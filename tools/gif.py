"""纯标准库 GIF89a 动画编码器（+ 结构解析器 + 自研 LZW 编解码）。

为什么必须自己实现 LZW
----------------------
环境里没有 ffmpeg（``docs/PROTOCOLS.md`` §9 的 MP4 分支因此不可用），
GIF 是**唯一**能动的交付格式；而 GIF 的图像数据是 LZW 压缩的，
标准库里没有任何现成实现（``zlib`` 是 deflate，不是 LZW），
``POLICY.md`` 又不允许引入 Pillow/ImageIO 之类运行期依赖。
所以"帧序列 → 可被浏览器播放的动画"这条链路要闭环，LZW 编解码器只能自研。
好在它的规模可控（~120 行），且**可被单测完全覆盖**：
本模块的测试用 Pillow 作为独立实现对拍（Pillow 内部是 giflib 的 C 实现），
这也是"我们写的 GIF 别人能读"的最强证据。

实现上的几个关键决定（都是踩过的坑）
------------------------------------
* **码长增长时机**：编码器在"输出一个码之后、登记新词条之前"判断是否把码长从 9 增到 12，
  与 giflib 完全一致。解码器则必须在"处理完当前码、并登记了新词条之后"判断，
  因为解码器永远比编码器晚一步建表（这一格错位会让 12 位边界上的文件全花屏）。
* **字典满 4095 就发 clear**（而不是用满 4096）：与 giflib 的保守做法一致，
  避免部分解码器在 4096 边界上的实现差异。
* **中位切分必须确定性**：颜色先按 ``(R,G,B)`` 打包成整数排序，切分时"最大跨度 → 像素数 → 盒子序号"
  三级比较全部可复现，代表色用整数加权平均（不碰浮点），同输入必然同输出。
* **≤256 色必须无损**：这是硬性质，不是"最好情况"。色数不超过上限时直接拿出现过的颜色当色表，
  一个像素都不许变；只有超色数时才真的做量化（此时损失是数学上不可避免的）。
* 局部色表只在**能带来无损表示**时使用：给每帧塞 768 字节色表既会让文件变大，
  也会破坏动画的全局配色一致性；超过 256 色时改用全局色表 + 最近邻/抖动映射更划算。
"""

from __future__ import annotations

import pathlib
import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "GifInfo",
    "decode_gif_frames",
    "lzw_decode",
    "lzw_encode",
    "quantize_median_cut",
    "read_gif",
    "write_gif",
]

#: GIF 逻辑屏幕/图像描述符的宽高是 2 字节无符号。
_MAX_DIM = 0xFFFF

#: LZW 的最大码长（GIF 规范：9..12 位）。
_MAX_CODE_BITS = 12

#: 字典槽位上限。留一格余量（4095 不再登记新词条），与 giflib/Pillow 的保守做法一致。
_MAX_TABLE = 4095

#: 图像数据子块的最大负载长度（GIF 规范：1 字节长度字段）。
_SUBBLOCK_MAX = 255


# --------------------------------------------------------------------------------------
# 颜色打包 / 调色板工具
# --------------------------------------------------------------------------------------


def _pack_rgb(rgb: np.ndarray) -> np.ndarray:
    """把 ``(H, W, 3)`` uint8 压成 ``(H, W)`` int64 的 ``0xRRGGBB``。

    打包成整数后可以用 ``np.unique`` / ``searchsorted`` 做集合运算，
    这是"精确命中调色板"与"颜色去重"两条快速路径的基础。
    """
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    return (r << 16) | (g << 8) | b


def _unpack_rgb(packed: np.ndarray) -> np.ndarray:
    """``0xRRGGBB`` 整数 → ``(N, 3)`` int64 颜色数组。"""
    flat = packed.reshape(-1)
    return np.stack(((flat >> 16) & 0xFF, (flat >> 8) & 0xFF, flat & 0xFF), axis=1)


def _validate_palette(palette: object, *, name: str = "palette") -> np.ndarray:
    """校验调色板：``(N, 3)`` uint8，``1 <= N <= 256``。"""
    arr = np.asarray(palette)
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} 必须是 uint8 数组，收到 dtype={arr.dtype}")
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} 必须是形状 (N, 3) 的数组，收到 shape={arr.shape}")
    if not 1 <= arr.shape[0] <= 256:
        raise ValueError(f"{name} 的颜色数必须在 1..256，收到 {arr.shape[0]}")
    return np.ascontiguousarray(arr)


def _palette_table(palette: np.ndarray) -> tuple[bytes, int]:
    """把调色板补齐成 2 的幂并序列化，返回 ``(色表字节, 位数)``。

    GIF 只允许色表长度为 2 的幂（2/4/8/…/256），不足的部分按规范补 0（黑色）；
    返回的"位数"同时决定 LZW 最小码长与描述符里的 size 字段。
    """
    count = int(palette.shape[0])
    bits = 1
    while (1 << bits) < count:
        bits += 1  # count==1 → bits==1 → 表长 2，这是 GIF 允许的最小色表
    table = np.zeros(((1 << bits), 3), dtype=np.uint8)
    table[:count] = palette
    return table.tobytes(), bits


def _subblocks(payload: bytes) -> bytes:
    """把负载切成 GIF 子块序列（每块 ≤255 字节），并以 0 长度块收尾。"""
    out = bytearray()
    for start in range(0, len(payload), _SUBBLOCK_MAX):
        chunk = payload[start : start + _SUBBLOCK_MAX]
        out.append(len(chunk))
        out += chunk
    out.append(0)
    return bytes(out)


def _read_subblocks(blob: bytes, pos: int) -> tuple[bytes, int]:
    """读出一段子块序列并拼回负载，返回 ``(负载, 新位置)``。"""
    parts: list[bytes] = []
    while True:
        if pos >= len(blob):
            raise ValueError("GIF 在子块序列中途结束（文件被截断）")
        size = blob[pos]
        pos += 1
        if size == 0:
            return b"".join(parts), pos
        if pos + size > len(blob):
            raise ValueError("GIF 子块声明的长度超出文件末尾（文件被截断）")
        parts.append(blob[pos : pos + size])
        pos += size


# --------------------------------------------------------------------------------------
# 中位切分量化
# --------------------------------------------------------------------------------------


def _split_box(box: np.ndarray, colors: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """沿"跨度最大的通道"在**像素数中位数**处把一个盒子切成两半。

    切点用累计像素数（而不是颜色个数）来定，保证每个盒子的像素量尽量均衡 ——
    这正是中位切分（median cut）比"均匀切色立方"在人眼观感上更好的原因。
    """
    sub = colors[box]
    channel = int(np.argmax(sub.max(axis=0) - sub.min(axis=0)))  # 并列时取最小通道号 → 确定性
    order = np.argsort(sub[:, channel], kind="stable")  # stable 保证同值颜色保持原（已排序）顺序
    ordered = box[order]
    cumulative = np.cumsum(counts[ordered])
    half = (int(cumulative[-1]) + 1) // 2
    cut = int(np.searchsorted(cumulative, half)) + 1
    cut = max(1, min(cut, int(box.size) - 1))  # 两侧都必须非空，否则会死循环
    return ordered[:cut], ordered[cut:]


def _box_representative(box: np.ndarray, colors: np.ndarray, counts: np.ndarray) -> tuple[int, int, int]:
    """盒子的代表色 = 像素数加权平均，用整数四舍五入（不碰浮点 → 完全可复现）。"""
    weight = counts[box]
    total = int(weight.sum())
    summed = (weight[:, None] * colors[box]).sum(axis=0)
    rep = (2 * summed + total) // (2 * total)  # 四舍五入（half-up），结果为 int64
    return int(rep[0]), int(rep[1]), int(rep[2])


def quantize_median_cut(rgb: np.ndarray, n_colors: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """中位切分量化：``(H, W, 3)`` uint8 → ``(索引图 (H, W) uint8, 调色板 (n, 3) uint8)``。

    三条硬性质（都有单测）：
    1. **确定性**：同输入必得同输出（不依赖哈希随机化、不依赖浮点舍入）。
    2. **色数 ≤ n_colors** 且 ≤ 256；返回的调色板只含真正用到的颜色。
    3. **无损退化**：输入自身的颜色数 ≤ n_colors 时，调色板就是出现过的那些颜色，
       索引经调色板映射后与原图**逐像素相等**。
    """
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        raise ValueError(f"rgb 必须是 uint8 数组，收到 dtype={arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"rgb 必须是形状 (H, W, 3) 的数组，收到 shape={arr.shape}")
    if not isinstance(n_colors, int) or not 1 <= n_colors <= 256:
        raise ValueError(f"n_colors 必须是 1..256 的整数，收到 {n_colors!r}")
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if height < 1 or width < 1:
        raise ValueError(f"rgb 的宽高都必须 >= 1，收到 shape={arr.shape}")

    packed = _pack_rgb(arr).reshape(-1)
    # np.unique 的返回值按数值升序 = (R,G,B) 字典序，这是"确定性"的第一层保证。
    unique, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    unique_count = int(unique.size)

    if unique_count <= n_colors:
        # 无损路径：出现过的颜色原样当色表。
        palette = _unpack_rgb(unique).astype(np.uint8)
        indices = inverse.astype(np.uint8).reshape(height, width)
        return indices, palette

    colors = _unpack_rgb(unique)  # (U, 3) int64
    populations = counts.astype(np.int64)
    boxes: list[np.ndarray] = [np.arange(unique_count)]
    spans: list[int] = [int((colors.max(axis=0) - colors.min(axis=0)).max())]
    pops: list[int] = [int(populations.sum())]

    while len(boxes) < n_colors:
        # 选盒规则：跨度最大 → 像素最多 → 序号最小；三级比较全部确定性。
        best = -1
        best_key = (-1, -1)
        for i, box in enumerate(boxes):
            if box.size < 2:
                continue
            key = (spans[i], pops[i])
            if key > best_key:
                best_key = key
                best = i
        if best < 0:
            break  # 已经没有可再切的盒子（颜色用尽）
        left, right = _split_box(boxes[best], colors, populations)
        boxes[best] = left
        boxes.append(right)
        # spans/pops 与 boxes 同步维护（切成两半后两个盒子的统计量都要重算）。
        left_colors = colors[left]
        right_colors = colors[right]
        spans[best] = int((left_colors.max(axis=0) - left_colors.min(axis=0)).max())
        pops[best] = int(populations[left].sum())
        spans.append(int((right_colors.max(axis=0) - right_colors.min(axis=0)).max()))
        pops.append(int(populations[right].sum()))

    palette_rows = [_box_representative(box, colors, populations) for box in boxes]
    # 不同盒子可能平均到同一个颜色：合并它们，避免色表里出现重复项（同时不浪费索引）。
    seen: dict[tuple[int, int, int], int] = {}
    remap = np.empty(len(palette_rows), dtype=np.int64)
    merged: list[tuple[int, int, int]] = []
    for i, row in enumerate(palette_rows):
        if row in seen:
            remap[i] = seen[row]
        else:
            seen[row] = len(merged)
            remap[i] = len(merged)
            merged.append(row)

    palette = np.array(merged, dtype=np.uint8)
    labels = np.empty(unique_count, dtype=np.uint8)
    for i, box in enumerate(boxes):
        labels[box] = remap[i]
    indices = labels[inverse].reshape(height, width)
    return indices, palette


# --------------------------------------------------------------------------------------
# 颜色 → 调色板映射（精确命中 / 最近邻 / Floyd–Steinberg 抖动）
# --------------------------------------------------------------------------------------


class _PaletteMapper:
    """把 RGB 像素映射到某个固定调色板的索引。

    两条路径：
    * **精确命中**（``searchsorted`` 向量化）：图像颜色全部在色表里时零损失、极快，
      这是"≤256 色无损"性质的执行体；
    * **最近邻**（分块向量化 + 逐色缓存）：只有真的缺色才付这个代价。

    抖动路径逐像素递推（Floyd–Steinberg 的误差必须顺序传播），用整数 1/16 定点运算，
    保证同输入同输出。
    """

    def __init__(self, palette: np.ndarray) -> None:
        self.palette = palette
        self._rgb = palette.astype(np.int64)
        packed = _pack_rgb(palette.reshape(1, -1, 3)).reshape(-1)
        order = np.argsort(packed, kind="stable")
        self._sorted_packed = packed[order]
        self._sorted_index = order.astype(np.int64)
        self._cache: dict[int, int] = {}

    def exact_indices(self, packed_colors: np.ndarray) -> np.ndarray | None:
        """色表精确命中查询；只要有一个颜色不在色表里就返回 ``None``。"""
        pos = np.searchsorted(self._sorted_packed, packed_colors)
        np.clip(pos, 0, self._sorted_packed.size - 1, out=pos)
        if not bool(np.all(self._sorted_packed[pos] == packed_colors)):
            return None
        return self._sorted_index[pos]

    def nearest_indices(self, colors: np.ndarray) -> np.ndarray:
        """``(N, 3)`` int64 颜色 → ``(N,)`` 最近邻索引（欧氏距离平方，并列取最小索引）。"""
        out = np.empty(colors.shape[0], dtype=np.int64)
        # 分块，避免 N×调色板×3 的中间数组吃光内存。
        chunk = max(1, 1 << 16 // max(1, self._rgb.shape[0]))
        for start in range(0, colors.shape[0], chunk):
            block = colors[start : start + chunk].astype(np.int64)
            diff = block[:, None, :] - self._rgb[None, :, :]
            dist = np.einsum("mnc,mnc->mn", diff, diff)
            out[start : start + chunk] = np.argmin(dist, axis=1)
        return out

    def nearest_one(self, r: int, g: int, b: int) -> int:
        """单色最近邻（带缓存）；抖动路径每像素都会调它。"""
        key = (r << 16) | (g << 8) | b
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        diff = self._rgb - np.array((r, g, b), dtype=np.int64)
        index = int(np.argmin(np.einsum("nc,nc->n", diff, diff)))
        self._cache[key] = index
        return index

    def map_frame(self, rgb: np.ndarray, *, dither: bool) -> np.ndarray:
        """整帧映射成 ``(H, W)`` uint8 索引图。"""
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        if dither:
            return self._map_frame_dither(rgb)
        packed = _pack_rgb(rgb).reshape(-1)
        unique, inverse = np.unique(packed, return_inverse=True)
        unique = np.asarray(unique).reshape(-1)
        inverse = np.asarray(inverse).reshape(-1)
        exact = self.exact_indices(unique)
        if exact is None:
            # 缺色时逐个补：能精确命中的仍走精确路径，只有缺口才用最近邻。
            pos = np.searchsorted(self._sorted_packed, unique)
            np.clip(pos, 0, self._sorted_packed.size - 1, out=pos)
            hit = self._sorted_packed[pos] == unique
            near = self.nearest_indices(_unpack_rgb(unique))
            exact = np.where(hit, self._sorted_index[pos], near)
        return exact[inverse].reshape(height, width).astype(np.uint8)

    def _map_frame_dither(self, rgb: np.ndarray) -> np.ndarray:
        """Floyd–Steinberg 误差扩散（整数 1/16 定点，确定性）。

        误差缓冲多留两行两列，省掉边界分支；传播权重 7/3/5/1 是经典 FS 核。
        右移对负数向下取整，正好是"往左上方少分一点误差"的稳定行为。
        """
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        source = rgb.tolist()
        error = [[[0, 0, 0] for _ in range(width + 2)] for _ in range(height + 2)]
        rows: list[list[int]] = []
        palette = self._rgb
        for y in range(height):
            src_row = source[y]
            err_row = error[y]
            err_next = error[y + 1]
            out_row: list[int] = []
            for x in range(width):
                base = err_row[x + 1]
                r0, g0, b0 = src_row[x]
                v0 = r0 * 16 + base[0]
                v1 = g0 * 16 + base[1]
                v2 = b0 * 16 + base[2]
                # 夹到合法域，避免误差在极端区域越滚越大。
                v0 = 0 if v0 < 0 else (4080 if v0 > 4080 else v0)
                v1 = 0 if v1 < 0 else (4080 if v1 > 4080 else v1)
                v2 = 0 if v2 < 0 else (4080 if v2 > 4080 else v2)
                index = self.nearest_one((v0 + 8) >> 4, (v1 + 8) >> 4, (v2 + 8) >> 4)
                out_row.append(index)
                e0 = v0 - int(palette[index, 0]) * 16
                e1 = v1 - int(palette[index, 1]) * 16
                e2 = v2 - int(palette[index, 2]) * 16
                right = err_row[x + 2]
                down_left = err_next[x]
                down = err_next[x + 1]
                down_right = err_next[x + 2]
                right[0] += (e0 * 7) >> 4
                right[1] += (e1 * 7) >> 4
                right[2] += (e2 * 7) >> 4
                down_left[0] += (e0 * 3) >> 4
                down_left[1] += (e1 * 3) >> 4
                down_left[2] += (e2 * 3) >> 4
                down[0] += (e0 * 5) >> 4
                down[1] += (e1 * 5) >> 4
                down[2] += (e2 * 5) >> 4
                down_right[0] += e0 >> 4
                down_right[1] += e1 >> 4
                down_right[2] += e2 >> 4
            rows.append(out_row)
        return np.array(rows, dtype=np.uint8)


# --------------------------------------------------------------------------------------
# LZW
# --------------------------------------------------------------------------------------


def lzw_encode(indices: np.ndarray, *, min_code_size: int) -> bytes:
    """GIF 变体 LZW 编码，返回**未切子块**的码流字节串。

    与 ``lzw_decode`` 严格互逆（``lzw_decode(lzw_encode(x)) == x``），
    子块切分由 ``write_gif`` 负责 —— 这样"算法正确性"与"容器格式"可以分开测。

    参数 ``min_code_size`` 是 GIF 的"LZW 最小码长"（2..8，等于色表位数、且不小于 2）。
    """
    arr = np.asarray(indices)
    if arr.dtype != np.uint8:
        raise ValueError(f"indices 必须是 uint8 数组，收到 dtype={arr.dtype}")
    if not isinstance(min_code_size, int) or not 2 <= min_code_size <= 8:
        raise ValueError(f"min_code_size 必须是 2..8 的整数，收到 {min_code_size!r}")
    limit = 1 << min_code_size
    flat = arr.reshape(-1)
    if flat.size:
        max_index = int(flat.max())
        if max_index >= limit:
            raise ValueError(f"索引最大值 {max_index} 超出最小码长 {min_code_size} 可表示的 0..{limit - 1}")
    clear_code = limit
    end_code = limit + 1

    out = bytearray()
    accumulator = 0
    nbits = 0
    width = min_code_size + 1
    table: dict[int, int] = {}
    next_code = clear_code + 2

    # 规范要求码流以 clear code 开始。
    accumulator |= clear_code << nbits
    nbits += width
    while nbits >= 8:
        out.append(accumulator & 0xFF)
        accumulator >>= 8
        nbits -= 8

    values = flat.tolist()
    if values:
        shift = min_code_size
        entity = values[0]
        for pixel in values[1:]:
            key = (entity << shift) | pixel
            code = table.get(key)
            if code is not None:
                entity = code
                continue
            # 热点路径：命中未登记的前缀就输出它。位打包在此内联（函数调用开销在
            # 百万像素级别上不可忽略），代价是这段打包代码在本函数里出现三次。
            accumulator |= entity << nbits
            nbits += width
            while nbits >= 8:
                out.append(accumulator & 0xFF)
                accumulator >>= 8
                nbits -= 8
            # 码长增长：在"输出之后、登记新词条之前"判断（与 giflib 一致）。
            # 每次只可能长 1 位，故 if 足够；12 位封顶。
            if next_code >= (1 << width) and width < _MAX_CODE_BITS:
                width += 1
            if next_code < _MAX_TABLE:
                table[key] = next_code
                next_code += 1
            else:
                # 字典将满：发 clear（仍用当前码长），随后整体复位。
                accumulator |= clear_code << nbits
                nbits += width
                while nbits >= 8:
                    out.append(accumulator & 0xFF)
                    accumulator >>= 8
                    nbits -= 8
                width = min_code_size + 1
                next_code = clear_code + 2
                table.clear()
            entity = pixel
        accumulator |= entity << nbits
        nbits += width
        while nbits >= 8:
            out.append(accumulator & 0xFF)
            accumulator >>= 8
            nbits -= 8
        # 收尾的这枚码同样可能触发码长增长（giflib 的 output() 对每一枚码都做这个判断）。
        # 漏掉这一次会让 EOI 用错码长 —— 只在"最后一枚数据码正好越过阈值"时炸，
        # 属于典型的偶发 bug，所以这里必须与主循环保持一致。
        if next_code >= (1 << width) and width < _MAX_CODE_BITS:
            width += 1
    accumulator |= end_code << nbits
    nbits += width
    while nbits >= 8:
        out.append(accumulator & 0xFF)
        accumulator >>= 8
        nbits -= 8
    if nbits:
        out.append(accumulator & 0xFF)  # 末字节高位补 0
    return bytes(out)


def lzw_decode(data: bytes, *, min_code_size: int, expected_len: int) -> np.ndarray:
    """GIF 变体 LZW 解码，返回 ``(expected_len,)`` 的 uint8 索引数组。

    必须能读**别人写的** GIF（Pillow/浏览器编码器），因此这里实现了完整规则：
    clear / end-of-information、码长 9→12 增长、4096 时停表（deferred clear）、
    以及 ``code == next_code`` 的 KwKwK 特例（此时词条 = 前缀 + 前缀首字符）。
    """
    if not isinstance(min_code_size, int) or not 2 <= min_code_size <= 8:
        raise ValueError(f"min_code_size 必须是 2..8 的整数，收到 {min_code_size!r}")
    if expected_len < 0:
        raise ValueError(f"expected_len 不能为负，收到 {expected_len}")

    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    table: list[bytes] = [bytes((i,)) for i in range(clear_code)]
    table.append(b"")  # clear 占位
    table.append(b"")  # end 占位
    width = min_code_size + 1
    next_code = clear_code + 2
    out = bytearray()
    previous: bytes | None = None
    bitpos = 0
    total_bits = len(data) * 8

    while True:
        if bitpos + width > total_bits:
            raise ValueError("LZW 码流在遇到结束码之前就结束了（文件被截断）")
        byte_index = bitpos >> 3
        window = int.from_bytes(data[byte_index : byte_index + 3], "little")
        code = (window >> (bitpos & 7)) & ((1 << width) - 1)
        bitpos += width

        if code == clear_code:
            del table[clear_code + 2 :]
            next_code = clear_code + 2
            width = min_code_size + 1
            previous = None
            continue
        if code == end_code:
            break
        if previous is None:
            if code >= clear_code:
                raise ValueError(f"LZW 首码必须是字面量（0..{clear_code - 1}），收到 {code}")
            entry = table[code]
        elif code < next_code:
            entry = table[code]
            if next_code < 4096:
                table.append(previous + entry[:1])
                next_code += 1
        elif code == next_code:
            # KwKwK：编码器用了"刚登记、解码器还差一步"的词条。
            entry = previous + previous[:1]
            if next_code < 4096:
                table.append(entry)
                next_code += 1
        else:
            raise ValueError(f"非法 LZW 码 {code}（当前下一个可用码为 {next_code}）")
        out += entry
        previous = entry
        # 解码器建表比编码器晚一步，因此码长增长判断放在"登记之后"。
        if next_code >= (1 << width) and width < _MAX_CODE_BITS:
            width += 1

    if len(out) != expected_len:
        raise ValueError(f"LZW 解出的像素数 {len(out)} 与期望 {expected_len} 不符")
    return np.frombuffer(bytes(out), dtype=np.uint8).copy()


# --------------------------------------------------------------------------------------
# 写出
# --------------------------------------------------------------------------------------


def _normalize_frames(frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    """把输入帧统一成 ``(H, W, 3)`` uint8，并做尺寸一致性校验。"""
    items = list(frames)
    if not items:
        raise ValueError("frames 不能为空：GIF 至少需要一帧")
    normalized: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for i, frame in enumerate(items):
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            raise ValueError(f"第 {i} 帧必须是 uint8 数组，收到 dtype={arr.dtype}")
        if arr.ndim == 2:
            # 灰度帧自动升为 RGB：GIF 没有"灰度模式"，色表总是 RGB 三元组。
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"第 {i} 帧必须是 (H, W, 3) 或 (H, W)，收到 shape={arr.shape}")
        height, width = int(arr.shape[0]), int(arr.shape[1])
        if height < 1 or width < 1:
            raise ValueError(f"第 {i} 帧的宽高都必须 >= 1，收到 shape={arr.shape}")
        if height > _MAX_DIM or width > _MAX_DIM:
            raise ValueError(f"第 {i} 帧尺寸超过 GIF 上限 {_MAX_DIM}，收到 shape={arr.shape}")
        if shape is None:
            shape = (height, width)
        elif (height, width) != shape:
            raise ValueError(
                f"第 {i} 帧尺寸 {width}x{height} 与首帧 {shape[1]}x{shape[0]} 不一致："
                "GIF 的逻辑屏幕尺寸固定，所有帧必须同尺寸"
            )
        normalized.append(np.ascontiguousarray(arr))
    return normalized


def _check_uint16(name: str, value: object, *, low: int = 0) -> int:
    """校验需要写进 2 字节字段的整数。"""
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= _MAX_DIM:
        raise ValueError(f"{name} 必须是 {low}..{_MAX_DIM} 的整数，收到 {value!r}")
    return value


def _comment_extension(comment: str | None) -> bytes:
    """注释扩展（0x21 0xFE）：纯字节负载，用 Latin-1 编码以保证可复现。"""
    if comment is None:
        return b""
    if not isinstance(comment, str):
        raise ValueError(f"comment 必须是 str 或 None，收到 {type(comment).__name__}")
    try:
        payload = comment.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError("comment 含非 Latin-1 字符，GIF 注释是纯字节负载，请先自行编码") from exc
    if b"\x00" in payload:
        raise ValueError("comment 不得包含 NUL 字节（GIF 的子块长度字段会与它冲突）")
    return bytes((0x21, 0xFE)) + _subblocks(payload)


def _netscape_extension(loop: int) -> bytes:
    """NETSCAPE2.0 循环扩展：``loop=0`` 表示无限循环。"""
    return bytes((0x21, 0xFF, 0x0B)) + b"NETSCAPE2.0" + bytes((0x03, 0x01)) + struct.pack("<H", loop) + b"\x00"


def write_gif(
    path: str | pathlib.Path,
    frames: Sequence[np.ndarray],
    *,
    delay_cs: int = 4,
    loop: int = 0,
    palette: np.ndarray | None = None,
    dither: bool = False,
    disposal: int = 2,
    comment: str | None = None,
) -> pathlib.Path:
    """把帧序列写成 GIF89a 动画，返回落盘路径。

    参数
    ----
    frames
        每帧 ``(H, W, 3)`` uint8（``(H, W)`` uint8 灰度会自动升为 RGB）；所有帧必须同尺寸。
    delay_cs
        帧间隔，单位 1/100 秒（GIF 的 delay 字段就是厘秒）。
    loop
        NETSCAPE 循环次数，``0`` = 无限循环。
    palette
        给定 ``(N, 3)`` uint8 时所有帧共用它，颜色按最近邻映射（可复现，可能有色差）；
        为 ``None`` 时用**首帧**的中位切分结果做全局色表，只有在"某帧颜色数 ≤256 且不在
        全局色表里"时才为该帧写局部色表 —— 这样既不牺牲无损性，也不给每帧白塞 768 字节。
    dither
        ``False``（默认）最近邻；``True`` 走 Floyd–Steinberg 误差扩散（仍然确定性）。
    disposal
        GCE 的处置方式（2 = 恢复背景，动画的常规选择）。
    comment
        可选注释（Latin-1），写进 0xFE 扩展块。
    """
    images = _normalize_frames(frames)
    height, width = int(images[0].shape[0]), int(images[0].shape[1])
    delay_cs = _check_uint16("delay_cs", delay_cs)
    loop = _check_uint16("loop", loop)
    if not isinstance(disposal, int) or isinstance(disposal, bool) or not 0 <= disposal <= 7:
        raise ValueError(f"disposal 必须是 0..7 的整数，收到 {disposal!r}")

    if palette is not None:
        base_palette = _validate_palette(palette)
        explicit_palette = True
    else:
        _, base_palette = quantize_median_cut(images[0], 256)
        explicit_palette = False
    # 全局色表的映射器只建一次：它的内部缓存（最近邻结果）对所有帧复用。
    base_mapper = _PaletteMapper(base_palette)
    local_palettes: list[np.ndarray | None] = [None] * len(images)
    if not explicit_palette:
        for i, image in enumerate(images[1:], start=1):
            unique = np.unique(_pack_rgb(image).reshape(-1))
            # 只有"能无损表示"的帧才配局部色表；超过 256 色的帧交给全局色表 + 抖动。
            if unique.size <= 256 and base_mapper.exact_indices(unique) is None:
                _, local = quantize_median_cut(image, 256)
                local_palettes[i] = local

    global_table, global_bits = _palette_table(base_palette)
    logical_packed = 0x80 | ((global_bits - 1) << 4) | (global_bits - 1)

    out = bytearray()
    out += b"GIF89a"
    out += struct.pack("<HHBBB", width, height, logical_packed, 0, 0)  # 背景色索引 0、像素比 0
    out += global_table
    out += _netscape_extension(loop)
    out += _comment_extension(comment)

    for index, image in enumerate(images):
        local = local_palettes[index]
        if local is None:
            mapper = base_mapper
            table, bits = global_table, global_bits
            image_packed = 0
        else:
            mapper = _PaletteMapper(local)
            table, bits = _palette_table(local)
            image_packed = 0x80 | (bits - 1)
        indices = mapper.map_frame(image, dither=dither)
        min_code_size = max(2, bits)  # GIF 规范：LZW 最小码长不得小于 2
        out += bytes((0x21, 0xF9, 0x04))  # 图形控制扩展
        out += bytes(((disposal & 0x07) << 2,))  # 无透明色、无用户输入
        out += struct.pack("<H", delay_cs)
        out += bytes((0, 0))  # 透明色索引（未启用）+ 结束符
        out += bytes((0x2C,))  # 图像描述符
        out += struct.pack("<HHHHB", 0, 0, width, height, image_packed)
        if local is not None:
            out += table
        out += bytes((min_code_size,))
        out += _subblocks(lzw_encode(indices, min_code_size=min_code_size))
    out += b"\x3B"  # trailer

    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(out))
    return destination


# --------------------------------------------------------------------------------------
# 结构解析 / 解码
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GifInfo:
    """GIF 的结构信息（不做像素解码，只看块级元数据）。"""

    width: int
    height: int
    n_frames: int
    delays_cs: tuple[int, ...]
    loop: int | None
    palette_size: int
    version: str


@dataclass(frozen=True)
class _GifFrame:
    """一帧的块级信息（内部使用）。"""

    left: int
    top: int
    width: int
    height: int
    interlaced: bool
    palette: bytes | None  # 局部色表（RGB 三元组），None 表示用全局色表
    min_code_size: int
    data: bytes  # 已拼好的 LZW 码流（子块已剥离）
    delay_cs: int
    disposal: int
    transparent: int | None


@dataclass(frozen=True)
class _GifFile:
    """整个 GIF 的块级结构（内部使用）。"""

    version: str
    width: int
    height: int
    background_index: int
    global_palette: bytes | None
    loop: int | None
    frames: tuple[_GifFrame, ...]


def _parse_gif(path: str | pathlib.Path) -> _GifFile:
    """块级解析 GIF89a/GIF87a。

    刻意**不**在这一层解 LZW：结构错误（缺 trailer、块被截断）与像素数据错误分开报，
    定位问题时能一眼看出是容器坏了还是码流坏了。
    """
    blob = pathlib.Path(path).read_bytes()
    if len(blob) < 13:
        raise ValueError(f"文件太短（{len(blob)} 字节），不可能是一个完整 GIF")
    version_bytes = blob[:6]
    if version_bytes not in (b"GIF87a", b"GIF89a"):
        raise ValueError(f"不是 GIF：签名 {version_bytes!r} 既不是 GIF87a 也不是 GIF89a")
    width, height, packed, background, _aspect = struct.unpack("<HHBBB", blob[6:13])
    if width < 1 or height < 1:
        raise ValueError(f"非法的 GIF 逻辑屏幕尺寸 {width}x{height}")

    pos = 13
    global_palette: bytes | None = None
    if packed & 0x80:
        entries = 1 << ((packed & 0x07) + 1)
        if pos + 3 * entries > len(blob):
            raise ValueError("全局色表声明的长度超出文件末尾（文件被截断）")
        global_palette = blob[pos : pos + 3 * entries]
        pos += 3 * entries

    frames: list[_GifFrame] = []
    loop: int | None = None
    pending_delay = 0
    pending_disposal = 0
    pending_transparent: int | None = None
    saw_trailer = False

    while pos < len(blob):
        marker = blob[pos]
        if marker == 0x3B:  # trailer
            pos += 1
            saw_trailer = True
            break
        if marker == 0x21:  # 扩展块
            if pos + 2 > len(blob):
                raise ValueError("GIF 扩展块头部被截断")
            label = blob[pos + 1]
            payload, pos = _read_subblocks(blob, pos + 2)
            if label == 0xF9:
                if len(payload) != 4:
                    raise ValueError(f"图形控制扩展长度必须是 4 字节，收到 {len(payload)}")
                control = payload[0]
                pending_delay = payload[1] | (payload[2] << 8)
                pending_disposal = (control >> 2) & 0x07
                pending_transparent = payload[3] if control & 0x01 else None
            elif label in (0xFF, 0xFE, 0x01):
                if label == 0xFF and payload[:11] in (b"NETSCAPE2.0", b"ANIMEXTS1.0") and len(payload) >= 14:
                    if payload[11] == 0x01:
                        loop = payload[12] | (payload[13] << 8)
            continue
        if marker == 0x2C:  # 图像描述符
            if pos + 10 > len(blob):
                raise ValueError("GIF 图像描述符被截断")
            left, top, frame_width, frame_height, image_packed = struct.unpack("<HHHHB", blob[pos + 1 : pos + 10])
            pos += 10
            local_palette: bytes | None = None
            if image_packed & 0x80:
                entries = 1 << ((image_packed & 0x07) + 1)
                if pos + 3 * entries > len(blob):
                    raise ValueError("局部色表声明的长度超出文件末尾（文件被截断）")
                local_palette = blob[pos : pos + 3 * entries]
                pos += 3 * entries
            if pos >= len(blob):
                raise ValueError("GIF 图像数据前缺少 LZW 最小码长字节（文件被截断）")
            min_code_size = blob[pos]
            pos += 1
            if not 2 <= min_code_size <= 8:
                raise ValueError(f"GIF 的 LZW 最小码长必须在 2..8，收到 {min_code_size}")
            data, pos = _read_subblocks(blob, pos)
            frames.append(
                _GifFrame(
                    left=left,
                    top=top,
                    width=frame_width,
                    height=frame_height,
                    interlaced=bool(image_packed & 0x40),
                    palette=local_palette,
                    min_code_size=min_code_size,
                    data=data,
                    delay_cs=pending_delay,
                    disposal=pending_disposal,
                    transparent=pending_transparent,
                )
            )
            # 图形控制扩展只作用于紧随其后的这一帧。
            pending_delay = 0
            pending_disposal = 0
            pending_transparent = None
            continue
        raise ValueError(f"未知的 GIF 块标识 0x{marker:02x}（位置 {pos}）")

    if not saw_trailer:
        raise ValueError("GIF 缺少结束符 0x3B（文件被截断）")
    return _GifFile(
        version=version_bytes.decode("ascii"),
        width=int(width),
        height=int(height),
        background_index=int(background),
        global_palette=global_palette,
        loop=loop,
        frames=tuple(frames),
    )


def read_gif(path: str | pathlib.Path) -> GifInfo:
    """读取 GIF 的块级结构：尺寸、帧数、每帧延时、循环次数、色表大小、版本号。"""
    parsed = _parse_gif(path)
    if parsed.global_palette is not None:
        palette_size = len(parsed.global_palette) // 3
    elif parsed.frames and parsed.frames[0].palette is not None:
        palette_size = len(parsed.frames[0].palette) // 3
    else:
        palette_size = 0
    return GifInfo(
        width=parsed.width,
        height=parsed.height,
        n_frames=len(parsed.frames),
        delays_cs=tuple(frame.delay_cs for frame in parsed.frames),
        loop=parsed.loop,
        palette_size=palette_size,
        version=parsed.version,
    )


def _deinterlace(indices: np.ndarray, height: int, width: int) -> np.ndarray:
    """还原 GIF 的 4 遍隔行扫描（我们自己不写隔行，但要能读别人的）。"""
    rows: list[int] = []
    for start, step in ((0, 8), (4, 8), (2, 4), (1, 2)):
        rows.extend(range(start, height, step))
    out = np.empty((height, width), dtype=np.uint8)
    out[np.array(rows[:height], dtype=np.int64)] = indices.reshape(height, width)
    return out


def decode_gif_frames(path: str | pathlib.Path) -> list[np.ndarray]:
    """解出每一帧并**合成**成 RGB 图像，返回 ``(H, W, 3)`` uint8 数组列表（逻辑屏幕尺寸）。

    说明：内部先用自研 LZW 解出索引图，再按该帧的色表（局部优先，否则全局）映射成 RGB，
    并处理处置方式（disposal 2=恢复背景 / 3=恢复上一帧）与透明索引 —— 这与浏览器、
    Pillow 的播放语义一致，因此可以直接和 ``PIL.Image.open(...)`` 的结果逐像素比对。
    """
    parsed = _parse_gif(path)
    if not parsed.frames:
        raise ValueError("GIF 里没有任何图像帧")
    height, width = parsed.height, parsed.width
    background = np.zeros(3, dtype=np.uint8)
    if parsed.global_palette is not None:
        table = np.frombuffer(parsed.global_palette, dtype=np.uint8).reshape(-1, 3)
        if parsed.background_index < table.shape[0]:
            background = table[parsed.background_index].copy()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = background
    result: list[np.ndarray] = []
    saved_rect: np.ndarray | None = None
    previous: _GifFrame | None = None

    for frame in parsed.frames:
        if frame.width < 1 or frame.height < 1 or frame.left + frame.width > width or frame.top + frame.height > height:
            raise ValueError(f"帧矩形 ({frame.left},{frame.top},{frame.width},{frame.height}) 超出逻辑屏幕 {width}x{height}")
        if previous is not None:
            # 处置方式在"下一帧绘制之前"生效。
            if previous.disposal == 2:
                canvas[
                    previous.top : previous.top + previous.height,
                    previous.left : previous.left + previous.width,
                ] = background
            elif previous.disposal == 3 and saved_rect is not None:
                canvas[
                    previous.top : previous.top + previous.height,
                    previous.left : previous.left + previous.width,
                ] = saved_rect
        rect = (slice(frame.top, frame.top + frame.height), slice(frame.left, frame.left + frame.width))
        saved_rect = canvas[rect].copy() if frame.disposal == 3 else None
        palette_bytes = frame.palette if frame.palette is not None else parsed.global_palette
        if palette_bytes is None:
            raise ValueError("该帧既没有局部色表也没有全局色表，无法还原颜色")
        table = np.frombuffer(palette_bytes, dtype=np.uint8).reshape(-1, 3)
        indices = lzw_decode(frame.data, min_code_size=frame.min_code_size, expected_len=frame.width * frame.height)
        if frame.interlaced:
            indices = _deinterlace(indices, frame.height, frame.width)
        else:
            indices = indices.reshape(frame.height, frame.width)
        if int(indices.max()) >= table.shape[0]:
            raise ValueError(f"帧索引 {int(indices.max())} 超出色表长度 {table.shape[0]}")
        rgb = table[indices]
        region = canvas[rect]
        if frame.transparent is not None:
            keep = indices != frame.transparent
            region[keep] = rgb[keep]
        else:
            region[:] = rgb
        result.append(canvas.copy())
        previous = frame
    return result
