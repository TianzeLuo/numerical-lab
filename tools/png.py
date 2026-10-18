"""纯标准库 PNG 写出器（writer）+ 读回器（reader）。

为什么要自己写 PNG
------------------
``docs/POLICY.md`` 把本仓库的运行期依赖压到只剩 numpy（P7 的精神同样适用于 Python 侧），
而 ``docs/PROTOCOLS.md`` §9 要求把光栅基线落到 ``dist/visual/png/``。
PNG 是"容器 + zlib 流"两者都已被标准库覆盖的格式：``zlib`` 提供 deflate，
``struct`` 提供大端打包，``zlib.crc32`` 提供 CRC —— 于是写出器只有 ~200 行，
却可以被单测完全覆盖。自研因此**比引依赖更可控**：字节级可复现、无需外部进程、
也不会因为 Pillow 的版本差异改变基线。

设计要点（每一条都是为了"可复现"和"可验证"）
------------------------------------------------
* **过滤类型只用 0（None）**：写出侧速度与实现复杂度最优，且能让 zlib 处理连续色调；
  但读出侧必须实现 0..4 全部五种过滤器 —— 因为 Pillow / 浏览器重写文件时会自适应挑过滤器，
  我们的读回器要能吃下"别人改写过的"PNG。
* **不写时间戳**（PNG 的 ``tIME``）与任何随机内容：同输入两次写出必须逐字节相同，
  这样光栅基线才能用哈希比较。
* **CRC32 必须校验**：读回时任何块 CRC 不符即抛 ``ValueError``，把"文件被截断/篡改"
  类故障暴露在测试里，而不是悄悄解出一堆噪声像素。
* 尺寸、通道数、dtype 一律自检：``uint8`` 是硬要求（浮点数组若被静默取整，
  光栅基线会以"看起来对"的方式错掉，这是最难查的一类 bug）。
"""

from __future__ import annotations

import pathlib
import struct
import zlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np

__all__ = [
    "PngImage",
    "png_bytes",
    "read_png",
    "write_gray",
    "write_rgb",
    "write_rgba",
]

#: PNG 文件签名（8 字节魔数）。
_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: 通道数 → PNG 颜色类型（colortype）。
_COLOR_TYPE_OF_CHANNELS = {1: 0, 3: 2, 4: 6}

#: PNG 颜色类型 → 通道数（仅支持本模块声明的那三种）。
_CHANNELS_OF_COLOR_TYPE = {0: 1, 2: 3, 6: 4}

#: 宽高字段是 4 字节无符号；实现上按 2**31-1 设限，避免下游 numpy 申请离谱内存。
_MAX_DIM = 0x7FFFFFFF

#: tEXt 关键字长度区间（PNG 规范：1..79 字节）。
_MIN_KEYWORD = 1
_MAX_KEYWORD = 79


@dataclass(frozen=True)
class PngImage:
    """读回的 PNG 图像：像素为**行主序、已还原（去过滤）**的原始字节。"""

    width: int
    height: int
    channels: int
    pixels: bytes

    def array(self) -> np.ndarray:
        """把 ``pixels`` 还原成 ``(H, W)`` 或 ``(H, W, C)`` 的 uint8 数组。

        提供这个便捷方法是刻意的：调用方（光栅基线、测试）几乎总是要数组形态，
        而在这里 reshape 能保证通道语义与 ``channels`` 一致。
        """
        flat = np.frombuffer(self.pixels, dtype=np.uint8)
        if self.channels == 1:
            return flat.reshape(self.height, self.width).copy()
        return flat.reshape(self.height, self.width, self.channels).copy()


# --------------------------------------------------------------------------------------
# 写出
# --------------------------------------------------------------------------------------


def _shape_hint(channels: int) -> str:
    """给出该通道数下的合法 shape 说明（用于错误信息）。"""
    if channels == 1:
        return "H, W"
    return f"H, W, {channels}"


def _validate(array: object, *, channels: int, name: str) -> np.ndarray:
    """校验图像数组并返回 C 连续副本。

    这里刻意"宁可报错也不猜"：dtype、维度、通道数、宽高全部显式检查，
    错误信息里带上实际 shape/dtype，避免调用方拿到一张错位的图却毫不知情。
    """
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        raise ValueError(
            f"{name} 必须是 uint8 数组，收到 dtype={arr.dtype}；"
            "请显式 astype(np.uint8)，不要依赖隐式截断（浮点会被静默取整/回绕）"
        )
    expected_ndim = 2 if channels == 1 else 3
    if arr.ndim != expected_ndim:
        raise ValueError(
            f"{name} 必须是 {expected_ndim} 维数组（shape=({_shape_hint(channels)})），收到 shape={arr.shape}"
        )
    if arr.ndim == 3 and arr.shape[2] != channels:
        raise ValueError(f"{name} 最后一维必须是 {channels}（通道数），收到 shape={arr.shape}")
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if height < 1 or width < 1:
        raise ValueError(f"{name} 的宽高都必须 >= 1，收到 shape={arr.shape}")
    if height > _MAX_DIM or width > _MAX_DIM:
        raise ValueError(f"{name} 尺寸超过 PNG 上限 {_MAX_DIM}，收到 shape={arr.shape}")
    return np.ascontiguousarray(arr)


def _chunk(tag: bytes, payload: bytes) -> bytes:
    """拼一个 PNG 块：长度(4) + 类型(4) + 数据 + CRC32(类型+数据)。"""
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def _raw_scanlines(arr: np.ndarray, channels: int) -> bytes:
    """把图像摊成"每行前加一个过滤类型字节 0"的原始字节流。

    实现上先申请 ``(H, W*C+1)`` 的缓冲区、整列写 0、再一次性搬像素：
    比 Python 里逐行 ``b"\\x00" + row.tobytes()`` 快一个量级，
    这对 128×128×60 帧这类批量导出很关键。
    """
    height, width = int(arr.shape[0]), int(arr.shape[1])
    stride = width * channels
    raw = np.empty((height, stride + 1), dtype=np.uint8)
    raw[:, 0] = 0  # 过滤类型 0 = None
    raw[:, 1:] = arr.reshape(height, stride)
    return raw.tobytes()


def _text_chunks(text: Mapping[str, str] | None) -> bytes:
    """把 ``{关键字: 文本}`` 编成若干 tEXt 块。

    tEXt 的文本是 **Latin-1**（PNG 规范对 tEXt 的规定，iTXt 才允许 UTF-8）。
    关键字有 1..79 字节、不得含 NUL、且首尾不能是空格 —— 非法即抛 ``ValueError``，
    因为违反规范的关键字会让严格解码器（含浏览器）直接拒绝整个文件。
    """
    if not text:
        return b""
    out: list[bytes] = []
    for key, value in text.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"tEXt 的关键字与文本都必须是 str，收到 {type(key).__name__}/{type(value).__name__}")
        try:
            key_bytes = key.encode("latin-1")
        except UnicodeEncodeError as exc:  # pragma: no cover - 触发路径由测试覆盖
            raise ValueError(f"tEXt 关键字 {key!r} 含非 Latin-1 字符，无法编码") from exc
        try:
            value_bytes = value.encode("latin-1")
        except UnicodeEncodeError as exc:  # pragma: no cover - 触发路径由测试覆盖
            raise ValueError(f"tEXt 文本（关键字 {key!r}）含非 Latin-1 字符，无法编码") from exc
        if not _MIN_KEYWORD <= len(key_bytes) <= _MAX_KEYWORD:
            raise ValueError(f"tEXt 关键字长度必须在 {_MIN_KEYWORD}..{_MAX_KEYWORD} 字节，收到 {len(key_bytes)}（{key!r}）")
        if b"\x00" in key_bytes:
            raise ValueError(f"tEXt 关键字不得包含 NUL 字节：{key!r}")
        if b"\x00" in value_bytes:
            # NUL 是 tEXt 的分隔符，写进去会让文本被截断成两半。
            raise ValueError(f"tEXt 文本不得包含 NUL 字节（关键字 {key!r}）")
        if key_bytes[:1] == b" " or key_bytes[-1:] == b" ":
            raise ValueError(f"tEXt 关键字首尾不得是空格：{key!r}")
        out.append(_chunk(b"tEXt", key_bytes + b"\x00" + value_bytes))
    return b"".join(out)


def _encode(arr: np.ndarray, channels: int, compress_level: int, text: Mapping[str, str] | None) -> bytes:
    """核心：把已校验的数组编成完整 PNG 字节串（IHDR/tEXt…/IDAT/IEND）。"""
    if not isinstance(compress_level, int) or not 0 <= compress_level <= 9:
        raise ValueError(f"compress_level 必须是 0..9 的整数（zlib 级别），收到 {compress_level!r}")
    height, width = int(arr.shape[0]), int(arr.shape[1])
    color_type = _COLOR_TYPE_OF_CHANNELS[channels]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    compressor = zlib.compressobj(compress_level)
    raw = _raw_scanlines(arr, channels)
    idat = compressor.compress(raw) + compressor.flush()
    return b"".join(
        (
            _SIGNATURE,
            _chunk(b"IHDR", ihdr),
            # tEXt 放在 IDAT 之前：解码器无需先解压图像就能读到元数据。
            _text_chunks(text),
            _chunk(b"IDAT", idat),
            _chunk(b"IEND", b""),
        )
    )


def png_bytes(
    rgb_or_rgba: np.ndarray,
    *,
    compress_level: int = 6,
    text: Mapping[str, str] | None = None,
) -> bytes:
    """不落盘的 PNG 编码：入参可以是 ``(H, W)`` 灰度、``(H, W, 3)`` RGB、``(H, W, 4)`` RGBA。

    通道数由 shape 推断（这也是它与 ``write_gray/rgb/rgba`` 的区别：
    后者用函数名把通道数钉死，避免"以为传了 RGB 其实是灰度"的静默错误）。
    """
    arr = np.asarray(rgb_or_rgba)
    if arr.ndim == 2:
        channels = 1
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        channels = int(arr.shape[2])
    else:
        raise ValueError(f"png_bytes 只接受 (H, W) / (H, W, 3) / (H, W, 4)，收到 shape={arr.shape}")
    checked = _validate(arr, channels=channels, name="rgb_or_rgba")
    return _encode(checked, channels, compress_level, text)


def _write(
    path: str | pathlib.Path,
    array: np.ndarray,
    *,
    channels: int,
    name: str,
    compress_level: int,
    text: Mapping[str, str] | None,
) -> pathlib.Path:
    """写文件的公共实现：校验 → 编码 → 建父目录 → 落盘。"""
    checked = _validate(array, channels=channels, name=name)
    data = _encode(checked, channels, compress_level, text)
    out = pathlib.Path(path)
    # 导出链路会写 dist/export/png/… 这类"目录可能还不存在"的路径，这里顺手建好。
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def write_rgb(
    path: str | pathlib.Path,
    rgb: np.ndarray,
    *,
    compress_level: int = 6,
    text: Mapping[str, str] | None = None,
) -> pathlib.Path:
    """写 8 位真彩 PNG。``rgb``：``(H, W, 3)`` uint8。返回落盘路径。"""
    return _write(path, rgb, channels=3, name="rgb", compress_level=compress_level, text=text)


def write_rgba(
    path: str | pathlib.Path,
    rgba: np.ndarray,
    *,
    compress_level: int = 6,
    text: Mapping[str, str] | None = None,
) -> pathlib.Path:
    """写 8 位真彩+alpha PNG。``rgba``：``(H, W, 4)`` uint8。返回落盘路径。"""
    return _write(path, rgba, channels=4, name="rgba", compress_level=compress_level, text=text)


def write_gray(
    path: str | pathlib.Path,
    gray: np.ndarray,
    *,
    compress_level: int = 6,
    text: Mapping[str, str] | None = None,
) -> pathlib.Path:
    """写 8 位灰度 PNG。``gray``：``(H, W)`` uint8。返回落盘路径。"""
    return _write(path, gray, channels=1, name="gray", compress_level=compress_level, text=text)


# --------------------------------------------------------------------------------------
# 读回
# --------------------------------------------------------------------------------------


def _paeth(a: int, b: int, c: int) -> int:
    """PNG 规范定义的 Paeth 预测子：三选一，取最接近 ``a+b-c`` 的那个。"""
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter_row(ftype: int, line: np.ndarray, prev: np.ndarray, bpp: int) -> np.ndarray:
    """按过滤类型还原一行（prev = 上一行已还原字节）。

    过滤器 0/1/2 可以向量化（None / Sub 的"左邻"等价于按通道分组的累加和 / Up 的整行相加），
    3/4（Average / Paeth）依赖同一行内**顺序**结果，只能逐字节递推 —— 这也是为什么
    我们写出侧坚持用 0：常见的 PNG 里 0/4 占绝大多数，读回器覆盖全部五种只是为了
    "能读别人改写过的文件"，而不是把它当热路径。
    """
    stride = int(line.shape[0])
    if ftype == 0:
        return line.copy()
    if ftype == 1:  # Sub：x + a
        lanes = line.reshape(-1, bpp).astype(np.int64)
        return (np.cumsum(lanes, axis=0) & 0xFF).astype(np.uint8).reshape(stride)
    if ftype == 2:  # Up：x + b
        return ((line.astype(np.int64) + prev.astype(np.int64)) & 0xFF).astype(np.uint8)
    if ftype == 3:  # Average：x + floor((a + b) / 2)
        cur = np.empty(stride, dtype=np.uint8)
        for i in range(stride):
            a = int(cur[i - bpp]) if i >= bpp else 0
            cur[i] = (int(line[i]) + ((a + int(prev[i])) >> 1)) & 0xFF
        return cur
    if ftype == 4:  # Paeth：x + Paeth(a, b, c)
        cur = np.empty(stride, dtype=np.uint8)
        for i in range(stride):
            a = int(cur[i - bpp]) if i >= bpp else 0
            c = int(prev[i - bpp]) if i >= bpp else 0
            cur[i] = (int(line[i]) + _paeth(a, int(prev[i]), c)) & 0xFF
        return cur
    raise ValueError(f"非法的 PNG 行过滤类型 {ftype}（合法值 0..4）")


def _unfilter(raw: bytes, height: int, width: int, channels: int) -> bytes:
    """把去 zlib 后的原始流还原成无过滤的行主序像素。"""
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError(f"IDAT 解压后长度 {len(raw)} 与 IHDR 推断的 {expected} 不符（文件可能被截断或尺寸字段被改写）")
    out = np.empty((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos)
        pos += stride
        cur = _unfilter_row(ftype, line, prev, channels)
        out[y] = cur
        prev = cur
    return out.tobytes()


def _parse_chunks(blob: bytes) -> list[tuple[bytes, bytes]]:
    """切分 PNG 块流，并逐个校验 CRC32。"""
    if not blob.startswith(_SIGNATURE):
        raise ValueError("不是 PNG：文件签名不匹配（应为 \\x89PNG\\r\\n\\x1a\\n）")
    pos = len(_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    while pos < len(blob):
        if pos + 8 > len(blob):
            raise ValueError(f"PNG 在第 {pos} 字节处被截断：块的头部不完整")
        (length,) = struct.unpack(">I", blob[pos : pos + 4])
        tag = blob[pos + 4 : pos + 8]
        end = pos + 8 + length
        if end + 4 > len(blob):
            raise ValueError(f"PNG 块 {tag!r} 声明长度 {length}，但文件在此处被截断")
        payload = blob[pos + 8 : end]
        (stored_crc,) = struct.unpack(">I", blob[end : end + 4])
        actual_crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise ValueError(
                f"PNG 块 {tag!r} 的 CRC32 校验失败（文件内 {stored_crc:#010x} != 实际 {actual_crc:#010x}）：文件被篡改或损坏"
            )
        chunks.append((tag, payload))
        pos = end + 4
        if tag == b"IEND":
            break
    if not chunks or chunks[0][0] != b"IHDR":
        raise ValueError("PNG 缺少首个 IHDR 块")
    return chunks


def read_png(path: str | pathlib.Path) -> PngImage:
    """读回 8 位、非隔行的灰度(0)/真彩(2)/真彩+alpha(6) PNG。

    支持一行内的全部五种过滤器（0..4），因为 Pillow / 浏览器改写文件时会自适应选择过滤器。
    不支持的形态（调色板、16 位、隔行、灰度+alpha）会抛 ``ValueError`` 并说明原因 ——
    悄悄返回错像素比报错危险得多。
    """
    blob = pathlib.Path(path).read_bytes()
    chunks = _parse_chunks(blob)
    header = chunks[0][1]
    if len(header) != 13:
        raise ValueError(f"IHDR 长度必须是 13 字节，收到 {len(header)}")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", header)
    if color_type not in _CHANNELS_OF_COLOR_TYPE:
        raise ValueError(f"不支持的 PNG 颜色类型 {color_type}（本读回器只支持 0=灰度 / 2=真彩 / 6=真彩+alpha）")
    if bit_depth != 8:
        raise ValueError(f"不支持的 PNG 位深 {bit_depth}（本读回器只支持 8 位）")
    if interlace != 0:
        raise ValueError(f"不支持的 PNG 隔行方式 {interlace}（本读回器只支持非隔行）")
    if compression != 0 or filter_method != 0:
        raise ValueError(f"非法的 IHDR 压缩/过滤方法：compression={compression}, filter_method={filter_method}")
    if width < 1 or height < 1:
        raise ValueError(f"非法的 PNG 尺寸 {width}x{height}")
    channels = _CHANNELS_OF_COLOR_TYPE[color_type]
    idat_parts = [payload for tag, payload in chunks if tag == b"IDAT"]
    if not idat_parts:
        raise ValueError("PNG 缺少 IDAT 块")
    try:
        raw = zlib.decompress(b"".join(idat_parts))
    except zlib.error as exc:
        raise ValueError(f"IDAT 不是合法的 zlib 流：{exc}") from exc
    pixels = _unfilter(raw, height, width, channels)
    return PngImage(width=int(width), height=int(height), channels=channels, pixels=pixels)
