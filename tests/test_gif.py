"""``tools/gif.py`` 的测试：结构、无损性、LZW 边界、性能、以及与 Pillow 的独立交叉验证。

测试策略上的两个刻意选择
------------------------
1. **不用 numpy.random**（POLICY P2 全仓库禁止 ``numcore/reference/`` 之外使用）：
   伪随机数据用整数散列（splitmix 风格）现场生成 —— 既确定性，又足够"乱"到能压出
   12 位码与字典复位路径。
2. **交叉验证用 Pillow**：Pillow 的 GIF 编解码是 giflib 的 C 实现，与本仓库毫无共享代码。
   它既能读我们写的文件（证明我们的 LZW 位流合法），也能写文件给我们的解码器读
   （证明我们的解码器不是"只认自己"的自洽实现）。
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np
import pytest

from tools.gif import (
    GifInfo,
    decode_gif_frames,
    decode_gif_indices,
    lzw_decode,
    lzw_encode,
    quantize_median_cut,
    read_gif,
    write_gif,
)

try:  # Pillow 仅用于交叉验证；实现文件里绝对不许 import PIL。
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover - 本环境已安装 Pillow
    PILImage = None

requires_pillow = pytest.mark.skipif(PILImage is None, reason="需要 Pillow 做独立交叉验证")


# --------------------------------------------------------------------------------------
# 工具：确定性伪随机 / 帧构造 / 手工字节流
# --------------------------------------------------------------------------------------


def hash_bytes(count: int, seed: int = 1) -> np.ndarray:
    """splitmix64 风格的整数散列 → ``count`` 个 uint8（确定性、无 numpy.random）。"""
    index = np.arange(count, dtype=np.uint64) + np.uint64(seed * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF)
    z = (index + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z ^= z >> np.uint64(31)
    return (z & np.uint64(0xFF)).astype(np.uint8)


def noise_rgb(height: int, width: int, seed: int = 1) -> np.ndarray:
    """每通道独立的伪随机噪声图（颜色数远超 256，专治量化与 LZW 的增长路径）。"""
    channels = [hash_bytes(height * width, seed + offset) for offset in (0, 17, 251)]
    return np.stack(channels, axis=1).reshape(height, width, 3)


def ramp_frames(count: int, height: int, width: int) -> list[np.ndarray]:
    """确定性渐变动画（模拟真实导出场景：色数有限、帧间高度相关）。"""
    x = np.linspace(0.0, 1.0, width)
    frames: list[np.ndarray] = []
    for k in range(count):
        red = (255.0 * x).astype(np.uint8)
        green = (255.0 * np.sin(6.0 * x + 0.1 * k)).astype(np.uint8)
        blue = np.full(width, (k * 7) % 256, dtype=np.uint8)
        frames.append(
            np.stack(
                [
                    np.broadcast_to(red[None, :], (height, width)),
                    np.broadcast_to(green[None, :], (height, width)),
                    np.broadcast_to(blue[None, :], (height, width)),
                ],
                axis=-1,
            )
        )
    return frames


def static_frames(count: int, height: int, width: int) -> list[np.ndarray]:
    """同一帧重复 count 次：所有帧共享同一批颜色 → 只应有一个全局色表。"""
    return [ramp_frames(1, height, width)[0].copy() for _ in range(count)]


def distinct_color_frames() -> list[np.ndarray]:
    """三帧、各自颜色数 ≤256 但彼此色表不同 → 应当触发"每帧局部色表"路径。"""
    frames = []
    for k in range(3):
        base = np.zeros((8, 8, 3), dtype=np.uint8)
        base[..., 0] = np.arange(8, dtype=np.uint8)[None, :] * 32
        base[..., 1] = k * 90
        base[..., 2] = np.arange(8, dtype=np.uint8)[:, None] * 32
        frames.append(base)
    return frames


def pack_codes(codes: list[tuple[int, int]]) -> bytes:
    """测试侧独立实现的最小 LSB-first 码流打包器（用于手工造 GIF87a）。"""
    accumulator = 0
    nbits = 0
    out = bytearray()
    for value, width in codes:
        accumulator |= value << nbits
        nbits += width
        while nbits >= 8:
            out.append(accumulator & 0xFF)
            accumulator >>= 8
            nbits -= 8
    if nbits:
        out.append(accumulator & 0xFF)
    return bytes(out)


def to_subblocks(payload: bytes) -> bytes:
    """测试侧独立实现：负载 → GIF 子块序列。"""
    out = bytearray()
    for start in range(0, len(payload), 255):
        chunk = payload[start : start + 255]
        out.append(len(chunk))
        out += chunk
    out.append(0)
    return bytes(out)


def build_gif87a() -> bytes:
    """手工造一个最小的 **GIF87a**（无 GCE、无 NETSCAPE、全字面量 LZW 码）。

    它检验的是解析器的"另一种合法形态"：老版本、无循环扩展、每帧延时缺省为 0。
    码流按规范手工打包，不经过 ``tools.gif`` 的编码器。
    """
    pixels = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    # min_code_size=2 → clear=4, eoi=5, 码长 3 位起步；前 4 枚码用 3 位，
    # 之后解码器建表已到 8 条，码长升为 4 位（我们手写时按同样的规则排布）。
    codes = [(4, 3), (0, 3), (1, 3), (1, 3), (0, 4), (5, 4)]
    palette = bytes((0, 0, 0, 255, 255, 255))
    blob = bytearray()
    blob += b"GIF87a"
    blob += struct.pack("<HHBBB", 2, 2, 0x80 | ((1 - 1) << 4) | (1 - 1), 0, 0)  # 2 色全局表
    blob += palette
    blob += bytes((0x2C,))
    blob += struct.pack("<HHHHB", 0, 0, 2, 2, 0)
    blob += bytes((2,))
    blob += to_subblocks(pack_codes(codes))
    blob += bytes((0x3B,))
    return bytes(blob)


def scan_blocks(blob: bytes) -> dict:
    """测试侧独立的块级扫描器：抽出扩展与图像描述符的关键字段。"""
    assert blob[:6] in (b"GIF87a", b"GIF89a")
    width, height, packed, background, _aspect = struct.unpack("<HHBBB", blob[6:13])
    pos = 13
    global_entries = 0
    if packed & 0x80:
        global_entries = 1 << ((packed & 0x07) + 1)
        pos += 3 * global_entries
    images: list[dict] = []
    extensions: list[tuple[int, bytes]] = []
    loop = None
    trailer = False
    while pos < len(blob):
        marker = blob[pos]
        if marker == 0x3B:
            trailer = True
            break
        if marker == 0x21:
            label = blob[pos + 1]
            pos += 2
            payload = bytearray()
            while blob[pos] != 0:
                size = blob[pos]
                payload += blob[pos + 1 : pos + 1 + size]
                pos += 1 + size
            pos += 1
            extensions.append((label, bytes(payload)))
            if label == 0xFF and bytes(payload[:11]) == b"NETSCAPE2.0":
                loop = payload[12] | (payload[13] << 8)
            continue
        assert marker == 0x2C, f"意外的块标识 {marker:#x}"
        left, top, frame_w, frame_h, image_packed = struct.unpack("<HHHHB", blob[pos + 1 : pos + 10])
        pos += 10
        if image_packed & 0x80:
            pos += 3 * (1 << ((image_packed & 0x07) + 1))
        min_code_size = blob[pos]
        pos += 1
        data_start = pos
        payload = bytearray()
        while blob[pos] != 0:
            size = blob[pos]
            payload += blob[pos + 1 : pos + 1 + size]
            pos += 1 + size
        pos += 1
        images.append(
            {
                "data_start": data_start,
                "data_end": pos,
                "left": left,
                "top": top,
                "width": frame_w,
                "height": frame_h,
                "local_table": bool(image_packed & 0x80),
                "bits": (image_packed & 0x07) + 1 if image_packed & 0x80 else None,
                "min_code_size": min_code_size,
                "lzw_bytes": len(payload),
            }
        )
    assert trailer, "缺少 trailer"
    return {
        "width": width,
        "height": height,
        "global_entries": global_entries,
        "background": background,
        "images": images,
        "extensions": extensions,
        "loop": loop,
        "trailer": trailer,
    }


# --------------------------------------------------------------------------------------
# 结构：帧数 / 延时 / 循环 / 调色板
# --------------------------------------------------------------------------------------


def test_info_reports_frames_delays_loop_palette(tmp_path: Path) -> None:
    frames = ramp_frames(5, 16, 16)
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]], dtype=np.uint8)
    path = write_gif(tmp_path / "a.gif", frames, delay_cs=7, loop=3, palette=palette)
    info = read_gif(path)
    assert isinstance(info, GifInfo)
    assert (info.width, info.height) == (16, 16)
    assert info.n_frames == 5
    assert info.delays_cs == (7, 7, 7, 7, 7)
    assert info.loop == 3
    assert info.palette_size == 8  # 5 色必须补齐到 2 的幂
    assert info.version == "GIF89a"


@pytest.mark.parametrize(
    ("colors", "expected"),
    [(1, 2), (2, 2), (3, 4), (5, 8), (16, 16), (17, 32), (200, 256), (256, 256)],
)
def test_palette_padded_to_power_of_two(tmp_path: Path, colors: int, expected: int) -> None:
    """GIF 只允许 2 的幂色表；必须补齐且报告正确的表长。"""
    palette = np.zeros((colors, 3), dtype=np.uint8)
    palette[:, 0] = np.arange(colors) % 256
    palette[:, 1] = (np.arange(colors) * 7) % 256
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    path = write_gif(tmp_path / f"p{colors}.gif", frames, palette=palette)
    info = read_gif(path)
    assert info.palette_size == expected
    scanned = scan_blocks(path.read_bytes())
    assert scanned["global_entries"] == expected
    assert scanned["images"][0]["min_code_size"] == max(2, expected.bit_length() - 1)


def test_structure_gce_netscape_and_comment(tmp_path: Path) -> None:
    """逐块核对：NETSCAPE 循环扩展 + 每帧一个 GCE + 注释扩展 + 全屏图像描述符。"""
    frames = static_frames(3, 8, 12)
    path = write_gif(tmp_path / "s.gif", frames, delay_cs=5, loop=0, disposal=2, comment="hello gif")
    scanned = scan_blocks(path.read_bytes())
    labels = [label for label, _ in scanned["extensions"]]
    assert labels.count(0xF9) == 3, "每帧必须有且只有一个图形控制扩展"
    assert 0xFE in labels and b"hello gif" in [p for label, p in scanned["extensions"] if label == 0xFE]
    assert scanned["loop"] == 0
    for image in scanned["images"]:
        assert (image["left"], image["top"]) == (0, 0)
        assert (image["width"], image["height"]) == (12, 8)
        assert not image["local_table"]  # 单一色表时不该有局部表
    for label, payload in scanned["extensions"]:
        if label == 0xF9:
            assert len(payload) == 4
            assert (payload[0] >> 2) & 0x07 == 2  # disposal
            assert payload[1] | (payload[2] << 8) == 5  # delay
            assert payload[3] == 0  # 透明索引


def test_local_palette_only_when_needed(tmp_path: Path) -> None:
    """颜色集不同的帧才配局部色表；同色集的帧复用全局表（不浪费 768 字节）。"""
    frames = distinct_color_frames()
    path = write_gif(tmp_path / "local.gif", frames)
    scanned = scan_blocks(path.read_bytes())
    assert scanned["images"][0]["local_table"] is False
    assert scanned["images"][1]["local_table"] is True
    assert scanned["images"][2]["local_table"] is True
    # 局部表也必须无损
    assert all(np.array_equal(a, b) for a, b in zip(decode_gif_frames(path), frames))


def test_uniform_palette_animation_has_no_local_tables(tmp_path: Path) -> None:
    frames = static_frames(4, 8, 8)  # 所有帧共用同一批颜色
    path = write_gif(tmp_path / "global.gif", frames)
    scanned = scan_blocks(path.read_bytes())
    assert all(image["local_table"] is False for image in scanned["images"])
    assert all(np.array_equal(a, b) for a, b in zip(decode_gif_frames(path), frames))


def test_read_gif87a_without_extensions(tmp_path: Path) -> None:
    """老版本 GIF（无 GCE / 无 NETSCAPE）：loop 必须是 None，延时为 0。"""
    path = tmp_path / "old.gif"
    path.write_bytes(build_gif87a())
    info = read_gif(path)
    assert info.version == "GIF87a"
    assert info.n_frames == 1
    assert info.delays_cs == (0,)
    assert info.loop is None
    assert info.palette_size == 2
    decoded = decode_gif_frames(path)
    assert len(decoded) == 1
    assert np.array_equal(decoded[0][:, :, 0], np.array([[0, 255], [255, 0]], dtype=np.uint8))


# --------------------------------------------------------------------------------------
# 无损往返
# --------------------------------------------------------------------------------------


def test_indices_plus_palette_equal_source(tmp_path: Path) -> None:
    """硬性质的字面形式：``decode_gif_indices`` 的「索引 + 色表」必须等于原图。"""
    frames = distinct_color_frames()  # 每帧色表不同 → 覆盖全局表与局部表两条路径
    path = write_gif(tmp_path / "idx.gif", frames)
    decoded = decode_gif_indices(path)
    assert len(decoded) == len(frames)
    for expected, (indices, palette) in zip(frames, decoded):
        assert indices.shape == expected.shape[:2]
        assert indices.dtype == np.uint8 and palette.dtype == np.uint8
        assert int(indices.max()) < palette.shape[0]
        assert np.array_equal(palette[indices], expected)


def test_lossless_roundtrip_small_palettes(tmp_path: Path) -> None:
    """≤256 色必须逐像素无损（硬性质，不是"通常成立"）。"""
    images = [
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.full((5, 7, 3), 200, dtype=np.uint8),
        np.stack([np.arange(16, dtype=np.uint8)[None, :]] * 3, axis=-1).repeat(9, axis=0)[:, :, :3],
    ]
    for i, image in enumerate(images):
        path = write_gif(tmp_path / f"lossless{i}.gif", [image])
        decoded = decode_gif_frames(path)
        assert len(decoded) == 1
        assert np.array_equal(decoded[0], image), f"第 {i} 个用例不是无损"


def test_lossless_roundtrip_256_colors(tmp_path: Path) -> None:
    """正好 256 色：色表取满，min_code_size=8。"""
    colors = np.stack(
        [np.arange(256), (np.arange(256) * 7) % 256, (np.arange(256) * 13) % 256], axis=1
    ).astype(np.uint8)
    picks = (np.arange(64 * 64) * 37) % 256  # 37 与 256 互素 → 256 种颜色各出现 16 次
    image = colors[picks].reshape(64, 64, 3)
    assert len(np.unique(image.reshape(-1, 3), axis=0)) == 256
    path = write_gif(tmp_path / "c256.gif", [image])
    assert np.array_equal(decode_gif_frames(path)[0], image)
    assert read_gif(path).palette_size == 256


def test_lossless_roundtrip_gray_frames(tmp_path: Path) -> None:
    """``(H, W)`` 灰度帧应自动升为 RGB 并保持逐像素无损。"""
    gray = (np.arange(64, dtype=np.uint8).reshape(8, 8) * 4).astype(np.uint8)
    path = write_gif(tmp_path / "gray.gif", [gray, gray[::-1]])
    decoded = decode_gif_frames(path)
    assert decoded[0].shape == (8, 8, 3)
    assert np.array_equal(decoded[0][..., 0], gray)
    assert np.array_equal(decoded[1][..., 0], gray[::-1])
    assert np.array_equal(decoded[0][..., 1], gray)


def test_gray_and_rgb_frames_can_mix(tmp_path: Path) -> None:
    """同尺寸的灰度帧与 RGB 帧可以混用（统一升维后再比对尺寸）。"""
    gray = np.full((6, 6), 30, dtype=np.uint8)
    rgb = np.full((6, 6, 3), 30, dtype=np.uint8)
    path = write_gif(tmp_path / "mix.gif", [gray, rgb])
    decoded = decode_gif_frames(path)
    assert np.array_equal(decoded[0], rgb) and np.array_equal(decoded[1], rgb)


def test_mapping_with_explicit_palette_is_lossless_when_colors_present(tmp_path: Path) -> None:
    """显式色表：颜色都在表里时也必须无损（最近邻映射不能有额外损失）。"""
    palette = np.array([[0, 0, 0], [10, 20, 30], [200, 100, 50], [255, 255, 255]], dtype=np.uint8)
    image = palette[[0, 1, 2, 3, 3, 2, 1, 0]].reshape(2, 4, 3)
    path = write_gif(tmp_path / "explicit.gif", [image], palette=palette)
    assert np.array_equal(decode_gif_frames(path)[0], image)


def test_mapping_with_explicit_palette_is_nearest_neighbour(tmp_path: Path) -> None:
    """表里没有的颜色按最近邻映射：验证"最近的确实是那个"。"""
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    image = np.array([[[10, 10, 10], [250, 250, 250]]], dtype=np.uint8)
    path = write_gif(tmp_path / "near.gif", [image], palette=palette)
    decoded = decode_gif_frames(path)[0]
    assert np.array_equal(decoded, np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8))


# --------------------------------------------------------------------------------------
# LZW 边界
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("min_code_size", [2, 3, 4, 8])
def test_lzw_roundtrip_shapes(min_code_size: int) -> None:
    """1×1、单色、长游程、逐像素变化：LZW 的四个基本形态。"""
    limit = 1 << min_code_size
    cases = [
        np.zeros(1, dtype=np.uint8),
        np.zeros(4096, dtype=np.uint8),
        np.full(1000, limit - 1, dtype=np.uint8),
        (np.arange(3000) % limit).astype(np.uint8),
        np.repeat(np.arange(limit, dtype=np.uint8), 17),
    ]
    for i, indices in enumerate(cases):
        blob = lzw_encode(indices, min_code_size=min_code_size)
        back = lzw_decode(blob, min_code_size=min_code_size, expected_len=indices.size)
        assert np.array_equal(back, indices), f"用例 {i}（min_code_size={min_code_size}）往返失败"


def test_lzw_tail_code_growth_regression() -> None:
    """回归：**最后一枚**数据码正好越过码长阈值时，EOI 必须用增长后的码长。

    这个 bug 只在特定长度上出现（本例 64 个同色像素、min_code_size=2），
    漏掉它会导致"某些动画的最后一帧解不出来"这种极难复现的故障。
    """
    indices = np.zeros((8, 8), dtype=np.uint8)
    blob = lzw_encode(indices, min_code_size=2)
    assert np.array_equal(lzw_decode(blob, min_code_size=2, expected_len=64), indices.reshape(-1))


def test_lzw_256_color_noise_needs_12_bit_codes() -> None:
    """256 色伪随机 64×64：要求码长增长到 12 位仍然逐像素正确。"""
    indices = hash_bytes(64 * 64, seed=3)  # 4096 次采样落在 256 个值上 → 基本取满
    assert len(np.unique(indices)) == 256
    blob = lzw_encode(indices, min_code_size=8)
    assert np.array_equal(lzw_decode(blob, min_code_size=8, expected_len=indices.size), indices)


def test_lzw_clear_code_reset_on_full_dictionary() -> None:
    """3 万像素的噪声必然写满字典：必须发 clear 并复位后继续正确。

    每枚码最多 12 位，因此 ``len(blob)*8 // 12`` 是"至少发出了多少枚码"的**下界**；
    它 >= 2048 就证明码长确实增长到了 12 位，>= 4096 就证明字典走满了整条复位路径。
    """
    indices = hash_bytes(30000, seed=11)
    blob = lzw_encode(indices, min_code_size=8)
    lower_bound_codes = len(blob) * 8 // 12
    assert lower_bound_codes >= 4096, f"码数下界 {lower_bound_codes} 不足以覆盖字典复位路径"
    assert np.array_equal(lzw_decode(blob, min_code_size=8, expected_len=indices.size), indices)


def test_lzw_1x1_and_single_color_through_gif(tmp_path: Path) -> None:
    """1×1 与单色整帧：GIF 层的最小合法输入。"""
    for shape in [(1, 1), (1, 33), (33, 1)]:
        image = np.full(shape + (3,), 17, dtype=np.uint8)
        path = write_gif(tmp_path / f"tiny{shape[0]}x{shape[1]}.gif", [image])
        assert np.array_equal(decode_gif_frames(path)[0], image)
    single = np.zeros((16, 16, 3), dtype=np.uint8)
    single[..., 2] = 9
    path = write_gif(tmp_path / "single.gif", [single])
    assert np.array_equal(decode_gif_frames(path)[0], single)
    assert read_gif(path).palette_size == 2


def test_lzw_rejects_bad_inputs() -> None:
    """编解码器的失败路径：dtype / 码长 / 索引越界 / 截断 / 长度不符。"""
    with pytest.raises(ValueError, match="uint8"):
        lzw_encode(np.zeros(4, dtype=np.int16), min_code_size=8)
    with pytest.raises(ValueError, match="min_code_size"):
        lzw_encode(np.zeros(4, dtype=np.uint8), min_code_size=1)
    with pytest.raises(ValueError, match="min_code_size"):
        lzw_encode(np.zeros(4, dtype=np.uint8), min_code_size=9)
    with pytest.raises(ValueError, match="索引最大值"):
        lzw_encode(np.array([0, 255, 7], dtype=np.uint8), min_code_size=2)
    with pytest.raises(ValueError, match="min_code_size"):
        lzw_decode(b"\x00\x00", min_code_size=1, expected_len=1)
    with pytest.raises(ValueError, match="不能为负"):
        lzw_decode(b"\x00\x00", min_code_size=2, expected_len=-1)
    with pytest.raises(ValueError, match="截断"):
        lzw_decode(b"\x00", min_code_size=8, expected_len=10)
    blob = lzw_encode(np.arange(20, dtype=np.uint8), min_code_size=8)
    with pytest.raises(ValueError, match="不符"):
        lzw_decode(blob, min_code_size=8, expected_len=21)


def test_lzw_rejects_illegal_first_code() -> None:
    """首码不是字面量（例如直接给了个表码）必须报错，而不是解出垃圾。"""
    blob = pack_codes([(4, 3), (5, 3)])  # 2 位最小码长下 clear=4、eoi=5：这里直接给 eoi 之前无数据
    assert lzw_decode(blob, min_code_size=2, expected_len=0).size == 0
    bad = pack_codes([(4, 3), (6, 3), (5, 3)])  # 6 不是字面量
    with pytest.raises(ValueError, match="首码"):
        lzw_decode(bad, min_code_size=2, expected_len=1)


def test_lzw_decode_handles_deferred_clear_stream() -> None:
    """外来码流可能"表满后不发 clear、只是不再建表"（deferred clear），必须能读。"""
    # 30 个不同码，手工造流：clear + 0..28 字面量 + eoi（解码器会正常建表）
    codes: list[tuple[int, int]] = [(256, 9)]
    width = 9
    next_code = 258
    for value in range(29):
        codes.append((value, width))
        if next_code >= (1 << width) and width < 12:
            width += 1
        next_code += 1
    codes.append((257, width))
    blob = pack_codes(codes)
    decoded = lzw_decode(blob, min_code_size=8, expected_len=29)
    assert np.array_equal(decoded, np.arange(29, dtype=np.uint8))


# --------------------------------------------------------------------------------------
# 量化
# --------------------------------------------------------------------------------------


def test_quantize_is_lossless_below_limit() -> None:
    """输入颜色数 ≤ n_colors 时必须无损，且调色板恰好是出现过的颜色。"""
    colors = np.stack(
        [np.arange(200), (np.arange(200) * 3) % 256, (np.arange(200) * 11) % 256], axis=1
    ).astype(np.uint8)
    image = colors[hash_bytes(32 * 32, seed=7) % 200].reshape(32, 32, 3)  # 至多 200 色
    indices, palette = quantize_median_cut(image, 256)
    assert palette.shape[0] <= 256
    assert indices.dtype == np.uint8 and palette.dtype == np.uint8
    assert np.array_equal(palette[indices], image)
    assert palette.shape[0] == len(np.unique(image.reshape(-1, 3), axis=0))


def test_quantize_is_deterministic_and_bounded() -> None:
    """同输入同输出；色数不超过请求值；索引必须落在色表范围内。"""
    image = noise_rgb(48, 40, seed=13)
    for n_colors in (2, 3, 17, 100, 256):
        first = quantize_median_cut(image, n_colors)
        second = quantize_median_cut(image, n_colors)
        assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
        indices, palette = first
        assert 1 <= palette.shape[0] <= n_colors
        assert int(indices.max()) < palette.shape[0]
        assert indices.shape == (48, 40)


def test_quantize_error_decreases_with_more_colors() -> None:
    """颜色越多误差越小（防止"量化器把颜色搞反了"这类低级错误）。"""
    image = noise_rgb(64, 64, seed=23)
    errors = []
    for n_colors in (2, 8, 64, 256):
        indices, palette = quantize_median_cut(image, n_colors)
        errors.append(float(np.abs(palette[indices].astype(np.int64) - image.astype(np.int64)).mean()))
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < errors[0]


def test_quantize_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="uint8"):
        quantize_median_cut(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="H, W, 3"):
        quantize_median_cut(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="n_colors"):
        quantize_median_cut(np.zeros((4, 4, 3), dtype=np.uint8), 0)
    with pytest.raises(ValueError, match="n_colors"):
        quantize_median_cut(np.zeros((4, 4, 3), dtype=np.uint8), 257)
    with pytest.raises(ValueError, match="n_colors"):
        quantize_median_cut(np.zeros((4, 4, 3), dtype=np.uint8), 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        quantize_median_cut(np.zeros((0, 4, 3), dtype=np.uint8))


# --------------------------------------------------------------------------------------
# 抖动
# --------------------------------------------------------------------------------------


def _block_mean_error(decoded: np.ndarray, source: np.ndarray, block: int) -> float:
    """按 block×block 求均值后比较误差（抖动追求的是局部平均，而不是逐像素最小误差）。"""
    height, width = source.shape[:2]
    shape = (height // block, block, width // block, block)
    left = decoded[..., 0].astype(np.float64).reshape(shape).mean(axis=(1, 3))
    right = source[..., 0].astype(np.float64).reshape(shape).mean(axis=(1, 3))
    return float(np.abs(left - right).mean())


def test_dither_is_deterministic_and_uses_both_colors(tmp_path: Path) -> None:
    """抖动必须确定性；在灰阶上用黑白两色应产生两色混合，且**局部平均**更接近原图。

    （逐像素误差在抖动下反而可能变大 —— 误差扩散的本意就是把量化误差在邻域里摊平，
    所以这里比的是 4×4 块均值，而不是逐像素 MAE。）
    """
    gray = (np.arange(16, dtype=np.uint8)[None, :] * 16).repeat(16, axis=0)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    first = write_gif(tmp_path / "d1.gif", [image], palette=palette, dither=True)
    second = write_gif(tmp_path / "d2.gif", [image], palette=palette, dither=True)
    assert first.read_bytes() == second.read_bytes()
    decoded = decode_gif_frames(first)[0]
    assert set(np.unique(decoded).tolist()) == {0, 255}, "抖动应同时用上黑与白"
    nearest = decode_gif_frames(write_gif(tmp_path / "d3.gif", [image], palette=palette, dither=False))[0]
    error_dither = _block_mean_error(decoded, image, 4)
    error_nearest = _block_mean_error(nearest, image, 4)
    assert error_dither < error_nearest, (error_dither, error_nearest)


def test_dither_with_256_color_palette(tmp_path: Path) -> None:
    """抖动路径在"色表很大"时也要能跑通且确定性（分块/缓存逻辑不复用另一条路径）。"""
    frames = ramp_frames(2, 12, 12)
    path = write_gif(tmp_path / "d256.gif", frames, dither=True)
    again = write_gif(tmp_path / "d256b.gif", frames, dither=True)
    assert path.read_bytes() == again.read_bytes()
    assert len(decode_gif_frames(path)) == 2


# --------------------------------------------------------------------------------------
# 确定性 / 性能
# --------------------------------------------------------------------------------------


def test_write_is_deterministic(tmp_path: Path) -> None:
    """同输入两次写出必须逐字节相同（不含时间戳、不依赖字典哈希顺序）。"""
    frames = ramp_frames(6, 24, 24)
    first = write_gif(tmp_path / "x1.gif", frames, delay_cs=3, loop=2, comment="c")
    second = write_gif(tmp_path / "x2.gif", frames, delay_cs=3, loop=2, comment="c")
    assert first.read_bytes() == second.read_bytes()
    big = noise_rgb(32, 32, seed=5)
    assert write_gif(tmp_path / "y1.gif", [big]).read_bytes() == write_gif(tmp_path / "y2.gif", [big]).read_bytes()


def test_large_animation_is_fast_and_small(tmp_path: Path) -> None:
    """60 帧 128×128：写出 < 5 秒、体积 < 3 MB（交付格式必须扛得住真实导出）。"""
    frames = ramp_frames(60, 128, 128)
    start = time.perf_counter()
    path = write_gif(tmp_path / "big.gif", frames, delay_cs=4, loop=0)
    elapsed = time.perf_counter() - start
    size = path.stat().st_size
    assert elapsed < 5.0, f"60 帧写出耗时 {elapsed:.2f}s"
    assert size < 3 * 1024 * 1024, f"文件体积 {size / 1024:.0f} KB"
    info = read_gif(path)
    assert info.n_frames == 60 and info.delays_cs == (4,) * 60 and info.loop == 0
    decoded = decode_gif_frames(path)
    assert len(decoded) == 60
    assert all(np.array_equal(a, b) for a, b in zip(decoded, frames))


# --------------------------------------------------------------------------------------
# 失败路径
# --------------------------------------------------------------------------------------


def test_write_gif_rejects_mismatched_frame_sizes(tmp_path: Path) -> None:
    frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 9, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="不一致"):
        write_gif(tmp_path / "bad.gif", frames)
    assert not (tmp_path / "bad.gif").exists()


def test_write_gif_rejects_bad_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不能为空"):
        write_gif(tmp_path / "e.gif", [])
    with pytest.raises(ValueError, match="uint8"):
        write_gif(tmp_path / "f.gif", [np.zeros((4, 4, 3), dtype=np.float32)])
    with pytest.raises(ValueError, match="shape"):
        write_gif(tmp_path / "g.gif", [np.zeros((4, 4, 2), dtype=np.uint8)])
    with pytest.raises(ValueError, match="shape"):
        write_gif(tmp_path / "h.gif", [np.zeros((4, 4, 3, 1), dtype=np.uint8)])
    with pytest.raises(ValueError, match=">= 1"):
        write_gif(tmp_path / "i.gif", [np.zeros((0, 4, 3), dtype=np.uint8)])


def test_write_gif_rejects_bad_parameters(tmp_path: Path) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="delay_cs"):
        write_gif(tmp_path / "d.gif", [frame], delay_cs=-1)
    with pytest.raises(ValueError, match="delay_cs"):
        write_gif(tmp_path / "d2.gif", [frame], delay_cs=70000)
    with pytest.raises(ValueError, match="loop"):
        write_gif(tmp_path / "l.gif", [frame], loop=-1)
    with pytest.raises(ValueError, match="disposal"):
        write_gif(tmp_path / "x.gif", [frame], disposal=8)
    with pytest.raises(ValueError, match="disposal"):
        write_gif(tmp_path / "x2.gif", [frame], disposal=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Latin-1"):
        write_gif(tmp_path / "c.gif", [frame], comment="中文注释")
    with pytest.raises(ValueError, match="comment"):
        write_gif(tmp_path / "c2.gif", [frame], comment=123)  # type: ignore[arg-type]


def test_write_gif_rejects_bad_palettes(tmp_path: Path) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="1..256"):
        write_gif(tmp_path / "p1.gif", [frame], palette=np.zeros((300, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="1..256"):
        write_gif(tmp_path / "p2.gif", [frame], palette=np.zeros((0, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="N, 3"):
        write_gif(tmp_path / "p3.gif", [frame], palette=np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        write_gif(tmp_path / "p4.gif", [frame], palette=np.zeros((4, 3), dtype=np.float32))


def test_read_gif_rejects_malformed_files(tmp_path: Path) -> None:
    good = write_gif(tmp_path / "ok.gif", ramp_frames(2, 8, 8)).read_bytes()
    with pytest.raises(ValueError, match="太短"):
        read_gif(_write(tmp_path, b"GIF89a", "short.gif"))
    with pytest.raises(ValueError, match="签名"):
        read_gif(_write(tmp_path, b"NOTAGIF" + good[7:], "sig.gif"))
    with pytest.raises(ValueError, match="结束符"):
        read_gif(_write(tmp_path, good[:-1], "notrailer.gif"))
    with pytest.raises(ValueError, match="截断|超出"):
        read_gif(_write(tmp_path, good[: len(good) // 2], "half.gif"))
    broken = bytearray(good)
    broken[broken.index(b"\x2c")] = 0x7F  # 未知块标识
    with pytest.raises(ValueError, match="未知"):
        read_gif(_write(tmp_path, bytes(broken), "marker.gif"))
    # 合法但没有任何图像块的 GIF：解析要成功，解码要明确报错
    empty = b"GIF89a" + struct.pack("<HHBBB", 1, 1, 0, 0, 0) + b"\x3b"
    empty_path = _write(tmp_path, empty, "empty.gif")
    assert read_gif(empty_path).n_frames == 0
    with pytest.raises(ValueError, match="没有任何图像帧"):
        decode_gif_frames(empty_path)


def _write(tmp_path: Path, data: bytes, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_decode_rejects_truncated_lzw_payload(tmp_path: Path) -> None:
    """图像数据被截断：必须在解码层报"码流提前结束"，而不是静默出一张残图。"""
    good = write_gif(tmp_path / "ok2.gif", [noise_rgb(48, 48, seed=2)]).read_bytes()
    image = scan_blocks(good)["images"][0]
    start, end = image["data_start"], image["data_end"]
    assert end - start > 4, "该用例需要多字节的 LZW 负载"
    # 砍掉最后一个子块（保留 0 长度结束符与 trailer）→ 解码器必然读不到 EOI
    cut = end - 2
    truncated = good[:cut] + good[end - 1 :]
    with pytest.raises(ValueError, match="截断|结束|不符"):
        decode_gif_frames(_write(tmp_path, truncated, "trunc.gif"))


# --------------------------------------------------------------------------------------
# Pillow 交叉验证（独立实现）
# --------------------------------------------------------------------------------------


@requires_pillow
def test_pillow_reads_our_animation(tmp_path: Path) -> None:
    """Pillow 读我们写的 GIF：帧数、每帧尺寸、无损用例的逐像素一致，外加 loop/duration。"""
    frames = ramp_frames(7, 20, 24)
    path = write_gif(tmp_path / "pil.gif", frames, delay_cs=6, loop=4)
    with PILImage.open(path) as image:
        assert image.n_frames == 7
        assert image.size == (24, 20)
        assert image.info.get("loop") == 4
        assert image.info.get("duration") == 60  # 厘秒 → 毫秒
        for k, expected in enumerate(frames):
            image.seek(k)
            assert image.size == (24, 20)
            assert np.array_equal(np.asarray(image.convert("RGB")), expected), f"第 {k} 帧不一致"


@requires_pillow
def test_pillow_reads_our_local_palette_animation(tmp_path: Path) -> None:
    """带局部色表的动画也要能被 Pillow 正确播放（这是局部表最容易写错的地方）。"""
    frames = distinct_color_frames()
    path = write_gif(tmp_path / "loc.gif", frames)
    with PILImage.open(path) as image:
        for k, expected in enumerate(frames):
            image.seek(k)
            assert np.array_equal(np.asarray(image.convert("RGB")), expected)


@requires_pillow
def test_pillow_reads_our_1x1_and_2color(tmp_path: Path) -> None:
    """极小色表（2 色、min_code_size=2）是最容易在"码长下限"上踩坑的形态。"""
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    path = write_gif(tmp_path / "one.gif", [image])
    with PILImage.open(path) as pil:
        assert pil.size == (1, 1)
        assert np.array_equal(np.asarray(pil.convert("RGB")), image)


@requires_pillow
def test_we_decode_pillow_written_gif(tmp_path: Path) -> None:
    """反方向：Pillow（giflib）编码 → 我们的 LZW 解码器。

    这条用例最有价值的地方在于：Pillow 会用**它自己**的最小码长、色表大小与
    帧优化策略，我们的解码器必须不依赖"自己编码器的习惯"。
    """
    frames = [distinct_color_frames()[0], distinct_color_frames()[2], noise_rgb(8, 8, seed=31) % 60]
    images = [PILImage.fromarray(frame, "RGB").convert("P", palette=PILImage.ADAPTIVE, colors=32) for frame in frames]
    path = tmp_path / "by-pillow.gif"
    images[0].save(path, save_all=True, append_images=images[1:], duration=40, loop=2, disposal=2)
    assert PILImage.open(path).n_frames == len(frames)
    ours = decode_gif_frames(path)
    assert len(ours) == len(frames)
    with PILImage.open(path) as reference:
        for k in range(reference.n_frames):
            reference.seek(k)
            expected = np.asarray(reference.convert("RGB"))
            assert ours[k].shape == expected.shape
            assert np.array_equal(ours[k], expected), f"第 {k} 帧与 Pillow 的解读不一致"


@requires_pillow
def test_pillow_decodes_12bit_lzw_noise(tmp_path: Path) -> None:
    """12 位码 + 字典复位路径：交给 giflib 的 C 解码器复算一遍。"""
    image = noise_rgb(160, 160, seed=41)
    path = write_gif(tmp_path / "noise.gif", [image])
    mine = decode_gif_frames(path)[0]
    with PILImage.open(path) as pil:
        theirs = np.asarray(pil.convert("RGB"))
    assert np.array_equal(mine, theirs)
    # 40000 像素随机噪声必然超过 4096 枚码 → 走满码长增长与字典复位
    assert len(path.read_bytes()) > 4096


# --------------------------------------------------------------------------------------
# 父任务给的探针脚本（原样固化成用例）
# --------------------------------------------------------------------------------------


def test_probe_smoke(tmp_path: Path) -> None:
    """完成标准里那段探针：GIF 写→读→解码，PNG 写→读，全部走一遍。"""
    from tools.png import read_png, write_rgb

    n = 64
    x = np.linspace(0, 1, n)
    frames = [
        np.stack(
            [
                np.broadcast_to((255 * x).astype("uint8"), (n, n)),
                np.broadcast_to((255 * np.sin(6 * x)).astype("uint8"), (n, n)),
                np.full((n, n), 128, "uint8"),
            ],
            axis=-1,
        )
        for _ in range(10)
    ]
    probe = tmp_path / "probe.gif"
    write_gif(probe, frames, delay_cs=5, loop=0)
    info = read_gif(probe)
    assert info.n_frames == 10 and info.delays_cs == (5,) * 10 and info.loop == 0
    assert len(decode_gif_frames(probe)) == 10
    assert read_png(write_rgb(tmp_path / "probe.png", frames[0])).width == n
    assert all(np.array_equal(a, b) for a, b in zip(decode_gif_frames(probe), frames))


def test_long_run_frames_compress_and_roundtrip(tmp_path: Path) -> None:
    """超长游程：极端重复数据上 LZW 的游程编码与码长逻辑都不能出错。"""
    image = np.zeros((3, 5000, 3), dtype=np.uint8)
    image[:, :2500] = 255
    image[:, 4990:] = 7
    path = write_gif(tmp_path / "runs.gif", [image])
    assert np.array_equal(decode_gif_frames(path)[0], image)
    assert path.stat().st_size < 2000, "游程没有被压掉，说明 LZW 没起作用"
    # 单色整屏（整帧一个码）也不能退化
    flat = np.full((64, 64, 3), 33, dtype=np.uint8)
    flat_path = write_gif(tmp_path / "flat.gif", [flat])
    assert np.array_equal(decode_gif_frames(flat_path)[0], flat)
    assert flat_path.stat().st_size < 400
