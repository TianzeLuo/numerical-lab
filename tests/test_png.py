"""``tools/png.py`` 的测试：往返、故障路径、以及与 Pillow 的独立交叉验证。

为什么测试里可以 import PIL：Pillow 在这里**不是**实现的一部分，而是独立实现（libpng 派生），
用它来读我们写出的文件是"别人能读"的最强证据；``tools/png.py`` 自身不许 import PIL。

为什么每条过滤器都要测：我们写出侧只用过滤类型 0，但读回侧必须吃下 Pillow/浏览器
改写的文件 —— 那些文件会自适应挑选 0..4。测试里自己构造过滤后的 IDAT，
就不依赖"Pillow 恰好选了某个过滤器"这种运气。
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from tools.png import PngImage, png_bytes, read_png, write_gray, write_rgb, write_rgba

try:  # Pillow 只用于交叉验证；缺失时相关用例跳过而不是让整套测试崩掉。
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover - 本环境已安装 Pillow
    PILImage = None

requires_pillow = pytest.mark.skipif(PILImage is None, reason="需要 Pillow 做独立交叉验证")


# --------------------------------------------------------------------------------------
# 构造测试数据 / 手工构造过滤器
# --------------------------------------------------------------------------------------


def sample_rgb(height: int = 7, width: int = 5) -> np.ndarray:
    """确定性渐变 + 花纹（不用随机数：POLICY P2 禁止 tests 里碰 numpy.random）。"""
    yy = np.arange(height, dtype=np.int64)[:, None]
    xx = np.arange(width, dtype=np.int64)[None, :]
    r = (xx * 37 + yy * 11) % 256
    g = (xx * 5 + yy * 61) % 256
    b = (xx * 97 + yy * 3) % 256
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def sample_rgba(height: int = 6, width: int = 9) -> np.ndarray:
    rgb = sample_rgb(height, width)
    alpha = (255 - np.arange(height * width).reshape(height, width) * 3 % 256).astype(np.uint8)
    return np.concatenate([rgb, alpha[:, :, None]], axis=-1)


def sample_gray(height: int = 4, width: int = 13) -> np.ndarray:
    return ((np.arange(height * width).reshape(height, width) * 7) % 256).astype(np.uint8)


def _paeth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """测试侧独立实现的 Paeth 预测子（与被测代码不共享任何函数）。"""
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def _forward_filter(row: np.ndarray, prev: np.ndarray, ftype: int, bpp: int) -> np.ndarray:
    """对一行做前向过滤（PNG 规范里编码器的动作，读回器要能逆回来）。"""
    x = row.astype(np.int64)
    left = np.zeros_like(x)
    left[bpp:] = x[:-bpp]
    up = prev.astype(np.int64)
    up_left = np.zeros_like(up)
    up_left[bpp:] = up[:-bpp]
    if ftype == 0:
        filtered = x
    elif ftype == 1:
        filtered = x - left
    elif ftype == 2:
        filtered = x - up
    elif ftype == 3:
        filtered = x - ((left + up) // 2)
    else:
        filtered = x - _paeth(left, up, up_left)
    return (filtered & 0xFF).astype(np.uint8)


def build_png_with_filters(array: np.ndarray, ftype: int, *, chunks: int = 1) -> bytes:
    """手工造一个"全部行都用 ftype 过滤"的合法 PNG（IDAT 可切成多块以测分块拼接）。"""
    arr = np.asarray(array)
    height, width = int(arr.shape[0]), int(arr.shape[1])
    channels = 1 if arr.ndim == 2 else int(arr.shape[2])
    stride = width * channels
    flat = arr.reshape(height, stride)
    raw = bytearray()
    prev = np.zeros(stride, dtype=np.uint8)
    for y in range(height):
        row = flat[y]
        raw.append(ftype)
        raw += _forward_filter(row, prev, ftype, channels).tobytes()
        prev = row
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    body = b"\x89PNG\r\n\x1a\n"
    body += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
    compressed = zlib.compress(bytes(raw), 6)
    step = max(1, len(compressed) // chunks)
    for start in range(0, len(compressed), step):
        body += _chunk(b"IDAT", compressed[start : start + step])
    body += _chunk(b"IEND", b"")
    return body


def _chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def write_bytes(tmp_path: Path, data: bytes, name: str = "hand.png") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------------------
# 往返 / 确定性 / 结构
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("writer", "array"),
    [
        (write_rgb, sample_rgb()),
        (write_rgba, sample_rgba()),
        (write_gray, sample_gray()),
    ],
    ids=["rgb", "rgba", "gray"],
)
def test_roundtrip_is_byte_exact(tmp_path: Path, writer, array: np.ndarray) -> None:
    """三种颜色类型的往返必须逐字节相等（这是所有光栅基线的前提）。"""
    path = writer(tmp_path / "img.png", array)
    image = read_png(path)
    assert isinstance(image, PngImage)
    assert (image.width, image.height) == (array.shape[1], array.shape[0])
    assert image.channels == (1 if array.ndim == 2 else array.shape[2])
    assert image.pixels == array.tobytes()
    assert np.array_equal(image.array(), array)


def test_roundtrip_single_pixel_and_single_row(tmp_path: Path) -> None:
    """1×1 与 1×N 是过滤器/步长实现最容易翻车的边界。"""
    one = np.array([[[7, 8, 9]]], dtype=np.uint8)
    assert np.array_equal(read_png(write_rgb(tmp_path / "one.png", one)).array(), one)
    row = sample_rgb(1, 33)
    assert np.array_equal(read_png(write_rgb(tmp_path / "row.png", row)).array(), row)
    column = sample_gray(33, 1)
    assert np.array_equal(read_png(write_gray(tmp_path / "col.png", column)).array(), column)


def test_write_is_deterministic(tmp_path: Path) -> None:
    """同输入两次写出必须逐字节相同：不允许嵌入时间戳/随机数。"""
    array = sample_rgba()
    first = write_rgb(tmp_path / "a.png", sample_rgb()).read_bytes()
    second = write_rgb(tmp_path / "b.png", sample_rgb()).read_bytes()
    assert first == second
    assert png_bytes(sample_rgb()) == first
    assert png_bytes(array) == png_bytes(array)


def test_png_bytes_matches_written_file(tmp_path: Path) -> None:
    array = sample_rgb()
    path = write_rgb(tmp_path / "c.png", array)
    assert path.read_bytes() == png_bytes(array)


def test_file_structure_is_wellformed(tmp_path: Path) -> None:
    """签名 / IHDR 字段 / 块顺序 / 无时间戳块：逐字节核对，防止"看起来能读"就算过。"""
    path = write_rgb(tmp_path / "s.png", sample_rgb(3, 4), text={"Software": "numerical-lab"})
    blob = path.read_bytes()
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert blob.endswith(_chunk(b"IEND", b""))
    length = struct.unpack(">I", blob[8:12])[0]
    assert blob[12:16] == b"IHDR" and length == 13
    width, height, depth, color, comp, filt, interlace = struct.unpack(">IIBBBBB", blob[16:29])
    assert (width, height, depth, color, comp, filt, interlace) == (4, 3, 8, 2, 0, 0, 0)
    assert b"tIME" not in blob  # 时间戳会破坏可复现性
    tags: list[bytes] = []
    pos = 8
    while pos < len(blob):
        size = struct.unpack(">I", blob[pos : pos + 4])[0]
        tags.append(blob[pos + 4 : pos + 8])
        pos += 12 + size
    assert tags[0] == b"IHDR"
    assert b"tEXt" in tags and tags[-1] == b"IEND"
    assert tags.index(b"tEXt") < tags.index(b"IDAT")  # 元数据必须在图像数据之前


def test_compress_level_changes_size_not_pixels(tmp_path: Path) -> None:
    """压缩级别只应影响体积，不应影响解码结果。"""
    array = sample_rgb(40, 40)
    small = write_rgb(tmp_path / "l1.png", array, compress_level=1)
    large = write_rgb(tmp_path / "l9.png", array, compress_level=9)
    assert np.array_equal(read_png(small).array(), array)
    assert np.array_equal(read_png(large).array(), array)
    assert large.stat().st_size <= small.stat().st_size


def test_write_creates_missing_parent_dirs(tmp_path: Path) -> None:
    """导出链路会写 dist/export/png/…，目录可能还不存在。"""
    target = tmp_path / "deep" / "nested" / "img.png"
    assert write_rgb(target, sample_rgb()).is_file()


# --------------------------------------------------------------------------------------
# 五种过滤器
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("ftype", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["gray", "rgb", "rgba"])
def test_all_five_filters_read_back(tmp_path: Path, ftype: int, kind: str) -> None:
    """读回器必须能还原 0..4 全部过滤器（Pillow/浏览器会改写我们的文件）。"""
    array = {"gray": sample_gray(6, 11), "rgb": sample_rgb(6, 11), "rgba": sample_rgba(6, 11)}[kind]
    blob = build_png_with_filters(array, ftype)
    image = read_png(write_bytes(tmp_path, blob, f"f{ftype}-{kind}.png"))
    assert np.array_equal(image.array(), array), f"过滤器 {ftype} / {kind} 还原错误"


@pytest.mark.parametrize("ftype", [1, 2, 3, 4])
def test_filtered_stream_really_differs(ftype: int) -> None:
    """自检：过滤后的字节流确实与原始像素不同，否则上面的用例是空转。"""
    image = sample_rgb(4, 6)
    row = image[1].reshape(-1)
    filtered = _forward_filter(row, image[0].reshape(-1), ftype, 3)
    assert not np.array_equal(filtered, row), "过滤后的字节流与原始像素相同，该用例会是空转"


def test_multi_idat_chunks_are_concatenated(tmp_path: Path) -> None:
    """IDAT 允许分块，读回器必须拼接后再解压。"""
    array = sample_rgb(12, 12)
    blob = build_png_with_filters(array, 4, chunks=5)
    assert blob.count(b"IDAT") >= 5
    assert np.array_equal(read_png(write_bytes(tmp_path, blob)).array(), array)


def test_unknown_filter_type_raises(tmp_path: Path) -> None:
    """非法过滤类型（0..4 之外）必须报错而不是猜。"""
    array = sample_rgb(3, 3)
    raw = bytearray()
    for _ in range(3):
        raw.append(9)  # 非法过滤类型
        raw += array[0].tobytes()
    blob = b"\x89PNG\r\n\x1a\n"
    blob += _chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 3, 8, 2, 0, 0, 0))
    blob += _chunk(b"IDAT", zlib.compress(bytes(raw)))
    blob += _chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="过滤类型"):
        read_png(write_bytes(tmp_path, blob, "badfilter.png"))


# --------------------------------------------------------------------------------------
# tEXt
# --------------------------------------------------------------------------------------


def test_text_chunks_roundtrip(tmp_path: Path) -> None:
    """tEXt 往返：既结构性核对块内容，也让 Pillow 独立读一遍。"""
    text = {"Software": "numerical-lab", "Comment": "frame 3, t=0.25; tEXt is latin-1 only: éüß"}
    path = write_rgb(tmp_path / "t.png", sample_rgb(), text=text)
    blob = path.read_bytes()
    for key, value in text.items():
        expected = _chunk(b"tEXt", key.encode("latin-1") + b"\x00" + value.encode("latin-1"))
        assert expected in blob
    if PILImage is not None:
        with PILImage.open(path) as image:
            image.load()
            assert image.text == text


def test_text_rejects_bad_keywords(tmp_path: Path) -> None:
    array = sample_rgb()
    with pytest.raises(ValueError, match="长度"):
        write_rgb(tmp_path / "k0.png", array, text={"": "x"})
    with pytest.raises(ValueError, match="长度"):
        write_rgb(tmp_path / "k80.png", array, text={"k" * 80: "x"})
    with pytest.raises(ValueError, match="NUL"):
        write_rgb(tmp_path / "kn.png", array, text={"a\x00b": "x"})
    with pytest.raises(ValueError, match="空格"):
        write_rgb(tmp_path / "ks.png", array, text={" lead": "x"})
    with pytest.raises(ValueError, match="Latin-1"):
        write_rgb(tmp_path / "ku.png", array, text={"关键词": "x"})
    with pytest.raises(ValueError, match="Latin-1"):
        write_rgb(tmp_path / "kv.png", array, text={"k": "中文"})
    with pytest.raises(ValueError, match="NUL"):
        write_rgb(tmp_path / "kt.png", array, text={"k": "a\x00b"})
    with pytest.raises(ValueError, match="str"):
        write_rgb(tmp_path / "kw.png", array, text={"k": 1})  # type: ignore[dict-item]


def test_text_keyword_length_boundary(tmp_path: Path) -> None:
    """79 字节是合法上界，不能被"过于保守"的检查误伤。"""
    key = "k" * 79
    path = write_rgb(tmp_path / "edge.png", sample_rgb(), text={key: "v"})
    assert _chunk(b"tEXt", key.encode("latin-1") + b"\x00v") in path.read_bytes()


# --------------------------------------------------------------------------------------
# 非法输入 / 损坏文件
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("array", "match"),
    [
        (np.zeros((4, 4, 3), dtype=np.float32), "uint8"),
        (np.zeros((4, 4, 3), dtype=np.int32), "uint8"),
        (np.zeros((4, 4), dtype=np.uint8), "3 维"),
        (np.zeros((4, 4, 4), dtype=np.uint8), "最后一维"),
        (np.zeros((4, 4, 2), dtype=np.uint8), "最后一维"),
        (np.zeros((16,), dtype=np.uint8), "3 维"),
        (np.zeros((0, 4, 3), dtype=np.uint8), ">= 1"),
        (np.zeros((4, 0, 3), dtype=np.uint8), ">= 1"),
    ],
)
def test_write_rgb_rejects_bad_arrays(tmp_path: Path, array: np.ndarray, match: str) -> None:
    """把 dtype/shape 错误挡在写盘之前，并说清原因。"""
    with pytest.raises(ValueError, match=match):
        write_rgb(tmp_path / "bad.png", array)
    assert not (tmp_path / "bad.png").exists()


def test_write_gray_and_rgba_reject_bad_arrays(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2 维"):
        write_gray(tmp_path / "g.png", sample_rgb())
    with pytest.raises(ValueError, match="4"):
        write_rgba(tmp_path / "a.png", sample_rgb())
    with pytest.raises(ValueError, match="uint8"):
        write_gray(tmp_path / "g2.png", sample_gray().astype(np.float64))


def test_bad_compress_level(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="compress_level"):
        write_rgb(tmp_path / "z.png", sample_rgb(), compress_level=10)
    with pytest.raises(ValueError, match="compress_level"):
        png_bytes(sample_rgb(), compress_level=-1)


def test_non_contiguous_input_is_accepted(tmp_path: Path) -> None:
    """切片产生的非连续数组是常见输入，必须正确处理（而不是解出错位像素）。"""
    big = sample_rgb(20, 20)
    view = big[::2, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    assert np.array_equal(read_png(write_rgb(tmp_path / "v.png", view)).array(), np.ascontiguousarray(view))


def test_tampered_idat_crc_raises(tmp_path: Path) -> None:
    """CRC 被篡改必须报错：否则损坏的光栅基线会被当成"新的真值"。"""
    path = write_rgb(tmp_path / "crc.png", sample_rgb(8, 8))
    blob = bytearray(path.read_bytes())
    idat = blob.index(b"IDAT")
    blob[idat + 6] ^= 0xFF  # 改一个 IDAT 数据字节，CRC 就对不上了
    with pytest.raises(ValueError, match="CRC32"):
        read_png(write_bytes(tmp_path, bytes(blob), "crc_bad.png"))


def test_tampered_ihdr_crc_raises(tmp_path: Path) -> None:
    path = write_rgb(tmp_path / "ihdr.png", sample_rgb(8, 8))
    blob = bytearray(path.read_bytes())
    blob[20] ^= 0x01  # IHDR 里的宽度字段
    with pytest.raises(ValueError, match="CRC32"):
        read_png(write_bytes(tmp_path, bytes(blob), "ihdr_bad.png"))


def test_truncated_and_non_png_raise(tmp_path: Path) -> None:
    path = write_rgb(tmp_path / "full.png", sample_rgb(8, 8))
    blob = path.read_bytes()
    with pytest.raises(ValueError, match="签名"):
        read_png(write_bytes(tmp_path, b"not a png at all", "junk.png"))
    with pytest.raises(ValueError, match="截断"):
        read_png(write_bytes(tmp_path, blob[: len(blob) // 2], "half.png"))
    with pytest.raises(ValueError, match="截断|不完整"):
        read_png(write_bytes(tmp_path, blob[:12], "tiny.png"))


def test_unsupported_color_types_raise(tmp_path: Path) -> None:
    """调色板 / 16 位 / 隔行：明确报错，绝不悄悄返回错像素。"""
    for color_type, depth, interlace, match in [
        (3, 8, 0, "颜色类型"),
        (4, 8, 0, "颜色类型"),
        (0, 16, 0, "位深"),
        (2, 8, 1, "隔行"),
    ]:
        blob = b"\x89PNG\r\n\x1a\n"
        blob += _chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, depth, color_type, 0, 0, interlace))
        blob += _chunk(b"IDAT", zlib.compress(b"\x00" * 8))
        blob += _chunk(b"IEND", b"")
        with pytest.raises(ValueError, match=match):
            read_png(write_bytes(tmp_path, blob, f"ct{color_type}-{depth}-{interlace}.png"))


def test_missing_idat_raises(tmp_path: Path) -> None:
    blob = b"\x89PNG\r\n\x1a\n"
    blob += _chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
    blob += _chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="IDAT"):
        read_png(write_bytes(tmp_path, blob, "noidat.png"))


def test_corrupt_zlib_stream_raises(tmp_path: Path) -> None:
    blob = b"\x89PNG\r\n\x1a\n"
    blob += _chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
    blob += _chunk(b"IDAT", b"\x00\x01\x02\x03")
    blob += _chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="zlib"):
        read_png(write_bytes(tmp_path, blob, "badzlib.png"))


def test_idat_size_mismatch_raises(tmp_path: Path) -> None:
    """解压后长度与 IHDR 推断不符 = 尺寸字段被改写，必须发现。"""
    blob = b"\x89PNG\r\n\x1a\n"
    blob += _chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
    blob += _chunk(b"IDAT", zlib.compress(b"\x00" + b"\x01" * 6))  # 只给了一行的一半
    blob += _chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="不符"):
        read_png(write_bytes(tmp_path, blob, "short.png"))


# --------------------------------------------------------------------------------------
# Pillow 交叉验证（独立实现）
# --------------------------------------------------------------------------------------


@requires_pillow
@pytest.mark.parametrize("size", [(1, 1), (1, 17), (17, 1), (8, 13), (64, 48)])
def test_pillow_reads_our_png(tmp_path: Path, size: tuple[int, int]) -> None:
    """让 Pillow（独立实现）打开我们写的 PNG，比尺寸 + 逐像素。"""
    height, width = size
    for writer, array in (
        (write_rgb, sample_rgb(height, width)),
        (write_rgba, sample_rgba(height, width)),
        (write_gray, sample_gray(height, width)),
    ):
        path = writer(tmp_path / f"x{width}x{height}.png", array)
        with PILImage.open(path) as image:
            image.load()
            assert image.size == (width, height)
            mode = {1: "L", 3: "RGB", 4: "RGBA"}[1 if array.ndim == 2 else array.shape[2]]
            assert image.mode == mode
            assert np.array_equal(np.asarray(image), array)


@requires_pillow
def test_we_read_pillow_written_png(tmp_path: Path) -> None:
    """反方向：Pillow 会用自适应过滤器写文件，我们的读回器要吃下它。"""
    for array, mode in ((sample_rgb(23, 19), "RGB"), (sample_rgba(23, 19), "RGBA"), (sample_gray(23, 19), "L")):
        path = tmp_path / f"pil-{mode}.png"
        PILImage.fromarray(array, mode).save(path)
        image = read_png(path)
        assert np.array_equal(image.array(), array), f"{mode} 从 Pillow 文件读回不一致"


@requires_pillow
def test_pillow_palette_png_is_rejected(tmp_path: Path) -> None:
    """Pillow 写的调色板 PNG 属于"不支持的形态"，必须显式报错。"""
    path = tmp_path / "p.png"
    PILImage.fromarray(sample_rgb(8, 8), "RGB").convert("P").save(path)
    with pytest.raises(ValueError, match="颜色类型"):
        read_png(path)


@requires_pillow
def test_pillow_text_roundtrip_long_value(tmp_path: Path) -> None:
    """tEXt 的值可以很长（可以有换行）；Pillow 读回时必须一字不差。"""
    text = {"Title": "a" * 200, "Note": "line1\nline2"}
    path = write_rgb(tmp_path / "long.png", sample_rgb(4, 4), text=text)
    with PILImage.open(path) as image:
        image.load()
        assert image.text == text
