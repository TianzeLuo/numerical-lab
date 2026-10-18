"""``.nlf`` 帧容器的测试。

这个文件守的是**跨语言契约**：TypeScript 侧 `viz/shared/src/nlf.ts` 必须能解码同样的字节，
两侧共用 `tests/fixtures/nlf/*.nlf` 与 `tests/parity/golden/nlf_*.json`。
因此这里除了往返测试，还要逐条验证**拒绝服务**的行为——一个"宽容"的解码器
会在前端悄悄画出错误的图，那正是本项目最怕的失败模式。
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from server.nlf import (
    NLF_MAGIC,
    NLF_VERSION,
    Frame,
    NlfError,
    frame_summary,
    manifest_frame,
    pack_frame,
    read_frame,
    unpack_frame,
    write_frame,
)


def _basic_frame() -> Frame:
    fr = Frame(chapter="0.0", viz="unit", kind="frame")
    fr.add("u", np.linspace(0.0, 1.0, 5), unit="1", label="u(x)")
    fr.add("M", np.arange(6, dtype=float).reshape(2, 3))
    fr.add("flags", np.array([True, False, True]))
    fr.add("idx", np.array([-2, 0, 7], dtype=np.int32))
    fr.set_meta(t=0.25, n=5, ok=True)
    fr.set_strings(status="ok")
    fr.critical = {"r": 0.5}
    fr.pitfalls = "固定坐标轴会掩盖发散。"
    return fr


# --------------------------------------------------------------------------- #
# 往返
# --------------------------------------------------------------------------- #
def test_roundtrip_preserves_arrays_and_metadata() -> None:
    fr = _basic_frame()
    back = unpack_frame(pack_frame(fr))
    assert back.chapter == "0.0" and back.viz == "unit" and back.kind == "frame"
    assert back.codec == "none" and back.pitfalls == fr.pitfalls
    assert back.meta == {"t": 0.25, "n": 5, "ok": True}
    assert back.strings == {"status": "ok"}
    assert back.critical == {"r": 0.5}
    assert back.names == ["u", "M", "flags", "idx"]
    assert np.array_equal(back["u"], fr["u"])
    assert np.array_equal(back["M"], fr["M"])
    assert back["flags"].dtype == np.bool_ and bool(back["flags"][1]) is False
    assert back["idx"].dtype == np.int32 and int(back["idx"][0]) == -2
    assert back.shape_of("M") == (2, 3)
    assert back.scalar("t") == 0.25 and back.scalar("missing") is None


def test_roundtrip_fortran_order_keeps_values_and_flag() -> None:
    A = np.asfortranarray(np.arange(12, dtype=float).reshape(3, 4))
    fr = Frame(chapter="0.0", viz="f")
    fr.add("AF", A, order="F")
    fr.add("AC", A, order="C")
    back = unpack_frame(pack_frame(fr))
    assert np.array_equal(back["AF"], np.asarray(A))
    assert np.array_equal(back["AF"], back["AC"])
    order_of = {s.name: s.order for s in back.specs}
    assert order_of["AF"] == "F" and order_of["AC"] == "C"


@pytest.mark.parametrize("codec", ["zlib", "gzip"])
def test_compression_roundtrip_and_plaintext_header(codec: str) -> None:
    fr = Frame(chapter="0.0", viz="big", codec=codec)
    fr.add("field", np.sin(np.linspace(0, 10, 4096)))
    blob = pack_frame(fr)
    (hlen,) = struct.unpack_from("<I", blob, 4)
    header_bytes = blob[8 : 8 + hlen]
    assert header_bytes.startswith(b"{")  # header 永远是明文
    assert f'"codec":"{codec}"'.encode() in header_bytes
    assert len(blob) < 4096 * 8  # 正弦数据可压缩
    back = unpack_frame(blob)
    assert np.allclose(back["field"], fr["field"])


def test_pack_is_deterministic_with_fixed_generator() -> None:
    fr = _basic_frame()
    fr.generator = {"tool": "test", "format": "0.1.0"}
    assert pack_frame(fr) == pack_frame(_basic_frame_with_fixed_generator())


def _basic_frame_with_fixed_generator() -> Frame:
    fr = _basic_frame()
    fr.generator = {"tool": "test", "format": "0.1.0"}
    return fr


def test_frame_summary_and_manifest_frame() -> None:
    fr = _basic_frame()
    text = frame_summary(fr)
    assert "unit" in text and "u:" in text and "meta" in text
    mf = manifest_frame("0.0", "unit", {"vizzes": [{"id": "a"}, {"id": "b"}]})
    back = unpack_frame(pack_frame(mf))
    assert back.kind == "manifest"
    assert json.loads(back.strings["manifest_json"])["vizzes"][0]["id"] == "a"
    assert back.scalar("n_viz") == 2


def test_write_and_read_frame_file(tmp_path) -> None:
    path = write_frame(tmp_path / "sub" / "x.nlf", _basic_frame())
    assert path.exists()
    back = read_frame(path)
    assert back.names == _basic_frame().names


# --------------------------------------------------------------------------- #
# 拒绝服务：所有自洽性问题都必须抛 NlfError
# --------------------------------------------------------------------------- #
def _good_blob() -> bytes:
    return pack_frame(_basic_frame())


def test_rejects_short_and_bad_magic() -> None:
    with pytest.raises(NlfError, match="太短"):
        unpack_frame(b"NLF")
    bad = bytearray(_good_blob())
    bad[:4] = b"XXXX"
    with pytest.raises(NlfError, match="magic"):
        unpack_frame(bytes(bad))


def test_rejects_bad_header_length_and_json() -> None:
    blob = _good_blob()
    with pytest.raises(NlfError, match="header_len"):
        unpack_frame(blob[:4] + struct.pack("<I", 10**9) + blob[8:])
    with pytest.raises(NlfError, match="header_len"):
        unpack_frame(blob[:4] + struct.pack("<I", len(blob)) + blob[8:])
    broken = NLF_MAGIC + struct.pack("<I", 5) + b"{oops" + b"\x00" * 8
    with pytest.raises(NlfError, match="JSON"):
        unpack_frame(broken)


def _mutate_header(blob: bytes, **changes) -> bytes:
    (hlen,) = struct.unpack_from("<I", blob, 4)
    header = json.loads(blob[8 : 8 + hlen].decode("utf-8"))
    header.update(changes)
    hb = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return NLF_MAGIC + struct.pack("<I", len(hb)) + hb + blob[8 + hlen :]


def test_rejects_version_endian_and_codec() -> None:
    blob = _good_blob()
    with pytest.raises(NlfError, match="版本"):
        unpack_frame(_mutate_header(blob, version=NLF_VERSION + 1))
    with pytest.raises(NlfError, match="endian"):
        unpack_frame(_mutate_header(blob, endian="big"))
    with pytest.raises(NlfError, match="codec"):
        unpack_frame(_mutate_header(blob, codec="lzma"))


def test_rejects_dtype_order_and_shape_inconsistency() -> None:
    blob = _good_blob()
    (hlen,) = struct.unpack_from("<I", blob, 4)
    header = json.loads(blob[8 : 8 + hlen].decode("utf-8"))
    for field, value, msg in (
        ("dtype", "f16", "dtype"),
        ("dtype", "f2", "dtype"),  # PROTOCOLS §1.1：线格式**没有**半精度，两侧都必须拒绝
        ("order", "X", "order"),
        ("nbytes", 1, "nbytes"),
    ):
        h = json.loads(json.dumps(header))
        h["arrays"][0][field] = value
        hb = json.dumps(h, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        with pytest.raises(NlfError, match=msg):
            unpack_frame(NLF_MAGIC + struct.pack("<I", len(hb)) + hb + blob[8 + hlen :])


def test_rejects_out_of_range_and_overlapping_spans() -> None:
    blob = _good_blob()
    (hlen,) = struct.unpack_from("<I", blob, 4)
    header = json.loads(blob[8 : 8 + hlen].decode("utf-8"))
    payload = blob[8 + hlen :]

    h = json.loads(json.dumps(header))
    h["arrays"][0]["offset"] = len(payload) + 10
    hb = json.dumps(h, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with pytest.raises(NlfError, match="越界"):
        unpack_frame(NLF_MAGIC + struct.pack("<I", len(hb)) + hb + payload)

    h = json.loads(json.dumps(header))
    h["arrays"][1]["offset"] = h["arrays"][0]["offset"]  # 两个数组完全重叠
    hb = json.dumps(h, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with pytest.raises(NlfError, match="重叠"):
        unpack_frame(NLF_MAGIC + struct.pack("<I", len(hb)) + hb + payload)


def test_rejects_payload_length_mismatch() -> None:
    blob = _good_blob()
    with pytest.raises(NlfError, match="payload"):
        unpack_frame(blob + b"\x00\x00\x00")
    with pytest.raises(NlfError, match="payload"):
        unpack_frame(blob[:-1])


def test_rejects_corrupt_compressed_payload() -> None:
    fr = Frame(chapter="0.0", viz="z", codec="zlib")
    fr.add("a", np.arange(100.0))
    blob = bytearray(pack_frame(fr))
    blob[-5] ^= 0xFF
    with pytest.raises(NlfError, match="zlib"):
        unpack_frame(bytes(blob))
    fr2 = Frame(chapter="0.0", viz="g", codec="gzip")
    fr2.add("a", np.arange(100.0))
    blob2 = bytearray(pack_frame(fr2))
    blob2[-5] ^= 0xFF
    with pytest.raises(NlfError, match="gzip"):
        unpack_frame(bytes(blob2))


def test_rejects_codec_field_that_contradicts_payload() -> None:
    """codec 字段被改成 "none"、payload 却仍是压缩数据时，长度自洽校验必须抓住它。"""
    fr = Frame(chapter="0.0", viz="x", codec="zlib")
    fr.add("a", np.arange(1000.0))
    blob = pack_frame(fr)
    (hlen,) = struct.unpack_from("<I", blob, 4)
    header = json.loads(blob[8 : 8 + hlen].decode("utf-8"))
    header["codec"] = "none"  # 撒谎：说没压缩
    hb = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    lying = NLF_MAGIC + struct.pack("<I", len(hb)) + hb + blob[8 + hlen :]
    with pytest.raises(NlfError, match="payload"):
        unpack_frame(lying)
    # 反向撒谎：说压缩了，其实没压缩
    header["codec"] = "zlib"
    hb2 = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    plain = pack_frame(Frame(chapter="0.0", viz="y"))  # 空帧 payload 为空，zlib 解压必然失败
    lying2 = NLF_MAGIC + struct.pack("<I", len(hb2)) + hb2 + b"not-zlib-data"
    with pytest.raises(NlfError, match="zlib"):
        unpack_frame(lying2)
    assert plain[:4] == NLF_MAGIC


# --------------------------------------------------------------------------- #
# 构造期的自我约束
# --------------------------------------------------------------------------- #
def test_frame_rejects_duplicate_names_and_array_in_meta() -> None:
    fr = Frame()
    fr.add("x", np.zeros(3))
    with pytest.raises(NlfError, match="重复"):
        fr.add("x", np.ones(3))
    with pytest.raises(NlfError, match="数组"):
        fr.set_meta(bad=[1, 2, 3])
    with pytest.raises(NlfError, match="数组"):
        fr.set_meta(bad=np.zeros(2))


def test_frame_access_errors_are_explicit() -> None:
    fr = _basic_frame()
    with pytest.raises(NlfError, match="没有数组"):
        _ = fr["nope"]
    with pytest.raises(NlfError, match="没有数组"):
        fr.shape_of("nope")
    assert "u" in fr and "nope" not in fr
    assert fr.nbytes() == sum(a.nbytes for a in (fr["u"], fr["M"], fr["flags"], fr["idx"]))


def test_validate_rejects_bad_kind_and_codec() -> None:
    fr = Frame(kind="weird")
    with pytest.raises(NlfError, match="kind"):
        fr.validate()
    fr2 = Frame(codec="brotli")
    with pytest.raises(NlfError, match="codec"):
        fr2.validate()


def test_non_float_dtype_is_coerced_to_float64() -> None:
    """未登记的 dtype（如 float16）会被抬成 f8，而不是写出前端不认识的格式。"""
    fr = Frame()
    fr.add("h", np.arange(4, dtype=np.float16))
    assert fr.specs[0].dtype == "f8"
    fr.add("i", np.arange(4, dtype=np.int16), dtype="i4")
    assert fr.specs[1].dtype == "i4"


# --------------------------------------------------------------------------- #
# 与仓库夹具 / 标准答案的一致性（跨语言契约的 Python 一侧）
# --------------------------------------------------------------------------- #
def test_repo_fixtures_decode_to_golden_values() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    fixture_dir = root / "tests" / "fixtures" / "nlf"
    golden_dir = root / "tests" / "parity" / "golden"
    index = json.loads((fixture_dir / "index.json").read_text(encoding="utf-8"))
    assert index["fixtures"], "夹具索引为空"
    for item in index["fixtures"]:
        name = item["name"]
        fr = read_frame(fixture_dir / item["file"])
        golden = json.loads((golden_dir / f"nlf_{name}.json").read_text(encoding="utf-8"))
        assert fr.chapter == golden["chapter"] and fr.viz == golden["viz"]
        assert fr.kind == golden["kind"] and fr.codec == golden["codec"]
        assert fr.meta == golden["meta"]
        assert fr.strings == golden["strings"]
        assert fr.critical == golden["critical"]
        assert fr.pitfalls == golden["pitfalls"]
        for arr_name, want in golden["arrays"].items():
            got = fr[arr_name]
            assert list(got.shape) == want["shape"], (name, arr_name)
            got_values = [float(v) if got.dtype.kind == "f" else int(v) for v in got.ravel(order="C")]
            assert got_values == want["values"], (name, arr_name)


def test_fixture_generator_is_idempotent() -> None:
    """`tools/gen_fixtures.py --check` 必须报告"无过期"（CI 门禁 G6 的 Python 侧）。"""
    from tools.gen_fixtures import build

    assert build(check=True) == 0


# --------------------------------------------------------------------------- #
# meta 只允许 number/bool/null —— 这条是"被前端抓出来的"契约
# --------------------------------------------------------------------------- #
def test_set_meta_rejects_string_and_points_at_strings() -> None:
    fr = Frame(chapter="0.0", viz="v", kind="frame")
    fr.add("x", np.zeros(3))
    with pytest.raises(NlfError, match="set_strings"):
        fr.set_meta(regime="underdamped")
    fr.set_strings(regime="underdamped")
    assert fr.strings["regime"] == "underdamped"


def test_validate_catches_injected_string_meta() -> None:
    """绕过 set_meta 直接塞字符串，validate()/pack_frame() 也必须拦住。"""
    fr = Frame(chapter="0.0", viz="v", kind="frame")
    fr.add("x", np.zeros(3))
    fr.meta["regime"] = "underdamped"
    with pytest.raises(NlfError, match="strings"):
        fr.validate()
    with pytest.raises(NlfError, match="strings"):
        pack_frame(fr)


def test_meta_types_roundtrip() -> None:
    fr = Frame(chapter="0.0", viz="v", kind="frame")
    fr.add("x", np.zeros(3))
    fr.set_meta(n=3, ok=True, missing=None, r=0.5)
    back = unpack_frame(pack_frame(fr))
    assert back.meta == {"n": 3, "ok": True, "missing": None, "r": 0.5}
    back.validate()
