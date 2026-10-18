"""生成跨语言基准夹具：``tests/fixtures/nlf/*.nlf`` 与 ``tests/parity/golden/nlf_*.json``。

为什么要生成物入库：前端（TypeScript）必须能在**没有 Python 的情况下**证明自己解码正确。
夹具是这条证明链的锚点：

1. ``tools/gen_fixtures.py`` 用 Python 写出 ``.nlf`` 与"标准答案"JSON；
2. ``tests/test_nlf.py`` 校验 Python 解码结果与标准答案一致；
3. ``viz/tests/nlf.test.ts`` 校验 TS 解码结果与**同一份**标准答案一致；
4. ``tools/parity.py``（G7）把 2、3 的结论合并成一条"双实现不漂移"的门禁。

用法::

    python tools/gen_fixtures.py           # 重新生成
    python tools/gen_fixtures.py --check   # 只校验仓库中的生成物是否最新（CI 门禁）
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from server.nlf import Frame, pack_frame, unpack_frame  # noqa: E402

FIXTURE_DIR = _ROOT / "tests" / "fixtures" / "nlf"
GOLDEN_DIR = _ROOT / "tests" / "parity" / "golden"


def _basic_f8_1d() -> Frame:
    fr = Frame(chapter="0.0", viz="fake_curve", kind="frame")
    fr.add("u", np.linspace(0.0, 1.0, 17), unit="1", label="u(x)")
    fr.add("x", np.linspace(-1.0, 1.0, 17), unit="1", label="x")
    fr.set_meta(t=0.25, n=17, residual=1.5e-13, overflow=False)
    fr.set_strings(status="ok", source="backend")
    fr.critical = {"r": 0.5}
    fr.pitfalls = "固定纵轴时它看起来在收敛；把纵轴自适应打开就会看到振荡其实在放大。"
    return fr


def _multi_2d() -> Frame:
    fr = Frame(chapter="0.0", viz="fake_heat", kind="frame")
    A = np.arange(12, dtype=float).reshape(3, 4) / 7.0
    fr.add("A", A, unit="1", label="矩阵 A")
    fr.add("b", np.array([1.0, -2.0, 3.5]), unit="1", label="右端项")
    fr.set_meta(t=0.0, rho=1.25, growth=3.0)
    fr.critical = {"rho": 128.0}
    fr.pitfalls = "热力图默认线性色标会把 1e-16 与 1e-3 画成同一片黑，必须同时给对数色标与误差副轴。"
    return fr


def _zlib_2d() -> Frame:
    fr = Frame(chapter="0.0", viz="fake_field", kind="frame", codec="zlib")
    n = 48
    x = np.linspace(0.0, 2.0 * np.pi, n)
    y = np.linspace(0.0, np.pi, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    fr.add("field", np.sin(X) * np.cos(Y), unit="1", label="sin(x)cos(y)")
    fr.set_meta(t=0.0, nx=n, ny=n)
    fr.pitfalls = "压缩只影响传输体积，不影响数值；但要注意 codec 字段必须与 payload 一致，否则解码出垃圾。"
    return fr


def _f_order() -> Frame:
    fr = Frame(chapter="0.0", viz="fake_fortran", kind="frame")
    A = (np.arange(12, dtype=float).reshape(3, 4, order="F")).copy(order="F")
    fr.add("AF", A, order="F", label="Fortran 序矩阵")
    fr.add("AC", A, order="C", label="同一矩阵的 C 序副本")
    fr.set_meta(t=0.0)
    fr.pitfalls = "列主序与行主序写盘后字节完全不同，解码端若忽略 order 字段就会得到转置矩阵。"
    return fr


def _dtypes() -> Frame:
    fr = Frame(chapter="0.0", viz="fake_dtypes", kind="frame")
    fr.add("i", np.array([-3, -1, 0, 2, 7], dtype=np.int32))
    fr.add("u", np.array([0, 1, 127, 255], dtype=np.uint8))
    fr.add("flag", np.array([True, False, True, True], dtype=bool))
    fr.set_meta(t=0.0, n=5)
    fr.pitfalls = "整数与布尔数组若被前端当成 float64 解读，会静默给出完全错误的数值。"
    return fr


def _dataset_1d() -> Frame:
    """``kind="dataset"``：对拍基准数据（非实时帧）。"""
    fr = Frame(chapter="0.0", viz="parity_basis", kind="dataset")
    k = np.arange(8, dtype=float)
    fr.add("h", 2.0**-k, label="步长")
    fr.add("err", 0.37 * (2.0**-k) ** 2, label="误差")
    fr.set_meta(n=8, slope=2.0)
    fr.set_strings(kind_note="对拍用基准")
    fr.pitfalls = "基准数据的斜率是解析构造的 2.0；若拟合器给出别的值，是拟合器错了，不是数据错了。"
    return fr


BUILDERS = {
    "basic_f8_1d": _basic_f8_1d,
    "multi_2d": _multi_2d,
    "zlib_2d": _zlib_2d,
    "f_order": _f_order,
    "dtypes": _dtypes,
    "dataset_1d": _dataset_1d,
}


def _canonical(fr: Frame) -> dict:
    """标准答案（跨语言比对用）：数组按 ``repr`` 精度 17 位十进制输出。"""
    arrays = {}
    for name in fr.names:
        arr = fr[name]
        arrays[name] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "values": [float(v) if arr.dtype.kind == "f" else int(v) for v in arr.ravel(order="C")],
        }
    return {
        "chapter": fr.chapter,
        "viz": fr.viz,
        "kind": fr.kind,
        "codec": fr.codec,
        "meta": fr.meta,
        "strings": fr.strings,
        "critical": fr.critical,
        "pitfalls": fr.pitfalls,
        "arrays": arrays,
    }


def build(*, check: bool) -> int:
    """生成（或校验）全部夹具；返回不一致的文件数。"""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    index: list[dict[str, str]] = []
    for name, builder in BUILDERS.items():
        fr = builder()
        # 夹具必须**逐字节可复现**：generator 里不写时间戳与解释器版本
        fr.generator = {"tool": "tools/gen_fixtures.py", "format": "0.1.0"}
        blob = pack_frame(fr)
        # 自检：写出去的东西必须能被自己读回来
        back = unpack_frame(blob)
        assert back.names == fr.names, name
        nlf_path = FIXTURE_DIR / f"{name}.nlf"
        golden_path = GOLDEN_DIR / f"nlf_{name}.json"
        golden = json.dumps(_canonical(fr), ensure_ascii=False, indent=1, sort_keys=True) + "\n"
        if check:
            if not nlf_path.exists() or nlf_path.read_bytes() != blob:
                stale.append(str(nlf_path.relative_to(_ROOT)))
            if not golden_path.exists() or golden_path.read_text(encoding="utf-8") != golden:
                stale.append(str(golden_path.relative_to(_ROOT)))
        else:
            nlf_path.write_bytes(blob)
            golden_path.write_text(golden, encoding="utf-8")
        index.append({"name": name, "file": f"{name}.nlf", "bytes": str(len(blob)), "arrays": ",".join(fr.names)})
    index_path = FIXTURE_DIR / "index.json"
    index_json = json.dumps({"fixtures": index}, ensure_ascii=False, indent=1) + "\n"
    if check:
        if not index_path.exists() or index_path.read_text(encoding="utf-8") != index_json:
            stale.append(str(index_path.relative_to(_ROOT)))
        if stale:
            print("以下生成物已过期（请运行 `python tools/gen_fixtures.py`）：")
            for s in stale:
                print(f"  · {s}")
            return len(stale)
        print(f"夹具齐备：{len(BUILDERS)} 个 .nlf + {len(BUILDERS)} 份标准答案（{FIXTURE_DIR.relative_to(_ROOT)}）")
        return 0
    index_path.write_text(index_json, encoding="utf-8")
    print(f"已生成 {len(BUILDERS)} 个夹具到 {FIXTURE_DIR.relative_to(_ROOT)}")
    for item in index:
        print(f"  · {item['file']:20s} {item['bytes']:>6s} B  arrays={item['arrays']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成/校验 .nlf 跨语言基准夹具")
    ap.add_argument("--check", action="store_true", help="只校验，不写入（CI 门禁）")
    args = ap.parse_args(argv)
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
