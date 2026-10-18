r"""4.2 章帧契约测试：四个可视化在**参数角点**上都必须能编出合法帧。

纪律与 4.1 章一致：探测用 ``Registry(root).discover(strict=False)``（不做全仓校验），
但帧本身用 ``discover()`` 的严格条目 + ``strict=True`` 构建，再走一遍 ``pack_frame``
（编码器会自己再查一遍非有限值与 meta 类型）。

重点不是"能跑通"，而是把三条**帧层**容易漏的口径钉住：

* 非有限值：``r_at_solution`` 在阶跃的平段上是 :math:`0/0`——必须有占位值 + 掩码
  （实测 t=0 时 128 个点里有 124 个未定义，早期版本直接让整帧编不出去）；
* ``requiredAux``：帧里必须真有名字含 ``error``/``residual`` 的数组，否则前端画不出
  误差面板（早期版本三项都缺 ``residual``）；
* 数值必须来自 numcore：本测试**不看**帧里的具体数字是否"好看"，只看结构与有限性，
  但会额外核一件事——``meta`` 里的系数必须与 :mod:`numcore.pde.advection` 的闭式一致
  （防止有人在帧里"顺手"另写一份数学）。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.pde import advection as ad  # noqa: E402
from numcore.pde import limiters as lm  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.nlf import pack_frame  # noqa: E402
from server.registry import Registry  # noqa: E402

CHAPTER = "4.2"
VIZ_IDS = ("scheme_zoo", "phase_error", "limiter_interaction", "packet_long_time")


def _entry():
    reg = Registry(ROOT)
    reg.discover(strict=False)
    return reg.get(CHAPTER)


def _raw():
    return _entry().module.MANIFEST


def _defaults(spec) -> dict:
    return {p["name"]: p["default"] for p in spec["params"]}


def _corners(spec) -> list[dict]:
    """默认值 + 每个参数的两端（枚举取首尾，数值取 min/max）。"""
    base = _defaults(spec)
    out = [dict(base)]
    for p in spec["params"]:
        if p["type"] == "enum":
            out.append({**base, p["name"]: p["options"][0]})
            out.append({**base, p["name"]: p["options"][-1]})
        elif p["type"] in ("int", "float") and p.get("min") is not None:
            out.append({**base, p["name"]: p["min"]})
            out.append({**base, p["name"]: p["max"]})
    seen = set()
    uniq = []
    for c in out:
        k = repr(sorted(c.items()))
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def _spec(viz_id: str):
    for spec in _raw()["vizzes"]:
        if spec["id"] == viz_id:
            return spec
    raise KeyError(viz_id)


# --------------------------------------------------------------------------- #
# 结构
# --------------------------------------------------------------------------- #
def test_manifest_shape() -> None:
    raw = _raw()
    assert raw["chapter"] == CHAPTER
    assert raw["part"] == "IV"
    assert {v["id"] for v in raw["vizzes"]} == set(VIZ_IDS)
    for spec in raw["vizzes"]:
        assert spec["dataSource"] == "backend"
        assert spec["theory"], f"{spec['id']} 缺 theory"
        # pitfalls 是**一个**字符串，且不短于 60 字符
        assert isinstance(spec["pitfalls"], str) and len(spec["pitfalls"]) >= 60
        assert spec["axes"], f"{spec['id']} 无 axes"
        for ax in spec["axes"]:
            assert "autoscaled" in ax, f"{spec['id']}/{ax['id']} 缺显式 autoscaled"
        aux = spec.get("requiredAux") or []
        assert aux, f"{spec['id']} requiredAux 为空"
        low = [a.lower() for a in aux]
        assert any(("error" in a) or ("residual" in a) for a in low), spec["id"]
        # 时间语义
        t = spec["time"]
        assert t["mode"] in ("none", "stream", "scrub")
        if t["mode"] == "none":
            assert "tMax" not in t and "dt" not in t and "fps" not in t
        else:
            assert t["tMax"] > 0 and t["dt"] > 0
        # critical ⟹ criticalNote
        for p in spec["params"]:
            assert p["type"] in ("int", "float", "bool", "enum")
            if p["type"] == "enum":
                assert len(p["options"]) >= 2
                assert p["default"] in p["options"]
            elif p["type"] in ("int", "float"):
                assert p["min"] < p["max"]
                assert p["min"] <= p["default"] <= p["max"]
                assert p.get("beyond") in (None, "", "above", "below", "magnitude")
            if p.get("critical"):
                assert p.get("criticalNote"), f"{spec['id']}/{p['name']} 有 critical 无 criticalNote"


@pytest.mark.parametrize("viz_id", VIZ_IDS)
def test_frames_are_finite_on_all_corners(viz_id: str) -> None:
    """默认 + 角点 × t∈{0, tMax}：数组全有限、meta 全是有限标量、能编码。"""
    entry = _entry()
    spec = _spec(viz_id)
    t_max = float((spec.get("time") or {}).get("tMax") or 0.0)
    times = [0.0, t_max] if t_max > 0 else [0.0]
    n_checked = 0
    for params in _corners(spec):
        for t in times:
            frame = build_frame(entry, viz_id, params, t, strict=True)
            blob = pack_frame(frame)
            assert isinstance(blob, (bytes, bytearray)) and len(blob) > 0
            for spec_arr, arr in frame.arrays.items() if hasattr(frame, "arrays") else []:
                assert np.all(np.isfinite(arr)), f"{viz_id}/{spec_arr} 非有限"
            # meta 里不许有 NaN/Inf（.nlf 头用 allow_nan=False）
            for k, v in frame.meta.items():
                if isinstance(v, float):
                    assert np.isfinite(v), f"{viz_id}: meta[{k}] = {v}"
            n_checked += 1
    assert n_checked >= 4


@pytest.mark.parametrize("viz_id", VIZ_IDS)
def test_frames_expose_required_aux(viz_id: str) -> None:
    """帧里必须真有 requiredAux 声明的数组名。"""
    entry = _entry()
    spec = _spec(viz_id)
    frame = build_frame(entry, viz_id, _defaults(spec), 0.0, strict=True)
    names = [n.lower() for n in frame.names]
    for aux in spec["requiredAux"]:
        assert any(aux.lower() in n for n in names), f"{viz_id}: 缺 {aux}（现有 {frame.names}）"


def test_no_nonfinite_arrays_anywhere() -> None:
    """四个可视化的默认帧里，每个数组都必须是有限值。"""
    entry = _entry()
    bad = []
    for viz_id in VIZ_IDS:
        spec = _spec(viz_id)
        frame = build_frame(entry, viz_id, _defaults(spec), 0.0, strict=True)
        for name in frame.names:
            arr = np.asarray(frame[name], dtype=float)
            if arr.size and not np.all(np.isfinite(arr)):
                bad.append(f"{viz_id}/{name}")
    assert not bad, bad


def test_limiter_interaction_masks_undefined_ratio() -> None:
    """阶跃平段上 r=0/0：必须有掩码，且占位值本身是有限数。"""
    entry = _entry()
    spec = _spec("limiter_interaction")
    frame = build_frame(entry, spec["id"], _defaults(spec), 0.0, strict=True)
    r = np.asarray(frame["r_at_solution"], dtype=float)
    valid = np.asarray(frame["r_valid"], dtype=float)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(valid))
    # t=0 时解就是阶跃，绝大多数点的 r 未定义（实测 124/128）
    assert float(np.mean(valid)) < 0.2
    assert int(frame.meta["n_r_undefined"]) >= 100


# --------------------------------------------------------------------------- #
# 帧里的数字必须来自 numcore（防止在帧里另写一份数学）
# --------------------------------------------------------------------------- #
def test_phase_error_frame_coefficients_match_numcore() -> None:
    entry = _entry()
    spec = _spec("phase_error")
    params = {**_defaults(spec), "scheme": "upwind", "nu": 0.4}
    frame = build_frame(entry, spec["id"], params, 0.0, strict=True)
    d2, a3, d4, a5 = ad.log_symbol_expansion("upwind", 0.4)
    assert abs(frame.meta["d2"] - d2) < 1e-15
    assert abs(frame.meta["a3"] - a3) < 1e-15
    assert abs(frame.meta["d4"] - d4) < 1e-15
    assert abs(frame.meta["a5"] - a5) < 1e-15
    # 帧里的相速度曲线必须与 numcore 的解析符号逐点一致
    phi = np.asarray(frame["phi"], dtype=float)
    got = np.asarray(frame["c_num"], dtype=float)
    want = np.asarray(ad.phase_speed("upwind", phi, 0.4), dtype=float)
    assert float(np.max(np.abs(got - want))) < 1e-14


def test_scheme_zoo_frame_reports_the_measured_step_statistics() -> None:
    """帧报的 umax/TV/过冲必须与直接调 numcore 算出来的一致。"""
    entry = _entry()
    spec = _spec("scheme_zoo")
    params = {**_defaults(spec), "scheme": "centered", "nu": 0.4, "nx": 128, "initial": "step"}
    frame = build_frame(entry, spec["id"], params, 0.0, strict=True)
    u0 = np.where(np.arange(128) / 128 < 0.25, 1.0, 0.0)
    res = ad.advection_solve(u0, nu=0.4, n_steps=0, scheme="centered", integrator="rk3")
    assert abs(frame.meta["umax"] - float(np.max(res.u))) < 1e-12
    assert abs(frame.meta["total_variation"] - lm.total_variation(u0)) < 1e-12
    # 解析符号给的 max|G| 与帧里报的一致
    assert abs(frame.meta["max_abs_symbol"] - ad.max_abs_symbol("centered", 0.4)) < 1e-15


def test_limiter_frame_reports_phi_inside_sweby_region() -> None:
    entry = _entry()
    spec = _spec("limiter_interaction")
    for limiter in ("minmod", "mc", "superbee", "vanleer", "vanalbada"):
        params = {**_defaults(spec), "limiter": limiter}
        frame = build_frame(entry, spec["id"], params, 0.0, strict=True)
        inside = np.asarray(frame["in_sweby"], dtype=float)
        assert float(np.mean(inside)) > 0.99, limiter
        assert 0.0 <= frame.meta["sweby_fraction"] <= 1.0


def test_time_zero_frame_is_not_a_stale_copy() -> None:
    """t=0 与 t=tMax 必须给出不同的帧（防止 lru_cache 键漏参）。"""
    entry = _entry()
    for viz_id in ("scheme_zoo", "limiter_interaction", "packet_long_time"):
        spec = _spec(viz_id)
        t_max = float(spec["time"]["tMax"])
        f0 = build_frame(entry, viz_id, _defaults(spec), 0.0, strict=True)
        f1 = build_frame(entry, viz_id, _defaults(spec), t_max, strict=True)
        u0 = np.asarray(f0["u"], dtype=float)
        u1 = np.asarray(f1["u"], dtype=float)
        assert float(np.max(np.abs(u1 - u0))) > 1e-6, viz_id
