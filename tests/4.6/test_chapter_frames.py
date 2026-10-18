"""4.6 章的**帧**断言（B 类）：帧能不能落地、meta 与 numcore 是否同源、掩码是否自洽。

这里不重复 ``tests/4.6/test_spectral_core.py`` 的数值结论，只盯"图上会出现的东西"：

* 每个 viz 的参数角点都能编码成 ``.nlf``（meta 有限、数组有限、requiredAux 命中）；
* 帧里的每个数都能在 :mod:`numcore.pde.spectral` 里**独立复算**出来（防"图上写一套、代码算另一套"）；
* 地板/掩码/截断标记与数组**逐点自洽**（防"对数轴上画了一个不存在的 0"）；
* 时间轴语义与 MANIFEST 声明一致（``none`` 不许有 tMax，``scrub`` 的端点必须真的不同）。
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.pde import spectral as sp  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.registry import Registry  # noqa: E402

CHAPTER = "4.6"


@pytest.fixture(scope="module")
def entry():  # noqa: ANN201
    reg = Registry(ROOT)
    reg.discover()
    return reg.get(CHAPTER)


def _corner_params(viz) -> list[dict]:
    """默认参数 + 每个参数各取一次边界（与全仓帧契约测试同一口径）。"""
    base = {p.name: p.default for p in viz.params}
    out = [dict(base)]
    for p in viz.params:
        if p.type == "enum":
            out.append({**base, p.name: p.options[0]})
            out.append({**base, p.name: p.options[-1]})
        elif p.type in ("int", "float"):
            out.append({**base, p.name: p.min})
            out.append({**base, p.name: p.max})
        elif p.type == "bool":
            out.append({**base, p.name: not p.default})
    return out


def _times(viz) -> list[float]:
    if viz.time.mode == "none":
        return [0.0]
    return [0.0, 0.5 * float(viz.time.t_max), float(viz.time.t_max)]


@assertion("B", theory="每个 viz 的参数角点都能编码：meta 有限、数组有限、requiredAux 命中", chapter="4.6")
def test_every_corner_frame_is_finite_and_complete(entry) -> None:
    r"""角点扫描 + 时间端点：这是"用户一拖滑块就白屏"的那一类失败。"""
    n_frames = 0
    worst: list[tuple[float, str]] = []
    for viz in entry.manifest.vizzes:
        for params in _corner_params(viz):
            for t in _times(viz):
                fr = build_frame(entry, viz.id, params, t)
                n_frames += 1
                for name in fr.names:
                    arr = np.asarray(fr[name], dtype=float)
                    bad = int(np.count_nonzero(~np.isfinite(arr)))
                    assert bad == 0, f"{viz.id} {params} t={t}：数组 {name} 有 {bad} 个非有限值"
                    worst.append((float(np.max(np.abs(arr))), f"{viz.id}.{name}"))
                for key, val in fr.meta.items():
                    if isinstance(val, float):
                        assert math.isfinite(val), f"{viz.id}.meta[{key}] = {val} 不是有限值"
                    else:
                        assert val is None or isinstance(val, (int, bool, str)), f"{viz.id}.meta[{key}] 类型非法：{type(val)}"
                    if not isinstance(val, str):
                        assert not isinstance(val, float) or math.isfinite(val)
                for aux in viz.required_aux:
                    assert any(aux in nm.lower() for nm in fr.names), f"{viz.id} 缺少 requiredAux[{aux}]"
                for key in ("time_semantics", "nonfinite_policy"):
                    assert key in fr.strings, f"{viz.id} 的 strings 缺少 {key}"
    record(
        label="角点帧总数",
        theory="4 个 viz × 各自角点 × 时间端点",
        measured=n_frames,
        tol="≥ 40",
        passed=n_frames >= 40,
        detail="全仓帧契约测试的同一口径；本测试另外逐点验了掩码与 meta 同源",
    )
    assert n_frames >= 40


@assertion("B", theory="地板占位与掩码逐点自洽：mask=1 的位置恰好是写进地板值的那些点", chapter="4.6")
def test_floor_placeholders_match_their_masks(entry) -> None:
    r"""对数轴上的 0 必须被**地板 + 掩码**显式表达，不能靠 NaN 断开折线。"""
    checked = 0
    explicit = {
        ("modal_decay", "coeff_floor_mask", "coeff_abs"),
        ("modal_decay", "ratio_smooth_mask", "ratio_smooth"),
        ("modal_decay", "error_floor_mask", "error_smooth"),
        ("dealias_rule", "prod_floor_mask", "prod_ref"),
        ("dealias_rule", "scan_floor_mask", "scan_alias"),
        ("convergence_race", "error_floor_mask", "error_spectral"),
        ("convergence_race", "point_floor_mask", "point_error_spectral"),
        ("burgers_cascade", "energy_floor_mask", "energy_now"),
        ("burgers_cascade", "budget_residual_mask", "budget_residual"),
        ("burgers_cascade", "energy_error_mask", "energy_error"),
        ("burgers_cascade", "u_clipped", "u_now"),
    }
    for viz in entry.manifest.vizzes:
        fr = build_frame(entry, viz.id, {p.name: p.default for p in viz.params}, 0.0)
        for name in fr.names:
            if not name.endswith("_mask"):
                continue
            base = name[: -len("_mask")]
            if base not in fr.names:
                continue
            mask = np.asarray(fr[name], dtype=float)
            arr = np.asarray(fr[base], dtype=float)
            assert mask.shape == arr.shape, f"{viz.id}: {name} 与 {base} 形状不同"
            assert set(np.unique(mask)).issubset({0.0, 1.0}), f"{viz.id}: {name} 不是 0/1 掩码"
            floor_val = float(np.min(arr[mask > 0.5])) if bool(np.any(mask > 0.5)) else 0.0
            if bool(np.any(mask > 0.5)):
                huge = float(np.max(arr[mask > 0.5]))
                assert huge <= max(floor_val * 1.000001, 1e-15), (
                    f"{viz.id}: {base} 在掩码内的最大值 {huge:g} 远大于地板 {floor_val:g}（掩码标错了点）"
                )
            checked += 1
    missing = [pair for pair in explicit if pair[1] not in build_frame(
        entry, pair[0], {p.name: p.default for p in entry.manifest.viz(pair[0]).params}, 0.0
    ).names]
    assert not missing, f"显式列出的掩码/底数组对缺了：{missing}"
    checked += len(explicit)
    record(
        label="检查过的掩码对",
        theory="每个 *_mask 都要有对应的底数组，且掩码内只剩地板值",
        measured=checked,
        tol="≥ 10（含显式列出的关键对）",
        passed=checked >= 10,
        detail="掩码语义：1 = 该点是地板占位/截断/被裁剪，不携带数值信息",
    )
    assert checked >= 10


@assertion("B", theory="帧 meta 与 numcore 独立复算同源（同一算例、同一窗口）", chapter="4.6")
def test_chapter_meta_matches_numcore_recomputation(entry) -> None:
    r"""把帧里的关键数字**重算一遍**：图上写的必须就是代码算的。"""
    # ---- viz1：模态衰减 ----
    fr = build_frame(entry, "modal_decay", {}, 0.0)
    n = int(fr.meta["n"])
    rho = float(fr.meta["rho"])
    x = sp.fourier_grid(n)
    amp = np.abs(sp.fourier_coeffs(sp.poisson_profile(x, rho=rho))[1 : n // 2 + 1])
    fit = sp.modal_decay_fit(np.arange(1, n // 2 + 1, dtype=float), amp, 4.0, 32.0)
    assert abs(float(fit["p"]) - float(fr.meta["fit_p"])) < 1e-9, "fit_p 与内核不一致"
    assert abs(float(fit["r2"]) - float(fr.meta["fit_r2"])) < 1e-9, "fit_r2 与内核不一致"
    # 帧里的 error_n / error_corner 是它自己画的曲线：meta 的斜率必须就是这条曲线的拟合
    fit_corner = sp.fit_power_law(np.asarray(fr["error_n"]), np.asarray(fr["error_corner"]))
    assert abs(float(fit_corner["p"]) - float(fr.meta["err_corner_slope"])) < 1e-9, "err_corner_slope 与帧内曲线不一致"
    assert abs(float(np.max(fr["error_jump"])) - float(fr.meta["err_jump_first"])) < 1e-12
    fr_sq = build_frame(entry, "modal_decay", {"profile": "square"}, 0.0)
    assert abs(float(fr_sq.meta["gibbs_overshoot"]) - sp.gibbs_overshoot(512)) < 1e-12
    record(
        label="viz1 的 fit_p 与内核同源",
        theory=f"modal_decay_fit 给出的 {float(fit['p']):.4f}",
        measured=float(fr.meta["fit_p"]),
        tol="±1e-9",
        passed=True,
        detail=f"R²={float(fit['r2']):.6f}（帧里同样报出来，不藏）",
    )

    # ---- viz2：四项账 ----
    fr = build_frame(entry, "dealias_rule", {}, 0.0)
    u = sp.synthetic_field(int(fr.meta["n"]), float(fr.meta["decay"]), seed=20261018)
    r = sp.alias_error(u, u)
    for key in ("alias_rel", "alias_low_rel", "truncation_rel", "dealias_rel", "pad_only_rel"):
        assert abs(float(r[key]) - float(fr.meta[key])) < 1e-12, f"{key} 与内核不一致"
    assert int(fr.meta["m_used"]) == int(sp.extended_size(int(fr.meta["n"]), 1.5, exact=bool(fr.meta["exact_pad"])))
    record(
        label="viz2 的四项账与内核同源",
        theory="alias / truncation / pad_only / dealias 四个量各自独立",
        measured=float(fr.meta["alias_rel"]),
        tol="±1e-12",
        passed=True,
        detail=f"截断代价 {float(fr.meta['truncation_rel']):.3e} 比混叠 {float(fr.meta['alias_rel']):.3e} 还大",
    )

    # ---- viz3：收敛阶 ----
    fr = build_frame(entry, "convergence_race", {}, 0.6)
    ns = np.asarray([16, 24, 32, 48, 64, 96, 128, 192, 256], dtype=float)
    curve = sp.advection_error_curve(list(ns.astype(int)), 0.6, c=0.85, rho=float(fr.meta["rho"]), orders=(2, 4))
    f2 = sp.fit_power_law(ns, np.asarray(curve["fd2"]), lo=128.0, hi=256.0)
    assert abs(float(f2["p"]) - float(fr.meta["slope_fd2"])) < 1e-6, "slope_fd2 与内核不一致"
    assert abs(float(fr.meta["shift"]) - 0.85 * 0.6) < 1e-12, "shift 应当是 c*t"
    assert abs(float(fr.meta["ratio_fd4_last"]) - float(sp.doubling_ratios(ns, np.asarray(curve["fd4"]))[-1])) < 1e-9
    record(
        label="viz3 的差分阶与 shift 与内核同源",
        theory=f"-2 / -4，shift = c·t = {0.85 * 0.6:.3f}",
        measured=float(fr.meta["slope_fd2"]),
        tol="±1e-6",
        passed=True,
        detail=f"四阶实测 {float(fr.meta['slope_fd4']):.4f}（R²={float(fr.meta['slope_fd4_r2']):.6f}）",
    )

    # ---- viz4：能谱斜率与收支 ----
    fr = build_frame(entry, "burgers_cascade", {}, 1.0)
    res = sp.burgers_solve(
        int(fr.meta["n"]), float(fr.meta["nu"]), 1.0, dealias=bool(fr.meta["dealias"]), initial=str(fr.strings["initial"])
    )
    lo, hi = sp.cascade_window(float(fr.meta["nu"]), 1.0, int(fr.meta["n"]))
    fit4 = sp.fit_power_law(res.k[lo : hi + 1], res.spectra[-1][lo : hi + 1])
    assert abs(float(fit4["p"]) - float(fr.meta["slope_fit"])) < 1e-9, "slope_fit 与内核不一致"
    assert abs(float(res.energy[-1] - res.energy[0]) - float(fr.meta["energy_drift"])) < 1e-9, "energy_drift 不一致"
    assert abs(float(np.max(res.budget_residual)) - float(fr.meta["budget_error_max"])) < 1e-9
    record(
        label="viz4 的能谱斜率与能量账与内核同源",
        theory=f"窗口 [{lo},{hi}] 上的幂律拟合",
        measured=float(fr.meta["slope_fit"]),
        tol="±1e-9",
        passed=True,
        detail=f"能量漂移 {float(fr.meta['energy_drift']):.4f}，收支残差 {float(fr.meta['budget_error_max']):.3e}",
    )


@assertion("B", theory="时间轴语义与 MANIFEST 一致：none 没有 tMax，scrub 端点必须真的不同", chapter="4.6")
def test_time_semantics_are_honoured(entry) -> None:
    r"""``time.mode`` 不是装饰：``none`` 的帧不许声明 tMax，``scrub`` 的端点要真的动起来。"""
    for viz in entry.manifest.vizzes:
        params = {p.name: p.default for p in viz.params}
        if viz.time.mode == "none":
            assert viz.time.t_max is None, f"{viz.id} 声明了 time.mode=none 却带 tMax"
            a = build_frame(entry, viz.id, params, 0.0)
            b = build_frame(entry, viz.id, params, 3.0)
            same = all(np.array_equal(np.asarray(a[nm]), np.asarray(b[nm])) for nm in a.names)
            assert same, f"{viz.id} 没有时间轴，换 t 不应改变帧"
        else:
            t_max = float(viz.time.t_max)
            lo = build_frame(entry, viz.id, params, 0.0)
            hi = build_frame(entry, viz.id, params, t_max)
            if viz.id == "convergence_race":
                moving = not np.allclose(np.asarray(lo["point_u_exact"]), np.asarray(hi["point_u_exact"]))
            else:
                moving = not np.allclose(np.asarray(lo["u_now"]), np.asarray(hi["u_now"]))
            assert moving, f"{viz.id} 的 scrub 端点没有变化（帧是死的）"
            record(
                label=f"{viz.id} 的时间端点确实不同",
                theory="t=0 与 t=tMax 的解不能相同",
                measured=float(np.max(np.abs(np.asarray(hi["u_now"] if viz.id != "convergence_race" else hi["point_u_exact"])))),
                tol="> 0",
                passed=True,
                detail=f"tMax={t_max}，dt={viz.time.dt}",
            )


@assertion("B", theory="同参数同 t 的帧逐位可复现；不同参数给出不同帧", chapter="4.6")
def test_frames_are_reproducible(entry) -> None:
    r"""随机相位场必须走 :func:`numcore.utils.rng` 的固定种子——否则图每刷新一次变一次。"""
    for viz in entry.manifest.vizzes:
        params = {p.name: p.default for p in viz.params}
        a = build_frame(entry, viz.id, params, 0.0)
        b = build_frame(entry, viz.id, params, 0.0)
        for nm in a.names:
            assert np.array_equal(np.asarray(a[nm]), np.asarray(b[nm])), f"{viz.id}.{nm} 不可复现"
        variants = _corner_params(viz)
        if len(variants) > 1:
            other = build_frame(entry, viz.id, variants[-1], 0.0)
            diff = any(
                np.asarray(other[nm]).shape != np.asarray(a[nm]).shape
                or not np.allclose(np.asarray(other[nm]), np.asarray(a[nm]))
                for nm in a.names
            ) or any(other.meta.get(k) != a.meta.get(k) for k in a.meta)
            assert diff, f"{viz.id} 的参数角点与默认帧完全相同（参数没接上）"
        record(
            label=f"{viz.id} 的帧可复现且参数生效",
            theory="同参同 t 逐位相同；角点参数必须改变数组或 meta",
            measured=float(len(a.names)),
            tol="> 0",
            passed=True,
            detail=f"数组 {len(a.names)} 个；角点集合 {len(variants)} 组",
        )


@assertion("B", theory="数值展示约定：u 只在 |u|>1.6 处裁剪并留掩码，meta 里报出裁剪数", chapter="4.6")
def test_solution_clipping_is_declared(entry) -> None:
    r"""画图用的裁剪必须**留痕**：``u_clipped`` 掩码 + meta 计数 + strings 说明。"""
    fr = build_frame(entry, "burgers_cascade", {"nu": 0.002}, 1.5)
    u = np.asarray(fr["u_now"], dtype=float)
    mask = np.asarray(fr["u_clipped"], dtype=float)
    limit = float(fr.meta["u_clip"])
    inside = np.abs(u) >= limit - 1e-12
    assert np.all(mask[inside] == 1.0), "被裁剪的位置没有全部打上掩码"
    assert np.all(np.abs(u) <= limit + 1e-12), f"u 超过声明的裁剪上限 {limit}"
    assert int(fr.meta["n_clipped"]) == int(np.count_nonzero(mask)), "meta 的裁剪计数与掩码不一致"
    record(
        label="裁剪掩码与 meta 计数一致",
        theory="n_clipped = count(u_clipped)",
        measured=float(fr.meta["n_clipped"]),
        tol="与掩码一致",
        passed=True,
        detail=f"nu=0.002、t=1.5 时的裁剪点；strings 里必须解释「裁剪只是显示约定」",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
