"""2.1 章：插值与多项式逼近。

四个可视化的分工是"从节点分布、到常数爆炸、再到误差的通道"：

1. ``runge_nodes`` —— 同一个函数、同样多的节点，只换**节点分布**：等距插值在
   :math:`n=7` 触底（实测最大误差 0.247）之后一路发散到 :math:`n=30` 的 2388，
   Chebyshev 节点同期是 0.132 -> 0.0177 -> 0.00243。时间轴是"已用节点数 k"，
   画的是**整条误差曲线**而不是最大误差那一个点。
2. ``lebesgue_growth`` —— 发散的原因不是函数，是 :math:`\\Lambda_n`：等距节点
   :math:`\\Lambda_{30}=6.60\\times10^6` 并继续以因子 2 增长，Chebyshev 节点
   :math:`\\Lambda_{30}=3.127` 只按 :math:`\\frac2\\pi\\ln n` 爬。log 轴上同时画
   **实测 log-log 斜率**与**理论斜率**（规则 2）。
3. ``error_curve`` —— 时间轴是节点分布的**连续变形** :math:`t\\in[0,1]`
   （等距 -> Chebyshev），看 :math:`\\Lambda(t)` 从 89.3 降到 2.54（整体下降，
   但**不是严格单调**：n=12 时实测最大局部回升 0.137），
   同时把误差曲线、严格误差包络、首一乘积 :math:`|\\prod(x-x_i)|` 三条线并排。
4. ``barycentric_vs_vandermonde`` —— **三条误差通道**并排：重心公式的函数值误差、
   Vandermonde 直接解的函数值误差、以及 Vandermonde 解的**系数**误差
   （真值取有理数精确解）。教训是"κ(V) 涨到 1e14 不等于公式不稳定"。

约定（与 0.0/1.2 章一致）：``meta`` 只放数字/布尔/``null``，字符串走 ``strings``；
对数轴上的曲线一律先夹到地板再进帧，地板计数写进 ``meta``；
``meta.overflow`` 报告**画出来的每一条曲线**是否越过对数轴上界。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from numcore.interp.poly import (
    NODE_KINDS,
    ROUNDOFF_FLOOR,
    UNIT_ROUNDOFF,
    barycentric_interpolate,
    clamp_for_log,
    error_envelope,
    lebesgue_asymptotic_chebyshev,
    lebesgue_asymptotic_equispaced,
    lebesgue_constant,
    lebesgue_growth_table,
    max_abs_error,
    monic_product,
    morphed_nodes,
    node_family,
    pointwise_error,
    poly_eval,
    runge,
    runge_error_table,
    sup_derivative_over_factorial,
    test_function_values,
    interpolation_method_table,
)
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 对数轴上界：插值误差在 n 大、a 大时会冲到 1e3 以上（实测 Runge n=30 是 2.4e3），
#: 统一夹到 1e12，既不丢趋势也不让一条曲线压平其余曲线。
_LOG_CEIL = 1e12

#: 评估网格点数（主图与误差曲线共用）
_N_GRID = 801

#: 时间轴上限：runge_nodes / barycentric_vs_vandermonde 的 n 上限
_MAX_N = 40
_MAX_N_VDM = 36

#: 数值参数关键值（都有实测依据，见各 criticalNote）
_CRIT_N_EQ = 12.0
_CRIT_N_MAX = 30.0
_CRIT_LAMBDA = 50.0
_CRIT_VDM = 30.0

_FAMILY_ZH = {"equispaced": "等距节点", "chebyshev": "Chebyshev 节点"}
_FUNC_ZH = {"runge": "Runge 函数 1/(1+a x^2)", "exp": "exp(x)"}


def _count_exact_zero(values: np.ndarray) -> int:
    """恰好等于 0 的元素个数（对数轴上的"真 0"要单独报数）。"""
    return int(np.count_nonzero(np.asarray(values, dtype=float) == 0.0))


def _clamped(values: Any, *, floor: float = ROUNDOFF_FLOOR, ceiling: float = _LOG_CEIL) -> tuple[np.ndarray, dict[str, float]]:
    """夹到 ``[floor, ceiling]`` 并返回计数信息（对数轴不能画 0 或 NaN）。"""
    arr, info = clamp_for_log(values, floor, ceiling)
    return np.asarray(arr, dtype=float), info


def _clamped_log10(values: Any, *, lo: float, hi: float) -> tuple[np.ndarray, dict[str, float]]:
    """夹取**已经取过 log10** 的数组（:func:`clamp_for_log` 只管正值域，不能用在这里）。

    对数轴的上下界本来就是负数（比如 -18 表示 1e-18），把它喂给
    :func:`clamp_for_log` 会被当成"负的地板"直接报错——这个错在等距/Chebyshev
    两种参数下都会犯，所以单独写一个 helper。``info`` 的键名与前者保持一致。
    """
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    n_nonfinite = int(arr.size - np.count_nonzero(finite))
    filled = np.where(finite, arr, float(hi))
    below = filled < float(lo)
    above = filled > float(hi)
    out = np.clip(filled, float(lo), float(hi))
    return out, {
        "floor": float(lo),
        "ceiling": float(hi),
        "n_floor": int(np.count_nonzero(below)),
        "n_ceiling": int(np.count_nonzero(above)) + n_nonfinite,
        "n_nonfinite": n_nonfinite,
    }


def _finite_or_none(value: Any) -> float | None:
    """非有限值一律写 ``None``：JSON 没有 NaN/Infinity，硬塞会让整帧编码失败。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# --------------------------------------------------------------------------- #
# 1) runge_nodes
# --------------------------------------------------------------------------- #
def _runge_nodes(params: Mapping[str, Any], t: float) -> Frame:
    family = str(params.get("distribution", NODE_KINDS[0]))
    other = "chebyshev" if family == "equispaced" else "equispaced"
    n = int(round(float(params.get("n", 12))))
    n = max(1, min(_MAX_N, n))
    a = float(params.get("a", 25.0))
    k = max(1, min(n, int(round(float(t)))))
    xs = np.linspace(-1.0, 1.0, _N_GRID)
    f = runge(xs, a)

    nd_sel = node_family(family, k)
    nd_oth = node_family(other, k)
    p_sel = barycentric_interpolate(nd_sel, runge(nd_sel, a), xs)
    p_oth = barycentric_interpolate(nd_oth, runge(nd_oth, a), xs)
    e_sel = pointwise_error(f, p_sel)
    e_oth = pointwise_error(f, p_oth)
    sup_sel = sup_derivative_over_factorial("runge", k + 1, a=a, n_grid=1201)
    env_sel = error_envelope(nd_sel, xs, sup_sel)

    # 误差随 n 的两条曲线（同一函数、同一网格，只有节点分布不同）
    tab = runge_error_table(_MAX_N, a=a, n_grid=_N_GRID)
    ns = tab["n"]
    eq_n, _ = _clamped(tab["error_equispaced"])
    ch_n, _ = _clamped(tab["error_chebyshev"])
    e_sel_c, info_e = _clamped(e_sel)
    e_oth_c, _ = _clamped(e_oth)
    env_c, _ = _clamped(env_sel)

    lam_sel = lebesgue_constant(nd_sel, -1.0, 1.0, n_grid=2001, refine_rounds=3)
    lam_oth = lebesgue_constant(nd_oth, -1.0, 1.0, n_grid=2001, refine_rounds=3)
    lam_sel_n = lebesgue_constant(node_family(family, n), -1.0, 1.0, n_grid=2001, refine_rounds=3)
    lam_oth_n = lebesgue_constant(node_family(other, n), -1.0, 1.0, n_grid=2001, refine_rounds=3)

    fr = Frame(chapter="2.1", viz="runge_nodes", kind="frame")
    fr.add("x_grid", xs, unit="1", label="x")
    fr.add("f_values", f, unit="1", label="f(x)")
    fr.add("p_selected", p_sel, unit="1", label=f"{_FAMILY_ZH[family]}插值多项式（k 个节点）")
    fr.add("p_other", p_oth, unit="1", label=f"{_FAMILY_ZH[other]}插值多项式（同样 k 个节点）")
    fr.add("error_selected", e_sel_c, unit="1", label="|f-p|（当前分布，已夹地板）")
    fr.add("error_other", e_oth_c, unit="1", label="|f-p|（另一种分布，已夹地板）")
    fr.add("error_selected_raw", e_sel, unit="1", label="|f-p|（真值，节点上恰为 0）")
    fr.add("envelope_selected", env_c, unit="1", label="严格误差包络 sup|f^(k+1)|/(k+1)!·|∏(x-x_i)|")
    fr.add("nodes_selected_x", nd_sel, unit="1", label="当前分布的节点")
    fr.add("nodes_selected_y", runge(nd_sel, a), unit="1", label="节点上的函数值")
    fr.add("nodes_other_x", nd_oth, unit="1", label="另一种分布的节点")
    fr.add("nodes_other_y", runge(nd_oth, a), unit="1", label="节点上的函数值")
    fr.add("n_axis", ns, unit="1", label="节点数 n")
    fr.add("max_error_vs_n_selected", eq_n if family == "equispaced" else ch_n, unit="1", label="最大误差随 n（当前分布）")
    fr.add("max_error_vs_n_other", ch_n if family == "equispaced" else eq_n, unit="1", label="最大误差随 n（另一种分布）")
    fr.add("mark_k", np.array([float(k)], dtype=float), unit="1", label="当前位置 k")
    fr.add(
        "mark_error",
        np.array([float(np.max(e_sel))], dtype=float),
        unit="1",
        label="当前最大误差",
    )
    e_eq_k = float(np.max(np.abs(f - barycentric_interpolate(node_family("equispaced", k), runge(node_family("equispaced", k), a), xs))))
    e_ch_k = float(np.max(np.abs(f - barycentric_interpolate(node_family("chebyshev", k), runge(node_family("chebyshev", k), a), xs))))
    fr.set_meta(
        n=n,
        k=k,
        a=a,
        is_chebyshev=bool(family == "chebyshev"),
        n_nodes=k,
        max_error_selected=float(np.max(e_sel)),
        max_error_other=float(np.max(e_oth)),
        max_error_equispaced_k=e_eq_k,
        max_error_chebyshev_k=e_ch_k,
        lambda_selected_k=lam_sel,
        lambda_other_k=lam_oth,
        lambda_selected_n=lam_sel_n,
        lambda_other_n=lam_oth_n,
        lambda_ratio_n=lam_oth_n / lam_sel_n if lam_sel_n > 0.0 else None,
        envelope_max=float(np.max(env_sel)),
        envelope_looseness=_finite_or_none(float(np.max(env_sel)) / float(np.max(e_sel))) if np.max(e_sel) > 0 else None,
        sup_derivative_over_factorial=sup_sel,
        error_floor=ROUNDOFF_FLOOR,
        n_error_at_floor=info_e["n_floor"],
        n_error_nonfinite=info_e["n_nonfinite"],
        n_exact_zero_error=_count_exact_zero(e_sel),
        n_exact_zero_error_raw=_count_exact_zero(e_sel),
        max_error_equispaced_n=float(tab["error_equispaced"][_MAX_N - 1]),
        max_error_chebyshev_n=float(tab["error_chebyshev"][_MAX_N - 1]),
        divergence_ratio=float(tab["error_equispaced"][_MAX_N - 1]) / float(tab["error_chebyshev"][_MAX_N - 1]),
        # 规则 1/4：越界必须报。主图四条曲线里 p 值最大可达 1e3 量级，
        # 超过对数轴上界的只有被夹掉的那些点（计数在 n_error_at_floor 里）。
        overflow=bool(np.max(e_sel) > _LOG_CEIL or np.max(e_oth) > _LOG_CEIL),
    )
    fr.set_strings(
        family=_FAMILY_ZH[family],
        other_family=_FAMILY_ZH[other],
        function=_FUNC_ZH["runge"],
        note=(
            f"k={k} 个{_FAMILY_ZH[family]}：最大误差 {np.max(e_sel):.3e}，"
            f"另一分布 {np.max(e_oth):.3e}；Λ_{k}={lam_sel:.4g} vs {lam_oth:.4g}。"
            f"等距误差在 n=7 触底 0.247；之后**不单调**——按奇偶分成两条子列各自发散"
            f"（偶 n=16 已 14.4，n=30 达 2.39e3）。Chebyshev 同样分两条子列下降，n=30 是 2.43e-3。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# 2) lebesgue_growth
# --------------------------------------------------------------------------- #
def _lebesgue_growth(params: Mapping[str, Any], t: float) -> Frame:
    n_max = int(round(float(params.get("n_max", _CRIT_N_MAX))))
    n_max = max(10, min(60, n_max))
    grid = int(round(float(params.get("grid", 2001))))
    grid = max(201, min(20001, grid))
    n_cur = max(2, min(n_max, int(round(float(t)))))
    tab = lebesgue_growth_table(n_max, n_grid=grid, refine_rounds=3)
    ns = tab["n"]

    lam_eq = tab["lambda_equispaced"]
    lam_ch = tab["lambda_chebyshev"]
    # Λ 不会等于 0（Λ_n ≥ 1），但 log 轴仍统一走 clamp，保证不会有 NaN 进帧
    log_eq, info_eq = _clamped_log10(tab["log10_lambda_equispaced"], lo=-16.0, hi=16.0)
    log_ch, info_ch = _clamped_log10(tab["log10_lambda_chebyshev"], lo=-16.0, hi=16.0)
    log_th_eq, _ = _clamped_log10(tab["log10_theory_equispaced"], lo=-16.0, hi=16.0)
    log_th_ch, _ = _clamped_log10(tab["log10_theory_chebyshev"], lo=-16.0, hi=16.0)
    slope_eq = tab["slope_measured_equispaced"]
    slope_ch = tab["slope_measured_chebyshev"]
    slope_th_eq = tab["slope_theory_equispaced"]
    slope_th_ch = tab["slope_theory_chebyshev"]

    j = int(np.argmin(np.abs(ns - n_cur)))
    fr = Frame(chapter="2.1", viz="lebesgue_growth", kind="frame")
    fr.add("n_axis", ns, unit="1", label="节点数 n")
    fr.add("lambda_equispaced", lam_eq, unit="1", label="Λ_n（等距）实测")
    fr.add("lambda_chebyshev", lam_ch, unit="1", label="Λ_n（Chebyshev）实测")
    fr.add("theory_equispaced", tab["theory_equispaced"], unit="1", label="渐近公式 2^(n+1)/(e n ln n)")
    fr.add("theory_chebyshev", tab["theory_chebyshev"], unit="1", label="渐近公式 (2/π)ln n + 0.9625")
    fr.add("log10_lambda_equispaced", log_eq, unit="1", label="log10 Λ_n（等距）")
    fr.add("log10_lambda_chebyshev", log_ch, unit="1", label="log10 Λ_n（Chebyshev）")
    fr.add("log10_theory_equispaced", log_th_eq, unit="1", label="log10 渐近（等距）")
    fr.add("log10_theory_chebyshev", log_th_ch, unit="1", label="log10 渐近（Chebyshev）")
    fr.add("slope_measured_equispaced", slope_eq, unit="1", label="实测 log-log 斜率（等距）")
    fr.add("slope_measured_chebyshev", slope_ch, unit="1", label="实测 log-log 斜率（Chebyshev）")
    fr.add("slope_theory_equispaced", slope_th_eq, unit="1", label="理论 log-log 斜率（等距）")
    fr.add("slope_theory_chebyshev", slope_th_ch, unit="1", label="理论 log-log 斜率（Chebyshev）")
    fr.add("error_asymptotic_equispaced", tab["error_asymptotic_equispaced"], unit="1", label="|Λ_实测/Λ_渐近 − 1|（等距）")
    fr.add("error_asymptotic_chebyshev", tab["error_asymptotic_chebyshev"], unit="1", label="|Λ_实测/Λ_渐近 − 1|（Chebyshev）")
    fr.add("mark_n", np.array([float(n_cur)], dtype=float), unit="1", label="当前位置 n")
    fr.add("mark_lambda", np.array([float(lam_eq[j]), float(lam_ch[j])], dtype=float), unit="1", label="当前位置 Λ")
    slope_gap_eq = float(abs(slope_eq[j] - slope_th_eq[j]))
    slope_gap_ch = float(abs(slope_ch[j] - slope_th_ch[j]))
    fr.set_meta(
        n_max=n_max,
        grid=grid,
        n_mark=n_cur,
        lambda_equispaced_n=float(lam_eq[j]),
        lambda_chebyshev_n=float(lam_ch[j]),
        theory_equispaced_n=float(tab["theory_equispaced"][j]),
        theory_chebyshev_n=float(tab["theory_chebyshev"][j]),
        lambda_ratio=float(lam_eq[j]) / float(lam_ch[j]),
        lambda_equispaced_times_u=float(lam_eq[j]) * UNIT_ROUNDOFF,
        lambda_chebyshev_times_u=float(lam_ch[j]) * UNIT_ROUNDOFF,
        slope_measured_equispaced_n=float(slope_eq[j]),
        slope_theory_equispaced_n=float(slope_th_eq[j]),
        slope_gap_equispaced=slope_gap_eq,
        slope_measured_chebyshev_n=float(slope_ch[j]),
        slope_theory_chebyshev_n=float(slope_th_ch[j]),
        slope_gap_chebyshev=slope_gap_ch,
        error_asymptotic_equispaced_n=float(tab["error_asymptotic_equispaced"][j]),
        error_asymptotic_chebyshev_n=float(tab["error_asymptotic_chebyshev"][j]),
        growth_factor_equispaced=float(lam_eq[min(j + 1, n_max - 1)] / lam_eq[j]) if j + 1 < n_max else None,
        log10_floor=-16.0,
        n_lambda_at_floor=info_eq["n_floor"] + info_ch["n_floor"],
        n_exact_zero_lambda=_count_exact_zero(lam_eq) + _count_exact_zero(lam_ch),
        n_exact_zero_slope=_count_exact_zero(slope_eq) + _count_exact_zero(slope_ch),
        overflow=bool(np.max(np.log10(np.maximum(lam_eq, 1e-300))) > 16.0),
    )
    fr.set_strings(
        family_bad=_FAMILY_ZH["equispaced"],
        family_good=_FAMILY_ZH["chebyshev"],
        note=(
            f"n={n_cur}：Λ_等距={lam_eq[j]:.4g}（渐近公式 {tab['theory_equispaced'][j]:.4g}，"
            f"相对偏差 {tab['error_asymptotic_equispaced'][j]:.3%}），"
            f"Λ_Chebyshev={lam_ch[j]:.6g}（公式 {tab['theory_chebyshev'][j]:.6g}）；"
            f"实测 log-log 斜率 {slope_eq[j]:.4f} vs 理论 {slope_th_eq[j]:.4f}（等距）、"
            f"{slope_ch[j]:.4f} vs {slope_th_ch[j]:.4f}（Chebyshev）。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# 3) error_curve
# --------------------------------------------------------------------------- #
def _error_curve(params: Mapping[str, Any], t: float) -> Frame:
    n = int(round(float(params.get("n", _CRIT_N_EQ))))
    n = max(2, min(24, n))
    a = float(params.get("a", 25.0))
    func = str(params.get("function", "runge"))
    morph = min(1.0, max(0.0, float(t)))
    xs = np.linspace(-1.0, 1.0, _N_GRID)
    f = test_function_values(func, xs, a)

    nd = morphed_nodes(n, morph, -1.0, 1.0)
    vals = test_function_values(func, nd, a)
    p = barycentric_interpolate(nd, vals, xs)
    err = pointwise_error(f, p)
    sup = sup_derivative_over_factorial(func, n + 1, a=a, n_grid=1201)
    env = error_envelope(nd, xs, sup)
    monic = np.abs(monic_product(nd, xs))

    nd_eq = node_family("equispaced", n)
    nd_ch = node_family("chebyshev", n)
    e_eq = pointwise_error(f, barycentric_interpolate(nd_eq, test_function_values(func, nd_eq, a), xs))
    e_ch = pointwise_error(f, barycentric_interpolate(nd_ch, test_function_values(func, nd_ch, a), xs))

    # Λ 随变形参数 t 的曲线（这是"为什么 Chebyshev 好"的量化答案）
    ts = np.linspace(0.0, 1.0, 21)
    lam_t = np.array(
        [lebesgue_constant(morphed_nodes(n, float(s), -1.0, 1.0), -1.0, 1.0, n_grid=2001, refine_rounds=3) for s in ts]
    )

    e_eq_c, _ = _clamped(e_eq)
    e_ch_c, _ = _clamped(e_ch)
    lam_t_c, _ = _clamped(lam_t, floor=1.0, ceiling=1e18)

    lam_cur = lebesgue_constant(nd, -1.0, 1.0, n_grid=2001, refine_rounds=3)
    lam_eq = lebesgue_constant(nd_eq, -1.0, 1.0, n_grid=2001, refine_rounds=3)
    lam_ch = lebesgue_constant(nd_ch, -1.0, 1.0, n_grid=2001, refine_rounds=3)
    err_c, info_err = _clamped(err)
    env_c, _ = _clamped(env)
    monic_c, info_monic = _clamped(monic)

    # 包络只在它明显高于舍入底的地方才有"上界"意义（否则量到的是舍入误差）
    valid = env > 100.0 * UNIT_ROUNDOFF
    tight = err[valid] / env[valid] if np.any(valid) else np.zeros(1)
    fr = Frame(chapter="2.1", viz="error_curve", kind="frame")
    fr.add("x_grid", xs, unit="1", label="x")
    fr.add("f_values", f, unit="1", label="f(x)")
    fr.add("p_morph", p, unit="1", label="插值多项式（变形参数 t）")
    fr.add("error", err_c, unit="1", label="|f-p|（已夹地板）")
    fr.add("error_raw", err, unit="1", label="|f-p|（真值，节点上恰为 0）")
    fr.add("envelope", env_c, unit="1", label="严格误差包络")
    fr.add("monic", monic_c, unit="1", label="|∏(x-x_i)|（首一乘积，节点上恰为 0）")
    fr.add("error_equispaced", e_eq_c, unit="1", label="|f-p|（等距节点，同一 n）")
    fr.add("error_chebyshev", e_ch_c, unit="1", label="|f-p|（Chebyshev 节点，同一 n）")
    fr.add("nodes_x", nd, unit="1", label="当前节点")
    fr.add("nodes_y", vals, unit="1", label="节点值")
    fr.add("t_axis", ts, unit="1", label="节点分布变形 t")
    fr.add("lambda_vs_t", lam_t_c, unit="1", label="Λ(t)")
    fr.add("mark_t", np.array([morph, lam_cur], dtype=float), unit="1", label="当前位置 (t, Λ)")
    fr.add("mark_error", np.array([float(np.max(err))], dtype=float), unit="1", label="当前最大误差")
    fr.set_meta(
        n=n,
        a=a,
        morph=morph,
        is_runge=bool(func == "runge"),
        max_error=float(np.max(err)),
        max_error_equispaced=float(np.max(e_eq)),
        max_error_chebyshev=float(np.max(e_ch)),
        lambda_morph=lam_cur,
        lambda_equispaced=lam_eq,
        lambda_chebyshev=lam_ch,
        lambda_ratio=lam_eq / lam_ch,
        max_envelope=float(np.max(env)),
        envelope_looseness=_finite_or_none(float(np.max(env)) / float(np.max(err))) if np.max(err) > 0 else None,
        tightness_median=_finite_or_none(float(np.median(tight))) if tight.size else None,
        tightness_points=int(np.count_nonzero(valid)),
        sup_derivative_over_factorial=sup,
        monic_max=float(np.max(monic)),
        monic_sup_theory=float(2.0 ** (1 - n)),
        monic_ratio_to_theory=_finite_or_none(float(np.max(monic)) / float(2.0 ** (1 - n))),
        error_floor=ROUNDOFF_FLOOR,
        n_error_at_floor=info_err["n_floor"],
        n_monic_at_floor=info_monic["n_floor"],
        n_exact_zero_error=_count_exact_zero(err),
        n_exact_zero_monic=_count_exact_zero(monic),
        n_error_nonfinite=info_err["n_nonfinite"],
        overflow=bool(np.max(err) > _LOG_CEIL or np.max(env) > _LOG_CEIL),
    )
    fr.set_strings(
        function_label=_FUNC_ZH[func],
        note=(
            f"n={n}, t={morph:.2f}：Λ={lam_cur:.4g}（等距 {lam_eq:.4g} -> Chebyshev {lam_ch:.4g}），"
            f"最大误差 {np.max(err):.3e}，严格包络 {np.max(env):.3e}"
            f"（宽松倍数 {float(np.max(env)) / float(np.max(err)):.3g}）。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# 4) barycentric_vs_vandermonde
# --------------------------------------------------------------------------- #
def _barycentric_vs_vandermonde(params: Mapping[str, Any], t: float) -> Frame:
    family = str(params.get("distribution", NODE_KINDS[0]))
    n_max = int(round(float(params.get("n_max", _CRIT_VDM))))
    n_max = max(6, min(_MAX_N_VDM, n_max))
    func = str(params.get("function", "exp"))
    n_cur = max(2, min(n_max, int(round(float(t)))))

    tab = interpolation_method_table(n_max, kind=family, function=func, n_grid=_N_GRID)
    ns = tab["n"]
    err_b = tab["error_barycentric"]
    err_v = tab["error_vandermonde"]
    cerr = tab["error_coefficients_vandermonde"]
    kappa = tab["kappa_vandermonde"]
    kbound = tab["kappa_bound_vandermonde"]
    resid = tab["residual_vandermonde"]

    log_b, info_b = _clamped_log10(tab["log10_error_barycentric"], lo=-18.0, hi=18.0)
    log_v, _ = _clamped_log10(tab["log10_error_vandermonde"], lo=-18.0, hi=18.0)
    log_c, info_c = _clamped_log10(tab["log10_error_coefficients_vandermonde"], lo=-18.0, hi=18.0)
    log_k, _ = _clamped_log10(tab["log10_kappa_vandermonde"], lo=0.0, hi=24.0)
    base = 3.0 if family == "equispaced" else (1.0 + np.sqrt(2.0))
    kappa_theory = base ** ns
    log_kt, _ = _clamped_log10(np.log10(kappa_theory), lo=0.0, hi=24.0)
    slope_k = _central_loglog_slope(ns, kappa)
    slope_kt = _central_loglog_slope(ns, kappa_theory)

    j = int(np.argmin(np.abs(ns - n_cur)))
    fr = Frame(chapter="2.1", viz="barycentric_vs_vandermonde", kind="frame")
    fr.add("n_axis", ns, unit="1", label="节点数 n")
    fr.add("error_barycentric", err_b, unit="1", label="函数值误差：重心公式")
    fr.add("error_vandermonde", err_v, unit="1", label="函数值误差：Vandermonde + Horner")
    fr.add("error_coefficients_vandermonde", cerr, unit="1", label="系数误差：Vandermonde（对有理数精确解）")
    fr.add("kappa_vandermonde", kappa, unit="1", label="κ₁(V) 实测")
    fr.add("kappa_bound_vandermonde", kbound, unit="1", label="最坏情况系数误差尺度 κ₁(V)·u")
    fr.add("kappa_theory", kappa_theory, unit="1", label=f"参考指数 {base:.4g}^n")
    fr.add("residual_vandermonde", resid, unit="1", label="后向残差 ‖Vc−y‖/(‖V‖‖c‖+‖y‖)")
    fr.add("log10_error_barycentric", log_b, unit="1", label="log10 函数值误差（重心）")
    fr.add("log10_error_vandermonde", log_v, unit="1", label="log10 函数值误差（Vandermonde）")
    fr.add("log10_error_coefficients_vandermonde", log_c, unit="1", label="log10 系数误差（Vandermonde）")
    fr.add("log10_kappa_vandermonde", log_k, unit="1", label="log10 κ₁(V)")
    fr.add("log10_kappa_theory", log_kt, unit="1", label="log10 参考指数曲线")
    fr.add("slope_kappa_measured", slope_k, unit="1", label="实测 log-log 斜率 d log κ/d log n")
    fr.add("slope_kappa_theory", slope_kt, unit="1", label="参考曲线的 log-log 斜率")
    fr.add("mark_n", np.array([float(n_cur)], dtype=float), unit="1", label="当前位置 n")
    fr.add(
        "mark_error",
        np.array([float(log_b[j]), float(log_v[j]), float(log_c[j])], dtype=float),
        unit="1",
        label="当前位置三条通道的 log10 误差",
    )
    fr.set_meta(
        n_max=n_max,
        n_mark=n_cur,
        is_chebyshev=bool(family == "chebyshev"),
        kappa_n=float(kappa[j]),
        kappa_log2_per_n=float(np.log2(kappa[j]) / ns[j]) if kappa[j] > 0.0 else None,
        kappa_theory_base=float(base),
        kappa_ratio_to_theory=float(kappa[j] / kappa_theory[j]),
        kappa_bound_n=float(kbound[j]),
        error_barycentric_n=float(err_b[j]),
        error_vandermonde_n=float(err_v[j]),
        error_coefficients_n=float(cerr[j]),
        coefficient_over_bound=_finite_or_none(float(cerr[j]) / float(kbound[j])) if kbound[j] > 0.0 else None,
        coefficient_over_value_error=_finite_or_none(float(cerr[j]) / float(err_b[j])) if err_b[j] > 0.0 else None,
        residual_n=float(resid[j]),
        slope_kappa_measured_n=float(slope_k[j]),
        slope_kappa_theory_n=float(slope_kt[j]),
        n_coefficient_error_clamped=info_c["n_ceiling"],
        log10_floor=-18.0,
        n_error_at_floor=info_b["n_floor"] + info_c["n_floor"],
        n_exact_zero_error=_count_exact_zero(err_b) + _count_exact_zero(err_v),
        n_exact_zero_residual=_count_exact_zero(resid),
        overflow=bool(np.max(cerr) > 1e18),
        kappa_saturated=bool(np.max(kappa) > 1e24),
    )
    fr.set_strings(
        family=_FAMILY_ZH[family],
        function_label=_FUNC_ZH[func],
        note=(
            f"n={n_cur}：函数值误差 重心 {err_b[j]:.3e} / Vandermonde {err_v[j]:.3e}（几乎并列），"
            f"而**系数**误差 {cerr[j]:.3e}，κ₁(V)={kappa[j]:.3e}（约 {base:.3g}^n 的 {kappa[j] / kappa_theory[j]:.3f} 倍），"
            f"后向残差始终 {resid[j]:.2e}。"
        ),
    )
    return fr


def _central_loglog_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """中心差分的 log-log 局部斜率（自研，不用 np.gradient——P3 禁用成品算法）。"""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    n = xs.size
    out = np.zeros(n, dtype=float)
    if n < 2:
        return out
    for i in range(n):
        i0 = max(0, i - 1)
        i1 = min(n - 1, i + 1)
        if i1 == i0:
            out[i] = 0.0
            continue
        dx = np.log10(xs[i1]) - np.log10(xs[i0])
        dy = np.log10(max(ys[i1], 1e-300)) - np.log10(max(ys[i0], 1e-300))
        out[i] = dy / dx if dx != 0.0 else 0.0
    return out


_DISPATCH = {
    "runge_nodes": _runge_nodes,
    "lebesgue_growth": _lebesgue_growth,
    "error_curve": _error_curve,
    "barycentric_vs_vandermonde": _barycentric_vs_vandermonde,
}


def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """按 viz 分发（``server.frames.build_frame`` 会做 requiredAux / meta 体检）。"""
    try:
        fn = _DISPATCH[viz_id]
    except KeyError as exc:  # pragma: no cover - manifest 与 dispatch 同步由测试保证
        raise KeyError(f"2.1 章没有可视化 {viz_id!r}") from exc
    return fn(params, float(t))


MANIFEST: dict[str, Any] = {
    "chapter": "2.1",
    "title": "插值与多项式逼近",
    "part": "II",
    "summary": (
        "节点分布、Lebesgue 常数与误差通道：同一个函数、同样多的节点，只换节点分布，"
        "误差能差 6 个量级；把「发散」量化成 Λ_n 的指数率，再把「解 Vandermonde」"
        "拆成函数值与系数两条互不相干的误差通道。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "runge_nodes",
            "title": "节点往两端集中：Runge 振荡为什么被压平",
            "description": (
                "同一个函数、同样多的节点，只换节点位置：等距插值在区间两端剧烈振荡，"
                "Chebyshev 节点把误差摊平。时间轴是「已经用了多少个节点」，记为 $k$；"
                "主图叠画两种节点分布得到的插值曲线，副图用对数纵轴画整条误差曲线，"
                "以及由高阶导数估计出的严格误差上界（包络）。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "distribution",
                    "label": "节点分布",
                    "type": "enum",
                    "options": ["equispaced", "chebyshev"],
                    "default": "equispaced",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "节点分布方式：equispaced 是区间上的等距节点，chebyshev 是两端更密的 "
                        "Chebyshev–Lobatto 节点。切到 Chebyshev 后，主图两端的振荡通常明显收窄，"
                        "副图误差曲线被压到更低的位置；另一种分布仍会同时画出，方便直接对照。"
                        "这个参数与 n 一起决定节点位置，也与 a 一起决定 Runge 函数奇点离区间有多近。"
                    ),
                },
                {
                    "name": "n",
                    "label": "节点数 n",
                    "type": "int",
                    "min": 1,
                    "max": 40,
                    "default": 12,
                    "critical": [_CRIT_N_EQ],
                    "criticalNote": (
                        "实测：等距插值的最大误差在 n=7 触底 0.247，之后**不是单调上涨**——"
                        "按奇偶分成两条子列各自发散（偶数 n 更差：n=8 是 1.045、n=12 是 3.663、"
                        "n=16 是 14.39），n=12 已经比函数自身的量级 1 还大；n=20 是 59.8、n=30 是 2.39e3。"
                        "同期 Chebyshev 节点也是奇偶两条子列下降：n=10 是 0.132、n=30 是 2.43e-3。"
                        "所以 n=12 是「等距插值已经比不插值更糟」的临界点。"
                    ),
                    "beyond": "above",
                    "help": (
                        "多项式次数 n（无单位；后端实际使用 $n+1$ 个含端点的节点，所以界面上叫节点数，"
                        "但次数比节点数少一）。调大时节点变多、次数升高：等距节点下误差先降后升，"
                        "而且在区间两端先坏；Chebyshev 节点下通常持续下降。看副图整条误差曲线和包络，"
                        "不要只看主图是否穿过节点；它与 distribution、a 一起决定误差量级。"
                    ),
                },
                {
                    "name": "a",
                    "label": "Runge 参数 a（越大，奇点越靠近区间）",
                    "type": "float",
                    "min": 1.0,
                    "max": 49.0,
                    "step": 1.0,
                    "default": 25.0,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "Runge 函数 $1/(1+ax^2)$ 的形状参数，无单位；a 越大，函数在 0 附近越尖，"
                        "复奇点 $\\pm i/\\sqrt a$ 越靠近实轴。调大 a 会让等距插值更早、更猛地发散，"
                        "也会让 Chebyshev 的收敛变慢；调小则曲线更平缓、两种节点都更容易。"
                        "先看主图两端，再看副图误差与包络，并与 n、distribution 一起比较。"
                    ),
                },
            ],
            "time": {
                "mode": "scrub",
                "tMax": 40.0,
                "dt": 1.0,
                "label": "已用节点数 k",
                "tMaxParam": "n",
            },
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": True,
                    "title": "$f(x)$ 与两种节点分布的插值多项式 $p_k(x)$",
                    "ylabel": "$y$",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "插值误差 $|f-p|$（$\\log_{10}$ 轴）与严格包络",
                    "ylabel": "误差 $|f-p|$",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "只看最大误差会把 Runge 现象讲反：最大误差只是一个数，位置在两端；真正的画面是"
                "「两端振荡、中间几乎贴住」，所以这里画整条误差曲线。"
                "对数轴上的 0 是假的：插值误差在节点上恰好为 $0$（$k=12$ 时有 12 个点），"
                "直接取对数会得到 $-\\infty$，所以画图前把 $0$ 抬到显示地板 $u/2\\approx1.11\\times10^{-16}$；"
                "贴地板的点会单独标出，不能把它们的数值当成真实误差。"
                "严格误差包络不是紧的：$n=12$ 时等距节点的包络约 $4.51\\times10^6$，实际误差只有 $3.66$，"
                "宽松了约 $1.2\\times10^6$ 倍。包络只说「不会超过」，拿它当误差估计会得到荒谬的结论；"
                "换成 $e^x$ 后同一公式只宽松约 $2.5$ 倍，可见松紧取决于函数的高阶导数。"
            ),
            "theory": [
                "**设定与唯一性**。给定 $n+1$ 个互异节点 $x_0<\\ldots<x_n$（本页参数 $n$ 是次数，所以实际画出 $n+1$ 个节点）与 $y_i=f(x_i)$，要求次数不超过 $n$ 的多项式 $p_n$ 满足 $p_n(x_i)=y_i$。若有两个这样的多项式，它们的差在 $n+1$ 个点上为零，而次数不超过 $n$ 的非零多项式至多有 $n$ 个根，所以差恒为零；存在性可由 Lagrange 构造或解 $Vc=y$ 得到。",
                "**误差恒等式的推导**。设 $f\\in C^{n+1}$，固定一个不是节点的 $x$，令 $g(t)=f(t)-p_n(t)-[f(x)-p_n(x)]\\omega(t)/\\omega(x)$，其中 $\\omega(t)=\\prod_i(t-x_i)$。则 $g$ 在 $x$ 与所有节点处为零，共 $n+2$ 个互异零点；反复用 Rolle 定理 $n+1$ 次，存在 $\\xi$ 使 $g^{(n+1)}(\\xi)=0$。因 $p_n^{(n+1)}=0$、$\\omega^{(n+1)}=(n+1)!$，移项得 $f(x)-p_n(x)=f^{(n+1)}(\\xi)\\omega(x)/(n+1)!$。节点处两边都是零，公式也成立。",
                "**误差的两个因子**。恒等式把误差拆成“函数的高阶导数”与“节点乘积”两部分：前者只由 $f$ 决定，后者只由节点位置决定。等距节点在两端使 $\\omega(x)$ 很快变大，所以 Runge 函数的误差先出现在两端；Chebyshev–Lobatto 节点 $x_k=\\cos(k\\pi/n)$ 使 $\\omega=2^{1-n}T_n$，在 $[-1,1]$ 上最大值为 $2^{1-n}$，是所有首一 $n$ 次多项式中最小的，这就是极小极大性质。",
                "**为什么 Chebyshev 能收敛**。Runge 函数 $f(x)=1/(1+ax^2)$ 的极点在 $\\pm i/\\sqrt a$，围住 $[-1,1]$ 且刚好碰到极点的 Bernstein 椭圆参数为 $\\rho=1/\\sqrt a+\\sqrt{1+1/a}$，$a=25$ 时约 $1.2198$。Chebyshev 插值的误差大致每步乘 $1/\\rho\\approx0.82$，所以图中 $n$ 增大时它稳定下降；等距节点没有这个几何收敛机制，误差最终由 $\\Lambda_n$ 的指数增长主导。",
                "**Lebesgue 常数把放大说清**。把插值写成算子 $P_n f=\\sum_i f(x_i)\\ell_i$，则 $\\|P_n\\|=\\Lambda_n=\\max_x\\sum_i|\\ell_i(x)|$，且 $\\Lambda_n\\ge1$。对最佳一致逼近 $p^*$ 有 $f-P_nf=(f-p^*)-P_n(f-p^*)$，取范数得 $\\|f-P_nf\\|\\le(1+\\Lambda_n)\\|f-p^*\\|$。等距节点 $\\Lambda_n\\sim2^{n+1}/(e n\\log n)$，Chebyshev 节点只有 $(2/\\pi)\\log n+0.9625$ 量级，所以同一函数、同一 $n$，只换节点就会差很多个数量级。",
                "**图上的四条线索**。主图看两条插值曲线在两端的行为：等距曲线在端点附近剧烈摆动，Chebyshev 曲线把误差摊平。副图是对数纵轴的整条误差曲线与红色虚线包络；包络来自 $\\sup|f^{(n+1)}|/(n+1)!\\cdot|\\omega(x)|$，是严格上界，不是误差估计。最下面的扫描图把偶数 $n$ 与奇数 $n$ 分开：等距误差在 $n=7$ 触底约 $0.247$，之后按两条子列各自上升，$n=30$ 约 $2.39\\times10^3$，而 Chebyshev 同期约 $2.43\\times10^{-3}$。",
                "**常见误读与数值极限**。穿过节点不等于拟合好：节点上误差按定义为零，对数轴把它抬到地板 $u/2$ 才画得出来，这些点不能当成真实误差。$n$ 越大也不等于越好，等距误差是锯齿形而非单调。包络在 Runge 函数上可能宽松上百万倍（$n=12$ 约 $4.51\\times10^6$ 对实际 $3.66$），拿它当误差估计会得到荒谬结论；换成 $e^x$ 后同一公式只宽松约 $2.5$ 倍，说明松紧取决于高阶导数。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "lebesgue_growth",
            "title": "Lebesgue 常数：节点分布一变，误差放大为何指数增长",
            "description": (
                "把「发散」变成一个可测量的数：$\\Lambda_n=\\max_x\\sum_i|L_i(x)|$，其中 $L_i$ 是插值的基函数。"
                "等距节点按约 $2^n$ 爆炸（$n=30$ 时约 $6.6\\times10^6$），Chebyshev 节点只按 $\\ln n$ 增长（约 $3.13$）。"
                "副图同时画出实测与理论的 log-log 斜率，再叠上渐近公式的相对偏差。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "n_max",
                    "label": "最大节点数 n_max",
                    "type": "int",
                    "min": 10,
                    "max": 60,
                    "default": 30,
                    "critical": [_CRIT_LAMBDA],
                    "criticalNote": (
                        "n=50 时实测 Λ_等距=3.64e12、Λ·u=4.0e-4：插值误差的理论界已经比 1e-3 还大，"
                        "双精度下的等距插值彻底失效（n=60 时 Λ·u=0.25，界完全失去意义）；"
                        "同一列 Chebyshev 是 3.4527（渐近公式 3.4530，相对偏差 1e-4）。"
                        "把 n_max 拖过 50，等距那条线才会露出完整的指数爆炸。"
                    ),
                    "beyond": "above",
                    "help": (
                        "Lebesgue 常数随节点数增长的扫描上限（次数，实际节点数为 $n_max+1$）。"
                        "调大后横轴延伸，等距曲线会露出更完整的指数爆炸，Chebyshev 只缓慢上升；"
                        "拖过 50 后 $\\Lambda_n u$ 角标会提醒双精度极限。它与 grid 一起决定测量可信度，"
                        "看主轴两条曲线、斜率面板和渐近偏差面板。"
                    ),
                },
                {
                    "name": "grid",
                    "label": "Lebesgue 函数求最大值用的网格点数",
                    "type": "int",
                    "min": 201,
                    "max": 20001,
                    "default": 2001,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "求 Lebesgue 函数最大值时使用的均匀网格点数（无单位，范围 201 到 20001）。"
                        "调大网格会更准地抓到靠近端点的窄峰，$\\Lambda_n$ 读数通常略微变大；"
                        "调小则容易系统性低估，等距节点尤其明显。看主图实测与渐近公式的差距、"
                        "以及偏差面板；与 n_max 联动，n_max 越大越需要更细的网格。"
                    ),
                },
            ],
            "time": {
                "mode": "scrub",
                "tMax": 60.0,
                "dt": 1.0,
                "label": "节点数 n",
                "tMaxParam": "n_max",
            },
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": True,
                    "title": "$\\Lambda_n$（$\\log_{10}$ 轴）实测 vs 渐近公式",
                    "ylabel": "$\\Lambda_n$",
                    "scale": "log",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "$\\log$-$\\log$ 斜率：实测 vs 理论",
                    "ylabel": "$d\\log\\Lambda_n/d\\log n$",
                },
            ],
            "pitfalls": (
                "线性坐标轴上几乎读不出这条曲线：等距的 $\\Lambda_n$ 从 $1.25$ 涨到 $6.6\\times10^6$，"
                "前 20 个点会被压成贴着 $0$ 的直线。必须用对数轴，并把实测斜率与理论斜率同时画出，"
                "否则「看起来在涨」和「按 $2^n$ 在涨」是两件事（$n=20$ 附近等距实测斜率约 $12.50$，"
                "理论约 $12.52$；Chebyshev 为 $0.2218$ 对 $0.2219$）。"
                "用网格搜索 $\\Lambda_n$ 会系统性低估它：等距节点的峰值挤在离端点约 $O(n^{-2})$ 宽的薄层里，"
                "均匀撒点容易漏掉；把网格从 2001 加到 20001，$n=30$ 的读数还会变化。"
                "渐近公式不是等式：$n=30$ 时等距公式与实测的比值约 $0.853$（差 $14.8$%），"
                "$n=60$ 时降到 $0.654$。把公式当精确值反推 $\\Lambda_n$ 会错得很远。"
            ),
            "theory": [
                "**定义与算子范数**。对节点 $x_0,\\ldots,x_n$，Lagrange 基为 $\\ell_i(x)=\\prod_{j\\ne i}(x-x_j)/(x_i-x_j)$，插值算子为 $P_n f=\\sum_i f(x_i)\\ell_i(x)$。对任意 $x$，$|P_nf(x)|\\le\\sum_i|\\ell_i(x)|\\|f\\|_\\infty$，所以 $\\|P_n\\|\\le\\Lambda_n=\\max_x\\sum_i|\\ell_i(x)|$；取 $f$ 在节点上等于 $\\operatorname{sign}\\ell_i(x^*)$ 的分片线性函数可把等号逼近，故算子范数正是 $\\Lambda_n$。又因常数函数被精确插值，$\\Lambda_n\\ge1$。",
                "**误差放大从哪来**。设 $p^*$ 是 $\\Pi_n$ 中对 $f$ 的最佳一致逼近。因为 $P_n$ 在 $\\Pi_n$ 上是恒等映射，$P_np^*=p^*$，于是 $f-P_nf=(f-p^*)-P_n(f-p^*)$。取范数并用算子范数，得到 $\\|f-P_nf\\|_\\infty\\le(1+\\Lambda_n)\\|f-p^*\\|_\\infty$。最佳逼近误差只由 $f$ 与多项式空间决定，$\\Lambda_n$ 则完全是节点分布的属性；这就是“同一函数、同一 $n$、只换节点”也能差出数量级的原因。",
                "**等距节点为什么指数爆炸**。把 $\\ell_i(x)=\\omega(x)w_i/(x-x_i)$ 代入，$\\Lambda_n=\\max_x|\\omega(x)|\\sum_i|w_i|/|x-x_i|$，其中 $w_i=1/\\prod_{j\\ne i}(x_i-x_j)$。等距时 $w_i=(-1)^{n-i}/(i!(n-i)!h^n)$，靠近端点的项含 $n!$ 分之一，节点乘积在端点又很大；两边一起把 $\\Lambda_n$ 推向 $2^{n+1}/(e n\\log n)$ 的指数增长。图中 $n=30$ 时约 $6.60\\times10^6$，就是这条机制的直接读数。",
                "**Chebyshev 节点为什么只有对数增长**。取 $x_k=\\cos(k\\pi/n)$，节点乘积 $\\omega=2^{1-n}T_n$，最大值为 $2^{1-n}$；基函数的峰值集中在端点附近但相互错开，Lebesgue 函数的最大值只按 $\\log n$ 增长，渐近为 $(2/\\pi)\\log n+0.9625$。Erdős–Turán 下界说任何节点集都有 $\\Lambda_n\\ge(2/\\pi)\\log(n+1)-c$，所以 Chebyshev 的量级已经是最优的；它不能把 $\\Lambda_n$ 变成常数。",
                "**斜率面板怎么读**。主轴用 $\\log$-$\\log$ 坐标，纵轴是 $\\Lambda_n$；指数增长在 $\\log$-$\\log$ 图上表现为斜率随 $n$ 变大，而不是一条直线。局部斜率 $d\\log\\Lambda_n/d\\log n$ 对等距约等于 $n\\log2-1-1/\\log n$，$n=20$ 时约 $12.5$；Chebyshev 的斜率约 $(2/\\pi)/(\\Lambda_n\\log n)$，只有 $0.22$ 左右。实测斜率与理论斜率必须同时看：二者接近只能说明趋势对了，不能说明渐近公式是等式。",
                "**渐近公式、网格与舍入极限**。等距公式只是量级：$n=30$ 时实测与公式之比约 $0.853$，$n=60$ 降到约 $0.654$，图下方的偏差面板画的就是这个相对差。用均匀网格求 $\\Lambda_n$ 会低估它，因为峰值挤在离端点约 $O(n^{-2})$ 宽的薄层里；把 grid 从 2001 加到 20001，$n=30$ 的读数还会变。最后看 $\\Lambda_nu$：$n=50$ 时约 $4.0\\times10^{-4}$，$n=60$ 时约 $0.251$，此时插值误差的理论界已经被舍入项吃掉。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "error_curve",
            "title": "拖动节点从等距变成 Chebyshev：误差、包络与首一乘积怎么变",
            "description": (
                "时间轴让节点分布从等距连续变形到 Chebyshev：$t=0$ 是等距，$t=1$ 是 Chebyshev。"
                "主图看节点怎样移动、插值曲线怎样跟着变；副图用对数纵轴同时画当前误差、"
                "严格误差上界和首一乘积 $\\left|\\prod_i(x-x_i)\\right|$，并展示 $\\Lambda(t)$ 的整体变化"
                "（不是严格单调：$n=12$ 时实测有约 $0.137$ 的局部回升）。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "n",
                    "label": "节点数 n",
                    "type": "int",
                    "min": 2,
                    "max": 24,
                    "default": 12,
                    "critical": [_CRIT_N_EQ],
                    "criticalNote": (
                        "n=12 是等距插值误差（3.663）已经超过函数自身量级的位置（见 runge_nodes）；"
                        "同时 n=12 时 Λ 从等距的 89.3 降到 Chebyshev 的 2.54（35 倍），"
                        "首一乘积的最大值从 4.01e-3 降到 4.84e-4（与理论 2^(1-n)=4.88e-4 相差 0.8%），"
                        "变形滑块在这两个数之间连续过渡。"
                    ),
                    "beyond": "above",
                    "help": (
                        "多项式次数 n（无单位；后端使用 $n+1$ 个含端点的节点）。调大时插值多项式次数更高："
                        "等距端点的误差可能爆炸，而 Chebyshev 端点通常继续下降。看主图两端的振荡，"
                        "以及副图误差、包络、首一乘积三条线；它与 function、a 一起决定误差大小，"
                        "也决定 $\\Lambda(t)$ 的起点和终点。"
                    ),
                },
                {
                    "name": "function",
                    "label": "被插函数",
                    "type": "enum",
                    "options": ["runge", "exp"],
                    "default": "runge",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "被插函数：runge 是 $1/(1+ax^2)$，exp 是 $e^x$。runge 的高阶导数增长快，"
                        "等距插值会发散；exp 很光滑，等距插值也能收敛，于是节点变形带来的差别小得多。"
                        "切换后先看副图误差量级，再看包络是否还宽松；a 只在 runge 时生效。"
                    ),
                },
                {
                    "name": "a",
                    "label": "Runge 参数 a（只在 function=runge 时生效）",
                    "type": "float",
                    "min": 1.0,
                    "max": 49.0,
                    "step": 1.0,
                    "default": 25.0,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "Runge 函数的形状参数（无单位），只在 function=runge 时生效；a 越大，0 附近越尖，"
                        "复奇点越靠近区间。调大 a 会使等距端点的误差更快变大，也会让 Chebyshev 的误差"
                        "下降变慢；调小则曲线更平缓。看主图两端与副图误差，和 n、function 联读。"
                    ),
                },
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.01, "label": "节点分布变形 t（0=等距，1=Chebyshev）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": True,
                    "title": "$f$ 与随 $t$ 变形的插值多项式 $p_t$",
                    "ylabel": "$y$",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "$|f-p|$、严格包络、$|\\prod_i(x-x_i)|$（$\\log_{10}$ 轴）",
                    "ylabel": "误差与首一乘积",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "包络不是误差估计，而是严格上界：$n=12$、$f$ 取 Runge 时上界约 $4.51\\times10^6$，"
                "实际误差只有 $3.66$，宽松了约 $1.2\\times10^6$ 倍；换成 $e^x$ 同一公式只宽松约 $2.5$ 倍。"
                "两条线画在一起时容易误以为「误差就在包络附近」，实际可以差 6 个量级。"
                "当包络掉到 $u\\Lambda$ 以下时，“实测误差超过包络”并不是异常——此时量到的是舍入误差，"
                "而包络界只对精确插值误差成立；所以统计紧度时只比较显著高于 $100u$ 的点。"
                "$\\left|\\prod_i(x-x_i)\\right|$ 在节点上恰好为 $0$，对数轴必须夹一个显示地板，否则曲线会断成几段。"
                "变形的终点不是「节点变密」，而是按余弦分布把节点推向两端——中间反而变稀；"
                "所以 $\\Lambda(t)$ 整体下降，沿途可能有两三个百分点的局部回升，不能读成严格单调。"
            ),
            "theory": [
                "**变形节点的定义**。第 $i$ 个节点取 $(1-t)x_i^{\\text{eq}}+t x_i^{\\text{ch}}$，$t\\in[0,1]$，其中 $x_i^{\\text{eq}}$ 是等距节点、$x_i^{\\text{ch}}=\\cos(i\\pi/n)$ 是 Chebyshev–Lobatto 节点。$t=0$ 与 $t=1$ 是两个端点，中间是连续过渡。注意这不是“把节点变密”：总节点数不变，$t$ 变大时节点按余弦分布被推向两端，区间中间反而变稀。",
                "**误差与首一乘积**。对每个 $t$，插值多项式 $p_t$ 仍满足误差恒等式 $f(x)-p_t(x)=f^{(n+1)}(\\xi)\\omega_t(x)/(n+1)!$，$\\omega_t(x)=\\prod_i(x-x_i(t))$。因此把 $\\sup|f^{(n+1)}|/(n+1)!$ 乘到 $|\\omega_t|$ 上，就得到一条严格包络；图中误差、包络、首一乘积三条线共用同一个对数纵轴，正是为了直接比较这两个因子。",
                "**Chebyshev 乘积的极小极大性**。在 $t=1$ 时 $\\omega_1=2^{1-n}T_n$，而 $T_n$ 在 $[-1,1]$ 上绝对值不超过 $1$，所以 $\\max|\\omega_1|=2^{1-n}$。若存在首一多项式 $q$ 的范数更小，考虑 $q-2^{1-n}T_n$：它在 $T_n$ 的 $n+1$ 个交错极值点上交替变号，至少有 $n$ 个零点，但次数不超过 $n-1$，只能恒为零，矛盾。于是 $2^{1-n}$ 是最小可能值；$n=12$ 时它是 $4.88\\times10^{-4}$。",
                "**变形为什么让误差下降**。$t$ 从 $0$ 到 $1$ 时，节点向两端聚集，$\\omega_t$ 在端点附近的最大值被压下去；图中 $n=12$ 时首一乘积最大值从约 $4.01\\times10^{-3}$ 降到 $4.84\\times10^{-4}$，约 $8.3$ 倍，已经接近理论值 $2^{1-n}$，差距约 $0.8$% 来自网格取最大值。误差曲线随之整体下移，主图两端的振荡先消失，这就是 Runge 现象被摊平的几何原因。",
                "**$\\Lambda(t)$ 不是严格单调**。$\\Lambda(t)=\\max_x\\sum_i|\\ell_i(x;t)|$ 是节点位置的非线性函数，节点的凸组合并不会让这个最大值自动单调下降。实测 $n=12$ 时它从约 $89.3$ 降到 $2.54$，整体下降约 $35$ 倍，但沿途有约 $0.137$ 的局部回升；图下方把 $21$ 个采样点全部画出，并标出全局最小点，最小点不一定在 $t=1$ 的端点。",
                "**包络的松紧与舍入地板**。包络只是不会超过，不是误差估计：$n=12$、Runge 函数时包络最大约 $4.51\\times10^6$，实际误差只有 $3.66$，宽松约 $1.2\\times10^6$ 倍；换成 $e^x$ 后同一公式只宽松约 $2.5$ 倍。包络低于 $100u$ 时，实测误差量到的是舍入噪声，比较紧度没有意义；图中只统计包络显著高于 $100u$ 的点。",
                "**常见误读**。节点上误差恰好为 $0$，对数轴必须把它抬到地板 $u/2$ 才能画，贴地板的点不是误差很小。$t$ 变大也不是节点更密，看主图节点位置比看曲线更可靠。最后不要把 $\\Lambda(t)$ 的局部回升当成数值错误：它来自最大值与节点位置的非线性关系，图上标出的全局最小点才是可下的结论。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "barycentric_vs_vandermonde",
            "title": "重心公式与解 Vandermonde：同一个多项式，三条误差通道",
            "description": (
                "同一批节点、同一个函数，比较三条误差通道：重心公式的函数值误差、"
                "解 Vandermonde 方程再用 Horner 求值的函数值误差，以及 Vandermonde 解的系数误差"
                "（真值由有理数精确消元得到）。副图画条件数 $\\kappa_1(V)$ 与参考指数曲线，"
                "并保留一直停在 $u$ 量级的后向残差。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "distribution",
                    "label": "节点分布",
                    "type": "enum",
                    "options": ["equispaced", "chebyshev"],
                    "default": "equispaced",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "节点分布：equispaced 为等距，chebyshev 为两端加密的 Chebyshev–Lobatto 节点。"
                        "切到 Chebyshev 后 $\\kappa_1(V)$ 的增长底数从约 3 降到约 $1+\\sqrt2$，"
                        "系数误差通道通常下降若干量级，而函数值误差两条线仍可能接近。"
                        "看主图三条通道与副图条件数、残差；与 n_max 一起决定比较区间。"
                    ),
                },
                {
                    "name": "n_max",
                    "label": "最大节点数 n_max",
                    "type": "int",
                    "min": 6,
                    "max": 36,
                    "default": 30,
                    "critical": [_CRIT_VDM],
                    "criticalNote": (
                        "n=30 实测：κ₁(V)=2.59e14（等距）/2.68e11（Chebyshev），"
                        "κ·u 达到 2.9e-2 / 3.0e-5，系数误差 2.69e-4 / 1.14e-6——"
                        "系数通道在这个点上已经丢掉全部有效位；而同一帧里函数值误差只有 2.45e-9 / 1.33e-15，"
                        "后向残差 1.6e-17。这正是「κ 大不等于公式不稳定」的临界演示。"
                    ),
                    "beyond": "above",
                    "help": (
                        "节点数扫描上限（次数，实际节点数为 $n_max+1$）。调大后曲线延伸到更大规模："
                        "Vandermonde 条件数按指数上升，系数误差会越过 $10^0$，而函数值误差与后向残差"
                        "仍可能很小。看三条通道何时分岔、$\\kappa$ 面板的斜率；与 distribution 一起"
                        "比较两种底数，不要把条件数直接当成误差倍数。"
                    ),
                },
                {
                    "name": "function",
                    "label": "被插函数",
                    "type": "enum",
                    "options": ["exp", "runge"],
                    "default": "exp",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "被插函数：exp 是 $e^x$，runge 是 $1/(1+25x^2)$。exp 的高阶导数温和，"
                        "才能把系数误差与函数值误差分开观察；换成 runge 后插值本身的发散误差会盖住"
                        "算法差异。默认用 exp，调参时先看主图三条线是否仍可分辨，再读 $\\kappa$ 与残差。"
                    ),
                },
            ],
            "time": {
                "mode": "scrub",
                "tMax": 36.0,
                "dt": 1.0,
                "label": "节点数 n",
                "tMaxParam": "n_max",
            },
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": True,
                    "title": "三条误差通道（$\\log_{10}$）：函数值 2 条 + 系数 1 条，叠 $\\kappa_1(V)\\cdot u$ 参考线",
                    "ylabel": "误差（$\\log_{10}$）",
                    "scale": "log",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "$\\kappa_1(V)$ 与参考指数曲线，及其 $\\log$-$\\log$ 斜率",
                    "ylabel": "$\\kappa_1(V)$ 与残差（$\\log$）",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "$\\kappa(V)$ 大不等于算得不准：解 Vandermonde 用的是部分主元 LU，它是后向稳定的——"
                "实测后向残差从 $n=6$ 到 $n=36$ 一直在 $10^{-17}$ 量级，而 $\\kappa(V)$ 涨了约 $10^{17}$ 倍。"
                "把条件数直接当成误差倍数，会得出「$n=30$ 时结果全错」的错误结论；同一帧里函数值误差只有约 $2.45\\times10^{-9}$。"
                "真正失控的是系数：用有理数精确解做真值，$n=30$ 时系数相对误差约 $2.69\\times10^{-4}$，"
                "$n=36$ 约 $1.19$，$n=44$ 约 $14.5$（有效位全丢）。"
                "系数与函数值走的是两条互不相干的通道：系数坏掉不一定影响区间内的取值，"
                "因为 Horner 求值时那些大系数会互相抵消。"
                "若把被插函数换成 Runge，函数值会被插值发散误差（约 $10^0$ 量级）盖住，"
                "此时比较两个算法毫无意义——这也是本页默认用 $e^x$ 的原因。"
                "条件数参考线只给量级：等距实测的 $\\log_2\\kappa/n$ 从 $1.53$ 漂到 $1.60$，"
                "Chebyshev 稳定在 $1.261\\pm0.002$，对应 $(1+\\sqrt2)^n$ 的底数约 $2.414$；它是趋势，不是精确公式。"
            ),
            "theory": [
                "**同一个多项式，不同的舍入路径**。给定 $n+1$ 个节点与函数值，插值多项式在数学上唯一。Lagrange 求和、Newton 差商、重心公式、解 Vandermonde 系统 $Vc=y$ 得到的是同一个 $p_n$，差别只在计算时怎样放大舍入误差。图中三条通道分别是重心公式的函数值误差、Vandermonde 解的函数值误差、以及 Vandermonde 解的系数误差，必须分开读。",
                "**重心公式的推导**。令 $\\omega(x)=\\prod_j(x-x_j)$，$w_i=1/\\prod_{j\\ne i}(x_i-x_j)$。Lagrange 基可写成 $\\ell_i(x)=\\omega(x)w_i/(x-x_i)$，于是 $p_n(x)=\\sum_i\\ell_i(x)y_i$。又因为常数函数 $1$ 的插值多项式就是 $1$，有 $\\sum_i\\ell_i(x)=1$；把分子分母同除以 $\\omega(x)$，得到 $p_n(x)=\\frac{\\sum_i w_i y_i/(x-x_i)}{\\sum_i w_i/(x-x_i)}$。每个求值点只需 $O(n)$ 次运算，权重只依赖节点。",
                "**容易抄错的恒等式**。由 $\\ell_i=\\omega w_i/(x-x_i)$ 求和得 $(\\sum_i w_i/(x-x_i))\\omega(x)=1$；这不是 $\\sum_i w_i/(x-x_i)=1$。后者随 $x$ 变化，并且随 $n$ 指数增长（$n=6$ 实测最小值约 $22.8$，$n=12$ 约 $4373$）。另外令 $x\\to\\infty$ 比较首项可得 $\\sum_i w_i=0$（$n\\ge1$）；把和为一错记成权重和为一，是本章最常见的公式级错误。",
                "**Vandermonde 条件数**。把 $p_n(x)=\\sum_{j=0}^n c_jx^j$ 代入节点条件，得到 $Vc=y$，$V_{ij}=x_i^j$。系数对右端扰动的敏感度由 $\\kappa_1(V)=\\|V\\|_1\\|V^{-1}\\|_1$ 控制：右端有相对扰动 $\\delta$ 时，系数的相对误差最坏可到 $\\kappa_1(V)\\delta$。等距节点上 $\\kappa_1(V)$ 约按 $3^n$ 增长，Chebyshev 节点约按 $(1+\\sqrt2)^n$ 增长；副图的参考指数曲线只是量级参考，不是精确公式。",
                "**后向稳定不等于前向准确**。部分主元 LU 是后向稳定的：算出的解满足 $(V+\\Delta V)\\hat c=y$，$\\|\\Delta V\\|$ 约为 $u\\|V\\|$，所以后向残差 $\\|V\\hat c-y\\|/(\\|V\\|\\|\\hat c\\|+\\|y\\|)$ 一直在 $u$ 量级。但前向误差还要乘条件数，系数误差约 $\\kappa_1(V)u$。图中 $n=30$ 时 $\\kappa_1(V)\\approx2.59\\times10^{14}$、$\\kappa_1(V)u\\approx2.9\\times10^{-2}$，系数误差约 $2.69\\times10^{-4}$，同一条曲线上的函数值误差却只有约 $2.45\\times10^{-9}$。",
                "**为什么系数坏、函数值仍可能好**。函数值误差走的是另一条通道：$p_n(x)=\\sum_i\\ell_i(x)y_i$，所以 $|\\delta p_n(x)|\\le\\Lambda_n\\max_i|\\delta y_i|$，量级是 $\\Lambda_nu$，与 $\\kappa_1(V)$ 无关（但 $\\kappa$ 大时系数表示本身已经不可信）。Horner 求值时那些大系数会互相抵消，因此区间内的取值可能仍然准确；图中 $n=36$ 时两条函数值误差约 $1.05\\times10^{-7}$ 与 $3.52\\times10^{-8}$，而系数误差已经约 $1.19$，后向残差仍是 $10^{-17}$ 量级。",
                "**图怎么读与常见误读**。主轴三条线：蓝线是重心公式，绿虚线是直接解，橙圈是系数误差，灰虚线是 $\\kappa_1(V)u$ 参考线；副轴同时画 $\\kappa_1(V)$ 与残差。看到 $\\kappa$ 涨到 $10^{14}$ 不能直接说公式不稳定，要看函数值误差和残差；反过来，残差小也不能保证系数准。参考指数曲线的底数会随节点分布漂移（等距的 $\\log_2\\kappa/n$ 从约 $1.53$ 漂到 $1.60$），它只用于判断趋势。",
            ],
        },
    ],
}
