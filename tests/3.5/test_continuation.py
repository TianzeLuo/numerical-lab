r"""3.5 章（分岔部分）：周期轨、乘子、周期倍增点与 Feigenbaum 常数的 A/C 类断言。

这一部分的裁判是**闭式解**，而不是"图看起来对"：

* :math:`r_1=3` 与 :math:`r_2=1+\sqrt6` 是周期倍增点的闭式；
* 2 周期的两个点与乘子 :math:`M=4+2r-r^2` 是闭式；
* 3 周期窗口起点 :math:`1+\sqrt8` 是切分岔的闭式；
* :math:`R_2=1+\sqrt5` 是超稳定参数的闭式。

Feigenbaum 常数 :math:`\delta=4.6692016091\ldots` 没有闭式，它是一条**收敛序列的极限**，
所以只能以 C 类断言出现：序列尾项必须落进容差，而且容差要按实测的收敛速度给。
本文件还把"周期是检测出来的"这件事单独测了一遍：窗口里闭合残差 1e-15、混沌区 1e-2，
两类点必须分在对数轴两端（混沌区的残差**不能**是 0——本仓初版就是那样错的）。
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

from numcore.ode import continuation as cont
from numcore.ode import dynamics as dyn
from numcore.verify import assertion, record

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

U = float(np.finfo(float).eps) / 2.0

#: 文献值（周期倍增点与超稳定参数），用来和二分实测对拍。
_R_DOUBLING_LIT = (3.0, 1.0 + math.sqrt(6.0), 3.5440903595519225, 3.564407266095432)
_R_SUPERSTABLE_LIT = (2.0, 1.0 + math.sqrt(5.0), 3.4985616993277016, 3.554640862768825,
                      3.566667379903097, 3.569243531630532)
_DELTA_LIT = 4.669201609102990
_R_ACCUM_LIT = 3.5699456718


# --------------------------------------------------------------------------- #
# 1. 周期倍增点与乘子
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    theory="周期倍增点由乘子穿过 −1 定义：r1=3（不动点失稳）、r2=1+√6、r3=3.5440903595519225、"
    "r4=3.564407266095432（后两个是标准文献值）",
    chapter="3.5",
)
def test_period_doubling_points_match_closed_forms() -> None:
    got = cont.period_doubling_points(max_period=8, scan=2000)
    assert got.size == 4, f"应当找到 4 个倍增点，得到 {got.size}: {got}"
    gaps = np.abs(got - np.asarray(_R_DOUBLING_LIT))
    record(
        label="四个周期倍增点与闭式/文献值的最大偏差",
        measured=float(np.max(gaps)),
        tol=1e-9,
        passed=float(np.max(gaps)) < 1e-9,
        detail="实测 " + "、".join(f"{v:.13f}" for v in got),
    )
    assert float(np.max(gaps)) < 1e-9
    record(
        label="r2 与 1+√6 的偏差",
        measured=float(abs(got[1] - (1.0 + math.sqrt(6.0)))),
        tol=1e-12,
        passed=abs(got[1] - (1.0 + math.sqrt(6.0))) < 1e-12,
        detail=f"1+√6={1.0 + math.sqrt(6.0):.15f}",
    )
    assert abs(got[1] - (1.0 + math.sqrt(6.0))) < 1e-12

    # 倍增点上乘子必须正好是 −1：这是"判据"本身的自洽性检查
    for k, period in enumerate((1, 2, 4, 8)):
        m = cont.cycle_multipliers(float(got[k]), period)
        worst = float(np.min(np.abs(m + 1.0))) if m.size else float("nan")
        record(
            label=f"p={period} 在 r{k + 1} 处的乘子（最接近 −1 的那条）",
            measured=abs(worst),
            tol=1e-6,
            passed=bool(abs(worst) < 1e-6),
            detail=f"乘子 {np.round(m, 8)}",
        )
        assert abs(worst) < 1e-6, f"p={period} 在 r={got[k]} 处的乘子离 −1 有 {abs(worst)}"


@assertion(
    "A",
    theory="logistic 的 2 周期有闭式：x=(r+1±√((r−3)(r+1)))/(2r)，乘子 M=g'(x₁)g'(x₂)=4+2r−r²",
    chapter="3.5",
)
def test_two_cycle_matches_closed_form() -> None:
    for r in (3.2, 3.4, 5.0**0.5 + 1.0):
        g = dyn.logistic(r)
        dg = dyn.logistic_deriv(r)
        cyc = cont.cycle_newton(g, dg, 2, seed=0.4, transient=4000)
        assert cyc is not None, f"r={r} 处没找到 2 周期"
        pts_closed, m_closed = cont.analytic_two_cycle(r)
        gap_pts = float(np.max(np.abs(np.sort(cyc.points) - np.sort(pts_closed))))
        record(
            label=f"r={r:.6g} 的 2 周期点 vs 闭式",
            measured=gap_pts,
            tol=1e-12,
            passed=gap_pts < 1e-12,
            detail=f"Newton 给 {np.round(np.sort(cyc.points), 12)}",
        )
        assert gap_pts < 1e-12
        gap_m = abs(cyc.multiplier - m_closed)
        record(
            label=f"r={r:.6g} 的乘子 vs 4+2r−r²",
            measured=gap_m,
            tol=1e-12,
            passed=gap_m < 1e-12,
            detail=f"实测 {cyc.multiplier:.12f}，闭式 {m_closed:.12f}",
        )
        assert gap_m < 1e-12
        # 同一件事的第二条路径：网格扫描 + 二分（不依赖 Newton 的收敛域）
        grid_m = cont.cycle_multipliers(r, 2)
        record(
            label=f"r={r:.6g} 的两条乘子路径之差（Newton vs 网格二分）",
            measured=float(np.max(np.abs(grid_m - m_closed))) if grid_m.size else float("nan"),
            tol=1e-9,
            passed=bool(grid_m.size and np.max(np.abs(grid_m - m_closed)) < 1e-9),
            detail=f"网格法给 {np.round(grid_m, 12)}",
        )
        assert grid_m.size and float(np.max(np.abs(grid_m - m_closed))) < 1e-9


@assertion(
    "A",
    theory="3 周期窗口起点是切分岔：g³(x)−x 出现二重根，闭式 r=1+√8=3.8284271247461903",
    chapter="3.5",
)
def test_period_three_window_starts_at_one_plus_sqrt_eight() -> None:
    got = cont.tangent_window_start(period=3)
    want = 1.0 + math.sqrt(8.0)
    gap = abs(got - want)
    record(
        label="3 周期窗口起点 vs 1+√8",
        measured=gap,
        tol=1e-9,
        passed=gap < 1e-9,
        detail=f"实测 {got:.12f}，闭式 {want:.12f}（判据：窗口内存在 |M|<1 的 3 周期）",
    )
    assert gap < 1e-9
    # 窗口里必须有稳定 3 周期，窗口外没有
    inside = cont.cycle_multipliers(want + 1e-4, 3)
    outside = cont.cycle_multipliers(want - 1e-3, 3)
    record(
        label="窗口内外的乘子（内 |M|<1、外无 3 周期）",
        measured=float(np.min(np.abs(inside))) if inside.size else float("nan"),
        tol=1.0,
        passed=bool(inside.size and np.min(np.abs(inside)) < 1.0 and outside.size == 0),
        detail=f"窗口内 {np.round(inside, 6)}，窗口外 {outside.size} 条",
    )
    assert inside.size and float(np.min(np.abs(inside))) < 1.0
    assert outside.size == 0


# --------------------------------------------------------------------------- #
# 2. 超稳定参数与 Feigenbaum 常数
# --------------------------------------------------------------------------- #
@assertion(
    "C",
    theory="Feigenbaum 常数 δ=lim (R_n−R_{n−1})/(R_{n+1}−R_n)=4.6692016091…，其中 R_n 是"
    "超稳定参数（2^{n−1} 周期轨经过 x=1/2）。序列收敛不快也不单调，所以容差按实测尾项给",
    chapter="3.5",
)
def test_superstable_parameters_and_feigenbaum_delta() -> None:
    R = cont.superstable_parameters(max_n=9, scan=400001)
    assert R.size == 9, f"应当有 9 个超稳定参数，得到 {R.size}"
    gaps = np.abs(R[: len(_R_SUPERSTABLE_LIT)] - np.asarray(_R_SUPERSTABLE_LIT))
    record(
        label="前六个超稳定参数与文献值的最大偏差",
        measured=float(np.max(gaps)),
        tol=1e-9,
        passed=float(np.max(gaps)) < 1e-9,
        detail="实测 " + "、".join(f"{v:.12f}" for v in R[:6]),
    )
    assert float(np.max(gaps)) < 1e-9
    record(
        label="R₂ 与 1+√5 的偏差",
        measured=float(abs(R[1] - (1.0 + math.sqrt(5.0)))),
        tol=0.0,
        passed=bool(R[1] == 1.0 + math.sqrt(5.0)),
        detail=f"1+√5={1.0 + math.sqrt(5.0):.15f}（二分 90 次后逐位相符）",
    )
    assert R[1] == 1.0 + math.sqrt(5.0)

    delta = cont.feigenbaum_delta(R)
    tail = float(delta[-1])
    record(
        label="δ 序列尾项与 4.6692016091 之差",
        measured=abs(tail - _DELTA_LIT),
        tol=5e-5,
        passed=abs(tail - _DELTA_LIT) < 5e-5,
        detail="δ 序列 " + "、".join(f"{v:.6f}" for v in delta),
    )
    assert abs(tail - _DELTA_LIT) < 5e-5, f"δ 尾项 {tail} 与文献值差 {abs(tail - _DELTA_LIT)}"
    # 收敛：尾项必须比序列中段更接近极限
    mid = float(delta[len(delta) // 2])
    record(
        label="δ 序列是否在收敛（尾项误差 < 中段误差）",
        measured=abs(tail - _DELTA_LIT),
        tol=abs(mid - _DELTA_LIT),
        passed=abs(tail - _DELTA_LIT) < abs(mid - _DELTA_LIT),
        detail=f"中段 δ={mid:.6f}（误差 {abs(mid - _DELTA_LIT):.2e}）",
    )
    assert abs(tail - _DELTA_LIT) < abs(mid - _DELTA_LIT)

    r_inf = cont.accumulation_point(R)
    record(
        label="外推的聚点 r∞ 与文献值 3.5699456718 之差",
        measured=abs(r_inf - _R_ACCUM_LIT),
        tol=1e-8,
        passed=abs(r_inf - _R_ACCUM_LIT) < 1e-8,
        detail=f"实测 {r_inf:.12f}（由 R_n 与相邻间隔按 δ−1 外推）",
    )
    assert abs(r_inf - _R_ACCUM_LIT) < 1e-8


# --------------------------------------------------------------------------- #
# 3. 周期检测与残差
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    theory="logistic 在 r=3.2 是 2 周期、r=3.5 是 4 周期、r=3.83/3.84 落在 3 周期窗口里、"
    "r=4 是混沌（≤32 步内不闭合）。判据是 |f^p(x)−x|≤1e-9，不是肉眼",
    chapter="3.5",
)
def test_period_detection_identifies_windows() -> None:
    cases = ((3.2, 2), (3.5, 4), (3.83, 3), (3.84, 3))
    for r, want in cases:
        p, res = cont.detect_period(dyn.logistic(r), 0.4, max_period=32, transient=2000)
        record(
            label=f"r={r} 的最小周期",
            measured=float(p),
            tol=float(want),
            passed=p == want and res < 1e-9,
            detail=f"实测 p={p}、闭合残差 {res:.3e}",
        )
        assert p == want, f"r={r} 应当是 {want} 周期，检测到 {p}"
        assert res < 1e-9
    p4, res4 = cont.detect_period(dyn.logistic(4.0), 0.4, max_period=32, transient=2000)
    record(
        label="r=4 的周期（应为 0 = 32 步内不闭合）",
        measured=float(p4),
        tol=0.0,
        passed=p4 == 0 and res4 > 1e-4,
        detail=f"最接近闭合的残差 {res4:.3e}（混沌不是「周期很长」，是根本不闭合）",
    )
    assert p4 == 0 and res4 > 1e-4


@assertion(
    "A",
    theory="周期窗口里闭合残差是舍入水平（1e-15），混沌区是 O(1e-2)：两类点必须分在对数轴两端；"
    "混沌区的残差**不能**是 0（那会把混沌画成完美闭合）",
    chapter="3.5",
)
def test_cycle_residuals_separate_windows_from_chaos() -> None:
    grid = np.linspace(2.4, 4.0, 700)
    periods, _ = cont.detect_periods(grid, max_period=32, transient=2000)
    res = cont.cycle_residuals(grid, periods, transient=2000)
    win = res[periods > 0]
    chaos = res[periods == 0]
    assert win.size > 100 and chaos.size > 100, f"窗口 {win.size} 个、混沌 {chaos.size} 个"
    record(
        label="周期窗口里的最大闭合残差",
        measured=float(np.max(win)),
        tol=1e-9,
        passed=float(np.max(win)) < 1e-9,
        detail=f"{win.size} 个参数点，最大 {np.max(win):.3e}",
    )
    assert float(np.max(win)) < 1e-9
    med = float(np.median(chaos))
    record(
        label="混沌区闭合残差的中位数",
        measured=med,
        tol=1e-3,
        passed=med > 1e-3,
        detail=(
            f"{chaos.size} 个参数点，中位 {med:.3e}、最小 {np.min(chaos):.3e}"
            "（初版把混沌区的残差一律留在 0.0，看起来'混沌也完美闭合'，这条断言就是那个 bug 的回归测试）"
        ),
    )
    assert med > 1e-3
    frac = float(np.mean(chaos > 1e-4))
    record(
        label="混沌区里残差 > 1e-4 的参数占比",
        measured=frac,
        tol=0.9,
        passed=frac > 0.9,
        detail=(
            f"{frac:.1%}；剩下的是「灰区」：残差在 1e-9 附近、刚好越过检测容差的近闭合轨道"
            "（实测最小 {:.2e}）——固定容差必然造出这样一条灰带，所以 meta 里同时给出用了多大的容差".format(
                float(np.min(chaos))
            )
        ),
    )
    assert frac > 0.9


@assertion(
    "A",
    theory="向量化的 detect_periods 与逐参数标量 detect_period 必须给出同一个周期与同一个残差"
    "（批量路径只是把参数维摊开；残差允许舍入量级的差）",
    chapter="3.5",
)
def test_vectorized_period_detection_matches_scalar() -> None:
    grid = np.linspace(2.5, 4.0, 61)
    periods, resid = cont.detect_periods(grid, max_period=32, transient=1500)
    worst_p = 0
    worst_res = 0.0
    for j, r in enumerate(grid):
        p, res = cont.detect_period(dyn.logistic(float(r)), 0.4, max_period=32, transient=1500)
        worst_p = max(worst_p, abs(p - int(periods[j])))
        worst_res = max(worst_res, abs(res - float(resid[j])))
    record(
        label="向量化 vs 标量的周期差异计数",
        measured=float(worst_p),
        tol=0.0,
        passed=worst_p == 0,
        detail=f"{grid.size} 个参数点",
    )
    assert worst_p == 0
    record(
        label="向量化 vs 标量的残差最大差",
        measured=worst_res,
        tol=1e-12,
        passed=worst_res < 1e-12,
        detail=(
            "残差是 min_p|f^p(x)−x|，两条路径从同一个 x₀ 出发；"
            "剩下的 1e-15 量级差来自标量与参数批两种求值路径的舍入（相对残差本身是 1e-13 量级）"
        ),
    )
    assert worst_res < 1e-12


@assertion(
    "A",
    theory="r=4 的超稳定不动点（x=1/2 是 r=2 的临界点）与周期窗口的存在性给出 Logistic 映射"
    "分岔结构的两个锚点：r<3 只有 1 周期、r=3.2 的乘子为 0.16",
    chapter="3.5",
)
def test_multiplier_sign_structure_of_the_cascade() -> None:
    # 1 周期在 r<3 稳定、在 r>3 不稳定
    for r, want_stable in ((2.5, True), (3.2, False)):
        m = cont.cycle_multipliers(r, 1)
        stable = bool(m.size and float(np.min(np.abs(m))) < 1.0)
        record(
            label=f"r={r} 的不动点是否稳定",
            measured=float(stable),
            tol=float(want_stable),
            passed=stable == want_stable,
            detail=f"乘子 {np.round(m, 8)}（理论 2−r={2.0 - r:.4f}）",
        )
        assert stable == want_stable
    m1 = cont.cycle_multipliers(3.2, 1)
    # r=3.2 的固定点有两个：x=0（乘子 r=3.2）与 x=1−1/r=0.6875（乘子 2−r=−1.2）。
    # x=0 恰好落在扫描网格的第一个节点上，所以"网格节点正好是根"这条路径每次都在被测。
    gap1 = float(np.min(np.abs(m1 - (2.0 - 3.2)))) if m1.size else float("nan")
    record(
        label="r=3.2 处不动点的乘子里有没有 2−r=−1.2",
        measured=gap1,
        tol=1e-12,
        passed=bool(m1.size == 2 and gap1 < 1e-12),
        detail=f"实测乘子 {np.round(m1, 12)}（含边界不动点 x=0 的 r={3.2}，共 {m1.size} 个）",
    )
    assert m1.size == 2 and gap1 < 1e-12

    m2 = cont.cycle_multipliers(3.2, 2)
    record(
        label="r=3.2 处稳定 2 周期的乘子 vs 4+2r−r²=0.16",
        measured=float(abs(np.min(np.abs(m2)) - 0.16)),
        tol=1e-12,
        passed=bool(m2.size and abs(float(np.min(np.abs(m2))) - 0.16) < 1e-12),
        detail=f"实测乘子 {np.round(m2, 12)}",
    )
    assert m2.size and abs(float(np.min(np.abs(m2))) - 0.16) < 1e-12
