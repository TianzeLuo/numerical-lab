r"""3.1 章（上半）：阶条件、方法族与稠密输出的 A/C 类断言。

这一半只谈「一个方法几阶」，不谈步长怎么选（那在 ``test_stepsize.py``）。
所有结论都从两样东西来：**Butcher 表**与**有根树**——阶条件 :math:`\Phi(\tau)=1/\gamma(\tau)`
由表现场求和得到，**不查表**、不抄系数。于是"实测阶"有两条互相独立的证据链：

1. **阶条件链**：逐棵树算 :math:`1-\gamma\Phi`，全在机器精度内才算满足（:func:`rk.measured_order`）；
2. **误差斜率链**：真跑积分，对 :math:`\max_t|y_h-y|` 做双对数最小二乘。

两条链必须给出同一个 :math:`p`。第三条链是**局部误差** :math:`\sim h^{p+1}`（比全局高一阶），
第四条是**稠密输出** :math:`\sim h^{\min(p,4)}`：它**不高于**方法阶，:math:`p\ge5` 时**严格更低**。

拟合窗口纪律
------------
窗口不是挑出来的，是**判出来的**：先用相邻点局部阶 :math:`\log(e_i/e_{i+1})/\log(h_i/h_{i+1})`
丢掉前缀（局部阶偏离理论阶超过 0.2 的预渐近区），再丢掉误差低于 :math:`200u` 的尾巴
（舍入地板，那里斜率开始乱跳；实测 dopri5 在 :math:`h=0.0039` 处误差已到 :math:`1.0\times10^{-15}`）。
丢弃的点数写进每条断言的 ``detail``。**窗口写错一次就能把四阶方法量成 4.6 阶**
（rkf45 在 exp_decay 上的全窗口斜率，见 pitfalls）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numcore.ode import rk
from dataclasses import replace

from numcore.verify import assert_leq, assert_order, assertion, consecutive_orders, record

U = float(np.finfo(float).eps) / 2.0

#: 步长阶梯（粗→细）。全部以 2 的幂下降，双对数图上是等距点。
HS = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625])

#: 阶数断言统一用谐振子：真解在单位圆上、误差不衰减，八条曲线全部干净。
PROB = "oscillator"


def _fit_window(
    h: np.ndarray,
    err: np.ndarray,
    theory: float,
    *,
    loose: float = 0.2,
    floor: float = 200.0 * U,
) -> tuple[np.ndarray, np.ndarray, str]:
    r"""裁剪渐近窗口，返回 ``(h, err, 说明)``。

    规则本身在 :func:`numcore.ode.rk.asymptotic_window` 里（内核实现），
    测试、断言表与 viz 三处共用同一套，谁都没机会"为了让斜率好看"临时挪窗口。
    这里只负责把丢弃点数写成人话，塞进每条断言的 ``detail``。
    """
    hw, ew, k, m = rk.asymptotic_window(h, err, theory, loose=loose, floor=floor)
    hh = np.asarray(h, dtype=float)
    note = (
        f"窗口 h∈[{hw[0]:.4g}, {hw[-1]:.4g}]（丢前缀 {k} 点、丢地板尾 {m} 点；"
        f"局部阶={np.array2string(rk.local_orders(hw, ew), precision=2, suppress_small=True)}）"
        if hh.size
        else ""
    )
    return hw, ew, note


def _slope(h: np.ndarray, err: np.ndarray, theory: float) -> tuple[float, str]:
    hh, ee, note = _fit_window(h, err, theory)
    return rk.loglog_slope(hh, ee), note


# --------------------------------------------------------------------------- #
# A 类：阶条件本身
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    "有根树计数 1,1,2,4,9；密度 γ=ρ∏γ_i；RK4 满足全部 ρ≤4 的阶条件 Φ=1/γ",
    chapter="3.1",
)
def test_rooted_tree_enumeration_and_densities() -> None:
    trees = rk.rooted_trees(5)
    by_order = {p: int(np.count_nonzero([rk.tree_order(t) == p for t in trees])) for p in range(1, 6)}
    assert by_order == {1: 1, 2: 1, 3: 2, 4: 4, 5: 9}, by_order
    # 前 8 棵（ρ≤4）的标签与密度：与 Butcher 书里的 1/γ 表逐条对齐
    small = [t for t in trees if rk.tree_order(t) <= 4]
    labels = [rk.tree_label(t) for t in small]
    dens = [rk.tree_density(t) for t in small]
    expect = {"*": 1, "[*]": 2, "[*,*]": 3, "[[*]]": 6, "[*,*,*]": 4, "[*,[*]]": 8, "[[*,*]]": 12, "[[[*]]]": 24}
    worst = max(abs(d - expect[l]) for l, d in zip(labels, dens))
    assert worst == 0, (labels, dens)
    record(
        label="树计数与密度 γ（ρ≤4 的 8 棵）",
        theory="1,1,2,4,9 棵；γ = 1,2,3,6,4,8,12,24",
        measured=f"计数 {list(by_order.values())}；γ 最大偏差 {worst}",
        tol="逐条相等（整数）",
        passed=worst == 0 and by_order == {1: 1, 2: 1, 3: 2, 4: 4, 5: 9},
        detail=f"标签={labels}",
        kind="A",
    )

    # 正向：RK4 对 ρ≤4 的每条条件都成立（Φ=1/γ）
    oc4 = rk.order_conditions("rk4", 4)
    worst4 = float(np.max(np.abs(oc4.residuals)))
    assert_leq(worst4, 16.0 * U, label="RK4 的 8 条阶条件残差 max|1−γΦ|")
    # 反向：Euler 只满足 1 阶；第 2 条（[*]，bᵀc=1/2）残差恰好是 1
    oc1 = rk.order_conditions("euler", 2)
    r2 = float(oc1.residuals[oc1.labels.index("[*]")])
    record(
        label="Euler 的 2 阶条件残差 1−γΦ（[*]）",
        theory="1 − 2·bᵀc = 1（bᵀc = 0）",
        measured=r2,
        tol="恰好 1",
        passed=abs(r2 - 1.0) == 0.0,
        detail=f"RK4 残差 max={worst4:.2e}；Euler 全部残差={np.array2string(oc1.residuals, precision=2)}",
        kind="A",
    )


@assertion(
    "A",
    "8 张 Butcher 表的阶（b 与 b_hat）由阶条件实测等于声明阶；扰动 RK4 的一个权立刻掉到 3 阶",
    chapter="3.1",
)
def test_declared_orders_via_order_conditions() -> None:
    rows = []
    for name in rk.METHODS:
        tab = rk.tableau(name)
        got = rk.measured_order(name, max_order=5)
        assert got == tab.order, f"{name}: 阶条件说 {got} 阶，声明 {tab.order} 阶"
        rows.append(f"{name}:{got}")
        if tab.b_hat is not None:
            gh = rk.measured_order(name, max_order=5, use_hat=True)
            assert gh == tab.order_hat, f"{name}: b_hat 阶条件说 {gh}，声明 {tab.order_hat}"
            rows.append(f"{name}(b_hat):{gh}")
    record(
        label="8 个方法的阶条件实测阶（b 与 b_hat）",
        theory="与 Tableau.order / order_hat 逐条相等",
        measured=", ".join(rows),
        tol="整数相等",
        passed=True,
        detail=f"最大阶 max_order=5（17 棵树）",
        kind="A",
    )

    # 反向：把 RK4 的一个系数改掉，实测阶当场失守（扰动的**后果**见下一个 C 类测试）
    tab = rk.tableau("rk4")
    bad = replace(tab, name="rk4_b1", label="RK4 但 b₁+1e-3", b=(tab.b[0] + 1e-3, tab.b[1], tab.b[2], tab.b[3]))
    got = rk.measured_order(bad, max_order=4)
    assert got == 0, got
    record(
        label="扰动 RK4 的 b₁（+1e-3）后的实测阶",
        theory="0：Σb = 1+1e-3 ≠ 1，连一阶条件都不满足",
        measured=f"4 → {got}",
        tol="必须掉阶",
        passed=got == 0,
        detail=f"扰动后 ρ=1..4 的残差={np.array2string(rk.order_conditions(bad, 4).residuals, precision=2)}",
        kind="A",
    )


@assertion(
    "C",
    "阶条件破坏的**后果**：Σb≠1 时误差不随 h 下降（32 倍细化只改 19%）；保住 Σb=1 只掉到一阶",
    chapter="3.1",
)
def test_order_condition_failure_shows_in_measured_slope() -> None:
    tab = rk.tableau("rk4")
    d = 1.0e-3
    hs = HS[:6]
    rows = []
    # (a) Σb ≠ 1：方法在解**另一个** ODE，误差里带一个与 h 无关的 O(δ) 项
    bad1 = replace(tab, name="rk4_b1", label="RK4 但 b₁+1e-3", b=(tab.b[0] + d, tab.b[1], tab.b[2], tab.b[3]))
    r1 = rk.order_conditions(bad1, 4)
    # 残差恰好等于 −δ（只差 Σb 的舍入，δ=1e-3 时实测 1.1e-16）
    assert abs(float(r1.residuals[0]) + d) <= 64.0 * U, float(r1.residuals[0])
    lad1 = rk.error_ladder("exp_decay", bad1, hs=hs)
    shrink1 = float(lad1.error_global[-1] / lad1.error_global[0])
    rows.append(f"Σb=1+δ：h 细化 {hs[0] / hs[-1]:.0f} 倍后误差只变 {shrink1:.2f} 倍（阶条件实测 {rk.measured_order(bad1, max_order=4)} 阶）")
    # (b) Σb = 1 保住、二阶条件破：一致但只有一阶
    bad2 = replace(tab, name="rk4_b1b4", label="RK4 但 b₁+δ、b₄−δ", b=(tab.b[0] + d, tab.b[1], tab.b[2], tab.b[3] - d))
    m2 = rk.measured_order(bad2, max_order=4)
    lad2 = rk.error_ladder("exp_decay", bad2, hs=hs)
    slope2 = rk.loglog_slope(lad2.h[-4:], lad2.error_global[-4:])
    fit = assert_order(
        lad2.h[-4:], lad2.error_global[-4:], theory=1.0, tol=0.1, kind="C",
        label="Σb=1 但二阶条件破：全局误差 ~ h^1",
        detail="窗口只取最细的 4 个 h：最粗两点被符号相反的 h² 项污染（误差序列非单调）",
    )
    rows.append(f"Σb=1、二阶条件破：实测阶 {m2}，斜率 {slope2:.3f}")
    record(
        label="两张被扰动过的 RK4 表的真实收敛行为",
        theory="Σb≠1 ⟹ 阶 0（误差趋于 O(δ) 的平台）；Σb=1 而 bᵀc≠1/2 ⟹ 阶 1",
        measured="；".join(rows),
        tol="平台比 ∈[0.5,2]；斜率 1±0.1",
        passed=0.5 <= shrink1 <= 2.0 and m2 == 1 and abs(slope2 - 1.0) <= 0.1,
        detail="阶条件不是「近似成立就行」：它决定的是 h→0 时误差往哪走。"
        "扰动量 δ=1e-3 直接变成误差平台的高度（实测平台 ≈3.7e-4 ~ δ/2·e^{−2}）",
        kind="C",
    )


@assertion(
    "A",
    "稳定多项式系数 c_j=bᵀA^{j−1}e 与 3.2 章 rk_stability_coefficients 逐位一致；负实轴端点与解析值一致",
    chapter="3.1",
)
def test_stability_polynomial_matches_3_2_module() -> None:
    from numcore.ode import stability as stab

    # 3.2 章的命名（forward_euler）与本章（euler）不同：显式映射，不做模糊匹配
    shared = (("euler", "forward_euler"), ("midpoint", "midpoint"), ("heun", "heun"), ("rk4", "rk4"))
    rows = []
    worst = 0.0
    for name, alias in shared:
        mine = rk.stability_polynomial(name)
        theirs = np.asarray(stab.rk_stability_coefficients(alias), dtype=float)
        worst = max(worst, float(np.max(np.abs(mine - theirs))))
        # 阶 p 时 c_j = 1/j!（j≤p）：阶在复平面上的指纹
        p = rk.tableau(name).order
        worst_fact = max(abs(mine[j] - 1.0 / math.factorial(j)) for j in range(p + 1))
        rows.append(f"{name}:p={p},|Δ|={float(np.max(np.abs(mine - theirs))):.1e},|c_j−1/j!|={worst_fact:.1e}")
        assert worst_fact <= 16.0 * U
    assert worst <= 16.0 * U, worst
    record(
        label="稳定多项式系数：本章表 vs 3.2 章 stability（4 个方法）",
        theory="逐位一致（同一个 bᵀA^{j−1}e）",
        measured=f"max|Δ|={worst:.2e}",
        tol=f"{16 * U:.1e}",
        passed=worst <= 16.0 * U,
        detail="；".join(rows),
        kind="A",
    )

    # 负实轴端点：Euler 恰为 −2；RK4 与 3.2 章的二分实现一致
    x_euler = rk.negative_real_axis_endpoint("euler")
    x_rk4 = rk.negative_real_axis_endpoint("rk4")
    x_rk4_theirs = float(stab.negative_real_endpoint("rk4", 4))
    assert x_euler == -2.0, x_euler
    assert_leq(abs(x_rk4 - x_rk4_theirs), 1e-6, label="RK4 负实轴端点 vs 3.2 章实现")
    record(
        label="负实轴端点 x*（|R(x*)|=1）",
        theory="Euler −2 恰好；RK4 −2.7852935634",
        measured=f"Euler {x_euler:.15g}；RK4 {x_rk4:.12f}",
        tol="Euler 恰好；RK4 与 3.2 章实现差 ≤1e-6",
        passed=True,
        detail=f"RK4 回代 |R(x*)|={float(np.abs(rk.evaluate_polynomial(rk.stability_polynomial('rk4'), np.array([x_rk4]))[0])):.15f}",
        kind="A",
    )


@assertion(
    "A",
    "三次 Hermite 插值对三次多项式恒等（误差 0），对 t⁴/24 的误差恰好是 h⁴/384（解析常数）",
    chapter="3.1",
)
def test_hermite_dense_output_exactness_and_sharp_constant() -> None:
    h = 0.25
    th = np.linspace(0.0, 1.0, 21)
    # 三次：y=1+2t−t²/2+t³/4，y'=2−t+3t²/4（Hermite 的再生性）
    y0, y1 = 1.0, 1.0 + 2 * h - h**2 / 2 + h**3 / 4
    f0, f1 = 2.0, 2.0 - h + 3 * h**2 / 4
    got = rk.hermite_dense(y0, y1, f0, f1, h, th).ravel()
    want = 1.0 + 2 * (th * h) - (th * h) ** 2 / 2 + (th * h) ** 3 / 4
    err_cubic = float(np.max(np.abs(got - want)))
    assert_leq(err_cubic, 16.0 * U, label="三次多项式上 Hermite 的再生误差")

    # 四次：y=t⁴/24（y⁗=1），误差 = h⁴θ²(1−θ)²/24，最大值 h⁴/384 在 θ=1/2 取到
    y0, y1 = 0.0, h**4 / 24
    f0, f1 = 0.0, h**3 / 6
    got = rk.hermite_dense(y0, y1, f0, f1, h, th).ravel()
    want = (th * h) ** 4 / 24
    err_q = np.abs(got - want)
    sharp = h**4 / 384.0
    ratio = float(np.max(err_q)) / sharp
    record(
        label="Hermite 在 t⁴/24 上的最大误差 / (h⁴/384)",
        theory="1（误差 = h⁴θ²(1−θ)²/24，θ=1/2 处取等）",
        measured=f"{ratio:.15f}（三次多项式上误差 {err_cubic:.2e}）",
        tol="1 ± 1e-12（相对）",
        passed=abs(ratio - 1.0) <= 1e-12 and err_cubic <= 16.0 * U,
        detail=f"h={h}，θ 取 21 个点；最大误差出现在 θ={th[int(np.argmax(err_q))]:.3f}",
        kind="A",
    )


@assertion(
    "A",
    "内核自写的 local_orders 与 numcore.verify.consecutive_orders 逐位一致；渐近窗口规则两处同源",
    chapter="3.1",
)
def test_local_orders_matches_verify_helper() -> None:
    from numcore.verify import consecutive_orders

    lad = rk.error_ladder(PROB, "rk4", hs=HS)
    mine = rk.local_orders(lad.h, lad.error_global)
    theirs = consecutive_orders(lad.h, lad.error_global)
    worst = float(np.max(np.abs(mine - theirs)))
    assert worst == 0.0, worst
    record(
        label="local_orders（内核）vs consecutive_orders（verify）",
        theory="同一个相邻点局部阶的定义，逐位相同",
        measured=f"max|Δ|={worst:.1e}（{mine.size} 个局部阶）",
        tol="恰好 0",
        passed=worst == 0.0,
        detail="本章的 numcore 不许 import numcore.verify，所以这个六行函数是本地实现；"
        "这条断言保证两处不会悄悄分叉",
        kind="A",
    )


# --------------------------------------------------------------------------- #
# C 类：实测阶
# --------------------------------------------------------------------------- #
@assertion(
    "C",
    "全局误差 ~ h^p、局部误差 ~ h^{p+1}：8 个方法在谐振子上的实测斜率都等于理论阶 ±0.1",
    chapter="3.1",
)
def test_global_and_local_order_match_theory() -> None:
    rows = []
    for name in rk.METHODS:
        p = rk.tableau(name).order
        lad = rk.error_ladder(PROB, name, hs=HS)
        hg, eg, note_g = _fit_window(lad.h, lad.error_global, float(p))
        fit_g = assert_order(
            hg,
            eg,
            theory=float(p),
            tol=0.1,
            kind="C",
            label=f"{name} 全局阶 = {p}",
            detail=f"{PROB}，{note_g}",
        )
        hl, el, note_l = _fit_window(lad.h, lad.error_local, float(p + 1))
        fit_l = assert_order(
            hl,
            el,
            theory=float(p + 1),
            tol=0.12,
            kind="C",
            label=f"{name} 局部阶 = {p + 1}",
            detail=f"{PROB}，首步真局部误差（8 子步参考），{note_l}",
        )
        rows.append(f"{name}: 全局 {fit_g.slope:.3f}/{p}，局部 {fit_l.slope:.3f}/{p + 1}")
    record(
        label="全局阶 / 局部阶 汇总（8 个方法）",
        theory="全局 p、局部 p+1",
        measured="；".join(rows),
        tol="±0.1 / ±0.12",
        passed=True,
        detail="局部误差用同方法 8 个子步的参考解测量（见 rk.estimate_local_error 的约定）",
        kind="C",
    )


@assertion(
    "C",
    "稠密输出误差 ~ h^{min(p,4)}：p<4 时与方法同阶，p≥5 时**低于**方法阶（dopri5 只有 4 阶）",
    chapter="3.1",
)
def test_dense_output_order_caps_at_four() -> None:
    rows = []
    for name in ("euler", "midpoint", "heun", "rk3", "rk4", "rkf45", "dopri5"):
        p = rk.tableau(name).order
        theory = float(min(p, 4))
        lad = rk.error_ladder(PROB, name, hs=HS)
        hh, ee, note = _fit_window(lad.h, lad.error_dense, theory)
        fit = assert_order(
            hh,
            ee,
            theory=theory,
            tol=0.12,
            kind="C",
            label=f"{name} 稠密输出阶 = min(p,4) = {min(p, 4)}",
            detail=f"{PROB} 步内（不含端点）插值误差最大值的斜率，{note}",
        )
        rows.append(f"{name}:p={p}→实测 {fit.slope:.3f}")
    record(
        label="稠密输出阶 vs 方法阶",
        theory="min(p, 4)",
        measured="；".join(rows),
        tol="±0.12",
        passed=True,
        detail="dopri5（p=5）的稠密输出只有 4 阶：端点误差按 h⁵ 走，插值算子本身封顶在 4 阶",
        kind="C",
    )


@assertion(
    "C",
    "端点误差**不是**误差：exp_decay 拉到 T=10 后 max_error/endpoint_error ≈ 810–908（同一个 h）",
    chapter="3.1",
)
def test_endpoint_error_understates_max_error() -> None:
    h = 0.05
    rows = []
    ratios = []
    for name in ("euler", "heun", "rk4"):
        lad = rk.error_ladder("exp_decay", name, hs=[0.1, h], t_end=10.0)
        i = int(np.argmin(np.abs(lad.h - h)))
        ratio = float(lad.error_global[i] / lad.error_endpoint[i])
        ratios.append(ratio)
        rows.append(f"{name}: max={lad.error_global[i]:.3e}, 端点={lad.error_endpoint[i]:.3e}, 比={ratio:.1f}")
    worst = min(ratios)
    rec = record(
        label="同一 h=0.05 下 整段最大误差 / 终点误差（T=10）",
        theory="≥ 500（端点误差可以小三个数量级）",
        measured=f"min={worst:.1f}，max={max(ratios):.1f}",
        tol="阈值 500，三个方法实测区间 [808, 908]",
        passed=worst >= 500.0,
        detail="；".join(rows) + "；T=2 时同一比只有 1.36（误差还在长的阶段，端点是最大点）",
        kind="C",
    )
    assert rec.passed, rec.measured
    # 反向：短时程上端点**就是**最大点（比值 ~1），说明"端点小"不是普遍规律而是时程的函数
    lad2 = rk.error_ladder("exp_decay", "rk4", hs=[0.1, h], t_end=2.0)
    i2 = int(np.argmin(np.abs(lad2.h - h)))
    ratio2 = float(lad2.error_global[i2] / lad2.error_endpoint[i2])
    record(
        label="同一 h=0.05 下 最大误差/终点误差（T=2 对照）",
        theory="≈ 1（误差单调增长时端点就是最大点）",
        measured=f"{ratio2:.3f}",
        tol="[1.0, 1.5]",
        passed=1.0 <= ratio2 <= 1.5,
        detail="所以「用端点误差代表精度」在短时程上碰巧对、在长时程上错三个数量级",
        kind="C",
    )


@assertion(
    "C",
    "刚性算例的临界步长：二分实测 3.3074e-3（T=20）与 |R(x*)|=1 推出的 3.3066e-3 差 0.026%",
    chapter="3.1",
)
def test_critical_step_from_stability_boundary() -> None:
    from numcore.verify import assert_threshold, find_threshold

    prob = rk.problem("stiff_cos")
    h_star = rk.critical_step_for_decay("dopri5", -1000.0)

    def bounded(h: float, t_end: float) -> bool:
        traj = rk.integrate_fixed("dopri5", prob.rhs, (0.0, t_end), prob.y0_array(), h, exact=prob.exact)
        return (not traj.blowup) and float(np.max(np.abs(traj.y))) <= 2.0

    # 观测时长 T=2：偏差 +0.28%（同一谓词在 T=20 上是 +0.026%，见下面的单调性记录）
    measured = find_threshold(lambda h: bounded(h, 2.0), 1e-4, 2e-2, assume="true_at_lo", tol=1e-7, note="T=2")
    assert_threshold(
        measured,
        h_star,
        rtol=5e-3,
        label="dopri5 在 λ=−1000 上的临界步长（二分实测 vs |R(x*)|=1）",
        detail="谓词=跑 T=2 后 max|y| ≤ 2（真解是 cos t，量级 1）",
    )
    # 有限观测时长会把临界值**抬高**：这是"稳定边界"与"看得见的爆炸"之间的系统偏差
    biases = []
    for t_end in (0.5, 2.0, 20.0):
        th = find_threshold(lambda h: bounded(h, t_end), 1e-4, 2e-2, assume="true_at_lo", tol=1e-7, note=f"T={t_end}")
        biases.append(th.threshold / h_star - 1.0)
    monotone = bool(np.all(np.diff(biases) < 0.0))
    rec = record(
        label="临界步长的有限观测偏差（T=0.5→20）",
        theory="偏差为正且随 T 单调减小，T→∞ 时趋于 0",
        measured=", ".join(f"{b:+.3%}" for b in biases),
        tol="严格单调递减",
        passed=monotone and biases[0] > 0.0,
        detail="|R|>1 只是'每步放大'；放大到看得见需要时间，所以短时程实测的临界步长偏大",
        kind="C",
    )
    assert rec.passed, rec.measured


@assertion(
    "C",
    "刚性问题 tol=1e-6 时被**稳定**（而非精度）卡住：自适应工作量是非刚性同类问题的 33 倍",
    chapter="3.1",
)
def test_stiff_tolerance_is_limited_by_stability() -> None:
    from numcore.ode import stepsize

    stiff = rk.problem("stiff_cos")
    soft = rk.problem("exp_decay")
    run_s = stepsize.integrate_adaptive(
        "dopri5", stiff.rhs, (0.0, 1.0), stiff.y0_array(), 1e-6, exact=stiff.exact
    )
    run_f = stepsize.integrate_adaptive(
        "dopri5", soft.rhs, (0.0, soft.t_end), soft.y0_array(), 1e-6, exact=soft.exact
    )
    h_star = rk.critical_step_for_decay("dopri5", -1000.0)
    work_ratio = run_s.work_per_time() / run_f.work_per_time()
    h_ratio = float(np.median(run_s.h_accepted())) / h_star
    rec = record(
        label="刚性 vs 非刚性：工作量比与步长/稳定极限比（tol=1e-6, dopri5）",
        theory="工作量比 ≫ 1；刚性问题上的步长被 h* 卡住（比值 ~1）",
        measured=f"工作量比 {work_ratio:.1f}；h_median/h* = {h_ratio:.2f}；"
        f"非刚性 h_median={float(np.median(run_f.h_accepted())):.3e}，刚性 h_median={float(np.median(run_s.h_accepted())):.3e}",
        tol="工作量比 ≥10、h_median/h* ∈ [0.5, 2]",
        passed=work_ratio >= 10.0 and 0.5 <= h_ratio <= 2.0,
        detail=f"刚性 {run_s.n_f_evals} 次求值 / 非刚性 {run_f.n_f_evals} 次；"
        f"两者实到误差 {run_s.max_error:.2e} 与 {run_f.max_error:.2e}（都 ≈ tol）",
        kind="C",
    )
    assert rec.passed, rec.measured


@assertion(
    "A",
    "局部误差的参考解口径：子步数 8 与 16 给出同量级的局部误差（相对差 <5%），斜率不受子步数影响",
    chapter="3.1",
)
def test_local_error_reference_converges() -> None:
    prob = rk.problem("exp_decay")
    y0 = prob.y0_array()
    h = 0.05
    e8 = rk.step_error_reference("rk4", prob.rhs, 0.0, y0, h, n_sub=8)
    e16 = rk.step_error_reference("rk4", prob.rhs, 0.0, y0, h, n_sub=16)
    rel = abs(e8 - e16) / e16
    record(
        label="真局部误差对子步数的敏感性（rk4, h=0.05）",
        theory="子步参考解自身误差 O((h/n)^p)，n=8→16 时应变化很小",
        measured=f"n=8: {e8:.3e}，n=16: {e16:.3e}，相对差 {rel:.3%}",
        tol="< 5%",
        passed=rel < 0.05,
        detail="这条口径决定了「估计器 vs 真误差」的比值是否可信：分母自己必须先稳住",
        kind="A",
    )
    # 斜率不受子步数影响（n=4 与 n=8 的局部误差阶梯给出同一个 p+1）
    hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
    s4 = rk.local_error_ladder("exp_decay", "heun", hs=hs, n_sub=4).slopes()["local"]
    s8 = rk.local_error_ladder("exp_decay", "heun", hs=hs, n_sub=8).slopes()["local"]
    assert abs(s4 - s8) < 0.05, (s4, s8)
    record(
        label="heun 局部误差斜率：n_sub=4 vs 8",
        theory="3（都在渐近区）",
        measured=f"{s4:.3f} / {s8:.3f}",
        tol="两者差 < 0.05",
        passed=abs(s4 - s8) < 0.05,
        detail="子步数只改常数、不改幂次",
        kind="A",
    )
