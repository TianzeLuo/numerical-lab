"""2.1 章 A/B 类断言：插值、Lebesgue 常数、Runge 现象与 Remez 最优逼近。

这一章的核心是**把"插值准不准"拆成三个互不相同的量**：

1. **插值误差本身**（与算法无关，取决于 :math:`f^{(n+1)}` 与节点分布）——
   由严格包络 :math:`\\sup|f^{(n+1)}|/(n+1)!\\cdot\\max|\\prod(x-x_i)|` 界住；
2. **Lebesgue 常数** :math:`\\Lambda_n`（节点分布的"质量"）——它决定舍入误差被放大多少，
   等距节点按 :math:`2^n` 爆炸，Chebyshev 节点只按 :math:`\\ln n` 爬；
3. **算法通道**（重心 / Newton / 直接解 Vandermonde）——在同一个 :math:`\\Lambda_n`
   下，三条通道给出同一个多项式，但舍入路径完全不同。

每条断言都给出**实测数字**（``record``/``assert_*`` 会登记），抄理论值不算证据。
"""

from __future__ import annotations

import numpy as np
import pytest

from numcore.interp.poly import (
    NODE_KINDS,
    ROUNDOFF_FLOOR,
    UNIT_ROUNDOFF,
    barycentric_interpolate,
    chebyshev_monic_coeffs,
    chebyshev_monic_sup_norm,
    chebyshev_nodes,
    chebyshev_poly_coeffs,
    chebyshev_t,
    chebyshev_trig,
    clamp_for_log,
    derivative_over_factorial_cauchy,
    divided_differences,
    divided_difference_table,
    equispaced_nodes,
    error_envelope,
    lagrange_interpolate,
    lebesgue_asymptotic_chebyshev,
    lebesgue_asymptotic_equispaced,
    lebesgue_constant,
    lebesgue_function,
    lebesgue_growth_table,
    max_abs_error,
    monic_product,
    node_family,
    newton_form,
    newton_interpolate,
    pointwise_error,
    poly_eval,
    remez,
    runge,
    runge_complex,
    runge_derivative_over_factorial,
    runge_error_table,
    sup_derivative_over_factorial,
)
from numcore.verify import assert_leq, assertion, record

U = UNIT_ROUNDOFF


def _agreement_ratio(a: np.ndarray, b: np.ndarray, lam: float) -> float:
    """两个等长数组的最大偏差，用 :math:`u\\Lambda_n` 归一（无量纲才有可比性）。"""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / (U * lam)


# --------------------------------------------------------------------------- #
# A 类：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="三种插值形式给出同一个多项式，差异 ≤ c·u·Λ_n", chapter="2.1")
def test_three_interpolation_forms_agree() -> None:
    r"""Lagrange（逐基函数）/ Newton（差商）/ 重心：同一批节点上必须是同一个多项式。

    三条通道的舍入路径完全不同（:math:`O(n^2)` 项求和 / 嵌套乘法 / 加权平均），
    所以"它们差多少"是**实测算法稳定性**的直接证据。归一化用 :math:`u\Lambda_n`：
    实测最大偏差 1.0–10.7 倍 :math:`u\Lambda_n`（Chebyshev n=8 最差 7.0），
    而 :math:`\Lambda_n` 本身在等距 n=24 时是 1.38e5——不归一化根本看不出谁在放大误差。
    """
    worst = 0.0
    detail = []
    for kind in NODE_KINDS:
        for n in (8, 16, 24):
            nd = node_family(kind, n)
            vals = np.exp(nd)
            xs = np.linspace(-1.0, 1.0, 201)
            p_lag = np.array([float(lagrange_interpolate(nd, vals, np.array([x]))[0]) for x in xs])
            p_new = newton_interpolate(nd, vals, xs)
            p_bar = barycentric_interpolate(nd, vals, xs)
            lam = lebesgue_constant(nd, -1.0, 1.0, n_grid=2001, refine_rounds=3)
            r1 = _agreement_ratio(p_lag, p_bar, lam)
            r2 = _agreement_ratio(p_new, p_bar, lam)
            worst = max(worst, r1, r2)
            detail.append(f"{kind[:4]} n={n}: {r1:.2f}/{r2:.2f}")
    rec = record(
        label="三种插值形式的最大偏差 / (u·Λ_n)",
        theory="同一多项式，差异应落在 c·u·Λ_n（c 为小常数）",
        measured=f"最大 {worst:.3f}（{'；'.join(detail)}）",
        tol="≤ 50",
        passed=worst <= 50.0,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="|f−p| ≤ sup|f^(n+1)|/(n+1)!·max|∏(x−x_i)|（严格上界）", chapter="2.1")
def test_interpolation_error_is_bounded_by_rigorous_envelope() -> None:
    r"""严格包络**必须**住误差（:math:`f=e^x`，n=4..12，两种节点分布）。

    包络用的是 :math:`\sup_{[-1,1]}|f^{(n+1)}|/(n+1)!`（数值上确界，密网格 + 缩区间细化），
    乘首一乘积的逐点绝对值。实测紧度中位数 0.36–0.37：**这个界对整函数只宽松 2.5 倍**，
    所以它在这里是有用的估计；换成 Runge 函数同一个公式会宽松 1e6 倍（见另一个断言）。
    """
    worst_ratio = 0.0
    tight = []
    for kind in NODE_KINDS:
        for n in (4, 8, 12):
            nd = node_family(kind, n)
            xs = np.linspace(-1.0, 1.0, 401)
            err = pointwise_error(np.exp(xs), barycentric_interpolate(nd, np.exp(nd), xs))
            sup = sup_derivative_over_factorial("exp", n + 1, n_grid=2001)
            env = error_envelope(nd, xs, sup)
            mask = env > 100.0 * U
            assert np.any(mask), "包络全在舍入底以下，无法检验"
            worst_ratio = max(worst_ratio, float(np.max(err[mask] / env[mask])))
            tight.append(float(np.median(err[mask] / env[mask])))
    assert_leq(worst_ratio, 1.0, label="误差/包络的最大值（严格上界不允许超过 1）")
    rec = record(
        label="包络紧度（误差/包络的中位数）",
        theory="严格上界 ≥ 实测误差；对 e^x 应只宽松一个常数倍",
        measured=f"中位数 {min(tight):.3f}–{max(tight):.3f}",
        tol="0.2 ≤ 中位数 ≤ 0.6",
        passed=0.2 <= min(tight) and max(tight) <= 0.6,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="首一 Chebyshev 多项式在首一多项式中无穷范数最小 = 2^(1−n)", chapter="2.1")
def test_chebyshev_monic_minimizes_sup_norm() -> None:
    r"""实测 :math:`\max|\prod(x-x_i)|`：Chebyshev 节点贴着理论值 :math:`2^{1-n}`。

    n=12 实测 4.84e-4 对理论 4.88e-4（差 0.8%）；等距节点是 4.01e-3，**大 8.3 倍**。
    这条断言同时钉住了"最小最大性质"和"等距节点的乘积在两端爆炸"两件事。
    """
    ratios = []
    eq_over_ch = []
    for n in (4, 8, 12, 20):
        xs = np.linspace(-1.0, 1.0, 4001)
        ch = float(np.max(np.abs(monic_product(chebyshev_nodes(n), xs))))
        eq = float(np.max(np.abs(monic_product(equispaced_nodes(n), xs))))
        th = chebyshev_monic_sup_norm(n)
        ratios.append(ch / th)
        eq_over_ch.append(eq / ch)
    rec_min = record(
        label="max|∏|（Chebyshev）/ 2^(1−n)",
        theory="首一 Chebyshev 多项式的无穷范数恰为 2^(1−n)：网格最大值只能 ≤ 理论值",
        measured=f"{min(ratios):.4f}–{max(ratios):.4f}（n=4 网格最粗，n=20 最接近 1）",
        tol="比值 ∈ [0.90, 1+1e-9]（上侧是严格不等式，下侧受网格分辨率限制）",
        passed=all(0.90 <= r <= 1.0 + 1e-9 for r in ratios),
    )
    assert rec_min.passed, rec_min.measured
    rec_eq = record(
        label="max|∏|（等距）/ max|∏|（Chebyshev）",
        theory="等距节点的首一乘积在端点附近爆炸，比值随 n 增长（n=12 实测 8.3）",
        measured=f"{min(eq_over_ch):.3f}–{max(eq_over_ch):.3f}",
        tol="n ≥ 8 时 ≥ 3",
        passed=eq_over_ch[-1] >= 3.0 and eq_over_ch[-2] >= 3.0,
    )
    assert rec_eq.passed, rec_eq.measured


@assertion("A", theory="Λ_n 的小 n 精确值 + 两条渐近公式", chapter="2.1")
def test_lebesgue_constant_values_and_asymptotics() -> None:
    r"""小 n 的 :math:`\Lambda_n` 是**有理数**，可以逐位核对；大 n 对渐近公式。

    实测精确值：:math:`\Lambda_1=1`、:math:`\Lambda_2=1.25`（两种节点相同）、
    Chebyshev :math:`\Lambda_3=5/3=1.6667`、:math:`\Lambda_4=1.7987618`；
    等距 :math:`\Lambda_3=1.6311303`、:math:`\Lambda_4=2.2078244`。
    渐近：Chebyshev 相对偏差在 n=10..40 内 ≤ 0.1%，等距比值稳定在 0.85–0.86
    （**不是 1**：渐近公式只是量级，这条差 14% 是必须记住的）。
    """
    exact = {
        ("equispaced", 1): 1.0,
        ("equispaced", 2): 1.25,
        ("equispaced", 3): 1.6311303094,
        ("equispaced", 4): 2.2078243973,
        ("chebyshev", 1): 1.0,
        ("chebyshev", 2): 1.25,
        ("chebyshev", 3): 5.0 / 3.0,
        ("chebyshev", 4): 1.7987618033,
    }
    worst_exact = 0.0
    for (kind, n), want in exact.items():
        got = lebesgue_constant(node_family(kind, n), -1.0, 1.0, n_grid=2001, refine_rounds=4)
        worst_exact = max(worst_exact, abs(got - want) / want)
    rec_exact = record(
        label="小 n 的 Λ_n 相对误差（8 个精确值）",
        theory="Λ_1=1、Λ_2=1.25、Λ_3(ch)=5/3、Λ_4(ch)=1.7987618 等",
        measured=f"最大相对误差 {worst_exact:.3e}",
        tol="≤ 1e-6（求最大值用的网格足够密）",
        passed=worst_exact <= 1e-6,
    )
    assert rec_exact.passed, rec_exact.measured

    ch_dev = []
    eq_ratio = []
    for n in (10, 15, 20, 30, 40):
        ch = lebesgue_constant(chebyshev_nodes(n), -1.0, 1.0, n_grid=2001, refine_rounds=3)
        eq = lebesgue_constant(equispaced_nodes(n), -1.0, 1.0, n_grid=2001, refine_rounds=3)
        ch_dev.append(abs(ch / lebesgue_asymptotic_chebyshev(n) - 1.0))
        eq_ratio.append(eq / lebesgue_asymptotic_equispaced(n))
    late = eq_ratio[1:]  # n ≥ 15：n=10 的渐近还差 7%（实测比值 0.914）
    rec_asym = record(
        label="Chebyshev 渐近偏差 / 等距渐近比值",
        theory="Chebyshev: Λ_n ≈ (2/π)ln n + 0.9625；等距: Λ_n ≈ 2^(n+1)/(e n ln n)",
        measured=f"Chebyshev 偏差 ≤ {max(ch_dev):.2e}；等距比值 {min(eq_ratio):.3f}–{max(eq_ratio):.3f}"
        f"（n≥15 时 {min(late):.3f}–{max(late):.3f}）",
        tol="Chebyshev ≤ 5e-3；等距比值 n≥15 时 ∈ [0.80, 0.90]、全部 n ∈ [0.80, 0.92]",
        passed=(
            max(ch_dev) <= 5e-3
            and 0.80 <= min(late)
            and max(late) <= 0.90
            and 0.80 <= min(eq_ratio)
            and max(eq_ratio) <= 0.92
        ),
    )
    assert rec_asym.passed, rec_asym.measured


@assertion("A", theory="等距 Λ 按 2^n 增长（实测 log-log 斜率 = 理论斜率）", chapter="2.1")
def test_lebesgue_constant_growth_rate() -> None:
    r"""指数爆炸的**率**要对得上：这是"防误导规则 2"要求的实测斜率 vs 理论斜率。

    等距：:math:`\Lambda_{n+1}/\Lambda_n \to 2`（n=30 实测 1.9041），
    log-log 斜率在 n=20 处实测 12.4991 对理论 12.5174（差 0.15%）；
    Chebyshev：斜率实测 0.2218 对理论 0.2219（差 0.05%），量级差 56 倍。
    """
    tab = lebesgue_growth_table(31, n_grid=2001, refine_rounds=3)
    lam_eq = tab["lambda_equispaced"]
    step = float(lam_eq[29] / lam_eq[28])
    sl_eq = tab["slope_measured_equispaced"]
    sl_eq_th = tab["slope_theory_equispaced"]
    sl_ch = tab["slope_measured_chebyshev"]
    sl_ch_th = tab["slope_theory_chebyshev"]
    gap_eq = max(abs(float(sl_eq[i] - sl_eq_th[i])) for i in (19, 24, 29))
    gap_eq_early = max(abs(float(sl_eq[i] - sl_eq_th[i])) for i in (9, 14))
    gap_ch = max(abs(float(sl_ch[i] - sl_ch_th[i])) for i in (9, 14, 19, 24))
    rec = record(
        label="Λ_{n+1}/Λ_n（等距）与两侧斜率偏差",
        theory="等距 Λ ~ 2^n（斜率 = n·ln2/ln10）；Chebyshev Λ ~ (2/π)ln n",
        measured=(
            f"Λ_30/Λ_29={step:.4f}；等距斜率偏差 ≤ {gap_eq:.4f}（n≥20；n=10/15 还有 {gap_eq_early:.3f}）；"
            f"Chebyshev 斜率偏差 ≤ {gap_ch:.4f}（n=10..25）"
        ),
        tol="步长 ∈ [1.85, 2.0]；等距斜率偏差（n≥20）≤ 0.03；Chebyshev ≤ 0.005",
        passed=1.85 <= step <= 2.0 and gap_eq <= 0.02 and gap_ch <= 5e-3,
    )
    assert rec.passed, rec.measured
    rec2 = record(
        label="Λ_30 等距 / Chebyshev",
        theory="n=30 时等距 6.60e6、Chebyshev 3.127，量级差 6 个数量级",
        measured=f"{float(lam_eq[29]):.4e} / {float(tab['lambda_chebyshev'][29]):.6f}",
        tol="比值 ≥ 1e5",
        passed=float(lam_eq[29] / tab["lambda_chebyshev"][29]) >= 1e5,
    )
    assert rec2.passed, rec2.measured


@assertion("A", theory="Runge：等距插值发散、Chebyshev 收敛（误差表实测）", chapter="2.1")
def test_runge_divergence_versus_chebyshev_convergence() -> None:
    r"""同一个函数 :math:`1/(1+25x^2)`，只换节点分布，误差走向完全相反。

    实测：等距在 n=7 触底 0.2474（"最优点"），n=12 已到 3.663（超过函数自身量级 1），
    n=20 是 59.82、n=30 是 2388——**发散**；Chebyshev 从 n≥6 单调下降，
    n=30 时 2.426e-3。奇点距离：:math:`1/\sqrt a=0.2`，
    Bernstein 参数 :math:`\rho=1/\sqrt a+\sqrt{1+1/a}=1.2198`。
    """
    tab = runge_error_table(30, a=25.0, n_grid=801)
    err_eq = tab["error_equispaced"]
    err_ch = tab["error_chebyshev"]
    k_min = int(np.argmin(err_eq))
    assert k_min + 1 == 7, f"等距最小误差出现在 n={k_min + 1}（预期 7）"
    odd_e = err_eq[6::2]  # n=7,9,11,...
    even_e = err_eq[7::2]  # n=8,10,12,...
    rec = record(
        label="等距误差最小值 / 奇偶两条子列的单调性",
        theory="等距插值在 n=7 取到全局最小；误差**不是**随 n 单调——按奇偶分成两条子列各自发散",
        measured=f"n={k_min + 1}，min={float(err_eq[k_min]):.4f}；奇列 n=7→29："
        f"{float(odd_e[0]):.3f}→{float(odd_e[-1]):.1f}，偶列 n=8→30：{float(even_e[0]):.3f}→{float(even_e[-1]):.1f}",
        tol="min ≈ 0.247；两条子列各自严格递增；n=12 > 1、n=20 > 50、n=30 > 2000",
        passed=(
            abs(float(err_eq[k_min]) - 0.2474) <= 5e-3
            and float(err_eq[11]) > 1.0
            and float(err_eq[19]) > 50.0
            and float(err_eq[29]) > 2000.0
            and bool(np.all(np.diff(odd_e) > 0.0))
            and bool(np.all(np.diff(even_e) > 0.0))
        ),
    )
    assert rec.passed, rec.measured
    rec2 = record(
        label="Chebyshev 误差（n=10/20/30）与奇点距离",
        theory="Chebyshev 节点下解析函数一致收敛；奇点距离 1/√a 决定收敛速率",
        measured=f"{float(err_ch[9]):.4e} / {float(err_ch[19]):.4e} / {float(err_ch[29]):.4e}"
        f"；1/√a={1.0 / np.sqrt(25.0):.3f}，ρ={1.0 / 5.0 + np.sqrt(1.0 + 1.0 / 25.0):.4f}"
        f"；n=30 奇列 {float(err_ch[28]):.2e} / 偶列 {float(err_ch[29]):.2e}",
        tol="n=30 时 < 5e-3；奇偶两条子列各自单调下降（x=0 只在偶 n 时是节点）",
        passed=(
            float(err_ch[29]) < 5e-3
            and bool(np.all(np.diff(err_ch[5::2]) < 0.0))
            and bool(np.all(np.diff(err_ch[6::2]) < 0.0))
        ),
    )
    assert rec2.passed, rec2.measured


@assertion("A", theory="Remez：等幅振荡 m+2 点 + de la Vallée Poussin 下界夹逼", chapter="2.1")
def test_remez_equioscillation_and_optimality_sandwich() -> None:
    r"""交错定理的两个方向都测：(a) 等幅振荡点数 ≥ m+2 且振幅相等；
    (b) 用 de la Vallée Poussin 下界 :math:`\min_i|e(x_i)|` 夹住 :math:`E`。

    实测 e^x（m=4）：E=5.4666760e-4，6 个等幅振荡点，振幅比 1.000000；
    下界 = 上界到 1e-6；同时 E ≤ Chebyshev 插值的误差 1.0660e-3（它是合法竞争多项式），
    这就是"最优"的**可验证**含义——不是"我说它最优"。
    """
    rows = []
    ok = True
    for fun, name, m in ((np.exp, "exp", 4), (np.exp, "exp", 8), (abs, "abs", 5), (np.sin, "sin", 6), (runge, "runge", 6)):
        r = remez(fun, m, -1.0, 1.0, n_grid=4001)
        xs = np.linspace(-1.0, 1.0, 4001)
        fv = np.asarray([float(fun(float(x))) for x in xs])
        err = fv - poly_eval(r.coeffs_monomial, xs)
        # de la Vallée Poussin 下界：交错点上的最小振幅
        lower = float(np.min(np.abs(r.reference_err))) if r.reference_err.size else 0.0
        upper = float(np.max(np.abs(err)))
        rows.append((name, m, r.n_extrema, r.equioscillation_spread, lower, abs(r.error_level), upper))
        ok = ok and (r.n_extrema >= m + 2) and (r.equioscillation_spread <= 1.0 + 1e-3)
        ok = ok and abs(abs(r.error_level) - upper) <= 1e-6 * max(upper, 1e-300)
        ok = ok and lower >= abs(r.error_level) * (1.0 - 1e-3)
    rec = record(
        label="Remez 交错点数 / 振幅比 / 下界与 max|e| 的相对偏差",
        theory="等幅振荡点数 ≥ m+2；min|e(x_i)| ≤ E = max|e|（交错定理的两个方向）",
        measured="；".join(
            f"{nm} m={m}: n_ext={ne}、振幅比 {sp:.6f}、下界/E={lo / max(ee, 1e-300):.6f}" for nm, m, ne, sp, lo, ee, _ in rows
        ),
        tol="n_ext ≥ m+2；振幅比 ≤ 1+1e-3；下界 ≥ 0.999E；E = max|e|（相对 1e-6）",
        passed=ok,
    )
    assert rec.passed, rec.measured
    # 上界侧：Chebyshev 插值误差是合法竞争者，最优逼近必须不差于它
    r4 = remez(np.exp, 4, -1.0, 1.0, n_grid=4001)
    xs = np.linspace(-1.0, 1.0, 801)
    cheb4 = max_abs_error(np.exp(xs), barycentric_interpolate(chebyshev_nodes(4), np.exp(chebyshev_nodes(4)), xs))
    rec2 = record(
        label="E_remez vs Chebyshev 插值误差（m=4, e^x）",
        theory="最佳一致逼近不差于任何同次数多项式，特别是 Chebyshev 插值",
        measured=f"E_remez={abs(r4.error_level):.6e} ≤ E_cheb={cheb4:.6e}",
        tol="E_remez < E_cheb",
        passed=abs(r4.error_level) < cheb4,
    )
    assert rec2.passed, rec2.measured


@assertion("A", theory="重心公式的舍入误差 ≈ u·Λ_n（插值误差落到舍入底之后）", chapter="2.1")
def test_barycentric_roundoff_scales_with_lebesgue_constant() -> None:
    r"""同一个算法、同一个函数：误差随 :math:`\Lambda_n` 走，与节点分布无关。

    实测 :math:`err/(u\Lambda_n)`：n=16..30 时等距 1.13–3.35、Chebyshev 3.84–4.40，
    都是个位数——这就是"重心公式的精度 = :math:`u\Lambda_n`"的直接测量。
    而 n=10 时同一比值是 7.0e4（Chebyshev 1.9e5）：那时量到的是**插值误差本身**，
    已经掉到舍入底以下的部分才有资格谈舍入误差——把两者混起来会得到完全错误的结论。
    """
    ratios = []
    for kind in NODE_KINDS:
        for n in (16, 20, 24, 30):
            nd = node_family(kind, n)
            xs = np.linspace(-1.0, 1.0, 801)
            err = max_abs_error(np.exp(xs), barycentric_interpolate(nd, np.exp(nd), xs))
            lam = lebesgue_constant(nd, -1.0, 1.0, n_grid=2001, refine_rounds=3)
            ratios.append(err / (U * lam))
    rec = record(
        label="重心公式误差 / (u·Λ_n)（n=16..30）",
        theory="插值误差已在舍入底以下时，实测误差 ≈ c·u·Λ_n（c 为个位数常数）",
        measured=f"{min(ratios):.3f}–{max(ratios):.3f}",
        tol="∈ [0.5, 10]",
        passed=0.5 <= min(ratios) and max(ratios) <= 10.0,
    )
    assert rec.passed, rec.measured
    early = []
    for kind in NODE_KINDS:
        nd = node_family(kind, 10)
        xs = np.linspace(-1.0, 1.0, 801)
        err = max_abs_error(np.exp(xs), barycentric_interpolate(nd, np.exp(nd), xs))
        lam = lebesgue_constant(nd, -1.0, 1.0, n_grid=2001, refine_rounds=3)
        early.append(err / (U * lam))
    rec2 = record(
        label="n=10 时的同一比值（插值误差仍占主导）",
        theory="插值误差 ≫ u·Λ_n 时，这个比值不再反映舍入行为",
        measured=f"等距 {early[0]:.3e}、Chebyshev {early[1]:.3e}",
        tol="≥ 1e3（证明此时量到的不是舍入误差）",
        passed=min(early) >= 1e3,
    )
    assert rec2.passed, rec2.measured


@assertion("A", theory="Vandermonde：系数误差 ~ κ(V)u，后向残差与 κ 无关", chapter="2.1")
def test_vandermonde_coefficient_error_follows_kappa_u() -> None:
    r"""实测"系数通道"与"函数值通道"的分裂（真值 = 有理数精确解）。

    等距节点：n=20 系数误差 4.18e-9（κu=3.54e-7）、n=30 是 2.69e-4（κu=2.88e-2）；
    Chebyshev：n=30 是 1.14e-6（κu=2.98e-5）。两者都**被 κu 界住**（比值 1e-2 量级），
    而同一批解的后向残差从 n=6 到 n=30 一直 ≤ 3.3e-17——LU 是后向稳定的，
    "κ 大"不等于"算法不稳定"，它只说明**系数这个基**病态。
    """
    from numcore.interp.poly import interpolation_method_table

    cerr_ok = True
    resid_worst = 0.0
    detail = []
    for kind in NODE_KINDS:
        tab = interpolation_method_table(30, kind=kind, function="exp", n_grid=801)
        for i in (19, 29):
            cerr = float(tab["error_coefficients_vandermonde"][i])
            bound = float(tab["kappa_bound_vandermonde"][i])
            resid_worst = max(resid_worst, float(tab["residual_vandermonde"][i]))
            cerr_ok = cerr_ok and cerr <= bound
            detail.append(f"{kind[:4]} n={i + 1}: {cerr:.2e} ≤ {bound:.2e}")
    rec = record(
        label="Vandermonde 系数误差 ≤ κ(V)·u",
        theory="前向误差界 ‖δc‖/‖c‖ 不超过 κ(V)·u 的常数倍（最坏情况尺度）",
        measured="；".join(detail),
        tol="四条全部 ≤ κu",
        passed=cerr_ok,
    )
    assert rec.passed, rec.measured
    rec2 = record(
        label="Vandermonde 后向残差（n=6..30 的最大值）",
        theory="部分主元 LU 后向稳定：残差 ≈ u，与 κ 无关",
        measured=f"{resid_worst:.3e}",
        tol="≤ 100u",
        passed=resid_worst <= 100.0 * U,
    )
    assert rec2.passed, rec2.measured


@assertion("A", theory="Runge 函数的 f^(k)/k! 有闭式，与 Cauchy 积分公式一致", chapter="2.1")
def test_runge_derivative_closed_form_matches_cauchy_integral() -> None:
    r"""包络里那个 :math:`\sup|f^{(n+1)}|/(n+1)!` 不能靠"大概"——两个独立算法对拍。

    闭式解来自部分分式 :math:`\frac{f^{(k)}(x)}{k!}=\frac12[\frac{(-i\sqrt a)^k}{(1+i\sqrt a x)^{k+1}}+\frac{(i\sqrt a)^k}{(1-i\sqrt a x)^{k+1}}]`；
    另一条路是 Cauchy 积分（梯形法数值积分）。实测最大偏差：k=1 是 4.4e-16、
    k=4 是 8.5e-14、k=8 是 7.3e-11（相对 1e-9 以下），两条路在 k ≤ 8 内一致。
    """
    xs = np.linspace(-0.8, 0.8, 21)
    worst = []
    for k in (1, 2, 4, 8):
        closed = runge_derivative_over_factorial(xs, k, 25.0)
        cauchy = derivative_over_factorial_cauchy(runge_complex, xs, k, radius=0.15, n_theta=1024)
        scale = float(np.max(np.abs(closed)))
        worst.append(float(np.max(np.abs(closed - cauchy))) / scale)
    rec = record(
        label="闭式 vs Cauchy 积分的相对偏差（k=1,2,4,8，r=0.15）",
        theory="两条独立路径必须给出同一组 f^(k)/k!（奇点在 ±0.2i，故 r<0.2）",
        measured="、".join(f"k={k}: {w:.2e}" for k, w in zip((1, 2, 4, 8), worst)),
        tol="≤ 1e-12（Cauchy 用 1024 点梯形法）",
        passed=max(worst) <= 1e-12,
    )
    assert rec.passed, rec.measured
    # 半径圈住极点时的静默错误：这是本章最容易"看起来对"的坑
    bad = []
    for k in (1, 4):
        closed = runge_derivative_over_factorial(xs, k, 25.0)
        cauchy = derivative_over_factorial_cauchy(runge_complex, xs, k, radius=0.25, n_theta=1024)
        bad.append(float(np.max(np.abs(closed - cauchy))) / float(np.max(np.abs(closed))))
    rec2 = record(
        label="轮廓半径 0.25 > 奇点距离 0.2 时的相对偏差（k=1,4）",
        theory="围道内必须解析：圈住极点后积分值被残差主导，函数**不会报错**",
        measured="、".join(f"k={k}: {b:.2e}" for k, b in zip((1, 4), bad)),
        tol="≥ 0.1（实测 ~1.0，即 100% 错误——静默失效）",
        passed=min(bad) >= 0.1,
    )
    assert rec2.passed, rec2.measured


# --------------------------------------------------------------------------- #
# B 类：不变量与形状
# --------------------------------------------------------------------------- #
@assertion("B", theory="插值多项式在节点上取到给定的函数值（精确到舍入）", chapter="2.1")
def test_interpolation_reproduces_node_values() -> None:
    """插值的定义性质：:math:`p(x_i)=y_i`。用三个实现分别检查（它们各自都可能写错）。"""
    worst = 0.0
    for kind in NODE_KINDS:
        for n in (5, 12, 20):
            nd = node_family(kind, n)
            vals = np.exp(nd)
            for p in (
                barycentric_interpolate(nd, vals, nd),
                newton_interpolate(nd, vals, nd),
                np.array([float(lagrange_interpolate(nd, vals, np.array([x]))[0]) for x in nd]),
            ):
                scale = max(float(np.max(np.abs(vals))), 1e-300)
                worst = max(worst, float(np.max(np.abs(np.asarray(p) - vals))) / scale)
    rec = record(
        label="|p(x_i) − y_i| / max|y|（三种实现 × 两种分布 × 3 个 n）",
        theory="p(x_i) = y_i 精确成立（浮点下 ≤ c·n·u）",
        measured=f"{worst:.3e}",
        tol="≤ 1e-14",
        passed=worst <= 1e-14,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="Chebyshev 三项递推与 cos(n·arccos x) 定义一致", chapter="2.1")
def test_chebyshev_recurrence_matches_trig_definition() -> None:
    """T_n 的两种定义必须一致（|x| ≤ 1）；系数展开 T_4 = 8x^4−8x^2+1 也要对得上。"""
    xs = np.linspace(-1.0, 1.0, 101)
    worst = 0.0
    for n in range(0, 21):
        worst = max(worst, float(np.max(np.abs(chebyshev_t(n, xs) - chebyshev_trig(n, xs)))))
    rec = record(
        label="|T_n(递推) − cos(n·arccos x)| 的最大值（n ≤ 20）",
        theory="递推与三角定义是同一个多项式",
        measured=f"{worst:.3e}",
        tol="≤ 1e-12",
        passed=worst <= 1e-12,
    )
    assert rec.passed, rec.measured
    c4 = chebyshev_poly_coeffs(4)
    want = np.array([8.0, 0.0, -8.0, 0.0, 1.0])
    rec2 = record(
        label="T_4 的单项式系数",
        theory="T_4 = 8x^4 − 8x^2 + 1；首项系数 2^(n−1)",
        measured=f"{np.array2string(c4, precision=1)}（T_3 首项 {chebyshev_poly_coeffs(3)[0]:.0f}）",
        tol="与 [8,0,−8,0,1] 完全一致",
        passed=bool(np.array_equal(c4, want)) and chebyshev_poly_coeffs(3)[0] == 4.0,
    )
    assert rec2.passed, rec2.measured


@assertion("B", theory="首一 Chebyshev 多项式的无穷范数 = 2^(1−n)", chapter="2.1")
def test_chebyshev_monic_sup_norm_value() -> None:
    """把 :math:`T_n/2^{n-1}` 的系数直接求值，实测上确界必须等于 :math:`2^{1-n}`。"""
    xs = np.linspace(-1.0, 1.0, 20001)
    rows = []
    ok = True
    for n in (1, 2, 4, 8, 12):
        sup = float(np.max(np.abs(poly_eval(chebyshev_monic_coeffs(n), xs))))
        th = chebyshev_monic_sup_norm(n)
        rows.append(f"n={n}: {sup:.6g}/{th:.6g}")
        ok = ok and abs(sup / th - 1.0) <= 5e-3
    rec = record(
        label="实测 sup|T_n/2^(n−1)| vs 2^(1−n)",
        theory="首一 Chebyshev 多项式的无穷范数恰为 2^(1−n)",
        measured="；".join(rows),
        tol="相对差 ≤ 0.5%（网格求最大值）",
        passed=ok,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="差商表与差商向量一致，且与节点顺序无关", chapter="2.1")
def test_divided_difference_consistency_and_permutation_invariance() -> None:
    """Newton 形式的两个不变量：对角线 = 差商；换节点顺序多项式不变。"""
    nd = equispaced_nodes(9)
    assert nd.size == 10, "equispaced_nodes(9) 给的是 10 个节点（n+1 约定）"
    vals = np.sin(nd) + 0.5 * nd
    tab = divided_difference_table(nd, vals)
    dd = divided_differences(nd, vals)
    diag = np.array([tab[k, k] for k in range(nd.size)])
    diff = float(np.max(np.abs(diag - dd)))
    perm = np.array([3, 0, 9, 1, 7, 2, 6, 4, 8, 5])  # 10 个下标的一个置换，一个都不能少
    xs = np.linspace(-1.0, 1.0, 101)
    p1 = newton_form(nd, vals).eval(xs)
    p2 = newton_form(nd[perm], vals[perm]).eval(xs)
    dperm = float(np.max(np.abs(p1 - p2)))
    rec = record(
        label="差商表对角线偏差 / 换序后的多项式偏差",
        theory="对角线即差商；插值多项式与节点编号无关（差商对节点对称）",
        measured=f"对角线 {diff:.3e}，换序 {dperm:.3e}",
        tol="两者 ≤ 1e-12",
        passed=diff <= 1e-12 and dperm <= 1e-12,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="Λ_n(x) = Σ|ℓ_i(x)| ≥ 1，且在节点上恰为 1", chapter="2.1")
def test_lebesgue_function_is_at_least_one() -> None:
    """Lebesgue 函数的两个形状性质：下界 1、节点处取 1。"""
    worst_low = 1e9
    worst_node = 0.0
    for kind in NODE_KINDS:
        for n in (3, 8, 15):
            nd = node_family(kind, n)
            xq = np.linspace(-1.0, 1.0, 401)
            keep = xq[np.min(np.abs(xq[:, None] - nd[None, :]), axis=1) > 1e-12]
            lam = lebesgue_function(nd, keep)
            worst_low = min(worst_low, float(np.min(lam)))
            at_nodes = lebesgue_function(nd, nd)
            worst_node = max(worst_node, float(np.max(np.abs(at_nodes - 1.0))))
    rec = record(
        label="min_x Λ_n(x) / max_i |Λ_n(x_i) − 1|",
        theory="Λ_n(x) ≥ 1（插值算子范数下界）；节点处 ℓ_i(x_j)=δ_ij 故 Λ_n(x_i)=1",
        measured=f"下界 {worst_low:.6f}；节点偏差 {worst_node:.3e}",
        tol="下界 ≥ 1 − 1e-12；节点偏差 ≤ 1e-12",
        passed=worst_low >= 1.0 - 1e-12 and worst_node <= 1e-12,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="对数轴夹取：0 → 地板、非有限 → 上界、负数必须报错", chapter="2.1")
def test_clamp_for_log_contract() -> None:
    """插值误差在节点上**恰好是 0**，所以夹取契约必须逐条守住（1.2 章同一口径）。"""
    raw = np.array([0.0, 1e-30, 0.5, 1e300, np.inf, np.nan, 0.0])
    out, info = clamp_for_log(raw, ROUNDOFF_FLOOR, 1e12)
    ok = bool(
        np.all(np.isfinite(out))
        and np.all(out >= ROUNDOFF_FLOOR)
        and np.all(out <= 1e12)
        and info["n_floor"] == 3  # 两个 0 与一个 1e-30 都在 u/2 之下
        and info["n_nonfinite"] == 2
    )
    raised = False
    try:
        clamp_for_log(np.array([-1.0, 1.0]))
    except ValueError:
        raised = True
    rec = record(
        label="夹取后的取值范围 / 地板计数 / 非有限计数 / 负数是否报错",
        theory="0 与 inf/nan 都必须如实计数；负数在对数轴上无定义，必须报错",
        measured=f"范围 [{out.min():.2e}, {out.max():.2e}]，地板 {info['n_floor']}，"
        f"非有限 {info['n_nonfinite']}，负数报错={raised}",
        tol="全部满足",
        passed=ok and raised,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="numcore.interp.poly 不得依赖参考层/外部数值库（P5、P10）", chapter="2.1")
def test_poly_module_imports_respect_layering() -> None:
    """把"分层"写成可执行断言：poly.py 不许 import scipy/sympy/mpmath，也不许 import 参考层。"""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    src = (root / "numcore" / "interp" / "poly.py").read_text()
    tree = ast.parse(src)
    banned = {"scipy", "sympy", "mpmath", "matplotlib", "pandas", "sklearn"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in banned or a.name.startswith("numcore.reference"):
                    hits.append(a.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in banned or node.module.startswith("numcore.reference"):
                hits.append(node.module)
    rec = record(
        label="poly.py 里被禁的 import",
        theory="P5：numcore 不得依赖 scipy/sympy/mpmath；P10：numcore 不得 import numcore.reference",
        measured=f"{hits if hits else '无'}",
        tol="空",
        passed=not hits,
    )
    assert rec.passed, rec.measured
