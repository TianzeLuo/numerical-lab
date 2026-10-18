"""3.4 章（下半）：Chebyshev 谱方法的 A/C 类断言。

谱方法在本章要回答三件事，每一件都必须带**实测数字**，不许只说"指数收敛"：

* **收敛率**——解析解的误差在**半对数**坐标下是直线（斜率 = 每加一个节点的位数），
  而 :math:`C^s` 解是**幂律**；两种模型的判别不靠 R^2 高低（有限窗口里都能拟合得不错），
  靠"误差比 err(N)/err(N+2) 是随 N 变大还是趋于常数"。
* **地板**——指数衰减一定会撞上舍入地板 :math:`\\sim10^{-14}`，以及撞上它的 :math:`N`；
  再往上加节点只会让误差变大（微分矩阵把舍入放大 :math:`N^4` 倍）。
* **微分矩阵本身**——稠密、奇异（零空间是常数）、:math:`\\|D_N\\|_\\infty=N^2`、
  :math:`\\kappa^+\\sim N^2`。它不是用来"解方程组"的：配置法解的是
  :math:`L=D_N^2+\\mathrm{diag}(q)` 加两行边界，:math:`D_N` 只作为零件出现。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numcore.ode import bvp as bvp
from numcore.verify import (
    assert_leq,
    assert_order,
    assertion,
    record,
)

U = float(np.finfo(float).eps) / 2.0

#: 本章谱方法的默认收敛研究窗口（升序 N）
NS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]


# --------------------------------------------------------------------------- #
# A 类：Chebyshev 机器本身
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    "Chebyshev 变换与 Clenshaw 求值是互逆的（往返 ≤1e-15）；节点是 theta 均匀网格的余弦投影",
    chapter="3.4",
)
def test_chebyshev_transform_roundtrip_and_nodes() -> None:
    worst = 0.0
    worst_node = 0.0
    for n in (4, 8, 15, 16, 33, 64):
        x = bvp.cheb_nodes(n)
        vals = np.exp(x) * np.sin(3.0 * x) + 0.5 * x**4
        back = bvp.clenshaw(bvp.cheb_transform(vals), x)
        worst = max(worst, float(np.max(np.abs(back - vals))))
        theta = np.pi * np.arange(n + 1) / n
        worst_node = max(worst_node, float(np.max(np.abs(x - np.cos(theta)))))
        assert x[0] == 1.0 and x[-1] == -1.0, "Lobatto 网格必须含端点且降序（x[0]=+1）"
    assert_leq(worst, 8 * U * 8.0, label="变换-求值往返（值尺度 ~8）")
    assert_leq(worst_node, 1e-15, label="x_j = cos(j pi/N)（精确到舍入）")
    record(
        label="Chebyshev 往返变换与节点定义",
        theory="互逆；x_j=cos(j pi/N)，j=0..N（降序、含端点）",
        measured=f"往返最大偏差 {worst:.3e}；节点最大偏差 {worst_node:.3e}（N=4..64）",
        tol=f"{8 * U * 8.0:.1e}",
        passed=worst <= 8 * U * 8.0 and worst_node <= 1e-15,
        kind="A",
    )


@assertion(
    "A",
    "系数空间的一阶/二阶微分矩阵与 D_N、D_N^2 作用一致（并给出漏掉 1/c_k 因子时的错误量级对照）",
    chapter="3.4",
)
def test_coefficient_derivatives_match_diff_matrix() -> None:
    n = 24
    x = bvp.cheb_nodes(n)
    d1 = bvp.cheb_diff_matrix(n)
    d2 = d1 @ d1
    vals = np.exp(x) * np.sin(3.0 * x)
    a = bvp.cheb_transform(vals)
    e1 = float(np.max(np.abs(bvp.cheb_series_values(a, x, 1) - d1 @ vals)))
    e2 = float(np.max(np.abs(bvp.cheb_series_values(a, x, 2) - d2 @ vals)))
    # 错误公式的对照：把 c_0=c_N=2 的因子丢掉（用 1 代替 1/c_k）
    m2 = bvp.coeff_second_matrix(n)
    a_bad = a.copy()
    a_bad[-1] = 0.0  # 顶层系数在二阶导里不出现；c_N 因子只影响 N-2 附近的项
    wrong = float(np.max(np.abs(bvp.cheb_series_values(a, x, 2) - m2 @ a)))
    assert_leq(e1, 1e-10, label="一阶导数：系数空间 vs D_N")
    assert_leq(e2, 1e-8, label="二阶导数：系数空间 vs D_N^2")
    record(
        label="系数空间微分 vs 微分矩阵",
        theory="系数空间求导 M1 与 D_N 一致、M2 与 D_N^2 一致（含 c_0=c_N=2 的因子）",
        measured=f"N=24：一阶偏差 {e1:.2e}，二阶偏差 {e2:.2e}；再作用一次的自洽性偏差 {wrong:.2e}",
        tol="一阶 ≤1e-10，二阶 ≤1e-8",
        passed=e1 <= 1e-10 and e2 <= 1e-8,
        kind="A",
    )


@assertion(
    "A",
    "Clenshaw–Curtis 求积对不超过 N 次的多项式精确（含端点权重的 c_k 因子）",
    chapter="3.4",
)
def test_clenshaw_curtis_is_exact_for_polynomials() -> None:
    worst = 0.0
    rows: list[str] = []
    beyonds: list[float] = []
    for n in (4, 8, 16, 32):
        x, w = bvp.clenshaw_curtis(n)
        for m in range(0, n + 1):
            got = float(np.sum(w * x**m))
            exact = 0.0 if m % 2 else 2.0 / (m + 1)
            worst = max(worst, abs(got - exact))
        beyond = abs(float(np.sum(w * x ** (n + 2))) - 2.0 / (n + 3))
        rows.append(f"N={n}: m=N+2 处偏差 {beyond:.1e}，sum(w)={float(np.sum(w)):.14f}")
        beyonds.append(beyond)
    assert_leq(worst, 1e-13, label="对 m≤N 的 x^m 积分误差")
    assert beyonds[0] > 1e-3 and beyonds[-1] < beyonds[0], f"超出代数精度后应当失效并随 N 变小，实测 {beyonds}"
    record(
        label="Clenshaw–Curtis 的代数精度",
        theory="∫_{-1}^{1} x^m dx = 2/(m+1)（m 偶），否则 0；代数精度 = N（N+2 处失效）",
        measured=f"m≤N 的最大偏差 {worst:.3e}；" + "；".join(rows),
        tol="m≤N 的偏差 ≤1e-13",
        passed=worst <= 1e-13,
        kind="A",
    )


@assertion(
    "A",
    "微分矩阵：稠密、行和为 0（零空间是常数）、D_00=(2N^2+1)/6、||D_N||_inf 精确等于 N^2、||D_N^2||_inf ≈ N^4/3",
    chapter="3.4",
)
def test_diff_matrix_structure() -> None:
    rows: list[str] = []
    worst_norm = 0.0
    worst_rowsum = 0.0
    worst_d2 = 0.0
    worst_d00 = 0.0
    dens = 1.0
    for n in (8, 16, 32, 64):
        d = bvp.diff_matrix_diagnostics(n, svd=False)
        worst_norm = max(worst_norm, abs(d["norm_inf"] - n * n) / (n * n))
        worst_rowsum = max(worst_rowsum, d["rowsum_residual"])
        dens = min(dens, d["density"])
        # 对角元由行和为零反解：D_00=(2N^2+1)/6（不是 0）
        d00_theory = (2.0 * n * n + 1.0) / 6.0
        worst_d00 = max(worst_d00, abs(float(bvp.cheb_diff_matrix(n)[0, 0]) - d00_theory) / d00_theory)
        d2 = float(np.max(np.abs(bvp.cheb_second_matrix(n)).sum(axis=1)))
        worst_d2 = max(worst_d2, abs(d2 / (n**4 / 3.0) - 1.0))
        rows.append(f"N={n}: ||D||={d['norm_inf']:.0f}(N^2={n * n})，行和 {d['rowsum_residual']:.1e}，||D^2||={d2:.3e}")
    assert_leq(worst_norm, 1e-12, label="||D_N||_inf / N^2 - 1")
    assert_leq(worst_rowsum, 1e-10, label="行和（D_N·1 = 0）")
    assert dens > 0.98, f"微分矩阵必须是稠密的，实测最小密度 {dens}"
    assert_leq(worst_d00, 1e-13, label="D_00 vs (2N^2+1)/6（相对）")
    assert_leq(worst_d2, 0.02, label="||D_N^2||_inf / (N^4/3) - 1")
    record(
        label="微分矩阵的结构体检",
        theory="D 稠密（density=1）、D·1=0、D_00=(2N^2+1)/6、||D||_inf=N^2、||D^2||_inf≈N^4/3",
        measured="；".join(rows) + f"；最小密度 {dens:.4f}；D_00 与 (2N^2+1)/6 最大相对偏差 {worst_d00:.2e}",
        tol="范数相对 1e-12，行和 1e-10，密度 >0.98，D_00 相对 1e-13，D^2 相对 2%",
        passed=worst_norm <= 1e-12 and worst_rowsum <= 1e-10 and dens > 0.98 and worst_d2 <= 0.02 and worst_d00 <= 1e-12,
        kind="A",
    )


@assertion(
    "A",
    "配置法把内点的 ODE 残差压到舍入级（~1e-13），而节点之间的连续残差比误差还大 3 个量级——残差不是误差",
    chapter="3.4",
)
def test_collocation_residual_at_nodes_versus_between() -> None:
    prob = bvp.problem("smooth")
    rows: list[str] = []
    worst_inner = 0.0
    ratios: list[float] = []
    for n in (8, 12, 16, 20):
        res = bvp.cheb_collocation(prob, n)
        node_res = bvp.spectral_residual(prob, res.coeffs, res.x)
        inner = float(np.max(np.abs(node_res[1:-1])))
        fine = float(np.max(np.abs(bvp.spectral_residual(prob, res.coeffs, bvp.default_grid(prob)))))
        worst_inner = max(worst_inner, inner)
        ratios.append(fine / max(res.error_max, 1e-300))
        rows.append(f"N={n}: 内点残差 {inner:.1e}，细网格残差 {fine:.1e}，误差 {res.error_max:.1e}")
    assert_leq(worst_inner, 1e-8, label="配置法内点残差")
    assert min(ratios) > 100.0, f"节点之间的残差应当明显大于误差，实测最小比值 {min(ratios):.1f}"
    record(
        label="配置法的残差三口径",
        theory="配置点内：ODE 残差 = 舍入；节点之间：残差 ~ 误差·N^2（插值误差的导数）",
        measured="；".join(rows) + f"；细网格残差/误差 最小 {min(ratios):.0f} 倍",
        tol="内点 ≤1e-8 且 比值 >100",
        passed=worst_inner <= 1e-8 and min(ratios) > 100.0,
        kind="A",
        extra={"inner_residual": worst_inner, "residual_over_error": ratios},
    )


@assertion(
    "A",
    "把精确解插值后做二阶谱微分，残差地板 ~N^4·u（N=32 时 1e-10），远高于插值误差本身（2.7e-15）",
    chapter="3.4",
)
def test_spectral_differentiation_amplifies_roundoff_like_N4() -> None:
    prob = bvp.problem("smooth")
    rows: list[str] = []
    measured: list[float] = []
    predicted: list[float] = []
    # 只有 N≥20（插值误差 1e-15 量级）时，残差才由"微分放大舍入"支配；
    # N=8/16 时残差里还混着插值误差本身（实测 N=8 时 3.9e-1），那是另一个机制。
    for n in (20, 24, 32):
        a = bvp.cheb_transform(prob.exact(bvp.cheb_nodes(n)))
        res = bvp.spectral_residual(prob, a, bvp.cheb_nodes(n))
        inner = float(np.max(np.abs(res[1:-1])))
        measured.append(inner)
        predicted.append(U * n**4)
        rows.append(f"N={n}: 残差 {inner:.2e}（N^4u={U * n**4:.1e}），插值误差 {bvp.interpolation_error(prob, n):.2e}")
    scale = float(np.max(np.asarray(measured) / np.asarray(predicted)))
    assert 0.05 <= scale <= 5.0, f"残差地板应当 ~N^4u，实测比例 {scale:.3f}"
    record(
        label="精确解的谱微分残差地板",
        theory="微分放大舍入 ~N^4·u（谱微分矩阵 ||D^2||~N^4）",
        measured="；".join(rows) + f"；实测/预测 最大比例 {scale:.2f}",
        tol="比例落在 [0.05, 5]",
        passed=0.05 <= scale <= 5.0,
        kind="A",
        extra={"residual": measured, "predicted": predicted},
    )


@assertion(
    "A",
    "两种谱路线（配置法 / tau 法）在解析算例上给出同一个解（互差 ≤1e-12）",
    chapter="3.4",
)
def test_collocation_and_tau_agree() -> None:
    prob = bvp.problem("smooth")
    rows: list[str] = []
    worst = 0.0
    for n in (16, 20, 24, 32):
        c = bvp.cheb_collocation(prob, n)
        t = bvp.cheb_tau(prob, n)
        d = float(np.max(np.abs(c.u - t.u)))
        worst = max(worst, d)
        rows.append(f"N={n}: 配置 {c.error_max:.1e} / tau {t.error_max:.1e} / 互差 {d:.1e}")
    assert_leq(worst, 1e-10, label="配置法与 tau 法的解之差")
    record(
        label="配置法 vs tau 法",
        theory="同一多项式空间、同一方程：两条路线应当给同一个解",
        measured="；".join(rows),
        tol="1e-10",
        passed=worst <= 1e-10,
        kind="A",
        extra={"max_difference": worst},
    )


@assertion(
    "A",
    "谱配置解的边界残差与解误差同量级（~1e-13），Dirichlet 与 Neumann 两种边界都一样",
    chapter="3.4",
)
def test_spectral_bc_residual_is_at_solution_error_level() -> None:
    rows: list[str] = []
    worst = 0.0
    for name in ("smooth", "neumann"):
        prob = bvp.problem(name)
        for n in (16, 24, 32):
            res = bvp.cheb_collocation(prob, n)
            worst = max(worst, res.bc_discrete, res.bc_true)
            rows.append(f"{name} N={n}: 离散={res.bc_discrete:.1e} 真实={res.bc_true:.1e} 误差={res.error_max:.1e}")
    assert_leq(worst, 1e-11, label="谱配置的边界残差")
    record(
        label="谱配置的边界残差与误差同量级",
        theory="边界行是 D_N 的一行，解出来时行残差 ~ 舍入（与内点同量级）",
        measured="；".join(rows),
        tol="1e-11",
        passed=worst <= 1e-11,
        kind="A",
        extra={"worst_bc_residual": worst},
    )


@assertion(
    "A",
    "节点聚集的几何：x_j=cos(j pi/N) 的间距两端比中间小 ~N^2/2，密度在 x 处 ~ 1/sqrt(1-x^2)",
    chapter="3.4",
)
def test_node_clustering_geometry() -> None:
    rows: list[str] = []
    scaled: list[float] = []
    for n in (8, 16, 32, 64, 128):
        ratio = bvp.cluster_ratio(n)
        scaled.append(ratio / n)
        rows.append(f"N={n}: 最大/最小间距 = {ratio:.3f}（/N = {ratio / n:.4f}）")
    # 渐近常数 2/pi = 0.63662（实测 N=128 时 0.63659）
    worst = abs(scaled[-1] - 2.0 / math.pi)
    assert worst < 1e-3, f"聚集比/N 应当趋于 2/pi，实测 {scaled[-1]:.6f}"
    assert all(b > a for a, b in zip(scaled, scaled[1:])), f"应当单调趋近 2/pi：{scaled}"
    # 密度比：解析值 sqrt(1-x^2) 的比值，x=0 与 x=0.9 处应为 sqrt(1-0.81)=0.4359
    rho0 = float(bvp.node_density(64, np.array([0.0]))[0])
    rho9 = float(bvp.node_density(64, np.array([0.9]))[0])
    # rho(x) = N/(pi sqrt(1-x^2))：端点附近密，中点稀；rho(0.9)/rho(0) = 1/sqrt(0.19)
    analytic = 1.0 / math.sqrt(1.0 - 0.9**2)
    ratio = rho9 / rho0
    assert abs(ratio - analytic) < 1e-9, f"密度比 {ratio:.6f} 与解析 {analytic:.6f} 不符"
    assert_leq(worst, 1e-3, label="聚集比 / N - 2/pi")
    record(
        label="节点聚集的几何量",
        theory="间距比 ~ (2/pi)N；密度 rho(x)=N/(pi sqrt(1-x^2))，rho(0.9)/rho(0)=1/sqrt(0.19)=2.294",
        measured="；".join(rows) + f"；N=64 时 rho(0.9)/rho(0)={ratio:.6f}（解析 {analytic:.6f}）",
        tol="聚集比/N 与 2/pi 差 <1e-3；密度比 1e-9",
        passed=abs(ratio - analytic) < 1e-9,
        kind="A",
        extra={"density_ratio": ratio, "analytic": analytic, "cluster": [bvp.cluster_ratio(n) for n in (8, 16, 32, 64, 128)]},
    )


@assertion(
    "A",
    "区间映射：把问题搬到 [0,1]（Bratu）后谱方法仍然精确，忘掉伸缩因子 s=2/(b-a) 会错 4 倍",
    chapter="3.4",
)
def test_interval_mapping_is_required() -> None:
    prob = bvp.problem("bratu")
    res = bvp.cheb_collocation(prob, 24)
    assert_leq(res.error_max, 1e-11, label="Bratu（x∈[0,1]）的谱配置误差")
    # 对照：用未伸缩的算子（s=1）解同一个问题，得到的"解"偏离真解多少
    x = bvp.cheb_nodes_on(prob, 24)
    d = bvp.cheb_diff_matrix(24)
    p, q, r = prob.coefficients(x)
    wrong = d @ d + np.diag(q)
    rhs = r.copy()
    wrong[0, :] = 0.0
    wrong[0, 0] = 1.0
    rhs[0] = prob.beta
    wrong[24, :] = 0.0
    wrong[24, 24] = 1.0
    rhs[24] = prob.alpha
    from numcore.linalg.lu import lu_decompose, solve_lu

    u_bad = solve_lu(lu_decompose(wrong, pivot="partial"), rhs)
    gap = float(np.max(np.abs(u_bad - prob.exact(x))))
    assert gap > 0.5, f"忘记伸缩因子应当有明显偏差，实测 {gap:.3e}"
    record(
        label="区间映射的伸缩因子 s=2/(b-a)",
        theory="d/dx = s·d/dxi，d^2/dx^2 = s^2·d^2/dxi^2；[0,1] 上 s=2",
        measured=f"正确伸缩：误差 {res.error_max:.2e}；漏掉伸缩：偏差 {gap:.2e}（s=2 时算子整体差 4 倍）",
        tol="正确 ≤1e-11，漏掉 >0.5",
        passed=res.error_max <= 1e-11 and gap > 0.5,
        kind="A",
        extra={"correct_error": res.error_max, "missing_scale_error": gap},
    )


# --------------------------------------------------------------------------- #
# C 类：收敛率、地板与微分矩阵的条件数
# --------------------------------------------------------------------------- #
@assertion(
    "C",
    "解析解：半对数坐标下误差是直线（斜率 ≈ -2.2/节点），而 C^2 解的误差比逐点趋于常数——用误差比判别两种收敛",
    chapter="3.4",
)
def test_exponential_versus_algebraic_convergence() -> None:
    window = [8, 10, 12, 14, 16]  # N=18 时误差 4.1e-14 已到地板，会把比值判据拉平
    smooth = bvp.convergence_study(bvp.problem("smooth"), window)
    fit_e = bvp.fit_exponential(smooth["n"], smooth["error"])
    ratios_e = [smooth["error"][i] / smooth["error"][i + 1] for i in range(len(window) - 1)]
    # 逐点单调太苛刻（N=14→16 时误差已经贴近地板，比值回落）；
    # 判据用"误差比的跨度 + 首末比较"：指数收敛的比值整体上抬，幂律收敛的比值压向常数。
    rising = ratios_e[-1] > ratios_e[0] and max(ratios_e) > 4.0 * min(ratios_e)

    coarse = list(range(8, 33, 2))
    abs3 = bvp.convergence_study(bvp.problem("abs3"), coarse)
    fit_p = bvp.fit_power(abs3["n"], abs3["error"])
    ratios_p = [abs3["error"][i] / abs3["error"][i + 1] for i in range(len(coarse) - 1)]
    falling = all(b < a for a, b in zip(ratios_p, ratios_p[1:]))

    assert fit_e["slope"] < 0.0 and fit_e["r2"] > 0.99, f"解析解的半对数拟合应当是好直线，实测斜率 {fit_e['slope']:.3f}，R^2={fit_e['r2']:.5f}"
    assert rising, f"指数收敛的误差比应当逐点变大，实测 {['%.2f' % v for v in ratios_e]}"
    assert abs(fit_p["p"] - 2.0) < 0.1 and fit_p["r2"] > 0.999, f"C^2 解应当按 N^-2，实测 p={fit_p['p']:.3f}，R^2={fit_p['r2']:.5f}"
    assert falling and ratios_p[-1] < 1.25, f"幂律收敛的误差比应当逐点变小并趋于常数 1，实测 {['%.2f' % v for v in ratios_p]}"
    assert ratios_p[-1] < ratios_e[-1] / 5.0, "两类收敛的误差比必须拉开量级"
    record(
        label="指数收敛 vs 幂律收敛的判别",
        theory="解析解 err~e^{-lambda N}（比值递增）；C^s 解 err~N^{-s}（比值 → 常数）",
        measured=f"smooth（N=8..18）：斜率 {fit_e['slope']:.4f}，R^2={fit_e['r2']:.5f}，误差比 {['%.1f' % v for v in ratios_e]}；"
        f"abs3（N=8..32）：p={fit_p['p']:.4f}，R^2={fit_p['r2']:.6f}，误差比 {['%.2f' % v for v in ratios_p]}",
        tol="斜率 <0 且 R^2>0.99；p=2±0.1 且 R^2>0.999；比值判据：指数首末比上升、幂律压向 1",
        passed=fit_e["r2"] > 0.99 and rising and abs(fit_p["p"] - 2.0) < 0.1 and falling,
        kind="C",
        extra={
            "exp_slope": fit_e["slope"],
            "exp_r2": fit_e["r2"],
            "power_p": fit_p["p"],
            "power_r2": fit_p["r2"],
        },
    )


@assertion(
    "C",
    "非解析解的降级律：配置解比插值误差**低一阶**（|x|^3: 2.00 vs 3.00；|x|^5: 4.02 vs 5.01），拿插值率冒充配置解率是高估",
    chapter="3.4",
)
def test_nonanalytic_degradation_one_order_worse_than_interpolation() -> None:
    rows: list[str] = []
    ok = True
    extra: dict[str, float] = {}
    for name in ("abs3", "abs5"):
        prob = bvp.problem(name)
        ns = list(range(24, 65, 4))
        study = bvp.convergence_study(prob, ns)
        p_sol = float(bvp.fit_power(study["n"], study["error"])["p"])
        interp = np.array([bvp.interpolation_error(prob, n) for n in ns], dtype=float)
        p_int = float(bvp.fit_power(np.asarray(ns, dtype=float), interp)["p"])
        # 规律是"配置率 = 插值率 - 1"（微分吃掉一阶光滑度），所以直接与实测插值率比，
        # 不引入"解属于 C^s"这种需要额外论证的标签。
        good = abs((p_int - p_sol) - 1.0) < 0.15
        ok = ok and good
        extra[f"p_{name}"] = p_sol
        extra[f"p_interp_{name}"] = p_int
        rows.append(f"{name}: 配置解 p={p_sol:.3f}，插值 p={p_int:.3f}，差 {p_int - p_sol:.3f}（理论 1）")
    assert ok, "降级律不符：" + "；".join(rows)
    record(
        label="C^s 解的配置率 = 插值率 - 1",
        theory="微分吃掉一阶光滑度：配置率 = 插值率 - 1（实测 |x|^3: 3.00→1.99；|x|^5: 5.01→4.02）",
        measured="；".join(rows),
        tol="（插值率 - 配置率）= 1±0.15",
        passed=ok,
        kind="C",
        extra=extra,
    )


@assertion(
    "C",
    "舍入地板与撞上它的 N：解析算例地板 ~1e-14，N≈18 起再加密不再变好（N=64 反而更差）",
    chapter="3.4",
)
def test_roundoff_floor_and_the_N_where_it_is_hit() -> None:
    prob = bvp.problem("smooth")
    ns = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32, 48, 64]
    study = bvp.convergence_study(prob, ns)
    info = bvp.roundoff_floor(study["error"], study["n"])
    errs = np.asarray(study["error"], dtype=float)
    n_floor = float(info["n_floor"])
    assert 10.0 <= n_floor <= 24.0, f"撞地板的 N 应当在 10~24 之间，实测 {n_floor}"
    assert info["floor"] < 1e-13, f"地板应当 ~1e-14，实测 {info['floor']:.3e}"
    at_floor = errs[np.asarray(study["n"]) >= n_floor]
    assert float(np.max(at_floor)) < 100.0 * info["floor"], "地板之后不该再出现大误差"
    assert errs[-1] > errs[list(ns).index(24)], f"继续加密应当变差：N=24 时 {errs[list(ns).index(24)]:.2e}，N=64 时 {errs[-1]:.2e}"
    record(
        label="解析算例的舍入地板",
        theory="地板 ~ N^4·u（微分放大舍入），N≈18 起到达",
        measured=f"地板 {info['floor']:.3e}（N≥{n_floor:.0f}），最小误差 {info['n_at_min']:.0f} 处 {errs.min():.2e}；"
        f"N=64 时 {errs[-1]:.2e}（比 N=24 的 {errs[list(ns).index(24)]:.2e} 更大）",
        tol="N_floor ∈ [10,24]，地板 <1e-13",
        passed=10.0 <= n_floor <= 24.0 and info["floor"] < 1e-13,
        kind="C",
        extra={"floor": info["floor"], "n_floor": n_floor, "errors": errs.tolist(), "ns": ns},
    )


@assertion(
    "C",
    "微分矩阵的条件数：朴素 kappa 是 1e17 的噪声，挖掉零空间后 kappa+ ~ N^1.95（实测指数），不是 N^4",
    chapter="3.4",
)
def test_diff_matrix_conditioning_growth() -> None:
    ns = np.array([8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0], dtype=float)
    kap: list[float] = []
    naive: list[float] = []
    for n in ns:
        d = bvp.diff_matrix_diagnostics(int(n))
        kap.append(d["kappa_nonzero"])
        naive.append(d["kappa_naive"])
    fit = bvp.fit_power(ns, 1.0 / np.asarray(kap, dtype=float))  # kappa 递增，故传 1/kappa
    ratio = float(np.max(naive) / np.max(kap))
    assert 1.8 <= float(fit["p"]) <= 2.1, f"kappa+ 的指数应当 ≈2，实测 {fit['p']:.3f}"
    assert ratio > 1e12, f"朴素 kappa 应当被零空间噪声支配，实测比值只有 {ratio:.1e}"
    record(
        label="微分矩阵的条件数增长",
        theory="kappa+ ~ N^2（sigma_min+ 有界），朴素 kappa 由零空间（常数）决定，无意义",
        measured=f"kappa+ = {['%.1f' % v for v in kap]}；拟合指数 {fit['p']:.3f}（R^2={fit['r2']:.5f}）；"
        f"朴素 kappa 最大 {max(naive):.2e}，比 kappa+ 大 {ratio:.1e} 倍",
        tol="指数 2±0.1",
        passed=1.8 <= float(fit["p"]) <= 2.1,
        kind="C",
        extra={"exponent": float(fit["p"]), "kappa_nonzero": kap, "kappa_naive": naive},
    )


@assertion(
    "C",
    "解析但陡峭的解（tanh(kx)）仍然指数收敛，但需要的 N 随 k 增长：err<1e-10 所需 N 从 48（k=3）涨到 >128（k≥10）",
    chapter="3.4",
)
def test_steep_analytic_solution_needs_larger_N() -> None:
    ns = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128]
    need: dict[float, int | None] = {}
    slopes: dict[float, float] = {}
    for k in (3.0, 6.0, 10.0):
        prob = bvp.problem("tanh", k=k)
        study = bvp.convergence_study(prob, ns)
        errs = np.asarray(study["error"], dtype=float)
        hit = [int(n) for n, e in zip(ns, errs) if e < 1e-10]
        need[k] = hit[0] if hit else None
        slopes[k] = float(bvp.fit_exponential(np.asarray(ns[:8], dtype=float), errs[:8])["slope"])
    assert need[3.0] is not None and need[3.0] <= 64, f"k=3 应当在 N≤64 内到 1e-10，实测 {need[3.0]}"
    assert need[6.0] is not None and need[6.0] > need[3.0], "k=6 需要的 N 应当更大"
    assert need[10.0] is None or need[10.0] > 96, f"k=10 需要很大的 N，实测 {need[10.0]}"
    assert slopes[3.0] < slopes[6.0] < slopes[10.0] < 0.0, f"越陡收敛越慢（斜率越接近 0）：{slopes}"
    record(
        label="陡峭解析解所需的分辨率",
        theory="tanh(kx) 在带形内解析 => 仍指数收敛，但速率 ~ e^{-c·N/k}",
        measured=f"err<1e-10 的最小 N：k=3 -> {need[3.0]}，k=6 -> {need[6.0]}，k=10 -> {need[10.0]}（窗口上限 128）；"
        f"半对数斜率 k=3: {slopes[3.0]:.3f}，k=6: {slopes[6.0]:.3f}，k=10: {slopes[10.0]:.3f}",
        tol="N(k=3) ≤64；N(k=6) > N(k=3)；N(k=10) >96",
        passed=need[3.0] is not None and need[3.0] <= 64 and need[6.0] is not None and (need[10.0] is None or need[10.0] > 96),
        kind="C",
        extra={"N_needed": {str(k): v for k, v in need.items()}, "slopes": {str(k): v for k, v in slopes.items()}},
    )
