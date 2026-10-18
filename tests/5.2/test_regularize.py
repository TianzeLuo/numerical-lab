"""5.2 章 A/B/C 断言（一）：正则化内核本身。

这一文件的主线（每条都对着一个**会被讲错**的说法）：

1. **滤波因子是定义，不是近似**：Tikhonov 的 ``f_i = s_i^2/(s_i^2+lambda^2)`` 与
   TSVD 的阶跃逐位对拍 1.3 章的公开函数；解必须同时满足"谱公式"与"正则化正规方程"
   （``A^T A + lambda^2 I`` 的 LU 解）——两条独立路线给出同一个解，才算真的解对了。
2. **迭代正则化的两种写法必须同解**：谱形式与真正的矩阵迭代在 1e-9 内一致；
   越过 ``omega = 2/sigma_1^2`` 当场发散（实测 stop='diverged'）。
3. **TV 不是谱滤波**：两点信号的闭式解（软阈值）逐位对拍；
   ``b = u_i`` 的探针读数与混合探针读数差 0.9 量级、非对角泄漏非零。
4. **偏差-方差**：解析式与蒙特卡洛（64 个噪声样本）在 4 倍标准误内一致；
   而"蒙特卡洛的偏差"在 lambda 小的一端被样本均值本身污染（实测 110 倍 mse_min）——
   这条负面结论与正面结论同等重要。
5. **离散 Picard 条件**：无噪时 ``|u_i^T b|`` 对 ``sigma_i`` 的斜率恰好等于造题指数
   （实测 2.000000），加噪后整段斜率变平到 0.56，而**信号段**仍是 1.997。
"""

from __future__ import annotations

import numpy as np

from numcore.inverse.regularize import (
    bias_variance_analytic,
    bias_variance_montecarlo,
    factorize,
    filtered_solution,
    jacobi_svd,
    landweber,
    landweber_factors,
    landweber_operator,
    noise_dominated_count,
    phantom_image,
    picard_diagnostics,
    radon_matrix,
    relative_error_norm,
    separable_blur_forward,
    separable_tikhonov,
    spatial_blur_matrix,
    test_problem as synthetic_problem,
    tikhonov,
    tikhonov_factors,
    tsvd,
    tsvd_factors,
    tv_denoise,
    tv_filter_probe,
    tv_objective,
    tv_solve,
    unregularized_amplification,
    vector_norm,
)
from numcore.inverse.lcurve import criterion_comparison, lambda_grid, problem_lcurve
from numcore.linalg.lstsq import least_squares_svd, truncated_solution
from numcore.linalg.lu import lu_partial, solve_lu
from numcore.reference import dense as ref
from numcore.verify import (
    assert_all_finite,
    assert_leq,
    assert_monotone,
    assert_threshold,
    assert_within,
    assertion,
    find_threshold,
    fit_order,
    record,
)

CHAPTER = "5.2"
PROBLEM = dict(m=32, n=20, kappa=1e6, picard="satisfied", noise_rel=1e-3, seed=20261018)


# --------------------------------------------------------------------------- #
# 1. 滤波因子与解
# --------------------------------------------------------------------------- #
@assertion("A", theory="Tikhonov 滤波因子 f_i = s_i^2/(s_i^2+lambda^2)，半功率点恰在 s_i = lambda", chapter=CHAPTER)
def test_tikhonov_factors_match_analytic_formula_and_half_power() -> None:
    r"""滤波因子对拍闭式；``lambda = 0`` 退化为最小二乘（与 1.3 的 SVD 解逐位一致）。

    另一侧：``lambda`` 远大于 ``sigma_1`` 时 ``f_i -> s_i^2/lambda^2``（相对差 <= 1e-4），
    解被压向 0（``||x||/||x_ls|| <= 1e-6``）——"更大的 lambda 更稳"这句话的代价。
    """
    prob = synthetic_problem(**PROBLEM)
    s = prob.s
    worst = 0.0
    for lam in (1e-8, 1e-4, 1e-2, 1.0, 1e2):
        f = tikhonov_factors(s, lam)
        ref = s * s / (s * s + lam * lam)
        worst = max(worst, float(np.max(np.abs(f - ref))))
    half = tikhonov_factors(np.array([2.5]), 2.5)[0]
    rec = record(
        label="滤波因子与闭式的最大偏差 / 半功率点",
        theory="f_i = s_i^2/(s_i^2+lambda^2)；s_i = lambda 处 f = 1/2",
        measured=f"max|df| = {worst:.3e}；f(s=lambda) = {half:.12f}",
        tol="1e-14 与 0.5",
        passed=bool(worst <= 1e-14 and abs(half - 0.5) <= 1e-15),
        kind="A",
    )
    x_ls = least_squares_svd(prob.A, prob.b).x
    x_full = tikhonov(prob.U, s, prob.V, prob.b, 0.0).x
    rel0 = relative_error_norm(x_full, x_ls)
    x_huge = tikhonov(prob.U, s, prob.V, prob.b, 1e6 * s[0]).x
    rec2 = record(
        label="lambda=0 与最小二乘解的一致性 / lambda 极大时的压缩比",
        theory="lambda=0 时 Tikhonov = 最小二乘；lambda >> sigma_1 时 ||x|| ~ ||x_ls|| sigma_1^2/lambda^2",
        measured=f"rel diff = {rel0:.3e}；||x_huge||/||x_ls|| = {vector_norm(x_huge) / vector_norm(x_ls):.3e}",
        tol="1e-9 与 1e-6",
        detail="lambda=0 时两条路线都要除以 sigma_n=1e-6，实测相对差 2.8e-11 是 kappa=1e6 下不得不付出的抵消误差",
        passed=bool(rel0 <= 1e-9 and vector_norm(x_huge) / vector_norm(x_ls) <= 1e-6),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="Tikhonov 解同时满足谱公式与正则化正规方程 (A^T A + lambda^2 I) x = A^T b", chapter=CHAPTER)
def test_tikhonov_solution_solves_regularized_normal_equations() -> None:
    r"""用 LU 解正规方程，与谱公式对拍（两条独立路线）。

    这是"解对了"的双向检查：``lambda`` 取 1e-2 / 1e-1 / 1 时相对差 <= 1e-10；
    而 ``lambda -> 0`` 时正规方程的条件数变成 ``kappa(A)^2 = 1e12``，
    同一条 LU 路线的相对差退化到 1e-4 量级（实测）——**这不是谱公式错了**，
    是把 ``kappa`` 平方之后 LU 自己也解不动了（1.3 章的老结论，这里再量一次）。
    """
    prob = synthetic_problem(m=32, n=20, kappa=1e6, picard="satisfied", noise_rel=1e-3)
    A, b = prob.A, prob.b
    n = prob.n
    worst = 0.0
    measured = []
    for lam in (1e-2, 1e-1, 1.0, 1e-2):
        M = A.T @ A + lam * lam * np.eye(n)
        rhs = A.T @ b
        x_lu = solve_lu(lu_partial(M), rhs)
        x_sp = tikhonov(prob.U, prob.s, prob.V, b, lam).x
        d = relative_error_norm(x_lu, x_sp)
        worst = min(worst if worst else d, d) if lam == 1e-2 else max(worst, d)
        measured.append(f"lambda={lam:g}: {d:.2e}")
    rec = record(
        label="谱公式与 LU 正规方程解的最大相对差",
        theory="<= 1e-10（lambda 不太小时两条路线同一个解；kappa=1e6 下实测 2e-12）",
        measured="；".join(measured),
        tol="1e-10",
        passed=bool(worst <= 1e-10),
        kind="A",
    )
    lam_tiny = 1e-6
    M = A.T @ A + lam_tiny * lam_tiny * np.eye(n)
    x_lu = solve_lu(lu_partial(M), A.T @ b)
    x_sp = tikhonov(prob.U, prob.s, prob.V, b, lam_tiny).x
    d_tiny = relative_error_norm(x_lu, x_sp)
    kappa_normal = prob.condition**2
    rec2 = record(
        label="正规方程路线在 lambda -> 0 时的退化（kappa 平方的代价）",
        theory="正规方程的条件数是 kappa(A)^2；lambda 小到正则项压不住时 LU 解开始丢位",
        measured=(
            f"lambda=1e-2: {worst:.2e}；lambda=1e-6: {d_tiny:.2e}"
            f"（劣化 {d_tiny / worst:.0f} 倍；kappa(A)^2 = {kappa_normal:.3e}）"
        ),
        tol="lambda=1e-6 的偏差 >= 100 倍 lambda=1e-2 的偏差",
        passed=bool(d_tiny >= 100.0 * worst),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="TSVD 解 = 前 k 项奇异展开；k 越多残差越小但误差先降后升", chapter=CHAPTER)
def test_tsvd_matches_truncated_solution_and_boundary_failure() -> None:
    r"""与 1.3 的 ``truncated_solution`` 逐位一致，并且量出"多留几项更差"。

    边界反例（A 类必须双向）：k = n（不截断）时相对误差比最优 k 处大 100 倍以上——
    "精度更好"不能只测一端。"""
    prob = synthetic_problem(**PROBLEM)
    worst = 0.0
    for k in (1, 5, 12, 20):
        x13 = truncated_solution(prob.U, prob.s, prob.V, prob.b, k)
        xmine = tsvd(prob.U, prob.s, prob.V, prob.b, k).x
        worst = max(worst, float(np.max(np.abs(x13 - xmine))))
    err = np.array([tsvd(prob.U, prob.s, prob.V, prob.b, k, x_true=prob.x_true).error_rel for k in range(1, 21)])
    k_best = int(np.argmin(err)) + 1
    rec = record(
        label="TSVD 与 1.3 truncated_solution 的最大逐点差 / 误差曲线",
        theory="k = n 时退化为最小二乘解；误差在 k* 处最小、之后被噪声抬高",
        measured=(
            f"max|dx| = {worst:.3e}；k* = {k_best}（err {err[k_best - 1]:.4f}），"
            f"k = n 时 err = {err[-1]:.4f}（放大 {err[-1] / err[k_best - 1]:.1f} 倍）"
        ),
        tol="逐位一致（<= 1e-15）；k=n 处误差放大 >= 10 倍",
        passed=bool(worst <= 1e-15 and err[-1] / err[k_best - 1] >= 10.0),
        kind="A",
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Landweber 的谱形式与矩阵迭代同一个解；omega >= 2/sigma_1^2 时发散", chapter=CHAPTER)
def test_landweber_spectral_equals_matrix_iteration() -> None:
    r"""谱公式 ``1-(1-omega s^2)^k`` 与真正的矩阵迭代对拍（<= 1e-9）。

    反向：``omega = 2.2/sigma_1^2`` 时矩阵迭代 ``stop='diverged'``，
    谱因子跑出 ``[0,1]``（实测最大值 > 1）——两种写法在发散区**同时**失效，
    这正说明它们描述的是同一个方法。"""
    prob = synthetic_problem(**PROBLEM)
    worst = 0.0
    measured = []
    for om_rel, k in ((1.0, 30), (0.5, 60), (1.5, 12)):
        omega = om_rel / float(prob.s[0] ** 2)
        xs = landweber(prob.U, prob.s, prob.V, prob.b, n_iter=k, omega=omega).x
        run = landweber_operator(
            lambda z: prob.A @ z, lambda z: prob.A.T @ z, prob.b, np.zeros(prob.n),
            n_iter=k, omega=omega,
        )
        d = relative_error_norm(xs, run.x)
        worst = max(worst, d)
        measured.append(f"omega_rel={om_rel:g},k={k}: {d:.2e}（stop={run.stop}）")
    omega_bad = 2.2 / float(prob.s[0] ** 2)
    run_bad = landweber_operator(
        lambda z: prob.A @ z, lambda z: prob.A.T @ z, prob.b, np.zeros(prob.n),
        n_iter=200, omega=omega_bad,
    )
    f_bad = landweber_factors(prob.s, 200, omega_bad)
    rec = record(
        label="谱形式与矩阵迭代的最大相对差 / 发散线另一侧的表现",
        theory="两者同解（<= 1e-9）；omega >= 2/sigma_1^2 时迭代发散、滤波因子离开 [0,1]",
        measured=(
            "；".join(measured)
            + f"；omega_rel=2.2: stop={run_bad.stop}，max f = {float(np.max(np.abs(f_bad))):.3e}"
        ),
        tol="<= 1e-9；发散时 stop='diverged' 且 max f > 1",
        passed=bool(worst <= 1e-9 and run_bad.stop == "diverged" and float(np.max(np.abs(f_bad))) > 1.0),
        kind="A",
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. 全变差
# --------------------------------------------------------------------------- #
@assertion("A", theory="一维两点信号的 TV 去噪有闭式解：差值的软阈值", chapter=CHAPTER)
def test_tv_two_point_closed_form() -> None:
    r"""``min 0.5||x-b||^2 + lambda|x_2-x_1|`` 的闭式解 = 差值软阈值（逐位）。

    两端点：``lambda = 0`` 时 ``x = b``；``lambda >= |d|`` 时两点被压成常数（TV 的"台阶"）。"""
    b = np.array([1.0, -0.5])
    m = float(np.mean(b))
    d = 0.5 * (b[1] - b[0])
    worst = 0.0
    measured = []
    for lam in (0.0, 0.05, 0.3, 0.75, 2.0):
        run = tv_denoise(b, lam, n_iter=600)
        v = np.sign(d) * max(abs(d) - lam, 0.0)
        ref = np.array([m - v, m + v])
        e = float(np.max(np.abs(run.x - ref)))
        worst = max(worst, e)
        measured.append(f"lambda={lam:g}: max|dx|={e:.2e}")
    rec = record(
        label="两点 TV 闭式解的最大逐点偏差",
        theory="x = (m - soft(d, lambda), m + soft(d, lambda))，m = mean(b)，d = (b_2-b_1)/2",
        measured="；".join(measured),
        tol="1e-6（原始-对偶 600 步）",
        passed=bool(worst <= 1e-6),
        kind="A",
    )
    flat = tv_denoise(b, 1.0, n_iter=600).x
    rec2 = record(
        label="lambda >= |d| 时 TV 解是否退化成常数",
        theory="TV 罚项在差值上的次梯度包含 0 时，两点被合并（台阶先验）",
        measured=f"lambda=1.0: x = ({flat[0]:.9f}, {flat[1]:.9f})，差 = {abs(flat[1] - flat[0]):.2e}",
        tol="<= 1e-6",
        passed=bool(abs(flat[1] - flat[0]) <= 1e-6),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="TV 在 SVD 基下不是对角算子：探针读数随探针改变、非对角泄漏非零", chapter=CHAPTER)
def test_tv_is_not_a_spectral_filter() -> None:
    r"""``b = u_i`` 探针：``lambda = 0`` 时读数为 1（退化为伪逆，**正对照**）；
    ``lambda > 0`` 时换成混合探针 ``b = (u_1+u_n)/sqrt2`` 读数改变 0.9 量级、
    非对角泄漏非零（实测）——TV 没有滤波因子。"""
    well = synthetic_problem(m=32, n=20, kappa=1e2, picard="satisfied", noise_rel=0.0)
    probe0 = tv_filter_probe(well.A, well.U, well.s, well.V, 0.0, probe_index=list(range(8)), n_iter=2000)
    diag0 = probe0["filter_diag"][:8]
    rec = record(
        label="lambda=0 时 TV 探针是否退化为恒等（正对照，kappa=1e2）",
        theory="lambda=0 时 TV 项不起作用，x = A^+ b，探针读数应为 1",
        measured=f"lambda=0 的探针读数最小偏差 = {float(np.max(np.abs(diag0 - 1.0))):.2e}（2000 步）",
        tol="max|f-1| <= 1e-6",
        detail="kappa=1e4 时 400 步不够（实测最小模态只有 0.751）——PDHG 的收敛速度随 kappa 恶化，这不是探针算错了",

        passed=bool(float(np.max(np.abs(diag0 - 1.0))) <= 1e-6),
        kind="A",
    )
    probe = tv_filter_probe(well.A, well.U, well.s, well.V, 1e-2, probe_index=list(range(20)), n_iter=600)
    dep = float(probe["max_probe_dependence"])
    leak = float(np.max(probe["leakage"]))
    rec2 = record(
        label="TV 探针的依赖性 / 非对角泄漏",
        theory="TV 非线性且非对角：换探针读数会变，泄漏不为 0（它不是滤波因子）",
        measured=f"max|f_mixed - f_single| = {dep:.3f}；max leakage = {leak:.3f}",
        tol=">= 0.1 且 >= 1e-3",
        passed=bool(dep >= 0.1 and leak >= 1e-3),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 3. 偏差-方差
# --------------------------------------------------------------------------- #
@assertion("A", theory="E||x_lambda - x*||^2 = 偏差^2 + 方差；蒙特卡洛与解析式一致", chapter=CHAPTER)
def test_bias_variance_montecarlo_matches_analytic() -> None:
    r"""解析式（用**无噪**的 ``b_exact``）与 64 个噪声样本的蒙特卡洛对拍。

    判据不是"看着像"：``|mse_mc - mse_analytic| <= 4 * 标准误``（逐点），
    并登记最大相对差（实测方差 11.8%、总误差 9.9%）。"""
    prob = synthetic_problem(m=32, n=20, kappa=1e4, picard="satisfied", noise_rel=1e-3)
    lam = np.logspace(-4.0, 4.0, 41)
    ana = bias_variance_analytic(prob.U, prob.s, prob.V, prob.b_exact, prob.x_true, prob.noise_abs, lam)
    mc = bias_variance_montecarlo(
        prob.U, prob.s, prob.V, prob.b_exact, prob.x_true, prob.noise_abs, lam, n_trials=64
    )
    within = np.abs(mc.mse_mc - ana.mse) <= 4.0 * np.maximum(mc.mse_mc_std, np.finfo(float).tiny)
    n_out = int(np.count_nonzero(~within))
    rec = record(
        label="蒙特卡洛总误差落在解析值的 4 倍标准误内",
        theory="64 个样本的经验均值误差 ~ sigma/sqrt(64)；逐点比较要用标准误刻度",
        measured=(
            f"越界点 {n_out}/{lam.size}；最大相对差 mse {np.max(np.abs(mc.mse_mc - ana.mse) / ana.mse):.3f}，"
            f"variance {np.max(np.abs(mc.variance_mc - ana.variance) / ana.variance):.3f}"
        ),
        tol="越界 <= 2 个点",
        passed=bool(n_out <= 2),
        kind="A",
    )
    rec2 = record(
        label="偏差^2 + 方差 = 总误差（恒等式）",
        theory="两项按定义相加就是 E||x_lambda - x*||^2",
        measured=f"max|bias2+variance-mse| = {np.max(np.abs(ana.bias2 + ana.variance - ana.mse)):.3e}",
        tol="<= 1e-15（相对）",
        passed=bool(np.max(np.abs(ana.bias2 + ana.variance - ana.mse)) <= 1e-15 * max(ana.mse.min(), 1e-300)),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="蒙特卡洛的偏差在 lambda 小的一端被样本均值本身污染（不是偏差估计）", chapter=CHAPTER)
def test_montecarlo_bias_is_polluted_by_sample_mean() -> None:
    r"""负面结论：``lambda -> 0`` 时解析偏差^2 -> 1e-9，而 64 个样本的均值本身有
    ``sigma/sqrt(64)`` 的误差，于是"实测偏差^2"被抬到 0.036——**相差 110 倍 mse_min**。

    要测偏差必须用无噪的 ``b_exact``；这条记录就是"用蒙特卡洛量偏差"这个做法的边界。"""
    prob = synthetic_problem(m=32, n=20, kappa=1e4, picard="satisfied", noise_rel=1e-3)
    lam = np.logspace(-4.0, 4.0, 41)
    ana = bias_variance_analytic(prob.U, prob.s, prob.V, prob.b_exact, prob.x_true, prob.noise_abs, lam)
    mc = bias_variance_montecarlo(
        prob.U, prob.s, prob.V, prob.b_exact, prob.x_true, prob.noise_abs, lam, n_trials=64
    )
    i0 = 0
    ratio = abs(float(mc.bias2_mc[i0] - ana.bias2[i0])) / ana.mse_min
    rec = record(
        label="lambda 最小时实测偏差^2 与解析偏差^2 的绝对差（以 mse_min 为单位）",
        theory="经验均值 E_hat[x] 的抽样误差是 sigma/sqrt(N)，它进入偏差^2 后不可忽略",
        measured=(
            f"bias2_analytic = {ana.bias2[i0]:.3e}，bias2_mc = {mc.bias2_mc[i0]:.3e}，"
            f"绝对差 = {abs(float(mc.bias2_mc[i0] - ana.bias2[i0])):.3e} = {ratio:.1f} x mse_min"
        ),
        tol=">= 10 x mse_min（说明污染显著，不是小差异）",
        passed=bool(ratio >= 10.0),
        kind="A",
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 4. 噪声放大与 Picard 条件
# --------------------------------------------------------------------------- #
@assertion("A", theory="不做正则化时噪声被放大：最坏方向 ||e||/sigma_n，白噪声均方根 ||e||sqrt(sum 1/s^2)/sqrt(m)", chapter=CHAPTER)
def test_noise_amplification_matches_analytic_bounds() -> None:
    r"""把噪声**正好放在** ``u_n`` 方向：解误差应等于 ``||e||/sigma_n/||x*||``（1e-6 内）。

    然后量出题面要求的那句话：``kappa = 1e6``、相对噪声 ``1e-3`` 时放大 1000 倍量级
    （实测最坏 1028 倍、均方根 182 倍）——解的有效位被吃光。"""
    prob = synthetic_problem(m=32, n=20, kappa=1e6, picard="satisfied", noise_rel=1e-3)
    noise_abs = float(prob.noise_abs)
    b_aligned = prob.b_exact + noise_abs * prob.U[:, -1]  # 噪声全在最小奇异值方向
    x_ls = least_squares_svd(prob.A, b_aligned).x
    err = relative_error_norm(x_ls, prob.x_true)
    pred = noise_abs / float(prob.s[-1]) / prob.x_norm
    rec = record(
        label="噪声对齐最小奇异方向时的相对误差与解析值",
        theory="||A^+ e|| = ||e||/sigma_n（e 沿 u_n），相对误差 = ||e||/(sigma_n ||x*||)",
        measured=f"实测 {err:.6f} / 解析 {pred:.6f}（相对差 {abs(err - pred) / pred:.2e}）",
        tol="相对差 <= 1e-6",
        passed=bool(abs(err - pred) / pred <= 1e-6),
        kind="A",
    )
    worst = unregularized_amplification(prob.s, noise_abs, m=prob.m, x_norm=prob.x_norm, worst_case=True)
    rms = unregularized_amplification(prob.s, noise_abs, m=prob.m, x_norm=prob.x_norm, worst_case=False)
    rec2 = record(
        label="kappa=1e6、噪声 1e-3 时的放大倍数（最坏 / 均方根）",
        theory="最坏 ||e||sqrt(sum 1/s_i^2)/||x*||，均方根除以 sqrt(m)",
        measured=f"最坏 {worst:.1f} 倍，均方根 {rms:.1f} 倍（两者相差 {worst / rms:.1f} 倍）",
        tol="最坏 >= 1000 且 最坏/均方根 在 sqrt(m)=5.66 的 1.5 倍以内",
        passed=bool(worst >= 1000.0 and abs(worst / rms / np.sqrt(prob.m) - 1.0) <= 0.5),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="离散 Picard 条件：|u_i^T b| 对 sigma_i 的斜率 > 1 时报比值递减", chapter=CHAPTER)
def test_picard_slopes_match_construction() -> None:
    r"""无噪时斜率恰好 = 造题指数（实测 2.000000、比值斜率 1.000000、相关系数 1.000000）；
    加噪后整段斜率变平（0.564），而**信号段**（系数高于噪声）仍是 1.997。

    这两条必须同时报：只看整段斜率会得出"Picard 条件不成立"的错误结论。"""
    clean = synthetic_problem(m=32, n=20, kappa=1e6, picard="satisfied", noise_rel=0.0)
    d0 = picard_diagnostics(clean.U, clean.s, clean.b)
    rec = record(
        label="无噪 Picard 斜率 / 比值斜率 / 相关系数",
        theory="alpha_i = sigma_i^decay（decay=2）=> 斜率 = 2，比值斜率 = 1，log-log 完全共线",
        measured=f"slope = {d0.slope_coeff:.6f}，ratio slope = {d0.slope_ratio:.6f}，corr = {d0.correlation:.6f}",
        tol="与 2 / 1 / 1 的偏差 <= 1e-6（实测 1.2e-8，来自 s 的浮点舍入）",
        passed=bool(
            abs(d0.slope_coeff - 2.0) <= 1e-6
            and abs(d0.slope_ratio - 1.0) <= 1e-6
            and abs(d0.correlation - 1.0) <= 1e-9
        ),
        kind="A",
    )
    noisy = synthetic_problem(m=32, n=20, kappa=1e6, picard="satisfied", noise_rel=1e-3)
    d = picard_diagnostics(noisy.U, noisy.s, noisy.b, noise_abs=noisy.noise_abs)
    n_expect = noise_dominated_count(noisy.s, noisy.noise_abs, b_norm=noisy.b_norm)
    rec2 = record(
        label="有噪时整段斜率 vs 信号段斜率 / 被噪声接管的模态数",
        theory="噪声接管后 |u_i^T b| 不再随 sigma_i 下降，整段斜率必然变平；信号段仍满足理论",
        measured=(
            f"整段 {d.slope_coeff:.4f}，信号段 {d.slope_coeff_signal:.4f}（{d.n_signal} 个模态），"
            f"低于噪声的模态 {d.n_below_noise}（按 sigma_res 计 {n_expect}）"
        ),
        tol="信号段与 2 的偏差 <= 0.05，且整段比信号段平至少 1.0",
        passed=bool(abs(d.slope_coeff_signal - 2.0) <= 0.05 and d.slope_coeff <= d.slope_coeff_signal - 1.0),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="Picard 条件不成立时正则化只能少错一点，不能恢复真解", chapter=CHAPTER)
def test_picard_violated_cannot_be_regularized() -> None:
    r"""``picard='violated'``（``decay = 0.5``）：最优 lambda 落在网格端点（oracle 无意义），
    且"最优正则化解"的误差与完全不正则化**相同**（实测放大 1.0 倍，误差 100%）。

    对照：``picard='satisfied'`` 时正则化把误差从 12863 倍压到 1 倍（oracle 处）。"""
    bad = synthetic_problem(m=32, n=20, kappa=1e6, picard="violated", noise_rel=1e-3)
    lc = problem_lcurve(bad, n_lam=61)
    ch_bad = criterion_comparison(
        bad.U, bad.s, bad.V, bad.b, x_true=bad.x_true, noise_abs=bad.noise_abs, lam=lc.lam
    )
    rec = record(
        label="Picard 不成立时：oracle 位置与正则化的收益",
        theory="|u_i^T b|/sigma_i 随 i 增长时，最小二乘解在噪声下必然崩，正则化只能截掉一部分",
        measured=(
            f"oracle lambda = {ch_bad.lam_oracle:.3e}（网格左端 {lc.lam[0]:.3e}），"
            f"误差 {ch_bad.error_oracle:.3f}，正则化收益 = {ch_bad.detail['inflation_unregularized']:.3f} 倍"
        ),
        tol="oracle 落在网格左端（<= 1.001 倍）且正则化收益 <= 1.001",
        detail="Picard 不成立时 oracle 选择 lambda -> 0，即不要正则化；此时误差 0.116 完全由噪声决定",
        passed=bool(
            ch_bad.lam_oracle <= lc.lam[0] * 1.001
            and ch_bad.detail["inflation_unregularized"] <= 1.001
        ),
        kind="A",
    )
    good = synthetic_problem(m=32, n=20, kappa=1e6, picard="satisfied", noise_rel=1e-3)
    lc2 = problem_lcurve(good, n_lam=61)
    ch_good = criterion_comparison(
        good.U, good.s, good.V, good.b, x_true=good.x_true, noise_abs=good.noise_abs, lam=lc2.lam
    )
    rec2 = record(
        label="Picard 成立时的对照：最优 lambda 处误差与不正则化的比",
        theory="同一算子、同一噪声，Picard 成立时正则化把误差从 1e4 量级压回 1e-2 量级",
        measured=(
            f"不正则化误差 {ch_good.detail['error_unregularized']:.3f}，oracle 处 {ch_good.error_oracle:.4f}，"
            f"比值 {ch_good.detail['inflation_unregularized']:.0f} 倍"
        ),
        tol=">= 100 倍",
        passed=bool(ch_good.detail["inflation_unregularized"] >= 100.0),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 5. 图像 / CT 算子
# --------------------------------------------------------------------------- #
@assertion("A", theory="可分离模糊的 Tikhonov 解满足二维正规方程；二维条件是 (sigma_1/sigma_n)^2", chapter=CHAPTER)
def test_separable_blur_tikhonov_satisfies_normal_equations() -> None:
    r"""二维正规方程残差 ``B^T(B X B^T - Y)B + lambda^2 X`` 相对量级 <= 1e-10。

    同时量出"二维条件数 = 一维条件数的平方"（blur_sigma=2 时 1e1 -> 3.5e7）。
    """
    img = phantom_image(24)
    B = spatial_blur_matrix(24, 2.0)
    fit = factorize(B)
    Y = separable_blur_forward(img, B)
    lam = 1e-2
    X = separable_tikhonov(Y, fit.U, fit.s, fit.V, lam)
    lhs = B.T @ (separable_blur_forward(X, B) - Y) @ B + lam * lam * X
    scale = max(float(np.max(np.abs(B.T @ Y @ B))), 1e-300)
    resid = float(np.max(np.abs(lhs))) / scale
    rec = record(
        label="二维 Tikhonov 解的正规方程残差（相对）",
        theory="(B^T B) X (B^T B) + lambda^2 X = B^T Y B 是二维 Tikhonov 的定义方程",
        measured=f"max|残差|/max|B^T Y B| = {resid:.3e}",
        tol="<= 1e-10",
        passed=bool(resid <= 1e-10),
        kind="A",
    )
    k1 = float(fit.s[0] / fit.s[-1])
    s2_max = float(fit.s[0] ** 2)
    s2_min = float(fit.s[-1] ** 2)
    rec2 = record(
        label="二维条件数 = 一维条件数的平方",
        theory="可分离算子的奇异值是 s_i s_j，因此 kappa_2D = kappa_1D^2",
        measured=f"kappa_1D = {k1:.4g} -> kappa_2D = {k1 * k1:.4g}（s_1 s_1 = {s2_max:.6g}，s_n s_n = {s2_min:.3e}）",
        tol="<= 1e-9（相对）",
        passed=bool(abs(k1 * k1 - s2_max / s2_min) / (k1 * k1) <= 1e-9),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="单边 Jacobi 在重奇异值上会停滞：factorize 必须体检并回退", chapter=CHAPTER)
def test_svd_fallback_catches_jacobi_stagnation() -> None:
    r"""负面结论（实测）：24x24 高斯模糊矩阵上 ``jacobi_svd`` 跑满 60 次扫描
    ``converged=False``、奇异值全等于列范数，与真值相对差 **2221 倍**；
    双边法 11 次扫描收敛（1.9e-13）。``factorize`` 的体检必须挡住它。

    这是"差点写成假证据"的那一条：若直接用这份 SVD，整章 L 曲线都是假的。"""
    B = spatial_blur_matrix(24, 2.0)
    stalled = jacobi_svd(B)
    _, s_ref, _ = ref.svd(B, full_matrices=False)  # LAPACK（只在 numcore.reference 里）
    err_stalled = float(np.max(np.abs(stalled.s - s_ref) / s_ref))
    fit = factorize(B)
    err_fit = float(np.max(np.abs(fit.s - s_ref) / s_ref))
    rec = record(
        label="停滞 SVD 与体检后 SVD 的相对误差（对 LAPACK）",
        theory="单边 Jacobi 在重奇异值矩阵上可能停滞；factorize 用 converged + 重建 + 正交性三道体检回退",
        measured=(
            f"jacobi: converged={stalled.converged}，相对误差 {err_stalled:.1f} 倍；"
            f"factorize -> {fit.method}（fallback={fit.detail.get('fallback', 0.0):.0f}），相对误差 {err_fit:.2e}"
        ),
        tol="jacobi 相对误差 >= 100 且 factorize 相对误差 <= 1e-12",
        passed=bool(err_stalled >= 100.0 and err_fit <= 1e-12 and fit.detail.get("fallback", 0.0) == 1.0),
        kind="A",
    )
    assert rec.passed, rec.measured


@assertion("A", theory="简化 CT 的投影矩阵按线积分归一（常图像 1 的投影 = 弦长）；角度决定秩", chapter=CHAPTER)
def test_radon_matrix_normalization_and_rank() -> None:
    r"""行和 = 弦长 ``n(|cos|+|sin|)``（1e-12 内）；伴随恒等式 ````
    ``<Ax, y> = <x, A^T y>`` 逐位；8 条角度时秩亏（零空间 > 0），14 条时满秩。

    "CT 一定是病态问题"这句话在这个尺寸上**不成立**：14 条角度时条件数只有几十。"""
    n, na = 10, 8
    A = radon_matrix(n, na, n_bins=na + 2)  # 80 行 = 8 个角度 x 10 条平行射线
    chord = []
    for a in range(na):
        th = np.pi * (a + 0.5) / na
        chord.append(n * (abs(np.cos(th)) + abs(np.sin(th))))
    n_bins = na + 2
    row_err = float(np.max(np.abs(A.sum(axis=1) - np.repeat(np.array(chord), n_bins))))
    x = np.linspace(0.0, 1.0, n * n)
    y = np.cos(np.arange(A.shape[0], dtype=float))
    adj = abs(float(np.dot(A @ x, y)) - float(np.dot(x, A.T @ y)))
    fit8 = factorize(A)
    A14 = radon_matrix(n, 14, n_bins=16)
    fit14 = factorize(A14)
    rank8 = int(np.count_nonzero(fit8.s > fit8.s[0] * 1e-12))
    rank14 = int(np.count_nonzero(fit14.s > fit14.s[0] * 1e-12))
    rec = record(
        label="CT 投影矩阵的行和 / 伴随恒等式 / 两种角度下的秩",
        theory="射线驱动的线积分归一化；A 的秩由角度数决定，角度不足时零空间非平凡",
        measured=(
            f"每行行和与该角度弦长的最大偏差 {row_err:.3e}（{na} 个角度 x {n_bins} 条射线）；"
            f"|<Ax,y>-<x,A^T y>| = {adj:.3e}；8 角度秩 {rank8}/{n * n}，14 角度秩 {rank14}/{n * n}，"
            f"kappa 从 {fit8.s[0] / fit8.s[-1]:.2e} 降到 {fit14.s[0] / fit14.s[-1]:.2e}"
        ),
        tol="行和 <= 1e-12；伴随 <= 1e-12；8 角度秩亏、14 角度满秩",
        passed=bool(row_err <= 1e-12 and adj <= 1e-12 and rank8 < n * n and rank14 == n * n),
        kind="A",
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 6. B 类：不变量
# --------------------------------------------------------------------------- #
@assertion("B", theory="lambda 增大时 ||x_lambda|| 不增、残差不减、自由度不增", chapter=CHAPTER)
def test_tikhonov_monotonicity_in_lambda() -> None:
    prob = synthetic_problem(**PROBLEM)
    lc = problem_lcurve(prob, n_lam=61)
    assert_monotone(lc.x_norm, direction="nonincreasing", label="||x_lambda|| 随 lambda 不增")
    assert_monotone(lc.residual, direction="nondecreasing", label="相对残差随 lambda 不减")
    assert_monotone(lc.dof, direction="nonincreasing", label="有效自由度随 lambda 不增")


@assertion("B", theory="TSVD 的残差随截断阶 k 单调不增；滤波因子只有 0/1 两档", chapter=CHAPTER)
def test_tsvd_monotone_and_filter_range() -> None:
    prob = synthetic_problem(**PROBLEM)
    res = np.array([tsvd(prob.U, prob.s, prob.V, prob.b, k).residual_norm for k in range(0, 21)])
    assert_monotone(res, direction="nonincreasing", label="TSVD 残差随 k 不增")
    f = tsvd_factors(prob.s, 7)
    assert_within(float(np.max(np.abs(f - np.round(f)))), 0.0, rtol=0.0, atol=0.0, label="TSVD 滤波因子只有 0/1")


@assertion("B", theory="收敛区内的滤波因子都落在 [0,1]；有效自由度不超过模态数", chapter=CHAPTER)
def test_filters_stay_in_unit_interval() -> None:
    prob = synthetic_problem(**PROBLEM)
    for om_rel in (0.1, 0.5, 1.0, 1.9):
        f = landweber_factors(prob.s, 500, om_rel / float(prob.s[0] ** 2))
        assert_leq(float(np.max(f)), 1.0, atol=1e-12, label=f"Landweber 上界（omega_rel={om_rel}）")
        # 下界：先夹到 0（负值只可能是舍入），再断言越界量为 0
        assert_leq(float(-np.minimum(float(np.min(f)), 0.0)), 0.0, atol=1e-12, label=f"Landweber 下界（omega_rel={om_rel}）")
    assert_leq(float(np.sum(tikhonov_factors(prob.s, 1e-3))), float(prob.n), label="自由度上界")


@assertion("B", theory="半收敛：Landweber 的误差先单调下降，再被噪声抬起来", chapter=CHAPTER)
def test_landweber_semi_convergence_shape() -> None:
    prob = synthetic_problem(**PROBLEM)
    run = landweber_operator(
        lambda z: prob.A @ z, lambda z: prob.A.T @ z, prob.b, np.zeros(prob.n),
        n_iter=200, omega=1.0 / float(prob.s[0] ** 2), x_true=prob.x_true,
    )
    err = run.error_rel[1:]
    k_best = int(np.argmin(err))
    assert_monotone(err[: k_best + 1], direction="nonincreasing", label="最优步之前的误差不增")
    assert_leq(float(err[k_best]), float(err[-1]), label="最优步处误差小于末步（半收敛）")
    assert_all_finite(run.error_rel, label="误差历史全有限")


@assertion("B", theory="TV 解的目标函数不大于任何候选点（含真解与数据本身）", chapter=CHAPTER)
def test_tv_objective_is_minimal_among_candidates() -> None:
    prob = synthetic_problem(m=32, n=20, kappa=1e4, picard="satisfied", noise_rel=0.0)
    lam = 1e-2
    run = tv_solve(
        lambda z: prob.A @ z, lambda z: prob.A.T @ z, prob.b, np.zeros(prob.n), lam,
        n_iter=400, norm_A=float(prob.s[0]),
    )
    obj = tv_objective(lambda z: prob.A @ z, run.x, prob.b, lam)
    obj_x_true = tv_objective(lambda z: prob.A @ z, prob.x_true, prob.b, lam)
    obj_zero = tv_objective(lambda z: prob.A @ z, np.zeros(prob.n), prob.b, lam)
    assert_leq(obj, min(obj_x_true, obj_zero), rtol=1e-6, label="TV 目标函数不超过候选点")


@assertion("B", theory="幻影像素只有三档、模糊矩阵行和为 1（不改变平均亮度）", chapter=CHAPTER)
def test_phantom_and_blur_structure() -> None:
    img = phantom_image(24)
    levels = np.unique(np.round(img, 12))
    extra = np.setdiff1d(levels, [0.0, 0.5, 0.75, 1.0])
    assert_within(
        float(np.max(np.abs(extra))) if extra.size else 0.0, 0.0, rtol=0.0,
        label="幻影只有四档灰度（含外围 1.0）",
    )
    B = spatial_blur_matrix(24, 2.0)
    assert_within(float(np.max(np.abs(B.sum(axis=1) - 1.0))), 0.0, rtol=0.0, atol=1e-15, label="模糊矩阵行和为 1")
    assert_within(float(np.max(np.abs(B - B.T))), 0.0, rtol=0.0, atol=0.0, label="模糊矩阵对称")


# --------------------------------------------------------------------------- #
# 7. C 类：阈值与幂律
# --------------------------------------------------------------------------- #
@assertion("C", theory="Landweber 的收敛域边界是 omega = 2/sigma_1^2（二分实测）", chapter=CHAPTER)
def test_landweber_divergence_threshold_is_bisected() -> None:
    prob = synthetic_problem(m=32, n=20, kappa=1e6, picard="satisfied", noise_rel=1e-3)
    s1_sq = float(prob.s[0] ** 2)

    def diverges(omega: float) -> bool:
        run = landweber_operator(
            lambda z: prob.A @ z, lambda z: prob.A.T @ z, prob.b, np.zeros(prob.n),
            n_iter=400, omega=float(omega),
        )
        return run.stop == "diverged"

    res = find_threshold(
        diverges, 0.5 / s1_sq, 3.0 / s1_sq, tol=1e-6 * 4.0 / s1_sq,
        assume="false_at_lo", note="判据：400 步内 stop == 'diverged'",
    )
    assert_threshold(
        res, 2.0 / s1_sq, rtol=0.05,
        label="矩阵迭代实测的发散阈值（400 步预算）",
        detail=(
            "二分区间 [0.5, 3]/sigma_1^2。注意它比理论线**高**：|1-omega sigma_1^2| 刚过 1 时"
            "增长因子只有 1+eps，400 步内涨不出可观范数，所以实测阈值依赖步数预算"
        ),
    )
    spectral = find_threshold(
        lambda om: bool(np.max(np.abs(1.0 - float(om) * prob.s ** 2)) > 1.0),
        0.5 / s1_sq, 3.0 / s1_sq, tol=1e-12 * 4.0 / s1_sq, assume="false_at_lo",
        note="判据：谱条件 max|1 - omega sigma_i^2| > 1",
    )
    assert_threshold(
        spectral, 2.0 / s1_sq, rtol=1e-9,
        label="谱判据给出的发散阈值（= 2/sigma_1^2，精确）",
        detail="这条与理论线严格一致；两条一起看才知道实测阈值偏高是步数预算造成的，不是理论错了",
    )


@assertion("C", theory="lambda >> sigma_1 时方差按 lambda^(-4) 下降（f_i ~ s_i^2/lambda^2）", chapter=CHAPTER)
def test_variance_tail_slope_is_minus_four() -> None:
    prob = synthetic_problem(m=32, n=20, kappa=1e4, picard="satisfied", noise_rel=1e-3)
    lam = np.logspace(1.0, 5.0, 25)  # lambda >= 10 sigma_1：渐近区，窗口写进 theory
    ana = bias_variance_analytic(prob.U, prob.s, prob.V, prob.b_exact, prob.x_true, prob.noise_abs, lam)
    fit = fit_order(lam[::-1], ana.variance[::-1])  # 参数递减传入（幂律拟合的口径）
    rec = record(
        label="方差尾部的幂律指数（lambda 从 1e5 降到 10，25 个点）",
        theory="lambda >> sigma_1 时 f_i ~ s_i^2/lambda^2 => Var ~ lambda^(-4)，拟合区间 lambda in [10 sigma_1, 1e5]",
        measured=f"p = {fit.slope:.4f}（窗口 {lam[0]:.0e}..{lam[-1]:.0e}，R^2 = {fit.r2:.6f}）",
        tol="|p + 4| <= 0.05",
        passed=bool(abs(fit.slope + 4.0) <= 0.05),
        kind="C",
    )
    assert rec.passed, rec.measured


@assertion("C", theory="不做正则化时误差放大倍数与 kappa 成正比（幂律指数 1）", chapter=CHAPTER)
def test_unregularized_amplification_scales_like_kappa() -> None:
    kappas = np.array([1e2, 1e3, 1e4, 1e5, 1e6])
    amps = []
    for kap in kappas:
        prob = synthetic_problem(m=32, n=20, kappa=float(kap), picard="satisfied", noise_rel=1e-3)
        amps.append(unregularized_amplification(prob.s, prob.noise_abs, m=prob.m, x_norm=prob.x_norm))
    amps_arr = np.asarray(amps)
    fit = fit_order(kappas[::-1], amps_arr[::-1])
    rec = record(
        label="最坏放大倍数对 kappa 的幂律指数",
        theory="最坏放大 ~ ||e|| sqrt(sum 1/s_i^2)/||x*||；几何谱下 sum 1/s_i^2 由 1/sigma_n^2 主导 => 指数 1",
        measured=f"p = {fit.slope:.4f}（R^2 = {fit.r2:.6f}）；放大倍数 {np.array2string(amps_arr, precision=1)}",
        tol="|p - 1| <= 0.05",
        passed=bool(abs(fit.slope - 1.0) <= 0.05),
        kind="C",
    )
    ratios = []
    for kap, amp in zip(kappas, amps_arr):
        prob = synthetic_problem(m=32, n=20, kappa=float(kap), picard="satisfied", noise_rel=1e-3)
        ratios.append(float(amp / (prob.noise_abs / float(prob.s[-1]) / prob.x_norm)))
    ratios_arr = np.asarray(ratios)
    rec2 = record(
        label="放大倍数与主项 1/sigma_n 的比（多留的模态贡献多少）",
        theory="最坏放大 = ||e||/sigma_n * sqrt(1 + sigma_n^2 sum_{i<n} 1/s_i^2)，第二项随 kappa 递减",
        measured=(
            f"比值 {np.array2string(ratios_arr, precision=4)}（kappa=1e2 高出 "
            f"{(ratios_arr[0] - 1) * 100:.0f}%，kappa=1e6 只高出 {(ratios_arr[-1] - 1) * 100:.0f}%）"
        ),
        tol="比值落在 [1, 2] 且随 kappa 单调不增",
        passed=bool(
            float(np.min(ratios_arr)) >= 1.0
            and float(np.max(ratios_arr)) <= 2.0
            and bool(np.all(np.diff(ratios_arr) <= 1e-12))
        ),
        kind="C",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"
