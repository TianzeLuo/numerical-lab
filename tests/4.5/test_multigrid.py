r"""4.5 章 A/B/C 断言：多重网格的转移代数、平滑性质与"网格无关"的实测边界。

A 类（解析比对）
    * 一维转移算子的三条恒等式：:math:`R=\tfrac12P^{\mathsf T}`、
      :math:`P^{\mathsf T}A_hP=2A_H`、:math:`RA_hP=A_H`（逐位成立，不是"近似"）；
    * 加权 Jacobi 的**逐模态**放大率与理论式 :math:`|1-\omega(1-\cos\theta_k)|`
      （:math:`\omega=2/3` 时即 :math:`|1-\tfrac43\sin^2(\theta_k/2)|`）逐位一致；
    * 两网格因子等于 :math:`\mu_{\rm smooth}^2=1/9`；
    * 平滑子迭代矩阵与 1.5 章的 ``jacobi/gs/sor_iteration_matrix`` 是同一个矩阵
      （口径沿用，不另立一套）；
    * FAS 收敛到 :func:`numcore.pde.multigrid.discrete_newton` 的同一离散解。

B 类（不变量）
    * 两网格校正把 :math:`P` 的值域**精确**消掉；
    * ``mg_apply`` 对 :math:`(x,b)` 是线性的，且与逐列拼出来的循环矩阵逐位一致；
    * 计数器数出来的代价 = 闭式求和（V 的几何级数、W 的 :math:`2^l` 重数）；
    * ``Q = P A_H^{-1} R A_h`` 幂等且 A-对称，以及 ``R P = 0.75I + 0.125 adj`` 的闭式
      （顺带把两个假恒等式 ``R A_h^{-1}=2A_H^{-1}R`` 与 ``RP=I`` 钉在反面）；
    * injection 与 full weighting 不是"差一个常数"；
    * 幂迭代估计器：归一化后的范数**就是**一步放大率，拿相邻范数相除会得到 ≈1 的假因子；
    * 残差触底后的比值不能进因子窗口（否则 FAS 会报出 0.86 这种假因子）。

C 类（实测阈值 / 收敛率）
    * GS 的单步模态放大率在高频半带的最大值 → :math:`1/\sqrt5`，边界误差按 :math:`O(h)` 收缩；
    * 缩放常数错配（:math:`c\ne 1/2`）从"因子变大"退化到"循环发散"；
    * W-cycle 因子更小但每步更贵，降到同一精度**总代价反而更大**；
    * 跨 :math:`h` 的因子扫描：均值/标准差/漂移，以及单层平滑子 :math:`1-\rho\sim h^2` 的斜率；
    * MGCG 的迭代数与 :math:`h` 无关而 CG 是 :math:`O(1/h)`；
    * FAS 的因子与网格无关，而"把非线性当线性解"会停在真残差 O(1) 的假解上。
"""

from __future__ import annotations

import numpy as np

from numcore.linalg.iterative import (
    gs_iteration_matrix,
    jacobi_iteration_matrix,
    laplacian_1d,
    mode_decay_factors,
    sine_eigenvectors_1d,
    sor_iteration_matrix,
)
from numcore.linalg.krylov import conjugate_gradient, pcg
from numcore.linalg.lu import lu_decompose, lu_solve
from numcore.pde import multigrid as mg
from numcore.utils import rng
from numcore.verify import assertion, record

OMEGA = mg.OMEGA_SMOOTH_1D
SIZES = (15, 31, 63, 127)


def _levels(n: int) -> int:
    return int(round(np.log2(n + 1)))


# --------------------------------------------------------------------------- #
# A 类
# --------------------------------------------------------------------------- #
@assertion("A", "一维转移算子恒等式 R=0.5P^T、P^T A_h P=2A_H、R A_h P=A_H 逐位成立", chapter="4.5")
def test_transfer_algebra_1d_exact() -> None:
    r"""三条恒等式在 1D 里是**精确**的（不是 O(h^2) 近似），而且常数不能错。

    :math:`RA_hP=A_H` 是 Galerkin 条件在一维的重合：重离散的粗层算子恰好就是
    Galerkin 粗层算子。二维里这条不再成立（会多出交叉项），所以本章明确只做 1D。
    """
    worst_rt = worst_galerkin = worst_ptap = worst_poisson = 0.0
    for n in SIZES:
        nc = (n - 1) // 2
        P = mg.prolongation_matrix(nc)
        R = mg.restriction_matrix(nc)
        A = mg.poisson_1d(n)
        Ac = mg.poisson_1d(nc)
        worst_rt = max(worst_rt, float(np.max(np.abs(R - 0.5 * P.T))))
        worst_galerkin = max(worst_galerkin, float(np.max(np.abs(R @ A @ P - Ac))))
        worst_ptap = max(worst_ptap, float(np.max(np.abs(P.T @ A @ P - 2.0 * Ac))))
        # 与 1.5 章的 laplacian_1d 口径一致：poisson_1d = laplacian_1d / h^2
        h = 1.0 / (n + 1)
        worst_poisson = max(worst_poisson, float(np.max(np.abs(A - laplacian_1d(n) / (h * h)))))
    rec = record(
        label="转移恒等式最大偏差",
        theory="max|R-0.5P^T|=0，max|R A_h P-A_H|=0，max|P^T A_h P-2A_H|=0",
        measured={
            "R_minus_half_PT": worst_rt,
            "galerkin_defect": worst_galerkin,
            "PtAP_minus_2AH": worst_ptap,
            "poisson_vs_1_5": worst_poisson,
        },
        tol=1e-13,
        passed=max(worst_rt, worst_galerkin, worst_ptap, worst_poisson) <= 1e-13,
    )
    assert rec.passed, rec.measured


@assertion("A", "加权 Jacobi 的逐模态放大率 = |1-omega(1-cos theta_k)|（omega=2/3 时 = |1-4/3 sin^2(theta/2)|）", chapter="4.5")
def test_jacobi_mode_factors_match_theory() -> None:
    r"""Jacobi 的迭代矩阵与 :math:`A_h` 同特征向量，所以"实测"与"理论"应当逐位相同。

    这是本章唯一一处可以做到机器精度的地方——它同时也解释了为什么平滑子的
    故事必须**按模态**讲：逐模态精确，整体范数却由最慢的那个模态主导。
    """
    worst = 0.0
    worst_alt = 0.0
    for n in SIZES:
        A = mg.poisson_1d(n)
        theta = np.arange(1, n + 1, dtype=float) * np.pi / (n + 1)
        for omega in (OMEGA, 1.0, 0.5):
            meas = mg.mode_factors_measured(A, kind="jacobi", omega=omega)[0]
            theo = mg.mode_factors_theory(n, kind="jacobi", omega=omega)
            worst = max(worst, float(np.max(np.abs(meas - theo))))
            alt = np.abs(1.0 - omega * (1.0 - np.cos(theta)))
            worst_alt = max(worst_alt, float(np.max(np.abs(theo - alt))))
            if omega == 1.0:
                # 口径沿用 1.5 章：omega=1 的 Jacobi 就是 |cos theta_k|
                worst_alt = max(worst_alt, float(np.max(np.abs(theo - mode_decay_factors(n, method="jacobi")))))
        quad = np.abs(1.0 - (4.0 / 3.0) * np.sin(theta / 2.0) ** 2)
        worst_alt = max(worst_alt, float(np.max(np.abs(mg.mode_factors_theory(n, kind="jacobi", omega=OMEGA) - quad))))
    rec = record(
        label="Jacobi 逐模态因子：实测 vs 理论 vs 1.5 章口径",
        theory="|1-omega(1-cos theta_k)|；omega=2/3 时 |1-(4/3)sin^2(theta_k/2)|",
        measured={"measured_vs_theory": worst, "theory_vs_closed_form": worst_alt},
        tol=1e-13,
        passed=max(worst, worst_alt) <= 1e-13,
    )
    assert rec.passed, rec.measured


@assertion("A", "平滑子迭代矩阵与 1.5 章的 jacobi/gs/sor_iteration_matrix 是同一个矩阵", chapter="4.5")
def test_smoother_matrices_match_chapter_1_5() -> None:
    r"""本章的 ``smoother_matrix`` 不许另立一套口径：它必须与 1.5 章的迭代矩阵逐位相同。

    ``omega`` 缩放的 Jacobi 用 :math:`I-\omega(I-T_J)` 对照（1.5 章的
    ``jacobi_iteration_matrix`` 只给 :math:`\omega=1` 的情形）。
    """
    worst = 0.0
    for n in (15, 31, 63):
        A = mg.poisson_1d(n)
        TJ = jacobi_iteration_matrix(A)
        for omega in (OMEGA, 1.0, 1.2):
            worst = max(worst, float(np.max(np.abs(mg.smoother_matrix(A, kind="jacobi", omega=omega) - (np.eye(n) - omega * (np.eye(n) - TJ))))))
            worst = max(worst, float(np.max(np.abs(mg.smoother_matrix(A, kind="sor", omega=omega) - sor_iteration_matrix(A, omega)))))
        worst = max(worst, float(np.max(np.abs(mg.smoother_matrix(A, kind="gs") - gs_iteration_matrix(A)))))
    rec = record(
        label="smoother_matrix vs 1.5 章迭代矩阵",
        theory="与 jacobi/gs/sor_iteration_matrix 逐位相同（omega 由 I-omega(I-T_J) 缩放）",
        measured=worst,
        tol=1e-13,
        passed=worst <= 1e-13,
    )
    assert rec.passed, rec.measured


@assertion("A", "两网格因子（nu1=nu2=1, omega=2/3）在多个尺寸上都等于 1/9", chapter="4.5")
def test_two_grid_factor_is_one_ninth() -> None:
    r"""粗层精确求解 + 前后各一次最优 Jacobi 平滑 → :math:`\rho_{\rm TG}=\mu_{\rm smooth}^2=1/9`。

    这是"多重网格比单层迭代快 4.7 倍"的出发点：0.111 对 0.999（n=127）。
    """
    vals = []
    for n in SIZES:
        vals.append(mg.two_grid_factor(mg.mg_hierarchy(n, levels=_levels(n))))
    arr = np.asarray(vals, dtype=float)
    worst = float(np.max(np.abs(arr - 1.0 / 9.0)))
    rec = record(
        label="两网格因子 vs 1/9",
        theory="rho_TG = mu_smooth^2 = (1/3)^2 = 0.111111...",
        measured={"factors": arr.tolist(), "max_abs_dev": worst, "spread": float(arr.max() - arr.min())},
        tol=1e-9,
        passed=worst <= 1e-9,
    )
    assert rec.passed, rec.measured


@assertion("A", "FAS 收敛到与离散牛顿法相同的离散解（误差到机器精度）", chapter="4.5")
def test_fas_matches_newton_reference() -> None:
    r"""FAS 的解必须与 :func:`discrete_newton` 的参考解一致——这是"没有假解"的硬证据。

    顺便量一下离散解与真解 :math:`\sin(\pi x)` 的 :math:`O(h^2)` 差距（它不是 FAS 的错）。
    """
    errs = []
    res = []
    gaps = []
    for n in (15, 31, 63):
        r = mg.measure_fas_factor(n, levels=_levels(n), cycles=16)
        errs.append(float(r.exact_error))
        res.append(float(r.residuals[-1]))
        x = np.arange(1, n + 1, dtype=float) / (n + 1)
        h = 1.0 / (n + 1)
        ref = mg.discrete_newton(h, np.pi**2 * np.sin(np.pi * x) + np.sin(np.pi * x) ** 3)
        gaps.append(float(np.max(np.abs(ref.u - np.sin(np.pi * x)))))
    worst_err = float(np.max(errs))
    worst_res = float(np.max(res))
    rec = record(
        label="FAS vs 离散牛顿参考解",
        theory="FAS 不动点 = 离散方程的解：max|u_FAS - u_Newt| 到机器精度；残差 <= 1e-8",
        measured={"exact_error": errs, "final_residual": res, "discrete_vs_exact": gaps},
        tol=1e-9,
        passed=worst_err <= 1e-9 and worst_res <= 1e-8,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# B 类
# --------------------------------------------------------------------------- #
@assertion("B", "两网格校正把 P 的值域精确消掉： (I - P A_H^{-1} R A_h)(P y) = 0", chapter="4.5")
def test_two_grid_correction_annihilates_coarse_range() -> None:
    r"""粗网格校正对"已经落在粗层能表示的子空间里"的误差是**精确**的。

    这条恒等式是 :math:`RA_hP=A_H` 的直接推论，也是"粗层管低频"的代数版本：
    低频误差在粗层上能表示，于是被一次校正干掉（剩下的只有平滑子的活儿）。
    """
    worst = 0.0
    for n in SIZES:
        levels = _levels(n)
        h = mg.mg_hierarchy(n, levels=levels)
        y = rng(11).standard_normal((n - 1) // 2)
        x = h.prolongations[0] @ y
        corr = mg.coarse_correction_matrix(h) @ x
        worst = max(worst, float(np.max(np.abs(corr))) / max(float(np.max(np.abs(x))), 1e-300))
    rec = record(
        label="粗层校正对 P 值域的零化（相对）",
        theory="R A_h P = A_H ⇒ (I-P A_H^{-1} R A_h) P y = 0",
        measured=worst,
        tol=1e-13,
        passed=worst <= 1e-13,
    )
    assert rec.passed, rec.measured


@assertion("B", "Galerkin 投影性质：Q = P A_H^{-1} R A_h 幂等且 A-对称；(P R) 与 R P 的闭式", chapter="4.5")
def test_variational_projection_properties() -> None:
    r"""R 的正确缩放给出的是一组**投影**性质，但"R P = I"是错的——这一条专门把它钉住。

    真踩过的坑（两个假恒等式，都是"看起来显然"）：

    * :math:`R A_h^{-1}=2A_H^{-1}R` 只在子空间上"半成立"，一般右端相对偏差 1.01；
    * :math:`RP=I` 也是假的：一维里 :math:`RP=0.75I+0.125\,\mathrm{adj}`，
      :math:`\max|RP-I|=0.25`。原因是 :math:`P^{\mathsf T}P\ne 2I`
      （列范数平方和 1.5、相邻列内积 0.25）。

    真正成立且有用的是：:math:`R A_hP=A_H` ⇒ :math:`Q=PA_H^{-1}RA_h` 幂等，
    且关于 :math:`A_h` 内积对称（Galerkin 投影）。
    """
    worst_q = 0.0
    worst_rp = 0.0
    worst_rp_identity = 0.0
    for n in (15, 31, 63):
        nc = (n - 1) // 2
        h = mg.mg_hierarchy(n, levels=2)
        P = mg.prolongation_matrix(nc)
        R = mg.restriction_matrix(nc)
        A = mg.poisson_1d(n)
        Q = np.eye(n) - mg.coarse_correction_matrix(h)
        worst_q = max(worst_q, float(np.max(np.abs(Q @ Q - Q))))
        x, y = rng(3).standard_normal(n), rng(5).standard_normal(n)
        asym = abs(float(x @ (A @ (Q @ y))) - float((Q @ x) @ (A @ y)))
        worst_q = max(worst_q, asym / max(abs(float(x @ (A @ (Q @ y)))), 1e-300))
        adj = np.zeros((nc, nc), dtype=float)
        for j in range(nc - 1):
            adj[j, j + 1] = adj[j + 1, j] = 1.0
        closed = 0.75 * np.eye(nc) + 0.125 * adj
        worst_rp = max(worst_rp, float(np.max(np.abs(R @ P - closed))))
        worst_rp_identity = max(worst_rp_identity, float(np.max(np.abs(R @ P - np.eye(nc)))))
    rec = record(
        label="Galerkin 投影 vs 两个假恒等式",
        theory="Q^2=Q、A-对称（偏差 0）；R P = 0.75I+0.125 adj（偏差 0）；max|RP-I| = 0.25 != 0",
        measured={"projection_defect": worst_q, "rp_closed_form": worst_rp,
                  "rp_vs_identity": worst_rp_identity},
        tol="投影偏差<=1e-13；闭式<=1e-15；|RP-I| 必须 ≈0.25（不是 0）",
        passed=worst_q <= 1e-13 and worst_rp <= 1e-15 and abs(worst_rp_identity - 0.25) <= 1e-12,
    )
    assert rec.passed, rec.measured


@assertion("B", "反例留档：两种「含逆交换律」写法都不成立，真命题是 A_H^{-1} R A_h P = I", chapter="4.5")
def test_inverse_commutation_claims_are_false() -> None:
    r"""把两个"看起来显然"的含逆恒等式钉在反面，免得后来者又写进文档。

    一般右端上 :math:`R A_h^{-1}=2A_H^{-1}R` 的相对偏差 1.01；
    即便在好子空间 :math:`A_h\,\mathrm{range}(P)` 上，:math:`RA_h^{-1}=A_H^{-1}R`
    也差了 0.61（因为 :math:`RP\ne I`）。真正成立的是 :math:`A_H^{-1}RA_hP=I`，
    它等价于本文件 A 类里的 :math:`RA_hP=A_H`。
    """
    n = 31
    nc = (n - 1) // 2
    P = mg.prolongation_matrix(nc)
    R = mg.restriction_matrix(nc)
    A = mg.poisson_1d(n)
    Ac = mg.poisson_1d(nc)
    luA, luAc = lu_decompose(A), lu_decompose(Ac)
    b = rng(7).standard_normal(n)
    left = R @ lu_solve(luA, b)
    right = 2.0 * lu_solve(luAc, R @ b)
    rel_general = float(np.max(np.abs(left - right))) / max(float(np.max(np.abs(left))), 1e-300)
    y = rng(9).standard_normal(nc)
    bs = A @ (P @ y)
    abs_sub = float(np.max(np.abs(R @ lu_solve(luA, bs) - lu_solve(luAc, R @ bs))))
    true_form = float(np.max(np.abs(lu_solve(luAc, R @ bs) - y)))
    rec = record(
        label="含逆交换律的反例留档",
        theory="一般右端 rel≈1.0；子空间上 abs≈0.61；真命题 A_H^{-1} R A_h P = I 偏差 0",
        measured={"general_rhs_rel": rel_general, "subspace_abs": abs_sub, "true_form_abs": true_form},
        tol="两个假命题偏差 >=0.1，真命题 <=1e-12",
        passed=rel_general >= 0.1 and abs_sub >= 0.1 and true_form <= 1e-12,
    )
    assert rec.passed, rec.measured


@assertion("B", "mg_apply 对 (x,b) 线性，且与逐列拼出的循环矩阵逐位一致", chapter="4.5")
def test_mg_apply_is_linear_and_matches_matrix() -> None:
    r"""一次循环是一个**线性**算子（固定参数、固定递归）：这一点必须被验，不能被假设。

    做法：把 :func:`mg_apply` 作用在单位向量上拼出矩阵 :math:`E`，再与逐次调用对拍；
    同时验叠加性 :math:`E(x_1+x_2, b_1+b_2)=E(x_1,b_1)+E(x_2,b_2)`（含右端非零的仿射情形）。
    """
    n = 31
    h = mg.mg_hierarchy(n, levels=_levels(n), nu1=1, nu2=1)
    E = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        E[:, j] = mg.mg_apply(h, e, None)
    x1, x2 = rng(1).standard_normal(n), rng(2).standard_normal(n)
    b1, b2 = rng(3).standard_normal(n), rng(4).standard_normal(n)
    worst = 0.0
    for x, b in ((x1, None), (x2, b2), (x1 + x2, b1 + b2)):
        ref = mg.mg_apply(h, x, b)
        got = E @ x + (mg.mg_apply(h, np.zeros(n), b) if b is not None else 0.0)
        worst = max(worst, float(np.max(np.abs(ref - got))) / max(float(np.max(np.abs(ref))), 1e-300))
    worst = max(worst, float(np.max(np.abs(mg.mg_apply(h, np.zeros(n), b1) + mg.mg_apply(h, np.zeros(n), b2)
                                            - mg.mg_apply(h, np.zeros(n), b1 + b2)))))
    rec = record(
        label="循环算子线性性与矩阵一致性（相对）",
        theory="E 线性：E(x1+x2,b1+b2)=E(x1,b1)+E(x2,b2)，且 E 的列就是 mg_apply(e_j)",
        measured=worst,
        tol=1e-12,
        passed=worst <= 1e-12,
    )
    assert rec.passed, rec.measured


@assertion("B", "计数器数出来的每周期代价 = V/W 循环的闭式求和", chapter="4.5")
def test_work_counter_matches_closed_form() -> None:
    r"""代价必须能对上一个独立的闭式：V 是几何级数，W 的每层被访问 :math:`2^l` 次。

    :math:`W_V=\sum_{l=0}^{L-2}(\nu_1+\nu_2)\frac{n_l}{n_0}+s\frac{n_{L-1}}{n_0}`，
    W-cycle 把第 :math:`l` 层的重数换成 :math:`2^l`。对不上就说明计数器和递归不同步
    （本章真踩过：``work`` 一度是"多个周期的总和"，看起来只是数字偏大）。
    """
    worst = 0.0
    rows = []
    for n in (15, 31, 63):
        for cycle, gamma in (("v", 1), ("w", 2)):
            levels = _levels(n)
            h = mg.mg_hierarchy(n, levels=levels, cycle=cycle, nu1=1, nu2=1, coarse_sweeps=24)
            r = mg.cycle_factor(h, iters=2, tol=0.0)
            counted = float(r.work)  # work 已经是每周期值（计数器总和 / 周期数）
            sizes = np.asarray([g.n for g in h.grids], dtype=float)
            closed = 0.0
            for lv in range(levels - 1):
                closed += (gamma**lv) * 2.0 * sizes[lv] / sizes[0]
            closed += (gamma ** (levels - 1)) * 24.0 * sizes[-1] / sizes[0]
            rows.append((n, cycle, counted, closed))
            worst = max(worst, abs(counted - closed) / max(closed, 1e-300))
    rec = record(
        label="计数器 vs 闭式代价（相对）",
        theory="V: sum_l 2*n_l/n_0 + 24*n_L/n_0；W: 重数 2^l",
        measured={"worst_rel": worst, "rows": rows[:4]},
        tol=1e-12,
        passed=worst <= 1e-12,
    )
    assert rec.passed, rec.measured


@assertion("B", "幂迭代估计器的陷阱：归一化后的范数就是一步放大率，相邻相除会得到假因子 ≈1", chapter="4.5")
def test_cycle_factor_estimator_trap() -> None:
    r"""本章真踩过的坑：把"每步归一化后的范数序列"再相邻相除，得到的是比值之比 ≈1。

    正确读法：序列的第 :math:`k` 项**本身就是**第 :math:`k` 步的放大率。这条断言把
    错法与对法同时钉住——错法必须真的错（≈1），对法必须对上显式矩阵的谱半径。
    """
    n = 15
    h = mg.mg_hierarchy(n, levels=_levels(n))
    E = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        E[:, j] = mg.mg_apply(h, e, None)
    rho = float(max(abs(complex(v)) for v in np.asarray(_eigvals(E)).ravel()))
    x = rng(45045).standard_normal(n)
    x = x / float(np.sqrt(x @ x))
    norms = [1.0]
    for _ in range(24):
        x = mg.mg_apply(h, x, None)
        nx = float(np.sqrt(x @ x))
        norms.append(nx)
        x = x / nx
    arr = np.asarray(norms[1:], dtype=float)
    correct = float(np.exp(np.mean(np.log(arr[-8:]))))
    naive = float(np.exp(np.mean(np.log((arr[-8:] / arr[-9:-1])))))
    r = mg.cycle_factor(h, iters=24)
    rec = record(
        label="估计器：对法 vs 错法 vs 显式矩阵谱半径",
        theory="归一化后每步范数即放大率（几何平均 -> rho(E)）；相邻相除会得到 ≈1 的假值",
        measured={
            "rho_of_matrix": rho,
            "cycle_factor_correct": correct,
            "naive_ratio_of_ratios": naive,
            "cycle_factor_api": r.factor,
        },
        tol="|correct-rho|<=2e-2 且 |naive-1|<=0.05",
        passed=abs(correct - rho) <= 2e-2 and abs(naive - 1.0) <= 0.05 and abs(r.factor - correct) <= 1e-12,
    )
    assert rec.passed, rec.measured


def _eigvals(E: np.ndarray) -> np.ndarray:
    """用 1.5 章的幂法/特征值工具拿谱半径（P2 禁止 numpy.linalg）。"""
    from numcore.linalg.krylov import tridiag_eigenvalues  # noqa: PLC0415

    n = E.shape[0]
    if np.allclose(E, np.triu(E)) and np.allclose(E, E.T):
        return tridiag_eigenvalues(np.diag(E), np.diag(E, 1).copy())
    # 非对称矩阵：走特征多项式的幂法估计（只需最大模特征值）
    x = rng(99).standard_normal(n)
    lam = 0.0
    for _ in range(400):
        y = E @ x
        ny = float(np.sqrt(y @ y))
        if ny == 0.0:
            return np.asarray([0.0])
        x = y / ny
        lam = float(x @ (E @ x))
    return np.asarray([abs(lam)])


@assertion("B", "injection 与 full weighting 不是差一个常数：R_inj != c P^T", chapter="4.5")
def test_injection_is_not_a_scaled_transpose() -> None:
    r"""injection 只取 :math:`r_{2j+1}`，权重和是 1 而不是 2 —— 它破坏的是**伴随关系**本身。

    伴随关系一破，Galerkin 恒等式随之失效，两网格因子从 0.111 退到 0.332（实测）。
    """
    worst = 0.0
    fw = []
    inj = []
    for n in (31, 63, 127):
        nc = (n - 1) // 2
        P = mg.prolongation_matrix(nc)
        Rf = mg.restriction_matrix(nc, kind="full_weighting")
        Ri = mg.restriction_matrix(nc, kind="injection")
        worst = max(worst, float(np.max(np.abs(Rf - 0.5 * P.T))))
        # 最优常数下的最小二乘残差：即便把 c 调到最好，injection 也不是 c P^T
        c = float(np.sum(Ri * P.T) / np.sum(P.T * P.T))
        worst = max(worst, float(np.max(np.abs(Ri - c * P.T))) / max(float(np.max(np.abs(Ri))), 1e-300))
        fw.append(mg.two_grid_factor(mg.mg_hierarchy(n, levels=_levels(n))))
        inj.append(mg.two_grid_factor(
            mg.mg_hierarchy(n, levels=_levels(n), restrict_kind="injection")
        ))
    ratio = float(np.min(np.asarray(inj) / np.asarray(fw)))
    rec = record(
        label="injection 的伴随缺口与两网格因子代价",
        theory="R = 0.5 P^T（满权）才使 R A_h P = A_H；injection 下两网格因子显著变大（实测约 3 倍）",
        measured={"worst_asymmetry": worst, "fw": fw, "inj": inj, "min_ratio": ratio},
        tol="asymmetry>=0.2 且 ratio>=2.5",
        passed=worst >= 0.2 and ratio >= 2.5,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# C 类
# --------------------------------------------------------------------------- #
@assertion("C", "GS 的单步模态放大率在高频半带最大值为 1/sqrt(5)，边界误差按 O(h) 收缩", chapter="4.5")
def test_gs_symbol_and_boundary_error() -> None:
    r"""GS 的实测放大率趋近无穷网格符号 :math:`1/\sqrt{5-4\cos\theta}`（高频半带最大 :math:`1/\sqrt5`）。

    有限网格上差的是边界效应，误差按 :math:`O(h)` 收缩——实测比值 2.006/2.008/2.004。
    顺便量一下非正规性的后果：特征值口径给出 0.9994（n=127），实测只有 0.4453。
    """
    maxes = []
    gaps = []
    young = []
    for n in SIZES:
        A = mg.poisson_1d(n)
        theta = np.arange(1, n + 1, dtype=float) * np.pi / (n + 1)
        meas = mg.mode_factors_measured(A, kind="gs")[0]
        symbol = 1.0 / np.sqrt(5.0 - 4.0 * np.cos(theta))
        high = theta >= np.pi / 2.0
        maxes.append(float(np.max(meas[high])))
        gaps.append(float(np.max(np.abs(meas - symbol))))
        young.append(float(np.max(mode_decay_factors(n, method="gs")[high])))
    ratios = [gaps[k] / gaps[k + 1] for k in range(len(gaps) - 1)]
    worst = abs(maxes[-1] - 1.0 / np.sqrt(5.0))
    ok = worst <= 5e-3 and all(1.7 <= r <= 2.3 for r in ratios) and maxes[-1] <= 0.46
    rec = record(
        label="GS 高频放大率与边界误差收缩",
        theory="max_high |g(theta)| -> 1/sqrt(5)=0.447214；边界误差 ~ O(h)（比值 -> 2）",
        measured={"max_high": maxes, "gap": gaps, "gap_ratios": ratios, "young_eigen": young,
                  "max_abs_dev": worst},
        tol="|max-1/sqrt5|<=5e-3 且 比值∈[1.7,2.3]",
        passed=ok,
    )
    assert rec.passed, rec.measured


@assertion("C", "缩放常数错配：c=1/2 收敛，c!=1/2 因子变大，c>=3/4 循环发散", chapter="4.5")
def test_scaling_mismatch_diverges() -> None:
    r"""限制算子的缩放常数必须与粗层算子配对：本章真跑过"看起来只是慢一点"到"直接发散"。

    实测（n=63）：c=0.25 → 0.892（欠校正）、0.5 → 0.286、0.75 → 2.043、1.0 → 6.038、
    2.0 → 53.4。Galerkin 缺口 |c-0.5| 是最直接的诊断量。
    """
    n = 63
    probe = mg.scaling_probe(n, levels=_levels(n), factors=(0.25, 0.5, 0.75, 1.0, 2.0))
    fac = np.asarray(probe.factors, dtype=float)
    cyc = np.asarray(probe.cycle_factors, dtype=float)
    defect = np.asarray(probe.galerkin_defect, dtype=float)
    i_half = int(np.argmin(np.abs(fac - 0.5)))
    ok = (
        cyc[i_half] < 1.0
        and np.all(cyc[np.abs(fac - 0.5) > 1e-12] > cyc[i_half])
        and np.all(cyc[fac >= 0.75] > 1.0)
        and defect[i_half] == 0.0
        and np.all(np.abs(defect - np.abs(fac - 0.5) * 2.0) <= 1e-12)
    )
    rec = record(
        label="缩放常数 c 的因子扫描",
        theory="|c-0.5|=0 时循环收敛；c>0.5 过校正可发散（粗层校正被放大 2c 倍）",
        measured={"c": fac.tolist(), "cycle_factor": cyc.tolist(), "galerkin_defect": defect.tolist(),
                  "worst": float(probe.worst)},
        tol="c=0.5 最小且 <1；c>=0.75 全部 >1；defect=|2c-1|",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("C", "W-cycle 因子更小但每步更贵：降到 1e-8 的总代价 V 反而更低", chapter="4.5")
def test_w_cycle_factor_smaller_but_costlier() -> None:
    r"""实测（n=63，L=6，nu=1）：V 因子 0.1887 / 每周期 4.16；W 因子 0.1145 / 每周期 21.4。

    「因子更小」被「一步更贵」吃掉：总代价 11x4.16 vs 9x21.4，V 便宜约 4.2 倍。
    """
    n = 63
    levels = _levels(n)
    rv = mg.measure_cycle_factor(n, levels=levels, cycle="v", iters=24)
    rw = mg.measure_cycle_factor(n, levels=levels, cycle="w", iters=24)
    cost_v = float(rv.iters_to_tol) * float(rv.work)
    cost_w = float(rw.iters_to_tol) * float(rw.work)
    ratio = cost_w / max(cost_v, 1e-300)
    ok = (
        rw.factor < rv.factor
        and rw.work >= 3.0 * rv.work
        and cost_w > cost_v
        and ratio >= 2.0
        and rv.work < 6.0
    )
    rec = record(
        label="V vs W：因子、每周期代价、总代价",
        theory="W 因子接近两网格（更小），但每周期代价约 2^L 级；总代价由 因子 与 代价 共同决定",
        measured={
            "v_factor": rv.factor, "w_factor": rw.factor,
            "v_work": rv.work, "w_work": rw.work, "work_ratio": rw.work / rv.work,
            "v_cycles": int(rv.iters_to_tol), "w_cycles": int(rw.iters_to_tol),
            "cost_v": cost_v, "cost_w": cost_w, "cost_ratio": ratio,
        },
        tol="w_factor<v_factor 且 w_work>=3*v_work 且 cost_w/cost_v>=2",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("C", "跨 h 的因子扫描：V-cycle 因子均值/标准差/漂移，单层平滑子 1-rho ~ h^2 的斜率", chapter="4.5")
def test_factor_scan_grid_independence() -> None:
    r"""「网格无关」的定量版本：均值 0.1826、标准差 1.1e-2、漂移 +2.9e-2（n=15..127）。

    同一序列上单层平滑子从 0.9872 爬到 0.9997，:math:`-\log(1-\rho)` 对 :math:`\log h`
    的斜率实测 -1.889（理论渐近值 2，小网格还没进渐近区）。
    两网格因子在整条序列上逐位相同（标准差 0）。
    """
    scan = mg.factor_scan((3, 4, 5, 6, 7), cycle="v", two_grid_max_levels=7)
    mg_it, sm_it = scan.iterations_for(1e-8)
    ok = (
        scan.mg_factor_std <= 0.02
        and abs(scan.mg_factor_drift) <= 0.05
        and float(np.max(scan.factor_two_grid) - np.min(scan.factor_two_grid)) <= 1e-12
        and -2.2 <= scan.smoother_trend <= -1.6
        and np.all(np.diff(scan.factor_smoother) > 0.0)
        and int(sm_it[-1]) >= 20 * int(mg_it[-1])
    )
    rec = record(
        label="因子扫描：MG 的平坦 vs 平滑子的 h^2 恶化",
        theory="MG 因子与 h 无关（std 小、漂移小）；1-rho_smoother ~ h^2 ⇒ 斜率 -2（渐近）",
        measured={
            "n": scan.n_fine.tolist(),
            "factor_mg": scan.factor_mg.tolist(),
            "factor_two_grid": scan.factor_two_grid.tolist(),
            "factor_smoother": scan.factor_smoother.tolist(),
            "mean": scan.mg_factor_mean, "std": scan.mg_factor_std, "drift": scan.mg_factor_drift,
            "smoother_slope": scan.smoother_trend,
            "iters_mg": mg_it.tolist(), "iters_smoother": sm_it.tolist(),
        },
        tol="std<=0.02 且 |drift|<=0.05 且 两网格 spread=0 且 斜率∈[-2.2,-1.6]",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("C", "MGCG 的迭代数与 h 无关（7 次），CG 是 O(1/h)，加速比 ~1/h", chapter="4.5")
def test_mgcg_iterations_grid_independent() -> None:
    r"""同一右端、同一精度：CG 的迭代数恰好是 n（15/31/63/127/255），MGCG 恒为 7。

    注意这是**迭代数比**，不是时间比：MGCG 每次迭代要多做一个循环（约 4 个最细层等价代价），
    所以 n 很小时它并不划算——两个口径都要写在报告里。
    """
    ns = (15, 31, 63, 127, 255)
    cg_it = []
    mgcg_it = []
    for n in ns:
        A = mg.poisson_1d(n)
        x = np.arange(1, n + 1, dtype=float) / (n + 1)
        b = rng(7).standard_normal(n) + 0.3 * np.sin(np.pi * x)
        direct = conjugate_gradient(A, b, tol=1e-8, maxiter=4 * n + 20)
        acc = pcg(A, b, precond=mg.MGPrecond(n, levels=_levels(n)), tol=1e-8, maxiter=200)
        cg_it.append(int(direct.n_iter))
        mgcg_it.append(int(acc.n_iter))
    cg_arr = np.asarray(cg_it, dtype=float)
    mg_arr = np.asarray(mgcg_it, dtype=float)
    ns_arr = np.asarray(ns, dtype=float)
    speed = cg_arr / mg_arr
    ok = (
        np.array_equal(cg_arr, ns_arr)
        and int(np.max(mg_arr)) <= 10
        and int(np.std(mg_arr)) <= 1
        and speed[-1] >= 30.0
        and np.all(np.diff(speed) > 0)
    )
    rec = record(
        label="CG vs MGCG 迭代数（tol=1e-8）",
        theory="CG ~ O(h^-1)（sqrt(kappa)）；MGCG 迭代数 O(1)；加速比按迭代数 ~ 1/h",
        measured={"n": ns, "cg_iters": cg_it, "mgcg_iters": mgcg_it, "speedup": speed.tolist(),
                  "mgcg_std": float(np.std(mg_arr))},
        tol="cg=n 且 mgcg<=10 且 加速比最大>=30",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("C", "FAS 的因子与网格无关；把非线性当线性解会停在真残差 O(1) 的假解上", chapter="4.5")
def test_fas_factor_and_false_solution() -> None:
    r"""FAS 实测因子 0.154/0.173/0.186/0.189（n=15..127），与线性 V-cycle 同一量级。

    反例（``nonlinear=False``，n=63）：线性残差降到 4.1e-10（按它自己的尺子"收敛了"），
    真残差却停在 5.60，解偏了 7.9e-2——两个残差的比值 1.4e10。这就是"残差小不等于解对"。
    """
    fac = []
    for n in (15, 31, 63, 127):
        fac.append(mg.measure_fas_factor(n, levels=_levels(n), cycles=16).factor)
    arr = np.asarray(fac, dtype=float)
    bad = mg.measure_fas_factor(63, levels=_levels(63), cycles=16, nonlinear=False)
    gap = float(bad.false_solution_gap)
    ok = (
        float(np.std(arr)) <= 0.02
        and float(np.mean(arr)) <= 0.25
        and float(bad.residuals[-1]) >= 1.0
        and float(bad.linear_residual) <= 1e-8
        and float(bad.exact_error) >= 5e-2
        and gap >= 1e9
    )
    rec = record(
        label="FAS 因子（跨 h）与线性化反例",
        theory="FAS 因子与 h 无关；线性化残差收敛到 1e-10 时真残差仍 O(1)，"
               "假解误差 ~8e-2（真解幅度 1）",
        measured={
            "n": [15, 31, 63, 127], "fas_factor": arr.tolist(),
            "mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "false_true_residual": float(bad.residuals[-1]),
            "false_linear_residual": float(bad.linear_residual),
            "false_solution_error": float(bad.exact_error),
            "gap": gap,
        },
        tol="std<=0.02 且 mean<=0.25 且 真残差>=1 且 线性残差<=1e-8 且 gap>=1e9",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("B", "残差触底后的比值不进因子窗口：窗口估计器给出真因子而不是 ≈1", chapter="4.5")
def test_window_factor_ignores_roundoff_floor() -> None:
    r"""FAS 的残差会撞上舍入地板：直接拿最后几个比值求几何平均会报出 0.86 这种假因子。

    这里用一条合成的"先按 0.19 衰减、后触底"的序列把窗口估计器钉住：
    它必须给出 0.19（而不是 ≈1），并把真正用到的循环区间作为窗口交出来。
    """
    decay = 0.19
    head = 60.0 * decay ** np.arange(12, dtype=float)
    tail = np.full(10, 4.0e-11)
    arr = np.concatenate([head, tail])
    factor, window = mg.window_factor(arr, skip=1, tail=4)
    naive = float(np.exp(np.mean(np.log(arr[-4:] / arr[-5:-1]))))
    ok = abs(factor - decay) <= 0.02 and window[1] <= 12 and abs(naive - 1.0) <= 0.05
    rec = record(
        label="窗口因子 vs 触底后的朴素比值",
        theory="触底段比值 ~1；窗口估计器跳过地板与初始瞬态后应给出真衰减率 0.19",
        measured={"window_factor": factor, "window": list(window), "naive_tail_ratio": naive},
        tol="|factor-0.19|<=0.02 且 |naive-1|<=0.05",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured
