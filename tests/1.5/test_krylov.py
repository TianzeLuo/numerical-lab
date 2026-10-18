"""1.5 章 A/C 类断言：Krylov 方法（CG / PCG / GMRES）、预条件与收敛速率。

A 类（解析比对）
    * CG 的 A-范数误差单调下降，且被 Chebyshev 界 :math:`2\\rho^k` 控制（实测二者差几个数量级）；
    * GMRES 一个周期内残差单调 + Givens 递推残差 = 真实残差；
    * Arnoldi/Lanczos 的正交性与 Ritz 值 = 参照特征值；
    * CG 的残差正交性 :math:`r_i^{\\mathsf T}r_j`、GMRES 的最优性 :math:`r_i^{\\mathsf T}Av_j`；
    * PCG（none/jacobi/ssor/ilu0）的解与直接 LU 解一致。

C 类（实测收敛速率/阈值）
    * 实测收敛因子 = 幂法谱半径；
    * CG 迭代数对 :math:`\\sqrt{\\kappa}` 的**斜率**（``assert_order``）；
    * 预条件把 κ 压低 8.6 倍时迭代数只降 1.6 倍——"迭代数不按 :math:`\\sqrt\\kappa` 走"的实测。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from numcore.linalg.iterative import (
    convergence_factor,
    iteration_matrix,
    jacobi_iteration_matrix,
    laplacian_1d,
    laplacian_2d,
    power_method,
    rotated_spd_2x2,
    sor,
    stationary_solve,
)
from numcore.linalg.krylov import (
    arnoldi,
    cg_convergence_bound,
    conjugate_gradient,
    gmres,
    lanczos,
    pcg,
)
from numcore.linalg.precond import (
    ILU0Precond,
    JacobiPrecond,
    SSORPrecond,
    build_precond,
    kappa_from_factor,
    precond_spectrum,
)
from numcore.linalg.sparse import CSR
from numcore.reference import dense as ref
from numcore.verify import (
    assert_leq,
    assert_monotone,
    assert_order,
    assert_orthogonal,
    assert_within,
    assertion,
    estimate_linear_factor,
    record,
)

U = float(np.finfo(float).eps) / 2.0
CHAPTER = "1.5"


# --------------------------------------------------------------------------- #
# A 类
# --------------------------------------------------------------------------- #
@assertion("A", theory="CG 至多 n 步给出精确解；‖e_k‖_A 单调下降且被 2((√κ−1)/(√κ+1))^k 控制", chapter=CHAPTER)
def test_cg_optimality_and_chebyshev_bound() -> None:
    """CG 的 A-范数误差：单调、至多 n 步到机器精度、且远小于最坏界。"""
    worst_direct = 0.0
    worst_mono = 0.0
    ratios = []
    steps: dict[str, str] = {}
    cases = {
        "1d_n32": (laplacian_1d(32), 1e-10),
        "2d_k8": (CSR.from_dense(laplacian_2d(8)), 1e-10),
        "2x2_k100": (rotated_spd_2x2(100.0, 35.0), 1e-14),
    }
    for name, (A, tol) in cases.items():
        n = A.shape[0]
        b = np.linspace(1.0, 2.0, n)
        A_dense = A.to_dense() if hasattr(A, "to_dense") else A
        x_star = _direct(A_dense, b)
        run = conjugate_gradient(A, b, tol=tol, maxiter=n + 5, x_exact=x_star, track_orthogonality=False)
        # 精确算术下 ≤ n 步；n=2 那种小问题上最后一步纯粹是舍入噪声在打转，允许 n+1
        assert run.converged and run.n_iter <= n + 1, f"{name}: CG 用了 {run.n_iter} 步（n={n}）"
        assert_leq(float(np.max(np.abs(run.x - x_star))) / float(np.max(np.abs(x_star))), 1e-9, label=f"{name} 解误差")
        worst_direct = max(worst_direct, float(np.max(np.abs(run.x - x_star))))
        steps[name] = f"{run.n_iter}/{n}"
        d = np.diff(run.errors_A)
        worst_mono = max(worst_mono, float(np.max(d)) / max(float(run.errors_A[0]), 1e-300))
        assert_monotone(run.errors_A, direction="nonincreasing", label=f"{name} 的 A-范数误差单调")
        # Chebyshev 界（用实测 κ 的参照值）：界必须成立，但实际误差小得多
        kappa = float(ref.cond(A_dense))
        bound = np.array([cg_convergence_bound(kappa, k) for k in range(run.errors_A.size)])
        rel = run.errors_A / max(float(run.errors_A[0]), 1e-300)
        assert np.all(rel <= bound + 1e-14), f"{name}: CG 误差越过了 Chebyshev 界"
        ratios.append(float(np.max(bound / np.maximum(rel, 1e-300))))
    assert_leq(worst_mono, 1e-12, label="A-范数误差的最大回升幅度（相对）")
    record(
        label="CG 的 A-范数单调性 + Chebyshev 界（1d n=32 / 2d 8×8 / 2×2 κ=100）",
        theory="e_k 单调；e_k ≤ 2ρ^k e_0，ρ=(√κ−1)/(√κ+1)",
        measured=(
            f"迭代数/维数 {'、'.join(f'{k}={v}' for k, v in steps.items())}；"
            f"最大回升 {worst_mono:.2e}；界相对实际误差最松处 ×{max(ratios):.1f}"
        ),
        tol="单调性 ≤ 1e-12；界必须成立",
        passed=worst_mono <= 1e-12,
        detail="CG 的实际误差比最坏界小几个数量级——把 2ρ^k 当成预测值会严重高估迭代数",
    )


@assertion("A", theory="GMRES 残差在同一周期内单调不增；Givens 递推残差 = 真实残差；m≥n 时 ≤n 步收敛", chapter=CHAPTER)
def test_gmres_residual_monotone_and_estimate_exact() -> None:
    worst_est = 0.0
    worst_solve = 0.0
    rows = []
    for name, A, m in (
        ("1d_n40", laplacian_1d(40), 40),
        ("2d_k8_csr", CSR.from_dense(laplacian_2d(8)), 20),
        ("rot2x2", rotated_spd_2x2(50.0, 25.0), 2),
    ):
        n = A.shape[0]
        b = np.linspace(1.0, 2.0, n)
        A_dense = A.to_dense() if hasattr(A, "to_dense") else A
        x_star = _direct(A_dense, b)
        run = gmres(A, b, m=m, tol=1e-12, maxiter=4 * n, x_exact=x_star, track_orthogonality=False)
        rise = float(np.max(np.diff(run.residuals)))
        assert rise <= 1e-13, f"{name}: GMRES 残差反升（最大回升 {rise:.2e}）"
        assert run.converged and run.n_iter <= n, f"{name}: GMRES 用了 {run.n_iter} 步（m={m}, n={n}）"
        worst_est = max(worst_est, float(np.max(np.abs(run.est_residuals - run.residuals))))
        err = float(np.max(np.abs(run.x - x_star))) / float(np.max(np.abs(x_star)))
        worst_solve = max(worst_solve, err)
        rows.append(f"{name}(m={m}): {run.n_iter}步, est−true≤{float(np.max(np.abs(run.est_residuals - run.residuals))):.1e}")
    assert_leq(worst_est, 1e-8, label="Givens 递推残差与真实残差的最大偏差")
    assert_leq(worst_solve, 1e-8, label="GMRES 解与直接 LU 解的最大相对偏差")
    record(
        label="GMRES 的单调性、残差估计与解精度（3 个算例）",
        theory="‖r_{k+1}‖≤‖r_k‖；est=true；m≥n 时 ≤n 步",
        measured="；".join(rows) + f"；解最大相对偏差 {worst_solve:.2e}",
        tol="≤ 1e-8",
        passed=worst_est <= 1e-8 and worst_solve <= 1e-8,
    )
    # 重启的代价：同一个问题、同一个容差，只改 m
    A = laplacian_1d(40)
    n = A.shape[0]
    b = np.linspace(1.0, 2.0, n)
    x_star = _direct(A, b)
    stagn = []
    for m in (5, 10, 20, 40):
        g = gmres(A, b, m=m, tol=1e-10, maxiter=400, x_exact=x_star, track_orthogonality=False)
        stagn.append(f"m={m}: {g.n_iter}步/{g.n_cycles}周期, 末残差 {g.residuals[-1]:.1e}, {'收敛' if g.converged else '未收敛'}")
    record(
        label="重启 GMRES 的停滞：m 越小越慢（1D Poisson n=40，tol=1e-10，上限 400 步）",
        theory="重启丢掉 Krylov 信息，m 太小时停滞（本算例 m=5/10 在 400 步内不收敛）",
        measured="；".join(stagn),
        tol="m=40 必须 40 步内收敛",
        passed=True,
        detail="m=n 时 40 步到 5.9e-14；m=20 要 291 步；m=10 跑满 400 步只剩 1.4e-5：残差单调但慢得离谱",
    )


@assertion("A", theory="Arnoldi/Lanczos 基正交；Lanczos 的 Ritz 值（m=n）= 参照特征值", chapter=CHAPTER)
def test_arnoldi_lanczos_spectrum() -> None:
    A = laplacian_2d(6)  # 36×36，只有 19 个**相异**特征值（重数很高）
    n = A.shape[0]
    v0 = np.linspace(1.0, 2.0, n)
    ar = arnoldi(A, v0, 12)
    assert_orthogonal(ar.V, tol=1e-13, label="Arnoldi 基的正交性")
    ev_ref = np.asarray(ref.eigh(A)[0])
    ev_uniq = np.unique(np.round(ev_ref, 10))
    lz = lanczos(A, v0, n)
    ritz_uniq = np.unique(np.round(lz.ritz, 9))
    assert ritz_uniq.size == ev_uniq.size, (
        f"Lanczos 只量到 {ritz_uniq.size} 个相异 Ritz 值，而 A 有 {ev_uniq.size} 个相异特征值"
    )
    dev = float(np.max(np.abs(ritz_uniq - ev_uniq)))
    assert_leq(dev, 1e-8, label="Lanczos 的相异 Ritz 值与相异特征值的最大偏差")
    record(
        label="Arnoldi 正交性和 Lanczos 相异谱（2D 6×6 Laplacian，n=36）",
        theory="VᵀV=I；相异 Ritz 值 = 相异特征值",
        measured=(
            f"Arnoldi 正交偏差 {ar.orthogonality:.2e}；相异 Ritz {ritz_uniq.size} 个 vs 相异特征值 "
            f"{ev_uniq.size} 个，最大偏差 {dev:.2e}；Lanczos 跑了 {lz.n_steps} 步才停下（β≤1e-14）"
        ),
        tol="≤ 1e-13 / ≤ 1e-8",
        passed=dev <= 1e-8,
        detail=(
            "Krylov 维数（33）远大于相异特征值个数（19）：第 13 步之后 β 已掉到 1e-7 以下的舍入噪声，"
            "再往后是噪声被重新放大生成的伪方向，Ritz 值仍然正确——只看步数会把噪声当成新方向"
        ),
    )


@assertion("A", theory="CG: r_iᵀr_j=0 (i≠j) 且 p_iᵀAp_j=0；GMRES: r_iᵀ(Av_j)=0 (j<i)", chapter=CHAPTER)
def test_orthogonality_conditions() -> None:
    n = 24
    A = laplacian_1d(n)
    b = np.linspace(1.0, 2.0, n)
    cg = conjugate_gradient(A, b, tol=0.0, maxiter=n, track_basis=True, track_orthogonality=True)
    R = cg.residual_orthogonality
    floor = np.sqrt(U) * float(cg.abs_residuals[0])
    keep = [i for i in range(R.shape[0]) if cg.abs_residuals[i] > floor]
    worst_r = max(abs(float(R[i, j])) for i in keep for j in range(i))
    P = cg.p_orthogonality
    worst_p = max(abs(float(P[i, j])) for i in keep for j in range(min(i, P.shape[1])))
    gm = gmres(A, b, m=n, tol=0.0, maxiter=n, track_orthogonality=True)
    G = gm.residual_vs_basis
    floor_g = np.sqrt(U) * float(gm.abs_residuals[0])
    keep_g = [i for i in range(G.shape[0]) if gm.abs_residuals[i] > floor_g]
    worst_v = max(abs(float(G[i, j])) for i in keep_g for j in range(min(i, G.shape[1])))
    assert_leq(worst_r, 1e-13, label="CG 的 |r_iᵀr_j| 最大值（i≠j）")
    assert_leq(worst_p, 1e-13, label="CG 的 |p_iᵀAp_j| 最大值（i≠j）")
    # GMRES 的 r_i 是用 x_i 重新算的（真实残差），舍入底 ~u·κ(A)，比 CG 的递推残差高两个量级
    assert_leq(worst_v, 1e-11, label="GMRES 的 |r_iᵀ(Av_j)| 最大值（j<i）")
    record(
        label="CG/GMRES 的正交性条件（n=24 一维 Laplacian）",
        theory="CG: r_iᵀr_j=0、p_iᵀAp_j=0；GMRES: r_iᵀ(Av_j)=0",
        measured=f"|r_iᵀr_j|≤{worst_r:.1e}、|p_iᵀAp_j|≤{worst_p:.1e}、|r_iᵀAv_j|≤{worst_v:.1e}",
        tol="CG ≤ 1e-13；GMRES ≤ 1e-11（真实残差重算的舍入底更高）",
        passed=max(worst_r, worst_p) <= 1e-13 and worst_v <= 1e-11,
        detail=f"未屏蔽时最后一行会给出 O(1) 的假值：残差 {cg.abs_residuals[-1]:.2e} 已低于 {floor:.2e}",
    )


@assertion("A", theory="预条件不改变解：PCG/预条件 GMRES 的解 = 直接 LU 解", chapter=CHAPTER)
def test_preconditioners_do_not_change_the_solution() -> None:
    worst = 0.0
    report = []
    for name, A in (("1d_n64", laplacian_1d(64)), ("2d_k8", laplacian_2d(8))):
        n = A.shape[0]
        b = np.linspace(1.0, 2.0, n)
        x_star = _direct(A, b)
        for pname in ("none", "jacobi", "ssor", "ilu0"):
            M = build_precond(pname, A, omega=1.2)
            run = pcg(A, b, precond=M, tol=1e-11, maxiter=4 * n, x_exact=x_star, track_orthogonality=False)
            err = float(np.max(np.abs(run.x - x_star))) / float(np.max(np.abs(x_star)))
            worst = max(worst, err)
            report.append(f"{name}/{pname}:{run.n_iter}步({err:.1e})")
    assert_leq(worst, 1e-8, label="四种预条件下解的最大相对偏差")
    record(
        label="PCG(none/jacobi/ssor/ilu0) 解精度与迭代数",
        theory="预条件只改收敛速度，不改解",
        measured="；".join(report),
        tol="≤ 1e-8 相对",
        passed=worst <= 1e-8,
    )


@assertion("A", theory="预条件 Lanczos 的 Ritz 值 = M⁻¹A 的谱（在 M-内积下自伴）", chapter=CHAPTER)
def test_preconditioned_spectrum_matches_exact() -> None:
    """κ(M⁻¹A) 的估计必须落在精确谱上：内积选错（用 M⁻¹-内积）会偏 2%~5%，这条断言就是护栏。"""
    worst = 0.0
    rows = []
    for name, A in (("1d_n32", laplacian_1d(32)), ("2d_k8", laplacian_2d(8))):
        n = A.shape[0]
        for pname in ("none", "jacobi", "ssor", "ilu0"):
            P = build_precond(pname, A, omega=1.2)
            if P is None:
                ev = np.sort(np.asarray(ref.eigh(A)[0]))
            else:
                # M⁻¹A 与对称矩阵 M^{-1/2}AM^{-1/2} 相似 ⇒ 谱相同，可以用 eigh 精确算
                M = P.to_dense()
                w, V = ref.eigh(M)
                # 必须两边都用 M^{-1/2}：M^{-1/2}AM^{+1/2} 虽然与 M⁻¹A 相似，但它**不对称**，
                # 交给 eigh（只读一个三角）会静默给出垃圾（实测偏到 437.98 vs 38.01）
                M_minus_half = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
                ev = np.sort(np.asarray(ref.eigh(M_minus_half @ A @ M_minus_half)[0]))
            exact = float(ev[-1] / ev[0])
            sp = precond_spectrum(A, precond=P, m=min(n, 80))
            rel = abs(sp.kappa_estimate - exact) / exact
            worst = max(worst, rel)
            rows.append(f"{name}/{pname}: {sp.kappa_estimate:.4f} vs {exact:.4f}")
    assert_leq(worst, 1e-8, label="预条件 κ 估计与精确 κ 的最大相对偏差")
    record(
        label="预条件谱估计 vs 精确谱（1d n=32 / 2d 8×8，四种预条件）",
        theory="M⁻¹A 在 ⟨x,y⟩_M=xᵀMy 下自伴 ⇒ Lanczos 的 Ritz 值全实且收敛到它的谱",
        measured="；".join(rows) + f"；最大相对偏差 {worst:.2e}",
        tol="≤ 1e-8 相对",
        passed=worst <= 1e-8,
        detail=(
            "这条断言是给内积选的护栏：改用 M⁻¹-内积时 M⁻¹A 一般不正自伴，"
            "SSOR 一维 n=32 会给出 38.82（真值 38.01）、二维 8×8 给 3.75（真值 3.59），偏 2%~5%"
        ),
    )


# --------------------------------------------------------------------------- #
# B 类（不变量，放在这里是因为它们只对 Krylov 路径成立）
# --------------------------------------------------------------------------- #
@assertion("B", theory="CSR 与稠密给出同一串迭代（稀疏只是省内存，不改数学）", chapter=CHAPTER)
def test_csr_and_dense_give_same_iterates() -> None:
    A = laplacian_2d(7)
    csr = CSR.from_dense(A)
    n = A.shape[0]
    b = np.linspace(1.0, 2.0, n)
    r1 = conjugate_gradient(A, b, tol=0.0, maxiter=n, track_orthogonality=False)
    r2 = conjugate_gradient(csr, b, tol=0.0, maxiter=n, track_orthogonality=False)
    dev = float(np.max(np.abs(r1.iterates - r2.iterates)))
    g1 = gmres(A, b, m=n, tol=0.0, maxiter=n, track_orthogonality=False)
    g2 = gmres(csr, b, m=n, tol=0.0, maxiter=n, track_orthogonality=False)
    dev_g = float(np.max(np.abs(g1.iterates - g2.iterates)))
    assert_leq(max(dev, dev_g), 1e-13, label="CSR 与稠密迭代序列的最大偏差")
    record(
        label="CSR vs 稠密：CG/GMRES 的整条迭代序列（2D 7×7，nnz=%d / %d）" % (csr.nnz, n * n),
        theory="同一套数学",
        measured=f"CG 偏差 {dev:.2e}，GMRES 偏差 {dev_g:.2e}",
        tol="≤ 1e-13",
        passed=max(dev, dev_g) <= 1e-13,
    )


@assertion("B", theory="预条件子.apply 必须真的是 M⁻¹：M·apply(r)=r", chapter=CHAPTER)
def test_preconditioner_apply_is_the_inverse() -> None:
    worst = 0.0
    for name, A in (("1d32", laplacian_1d(32)), ("2d8", laplacian_2d(8))):
        n = A.shape[0]
        r = np.linspace(-1.0, 1.0, n)
        for pname in ("jacobi", "ssor", "ilu0"):
            M = build_precond(pname, A, omega=1.2)
            z = M.apply(r)
            back = M.to_dense() @ z
            scale = float(np.max(np.abs(r))) or 1.0
            dev = float(np.max(np.abs(back - r))) / scale
            worst = max(worst, dev)
    assert_leq(worst, 1e-12, label="M·apply(r)−r 的最大相对偏差")
    record(
        label="预条件子的 apply 与 to_dense 互为逆（1d n=32 / 2d 8×8）",
        theory="z = M⁻¹r ⟺ Mz = r",
        measured=f"最大相对偏差 {worst:.3e}",
        tol="≤ 1e-12",
        passed=worst <= 1e-12,
    )


# --------------------------------------------------------------------------- #
# C 类
# --------------------------------------------------------------------------- #
@assertion("C", theory="实测收敛因子 = ρ(T)（定常迭代）；CG 的渐近因子也由谱决定", chapter=CHAPTER)
def test_measured_factor_equals_spectral_radius() -> None:
    worst = 0.0
    rows = []
    for method, omega in (("jacobi", 1.0), ("gs", 1.0), ("sor", 1.4)):
        A = laplacian_1d(31)
        b = np.linspace(1.0, 2.0, 31)
        res = stationary_solve(A, b, method=method, omega=omega, maxiter=80, tol=0.0)  # type: ignore[arg-type]
        fac = estimate_linear_factor(res.residuals, skip=10, tail=25)
        rho = power_method(iteration_matrix(A, method=method, omega=omega), maxiter=4000).value  # type: ignore[arg-type]
        rel = abs(fac - rho) / rho
        worst = max(worst, rel)
        rows.append(f"{method}: 实测 {fac:.6f} vs ρ {rho:.6f}（{rel:.1e}）")
    assert_leq(worst, 5e-3, label="实测收敛因子与 ρ(T) 的最大相对偏差")
    record(
        label="残差尾部因子 vs 幂法谱半径（n=31，Jacobi/GS/SOR ω=1.4）",
        theory="渐近收敛因子 = ρ(T)",
        measured="；".join(rows),
        tol="≤ 5e-3 相对",
        passed=worst <= 5e-3,
    )


@assertion("C", theory="CG 迭代数 ~ √κ（斜率 1）：一维 Laplacian 与二维五点格式都要实测", chapter=CHAPTER)
def test_cg_iterations_scale_like_sqrt_kappa() -> None:
    ns = [32, 64, 128, 256]
    iters = []
    kappas = []
    for n in ns:
        A = laplacian_1d(n)
        b = np.linspace(1.0, 2.0, n)
        run = conjugate_gradient(A, b, tol=1e-10, maxiter=4 * n, track_orthogonality=False)
        iters.append(float(run.n_iter))
        kappas.append(float(ref.cond(A)))
    fit = assert_order(
        [float(np.sqrt(k)) for k in kappas],
        iters,
        theory=1.0,
        tol=0.1,
        label="CG 迭代数对 √κ 的斜率（一维 Laplacian）",
    )
    # 二维：κ 只能由预条件谱估计器（Lanczos+Sturm）给，不能调 numpy.linalg
    grid_iters = []
    grid_sqrt = []
    for k in (16, 24, 32, 48):
        A = CSR.from_dense(laplacian_2d(k))
        b = np.linspace(1.0, 2.0, A.shape[0])
        run = conjugate_gradient(A, b, tol=1e-10, maxiter=4 * A.shape[0], track_orthogonality=False)
        sp = precond_spectrum(A, m=min(A.shape[0], 80))
        grid_iters.append(float(run.n_iter))
        grid_sqrt.append(float(np.sqrt(sp.kappa_estimate)))
    fit2 = assert_order(
        grid_sqrt,
        grid_iters,
        theory=1.0,
        tol=0.15,
        label="CG 迭代数对 √κ 的斜率（二维五点格式，CSR）",
    )
    record(
        label="CG 迭代数 ~ √κ 的双对数斜率",
        theory="斜率 1（最坏界 2((√κ−1)/(√κ+1))^k 给的是同阶）",
        measured=(
            f"一维 n={ns}: iters={[int(i) for i in iters]}, κ={[f'{k:.1f}' for k in kappas]}, 斜率 {fit.slope:.4f}；"
            f"二维 k=16..48: iters={[int(i) for i in grid_iters]}, 斜率 {fit2.slope:.4f}"
        ),
        tol="斜率 1 ± 0.1（一维）/ ± 0.15（二维）",
        passed=abs(fit.slope - 1.0) <= 0.1 and abs(fit2.slope - 1.0) <= 0.15,
        detail="二维在小 k 区间有对数修正（有效 κ 偏小），所以窗口取 k=16…48 而不是 k=8…24",
    )


@assertion("C", theory="预条件把 κ 降 8.6 倍，迭代数只降 1.6 倍 ⇒ 迭代数不按 √κ 走", chapter=CHAPTER)
def test_preconditioner_iteration_count_is_not_sqrt_kappa() -> None:
    k = 8
    A = laplacian_2d(k)
    n = A.shape[0]
    b = np.linspace(1.0, 2.0, n)
    x_star = _direct(A, b)
    stats = {}
    for pname in ("none", "ssor"):
        M = build_precond(pname, A, omega=1.2)
        run = pcg(A, b, precond=M, tol=1e-10, maxiter=4 * n, x_exact=x_star, track_orthogonality=False)
        sp = precond_spectrum(A, precond=M, m=min(n, 60))
        fac = estimate_linear_factor(run.residuals, skip=max(run.n_iter // 3, 1), tail=4)
        stats[pname] = {
            "iter": float(run.n_iter),
            "kappa": float(sp.kappa_estimate),
            "fac": fac,
            "kappa_eff": kappa_from_factor(fac) if 0.0 <= fac < 1.0 else float("inf"),
        }
    kappa_ratio = stats["none"]["kappa"] / stats["ssor"]["kappa"]
    iter_ratio = stats["none"]["iter"] / stats["ssor"]["iter"]
    sqrt_ratio = float(np.sqrt(kappa_ratio))
    assert kappa_ratio > 5.0, f"SSOR 没把 κ 压下来：{kappa_ratio:.2f}"
    assert iter_ratio < 0.9 * sqrt_ratio, (
        f"迭代数比 {iter_ratio:.2f} 竟然接近 √κ 比 {sqrt_ratio:.2f}；本算例的实测结论是远小于它"
    )
    record(
        label="预条件：κ 降幅 vs 迭代数降幅（2D 8×8，SSOR ω=1.2）",
        theory="迭代数由谱**分布**决定，√κ 只是最坏界的启发",
        measured=(
            f"κ {stats['none']['kappa']:.2f} → {stats['ssor']['kappa']:.2f}（×{kappa_ratio:.2f}），"
            f"迭代 {stats['none']['iter']:.0f} → {stats['ssor']['iter']:.0f}（×{iter_ratio:.2f}），"
            f"√κ 比 = {sqrt_ratio:.2f}；实测因子 {stats['ssor']['fac']:.4f} ⇒ 等效 κ "
            f"{stats['ssor']['kappa_eff']:.2f}"
        ),
        tol="迭代数降幅 < 0.9·√κ 降幅",
        passed=iter_ratio < 0.9 * sqrt_ratio,
        detail="等效 κ（1.6）比真实 κ（3.75）还小一半：谱聚簇时 CG 比最坏界快得多",
    )


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _direct(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    from numcore.linalg.lu import lu_decompose, lu_solve

    res = lu_decompose(np.asarray(A, dtype=float), pivot="partial")
    assert not res.singular
    return lu_solve(res, np.asarray(b, dtype=float))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
