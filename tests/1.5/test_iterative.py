"""1.5 章 A/B（+ 一条 C）类断言：定常迭代、迭代矩阵与谱半径。

A 类（解析比对）
    * Jacobi/GS/SOR 的谱半径 = 一维 Laplacian 上的解析余弦值（含 SOR 的 Young 关系）；
    * 2×2 椭圆模型上 :math:`\\rho(T_J)=\\sqrt{|a_{12}a_{21}|/(a_{11}a_{22})}` 的闭式；
    * Jacobi 的**逐模态**误差衰减 :math:`c_j^{(k)}=\\mu_j^k c_j^{(0)}`（模态分解的精确性）。

B 类（不变量）
    * 迭代矩阵与真实扫描的一致性（``x_{k+1}=Tx_k+c``）；
    * ``(D-\\omega L)T_\\omega = (1-\\omega)D+\\omega U`` 的代数恒等式；
    * ``\\omega=1`` 时 SOR 与 GS 逐位相同；
    * CSR 行访问与稠密行访问给出同一条 GS 迭代；
    * 帧契约（数组名与 requiredAux 对得上、无非有限值、meta 只放标量）。

C 类（实测阈值）
    * 用**中心差分单调性**二分实测 SOR 的最优 :math:`\\omega`，与
      :math:`2/(1+\\sqrt{1-\\mu^2})` 比对。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from numcore.linalg.iterative import (
    convergence_factor,
    gauss_seidel,
    iteration_matrix,
    jacobi,
    jacobi_eigenvalues_1d,
    jacobi_iteration_matrix,
    laplacian_1d,
    mode_components,
    mode_decay_factors,
    power_method,
    rotated_spd_2x2,
    sine_eigenvectors_1d,
    sor,
    sor_spectral_radius,
    stationary_solve,
    young_optimal_omega,
    gs_iteration_matrix,
    sor_iteration_matrix,
)
from numcore.linalg.lu import lu_decompose, lu_solve
from numcore.linalg.sparse import CSR
from numcore.reference import dense as ref
from numcore.verify import (
    assert_leq,
    assert_monotone,
    assert_threshold,
    assert_within,
    assertion,
    estimate_linear_factor,
    find_threshold,
    record,
)

U = float(np.finfo(float).eps) / 2.0
CHAPTER = "1.5"


# --------------------------------------------------------------------------- #
# A 类
# --------------------------------------------------------------------------- #
@assertion("A", theory="一维 Laplacian 上 ρ(T_J)=cos(π/(n+1))，ρ(T_GS)=cos²(π/(n+1))", chapter=CHAPTER)
def test_spectral_radius_matches_analytic_cosine() -> None:
    """手写幂法量出的谱半径应当逐位等于解析余弦值（并用参照特征值独立复核）。"""
    worst_j = 0.0
    worst_g = 0.0
    for n in (7, 15, 31, 63):
        A = laplacian_1d(n)
        mu = float(np.cos(np.pi / (n + 1)))
        rho_j = power_method(jacobi_iteration_matrix(A), maxiter=4000).value
        rho_g = power_method(gs_iteration_matrix(A), maxiter=4000).value
        worst_j = max(worst_j, abs(rho_j - mu) / mu)
        worst_g = max(worst_g, abs(rho_g - mu * mu) / (mu * mu))
        # 独立复核：直接对迭代矩阵做参照特征值分解（只在 tests 里允许 numpy.linalg）
        tj = np.max(np.abs(np.asarray(ref.eigvals(jacobi_iteration_matrix(A)))))
        assert abs(tj - mu) <= 1e-12 * mu, f"n={n}: 参照特征值 {tj} 与解析 {mu} 不符"
        ev = np.asarray(ref.eigvals(A))
        assert np.allclose(np.sort(ev.real), 4.0 * np.sin(np.arange(1, n + 1) * np.pi / (2 * (n + 1))) ** 2, atol=1e-10)
    assert_leq(worst_j, 1e-8, label="ρ(T_J) 与 cos(π/(n+1)) 的最大相对偏差")
    assert_leq(worst_g, 1e-8, label="ρ(T_GS) 与 cos²(π/(n+1)) 的最大相对偏差")
    record(
        label="ρ 的解析比对（n=7…63）",
        theory="cos(π/(n+1)) 与 cos²(π/(n+1))",
        measured=f"ρ_J 偏差 ≤ {worst_j:.3e}，ρ_GS 偏差 ≤ {worst_g:.3e}",
        tol="≤ 1e-8 相对（幂法 tol=1e-13）",
        passed=worst_j <= 1e-8 and worst_g <= 1e-8,
    )


@assertion("A", theory="2×2 椭圆模型上 ρ(T_J)=√(|a12·a21|/(a11·a22))，且 κ(A)=κ 精确", chapter=CHAPTER)
def test_two_by_two_spectral_radius_formula() -> None:
    """旋转椭圆模型给了谱半径的闭式，幂法必须落在闭式上（含极扁的 κ=100）。"""
    worst_mu = 0.0
    worst_kappa = 0.0
    cases = [(2.0, 20.0), (5.0, 35.0), (20.0, 35.0), (100.0, 20.0), (100.0, 60.0)]
    for kappa, ang in cases:
        A = rotated_spd_2x2(kappa, ang)
        mu_theory = float(np.sqrt(abs(A[0, 1] * A[1, 0]) / (A[0, 0] * A[1, 1])))
        mu_meas = power_method(jacobi_iteration_matrix(A), maxiter=4000).value
        ev = np.sort(np.asarray(ref.eigvals(A)).real)
        kappa_meas = float(ev[-1] / ev[0])
        worst_mu = max(worst_mu, abs(mu_meas - mu_theory) / max(mu_theory, 1e-12))
        worst_kappa = max(worst_kappa, abs(kappa_meas - kappa) / kappa)
    assert_leq(worst_mu, 1e-7, label="ρ(T_J) 与闭式 √|a12a21/(a11a22)| 的最大相对偏差")
    assert_leq(worst_kappa, 1e-12, label="κ(A) 与设定值 κ 的最大相对偏差")
    record(
        label="2×2 椭圆模型（κ ≤ 100，5 组角度）",
        theory="ρ(T_J)=√|a12a21/(a11a22)|，λ(A)={1,κ}",
        measured=f"ρ 偏差 ≤ {worst_mu:.3e}，κ 偏差 ≤ {worst_kappa:.3e}",
        tol="≤ 1e-7 / ≤ 1e-12 相对",
        passed=worst_mu <= 1e-7 and worst_kappa <= 1e-12,
    )


@assertion("A", theory="SOR 的 Young 关系：ω≤ω_opt 时 ρ=((ωμ+√(ω²μ²−4(ω−1)))/2)²，ω_opt 处 ρ=ω_opt−1", chapter=CHAPTER)
def test_sor_young_relation_and_optimal_omega() -> None:
    """SOR 谱半径的解析式必须能被幂法复现：欠松弛段精确，最优 ω 处有缺陷、只能到 1e-3。"""
    worst_sub = 0.0
    worst_opt = 0.0
    for n in (15, 31):
        A = laplacian_1d(n)
        mu = float(np.cos(np.pi / (n + 1)))
        w_opt = young_optimal_omega(mu)
        rho_opt_theory = w_opt - 1.0
        rho_opt_meas = power_method(sor_iteration_matrix(A, w_opt), maxiter=40000).value
        worst_opt = max(worst_opt, abs(rho_opt_meas - rho_opt_theory) / rho_opt_theory)
        for w in (1.2, 0.8, 1.5):
            if w >= w_opt:
                continue
            th = sor_spectral_radius(mu, w)
            me = power_method(sor_iteration_matrix(A, w), maxiter=40000).value
            worst_sub = max(worst_sub, abs(me - th) / th)
    # 2×2 上也验一遍 Young 关系（一致次序矩阵）
    for kappa, ang in ((5.0, 35.0), (20.0, 20.0)):
        A = rotated_spd_2x2(kappa, ang)
        mu = float(np.sqrt(abs(A[0, 1] * A[1, 0]) / (A[0, 0] * A[1, 1])))
        w_opt = young_optimal_omega(mu)
        me = power_method(sor_iteration_matrix(A, w_opt), maxiter=40000).value
        worst_opt = max(worst_opt, abs(me - (w_opt - 1.0)) / (w_opt - 1.0))
    assert_leq(worst_sub, 1e-7, label="欠松弛段 ρ(SOR) 与 Young 解析式的最大相对偏差")
    assert_leq(worst_opt, 5e-3, label="ρ(T_ωopt) 与 ω_opt−1 的最大相对偏差")
    record(
        label="SOR 的 Young 关系（一维 n=15/31 + 2×2 两组）",
        theory="ρ(T_ω)=((ωμ+√(ω²μ²−4(ω−1)))/2)²（ω≤ω_opt），ρ(T_ωopt)=ω_opt−1",
        measured=f"欠松弛段偏差 ≤ {worst_sub:.3e}；ω_opt 处偏差 ≤ {worst_opt:.3e}",
        tol="≤ 1e-7（欠松弛）/ ≤ 5e-3（最优 ω，主特征值有缺陷）",
        passed=worst_sub <= 1e-7 and worst_opt <= 5e-3,
        detail="ω_opt 处 T_ω 的特征值是重根（缺陷），幂法的范数增长比收敛极慢，这正是课本不会提醒的实测代价",
    )


@assertion("A", theory="Jacobi 的模态分量精确满足 c_j^(k)=μ_j^k c_j^(0)（T_J 与 A 同特征向量）", chapter=CHAPTER)
def test_jacobi_mode_decay_is_exact() -> None:
    """模态分解对 Jacobi 是**恒等式**而不是近似：比值应当逐位等于 μ_j。"""
    n = 31
    A = laplacian_1d(n)
    V = sine_eigenvectors_1d(n)
    coef = np.linspace(1.0, 0.2, n)
    e0 = V @ coef
    x_star = np.zeros(n)
    res = jacobi(A, np.zeros(n), x0=e0, maxiter=6, tol=0.0, x_exact=x_star)
    C = mode_components(res.iterates - x_star[None, :])
    mus = jacobi_eigenvalues_1d(n)
    scale = float(np.max(np.abs(C[0])))
    floor = 1e-12 * scale  # 低于它就不再是"衰减"，而是舍入噪声
    worst = 0.0
    dropped = 0
    for j in range(n):
        if abs(C[0, j]) < 1e-8:
            continue
        for k in range(1, 6):
            pred = mus[j] ** k * C[0, j]
            if abs(pred) < floor:  # μ_j ≈ 0（j=(n+1)/2）的模态一步就消失，之后只剩噪声
                dropped += 1
                continue
            worst = max(worst, abs(C[k, j] - pred) / abs(pred))
    assert_leq(worst, 1e-10, label="Jacobi 模态衰减与 μ_j^k 的最大相对偏差")
    meas = np.array([C[1, j] / C[0, j] for j in range(n)])
    dev = float(np.max(np.abs(meas - mus)))
    record(
        label="n=31 的 31 个模态逐一比对（k=1…5）",
        theory="c_j^(k)/c_j^(0) = cos(jπ/32)^k",
        measured=f"最大相对偏差 {worst:.3e}；单步比值与 μ_j 的最大绝对偏差 {dev:.3e}",
        tol="≤ 1e-10 相对",
        passed=worst <= 1e-10,
        detail=(
            f"μ_1 = {mus[0]:.8f}（最低模态几乎不衰减），μ_31 = {mus[-1]:.8f}（最高模态交替衰减）；"
            f"另有 {dropped} 个 (j,k) 组合因为 μ_j^k·c_j 落到 1e-12·max|c_0| 以下被跳过——"
            "那些位置上的分量已经是舍入噪声，不是衰减"
        ),
    )


# --------------------------------------------------------------------------- #
# B 类
# --------------------------------------------------------------------------- #
@assertion("B", theory="x_{k+1} = T x_k + c：迭代矩阵必须复现真实扫描的一步", chapter=CHAPTER)
def test_iteration_matrix_reproduces_sweep() -> None:
    """把 T 和 c 拼回一步，必须与 stationary_solve 的第二步逐位一致。"""
    n = 17
    A = laplacian_1d(n)
    b = np.linspace(1.0, 2.0, n)
    x0 = np.linspace(-0.5, 0.5, n)
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    worst = 0.0
    for method, omega, c in (
        ("jacobi", 1.0, _lu_vec(D, b)),
        ("gs", 1.0, _lu_vec(D - L, b)),
        ("sor", 1.3, 1.3 * _lu_vec(D - 1.3 * L, b)),
    ):
        T = iteration_matrix(A, method=method, omega=omega)  # type: ignore[arg-type]
        res = stationary_solve(A, b, method=method, omega=omega, x0=x0, maxiter=2, tol=0.0)  # type: ignore[arg-type]
        pred = T @ x0 + c
        worst = max(worst, float(np.max(np.abs(pred - res.iterates[1]))))
    assert_leq(worst, 1e-12, label="T·x0+c 与真实扫描一步的最大偏差")
    record(
        label="Jacobi/GS/SOR 的迭代矩阵一致性（n=17）",
        theory="x_{k+1}=Tx_k+c，c=(D−ωL)⁻¹ωb",
        measured=f"最大偏差 {worst:.3e}",
        tol="≤ 1e-12（绝对）",
        passed=worst <= 1e-12,
    )


@assertion("B", theory="(D−ωL)T_ω = (1−ω)D+ωU 的代数恒等式", chapter=CHAPTER)
def test_sor_iteration_matrix_algebra() -> None:
    """SOR 迭代矩阵由 LU 逐列求逆得到，必须满足定义它的矩阵恒等式。"""
    n = 23
    A = laplacian_1d(n)
    worst = 0.0
    for omega in (0.7, 1.0, 1.4, 1.8):
        D = np.diag(np.diag(A))
        L = -np.tril(A, -1)
        U = -np.triu(A, 1)
        T = sor_iteration_matrix(A, omega)
        lhs = (D - omega * L) @ T
        rhs = (1.0 - omega) * D + omega * U
        worst = max(worst, float(np.max(np.abs(lhs - rhs))) / float(np.max(np.abs(rhs))))
    assert_leq(worst, 1e-13, label="(D−ωL)T_ω − ((1−ω)D+ωU) 的最大相对残差")
    record(
        label="SOR 迭代矩阵恒等式（n=23，ω=0.7/1.0/1.4/1.8）",
        theory="(D−ωL)T_ω = (1−ω)D+ωU",
        measured=f"最大相对残差 {worst:.3e}",
        tol="≤ 1e-13 相对",
        passed=worst <= 1e-13,
    )


@assertion("B", theory="ω=1 时 SOR ⇔ Gauss–Seidel（迭代矩阵与迭代序列都逐位相同）", chapter=CHAPTER)
def test_sor_at_one_equals_gauss_seidel() -> None:
    A = laplacian_1d(19)
    b = np.linspace(1.0, 2.0, 19)
    x0 = np.linspace(0.5, -0.5, 19)
    T_s = sor_iteration_matrix(A, 1.0)
    T_g = gs_iteration_matrix(A)
    dev_t = float(np.max(np.abs(T_s - T_g)))
    r_s = sor(A, b, x0=x0, omega=1.0, maxiter=5, tol=0.0)
    r_g = gauss_seidel(A, b, x0=x0, maxiter=5, tol=0.0)
    dev_x = float(np.max(np.abs(r_s.iterates - r_g.iterates)))
    assert_leq(max(dev_t, dev_x), 1e-14, label="ω=1 的 SOR 与 GS 的最大偏差")
    record(
        label="SOR(ω=1) 与 GS 的等价性",
        theory="T_1 = T_GS，(D−L)⁻¹U 完全一致",
        measured=f"迭代矩阵偏差 {dev_t:.3e}，5 步迭代偏差 {dev_x:.3e}",
        tol="≤ 1e-14",
        passed=max(dev_t, dev_x) <= 1e-14,
    )


@assertion("B", theory="CSR 行访问与稠密行访问给出同一条迭代（稀疏容器不改变数学）", chapter=CHAPTER)
def test_csr_rows_reproduce_dense_gauss_seidel() -> None:
    """用 CSR 的行手写一遍 GS 前向扫描，必须与 gauss_seidel 的稠密实现逐位一致。"""
    n = 18
    A = laplacian_1d(n)
    csr = CSR.from_dense(A)
    b = np.linspace(1.0, 2.0, n)
    x0 = np.linspace(0.3, -0.7, n)
    res = gauss_seidel(A, b, x0=x0, maxiter=4, tol=0.0)
    x = np.array(x0, dtype=float)
    for _ in range(4):
        for i in range(n):
            cols, vals = csr.row(i)
            s = float(np.dot(vals, x[cols])) - float(vals[cols == i][0]) * x[i] if np.any(cols == i) else float(np.dot(vals, x[cols]))
            d = float(vals[cols == i][0]) if np.any(cols == i) else 1.0
            x[i] = (b[i] - s) / d
    dev = float(np.max(np.abs(x - res.iterates[-1])))
    assert_leq(dev, 1e-13, label="CSR 行扫描与稠密 GS 的最大偏差")
    record(
        label="CSR 行访问 vs 稠密 GS（n=18，4 步）",
        theory="稀疏与稠密是同一套数学",
        measured=f"最大偏差 {dev:.3e}，CSR nnz={csr.nnz}（稠密 nnz={int(np.count_nonzero(A))}）",
        tol="≤ 1e-13",
        passed=dev <= 1e-13,
    )


@assertion("B", theory="ρ(T) < 1 ⟺ 任意初值收敛；ρ(T) > 1 时残差必须发散", chapter=CHAPTER)
def test_convergence_iff_spectral_radius_below_one() -> None:
    """构造一对矩阵：弱对角占优（ρ(T_J)>1）与强对角占优（ρ(T_J)<1），看残差走向。"""
    n = 12
    A_bad = laplacian_1d(n)
    np.fill_diagonal(A_bad, 1.2)  # 对角被削弱 → Jacobi 谱半径 > 1
    # 注意：给 A 乘常数**不改变** T_J = I − D⁻¹A（D 同步缩放），所以要真加对角
    A_ok = laplacian_1d(n) + 10.0 * np.eye(n)
    rho_bad = power_method(jacobi_iteration_matrix(A_bad), maxiter=4000).value
    rho_ok = power_method(jacobi_iteration_matrix(A_ok), maxiter=4000).value
    b = np.ones(n)
    r_bad = jacobi(A_bad, b, maxiter=25, tol=0.0)
    r_ok = jacobi(A_ok, b, maxiter=25, tol=0.0)
    grow = float(r_bad.residuals[-1] / r_bad.residuals[0])
    shrink = float(r_ok.residuals[-1] / r_ok.residuals[0])
    assert rho_bad > 1.0 and rho_ok < 1.0, f"构型失败：ρ={rho_bad}, {rho_ok}"
    assert grow > 1.0, f"ρ(T)>1 却没有发散：{grow}"
    assert shrink < 1e-6, f"ρ(T)<1 却几乎没下降：{shrink}"
    record(
        label="ρ(T)>1 与 ρ(T)<1 的残差走向（n=12，25 步）",
        theory="ρ(T)<1 收敛，ρ(T)≥1 不收敛",
        measured=f"ρ={rho_bad:.4f} 时残差 ×{grow:.3e}；ρ={rho_ok:.4f} 时残差 ×{shrink:.3e}",
        tol="25 步后：发散 >1、收敛 <1e-6",
        passed=grow > 1.0 and shrink < 1e-6,
    )


# --------------------------------------------------------------------------- #
# C 类：实测 SOR 的最优 ω
# --------------------------------------------------------------------------- #
@assertion("C", theory="ρ_meas(ω) 在 ω_opt 处取极小：中心差分符号在 ω_opt 处翻转", chapter=CHAPTER)
def test_optimal_omega_measured_by_bisection() -> None:
    """不查公式，只问"往右走 ρ 变大还是变小"，二分出 ω_opt 再与解析值比对。"""
    n = 15
    A = laplacian_1d(n)
    mu = float(np.cos(np.pi / (n + 1)))
    theory = young_optimal_omega(mu)

    def threshold_at(h: float):
        def decreasing(omega: float) -> bool:
            lo = power_method(sor_iteration_matrix(A, omega - h), maxiter=6000).value
            hi = power_method(sor_iteration_matrix(A, omega + h), maxiter=6000).value
            return hi < lo

        return find_threshold(decreasing, 1.0, 1.95, tol=1e-4, note=f"中心差分 h={h}")

    h = 0.002
    found = threshold_at(h)
    assert_threshold(found, theory, rtol=1.5e-3, label="SOR 最优 ω 的二分实测值（h=0.002）")
    # 实测偏移随 h 的线性收缩：这是"用差分测极小点"的固有偏差，必须量出来而不是假装没有
    biases = []
    for hh in (0.01, 0.005, 0.002):
        f = threshold_at(hh)
        biases.append((hh, f.threshold, f.threshold - theory, f.calls))
    rel = [abs(b[2]) / theory for b in biases]
    assert rel[2] < rel[0], f"步长减小时偏差没有变小：{[f'{x:.2e}' for x in rel]}"
    record(
        label="ω_opt 的实测偏差 vs 差分步长 h",
        theory=f"ω_opt = 2/(1+√(1−μ²)) = {theory:.6f}（h→0 的极限）",
        measured="；".join(f"h={hh}: 实测 {th:.6f}（偏差 {b:+.2e}）" for hh, th, b, _ in biases),
        tol=f"h=0.002 时 ≤ 1.5e-3 相对（实测 {rel[2]:.2e}）",
        passed=rel[2] <= 1.5e-3,
        detail=(
            "偏差 ≈ 0.47h（h 减半偏差就减半），这是中心差分测极小点的固有系统偏移；"
            "再往下取 h=0.001 偏差反而跳回 5e-3——那时幂法自身的 1e-4 噪声已经盖过了 ρ 的真实差别"
        ),
    )
    rho_opt = power_method(sor_iteration_matrix(A, found.threshold), maxiter=40000).value
    record(
        label="实测最优点上的 ρ(T_ωopt) 与 ω_opt−1",
        theory=f"{found.threshold - 1.0:.6f}",
        measured=f"{rho_opt:.6f}",
        tol="≤ 5e-3 相对（重特征值处幂法慢）",
        passed=abs(rho_opt - (found.threshold - 1.0)) <= 5e-3 * found.threshold,
    )


@assertion("C", theory="残差在渐近段线性下降，实测因子 = 幂法量到的 ρ(T)", chapter=CHAPTER)
def test_measured_factor_matches_spectral_radius() -> None:
    """同一件事的两种量法：残差尾部斜率 vs 迭代矩阵的谱半径。"""
    n = 31
    A = laplacian_1d(n)
    b = np.linspace(1.0, 2.0, n)
    worst = 0.0
    for method, omega in (("jacobi", 1.0), ("gs", 1.0), ("sor", 1.5)):
        res = stationary_solve(A, b, method=method, omega=omega, maxiter=60, tol=0.0)  # type: ignore[arg-type]
        fac = estimate_linear_factor(res.residuals, skip=8, tail=20)
        rho = power_method(iteration_matrix(A, method=method, omega=omega), maxiter=4000).value  # type: ignore[arg-type]
        worst = max(worst, abs(fac - rho) / rho)
        assert_monotone(res.residuals[10:], direction="nonincreasing", label=f"{method} 残差渐近单调")
    assert_leq(worst, 5e-3, label="实测收敛因子与 ρ(T) 的最大相对偏差")
    record(
        label="残差尾部因子 vs ρ(T)（Jacobi/GS/SOR，n=31）",
        theory="渐近因子 = ρ(T)",
        measured=f"最大相对偏差 {worst:.3e}",
        tol="≤ 5e-3 相对",
        passed=worst <= 5e-3,
    )


# --------------------------------------------------------------------------- #
# 帧契约（B 类，顺带把 server 端拉进测试）
# --------------------------------------------------------------------------- #
@assertion("B", theory="帧契约：requiredAux 有对应数组、数组全有限、meta 只放标量", chapter=CHAPTER)
def test_chapter_frames_obey_contract() -> None:
    mod = _load_chapter()
    assert len(mod.MANIFEST["vizzes"]) == 4
    ids = {v["id"] for v in mod.MANIFEST["vizzes"]}
    assert ids == {"iteration_race", "error_spectrum", "krylov_growth", "precond_effect"}
    checked = 0
    bad: list[str] = []
    for viz in mod.MANIFEST["vizzes"]:
        base = {p["name"]: p.get("default") for p in viz["params"]}
        ts = [0.0, 7.0, float(viz["time"].get("tMax", 1.0))] if viz["time"]["mode"] != "none" else [0.0]
        for t in ts:
            fr = mod.compute(viz["id"], base, t)
            fr.validate()
            for aux in viz["requiredAux"]:
                if not any(nm.startswith(aux) for nm in fr.names):
                    bad.append(f"{viz['id']}: requiredAux {aux} 没有对应数组")
            for nm in fr.names:
                a = fr[nm]
                if not np.all(np.isfinite(a)):
                    bad.append(f"{viz['id']}.{nm} 含非有限值")
            for key, val in fr.meta.items():
                if isinstance(val, (str, list, tuple, dict, np.ndarray)):
                    bad.append(f"{viz['id']}.meta[{key}] 不是标量")
            checked += 1
    assert not bad, "；".join(bad)
    record(
        label="四个可视化的帧契约（含 t=0/中途/末端）",
        theory="requiredAux 命中、数组有限、meta 标量、可序列化",
        measured=f"{checked} 帧全部通过（{len(mod.MANIFEST['vizzes'])} 个 viz）",
        tol="0 处违约",
        passed=True,
    )


@assertion("B", theory="竞速帧：收敛后是重复点（race_len 才是真实步数）；残差整体下降但早期可以瞬态上升", chapter=CHAPTER)
def test_race_frame_trajectory_invariants() -> None:
    mod = _load_chapter()
    default = {p["name"]: p.get("default") for p in mod.MANIFEST["vizzes"][0]["params"]}
    fr = mod.compute("iteration_race", default, 60.0)
    res = fr["race_residual"]
    lens = fr["race_len"]
    assert res.shape == (5, 61) and lens.shape == (5,)
    names = ["Jacobi", "GS", "SOR", "CG", "GMRES"]
    # 整体必须下降；但早期可以上升——非正规迭代矩阵的瞬态增长是真实的（实测 Jacobi 第 0→1 步 ×1.14）
    assert np.all(res[:, -1] < res[:, 0]), "有轨迹跑完 60 步反而变大了"
    transients = [float(np.max(np.diff(res[i]))) for i in range(5)]
    # CG/GMRES 有极小化性质 ⇒ 残差单调；定常迭代在 2×2 上 T_J 形如 [[0,a],[b,0]]，
    # 特征值 ±μ 意味着每两步换一次相位，范数上下起伏 —— 只有**奇偶子列**各自单调。
    for i in (3, 4):
        assert_monotone(res[i, 5:], direction="nonincreasing", label=f"{names[i]} 残差单调")
    for i in (0, 1, 2):
        even = res[i, 10::2]
        odd = res[i, 11::2]
        assert_monotone(even, direction="nonincreasing", label=f"{names[i]} 残差偶子列单调")
        assert_monotone(odd, direction="nonincreasing", label=f"{names[i]} 残差奇子列单调")
        assert transients[i] > 0.0, f"{names[i]}：这个算例本应看到周期性起伏，实测却没有"
    cg_i = float(lens[3])
    assert cg_i <= 3.0, f"2×2 上 CG 用了 {cg_i} 步，超过 n+1=3"
    record(
        label="竞速帧：五条轨迹的步数、瞬态与整体下降（κ=20、θ=35°）",
        theory="收敛后为重复点；ρ<1 不排除早期瞬态上升",
        measured=(
            f"race_len = {lens.tolist()}（Jacobi/GS/SOR/CG/GMRES）；"
            f"首步最大增量 [{'/'.join(f'{t:.2e}' for t in transients)}]；"
            f"60 步后的相对残差 {'/'.join(f'{res[i, -1]:.1e}' for i in range(5))}"
        ),
        tol="CG、GMRES ≤ 3 步；尾段（k≥10）单调；末值 < 初值",
        passed=cg_i <= 3.0,
        detail="Jacobi 的第一步残差从 1 涨到 1.14：ρ(T)<1 只管渐近，不管前几步——这就是「非正规性」的可见后果",
    )


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _lu_vec(M: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """解 M z = rhs（走自研 LU），给"迭代矩阵 + 常数项"的一致性检查用。"""
    res = lu_decompose(M, pivot="partial")
    assert not res.singular
    return lu_solve(res, rhs)


_CHAPTER_CACHE: list = []


def _load_chapter():
    """按路径加载 ``server/chapters/1.5.py``（与 1.2 章的测试同样的做法）。"""
    if not _CHAPTER_CACHE:
        path = Path(__file__).resolve().parents[2] / "server" / "chapters" / "1.5.py"
        spec = importlib.util.spec_from_file_location("chapter_1_5_iterative", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CHAPTER_CACHE.append(mod)
    return _CHAPTER_CACHE[0]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
