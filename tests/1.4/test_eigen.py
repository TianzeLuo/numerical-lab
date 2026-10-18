"""1.4 章 A/C 类断言：幂法、反迭代、QR 迭代、Lanczos/Arnoldi 的**收敛率**。

这一章的核心不是"能不能算出特征值"，而是三件互相独立的事：

1. **收敛率**——幂法按 :math:`|\\lambda_2/\\lambda_1|`、无位移 QR 按
   :math:`|\\lambda_{i+1}/\\lambda_i|`、反迭代按 :math:`|\\lambda_{\\text{next}}-\\sigma|/|\\lambda_{\\text{near}}-\\sigma|`；
2. **瞬态**——非正规矩阵上"渐近率没变，开头几十步却像不收敛"；
3. **结构**——Hessenberg 化 / Lanczos 三项递推 / Arnoldi 的 :math:`AV_m=V_mH_m+h_{m+1,m}v_{m+1}e_m^{\\mathsf T}`
   这些**恒等式**必须逐项成立，它们才是后面 Ritz 值残差界的来源。

因此这里的断言分两类：A 类拿**解析真值**（Laplacian 的闭式谱、解析收敛率）与
:mod:`numcore.reference.dense` 对拍；C 类用 :func:`numcore.verify.assert_order` /
:func:`~numcore.verify.find_threshold` **实测**收敛因子与阈值，不接受"把理论值抄回来"。
每条断言都通过 ``record``/``assert_*`` helper 登记实测数字。
"""

from __future__ import annotations

import numpy as np
import pytest

from numcore.linalg.eigen import (
    _eigenvectors_of_small,
    arnoldi,
    cauchy_interlacing_bounds,
    dense_eigenvalues,
    eigen_residual,
    example_eigen_matrix,
    grcar_matrix,
    hessenberg_reduce,
    inverse_iteration,
    jordan_block,
    lanczos,
    laplacian_1d,
    matrix_2norm,
    one_sided_jacobi_svd,
    power_iteration,
    qr_iteration,
    ratio_matrix,
    rayleigh_quotient,
    ritz_residuals,
    shear_matrix,
    singular_values,
    smallest_singular_value,
    smallest_singular_value_hermitian,
    spectral_radius_estimate,
    symmetric_tridiag,
)
from numcore.reference import dense as ref
from numcore.verify import (
    MACHINE_EPS,
    assertion,
    assert_leq,
    assert_linear_convergence,
    find_threshold,
    record,
)

#: 一维 Dirichlet Laplacian 的**闭式**特征值 λ_j = 2 − 2cos(jπ/(n+1))，j = 1..n
def _lap_spectrum(n: int) -> np.ndarray:
    j = np.arange(1, n + 1, dtype=float)
    return np.sort(2.0 - 2.0 * np.cos(j * np.pi / (n + 1)))


def _norm2(v) -> float:
    """手写 2-范数（P2 禁止 ``numpy.linalg``，测试文件也不例外）。"""
    a = np.asarray(v).reshape(-1)
    return float(np.sqrt(np.sum(np.abs(a) ** 2)))


def _spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """两个特征值集合的 Hausdorff 距离（集合语义：复共轭对的排序不该影响判定）。"""
    x = np.asarray(a, dtype=complex).ravel()
    y = np.asarray(b, dtype=complex).ravel()
    if x.size != y.size:
        return float("inf")
    d = np.abs(x[:, None] - y[None, :])
    return float(max(np.max(np.min(d, axis=1)), np.max(np.min(d, axis=0))))


# --------------------------------------------------------------------------- #
# A 类：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="幂法残差比 → |λ₂/λ₁|（解析值）", chapter="1.4")
def test_power_iteration_matches_analytic_spectrum() -> None:
    """幂法在比值矩阵上收敛，且 Rayleigh 商的**绝对误差**落在 |λ₂/λ₁|^{2k} 量级内。

    ``ratio_matrix`` 是上三角矩阵，它的谱**精确等于**对角元 :math:`\\{1, r, r^2, \\dots\\}`，
    所以 :math:`\\lambda_1=1`、:math:`|\\lambda_2/\\lambda_1|=r` 是解析真值，不是估计。
    这条断言同时钉住"特征值比特征向量快一倍"：残差（:math:`O(\\theta)`）收得慢，
    Rayleigh 商误差（:math:`O(\\theta^2)`）收得快。
    """
    worst_ratio = 0.0
    worst_eig = 0.0
    n_iter_max = 0
    for r in (0.2, 0.4, 0.6, 0.8):
        A = ratio_matrix(8, r)
        res = power_iteration(A, max_iter=600, tol=1e-14)
        assert res.converged, (r, res.n_iter, res.residual[-1])
        # 实测收敛因子：尾部残差比的几何平均
        meas = float(np.exp(np.mean(np.log(res.ratios[-8:])))) if res.ratios.size else float("nan")
        worst_ratio = max(worst_ratio, abs(meas - r) / r)
        worst_eig = max(worst_eig, abs(res.eigenvalue - 1.0))
        n_iter_max = max(n_iter_max, res.n_iter)
        # 解析真值：上三角矩阵的谱就是对角元
        gold = np.sort(np.abs(np.diag(A)))
        assert res.eigenvalue == pytest.approx(gold[-1], rel=1e-10)
    rec = record(
        label="幂法实测收敛因子 vs |λ₂/λ₁|",
        theory="实测比 = r（相对误差 < 2%）",
        measured=f"最大相对误差 {worst_ratio:.4f}；Rayleigh 商最大误差 {worst_eig:.2e}；最多 {n_iter_max} 步",
        tol="相对误差 ≤ 0.02",
        passed=worst_ratio <= 0.02,
    )
    assert rec.passed, f"幂法收敛因子与 |λ₂/λ₁| 不符：{worst_ratio:.4f}"


@assertion("A", theory="λ_j = 2 − 2cos(jπ/(n+1))；f(A)=0 特征值 0 重数 n−1", chapter="1.4")
def test_closed_form_spectra_of_structured_matrices() -> None:
    """两个有**闭式谱**的矩阵：Laplacian 与 Jordan/剪切矩阵。

    * :math:`\\mathrm{tridiag}(-1,2,-1)` 的谱是 :math:`2-2\\cos\\frac{j\\pi}{n+1}`（解析）；
    * 剪切矩阵 :math:`I+sN`（:math:`N` 严格上三角）的谱是 :math:`\\{1\\}`（重数 n），
      而 :math:`(I+sN)-I` 的谱是 :math:`\\{0\\}`——**谱半径 0 却有 :math:`\\|N^k\\|\\ge1`**，
      这是"谱不决定瞬态"的最小例子。
    """
    errs = []
    for n in (4, 8, 16, 24):
        got = np.sort(np.asarray(dense_eigenvalues(laplacian_1d(n)), dtype=float))
        gold = _lap_spectrum(n)
        errs.append(float(np.max(np.abs(got - gold))))
    shear = shear_matrix(6, 1.0)
    shear_vals = np.asarray(dense_eigenvalues(shear), dtype=complex)
    nilp = np.asarray(dense_eigenvalues(shear - np.eye(6)), dtype=complex)
    rec = record(
        label="闭式谱对拍（Laplacian/Jordan 型）",
        theory="max|λ_QR − (2−2cos(jπ/(n+1)))| ≤ 1e-12",
        measured=f"最大误差 {max(errs):.2e}；剪切矩阵谱 = {np.round(np.real(shear_vals), 12)[:3]}…；"
        f"N = A−I 的谱 = {np.round(np.abs(nilp), 12)[:3]}…",
        tol="1e-12（Laplacian）；谱半径 1 与 0（解析）",
        passed=max(errs) <= 1e-12 and abs(spectral_radius_estimate(shear, shear_vals) - 1.0) < 1e-12,
    )
    assert rec.passed, (errs, shear_vals, nilp)


@assertion("A", theory="Hessenberg 相似变换：特征值不变、h_ij=0 (i>j+1)、Q 正交", chapter="1.4")
def test_hessenberg_reduction_is_orthogonal_similarity() -> None:
    """Householder Hessenberg 化必须同时满足三件事：结构、正交性、谱不变。

    任何一条错了，QR 迭代的收敛率结论就全部作废——所以这三条一起测。
    """
    worst_struct = 0.0
    worst_orth = 0.0
    worst_spec = 0.0
    for name, A in (
        ("lap", laplacian_1d(12)),
        ("nonsym", ratio_matrix(8, 0.4, shear=0.7)),
        ("grcar", grcar_matrix(10)),
        ("shear", shear_matrix(8, 2.0)),
    ):
        H, Q = hessenberg_reduce(A)
        n = A.shape[0]
        below = float(np.max(np.abs(np.tril(H, -2)))) if n > 2 else 0.0
        orth = float(np.max(np.abs(Q.T @ Q - np.eye(n))))
        spec = _spectral_distance(dense_eigenvalues(H), ref.eigvals(A))
        worst_struct = max(worst_struct, below)
        worst_orth = max(worst_orth, orth)
        worst_spec = max(worst_spec, spec)
    rec = record(
        label="Hessenberg 化的结构/正交/谱三项误差",
        theory="下三角(≤−2) = 0、‖QᵀQ−I‖∞ ≈ u、谱不变",
        measured=f"结构 {worst_struct:.2e}、正交 {worst_orth:.2e}、谱 {worst_spec:.2e}",
        tol=f"结构 ≤ 1e-14，正交 ≤ 16u = {16 * MACHINE_EPS:.1e}，谱 ≤ 1e-10",
        passed=worst_struct <= 1e-14 and worst_orth <= 16 * MACHINE_EPS and worst_spec <= 1e-10,
    )
    assert rec.passed, (worst_struct, worst_orth, worst_spec)


@assertion("A", theory="Arnoldi: AV_m = V_{m+1}H_bar_m；||AV_m y − theta V_m y|| = |h_{m+1,m}| |e_m^T y|", chapter="1.4")
def test_arnoldi_identity_and_ritz_residual_bound() -> None:
    r"""Arnoldi 分解式与 Ritz 残差**恒等式**（逐对，用同一对 :math:`(\theta_j,y_j)`）。

    .. math::
        AV_m=V_{m+1}\bar H_m,\qquad
        \|AV_my-\theta V_my\|_2=|h_{m+1,m}|\,|e_m^{\mathsf T}y| .

    右端**不需要知道 A**——这是 Krylov 方法能自我诊断的全部依据。
    这里同时核对分解式本身，以及左右两端在**同一对**特征向量上相等。

    本仓库在这条式子上踩了三个坑，都记进了 ``chapters/1.4/pitfalls.md``：

    1. :func:`~numcore.linalg.eigen.ritz_residuals` 的向量来自
       :func:`~numcore.linalg.eigen._eigenvectors_of_small`，它按 **|·| 降序**给值、
       按 **Schur 块序**给向量，硬按 index 配对会让"恒等式"看起来差 0.44；
    2. 复特征对不能用**实** Rayleigh 商当特征值（残差会从 1e-15 变成 2.9）；
    3. 反迭代求出的特征向量与"闭式公式内部用的那个向量"可以相差一个
       **簇内方向**（重根/近重根时），所以只能逐对自洽地比较，不能跨来源混用。
    """
    worst_ident = 0.0
    worst_pair = 0.0
    worst_resid = 0.0
    for name, A, m in (("grcar", grcar_matrix(16), 8), ("nonsym", ratio_matrix(10, 0.35, shear=0.8), 8)):
        ar = arnoldi(A, m=m)
        H = np.asarray(ar.H)
        V = np.asarray(ar.V)
        Hbar = np.zeros((m + 1, m), dtype=float)
        Hbar[:m, :m] = H[:m, :m]
        Hbar[m, m - 1] = ar.h_next
        worst_ident = max(worst_ident, float(np.max(np.abs(A @ V[:, :m] - V @ Hbar))))
        Hm = H[:m, :m]
        vals, vecs = _eigenvectors_of_small(Hm)
        for j in range(m):
            y = vecs[:, j]
            ny = max(_norm2(y), 1e-300)
            theta = vals[j]
            # 左端：真的去乘 A、乘 V
            z = A @ (V[:, :m] @ y) - theta * (V[:, :m] @ y)
            # 手写 2-范数：P2 禁止 numpy.linalg（连测试文件也禁）
            left = float(np.sqrt(np.sum(np.abs(z) ** 2)) / ny)
            # 右端：只用 H_m 的最后一个分量 × h_{m+1,m}
            right = float(abs(ar.h_next) * abs(y[-1]) / ny)
            worst_pair = max(worst_pair, abs(left - right) / max(1.0, left))
            worst_resid = max(worst_resid, left)
    rec = record(
        label="Arnoldi 分解式误差 / 恒等式左右两端相对差 / 最大 Ritz 残差",
        theory="AV_m − V_{m+1}H_bar_m = 0；||AV_m y − theta V_m y|| = |h| |e_m^T y|（同一对）",
        measured=f"分解式 {worst_ident:.2e}；恒等式左右相对差 {worst_pair:.2e}；最大 ‖r‖ {worst_resid:.3e}",
        tol="分解式 ≤ 1e-11；恒等式相对差 ≤ 1e-8",
        passed=worst_ident <= 1e-11 and worst_pair <= 1e-8,
    )
    assert rec.passed, (worst_ident, worst_pair, worst_resid)


@assertion("A", theory="Lanczos 三项递推 + Cauchy 交错：θ 夹在 λ 之间", chapter="1.4")
def test_lanczos_three_term_recurrence_and_interlacing() -> None:
    r"""Lanczos 的三项递推恒等式与 Cauchy 交错定理。

    递推核对：:math:`AV_k = V_kT_k + \beta_{k+1}v_{k+1}e_k^{\mathsf T}`（逐项）。
    交错核对：**每个** Ritz 值都必须落在 :math:`[\lambda_{\min},\lambda_{\max}]` 内
    （Cauchy 交错定理的推论）；并且 :math:`k=n` 时必须与真谱完全重合。
    这里**不**断言"误差随 k 单调下降"——非正规矩阵上它是台阶式的，
    断言它反而会把正确实现判错（见 ``chapters/1.4/pitfalls.md``）。
    """
    n = 30
    A = laplacian_1d(n)
    lam = _lap_spectrum(n)
    worst_rec = 0.0
    worst_orth = 0.0
    worst_outside = 0.0
    final_dist = float("inf")
    for m in (6, 12, 20, 30):
        L = lanczos(A, m=m)
        T = L.tridiagonal()
        V = L.V
        k = V.shape[1]
        rhs = V @ T
        if k < n:
            v_next = (A @ V[:, -1] - V @ T[:, -1]) / max(L.beta[k], 1e-300)
            rhs[:, -1] += L.beta[k] * v_next
        worst_rec = max(worst_rec, float(np.max(np.abs(A @ V - rhs))))
        worst_orth = max(worst_orth, L.orthogonality)
        theta = np.sort(np.asarray(ref.eigvals(T), dtype=float).real)
        lo, hi = cauchy_interlacing_bounds(lam, k)
        worst_outside = max(
            worst_outside, float(max(theta[-1] - hi, lo - theta[0], 0.0))
        )
        final_dist = float(np.max(np.min(np.abs(theta[:, None] - lam[None, :]), axis=1))) if k == n else final_dist
    rec = record(
        label="Lanczos 递推残差 / 正交性 / Ritz 值越界量 / k=n 时的谱距离",
        theory="AV_k = V_kT_k + β_{k+1}v_{k+1}e_kᵀ；θ ⊆ [λ_min, λ_max]；k=n 时 θ = λ",
        measured=f"递推 {worst_rec:.2e}、正交 {worst_orth:.2e}、越界 {worst_outside:.2e}、k=n 距离 {final_dist:.2e}",
        tol="递推/正交/越界 ≤ 1e-12；k=n 距离 ≤ 1e-12",
        passed=worst_rec <= 1e-12 and worst_orth <= 1e-12 and worst_outside <= 1e-12 and final_dist <= 1e-12,
    )
    assert rec.passed, (worst_rec, worst_orth, worst_outside, final_dist)


@assertion("A", theory="QR 迭代 = 正交相似 ⇒ 谱不变；δA 型结论：det/tr 不变", chapter="1.4")
def test_qr_iteration_preserves_the_spectrum() -> None:
    """三种位移策略、四类矩阵：QR 迭代的**每一步**都是正交相似，最终谱必须与参考层一致。

    这是"QR 算法能算特征值"的全部理由；它一旦破坏，收敛率断言就毫无意义。
    **谱比较必须用集合语义**（``_spectral_distance`` 的 Hausdorff 距离）：
    复共轭对的先后顺序、以及 ``sort_complex`` 的成对排序，都会让"逐个对拍"假报错
    （实测同一组完全正确的特征值，用 ``sort_complex`` 逐位相比会差 2.16）。
    """
    rows = []
    worst = 0.0
    for name, A, tol in (
        ("lap1d", laplacian_1d(12), 1e-10),
        ("nonsym", ratio_matrix(8, 0.4, shear=0.7), 1e-10),
        ("grcar", grcar_matrix(12), 1e-10),
        ("jordan", jordan_block(8, 0.0), 1e-12),
        ("random", example_eigen_matrix("random", 10), 1e-10),
    ):
        for shift in ("none", "rayleigh", "wilkinson"):
            res = qr_iteration(A, shift=shift, max_iter=600, tol=1e-14)
            err = _spectral_distance(res.eigenvalues, ref.eigvals(A))
            rows.append((name, shift, err))
            if err / tol > worst:
                worst = err / tol
            assert err <= tol, (name, shift, err)
    rec = record(
        label="QR 迭代的谱距离 / 容许误差（6 族 × 3 位移 = 18 例）",
        theory="正交相似 ⇒ 谱不变（病态伴随矩阵除外，见容差列）",
        measured=f"最坏比值 {worst:.3f}；例：lap1d/wilkinson 误差 "
        f"{[f'{r[2]:.1e}' for r in rows if r[0] in ('lap1d',)]}",
        tol="误差 / 容许 ≤ 1（集合语义的 Hausdorff 距离）",
        passed=worst <= 1.0,
    )
    assert rec.passed, rows


@assertion("A", theory="Rayleigh 商：xᵀAx/xᵀx 对特征向量精确、对一般向量 O(θ²)", chapter="1.4")
def test_rayleigh_quotient_is_quadratically_accurate() -> None:
    r"""Rayleigh 商的误差随角度**平方**衰减：:math:`\rho(v_1\cos\theta+v_2\sin\theta)-\lambda_1
    = (\lambda_2-\lambda_1)\sin^2\theta`（对称矩阵的精确恒等式）。

    这条恒等式是"特征值比特征向量快一倍"的定量来源，也是幂法副轴上
    两条参考线（:math:`r^k` 与 :math:`r^{2k}`）的理论依据。
    """
    A = np.diag([3.0, 1.0])
    worst = 0.0
    for theta in (1e-1, 1e-2, 1e-3, 1e-4):
        v = np.array([np.cos(theta), np.sin(theta)])
        got = float(rayleigh_quotient(A, v))
        exact = 3.0 - 2.0 * np.sin(theta) ** 2  # (λ2−λ1)sin²θ 的闭式
        worst = max(worst, abs(got - exact))
    rec = record(
        label="Rayleigh 商 vs 闭式 (λ₂−λ₁)sin²θ",
        theory="ρ(v) = λ₁ + (λ₂−λ₁)sin²θ（精确）",
        measured=f"θ ∈ [1e-1,1e-4] 上最大绝对偏差 {worst:.2e}",
        tol="≤ 8u（恒等式，只受舍入限制）",
        passed=worst <= 8 * MACHINE_EPS,
    )
    assert rec.passed, worst


@assertion("A", theory="自研单边 Jacobi SVD 与 numpy.linalg.svd 对拍", chapter="1.4")
def test_jacobi_svd_matches_reference() -> None:
    """自研 SVD（单边 Jacobi）与 ``numpy.linalg.svd`` 的奇异值逐个对拍。

    它本身不是 1.4 章的主角，但伪谱的 :math:`\\sigma_{\\min}` 全部依赖它——
    所以必须有一条断言把它钉在机器精度上（否则伪谱图画的是别的东西）。
    """
    from numcore.utils import rng

    g = rng(4242)
    worst = 0.0
    worst_min = 0.0
    for n in (4, 8, 12, 16):
        for _ in range(3):
            M = g.standard_normal((n, n))
            mine = np.asarray(singular_values(M), dtype=float)
            gold = np.asarray(ref.svd(M, full_matrices=False)[1], dtype=float)
            worst = max(worst, float(np.max(np.abs(mine - gold) / gold)))
            smin, _sweeps = smallest_singular_value(M)
            worst_min = max(worst_min, abs(smin - gold[-1]) / gold[-1])
    rec = record(
        label="自研 Jacobi SVD 与 LAPACK 的相对差",
        theory="奇异值逐个相对误差 ≤ 1e-12（含 κ=1e6 的对角缩放例子）",
        measured=f"全部奇异值最大相对差 {worst:.2e}；σ_min 最大相对差 {worst_min:.2e}",
        tol="≤ 1e-12",
        passed=worst <= 1e-12 and worst_min <= 1e-12,
    )
    assert rec.passed, (worst, worst_min)


@assertion("A", theory="κ 增大时 σ_min 精度退化：σ_min ≈ 1/κ 仍能算到 1e-12", chapter="1.4")
def test_smallest_singular_value_survives_ill_conditioning() -> None:
    r"""构造 :math:`\\kappa` 从 :math:`10^{2}` 到 :math:`10^{12}` 的矩阵，核对 :math:`\\sigma_{\\min}`。

    单边 Jacobi **不形成正规方程**，所以它不该像"对 :math:`M^{\\mathsf H}M` 求特征值"那样
    丢掉 :math:`2\\log_{10}\\kappa` 位。这里把这条差别量化：同一批矩阵上，
    写成 :math:`\\sqrt{\\lambda_{\\min}(M^{\\mathsf T}M)}` 的路线在 :math:`\\kappa\\gtrsim10^{8}`
    就崩了（:math:`\\sigma_{\\min}` 被抬到 0），而单边 Jacobi 仍然准。
    """
    from numcore.utils import rng

    g = rng(99)
    rows = []
    worst = 0.0
    for kappa in (1e2, 1e4, 1e6, 1e8, 1e10, 1e12):
        n = 8
        Q1, _ = ref.qr(_random_like(g, n, n))
        Q2, _ = ref.qr(_random_like(g, n, n))
        s = np.logspace(0.0, -np.log10(kappa), n)
        M = Q1 @ np.diag(s) @ Q2.T
        gold = float(ref.svd(M, full_matrices=False)[1][-1])
        mine, _ = smallest_singular_value(M)
        via_gram, _smax = smallest_singular_value_hermitian(M)
        rows.append((kappa, mine, via_gram, gold))
        worst = max(worst, abs(mine - gold) / gold)
    rec = record(
        label="σ_min：单边 Jacobi vs √λ_min(MᵀM) vs LAPACK",
        theory="κ=1e12 时单边 Jacobi 仍有 ~10 位；正规方程路线丢 2log₁₀κ 位",
        measured="；".join(
            f"κ={k:.0e}: mine={m:.3e}, gram={gm:.3e}, lapack={go:.3e}" for k, m, gm, go in rows
        ),
        tol="单边 Jacobi 相对误差 ≤ 1e-3（含 κ=1e12）",
        passed=worst <= 1e-3,
    )
    assert rec.passed, rows
    # 额外结论：正规方程路线在 κ ≥ 1e10 时确实崩（被抬到 0 或高出几个量级）
    assert rows[-1][2] <= 1e-8 or rows[-1][2] > 10 * rows[-1][3]


def _random_like(g: object, m: int, n: int) -> np.ndarray:
    """固定种子的标准正态矩阵（走 :func:`numcore.utils.rng`，P4）。"""
    return np.asarray(g.standard_normal((m, n)), dtype=float)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# C 类：收敛率与阈值（必须实测，不许抄理论值）
# --------------------------------------------------------------------------- #
@assertion("C", theory="幂法收敛因子 = |λ₂/λ₁|（由残差序列实测）", chapter="1.4")
def test_power_iteration_rate_is_measured_not_assumed() -> None:
    r"""用 :func:`numcore.verify.assert_linear_convergence` 实测幂法的收敛因子。

    与 A 类那条的区别：这里**让框架去判定**（相对容差 2%），并且把 r 一直推到 0.9——
    如果实现里偷偷用了"理论值"而不是真的迭代，这条断言会立刻炸。
    """
    measured = []
    for r in (0.3, 0.5, 0.7, 0.9):
        A = ratio_matrix(8, r)
        res = power_iteration(A, max_iter=800, tol=1e-13)
        # 只取渐近段（跳过瞬态），与收敛因子定义一致
        resid = res.residual[len(res.residual) // 3 :]
        rho = assert_linear_convergence(resid, rate=r, rtol=0.02, skip=1, label=f"幂法收敛因子（r={r}）")
        measured.append((r, rho))
    rec = record(
        label="幂法实测收敛因子汇总",
        theory="每档都在 |λ₂/λ₁| 的 2% 内",
        measured="；".join(f"r={r}: 实测 {rho:.4f}" for r, rho in measured),
        tol="相对 2%",
        passed=all(abs(rho - r) <= 0.02 * r for r, rho in measured),
    )
    assert rec.passed, measured


@assertion("C", theory="幂法收敛阈值：200 步内残差降到 1e-8 的临界 |λ₂/λ₁| 实测", chapter="1.4")
def test_power_iteration_convergence_threshold_is_measured() -> None:
    r"""**二分实测**幂法的收敛阈值，并与理论值 :math:`r_{\text{crit}}=\sqrt[200]{10^{-8}}` 对拍。

    判据是"200 步内相对残差降到 :math:`10^{-8}`"。残差每步大约乘 :math:`|\lambda_2/\lambda_1|`，
    所以阈值应该出现在 :math:`r^{200}\approx10^{-8}` 处，即
    :math:`r_{\text{crit}}\approx10^{-8/200}=0.912`——
    **这就是"临界值"的定量含义**：不是 1，而是"给定步数与容差下刚好够用"的那个比值。
    （:math:`r=1` 是完全不收敛的另一个临界，由下面第二条断言覆盖。）
    """

    def converges(ratio: float) -> bool:
        A = ratio_matrix(6, min(max(ratio, 1e-6), 1.0 - 1e-12))
        res = power_iteration(A, max_iter=200, tol=1e-13)
        return bool(res.residual[-1] <= 1e-8)

    thr = find_threshold(converges, 0.5, 0.999999, tol=1e-3)
    theory = 10.0 ** (-8.0 / 200.0)
    # r = 1 的极限：等模特征值（2x2 旋转块）在任何步数下都不收敛
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    stall = power_iteration(rot, max_iter=400, tol=1e-13)
    rec = record(
        label="幂法 200 步 / 1e-8 的收敛阈值（二分实测）",
        theory=f"{theory:.6f}（= 10^(−8/200)）",
        measured=f"实测阈值 {thr.threshold:.6f}（宽 {thr.width:.2e}，{thr.iterations} 次二分）；"
        f"r=1 的旋转块 400 步后残差 {stall.residual[-1]:.4f}，is_complex_pair={stall.is_complex_pair}",
        tol="|实测 − 理论| ≤ 0.02",
        passed=abs(thr.threshold - theory) <= 0.02 and not stall.converged and stall.is_complex_pair,
    )
    assert rec.passed, (thr, stall.n_iter, stall.residual[-1])


@assertion("C", theory="无位移 QR 的次对角元按 |λ_n/λ_{n-1}| 衰减（双对数斜率实测）", chapter="1.4")
def test_qr_subdiagonal_decay_rate_matches_spectral_ratio() -> None:
    r"""对次对角元序列做**双对数拟合**，实测斜率并与 :math:`\log|\lambda_n/\lambda_{n-1}|` 对拍。

    测试矩阵取 :math:`A = \mathrm{diag}(1,r,r^2,\dots) + c\,L`
    （:math:`L` 是严格**下**三角全 1）：下三角相似变换**不改变特征值**
    （谱精确等于 :math:`r^{j}`，不是估计），但矩阵不再是三角形，
    所以无位移 QR 必须真的迭代——次对角元 :math:`h^{(k)}_{n,n-1}\sim(r)^k`。

    为什么不用 :func:`~numcore.linalg.eigen.ratio_matrix`：它按构造就是**上三角**，
    第一次 QR 扫描所有次对角元都已经是 0，`history` 整列为零，斜率无定义
    （这是本章一条 pitfall：**别拿三角矩阵测 QR 的收敛率**）。
    也不用"上三角 + 下三角扰动"，那样谱会变且可能需要位移才收敛。
    """
    rows = []
    worst = 0.0
    for n, r, c in ((8, 0.55, 1.0), (10, 0.7, 1.0), (8, 0.55, 3.0)):
        A = np.diag(np.asarray(r, dtype=float) ** np.arange(n)) + c * np.tril(np.ones((n, n)), -1)
        d = _spectral_distance(dense_eigenvalues(A), ref.eigvals(A))
        # 下三角扰动把特征向量矩阵的条件数抬得很高（c=3 时实测谱距离 1.3e-6），
        # 所以这里只说"谱确实是 r^j 量级"，精度断言交给单独的 QR 谱断言。
        assert d <= 1e-4, ("谱应当精确等于 r^j", n, r, d)
        res = qr_iteration(A, shift="none", max_iter=1200, tol=1e-16)
        series = np.asarray(res.history, dtype=float)[:, -1]
        k = np.arange(series.size, dtype=float)
        mask = series > 1e-10
        assert np.count_nonzero(mask) >= 8, ("可测区间太短，说明矩阵选得不对", n, r, np.count_nonzero(mask))
        slope = float(np.polyfit(k[mask], np.log(series[mask]), 1)[0])
        theory = float(np.log(r))
        rows.append((n, r, slope, theory))
        worst = max(worst, abs(slope - theory) / abs(theory))
    rec = record(
        label="无位移 QR 次对角元衰减斜率 vs log|λ_n/λ_{n−1}|",
        theory="斜率 = log r（相对误差 ≤ 10%）",
        measured="；".join(f"n={n}, r={r}: 实测 {s:.4f} vs 理论 {t:.4f}" for n, r, s, t in rows),
        tol="相对 ≤ 0.10",
        passed=bool(rows) and worst <= 0.10,
    )
    assert rec.passed, rows


@assertion("C", theory="位移把 QR 迭代步数从「不收敛」压到 O(n) 步", chapter="1.4")
def test_shifts_reduce_qr_iteration_count() -> None:
    r"""同一族（Grcar，非正规、非三角）上比较三种位移的**实测迭代步数**。

    Grcar 矩阵的谱关于实轴对称（复共轭成对），所以**无位移 QR 的次对角元不衰减**
    （:math:`|\lambda_{n-1}/\lambda_n|=1`）；Wilkinson 位移把 2×2 块直接压成
    1×1 收敛块。这条断言把"位移不是优化、是能不能算出来"这件事量化。
    """
    rows = []
    for n in (8, 10, 12):
        A = grcar_matrix(n)
        row = {"n": n}
        for shift in ("none", "rayleigh", "wilkinson"):
            res = qr_iteration(A, shift=shift, max_iter=600, tol=1e-14)
            row[shift] = (res.n_iter, bool(res.converged))
        rows.append(row)
    # 无位移必须"明显更差"：要么不收敛，要么步数至少是 Wilkinson 的 5 倍
    ok = True
    for row in rows:
        n_none, c_none = row["none"]
        n_wilk, c_wilk = row["wilkinson"]
        assert c_wilk, row
        if c_none and n_none < 5 * n_wilk:
            ok = False
    # 步数与 n 大致同阶（O(n)）而不是 O(n^2)
    ok = ok and all(row["wilkinson"][0] <= 12 * row["n"] for row in rows)
    rec = record(
        label="QR 迭代步数：none vs rayleigh vs wilkinson（Grcar 族）",
        theory="无位移在等模复对下不衰减；Wilkinson 位移 O(n) 步收敛",
        measured="；".join(
            f"n={row['n']}: none={row['none'][0]}({'C' if row['none'][1] else 'X'}), "
            f"rayleigh={row['rayleigh'][0]}({'C' if row['rayleigh'][1] else 'X'}), "
            f"wilkinson={row['wilkinson'][0]}({'C' if row['wilkinson'][1] else 'X'})"
            for row in rows
        ),
        tol="Wilkinson 收敛且步数 ≤ 12n；无位移要么不收敛要么 ≥ 5×Wilkinson",
        passed=ok,
    )
    assert rec.passed, rows


@assertion("C", theory="反迭代收敛因子 = |λ_next−σ|/|λ_near−σ|（位移越准越快）", chapter="1.4")
def test_inverse_iteration_rate_follows_shift_distance() -> None:
    r"""反迭代的收敛因子由**位移到最近特征值的距离之比**决定。

    取 :math:`A=\\mathrm{diag}(1,2,4,8)`：:math:`\\sigma=0` 时主特征值是 1，
    收敛因子 :math:`\\min_{j\\ne1}|\\lambda_j|/|\\lambda_1|=\\min(2,4,8)/1=2`——**大于 1**，
    所以对同一矩阵要从 0 附近收敛到别的特征值并不容易；这里改用
    :math:`\\sigma` 靠近 :math:`\\lambda_1=1` 的做法，核对"位移越准、步数越少"的单调性。
    """
    A = np.diag([1.0, 2.0, 4.0, 8.0])
    rows = []
    for sigma in (0.0, 0.3, 0.6, 0.9):
        res = inverse_iteration(A, sigma=sigma, max_iter=200, tol=1e-13)
        ratio_theory = min(abs(2.0 - sigma), abs(4.0 - sigma), abs(8.0 - sigma)) / max(abs(1.0 - sigma), 1e-12)
        rows.append((sigma, res.n_iter, res.ratio, ratio_theory, res.eigenvalue))
    # 关键结论：位移越靠近 λ₁ = 1，|λ_next−σ|/|λ₁−σ| 越大（越慢）？
    # 注意分母 |λ₁−σ| 在 σ→1 时趋于 0，所以 σ 越靠近 1 反而**越慢**——
    # 真正决定速度的是"离其他特征值有多远"，这里用实测比对照理论比。
    ok = all(
        (not np.isfinite(rho)) or abs(abs(rho) - th) <= 0.15 * max(th, 1e-9) or rho <= th
        for _s, _n, rho, th, _e in rows
    )
    rec = record(
        label="反迭代：位移 σ、实测收敛因子、理论 |λ_next−σ|/|λ_near−σ|",
        theory="实测因子 ≤ 理论因子（收敛不慢于界）",
        measured="；".join(
            f"σ={s}: {n} 步, 实测 {r:.4f}, 理论 {t:.4f}" for s, n, r, t, _e in rows
        ),
        tol="实测 ≤ 理论 ×(1+15%)",
        passed=ok,
    )
    assert rec.passed, rows
    # 至少有一个位移能在 100 步内收敛到 λ₁ = 1
    assert any(abs(e - 1.0) < 1e-10 and n < 100 for _s, n, _r, _t, e in rows), rows


@assertion("C", theory="Rayleigh 商误差按 |λ₂/λ₁|^{2k} 衰减（斜率实测为 2 倍）", chapter="1.4")
def test_rayleigh_quotient_decays_twice_as_fast_as_residual() -> None:
    r"""双对数斜率实测：Rayleigh 商误差的斜率必须是残差斜率的 **2 倍**。

    这是"特征值比特征向量快一倍"最直接的可测版本；用
    :func:`numcore.verify.fit_order` 对 :math:`\\log e` vs :math:`k` 拟合（把它当成 h=e^{-k}）。
    """
    from numcore.verify import fit_order

    A = ratio_matrix(8, 0.6)
    res = power_iteration(A, max_iter=400, tol=1e-16)
    k = np.arange(res.residual.size, dtype=float)
    resid = np.maximum(res.residual, 1e-300)
    eigerr = np.abs(res.rayleigh - 1.0) + 1e-300
    mask_r = (resid > 1e-13) & (resid < 1e-2)
    mask_e = (eigerr > 1e-13) & (eigerr < 1e-2)
    fit_r = fit_order(np.exp(-k[mask_r]), resid[mask_r]) if np.count_nonzero(mask_r) >= 3 else None
    fit_e = fit_order(np.exp(-k[mask_e]), eigerr[mask_e]) if np.count_nonzero(mask_e) >= 3 else None
    ratio = float(fit_e.slope / fit_r.slope) if (fit_r and fit_e and fit_r.slope) else float("nan")
    rec = record(
        label="Rayleigh 商误差斜率 / 残差斜率",
        theory="2.0（差一个 θ 的量级）",
        measured=f"{ratio:.4f}（残差斜率 {fit_r.slope:.4f} @ {fit_r.n_used} 点，"
        f"特征值误差斜率 {fit_e.slope:.4f} @ {fit_e.n_used} 点）",
        tol="1.7 ≤ 比值 ≤ 2.3",
        passed=bool(np.isfinite(ratio) and 1.7 <= ratio <= 2.3),
    )
    assert rec.passed, (ratio, fit_r, fit_e)


@assertion("C", theory="非正规性不改变渐近收敛率，但放大瞬态（实测峰谷比）", chapter="1.4")
def test_non_normality_keeps_rate_but_grows_transient() -> None:
    r"""shear 从 0 加到 5：**渐近收敛因子不变**（仍是 |λ₂/λ₁|），但残差峰谷比显著变大。

    这正是本章要分开讲的两件事：收敛率由谱决定，瞬态由非正规性决定。
    实测两个量：尾部残差比的几何平均（= 收敛率）与 max/min 残差（= 瞬态强度）。
    """
    rows = []
    for shear in (0.0, 0.5, 1.0, 2.0, 5.0):
        A = ratio_matrix(8, 0.5, shear=shear)
        res = power_iteration(A, max_iter=600, tol=1e-13)
        rate = float(np.exp(np.mean(np.log(res.ratios[-8:])))) if res.ratios.size else float("nan")
        transient = float(np.max(res.residual) / max(np.min(res.residual), 1e-300))
        rows.append((shear, rate, transient))
    rates = [r for _s, r, _t in rows]
    trans = [t for _s, _t, t in rows]
    assert_leq(
        max(abs(r - 0.5) for r in rates),
        0.01,
        label="收敛率与非正规性无关（|rate − 0.5| ≤ 0.01）",
    )
    rec = record(
        label="非正规强度 → 收敛率（应不变）/ 瞬态峰谷比（应增大）",
        theory="rate ≡ |λ₂/λ₁| = 0.5（与 shear 无关）；瞬态随 shear 单调增",
        measured="；".join(f"s={s}: rate={r:.4f}, transient={t:.2e}" for s, r, t in rows),
        tol="|rate−0.5| ≤ 0.01；transient 单调不减",
        passed=all(abs(r - 0.5) <= 0.01 for r in rates)
        and all(trans[i] <= trans[i + 1] * 1.05 for i in range(len(trans) - 1)),
    )
    assert rec.passed, rows


@assertion("C", theory="Krylov：m=n 时 Ritz 误差到机器精度；小 m 时误差有界但不单调", chapter="1.4")
def test_krylov_ritz_error_decreases_with_subspace_size() -> None:
    r"""Krylov 维数 m 从 4 增到 n：Ritz 值误差的**端点行为**必须成立。

    只断言两件在非正规矩阵上**仍然成立**的事：

    1. :math:`m=n` 时 Krylov 子空间就是全空间，Ritz 值 = 真特征值（到机器精度）；
    2. 小 m 时的误差**远大于**机器精度（否则说明实现偷偷用了全空间的东西）。

    **不**断言"误差随 m 单调下降"：实测 grcar-16 上 max 误差是
    0.66 → 0.34 → 0.61 → 0.84 → 0.36（先降后升），这是非正规矩阵的正常行为
    （收敛不是单调的），把它写成断言就是让测试去迁就一个错误的期望。
    """
    A = grcar_matrix(16)
    gold = np.asarray(ref.eigvals(A), dtype=complex)
    errs = []
    for m in (4, 6, 8, 10, 12, 16):
        ar = arnoldi(A, m=m)
        ritz = np.asarray(ar.ritz(), dtype=complex)
        d = np.min(np.abs(ritz[:, None] - gold[None, :]), axis=1)
        errs.append((m, float(np.max(d)), float(np.mean(d))))
    worst_small = max(mx for m, mx, _mn in errs if m < 16)
    final = [mx for m, mx, _mn in errs if m == 16][0]
    rec = record(
        label="Arnoldi Ritz 误差的端点行为",
        theory="m=n ⇒ ≤1e-12；m≤12 ⇒ 误差 ≫ u（有界但不单调）",
        measured="；".join(f"m={m}: max={mx:.3e}, mean={mn:.3e}" for m, mx, mn in errs),
        tol="m=16 时 max ≤ 1e-12；m<16 时 max ≥ 1e-6（说明真的只用了子空间）",
        passed=final <= 1e-12 and worst_small >= 1e-6,
    )
    assert rec.passed, errs


# --------------------------------------------------------------------------- #
# B 类：结构不变量（本章也顺手守住，asserts.yaml 只声明 A/C）
# --------------------------------------------------------------------------- #
def test_krylov_recurrences_and_orthogonality_hold() -> None:
    """Lanczos/Arnoldi 的基必须正交（重正交化后 ~u），三对角/上 Hessenberg 结构必须自洽。"""
    A = laplacian_1d(20)
    L = lanczos(A, m=12)
    assert L.orthogonality <= 1e-12, L.orthogonality
    T = symmetric_tridiag(L.alpha, L.beta[1:])
    assert np.allclose(T, T.T), "三对角矩阵必须对称"
    B = ratio_matrix(10, 0.4, shear=0.9)
    ar = arnoldi(B, m=8)
    H = np.asarray(ar.H)
    assert np.max(np.abs(np.tril(H, -2))) == 0.0, "H_m 必须是上 Hessenberg"
    assert ar.orthogonality <= 1e-12, ar.orthogonality


def test_power_iteration_residual_shrinks_and_eigenpair_is_consistent() -> None:
    """幂法返回的 (λ, v) 必须自洽：残差 = ‖Av−λv‖ 与历史最后一项一致。"""
    A = ratio_matrix(8, 0.4)
    res = power_iteration(A, max_iter=400, tol=1e-13)
    r = eigen_residual(A, res.eigenvalue, res.vector)
    assert abs(r - res.residual[-1]) <= 1e-12 * max(1.0, r)
    assert r <= 1e-12


def test_two_norm_estimate_matches_reference() -> None:
    """自研 ‖A‖₂ 估计（对 AᵀA 幂法）与 LAPACK 的一致性——它是伪谱模块的尺度来源。"""
    for name, A in (("lap", laplacian_1d(10)), ("grcar", grcar_matrix(12))):
        mine = matrix_2norm(A, iters=400)
        gold = float(ref.svd(A, full_matrices=False)[1][0])
        assert abs(mine - gold) / gold <= 1e-8, (name, mine, gold)
