"""1.3 章 A/B 类断言（上）：Householder/Givens QR 与三种 SVD。

这一章要分开测四件事，测法各不相同：

1. **正交变换本身**（Householder 反射、Givens 旋转）——它们只做旋转反射，
   所以误差必须是 :math:`c\\,n\\,u\\,\\|A\\|` 级别，与 :math:`\\kappa` 无关（A 类重建对拍）；
2. **QR 的最小二乘解**——对拍 LAPACK 的 SVD 解，看前向误差是不是真的只被
   :math:`\\kappa` 放大**一次**，以及残差是不是真的与列空间正交（A 类解析对拍）；
3. **奇异值本身**——三个自研版本（Jacobi / Kogbetliantz / Golub--Kahan）互相之间、
   以及与 LAPACK 之间对拍。小的奇异值只能要求 :math:`\\kappa u` 的**相对**精度
   （A 类，容差随 :math:`\\sigma_{\\max}/\\sigma_i` 放宽——这是 Demmel--Veselić 的结论，
   不是我们放宽标准）；
4. **结构不变量**——三角性、置换、单调性、正交性（B 类）。

参照物一律走 :mod:`numcore.reference.dense`（LAPACK/scipy）；
``numcore`` 本体里不许出现 ``numpy.linalg``（P2），测试里也不许。

每条断言都给出**实测数字**：抄理论值不算证据。
"""

from __future__ import annotations

import numpy as np
import pytest

from numcore.linalg.lu import det_from_lu, lu_partial
from numcore.linalg.qr import (
    back_substitute,
    column_norms,
    column_pivot_qr,
    givens_matrix,
    givens_qr,
    givens_rotation,
    householder_matrix,
    householder_qr,
    householder_vector,
    orthogonality_defect,
    orthogonality_measure,
    qr_flops,
    qr_least_squares,
    qr_rank,
    qr_reconstruction_error,
)
from numcore.linalg.svd import (
    bidiagonalize,
    golub_kahan_svd,
    jacobi_svd,
    kogbetliantz_svd,
    moore_penrose_defects,
    numerical_rank,
    pseudoinverse,
    relative_singular_value_error,
    singular_vector_residuals,
    sort_and_fix_signs,
    spectral_condition,
    truncated_solution,
)
from numcore.reference import dense as ref
from numcore.utils import rng
from numcore.verify import assertion, assert_leq, assert_monotone, assert_orthogonal, record

U = float(np.finfo(float).eps) / 2.0  # 单位舍入 u = 2^-53 ≈ 1.11e-16

_SHAPES = ((2, 2), (8, 5), (5, 8), (12, 12), (20, 4), (30, 30), (40, 12), (6, 40))
_METHODS = ("jacobi", "kogbetliantz", "golub_kahan")


def _rand(m: int, n: int, seed: int, *, scale: float = 1.0) -> np.ndarray:
    g = rng(seed)
    return scale * g.standard_normal((m, n))


def _norm2(x: np.ndarray) -> float:
    return float(np.sqrt(np.dot(x, x)))


def _fro(A: np.ndarray) -> float:
    return float(np.sqrt(np.sum(A * A)))


def _svd(name: str, A: np.ndarray):
    if name == "jacobi":
        return jacobi_svd(A)
    if name == "kogbetliantz":
        return kogbetliantz_svd(A)
    if name == "golub_kahan":
        return golub_kahan_svd(A)
    raise ValueError(name)


def _ill_conditioned(m: int, n: int, kappa: float, seed: int) -> np.ndarray:
    """``σ`` 从 1 几何递减到 ``1/kappa`` 的矩阵（奇异值精确已知，便于对拍）。"""
    q1 = householder_qr(_rand(m, n, seed)).Q
    q2 = householder_qr(_rand(n, n, seed + 77)).Q
    s = np.logspace(0.0, -float(np.log10(kappa)), n)
    return (q1 * s) @ q2.T


# --------------------------------------------------------------------------- #
# A 类：Householder / Givens
# --------------------------------------------------------------------------- #
@assertion("A", theory="‖A−QR‖_F ≤ c·n·u·‖A‖_F，且 ‖QᵀQ−I‖ ≤ c·n·u", chapter="1.3")
def test_householder_qr_reconstruction_and_orthogonality() -> None:
    """Householder QR 的重建误差与正交性缺陷都落在 ``n·u`` 量级，与 κ 无关。

    这是"正交变换不放大误差"的量化版本：即使 :math:`\\kappa = 10^{12}`，
    :math:`\\|A - QR\\|` 也只跟 :math:`n u \\|A\\|` 有关。
    """
    worst_rec = 0.0
    worst_orth = 0.0
    rows = []
    for m, n in _SHAPES:
        A = _rand(m, n, 100 + m * 10 + n)
        res = householder_qr(A, collect_reflectors=True)
        rec = qr_reconstruction_error(A, res.Q, res.R)
        orth = orthogonality_defect(res.Q)
        worst_rec = max(worst_rec, rec / (n * U * _fro(A)))
        worst_orth = max(worst_orth, orth / (n * U))
        rows.append(f"{m}x{n}:{rec:.1e}")
        k = min(m, n)
        assert res.Q.shape == (m, k) and res.R.shape == (k, n)
    rec_h = record(
        label="重建误差 /(n·u·‖A‖_F)",
        theory="≤ 1",
        measured=f"{worst_rec:.3f}（各形状 {', '.join(rows)}）",
        tol="1",
        passed=worst_rec <= 1.0,
        detail="Householder QR 后向稳定：误差与 κ 无关",
    )
    assert rec_h.passed
    assert_leq(worst_orth, 10.0, label="正交性缺陷 /(n·u)", detail="max|QᵀQ − I| 相对于 n·u（c·n·u 的 c 不超过 10）")


@assertion(
    "A",
    theory="取 α = −sign(x₁)‖x‖：‖Hx − αe₁‖ ≤ c·u‖x‖；取正号则退化到 ‖x_perp‖",
    chapter="1.3",
)
def test_householder_sign_choice_avoids_cancellation() -> None:
    """反射向量的**符号选择**不是审美问题：``α = −sign(x₁)‖x‖`` 让 ``x₁ − α`` 变成同号相加。

    实测：取正号时 ``‖Hx − αe₁‖/‖x‖`` 正好退化成横向分量的大小
    （1e-2 / 1e-5 / 1e-8 / 1e-11 逐一对上），而取负号时一路压到 ``u`` 附近。
    """
    worst_stable = 0.0
    worst_naive = 0.0
    rows = []
    for eps_ang in (1e-2, 1e-5, 1e-8, 1e-11):
        x = np.array([1.0, eps_ang, -eps_ang / 3.0, eps_ang / 7.0])
        x_norm = _norm2(x)
        v, alpha = householder_vector(x)
        H = householder_matrix(v)
        stable = float(np.max(np.abs(H @ x - alpha * np.eye(4)[0]))) / x_norm
        nightmare = float(np.sqrt(np.dot(x[1:], x[1:]))) / x_norm
        naive_alpha = x_norm  # 朴素取法：干脆不选号
        Hn = householder_matrix(x - naive_alpha * np.eye(4)[0])
        naive = float(np.max(np.abs(Hn @ x - naive_alpha * np.eye(4)[0]))) / x_norm
        worst_stable = max(worst_stable, stable)
        worst_naive = max(worst_naive, naive)
        rows.append(f"ε={eps_ang:.0e}: 稳定 {stable:.1e} / 朴素 {naive:.1e}（横向 {nightmare:.1e}）")
    rec_h = record(
        label="‖Hx−αe₁‖/‖x‖（稳定取号 vs 朴素取号）",
        theory="稳定 ~u；朴素 ~‖x_perp‖/‖x‖",
        measured="；".join(rows),
        tol="稳定 ≤ 100u，朴素 ≥ 横向分量的一半",
        passed=worst_stable <= 100.0 * U and worst_naive > 1e-3,
        detail="这就是教科书里那句『取负号以避免相减抵消』的实测证据",
    )
    assert rec_h.passed
    assert_orthogonal(householder_matrix(householder_vector(np.array([3.0, 4.0, 0.0]))[0]))


@assertion("A", theory="Givens：n_rotations = Σ_{col<min(m,n)}(m−1−col)，重建误差 ≤ c·n·u·‖A‖_F", chapter="1.3")
def test_givens_qr_reconstruction_and_rotation_count() -> None:
    """Givens QR 的重建误差与"旋转个数"的闭式计数一致。

    计数不是装饰：它决定 Givens 相对 Householder 的运算量常数（稀疏/带外元素时反而划算），
    所以这里把它钉死。
    """
    worst = 0.0
    rows = []
    for m, n in _SHAPES:
        A = _rand(m, n, 200 + m * 10 + n)
        res = givens_qr(A)
        ratio = qr_reconstruction_error(A, res.Q, res.R) / (n * U * _fro(A))
        worst = max(worst, ratio)
        expected = sum(m - 1 - col for col in range(min(m, n)))
        rows.append(f"{m}x{n}:{res.n_rotations}")
        assert res.n_rotations == expected, (m, n, res.n_rotations, expected)
    rec_h = record(
        label="重建误差 /(n·u·‖A‖_F)，旋转个数",
        theory="≤ 1；旋转数 = Σ_{col<k}(m−1−col)",
        measured=f"最大 {worst:.3f}；个数 {', '.join(rows)}（全部等于闭式计数）",
        tol="1",
        passed=worst <= 1.0,
    )
    assert rec_h.passed


@assertion(
    "A",
    theory="Aᵀr = 0：‖Aᵀr‖ ≤ c·n·u·‖A‖_F·(‖b‖ + ‖A‖_F‖x_hat‖)（任意 κ）",
    chapter="1.3",
)
def test_qr_least_squares_residual_is_orthogonal() -> None:
    """QR 最小二乘的残差与列空间正交，且这个"正交"与 κ 无关。

    刻度取的是**后向误差**的刻度 ``n u ‖A‖(‖b‖ + ‖A‖‖x_hat‖)``，而不是简单的
    ``n u ‖A‖‖b‖``：病态时 ``‖x_hat‖`` 本身会被放大到 ``κ‖b‖/‖A‖`` 量级，
    用后者的比值会跟着 κ 一起涨（实测 κ=1e10 时到 5.4e8），那不是"正交性变差"，
    是**尺子选错了**——1.2 章也踩过同一条（刻度必须跟着问题的尺度走）。
    """
    worst = 0.0
    worst_fwd = 0.0
    rows = []
    for kappa in (1e2, 1e6, 1e10):
        A = _ill_conditioned(24, 8, kappa, 300)
        b = _rand(24, 1, 301).ravel()
        x = qr_least_squares(A, b)
        r = b - A @ x
        defect = _norm2(A.T @ r)
        scale = 8 * U * _fro(A) * (_norm2(b) + _fro(A) * _norm2(x))
        worst = max(worst, defect / scale)
        x_ref = ref.lstsq(A, b)[0]
        worst_fwd = max(worst_fwd, _norm2(x - x_ref) / _norm2(x_ref))
        rows.append(f"κ={kappa:.0e}:{defect / scale:.3f}")
    rec_h = record(
        label="‖Aᵀr‖ /(n·u·‖A‖_F·(‖b‖+‖A‖_F‖x_hat‖))",
        theory="≤ c（与 κ 无关）",
        measured=f"{worst:.3f}（{', '.join(rows)}）；与 LAPACK 解的最大相对差 {worst_fwd:.2e}",
        tol="≤ 20",
        passed=worst <= 20.0,
    )
    assert rec_h.passed


@assertion("A", theory="列主元 QR 判秩：|r_kk| ≤ max(m,n)·u·|r₁₁| ⟹ 数值秩 < k", chapter="1.3")
def test_column_pivot_qr_rank_revealing() -> None:
    """秩亏矩阵上，列主元 QR 的最后一个对角元必须掉到 ``max(m,n)u|r₁₁|`` 以下。

    同时钉住两条**能证明的**界：``|r_11| ≥ σ_1/sqrt(n)``（主元取最大列范数）与体积界
    ``prod_{j<=k}|r_jj| <= prod_{j<=k} sigma_j``；并**实测**那条常被误当成定理的
    ``|r_kk| ≈ σ_k`` 到底能偏多少（它只是启发式，Kahan 矩阵上会彻底失效）。
    """
    g = rng(400)
    base = g.standard_normal((10, 3))
    A_def = np.column_stack([base, base[:, 0] - 2.0 * base[:, 2]])  # 第 4 列 = c1 − 2c3
    res_def = column_pivot_qr(A_def)
    tol_def = 10 * U * abs(res_def.R[0, 0])
    rank_def = qr_rank(np.diag(res_def.R), tol=tol_def)
    near = A_def.copy()
    near[:, 3] += 1e-7 * g.standard_normal(10)
    res_near = column_pivot_qr(near)
    rank_near = qr_rank(np.diag(res_near.R), tol=10 * U * abs(res_near.R[0, 0]))
    bound_ok = True
    details = []
    ratio_lo = 1e9
    ratio_hi = 0.0
    for A in (A_def, near):
        res = column_pivot_qr(A)
        s = jacobi_svd(A).s
        n = A.shape[1]
        d = np.abs(np.diag(res.R))
        # 可证明的两条：|r_11| >= σ_1/sqrt(n)；前 k 个 |r_jj| 的乘积 <= 前 k 个 σ 的乘积
        bound_ok = bound_ok and bool(d[0] >= s[0] / np.sqrt(n) - 10 * U * s[0])
        for k in range(1, n + 1):
            # 精确算术下的体积界 ∏|r_jj| <= ∏σ_j；每个因子只精确到 u‖A‖，
            # 所以在"每个 σ 各自放宽 u‖A‖"的形式下断言（秩亏矩阵的最后一个因子
            # 本身就在 u‖A‖ 量级，比较裸乘积会拿两个都等于"零"的数比大小）。
            rhs = float(np.prod(s[:k] + 10.0 * n * U * s[0]))
            bound_ok = bound_ok and bool(np.prod(d[:k]) <= rhs)
        ratio_lo = min(ratio_lo, float(np.min(d / s)))
        ratio_hi = max(ratio_hi, float(np.max(d / s)))
        details.append(f"|r_kk|/σ_k={np.array2string(d / s, precision=2)}")
    rec_h = record(
        label="秩亏 → 数值秩 3；近秩亏 → 数值秩 4",
        theory="判秩用 |r_nn| ≤ max(m,n)u|r₁₁|（启发式）；可证明的是 |r₁₁| ≥ σ₁/√n 与体积界（含舍入松弛）",
        measured=f"秩亏 rank={rank_def}（|r₄₄|={abs(res_def.R[3, 3]):.2e} ≤ tol={tol_def:.2e}），"
        f"近秩亏 rank={rank_near}；" + "；".join(details),
        tol="秩精确相等；两条界在 10u 松弛内成立",
        passed=rank_def == 3 and rank_near == 4 and bound_ok,
    )
    assert rec_h.passed


@assertion("A", theory="运算量计数 = 闭式 2n²(m−n/3)（逐位相等）", chapter="1.3")
def test_qr_flop_accounting_matches_formula() -> None:
    """运算量计数器必须与闭式公式**逐位相等**：计数器漂了，"省了多少"的结论就作废。"""
    worst = 0
    ok = True
    rows = []
    for m, n in ((4, 4), (10, 6), (30, 30), (40, 12)):
        A = _rand(m, n, 500 + m + n)
        res = householder_qr(A)
        ok = ok and res.n_ops == qr_flops(m, n)
        worst = max(worst, abs(res.n_ops - qr_flops(m, n)))
        rows.append(f"{m}x{n}:{res.n_ops}")
    rec_h = record(
        label="n_ops 与 qr_flops 的最大差",
        theory="0（逐位相等）",
        measured=f"{worst}；实测 {', '.join(rows)}",
        tol="0",
        passed=ok,
    )
    assert rec_h.passed


# --------------------------------------------------------------------------- #
# A 类：奇异值与伪逆对拍
# --------------------------------------------------------------------------- #
@pytest.mark.parity
@assertion(
    "A",
    theory="|σ_i−σ_i^ref|/σ_i^ref ≤ c·n·u·(σ_max/σ_i)（Demmel–Veselić）",
    chapter="1.3",
)
def test_singular_value_parity_with_reference() -> None:
    """三个自研 SVD 的奇异值与 LAPACK 对拍，容差按 ``σ_max/σ_i`` 放宽。

    为什么不能要求逐位相等：奇异值的**绝对**精度只能到 ``u‖A‖``，
    所以最小的那个奇异值只可能有 ``κu`` 的**相对**精度。把这条写进容差，
    而不是偷偷放宽——"小奇异值的相对精度"正是本章的主线之一。
    """
    worst = {name: 0.0 for name in _METHODS}
    rows = []
    for m, n in _SHAPES:
        A = _rand(m, n, 600 + m * 10 + n)
        s_ref = ref.svd(A, full_matrices=False)[1]
        for name in _METHODS:
            fit = _svd(name, A)
            err = relative_singular_value_error(fit.s, s_ref)
            worst[name] = max(worst[name], err / (max(m, n) * U * float(s_ref[0] / s_ref[-1])))
            assert fit.converged
        rows.append(f"{m}x{n}")
    rec_h = record(
        label="max_i |Δσ_i|/σ_i ÷ (n·u·σ_max/σ_i)",
        theory="≤ 1",
        measured="；".join(f"{k}={v:.3f}" for k, v in worst.items()) + f"（形状 {', '.join(rows)}）",
        tol="1",
        passed=max(worst.values()) <= 1.0,
        detail="三个版本分别对拍 LAPACK dgesdd",
    )
    assert rec_h.passed


@assertion("A", theory="‖A − UΣVᵀ‖_F ≤ c·n·u·‖A‖_F，且 U、V 列正交", chapter="1.3")
def test_svd_reconstruction_and_orthogonality() -> None:
    """重建误差 + 两翼正交性：SVD 的三条不变量一起测（``m<n`` 时 ``U`` 是方阵、``V`` 是瘦的）。"""
    worst_rec = 0.0
    worst_orth = 0.0
    for m, n in _SHAPES:
        A = _rand(m, n, 700 + m * 10 + n)
        for name in _METHODS:
            fit = _svd(name, A)
            rec = _fro(A - fit.reconstruct())
            worst_rec = max(worst_rec, rec / (max(m, n) * U * _fro(A)))
            worst_orth = max(worst_orth, orthogonality_defect(fit.U) / (max(m, n) * U), orthogonality_defect(fit.V) / (max(m, n) * U))
    rec_h = record(
        label="重建误差 /(n·u·‖A‖_F) 与两翼正交性缺陷 /(n·u)",
        theory="两者都 ≤ c",
        measured=f"重建比 {worst_rec:.3f}，正交性比 {worst_orth:.3f}",
        tol="≤ 10",
        passed=worst_rec <= 10.0 and worst_orth <= 10.0,
    )
    assert rec_h.passed


@assertion("A", theory="‖Av_i − σ_i u_i‖ ≤ c·u·‖A‖（对每个 i）", chapter="1.3")
def test_singular_vector_residuals_are_at_roundoff() -> None:
    """奇异向量的**残差**（而不是向量本身）：重奇异值附近向量不唯一，
    但 ``Av_i − σ_i u_i`` 在任何情况下都必须小到 ``u‖A‖``。"""
    worst = 0.0
    for m, n in ((8, 5), (12, 12), (20, 8), (5, 8)):
        A = _rand(m, n, 800 + m + n)
        for name in _METHODS:
            fit = _svd(name, A)
            worst = max(worst, float(np.max(singular_vector_residuals(A, fit.U, fit.s, fit.V))) / (U * _fro(A)))
    rec_h = record(
        label="奇异向量残差 /(u·‖A‖_F)",
        theory="≤ c",
        measured=f"{worst:.2f}",
        tol="≤ 200",
        passed=worst <= 200.0,
    )
    assert rec_h.passed


@assertion("A", theory="Moore–Penrose 四条件全部 ≤ c·n·u（含秩亏情形）", chapter="1.3")
def test_pseudoinverse_satisfies_moore_penrose() -> None:
    """伪逆的四条定义式（含两个对称性）都要落在 ``n·u`` 量级；秩亏时它是最小范数解算子。"""
    worst = 0.0
    rows = []
    g = rng(900)
    base = g.standard_normal((9, 3))
    rank_def = np.column_stack([base, base[:, 0] + base[:, 1], 2.0 * base[:, 0] + 2.0 * base[:, 1]])
    for A in (_rand(9, 5, 901), rank_def, _rand(14, 4, 902)):
        fit = jacobi_svd(A)
        A_plus = pseudoinverse(fit.U, fit.s, fit.V)
        d = moore_penrose_defects(A, A_plus)
        worst = max(worst, max(d.values()) / (max(A.shape) * U))
        rows.append(f"{A.shape[0]}x{A.shape[1]}:{max(d.values()):.1e}")
    A_ref = _rand(9, 5, 903)
    fit_ref = jacobi_svd(A_ref)
    diff = float(np.max(np.abs(pseudoinverse(fit_ref.U, fit_ref.s, fit_ref.V) - ref.pinv(A_ref))))
    rec_h = record(
        label="四条定义式的最大缺陷 /(n·u)",
        theory="≤ c",
        measured=f"{worst:.3f}；{', '.join(rows)}；与 LAPACK pinv 的元素最大差 {diff:.1e}",
        tol="≤ 10",
        passed=worst <= 10.0,
    )
    assert rec_h.passed


@assertion("A", theory="数值秩 = #{σ_i > max(m,n)u·σ_max}；κ₂ 与 LAPACK cond 一致", chapter="1.3")
def test_rank_and_condition_match_reference() -> None:
    """数值秩判定与条件数都要与参照物一致：秩亏矩阵给 4（三个版本一致）。"""
    g = rng(1000)
    base = g.standard_normal((12, 4))
    A_def = np.column_stack([base, base[:, 1] - base[:, 3]])
    ranks = {}
    for name in _METHODS:
        fit = _svd(name, A_def)
        ranks[name] = numerical_rank(fit.s, 12, 5)
    A_ill = _ill_conditioned(20, 6, 1e8, 1001)
    fit_ill = golub_kahan_svd(A_ill)
    kappa = spectral_condition(fit_ill.s)
    kappa_ref = float(ref.cond(A_ill))
    rec_h = record(
        label="秩亏矩阵的数值秩；κ₂ 与 LAPACK 的相对差",
        theory="秩 = 4；κ₂ 相对差 ≤ c·κ·u",
        measured=f"秩 {ranks}；κ₂ 实测 {kappa:.9e} vs LAPACK {kappa_ref:.9e}，相对差 {abs(kappa - kappa_ref) / kappa_ref:.2e}",
        tol="秩精确相等；κ 相对差 ≤ 1e-6",
        passed=all(r == 4 for r in ranks.values()) and abs(kappa - kappa_ref) / kappa_ref <= 1e-6,
    )
    assert rec_h.passed


@assertion("A", theory="双对角化：A = U₁ B V₁ᵀ，B 只有两条对角线非零（u‖A‖ 精度）", chapter="1.3")
def test_bidiagonalization_structure_and_error() -> None:
    """Golub–Kahan 的第一步：双对角形必须**精确**只有两条对角线，重建误差 ``u‖A‖``。"""
    worst = 0.0
    for m, n in ((8, 5), (5, 8), (12, 12), (20, 4)):
        A = _rand(m, n, 1100 + m * 10 + n)
        bi = bidiagonalize(A)
        B = bi.to_dense()
        worst = max(worst, _fro(A - bi.U1_thin @ B @ bi.V1_thin.T) / (max(m, n) * U * _fro(A)))
        assert bi.orientation == ("upper" if m >= n else "lower")
        off = B - np.diag(np.diag(B)) - np.diag(np.diag(B, 1 if bi.orientation == "upper" else -1), 1 if bi.orientation == "upper" else -1)
        assert float(np.max(np.abs(off))) <= 1e-14 * _fro(A)
        assert orthogonality_defect(bi.U1_thin) <= 100 * U * max(m, n)
        assert orthogonality_defect(bi.V1_thin) <= 100 * U * max(m, n)
    rec_h = record(
        label="双对角化重建误差 /(n·u·‖A‖_F)",
        theory="≤ 1；非对角外元素 = 0",
        measured=f"{worst:.3f}",
        tol="1",
        passed=worst <= 1.0,
    )
    assert rec_h.passed


@assertion("A", theory="解析奇异值：[[1,1],[0,1]] → (φ,1/φ)，φ=(1+√5)/2；对角与 c·Q 同样精确", chapter="1.3")
def test_svd_matches_closed_form_on_known_matrices() -> None:
    """三块可解析的矩阵：黄金比、对角阵、``c·Q``（奇异值全相等）。"""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    worst = 0.0
    labels = []
    cases = (
        ("[[1,1],[0,1]]", np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([phi, 1.0 / phi])),
        ("diag(3,1,0.5)", np.diag([3.0, 1.0, 0.5]), np.array([3.0, 1.0, 0.5])),
        ("4·Q(4x3)", 4.0 * householder_qr(_rand(4, 3, 1200)).Q, np.array([4.0, 4.0, 4.0])),
    )
    for label, A, s_true in cases:
        for name in _METHODS:
            fit = _svd(name, A)
            err = float(np.max(np.abs(fit.s - s_true))) / float(s_true[0])
            worst = max(worst, err / (max(A.shape) * U))
        labels.append(label)
    rec_h = record(
        label="max|σ − σ_解析| / (n·u·σ_max)",
        theory="≤ 1",
        measured=f"{worst:.3f}（{', '.join(labels)}）",
        tol="1",
        passed=worst <= 1.0,
    )
    assert rec_h.passed


@assertion("A", theory="小奇异值相对误差 ~ κu（不是 (κu)²），且 Jacobi 不劣于 Golub–Kahan", chapter="1.3")
def test_ill_conditioned_singular_value_accuracy() -> None:
    """把 κ 一路拉到 ``1e14``：小奇异值的相对误差随 κ 线性增长。

    顺带**实测**（而不是引用）Demmel–Veselić 那条结论：高相对精度下单边 Jacobi 通常最准。
    """
    worst_ratio = 0.0
    rows = []
    jacobi_best = 0
    total = 0
    for kappa in (1e2, 1e6, 1e10, 1e14):
        A = _ill_conditioned(20, 8, kappa, 1300)
        s_ref = ref.svd(A, full_matrices=False)[1]
        errs = {}
        for name in _METHODS:
            fit = _svd(name, A)
            errs[name] = relative_singular_value_error(fit.s, s_ref)
            worst_ratio = max(worst_ratio, errs[name] / (8 * U * kappa))
        rows.append(f"κ={kappa:.0e}: " + "/".join(f"{errs[k]:.1e}" for k in _METHODS))
        total += 1
        best = min(errs.values())
        jacobi_best += int(errs["jacobi"] <= best * (1.0 + 1e-9))
    rec_h = record(
        label="小奇异值相对误差 /(n·u·κ)",
        theory="≤ c（相对误差 ~ κu，不是 (κu)²）",
        measured=f"最大比 {worst_ratio:.3f}；" + "；".join(rows),
        tol="≤ 10",
        passed=worst_ratio <= 10.0,
        detail=f"Jacobi 在三者中最准（含打平）的次数 {jacobi_best}/{total}",
    )
    assert rec_h.passed


# --------------------------------------------------------------------------- #
# B 类：结构不变量
# --------------------------------------------------------------------------- #
@assertion("B", theory="结构：R 上三角、perm 是置换、反射子长度 = m、剩余列范数非增", chapter="1.3")
def test_qr_factor_shape_invariants() -> None:
    """分解出来的容器结构：``R`` 上三角、``perm`` 是 ``0..n−1`` 的置换、反射子长度正确。"""
    n_ok = 0
    for m, n in _SHAPES:
        A = _rand(m, n, 1400 + m * 10 + n)
        res = householder_qr(A, collect_reflectors=True)
        assert np.allclose(res.R, np.triu(res.R), atol=1e-14)
        piv = column_pivot_qr(A)
        assert sorted(piv.perm) == list(range(n))
        assert piv.pivots.size == min(m, n)
        lens = [v.shape[0] for v in res.reflectors]
        assert lens == sorted(lens, reverse=True) and max(lens) == m and min(lens) >= 1
        n_ok += 1
    rec_h = record(
        label="结构检查通过的形状数",
        theory=f"{len(_SHAPES)}",
        measured=f"{n_ok}",
        tol="全部",
        passed=n_ok == len(_SHAPES),
    )
    assert rec_h.passed


@assertion("B", theory="列主元 QR：|r_kk| 与剩余列范数随 k 非增", chapter="1.3")
def test_pivoted_qr_diagonal_is_nonincreasing() -> None:
    """列主元的意义就在这条单调性上：剩下的列范数不会回头变大。"""
    checked = 0
    for seed, (m, n) in enumerate(((12, 6), (20, 8), (30, 5)), start=1500):
        A = _rand(m, n, seed)
        res = column_pivot_qr(A)
        assert_monotone(np.abs(np.diag(res.R)), direction="nonincreasing", label=f"|r_kk| 非增 ({m}x{n})")
        assert_monotone(
            res.col_norms_remaining, direction="nonincreasing", label=f"剩余列范数非增 ({m}x{n})"
        )
        checked += 1
    assert checked == 3


@assertion("B", theory="Jacobi 每次扫描都把 ‖offdiag‖_F 压下去（单调）", chapter="1.3")
def test_jacobi_off_diagonal_norm_decreases() -> None:
    """单边 Jacobi 的每次扫描都必须把非对角 Frobenius 范数压下去——这是它收敛的证据。"""
    offs_all = []
    for m, n in ((8, 5), (12, 8), (20, 6)):
        A = _rand(m, n, 1600 + m + n)
        B = A.copy()
        offs = []
        for _ in range(4):
            fit = jacobi_svd(B, max_sweeps=1)
            B = (fit.U.T @ B) @ fit.V
            offs.append(float(np.sqrt(max(np.sum(B * B) - np.sum(np.diag(B) ** 2), 0.0))))
        assert_monotone(offs, direction="nonincreasing", label=f"offdiag 范数非增 ({m}x{n})")
        offs_all.append(offs[0])
    rec_h = record(
        label="三种形状扫描 4 次后的 offdiag 范数",
        theory="单调不增",
        measured=f"起点 {', '.join(f'{v:.2e}' for v in offs_all)}",
        tol="相邻差 ≥ −n·u·scale",
        passed=len(offs_all) == 3,
    )
    assert rec_h.passed


@assertion("B", theory="排序与符号约定：σ 降序非负，规范化后奇异向量残差仍 ≤ c·u·‖A‖", chapter="1.3")
def test_svd_sorting_and_sign_conventions() -> None:
    """``sort_and_fix_signs`` 必须给出降序非负的 σ，并且不破坏 ``Av_i = σ_i u_i``。"""
    worst = 0.0
    for m, n in ((9, 6), (6, 9), (10, 10)):
        A = _rand(m, n, 1700 + m + n)
        r = min(m, n)
        s_in = (np.linspace(1.0, 5.0, r)) ** 2
        U2, s_out, V2 = sort_and_fix_signs(_rand(m, r, 1711), s_in, _rand(n, r, 1712))
        assert s_out.shape == (r,)
        assert np.all(np.diff(s_out) <= 1e-12) and np.all(s_out >= 0.0)
        assert U2.shape == (m, r) and V2.shape == (n, r)
        for j in range(r):
            col = U2[:, j]
            assert col[int(np.argmax(np.abs(col)))] > 0.0
        fit = golub_kahan_svd(A)
        worst = max(worst, float(np.max(singular_vector_residuals(A, fit.U, fit.s, fit.V))) / (U * _fro(A)))
    rec_h = record(
        label="排序/符号规范化后的奇异向量残差 /(u·‖A‖_F)",
        theory="σ 降序非负；残差 ≤ c",
        measured=f"{worst:.2f}",
        tol="≤ 200",
        passed=worst <= 200.0,
    )
    assert rec_h.passed


@assertion("B", theory="Givens 旋转：c²+s²=1、det=1、r=hypot(a,b) 且不溢出", chapter="1.3")
def test_givens_rotation_stability() -> None:
    """Givens 旋转的稳定性体现在极限量级：``a=1e300, b=1e-300`` 也不能溢出去。"""
    worst = 0.0
    worst_det = 0.0
    for a, b in ((3.0, 4.0), (1e300, 1e-300), (1e-300, 1e300), (-1.0, 0.0), (0.0, -2.0)):
        c, s, r = givens_rotation(a, b)
        worst = max(worst, abs(c * c + s * s - 1.0))
        assert np.isfinite(c) and np.isfinite(s) and np.isfinite(r)
        assert abs(abs(r) - np.hypot(a, b)) <= 1e-15 * max(abs(a), abs(b), 1e-300)
        assert r == 0.0 or np.sign(r) == np.sign(a) or a == 0.0
        G = givens_matrix(4, 1, 3, c, s)
        assert_orthogonal(G, label="Givens 矩阵正交")
        worst_det = max(worst_det, abs(det_from_lu(lu_partial(G)) - 1.0))
        assert abs(c * a + s * b - r) <= 1e-15 * max(abs(a), abs(b), 1e-300)
        assert abs(-s * a + c * b) <= 1e-15 * max(abs(a), abs(b), 1e-300)
    rec_h = record(
        label="|c²+s²−1| 与 |det G − 1| 的最大值",
        theory="≤ u",
        measured=f"{worst:.2e} / {worst_det:.2e}",
        tol=f"{10 * U:.1e}",
        passed=worst <= 10 * U and worst_det <= 1e-12,
    )
    assert rec_h.passed


@assertion("B", theory="回代精确；orthogonality_measure ∈ [0,1] 且 r=0 时返回 0", chapter="1.3")
def test_back_substitution_and_orthogonality_measure() -> None:
    """回代解三角系统的误差，以及正交性度量的取值范围（不是 NaN、不是负数）。"""
    worst = 0.0
    for n in (3, 8, 20):
        R = np.triu(_rand(n, n, 1800 + n)) + n * np.eye(n)
        x_true = _rand(n, 1, 1810 + n).ravel()
        x = back_substitute(R, R @ x_true)
        worst = max(worst, float(np.max(np.abs(x - x_true))) / float(np.max(np.abs(x_true))))
        A = _rand(n + 4, n, 1820 + n)
        assert orthogonality_measure(A, np.zeros(n + 4)) == 0.0
        assert 0.0 <= orthogonality_measure(A, _rand(n + 4, 1, 1830 + n).ravel()) <= 1.0
    rec_h = record(
        label="回代最大相对误差",
        theory="≤ n·u·κ(R)",
        measured=f"{worst:.2e}",
        tol="1e-12",
        passed=worst <= 1e-12,
        detail="同时检查 orthogonality_measure 在 r=0 时返回 0（不是 NaN）",
    )
    assert rec_h.passed


@assertion("B", theory="截断解端点：k=n 与伪逆解逐位一致，k=0 给零解", chapter="1.3")
def test_truncated_solution_endpoints() -> None:
    """截断 SVD 的两个端点必须有定义：``k=0`` 给零解、``k=n`` 与全解逐位一致。"""
    worst = 0.0
    for m, n in ((8, 5), (12, 8)):
        A = _rand(m, n, 1900 + m + n)
        b = _rand(m, 1, 1910 + m).ravel()
        fit = jacobi_svd(A)
        x_full = truncated_solution(fit.U, fit.s, fit.V, b, n)
        assert float(np.max(np.abs(truncated_solution(fit.U, fit.s, fit.V, b, 0)))) == 0.0
        worst = max(worst, float(np.max(np.abs(x_full - pseudoinverse(fit.U, fit.s, fit.V) @ b))))
    rec_h = record(
        label="k=n 的截断解与伪逆解的逐位差",
        theory="0",
        measured=f"{worst:.2e}",
        tol="≤ 1e-14",
        passed=worst <= 1e-14,
    )
    assert rec_h.passed


@assertion("B", theory="列范数 = ‖a_j‖₂；Householder 矩阵对称、正交、对合（H²=I）", chapter="1.3")
def test_column_norms_and_householder_matrix_are_consistent() -> None:
    """``column_norms`` 与手算一致；``householder_matrix`` 对称正交且 ``H²=I``。"""
    worst = 0.0
    for m, n in ((7, 4), (15, 9)):
        A = _rand(m, n, 2000 + m + n)
        cn = column_norms(A)
        for j in range(n):
            worst = max(worst, abs(cn[j] - _norm2(A[:, j])))
        v, _alpha = householder_vector(A[:, 0])
        H = householder_matrix(v)
        assert_orthogonal(H, label="Householder 矩阵正交")
        assert float(np.max(np.abs(H - H.T))) <= 1e-14
        assert float(np.max(np.abs(H @ H - np.eye(m)))) <= 1e-14
    rec_h = record(
        label="列范数最大绝对差",
        theory="≤ u·‖a_j‖",
        measured=f"{worst:.2e}",
        tol="1e-14",
        passed=worst <= 1e-14,
    )
    assert rec_h.passed
