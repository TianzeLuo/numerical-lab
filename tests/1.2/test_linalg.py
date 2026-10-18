"""1.2 章 A/B 类断言：LU、Cholesky、主元策略与条件数。

这一章的核心是**把三件常被混为一谈的事分开**：

1. **后向稳定性**（消元本身稳不稳）——由增长因子 :math:`\\rho` 刻画，部分主元保证
   :math:`\\|\\delta A\\| \\le c\\,n\\,\\rho\\,u\\,\\|A\\|`；
2. **条件数**（问题本身难不难）——由 :math:`\\kappa(A)` 刻画，前向误差
   :math:`\\|\\hat x - x\\|/\\|x\\| \\lesssim \\kappa(A)\\,\\eta`；
3. **结构**（稀疏性）——由消元顺序决定，与数值毫无关系。

因此这里的断言也都围绕"实测值 vs 理论界"：rho 的上界是不是可达、后向误差是不是真的
与 :math:`\\kappa` 无关、前向误差是不是真的按 :math:`\\kappa u` 走、填充数是不是真的
等于符号预测。每条断言都给出**实测数字**（``record``/``assert_*`` helper 会登记），
抄理论值不算证据。
"""

from __future__ import annotations

import numpy as np
import pytest

from numcore.linalg.chol import chol_flops, chol_solve, cholesky, chol_reconstruct_error, is_positive_definite
from numcore.linalg.lu import (
    backward_error,
    ball_matrix,
    condition_estimate_1,
    dominant_matrix,
    det_from_lu,
    elimination_flops,
    exact_solve,
    example_matrix,
    growth_factor,
    leading_principal_minors,
    lu_decompose,
    lu_no_pivot,
    lu_partial,
    lu_solve,
    permute_rows,
    residual_norm,
    singular_values_2x2,
    unit_ball_radius,
    wilkinson_matrix,
)
from numcore.reference import dense as ref
from numcore.utils import rng
from numcore.verify import assertion, assert_leq, assert_order, find_threshold, record

U = float(np.finfo(float).eps) / 2.0  # 单位舍入 u = 2^-53 ≈ 1.11e-16


def _random(n: int, count: int, seed: int, *, scale: float = 1.0) -> list[np.ndarray]:
    g = rng(seed)
    return [scale * g.standard_normal((n, n)) for _ in range(count)]


# --------------------------------------------------------------------------- #
# A 类：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="‖PA−LU‖∞ ≤ c·n·u·‖A‖∞", chapter="1.2")
def test_lu_reconstruction_is_exact_to_roundoff() -> None:
    """LU 的重建残差必须落在 :math:`n\\,u\\,\\|A\\|_\\infty` 量级（Wilkinson 后向稳定定理）。

    这是"消元本身不制造误差"的量化版本：不管 κ 多大，重建误差都只跟 n 与 u 有关。
    """
    worst = 0.0
    worst_ratio = 0.0
    for n in (2, 5, 12, 30, 64):
        for A in _random(n, 8, 1000 + n):
            res = lu_partial(A)
            scale = float(np.max(np.abs(A)))
            err = float(np.max(np.abs(res.P() @ A - res.L @ res.U)))
            worst = max(worst, err)
            worst_ratio = max(worst_ratio, err / (n * U * scale))
    rec = record(
        label="重建残差 / (n·u·‖A‖∞)",
        theory="≤ 1（Wilkinson 界）",
        measured=f"{worst_ratio:.3f}（最大绝对残差 {worst:.3e}）",
        tol="1",
        passed=worst_ratio <= 1.0,
    )
    assert rec.passed, f"重建残差超出 n·u 界：比值 {worst_ratio:.3f}"


@assertion("A", theory="部分主元 ρ ≤ 2^(n−1)，Wilkinson 矩阵取等号", chapter="1.2")
def test_partial_pivot_growth_bound_is_sharp() -> None:
    """增长因子的上界**可达**：Wilkinson 矩阵上 ρ = 2^(n−1)（实测等于理论值，不是小于）。"""
    measured = []
    for n in (2, 3, 5, 8, 10, 12):
        res = lu_partial(wilkinson_matrix(n))
        measured.append((n, res.rho))
        assert res.rho == pytest.approx(2.0 ** (n - 1), rel=1e-12), (n, res.rho)
    # 完全主元把同一个矩阵压到 2（这就是"多花比较换稳定性"的收益）
    rho_complete = lu_decompose(wilkinson_matrix(12), pivot="complete").rho
    record(
        label="ρ(部分主元) / 2^(n−1)",
        theory=2.0 ** 11,
        measured=f"{measured[-1][1]:.6g}（完全主元 {rho_complete:.6g}）",
        tol="rel 1e-12",
        passed=True,
    )
    assert rho_complete <= 4.0, rho_complete


@assertion("A", theory="ρ 与 κ 是两件事：tiny_pivot 的 κ₁ = 4 但无主元 ρ = 1/u", chapter="1.2")
def test_growth_factor_and_condition_number_are_independent() -> None:
    """小主元反例：矩阵**条件数很小**（κ₁ = 4），无主元消元却把解算成完全错误。

    说明"消元不稳定"与"问题病态"是两件独立的事——前者靠选主元解决，后者无解。
    """
    A = example_matrix("tiny_pivot")
    b = np.linspace(1.0, 2.0, 2)
    x_exact = exact_solve(A, b)
    x_none = lu_solve(lu_no_pivot(A), b)
    x_part = lu_solve(lu_partial(A), b)
    kappa = condition_estimate_1(A)
    err_none = float(np.max(np.abs(x_none - x_exact))) / float(np.max(np.abs(x_exact)))
    err_part = float(np.max(np.abs(x_part - x_exact))) / float(np.max(np.abs(x_exact)))
    rho_none = lu_no_pivot(A).rho
    rec = record(
        label="κ₁ / ρ(无主元) / 前向误差(无主元→部分)",
        theory="κ₁=4（小）、ρ=1/u、前向误差 1 → 0",
        measured=f"κ₁={kappa:.4g}, ρ={rho_none:.3g}, 误差 {err_none:.3g} → {err_part:.3g}",
        tol="err_none ≥ 0.1 且 err_part ≤ 1e-15",
        passed=err_none >= 0.1 and err_part <= 1e-15,
    )
    assert rec.passed, (kappa, rho_none, err_none, err_part)
    assert 1.0 < kappa < 10.0, f"这个反例的 κ₁ 应当很小，实测 {kappa}"


@assertion("A", theory="前向误差 ≤ κ·u（常数因子内可达），后向误差 ≈ u 与 κ 无关", chapter="1.2")
def test_forward_error_follows_kappa_u_while_backward_stays_at_u() -> None:
    """沿 κ 从 1e2 扫到 1e14：前向误差随 κ 单调上升且始终被 κu 界住，后向误差始终 ~1e-16。

    这是本章最重要的一条定量结论。两边都用**实测**：前向误差对比有理数精确解，
    后向误差是范数型后向误差 :math:`\\|Ax-b\\|_\\infty/(\\|A\\|_\\infty\\|x\\|_\\infty+\\|b\\|_\\infty)`。

    注意 κ 很小时前向误差会**恰好是 0**（两个解舍入到同一组 double），所以下界只在
    κ ≥ 1e6、误差已经离开舍入底之后才检查——把"误差为 0"当成"误差很小"是自欺。
    """
    exps = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
    ratios = []
    fwds = []
    kappas = []
    worst_backward = 0.0
    for exp in exps:
        A = ball_matrix(0.8, 10.0**exp)
        b = np.linspace(1.0, 2.0, 2)
        x_exact = exact_solve(A, b)
        x = lu_solve(lu_partial(A), b)
        denom = float(np.max(np.abs(x_exact)))
        fwd = float(np.max(np.abs(x - x_exact))) / denom
        smin, smax = singular_values_2x2(A)
        kappa = smax / smin if smin > 0.0 else float("inf")
        kappas.append(kappa)
        fwds.append(fwd)
        ratios.append(fwd / (kappa * U))
        worst_backward = max(worst_backward, backward_error(A, x, b))
    # 大 κ 段：误差必须离开舍入底，并且落在 κu 的一个常数因子之内
    big = [r for k, r in zip(kappas, ratios) if k >= 1e6]
    upper_ok = max(ratios) <= 1.0
    lower_ok = min(big) >= 1e-3
    # 单调性：κ 增大时前向误差不下降（允许相等：都贴在舍入底上）
    mono_ok = all(b >= a for a, b in zip(fwds, fwds[1:]))
    rec = record(
        label="前向误差/(κ·u) 与后向误差",
        theory="fwd/(κu) ∈ (0, 1]，后向误差 ≤ 10u 且与 κ 无关",
        measured=(
            f"fwd/(κu) ∈ [{min(ratios):.4g}, {max(ratios):.4g}]（大 κ 段最小 {min(big):.4g}）；"
            f"后向误差最大 {worst_backward:.2e}；前向误差单调={mono_ok}"
        ),
        tol="0 < ratio ≤ 1 且大 κ 段 ≥ 1e-3，后向 ≤ 10u",
        passed=upper_ok and lower_ok and mono_ok and worst_backward <= 10 * U,
    )
    assert rec.passed, (ratios, worst_backward, mono_ok)


@assertion("A", theory="Cholesky 运算量 = LU 的一半（n³/3 vs 2n³/3）", chapter="1.2")
def test_cholesky_uses_half_the_operations_of_lu() -> None:
    """实测两种分解的标量运算次数之比 → 1/2（对称性省下的一半乘法）。"""
    ratios = []
    for n in (32, 64, 128, 256):
        A = example_matrix("spd", n)
        cl = cholesky(A)
        lu = lu_partial(A)
        assert cl.ok and not lu.singular
        ratios.append(chol_flops(n) / elimination_flops(n))
    rec = record(
        label="chol_flops / lu_flops（n=256）",
        theory="0.5",
        measured=f"{ratios[-1]:.6f}（n=32..256: {', '.join(f'{r:.4f}' for r in ratios)}）",
        tol="0.5 ± 0.01",
        passed=abs(ratios[-1] - 0.5) < 0.01,
    )
    assert rec.passed


@assertion("A", theory="LU 运算量 ~ (2/3)n³：双对数斜率 3", chapter="1.2")
def test_elimination_flop_count_grows_cubically() -> None:
    """运算量随 n 的三次增长（斜率 3 ± 0.1），常数项 → 2/3。

    这里数的不是公式，而是 :func:`lu_decompose` 在运行时按数组形状**实际累加**的运算次数，
    再与闭式 :math:`n(n-1)(4n+1)/6` 对拍（两者必须完全一致）。
    """
    ns = [16, 32, 64, 128, 256]
    ops = []
    for n in ns:
        A = example_matrix("spd", n)
        res = lu_partial(A)
        assert res.ops == elimination_flops(n), (n, res.ops, elimination_flops(n))
        ops.append(float(res.ops))
    fit = assert_order([float(n) for n in ns], ops, theory=3.0, tol=0.1, label="LU 运算量增长阶")
    const = ops[-1] / (ns[-1] ** 3)
    rec = record(
        label="ops / n³",
        theory="2/3",
        measured=f"{const:.6f}（斜率拟合 {fit.slope:.4f}）",
        tol="2/3 ± 0.01",
        passed=abs(const - 2.0 / 3.0) < 0.01,
    )
    assert rec.passed, (const, fit.slope)


@assertion("A", theory="严格对角占优 ⟹ 无主元消元 ρ ≤ 2（Wilkinson）", chapter="1.2")
def test_diagonal_dominance_guarantees_bounded_growth() -> None:
    """严格对角占优（比值 > 1）时无主元消元的增长因子**有** 2 这个上界；失去占优则没有保证。

    注意这里测的是"保证"而不是"崩溃"：占优比 < 1 **不必然**出事（它不是必要条件），
    实测里 ρ 只是偶尔冲到几十——所以断言写成"占优时 ≤ 2（多组 n/种子实测），
    失去占优后出现 > 2 的实例"，并把两组数字都记下来。
    """
    worst_dominant = 0.0
    for n in (8, 12, 16, 24):
        for dom in (1.02, 1.1, 1.5, 2.0, 3.0):
            for seed in (1, 17, 2024):
                rho = lu_no_pivot(dominant_matrix(n, dom, seed=seed)).rho
                worst_dominant = max(worst_dominant, rho)
    worst_free = max(lu_no_pivot(dominant_matrix(12, 0.05, seed=s)).rho for s in (1, 2, 3, 17, 2024))
    rec = record(
        label="max ρ（严格对角占优，60 例）/ max ρ（占优比 0.05，5 例）",
        theory="占优时 ≤ 2；失去占优无上界（实测值只说明这一组种子）",
        measured=f"{worst_dominant:.4f} / {worst_free:.4g}",
        tol="占优 ≤ 2",
        passed=worst_dominant <= 2.0 + 1e-12 and worst_free > worst_dominant,
    )
    assert rec.passed, (worst_dominant, worst_free)


@assertion("A", theory="κ₁(A) = ‖A‖₁‖A⁻¹‖₁（自研 LU 求逆列）", chapter="1.2")
def test_condition_estimate_matches_reference() -> None:
    """自研 κ₁ 估计与参考层（SciPy）逐例对拍（相对差 ≤ 1e-8）。"""
    worst = 0.0
    for name, n in (("hilbert", 6), ("spd", 12), ("wilkinson", 8), ("diag_dominant", 10), ("banded", 16)):
        A = example_matrix(name, n)
        mine = condition_estimate_1(A)
        theirs = ref.cond(A, p=1)
        rel = abs(mine - theirs) / theirs
        worst = max(worst, rel)
    rec = record(
        label="max |κ₁(自研) − κ₁(SciPy)| / κ₁",
        theory="≤ 1e-8",
        measured=f"{worst:.3e}",
        tol="1e-8",
        passed=worst <= 1e-8,
    )
    assert rec.passed, worst


@assertion("A", theory="采样 σmax 的误差 ≤ C·Δθ²（C 为实测常数），σmin 只会被高估", chapter="1.2")
def test_sampled_singular_values_converge_quadratically() -> None:
    """在单位圆上采样量 :math:`\\sigma_{\\max}`：误差被 :math:`C\\,\\Delta\\theta^2` 界住。

    为什么不用"拟合阶数 = 2"：均匀网格是**嵌套**的（n 翻倍时旧点全保留），于是误差呈阶梯状——
    只有当新网格恰好插到极大点附近时误差才下降，局部阶会在 0 与 5 之间乱跳，斜率拟合不稳定。
    真正稳定成立的是那个**上界**（实测 C ≈ 0.36），以及"误差随 n 单调不增"。
    这条断言同时暴露采样法的**系统性高估 σmin**：这就是 pitfalls 第 3 条的来源。
    """
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    smin, smax = singular_values_2x2(A)
    counts = [91, 181, 361, 721, 1441, 2881, 5761]
    errs = []
    for m in counts:
        _, hi = unit_ball_radius(A, n_theta=m)
        errs.append(abs(hi - smax))
    consts = [e / ((2 * np.pi / m) ** 2) for e, m in zip(errs, counts)]
    mono_ok = all(b <= a for a, b in zip(errs, errs[1:]))
    _, lo = unit_ball_radius(A, n_theta=counts[-1])
    rec = record(
        label="采样 σmax 误差/Δθ² 的最大值；σmin 的高估量",
        theory="≤ 1（O(Δθ²) 界）；σmin 采样值 ≥ 闭式值",
        measured=f"C_max={max(consts):.4f}（n=91..5761）；σmin 实测 {lo:.6f} vs 闭式 {smin:.6f}（+{lo - smin:.2e}）",
        tol="C ≤ 1；单调不增；σmin ≥ 闭式",
        passed=max(consts) <= 1.0 and mono_ok and lo >= smin,
    )
    assert rec.passed, (consts, lo, smin)


# --------------------------------------------------------------------------- #
# B 类：不变量
# --------------------------------------------------------------------------- #
@assertion("B", theory="L 单位下三角、U 上三角、perm 是置换", chapter="1.2")
def test_lu_factor_shape_invariants() -> None:
    """对随机矩阵与全部示例矩阵：因子形状、单位对角、置换合法性。"""
    checked = 0
    matrices = _random(9, 8, 7) + [example_matrix(name, 9) for name in ("wilkinson", "hilbert", "spd", "diag_dominant", "banded")]
    for A in matrices:
        for res in (lu_partial(A), lu_no_pivot(A), lu_decompose(A, pivot="complete")):
            n = res.n
            assert np.allclose(np.diag(res.L), 1.0), "L 的对角必须恰为 1"
            assert np.allclose(res.L, np.tril(res.L)), "L 必须下三角"
            assert np.allclose(res.U, np.triu(res.U)), "U 必须上三角"
            assert sorted(res.perm.tolist()) == list(range(n)), "perm 必须是置换"
            assert np.allclose(permute_rows(A, res.perm), res.P() @ A)
            checked += 1
    rec = record(
        label="检查过的分解数",
        theory="全部满足",
        measured=f"{checked} 次",
        tol="0 反例",
        passed=checked >= 30,
    )
    assert rec.passed


@assertion("B", theory="det(A) = (−1)^s·∏u_ii", chapter="1.2")
def test_determinant_from_lu_matches_reference() -> None:
    """行列式恒等式：行交换只改符号，乘积的对角元给出 |det|。"""
    worst_ratio = 0.0
    worst_rel = 0.0
    detail = []
    for name, n in (("hilbert", 6), ("wilkinson", 9), ("spd", 10), ("banded", 12), ("hilbert", 10)):
        A = example_matrix(name, n)
        mine = det_from_lu(lu_partial(A))
        theirs = ref.det(A)
        rel = abs(mine - theirs) / abs(theirs)
        bound = n * U * condition_estimate_1(A)  # ∏u_ii 的相对误差被 κ 放大
        worst_rel = max(worst_rel, rel)
        worst_ratio = max(worst_ratio, rel / bound)
        detail.append(f"{name}{n}: {rel:.1e}/{bound:.1e}")
    rec = record(
        label="max (|Δdet|/|det|) / (n·u·κ₁)",
        theory="≤ 1（det 是 n 个因子的乘积，相对误差被 κ₁ 放大）",
        measured=f"比值 {worst_ratio:.4f}（最大相对误差 {worst_rel:.2e}）；" + ", ".join(detail),
        tol="1",
        passed=worst_ratio <= 1.0,
    )
    assert rec.passed, (worst_ratio, worst_rel)


@assertion("B", theory="Cholesky 成功 ⟺ 矩阵正定", chapter="1.2")
def test_cholesky_success_iff_positive_definite() -> None:
    """200 个随机对称矩阵：Cholesky 能否分解与"特征值是否全正"必须逐例一致。"""
    g = rng(2024)
    mismatches = []
    n_ok = 0
    for _ in range(200):
        M = g.standard_normal((6, 6))
        S = 0.5 * (M + M.T)
        ok = cholesky(S).ok
        n_ok += int(ok)
        ev = ref.eigh(S)[0]
        if ok != bool(np.all(ev > 0)):
            mismatches.append((ok, float(np.min(ev))))
    rec = record(
        label="判定不一致个数（200 例，其中正定 {})".format(n_ok),
        theory="0",
        measured=f"{len(mismatches)}",
        tol="0",
        passed=not mismatches,
    )
    assert rec.passed, mismatches[:3]


@assertion("B", theory="前导块分解残差 ≤ c·n·u·ρ（消元逐步精确）", chapter="1.2")
def test_leading_block_residual_stays_at_roundoff() -> None:
    """章节热力图那条残差曲线的不变量：精确算术下恒 0，浮点下 ≤ n·u·ρ。"""
    worst = 0.0
    worst_ratio = 0.0
    for n, dom in ((8, 1.6), (12, 0.5), (16, 0.9), (16, 2.5)):
        A = dominant_matrix(n, dom)
        res = lu_decompose(A, pivot="partial", history=True)
        scale = float(np.max(np.abs(A)))
        A_perm = res.P() @ A
        bound = n * U * res.rho
        for j, snap in enumerate(res.history):
            m = max(1, min(j + 1, n))
            L = np.tril(np.asarray(snap)[:m, :m], -1) + np.eye(m)
            Uarr = np.triu(np.asarray(snap)[:m, :m])
            err = float(np.max(np.abs(A_perm[:m, :m] - L @ Uarr))) / scale
            worst = max(worst, err)
            worst_ratio = max(worst_ratio, err / bound)
    rec = record(
        label="max (前导块残差/‖A‖∞) / (n·u·ρ)",
        theory="≤ 1（与 LU 后向误差同一条界）",
        measured=f"比值 {worst_ratio:.4f}（最大绝对残差 {worst:.3e}）",
        tol="1",
        passed=worst_ratio <= 1.0,
    )
    assert rec.passed, (worst_ratio, worst)


@assertion("B", theory="顺序主子式非零 ⟺ 无主元 LU 可行", chapter="1.2")
def test_leading_principal_minors_detect_zero_pivots() -> None:
    """swap 矩阵的顺序主子式有 0 → 无主元消元必然除零；对角占优矩阵则全非零。"""
    minors_swap = leading_principal_minors(example_matrix("swap"))
    minors_dd = leading_principal_minors(example_matrix("diag_dominant", 8))
    minors_hilbert = leading_principal_minors(example_matrix("hilbert", 8))
    ok_swap = bool(np.any(minors_swap == 0.0))
    ok_dd = bool(np.all(minors_dd != 0.0)) and bool(np.all(minors_hilbert != 0.0))
    # 与"实际能不能跑"对拍
    assert lu_no_pivot(example_matrix("swap")).singular is ok_swap
    rec = record(
        label="零主子式检测 / 非零主子式",
        theory="swap: 有 0；对角占优与 Hilbert: 全非零",
        measured=f"swap {minors_swap.tolist()}；对角占优全非零={bool(np.all(minors_dd != 0.0))}",
        tol="逐例一致",
        passed=ok_swap and ok_dd,
    )
    assert rec.passed


@assertion("B", theory="后向误差无量纲：A、b 同乘常数后仍是 O(u)", chapter="1.2")
def test_backward_error_is_scale_invariant() -> None:
    """把 A 与 b 同乘一个常数，后向误差**保持在舍入底**（它必须是无量纲量）。

    注意精度：后向误差本身就在 1e-17 量级，缩放会改变舍入的具体走向，
    所以"绝对变化 ≤ 1e-18"这种写法是错的（实测就有 1.3e-18 的抖动）。
    正确的说法是它仍然是 O(u)——即不随尺度漂移数量级。
    """
    A = example_matrix("hilbert", 6)
    b = np.linspace(1.0, 2.0, 6)
    x = lu_solve(lu_partial(A), b)
    base = backward_error(A, x, b)
    scaled = [
        backward_error(s * A, x, s * b)
        for s in (1e-6, 1e-2, 7.5, 1e3, 1e6)
    ]
    worst = max(abs(v - base) for v in scaled)
    biggest = max(scaled)
    rec = record(
        label="尺度变换后的后向误差（基准 {base:.2e}）".format(base=base),
        theory="全部 ≤ 100u = {:.2e}（尺度不变，只留在舍入底）".format(100 * U),
        measured=f"最大 {biggest:.3e}，与基准的最大差 {worst:.3e}",
        tol="100u",
        passed=biggest <= 100 * U and worst <= 100 * U,
    )
    assert rec.passed, (biggest, worst)


@assertion("B", theory="chol_solve 的解满足 A x = b（残差 ≤ n·u·‖A‖‖x‖）", chapter="1.2")
def test_cholesky_solve_residual_bound() -> None:
    """Cholesky 求解的残差界（不变量：解必须真的满足方程）。"""
    worst = 0.0
    for n in (4, 9, 25, 49):
        A = example_matrix("spd", n)
        b = np.linspace(1.0, 2.0, n)
        res = cholesky(A)
        x = chol_solve(res, b)
        worst = max(worst, residual_norm(A, x, b) / (n * U * float(np.max(np.abs(A))) * float(np.max(np.abs(x)))))
    rec = record(
        label="残差 / (n·u·‖A‖·‖x‖)",
        theory="≤ 1",
        measured=f"{worst:.4f}",
        tol="1",
        passed=worst <= 1.0,
    )
    assert rec.passed, worst


# --------------------------------------------------------------------------- #
# 反例矩阵动物园的"每个矩阵说一句话"（避免有人把示例矩阵改成平凡的）
# --------------------------------------------------------------------------- #
@assertion("B", theory="示例矩阵必须各自保持其反例性质", chapter="1.2")
def test_example_matrices_keep_their_character() -> None:
    """动物园自检：swap 无主元必奇异、tiny_pivot 的 ρ=1e18、wilkinson 达到 2^(n−1)、SPD 真的正定。"""
    facts = {
        "swap_singular": lu_no_pivot(example_matrix("swap")).singular,
        "tiny_rho": lu_no_pivot(example_matrix("tiny_pivot")).rho,
        "wilkinson_sharp": lu_partial(wilkinson_matrix(10)).rho == 2.0**9,
        "spd_ok": is_positive_definite(example_matrix("spd", 12)),
        "dd_ok": is_positive_definite(example_matrix("diag_dominant", 12)),
        "hilbert_kappa": condition_estimate_1(example_matrix("hilbert", 8)) > 1e9,
        "banded_nnz": int(np.count_nonzero(example_matrix("banded", 12))) == 12 + 2 * 11,
    }
    rec = record(
        label="动物园自检",
        theory="7 条性质全部成立",
        measured=str(facts),
        tol="全 True",
        passed=all(bool(v) for v in facts.values()),
    )
    assert rec.passed, facts
    assert facts["tiny_rho"] > 1e17


@assertion("B", theory="ρ 的上界随 n 指数增长，但实测 ρ 通常远小于它", chapter="1.2")
def test_typical_growth_is_far_below_the_bound() -> None:
    """随机矩阵上 ρ 的典型值（实测）与最坏界 2^(n−1) 的对比——别拿最坏界吓自己。"""
    worst = 0.0
    for n in (10, 20, 40):
        for A in _random(n, 5, 555 + n):
            worst = max(worst, lu_partial(A).rho)
    bound = 2.0**39
    assert_leq(worst, bound, label="随机矩阵 ρ < 2^(n−1)")
    rec = record(
        label="随机矩阵最大 ρ（n ≤ 40）",
        theory=f"≤ 2^39 = {bound:.3e}",
        measured=f"{worst:.3f}（对数尺度上离界差 {np.log2(bound / worst):.1f} 个二进制数量级）",
        tol="≤ 2^39",
        passed=worst <= bound,
    )
    assert rec.passed
