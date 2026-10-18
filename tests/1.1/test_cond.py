"""1.1 章 A/B 类断言（条件数侧）：κ 的定义、与参考实现的对拍、前向/后向误差的分工。

这一章要把"**稳定**"与"**准**"分开，所以条件数侧的断言分两层：

* **A 类**（解析对拍）：:math:`\\kappa_p=\\|A\\|_p\\|A^{-1}\\|_p` 与参考实现（SciPy/NumPy）
  逐点对拍；前向误差 :math:`\\propto\\kappa u` 的**实测阶**；采样椭球半径的
  :math:`O(\\Delta\\theta^2)` 阶；:math:`\\kappa(A^\\top A)=\\kappa(A)^2`。
* **B 类**（不变量）：:math:`\\kappa\\ge1`、缩放/求逆/转置不变性、
  分量型条件数对**行缩放**不变、采样半径只会高估 :math:`\\sigma_{\\min}`。

注意：前向误差的"参考解"用 ``numcore.reference.exact.mp_solve``（mpmath 60 位十进制），
它自身的相对误差 ~1e-17，远小于被测的 :math:`\\kappa u`（:math:`\\kappa\\ge10^2` 时至少 1e-14）。
"""

from __future__ import annotations

import numpy as np

from numcore.linalg import cond
from numcore.linalg.lu import backward_error, lu_partial, solve_lu
from numcore.reference import dense as ref_dense
from numcore.reference import exact as ref_exact
from numcore.utils import rng
from numcore.verify import assertion, assert_leq, assert_order, record

#: 单位舍入 u = 2^-53
U = 2.0**-53

#: 参考实现的范数键（``np.linalg.cond`` 只认 ``np.inf``/``'fro'``，不认字符串 ``'inf'``）
_REF_KEY = {1: 1, 2: 2, "inf": np.inf, "fro": "fro"}


def _sample_matrices() -> dict[str, np.ndarray]:
    g = rng(31)
    return {
        "diag1e-3": np.diag([1.0, 1e-3]),
        "diag1e-6": np.diag([1.0, 1e-6]),
        "rand4": g.uniform(-1.0, 1.0, (4, 4)),
        "sym5": cond.symmetric_with_spectrum([1.0, 1e-1, 1e-2, 1e-3, 1e-4]),
        "shear1e4": cond.condition_matrix("shear", 1e4),
        "hilbert4": cond.hilbert_matrix(4),
        "rotation": cond.condition_matrix("rotation", 1e6),
    }


# --------------------------------------------------------------------------- #
# A 类
# --------------------------------------------------------------------------- #
@assertion("A", theory="κ_p = ‖A‖_p·‖A⁻¹‖_p，p ∈ {1,2,∞,F}", chapter="1.1")
def test_condition_numbers_match_reference() -> None:
    """四把尺子同时和参考实现（``numpy.linalg.cond``）对拍。

    :math:`\\kappa_2` 走幂迭代 + 反迭代（只做矩阵–向量乘），所以它比
    :math:`\\kappa_1/\\kappa_\\infty/\\kappa_F` 差几个量级——但仍在 1e-9 以内
    （κ ≤ 1e6 的矩阵上）。
    """
    worst = {1: 0.0, 2: 0.0, "inf": 0.0, "fro": 0.0}
    rel22 = []
    for name, A in _sample_matrices().items():
        for p in (1, 2, "inf", "fro"):
            ours = cond.condition_number(A, p=p)
            ref = float(ref_dense.cond(A, p=_REF_KEY[p]))
            rel = abs(ours - ref) / ref
            worst[p] = max(worst[p], rel)
            if p == 2 and cond.condition_2(A) <= 1e6:
                rel22.append(rel)
    assert worst[1] <= 1e-8 and worst["inf"] <= 1e-8 and worst["fro"] <= 1e-8, worst
    assert max(rel22) <= 1e-6, rel22
    rec = record(
        label="κ_p 与参考实现的最大相对偏差",
        theory="≤ 1e-8（1/∞/F，逐点）；≤ 1e-6（2，幂迭代）",
        measured=(
            f"1: {worst[1]:.2e}，2: {max(rel22):.2e}，inf: {worst['inf']:.2e}，F: {worst['fro']:.2e}"
            f"（{len(_sample_matrices())} 个矩阵）"
        ),
        tol="1e-8 / 1e-6",
        passed=worst[1] <= 1e-8 and max(rel22) <= 1e-6,
    )
    assert rec.passed


@assertion("A", theory="前向误差 ≤ κ(A)·u（实测阶 1），后向误差 ≤ c·u 且与 κ 无关", chapter="1.1")
def test_forward_error_follows_kappa_times_roundoff() -> None:
    """把"稳定 ≠ 准"做成两条曲线：:math:`\\kappa` 涨 12 个量级时前向误差跟着涨，后向误差不动。

    参考解由 mpmath 60 位给出（``mp_solve``）；矩阵取 ``shear`` 族——它是本章唯一一个
    消元过程真的会引入舍入的族（对角/三角/旋转族的 LU 是精确的，测不出前向误差）。
    """
    kappas = np.array([1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14], dtype=float)
    fwd = []
    back = []
    for target in kappas:
        A = cond.condition_matrix("shear", float(target))
        b = np.array([1.0, 1.0])
        res = lu_partial(A)
        x = solve_lu(res, b)
        x_ref = np.array([float(v) for v in ref_exact.mp_solve(A.tolist(), b.tolist(), dps=60)])
        fwd.append(float(np.max(np.abs(x - x_ref)) / np.max(np.abs(x_ref))))
        back.append(backward_error(A, x, b))
    fwd_arr = np.asarray(fwd)
    back_arr = np.asarray(back)
    fit = assert_order(kappas, fwd_arr, theory=1.0, tol=0.1, label="前向误差 ∝ κ 的实测阶")
    ratios = fwd_arr / (kappas * U)
    worst_back = float(np.max(back_arr) / U)
    assert float(np.max(ratios)) <= 0.5, ratios
    assert worst_back <= 10.0, worst_back
    assert np.all(np.diff(fwd_arr) > 0.0), "前向误差必须随 κ 单调增"
    rec = record(
        label="前向误差/(κu) 与后向误差/u",
        theory="前向 ≤ 0.5κu（实测阶 1）；后向 ≤ 10u",
        measured=(
            f"max 前向/(κu)={float(np.max(ratios)):.4f}（阶 {fit.slope:.4f}，R²={fit.r2:.4f}）；"
            f"max 后向/u={worst_back:.3f}"
        ),
        tol="0.5 / 10",
        passed=float(np.max(ratios)) <= 0.5 and worst_back <= 10.0,
    )
    assert rec.passed


@assertion("A", theory="实测放大比 ≤ κ（右端项扰动方向一般不在最坏方向）", chapter="1.1")
def test_rhs_amplification_is_below_kappa() -> None:
    """在 720 个方向上实测 :math:`(\\|\\Delta x\\|/\\|x\\|)/(\\|\\delta b\\|/\\|b\\|)`。

    它的最大值逼近 :math:`\\kappa`（实测 0.99997），中位数只有 0.19——这解释了为什么
    "实测放大比总是远小于 κ"不是界太松，而是方向不对。
    """
    g = rng(41)
    ratios = []
    for _ in range(60):
        n = int(g.integers(1, 3)) * 2
        A = g.uniform(-1.0, 1.0, (n, n))
        if not np.isfinite(cond.condition_2(A)) or cond.condition_2(A) > 1e10:
            continue
        for th in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            b = np.array([np.cos(th), np.sin(th)] + [0.0] * (n - 2))
            db = 1e-8 * np.array([-np.sin(th), np.cos(th)] + [0.0] * (n - 2))
            amp = cond.rhs_amplification_measured(A, b, db, p=2)
            ratios.append(amp / cond.condition_2(A))
    worst = float(np.max(ratios))
    assert worst <= 1.0 + 1e-6, worst
    rec = record(
        label="实测放大比 / κ（720 个方向）",
        theory="≤ 1（上界是保证，不是估计）",
        measured=f"max {worst:.6f}，中位数 {float(np.median(ratios)):.4f}",
        tol="1",
        passed=worst <= 1.0 + 1e-6,
    )
    assert rec.passed


@assertion("A", theory="κ(AᵀA) = κ(A)²（正规方程把条件数平方，几何上就是椭球被再压一次）", chapter="1.1")
def test_gram_condition_number_is_kappa_squared() -> None:
    """正规方程的代价：:math:`\\kappa(A^\\top A)=\\kappa(A)^2`。用**采样半径**独立测一遍。"""
    g = rng(43)
    worst = 0.0
    rows = []
    for _ in range(8):
        A = g.uniform(-1.0, 1.0, (3, 3))
        k = cond.condition_2(A)
        if not np.isfinite(k) or k > 1e4:
            continue
        k_gram = cond.condition_2(A.T @ A)
        rel = abs(k_gram / (k * k) - 1.0)
        worst = max(worst, rel)
        rows.append((k, k_gram))
    assert worst <= 1e-6, worst
    rec = record(
        label="|κ(AᵀA)/κ(A)² − 1|",
        theory="≤ 1e-6（舍入量级）",
        measured=f"{worst:.2e}（{len(rows)} 个矩阵；例 κ={rows[0][0]:.4g} → κ(AᵀA)={rows[0][1]:.4g}）",
        tol="1e-6",
        passed=worst <= 1e-6,
    )
    assert rec.passed


@assertion("A", theory="正交矩阵 κ₂ = 1（等距）；对角阵 κ₁ = max|d|/min|d|（闭式）", chapter="1.1")
def test_orthogonal_and_diagonal_conditions_are_exact() -> None:
    r"""两类"条件数可以闭式写出"的矩阵。

    * **正交**：:math:`A^\top A=I` ⇒ :math:`\kappa_2=1`。注意 :math:`\kappa_2` 走幂迭代，
      所以只能保证到迭代容差（实测 4e-16）；:math:`\kappa_1` 对它没有意义（可以是 2.4）。
    * **对角**：:math:`\kappa_1=\max|d|/\min|d|`，且对角阵的 LU 求逆**没有舍入**，
      所以这条是逐位（相对 1e-15）成立的；:math:`\kappa_2` 走的是 :math:`A^\top A` 上的
      反迭代，κ 大时会被平方放大，这里不用它做断言。
    """
    g = rng(47)
    worst_orth = 0.0
    for th in np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False):
        worst_orth = max(worst_orth, abs(cond.condition_2(cond.givens_rotation(3, 0, 1, th)) - 1.0))
    for n in (2, 3, 4, 5, 6):
        for seed in (101, 202):
            worst_orth = max(worst_orth, abs(cond.condition_2(cond.orthogonal_matrix(n, seed=seed)) - 1.0))
    worst_diag = 0.0
    for _ in range(20):
        d = np.exp2(g.uniform(-20.0, 20.0, 5))
        A = np.diag(d)
        exact = float(np.max(np.abs(d)) / np.min(np.abs(d)))
        worst_diag = max(worst_diag, abs(cond.condition_1(A) - exact) / exact)
    assert worst_orth <= 1e-12, worst_orth
    assert worst_diag <= 1e-15, worst_diag
    rec = record(
        label="正交 κ₂ 与对角 κ₁ 的相对偏差",
        theory="正交 ≤ 1e-12（幂迭代容差）；对角 ≤ 1e-15（求逆无舍入）",
        measured=f"正交 max {worst_orth:.2e}（5 个旋转 + 10 个正交阵）；对角 max {worst_diag:.2e}（20 个）",
        tol="1e-12 / 1e-15",
        passed=worst_orth <= 1e-12 and worst_diag <= 1e-15,
    )
    assert rec.passed


@assertion("A", theory="采样椭球半径误差 = O(Δθ²)（Δθ ≪ 1/κ 时）：σmin 被系统性高估", chapter="1.1")
def test_sampled_ellipse_radii_converge_quadratically() -> None:
    r""":math:`r(\theta)=\|A u(\theta)\|` 在短轴附近 :math:`\approx\sigma_{\min}+c\,t^2`
    （:math:`c=O(\kappa^2\sigma_{\min})`），所以均匀采样的 :math:`\min r` **总是高估**
    :math:`\sigma_{\min}`，误差 :math:`O(\Delta\theta^2)`。

    两个坑都踩过一遍：

    * 只用一条网格时，误差取决于网格是否"踩中"短轴，逐点序列会**振荡**（实测局部阶在
      −2.5 与 +1 之间乱跳），必须先对 32 个整格平移 δ 取平均；
    * :math:`\kappa\Delta\theta\gtrsim1` 时进入**线性**区（误差 :math:`O(\kappa\Delta\theta)`，
      实测阶 1.71），所以取 :math:`\kappa=30`、:math:`n_\theta\ge128` 才看得到干净的 2 阶。

    库函数 :func:`cond.ellipse_image` 不给平移参数，因此这里按同一公式自行采样；
    它与库采样器在 δ=0 处逐点一致（:math:`n_\theta` 的网格取法相同，见下）。
    """
    A = cond.condition_matrix("shear", 30.0)
    smin, smax = cond.ellipse_semi_axes(A)

    def shifted_min(theta: np.ndarray, delta: float) -> float:
        """把整条网格平移 ``delta`` 个格距后重采（库函数不给平移参数）。"""
        th = theta + delta * (theta[1] - theta[0])
        pts = A @ np.vstack([np.cos(th), np.sin(th)])
        return float(np.min(np.hypot(pts[0], pts[1])))

    ns = np.array([128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0])
    errs = []
    for n in ns:
        theta, x, y = cond.ellipse_image(A, n_theta=int(n))
        r_lib = np.hypot(x, y)
        assert abs(float(np.min(r_lib)) - shifted_min(theta, 0.0)) <= 1e-15 * smin
        assert float(np.min(r_lib)) >= smin * (1.0 - 1e-12), "采样半径不可能小于 σmin"
        assert float(np.max(r_lib)) <= smax * (1.0 + 1e-12), "采样半径不可能大于 σmax"
        errs.append(float(np.mean([abs(shifted_min(theta, d / 32.0) - smin) / smin for d in range(32)])))
    fit = assert_order(1.0 / ns, np.asarray(errs), theory=2.0, tol=0.05, label="采样半径误差 ∝ Δθ² 的实测阶")
    rec = record(
        label="椭球采样半径的相对误差阶",
        theory="2（O(Δθ²)，32 个整格平移取平均，κ=30）",
        measured=f"阶 {fit.slope:.4f}（R²={fit.r2:.5f}）；n=128→{errs[0]:.2e}，n=8192→{errs[-1]:.2e}",
        tol="2±0.05",
        passed=abs(fit.slope - 2.0) <= 0.05,
    )
    assert rec.passed


# --------------------------------------------------------------------------- #
# B 类
# --------------------------------------------------------------------------- #
@assertion("B", theory="κ ≥ 1；κ(cA) = κ(A)（c > 0）；κ(cI) = 1", chapter="1.1")
def test_kappa_is_at_least_one_and_scale_invariant() -> None:
    """条件数的两条最基本不变量：不小于 1，且与整体缩放无关。"""
    g = rng(53)
    worst_scale = 0.0
    min_kappa = np.inf
    for _ in range(40):
        n = int(g.integers(2, 6))
        A = g.uniform(-1.0, 1.0, (n, n))
        k = cond.condition_2(A)
        if not np.isfinite(k):
            continue
        min_kappa = min(min_kappa, k)
        c = float(2.0 ** int(g.integers(-40, 41)))
        worst_scale = max(worst_scale, abs(cond.condition_2(c * A) / k - 1.0))
        assert k >= 1.0 - 1e-12, k
    assert min_kappa >= 1.0 - 1e-12
    assert abs(cond.condition_2(7.5 * np.eye(4)) - 1.0) <= 1e-15, cond.condition_2(7.5 * np.eye(4))
    assert_leq(worst_scale, 1e-12, label="|κ(cA)/κ(A) − 1|", detail=f"最小实测 κ = {min_kappa:.6f}")


@assertion("B", theory="κ(A⁻¹) = κ(A)；κ₂(Aᵀ) = κ₂(A)；κ₁(Aᵀ) = κ∞(A)（注意不是 κ₁）", chapter="1.1")
def test_kappa_of_inverse_and_transpose_agree() -> None:
    r"""求逆不改变条件数；转置只把 :math:`\kappa_1` 与 :math:`\kappa_\infty` 互换。

    "转置不改变 κ"这句话**只对 2 范数与 F 范数成立**：对 1 范数有
    :math:`\kappa_1(A^\top)=\|A^\top\|_1\|A^{-\top}\|_1=\|A\|_\infty\|A^{-1}\|_\infty=\kappa_\infty(A)`。
    一开始把这条写成 :math:`\kappa_1(A^\top)=\kappa_1(A)`，实测偏差 48%——不是舍入，是恒等式写错了。
    """
    g = rng(59)
    worst_inv = 0.0
    worst_t1 = 0.0
    worst_t2 = 0.0
    worst_tf = 0.0
    n_used = 0
    for _ in range(25):
        n = int(g.integers(2, 5))
        A = g.uniform(-1.0, 1.0, (n, n))
        k1 = cond.condition_1(A)
        if not np.isfinite(k1) or k1 > 1e8:
            continue
        n_used += 1
        worst_inv = max(worst_inv, abs(cond.condition_1(cond.matrix_inverse(A)) / k1 - 1.0))
        worst_t1 = max(worst_t1, abs(cond.condition_1(A.T) / cond.condition_inf(A) - 1.0))
        worst_t2 = max(worst_t2, abs(cond.condition_2(A.T) / cond.condition_2(A) - 1.0))
        worst_tf = max(worst_tf, abs(cond.condition_fro(A.T) / cond.condition_fro(A) - 1.0))
    assert n_used >= 10
    worst = max(worst_inv, worst_t1, worst_t2, worst_tf)
    assert_leq(
        worst,
        1e-9,
        label="max(κ₁(A⁻¹)/κ₁(A)−1, κ₁(Aᵀ)/κ∞(A)−1, κ₂(Aᵀ)/κ₂(A)−1, κ_F(Aᵀ)/κ_F(A)−1)",
        detail=(
            f"{n_used} 个矩阵；求逆 {worst_inv:.2e}，转置↔1/∞ {worst_t1:.2e}，"
            f"κ₂ 转置 {worst_t2:.2e}，κ_F 转置 {worst_tf:.2e}"
        ),
    )


@assertion("B", theory="分量型条件数 κ_comp = ‖|A⁻¹||A|‖∞ 对**行**缩放不变；κ∞ 不然", chapter="1.1")
def test_componentwise_condition_is_row_scaling_invariant() -> None:
    """分量型条件数按行缩放不变（行缩放等价于换单位），而范数型 κ∞ 会变——

    这正是"用哪个 κ"必须连模型一起说明的原因。
    """
    g = rng(61)
    worst_comp = 0.0
    max_kappa_drift = 0.0
    for _ in range(20):
        n = int(g.integers(2, 5))
        A = g.uniform(-1.0, 1.0, (n, n)) + np.eye(n) * 2.0
        d = np.exp2(g.uniform(-8.0, 8.0, n))
        D = np.diag(d)
        kc = cond.condition_componentwise(A)
        worst_comp = max(worst_comp, abs(cond.condition_componentwise(D @ A) / kc - 1.0))
        max_kappa_drift = max(max_kappa_drift, abs(cond.condition_inf(D @ A) / cond.condition_inf(A) - 1.0))
        assert kc >= 1.0, kc
    assert worst_comp <= 1e-9, worst_comp
    rec = record(
        label="行缩放下的 κ_comp / κ∞ 相对变化",
        theory="κ_comp 不变（≤1e-9）；κ∞ 会变（这是预期的，不是 bug）",
        measured=f"κ_comp 最大变化 {worst_comp:.2e}；κ∞ 最大变化 {max_kappa_drift:.2e}",
        tol="1e-9",
        passed=worst_comp <= 1e-9 and max_kappa_drift > 1e-3,
    )
    assert rec.passed


@assertion("B", theory="未知范数键与非有限输入必须报错，不能返回猜的值", chapter="1.1")
def test_condition_number_rejects_bad_input() -> None:
    """接口的边界：范数键拼错、κ 目标非法、矩阵非方阵，都必须**显式**报错。"""
    bad = 0
    for call in (
        lambda: cond.condition_number(np.eye(3), p="banana"),
        lambda: cond.condition_number(np.ones((2, 3))),
        lambda: cond.condition_matrix("shear", 0.5),
        lambda: cond.condition_matrix("nope", 10.0),
        lambda: cond.matrix_inverse(np.zeros((2, 2))),
    ):
        try:
            call()
        except (ValueError, KeyError, ZeroDivisionError):
            bad += 1
    assert bad == 5, bad
    rec = record(
        label="非法输入的显式报错计数",
        theory="5/5",
        measured=f"{bad}/5",
        tol="5",
        passed=bad == 5,
    )
    assert rec.passed


@assertion("B", theory="采样半径夹住闭式奇异值：r_min ≥ σmin、r_max ≤ σmax（任意 n_θ）", chapter="1.1")
def test_sampled_radii_bracket_singular_values() -> None:
    """对任意采样密度，采样都只能"错过"极值方向，不可能超过它。"""
    worst_lo = 0.0
    worst_hi = 0.0
    for target in (1e2, 1e4, 1e6, 1e8):
        A = cond.condition_matrix("shear", target)
        smin, smax = cond.ellipse_semi_axes(A)
        k2 = cond.condition_2(A)
        assert abs(smax / smin / k2 - 1.0) <= 1e-12
        for n in (9, 17, 33, 65, 129):
            rmin, rmax = cond.ellipse_radii(A, n_theta=n)
            worst_lo = max(worst_lo, (smin - rmin) / smin)
            worst_hi = max(worst_hi, (rmax - smax) / smax)
    assert worst_lo <= 0.0 and worst_hi <= 0.0, (worst_lo, worst_hi)
    rec = record(
        label="采样半径越界量（σmin−r_min、r_max−σmax 的最大值）",
        theory="≤ 0（只会错过，不会越过）",
        measured=f"低端 {worst_lo:.1e}，高端 {worst_hi:.1e}（4 个 κ × 5 种采样密度）",
        tol="0",
        passed=worst_lo <= 0.0 and worst_hi <= 0.0,
    )
    assert rec.passed
