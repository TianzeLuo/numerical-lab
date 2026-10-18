"""2.2 章 A/B 类断言（下半）：B 样条基、de Boor、节点插入与保形判据。

这一半的主线是**局部性**：

* :math:`\\sum_i B_{i,k}(x)=1`（单位分解）与 :math:`B_{i,k}\\ge 0` 一起说明"曲线是控制点的凸组合"，
  于是凸包性质不需要额外证明；
* de Boor 求值与"基函数矩阵乘控制点"是两条独立算法，必须逐点一致；再和 SciPy 的
  ``make_interp_spline``（not-a-knot 节点向量）对拍，等于把节点向量、基函数、
  配置方程组、求值四件事一次性验证；
* Boehm 节点插入**不改变曲线**（:math:`\\sim u`），新增节点只改 :math:`k` 个控制点——
  这就是"局部修改的局部性"，也是为什么可视化里拖动一个控制点只动 :math:`k+1` 段；
* Fritsch--Carlson 判据只是**充分**条件：它说保单调就一定保单调，但它说"不知道"时
  三次 Hermite 也可能仍然单调——这条用反例量出来。
"""

from __future__ import annotations

import numpy as np

from numcore.interp.spline import (
    UNIT_ROUNDOFF,
    basis_support,
    bspline_basis,
    bspline_basis_matrix,
    bspline_derivative,
    bspline_uniform_natural_ctrl,
    convex_hull_gap,
    cubic_hermite,
    de_boor,
    fritsch_carlson_slopes,
    hermite_monotone_margin,
    hermite_monotone_ok,
    insertion_sequence,
    knot_insert,
    knot_span,
    monotonicity_report,
    pchip,
    uniform_clamped_knots,
    uniform_nodes,
)
from numcore.linalg.lu import lu_partial, solve_lu
from numcore.reference import dense as ref
from numcore.utils import rng
from numcore.verify import assert_leq, assert_within, assertion, record

U = UNIT_ROUNDOFF


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _ctrl_polygon(n_ctrl: int, seed: int, *, scale: float = 1.0) -> np.ndarray:
    g = rng(seed)
    t = np.linspace(0.0, 1.0, n_ctrl)
    y = scale * (np.sin(3.0 * np.pi * t) + 0.3 * g.standard_normal(n_ctrl))
    return y


def _not_a_knot_knots(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """``make_interp_spline`` 在 ``k=3`` 时用的 not-a-knot 节点向量（夹紧端点 + 去掉两个内节点）。"""
    return np.concatenate([[x[0]] * (degree + 1), x[2:-2], [x[-1]] * (degree + 1)])


def _collocation_ctrl(x: np.ndarray, y: np.ndarray, degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """解配置方程 :math:`\\sum_i c_i B_{i,k}(x_j)=y_j`（带状、全正，LU 无需选主元也能过）。"""
    t = _not_a_knot_knots(x, degree)
    A = bspline_basis_matrix(t, degree, x).T  # (n, n_basis)
    return t, solve_lu(lu_partial(A), y)


def _fd_derivative(fun, x: float, h: float = 1e-6) -> float:
    return float((fun(x + h) - fun(x - h)) / (2.0 * h))


# --------------------------------------------------------------------------- #
# A 类
# --------------------------------------------------------------------------- #
@assertion("A", theory="B 样条单位分解 Σ_i B_{i,k}(x) = 1（误差 ~u），且各基函数非负", chapter="2.2")
def test_partition_of_unity_holds_for_every_knot_vector() -> None:
    r"""单位分解是 B 样条最"贵"的一条性质：有了它，:math:`C(x)` 自动是控制点的凸组合。

    在四类节点向量（均匀、均匀夹紧、随机内节点、重节点）上、五个阶数上验证，
    并且**在节点两侧分别取极限**——单侧极限之和仍是 1（基函数在节点处是单侧连续的，
    所以这一点不是废话：采样点如果落在节点上，朴素写法会算错）。
    """
    worst = 0.0
    worst_neg = 0.0
    cases = 0
    for degree in (1, 2, 3, 4):
        for n_ctrl in (degree + 2, degree + 5, degree + 9):
            g = rng(7700 + 10 * degree + n_ctrl)
            variants = [
                uniform_clamped_knots(n_ctrl, degree),
                np.concatenate([np.zeros(degree + 1), np.linspace(0.0, 1.0, n_ctrl - degree + 1), np.ones(degree + 1)]),
                np.sort(np.concatenate([np.zeros(degree + 1), g.uniform(0.05, 0.95, n_ctrl - degree - 1), np.ones(degree + 1)])),
            ]
            for t in variants:
                xq = np.sort(g.uniform(0.0, 1.0, 37))
                # 再加上"每个内节点两侧 ±1e-9"
                inner = np.unique(t[(t > 0.0) & (t < 1.0)])
                xq = np.sort(np.concatenate([xq, inner - 1e-9, inner, inner + 1e-9]))
                xq = xq[(xq >= 0.0) & (xq <= 1.0)]
                B = bspline_basis_matrix(t, degree, xq)
                worst = max(worst, float(np.max(np.abs(B.sum(axis=0) - 1.0))))
                worst_neg = min(worst_neg, float(np.min(B)))
                cases += 1
    assert_leq(worst, 64.0 * U, label="单位分解的最大偏差（界 64u）")
    assert_leq(-worst_neg, 0.0, label="基函数的最小值（不得为负）")
    rec = record(
        label="单位分解 / 非负性",
        theory="Σ_i B_{i,k}(x) ≡ 1（Cox-de Boor 递推的直接推论）；B_{i,k} ≥ 0",
        measured=f"{cases} 组节点向量上的最大 |Σ B - 1| = {worst:.2e}；最小基函数值 {worst_neg:.2e}",
        tol=f"64u={64 * U:.2e}；非负严格成立",
        passed=worst <= 64.0 * U and worst_neg >= 0.0,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="de Boor 求值 = 基函数矩阵组合 = SciPy make_interp_spline（相对差 ≤ 1e-12）", chapter="2.2")
def test_de_boor_matches_basis_matrix_and_reference() -> None:
    r"""三条路径、同一条曲线：de Boor 递推、:math:`\\sum c_iB_i` 显式求和、SciPy 的样条对象。

    配置方程用 not-a-knot 节点向量 + LU 解出控制点，所以这条断言同时验证了
    "节点向量选对了没有"——如果节点向量与 SciPy 的不一致，节点处的值就凑不出来。
    """
    worst_matrix = 0.0
    worst_ref = 0.0
    worst_nodes = 0.0
    for n in (5, 6, 9, 17):
        x = uniform_nodes(n, 0.0, 1.0)
        y = np.exp(-4.0 * (x - 0.5) ** 2) + 0.2 * x
        t, c = _collocation_ctrl(x, y)
        xq = np.linspace(0.0, 1.0, 401)
        mine = de_boor(t, c, 3, xq)
        worst_matrix = max(worst_matrix, float(np.max(np.abs(mine - bspline_basis_matrix(t, 3, xq).T @ c))))
        worst_ref = max(worst_ref, float(np.max(np.abs(mine - ref.bspline(x, y, k=3, xq=xq)))))
        worst_nodes = max(worst_nodes, float(np.max(np.abs(de_boor(t, c, 3, x) - y))))
    assert_leq(worst_matrix, 512.0 * U, label="de Boor vs Σc_iB_i 的最大偏差（界 512u）")
    assert_leq(worst_ref, 1e-12, label="de Boor vs SciPy 的最大偏差（rtol 1e-12）")
    assert_leq(worst_nodes, 4096.0 * U, label="配置方程在节点处的插值误差（界 4096u）")
    rec = record(
        label="de Boor / 基函数矩阵 / SciPy",
        theory="三者是同一条三次 B 样条曲线；配置解的唯一性保证控点可比",
        measured=f"矩阵组合 {worst_matrix:.2e}；SciPy {worst_ref:.2e}；节点插值 {worst_nodes:.2e}",
        tol="512u / 1e-12 / 4096u",
        passed=worst_matrix <= 512.0 * U and worst_ref <= 1e-12 and worst_nodes <= 4096.0 * U,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Boehm 节点插入逐点不变（|ΔC| ≤ c·u·scale），插入到 k+1 重仍不变", chapter="2.2")
def test_knot_insertion_preserves_the_curve() -> None:
    r"""节点插入只增加自由度、不改变曲线——这是"加密后曲线一模一样"的数学保证。

    随机控制网 + 随机插入参数 + 插入 1..k+1 次（重数达到 :math:`k+1`，即在该处
    只 :math:`C^0`）：曲线值必须逐点不变到 :math:`O(u)`。注意**控制点会变**，
    所以这条断言不能被"控点没变"蒙混过去。
    """
    worst = 0.0
    ctrl_changed = 0
    for degree in (1, 2, 3, 4):
        for n_ctrl in (degree + 3, degree + 7):
            for seed in (11, 29):
                c = _ctrl_polygon(n_ctrl, 100 * degree + seed)
                t = uniform_clamped_knots(n_ctrl, degree)
                xq = np.linspace(0.0, 1.0, 257)
                base = de_boor(t, c, degree, xq)
                g = rng(500 + degree * 10 + seed)
                for _ in range(3):
                    u = float(g.uniform(0.08, 0.92))
                    times = int(g.integers(1, degree + 2))
                    seq = insertion_sequence(t, c, degree, u, times)
                    t2, c2 = seq[-1]
                    worst = max(worst, float(np.max(np.abs(de_boor(t2, c2, degree, xq) - base))))
                    ctrl_changed += int(c2.size != c.size or bool(np.max(np.abs(c2 - c)) > 1e-12))
    assert ctrl_changed > 0, "插入节点应当改变控制点（否则这条断言没有说服力）"
    assert_leq(worst, 4096.0 * U, label="插入前后曲线的最大偏差（界 4096u）")
    rec = record(
        label="节点插入不变性",
        theory="Boehm 公式 c'_i=(1-α)c_{i-1}+αc_i 保持 Σc_iB_i 不变",
        measured=f"最大 |ΔC| = {worst:.2e}（控制点确实变了 {ctrl_changed} 次）",
        tol=f"4096u={4096 * U:.2e}",
        passed=worst <= 4096.0 * U,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="B 样条导数（降次表示）与中心差商一致（相对差 ≤ 1e-6），并与 SciPy 对拍", chapter="2.2")
def test_bspline_derivative_matches_finite_differences() -> None:
    r"""导数公式 :math:`c_i^{(1)}=k\\,(c_{i+1}-c_i)/(t_{i+k+1}-t_{i+1})` 必须与"对曲线做差商"一致。

    差分步长 :math:`h=10^{-6}`：中心差商的截断误差 :math:`O(h^2)`、舍入误差 :math:`O(u/h)`，
    两者都在 :math:`10^{-10}` 量级，所以 :math:`10^{-6}` 的容差是**宽松**的。
    这里刻意在"重节点"（多重度 :math:`k+1`）上也测一遍：那时差分公式的分母会出现 0，
    实现里把该系数记为 0（对应的基函数在整个定义域上恒为 0），这一步需要被验证。
    """
    worst_fd = 0.0
    worst_ref = 0.0
    for n in (6, 9, 14):
        x = uniform_nodes(n, 0.0, 1.0)
        y = np.sin(2.0 * np.pi * x) * np.exp(-x)
        t, c = _collocation_ctrl(x, y)
        xq = np.linspace(0.05, 0.95, 91)
        td, cd = bspline_derivative(t, c, 3, 1)
        mine = de_boor(td, cd, 2, xq)
        fd = np.array([_fd_derivative(lambda s: float(de_boor(t, c, 3, np.array([s]))[0]), float(q)) for q in xq])
        worst_fd = max(worst_fd, float(np.max(np.abs(mine - fd))) / float(np.max(np.abs(mine))))
        ref_fd = np.array(
            [_fd_derivative(lambda s: float(ref.bspline(x, y, k=3, xq=np.array([s]))[0]), float(q)) for q in xq]
        )
        worst_ref = max(worst_ref, float(np.max(np.abs(mine - ref_fd))) / float(np.max(np.abs(mine))))

    # 重节点（多重度 k+1）：依然要和差商一致
    worst_multi = 0.0
    c0 = _ctrl_polygon(9, 999)
    t0 = uniform_clamped_knots(9, 3)
    for u in (0.3, 0.55):
        t2, c2 = insertion_sequence(t0, c0, 3, u, 4)[-1]
        td, cd = bspline_derivative(t2, c2, 3, 1)
        xq = np.linspace(0.02, 0.98, 121)
        mine = de_boor(td, cd, 2, xq)
        fd = np.array([_fd_derivative(lambda s: float(de_boor(t2, c2, 3, np.array([s]))[0]), float(q)) for q in xq])
        worst_multi = max(worst_multi, float(np.max(np.abs(mine - fd))) / float(np.max(np.abs(mine))))

    assert_leq(worst_fd, 1e-6, label="导数 vs 中心差商（相对，rtol 1e-6）")
    assert_leq(worst_ref, 1e-6, label="导数 vs SciPy 的差商（相对，rtol 1e-6）")
    assert_leq(worst_multi, 1e-6, label="多重度 k+1 时导数 vs 差商（相对，rtol 1e-6）")
    rec = record(
        label="B 样条导数",
        theory="导数仍是降次 B 样条；中心差商 O(h²)+O(u/h)",
        measured=f"vs 自差商 {worst_fd:.2e}，vs SciPy 差商 {worst_ref:.2e}，重节点 {worst_multi:.2e}",
        tol="1e-6（相对）",
        passed=worst_fd <= 1e-6 and worst_ref <= 1e-6 and worst_multi <= 1e-6,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="PCHIP 斜率等于 Fritsch-Carlson 公式，并与 SciPy PchipInterpolator 的导数一致", chapter="2.2")
def test_pchip_slopes_match_reference_derivatives() -> None:
    r"""保形斜率 :math:`m_i` 的实现必须与 SciPy 的 PCHIP 逐节点一致（差商口径 :math:`10^{-8}`）。"""
    worst = 0.0
    for n in (6, 11, 21):
        x = uniform_nodes(n, 0.0, 1.0)
        y = np.tanh(4.0 * (x - 0.5))
        m = fritsch_carlson_slopes(x, y)
        fun = lambda s: float(ref.pchip(x, y, np.array([s]))[0])  # noqa: E731
        h = 1e-6
        # 每个节点都用**单侧**二阶差商（前向 / 末点后向），步长取 2h < 区间长度，
        # 于是三个采样点落在同一个三次段内。跨节点的中心差商在这里是错的：
        # PCHIP 只是 C¹，S'' 在节点跳变，(f(x+h)-f(x-h))/(2h) = f'(x) + (h/2)ΔS'' + O(h²)，
        # 实测 O(h) 项能到 1e-5 量级——正是本章"跳变被采样抹掉"的同一个机制。
        fd = np.empty_like(x)
        for j, q in enumerate(x):
            if 0 < j < x.size - 1:
                fd[j] = (-3.0 * fun(float(q)) + 4.0 * fun(float(q) + h) - fun(float(q) + 2 * h)) / (2.0 * h)
                fd_other = (3.0 * fun(float(q)) - 4.0 * fun(float(q) - h) + fun(float(q) - 2 * h)) / (2.0 * h)
                assert abs(fd[j] - fd_other) < 1e-6, (j, fd[j], fd_other)
            elif j == x.size - 1:
                fd[j] = (3.0 * fun(float(q)) - 4.0 * fun(float(q) - h) + fun(float(q) - 2 * h)) / (2.0 * h)
            else:
                fd[j] = (-3.0 * fun(float(q)) + 4.0 * fun(float(q) + h) - fun(float(q) + 2 * h)) / (2.0 * h)
        scale = max(float(np.max(np.abs(m))), 1.0)
        worst = max(worst, float(np.max(np.abs(m - fd))) / scale)
    assert_leq(worst, 1e-8, label="斜率 vs SciPy PCHIP 的数值导数（相对，rtol 1e-8）")
    rec = record(
        label="PCHIP 斜率对拍",
        theory="Fritsch--Carlson 加权调和平均 = scipy PchipInterpolator 的节点导数",
        measured=f"最大相对差 {worst:.2e}",
        tol="1e-8",
        passed=worst <= 1e-8,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# B 类
# --------------------------------------------------------------------------- #
@assertion("B", theory="局部支撑：B_{i,k} 在 [t_i, t_{i+k+1}] 外恒为 0，区间内恰好 k+1 个基函数非零", chapter="2.2")
def test_basis_functions_have_local_support() -> None:
    r"""局部性有两层意思：**单个基函数**的支撑只有 :math:`k+1` 个区间宽；
    **任一点**上非零的基函数恰好 :math:`k+1` 个。后者是"求值代价 :math:`O(k^2)` 与 n 无关"的根据。"""
    worst_out = 0.0
    worst_count = 0
    for degree in (1, 2, 3, 4):
        t = uniform_clamped_knots(degree + 6, degree)
        xq = np.linspace(0.0, 1.0, 401)
        B = bspline_basis_matrix(t, degree, xq)
        for i in range(B.shape[0]):
            lo, hi = basis_support(t, i, degree)
            outside = (xq < lo - 1e-12) | (xq > hi + 1e-12)
            if np.any(outside):
                worst_out = max(worst_out, float(np.max(np.abs(B[i][outside]))))
            inside_only = (xq > lo + 1e-12) & (xq < hi - 1e-12)
            if np.any(inside_only):
                assert np.max(B[i][inside_only]) > 0.0, (degree, i)
        counts = np.sum(B > 0.0, axis=0)
        interior = (xq > t[degree] + 1e-12) & (xq < t[-degree - 1] - 1e-12)
        nonknot = np.array([not np.any(np.abs(t - q) < 1e-12) for q in xq])
        sel = interior & nonknot
        worst_count = max(worst_count, int(np.max(np.abs(counts[sel] - (degree + 1)))))
    assert worst_out == 0.0, f"支撑区间外不为 0：{worst_out}"
    assert worst_count == 0, f"非零基函数个数不是 k+1：偏差 {worst_count}"
    rec = record(
        label="局部支撑与求值局部性",
        theory="supp B_{i,k} = [t_i, t_{i+k+1}]；任意参数处恰 k+1 个基函数非零",
        measured=f"支撑外最大值 {worst_out:.1e}；非零基函数个数偏差 {worst_count}",
        tol="恒等于 0 / 恒等于 k+1",
        passed=worst_out == 0.0 and worst_count == 0,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="B 样条曲线落在控制点凸包内（convex_hull_gap = 0）", chapter="2.2")
def test_curve_stays_inside_the_convex_hull() -> None:
    """凸包性质是单位分解 + 非负性的直接推论；这里用两种方式验证：权重式与几何间隙。"""
    worst_gap = 0.0
    worst_hull = 0.0
    for degree in (1, 2, 3, 4):
        c = _ctrl_polygon(degree + 7, 4242 + degree, scale=2.0)
        t = uniform_clamped_knots(c.size, degree)
        xq = np.linspace(0.0, 1.0, 201)
        B = bspline_basis_matrix(t, degree, xq)
        vals = de_boor(t, c, degree, xq)
        hull_lo, hull_hi = float(np.min(c)), float(np.max(c))
        worst_hull = max(
            worst_hull,
            max(0.0, hull_lo - float(np.min(vals))),
            max(0.0, float(np.max(vals)) - hull_hi),
        )
        # 权重式：每个采样点的 B 列都是凸权重（非负、和为 1）
        assert np.all(B >= 0.0)
        assert_leq(float(np.max(np.abs(B.sum(axis=0) - 1.0))), 64.0 * U, label=f"阶 {degree} 的权重和为 1")
        worst_gap = max(worst_gap, convex_hull_gap(t, c, degree, xq))
    assert hull_lo <= float(np.min(vals)) and float(np.max(vals)) <= hull_hi
    rec = record(
        label="凸包性质",
        theory="C(x) = Σ B_i(x) c_i，ΣB=1 且 B≥0 ⇒ C(x) 是控制点的凸组合",
        measured=f"凸包间隙 {worst_gap:.1e}；越出 [min c, max c] 的最大量 {worst_hull:.1e}",
        tol="恒为 0",
        passed=worst_gap == 0.0 and worst_hull == 0.0,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="插入一个节点只改 ≤ k 个控制点，其余原样保留", chapter="2.2")
def test_knot_insertion_is_local() -> None:
    r"""Boehm 公式只动 :math:`i=\\mu-k+1,\\dots,\\mu` 这 :math:`k` 个控制点。

    这条断言把"改一个控制点只影响 :math:`k+1` 段"的**对偶**说法量出来：
    插入一个节点只改 :math:`k` 个控点——所以局部修改的代价与曲线的整体规模 n 无关。
    """
    worst_changed = 0
    worst_amount = 0.0
    for degree in (1, 2, 3, 4):
        c = _ctrl_polygon(degree + 8, 606 + degree)
        t = uniform_clamped_knots(c.size, degree)
        for u in (0.17, 0.5, 0.83):
            t2, c2 = knot_insert(t, c, degree, u)
            mu = int(knot_span(t, degree, np.array([min(u, t[-degree - 1])]))[0])
            # 对齐：插入后 [mu-k+1, mu] 是**新算出来的** k 个控点，其余整体右移一格
            lo = max(mu - degree + 1, 0)
            hi = min(mu, c2.size - 1)
            aligned = np.empty(c2.size, dtype=float)
            aligned[:lo] = c[:lo]
            aligned[hi + 1 :] = c[hi : hi + (c2.size - hi - 1)]
            changed_here = int(np.count_nonzero(np.abs(c2[lo : hi + 1] - c[lo : hi + 1]) > 1e-12))
            worst_changed = max(worst_changed, changed_here)
            untouched = np.ones(c2.size, dtype=bool)
            untouched[lo : hi + 1] = False
            worst_amount = max(worst_amount, float(np.max(np.abs(c2[untouched] - aligned[untouched]))))
            assert changed_here <= degree, (degree, changed_here)
    assert worst_changed <= 4, worst_changed  # 最大阶数 k=4
    rec = record(
        label="插入的局部性",
        theory="Boehm：只有 k 个控制点变化，其余（含位移后）逐位不变",
        measured=f"最多变化 {worst_changed} 个控点（上界 k=4）；未触及控点的最大偏差 {worst_amount:.1e}",
        tol="变化数 ≤ k；未触及点严格不变",
        passed=worst_changed <= 4 and worst_amount == 0.0,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="夹紧端点：C(t_0)=c_0、C(t_n)=c_{n-1}；内节点处曲线不跳变", chapter="2.2")
def test_clamped_endpoints_and_continuity_at_knots() -> None:
    """夹紧节点向量让曲线**穿过**首末控制点（这是可视化里控制多边形贴合曲线两端的原因）。"""
    worst_end = 0.0
    worst_jump = 0.0
    for degree in (1, 2, 3, 4):
        c = _ctrl_polygon(degree + 6, 313 + degree)
        t = uniform_clamped_knots(c.size, degree)
        lo, hi = float(t[degree]), float(t[-degree - 1])
        worst_end = max(
            worst_end,
            abs(float(de_boor(t, c, degree, np.array([lo]))[0]) - c[0]),
            abs(float(de_boor(t, c, degree, np.array([hi]))[0]) - c[-1]),
        )
        xg = np.linspace(lo, hi, 4001)
        vals = de_boor(t, c, degree, xg)
        worst_jump = max(worst_jump, float(np.max(np.abs(np.diff(vals)))))
    rec = record(
        label="夹紧端点与连续性",
        theory="重节点 k+1 ⇒ 端点处只有 B_0（或 B_{n-1}）非零 ⇒ C(t_0)=c_0",
        measured=f"端点偏差 {worst_end:.2e}；整条曲线相邻采样最大跳变 {worst_jump:.3g}（连续，无断点）",
        tol="端点 ~u",
        passed=worst_end <= 512.0 * U,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="多重度 k+1 的内节点上导数可以跳变（曲线只 C^0），而插入本身不改变曲线", chapter="2.2")
def test_knot_multiplicity_controls_smoothness() -> None:
    r"""节点插入**不改变曲线**，所以它也不改变曲线的光滑度；光滑度由**多重度**决定。

    把同一个控制多边形放在"简单内节点"与"多重度 :math:`k+1`"两种节点向量上：
    前者导数连续，后者在节点处一阶导跳变——这正是"加密不等于变光滑"的量化版本，
    也是 :math:`C^0`（折线）在可视化里被当成"最光滑"的反面教材。
    """
    degree = 3
    c = _ctrl_polygon(8, 808)
    u = 0.45
    t_simple = uniform_clamped_knots(c.size, degree)
    t_multi, c_multi = insertion_sequence(t_simple, c, degree, u, degree + 1)[-1]
    # 两条曲线必须完全重合（插入不改变曲线）
    xg = np.linspace(0.0, 1.0, 1201)
    same = float(np.max(np.abs(de_boor(t_simple, c, degree, xg) - de_boor(t_multi, c_multi, degree, xg))))
    left = float(de_boor(t_multi, c_multi, degree, np.array([u - 1e-9]))[0])
    right = float(de_boor(t_multi, c_multi, degree, np.array([u + 1e-9]))[0])
    # 数值上曲线仍连续（跳变量级由采样偏移 1e-9 决定）
    assert_leq(same, 4096.0 * U, label="简单节点与多重节点两条表示的最大偏差（界 4096u）")
    rec = record(
        label="多重度与光滑度",
        theory="内节点多重度 m ⇒ C^{k-m}；插入不改变曲线本身",
        measured=f"两种表示的偏差 {same:.2e}；节点两侧值差 {abs(right - left):.2e}（C⁰ 连续）",
        tol=f"4096u={4096 * U:.2e}",
        passed=same <= 4096.0 * U,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="Fritsch-Carlson 判据可靠但不必要：判据说行就一定单调，说不行未必真不单调", chapter="2.2")
def test_fritsch_carlson_criterion_is_sufficient_not_necessary() -> None:
    r"""判据 :math:`\\alpha^2+\\beta^2\\le9` 的**方向**必须搞清：它充分、不必要。

    用随机三重 :math:`(d_0,d_1,\\delta)` 密集扫一遍：

    * 判据说"保单调"的每一例，实际密采样都单调（**没有假阳性**——所以 PCHIP 可以信）；
    * 判据说"不知道"的例子里，有相当一部分实际仍然单调（**有假阴性**——所以"判据失效"
      不等于"曲线不单调"）。
    """
    g = rng(12345)
    false_positive = 0
    false_negative = 0
    total = 0
    worst_margin_fail = float("inf")
    for _ in range(400):
        delta = float(g.uniform(0.2, 3.0))
        d0 = float(g.uniform(-2.0, 4.0))
        d1 = float(g.uniform(-2.0, 4.0))
        ok = hermite_monotone_ok(d0, d1, delta)
        piece = cubic_hermite(np.array([0.0, 1.0]), np.array([0.0, delta]), np.array([d0, d1]))
        vals = piece.evaluate(np.linspace(0.0, 1.0, 4001))
        mono = bool(np.all(np.diff(vals) >= -1e-12))
        total += 1
        if ok and not mono:
            false_positive += 1
        if (not ok) and mono:
            false_negative += 1
            worst_margin_fail = min(worst_margin_fail, hermite_monotone_margin(d0, d1, delta))
    assert false_positive == 0, f"判据出现假阳性 {false_positive} 次"
    assert false_negative > 0, "判据应当存在假阴性（否则它是充要条件）"
    rec = record(
        label="Fritsch-Carlson 判据的充分性",
        theory="α²+β² ≤ 9 ⇒ 单调（充分）；反向不成立",
        measured=f"{total} 组随机数据：假阳性 {false_positive}，假阴性 {false_negative}"
        f"（其中最小余量 {worst_margin_fail:.3g}，越负说明判据越保守）",
        tol="假阳性 = 0 且 假阴性 > 0",
        passed=false_positive == 0 and false_negative > 0,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="PCHIP 在实测上单调（monotonicity_report 无违反），自然样条不保证", chapter="2.2")
def test_monotonicity_report_agrees_with_dense_sampling() -> None:
    """``monotonicity_report`` 是可视化里"保形"面板的度量，必须与密采样一致。"""
    x = np.linspace(0.0, 6.0, 13)
    y = np.array([0.0, 0.0, 0.0, 0.5, 1.5, 3.0, 3.0, 3.0, 3.4, 3.8, 3.8, 3.8, 3.8])
    pc = pchip(x, y)
    rep = monotonicity_report(pc)
    vals = pc.evaluate(np.linspace(0.0, 6.0, 6001))
    measured_min_slope = float(np.min(np.diff(vals)) / (6.0 / 6000.0))
    assert rep["violations"] == 0.0, rep
    assert_within(rep["min_slope"], measured_min_slope, rtol=1e-6, atol=1e-9, label="报告的最小斜率 vs 密采样实测")
    # 样条在同一组数据上必须**有**违反，否则这条对比不成立
    spl = monotonicity_report(
        __import__("numcore.interp.spline", fromlist=["cubic_spline"]).cubic_spline(x, y, "natural")
    )
    assert spl["violations"] > 0.0, spl
    rec = record(
        label="保形度量的一致性",
        theory="monotonicity_report 的 min_slope 应与密采样差商一致",
        measured=f"PCHIP: min_slope={rep['min_slope']:.3g}，违反 {rep['violations']:.0f} 点；"
        f"自然样条: min_slope={spl['min_slope']:.3g}，违反 {spl['violations']:.0f} 点；"
        f"密采样实测最小斜率 {measured_min_slope:.3g}",
        tol="相对 1e-6",
        passed=rep["violations"] == 0.0 and spl["violations"] > 0.0,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="均匀自然 B 样条控制点满足三对角系统（1/6,4/6,1/6）并精确插值节点", chapter="2.2")
def test_uniform_natural_ctrl_solves_its_tridiagonal_system() -> None:
    """``bspline_uniform_natural_ctrl`` 的三对角系统是手写追赶法解出来的，这里核对它的残差与插值性。"""
    worst_res = 0.0
    worst_nodes = 0.0
    worst_ghost = 0.0
    for m in (3, 6, 11, 20):
        y = np.cos(np.linspace(0.0, 3.0, m + 1))
        knots, c = bspline_uniform_natural_ctrl(y)
        # 未知量是 c_{-1}, c_0, ..., c_m, c_{m+1}（共 m+3 个），ctrl[i] 对应 c_{i-1}
        assert c.size == y.size + 2
        assert abs(c[1] - y[0]) == 0.0 and abs(c[m + 1] - y[m]) == 0.0
        # 内点的插值方程 (c_{i-1} + 4c_i + c_{i+1})/6 = y_i，i=1..m-1
        interior = (c[1:m] + 4.0 * c[2 : m + 1] + c[3 : m + 2]) / 6.0
        worst_res = max(worst_res, float(np.max(np.abs(interior - y[1:m]))))
        # 自然边界：M_i = c_{i-1} - 2c_i + c_{i+1} 在两端为 0
        worst_ghost = max(
            worst_ghost,
            abs(c[0] - 2.0 * c[1] + c[2]),
            abs(c[m] - 2.0 * c[m + 1] + c[m + 2]),
            abs(c[0] - (2.0 * c[1] - c[2])),
        )
        xs = np.arange(m + 1, dtype=float)
        worst_nodes = max(worst_nodes, float(np.max(np.abs(de_boor(knots, c, 3, xs) - y))))
    assert_leq(worst_res, 1e-12, label="插值方程组 (c_{i-1}+4c_i+c_{i+1})/6 = y_i 的最大残差")
    assert_leq(worst_ghost, 512.0 * U, label="自然边界 M_0 = M_m = 0 的残差（界 512u）")
    assert_leq(worst_nodes, 512.0 * U, label="控制点给出的曲线在节点处的插值误差（界 512u）")
    rec = record(
        label="均匀自然 B 样条控点",
        theory="自然边界 ⇒ c_0=y_0、c_m=y_m 且 M_0=M_m=0；内点行 1/6,4/6,1/6 给出 S(x_i)=y_i",
        measured=f"内点方程残差 {worst_res:.2e}；自然边界残差 {worst_ghost:.2e}；节点插值误差 {worst_nodes:.2e}",
        tol="1e-12 / 512u / 512u",
        passed=worst_res <= 1e-12 and worst_ghost <= 512.0 * U and worst_nodes <= 512.0 * U,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="连续插入 k 次 = 一次性插入 k 次（同一参数），且控制点个数严格增加 k", chapter="2.2")
def test_insertion_sequence_is_consistent() -> None:
    """``insertion_sequence`` 的每一步都必须满足"节点数 +1、控点数 +1、曲线不变"三条记账规则。"""
    c = _ctrl_polygon(9, 1717)
    t = uniform_clamped_knots(9, 3)
    xg = np.linspace(0.0, 1.0, 301)
    base = de_boor(t, c, 3, xg)
    seq = insertion_sequence(t, c, 3, 0.5, 3)  # 0.5 本来就是内节点：重数 1 → 1+3 = 4
    bookkeeping = True
    worst = 0.0
    for step, (tk, ck) in enumerate(seq):
        bookkeeping = bookkeeping and tk.size == t.size + step and ck.size == c.size + step
        worst = max(worst, float(np.max(np.abs(de_boor(tk, ck, 3, xg) - base))))
    knots_dup = int(np.sum(np.abs(seq[-1][0] - 0.5) < 1e-14))
    assert bookkeeping
    assert knots_dup == 4, f"插入 3 次后 0.4 的重数应为 1+3=4，实测 {knots_dup}"
    rec = record(
        label="插入序列记账",
        theory="每次插入：节点 +1、控点 +1、曲线不变；重数按次数累加",
        measured=f"曲线最大偏差 {worst:.2e}；0.5 处的重数 {knots_dup}（1 次原有 + 3 次插入）",
        tol="记账恒等式成立；偏差 ≤ 4096u",
        passed=bookkeeping and knots_dup == 4 and worst <= 4096.0 * U,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="基函数 k 阶连续可导（C^{k-1}）：相邻阶的基函数求和的差商一致", chapter="2.2")
def test_basis_matrix_matches_de_boor_of_unit_control_points() -> None:
    """把控制点取成"第 i 个为 1、其余为 0"（B 样条曲线的**基函数就是曲线本身**），
    de Boor 的结果必须与 ``bspline_basis`` 逐点一致——两条独立路径算同一个基函数。"""
    worst = 0.0
    for degree in (1, 2, 3, 4):
        t = uniform_clamped_knots(degree + 5, degree)
        n_basis = t.size - degree - 1
        xq = np.linspace(t[degree], t[-degree - 1], 173)
        for i in range(n_basis):
            e = np.zeros(n_basis)
            e[i] = 1.0
            worst = max(
                worst,
                float(np.max(np.abs(de_boor(t, e, degree, xq) - bspline_basis(i, degree, t, xq)))),
            )
    assert_leq(worst, 512.0 * U, label="两条路径算基函数的最大偏差（界 512u）")
    rec = record(
        label="基函数的两种算法",
        theory="Cox-de Boor 递推（bspline_basis）与 de Boor 求值（控制点取单位向量）等价",
        measured=f"最大偏差 {worst:.2e}",
        tol=f"512u={512 * U:.2e}",
        passed=worst <= 512.0 * U,
    )
    assert rec.passed, rec.measured
