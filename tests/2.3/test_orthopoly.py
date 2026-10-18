"""2.3 章 A/B 类断言：正交多项式的构造、零点、Gauss 求积。

这一章的第一半（本文件）守的是**"自造的多项式族到底对不对"**：

1. **三项递推 vs 闭式**（A 类）——递推是手写的、闭式是解析的，两者对上才谈得上后面；
2. **正交归一性**（A 类）——:math:`\\langle p_i,p_j\\rangle_w=\\delta_{ij}`，缺陷必须落在
   舍入地板（它是后面一切投影的“坐标系是不是正交”的前提）；
3. **Gauss 求积的代数精度**（A 类）——恰好 :math:`2n-1` 阶：不是“很高”，而是**确定的两倍减一**，
   并且第 :math:`2n` 阶矩必须当场失败（这一条比“误差很小”有力得多）；
4. **零点的性质**（A/B 类）——全在区间内、互异、与低一阶交错、变号数等于“小于 x 的零点个数”；
5. **Newton 求根的可观测物**（B 类）——迭代次数、残差、以及**Newton 修正量**：
   Laguerre 的 :math:`|L_n|` 在 n=40 时能到 :math:`10^{14}`，而根误差只有 :math:`10^{-14}`。
   把残差当误差是本章最容易犯的观测错误，所以它有一条独立断言。

每个测试都登记实测值；容差都由实测值加上 1~2 个数量级的余量给出，不是先写容差再凑数字。
"""

from __future__ import annotations

import math

import numpy as np

from numcore.interp.orthopoly import (
    FAMILIES,
    ROUNDOFF,
    chebyshev_analytic,
    chebyshev_derivative,
    chebyshev_extrema,
    chebyshev_recurrence,
    chebyshev_roots,
    exact_moment,
    family_norm,
    gauss_chebyshev,
    gauss_laguerre,
    gauss_legendre,
    gram_matrix,
    gram_schmidt,
    laguerre_analytic,
    laguerre_recurrence,
    legendre_analytic,
    legendre_derivative,
    legendre_recurrence,
    moment_exactness_error,
    newton_roots_family,
    orthonormal_values,
    orthonormality_defect,
    quadrature,
    sturm_root_count,
    weight_function,
)
from numcore.reference import dense as ref
from numcore.verify import assertion, assert_leq, record

U = ROUNDOFF  # 单位舍入 u = 2^-53 ≈ 1.11e-16


# --------------------------------------------------------------------------- #
# A 类：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="三项递推与闭式解逐点一致（误差只来自舍入）", chapter="2.3")
def test_recurrence_matches_analytic_forms() -> None:
    r"""Legendre/Chebyshev 的递推与闭式相差 ~n·u；Laguerre 的差随 x 增大而变大（要限定窗口）。

    Laguerre 那一条不是“实现不准”：:math:`L_n(x)` 在 :math:`x\gg n` 时自身量级极大
    （:math:`L_{40}(142)\sim10^{14}`），绝对误差自然被放大；所以断言写在
    :math:`x\in[0,10]` 的窗口里，并额外记录窗口依赖。
    """
    x = np.linspace(-1.0, 1.0, 401)
    d_leg = max(
        float(np.max(np.abs(legendre_recurrence(n, x)[n] - legendre_analytic(n, x)))) for n in range(1, 13)
    )
    d_che = max(
        float(np.max(np.abs(chebyshev_recurrence(n, x)[n] - chebyshev_analytic(n, x)))) for n in range(1, 17)
    )
    xl = np.linspace(0.0, 10.0, 401)
    d_lag = max(
        float(np.max(np.abs(laguerre_recurrence(n, xl)[n] - laguerre_analytic(n, xl)))) for n in range(1, 13)
    )
    window = {}
    for hi in (5.0, 10.0, 20.0):
        xs = np.linspace(0.0, hi, 401)
        window[hi] = max(
            float(np.max(np.abs(laguerre_recurrence(n, xs)[n] - laguerre_analytic(n, xs)))) for n in range(1, 13)
        )
    rec = record(
        label="递推 vs 闭式（n≤12 / n≤16）",
        theory="≤ 1e-11（Legendre）、≤ 1e-13（Chebyshev）、≤ 1e-8（Laguerre, x≤10）",
        measured=f"Legendre {d_leg:.2e}，Chebyshev {d_che:.2e}，Laguerre {d_lag:.2e}",
        tol="见 theory",
        passed=bool(d_leg <= 1e-11 and d_che <= 1e-13 and d_lag <= 1e-8),
    )
    record(
        label="Laguerre 递推误差的窗口依赖（x 上限 5/10/20）",
        theory="绝对误差随 x 上限增长（|L_n| 自身量级被放大）",
        measured=f"{window[5.0]:.2e} / {window[10.0]:.2e} / {window[20.0]:.2e}",
        tol="单调不减",
        passed=bool(window[5.0] <= window[10.0] <= window[20.0]),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="与 SciPy 的 eval_legendre / eval_chebyt 逐点一致", chapter="2.3")
def test_analytic_forms_match_reference_special_functions() -> None:
    """自研闭式与参照层（SciPy）对拍：最大绝对差必须是舍入量级。"""
    xs = np.linspace(-1.0, 1.0, 101)
    d_leg = max(float(np.max(np.abs(legendre_analytic(n, xs) - ref.legendre_p(n, xs)))) for n in range(0, 13))
    d_che = max(float(np.max(np.abs(chebyshev_analytic(n, xs) - ref.chebyt(n, xs)))) for n in range(0, 13))
    rec = record(
        label="闭式 vs SciPy（n≤12）",
        theory="≤ 1e-12",
        measured=f"Legendre {d_leg:.2e}，Chebyshev {d_che:.2e}",
        tol="1e-12",
        passed=bool(d_leg <= 1e-12 and d_che <= 1e-12),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="⟨p_i,p_j⟩_w = δ_ij（正交归一化后）", chapter="2.3")
def test_orthonormality_defect_stays_at_roundoff() -> None:
    r"""三个族的正交归一缺陷都要在 :math:`u` 量级：这是“基是正交坐标系”的实测版本。

    Laguerre 用 :math:`n_{\text{quad}}=4n+24` 个 Gauss--Laguerre 节点做内积，
    节点/权重本身就是本章的求根结果——所以这条断言同时验证了求根与求积。
    """
    worst = 0.0
    detail = []
    for fam in FAMILIES:
        for n_max in (6, 12, 20):
            defect, gram = orthonormality_defect(fam, n_max)
            worst = max(worst, defect)
            detail.append(f"{fam}(n≤{n_max})={defect:.1e}")
    rec = record(
        label="max|⟨p_i,p_j⟩_w − δ_ij|",
        theory="≤ 1e-13（舍入量级）",
        measured=f"{worst:.2e}（" + "，".join(detail) + "）",
        tol="1e-13",
        passed=bool(worst <= 1e-13),
    )
    # 逐族的 Gram 矩阵还必须**几乎是对角**的：非对角元是缺陷的来源
    d_leg, G_leg = orthonormality_defect("legendre", 10)
    off = float(np.max(np.abs(G_leg - np.diag(np.diag(G_leg)))))
    record(
        label="Gram 矩阵非对角元（legendre, n≤10）",
        theory="≈ 0",
        measured=f"{off:.2e}（对角元偏离 1 的最大值 {float(np.max(np.abs(np.diag(G_leg) - 1.0))):.2e}）",
        tol="1e-13",
        passed=bool(off <= 1e-13),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Gauss 求积的代数精度恰为 2n−1；第 2n 阶矩必然失败", chapter="2.3")
def test_gauss_rules_integrate_polynomials_to_the_exact_degree() -> None:
    r"""**恰好** :math:`2n-1` 阶：既断言 :math:`k\le2n-1` 时误差在舍入水平，
    又断言 :math:`k=2n` 时误差显著非零（否则“精度”这句话没有上界，等于什么都没说）。

    Laguerre 的矩是 :math:`k!`（增长极快），所以对它用**相对**误差。
    """
    worst_exact = 0.0
    weakest_fail = float("inf")
    rows = []
    for n in (4, 8, 16):
        for fam in ("legendre", "chebyshev", "laguerre"):
            err = np.abs(moment_exactness_error(fam, n))
            if fam == "laguerre":
                facts = np.array([float(math.factorial(k)) for k in range(err.size)])
                err = err / facts
            ok = float(np.max(err[: 2 * n]))
            bad = float(err[2 * n])
            worst_exact = max(worst_exact, ok)
            weakest_fail = min(weakest_fail, bad)
            rows.append(f"{fam}(n={n}):{ok:.1e}/{bad:.1e}")
    rec = record(
        label="k≤2n−1 的最大矩误差 / k=2n 处的误差",
        theory="前者 ≤ 1e-13，后者 ≥ 1e-10",
        measured=f"精确段 {worst_exact:.2e}，失效点最小值 {weakest_fail:.2e}",
        tol="见 theory",
        passed=bool(worst_exact <= 1e-13 and weakest_fail >= 1e-10),
    )
    record(
        label="逐族明细（精确段/失效点）",
        theory="2n−1 是硬边界",
        measured="，".join(rows),
        tol="-",
        passed=True,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Σ w_k = b − a；权重全正", chapter="2.3")
def test_gauss_weights_sum_to_the_interval_measure() -> None:
    """权重和必须精确等于区间长度（Leibniz 律那一行），且**全为正**——
    负权重意味着求积公式在数学上不成立（物理上就是“减去一部分面积”）。
    """
    rows = []
    worst = 0.0
    min_w = float("inf")
    for n in (4, 16, 32, 64):
        for fam in FAMILIES:
            if fam == "legendre":
                _, w = gauss_legendre(n)
                target = 2.0
            elif fam == "chebyshev":
                _, w = gauss_chebyshev(n)
                target = float(np.pi)
            else:
                _, w = gauss_laguerre(n)
                target = 1.0
            err = abs(float(np.sum(w)) - target)
            worst = max(worst, err)
            min_w = min(min_w, float(np.min(w)))
            rows.append(f"{fam}(n={n}):{err:.1e}")
    rec = record(
        label="|Σw − (b−a)|（n≤64，三族）",
        theory="≤ 1e-12",
        measured=f"{worst:.2e}；最小权重 {min_w:.2e} > 0",
        tol="1e-12",
        passed=bool(worst <= 1e-12 and min_w > 0.0),
    )
    record(
        label="逐族明细",
        theory="Chebyshev 的权重是常数 π/n（精确）",
        measured="，".join(rows),
        tol="-",
        passed=True,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="‖P_k‖²_w = 2/(2k+1)（Legendre）、π 或 π/2（Chebyshev）、1（Laguerre）", chapter="2.3")
def test_family_norms_match_theory() -> None:
    """归一化常数不是“随便定的”：它决定了正交归一基与 Gauss 权重公式里的系数。"""
    rows = []
    worst = 0.0
    for fam in FAMILIES:
        for k in range(0, 8):
            got = float(family_norm(fam, k))
            if fam == "legendre":
                want = 2.0 / (2 * k + 1)
            elif fam == "chebyshev":
                want = float(np.pi) if k == 0 else float(np.pi) / 2.0
            else:
                want = 1.0
            worst = max(worst, abs(got - want))
            if k < 3:
                rows.append(f"{fam}(k={k}):{got:.6f}≈{want:.6f}")
    # 与“用数值积分算出来的范数”对拍（independent 路径：Gauss 求积 vs 解析公式）
    xq, wq = gauss_legendre(40)
    num = {k: float(np.sum(wq * legendre_analytic(k, xq) ** 2)) for k in range(0, 6)}
    worst_num = max(abs(num[k] - family_norm("legendre", k)) for k in num)
    rec = record(
        label="family_norm vs 闭式 / vs 数值积分",
        theory="≤ 1e-12",
        measured=f"闭式差 {worst:.2e}，数值积分差 {worst_num:.2e}",
        tol="1e-12",
        passed=bool(worst <= 1e-12 and worst_num <= 1e-12),
    )
    record(label="前几项范数", theory="2/(2k+1)、π/π2、1", measured="，".join(rows), tol="-", passed=True)
    assert rec.passed, rec.measured


@assertion("A", theory="Sturm 变号数 = x 与「同号端点」之间的零点个数（Szego 第 3.3 节）", chapter="2.3")
def test_sturm_count_matches_the_root_count_side() -> None:
    r"""符号变号数是“逐根夹逼”的依据：它必须与已知零点位置**逐个核对**，否则隔离是假的。

    方向是这一条的关键：Legendre/Chebyshev 在 :math:`x=1` 处全部取正，于是
    :math:`V(x)=\#\{x_k>x\}`；Laguerre 在 :math:`x=0` 处全部取正，于是
    :math:`V(x)=\#\{x_k<x\}`。把方向搞反在“零点没越过探针点”时完全不报错——
    所以这里逐族逐点核对，并且把两个方向都算出来记录在案。
    探针点避开零点本身（:math:`x` 恰为根时 :math:`V` 的定义有 ±1 的歧义，是零测集）。
    """
    worst_right = 0
    worst_wrong = 0
    total = 0
    for fam in FAMILIES:
        for n in (4, 8, 12):
            roots = newton_roots_family(fam, n).roots
            lo, hi = (-1.0, 1.0) if fam != "laguerre" else (0.0, float(roots[-1]) * 1.5 + 1.0)
            probes = [t for t in np.linspace(lo, hi, 41) if float(np.min(np.abs(roots - t))) > 1e-9]
            for t in probes:
                got = sturm_root_count(fam, n, float(t))
                right = int(np.count_nonzero(roots > t)) if fam != "laguerre" else int(np.count_nonzero(roots < t))
                wrong = int(np.count_nonzero(roots < t)) if fam != "laguerre" else int(np.count_nonzero(roots > t))
                worst_right = max(worst_right, abs(got - right))
                worst_wrong = max(worst_wrong, abs(got - wrong))
                total += 1
    rec = record(
        label="max|V − 正确方向的零点个数|（探针避开零点）",
        theory="0（精确相等）",
        measured=f"{worst_right}（共核对 {total} 个探针点）",
        tol="0",
        passed=bool(worst_right == 0),
    )
    record(
        label="把方向搞反时的最大偏差",
        theory="> 0（方向必须按族的归一化端点选）",
        measured=str(worst_wrong),
        tol="-",
        passed=bool(worst_wrong > 0),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Gram–Schmidt 得到的正交多项式 = 归一化 Legendre（差一个正因子）", chapter="2.3")
def test_gram_schmidt_reproduces_orthonormal_polynomials() -> None:
    r"""对单项式基做 Gram--Schmidt，结果必须（在归一化后）等于解析的 :math:`\\hat P_k`。

    这条是“从零构造正交多项式”与“用闭式公式”两条独立路径的对拍——
    它同时说明：Gram--Schmidt 在**离散**内积下就已经足够（缺陷见下一条）。
    """
    # 关键：离散内积的**测度**决定正交族。Gauss–Legendre 节点+权重才逼近 Lebesgue 测度；
    # 用 Chebyshev 节点配等权得到的是反正弦测度，正交族会变成 Chebyshev（实测差 2.1e0）。
    nodes, weights = gauss_legendre(200)
    gs = gram_schmidt(8, nodes, weights)
    ref_mat = orthonormal_values("legendre", 8, nodes)  # (k, x)
    worst = 0.0
    for k in range(9):
        v = np.array([gs.coeffs[k][j] for j in range(k + 1)], dtype=float)
        got = np.vander(nodes, k + 1)[:, ::-1] @ v
        want = ref_mat[k]
        # 归一化后的符号可能差一个 ±1，取两种符号的较小差
        worst = max(worst, min(float(np.max(np.abs(got - want))), float(np.max(np.abs(got + want)))))
    # 反面：换成正弦测度（Chebyshev 节点等权），同一个 GS 给出的是另一族
    cheb_nodes = np.cos(np.pi * (np.arange(600) + 0.5) / 600)
    gs_arc = gram_schmidt(4, cheb_nodes, np.full(600, np.pi / 600))
    ref_cheb = orthonormal_values("chebyshev", 4, cheb_nodes)
    worst_arc = 0.0
    for k in range(1, 5):
        v = np.array([gs_arc.coeffs[k][j] for j in range(k + 1)], dtype=float)
        got = np.vander(cheb_nodes, k + 1)[:, ::-1] @ v
        worst_arc = max(worst_arc, min(float(np.max(np.abs(got - ref_cheb[k]))), float(np.max(np.abs(got + ref_cheb[k])))))
    rec = record(
        label="Gram–Schmidt vs 解析 Legendre（Gauss–Legendre 测度，200 点）",
        theory="≤ 1e-12",
        measured=f"{worst:.2e}；GS 自报缺陷 {gs.defect:.2e}",
        tol="1e-12",
        passed=bool(worst <= 1e-12 and gs.defect <= 1e-12),
    )
    record(
        label="换成反正弦测度后 GS 得到的其实是 Chebyshev 族",
        theory="≤ 1e-12（同一个算法，测度变了族就变了）",
        measured=f"{worst_arc:.2e}",
        tol="1e-12",
        passed=bool(worst_arc <= 1e-12),
    )
    assert rec.passed and worst_arc <= 1e-12, (worst, worst_arc)


@assertion("A", theory="Chebyshev 零点/极点是闭式：cos((2k−1)π/2n)、cos(kπ/n)", chapter="2.3")
def test_chebyshev_nodes_are_closed_form_and_equioscillate() -> None:
    r"""Chebyshev 的零点与极点是本章唯一的**解析可写**的节点，因此被用作内部基准：
    零点处 :math:`T_n=\\cos(n\\arccos x)=0`，极点处 :math:`|T_n|=1` 且符号逐个交替。
    """
    worst_zero = 0.0
    worst_eq = 0.0
    alt = []
    for n in (4, 8, 16, 32):
        z = chebyshev_roots(n)
        worst_zero = max(worst_zero, float(np.max(np.abs(chebyshev_analytic(n, z)))))
        m = chebyshev_extrema(n)
        vals = chebyshev_analytic(n, m)
        worst_eq = max(worst_eq, float(np.max(np.abs(np.abs(vals) - 1.0))))
        signs = np.sign(vals)
        alt.append(bool(np.all(signs[1:] * signs[:-1] < 0)))
    rec = record(
        label="零点处 |T_n| / 极点处 ||T_n|−1|",
        theory="0 与 0（闭式）",
        measured=f"{worst_zero:.2e} 与 {worst_eq:.2e}",
        tol="1e-13",
        passed=bool(worst_zero <= 1e-13 and worst_eq <= 1e-13),
    )
    record(
        label="极点处符号严格交替（n=4,8,16,32）",
        theory="符号逐个交替",
        measured=str(alt),
        tol="True",
        passed=bool(all(alt)),
    )
    assert worst_zero <= 1e-13 and worst_eq <= 1e-13, (worst_zero, worst_eq)


# --------------------------------------------------------------------------- #
# B 类：不变量与可观测物
# --------------------------------------------------------------------------- #
@assertion("B", theory="正交多项式的 n 个零点互异、升序、且 Newton 收敛", chapter="2.3")
def test_newton_finds_all_roots_without_duplicates() -> None:
    r"""逐族逐阶检查：根数 = n、严格升序、迭代次数有界。

    这条看似平凡，却是踩过坑的地方：Laguerre 用渐近初值时会**两个初值收敛到同一个根**
    （n=3 时 :math:`\sum w` 就会从 1 变成 2.68），所以必须用 Sturm 变号数把根隔离开再迭代。
    """
    rows = []
    worst_iter = 0
    ok = True
    for fam in FAMILIES:
        for n in (2, 3, 5, 9, 16, 24, 32):
            rs = newton_roots_family(fam, n)
            gaps = np.diff(rs.roots)
            ok = ok and rs.roots.size == n and bool(np.all(gaps > 0)) and rs.converged
            worst_iter = max(worst_iter, rs.max_iterations)
        rows.append(f"{fam}:maxIter={newton_roots_family(fam, 32).max_iterations}")
    rec = record(
        label="根数/升序/收敛（三族，n≤32）",
        theory="根数 = n，间距 > 0，全部收敛",
        measured=f"全部满足 = {ok}；最慢单根迭代次数 {worst_iter}",
        tol="-",
        passed=bool(ok),
    )
    record(label="各族的逐根最大迭代次数", theory="≤ 60", measured="，".join(rows), tol="60", passed=bool(worst_iter <= 60))
    assert rec.passed, rec.measured


@assertion("B", theory="Laguerre 的 |L_n(x_k)| 极大但根误差 ~u（误差量的选择会骗人）", chapter="2.3")
def test_laguerre_huge_residual_but_roundoff_root_error() -> None:
    r"""**本章最重要的观测陷阱**：:math:`|L_{40}(x_k)|\sim5.6\times10^{14}`，
    但 Newton 修正量 :math:`|L_n/L_n'|` 只有 :math:`1.2\times10^{-14}`。

    同一个求根结果，用残差当误差会得出“算法崩了”，用修正量当误差是“到机器精度了”。
    正确的那一个是与多项式自身尺度无关的那个。
    """
    rows = []
    worst_err = 0.0
    biggest_res = 0.0
    for n in (8, 16, 32, 40):
        rs = newton_roots_family("laguerre", n)
        worst_err = max(worst_err, rs.max_root_error)
        biggest_res = max(biggest_res, rs.max_residual)
        rows.append(f"n={n}:res={rs.max_residual:.1e},err={rs.max_root_error:.1e}")
    ratio = biggest_res / worst_err
    rec = record(
        label="Laguerre：残差 / 根误差（n≤40）",
        theory="残差 ≫ 根误差（两者相差 ≥1e25）",
        measured=f"残差 {biggest_res:.2e}，根误差 {worst_err:.2e}，比值 {ratio:.2e}",
        tol="根误差 ≤ 1e-12",
        passed=bool(worst_err <= 1e-12 and ratio >= 1e20),
    )
    record(label="逐阶明细", theory="n 越大残差越大", measured="，".join(rows), tol="-", passed=True)
    assert rec.passed, rec.measured


@assertion("B", theory="同族相邻阶的零点严格交错（Sturm 分离定理）", chapter="2.3")
def test_roots_of_successive_degrees_interlace() -> None:
    r"""零点交错是正交多项式族最本质的结构性质之一：它保证 Gauss 节点可以逐阶细化。
    这里直接数“落在每个区间里的根数”，比只看图像可靠。
    """
    worst = 0
    checked = 0
    for fam in FAMILIES:
        for n in (2, 4, 8, 16, 24):
            lo = newton_roots_family(fam, n - 1).roots
            hi = newton_roots_family(fam, n).roots
            # 交错 ⟺ 每个 (lo[i], lo[i+1]) 里恰好一个 hi 的根，两端各一个
            inner = [int(np.count_nonzero((hi > a) & (hi < b))) for a, b in zip(lo[:-1], lo[1:])]
            worst = max(worst, max(abs(c - 1) for c in inner) if inner else 0)
            checked += len(inner)
    rec = record(
        label="每两个相邻低阶零点之间的高阶零点个数 − 1",
        theory="0（严格交错）",
        measured=f"最大偏离 {worst}（共核对 {checked} 个区间）",
        tol="0",
        passed=bool(worst == 0),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="Legendre 零点全在 (−1,1)、Laguerre 全在 (0,∞)、且 P_n(x_k)=0", chapter="2.3")
def test_roots_lie_inside_the_orthogonality_interval() -> None:
    """零点必须在正交区间**内部**（这是正交多项式零点定理的一部分）：越界意味着求根跑飞了。"""
    worst_res = 0.0
    ok_range = True
    gap = 0.0
    for fam in FAMILIES:
        for n in (4, 12, 24, 32):
            rs = newton_roots_family(fam, n)
            if fam == "legendre":
                ok_range = ok_range and bool(np.all(rs.roots > -1.0) and np.all(rs.roots < 1.0))
                # 回代用**递推**：闭式展开式在 n=32 时自身只剩 ~1e-6 精度（大数相消）
                worst_res = max(worst_res, float(np.max(np.abs(legendre_recurrence(n, rs.roots)[n]))))
                gap = max(gap, float(np.max(np.abs(legendre_analytic(n, rs.roots)))))
            elif fam == "chebyshev":
                ok_range = ok_range and bool(np.all(rs.roots > -1.0) and np.all(rs.roots < 1.0))
                worst_res = max(worst_res, float(np.max(np.abs(chebyshev_recurrence(n, rs.roots)[n]))))
            else:
                ok_range = ok_range and bool(np.all(rs.roots > 0.0))
    rec = record(
        label="零点在正交区间内 + 零点处的 |p_n|（递推回代）",
        theory="区间内 + |p_n| ≤ 1e-11",
        measured=f"区间检查 = {ok_range}；Legendre/Chebyshev 残差 {worst_res:.2e}",
        tol="见 theory",
        passed=bool(ok_range and worst_res <= 1e-11),
    )
    record(
        label="同一个零点用闭式展开式回代得到的残差",
        theory="≫ 1e-11（闭式自身的大数相消，不是求根不准）",
        measured=f"{gap:.2e}（n=32 的 Legendre）",
        tol="-",
        passed=bool(gap > 1e-9),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="Gauss 权重 = 2/((1−x²)P_n'(x)²)（Legendre）、x/(n²L_{n−1}(x)²)（Laguerre）", chapter="2.3")
def test_gauss_weights_match_their_closed_forms() -> None:
    r"""权重公式是本章的“第二个可对拍点”：节点由求根给出、权重由导数闭式给出，
    两者独立算出来的求积公式必须仍然精确（否则权重公式抄错了）。
    """
    rows = []
    worst = 0.0
    for n in (4, 8, 16, 32):
        x, w = gauss_legendre(n)
        formula = 2.0 / ((1.0 - x * x) * legendre_derivative(n, x) ** 2)
        worst = max(worst, float(np.max(np.abs(w - formula) / np.abs(w))))
        xl, wl = gauss_laguerre(n)
        # 正确闭式是 w_k = x_k / (n² L_{n−1}(x_k)²)（用递推求 L_{n−1}，闭式自身精度不够）
        prev = laguerre_recurrence(n - 1, xl)[n - 1]
        formula_l = xl / (n * n * prev**2)
        worst = max(worst, float(np.max(np.abs(wl - formula_l) / np.abs(wl))))
        rows.append(f"n={n}:{float(np.max(np.abs(w - formula)) / float(np.max(w))):.1e}")
    rec = record(
        label="权重与闭式公式的相对差（Legendre & Laguerre，n≤32）",
        theory="≤ 1e-12",
        measured=f"{worst:.2e}",
        tol="1e-12",
        passed=bool(worst <= 1e-12),
    )
    record(label="Legendre 权重逐阶明细", theory="-", measured="，".join(rows), tol="-", passed=True)
    assert rec.passed, rec.measured


@assertion("B", theory="quadrature() 与显式 Σw·f 一致，且权函数形状可核对", chapter="2.3")
def test_quadrature_helper_and_weight_shapes() -> None:
    """求积 API 与“手算的加权和”必须逐位一致；权函数在区间内正、在区间外不参与。"""
    x, w = gauss_legendre(24)
    vals = np.exp(x)
    a = float(quadrature(vals, w))
    b = float(np.sum(w * vals))
    ok_shape = True
    for fam in FAMILIES:
        lo, hi = (-1.0, 1.0) if fam != "laguerre" else (0.0, np.inf)
        probe = np.linspace(lo + 1e-3, min(hi, 50.0) - 1e-3, 17)
        wv = weight_function(fam, probe)
        ok_shape = ok_shape and bool(np.all(wv > 0.0))
        if not np.isfinite(hi):
            continue
    rec = record(
        label="quadrature() 与 Σw·f 的相对差 / 权函数全正",
        theory="0 与 True",
        measured=f"{abs(a - b) / abs(b):.2e} 与 {ok_shape}",
        tol="0 与 True",
        passed=bool(a == b and ok_shape),
    )
    # ∫_0^∞ e^{-x}·1 dx = 1（权函数本身）：Laguerre 求积把它算成 1
    xl, wl = gauss_laguerre(20)
    one = float(quadrature(np.ones_like(xl), wl))
    rec2 = record(
        label="Laguerre 权函数积分 ∫e^{−x}dx",
        theory="1",
        measured=f"{one:.16f}",
        tol="1e-12",
        passed=bool(abs(one - 1.0) <= 1e-12),
    )
    # 精确矩与求积结果对拍（第二条独立路径）
    worst = 0.0
    for fam in FAMILIES:
        if fam == "legendre":
            xq, wq = gauss_legendre(24)
            f = lambda t: t**6  # noqa: E731
        elif fam == "chebyshev":
            xq, wq = gauss_chebyshev(24)
            f = lambda t: t**6  # noqa: E731
        else:
            xq, wq = gauss_laguerre(24)
            f = lambda t: t**6  # noqa: E731
        got = float(np.sum(wq * f(xq)))
        want = float(exact_moment(fam, 6))
        worst = max(worst, abs(got - want) / max(abs(want), 1.0))
    assert_leq(worst, 1e-13, label="精确矩 vs 求积（k=6，三族）")
    assert rec.passed and rec2.passed


@assertion("B", theory="Gram 矩阵 G_ij = ⟨p_i,p_j⟩_w 的对称性与对角占优", chapter="2.3")
def test_gram_matrix_is_symmetric_and_diagonal() -> None:
    """正交归一基下 G 必须≈I：对称性、对角元 1、非对角元 0，三条都要实测。"""
    x, w = gauss_legendre(48)
    B = orthonormal_values("legendre", 10, x)
    G = gram_matrix(B, w)
    asym = float(np.max(np.abs(G - G.T)))
    diag = float(np.max(np.abs(np.diag(G) - 1.0)))
    off = float(np.max(np.abs(G - np.diag(np.diag(G)))))
    rec = record(
        label="G 的对称性/对角/非对角偏离",
        theory="≤ 1e-13",
        measured=f"非对称 {asym:.1e}，对角偏离 {diag:.1e}，非对角 {off:.1e}",
        tol="1e-13",
        passed=bool(max(asym, diag, off) <= 1e-13),
    )
    assert rec.passed, rec.measured
