"""2.1 章重心公式 / Vandermonde 直接解 / 帧契约的 A/B 断言。

这一章有两条**看起来同义、其实完全不同**的"误差"：

* **函数值误差** :math:`\\max_x|p(x)-f(x)|`——重心公式与 Vandermonde 直接解都是
  :math:`\\approx u\\Lambda_n`，两条通道几乎一样好；
* **系数误差** :math:`\\|\\hat c-c\\|`——只有 Vandermonde 直接解有这一项，它按
  :math:`\\kappa(V)u` 爆炸（等距 n=30 时 2.69e-4，n=36 时 1.19）。

把"系数误差大"读成"公式不稳定"是本 Viz 最想拦住的那个错误结论：
同一个解的**后向残差**从 n=6 到 n=30 一直是 1e-17 量级，与 :math:`\\kappa` 无关。
"""

from __future__ import annotations

import importlib.util
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from numcore.interp.poly import (
    NODE_KINDS,
    UNIT_ROUNDOFF,
    barycentric_interpolate,
    barycentric_partition_of_unity,
    barycentric_weight_identity,
    barycentric_weights,
    barycentric_weights_chebyshev,
    chebyshev_nodes,
    equispaced_nodes,
    interpolation_method_table,
    lagrange_basis,
    lebesgue_constant,
    lebesgue_function,
    node_family,
    vandermonde_condition,
    vandermonde_matrix,
    vandermonde_solve,
)
from numcore.linalg.lu import exact_solve, lu_decompose, solve_lu
from numcore.verify import assertion, record

U = UNIT_ROUNDOFF
ROOT = Path(__file__).resolve().parents[2]


def _bareiss_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""分数-free（Bareiss）整数消元的**独立**精确解——用来对拍 ``exact_solve``。

    与 :func:`numcore.linalg.lu.exact_solve` 的算法完全不同：后者是"有理数高斯消元"，
    这里先把每一行通分成整数（行缩放不改变解），再用 Bareiss 恒等式
    :math:`a_{ij}^{(k)}=(a_{ij}^{(k-1)}a_{kk}^{(k-1)}-a_{ik}^{(k-1)}a_{kj}^{(k-1)})/a_{k-1,k-1}^{(k-1)}`
    做**整数**消元，最后回代。两条独立路径给出同一组有理数，才说明"真值"不是自证。
    """
    n = A.shape[0]
    rows: list[list[Fraction]] = []
    for i in range(n):
        row = [Fraction(float(A[i, j])) for j in range(n)] + [Fraction(float(b[i]))]
        den = 1
        for v in row:
            den = den * v.denominator // _gcd(den, v.denominator)
        rows.append([v * den for v in row])  # 行缩放：解不变，且每个元素变成整数
    prev = Fraction(1)
    for k in range(n - 1):
        if rows[k][k] == 0:
            piv = next((i for i in range(k + 1, n) if rows[i][k] != 0), None)
            if piv is None:
                raise ZeroDivisionError("奇异")
            rows[k], rows[piv] = rows[piv], rows[k]
        for i in range(k + 1, n):
            for j in range(k + 1, n + 1):
                rows[i][j] = (rows[i][j] * rows[k][k] - rows[i][k] * rows[k][j]) / prev
            rows[i][k] = Fraction(0)
        prev = rows[k][k]
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        acc = rows[i][n]
        for j in range(i + 1, n):
            acc -= rows[i][j] * x[j]
        x[i] = acc / rows[i][i]
    return np.array([float(v) for v in x], dtype=float)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _load_chapter():
    spec = importlib.util.spec_from_file_location("ch21_mod", ROOT / "server" / "chapters" / "2.1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# A 类：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="乘积形式权重的恒等式：(Σ w_i/(x−x_i))·∏(x−x_j) = 1", chapter="2.1")
def test_weight_identity_and_the_naive_form_is_false() -> None:
    r"""任务书里常见的写法 :math:`\sum_i w_i/(x-x_i)=1` 是**错的**，这里把它量化。

    未归一化的乘积形式权重满足 :math:`\bigl(\sum_i\frac{w_i}{x-x_i}\bigr)\prod_j(x-x_j)=1`；
    直接对 :math:`\sum_iw_i/(x-x_i)` 取绝对值，量到的是 :math:`1/|\prod_j(x-x_j)|`——
    实测 n=6 时最小 33.1、n=12 时 2.08e3、n=20 时 5.26e5，随 n 指数增长，
    **离 1 越来越远**。两条都要测：正确的那条到 1e-14，错误的那条至少差 1 个数量级。
    """
    worst_ratio = 0.0
    naive_min = []
    detail = []
    for kind in NODE_KINDS:
        for n in (6, 12, 20):
            nd = node_family(kind, n)
            xq = np.linspace(-1.0, 1.0, 201)
            keep = xq[np.min(np.abs(xq[:, None] - nd[None, :]), axis=1) > 1e-12]
            ident = barycentric_weight_identity(nd, keep)
            lam = lebesgue_constant(nd, -1.0, 1.0, n_grid=2001, refine_rounds=3)
            # 恒等式的绝对残差随 1/∏(x−x_j) 放大，归一化因子正是 u·Λ_n（与全章同一把尺子）
            worst_ratio = max(worst_ratio, float(np.max(np.abs(ident - 1.0))) / (U * lam))
            w = barycentric_weights(nd)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                naive = np.abs(np.sum(w[None, :] / (keep[:, None] - nd[None, :]), axis=1))
            naive_min.append(float(np.min(naive)))
            detail.append(f"{kind[:4]} n={n}: 比值 {float(np.max(np.abs(ident - 1.0))) / (U * lam):.2f}")
    rec = record(
        label="恒等式残差/(u·Λ_n) / 朴素 Σ w/(x−xi) 的最小值",
        theory="正确恒等式含 ∏(x−x_j) 因子（残差 ≈ u·Λ_n）；朴素和 = 1/∏(x−x_j) 而不是 1",
        measured=f"归一化残差 ≤ {worst_ratio:.2f}（{'；'.join(detail)}）；朴素和最小 {min(naive_min):.1f}"
        f"（n=6/12/20 分别为 {naive_min[0]:.1f}/{naive_min[2]:.1f}/{naive_min[4]:.1f}）",
        tol="归一化残差 ≤ 10；朴素和 ≥ 10（即「绝不等于 1」）",
        passed=worst_ratio <= 10.0 and min(naive_min) >= 10.0,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="重心单位分解 Σ_i ℓ_i(x) = 1；Lebesgue 函数 Σ|ℓ_i| ≥ 1", chapter="2.1")
def test_partition_of_unity_is_tight_while_lebesgue_is_not() -> None:
    r"""两条式子差别只在绝对值的**位置**，数值上一个到 1、一个到 1e4。

    实测 :math:`\max_x|\sum_i\ell_i-1|`：Chebyshev n=20 是 6.7e-16、等距 n=20 是 3.1e-13
    （放大倍数正是 :math:`\Lambda_n`）；同一批节点上 :math:`\Lambda_{20}`（等距）是 1.07e4。
    写混这两条（先取绝对值再除以绝对值和）会得到一个"看起来像 2 的数"——
    那个数其实是 :math:`\Lambda_n(x)`，已经被 chapter 2.1 的实现修掉了。
    """
    worst_unity = 0.0
    lam_pairs = []
    for kind in NODE_KINDS:
        for n in (6, 12, 20):
            nd = node_family(kind, n)
            xq = np.linspace(-1.0, 1.0, 201)
            keep = xq[np.min(np.abs(xq[:, None] - nd[None, :]), axis=1) > 1e-12]
            worst_unity = max(worst_unity, float(np.max(np.abs(barycentric_partition_of_unity(nd, keep) - 1.0))))
            lam_pairs.append(float(np.max(lebesgue_function(nd, keep))))
    rec = record(
        label="max|Σℓ−1| / 同批节点的 max Λ_n",
        theory="Σℓ ≡ 1（舍入级）；Λ_n = Σ|ℓ| 可以任意大——绝对值的位置决定一切",
        measured=f"单位分解偏差 ≤ {worst_unity:.2e}；Λ 最大 {max(lam_pairs):.1f}"
        f"（等距 n=20 为 {lam_pairs[5]:.1f}）",
        tol="单位分解 ≤ 1e-12；Λ 最大 ≥ 10",
        passed=worst_unity <= 1e-12 and max(lam_pairs) >= 10.0,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Chebyshev–Lobatto 权重闭式 w_j ∝ (−1)^j·δ_j（δ=1/2 在端点）", chapter="2.1")
def test_chebyshev_lobatto_weights_match_product_form() -> None:
    r"""闭式权重与乘积形式权重必须只差一个**整体常数**（重心公式对整体缩放不敏感）。

    实测比值：n=6 全为 5.3333333333、n=12 为 170.6666666667、n=20 为 26214.4（相对散布 < 1e-13）。
    这同时说明两件事：闭式实现正确；"权重可以任意缩放"这条不变量的实测含义。
    """
    rows = []
    ok = True
    for n in (6, 12, 20):
        nd = chebyshev_nodes(n)
        w_prod = barycentric_weights(nd)
        w_closed = barycentric_weights_chebyshev(n)
        ratio = w_prod / w_closed
        spread = float(np.max(np.abs(ratio - np.mean(ratio))) / abs(np.mean(ratio)))
        rows.append(f"n={n}: 比值 {np.mean(ratio):.6f}，散布 {spread:.1e}")
        ok = ok and spread <= 1e-12
    rec = record(
        label="乘积形式权重 / 闭式权重 的常数性与散布",
        theory="w_j = (−1)^j δ_j 与 ∏ 形式只差整体常数（缩放不改变重心插值）",
        measured="；".join(rows),
        tol="相对散布 ≤ 1e-12",
        passed=ok,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="有理数精确解（高斯消元）与 Bareiss 整数消元给出同一组系数", chapter="2.1")
def test_exact_reference_is_confirmed_by_bareiss_elimination() -> None:
    r"""Viz 4 的"真值"不能靠一个实现自证：两种**不同算法**的精确解必须逐位一致。

    实测 n=6/10/12（两种节点分布）：:func:`exact_solve` 与 Bareiss 整数消元的相对偏差
    全部为 **0.0**（同一个有理数转成同一个 double）。
    顺便说明为什么不用 ``mpmath`` 的 ``lu_solve``：同一个 n=12 等距 Vandermonde 上
    它给出与有理真值相差 1.66e-13 的结果，且把 ``dps`` 从 60 提到 100 也不改善——
    病态系统的"高精度"未必是高精度。
    """
    worst = 0.0
    detail = []
    for kind in NODE_KINDS:
        for n in (6, 10, 12):
            nd = node_family(kind, n)
            vals = np.exp(nd)
            V = vandermonde_matrix(nd)
            c_ref = exact_solve(V, vals)
            c_bareiss = _bareiss_solve(V, vals)
            rel = float(np.max(np.abs(c_ref - c_bareiss))) / float(np.max(np.abs(c_ref)))
            worst = max(worst, rel)
            detail.append(f"{kind[:4]} n={n}: {rel:.1e}(κ={vandermonde_condition(nd):.1e})")
    rec = record(
        label="exact_solve vs Bareiss 的相对偏差（6 组）",
        theory="两种独立的精确有理算法必须给出同一组系数",
        measured=f"最大 {worst:.1e}；" + "；".join(detail),
        tol="≤ 1e-16（同一有理数的两次浮点转换）",
        passed=worst <= 1e-16,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="κ(V) 的增长率：Chebyshev 按 (1+√2)^n，等距更快", chapter="2.1")
def test_vandermonde_condition_growth_rates() -> None:
    r"""两个增长率都要量出来：Chebyshev 的 :math:`\log_2\kappa/n` 稳定在 1.26，
    与理论 :math:`\log_2(1+\sqrt2)=1.2716` 差 <1%；等距则在 1.53→1.60 之间漂移（底数更大）。

    实测 n=30：等距 κ=2.59e14、Chebyshev κ=2.68e11（差 966 倍）；n=44 时
    2.0e21 vs 6.2e16。这条断言是 Viz 4 里那条 κ 参考线的依据。
    """
    rows = []
    ok = True
    for kind in NODE_KINDS:
        per_n = []
        for n in (12, 20, 30, 40):
            nd = node_family(kind, n)
            kappa = vandermonde_condition(nd)
            per_n.append(float(np.log2(kappa) / n))
        if kind == "chebyshev":
            th = float(np.log2(1.0 + np.sqrt(2.0)))
            dev = max(abs(v / th - 1.0) for v in per_n)
            rows.append(f"Chebyshev log2κ/n {min(per_n):.4f}–{max(per_n):.4f}（理论 {th:.4f}，偏差 {dev:.1e}）")
            ok = ok and dev <= 0.02
        else:
            rows.append(f"等距 log2κ/n {min(per_n):.4f}–{max(per_n):.4f}（漂移，底数 > 1+√2）")
            ok = ok and per_n[0] >= 1.4 and per_n[-1] <= 1.7
    ratio = vandermonde_condition(equispaced_nodes(30)) / vandermonde_condition(chebyshev_nodes(30))
    rec = record(
        label="log2 κ(V)/n 与 Chebyshev 理论底数 / n=30 的 κ 之比",
        theory="Chebyshev 节点 κ ~ (1+√2)^n；等距节点更快；同 n 下 Chebyshev 小几个数量级",
        measured="；".join(rows) + f"；κ_eq/κ_ch(n=30)={ratio:.1f}",
        tol="Chebyshev 偏差 ≤ 2%；等距 ∈ [1.4, 1.7]；比值 ≥ 100",
        passed=ok and ratio >= 100.0,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="三条误差通道：函数值 ~ uΛ_n、系数 ~ κ(V)u、残差 ~ u", chapter="2.1")
def test_three_error_channels_of_the_vandermonde_solve() -> None:
    r"""Viz 4 的核心定量结论（每条都实测）：

    函数值误差：等距 n=36 时重心 1.05e-7、Vandermonde 3.52e-8（**同一量级**）；
    系数误差：同一点 1.19（相对 κu=28.6 比值 1e-2 量级）；后向残差：1.2e-17–3.2e-17。
    所以"直接解 Vandermonde 不稳定"这句话，只有在**系数的基**里才成立。
    """
    rows = []
    ok = True
    for kind in NODE_KINDS:
        tab = interpolation_method_table(36, kind=kind, function="exp", n_grid=801)
        i = 35
        eb = float(tab["error_barycentric"][i])
        ev = float(tab["error_vandermonde"][i])
        ec = float(tab["error_coefficients_vandermonde"][i])
        kb = float(tab["kappa_bound_vandermonde"][i])
        rs = float(tab["residual_vandermonde"][i])
        rows.append(
            f"{kind[:4]} n=36: 函数值 {eb:.2e}/{ev:.2e}（比 {max(eb, ev) / min(eb, ev):.2f}×），"
            f"系数 {ec:.2e} ≤ κu={kb:.2e}，残差 {rs:.1e}"
        )
        ok = ok and 0.1 <= ev / eb <= 10.0 and ec <= kb and rs <= 100.0 * U
    rec = record(
        label="三条误差通道（n=36，两种分布）",
        theory="函数值误差两条通道同量级；系数误差被 κu 界住；残差 ≈ u 与 κ 无关",
        measured="；".join(rows),
        tol="函数值比值 ∈ [0.1, 10]；系数 ≤ κu；残差 ≤ 100u",
        passed=ok,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# B 类：不变量与帧契约
# --------------------------------------------------------------------------- #
@assertion("B", theory="ℓ_i(x_j) = δ_ij，故 p(x_i) = y_i，且 Σ_i w_i = 0", chapter="2.1")
def test_barycentric_basis_and_weights_invariants() -> None:
    r"""三条形状不变量：基函数在节点上是 Kronecker δ、**权重和 = 0**、插值在节点上精确。

    第二条容易记反：乘积形式权重 :math:`w_i=1/\prod_{j\ne i}(x_i-x_j)` 的和恒为 **0**
    （实测两种分布、n=5..12 全部是 0 到 1e-16），不是 1——"和为 1"说的是**归一化以后的**
    :math:`\ell_i`（见 :func:`barycentric_partition_of_unity`）。
    """
    nd = node_family("chebyshev", 11)
    basis = np.array([lagrange_basis(nd, float(x)) for x in nd])
    delta = np.abs(basis - np.eye(nd.size))
    w = barycentric_weights(nd)
    vals = np.cos(nd) + 0.25 * nd**3
    at_nodes = barycentric_interpolate(nd, vals, nd)
    w_eq = barycentric_weights(equispaced_nodes(12))
    # 权重本身的量级可以很大（1/∏ 小量），所以"和 = 0"只能按相对口径检验：
    # 消去后的残差应当只有被加数规模的一个 u 倍。
    rel_ch = abs(float(np.sum(w))) / float(np.sum(np.abs(w)))
    rel_eq = abs(float(np.sum(w_eq))) / float(np.sum(np.abs(w_eq)))
    rec = record(
        label="|ℓ_i(x_j)−δ_ij| / |Σw|/Σ|w| / |p(x_i)−y_i|",
        theory="Lagrange 基在节点上是单位阵；乘积形式权重之和恒为 0（相对残差 ≈ u）；插值在节点上取给定值",
        measured=f"基 {float(np.max(delta)):.2e}；|Σw|/Σ|w| {rel_ch:.2e}（等距 n=12 时 {rel_eq:.2e}，"
        f"Σ|w|={float(np.sum(np.abs(w))):.3e}）；节点值 {float(np.max(np.abs(at_nodes - vals))):.2e}",
        tol="基与节点值 ≤ 1e-13；权重和相对残差 ≤ 1e-14",
        passed=(
            float(np.max(delta)) <= 1e-13
            and rel_ch <= 1e-14
            and rel_eq <= 1e-14
            and float(np.max(np.abs(at_nodes - vals))) <= 1e-13
        ),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="重心插值对权重的整体缩放与节点置换都不变", chapter="2.1")
def test_weight_scaling_and_node_permutation_invariance() -> None:
    """变换不变量：权重乘 1e7、节点整体打乱顺序，插值结果都不能变（差 ≪ uΛ）。"""
    nd = equispaced_nodes(12)
    vals = np.exp(nd)
    xs = np.linspace(-1.0, 1.0, 401)
    p0 = barycentric_interpolate(nd, vals, xs)
    p_scaled = barycentric_interpolate(nd, vals, xs, weights=1e7 * barycentric_weights(nd))
    perm = np.array([7, 0, 11, 3, 9, 1, 6, 12, 2, 10, 4, 8, 5])
    assert perm.size == nd.size
    p_perm = barycentric_interpolate(nd[perm], vals[perm], xs)
    lam = float(np.max(lebesgue_function(nd, np.linspace(-1.0, 1.0, 401))))
    scale = U * lam
    r1 = float(np.max(np.abs(p_scaled - p0))) / scale
    r2 = float(np.max(np.abs(p_perm - p0))) / scale
    rec = record(
        label="缩放不变性 / 置换不变性（以 u·Λ_n 归一）",
        theory="重心公式分子分母同阶齐次；插值多项式与节点编号无关",
        measured=f"缩放 {r1:.2f}uΛ、置换 {r2:.2f}uΛ（Λ={lam:.1f}）",
        tol="两者 ≤ 50uΛ",
        passed=r1 <= 50.0 and r2 <= 50.0,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="barycentric_vs_vandermonde 帧：三条通道与 κ 参考线都在帧里", chapter="2.1")
def test_vandermonde_frames_obey_invariants() -> None:
    """直接检查帧（可视化与断言不能各说各话）：沿时间轴 n 增长时 κ 不降、残差恒在 u 量级。"""
    mod = _load_chapter()
    prev_kappa = 0.0
    worst_resid = 0.0
    ok = True
    rows = []
    for t in (6.0, 12.0, 20.0, 30.0):
        fr = mod.compute("barycentric_vs_vandermonde", {"distribution": "equispaced", "n_max": 30}, t)
        kappa = float(np.max(fr["kappa_vandermonde"][: int(fr.meta["n_mark"])]))
        resid = float(np.max(fr["residual_vandermonde"][: int(fr.meta["n_mark"])]))
        prev_kappa = max(prev_kappa, kappa)
        worst_resid = max(worst_resid, resid)
        ok = ok and kappa >= prev_kappa * 0.999
        rows.append(f"t={t:.0f}: κ={kappa:.2e}, 残差={resid:.1e}")
    # 三条通道都在帧里，且系数误差通道确实被单列出来
    names = set(mod.compute("barycentric_vs_vandermonde", {"distribution": "equispaced"}, 20.0).names)
    ok = ok and {"error_barycentric", "error_vandermonde", "error_coefficients_vandermonde"} <= names
    ok = ok and worst_resid <= 100.0 * U
    rec = record(
        label="帧内 κ 单调不减 / 后向残差 / 三条通道齐备",
        theory="κ 随 n 增长；残差与 κ 无关（后向稳定）；帧必须同时给出三条误差通道",
        measured="；".join(rows) + f"；残差最大 {worst_resid:.1e}",
        tol="κ 不减；残差 ≤ 100u；三条通道齐备",
        passed=ok,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="四支可视化的帧契约：requiredAux、有限性、meta 标量、对数轴地板", chapter="2.1")
def test_all_chapter_frames_are_well_formed() -> None:
    """逐 viz 检查帧契约（与 ``tools`` 的体检规则一致），并核对"对数轴地板"的登记。

    重点看防误导那几条：每支 viz 的 ``pitfalls`` 长度、每个轴的 ``autoscaled``、
    对数轴帧是否把夹取计数写进了 meta（``n_error_at_floor``/``n_lambda_at_floor`` 等）。
    """
    from server.registry import Registry
    from server.frames import build_frame

    reg = Registry(str(ROOT))
    reg.discover(strict=False)
    entry = reg.get("2.1")
    report = []
    ok = True
    for viz in entry.manifest.viz_ids:
        spec = entry.manifest.viz(viz)
        params = {p.name: p.default for p in spec.params}
        fr = build_frame(entry, viz, params, 0.0)
        names = {n.lower() for n in fr.names}
        for aux in spec.required_aux:
            if not any(n.startswith(aux) for n in names):
                ok = False
                report.append(f"{viz}: requiredAux {aux} 缺失")
        nonfinite = [n for n in fr.names if not np.all(np.isfinite(fr[n]))]
        if nonfinite:
            ok = False
            report.append(f"{viz}: 非有限数组 {nonfinite}")
        for key, val in fr.meta.items():
            if isinstance(val, float) and not np.isfinite(val):
                ok = False
                report.append(f"{viz}: meta[{key}]={val}")
        if len(spec.pitfalls) < 60:
            ok = False
            report.append(f"{viz}: pitfalls 太短")
        if not all(a.autoscaled in (True, False) for a in spec.axes) or not spec.axes:
            ok = False
            report.append(f"{viz}: 轴声明不完整")
        has_log = any(a.scale == "log" for a in spec.axes)
        floor_keys = [k for k in fr.meta if "floor" in k]
        if has_log and not floor_keys:
            ok = False
            report.append(f"{viz}: 有对数轴却没登记地板")
        report.append(f"{viz}: {len(names)} 数组 / {len(fr.meta)} meta / 地板键 {floor_keys} / pitfalls {len(spec.pitfalls)} 字")
    rec = record(
        label="四支 viz 的帧契约",
        theory="requiredAux 齐备、数组与 meta 全有限、对数轴登记地板、pitfalls ≥ 60 字",
        measured="；".join(report),
        tol="全部通过",
        passed=ok,
    )
    assert rec.passed, report
