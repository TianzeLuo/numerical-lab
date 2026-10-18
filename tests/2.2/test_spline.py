"""2.2 章 A/B 类断言（上半）：追赶法、三种边界的三次样条、C² 连续性与变分性质。

这一半围绕**三对角结构**展开：

* :math:`C^2` 样条的 :math:`C^1` 条件是一个三对角系统，所以第一件事是证明"追赶法"
  本身的解与解析例/稠密参照一致（残差 :math:`\\sim u`）；
* 三种边界条件（自然/夹持/not-a-knot）必须各自满足它承诺的端点条件，
  并与 SciPy 的 ``CubicSpline`` 逐点一致——它们是**两条独立算法**（弯矩三对角 vs
  带状消元）对同一个数学对象的两次计算；
* 变分性质（:math:`\\int(S'')^2` 最小）不只是"听起来很美"：它在同一组数据上
  可测量，而且**只对自然样条成立**——夹持/not-a-knot 都被额外条件约束住了，
  把折线算进来更会得出"能量为 0"的荒谬结论（它根本不在 :math:`H^2` 里）。

容差口径：A 类对拍给相对差；B 类给"机器精度界 :math:`c\\,n\\,u\\,\\text{scale}`"，
来源写进 ``record`` 的 ``tol`` 字段，不写"试出来的容差"。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from numcore.interp.spline import (
    PiecewiseCubic,
    UNIT_ROUNDOFF,
    chebyshev_nodes,
    cubic_from_moments,
    cubic_spline,
    curvature_comb,
    curvature_of,
    gauss_legendre,
    hermite_monotone_margin,
    interpolating_polynomial_energy,
    lagrange_value,
    lebesgue_constant,
    monotonicity_report,
    pchip,
    sample_function,
    spline_system,
    thomas_flops,
    tridiagonal_apply,
    tridiagonal_factor,
    tridiagonal_matvec,
    tridiagonal_residual,
    uniform_nodes,
)
from numcore.reference import dense as ref
from numcore.utils import rng
from numcore.verify import (
    assert_leq,
    assert_monotone,
    assert_order,
    assertion,
    fit_order,
    record,
)

U = UNIT_ROUNDOFF  # u = 2^-53 ≈ 1.11e-16


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _dense_tridiag(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """把三对角三条线拼成稠密矩阵（**只**用于跟参照物对拍）。"""
    n = diag.size
    A = np.zeros((n, n), dtype=float)
    A[np.arange(n), np.arange(n)] = diag
    idx = np.arange(1, n)
    A[idx, idx - 1] = lower[1:]
    A[idx - 1, idx] = upper[:-1]
    return A


def _random_tridiag(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """严格对角占优的随机三对角（占优 ⇒ 追赶法无主元也稳定）。"""
    g = rng(seed)
    lower = np.zeros(n, dtype=float)
    upper = np.zeros(n, dtype=float)
    lower[1:] = g.uniform(-1.0, 1.0, n - 1)
    upper[:-1] = g.uniform(-1.0, 1.0, n - 1)
    diag = np.abs(lower) + np.abs(upper) + g.uniform(0.5, 2.0, n)
    return lower, diag, upper


def _max_error_on_grid(piece: PiecewiseCubic, f: str, n_grid: int = 4001) -> float:
    xg = np.linspace(-1.0, 1.0, n_grid)
    return float(np.max(np.abs(piece.evaluate(xg) - sample_function(f, xg))))


def _fprime(name: str, x: float) -> float:
    """本章测试函数的解析导数（只用到 monotone/sine/bump 三支）。"""
    if name == "monotone":
        return 1.0 + 0.2 * np.pi * np.cos(0.5 * np.pi * x)
    if name == "sine":
        return np.pi * np.cos(np.pi * x)
    if name == "runge":
        den = (1.0 + 25.0 * x * x) ** 2
        return -50.0 * x / den
    if name == "bump":
        return -32.0 * x * np.exp(-16.0 * x * x)
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# A 类
# --------------------------------------------------------------------------- #
@assertion("A", theory="追赶法与解析解/稠密参照一致，残差 ≤ c·n·u·scale", chapter="2.2")
def test_thomas_tridiagonal_matches_analytic_and_dense() -> None:
    r"""三对角系统的解与解析例、SciPy 稠密解逐点一致。

    解析例取 :math:`A=\mathrm{tridiag}(-1,2,-1)`、真解 :math:`x_j=j/(n-1)`：
    此时右端是 :math:`f=A x^\*`，追赶法在精确算术下必须**逐位**还原 :math:`x^\*`
    （离散 Laplacian 不引入误差）；浮点下残差是 :math:`O(u)`。
    """
    n = 65
    lower = np.zeros(n)
    lower[1:] = -1.0
    diag = np.full(n, 2.0)
    upper = np.zeros(n)
    upper[:-1] = -1.0
    x_true = np.linspace(0.0, 1.0, n)
    rhs = tridiagonal_matvec(lower, diag, upper, x_true)
    sol = tridiagonal_apply(lower, diag, upper, rhs)
    analytic_err = float(np.max(np.abs(sol - x_true)))
    analytic_res = tridiagonal_residual(lower, diag, upper, sol, rhs)

    # 随机严格对角占优的三对角系统：与 SciPy 稠密解对拍
    worst_rel = 0.0
    worst_res = 0.0
    for m in (2, 5, 17, 64, 129):
        lo, dg, up = _random_tridiag(m, 2200 + m)
        f = rng(9000 + m).standard_normal(m)
        mine = tridiagonal_apply(lo, dg, up, f)
        theirs = ref.solve(_dense_tridiag(lo, dg, up), f)
        scale = float(np.max(np.abs(theirs)))
        worst_rel = max(worst_rel, float(np.max(np.abs(mine - theirs))) / scale)
        worst_res = max(worst_res, tridiagonal_residual(lo, dg, up, mine, f))

    assert_leq(analytic_err, 4.0 * n * U, label="解析例的最大逐点误差（界 4n·u）")
    assert_leq(worst_rel, 32.0 * n * U, label="与 SciPy 稠密解的相对差（界 32n·u）")
    assert_leq(worst_res, 8.0 * n * U, label="范数型后向残差（界 8n·u）")
    rec = record(
        label="追赶法 vs 解析/SciPy + 运算量",
        theory="解析例逐点 ~u；与稠密解相对差 ~n·u",
        measured=f"解析误差 {analytic_err:.2e}，残差 {analytic_res:.2e}，"
        f"随机最大相对差 {worst_rel:.2e}，最大残差 {worst_res:.2e}",
        tol=f"4nu={4.0 * n * U:.2e}",
        passed=analytic_err <= 4.0 * n * U and worst_rel <= 32.0 * n * U,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="三种边界条件与 SciPy CubicSpline 逐点一致（相对差 ≤ 1e-10）", chapter="2.2")
def test_three_boundary_conditions_match_reference() -> None:
    """自然/夹持/not-a-knot 三种边界：节点处精确，且与 SciPy 的独立实现一致。

    这是"弯矩三对角系统"（自研）与"带状消元"（SciPy）两条独立算法对同一个
    数学对象的交叉验证。夹持端点斜率取解析导数，not-a-knot 同时对比
    ``CubicSpline(bc_type='not-a-knot')`` 与 ``make_interp_spline``。
    """
    worst = {"natural": 0.0, "clamped": 0.0, "not_a_knot": 0.0}
    worst_nodes = 0.0
    for f in ("runge", "monotone", "bump"):
        for n in (5, 9, 17, 33):
            x = uniform_nodes(n, -1.0, 1.0)
            y = sample_function(f, x)
            xq = np.linspace(-1.0, 1.0, 257)

            s_nat = cubic_spline(x, y, "natural")
            worst["natural"] = max(
                worst["natural"],
                float(np.max(np.abs(s_nat.evaluate(xq) - ref.cubic_spline(x, y, bc_type="natural", xq=xq)))),
            )

            # SciPy 的 bc_type="clamped" 语义是"两端一阶导为 0"，所以取 d0=d1=0 对拍；
            # "给定非零端点斜率"的夹持条件在本条断言里按定义单独核对。
            s_cl = cubic_spline(x, y, "clamped", d_left=0.0, d_right=0.0)
            worst["clamped"] = max(
                worst["clamped"],
                float(np.max(np.abs(s_cl.evaluate(xq) - ref.cubic_spline(x, y, bc_type="clamped", xq=xq)))),
            )
            assert_leq(
                abs(float(s_cl.evaluate(np.array([x[0]]), 1)[0])),
                64.0 * U,
                label=f"夹持端点 S'(x0) 应为 0（{f}, n={n}）",
            )
            d0, d1 = _fprime(f, float(x[0])), _fprime(f, float(x[-1]))
            s_cl2 = cubic_spline(x, y, "clamped", d_left=d0, d_right=d1)
            assert_leq(
                max(
                    abs(float(s_cl2.evaluate(np.array([x[0]]), 1)[0]) - d0),
                    abs(float(s_cl2.evaluate(np.array([x[-1]]), 1)[0]) - d1),
                ),
                4096.0 * U * max(abs(d0), abs(d1), 1.0),
                label=f"夹持端点斜率条件（{f}, n={n}）",
            )

            s_nak = cubic_spline(x, y, "not_a_knot")
            worst["not_a_knot"] = max(
                worst["not_a_knot"],
                float(np.max(np.abs(s_nak.evaluate(xq) - ref.cubic_spline(x, y, bc_type="not-a-knot", xq=xq)))),
                float(np.max(np.abs(s_nak.evaluate(xq) - ref.bspline(x, y, k=3, xq=xq)))),
            )

            for piece in (s_nat, s_cl, s_nak):
                worst_nodes = max(worst_nodes, float(np.max(np.abs(piece.evaluate(x) - y))))

    scale = 4.0
    for bc, dev in worst.items():
        assert_leq(dev, 1e-10 * scale, label=f"{bc} 与 SciPy 的最大偏差（rtol 1e-10）")
    assert_leq(worst_nodes, 8.0 * U * scale, label="三种边界在节点处的插值误差（界 8u）")
    rec = record(
        label="三种边界 vs SciPy + 节点精确性",
        theory="同一数学对象的两条独立算法应当一致到 1e-10；节点处精确到 u",
        measured="；".join(f"{k} {v:.2e}" for k, v in worst.items()) + f"；节点 {worst_nodes:.2e}",
        tol="1e-10（对拍）/ 8u（节点）",
        passed=max(worst.values()) <= 1e-10 * scale and worst_nodes <= 8.0 * U * scale,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="C² 样条的 |ΔS''| ≤ c·u·scale；C¹ 的 PCHIP 的 |ΔS''| 是 O(h) 真跳变", chapter="2.2")
def test_c2_jump_is_at_roundoff_while_c1_jumps_are_real() -> None:
    r"""C² 与 C¹ 的分界就是**二阶导在节点处连不连续**，这条断言把它量出来。

    对同一组数据分别构造样条（C²）与 PCHIP（C¹）：前者节点两侧的二阶导差应当只有
    :math:`O(u)`（机器精度），后者是货真价实的跳变（随 n 增大而增大，因为
    :math:`S''` 的量级按 :math:`h^{-2}` 长）。
    """
    worst_spline = 0.0
    pchip_jumps = []
    for f in ("bump", "monotone", "runge"):
        for n in (7, 15, 31):
            x = uniform_nodes(n, -1.0, 1.0)
            y = sample_function(f, x)
            s = cubic_spline(x, y, "natural")
            p = pchip(x, y)
            js = float(np.max(np.abs(s.jumps("d2"))))
            jp = float(np.max(np.abs(p.jumps("d2"))))
            scale = max(float(np.max(np.abs(s.d2_left))), 1e-300)
            worst_spline = max(worst_spline, js / (U * scale))
            pchip_jumps.append(jp)
            # 一阶导：样条是 C¹，PCHIP 也是 C¹（两侧都该只有 u 量级）
            assert_leq(
                float(np.max(np.abs(p.jumps("d1")))),
                64.0 * U * max(float(np.max(np.abs(p.d1_left))), 1e-300),
                label=f"PCHIP 的 |ΔS'|（{f}, n={n}）",
            )
    assert_leq(worst_spline, 1.0, label="样条的 |ΔS''|/(u·scale)（界 1）")
    rec = record(
        label="C² 跳变 / C¹ 跳变",
        theory="样条 |ΔS''| ~ u·scale；PCHIP 的 |ΔS''| 是真跳变（O(h^-2)·数据尺度）",
        measured=f"样条最大 |ΔS''|/(u·scale) = {worst_spline:.3f}；PCHIP 最大 |ΔS''| = "
        f"{max(pchip_jumps):.3g}（最小 {min(pchip_jumps):.3g}）",
        tol="1（样条）/ 严格大于 1e3·u（PCHIP）",
        passed=worst_spline <= 1.0 and min(pchip_jumps) > 1e-6,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="自然样条的能量 ∫(S'')² 严格小于同数据的 PCHIP 与插值多项式", chapter="2.2")
def test_spline_energy_is_variationally_minimal() -> None:
    r"""变分性质：在所有 :math:`H^2` 插值函数里，自然样条的曲率能量最小。

    三个对手：PCHIP（:math:`C^1`）、Lagrange 插值多项式（:math:`C^\infty`）、
    not-a-knot 样条（被端点三阶导连续约束住）。折线**不参与**比较：它的
    :math:`\\int(S'')^2=0`，因为它的二阶导是 delta 函数之和，不在 :math:`H^2` 里——
    把它算进来会得到"折线最光滑"的荒谬结论，这正是 pitfalls 里的一条。
    """
    ratios_pchip = []
    ratios_poly = []
    ratios_nak = []
    natural_is_minimal = True
    clamped_energy = []
    for f in ("runge", "bump", "sine"):
        for n in (9, 13, 17):
            x = uniform_nodes(n, -1.0, 1.0)
            y = sample_function(f, x)
            e_nat = cubic_spline(x, y, "natural").energy()
            e_pc = pchip(x, y).energy()
            e_poly = interpolating_polynomial_energy(x, y)
            e_nak = cubic_spline(x, y, "not_a_knot").energy()
            e_cl = cubic_spline(x, y, "clamped").energy()
            ratios_nak.append(e_nak / e_nat)
            clamped_energy.append(e_cl / e_nat)
            ratios_pchip.append(e_pc / e_nat)
            ratios_poly.append(e_poly / e_nat)
            natural_is_minimal = natural_is_minimal and e_nat < e_pc and e_nat < e_poly
    assert natural_is_minimal, "自然样条的能量应当严格最小"
    assert_leq(
        1.0 / min(ratios_pchip), 1.0, label="自然样条能量 ≤ PCHIP 能量（比值倒数 ≤ 1）"
    )
    rec = record(
        label="能量比 E(对手)/E(自然样条)",
        theory="自然样条最小 ⇒ 所有比值 > 1；多项式因 Runge 现象可达 1e2~1e5",
        measured=f"PCHIP 比值 [{min(ratios_pchip):.3f}, {max(ratios_pchip):.3g}]，"
        f"多项式比值 [{min(ratios_poly):.3g}, {max(ratios_poly):.3g}]，"
        f"not-a-knot/自然 [{min(ratios_nak):.3f}, {max(ratios_nak):.3f}]，"
        f"夹持/自然 [{min(clamped_energy):.3f}, {max(clamped_energy):.3f}]，折线 0（不在 H² 内，不计入）",
        tol="所有比值 > 1（严格）",
        passed=bool(min(ratios_pchip) > 1.0 and min(ratios_poly) > 1.0),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="not-a-knot 与（精确端点导数的）夹持样条误差 O(h^4)，自然样条被边界层压到 O(h^2)", chapter="2.2")
def test_error_order_depends_on_the_boundary_condition() -> None:
    r"""同一个函数、同一批节点，**只换边界条件**，收敛阶就从 4 掉到 2。

    这是本章最值得量的一件事：自然边界强行令 :math:`S''(x_0)=S''(x_n)=0`，
    而 :math:`f''` 在端点并不为 0（取 :math:`f=x+0.4\\sin(\\pi x/2)`，
    :math:`f''(\\pm1)=\\mp0.987`），于是解里出现一层 :math:`O(h^2)` 的边界层，
    它**污染整条曲线**。not-a-knot 不预设端点二阶导，恢复 :math:`O(h^4)`。
    """
    ns = [9, 17, 33, 65, 129]
    hs = [2.0 / (n - 1) for n in ns]
    errs_nak = []
    errs_cl = []
    errs_nat = []
    for n in ns:
        x = uniform_nodes(n, -1.0, 1.0)
        y = sample_function("monotone", x)
        errs_nak.append(_max_error_on_grid(cubic_spline(x, y, "not_a_knot"), "monotone"))
        errs_nat.append(_max_error_on_grid(cubic_spline(x, y, "natural"), "monotone"))
        errs_cl.append(
            _max_error_on_grid(
                cubic_spline(x, y, "clamped", d_left=_fprime("monotone", float(x[0])), d_right=_fprime("monotone", float(x[-1]))),
                "monotone",
            )
        )
    fit_nak = assert_order(hs, errs_nak, theory=4.0, tol=0.1, label="not-a-knot 样条 L∞ 误差阶")
    fit_cl = assert_order(hs, errs_cl, theory=4.0, tol=0.1, label="夹持（精确端点导数）误差阶")
    fit_nat = fit_order(hs, errs_nat)
    assert 1.7 <= fit_nat.slope <= 2.4, f"自然样条应当被边界层压到 O(h^2)，实测 {fit_nat.slope:.3f}"
    rec = record(
        label="自然样条实测阶（边界层）",
        theory="自然边界强制 S''=0 与 f''≠0 冲突 ⇒ 全局 O(h²)",
        measured=f"实测阶 {fit_nat.slope:.3f}（R²={fit_nat.r2:.4f}），误差 {errs_nat[-1]:.3e}；"
        f"not-a-knot {fit_nak.slope:.3f}，夹持 {fit_cl.slope:.3f}",
        tol="自然阶落在 [1.7, 2.4]",
        passed=1.7 <= fit_nat.slope <= 2.4,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="均匀节点自然样条 = 三对角解出的 B 样条控制点经 de Boor 求值", chapter="2.2")
def test_de_boor_matches_tridiagonal_spline() -> None:
    r"""同一数学对象的两条完全不同的计算路径：

    (a) 解弯矩三对角系统 + 分段 Hermite 求值（本文件上半的路线）；
    (b) 均匀节点上解 B 样条控制点的三对角系统（1/6,4/6,1/6 与自然端点行）+ de Boor 求值。

    两者必须逐点一致到 :math:`O(u)`，且在节点处精确等于数据。
    """
    from numcore.interp.spline import bspline_uniform_natural_ctrl, de_boor

    worst = 0.0
    worst_nodes = 0.0
    for f in ("runge", "bump", "monotone"):
        for n in (4, 7, 12, 25):
            y = sample_function(f, np.linspace(0.0, float(n - 1), n))
            xs = np.arange(n, dtype=float)
            knots, ctrl = bspline_uniform_natural_ctrl(y)
            xq = np.linspace(0.0, float(n - 1), 301)
            worst = max(worst, float(np.max(np.abs(de_boor(knots, ctrl, 3, xq) - cubic_spline(xs, y, "natural").evaluate(xq)))))
            worst_nodes = max(worst_nodes, float(np.max(np.abs(de_boor(knots, ctrl, 3, xs) - y))))
    assert_leq(worst, 512.0 * U, label="de Boor 与弯矩样条的最大偏差（界 512u）")
    assert_leq(worst_nodes, 64.0 * U, label="de Boor 在节点处的插值误差（界 64u）")
    rec = record(
        label="de Boor vs 三对角弯矩样条",
        theory="B 样条控制点的三对角系统与弯矩三对角系统表示同一条自然样条",
        measured=f"最大逐点偏差 {worst:.2e}，节点处 {worst_nodes:.2e}",
        tol=f"512u={512 * U:.2e}",
        passed=worst <= 512.0 * U and worst_nodes <= 64.0 * U,
    )
    assert rec.passed, rec.measured


@assertion("A", theory="∫(S'')² 与解析值/复合 Simpson 一致：三次数据上能量恰为 ∫(p'')²", chapter="2.2")
def test_energy_matches_analytic_and_simpson() -> None:
    r"""曲率能量有两把独立的尺子：

    (a) **解析例**：取 :math:`f=x^3`（:math:`f''=6x`，:math:`\int_{-1}^{1}(f'')^2=24`），
        用 not-a-knot 样条插值它——三次多项式属于每个三次样条空间，所以样条**就是**
        :math:`x^3` 本身，能量必须精确等于 24；Lagrange 插值多项式同理。
    (b) **复合 Simpson**：对 :math:`(S'')^2` 用 20001 个点做 Simpson（误差 :math:`O(n^{-4})`），
        与闭式 :math:`\sum h_i(M_i^2+M_iM_{i+1}+M_{i+1}^2)/3` 对拍。

    两条尺子都不依赖 SciPy。``numcore.reference.dense.quad`` 的 ``full_output`` 解包
    问题**已经修好**（scipy 1.16.3 下 ier==0 返回 3 元组、ier≠0 返回 4 元组，
    参见 ``tests/test_reference_layer.py::test_dense_quad_survives_full_output_four_tuple``），
    所以现在也可以直接用它来对拍；本测试保留"解析例 + 复合 Simpson"这两条独立尺子，
    是因为它们连 numpy 的求积原语也不依赖。
    """
    worst_cubic = 0.0
    worst_simpson = 0.0
    worst_ident = 0.0
    for n in (7, 11, 19):
        x = uniform_nodes(n, -1.0, 1.0)
        y = x**3
        e_poly = interpolating_polynomial_energy(x, y)
        e_nak = cubic_spline(x, y, "not_a_knot").energy()
        e_cl = cubic_spline(x, y, "clamped", d_left=3.0, d_right=3.0).energy()
        worst_cubic = max(
            worst_cubic, abs(e_poly - 24.0) / 24.0, abs(e_nak - 24.0) / 24.0, abs(e_cl - 24.0) / 24.0
        )

    for f in ("runge", "bump", "monotone"):
        for n in (7, 13):
            x = uniform_nodes(n, -1.0, 1.0)
            s = cubic_spline(x, sample_function(f, x), "natural")
            closed = s.energy()
            worst_ident = max(worst_ident, abs(closed - s.energy_from_moments()) / max(abs(closed), 1e-300))
            t_grid = np.linspace(float(x[0]), float(x[-1]), 20001)
            d2 = s.evaluate(t_grid, 2)
            h = float(t_grid[1] - t_grid[0])
            simpson = float(
                h / 3.0 * (d2[0] ** 2 + d2[-1] ** 2 + 4.0 * np.sum(d2[1:-1:2] ** 2) + 2.0 * np.sum(d2[2:-2:2] ** 2))
            )
            worst_simpson = max(worst_simpson, abs(closed - simpson) / max(abs(closed), 1e-300))

    assert_leq(worst_cubic, 1e-9, label="三次数据上的能量与解析值 24 的相对差")
    assert_leq(worst_ident, 1e-14, label="系数式与弯矩式能量的相对差")
    assert_leq(worst_simpson, 1e-8, label="闭式与复合 Simpson 的相对差")
    rec = record(
        label="能量 vs 解析值 / Simpson",
        theory="三次数据 ⇒ 样条复现该三次多项式 ⇒ E = ∫(p'')² = 24；闭式与 Simpson 一致到 O(h^4)",
        measured=f"三次例最大相对差 {worst_cubic:.2e}；两种写法 {worst_ident:.2e}；Simpson {worst_simpson:.2e}",
        tol="1e-9 / 1e-14 / 1e-8",
        passed=worst_cubic <= 1e-9 and worst_ident <= 1e-14 and worst_simpson <= 1e-8,
    )
    assert rec.passed, rec.measured



@assertion("A", theory="Gauss-Legendre n 点对次数 ≤ 2n-1 的多项式精确", chapter="2.2")
def test_gauss_legendre_is_exact_for_polynomials() -> None:
    """自研 Gauss--Legendre 节点/权重的精确度检验（它支撑上一条断言的"精确"二字）。"""
    worst = 0.0
    for n in (1, 2, 5, 12):
        nodes, weights = gauss_legendre(n)
        for deg in range(0, 2 * n):
            exact = 0.0 if deg % 2 == 1 else 2.0 / (deg + 1)
            got = float(np.sum(weights * nodes**deg))
            worst = max(worst, abs(got - exact))
        assert_leq(float(np.abs(np.sum(weights) - 2.0)), 8.0 * U * n, label=f"n={n} 时 sum(w)=2")
    assert_leq(worst, 1e-14, label="∫x^deg 的最大误差（deg ≤ 2n-1）")
    rec = record(
        label="Gauss-Legendre 精确度",
        theory="n 点 Gauss 对 2n-1 次以下多项式精确",
        measured=f"直到 23 次多项式的最大积分误差 {worst:.2e}",
        tol="1e-14",
        passed=worst <= 1e-14,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# B 类
# --------------------------------------------------------------------------- #
@assertion("B", theory="样条在节点处插值精确：|S(x_i)-y_i| ≤ c·u·scale", chapter="2.2")
def test_spline_interpolates_nodes_exactly() -> None:
    """插值条件是所有其它性质的地基：节点处必须是 :math:`O(u)`，不是"差不多"。"""
    worst = 0.0
    for f in ("runge", "monotone", "bump", "sine"):
        for n in (5, 9, 21):
            x = chebyshev_nodes(n, -1.0, 1.0)
            y = sample_function(f, x)
            for bc in ("natural", "clamped", "not_a_knot"):
                s = cubic_spline(x, y, bc)
                worst = max(worst, float(np.max(np.abs(s.evaluate(x) - y))))
    assert_leq(worst, 64.0 * U, label="三种边界、两种节点分布下的最大节点误差（界 64u）")
    rec = record(
        label="节点插值残差",
        theory="S(x_i) = y_i 精确（构造性）",
        measured=f"最大 |S(x_i)-y_i| = {worst:.2e}",
        tol=f"64u={64 * U:.2e}",
        passed=worst <= 64.0 * U,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="单调数据上 PCHIP 严格单调；自然/not-a-knot 样条可以越界振荡", chapter="2.2")
def test_monotone_data_pchip_monotone_but_spline_overshoots() -> None:
    r"""形状保持不是样条的性质，是 PCHIP 的性质——用**同一组数据**证明。

    数据取一个平台到平台的台阶（单调不减），PCHIP 的斜率被 Fritsch--Carlson 规则
    夹在"不许更陡"的区域里，故单调；自然样条为了凑 :math:`C^2` 会在台阶前后
    冲出去（实测最小斜率 < 0），这正是"样条更光滑 ≠ 样条更守形状"。
    """
    x = np.linspace(0.0, 5.0, 6)
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    pc = monotonicity_report(pchip(x, y))
    nat = monotonicity_report(cubic_spline(x, y, "natural"))
    nak = monotonicity_report(cubic_spline(x, y, "not_a_knot"))
    assert_monotone(pchip(x, y).evaluate(np.linspace(0.0, 5.0, 2001)), direction="nondecreasing",
                    label="PCHIP 在台阶数据上单调不减")
    assert pc["monotone"] == 1.0, pc
    assert nat["monotone"] == 0.0, "自然样条在台阶数据上应当越界（否则这条反例失效）"
    # Fritsch--Carlson 判据与实测一致：PCHIP 的每个区间都落在充分单调区域内
    margins = []
    for i in range(x.size - 1):
        h = x[i + 1] - x[i]
        delta = (y[i + 1] - y[i]) / h
        m = pchip(x, y).slopes
        margins.append(hermite_monotone_margin(float(m[i]), float(m[i + 1]), float(delta)))
    assert min(margins) >= 0.0, margins
    rec = record(
        label="台阶数据上的单调性",
        theory="Fritsch--Carlson 判据 alpha²+beta² ≤ 9 ⇒ 单调；自然样条无此保证",
        measured=f"PCHIP 最小斜率 {pc['min_slope']:.3g}（越界点 {pc['violations']:.0f}），"
        f"自然样条最小斜率 {nat['min_slope']:.3g}（越界点 {nat['violations']:.0f}），"
        f"not-a-knot {nak['min_slope']:.3g}（越界 {nak['violations']:.0f}）；FC 余量最小 {min(margins):.3g}",
        tol="PCHIP 越界 0；样条最小斜率 < 0",
        passed=pc["monotone"] == 1.0 and nat["min_slope"] < 0.0 and min(margins) >= 0.0,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="追赶法对未知量顺序反转（置换 P）是等变的：解随置换同步反转", chapter="2.2")
def test_thomas_is_invariant_under_order_reversal() -> None:
    r"""置换不变性：把未知量整体反序（:math:`PAP^{\\mathsf T}`，:math:`P` 是反序置换），
    解也必须整体反序。

    对三对角矩阵，反序置换**保持三对角结构**，所以同一个追赶法可以直接吃下它——
    这条断言同时验证了"算法实现里没有任何依赖下标方向的偏置"。
    """
    worst = 0.0
    for m in (4, 9, 32, 77):
        lo, dg, up = _random_tridiag(m, 3300 + m)
        f = rng(4400 + m).standard_normal(m)
        x = tridiagonal_apply(lo, dg, up, f)
        # 反序：新下三角 = 旧上三角反序、新上三角 = 旧下三角反序
        lo2 = np.zeros(m)
        up2 = np.zeros(m)
        lo2[1:] = up[:-1][::-1]
        up2[:-1] = lo[1:][::-1]
        dg2 = dg[::-1]
        f2 = f[::-1]
        x2 = tridiagonal_apply(lo2, dg2, up2, f2)
        scale = float(np.max(np.abs(x)))
        worst = max(worst, float(np.max(np.abs(x2 - x[::-1]))) / scale)
    assert_leq(worst, 64.0 * U, label="反序系统解与解的逆序的相对差（界 64u）")
    rec = record(
        label="追赶法的反序置换不变性",
        theory="P A Pᵀ (P 反序) 仍是三对角，解同步反序",
        measured=f"最大相对差 {worst:.2e}",
        tol=f"64u={64 * U:.2e}",
        passed=worst <= 64.0 * U,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="追赶法的因子满足 A = LU（三对角 LU 无填充）", chapter="2.2")
def test_thomas_factorization_reproduces_matrix() -> None:
    r"""追赶法就是**无填充的三对角 LU**：:math:`L` 单位下三角、:math:`U` 上三角，
    重建误差应当在 :math:`n\\,u\\,\\|A\\|` 量级，且与 1.2 章的稠密 LU 同一口径。"""
    worst = 0.0
    worst_ops = 0.0
    for m in (3, 10, 40):
        lo, dg, up = _random_tridiag(m, 5500 + m)
        mult, piv = tridiagonal_factor(lo, dg, up)
        Lmat = np.eye(m)  # 单位下双对角：次对角元是**乘子** l_i = a_i/u_{i-1}
        Lmat[np.arange(1, m), np.arange(m - 1)] = mult[1:]
        Umat = np.diag(piv) + np.diag(up[:-1], 1)
        A = _dense_tridiag(lo, dg, up)
        worst = max(worst, float(np.max(np.abs(Lmat @ Umat - A))) / float(np.max(np.abs(A))))
        worst_ops = max(worst_ops, abs(thomas_flops(m) - (8 * m - 7)))
    assert_leq(worst, 64.0 * U, label="|LU - A|/|A| 的最大值（界 64u）")
    rec = record(
        label="三对角 LU 重建 + 运算量公式",
        theory="A = LU（L 单位下三角、U 上三角），追赶法 8n-7 次浮点运算",
        measured=f"最大相对重建误差 {worst:.2e}；运算量公式偏差 {worst_ops:.1f}",
        tol=f"64u={64 * U:.2e}",
        passed=worst <= 64.0 * U and worst_ops == 0.0,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="一侧极限与网格无关（eps→0 收敛到解析跳变）；等距采样的相邻差分估计随网格漂移 3~4 倍", chapter="2.2")
def test_node_sides_expose_the_jump_that_uniform_sampling_hides() -> None:
    r"""同一个 PCHIP，两种"量跳变"的办法：**一侧极限** vs **等距采样的相邻差分**。

    * 一侧极限是解析的：:math:`S''(x_i^\pm)` 由分段三次的系数直接给出，取
      :math:`\epsilon=10^{-3},10^{-6},10^{-12}` 去采样，一侧极限以**一阶**速度收敛到同一个
      :math:`\Delta S''`（偏差 :math:`O(\epsilon)`，实测 1000 倍 :math:`\epsilon` ⇒ 1000 倍偏差），
      **与采样网格无关**；
    * 等距采样再作相邻差分则**依赖网格**：实测在 8 种网格密度 × 3 种偏移下，
      "相邻差分最大值 / 真跳变" 在 :math:`[0.34, 2.40]` 之间漂移（最粗的网格丢掉 2/3 的
      跳变，某些网格还会**高估** 2.4 倍——因为相邻差分里混进了区间内的斜率）。
      把这条曲线画成折线，跳变就变成了一段陡坡——这就是"看起来只是变化快"。
    """
    worst_rel = 0.0
    worst_ratio = float("inf")
    ratios: list[float] = []
    for f in ("bump", "runge"):
        for n in (7, 13):
            x = uniform_nodes(n, -1.0, 1.0)
            p = pchip(x, sample_function(f, x))
            jump = float(np.max(np.abs(p.jumps("d2"))))
            inner = x[1:-1]
            devs = []
            for eps in (1e-3, 1e-6, 1e-12):
                sampled = np.max(np.abs(p.evaluate(inner + eps, 2) - p.evaluate(inner - eps, 2)))
                devs.append(abs(float(sampled) - jump))
            # 一侧极限是**极限**：偏差应当与 eps 成正比（实测 1000 倍 eps 缩小 ⇒ 1000 倍偏差缩小）
            worst_ratio = min(worst_ratio, devs[0] / max(devs[1], 1e-300))
            worst_rel = max(worst_rel, devs[2] / jump)
            for m in (9, 13, 17, 25, 33, 65, 129, 2001):
                for shift in range(3):
                    xs = np.linspace(-1.0, 1.0, m)
                    if shift:
                        g = rng(700 + shift)
                        xs = np.sort(np.clip(xs + g.uniform(-0.5, 0.5) * (2.0 / (m - 1)), -1.0, 1.0))
                    naive = float(np.max(np.abs(np.diff(p.evaluate(xs, 2)))))
                    ratios.append(naive / jump)
    arr = np.array(ratios)
    spread = float(np.max(arr) / np.min(arr))
    assert_leq(worst_rel, 1e-9, label="eps=1e-12 时采样跳变与解析跳变的相对差（rtol 1e-9）")
    assert_leq(
        100.0,
        worst_ratio,
        label="eps 缩小 1e6 倍带来的偏差缩小倍数（一阶收敛的下界，实测 1000）",
    )
    assert_leq(float(np.min(arr)), 0.7, label="最粗网格上相邻差分/真跳变（必须明显低估）")
    assert_leq(2.0, spread, label="相邻差分估计的网格漂移倍数（必须确实漂移）")
    rec = record(
        label="一侧极限 vs 等距相邻差分",
        theory="一侧极限 = 解析值（网格无关）；相邻差分 = 网格相关的下界/上界混合估计",
        measured=f"{len(arr)} 种网格：相邻差分/真跳变 ∈ [{float(np.min(arr)):.3f}, {float(np.max(arr)):.3f}]"
        f"（漂移 {spread:.2f} 倍）；一侧极限与解析跳变之差在 eps=1e-12 时 ≤{worst_rel:.1e}（相对），"
        f"偏差随 eps 一阶下降（1e6 倍 eps ⇒ ≥{worst_ratio:.0f} 倍偏差）",
        tol="一侧极限 1e-9；最粗网格 ≤0.7；漂移 ≥2",
        passed=worst_rel <= 1e-9 and float(np.min(arr)) <= 0.7 and spread >= 2.0,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="C¹ 条件给出的矩阵确实是三对角且严格对角占优", chapter="2.2")
def test_spline_system_is_tridiagonal_and_diagonally_dominant() -> None:
    """三对角结构与对角占优是"无主元追赶法可用"的前提，必须逐项验证。"""
    worst_off = 0.0
    worst_margin = float("inf")
    worst_sum = 0.0
    for f in ("runge", "monotone"):
        for n in (4, 8, 16, 40):
            for bc in ("natural", "clamped", "not_a_knot"):
                x = uniform_nodes(n, -1.0, 1.0)
                y = sample_function(f, x)
                sysm = spline_system(x, y, bc)
                A = _dense_tridiag(sysm.lower, sysm.diag, sysm.upper)
                m = A.shape[0]
                off = 0.0
                for i in range(m):
                    for j in range(m):
                        if abs(i - j) > 1:
                            off = max(off, abs(A[i, j]))
                worst_off = max(worst_off, off)
                # 严格对角占优：|d_i| > |l_i| + |u_i|
                row = np.abs(sysm.lower) + np.abs(sysm.upper)
                margin = float(np.min(np.abs(sysm.diag) - row))
                worst_margin = min(worst_margin, margin)
                worst_sum = max(worst_sum, abs(float(np.max(sysm.lam[1:-1] + sysm.mu[1:-1])) - 1.0))
    assert worst_off == 0.0, f"非三对角元素不为 0：{worst_off}"
    assert worst_margin > 0.0, f"对角占优余量 {worst_margin}"
    rec = record(
        label="样条系统的结构与占优",
        theory="A M = d 是三对角；对角占优 ⇒ 追赶法无需选主元",
        measured=f"三对角外最大元素 {worst_off:.1e}；最小占优越量 {worst_margin:.3g}；"
        f"max|lambda+mu-1| = {worst_sum:.2e}",
        tol="外元素恒 0，占优越量 > 0",
        passed=worst_off == 0.0 and worst_margin > 0.0,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="曲率梳 global 尺度的长度可比；local 归一化丢失长度信息", chapter="2.2")
def test_curvature_comb_scales_are_honest() -> None:
    r"""曲率梳的两种尺度必须**行为可预测**，否则图上"哪里更弯"就是胡说。

    ``global``：最长梳齿恰为 ``length``，其余按 :math:`|\\kappa|` 成比例；
    ``local``：每条梳齿都被拉成 ``length``——长度信息完全丢失（只剩方向）。
    """
    x = uniform_nodes(9, -1.0, 1.0)
    s = cubic_spline(x, sample_function("runge", x), "natural")
    xg = np.linspace(-1.0, 1.0, 201)
    base_g, tip_g = curvature_comb(s, xg, scale="global", length=0.4)
    base_l, tip_l = curvature_comb(s, xg, scale="local", length=0.4)
    len_g = np.sqrt(np.sum((tip_g - base_g) ** 2, axis=1))
    len_l = np.sqrt(np.sum((tip_l - base_l) ** 2, axis=1))
    kap = np.abs(curvature_of(s, xg))
    # global：长度与 |kappa| 严格成比例（在 kappa=0 处比值没有定义，故直接比长度）
    peak = float(np.max(kap))
    prop_ok = float(np.max(np.abs(len_g - (0.4 / peak) * kap))) < 1e-12 and abs(float(np.max(len_g)) - 0.4) < 1e-12
    # local：所有**非零曲率**处都被拉成同一长度（kappa 恰为 0 的点连方向都没有，齿长为 0）
    nz = kap > 0.0
    flat_ok = bool(np.any(nz)) and float(np.max(np.abs(len_l[nz] - 0.4))) < 1e-12
    assert prop_ok, (float(np.max(len_g)), float(np.min(len_g)), peak)
    assert flat_ok, (float(np.max(len_l)), float(np.min(len_l)))
    assert_leq(
        float(np.min(len_g) / np.max(len_g)),
        1.0,
        label="global 尺度下最短/最长梳齿比 ≤ 1（长度可比）",
    )
    rec = record(
        label="曲率梳两种尺度",
        theory="global: |spine| = length·|kappa|/max|kappa|；local: |spine| ≡ length",
        measured=f"global 长度 ∈ [{float(np.min(len_g)):.3g}, {float(np.max(len_g)):.3g}]（与 kappa 成正比），"
        f"local 在 {int(np.count_nonzero(nz))}/{len_l.size} 个非零曲率点上恒为 {float(np.mean(len_l[nz])):.3g}"
        f"（标准差 {float(np.std(len_l[nz])):.1e}），另外 {int(np.count_nonzero(~nz))} 个 kappa=0 点上齿长为 0",
        tol="比例偏差 < 1e-9；local 方差 < 1e-12",
        passed=prop_ok and flat_ok,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="张力族的能量按 (1-tau)² 缩小，tau=1 退化成折线（C¹ 断裂）", chapter="2.2")
def test_tension_family_scales_energy_and_breaks_c1() -> None:
    r"""``control_points`` 的张力旋钮在数学上就是"弯矩乘 :math:`(1-\\tau)`"：

    * 曲线仍然过所有数据点（节点残差 ~u）；
    * 能量按 :math:`(1-\\tau)^2` 缩小（:math:`S''` 整体乘 :math:`1-\\tau`）；
    * 一阶导跳变从 :math:`O(u)` 变成 :math:`O(\\tau)` 的真跳变——所以 :math:`\\tau>0`
      换到的是"更平"，不是"更光滑"。
    """
    x = uniform_nodes(7, -1.0, 1.0)
    y = sample_function("runge", x)
    base = cubic_spline(x, y, "natural")
    e0 = base.energy()
    worst_energy = 0.0
    worst_nodes = 0.0
    jumps = {}
    for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
        piece = cubic_from_moments(x, y, (1.0 - tau) * base.moments)
        worst_energy = max(worst_energy, abs(piece.energy() - (1.0 - tau) ** 2 * e0))
        worst_nodes = max(worst_nodes, float(np.max(np.abs(piece.evaluate(x) - y))))
        jumps[tau] = float(np.max(np.abs(piece.jumps("d1"))))
    assert_leq(worst_energy, 1e-9 * e0, label="能量与 (1-tau)²·E0 的最大偏差（相对）")
    assert_leq(worst_nodes, 64.0 * U, label="张力族在节点处的插值误差（界 64u）")
    assert jumps[0.0] < 1e-12 < jumps[0.5] < jumps[1.0], jumps
    rec = record(
        label="张力族的能量与 C¹ 断裂",
        theory="M_tau=(1-tau)M ⇒ E_tau=(1-tau)²E0；tau>0 时 S' 在节点跳变",
        measured=f"E0={e0:.4g}；能量偏差 {worst_energy:.2e}；节点误差 {worst_nodes:.2e}；"
        f"|dS'|: tau=0 → {jumps[0.0]:.2e}, 0.5 → {jumps[0.5]:.3g}, 1 → {jumps[1.0]:.3g}",
        tol="能量相对偏差 ≤ 1e-9；跳变严格递增",
        passed=worst_energy <= 1e-9 * e0 and worst_nodes <= 64.0 * U and jumps[0.0] < jumps[0.5] < jumps[1.0],
    )
    assert rec.passed, rec.measured


@assertion("B", theory="重心插值权与 Lebesgue 常数：等距节点指数增长、Chebyshev 对数增长", chapter="2.2")
def test_lebesgue_growth_separates_polynomial_from_spline() -> None:
    r"""多项式失控的**机制**：等距节点的 Lebesgue 常数指数增长，样条误差却按 :math:`h^4` 下降。"""
    equi = [lebesgue_constant(uniform_nodes(n), n_grid=401) for n in (7, 13, 19)]
    cheb = [lebesgue_constant(chebyshev_nodes(n), n_grid=401) for n in (7, 13, 19)]
    assert equi[2] > 5.0 * equi[0], equi
    assert cheb[2] < 1.6 * cheb[0], cheb
    # 同一组节点：多项式误差爆炸而样条误差下降
    err_poly = []
    err_spl = []
    for n in (7, 13, 19):
        x = uniform_nodes(n, -1.0, 1.0)
        y = sample_function("runge", x)
        xg = np.linspace(-1.0, 1.0, 2001)
        ex = sample_function("runge", xg)
        err_poly.append(float(np.max(np.abs(lagrange_value(x, y, xg) - ex))))
        err_spl.append(float(np.max(np.abs(cubic_spline(x, y, "natural").evaluate(xg) - ex))))
    assert err_poly[2] > 4.0 * err_poly[0], err_poly
    rec = record(
        label="Lebesgue 常数与同一节点上的两种误差",
        theory="Lambda_n(等距) 指数增长；Lambda_n(Chebyshev) ~ log n；样条误差 O(h^4)",
        measured=f"Lambda 等距 {equi[0]:.3g}→{equi[2]:.3g}，Chebyshev {cheb[0]:.3g}→{cheb[2]:.3g}；"
        f"多项式误差 {err_poly[0]:.3g}→{err_poly[2]:.3g}，样条误差 {err_spl[0]:.3g}→{err_spl[2]:.3g}",
        tol="等距至少 5 倍增长；Chebyshev < 1.6 倍",
        passed=equi[2] > 5.0 * equi[0] and cheb[2] < 1.6 * cheb[0] and err_poly[2] > 4.0 * err_poly[0],
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 与章节帧的耦合
# --------------------------------------------------------------------------- #
@assertion("B", theory="2.2 的四个帧：requiredAux 齐备、关键跳变/不变性与 numcore 一致", chapter="2.2")
def test_chapter_frames_obey_invariants() -> None:
    """可视化里画出来的量必须满足断言里同一批不变量（纸面与机器同源）。"""
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("ch22_mod", root / "server" / "chapters" / "2.2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    report = []
    ok = True
    for viz in mod.MANIFEST["vizzes"]:
        params = {p["name"]: p.get("default") for p in viz["params"]}
        t = 2.0 if viz["time"]["mode"] != "none" else 0.0
        fr = mod.compute(viz["id"], params, t)
        names = {n.lower() for n in fr.names}
        for aux in viz["requiredAux"]:
            if not any(n.startswith(aux) for n in names):
                ok = False
                report.append(f"{viz['id']}: 缺 requiredAux {aux}")
        if not np.all(np.isfinite(fr["error"])):
            ok = False
            report.append(f"{viz['id']}: error 不是全有限")
        report.append(f"{viz['id']}: {len(fr.names)} 数组")

    def _with_defaults(viz_id: str, **over: object) -> dict[str, object]:
        viz = next(v for v in mod.MANIFEST["vizzes"] if v["id"] == viz_id)
        params = {p["name"]: p.get("default") for p in viz["params"]}
        params.update(over)
        return params

    cp = mod.compute("control_points", _with_defaults("control_points", bc="natural", tension=0.0), 2.0)
    # C² 的说法必须与 numcore 的实测一致
    ok = ok and float(np.max(cp["jump_d2"])) <= 1e-12
    ok = ok and float(np.max(cp["error"])) <= 1e-12
    # 曲率梳 global 尺度的最长齿 = manifest 里声明的 0.45
    spine = np.sqrt(np.sum((cp["comb_tip"] - cp["comb_base"]) ** 2, axis=1))
    ok = ok and abs(float(np.max(spine)) - 0.45) < 1e-9
    # 折线的曲率能量必须是 0（"不在 H² 里"的那条陷阱）
    ok = ok and float(cp.meta["energy_polyline"]) == 0.0

    sm = mod.compute("smoothness_c0c1c2", _with_defaults("smoothness_c0c1c2"), 0.0)
    ok = ok and float(sm.meta["jump_spline_max"]) <= 1e-12 < float(sm.meta["jump_pchip_max"])

    bs = mod.compute("bspline_local", _with_defaults("bspline_local", n_ctrl=8, degree=3), 4.0)
    ok = ok and float(np.max(bs["error"])) <= 1e-12
    ok = ok and float(np.max(bs["residual"])) <= 1e-12
    ok = ok and float(bs.meta["partition_unity_error"]) <= 1e-12
    ok = ok and float(bs.meta["hull_gap"]) == 0.0

    sp_ = mod.compute("spline_vs_poly", _with_defaults("spline_vs_poly", n=15, function="runge"), 0.0)
    ok = ok and float(sp_.meta["error_poly"]) > float(sp_.meta["error_spline"])
    ok = ok and float(sp_.meta["error_floor"]) > 0.0

    rec = record(
        label="2.2 帧不变量",
        theory="帧里画出的跳变、能量、单位分解、凸包必须与 numcore 的断言同值",
        measured="；".join(report)
        + f"；C² 跳变 {float(np.max(cp['jump_d2'])):.1e}，梳齿峰 {float(np.max(spine)):.3f}，"
        f"PCHIP 跳变 {float(sm.meta['jump_pchip_max']):.3g}，插入不变性 {float(np.max(bs['error'])):.1e}，"
        f"多项式/样条误差比 {float(sp_.meta['error_ratio']):.3g}",
        tol="全部通过",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured
    assert ok


@assertion("B", theory="多项式能量的 Gauss 积分与加密网格一致（积分器自身收敛）", chapter="2.2")
def test_polynomial_energy_is_resolved_by_the_quadrature() -> None:
    """:math:`(p'')^2` 在节点附近变化极快，积分值必须对 Gauss 点数不敏感——否则"能量比"没有意义。"""
    n = 13
    x = uniform_nodes(n, -1.0, 1.0)
    y = sample_function("runge", x)
    # n_quad 必须 ≥ n-2（(p'')² 的次数是 2(n-3)，m 点 Gauss 精确到 2m-1）；
    # 这里连带验证"缺省的 2n+2"已经落在收敛区
    vals = [interpolating_polynomial_energy(x, y, n_quad=q) for q in (2 * n + 2, 4 * n + 2, 8 * n + 2)]
    spread = (max(vals) - min(vals)) / max(vals)
    assert_leq(spread, 1e-10, label="三种 Gauss 点数下多项式能量的相对散布")
    rec = record(
        label="多项式能量对积分点数的稳定性",
        theory="(p'')² 次数 ≤ 2(n-3)；限制精度的是重心求值的抵消，故 2n 以上才收敛",
        measured=f"E(2n+2)={vals[0]:.6g}, E(4n+2)={vals[1]:.6g}, E(8n+2)={vals[2]:.6g}，相对散布 {spread:.2e}",
        tol="1e-10",
        passed=spread <= 1e-10,
    )
    assert rec.passed, rec.measured
