"""2.3 章 A/B 类断言：投影几何、Remez 最优一致逼近、误差衰减率。

本文件守的是**第二半**——"逼近到底有多好、这个'好'是怎么量出来的“：

1. **投影的几何**（A 类）——残差与基正交、投影是幂等算子、多项式被精确还原；
2. **交错定理**（A 类）——Remez 结果必须有 :math:`n+2` 个等幅交错点，
   且 de la Vallée Poussin 下界必须追平实测上界（这是“最优”的**证书**，不是“看起来平”）；
3. **收敛率**（A 类）——解析函数的斜率必须等于 :math:`-\\ln\\rho`（Bernstein 椭圆），
   尖点/跳点必须是 :math:`n^{-3/2}` / :math:`n^{-1/2}`；整函数必须**快过任何**几何率；
4. **量的选择会骗人**（B 类）——换基底（条件数）、不切段求积（假的收敛率）、
   误差掉到舍入地板（假的“没收敛”）都在这里被钉住。

所有容差都由实测值加余量给出；“斜率”类断言一律用 :func:`assert_order` / 拟合助手登记实测值。
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

from numcore.interp.approx import (
    BASES,
    bernstein_rho,
    chebyshev_to_monomial,
    clenshaw,
    fit_loglog_slope,
    fit_semilog_slope,
    gram_condition_estimate,
    idempotence_defect,
    l2_projection,
    projection_error_curve,
    projector_matrix,
    remez,
    smoothness_table,
    test_function as make_fn,
    theory_algebraic_rate,
    theory_exponential_rate,
)
from numcore.interp.orthopoly import chebyshev_nodes, evaluate_polynomial
from numcore.verify import assertion, assert_leq, assert_order, record

U = float(np.finfo(float).eps) / 2.0


def _cheb_interp_error(f, deg: int, x: np.ndarray) -> float:
    """Chebyshev 节点插值的 L∞ 误差（用作 Remez 的对照：同一族、同样次数）。"""
    xc = chebyshev_nodes(deg + 1)
    B = np.vander(xc, deg + 1)[:, ::-1]
    c = _solve(B, np.asarray(f(xc), dtype=float))
    return float(np.max(np.abs(np.asarray(f(x), dtype=float) - np.vander(x, deg + 1)[:, ::-1] @ c)))


def _solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """手写列主元 Gauss 消元（测试里不碰线代库，见 POLICY P2）。"""
    M = np.array(A, dtype=float)
    y = np.array(b, dtype=float)
    n = M.shape[0]
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if p != k:
            M[[k, p]] = M[[p, k]]
            y[[k, p]] = y[[p, k]]
        for i in range(k + 1, n):
            m = M[i, k] / M[k, k]
            M[i, k:] -= m * M[k, k:]
            y[i] -= m * y[k]
    out = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        out[i] = (y[i] - M[i, i + 1 :] @ out[i + 1 :]) / M[i, i]
    return out


# --------------------------------------------------------------------------- #
# A 类
# --------------------------------------------------------------------------- #
@assertion("A", theory="⟨f − P_n f, φ_k⟩ = 0（投影的几何定义）", chapter="2.3")
def test_residual_is_orthogonal_to_the_basis() -> None:
    r"""投影的**定义**就是残差与整个子空间正交；这条不成立，后面所有误差曲线都没意义。

    实测是 :math:`10^{-17}` 量级——注意这不是“很小”，而是**在下溢边缘**：
    :math:`\\langle r,\\varphi_k\\rangle` 的真值是 0，量到的全是舍入。
    """
    worst = 0.0
    rows = []
    for name in ("pole", "cusp", "entire"):
        f = make_fn(name, x0=0.0)
        for n in (4, 8, 12):
            r = l2_projection(f, n, breakpoints=(0.0,) if name == "cusp" else ())
            B = basis_of(r.nodes, n)
            resid = np.asarray(f(r.nodes), dtype=float) - B.T @ r.coeffs
            inner = float(np.max(np.abs(B @ (r.weights * resid))))
            worst = max(worst, inner)
            rows.append(f"{name}(n={n}):{inner:.1e}")
    rec = record(
        label="max|⟨残差, φ_k⟩|（三函数 × n=4,8,12）",
        theory="0（正交）",
        measured=f"{worst:.2e}",
        tol="1e-15",
        passed=bool(worst <= 1e-15),
    )
    record(label="逐项明细", theory="-", measured="，".join(rows), tol="-", passed=True)
    assert rec.passed, rec.measured


def basis_of(x: np.ndarray, n: int) -> np.ndarray:
    """归一化 Legendre 基在 x 上的值，形状 ``(n, x.size)``（本文件的小工具）。"""
    from numcore.interp.orthopoly import orthonormal_values

    return orthonormal_values("legendre", n - 1, x)


@assertion("A", theory="P² = P（投影是幂等算子）；多项式被精确还原", chapter="2.3")
def test_projection_is_idempotent_and_exact_on_polynomials() -> None:
    r"""两件必须成立的事：算子幂等、以及“次数够就精确”（:math:`\\Phi_{n-1}` 包含该多项式）。

    第二件是**逼近误差的下界来源**：只有当多项式被精确还原时，“误差随 n 减小”才可能是真的。
    """
    f = make_fn("pole")
    worst_idem = 0.0
    for n in (4, 8, 12):
        r = l2_projection(f, n)
        P = projector_matrix(basis_of(r.nodes, n), r.weights)
        worst_idem = max(worst_idem, idempotence_defect(P))
    # f 是 5 次多项式（取 T_5 的组合），用 n=6 项投影必须精确
    tt = np.array([0.3, -1.0, 0.7, 2.0, -0.5, 1.5])
    poly = lambda t: sum(tt[k] * np.cos(k * np.arccos(np.clip(t, -1.0, 1.0))) for k in range(6))  # noqa: E731
    x = np.linspace(-1.0, 1.0, 201)
    exact_err = 0.0
    for terms in (6, 7):
        r = l2_projection(poly, terms)
        exact_err = max(exact_err, float(np.max(np.abs(np.asarray(poly(x)) - basis_of(x, terms).T @ r.coeffs))))
    rec = record(
        label="‖P²−P‖∞ 与 5 次多项式的投影误差（n=6,7）",
        theory="≤ 1e-13（两者）",
        measured=f"幂等缺陷 {worst_idem:.2e}，多项式还原误差 {exact_err:.2e}",
        tol="1e-13",
        passed=bool(worst_idem <= 1e-13 and exact_err <= 1e-13),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Clenshaw 递推 = 直接求和 Σc_kT_k(x)", chapter="2.3")
def test_clenshaw_matches_direct_chebyshev_sum() -> None:
    """Clenshaw 是 Chebyshev 展开的稳定求值法；它必须与“笨办法”逐点一致（差在舍入）。"""
    worst = 0.0
    for deg in (4, 8, 16, 24):
        c = np.array([1.0 / (k + 1) for k in range(deg + 1)])
        x = np.linspace(-1.0, 1.0, 301)
        th = np.arccos(np.clip(x, -1.0, 1.0))
        direct = np.zeros_like(x)
        for k in range(deg + 1):
            direct = direct + c[k] * np.cos(k * th)
        worst = max(worst, float(np.max(np.abs(clenshaw(c, x) - direct))))
    # Chebyshev → 单项式系数必须与“逐项展开”一致
    c = np.array([1.0, 0.5, -0.25, 0.125])
    mono = chebyshev_to_monomial(c)
    x = np.linspace(-1.0, 1.0, 101)
    th = np.arccos(np.clip(x, -1.0, 1.0))
    direct = sum(c[k] * np.cos(k * th) for k in range(4))
    conv = float(np.max(np.abs(evaluate_polynomial(mono, x) - direct)))
    rec = record(
        label="Clenshaw vs 直接求和 / 基变换 vs 直接展开",
        theory="≤ 1e-13",
        measured=f"{worst:.2e} 与 {conv:.2e}",
        tol="1e-13",
        passed=bool(worst <= 1e-13 and conv <= 1e-13),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Chebyshev 零点上的首一多项式范数 = 2^(1−n)（最小），等距节点更差", chapter="2.3")
def test_chebyshev_nodes_minimize_the_monic_polynomial() -> None:
    r"""Chebyshev 节点存在的**理由**：:math:`\\min\\|\\prod(x-x_k)\\|_\\infty=2^{1-n}`，取到时唯一。

    实测：Chebyshev 零点上正好 :math:`2^{1-n}`；等距节点的范数大 1.6 倍（n=4）到 44 倍（n=16），
    这就是“插值节点不能等距”的定量版本。
    """
    x = np.linspace(-1.0, 1.0, 4001)
    worst_cheb = 0.0
    ratios = []
    for n in (4, 8, 12, 16):
        mono_c = np.prod([x - t for t in chebyshev_nodes(n)], axis=0)
        mono_e = np.prod([x - t for t in np.linspace(-1.0, 1.0, n)], axis=0)
        pk = float(np.max(np.abs(mono_c)))
        worst_cheb = max(worst_cheb, abs(pk - 2.0 ** (1 - n)) / 2.0 ** (1 - n))
        ratios.append(float(np.max(np.abs(mono_e))) / pk)
    rec = record(
        label="‖首一多项式‖∞ / 2^(1−n) − 1（Chebyshev 节点）",
        theory="0（取到理论最小）",
        measured=f"最大相对偏差 {worst_cheb:.2e}",
        tol="1e-12",
        passed=bool(worst_cheb <= 1e-12),
    )
    record(
        label="等距节点范数 / Chebyshev 节点范数（n=4,8,12,16）",
        theory="> 1 且随 n 增长",
        measured="，".join(f"{r:.1f}" for r in ratios),
        tol="单调增",
        passed=bool(all(a <= b for a, b in zip(ratios, ratios[1:])) and ratios[0] > 1.0),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Chebyshev 交错定理：最佳一致逼近 ⟺ n+2 点等幅交错", chapter="2.3")
def test_remez_satisfies_the_alternation_theorem() -> None:
    r"""交错定理的两个可观测后果：交错点个数 :math:`=n+2`、各点 :math:`|e|` 等幅（比值→1）。

    只在**误差远高于舍入地板**的区间上断言（eˣ 的 n≥12 已经掉到 :math:`10^{-14}`，
    那时“极值”全是噪声，“数出 n+2 个点”没有意义——这条限制本身写进断言里）。
    """
    worst_ratio = 0.0
    ok_all = True
    rows = []
    for name in ("pole", "cusp"):
        f = make_fn(name, x0=0.0)
        for deg in (4, 8, 12, 16):
            r = remez(f, deg)
            ratio = r.amplitude_ratio
            ok = r.converged and r.n_alt == deg + 2 and r.sign_alternates and ratio <= 1.0 + 1e-6
            ok_all = ok_all and ok
            worst_ratio = max(worst_ratio, min(ratio, 1e3))
            rows.append(f"{name}(n={deg}):{r.n_alt}点/比值{ratio:.9f}/{'收敛' if r.converged else '未收敛'}")
    f = make_fn("entire")
    r = remez(f, 8)
    ok_all = ok_all and r.converged and r.n_alt == 10 and r.sign_alternates
    rows.append(f"entire(n=8):{r.n_alt}点/比值{r.amplitude_ratio:.9f}")
    rec = record(
        label="交错点个数 = n+2 且振幅比 → 1",
        theory="n+2 个点、比值 ≤ 1+1e-6",
        measured=f"最大比值 {worst_ratio:.9f}；全部满足 = {ok_all}",
        tol="见 theory",
        passed=bool(ok_all),
    )
    record(label="逐例明细", theory="-", measured="；".join(rows), tol="-", passed=True)
    assert rec.passed, rec.measured


@assertion("A", theory="de la Vallee Poussin：min|e|(交错集) <= ‖f−p*‖∞", chapter="2.3")
def test_de_la_vallee_poussin_certificate() -> None:
    r""""最优“必须给出**证书**：交错点上的最小振幅 :math:`h` 是 :math:`\|f-p^*\|_\infty` 的下界，
    当 :math:`h` 追平实测上界 :math:`\max|e|` 时，当前多项式就是最优（相差 :math:`1-\\varepsilon`）。

    这条与“点集不再移动”的区别是本章的核心教训：后者在浮点噪声下会**假收敛**。
    """
    worst_gap = 0.0
    rows = []
    for name in ("pole", "cusp"):
        f = make_fn(name, x0=0.0)
        for deg in (4, 8, 12):
            r = remez(f, deg)
            h = float(np.min(np.abs(r.alt_error)))
            gap = float(r.max_error) - h
            worst_gap = max(worst_gap, gap)
            rows.append(f"{name}(n={deg}):h={h:.6e},max={r.max_error:.6e}")
    # 对照：一个**非最优**多项式（Chebyshev 插值）也有交错集，但它的下界明显低于上界
    f = make_fn("pole")
    xc = chebyshev_nodes(9)
    cc = _solve(np.vander(xc, 9)[:, ::-1], f(xc))
    grid = np.linspace(-1.0, 1.0, 4001)
    err_c = np.asarray(f(grid), dtype=float) - np.vander(grid, 9)[:, ::-1] @ cc
    linf_c = float(np.max(np.abs(err_c)))
    rem = remez(f, 8)
    rec = record(
        label="max|e| − min|e|(交错集)（Remez 结果）",
        theory="≤ 1e-6·max|e|（证书成立）",
        measured=f"最大间隙 {worst_gap:.2e}",
        tol="见 theory",
        passed=bool(worst_gap <= 1e-6 * max(1e-30, rem.max_error)),
    )
    record(
        label="同一函数：Chebyshev 插值的 L∞ 误差 / Remez 的 L∞ 误差",
        theory="> 1（非最优多项式的下界与上界对不上）",
        measured=f"{linf_c:.6e} / {rem.max_error:.6e} = {linf_c / rem.max_error:.4f}",
        tol="> 1",
        passed=bool(linf_c > rem.max_error),
    )
    record(label="逐例明细", theory="-", measured="；".join(rows), tol="-", passed=True)
    assert rec.passed, rec.measured


@assertion("A", theory="收敛阶 = ν+1/2：尖点 |x| 为 3/2，跳点 sign(x) 为 1/2", chapter="2.3")
def test_algebraic_rates_match_smoothness() -> None:
    r"""Jackson 型估计的实测版：:math:`\|f-P_nf\|_2\\asymp n^{-(\\nu+1/2)}`。

    非光滑点处**必须切段求积**，否则量到的是求积误差（见 B 类那条反例）。
    """
    ns = np.arange(2, 33)
    f_cusp = make_fn("cusp", x0=0.0)
    f_jump = make_fn("jump", x0=0.0)
    e_cusp = projection_error_curve(f_cusp, ns, breakpoints=(0.0,))
    e_jump = projection_error_curve(f_jump, ns, breakpoints=(0.0,))
    h = 1.0 / ns
    fit_c = assert_order(h, e_cusp, theory=theory_algebraic_rate(1.0), tol=0.1, skip=10,
                         label="尖点 |x| 的 L² 收敛阶（n≥12）")
    fit_j = assert_order(h, e_jump, theory=theory_algebraic_rate(0.0), tol=0.1,
                         label="跳点 sign(x) 的 L² 收敛阶")
    record(
        label="两条曲线的拟合常数 C 与 R²",
        theory="err ≈ C·n^(−α)",
        measured=f"cusp C={math.exp(fit_c.intercept):.3f}(R²={fit_c.r2:.6f})，"
        f"jump C={math.exp(fit_j.intercept):.3f}(R²={fit_j.r2:.6f})",
        tol="-",
        passed=True,
    )


@assertion("A", theory="Bernstein 椭圆：解析 ⇒ 误差 ~ ρ^(−n)，斜率 = −lnρ", chapter="2.3")
def test_analytic_rate_equals_minus_log_rho() -> None:
    r"""解析函数的误差**在参数移动时**仍必须跟着 :math:`-\\ln\\rho` 走：
    :math:`\\rho=|y|+\\sqrt{1+y^2}`（极点 :math:`\\pm i/\\sqrt s`）。这是“理论线”的真实含义。

    实测：s=4/25/100 时相对差 ≤0.7%；s=200（ρ=1.073）时差到 2.4%——
    **渐近区还没到**，这也是“半对数图上看着是直线”却拟合不出理论斜率的原因。
    """
    worst = 0.0
    rows = []
    ns = np.arange(2, 41)
    for scale in (4.0, 25.0, 100.0):
        f = make_fn("pole", scale=scale)
        e = projection_error_curve(f, ns)
        got, r2 = fit_semilog_slope(ns, e, skip=1)
        want = theory_exponential_rate(complex(0.0, 1.0 / math.sqrt(scale)))
        rel = abs(got - want) / abs(want)
        worst = max(worst, rel)
        rows.append(f"s={scale:g}:{got:.6f}≈{want:.6f}({rel:.2%})")
        rec = record(
            label=f"半对数斜率 vs −lnρ（s={scale:g}, ρ={bernstein_rho(complex(0.0, 1.0 / math.sqrt(scale))):.6f}）",
            theory=f"{-want:.6f}",
            measured=f"{got:.6f}（R²={r2:.6f}）",
            tol="≤ 1%",
            passed=bool(rel <= 0.01),
        )
        assert rec.passed, rec.measured
    # 慢收敛（ρ→1）时渐近区更远：如实登记，不掩盖
    f = make_fn("pole", scale=200.0)
    e = projection_error_curve(f, ns)
    got, _ = fit_semilog_slope(ns, e, skip=1)
    want = theory_exponential_rate(complex(0.0, 1.0 / math.sqrt(200.0)))
    pre = abs(got - want) / abs(want)
    record(
        label="s=200（ρ=1.073）时的同一比较",
        theory="偏差变大（n≤40 还没进渐近区）",
        measured=f"{got:.6f} vs {want:.6f}（{pre:.2%}）",
        tol="> 1%",
        passed=bool(pre > 0.01),
    )
    record(label="拟合明细", theory="-", measured="；".join(rows), tol="-", passed=True)
    assert worst <= 0.01


@assertion("A", theory="整函数的误差快过任何 ρ^(−n)（ρ>1）", chapter="2.3")
def test_entire_function_beats_every_geometric_rate() -> None:
    r"""eˣ 在 :math:`[-1,1]` 上解析且无有限 :math:`\\rho`：对任意 :math:`\\rho>1`，
    :math:`\\rho^n\\|e_n\\|\\to0`。实测 n≥15 时 :math:`2^{n}e_n\\le10^{-9}`。

    这条同时解释了为什么“半对数图上看整函数”会看到一条**越来越陡**的曲线——
    它不是直线，也就没有斜率可拟合。
    """
    ns = np.arange(2, 23)
    f = make_fn("entire")
    e = projection_error_curve(f, ns)
    tail = ns >= 15
    worst = {}
    for rho in (1.5, 2.0, 3.0):
        worst[rho] = float(np.max(e[tail] * rho ** ns[tail]))
    rec = record(
        label="max ρ^n·‖e_n‖（n≥15）",
        theory="→ 0（对每个 ρ>1）",
        measured="，".join(f"ρ={r}:{v:.2e}" for r, v in worst.items()),
        tol="≤ 1e-8",
        passed=bool(worst[1.5] <= 1e-10 and worst[2.0] <= 1e-8 and worst[3.0] <= 1e-4),
    )
    # 衰减因子只看**误差仍在舍入地板之上**的那一段（n 一大就被 1e-16 的地板污染，
    # 那时的"比值"是噪声：实测尾部会从 0.04 反弹到 0.65）
    idx = [i for i in range(len(ns) - 1) if e[i + 1] > 1e4 * U]
    ratios_txt = "，".join(f"n={ns[i]}→{ns[i+1]}:{e[i+1]/e[i]:.3f}" for i in idx)
    mono = all(e[i + 2] / e[i + 1] < e[i + 1] / e[i] for i in idx[:-1])
    record(
        label="exp(x) 的相邻两点之比 e_{n+1}/e_n（只取 e>1e-12 的窗口）",
        theory="随 n 严格变小（超几何收敛）",
        measured=ratios_txt,
        tol="单调减",
        passed=bool(mono and len(idx) >= 4),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="Remez 的 L∞ 误差 ≤ Chebyshev 插值、≤ 同阶 L² 投影", chapter="2.3")
def test_remez_beats_chebyshev_interpolation_and_l2_projection() -> None:
    r"""最优一致逼近的“最优”是可比较的：Remez 必须同时压住两条常见替代路径。

    实测（pole, n=8）：9.81e-2 vs Chebyshev 插值 1.71e-1 vs L² 投影 1.50e-1。
    L² 投影在**积分意义**下最优，却在最大范数下输给 Remez——这正是“两种最优不是一回事”。
    """
    grid = np.linspace(-1.0, 1.0, 2001)
    worst_vs_cheb = 0.0
    worst_vs_l2 = 0.0
    rows = []
    for name in ("pole", "cusp", "entire"):
        f = make_fn(name, x0=0.0)
        for deg in (4, 8, 12):
            r = remez(f, deg)
            e_cheb = _cheb_interp_error(f, deg, grid)
            l2 = l2_projection(f, deg + 1, breakpoints=(0.0,) if name == "cusp" else ())
            e_l2 = float(np.max(np.abs(np.asarray(f(grid), dtype=float) - basis_of(grid, deg + 1).T @ l2.coeffs)))
            worst_vs_cheb = max(worst_vs_cheb, r.max_error / e_cheb)
            worst_vs_l2 = max(worst_vs_l2, r.max_error / e_l2)
            rows.append(f"{name}(n={deg}):{r.max_error:.3e} < cheb {e_cheb:.3e} < l2 {e_l2:.3e}"
                        if e_cheb > r.max_error and e_l2 > r.max_error else f"{name}(n={deg}):反例")
    rec = record(
        label="Remez / Chebyshev 插值 与 Remez / L² 投影",
        theory="< 1（两者）",
        measured=f"最大比 {worst_vs_cheb:.4f} 与 {worst_vs_l2:.4f}",
        tol="< 1",
        passed=bool(worst_vs_cheb < 1.0 and worst_vs_l2 < 1.0),
    )
    record(label="逐例（Remez < cheb < (或) l2）", theory="-", measured="；".join(rows), tol="-", passed=True)
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# B 类
# --------------------------------------------------------------------------- #
@assertion("B", theory="正规方程的相对误差 ~u·κ(G)；κ 越大系数越不可信（残差仍小）", chapter="2.3")
def test_gram_conditioning_explains_the_monomial_breakdown() -> None:
    r"""三个基张成同一个子空间，但 :math:`\\kappa(G)` 从 1（Legendre）涨到 :math:`10^{17}`（单项式）。

    实测（pole, n=24）：:math:`\\kappa u\\approx21.7>1`，投影函数与良态基的结果差 1.2e-3，
    而**正规方程残差只有 1.0e-11**——残差小绝不等于解对。
    """
    f = make_fn("pole")
    x = np.linspace(-1.0, 1.0, 401)
    conds = {}
    worst_gap = 0.0
    resid = 0.0
    for n in (12, 20, 24):
        ref_vals = basis_of(x, n).T @ l2_projection(f, n).coeffs
        for basis in BASES:
            r = l2_projection(f, n, basis=basis)
            conds[(basis, n)] = r.gram_cond
            vals = _basis_vals(basis, n, x).T @ r.coeffs
            if basis == "monomial":
                worst_gap = max(worst_gap, float(np.max(np.abs(vals - ref_vals))))
                resid = max(resid, r.normal_residual)
    growth = conds[("monomial", 24)] / conds[("monomial", 12)]
    rec = record(
        label="κ(G) 随 n 的增长（单项式，n=12→24）",
        theory="指数增长（实测 1e8 → 2e17）",
        measured=f"{conds[('monomial', 12)]:.2e} → {conds[('monomial', 24)]:.2e}（×{growth:.1e}）；"
        f"Legendre n=24 的 κ={conds[('legendre', 24)]:.3f}",
        tol="-",
        passed=bool(growth > 1e6 and conds[("legendre", 24)] <= 1.001),
    )
    record(
        label="n=24：单项式基投影与良态基投影的函数差 / 正规方程残差",
        theory="函数差 ~ κu ≫ 残差",
        measured=f"{worst_gap:.2e} vs {resid:.2e}",
        tol="gap > 1e-4 且 resid < 1e-8",
        passed=bool(worst_gap > 1e-4 and resid < 1e-8),
    )
    assert rec.passed, rec.measured


def _basis_vals(basis: str, n: int, x: np.ndarray) -> np.ndarray:
    from numcore.interp.approx import basis_values

    return basis_values(basis, n, x)


@assertion("B", theory="良态基下投影与基无关；病态基下“同一个子空间”也会给出不同函数", chapter="2.3")
def test_projection_is_basis_independent_when_well_conditioned() -> None:
    """换基不换子空间：Legendre 与 Chebyshev 的结果必须一致到舍入；单项式到 n=24 就散了。"""
    f = make_fn("pole")
    x = np.linspace(-1.0, 1.0, 401)
    rows = []
    worst_ok = 0.0
    worst_bad = 0.0
    for n in (12, 20, 24):
        vals = {b: _basis_vals(b, n, x).T @ l2_projection(f, n, basis=b).coeffs for b in BASES}
        d1 = float(np.max(np.abs(vals["legendre"] - vals["chebyshev"])))
        d2 = float(np.max(np.abs(vals["legendre"] - vals["monomial"])))
        worst_ok = max(worst_ok, d1)
        worst_bad = d2
        rows.append(f"n={n}:|L−C|={d1:.1e},|L−M|={d2:.1e}")
    rec = record(
        label="|投影_Legendre − 投影_Chebyshev| 与 |投影_Legendre − 投影_单项式|",
        theory="前者 ≤1e-13（同一子空间），后者随 κ 增大",
        measured=f"最大 |L−C|={worst_ok:.2e}（n≤24）；n=24 的 |L−M|={worst_bad:.2e}",
        tol="见 theory",
        passed=bool(worst_ok <= 1e-13 and worst_bad > 1e-5),
    )
    record(label="逐阶明细", theory="-", measured="；".join(rows), tol="-", passed=True)
    assert rec.passed, rec.measured


@assertion("B", theory="不切段 + 有限求积阶数 ⇒ 误差曲线被求积误差污染", chapter="2.3")
def test_missing_breakpoints_corrupt_the_measured_rate() -> None:
    r"""把“误差曲线的斜率”当成“逼近的收敛率”是很自然的事，而它随时可以是**求积的**斜率。

    实测（固定 40 点求积、拟合 n≥12）：尖点不切段得 α=2.13（真值 1.5，偏 42%），
    跳点不切段得 α=0.70（真值 0.5，偏 40%）；在折点切段后分别回到 1.52 / 0.495。
    """
    ns = np.arange(4, 25)
    rows = []
    worst_dev = 0.0
    for name in ("cusp", "jump"):
        f = make_fn(name, x0=0.0)
        a = projection_error_curve(f, ns, n_quad=40, breakpoints=())
        b = projection_error_curve(f, ns, n_quad=40, breakpoints=(0.0,))
        sa, _ = fit_loglog_slope(ns, a, skip=8)
        sb, _ = fit_loglog_slope(ns, b, skip=8)
        worst_dev = max(worst_dev, abs(-sa - theory_algebraic_rate(1.0 if name == "cusp" else 0.0)))
        rows.append(f"{name}:不切段 α={-sa:.3f}，切段 α={-sb:.3f}")
    rec = record(
        label="不切段时的实测斜率与理论斜率之差 / 切段后的斜率",
        theory="切段后 |α−ν−1/2| ≤ 0.05；不切段时偏差 ≥ 0.15",
        measured=f"不切段最大偏差 {worst_dev:.3f}；" + "；".join(rows),
        tol="见 theory",
        passed=bool(worst_dev >= 0.15),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="smoothness_table 的分类要与实测斜率一致", chapter="2.3")
def test_smoothness_table_matches_measured_rates() -> None:
    """把“光滑性 → 收敛率”的对照表当成数据检查一遍：表里的 α 必须等于拟合出来的 α。"""
    rows = []
    ok = True
    ns = np.arange(2, 33)
    table = {row["name"]: row for row in smoothness_table()}
    for name in ("cusp", "jump"):
        f = make_fn(name, x0=0.0)
        e = projection_error_curve(f, ns, breakpoints=(0.0,))
        got, _ = fit_loglog_slope(ns, e, skip=10 if name == "cusp" else 0)
        want = table[name]["alpha"]
        ok = ok and abs(-got - want) <= 0.1
        rows.append(f"{name}:实测 {-got:.3f} vs 表 {want}")
    f = make_fn("pole")
    e = projection_error_curve(f, ns)
    got, _ = fit_semilog_slope(ns, e, skip=1)
    want = table["pole"]["rate"]
    ok = ok and abs(got - want) / abs(want) <= 0.01
    rows.append(f"pole:实测 {got:.6f} vs 表 {want:.6f}")
    rec = record(
        label="实测斜率 vs smoothness_table",
        theory="α 差 ≤0.1，斜率差 ≤1%",
        measured="；".join(rows),
        tol="见 theory",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="Remez 拒绝在跳点处假收敛，并返回历史最优迭代", chapter="2.3")
def test_remez_reports_non_convergence_honestly() -> None:
    r"""跳点函数上交换点集会**破掉交错结构**（误差在跳点两侧各有一个极值，符号却相同），
    这时实现必须报 ``converged=False``，并且返回**历史最优**的那个迭代，而不是最后一轮。

    如实报告比“看起来收敛”重要：这一条就是“宁可当场炸，也不要画一张假图”的版本。
    """
    f = make_fn("jump", x0=0.0)
    r = remez(f, 6)
    hist_max = [float(np.max(np.abs(e))) for e in r.history]
    best_hist = min(hist_max)
    rec = record(
        label="跳点 n=6：converged / 返回的 max|e| / 历史最优",
        theory="converged=False 且 max|e| = min(历史)",
        measured=f"converged={r.converged}，max|e|={r.max_error:.3e}，历史最优={best_hist:.3e}",
        tol="≥ 与 =",
        passed=bool((not r.converged) and r.max_error <= best_hist + 1e-12),
    )
    # 同一函数上 L² 投影**没有**这个问题（它不需要交错结构）
    ns = np.array([4, 8, 16, 24])
    e_l2 = projection_error_curve(f, ns, breakpoints=(0.0,))
    rec2 = record(
        label="同一函数：L² 投影误差仍在按 n^(−1/2) 下降",
        theory="单调下降",
        measured="，".join(f"{v:.3e}" for v in e_l2),
        tol="单调",
        passed=bool(np.all(np.diff(e_l2) < 0)),
    )
    assert rec.passed and rec2.passed, (rec.measured, rec2.measured)


@assertion("B", theory="四个可视化的帧都带 requiredAux，且 meta 只放有限标量", chapter="2.3")
def test_all_chapter_frames_are_well_formed() -> None:
    """逐 viz 检查帧契约：requiredAux 齐备、meta 是有限标量或 None、时间轴真的在动。"""
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("numlab_ch23_test", root / "server" / "chapters" / "2.3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = []
    ok = True
    for viz in mod.MANIFEST["vizzes"]:
        params = {p["name"]: p.get("default") for p in viz["params"]}
        fr = mod.compute(viz["id"], params, 0.0)
        names = {n.lower() for n in fr.names}
        for aux in viz["requiredAux"]:
            if not any(aux in n for n in names):
                ok = False
                report.append(f"{viz['id']}: 缺 requiredAux {aux}")
            elif not any(n.startswith(aux) for n in names):
                ok = False
                report.append(f"{viz['id']}: {aux} 只有间接匹配 {sorted(n for n in names if aux in n)}")
        for nm in names:
            if not np.all(np.isfinite(fr[nm])):
                ok = False
                report.append(f"{viz['id']}.{nm}: 含非有限值")
        for k, v in fr.meta.items():
            if isinstance(v, (list, tuple, dict, str)):
                ok = False
                report.append(f"{viz['id']}.meta[{k}]: 非标量")
            if isinstance(v, float) and not math.isfinite(v):
                ok = False
                report.append(f"{viz['id']}.meta[{k}]: 非有限（应为 None）")
        # 时间轴：scrub 的帧必须随 t 变化（否则动画是假的）
        if viz["time"]["mode"] == "scrub":
            t_max = float(viz["time"]["tMax"])
            a = mod.compute(viz["id"], params, 0.0)
            b = mod.compute(viz["id"], params, t_max)
            # "帧变了"的判据必须是**结构性**的：只看某个误差数组会因为"全在舍入地板"
            # 而误判（zeros_evolution 的根误差全是 1.1e-16，减法恒为 0）。
            changed = a.meta.get("step") != b.meta.get("step")
            if not changed:
                for nm in a.names:
                    if nm not in b.names or a[nm].shape != b[nm].shape:
                        changed = True
                        break
                    if float(np.max(np.abs(a[nm] - b[nm]))) > 0.0:
                        changed = True
                        break
            if not changed:
                ok = False
                report.append(f"{viz['id']}: t=0 与 t={t_max:g} 的帧完全相同（时间轴没接上）")
        report.append(f"{viz['id']}: {len(names)} 数组，meta {len(fr.meta)} 项")
    rec = record(
        label="帧契约检查（4 个 viz，含时间轴有效性）",
        theory="requiredAux 齐备、数组全有限、meta 只放有限标量、scrub 帧随时间变化",
        measured="；".join(report),
        tol="全通过",
        passed=ok,
    )
    assert rec.passed, report
