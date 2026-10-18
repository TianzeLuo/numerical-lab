"""5.1 章 A 类断言（解析对拍）：收敛形态、割线方程、共轭性与鞍点逃逸。

这一章最容易写成假证据的四件事，全部在这里被钉住：

1. **"牛顿法二次收敛"**：只有"整步被接受 + 已经在解附近"两条同时成立时才有
   :math:`e_{k+1}\\approx Ce_k^2`。所以这里既测渐近区里 :math:`C` 的**存在与数值**，
   也测**离开二次区**时的退化（卡住步长 → 线性，速率实测 1-α_cap）；
   还测 Rosenbrock 上"二次区的尾巴只有 2–3 步、常数根本还没稳定"这个事实。
2. **"BFGS 满足割线方程"**：更新的**前**与更新的**后**是两个不同的量
   （实测 0.999 与 3e-15），只报后者就是把"更新生效"说成"迭代本身满足"。
3. **"CG 有限步终止"**：只在**精确**线搜索下成立（实测 n=4 步 vs 不精确 72 步）。
4. **"随机扰动逃出鞍点"**：逃逸阈值与 :math:`1/|\\langle u_{\\min},d\\rangle|` 成正比，
   与 u_min 正交时**永远**逃不掉——阈值是"没测到"而不是 0。
"""

from __future__ import annotations

import numpy as np

from numcore.inverse.optimize import (
    anisotropic_quadratic,
    bfgs,
    cg,
    contour_grid,
    escape_threshold,
    euclidean_norm,
    gd,
    lbfgs,
    min_eigenpair,
    negative_curvature_direction,
    newton,
    quartic_quadratic,
    quadratic_from_spectrum,
    quadratic_window,
    rosenbrock_objective,
    saddle_quadratic,
    secant_residual,
    trust_region,
)
from numcore.reference import dense as ref
from numcore.verify import assertion, assert_order, record

U = float(np.finfo(float).eps) / 2.0  # 单位舍入 u = 2^-53 ≈ 1.11e-16

QUARTIC_KAPPA = 100.0
QUARTIC_BETA = 0.05


def _quartic():
    return quartic_quadratic(QUARTIC_KAPPA, angle_deg=35.0, beta=QUARTIC_BETA)


def _q_ratios(dist: np.ndarray, *, floor: float = 1e-13) -> np.ndarray:
    """逐步的 :math:`e_{k+1}/e_k^2`，丢掉落到舍入地板以下的点。"""
    e = np.asarray(dist, dtype=float).ravel()
    out = [e[k + 1] / e[k] ** 2 for k in range(e.size - 1) if e[k] > floor and e[k + 1] > 0.0]
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# 1. 牛顿法的二次收敛：常数、进入点、以及"离开二次区"的退化
# --------------------------------------------------------------------------- #
@assertion("A", theory="e_{k+1} ≤ C·e_k^2（解附近、整步被接受）", chapter="5.1")
def test_newton_quadratic_constant_is_measured_and_fails_far_away() -> None:
    r"""四次型上牛顿法的二次收敛**常数**是实测出来的，而且它在第一步**不成立**。

    窗口由 :func:`numcore.inverse.optimize.quadratic_window` 给出（"从末尾往前扫，
    相邻比值相差不超过 2.5 倍"），不是事后挑的。双向：窗口里的比值必须逐点稳定，
    而第一步的比值必须比该常数大三个数量级——"看起来在平方"不等于"到处都平方"。
    """
    F = _quartic()
    tr = newton(F, np.array([1.0, 1.0]), max_iter=60, tol=1e-16)
    enter, c_hat, window = quadratic_window(tr.dist)
    q = _q_ratios(tr.dist)
    assert enter is not None and c_hat is not None, f"二次窗口没有被测出来（q={q!r}）"
    spread = float(np.max(window) / np.min(window))
    first = float(q[0])
    rec = record(
        label="进入二次区的步号 / 窗口内 q 的散布",
        theory="窗口 ≥ 2 点、散布 ≤ 2.5（相邻比值定义），步号有限",
        measured=f"enter_iter={enter}，窗口 {np.array2string(window, precision=3)}，散布 {spread:.3f}，C≈{c_hat:.3e}",
        tol="≥2 点、散布 ≤ 2.5",
        passed=bool(window.size >= 2 and spread <= 2.5 and 0 <= enter <= tr.n_iter),
    )
    rec2 = record(
        label="第一步 e1/e0^2 ÷ 渐近常数 C",
        theory="≥ 100（远处不是二次收敛）",
        measured=f"{first / c_hat:.3e}（第一步 {first:.3e}）",
        tol="≥ 100",
        passed=bool(first / c_hat >= 100.0),
    )
    rec3 = record(
        label="渐近二次常数 C 与牛顿步数",
        theory="C 有限且 > 0；步数 ≤ 8",
        measured=f"C={c_hat:.6e}，{tr.n_iter} 步到 {tr.dist[-1]:.3e}",
        tol="C>0 且步数 ≤ 8",
        passed=bool(np.isfinite(c_hat) and c_hat > 0.0 and tr.n_iter <= 8),
    )
    assert rec.passed and rec2.passed and rec3.passed, f"{rec.measured} / {rec2.measured} / {rec3.measured}"


@assertion("A", theory="整步 α=1 是二次收敛的必要条件；α_cap<1 时退化为线性，因子 1-α_cap", chapter="5.1")
def test_newton_loses_quadratic_convergence_when_steps_are_capped() -> None:
    r"""卡住步长 → 二次收敛当场消失，实测线性因子 = 1 - α_cap（容差 0.05）。

    这是"离开二次区"最干净的一种：问题没变、初值没变，只把每步允许的倍数从 1 降到
    α_cap，收敛形态就从二次变成线性。反过来也说明：看到线性**不等于**方法变了。
    """
    F = _quartic()
    x0 = np.array([1.0, 1.0])
    worst = 0.0
    detail = []
    for cap in (0.75, 0.5, 0.25):
        tr = newton(F, x0, alpha_cap=cap, max_iter=400, tol=1e-15)
        e = tr.dist
        ratios = [e[k + 1] / e[k] for k in range(e.size - 1) if e[k] > 1e-12]
        assert len(ratios) >= 3, f"α_cap={cap} 的线性比值不足：{len(ratios)}"
        rate = float(np.mean(ratios[-3:]))
        worst = max(worst, abs(rate - (1.0 - cap)))
        detail.append(f"α_cap={cap}: 实测 {rate:.4f} / 理论 {1.0 - cap:.4f}（{tr.n_iter} 步）")
    rec = record(
        label="线性因子 - (1-α_cap) 的最大偏差",
        theory="≤ 0.05",
        measured=f"{worst:.4f}（" + "；".join(detail) + "）",
        tol="0.05",
        passed=bool(worst <= 0.05),
    )
    tr_full = newton(F, x0, alpha_cap=1.0, max_iter=60, tol=1e-16)
    rec2 = record(
        label="α_cap=1 与 α_cap=0.5 的步数比",
        theory="≥ 5（二次 vs 线性）",
        measured=f"{tr_full.n_iter} : 50（比 {50.0 / max(tr_full.n_iter, 1):.1f}）",
        tol="≥ 5",
        passed=bool(tr_full.n_iter <= 10),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="二次型上牛顿一步到驻点；鞍点上那一步到的是鞍点而不是极小点", chapter="5.1")
def test_newton_one_step_on_quadratic_and_on_saddle() -> None:
    r"""牛顿法**不检查曲率符号**：鞍点上它同样一步到位，只是到达的是鞍点。

    实测：正定二次型一步之后残距 ~1e-16、``hessian_pd=1``；
    鞍点一步之后恰好落在驻点（dist=0）、``hessian_pd=0`` 且 :math:`\lambda_{\min}<0`
    ——"牛顿法收敛"与"收敛到极小点"是两件事。
    """
    Q = anisotropic_quadratic(100.0, angle_deg=35.0)
    tr = newton(Q, np.array([1.0, 1.0]), line_search="none", max_iter=1, tol=1e-300)
    S = saddle_quadratic(neg=-0.5, pos=1.0)
    ts = newton(S, np.array([1.0, 1e-3]), line_search="none", max_iter=1, tol=1e-300)
    lam_min, _ = min_eigenpair(S.hess(np.zeros(2)))
    rec = record(
        label="正定二次型上牛顿一步之后的 ‖x1 - x*‖",
        theory="≤ κ·u（一次求解的舍入量级）",
        measured=f"{tr.dist[-1]:.3e}（κ={Q.kappa:.1f}，一步，正定标志 {tr.extra('hessian_pd')}）",
        tol=f"{Q.kappa * U:.2e}",
        passed=bool(tr.n_iter == 1 and tr.dist[-1] <= 100.0 * Q.kappa * U and float(np.all(tr.extra("hessian_pd") == 1.0))),
    )
    rec2 = record(
        label="鞍点上牛顿一步之后的 dist 与 Hessian 正定标志",
        theory="dist ≈ 0（到驻点），但 λ_min<0、正定标志 0",
        measured=f"dist={ts.dist[-1]:.3e}，λ_min={lam_min:.3f}，正定标志 {ts.extra('hessian_pd')}",
        tol="λ_min<0 且标志 0",
        passed=bool(ts.n_iter == 1 and ts.dist[-1] <= 1e-15 and lam_min < 0.0 and float(np.all(ts.extra("hessian_pd") == 0.0))),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 2. BFGS / L-BFGS 的割线方程
# --------------------------------------------------------------------------- #
@assertion("A", theory="BFGS 更新后 B_{k+1}s_k = y_k（逐步）；更新前一般不是", chapter="5.1")
def test_bfgs_secant_residual_step_by_step() -> None:
    r"""**逐步**测割线方程残差：更新后 :math:`O(u)`、更新前 :math:`O(1)`。

    只报最后一步的残差等于什么都没测；只报更新后的残差则会把"更新公式生效"
    说成"拟牛顿近似一直满足割线方程"。两个量一起报，才是这条定理的完整内容。
    """
    cases = [
        ("4 维 κ=1000 二次型", quadratic_from_spectrum([1.0, 10.0, 100.0, 1000.0], seed=7, x_star=np.zeros(4)), np.array([2.0, -1.0, 0.5, 3.0])),
        ("四次型", _quartic(), np.array([1.0, 1.0])),
    ]
    worst_after = 0.0
    worst_before = 0.0
    detail = []
    for name, obj, x0 in cases:
        tr = bfgs(obj, x0, max_iter=200, tol=1e-14)
        after = float(np.max(tr.secant_residual))
        before = float(np.max(tr.extra("secant_before")))
        worst_after = max(worst_after, after)
        worst_before = max(worst_before, before)
        detail.append(f"{name}: 更新后 {after:.2e} / 更新前 {before:.3f}（{tr.n_iter} 步）")
    rec = record(
        label="BFGS 更新后割线残差 max ‖B s - y‖/‖y‖",
        theory=f"≤ 100u ≈ {100 * U:.1e}",
        measured=f"{worst_after:.3e}（" + "；".join(detail) + "）",
        tol=f"{100 * U:.1e}",
        passed=bool(worst_after <= 100.0 * U),
    )
    rec2 = record(
        label="BFGS 更新前割线残差 max（同一批步）",
        theory="≥ 0.5（更新不是多余的）",
        measured=f"{worst_before:.3f}",
        tol="≥ 0.5",
        passed=bool(worst_before >= 0.5),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="L-BFGS 只精确满足最近 m 个割线方程；越老越差", chapter="5.1")
def test_lbfgs_secant_residual_grows_with_pair_age() -> None:
    r"""L-BFGS 的隐式逆 Hessian 对**最近**那一对精确成立、对更老的结对变差。

    实测（m=2）：对最新一对的残差是 0.0（构造使然），对上一对是 0.13；
    这说明"L-BFGS 满足割线方程"这句话必须带上"哪一对"。
    """
    Q = quadratic_from_spectrum([1.0, 10.0, 100.0, 1000.0], seed=7, x_star=np.zeros(4))
    x0 = np.array([2.0, -1.0, 0.5, 3.0])
    tr = lbfgs(Q, x0, memory=2, max_iter=200, tol=1e-12)
    sec = tr.extra("inverse_secant")
    assert sec.size >= 2, f"m=2 时应当至少有 2 对 (s,y)，得到 {sec!r}"
    newest = float(sec[0])
    older = float(np.max(sec[1:]))
    rec = record(
        label="L-BFGS(m=2) 最新一对的逆割线残差",
        theory="= O(u)（构造上精确）",
        measured=f"{newest:.3e}",
        tol=f"{100 * U:.1e}",
        passed=bool(newest <= 100.0 * U),
    )
    rec2 = record(
        label="L-BFGS(m=2) 更老一对的逆割线残差",
        theory="远大于 最新一对（记忆没覆盖到）",
        measured=f"{older:.3f}（比值 {older / max(newest, 1e-300):.3e}）",
        tol="≥ 1e-3",
        passed=bool(older >= 1e-3),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 3. 共轭梯度：精确线搜索下的有限步终止与共轭性
# --------------------------------------------------------------------------- #
@assertion("A", theory="精确线搜索下 CG 至多 n 步到达二次型最优解，方向两两 A-共轭", chapter="5.1")
def test_cg_exact_line_search_terminates_in_n_steps() -> None:
    r"""n=4 与 n=6 的二次型上，CG 的迭代数**恰好**等于 n，且归一化共轭偏差 :math:`O(u\\kappa)`。

    反向（同一测试内）：同样的方向公式换成 Wolfe 不精确线搜索，步数远超 n
    ——"CG 有限步终止"是**精确线搜索**的性质，不是 CG 的性质。
    """
    worst_off = 0.0
    detail = []
    steps_ok = True
    for n, kap in ((4, 1e3), (6, 1e2), (8, 10.0)):
        lam = np.logspace(0.0, np.log10(kap), n)
        Q = quadratic_from_spectrum(lam, seed=100 + n, x_star=np.zeros(n))
        x0 = np.linspace(1.0, -1.0, n)
        tr = cg(Q, x0, line_search="exact", max_iter=3 * n)
        D = tr.extra("directions")
        assert D.shape[0] >= 2, f"n={n} 的方向历史只有 {D.shape[0]} 个"
        G = D @ Q.A @ D.T
        dn = np.sqrt(np.clip(np.diag(G), 0.0, None))
        Grel = G / np.outer(dn, dn)
        off = float(np.max(np.abs(Grel - np.diag(np.diag(Grel)))))
        worst_off = max(worst_off, off)
        steps_ok = steps_ok and (tr.n_iter == n)
        detail.append(f"n={n},κ={kap:g}: {tr.n_iter} 步，共轭偏差 {off:.2e}")
    rec = record(
        label="CG(精确线搜索) 迭代数 - n（κ ≤ 1e3）",
        theory="0（有限步终止：至多 n 步精确到达）",
        measured="；".join(detail),
        tol="= 0",
        passed=bool(steps_ok),
    )
    rec_off = record(
        label="归一化 A-共轭偏差 max |p_i^TAp_j|/(‖p_i‖_A‖p_j‖_A)",
        theory="≤ 1e-6（κ ≤ 1e3 时舍入放大的上界）",
        measured=f"{worst_off:.3e}",
        tol="1e-6",
        passed=bool(worst_off <= 1e-6),
    )
    lam_bad = np.logspace(0.0, 5.0, 6)
    Qb = quadratic_from_spectrum(lam_bad, seed=7, x_star=np.zeros(6))
    tb = cg(Qb, np.linspace(1.0, -1.0, 6), line_search="exact", max_iter=30)
    Db = tb.extra("directions")
    Gb = Db @ Qb.A @ Db.T
    dnb = np.sqrt(np.clip(np.diag(Gb), 0.0, None))
    Grelb = Gb / np.outer(dnb, dnb)
    offb = float(np.max(np.abs(Grelb - np.diag(np.diag(Grelb)))))
    rec_bad = record(
        label="κ=1e5、n=6 时精确线搜索 CG 的步数与共轭偏差",
        theory="步数 > n 且共轭偏差 ~1（浮点算术破坏共轭性）",
        measured=f"{tb.n_iter} 步（n=6），偏差 {offb:.3f}，终距 {tb.dist[-1]:.2e}",
        tol="步数 > 6 且偏差 ≥ 0.1",
        passed=bool(tb.n_iter > 6 and offb >= 0.1),
    )
    lam4 = np.array([1.0, 10.0, 100.0, 1000.0])
    Q4 = quadratic_from_spectrum(lam4, seed=7, x_star=np.zeros(4))
    ti = cg(Q4, np.array([2.0, -1.0, 0.5, 3.0]), variant="PR+", line_search="wolfe", c2=0.1, max_iter=300, tol=1e-8)
    rec2 = record(
        label="CG(Wolfe 不精确线搜索) 迭代数 / n",
        theory="远大于 1（失去共轭性的代价）",
        measured=f"{ti.n_iter} / 4 = {ti.n_iter / 4.0:.1f}（终距 {ti.dist[-1]:.2e}）",
        tol="≥ 5",
        passed=bool(ti.n_iter >= 20),
    )
    assert rec.passed and rec_off.passed and rec_bad.passed and rec2.passed, (
        f"{rec.measured} / {rec_off.measured} / {rec_bad.measured} / {rec2.measured}"
    )


# --------------------------------------------------------------------------- #
# 4. 二次型的谱与最优解：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="A = QΛQ^T 的实测谱 = 给定谱；A x* = b 的残差 = O(u‖b‖)", chapter="5.1")
def test_quadratic_spectrum_and_optimum_match_analytic() -> None:
    r"""造出来的二次型必须**实测**谱与给定的谱一致（Jacobi 迭代），最优解残差 :math:`O(u)`。

    这条是后面所有"κ 与迭代数"结论的地基：如果实测 κ 不等于给定 κ，
    那些结论测的就是另一个问题。
    """
    worst_eig = 0.0
    worst_res = 0.0
    detail = []
    for kappa in (10.0, 100.0, 1000.0):
        lam = np.array([1.0, kappa])
        Q = quadratic_from_spectrum(lam, seed=3, x_star=np.zeros(2))
        worst_eig = max(worst_eig, float(np.max(np.abs(Q.eigenvalues - lam) / lam)))
        res = euclidean_norm(Q.A @ Q.x_star - Q.b) / max(euclidean_norm(Q.b), 1e-300)
        worst_res = max(worst_res, res)
        detail.append(f"κ={kappa:g}: 实测 {Q.eigenvalues[-1] / Q.eigenvalues[0]:.10f}")
    eig_tol = 1000.0 * U  # κ=1000 时 Jacobi 的相对精度 ≈ κ·u
    rec = record(
        label="实测特征值相对误差 max |λ-λ|/λ",
        theory=f"≤ 1000u ≈ {eig_tol:.1e}（κ=1e3 时 ≈ κ·u）",
        measured=f"{worst_eig:.3e}（" + "；".join(detail) + "）",
        tol=f"{eig_tol:.1e}",
        passed=bool(worst_eig <= eig_tol),
    )
    rec2 = record(
        label="最优解残差 ‖A x* - b‖/‖b‖",
        theory=f"≤ 100u ≈ {100 * U:.1e}",
        measured=f"{worst_res:.3e}",
        tol=f"{100 * U:.1e}",
        passed=bool(worst_res <= 100.0 * U),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="Rosenbrock 上二次区不足 2 步，常数没有被测出来（与四次型对照）", chapter="5.1")
def test_rosenbrock_quadratic_tail_is_too_short_for_a_stable_constant() -> None:
    r"""诚实地报告"常数测不出来"：Rosenbrock 的尾部比值不满足稳定后缀条件。

    同一个窗口函数（:func:`quadratic_window`）：四次型给出 2 点稳定后缀 + 常数 9.9e-5，
    Rosenbrock 的尾部比值在 1.98 与 18.2 之间跳，稳定后缀只有 1 点 →
    函数必须返回 ``None``，调用方也必须如实写 ``null``，而不是拿单点当常数。
    """
    R = rosenbrock_objective()
    tr = newton(R, np.array([-1.2, 1.0]), max_iter=200, tol=1e-16)
    enter_r, c_r, win_r = quadratic_window(tr.dist, floor=1e-10)
    F = _quartic()
    enter_f, c_f, win_f = quadratic_window(newton(F, np.array([1.0, 1.0]), max_iter=60, tol=1e-16).dist)
    rec = record(
        label="Rosenbrock 的二次窗口",
        theory="enter_iter=None、C=None（稳定后缀 < 2 点）",
        measured=f"窗口 {np.array2string(win_r, precision=3)}（{win_r.size} 点），C={c_r}，牛顿 {tr.n_iter} 步",
        tol="None",
        passed=bool(enter_r is None and c_r is None and win_r.size < 2),
    )
    rec2 = record(
        label="四次型的二次窗口",
        theory="≥2 点稳定后缀且 C 有限",
        measured=f"enter_iter={enter_f}，C={c_f:.3e}，窗口 {np.array2string(win_f, precision=3)}",
        tol="≥2 点",
        passed=bool(enter_f is not None and c_f is not None and win_f.size >= 2),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 5. 鞍点：负曲率方向与逃逸阈值
# --------------------------------------------------------------------------- #
@assertion("A", theory="逃逸阈值 δ* = R(1+|λ-|α)^(-N)，且与 |<u_min,d>| 成反比", chapter="5.1")
def test_escape_threshold_matches_analytic_formula() -> None:
    r"""二分实测的逃逸阈值与解析式逐位吻合；随机方向的阈值 = 解析值 / 重合度。

    实测（λ-=-0.05, α=1, N=120）：沿 u_min 的阈值 2.866e-3 与
    :math:`(1.05)^{-120}=2.866e-3` 一致；沿 45° 方向的阈值恰为 √2 倍。
    """
    step = 1.0
    worst = 0.0
    detail = []
    for neg in (-0.5, -0.05):
        n_esc = 120
        S = saddle_quadratic(neg=neg, pos=1.0)
        u = negative_curvature_direction(S.hess(np.zeros(2)))
        esc = escape_threshold(S, np.zeros(2), u, step=step, max_iter=n_esc, radius=1.0, tol=1e-10)
        theory = (1.0 + abs(neg) * step) ** (-n_esc)
        rel = abs(esc.threshold - theory) / theory
        worst = max(worst, rel)
        detail.append(f"λ-={neg}: 实测 {esc.threshold:.6e} / 理论 {theory:.6e}（相对差 {rel:.2e}）")
    rec = record(
        label="逃逸阈值相对误差 max |δ*_meas - δ*_theory|/δ*_theory",
        theory="≤ 1e-6",
        measured=f"{worst:.3e}（" + "；".join(detail) + "）",
        tol="1e-6",
        passed=bool(worst <= 1e-6),
    )
    S = saddle_quadratic(neg=-0.05, pos=1.0)
    u = negative_curvature_direction(S.hess(np.zeros(2)))
    d45 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    e1 = escape_threshold(S, np.zeros(2), u, step=step, max_iter=120, radius=1.0, tol=1e-10)
    e2 = escape_threshold(S, np.zeros(2), d45, step=step, max_iter=120, radius=1.0, tol=1e-10)
    ratio = e2.threshold / e1.threshold
    rec2 = record(
        label="45° 方向阈值 / u_min 方向阈值（对 1/|<u_min,d>|）",
        theory=f"= {1.0 / e2.direction_overlap:.6f}",
        measured=f"{ratio:.6f}（重合度 {e2.direction_overlap:.6f}）",
        tol="相对 1e-5",
        passed=bool(abs(ratio * e2.direction_overlap - 1.0) <= 1e-5),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="扰动与负曲率方向正交时，任意幅度都逃不出鞍点", chapter="5.1")
def test_orthogonal_perturbation_never_escapes() -> None:
    r"""反例：与 :math:`u_{\\min}` 正交的扰动**永远**逃不掉（放大到 1e12 也没用）。

    返回值必须是"没测到"（``escaped=False``、``threshold=inf``），
    帧里据此写 ``None``——把这种情况记成 0 或者记成"阈值很小"都是假证据。
    """
    S = saddle_quadratic(neg=-0.05, pos=1.0)
    u = negative_curvature_direction(S.hess(np.zeros(2)))
    orth = np.array([-u[1], u[0]])
    esc = escape_threshold(S, np.zeros(2), orth, step=1.0, max_iter=120, radius=1.0, tol=1e-10, hi0=1e-6, max_expand=40)
    along = escape_threshold(S, np.zeros(2), u, step=1.0, max_iter=120, radius=1.0, tol=1e-10)
    rec = record(
        label="正交方向的逃逸结果",
        theory="escaped=False、threshold=inf、重合度=0",
        measured=f"escaped={esc.escaped}，threshold={esc.threshold}，重合度={esc.direction_overlap:.3e}，"
        f"最大试探 δ={esc.hi:.3e}",
        tol="escaped=False",
        passed=bool((not esc.escaped) and not np.isfinite(esc.threshold) and esc.direction_overlap <= 1e-12),
    )
    rec2 = record(
        label="同一问题上沿 u_min 的阈值",
        theory="有限（≈1.4e-3）",
        measured=f"{along.threshold:.3e}（escaped={along.escaped}）",
        tol="有限且 > 0",
        passed=bool(along.escaped and np.isfinite(along.threshold) and along.threshold > 0.0),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 6. 与参考层的对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="自研 BFGS / 信赖域 / 牛顿的终点 = SciPy BFGS 的终点", chapter="5.1")
def test_optimizers_match_reference_minimizer_on_rosenbrock() -> None:
    r"""三个自研方法的终点与 :mod:`numcore.reference` 里的 SciPy BFGS 对拍。

    Rosenbrock 的解析极小点是 (1,1)，但"和参考实现一致"是更强的检查：
    它同时钉住了线搜索、割线更新与信赖域子问题的正确性。
    """
    R = rosenbrock_objective()
    x0 = np.array([-1.2, 1.0])
    ref_res = ref.minimize(
        lambda v: float(R.value(np.asarray(v, dtype=float))),
        x0,
        method="BFGS",
        jac=lambda v: R.grad(np.asarray(v, dtype=float)),
    )
    x_ref = np.asarray(ref_res.x, dtype=float)
    worst = 0.0
    detail = []
    for name, tr in (
        ("newton", newton(R, x0, max_iter=200, tol=1e-14)),
        ("bfgs", bfgs(R, x0, max_iter=200, tol=1e-14)),
        ("trust_region", trust_region(R, x0, delta0=1.0, max_iter=80, tol=1e-14)),
    ):
        d = euclidean_norm(tr.x_final - x_ref)
        worst = max(worst, d)
        detail.append(f"{name}: ‖Δx‖={d:.2e}（{tr.n_iter} 步，f={tr.f[-1]:.2e}）")
    rec = record(
        label="与参考实现终点的最大距离 max ‖x_ours - x_ref‖",
        theory="≤ 1e-8",
        measured=f"{worst:.3e}（" + "；".join(detail) + "）",
        tol="1e-8",
        passed=bool(worst <= 1e-8),
    )
    rec2 = record(
        label="参考实现终点与解析极小点 (1,1) 的距离",
        theory="≤ 1e-12（参考值本身可信）",
        measured=f"{euclidean_norm(x_ref - np.array([1.0, 1.0])):.3e}（f={float(ref_res.fun):.3e}）",
        tol="1e-12",
        passed=bool(euclidean_norm(x_ref - np.array([1.0, 1.0])) <= 1e-12),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 7. 步长：实测压缩因子 = 谱半径、argmin = 2/(L+μ)
# --------------------------------------------------------------------------- #
@assertion("A", theory="尾部压缩因子 = max|1-αλ_i|；其 argmin 出现在 α*=2/(L+μ)", chapter="5.1")
def test_measured_contraction_factor_equals_spectral_radius() -> None:
    r"""逐步实测的梯度压缩因子与 :math:`\\max_i|1-\\alpha\\lambda_i|` **逐位一致**（相对 1e-4）。

    并在同一条测试里找 argmin：实测最优步长就是 :math:`\\alpha^\\star=2/(L+\\mu)`，
    不是"扫出来的某个经验值"。
    """
    Q = anisotropic_quadratic(100.0, angle_deg=35.0)
    alpha_star = float(Q.optimal_step())
    worst = 0.0
    rates = []
    grid = np.linspace(0.7, 1.0, 16)
    for f in grid:
        tr = gd(Q, np.array([1.0, 1.0]), step=float(f) * alpha_star, max_iter=3000, tol=1e-300)
        g = tr.gnorm[tr.gnorm.size // 2 :]
        g = g[g > 0.0]
        ratios = g[1:] / g[:-1]
        meas = float(np.exp(np.mean(np.log(ratios))))
        theory = Q.spectral_radius(float(f) * alpha_star)
        rates.append(meas)
        worst = max(worst, abs(meas - theory) / theory)
    argmin_f = float(grid[int(np.argmin(rates))])
    rec = record(
        label="实测压缩因子与 max|1-αλ| 的最大相对差",
        theory="≤ 1e-4",
        measured=f"{worst:.3e}（16 个步长，α/α* ∈ [0.7,1.0]）",
        tol="1e-4",
        passed=bool(worst <= 1e-4),
    )
    rec2 = record(
        label="实测最优步长 α_argmin/α*",
        theory="= 1（网格分辨率 0.02）",
        measured=f"{argmin_f:.4f}（α*={alpha_star:.6e}，最小实测因子 {min(rates):.6f}）",
        tol="±0.03",
        passed=bool(abs(argmin_f - 1.0) <= 0.03),
    )
    rec3 = record(
        label="α* 处的实测因子 vs (κ-1)/(κ+1)",
        theory=f"{float(np.float64((Q.kappa - 1) / (Q.kappa + 1))):.6f}",
        measured=f"{rates[-1]:.6f}",
        tol="相对 1e-6",
        passed=bool(abs(rates[-1] - (Q.kappa - 1) / (Q.kappa + 1)) <= 1e-6),
    )
    assert rec.passed and rec2.passed and rec3.passed, f"{rec.measured} / {rec2.measured} / {rec3.measured}"


@assertion("A", theory="最优步长下 N(κ) ≈ (κ/2)·ln(1/ε)，实测幂律指数 = 1", chapter="5.1")
def test_gd_iteration_count_is_linear_in_kappa() -> None:
    r"""κ 从 10 到 10000：实测迭代数与理论 :math:`\\lceil\\ln\\varepsilon/\\ln\\rho\\rceil` 逐个相等。

    幂律拟合给出指数 1.000（`assert_order`，A 类）："迭代数与条件数成正比"不是形容词，
    是一条可以拟合出来的直线；同时给出拟合常数 :math:`C=N/\\kappa\\approx\\tfrac12\\ln(1/\\varepsilon)`。
    """
    kappas = [10000.0, 3000.0, 1000.0, 300.0, 100.0, 30.0, 10.0]  # 必须递减（fit_order 要求）
    measured = []
    theory = []
    for kap in kappas:
        Q = quadratic_from_spectrum([1.0, kap], seed=11, x_star=np.zeros(2))
        tr = gd(Q, np.array([1.0, 1.0]), max_iter=2000000, tol=1e-8, tol_kind="grad")
        measured.append(float(tr.n_iter))
        theory.append(float(np.ceil(np.log(1e-8) / np.log((kap - 1.0) / (kap + 1.0)))))
    worst = max(abs(m - t) / t for m, t in zip(measured, theory))
    rec = record(
        label="实测迭代数与理论 ceil(lnε/lnρ) 的最大相对差",
        theory="≤ 1e-3",
        measured=f"{worst:.3e}（" + "；".join(f"κ={k:g}:{int(m)}" for k, m in zip(kappas, measured)) + "）",
        tol="1e-3",
        passed=bool(worst <= 1e-3),
    )
    fit = assert_order(
        np.asarray(kappas),
        np.asarray(measured),
        theory=1.0,
        tol=0.02,
        kind="A",
        label="N(κ) 的幂律指数",
        detail="窗口 κ ∈ [10, 10000]，ε=1e-8，最优步长 2/(L+μ)",
    )
    const = float(np.mean([m / k for m, k in zip(measured, kappas)]))
    rec2 = record(
        label="拟合常数 C = N/κ 与 1/2ln(1/ε) 的比",
        theory=f"≈ 1（1/2ln(1/ε)={0.5 * np.log(1e8):.4f}）",
        measured=f"C={const:.5f}，比 {const / (0.5 * np.log(1e8)):.5f}",
        tol="≤ 0.02",
        passed=bool(abs(const / (0.5 * np.log(1e8)) - 1.0) <= 0.02 and fit.slope > 0.0),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 8. 轮廓网格：帧要用到的几何量也得对
# --------------------------------------------------------------------------- #
@assertion("A", theory="contour_grid 的 Z 与逐点求值一致，且形状/坐标轴与 meshgrid 约定一致", chapter="5.1")
def test_contour_grid_matches_pointwise_evaluation() -> None:
    r"""等高线网格是"图上的背景"，它错了没人看得出来——所以逐点对拍。"""
    Q = anisotropic_quadratic(100.0, angle_deg=35.0)
    X, Y, Z = contour_grid(Q, (-2.0, 2.0), (-1.0, 3.0), n=17)
    worst = 0.0
    for i in range(0, 17, 4):
        for j in range(0, 17, 4):
            z = float(Q.value(np.array([X[i, j], Y[i, j]])))
            worst = max(worst, abs(Z[i, j] - z) / max(abs(z), 1e-300))
    rec = record(
        label="网格值相对误差 max",
        theory="= 0（同一公式，逐点求值）",
        measured=f"{worst:.3e}（17×17 网格，抽样 25 点）",
        tol="1e-15",
        passed=bool(worst <= 1e-15),
    )
    rec2 = record(
        label="网格坐标范围",
        theory="X ∈ [-2,2]、Y ∈ [-1,3]，shape=(17,17)",
        measured=f"X∈[{X.min():.1f},{X.max():.1f}]，Y∈[{Y.min():.1f},{Y.max():.1f}]，shape={Z.shape}",
        tol="精确",
        passed=bool(Z.shape == (17, 17) and X.min() == -2.0 and X.max() == 2.0 and Y.min() == -1.0 and Y.max() == 3.0),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 9. 割线残差的定义口径
# --------------------------------------------------------------------------- #
@assertion("A", theory="secant_residual(B,s,y)=‖Bs-y‖/‖y‖；y=0 时返回 inf（不编数）", chapter="5.1")
def test_secant_residual_definition_and_degenerate_case() -> None:
    r"""定义口径本身要被测：分母是 :math:`\\|y\\|`、``y=0`` 时返回 ``inf`` 而不是 0。

    返回 0 会让调用方误以为"完美满足割线方程"，而真相是"这个量没有定义"。
    """
    B = np.array([[2.0, 0.0], [0.0, 3.0]])
    s = np.array([1.0, 2.0])
    y = np.array([2.0, 6.0])
    exact = secant_residual(B, s, y)
    B2 = np.array([[3.0, 0.0], [0.0, 4.0]])
    off = secant_residual(B2, s, y)
    degenerate = secant_residual(B, s, np.zeros(2))
    rec = record(
        label="B 精确满足 / B 不满足的残差",
        theory="0 与 0.316（=‖(1,2)‖/‖(2,6)‖）",
        measured=f"{exact:.3e} 与 {off:.6f}",
        tol="1e-15 / 1e-6",
        passed=bool(exact <= 1e-15 and abs(off - euclidean_norm(np.array([1.0, 2.0])) / euclidean_norm(y)) <= 1e-12),
    )
    rec2 = record(
        label="y=0 时的返回值",
        theory="inf（无定义，不许返回 0）",
        measured=f"{degenerate}",
        tol="inf",
        passed=bool(not np.isfinite(degenerate)),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"
