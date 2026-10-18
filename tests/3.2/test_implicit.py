"""3.2 章（下半）：隐式方法、Newton 迭代与刚性代价的 A/C 类断言。

这一半谈的是「稳了以后要付多少钱」：

* 每一步的线性系统**必须**走自研 LU（:func:`numcore.linalg.lu.lu_decompose` /
  :func:`~numcore.linalg.lu.solve_lu`），所以有一条断言专门把「Newton + LU 真算出来的解」
  与「一步放大因子的闭式递推」逐点对拍——两条完全不同的代码路径；
* 步长天花板不是抄来的：:func:`~numcore.ode.implicit.max_stable_step` 在
  :math:`|R(\\lambda h)|=1` 上二分，再拿真跑出来的积分验证「跨过天花板就真的炸」；
* 刚性比与容差的两条阈值都用 :func:`~numcore.verify.find_threshold` 二分实测，
  理论值只用来对账。
"""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Any

import numpy as np
import pytest

from numcore.linalg.lu import lu_decompose, solve_lu
from numcore.ode import implicit as imp
from numcore.ode import stability as stab
from numcore.verify import (
    assert_leq,
    assert_order,
    assert_threshold,
    assertion,
    find_threshold,
    record,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import nlf as nlf_mod  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.registry import Registry  # noqa: E402

U = float(np.finfo(float).eps) / 2.0


def _frac_cubic() -> tuple[object, object]:
    """Newton 的两个测试函数：f(z)=z³−3z²−2（会冲过头）与 f(z)=z³−1（在 z=0 处 J 奇异）。"""
    over = (
        lambda z: np.array([z[0] ** 3 - 3.0 * z[0] ** 2 - 2.0]),
        lambda z: np.array([[3.0 * z[0] ** 2 - 6.0 * z[0]]]),
    )
    singular = (
        lambda z: np.array([z[0] ** 3 - 1.0]),
        lambda z: np.array([[3.0 * z[0] ** 2]]),
    )
    return over, singular


# --------------------------------------------------------------------------- #
# A 类：Newton 迭代与 LU
# --------------------------------------------------------------------------- #
@assertion("A", "线性问题每步 Newton 一次迭代即到舍入底（残余 ~u），LU 每步一次", chapter="3.2")
def test_newton_converges_in_one_iteration_for_linear_problem() -> None:
    lam = -1000.0
    sol = imp.integrate(
        "implicit_euler",
        imp.stiff_cos_rhs(lam),
        imp.stiff_cos_jac(lam),
        (0.0, 1.0),
        np.array([1.0]),
        0.1,  # 步长远超显式天花板 2e-3：隐式照样一步一个线性系统
        exact=imp.stiff_cos_exact,
    )
    per_step = sol.n_newton_iters / max(sol.n_steps, 1)
    ok = abs(per_step - 1.0) < 1e-12 and sol.n_lu == sol.n_steps and sol.max_newton_residual < 1e-13
    assert ok, f"每步迭代 {per_step:.6f}，LU {sol.n_lu}/{sol.n_steps} 步，最大残余 {sol.max_newton_residual:.3e}"
    record(
        label="线性隐式步的 Newton 迭代数与 LU 次数",
        theory="每步 1 次迭代、1 次 LU（线性问题 Newton 一步到位）",
        measured=f"{sol.n_steps} 步 → 迭代 {sol.n_newton_iters} 次、LU {sol.n_lu} 次，"
        f"最大残余 {sol.max_newton_residual:.3e}，整段误差 {sol.max_error:.3e}",
        tol="每步 1 次迭代；残余 < 1e-13",
        passed=ok,
        kind="A",
        extra={"per_step": per_step, "residual": sol.max_newton_residual},
    )


@assertion(
    "A",
    "残差回溯把一次本会失败的 Newton 迭代救回来：f(z)=z³−3z²−2 从 z=−1 出发，未阻尼 25 步不收敛，阻尼后 7 步收敛",
    chapter="3.2",
)
def test_newton_backtracking_rescues_a_diverging_iteration() -> None:
    (res, jac), _ = _frac_cubic()
    z0 = np.array([-1.0])
    _zu, ru = imp.newton_solve(res, jac, z0, tol=1e-12, max_iter=25, damping=False)
    zd, rd = imp.newton_solve(res, jac, z0, tol=1e-12, max_iter=25, damping=True)
    ok = (not ru.converged) and rd.converged and rd.n_backtracks > 0
    assert ok, f"未阻尼 converged={ru.converged}(残余 {ru.residual:.3e})，阻尼 converged={rd.converged}({rd.iterations} 步，回溯 {rd.n_backtracks})"
    record(
        label="阻尼 vs 未阻尼的 Newton",
        theory="整步冲过头时回溯能把残差单调压下去；未阻尼会来回震荡",
        measured=f"未阻尼：{ru.iterations} 步未收敛（残余 {ru.residual:.3e}）；"
        f"阻尼：{rd.iterations} 步收敛，回溯 {rd.n_backtracks} 次，残余 {rd.residual:.3e}，根 {zd[0]:.12f}",
        tol="未阻尼不收敛且阻尼收敛",
        passed=ok,
        kind="A",
        extra={"undamped_residual": ru.residual, "damped_iterations": rd.iterations, "backtracks": rd.n_backtracks},
    )


@assertion("A", "Jacobi 奇异时 Newton 报告 singular 而不是抛异常（隐式步随后缩步长重试）", chapter="3.2")
def test_newton_reports_singular_jacobian_instead_of_crashing() -> None:
    _over, (res, jac) = _frac_cubic()
    z, rep = imp.newton_solve(res, jac, np.array([0.0]), tol=1e-12, max_iter=25, damping=True)
    ok = rep.singular and (not rep.converged)
    assert ok, f"singular={rep.singular} converged={rep.converged} residual={rep.residual:.3e} z={z}"
    record(
        label="奇异 Jacobi 的处理",
        theory="z=0 处 f'(z)=0：必须报 singular=True/converged=False，不许抛异常",
        measured=f"singular={rep.singular}，converged={rep.converged}，迭代 {rep.iterations} 次，残余 {rep.residual:.3e}",
        tol="标志位为真且未收敛",
        passed=ok,
        kind="A",
    )


@assertion(
    "A",
    "Newton+LU 的隐式积分与闭式放大因子递推 y_{n+1}=R(λh)y_n 逐步一致（两条独立路径）",
    chapter="3.2",
)
def test_lu_solve_path_matches_amplification_recursion() -> None:
    lam = -50.0
    t_end, h = 1.0, 0.01
    n = int(round(t_end / h))
    rows = []
    worst = 0.0
    for method, order in (("implicit_euler", 1), ("trapezoidal", 2), ("bdf2", 2)):
        sol = imp.integrate(method, imp.linear_rhs(lam), imp.linear_jac(lam), (0.0, t_end), np.array([1.0]), h)
        # 第二条路径：**标量差分方程**，完全不碰 Newton/LU。
        # 单步法 y_{n+1}=R(λh)y_n 是等比数列；BDF2 是三项递推，它的解不是等比
        # （只有在 λh=−0.5 那种重根点上才退化成 (A+Bn)rⁿ），所以逐项滚一遍才对。
        zz = complex(lam * h)
        if method == "bdf2":
            r0 = complex(np.atleast_1d(stab.stability_function("implicit_euler", 1, zz))[0])
            seq = [1.0 + 0.0j, r0]
            for _k in range(2, n + 1):
                seq.append((2.0 * seq[-1] - 0.5 * seq[-2]) / (1.5 - zz))
            closed = np.asarray(seq, dtype=complex)
        else:
            r = complex(np.atleast_1d(stab.stability_function(method, order, zz))[0])
            closed = r ** np.arange(n + 1)
        got = np.asarray(sol.y, dtype=float).reshape(-1)
        d = float(np.max(np.abs(got - np.real(closed))))
        worst = max(worst, d)
        rows.append(f"{method}:{d:.2e}({sol.n_steps}步/{sol.n_lu}次LU)")
    assert_leq(worst, 1e-11, label="LU 路径 vs 放大因子递推")
    record(
        label="隐式 LU 路径与闭式递推的最大差",
        theory="0（两条独立路径：Newton+LU 解方程 vs 标量差分方程递推）",
        measured=f"最大 {worst:.3e}；{'; '.join(rows)}",
        tol="1e-11",
        passed=worst <= 1e-11,
        kind="A",
    )


# --------------------------------------------------------------------------- #
# A 类：阶数（实测收敛率）
# --------------------------------------------------------------------------- #
@assertion("A", "隐式 Euler 一阶、梯形法二阶、BDF2 二阶（实测收敛率）", chapter="3.2")
def test_implicit_methods_show_their_theoretical_orders() -> None:
    rows = []
    fits = {}
    # 窗口按方法选：BDF2 的启动步把误差常数抬高，粗窗口里量到的是 1.88（还没进渐近区），
    # 所以它的窗口取更细的 (7..12)——这个"窗口依赖"本身就是一条要写进陷阱的实测事实。
    plan = (
        ("implicit_euler", 1, "隐式 Euler", (3, 4, 5, 6, 7, 8)),
        ("trapezoidal", 2, "梯形法", (3, 4, 5, 6, 7, 8)),
        ("bdf2", 2, "BDF2", (7, 8, 9, 10, 11, 12)),
    )
    coarse_bdf2 = None
    for method, order, name, window in plan:
        hs = np.array([1.0 / 2**k for k in window])
        err = []
        for h in hs:
            sol = imp.integrate(
                method,
                imp.linear_rhs(-1.0),
                imp.linear_jac(-1.0),
                (0.0, 1.0),
                np.array([1.0]),
                float(h),
                exact=imp.linear_exact(-1.0),
            )
            err.append(sol.max_error)
        fit = assert_order(hs, np.asarray(err), theory=float(order), label=f"{name}的实测收敛阶", detail=f"y'=−y 到 t=1，整段最大误差；窗口 k={window}")
        fits[name] = fit
        rows.append(f"{name}:{fit.slope:.4f}(k={window[0]}..{window[-1]})")
        if method == "bdf2":
            hs_c = np.array([1.0 / 2**k for k in (3, 4, 5, 6, 7, 8)])
            err_c = []
            for h in hs_c:
                sol = imp.integrate(
                    "bdf2", imp.linear_rhs(-1.0), imp.linear_jac(-1.0), (0.0, 1.0), np.array([1.0]), float(h), exact=imp.linear_exact(-1.0)
                )
                err_c.append(sol.max_error)
            coarse_bdf2 = float(__import__("numcore.verify", fromlist=["fit_order"]).fit_order(hs_c, np.asarray(err_c)).slope)
    record(
        label="三个隐式方法的实测阶",
        theory="隐式 Euler 1、梯形 2、BDF2 2",
        measured="；".join(rows) + f"；BDF2 在粗窗口 (3..8) 只能量到 {coarse_bdf2:.4f}（启动步抬高常数）",
        tol="±0.1（assert_order 缺省）",
        passed=True,
        kind="A",
        extra={k: v.slope for k, v in fits.items()},
    )


@assertion(
    "A",
    "显式步长天花板实测 h*=c/|λ|：前向 Euler c=2、RK4 c=2.785293…；隐式方法没有天花板",
    chapter="3.2",
)
def test_explicit_step_ceiling_matches_theory() -> None:
    lam = -1000.0
    rows = []
    ok = True
    for method, order, c in (("forward_euler", 1, 2.0), ("rk4", 4, 2.785293563405282)):
        h_meas = imp.max_stable_step(method, order, lam)
        h_theory = imp.explicit_step_ceiling_theory(method, order, lam)
        rel = abs(h_meas - h_theory) / h_theory
        ok = ok and rel < 1e-12
        rows.append(f"{method}: 实测 {h_meas:.15e} vs 理论 {h_theory:.15e}（相对差 {rel:.1e}）")
    h_imp = imp.max_stable_step("implicit_euler", 1, lam)
    steps = imp.stability_limited_steps("forward_euler", 1, lam, 2.0 * math.pi)
    theory_steps = int(math.ceil(2.0 * math.pi / (2.0 / 1000.0)))
    ok = ok and math.isinf(h_imp) and steps == theory_steps
    assert ok, rows + [f"隐式天花板 {h_imp}", f"步数 {steps} vs {theory_steps}"]
    record(
        label="显式/隐式步长天花板",
        theory="2/|λ| 与 2.785293563405282/|λ|；隐式无界",
        measured="；".join(rows) + f"；隐式 {h_imp}；一个周期内显式需 {steps} 步（理论 {theory_steps}）",
        tol="相对差 < 1e-12",
        passed=ok,
        kind="A",
    )


@assertion(
    "A",
    "跨过实测天花板以后显式解真的炸：h=0.999h* 有界、h=1.02h* 增长 674 倍、h=1.1h* 撞上 |y|>1e12 的爆炸判据",
    chapter="3.2",
)
def test_step_ceiling_is_where_the_explicit_solution_really_blows_up() -> None:
    n = 20
    A = imp.stiff_diffusion_matrix(n)
    y0 = np.sin(np.linspace(0.1, 1.0, n))
    bound = imp.stiff_spectrum_bound(n)
    hc = imp.max_stable_step("forward_euler", 1, -bound)
    scale = float(np.max(np.abs(y0)))
    growth = {}
    for fac in (0.999, 1.02, 1.10):
        sol = imp.integrate_linear_system("forward_euler", A, (0.0, 0.5), y0, hc * fac)
        growth[fac] = (sol.max_abs_y / scale, sol.blowup, sol.n_steps)
    g_ok, blow_ok, _ = growth[0.999]
    g_mid, _, _ = growth[1.02]
    _, blow_hi, _ = growth[1.10]
    ok = (not blow_ok) and abs(g_ok - 1.0) < 1e-6 and g_mid > 100.0 and blow_hi
    assert ok, f"0.999h*: 增长 {g_ok:.3e}(爆炸 {blow_ok})；1.02h*: {g_mid:.3e}；1.10h*: 爆炸 {blow_hi}"
    record(
        label="跨过天花板的显式解行为（一维扩散 n=20）",
        theory="|λ|max≈4(n+1)²=1764，h*=2/|λ|≈1.1338e-3；h>h* 必有模态被放大",
        measured=f"h*=1.1338e-3：0.999h* 增长 {g_ok:.4f}（{growth[0.999][2]} 步）、"
        f"1.02h* 增长 {g_mid:.3e}（{growth[1.02][2]} 步）、1.10h* 爆炸={blow_hi}（{growth[1.10][2]} 步内 |y|>1e12）",
        tol="h* 以下有界、1.02h* 增长 >100 倍、1.10h* 触发爆炸判据",
        passed=ok,
        kind="A",
        extra={"growth_below": g_ok, "growth_above": g_mid},
    )


@assertion("A", "刚性线性系统上隐式 Euler 的步长可以远超显式天花板（20 倍）而解仍有界", chapter="3.2")
def test_implicit_is_unconditionally_stable_on_a_stiff_system() -> None:
    n = 20
    A = imp.stiff_diffusion_matrix(n)
    y0 = np.sin(np.linspace(0.1, 1.0, n))
    bound = imp.stiff_spectrum_bound(n)
    hc = imp.max_stable_step("forward_euler", 1, -bound)
    h = 20.0 * hc
    sol = imp.integrate_linear_system("implicit_euler", A, (0.0, 0.5), y0, h)
    scale = float(np.max(np.abs(y0)))
    growth = sol.max_abs_y / scale
    ok = sol.ok and (not sol.blowup) and growth <= 1.0 + 1e-9
    assert ok, f"h=20h* 时增长 {growth:.6e}，ok={sol.ok}，爆炸={sol.blowup}"
    record(
        label="隐式 Euler 在 20 倍显式天花板上",
        theory="A-稳定 ⟹ 任意 h 有界；扩散矩阵的精确解单调衰减（A 负定）",
        measured=f"h={h:.4e}=20×h*（{sol.n_steps} 步，{sol.n_lu} 次 LU），最大模比 {growth:.6f}，"
        f"Newton 最大残余 {sol.max_newton_residual:.2e}",
        tol="增长 ≤ 1+1e-9",
        passed=ok,
        kind="A",
    )


@assertion(
    "A",
    "启动步的选择在快模态被激发时才现形：解带 e^{−λt} 边界层时，hλ=3 的显式启动第一步冲到 −1（误差 2.05），隐式启动误差只有它的 1/10",
    chapter="3.2",
)
def test_bdf2_startup_step_shows_up_when_the_fast_mode_is_excited() -> None:
    lam = 1000.0
    h = 3.0e-3  # hλ = 3 > 2：前向 Euler 的稳定域之外
    c0 = 1.0  # 解 y = cos t + c₀e^{−λt}：快模态的系数不再是 0

    def exact(t: Any) -> np.ndarray:
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        return np.asarray(imp.stiff_cos_solution_full(tt, lam, c0), dtype=float).reshape(tt.shape)

    y0 = np.array([1.0 + c0])
    out = {}
    for startup in ("forward_euler", "implicit_euler"):
        sol = imp.integrate(
            "bdf2",
            imp.stiff_cos_rhs(-lam),
            imp.stiff_cos_jac(-lam),
            (0.0, 0.03),
            y0,
            h,
            startup=startup,
            exact=exact,
        )
        out[startup] = sol
    e_fwd, e_imp = out["forward_euler"], out["implicit_euler"]
    y1_fwd = float(np.asarray(e_fwd.y)[1, 0])
    y1_exact = float(exact(h)[0])
    y1_imp = float(np.asarray(e_imp.y)[1, 0])
    ratio = e_fwd.max_error / max(e_imp.max_error, 1e-300)
    ok = (
        (not e_fwd.blowup)
        and (not e_imp.blowup)
        and y1_fwd < 0.0 < y1_imp
        and ratio > 5.0
    )
    assert ok, f"首步 显式 {y1_fwd:.6f} / 隐式 {y1_imp:.6f} / 精确 {y1_exact:.6f}；最大误差比 {ratio:.3f}"
    record(
        label="BDF2 两种启动步（快模态被激发，hλ=3）",
        theory="|1+λh|=2>1：显式启动把快模态反向放大；隐式启动 |1/(1+λh)|=1/4 是收缩的",
        measured=f"精确首步 {y1_exact:.6f}；显式启动首步 {y1_fwd:.6f}（过冲到负值），隐式启动首步 {y1_imp:.6f}；"
        f"整段最大误差 显式 {e_fwd.max_error:.6f} vs 隐式 {e_imp.max_error:.6f}（比 {ratio:.2f}）",
        tol="显式首步过冲到负、隐式首步仍为正、误差比 > 5",
        passed=ok,
        kind="A",
        extra={"y1_exact": y1_exact, "y1_forward": y1_fwd, "y1_implicit": y1_imp, "error_ratio": ratio},
    )


@assertion("A", "非线性刚性算例（精确解仍是 cos t）里 Newton 真的要多步迭代，且隐式 Euler 仍是一阶", chapter="3.2")
def test_nonlinear_stiff_problem_needs_newton_and_keeps_order_one() -> None:
    lam, eps = -500.0, 8.0
    cos = math.cos

    def f(t: float, y: np.ndarray) -> np.ndarray:
        d = y[0] - cos(t)
        return np.array([lam * d - math.sin(t) - eps * d * d])

    def jac(t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG001
        d = y[0] - cos(t)
        return np.array([[lam - 2.0 * eps * d]])

    hs = np.array([1.0 / 2**k for k in (3, 4, 5, 6, 7)])
    err = []
    iters = []
    for h in hs:
        sol = imp.integrate("implicit_euler", f, jac, (0.0, 1.0), np.array([1.0]), float(h), exact=imp.stiff_cos_exact)
        iters.append(sol.n_newton_iters / max(sol.n_steps, 1))
        err.append(sol.max_error)
    fit = assert_order(hs, np.asarray(err), theory=1.0, label="非线性刚性算例的实测阶", detail="f=−λ(y−cos t)−sin t−ε(y−cos t)²，精确解 cos t")
    per_step = float(np.mean(iters))
    ok = per_step > 1.0
    assert ok, f"每步平均 Newton 迭代 {per_step:.4f}（应当 > 1，否则说明非线性没起作用）"
    record(
        label="非线性算例的 Newton 迭代数与阶",
        theory="非线性 ⟹ 每步 > 1 次迭代；隐式 Euler 一阶",
        measured=f"每步平均迭代 {per_step:.4f} 次（各步长 {', '.join(f'{v:.3f}' for v in iters)}），实测阶 {fit.slope:.4f}",
        tol="迭代 > 1；阶 1±0.1",
        passed=ok,
        kind="A",
        extra={"per_step_iters": per_step, "slope": fit.slope},
    )


@assertion("A", "代价梯的每一级都是真积分：误差按 h² 缩（每降一级 h 减半、误差降到 1/4），求值次数翻倍", chapter="3.2")
def test_cost_ladder_is_a_real_measurement() -> None:
    lam = -1000.0
    lad = imp.cost_ladder(
        "trapezoidal",
        2,
        imp.stiff_cos_rhs(lam),
        imp.stiff_cos_jac(lam),
        (0.0, 2.0 * math.pi),
        np.array([1.0]),
        k_range=(3, 10),
        exact=imp.stiff_cos_exact,
    )
    err = np.asarray(lad["max_error"], dtype=float)
    nf = np.asarray(lad["n_f_evals"], dtype=float)
    ratios = err[:-1] / err[1:]
    nf_ratios = nf[1:] / nf[:-1]
    ok = bool(np.all(np.abs(ratios - 4.0) < 1.0)) and bool(np.all(np.abs(nf_ratios - 2.0) < 0.05))
    assert ok, f"误差比 {ratios}，求值比 {nf_ratios}"
    record(
        label="代价梯的误差比与求值比",
        theory="二阶方法：h 减半 ⟹ 误差 ×1/4；步数 ×2 ⟹ 求值 ×2",
        measured=f"误差比 {np.array2string(ratios, precision=3)}，求值比 {np.array2string(nf_ratios, precision=3)}，"
        f"最粗一级 h={lad['h'][0]:.4f} 误差 {err[0]:.3e}（{nf[0]:.0f} 次求值）",
        tol="误差比 4±1、求值比 2±0.05（步数取整）",
        passed=ok,
        kind="A",
        extra={"error_ratios": ratios.tolist(), "eval_ratios": nf_ratios.tolist()},
    )


# --------------------------------------------------------------------------- #
# C 类：实测阈值
# --------------------------------------------------------------------------- #
@assertion("C", "显式方法在 10⁴ 次求值预算内能覆盖的刚性比上限：二分实测 ≈ 2·预算/t_end = 3183", chapter="3.2")
def test_stiffness_ratio_threshold_for_explicit_step_budget() -> None:
    t_end = 2.0 * math.pi
    budget = 10000
    calls = {"n": 0}

    def predicate(r: float) -> bool:
        calls["n"] += 1
        return imp.stability_limited_steps("forward_euler", 1, -float(r), t_end) <= budget

    res = find_threshold(predicate, 20.0, 100000.0, assume="true_at_lo", note="显式步数 ≤ 10⁴")
    theory = 2.0 * budget / t_end
    assert_threshold(res, theory, rtol=0.01, label="显式方法的刚性比预算阈值", detail=f"{calls['n']} 次谓词调用（每次内部还有一次 |R|=1 的二分）")
    record(
        label="阈值两侧的步数",
        theory=f"r* = 2·预算/t_end = {theory:.2f}",
        measured=f"r* ≈ {res.threshold:.2f}（区间 [{res.lo:.2f}, {res.hi:.2f}]，{res.iterations} 次二分）；"
        f"r=1000 时 {imp.stability_limited_steps('forward_euler', 1, -1000.0, t_end)} 步，"
        f"r=10000 时 {imp.stability_limited_steps('forward_euler', 1, -10000.0, t_end)} 步",
        tol="相对误差 ≤ 1%",
        passed=abs(res.threshold - theory) <= 0.01 * theory,
        kind="C",
        extra={"threshold": res.threshold, "theory": theory, "predicate_calls": calls["n"]},
    )


@assertion(
    "C",
    "容差与代价的交叉点：容差放宽到某处后隐式的总求值次数超过显式（二分实测），且与代价梯的二阶拟合预测同量级",
    chapter="3.2",
)
def test_cost_crossover_tolerance_matches_ladder_prediction() -> None:
    lam = -1000.0
    t_end = 2.0 * math.pi
    rhs, jac = imp.stiff_cos_rhs(lam), imp.stiff_cos_jac(lam)
    lad = imp.cost_ladder("trapezoidal", 2, rhs, jac, (0.0, t_end), np.array([1.0]), k_range=(3, 11), exact=imp.stiff_cos_exact)
    h_exp = imp.max_stable_step("forward_euler", 1, lam)
    n_exp = int(math.ceil(t_end / h_exp))
    # 独立预测：把代价梯拟合成 err ≈ C·h^p，再解 T/h = n_exp 处的容差
    logh = np.log(np.asarray(lad["h"], dtype=float))
    logerr = np.log(np.asarray(lad["max_error"], dtype=float))
    p, logC = np.polyfit(logh, logerr, 1)
    C = math.exp(float(logC))
    h_star = t_end / n_exp
    tol_pred = C * h_star ** float(p)

    def predicate(log10_tol: float) -> bool:
        pick = imp.step_for_tolerance(lad, 10.0**log10_tol)
        return pick["n_f_evals"] <= float(n_exp)  # 隐式总求值不超过显式 ⟹ 隐式更便宜

    # lo 在紧容差端（隐式更贵 ⟹ 谓词为假），hi 在松容差端（隐式更便宜 ⟹ 谓词为真）
    res = find_threshold(predicate, -11.0, -1.0, assume="false_at_lo", note="隐式求值次数 ≤ 显式")
    measured_tol = 10.0 ** res.threshold
    ratio = measured_tol / tol_pred
    ok = 0.1 <= ratio <= 10.0  # 阶梯每级 4 倍，允许同一量级
    assert ok, f"实测交叉容差 {measured_tol:.3e} vs 拟合预测 {tol_pred:.3e}（比 {ratio:.2f}）"
    record(
        label="代价交叉容差：二分实测 vs 二阶拟合预测",
        theory=f"err≈C·h^p（p={float(p):.4f}，C={C:.3e}）⟹ tol*≈C·(T/n_exp)^p = {tol_pred:.3e}",
        measured=f"二分实测 tol* = {measured_tol:.3e}（log10 ∈ [{res.lo:.3f}, {res.hi:.3f}]，{res.iterations} 次二分）；"
        f"隐式 h={imp.step_for_tolerance(lad, measured_tol)['h']:.4e}，显式 {n_exp} 步 × 1 次求值",
        tol="两者同量级（比值落在 [0.1, 10]）",
        passed=ok,
        kind="C",
        extra={"measured_tol": measured_tol, "predicted_tol": tol_pred, "ratio": ratio, "fit_p": float(p)},
    )


# --------------------------------------------------------------------------- #
# 帧契约（走真正的服务端管线）
# --------------------------------------------------------------------------- #
@assertion("A", "5 个 viz 的帧都过服务端体检：requiredAux 齐备、meta 全有限标量、数组可编码", chapter="3.2")
def test_all_chapter_frames_are_well_formed() -> None:
    reg = Registry(ROOT)
    reg.discover(strict=False)
    entry = reg.get("3.2")
    report: list[str] = []
    ok = True
    for viz in entry.module.MANIFEST["vizzes"]:
        vid = viz["id"]
        params = {p["name"]: p.get("default") for p in viz["params"]}
        t_max = float((viz.get("time") or {}).get("tMax") or 0.0)
        for t in (0.0, t_max):
            fr = build_frame(entry, vid, params, t, strict=True)
            names = list(fr.names)
            low = {n.lower() for n in names}
            for aux in viz["requiredAux"]:
                if not any(aux in n for n in low):
                    ok = False
                    report.append(f"{vid}@{t:g}: 缺 requiredAux {aux}")
                elif not any(n.startswith(aux) for n in low):
                    ok = False
                    report.append(f"{vid}@{t:g}: {aux} 只有间接匹配")
                else:
                    arr = fr[next(n for n in names if n.lower().startswith(aux))]
                    if np.all(~np.isfinite(np.asarray(arr, dtype=float))):
                        ok = False
                        report.append(f"{vid}@{t:g}: {aux} 全是非有限值")
            bad_meta = [k for k, v in fr.meta.items() if not isinstance(v, (int, float, bool)) or (isinstance(v, float) and not math.isfinite(v))]
            if bad_meta:
                ok = False
                report.append(f"{vid}@{t:g}: meta 非标量或非有限 {bad_meta}")
            # 编码本身就是一条不变量：header 用 allow_nan=False
            blob = nlf_mod.pack_frame(fr)
            back = nlf_mod.unpack_frame(blob)
            if len(back.names) != len(names):
                ok = False
                report.append(f"{vid}@{t:g}: 编码往返数组数不一致 {len(back.names)} vs {len(names)}")
            report.append(f"{vid}@{t:g}: {len(names)} 数组 / {len(blob)} 字节 / meta {len(fr.meta)} 项")
    assert ok, "\n".join(report)
    record(
        label="5 个 viz × 2 个时间端点的帧体检",
        theory="requiredAux 齐备、meta 只放有限标量、帧可编码可解码",
        measured="；".join(report),
        tol="全部通过",
        passed=ok,
        kind="A",
    )


@assertion(
    "B",
    "每个参数角点上帧都收敛到「一个非有限值都没有」：meta 全有限、数组全有限（极点位置换成占位值并配掩码，折线不用 NaN 分隔）",
    chapter="3.2",
)
def test_frame_corners_have_no_undocumented_nonfinite_values() -> None:
    """全参数角点的「帧卫生」体检（B 类：不变量，不是解析对拍）。

    这条是被真事逼出来的：``stability_region`` 与 ``stability_surface`` 在**默认参数**
    （``bdf2`` / ``bdf``+``order=2``）下，网格点正好压在 BDF 的极点 z=α_k 上，
    ``|R|`` 除零 → ``RuntimeWarning``。仓库把帧构建期的 RuntimeWarning 当错误
    （帧会被缓存/编码，warning 说明某个量已经无定义），于是整帧构建失败。

    约定（写进每帧的 ``strings.nonfinite_policy``）：**数组与 meta 里一个非有限值都不许有**。
    极点位置换成 ``strings.pole_fill`` 并由 ``*_pole`` 掩码标出；折线不用 NaN 分隔
    （回到起点再画下一段）；meta 一律有限。为什么连 NaN 都不留：``.nlf`` 的数组段是裸
    float64，NaN 编码得进去、也不会报错，但前端拿到含 NaN 的折线会**静默丢掉那一段**。
    """
    import warnings

    reg = Registry(ROOT)
    reg.discover(strict=False)
    entry = reg.get("3.2")
    problems: list[str] = []
    pole_cases: list[str] = []
    n_frames = 0
    for viz in entry.module.MANIFEST["vizzes"]:
        base = {p["name"]: p.get("default") for p in viz["params"]}
        corners: list[dict] = [dict(base)]
        for p in viz["params"]:
            name, kind = p["name"], p.get("type")
            if kind == "enum":
                opts = list(p.get("options") or [])
                corners += [{**base, name: opts[0]}, {**base, name: opts[-1]}]
            elif kind in ("int", "float") and "min" in p and "max" in p:
                corners += [{**base, name: p["min"]}, {**base, name: p["max"]}]
            elif kind == "bool":
                corners += [{**base, name: not base.get(name, False)}]
        time = viz.get("time") or {}
        times = [0.0] if time.get("mode") == "none" else [0.0, float(time.get("tMax") or 0.0)]
        for params in corners:
            for t in times:
                n_frames += 1
                tag = f"{viz['id']} {params} t={t:g}"
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        fr = build_frame(entry, viz["id"], params, t, strict=True)
                        nlf_mod.pack_frame(fr)
                except Exception as exc:  # noqa: BLE001 - 每一处失败都要收集起来
                    problems.append(f"{tag}: 构建/编码失败 {type(exc).__name__}: {exc}")
                    continue
                if "nonfinite_policy" not in fr.strings:
                    problems.append(f"{tag}: strings 里没有 nonfinite_policy（非有限值的处理口径没写清）")
                for key, val in fr.meta.items():
                    if isinstance(val, bool) or isinstance(val, int):
                        continue
                    if isinstance(val, float):
                        if not math.isfinite(val):
                            problems.append(f"{tag}: meta.{key}={val} 不是有限值")
                    elif val is not None:
                        problems.append(f"{tag}: meta.{key} 类型是 {type(val).__name__}（只许 number/bool/null）")
                names = set(fr.names)
                for name in fr.names:
                    arr = np.asarray(fr[name], dtype=float)
                    if arr.size and not np.all(np.isfinite(arr)):
                        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
                        problems.append(
                            f"{tag}: 数组 {name} 有 {n_bad}/{arr.size} 个非有限值"
                            f"（极点要换成 strings.pole_fill 占位 + *_pole 掩码，折线不许用 NaN 分隔）"
                        )
                for key, val in fr.meta.items():
                    if key.startswith("n_pole") and isinstance(val, int) and val > 0:
                        masks = [n for n in fr.names if n.endswith("_pole")]
                        if not masks:
                            problems.append(f"{tag}: meta.{key}={val} 说明命中极点，但帧里没有 *_pole 掩码")
                        else:
                            pole_cases.append(f"{tag}: {key}={val}，掩码 {masks}")
    assert n_frames >= 60, f"角点采样只跑了 {n_frames} 帧，覆盖不足"
    assert not problems, f"{len(problems)}/{n_frames} 帧的非有限值处理不合格：\n  - " + "\n  - ".join(problems[:12])
    record(
        label="参数角点上的帧卫生（数组/meta 全有限 + 极点掩码）",
        theory="帧里一个非有限值都不许有：极点用占位值 + 掩码，折线不用 NaN 分隔，meta 有限",
        measured=f"{n_frames} 帧全部通过；其中 {len(pole_cases)} 帧命中极点并带掩码"
        + (f"（例：{pole_cases[0]}；{pole_cases[-1]}）" if pole_cases else ""),
        tol="0 处非有限值问题",
        passed=True,
        kind="B",
        extra={"n_frames": n_frames, "n_pole_frames": len(pole_cases)},
    )
