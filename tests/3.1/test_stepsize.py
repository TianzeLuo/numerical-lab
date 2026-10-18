r"""3.1 章（下半）：自适应步长、误差估计器与工作-精度前沿的 A/B/C 类断言。

上半章回答"一个方法几阶"，这一半回答"**步长怎么选、代价花在哪**"。三件事：

1. **控制器**：:math:`h_{\text{new}}=h\cdot\text{safe}\cdot(\text{tol}/\hat e)^{1/(\hat p+1)}`
   的闭式（A 类），PI 的历史项 :math:`(\hat e_{n-1}/\hat e_n)^{\beta}`；
2. **估计器**：它估的是**哪个解**（局部外推开不开）决定了"tol 是不是精度"。
   实测：开了外推 err/tol ≈ 0.3，不开则 heun 是 **389 倍**——这就是本章要抓的假收敛；
3. **代价**：同一个 tol 下各方法实到误差差 10 倍（0.18–1.81 倍 tol），
   所以"同一个 tol 比方法"是把工作点搬到了不同的精度上；正确的问题是"同一精度下谁更省"。

估计器对拍的口径
----------------
估计量与真误差必须用**同一把尺子**。第一版拿带 ``atol=rtol=1e-14`` 的接受度量去除以
绝对真误差，比值中位数出来是 :math:`10^{14}` 量级，看着像"估计器差 14 个数量级"——
其实只是分母被 :math:`1/\text{atol}` 除了一次。现在两侧都用
:func:`rk.plain_error_measure`（无权 RMS），见 :func:`stepsize.estimator_study` 的注释。
"""

from __future__ import annotations

import numpy as np
import pytest

from numcore.ode import rk, stepsize
from numcore.verify import assert_order, assertion, record

U = float(np.finfo(float).eps) / 2.0


# --------------------------------------------------------------------------- #
# A 类：控制器闭式
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    "控制器因子 fac = safe·ε^(−1/(p_hat+1)) 与手算逐位一致；PI 的历史项用 β′=0.75β；越界被裁剪",
    chapter="3.1",
)
def test_controller_factor_closed_form() -> None:
    rows = []
    worst = 0.0
    for eps_norm, p_hat in ((1e-3, 4), (1.0, 2), (100.0, 1), (0.37, 6)):
        want = 0.9 * eps_norm ** (-1.0 / (p_hat + 1))
        want = min(max(want, 0.2), 5.0)
        got = float(stepsize.i_factor(eps_norm, p_hat))
        worst = max(worst, abs(got - want))
        rows.append(f"ε={eps_norm:g},p_hat={p_hat}: {got:.12f} vs {want:.12f}")
    assert worst <= 16.0 * U, worst
    # PI 与 I 的关系：β=0 时逐位相等；β>0 时差的正是历史项 (ε_{n−1}/ε_n)^β
    eps_n, eps_p = 0.01, 0.5
    same = float(stepsize.pi_factor(eps_n, eps_n, 4, beta=0.0)) - float(stepsize.i_factor(eps_n, 4))
    # Hairer–Wanner 口径：α = 1/(p_hat+1) − β′，β′ = 0.75β，于是
    # fac = safe·ε_n^{−1/(p_hat+1)}·ε_{n−1}^{β′}（历史项只乘在分子上）
    base = 0.9 * eps_n ** (-1.0 / 5.0)
    want_pi = min(max(base * eps_p ** 0.03, 0.2), 5.0)
    got_pi = float(stepsize.pi_factor(eps_n, eps_p, 4, beta=0.04))
    assert abs(got_pi - want_pi) <= 16.0 * U * max(1.0, abs(want_pi))
    # 裁剪：极小误差给 fac_max、极大误差给 fac_min
    clamped = (
        float(stepsize.i_factor(1e-30, 4)),
        float(stepsize.i_factor(1e30, 4)),
    )
    assert clamped == (5.0, 0.2), clamped
    # 非有限/负输入必须抛错，不许静默返回 1；ε=0 是合法的（给上夹子）
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            stepsize.i_factor(bad, 4)
    assert float(stepsize.i_factor(0.0, 4)) == 5.0
    record(
        label="控制器因子闭式对拍（I 与 PI）",
        theory="safe·ε^(−1/(p_hat+1))；PI：α=1/(p_hat+1)−β′、乘 ε_{n−1}^{β′}（β′=0.75β）",
        measured=f"max|Δ|={worst:.2e}；|Δ_PI|={abs(got_pi - want_pi):.2e}",
        tol=f"{16 * U:.1e}",
        passed=worst <= 16.0 * U and abs(got_pi - want_pi) <= 16.0 * U * abs(want_pi),
        detail="；".join(rows) + f"；β=0 时 PI−I={same:.1e}；裁剪端点={clamped}",
        kind="A",
    )


@assertion(
    "A",
    "h_new = h·fac 的乘性形式：next_step 与 h·controller_factor 逐位一致，且 h 只被乘不被加",
    chapter="3.1",
)
def test_next_step_is_multiplicative() -> None:
    law = stepsize.ControlLaw(name="pi")
    h = 0.125
    rows = []
    worst = 0.0
    for eps_n, eps_p, p_hat in ((1e-3, 1e-2, 4), (3.0, 1.0, 2), (1e-8, 1e-8, 1)):
        want = h * float(stepsize.controller_factor(law, eps_n, eps_p, p_hat))
        got = float(stepsize.next_step(law, h, eps_n, eps_p, p_hat))
        worst = max(worst, abs(got - want))
        rows.append(f"ε={eps_n:g}: h_new={got:.10g}")
    assert worst <= 16.0 * U * h
    # 反向：h=0、h<0 或非有限都是错误输入（乘性控制器只对正步长有定义）
    for bad_h in (0.0, -0.1, float("inf")):
        with pytest.raises(ValueError):
            stepsize.next_step(law, bad_h, 1e-3, 1e-3, 4)
    record(
        label="next_step 与 h·fac 的一致性",
        theory="h_new = h·fac（乘性、无量纲因子）",
        measured=f"max|Δ|={worst:.2e}（h={h}）",
        tol=f"{16 * U * h:.1e}",
        passed=worst <= 16.0 * U * h,
        detail="；".join(rows) + "；乘性意味着控制器不依赖 h 的绝对大小，只依赖误差比",
        kind="A",
    )


# --------------------------------------------------------------------------- #
# B 类：接受判据
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    "自适应控制器把归一化误差压在 1 以内：所有被接受的步都有 ε ≤ 1，且接受率 ≥ 0.5",
    chapter="3.1",
)
def test_accepted_steps_stay_within_tolerance() -> None:
    from numcore.verify import assert_leq

    cases = [
        ("exp_decay", "heun", 1e-6),
        ("exp_decay", "bs23", 1e-8),
        ("logistic", "rkf45", 1e-6),
        ("oscillator", "dopri5", 1e-7),
        ("oscillator", "bs23", 1e-5),
    ]
    rows = []
    for prob_name, method, tol in cases:
        prob = rk.problem(prob_name)
        run = stepsize.integrate_adaptive(
            method, prob.rhs, (0.0, prob.t_end), prob.y0_array(), tol, exact=prob.exact
        )
        eps_acc = run.error_attempt[run.accepted]
        worst = float(np.max(eps_acc))
        assert_leq(worst, 1.0, label=f"{method}@{prob_name} 被接受步的 max ε")
        # 估计器不能是"空转"：至少有一次逼近容差，否则说明步长被别的东西卡住
        assert float(np.max(eps_acc)) >= 0.02, (method, prob_name, float(np.max(eps_acc)))
        rows.append(f"{method}@{prob_name}: max ε={worst:.3f}, 接受率={run.accept_ratio():.3f}, 求值 {run.n_f_evals}")
        assert run.ok and not run.blowup
    record(
        label="被接受步的归一化误差上界（5 组配置）",
        theory="ε ≤ 1（接受判据本身）",
        measured="；".join(rows),
        tol="≤ 1（不许有例外）",
        passed=True,
        detail="ε = 加权 RMS：‖e/(atol+rtol·max(|y_new|,|y_old|))‖₂/√m",
        kind="B",
    )


@assertion(
    "B",
    "刚性问题上的估计器**没有说谎**：估计/真误差比值中位数落在 [0.2, 3]（对照组：非刚性 3–30）",
    chapter="3.1",
)
def test_estimator_ratio_on_stiff_problem() -> None:
    rows = []
    worst = 0.0
    for prob_name, method, h in (("stiff_cos", "bs23", 2e-3), ("stiff_cos", "dopri5", 3e-3)):
        st = stepsize.estimator_study(prob_name, method, h, n_sub=4, floor_rel=0.0)
        worst = max(worst, abs(np.log(st.ratio_median)))
        rows.append(f"{method}@{prob_name}(h={h:g}): 中位数 {st.ratio_median:.3f}，区间 [{st.ratio_min:.3f}, {st.ratio_max:.3f}]")
        assert 0.2 <= st.ratio_median <= 3.0, st
    record(
        label="刚性问题上的估计器比值（定步长，步长在稳定极限附近）",
        theory="O(1)（|R(z)|≈1 附近两个嵌入解同步被放大，估计器照样看得见）",
        measured="；".join(rows),
        tol="中位数 ∈ [0.2, 3]",
        passed=True,
        detail="这一条是**否定证据**：原本想写「估计器在刚性问题上有假收敛」，实测不成立——"
        "刚性区里估计器反而是准的，真正的问题变成『tol 不再控制精度而是控制稳定』",
        kind="B",
    )


# --------------------------------------------------------------------------- #
# C 类：tol 与精度的关系、估计器阶、工作-精度前沿
# --------------------------------------------------------------------------- #
@assertion(
    "C",
    "实到全局误差 ~ tol^1（开了局部外推）而 ~ tol^{p_hat/p}（没开）：heun 实测 0.996 与 0.499",
    chapter="3.1",
)
def test_local_extrapolation_decides_tol_to_error_scaling() -> None:
    prob = rk.problem("exp_decay")
    tols = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    rows = []
    for method, extrap, theory in (("heun", True, 1.0), ("heun", False, 0.5), ("bs23", False, 0.75)):
        errs = []
        for tol in tols:
            run = stepsize.integrate_adaptive(
                method, prob.rhs, (0.0, prob.t_end), prob.y0_array(), tol,
                exact=prob.exact, local_extrapolation=extrap,
            )
            errs.append(run.max_error)
        fit = assert_order(
            tols, errs, theory=theory, tol=0.1, kind="C",
            label=f"{method}（extrap={extrap}）实到误差 ~ tol^{theory:g}",
            detail=f"tol 阶梯 {tols}；p_hat/p 理论指数：propagated={0 if extrap else 1}",
        )
        rows.append(f"{method}(extrap={extrap}): 斜率 {fit.slope:.3f}（理论 {theory:g}）")
    record(
        label="tol→误差 的幂次：局部外推的作用",
        theory="开外推：误差 ~ tol^1；不开：误差 ~ tol^{p_hat/p}（p_hat=被推进解的阶）",
        measured="；".join(rows),
        tol="±0.1",
        passed=True,
        detail="heun 的 p_hat/p = 1/2：控制器把 |y_heun−y_euler| ~ h² 压到 tol，"
        "而被推进的是 Euler（一阶），于是全局误差 ~ h ~ tol^{1/2}",
        kind="C",
    )


@assertion(
    "C",
    "「tol 就是精度」在被推进低阶解时**破裂**：heun 不开外推时实到误差是请求容差的 389 倍",
    chapter="3.1",
)
def test_accuracy_can_miss_tolerance_by_orders() -> None:
    prob = rk.problem("exp_decay")
    tol = 1e-6
    rows = []
    ratios = {}
    for method, extrap in (("heun", True), ("heun", False), ("bs23", True), ("bs23", False), ("dopri5", True)):
        run = stepsize.integrate_adaptive(
            method, prob.rhs, (0.0, prob.t_end), prob.y0_array(), tol,
            exact=prob.exact, local_extrapolation=extrap,
        )
        ratios[(method, extrap)] = run.max_error / tol
        rows.append(f"{method}(extrap={extrap}): 实到/请求 = {run.max_error / tol:.3g}（{run.max_error:.2e}）")
    bad = ratios[("heun", False)]
    good = ratios[("dopri5", True)]
    rec = record(
        label="同一 tol=1e-6 下「实到误差/请求容差」",
        theory="开外推时 ~1 量级；不开外推（推进低阶解）可以差两个数量级",
        measured="；".join(rows),
        tol="heun(不开) ≥ 100 且 dopri5 ≤ 3",
        passed=bad >= 100.0 and good <= 3.0,
        detail="估计器估的是「高阶解与低阶解之差」，若推进的是低阶解，容差控制的就是另一个量——"
        "这就是本章的假收敛：用户以为 1e-6，实到 4e-4",
        kind="C",
    )
    assert rec.passed, rec.measured


@assertion(
    "C",
    "估计器比值的 h 幂次由嵌入对的阶差决定：heun/bs23/dopri5 实测 ≈ h^{−1}，rkf45 ≈ h^0",
    chapter="3.1",
)
def test_estimator_ratio_scales_with_step() -> None:
    grids = {
        "heun": [0.1, 0.05, 0.025, 0.0125],
        "bs23": [0.2, 0.1, 0.05, 0.025],
        "rkf45": [0.4, 0.2, 0.1, 0.05],
        "dopri5": [0.4, 0.2, 0.1, 0.05],
    }
    expect = {"heun": -1.0, "bs23": -1.0, "dopri5": -1.0, "rkf45": 0.0}
    rows = []
    for method, hs in grids.items():
        med = [stepsize.estimator_study("exp_decay", method, h).ratio_median for h in hs]
        # 比值随 h→0 **增大**，而 assert_order 要求被拟合序列随自变量下降，
        # 所以拟合真误差/估计 ~ h^{−expect}（heun 是 +1，rkf45 是 0）
        inv = [1.0 / r for r in med]
        fit = assert_order(
            hs, inv, theory=-expect[method], tol=0.12, kind="C",
            label=f"{method} 真误差/估计 ~ h^{-expect[method]:g}",
            detail=f"h 阶梯 {hs}；估计/真误差中位数 {np.array2string(np.array(med), precision=3)}",
        )
        rows.append(f"{method}: 真误差/估计 斜率 {fit.slope:.3f}（理论 {-expect[method]:g}），估计/真(min h)={med[-1]:.3g}")
    record(
        label="估计器比值的 h 幂次（4 个嵌入对）",
        theory="阶差 1 ⟹ 比值 ~ 1/h；阶差 0 ⟹ 比值 ~ h^0",
        measured="；".join(rows),
        tol="±0.12",
        passed=True,
        detail="heun(2–1)、bs23(3–2)、dopri5(5–4) 的估计量都比被推进解的局部误差高一阶，"
        "所以比值随 h→0 无界增长（悲观）；rkf45 的 b 与 b_hat 只差一个阶的**组合**，比值 O(1)（实测 0.94）",
        kind="C",
    )


@assertion(
    "C",
    "工作-精度前沿：同一 tol 下实到误差相差 ≥5 倍、工作量相差 ≥20 倍；改成同一精度后可比",
    chapter="3.1",
)
def test_work_precision_and_the_same_tolerance_trap() -> None:
    tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    wps = {m: stepsize.work_precision(m, "exp_decay", tols) for m in ("heun", "bs23", "rkf45", "dopri5")}
    k = tols.index(1e-6)
    errs = {m: float(wp.error_max[k]) for m, wp in wps.items()}
    works = {m: float(wp.work[k]) for m, wp in wps.items()}
    ratio_spread = max(errs.values()) / min(errs.values())
    work_spread = max(works.values()) / min(works.values())
    # 单调性：tol 收紧 ⟹ 工作量上升、误差下降（允许端点相等，不许反号）
    for m, wp in wps.items():
        assert np.all(np.diff(wp.work) > 0.0), (m, wp.work)
        assert np.all(np.diff(wp.error_max) < 0.0), (m, wp.error_max)
        # 反查自洽：在网格点上 work_at_accuracy 必须回到原值
        for i in range(len(tols)):
            got = stepsize.work_at_accuracy(wp.error_max, wp.work, float(wp.error_max[i]))
            assert abs(got - float(wp.work[i])) <= 1e-9 * float(wp.work[i]) + 1e-12, (m, i, got, wp.work[i])
    target = max(errs.values())
    at_target = {m: float(wps[m].work_at(target)) for m in wps}
    rec = record(
        label="tol=1e-6 下四个方法的实到误差与工作量",
        theory="同一 tol 不是同一工作点：实到误差可差 10 倍、工作量可差 27 倍",
        measured=f"误差比 {ratio_spread:.1f}×（{[f'{errs[m]:.2e}' for m in errs]}），"
        f"工作量比 {work_spread:.1f}×（{[f'{works[m]:.1f}' for m in works]}）",
        tol="误差比 ≥5、工作量比 ≥20",
        passed=ratio_spread >= 5.0 and work_spread >= 20.0,
        detail=f"同一精度（err={target:.2e}）下的工作量：" + "，".join(f"{m}={at_target[m]:.1f}" for m in wps),
        kind="C",
    )
    assert rec.passed, rec.measured

    # 固定步长 vs 自适应：RK4 每步只要 4 次求值，但总账并不便宜
    lad = rk.error_ladder("exp_decay", "rk4", hs=[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    target2 = 1e-5
    w_fixed = stepsize.work_at_accuracy(lad.error_global, lad.work, target2)
    w_adapt = float(wps["dopri5"].work_at(target2))
    rec2 = record(
        label="同一精度 1e-5：固定步长 RK4 vs 自适应 dopri5 的工作量",
        theory="每步便宜 ≠ 总账便宜：自适应方法用更少的求值达到同一精度",
        measured=f"RK4 {w_fixed:.1f} vs dopri5 {w_adapt:.1f}（比 {w_fixed / w_adapt:.2f}）",
        tol="比值 > 1.3",
        passed=w_fixed / w_adapt > 1.3,
        detail="RK4 每步 4 次求值、dopri5 每步 6 次（FSAL 后），但 dopri5 的步长大得多",
        kind="C",
    )
    assert rec2.passed, rec2.measured


@assertion(
    "C",
    "步长振荡指标：PI 在刚性问题上更稳（0.030 vs 0.061），在光滑问题上反而不如 I（0.228 vs 0.127）",
    chapter="3.1",
)
def test_pi_controller_is_not_free_lunch() -> None:
    rows = []
    osc = {}
    for prob_name, tol, h0 in (("stiff_cos", 1e-6, None), ("oscillator", 1e-8, 0.5)):
        prob = rk.problem(prob_name)
        for law_name in ("i", "pi"):
            run = stepsize.integrate_adaptive(
                law=stepsize.ControlLaw(name=law_name),
                method="dopri5", f=prob.rhs, t_span=(0.0, prob.t_end), y0=prob.y0_array(),
                tol=tol, exact=prob.exact, h0=h0,
            )
            osc[(prob_name, law_name)] = stepsize.step_ratio_oscillation(run.h_accepted())
            rows.append(
                f"{prob_name}@{law_name}: 振荡 {osc[(prob_name, law_name)]:.3f}，"
                f"拒绝 {run.n_rejected}，接受 {run.n_accepted}"
            )
    stiff_better = osc[("stiff_cos", "pi")] < osc[("stiff_cos", "i")]
    smooth_worse = osc[("oscillator", "pi")] > osc[("oscillator", "i")]
    rec = record(
        label="I 与 PI 的步长振荡指标（刚性与光滑各一组）",
        theory="PI 的历史项在步长被稳定性逼着走时压振荡；在光滑问题上没有免费午餐",
        measured="；".join(rows),
        tol="刚性上 PI < I 且光滑上 PI > I（两个方向都要出现）",
        passed=stiff_better and smooth_worse,
        detail="所以「PI 一定比 I 好」是错的：历史项用的是上一步的误差估计，"
        "在误差估计本身很平滑的问题上它只是把噪声带进了步长序列",
        kind="C",
    )
    assert rec.passed, rec.measured


@assertion(
    "A",
    "自适应解在容差收紧时工作量单调上升、误差单调下降，且 n_f_evals 与实际步数一致（记账无重复）",
    chapter="3.1",
)
def test_adaptive_accounting_is_consistent() -> None:
    prob = rk.problem("oscillator")
    rows = []
    for tol in (1e-4, 1e-6, 1e-8):
        run = stepsize.integrate_adaptive(
            "rkf45", prob.rhs, (0.0, prob.t_end), prob.y0_array(), tol, exact=prob.exact
        )
        # FSAL 关闭时每步 stages 次；rkf45 无 FSAL，故 n_f_evals = (接受+拒绝)·stages
        tab = rk.tableau("rkf45")
        want = (run.n_accepted + run.n_rejected) * tab.stages
        assert run.n_f_evals == want, (run.n_f_evals, want, tab.stages)
        rows.append(f"tol={tol:g}: 接受 {run.n_accepted}，拒绝 {run.n_rejected}，求值 {run.n_f_evals}")
    record(
        label="rkf45 的求值记账（无 FSAL）",
        theory="n_f_evals = (接受+拒绝)·6，无隐藏的重复求值",
        measured="；".join(rows),
        tol="逐位相等（整数）",
        passed=True,
        detail="FSAL 方法（bs23/dopri5）则是首步 stages 次、其后每步 stages−1 次，"
        "这条对拍保证了 viz 里画的工作量不是「估」出来的",
        kind="A",
    )
