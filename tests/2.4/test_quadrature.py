"""2.4 章 A/B 类断言：实测阶 vs 理论阶、Gauss 的 2n−1、奇异性与地板。

本文件守的是三件**互相独立**的事：

1. **公式的阶**（A 类）——复合梯形/Simpson/3-8 的实测斜率必须等于 2/4/4，
   误差常数必须落在 :math:`C(b-a)[\\min f^{(p)},\\max f^{(p)}]` 这个解析区间里，
   带符号误差必须被 :math:`C(b-a)h^pf^{(p)}(\\xi)` 的双侧不等式夹住，Romberg 第 j 列必须是 2j+2；
2. **精确度的边界**（A 类）——Gauss 的代数精度**恰好**是 2n−1：k=2n−1 在舍入地板、
   k=2n 当场失效，两个方向都要有实测值；
3. **奇异性与地板**（A/B 类）——tanh–sinh 对端点奇异的地板**不是** u 而是未采样尾巴
   :math:`\\int_0^{x_{\\min}}f`；换元 :math:`x=t^2` 之后等距规则重新收敛；
   原始等距规则在奇异被积函数上**根本没有定义**。

B 类守的是三件会骗人的事（本章的三条陷阱）：
**函数值准 ≠ 积分准**（权重放大 :math:`\\Gamma` 倍）、
**量到的是光滑度的阶**（√x 把 Simpson 从 4 阶压到 1.5 阶）、
**误差估计器估的是自己的误差**（1e-9 的输入噪声让真误差停在 1e-12，而估计器报 2.7e-15）。
"""

from __future__ import annotations

import math

import numpy as np

from numcore.quad.adaptive import adaptive_simpson
from numcore.quad.condition import amplification_experiment, input_error_bound, weight_diagnostics
from numcore.quad.gauss import (
    gauss_legendre_rule,
    moment_errors,
    moment_failure_factor,
)
from numcore.quad.integrals import integrand, substituted_error_curve
from numcore.quad.newton_cotes import (
    RULES,
    composite_eval,
    composite_rule,
    effective_sub_counts,
    error_curve,
    error_law,
    fitted_constant,
    newton_cotes_weights,
    rule_law,
)
from numcore.quad.romberg import column_orders, extrapolation_gain, romberg_table
from numcore.quad.singular import saturation_abscissa, tanh_sinh_rule
from numcore.verify import assertion, assert_leq, assert_order, record

#: 单位舍入 :math:`u=2^{-53}`。
U = 2.0**-53

#: 误差曲线用的面板数（6 的倍数让三种规则共用节点）。
_COUNTS = np.asarray([6, 12, 24, 48, 96, 192, 384], dtype=float)


def _errors(name: str, rule: str, counts: np.ndarray = _COUNTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(生效面板数, h=1/n_eff, |Q−I|)``。

    第三项是**反相关**的两列：``h`` 递减、误差递减，所以 ``assert_order(h, err)``
    得到的是正的阶（若把面板数当横坐标会得到负号——这是实现期真踩过的坑）。
    """
    item = integrand(name)
    eff = effective_sub_counts(rule, counts).astype(float)
    errs = np.asarray(error_curve(item.exact, item.f, item.a, item.b, eff, rule=rule), dtype=float)
    return eff, 1.0 / eff, errs


def _above_floor(errs: np.ndarray, h: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """只留地板之上的点（细端的平段会把阶拟合小）。"""
    keep = errs > 4.0 * floor
    return h[keep], errs[keep]


@assertion("A", theory="复合 Newton–Cotes 的误差阶 = 2/4/4（梯形/Simpson/3-8）", chapter="2.4")
def test_composite_orders_match_theory() -> None:
    """实测斜率对理论阶，三种规则各一条记录（容差 ±0.1）。"""
    rows = []
    for rule, p in (("trapezoid", 2.0), ("simpson", 4.0), ("simpson38", 4.0)):
        counts, h, errs = _errors("exp", rule)
        hh, ee = _above_floor(errs, h, U)
        fit = assert_order(
            hh,
            ee,
            theory=p,
            tol=0.1,
            skip=1,
            label=f"{rule} 的实测阶 = {p:g}",
            detail=f"面板数 {counts.astype(int).tolist()}，末点误差 {errs[-1]:.3e}",
        )
        rows.append(f"{rule}: 斜率 {fit.slope:+.4f}（R²={fit.r2:.8f}，末点 h={hh[-1]:.3e}）")
    record(label="逐规则明细", theory="-", measured="；".join(rows), tol="-", passed=True)


@assertion("A", theory="E := Q−I = C(b−a)h^p f^(p)(ξ)，C = +1/12、+1/180、+1/80", chapter="2.4")
def test_error_constants_and_bracket_match_theory() -> None:
    """两道独立证据：常数落在解析区间内 + 带符号误差被双侧不等式夹住。"""
    item = integrand("exp")  # f^(p) = e^x ∈ [1, e]，给出最紧的解析区间
    rows = []
    for rule in RULES:
        law = rule_law(rule)
        counts, _, errs = _errors("exp", rule)
        measured = fitted_constant(errs, item.a, item.b, counts, power=law.error_power)
        assert item.derivative_bounds is not None
        lo_d, hi_d = item.derivative_bounds(law.error_derivative)  # type: ignore[misc]
        lo = abs(law.error_coefficient) * (item.b - item.a) * min(lo_d, hi_d)
        hi = abs(law.error_coefficient) * (item.b - item.a) * max(lo_d, hi_d)
        record(
            label=f"{rule} 的实测常数落在 C(b−a)[min f^(p), max f^(p)] 内",
            theory=f"[{lo:.8f}, {hi:.8f}]（C={law.error_coefficient:+.8f}）",
            measured=measured,
            tol="区间内",
            passed=bool(lo <= measured <= hi),
            detail=f"p={law.error_power}、f^(p)∈[{lo_d:.4f}, {hi_d:.4f}]；"
            f"常数若写成 −1/{int(round(1 / abs(law.error_coefficient)))} 会掉到区间外——实现期真的写错过",
        )
        rows.append(f"{rule}: C_meas={measured:.8f}")
    for rule in RULES:
        law = error_law(rule)
        for n in (4, 24, 96):
            ev = composite_eval(item.f, item.a, item.b, n, rule=rule)
            signed = ev.value - item.exact
            lo, hi = law.predicted(item.a, item.b, n, 1.0, math.e)
            record(
                label=f"{rule}、n={n}：带符号误差被解析区间夹住",
                theory=f"[{lo:+.6e}, {hi:+.6e}]",
                measured=signed,
                tol="区间内",
                passed=bool(min(lo, hi) <= signed <= max(lo, hi)),
                detail="E 的符号约定是 Q−I：凸函数被梯形**高估**，所以 C>0",
            )
    record(label="逐规则常数", theory="-", measured="；".join(rows), tol="-", passed=True)


@assertion("A", theory="端点奇异 f ~ x^α ⇒ 实测阶 = min(p, α+1)（与公式阶无关）", chapter="2.4")
def test_smoothness_caps_the_measured_order() -> None:
    """√x（α=1/2）把三种规则的实测阶统统压到 1.5——**量到的是光滑度的阶**（陷阱②）。"""
    rows = []
    for rule in RULES:
        counts, h, errs = _errors("sqrt", rule)
        fit = assert_order(
            h,
            errs,
            theory=1.5,
            tol=0.1,
            skip=1,
            label=f"{rule} 在 √x 上的实测阶 = min(p, 1.5)",
            detail=f"理论 p={rule_law(rule).error_power}，被光滑性压到 1.5；面板数 {counts.astype(int).tolist()}",
        )
        rows.append(f"{rule}: {fit.slope:+.4f}（理论 p={rule_law(rule).error_power}）")
    record(
        label="三种规则的实测阶都被压到 1.5",
        theory="min(p, α+1) = 1.5",
        measured="；".join(rows),
        tol="±0.1",
        passed=True,
        detail="α=0.5 时 2 阶的梯形与 4 阶的 Simpson 量出来是同一个阶——"
        "「阶」这个词必须先说清楚是谁的阶",
    )


@assertion("A", theory="Gauss–Legendre 的代数精度恰好 2n−1：k=2n−1 精确、k=2n 失效", chapter="2.4")
def test_gauss_exactness_is_exactly_two_n_minus_one() -> None:
    """两个方向都给实测值：地板上的 0 与「当场失效」的跳变因子。"""
    rows = []
    for n in (2, 4, 8, 12, 16, 20):
        errs = np.asarray(moment_errors(n, k_max=2 * n + 1), dtype=float)
        limit = 2 * n - 1
        at_limit = float(errs[limit])
        beyond = float(errs[2 * n])
        factor = moment_failure_factor(n)
        record(
            label=f"n={n}：k=2n−1 精确、k=2n 失效",
            theory=f"err(2n−1)=0、err(2n)>0（实测失效因子 {factor:.3e}）",
            measured=f"err({limit})={at_limit:.3e}、err({2 * n})={beyond:.3e}",
            tol="at_limit ≤ 4u、beyond ≥ 1e3·u",
            passed=bool(at_limit <= 4.0 * U and beyond >= 1e3 * U),
            detail="两个方向缺一不可：只画前半边，「精确到 2n−1 次」与「精确到 100 次」看不出区别",
        )
        rows.append(f"n={n}: {at_limit:.1e}/{beyond:.1e}")
    record(label="逐例明细（k=2n−1 / k=2n）", theory="-", measured="；".join(rows), tol="-", passed=True)


@assertion("A", theory="Romberg 第 j 列的误差阶 = 2j+2（第 1 列就是 Simpson）", chapter="2.4")
def test_romberg_column_orders_match_theory() -> None:
    """四列的实测阶对理论值 2/4/6/8，另附相邻两列的加速比。"""
    item = integrand("exp")
    result = romberg_table(item.f, item.a, item.b, 10, exact=item.exact)
    orders = np.asarray(column_orders(result, item.exact), dtype=float)
    rows = []
    for j in range(4):
        theory = 2.0 * j + 2.0
        record(
            label=f"第 {j} 列的实测阶 = {theory:g}",
            theory=theory,
            measured=float(orders[j]),
            tol="±0.1",
            passed=bool(abs(float(orders[j]) - theory) <= 0.1),
            detail=f"该列可用行数 {result.size - j}（总层数 {result.size}）；"
            f"不剔掉触地板的行时第 3 列实测 6.96 而不是 7.97",
        )
        rows.append(f"col{j}: {orders[j]:.4f}")
    _, _, gains = extrapolation_gain(result, item.exact)
    finite = gains[np.isfinite(gains)]
    record(
        label="外推的加速比（同一行相邻两列）",
        theory="> 1（外推确实在消误差项）",
        measured=f"min={float(np.min(finite)):.4f}、max={float(np.max(finite)):.4f}、"
        f"inf {int(np.count_nonzero(~np.isfinite(gains)))} 个（细端恰好落到 0）",
        tol="≥ 1",
        passed=bool(np.all(finite >= 1.0)),
        detail="；".join(rows) + "；加速比掉到 ~1 就说明这一列已经在外推的精度极限上",
    )


@assertion("A", theory="tanh–sinh 的地板 = 未采样尾巴 ∫_0^{x_min}f（x_min ≈ u）", chapter="2.4")
def test_tanh_sinh_floor_is_the_unsampled_tail() -> None:
    """x^{−1/2} 型的地板是 √u 量级而不是 u：节点数加 5 倍，误差一动不动。"""
    item = integrand("inv_sqrt")
    h = 2.0**-4
    rows_data = []
    for m in (51, 102, 204, 256):
        rule = tanh_sinh_rule(m, h=h, a=item.a, b=item.b)
        value = rule.integrate_values(item.values(rule.nodes))
        live = ~rule.saturated
        x_min = float(np.min(rule.nodes[live])) if bool(np.any(live)) else float(np.min(rule.nodes))
        tail = item.tail_integral(x_min)
        rows_data.append((2 * m + 1, abs(value - item.exact), x_min, float(tail or 0.0), int(rule.n_saturated)))
    first, plateau = rows_data[0][1], rows_data[-1][1]
    record(
        label="节点数从 103 涨到 513，误差停在地板上",
        theory="地板 ≈ ∫_0^{x_min}f = 2√x_min ≈ √(2u) = 1.49e-08",
        measured=f"{first:.6e} → {plateau:.6e}（比值 {plateau / first:.6f}）",
        tol="比值 ∈ [0.5, 2]（不再下降）",
        passed=bool(0.5 <= plateau / first <= 2.0),
        detail="；".join(
            f"节点 {nn}：err={e:.3e}、x_min={xm:.3e}、尾巴={tl:.3e}、饱和 {sat}"
            for nn, e, xm, tl, sat in rows_data
        ),
    )
    ratio = [abs(e / tl) for _, e, _, tl, _ in rows_data if tl != 0.0]
    record(
        label="实测误差与解析尾巴同量级",
        theory="误差 / 尾巴 ∈ [0.2, 5]（地板就是这块没采到的面积）",
        measured="；".join(f"{r:.3f}" for r in ratio),
        tol="[0.2, 5]",
        passed=bool(all(0.2 <= r <= 5.0 for r in ratio)),
        detail="x_min ≈ u（tanh 在 double 里舍入到 ±1），所以尾巴是 √u 而不是 u",
    )
    log_item = integrand("log")
    log_rule = tanh_sinh_rule(256, h=h, a=log_item.a, b=log_item.b)
    log_err = abs(log_rule.integrate_values(log_item.values(log_rule.nodes)) - log_item.exact)
    record(
        label="同一个机制给出两种量级的地板",
        theory="ln x 型：x_min|ln x_min| ≈ 2e-15（u 量级）",
        measured=f"ln x 的误差 {log_err:.3e}（inv_sqrt 是 {plateau:.3e}）",
        tol="log_err ≤ 1e-13 且 plateau ≥ 1e-9",
        passed=bool(log_err <= 1e-13 and plateau >= 1e-9),
        detail="地板由 ∫_0^{x_min}f 的**解析大小**决定：x^{−1/2} 给 2√x_min，ln x 给 x_min|ln x_min|",
    )


@assertion("A", theory="换元 x=t² 把 x^{−1/2} 变成光滑被积函数：等距规则恢复 2 阶", chapter="2.4")
def test_substitution_restores_the_equispaced_order() -> None:
    """同一个梯形法，换元后重新收敛；原始等距规则则是 inf（无定义）。"""
    item = integrand("inv_sqrt")
    counts = np.asarray([8, 16, 32, 64, 128], dtype=float)
    errs = np.asarray(substituted_error_curve(item, counts.astype(int), rule="trapezoid"), dtype=float)
    fit = assert_order(
        1.0 / counts,
        errs,
        theory=2.0,
        tol=0.1,
        label="换元后等距规则的实测阶 = 2",
        detail=f"误差 {errs[0]:.3e} → {errs[-1]:.3e}（不换元时公式里是 inf）",
    )
    raw = composite_eval(item.f, item.a, item.b, 8, rule="trapezoid")
    n_bad = int(np.count_nonzero(~np.isfinite(raw.values)))
    record(
        label="不换元时公式本身没有定义",
        theory="x=0 处 f=+inf ⇒ 复合公式里有非有限项",
        measured=f"{n_bad} 个节点值非有限、finite={raw.finite}",
        tol="≥1",
        passed=bool(n_bad >= 1 and not raw.finite),
        detail=f"换元后同一套梯形法的实测阶是 {fit.slope:.4f}——差别不在方法而在**变量**",
    )


@assertion("A", theory="自适应 Simpson 的估计误差 ≥ 真误差（光滑函数上），且同量级以内", chapter="2.4")
def test_adaptive_estimator_is_conservative_on_smooth_functions() -> None:
    """光滑函数上估计器偏保守，但绝不低于真误差（比值实测 6.7e2 / 4.9e0 / 1.0e0）。"""
    for name in ("exp", "runge", "sqrt"):
        item = integrand(name)
        tol = 1e-9 * max(1.0, abs(item.exact))
        res = adaptive_simpson(item.f, item.a, item.b, tol, max_depth=20, exact=item.exact)
        assert res.true_error is not None
        ratio = res.error_estimate / res.true_error if res.true_error > 0.0 else math.inf
        record(
            label=f"{name}：估计误差 / 真误差",
            theory="≥ 1（保守）且 ≤ 1e6（同量级以内）",
            measured=f"est={res.error_estimate:.3e}、true={res.true_error:.3e}、比值 {ratio:.3e}",
            tol="[1, 1e6]",
            passed=bool(1.0 <= ratio <= 1e6),
            detail=f"区间 {res.n_intervals} 个、求值 {res.n_evals} 次、最大深度 {res.max_depth_used}、"
            f"converged={res.converged}",
        )


@assertion("B", theory="Σw = b−a 且 Γ=Σ|w|/Σw ≥ 1（判据是 Γ，不是点数）", chapter="2.4")
def test_weight_sums_and_condition_numbers() -> None:
    """权重和、条件数、负权重个数：等距规则 m=9 起 Γ>1，m=10 是反例。"""
    rows = []
    bad = []
    worst_sum = 0.0
    for m in range(2, 21):
        diag = weight_diagnostics(newton_cotes_weights(m, -1.0, 1.0))
        worst_sum = max(worst_sum, abs(diag.sum_w - 2.0))
        if abs(diag.sum_w - 2.0) > 1e-6 or diag.condition < 1.0 - 1e-12:
            bad.append(f"m={m}: Σw={diag.sum_w:.15f}、Γ={diag.condition:.6f}")
        rows.append((m, diag.condition, diag.n_negative, diag.min_w))
    record(
        label="权重和 = 区间长度、Γ ≥ 1（m=2..20）",
        theory="Σw=2（m≤20 时实测最大偏差 1.5e-8，留到 1e-6 的余量）、Γ≥1",
        measured=f"最大偏差 {worst_sum:.3e}" + ("" if not bad else "；失败：" + "；".join(bad)),
        tol="±1e-6",
        passed=not bad,
        detail="权重和破掉说明权重算错了——它是最弱的不变量，却抓住了最强的问题"
        "（见下一条：同一条公式换个写法，偏差能从 1e-8 涨到 21）",
    )
    gauss = weight_diagnostics(gauss_legendre_rule(12).weights)
    record(
        label="Gauss 权重全正 ⇒ Γ ≡ 1",
        theory="Γ=1、负权重 0 个",
        measured=f"Γ={gauss.condition:.15f}、负权重 {gauss.n_negative} 个、最小权重 {gauss.min_w:.6f}",
        tol="|Γ−1| ≤ 1e-12",
        passed=bool(abs(gauss.condition - 1.0) <= 1e-12 and gauss.n_negative == 0),
        detail="权重全正时 Σ|w|=Σw，函数值的误差不会被公式放大",
    )
    first_neg = next(m for m, _, neg, _ in rows if neg > 0)
    record(
        label="等距规则首次出现负权重的点数",
        theory="m=9（m=10 权重全正是反例，所以判据是 Γ 而不是点数）",
        measured=f"首个负权重在 m={first_neg}；"
        + "；".join(f"m={m}: Γ={g:.4f}（负 {n} 个，最小 {mw:+.4f}）" for m, g, n, mw in rows if m >= 8),
        tol="m == 9",
        passed=bool(first_neg == 9),
        detail="m=10 的 Γ 回到 1.0000——用「点数」当判据会在这里翻车",
    )


@assertion("B", theory="函数值的相对误差被放大 Σ|w| 倍传到积分（陷阱①）", chapter="2.4")
def test_function_value_error_is_amplified_by_gamma() -> None:
    """同一个函数、同一个扰动：Gauss 19 点放大 ~1 倍，等距 19 点放大 ~35 倍。"""
    equi = newton_cotes_weights(19, 0.0, 1.0)
    gauss_rule = gauss_legendre_rule(19, 0.0, 1.0)

    def values_at(x: np.ndarray) -> np.ndarray:
        return np.exp(x) * (1.0 + 0.3 * x)

    exp_equi = amplification_experiment(equi, values_at(np.linspace(0.0, 1.0, 19)), rel_error=1e-8, trials=64)
    exp_gauss = amplification_experiment(
        gauss_rule.weights, values_at(gauss_rule.nodes), rel_error=1e-8, trials=64
    )
    rows = [
        f"等距 19 点: worst={exp_equi['worst_ratio']:.4f}、精确最坏={exp_equi['adversarial_ratio']:.4f}、"
        f"Σ|w|={exp_equi['bound']:.4f}、Γ={exp_equi['condition']:.4f}",
        f"Gauss 19 点: worst={exp_gauss['worst_ratio']:.4f}、精确最坏={exp_gauss['adversarial_ratio']:.4f}、"
        f"Σ|w|={exp_gauss['bound']:.4f}、Γ={exp_gauss['condition']:.4f}",
    ]
    record(
        label="放大因子 ≤ Σ|w|（解析上界从不被破，且量级被顶到）",
        theory="worst ≤ 精确最坏 ≤ Σ|w|（两种规则各一条）",
        measured="；".join(rows),
        tol="worst ≤ adversarial ≤ Σ|w|",
        passed=bool(
            exp_equi["worst_ratio"] <= exp_equi["adversarial_ratio"] <= exp_equi["bound"]
            and exp_gauss["worst_ratio"] <= exp_gauss["adversarial_ratio"] <= exp_gauss["bound"]
            and exp_equi["worst_ratio"] >= 0.1 * exp_equi["bound"]
        ),
        detail="input_error_bound = ε·|f|_∞·Σ|w|：它**就是**最坏情况，不是估计；"
        "精确最坏值取符号使每项同号（Σ|w_k v_k|），随机试验只能逼近它",
    )
    gamma = weight_diagnostics(equi).condition
    measured_ratio = exp_equi["adversarial_ratio"] / exp_gauss["adversarial_ratio"]
    record(
        label="同一函数下两种规则的放大比 ≈ Γ_equi/Γ_gauss",
        theory=f"{gamma:.4f}（两者相差 20% 以内）",
        measured=f"{measured_ratio:.4f}",
        tol=f"±{0.2 * gamma:.4f}",
        passed=bool(abs(measured_ratio - gamma) <= 0.2 * gamma),
        detail="函数值一模一样，唯一变的是权重与节点位置——"
        "放大不是被积函数的错，是公式的错（残余差异来自两套节点上的 |v| 分布不同）",
    )
    bound = input_error_bound(equi, 1e-8, float(np.max(np.abs(values_at(np.linspace(0.0, 1.0, 19))))))
    record(
        label="19 点等距规则的误差上界比 Gauss 大多少",
        theory=f"Γ={gamma:.4f}",
        measured=f"ε·|f|_∞·Σ|w| = {bound:.6e}",
        tol="-",
        passed=bool(bound > 0.0),
        detail="这就是「函数值算得准」与「积分算得准」之间的那条缝",
    )


@assertion("B", theory="同一个权重换个算法写法：条件数差九个数量级（Σw 从 1e-8 涨到 21）", chapter="2.4")
def test_weight_computation_is_the_ill_conditioned_part() -> None:
    """在 [a,b] 上直接展开 Lagrange 基 vs 先在 [−1,1] 上算再映射。

    这是本章的"实现期真踩过的坑"之一：两条路径数学上完全等价，
    数值上在 m=24 时差 9 个数量级；抓住它的是最弱的那条不变量 Σw=b−a。
    """
    from numcore.interp.poly import poly_from_roots

    def direct(m: int, a: float, b: float) -> np.ndarray:
        x = np.linspace(a, b, m)
        w = np.zeros(m, dtype=float)
        for j in range(m):
            others = np.delete(x, j)
            coef = poly_from_roots(others)
            top = coef.size - 1
            total = 0.0
            for i, c in enumerate(coef):
                power = top - i
                total += float(c) * (b ** (power + 1) - a ** (power + 1)) / (power + 1)
            w[j] = total / float(np.prod(x[j] - others))
        return w

    rows = []
    ok = True
    for m in (9, 16, 19, 24):
        bad_sum = abs(float(np.sum(direct(m, 0.0, 1.0))) - 1.0)
        good_sum = abs(float(np.sum(newton_cotes_weights(m, 0.0, 1.0))) - 1.0)
        rows.append(f"m={m}: 直接 {bad_sum:.3e} / 映射 {good_sum:.3e}")
        if not (bad_sum > 100.0 * good_sum and good_sum <= 1e-5):
            ok = False
    record(
        label="同一个权重的两种算法写法",
        theory="直接展开在 [0,1] 上条件极坏（m=24 时 Σw 偏 21），映射后偏 3.3e-6",
        measured="；".join(rows),
        tol="直接/映射 > 100 且映射 ≤ 1e-5",
        passed=bool(ok),
        detail="与 2.3 章「单项式基 vs Chebyshev 基」是同一件事："
        "数学恒等不代表数值等价，而发现它的往往是最弱的那条不变量",
    )


@assertion("B", theory="误差估计器估的是自己的误差：窄峰上先低估 800 倍，细分后才追上", chapter="2.4")
def test_adaptive_estimator_underestimates_on_a_narrow_spike() -> None:
    """用**同一个结果对象的前缀**做证据：第 1 个区间上 est << true，最后一块才收敛。"""
    item = integrand("spike")
    res = adaptive_simpson(item.f, item.a, item.b, 1e-6, max_depth=12, exact=item.exact)
    assert res.true_err_cum is not None
    early_est = float(abs(res.est[0]))
    early_true = float(abs(res.true_err_cum[0]))
    early_ratio = early_est / early_true
    record(
        label="第一个区间上估计误差 << 真误差",
        theory="比值 << 1（初始三点 Simpson 全落在峰外）",
        measured=f"est={early_est:.3e}、true={early_true:.3e}、比值 {early_ratio:.3e}",
        tol="≤ 0.01",
        passed=bool(early_ratio <= 0.01),
        detail="峰宽 0.02 而初始步长 0.5：三点（0、0.5、1）谁都没碰到峰——"
        "嵌套法则之差在峰外是恒等于 0 的，它「有理有据」地报了个 0",
    )
    late_est = float(abs(res.error_estimate))
    late_true = float(abs(res.true_error or 0.0))
    record(
        label="细分之后估计器追上真误差",
        theory="同一对象上比值升到 O(1)",
        measured=f"est={late_est:.3e}、true={late_true:.3e}、比值 {late_est / late_true:.3f}",
        tol="≥ 1",
        passed=bool(late_est / late_true >= 1.0),
        detail=f"{res.n_intervals} 个区间、{res.n_evals} 次求值——估计器不是「总是」骗人，是「有时」骗人",
    )


@assertion("B", theory="误差估计器在舍入地板附近会假收敛：输入噪声它看不见（陷阱③后半）", chapter="2.4")
def test_error_estimator_fakes_convergence_at_the_roundoff_floor() -> None:
    """给光滑被积函数加 1e-9 的噪声：真误差停在 1.2e-12，估计器照旧报 2.7e-15。"""
    item = integrand("exp")
    xs = np.linspace(item.a, item.b, 17)
    scale = float(np.max(np.abs(item.values(xs))))

    def noisy(x: object) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return item.values(arr) + 1e-9 * scale * np.sin(1.0e3 * arr)

    res = adaptive_simpson(noisy, item.a, item.b, 1e-14, max_depth=20, exact=item.exact)
    est = float(abs(res.error_estimate))
    true = float(abs(res.true_error or 0.0))
    record(
        label="估计器停在地板上而真误差是它的数百倍",
        theory="true/est >> 1：估计器只看嵌套法则之差，看不到函数值里的噪声",
        measured=f"est={est:.3e}、true={true:.3e}、true/est={true / max(est, U):.3e}",
        tol="≥ 10",
        passed=bool(true / max(est, U) >= 10.0),
        detail=f"输入噪声 1e-9、tol=1e-14、区间 {res.n_intervals} 个："
        "被积函数值的噪声被 Σ|w| 收集起来，越细分收集得越多",
    )
    clean = romberg_table(item.f, item.a, item.b, 8, exact=item.exact)
    noisy_tab = romberg_table(noisy, item.a, item.b, 8, exact=item.exact)
    diag_clean = np.abs(np.asarray(clean.diagonal(), dtype=float) - item.exact)
    diag_noisy = np.abs(np.asarray(noisy_tab.diagonal(), dtype=float) - item.exact)
    floor_clean = float(np.min(diag_clean[4:]))
    floor_noisy = float(np.min(diag_noisy[4:]))
    record(
        label="Romberg 对角线：无噪声时触 u，有噪声时被抬高三个数量级",
        theory="干净：≤ 4u；带 1e-9 噪声：≥ 100u",
        measured=f"干净 {floor_clean:.3e}、带噪声 {floor_noisy:.3e}（比 {floor_noisy / max(floor_clean, U):.3e}）",
        tol="clean ≤ 4u 且 noisy ≥ 100u",
        passed=bool(floor_clean <= 4.0 * U and floor_noisy >= 100.0 * U),
        detail="外推把噪声放大 A_j ≤ 1.9692603 倍，截断误差却在第 6 层就没了："
        "两条完全不同的机制（自适应看法则之差、Romberg 看相邻层之差）给出同一个教训",
    )


@assertion("B", theory="折点落在节点上时梯形法逐段精确：误差的奇偶效应是巧合，不是阶", chapter="2.4")
def test_cusp_parity_effect_is_a_coincidence_not_an_order() -> None:
    """k 为偶数时误差 ~1e-16，k 为奇数时 ~1e-2：差 14 个数量级。"""
    item = integrand("cusp")
    even = abs(composite_rule(item.f, item.a, item.b, 8, rule="trapezoid") - item.exact)
    odd = abs(composite_rule(item.f, item.a, item.b, 9, rule="trapezoid") - item.exact)
    record(
        label="折点在节点上（偶数面板）vs 不在（奇数面板）",
        theory="偶数：|Q−I| ≈ 0；奇数：O(h²) ≈ 1.2e-2",
        measured=f"k=8: {even:.3e}；k=9: {odd:.3e}；比 {odd / max(even, U):.3e}",
        tol="even ≤ 1e-14 且 odd ≥ 1e-3",
        passed=bool(even <= 1e-14 and odd >= 1e-3),
        detail="|x| 的两段都是直线：只要折点落在节点上，复合梯形就逐段精确——"
        "这时双对数图上的「阶」是无穷大，但它不是公式的性质",
    )
    rows = []
    for k in (8, 16, 32, 64):
        e_even = abs(composite_rule(item.f, item.a, item.b, k, rule="trapezoid") - item.exact)
        e_odd = abs(composite_rule(item.f, item.a, item.b, k + 1, rule="trapezoid") - item.exact)
        rows.append(f"k={k}: {e_even:.1e}/{e_odd:.1e}")
    record(
        label="奇偶效应在整条曲线上的复现",
        theory="偶数面板恒在舍入地板、奇数面板按 h² 下降",
        measured="；".join(rows),
        tol="-",
        passed=True,
        detail="偶数面板上的 1e-16 不是「精度极高」，是「折点恰好被采到」",
    )


@assertion("B", theory="没有换元就用 substituted_error_curve 必须报错，不能返回 NaN", chapter="2.4")
def test_missing_substitution_raises_instead_of_returning_nan() -> None:
    """NaN 会被地板截断悄悄「修好」，所以这里要求显式报错。"""
    raised = ""
    try:
        substituted_error_curve(integrand("spike"), np.asarray([8, 16], dtype=int))
    except ValueError as exc:
        raised = f"{type(exc).__name__}: {exc}"
    record(
        label="缺换元时抛 ValueError",
        theory="ValueError（而不是 NaN 数组）",
        measured=raised or "没有抛异常",
        tol="-",
        passed=bool(raised),
        detail="实测教训：半无穷换元（x=e^(−t)）被误用时 composite_rule 返回 NaN，"
        "而 NaN 经过「地板截断」看起来和正常曲线一模一样——所以入口必须拦",
    )
    ok = [f"{n}:{'有' if integrand(n).substitution is not None else '无'}" for n in ("inv_sqrt", "sqrt", "log")]
    domains = [
        f"{n}:{integrand(n).substitution.domain}" for n in ("inv_sqrt", "sqrt", "log")  # type: ignore[union-attr]
    ]
    record(
        label="带换元的被积函数：换元域必须有限",
        theory="inv_sqrt / sqrt / log 都有有限区间换元",
        measured="；".join(ok) + " | " + "；".join(domains),
        tol="-",
        passed=bool(all(integrand(n).substitution is not None for n in ("inv_sqrt", "sqrt", "log"))),
        detail="等距规则没法直接在 [0,∞) 上验证：域里有 inf 就得不到可拟合的误差曲线",
    )


@assertion("B", theory="tanh–sinh 的饱和起点 = 2·t*/h + 1（t* 由 1−tanh(z) < u 定出）", chapter="2.4")
def test_tanh_sinh_saturation_threshold_matches_the_analytic_abscissa() -> None:
    """饱和节点必须按 0 计（否则 0·inf = NaN），起点由解析横坐标决定。"""
    t_star = float(saturation_abscissa())
    rows = []
    for h in (2.0**-3, 2.0**-4, 2.0**-5):
        predicted = 2 * int(math.ceil(t_star / h - 1e-12)) + 1
        observed = None
        for m in range(1, 400):
            if tanh_sinh_rule(m, h=h).n_saturated > 0:
                observed = 2 * m + 1
                break
        assert observed is not None
        assert_leq(
            abs(observed - predicted),
            4,
            label=f"h=2^{int(math.log2(h))}：饱和起点与解析预测相差不超过 4 个节点",
            detail=f"预测 {predicted}、实测 {observed}",
        )
        rows.append(f"h=2^{int(math.log2(h))}: 预测 {predicted}、实测 {observed}")
    record(
        label="饱和起点 = 2·t*/h + 1",
        theory=f"t* = {t_star:.6f}",
        measured="；".join(rows),
        tol="±4 个节点（离散化）",
        passed=True,
        detail="超过这个点数之后再加节点只是往公式里塞 0·inf：必须按 0 计（否则整帧 NaN）",
    )
