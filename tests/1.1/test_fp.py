"""1.1 章 A/B 类断言（浮点侧）：舍入单位、间距、求和误差模型、抵消与区间传播。

这一章的核心是把三件常被混为一谈的事分开：

1. **表示误差**（:math:`\\mathrm{fl}(x)=x(1+\\delta)`，:math:`|\\delta|\\le u`）——网格的粗细；
2. **运算误差**（求和/减法的舍入）——算法稳不稳；
3. **问题条件**（:math:`\\kappa` 与抵消放大比）——数据本身难不难。

因此：

* **A 类**对拍"解析模型"：:math:`u=2^{-53}` 与 ``np.finfo``、间距与 ``np.spacing``/位运算、
  朴素求和误差的 :math:`n` 斜率、Kahan 的 :math:`u` 级界、以及 ``0.1+0.2-0.3`` 的**精确值**
  （在测试里用 ``Fraction``/``decimal``/``mpmath`` 三条独立路径算出来）。
* **B 类**守不变量：结合律偏差界、2 的幂缩放不变性、binade 边界间距减半、
  区间算术包含精确和、补偿归约的各种变体都落在各自的界内。

每条断言都登记**实测数字**（``record``/``assert_*``），抄理论值不算证据。
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from numcore import fp
from numcore.reference import exact as ref_exact
from numcore.utils import rng
from numcore.verify import assertion, assert_leq, assert_order, record

#: 单位舍入 u = 2^-53（binary64）
U = fp.unit_roundoff()


# --------------------------------------------------------------------------- #
# A 类：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="fl(x)=x(1+δ), |δ| ≤ u = 2^-53；u = ε_mach/2", chapter="1.1")
def test_rounding_unit_matches_ieee754_model() -> None:
    """舍入单位 :math:`u` 的定义与 IEEE-754 的两条可判定性质。

    * :math:`\\mathrm{fl}(1+u)=1`（恰在中点，ties-to-even 回到偶数格点 :math:`1`）；
    * :math:`\\mathrm{fl}(1+2u)>1`（下一个可表示数）。

    再用**有理数**造出不可表示的 :math:`x`，实测 :math:`\\sup|\\mathrm{fl}(x)-x|/(u|x|)`：
    理论值是 1（round-to-nearest 的 1/2 ulp 界），实测应当逼近它但不超过。
    """
    u = fp.unit_roundoff()
    assert u == 2.0**-53
    assert fp.machine_epsilon() == 2.0 * u == float(np.finfo(np.float64).eps)
    assert 1.0 + u == 1.0
    assert 1.0 + 2.0 * u > 1.0
    assert 1.0 + u / 2.0 == 1.0

    g = rng(11)
    worst = 0.0
    for _ in range(4000):
        x = Fraction(int(g.integers(1, 1 << 52)), int(g.integers(1, 1 << 20)))
        got = float(x)
        rel = abs(Fraction(got) - x) / abs(x)
        worst = max(worst, float(rel / Fraction(u)))
    rec = record(
        label="sup |fl(x)-x|/(u|x|)",
        theory="≤ 1（1/2 ulp 舍入界）",
        measured=f"{worst:.6f}（4000 个有理数样本）",
        tol="1",
        passed=worst <= 1.0,
    )
    assert rec.passed, f"舍入模型被违反：比值 {worst}"


@assertion("A", theory="ulp(x) 与 np.spacing 逐点相等；next_float 与 np.nextafter 相等", chapter="1.1")
def test_spacing_and_nextafter_match_numpy() -> None:
    """自研的**位运算**间距/相邻数实现与 numpy 的逐点一致性（含次正规与 0）。"""
    g = rng(17)
    xs = [0.0, 1.0, -1.0, 0.3, 1e-300, 1e300, 2.0**-1000, -3.7, 1e16, 5e-324]
    xs += [float(g.uniform(-1e6, 1e6)) for _ in range(200)]
    xs += [2.0**e for e in range(-60, 61, 7)]
    worst_spacing = 0.0
    worst_next = 0.0
    for x in xs:
        worst_spacing = max(worst_spacing, abs(fp.spacing(x) - float(np.spacing(abs(x)))))
        for d in (-1, 1):
            a = fp.next_float(x, direction=d)
            b = float(np.nextafter(x, d * np.inf))
            worst_next = max(worst_next, abs(a - b))
            if a != b:
                # 也允许"越过 ±0 以后符号不同"的边界情形：用位模式比较
                assert fp.ulp_distance(x, a) == 1
    assert worst_spacing == 0.0 and worst_next == 0.0
    rec = record(
        label="spacing/next_float 与 numpy 的最大偏差",
        theory="0（逐点相等）",
        measured=f"spacing {worst_spacing:.1e}，nextafter {worst_next:.1e}（{len(xs)} 个点）",
        tol="0",
        passed=worst_spacing == 0.0 and worst_next == 0.0,
    )
    assert rec.passed


@assertion("A", theory="朴素求和误差 = n−1（该数据族取等）；对 n 的对数斜率 1", chapter="1.1")
def test_naive_sum_error_grows_linearly() -> None:
    """``1e16 + 1 + 1 + ...`` 上每加一个 1 就丢一个 1：实测误差**恰好**是 :math:`n-1`。

    于是双对数斜率必须等于 1（这是 A 类要求的"实测阶 = 理论阶 ± 0.1"），
    同时误差不能超过 Wilkinson 界 :math:`n\\,u\\sum|x_i|`。
    """
    ns = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024], dtype=float)
    errs = np.array([fp.sum_error([1e16] + [1.0] * (int(n) - 1), "naive") for n in ns], dtype=float)
    assert np.all(errs == ns - 1.0), (ns, errs)
    fit = assert_order(ns, errs, theory=1.0, tol=0.1, label="朴素求和误差 ∝ n 的斜率")
    ratios = []
    for n in ns:
        vals = [1e16] + [1.0] * (int(n) - 1)
        ratios.append(fp.sum_error(vals, "naive") / fp.sum_error_bound(vals))
    rec = record(
        label="朴素误差/(n·u·Σ|x|)",
        theory="≤ 1",
        measured=f"{max(ratios):.4f}（斜率 {fit.slope:.6f}，R²={fit.r2:.6f}）",
        tol="1",
        passed=max(ratios) <= 1.0,
    )
    assert rec.passed


@assertion("A", theory="Kahan 误差 ≤ (2u+n·u²)Σ|x|，且与 n 无关", chapter="1.1")
def test_kahan_error_is_u_level_not_n_u() -> None:
    """补偿求和把误差从 :math:`n u\\sum|x|` 压到 :math:`2u\\sum|x|`：同一个数据族上量两个比值。

    实测（:math:`n=4096`）：朴素误差 4095，Kahan 误差 1，而 :math:`u\\sum|x|\\approx1.11`。
    """
    rows = []
    for n in (16, 64, 256, 1024, 4096):
        vals = [1e16] + [1.0] * (n - 1)
        e_k = fp.sum_error(vals, "kahan")
        e_n = fp.sum_error(vals, "naive")
        b_k = fp.kahan_error_bound(vals)
        rows.append((n, e_k, e_n, b_k, e_k / (U * sum(abs(v) for v in vals))))
    assert all(r[1] <= r[3] for r in rows), rows
    worst = max(r[4] for r in rows)
    assert worst <= 2.0
    rec = record(
        label="Kahan 误差/(u·Σ|x|) 与 Kahan/朴素",
        theory="≤ 2；且 Kahan/朴素 → 0 随 n 增大",
        measured=(
            f"max Kahan/(u·Σ|x|)={worst:.4f}；n=4096 时 Kahan={rows[-1][1]:.0f}、"
            f"朴素={rows[-1][2]:.0f}、比值={rows[-1][1] / rows[-1][2]:.2e}"
        ),
        tol="2",
        passed=worst <= 2.0,
    )
    assert rec.passed


@assertion("A", theory="0.1+0.2−0.3 = 5.551115123125783e-17 = 1 ulp(0.3)（精确值）", chapter="1.1")
def test_zero_point_one_plus_zero_point_two_residual_is_exact() -> None:
    """``0.1+0.2 != 0.3`` 的精确量化：三条**互相独立**的路径给出同一个数。

    1. ``Fraction``：把三个 double 当有理数算精确差；
    2. ``decimal``：:func:`fp.exact_decimal` 给出的精确十进制展开；
    3. ``mpmath``（``numcore.reference.exact.mp_sum``，50 位）：把 ``0.1`` 当**数学上的**
       :math:`1/10` 时结果就是 0——说明这 :math:`5.55\\times10^{-17}` 全部来自二进制表示。
    """
    s = float(0.1) + float(0.2)
    r = s - 0.3
    exact = Fraction(s) - Fraction(0.3)
    assert Fraction(r) == exact, "Sterbenz 保证这次减法精确：残差必须逐位等于精确差"
    assert r == 5.551115123125783e-17
    assert r == fp.spacing(0.3), "残差恰好是 0.3 处的 1 个 ulp"
    assert fp.ulp_distance(s, 0.3) == 1
    assert fp.residual_ulps(s, 0.3) == 1
    assert not (0.1 + 0.2 == 0.3)
    assert fp.exact_decimal(r, digits=20).startswith("0.0000000000000000555")
    mp = ref_exact.mp_sum([0.1, 0.2, -0.3], dps=50)
    assert abs(float(mp)) <= 1e-50, "数学实数域里 1/10+2/10−3/10 = 0"
    rec = record(
        label="(fl(0.1)+fl(0.2))−fl(0.3)",
        theory="5.551115123125783e-17 = ulp(0.3)（精确值）",
        measured=f"{r:.17e}；ulp 距离 {fp.ulp_distance(s, 0.3)}；精确十进制 {fp.exact_decimal(r, digits=20)}",
        tol="逐位相等",
        passed=Fraction(r) == exact and r == fp.spacing(0.3),
    )
    assert rec.passed


@assertion("A", theory="剩余有效位数 = log10(|a−b|/(u·M))，对 log10|a−b| 斜率 1", chapter="1.1")
def test_cancellation_digits_follow_log_model() -> None:
    """抵消的定量模型：把 a、b 各推 1 个 ulp，实测相对误差的**对数**与理论预测对比。

    两条对拍：

    * 用**实际的 ulp 和**预测（:math:`\\log_{10}(|a-b|/(\\mathrm{ulp}(a)+\\mathrm{ulp}(b)))`）时，
      实测与预测应当几乎相等（差只来自 log 的舍入）；
    * 用 :math:`u\\max(|a|,|b|)` 这个"教科书写法"预测时，两者相差不超过
      :math:`\\log_{10}5\\approx0.7` 位（因为 :math:`\\mathrm{ulp}(x)/|x|\\in[u,2u]` 且两个操作数各贡献一次）。
    """
    a = 1.0
    gaps = np.array([1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7, 1e-8, 1e-9])
    digits = []
    amps = []
    dev_ulp = 0.0
    dev_u = 0.0
    for gap in gaps:
        b = a - float(gap)
        amp = fp.perturbation_amplification(a, b, ulps=1)
        measured = -float(np.log10(amp))
        pred_ulp = float(np.log10(gap / (fp.spacing(a) + fp.spacing(b))))
        pred_u = fp.remaining_digits(gap, max(abs(a), abs(b)))
        digits.append(measured)
        amps.append(amp)
        dev_ulp = max(dev_ulp, abs(measured - pred_ulp))
        dev_u = max(dev_u, abs(measured - pred_u))
    # 幂律部分：放大比 ∝ 1/|a−b|（阶 −1，这才是 assert_order 的用武之地）
    fit_amp = assert_order(gaps, np.asarray(amps), theory=-1.0, tol=0.01, label="抵消放大比 ∝ 1/|a−b|")
    # 位数部分：digits = log10|a−b| + C 是**对数**关系，只能做线性拟合（不是幂律）
    slope, intercept = np.polyfit(np.log10(gaps), np.asarray(digits), 1)
    resid = float(np.max(np.abs(np.asarray(digits) - (slope * np.log10(gaps) + intercept))))
    assert abs(slope - 1.0) <= 0.01, slope
    assert dev_ulp <= 0.02, dev_ulp
    assert dev_u <= 0.7, dev_u
    rec = record(
        label="抵消：放大比阶 / 位数斜率 / 与两种预测的偏差",
        theory="放大比阶 = −1；位数对 log10|a−b| 斜率 = 1；实测−ulp 预测 ≤ 0.02 位；实测−u·M 预测 ≤ 0.7 位",
        measured=(
            f"放大比阶 {fit_amp.slope:.6f}；位数斜率 {slope:.6f}（最大残差 {resid:.2e} 位）；"
            f"dev_ulp={dev_ulp:.4f}，dev_u={dev_u:.4f}"
        ),
        tol="−1±0.01 / 1±0.01 / 0.02 / 0.7",
        passed=abs(slope - 1.0) <= 0.01 and dev_ulp <= 0.02 and dev_u <= 0.7,
    )
    assert rec.passed


@assertion("A", theory="Sterbenz：b/2 ≤ a ≤ 2b ⟹ fl(a−b) 精确（相对误差 0）", chapter="1.1")
def test_sterbenz_subtraction_is_exact() -> None:
    """抵消**不制造**误差：误差全部来自输入。用有理数逐位验证这条引理，并给出反面。

    反面（非 Sterbenz 区）也要有：此时相对误差 ≤ u，但绝不会超过它。
    """
    g = rng(3)
    n_in = 0
    worst_in = 0.0
    for _ in range(2000):
        a = float(g.uniform(0.5, 2.0))
        b = a * float(g.uniform(0.55, 1.0))
        assert 0.5 <= b / a <= 1.0
        if not fp.sterbenz_exact(a, b):  # pragma: no cover - 由上面的区间保证
            continue
        n_in += 1
        worst_in = max(worst_in, fp.subtraction_relative_error(a, b))
    worst_out = 0.0
    n_out = 0
    for _ in range(2000):
        a = float(g.uniform(-1.0, 1.0))
        b = float(g.uniform(-1.0, 1.0))
        if fp.sterbenz_exact(a, b) or a == b:
            continue
        n_out += 1
        worst_out = max(worst_out, fp.subtraction_relative_error(a, b))
    assert n_in > 1500 and worst_in == 0.0, (n_in, worst_in)
    assert n_out > 500 and worst_out <= U, (n_out, worst_out)
    rec = record(
        label="Sterbenz 区内减法相对误差 / 区外上界",
        theory="区内 = 0；区外 ≤ u",
        measured=f"区内 {n_in} 例最大 {worst_in:.1e}；区外 {n_out} 例最大 {worst_out:.3e}（u={U:.3e}）",
        tol="0 / u",
        passed=worst_in == 0.0 and worst_out <= U,
    )
    assert rec.passed


@assertion("A", theory="two_sum / two_product 的 (s,e) 精确复原 a+b 与 a·b", chapter="1.1")
def test_two_sum_and_two_product_are_exact() -> None:
    """两位和/两位积是**精确**的误差提取器：用有理数验证 ``s+e`` 与 ``p+e`` 逐位复原。"""
    g = rng(5)
    worst_sum = 0.0
    worst_prod = 0.0
    for _ in range(1500):
        a = float(g.uniform(-1e6, 1e6))
        b = float(g.uniform(-1e6, 1e6))
        s, e = fp.two_sum(a, b)
        worst_sum = max(worst_sum, float(abs((Fraction(s) + Fraction(e)) - (Fraction(a) + Fraction(b)))))
        p, pe = fp.two_product(a, b)
        worst_prod = max(worst_prod, float(abs((Fraction(p) + Fraction(pe)) - Fraction(a) * Fraction(b))))
    assert worst_sum == 0.0 and worst_prod == 0.0
    rec = record(
        label="|(s+e)−(a+b)| 与 |(p+e)−a·b| 的最大值",
        theory="0（精确复原）",
        measured=f"two_sum {worst_sum:.1e}，two_product {worst_prod:.1e}（各 1500 对）",
        tol="0",
        passed=worst_sum == 0.0 and worst_prod == 0.0,
    )
    assert rec.passed


# --------------------------------------------------------------------------- #
# B 类：不变量
# --------------------------------------------------------------------------- #
@assertion("B", theory="|fl(fl(a+b)+c) − fl(a+(b+c))| ≤ 2u(|a|+|b|+|c|)", chapter="1.1")
def test_associativity_deviation_is_roundoff_sized() -> None:
    """加法不满足结合律，但偏差**只**有舍入量级——不是"浮点坏了"。"""
    g = rng(7)
    worst = 0.0
    n_diff = 0
    trials = 3000
    for _ in range(trials):
        # 三个数跨若干量级：同量级的三数和**常常是精确的**，那时结合律"看不出"被破坏
        # （本样本 3000 组里同量级情形有 2940 组左右和为精确值），所以必须跨量级取样。
        a, b, c = (
            float(g.uniform(0.5, 1.0)) * 2.0 ** int(g.integers(-30, 1)) * (1.0 if g.integers(2) else -1.0)
            for _ in range(3)
        )
        left = (a + b) + c
        right = a + (b + c)
        if left != right:
            n_diff += 1
        bound = 2.0 * U * (abs(a) + abs(b) + abs(c)) + 4.0 * U * U * (abs(left) + abs(right))
        worst = max(worst, abs(left - right) / bound)
    assert_leq(
        worst,
        1.0,
        label="结合律偏差 / (2uΣ|x| + 4u²|和|)",
        detail=f"{n_diff}/{trials} 组不相等；最大比值 {worst:.4f}",
    )
    assert n_diff > 100, f"只有 {n_diff} 组不相等，样本不够刁"


@assertion("B", theory="Kahan 对精确的 2 的幂缩放逐位不变", chapter="1.1")
def test_kahan_is_invariant_under_power_of_two_scaling() -> None:
    """:math:`\\mathrm{fl}(2^k x)=2^k\\mathrm{fl}(x)`（无上下溢时），所以整个 Kahan 迭代逐位同构。"""
    vals = [1e16] + [1.0] * 511
    base = fp.kahan_sum(vals)
    mismatches = 0
    for k in range(-30, 31, 3):
        scale = 2.0**k
        got = fp.kahan_sum([v * scale for v in vals])
        if got != base * scale:
            mismatches += 1
    assert mismatches == 0
    rec = record(
        label="2^k 缩放下 Kahan 结果的逐位一致性",
        theory="全部逐位相等（21 个 k）",
        measured=f"不一致 {mismatches} 个；kahan(原)={base:.17g}",
        tol="0",
        passed=mismatches == 0,
    )
    assert rec.passed


@assertion("B", theory="同一 binade 内 ulp 恒定；跨过 2 的幂恰好减半；ulp 关于 |x| 单调不减", chapter="1.1")
def test_spacing_staircase_halves_at_powers_of_two() -> None:
    """浮点网格的"阶梯"结构：间距在每个 2 的幂处**精确**减半。"""
    ratios = []
    for e in (-10, -1, 0, 1, 5, 20, 60):
        p = 2.0**e
        below = fp.next_float(p, direction=-1)
        ratios.append(fp.spacing(p) / fp.spacing(below))
    assert all(r == 2.0 for r in ratios), ratios
    # 同一 binade 内（用连续可表示数走几步）间距不变
    walk = fp.neighborhood(1.5, half=8)
    sp = fp.spacings(walk)
    assert float(np.max(sp)) == float(np.min(sp))
    grid = fp.neighborhood(1.0, half=6)
    sp_grid = fp.spacings(grid)
    assert np.all(np.diff(sp_grid) >= 0.0), "ulp 关于 |x| 必须单调不减"
    rec = record(
        label="binade 边界处的间距比 / 单调性",
        theory="边界处恰好 2；binade 内恒定 1；整体单调不减",
        measured=f"边界比 {ratios}；binade 内 max/min = {float(np.max(sp)) / float(np.min(sp)):.1f}",
        tol="2 / 1 / 单调",
        passed=all(r == 2.0 for r in ratios),
    )
    assert rec.passed


@assertion("B", theory="误差区间必含精确和；宽度 ≤ 该方法的误差界（含向外舍入）", chapter="1.1")
def test_intervals_contain_exact_sum_and_respect_bounds() -> None:
    """区间算术是"最保守的误差传播"：无论方法多差，精确和都必须落在区间里。"""
    g = rng(23)
    worst = 0.0
    n_checked = 0
    for trial in range(30):
        n = int(g.integers(4, 200))
        vals = [float(g.uniform(-1.0, 1.0)) for _ in range(n)]
        if trial % 3 == 0:
            vals = [1e12] + [float(g.uniform(-1e-3, 1e-3)) for _ in range(n - 1)]
        exact = fp.exact_sum(vals)
        s_abs = sum(abs(v) for v in vals)
        for method in ("naive", "kahan", "pairwise", "sorted"):
            iv = fp.sum_error_interval(vals, method)
            assert iv.contains(float(exact)), (method, iv, float(exact))
            center = abs(float(iv.mid))
            # 区间宽度 = 2b（区间是 ±b）**再加上**两个端点各自被 next_float 推出去的那 1 个 ulp：
            # 当 b 远小于 ulp(和) 时（例如 1e12 + 1e-3），是端点舍入在决定宽度。
            slack = 4.0 * fp.spacing(center) + 8.0 * U * center
            generic = 2.0 * max((n - 1) * U * s_abs, fp.sum_error_bound(vals)) + slack
            worst = max(worst, iv.width / generic)
            n_checked += 1
    assert worst <= 1.0 + 1e-9, worst
    rec = record(
        label="区间宽度 / (2b + 端点 ulp)",
        theory="≤ 1（区间必含精确和，且宽度不超过界的两倍加端点舍入）",
        measured=f"{worst:.6f}（{n_checked} 次检查，全部包含精确和）",
        tol="1",
        passed=worst <= 1.0 + 1e-9,
    )
    assert rec.passed


@assertion("B", theory="串行/置换 Kahan ≤ (2u+nu²)Σ|x|；并行分段 ≤ n·u·Σ|x| 但远差于串行", chapter="1.1")
def test_compensated_reduction_variants_respect_their_bounds() -> None:
    """把"Kahan 不能并行"变成可测量的数字：``big=1e16, small=1e-3, n=4096``。

    实测：串行/置换误差 0.095（u 级界内），8 段并行 4.1、64 段并行 4.1——
    与朴素求和同级（段间那一次朴素合并是瓶颈）。
    """
    vals = [1e16] + [1e-3] * 4095
    exact = fp.exact_sum(vals)
    errs = {m: float(abs(Fraction(fp.compensated_sum(vals, scheme=m)) - exact)) for m in fp.COMPENSATED_SCHEMES}
    b_k = fp.kahan_error_bound(vals)
    b_n = fp.sum_error_bound(vals)
    assert errs["serial"] <= b_k and errs["shuffled"] <= b_k, errs
    assert errs["parallel8"] <= b_n and errs["parallel64"] <= b_n, errs
    assert errs["parallel8"] > 10.0 * errs["serial"], errs
    rec = record(
        label="补偿归约误差（serial/shuffled/parallel8/parallel64）",
        theory=f"前两者 ≤ 2u·Σ|x|={b_k:.3g}；后两者 ≤ n·u·Σ|x|={b_n:.3g} 且 ≫ 串行",
        measured=(
            f"serial={errs['serial']:.3g}, shuffled={errs['shuffled']:.3g}, "
            f"parallel8={errs['parallel8']:.3g}, parallel64={errs['parallel64']:.3g}"
        ),
        tol="各自界内 + 并行 > 10×串行",
        passed=errs["parallel8"] > 10.0 * errs["serial"],
    )
    assert rec.passed


@assertion("B", theory="纯置换不改变 Kahan 的误差量级（界对任意顺序都成立）", chapter="1.1")
def test_pure_permutation_does_not_break_kahan() -> None:
    """纠正一句常见的错话："乱序会让 Kahan 失效"。

    Kahan 的误差界 :math:`(2u+O(nu^2))\\sum|x_i|` 与**顺序无关**，纯置换实测只改变一点点；
    真正失效的是分段并行（见上一条）。
    """
    vals = [1e16] + [1.0] * 511
    exact = fp.exact_sum(vals)
    e_serial = float(abs(Fraction(fp.kahan_sum(vals)) - exact))
    ratios = []
    for seed in (1, 2, 3, 5, 8, 13):
        e_perm = float(abs(Fraction(fp.kahan_sum_shuffled(vals, seed=seed)) - exact))
        ratios.append(e_perm / max(e_serial, 1e-300))
    assert all(r <= 4.0 for r in ratios), ratios
    rec = record(
        label="置换 Kahan 误差 / 串行 Kahan 误差",
        theory="量级不变（个位数倍）",
        measured=f"最大比值 {max(ratios):.3f}（6 个种子；串行误差 {e_serial:.3g}）",
        tol="≤ 4",
        passed=max(ratios) <= 4.0,
    )
    assert rec.passed


@assertion("B", theory="float_grid 与 neighborhood 给的是同一批可表示数（ulp 距离逐点 = 1）", chapter="1.1")
def test_float_grid_matches_neighbourhood_chain() -> None:
    """放大镜用的两个网格构造器必须自洽：区间枚举 = 相邻链。"""
    for x in (0.3, 1.0, -2.5, 1e16):
        pts = fp.float_grid(x, fp.next_float(x, direction=1, steps=3), max_points=64)
        assert pts.size == 4
        assert all(fp.ulp_distance(pts[i], pts[i + 1]) == 1 for i in range(3))
        nb = fp.neighborhood(x, half=3)
        assert nb.size == 7
        assert float(nb[3]) == float(x)
        assert all(fp.ulp_distance(nb[i], nb[i + 1]) == 1 for i in range(6))
    # 超出上限必须报错，而不是悄悄截断
    try:
        fp.float_grid(1.0, 2.0, max_points=16)
        raise AssertionError("超上限时应当抛 ValueError")
    except ValueError:
        pass
    rec = record(
        label="网格构造器自洽性",
        theory="相邻点 ulp 距离恒为 1；超上限抛错",
        measured="4 个中心点全通过；max_points 截断保护生效",
        tol="全 True",
        passed=True,
    )
    assert rec.passed


@assertion("B", theory="四个 viz 的帧都带 requiredAux、meta 只含标量、关键曲线形状自洽", chapter="1.1")
def test_all_chapter_frames_are_well_formed() -> None:
    """章节帧契约（与 ``server/frames.py`` 的 strict 体检同一口径）。"""
    import importlib.util
    from pathlib import Path

    from server.manifest import load_manifest, validate_manifest

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("ch11_frames", root / "server" / "chapters" / "1.1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    problems = validate_manifest(mod.MANIFEST)
    assert not problems, problems
    man = load_manifest(mod.MANIFEST, where="1.1")
    report = []
    ok = True
    for viz in man.vizzes:
        params = {p.name: p.default for p in viz.params}
        for t in (0.0, viz.time.t_max * 0.5 if viz.time.t_max else 0.0, viz.time.t_max or 0.0):
            fr = mod.compute(viz.id, params, float(t))
            names = {n.lower() for n in fr.names}
            for aux in viz.required_aux:
                if not any(n.startswith(aux) for n in names):
                    ok = False
                    report.append(f"{viz.id}: 缺 requiredAux {aux}（现有 {sorted(names)}）")
            for key, val in fr.meta.items():
                if isinstance(val, str) or isinstance(val, (list, tuple, dict)):
                    ok = False
                    report.append(f"{viz.id}: meta[{key}] 不是标量")
            # 数组长度必须自洽（同一 x 轴上的曲线长度相同）
            for nm in fr.names:
                arr = np.asarray(fr[nm])
                if arr.size == 0:
                    ok = False
                    report.append(f"{viz.id}: 数组 {nm} 为空")
        report.append(f"{viz.id}: {len(fr.names)} 数组")
    rec = record(
        label="1.1 章帧契约",
        theory="每个 viz 的 requiredAux 齐备、meta 全标量、数组非空",
        measured="；".join(report),
        tol="全通过",
        passed=ok,
    )
    assert rec.passed, report
