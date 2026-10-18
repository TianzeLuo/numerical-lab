"""断言框架自身的测试。

哲学：**先有裁判，再有运动员**。如果裁判本身不可信，后面 1000 条数值断言都是装饰品。
因此这里用"人造已知答案"逐条检验每一个 helper：

* 解析阶数：人造 ``e = C·h^p`` 必须被还原成 ``p``（误差 < 1e-10）；
* 二分阈值：人造边界 ``r* = 0.5123`` 必须被二分到 1e-9；
* 收敛因子：人造 ``r_k = 0.5^k`` 必须给出 ``ρ = 0.5``；
* 失败路径：错误的断言必须抛出**带理论值/实测值**的 ``AssertionFailure``；
* 元规则：被装饰但"没登记任何实测量"的测试必须判失败（禁止静默通过）。
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from numcore.utils import rng
from numcore.verify import (
    AssertionFailure,
    Measured,
    assert_all_finite,
    assert_bounded_drift,
    assert_conserved,
    assert_leq,
    assert_linear_convergence,
    assert_linear_drift,
    assert_monotone,
    assert_order,
    assert_orthogonal,
    record,
    assert_symmetric,
    assert_threshold,
    assert_within,
    assertion,
    cancellation_digits,
    consecutive_orders,
    count_significant_digits,
    effective_sample_size,
    estimate_linear_factor,
    find_threshold,
    fit_order,
    diagnose_order,
    markdown_table,
    max_abs,
    mean_ci,
    norm_cdf,
    norm_ppf,
    pearson,
    r_squared,
    isolated_report,
    records,
    reference_line,
    relative_error,
    reset_report,
    safe_divide,
    safe_log,
    scaled_tol,
    std_err,
    summary,
    suppressed,
    t_ppf,
    unit_roundoff,
    write_report,
)

# --------------------------------------------------------------------------- #
# A 类：收敛阶
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0, 4.0, 6.0, 1.5])
def test_fit_order_recovers_exact_slope(p: float) -> None:
    """人造 ``e = C·h^p`` 的拟合斜率必须**逐位**还原 p。"""
    h = 2.0 ** -np.arange(1, 11, dtype=float)
    err = 0.37 * h**p
    fit = fit_order(h, err)
    assert abs(fit.slope - p) < 1e-10, fit
    assert fit.r2 > 1 - 1e-12
    assert fit.stderr < 1e-10
    assert fit.n_used == 10 and fit.n_dropped == 0
    assert abs(fit.constant - 0.37) < 1e-12


def test_fit_order_with_noise_is_close() -> None:
    """带 1% 乘性噪声时，斜率仍应落在 2% 以内。"""
    h = 2.0 ** -np.arange(1, 13, dtype=float)
    err = h**2 * (1.0 + 0.01 * rng(7).standard_normal(h.size))
    fit = fit_order(h, err)
    assert abs(fit.slope - 2.0) < 0.04, fit


def test_fit_order_skip_drops_coarsest_points() -> None:
    """``skip`` 必须丢弃最前面（最粗）的点。"""
    h = 2.0 ** -np.arange(1, 11, dtype=float)
    err = h**2
    fit = fit_order(h, err, skip=3)
    assert fit.n_used == 7 and fit.n_dropped == 3
    assert np.allclose(fit.h, h[3:])


def test_fit_order_rejects_too_few_points() -> None:
    """有效点不足必须报错，而不是给出一个瞎猜的斜率。"""
    with pytest.raises(ValueError, match="有效点不足"):
        fit_order([0.1, 0.05], [1e-2, 1e-3])


def test_fit_order_drops_zero_and_negative_errors() -> None:
    """0 误差（精确到机器零）不携带阶数信息，必须被丢弃。"""
    h = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    err = np.array([0.16, 0.04, 0.0, -1e-3, 0.000625])
    fit = fit_order(h, err, min_points=2)
    assert fit.n_used == 3 and fit.n_dropped == 2
    assert abs(fit.slope - 2.0) < 1e-9


def test_consecutive_orders_are_exact() -> None:
    """``e = h^2``：每一步的局部阶都必须是 2。"""
    h = 2.0 ** -np.arange(0, 6, dtype=float)
    p = consecutive_orders(h, h**2)
    assert np.allclose(p, 2.0, atol=1e-12)


def test_diagnose_order_clean_power_law() -> None:
    """纯幂律：诊断必须说"全程可用"，且首尾局部阶一致。"""
    h = 2.0 ** -np.arange(0, 12, dtype=float)
    d = diagnose_order(h, h**3)
    assert not d.floored and not d.noisy
    assert abs(d.head_p - 3.0) < 1e-9 and abs(d.tail_p - 3.0) < 1e-9
    assert "全程可用" in d.note()
    fit = fit_order(h[d.slice()], (h**3)[d.slice()])
    assert abs(fit.slope - 3.0) < 1e-9


def test_diagnose_order_detects_roundoff_floor() -> None:
    """``e = h² + 1e-3``：细网格端触地板，诊断必须报警并建议截尾。"""
    h = 2.0 ** -np.arange(0, 14, dtype=float)
    err = h**2 + 1e-3
    d = diagnose_order(h, err, tol=0.1)
    assert d.floored, d.note()
    assert d.tail_p < 0.25 < d.head_p
    assert "舍入地板" in d.note()
    # 地板在细网格端，因此应保留"头部"（粗网格）若干点：显式切片，而不是让工具替你猜
    k = 4
    fit = fit_order(h[:k], err[:k])
    assert abs(fit.slope - 2.0) < 0.05, fit


def test_diagnose_order_detects_preasymptotic_head() -> None:
    """``e = h² + 0.3h``：局部阶从 2 漂到 1，诊断必须报告"非单一幂律"并给出建议窗口。"""
    h = 2.0 ** -np.arange(-3, 9, dtype=float)
    err = h**2 + 0.3 * h
    d = diagnose_order(h, err, tol=0.1)
    assert d.head_p > d.tail_p + 0.5, d
    assert d.noisy or d.floored
    assert d.suggest_tail >= 2


def test_fit_order_tail_parameter_truncates_fine_end() -> None:
    """``tail`` 必须只保留最后若干点（skip 之后计数）。"""
    h = 2.0 ** -np.arange(0, 10, dtype=float)
    err = h**2 + 1e-9
    fit = fit_order(h, err, skip=1, tail=4)
    assert fit.n_used == 4
    assert np.allclose(fit.h, h[-4:])


def test_assert_order_passes_and_records() -> None:
    """正确阶数：通过，并在报告中留下"理论值/实测值"。"""
    h = 2.0 ** -np.arange(1, 9, dtype=float)
    with isolated_report():  # 只在本测试内部要一张空表，退出时原来的记录会回来
        with suppressed():
            fit = assert_order(h, h**4, theory=4.0, label="simpson")
    assert abs(fit.slope - 4.0) < 1e-9


def test_assert_order_failure_message_has_theory_and_measured() -> None:
    """错误阶数：失败信息**必须**同时含理论值与实测值。"""
    h = 2.0 ** -np.arange(1, 9, dtype=float)
    with suppressed(), pytest.raises(AssertionFailure) as ei:
        assert_order(h, h**2, theory=4.0, label="假断言")
    msg = str(ei.value)
    assert "理论值" in msg and "实测值" in msg
    assert "4" in msg and "2" in msg
    assert ei.value.theory == 4.0
    assert abs(float(ei.value.measured) - 2.0) < 1e-6


def test_reference_line_has_theoretical_slope() -> None:
    """双对数参考线的斜率必须等于理论阶（防误导规则 2 的数据基础）。"""
    h = np.array([1.0, 0.5, 0.25])
    y = reference_line(h, 2.0, anchor=(1.0, 3.0))
    slope = math.log(y[0] / y[2]) / math.log(h[0] / h[2])
    assert abs(slope - 2.0) < 1e-12


# --------------------------------------------------------------------------- #
# C 类：收敛因子与阈值
# --------------------------------------------------------------------------- #


def test_estimate_linear_factor_exact() -> None:
    """``r_k = 0.5^k`` 的估计因子必须是 0.5。"""
    res = 0.5 ** np.arange(0, 20, dtype=float)
    assert abs(estimate_linear_factor(res) - 0.5) < 1e-12


def test_assert_linear_convergence_pass_and_fail() -> None:
    res = 0.8 ** np.arange(0, 30, dtype=float)
    with suppressed():
        rho = assert_linear_convergence(res, rate=0.8, rtol=0.01)
        assert abs(rho - 0.8) < 1e-6
        with pytest.raises(AssertionFailure, match="理论值"):
            assert_linear_convergence(res, rate=0.5, rtol=0.05)


def test_find_threshold_recovers_known_boundary() -> None:
    """人造边界 r* = 0.5123 必须被二分到 1e-9 以内。"""
    r_star = 0.5123
    res = find_threshold(lambda r: r <= r_star, 0.1, 1.0, tol=1e-12)
    assert abs(res.threshold - r_star) < 1e-9, res
    assert res.calls == res.iterations + 2
    assert res.lo <= r_star <= res.hi


def test_find_threshold_validates_assumptions() -> None:
    """谓词在端点同值 / 端点假设不符：必须报错，而不是返回垃圾。"""
    with pytest.raises(ValueError, match="取值相同"):
        find_threshold(lambda r: True, 0.1, 1.0)
    with pytest.raises(ValueError, match="端点假设不符"):
        find_threshold(lambda r: r <= 0.5, 0.1, 1.0, assume="false_at_lo")
    with pytest.raises(ValueError, match="lo < hi"):
        find_threshold(lambda r: r <= 0.5, 1.0, 0.1)


def test_find_threshold_reverse_orientation() -> None:
    """``false_at_lo`` 方向：找"由假变真"的边界。"""
    res = find_threshold(lambda r: r >= 0.25, 0.0, 1.0, assume="false_at_lo", tol=1e-12)
    assert abs(res.threshold - 0.25) < 1e-9


def test_assert_threshold_pass_and_fail() -> None:
    with suppressed():
        assert_threshold(0.5123, 0.5, rtol=0.03, label="热方程阈值 r=1/2")
        with pytest.raises(AssertionFailure, match="理论值"):
            assert_threshold(0.62, 0.5, rtol=0.02)


# --------------------------------------------------------------------------- #
# B 类：不变量
# --------------------------------------------------------------------------- #


def test_assert_within_and_leq() -> None:
    with suppressed():
        assert_within(1.0000001, 1.0, rtol=1e-5)
        with pytest.raises(AssertionFailure):
            assert_within(1.01, 1.0, rtol=1e-5)
        assert_leq(0.4999999, 0.5, rtol=1e-6)
        with pytest.raises(AssertionFailure):
            assert_leq(0.51, 0.5, rtol=1e-6)


def test_assert_conserved_detects_leak() -> None:
    """守恒量：机器精度内守恒通过；缓慢漏水必须被抓出来。"""
    mass_ok = np.full(50, 3.14159)
    mass_ok[10] += 1e-15  # 舍入级扰动应当容忍
    with suppressed():
        drift = assert_conserved(mass_ok, quantity="质量")
        assert drift < 1e-13
        leaky = 1.0 - 1e-6 * np.arange(50)
        with pytest.raises(AssertionFailure, match="理论值"):
            assert_conserved(leaky, quantity="质量")


def test_assert_all_finite() -> None:
    with suppressed():
        assert_all_finite(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(AssertionFailure):
            assert_all_finite(np.array([1.0, np.nan, np.inf]))


def test_assert_bounded_drift_vs_linear() -> None:
    """辛方法能量（有界振荡）与 RK4 能量（线性漂移）必须被区分开。"""
    k = np.arange(2000)
    bounded = 1.0 + 1e-3 * np.sin(0.1 * k)
    drifting = 1.0 - 1e-6 * k
    with suppressed():
        amp = assert_bounded_drift(bounded, tol=2e-3)
        assert amp < 2e-3
        with pytest.raises(AssertionFailure):
            assert_bounded_drift(drifting, tol=1e-3)
        b = assert_linear_drift(drifting, min_slope=1e-7)
        assert abs(b + 1e-6) < 1e-12
        with pytest.raises(AssertionFailure):
            assert_linear_drift(bounded, min_slope=1e-3)


def test_assert_monotone() -> None:
    with suppressed():
        assert_monotone(np.array([5.0, 4.0, 3.0, 2.0]))
        assert_monotone(np.array([1.0, 2.0, 3.0]), direction="nondecreasing")
        with pytest.raises(AssertionFailure):
            assert_monotone(np.array([1.0, 2.0, 1.5]))
        with pytest.raises(AssertionFailure):
            assert_monotone(np.array([1.0, 1.0, 1.0]), direction="increasing")


def test_assert_orthogonal_accepts_rotation_rejects_other() -> None:
    """手造 Givens 旋转（不借 numpy.linalg）验证正交性检查。"""
    theta = 0.7
    c, s = math.cos(theta), math.sin(theta)
    Q = np.array([[c, -s], [s, c]])
    Q4 = np.zeros((4, 4))
    Q4[:2, :2] = Q
    Q4[2:, 2:] = Q
    with suppressed():
        err = assert_orthogonal(Q4)
        assert err < 1e-15, err
        with pytest.raises(AssertionFailure):
            assert_orthogonal(np.array([[1.0, 0.5], [0.0, 1.0]]))


def test_assert_symmetric() -> None:
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    with suppressed():
        assert_symmetric(A)
        with pytest.raises(AssertionFailure):
            assert_symmetric(np.array([[2.0, 1.0], [0.0, 3.0]]))


def test_relative_error_and_max_abs() -> None:
    assert abs(relative_error([1.0, 2.0], [1.0, 2.0])) == 0.0
    assert abs(relative_error([1.0, 2.0], [2.0, 4.0]) - 0.5) < 1e-15
    assert max_abs(np.array([-3.0, 1.0])) == 3.0
    assert max_abs(np.array([])) == 0.0


# --------------------------------------------------------------------------- #
# 容差与安全运算
# --------------------------------------------------------------------------- #


def test_unit_roundoff_is_two_pow_minus_53() -> None:
    assert unit_roundoff() == 2.0**-53
    assert scaled_tol(1.0, n_ops=1) == 8.0 * 2.0**-53
    assert scaled_tol(1.0, n_ops=100) > scaled_tol(1.0, n_ops=10)


def test_safe_log_never_warns() -> None:
    """``safe_log`` 在 0/负值处必须安静地给出 -inf/nan（pytest 里 RuntimeWarning 即错误）。"""
    out = safe_log(np.array([1.0, 0.0, -1.0, np.e]))
    assert out[0] == 0.0
    assert out[1] == -np.inf
    assert np.isnan(out[2])
    assert abs(out[3] - 1.0) < 1e-15
    assert safe_log(0.0, floor=1e-300) < -600


def test_safe_divide_no_warning() -> None:
    out = safe_divide(np.array([1.0, 2.0]), np.array([0.0, 2.0]))
    assert out[0] == 0.0 and abs(out[1] - 1.0) < 1e-15


def test_count_significant_digits_semantics() -> None:
    """一致位数：相同数 → 封顶 15.95；相差一倍 → 0 位；相差 1e-9 → 约 9 位。"""
    assert count_significant_digits(1.0, 1.0) == pytest.approx(15.95, abs=0.01)
    assert count_significant_digits(2.0, 1.0) < 0.01
    assert 8.9 < count_significant_digits(1.0 + 1e-9, 1.0) < 9.1
    assert count_significant_digits(1e-15, 1.0) < 0.01  # 1e-15 与 1 几乎没有任何共同有效位


def test_cancellation_digits_models_loss_of_precision() -> None:
    """灾难性抵消：a−b 的结果越小，剩下的有效位数越少（1.1 章的定量结论）。"""
    assert cancellation_digits(1.0, 1.0) == pytest.approx(15.95, abs=0.01)   # 无抵消
    assert cancellation_digits(1e-8, 1.0) == pytest.approx(8.0, abs=0.05)    # 抵消掉 8 位
    assert cancellation_digits(1e-15, 1.0) == pytest.approx(0.9, abs=0.15)   # 几乎全丢
    assert cancellation_digits(0.0, 1.0) == 0.0


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #


def test_norm_ppf_matches_known_quantiles() -> None:
    assert abs(float(norm_ppf(0.975)) - 1.959963985) < 1e-6
    assert abs(float(norm_ppf(0.5))) < 1e-9
    assert abs(float(norm_ppf(0.995)) - 2.575829304) < 1e-6


def test_norm_cdf_roundtrip() -> None:
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    z = norm_ppf(p)
    back = norm_cdf(z)
    assert np.allclose(back, p, atol=1e-9)


def test_t_ppf_table_and_limits() -> None:
    assert abs(t_ppf(0.95, 1) - 6.314) < 1e-12
    assert abs(t_ppf(0.95, 10) - 1.812) < 1e-12
    assert abs(t_ppf(0.95, 100) - 1.959963985) < 2e-3  # df>30 用正态近似
    with pytest.raises(ValueError):
        t_ppf(0.93, 5)


def test_mean_ci_shrinks_as_one_over_sqrt_n() -> None:
    g = rng(11)
    x1 = g.standard_normal(100)
    x2 = g.standard_normal(10000)
    _, h1 = mean_ci(x1)
    _, h2 = mean_ci(x2)
    assert h2 < h1 / 5.0  # 样本量 100 倍 → 半宽约 1/10
    assert abs(std_err(x1) - np.std(x1, ddof=1) / 10.0) < 1e-15


def test_effective_sample_size_detects_correlation() -> None:
    g = rng(3)
    iid = g.standard_normal(4000)
    assert effective_sample_size(iid) > 0.7 * 4000
    ar = np.empty(4000)
    ar[0] = 0.0
    for i in range(1, 4000):  # AR(1)，ρ=0.9 → N_eff ≈ N(1−ρ)/(1+ρ) ≈ 210
        ar[i] = 0.9 * ar[i - 1] + float(g.standard_normal())
    ess = effective_sample_size(ar)
    assert ess < 0.2 * 4000, ess


def test_pearson_and_r_squared() -> None:
    x = np.linspace(0, 1, 50)
    assert abs(pearson(x, 3 * x + 1) - 1.0) < 1e-12
    assert abs(pearson(x, -2 * x) + 1.0) < 1e-12
    assert abs(r_squared(3 * x + 1, 3 * x + 1) - 1.0) < 1e-15
    assert r_squared(x, np.full_like(x, 0.5)) < 1.0


# --------------------------------------------------------------------------- #
# 元规则：登记、静默通过、报告
# --------------------------------------------------------------------------- #


def test_decorator_requires_measurement() -> None:
    """被装饰却"什么都没登记"的测试必须失败 —— 这就是"禁止静默通过"。"""

    @assertion("B", "空断言")
    def _empty() -> None:
        return None

    # 注意：本测试故意不在 suppressed() 下运行——"禁止静默通过"这条元规则
    # 必须被真正触发一次。用 isolated_report() 而不是 reset_report()：
    # 后者会连**整个会话**的登记一起清掉，导致 pytest 结束时根本不写报告。
    with isolated_report():
        with pytest.raises(AssertionFailure, match="未登记任何实测值"):
            _empty()
        assert any(not r.passed and r.unmeasured for r in records())


def test_decorator_records_returned_measurement() -> None:
    """返回 ``Measured`` 的测试必须出现在报告里，且带上理论值。"""

    @assertion("C", "人造收敛因子", id="fake_factor", chapter="0.0")
    def _factor() -> Measured:
        return Measured(value=0.5, tol=0.01, detail="人造")

    with suppressed():
        _factor()  # suppressed 下不登记，只验证不抛错

    @assertion("C", "人造收敛因子", id="fake_factor", chapter="0.0")
    def _factor2() -> Measured:
        return Measured(value=0.5, tol=0.01, detail="人造")

    with isolated_report():
        _factor2()
        recs = records()
        assert len(recs) == 1
        r = recs[0]
        assert r.id == "fake_factor" and r.kind == "C"
        assert r.measured == 0.5 and r.tol == 0.01 and r.passed
        assert r.chapter == "0.0"
        report_path = write_report("/tmp/nlab-probe/assertions.json")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["totals"]["C"] == 1
        assert payload["chapters"]["0.0"]["passed"] == 1
        assert "fake_factor" in markdown_table("0.0")
        assert summary()["totals"]["unmeasured"] == 0


def test_chapter_attribution_from_nodeid() -> None:
    from numcore.verify import chapter_from_nodeid

    assert chapter_from_nodeid("tests/1.2/test_lu.py::test_x") == "1.2"
    assert chapter_from_nodeid("tests/1.2/test_lu.py::TestC::test_x[a-b]") == "1.2"
    assert chapter_from_nodeid("tests/test_verify_framework.py::test_x") is None


def test_rng_is_deterministic() -> None:
    a = rng(42).standard_normal(5)
    b = rng(42).standard_normal(5)
    c = rng(43).standard_normal(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.array_equal(rng().standard_normal(3), rng(20261018).standard_normal(3))


# --------------------------------------------------------------------------- #
# 元测试：框架自己的测试不许毁掉会话报告
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def records_before_this_file() -> int:
    """进入本文件时，会话里已有的登记条数（整套跑时是前几章贡献的）。"""
    return len(records())


def test_framework_tests_do_not_wipe_the_session_report(records_before_this_file: int) -> None:
    """本文件跑完之后，会话登记表不许比进来时更少。

    背景：``pytest_sessionfinish`` 靠登记表写 ``dist/reports/assertions.json``。
    框架测试如果用 ``reset_report()`` 清表，整套 pytest 跑完报告就是空的、**静默不写**，
    G3 于是拿着上一次部分运行的旧报告报"已实现章节没有任何断言记录"。
    正确做法是 ``with isolated_report():``——退出时把原记录放回去。
    """
    assert len(records()) >= records_before_this_file, (
        "会话登记表被本文件的测试清空了：请用 numcore.verify.isolated_report() 而不是 reset_report()"
    )


def test_decorator_rejects_kind_mismatch() -> None:
    """声明 A 类、却只登记了 B 类记录 —— 也算"假证据"，必须失败。

    为什么要有这条规则：A 类断言的含义是"与解析解/参考实现对拍"，
    而 `assert_leq` / `assert_within` / `assert_monotone` 这些 helper 登记的 kind 固定是 `"B"`。
    于是一个 `@assertion("A")` 的测试只要全用 helper，就会出现
    "这一章有 N 条 A 类断言"是假话、而 pytest 全绿、报告里也有记录的情况。
    1.3 章真踩到过（交付报告第 4 条），当时只靠 `min_counts` 间接兜住。
    """

    @assertion("A", "解析对拍：本测试其实只记了 B")
    def _mismatched() -> None:
        record(label="只记一条 B", theory=1.0, measured=1.0, tol=0.0, passed=True, kind="B")

    with isolated_report():
        with pytest.raises(AssertionFailure, match="声明为 A 类"):
            _mismatched()
        assert any(not r.passed for r in records())

    # 显式声明 A 类记录（或把 helper 的 kind 改成 A）之后必须通过：
    @assertion("A", "解析对拍：显式记 A")
    def _explicit() -> None:
        record(label="显式 A 记录", theory=1.0, measured=1.0, tol=0.0, passed=True, kind="A")

    with isolated_report():
        _explicit()
        recs = records()
        assert len(recs) == 1 and recs[0].kind == "A" and recs[0].passed

    # helper 的缺省类别是 B，所以 @assertion("B") 用 helper 是合法的：
    @assertion("B", "不变量：helper 缺省 B")
    def _b_helper() -> None:
        assert_leq(1.0, 2.0, label="1 ≤ 2")

    with isolated_report():
        _b_helper()
        assert records() and all(r.kind == "B" for r in records())


def test_assert_order_can_record_kind_c() -> None:
    """`assert_order` 的缺省类别是 A；但"实测阶是否达到理论阶"也常被声明为 C 类。

    这条测试钉住 `kind=` 参数：不传就是 A（历史行为，全仓多数章节依赖它），
    传了就按传的记 —— 否则声明 C 的测试会因为"类别不符"直接失败（0.0 章真踩过）。
    """
    h = np.array([0.05, 0.025, 0.0125, 0.00625])
    err = 3.0 * h**2

    with isolated_report():
        assert_order(h, err, theory=2.0, tol=0.05, label="缺省类别")
        assert records()[-1].kind == "A"

    with isolated_report():
        assert_order(h, err, theory=2.0, tol=0.05, label="显式 C 类", kind="C")
        assert records()[-1].kind == "C"
