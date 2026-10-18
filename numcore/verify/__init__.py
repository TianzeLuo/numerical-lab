"""``numcore.verify`` —— 断言框架（先有裁判，再有运动员）。

对外 API 一屏速查::

    from numcore.verify import (
        assertion, Measured,                       # 登记
        fit_order, assert_order, consecutive_orders, reference_line,   # A 类：收敛阶
        estimate_linear_factor, assert_linear_convergence,             # C 类：收敛因子
        find_threshold, assert_threshold, ThresholdResult,             # C 类：临界阈值（二分实测）
        assert_within, assert_leq, assert_conserved, assert_bounded_drift,
        assert_linear_drift, assert_monotone, assert_orthogonal, assert_symmetric,  # B 类
        unit_roundoff, scaled_tol, relative_error, safe_log,           # 容差与安全运算
        mean_ci, std_err, effective_sample_size,                       # 统计
    )
"""

from __future__ import annotations

from .errors import AssertionFailure, fmt_value
from .invariants import (
    assert_all_finite,
    assert_bounded_drift,
    assert_conserved,
    assert_leq,
    assert_linear_drift,
    assert_monotone,
    assert_orthogonal,
    assert_symmetric,
    assert_within,
)
from .order import (
    OrderFit,
    WindowDiagnosis,
    assert_order,
    asymptotic_window,
    consecutive_orders,
    diagnose_order,
    fit_order,
    format_fit,
    reference_line,
)
from .rates import (
    ThresholdResult,
    assert_eventually_decreasing,
    assert_linear_convergence,
    assert_threshold,
    estimate_linear_factor,
    find_threshold,
)
from .report import (
    AssertionRecord,
    Measured,
    assertion,
    chapter_from_nodeid,
    fail,
    markdown_table,
    record,
    records,
    clear_reset_events,
    isolated_report,
    reset_events,
    reset_report,
    summary,
    suppressed,
    write_report,
)
from .stats import (
    autocorrelation,
    effective_sample_size,
    mean_ci,
    norm_cdf,
    norm_ppf,
    pearson,
    r_squared,
    std_err,
    t_ppf,
)
from .tolerance import (
    MACHINE_EPS,
    UNIT_ROUNDOFF,
    cancellation_digits,
    count_significant_digits,
    frobenius,
    is_all_finite,
    machine_eps,
    machine_tol,
    max_abs,
    relative_error,
    safe_divide,
    safe_log,
    safe_log10,
    safe_sqrt,
    scaled_tol,
    ulp,
    unit_roundoff,
)

__all__ = [
    # 异常与格式化
    "AssertionFailure",
    "fmt_value",
    # 登记
    "assertion",
    "Measured",
    "AssertionRecord",
    "record",
    "fail",
    "records",
    "reset_report",
    "summary",
    "suppressed",
    "write_report",
    "markdown_table",
    "chapter_from_nodeid",
    # A 类
    "OrderFit",
    "fit_order",
    "consecutive_orders",
    "WindowDiagnosis",
    "diagnose_order",
    "asymptotic_window",
    "assert_order",
    "reference_line",
    "format_fit",
    # C 类
    "ThresholdResult",
    "find_threshold",
    "assert_threshold",
    "estimate_linear_factor",
    "assert_linear_convergence",
    "assert_eventually_decreasing",
    # B 类
    "assert_within",
    "assert_leq",
    "assert_all_finite",
    "assert_conserved",
    "assert_bounded_drift",
    "assert_linear_drift",
    "assert_monotone",
    "assert_orthogonal",
    "assert_symmetric",
    # 容差
    "MACHINE_EPS",
    "UNIT_ROUNDOFF",
    "machine_eps",
    "unit_roundoff",
    "ulp",
    "scaled_tol",
    "machine_tol",
    "relative_error",
    "max_abs",
    "frobenius",
    "safe_log",
    "safe_log10",
    "safe_divide",
    "safe_sqrt",
    "is_all_finite",
    "count_significant_digits",
    "cancellation_digits",
    # 统计
    "norm_cdf",
    "norm_ppf",
    "t_ppf",
    "mean_ci",
    "std_err",
    "effective_sample_size",
    "autocorrelation",
    "pearson",
    "r_squared",
]
