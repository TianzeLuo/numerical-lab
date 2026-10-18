"""收敛因子与临界阈值的**实测**工具（C 类断言的核心）。

两类问题：

1. **迭代法收敛因子**：残差满足 ``‖r_k‖ ≈ ρ^k·C``，则 ``ρ̂ = (r_{k+1}/r_k)`` 的几何平均
   （等价于尾部残差的对数斜率）；与理论谱半径 / 收敛界比较。
2. **稳定性阈值**：诸如热方程 ``r = αΔt/Δx² ≤ 1/2``、CFL 数 ``≤ 1``、
   BDF 的 ``A(α)`` 角——理论值写死在教材里，本项目要求**二分实测**：
   让脚本自己找到"从稳定到爆炸"的边界，再与理论值比对。
   这样"阈值 = 1/2"就不是抄来的结论，而是被测量出来的事实。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .errors import fmt_value
from .report import fail, record
from .tolerance import safe_log, scaled_tol

__all__ = [
    "estimate_linear_factor",
    "assert_linear_convergence",
    "ThresholdResult",
    "find_threshold",
    "assert_threshold",
    "assert_eventually_decreasing",
]


def estimate_linear_factor(
    residuals: Any,
    *,
    skip: int = 1,
    tail: int | None = None,
) -> float:
    """由残差序列估计线性收敛因子 ``ρ``（几何平均，尾部窗口）。

    参数
    ----
    residuals
        ``‖r_k‖`` 序列（正数）。
    skip
        丢弃最前面的 ``skip`` 个残差（初始瞬态不一定在线性渐近区）。
    tail
        只用最后 ``tail`` 个比值；``None`` 表示用全部可用值。
    """
    r = np.asarray(residuals, dtype=float).ravel()
    if skip:
        r = r[int(skip) :]
    r = r[np.isfinite(r) & (r > 0)]
    if r.size < 2:
        raise ValueError(f"有效残差不足（需 ≥2 个正数），得到 {r.size} 个")
    ratios = r[1:] / r[:-1]
    if tail is not None:
        ratios = ratios[-int(tail) :]
    return float(np.exp(np.mean(safe_log(ratios))))


def assert_linear_convergence(
    residuals: Any,
    *,
    rate: float | None = None,
    rtol: float = 0.2,
    skip: int = 1,
    tail: int | None = None,
    label: str | None = None,
    detail: str = "",
) -> float:
    """断言残差以因子 ``rate`` 线性收敛（``rate=None`` 时只要求 ``ρ < 1``）。"""
    rho = estimate_linear_factor(residuals, skip=skip, tail=tail)
    r = np.asarray(residuals, dtype=float).ravel()
    if rate is None:
        ok = rho < 1.0
        theory: Any = "< 1（线性收敛）"
        tol_txt = "严格小于 1"
    else:
        ok = abs(rho - rate) <= rtol * max(abs(rate), 1e-300)
        theory = rate
        tol_txt = f"±{rtol:.0%} 相对"
    name = label or "线性收敛因子"
    info = f"由 {r.size} 个残差估计（skip={skip}, tail={tail}）" + (f"；{detail}" if detail else "")
    if not ok:
        fail(label=name, theory=theory, measured=rho, tol=tol_txt, detail=info, kind="C")
    record(label=name, theory=theory, measured=rho, tol=tol_txt, passed=True, detail=info, kind="C")
    return rho


@dataclass(frozen=True)
class ThresholdResult:
    """二分实测的临界值。"""

    threshold: float
    lo: float
    hi: float
    iterations: int
    calls: int
    note: str = ""

    @property
    def bracket(self) -> tuple[float, float]:
        return (self.lo, self.hi)

    @property
    def width(self) -> float:
        return abs(self.hi - self.lo)

    def __str__(self) -> str:  # pragma: no cover - 人读
        return f"阈值 ≈ {fmt_value(self.threshold)}（区间宽 {self.width:.2e}，{self.iterations} 次二分）"


def find_threshold(
    predicate: Callable[[float], bool],
    lo: float,
    hi: float,
    *,
    tol: float = 1e-9,
    max_iter: int = 200,
    assume: str = "true_at_lo",
    note: str = "",
) -> ThresholdResult:
    """二分寻找 ``predicate`` 由真变假的临界点。

    参数
    ----
    predicate
        单调谓词，例如 ``lambda r: solution_stays_bounded(r, nt=200)``。
    assume
        ``"true_at_lo"``：``predicate(lo)`` 为真、``predicate(hi)`` 为假（如 r ≤ 1/2 稳定）；
        ``"false_at_lo"``：相反。
    tol
        区间宽度收敛判据（相对 ``max(|lo|,|hi|,1)`` 的绝对宽度）。
    """
    if assume not in ("true_at_lo", "false_at_lo"):
        raise ValueError("assume 只能是 'true_at_lo' 或 'false_at_lo'")
    lo = float(lo)
    hi = float(hi)
    if not lo < hi:
        raise ValueError(f"需要 lo < hi，得到 lo={lo}, hi={hi}")
    f_lo = bool(predicate(lo))
    f_hi = bool(predicate(hi))
    calls = 2
    if f_lo == f_hi:
        raise ValueError(
            "谓词在区间端点取值相同，无法二分："
            f"predicate({lo})={f_lo}, predicate({hi})={f_hi}。请扩大区间或检查谓词。"
        )
    want_lo = assume == "true_at_lo"
    if f_lo != want_lo:
        raise ValueError(
            f"端点假设不符：assume={assume} 但 predicate(lo)={f_lo}。"
            "请交换 lo/hi 或修改 assume。"
        )
    scale = max(abs(lo), abs(hi), 1.0)
    it = 0
    a, b = lo, hi
    while it < int(max_iter) and (b - a) > tol * scale:
        m = 0.5 * (a + b)
        fm = bool(predicate(m))
        calls += 1
        if fm == want_lo:
            a = m
        else:
            b = m
        it += 1
    return ThresholdResult(threshold=0.5 * (a + b), lo=a, hi=b, iterations=it, calls=calls, note=note)


def assert_threshold(
    measured: float | ThresholdResult,
    theory: float,
    *,
    rtol: float = 0.02,
    label: str | None = None,
    detail: str = "",
) -> float:
    """断言实测阈值 ``= 理论阈值``（相对容差 ``rtol``，默认 2%）。"""
    m = measured.threshold if isinstance(measured, ThresholdResult) else float(measured)
    name = label or "稳定性阈值"
    extra = ""
    if isinstance(measured, ThresholdResult):
        extra = f"二分区 [{fmt_value(measured.lo)}, {fmt_value(measured.hi)}]，{measured.iterations} 次迭代，{measured.calls} 次谓词调用"
        if measured.note:
            extra += f"；{measured.note}"
    if detail:
        extra += f"；{detail}"
    ok = abs(m - theory) <= rtol * max(abs(theory), 1e-300)
    if not ok:
        fail(label=name, theory=theory, measured=m, tol=f"±{rtol:.1%} 相对", detail=extra, kind="C")
    record(label=name, theory=theory, measured=m, tol=f"±{rtol:.1%} 相对", passed=True, detail=extra, kind="C")
    return m


def assert_eventually_decreasing(
    values: Any,
    *,
    after: int = 0,
    strict: bool = False,
    label: str | None = None,
) -> None:
    """断言序列从第 ``after`` 项起单调不增（收敛性的必要条件，B 类）。"""
    v = np.asarray(values, dtype=float).ravel()[int(after) :]
    if v.size < 2:
        raise ValueError("序列太短")
    d = np.diff(v)
    scale = max(float(np.max(np.abs(v))), 1e-300)
    slack = scaled_tol(scale, n_ops=v.size)
    ok = bool(np.all(d < 0)) if strict else bool(np.all(d <= slack))
    name = label or "序列单调不增"
    info = f"检查 {v.size} 项，最大回升 = {fmt_value(float(np.max(d)) if d.size else 0.0)}"
    if not ok:
        fail(label=name, theory="单调不增", measured=float(np.max(d)), tol="≤ 0", detail=info, kind="B")
    record(label=name, theory="单调不增", measured=float(np.max(d)), tol="≤ 0", passed=True, detail=info, kind="B")
