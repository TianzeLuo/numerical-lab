"""断言失败的统一异常：**必须同时说出理论值与实测值**。

设计动机（docs/CHAPTERS.md §2）："失败信息指出理论值与实测值"。
一条只说 ``assert 0.87 <= 0.1`` 的失败信息，等于没有信息。
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["AssertionFailure", "fmt_value"]


def fmt_value(v: Any, *, sig: int = 6) -> str:
    """把数值格式化成"人眼能比较"的短形式（科学计数法 + 有效数字）。"""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "+inf" if v > 0 else "-inf"
        if v == 0:
            return "0"
        a = abs(v)
        if 1e-3 <= a < 1e5:
            return f"{v:.{sig}g}"
        return f"{v:.{sig - 1}e}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt_value(x, sig=sig) for x in v) + "]"
    return str(v)


class AssertionFailure(AssertionError):
    """数值断言的失败载体。

    属性
    ----
    label : str
        断言名（人类可读）。
    theory : Any
        理论值 / 理论阶 / 理论界。
    measured : Any
        实测值。
    tol : Any
        容差（若为 ``(rtol, atol)`` 元组则原样保留）。
    detail : str
        补充上下文（如拟合 R²、参与的样本点、二分区间）。
    """

    def __init__(
        self,
        *,
        label: str,
        theory: Any = None,
        measured: Any = None,
        tol: Any = None,
        detail: str = "",
        kind: str | None = None,
    ) -> None:
        self.label = label
        self.theory = theory
        self.measured = measured
        self.tol = tol
        self.detail = detail
        self.kind = kind
        super().__init__(self._render())

    # ------------------------------------------------------------------ #
    def deviation(self) -> float | None:
        """实测与理论的绝对差；不可比较时返回 ``None``。"""
        try:
            if isinstance(self.theory, (int, float)) and isinstance(self.measured, (int, float)):
                return abs(float(self.measured) - float(self.theory))
        except (TypeError, ValueError):  # pragma: no cover - 防御性
            return None
        return None

    def _render(self) -> str:
        head = f"[{self.kind or '?'} 类断言失败] {self.label}"
        lines = [
            head,
            f"  理论值 : {fmt_value(self.theory)}",
            f"  实测值 : {fmt_value(self.measured)}",
        ]
        dev = self.deviation()
        if dev is not None:
            lines.append(f"  偏差   : {fmt_value(dev)}")
        if self.tol is not None:
            lines.append(f"  容差   : {fmt_value(self.tol)}")
        if self.detail:
            lines.append(f"  详情   : {self.detail}")
        return "\n".join(lines)
