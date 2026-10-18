"""pytest 插件：会话结束时落盘断言报告，并在终端打印"每章 A/B/C 覆盖"摘要。

用法（``tests/conftest.py``）::

    from numcore.verify import plugin as _numcore_plugin

    def pytest_configure(config):
        _numcore_plugin.pytest_configure(config)

    def pytest_runtest_setup(item):
        _numcore_plugin.pytest_runtest_setup(item)

    def pytest_sessionfinish(session, exitstatus):
        _numcore_plugin.pytest_sessionfinish(session, exitstatus)

报告路径由环境变量 ``NUMCORE_REPORT`` 指定，缺省 ``dist/reports/assertions.json``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import report as _report

__all__ = ["pytest_configure", "pytest_runtest_setup", "pytest_sessionfinish", "report_path"]


def report_path() -> Path:
    """断言报告的落盘路径。"""
    return Path(os.environ.get("NUMCORE_REPORT", "dist/reports/assertions.json"))


def pytest_configure(config: Any) -> None:
    """清空上一次会话的登记表，避免跨会话污染。"""
    _report.clear_reset_events()
    _report.reset_report()
    _report.clear_reset_events()  # 上面这次是"开新会话"，不算事故


def pytest_runtest_setup(item: Any) -> None:
    """把当前 nodeid 放进上下文，使登记的断言自带 nodeid / 章节归属。"""
    _report._NODEID.set(getattr(item, "nodeid", None))  # noqa: SLF001 - 同包内部状态


def pytest_sessionfinish(session: Any, exitstatus: int = 0) -> None:  # noqa: ARG001
    """写报告 + 终端摘要。"""
    recs = _report.records()
    if not recs:
        # 空登记表 = 报告不写。这在两种情况下是正常的（本会话没有断言测试），
        # 但也可能是**事故**：某个测试直接调了 `reset_report()`，把整个会话的记录清掉了，
        # 于是 G3 拿着上一次部分运行的旧报告报"已实现章节没有任何断言记录"。
        # 真踩过。所以这里不静默：收集到测试却一条记录都没有时，明确喊一声。
        collected = int(getattr(session, "testscollected", 0) or 0)
        if collected > 0 and _report.reset_events() > 0:
            print(
                f"\n[numcore] ⚠ 本次会话调用过 reset_report() {_report.reset_events()} 次，"
                f"结束时登记表是空的 → 报告未更新（{report_path()}）。"
                "测试里要一张干净的表请用 numcore.verify.isolated_report()，"
                "它退出时会把原记录放回去。",
                flush=True,
            )
        return
    path = report_path()
    _report.write_report(path)
    summary = _report.summary()
    t = summary["totals"]
    tr = getattr(session.config, "pluginmanager", None)
    writer = None
    if tr is not None:
        trr = session.config.get_terminal_writer() if hasattr(session.config, "get_terminal_writer") else None
        writer = trr
    lines = [
        "",
        "── numcore 断言报告 ─────────────────────────────────────────",
        f"  共 {t['total']} 条：A(解析对拍) {t['A']}  B(不变量) {t['B']}  C(收敛率) {t['C']}"
        f"   通过 {t['passed']}  失败 {t['failed']}  未登记实测值 {t['unmeasured']}",
    ]
    for ch in sorted(summary["chapters"]):
        d = summary["chapters"][ch]
        if ch == "—":
            continue
        lines.append(f"  · 第 {ch} 章：A{d['A']} B{d['B']} C{d['C']}，通过 {d['passed']}，失败 {d['failed']}")
    lines.append(f"  报告：{path}")
    lines.append("────────────────────────────────────────────────────────────")
    text = "\n".join(lines)
    if writer is not None:
        writer.line(text)
    else:  # pragma: no cover - 无终端写出器时回退
        print(text)
