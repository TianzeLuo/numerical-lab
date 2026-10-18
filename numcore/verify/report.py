"""断言登记与报告：让"理论值 vs 实测值"成为**可审计的产物**。

工作方式
--------
* 章节测试用 ``@assertion(kind="A", theory="O(h²)")`` 装饰；
* 装饰器把 ``kind/theory/chapter/nodeid`` 放进上下文（``contextvars``，线程/协程安全）；
* :mod:`numcore.verify` 的各个 ``assert_*`` helper 在判定时调用 :func:`record`，
  把``理论值 / 实测值 / 容差 / 判定``写入全局登记表；
* 测试函数还可以直接 ``return Measured(value=..., tol=...)`` 来登记"手工断言"的结果；
* pytest 会话结束（``numcore/verify/plugin.py``）把登记表落成
  ``dist/reports/assertions.json``，并打印每章摘要。

**静默通过是不允许的**：若一个被装饰的测试既没触发任何 helper 记录、也没返回 ``Measured``，
装饰器会直接判失败（``kind`` 未知的占位断言会被 G3 门禁拦下）。
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

from .errors import AssertionFailure, fmt_value

__all__ = [
    "AssertionRecord",
    "Measured",
    "assertion",
    "record",
    "fail",
    "reset_report",
    "reset_events",
    "clear_reset_events",
    "isolated_report",
    "records",
    "summary",
    "write_report",
    "markdown_table",
    "chapter_from_nodeid",
    "current_context",
    "suppressed",
]

_CHAPTER_RE = re.compile(r"(?:^|[/\\])tests[/\\](\d+\.\d+)[/\\]")
_T = TypeVar("_T")


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class AssertionRecord:
    """一条断言的完整证据链。"""

    id: str
    kind: str  # "A" | "B" | "C"
    label: str
    theory: Any = None
    measured: Any = None
    tol: Any = None
    passed: bool = True
    detail: str = ""
    chapter: str | None = None
    nodeid: str | None = None
    seconds: float | None = None
    unmeasured: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def row(self) -> str:
        """Markdown 表格行。"""
        mark = "✅" if self.passed else "❌"
        tol = "—" if self.tol is None else fmt_value(self.tol)
        return (
            f"| `{self.id}` | {self.kind} | {self.label} | {fmt_value(self.theory)} | "
            f"{fmt_value(self.measured)} | {tol} | {mark} |"
        )


@dataclass
class Measured:
    """测试函数显式返回的实测量（用于手工断言）。"""

    value: Any
    tol: Any = None
    theory: Any = None
    label: str | None = None
    detail: str = ""


@dataclass
class _Context:
    kind: str
    theory: Any
    id: str
    chapter: str | None
    tol: Any
    nodeid: str | None
    n_recorded: int = 0
    #: 本测试登记过的类别（A/B/C），用于查"声明 A 却只登记了 B"这类**类别不符**。
    kinds: set[str] = dataclasses.field(default_factory=set)


_RECORDS: list[AssertionRecord] = []
_CTX: contextvars.ContextVar[_Context | None] = contextvars.ContextVar("numcore_verify_ctx", default=None)
_NODEID: contextvars.ContextVar[str | None] = contextvars.ContextVar("numcore_verify_nodeid", default=None)
_SUPPRESS: contextvars.ContextVar[bool] = contextvars.ContextVar("numcore_verify_suppress", default=False)


@contextlib.contextmanager
def suppressed() -> Iterator[None]:
    """临时关闭登记（用于"故意制造失败"的框架自测，避免污染报告）。"""
    token = _SUPPRESS.set(True)
    try:
        yield
    finally:
        _SUPPRESS.reset(token)


# --------------------------------------------------------------------------- #
# 登记
# --------------------------------------------------------------------------- #
def chapter_from_nodeid(nodeid: str | None) -> str | None:
    """从 pytest nodeid 推断章节号（``tests/1.2/test_lu.py::test_x`` → ``"1.2"``）。"""
    if not nodeid:
        return None
    m = _CHAPTER_RE.search(nodeid)
    return m.group(1) if m else None


def record(
    *,
    label: str,
    theory: Any = None,
    measured: Any = None,
    tol: Any = None,
    passed: bool = True,
    detail: str = "",
    kind: str | None = None,
    extra: dict[str, Any] | None = None,
) -> AssertionRecord:
    """登记一条断言证据。

    上下文存在时（即测试被 ``@assertion`` 装饰）自动补全 ``kind/theory/chapter/nodeid``。
    """
    ctx = _CTX.get()
    if _SUPPRESS.get():
        return AssertionRecord(id=label, kind=kind or "?", label=label)  # 不登记，仅返回占位
    rec = AssertionRecord(
        id=(ctx.id + "::" + label) if ctx and label != ctx.id else (label if not ctx else ctx.id),
        kind=kind or (ctx.kind if ctx else "?"),
        label=label,
        theory=theory if theory is not None else (ctx.theory if ctx else None),
        measured=measured,
        tol=tol if tol is not None else (ctx.tol if ctx else None),
        passed=passed,
        detail=detail,
        chapter=(ctx.chapter if ctx else None) or chapter_from_nodeid(_NODEID.get()),
        nodeid=_NODEID.get(),
        extra=extra or {},
    )
    _RECORDS.append(rec)
    if ctx is not None:
        ctx.n_recorded += 1
        ctx.kinds.add(str(rec.kind))
    return rec


def current_context() -> _Context | None:
    """返回当前断言上下文（供 helper 读取 ``kind`` 等）。"""
    return _CTX.get()


def fail(
    *,
    label: str,
    theory: Any = None,
    measured: Any = None,
    tol: Any = None,
    detail: str = "",
    kind: str | None = None,
) -> None:
    """登记一条**失败**断言并抛出 :class:`AssertionFailure`（helper 内部专用）。

    先登记、再抛出，并给异常打上 ``recorded`` 标记，避免装饰器重复登记。
    """
    record(
        label=label,
        theory=theory,
        measured=measured,
        tol=tol,
        passed=False,
        detail=detail,
        kind=kind,
    )
    exc = AssertionFailure(label=label, theory=theory, measured=measured, tol=tol, detail=detail, kind=kind)
    exc.recorded = True  # type: ignore[attr-defined]
    raise exc


def _record_failure(exc: AssertionFailure, seconds: float | None = None) -> None:
    ctx = _CTX.get()
    if _SUPPRESS.get():
        exc.recorded = True  # type: ignore[attr-defined]
        return
    _RECORDS.append(
        AssertionRecord(
            id=ctx.id if ctx else exc.label,
            kind=exc.kind or (ctx.kind if ctx else "?"),
            label=exc.label,
            theory=exc.theory if exc.theory is not None else (ctx.theory if ctx else None),
            measured=exc.measured,
            tol=exc.tol if exc.tol is not None else (ctx.tol if ctx else None),
            passed=False,
            detail=exc.detail,
            chapter=(ctx.chapter if ctx else None) or chapter_from_nodeid(_NODEID.get()),
            nodeid=_NODEID.get(),
            seconds=seconds,
            unmeasured=exc.measured is None,
        )
    )
    exc.recorded = True  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 装饰器
# --------------------------------------------------------------------------- #
def assertion(
    kind: str,
    theory: Any,
    *,
    id: str | None = None,  # noqa: A002 - 与项目术语一致
    chapter: str | None = None,
    tol: Any = None,
    strict_measure: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """把一个测试函数登记为 A/B/C 类断言。

    参数
    ----
    kind
        ``"A"`` 解析对拍 / ``"B"`` 不变量 / ``"C"`` 收敛率与阈值。
    theory
        理论值、理论阶或理论界（会写进报告与失败信息）。
    id
        断言 id，缺省用函数名；``chapters/<id>/asserts.yaml`` 引用它。
    strict_measure
        为真时，函数必须在执行过程中产生至少一条登记（helper 记录或返回 ``Measured``），
        否则判为"静默通过"并失败。
    """

    def deco(fn: Callable[..., _T]) -> Callable[..., _T]:
        aid = id or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            ctx = _Context(
                kind=kind,
                theory=theory,
                id=aid,
                chapter=chapter or chapter_from_nodeid(_NODEID.get()),
                tol=tol,
                nodeid=_NODEID.get(),
            )
            token = _CTX.set(ctx)
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
            except AssertionFailure as exc:
                if not getattr(exc, "recorded", False):
                    _record_failure(exc, seconds=time.perf_counter() - t0)
                raise
            except Exception as exc:  # 非断言异常也要留痕，否则报告中"凭空消失"
                if not _SUPPRESS.get():
                    _RECORDS.append(
                        AssertionRecord(
                            id=aid,
                            kind=kind,
                            label=aid,
                            theory=theory,
                            measured=None,
                            tol=tol,
                            passed=False,
                            detail=f"{type(exc).__name__}: {exc}",
                            chapter=ctx.chapter,
                            nodeid=ctx.nodeid,
                            seconds=time.perf_counter() - t0,
                            unmeasured=True,
                        )
                    )
                raise
            else:
                _ingest_return(out, ctx)
                if (
                    strict_measure
                    and ctx.n_recorded > 0
                    and kind in ("A", "B", "C")
                    and kind not in ctx.kinds
                    and not _SUPPRESS.get()
                ):
                    # 类别不符也是"假证据"：@assertion("A") 要求与解析/参考实现对拍，
                    # 但 helper（assert_leq / assert_within / assert_monotone）登记的 kind 固定是 "B"。
                    # 只报"零测量"是不够的——那种测试 pytest 全绿、报告里也有记录，
                    # 而"这一章有 N 条 A 类断言"这句话就变成了假的（1.3 章真踩到，见其交付报告第 4 条）。
                    exc = AssertionFailure(
                        label=aid,
                        theory=theory,
                        measured=None,
                        tol=tol,
                        detail=(
                            f"声明为 {kind} 类，但本测试登记的全是 "
                            f"{'/'.join(sorted(ctx.kinds))} 类记录：A 类必须与解析解/参考实现对拍、"
                            "B 类是不变量、C 类是收敛率。要用 helper 记 A 类，请显式调用 "
                            'record(kind="A", ...) 并 assert 它；helper 的缺省类别是 B。'
                        ),
                        kind=kind,
                    )
                    _record_failure(exc, seconds=time.perf_counter() - t0)
                    raise exc
                if strict_measure and ctx.n_recorded == 0 and not _SUPPRESS.get():
                    exc = AssertionFailure(
                        label=aid,
                        theory=theory,
                        measured=None,
                        tol=tol,
                        detail="测试未登记任何实测值：请调用 numcore.verify 的 assert_* helper，"
                        "或 return Measured(value=..., tol=...)。禁止静默通过（POLICY §3）。",
                        kind=kind,
                    )
                    _record_failure(exc, seconds=time.perf_counter() - t0)
                    raise exc
                return out
            finally:
                _CTX.reset(token)

        wrapper.assertion_meta = {"id": aid, "kind": kind, "theory": theory, "chapter": chapter}  # type: ignore[attr-defined]
        return wrapper

    return deco


def _ingest_return(out: Any, ctx: _Context) -> None:
    """把测试函数返回的 ``Measured``（或其序列）登记进报告。"""
    if out is None:
        return
    items: Sequence[Any]
    if isinstance(out, Measured):
        items = [out]
    elif isinstance(out, (list, tuple)):
        items = list(out)
    else:
        return
    for it in items:
        if not isinstance(it, Measured):
            continue
        record(
            label=it.label or ctx.id,
            theory=it.theory if it.theory is not None else ctx.theory,
            measured=it.value,
            tol=it.tol if it.tol is not None else ctx.tol,
            passed=True,
            detail=it.detail,
        )


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def records() -> list[AssertionRecord]:
    """当前会话登记的全部断言（拷贝）。"""
    return list(_RECORDS)


@contextlib.contextmanager
def isolated_report() -> Iterator[list[AssertionRecord]]:
    """临时隔离登记表：进入时清空，退出时把原来的记录**原样放回**。

    为什么必须有它：:func:`reset_report` 清的是**整个 pytest 会话**的登记表，
    而 ``pytest_sessionfinish`` 是在会话结束时才写报告的。框架自己的单元测试如果直接调
    ``reset_report()``，整套 pytest 跑完之后登记表是空的，插件就**静默不写**
    ``dist/reports/assertions.json``——G3 于是读到上一次部分运行的旧报告并报
    "已实现章节在报告里没有任何断言记录"。这个坑真的踩过，所以把它做成 API 而不是纪律。
    """
    saved = list(_RECORDS)
    _RECORDS.clear()
    try:
        yield _RECORDS
    finally:
        _RECORDS[:] = saved


#: 本会话里 ``reset_report()`` 被调用的次数。插件靠它区分"这次会话没跑断言测试"与
#: "有人把整张表清空了"——后者会让报告静默不写（真踩过，见 :func:`isolated_report`）。
_RESET_EVENTS = 0


def reset_events() -> int:
    """本会话里 :func:`reset_report` 被调用过几次（插件与测试用来判"报告为什么没写"）。"""
    return _RESET_EVENTS


def clear_reset_events() -> None:
    """把计数器归零（``pytest_configure`` 在每个会话开始时调用）。"""
    global _RESET_EVENTS
    _RESET_EVENTS = 0


def reset_report() -> None:
    """清空登记表（测试用）。

    ⚠ 它清的是**整个 pytest 会话**的登记表。测试里要"一张干净的表"，请用
    :func:`isolated_report`（退出时会把原记录放回去）。
    """
    global _RESET_EVENTS
    _RESET_EVENTS += 1
    _RECORDS.clear()


def summary() -> dict[str, Any]:
    """按章节 / 类型汇总。"""
    per_chapter: dict[str, dict[str, Any]] = {}
    totals = {"A": 0, "B": 0, "C": 0, "passed": 0, "failed": 0, "unmeasured": 0, "total": 0}
    for r in _RECORDS:
        ch = r.chapter or "—"
        d = per_chapter.setdefault(
            ch, {"chapter": ch, "A": 0, "B": 0, "C": 0, "passed": 0, "failed": 0, "items": []}
        )
        if r.kind in ("A", "B", "C"):
            d[r.kind] += 1
            totals[r.kind] += 1
        d["passed" if r.passed else "failed"] += 1
        totals["passed" if r.passed else "failed"] += 1
        totals["unmeasured"] += int(r.unmeasured)
        totals["total"] += 1
        d["items"].append(r.to_dict())
    return {"totals": totals, "chapters": per_chapter}


def markdown_table(chapter: str | None = None, *, only_failed: bool = False) -> str:
    """生成断言报告表（理论值 / 实测值 / 容差 / 判定）。"""
    rows = [r for r in _RECORDS if (chapter is None or r.chapter == chapter)]
    if only_failed:
        rows = [r for r in rows if not r.passed]
    head = "| 断言 id | 类 | 内容 | 理论值 | 实测值 | 容差 | 判定 |\n|---|---|---|---|---|---|---|"
    body = "\n".join(r.row() for r in rows) or "| — | — | 无 | — | — | — | — |"
    return head + "\n" + body


def write_report(path: str | Path, *, extra: dict[str, Any] | None = None) -> Path:
    """把登记表写成 ``dist/reports/assertions.json``（协议见 PROTOCOLS §8）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = summary()
    payload["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if extra:
        payload.update(extra)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def iter_chapters() -> Iterable[str]:
    """已登记断言的章节号（排序）。"""
    return sorted({r.chapter for r in _RECORDS if r.chapter})
