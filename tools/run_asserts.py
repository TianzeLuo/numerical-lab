#!/usr/bin/env python3
"""G3 门：跑断言、对账"声明 vs 实测"、产出报告。

做三件事：

1. 跑 ``pytest``（可 ``--no-run`` 复用上一次的报告），报告落在 ``dist/reports/assertions.json``；
2. **对账**：每章的 ``chapters/<id>/asserts.yaml`` 声明了哪些测试函数、属于哪一类，
   报告里就必须有一条对应的记录，且类别一致、数量不少于 ``min_counts``；
3. 生成 ``chapters/<id>/asserts_report.md``（理论值 / 实测值 / 容差 / 判定）——
   它会被 ``notes.tex`` 通过 ``\\input`` 引进去，**纸面结论与机器结论同源**。

判定规则（任何一条不满足就 exit 1）：

* 有任何 ``failed`` 或 ``unmeasured`` 记录；
* 已实现的章节（有模块且 manifest 合法）缺少 A/B/C 中任何一类；
* 声明过的测试函数在报告里查无此人，或类别不符；
* 声明的 ``min_counts`` 未被满足。

用法::

    python3 tools/run_asserts.py                 # 跑全部测试并出报告
    python3 tools/run_asserts.py --no-run        # 只分析现有报告
    python3 tools/run_asserts.py -k 0.0          # 只跑匹配的用例
    python3 tools/run_asserts.py --json          # 机器可读结果
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from collections import defaultdict
from typing import Any, Iterable

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.registry import Registry  # noqa: E402

DEFAULT_REPORT = ROOT / "dist" / "reports" / "assertions.json"
KINDS = ("A", "B", "C")


def _display_path(path: pathlib.Path, root: pathlib.Path) -> str:
    """尽量给相对路径，给不出就原样返回。

    存在的理由：``--report`` 可以被指到仓库外（例如 ``/tmp/r.json``，或 CI 的临时目录），
    那时 ``path.relative_to(root)`` 会抛 ``ValueError: '/tmp/...' is not in the subpath of ...``。
    章报告其实**已经正确写出**了，只是最后打印一行路径时把整个进程崩掉——工具不该因为
    "打印用的路径算不出来"而失败。所有打印/写盘用的路径都走这里。
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# asserts.yaml
# --------------------------------------------------------------------------- #
def load_asserts_decl(root: pathlib.Path, chapter: str) -> dict[str, Any] | None:
    """读 ``chapters/<id>/asserts.yaml``；不存在返回 None（不报错，但要记账）。"""
    path = root / "chapters" / chapter / "asserts.yaml"
    if not path.exists():
        return None
    import yaml  # type: ignore

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["_path"] = _display_path(path, root)
    return data


def declared_tests(decl: dict[str, Any]) -> dict[str, str]:
    """``测试函数名 → 类别``。"""
    out: dict[str, str] = {}
    for item in decl.get("tests", []) or []:
        name = str(item.get("name", "")).strip()
        kind = str(item.get("kind", "")).strip().upper()
        if name:
            out[name] = kind
    return out


# --------------------------------------------------------------------------- #
# 分析
# --------------------------------------------------------------------------- #
def analyse(
    root: pathlib.Path,
    report: dict[str, Any],
    *,
    require_declaration_for: set[str] | None = None,
) -> dict[str, Any]:
    """把报告与声明对账，返回 ``{"ok": bool, "problems": [...], "chapters": {...}}``。"""
    problems: list[str] = []
    totals = report.get("totals", {})
    if totals.get("failed"):
        problems.append(f"有 {totals['failed']} 条断言失败")
    if totals.get("unmeasured"):
        problems.append(f"有 {totals['unmeasured']} 条断言没有登记实测值（静默通过）")

    reg = Registry(root)
    try:
        summary = reg.summary()
    except Exception as exc:  # noqa: BLE001 - 章节坏了也要能出报告
        problems.append(f"章节注册表不可用：{type(exc).__name__}: {exc}")
        summary = {"chapters": []}
    implemented = {
        c["id"]
        for c in summary.get("chapters", [])
        if c.get("hasModule") and c.get("manifestOk")
    }

    chapters_out: dict[str, Any] = {}
    for cid, blk in sorted(report.get("chapters", {}).items()):
        recs = blk.get("items", [])
        per_fn: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in recs:
            fn = str(rec.get("nodeid") or rec.get("id") or "").split("::")[-1]
            per_fn[fn].append(rec)
        counts = {k: 0 for k in KINDS}
        for rec in recs:
            k = str(rec.get("kind", "")).strip().upper()
            if k in counts:
                counts[k] += 1
        decl = load_asserts_decl(root, cid)
        ch_problems: list[str] = []

        if decl is not None:
            want = declared_tests(decl)
            if not want:
                ch_problems.append(f"{cid}: asserts.yaml 里没有 tests 条目")
            for name, kind in want.items():
                got = per_fn.get(name)
                if not got:
                    ch_problems.append(f"{cid}: 声明了 {name}（{kind} 类）但报告里没有这条断言")
                    continue
                kinds_got = {str(r.get("kind", "")).upper() for r in got}
                if kind and kind not in kinds_got:
                    ch_problems.append(f"{cid}: {name} 声明为 {kind} 类，实测类别是 {sorted(kinds_got)}")
            extra = sorted(set(per_fn) - set(want))
            for name in extra:
                ch_problems.append(f"{cid}: {name} 有断言记录但没在 asserts.yaml 里声明")
            for kind, minimum in (decl.get("min_counts") or {}).items():
                k = str(kind).upper()
                if counts.get(k, 0) < int(minimum):
                    ch_problems.append(f"{cid}: {k} 类断言只有 {counts.get(k, 0)} 条，声明要求 ≥ {minimum}")
            req = [str(x).upper() for x in (decl.get("required_kinds") or [])]
            for k in req:
                if counts.get(k, 0) == 0:
                    ch_problems.append(f"{cid}: 缺少 {k} 类断言（asserts.yaml 要求必须有）")
        elif cid in implemented:
            ch_problems.append(f"{cid}: 已实现但没有 chapters/{cid}/asserts.yaml（无法对账）")

        # 未声明的已实现章节也必须三类齐全
        if decl is None:
            for k in KINDS:
                if counts.get(k, 0) == 0:
                    ch_problems.append(f"{cid}: 缺少 {k} 类断言")

        chapters_out[cid] = {
            "declared": len(declared_tests(decl)) if decl else 0,
            "collected": len(per_fn),
            "records": len(recs),
            "counts": counts,
            "passed": sum(1 for r in recs if r.get("passed")),
            "failed": sum(1 for r in recs if not r.get("passed")),
            "hasDeclaration": decl is not None,
            "problems": ch_problems,
            "items": recs,
        }
        problems.extend(ch_problems)

    missing = sorted(implemented - set(report.get("chapters", {})))
    for cid in missing:
        problems.append(f"{cid}: 已实现章节在报告里没有任何断言记录")

    return {
        "ok": not problems,
        "problems": problems,
        "totals": totals,
        "chapters": chapters_out,
    }


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if abs(v) < 1e-4 or abs(v) >= 1e5:
            return f"{v:.3e}"
        return f"{v:.6g}"
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def markdown_for(chapter: str, blk: dict[str, Any]) -> str:
    lines = [
        f"## 章节 {chapter} 数值断言报告",
        "",
        "> 本表由 `tools/run_asserts.py` 从实际运行结果生成，**不是手写的**。",
        "",
        f"- 断言条数：{blk['records']}（A {blk['counts']['A']} / B {blk['counts']['B']} / C {blk['counts']['C']}）",
        f"- 通过：{blk['passed']}，失败：{blk['failed']}",
        f"- 声明文件：{'`chapters/%s/asserts.yaml`' % chapter if blk['hasDeclaration'] else '**缺失**'}",
        "",
        "| 类别 | 断言 | 理论值 | 实测值 | 容差 | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for rec in blk["items"]:
        ok = "✅" if rec.get("passed") else "❌"
        label = str(rec.get("label") or rec.get("id") or "").replace("|", "\\|")
        theory = _fmt(rec.get("theory")).replace("|", "\\|")
        measured = _fmt(rec.get("measured")).replace("|", "\\|")
        tol = _fmt(rec.get("tol")).replace("|", "\\|")
        lines.append(f"| {rec.get('kind', '?')} | {label} | {theory} | {measured} | {tol} | {ok} |")
    if not blk["items"]:
        lines.append("| — | （无断言记录） | — | — | — | ❌ |")
    lines.append("")
    details = [r for r in blk["items"] if r.get("detail")]
    if details:
        lines.append("<details><summary>逐条细节（拟合窗口、数据点数、残差量级等）</summary>")
        lines.append("")
        for rec in details:
            lines.append(f"- **{rec.get('label', rec.get('id'))}**：{rec['detail']}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    if blk.get("problems"):
        lines.append("**对账问题**：")
        lines.append("")
        for p in blk["problems"]:
            lines.append(f"- {p}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_pytest(root: pathlib.Path, report: pathlib.Path, pytest_args: list[str]) -> int:
    report.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["NUMCORE_REPORT"] = str(report)
    env.setdefault("MPLCONFIGDIR", str(root / ".cache" / "mpl"))
    cmd = [sys.executable, "-m", "pytest", "-q", *pytest_args]
    print(f"$ NUMCORE_REPORT={report} {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=root, env=env, check=False)
    return proc.returncode


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="跑断言、对账声明、生成报告（G3 门）")
    ap.add_argument("--no-run", action="store_true", help="不跑 pytest，直接分析现有报告")
    ap.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--allow-empty", action="store_true", help="报告缺失时不算失败（仅供工具自测）")
    ap.add_argument("pytest_args", nargs="*", help="透传给 pytest 的参数")
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = ROOT
    report = args.report if args.report.is_absolute() else root / args.report
    rc = 0
    if not args.no_run:
        rc = run_pytest(root, report, list(args.pytest_args))
    if not report.exists():
        msg = f"断言报告不存在：{report}"
        print(("[警告] " if args.allow_empty else "[错误] ") + msg)
        return 0 if args.allow_empty else 1
    data = json.loads(report.read_text(encoding="utf-8"))
    result = analyse(root, data)

    # 回写增强后的报告 + 每章 markdown
    enriched = dict(data)
    enriched["gate"] = {"ok": result["ok"], "problems": result["problems"], "chapters": {
        cid: {k: v for k, v in blk.items() if k != "items"} for cid, blk in result["chapters"].items()
    }}
    report.write_text(json.dumps(enriched, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for cid, blk in sorted(result["chapters"].items()):
        out = root / "chapters" / cid / "asserts_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown_for(cid, blk), encoding="utf-8", newline="\n")

    t = result["totals"]
    if args.json:
        print(json.dumps({"ok": result["ok"], "problems": result["problems"], "totals": t}, ensure_ascii=False))
    else:
        print(
            f"断言：{t.get('total', 0)} 条"
            f"（A {t.get('A', 0)} / B {t.get('B', 0)} / C {t.get('C', 0)}），"
            f"通过 {t.get('passed', 0)}，失败 {t.get('failed', 0)}，未登记实测值 {t.get('unmeasured', 0)}"
        )
        for cid, blk in sorted(result["chapters"].items()):
            print(
                f"  {cid}: {blk['records']} 条"
                f"（声明 {blk['declared']} / 测试函数 {blk['collected']}）"
                f" A{blk['counts']['A']} B{blk['counts']['B']} C{blk['counts']['C']}"
                f"{'  ⚠ ' + str(len(blk['problems'])) + ' 个对账问题' if blk['problems'] else ''}"
            )
        if result["ok"]:
            print("G3 通过：声明与实测一致，三类断言齐全")
        else:
            print(f"G3 失败：{len(result['problems'])} 个问题")
            for p in result["problems"]:
                print(f"  - {p}")
        print(f"报告：{_display_path(report, root)}；每章附表：chapters/<id>/asserts_report.md")
    if not result["ok"]:
        return 1
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
