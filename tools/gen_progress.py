#!/usr/bin/env python3
"""G10 门：把仓库状态写成 ``docs/PROGRESS.md``，并检查它是否过期。

文件分两段，中间有一行 HTML 注释当分界线：

* 分界线**以上**是**确定性内容**（章节进度表、总数、每章三件套与断言的到位情况）——
  ``--check`` 只比这一段，任何漂移都算失败；
* 分界线**以下**是**易变内容**（各门禁最近一次运行的结论、时间戳、commit），
  从 ``dist/reports/*.json`` 里读，不参与漂移判定（否则每跑一次 CI 都会"漂移"）。

用法::

    python3 tools/gen_progress.py            # 写入 docs/PROGRESS.md
    python3 tools/gen_progress.py --check    # 只检查确定性段落是否漂移
    python3 tools/gen_progress.py --json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Iterable

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.registry import Registry  # noqa: E402

DIVIDER = "<!-- gates: 以下是易变内容（门禁运行记录），不参与 G10 漂移判定 -->"
PROGRESS = ROOT / "docs" / "PROGRESS.md"

GATES: list[tuple[str, str, str]] = [
    ("G1", "策略红线", "dist/reports/policy.json"),
    ("G3", "数值断言", "dist/reports/assertions.json"),
    ("G6", "生成物无漂移（manifest + 色标）", "dist/reports/manifests.json"),
    ("G7", "Python↔TS 对拍", "dist/reports/parity.json"),
    ("G8", "视觉回归", "dist/reports/visual_regression.json"),
    ("G9", "LaTeX 编译", "dist/reports/book.json"),
    ("CI", "整体", "dist/reports/ci.json"),
]


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ok": False, "error": "报告不是合法 JSON"}
    return None


def _bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "—"
    filled = round(width * done / total)
    return f"`{'█' * filled}{'·' * (width - filled)}` {done}/{total}"


def _git_commit(root: pathlib.Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
        return out.stdout.strip() or "（未知）"
    except OSError:  # pragma: no cover
        return "（无 git）"


def _declared_kinds(root: pathlib.Path, chapter_id: str) -> tuple[int, int, int] | None:
    """从 ``chapters/<id>/asserts.yaml`` 数**声明**的 A/B/C 条数。

    为什么要用声明的条数、而不是 `dist/reports/assertions.json` 里的实测条数：
    报告被**任何一次 pytest**（哪怕只跑单章）重写，于是"确定性段落"会随之漂移，
    G10 于是假红（实测：同一条命令连跑两次，第一次 exit 0、第二次 exit 1）。
    声明的条数在版本控制里，只有作者改章节时才会变——这才是"确定性"该有的含义。
    实测条数照旧展示，但放在分界线**以下**的易变段落里。
    """
    path = root / "chapters" / chapter_id / "asserts.yaml"
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return None
    tests = data.get("tests") or []
    out = {"A": 0, "B": 0, "C": 0}
    for item in tests:
        kind = str((item or {}).get("kind", "")).upper()
        if kind in out:
            out[kind] += 1
    return out["A"], out["B"], out["C"]


def deterministic_section(root: pathlib.Path, summary: dict[str, Any], report: dict[str, Any] | None) -> str:
    declared = [c for c in summary["chapters"] if c["declared"]]
    extra = [c for c in summary["chapters"] if not c["declared"]]
    lines: list[str] = []
    lines.append("<!-- 本文件的「确定性段落」由 tools/gen_progress.py 生成，请勿手改（G10 会检查漂移）。 -->")
    lines.append("")
    lines.append("# 项目进度")
    lines.append("")
    lines.append(
        f"- 章节：**{summary['chapters_complete']} / {summary['chapters_declared']}** 完成"
        f"（另有 {summary['chapters_extra']} 个额外章节：模板章 0.0 不计入 28 章）"
    )
    lines.append(
        f"- 交互可视化：**{summary['vizzes_implemented']} / {summary['vizzes_declared']}**"
        f"（额外章节另有 {summary['vizzes_extra_implemented']} / {summary['vizzes_extra']}）"
    )
    lines.append(f"- 进度条（可视化）：{_bar(summary['vizzes_implemented'], summary['vizzes_declared'])}")
    lines.append(f"- 进度条（章节）：{_bar(summary['chapters_complete'], summary['chapters_declared'])}")
    lines.append("")
    lines.append("「五件套」= ① `chapters/<id>/notes.tex` 推导笔记 ② `numcore/` 核心实现（`docs/chapters.yaml` 的 `modules`）"
                 " ③ `viz/chapters/<id>/` 交互可视化（`*.viz.ts` + `manifest.gen.ts`）"
                 " ④ `tests/<id>/` + `chapters/<id>/asserts.yaml` 数值断言"
                 " ⑤ `chapters/<id>/pitfalls.md` 防误导分析（≥ 800 字）。")
    lines.append("")
    lines.append("## 章节状态")
    lines.append("")
    lines.append("| 章 | 标题 | 部 | 可视化 | 模块 | manifest | 测试 | 笔记 | 声明断言 A/B/C | 完成 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in declared + extra:
        cid = c["id"]
        kinds_declared = _declared_kinds(root, cid)
        kinds = "/".join(str(v) for v in kinds_declared) if kinds_declared else "—"
        viz = f"{c['vizImplemented']}/{c['vizDeclared']}"
        mark = lambda b: "✅" if b else "—"  # noqa: E731
        title = c["title"].replace("|", "\\|")
        lines.append(
            f"| {cid}{'' if c['declared'] else ' *'} | {title} | {c['part'] or '—'} | {viz} | "
            f"{mark(c['hasModule'])} | {mark(c['manifestOk'])} | {mark(c['hasTests'])} | {mark(c['hasNotes'])} | "
            f"{kinds} | {mark(c['complete'])} |"
        )
    lines.append("")
    lines.append("`*` = 未在 `docs/chapters.yaml` 里声明的额外章节（当前只有 0.0 模板章）。")
    lines.append("")
    return "\n".join(lines) + "\n"


def _assertions_section(root: pathlib.Path, report: dict[str, Any] | None) -> list[str]:
    """易变段落：最近一次 pytest 的断言实测总览（报告随时会被重写，故不入确定性段落）。"""
    if not report:
        return []
    t = report.get("totals", {})
    lines = ["## 断言实测（上次 pytest 运行）", ""]
    lines.append(f"- 实测条数：{t.get('total', 0)}（A {t.get('A', 0)} / B {t.get('B', 0)} / C {t.get('C', 0)}）")
    lines.append(f"- 通过 {t.get('passed', 0)}，失败 {t.get('failed', 0)}，未登记实测值 {t.get('unmeasured', 0)}")
    lines.append("- 每章实测条数：" + "、".join(
        f"{cid} {len(blk.get('items') or [])}"
        for cid, blk in sorted((report.get("chapters") or {}).items())
    ))
    lines.append("")
    return lines


def volatile_section(root: pathlib.Path, summary: dict[str, Any], report: dict[str, Any] | None = None) -> str:
    lines = [DIVIDER, "", "## 门禁状态", ""]
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')} · commit `{_git_commit(root)}`")
    lines.append("")
    lines.append("| 门禁 | 内容 | 结论 | 证据 |")
    lines.append("|---|---|---|---|")
    for gid, what, rel in GATES:
        data = _read_json(root / rel)
        if data is None:
            verdict = "未运行"
        else:
            # 证据文件的 "ok" 可能嵌在 gate 里（run_asserts 把门禁结论写在 gate 下）
            ok = data.get("ok")
            if ok is None:
                ok = data.get("gate", {}).get("ok")
            verdict = "通过" if ok else ("失败" if ok is False else "未知")
            if gid == "CI" and data.get("failed"):
                verdict = f"失败（{data['failed']} 步）"
        lines.append(f"| {gid} | {what} | {verdict} | `{rel}` |")
    lines.append("")
    lines.append("门禁定义见 [`POLICY.md`](POLICY.md)；一键跑全部：`make ci`（或 `bash tools/ci.sh`）。")
    lines.append("")
    # 断言实测条数来自 `dist/reports/assertions.json`，而**任何一次 pytest**（哪怕只跑单章）
    # 都会重写它——所以它只能放在分界线以下，否则 G10 会假红（同一条命令连跑两次、第二次失败）。
    lines.extend(_assertions_section(root, report))
    return "\n".join(lines)


def build(root: pathlib.Path) -> tuple[str, str]:
    """返回 (确定性段落, 全文)。"""
    reg = Registry(root)
    reg.discover(strict=False)
    summary = reg.summary()
    report = _read_json(root / "dist" / "reports" / "assertions.json")
    head = deterministic_section(root, summary, report)
    return head, head + "\n" + volatile_section(root, summary, report)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成/检查 docs/PROGRESS.md（G10 门）")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=PROGRESS)
    args = ap.parse_args(list(argv) if argv is not None else None)

    head, full = build(ROOT)
    out: pathlib.Path = args.out
    if args.check:
        if not out.exists():
            print(f"[G10] {out.relative_to(ROOT)} 不存在；请运行 python3 tools/gen_progress.py")
            return 1
        old = out.read_text(encoding="utf-8")
        old_head = old.split(DIVIDER)[0]
        if old_head.strip() != head.strip():
            print("[G10] docs/PROGRESS.md 的确定性段落与仓库状态漂移；请运行 python3 tools/gen_progress.py")
            old_lines = old_head.strip().splitlines()
            new_lines = head.strip().splitlines()
            import difflib

            for line in list(difflib.unified_diff(old_lines, new_lines, "PROGRESS.md(旧)", "PROGRESS.md(新)", lineterm=""))[:40]:
                print(line)
            return 1
        if args.json:
            print(json.dumps({"ok": True, "path": str(out.relative_to(ROOT))}, ensure_ascii=False))
        else:
            print("[G10] docs/PROGRESS.md 与仓库状态一致")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full, encoding="utf-8", newline="\n")
    if args.json:
        print(json.dumps({"ok": True, "path": str(out.relative_to(ROOT)), "bytes": len(full)}, ensure_ascii=False))
    else:
        print(f"已写入 {out.relative_to(ROOT)}（确定性段落 {len(head)} 字节）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
