#!/usr/bin/env python3
"""G9 门：结构检查 + 每章 xelatex 编译 + 合并本。

三件事：

1. **结构检查**（不需要 LaTeX）：有模块的章节必须有 ``chapters/<id>/notes.tex``，
   且至少含一个 ``\\begin{theorem}``（POLICY P-notes）；``\\input`` 的依赖文件必须存在。
2. **附表生成**：从 ``dist/reports/assertions.json`` 生成 ``chapters/<id>/asserts_table.tex``
   （longtable）——笔记里 ``\\input`` 它就能做到"纸面结论与机器结论同源"。
3. **编译**：逐章生成一个只包住该章的 wrapper 并编译（错误定位到章），再编译合并本
   ``dist/book/book.tex``；从 ``.log`` 里抓 undefined references / overfull 统计。

``--check`` 用于 CI：任何结构问题或编译失败都 exit 1。

用法::

    python3 tools/build_book.py --check          # CI：结构 + 逐章编译 + 合并本
    python3 tools/build_book.py --no-compile     # 只做结构检查与附表生成
    python3 tools/build_book.py --only 0.0       # 只处理一章
    python3 tools/build_book.py --json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.registry import Registry  # noqa: E402

PREAMBLE = r"""% 本文件由 tools/build_book.py 生成，请勿手改。
\documentclass[11pt,openany]{ctexbook}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!60!black}
\newtheorem{theorem}{定理}[chapter]
\newtheorem{lemma}[theorem]{引理}
\newtheorem{corollary}[theorem]{推论}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{定义}
\newtheorem{example}[theorem]{例}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{注}
\newcommand{\numlabchapter}[2]{\chapter{#1}\label{ch:#2}}
"""

# 注意：这里是 LaTeX 源码，含 % 注释，所以**不能**用 %-格式化，只能用 replace 填占位符。
ASSERTS_TABLE_HEADER = r"""% 本文件由 tools/build_book.py 从 dist/reports/assertions.json 生成，请勿手改。
\begin{longtable}{@{}p{0.06\textwidth}p{0.32\textwidth}p{0.22\textwidth}p{0.16\textwidth}@{}}
\caption{章节 __CID__ 数值断言（生成时间 __TS__）}\\
\toprule
类别 & 断言 & 理论值 & 实测值 \\
\midrule
\endfirsthead
\toprule
类别 & 断言 & 理论值 & 实测值 \\
\midrule
\endhead
\bottomrule
\endlastfoot
"""


TEX_ESCAPE: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(s: Any) -> str:
    r"""把任意文本变成"在 LaTeX 正文里安全"的文本。

    必须是**单遍**查表：第一版用连续 replace，`\\` 先被替换成 ``\\textbackslash{}``，
    随后 ``{``/``}`` 又把刚插进去的花括号转义了，于是编译出来是 ``\textbackslash\{\}``
    （印出两个字面花括号）。测试把它钉住了。
    """
    return "".join(TEX_ESCAPE.get(ch, ch) for ch in str(s))


def fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "真" if v else "假"
    if isinstance(v, float):
        if v != 0.0 and (abs(v) < 1e-4 or abs(v) >= 1e5):
            m, e = f"{v:.3e}".split("e")
            return f"${m}\\times 10^{{{int(e)}}}$"
        return f"{v:.6g}"
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        return ", ".join(fmt(x) for x in v)
    return tex_escape(v)


def write_asserts_table(
    root: pathlib.Path, cid: str, report: dict[str, Any] | None, *, dry_run: bool = False
) -> tuple[pathlib.Path, bool] | None:
    """从断言报告生成该章的 LaTeX 附表；没有报告就不生成（返回 ``None``）。

    返回 ``(路径, 是否有变化)``——``dry_run=True`` 时只算不写，供 ``--check`` 做漂移判定：
    附表是 ``notes.tex`` ``\input`` 的**正文内容**，如果它与最新断言报告不一致，
    印出来的 PDF 就在撒谎（手抄数字的老毛病换个形式又回来了）。
    """
    if not report:
        return None
    blk = (report.get("chapters") or {}).get(cid)
    if not blk:
        return None
    out = root / "chapters" / cid / "asserts_table.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [ASSERTS_TABLE_HEADER.replace("__CID__", cid).replace("__TS__", tex_escape(report.get("generated_at", "?")))]
    for rec in blk.get("items", []):
        mark = r"\checkmark" if rec.get("passed") else r"$\times$"
        lines.append(
            f"{tex_escape(rec.get('kind', '?'))} & {tex_escape(rec.get('label') or rec.get('id'))} & "
            f"{fmt(rec.get('theory'))} & {fmt(rec.get('measured'))} {mark} \\\\"
        )
    if not blk.get("items"):
        lines.append(r"\multicolumn{4}{c}{（本章暂无断言记录）} \\")
    lines.append(r"\end{longtable}")
    lines.append("")
    text = "\n".join(lines)
    old = out.read_text(encoding="utf-8") if out.exists() else None
    changed = old != text
    if changed and not dry_run:
        out.write_text(text, encoding="utf-8", newline="\n")
    return out, changed


def check_structure(root: pathlib.Path, cid: str) -> tuple[list[str], pathlib.Path | None]:
    """返回 (问题列表, notes 路径)。"""
    problems: list[str] = []
    notes = root / "chapters" / cid / "notes.tex"
    if not notes.exists():
        return [f"{cid}: 缺少 chapters/{cid}/notes.tex"], None
    text = notes.read_text(encoding="utf-8")
    if r"\begin{theorem}" not in text:
        problems.append(f"{cid}: notes.tex 里没有 \\begin{{theorem}}（关键定理需要有完整证明）")
    for m in re.finditer(r"\\input\{([^}]+)\}", text):
        rel = m.group(1)
        if not rel.endswith(".tex"):
            rel += ".tex"
        target = root / rel
        if not target.exists() and not (root / "chapters" / cid / rel).exists():
            problems.append(f"{cid}: notes.tex 里 \\input{{{m.group(1)}}} 指向的文件不存在")
    if r"\input{chapters/%s/asserts_table}" % cid not in text and r"\input{asserts_table}" not in text:
        problems.append(f"{cid}: notes.tex 没有 \\input{{chapters/{cid}/asserts_table}}（断言附表要进笔记）")
    return problems, notes


def wrapper_tex(cid: str, title: str, *, single: bool) -> str:
    """单章 wrapper。

    注意：``notes.tex`` 自己开头就写 ``\numlabchapter``（章节号 + 标题），
    所以 wrapper **不能**再补一个，否则合并本里会出现"空章 + 真章"两页。
    """
    body = f"\\input{{chapters/{cid}/notes.tex}}\n"
    if single:
        body = (
            "\\setcounter{chapter}{0}\n"
            "\\renewcommand{\\thechapter}{" + tex_escape(cid) + "}\n"
            + body
        )
    return PREAMBLE + "\n\\title{数值方法实验室 · 章节 " + tex_escape(cid) + "}\n\\begin{document}\n" + body + "\\end{document}\n"


def has_xelatex() -> bool:
    return shutil.which("xelatex") is not None


REF_RE = re.compile(r"LaTeX Warning: (Reference|Citation) [`'][^']*' on page \d+ undefined")
OUT_RE = re.compile(r"Output written on .*?\((\d+) pages")


def compile_tex(tex: str, out_dir: pathlib.Path, *, runs: int = 2, root: pathlib.Path = ROOT) -> dict[str, Any]:
    """在 ``root`` 下用 xelatex 编译 ``tex``（相对 root 的路径），产物写进 out_dir。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    job = pathlib.Path(tex).stem
    started = time.perf_counter()
    log_text = ""
    rc = 0
    for _ in range(max(1, runs)):
        proc = subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={out_dir}",
                tex,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        rc = proc.returncode
        log_path = out_dir / f"{job}.log"
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else (proc.stdout or "")
        if rc != 0:
            break
    undefined = REF_RE.findall(log_text)
    pages = OUT_RE.search(log_text)
    pdf = out_dir / f"{job}.pdf"
    return {
        "job": job,
        "tex": tex,
        "rc": rc,
        "ok": rc == 0 and not undefined and pdf.exists(),
        "pages": int(pages.group(1)) if pages else None,
        "undefined": len(undefined),
        "overfull": len(re.findall(r"Overfull \\hbox", log_text)),
        "seconds": round(time.perf_counter() - started, 2),
        "tail": "\n".join(log_text.strip().splitlines()[-15:]) if rc != 0 else "",
    }


def build(root: pathlib.Path, *, only: str | None = None, do_compile: bool = True) -> dict[str, Any]:
    reg = Registry(root)
    reg.discover(strict=False)
    ids = reg.ids()
    if only:
        ids = [c for c in ids if c == only]
    report = None
    rep_path = root / "dist" / "reports" / "assertions.json"
    if rep_path.exists():
        report = json.loads(rep_path.read_text(encoding="utf-8"))

    problems: list[str] = []
    chapters: dict[str, Any] = {}
    book_dir = root / "dist" / "book"
    for cid in ids:
        entry = reg.get(cid)
        ch_problems, notes = check_structure(root, cid)
        made = write_asserts_table(root, cid, report, dry_run=do_compile)
        table = made[0] if made else None
        if made and made[1]:
            # 编译模式（--check）下不落盘：附表过期就是要报的问题，不是悄悄改掉
            ch_problems.append(
                f"{cid}: chapters/{cid}/asserts_table.tex 与最新断言报告不一致（先跑 python3 tools/run_asserts.py）"
            )
        info: dict[str, Any] = {
            "title": entry.title,
            "hasNotes": notes is not None,
            "hasTable": table is not None,
            "problems": ch_problems,
        }
        if do_compile and notes is not None and not ch_problems:
            book_dir.mkdir(parents=True, exist_ok=True)
            single = book_dir / f"one_{cid}.tex"
            single.write_text(wrapper_tex(cid, entry.title, single=True), encoding="utf-8", newline="\n")
            # 两遍：交叉引用（\ref 定理编号）第一遍必然是 undefined 的
            info["single"] = compile_tex(str(single.relative_to(root)), book_dir, runs=2)
            if not info["single"]["ok"]:
                ch_problems.append(f"{cid}: 单章编译失败（见 dist/book/one_{cid}.log）")
        problems.extend(ch_problems)
        info["problems"] = ch_problems
        info["ok"] = not ch_problems
        chapters[cid] = info

    book: dict[str, Any] | None = None
    if do_compile and not only:
        book_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            PREAMBLE,
            # 作者/标题也必须过 tex_escape：`build_book.py` 里的下划线在正文数学模式下
            # 会直接变成 "Missing $ inserted"（这个坑真的踩过一次）
            r"\title{" + tex_escape("数值方法实验室：28 章交互式数值方法笔记") + "}",
            r"\author{" + tex_escape("由 tools/build_book.py 自动汇编") + "}",
            r"\begin{document}",
            r"\maketitle",
            r"\tableofcontents",
        ]
        for cid in ids:
            if chapters[cid]["hasNotes"]:
                lines.append(r"\input{chapters/%s/notes.tex}" % cid)
        lines += [r"\appendix", r"\chapter{各章数值断言（机器生成）}"]
        for cid in ids:
            if chapters[cid]["hasTable"]:
                lines.append(r"\section*{章节 %s}" % cid)
                lines.append(r"\input{chapters/%s/asserts_table.tex}" % cid)
        lines.append(r"\end{document}")
        book_tex = book_dir / "book.tex"
        book_tex.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        book = compile_tex("dist/book/book.tex", book_dir, runs=2)
        if not book["ok"]:
            problems.append("合并本编译失败（见 dist/book/book.log）")

    return {
        "ok": not problems,
        "problems": problems,
        "chapters": chapters,
        "book": book,
        "xelatex": has_xelatex(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="结构检查 + 逐章/合并本 xelatex 编译（G9 门）")
    ap.add_argument("--check", action="store_true", help="CI 模式：任何问题都 exit 1")
    ap.add_argument("--no-compile", action="store_true", help="只做结构检查与附表生成")
    ap.add_argument("--only", default=None, help="只处理某一章")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--report", type=pathlib.Path, default=ROOT / "dist" / "reports" / "book.json")
    args = ap.parse_args(list(argv) if argv is not None else None)

    do_compile = not args.no_compile and has_xelatex()
    if not do_compile and not args.no_compile:
        print("[提示] 环境里没有 xelatex：跳过编译，只做结构检查与附表生成")
    result = build(ROOT, only=args.only, do_compile=do_compile)
    report: pathlib.Path = args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps({"ok": result["ok"], "problems": result["problems"], "book": result["book"]}, ensure_ascii=False))
    else:
        for cid, info in sorted(result["chapters"].items()):
            if info["problems"]:
                print(f"  {cid}: ❌ " + "；".join(info["problems"]))
            else:
                single = info.get("single", {})
                extra = f"，单章 {single.get('pages', '?')} 页，{single.get('seconds', 0)} s" if single else ""
                print(f"  {cid}: ✅ notes/附表齐全{extra}")
        if result["book"]:
            b = result["book"]
            print(f"合并本：{'✅' if b['ok'] else '❌'} {b.get('pages', '?')} 页，未定义引用 {b['undefined']}，Overfull {b['overfull']}，{b['seconds']} s")
        if result["ok"]:
            print("G9 通过：结构检查与编译均无问题")
        else:
            print(f"G9 失败：{len(result['problems'])} 个问题")
            for p in result["problems"]:
                print(f"  - {p}")
        print(f"报告：{report.relative_to(ROOT)}")
    if args.check and not result["ok"]:
        return 1
    return 0 if result["ok"] else 0 if not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
