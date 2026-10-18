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
    python3 tools/build_book.py --no-compile     # 只生成/刷新附表 + 结构检查（先写表、后检查）
    python3 tools/build_book.py --only 0.0       # 只处理一章
    python3 tools/build_book.py --json

注意 ``--no-compile`` 的"写"是有保护的：``dist/reports/assertions.json`` 是全仓共享的，
若报告里没有某章的断言记录（并发下常被别的章覆盖成残缺报告），就**保留**那一章现有的
``asserts_table.tex`` 不重写，并在输出里说明；``--only`` 显式点名的章不受此保护。
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
\usepackage{array}
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


#: Unicode 数学/希腊符号 → LaTeX。**必须**做这层映射：xelatex 的默认字体缺
#: ``≤``(U+2264)、``‖``(U+2016) 这类字形，日志里只留一行 "Missing character"，
#: 正文里那个字符就**静默消失**了（断言附表里出现过 ``漂移 ≤ 1.7e-17`` 印成 ``漂移 �``）。
#: 这些字符在断言报告的 theory/measured 字段里到处都是，所以映射做在 tex_escape 里，
#: 一处生效、全表受益。映射表是单遍查表的一部分（先查它，再查 TEX_ESCAPE）。
UNICODE_MATH: dict[str, str] = {
    "≤": r"$\le$", "≥": r"$\ge$", "≠": r"$\ne$", "≈": r"$\approx$", "∼": r"$\sim$",
    "≡": r"$\equiv$", "≪": r"$\ll$", "≫": r"$\gg$", "∝": r"$\propto$",
    "→": r"$\to$", "←": r"$\leftarrow$", "↦": r"$\mapsto$", "⇒": r"$\Rightarrow$",
    "⟺": r"$\iff$", "⟹": r"$\implies$", "∈": r"$\in$", "∀": r"$\forall$", "∃": r"$\exists$",
    "∞": r"$\infty$", "∂": r"$\partial$", "∇": r"$\nabla$", "∫": r"$\int$", "∑": r"$\sum$",
    "Σ": r"$\Sigma$", "∏": r"$\prod$", "√": r"$\surd$", "±": r"$\pm$", "∓": r"$\mp$",
    "×": r"$\times$", "÷": r"$\div$", "·": r"$\cdot$", "⋯": r"$\cdots$", "…": r"$\dots$",
    "−": r"$-$", "–": "--", "—": "---", "‖": r"$\|$", "⟨": r"$\langle$", "⟩": r"$\rangle$",
    "⊂": r"$\subset$", "⊆": r"$\subseteq$", "∩": r"$\cap$", "∪": r"$\cup$", "∅": r"$\varnothing$",
    "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$", "δ": r"$\delta$", "ε": r"$\varepsilon$",
    "ζ": r"$\zeta$", "η": r"$\eta$", "θ": r"$\theta$", "ι": r"$\iota$", "κ": r"$\kappa$",
    "λ": r"$\lambda$", "μ": r"$\mu$", "ν": r"$\nu$", "ξ": r"$\xi$", "π": r"$\pi$",
    "ρ": r"$\rho$", "σ": r"$\sigma$", "τ": r"$\tau$", "φ": r"$\varphi$", "χ": r"$\chi$",
    "ψ": r"$\psi$", "ω": r"$\omega$", "Γ": r"$\Gamma$", "Δ": r"$\Delta$", "Θ": r"$\Theta$",
    "Λ": r"$\Lambda$", "Ξ": r"$\Xi$", "Π": r"$\Pi$", "Φ": r"$\Phi$", "Ψ": r"$\Psi$",
    "Ω": r"$\Omega$", "ℓ": r"$\ell$", "ℏ": r"$\hbar$", "ᵀ": r"$^{\mathsf T}$",
    "⁻¹": r"$^{-1}$", "½": r"$1/2$", "°": r"$^\circ$", "′": r"$'$",
    # 角标：断言报告里 κ₁、n²、ρ³ 这类写法很常见，而 U+2081/U+00B2 同样会缺字形
    "₀": "$_0$", "₁": "$_1$", "₂": "$_2$", "₃": "$_3$", "₄": "$_4$",
    "₅": "$_5$", "₆": "$_6$", "₇": "$_7$", "₈": "$_8$", "₉": "$_9$",
    "ₖ": "$_k$", "ᵢ": "$_i$", "ⱼ": "$_j$", "ₙ": "$_n$",
    "⁰": "$^0$", "¹": "$^1$", "²": "$^2$", "³": "$^3$", "⁴": "$^4$",
    "⁵": "$^5$", "⁶": "$^6$", "⁷": "$^7$", "⁸": "$^8$", "⁹": "$^9$",
    # 全角括号：中日韩字体里有，但和 ASCII 数学混排时用半角更安全、也更省宽度
    "（": "(", "）": ")", "【": "[", "】": "]",
}

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


def _display_path(path: pathlib.Path, root: pathlib.Path) -> str:
    """尽量给相对路径，给不出就原样返回。

    与 ``tools/run_asserts.py`` 里的同名 helper 一个道理：``--report`` 可以被指到仓库外
    （CI 的临时目录、``/tmp/book.json``），那时 ``path.relative_to(root)`` 会抛
    ``ValueError: ... is not in the subpath of ...``——报告早就写好了，只是打印路径时崩掉。
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def tex_escape(s: Any) -> str:
    r"""把任意文本变成"在 LaTeX 正文里安全"的文本。

    必须是**单遍**查表：第一版用连续 replace，`\\` 先被替换成 ``\\textbackslash{}``，
    随后 ``{``/``}`` 又把刚插进去的花括号转义了，于是编译出来是 ``\textbackslash\{\}``
    （印出两个字面花括号）。测试把它钉住了。
    """
    out: list[str] = []
    i = 0
    text = str(s)
    while i < len(text):
        # 先看双字符映射（⁻¹ 这类组合不能按单字符切）
        pair = text[i : i + 2]
        if pair in UNICODE_MATH:
            out.append(UNICODE_MATH[pair])
            i += 2
            continue
        ch = text[i]
        if ch in UNICODE_MATH:
            out.append(UNICODE_MATH[ch])
        else:
            out.append(TEX_ESCAPE.get(ch, ch))
        i += 1
    return "".join(out)


def allow_wrap(s: str) -> str:
    r"""在长标识符的每个 ``\_`` 之后插 ``\allowbreak``。

    ``\texttt{test\_symbolic\_fill\_matches\_exact\_elimination}`` 这种串在窄表列里
    是**不可断行**的，只能冲出右边界；插了可断点之后 TeX 才有地方换行。
    它只加断点、不改字形，所以在正文里用也一样安全。
    """
    return s.replace("\\_", "\\_\\allowbreak{}")


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
    return soft_breaks(tex_escape(v))


def soft_breaks(s: str) -> str:
    r"""在 ``/``、``=``、``(`` 之后给 TeX 一个可断点。

    ``实测值`` 列里全是 ``0.498537(n=32..256: 0.4884, 0.4942, ...)`` 这种长 ASCII 串，
    TeX 在 ``p{}`` 列里只能按空格断行，于是一路冲出右边界。这些位置本来就不该黏在一起，
    加断点只是"允许"它断，不影响不断时的排版。
    """
    for ch in ("/", "=", ",", "(", "、"):
        s = s.replace(ch, ch + r"\allowbreak{}")
    return s


#: 附表文本里"xelatex + ctex 一定没有字形"的字符类别。
#:
#: 由 2.4 章的作者实测补上这一课：`record(theory=...)` 是自由文本，而上标/下标 Unicode
#: （`ᵖ ⁽ ⁾ ₀ ①`）与 `⇔ ≳ ⊥` 这类符号**会被 xelatex 静默吞掉**——单章编译（G9）才报
#: "missing glyphs"，而那时数字已经印进 PDF 的正文里了。所以在**生成附表之前**就查一遍。
#: 判定用白名单：ASCII + CJK + 希腊字母 + 已在本仓库 30+ 页构建里验证过的数学符号。
_SAFE_EXTRA = set("αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"
                  "∑∏∫√≈≠≤≥±×÷∞→←↔∂∇∈∉⊂⊆∪∩⋅·°′″⌈⌉⌊⌋")
_SAFE_RANGES = ((0x20, 0x7E), (0x3000, 0x303F), (0x4E00, 0x9FFF), (0xFF00, 0xFFEF), (0x2010, 0x2027), (0x2018, 0x201D))


def latex_risky_chars(text: str) -> list[str]:
    """返回文本里"附表/笔记里 xelatex 没有字形"的字符（去重、按出现顺序）。"""
    bad: list[str] = []
    for ch in text:
        if ch in _SAFE_EXTRA or any(lo <= ord(ch) <= hi for lo, hi in _SAFE_RANGES):
            continue
        if ch in ("\n", "\t", "\r"):
            continue
        if ch not in bad:
            bad.append(ch)
    return bad


def write_asserts_table(
    root: pathlib.Path,
    cid: str,
    report: dict[str, Any] | None,
    *,
    dry_run: bool = False,
    require_records: bool = False,
) -> tuple[pathlib.Path, bool] | None:
    r"""从断言报告生成该章的 LaTeX 附表；没有报告就不生成（返回 ``None``）。

    返回 ``(路径, 是否有变化)``——``dry_run=True`` 时只算不写，供 ``--check`` 做漂移判定：
    附表是 ``notes.tex`` ``\input`` 的**正文内容**，如果它与最新断言报告不一致，
    印出来的 PDF 就在撒谎（手抄数字的老毛病换个形式又回来了）。

    ``require_records=True`` 时，报告里**没有该章的任何断言记录**（整块缺失，或 ``items`` 为空）
    就不写、返回 ``None``：``dist/reports/assertions.json`` 是全仓共享、每次 pytest 覆盖的，
    别的 agent 跑一次单章测试就可能把它换成"只有那一章"的残缺报告；照着这种报告写表，
    会把本已写好的附表重写成空表。并发下"什么都不动"比"写一张空表"安全。
    """
    if not report:
        return None
    blk = (report.get("chapters") or {}).get(cid)
    if not blk:
        return None
    if require_records and not blk.get("items"):
        return None
    out = root / "chapters" / cid / "asserts_table.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [ASSERTS_TABLE_HEADER.replace("__CID__", cid).replace("__TS__", tex_escape(report.get("generated_at", "?")))]
    for rec in blk.get("items", []):
        mark = r"\checkmark" if rec.get("passed") else r"$\times$"
        lines.append(
            f"{tex_escape(rec.get('kind', '?'))} & {allow_wrap(tex_escape(rec.get('label') or rec.get('id')))} & "
            f"{fmt(rec.get('theory'))} & {fmt(rec.get('measured'))} {mark} \\\\"
        )
    if not blk.get("items"):
        lines.append(r"\multicolumn{4}{c}{（本章暂无断言记录）} \\")
    lines.append(r"\end{longtable}")
    lines.append("")
    text = "\n".join(lines)
    # 生成前先查字形：附表是 notes.tex 的正文，印出来才报"缺字形"就晚了
    # （这条不是拒绝写文件——写还是要写，问题由 check_structure 报给 G9）。
    old = out.read_text(encoding="utf-8") if out.exists() else None
    changed = old != text
    if changed and not dry_run:
        out.write_text(text, encoding="utf-8", newline="\n")
    return out, changed


def check_structure(root: pathlib.Path, cid: str) -> tuple[list[str], pathlib.Path | None]:
    """返回 (问题列表, notes 路径)。"""
    problems: list[str] = []
    # 附表文本里的"无字形字符"要在这里报出来（否则只能在 xelatex 日志里看到
    # "Missing character"，而且它**不会**让编译失败——数字就那么少了一撇）。
    table = root / "chapters" / cid / "asserts_table.tex"
    if table.exists():
        bad = latex_risky_chars(table.read_text(encoding="utf-8"))
        if bad:
            shown = "、".join(repr(c) for c in bad[:6])
            problems.append(
                f"{cid}: asserts_table.tex 含 xelatex 无字形的字符：{shown}"
                f"（共 {len(bad)} 种）——请把断言文本里的上标/下标 Unicode 改成 ASCII 写法"
                "（`bᵀ` → `b^T`、`Q⁻¹` → `Q^{-1}`）"
            )
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


MISSING_RE = re.compile(r"Missing character: There is no (\S+)")
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
    # 缺字形是**静默**的：xelatex 照常返回 0、PDF 照样生成，只是那个字符不见了。
    # 因此把它算作失败条件（曾经 ``≤`` 就是这样从断言附表里蒸发的）。
    missing = MISSING_RE.findall(log_text)
    return {
        "job": job,
        "tex": tex,
        "rc": rc,
        "ok": rc == 0 and not undefined and not missing and pdf.exists(),
        "missing_glyphs": len(missing),
        "missing_samples": sorted(set(missing))[:5],
        "pages": int(pages.group(1)) if pages else None,
        "undefined": len(undefined),
        # 具体名字：只报"单章编译失败"时作者还得去翻 one_<id>.log（3.1 章踩过：
        # 跨章 \eqref 到 3.2 章的 label，单章编译下那个 label 根本不存在）。
        "undefined_refs": sorted({m.group(0).split("`")[-1].split("'")[0] for m in REF_RE.finditer(log_text)})[:8],
        "errors": [ln.strip() for ln in log_text.splitlines() if ln.startswith("!")][:3],
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
        table_path = root / "chapters" / cid / "asserts_table.tex"
        # 顺序很关键：**先**生成/刷新附表，**再**做结构检查。
        # check_structure 要求 notes.tex 里 \input 到的文件都存在，而 asserts_table.tex 正是
        # 在这里生成的；反过来的话，第一次 --no-compile 必然先报"asserts_table.tex 不存在"，
        # 让人以为失败（3.2 章就是这么被误导的）。
        #
        # 报告里没有本章记录的：默认保留现有附表，只有 --only 点名的章照旧无条件写。
        # 这条保护只作用于**写**的路径：--check（dry_run）的判定逻辑一点没动，
        # 附表过期在 --check 下照旧报问题（G9 单章跑就是要严格）。
        force = only is not None and cid == only
        require_records = (not do_compile) and not force
        made = write_asserts_table(root, cid, report, dry_run=do_compile, require_records=require_records)
        # 没写不代表没有：--no-compile 跳过写入时，磁盘上原有的附表仍是有效内容，合并本附录照旧
        # input 它（hasTable 为真）。--check（do_compile=True）保持原样：有没有附表只看报告块，
        # 免得"报告里没有的章"悄悄改变合并本的构成。
        if made is not None:
            table = made[0]
        else:
            table = table_path if (not do_compile and table_path.exists()) else None
        ch_problems, notes = check_structure(root, cid)
        if made and made[1] and do_compile:
            # --check 模式下不落盘：附表过期就是要报的问题，不是悄悄改掉。
            # 反过来 --no-compile 就是"来生成附表的"，写完之后不该再报它过期。
            ch_problems.append(
                f"{cid}: chapters/{cid}/asserts_table.tex 与最新断言报告不一致（先跑 python3 tools/run_asserts.py）"
            )
        info: dict[str, Any] = {
            "title": entry.title,
            "hasNotes": notes is not None,
            "hasTable": table is not None,
            "problems": ch_problems,
        }
        if made is None and not do_compile:
            # 明细账要能解释"为什么这次没刷新附表"，不能只是安静地什么都不做。
            # （--check 是 dry_run，它的输出要点保持原样，不在这里加话。）
            if report is None:
                info["tableKept"] = "没有断言报告（dist/reports/assertions.json），保持现有附表不动"
            elif table_path.exists():
                info["tableKept"] = f"报告里没有 {cid} 的记录，保留现有 asserts_table.tex"
            else:
                info["tableKept"] = f"报告里没有 {cid} 的记录，未生成 asserts_table.tex"
        if do_compile and notes is not None and not ch_problems:
            book_dir.mkdir(parents=True, exist_ok=True)
            single = book_dir / f"one_{cid}.tex"
            single.write_text(wrapper_tex(cid, entry.title, single=True), encoding="utf-8", newline="\n")
            # 两遍：交叉引用（\ref 定理编号）第一遍必然是 undefined 的
            info["single"] = compile_tex(str(single.relative_to(root)), book_dir, runs=2)
            if not info["single"]["ok"]:
                miss = int(info["single"].get("missing_glyphs") or 0)
                extra = (
                    f"；日志里有 {miss} 处缺字形（{info['single'].get('missing_samples')}），"
                    "xelatex 只是把字吞掉、不会报错"
                    if miss
                    else ""
                )
                # 未定义引用与其它错误要给**具体名字**：只报"单章编译失败"时，
                # 作者得自己去翻 one_<id>.log 才知道是哪个 label（3.1 章踩过——跨章
                # \eqref 到 3.2 章，单章编译下那个 label 不存在）。
                undef = list(info["single"].get("undefined_refs") or [])
                if undef:
                    extra += f"；未定义引用 {len(undef)} 处：{', '.join(undef[:6])}"
                err_lines = list(info["single"].get("errors") or [])
                if err_lines:
                    extra += f"；错误行：{err_lines[0][:160]}"
                ch_problems.append(f"{cid}: 单章编译失败（见 dist/book/one_{cid}.log{extra}）")
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
            miss = int(book.get("missing_glyphs") or 0)
            extra = (
                f"；{miss} 处缺字形（{book.get('missing_samples')}）——字被静默吞掉，G9 必须报出来"
                if miss
                else ""
            )
            problems.append(f"合并本编译失败（见 dist/book/book.log{extra}）")

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
            if info.get("tableKept"):
                # 并发下这是正常的：报告被别人换成了残缺的一份，附表保持原样，别误读成失败。
                print(f"      ↳ {info['tableKept']}")
        if result["book"]:
            b = result["book"]
            print(f"合并本：{'✅' if b['ok'] else '❌'} {b.get('pages', '?')} 页，未定义引用 {b['undefined']}，Overfull {b['overfull']}，{b['seconds']} s")
        if result["ok"]:
            print("G9 通过：结构检查与编译均无问题")
        else:
            print(f"G9 失败：{len(result['problems'])} 个问题")
            for p in result["problems"]:
                print(f"  - {p}")
        print(f"报告：{_display_path(report, ROOT)}")
    if args.check and not result["ok"]:
        return 1
    return 0 if result["ok"] else 0 if not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
