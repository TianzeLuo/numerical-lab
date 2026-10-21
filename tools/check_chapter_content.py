#!/usr/bin/env python3
"""给章节作者用的内容体检：参数解释、推导步骤、公式定界符。

用法：
    /home/user/anaconda3/bin/python tools/check_chapter_content.py server/chapters/1.2.py

只检查**数据**（MANIFEST），不跑帧、不跑浏览器。CI 的正式公式门在
`viz/tests/site_math.test.ts`；这里提前把同类错误暴露给作者。
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.manifest import load_manifest, validate_manifest  # noqa: E402

MIN_HELP = 24
MIN_STEPS = 4
MIN_THEORY_CHARS = 500

#: 轴名/参数名/时间轴名里不允许出现的“裸数学”：这些字符一旦落在 $...$ 外，
#: HTML 与画布都会按普通正文绘制，读者看到的是源码而不是公式。
_GREEK = "αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"
_BARE_MATH = re.compile(
    "[" + re.escape(_GREEK) + "]"
    + r"|(?<![\\$A-Za-z0-9_])[A-Za-z]_"
    + r"|\^|‖|∫|√|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻₀₁₂₃₄₅₆₇₈₉]"
)

ALLOWED = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta",
    "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "varpi", "rho",
    "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", "varphi", "chi", "psi",
    "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
    "Phi", "Psi", "Omega", "frac", "dfrac", "tfrac", "sqrt", "sum", "prod", "int",
    "partial", "nabla", "infty", "le", "leq", "ge", "geq", "ne", "neq", "approx",
    "sim", "simeq", "equiv", "times", "cdot", "pm", "mp", "to", "rightarrow",
    "leftarrow", "leftrightarrow", "Rightarrow", "Leftarrow", "Leftrightarrow",
    "ldots", "dots", "cdots", "left", "right", "text", "mathrm", "mathbf", "mathit",
    "operatorname", "hat", "bar", "tilde", "vec", "dot", "ddot", "mathbb", "mathcal",
    "quad", "qquad", "|", "{", "}", "in", "notin", "subset", "subseteq", "cup", "cap",
    "log", "ln", "lg", "exp", "max", "min", "arg", "det", "rank", "Tr", "tr", "sin",
    "cos", "tan", "sinh", "cosh", "tanh", "lim", "sup", "inf", "gcd", "mod", "sgn",
    "cond", "diag", "span", "dim", "deg", "sec", "csc", "cot",
    "langle", "rangle", "lvert", "rvert", "lVert", "rVert", "lfloor", "rfloor",
    "lceil", "rceil", "perp", "parallel", "prime", "circ", "ast", "star", "hbar",
    "ell", "setminus", "emptyset", "varnothing", "forall", "exists", "neg", "land",
    "lor", "oplus", "otimes", "mapsto", "uparrow", "downarrow", "angle", "degree",
    "quad", ",", ";", ":", "!", " ", "%",
}


def _load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"content_{path.stem.replace('.', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_label(value: str, where: str, errors: list[str], *, axis: bool = False) -> None:
    """标签里的数学必须走 TeX；轴名里也不允许用反引号冒充公式。"""
    if not value:
        return
    if axis and "`" in value:
        errors.append(f"{where}: 轴名里有反引号（画布不认 Markdown code，请用 $...$）")
    bare = re.sub(r"\$[^$]*\$|`[^`]*`", "", value)
    hit = _BARE_MATH.search(bare)
    if hit:
        errors.append(f"{where}: 数学片段 {hit.group(0)!r} 没有放进 $...$")


def _walk_strings(value: Any) -> Any:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def _check_control_chars(value: Any, errors: list[str]) -> None:
    for text in _walk_strings(value):
        bad = next((ch for ch in text if ord(ch) < 32 and ch not in "\n\t"), None)
        if bad is not None:
            errors.append(f"MANIFEST 字符串里有控制字符 U+{ord(bad):04X}（多半是 \\u 转义被吃掉了）")


def _check_math(text: str, where: str, errors: list[str]) -> None:
    i = 0
    while i < len(text):
        if text[i] != "$":
            if text[i] == "\\" and not text.startswith("\\(", i):
                errors.append(f"{where}: 公式外出现反斜杠（会原样显示）")
            i += 1
            continue
        end = text.find("$", i + 1)
        if end < 0:
            errors.append(f"{where}: 未闭合的 $...$")
            return
        body = text[i + 1:end]
        depth = 0
        j = 0
        while j < len(body):
            ch = body[j]
            if ch == "\\":
                m = re.match(r"\\([A-Za-z]+|.)", body[j:])
                name = m.group(1) if m else ""
                if name.isalpha() and name not in ALLOWED:
                    errors.append(f"{where}: 不支持的公式命令 \\{name}")
                j += len(m.group(0)) if m else 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth != 0:
            errors.append(f"{where}: 花括号不平衡")
        i = end + 1


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--all":
        rc = 0
        for chapter in sorted((ROOT / "server" / "chapters").glob("*.py")):
            rc = main([argv[0], str(chapter)]) or rc
        return rc
    if len(argv) != 2:
        print("用法：check_chapter_content.py <server/chapters/X.Y.py> | --all")
        return 2
    path = Path(argv[1])
    if not path.is_absolute():
        path = ROOT / path
    module = _load(path)
    raw = module.MANIFEST
    problems = validate_manifest(raw)
    errors: list[str] = []
    notes: list[str] = []
    manifest = load_manifest(raw)
    for viz in manifest.vizzes:
        _check_label(viz.title, f"{viz.id}.title", errors)
        for param in viz.params:
            _check_label(param.label, f"{viz.id}.{param.name}.label", errors)
            if not param.help.strip():
                errors.append(f"{viz.id}.{param.name}: 缺 help 展开解释")
            elif len(param.help.strip()) < MIN_HELP:
                errors.append(f"{viz.id}.{param.name}: help 太短（{len(param.help.strip())} < {MIN_HELP} 字）")
            else:
                _check_math(param.help, f"{viz.id}.{param.name}.help", errors)
        time_label = getattr(getattr(viz, "time", None), "label", "") or ""
        _check_label(time_label, f"{viz.id}.time.label", errors)
        if len(viz.theory) < MIN_STEPS:
            errors.append(f"{viz.id}: 推导只有 {len(viz.theory)} 步（至少 {MIN_STEPS} 步）")
        total = sum(len(s.strip()) for s in viz.theory)
        if total < MIN_THEORY_CHARS:
            errors.append(f"{viz.id}: 推导总字数 {total}（至少 {MIN_THEORY_CHARS}）")
        for i, line in enumerate(viz.theory):
            _check_math(line, f"{viz.id}.theory[{i}]", errors)
        for i, ax in enumerate(viz.axes):
            for field in ("title", "xlabel", "ylabel"):
                value = getattr(ax, field, "")
                if value:
                    _check_math(value, f"{viz.id}.axes[{i}].{field}", errors)
                    _check_label(value, f"{viz.id}.axes[{i}].{field}", errors, axis=True)
    _check_control_chars(raw, errors)
    errors.extend(problems)
    if errors:
        print(f"{path}: {len(errors)} 个问题")
        for item in errors:
            print(" -", item)
        return 1
    print(f"{path}: OK（{len(manifest.vizzes)} 个实验，"
          f"{sum(len(v.params) for v in manifest.vizzes)} 个参数全部有解释）")
    for note in notes:
        print(" ·", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
