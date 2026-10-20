#!/usr/bin/env python3
"""Validate data/site/formulas.json and data/site/guides.json against the chapter manifest.

Only PyYAML is required beyond the standard library.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS_YAML = ROOT / "docs" / "chapters.yaml"
FORMULAS_JSON = ROOT / "data" / "site" / "formulas.json"
GUIDES_JSON = ROOT / "data" / "site" / "guides.json"

GREEK_COMMANDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta",
    "eta", "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu",
    "xi", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau",
    "upsilon", "phi", "varphi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
    "Phi", "Psi", "Omega",
}

# Conservative subset agreed for the custom TeX -> MathML renderer.
ALLOWED_COMMANDS = GREEK_COMMANDS | {
    "frac", "sqrt",
    "sum", "prod", "int",
    "partial", "nabla", "infty",
    "le", "ge", "ne", "approx", "sim", "times", "cdot", "pm", "to",
    "ldots",
    "left", "right",
    "text", "mathrm", "mathbf", "operatorname",
    "hat", "bar", "tilde", "vec",
    "mathbb", "mathcal",
    "quad", "qquad",
    # escaped literals used by set notation and norms
    "{", "}", "|",
}

FORBIDDEN_SNIPPETS = (
    "\\begin", "\\end", "\\label", "\\tag", "\\nonumber", "\\notag",
    "\\bigl", "\\bigr", "\\Bigl", "\\Bigr", "\\biggl", "\\biggr",
    "\\displaystyle", "\\limits", "\\substack",
    "\\operatorname*", "\\sqrt[",
)

BANNED_CAPTION_PHRASES = (
    "规则", "防误导", "不许", "必须", "别把", "别用", "别被", "别以为",
    "骗", "不能", "这是 bug",
)

BANNED_GUIDE_PHRASES = (
    "规则", "防误导", "必须", "不许", "别被", "别以为", "骗", "bug",
    "本章将", "我们会证明",
)

CANONICAL_CHAPTERS = (
    [f"1.{i}" for i in range(1, 6)]
    + [f"2.{i}" for i in range(1, 7)]
    + [f"3.{i}" for i in range(1, 6)]
    + [f"4.{i}" for i in range(1, 7)]
    + [f"5.{i}" for i in range(1, 5)]
    + ["6.1", "6.2"]
)

CMD_RE = re.compile(r"\\([A-Za-z]+|.)")


def load_expected_chapters() -> list[str]:
    with CHAPTERS_YAML.open(encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    chapters = manifest["chapters"]
    expected = [str(item["id"]) for item in chapters if str(item["id"]) != "0.0"]
    return expected


def strip_commands(tex: str) -> str:
    """Drop command names/symbols, keeping their brace-delimited arguments."""
    out: list[str] = []
    i = 0
    while i < len(tex):
        if tex[i] == "\\":
            i += 1
            if i >= len(tex):
                break
            if tex[i].isalpha():
                while i < len(tex) and tex[i].isalpha():
                    i += 1
            else:
                # \{, \}, \| ... are literal symbols, not grouping delimiters
                i += 1
            out.append(" ")
        else:
            out.append(tex[i])
            i += 1
    return "".join(out)


def check_brackets(tex: str) -> list[str]:
    clean = strip_commands(tex)
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in clean:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return ["unbalanced brackets (mismatched nesting)"]
            stack.pop()
    if stack:
        return [f"unclosed bracket {stack[-1]!r}"]
    return []


def check_formula(tex: str, where: str) -> list[str]:
    errors: list[str] = []

    if not isinstance(tex, str) or not tex.strip():
        return [f"{where}: tex must be a nonempty string"]
    if "\n" in tex or "\r" in tex:
        errors.append(f"{where}: tex must be a single line")
    if "$" in tex:
        errors.append(f"{where}: tex must not contain $ delimiters")
    if "\\[" in tex or "\\]" in tex or "\\(" in tex or "\\)" in tex:
        errors.append(f"{where}: tex must not contain display/inline math delimiters")
    if "&" in tex:
        errors.append(f"{where}: alignment character & is not supported")
    if "\\\\" in tex:
        errors.append(f"{where}: multi-line \\\\ is not supported")
    if tex.rstrip().endswith("\\"):
        errors.append(f"{where}: trailing backslash")
    if len(tex) > 220:
        errors.append(f"{where}: tex is {len(tex)} chars (limit 220)")

    for snippet in FORBIDDEN_SNIPPETS:
        if snippet in tex:
            errors.append(f"{where}: forbidden construct {snippet}")

    left_depth = 0
    for match in CMD_RE.finditer(tex):
        name = match.group(1)
        if name == "left":
            left_depth += 1
        elif name == "right":
            left_depth -= 1
            if left_depth < 0:
                errors.append(f"{where}: \\right without matching \\left")
                left_depth = 0
        if name not in ALLOWED_COMMANDS:
            errors.append(f"{where}: unsupported command \\{name}")
    if left_depth != 0:
        errors.append(f"{where}: {left_depth} unmatched \\left command(s)")

    errors.extend(f"{where}: {msg}" for msg in check_brackets(tex))

    # Literal set braces (\{ ... \}) must also be paired.
    if tex.count(r"\{") != tex.count(r"\}"):
        errors.append(f"{where}: unbalanced literal set braces \\{{ \\}}")

    # Norms and absolute-value bars must come in pairs.
    if tex.count("|") % 2:
        errors.append(f"{where}: odd number of | delimiters")

    for match in re.finditer(r"\\mathbb\{([^{}]*)\}", tex):
        if match.group(1) not in {"R", "N", "O"}:
            errors.append(f"{where}: unsupported \\mathbb{{{match.group(1)}}}")
    for match in re.finditer(r"\\mathcal\{([^{}]*)\}", tex):
        if match.group(1) != "O":
            errors.append(f"{where}: unsupported \\mathcal{{{match.group(1)}}}")
    for match in re.finditer(r"\\operatorname\{([^{}]*)\}", tex):
        if not re.fullmatch(r"[A-Za-z]+", match.group(1)):
            errors.append(f"{where}: \\operatorname argument must be letters only")

    return errors


def check_caption(caption: str, where: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(caption, str) or not caption.strip():
        return [f"{where}: caption must be a nonempty string"]
    if "\n" in caption or "\r" in caption:
        errors.append(f"{where}: caption must be a single line")
    if not (8 <= len(caption) <= 24):
        errors.append(f"{where}: caption length {len(caption)} outside 8..24")
    for phrase in BANNED_CAPTION_PHRASES:
        if phrase in caption:
            errors.append(f"{where}: caption contains discouraged phrase {phrase!r}")
    return errors


def check_guides(expected: list[str]) -> tuple[list[str], dict[str, int]]:
    """Validate the chapter introductions shown at the top of each chapter page."""
    errors: list[str] = []
    stats: dict[str, int] = {}
    try:
        with GUIDES_JSON.open(encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        return [f"cannot load {GUIDES_JSON.relative_to(ROOT)}: {exc}"], stats
    if not isinstance(raw, dict) or not isinstance(raw.get("chapters"), dict):
        return ["guides.json must contain a chapters object"], stats
    chapters = raw["chapters"]
    actual = list(chapters.keys())
    if actual != expected:
        missing = [ch for ch in expected if ch not in chapters]
        extra = [ch for ch in actual if ch not in expected]
        if missing:
            errors.append(f"guides: missing chapters: {', '.join(missing)}")
        if extra:
            errors.append(f"guides: unexpected chapters: {', '.join(extra)}")
        if not missing and not extra:
            errors.append("guides: chapter order must follow 1.1 ... 6.2")
    lengths: list[int] = []
    for ch in expected:
        entry = chapters.get(ch)
        if not isinstance(entry, dict):
            errors.append(f"guides.{ch}: entry must be an object")
            continue
        intro = entry.get("intro")
        if not isinstance(intro, str) or not intro.strip():
            errors.append(f"guides.{ch}: intro must be a nonempty string")
            continue
        length = len(intro.strip())
        lengths.append(length)
        if not 80 <= length <= 220:
            errors.append(f"guides.{ch}: intro length {length} outside 80..220")
        if "\n" in intro:
            errors.append(f"guides.{ch}: intro must be a single paragraph")
        for phrase in BANNED_GUIDE_PHRASES:
            if phrase in intro:
                errors.append(f"guides.{ch}: banned phrase {phrase!r}")
        glossary = entry.get("glossary")
        if not isinstance(glossary, list) or not 2 <= len(glossary) <= 4:
            errors.append(f"guides.{ch}: glossary must list 2..4 terms")
            continue
        seen_terms: set[str] = set()
        for i, item in enumerate(glossary):
            where = f"guides.{ch}.glossary[{i}]"
            if not isinstance(item, dict):
                errors.append(f"{where}: entry must be an object")
                continue
            term = item.get("term")
            desc = item.get("desc")
            if not isinstance(term, str) or not term.strip() or len(term.strip()) > 24:
                errors.append(f"{where}: term must be a nonempty string up to 24 chars")
                continue
            if "\n" in term or "\n" in str(desc):
                errors.append(f"{where}: term/desc must be single-line")
            if term.strip() in seen_terms:
                errors.append(f"{where}: duplicate term {term!r}")
            seen_terms.add(term.strip())
            if not isinstance(desc, str) or not 8 <= len(desc.strip()) <= 80:
                errors.append(f"{where}: desc length must be 8..80 chars")
                continue
            for phrase in BANNED_GUIDE_PHRASES:
                if phrase in term or phrase in desc:
                    errors.append(f"{where}: banned phrase {phrase!r}")
        stats.setdefault("glossary", 0)
        stats["glossary"] += len(glossary)
    if lengths:
        stats["count"] = len(lengths)
        stats["min"] = min(lengths)
        stats["max"] = max(lengths)
    return errors, stats


def main() -> int:
    errors: list[str] = []

    try:
        expected = load_expected_chapters()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"FAIL: cannot load {CHAPTERS_YAML}: {exc}", file=sys.stderr)
        return 2

    if expected != CANONICAL_CHAPTERS:
        missing_from_manifest = [ch for ch in CANONICAL_CHAPTERS if ch not in expected]
        extra_in_manifest = [ch for ch in expected if ch not in CANONICAL_CHAPTERS]
        errors.append(
            "docs/chapters.yaml: chapter list must be exactly 1.1-6.2"
            + (f" (missing: {', '.join(missing_from_manifest)})" if missing_from_manifest else "")
            + (f" (extra: {', '.join(extra_in_manifest)})" if extra_in_manifest else "")
        )
    expected = CANONICAL_CHAPTERS

    try:
        with FORMULAS_JSON.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"FAIL: cannot load {FORMULAS_JSON}: {exc}", file=sys.stderr)
        return 2

    if not isinstance(data, dict):
        print("FAIL: top-level JSON value must be an object", file=sys.stderr)
        return 2

    actual = list(data.keys())
    missing = [ch for ch in expected if ch not in data]
    extra = [ch for ch in actual if ch not in expected]
    if missing:
        errors.append(f"missing chapters: {', '.join(missing)}")
    if extra:
        errors.append(f"unexpected chapters: {', '.join(extra)}")

    seen_tex: dict[str, str] = {}
    seen_caption: dict[str, str] = {}
    total = 0

    for ch in expected:
        if ch not in data:
            continue
        entries = data[ch]
        if not isinstance(entries, list):
            errors.append(f"{ch}: value must be a list")
            continue
        if not (2 <= len(entries) <= 3):
            errors.append(f"{ch}: expected 2-3 entries, found {len(entries)}")
        for idx, entry in enumerate(entries, 1):
            total += 1
            where = f"{ch}[{idx}]"
            if not isinstance(entry, dict):
                errors.append(f"{where}: entry must be an object")
                continue
            if set(entry.keys()) != {"tex", "caption"}:
                errors.append(f"{where}: keys must be exactly tex and caption")
            tex = entry.get("tex")
            caption = entry.get("caption")
            errors.extend(check_formula(tex, where))
            errors.extend(check_caption(caption, where))

            if isinstance(tex, str):
                if tex in seen_tex:
                    errors.append(f"{where}: duplicate tex (also used by {seen_tex[tex]})")
                else:
                    seen_tex[tex] = where
            if isinstance(caption, str):
                if caption in seen_caption:
                    errors.append(
                        f"{where}: duplicate caption (also used by {seen_caption[caption]})"
                    )
                else:
                    seen_caption[caption] = where

    guide_errors, guide_stats = check_guides(expected)
    errors.extend(guide_errors)

    if errors:
        try:
            display_path = FORMULAS_JSON.relative_to(ROOT)
        except ValueError:
            display_path = FORMULAS_JSON
        print(f"FAIL: {len(errors)} problem(s) in {display_path} and data/site/guides.json")
        for err in errors:
            print(f"  - {err}")
        return 1

    counts = {ch: len(data[ch]) for ch in expected}
    triples = [ch for ch, n in counts.items() if n == 3]
    lengths = [len(entry["tex"]) for ch in expected for entry in data[ch]]
    captions = [len(entry["caption"]) for ch in expected for entry in data[ch]]
    print(
        f"OK: {len(expected)} chapters, {total} formulas "
        f"(2 each, 3 for {', '.join(triples) if triples else 'none'})"
    )
    print(
        f"    tex length max {max(lengths)}, caption length {min(captions)}..{max(captions)}; "
        "whitelist and bracket checks passed"
    )
    if guide_stats:
        print(
            f"    guides: {guide_stats['count']} chapters, length "
            f"{guide_stats['min']}..{guide_stats['max']}; tone checks passed"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
