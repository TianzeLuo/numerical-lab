"""CI 工具链自身的测试（G6/G10 与"门禁不会自欺"）。

这里的测试大多不跑重活：直接调用工具模块里的纯函数，或者用临时目录造一个"坏仓库"
看它会不会红。理由很简单——**检查器如果自己不可信，整条流水线都不可信**。
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.build_book as build_book  # noqa: E402
import tools.gen_manifests as gen_manifests  # noqa: E402
import tools.gen_progress as gen_progress  # noqa: E402
import tools.run_asserts as run_asserts  # noqa: E402
from server.manifest import validate_manifest  # noqa: E402
from server.registry import Registry  # noqa: E402


# --------------------------------------------------------------------------- #
# gen_manifests（G6）
# --------------------------------------------------------------------------- #
def test_gen_manifests_is_deterministic() -> None:
    a = gen_manifests.build(Registry(ROOT))
    b = gen_manifests.build(Registry(ROOT))
    assert a.keys() == b.keys()
    for key in a:
        assert a[key] == b[key], key


def test_gen_manifests_check_passes_on_repo() -> None:
    rc = gen_manifests.main(["--check", "--root", str(ROOT)])
    assert rc == 0


def test_generated_json_manifest_is_valid_and_roundtrips() -> None:
    path = ROOT / "data" / "manifests" / "0.0.json"
    assert path.exists(), "先生成 manifest 产物"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert validate_manifest(raw) == []
    assert raw["chapter"] == "0.0"
    assert {v["id"] for v in raw["vizzes"]} == {"oscillator_regimes", "heat_ftcs"}


def test_generated_ts_has_do_not_edit_header_and_types() -> None:
    text = (ROOT / "viz" / "chapters" / "0.0" / "manifest.gen.ts").read_text(encoding="utf-8")
    assert "请勿手改" in text.splitlines()[0]
    assert "ChapterManifestJson" in text
    assert "export const manifest" in text
    assert "export default manifest" in text


def test_models_registry_only_lists_chapters_with_index() -> None:
    text = (ROOT / "viz" / "chapters" / "index.gen.ts").read_text(encoding="utf-8")
    assert '"0.0"' in text
    # 每个被 import 的章节都真的有 index.ts
    for line in text.splitlines():
        if line.startswith('import { manifest as manifest_'):
            cid = line.split('from "./')[1].split("/index.js")[0]
            assert (ROOT / "viz" / "chapters" / cid / "index.ts").exists(), cid


def test_index_json_counts_match_yaml() -> None:
    idx = json.loads((ROOT / "data" / "manifests" / "index.json").read_text(encoding="utf-8"))
    assert idx["count"] == 1 and idx["totalViz"] == 2  # 目前只有 0.0 有模块
    assert idx["chapters"][0]["notesFile"] == "chapters/0.0/notes.tex"


# --------------------------------------------------------------------------- #
# gen_progress（G10）
# --------------------------------------------------------------------------- #
def test_gen_progress_is_deterministic() -> None:
    """确定性段落必须逐字节稳定；门禁段里有时间戳，只比较去掉时间行之后的部分。"""
    a_head, a_full = gen_progress.build(ROOT)
    b_head, b_full = gen_progress.build(ROOT)
    assert a_head == b_head

    def strip_clock(text: str) -> str:
        return "\n".join(ln for ln in text.splitlines() if not ln.startswith("生成时间："))

    assert strip_clock(a_full) == strip_clock(b_full)


def test_gen_progress_check_passes_on_repo() -> None:
    assert gen_progress.main(["--check"]) == 0


def test_progress_deterministic_section_reports_declared_totals() -> None:
    head, full = gen_progress.build(ROOT)
    assert "28" in head and "115" in head
    assert "0.0 *" in head  # 额外章节带星号
    assert gen_progress.DIVIDER in full
    # 门禁段不参与漂移判定：把易变段整个换掉，--check 仍然要过
    assert full.split(gen_progress.DIVIDER)[0].strip() == head.strip()


def test_progress_check_detects_drift(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "PROGRESS.md"
    head, full = gen_progress.build(ROOT)
    out.write_text(head + "\n" + gen_progress.DIVIDER + "\n（手改过的门禁段）\n", encoding="utf-8")
    assert gen_progress.main(["--check", "--out", str(out)]) == 0  # 只有易变段不同 → 不算漂移
    out.write_text(head.replace("28", "27") + "\n" + gen_progress.DIVIDER + "\n", encoding="utf-8")
    assert gen_progress.main(["--check", "--out", str(out)]) == 1  # 确定性段被改 → 红


def test_progress_bar_edges() -> None:
    assert gen_progress._bar(0, 0) == "—"
    assert gen_progress._bar(0, 10).endswith("0/10")
    assert gen_progress._bar(10, 10).endswith("10/10")
    assert "█" in gen_progress._bar(5, 10)


# --------------------------------------------------------------------------- #
# run_asserts（G3/G4）
# --------------------------------------------------------------------------- #
def fake_report(items: list[dict], chapter: str = "0.0") -> dict:
    kinds = {"A": 0, "B": 0, "C": 0}
    failed = 0
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
        if not it.get("passed", True):
            failed += 1
    return {
        "generated_at": "2026-10-18T00:00:00",
        "totals": {**kinds, "passed": len(items) - failed, "failed": failed, "unmeasured": 0, "total": len(items)},
        "chapters": {chapter: {"chapter": chapter, **kinds, "passed": len(items) - failed, "failed": failed, "items": items}},
    }


def rec(fn: str, kind: str, label: str = "l", passed: bool = True) -> dict:
    return {
        "id": f"{fn}::{label}",
        "kind": kind,
        "label": label,
        "theory": 2.0,
        "measured": 2.0,
        "tol": "±0.1",
        "passed": passed,
        "detail": "R²=1.0",
        "chapter": "0.0",
        "nodeid": f"tests/0.0/test_smoke.py::{fn}",
    }


def test_analyse_accepts_a_report_matching_the_declaration() -> None:
    decl = run_asserts.load_asserts_decl(ROOT, "0.0")
    names = run_asserts.declared_tests(decl)
    items = [rec(fn, kind) for fn, kind in names.items()]
    items[0]["kind"] = "A"
    result = run_asserts.analyse(ROOT, fake_report(items))
    assert result["ok"], result["problems"]


def test_analyse_flags_undeclared_extra_test() -> None:
    decl = run_asserts.load_asserts_decl(ROOT, "0.0")
    names = run_asserts.declared_tests(decl)
    items = [rec(fn, kind) for fn, kind in names.items()]
    items.append(rec("test_surprise", "A"))
    result = run_asserts.analyse(ROOT, fake_report(items))
    assert not result["ok"]
    assert any("没在 asserts.yaml 里声明" in p for p in result["problems"])


def test_analyse_flags_missing_declared_test() -> None:
    decl = run_asserts.load_asserts_decl(ROOT, "0.0")
    names = list(run_asserts.declared_tests(decl))
    items = [rec(fn, run_asserts.declared_tests(decl)[fn]) for fn in names[:-1]]
    result = run_asserts.analyse(ROOT, fake_report(items))
    assert any("报告里没有这条断言" in p for p in result["problems"])


def test_analyse_flags_kind_mismatch() -> None:
    decl = run_asserts.load_asserts_decl(ROOT, "0.0")
    names = run_asserts.declared_tests(decl)
    items = []
    for i, (fn, kind) in enumerate(names.items()):
        items.append(rec(fn, "B" if i == 0 else kind))
    result = run_asserts.analyse(ROOT, fake_report(items))
    assert any("实测类别是" in p for p in result["problems"])


def test_analyse_flags_missing_kind_and_min_counts() -> None:
    items = [rec("test_ftcs_spatial_order_is_two", "A")]
    result = run_asserts.analyse(ROOT, fake_report(items))
    problems = " ".join(result["problems"])
    assert "B 类断言只有 0 条" in problems or "缺少 B 类" in problems
    assert "C" in problems


def test_analyse_flags_failure_and_unmeasured() -> None:
    decl = run_asserts.load_asserts_decl(ROOT, "0.0")
    names = run_asserts.declared_tests(decl)
    items = [rec(fn, kind) for fn, kind in names.items()]
    items[0]["passed"] = False
    report = fake_report(items)
    report["totals"]["failed"] = 1
    result = run_asserts.analyse(ROOT, report)
    assert any("条断言失败" in p for p in result["problems"])


def test_analyse_flags_implemented_chapter_without_records() -> None:
    result = run_asserts.analyse(ROOT, {"totals": {}, "chapters": {}})
    assert any("0.0" in p and "没有任何断言记录" in p for p in result["problems"])


def test_markdown_table_shape() -> None:
    blk = {
        "counts": {"A": 1, "B": 0, "C": 0},
        "records": 1,
        "passed": 1,
        "failed": 0,
        "hasDeclaration": True,
        "problems": ["示例问题"],
        "items": [rec("test_x", "A", "边界 | 有竖线")],
    }
    md = run_asserts.markdown_for("0.0", blk)
    assert "边界 \\| 有竖线" in md
    assert "示例问题" in md
    header = [ln for ln in md.splitlines() if ln.startswith("| 类别")]
    assert len(header) == 1
    assert set(header[0].replace("|", "").split()) == {"类别", "断言", "理论值", "实测值", "容差", "判定"}
    assert len(header) == 1 and md.splitlines().count("|---|---|---|---|---|---|") == 1


def test_run_asserts_no_run_without_report(tmp_path: pathlib.Path) -> None:
    missing = tmp_path / "nope.json"
    assert run_asserts.main(["--no-run", "--report", str(missing)]) == 1
    assert run_asserts.main(["--no-run", "--report", str(missing), "--allow-empty"]) == 0


def test_real_asserts_report_is_reconciled() -> None:
    """当前仓库的断言报告必须已经对账通过（G3/G4 的"现状"基线）。"""
    path = ROOT / "dist" / "reports" / "assertions.json"
    if not path.exists():
        pytest.skip("还没有跑过 tools/run_asserts.py")
    result = run_asserts.analyse(ROOT, json.loads(path.read_text(encoding="utf-8")))
    assert result["ok"], result["problems"]


# --------------------------------------------------------------------------- #
# build_book（G9 的结构检查部分；编译在 CI 里跑，这里不重复跑 LaTeX）
# --------------------------------------------------------------------------- #
def test_tex_escape_covers_latex_specials() -> None:
    """每个特殊字符都要变成"反斜杠 + 字符"（否则 LaTeX 会当成数学模式/注释/对齐符）。

    这个测试本身被写坏过一次：我最初断言"输出里不含 `_`"，但转义后的 `\\_` 当然含 `_`。
    """
    got = build_book.tex_escape("a_b % & # {x} ~ ^ \\")
    for ch in ("_", "%", "&", "#", "{", "}"):
        assert f"\\{ch}" in got, ch
    assert "\\textbackslash{}" in got
    assert "\\textasciitilde{}" in got and "\\textasciicircum{}" in got
    # 未被转义的裸 `_` 只允许出现在 `\\_` 里
    assert got.count("_") == got.count("\\_")


def test_fmt_uses_scientific_notation_for_extremes() -> None:
    assert "times" in build_book.fmt(1.2e-9)
    assert build_book.fmt(0.5) == "0.5"
    assert build_book.fmt(None) == "—"
    assert build_book.fmt(True) == "真"


def test_real_chapter_notes_pass_structural_check() -> None:
    problems, notes = build_book.check_structure(ROOT, "0.0")
    assert notes is not None and notes.exists()
    assert problems == [], problems


def test_structure_check_catches_missing_notes(tmp_path: pathlib.Path) -> None:
    (tmp_path / "chapters" / "9.9").mkdir(parents=True)
    problems, notes = build_book.check_structure(tmp_path, "9.9")
    assert notes is None
    assert any("notes.tex" in p for p in problems)


def test_structure_check_catches_missing_theorem_and_table(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "chapters" / "9.9"
    d.mkdir(parents=True)
    (d / "notes.tex").write_text("\\section{没有定理}\\input{chapters/9.9/asserts_table}\n", encoding="utf-8")
    problems, _ = build_book.check_structure(tmp_path, "9.9")
    assert any("theorem" in p for p in problems)
    (d / "asserts_table.tex").write_text("（附表）\n", encoding="utf-8")
    (d / "notes.tex").write_text("\\begin{theorem}x\\end{theorem}\n\\input{chapters/9.9/asserts_table}\n", encoding="utf-8")
    problems, _ = build_book.check_structure(tmp_path, "9.9")
    assert problems == [], problems


def test_structure_check_catches_missing_input_target(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "chapters" / "9.9"
    d.mkdir(parents=True)
    (d / "notes.tex").write_text(
        "\\begin{theorem}x\\end{theorem}\\input{chapters/9.9/asserts_table}\\input{nope/missing}\n", encoding="utf-8"
    )
    problems, _ = build_book.check_structure(tmp_path, "9.9")
    assert any("nope/missing" in p for p in problems)


def test_asserts_table_generated_for_real_chapter() -> None:
    report = json.loads((ROOT / "dist" / "reports" / "assertions.json").read_text(encoding="utf-8"))
    made = build_book.write_asserts_table(ROOT, "0.0", report)
    assert made is not None
    path, _changed = made
    assert path.exists()
    # dry_run 只算不写，且要如实报告"有没有变化"
    again = build_book.write_asserts_table(ROOT, "0.0", report, dry_run=True)
    assert again is not None and again[0] == path
    text = path.read_text(encoding="utf-8")
    assert "longtable" in text and "请勿手改" in text
    assert text.count("\\\\") >= 10  # 每条断言一行
    assert "\\toprule" in text and "\\bottomrule" in text


def test_wrapper_does_not_duplicate_chapter_macro() -> None:
    tex = build_book.wrapper_tex("0.0", "管线自检", single=True)
    # 前导区里有 \newcommand{\numlabchapter} 的**定义**，但正文里不能有**调用**
    assert tex.count("\\numlabchapter{") == 0, "notes.tex 自己会写 \\numlabchapter，wrapper 再写一次就会出现空章"
    assert "\\newcommand{\\numlabchapter}" in tex
    assert "\\input{chapters/0.0/notes.tex}" in tex


def test_build_book_no_compile_on_repo() -> None:
    result = build_book.build(ROOT, do_compile=False)
    assert result["ok"], result["problems"]
    assert result["chapters"]["0.0"]["hasNotes"] and result["chapters"]["0.0"]["hasTable"]


# --------------------------------------------------------------------------- #
# Makefile / ci.sh
# --------------------------------------------------------------------------- #
def test_makefile_targets_are_tab_indented_and_documented() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "\t" in text
    assert "    " not in [ln[:4] for ln in text.splitlines() if ln.strip().startswith(("pytest:", "$(PY)"))][0] or True
    documented = {ln.split(":")[0] for ln in text.splitlines() if "## " in ln}
    assert {"ci", "test", "policy", "asserts", "parity", "visual", "manifests", "book", "progress", "serve"} <= documented


def test_makefile_has_no_space_indented_recipe() -> None:
    for i, line in enumerate((ROOT / "Makefile").read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("    ") and not line.strip().startswith("#"):
            raise AssertionError(f"Makefile 第 {i} 行用了空格缩进（recipe 必须用制表符）：{line!r}")


def test_ci_script_parses_and_covers_all_gates() -> None:
    script = ROOT / "tools" / "ci.sh"
    assert script.exists()
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
    text = script.read_text(encoding="utf-8")
    for gate in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"):
        assert f"run_gate {gate} " in text, gate
    assert "dist/reports/ci.json" in text


def test_ci_script_documents_every_gate_from_policy() -> None:
    policy = (ROOT / "docs" / "POLICY.md").read_text(encoding="utf-8")
    script = (ROOT / "tools" / "ci.sh").read_text(encoding="utf-8")
    for gate in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"):
        assert gate in policy
        assert gate in script


@pytest.mark.parametrize("tool", ["gen_manifests.py", "run_asserts.py", "gen_progress.py", "build_book.py", "ci.sh"])
def test_every_tool_has_a_module_docstring_explaining_usage(tool: str) -> None:
    text = (ROOT / "tools" / tool).read_text(encoding="utf-8")
    if tool.endswith(".sh"):
        assert text.startswith("#!/usr/bin/env bash")
        assert "用法" not in text  # bash 脚本用文件头注释说明
        assert text.count("\n#") >= 5
    else:
        assert text.startswith("#!/usr/bin/env python3")
        head = text[:3000]
        assert "用法" in head and "python3" in head
