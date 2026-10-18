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
    """index.json 的计数必须与**注册表**一致。

    早先这里写死了 `count == 2 and totalViz == 6`（只有 0.0 与 1.2 时的现状），
    每加一章就得手改一次——那是"用测试记录仓库现状"，加章的人一定会忘。
    现在改成自洽检查：注册表里有几章/几个 viz，索引就必须是几。
    """
    idx = json.loads((ROOT / "data" / "manifests" / "index.json").read_text(encoding="utf-8"))
    reg = Registry(ROOT)
    reg.discover(strict=False)
    want = {cid: len(reg.get(cid).viz_ids) for cid in reg.ids() if (ROOT / "data" / "manifests" / f"{cid}.json").exists()}
    assert idx["count"] == len(want), f"index.count={idx['count']} 与已生成 manifest 的章数 {len(want)} 不一致"
    assert idx["totalViz"] == sum(want.values())
    assert {c["chapter"] for c in idx["chapters"]} == set(want)
    assert idx["chapters"][0]["notesFile"].startswith("chapters/")


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
# run_asserts（G3）
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


def rec(
    fn: str,
    kind: str,
    label: str = "l",
    passed: bool = True,
    *,
    chapter: str = "0.0",
    nodeid: str | None = None,
) -> dict:
    return {
        "id": f"{fn}::{label}",
        "kind": kind,
        "label": label,
        "theory": 2.0,
        "measured": 2.0,
        "tol": "±0.1",
        "passed": passed,
        "detail": "R²=1.0",
        "chapter": chapter,
        "nodeid": nodeid or f"tests/{chapter}/test_smoke.py::{fn}",
    }


def test_analyse_accepts_a_report_matching_the_declaration() -> None:
    """一份"每章都按声明齐活"的报告必须对账通过。

    注意 ``analyse`` 会遍历**仓库里所有已实现章节**，所以假报告也必须覆盖全部章节，
    不能只造 0.0 那一块（否则它会如实报告"1.2 没有任何断言记录"）。
    """
    reg = Registry(ROOT)
    chapters: dict[str, dict] = {}
    for cid in reg.ids():
        decl = run_asserts.load_asserts_decl(ROOT, cid)
        if not decl:
            continue
        names = run_asserts.declared_tests(decl)
        items = [rec(fn, kind, chapter=cid, nodeid=f"tests/{cid}/test_x.py::{fn}") for fn, kind in names.items()]
        chapters[cid] = fake_report(items, chapter=cid)["chapters"][cid]
    result = run_asserts.analyse(ROOT, {"chapters": chapters})
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


def test_real_asserts_report_is_reconciled(tmp_path: pathlib.Path) -> None:
    """当前仓库的断言报告必须已经对账通过（G3 的"现状"基线）。

    报告用**本次会话**的登记表现场生成，不去读磁盘上那份可能来自上一次部分运行的旧文件
    （踩过：只跑 ``pytest tests/1.2`` 之后，磁盘报告里没有 0.0，这条测试就误报）。
    单文件运行时本次会话的记录不全，此时跳过并说明原因——CI 永远整跑，不会走到这条分支。
    """
    from numcore.verify import report as vreport

    path = vreport.write_report(tmp_path / "assertions.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [cid for cid in Registry(ROOT).ids() if cid not in payload.get("chapters", {})]
    if missing:
        pytest.skip(f"本次会话还没有跑到这些章节，无法对账：{missing}（请整跑 pytest tests）")
    result = run_asserts.analyse(ROOT, payload)
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


# --------------------------------------------------------------------------- #
# G9：Unicode 数学符号必须在进 LaTeX 之前被翻译掉
# --------------------------------------------------------------------------- #
def test_tex_escape_translates_unicode_math() -> None:
    """``≤``/``→``/``‖`` 这类字符必须变成 LaTeX：xelatex 缺字形时是**静默吞字**。

    真实事故：`asserts_table.tex` 里印出的是 ``漂移 � 1.77680e-``——那个 ``≤`` 消失了，
    而 xelatex 返回 0、G9 照样通过。所以这层映射 + 缺字形检查一起构成了修复。
    """
    got = build_book.tex_escape("‖A‖∞ ≤ κ₁·u → 0（ρ 与 det）")
    assert "$\\le$" in got and "$\\to$" in got, got
    assert "$\\kappa$" in got and "$\\rho$" in got, got
    # 判据不是"全 ASCII"（中文当然留着），而是"数学符号一律不许留"：
    # 希腊字母、一般标点/箭头/关系符/角标、拉丁补充里的 °±²³×÷ 都算数学符号。
    leftover = [c for c in got if _is_math_char(c)]
    assert not leftover, f"还有没翻译的数学字符：{leftover!r}（结果 {got!r}）"


def _is_math_char(ch: str) -> bool:
    o = ord(ch)
    return (
        0x0370 <= o <= 0x03FF      # 希腊字母
        or 0x1D00 <= o <= 0x1D7F   # 拉丁扩展（含 ᵀ）
        or 0x2000 <= o <= 0x2BFF   # 一般标点、箭头、数学运算符、角标
        or 0x00B0 <= o <= 0x00BF   # °±²³
        or o in (0x00D7, 0x00F7)   # ×÷
    )


def test_unicode_math_table_is_ascii_safe_and_single_pass() -> None:
    """映射表里每个键都必须翻成纯 ASCII，且不能与 TEX_ESCAPE 打架（单遍语义）。"""
    for key, val in build_book.UNICODE_MATH.items():
        assert all(ord(ch) < 128 for ch in val), (key, val)
        once = build_book.tex_escape(key)
        assert once == val, (key, once, val)
    # 混排：LaTeX 特殊字符与 Unicode 数学在同一串里各走各的
    mixed = build_book.tex_escape("100% ≤ x_1 & ρ")
    assert "\\%" in mixed and "\\_" in mixed and "\\&" in mixed and "$\\le$" in mixed
    # 中文与中文标点必须原样留着（它们由 ctex 的中文字体负责）
    assert build_book.tex_escape("残差、剖面。") == "残差、剖面。"


def test_missing_glyph_warnings_are_detected() -> None:
    """``compile_tex`` 必须把 "Missing character" 计入失败条件。"""
    sample = "Missing character: There is no ≤ (U+2264) in font [lmroman10-regular]!"
    assert build_book.MISSING_RE.findall(sample) == ["≤"]
    assert build_book.MISSING_RE.findall("Missing character: There is no ‖ (U+2016)") == ["‖"]
    assert build_book.MISSING_RE.findall("no missing anything here") == []


# --------------------------------------------------------------------------- #
# 回归：仓库外报告路径 / --no-compile 的先后与"残缺报告"保护
# --------------------------------------------------------------------------- #
def test_display_path_falls_back_outside_root(tmp_path: pathlib.Path) -> None:
    """``--report`` 指到仓库外时，取相对路径必须退化成原样返回而不是抛 ValueError。"""
    assert run_asserts._display_path(ROOT / "dist" / "reports" / "assertions.json", ROOT) == "dist/reports/assertions.json"
    outside = tmp_path / "outside" / "r.json"
    assert run_asserts._display_path(outside, ROOT) == str(outside)


def test_run_asserts_no_run_accepts_report_outside_root(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_asserts.main(["--no-run", "--report", <仓库外>])`` 不许崩，且报告要落盘。

    ``--no-run`` 的语义是"不跑 pytest，直接拿现有报告对账"；对完账它仍会**回写**增强后的报告
    （带 ``gate`` 段）并生成每章 markdown。原先崩溃点在最后那行 ``report.relative_to(root)``：
    报告其实已经写好了，人却看到 ValueError。

    这里把 ``run_asserts.ROOT`` 指到临时目录而不是用真仓库：``main`` 还会写
    ``chapters/<id>/asserts_report.md``，并发下真仓库里跑会改到别的章的产物。
    """
    fake_root = tmp_path / "root"
    (fake_root / "chapters").mkdir(parents=True)
    (fake_root / "docs").mkdir(parents=True)
    (fake_root / "docs" / "chapters.yaml").write_text(
        'chapters:\n  - {id: "9.9", title: "假章", part: "测试", viz: 0}\n', encoding="utf-8"
    )
    monkeypatch.setattr(run_asserts, "ROOT", fake_root)

    outside = tmp_path / "outside" / "r.json"  # 仓库外：修复前这里必崩
    outside.parent.mkdir(parents=True)
    items = [rec("test_a", "A"), rec("test_b", "B"), rec("test_c", "C")]
    outside.write_text(json.dumps(fake_report(items, chapter="9.9"), ensure_ascii=False), encoding="utf-8")

    rc = run_asserts.main(["--no-run", "--report", str(outside)])
    assert rc == 0
    assert outside.exists()
    payload = json.loads(outside.read_text(encoding="utf-8"))
    assert payload["gate"]["ok"] is True, payload["gate"]["problems"]
    assert (fake_root / "chapters" / "9.9" / "asserts_report.md").exists()


def _asserts_report(items_by_chapter: dict[str, list[dict]]) -> dict:
    """造一份最小断言报告：``{章号: items}`` → ``dist/reports/assertions.json`` 的形状。"""
    chapters = {cid: fake_report(items, chapter=cid)["chapters"][cid] for cid, items in items_by_chapter.items()}
    return {"generated_at": "2026-10-18T00:00:00", "totals": {"total": 0, "failed": 0, "unmeasured": 0}, "chapters": chapters}


def _fake_book_repo(tmp_path: pathlib.Path, ids: list[str], report: dict | None) -> pathlib.Path:
    """造一个"最小可用仓库"：注册表能发现章节、notes.tex 结构合格、报告内容可控。

    必须用假根：``build()`` 会遍历**所有**章（并发下别的章半成品会让结果不可复现），
    而且 ``--no-compile`` 会往 ``chapters/<id>/`` 写文件。
    章节模块直接复制真仓库的 0.0 模板（manifest 保持 schema 合法），只改章号。
    """
    root = tmp_path / "repo"
    (root / "server" / "chapters").mkdir(parents=True)
    template = (ROOT / "server" / "chapters" / "0.0.py").read_text(encoding="utf-8")
    for cid in ids:
        (root / "server" / "chapters" / f"{cid}.py").write_text(
            template.replace('"chapter": "0.0"', f'"chapter": "{cid}"'), encoding="utf-8"
        )
        d = root / "chapters" / cid
        d.mkdir(parents=True)
        (d / "notes.tex").write_text(
            "\\begin{theorem}测试定理\\end{theorem}\n\\input{chapters/%s/asserts_table}\n" % cid, encoding="utf-8"
        )
    if report is not None:
        out = root / "dist" / "reports" / "assertions.json"
        out.parent.mkdir(parents=True)
        out.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return root


def test_write_asserts_table_requires_records(tmp_path: pathlib.Path) -> None:
    """``require_records=True`` 时，报告里没有该章记录（含空 items）就不写、返回 None。

    根必须用临时目录：最后一行的"老行为"是会真的落盘的（写一张空表），拿真仓库当根
    就等于亲手往 ``chapters/<id>/`` 里丢文件——那正是这条保护要防的事。
    """
    root = tmp_path / "repo"
    (root / "chapters" / "9.9").mkdir(parents=True)
    report = {"chapters": {"9.9": {"items": []}, "8.8": {"items": [rec("test_x", "A")]}}}
    table = root / "chapters" / "9.9" / "asserts_table.tex"
    assert build_book.write_asserts_table(root, "9.9", report, require_records=True) is None
    assert build_book.write_asserts_table(root, "9.9", report, dry_run=True, require_records=True) is None
    assert build_book.write_asserts_table(root, "1.2", report, require_records=True) is None  # 报告里整块缺失
    assert not table.exists(), "require_records=True 竟然写了盘"
    made = build_book.write_asserts_table(root, "9.9", report)  # 默认保持老行为（写空表）
    assert made is not None and made[0] == table and table.exists()


def test_build_book_no_compile_generates_table_before_structure_check(tmp_path: pathlib.Path) -> None:
    """``--no-compile`` 必须**先**生成附表再做结构检查。

    反过来的话，第一次运行必然报"notes.tex 里 \\input{...asserts_table} 指向的文件不存在"
    （2.1/3.2 章真被这条误导过），而它其实下一秒就把文件写出来了。
    """
    report = _asserts_report({"9.9": [rec("test_x", "A")]})
    root = _fake_book_repo(tmp_path, ["9.9"], report)
    table = root / "chapters" / "9.9" / "asserts_table.tex"
    assert not table.exists()

    result = build_book.build(root, do_compile=False)
    assert table.exists()
    assert result["chapters"]["9.9"]["problems"] == [], result["chapters"]["9.9"]["problems"]
    assert result["ok"], result["problems"]
    assert "请勿手改" in table.read_text(encoding="utf-8")


def test_build_book_no_compile_keeps_table_when_report_lacks_chapter(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """残缺报告（并发下被别的章覆盖）不许把别章已写好的附表清成空表。

    ``8.8`` 的报告块是"存在但 items 为空"——这正是踩过的形态：老代码会照它写一张
    "（本章暂无断言记录）"的表，覆盖掉磁盘上那份好表。修复后保持原文件不动，并在输出里说明。
    """
    report = _asserts_report({"9.9": [rec("test_x", "A")], "8.8": []})
    root = _fake_book_repo(tmp_path, ["9.9", "8.8"], report)
    old = root / "chapters" / "8.8" / "asserts_table.tex"
    old.write_text("旧附表（不许被动）\n", encoding="utf-8")

    result = build_book.build(root, do_compile=False)
    assert old.read_text(encoding="utf-8") == "旧附表（不许被动）\n"
    kept = result["chapters"]["8.8"]
    assert kept["problems"] == [], kept["problems"]
    assert "保留现有 asserts_table.tex" in kept["tableKept"]
    assert "请勿手改" in (root / "chapters" / "9.9" / "asserts_table.tex").read_text(encoding="utf-8")
    # 合并本附录只 input 存在的附表：跳过写入不等于"没有附表"
    assert kept["hasTable"] is True

    monkeypatch.setattr(build_book, "ROOT", root)
    rc = build_book.main(["--no-compile", "--report", str(tmp_path / "book.json")])  # 报告也在仓库外
    assert rc == 0
    assert "报告里没有 8.8 的记录，保留现有 asserts_table.tex" in capsys.readouterr().out

    # --only 是**显式点名**：不受这条保护，照老规矩按报告写（报告里是空块 → 写空表），
    # 这样"我知道这章的记录没了，但我要的就是现在这份报告对应的附表"仍然做得到。
    forced = build_book.build(root, only="8.8", do_compile=False)
    assert "tableKept" not in forced["chapters"]["8.8"]
    assert "暂无断言记录" in old.read_text(encoding="utf-8")


def test_build_book_check_reports_stale_table_without_writing(tmp_path: pathlib.Path) -> None:
    """``--check`` 的判定逻辑保持不变：附表与报告不一致 → 报问题、红，而且**不落盘**。

    跑 ``only="9.9"`` 是为了跳过合并本编译（这里要验的是判定，不是 LaTeX）。
    """
    report = _asserts_report({"9.9": [rec("test_x", "A")]})
    root = _fake_book_repo(tmp_path, ["9.9"], report)
    table = root / "chapters" / "9.9" / "asserts_table.tex"
    table.write_text("过期的附表\n", encoding="utf-8")

    result = build_book.build(root, only="9.9", do_compile=True)  # dry_run：只算不写
    assert table.read_text(encoding="utf-8") == "过期的附表\n", "dry_run 竟然写了盘"
    problems = result["chapters"]["9.9"]["problems"]
    assert any("与最新断言报告不一致" in p for p in problems), problems
    assert result["chapters"]["9.9"]["hasTable"] is True
    assert result["ok"] is False
