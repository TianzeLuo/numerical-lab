"""章节注册表的测试。

关注三件事：
1. ``docs/chapters.yaml`` 是"计划"（28 章 / 115 可视化），注册表必须如实对照；
2. 只发现合法的章节模块（点号文件名、私有文件、坏 manifest 各有确定行为）；
3. ``progress()/summary()`` 的数字不能自己骗自己——"声明了 6 个可视化但只实现 2 个"要能看出来。
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from server.manifest import ManifestError
from server.registry import (
    MIN_PITFALL_DOC_CHARS,
    ChapterEntry,
    ChapterProgress,
    Registry,
    RegistryError,
    default_registry,
    load_chapter_yaml,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]

GOOD_MANIFEST = '''
MANIFEST = {
    "chapter": "%(cid)s",
    "title": "测试章 %(cid)s",
    "summary": "s",
    "vizzes": [
        {
            "id": "demo",
            "title": "演示",
            "dataSource": "backend",
            "params": [{"name": "n", "label": "n", "type": "int", "min": 2, "max": 8, "default": 4,
                        "critical": [], "criticalNote": ""}],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [{"id": "main", "autoscaled": False}],
            "pitfalls": "把纵轴设成自适应会让发散看起来像一条平线，必须固定纵轴；同时误差轴要用对数刻度，"
                        "否则 1e-16 与 1e-3 看起来一样深，读者会以为精度还很高。",
            "theory": ["误差 O(h^2)"],
        }
    ],
}


def compute(viz_id, params, t):
    raise NotImplementedError
'''


@pytest.fixture()
def rewrite_viz_count():
    """就地改写假 chapters.yaml 里某章的 viz 声明数（用于测"实现数 vs 声明数"）。"""

    def _rewrite(root: pathlib.Path, cid: str, viz: int) -> None:
        path = root / "docs" / "chapters.yaml"
        text = path.read_text(encoding="utf-8")
        old = f'id: "{cid}"'
        idx = text.index(old)
        head, tail = text[:idx], text[idx:]
        tail = tail.replace("viz: 2", f"viz: {viz}", 1)
        path.write_text(head + tail, encoding="utf-8")

    return _rewrite


@pytest.fixture()
def fake_root(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "server" / "chapters").mkdir(parents=True)
    (tmp_path / "viz" / "chapters").mkdir(parents=True)
    (tmp_path / "chapters").mkdir()
    (tmp_path / "numcore").mkdir()
    (tmp_path / "numcore" / "fp.py").write_text("# 假的 fp 模块，只为测试 modules 存在性\n", encoding="utf-8")
    (tmp_path / "docs" / "chapters.yaml").write_text(
        textwrap.dedent(
            """
            chapters:
              - id: "0.0"
                title: 自检
                part: ""
                viz: 2
                modules: [numcore/fp.py]
              - id: "1.1"
                title: 浮点
                part: I
                viz: 4
                modules: [numcore/fp.py, numcore/missing_thing.py]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def write_chapter(root: pathlib.Path, cid: str, text: str) -> pathlib.Path:
    path = root / "server" / "chapters" / f"{cid}.py"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# chapters.yaml
# --------------------------------------------------------------------------- #
def test_load_chapter_yaml_reads_all_fields() -> None:
    data = load_chapter_yaml(ROOT)
    assert len(data) == 28, f"docs/chapters.yaml 应当声明 28 章，实际 {len(data)}"
    assert "1.1" in data
    # 0.0 是 M0 模板章：它存在于 server/chapters/，但**不在** chapters.yaml 里，
    # 因为它不属于那 28 章 / 115 个可视化（registry 会把它单列为"额外章节"）。
    assert "0.0" not in data
    total = sum(int(v.get("viz", 0) or 0) for v in data.values())
    assert total == 115, f"全书可视化总数应当是 115，实际 {total}"
    assert data["6.1"]["viz"] == 4 and data["6.2"]["viz"] == 3
    for cid, info in data.items():
        assert info.get("part") is not None or cid == "0.0"


def test_load_chapter_yaml_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_chapter_yaml(tmp_path)


def test_yaml_and_markdown_agree_on_counts() -> None:
    """docs/CHAPTERS.md 是给人读的，yaml 是给机器读的；两者的可视化总数必须一致。"""
    yaml_total = sum(int(v.get("viz", 0) or 0) for v in load_chapter_yaml(ROOT).values())
    md = (ROOT / "docs" / "CHAPTERS.md").read_text(encoding="utf-8")
    assert "115" in md, "CHAPTERS.md 里应当写明总数 115"
    assert yaml_total == 115


# --------------------------------------------------------------------------- #
# 发现
# --------------------------------------------------------------------------- #
def test_discover_finds_real_chapters() -> None:
    reg = Registry(ROOT)
    found = reg.discover()
    assert isinstance(found, dict) and "0.0" in found
    ids = reg.ids()
    # 不钉死"仓库当前有哪几章"（每加一章就得手改的钉子），只要求：
    #  * 模板章 0.0 在（它是五件套的样板）；
    #  * 顺序按 (部, 章) 升序，且没有重复。
    assert "0.0" in ids
    assert len(set(ids)) == len(ids)
    assert sorted(ids, key=lambda c: (int(c.split(".")[0]), int(c.split(".")[1]))) == ids
    entry = reg.get("0.0")
    assert isinstance(entry, ChapterEntry)
    assert entry.viz_ids == ("oscillator_regimes", "heat_ftcs")
    assert entry.title
    assert reg.viz("0.0", "heat_ftcs").id == "heat_ftcs"
    ch12 = reg.get("1.2")
    assert ch12.viz_ids == ("lu_heatmap", "unit_ball", "spy_ordering", "pivot_strategy")
    # 1.2 章此刻应当是"五件套齐全"的：模块在、viz 四个、笔记/坑/断言声明都在
    row = next(r for r in reg.progress() if r.id == "1.2")
    assert row.has_module and row.manifest_ok
    assert row.viz_declared == 4 and row.viz_implemented == 4
    assert row.pitfall_chars >= 800


def test_discover_ignores_private_and_bad_names(fake_root: pathlib.Path) -> None:
    write_chapter(fake_root, "0.0", GOOD_MANIFEST % {"cid": "0.0"})
    (fake_root / "server" / "chapters" / "_helper.py").write_text("X = 1\n", encoding="utf-8")
    (fake_root / "server" / "chapters" / "notes.txt").write_text("x\n", encoding="utf-8")
    (fake_root / "server" / "chapters" / "__init__.py").write_text("", encoding="utf-8")
    reg = Registry(fake_root)
    assert reg.ids() == ["0.0"]


def test_discover_strict_raises_with_all_problems(fake_root: pathlib.Path) -> None:
    write_chapter(fake_root, "0.0", "MANIFEST = {'chapter': '0.0'}\n")
    reg = Registry(fake_root)
    with pytest.raises(RegistryError) as ei:
        reg.discover(strict=True)
    msg = str(ei.value)
    assert "0.0" in msg and "title" in msg
    assert reg.discover(strict=False) == {}


def test_discover_import_error_is_reported(fake_root: pathlib.Path) -> None:
    write_chapter(fake_root, "0.0", "import nonexistent_module_xyz\n")
    reg = Registry(fake_root)
    with pytest.raises(RegistryError) as ei:
        reg.discover(strict=True)
    assert "nonexistent_module_xyz" in str(ei.value)


def test_missing_manifest_is_an_error(fake_root: pathlib.Path) -> None:
    write_chapter(fake_root, "0.0", "compute = lambda *a: None\n")
    with pytest.raises(RegistryError, match="MANIFEST"):
        Registry(fake_root).discover(strict=True)


def test_no_chapters_at_all(fake_root: pathlib.Path) -> None:
    with pytest.raises(RegistryError, match="没有发现任何章节"):
        Registry(fake_root).discover()


def test_registry_discover_is_idempotent() -> None:
    reg = Registry(ROOT)
    first = reg.discover()
    again = reg.discover()
    # 模块对象每次重新 exec 都是新对象，所以比"身份"没意义；比可序列化的内容
    assert [e.manifest.to_json() for e in first.values()] == [e.manifest.to_json() for e in again.values()]
    assert reg.ids() == ["0.0", "1.2"]
    assert [e.id for e in reg.all()] == reg.ids()


def test_get_and_viz_errors() -> None:
    reg = Registry(ROOT)
    with pytest.raises(KeyError, match="没有章节"):
        reg.get("9.9")
    entry = reg.get("0.0")
    with pytest.raises(KeyError, match="没有可视化"):
        entry.compute("nope", {}, 0.0)


def test_compute_without_module_raises(fake_root: pathlib.Path) -> None:
    write_chapter(fake_root, "0.0", GOOD_MANIFEST % {"cid": "0.0"})
    reg = Registry(fake_root)
    reg.discover()
    entry = reg.get("0.0")
    entry.module = None
    with pytest.raises(RegistryError, match="compute"):
        entry.compute("demo", {}, 0.0)


def test_paths_are_where_the_rest_of_the_repo_expects() -> None:
    reg = Registry(ROOT)
    assert reg.viz_file("0.0", "heat_ftcs") == ROOT / "viz" / "chapters" / "0.0" / "heat_ftcs.viz.ts"
    assert reg.model_file("0.0", "heat_ftcs") == ROOT / "viz" / "chapters" / "0.0" / "heat_ftcs.model.ts"
    assert reg.tests_dir("1.2") == ROOT / "tests" / "1.2"
    assert reg.notes_file("1.2") == ROOT / "chapters" / "1.2" / "notes.tex"


def test_default_registry_is_shared() -> None:
    assert default_registry() is default_registry()
    assert default_registry().root == ROOT


def test_iter_viz_yields_every_pair() -> None:
    reg = Registry(ROOT)
    pairs = list(reg.iter_viz())
    assert ("0.0", "oscillator_regimes") in pairs
    assert ("1.2", "spy_ordering") in pairs
    assert len(pairs) == 6  # 0.0 两个 + 1.2 四个


# --------------------------------------------------------------------------- #
# 进度
# --------------------------------------------------------------------------- #
def test_progress_covers_every_declared_chapter(fake_root: pathlib.Path) -> None:
    write_chapter(fake_root, "0.0", GOOD_MANIFEST % {"cid": "0.0"})
    rows = Registry(fake_root).progress()
    assert [r.id for r in rows] == ["0.0", "1.1"]
    zero, one = rows
    assert zero.has_module and zero.manifest_ok and zero.viz_declared == 2
    assert zero.viz_implemented == 0  # viz/chapters/0.0/demo.viz.ts 不存在
    assert not zero.complete
    assert not one.has_module
    assert one.viz_declared == 4
    assert one.modules_missing == ["numcore/missing_thing.py"]
    assert one.modules_present == ["numcore/fp.py"]


def test_progress_counts_implemented_viz(fake_root: pathlib.Path, rewrite_viz_count) -> None:
    write_chapter(fake_root, "0.0", GOOD_MANIFEST % {"cid": "0.0"})
    rewrite_viz_count(fake_root, "0.0", 1)
    (fake_root / "viz" / "chapters" / "0.0").mkdir(parents=True)
    (fake_root / "viz" / "chapters" / "0.0" / "demo.viz.ts").write_text("export {};\n", encoding="utf-8")
    (fake_root / "tests" / "0.0").mkdir(parents=True)
    (fake_root / "tests" / "0.0" / "test_x.py").write_text("def test_x(): pass\n", encoding="utf-8")
    (fake_root / "chapters" / "0.0").mkdir(parents=True)
    (fake_root / "chapters" / "0.0" / "notes.tex").write_text("\\section{x}\n", encoding="utf-8")
    (fake_root / "chapters" / "0.0" / "pitfalls.md").write_text("坑" * MIN_PITFALL_DOC_CHARS, encoding="utf-8")
    (fake_root / "chapters" / "0.0" / "asserts.yaml").write_text("tests: []\n", encoding="utf-8")
    row = Registry(fake_root).progress()[0]
    assert row.viz_implemented == 1
    assert row.has_tests and row.has_notes
    assert row.complete, row.modules_missing
    assert row.to_json()["complete"] is True
    assert row.has_pitfalls and row.pitfall_chars == MIN_PITFALL_DOC_CHARS and row.has_asserts_decl


def test_progress_incomplete_when_pitfalls_or_asserts_missing(fake_root: pathlib.Path, rewrite_viz_count) -> None:
    """POLICY §2：防误导分析（≥800 字）与断言声明也是"五件套"的一部分。"""
    write_chapter(fake_root, "0.0", GOOD_MANIFEST % {"cid": "0.0"})
    rewrite_viz_count(fake_root, "0.0", 0)
    (fake_root / "tests" / "0.0").mkdir(parents=True)
    (fake_root / "tests" / "0.0" / "test_x.py").write_text("def test_x(): pass\n", encoding="utf-8")
    d = fake_root / "chapters" / "0.0"
    d.mkdir(parents=True)
    (d / "notes.tex").write_text("\\begin{theorem}x\\end{theorem}\n", encoding="utf-8")
    row = Registry(fake_root).progress()[0]
    assert not row.complete
    assert not row.has_pitfalls and not row.has_asserts_decl

    (d / "pitfalls.md").write_text("坑" * (MIN_PITFALL_DOC_CHARS - 1), encoding="utf-8")
    (d / "asserts.yaml").write_text("tests: []\n", encoding="utf-8")
    row = Registry(fake_root).progress()[0]
    assert row.pitfall_chars == MIN_PITFALL_DOC_CHARS - 1
    assert not row.has_pitfalls and not row.complete

    (d / "pitfalls.md").write_text("坑" * MIN_PITFALL_DOC_CHARS, encoding="utf-8")
    row = Registry(fake_root).progress()[0]
    assert row.has_pitfalls and row.complete


def test_pitfalls_and_asserts_paths(fake_root: pathlib.Path) -> None:
    reg = Registry(fake_root)
    assert reg.pitfalls_file("1.2") == fake_root / "chapters" / "1.2" / "pitfalls.md"
    assert reg.asserts_decl_file("1.2") == fake_root / "chapters" / "1.2" / "asserts.yaml"


def test_progress_reports_broken_manifest_without_raising(fake_root: pathlib.Path) -> None:
    write_chapter(fake_root, "0.0", "MANIFEST = {'chapter': '0.0', 'title': 'x', 'vizzes': []}\n")
    row = Registry(fake_root).progress()[0]
    assert row.has_module and not row.manifest_ok
    assert any("vizzes" in p for p in row.manifest_problems)


def test_summary_never_reports_more_done_than_declared(fake_root: pathlib.Path) -> None:
    write_chapter(fake_root, "0.0", GOOD_MANIFEST % {"cid": "0.0"})
    # 故意造两个 viz 文件但只声明 2 个（等价于"实现数不得超过声明数"）
    (fake_root / "viz" / "chapters" / "0.0").mkdir(parents=True)
    for name in ("demo.viz.ts", "extra.viz.ts", "another.viz.ts"):
        (fake_root / "viz" / "chapters" / "0.0" / name).write_text("export {};\n", encoding="utf-8")
    s = Registry(fake_root).summary()
    assert s["chapters_declared"] == 2
    assert s["vizzes_declared"] == 6  # 假 yaml 里 0.0 声明 2 个 + 1.1 声明 4 个
    assert s["vizzes_implemented"] == 1  # 只算声明过的 demo
    assert s["chapters_complete"] == 0
    assert len(s["chapters"]) == 2


def test_real_repo_progress_matches_yaml() -> None:
    s = Registry(ROOT).summary()
    assert s["chapters_declared"] == 28
    assert s["vizzes_declared"] == 115, "115 = chapters.yaml 里 28 章的可视化总数"
    assert s["chapters_extra"] == 1 and s["vizzes_extra"] == 2
    assert s["vizzes_total"] == 117
    assert 0 <= s["vizzes_implemented"] <= 115
    assert s["chapters_complete"] <= s["chapters_declared"]
    # 0.0 章（额外章节）必须已经是一个"能跑"的样板：manifest 合法、声明 2 个可视化
    row = next(r for r in s["chapters"] if r["id"] == "0.0")
    assert row["declared"] is False
    assert row["manifestOk"] is True
    assert row["vizDeclared"] == 2
    # 声明章节按 yaml 顺序在前，额外章节排在后面
    assert [r["id"] for r in s["chapters"]][-1] == "0.0"


def test_windows_style_newlines_do_not_break_discovery(fake_root: pathlib.Path) -> None:
    path = write_chapter(fake_root, "0.0", GOOD_MANIFEST % {"cid": "0.0"})
    path.write_bytes(path.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    assert Registry(fake_root).ids() == ["0.0"]


def test_manifest_error_is_not_swallowed_in_strict_mode(fake_root: pathlib.Path) -> None:
    write_chapter(fake_root, "0.0", (GOOD_MANIFEST % {"cid": "0.0"}).replace('"error"', '"mesh"'))
    with pytest.raises(RegistryError) as ei:
        Registry(fake_root).discover(strict=True)
    assert "规则 3" in str(ei.value)


def test_registry_root_is_resolved_absolute() -> None:
    reg = Registry(".")
    assert reg.root.is_absolute()
    assert reg.root == pathlib.Path.cwd().resolve()


def test_chapter_entry_manifest_is_typed() -> None:
    entry = Registry(ROOT).get("0.0")
    viz = entry.manifest.viz("oscillator_regimes")
    assert viz.params[1].name == "zeta"
    assert viz.params[1].critical == (1.0,)
    assert viz.required_aux == ("residual",)
    assert viz.time.mode == "stream"
    with pytest.raises(KeyError):
        entry.manifest.viz("missing")
