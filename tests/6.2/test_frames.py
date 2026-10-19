"""6.2 章的帧级断言：页面上的每一个数字都必须能在磁盘/核心里找到出处。

三个页面共同的一条纪律：**审计页不许自证**。因此这里不复用前端模型的计算，而是
独立地从帧、磁盘与 ``numcore`` 重新算一遍：

1. 导出页：帧数组与 ``export_plan`` 逐位相等；24 fps 的 GIF 漂移与 25 fps 的零漂移；
2. 目录页：节点/边数与 ``docs/chapters.yaml`` 展开后的图一致；4.3 的闭包是 8 章；
3. 讲义页：每个格子的 0/1 都能由 notes/附表的实际内容与断言报告条数复算出来。
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.apps import packaging  # noqa: E402
from numcore.ode import rk  # noqa: E402
from numcore.ode.stepsize import integrate_adaptive  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.nlf import pack_frame  # noqa: E402
from server.registry import Registry, load_chapter_yaml  # noqa: E402
from tools.build_book import latex_risky_chars  # noqa: E402

VIZ_IDS = ("catalog_nav", "export_timeline", "book_build")
_ENTRY_CACHE: list = []
_CHECKS = ("notes", "table", "refs", "inputs", "glyph", "pitfalls")


def _entry():
    if not _ENTRY_CACHE:
        reg = Registry(ROOT)
        reg.discover(strict=False)
        _ENTRY_CACHE.append(reg.get("6.2"))
    return _ENTRY_CACHE[0]


def _viz(viz_id: str):
    return next(v for v in _entry().manifest.vizzes if v.id == viz_id)


def _defaults(viz_id: str) -> dict:
    return {p.name: p.default for p in _viz(viz_id).params}


def _frame(viz_id: str, **over):
    params = _defaults(viz_id)
    params.update(over)
    return build_frame(_entry(), viz_id, params, 0.0)


def _read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _expected_book_row(cid: str) -> dict:
    notes = _read(ROOT / "chapters" / cid / "notes.tex")
    table = _read(ROOT / "chapters" / cid / "asserts_table.tex")
    pitfalls = _read(ROOT / "chapters" / cid / "pitfalls.md")
    report = _read(ROOT / "chapters" / cid / "asserts_report.md")
    m = re.search(r"断言条数：\s*(\d+)", report)
    report_count = int(m.group(1)) if m else -1
    cross = packaging.tex_crossrefs(notes)
    labels = set(cross.labels)
    unresolved = [r for r in cross.refs if r not in labels]
    dup = cross.duplicate_labels
    missing_inputs = [t for t in cross.inputs if not (ROOT / (t if pathlib.Path(t).suffix else t + ".tex")).exists()]
    rows = packaging.asserts_table_rows(table)
    clean_notes = packaging.strip_tex_comments(notes)
    clean_table = packaging.strip_tex_comments(table)
    risky = latex_risky_chars(clean_notes + "\n" + clean_table)
    checks = (
        bool(notes and clean_notes.count("\\begin{theorem}") > 0),
        bool(table and report_count > 0 and rows == report_count),
        bool(notes and not unresolved and not dup),
        bool(notes and not missing_inputs),
        bool(notes and not risky),
        bool(pitfalls.strip() and len(pitfalls.strip()) >= 800),
    )
    return {
        "checks": checks,
        "table_rows": rows,
        "report_count": report_count,
        "labels": len(cross.labels),
        "refs": len(cross.refs),
        "inputs": len(cross.inputs),
    }


# --------------------------------------------------------------------------- #
# 1. 三页默认帧：有限、可装、requiredAux
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="三页默认帧的数组/meta 全部有限；pack_frame 后 JSON 头无 NaN/Infinity；requiredAux 可定位",
    chapter="6.2",
)
def test_all_frames_finite_packable_and_declare_aux() -> None:
    def boom(token: str) -> None:
        raise AssertionError(f"JSON 里出现非有限常量 {token!r}")

    n_arrays = 0
    bad = 0
    lines: list[str] = []
    for vid in VIZ_IDS:
        fr = _frame(vid)
        for name in fr.names:
            arr = np.asarray(fr[name], dtype=float)
            assert np.all(np.isfinite(arr)), f"{vid}/{name} 有非有限值"
        for key, val in fr.meta.items():
            if isinstance(val, float):
                assert math.isfinite(val), f"{vid} meta.{key} 非有限：{val!r}"
        for req in _viz(vid).required_aux:
            assert any(req in name for name in fr.names), f"{vid} 缺 requiredAux={req}"
        blob = pack_frame(fr)
        try:
            (hlen,) = struct.unpack_from("<I", blob, 4)
            json.loads(blob[8 : 8 + hlen].decode("utf-8"), parse_constant=boom)
        except AssertionError:
            bad += 1
        n_arrays += len(fr.names)
        lines.append(f"{vid}: {len(fr.names)} 数组/{len(fr.meta)} meta")
    rec = record(
        label="三页默认帧的有限性、装帧与 requiredAux",
        measured=f"非有限 JSON 常量 {bad} 个；" + "；".join(lines),
        tol="非有限常量=0，requiredAux 全部命中",
        passed=bool(bad == 0 and n_arrays >= 40),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. 导出帧 ↔ 核心逐位一致
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="帧数组与直接调用 integrate_adaptive+export_plan 的结果逐位相等；meta 的帧数/delay/漂移与计划一致",
    chapter="6.2",
)
def test_export_frame_matches_core_bitwise() -> None:
    tol = 1e-5
    p = rk.problem("logistic")
    run = integrate_adaptive(
        "dopri5", p.rhs, (0.0, 4.0), p.y0_array(), tol, atol=tol, h0=0.05,
        local_extrapolation=True, exact=p.exact, label="logistic",
    )
    plan = packaging.export_plan(run.t, run.y, 24.0, mode="nearest", max_frames=240, exact=p.exact)
    fr = _frame("export_timeline")
    arrays = {
        "frame_t": plan.frame_t,
        "content_t": plan.content_t,
        "playback_t": plan.playback_t,
        "timing_error": plan.timing_error,
        "playback_error": plan.playback_error,
        "state_error": plan.state_error,
        "stall": plan.stall.astype(float),
        "content_index": plan.content_index.astype(float),
    }
    diffs = {k: float(np.max(np.abs(np.asarray(fr[k]) - v))) if v.size else 0.0 for k, v in arrays.items()}
    meta_ok = (
        int(fr.meta["n_frames"]) == plan.n_frames
        and int(fr.meta["delay_cs"]) == 4
        and abs(fr.meta["actual_fps"] - 25.0) <= 1e-12
        and abs(fr.meta["max_timing_error"] - plan.max_abs_timing_error) <= 1e-15
        and abs(fr.meta["max_playback_error"] - plan.max_abs_playback_error) <= 1e-15
    )
    ok = max(diffs.values()) == 0.0 and meta_ok
    rec = record(
        label="导出帧与核心时间轴逐位对拍",
        measured=(
            f"8 个数组最大差={max(diffs.values()):.1e}；n={plan.n_frames}，delay={plan.delay_cs}cs，"
            f"max|Δt|={plan.max_abs_timing_error:.5f}，GIF 漂移={plan.max_abs_playback_error:.5f} s"
        ),
        tol="全部逐位相等；meta 精确",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 3. 三种取样的证据必须不同
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="linear 的内容时刻误差=0；decimate 的时间误差 > nearest；linear 状态误差 < decimate；fps=25 的 GIF 漂移为 0",
    chapter="6.2",
)
def test_sampling_modes_expose_distinct_failure_modes() -> None:
    near = _frame("export_timeline", sampling="nearest")
    lin = _frame("export_timeline", sampling="linear")
    dec = _frame("export_timeline", sampling="decimate")
    at25 = _frame("export_timeline", fps=25)
    at24 = _frame("export_timeline", fps=24)
    ok = (
        abs(lin.meta["max_timing_error"]) <= 1e-15
        and dec.meta["max_timing_error"] > 2.0 * near.meta["max_timing_error"]
        and lin.meta["max_state_error"] < 0.25 * dec.meta["max_state_error"]
        and abs(at25.meta["max_playback_error"]) <= 1e-15
        and at24.meta["max_playback_error"] > 0.1
    )
    rec = record(
        label="最近邻/线性/序号抽稀与 GIF 帧率的独立证据",
        measured=(
            f"max|Δt|：nearest={near.meta['max_timing_error']:.5f}，linear={lin.meta['max_timing_error']:.1e}，"
            f"decimate={dec.meta['max_timing_error']:.5f}；max|Δy|：linear={lin.meta['max_state_error']:.2e}，"
            f"decimate={dec.meta['max_state_error']:.2e}；GIF 漂移：24fps={at24.meta['max_playback_error']:.4f}s，"
            f"25fps={at25.meta['max_playback_error']:.1e}s"
        ),
        tol="linear timing=0；decimate>2×nearest；linear state<decimate/4；25fps 零漂移而 24fps>0.1s",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 4. 目录帧 ↔ 图
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="节点=28、边=展开后的依赖总数；四项错误全 0；焦点 4.3 的祖先闭包=8 章、9 条边；最长链 9 章",
    chapter="6.2",
)
def test_catalog_frame_matches_dependency_graph() -> None:
    catalog = load_chapter_yaml(ROOT)
    ids = sorted(catalog, key=lambda c: tuple(int(x) for x in c.split(".")))
    raw = {cid: list(catalog[cid].get("deps") or []) for cid in ids}
    deps, _ = packaging.resolve_dependencies(ids, raw)
    closure = set()
    stack = ["4.3"]
    while stack:
        cid = stack.pop()
        if cid in closure:
            continue
        closure.add(cid)
        stack.extend(deps[cid])
    expected_edges = sum(len(v) for v in deps.values())
    chain = packaging.longest_chain(ids, deps)
    fr = _frame("catalog_nav")
    n_nodes = len(np.asarray(fr["node_x"]))
    e = len(np.asarray(fr["edge_x0"]))
    errors = np.asarray(fr["error_values"])
    fr_ids = json.loads(fr.strings["chapter_ids"])
    ok = (
        n_nodes == 28
        and e == expected_edges
        and np.allclose(errors, 0.0)
        and int(fr.meta["focus_closure"]) == len(closure) == 8
        and int(fr.meta["chain_len"]) == len(chain) == 9
        and fr_ids == ids
        and np.max(np.asarray(fr["edge_x0"])) <= int(fr.meta["max_depth"])
    )
    rec = record(
        label="目录帧与展开后的依赖图对拍",
        measured=(
            f"节点 {n_nodes}、边 {e}（期望 {expected_edges}）；闭包 {int(fr.meta['focus_closure'])} 章、"
            f"链长 {int(fr.meta['chain_len'])}；错误计数 {errors.tolist()}"
        ),
        tol="节点=28、边=67、闭包=8、链长=9、错误=0",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 5. 目录帧的分部高亮
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="part=IV 时高亮 6 个节点、其余为 0；focus=all 时全部高亮；焦点掩码恰有一个 1",
    chapter="6.2",
)
def test_catalog_part_and_focus_masks() -> None:
    part = _frame("catalog_nav", part="IV", focus="all")
    all_hl = _frame("catalog_nav", part="all", focus="all")
    focus = _frame("catalog_nav", part="all", focus="4.3")
    ok = (
        int(np.sum(np.asarray(part["node_highlight"]))) == 6
        and int(np.sum(np.asarray(all_hl["node_highlight"]))) == 28
        and int(np.sum(np.asarray(focus["node_focus"]))) == 1
        and int(np.sum(np.asarray(part["edge_highlight"]))) >= 1
    )
    rec = record(
        label="分部/焦点高亮掩码",
        measured=(
            f"part=IV 高亮 {int(np.sum(np.asarray(part['node_highlight'])))} 节点、"
            f"{int(np.sum(np.asarray(part['edge_highlight'])))} 条边；all={int(np.sum(np.asarray(all_hl['node_highlight'])))}；"
            f"焦点标记 {int(np.sum(np.asarray(focus['node_focus'])))}"
        ),
        tol="IV=6 节点、all=28、焦点标记=1",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 6. 讲义矩阵 ↔ 磁盘
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="每个 0/1 格子都由 notes/附表/报告条数/引用/输入/字形/pitfalls 独立复算；问题数=6−行和",
    chapter="6.2",
)
def test_book_frame_reflects_filesystem() -> None:
    fr = _frame("book_build")
    n = int(fr.meta["n_chapters"])
    ids = json.loads(fr.strings["chapter_ids"])
    values = np.asarray(fr["check_values"]).reshape(n, 6)
    issues = np.asarray(fr["error_y"])
    mismatch: list[str] = []
    for i, cid in enumerate(ids):
        exp = _expected_book_row(cid)
        if tuple(int(x) for x in values[i]) != tuple(int(x) for x in exp["checks"]):
            mismatch.append(f"{cid}/检查")
        if int(issues[i]) != 6 - sum(exp["checks"]):
            mismatch.append(f"{cid}/问题数")
        if int(fr["table_rows"][i]) != exp["table_rows"] or int(fr["declared"][i]) != exp["report_count"]:
            mismatch.append(f"{cid}/附表")
        if int(fr["labels"][i]) != exp["labels"] or int(fr["refs"][i]) != exp["refs"] or int(fr["inputs"][i]) != exp["inputs"]:
            mismatch.append(f"{cid}/引用计数")
    ok = not mismatch and int(np.sum(issues)) == int(fr.meta["n_issues"])
    rec = record(
        label="讲义审计矩阵与磁盘实际内容对拍",
        measured=(
            f"{n} 章 × 6 项；不一致 {len(mismatch)} 处"
            + ("：" + "、".join(mismatch[:8]) if mismatch else "")
            + f"；总问题数 {int(np.sum(issues))}（meta {int(fr.meta['n_issues'])}）"
        ),
        tol="所有格子与独立复算一致；问题总数吻合",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 7. 讲义页的筛选与排序
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="part=I 只留 5 章；fail 排序把问题多的章放前面（同分按目录序）；行序与 chapter_ids 对应",
    chapter="6.2",
)
def test_book_part_filter_and_sorting() -> None:
    part = _frame("book_build", part="I", sort="catalog")
    fail = _frame("book_build", part="I", sort="fail")
    ids = json.loads(part.strings["chapter_ids"])
    fail_ids = json.loads(fail.strings["chapter_ids"])
    issues = list(np.asarray(fail["error_y"]))
    expected_part = ["1.1", "1.2", "1.3", "1.4", "1.5"]
    ok = (
        ids == expected_part
        and sorted(fail_ids) == sorted(expected_part)
        and issues == sorted(issues, reverse=True)
        and int(fail.meta["n_issues"]) == sum(issues)
        and all(i in expected_part for i in fail_ids)
    )
    rec = record(
        label="讲义的部筛选与失败优先排序",
        measured=f"part=I 顺序={ids}；fail 顺序={fail_ids}，问题数={issues}",
        tol="筛选 5 章；fail 的问题数非增；两页章集合一致",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 8. manifest 契约
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="3 个 viz 的 id 固定；每页 requiredAux 在默认帧里存在；fps 参数声明临界值 25 且 criticalNote 非空",
    chapter="6.2",
)
def test_manifest_contract() -> None:
    man = _entry().manifest
    ids = man.viz_ids
    lines: list[str] = []
    ok = ids == VIZ_IDS
    for vid in VIZ_IDS:
        spec = man.viz(vid)
        fr = _frame(vid)
        aux = all(any(req in name for name in fr.names) for req in spec.required_aux)
        ok = ok and aux and len(spec.pitfalls) >= 60 and len(spec.params) >= 2
        ok = ok and all(isinstance(ax.autoscaled, bool) for ax in spec.axes)
        lines.append(f"{vid}: params={len(spec.params)}, aux={spec.required_aux}")
    fps_spec = man.viz("export_timeline").param("fps")
    ok = ok and 25.0 in fps_spec.critical and bool(fps_spec.critical_note)
    rec = record(
        label="6.2 manifest 的参数/临界值/requiredAux 契约",
        measured="；".join(lines) + f"；fps.critical={fps_spec.critical}",
        tol="3 页齐备、requiredAux 命中、fps 临界值 25 有说明",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured
