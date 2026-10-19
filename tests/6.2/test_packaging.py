"""6.2 章核心断言的单元测试：导出时间轴、依赖图与 TeX 轻量解析。

三类“看起来成功”的假象分别在这里被拆开：

* 视频“能播”不等于帧数/时长对：内容时刻与 GIF 播放时刻是两本账；
* 目录“28 章都在”不等于构建顺序合法：每条依赖边都必须让前置章先出现；
* 讲义“编译通过”不等于引用/附表闭合：注释里的命令不算、行数要对报告。
"""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.apps import packaging  # noqa: E402
from numcore.ode import rk  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402
from server.registry import load_chapter_yaml  # noqa: E402

_LOGISTIC = rk.problem("logistic")
_T = np.asarray([0.0, 0.03, 0.10, 0.30, 0.70, 1.20, 2.00, 3.00, 4.00], dtype=float)
_Y = _LOGISTIC.exact(_T)
_EXACT = _LOGISTIC.exact


# --------------------------------------------------------------------------- #
# 1. 帧数、时长与边界
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="n=floor(fps·(t_end−t_0))+1；frame_t 严格递增且不越界；名义/播放总时长与 delay_cs 精确定义",
    chapter="6.2",
)
def test_export_frame_count_and_time_bounds() -> None:
    rows: list[str] = []
    ok = True
    for fps in (12.0, 24.0, 25.0, 30.0, 60.0):
        for mode in ("nearest", "linear", "decimate"):
            plan = packaging.export_plan(_T, _Y, fps, mode=mode, max_frames=500, exact=_EXACT)
            expect = int(np.floor((_T[-1] - _T[0]) * fps + 1e-12)) + 1
            cond = (
                plan.n_frames == expect
                and np.all(np.diff(plan.frame_t) > 0.0)
                and plan.frame_t[0] == _T[0]
                and plan.frame_t[-1] <= _T[-1] + 1e-12
                and abs(plan.nominal_duration - plan.n_frames / fps) <= 1e-12
                and abs(plan.playback_duration - plan.n_frames * plan.delay_cs / 100.0) <= 1e-12
            )
            ok = ok and bool(cond)
            rows.append(f"{fps:g}/{mode}: n={plan.n_frames}, raw={plan.n_raw}")
    rec = record(
        label="导出帧数、边界与时长账本",
        measured="；".join(rows[:5]) + f"（共 {len(rows)} 组）",
        tol="全部 15 组：帧数公式精确、frame_t 严格递增、时长公式精确",
        passed=ok,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. GIF 厘秒量化
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="delay_cs=max(1,round(100/fps))；actual_fps=100/delay_cs；25 fps 精确，24/30/60 都会被改写",
    chapter="6.2",
)
def test_gif_centisecond_quantization() -> None:
    expect = {12.0: (8, 12.5), 24.0: (4, 25.0), 25.0: (4, 25.0), 30.0: (3, 100.0 / 3.0), 60.0: (2, 50.0)}
    lines: list[str] = []
    ok = True
    for fps, (delay, actual) in expect.items():
        info = packaging.gif_timing(fps)
        cond = info["delay_cs"] == delay and abs(info["actual_fps"] - actual) <= 1e-12
        ok = ok and bool(cond)
        lines.append(f"{fps:g}→{info['delay_cs']}cs/{info['actual_fps']:.4g}fps")
    ok = ok and abs(packaging.gif_timing(25.0)["error_per_frame"]) <= 1e-15
    ok = ok and packaging.gif_timing(24.0)["error_per_frame"] < 0.0  # 4cs 比 1/24s 短 → 播放更快
    rec = record(
        label="GIF delay 厘秒量化账本",
        measured="、".join(lines) + "；25fps 每帧误差=0，24fps 每帧 -1/600 s",
        tol="delay/actual_fps 与公式逐项相等；25fps 零漂移",
        passed=ok,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 3. 最近邻误差上界
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="最近邻内容时刻误差 |content_t−frame_t| ≤ max(h)/2（落在最近快照的 Voronoi 半格内）",
    chapter="6.2",
)
def test_nearest_timing_error_half_step_bound() -> None:
    hmax = float(np.max(np.diff(_T)))
    worst = 0.0
    rows: list[str] = []
    for fps in (12.0, 24.0, 25.0, 30.0, 60.0):
        plan = packaging.export_plan(_T, _Y, fps, mode="nearest", max_frames=500, exact=_EXACT)
        mx = float(np.max(np.abs(plan.timing_error)))
        worst = max(worst, mx)
        rows.append(f"{fps:g}:{mx:.5f}")
    bound = 0.5 * hmax + 1e-12
    rec = record(
        label="最近邻取样的内容时刻误差上界",
        measured=f"max|Δt|={worst:.6f}；上界 hmax/2={bound:.6f}（hmax={hmax:.4f}）；" + "、".join(rows),
        tol="worst ≤ hmax/2 + 1e-12",
        passed=bool(worst <= bound),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 4. 线性插值：内容时刻精确、状态误差 O(h²)
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="linear 的内容时刻误差=0；状态误差 ≤ |y''|max·hmax²/8，logistic 的 |y''|max=0.0962",
    chapter="6.2",
)
def test_linear_sampling_is_time_exact_and_state_second_order() -> None:
    plan = packaging.export_plan(_T, _Y, 24.0, mode="linear", max_frames=500, exact=_EXACT)
    hmax = float(np.max(np.diff(_T)))
    bound = 0.0121 * hmax * hmax
    timing_ok = bool(np.max(np.abs(plan.timing_error)) <= 1e-15)
    state_ok = bool(plan.max_state_error <= bound)
    nearest = packaging.export_plan(_T, _Y, 24.0, mode="nearest", max_frames=500, exact=_EXACT)
    better = bool(plan.max_state_error < 0.25 * nearest.max_state_error)
    rec = record(
        label="线性重采样的时间精确性与二阶状态误差",
        measured=(
            f"max|timing|={np.max(np.abs(plan.timing_error)):.3e}；"
            f"max|Δy|={plan.max_state_error:.4e}（界 {bound:.4e}）；"
            f"最近邻 max|Δy|={nearest.max_state_error:.4e}"
        ),
        tol="timing=0（≤1e-15）；state ≤ 0.0121·hmax²；且优于最近邻 4 倍",
        passed=bool(timing_ok and state_ok and better),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 5. 序号抽稀 vs 最近邻
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="decimate 的帧数不变、源下标单调非降且末帧取到 n−1；其时间误差显著大于最近邻（非均匀步长被拉平）",
    chapter="6.2",
)
def test_decimation_distorts_time_on_nonuniform_sources() -> None:
    near = packaging.export_plan(_T, _Y, 24.0, mode="nearest", max_frames=500, exact=_EXACT)
    dec = packaging.export_plan(_T, _Y, 24.0, mode="decimate", max_frames=500, exact=_EXACT)
    mono = bool(np.all(np.diff(dec.content_index) >= 0))
    last = bool(dec.content_index[-1] == len(_T) - 1)
    bigger = bool(dec.max_abs_timing_error > 2.0 * near.max_abs_timing_error)
    same_n = dec.n_frames == near.n_frames
    rec = record(
        label="按序号抽稀在非均匀时间轴上的失真",
        measured=(
            f"nearest max|Δt|={near.max_abs_timing_error:.5f}，decimate={dec.max_abs_timing_error:.5f}"
            f"（{dec.max_abs_timing_error / max(near.max_abs_timing_error, 1e-30):.2f}×）；"
            f"下标单调={mono}、末帧取 n−1={last}、帧数相同={same_n}"
        ),
        tol="decimate > 2×nearest；下标单调且末帧覆盖",
        passed=bool(mono and last and bigger and same_n),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 6. 截断
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="max_frames=K 时实出 K 帧、truncated=True、n_raw−K 帧被显式记账；末帧时刻 < t_end（跨度为整帧时除外）",
    chapter="6.2",
)
def test_frame_budget_truncation_is_reported() -> None:
    plan = packaging.export_plan(_T, _Y, 24.0, mode="nearest", max_frames=10, exact=_EXACT)
    cond = (
        plan.n_frames == 10
        and plan.truncated
        and plan.n_raw - plan.n_frames > 0
        and plan.frame_t[-1] < _T[-1] - 0.5 / 24.0
    )
    rec = record(
        label="帧数上限截断的显式账本",
        measured=(
            f"raw={plan.n_raw}，budget={plan.max_frames}，实出={plan.n_frames}，"
            f"截断={plan.n_raw - plan.n_frames} 帧，末帧 t={plan.frame_t[-1]:.4f} < {_T[-1]:.4f}"
        ),
        tol="实出=budget、truncated=True、末帧明显提前",
        passed=bool(cond),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 7. 拓扑序
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="拓扑序包含全部 28 章；每条依赖边 u→v 中 u 在 v 前；6.2（deps=all）必须最后",
    chapter="6.2",
)
def test_topological_order_respects_catalog_dependencies() -> None:
    catalog = load_chapter_yaml(ROOT)
    ids = sorted(catalog, key=lambda c: tuple(int(x) for x in c.split(".")))
    raw = {cid: list(catalog[cid].get("deps") or []) for cid in ids}
    deps, problems = packaging.resolve_dependencies(ids, raw)
    order, cycles = packaging.topological_order(ids, deps)
    rank = {cid: i for i, cid in enumerate(order)}
    violations = [
        f"{dep}→{cid}"
        for cid in ids
        for dep in deps[cid]
        if rank.get(dep, -1) >= rank.get(cid, -2)
    ]
    ok = (
        len(order) == len(ids) == 28
        and not cycles
        and not problems
        and not violations
        and order[-1] == "6.2"
    )
    rec = record(
        label="目录依赖的拓扑序合法性",
        measured=(
            f"{len(order)} 章、{sum(len(v) for v in deps.values())} 条边；环 {len(cycles)}；"
            f"违例 {len(violations)}；末位={order[-1]}"
        ),
        tol="28 章全部出现、0 环、0 违例、末位 6.2",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 8. 分层与最长链
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="depth(v)=1+max depth(dep)；最长链长度=max depth+1=9，且链上相邻两项确实有依赖边",
    chapter="6.2",
)
def test_dependency_layers_and_longest_chain() -> None:
    catalog = load_chapter_yaml(ROOT)
    ids = sorted(catalog, key=lambda c: tuple(int(x) for x in c.split(".")))
    raw = {cid: list(catalog[cid].get("deps") or []) for cid in ids}
    deps, _ = packaging.resolve_dependencies(ids, raw)
    depth = packaging.dependency_depth(ids, deps)
    layers = packaging.dependency_layers(ids, deps)
    chain = packaging.longest_chain(ids, deps)
    chain_edges_ok = all(chain[i] in deps[chain[i + 1]] for i in range(len(chain) - 1))
    partition = sorted(cid for layer in layers for cid in layer) == sorted(ids)
    depth_ok = all(depth[cid] == (1 + max((depth[d] for d in deps[cid]), default=-1)) for cid in ids)
    ok = (
        partition
        and depth_ok
        and chain_edges_ok
        and len(chain) == max(depth.values()) + 1
        and chain[-1] == "6.2"
    )
    rec = record(
        label="依赖分层与关键路径",
        measured=(
            f"{len(layers)} 层、max depth={max(depth.values())}；链长 {len(chain)}："
            + " → ".join(chain)
        ),
        tol="分层是划分；depth 递推精确；链长=max depth+1 且末位 6.2",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 9. all 展开与坏依赖
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="all 展开为清单中位于其前的所有章（不含自身）；未知依赖/自依赖只进问题列表、不静默变成边",
    chapter="6.2",
)
def test_resolve_dependencies_expands_all_and_flags_bad_edges() -> None:
    ids = ["1.1", "1.2", "1.3"]
    deps, problems = packaging.resolve_dependencies(
        ids,
        {"1.1": [], "1.2": ["1.1", "1.1"], "1.3": ["all", "9.9", "1.3"]},
    )
    ok = (
        deps["1.2"] == ("1.1",)
        and deps["1.3"] == ("1.1", "1.2")
        and any("未知依赖" in p for p in problems)
        and any("自依赖" in p for p in problems)
        and len(problems) == 2
    )
    rec = record(
        label="all 展开与非法依赖的显式报告",
        measured=f"deps={deps}；问题={problems}",
        tol="1.3 依赖 1.1/1.2；未知与自依赖各报 1 条，不进入 deps",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 10. TeX 轻量解析
# --------------------------------------------------------------------------- #
@assertion(
    "B",
    theory="注释中的 \\ref/\\input/\\label 不计；重复 label 被单独标出；附表行只数 A/B/C & 开头的正文行",
    chapter="6.2",
)
def test_tex_crossrefs_ignore_comments_and_count_table_rows() -> None:
    text = r"""
\label{sec:a}
\label{sec:a}
\ref{sec:a} \eqref{eq:b}
% \ref{ghost} \input{nope}
\input{chapters/x/asserts_table}
\% 不是注释，但后面没有命令
"""
    cross = packaging.tex_crossrefs(text)
    table = r"""
% A & fake & row \\
A & real & row & v \\
B & another & row & v \\
\midrule
C & third & row & v \\
\bottomrule
"""
    rows = packaging.asserts_table_rows(table)
    ok = (
        cross.labels == ("sec:a", "sec:a")
        and cross.refs == ("sec:a", "eq:b")
        and cross.inputs == ("chapters/x/asserts_table",)
        and cross.duplicate_labels == ("sec:a",)
        and rows == 3
    )
    rec = record(
        label="注释剔除、重复标签与附表行计数",
        measured=f"labels={cross.labels}，refs={cross.refs}，inputs={cross.inputs}，dups={cross.duplicate_labels}，rows={rows}",
        tol="注释中的 3 个命令全部忽略；重复 label 检出；附表正文 3 行",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured
