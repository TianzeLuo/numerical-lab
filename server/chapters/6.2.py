r"""6.2 章：交互式教材打包与视频导出（3 个可视化）。

这一章不再解方程，而是审计**产出链路**本身。三个页面各回答一个"看起来成功"背后的
可测量问题：

1. ``catalog_nav`` —— 28 章的目录不是按章节号排出来的，而是依赖图的最长路径分层。
   主图是依赖 DAG（节点=章，边=前置依赖），副图是四项图错误的计数；选中的章会高亮
   它的祖先闭包。``deps: [all]`` 被展开为"清单里排在它前面的所有章"。
2. ``export_timeline`` —— 视频的时间轴有**两本账**：内容时刻（这一帧取自数值解的哪个
   时刻）与播放时刻（GIF 的 delay 以 1/100 s 为单位）。最近邻取样的内容误差被半个
   源步长界住，而 24 fps 的 GIF 实际按 25 fps 播放，漂移随帧号线性累积。
3. ``book_build`` —— 讲义构建的逐章检查：笔记、附表行数与声明一致、引用闭合、
   ``\input`` 目标存在、字形无风险、pitfalls 达标。热图的行=章、列=检查项；
   副图是每章的问题数。页面不把"编译没报错"当成"内容闭合"。

三个页面的数据都来自确定性计算（:mod:`numcore.apps.packaging` + 当前仓库的文本/图），
不读 ``dist/`` 的临时报告——否则"审计"会随上一次构建的残留而漂移。
"""

from __future__ import annotations

import functools
import json
import math
import pathlib
import re
from typing import Any, Mapping, Sequence

import numpy as np

from numcore.apps import packaging
from numcore.ode import rk
from numcore.ode.stepsize import integrate_adaptive
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

CHAPTER = "6.2"
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PARTS = ("I", "II", "III", "IV", "V", "VI")
_CHECK_IDS = ("notes", "table", "refs", "inputs", "glyph", "pitfalls")
_CHECK_LABELS = ("笔记", "附表", "引用", "输入", "字形", "陷阱")
_ROMAN = {p: i + 1 for i, p in enumerate(_PARTS)}


def _viz(**kwargs: Any) -> dict[str, Any]:
    return kwargs


_PITFALLS = {
    "catalog_nav": (
        "目录顺序不是构建顺序：章节号只反映写作计划，真正能编译的前提是依赖已就位。"
        "把 3.4 放在 2.3 前面会缺正交多项式，把 6.1 放在 4.x 前面会缺四个求解器。"
        "第二：``deps: [all]`` 不是「依赖所有章」的字面意思，本章把它展开为「清单里排在"
        "它前面的所有章」，否则 6.2 会依赖自己。第三：没有环 ≠ 顺序唯一；同层节点按"
        "清单顺序稳定排序，换实现若用集合遍历，构建顺序会漂移。"
        "第四：图的边数不等于依赖条目数——``all`` 会展开成 27 条边。"
    ),
    "export_timeline": (
        "「视频能播」不等于「时间轴对」。内容时刻与播放时刻是两本账：最近邻取样把每帧"
        "量化到源快照，误差有半个源步长的上界；按序号抽稀则在非均匀步长下把物理时间"
        "拉弯。第二：GIF 的 delay 是整数厘秒，24 fps 请求被写成 4 cs，实际播放 25 fps，"
        "60 fps 被写成 2 cs，实际只有 50 fps；误差随帧号线性累积，长时间片尾会明显快/慢。"
        "第三：帧数上限截断不是「短一点」——它直接改变导出的时间跨度，必须在账本里报出来。"
        "第四：最近邻内容误差小，不代表画面不抖；同一快照被连续显示多帧（stall）是"
        "另一条独立的量。"
    ),
    "book_build": (
        "「编译通过」不是「书对」：xelatex 对缺字形只吞字、不报错，未定义引用也要两遍"
        "编译才稳定；附表的行数如果与 asserts.yaml 声明不一致，PDF 里就是旧数字。"
        "第二：引用必须在单章内闭合，因为 G9 逐章编译；跨章 eqref 引用在合并本里能过，"
        "单章编译会失败。第三：注释里的 input/ref 命令不算依赖，先剔注释再计数。"
        "第四：6 列检查里任何一格为 0 都不能靠「其它列全绿」抵消；问题数按章统计，"
        "不是全局平均。"
    ),
}


MANIFEST: dict[str, Any] = {
    "chapter": CHAPTER,
    "title": "交互式教材打包与视频导出",
    "part": "VI",
    "summary": (
        "把 28 章组织成可导航、可导出、可编译的产品：目录依赖图决定构建顺序，"
        "视频导出的内容时刻与 GIF 播放时刻必须分开记账，讲义构建则逐章核对引用闭合、"
        "附表行数、输入文件与字形风险。三个页面都在审计「看起来成功」背后的可测量不变量。"
    ),
    "vizzes": [
        _viz(
            id="catalog_nav",
            title="目录与依赖：28 章的构建顺序",
            description=(
                "节点是章、边是前置依赖，按最长路径分层。选中某一章会高亮它的祖先闭包；"
                "选中某一部会高亮该部的节点。副图是环、未知依赖、自依赖、目录漂移四项"
                "错误计数（都应为 0）。侧栏报最长关键路径与闭包大小。"
            ),
            dataSource="backend",
            params=[
                {
                    "name": "part",
                    "label": "高亮分部",
                    "type": "enum",
                    "options": ["all", "I", "II", "III", "IV", "V", "VI"],
                    "default": "all",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "focus",
                    "label": "焦点章（高亮其前置闭包）",
                    "type": "enum",
                    "options": ["all", *[
                        "1.1", "1.2", "1.3", "1.4", "1.5",
                        "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
                        "3.1", "3.2", "3.3", "3.4", "3.5",
                        "4.1", "4.2", "4.3", "4.4", "4.5", "4.6",
                        "5.1", "5.2", "5.3", "5.4", "6.1", "6.2",
                    ]],
                    "default": "4.3",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            time={"mode": "none"},
            requiredAux=["error"],
            axes=[
                {"id": "main", "autoscaled": False, "title": "依赖图（按最长路径分层）", "ylabel": "层内序号"},
                {"id": "error", "autoscaled": False, "title": "依赖图的错误计数（应为 0）", "ylabel": "计数"},
            ],
            pitfalls=_PITFALLS["catalog_nav"],
            theory=[
                "拓扑序：每条边 $u\\to v$ 中 $u$ 必须排在 $v$ 前；同层按清单顺序稳定排序",
                "分层深度 $\\operatorname{depth}(v)=1+\\max\\operatorname{depth}(u)$，最长链是关键路径（不是章节号顺序）",
                "deps: [all] 展开为清单中排在它前面的所有章，不包含自身",
            ],
        ),
        _viz(
            id="export_timeline",
            title="视频导出：帧数、时长与两本时间账",
            description=(
                "源时间轴来自自适应 dopri5（非均匀步长）。把 logistic 轨迹按固定 fps "
                "重采样：主图是源步长与导出帧的位置，副图是内容时刻误差（最近邻/线性/"
                "序号抽稀）与 GIF 厘秒量化造成的播放漂移。侧栏把名义帧数、实际帧数、"
                "截断、stall、状态误差与 delay_cs 分开报。"
            ),
            dataSource="backend",
            params=[
                {
                    "name": "tol",
                    "label": "自适应容差 tol",
                    "type": "enum",
                    "options": [1e-3, 1e-5, 1e-7],
                    "default": 1e-5,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "fps",
                    "label": "请求帧率 fps",
                    "type": "enum",
                    "options": [12, 24, 25, 30, 60],
                    "default": 24,
                    "critical": [25.0],
                    "criticalNote": "25 fps 是 GIF 厘秒量化能精确表示的一档（4 cs）；24/30/60 都会被改写。",
                },
                {
                    "name": "sampling",
                    "label": "取样方式",
                    "type": "enum",
                    "options": ["nearest", "linear", "decimate"],
                    "default": "nearest",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "budget",
                    "label": "帧数上限",
                    "type": "enum",
                    "options": [120, 240],
                    "default": 240,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            time={"mode": "none"},
            requiredAux=["error"],
            axes=[
                {"id": "main", "autoscaled": False, "scale": "log", "title": "自适应步长 h(t) 与导出帧", "ylabel": "步长 h（对数）"},
                {"id": "error", "autoscaled": False, "title": "时间误差：内容时刻 / GIF 播放时刻", "ylabel": "误差（s）"},
            ],
            pitfalls=_PITFALLS["export_timeline"],
            theory=[
                "帧数 $=\\lfloor\\mathrm{fps}\\cdot(t_{\\mathrm{end}}-t_0)\\rfloor+1$；超过上限时如实截断",
                "GIF $\\mathrm{delay}_{cs}=\\max(1,\\operatorname{round}(100/\\mathrm{fps}))$，实际 $\\mathrm{fps}=100/\\mathrm{delay}_{cs}$",
                "最近邻内容误差 $\\le h/2$；GIF 播放误差随帧号线性累积",
            ],
        ),
        _viz(
            id="book_build",
            title="讲义构建：引用、附表与字形审计",
            description=(
                "逐章检查六件事：notes 存在且含定理、asserts_table 行数与 asserts.yaml "
                "声明一致、引用在章内闭合、LaTeX 的 input 目标存在、notes+附表无 xelatex 无字形"
                "风险字符、pitfalls $\\ge800$ 字。热图行=章、列=检查项；副图是每章的问题数。"
            ),
            dataSource="backend",
            params=[
                {
                    "name": "part",
                    "label": "分部筛选",
                    "type": "enum",
                    "options": ["all", "I", "II", "III", "IV", "V", "VI"],
                    "default": "all",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "sort",
                    "label": "排序",
                    "type": "enum",
                    "options": ["catalog", "topo", "fail"],
                    "default": "topo",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            time={"mode": "none"},
            requiredAux=["error"],
            axes=[
                {"id": "main", "autoscaled": False, "title": "逐章构建检查（1=通过，0=缺）", "ylabel": "章"},
                {"id": "error", "autoscaled": False, "title": "每章问题数（0..6）", "ylabel": "问题数"},
            ],
            pitfalls=_PITFALLS["book_build"],
            theory=[
                "附表行数必须等于 asserts_report.md 的断言条数（纸面与机器同源）",
                "引用在单章内闭合；LaTeX 的 input 目标必须存在（注释中的命令不算）",
                "notes+附表不含 xelatex 无字形风险字符；pitfalls $\\ge800$ 字",
            ],
        ),
    ],
}


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
def _finite(v: Any, fallback: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return fallback
    return x if math.isfinite(x) else fallback


def _cid_key(cid: str) -> tuple[int, int]:
    a, b = cid.split(".")
    return int(a), int(b)


def _json_list(values: Any) -> str:
    return json.dumps([str(v) for v in values], ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# 1. 目录依赖图
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _catalog_graph() -> dict[str, Any]:
    from server.registry import load_chapter_yaml

    catalog = load_chapter_yaml(_ROOT)
    ids = sorted(catalog, key=_cid_key)
    raw_deps = {cid: list((catalog[cid].get("deps") or [])) for cid in ids}
    deps, dep_problems = packaging.resolve_dependencies(ids, raw_deps)
    order, cycles = packaging.topological_order(ids, deps)
    if cycles:
        raise ValueError(f"docs/chapters.yaml 的依赖图有环：{cycles}")
    depth = packaging.dependency_depth(ids, deps)
    chain = packaging.longest_chain(ids, deps)
    layers = [list(layer) for layer in packaging.dependency_layers(ids, deps)]
    rank = {cid: i for i, cid in enumerate(ids)}
    parents = {cid: list(deps[cid]) for cid in ids}
    children: dict[str, list[str]] = {cid: [] for cid in ids}
    for cid in ids:
        for dep in deps[cid]:
            children[dep].append(cid)
    # 层内用重心法排一遍（上行看父节点、下行看子节点），减少交叉；同重心按清单序。
    ypos: dict[str, float] = {}
    for d, layer in enumerate(layers):
        m = len(layer)
        for i, cid in enumerate(layer):
            ypos[cid] = float(i) - 0.5 * (m - 1)

    def _bary(cid: str, refs: Mapping[str, Sequence[str]], sign: float) -> float:
        vals = [sign * ypos[r] for r in refs.get(cid, ()) if r in ypos]
        return float(sum(vals) / len(vals)) if vals else ypos.get(cid, 0.0)

    for _sweep in range(3):
        for d in range(1, len(layers)):
            layers[d].sort(key=lambda c: (_bary(c, parents, 1.0), rank[c]))
            m = len(layers[d])
            for i, cid in enumerate(layers[d]):
                ypos[cid] = float(i) - 0.5 * (m - 1)
        for d in range(len(layers) - 2, -1, -1):
            layers[d].sort(key=lambda c: (_bary(c, children, 1.0), rank[c]))
            m = len(layers[d])
            for i, cid in enumerate(layers[d]):
                ypos[cid] = float(i) - 0.5 * (m - 1)
    pos = {cid: (float(depth[cid]), ypos[cid]) for cid in ids}
    edges: list[tuple[str, str]] = []
    for cid in ids:
        for dep in deps[cid]:
            edges.append((dep, cid))
    parts = {cid: str(catalog[cid].get("part", "")) for cid in ids}
    vizzes = {cid: int(catalog[cid].get("viz", 0) or 0) for cid in ids}
    titles = {cid: str(catalog[cid].get("title", "")) for cid in ids}
    return {
        "ids": ids,
        "deps": deps,
        "order": order,
        "depth": depth,
        "layers": layers,
        "chain": chain,
        "edges": edges,
        "pos": pos,
        "parts": parts,
        "vizzes": vizzes,
        "titles": titles,
        "problems": dep_problems,
    }


def _ancestor_closure(focus: str, deps: Mapping[str, Sequence[str]]) -> set[str]:
    if focus == "all":
        return set(deps)
    seen: set[str] = set()
    stack = [focus]
    while stack:
        cid = stack.pop()
        if cid in seen or cid not in deps:
            continue
        seen.add(cid)
        stack.extend(deps[cid])
    return seen


def _catalog_nav(params: Mapping[str, Any]) -> Frame:
    g = _catalog_graph()
    ids: list[str] = g["ids"]
    deps = g["deps"]
    pos = g["pos"]
    part = str(params.get("part", "all"))
    focus = str(params.get("focus", "4.3"))
    if part not in ("all", *_PARTS):
        raise ValueError(f"未知分部 {part!r}")
    if focus != "all" and focus not in pos:
        raise ValueError(f"未知焦点章 {focus!r}")
    closure = _ancestor_closure(focus, deps)
    part_mask = np.asarray(
        [1.0 if part == "all" or g["parts"][cid] == part else 0.0 for cid in ids], dtype=float
    )
    focus_mask = np.asarray([1.0 if cid in closure else 0.0 for cid in ids], dtype=float)
    node_hl = np.where(part == "all", focus_mask, part_mask)
    is_focus = np.asarray([1.0 if cid == focus else 0.0 for cid in ids], dtype=float)
    edge_hl = np.asarray(
        [
            1.0 if (a in closure and b in closure and part == "all") or (part != "all" and (g["parts"].get(a) == part or g["parts"].get(b) == part)) else 0.0
            for a, b in g["edges"]
        ],
        dtype=float,
    )
    if focus == "all" and part == "all":
        node_hl = np.ones(len(ids), dtype=float)
        edge_hl = np.ones(len(g["edges"]), dtype=float)

    node_x = np.asarray([pos[cid][0] for cid in ids], dtype=float)
    node_y = np.asarray([pos[cid][1] for cid in ids], dtype=float)
    node_depth = np.asarray([g["depth"][cid] for cid in ids], dtype=float)
    node_part = np.asarray([float(_ROMAN.get(g["parts"][cid], 0)) for cid in ids], dtype=float)
    node_viz = np.asarray([float(g["vizzes"][cid]) for cid in ids], dtype=float)
    node_rank = np.zeros(len(ids), dtype=float)
    for rank, cid in enumerate(g["order"]):
        node_rank[ids.index(cid)] = float(rank)
    node_deps = np.asarray([float(len(deps[cid])) for cid in ids], dtype=float)
    e = g["edges"]
    edge_x0 = np.asarray([pos[a][0] for a, _ in e], dtype=float) if e else np.zeros(0)
    edge_y0 = np.asarray([pos[a][1] for a, _ in e], dtype=float) if e else np.zeros(0)
    edge_x1 = np.asarray([pos[b][0] for _, b in e], dtype=float) if e else np.zeros(0)
    edge_y1 = np.asarray([pos[b][1] for _, b in e], dtype=float) if e else np.zeros(0)

    n_unknown = sum(1 for p in g["problems"] if "未知依赖" in p)
    n_self = sum(1 for p in g["problems"] if "自依赖" in p)
    # 第四格"目录漂移"：清单里的 id 是否都能在 MANIFEST 生成的目录中找到。
    # 这里不读 dist；用服务器已实现的章节文件数量与清单数量的差作为构建期缺口。
    module_ids = {
        p.stem
        for p in (_ROOT / "server" / "chapters").glob("*.py")
        if re.fullmatch(r"\d+\.\d+", p.stem)
    }
    n_missing_module = sum(1 for cid in ids if cid not in module_ids)
    error_values = np.asarray([0.0, float(n_unknown), float(n_self), float(n_missing_module)], dtype=float)

    fr = Frame(chapter=CHAPTER, viz="catalog_nav", kind="frame")
    fr.add("node_x", node_x, label="节点依赖层")
    fr.add("node_y", node_y, label="层内序号")
    fr.add("node_depth", node_depth, label="拓扑深度")
    fr.add("node_part", node_part, label="分部编号（I..VI）")
    fr.add("node_viz", node_viz, label="可视化数")
    fr.add("node_rank", node_rank, label="拓扑序")
    fr.add("node_deps", node_deps, label="直接前置数")
    fr.add("node_highlight", node_hl, label="高亮掩码")
    fr.add("node_focus", is_focus, label="焦点章掩码")
    fr.add("edge_x0", edge_x0, label="边起点 x")
    fr.add("edge_y0", edge_y0, label="边起点 y")
    fr.add("edge_x1", edge_x1, label="边终点 x")
    fr.add("edge_y1", edge_y1, label="边终点 y")
    fr.add("edge_highlight", edge_hl, label="边高亮掩码")
    fr.add("error", error_values, label="依赖图错误计数（requiredAux）")
    fr.add("error_values", error_values, label="错误计数")
    fr.add("layer_size", np.asarray([float(len(layer)) for layer in g["layers"]]), label="每层节点数")
    fr.set_meta(
        n_chapters=len(ids),
        n_edges=len(e),
        n_layers=len(g["layers"]),
        max_depth=float(max(g["depth"].values(), default=0)),
        n_cycles=0.0,
        n_unknown=float(n_unknown),
        n_self=float(n_self),
        n_missing_module=float(n_missing_module),
        n_violations=float(sum(error_values)),
        focus_closure=float(len(closure)) if focus != "all" else float(len(ids)),
        focus_edges=float(sum(edge_hl)) if focus != "all" else float(len(e)),
        chain_len=float(len(g["chain"])),
        overflow=False,
    )
    fr.set_strings(
        status="ok" if sum(error_values) == 0 else "degraded",
        chapter_ids=_json_list(ids),
        parts=_json_list([g["parts"][cid] for cid in ids]),
        titles=_json_list([g["titles"][cid] for cid in ids]),
        focus=focus,
        part=part,
        chain=" → ".join(g["chain"]),
        problem_note=(
            "依赖图四项检查全为 0：无环、无未知依赖、无自依赖、清单章都有模块。"
            if sum(error_values) == 0
            else "依赖图有问题：" + "；".join(g["problems"] + [f"缺模块 {n_missing_module} 章"])
        ),
        closure_note=(
            "焦点=all：显示全图。"
            if focus == "all"
            else f"焦点 {focus} 的前置闭包含 {len(closure)} 章；这些章必须在它之前构建。"
        ),
        layer_note=f"{len(g['layers'])} 层、{len(e)} 条边；最长链 {len(g['chain'])} 章（关键路径）。",
        error_note="四格：环 / 未知依赖 / 自依赖 / 有清单但缺模块。",
    )
    fr.pitfalls = _PITFALLS["catalog_nav"]
    return fr


# --------------------------------------------------------------------------- #
# 2. 视频导出：内容时刻与播放时刻
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=12)
def _adaptive_source(tol: float) -> tuple[np.ndarray, np.ndarray, Any]:
    p = rk.problem("logistic")
    run = integrate_adaptive(
        "dopri5",
        p.rhs,
        (0.0, 4.0),
        p.y0_array(),
        float(tol),
        atol=float(tol),
        h0=0.05,
        local_extrapolation=True,
        exact=p.exact,
        label="logistic",
    )
    if not run.ok or run.t.size < 2:
        raise RuntimeError("自适应 logistic 积分失败，无法构造导出时间轴")
    return np.asarray(run.t, dtype=float), np.asarray(run.y, dtype=float), run


def _export_timeline(params: Mapping[str, Any]) -> Frame:
    tol = float(params["tol"])
    fps = float(params["fps"])
    mode = str(params["sampling"])
    budget = int(params["budget"])
    if tol not in (1e-3, 1e-5, 1e-7):
        raise ValueError(f"未知容差 {tol!r}")
    if fps not in (12.0, 24.0, 25.0, 30.0, 60.0):
        raise ValueError(f"未知 fps {fps!r}")
    if mode not in ("nearest", "linear", "decimate"):
        raise ValueError(f"未知取样方式 {mode!r}")
    if budget not in (120, 240):
        raise ValueError(f"未知帧数上限 {budget!r}")
    tt, yy, run = _adaptive_source(tol)
    plan = packaging.export_plan(
        tt,
        yy,
        fps,
        mode=mode,
        max_frames=budget,
        exact=rk.problem("logistic").exact,
    )
    fr = Frame(chapter=CHAPTER, viz="export_timeline", kind="frame")
    fr.add("source_t", plan.source_t, label="自适应接受步时刻")
    fr.add("source_step_t", plan.source_step_t, label="步长起点时刻")
    fr.add("source_step_h", plan.source_step_h, label="自适应步长 h")
    fr.add("frame_t", plan.frame_t, label="名义帧时刻")
    fr.add("frame_index", np.arange(plan.n_frames, dtype=float), label="帧号")
    fr.add("content_t", plan.content_t, label="内容时刻")
    fr.add("playback_t", plan.playback_t, label="GIF 播放时刻")
    fr.add("content_index", plan.content_index.astype(float), label="源快照下标")
    fr.add("weight", plan.weight, label="线性插值权重")
    fr.add("content_y", plan.content_y[:, 0], label="帧内容 y")
    fr.add("exact_y", plan.exact_y[:, 0] if plan.exact_y is not None else np.zeros(plan.n_frames), label="解析解 y")
    fr.add("error", plan.timing_error, label="内容时刻误差（requiredAux）")
    fr.add("timing_error", plan.timing_error, label="内容时刻误差")
    fr.add("playback_error", plan.playback_error, label="GIF 播放漂移")
    fr.add("state_error", plan.state_error, label="帧内容相对解析解的误差")
    fr.add("stall", plan.stall.astype(float), label="同源快照重复掩码")
    fr.set_meta(
        tol=float(tol),
        fps=float(fps),
        delay_cs=float(plan.delay_cs),
        actual_fps=float(plan.actual_fps),
        fps_error=float(plan.actual_fps - fps),
        n_source=int(plan.source_t.size),
        n_steps=int(run.n_accepted),
        n_rejected=int(run.n_rejected),
        h_min=float(np.min(plan.source_step_h)),
        h_max=float(np.max(plan.source_step_h)),
        h_ratio=float(np.max(plan.source_step_h) / np.min(plan.source_step_h)),
        span=float(plan.span),
        n_raw=int(plan.n_raw),
        n_frames=int(plan.n_frames),
        truncated=1.0 if plan.truncated else 0.0,
        n_truncated=float(plan.n_raw - plan.n_frames),
        nominal_duration=float(plan.nominal_duration),
        playback_duration=float(plan.playback_duration),
        duration_gap=float(plan.duration_gap),
        max_timing_error=float(plan.max_abs_timing_error),
        max_playback_error=float(plan.max_abs_playback_error),
        max_state_error=float(plan.max_state_error),
        n_stalls=float(plan.n_stalls),
        max_hold=float(plan.max_hold),
        max_hold_playback=float(plan.max_hold_playback),
        n_skipped_source=float(plan.n_skipped_source),
        overflow=False,
    )
    mode_label = {
        "nearest": "最近邻快照",
        "linear": "物理时间线性插值",
        "decimate": "按序号等距抽稀",
    }[mode]
    fr.set_strings(
        status="ok",
        sampling=mode,
        sampling_note=f"取样={mode_label}；帧内容取自源快照，误差另算。",
        source_note=(
            f"自适应 dopri5：{plan.source_t.size} 个接受点、{run.n_rejected} 次拒绝；"
            f"h∈[{np.min(plan.source_step_h):.4g}, {np.max(plan.source_step_h):.4g}]，"
            f"最大/最小={np.max(plan.source_step_h)/np.min(plan.source_step_h):.3g}。"
        ),
        frame_note=(
            f"名义 {plan.n_raw} 帧，上限 {budget}，实出 {plan.n_frames} 帧"
            + (f"（截断 {plan.n_raw - plan.n_frames} 帧）" if plan.truncated else "（未截断）")
            + f"；stall {plan.n_stalls} 帧，最长同源 {plan.max_hold:.4f} s。"
        ),
        gif_note=(
            f"GIF delay={plan.delay_cs} cs（请求 {fps:g} fps → 实际 {plan.actual_fps:.4f} fps）；"
            f"播放总时长 {plan.playback_duration:.4f} s，名义 {plan.nominal_duration:.4f} s，"
            f"片尾漂移 {plan.max_abs_playback_error:.4f} s。"
        ),
        state_note=(
            f"最近邻/插值的帧内容相对解析解 max|Δy|={plan.max_state_error:.3e}"
            f"（源步长二阶项主导）。"
        ),
        error_note=(
            "内容误差有界（≤h/2，最近邻），播放漂移按 delay_cs 线性累积；"
            "decimate 模式把非均匀步长按序号拉平，误差不再有界。"
        ),
    )
    fr.pitfalls = _PITFALLS["export_timeline"]
    return fr


# --------------------------------------------------------------------------- #
# 3. 讲义构建审计
# --------------------------------------------------------------------------- #
def _strip_text(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _declared_assert_count(cid: str) -> int:
    import yaml

    path = _ROOT / "chapters" / cid / "asserts.yaml"
    if not path.exists():
        return 0
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return 0
    return len(data.get("tests") or [])


def _report_assert_count(cid: str) -> int:
    """``asserts_report.md`` 里的实测断言条数；没有报告返回 -1。

    附表由**同一份断言报告**生成（``tools/build_book.py``），所以"附表行数=报告条数"
    才是纸面与机器同源。拿它去对 ``asserts.yaml`` 的**测试函数数**会错：一个测试可以
    登记多条断言（1.1 章 27 个函数 → 31 条记录）。
    """
    path = _ROOT / "chapters" / cid / "asserts_report.md"
    text = _strip_text(path)
    if text is None:
        return -1
    m = re.search(r"断言条数：\s*(\d+)", text)
    return int(m.group(1)) if m else -1


def _resolve_input(target: str) -> pathlib.Path:
    rel = target.strip()
    p = _ROOT / rel
    if not p.suffix:
        p = _ROOT / (rel + ".tex")
    return p


@functools.lru_cache(maxsize=1)
def _book_audit_base() -> dict[str, Any]:
    from tools.build_book import latex_risky_chars

    g = _catalog_graph()
    ids: list[str] = g["ids"]
    rows: dict[str, dict[str, Any]] = {}
    for cid in ids:
        notes_path = _ROOT / "chapters" / cid / "notes.tex"
        table_path = _ROOT / "chapters" / cid / "asserts_table.tex"
        pitfalls_path = _ROOT / "chapters" / cid / "pitfalls.md"
        notes = _strip_text(notes_path)
        table = _strip_text(table_path)
        pitfalls = _strip_text(pitfalls_path)
        clean_notes = packaging.strip_tex_comments(notes or "")
        clean_table = packaging.strip_tex_comments(table or "")
        cross = packaging.tex_crossrefs(notes or "")
        labels = set(cross.labels)
        unresolved = [r for r in cross.refs if r not in labels]
        dup = cross.duplicate_labels
        inputs_missing = [t for t in cross.inputs if not _resolve_input(t).exists()]
        declared_tests = _declared_assert_count(cid)
        report_count = _report_assert_count(cid)
        table_rows = packaging.asserts_table_rows(table or "")
        risky = latex_risky_chars(clean_notes + "\n" + clean_table)
        checks = (
            bool(notes is not None and clean_notes.count("\\begin{theorem}") > 0),
            bool(table is not None and report_count > 0 and table_rows == report_count),
            bool(notes is not None and not unresolved and not dup),
            bool(notes is not None and not inputs_missing),
            bool(notes is not None and not risky),
            bool(pitfalls is not None and len(pitfalls.strip()) >= 800),
        )
        rows[cid] = {
            "checks": tuple(bool(x) for x in checks),
            "labels": len(cross.labels),
            "refs": len(cross.refs),
            "inputs": len(cross.inputs),
            "unresolved": len(unresolved),
            "duplicate": len(dup),
            "inputs_missing": len(inputs_missing),
            "risky": len(risky),
            "risky_chars": "".join(risky),
            "table_rows": table_rows,
            "declared_tests": declared_tests,
            "report_count": report_count,
            "theorems": clean_notes.count("\\begin{theorem}"),
            "chars": len(notes or ""),
            "pitfall_chars": len((pitfalls or "").strip()),
            "missing_inputs": tuple(inputs_missing),
        }
    return {"ids": ids, "rows": rows}


def _book_order(ids: list[str], sort: str) -> list[str]:
    g = _catalog_graph()
    if sort == "catalog":
        return list(ids)
    if sort == "topo":
        rank = {cid: i for i, cid in enumerate(g["order"])}
        return sorted(ids, key=lambda c: rank.get(c, 10**9))
    if sort == "fail":
        base = _book_audit_base()
        rank = {cid: i for i, cid in enumerate(ids)}
        return sorted(ids, key=lambda c: (-(len(_CHECK_IDS) - sum(base["rows"][c]["checks"])), rank[c]))
    raise ValueError(f"未知排序 {sort!r}")


def _book_build(params: Mapping[str, Any]) -> Frame:
    part = str(params.get("part", "all"))
    sort = str(params.get("sort", "topo"))
    if part not in ("all", *_PARTS):
        raise ValueError(f"未知分部 {part!r}")
    base = _book_audit_base()
    g = _catalog_graph()
    ids = [cid for cid in g["ids"] if part == "all" or g["parts"][cid] == part]
    ids = _book_order(ids, sort)
    rows = base["rows"]
    n = len(ids)
    values = np.zeros((n, len(_CHECK_IDS)), dtype=float)
    for i, cid in enumerate(ids):
        values[i, :] = np.asarray(rows[cid]["checks"], dtype=float)
    issues = np.asarray([len(_CHECK_IDS) - int(np.sum(values[i])) for i in range(n)], dtype=float)
    labels = np.asarray([float(rows[c]["labels"]) for c in ids], dtype=float)
    refs = np.asarray([float(rows[c]["refs"]) for c in ids], dtype=float)
    inputs = np.asarray([float(rows[c]["inputs"]) for c in ids], dtype=float)
    risky = np.asarray([float(rows[c]["risky"]) for c in ids], dtype=float)
    table_rows = np.asarray([float(rows[c]["table_rows"]) for c in ids], dtype=float)
    declared = np.asarray([float(rows[c]["report_count"]) for c in ids], dtype=float)
    declared_tests = np.asarray([float(rows[c]["declared_tests"]) for c in ids], dtype=float)
    theorems = np.asarray([float(rows[c]["theorems"]) for c in ids], dtype=float)
    chars = np.asarray([float(rows[c]["chars"]) for c in ids], dtype=float)

    fr = Frame(chapter=CHAPTER, viz="book_build", kind="frame")
    # 热图行 0 在 y-min（光栅器的约定）；行标签由前端按同一顺序翻转。
    fr.add("check_values", values.reshape(-1), label="逐章检查矩阵（行主序）")
    fr.add("chapter_index", np.arange(n, dtype=float), label="显示行号")
    fr.add("error", issues, label="每章问题数（requiredAux）")
    fr.add("error_y", issues, label="每章问题数")
    fr.add("labels", labels, label="LaTeX 标签命令数")
    fr.add("refs", refs, label="\\ref 数")
    fr.add("inputs", inputs, label="LaTeX 输入命令数")
    fr.add("risky", risky, label="字形风险字符数")
    fr.add("table_rows", table_rows, label="附表行数")
    fr.add("declared", declared, label="报告断言条数")
    fr.add("declared_tests", declared_tests, label="asserts.yaml 声明的测试函数数")
    fr.add("theorems", theorems, label="定理数")
    fr.add("chars", chars, label="正文字符数")
    total_labels = int(np.sum(labels))
    total_refs = int(np.sum(refs))
    total_inputs = int(np.sum(inputs))
    total_risky = int(np.sum(risky))
    total_issues = int(np.sum(issues))
    missing_tables = sum(1 for c in ids if rows[c]["table_rows"] == 0)
    mismatched = sum(1 for c in ids if rows[c]["table_rows"] != rows[c]["report_count"])
    unresolved = sum(int(rows[c]["unresolved"]) for c in ids)
    duplicates = sum(int(rows[c]["duplicate"]) for c in ids)
    missing_inputs = sum(int(rows[c]["inputs_missing"]) for c in ids)
    bad_rows = [
        f"{cid}: " + "、".join(label for label, ok in zip(_CHECK_LABELS, rows[cid]["checks"]) if not ok)
        for cid in ids
        if not all(rows[cid]["checks"])
    ]
    fr.set_meta(
        n_chapters=float(n),
        n_checks=float(len(_CHECK_IDS)),
        n_issues=float(total_issues),
        n_labels=float(total_labels),
        n_refs=float(total_refs),
        n_inputs=float(total_inputs),
        n_risky=float(total_risky),
        n_unresolved=float(unresolved),
        n_duplicates=float(duplicates),
        n_missing_inputs=float(missing_inputs),
        n_missing_tables=float(missing_tables),
        n_table_mismatch=float(mismatched),
        n_table_rows=float(np.sum(table_rows)),
        n_declared=float(np.sum(declared)),
        n_declared_tests=float(np.sum(declared_tests)),
        n_theorems=float(np.sum(theorems)),
        overflow=False,
    )
    fr.set_strings(
        status="ok" if total_issues == 0 else "degraded",
        chapter_ids=_json_list(ids),
        check_ids=_json_list(_CHECK_IDS),
        check_labels=_json_list(_CHECK_LABELS),
        part=part,
        sort=sort,
        summary=f"{n} 章 × {len(_CHECK_IDS)} 项检查；问题格 {total_issues} 个。",
        issue_note=("全部检查通过。" if not bad_rows else "未通过：" + "；".join(bad_rows[:6]) + ("…" if len(bad_rows) > 6 else "")),
        ref_note=(
            f"标签 {total_labels}、引用 {total_refs}、输入 {total_inputs}；"
            f"未定义 {unresolved}、重复标签 {duplicates}、缺输入文件 {missing_inputs}。"
        ),
        table_note=(
            f"附表行数 {int(np.sum(table_rows))}，报告条数 {int(np.sum(declared))}；"
            f"不一致 {mismatched} 章，缺表 {missing_tables} 章。"
        ),
        glyph_note=f"字形风险字符 {total_risky} 个（notes+附表，注释已剔除）。",
        pitfall_note=(
            "pitfalls 全部 ≥800 字。"
            if all(rows[c]["pitfall_chars"] >= 800 for c in ids)
            else "pitfalls 不足 800 字：" + "、".join(c for c in ids if rows[c]["pitfall_chars"] < 800)
        ),
        check_note="列：笔记存在且含定理 / 附表行数=声明 / 章内引用闭合 / LaTeX input 目标存在 / 无字形风险 / pitfalls≥800 字。",
    )
    fr.pitfalls = _PITFALLS["book_build"]
    return fr


# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口；三个页面都没有时间轴，``t`` 由调度器给 0。"""
    del t
    if viz_id == "catalog_nav":
        return _catalog_nav(params)
    if viz_id == "export_timeline":
        return _export_timeline(params)
    if viz_id == "book_build":
        return _book_build(params)
    raise KeyError(f"章节 6.2 没有可视化 {viz_id!r}")
