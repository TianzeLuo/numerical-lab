r"""6.2 章的纯函数内核：把"打包与导出"变成可测的整数、时间轴与图。

为什么这些函数不该写进前端或服务层
----------------------------------
6.2 的三个可视化看似是"工具链审计"，但每一条结论都是**确定性的算术或图论**：

* 导出视频的帧数不是"导完数一下"，而是 :math:`\lfloor (t_1-t_0)f_{\rm fps}\rfloor+1`；
* GIF 的播放时长被 delay 字段（单位 1/100 s）量化，名义 24 fps 实际是 25 fps；
* 目录的构建顺序是依赖图的最长路径分层，不是章节号排序；
* 讲义审计的"引用闭合 / 附表行数 / 字形风险"都能由文本与声明精确计数。

把它们放进 ``numcore`` 的好处和其余章节一致：无 IO、可单测、帧与测试共用同一份实现。
服务层只负责读文件、把结果搬进 :class:`server.nlf.Frame`。

时间轴的两个账本
----------------
一个导出帧有**两个时刻**，混为一谈是本章最核心的陷阱：

1. **内容时刻**：这一帧的画面取自数值解的哪个时刻。用最近邻取样时它被量化到
   源快照网格上，误差不超过半个源步长；用"每 k 步取一帧"时它按快照序号而非物理
   时间分布，遇到自适应步长就会被拉得面目全非。
2. **播放时刻**：浏览器/播放器把它显示在哪个时刻。GIF 的 delay 只能取整数厘秒，
   ``delay_cs = max(1, round(100/fps))``，于是 24 fps 被播成 25 fps、60 fps 被播成
   50 fps；误差随帧号**线性累积**，不是有界的抖动。

:func:`export_plan` 同时返回两本账，因此“看起来在动”不再等于“时间轴对”。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "CrossRefs",
    "ExportPlan",
    "asserts_table_rows",
    "dependency_depth",
    "dependency_layers",
    "export_plan",
    "gif_timing",
    "longest_chain",
    "resolve_dependencies",
    "strip_tex_comments",
    "tex_crossrefs",
    "topological_order",
]


# --------------------------------------------------------------------------- #
# 视频导出：内容时刻 vs 播放时刻
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExportPlan:
    """一次导出的完整账本（所有数组长度都等于导出帧数，源数组除外）。

    ``content_index`` 是最近邻/分段线性/序号抽稀时用到的源快照下标；
    ``weight`` 只在 ``linear`` 模式下非零（右端点的插值权重）。
    ``timing_error = content_time - nominal_time``：内容时刻相对名义时刻的偏差。
    ``playback_error = playback_time - nominal_time``：GIF 厘秒量化造成的累计漂移。
    """

    fps: float
    delay_cs: int
    actual_fps: float
    mode: str
    max_frames: int
    truncated: bool
    n_raw: int
    n_frames: int
    source_t: np.ndarray
    source_y: np.ndarray
    source_step_t: np.ndarray
    source_step_h: np.ndarray
    frame_t: np.ndarray
    content_t: np.ndarray
    playback_t: np.ndarray
    content_index: np.ndarray
    weight: np.ndarray
    content_y: np.ndarray
    timing_error: np.ndarray
    playback_error: np.ndarray
    state_error: np.ndarray
    exact_y: np.ndarray | None
    stall: np.ndarray
    n_stalls: int
    max_hold: float
    max_hold_playback: float
    n_skipped_source: int
    span: float
    nominal_duration: float
    playback_duration: float
    max_abs_timing_error: float
    max_abs_playback_error: float
    max_state_error: float

    @property
    def duration_gap(self) -> float:
        """名义播放总时长与数值时间跨度的差（秒；末帧占满一个 delay）。"""
        return float(self.playback_duration - self.span)


def gif_timing(fps: float) -> dict[str, float | int]:
    """GIF 的厘秒量化账本：请求帧率 → 实际帧率与每帧漂移。

    与 :mod:`tools.export_video` 保持同一个公式（``max(1, round(100/fps))``）。
    ``error_per_frame`` 是 (播放时刻−名义时刻) 在最后一帧相对第一帧的斜率：
    ``delay_cs/100 − 1/fps``。
    """
    if not (isinstance(fps, (int, float)) and np.isfinite(fps) and fps > 0.0):
        raise ValueError(f"fps 必须是正有限数，收到 {fps!r}")
    delay_cs = max(1, int(round(100.0 / float(fps))))
    actual_fps = 100.0 / float(delay_cs)
    return {
        "delay_cs": int(delay_cs),
        "actual_fps": float(actual_fps),
        "fps_error": float(actual_fps - float(fps)),
        "error_per_frame": float(delay_cs / 100.0 - 1.0 / float(fps)),
    }


def _as_state(y: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"状态必须是 (n,) 或 (n,d)，收到 shape={arr.shape}")
    return arr


def _run_lengths(equal: np.ndarray) -> np.ndarray:
    """``equal[i]`` 表示第 i 帧与前一帧同源；返回每一段连续重复的长度。"""
    if equal.size == 0:
        return np.zeros(0, dtype=int)
    runs: list[int] = []
    cur = 1
    for flag in equal.reshape(-1):
        if flag:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return np.asarray(runs, dtype=int)


def export_plan(
    t: Any,
    y: Any,
    fps: float,
    *,
    mode: str = "nearest",
    max_frames: int = 240,
    exact: Callable[[Any], Any] | None = None,
) -> ExportPlan:
    """把一条非均匀源时间轴重采样成固定 fps 的导出帧，并给出两本时间账。

    参数
    ----
    t
        源快照时刻，严格递增、长度 ≥ 2。自适应求解器的接受步时刻正适合这里。
    y
        源状态，``(n,)`` 或 ``(n,d)``。
    fps
        请求帧率（正有限数）。GIF 的实际播放帧率由 :func:`gif_timing` 量化。
    mode
        ``nearest``：每帧取最近的源快照（时间误差有界，但画面会"台阶化"）；
        ``linear``：按物理时间线性插值（内容时刻精确，状态误差 O(h²)）；
        ``decimate``：按**序号**等距抽稀（朴素做法；非均匀步长下时间被扭曲）。
    max_frames
        帧数上限；超过则截断并置 ``truncated=True``（不假装导完了）。
    exact
        可选解析解，用于逐帧状态误差 ``state_error``。签名 ``t -> (n,1)``。
    """
    if mode not in ("nearest", "linear", "decimate"):
        raise ValueError(f"未知取样模式 {mode!r}（nearest / linear / decimate）")
    if not (isinstance(max_frames, (int, float)) and int(max_frames) == max_frames and max_frames >= 2):
        raise ValueError(f"max_frames 必须是 ≥2 的整数，收到 {max_frames!r}")
    tt = np.asarray(t, dtype=float).reshape(-1)
    yy = _as_state(y)
    if tt.size < 2 or yy.shape[0] != tt.size:
        raise ValueError(f"t 与 y 的长度必须一致且 ≥2：len(t)={tt.size}, y.shape={yy.shape}")
    if not np.all(np.isfinite(tt)) or not np.all(np.isfinite(yy)):
        raise ValueError("源时间轴/状态含非有限值")
    if np.any(np.diff(tt) <= 0.0):
        raise ValueError("源时间轴必须严格递增")
    timing = gif_timing(float(fps))
    t0 = float(tt[0])
    span = float(tt[-1] - t0)
    n_raw = int(np.floor(span * float(fps) + 1.0e-12)) + 1
    n_frames = min(n_raw, int(max_frames))
    truncated = n_frames < n_raw
    frame_t = t0 + np.arange(n_frames, dtype=float) / float(fps)

    if mode == "decimate":
        target = np.linspace(0.0, float(tt.size - 1), n_frames) if n_frames > 1 else np.zeros(1)
        idx = np.round(target).astype(int)
        idx = np.clip(idx, 0, tt.size - 1)
        weight = np.zeros(n_frames, dtype=float)
        content_t = tt[idx].copy()
        content_y = yy[idx].copy()
    elif mode == "nearest":
        right = np.searchsorted(tt, frame_t, side="left")
        right = np.clip(right, 0, tt.size - 1)
        left = np.clip(right - 1, 0, tt.size - 1)
        take_right = np.abs(tt[right] - frame_t) < np.abs(tt[left] - frame_t)
        idx = np.where(take_right, right, left)
        weight = np.zeros(n_frames, dtype=float)
        content_t = tt[idx].copy()
        content_y = yy[idx].copy()
    else:  # linear
        right = np.searchsorted(tt, frame_t, side="right")
        right = np.clip(right, 1, tt.size - 1)
        left = right - 1
        denom = tt[right] - tt[left]
        weight = np.clip((frame_t - tt[left]) / denom, 0.0, 1.0)
        idx = left.copy()
        content_t = frame_t.copy()
        content_y = (1.0 - weight)[:, None] * yy[left] + weight[:, None] * yy[right]

    timing_error = content_t - frame_t
    playback_t = t0 + np.arange(n_frames, dtype=float) * float(timing["delay_cs"]) / 100.0
    playback_error = playback_t - frame_t

    exact_y = None
    if exact is not None:
        exact_y = _as_state(exact(frame_t))
        if exact_y.shape[0] != n_frames:
            raise ValueError(f"解析解返回 {exact_y.shape[0]} 个点，期望 {n_frames}")
        state_error = np.max(np.abs(content_y - exact_y), axis=1)
    else:
        state_error = np.zeros(n_frames, dtype=float)

    stall = np.zeros(n_frames, dtype=bool)
    if mode != "linear" and n_frames > 1:
        # 线性插值时左端点下标重复是正常的（权重在变），画面并没有"停住"。
        stall[1:] = idx[1:] == idx[:-1]
    runs = _run_lengths(stall[1:])
    run_max = int(np.max(runs)) if runs.size else 1
    n_stalls = int(np.count_nonzero(stall))
    max_hold = float(run_max / float(fps)) if n_stalls else 0.0
    max_hold_playback = float(run_max * int(timing["delay_cs"]) / 100.0) if n_stalls else 0.0

    step_h = np.diff(tt)
    return ExportPlan(
        fps=float(fps),
        delay_cs=int(timing["delay_cs"]),
        actual_fps=float(timing["actual_fps"]),
        mode=mode,
        max_frames=int(max_frames),
        truncated=bool(truncated),
        n_raw=int(n_raw),
        n_frames=int(n_frames),
        source_t=tt.copy(),
        source_y=yy.copy(),
        source_step_t=tt[:-1].copy(),
        source_step_h=step_h,
        frame_t=frame_t,
        content_t=content_t,
        playback_t=playback_t,
        content_index=idx.astype(int),
        weight=weight,
        content_y=content_y,
        timing_error=timing_error,
        playback_error=playback_error,
        state_error=state_error,
        exact_y=exact_y,
        stall=stall,
        n_stalls=n_stalls,
        max_hold=max_hold,
        max_hold_playback=max_hold_playback,
        n_skipped_source=int(tt.size - np.unique(idx).size),
        span=span,
        nominal_duration=float(n_frames / float(fps)),
        playback_duration=float(n_frames * int(timing["delay_cs"]) / 100.0),
        max_abs_timing_error=float(np.max(np.abs(timing_error))) if n_frames else 0.0,
        max_abs_playback_error=float(np.max(np.abs(playback_error))) if n_frames else 0.0,
        max_state_error=float(np.max(state_error)) if n_frames else 0.0,
    )


# --------------------------------------------------------------------------- #
# 目录：依赖图的确定性解析、拓扑序与分层
# --------------------------------------------------------------------------- #
def resolve_dependencies(
    ids: Sequence[str],
    raw_deps: Mapping[str, Sequence[str]],
) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    """展开 ``all``、去重并校验依赖；返回 ``(每章依赖, 问题列表)``。

    ``all`` 的定义是"清单里排在它前面的所有章"——这也让 6.2 的 ``deps: [all]``
    不会意外地依赖自己。未知依赖与自依赖都记进问题列表，而不是静默丢掉。
    """
    order = list(ids)
    rank = {cid: i for i, cid in enumerate(order)}
    if len(rank) != len(order):
        raise ValueError("章节 id 有重复")
    out: dict[str, tuple[str, ...]] = {}
    problems: list[str] = []
    for cid in order:
        seen: list[str] = []
        for raw in raw_deps.get(cid, ()) or ():
            token = str(raw)
            if token == "all":
                cand = order[: rank[cid]]
            elif token == cid:
                problems.append(f"{cid}: 自依赖")
                continue
            elif token in rank:
                cand = [token]
            else:
                problems.append(f"{cid}: 未知依赖 {token}")
                continue
            for dep in cand:
                if dep not in seen:
                    seen.append(dep)
        out[cid] = tuple(seen)
    return out, problems


def topological_order(
    ids: Sequence[str],
    deps: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[str]]:
    """确定性的 Kahn 拓扑排序（同层按清单顺序），返回 ``(顺序, 环上的剩余节点)``。

    不抛异常是对服务层友好的设计：目录坏了也要能画出来并把环标红，而不是让整页 500。
    """
    order = list(ids)
    rank = {cid: i for i, cid in enumerate(order)}
    indeg = {cid: 0 for cid in order}
    children: dict[str, list[str]] = {cid: [] for cid in order}
    for cid in order:
        for dep in deps.get(cid, ()) or ():
            if dep not in rank:
                continue
            indeg[cid] += 1
            children[dep].append(cid)
    ready = [cid for cid in order if indeg[cid] == 0]
    result: list[str] = []
    while ready:
        ready.sort(key=lambda c: rank[c])
        cid = ready.pop(0)
        result.append(cid)
        for child in children[cid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
    remaining = [cid for cid in order if cid not in set(result)]
    return result, remaining


def dependency_depth(
    ids: Sequence[str],
    deps: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    """最长路径深度：根为 0，``depth(c)=1+max depth(dep)``；有环时抛 ``ValueError``。"""
    order, cycles = topological_order(ids, deps)
    if cycles:
        raise ValueError(f"依赖图有环：{', '.join(cycles)}")
    depth: dict[str, int] = {}
    for cid in order:
        parents = [depth[d] for d in deps.get(cid, ()) or () if d in depth]
        depth[cid] = 1 + max(parents) if parents else 0
    return depth


def dependency_layers(
    ids: Sequence[str],
    deps: Mapping[str, Sequence[str]],
) -> list[list[str]]:
    """按深度分层；层内保持清单顺序。"""
    depth = dependency_depth(ids, deps)
    n_layers = max(depth.values(), default=-1) + 1
    layers: list[list[str]] = [[] for _ in range(n_layers)]
    for cid in ids:
        layers[depth[cid]].append(cid)
    return layers


def longest_chain(
    ids: Sequence[str],
    deps: Mapping[str, Sequence[str]],
) -> list[str]:
    """最长依赖链（关键路径）；同长度时取清单顺序更靠前的前驱。"""
    order, cycles = topological_order(ids, deps)
    if cycles:
        raise ValueError(f"依赖图有环：{', '.join(cycles)}")
    rank = {cid: i for i, cid in enumerate(ids)}
    best: dict[str, tuple[int, str | None]] = {}
    for cid in order:
        candidates = [d for d in deps.get(cid, ()) or () if d in best]
        if not candidates:
            best[cid] = (1, None)
            continue
        candidates.sort(key=lambda d: (-best[d][0], rank[d]))
        parent = candidates[0]
        best[cid] = (best[parent][0] + 1, parent)
    if not best:
        return []
    end = sorted(best, key=lambda c: (-best[c][0], rank[c]))[0]
    chain = [end]
    while best[chain[-1]][1] is not None:
        chain.append(str(best[chain[-1]][1]))
    return list(reversed(chain))


# --------------------------------------------------------------------------- #
# 讲义：轻量 TeX 引用/注释解析（不是完整的 TeX 解析器）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CrossRefs:
    """一章 notes 里的标签、引用与 ``\\input`` 目标（都按出现顺序，保留重复）。"""

    labels: tuple[str, ...]
    refs: tuple[str, ...]
    inputs: tuple[str, ...]

    @property
    def duplicate_labels(self) -> tuple[str, ...]:
        seen: set[str] = set()
        dup: list[str] = []
        for name in self.labels:
            if name in seen and name not in dup:
                dup.append(name)
            seen.add(name)
        return tuple(dup)


def strip_tex_comments(text: str) -> str:
    r"""逐行去掉未转义的 ``%`` 注释（``\%`` 是字面百分号，保留）。"""
    out: list[str] = []
    for line in text.splitlines():
        cut = len(line)
        i = 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                cut = i
                break
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


_CMD_RE = r"\\(?P<cmd>label|ref|eqref|pageref|autoref|cref|input|include)\s*\{(?P<arg>[^}]*)\}"


def tex_crossrefs(text: str) -> CrossRefs:
    """提取 ``\\label`` / ``\\ref`` / ``\\input``；注释里的命令不算。"""
    clean = strip_tex_comments(text)
    labels: list[str] = []
    refs: list[str] = []
    inputs: list[str] = []
    for m in re.finditer(_CMD_RE, clean):
        cmd = m.group("cmd")
        arg = m.group("arg").strip()
        if cmd == "label":
            labels.append(arg)
        elif cmd in ("input", "include"):
            inputs.append(arg)
        else:
            refs.append(arg)
    return CrossRefs(tuple(labels), tuple(refs), tuple(inputs))


def asserts_table_rows(text: str) -> int:
    """数断言附表的正文行数（每行以 ``A &`` / ``B &`` / ``C &`` 开头）。

    注释与表头不算；生成器写出的行正是这个形状（见 ``tools/build_book.py``）。
    """
    clean = strip_tex_comments(text)
    return sum(1 for line in clean.splitlines() if re.match(r"^\s*[ABC]\s*&", line))


