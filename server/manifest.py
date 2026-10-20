"""Manifest 校验：把"防误导七条规则"变成**无法绕过**的数据结构约束。

真源在 ``server/chapters/<id>.py::MANIFEST``（Python），生成的 TS/JSON 只是它的影子。
本模块是那个真源的守门人：``load_manifest`` 不通过就不许生成任何东西。

为什么把规则写成硬校验而不是写进文档：
    文档会被略过，"忘记写 criticalNote"却会让 CI 直接红——这才是规则能活下来的原因。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ManifestError",
    "ParamSpec",
    "AxisSpec",
    "TimeSpec",
    "VizSpec",
    "ChapterManifest",
    "validate_manifest",
    "load_manifest",
    "MIN_PITFALL_CHARS",
]

CHAPTER_ID_RE = re.compile(r"^\d+\.\d+$")
VIZ_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
PARAM_TYPES = ("int", "float", "bool", "enum")
TIME_MODES = ("none", "stream", "scrub")
DATA_SOURCES = ("backend", "backend+preview")

#: 规则 7：pitfalls 至少 60 字（中文按字符计）。少于这个长度基本等于没说。
MIN_PITFALL_CHARS = 60


class ManifestError(ValueError):
    """Manifest 不合规；``problems`` 给出全部问题（不是第一个）。"""

    def __init__(self, problems: Sequence[str], *, where: str = "") -> None:
        self.problems = list(problems)
        head = f"{where}：" if where else ""
        super().__init__(head + f"manifest 有 {len(self.problems)} 处不合规：\n  - " + "\n  - ".join(self.problems))


@dataclass(frozen=True)
class ParamSpec:
    """一个交互参数的契约（含规则 6 要求的理论临界值）。"""

    name: str
    label: str
    type: str
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple[str, ...] = ()
    unit: str = ""
    critical: tuple[float, ...] = ()
    critical_note: str = ""
    beyond: str = "above"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "critical": list(self.critical),
            "criticalNote": self.critical_note,
        }
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.step is not None:
            out["step"] = self.step
        if self.options:
            out["options"] = list(self.options)
        if self.unit:
            out["unit"] = self.unit
        if self.beyond != "above":
            out["beyond"] = self.beyond
        return out


@dataclass(frozen=True)
class AxisSpec:
    """一个坐标轴的声明（规则 1/4：``autoscaled`` 必须显式给出）。"""

    id: str
    autoscaled: bool
    title: str = ""
    ylabel: str = ""
    scale: str = "linear"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "autoscaled": self.autoscaled}
        if self.title:
            out["title"] = self.title
        if self.ylabel:
            out["ylabel"] = self.ylabel
        if self.scale != "linear":
            out["scale"] = self.scale
        return out


@dataclass(frozen=True)
class TimeSpec:
    """时间轴契约（``mode="none"`` 时 MUST NOT 出现时间控件）。"""

    mode: str = "none"
    t_max: float | None = None
    dt: float | None = None
    fps: int | None = None
    label: str = "t"
    t_max_param: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode}
        for key, val in (
            ("tMax", self.t_max),
            ("dt", self.dt),
            ("fps", self.fps),
            ("label", self.label),
            ("tMaxParam", self.t_max_param),
        ):
            if val is not None:
                out[key] = val
        return out

    @property
    def n_frames(self) -> int:
        if self.mode == "none" or not self.t_max or not self.dt:
            return 1
        return int(round(self.t_max / self.dt)) + 1


@dataclass(frozen=True)
class VizSpec:
    """一个可视化的完整契约。"""

    id: str
    title: str
    data_source: str
    params: tuple[ParamSpec, ...]
    time: TimeSpec
    required_aux: tuple[str, ...]
    axes: tuple[AxisSpec, ...]
    pitfalls: str
    theory: tuple[str, ...]
    description: str = ""
    badges: tuple[str, ...] = ()

    def param(self, name: str) -> ParamSpec:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"可视化 {self.id} 没有参数 {name}")

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "dataSource": self.data_source,
            "params": [p.to_json() for p in self.params],
            "time": self.time.to_json(),
            "requiredAux": list(self.required_aux),
            "axes": [a.to_json() for a in self.axes],
            "pitfalls": self.pitfalls,
            "theory": list(self.theory),
        }
        if self.description:
            out["description"] = self.description
        if self.badges:
            out["badges"] = list(self.badges)
        return out


@dataclass(frozen=True)
class ChapterManifest:
    """一章的 manifest。"""

    chapter: str
    title: str
    vizzes: tuple[VizSpec, ...]
    part: str = ""
    summary: str = ""

    def viz(self, viz_id: str) -> VizSpec:
        for v in self.vizzes:
            if v.id == viz_id:
                return v
        raise KeyError(f"章节 {self.chapter} 没有可视化 {viz_id}")

    @property
    def viz_ids(self) -> tuple[str, ...]:
        return tuple(v.id for v in self.vizzes)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "chapter": self.chapter,
            "title": self.title,
            "vizzes": [v.to_json() for v in self.vizzes],
        }
        if self.part:
            out["part"] = self.part
        if self.summary:
            out["summary"] = self.summary
        return out

    def iter_params(self) -> Iterable[tuple[VizSpec, ParamSpec]]:
        for v in self.vizzes:
            for p in v.params:
                yield v, p


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_params(viz_id: str, raw: Any, problems: list[str]) -> None:
    if not isinstance(raw, list) or len(raw) == 0:
        problems.append(f"{viz_id}: params 必须是非空列表（至少要有一个参数，或显式写 [] 并说明理由）")
        return
    seen: set[str] = set()
    for i, p in enumerate(raw):
        tag = f"{viz_id}.params[{i}]"
        if not isinstance(p, Mapping):
            problems.append(f"{tag}: 必须是对象")
            continue
        name = p.get("name")
        if not isinstance(name, str) or not name:
            problems.append(f"{tag}: name 必须是非空字符串")
            continue
        tag = f"{viz_id}.{name}"
        if name in seen:
            problems.append(f"{tag}: 参数名重复")
        seen.add(name)
        if not isinstance(p.get("label"), str) or not p.get("label"):
            problems.append(f"{tag}: 缺少 label（界面上的中文名）")
        ptype = p.get("type")
        if ptype not in PARAM_TYPES:
            problems.append(f"{tag}: type 必须是 {PARAM_TYPES} 之一，得到 {ptype!r}")
            continue
        if "critical" not in p:
            problems.append(f"{tag}: 缺少 critical（没有理论临界值也要显式写 []）——规则 6")
        else:
            crit = p["critical"]
            if not isinstance(crit, (list, tuple)):
                problems.append(f"{tag}: critical 必须是列表")
            else:
                for c in crit:
                    if not _is_number(c):
                        problems.append(f"{tag}: critical 的元素必须是数字，得到 {c!r}")
                if crit and not str(p.get("criticalNote") or "").strip():
                    problems.append(f"{tag}: critical 非空时 criticalNote 必须写明依据——规则 6")
        beyond = p.get("beyond", "above")
        if beyond not in ("above", "below", "magnitude"):
            problems.append(f"{tag}: beyond 必须是 above/below/magnitude，得到 {beyond!r}")
        if ptype == "enum":
            opts = p.get("options")
            if not isinstance(opts, (list, tuple)) or len(opts) < 2:
                problems.append(f"{tag}: enum 至少要有两个 options")
            elif p.get("default") not in opts:
                problems.append(f"{tag}: default={p.get('default')!r} 不在 options 里")
        else:
            if "default" not in p:
                problems.append(f"{tag}: 缺少 default")
            lo, hi = p.get("min"), p.get("max")
            if lo is not None and not _is_number(lo):
                problems.append(f"{tag}: min 必须是数字")
            if hi is not None and not _is_number(hi):
                problems.append(f"{tag}: max 必须是数字")
            if _is_number(lo) and _is_number(hi) and not lo < hi:
                problems.append(f"{tag}: 需要 min < max，得到 [{lo}, {hi}]")
            d = p.get("default")
            if ptype == "int" and not isinstance(d, int):
                problems.append(f"{tag}: int 参数的 default 必须是整数")
            if ptype == "float" and not _is_number(d):
                problems.append(f"{tag}: float 参数的 default 必须是数字")
            if ptype == "bool" and not isinstance(d, bool):
                problems.append(f"{tag}: bool 参数的 default 必须是 true/false")
            if _is_number(d):
                if _is_number(lo) and d < lo:
                    problems.append(f"{tag}: default={d} 小于 min={lo}")
                if _is_number(hi) and d > hi:
                    problems.append(f"{tag}: default={d} 大于 max={hi}")


def _check_time(viz_id: str, raw: Any, problems: list[str], params: Any = None) -> None:
    if raw is None:
        problems.append(f"{viz_id}: 缺少 time 字段（没有时间轴也要写 {{\"mode\": \"none\"}}）")
        return
    if not isinstance(raw, Mapping):
        problems.append(f"{viz_id}: time 必须是对象")
        return
    mode = raw.get("mode")
    if mode not in TIME_MODES:
        problems.append(f"{viz_id}: time.mode 必须是 {TIME_MODES} 之一，得到 {mode!r}")
        return
    if mode == "none":
        extra = {"tMax", "dt", "fps", "tMaxParam"} & set(raw)
        if extra:
            problems.append(f"{viz_id}: time.mode='none' 时不得声明时间控件 {sorted(extra)}——规则 4")
        return
    if not _is_number(raw.get("tMax")) or not (raw["tMax"] > 0):
        problems.append(f"{viz_id}: time.mode={mode!r} 需要正的 tMax")
    if not _is_number(raw.get("dt")) or not (raw["dt"] > 0):
        problems.append(f"{viz_id}: time.mode={mode!r} 需要正的 dt")
    if _is_number(raw.get("tMax")) and _is_number(raw.get("dt")) and raw["dt"] > raw["tMax"]:
        problems.append(f"{viz_id}: dt 不应大于 tMax（会只有一帧）")
    fps = raw.get("fps")
    if fps is not None and (not isinstance(fps, int) or not 1 <= fps <= 120):
        problems.append(f"{viz_id}: fps 必须是 1..120 的整数")
    cap = raw.get("tMaxParam")
    if cap is not None:
        if not isinstance(cap, str) or not cap.strip():
            problems.append(f"{viz_id}: time.tMaxParam 必须是非空参数名")
        else:
            pmap = {
                p.get("name"): p
                for p in (params if isinstance(params, (list, tuple)) else ())
                if isinstance(p, Mapping)
            }
            p = pmap.get(cap)
            if p is None:
                problems.append(f"{viz_id}: time.tMaxParam 指向不存在的参数 {cap!r}")
            elif p.get("type") not in ("int", "float"):
                problems.append(f"{viz_id}: time.tMaxParam 只能指向 int/float 参数，{cap} 是 {p.get('type')!r}")


def validate_manifest(raw: Mapping[str, Any], *, where: str = "") -> list[str]:
    """校验 manifest 原字典，返回问题列表（空 = 合规）。"""
    problems: list[str] = []
    if not isinstance(raw, Mapping):
        return ["manifest 必须是 dict"]
    chapter = raw.get("chapter")
    if not isinstance(chapter, str) or not CHAPTER_ID_RE.match(chapter):
        problems.append(f"chapter 必须是 '主版本.次版本' 形式（如 '1.2'），得到 {chapter!r}")
    if not str(raw.get("title") or "").strip():
        problems.append("title 必须是非空字符串")
    vizzes = raw.get("vizzes")
    if not isinstance(vizzes, (list, tuple)) or len(vizzes) == 0:
        return problems + ["vizzes 必须是非空列表"]
    seen: set[str] = set()
    for i, v in enumerate(vizzes):
        if not isinstance(v, Mapping):
            problems.append(f"vizzes[{i}] 必须是对象")
            continue
        vid = v.get("id")
        if not isinstance(vid, str) or not VIZ_ID_RE.match(vid):
            problems.append(f"vizzes[{i}]: id 必须是小写下划线标识符，得到 {vid!r}")
            continue
        if vid in seen:
            problems.append(f"{vid}: 同一章内 id 必须唯一")
        seen.add(vid)
        if not str(v.get("title") or "").strip():
            problems.append(f"{vid}: 缺少 title")
        if v.get("dataSource") not in DATA_SOURCES:
            problems.append(f"{vid}: dataSource 必须是 {DATA_SOURCES} 之一，得到 {v.get('dataSource')!r}")
        _check_params(vid, v.get("params"), problems)
        _check_time(vid, v.get("time"), problems, v.get("params"))
        aux = v.get("requiredAux")
        if not isinstance(aux, (list, tuple)) or len(aux) == 0:
            problems.append(f"{vid}: requiredAux 必须非空——规则 3 要求给出误差或残差场")
        elif not ({"error", "residual"} & set(aux)):
            problems.append(f"{vid}: requiredAux 必须含 'error' 或 'residual'，得到 {list(aux)}——规则 3")
        axes = v.get("axes")
        if not isinstance(axes, (list, tuple)) or len(axes) == 0:
            problems.append(f"{vid}: axes 必须非空")
        else:
            ax_ids: set[str] = set()
            for j, a in enumerate(axes):
                if not isinstance(a, Mapping):
                    problems.append(f"{vid}.axes[{j}]: 必须是对象")
                    continue
                aid = a.get("id")
                if not isinstance(aid, str) or not aid:
                    problems.append(f"{vid}.axes[{j}]: 缺少 id")
                    continue
                if aid in ax_ids:
                    problems.append(f"{vid}.axes: id 重复 {aid!r}")
                ax_ids.add(aid)
                if "autoscaled" not in a or not isinstance(a["autoscaled"], bool):
                    problems.append(f"{vid}.axes[{aid}]: 必须显式声明 autoscaled（true/false）——规则 1/4")
                if a.get("scale", "linear") not in ("linear", "log"):
                    problems.append(f"{vid}.axes[{aid}]: scale 只能是 linear/log")
        pitfalls = v.get("pitfalls")
        if not isinstance(pitfalls, str):
            # 类型错与长度错必须给不同的信息。踩过：作者把 pitfalls 写成 list，
            # 而旧消息只报"至少 60 字（现在 295 字）"——自相矛盾，看的人完全不知道该改什么。
            problems.append(
                f"{vid}: pitfalls 必须是**字符串**（现在是 {type(pitfalls).__name__}）"
                f"——规则 7 要求把'它会怎样骗你'写成一段正文，而不是列表"
            )
        elif len(pitfalls.strip()) < MIN_PITFALL_CHARS:
            problems.append(
                f"{vid}: pitfalls 至少 {MIN_PITFALL_CHARS} 字（现在 {len(pitfalls.strip())} 字）"
                "——规则 7 要求说清'它会怎样骗你'"
            )
        theory = v.get("theory")
        if not isinstance(theory, (list, tuple)) or len(theory) == 0:
            problems.append(f"{vid}: theory 必须给出至少一条理论依据（notes.tex 里要有对应推导）")
    return problems


def load_manifest(raw: Mapping[str, Any], *, where: str = "") -> ChapterManifest:
    """校验并转成强类型对象；不合规直接抛 :class:`ManifestError`。"""
    problems = validate_manifest(raw, where=where)
    if problems:
        raise ManifestError(problems, where=where)
    vizzes: list[VizSpec] = []
    for v in raw["vizzes"]:
        params = tuple(
            ParamSpec(
                name=p["name"],
                label=p["label"],
                type=p["type"],
                default=p.get("default"),
                min=p.get("min"),
                max=p.get("max"),
                step=p.get("step"),
                options=tuple(p.get("options", ())),
                unit=p.get("unit", ""),
                critical=tuple(float(c) for c in p.get("critical", ())),
                critical_note=p.get("criticalNote", ""),
                beyond=p.get("beyond", "above"),
            )
            for p in v["params"]
        )
        t = v["time"]
        time = TimeSpec(
            mode=t["mode"],
            t_max=t.get("tMax"),
            dt=t.get("dt"),
            fps=t.get("fps"),
            label=t.get("label", "t"),
            t_max_param=t.get("tMaxParam"),
        )
        axes = tuple(
            AxisSpec(id=a["id"], autoscaled=bool(a["autoscaled"]), title=a.get("title", ""), ylabel=a.get("ylabel", ""), scale=a.get("scale", "linear"))
            for a in v["axes"]
        )
        vizzes.append(
            VizSpec(
                id=v["id"],
                title=v["title"],
                data_source=v["dataSource"],
                params=params,
                time=time,
                required_aux=tuple(v["requiredAux"]),
                axes=axes,
                # 画布上的 pitfalls 面板没有 markdown 解析器：留着 ** 会原样画出两个星号。
                # 只清洗这一段（description/theory 还供书稿渲染），长度校验仍按原文做。
                pitfalls=v["pitfalls"].replace("**", ""),
                theory=tuple(v["theory"]),
                description=v.get("description", ""),
                badges=tuple(v.get("badges", ())),
            )
        )
    return ChapterManifest(
        chapter=raw["chapter"],
        title=raw["title"],
        vizzes=tuple(vizzes),
        part=raw.get("part", ""),
        summary=raw.get("summary", ""),
    )
