"""帧装配层：参数规范化 → 调用章节 ``compute`` → 契约体检 → ``.nlf`` 帧。

为什么体检要放在这里而不是信任章节实现：
    规则 3（必须有误差/残差场）、规则 6（临界值随参数进帧）、规则 7（pitfalls 进帧）
    都是"前端要显示、后端必须给"的东西。放在唯一的出帧口检查，就没有第二个地方能漏。
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import pathlib
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from server.manifest import ChapterManifest, ParamSpec, VizSpec
from server.nlf import Frame, NlfError
from server.registry import ChapterEntry, Registry

__all__ = [
    "ParamError",
    "normalize_params",
    "params_hash",
    "governing_critical",
    "build_frame",
    "FrameCache",
    "precompute_chapter",
    "static_frame_path",
]


class ParamError(ValueError):
    """参数不合规（未知参数、类型错误、越界）。"""


def governing_critical(param: ParamSpec, value: float | None = None) -> float | None:
    """在 ``param.critical`` 里挑出"当前位置最该被提及"的那一个（与 TS 侧同一口径）。

    语义是"**你现在越过了哪一道线**"，也就是最近越过的那条：

    * ``above``（缺省）：已越过的阈值里**最大**的那个；一条都没越过 → 最小的阈值（下一个目标）；
    * ``below``：已越过的阈值里**最小**的那个；一条都没越过 → 最大的阈值；
    * ``magnitude``：绝对值最接近 ``value`` 的那个。

    ``value=None``（未知当前值）时给"最先会被撞上"的那条：``above`` 给最小、``below`` 给最大。
    """
    crit = sorted(param.critical)
    if not crit:
        return None
    if value is None:
        return crit[0] if param.beyond != "below" else crit[-1]
    if param.beyond == "above":
        crossed = [c for c in crit if value > c]
        return crossed[-1] if crossed else crit[0]
    if param.beyond == "below":
        crossed = [c for c in crit if value < c]
        return crossed[0] if crossed else crit[-1]
    return min(crit, key=lambda c: abs(abs(c) - abs(value)))


def _coerce(param: ParamSpec, value: Any) -> Any:
    """按 manifest 声明的类型转换并做区间校验（越界**报错**，不静默夹取）。"""
    if param.type == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not float(value).is_integer():
            raise ParamError(f"参数 {param.name} 需要整数，得到 {value!r}")
        out: Any = int(value)
    elif param.type == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParamError(f"参数 {param.name} 需要数字，得到 {value!r}")
        out = float(value)
        if not math.isfinite(out):
            raise ParamError(f"参数 {param.name} 必须是有限数，得到 {value!r}")
    elif param.type == "bool":
        if not isinstance(value, bool):
            raise ParamError(f"参数 {param.name} 需要布尔值，得到 {value!r}")
        out = value
    elif param.type == "enum":
        if value not in param.options:
            raise ParamError(f"参数 {param.name} 必须是 {list(param.options)} 之一，得到 {value!r}")
        out = value
    else:  # pragma: no cover - manifest 校验已挡住
        raise ParamError(f"参数 {param.name} 的类型 {param.type} 未知")
    if param.min is not None and isinstance(out, (int, float)) and not isinstance(out, bool) and out < param.min:
        raise ParamError(f"参数 {param.name}={out} 小于下界 {param.min}")
    if param.max is not None and isinstance(out, (int, float)) and not isinstance(out, bool) and out > param.max:
        raise ParamError(f"参数 {param.name}={out} 大于上界 {param.max}")
    return out


def normalize_params(viz: VizSpec, raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """把用户给的参数补全为完整、类型正确的参数字典。"""
    raw = dict(raw or {})
    known = {p.name for p in viz.params}
    unknown = set(raw) - known
    if unknown:
        raise ParamError(f"可视化 {viz.id} 不认识参数 {sorted(unknown)}（可用：{sorted(known)}）")
    return {p.name: _coerce(p, raw[p.name]) if p.name in raw else p.default for p in viz.params}


def _canonical(params: Mapping[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def params_hash(chapter: str, viz: str, params: Mapping[str, Any]) -> str:
    """参数指纹（用于静态预计算的稳定文件名）。"""
    h = hashlib.sha256()
    h.update(f"{chapter}/{viz}/".encode())
    h.update(_canonical(params).encode())
    return h.hexdigest()[:16]


def build_frame(
    entry: ChapterEntry,
    viz_id: str,
    params: Mapping[str, Any] | None = None,
    t: float = 0.0,
    *,
    strict: bool = True,
) -> Frame:
    """调用章节 ``compute`` 并把结果装成合规的 :class:`Frame`。

    ``strict=True``（缺省）会体检：``requiredAux`` 是否真的给了、``meta`` 是否全是标量、
    返回值是否是 Frame。这些检查失败即抛 :class:`NlfError`——宁可当场炸，也不要前端画一张空图。
    """
    manifest: ChapterManifest = entry.manifest
    viz = manifest.viz(viz_id)
    norm = normalize_params(viz, params)
    raw = entry.compute(viz_id, norm, t)
    if not isinstance(raw, Frame):
        raise NlfError(f"章节 {entry.id} 的 compute({viz_id}) 必须返回 Frame，得到 {type(raw).__name__}")
    fr = raw
    fr.chapter = entry.id
    fr.viz = viz_id
    if fr.kind == "frame" and not fr.strings.get("data_source"):
        fr.set_strings(data_source=viz.data_source)
    fr.set_strings(params_json=_canonical(norm))
    fr.set_meta(t=float(t))
    fr.pitfalls = viz.pitfalls
    crit: dict[str, float] = {}
    for p in viz.params:
        g = governing_critical(p, norm.get(p.name) if not isinstance(norm.get(p.name), str) else None)
        if g is not None:
            crit[p.name] = float(g)
    if crit:
        fr.critical = crit
    if strict:
        names = {n.lower() for n in fr.names}
        need = [a for a in viz.required_aux if a in ("error", "residual")]
        for aux in need:
            if not any(aux in n for n in names):
                raise NlfError(
                    f"章节 {entry.id} 的 {viz_id} 声明了 requiredAux={list(viz.required_aux)}，"
                    f"但帧里没有名字含 {aux!r} 的数组（现有：{sorted(fr.names)}）——规则 3"
                )
        for key, val in fr.meta.items():
            if isinstance(val, (list, tuple, dict)):
                raise NlfError(f"meta[{key!r}] 是容器（{type(val).__name__}）——meta 只允许标量（PROTOCOLS §1.1）")
    return fr


# --------------------------------------------------------------------------- #
# 缓存与静态预计算
# --------------------------------------------------------------------------- #
@dataclass
class FrameCache:
    """按 (chapter, viz, params, t) 缓存帧字节的有界 LRU。

    有界是必须的：一帧热力图可能几 MB，无界缓存会把服务进程吃干。

    **线程安全**：HTTP 服务器是多线程的（``ThreadingHTTPServer``），两个请求同时
    落进 :meth:`put` 的淘汰循环会同时改 ``_items``/``_bytes``——轻则计数错乱、
    重则 ``RuntimeError: dictionary changed size during iteration``。所以所有读写都在
    一把锁里；锁只包住字典操作，**不包**算帧（算帧在调用方，绝不持锁做重活）。
    """

    max_entries: int = 64
    max_bytes: int = 256 * 1024 * 1024
    _items: dict[str, bytes] = field(default_factory=dict)
    _bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def key(self, chapter: str, viz: str, params: Mapping[str, Any], t: float) -> str:
        return f"{chapter}/{viz}/{params_hash(chapter, viz, params)}/t={t:.6g}"

    @property
    def n_entries(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def n_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._items.get(key)

    def put(self, key: str, blob: bytes) -> None:
        if len(blob) > self.max_bytes:
            return  # 单帧就超预算：不缓存，直接每次重算
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= len(old)
            self._items[key] = blob
            self._bytes += len(blob)
            while self._items and (len(self._items) > self.max_entries or self._bytes > self.max_bytes):
                evicted, evicted_blob = next(iter(self._items.items()))
                del self._items[evicted]
                self._bytes -= len(evicted_blob)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0


def static_frame_path(root: pathlib.Path, chapter: str, viz: str, params: Mapping[str, Any], t: float = 0.0) -> pathlib.Path:
    """静态预计算帧的落盘路径（L3 降级通道用）。"""
    return root / "data" / "frames" / chapter / viz / f"{params_hash(chapter, viz, params)}_t{t:.6g}.nlf"


def precompute_chapter(
    registry: Registry,
    chapter_id: str,
    *,
    out_root: pathlib.Path | None = None,
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    frames_per_viz: int = 1,
    limit: int | None = None,
) -> list[pathlib.Path]:
    """把一章的帧预先算好写到 ``data/frames/...``。

    ``param_grid`` 缺省时只算"缺省参数"的 ``frames_per_viz`` 帧（时间轴等间隔）。
    ``limit`` 用于在 CI 里限制总帧数（防止一个不小心就写出几万个文件）。
    """
    root = registry.root if out_root is None else out_root
    entry = registry.get(chapter_id)
    written: list[pathlib.Path] = []
    for viz in entry.manifest.vizzes:
        base = normalize_params(viz, param_grid or {})
        n = 1 if viz.time.mode == "none" else max(1, frames_per_viz)
        t_max = viz.time.t_max or 0.0
        for i in range(n):
            t = 0.0 if n == 1 else t_max * i / (n - 1)
            fr = build_frame(entry, viz.id, base, t)
            blob = _pack(fr)
            path = static_frame_path(root, chapter_id, viz.id, base, t)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            written.append(path)
            if limit is not None and len(written) >= limit:
                return written
    return written


def _pack(fr: Frame) -> bytes:
    from server.nlf import pack_frame

    return pack_frame(fr)


@functools.lru_cache(maxsize=1)
def _noop() -> None:  # pragma: no cover - 占位，保持模块级 lru_cache 的用法一致
    return None


def iter_all_frames(registry: Registry) -> Iterable[tuple[str, str]]:
    """遍历 (chapter, viz)，供工具与 CI 使用。"""
    return registry.iter_viz()
