"""``.nlf`` 帧容器（Numerical Lab Frame v1）——协议实现见 ``docs/PROTOCOLS.md`` §1。

布局::

    offset 0   : b"NLF1"
    offset 4   : uint32LE  header_len
    offset 8   : header_len 字节 UTF-8 JSON
    offset 8+H : payload（各数组连续紧排，little-endian）

设计要点（也是与前端契约的一部分）：

* **header 恒为明文**，人可以用 ``xxd`` 直接看；只有 payload 允许 zlib/gzip 压缩；
* ``meta`` 只放标量（时间、残差、阶数……），数组一律进 ``arrays``——
  这样"前端偷算一个物理量再塞进图里"就没有藏身处；
* 解码端做**全量自洽校验**（偏移、长度、重叠、dtype、shape 乘积），宁可报错也不返回可疑数据。
"""

from __future__ import annotations

import gzip
import json
import math
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "NLF_MAGIC",
    "NLF_VERSION",
    "DTYPE_CODES",
    "NlfError",
    "ArraySpec",
    "Frame",
    "pack_frame",
    "unpack_frame",
    "write_frame",
    "read_frame",
    "frame_summary",
]

NLF_MAGIC = b"NLF1"
NLF_VERSION = 1
FORMAT_VERSION = "0.1.0"

#: dtype 代号 → numpy 类型（写端只产生 f8/f4/i8/i4/i2/u1/u2/u4/bool）
DTYPE_CODES: dict[str, Any] = {
    "f8": np.float64,
    "f4": np.float32,
    "i8": np.int64,
    "i4": np.int32,
    "i2": np.int16,
    "u1": np.uint8,
    "u2": np.uint16,
    "u4": np.uint32,
    "bool": np.bool_,
}
_CODE_OF: dict[Any, str] = {np.dtype(v): k for k, v in DTYPE_CODES.items()}
MAX_HEADER = 32 * 1024 * 1024  # 32 MiB：正常 header 只有几 KB，超过必是 bug


class NlfError(ValueError):
    """``.nlf`` 容器格式错误（解码端拒绝服务的唯一异常类型）。"""


@dataclass(frozen=True)
class ArraySpec:
    """一个数组的元信息（不含数据本体）。"""

    name: str
    dtype: str
    shape: tuple[int, ...]
    order: str
    offset: int
    nbytes: int
    unit: str = ""
    label: str = ""

    @property
    def itemsize(self) -> int:
        return int(np.dtype(DTYPE_CODES[self.dtype]).itemsize)

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "order": self.order,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "unit": self.unit,
            "label": self.label,
        }


@dataclass
class Frame:
    """一帧数值数据：若干命名数组 + 标量元数据 + 防误导文本。"""

    chapter: str = ""
    viz: str = ""
    kind: str = "frame"
    codec: str = "none"
    meta: dict[str, Any] = field(default_factory=dict)
    strings: dict[str, str] = field(default_factory=dict)
    critical: dict[str, float] = field(default_factory=dict)
    pitfalls: str = ""
    generator: dict[str, str] = field(default_factory=dict)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    _specs: list[ArraySpec] = field(default_factory=list, repr=False)
    _arrays: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #
    # 构造
    # ------------------------------------------------------------------ #
    def add(
        self,
        name: str,
        array: Any,
        *,
        unit: str = "",
        label: str = "",
        dtype: str | None = None,
        order: str = "C",
    ) -> "Frame":
        """加入一个数组（返回 self 以便链式调用）。"""
        if name in self._arrays:
            raise NlfError(f"数组名重复：{name!r}")
        arr = np.asarray(array)
        if dtype is not None:
            arr = arr.astype(DTYPE_CODES[dtype], copy=False)
        elif arr.dtype not in _CODE_OF:
            arr = arr.astype(np.float64)
        code = _CODE_OF[arr.dtype]
        if order == "F":
            arr = np.asfortranarray(arr)
        else:
            arr = np.ascontiguousarray(arr)
        self._arrays[name] = arr
        self._specs.append(
            ArraySpec(
                name=name,
                dtype=code,
                shape=tuple(int(s) for s in arr.shape),
                order=order,
                offset=0,  # 打包时回填
                nbytes=int(arr.nbytes),
                unit=unit,
                label=label,
            )
        )
        return self

    def set_meta(self, **kv: Any) -> "Frame":
        """设置标量元数据：**只允许 number / bool / null**（PROTOCOLS §1.1）。

        字符串请走 :meth:`set_strings`。这条限制是被 TS 解码器当场抓出来的：
        Python 侧原本放行了字符串，于是 `0.0` 章把 "underdamped" 写进 meta，
        `pytest` 全绿、`tsc` 全绿，直到前端真正解码那一帧才炸。
        """
        for k, v in kv.items():
            if isinstance(v, str):
                raise NlfError(
                    f"meta[{k!r}] 是字符串——字符串标量必须走 set_strings()/strings（PROTOCOLS §1.1："
                    f"meta 只允许 number/bool/null）"
                )
            if isinstance(v, (list, tuple, np.ndarray)):
                raise NlfError(f"meta[{k!r}] 是数组——数组必须通过 add() 进入 arrays（PROTOCOLS §1.1）")
            if isinstance(v, (np.floating, np.integer, np.bool_)):
                v = v.item()
            self.meta[k] = v
        return self

    def set_strings(self, **kv: str) -> "Frame":
        """设置字符串标量（状态徽标等）。"""
        self.strings.update({k: str(v) for k, v in kv.items()})
        return self

    # ------------------------------------------------------------------ #
    # 访问
    # ------------------------------------------------------------------ #
    @property
    def names(self) -> list[str]:
        return [s.name for s in self._specs]

    @property
    def specs(self) -> list[ArraySpec]:
        return list(self._specs)

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self._arrays[name]
        except KeyError as exc:
            raise NlfError(f"帧中没有数组 {name!r}；现有：{self.names}") from exc

    def __contains__(self, name: object) -> bool:
        return name in self._arrays

    def scalar(self, key: str, default: float | None = None) -> float | None:
        """取 meta 中的标量（供前端/测试便捷使用）。"""
        v = self.meta.get(key, default)
        return None if v is None else float(v)

    def shape_of(self, name: str) -> tuple[int, ...]:
        for s in self._specs:
            if s.name == name:
                return s.shape
        raise NlfError(f"帧中没有数组 {name!r}")

    def nbytes(self) -> int:
        return int(sum(s.nbytes for s in self._specs))

    # ------------------------------------------------------------------ #
    # 校验
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """写入前自检（与解码端校验对齐）。"""
        if self.kind not in ("frame", "dataset", "manifest"):
            raise NlfError(f"kind 非法：{self.kind!r}")
        if self.codec not in ("none", "zlib", "gzip"):
            raise NlfError(f"codec 非法：{self.codec!r}")
        if not isinstance(self.pitfalls, str):
            raise NlfError("pitfalls 必须是字符串")
        seen: set[str] = set()
        for s in self._specs:
            if s.name in seen:
                raise NlfError(f"数组名重复：{s.name!r}")
            seen.add(s.name)
            if s.dtype not in DTYPE_CODES:
                raise NlfError(f"dtype 非法：{s.dtype!r}")
            if len(s.shape) > 3:
                raise NlfError(f"shape 维数 > 3：{s.name!r} {s.shape}")
            if s.order not in ("C", "F"):
                raise NlfError(f"order 非法：{s.order!r}")
            if s.nbytes != s.size * s.itemsize:
                raise NlfError(f"nbytes 与 shape/dtype 不符：{s.name!r}")
            arr = self._arrays.get(s.name)
            if arr is not None and arr.size and arr.dtype.kind == "f" and not bool(np.all(np.isfinite(arr))):
                # 数组段是裸 float64，NaN/Inf **编码得进去**（不像 meta 那样在 JSON 层就炸），
                # 于是问题会跑到前端：一条 NaN 折线在画布上就是"这段数据凭空消失"，还不报错。
                # 这里的口径与 meta 一致："没有有限值"要写成占位值 + 掩码 + strings 说明
                # （见 PROTOCOLS §1.3 与 3.2 章极点、1.2 章奇异点的处理）。
                n_bad = int(np.count_nonzero(~np.isfinite(arr)))
                raise NlfError(
                    f"数组 {s.name!r} 有 {n_bad}/{arr.size} 个非有限值——"
                    '请写占位值（并在 strings 里说明）+ 掩码数组，或改放 meta 的 null（PROTOCOLS §1.3）'
                )
        for k, v in self.meta.items():
            if isinstance(v, str):
                raise NlfError(f"meta[{k!r}] 是字符串——请改放进 strings（PROTOCOLS §1.1）")
            if isinstance(v, (list, tuple, np.ndarray, dict)) or not isinstance(v, (int, float, bool, type(None))):
                raise NlfError(f"meta[{k!r}] 不是 number/bool/null（PROTOCOLS §1.1）")
            if isinstance(v, float) and not math.isfinite(v):
                # 头用 allow_nan=False 编码，NaN/Inf 会让**整帧**编码失败；而"没有有限值"
                # 与"值很大/很小"在图上必须是两件事，所以这里直接拒绝，逼调用方用 null +
                # strings 里的说明。踩过：1.2 的 unit_ball 在 κ ≳ 1/u 时把 NaN 写进 meta。
                raise NlfError(
                    f"meta[{k!r}] = {v!r} 不是有限数——JSON 没有 NaN/Infinity，"
                    '请用 None（null）表示"无有限值"，并在 strings 里说明原因'
                )
        for k, v in self.critical.items():
            if isinstance(v, (list, tuple)) and not all(isinstance(x, (int, float)) for x in v):
                raise NlfError(f"critical[{k!r}] 必须是数值或数值数组")


def _generator_info() -> dict[str, str]:
    return {
        "tool": "server/nlf.py",
        "format": FORMAT_VERSION,
        "python": ".".join(str(x) for x in __import__("sys").version_info[:3]),
        "numpy": np.__version__,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def pack_frame(frame: Frame, *, codec: str | None = None, compresslevel: int = 6) -> bytes:
    """把 :class:`Frame` 打成 ``.nlf`` 字节串。"""
    frame.validate()
    if codec is not None:
        frame.codec = codec
    chunks: list[bytes] = []
    offset = 0
    specs: list[dict[str, Any]] = []
    for s in frame._specs:  # noqa: SLF001 - 同模块
        arr = frame._arrays[s.name]  # noqa: SLF001
        raw = arr.tobytes(order=s.order)
        if len(raw) != s.nbytes:  # pragma: no cover - 防御
            raise NlfError(f"序列化长度不符：{s.name!r} {len(raw)} != {s.nbytes}")
        spec = s.to_json()
        spec["offset"] = offset
        specs.append(spec)
        chunks.append(raw)
        offset += s.nbytes
    payload = b"".join(chunks)
    if frame.codec == "zlib":
        payload = zlib.compress(payload, compresslevel)
    elif frame.codec == "gzip":
        payload = gzip.compress(payload, compresslevel)
    header: dict[str, Any] = {
        "version": NLF_VERSION,
        "kind": frame.kind,
        "chapter": frame.chapter,
        "viz": frame.viz,
        "codec": frame.codec,
        "endian": "little",
        "generator": frame.generator or _generator_info(),
        "meta": frame.meta,
        "strings": frame.strings,
        "critical": frame.critical,
        "pitfalls": frame.pitfalls,
        "arrays": specs,
    }
    if frame.annotations:
        header["annotations"] = frame.annotations
    hb = json.dumps(header, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(hb) > MAX_HEADER:
        raise NlfError(f"header 过大：{len(hb)} 字节 > {MAX_HEADER}")
    return NLF_MAGIC + struct.pack("<I", len(hb)) + hb + payload


def unpack_frame(buf: bytes | bytearray | memoryview) -> Frame:
    """解析 ``.nlf`` 字节串；任何自洽性问题都抛 :class:`NlfError`。"""
    data = bytes(buf)
    if len(data) < 8:
        raise NlfError(f"数据太短：{len(data)} 字节")
    if data[:4] != NLF_MAGIC:
        raise NlfError(f"magic 不符：{data[:4]!r} != {NLF_MAGIC!r}")
    (hlen,) = struct.unpack_from("<I", data, 4)
    if hlen > MAX_HEADER or 8 + hlen > len(data):
        raise NlfError(f"header_len 非法：{hlen}（总长 {len(data)}）")
    try:
        header = json.loads(data[8 : 8 + hlen].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NlfError(f"header JSON 解析失败：{exc}") from exc
    if not isinstance(header, dict):
        raise NlfError("header 必须是 JSON 对象")
    if header.get("version") != NLF_VERSION:
        raise NlfError(f"版本不支持：{header.get('version')!r}（本实现支持 {NLF_VERSION}）")
    if header.get("endian") != "little":
        raise NlfError(f"endian 非法：{header.get('endian')!r}")
    codec = header.get("codec", "none")
    if codec not in ("none", "zlib", "gzip"):
        raise NlfError(f"codec 非法：{codec!r}")
    payload = data[8 + hlen :]
    if codec == "zlib":
        try:
            payload = zlib.decompress(payload)
        except zlib.error as exc:
            raise NlfError(f"zlib 解压失败：{exc}") from exc
    elif codec == "gzip":
        try:
            payload = gzip.decompress(payload)
        except OSError as exc:
            raise NlfError(f"gzip 解压失败：{exc}") from exc

    fr = Frame(
        chapter=str(header.get("chapter", "")),
        viz=str(header.get("viz", "")),
        kind=str(header.get("kind", "frame")),
        codec=codec,
        meta=dict(header.get("meta", {})),
        strings={k: str(v) for k, v in dict(header.get("strings", {})).items()},
        critical=dict(header.get("critical", {})),
        pitfalls=str(header.get("pitfalls", "")),
        generator=dict(header.get("generator", {})),
        annotations=list(header.get("annotations", [])),
    )
    total = 0
    spans: list[tuple[int, int, str]] = []
    for raw_spec in header.get("arrays", []):
        if not isinstance(raw_spec, dict):
            raise NlfError("arrays 元素必须是对象")
        try:
            spec = ArraySpec(
                name=str(raw_spec["name"]),
                dtype=str(raw_spec["dtype"]),
                shape=tuple(int(x) for x in raw_spec["shape"]),
                order=str(raw_spec.get("order", "C")),
                offset=int(raw_spec["offset"]),
                nbytes=int(raw_spec["nbytes"]),
                unit=str(raw_spec.get("unit", "")),
                label=str(raw_spec.get("label", "")),
            )
        except KeyError as exc:
            raise NlfError(f"arrays 元素缺少字段：{exc}") from exc
        if spec.dtype not in DTYPE_CODES:
            raise NlfError(f"dtype 非法：{spec.dtype!r}")
        if spec.order not in ("C", "F"):
            raise NlfError(f"order 非法：{spec.order!r}")
        if spec.nbytes != spec.size * spec.itemsize:
            raise NlfError(f"nbytes 与 shape/dtype 不符：{spec.name!r}")
        if spec.offset < 0 or spec.offset + spec.nbytes > len(payload):
            raise NlfError(f"数组越界：{spec.name!r} offset={spec.offset} nbytes={spec.nbytes} payload={len(payload)}")
        spans.append((spec.offset, spec.offset + spec.nbytes, spec.name))
        total += spec.nbytes
        np_dtype = DTYPE_CODES[spec.dtype]
        flat = np.frombuffer(payload, dtype=np_dtype, count=spec.size, offset=spec.offset)
        arr = flat.reshape(spec.shape, order=spec.order).copy()
        fr._arrays[spec.name] = arr  # noqa: SLF001
        fr._specs.append(spec)  # noqa: SLF001
    if total != len(payload):
        raise NlfError(f"payload 长度与数组总和不符：{len(payload)} != {total}")
    spans.sort()
    for (s0, e0, n0), (s1, _, n1) in zip(spans, spans[1:]):
        if e0 > s1:
            raise NlfError(f"数组区间重叠：{n0!r} 与 {n1!r}")
    return fr


def write_frame(path: str | Path, frame: Frame, *, codec: str | None = None) -> Path:
    """把帧写盘（``codec`` 缺省沿用帧自身设置）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pack_frame(frame, codec=codec))
    return p


def read_frame(path: str | Path) -> Frame:
    """从磁盘读取帧。"""
    return unpack_frame(Path(path).read_bytes())


def frame_summary(frame: Frame) -> str:
    """一帧的人类可读摘要（日志/报告用）。"""
    parts = [
        f"<Frame {frame.chapter}/{frame.viz} kind={frame.kind} codec={frame.codec} "
        f"arrays={len(frame._specs)} bytes={frame.nbytes()}>"
    ]
    for s in frame._specs:
        parts.append(f"  · {s.name}: {s.dtype}{list(s.shape)} order={s.order} {s.unit} {s.label}".rstrip())
    if frame.meta:
        parts.append(f"  meta: {frame.meta}")
    return "\n".join(parts)


def iter_arrays(frame: Frame) -> Iterable[tuple[ArraySpec, np.ndarray]]:
    """遍历 ``(spec, array)`` 对。"""
    for s in frame._specs:
        yield s, frame._arrays[s.name]


def manifest_frame(chapter: str, viz: str, payload: Mapping[str, Any]) -> Frame:
    """构造 ``kind="manifest"`` 帧（把 manifest 走同一条通道送到前端）。"""
    fr = Frame(chapter=chapter, viz=viz, kind="manifest")
    fr.set_meta(version=1, n_viz=len(payload.get("vizzes", [])) if isinstance(payload, Mapping) else 0)
    fr.strings["manifest_json"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return fr
