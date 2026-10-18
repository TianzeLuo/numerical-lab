"""零依赖 HTTP + WebSocket 服务：``docs/PROTOCOLS.md`` §3（HTTP API）与 §2（WS 流）的执行体。

为什么是标准库：``docs/POLICY.md`` §1 的运行期依赖红线里没有 fastapi/uvicorn/websockets，
所以 HTTP 直接建在 ``http.server`` 上，WS 复用自研的 :mod:`server.ws`（RFC 6455 子集）。

启动::

    python3 -m server.app --port 8080 --host 127.0.0.1 [--allow-write] [--data-dir DIR]

路由（逐条对应 PROTOCOLS §3）::

    GET  /                                  → viz/site/index.html（站点壳）
    GET  /viz/...  /dist/viz/...  /data/...  → 静态资源（防目录穿越，见 resolve_static）
    GET  /api/health                        → {"ok": true, "chapters": N, "version": "..."}
    GET  /api/chapters                      → 章节清单（chapters.yaml + 进度 + 依赖）
    GET  /api/chapter/<id>/manifest          → 该章全部 viz 的 manifest
    GET  /api/chapter/<id>/frame?viz=&params=&t=   → 单个 .nlf
    GET  /api/chapter/<id>/frames?viz=&params=&n=  → 多帧流（拼接格式见下）
    GET|POST /api/asserts/<id>               → 断言报告（**仅** --allow-write；否则 403）
    GET  /ws/chapter/<id>                    → WebSocket（握手后先发 ready+manifest）

``params`` 传输：JSON → UTF-8 → base64url（补位可省）；同时接受 ``params_json`` 便于手工调试
（PROTOCOLS §3 的两行都实现）。参数错误一律 400 且消息里带**可用参数名**——诊断信息来自
:class:`server.frames.ParamError`，不在这一层重新发明。

``t`` 的语义（PROTOCOLS §3）：``/frame`` 的 ``t`` 缺省为 0，接受 ``%.6g`` 写法
（``t=0.05``、``t=1.5e-05``）。解析后**对齐到 ``TimeSpec.dt`` 的最近整步**
（:func:`align_t`），再把时间轴三件套写进响应帧的 meta：

* ``meta.t`` / ``meta.t_actual`` = 实到时刻（对齐后的格点，写法与 L3 文件名里的 ``%.6g`` 一致）；
* ``meta.t_requested`` = 请求原值（未对齐时才有意义；它总是存在，对齐时等于实到时刻）。

对齐不是洁癖：L2 与 L3 必须指向同一个时刻，否则前端拖到 0.103 时，动态通道回一帧
0.103、静态通道回文件名里的 0.1，两条通道就会各说各话。对齐后 ``/frame`` 的响应
与 ``data/frames/<chapter>/<viz>/<hash>_t<%.6g>.nlf`` 逐字对应。

多帧流 ``/frames`` 的拼接格式（自描述，v1，模块提供 print/parse 参考实现）::

    offset 0 : 4 字节   magic  = b"NLFS"
    offset 4 : uint32BE version = 1
    offset 8 : uint32BE header_len
    offset 12: header_len 字节 UTF-8 JSON，形如
               {"format":"nlf-stream","version":1,"chapter":"0.0","viz":"heat_ftcs",
                "n":5,"params":{...},"t":[0.0,0.05,0.1,0.15,0.2]}
    之后 n 条记录，每条：uint32BE frame_len | frame_len 字节的完整 ``.nlf``

所有 u32 都是**大端**（网络序），与 PROTOCOLS §3 里"长度前缀拼接（4 字节大端长度）"的草拟
约定一致；注意 ``.nlf`` v1 容器自己的 ``header_len`` 是**小端**，两种容器别混。

为什么不用 ``multipart/mixed``：multipart 需要 boundary 解析与 MIME 头，浏览器端在裸
``ArrayBuffer`` 里切分很别扭；长度前缀两行 ``struct`` 就能双向解析。而"自描述"落在
JSON 头上：chapter/viz/n/params/t 全在里面，客户端拿到一段字节就能独立解释它，
不必回看请求上下文。``Content-Type: application/octet-stream`` + ``X-Nlf-Stream: NLFS/1``。

**Content-Type 的一处取舍**：单帧用 ``application/x-nlf``（PROTOCOLS §3 表格的写法），
不是 ``application/octet-stream``——多帧流因为是自定义容器才用 octet-stream。

线程模型与缓存
--------------

``ThreadingHTTPServer`` + ``daemon_threads``：每连接一个线程，计算在该连接线程内**同步**执行
（与 PROTOCOLS §2.3 的"单连接单线程"一致）。因此 WS 的 ``play``/``step`` 是同步推流：

* ``pause``/``cancel`` 只能作用于**流与流之间**（同步流跑的时候不读 socket，否则会把
  帧边界读坏——``server/ws.py`` 明确禁止在超时后继续 recv）；
* 返回 ``done``（``frames_sent`` 为本次被丢弃/未推送的帧数），不新增 PROTOCOLS §2.2 之外的消息类型。

帧缓存是**进程共享**的有界 :class:`server.frames.FrameCache`：帧字节不可变，同一组参数会被
反复请求，每连接一份缓存等于把命中率除以连接数。``FrameCache`` 自带锁（读写都在锁里，
**算帧在锁外**），所以这里不再自己包一层。缓存键由 :func:`_frame_cache_key` 生成：

* 多帧流的**栅格帧**用 ``server.frames.FrameCache.key`` 的原样键；
* ``/frame``/``seek`` 的**点名帧**多带 ``req:``/``req-raw:`` 命名空间前缀——响应字节里含
  ``meta.t_requested``，与栅格帧不是同一份内容，绝不能共用同一个键。

HTTP 层最容易踩的两个坑（都在下面显式处理）：

1. **目录穿越**：``..`` / 绝对路径 / 符号链接逃逸一律 403（见 :func:`resolve_static`）；
2. **升级 WS 时 ``rfile`` 预读**：``http.server`` 的 ``rfile`` 是带缓冲的读取器，
   一次 ``read()`` 可能把紧随握手请求头的首帧字节一起吞掉，必须交给
   ``WebSocket.accept(initial_bytes=...)``；但 ``rfile.peek()`` 在缓冲区为空时会**阻塞**
   （等下一批字节），所以先把 socket 切到非阻塞再 peek，见 :meth:`_Handler._initial_ws_bytes`。
"""

from __future__ import annotations

import argparse
import base64
import binascii
import itertools
import json
import logging
import math
import pathlib
import re
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, unquote, urlsplit

from server import __version__
from server.frames import FrameCache, ParamError, build_frame, normalize_params
from server.manifest import VizSpec
from server.nlf import Frame, NlfError, pack_frame, unpack_frame
from server.registry import ChapterEntry, Registry, RegistryError, load_chapter_yaml
from server.ws import WebSocket, WsError

__all__ = [
    "FRAME_CONTENT_TYPE",
    "MAX_STREAM_FRAMES",
    "NLF_STREAM_MAGIC",
    "NLF_STREAM_VERSION",
    "RequestError",
    "LabHTTPServer",
    "build_server",
    "encode_params",
    "main",
    "pack_frame_stream",
    "read_frame_stream",
    "resolve_static",
]

LOG = logging.getLogger("numlab.app")

#: 单帧响应的 Content-Type（PROTOCOLS §3 表格写法）。
FRAME_CONTENT_TYPE = "application/x-nlf"
#: 多帧流的 Content-Type 与自定义头。
STREAM_CONTENT_TYPE = "application/octet-stream"
#: 多帧流容器标识。
NLF_STREAM_MAGIC = b"NLFS"
NLF_STREAM_VERSION = 1
#: 一次 ``/frames`` 请求允许的最大帧数（防止一个 URL 让服务端算到天荒地老）。
MAX_STREAM_FRAMES = 600
#: ``play`` 消息未给 ``time.frames`` 时的缺省帧数。
DEFAULT_PLAY_FRAMES = 60
#: ``play`` 按 fps 节拍的总睡眠预算上限（秒）：流式接口不该把同步线程占住太久。
MAX_PACE_SECONDS = 1.0
#: WS 默认读写超时。
DEFAULT_WS_TIMEOUT = 30.0

_WS_PATH_RE = re.compile(r"^/ws/chapter/([^/]+)/?$")
_API_CHAPTER_RE = re.compile(r"^/api/chapter/([^/]+)/(manifest|frame|frames)$")
_API_ASSERTS_RE = re.compile(r"^/api/asserts/([^/]+)$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")

CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".nlf": FRAME_CONTENT_TYPE,
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".wasm": "application/wasm",
}
DEFAULT_CONTENT_TYPE = "application/octet-stream"

_REQ_IDS = itertools.count(1)


class RequestError(Exception):
    """请求层错误：HTTP 层把它渲染成 ``{"error": ..., "code": ...}`` + 状态码。"""

    def __init__(self, status: int, message: str, *, code: str = "bad_request") -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = message
        self.code = code


class _WsCommandError(Exception):
    """WS 控制消息层面的错误：回 ``{"type": "error", "code", "message"}``，连接不断。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- #
# 多帧流容器（自描述长度前缀）
# --------------------------------------------------------------------------- #
def pack_frame_stream(
    chapter: str,
    viz: str,
    params: Mapping[str, Any],
    times: Sequence[float],
    frames: Sequence[bytes],
) -> bytes:
    """把 n 个 ``.nlf`` 帧打包成自描述的 ``NLFS`` 容器（格式说明见模块 docstring）。"""
    if len(frames) != len(times):
        raise ValueError(f"frames 与 times 长度不一致：{len(frames)} != {len(times)}")
    header = {
        "format": "nlf-stream",
        "version": NLF_STREAM_VERSION,
        "chapter": chapter,
        "viz": viz,
        "n": len(frames),
        "params": dict(params),
        "t": [float(t) for t in times],
    }
    hb = json.dumps(header, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    out = [NLF_STREAM_MAGIC, struct.pack(">II", NLF_STREAM_VERSION, len(hb)), hb]
    for blob in frames:
        out.append(struct.pack(">I", len(blob)))
        out.append(bytes(blob))
    return b"".join(out)


def read_frame_stream(buf: bytes | bytearray | memoryview) -> tuple[dict[str, Any], list[bytes]]:
    """解析 :func:`pack_frame_stream` 的产物，返回 ``(header, frames)``。

    任何长度不自洽都抛 :class:`ValueError`——测试用它做"独立解析"之外的第二道校验。
    """
    data = bytes(buf)
    if len(data) < 12 or data[:4] != NLF_STREAM_MAGIC:
        raise ValueError(f"不是 NLFS 容器（前 4 字节 {data[:4]!r}）")
    version, hlen = struct.unpack_from(">II", data, 4)
    if version != NLF_STREAM_VERSION:
        raise ValueError(f"NLFS 版本不支持：{version}")
    if 12 + hlen > len(data):
        raise ValueError(f"NLFS 头长度非法：{hlen}（总长 {len(data)}）")
    header = json.loads(data[12 : 12 + hlen].decode("utf-8"))
    offset = 12 + hlen
    frames: list[bytes] = []
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError(f"第 {len(frames)} 条记录的帧长字段被截断")
        (flen,) = struct.unpack_from(">I", data, offset)
        offset += 4
        if offset + flen > len(data):
            raise ValueError(f"第 {len(frames)} 条记录声明 {flen} 字节，剩余不足")
        frames.append(data[offset : offset + flen])
        offset += flen
    if len(frames) != header.get("n"):
        raise ValueError(f"记录数 {len(frames)} 与头部 n={header.get('n')!r} 不一致")
    return header, frames


# --------------------------------------------------------------------------- #
# 静态路径解析（防目录穿越）
# --------------------------------------------------------------------------- #
def resolve_static(root: pathlib.Path, rel: str) -> pathlib.Path | None:
    """把 URL 子路径解析成 ``root`` 下的真实文件。

    ``None`` 表示不存在（404）；抛出 :class:`RequestError` (403) 表示越界。

    挡住的四类写法（``tests/test_app.py`` 逐条覆盖）：

    1. ``..`` 段（含 ``%2e%2e`` 解码后的形式）与 ``.`` 段；
    2. 反斜杠分隔（``..\\..\\``）与 Windows 盘符 / UNC 前缀；
    3. 挂载前缀之后仍是绝对路径（``/viz//etc/passwd``、``/viz/%2Fetc%2Fpasswd``）；
    4. **符号链接逃逸**：``resolve()`` 之后必须仍在 root 之内。

    最后一条检查是兜底：前三条是"看起来像穿越"，它管的是"真的落到了别处"
    （软链、``..`` 的变体、未来平台差异），两者都留着才算防住。
    """
    if "\x00" in rel:
        raise RequestError(403, "路径含 NUL 字节", code="path_forbidden")
    rel = rel.replace("\\", "/")
    if rel.startswith("/"):
        raise RequestError(403, f"挂载点之后不允许绝对路径：{rel!r}", code="path_forbidden")
    if _WINDOWS_DRIVE_RE.match(rel):
        raise RequestError(403, f"不允许盘符形式的路径：{rel!r}", code="path_forbidden")
    parts: list[str] = []
    for seg in rel.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            raise RequestError(403, f"路径含上跳段 '..'：{rel!r}", code="path_forbidden")
        parts.append(seg)
    if not parts:
        return None  # 目录本身不提供列表
    root_real = pathlib.Path(root).resolve()
    try:
        real = root_real.joinpath(*parts).resolve()
    except OSError:
        return None
    if real != root_real and root_real not in real.parents:
        raise RequestError(403, f"路径逃出静态根：{rel!r}", code="path_forbidden")
    return real if real.is_file() else None


def content_type_for(path: pathlib.Path) -> str:
    """按扩展名给出 Content-Type（未知一律 octet-stream）。"""
    return CONTENT_TYPES.get(path.suffix.lower(), DEFAULT_CONTENT_TYPE)


# --------------------------------------------------------------------------- #
# 查询串 / params 解码
# --------------------------------------------------------------------------- #
def _first(query: Mapping[str, list[str]], key: str) -> str | None:
    """取查询参数（同名取最后一个，缺省或空串返回 ``None``）。"""
    values = query.get(key)
    if not values:
        return None
    value = values[-1]
    return value if value != "" else None


def decode_params(params_b64: str | None, params_json: str | None = None) -> dict[str, Any]:
    """解码 ``params``（base64url(JSON)）或 ``params_json``（调试用明文）。

    非法 base64、非 UTF-8、JSON 不是对象，都抛可诊断的 :class:`RequestError` (400)。
    只做**传输层**解码与形状检查；参数名是否已知、类型是否匹配交给
    :func:`server.frames.normalize_params`（错误消息里才有"可用参数名"）。
    """
    if params_b64 is None and params_json is None:
        return {}
    if params_b64 is not None:
        raw_text = params_b64
        text = _b64url_decode(params_b64)
    else:
        assert params_json is not None
        raw_text = params_json
        text = params_json.encode("utf-8")
    try:
        obj = json.loads(text.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RequestError(400, f"params 不是合法 UTF-8：{exc}", code="bad_params") from exc
    except ValueError as exc:
        raise RequestError(
            400,
            f"params 不是合法 JSON（{'base64url' if params_b64 is not None else 'params_json'} 解出 {raw_text[:80]!r}）：{exc}",
            code="bad_params",
        ) from exc
    if not isinstance(obj, dict):
        raise RequestError(
            400, f"params 必须是 JSON 对象，收到 {type(obj).__name__}", code="bad_params"
        )
    return obj


def _b64url_decode(text: str) -> bytes:
    """宽容的 base64url 解码：补位可省；顺带兼容被 URL 解码成空格的 ``+``。"""
    cleaned = text.strip().replace(" ", "+").replace("-", "+").replace("_", "/")
    cleaned += "=" * (-len(cleaned) % 4)
    try:
        return base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestError(400, f"params 不是合法 base64url：{exc}", code="bad_params") from exc


def encode_params(params: Mapping[str, Any]) -> str:
    """``params`` 的规范编码（无补位 base64url）：测试与调试脚本用它拼 URL。"""
    raw = json.dumps(dict(params), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _finite_float(raw: str, *, what: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise RequestError(400, f"{what} 必须是浮点数，收到 {raw!r}", code="bad_params") from exc
    if not math.isfinite(value):
        raise RequestError(400, f"{what} 必须是有限数，收到 {raw!r}", code="bad_params")
    return value


def parse_t(viz: VizSpec, raw: str | None) -> float:
    """校验 ``t``：有限、非负、落在该 viz 的时间范围内。

    ``time.mode == "none"`` 的 viz 没有时间轴，只接受 ``t=0``（PROTOCOLS §5 末条：
    没有时间控件就不该有时间语义）；越界**报错而不是静默夹取**——
    悄悄换成边界值会让客户端以为"这就是那一刻的帧"。
    """
    t = 0.0 if raw is None else _finite_float(raw, what="t")
    if t < 0.0:
        raise RequestError(400, f"t 必须 ≥ 0，收到 {t}", code="bad_params")
    spec = viz.time
    if spec.mode == "none":
        if t != 0.0:
            raise RequestError(
                400,
                f"可视化 {viz.id} 的 time.mode=none（没有时间轴），t 必须为 0，收到 {t}",
                code="bad_params",
            )
    elif spec.t_max is not None and t > spec.t_max:
        raise RequestError(
            400, f"t={t} 超出可视化 {viz.id} 的时间范围 [0, {spec.t_max}]", code="bad_params"
        )
    return t


def parse_n(viz: VizSpec, raw: str | None) -> int:
    """校验 ``n``（帧数）：≥1、≤ :data:`MAX_STREAM_FRAMES`，且无时间轴的 viz 只能 n=1。"""
    if raw is None:
        return 1
    try:
        n = int(raw)
    except ValueError as exc:
        raise RequestError(400, f"n 必须是整数，收到 {raw!r}", code="bad_params") from exc
    if n < 1:
        raise RequestError(400, f"n 必须 ≥ 1，收到 {n}", code="bad_params")
    if n > MAX_STREAM_FRAMES:
        raise RequestError(
            400, f"n={n} 超过上限 {MAX_STREAM_FRAMES}（一次请求别把服务端算穿）", code="bad_params"
        )
    if viz.time.mode == "none" and n != 1:
        raise RequestError(
            400,
            f"可视化 {viz.id} 的 time.mode=none（没有时间轴），只能取 n=1，收到 n={n}",
            code="bad_params",
        )
    return n


def frame_times(viz: VizSpec, n: int) -> list[float]:
    """``/frames`` 的时间栅格：``[0, tMax]`` 上等距 n 点。

    与 ``server.frames.precompute_chapter`` **同一口径**（同样不做 dt 对齐），
    因此同一组参数下 ``/frames`` 取到的帧与 L3 静态文件逐字节同源——降级通道的
    "先看静态、没有再算"才不会因为两套格点而对不上。单帧 ``/frame`` 才做 dt 对齐
    （PROTOCOLS §3 的 ``t`` 语义），见 :func:`align_t`。
    """
    if n <= 1 or viz.time.mode == "none":
        return [0.0]
    t_max = viz.time.t_max or 0.0
    return [t_max * i / (n - 1) for i in range(n)]


def align_t(viz: VizSpec, t: float) -> float:
    """把请求时刻对齐到 ``TimeSpec.dt`` 的最近整步（PROTOCOLS §3 的 ``t`` 语义）。

    规则与理由：

    * ``time.mode == "none"`` → 恒为 0（没有时间轴就没有"哪一刻"）；
    * ``dt`` 缺失/非正 → 原样返回（manifest 没给出格距时不敢替它定步长）；
    * 否则取 ``round(t/dt)`` 并夹到 ``[0, round(t_max/dt)]``，再用 ``%.6g`` 归一化——
      归一化的写法与 L3 文件名（``<hash>_t<%.6g>.nlf``）逐字一致，
      因此"动态取到的帧"和"静态预计算的帧"永远指向同一个时刻。
    """
    spec = viz.time
    if spec.mode == "none":
        return 0.0
    dt = spec.dt
    if not dt or dt <= 0:
        return t
    steps = math.floor(t / dt + 0.5)  # 最近整步；.5 向上，不受银行家舍入影响
    if spec.t_max is not None and spec.t_max > 0:
        steps = min(steps, math.floor(spec.t_max / dt + 0.5))  # 不许越过时间轴上界
    steps = max(steps, 0)
    return float(f"{steps * dt:.6g}")


def _stamp_time(frame: Frame, t_actual: float, t_requested: float) -> Frame:
    """写时间轴三件套：``t``/``t_actual`` = 实到时刻，``t_requested`` = 请求原值。

    ``build_frame`` 已经把 ``meta["t"]`` 设成我们传进去（= 对齐后）的时刻；这里显式再写一次，
    并把"用户要的是哪一刻"留在 ``t_requested`` 里——UI 才能说清"你要 0.103，这一帧是 0.1"。
    """
    frame.set_meta(t=t_actual, t_actual=t_actual, t_requested=t_requested)
    return frame


def frame_bytes_at(
    entry: ChapterEntry, viz: VizSpec, params: Mapping[str, Any], t: float, cache: FrameCache
) -> bytes:
    """**栅格帧**：按给定的 t 取帧，不做对齐、不打时间戳。

    与 :func:`server.frames.build_frame` + ``precompute_chapter`` 逐字节同源，
    多帧流与 L3 静态文件都走这一条。
    """
    key = _frame_cache_key(cache, entry, viz, params, t)
    blob = cache.get(key)
    if blob is None:
        blob = pack_frame(build_frame(entry, viz.id, params, t))
        cache.put(key, blob)
    return blob


def frame_bytes_for(
    entry: ChapterEntry, viz: VizSpec, params: Mapping[str, Any], t_requested: float, cache: FrameCache
) -> bytes:
    """**点名时刻的帧**（``/frame``、WS ``seek``）：对齐 → 打时间戳 → 两级缓存。

    为什么是两级缓存：``meta.t_requested`` 是响应内容的一部分，不能把 A 请求的响应发给 B 请求；
    但同一个格点上的所有拖动值又该共用同一份帧字节（对齐的收益全在这里）。

    * ``基准键``（``req:<实到时刻>``）不含请求原值：对齐到同一格点的任何请求都能命中；
    * ``原始键``（``req:<实到>|<原值>``）只服务于"未对齐"的请求——命中失败时**重贴**基准帧
      （解包 → 改 meta → 打包；1 MB 的帧也只有约 0.9 ms），而不是重算物理量。

    两个键都带命名空间前缀（``req:`` / ``req-raw:``），避免与 :func:`frame_bytes_at`
    的栅格帧（无时间戳）互相串味。
    """
    t_actual = align_t(viz, t_requested)
    base_key = _frame_cache_key(cache, entry, viz, params, t_actual, ns="req")
    if t_requested == t_actual:
        blob = cache.get(base_key)
        if blob is None:
            frame = _stamp_time(build_frame(entry, viz.id, params, t_actual), t_actual, t_actual)
            blob = pack_frame(frame)
            cache.put(base_key, blob)
        return blob
    raw_key = _frame_cache_key(cache, entry, viz, params, t_requested, ns="req-raw")
    blob = cache.get(raw_key)
    if blob is None:
        base = cache.get(base_key)
        if base is None:
            frame = _stamp_time(build_frame(entry, viz.id, params, t_actual), t_actual, t_actual)
            base = pack_frame(frame)
            cache.put(base_key, base)
        blob = pack_frame(_stamp_time(unpack_frame(base), t_actual, t_requested))
        cache.put(raw_key, blob)
    return blob


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #
def _frame_cache_key(
    cache: FrameCache,
    entry: ChapterEntry,
    viz: VizSpec,
    params: Mapping[str, Any],
    t: float,
    ns: str = "",
) -> str:
    """帧缓存的键 = 权威实现（:meth:`server.frames.FrameCache.key`）+ 可选命名空间前缀。

    ``FrameCache.key`` 才是"章节/可视化/参数指纹/时刻"的唯一定义（它和 L3 文件名同源），
    这里只在它前面加一段 ``ns``：``/frame`` 的响应字节里带 ``meta.t_requested``，
    与多帧流的栅格帧不是同一份内容，必须分开存。
    """
    base = cache.key(entry.id, viz.id, params, float(t))
    return f"{ns}|{base}" if ns else base


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #
class LabHTTPServer(ThreadingHTTPServer):
    """带路由所需上下文的 ``ThreadingHTTPServer``（daemon 线程，进程退出不留悬挂）。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        root: pathlib.Path,
        data_dir: pathlib.Path | None = None,
        registry: Registry,
        allow_write: bool = False,
        ws_timeout: float = DEFAULT_WS_TIMEOUT,
        cache: FrameCache | None = None,
    ) -> None:
        super().__init__(address, handler)
        self.root = pathlib.Path(root).resolve()
        self.data_dir = pathlib.Path(data_dir).resolve() if data_dir is not None else self.root / "data"
        self.registry = registry
        self.allow_write = bool(allow_write)
        self.ws_timeout = float(ws_timeout)
        self.cache: FrameCache = cache if cache is not None else FrameCache()

    @property
    def base_url(self) -> str:
        """内核分配之后的真实地址（``--port 0`` 时也只有这里才对）。"""
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}"


class _Handler(BaseHTTPRequestHandler):
    """HTTP/WS 路由。所有异常在这里收敛：4xx 可诊断、5xx 只进日志不上面。"""

    server_version = f"numlab/{__version__}"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = 30.0

    # ------------------------------------------------------------ 日志
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102 - 覆盖 http.server
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def log_error(self, fmt: str, *args: Any) -> None:  # noqa: D102 - 覆盖 http.server
        LOG.warning("%s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------ 入口
    def do_GET(self) -> None:  # noqa: N802 - http.server 约定
        """GET 路由。"""
        self._dispatch(method="GET", head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802 - http.server 约定
        """HEAD：与 GET 同路由但不回 body（``curl -I`` 用）。"""
        self._dispatch(method="GET", head_only=True)

    def do_POST(self) -> None:  # noqa: N802 - http.server 约定
        """POST 只用于 ``/api/asserts/<id>``（PROTOCOLS §3 那一行）。"""
        self._dispatch(method="POST", head_only=False)

    def _dispatch(self, *, method: str, head_only: bool) -> None:
        try:
            self._route(method=method, head_only=head_only)
        except RequestError as exc:
            self._send_json(exc.status, {"error": exc.message, "code": exc.code}, head_only=head_only)
        except ParamError as exc:
            # 参数不合规（未知参数/类型/越界）：ParamError 的消息里已经带了可用参数名
            self._send_json(400, {"error": str(exc), "code": "bad_params"}, head_only=head_only)
        except (BrokenPipeError, ConnectionResetError) as exc:
            LOG.info("客户端提前断开：%s %s（%s）", method, self.path, exc)
        except Exception:  # noqa: BLE001 - 兜底：绝不把 traceback 写回响应
            request_id = f"{int(time.time() * 1000):x}-{next(_REQ_IDS)}"
            LOG.exception("处理 %s %s 时发生未捕获异常（request_id=%s）", method, self.path, request_id)
            try:
                self._send_json(
                    500,
                    {"error": "内部错误（详情见服务端日志）", "code": "internal_error", "request_id": request_id},
                    head_only=head_only,
                )
            except OSError:
                pass

    # ------------------------------------------------------------ 路由
    def _route(self, *, method: str, head_only: bool) -> None:
        split = urlsplit(self.path)
        path = unquote(split.path or "/")
        query = parse_qs(split.query, keep_blank_values=True)

        if method == "POST" and not path.startswith("/api/asserts/"):
            raise RequestError(
                405, f"{path} 不接受 POST（PROTOCOLS §3 里 POST 只用于 /api/asserts/<id>）", code="method_not_allowed"
            )

        if path.startswith("/ws/") or path == "/ws":
            if method != "GET":
                raise RequestError(405, f"WebSocket 端点只支持 GET，收到 {method}", code="method_not_allowed")
            match = _WS_PATH_RE.match(path)
            if match is None:
                raise RequestError(
                    404, f"未知 WebSocket 端点 {path!r}；正确形状是 /ws/chapter/<chapter_id>", code="not_found"
                )
            self._handle_ws(match.group(1))
            return

        if path == "/api/health":
            self._send_json(200, self._api_health(), head_only=head_only)
            return
        if path == "/api/chapters":
            self._send_json(200, self._api_chapters(), head_only=head_only)
            return
        if path.startswith("/api/asserts/"):
            match = _API_ASSERTS_RE.match(path)
            if match is None:
                raise RequestError(
                    404, f"未知断言端点 {path!r}；正确形状是 /api/asserts/<chapter_id>", code="not_found"
                )
            self._send_json(200, self._api_asserts(match.group(1)), head_only=head_only)
            return
        if path.startswith("/api/chapter/"):
            match = _API_CHAPTER_RE.match(path)
            if match is None:
                raise RequestError(
                    404,
                    f"未知 API 路径 {path!r}；可用：/api/chapter/<id>/manifest|frame|frames",
                    code="not_found",
                )
            self._api_chapter(match.group(1), match.group(2), query, head_only=head_only)
            return
        if path.startswith("/api/"):
            raise RequestError(
                404, f"未知 API 路径 {path!r}（可用：/api/health、/api/chapters、/api/chapter/...、/api/asserts/...）",
                code="not_found",
            )
        if method != "GET":
            raise RequestError(
                405, f"{path} 只支持 GET（POST 仅用于 /api/asserts/<id>）", code="method_not_allowed"
            )
        self._serve_static_or_shell(path, head_only=head_only)

    # ------------------------------------------------------------ API
    def _api_health(self) -> dict[str, Any]:
        server = self._lab_server
        return {
            "ok": True,
            "chapters": len(server.registry.ids()),
            "version": __version__,
            "allowWrite": server.allow_write,
            "ws": "/ws/chapter/<chapter_id>",
        }

    def _api_chapters(self) -> dict[str, Any]:
        """章节清单 = 注册表进度 ⊕ ``docs/chapters.yaml`` 的依赖/难度/断言类型。

        语义（与 ``Registry.summary()`` 对齐）：``chapters`` 是**全部可发现的章节**——
        声明的 28 章 + 磁盘上多出来的 ``0.0`` 模板章，每条都带 ``declared: true/false``；
        顶层另有 ``count/declared/extra`` 与 ``vizzesDeclared/vizzesExtra/vizzesTotal``，
        让前端既能列全表、又能一眼看出"哪些不在计划里"（0.0 不该被悄悄过滤掉）。
        """
        server = self._lab_server
        summary = dict(server.registry.summary())
        # 逐条拷贝：本端点要往每条里塞 YAML 字段，不能改到注册表自己的对象
        # （万一哪天注册表把 summary() 的结果缓存起来，就地改就会污染它的状态）
        chapters = [dict(chapter) for chapter in summary.get("chapters", [])]
        try:
            declared_map = load_chapter_yaml(server.root)
        except (RegistryError, OSError) as exc:
            LOG.warning("读取 docs/chapters.yaml 失败：%s", exc)
            declared_map = {}
        n_declared = 0
        vizzes_declared = 0
        vizzes_extra = 0
        for chapter in chapters:
            info = declared_map.get(chapter.get("id", ""), {})
            if chapter.get("declared") is None:  # 老版本注册表不带这个字段时的兜底
                chapter["declared"] = chapter.get("id") in declared_map
            # declared 的判据以**注册表**为准：它的 root 才是章节真源所在，
            # 本服务只借 declared_map 补 deps/difficulty 这类 YAML 独有字段。
            vizzes = _as_int(chapter.get("vizDeclared"), 0)
            if chapter["declared"]:
                n_declared += 1
                vizzes_declared += vizzes
            else:
                vizzes_extra += vizzes
            chapter["deps"] = list(info.get("deps") or [])
            chapter["difficulty"] = info.get("difficulty", "")
            chapter["asserts"] = list(info.get("asserts") or [])
            chapter["modules"] = list(info.get("modules") or [])
        summary["chapters"] = chapters
        summary.update(
            {
                "yamlAvailable": bool(declared_map),
                "count": len(chapters),
                "declared": n_declared,
                "extra": len(chapters) - n_declared,
                "vizzesDeclared": vizzes_declared,
                "vizzesExtra": vizzes_extra,
                "vizzesTotal": vizzes_declared + vizzes_extra,
            }
        )
        return summary

    def _api_asserts(self, chapter_id: str) -> dict[str, Any]:
        """PROTOCOLS §8 的断言报告；只读 ``dist/reports/assertions.json``。

        ``--allow-write`` 是这一行的开关（PROTOCOLS §3 备注）：不写盘的部署下直接 403。
        这一层**不**跑 pytest——真正的"触发"属于 ``tools/run_asserts.py``（目前仓库里还没有），
        服务进程不该被一个 GET 拖进几分钟的测试会话。
        """
        server = self._lab_server
        if not server.allow_write:
            raise RequestError(
                403,
                "/api/asserts 未启用：请用 --allow-write 启动（PROTOCOLS §3 只在该模式下开放这一行）",
                code="forbidden",
            )
        try:
            server.registry.get(chapter_id)
        except KeyError as exc:
            raise RequestError(404, f"未知章节 {chapter_id!r}：{exc}", code="not_found") from exc
        report_path = server.root / "dist" / "reports" / "assertions.json"
        if not report_path.is_file():
            raise RequestError(
                404,
                f"没有断言报告 {report_path.relative_to(server.root)}；先跑 tools/run_asserts.py 生成",
                code="not_found",
            )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RequestError(500, f"断言报告不是合法 JSON：{exc}", code="internal_error") from exc
        chapters = report.get("chapters") or {}
        return {
            "chapter": chapter_id,
            "report": chapters.get(chapter_id),
            "totals": report.get("totals"),
            "generatedAt": report.get("generated_at"),
            "source": str(report_path.relative_to(server.root)),
        }

    def _api_chapter(
        self, chapter_id: str, action: str, query: Mapping[str, list[str]], *, head_only: bool
    ) -> None:
        entry = self._require_chapter(chapter_id)
        if action == "manifest":
            self._send_json(200, entry.manifest.to_json(), head_only=head_only)
            return
        viz_id = _first(query, "viz")
        if viz_id is None:
            available = ", ".join(entry.manifest.viz_ids) or "（无）"
            raise RequestError(
                400, f"缺少查询参数 viz（章节 {entry.id} 可用：{available}）", code="bad_params"
            )
        try:
            viz = entry.manifest.viz(viz_id)
        except KeyError as exc:
            available = ", ".join(entry.manifest.viz_ids) or "（无）"
            raise RequestError(
                404, f"章节 {entry.id} 没有可视化 {viz_id!r}（可用：{available}）", code="not_found"
            ) from exc

        params = decode_params(_first(query, "params"), _first(query, "params_json"))
        norm = normalize_params(viz, params)  # ParamError → 400（_dispatch 之前先转）
        if action == "frame":
            t = parse_t(viz, _first(query, "t"))
            t_actual = align_t(viz, t)
            blob = self._frame_bytes(entry, viz, norm, t)
            headers = [("X-Chapter", entry.id), ("X-Viz", viz.id), ("X-T", f"{t_actual:.6g}")]
            if t_actual != t:
                headers.append(("X-T-Requested", f"{t:.6g}"))
            self._send_bytes(200, blob, FRAME_CONTENT_TYPE, head_only=head_only, extra_headers=tuple(headers))
            return
        n = parse_n(viz, _first(query, "n"))
        times = frame_times(viz, n)
        blobs = [self._grid_frame_bytes(entry, viz, norm, t) for t in times]
        try:
            body = pack_frame_stream(entry.id, viz.id, norm, times, blobs)
        except ValueError as exc:  # pragma: no cover - 参数由上面保证
            raise RequestError(500, f"帧流打包失败：{exc}", code="internal_error") from exc
        self._send_bytes(
            200, body, STREAM_CONTENT_TYPE, head_only=head_only,
            extra_headers=(
                ("X-Nlf-Stream", f"NLFS/{NLF_STREAM_VERSION}"),
                ("X-Frame-Count", str(len(blobs))),
                ("X-Chapter", entry.id),
                ("X-Viz", viz.id),
            ),
        )

    def _require_chapter(self, chapter_id: str) -> ChapterEntry:
        try:
            return self._lab_server.registry.get(chapter_id)
        except KeyError as exc:
            known = ", ".join(self._lab_server.registry.ids()) or "（空）"
            raise RequestError(404, f"未知章节 {chapter_id!r}（已加载：{known}）", code="not_found") from exc

    def _frame_bytes(self, entry: ChapterEntry, viz: VizSpec, params: Mapping[str, Any], t: float) -> bytes:
        """``/frame`` 的帧字节：dt 对齐 + meta 时间戳（见 :func:`frame_bytes_for`）。"""
        return frame_bytes_for(entry, viz, params, t, self._lab_server.cache)

    def _grid_frame_bytes(self, entry: ChapterEntry, viz: VizSpec, params: Mapping[str, Any], t: float) -> bytes:
        """``/frames`` 的帧字节：栅格原样，不做对齐/时间戳（见 :func:`frame_bytes_at`）。"""
        return frame_bytes_at(entry, viz, params, t, self._lab_server.cache)

    # ------------------------------------------------------------ 静态资源
    def _serve_static_or_shell(self, path: str, *, head_only: bool) -> None:
        server = self._lab_server
        if path == "/":
            index = server.root / "viz" / "site" / "index.html"
            if not index.is_file():
                raise RequestError(
                    404,
                    "站点壳缺失：viz/site/index.html 尚未生成（API 仍然可用，见 /api/health）",
                    code="not_found",
                )
            self._send_bytes(200, index.read_bytes(), "text/html; charset=utf-8", head_only=head_only)
            return
        for prefix, root in self._static_mounts():
            if path.startswith(prefix):
                target = resolve_static(root, path[len(prefix) :])
                if target is None:
                    raise RequestError(404, f"静态资源不存在：{path}", code="not_found")
                try:
                    body = target.read_bytes()
                except OSError as exc:
                    raise RequestError(404, f"静态资源不可读：{path}（{exc}）", code="not_found") from exc
                self._send_bytes(
                    200, body, content_type_for(target), head_only=head_only,
                    extra_headers=(("Cache-Control", "no-cache"),),
                )
                return
        raise RequestError(404, f"未知路径 {path!r}", code="not_found")

    def _static_mounts(self) -> tuple[tuple[str, pathlib.Path], ...]:
        """URL 前缀 → 磁盘根。顺序有意义：``/dist/viz/`` 必须先于 ``/viz/`` 匹配。"""
        server = self._lab_server
        return (
            ("/dist/viz/", server.root / "dist" / "viz"),
            ("/viz/", server.root / "viz"),
            ("/data/", server.data_dir),
        )

    # ------------------------------------------------------------ WebSocket
    def _handle_ws(self, chapter_id: str) -> None:
        entry = self._require_chapter(chapter_id)  # 未知章节 → 404（还没升级，按 HTTP 回）
        self.close_connection = True  # 升级之后这个 socket 归 WebSocket，http.server 别再复用
        try:
            # 非阻塞 peek 的配方只有一份实现（server/ws.py::accept_from_handler）：
            # 照抄"rfile.peek() 一起交回"会在缓冲区为空时死锁（客户端正等 101）。
            ws = WebSocket.accept_from_handler(
                self,
                timeout=self._lab_server.ws_timeout,
                subprotocols=("nlab.v1",),
            )
        except WsError as exc:
            LOG.warning("WS 握手失败（章节 %s）：%s", chapter_id, exc)
            return
        except OSError as exc:
            LOG.info("WS 握手期间连接断开（章节 %s）：%s", chapter_id, exc)
            return
        with ws:
            serve_websocket(ws, self._lab_server, entry)

    # ------------------------------------------------------------ 响应
    @property
    def _lab_server(self) -> LabHTTPServer:
        return self.server  # type: ignore[return-value] - 只由 build_server 构造

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        head_only: bool = False,
        extra_headers: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra_headers:
            self.send_header(name, value)
        self.end_headers()
        if head_only or not body:
            return
        self.wfile.write(body)

    def _send_json(
        self,
        status: int,
        obj: Any,
        *,
        head_only: bool = False,
        extra_headers: Sequence[tuple[str, str]] = (),
    ) -> None:
        try:
            body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except ValueError:
            body = json.dumps(
                {"error": "响应含无法序列化的值（NaN/Infinity）", "code": "internal_error"}, ensure_ascii=False
            ).encode("utf-8")
            status = 500
        self._send_bytes(status, body, "application/json; charset=utf-8", head_only=head_only, extra_headers=extra_headers)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """把 ``http.server`` 内部产生的错误（400/414/501/505…）也渲染成 JSON。

        覆盖它是为了让"错误响应"只有一种形状：客户端永远可以 ``json.loads`` 报错体，
        且永远看不到 traceback（``BaseHTTPRequestHandler`` 默认发 HTML）。
        """
        try:
            short, long_message = self.responses[code][:2]  # type: ignore[index]
        except (KeyError, AttributeError):  # pragma: no cover - 自定义 code
            short, long_message = "Error", ""
        self.close_connection = True
        self._send_json(
            code,
            {"error": message or short, "code": code, "hint": explain or long_message or None},
            extra_headers=(("Connection", "close"),),
        )


# --------------------------------------------------------------------------- #
# WebSocket 会话（PROTOCOLS §2）
# --------------------------------------------------------------------------- #
@dataclass
class _Session:
    """一条 WS 连接的订阅状态：当前 viz、参数、时间轴位置。"""

    entry: ChapterEntry
    cache: FrameCache
    viz: VizSpec | None = None
    params: dict[str, Any] = field(default_factory=dict)
    t: float = 0.0

    def select(self, viz_id: Any, params: Any) -> None:
        """校验并切换到某个 viz（未知 viz / 未知参数都给出可诊断的错误）。"""
        if not isinstance(viz_id, str) or not viz_id:
            raise _WsCommandError("bad_request", "subscribe 需要字符串 viz")
        try:
            viz = self.entry.manifest.viz(viz_id)
        except KeyError as exc:
            available = ", ".join(self.entry.manifest.viz_ids) or "（无）"
            raise _WsCommandError(
                "unknown_viz", f"章节 {self.entry.id} 没有可视化 {viz_id!r}（可用：{available}）"
            ) from exc
        if params is None:
            raw: dict[str, Any] = {}
        elif isinstance(params, Mapping):
            raw = dict(params)
        else:
            raise _WsCommandError("bad_params", f"params 必须是对象，收到 {type(params).__name__}")
        try:
            norm = normalize_params(viz, raw)
        except ParamError as exc:
            raise _WsCommandError("bad_params", str(exc)) from exc
        self.viz, self.params = viz, norm

    def require_viz(self) -> VizSpec:
        if self.viz is None:
            raise _WsCommandError("no_subscription", "请先发 subscribe（含 viz 与 params）")
        return self.viz

    def set_params(self, patch: Any) -> None:
        """``params`` 消息：增量合并（PROTOCOLS §2.1）。"""
        viz = self.require_viz()
        if not isinstance(patch, Mapping):
            raise _WsCommandError("bad_params", f"params 必须是对象，收到 {type(patch).__name__}")
        merged = {**self.params, **dict(patch)}
        try:
            self.params = normalize_params(viz, merged)
        except ParamError as exc:
            raise _WsCommandError("bad_params", str(exc)) from exc

    def frame_at(self, t: float) -> bytes:
        """栅格帧（流式推送用）：与 L3/precompute 同源，不对齐、不打时间戳。"""
        return frame_bytes_at(self.entry, self.require_viz(), self.params, t, self.cache)

    def frame_for(self, t: float) -> bytes:
        """点名时刻的帧（``seek``/``step`` 用）：dt 对齐 + meta 时间戳。"""
        return frame_bytes_for(self.entry, self.require_viz(), self.params, t, self.cache)

    def forward_times(self, start: float, n: int) -> list[float]:
        """从 ``start`` 起按 ``time.dt`` 走 n 帧（无时间轴则恒为单帧）。"""
        viz = self.require_viz()
        spec = viz.time
        if spec.mode == "none" or n <= 1:
            return [start]
        t_max = spec.t_max
        dt = spec.dt if spec.dt else ((t_max - start) / (n - 1) if t_max else 1.0)
        out: list[float] = []
        t = start
        for _ in range(n):
            if t_max is not None and t > t_max:
                t = t_max
            out.append(t)
            t += dt
        return out


def _ws_send(ws: WebSocket, obj: Any) -> bool:
    """发一条 JSON；连接已断返回 ``False``（调用方据此结束会话）。"""
    try:
        ws.send_json(obj)
        return True
    except (WsError, OSError):
        return False


def _ws_send_frame(ws: WebSocket, blob: bytes) -> bool:
    """发一条二进制帧（``.nlf``）。"""
    try:
        ws.send_binary(blob)
        return True
    except (WsError, OSError):
        return False


def _ws_push(
    ws: WebSocket, session: _Session, times: Sequence[float], *, fps: float | None = None, point: bool = False
) -> bool:
    """按 PROTOCOLS §2.2 的顺序推流：``frames``(n) → n 个二进制帧 → ``done``。

    **先把整段帧算完再发预告**：``frames``(n) 一旦发出去，客户端就在等 n 个二进制帧，
    此时再报 ``error`` 会让它对不上账（少收的帧被当成丢失）。先算后发，
    失败就只回一条 ``error``，一个字节的预告都没发过。

    ``point=True`` 表示 ``times`` 是"用户点名的时刻"（``seek``）：走 dt 对齐 + meta 时间戳
    的那条路径；其余（subscribe/params/play/step）是按 ``TimeSpec.dt`` 走的栅格，与 L3 同源。
    """
    if not times:
        return _ws_send(ws, {"type": "done", "frames_sent": 0})
    take = session.frame_for if point else session.frame_at
    blobs = [take(t) for t in times]  # 计算失败会在这里抛出 → 调用方回 error
    if not _ws_send(ws, {"type": "frames", "n": len(blobs)}):
        return False
    viz = session.require_viz() if point else None
    budget = MAX_PACE_SECONDS
    sent = 0
    for index, (t, blob) in enumerate(zip(times, blobs)):
        if not _ws_send_frame(ws, blob):
            return False
        # 点名帧：游标落在**实到时刻**（帧字节里已经对齐过），而不是请求原值
        session.t = align_t(viz, t) if viz is not None else t
        sent += 1
        if fps and fps > 0 and index + 1 < len(times) and budget > 0:
            delay = min(1.0 / fps, budget)
            time.sleep(delay)
            budget -= delay
    return _ws_send(ws, {"type": "done", "frames_sent": sent})


def _int_or(value: Any, default: int) -> int:
    """宽容地把 JSON 里的数字/数字串读成 int（bool 不算）。"""
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _as_int(value: Any, fallback: int) -> int:
    """注册表字段 → int（缺字段/None/异常一律退回 ``fallback``，不让 API 因为计数器炸掉）。"""
    return _int_or(value, fallback)


def _fps_of(msg: Mapping[str, Any]) -> float | None:
    time_spec = msg.get("time")
    if not isinstance(time_spec, Mapping):
        return None
    fps = time_spec.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        return None
    value = float(fps)
    return value if value > 0 else None


def serve_websocket(ws: WebSocket, server: LabHTTPServer, entry: ChapterEntry) -> None:
    """一条 WS 连接的完整生命周期（握手已完成）。

    首条消息恒为 ``ready``（附整章 manifest，前端据此建控件），随后按 §2.1 的控制消息
    推帧。任何一层的失败都收敛成 ``{"type": "error", ...}``（可恢复）或直接结束连接
    （帧层违规时 :mod:`server.ws` 已经发过 close）。
    """
    session = _Session(entry=entry, cache=server.cache)
    if not _ws_send(ws, {"type": "ready", "chapter": entry.id, "manifest": entry.manifest.to_json()}):
        return
    while True:
        try:
            msg = ws.recv_json()
        except TimeoutError:
            _ws_send(ws, {"type": "error", "code": "timeout", "message": f"空闲超过 {server.ws_timeout}s"})
            return
        except WsError as exc:
            if ws.closed:  # 帧层违规：ws.py 已发 close，收尾即可
                return
            # 消息边界没坏（例如客户端手滑发了二进制帧）：报错但继续服务
            if not _ws_send(ws, {"type": "error", "code": "bad_message", "message": str(exc)}):
                return
            continue
        except OSError:
            return
        if msg is None:
            return
        if not _handle_ws_message(ws, session, msg):
            return


def _handle_ws_message(ws: WebSocket, session: _Session, msg: dict[str, Any]) -> bool:
    """处理一条控制消息；返回 ``False`` 表示连接该结束了。"""
    kind = msg.get("type")
    try:
        if kind == "ping":
            return _ws_send(ws, {"type": "pong", "t": msg.get("t", time.time())})
        if kind == "subscribe":
            return _ws_subscribe(ws, session, msg)
        if kind == "params":
            session.set_params(msg.get("params"))
            time_spec = msg.get("time") if isinstance(msg.get("time"), Mapping) else None
            if time_spec is not None and session.require_viz().time.mode != "none":
                n = max(1, min(_int_or(time_spec.get("frames"), 1), MAX_STREAM_FRAMES))
                return _ws_push(ws, session, session.forward_times(session.t, n), fps=_fps_of(msg))
            return _ws_push(ws, session, [session.t])
        if kind == "seek":
            viz = session.require_viz()
            t = msg.get("t")
            if isinstance(t, bool) or not isinstance(t, (int, float)):
                raise _WsCommandError("bad_request", "seek 需要数值 t")
            # 与 /frame 同一口径：对齐到 dt 的最近整步，帧里回写实到时刻
            # （游标由 _ws_push 落到实到时刻，下次 step 才从格点上继续走）
            return _ws_push(ws, session, [parse_ws_t(viz, float(t))], point=True)
        if kind == "step":
            viz = session.require_viz()
            n = _int_or(msg.get("n"), 1)
            if n < 1 or n > MAX_STREAM_FRAMES:
                raise _WsCommandError("bad_request", f"step 的 n 必须在 1..{MAX_STREAM_FRAMES}，收到 {n}")
            if viz.time.mode == "none":
                return _ws_push(ws, session, [session.t])
            base = session.t + (viz.time.dt or 0.0)
            return _ws_push(ws, session, session.forward_times(base, n))
        if kind == "play":
            viz = session.require_viz()
            if viz.time.mode == "none":
                return _ws_push(ws, session, [session.t])
            time_spec = msg.get("time") if isinstance(msg.get("time"), Mapping) else {}
            n = max(1, min(_int_or(time_spec.get("frames"), DEFAULT_PLAY_FRAMES), MAX_STREAM_FRAMES))
            dt = time_spec.get("dt")
            if isinstance(dt, (int, float)) and not isinstance(dt, bool) and dt > 0 and viz.time.t_max:
                span = max(0.0, viz.time.t_max - session.t)
                n = min(n, max(1, int(span / float(dt)) + 1))
            times = session.forward_times(session.t, n)
            return _ws_push(ws, session, times, fps=_fps_of(msg))
        if kind in ("pause", "cancel"):
            # 同步流模型下这两者只作用于"流与流之间"（见模块 docstring 的线程模型一节）
            return _ws_send(ws, {"type": "done", "frames_sent": 0, "t": session.t})
        if kind is None:
            raise _WsCommandError("bad_request", "控制消息缺少 type 字段")
        raise _WsCommandError(
            "unknown_type",
            f"未知控制消息类型 {kind!r}（支持：subscribe/params/play/pause/seek/step/cancel/ping）",
        )
    except _WsCommandError as exc:
        return _ws_send(ws, {"type": "error", "code": exc.code, "message": exc.message})
    except ParamError as exc:
        return _ws_send(ws, {"type": "error", "code": "bad_params", "message": str(exc)})
    except (NlfError, RegistryError) as exc:
        LOG.exception("WS 计算失败（章节 %s）", session.entry.id)
        return _ws_send(ws, {"type": "error", "code": "internal", "message": f"计算失败：{exc}"})


def parse_ws_t(viz: VizSpec, t: float) -> float:
    """WS 版 ``t`` 校验（与 HTTP 的 :func:`parse_t` 同一口径，但抛 WS 错误）。"""
    if not math.isfinite(t):
        raise _WsCommandError("bad_request", f"t 必须是有限数，收到 {t}")
    if t < 0.0:
        raise _WsCommandError("bad_request", f"t 必须 ≥ 0，收到 {t}")
    spec = viz.time
    if spec.mode == "none":
        return 0.0
    if spec.t_max is not None and t > spec.t_max:
        raise _WsCommandError("bad_request", f"t={t} 超出时间范围 [0, {spec.t_max}]")
    return t


def _ws_subscribe(ws: WebSocket, session: _Session, msg: dict[str, Any]) -> bool:
    """``subscribe``：选 viz、规范化参数、回 ready，然后推首帧（或 ``time.frames`` 帧）。"""
    chapter = msg.get("chapter")
    if chapter is not None and chapter != session.entry.id:
        raise _WsCommandError(
            "chapter_mismatch",
            f"连接的是章节 {session.entry.id}，subscribe 里却是 {chapter!r}（章节由 URL 决定）",
        )
    session.select(msg.get("viz"), msg.get("params"))
    session.t = 0.0
    time_spec = msg.get("time") if isinstance(msg.get("time"), Mapping) else {}
    n = 1
    viz = session.require_viz()
    if viz.time.mode != "none":
        n = max(1, min(_int_or(time_spec.get("frames"), 1), MAX_STREAM_FRAMES))
    if not _ws_send(
        ws,
        {
            "type": "ready",
            "chapter": session.entry.id,
            "viz": viz.id,
            "params": session.params,
            "manifest": session.entry.manifest.to_json(),
        },
    ):
        return False
    return _ws_push(ws, session, session.forward_times(0.0, n), fps=_fps_of(msg))


# --------------------------------------------------------------------------- #
# 启动
# --------------------------------------------------------------------------- #
def build_server(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    root: pathlib.Path | str | None = None,
    data_dir: pathlib.Path | str | None = None,
    registry: Registry | None = None,
    allow_write: bool = False,
    ws_timeout: float = DEFAULT_WS_TIMEOUT,
    cache: FrameCache | None = None,
) -> LabHTTPServer:
    """建服务（不启动）。``port=0`` 时真实端口在 ``server.server_address[1]``。"""
    root_path = pathlib.Path(root).resolve() if root is not None else pathlib.Path(__file__).resolve().parents[1]
    if registry is None:
        registry = Registry(root_path)
        registry.discover()
    return LabHTTPServer(
        (host, int(port)),
        _Handler,
        root=root_path,
        data_dir=data_dir,
        registry=registry,
        allow_write=allow_write,
        ws_timeout=ws_timeout,
        cache=cache,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m server.app",
        description="数值方法实验室：零依赖 HTTP + WebSocket 服务（PROTOCOLS §3）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（缺省 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8080, help="监听端口；0 = 由内核分配（缺省 8080）")
    parser.add_argument("--root", default=None, help="仓库根（缺省本文件的上两级目录）")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="静态 /data/ 的根目录（缺省 <root>/data；指向别处即可用预计算产物做降级演示）",
    )
    parser.add_argument(
        "--allow-write",
        action="store_true",
        help="启用 /api/asserts/<id>（PROTOCOLS §3 只在该模式下开放这一行）",
    )
    parser.add_argument("--ws-timeout", type=float, default=DEFAULT_WS_TIMEOUT, help="WS 读写超时秒数")
    parser.add_argument("--quiet", action="store_true", help="只打印监听地址（脚本/测试用）")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：解析参数 → 建服务 → 打印真实地址 → ``serve_forever`` 直到 SIGINT/SIGTERM。"""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        httpd = build_server(
            host=args.host,
            port=args.port,
            root=args.root,
            data_dir=args.data_dir,
            allow_write=args.allow_write,
            ws_timeout=args.ws_timeout,
        )
    except (RegistryError, OSError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1

    host, port = httpd.server_address[0], httpd.server_address[1]
    print(f"数值方法实验室服务已启动：http://{host}:{port}/", flush=True)
    print(f"  WebSocket：ws://{host}:{port}/ws/chapter/<chapter_id>", flush=True)
    print(f"  仓库根：{httpd.root}", flush=True)
    print(f"  静态 /data/：{httpd.data_dir}", flush=True)
    print(f"  断言写入口（/api/asserts）：{'开启' if httpd.allow_write else '关闭（--allow-write 可开）'}", flush=True)

    def _stop(signum: int, _frame: Any) -> None:
        LOG.info("收到信号 %s：正在停止…", signum)
        # shutdown() 必须从**另一个**线程调用（serve_forever 在主线程里阻塞）
        threading.Thread(target=httpd.shutdown, name="numlab-shutdown", daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):  # pragma: no cover - 非主线程/平台不支持
            pass
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:  # pragma: no cover - 信号处理器之外的第二道保险
        LOG.info("KeyboardInterrupt：正在停止…")
    finally:
        httpd.server_close()
        print("服务已停止。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
