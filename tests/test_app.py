"""``server/app.py``（HTTP + WS 服务）与 ``tools/precompute.py`` 的端到端测试。

对应 ``docs/PROTOCOLS.md`` §3 的**每一行**：正常路径 + 404/400/403/405 路径、
``params`` 的 base64url 往返、多帧流的拼接格式、WS 的一次完整往返与并发、
静态资源的四种目录穿越、``--allow-write`` 的开关语义、L3 预计算的落盘与 prune。

拓扑照抄 ``tests/test_ws.py``：``build_server(port=0)`` + 后台守护线程 + 轮询/往返同步，
不靠 ``sleep`` 猜时间；客户端一律用自研的 :class:`server.wsclient.WebSocketClient`。

两处刻意的写法：

* **多帧流用手工 ``struct`` 解析**（``_parse_stream_manual``），不复用 ``server.app``
  里的 :func:`read_frame_stream`——否则就是拿实现证明实现；后者另有测试做交叉验证。
* **合成一章 ``9.9``**（bool/enum/中文参数名俱全）来覆盖"参数类型"这一族断言：
  真实的 0.0 章只有 float/int，用它测不出 bool/enum 的往返。
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import socket
import struct
import threading
import types
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pytest

from server.app import (
    FRAME_CONTENT_TYPE,
    MAX_STREAM_FRAMES,
    STREAM_CONTENT_TYPE,
    LabHTTPServer,
    RequestError,
    align_t,
    build_server,
    decode_params,
    encode_params,
    read_frame_stream,
    resolve_static,
)
from server.frames import FrameCache, build_frame
from server.manifest import load_manifest
from server.nlf import Frame, NlfError, pack_frame, unpack_frame
from server.registry import ChapterEntry, Registry
from server.wsclient import WebSocketClient, WsHandshakeError

pytestmark = pytest.mark.e2e

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHAPTER = "0.0"
#: time.mode="scrub"（tMax=0.2, dt=0.005）
SCRUB_VIZ = "heat_ftcs"
#: time.mode="stream"（tMax=20, dt=0.05）
STREAM_VIZ = "oscillator_regimes"


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _get(httpd: LabHTTPServer, path: str, *, timeout: float = 30.0) -> tuple[int, dict[str, str], bytes]:
    """GET 一个路径；4xx/5xx 也照常返回 ``(status, headers, body)`` 而不是抛异常。"""
    try:
        with urllib.request.urlopen(httpd.base_url + path, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def _get_json(httpd: LabHTTPServer, path: str) -> tuple[int, dict[str, Any]]:
    status, headers, body = _get(httpd, path)
    assert headers.get("Content-Type", "").startswith("application/json"), headers
    return status, json.loads(body.decode("utf-8"))


def _raw(httpd: LabHTTPServer, request: bytes, *, timeout: float = 30.0) -> bytes:
    """裸 socket 发一条请求，收完整响应（用于穿越路径 / HEAD / 坏方法：urllib 会自作主张）。"""
    with socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=timeout) as sock:
        sock.sendall(request)
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def _raw_status_body(httpd: LabHTTPServer, target: str, method: str = "GET") -> tuple[int, bytes]:
    """发一条最小的 HTTP/1.1 请求，返回 ``(状态码, body)``。"""
    request = f"{method} {target} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n".encode("latin-1")
    response = _raw(httpd, request)
    head, _, body = response.partition(b"\r\n\r\n")
    line = head.split(b"\r\n", 1)[0]
    return int(line.split(b" ")[1]), body


def _raw_until(httpd: LabHTTPServer, request: bytes, marker: bytes, *, timeout: float = 15.0) -> bytes:
    """裸 socket 发请求并读到出现 ``marker`` 为止（WS 升级后连接不会自己关，不能读到 EOF）。"""
    with socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=timeout) as sock:
        sock.sendall(request)
        data = b""
        while marker not in data:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    return data


def _parse_stream_manual(blob: bytes) -> tuple[dict[str, Any], list[bytes]]:
    """手工解析 NLFS 容器（测试独立实现，不复用被测代码）；u32 一律**大端**。"""
    assert blob[:4] == b"NLFS", blob[:4]
    version, header_len = struct.unpack_from(">II", blob, 4)
    assert version == 1, version
    header = json.loads(blob[12 : 12 + header_len].decode("utf-8"))
    offset = 12 + header_len
    frames: list[bytes] = []
    while offset < len(blob):
        assert offset + 4 <= len(blob), "帧长字段被截断"
        (frame_len,) = struct.unpack_from(">I", blob, offset)
        offset += 4
        assert offset + frame_len <= len(blob), "帧体被截断"
        frames.append(blob[offset : offset + frame_len])
        offset += frame_len
    assert offset == len(blob), "尾部有多余字节"
    return header, frames


def _ws_url(httpd: LabHTTPServer, path: str = f"/ws/chapter/{CHAPTER}") -> str:
    return f"ws://127.0.0.1:{httpd.server_address[1]}{path}"


def _read_push(client: WebSocketClient, expect: int) -> list[Frame]:
    """读一次完整的 §2.2 推送：``frames``(n) → n 个二进制 → ``done``。"""
    head = client.recv_json()
    assert head is not None and head["type"] == "frames" and head["n"] == expect, head
    frames: list[Frame] = []
    for _ in range(expect):
        message = client.recv()
        assert message is not None and message.opcode == "binary", message
        frames.append(unpack_frame(message.data))
    tail = client.recv_json()
    assert tail is not None and tail["type"] == "done" and tail["frames_sent"] == expect, tail
    return frames


def _assert_same_frame(got: Frame, expected: Frame) -> None:
    """逐数组比对帧内容（``.nlf`` 的数组 + meta.t）。"""
    assert got.chapter == expected.chapter
    assert got.viz == expected.viz
    assert sorted(got.names) == sorted(expected.names)
    for name in expected.names:
        np.testing.assert_array_equal(got[name], expected[name], err_msg=f"数组 {name} 不一致")
    assert float(got.meta["t"]) == float(expected.meta["t"])


# --------------------------------------------------------------------------- #
# 合成章节 9.9：bool / enum / 中文参数名，真实 0.0 章覆盖不到的类型
# --------------------------------------------------------------------------- #
DEMO_PITFALLS = (
    "合成章节的唯一用途是把参数类型跑全：bool 与 enum 在 URL 里只有 base64 一层保护，"
    "写错一个字符就会变成另一种类型，而界面上看起来毫无异常——所以测试必须逐个类型钉死。"
)

DEMO_MANIFEST: dict[str, Any] = {
    "chapter": "9.9",
    "title": "测试用合成章节（不参与教学）",
    "part": "",
    "summary": "覆盖 bool/enum/中文参数名的参数往返。",
    "vizzes": [
        {
            "id": "demo",
            "title": "合成可视化",
            "description": "把参数原样写回 meta，便于断言。",
            "dataSource": "backend",
            "params": [
                {
                    "name": "n",
                    "label": "样本数 n",
                    "type": "int",
                    "min": 1,
                    "max": 64,
                    "default": 4,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "alpha",
                    "label": "系数 α",
                    "type": "float",
                    "min": 0.0,
                    "max": 10.0,
                    "default": 1.5,
                    "critical": [2.0],
                    "criticalNote": "α>2 起支配（合成样例）",
                    "beyond": "above",
                },
                {
                    "name": "flag",
                    "label": "开关",
                    "type": "bool",
                    "default": False,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "mode",
                    "label": "模式",
                    "type": "enum",
                    "options": ["fast", "safe", "中文模式"],
                    "default": "fast",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "系数",
                    "label": "中文参数名",
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.25,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [{"id": "main", "autoscaled": False}],
            "pitfalls": DEMO_PITFALLS,
            "theory": ["合成理论：Everything is a parameter"],
        },
        {
            "id": "boom",
            "title": "会炸的合成可视化（t > 0.25 抛错）",
            "description": "用来验证「计算失败时不许先发 frames 预告」。",
            "dataSource": "backend",
            "params": [
                {
                    "name": "n",
                    "label": "样本数 n",
                    "type": "int",
                    "min": 1,
                    "max": 8,
                    "default": 2,
                    "critical": [],
                    "criticalNote": "",
                }
            ],
            "time": {"mode": "stream", "tMax": 1.0, "dt": 0.5, "fps": 30, "label": "t"},
            "requiredAux": ["error"],
            "axes": [{"id": "main", "autoscaled": False}],
            "pitfalls": DEMO_PITFALLS,
            "theory": ["t > 0.25 时故意抛 NlfError，模拟章节实现挂掉"],
        },
    ],
}


def _demo_compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """把参数原样写进 meta/strings，测试据此断言"参数确实到达了计算层"。"""
    if viz_id == "boom":
        if float(t) > 0.25:
            raise NlfError(f"合成故障：t={t} 太大（> 0.25）")
        frame = Frame(chapter="9.9", viz="boom")
        frame.add("x", np.zeros(2, dtype=float), unit="1", label="样本")
        frame.add("error", np.zeros(2, dtype=float), unit="1", label="误差")
        return frame
    if viz_id != "demo":
        raise KeyError(viz_id)
    n = int(params["n"])
    frame = Frame(chapter="9.9", viz="demo")
    frame.add("x", np.arange(n, dtype=float), unit="1", label="样本")
    frame.add("error", np.zeros(n, dtype=float), unit="1", label="误差（规则 3 要求）")
    frame.set_meta(
        t=float(t),
        n=n,
        alpha=float(params["alpha"]),
        flag=bool(params["flag"]),
        系数=float(params["系数"]),
    )
    # enum 是字符串标量 → 走 strings（PROTOCOLS §1.1：meta 只允许 number/bool/null）
    frame.set_strings(mode=str(params["mode"]), 中文标签="参数已送达计算层")
    return frame


def _demo_entry() -> ChapterEntry:
    return ChapterEntry(
        id="9.9",
        path=pathlib.Path("<demo>"),
        manifest=load_manifest(DEMO_MANIFEST, where="测试合成章节"),
        module=types.SimpleNamespace(compute=_demo_compute),
    )


class _FakeRegistry:
    """只实现 app.py 用到的那几个方法（get/ids/summary/all/iter_viz）。"""

    def __init__(self, entries: list[ChapterEntry]) -> None:
        self._entries = {entry.id: entry for entry in entries}

    def get(self, chapter_id: str) -> ChapterEntry:
        if chapter_id not in self._entries:
            raise KeyError(f"没有章节 {chapter_id}")
        return self._entries[chapter_id]

    def ids(self) -> list[str]:
        return sorted(self._entries)

    def all(self) -> list[ChapterEntry]:
        return [self._entries[cid] for cid in self.ids()]

    def iter_viz(self) -> Iterator[tuple[str, str]]:
        for entry in self.all():
            for viz_id in entry.viz_ids:
                yield entry.id, viz_id

    def summary(self) -> dict[str, Any]:
        return {
            "chapters_declared": len(self._entries),
            "chapters_complete": 0,
            "vizzes_declared": sum(len(e.viz_ids) for e in self._entries.values()),
            "vizzes_implemented": 0,
            "chapters": [
                {"id": e.id, "title": e.title, "vizDeclared": len(e.viz_ids), "complete": False}
                for e in self.all()
            ],
        }


class _BoomRegistry(_FakeRegistry):
    """取章节时抛内部错误：用来验证 500 不泄漏 traceback。"""

    def get(self, chapter_id: str) -> ChapterEntry:
        raise RuntimeError("SECRET-内部细节 /etc/shadow server/app.py line 42")


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def real_registry() -> Registry:
    registry = Registry(ROOT)
    registry.discover()
    return registry


@pytest.fixture
def serve() -> Iterator[Callable[..., LabHTTPServer]]:
    """``serve(**kwargs)`` 启动一台服务（``port=0``），测试结束统一关掉。"""
    started: list[LabHTTPServer] = []

    def start(**kwargs: Any) -> LabHTTPServer:
        kwargs.setdefault("port", 0)
        httpd = build_server(**kwargs)
        threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        ).start()
        started.append(httpd)
        return httpd

    yield start
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def lab(serve: Callable[..., LabHTTPServer], real_registry: Registry) -> LabHTTPServer:
    """真实仓库（真章节 + 真静态根），只读。"""
    return serve(root=ROOT, registry=real_registry)


@pytest.fixture
def demo_lab(serve: Callable[..., LabHTTPServer]) -> LabHTTPServer:
    """合成章节 9.9（bool/enum/中文参数名）。"""
    return serve(root=ROOT, registry=_FakeRegistry([_demo_entry()]))


@pytest.fixture
def temp_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """一个最小的"站点根"：站点壳 + 静态资源 + 一个假 .nlf。"""
    root = tmp_path / "lab"
    (root / "viz" / "site").mkdir(parents=True)
    (root / "viz" / "site" / "index.html").write_text(
        "<!doctype html><html><title>合成站点</title></html>", encoding="utf-8"
    )
    (root / "viz" / "app.js").write_text("export const answer = 42;\n", encoding="utf-8")
    (root / "dist" / "viz").mkdir(parents=True)
    (root / "dist" / "viz" / "bundle.js").write_text("// bundle\n", encoding="utf-8")
    (root / "data").mkdir(parents=True)
    (root / "data" / "pre.nlf").write_bytes(b"NLF1-pretend")
    return root


# --------------------------------------------------------------------------- #
# 1. /api/health · /api/chapters
# --------------------------------------------------------------------------- #
def test_health_reports_ok_chapters_and_version(lab: LabHTTPServer, real_registry: Registry) -> None:
    status, payload = _get_json(lab, "/api/health")
    assert status == 200
    assert payload["ok"] is True
    assert payload["chapters"] == len(real_registry.ids())
    assert isinstance(payload["version"], str) and payload["version"]


def test_health_allow_write_flag_reflects_startup(
    serve: Callable[..., LabHTTPServer], real_registry: Registry
) -> None:
    off = serve(root=ROOT, registry=real_registry)
    on = serve(root=ROOT, registry=real_registry, allow_write=True)
    assert _get_json(off, "/api/health")[1]["allowWrite"] is False
    assert _get_json(on, "/api/health")[1]["allowWrite"] is True


def test_chapters_merges_yaml_progress_and_deps(lab: LabHTTPServer) -> None:
    status, payload = _get_json(lab, "/api/chapters")
    assert status == 200
    assert payload["yamlAvailable"] is True  # 仓库里有 docs/chapters.yaml（PyYAML 或退化解析）
    by_id = {chapter["id"]: chapter for chapter in payload["chapters"]}
    assert CHAPTER in by_id, "0.0 是磁盘上的额外章节，必须出现在清单里"
    demo = by_id[CHAPTER]
    assert demo["declared"] is False  # 0.0 不在 docs/chapters.yaml 的 28 章里
    assert demo["hasModule"] is True
    assert demo["vizDeclared"] >= 2
    assert isinstance(demo["deps"], list) and isinstance(demo["modules"], list)
    # YAML 侧字段真的被合并进来了：至少有一章声明了依赖（例如 1.2 → 1.1）
    declared = [chapter for chapter in payload["chapters"] if chapter["declared"]]
    assert declared, "docs/chapters.yaml 里的章节必须出现在清单里"
    assert any(chapter["deps"] for chapter in declared)
    assert all("difficulty" in chapter and "asserts" in chapter for chapter in declared)


def test_chapters_top_level_counts_split_declared_and_extra(lab: LabHTTPServer) -> None:
    _, payload = _get_json(lab, "/api/chapters")
    chapters = payload["chapters"]
    assert payload["count"] == len(chapters)
    assert payload["declared"] == sum(1 for c in chapters if c["declared"])
    assert payload["extra"] == sum(1 for c in chapters if not c["declared"])
    assert payload["count"] == payload["declared"] + payload["extra"]
    assert payload["vizzesTotal"] == payload["vizzesDeclared"] + payload["vizzesExtra"]
    assert payload["extra"] >= 1 and payload["vizzesExtra"] >= 2  # 0.0 的两个 viz


def test_chapters_passes_through_registry_progress_fields(lab: LabHTTPServer, real_registry: Registry) -> None:
    """注册表的进度字段（含 P13 的 hasPitfalls/pitfallChars/hasAssertsDecl）必须原样透传。"""
    _, payload = _get_json(lab, "/api/chapters")
    source = {row["id"]: row for row in real_registry.summary()["chapters"]}
    yaml_only = {"deps", "difficulty", "asserts", "modules"}  # 本端点额外合并的四项
    for chapter in payload["chapters"]:
        passthrough = {key: value for key, value in chapter.items() if key not in yaml_only}
        assert passthrough == source[chapter["id"]], chapter["id"]
    zero = next(chapter for chapter in payload["chapters"] if chapter["id"] == CHAPTER)
    p13 = [key for key in source[CHAPTER] if key.startswith("has") or key.startswith("pitfall")]
    assert p13, "注册表应当带上 P13 的进度字段"
    assert all(zero[key] == source[CHAPTER][key] for key in p13)
    assert zero["pitfallChars"] > 0


def test_unknown_api_path_is_404_json(lab: LabHTTPServer) -> None:
    status, payload = _get_json(lab, "/api/nope")
    assert status == 404
    assert payload["code"] == "not_found"
    assert "/api/health" in payload["error"]


def test_chapters_without_yaml_degrades_gracefully(
    serve: Callable[..., LabHTTPServer], real_registry: Registry, temp_root: pathlib.Path
) -> None:
    """没有 docs/chapters.yaml 时不许 500：清单退化为"注册表里有什么就报什么"。"""
    httpd = serve(root=temp_root, registry=real_registry)
    status, payload = _get_json(httpd, "/api/chapters")
    assert status == 200
    assert payload["yamlAvailable"] is False
    assert payload["count"] == len(payload["chapters"]) >= 1
    assert payload["declared"] == sum(1 for chapter in payload["chapters"] if chapter["declared"])
    assert payload["count"] == payload["declared"] + payload["extra"]
    # 读不到 YAML → 依赖/难度这类字段只能是空的，但不许 500
    assert all(chapter["deps"] == [] and chapter["difficulty"] == "" for chapter in payload["chapters"])


# --------------------------------------------------------------------------- #
# 2. /api/chapter/<id>/manifest
# --------------------------------------------------------------------------- #
def test_manifest_equals_registry_source_of_truth(lab: LabHTTPServer, real_registry: Registry) -> None:
    status, payload = _get_json(lab, f"/api/chapter/{CHAPTER}/manifest")
    assert status == 200
    assert payload == real_registry.get(CHAPTER).manifest.to_json()
    assert {v["id"] for v in payload["vizzes"]} == {"oscillator_regimes", "heat_ftcs"}


def test_manifest_unknown_chapter_404(lab: LabHTTPServer) -> None:
    status, payload = _get_json(lab, "/api/chapter/9.9/manifest")
    assert status == 404
    assert payload["code"] == "not_found"
    assert "9.9" in payload["error"]


def test_manifest_bad_action_404(lab: LabHTTPServer) -> None:
    status, payload = _get_json(lab, f"/api/chapter/{CHAPTER}/nonsense")
    assert status == 404
    assert payload["code"] == "not_found"


# --------------------------------------------------------------------------- #
# 3. /api/chapter/<id>/frame
# --------------------------------------------------------------------------- #
def test_frame_returns_nlf_bytes_with_x_nlf_content_type(lab: LabHTTPServer) -> None:
    status, headers, body = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}")
    assert status == 200
    assert headers["Content-Type"] == FRAME_CONTENT_TYPE
    assert headers["Content-Length"] == str(len(body))
    assert body[:4] == b"NLF1"
    frame = unpack_frame(body)  # 能被权威实现读回
    assert frame.chapter == CHAPTER and frame.viz == SCRUB_VIZ
    assert "error" in frame.names  # 规则 3 的副轴数据确实在帧里


def test_frame_bytes_match_direct_build_frame(lab: LabHTTPServer, real_registry: Registry) -> None:
    params = {"nx": 21, "r": 0.35, "t_end": 0.02}
    status, _, body = _get(
        lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&params={encode_params(params)}&t=0.02"
    )
    assert status == 200
    entry = real_registry.get(CHAPTER)
    expected = build_frame(entry, SCRUB_VIZ, params, 0.02)
    _assert_same_frame(unpack_frame(body), expected)


def test_frame_default_params_are_manifest_defaults(lab: LabHTTPServer, real_registry: Registry) -> None:
    _, _, body = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={STREAM_VIZ}")
    frame = unpack_frame(body)
    assert json.loads(frame.strings["params_json"]) == {"omega": 1.0, "zeta": 0.05, "n_periods": 3}


def test_frame_t_reaches_meta_and_headers(lab: LabHTTPServer) -> None:
    status, headers, body = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&t=0.1")
    assert status == 200
    assert float(unpack_frame(body).meta["t"]) == 0.1
    assert float(headers["X-T"]) == 0.1
    assert headers["X-Viz"] == SCRUB_VIZ


def test_frame_params_float_int_roundtrip(lab: LabHTTPServer) -> None:
    params = {"omega": 2.5, "zeta": 1.0, "n_periods": 2}
    status, _, body = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={STREAM_VIZ}&params={encode_params(params)}")
    assert status == 200
    frame = unpack_frame(body)
    assert frame.meta["omega"] == 2.5 and frame.meta["zeta"] == 1.0
    assert json.loads(frame.strings["params_json"]) == params


def test_params_json_alias_matches_base64_params(lab: LabHTTPServer) -> None:
    params = {"omega": 3.0, "zeta": 0.5, "n_periods": 1}
    _, _, via_b64 = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={STREAM_VIZ}&params={encode_params(params)}")
    _, _, via_json = _get(
        lab,
        f"/api/chapter/{CHAPTER}/frame?viz={STREAM_VIZ}&params_json="
        + urllib.parse.quote(json.dumps(params)),
    )
    _assert_same_frame(unpack_frame(via_b64), unpack_frame(via_json))


def test_params_base64url_roundtrip_all_json_types() -> None:
    """中文键/值 + 浮点 + 布尔 + enum + 整数，经 base64url 一个都不许变形。"""
    params: dict[str, Any] = {
        "系数": 0.75,
        "alpha": 1.25,
        "flag": True,
        "mode": "中文模式",
        "n": 7,
        "nested_free": None,
    }
    encoded = encode_params(params)
    assert "=" not in encoded  # 无补位的 base64url
    assert decode_params(encoded) == params


def test_frame_accepts_padded_and_standard_base64(lab: LabHTTPServer) -> None:
    raw = json.dumps({"omega": 2.0, "zeta": 0.5, "n_periods": 2}, ensure_ascii=False).encode("utf-8")
    padded = base64.b64encode(raw).decode("ascii")  # 标准字母表 + 补位
    status, _, body = _get(
        lab, f"/api/chapter/{CHAPTER}/frame?viz={STREAM_VIZ}&params=" + urllib.parse.quote(padded)
    )
    assert status == 200
    assert unpack_frame(body).meta["omega"] == 2.0


def test_frame_bool_enum_and_chinese_values_reach_compute(demo_lab: LabHTTPServer) -> None:
    params = {"n": 6, "alpha": 2.5, "flag": True, "mode": "中文模式", "系数": 0.75}
    status, _, body = _get(demo_lab, f"/api/chapter/9.9/frame?viz=demo&params={encode_params(params)}")
    assert status == 200
    frame = unpack_frame(body)
    assert frame.meta["flag"] is True
    assert frame.strings["mode"] == "中文模式"
    assert frame.meta["系数"] == 0.75
    assert frame.meta["n"] == 6
    assert frame.strings["中文标签"] == "参数已送达计算层"
    assert frame["x"].shape == (6,)


def test_frame_wrong_bool_type_is_400(demo_lab: LabHTTPServer) -> None:
    status, payload = _get_json(
        demo_lab, "/api/chapter/9.9/frame?viz=demo&params=" + encode_params({"flag": 1})
    )
    assert status == 400
    assert payload["code"] == "bad_params"
    assert "flag" in payload["error"]


def test_frame_enum_out_of_options_is_400(demo_lab: LabHTTPServer) -> None:
    status, payload = _get_json(
        demo_lab, "/api/chapter/9.9/frame?viz=demo&params=" + encode_params({"mode": "turbo"})
    )
    assert status == 400
    assert "中文模式" in payload["error"] and "fast" in payload["error"]


def test_frame_unknown_param_lists_available_names(lab: LabHTTPServer) -> None:
    status, payload = _get_json(
        lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&params=" + encode_params({"zzz": 1})
    )
    assert status == 400
    assert payload["code"] == "bad_params"
    for name in ("nx", "r", "t_end"):
        assert name in payload["error"]


def test_frame_out_of_range_param_is_400(lab: LabHTTPServer) -> None:
    status, payload = _get_json(
        lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&params=" + encode_params({"r": 99.0})
    )
    assert status == 400
    assert "上界" in payload["error"]


def test_frame_invalid_base64_is_400(lab: LabHTTPServer) -> None:
    status, payload = _get_json(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&params=!!!not-base64")
    assert status == 400
    assert payload["code"] == "bad_params"


def test_frame_params_not_json_object_is_400(lab: LabHTTPServer) -> None:
    encoded = base64.urlsafe_b64encode(b"[1, 2, 3]").decode("ascii").rstrip("=")
    status, payload = _get_json(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&params={encoded}")
    assert status == 400
    assert "必须是 JSON 对象" in payload["error"]


def test_frame_params_invalid_utf8_is_400(lab: LabHTTPServer) -> None:
    encoded = base64.urlsafe_b64encode(b"\xff\xfe\xfa").decode("ascii").rstrip("=")
    status, payload = _get_json(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&params={encoded}")
    assert status == 400
    assert "UTF-8" in payload["error"]


def test_frame_unknown_viz_is_404(lab: LabHTTPServer) -> None:
    status, payload = _get_json(lab, f"/api/chapter/{CHAPTER}/frame?viz=ghost")
    assert status == 404
    assert payload["code"] == "not_found"
    assert SCRUB_VIZ in payload["error"]


def test_frame_missing_viz_is_400_with_available_list(lab: LabHTTPServer) -> None:
    status, payload = _get_json(lab, f"/api/chapter/{CHAPTER}/frame")
    assert status == 400
    assert SCRUB_VIZ in payload["error"] and STREAM_VIZ in payload["error"]


def test_frame_unknown_chapter_is_404(lab: LabHTTPServer) -> None:
    status, payload = _get_json(lab, f"/api/chapter/7.7/frame?viz={SCRUB_VIZ}")
    assert status == 404
    assert "7.7" in payload["error"]


@pytest.mark.parametrize("query", ["t=abc", "t=nan", "t=inf", "t=-0.5", "t=5.0"])
def test_frame_bad_t_is_400(lab: LabHTTPServer, query: str) -> None:
    status, payload = _get_json(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&{query}")
    assert status == 400
    assert payload["code"] == "bad_params"


def test_frame_t_on_timeless_viz_is_400(demo_lab: LabHTTPServer) -> None:
    status, payload = _get_json(demo_lab, "/api/chapter/9.9/frame?viz=demo&t=0.5")
    assert status == 400
    assert "time.mode=none" in payload["error"]
    assert _get(demo_lab, "/api/chapter/9.9/frame?viz=demo&t=0")[0] == 200


def test_frame_cache_is_shared_and_reused(lab: LabHTTPServer) -> None:
    assert isinstance(lab.cache, FrameCache)  # 进程共享一份（FrameCache 自带锁，见 server/frames.py）
    path = f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&params={encode_params({'nx': 11})}"
    first = _get(lab, path)[2]
    entries = lab.cache.n_entries
    second = _get(lab, path)[2]
    assert first == second
    assert entries >= 1
    assert lab.cache.n_entries == entries  # 命中不新增条目


# ---- PROTOCOLS §3：t 对齐 + meta 时间戳 ---------------------------------- #
def test_frame_t_is_aligned_to_dt_and_stamped(lab: LabHTTPServer, real_registry: Registry) -> None:
    """``t=0.103`` 落在 dt=0.005 的第 21 步上：实到 0.105，请求值留在 meta.t_requested。"""
    viz = real_registry.get(CHAPTER).manifest.viz(SCRUB_VIZ)
    assert viz.time.dt == 0.005
    status, headers, body = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&t=0.103")
    assert status == 200
    assert float(headers["X-T"]) == 0.105
    assert float(headers["X-T-Requested"]) == 0.103
    frame = unpack_frame(body)
    assert float(frame.meta["t"]) == 0.105
    assert float(frame.meta["t_actual"]) == 0.105
    assert float(frame.meta["t_requested"]) == 0.103
    _assert_same_frame(frame, build_frame(real_registry.get(CHAPTER), SCRUB_VIZ, viz.defaults(), 0.105))


def test_frame_alignment_rounds_to_nearest_step(lab: LabHTTPServer) -> None:
    for raw, expected in (("0.1024", 0.1), ("0.1025", 0.105), ("0.1076", 0.11), ("0.2", 0.2), ("0", 0.0)):
        _, headers, body = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&t={raw}")
        assert float(headers["X-T"]) == pytest.approx(expected), raw
        frame = unpack_frame(body)
        assert float(frame.meta["t"]) == pytest.approx(expected), raw
        assert float(frame.meta["t_actual"]) == pytest.approx(expected), raw


def test_frame_aligned_request_has_no_redundant_header(lab: LabHTTPServer) -> None:
    _, headers, body = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&t=0.105")
    frame = unpack_frame(body)
    assert float(frame.meta["t_requested"]) == 0.105 == float(frame.meta["t_actual"])
    assert "X-T-Requested" not in headers  # 请求值 == 实到值，不必再报一次


def test_frame_alignment_writes_t_actual_for_every_request(lab: LabHTTPServer) -> None:
    """``meta.t_actual`` 是契约字段：任何 ``/frame`` 响应都必须带（PROTOCOLS §3 的 MUST）。"""
    for query in ("", "t=0", "t=0.19999"):
        _, _, body = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&{query}")
        assert float(unpack_frame(body).meta["t_actual"]) == float(unpack_frame(body).meta["t"])


def test_frame_small_t_aligns_to_zero(lab: LabHTTPServer) -> None:
    """``%.6g`` 写法（t=1.5e-05）要能解析，且比半步还小 → 对齐到 0。"""
    status, headers, body = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&t=1.5e-05")
    assert status == 200
    assert float(headers["X-T"]) == 0.0
    assert float(unpack_frame(body).meta["t_requested"]) == 1.5e-05


@pytest.mark.parametrize(
    "raw,expected",
    [(0.0, 0.0), (0.2, 0.0), (0.3, 0.5), (0.75, 1.0), (1.0, 1.0), (99.0, 1.0)],
)
def test_align_t_snaps_to_nearest_step_and_clamps(raw: float, expected: float) -> None:
    """合成 viz ``boom`` 的 dt=0.5 / tMax=1.0：最近整步，越界夹到上界。"""
    assert align_t(load_manifest(DEMO_MANIFEST, where="测试合成章节").viz("boom"), raw) == pytest.approx(expected)


def test_align_t_timeless_viz_is_zero() -> None:
    viz = load_manifest(DEMO_MANIFEST, where="测试合成章节").viz("demo")  # time.mode = none
    assert align_t(viz, 0.0) == 0.0


def test_frame_alignment_shares_one_base_frame_per_step(lab: LabHTTPServer) -> None:
    """同一格点上的不同拖动值：物理量只算一次，各自带回自己的 ``t_requested``。"""
    query = f"params={encode_params({'nx': 11})}"
    before = lab.cache.n_entries
    _, _, a = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&{query}&t=0.1027")
    _, _, b = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&{query}&t=0.1031")
    fa, fb = unpack_frame(a), unpack_frame(b)
    assert float(fa.meta["t_actual"]) == float(fb.meta["t_actual"]) == 0.105
    assert float(fa.meta["t_requested"]) == 0.1027 and float(fb.meta["t_requested"]) == 0.1031
    np.testing.assert_array_equal(fa["u"], fb["u"])
    assert lab.cache.n_entries - before == 3  # 基准帧 1 条 + 两个请求各 1 条（没有重算第三份）


def test_frame_matches_precomputed_l3_file_at_same_grid_point(
    tmp_path: pathlib.Path, real_registry: Registry, serve: Callable[..., LabHTTPServer]
) -> None:
    """L2 动态帧与 L3 静态帧必须指向同一时刻、同一数组（只差服务端写的时间戳）。"""
    from server.frames import static_frame_path

    assert _precompute_main(["--chapter", CHAPTER, "--n", "2", "--out-root", str(tmp_path), "--quiet"]) == 0
    viz = real_registry.get(CHAPTER).manifest.viz(SCRUB_VIZ)
    path = static_frame_path(tmp_path, CHAPTER, SCRUB_VIZ, viz.defaults(), 0.2)
    assert path.is_file(), sorted(str(p) for p in tmp_path.glob("data/frames/*/*/*.nlf"))
    static = unpack_frame(path.read_bytes())
    httpd = serve(root=tmp_path, registry=real_registry)
    _, headers, body = _get(httpd, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&t=0.2")
    live = unpack_frame(body)
    assert float(headers["X-T"]) == float(static.meta["t"]) == 0.2
    for name in static.names:
        np.testing.assert_array_equal(live[name], static[name], err_msg=f"数组 {name} 与 L3 不一致")
    assert str(path).endswith("_t0.2.nlf")  # 文件名里的写法与 meta.t 同款（%.6g）


# --------------------------------------------------------------------------- #
# 4. /api/chapter/<id>/frames （多帧流）
# --------------------------------------------------------------------------- #
def test_frames_stream_format_parseable_by_hand(lab: LabHTTPServer) -> None:
    status, headers, body = _get(lab, f"/api/chapter/{CHAPTER}/frames?viz={SCRUB_VIZ}&n=5")
    assert status == 200
    assert headers["Content-Type"] == STREAM_CONTENT_TYPE
    assert headers["X-Nlf-Stream"] == "NLFS/1"
    assert headers["X-Frame-Count"] == "5"
    header, frames = _parse_stream_manual(body)  # 独立解析
    assert header["format"] == "nlf-stream" and header["version"] == 1
    assert header["chapter"] == CHAPTER and header["viz"] == SCRUB_VIZ
    assert header["n"] == len(frames) == 5
    assert len(header["t"]) == 5
    assert header["t"][0] == 0.0 and header["t"] == sorted(header["t"])
    for blob, t in zip(frames, header["t"]):
        frame = unpack_frame(blob)  # 每条记录都是完整 .nlf
        assert float(frame.meta["t"]) == float(t)


def test_frames_stream_helper_agrees_with_manual_parse(lab: LabHTTPServer) -> None:
    _, _, body = _get(lab, f"/api/chapter/{CHAPTER}/frames?viz={SCRUB_VIZ}&n=4")
    header, frames = read_frame_stream(body)
    manual_header, manual_frames = _parse_stream_manual(body)
    assert header == manual_header
    assert frames == manual_frames


def test_frames_stream_records_equal_single_frame_endpoint(lab: LabHTTPServer) -> None:
    _, _, body = _get(lab, f"/api/chapter/{CHAPTER}/frames?viz={SCRUB_VIZ}&n=3")
    header, frames = _parse_stream_manual(body)
    for blob, t in zip(frames, header["t"]):
        _, _, single = _get(lab, f"/api/chapter/{CHAPTER}/frame?viz={SCRUB_VIZ}&t={t!r}")
        _assert_same_frame(unpack_frame(blob), unpack_frame(single))


def test_frames_grid_frames_are_unstamped_and_equal_build_frame(
    lab: LabHTTPServer, real_registry: Registry
) -> None:
    """多帧流按栅格原样取帧：与 ``build_frame``+``pack_frame`` 逐字节同源（L3 同款）。"""
    viz = real_registry.get(CHAPTER).manifest.viz(SCRUB_VIZ)
    _, _, body = _get(lab, f"/api/chapter/{CHAPTER}/frames?viz={SCRUB_VIZ}&n=3")
    header, frames = _parse_stream_manual(body)
    last = unpack_frame(frames[-1])
    assert float(last.meta["t"]) == float(header["t"][-1])
    expected = pack_frame(build_frame(real_registry.get(CHAPTER), SCRUB_VIZ, viz.defaults(), header["t"][-1]))
    assert frames[-1] == expected


def test_frames_grid_frames_carry_no_server_timestamps(demo_lab: LabHTTPServer) -> None:
    """服务端只给 ``/frame``（点名时刻）打时间戳；栅格帧的 meta 由章节自己说了算。"""
    _, _, body = _get(demo_lab, "/api/chapter/9.9/frames?viz=demo&n=1")
    _, frames = _parse_stream_manual(body)
    meta = unpack_frame(frames[0]).meta
    assert "t_actual" not in meta and "t_requested" not in meta


def test_frames_default_n_is_one(lab: LabHTTPServer) -> None:
    _, headers, body = _get(lab, f"/api/chapter/{CHAPTER}/frames?viz={STREAM_VIZ}")
    assert headers["X-Frame-Count"] == "1"
    header, frames = _parse_stream_manual(body)
    assert header["n"] == 1 and len(frames) == 1
    assert header["t"] == [0.0]


def test_frames_respects_time_range(lab: LabHTTPServer) -> None:
    _, _, body = _get(lab, f"/api/chapter/{CHAPTER}/frames?viz={SCRUB_VIZ}&n=3")
    header, _ = _parse_stream_manual(body)
    assert header["t"][0] == 0.0
    assert header["t"][-1] == pytest.approx(0.2)  # tMax


@pytest.mark.parametrize("n", ["0", "-3", "abc", "1.5", str(MAX_STREAM_FRAMES + 1)])
def test_frames_bad_n_is_400(lab: LabHTTPServer, n: str) -> None:
    status, payload = _get_json(lab, f"/api/chapter/{CHAPTER}/frames?viz={SCRUB_VIZ}&n={n}")
    assert status == 400
    assert payload["code"] == "bad_params"


def test_frames_n_on_timeless_viz_is_400(demo_lab: LabHTTPServer) -> None:
    status, payload = _get_json(demo_lab, "/api/chapter/9.9/frames?viz=demo&n=3")
    assert status == 400
    assert "n=1" in payload["error"]
    assert _get(demo_lab, "/api/chapter/9.9/frames?viz=demo&n=1")[0] == 200


def test_frames_unknown_viz_is_404(lab: LabHTTPServer) -> None:
    status, payload = _get_json(lab, f"/api/chapter/{CHAPTER}/frames?viz=ghost&n=2")
    assert status == 404
    assert payload["code"] == "not_found"


# --------------------------------------------------------------------------- #
# 5. 静态资源与目录穿越
# --------------------------------------------------------------------------- #
def test_site_shell_served_at_root(
    serve: Callable[..., LabHTTPServer], real_registry: Registry, temp_root: pathlib.Path
) -> None:
    httpd = serve(root=temp_root, registry=real_registry)
    status, headers, body = _get(httpd, "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert body == (temp_root / "viz" / "site" / "index.html").read_bytes()


def test_root_on_real_repo_is_shell_or_diagnosable_404(lab: LabHTTPServer) -> None:
    """真仓库的 ``viz/site/index.html`` 由站点任务负责；在它落地前后都必须给出确定行为。"""
    index = ROOT / "viz" / "site" / "index.html"
    status, headers, body = _get(lab, "/")
    if index.is_file():
        assert status == 200 and body == index.read_bytes()
    else:
        assert status == 404
        assert "viz/site/index.html" in json.loads(body)["error"]


def test_viz_dist_and_data_mounts(
    serve: Callable[..., LabHTTPServer], real_registry: Registry, temp_root: pathlib.Path
) -> None:
    httpd = serve(root=temp_root, registry=real_registry)
    status, headers, body = _get(httpd, "/viz/app.js")
    assert (status, headers["Content-Type"], body) == (200, "text/javascript; charset=utf-8", b"export const answer = 42;\n")
    status, headers, body = _get(httpd, "/dist/viz/bundle.js")
    assert status == 200 and body == b"// bundle\n"
    assert headers["Content-Type"].startswith("text/javascript")
    status, headers, body = _get(httpd, "/data/pre.nlf")
    assert status == 200 and body == b"NLF1-pretend"
    assert headers["Content-Type"] == FRAME_CONTENT_TYPE


def test_data_dir_flag_moves_the_data_mount(
    serve: Callable[..., LabHTTPServer], real_registry: Registry, temp_root: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    other = tmp_path / "outside-data"
    other.mkdir()
    (other / "frame.nlf").write_bytes(b"NLF1-from-override")
    httpd = serve(root=temp_root, data_dir=other, registry=real_registry)
    status, _, body = _get(httpd, "/data/frame.nlf")
    assert status == 200 and body == b"NLF1-from-override"
    assert _get(httpd, "/data/pre.nlf")[0] == 404  # 原 data/ 已被换掉


def test_static_missing_file_is_404(lab: LabHTTPServer) -> None:
    status, payload = _get_json(lab, "/viz/site/definitely-missing.js")
    assert status == 404
    assert payload["code"] == "not_found"


def test_static_directory_is_not_listed(
    serve: Callable[..., LabHTTPServer], real_registry: Registry, temp_root: pathlib.Path
) -> None:
    httpd = serve(root=temp_root, registry=real_registry)
    assert _get(httpd, "/viz/site/")[0] == 404
    assert _get(httpd, "/viz/")[0] == 404


def test_resolve_static_rejects_traversal_without_server(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "root"
    (root / "viz").mkdir(parents=True)
    (root / "viz" / "ok.js").write_text("ok", encoding="utf-8")
    assert resolve_static(root / "viz", "ok.js") == (root / "viz" / "ok.js").resolve()
    assert resolve_static(root / "viz", "missing.js") is None
    for evil in ("../secret.txt", "..", "a/../../x", "/etc/passwd", "C:/windows", "\\..\\x"):
        with pytest.raises(RequestError) as err:
            resolve_static(root / "viz", evil)
        assert err.value.status == 403


@pytest.mark.parametrize(
    "target",
    [
        "/viz/../server/app.py",  # 1) .. 段
        "/viz/%2e%2e/%2e%2e/server/app.py",  # 1b) 百分号编码的 ..
        "/viz/..%5C..%5Cserver/app.py",  # 2) 反斜杠变体
        "/viz//etc/passwd",  # 3) 挂载点之后是绝对路径
        "/viz/%2Fetc%2Fpasswd",  # 3b) 编码后的绝对路径
        "/viz/%00app.js",  # 4) NUL 字节
        "/data/../../etc/passwd",  # 另一个挂载点同样要挡
    ],
)
def test_static_traversal_is_403(lab: LabHTTPServer, target: str) -> None:
    status, body = _raw_status_body(lab, target)
    assert status == 403, (target, body)
    payload = json.loads(body.decode("utf-8"))
    assert payload["code"] == "path_forbidden"
    assert b"import argparse" not in body  # 绝没有把 server/app.py 交出去


def test_static_symlink_escape_is_403(
    serve: Callable[..., LabHTTPServer], real_registry: Registry, tmp_path: pathlib.Path
) -> None:
    root = tmp_path / "lab"
    (root / "viz").mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    link = root / "viz" / "leak.txt"
    try:
        os.symlink(secret, link)
    except OSError as exc:  # pragma: no cover - 平台不支持软链
        pytest.skip(f"平台不支持创建符号链接：{exc}")
    httpd = serve(root=root, registry=real_registry)
    status, body = _raw_status_body(httpd, "/viz/leak.txt")
    assert status == 403
    assert b"TOP-SECRET" not in body
    assert json.loads(body.decode("utf-8"))["code"] == "path_forbidden"


def test_static_symlink_inside_root_is_allowed(
    serve: Callable[..., LabHTTPServer], real_registry: Registry, tmp_path: pathlib.Path
) -> None:
    """软链本身不是罪：只要 resolve() 之后仍在静态根之内就该正常服务。"""
    root = tmp_path / "lab"
    (root / "viz").mkdir(parents=True)
    real = root / "viz" / "real.js"
    real.write_text("ok", encoding="utf-8")
    try:
        os.symlink(real, root / "viz" / "alias.js")
    except OSError as exc:  # pragma: no cover - 平台不支持软链
        pytest.skip(f"平台不支持创建符号链接：{exc}")
    httpd = serve(root=root, registry=real_registry)
    status, _, body = _get(httpd, "/viz/alias.js")
    assert status == 200 and body == b"ok"


def test_head_has_headers_but_no_body(lab: LabHTTPServer) -> None:
    response = _raw(lab, b"HEAD /api/health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
    head, _, body = response.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 200 ")
    assert b"Content-Length: " in head
    assert body == b""


def test_post_is_405_outside_asserts(lab: LabHTTPServer) -> None:
    status, body = _raw_status_body(lab, "/api/health", method="POST")
    assert status == 405
    assert json.loads(body.decode("utf-8"))["code"] == "method_not_allowed"


def test_unknown_method_error_is_json(lab: LabHTTPServer) -> None:
    """http.server 自己产生的 501 也要是 JSON（``send_error`` 被覆盖）。"""
    status, body = _raw_status_body(lab, "/api/health", method="DELETE")
    assert status == 501
    payload = json.loads(body.decode("utf-8"))
    assert payload["code"] == 501


# --------------------------------------------------------------------------- #
# 6. /api/asserts/<id>（--allow-write 开关）
# --------------------------------------------------------------------------- #
def test_asserts_forbidden_without_allow_write(lab: LabHTTPServer) -> None:
    for method in ("GET", "POST"):
        status, body = _raw_status_body(lab, f"/api/asserts/{CHAPTER}", method=method)
        assert status == 403, method
        payload = json.loads(body.decode("utf-8"))
        assert payload["code"] == "forbidden"
        assert "--allow-write" in payload["error"]


def test_asserts_missing_report_is_404_with_allow_write(
    serve: Callable[..., LabHTTPServer], real_registry: Registry, temp_root: pathlib.Path
) -> None:
    httpd = serve(root=temp_root, registry=real_registry, allow_write=True)
    status, payload = _get_json(httpd, f"/api/asserts/{CHAPTER}")
    assert status == 404
    assert "run_asserts" in payload["error"]


def test_asserts_returns_chapter_section(
    serve: Callable[..., LabHTTPServer], real_registry: Registry, temp_root: pathlib.Path
) -> None:
    reports = temp_root / "dist" / "reports"
    reports.mkdir(parents=True)
    (reports / "assertions.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00+0000",
                "totals": {"A": 1, "B": 0, "C": 0, "passed": 1, "failed": 0},
                "chapters": {CHAPTER: {"declared": 1, "collected": 1, "passed": 1, "items": []}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    httpd = serve(root=temp_root, registry=real_registry, allow_write=True)
    status, payload = _get_json(httpd, f"/api/asserts/{CHAPTER}")
    assert status == 200
    assert payload["chapter"] == CHAPTER
    assert payload["report"]["passed"] == 1
    assert payload["totals"]["A"] == 1
    assert payload["source"] == "dist/reports/assertions.json"
    # POST 与 GET 等价（PROTOCOLS §3 写的是 POST）
    status_post, body = _raw_status_body(httpd, f"/api/asserts/{CHAPTER}", method="POST")
    assert status_post == 200
    assert json.loads(body.decode("utf-8"))["chapter"] == CHAPTER


def test_asserts_unknown_chapter_is_404(serve: Callable[..., LabHTTPServer], real_registry: Registry) -> None:
    httpd = serve(root=ROOT, registry=real_registry, allow_write=True)
    status, payload = _get_json(httpd, "/api/asserts/7.7")
    assert status == 404
    assert payload["code"] == "not_found"


def test_broken_report_json_is_500_without_traceback(
    serve: Callable[..., LabHTTPServer], real_registry: Registry, temp_root: pathlib.Path
) -> None:
    reports = temp_root / "dist" / "reports"
    reports.mkdir(parents=True)
    (reports / "assertions.json").write_text("{ not json", encoding="utf-8")
    httpd = serve(root=temp_root, registry=real_registry, allow_write=True)
    status, payload = _get_json(httpd, f"/api/asserts/{CHAPTER}")
    assert status == 500
    assert "Traceback" not in json.dumps(payload)


# --------------------------------------------------------------------------- #
# 7. 内部错误：500 且不泄漏 traceback
# --------------------------------------------------------------------------- #
def test_internal_error_is_500_without_traceback(serve: Callable[..., LabHTTPServer]) -> None:
    httpd = serve(root=ROOT, registry=_BoomRegistry([]))
    status, headers, body = _get(httpd, "/api/chapter/1.1/manifest")
    assert status == 500
    text = body.decode("utf-8")
    assert "SECRET" not in text
    assert "Traceback" not in text
    assert "app.py" not in text and "/etc/shadow" not in text
    payload = json.loads(text)
    assert payload["code"] == "internal_error"
    assert payload["request_id"]
    assert headers["Content-Type"].startswith("application/json")


# --------------------------------------------------------------------------- #
# 8. WebSocket（PROTOCOLS §2）
# --------------------------------------------------------------------------- #
def test_ws_ready_is_first_message_with_manifest(lab: LabHTTPServer, real_registry: Registry) -> None:
    with WebSocketClient.connect(_ws_url(lab), timeout=10.0) as client:
        ready = client.recv_json()
    assert ready is not None and ready["type"] == "ready"
    assert ready["chapter"] == CHAPTER
    assert ready["manifest"] == real_registry.get(CHAPTER).manifest.to_json()


def test_ws_subscribe_pushes_nlf_frame_matching_build_frame(lab: LabHTTPServer, real_registry: Registry) -> None:
    params = {"omega": 2.0, "zeta": 0.3, "n_periods": 2}
    with WebSocketClient.connect(_ws_url(lab), timeout=10.0) as client:
        client.recv_json()  # ready
        client.send_json({"type": "subscribe", "chapter": CHAPTER, "viz": STREAM_VIZ, "params": params})
        ack = client.recv_json()
        assert ack is not None and ack["type"] == "ready" and ack["viz"] == STREAM_VIZ
        assert ack["params"] == params
        frames = _read_push(client, 1)
    expected = build_frame(real_registry.get(CHAPTER), STREAM_VIZ, params, 0.0)
    _assert_same_frame(frames[0], expected)


def test_ws_subscribe_time_frames_streams_n_frames(lab: LabHTTPServer) -> None:
    with WebSocketClient.connect(_ws_url(lab), timeout=15.0) as client:
        client.recv_json()
        client.send_json(
            {
                "type": "subscribe",
                "viz": SCRUB_VIZ,
                "params": {"nx": 21},
                "time": {"mode": "stream", "frames": 4, "fps": 0},
            }
        )
        client.recv_json()
        frames = _read_push(client, 4)
    assert [float(f.meta["t"]) for f in frames] == pytest.approx([0.0, 0.005, 0.01, 0.015])


def test_ws_params_message_recomputes_exactly_one_frame(lab: LabHTTPServer) -> None:
    with WebSocketClient.connect(_ws_url(lab), timeout=10.0) as client:
        client.recv_json()
        client.send_json({"type": "subscribe", "viz": STREAM_VIZ, "params": {"zeta": 0.05}})
        client.recv_json()
        _read_push(client, 1)
        client.send_json({"type": "params", "params": {"zeta": 1.5}})  # 增量合并
        frames = _read_push(client, 1)
    assert float(frames[0].meta["zeta"]) == 1.5
    assert float(frames[0].meta["omega"]) == 1.0  # 未提到的参数保持原值


def test_ws_bad_params_error_keeps_connection_alive(lab: LabHTTPServer) -> None:
    with WebSocketClient.connect(_ws_url(lab), timeout=10.0) as client:
        client.recv_json()
        client.send_json({"type": "subscribe", "viz": SCRUB_VIZ, "params": {"zzz": 1}})
        error = client.recv_json()
        assert error is not None and error["type"] == "error" and error["code"] == "bad_params"
        for name in ("nx", "r", "t_end"):
            assert name in error["message"]
        client.send_json({"type": "ping", "t": 7})
        pong = client.recv_json()  # 连接还能用
        assert pong is not None and pong["type"] == "pong" and pong["t"] == 7


def test_ws_unknown_viz_and_chapter_mismatch_are_errors(lab: LabHTTPServer) -> None:
    with WebSocketClient.connect(_ws_url(lab), timeout=10.0) as client:
        client.recv_json()
        client.send_json({"type": "subscribe", "viz": "ghost"})
        error = client.recv_json()
        assert error is not None and error["code"] == "unknown_viz" and SCRUB_VIZ in error["message"]
        client.send_json({"type": "subscribe", "chapter": "1.2", "viz": SCRUB_VIZ})
        mismatch = client.recv_json()
        assert mismatch is not None and mismatch["code"] == "chapter_mismatch"


def test_ws_commands_before_subscribe_are_errors(lab: LabHTTPServer) -> None:
    with WebSocketClient.connect(_ws_url(lab), timeout=10.0) as client:
        client.recv_json()
        for message in ({"type": "params", "params": {"nx": 11}}, {"type": "step", "n": 2}, {"type": "seek", "t": 0.1}):
            client.send_json(message)
            error = client.recv_json()
            assert error is not None and error["type"] == "error" and error["code"] == "no_subscription"


def test_ws_unknown_type_and_missing_type(lab: LabHTTPServer) -> None:
    with WebSocketClient.connect(_ws_url(lab), timeout=10.0) as client:
        client.recv_json()
        client.send_json({"type": "teleport"})
        error = client.recv_json()
        assert error is not None and error["code"] == "unknown_type" and "subscribe" in error["message"]
        client.send_json({"nope": 1})
        missing = client.recv_json()
        assert missing is not None and missing["code"] == "bad_request"


def test_ws_binary_message_from_client_is_error_not_crash(lab: LabHTTPServer) -> None:
    with WebSocketClient.connect(_ws_url(lab), timeout=10.0) as client:
        client.recv_json()
        client.send_binary(b"NLF1-not-a-control-message")
        error = client.recv_json()
        assert error is not None and error["type"] == "error" and error["code"] == "bad_message"
        client.send_json({"type": "ping"})
        assert client.recv_json()["type"] == "pong"


def test_ws_seek_and_step(lab: LabHTTPServer) -> None:
    with WebSocketClient.connect(_ws_url(lab), timeout=15.0) as client:
        client.recv_json()
        client.send_json({"type": "subscribe", "viz": SCRUB_VIZ})
        client.recv_json()
        _read_push(client, 1)
        client.send_json({"type": "seek", "t": 0.1})
        assert [float(f.meta["t"]) for f in _read_push(client, 1)] == pytest.approx([0.1])
        client.send_json({"type": "step", "n": 3})
        assert [float(f.meta["t"]) for f in _read_push(client, 3)] == pytest.approx([0.105, 0.11, 0.115])
        client.send_json({"type": "seek", "t": 9.0})  # 越界 → error，连接不断
        error = client.recv_json()
        assert error is not None and error["code"] == "bad_request"
        client.send_json({"type": "ping"})
        assert client.recv_json()["type"] == "pong"


def test_ws_seek_aligns_to_dt_and_stamps_time(lab: LabHTTPServer) -> None:
    """``seek`` 与 ``/frame`` 同一口径：对齐到最近整步 + 写回 meta 三件套。"""
    with WebSocketClient.connect(_ws_url(lab), timeout=15.0) as client:
        client.recv_json()
        client.send_json({"type": "subscribe", "viz": SCRUB_VIZ})
        client.recv_json()
        _read_push(client, 1)
        client.send_json({"type": "seek", "t": 0.103})
        frames = _read_push(client, 1)
    assert float(frames[0].meta["t"]) == 0.105
    assert float(frames[0].meta["t_actual"]) == 0.105
    assert float(frames[0].meta["t_requested"]) == 0.103


def test_ws_play_streams_until_done_and_pause_cancel_are_done(lab: LabHTTPServer) -> None:
    with WebSocketClient.connect(_ws_url(lab), timeout=15.0) as client:
        client.recv_json()
        client.send_json({"type": "subscribe", "viz": SCRUB_VIZ})
        client.recv_json()
        _read_push(client, 1)
        client.send_json({"type": "play", "time": {"frames": 3, "fps": 0}})
        assert len(_read_push(client, 3)) == 3
        client.send_json({"type": "pause"})
        done = client.recv_json()
        assert done is not None and done["type"] == "done" and done["frames_sent"] == 0
        client.send_json({"type": "cancel"})
        assert client.recv_json()["type"] == "done"


def test_ws_unknown_chapter_handshake_rejected_with_404(serve: Callable[..., LabHTTPServer], real_registry: Registry) -> None:
    httpd = serve(root=ROOT, registry=real_registry)
    with pytest.raises(WsHandshakeError) as err:
        WebSocketClient.connect(_ws_url(httpd, "/ws/chapter/7.7"), timeout=10.0)
    assert err.value.status == 404


def test_ws_unknown_path_handshake_rejected(lab: LabHTTPServer) -> None:
    with pytest.raises(WsHandshakeError) as err:
        WebSocketClient.connect(_ws_url(lab, "/ws/nope"), timeout=10.0)
    assert err.value.status == 404


def test_ws_two_clients_do_not_mix_frames(lab: LabHTTPServer) -> None:
    results: dict[str, list[float]] = {}
    lock = threading.Lock()

    def worker(tag: str, omega: float) -> None:
        with WebSocketClient.connect(_ws_url(lab), timeout=20.0) as client:
            client.recv_json()
            client.send_json({"type": "subscribe", "viz": STREAM_VIZ, "params": {"omega": omega}})
            client.recv_json()
            frames = _read_push(client, 1)
            with lock:
                results[tag] = [float(frames[0].meta["omega"]), float(frames[0]["t"][1])]

    threads = [threading.Thread(target=worker, args=(f"c{i}", float(i + 1))) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
    assert results["c0"][0] == 1.0 and results["c1"][0] == 2.0
    assert results["c0"][1] != results["c1"][1]  # 各自的帧内容确实不同


def test_ws_pipelined_first_frame_is_not_lost(lab: LabHTTPServer) -> None:
    """握手请求与首帧挤在同一次 write 里：``rfile`` 预读的字节必须交回 WS 层。"""
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    handshake = (
        f"GET /ws/chapter/{CHAPTER} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    ).encode("ascii")
    payload = json.dumps({"type": "ping", "t": 99}).encode("utf-8")
    mask = os.urandom(4)
    masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
    frame = bytes([0x81, 0x80 | len(payload)]) + mask + masked
    response = _raw_until(lab, handshake + frame, b'"pong"')
    assert response.startswith(b"HTTP/1.1 101 ")
    body = response.split(b"\r\n\r\n", 1)[1]
    assert b'"ready"' in body
    assert b'"pong"' in body and b"99" in body  # 首帧确实被处理了（没有丢在 rfile 缓冲里）


def test_ws_compute_failure_reports_error_before_any_frames_announcement(demo_lab: LabHTTPServer) -> None:
    """帧算不出来时只许回 ``error``：``frames``(n) 一旦发出去，客户端就在等 n 个二进制帧。"""
    with WebSocketClient.connect(_ws_url(demo_lab, "/ws/chapter/9.9"), timeout=10.0) as client:
        client.recv_json()  # 首条 ready
        client.send_json({"type": "subscribe", "viz": "boom", "time": {"frames": 3, "fps": 0}})
        ack = client.recv_json()
        assert ack is not None and ack["type"] == "ready"
        error = client.recv_json()  # 紧跟其后的必须是 error，而不是 frames 预告
        assert error is not None and error["type"] == "error" and error["code"] == "internal"
        assert "合成故障" in error["message"]
        client.send_json({"type": "ping"})  # 连接仍然可用
        assert client.recv_json()["type"] == "pong"


def test_ws_client_disconnect_leaves_server_healthy(lab: LabHTTPServer) -> None:
    client = WebSocketClient.connect(_ws_url(lab), timeout=10.0)
    assert client.recv_json()["type"] == "ready"
    client.close(1000)
    # 服务端必须还能继续服务其他请求
    assert _get_json(lab, "/api/health")[0] == 200


def test_main_prints_real_port_and_stops_on_sigint() -> None:
    """``--port 0`` 必须打印内核分配的真实端口；SIGINT 必须干净退出（退出码 0）。"""
    import signal
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.app", "--port", "0", "--quiet"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        banner = ""
        for _ in range(20):
            line = proc.stdout.readline()  # type: ignore[union-attr]
            if "http://" in line:
                banner = line.strip()
                break
        assert banner, "没有打印监听地址"
        port = int(banner.rsplit(":", 1)[1].split("/")[0])
        assert port > 0, banner
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=15) as resp:
            assert json.loads(resp.read())["ok"] is True
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=20) == 0, "SIGINT 后应当以 0 退出"
        assert "服务已停止" in proc.stdout.read()  # type: ignore[union-attr]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


# --------------------------------------------------------------------------- #
# 9. tools/precompute.py
# --------------------------------------------------------------------------- #
def _precompute_main(argv: list[str]) -> int:
    from tools.precompute import main as precompute_main

    return precompute_main(argv)


def test_precompute_writes_readable_frames(tmp_path: pathlib.Path, real_registry: Registry) -> None:
    from server.frames import static_frame_path

    assert _precompute_main(["--chapter", CHAPTER, "--n", "2", "--out-root", str(tmp_path), "--quiet"]) == 0
    written = sorted(tmp_path.glob("data/frames/*/*/*.nlf"))
    assert written, "预计算没有写出任何文件"
    for path in written:
        frame = unpack_frame(path.read_bytes())
        assert frame.chapter == CHAPTER
        assert "error" in frame.names or "residual" in frame.names
    # 路径约定与 server.frames.static_frame_path 一致（L3 查表要靠它）
    default_params = real_registry.get(CHAPTER).manifest.viz(SCRUB_VIZ).defaults()
    expected = static_frame_path(tmp_path, CHAPTER, SCRUB_VIZ, default_params, 0.0)
    assert expected in written


def test_precompute_limit_caps_total_frames(tmp_path: pathlib.Path) -> None:
    assert _precompute_main(["--chapter", CHAPTER, "--n", "5", "--limit", "3", "--out-root", str(tmp_path), "--quiet"]) == 0
    assert len(list(tmp_path.glob("data/frames/*/*/*.nlf"))) == 3


def test_precompute_json_summary_reports_files_and_bytes(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    from tools.precompute import _parse_args, run

    args = _parse_args(["--chapter", CHAPTER, "--n", "1", "--out-root", str(tmp_path), "--quiet"])
    result = run(args)
    assert result["ok"] is True and result["files"] == len(result["written"]) > 0
    assert result["bytes"] == sum(pathlib.Path(p).stat().st_size for p in result["written"])
    assert capsys.readouterr().out == ""  # --quiet 真的安静


def test_precompute_prints_counts(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _precompute_main(["--chapter", CHAPTER, "--n", "1", "--out-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "合计：" in out and "个文件" in out and "字节" in out


def test_precompute_is_idempotent(tmp_path: pathlib.Path) -> None:
    argv = ["--chapter", CHAPTER, "--n", "2", "--out-root", str(tmp_path), "--quiet"]
    assert _precompute_main(argv) == 0
    first = sorted(str(p) for p in tmp_path.glob("data/frames/*/*/*.nlf"))
    assert _precompute_main(argv) == 0
    assert sorted(str(p) for p in tmp_path.glob("data/frames/*/*/*.nlf")) == first


def test_precompute_prune_removes_stale_keeps_valid(tmp_path: pathlib.Path, real_registry: Registry) -> None:
    from server.frames import static_frame_path

    assert _precompute_main(["--chapter", CHAPTER, "--n", "1", "--out-root", str(tmp_path), "--quiet"]) == 0
    defaults = real_registry.get(CHAPTER).manifest.viz(SCRUB_VIZ).defaults()
    valid = static_frame_path(tmp_path, CHAPTER, SCRUB_VIZ, defaults, 0.0)
    assert valid.is_file()
    ghost_chapter = tmp_path / "data" / "frames" / "9.9" / "ghost"
    ghost_chapter.mkdir(parents=True)
    (ghost_chapter / "stale.nlf").write_bytes(b"NLF1-stale")
    ghost_viz = tmp_path / "data" / "frames" / CHAPTER / "removed_viz"
    ghost_viz.mkdir(parents=True)
    (ghost_viz / "stale.nlf").write_bytes(b"NLF1-stale")

    assert _precompute_main(["--chapter", CHAPTER, "--n", "1", "--out-root", str(tmp_path), "--prune", "--quiet"]) == 0

    assert not (tmp_path / "data" / "frames" / "9.9").exists(), "没有对应章节的目录必须被删掉"
    assert not ghost_viz.exists(), "没有对应可视化的目录必须被删掉"
    assert valid.is_file(), "有效帧不能被误删"


def test_precompute_prune_without_frames_dir_is_noop(tmp_path: pathlib.Path, real_registry: Registry) -> None:
    from tools.precompute import prune_stale

    removed, files = prune_stale(tmp_path, real_registry)
    assert removed == [] and files == 0


def test_precompute_prune_ignores_loose_files(tmp_path: pathlib.Path, real_registry: Registry) -> None:
    """``data/frames`` 下的散落文件不是"章节目录"，清理工具不许碰它们。"""
    from tools.precompute import data_frames_dir, prune_stale

    base = data_frames_dir(tmp_path)
    base.mkdir(parents=True)
    loose = base / "loose.nlf"
    loose.write_bytes(b"NLF1-loose")
    removed, files = prune_stale(tmp_path, real_registry)
    assert removed == [] and files == 0
    assert loose.is_file()


def test_precompute_unknown_chapter_exits_1(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _precompute_main(["--chapter", "42.42", "--out-root", str(tmp_path)]) == 1
    assert "未知章节" in capsys.readouterr().err
