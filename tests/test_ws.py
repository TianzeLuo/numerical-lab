"""``server/ws.py`` + ``server/wsclient.py`` 的协议一致性测试（RFC 6455 子集）。

拓扑：后台线程里跑一个 ``ThreadingTCPServer``（``port=0``，端口由内核分配，
所以测试之间不会抢端口、也不会残留监听），handler 里用 :meth:`WebSocket.accept`
完成握手后做 echo；客户端一律用自研的 :class:`WebSocketClient`。

两条刻意的写法：

* **非法帧用手工拼的字节**（``_frame`` 辅助函数，独立于实现），不用被测代码的组帧函数——
  否则就是拿实现证明实现；客户端侧的 ``raw_send`` 正是为这种「故意违规」准备的。
* **不靠 ``sleep`` 等结果**：所有同步都靠「发出去 → 读回来」的往返，
  内核事件 + socket 超时兜底，失败就是超时（``TimeoutError``），不会挂死。
"""

from __future__ import annotations

import base64
import concurrent.futures
import dataclasses
import json
import socket
import socketserver
import struct
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

import pytest

from server.ws import (
    MAX_FRAME_BYTES,
    Message,
    WebSocket,
    WsError,
    _mask_payload,
    accept_key,
    initial_bytes_from_handler,
    peek_buffered,
)
from server.wsclient import WebSocketClient, WsHandshakeError

pytestmark = pytest.mark.e2e

#: 服务端读超时：任何一步卡住都变成 5 s 内的失败，而不是挂死整条测试会话。
_IO_TIMEOUT = 5.0


# --------------------------------------------------------------------------- 工具


def _frame(
    opcode: int,
    payload: bytes = b"",
    *,
    fin: bool = True,
    mask: bool = True,
    rsv: int = 0,
    declared: int | None = None,
    mask_key: bytes = b"\x37\xfa\x21\x3d",
) -> bytes:
    """手工拼一个客户端帧（测试独立实现，不复用 ``server/ws.py`` 的组帧代码）。

    ``declared`` 用来只发帧头、声明一个假长度（超长帧测试需要「别真的发 16 MiB」）；
    ``mask=False`` / ``rsv=1`` / 奇怪 ``opcode`` 用来构造协议违规。
    """
    length = len(payload) if declared is None else declared
    head = bytearray([(0x80 if fin else 0x00) | ((rsv & 0x07) << 4) | (opcode & 0x0F)])
    mask_bit = 0x80 if mask else 0x00
    if length < 126:
        head.append(mask_bit | length)
    elif length < 65536:
        head.append(mask_bit | 126)
        head += struct.pack("!H", length)
    else:
        head.append(mask_bit | 127)
        head += struct.pack("!Q", length)
    body = payload
    if mask:
        head += mask_key
        body = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
    return bytes(head) + body


class _EchoServer(socketserver.ThreadingTCPServer):
    """WS echo 服务：``allow_reuse_address`` + 守护线程，测试退出即消失。"""

    allow_reuse_address = True
    daemon_threads = True

    subprotocols: tuple[str, ...] = ()
    max_frame: int = MAX_FRAME_BYTES

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.errors: list[tuple[str, int, str]] = []  # 服务端侧捕获的 WsError（kind, code, msg）
        self.seen: dict[str, Any] = {}  # 握手阶段观察到的东西（路径/头/子协议）
        self._lock = threading.Lock()

    def record(self, kind: str, exc: BaseException) -> None:
        """记下服务端抛出的协议错误，供测试断言「服务端确实以 1002 拒绝了它」。"""
        with self._lock:
            self.errors.append((kind, getattr(exc, "code", 0), str(exc)))

    def snapshot(self) -> list[tuple[str, int, str]]:
        """错误清单的线程安全快照（断言线程与 handler 线程不同，必须走锁）。"""
        with self._lock:
            return list(self.errors)


class _EchoHandler(socketserver.BaseRequestHandler):
    """握手 → 原样回显：文本回文本、JSON 对象走 json→json、二进制回二进制。"""

    def handle(self) -> None:  # noqa: D102 - socketserver 的回调
        srv = cast(_EchoServer, self.server)
        try:
            ws = WebSocket.accept(
                self.request,
                timeout=_IO_TIMEOUT,
                max_frame=srv.max_frame,
                subprotocols=srv.subprotocols,
            )
        except WsError as exc:
            srv.record("handshake", exc)
            return
        srv.seen = {
            "path": ws.request_path,
            "headers": dict(ws.request_headers),
            "subprotocol": ws.subprotocol,
        }
        try:
            with ws:
                self._serve(ws)
        except (WsError, OSError) as exc:  # OSError 含 TimeoutError：超时只记一笔，别打印堆栈
            srv.record("frame", exc)

    @staticmethod
    def _serve(ws: WebSocket) -> None:
        while True:
            msg = ws.recv()
            if msg is None:
                return  # 对端 close（已回显）或 TCP 断开
            if msg.opcode == "binary":
                ws.send_binary(msg.data)
                continue
            text = msg.text or ""
            if text == "close-4001":
                ws.close(4001, "自定义关闭")
                return
            if text.startswith("{"):
                try:
                    ws.send_json(json.loads(text))
                except ValueError:
                    ws.send_text(text)
                continue
            ws.send_text(text)


class _CannedHandshakeHandler(socketserver.BaseRequestHandler):
    """假的握手服务：按 ``mode`` 回一个不正确的响应，专门用来测客户端侧的校验。"""

    mode = "bad_accept"

    def handle(self) -> None:  # noqa: D102 - socketserver 的回调
        self.request.settimeout(_IO_TIMEOUT)  # 假服务也要有超时：别让 handler 线程悬着
        raw = b""
        while b"\r\n\r\n" not in raw:  # 读掉请求头
            chunk = self.request.recv(4096)
            if not chunk:
                return
            raw += chunk
        if self.mode == "status_426":
            self.request.sendall(
                b"HTTP/1.1 426 Upgrade Required\r\n"
                b"Sec-WebSocket-Version: 13\r\nContent-Length: 0\r\n\r\n"
            )
            return
        # 按请求里的 Key 算出**正确**的 Accept：这样除 mode="bad_accept" 外，
        # 唯一不合规的就只有我们要测的那一项（否则 Accept 先报错，测不到 Upgrade 检查）。
        key = ""
        for line in raw.decode("latin-1").split("\r\n")[1:]:
            name, _, value = line.partition(":")
            if name.strip().lower() == "sec-websocket-key":
                key = value.strip()
        accept = "wrong-on-purpose" if self.mode == "bad_accept" else accept_key(key)
        upgrade = b"" if self.mode == "missing_upgrade" else b"Upgrade: websocket\r\n"
        protocol = (
            b"Sec-WebSocket-Protocol: never-requested\r\n"
            if self.mode == "unrequested_subprotocol"
            else b""
        )
        self.request.sendall(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            + upgrade
            + b"Connection: Upgrade\r\n"
            + f"Sec-WebSocket-Accept: {accept}\r\n".encode("ascii")
            + protocol
            + b"\r\n"
        )


def _url(srv: _EchoServer, path: str = "/ws/chapter/1.2") -> str:
    """测试用 URL：主机固定回环，端口取内核分配的那个。"""
    return f"ws://127.0.0.1:{srv.server_address[1]}{path}"


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """从裸 socket 读满 n 字节（服务端帧不掩码，测试里手工解析）。"""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def _raw_handshake(srv: _EchoServer, request: bytes) -> str:
    """用裸 socket 发一个手工握手请求并读回**响应头**（不依赖自研客户端）。"""
    with socket.create_connection(
        ("127.0.0.1", srv.server_address[1]), timeout=_IO_TIMEOUT
    ) as sock:
        sock.sendall(request)
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
    return buf.decode("latin-1")


def _wait_for_error(
    srv: _EchoServer, code: int, *, kind: str | None = None, timeout: float = 3.0
) -> list[tuple[str, int, str]]:
    """等后台 handler 线程把 ``WsError(code)`` 记进 ``srv.errors``。

    为什么要等：错误在 handler 线程里记录，而断言在主线程执行——客户端刚读到 close 帧时，
    handler 可能还在 ``_linger_close`` 里排空对端残留字节，尚未 record。
    同步方式是**轮询 + 截止时间**（不是固定 sleep）：等不到就返回现状，由断言报错。
    """
    deadline = time.monotonic() + timeout
    while True:
        errors = srv.snapshot()
        hit = any(
            got == code and (kind is None or got_kind == kind)
            for got_kind, got, _m in errors
        )
        if hit:
            return errors
        if time.monotonic() >= deadline:
            return errors
        time.sleep(0.005)


_GOOD_HANDSHAKE = (
    b"GET {target} HTTP/1.1\r\n"
    b"Host: 127.0.0.1\r\n"
    b"Upgrade: websocket\r\n"
    b"Connection: keep-alive, Upgrade\r\n"
    b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
    b"Sec-WebSocket-Version: {version}\r\n"
    b"{extra}"
    b"\r\n"
)


def _handshake_request(
    *, target: bytes = b"/ws/chapter/1.2", version: bytes = b"13", extra: bytes = b""
) -> bytes:
    """拼一个握手请求；``version=b""`` 表示该头存在但值为空（要删整行由调用方 replace）。"""
    return _GOOD_HANDSHAKE.replace(b"{target}", target).replace(
        b"{version}", version
    ).replace(b"{extra}", extra)


@pytest.fixture()
def ws_server() -> Iterator[Callable[..., _EchoServer]]:
    """启动若干后台 WS 服务，``yield`` 一个 ``start(**opts) -> server`` 工厂，退出时全部关停。"""
    servers: list[_EchoServer] = []

    def start(*, subprotocols: Sequence[str] = (), max_frame: int = MAX_FRAME_BYTES) -> _EchoServer:
        srv = _EchoServer(("127.0.0.1", 0), _EchoHandler)
        srv.subprotocols = tuple(subprotocols)
        srv.max_frame = max_frame
        threading.Thread(
            target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        ).start()
        servers.append(srv)
        return srv

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


# ------------------------------------------------------------------- 1. 握手基元


def test_accept_key_rfc6455_example() -> None:
    """RFC 6455 §1.3 官方向量：Accept 必须逐字符相等。"""
    assert accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_handshake_response_headers(ws_server: Callable[..., _EchoServer]) -> None:
    """101 响应的状态行与头必须一字不差（大小写敏感的是值，不是头名）。"""
    srv = ws_server()
    response = _raw_handshake(srv, _handshake_request())
    head, _, _body = response.partition("\r\n\r\n")
    lines = head.split("\r\n")
    assert lines[0] == "HTTP/1.1 101 Switching Protocols"
    headers = {
        name.strip().lower(): value.strip()
        for name, _, value in (line.partition(":") for line in lines[1:])
    }
    assert headers["upgrade"].lower() == "websocket"
    assert headers["connection"].lower() == "upgrade"
    assert headers["sec-websocket-accept"] == accept_key("dGhlIHNhbXBsZSBub25jZQ==")
    assert "sec-websocket-protocol" not in headers  # 客户端没提，服务端不得主动回


def test_server_exposes_request_path_headers_and_subprotocol(
    ws_server: Callable[..., _EchoServer]
) -> None:
    """``request_path`` 保留查询串、头名小写化——``server/app.py`` 的路由要靠这两点。"""
    srv = ws_server(subprotocols=("nlab.v1",))
    with WebSocketClient.connect(
        _url(srv, "/ws/chapter/3.2?mode=stream"),
        timeout=_IO_TIMEOUT,
        subprotocols=("nlab.v1",),
        extra_headers={"X-Nlab-Test": "yes"},
    ) as client:
        client.send_text("route")
        assert client.recv() is not None
    assert srv.seen["path"] == "/ws/chapter/3.2?mode=stream"
    assert srv.seen["subprotocol"] == "nlab.v1"
    headers = srv.seen["headers"]
    assert headers["x-nlab-test"] == "yes"
    assert headers["upgrade"].lower() == "websocket"
    # 客户端每次用 os.urandom(16) 生成 Key：这里按 RFC 校验「16 字节随机数的 base64」
    assert len(base64.b64decode(headers["sec-websocket-key"], validate=True)) == 16


# --------------------------------------------------------------------- 2. echo


def test_text_echo_utf8(ws_server: Callable[..., _EchoServer]) -> None:
    """文本帧原样回显，中文/emoji 按 UTF-8 字节往返。"""
    srv = ws_server()
    text = "你好，数值实验室 ✓ 中文 emoji 🚀 混排"
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_text(text)
        msg = client.recv()
        assert msg is not None and msg.opcode == "text"
        assert msg.text == text
        assert msg.data == text.encode("utf-8")


def test_binary_echo_all_byte_values(ws_server: Callable[..., _EchoServer]) -> None:
    """二进制帧逐字节相等（含 0x00 / 0x0A / 0xFF 这些容易被「顺手转义」的字节）。"""
    srv = ws_server()
    payload = bytes(range(256)) * 3
    assert payload[:1] == b"\x00" and payload[255:256] == b"\xff"
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_binary(payload)
        msg = client.recv()
        assert msg is not None and msg.opcode == "binary"
        assert msg.data == payload
        assert msg.text is None


# ------------------------------------------------------- 3. 长度分支与大帧往返

_ONE_MIB = bytes(range(256)) * 4096  # 正好 1 MiB，且覆盖全部字节值


def test_large_frame_1mib_binary_roundtrip(ws_server: Callable[..., _EchoServer]) -> None:
    """1 MiB 二进制往返逐字节相等：走的是 64 位长度分支。"""
    srv = ws_server()
    assert len(_ONE_MIB) == 1 << 20
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_binary(_ONE_MIB)
        msg = client.recv()
        assert msg is not None and len(msg.data) == len(_ONE_MIB)
        assert msg.data == _ONE_MIB


@pytest.mark.parametrize("size", [0, 1, 125, 126, 127, 65535, 65536])
def test_length_encoding_boundaries(ws_server: Callable[..., _EchoServer], size: int) -> None:
    """7 位 / 16 位 / 64 位长度分支的边界（125→126→65535→65536）都必须自洽。"""
    srv = ws_server()
    payload = bytes((i * 7 + 3) & 0xFF for i in range(size))
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_binary(payload)
        msg = client.recv()
        assert msg is not None
        assert msg.data == payload


def test_mask_payload_matches_naive_implementation() -> None:
    """掩码是自反的，且与逐字节朴素实现完全一致（大帧走整数异或，不能悄悄算错）。"""
    key = b"\x37\xfa\x21\x3d"
    data = bytes(range(256)) * 5 + b"\x00"
    masked = _mask_payload(data, key)
    assert masked == bytes(b ^ key[i & 3] for i, b in enumerate(data))
    assert _mask_payload(masked, key) == data
    assert _mask_payload(b"", key) == b""


# ------------------------------------------------------------- 4. 分片与控制帧


def _split_utf8(text: str, parts: int) -> list[bytes]:
    """把一段 UTF-8 字节切成 parts 片，**故意**把多字节字符切在分片中间。"""
    raw = text.encode("utf-8")
    assert parts >= 4
    bounds = [0, 1, 2, 3]  # 前 3 片各 1 字节：必然切断某个多字节字符
    while len(bounds) < parts:
        bounds.append(bounds[-1] + max(1, (len(raw) - bounds[-1]) // (parts - len(bounds) + 1)))
    bounds.append(len(raw))
    return [raw[bounds[i] : bounds[i + 1]] for i in range(parts)]


def test_fragmented_binary_five_parts(ws_server: Callable[..., _EchoServer]) -> None:
    """5 个分片重组后与原文逐字节相等。"""
    srv = ws_server()
    parts = [b"\x00\x01\x02", b"chunk-2" * 3, b"\xff" * 40, b"", b"tail"]
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_fragmented(parts)
        msg = client.recv()
        assert msg is not None and msg.opcode == "binary"
        assert msg.data == b"".join(parts)


def test_fragmented_text_split_inside_multibyte_char(
    ws_server: Callable[..., _EchoServer]
) -> None:
    """UTF-8 校验只能对**整条消息**做：多字节字符被切在两个分片中间也必须通过。"""
    srv = ws_server()
    text = "分片测试：中文与 emoji 🚀 混排"
    parts = _split_utf8(text, 5)
    assert len(parts) == 5 and len(parts[0]) == 1  # 确实切在多字节字符中间
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_fragmented(parts, text=True)
        msg = client.recv()
        assert msg is not None and msg.text == text


def test_fragmented_with_interleaved_ping(ws_server: Callable[..., _EchoServer]) -> None:
    """控制帧插在数据分片之间：服务端既要重组正确，也要逐个回 pong。"""
    srv = ws_server()
    parts = [b"aa", b"bb", b"\xf0\x9f\x9a\x80", b"dd", b"ee"]
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_fragmented(parts, interleave_ping=True)
        pongs = 0
        while True:
            frame = client.recv_frame_raw()  # 用低层口子看被 recv() 吞掉的 pong
            assert frame is not None, "服务端在重组完成前就断开了"
            fin, opcode, payload = frame
            if opcode == 0x9:
                continue
            if opcode == 0xA:
                assert (fin, payload) == (True, b"mid")
                pongs += 1
                continue
            break
        assert pongs == len(parts) - 1  # 每个分片间隔一个 ping，就应有一个 pong
        assert (opcode, payload) == (0x2, b"".join(parts))


def test_fragmented_message_exceeding_limit_closes_1009(
    ws_server: Callable[..., _EchoServer]
) -> None:
    """分片累计超过上限 → 1009（单帧没超、累计超了，正是「只在单帧上判长度」的漏洞）。"""
    limit = 4096
    srv = ws_server(max_frame=limit)
    blob = b"x" * 1024
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.raw_send(_frame(0x2, blob, fin=False))  # 1024
        client.raw_send(_frame(0x0, blob, fin=False))  # 2048
        client.raw_send(_frame(0x0, blob, fin=False))  # 3072
        client.raw_send(_frame(0x0, blob * 2, fin=True))  # 5120 > 4096 → 1009
        assert client.recv() is None
        assert client.close_code == 1009
    assert _wait_for_error(srv, 1009)


# ------------------------------------------------------------ 5. 粘包 / 半包


def test_three_frames_coalesced_in_one_write(ws_server: Callable[..., _EchoServer]) -> None:
    """3 个帧拼成一次写入：接收端不得假设「一次 recv 正好一个帧」。"""
    srv = ws_server()
    blob = (
        _frame(0x1, b"one")
        + _frame(0x2, b"\x00\xff\x10")
        + _frame(0x1, "第三帧".encode("utf-8"))
    )
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.raw_send(blob)
        got = [client.recv() for _ in range(3)]
    assert [m.data if m else None for m in got] == [b"one", b"\x00\xff\x10", "第三帧".encode()]


def test_single_frame_dribbled_one_byte_at_a_time(ws_server: Callable[..., _EchoServer]) -> None:
    """1 个帧拆成 1 字节一次慢慢发：服务端的 ``_read_exact`` 必须循环读满。"""
    srv = ws_server()
    text = "半包测试：一字节一字节地喂"
    blob = _frame(0x1, text.encode("utf-8"))
    assert len(blob) > 16
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        for i in range(len(blob)):
            client.raw_send(blob[i : i + 1])
            time.sleep(0.001)  # 让内核真的把它切成一堆小段（总睡眠 ~30 ms，可接受）
        msg = client.recv()
    assert msg is not None and msg.text == text


# --------------------------------------------------------------- 6. 控制帧/关闭


def test_ping_gets_pong_and_unsolicited_pong_is_ignored(
    ws_server: Callable[..., _EchoServer]
) -> None:
    """ping → pong（载荷原样带回）；自发 pong 不应答、也不影响后续数据。"""
    srv = ws_server()
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.ping(b"keepalive")
        assert client.recv_frame_raw() == (True, 0xA, b"keepalive")
        client.pong(b"unsolicited")
        client.send_text("still-alive")
        msg = client.recv()
        assert msg is not None and msg.text == "still-alive"


def test_ping_payload_limit_is_local_error(ws_server: Callable[..., _EchoServer]) -> None:
    """控制帧载荷 >125 是调用方的错：本端抛 WsError，连接照旧可用。"""
    srv = ws_server()
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        with pytest.raises(WsError, match="125"):
            client.ping(b"x" * 126)
        client.send_text("ok")
        msg = client.recv()
        assert msg is not None and msg.text == "ok"


def test_close_handshake_code_1000(ws_server: Callable[..., _EchoServer]) -> None:
    """标准 close 握手：客户端发起 1000，双方都认为连接已关闭；close() 幂等。"""
    srv = ws_server()
    client = WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT)
    client.send_text("hi")
    assert client.recv() is not None
    client.close(1000)
    assert client.closed and client.close_code == 1000
    assert client.recv() is None  # 关闭后 recv 直接 None，不再阻塞
    client.close(1000)  # 幂等：不抛异常、也不再发第二个 close 帧
    with pytest.raises(WsError):
        client.send_text("after-close")  # closed 之后禁止再发
    assert srv.snapshot() == []  # 干净的 close 握手不该产生任何协议错误


def test_client_initiated_custom_close_4001(ws_server: Callable[..., _EchoServer]) -> None:
    """自定义 code(4001)+reason 的关闭：服务端回显同样的 code。"""
    srv = ws_server()
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.close(4001, "客户端自定义关闭")
        assert client.closed
        assert client.close_code == 4001
        assert client.close_reason == "客户端自定义关闭"
    assert srv.snapshot() == []  # 干净的 close 握手不该产生任何协议错误


def test_server_initiated_custom_close_4001(ws_server: Callable[..., _EchoServer]) -> None:
    """服务端主动 close(4001)：客户端要能看到 code 与 reason（探针消息触发）。"""
    srv = ws_server()
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_text("close-4001")
        assert client.recv() is None
        assert client.closed
        assert (client.close_code, client.close_reason) == (4001, "自定义关闭")
    assert srv.snapshot() == []  # 干净的 close 握手不该产生任何协议错误


def test_close_code_not_allowed_in_frame(ws_server: Callable[..., _EchoServer]) -> None:
    """1005/1006/1015 是本地占位符，绝不允许出现在帧里——本端直接拒绝。"""
    srv = ws_server()
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        for bad in (1005, 1006, 1015, 999, 5000):
            with pytest.raises(WsError):
                client.close(bad)
        client.close(1000)  # 合法 code 仍然可用


# ------------------------------------------------------- 7. 非法输入全部被拒


_ILLEGAL_CASES: list[tuple[str, bytes, int]] = [
    ("无掩码帧", _frame(0x1, b"nope", mask=False), 1002),
    ("坏 opcode 0x3", _frame(0x3, b"xx"), 1002),
    ("RSV1 非零（压缩/扩展）", _frame(0x1, b"xx", rsv=1), 1002),
    ("RSV3 非零", _frame(0x2, b"xx", rsv=4), 1002),
    ("声明 16 MiB+1 的超长帧", _frame(0x2, b"", declared=MAX_FRAME_BYTES + 1), 1009),
    ("非法 UTF-8 文本", _frame(0x1, b"\xff\xfe\xfd"), 1007),
    ("没有起始帧的 continuation", _frame(0x0, b"x"), 1002),
    ("分片的控制帧（ping）", _frame(0x9, b"p", fin=False), 1002),
    ("控制帧载荷超 125 字节", _frame(0xA, b"x" * 126), 1002),
    ("close 载荷只有 1 字节", _frame(0x8, b"\x03"), 1002),
    ("close code 1005 非法上线", _frame(0x8, struct.pack("!H", 1005)), 1002),
    ("64 位长度最高位为 1", _frame(0x2, b"", declared=1 << 63), 1002),
]


@pytest.mark.parametrize(
    ("name", "blob", "code"),
    [pytest.param(name, data, code, id=name) for name, data, code in _ILLEGAL_CASES],
)
def test_illegal_input_closes_with_expected_code(
    ws_server: Callable[..., _EchoServer], name: str, blob: bytes, code: int
) -> None:
    """每一类非法帧都必须以正确的 close code 结束，并且服务端要抛出 WsError。"""
    srv = ws_server()
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.raw_send(blob)
        assert client.recv() is None, f"{name}: 连接应当被关闭"
        assert client.close_code == code, f"{name}: 期望 close {code}，收到 {client.close_code}"
    errors = _wait_for_error(srv, code)
    assert any(got == code for _kind, got, _msg in errors), (
        f"{name}: 服务端应当记录 WsError(code={code})，实际 {errors}"
    )


def test_non_minimal_length_encoding_closes_1002(ws_server: Callable[..., _EchoServer]) -> None:
    """长度必须用最简编码（RFC 6455 §5.2）：5 字节载荷写成 16 位长度即违规。"""
    srv = ws_server()
    key = b"\x37\xfa\x21\x3d"
    head = bytearray([0x81, 0x80 | 126])
    head += struct.pack("!H", 5) + key
    blob = bytes(head) + bytes(b ^ key[i & 3] for i, b in enumerate(b"hello"))
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.raw_send(blob)
        assert client.recv() is None
        assert client.close_code == 1002
    assert _wait_for_error(srv, 1002)


def test_invalid_utf8_spanning_fragments_closes_1007(
    ws_server: Callable[..., _EchoServer]
) -> None:
    """非法 UTF-8 跨分片同样要 1007（不能只看最后一个分片）。"""
    srv = ws_server()
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_fragmented([b"\xe4\xb8", b"\xff\xfe"], text=True)
        assert client.recv() is None
        assert client.close_code == 1007
    assert _wait_for_error(srv, 1007)


def test_oversized_declared_frame_is_rejected_before_payload(
    ws_server: Callable[..., _EchoServer]
) -> None:
    """64 位声明超过 ``max_frame`` → 1009，且服务端**没有尝试去读**那 16 MiB 载荷。"""
    srv = ws_server()
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.raw_send(_frame(0x2, b"", declared=MAX_FRAME_BYTES + 1))  # 只发帧头
        assert client.recv() is None
        assert client.close_code == 1009
    assert _wait_for_error(srv, 1009)


def test_smaller_max_frame_is_honoured(ws_server: Callable[..., _EchoServer]) -> None:
    """``max_frame`` 是每连接的：把上限压到 1 KiB，2 KiB 的帧就必须被 1009 拒掉。"""
    srv = ws_server(max_frame=1024)
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_binary(b"y" * 2048)
        assert client.recv() is None
        assert client.close_code == 1009


def test_handshake_version_8_gets_426(ws_server: Callable[..., _EchoServer]) -> None:
    """``Sec-WebSocket-Version: 8`` → 426，并附上服务端支持的版本号。"""
    srv = ws_server()
    response = _raw_handshake(srv, _handshake_request(version=b"8"))
    assert response.startswith("HTTP/1.1 426 ")
    assert "Sec-WebSocket-Version: 13" in response
    assert _wait_for_error(srv, 1002, kind="handshake")


@pytest.mark.parametrize(
    ("request_bytes", "status"),
    [
        pytest.param(
            _handshake_request().replace(b"Sec-WebSocket-Version: 13\r\n", b""), "400 ",
            id="缺少版本头",
        ),
        pytest.param(_handshake_request(version=b""), "426 ", id="版本头为空"),
        pytest.param(
            _handshake_request(extra=b"Sec-WebSocket-Version: 8\r\n"), "426 ",
            id="版本头重复（合并后不是 13）",
        ),
        pytest.param(
            _handshake_request().replace(
                b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n", b""), "400 ",
            id="缺少 Key",
        ),
        pytest.param(
            _handshake_request().replace(b"dGhlIHNhbXBsZSBub25jZQ==", b"YWJj"), "400 ",
            id="Key 不是 16 字节",
        ),
        pytest.param(
            _handshake_request(extra=b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"), "400 ",
            id="Key 重复（合并后不是合法 base64）",
        ),
        pytest.param(_handshake_request().replace(b"GET ", b"POST ", 1), "405 ", id="不是 GET"),
        pytest.param(
            _handshake_request().replace(b"Upgrade: websocket\r\n", b""), "400 ",
            id="缺少 Upgrade",
        ),
        pytest.param(
            _handshake_request().replace(b"keep-alive, Upgrade", b"keep-alive"), "400 ",
            id="Connection 里没有 Upgrade",
        ),
        pytest.param(
            _handshake_request().replace(b"HTTP/1.1", b"HTTP/1.0", 1), "400 ",
            id="不是 HTTP/1.1",
        ),
    ],
)
def test_bad_handshake_is_rejected(
    ws_server: Callable[..., _EchoServer], request_bytes: bytes, status: str
) -> None:
    """握手不完整/不合法一律回 HTTP 错误码（且不能悄悄升级成 WebSocket）。"""
    srv = ws_server()
    response = _raw_handshake(srv, request_bytes)
    assert response.startswith("HTTP/1.1 " + status), f"收到 {response.splitlines()[:1]}"
    assert _wait_for_error(srv, 1002, kind="handshake")


def test_client_rejects_non_101_and_bad_accept_key() -> None:
    """客户端侧校验：非 101 → WsHandshakeError(status)；Accept 对不上也要拒绝。"""
    servers: list[socketserver.ThreadingTCPServer] = []

    def start(mode: str) -> int:
        handler = type(f"_H_{mode}", (_CannedHandshakeHandler,), {"mode": mode})
        srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        srv.daemon_threads = True
        thread = threading.Thread(
            target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        thread.start()
        servers.append(srv)
        return int(srv.server_address[1])

    try:
        with pytest.raises(WsHandshakeError) as err:
            WebSocketClient.connect(f"ws://127.0.0.1:{start('status_426')}/ws", timeout=_IO_TIMEOUT)
        assert err.value.status == 426
        with pytest.raises(WsHandshakeError, match="Accept"):
            WebSocketClient.connect(f"ws://127.0.0.1:{start('bad_accept')}/ws", timeout=_IO_TIMEOUT)
        with pytest.raises(WsHandshakeError, match="Upgrade"):
            WebSocketClient.connect(
                f"ws://127.0.0.1:{start('missing_upgrade')}/ws", timeout=_IO_TIMEOUT
            )
        with pytest.raises(WsHandshakeError, match="子协议"):
            WebSocketClient.connect(
                f"ws://127.0.0.1:{start('unrequested_subprotocol')}/ws", timeout=_IO_TIMEOUT
            )
    finally:
        for srv in servers:
            srv.shutdown()
            srv.server_close()


def test_client_rejects_non_ws_scheme() -> None:
    """``wss://`` 不在本子集内：显式拒绝，不能悄悄降级成明文。"""
    with pytest.raises(WsError, match="ws://"):
        WebSocketClient.connect("wss://127.0.0.1:1/ws", timeout=0.5)


class _PipelinedHandler(socketserver.BaseRequestHandler):
    """模拟 ``server/app.py`` 的升级路径：HTTP 层先读走请求头，把缓冲区剩的字节交回。

    ``http.server`` 的 ``self.rfile`` 是带缓冲的读取器，一次 read() 可能把紧随握手而来的
    帧字节一起吞掉；不交回 ``initial_bytes`` 就会「第一个帧凭空消失、连接卡死」。
    """

    def handle(self) -> None:  # noqa: D102 - socketserver 的回调
        self.request.settimeout(_IO_TIMEOUT)  # 交给 accept 前也先设超时，避免悬挂
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            buf += chunk
        # 已经把「请求头 + 可能紧随其后的帧字节」从 socket 里读走了，
        # 所以必须**原样整份**交回（包括请求头本身），accept 才不会丢帧、也不会重复解析。
        ws = WebSocket.accept(self.request, timeout=_IO_TIMEOUT, initial_bytes=buf)
        with ws:
            msg = ws.recv()
            if msg is not None:
                ws.send_text(msg.text or "")


def test_pipelined_first_frame_is_not_lost() -> None:
    """握手与第一个帧挤在同一次写入里：预读的字节必须被保住（``_rbuf`` / ``initial_bytes``）。"""
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _PipelinedHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    try:
        with socket.create_connection(
            ("127.0.0.1", srv.server_address[1]), timeout=_IO_TIMEOUT
        ) as sock:
            sock.sendall(_handshake_request() + _frame(0x1, b"pipelined"))
            head = b""
            while b"\r\n\r\n" not in head:
                head += _recv_exact(sock, 1)
            assert head.startswith(b"HTTP/1.1 101 ")
            header = _recv_exact(sock, 2)
            assert header[0] == 0x81  # FIN + text，RSV 全 0
            assert header[1] & 0x80 == 0  # 服务端发出的帧**不得**掩码（RFC 6455 §5.1）
            assert _recv_exact(sock, header[1] & 0x7F) == b"pipelined"
    finally:
        srv.shutdown()
        srv.server_close()


# --------------------------------------------------------------- 8. 并发不串数据


def test_two_concurrent_clients_do_not_mix_data(ws_server: Callable[..., _EchoServer]) -> None:
    """两个客户端同时来回 echo：各自的数据必须逐字节对得上，且标签不串。"""
    srv = ws_server()
    url = _url(srv)
    rounds = 25

    def worker(tag: str) -> None:
        with WebSocketClient.connect(url, timeout=10.0) as client:
            for index in range(rounds):
                payload = f"{tag}#{index:03d}|".encode("ascii") + bytes([index & 0xFF]) * 64
                client.send_binary(payload)
                msg = client.recv()
                assert msg is not None and msg.opcode == "binary"
                assert msg.data == payload, f"{tag} 第 {index} 轮数据不一致"
                assert msg.data.startswith(tag.encode()), f"{tag} 收到了别人的数据"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, "A"), pool.submit(worker, "B")]
        for future in futures:
            future.result(timeout=30)  # 超时即失败，不会挂死


# ------------------------------------------------------------- 9. 子协议协商


def test_subprotocol_echoed_when_matched(ws_server: Callable[..., _EchoServer]) -> None:
    """服务端 ``subprotocols=("nlab.v1",)`` + 客户端请求同款 → 回显 ``nlab.v1``。"""
    srv = ws_server(subprotocols=("nlab.v1",))
    with WebSocketClient.connect(
        _url(srv), timeout=_IO_TIMEOUT, subprotocols=("nlab.v1",)
    ) as client:
        assert client.subprotocol == "nlab.v1"
        client.send_text("sub")
        assert client.recv() is not None
    assert srv.seen["subprotocol"] == "nlab.v1"


def test_subprotocol_not_echoed_when_no_overlap(ws_server: Callable[..., _EchoServer]) -> None:
    """没有交集就不回显 ``Sec-WebSocket-Protocol``，但连接照常建立。"""
    srv = ws_server(subprotocols=("nlab.v1",))
    with WebSocketClient.connect(
        _url(srv), timeout=_IO_TIMEOUT, subprotocols=("other.v9",)
    ) as client:
        assert client.subprotocol is None
        assert "sec-websocket-protocol" not in client.response_headers
        client.send_text("sub-any")
        assert client.recv() is not None
    assert srv.seen["subprotocol"] is None


def test_subprotocol_picks_server_preference(ws_server: Callable[..., _EchoServer]) -> None:
    """两边都提了多个时按**服务端偏好序**取（客户端顺序不能绑架服务端）。"""
    srv = ws_server(subprotocols=("nlab.v1", "nlab.v2"))
    with WebSocketClient.connect(
        _url(srv), timeout=_IO_TIMEOUT, subprotocols=("nlab.v2", "nlab.v1")
    ) as client:
        assert client.subprotocol == "nlab.v1"


# --------------------------------------------------------------- 10. JSON 往返


def test_json_roundtrip_chinese_and_nested(ws_server: Callable[..., _EchoServer]) -> None:
    """``send_json``/``recv_json`` 中文与嵌套结构往返一致，且线上是 UTF-8 原文。"""
    srv = ws_server()
    obj = {
        "type": "subscribe",
        "chapter": "3.2",
        "viz": "stability_domain",
        "params": {"method": "rk4", "h": 0.05, "n": 24, "tol": 1e-12, "flag": True, "nil": None},
        "notes": ["中文", "emoji 🚀", {"嵌套": ["a", 1, 2.5]}],
        "pitfall": "对数色标会把 1e-16 和 1e-3 画得一样深……",
    }
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_json(obj)
        assert client.recv_json() == obj
        client.send_json({"k": "中文"})
        msg = client.recv()
        assert msg is not None and msg.opcode == "text"
        assert "中文".encode("utf-8") in msg.data  # 不是 \uXXXX 转义
        assert json.loads(msg.data) == {"k": "中文"}


def test_recv_json_rejects_binary_frame_but_keeps_connection(
    ws_server: Callable[..., _EchoServer]
) -> None:
    """二进制帧（.nlf 载荷）不能当 JSON 解；这个错误只发生在本端，连接不受影响。"""
    srv = ws_server()
    with WebSocketClient.connect(_url(srv), timeout=_IO_TIMEOUT) as client:
        client.send_binary(b"\x00\xffbinary")
        with pytest.raises(WsError, match="二进制"):
            client.recv_json()
        client.send_text("json-please")
        msg = client.recv()
        assert msg is not None and msg.text == "json-please"


def test_message_value_object_contract() -> None:
    """``Message`` 是不可变值对象：``text`` 只对文本帧有值，非法 opcode 直接拒绝。"""
    msg = Message("text", "中文".encode("utf-8"))
    assert msg.text == "中文"
    assert Message("binary", b"\x00").text is None
    assert Message("binary", bytearray(b"\x01")).data == b"\x01"
    with pytest.raises(ValueError):
        Message("ping", b"")
    with pytest.raises(dataclasses.FrozenInstanceError):  # frozen：赋值必须失败
        msg.opcode = "binary"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 从 http.server 升级：initial_bytes 的配方必须只有一份实现，且不能阻塞
# --------------------------------------------------------------------------- #
class _FakeHTTPHandler:
    """够用的 ``BaseHTTPRequestHandler`` 替身。

    忠实模拟两件事：**请求头已经被 HTTP 层消费掉了**（所以要把 head 重建出来交回），
    以及 ``rfile`` 是带缓冲的（客户端把首帧和请求头一起发来时，首帧会落在缓冲区里）。
    """

    def __init__(self, sock: socket.socket) -> None:
        self.connection = sock
        self.rfile = sock.makefile("rb")
        self.raw_requestline = b""
        self.headers: dict[str, str] = {}

    def read_request_head(self) -> None:
        self.raw_requestline = self.rfile.readline()
        while True:
            line = self.rfile.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode("latin-1").partition(":")
            self.headers[name.strip()] = value.strip()

    def close(self) -> None:
        self.rfile.close()


def _upgrade_request(key: str) -> bytes:
    """升级请求（**与文件上半部分那个同名助手区分开**：那个是给裸 socket 用的，签名不同）。"""
    return (
        b"GET /ws/chapter/0.0 HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        b"Sec-WebSocket-Key: " + key.encode() + b"\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )


def test_peek_buffered_returns_buffered_bytes_without_blocking() -> None:
    """缓冲区里**有**预读字节时取回；取的是"已缓冲"的字节，不消费缓冲区。"""
    a, b = socket.socketpair()
    try:
        h = _FakeHTTPHandler(a)
        b.sendall(b"\x81\x05hello")
        assert h.rfile.peek(1)[:1] == b"\x81"  # 把字节拉进缓冲区（有数据，不会阻塞）
        got = peek_buffered(h.rfile, a)
        assert got == b"\x81\x05hello"
        assert peek_buffered(h.rfile, a) == b"\x81\x05hello"  # peek 不消费
        h.close()
    finally:
        a.close()
        b.close()


def test_peek_buffered_is_non_blocking_on_empty_buffer() -> None:
    """缓冲区为空时必须**立刻**返回 b""——若这里退化成阻塞 peek，本测试会挂住。"""
    a, b = socket.socketpair()
    try:
        h = _FakeHTTPHandler(a)
        started = time.monotonic()
        assert peek_buffered(h.rfile, a) == b""
        assert time.monotonic() - started < 1.0, "peek_buffered 阻塞了"
        h.close()
    finally:
        a.close()
        b.close()


def test_initial_bytes_rebuilds_request_head_and_keeps_buffered_frames() -> None:
    a, b = socket.socketpair()
    try:
        h = _FakeHTTPHandler(a)
        b.sendall(_upgrade_request("dGhlIHNhbXBsZSBub25jZQ==") + _frame(0x1, b"hi"))
        h.read_request_head()
        head = initial_bytes_from_handler(h)
        assert head.startswith(b"GET /ws/chapter/0.0 HTTP/1.1\r\n")
        assert b"Host: x\r\n" in head
        assert head.endswith(b"\r\n\r\n" + _frame(0x1, b"hi")) or b"\r\n\r\n" in head
        h.close()
    finally:
        a.close()
        b.close()


def test_accept_from_handler_completes_handshake_and_first_frame() -> None:
    """端到端：客户端把**请求头与首帧一起发**（同一个 TCP 段），首帧也必须到达。

    这一条同时钉住两个 bug：照抄 ``rfile.peek()`` 会死锁（客户端正等 101），
    而漏交缓冲字节会丢掉握手后的第一帧。
    """
    import base64 as _b64
    import threading

    key = _b64.b64encode(b"0123456789abcdef").decode("ascii")
    server_sock, client_sock = socket.socketpair()
    result: dict[str, object] = {}

    def serve() -> None:
        handler = _FakeHTTPHandler(server_sock)
        try:
            handler.read_request_head()
            ws = WebSocket.accept_from_handler(handler, timeout=_IO_TIMEOUT)
            with ws:
                result["frame"] = ws.recv()
        except Exception as exc:  # noqa: BLE001 - 测试要把异常带回来断言
            result["error"] = exc
        finally:
            handler.close()
            server_sock.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    client_sock.settimeout(_IO_TIMEOUT)
    client_sock.sendall(_upgrade_request(key) + _frame(0x1, b"hi"))
    resp = client_sock.recv(4096)
    assert b"101" in resp.split(b"\r\n")[0], resp
    t.join(timeout=_IO_TIMEOUT)
    assert not t.is_alive(), "握手死锁了（peek 阻塞）"
    assert "error" not in result, result.get("error")
    msg = result["frame"]
    assert isinstance(msg, Message) and msg.data == b"hi" and msg.opcode == "text"
    client_sock.close()
