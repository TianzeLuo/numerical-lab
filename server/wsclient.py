"""自研 WebSocket **客户端子集**（零第三方依赖），供 ``tests/`` 与 ``tools/`` 使用。

为什么不用 ``websockets``/``aiohttp``：见 ``docs/POLICY.md`` §1 的依赖红线。此外测试需要
**故意违规**的输入（无掩码帧、坏 opcode、RSV 非零、半包、声明超长的帧），
成熟库恰恰不会给你这些口子，所以这里额外提供了 :meth:`WebSocketClient.raw_send`、
:meth:`WebSocketClient.send_fragmented` 与 :meth:`WebSocketClient.recv_frame_raw`。

帧编解码、掩码、分片重组、close 握手全部复用 :class:`server.ws._FrameCodec`：
客户端与服务端的差别只有「发出的帧必须掩码」和握手方向，解析逻辑同源才不会两边漂移。
"""

from __future__ import annotations

import base64
import os
import socket
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

from server.ws import (
    MAX_FRAME_BYTES,
    _FrameCodec,
    _has_token,
    _parse_header_block,
    accept_key,
    WsError,
)

__all__ = ["WsHandshakeError", "WebSocketClient"]


class WsHandshakeError(WsError):
    """握手失败（HTTP 状态码不是 101，或 ``Sec-WebSocket-Accept`` 校验不过）。

    继承 :class:`WsError`，所以调用方只 ``except WsError`` 也能兜住；
    ``status`` 是响应里的 HTTP 状态码（0 表示连状态行都解析不出来），
    测试据此断言「版本不是 13 → 426」这类握手拒绝。
    """

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message, 1002)
        self.status = status


class WebSocketClient(_FrameCodec):
    """``ws://`` 客户端：:meth:`connect` 完成握手，其余 API 与服务端同形。

    ``with WebSocketClient.connect(url) as c:`` 退出时确保 ``close(1000)``；
    收到的帧必须是**不掩码**的（RFC 6455 §5.1），掩码键用 ``os.urandom(4)`` 生成。
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        timeout: float | None = None,
        max_frame: int = MAX_FRAME_BYTES,
    ) -> None:
        super().__init__(sock, timeout=timeout, max_frame=max_frame, mask_outgoing=True)
        #: 协商到的子协议；服务端没回显则为 ``None``
        self.subprotocol: str | None = None
        #: 服务端 101 响应的状态行与头（调试/测试用，键小写）
        self.response_line: str = ""
        self.response_headers: dict[str, str] = {}
        #: 实际请求的 request-target（含查询串）
        self.request_target: str = ""

    @classmethod
    def connect(
        cls,
        url: str,
        *,
        timeout: float = 5.0,
        subprotocols: Sequence[str] = (),
        extra_headers: dict[str, str] | None = None,
    ) -> "WebSocketClient":
        """连到 ``ws://host[:port]/path[?query]`` 并完成握手。

        ``timeout`` 同时用于连接与后续读写（``socket.settimeout``）：
        阻塞式客户端没有超时就会把测试挂死，宁可抛 ``TimeoutError``。
        TLS（``wss://``）不在本子集内，显式拒绝而不是悄悄降级成明文。
        """
        parts = urlsplit(url)
        if parts.scheme != "ws":
            raise WsError(f"只支持 ws:// 的 URL（收到 scheme={parts.scheme!r}）；wss:// 不在本子集内")
        host = parts.hostname
        if not host:
            raise WsError(f"URL 里没有主机名：{url!r}")
        port = parts.port or 80
        target = parts.path or "/"
        if parts.query:
            target = f"{target}?{parts.query}"
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
        except OSError as exc:
            raise WsError(f"连接 {host}:{port} 失败：{exc}") from exc
        client = cls(sock, timeout=timeout)
        client.request_target = target
        try:
            client._handshake(host, port, target, key, tuple(subprotocols), dict(extra_headers or {}))
        except BaseException:
            # 握手失败也要把 fd 放掉：调用方拿到的是异常，没有机会再 close 它
            client._linger_close(0.0)
            raise
        return client

    def _handshake(
        self,
        host: str,
        port: int,
        target: str,
        key: str,
        subprotocols: tuple[str, ...],
        extra_headers: Mapping[str, str],
    ) -> None:
        """发 101 请求、校验 ``Sec-WebSocket-Accept``、记录子协议。"""
        host_header = host if port == 80 else f"{host}:{port}"
        lines = [
            f"GET {target} HTTP/1.1",
            f"Host: {host_header}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        if subprotocols:
            lines.append("Sec-WebSocket-Protocol: " + ", ".join(subprotocols))
        lines.extend(f"{name}: {value}" for name, value in extra_headers.items())
        try:
            self._write_all(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
            head = self._read_http_head()
        except (WsError, OSError) as exc:
            raise WsHandshakeError(f"握手过程中断：{exc}") from exc
        start_line, headers = _parse_header_block(head)
        self.response_line = start_line
        bits = start_line.split()
        status = int(bits[1]) if len(bits) >= 2 and bits[1].isdigit() else 0
        if status != 101:
            raise WsHandshakeError(f"服务端拒绝握手：{start_line!r}", status)
        expected = accept_key(key)
        got = headers.get("sec-websocket-accept", "").strip()
        if got != expected:
            # 这一条是握手的安全要点：Accept 对不上说明对面不是 WebSocket（或被中间人改了）
            raise WsHandshakeError(
                f"Sec-WebSocket-Accept 不匹配（期望 {expected}，收到 {got!r}）", status
            )
        if not _has_token(headers.get("upgrade", ""), "websocket"):
            raise WsHandshakeError("101 响应缺少 Upgrade: websocket", status)
        chosen = headers.get("sec-websocket-protocol", "").strip() or None
        if chosen is not None and chosen not in subprotocols:
            raise WsHandshakeError(f"服务端回显了未请求的子协议 {chosen!r}", status)
        self.subprotocol = chosen  # 客户端严格按服务端回显记，别自作主张
        self.response_headers = headers

    def __enter__(self) -> "WebSocketClient":
        return self

    def __exit__(self, *exc: object) -> None:
        """退出时确保 close(1000)：先发 close 握手，再放掉 socket。"""
        self.close(1000)
