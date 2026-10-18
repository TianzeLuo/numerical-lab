"""RFC 6455 WebSocket **服务端子集**（零第三方依赖）——协议契约见 ``docs/PROTOCOLS.md`` §2.3。

为什么自研：``fastapi / uvicorn / websockets`` 都不在本仓库的依赖红线内（``docs/POLICY.md`` §1），
而 WebSocket 又是 L1→L2→L3 降级链的第一层（``docs/PROTOCOLS.md`` §4），绕不过去。

支持：握手（``Sec-WebSocket-Accept``、子协议回显）、文本/二进制帧、客户端掩码解码、
分片重组（continuation）、ping/pong、close 握手与 close code 校验。

不支持（并显式拒绝）：扩展协商与 ``permessage-deflate`` 压缩（RSV1 非零 → 1002）、
未知 opcode → 1002、超长帧 → 1009、非法 UTF-8 文本 → 1007。

实现纪律（本文件最容易出错的三处，也是测试重点）：

* **短读/短写**：TCP 的一次 ``recv`` 可能只给半个帧头，一次 ``send`` 也可能只写出一部分；
  所有读写都必须走 ``_read_exact`` / ``_write_all`` 的循环，绝不能假设「一次读到一个帧」。
* **发完 close 要把对端残留字节读干净再关 socket**：Linux 在仍有未读数据的 socket 上直接
  ``close()`` 会发 RST，把刚写出的 close 帧一起丢掉，对端只能看到 1006 而不是 1002/1009。
* **控制帧可以插在数据分片之间**：重组状态只由数据帧推进，ping/pong/close 不得干扰它。

服务端与客户端（``server/wsclient.py``）共用 :class:`_FrameCodec` 这一个内核：
两者的差别只有「发出的帧是否掩码」和握手方向，帧解析的边界条件完全同源，
复制两份迟早会漂移。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

__all__ = [
    "MAX_FRAME_BYTES",
    "WS_GUID",
    "WsError",
    "Message",
    "accept_key",
    "WebSocket",
]

#: 单帧（以及重组后的一条消息）的字节上限；超过即以 1009 关闭。
#: 16 MiB 足以装下最大的 ``.nlf`` 帧（PROTOCOLS §1 的数组都是二维小网格）。
MAX_FRAME_BYTES: int = 16 * 1024 * 1024

#: RFC 6455 §1.3 的握手魔术串：Accept = base64(SHA1(Key + GUID))。
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_OP_CONT = 0x0
_OP_TEXT = 0x1
_OP_BINARY = 0x2
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA

_DATA_OPCODES = frozenset({_OP_CONT, _OP_TEXT, _OP_BINARY})
_CONTROL_OPCODES = frozenset({_OP_CLOSE, _OP_PING, _OP_PONG})
_KNOWN_OPCODES = _DATA_OPCODES | _CONTROL_OPCODES

_OPCODE_NAMES = {
    _OP_CONT: "continuation",
    _OP_TEXT: "text",
    _OP_BINARY: "binary",
    _OP_CLOSE: "close",
    _OP_PING: "ping",
    _OP_PONG: "pong",
}

#: 控制帧载荷上限（RFC 6455 §5.5）；close 帧里还要塞 2 字节 code。
_CONTROL_MAX_BYTES = 125

#: 发出 close 后等对端回 close 的上限；等不到就强断，避免把服务线程永久挂住。
_CLOSE_WAIT_SECONDS = 0.5
#: 关 socket 前排空对端残留字节的上限（防 RST 吃掉 close 帧）。
_LINGER_SECONDS = 0.3
#: 握手请求头的字节上限（正常握手 < 1 KiB）。
_HTTP_HEAD_LIMIT = 16 * 1024


class WsError(Exception):
    """WebSocket 层错误。

    ``code`` 表示**协议违规时应该回给对端的 close code**（默认 1002）。
    当错误发生在本端（例如 ping 载荷超过 125 字节、``recv_json`` 收到非 JSON 文本）时，
    该字段只是分类标签，并不会真的去关连接。
    """

    def __init__(self, message: str, code: int = 1002) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Message:
    """一条**完整**消息（分片已在 :meth:`_FrameCodec.recv` 内重组完毕）。

    做成 frozen 值对象：它会跨线程/跨层传递（handler → 计算 → 发送），
    不可变能省掉一整类「读到一半被改写」的诡异问题。
    """

    opcode: str  # "text" | "binary"
    data: bytes

    def __post_init__(self) -> None:
        if self.opcode not in ("text", "binary"):
            raise ValueError(f"opcode 只允许 'text'/'binary'，收到 {self.opcode!r}")
        object.__setattr__(self, "data", bytes(self.data))  # bytearray/memoryview 归一成 bytes

    @property
    def text(self) -> str | None:
        """``opcode == "text"`` 时返回 UTF-8 解码结果；二进制帧返回 ``None``。"""
        if self.opcode != "text":
            return None
        return self.data.decode("utf-8")


def accept_key(sec_websocket_key: str) -> str:
    """``Sec-WebSocket-Accept`` = base64(SHA1(key + ``WS_GUID``))（RFC 6455 §4.2.2）。

    RFC 官方示例：``dGhlIHNhbXBsZSBub25jZQ==`` → ``s3pPLMBiTxaQ9kYGzzhZRbK+xOo=``。
    """
    digest = hashlib.sha1(sec_websocket_key.strip().encode("ascii") + WS_GUID.encode("ascii"))
    return base64.b64encode(digest.digest()).decode("ascii")


def peek_buffered(rfile: Any, sock: socket.socket) -> bytes:
    """非阻塞地取回 ``rfile`` 缓冲区里**已经读到**的字节；没有就返回 ``b""``。

    ``peek()`` 不消费缓冲区，所以交回的字节只会被 WebSocket 层消费一次。
    CPython 3.13 在非阻塞 + 无数据时返回 ``b""``；旧版本可能抛 ``BlockingIOError``，
    两种都当"没数据"处理。
    """
    try:
        old_timeout = sock.gettimeout()
    except OSError:
        return b""
    try:
        sock.setblocking(False)
        return bytes(rfile.peek() or b"")
    except (BlockingIOError, InterruptedError, OSError, ValueError):
        return b""
    finally:
        try:
            sock.settimeout(old_timeout)
        except OSError:
            pass


def initial_bytes_from_handler(handler: Any) -> bytes:
    """重建 ``BaseHTTPRequestHandler`` 已消费的请求头，并拼上其缓冲区里的预读字节。"""
    lines = [bytes(handler.raw_requestline or b"").rstrip(b"\r\n")]
    for name, value in handler.headers.items():
        lines.append(f"{name}: {value}".encode("latin-1", "replace"))
    head = b"\r\n".join(lines) + b"\r\n\r\n"
    return head + peek_buffered(handler.rfile, handler.connection)


def _code_allowed_in_frame(code: int) -> bool:
    """close code 是否可以**出现在帧里**（RFC 6455 §7.4 + IANA 注册表）。

    1005/1006/1015 是「本地占位符」——只能由本端内部使用，永远不许上线；
    1004 保留；1016–2999 未注册；<1000 与 ≥5000 非法。
    """
    if not 1000 <= code < 5000:
        return False
    if code in (1004, 1005, 1006, 1015):
        return False
    if 1016 <= code < 3000:
        return False
    return True


def _truncate_utf8(text: str, limit: int) -> bytes:
    """编码成不超过 ``limit`` 字节的 UTF-8，且**不切断多字节字符**。

    为什么不能直接 ``raw[:limit]``：截在汉字/emoji 中间会产生非法 UTF-8，
    对端按 1007 处理，于是「关闭原因太长」这种小事被升级成协议错误。
    """
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return raw
    return raw[:limit].decode("utf-8", "ignore").encode("utf-8")


def _mask_payload(data: bytes, key: bytes) -> bytes:
    """按 RFC 6455 §5.3 用 4 字节掩码键异或载荷（自反：做两次等于没做）。

    为什么不用逐字节生成器：16 MiB 的帧要循环 1600 万次（纯 Python 约 1 s），
    这里把「重复到等长的掩码」和载荷都当成大整数一次异或，循环交给 C 实现。
    """
    n = len(data)
    if n == 0:
        return b""
    key_stream = (key * ((n + 3) // 4))[:n]
    return (int.from_bytes(data, "big") ^ int.from_bytes(key_stream, "big")).to_bytes(n, "big")


def _parse_header_block(block: bytes) -> tuple[str, dict[str, str]]:
    """把 ``...\\r\\n\\r\\n`` 头块解析成 ``(起始行, 头字典)``。

    头名统一小写（HTTP 头大小写不敏感，调用方不该关心对端怎么写的）；
    同名头按 HTTP 语义用 ", " 连接——客户端完全可以把 ``Sec-WebSocket-Protocol`` 拆成多行。
    """
    lines = block.decode("latin-1").split("\r\n")  # latin-1：逐字节保真，头里允许任意 8 位字节
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep:
            continue  # 畸形头行直接跳过：把关的是下面的显式校验，不是解析器
        key = name.strip().lower()
        val = value.strip()
        headers[key] = f"{headers[key]}, {val}" if key in headers else val
    return (lines[0].strip() if lines else ""), headers


def _has_token(value: str, token: str) -> bool:
    """逗号分隔的头值里是否含某个 token（``Connection: keep-alive, Upgrade``）。"""
    return any(part.strip().lower() == token for part in value.split(","))


def _is_valid_client_key(key: str) -> bool:
    """``Sec-WebSocket-Key`` 必须是 16 字节随机数的 base64（RFC 6455 §4.2.1）。"""
    try:
        raw = base64.b64decode(key.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError):
        return False  # binascii.Error 是 ValueError 的子类
    return len(raw) == 16


class _PeerEof(Exception):
    """内部信号：对端在预期字节数到齐前关掉了 TCP。

    单独一个异常类型是为了区分「帧边界上的 EOF（= 正常断开）」与
    「帧中途的 EOF（= 截断，只能以 1006 记本地日志）」。
    """

    def __init__(self, got: int) -> None:
        super().__init__(f"对端在读到 {got} 字节后关闭")
        self.got = got


class _FrameCodec:
    """服务端/客户端共用的内核：帧编解码、掩码、分片重组、ping/pong、close 握手。

    方向差异只有两处，都是构造参数：``mask_outgoing``（客户端发的帧必须掩码，
    服务端发的帧必须不掩码，RFC 6455 §5.1）与握手方向（在子类里）。
    公开的消息级 API（``recv`` / ``send_text`` / …）也定义在这里，两个子类直接继承，
    于是「文本、二进制、JSON、ping、close」的语义在两侧天然一致。
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        timeout: float | None = None,
        max_frame: int = MAX_FRAME_BYTES,
        mask_outgoing: bool = False,
    ) -> None:
        self._sock = sock
        self._timeout = timeout
        self._max_frame = int(max_frame)
        self._mask_outgoing = mask_outgoing
        self._expect_masked = not mask_outgoing
        self._rbuf = bytearray()  # 半包/粘包的溢出字节都留在这里，跨调用不丢
        self._closed = False
        self._close_sent = False
        self._transport_closed = False
        self._frag_opcode: int | None = None
        self._frag_parts: list[bytes] = []
        self._frag_len = 0
        self._close_code: int | None = None
        self._close_reason = ""
        self._peer_close_code: int | None = None
        self._peer_close_reason = ""
        if timeout is not None:
            sock.settimeout(timeout)
        # Nagle 会把交互式小帧（ping/pong、JSON 控制消息）攒批，往返上引入 ~40 ms 抖动；
        # 这个通道的帧都很小且要的是即时性，关掉更合适。
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass  # 非 TCP（理论上不会）或平台不支持：不影响正确性

    # ------------------------------------------------------------------ 属性

    @property
    def closed(self) -> bool:
        """对端已 close、本端已 close，或传输层已断开时为 ``True``。"""
        return self._closed

    @property
    def close_code(self) -> int | None:
        """这次连接的结束码：本端发起的 close 用本端的值，否则用对端发来的值。"""
        return self._close_code

    @property
    def close_reason(self) -> str:
        """这次连接的结束原因短语（与 :attr:`close_code` 同时确定）。"""
        return self._close_reason

    @property
    def peer_close_code(self) -> int | None:
        """对端 close 帧里的 code（做日志/断言时可以和本端的值对照）。"""
        return self._peer_close_code

    @property
    def peer_close_reason(self) -> str:
        """对端 close 帧里的原因短语。"""
        return self._peer_close_reason

    @property
    def max_frame(self) -> int:
        """本连接的帧长上限（字节）。"""
        return self._max_frame

    # ------------------------------------------------------------- 底层 IO

    def _read_exact(self, n: int) -> bytes:
        """读满 ``n`` 字节；不足则继续 ``recv``，EOF 时抛 :class:`_PeerEof`。

        这是整个实现的地基：**绝不能**写成 ``self._sock.recv(n)`` 就当作读到了 n 字节，
        局域网/本机上也经常只收到帧头或半个载荷。
        """
        if n <= 0:
            return b""
        if self._transport_closed:
            raise WsError("连接已关闭，无法继续读取")
        buf = self._rbuf
        if len(buf) >= n:
            out = bytes(buf[:n])
            del buf[:n]
            return out
        parts = [bytes(buf)]
        buf.clear()
        got = len(parts[0])
        while got < n:
            chunk = self._sock.recv(n - got)
            if not chunk:
                raise _PeerEof(got)
            parts.append(chunk)
            got += len(chunk)
        return b"".join(parts)

    def _write_all(self, data: bytes) -> None:
        """把 ``data`` 全部写出去；``send`` 允许短写，所以必须循环。"""
        if self._transport_closed:
            raise WsError("连接已关闭，无法继续写入")
        view = memoryview(data)
        try:
            while len(view):
                try:
                    sent = self._sock.send(view)
                except OSError as exc:
                    raise WsError(f"写入 socket 失败（对端可能已断开）：{exc}") from exc
                if sent <= 0:
                    raise WsError("写入 socket 返回 0，连接已不可用")
                view = view[sent:]
        finally:
            view.release()

    def _linger_close(self, budget: float = _LINGER_SECONDS) -> None:
        """半关 → 排空对端残留字节 → 关 socket（幂等）。

        为什么不能拿到 close 就直接 ``close()``：接收缓冲区里还有没读完的字节时，
        Linux 会发 RST 而不是 FIN，对端可能连同刚收到的 close 帧一起被丢掉——
        于是「以 1002 关闭」在测试里随机退化成「1006 异常断开」。
        """
        if self._transport_closed:
            return
        self._transport_closed = True
        sock = self._sock
        try:
            sock.shutdown(socket.SHUT_WR)  # 先发 FIN：告诉对端我们不再写数据
        except OSError:
            pass
        old = None
        try:
            old = sock.gettimeout()
        except OSError:
            pass
        if budget > 0:
            try:
                sock.settimeout(budget)
                while True:
                    try:
                        if not sock.recv(65536):
                            break  # 对端也关了：可以安全 close
                    except OSError:
                        break
            except OSError:
                pass
        try:
            if old is not None:
                sock.settimeout(old)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    # ------------------------------------------------------------ 帧编解码

    def _send_frame(self, opcode: int, payload: bytes, *, fin: bool = True) -> None:
        """按 RFC 6455 §5.2 组帧并写出（服务端不掩码，客户端用 ``os.urandom(4)`` 掩码）。"""
        if self._closed or self._close_sent:
            raise WsError("连接已关闭（或 close 帧已发出），不能再发送帧")
        if opcode in _CONTROL_OPCODES and (not fin or len(payload) > _CONTROL_MAX_BYTES):
            raise WsError(
                f"控制帧（{_OPCODE_NAMES[opcode]}）不得分片且载荷不得超过 "
                f"{_CONTROL_MAX_BYTES} 字节（收到 {len(payload)}）"
            )
        head = bytearray()
        head.append((0x80 if fin else 0x00) | (opcode & 0x0F))
        mask_bit = 0x80 if self._mask_outgoing else 0x00
        n = len(payload)
        if n < 126:
            head.append(mask_bit | n)
        elif n < 65536:
            head.append(mask_bit | 126)
            head += struct.pack("!H", n)
        else:
            head.append(mask_bit | 127)
            head += struct.pack("!Q", n)
        if self._mask_outgoing:
            key = os.urandom(4)
            head += key
            payload = _mask_payload(payload, key)
        try:
            self._write_all(bytes(head) + payload)
        except (WsError, OSError):
            self._closed = True  # 写不出去说明链路已断，别让调用方继续以为能发
            raise

    def _read_exact_checked(self, n: int, what: str) -> bytes:
        """``_read_exact`` + 把截断翻译成 1006 错误（帧中途断开不是「正常关闭」）。"""
        try:
            return self._read_exact(n)
        except _PeerEof as eof:
            self._fail(1006, f"对端在{what}中途断开（{eof.got}/{n} 字节）")

    def _recv_frame(self) -> tuple[bool, int, bytes] | None:
        """读一个帧，返回 ``(fin, opcode, payload)``；帧边界上的 EOF 返回 ``None``。

        这里集中了全部「非法输入 → close code」的判定，顺序也按 RFC：
        先 RSV/opcode，再掩码位，再长度（控制帧长度上限在展开扩展长度**之前**就能判），
        最后才是掩码键与载荷——超长帧必须尽早拒绝，绝不能先读 16 MiB 再报错。
        """
        try:
            head = self._read_exact(2)
        except _PeerEof as eof:
            self._closed = True
            self._linger_close()
            if eof.got == 0:
                return None  # 帧边界上的 EOF：对端没做 close 握手就断了（关标签页就是这样）
            raise WsError(f"对端在帧头中途断开（{eof.got}/2 字节）", 1006) from None
        b0, b1 = head[0], head[1]
        fin = bool(b0 & 0x80)
        if b0 & 0x70:
            self._fail(1002, "RSV1/2/3 非零：本子集不支持扩展与 permessage-deflate 压缩")
        opcode = b0 & 0x0F
        if opcode not in _KNOWN_OPCODES:
            self._fail(1002, f"未知 opcode 0x{opcode:x}")
        masked = bool(b1 & 0x80)
        if masked != self._expect_masked:
            self._fail(
                1002,
                "客户端发来的帧必须带掩码" if self._expect_masked else "服务端发来的帧不得带掩码",
            )
        length7 = b1 & 0x7F
        if opcode in _CONTROL_OPCODES and (not fin or length7 > _CONTROL_MAX_BYTES):
            self._fail(1002, "控制帧不得分片，且载荷不得超过 125 字节")
        if length7 < 126:
            length = length7
        elif length7 == 126:
            length = struct.unpack("!H", self._read_exact_checked(2, "16 位长度字段"))[0]
            if length < 126:
                self._fail(1002, "16 位长度字段没有使用最简编码")
        else:
            length = struct.unpack("!Q", self._read_exact_checked(8, "64 位长度字段"))[0]
            if length >> 63:
                self._fail(1002, "64 位长度字段的最高位必须为 0")
            if length < 65536:
                self._fail(1002, "64 位长度字段没有使用最简编码")
        if length > self._max_frame:
            self._fail(1009, f"帧长 {length} 超过上限 {self._max_frame} 字节")
        key = self._read_exact_checked(4, "掩码键") if masked else b""
        payload = self._read_exact_checked(length, "帧载荷") if length else b""
        if masked:
            payload = _mask_payload(payload, key)
        return fin, opcode, payload

    def _fail(self, code: int, reason: str) -> NoReturn:
        """协议违规的统一出口：回一个 close 帧、断开连接、抛 :class:`WsError`。

        1005/1006 不许上线，所以这两种 code 只记录不外发；
        对端已经收到过我们的 close 时也不再补发（RFC：close 只能发一次）。
        """
        if code not in (1005, 1006) and not self._close_sent and not self._closed:
            try:
                self._send_frame(_OP_CLOSE, struct.pack("!H", code) + _truncate_utf8(reason, 123))
            except (WsError, OSError):
                pass
            self._close_sent = True
        self._close_code = code if code not in (1005, 1006) else self._close_code
        self._close_reason = reason
        self._reset_fragments()
        self._closed = True
        self._linger_close()
        raise WsError(reason, code)

    # ------------------------------------------------------- 消息级公开 API

    def recv(self) -> Message | None:
        """收下一条**完整消息**；返回 ``None`` 表示对端已正常关闭。

        分片在这一层重组、ping 在这里自动回 pong、pong 被忽略、close 被回显，
        所以调用方只需要 ``while (msg := ws.recv()) is not None: ...``。

        ``socket.timeout``（``TimeoutError``）会原样抛出：超时发生在帧中途时字节流已经错位，
        调用方应当直接 ``close()``，不要接着 recv。
        """
        if self._closed:
            return None
        while True:
            frame = self._recv_frame()
            if frame is None:
                self._closed = True
                return None  # 对端没有 close 就断了
            fin, opcode, payload = frame
            if opcode == _OP_PING:
                # 即便正处在数据分片序列中间也必须立刻回 pong：控制帧不参与重组状态
                self._send_frame(_OP_PONG, payload)
                continue
            if opcode == _OP_PONG:
                continue  # pong 无需应答
            if opcode == _OP_CLOSE:
                self._handle_peer_close(payload)
                return None
            if opcode == _OP_CONT:
                if self._frag_opcode is None:
                    self._fail(1002, "收到没有起始帧的 continuation 帧")
                self._frag_parts.append(payload)
                self._frag_len += len(payload)
                if self._frag_len > self._max_frame:
                    self._fail(1009, f"重组后的消息超过上限 {self._max_frame} 字节")
                if not fin:
                    continue
                data = b"".join(self._frag_parts)
                start_opcode = self._frag_opcode
                self._reset_fragments()
                return self._build_message(start_opcode, data)
            # 新的数据帧（text/binary）
            if self._frag_opcode is not None:
                self._fail(1002, "上一个分片序列尚未结束，又收到新的数据帧")
            if fin:
                return self._build_message(opcode, payload)
            self._frag_opcode = opcode
            self._frag_parts = [payload]
            self._frag_len = len(payload)

    def recv_json(self) -> dict[str, Any] | None:
        """收一条 JSON 文本消息；对端关闭返回 ``None``。

        非文本帧或不是 JSON 对象的载荷一律抛 :class:`WsError`：
        ``docs/PROTOCOLS.md`` §2 规定「文本帧 = JSON 控制消息、二进制帧 = .nlf」，
        把二进制帧硬当 JSON 解，只会让协议错误在很远的地方才暴露。
        """
        msg = self.recv()
        if msg is None:
            return None
        if msg.opcode != "text":
            raise WsError("期望文本（JSON）帧，收到的是二进制帧")
        try:
            obj = json.loads(msg.data)
        except ValueError as exc:
            raise WsError(f"文本帧不是合法 JSON：{exc}") from exc
        if not isinstance(obj, dict):
            raise WsError(f"JSON 控制消息必须是对象，收到 {type(obj).__name__}")
        return obj

    def send_text(self, s: str) -> None:
        """发一条 UTF-8 文本帧。"""
        self._send_frame(_OP_TEXT, s.encode("utf-8"))

    def send_binary(self, b: bytes) -> None:
        """发一条二进制帧（在 ``docs/PROTOCOLS.md`` §2 里它就是 ``.nlf`` 载荷）。"""
        self._send_frame(_OP_BINARY, bytes(b))

    def send_json(self, obj: Any) -> None:
        """把对象序列化成 UTF-8 文本帧发出。

        ``ensure_ascii=False``：中文原文上线（更短、抓包可读），反正帧本身已经是 UTF-8。
        """
        self.send_text(json.dumps(obj, ensure_ascii=False))

    def ping(self, payload: bytes = b"") -> None:
        """发一个 ping 控制帧（载荷 ≤125 字节）。"""
        self._send_control(_OP_PING, payload)

    def pong(self, payload: bytes = b"") -> None:
        """发一个 pong 控制帧（载荷 ≤125 字节）。"""
        self._send_control(_OP_PONG, payload)

    def close(self, code: int = 1000, reason: str = "") -> None:
        """发 close 帧并收尾；**幂等**——重复调用不会再发第二个 close 帧。

        发完只**有界地**等对端回 close（``_CLOSE_WAIT_SECONDS``），等不到就强断：
        ``__exit__`` 会走这里，无界等待会把单连接单线程的服务线程永久挂住。
        ``code`` 必须是 RFC 6455 §7.4 允许上线的取值（1005/1006/1015 是本地占位符）。
        """
        if not _code_allowed_in_frame(code):
            raise WsError(f"close code {code} 不允许出现在帧里（1005/1006/1015 是本地占位符）")
        if self._close_sent or self._closed:
            self._linger_close()  # 幂等收尾：确保 TCP 也关掉，别把 fd 漏给 gc
            return
        self._close_code = code
        self._close_reason = reason
        body = struct.pack("!H", code) + _truncate_utf8(reason, _CONTROL_MAX_BYTES - 2)
        try:
            self._send_frame(_OP_CLOSE, body)
        except (WsError, OSError):
            self._closed = True
            self._linger_close()
            return
        self._close_sent = True
        self._wait_peer_close(_CLOSE_WAIT_SECONDS)
        self._closed = True
        self._linger_close()

    def send_fragmented(
        self, parts: Sequence[bytes], *, text: bool = False, interleave_ping: bool = False
    ) -> None:
        """把消息拆成若干分片发出（首帧带 text/binary，其余是 continuation）。

        ``interleave_ping=True`` 时在每个分片之间插一个 ping——这是 RFC 明确允许、
        而实现最容易写错的组合（重组状态不能被打断）。测试要用它。
        """
        parts = [bytes(p) for p in parts]
        if not parts:
            raise WsError("send_fragmented 至少需要一个分片")
        first = _OP_TEXT if text else _OP_BINARY
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            self._send_frame(first if index == 0 else _OP_CONT, part, fin=last)
            if interleave_ping and not last:
                self.ping(b"mid")

    def raw_send(self, data: bytes) -> None:
        """**绕过组帧**直接写原始字节。

        只给测试/tools 用：要构造无掩码帧、坏 opcode、半包、声明超长的帧，
        就必须能跳过本实现的打包逻辑（否则就是拿实现证明实现）。
        """
        self._write_all(bytes(data))

    def recv_frame_raw(self) -> tuple[bool, int, bytes] | None:
        """读**一个帧**而不做重组/自动应答，返回 ``(fin, opcode, payload)``。

        同样是给测试用的低层口子：可以看见被 ``recv()`` 吞掉的 pong。
        注意它不会自动回 ping、也不会回显 close。
        """
        return self._recv_frame()

    # ---------------------------------------------------------------- 辅助

    def _send_control(self, opcode: int, payload: bytes) -> None:
        """发控制帧；载荷超限属于调用方错误，直接抛，不发半个帧出去。"""
        if len(payload) > _CONTROL_MAX_BYTES:
            raise WsError(
                f"控制帧载荷不得超过 {_CONTROL_MAX_BYTES} 字节（收到 {len(payload)}）"
            )
        self._send_frame(opcode, bytes(payload))

    def _build_message(self, opcode: int, data: bytes) -> Message:
        """把重组好的载荷变成消息；text 必须是合法 UTF-8（否则 1007）。

        校验放在**整条消息**上而不是单个分片：RFC 允许一个多字节字符被拆到两个分片里。
        """
        if opcode == _OP_TEXT:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                self._fail(1007, f"文本消息不是合法 UTF-8：{exc.reason}")
            return Message("text", data)
        return Message("binary", data)

    def _reset_fragments(self) -> None:
        """清空分片重组状态（一条消息结束/出错后都必须调用）。"""
        self._frag_opcode = None
        self._frag_parts = []
        self._frag_len = 0

    def _record_peer_close(self, payload: bytes) -> None:
        """解析对端 close 帧的载荷并记录 code/reason；非法载荷 → 1002 / 1007。"""
        if len(payload) == 1:
            self._fail(1002, "close 帧载荷只有 1 字节（要么 0 字节，要么 ≥2 字节）")
        if not payload:
            return
        code = struct.unpack("!H", payload[:2])[0]
        try:
            reason = payload[2:].decode("utf-8")
        except UnicodeDecodeError:
            self._fail(1007, "close 帧的原因短语不是合法 UTF-8")
        if not _code_allowed_in_frame(code):
            self._fail(1002, f"close code {code} 不允许出现在帧里")
        self._peer_close_code = code
        self._peer_close_reason = reason
        if self._close_code is None:
            # 本端没发起过 close（对端主动关的）→ 结束码就是它给的那个。
            # 本端发起时保留本端的值：RFC 只要求对端回显 code，不回显 reason，
            # 否则「我发的 close 原因」会被对端的空原因覆盖掉。
            self._close_code = code
            self._close_reason = reason

    def _handle_peer_close(self, payload: bytes) -> None:
        """对端发来 close：回一个 close（通常回显 code），随后即可安全断开。"""
        self._record_peer_close(payload)  # 非法 code / 非法 UTF-8 会先抛 1002 / 1007
        if not self._close_sent:
            # 空载荷就回空载荷：RFC 允许「无 code」的 close，硬塞 1000 反而篡改了语义
            body = payload[:2] if len(payload) >= 2 else b""
            try:
                self._send_frame(_OP_CLOSE, body)
            except (WsError, OSError):
                pass
            self._close_sent = True
        self._closed = True
        self._linger_close()

    def _wait_peer_close(self, budget: float) -> None:
        """发出 close 后有界地等对端回 close（顺带丢弃迟到的数据帧）。"""
        sock = self._sock
        try:
            old = sock.gettimeout()
        except OSError:
            return
        deadline = time.monotonic() + budget
        try:
            while True:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    break
                sock.settimeout(remain if old is None else min(old, remain))
                try:
                    frame = self._recv_frame()
                except (WsError, OSError):
                    break
                if frame is None:
                    break
                _fin, opcode, payload = frame
                if opcode == _OP_CLOSE:
                    try:
                        self._record_peer_close(payload)
                    except WsError:
                        pass
                    break
                if opcode == _OP_PING:
                    try:
                        self._send_frame(_OP_PONG, payload)
                    except (WsError, OSError):
                        break
        finally:
            try:
                sock.settimeout(old)
            except OSError:
                pass

    # ------------------------------------------------------------ 握手读取

    def _read_http_head(self, limit: int = _HTTP_HEAD_LIMIT) -> bytes:
        """读到 HTTP 头结束（``\\r\\n\\r\\n``）为止，返回**含终止符**的头块。

        多读进来的字节留在 ``_rbuf`` 里：严格说 RFC 6455 要求客户端等到 101 再发帧，
        但万一它抢跑，把已经到达的帧字节丢掉会让连接彻底错位；留着零成本。
        """
        while True:
            index = self._rbuf.find(b"\r\n\r\n")
            if index >= 0:
                end = index + 4
                head = bytes(self._rbuf[:end])
                del self._rbuf[:end]
                return head
            if len(self._rbuf) > limit:
                raise WsError(f"HTTP 头超过 {limit} 字节")
            chunk = self._sock.recv(4096)
            if not chunk:
                raise WsError(f"对端在 HTTP 头读完前关闭（已收 {len(self._rbuf)} 字节）")
            self._rbuf += chunk


class WebSocket(_FrameCodec):
    """服务端连接：先 :meth:`accept` 完成握手，再收发消息。

    ``server/app.py`` 的路由大致这样用::

        ws = WebSocket.accept(sock, timeout=30.0, subprotocols=("nlab.v1",))
        with ws:
            while (req := ws.recv_json()) is not None:
                ws.send_json({"type": "pong", "t": req.get("t", 0.0)})
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        timeout: float | None = None,
        max_frame: int = MAX_FRAME_BYTES,
    ) -> None:
        super().__init__(sock, timeout=timeout, max_frame=max_frame, mask_outgoing=False)
        #: 请求行里的 request-target，**含查询串**（路由时自己用 urlsplit 拆，别在这里丢信息）
        self.request_path: str = ""
        #: 请求头，键统一小写（HTTP 头大小写不敏感）；同名头用 ", " 连接
        self.request_headers: dict[str, str] = {}
        #: 协商结果：客户端提议与服务端 ``subprotocols`` 的交集（按服务端偏好序挑），无交集为 ``None``
        self.subprotocol: str | None = None
        #: 完整请求行（日志/调试用）
        self.request_line: str = ""

    @classmethod
    def accept(
        cls,
        sock: socket.socket,
        *,
        timeout: float | None = None,
        max_frame: int = MAX_FRAME_BYTES,
        subprotocols: Sequence[str] = (),
        initial_bytes: bytes = b"",
    ) -> "WebSocket":
        """完成 RFC 6455 服务端握手并返回连接对象。

        握手失败时写回一个最小的 HTTP 错误响应（缺头/方法不对 → 400，
        ``Sec-WebSocket-Version`` 不是 13 → 426 并附 ``Sec-WebSocket-Version: 13``），
        关掉 socket 后抛 :class:`WsError`——此时还没有 WebSocket 连接，
        按 RFC 只能回 HTTP 状态码，不能回 close 帧。

        ``initial_bytes``：调用方**已经从 socket 读走**的流前缀，原样交回即可（可以包含整个
        HTTP 请求头，也可以只是其中一部分；剩下的照常从 socket 读）。典型场景：外层用
        ``socketserver`` 的缓冲读取器（``http.server`` 的 ``rfile`` 就是）读过这个 socket，
        它一次 ``read()`` 可能把紧随握手而来的帧字节一起吞进缓冲区，那些字节必须交回来，
        否则握手后的第一帧就永远到不了这里。

        .. note::
           不能把**仍然留在 socket 里**的字节传进来（例如 ``MSG_PEEK`` 的结果），否则会被解析两次。
           若外层是 ``http.server.BaseHTTPRequestHandler``，**不要照抄"``rfile.peek()`` 一起交回"**
           ——``peek()`` 在缓冲区为空时会阻塞等下一批字节，而客户端正等着 101，于是每次升级都死锁。
           请直接用 :meth:`accept_from_handler`（它做了非阻塞 peek）。
        """
        conn = cls(sock, timeout=timeout, max_frame=max_frame)
        if initial_bytes:
            conn._rbuf += initial_bytes
        conn._handshake(tuple(subprotocols))
        return conn

    @classmethod
    def accept_from_handler(
        cls,
        handler: Any,
        *,
        timeout: float | None = None,
        max_frame: int = MAX_FRAME_BYTES,
        subprotocols: Sequence[str] = (),
    ) -> "WebSocket":
        """从 ``http.server.BaseHTTPRequestHandler`` 升级到 WebSocket。

        ``handler`` 需要提供 ``raw_requestline`` / ``headers`` / ``rfile`` / ``connection``
        （``BaseHTTPRequestHandler`` 全都有）。做两件事：

        1. 把 HTTP 层已经消费掉的请求头**重建**出来（``raw_requestline`` + ``headers``）；
        2. 用非阻塞 peek 取回 ``rfile`` 里已经缓冲的字节（可能含握手后立刻到达的首帧）。

        为什么必须非阻塞：``BufferedReader.peek()`` 在缓冲区为空时会去 socket 上再读一批，
        而此时客户端正在等 101 —— 于是升级卡死。这个坑踩过一次：
        ``server/app.py`` 的实现者实测 ``peek()`` 会等 1.5 s 直到对端再发字节。
        """
        return cls.accept(
            handler.connection,
            timeout=timeout,
            max_frame=max_frame,
            subprotocols=subprotocols,
            initial_bytes=initial_bytes_from_handler(handler),
        )

    def _handshake(self, subprotocols: tuple[str, ...]) -> None:
        """读请求、校验必需头、回 101；子协议只在有交集时回显。"""
        try:
            head = self._read_http_head()
        except (WsError, OSError) as exc:
            self._closed = True
            self._linger_close()
            raise WsError(f"读取握手请求失败：{exc}") from exc
        start_line, headers = _parse_header_block(head)
        self.request_line = start_line
        parts = start_line.split()
        if len(parts) != 3:
            self._reject("400 Bad Request", f"请求行格式错误：{start_line!r}")
        method, target, version = parts
        if method.upper() != "GET":
            self._reject("405 Method Not Allowed", "WebSocket 握手必须是 GET")
        if version != "HTTP/1.1":
            self._reject("400 Bad Request", f"只支持 HTTP/1.1 握手（收到 {version}）")
        if not _has_token(headers.get("upgrade", ""), "websocket"):
            self._reject("400 Bad Request", "缺少 Upgrade: websocket")
        if not _has_token(headers.get("connection", ""), "upgrade"):
            self._reject("400 Bad Request", "缺少 Connection: Upgrade")
        # 区分「压根没写版本头」（握手不完整 → 400）与「写了但不是 13」（版本不支持 → 426）。
        # RFC 6455 §4.2.1 要求版本头必须存在，§4.4 则规定版本不认识时回 426 并告诉对端我们支持 13。
        if "sec-websocket-version" not in headers:
            self._reject("400 Bad Request", "缺少 Sec-WebSocket-Version 头")
        offered_version = headers["sec-websocket-version"].strip()
        if offered_version != "13":
            self._reject(
                "426 Upgrade Required",
                f"不支持的 Sec-WebSocket-Version: {offered_version!r}",
                extra=(("Sec-WebSocket-Version", "13"),),
            )
        key = headers.get("sec-websocket-key", "").strip()
        if not _is_valid_client_key(key):
            self._reject("400 Bad Request", "缺少或非法的 Sec-WebSocket-Key（须为 16 字节 base64）")
        self.request_path = target
        self.request_headers = headers
        # 子协议按服务端偏好挑：服务端最想用的那个只要客户端也提了就用它，
        # 避免「客户端列了十个、把服务端不想要的排在前面」被动接受。
        offered = [p.strip() for p in headers.get("sec-websocket-protocol", "").split(",")]
        chosen = next((want for want in subprotocols if want and want in offered), None)
        self.subprotocol = chosen
        lines = [
            "HTTP/1.1 101 Switching Protocols",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Accept: {accept_key(key)}",
        ]
        if chosen is not None:
            lines.append(f"Sec-WebSocket-Protocol: {chosen}")
        self._write_all(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))

    def _reject(
        self, status: str, reason: str, *, extra: Sequence[tuple[str, str]] = ()
    ) -> NoReturn:
        """握手阶段拒绝：回最小 HTTP 响应 → 关 socket → 抛 :class:`WsError`。"""
        body = (reason + "\n").encode("utf-8")
        head = [
            f"HTTP/1.1 {status}",
            "Content-Type: text/plain; charset=utf-8",
            f"Content-Length: {len(body)}",
            "Connection: close",
            *(f"{name}: {value}" for name, value in extra),
        ]
        try:
            self._write_all(("\r\n".join(head) + "\r\n\r\n").encode("ascii") + body)
        except (WsError, OSError):
            pass
        self._closed = True
        self._linger_close()
        raise WsError(f"握手被拒绝：{status}（{reason}）")

    def __enter__(self) -> "WebSocket":
        return self

    def __exit__(self, *exc: object) -> None:
        """退出时确保 close(1000)：正常路径发 close 握手，出错路径已经是 no-op。"""
        self.close(1000)
