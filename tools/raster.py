"""Scene → PNG 的**参考光栅器**：无浏览器、无第三方图像库、字节可复现。

它是什么
--------
``viz/shared/src/render2d.ts`` 把 ``Scene`` 画到浏览器 canvas；本模块把**同一个** ``Scene``
画成 PNG。于是"视觉回归"在没有浏览器的环境里也能跑（``docs/PROTOCOLS.md`` §6/§9），
而 G8 门禁（``tools/visual_regression.py``）才有一个不依赖 GPU/字体的参照系。

与 TS 侧的对应关系（画法规则以 TS 为准，本模块逐条对齐）
--------------------------------------------------------
* 轴、刻度、1-2-5 步长、对数刻度、``formatNumber``/``tickLabel`` 全部照抄
  ``viz/shared/src/axes.ts``（``niceLimits``/``niceTicks``/``formatNumber``）；
* 数据↔像素、热力图 RGBA、marching squares、溢出检测照抄 ``viz/shared/src/geometry.ts``；
* 归一化/查表照抄 ``viz/shared/src/colorscale.ts``，LUT 取 ``tools/colorscale_tables.py``
  （与 ``viz/shared/colorscales.json``、``colorscale_data.gen.ts`` 同源，G6 守住）；
* 分层顺序、轴框/刻度/标题/徽标/pitfalls 面板的几何照抄 ``viz/shared/src/render2d.ts``。

**红线（与 TS 完全一致的语义）**
* 折线在 NaN/±Inf 处**断开**，绝不用直线把空洞连起来；
* 热力图的"无数据"格子（NaN/±Inf/对数刻度下的非正值/低于 ``clipBelow``）画成
  中性灰 ``(130,130,130,255)`` 而**不是透明**；
* 徽标永远画在最上层；溢出（被轴裁掉 / 没画出来）由 ``check_overflow`` 算出来写进
  ``scene["overflow"]``，不靠人眼发现。

与 TS 的**已知差异**（都是实现手段的差异，不是语义差异；逐条有理由）
--------------------------------------------------------------------
1. **无抗锯齿**：本实现只写整数像素（Bresenham + 扫描线），canvas 会做抗锯齿。
   因此基线与浏览器截图不会逐像素相同；回归目标是**拓扑/配色/标注**一致，而不是反锯齿一致。
   好处是像素完全确定：同输入两次必然同字节（PNG 编码器不写时间戳）。
2. **字体**：环境里没有浏览器也没有字体光栅器，故内嵌一份位图字体（见
   {@link _FONT_BLOB_B64} 的来历说明）。等宽；汉字为全宽（2 个 ASCII 宽）。
   未内嵌的字符画成"缺字框"（缺字扫描见交付说明：现有 0.0 章的 TS 场景与章节源码
   里**没有**缺字）。``measureText`` 的换行口径因此是等宽的，不是比例字体的。
   副作用要说清楚：同样字号下等宽字的**字面比浏览器的比例字体宽**（尤其中文），
   所以 TS 转储场景里那些"按浏览器字宽摆位"的长文本（徽标、meta 面板）在 PNG 里
   会换行更早、偶尔与邻近元素贴在一起。这是**量度差异**不是布局 bug，基线因此是
   "本光栅器眼中的同一场景"，不是浏览器截图的复刻。
3. **热力图位图的行序**：``render2d.ts`` 的快路径（有 ``OffscreenCanvas`` 时）
   ``drawImage`` 没有翻转纵向，于是"第 0 行（y 最小）"被贴到了轴顶部；而它的慢路径
   与 ``geometry.ts`` 的节点采样约定（第 j 行 ↔ y 从 ``extent[2]`` 到 ``extent[3]``）
   都要求第 0 行在**底部**。两条路径不一致是 TS 侧的瑕疵，本实现按**文档化的采样约定 +
   慢路径**画（第 0 行在底部）——这一条在交付说明里作为发现单独提出。
4. **折线不裁剪**：与 TS 一致（``drawLineLayer`` 不套 ``clip()``），只丢弃离轴矩形
   超过 1e6 像素的荒谬点；热力图/等值线/柱状/矢量场裁剪到轴矩形。

``default_scene_for_frame``
---------------------------
**它不是教学图**：它把"一帧 + manifest"折成一个固定的、覆盖全部装饰的回归探针场景，
只服务于视觉回归（见其 docstring）。教学图的场景由章节前端 ``*.model.ts`` 生成，
Python 侧拿不到、也不应该去猜。
"""

from __future__ import annotations

import base64
import decimal
import math
import pathlib
import re
import struct
import sys
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tools.colorscale_tables import NAMES as CMAP_NAMES
from tools.colorscale_tables import TABLES as CMAP_TABLES
from tools.png import PngImage, png_bytes

__all__ = [
    "RasterError",
    "validate_scene",
    "check_overflow",
    "nice_ticks",
    "nice_limits",
    "format_number",
    "format_sci",
    "tick_label",
    "linspace",
    "logspace",
    "map_to_pixel",
    "map_to_data",
    "build_heat_rgba",
    "marching_squares",
    "Canvas",
    "render_scene",
    "render_scene_to_bytes",
    "render_scene_to_file",
    "default_scene_for_frame",
    "KNOWN_BADGE_KINDS",
    "KNOWN_LAYER_KINDS",
    "KNOWN_CMAPS",
]


class RasterError(ValueError):
    """场景不合法，或光栅化过程中遇到无法解释的输入。"""


# =========================================================================== #
# 1. 坐标轴：``viz/shared/src/axes.ts`` 的逐行 Python 对照实现
# =========================================================================== #
_DEFAULT_TARGET = 7
_MANTISSAS: tuple[float, ...] = (1.0, 2.0, 5.0)
_EPS_REL = 1e-12

_MAX_FLOAT = sys.float_info.max
_MIN_FLOAT = 5e-324  # 对应 JS 的 Number.MIN_VALUE


def _require_finite(v: Any, name: str) -> float:
    """校验有限数（与 axes.ts::requireFinite 同口径：「非数」也报错）。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"{name} 必须是有限数，收到 {v!r}")
    return float(v)


def _check_target(target: float | None) -> int:
    t = _DEFAULT_TARGET if target is None else _require_finite(target, "target")
    if t < 2:
        raise ValueError(f"target 至少为 2，收到 {t}")
    return int(math.floor(t))


def _check_count(n: float) -> int:
    v = _require_finite(n, "n")
    if v < 0:
        raise ValueError(f"点数必须非负，收到 {v}")
    return int(math.floor(v))


def _pow10(e: float) -> float:
    """``10^e``，在次正规/上溢边界处饱和（与 axes.ts::pow10 一致）。"""
    if e > 308:
        return _MAX_FLOAT
    if e < -323:
        return _MIN_FLOAT
    v = math.pow(10.0, e)
    if v == 0.0:
        return _MIN_FLOAT
    if not math.isfinite(v):
        return _MAX_FLOAT
    return v


def _decade_exp_down(x: float) -> int:
    """最大的整数 e 使 ``10^e ≤ x``（要求 x > 0）。"""
    e = int(math.floor(math.log10(x) + _EPS_REL))
    while _pow10(e) > x:
        e -= 1
    while _pow10(e + 1) <= x:
        e += 1
    return e


def _decade_exp_up(x: float) -> int:
    """最小的整数 e 使 ``10^e ≥ x``（要求 x > 0）。"""
    e = int(math.ceil(math.log10(x) - _EPS_REL))
    while _pow10(e) < x:
        e += 1
    while _pow10(e - 1) >= x:
        e -= 1
    return e


def linspace(a: float, b: float, n: float) -> np.ndarray:
    """``a … b`` 的 n 点等距序列（含两端，端点精确；语义同 axes.ts::linspace）。"""
    a = _require_finite(a, "a")
    b = _require_finite(b, "b")
    length = _check_count(n)
    out = np.empty(length, dtype=np.float64)
    if length == 0:
        return out
    if length == 1:
        out[0] = a
        return out
    step = (b - a) / (length - 1)
    for i in range(length - 1):
        out[i] = a + i * step
    out[length - 1] = b
    return out


def logspace(a: float, b: float, n: float) -> np.ndarray:
    """``10^a … 10^b`` 的 n 点等比序列（参数是十进制指数）。"""
    exps = linspace(a, b, n)
    return np.power(10.0, exps)


def _ticks_for_step(lo: float, hi: float, step: float) -> list[float]:
    """生成"以 step 为公差的等差数列 ∩ [lo, hi]"（端点处只做 ≤1e-12 相对量钳位）。"""
    out: list[float] = []
    if not (step > 0) or not math.isfinite(step):
        return out
    tol = _EPS_REL * max(abs(lo), abs(hi), step)
    i0 = int(math.ceil(lo / step - 1e-9))
    i1 = int(math.floor(hi / step + 1e-9))
    if i1 < i0:
        return out
    if i1 - i0 > 1_000_000:  # 病态输入：步长相对区间小到失去意义 → 无候选
        return out
    for i in range(i0, i1 + 1):
        v = i * step
        if v < lo and v >= lo - tol:
            v = lo
        if v > hi and v <= hi + tol:
            v = hi
        if v < lo or v > hi:
            continue
        if out and v <= out[-1]:  # 去重 + 保序
            continue
        out.append(v)
    return out


def _candidate_steps(span: float) -> list[float]:
    e0 = int(math.floor(math.log10(span))) - 2
    out: list[float] = []
    for e in range(e0, e0 + 5):
        for m in _MANTISSAS:
            out.append(m * _pow10(e))
    return out


def _coverage(ticks: Sequence[float], lo: float, hi: float) -> float:
    if len(ticks) < 2:
        return 0.0
    span = hi - lo
    if not (span > 0):
        return 1.0
    return (ticks[-1] - ticks[0]) / span


def _select_linear_step(lo: float, hi: float, target: int) -> tuple[float, list[float]]:
    """1-2-5 步长的选择口径（与 axes.ts::selectLinearStep 完全一致）。"""
    span = hi - lo
    min_count = max(2, int(math.ceil(target / 2)))
    max_count = max(min_count, int(math.floor(target * 2)))
    best_step = 0.0
    best_ticks: list[float] = []
    best_key: tuple[int, float, int] | None = None
    for step in _candidate_steps(span):
        ticks = _ticks_for_step(lo, hi, step)
        n = len(ticks)
        if n < min_count or n > max_count:
            continue
        cov = _coverage(ticks, lo, hi)
        key = (0, abs(n - target), n) if cov >= 0.9 else (1, -cov, abs(n - target))
        if best_key is None or key < best_key:
            best_key = key
            best_step = step
            best_ticks = ticks
    if best_key is not None:
        return best_step, best_ticks
    raw = span / target
    e = int(math.floor(math.log10(raw)))
    step = _pow10(e + 1)
    for m in _MANTISSAS:
        s = m * _pow10(e)
        if s >= raw:
            step = s
            break
    ticks = _ticks_for_step(lo, hi, step)
    if len(ticks) < 2:
        ticks = [lo, hi]
    return step, ticks


def _decade_stride(decade_count: int, target: int) -> int:
    return max(1, int(math.ceil((decade_count - 1) / max(1, target - 1))))


def _log_ticks(lo: float, hi: float, target: int) -> list[float]:
    min_count = max(2, int(math.ceil(target / 2)))
    e_lo = _decade_exp_up(lo)
    e_hi = _decade_exp_down(hi)
    count = e_hi - e_lo + 1
    if count >= min_count:
        stride = _decade_stride(count, target)
        best_offset = 0
        best_gap = math.inf
        for off in range(stride):
            first = e_lo + off
            if first > e_hi:
                break
            last = e_hi - ((e_hi - first) % stride)
            gap = max(math.log10(_pow10(first) / lo), math.log10(hi / _pow10(last)))
            if gap < best_gap - 1e-12:
                best_gap = gap
                best_offset = off
        out: list[float] = []
        e = e_lo + best_offset
        while e <= e_hi:
            out.append(_pow10(e))
            e += stride
        return out

    def build(mantissas: Sequence[float]) -> list[float]:
        out2: list[float] = []
        for e in range(e_lo - 1, e_hi + 2):
            for m in mantissas:
                v = m * _pow10(e)
                if v < lo or v > hi:
                    continue
                if out2 and v <= out2[-1]:
                    continue
                out2.append(v)
        return out2

    coarse = build((1.0, 5.0))
    if len(coarse) >= 2:
        return coarse
    fine = build((1.0, 2.0, 5.0))
    if len(fine) >= 2:
        return fine
    return [lo, hi]


def nice_ticks(lo: float, hi: float, target: float | None = None, *, log: bool = False) -> list[float]:
    """``[lo, hi]`` 内的"好看"刻度（升序、严格落在区间内、无重复）。

    线性用 1-2-5 步长；对数用整十次幂（跨度不足时补 ``5×10^k``，再不足补 ``2×10^k``）。
    退化：``lo == hi`` → ``[lo]``；``lo > hi`` → ``ValueError``；对数且 ``lo ≤ 0`` → ``ValueError``。
    """
    lo = _require_finite(lo, "lo")
    hi = _require_finite(hi, "hi")
    tgt = _check_target(target)
    if lo > hi:
        raise ValueError(f"nice_ticks 要求 lo ≤ hi，收到 [{lo}, {hi}]")
    if log:
        if not (lo > 0) or not (hi > 0):
            raise ValueError(f"对数刻度要求 lo > 0 且 hi > 0，收到 [{lo}, {hi}]")
        if lo == hi:
            return [lo]
        return _log_ticks(lo, hi, tgt)
    if lo == hi:
        return [lo]
    return _select_linear_step(lo, hi, tgt)[1]


#: log 轴下界的"地板"：低于它的值一律抬到这里，避免 pad 缩小后下溢成 0。
#: 1e-300 足够低（float64 的 min subnormal 是 5e-324），也不会与真实数据范围打架。
_LOG_FLOOR = 1e-300


def nice_limits(
    lo: float,
    hi: float,
    target: float | None = None,
    *,
    log: bool = False,
    pad: float = 0.0,
) -> tuple[float, float]:
    """把 ``[lo, hi]`` 扩成"好看"的轴范围（返回值恒满足 ``lo' ≤ lo ≤ hi ≤ hi'``）。"""
    lo = _require_finite(lo, "lo")
    hi = _require_finite(hi, "hi")
    tgt = _check_target(target)
    pad = _require_finite(pad, "pad")
    if pad < 0:
        raise ValueError(f"pad 必须非负，收到 {pad}")
    if lo > hi:
        raise ValueError(f"nice_limits 要求 lo ≤ hi，收到 [{lo}, {hi}]")

    if log:
        if not (lo > 0) or not (hi > 0):
            raise ValueError(f"对数刻度要求 lo > 0 且 hi > 0，收到 [{lo}, {hi}]")
        # 下界要有一个"地板"：亚正常数（如 1e-300，章里常拿它当 log 轴的显示地板）再
        # 按 pad 缩小就会**下溢到 0.0**，于是 math.log10(0) 抛 math domain error。
        # 踩过：1.4 的 qr_schur 把 error 的地板设成 1e-300，整条视觉回归流水线直接崩在
        # 兜底探针上（不是那一章变红，而是**整跑挂掉**）。地板取 1e-300 的下一档：
        # 只影响"低于任何真实数据的地板值"，不会把可见范围截断。
        a, b = max(lo, _LOG_FLOOR), hi
        if a == b:
            a = max(a / 10.0, _LOG_FLOOR)
            b = max(b * 10.0, a * 10.0)
        if pad > 0:
            k = pad * math.log10(b / a)
            a = max(a / _pow10(k), _LOG_FLOOR)
            b *= _pow10(k)
            if b <= a:
                b = a * 10.0
        ea = _decade_exp_down(a)
        eb = _decade_exp_up(b)
        min_count = max(2, int(math.ceil(tgt / 2)))
        for _ in range(8):
            count = eb - ea + 1
            if count < min_count:
                break
            stride = _decade_stride(count, tgt)
            rem = (eb - ea) % stride
            if rem == 0:
                break
            need = stride - rem
            ea -= need // 2
            eb += int(math.ceil(need / 2))
        return _pow10(ea), _pow10(eb)

    a, b = lo, hi
    if a == b:
        d = 1.0 if a == 0 else abs(a)
        a -= d
        b += d
    if pad > 0:
        d = pad * (b - a)
        a -= d
        b += d
    for _ in range(8):
        step, ticks = _select_linear_step(a, b, tgt)
        if _coverage(ticks, a, b) >= 1 - 1e-9:
            break
        na = math.floor(a / step + 1e-12) * step
        nb = math.ceil(b / step - 1e-12) * step
        a = min(a, na)
        b = max(b, nb)
    return a, b


# --------------------------------------------------------------------------- #
# 数值格式化：JS 的 toFixed/toPrecision 是"对 double 的精确值做四舍五入，
# 逢半取更靠近 +∞ 的那个整数"，与 Python 的 round-half-even 不同。
# 为了与 axes.ts 逐字符一致，这里用 decimal 精确复刻 ECMAScript 的口径。
# --------------------------------------------------------------------------- #
_DEC_PREC = 400
_DEC_CTX = decimal.Context(prec=_DEC_PREC)


def _dec_round(value: decimal.Decimal, exponent: decimal.Decimal) -> decimal.Decimal:
    """按 ECMAScript ``Number.prototype.toFixed`` 的实际口径量化：逢半**远离 0**。

    规范文字写的是"取更靠近 +∞ 的整数"，但 V8（也就是本仓库真正要对齐的那一端）与
    IEEE-754 的 round-half-away-from-zero 一致：``(2.5).toFixed(0) === "3"``、
    ``(-2.5).toFixed(0) === "-3"``、``(-0.125).toFixed(0) === "-0"``。
    这组行为由 ``tests/test_raster.py::test_cross_language_*`` 逐值对拍守住。
    """
    ctx = decimal.Context(prec=_DEC_PREC, rounding=decimal.ROUND_HALF_UP)
    return ctx.quantize(value, exponent)


def _to_fixed(v: float, decimals: int) -> str:
    """``Number.prototype.toFixed`` 的等价实现（逢半向 +∞ 取整）。"""
    q = _dec_round(decimal.Decimal(v), decimal.Decimal(1).scaleb(-decimals))
    return f"{q:f}"


def _to_precision(v: float, digits: int) -> str:
    """``Number.prototype.toPrecision`` 的等价实现（含进位与定点/指数分支）。"""
    d = decimal.Decimal(v)
    if d == 0:
        return "0" if digits <= 1 else "0." + "0" * (digits - 1)
    e = d.adjusted()
    q = _dec_round(d, decimal.Decimal(1).scaleb(-(digits - 1 - e)))
    if abs(q) >= _DEC_CTX.scaleb(decimal.Decimal(1), e + 1):
        e += 1
        q = _dec_round(d, decimal.Decimal(1).scaleb(-(digits - 1 - e)))
    if e < -6 or e >= digits:
        mant = _dec_round(_DEC_CTX.scaleb(q, -e), decimal.Decimal(1).scaleb(-(digits - 1)))
        sign = "+" if e >= 0 else "-"
        return f"{mant:f}e{sign}{abs(e)}"
    return f"{q:f}"


def _stringify_exponential(v: float, digits_minus_1: int) -> str:
    """``Number.prototype.toExponential`` 的等价实现。"""
    d = decimal.Decimal(v)
    e = d.adjusted()
    exp = decimal.Decimal(1).scaleb(-digits_minus_1)
    q = _dec_round(_DEC_CTX.scaleb(d, -e), exp)
    if abs(q) >= 10:
        e += 1
        q = _dec_round(_DEC_CTX.scaleb(d, -e), exp)
    sign = "+" if e >= 0 else "-"
    return f"{q:f}e{sign}{abs(e)}"


def _strip_fixed_zeros(s: str) -> str:
    return re.sub(r"\.?0+$", "", s) if "." in s else s


def _strip_sci_zeros(mant: str) -> str:
    return re.sub(r"\.?0+$", "", mant) if "." in mant else mant


def format_number(v: float, sig: float = 6) -> str:
    """把 v 格式化为 sig 位有效数字的紧凑字符串（口径同 axes.ts::formatNumber）。

    ``|v| ∈ [1e-3, 1e5]`` 用定点；其余用 ``e`` 记法（指数不带 ``+``）；``0`` → ``"0"``；
    ``NaN`` → ``"NaN"``；``±Inf`` → ``"∞"``/``"-∞"``；尾随 0 一律去掉。
    """
    digits = max(1, int(math.floor(_require_finite(sig, "sig"))))
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    if v == math.inf:
        return "∞"
    if v == -math.inf:
        return "-∞"
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-3 <= a <= 1e5:
        decimals = min(100, max(0, digits - 1 - int(math.floor(math.log10(a)))))
        return _strip_fixed_zeros(_to_fixed(v, decimals))
    s = _stringify_exponential(float(v), digits - 1)
    mant, exp = s.split("e")
    return f"{_strip_sci_zeros(mant)}e{exp[1:] if exp.startswith('+') else exp}"


_SUP: dict[str, str] = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻",
}


def _superscript(exp: int) -> str:
    return "".join(_SUP.get(c, c) for c in str(exp))


def format_sci(v: float, sig: float = 6) -> str:
    """恒为 ``m×10^k`` 形式的科学计数（口径同 axes.ts::formatSci）。"""
    digits = max(1, int(math.floor(_require_finite(sig, "sig"))))
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    if v == math.inf:
        return "∞"
    if v == -math.inf:
        return "-∞"
    if v == 0:
        return "0"
    a = abs(v)
    e = int(math.floor(math.log10(a)))
    if a < _pow10(e):
        e -= 1
    if a >= _pow10(e + 1):
        e += 1
    mant = v / _pow10(e)
    mant_str = _to_precision(mant, digits)
    if abs(float(mant_str)) >= 10:
        e += 1
        mant = mant / 10
        mant_str = _to_precision(mant, digits)
    return f"{_strip_sci_zeros(mant_str)}×10{_superscript(e)}"


def tick_label(v: float, *, log: bool = False, sig: float = 6) -> str:
    """刻度标签：对数模式下整十次幂写 ``10^k``、``5·10^k`` 写 ``5×10^k``，其余同 format_number。"""
    s = 6 if sig is None else max(1, int(math.floor(_require_finite(sig, "sig"))))
    if not log or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or v <= 0:
        return format_number(v, s)
    v = float(v)
    k = math.log10(v)
    kr = int(round(k))
    if abs(v - _pow10(kr)) <= 1e-9 * v:
        if kr == 0:
            return "1"
        if kr == 1:
            return "10"
        return f"10{_superscript(kr)}"
    k5 = int(round(k - math.log10(5)))
    if abs(v - 5 * _pow10(k5)) <= 1e-9 * v:
        if k5 == 0:
            return "5"
        return f"5×10{_superscript(k5)}"
    return format_number(v, s)


def map_to_pixel(
    v: float,
    lo: float,
    hi: float,
    p0: float,
    p1: float,
    scale: str = "linear",
) -> float:
    """数据值 → 像素坐标（仿射；对数刻度取 log10）。退化与越界口径同 axes.ts::mapToPixel。"""
    lo = _require_finite(lo, "lo")
    hi = _require_finite(hi, "hi")
    p0 = _require_finite(p0, "p0")
    p1 = _require_finite(p1, "p1")
    if p0 == p1:
        raise ValueError("像素区间退化（p0 == p1），无法映射")
    if scale == "log":
        if not (lo > 0) or not (hi > 0):
            raise ValueError(f"对数刻度要求 lo > 0 且 hi > 0，收到 [{lo}, {hi}]")
        if lo == hi:
            return (p0 + p1) / 2
        if not (v > 0):
            return math.nan
        t = (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return p0 + t * (p1 - p0)
    if lo == hi:
        return (p0 + p1) / 2
    return p0 + ((v - lo) / (hi - lo)) * (p1 - p0)


def map_to_data(
    p: float,
    lo: float,
    hi: float,
    p0: float,
    p1: float,
    scale: str = "linear",
) -> float:
    """像素坐标 → 数据值（``map_to_pixel`` 的逆）。"""
    lo = _require_finite(lo, "lo")
    hi = _require_finite(hi, "hi")
    p0 = _require_finite(p0, "p0")
    p1 = _require_finite(p1, "p1")
    if p0 == p1:
        raise ValueError("像素区间退化（p0 == p1），无法反解")
    t = (p - p0) / (p1 - p0)
    if scale == "log":
        if not (lo > 0) or not (hi > 0):
            raise ValueError(f"对数刻度要求 lo > 0 且 hi > 0，收到 [{lo}, {hi}]")
        if lo == hi:
            return lo
        a = math.log10(lo)
        b = math.log10(hi)
        return math.pow(10, a + t * (b - a))
    if lo == hi:
        return lo
    return lo + t * (hi - lo)


# =========================================================================== #
# 2. 色标：``viz/shared/src/colorscale.ts`` 的对照实现（LUT 与 TS 同源）
# =========================================================================== #
KNOWN_CMAPS: tuple[str, ...] = tuple(CMAP_NAMES)
_LEVELS = 256
_NAN_RGBA = (130, 130, 130, 255)  # geometry.ts::buildHeatRgba 的中性灰
_LUT_CACHE: dict[str, np.ndarray] = {}


def cmap_table(name: str) -> np.ndarray:
    """取色标的 ``(256, 3)`` uint8 查找表（真源 tools/colorscale_tables.py，与 TS 同值）。"""
    if name not in CMAP_TABLES:
        raise RasterError(f"未知色标：{name!r}（可用：{', '.join(CMAP_NAMES)}）")
    table = _LUT_CACHE.get(name)
    if table is None:
        table = np.asarray(CMAP_TABLES[name], dtype=np.uint8)
        if table.shape != (_LEVELS, 3):
            raise RasterError(f"色标 {name} 的 LUT 形状应为 (256, 3)，得到 {table.shape}")
        _LUT_CACHE[name] = table
    return table


def lookup_cmap(name: str, t: float) -> tuple[int, int, int]:
    """查表：t 截断到 [0,1] 后按 round(t·255) 取 RGB（NaN 按 0 处理）。"""
    table = cmap_table(name)
    if isinstance(t, float) and math.isnan(t):
        tt = 0.0
    elif t == math.inf:
        tt = 1.0
    elif t == -math.inf:
        tt = 0.0
    else:
        tt = 0.0 if t < 0 else (1.0 if t > 1 else float(t))
    idx = min(_LEVELS - 1, max(0, int(math.floor(tt * (_LEVELS - 1) + 0.5))))
    return int(table[idx, 0]), int(table[idx, 1]), int(table[idx, 2])


def normalize_value(v: float, vmin: float, vmax: float, log_scale: bool = False) -> float:
    """把数据值归一化到 ``[0, 1]``（越界截断；口径同 colorscale.ts::normalizeValue）。"""
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        raise RasterError(f"vmin/vmax 必须是有限数，收到 [{vmin}, {vmax}]")
    if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return math.nan
    v = float(v)
    if log_scale:
        if not (vmin > 0) or not (vmax > 0):
            raise RasterError(f"对数色标要求 vmin > 0 且 vmax > 0，收到 [{vmin}, {vmax}]")
        if vmin == vmax:
            raise RasterError(f"对数色标要求 vmin ≠ vmax，收到 vmin = vmax = {vmin}")
        if not (v > 0):
            return math.nan
        t = math.log10(v / vmin) / math.log10(vmax / vmin)
        return 0.0 if t < 0 else (1.0 if t > 1 else t)
    if vmin == vmax:
        return 0.5
    t = (v - vmin) / (vmax - vmin)
    return 0.0 if t < 0 else (1.0 if t > 1 else t)


def apply_cmap(
    name: str,
    values: Any,
    vmin: float,
    vmax: float,
    *,
    log_scale: bool = False,
    clip_below: float | None = None,
) -> np.ndarray:
    """一维数据场 → ``(N, 4)`` uint8 RGBA（不可见像素 alpha = 0）。"""
    table = cmap_table(name)
    if clip_below is not None and not math.isfinite(clip_below):
        raise RasterError(f"clipBelow 必须是有限数，收到 {clip_below!r}")
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        raise RasterError(f"vmin/vmax 必须是有限数，收到 [{vmin}, {vmax}]")
    if log_scale and (not (vmin > 0) or not (vmax > 0)):
        raise RasterError(f"对数色标要求 vmin > 0 且 vmax > 0，收到 [{vmin}, {vmax}]")
    if log_scale and vmin == vmax:
        raise RasterError(f"对数色标要求 vmin ≠ vmax，收到 vmin = vmax = {vmin}")

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = arr.size
    out = np.zeros((n, 4), dtype=np.uint8)
    finite = np.isfinite(arr)
    if clip_below is not None:
        finite &= arr >= clip_below
    if log_scale:
        finite &= arr > 0
    idx_in = np.nonzero(finite)[0]
    if idx_in.size:
        v = arr[idx_in]
        if log_scale:
            t = np.log10(v / vmin) / math.log10(vmax / vmin)
        elif vmin == vmax:
            t = np.full(v.shape, 0.5)
        else:
            t = (v - vmin) / (vmax - vmin)
        t = np.clip(t, 0.0, 1.0)
        idx = np.floor(t * (_LEVELS - 1) + 0.5).astype(np.int64)
        np.clip(idx, 0, _LEVELS - 1, out=idx)
        out[idx_in, 0:3] = table[idx]
        out[idx_in, 3] = 255
    return out


def build_heat_rgba(
    values: Any,
    nx: int,
    ny: int,
    cmap: str,
    vmin: float,
    vmax: float,
    *,
    log_scale: bool = False,
    clip_below: float | None = None,
    nan_color: tuple[int, int, int, int] = _NAN_RGBA,
) -> np.ndarray:
    """网格场 → ``(ny, nx, 4)`` RGBA 字节。

    "无数据"格子（NaN/±Inf、对数刻度下的非正值、低于 ``clipBelow``）画成中性灰
    ``(130,130,130,255)`` 而**不是透明**——白底上透明的格子和"零值"看起来一样，
    这正是本仓库最在意的一类误导（geometry.ts::buildHeatRgba）。
    """
    if int(nx) * int(ny) != len(np.asarray(values).reshape(-1)):
        raise RasterError(f"build_heat_rgba：values 长度 {len(np.asarray(values).reshape(-1))} ≠ nx*ny = {int(nx) * int(ny)}")
    rgba = apply_cmap(cmap, values, vmin, vmax, log_scale=log_scale, clip_below=clip_below)
    invisible = rgba[:, 3] == 0
    if bool(np.any(invisible)):
        rgba[invisible, 0] = nan_color[0]
        rgba[invisible, 1] = nan_color[1]
        rgba[invisible, 2] = nan_color[2]
        rgba[invisible, 3] = nan_color[3]
    return rgba.reshape(int(ny), int(nx), 4)


# =========================================================================== #
# 3. 几何：``viz/shared/src/geometry.ts`` 的对照实现
# =========================================================================== #
def grid_coord(i: float, n: int, lo: float, hi: float) -> float:
    """第 i 个**节点**的坐标：n 个节点均匀铺满 ``[lo, hi]``（含两端）。"""
    if n <= 1:
        return lo
    return lo + ((hi - lo) * i) / (n - 1)


def data_to_pixel(axis: Mapping[str, Any], x: float, y: float) -> tuple[float, float]:
    """数据坐标 → 画布像素（y 轴向下）。"""
    rx, ry, rw, rh = (float(v) for v in axis["rect"])
    return (
        map_to_pixel(x, axis["xlim"][0], axis["xlim"][1], rx, rx + rw, axis["xscale"]),
        map_to_pixel(y, axis["ylim"][0], axis["ylim"][1], ry + rh, ry, axis["yscale"]),
    )


def layer_to_pixel_path(axis: Mapping[str, Any], x: Any, y: Any) -> list[float | None]:
    """把图层数据映射到像素折线；非有限处插入 ``None``（= 断开）。"""
    xs = np.asarray(x, dtype=np.float64).reshape(-1)
    ys = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(xs.size, ys.size)
    out: list[float | None] = []
    for i in range(n):
        px, py = data_to_pixel(axis, float(xs[i]), float(ys[i]))
        if not math.isfinite(px) or not math.isfinite(py):
            out.extend((None, None))
            continue
        out.extend((px, py))
    return out


def marching_squares(values: Any, nx: int, ny: int, level: float) -> list[float]:
    """Marching squares：把标量场在给定层级上切成线段（坐标是网格单位）。

    返回扁平线段数组 ``[x0, y0, x1, y1, …]``；鞍点按中心值决定连接方式（同 geometry.ts）。
    """
    if nx < 2 or ny < 2:
        return []
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    out: list[float] = []

    def at(i: int, j: int) -> float:
        return float(flat[j * nx + i])

    def lerp(a: float, b: float) -> float:
        d = b - a
        if d == 0 or not math.isfinite(d):
            return 0.5
        t = (level - a) / d
        return 0.0 if t < 0 else (1.0 if t > 1 else t)

    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = at(i, j)
            v10 = at(i + 1, j)
            v11 = at(i + 1, j + 1)
            v01 = at(i, j + 1)
            if not (math.isfinite(v00) and math.isfinite(v10) and math.isfinite(v11) and math.isfinite(v01)):
                continue
            code = 0
            if v00 > level:
                code |= 1
            if v10 > level:
                code |= 2
            if v11 > level:
                code |= 4
            if v01 > level:
                code |= 8
            if code == 0 or code == 15:
                continue
            bottom = (i + lerp(v00, v10), float(j))
            right = (float(i + 1), j + lerp(v10, v11))
            top = (i + lerp(v01, v11), float(j + 1))
            left = (float(i), j + lerp(v00, v01))

            def push(a: tuple[float, float], b: tuple[float, float]) -> None:
                out.extend((a[0], a[1], b[0], b[1]))

            if code in (1, 14):
                push(left, bottom)
            elif code in (2, 13):
                push(bottom, right)
            elif code in (3, 12):
                push(left, right)
            elif code in (4, 11):
                push(right, top)
            elif code in (6, 9):
                push(bottom, top)
            elif code in (7, 8):
                push(left, top)
            elif code in (5, 10):
                center = (v00 + v10 + v11 + v01) / 4
                if (code == 5) == (center > level):
                    push(left, top)
                    push(bottom, right)
                else:
                    push(left, bottom)
                    push(right, top)
    return out


def contour_segments(values: Any, nx: int, ny: int, levels: Sequence[float]) -> list[dict[str, Any]]:
    """多层等值线：返回每层的层级与线段（同 geometry.ts::contourSegments）。"""
    return [{"level": float(lv), "segments": marching_squares(values, nx, ny, float(lv))} for lv in levels]


def _precision_str(v: float, digits: int) -> str:
    """``Number.prototype.toPrecision`` 的字符串形式（用于溢出提示）。"""
    if isinstance(v, float) and not math.isfinite(v):
        if math.isnan(v):
            return "NaN"
        return "Infinity" if v > 0 else "-Infinity"
    return _to_precision(float(v), digits)


def check_overflow(scene: Mapping[str, Any], *, tolerance: float = 1e-12) -> dict[str, Any]:
    """扫描整个 Scene，找出"被坐标轴裁掉"与"根本没画出来"的数据（规则 1 的自动化）。"""
    offenders: list[dict[str, Any]] = []
    non_finite = 0
    axes_by_id = {str(a.get("id")): a for a in scene.get("axes", [])}

    def consider(index: int, kind: str, axis_id: str, values: Mapping[str, Any]) -> None:
        nonlocal non_finite
        ax = axes_by_id.get(axis_id)
        if ax is None:
            return
        for which in ("x", "y"):
            data = values.get(which)
            if data is None:
                continue
            lim = ax["xlim"] if which == "x" else ax["ylim"]
            span = abs(float(lim[1]) - float(lim[0]))
            worst = 0.0
            for raw in np.asarray(data, dtype=np.float64).reshape(-1):
                v = float(raw)
                if not math.isfinite(v):
                    non_finite += 1
                    continue
                over = max(float(lim[0]) - v, v - float(lim[1]), 0.0)
                rel = over / span if span > 0 else (math.inf if over > 0 else 0.0)
                if rel > worst:
                    worst = rel
            if worst > tolerance:
                offenders.append({"index": index, "kind": kind, "axes": axis_id, "axis": which, "over": worst, "limit": span})

    for index, layer in enumerate(scene.get("layers", [])):
        kind = layer.get("kind")
        axis_id = str(layer.get("axes"))
        if kind == "line":
            for s in layer.get("series", []):
                consider(index, kind, axis_id, {"x": s.get("x"), "y": s.get("y")})
        elif kind == "scatter":
            consider(index, kind, axis_id, {"x": layer.get("x"), "y": layer.get("y")})
        elif kind == "band":
            band = layer.get("band") or {}
            consider(index, kind, axis_id, {"x": band.get("x"), "y": band.get("yhi")})
            consider(index, kind, axis_id, {"y": band.get("ylo")})
        elif kind == "bars":
            consider(index, kind, axis_id, {"x": layer.get("x"), "y": layer.get("values")})
            errors = layer.get("errors")
            if errors is not None:
                vals = np.asarray(layer.get("values"), dtype=np.float64).reshape(-1)
                errs = np.asarray(errors, dtype=np.float64).reshape(-1)
                n = min(vals.size, errs.size)
                e = np.where(np.isfinite(errs[:n]), errs[:n], 0.0)
                consider(index, kind, axis_id, {"y": vals[:n] + e})
                consider(index, kind, axis_id, {"y": vals[:n] - e})
        elif kind in ("heat", "contour"):
            x0, x1, y0, y1 = (float(v) for v in layer["extent"])
            consider(index, kind, axis_id, {"x": [x0, x1], "y": [y0, y1]})
        elif kind == "vectorfield":
            x0, x1, y0, y1 = (float(v) for v in layer["extent"])
            consider(index, kind, axis_id, {"x": [x0, x1], "y": [y0, y1]})
        elif kind == "loglogfit":
            consider(index, kind, axis_id, {"x": layer.get("h"), "y": layer.get("err")})

    parts: list[str] = []
    if offenders:
        first = offenders[0]
        extra = f"（共 {len(offenders)} 处）" if len(offenders) > 1 else ""
        parts.append(
            f"图层{first['index']}({first['kind']}) 的 {first['axis']} 超出轴 {first['axes']} 范围："
            f"{_precision_str(first['over'], 3)} 倍于其 {_precision_str(first['limit'], 3)} 的跨度{extra}"
        )
    if non_finite > 0:
        parts.append(f"{non_finite} 个非有限值未绘制（以灰色表示“无数据”）")
    return {
        "flag": bool(offenders) or non_finite > 0,
        "note": "；".join(parts),
        "offenders": offenders,
        "nonFinite": non_finite,
    }


# =========================================================================== #
# 4. Scene 校验：``viz/shared/src/scene.ts::validateScene`` 的对照实现
# =========================================================================== #
KNOWN_LAYER_KINDS = ("line", "band", "scatter", "heat", "contour", "bars", "vectorfield", "loglogfit")
KNOWN_BADGE_KINDS = ("source", "autoscale", "overflow", "preview", "critical", "status")
KNOWN_TEXT_ROLES = ("title", "label", "note", "axis", "pitfall")


def _len(a: Any) -> int:
    return 0 if a is None else int(np.asarray(a).reshape(-1).size)


def _finite_all(a: Any) -> bool:
    if a is None:
        return False
    return bool(np.all(np.isfinite(np.asarray(a, dtype=np.float64))))


def validate_scene(scene: Any) -> list[str]:
    """校验 Scene 是否满足契约；返回**问题列表**（空列表 = 合法）。

    与 ``scene.ts::validateScene`` 同一套规则、同一套"先列全再决定"的用法：
    调用方（视觉回归/测试）需要一次性看到全部问题。
    """
    issues: list[str] = []
    if not isinstance(scene, Mapping):
        return ["Scene 不是对象"]
    s: Mapping[str, Any] = scene
    if s.get("version") != 1:
        issues.append(f"version 必须是 1，得到 {s.get('version')!r}")
    for key in ("chapter", "viz", "title", "pitfalls", "dataSource"):
        v = s.get(key)
        if not isinstance(v, str) or v == "":
            issues.append(f"{key} 必须是非空字符串")
    t = s.get("t")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(float(t)):
        issues.append("t 必须是有限数")
    canvas = s.get("canvas") or {}
    if not (isinstance(canvas.get("w"), (int, float)) and canvas.get("w", 0) > 0
            and isinstance(canvas.get("h"), (int, float)) and canvas.get("h", 0) > 0):
        issues.append("canvas.w/h 必须为正")

    axes = s.get("axes")
    if not isinstance(axes, list) or not axes:
        issues.append("至少需要一个坐标轴")
        axes = []
    axis_ids: set[str] = set()
    for a in axes:
        aid = a.get("id")
        if not aid:
            issues.append("轴缺少 id")
        elif aid in axis_ids:
            issues.append(f"轴 id 重复：{aid}")
        axis_ids.add(str(aid))
        rect = a.get("rect")
        if not isinstance(rect, list) or len(rect) != 4:
            issues.append(f"轴 {aid} 的 rect 必须是 4 元组")
        elif not (rect[2] > 0 and rect[3] > 0):
            issues.append(f"轴 {aid} 的宽高必须为正")
        for key in ("xlim", "ylim"):
            lim = a.get(key)
            if not isinstance(lim, list) or len(lim) != 2 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in lim
            ):
                issues.append(f"轴 {aid} 的 {key} 必须是两个有限数")
            elif float(lim[0]) == float(lim[1]):
                issues.append(f"轴 {aid} 的 {key} 退化（上下限相等）——请显式给定范围或标记 autoscaled")
            elif float(lim[0]) > float(lim[1]):
                issues.append(f"轴 {aid} 的 {key} 顺序颠倒")
        if a.get("xscale") == "log" and isinstance(a.get("xlim"), list) and a["xlim"][0] <= 0:
            issues.append(f"轴 {aid}：对数 x 轴下界必须 > 0")
        if a.get("yscale") == "log" and isinstance(a.get("ylim"), list) and a["ylim"][0] <= 0:
            issues.append(f"轴 {aid}：对数 y 轴下界必须 > 0")
        if not isinstance(a.get("autoscaled"), bool):
            issues.append(f"轴 {aid} 必须显式声明 autoscaled（规则 1/4）")
        if not isinstance(a.get("xticks"), list) or not isinstance(a.get("yticks"), list):
            issues.append(f"轴 {aid} 必须给出 xticks/yticks")

    layers = s.get("layers")
    if not isinstance(layers, list) or not layers:
        issues.append("至少需要一个图层")
        layers = []

    def grid_layer_issue(tag: str, layer: Mapping[str, Any]) -> None:
        nx, ny = layer.get("nx", 0), layer.get("ny", 0)
        if not (isinstance(nx, int) and isinstance(ny, int) and nx >= 2 and ny >= 2):
            issues.append(f"{tag} 网格至少 2×2（节点采样含两端），得到 {nx}×{ny}")
        e = layer.get("extent") or []
        if not (isinstance(e, list) and len(e) == 4 and e[1] > e[0] and e[3] > e[2]):
            issues.append(f"{tag} extent 必须严格递增：{e!r}")

    for i, layer in enumerate(layers):
        kind = layer.get("kind")
        tag = f"图层[{i}]({kind if isinstance(kind, str) else '?'})"
        if kind not in KNOWN_LAYER_KINDS:
            issues.append(f"{tag} 未知的 kind")
            continue
        ax = layer.get("axes")
        if not ax or str(ax) not in axis_ids:
            issues.append(f"{tag} 引用了不存在的轴 {ax!r}")
        if kind == "line":
            series = layer.get("series")
            if not isinstance(series, list) or not series:
                issues.append(f"{tag} 至少一条曲线")
            for ser in series or []:
                if _len(ser.get("x")) != _len(ser.get("y")):
                    issues.append(f"{tag} x/y 长度不一致（{_len(ser.get('x'))} vs {_len(ser.get('y'))}）")
                if not _finite_all(ser.get("y")):
                    issues.append(f"{tag} 含非有限纵坐标（请显式裁剪或用 overflow 标记）")
        elif kind == "band":
            band = layer.get("band")
            if not band:
                issues.append(f"{tag} 缺少 band")
            elif _len(band.get("x")) != _len(band.get("ylo")) or _len(band.get("x")) != _len(band.get("yhi")):
                issues.append(f"{tag} x/ylo/yhi 长度不一致")
        elif kind == "scatter":
            if _len(layer.get("x")) != _len(layer.get("y")):
                issues.append(f"{tag} x/y 长度不一致")
            if not (isinstance(layer.get("r"), (int, float)) and layer.get("r") > 0):
                issues.append(f"{tag} 半径必须 > 0")
        elif kind == "heat":
            if _len(layer.get("values")) != layer.get("nx", 0) * layer.get("ny", 0):
                issues.append(f"{tag} values 长度 {_len(layer.get('values'))} ≠ nx*ny = {layer.get('nx', 0) * layer.get('ny', 0)}")
            if layer.get("cmap") not in KNOWN_CMAPS:
                issues.append(f"{tag} 未知色标 {layer.get('cmap')!r}")
            if not (layer.get("vmax", 0) > layer.get("vmin", 0)):
                issues.append(f"{tag} 需要 vmax > vmin")
            if layer.get("logScale") and layer.get("vmin", 0) <= 0:
                issues.append(f"{tag} 对数色标要求 vmin > 0")
            grid_layer_issue(tag, layer)
        elif kind == "contour":
            if _len(layer.get("values")) != layer.get("nx", 0) * layer.get("ny", 0):
                issues.append(f"{tag} values 长度与 nx*ny 不符")
            levels = layer.get("levels")
            if not isinstance(levels, list) or not levels:
                issues.append(f"{tag} 至少一个等值线层级")
            grid_layer_issue(tag, layer)
        elif kind == "bars":
            if _len(layer.get("x")) != _len(layer.get("values")):
                issues.append(f"{tag} x/values 长度不一致")
            errs = layer.get("errors")
            if errs is not None and _len(errs) != _len(layer.get("values")):
                issues.append(f"{tag} errors 长度不符")
        elif kind == "vectorfield":
            if _len(layer.get("u")) != layer.get("nx", 0) * layer.get("ny", 0) or _len(layer.get("v")) != layer.get("nx", 0) * layer.get("ny", 0):
                issues.append(f"{tag} u/v 长度必须等于 nx*ny")
            if not (isinstance(layer.get("stride"), (int, float)) and layer.get("stride") >= 1):
                issues.append(f"{tag} stride 必须 ≥ 1")
            grid_layer_issue(tag, layer)
        elif kind == "loglogfit":
            if _len(layer.get("h")) != _len(layer.get("err")):
                issues.append(f"{tag} h/err 长度不一致")
            for key in ("measuredSlope", "theorySlope"):
                v = layer.get(key)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    issues.append(f"{tag} {key} 必须是有限数（规则 2 不允许缺省）")
            if not str(layer.get("seriesLabel") or "") or not str(layer.get("theoryLabel") or ""):
                issues.append(f"{tag} 必须同时标注实测曲线与理论参考线")

    texts = s.get("texts")
    if not isinstance(texts, list):
        issues.append("texts 必须是数组（可以为空）")
    else:
        for txt in texts:
            if not (isinstance(txt.get("x"), (int, float)) and math.isfinite(float(txt.get("x", math.nan)))
                    and isinstance(txt.get("y"), (int, float)) and math.isfinite(float(txt.get("y", math.nan)))):
                issues.append(f"文本 {str(txt.get('text'))[:12]}… 坐标非有限")
            if txt.get("role") not in KNOWN_TEXT_ROLES:
                issues.append(f"文本角色未知：{txt.get('role')!r}")

    badges = s.get("badges")
    if not isinstance(badges, list):
        issues.append("badges 必须是数组")
    else:
        if not any(b.get("kind") == "source" for b in badges):
            issues.append("缺少 source 徽标（规则 5：必须标明数据来源）")
        for b in badges:
            if b.get("kind") not in KNOWN_BADGE_KINDS:
                issues.append(f"徽标类型未知：{b.get('kind')!r}")
            if not b.get("text"):
                issues.append("徽标文本不得为空")
        issues.extend(_badge_layout_issues(s, badges))
    issues.extend(_text_deco_issues(s))

    overflow = s.get("overflow")
    if not isinstance(overflow, Mapping):
        issues.append("缺少 overflow 字段（规则 1）")
    elif overflow.get("flag") and not str(overflow.get("note") or "").strip():
        issues.append("overflow.flag 为真时必须给出 overflow.note（规则 1）")
    if isinstance(s.get("pitfalls"), str) and len(str(s.get("pitfalls")).strip()) < 10:
        issues.append("pitfalls 太短（规则 7：必须真的说清它会怎样骗你）")
    return issues


#: 角标文字宽度的共享度量：1 个"显示宽度单位"（CJK/全角 = 2，其余 = 1）折合多少像素。
#: 7.0 是本文件位图字体的**实测**比值（任意字号下都是 7.00 px/单位），比浏览器 sans-serif
#: 略宽，因此当保守上界用。TS 侧同一常量在 ``viz/shared/src/scene.ts::BADGE_UNIT_PX``。
BADGE_UNIT_PX = 7.0
#: 角标盒左右内边距之和（`render2d.ts::drawBadge` 与本文件 `_draw_badge` 都是 5px）。
BADGE_PAD_PX = 10.0
#: 两个渲染层的缺省基准字号（``render2d.ts::opts.baseFontSize ?? 13`` 与本文件 ``_BASE_FONT_SIZE``）。
BASE_FONT_SIZE = 13.0
#: 角标盒高度 = 基准字号 + 4（两个渲染层都这么画）。用**绘制高度**而不是名义值：
#: 早先写死 15（对应 base = 11），而真实盒高是 17，差这 2px 足以漏掉"刚好压住半个字"的重叠。
BADGE_BOX_H = BASE_FONT_SIZE + 4.0


def _text_width_units(text: str) -> int:
    """显示宽度单位：CJK/全角记 2，其余记 1（与 ``scene.ts::textWidthUnits`` 同口径）。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(text))


def badge_box_width(text: str) -> float:
    """角标盒的估算宽度（共享尺子，见 :data:`BADGE_UNIT_PX`）。"""
    return _text_width_units(text) * BADGE_UNIT_PX + BADGE_PAD_PX


#: 两列角标时的列宽（与 ``viz/shared/src/model.ts::BADGE_LAYOUT.colW`` 一致）。
BADGE_COL_W = 344.0
#: 画布左右边距（同上，``BADGE_LAYOUT.x``）。
BADGE_MARGIN_X = 8.0


def badge_unit_budget(width: float) -> int:
    """一行角标能容纳的显示宽度单位（与 TS ``badgeUnitBudget`` 同一算法）。

    两列布局用列宽，单列（窄画布）用"整幅宽 − 左右边距"。导出工具会用
    ``--width 200`` 这类窄画布跑探针场景，死守列宽会让角标冲出画布。
    """
    usable = BADGE_COL_W if width >= 2 * BADGE_COL_W + 24 else max(60.0, width - 2 * BADGE_MARGIN_X)
    return max(8, int((usable - BADGE_PAD_PX) // BADGE_UNIT_PX))


def clip_badge_text(text: str, max_units: int) -> str:
    """把角标文案裁到 ``max_units`` 个显示宽度单位（超出用 ``…`` 收尾）。

    与 ``viz/shared/src/model.ts`` 的裁剪**同口径**：宁可少显示几个字，也不许角标越界或叠住。
    """
    if _text_width_units(text) <= max_units:
        return text
    out: list[str] = []
    used = 0
    for ch in str(text):
        w = 2 if ord(ch) > 0x2E80 else 1
        if used + w > max_units - 1:
            break
        out.append(ch)
        used += w
    return "".join(out) + "…"


#: 坐标轴装饰带（与 ``viz/shared/src/scene.ts::AXIS_DECO`` 一致）。
AXIS_DECO_TOP = 18.0
AXIS_DECO_BOTTOM = 28.0
AXIS_DECO_LEFT = 56.0


def axis_deco_bands(axis: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    """某个轴占用的装饰带（相对画布的 ``(x, y, w, h)``），与 TS 侧 ``axisDecoBands`` 同口径。

    **按需生成**：没有刻度就不占刻度带，没有轴名/轴标题就不占那一条。
    这条细节很要紧——好几个模型刻意把 ``xticks`` 清空、改用 ``text`` 层手写类别名
    （``AxisOptions`` 没有 ``xtickLabels``），把空刻度也当成"有装饰"会误报。
    """
    rect = axis.get("rect")
    if not isinstance(rect, Sequence) or len(rect) != 4:
        return []
    x, y, w, h = (float(v) for v in rect)
    out: list[tuple[float, float, float, float]] = []
    if axis.get("title"):
        out.append((x, y - AXIS_DECO_TOP, w, AXIS_DECO_TOP - 2.0))
    if axis.get("xticks"):
        out.append((x, y + h + 3.0, w, 15.0))
    if axis.get("xlabel"):
        out.append((x, y + h + 16.0, w, AXIS_DECO_BOTTOM - 16.0))
    if axis.get("yticks"):
        out.append((x - 41.0, y, 35.0, h))
    if axis.get("ylabel"):
        out.append((x - AXIS_DECO_LEFT, y, 14.0, h))
    return out


def text_box_of(layer: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """一个文本层的估算盒子 ``(x, y, w, h)``——按 ``maxWidth`` 折行，行高 ``size + 4``。

    与 ``viz/shared/src/scene.ts::textBoxOf`` 同口径：两个渲染层都在 ``maxWidth`` 处折行，
    所以一段长说明占的是**几行的块**，不是一行横条（按单行算会误报，0.0 章真报过）。
    """
    size = float(layer.get("size", BASE_FONT_SIZE))
    body = str(layer.get("text", ""))
    max_w = float(layer.get("maxWidth", 0.0) or 0.0)
    if max_w > 0:
        per_line = max(1, int(max_w // BADGE_UNIT_PX))
        lines: list[str] = []
        cur = ""
        for ch in body:
            if cur and _text_width_units(cur + ch) > per_line:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        if cur:
            lines.append(cur)
    else:
        lines = [body]
    width = max((_text_width_units(ln) for ln in lines), default=0) * BADGE_UNIT_PX
    x = float(layer.get("x", 0.0))
    align = layer.get("align")
    if align == "right":
        x -= width
    elif align == "center":
        x -= width / 2.0
    return x, float(layer.get("y", 0.0)), width, len(lines) * (size + 4.0)


def pitfalls_band(scene: Mapping[str, Any], base_font_size: float = BASE_FONT_SIZE) -> tuple[float, float, float, float] | None:
    """底部 pitfalls 面板的盒子 ``(x, y, w, h)``——规则 7 的面板由渲染层自己画。

    与 ``viz/shared/src/render2d.ts::drawPitfalls`` 同口径：字号 ``base-1``、行高 ``size+4``、
    左右各留 12px、底边距 10px。**模型看不见这块面板**（它不在 Scene 里，由渲染器从
    ``scene.pitfalls`` 现画），所以只能由校验器替它守：文本层压在面板下等于没说。
    """
    body = scene.get("pitfalls")
    canvas = scene.get("canvas") or {}
    if not isinstance(body, str) or not body.strip():
        return None
    w = float(canvas.get("w", 0.0))
    h = float(canvas.get("h", 0.0))
    if w <= 0 or h <= 0:
        return None
    size = float(base_font_size) - 1.0
    per_line = max(1, int((w - 24) // BADGE_UNIT_PX))
    lines: list[str] = []
    cur = ""
    for ch in f"它会怎样骗你：{body}":
        if cur and _text_width_units(cur + ch) > per_line:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    line_h = size + 4.0
    y0 = h - 10.0 - len(lines) * line_h
    return 8.0, y0 - 4.0, w - 16.0, len(lines) * line_h + 8.0


def _text_deco_issues(scene: Mapping[str, Any]) -> list[str]:
    """规则 8（布局·其二）：文本层不得压住任何坐标轴的装饰带。

    那里由**渲染层自己**画刻度与轴名，模型看不见、也移不开——所以只能由模型让路：
    给轴下留出 ``28px``（刻度标签 ``bottom+4..15``、轴名 ``bottom+16..28``）再放自己的文本，
    或者干脆把该轴的 ``xlabel`` 置空、改用手写类别名（见 ``spy_ordering`` / ``lu_heatmap``）。
    """
    issues: list[str] = []
    texts = [t for t in (scene.get("texts") or []) if isinstance(t, Mapping)]
    # 文本压文本也是违规：两块说明叠在一起，等于两块都没说清。触发口径是"两个方向都超过
    # 2px"——恰好贴边（共边）不算，真正的叠字才算。1.1 章的 `cancellation` 真报过：
    # 状态行"…纵轴是先验估计…"与游标标签"t=6"叠了 21×15px。
    boxes = []
    for layer in texts:
        bx, by, bw, bh = text_box_of(layer)
        if math.isfinite(bx) and math.isfinite(by):
            boxes.append((bx, by, bw, bh, str(layer.get("text", ""))[:16]))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax, ay, aw, ah, at = boxes[i]
            bx2, by2, bw2, bh2, bt = boxes[j]
            ox = min(ax + aw, bx2 + bw2) - max(ax, bx2)
            oy = min(ay + ah, by2 + bh2) - max(ay, by2)
            if ox > 2.0 and oy > 2.0:
                issues.append(
                    f"文本互相压住（规则 8）：{at}… 与 {bt}… 重叠 {ox:.0f}×{oy:.0f}px"
                )
    band = pitfalls_band(scene)
    if band is not None:
        bx, by, bw, bh = band
        for layer in texts:
            tx, ty, tw, th = text_box_of(layer)
            if not (math.isfinite(tx) and math.isfinite(ty)):
                continue
            ox = min(tx + tw, bx + bw) - max(tx, bx)
            oy = min(ty + th, by + bh) - max(ty, by)
            if ox > 2.0 and oy > 2.0:
                issues.append(
                    f"文本被底部 pitfalls 面板压住（规则 7）：{str(layer.get('text', ''))[:16]}… "
                    f"与面板 [{bx:.0f}, {by:.0f}, {bw:.0f}×{bh:.0f}] 重叠 {ox:.0f}×{oy:.0f}px"
                )
    for axis in scene.get("axes") or []:
        if not isinstance(axis, Mapping):
            continue
        for bx, by, bw, bh in axis_deco_bands(axis):
            for layer in texts:
                tx, ty, tw, th = text_box_of(layer)
                if not (math.isfinite(tx) and math.isfinite(ty)):
                    continue
                if not (tx + tw <= bx or bx + bw <= tx or ty + th <= by or by + bh <= ty):
                    issues.append(
                        f"文本压住坐标轴装饰（规则 8）：{str(layer.get('text', ''))[:16]}… 的盒子 "
                        f"[{tx:.1f}, {ty:.1f}, {tw:.1f}×{th:.1f}] 与轴 {axis.get('id')} 的装饰带 "
                        f"[{bx:.1f}, {by:.1f}, {bw:.1f}×{bh:.1f}] 相交"
                    )
    return issues


def _badge_layout_issues(scene: Mapping[str, Any], badges: Sequence[Any]) -> list[str]:
    """规则 8（布局）：角标盒必须落在画布内，且两两不相交。

    为什么把它写进契约：角标坐标由模型算出，"两枚角标叠在一起"在图上就是读不出来，
    而这件事**只有渲染时才知道**——1.2 与 0.0 章真出现过（左侧 65 单位的
    "副轴：自适应…"横穿到右列，压住临界值角标），当时的校验器查不出来，因为它不知道文字宽度。
    现在两侧用同一把尺子（:func:`badge_box_width`），于是"模型以为放得下、校验器以为放不下"
    这种扯皮不会发生。
    """
    issues: list[str] = []
    canvas = scene.get("canvas") or {}
    cw = float(canvas.get("w") or 0.0)
    ch = float(canvas.get("h") or 0.0)
    boxes: list[tuple[float, float, float, float, str]] = []
    for b in badges:
        if not isinstance(b, Mapping):
            continue
        text = str(b.get("text", ""))
        raw_x, raw_y = b.get("x", 8.0), b.get("y", 8.0)
        try:
            x, y = float(raw_x), float(raw_y)
        except (TypeError, ValueError):
            issues.append(f"徽标 {text[:12]}… 坐标非有限")
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            issues.append(f"徽标 {text[:12]}… 坐标非有限")
            continue
        w = badge_box_width(text)
        if x < 0 or y < 0 or (cw > 0 and x + w > cw) or (ch > 0 and y + BADGE_BOX_H > ch):
            issues.append(
                f"徽标 {text[:16]}… 超出画布：盒子 [{x:.1f}, {y:.1f}, {w:.1f}×{BADGE_BOX_H:g}]，"
                f"画布 {cw:g}×{ch:g}（规则 8：先用 placeBadge/clipBadgeText 收进列宽）"
            )
        boxes.append((x, y, w, BADGE_BOX_H, text))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax, ay, aw, ah, at = boxes[i]
            bx, by, bw, bh, bt = boxes[j]
            if not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay):
                issues.append(
                    f"徽标重叠（规则 8）：{at[:14]}… 与 {bt[:14]}…"
                    f"（盒子 [{ax:.1f}, {ay:.1f}] 与 [{bx:.1f}, {by:.1f}]）"
                )
    return issues


def assert_valid_scene(scene: Any) -> Mapping[str, Any]:
    """校验失败即抛 :class:`RasterError`（带全部问题清单）。"""
    issues = validate_scene(scene)
    if issues:
        raise RasterError("Scene 不合法（%d 处）：\n  - %s" % (len(issues), "\n  - ".join(issues)))
    return scene


# =========================================================================== #
# 5. 内嵌位图字体
#
# 来历：环境里没有浏览器、没有字体光栅器，纯标准库也无法解析 TTF。开发期用 Pillow
# 把 DejaVu Sans Mono（ASCII/符号）+ Droid Sans Fallback（汉字，Apache-2.0）烘焙成
# 4 档 ASCII 位图（11/12/13/18 px）与 1 档汉字位图（14 px，全宽 = 2 个 ASCII 宽），
# zlib+base64 内嵌（见文件末尾 _FONT_BLOB_B64）。运行期不依赖 Pillow / 字体文件。
# 汉字集合 = 开发时把下面这些来源里出现过的**全部非 ASCII 字符**收集起来烘焙：
#   server/chapters/*.py、docs/chapters.yaml、viz/shared/src/scene.ts、
#   viz/shared/src/render2d.ts、tools/raster.py
# 未覆盖的字符画成"缺字框"（确定性，不静默变形）。
#
# 重新烘焙（新增中文文本把"缺字框"引出来时）：
#   1. 装 Pillow（只在开发期用；运行期零依赖），按上面这些来源重新收集字符集；
#   2. ImageFont.truetype 渲染成 4+1 档位图，按 (size, codepoint) → 墨点偏移写出；
#   3. zlib 压缩 + base64，替换文件末尾的 _FONT_BLOB_B64。
#   缺字自检：``_font().glyph_offsets(ch, 13) is None`` 即为缺字；交付时对
#   dist/scenes/**/*.json 全文扫过一遍，0 缺字（见交付说明）。
# =========================================================================== #
def _round_half_up(v: float) -> int:
    """JS ``Math.round`` 的口径（floor(v+0.5)），与 Python 的 banker's rounding 区分开。"""
    return int(math.floor(v + 0.5))


@dataclass
class _Strike:
    """一档位图：定宽 cell + 每个码位的墨点偏移。"""

    size: int
    w: int
    h: int
    baseline: int
    ink: dict[int, np.ndarray]  # 码位 → (N, 2) 的 (dy, dx) 偏移
    wide: bool = False

    def has(self, cp: int) -> bool:
        return cp in self.ink

    def advance(self, cp: int) -> int:
        return self.w


class _BitmapFont:
    """位图字体的解析与排版（等宽；汉字全宽）。"""

    def __init__(self, blob: bytes) -> None:
        self.ascii: dict[int, _Strike] = {}
        self.cjk: dict[int, _Strike] = {}
        self._scaled_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
        self._rot_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
        self._box_cache: dict[tuple[int, int], np.ndarray] = {}
        self._parse(blob)

    # ---------------------------------------------------------------- 解析
    def _parse(self, blob: bytes) -> None:
        raw = zlib.decompress(blob)
        pos = 0
        version, n_ascii = struct.unpack_from("<BB", raw, pos)
        pos += 2
        if version != 1:
            raise RasterError(f"内嵌字体版本 {version} 未知")
        for _ in range(n_ascii):
            size, w, h, baseline, n = struct.unpack_from("<BBBBH", raw, pos)
            pos += 6
            ink = self._read_glyphs(raw, pos, n, w, h)
            pos += self._glyph_bytes(n, w, h)
            self.ascii[int(size)] = _Strike(int(size), int(w), int(h), int(baseline), ink)
        version2, n_cjk = struct.unpack_from("<BB", raw, pos)
        pos += 2
        if version2 != 1:
            raise RasterError(f"内嵌字体版本 {version2} 未知")
        for _ in range(n_cjk):
            size, w, h, baseline, n = struct.unpack_from("<BBBBH", raw, pos)
            pos += 6
            ink = self._read_glyphs(raw, pos, n, w, h)
            pos += self._glyph_bytes(n, w, h)
            self.cjk[int(size)] = _Strike(int(size), int(w), int(h), int(baseline), ink, wide=True)
        if not self.ascii or not self.cjk:
            raise RasterError("内嵌字体为空")

    @staticmethod
    def _glyph_bytes(n: int, w: int, h: int) -> int:
        return n * (4 + h * ((w + 7) // 8))

    @staticmethod
    def _read_glyphs(raw: bytes, pos: int, n: int, w: int, h: int) -> dict[int, np.ndarray]:
        nbytes = (w + 7) // 8
        out: dict[int, np.ndarray] = {}
        for _g in range(n):
            (cp,) = struct.unpack_from("<I", raw, pos)
            pos += 4
            rows = []
            for y in range(h):
                bits = 0
                for b in range(nbytes):
                    bits = (bits << 8) | raw[pos + y * nbytes + b]
                rows.append(bits)
            pos += h * nbytes
            ys: list[int] = []
            xs: list[int] = []
            for y, bits in enumerate(rows):
                for x in range(w):
                    if bits & (1 << (w - 1 - x)):
                        ys.append(y)
                        xs.append(x)
            out[cp] = np.asarray(list(zip(ys, xs)), dtype=np.int32).reshape(-1, 2)
        return out

    # ---------------------------------------------------------------- 排版
    def strike_for(self, size: float) -> _Strike:
        key = min(self.ascii, key=lambda s: (abs(s - size), s))
        return self.ascii[key]

    def cjk_strike_for(self, size: float) -> _Strike:
        key = min(self.cjk, key=lambda s: (abs(s - size), s))
        return self.cjk[key]

    def is_wide(self, ch: str, size: float) -> bool:
        cp = ord(ch)
        if cp < 0x2E80:
            return False
        return True

    def _scaled(self, strike: _Strike, cp: int, dw: int, dh: int) -> np.ndarray:
        """把汉字位图按最近邻缩放到 (dw, dh)（dst 尺寸随 ASCII 档变化时复用）。"""
        key = (id(strike), cp, dw, dh)
        hit = self._scaled_cache.get(key)
        if hit is not None:
            return hit
        offs = strike.ink.get(cp)
        if offs is None:
            return self._box(dw, dh)
        if dw == strike.w and dh == strike.h:
            out = offs
        else:
            dy = np.floor(offs[:, 0] * dh / strike.h).astype(np.int32)
            dx = np.floor(offs[:, 1] * dw / strike.w).astype(np.int32)
            out = np.unique(np.stack([dy, dx], axis=1), axis=0)
        self._scaled_cache[key] = out
        return out

    def _box(self, w: int, h: int) -> np.ndarray:
        """缺字框：1 像素边框（确定性，绝不静默画成空白）。"""
        key = (w, h)
        hit = self._box_cache.get(key)
        if hit is not None:
            return hit
        pts = set()
        for x in range(w):
            pts.add((0, x))
            pts.add((h - 1, x))
        for y in range(h):
            pts.add((y, 0))
            pts.add((y, w - 1))
        out = np.asarray(sorted(pts), dtype=np.int32).reshape(-1, 2)
        self._box_cache[key] = out
        return out

    def glyph_offsets(self, ch: str, size: float) -> tuple[np.ndarray, int, int, int]:
        """返回 ``(墨点偏移, 推进宽度, cell 高, baseline)``（未缩放）。"""
        strike = self.strike_for(size)
        cp = ord(ch)
        if self.is_wide(ch, size):
            cjk = self.cjk_strike_for(size)
            offs = self._scaled(cjk, cp, 2 * strike.w, strike.h)
            return offs, 2 * strike.w, strike.h, strike.baseline
        offs = strike.ink.get(cp)
        if offs is None:
            offs = self._box(strike.w, strike.h)
        return offs, strike.w, strike.h, strike.baseline

    def rotated_offsets(self, ch: str, size: float) -> tuple[np.ndarray, int, int]:
        """旋转 -90°（自下而上）的墨点偏移：``dst[r][c] = src[c][w-1-r]``。"""
        strike = self.strike_for(size)
        cp = ord(ch)
        wide = self.is_wide(ch, size)
        w = 2 * strike.w if wide else strike.w
        h = strike.h
        key = (cp, w, h, 1 if wide else 0)
        hit = self._rot_cache.get(key)
        if hit is not None:
            return hit, h, w  # (offsets, 纵向尺寸=推进宽, 横向尺寸=cell 高)
        if wide:
            cjk = self.cjk_strike_for(size)
            offs = self._scaled(cjk, cp, w, h)
        else:
            offs = strike.ink.get(cp)
            if offs is None:
                offs = self._box(w, h)
        if offs.size == 0:
            rot = offs
        else:
            rot = np.stack([w - 1 - offs[:, 1], offs[:, 0]], axis=1).astype(np.int32)
        self._rot_cache[key] = rot
        return rot, h, w

    def measure(self, text: str, size: float) -> int:
        """等宽口径的文本宽度（像素）。"""
        strike = self.strike_for(size)
        total = 0
        for ch in text:
            total += 2 * strike.w if self.is_wide(ch, size) else strike.w
        return total

    def wrap(self, text: str, size: float, max_width: float) -> list[str]:
        """按**共享的显示宽度单位**换行（与 ``text_box_of`` / TS ``textBoxOf`` 同一把尺子）。

        这里曾经用 ``self.measure()``（真实字形宽度）换行，而布局校验与模型的排版预算用的是
        ``_text_width_units * BADGE_UNIT_PX``（共享代理）——两把尺子差 ~7%（CJK 在 13px 档是
        16px 宽，代理算 14px），于是**渲染出来的行数比预算多一行**，整个文本块比模型预留的高，
        压住下面那块（2.3 章的中面板就这么糊成一团的：预算 4 行、画出 5 行）。
        现在两侧都按代理尺子断行：行数与预算**恒等**，逐行像素宽度最多比代理宽 7%（有富余）。
        """
        out: list[str] = []
        line = ""
        per_line = max(1, int(max_width // BADGE_UNIT_PX)) if max_width > 0 else 0
        for ch in text:
            if per_line and line and _text_width_units(line + ch) > per_line:
                out.append(line)
                line = ch
            else:
                line += ch
        if line:
            out.append(line)
        return out


_FONT: _BitmapFont | None = None


def _font() -> _BitmapFont:
    """惰性解析内嵌字体（首次使用才解压）。"""
    global _FONT
    if _FONT is None:
        _FONT = _BitmapFont(base64.b64decode(_FONT_BLOB_B64))
    return _FONT


# =========================================================================== #
# 6. 画布：整数像素 + 源覆盖混合 + 矩形裁剪栈
# =========================================================================== #
_NAMED_COLORS: dict[str, tuple[int, int, int, float]] = {
    "black": (0, 0, 0, 1.0),
    "white": (255, 255, 255, 1.0),
    "red": (255, 0, 0, 1.0),
    "green": (0, 128, 0, 1.0),
    "blue": (0, 0, 255, 1.0),
    "orange": (255, 165, 0, 1.0),
    "yellow": (255, 255, 0, 1.0),
    "gray": (128, 128, 128, 1.0),
    "grey": (128, 128, 128, 1.0),
    "silver": (192, 192, 192, 1.0),
    "navy": (0, 0, 128, 1.0),
    "teal": (0, 128, 128, 1.0),
    "purple": (128, 0, 128, 1.0),
    "brown": (165, 42, 42, 1.0),
    "pink": (255, 192, 203, 1.0),
    "cyan": (0, 255, 255, 1.0),
    "magenta": (255, 0, 255, 1.0),
    "lime": (0, 255, 0, 1.0),
}

_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_RGB_RE = re.compile(r"^rgba?\(([^)]*)\)$")


def parse_color(spec: Any) -> tuple[int, int, int, float]:
    """CSS 颜色串 → ``(r, g, b, alpha)``；不认识的颜色**报错**而不是猜一个。"""
    if isinstance(spec, (tuple, list)) and len(spec) in (3, 4):
        r, g, b = (int(np.clip(int(c), 0, 255)) for c in spec[:3])
        a = float(spec[3]) if len(spec) == 4 else 1.0
        return r, g, b, a
    if not isinstance(spec, str):
        raise RasterError(f"颜色必须是字符串或 (r,g,b[,a])，收到 {spec!r}")
    text = spec.strip()
    if text in ("none", "transparent"):
        return 0, 0, 0, 0.0
    low = text.lower()
    if low in _NAMED_COLORS:
        return _NAMED_COLORS[low]
    m = _HEX_RE.match(text)
    if m:
        body = m.group(1)
        if len(body) == 3:
            r, g, b = (int(c * 2, 16) for c in body)
            return r, g, b, 1.0
        r, g, b = (int(body[i : i + 2], 16) for i in (0, 2, 4))
        a = int(body[6:8], 16) / 255.0 if len(body) == 8 else 1.0
        return r, g, b, a
    m = _RGB_RE.match(low)
    if m:
        parts = [p.strip() for p in m.group(1).replace("/", " ").split(",")]
        if len(parts) not in (3, 4):
            raise RasterError(f"颜色 {spec!r} 的 rgb()/rgba() 分量数不对")
        vals: list[float] = []
        for i, p in enumerate(parts):
            if i < 3:
                vals.append(float(p[:-1]) * 2.55 if p.endswith("%") else float(p))
            else:
                vals.append(float(p[:-1]) / 100.0 if p.endswith("%") else float(p))
        r, g, b = (int(np.clip(_round_half_up(v), 0, 255)) for v in vals[:3])
        a = float(np.clip(vals[3] if len(vals) == 4 else 1.0, 0.0, 1.0))
        return r, g, b, a
    raise RasterError(f"不认识的 CSS 颜色：{spec!r}")


class Canvas:
    """RGBA 像素缓冲 + 最小绘图原语（全部确定性：无抗锯齿、无随机、无时间戳）。"""

    def __init__(self, width: float, height: float, *, scale: float = 1.0, background: Any = "#ffffff") -> None:
        self.scale = float(scale)
        self.w = max(1, _round_half_up(width * self.scale))
        self.h = max(1, _round_half_up(height * self.scale))
        self.rgba = np.empty((self.h, self.w, 4), dtype=np.uint8)
        r, g, b, a = parse_color(background)
        self.rgba[:, :, 0] = r
        self.rgba[:, :, 1] = g
        self.rgba[:, :, 2] = b
        self.rgba[:, :, 3] = int(np.clip(_round_half_up(a * 255), 0, 255))
        self._clip: list[tuple[int, int, int, int]] = [(0, 0, self.w, self.h)]
        self._font = _font()

    # ------------------------------------------------------------- 坐标/裁剪
    def px(self, v: float) -> int:
        """逻辑坐标 → 设备像素（与 canvas 的 dpr 变换同口径）。"""
        return _round_half_up(float(v) * self.scale)

    def push_clip(self, x: float, y: float, w: float, h: float) -> None:
        x0, y0, x1, y1 = self._dev_rect(x, y, w, h)
        cx0, cy0, cx1, cy1 = self._clip[-1]
        self._clip.append((max(x0, cx0), max(y0, cy0), min(x1, cx1), min(y1, cy1)))

    def pop_clip(self) -> None:
        if len(self._clip) > 1:
            self._clip.pop()

    def _dev_rect(self, x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
        x0, y0 = self.px(x), self.px(y)
        x1, y1 = self.px(x + w), self.px(y + h)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return x0, y0, x1, y1

    def _clipped(self, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int] | None:
        cx0, cy0, cx1, cy1 = self._clip[-1]
        x0, y0 = max(x0, cx0), max(y0, cy0)
        x1, y1 = min(x1, cx1), min(y1, cy1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    # ------------------------------------------------------------- 混合/填充
    def blend_region(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int, float]) -> None:
        box = self._clipped(x0, y0, x1, y1)
        if box is None:
            return
        x0, y0, x1, y1 = box
        r, g, b, a = color
        a8 = int(np.clip(_round_half_up(a * 255), 0, 255))
        if a8 == 0:
            return
        region = self.rgba[y0:y1, x0:x1]
        if a8 == 255:
            region[:, :, 0] = r
            region[:, :, 1] = g
            region[:, :, 2] = b
            region[:, :, 3] = 255
            return
        src = np.array([r, g, b, 255], dtype=np.uint16)
        dst = region.astype(np.uint16)
        blended = (src[None, None, :] * a8 + dst * (255 - a8) + 127) // 255
        region[:] = blended.astype(np.uint8)

    def put(self, x: int, y: int, color: tuple[int, int, int, float]) -> None:
        cx0, cy0, cx1, cy1 = self._clip[-1]
        if not (cx0 <= x < cx1 and cy0 <= y < cy1):
            return
        r, g, b, a = color
        a8 = int(np.clip(_round_half_up(a * 255), 0, 255))
        if a8 == 0:
            return
        if a8 == 255:
            self.rgba[y, x, 0] = r
            self.rgba[y, x, 1] = g
            self.rgba[y, x, 2] = b
            self.rgba[y, x, 3] = 255
            return
        dst = self.rgba[y, x].astype(np.uint16)
        self.rgba[y, x] = ((np.array([r, g, b, 255], dtype=np.uint16) * a8 + dst * (255 - a8) + 127) // 255).astype(np.uint8)

    def fill_rect(self, x: float, y: float, w: float, h: float, color: Any, *, alpha: float = 1.0) -> None:
        x0, y0, x1, y1 = self._dev_rect(x, y, w, h)
        r, g, b, a = parse_color(color)
        self.blend_region(x0, y0, x1, y1, (r, g, b, a * alpha))

    def stroke_rect(self, x: float, y: float, w: float, h: float, color: Any, *, thickness: float = 1.0) -> None:
        t = max(1, self.px(thickness))
        self.fill_rect(x, y, w, t, color)
        self.fill_rect(x, y + h - thickness, w, t, color)
        self.fill_rect(x, y, t, h, color)
        self.fill_rect(x + w - thickness, y, t, h, color)

    def hline(self, y: float, x0: float, x1: float, color: Any, *, thickness: float = 1.0, alpha: float = 1.0) -> None:
        t = max(1, self.px(thickness))
        self.fill_rect(x0, y - t / 2.0, x1 - x0, t, color, alpha=alpha)

    def vline(self, x: float, y0: float, y1: float, color: Any, *, thickness: float = 1.0, alpha: float = 1.0) -> None:
        t = max(1, self.px(thickness))
        self.fill_rect(x - t / 2.0, y0, t, y1 - y0, color, alpha=alpha)

    def line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: Any,
        *,
        width: float = 1.0,
        dash: Sequence[float] | None = None,
        alpha: float = 1.0,
    ) -> None:
        """Bresenham 直线（设备像素），可选虚线（dash 长度按逻辑像素给出）。"""
        r, g, b, a = parse_color(color)
        col = (r, g, b, a * alpha)
        brush = max(1, self.px(width))
        px0, py0 = self.px(x0), self.px(y0)
        px1, py1 = self.px(x1), self.px(y1)
        dx = abs(px1 - px0)
        dy = abs(py1 - py0)
        steps = max(dx, dy, 1)
        sx = 1 if px1 >= px0 else -1
        sy = 1 if py1 >= py0 else -1
        err = dx - dy
        cx, cy = px0, py0
        pattern = [max(1, _round_half_up(d * self.scale)) for d in dash] if dash else None
        period = float(sum(pattern)) if pattern else 0.0
        length = math.hypot(px1 - px0, py1 - py0)
        for step in range(steps + 1):
            draw = True
            if pattern:
                phase = (length * step / steps) % period
                acc = 0.0
                for k, seg in enumerate(pattern):
                    if phase < acc + seg:
                        draw = k % 2 == 0
                        break
                    acc += seg
            if draw:
                self._brush(cx, cy, brush, col)
            if step == steps:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

    def _brush(self, x: int, y: int, size: int, color: tuple[int, int, int, float]) -> None:
        if size <= 1:
            self.put(x, y, color)
            return
        half = size // 2
        self.blend_region(x - half, y - half, x - half + size, y - half + size, color)

    def fill_circle(self, cx: float, cy: float, r: float, color: Any, *, alpha: float = 1.0) -> None:
        r_, g_, b_, a_ = parse_color(color)
        ccx, ccy = self.px(cx), self.px(cy)
        rad = r * self.scale
        if rad <= 0:
            return
        ir = int(math.ceil(rad))
        x0, y0, x1, y1 = self._clipped(ccx - ir, ccy - ir, ccx + ir + 1, ccy + ir + 1) or (0, 0, 0, 0)
        if x1 <= x0:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1]
        inside = ((xs - ccx) ** 2 + (ys - ccy) ** 2) <= rad * rad
        if not bool(np.any(inside)):
            return
        self._blend_mask(x0, y0, inside, (r_, g_, b_, a_ * alpha))

    def fill_polygon(self, points: Sequence[tuple[float, float]], color: Any, *, alpha: float = 1.0) -> None:
        """扫描线填充（nonzero 环绕数），点数 < 3 直接返回。"""
        if len(points) < 3:
            return
        r, g, b, a = parse_color(color)
        col = (r, g, b, a * alpha)
        pts = [(self.px(x), self.px(y)) for x, y in points]
        ys = [p[1] for p in pts]
        y_min, y_max = min(ys), max(ys)
        cx0, cy0, cx1, cy1 = self._clip[-1]
        y_min, y_max = max(y_min, cy0), min(y_max, cy1 - 1)
        n = len(pts)
        for y in range(y_min, y_max + 1):
            xs_cross: list[tuple[float, int]] = []
            for i in range(n):
                x_a, y_a = pts[i]
                x_b, y_b = pts[(i + 1) % n]
                if y_a == y_b:
                    continue
                if (y_a <= y < y_b) or (y_b <= y < y_a):
                    t = (y - y_a) / (y_b - y_a)
                    xs_cross.append((x_a + t * (x_b - x_a), 1 if y_b > y_a else -1))
            if not xs_cross:
                continue
            xs_cross.sort()
            winding = 0
            for i in range(len(xs_cross) - 1):
                winding += xs_cross[i][1]
                if winding != 0:
                    x_start = int(math.floor(xs_cross[i][0] + 0.5))
                    x_end = int(math.floor(xs_cross[i + 1][0] + 0.5))
                    if x_end > x_start:
                        self.blend_region(max(x_start, cx0), y, min(x_end, cx1), y + 1, col)

    def _blend_mask(self, x0: int, y0: int, mask: np.ndarray, color: tuple[int, int, int, float]) -> None:
        """按布尔掩码着色（掩码与 ``self.rgba[y0:y0+mask.shape[0], x0:x0+mask.shape[1]]`` 对齐）。"""
        r, g, b, a = color
        a8 = int(np.clip(_round_half_up(a * 255), 0, 255))
        if a8 == 0:
            return
        ys = slice(y0, y0 + mask.shape[0])
        xs = slice(x0, x0 + mask.shape[1])
        region = self.rgba[ys, xs]
        if a8 == 255:
            region[mask] = (r, g, b, 255)
            return
        sub = region[mask]
        src = np.array([r, g, b, 255], dtype=np.uint16)
        blended = (src[None, :] * a8 + sub.astype(np.uint16) * (255 - a8) + 127) // 255
        region[mask] = blended.astype(np.uint8)

    def draw_image(self, image: np.ndarray, x: float, y: float, w: float, h: float) -> None:
        """把 ``(ny, nx, 4)`` 图像以最近邻铺满逻辑矩形（不做插值，保持"每格一像素"的锐利）。"""
        x0, y0, x1, y1 = self._dev_rect(x, y, w, h)
        box = self._clipped(x0, y0, x1, y1)
        if box is None:
            return
        cx0, cy0, cx1, cy1 = box
        ny, nx = int(image.shape[0]), int(image.shape[1])
        if nx <= 0 or ny <= 0 or x1 <= x0 or y1 <= y0:
            return
        col_idx = np.floor((np.arange(cx0, cx1) - x0) * nx / (x1 - x0)).astype(np.int64)
        row_idx = np.floor((np.arange(cy0, cy1) - y0) * ny / (y1 - y0)).astype(np.int64)
        np.clip(col_idx, 0, nx - 1, out=col_idx)
        np.clip(row_idx, 0, ny - 1, out=row_idx)
        self.rgba[cy0:cy1, cx0:cx1] = image[row_idx][:, col_idx]

    # ------------------------------------------------------------- 文本
    def text(
        self,
        x: float,
        y: float,
        body: str,
        size: float,
        color: Any = "#222",
        *,
        align: str = "left",
        baseline: str = "top",
        bold: bool = False,
        max_width: float | None = None,
        alpha: float = 1.0,
    ) -> None:
        """画一段文本（等宽位图字体；``max_width`` 时自动换行）。

        ``baseline``：``top``（y 是字格顶）/ ``middle``（y 是字格垂直中心）/ ``bottom``（y 是字格底）。
        """
        font = self._font
        rgb = parse_color(color)
        col = (rgb[0], rgb[1], rgb[2], rgb[3] * alpha)
        paragraphs: list[str] = []
        for part in str(body).split("\n"):
            if max_width:
                paragraphs.extend(font.wrap(part, size, max_width))
            else:
                paragraphs.append(part)
        cell_h = font.strike_for(size).h
        # 行高与 ``text_box_of`` 的 ``size + 4`` 对齐（折行时）；单行文本用字格高即可
        line_h = (size + 4.0) if max_width else cell_h
        for i, line in enumerate(paragraphs):
            width = font.measure(line, size)
            if align == "center":
                x0 = x - width / 2.0
            elif align == "right":
                x0 = x - width
            else:
                x0 = x
            top = y + i * line_h
            if baseline == "middle":
                top -= cell_h / 2.0
            elif baseline == "bottom":
                top -= cell_h
            self._draw_line(x0, top, line, size, col, bold)

    def _draw_line(self, x: float, top: float, line: str, size: float, color: tuple[int, int, int, float], bold: bool) -> None:
        font = self._font
        cursor = self.px(x)
        y0 = self.px(top)
        offsets = (0, 1) if bold else (0,)
        for ch in line:
            offs, adv, _h, _b = font.glyph_offsets(ch, size)
            if offs.size:
                for shift in offsets:
                    ys = y0 + offs[:, 0]
                    xs = cursor + shift + offs[:, 1]
                    self._plot_offsets(ys, xs, color)
            cursor += max(1, _round_half_up(adv * self.scale))

    def _plot_offsets(self, ys: np.ndarray, xs: np.ndarray, color: tuple[int, int, int, float]) -> None:
        cx0, cy0, cx1, cy1 = self._clip[-1]
        keep = (ys >= cy0) & (ys < cy1) & (xs >= cx0) & (xs < cx1)
        if not bool(np.any(keep)):
            return
        ys = ys[keep]
        xs = xs[keep]
        r, g, b, a = color
        a8 = int(np.clip(_round_half_up(a * 255), 0, 255))
        if a8 == 0:
            return
        if a8 == 255:
            self.rgba[ys, xs] = (r, g, b, 255)
            return
        sub = self.rgba[ys, xs].astype(np.uint16)
        src = np.array([r, g, b, 255], dtype=np.uint16)
        self.rgba[ys, xs] = ((src[None, :] * a8 + sub * (255 - a8) + 127) // 255).astype(np.uint8)

    def text_vertical(
        self,
        x: float,
        y: float,
        body: str,
        size: float,
        color: Any = "#222",
        *,
        bold: bool = False,
        alpha: float = 1.0,
    ) -> None:
        """旋转 -90° 的文本（自下而上读），垂直居中于 y、字格左缘在 x。"""
        font = self._font
        rgb = parse_color(color)
        col = (rgb[0], rgb[1], rgb[2], rgb[3] * alpha)
        # 不能用 font.measure(body)：它按横排宽度估算，而下面逐字用的是 rotated_offsets 的
        # 固定字格（每个字符 17px）。两者不一致会把标签整体上移半段（5.1 的 y 轴名
        # 于是伸进角标区）。这里按真正要走的字格求和，居中才与实际像素一致。
        advances = [max(1, _round_half_up(adv * self.scale)) for _rot, adv, _cell in (font.rotated_offsets(ch, size) for ch in body)]
        total = sum(advances)
        cursor = self.px(y) + _round_half_up(total / 2.0)
        x0 = self.px(x)
        offsets = (0, 1) if bold else (0,)
        for ch, adv_px in zip(body, advances):
            rot, _adv, _cell_h = font.rotated_offsets(ch, size)
            cursor -= adv_px
            if rot.size:
                for shift in offsets:
                    ys = cursor + rot[:, 0]
                    xs = x0 + shift + rot[:, 1]
                    self._plot_offsets(ys, xs, col)

    # ------------------------------------------------------------- 输出
    def to_png_image(self) -> PngImage:
        """当前像素 → :class:`tools.png.PngImage`（RGBA）。"""
        return PngImage(width=self.w, height=self.h, channels=4, pixels=self.rgba.tobytes())


# =========================================================================== #
# 7. 渲染：``viz/shared/src/render2d.ts`` 的分层顺序与几何
# =========================================================================== #
_BASE_FONT_SIZE = 13
_DEFAULT_FONT_COLOR = "#222"
_GRID_COLOR = "#e8e8e8"
_FRAME_COLOR = "#333"
_TITLE_COLOR = "#111"
_BADGE_BORDER = "#4a5568"

#: Canvas 的裁剪/clip 语义：TS 只对热力图/等值线/柱状/矢量场套 clip()，折线/散点/带状不裁剪。
_CLIPPED_KINDS = ("heat", "contour", "bars", "vectorfield")


@dataclass
class RenderOptions:
    """渲染选项（对应 render2d.ts::RenderOptions）。"""

    base_font_size: float = _BASE_FONT_SIZE
    show_badges: bool = True
    show_titles: bool = True


def _clip_series_to_axis(axis: Mapping[str, Any], path: Sequence[float | None]) -> list[float | None]:
    """只保留轴上可见区间的折线（丢弃离轴 > 1e6 px 的荒谬点，同 render2d.ts）。"""
    rx, ry, rw, rh = (float(v) for v in axis["rect"])
    x0, x1 = rx, rx + rw
    y0, y1 = ry, ry + rh
    out = list(path)
    for i in range(0, len(out), 2):
        px, py = out[i], out[i + 1]
        if px is None or py is None:
            continue
        far = 1e6
        if px < x0 - far or px > x1 + far or py < y0 - far or py > y1 + far:
            out[i] = None
            out[i + 1] = None
    return out


def _stroke_path(canvas: Canvas, path: Sequence[float | None], color: Any, width: float, dash: Sequence[float] | None, alpha: float) -> None:
    """把 ``(x0,y0,x1,y1,…)`` 折线画出来；``None`` 处**断线**。"""
    start: tuple[float, float] | None = None
    prev: tuple[float, float] | None = None
    for i in range(0, len(path), 2):
        px, py = path[i], path[i + 1]
        if px is None or py is None:
            start = None
            prev = None
            continue
        if start is None:
            start = (px, py)
            prev = (px, py)
            continue
        assert prev is not None
        canvas.line(prev[0], prev[1], px, py, color, width=width, dash=dash, alpha=alpha)
        prev = (px, py)


def _draw_axis(canvas: Canvas, axis: Mapping[str, Any], opts: RenderOptions) -> None:
    rx, ry, rw, rh = (float(v) for v in axis["rect"])
    x1, y1 = rx + rw, ry + rh
    if axis.get("grid"):
        for t in axis["xticks"]:
            px, _ = data_to_pixel(axis, float(t), axis["ylim"][0])
            if rx <= px <= x1:
                canvas.vline(px, ry, y1, _GRID_COLOR, thickness=1)
        for t in axis["yticks"]:
            _, py = data_to_pixel(axis, axis["xlim"][0], float(t))
            if ry <= py <= y1:
                canvas.hline(py, rx, x1, _GRID_COLOR, thickness=1)
    if axis.get("frame"):
        canvas.stroke_rect(rx, ry, rw, rh, _FRAME_COLOR, thickness=1)
    tick_size = opts.base_font_size - 2
    for t in axis["xticks"]:
        px, _ = data_to_pixel(axis, float(t), axis["ylim"][0])
        if px < rx - 1 or px > x1 + 1:
            continue
        canvas.text(px, y1 + 4, tick_label(float(t), log=axis.get("xscale") == "log"), tick_size, _FRAME_COLOR, align="center", baseline="top")
    for t in axis["yticks"]:
        _, py = data_to_pixel(axis, axis["xlim"][0], float(t))
        if py < ry - 1 or py > y1 + 1:
            continue
        canvas.text(rx - 6, py, tick_label(float(t), log=axis.get("yscale") == "log"), tick_size, _FRAME_COLOR, align="right", baseline="middle")
    if opts.show_titles:
        if axis.get("xlabel"):
            canvas.text(rx + rw / 2, y1 + opts.base_font_size + 16, str(axis["xlabel"]), opts.base_font_size, _DEFAULT_FONT_COLOR, align="center", baseline="bottom", bold=True)
        if axis.get("title"):
            canvas.text(rx + rw / 2, ry - 6, str(axis["title"]), opts.base_font_size, _DEFAULT_FONT_COLOR, align="center", baseline="bottom", bold=True)
        if axis.get("ylabel"):
            canvas.text_vertical(rx - 50, ry + rh / 2, str(axis["ylabel"]), opts.base_font_size, _DEFAULT_FONT_COLOR, bold=True)


def _draw_line_layer(
    canvas: Canvas,
    axis: Mapping[str, Any],
    layer: Mapping[str, Any],
    *,
    avoid: Sequence[tuple[float, float, float, float]] = (),
) -> None:
    canvas.push_clip(*[float(v) for v in axis["rect"]])
    for s in layer.get("series", []):
        path = _clip_series_to_axis(axis, layer_to_pixel_path(axis, s.get("x"), s.get("y")))
        color = s.get("color", "#4c78a8")
        width = float(s.get("width", 1.6))
        alpha = float(s.get("alpha", 1.0))
        _stroke_path(canvas, path, color, width, s.get("dash"), alpha)
        marker = s.get("marker", "none")
        if marker and marker != "none":
            radius = max(2.0, width * 1.6)
            for i in range(0, len(path), 2):
                px, py = path[i], path[i + 1]
                if px is None or py is None:
                    continue
                if i >= 2 and path[i - 1] is None:
                    continue  # 断点后不画标记
                if marker == "o":
                    canvas.fill_circle(px, py, radius, color, alpha=alpha)
                elif marker == "s":
                    canvas.fill_rect(px - radius, py - radius, 2 * radius, 2 * radius, color, alpha=alpha)
                else:
                    canvas.line(px - radius, py - radius, px + radius, py + radius, color, width=1, alpha=alpha)
                    if marker == "+":
                        canvas.line(px - radius, py + radius, px + radius, py - radius, color, width=1, alpha=alpha)
        if s.get("label"):
            last = None
            for i in range(0, len(path), 2):
                if path[i] is not None and path[i + 1] is not None:
                    last = (path[i], path[i + 1])
            if last is not None:
                # 标签锚在末点，但**不许越出绘图区**：右栏文字是场景层的文本，渲染器看不见，
                # 越栏的标签会正好压在上面（2.2 章的曲线标签就压过右栏说明，PNG 上才看得见）。
                # 放不下就翻到末点左侧（右对齐），再放不下就贴着绘图区右边界内侧。
                label = str(s["label"])
                size = _BASE_FONT_SIZE - 1
                rx, _ry, rw, _rh = (float(v) for v in axis["rect"])
                lw = float(_font().measure(label, size))
                box_h = size + 4.0
                # 标签锚在末点，但**必须同时满足**：不越出绘图区、不压住场景文本。
                # 越栏会伸进右栏说明、挤进绘图区又会压住模型的说明行（1.3 章的曲线标签
                # 两种都犯过）——所以放不下就**不画**：角标与说明里本来就有它的名字，
                # 宁可不画，也不许叠着说（防误导七条第一条：不许静默，也不许含糊）。
                for x in (last[0] + 4.0, last[0] - 4.0 - lw):
                    if x < rx or x + lw > rx + rw:
                        continue
                    box = (x, last[1] - 2.0 - box_h, lw, box_h)
                    if any(
                        min(box[0] + box[2], b[0] + b[2]) - max(box[0], b[0]) > 2.0
                        and min(box[1] + box[3], b[1] + b[3]) - max(box[1], b[1]) > 2.0
                        for b in avoid
                    ):
                        continue
                    canvas.text(x, last[1] - 2, label, size, color, align="left", baseline="bottom")
                    break
    canvas.pop_clip()


def _draw_band_layer(canvas: Canvas, axis: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
    band = layer.get("band") or {}
    xs = np.asarray(band.get("x"), dtype=np.float64).reshape(-1)
    ylo = np.asarray(band.get("ylo"), dtype=np.float64).reshape(-1)
    yhi = np.asarray(band.get("yhi"), dtype=np.float64).reshape(-1)
    n = min(xs.size, ylo.size, yhi.size)
    canvas.push_clip(*[float(v) for v in axis["rect"]])
    points: list[tuple[float, float]] = []
    for i in range(n):
        px, py = data_to_pixel(axis, float(xs[i]), float(yhi[i]))
        if math.isfinite(px) and math.isfinite(py):
            points.append((px, py))
    for i in range(n - 1, -1, -1):
        px, py = data_to_pixel(axis, float(xs[i]), float(ylo[i]))
        if math.isfinite(px) and math.isfinite(py):
            points.append((px, py))
    canvas.fill_polygon(points, band.get("color", "#4c78a8"), alpha=float(band.get("alpha", 0.3)))
    canvas.pop_clip()


def _draw_scatter_layer(canvas: Canvas, axis: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
    xs = np.asarray(layer.get("x"), dtype=np.float64).reshape(-1)
    ys = np.asarray(layer.get("y"), dtype=np.float64).reshape(-1)
    r = float(layer.get("r", 2.5))
    canvas.push_clip(*[float(v) for v in axis["rect"]])
    for i in range(min(xs.size, ys.size)):
        xv, yv = float(xs[i]), float(ys[i])
        if not (math.isfinite(xv) and math.isfinite(yv)):
            continue
        px, py = data_to_pixel(axis, xv, yv)
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        canvas.fill_circle(px, py, r, layer.get("color", "#4c78a8"), alpha=float(layer.get("alpha", 1.0)))
    canvas.pop_clip()


def _draw_heat_layer(canvas: Canvas, axis: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
    nx, ny = int(layer["nx"]), int(layer["ny"])
    rgba = build_heat_rgba(
        layer["values"], nx, ny, str(layer["cmap"]), float(layer["vmin"]), float(layer["vmax"]),
        log_scale=bool(layer.get("logScale")), clip_below=layer.get("clipBelow"),
    )
    ex0, ex1, ey0, ey1 = (float(v) for v in layer["extent"])
    x0, y0 = data_to_pixel(axis, ex0, ey1)
    x1, y1 = data_to_pixel(axis, ex1, ey0)
    left, top = min(x0, x1), min(y0, y1)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    # 见模块 docstring：按 geometry.ts 的节点采样约定（第 0 行 = extent[2] = y 最小）画，
    # 即第 0 行在**底部**，因此这里纵向翻转后再贴图（render2d.ts 的快路径漏了这次翻转）。
    canvas.push_clip(*[float(v) for v in axis["rect"]])
    canvas.draw_image(rgba[::-1], left, top, dx, dy)
    canvas.pop_clip()


def _draw_contour_layer(canvas: Canvas, axis: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
    nx, ny = int(layer["nx"]), int(layer["ny"])
    ex0, ex1, ey0, ey1 = (float(v) for v in layer["extent"])
    color = layer.get("color", "#222")
    width = float(layer.get("width", 1.0))
    canvas.push_clip(*[float(v) for v in axis["rect"]])
    for entry in contour_segments(layer["values"], nx, ny, layer.get("levels", [])):
        seg = entry["segments"]
        for i in range(0, len(seg), 4):
            px0, py0 = data_to_pixel(axis, grid_coord(seg[i], nx, ex0, ex1), grid_coord(seg[i + 1], ny, ey0, ey1))
            px1, py1 = data_to_pixel(axis, grid_coord(seg[i + 2], nx, ex0, ex1), grid_coord(seg[i + 3], ny, ey0, ey1))
            if not all(math.isfinite(v) for v in (px0, py0, px1, py1)):
                continue
            canvas.line(px0, py0, px1, py1, color, width=width)
    canvas.pop_clip()


def _draw_bars_layer(canvas: Canvas, axis: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
    xs = np.asarray(layer.get("x"), dtype=np.float64).reshape(-1)
    vals = np.asarray(layer.get("values"), dtype=np.float64).reshape(-1)
    errors = layer.get("errors")
    errs = None if errors is None else np.asarray(errors, dtype=np.float64).reshape(-1)
    base = float(layer.get("baseline", 0.0))
    width = float(layer.get("width", 0.8))
    color = layer.get("color", "#4c78a8")
    canvas.push_clip(*[float(v) for v in axis["rect"]])
    for i in range(min(xs.size, vals.size)):
        xv, yv = float(xs[i]), float(vals[i])
        if not (math.isfinite(xv) and math.isfinite(yv)):
            continue
        half = width / 2.0
        pxa, pya = data_to_pixel(axis, xv - half, yv)
        pxb, pyb = data_to_pixel(axis, xv + half, base)
        if not all(math.isfinite(v) for v in (pxa, pya, pxb, pyb)):
            continue
        canvas.fill_rect(min(pxa, pxb), min(pya, pyb), abs(pxb - pxa), abs(pyb - pya), color)
        if errs is not None and i < errs.size:
            err = float(errs[i])
            if math.isfinite(err) and err > 0:
                xm, _ = data_to_pixel(axis, xv, base)
                _, y_top = data_to_pixel(axis, xv, yv + err)
                _, y_bot = data_to_pixel(axis, xv, max(base, yv - err))
                canvas.line(xm, y_top, xm, y_bot, color, width=1)
                cap = max(2.0, abs(pxb - pxa) / 4.0)
                canvas.line(xm - cap, y_top, xm + cap, y_top, color, width=1)
                canvas.line(xm - cap, y_bot, xm + cap, y_bot, color, width=1)
    canvas.pop_clip()


def _draw_vectorfield_layer(canvas: Canvas, axis: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
    nx, ny = int(layer["nx"]), int(layer["ny"])
    ex0, ex1, ey0, ey1 = (float(v) for v in layer["extent"])
    u = np.asarray(layer["u"], dtype=np.float64).reshape(-1)
    v = np.asarray(layer["v"], dtype=np.float64).reshape(-1)
    stride = max(1, int(layer.get("stride", 1)))
    scale = float(layer.get("scale", 1.0))
    color = layer.get("color", "#666")
    canvas.push_clip(*[float(v) for v in axis["rect"]])
    for j in range(0, ny, stride):
        for i in range(0, nx, stride):
            k = j * nx + i
            if k >= u.size or k >= v.size:
                continue
            uv, vv = float(u[k]), float(v[k])
            if not (math.isfinite(uv) and math.isfinite(vv)):
                continue
            xv = grid_coord(i, nx, ex0, ex1)
            yv = grid_coord(j, ny, ey0, ey1)
            px, py = data_to_pixel(axis, xv, yv)
            px2, py2 = data_to_pixel(axis, xv + uv * scale, yv + vv * scale)
            if not all(math.isfinite(t) for t in (px, py, px2, py2)):
                continue
            canvas.line(px, py, px2, py2, color, width=1)
            ang = math.atan2(py2 - py, px2 - px)
            head = 4.0
            canvas.fill_polygon(
                [
                    (px2, py2),
                    (px2 - head * math.cos(ang - 0.4), py2 - head * math.sin(ang - 0.4)),
                    (px2 - head * math.cos(ang + 0.4), py2 - head * math.sin(ang + 0.4)),
                ],
                color,
            )
    canvas.pop_clip()


def _draw_loglogfit_layer(canvas: Canvas, axis: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
    color = layer.get("color") or "#2b6cb0"
    theory_color = layer.get("theoryColor") or "#c53030"
    hs = np.asarray(layer.get("h"), dtype=np.float64).reshape(-1)
    errs = np.asarray(layer.get("err"), dtype=np.float64).reshape(-1)
    canvas.push_clip(*[float(v) for v in axis["rect"]])
    path = _clip_series_to_axis(axis, layer_to_pixel_path(axis, hs, errs))
    _stroke_path(canvas, path, color, 1.6, None, 1.0)
    for i in range(0, len(path), 2):
        px, py = path[i], path[i + 1]
        if px is None or py is None:
            continue
        canvas.fill_circle(px, py, 2.5, color)
    anchor = None
    for i in range(len(path) - 2, -1, -2):
        if path[i] is not None and path[i + 1] is not None:
            anchor = (path[i], path[i + 1])
            break
    theory_slope = float(layer.get("theorySlope", 0.0))
    ref_x = layer.get("refX") or [0.0, 0.0]
    x0, x1 = float(ref_x[0]), float(ref_x[1])
    if anchor is not None and x0 > 0 and x1 > 0 and errs.size:
        last_err = float(errs[-1])
        if last_err > 0:
            log_span = math.log10(x1) - math.log10(x0)
            theory_path: list[float | None] = []
            s = 0.0
            while s <= 1.0000001:
                lx = math.log10(x0) + s * log_span
                ly = math.log10(last_err) + theory_slope * (lx - math.log10(x1))
                px, py = data_to_pixel(axis, math.pow(10, lx), math.pow(10, ly))
                if math.isfinite(px) and math.isfinite(py):
                    theory_path.extend((px, py))
                else:
                    theory_path.extend((None, None))
                s += 0.02
            _stroke_path(canvas, theory_path, theory_color, 1.3, [5, 4], 1.0)
    rx, ry, rw, _rh = (float(t) for t in axis["rect"])
    y_top = ry + 14
    canvas.text(rx + rw - 8, y_top, str(layer.get("seriesLabel", "")), _BASE_FONT_SIZE - 1, color, align="right", baseline="top")
    canvas.text(rx + rw - 8, y_top + _BASE_FONT_SIZE + 2, str(layer.get("theoryLabel", "")), _BASE_FONT_SIZE - 1, theory_color, align="right", baseline="top")
    canvas.pop_clip()


def _scene_text_boxes(scene: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    """场景里所有文本层的估算盒子（给"渲染器自己画的标签"避让用）。"""
    return [text_box_of(t) for t in (scene.get("texts") or []) if isinstance(t, Mapping)]


def _draw_layer(canvas: Canvas, scene: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
    axis = next((a for a in scene.get("axes", []) if a.get("id") == layer.get("axes")), None)
    if axis is None:
        return
    kind = layer.get("kind")
    if kind == "line":
        _draw_line_layer(canvas, axis, layer, avoid=_scene_text_boxes(scene))
    elif kind == "band":
        _draw_band_layer(canvas, axis, layer)
    elif kind == "scatter":
        _draw_scatter_layer(canvas, axis, layer)
    elif kind == "heat":
        _draw_heat_layer(canvas, axis, layer)
    elif kind == "contour":
        _draw_contour_layer(canvas, axis, layer)
    elif kind == "bars":
        _draw_bars_layer(canvas, axis, layer)
    elif kind == "vectorfield":
        _draw_vectorfield_layer(canvas, axis, layer)
    elif kind == "loglogfit":
        _draw_loglogfit_layer(canvas, axis, layer)


def _draw_colorbar(
    canvas: Canvas,
    axis: Mapping[str, Any],
    layer: Mapping[str, Any],
    *,
    font_size: float = _BASE_FONT_SIZE,
) -> None:
    """色标条：与 ``render2d.ts::drawColorbar`` 同一算法。

    **Scene 契约里没有色标节点**（§6 的 Layer 只带 cmap/vmin/vmax/logScale/label），
    因此这里约定：每个 heat 图层在其轴右侧自动配一条色标（对数色标用几何中点
    ``sqrt(vmin·vmax)`` 作中间标签）。这是加法，不改契约。
    """
    rx, ry, _rw, rh = (float(v) for v in axis["rect"])
    x = rx + _rw + 46.0
    y = ry + 10.0
    w = 14.0
    h = max(20.0, rh - 60.0)
    cmap = str(layer.get("cmap", "viridis"))
    vmin, vmax = float(layer["vmin"]), float(layer["vmax"])
    log_scale = bool(layer.get("logScale"))
    table = cmap_table(cmap)
    steps = max(2, _round_half_up(h * canvas.scale))
    band = h / steps
    for i in range(steps):
        t = 1.0 - i / (steps - 1)
        idx = min(_LEVELS - 1, max(0, _round_half_up(t * (_LEVELS - 1))))
        r, g, b = (int(table[idx, 0]), int(table[idx, 1]), int(table[idx, 2]))
        # +1 设备像素的重叠：避免相邻色带之间出现背景色缝（canvas 侧是 fillRect 的 +1）
        canvas.fill_rect(x, y + i * band, w, band + 1.0 / canvas.scale, f"rgb({r},{g},{b})")
    canvas.stroke_rect(x, y, w, h, _FRAME_COLOR, thickness=1)
    labels = [vmax, math.sqrt(vmin * vmax), vmin] if log_scale else [vmax, (vmin + vmax) / 2.0, vmin]
    for i, v in enumerate(labels):
        canvas.text(x + w + 4, y + (i * h) / (len(labels) - 1), format_number(float(v), 4), font_size - 1, _DEFAULT_FONT_COLOR, align="left", baseline="middle")
    if layer.get("label"):
        canvas.text(x + w / 2, y + h + 4, str(layer["label"]), font_size - 1, _DEFAULT_FONT_COLOR, align="center", baseline="top")


def _draw_badge(canvas: Canvas, badge: Mapping[str, Any], opts: RenderOptions) -> None:
    size = opts.base_font_size - 2
    pad = 5.0
    width = _font().measure(str(badge.get("text", "")), size) + 2 * pad
    height = opts.base_font_size + 4
    bx, by = float(badge.get("x", 8)), float(badge.get("y", 8))
    canvas.fill_rect(bx, by, width, height, "rgba(255,255,255,0.86)")
    color = badge.get("color") or _BADGE_BORDER
    canvas.stroke_rect(bx, by, width, height, color, thickness=1)
    canvas.text(bx + pad, by + height / 2.0, str(badge.get("text", "")), size, color, align="left", baseline="middle", bold=True)


def _draw_pitfalls(canvas: Canvas, scene: Mapping[str, Any], opts: RenderOptions) -> None:
    """底部 pitfalls 面板（规则 7：它会怎样骗你，必须随图出现）。"""
    size = opts.base_font_size - 1
    width = float(scene["canvas"]["w"])
    height = float(scene["canvas"]["h"])
    lines = _font().wrap(f"它会怎样骗你：{scene.get('pitfalls', '')}", size, width - 24)
    line_h = size + 4
    y0 = height - 10 - len(lines) * line_h
    canvas.fill_rect(8, y0 - 4, width - 16, len(lines) * line_h + 8, "rgba(255,247,237,0.95)")
    canvas.stroke_rect(8, y0 - 4, width - 16, len(lines) * line_h + 8, "#dd6b20", thickness=1)
    for i, line in enumerate(lines):
        canvas.text(14, y0 + i * line_h, line, size, "#7b341e", align="left", baseline="top")


def render_scene(
    scene: Mapping[str, Any],
    *,
    scale: float = 1.0,
    validate: bool = True,
    options: RenderOptions | None = None,
) -> PngImage:
    """把 Scene 光栅化成 :class:`~tools.png.PngImage`（RGBA，8 位/通道）。

    ``scale`` 等价于浏览器里的 devicePixelRatio：画布按比例放大、所有几何同步缩放
    （文字按目标字号选最近的一档位图）。
    ``validate=True``（缺省）会先跑 :func:`validate_scene`——宁可当场报错，也不要悄悄画出一张
    与契约不符的图（TS 侧把校验放在 ``assertValidScene``，本实现把它放在入口，理由见 docstring）。
    """
    if validate:
        assert_valid_scene(scene)
    opts = options or RenderOptions()
    canvas_spec = scene["canvas"]
    canvas = Canvas(float(canvas_spec["w"]), float(canvas_spec["h"]), scale=scale, background=canvas_spec.get("background", "#ffffff"))

    if scene.get("title"):
        canvas.text(12, 8, str(scene["title"]), opts.base_font_size + 5, _TITLE_COLOR, align="left", baseline="top", bold=True)
    for axis in scene.get("axes", []):
        _draw_axis(canvas, axis, opts)
    for layer in scene.get("layers", []):
        _draw_layer(canvas, scene, layer)
    for layer in scene.get("layers", []):
        if layer.get("kind") == "heat":
            axis = next((a for a in scene.get("axes", []) if a.get("id") == layer.get("axes")), None)
            if axis is not None:
                _draw_colorbar(canvas, axis, layer, font_size=opts.base_font_size)
    for txt in scene.get("texts", []):
        canvas.text(
            float(txt.get("x", 0)),
            float(txt.get("y", 0)),
            str(txt.get("text", "")),
            float(txt.get("size", 12)),
            txt.get("color", _DEFAULT_FONT_COLOR),
            align=str(txt.get("align", "left")),
            baseline="top",
            bold=txt.get("role") == "title",
            max_width=txt.get("maxWidth"),
        )
    if opts.show_badges:
        for badge in scene.get("badges", []):
            _draw_badge(canvas, badge, opts)
    _draw_pitfalls(canvas, scene, opts)
    return canvas.to_png_image()


def render_scene_to_bytes(scene: Mapping[str, Any], *, scale: float = 1.0, validate: bool = True) -> bytes:
    """``render_scene`` + PNG 编码（无时间戳 → 同输入必然同字节）。"""
    image = render_scene(scene, scale=scale, validate=validate)
    assert image.channels == 4
    arr = np.frombuffer(image.pixels, dtype=np.uint8).reshape(image.height, image.width, 4)
    return png_bytes(arr, compress_level=6)


def render_scene_to_file(path: str | pathlib.Path, scene: Mapping[str, Any], *, scale: float = 1.0, validate: bool = True) -> pathlib.Path:
    """把 Scene 写成 PNG 文件（父目录自动创建）。"""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render_scene_to_bytes(scene, scale=scale, validate=validate))
    return out


# =========================================================================== #
# 8. 帧 → "标准探针场景"
# =========================================================================== #
_COORD_NAMES = ("x", "t", "r", "h", "z", "xi", "eta")
_AUX_NAMES = ("error", "residual")
_AXIS_COLORS = ("#2b6cb0", "#dd6b20", "#2f855a", "#805ad5")
_MAIN_COLOR = "#2b6cb0"
_AUX_COLOR = "#c53030"


def _frame_array(frame: Any, name: str) -> np.ndarray:
    return np.asarray(frame[name], dtype=np.float64)


def _finite_range(values: np.ndarray) -> tuple[float, float] | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.min(finite)), float(np.max(finite))


def _positive_range(values: np.ndarray) -> tuple[float, float] | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        return None
    return float(np.min(finite)), float(np.max(finite))


def _finite_pairs(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float], int]:
    """丢掉非有限的 (x, y) 点，返回 (x, y, 丢弃个数)。

    为什么要丢而不是留着：``validateScene``（TS 与 Python 同一套规则）把"折线里出现非有限
    纵坐标"判为契约违规——折线的断线语义是渲染层的**防御**，模型层必须先裁掉。
    探针场景必须自己合法，否则它会以"场景不合法"的名义把回归门禁弄红。
    """
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(xa.size, ya.size)
    keep = np.isfinite(xa[:n]) & np.isfinite(ya[:n])
    return xa[:n][keep].tolist(), ya[:n][keep].tolist(), int(n - np.count_nonzero(keep))


def _axis_spec(manifest_viz: Mapping[str, Any], index: int, fallback: Mapping[str, Any]) -> dict[str, Any]:
    axes = manifest_viz.get("axes") or []
    if index < len(axes) and isinstance(axes[index], Mapping):
        merged = dict(fallback)
        merged.update({k: v for k, v in axes[index].items() if v not in (None, "")})
        return merged
    return dict(fallback)


def default_scene_for_frame(
    manifest_viz: Mapping[str, Any],
    frame: Any,
    *,
    width: float = 960.0,
    height: float = 720.0,
    tick_target: int = 7,
    t: float | None = None,
) -> dict[str, Any]:
    """把"一帧 + manifest"折成一个固定的**回归探针场景**（**不是教学图**）。

    为什么需要它：教学图的 Scene 由章节前端 ``*.model.ts`` 生成，Python 侧拿不到 TS 的 model
    （``docs/PROTOCOLS.md`` §6 只规定 Scene 是两侧共用的中间表示，没规定 Python 能造出同一个
    教学场景）。于是视觉回归约定：**输入是帧 + manifest**，由本函数生成一个覆盖全部装饰的
    标准场景——主图层画 requiredAux 之外的第一条主曲线（有第二条就再画一条虚线）、
    副轴画 error/residual、徽标四类齐全（source/autoscale/critical，溢出时再加 overflow）、
    底部 pitfalls 面板。整段逻辑**只有这一份实现**（``tools/visual_regression.py`` 与
    ``tests/test_raster.py`` 都调它），这样"探针"本身也是可回归的。

    布局是固定的（960×720，主轴上、误差轴下、右侧 meta 文本），不随章节变化；
    它只服务于"图会不会变"这一个问题，不承担教学表达。

    **它绝不产出非法 Scene**：折线里的非有限点（NaN/±Inf）在这里就被丢掉
    （见 :func:`_finite_pairs`），丢弃个数写进 meta 文本面板——因为 ``validateScene``
    把"折线含非有限纵坐标"判为契约违规，探针场景自己必须先合法；折线的断线语义
    仍由渲染层防御性支持（见模块 docstring 的红线）。
    """
    names = list(frame.names)
    if not names:
        raise RasterError("帧里没有任何数组，无法构造探针场景")

    required_aux = [str(a) for a in (manifest_viz.get("requiredAux") or [])]
    aux_name = None
    for want in required_aux + list(_AUX_NAMES):
        aux_name = next((n for n in names if want.lower() in n.lower()), None)
        if aux_name:
            break

    coord_name = next((n for n in names if n.lower() in _COORD_NAMES and n != aux_name), None)
    if coord_name is None:
        coord_name = names[0]
    main_names = [n for n in names if n not in (coord_name, aux_name)]
    two_d = [n for n in main_names if _frame_array(frame, n).ndim == 2]
    one_d = [n for n in main_names if _frame_array(frame, n).ndim == 1]

    coord = _frame_array(frame, coord_name)
    x_range = _finite_range(coord)
    if x_range is None:
        x_range = (0.0, 1.0)
    xlim = nice_limits(x_range[0], x_range[1], tick_target, pad=0.02 if x_range[1] > x_range[0] else 0.0)
    if xlim[0] == xlim[1]:
        xlim = (xlim[0] - 1.0, xlim[1] + 1.0)

    main_rect = [80.0, 130.0, 560.0, 260.0]
    aux_rect = [80.0, 450.0, 560.0, 150.0]
    aux_log = str(_axis_spec(manifest_viz, 1, {}).get("scale", "log")) == "log"

    main_axis: dict[str, Any] = {
        "id": "main",
        "rect": main_rect,
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [0.0, 1.0],
        "xscale": "linear",
        "yscale": "linear",
        "xlabel": str(coord_name),
        "ylabel": str(_axis_spec(manifest_viz, 0, {}).get("ylabel", "")),
        "title": str(_axis_spec(manifest_viz, 0, {}).get("title", manifest_viz.get("title", ""))),
        "autoscaled": bool(_axis_spec(manifest_viz, 0, {}).get("autoscaled", False)),
        "xticks": [],
        "yticks": [],
        "grid": True,
        "frame": True,
    }
    main_axis["xticks"] = nice_ticks(main_axis["xlim"][0], main_axis["xlim"][1], tick_target)

    layers: list[dict[str, Any]] = []
    dropped_total = 0  # 非有限点一律丢弃（见 _finite_pairs），但必须在文本面板里显式说明
    # 二维场：优先画热力图（并因此自动获得色标），这样多数组的探针也能覆盖 heat 路径。
    if two_d:
        field = _frame_array(frame, two_d[0])
        ny, nx = int(field.shape[0]), int(field.shape[1])
        extent = [float(xlim[0]), float(xlim[1]), float(xlim[0]), float(xlim[1])]
        x_name = next((n for n in names if n.lower() == "x"), None)
        y_name = next((n for n in names if n.lower() == "y"), None)
        if x_name and y_name and _frame_array(frame, x_name).size == nx and _frame_array(frame, y_name).size == ny:
            extent = [
                float(np.nanmin(_frame_array(frame, x_name))), float(np.nanmax(_frame_array(frame, x_name))),
                float(np.nanmin(_frame_array(frame, y_name))), float(np.nanmax(_frame_array(frame, y_name))),
            ]
        stats = _finite_range(field)
        vmin, vmax = (stats if stats else (0.0, 1.0))
        if not (vmax > vmin):
            vmax = vmin + 1.0
        main_axis["xlim"] = [min(extent[0], extent[1]), max(extent[0], extent[1])]
        main_axis["ylim"] = [min(extent[2], extent[3]), max(extent[2], extent[3])]
        layers.append({
            "kind": "heat", "axes": "main", "nx": nx, "ny": ny,
            "values": field.reshape(-1).tolist(), "extent": extent, "cmap": "viridis",
            "vmin": vmin, "vmax": vmax, "logScale": False, "label": two_d[0],
        })
    else:
        y_range = None
        for name in one_d[:2]:
            rng = _finite_range(_frame_array(frame, name))
            if rng is None:
                continue
            y_range = rng if y_range is None else (min(y_range[0], rng[0]), max(y_range[1], rng[1]))
        if y_range is None:
            y_range = (0.0, 1.0)
        pad = 0.05 if y_range[1] > y_range[0] else 0.0
        ylim = nice_limits(y_range[0], y_range[1], tick_target, pad=pad)
        if ylim[0] == ylim[1]:
            ylim = (ylim[0] - 1.0, ylim[1] + 1.0)
        main_axis["ylim"] = [float(ylim[0]), float(ylim[1])]
        main_axis["yticks"] = nice_ticks(main_axis["ylim"][0], main_axis["ylim"][1], tick_target)
        series = []
        for i, name in enumerate(one_d[:2]):
            y = _frame_array(frame, name)
            n = min(y.size, coord.size)
            xs, ys, dropped = _finite_pairs(coord[:n], y[:n])
            dropped_total += dropped
            unit = frame.specs[names.index(name)].unit
            series.append({
                "x": xs,
                "y": ys,
                "color": _AXIS_COLORS[i % len(_AXIS_COLORS)],
                "width": 1.8,
                "label": f"{name} ({unit})" if unit else name,
                "dash": [6, 4] if i == 1 else None,
            })
        if series:
            layers.append({"kind": "line", "axes": "main", "series": series})
    if not main_axis["yticks"]:
        main_axis["yticks"] = nice_ticks(main_axis["ylim"][0], main_axis["ylim"][1], tick_target)

    axes: list[dict[str, Any]] = [main_axis]
    if aux_name is not None:
        aux_vals = _frame_array(frame, aux_name)
        pos = _positive_range(aux_vals)
        if aux_log and pos is None:
            aux_axis_spec = _axis_spec(manifest_viz, 1, {})
            aux_axis = {
                "id": "error",
                "rect": aux_rect,
                "xlim": list(main_axis["xlim"]),
                "ylim": [1e-16, 1.0],
                "xscale": "linear",
                "yscale": "log",
                "xlabel": str(coord_name),
                "ylabel": str(aux_axis_spec.get("ylabel", "‖误差‖")),
                "title": str(aux_axis_spec.get("title", "误差 / 残差")),
                "autoscaled": True,
                "xticks": nice_ticks(main_axis["xlim"][0], main_axis["xlim"][1], tick_target),
                "yticks": nice_ticks(1e-16, 1.0, tick_target, log=True),
                "grid": True,
                "frame": True,
            }
        else:
            if aux_log:
                ylim_aux = nice_limits(max(pos[0], _MIN_FLOAT), max(pos[1], pos[0] * 1.0000001), tick_target, log=True, pad=0.1)
                yscale = "log"
            else:
                rng = _finite_range(aux_vals) or (0.0, 1.0)
                ylim_aux = nice_limits(rng[0], rng[1], tick_target, pad=0.05 if rng[1] > rng[0] else 0.0)
                yscale = "linear"
                if ylim_aux[0] == ylim_aux[1]:
                    ylim_aux = (ylim_aux[0] - 1.0, ylim_aux[1] + 1.0)
            aux_axis_spec = _axis_spec(manifest_viz, 1, {})
            aux_axis = {
                "id": "error",
                "rect": aux_rect,
                "xlim": list(main_axis["xlim"]),
                "ylim": [float(ylim_aux[0]), float(ylim_aux[1])],
                "xscale": "linear",
                "yscale": yscale,
                "xlabel": str(coord_name),
                "ylabel": str(aux_axis_spec.get("ylabel", "‖误差‖")),
                "title": str(aux_axis_spec.get("title", "误差 / 残差")),
                "autoscaled": True,
                "xticks": nice_ticks(main_axis["xlim"][0], main_axis["xlim"][1], tick_target),
                "yticks": nice_ticks(float(ylim_aux[0]), float(ylim_aux[1]), tick_target, log=yscale == "log"),
                "grid": True,
                "frame": True,
            }
        n = min(aux_vals.size, coord.size)
        aux_y = np.abs(aux_vals[:n]) if aux_log else aux_vals[:n]
        aux_x, aux_y_kept, dropped = _finite_pairs(coord[:n], aux_y)
        dropped_total += dropped
        layers.append({
            "kind": "line",
            "axes": "error",
            "series": [{
                "x": aux_x,
                "y": aux_y_kept,
                "color": _AUX_COLOR,
                "width": 1.6,
                "label": f"{aux_name}（规则 3）",
            }],
        })
        axes.append(aux_axis)

    data_source = str(frame.strings.get("data_source") or manifest_viz.get("dataSource") or "backend")
    badges: list[dict[str, Any]] = [
        {
            "kind": "preview" if data_source.startswith("backend+preview") else "source",
            "text": ("数据：后端帧 + 前端近似预览" if data_source.startswith("backend+preview") else "数据：后端 .nlf 帧"),
            "x": 10.0, "y": 34.0, "color": "#2b6cb0",
        },
        {
            "kind": "autoscale",
            "text": "坐标轴：自适应（可能掩盖发散）" if _any_autoscaled(axes) else "坐标轴：固定",
            "x": 10.0, "y": 52.0,
            "color": "#b7791f" if _any_autoscaled(axes) else "#4a5568",
        },
    ]
    crit_badge = _critical_badge(frame, manifest_viz, x=10.0, y=70.0)
    if crit_badge is not None:
        badges.append(crit_badge)

    texts: list[dict[str, Any]] = []
    meta_lines = [f"{k} = {format_number(float(v), 4)}" for k, v in sorted(frame.meta.items())
                  if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if dropped_total:
        meta_lines.append(f"已丢弃非有限点：{dropped_total}（探针场景不产生非法 Scene）")
    for key in sorted(frame.strings):
        if key in ("params_json", "data_source"):
            continue
        meta_lines.append(f"{key} = {frame.strings[key]}")
    params_json = frame.strings.get("params_json")
    if params_json:
        try:
            import json as _json

            params = _json.loads(params_json)
            meta_lines.append("参数：" + ", ".join(f"{k}={v}" for k, v in sorted(params.items())))
        except Exception:  # pragma: no cover - params_json 由 build_frame 保证合法
            pass
    texts.append({
        "x": 680.0, "y": 140.0, "text": "\n".join(meta_lines[:14]), "size": 12.0,
        "color": "#2d3748", "align": "left", "role": "note", "maxWidth": 260.0,
    })

    scene: dict[str, Any] = {
        "version": 1,
        "chapter": str(frame.chapter),
        "viz": str(frame.viz),
        "title": f"{frame.chapter} · {manifest_viz.get('title', frame.viz)}",
        "t": float(frame.meta.get("t", 0.0) if t is None else t),
        "canvas": {"w": float(width), "h": float(height), "background": "#ffffff"},
        "axes": axes,
        "layers": layers,
        "texts": texts,
        "badges": badges,
        "overflow": {"flag": False, "note": ""},
        "pitfalls": str(frame.pitfalls or manifest_viz.get("pitfalls") or "（manifest 未给出 pitfalls）"),
        "dataSource": data_source,
    }
    report = check_overflow(scene)
    scene["overflow"] = {"flag": report["flag"], "note": report["note"]}
    if report["flag"]:
        badges.append({"kind": "overflow", "text": f"溢出：{report['note']}", "x": 10.0, "y": 88.0, "color": "#c53030"})
    # 探针场景也要守规则 8：导出工具会用 `--width 200` 这类窄画布跑它，而角标文案长度
    # 由帧里的数值决定（临界值/溢出说明可能很长）。这里按同一把尺子裁到画布宽度预算内。
    budget = badge_unit_budget(width)
    for b in badges:
        b["text"] = clip_badge_text(str(b.get("text", "")), budget)
    return scene


def _any_autoscaled(axes: Sequence[Mapping[str, Any]]) -> bool:
    """规则 4 的口径：只要有一个轴是自适应的，就按"自适应"提示。"""
    return any(bool(a.get("autoscaled")) for a in axes)


def _critical_badge(frame: Any, manifest_viz: Mapping[str, Any], *, x: float, y: float) -> dict[str, Any] | None:
    """规则 6：把"当前参数值 vs 理论临界值"打成徽标；没有临界值时也显式说明。"""
    params: dict[str, Any] = {}
    raw = frame.strings.get("params_json")
    if raw:
        try:
            import json as _json

            params = _json.loads(raw)
        except Exception:  # pragma: no cover
            params = {}
    specs = {str(p.get("name")): p for p in (manifest_viz.get("params") or [])}
    critical = dict(frame.critical or {})
    if not critical:
        for name, spec in specs.items():
            if spec.get("critical"):
                critical[name] = float(spec["critical"][0])
    if not critical:
        return None
    name = sorted(critical)[0]
    spec = specs.get(name, {})
    values = [float(v) for v in (spec.get("critical") or [critical[name]])]
    current = params.get(name, spec.get("default"))
    shown = " / ".join((f"{_precision_str(v, 4)}*" if v == critical[name] else _precision_str(v, 4)) for v in values)
    beyond = str(spec.get("beyond", "above"))
    crossed = False
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        cur = float(current)
        if beyond == "above":
            crossed = cur > float(critical[name])
        elif beyond == "below":
            crossed = cur < float(critical[name])
        else:
            crossed = abs(cur) > abs(float(critical[name]))
        body = f"{name} = {_precision_str(cur, 4)}　理论临界 {shown}"
    else:
        body = f"{name} = ?　理论临界 {shown}"
    if crossed:
        body += "　⚠ 已越界"
    return {"kind": "critical", "text": body, "x": x, "y": y, "color": "#c53030" if crossed else "#2f855a"}


# =========================================================================== #
# 9. 内嵌字体数据（base64(zlib(位图))，来历见 §5 注释）
# =========================================================================== #
_FONT_BLOB_B64 = (
    "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"
)
