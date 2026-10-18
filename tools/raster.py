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


def _text_deco_issues(scene: Mapping[str, Any]) -> list[str]:
    """规则 8（布局·其二）：文本层不得压住任何坐标轴的装饰带。

    那里由**渲染层自己**画刻度与轴名，模型看不见、也移不开——所以只能由模型让路：
    给轴下留出 ``28px``（刻度标签 ``bottom+4..15``、轴名 ``bottom+16..28``）再放自己的文本，
    或者干脆把该轴的 ``xlabel`` 置空、改用手写类别名（见 ``spy_ordering`` / ``lu_heatmap``）。
    """
    issues: list[str] = []
    texts = [t for t in (scene.get("texts") or []) if isinstance(t, Mapping)]
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
        """按等宽宽度换行（同 render2d.ts::wrapText 的"塞不下就断，断点字符进下一行"）。"""
        out: list[str] = []
        line = ""
        for ch in text:
            probe = line + ch
            if max_width > 0 and self.measure(probe, size) > max_width and line:
                out.append(line)
                line = ch
            else:
                line = probe
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
        line_h = cell_h + 4 if max_width else cell_h
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
        total = font.measure(body, size)
        cursor = self.px(y) + _round_half_up(total / 2.0)
        x0 = self.px(x)
        offsets = (0, 1) if bold else (0,)
        for ch in body:
            rot, adv, _cell_h = font.rotated_offsets(ch, size)
            cursor -= max(1, _round_half_up(adv * self.scale))
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
            canvas.text_vertical(rx - 42, ry + rh / 2, str(axis["ylabel"]), opts.base_font_size, _DEFAULT_FONT_COLOR, bold=True)


def _draw_line_layer(canvas: Canvas, axis: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
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
                canvas.text(last[0] + 4, last[1] - 2, str(s["label"]), _BASE_FONT_SIZE - 1, color, align="left", baseline="bottom")


def _draw_band_layer(canvas: Canvas, axis: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
    band = layer.get("band") or {}
    xs = np.asarray(band.get("x"), dtype=np.float64).reshape(-1)
    ylo = np.asarray(band.get("ylo"), dtype=np.float64).reshape(-1)
    yhi = np.asarray(band.get("yhi"), dtype=np.float64).reshape(-1)
    n = min(xs.size, ylo.size, yhi.size)
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


def _draw_scatter_layer(canvas: Canvas, axis: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
    xs = np.asarray(layer.get("x"), dtype=np.float64).reshape(-1)
    ys = np.asarray(layer.get("y"), dtype=np.float64).reshape(-1)
    r = float(layer.get("r", 2.5))
    for i in range(min(xs.size, ys.size)):
        xv, yv = float(xs[i]), float(ys[i])
        if not (math.isfinite(xv) and math.isfinite(yv)):
            continue
        px, py = data_to_pixel(axis, xv, yv)
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        canvas.fill_circle(px, py, r, layer.get("color", "#4c78a8"), alpha=float(layer.get("alpha", 1.0)))


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


def _draw_layer(canvas: Canvas, scene: Mapping[str, Any], layer: Mapping[str, Any]) -> None:
    axis = next((a for a in scene.get("axes", []) if a.get("id") == layer.get("axes")), None)
    if axis is None:
        return
    kind = layer.get("kind")
    if kind == "line":
        _draw_line_layer(canvas, axis, layer)
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
    "k73eT7Z3+5HZJ3vItwVfdMNfC2xY27fqHn22aZthmdvsc7Y985Ljuy6nK+Nc9pwiadmZcTEObG10vv6m3XJp6fa87tBHCrQA"
    "+szjsSCn7O3UQGu6AcFhndc/eeAVuWz9++a22jCvOr5rjFYjga5+Fya/kgWZs+ffghLCkLvHTFifNZdNH95i138Yqbr+l82R"
    "gFPb+D394/F4RjCPt7ZLk46XCCS9Ro5PgT5b3pQZSNHrb9X7VUOSNFmYn9g1fvfOFvpZ83ibvT6IQcT6GCnMCE7BpdnH+9vb"
    "68OBFZFcOdDmJrxjm4Kl22FhFuz1Ke+b2uT5Dnv+ePb8rOurrr90vQBpeb/2PmjHfwrSdyGdBuyoaxoSlmu3q9vCPLTXnnfb"
    "JNm9256fsdcf4z8LPfQpbNWMuX53/z78xvLH1lQ/3rlb/zmSNEiCbrs8LpeFeWS3fVvV9b6h5+lh9PFit/9nbfCO1zpe25c/"
    "2Mb7T+6vb1baJZ937cVz+livMnV95Lfa7ee7bXiSpdP5ghP377ugn3/guqxfax6LbfmD3KuQp0Ys7jbGujwdNn141N5ePJ6q"
    "+hUtYpX3tPex3XbWrVtD3ph9GcwF2qzue+yzXfE+0szp3+nH4/b8DQtV1mc4i7cJwEE4qg/vtc3ukKT7pBadw55ot7calol9"
    "fOw+nnzDTF7XFruoGjqH/zHH+/bhbQ6DpYDd0TUbOe5YX56zRhdNRTxqYIx5HNgagr45bPbwqb35lzDrMMfiBtgwm/1famd7"
    "ON2Delw2x9llB7Z65vH0vvXXuQy2uYxrFayVZndCbKsjimlf7zu+ntmbP9pTn3vAUziDp6zslY+sy6fq4KxkZfas9VEyyCuw"
    "tzY9atfn9/9YvMlbm44mQyKfbdb8s47LjhsOnes68DwPFtLKNG9pP8wR+46Vkp0k/dRe9rMFNX4B7LktWd8+vwf/+vL5wP79"
    "W72ffSvszX8f/aR9w9his+axQ1iokXD+O2BljGQL25rHD4DB/I1uI6xkHh/af3xVwaq0gCzj1rbV//wef1z3J+4r/w/vba9N"
    "jlCOoZ/m8cLe/iJW11HZz/6/uAdP5oBlfY4/67gNJVTtRnZv/qZFxzNAvd3evXTf+jBGfQqUoTvAS2ds4eg21sTaZoA2vvRR"
    "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"
    "HfX/D2YWANwRwnc="
)
