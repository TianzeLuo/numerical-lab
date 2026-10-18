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


def _draw_line_layer(
    canvas: Canvas,
    axis: Mapping[str, Any],
    layer: Mapping[str, Any],
    *,
    avoid: Sequence[tuple[float, float, float, float]] = (),
) -> None:
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
