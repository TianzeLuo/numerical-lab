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
    "Grd2d47C8s/z9W6P8FxcFsVA=="
)
