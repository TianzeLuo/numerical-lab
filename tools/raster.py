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
)
