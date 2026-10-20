"""1.1 章：浮点数、误差传播与问题条件。

四个可视化沿一条主线排开：**浮点网格长什么样 → 误差怎么被放大 → 问题有多难 → 算法能救多少**。

1. ``float_zoom`` —— 数轴放大镜。每个圆点是一个真实存在的 double，横轴是它与窗口中心的
   距离（单位 ulp）；可切到 0.3 或 2 的幂 0.25，直接看 ``0.1+0.2`` 的落点、相邻间距与
   跨过 2 的幂时格点变稀。下半图是"``0.1`` 累加 k 次"的残差演化。
2. ``cancellation`` —— 灾难性抵消。拖 ``a``、``b`` 相互靠近，主轴给"**剩余有效位数的整数计数**"，
   副轴给"输入扰动被放大后的相对误差"爆炸曲线；同时画出**减法本身**的舍入误差（恒为 0，
   Sterbenz 引理）——两条曲线的对比就是这一章最容易被误读的地方。
3. ``cond_ellipsoid`` —— 条件数椭球。单位圆在 :math:`A` 下的像，半轴之比就是 :math:`\\kappa_2`；
   拖右端项扰动 :math:`\\delta b` 看解 :math:`x` 跳多远，副轴给**实测放大比 vs 理论上界** :math:`\\kappa`。
4. ``summation_order`` —— 求和顺序。``1e16 + 1 + 1 + ...`` 上朴素 / Kahan / 成对 / 排序四种顺序的
   误差随项数**同步演化**，副轴（对数）同时给出 :math:`n u\\sum|x|` 与 :math:`2u\\sum|x|` 两条理论界。

约定（与 0.0 / 1.2 章一致）：``meta`` 只放数字/布尔/``None``，字符串一律走 ``strings``；
数组一律 ``fr.add(...)``。**所有**对数轴都有一个显式的显示地板（见各帧的 ``*_floor`` meta 与
数组标签里的"以 …… 显示"），因为"误差恰好为 0"在 log 轴上画不出来——把它悄悄丢掉，
就会把"精确"画成"没有数据"。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from numcore import fp
from numcore.linalg import cond
from numcore.linalg.lu import backward_error, exact_solve, lu_partial, residual_norm, solve_lu
from numcore.utils import rng
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 单位舍入 u = 2^-53（与 :func:`numcore.fp.unit_roundoff` 同一口径）
U = fp.unit_roundoff()

#: ``float_zoom`` 的时间轴上限：0.1 累加的最大次数
ACC_COUNT = 64

#: ``cancellation`` 的损失位数上限（= log10(最大可观测量级比)）
LOSS_MAX = 16.0

#: 对数轴上的相对误差地板：小于它的量按 u/2 显示
ERROR_FLOOR = U / 2.0

#: 椭球主轴的固定半宽（与 1.2 章 unit_ball 同一约定：不重要，但不许自适应）
ELLIPSE_HALF_WIDTH = 3.0


def _step_of(t: float, n_max: int) -> int:
    """把连续时间轴四舍五入到整数步并夹到 ``[0, n_max]``。"""
    k = int(np.floor(float(t) + 0.5))
    return max(0, min(k, max(0, int(n_max))))


def _scalar(x: Any) -> Any:
    """meta 只接受有限数 / bool / None：``nan``/``inf`` 一律写成 ``None``。

    为什么必须这样：``server.nlf.pack_frame`` 用 ``allow_nan=False`` 序列化头部
    （JSON 规范里没有 NaN），一旦把 NaN 塞进 meta，帧在**写盘那一刻**才会炸，
    而且报错信息是 ``Out of range float values are not JSON compliant``——
    与"哪个量没定义"毫无关系。所以"没定义"要在装配时就说清楚。
    """
    if x is None or isinstance(x, bool):
        return x
    v = float(x)
    return v if np.isfinite(v) else None


def _at_least_one(t: float, n_max: int) -> int:
    """同上，但下限是 1（"部分和长度 k"这类必须有意义的量）。"""
    return max(1, _step_of(t, n_max))


# =========================================================================== #
# MANIFEST：四个可视化的契约（真源）
# =========================================================================== #
MANIFEST: dict[str, Any] = {
    "chapter": "1.1",
    "title": "浮点数、误差传播与问题条件",
    "part": "I",
    "summary": (
        "从浮点网格的间距，到抵消放大的有效位损失，再到 κ(A) 决定的前向误差界："
        "把「精度」「稳定性」「条件数」这三件常被混为一谈的事分开，并让每件都能被测量。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "float_zoom",
            "title": "0.1+0.2 为什么不是 0.3：把数轴放大到最后一个二进制位",
            "description": (
                "在 0.3 附近，计算机能表示的数不是一条连续的线，而是一串间距相同的格点。"
                "这段间距叫一个 ulp（最后一位的单位），double 在 0.3 附近约为 $5.55\\times10^{-17}$。"
                "上图的每个圆点就是一个真实存在的 double，横轴的单位是“窗口中心处的一个 ulp”，"
                "括号标出相邻两个格点的真实间距；纵轴只有一行，只用来把格点排开，不表示数值大小。"
                "所谓“0.3”在 double 里其实存成 $0.2999999999999999889\\ldots$（蓝点）；"
                "$0.1+0.2$ 存成 $0.3000000000000000444\\ldots$（红点），两者正好相邻，所以打印出来只差最后一位。"
                "把观察位置切到 0.25，还能看到它作为 2 的幂边界，左右两侧的格点疏密不同。"
                "悬停任意圆点可以读出它完整的 17 位十进制值。"
                "下图把 0.1 连续加 $k$ 次，画出每一步留下的误差。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "focus",
                    "label": "观察位置",
                    "type": "enum",
                    "options": ["0.3（加法）", "0.25（2 的幂）"],
                    "default": "0.3（加法）",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "window_ulps",
                    "label": "横向半宽（以窗口中心的 ulp 计）",
                    "type": "int",
                    "min": 3,
                    "max": 32,
                    "step": 1,
                    "default": 8,
                    "unit": "ulp",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 64.0, "dt": 1.0, "label": "0.1 的累加次数 k"},
            "requiredAux": ["residual"],
            "axes": [
                {
                    "id": "ladder",
                    "autoscaled": False,
                    "title": "浮点格点：横轴是与窗口中心的距离，每个圆点是一个可表示的 double",
                    "ylabel": "浮点格点（示意行）",
                },
                {
                    "id": "residual",
                    "autoscaled": True,
                    "title": "把 0.1 累加 k 次后，结果与数学值 k/10 的差（对数轴）",
                    "ylabel": "绝对误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "圆点之间的横向距离才是真实间距；纵轴只是一行示意，不表示大小。"
                "图里只展示窗口内的几十个 double，窗口外还有无数个——看不见不等于没有。"
                "0.1+0.2 与 0.3 相差一个 ulp，是二进制舍入的必然结果，不是加法器坏了；"
                "0.1 和 0.2 自己都不能被 double 精确表示。"
                "下图是误差的对数轴：误差恰好为 0 的步会被抬到地板高度显示，不能读成“误差不为零”。"
            ),
            "theory": [
                "$\\mathrm{fl}(x)$ 表示离实数 $x$ 最近的 double；相邻两个 double 的距离记作 $\\mathrm{ulp}(x)$，在区间 $[2^e,2^{e+1})$ 内固定为 $2^{e-52}$。",
                "$\\mathrm{fl}(0.1)+\\mathrm{fl}(0.2)=0.30000000000000004$，而 $0.3$ 的最近 double 是 $0.2999999999999999889\\ldots$；两者相邻，差 $1$ 个 ulp。",
                "把 0.1 累加 $k$ 次的误差随 $k$ 线性增长，尺度约为 $k\\,u/2$，其中 $u=2^{-53}$ 是单位舍入。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "cancellation",
            "title": "两个相近数相减：有效数字为什么会消失",
            "description": (
                "输入 $a$ 和 $b$ 各自带着一点表示误差。当它们很接近时，$a-b$ 的绝对值很小，"
                "这点误差在结果里占的比例就会被放大——有效数字一位位消失。"
                "上图横轴是“抵消了多少个十进制位” $t=\\log_{10}(M/|a-b|)$，纵轴是相减后还剩多少位可信；"
                "下图用对数纵轴比较实测相对误差、理论上界，以及减法本身的舍入误差。"
                "关键区别：减法本身通常是精确的（Sterbenz 引理），被放大的是输入原本的误差。"
                "两个数可以直接键入，例如把 $b$ 设成 $0.9999999999999999$。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "a",
                    "label": "被减数 a",
                    "type": "float",
                    "min": 0.5,
                    "max": 2.0,
                    "step": 1e-6,
                    "default": 1.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "b",
                    "label": "减数 b（可键入更接近 a 的值）",
                    "type": "float",
                    "min": 0.5,
                    "max": 2.0,
                    "step": 1e-6,
                    "default": 0.999999,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "input_digits",
                    "label": "输入自带的十进制有效位数 d",
                    "type": "int",
                    "min": 2,
                    "max": 16,
                    "default": 16,
                    "critical": [15.95],
                    "criticalNote": (
                        "$\\log_{10}(2^{53})=15.95$ 是 binary64 能可靠表达的十进制位数上限；"
                        "把 d 调到它以下，图上的「剩余位数」与理论上界会同步变差。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "perturb_ulps",
                    "label": "输入扰动（ulp）",
                    "type": "int",
                    "min": 0,
                    "max": 8,
                    "default": 1,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "digits",
                    "autoscaled": False,
                    "title": "还剩多少位十进制有效数字（横轴 = 抵消掉多少位，记作 t）",
                    "ylabel": "剩余位数",
                },
                {
                    "id": "error",
                    "autoscaled": True,
                    "title": "实测相对误差与理论上界（横轴 = 损失位数 t）",
                    "ylabel": "相对误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "相减后误差变大，不代表减法算错了：两个相近数相减通常精确，真正被放大的是输入本来就带的误差。"
                "上图是“还剩几位有效数字”的先验估计，不是这一次计算的实测误差；具体误差看下图。"
                "把 b 调到与 a 完全相等时，程序会改用相邻的一个 ulp，避免取对数；这时标出的点是退化情况，不能当一般结论。"
            ),
            "theory": [
                "减法的条件数 $\\operatorname{cond}(a-b)=\\frac{|a|+|b|}{|a-b|}$；相对误差约被放大同样倍数。",
                "Sterbenz 引理：当 $b/2\\le a\\le 2b$ 时，$\\mathrm{fl}(a-b)$ 是精确的，误差全部来自输入。",
                "剩余有效位数约为 $\\log_{10}\\!\\left(|a-b|/(\\delta_{\\mathrm{in}}\\max(|a|,|b|))\\right)$；每抵消 10 倍，少一位。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "cond_ellipsoid",
            "title": "条件数 κ(A)：输入的小误差会被放大多少倍",
            "description": (
                "解方程 $Ax=b$ 时，条件数 $\\kappa(A)$ 衡量“$b$ 的小扰动会把解 $x$ 放大多少”。"
                "画面中单位圆经过 $A$ 变成椭圆，长轴与短轴之比就是 $\\kappa_2$；$\\kappa$ 越大，"
                "椭圆越扁，解对输入方向越敏感。拖动时间轴改变 $\\kappa$，观察右端项被扰动后解点的跳动；"
                "下图比较当前方向上的实测放大倍数与最坏情况的上界 $\\kappa$。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "family",
                    "label": "矩阵族",
                    "type": "enum",
                    "options": ["shear", "diagonal", "rotation", "graded", "triangular"],
                    "default": "shear",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "shear",
                    "label": "剪切强度 s（仅 shear 族）",
                    "type": "float",
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "default": 0.8,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "delta_b",
                    "label": "右端项相对扰动 ‖δb‖/‖b‖",
                    "type": "float",
                    "min": 1e-16,
                    "max": 1e-2,
                    "step": 1e-16,
                    "default": 1e-12,
                    "critical": [1.1e-16],
                    "criticalNote": (
                        "δb 低于单位舍入 u = 1.11e-16 时，扰动在 float64 里根本表示不出来，"
                        "实测放大比会退化成舍入噪声；滑块下限取 u 量级正是为了让这件事可见。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "b_angle",
                    "label": "右端项方向（度）",
                    "type": "float",
                    "min": 0.0,
                    "max": 360.0,
                    "step": 5.0,
                    "default": 30.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "norm",
                    "label": "范数",
                    "type": "enum",
                    "options": ["1", "2", "inf", "fro"],
                    "default": "2",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 16.0, "dt": 0.25, "label": "log10 κ（目标）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "geometry",
                    "autoscaled": False,
                    "title": "单位圆与 A 作用后的椭圆",
                    "ylabel": "x₂",
                },
                {
                    "id": "error",
                    "autoscaled": True,
                    "title": "实测放大比与最坏情况上界 κ(A)",
                    "ylabel": "放大比",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "$\\kappa(A)$ 是最坏情况的上界，不一定在某个具体方向上达到；实测放大比低于 $\\kappa$ 是正常的。"
                "$\\kappa$ 很大时，单位圆会被拉成很扁的椭圆，图中可能冲出显示范围；看不到不等于不存在，状态标签会说明。"
                "当 $\\kappa$ 接近 $1/u\\approx9\\times10^{15}$ 时，$A$ 在 double 中已经和奇异矩阵难以区分，"
                "后面的曲线主要由舍入噪声决定。"
            ),
            "theory": [
                "单位球在 $A$ 下的像，半轴是 $\\sigma_{\\min}$ 与 $\\sigma_{\\max}$，半轴之比为 $\\kappa_2(A)=\\|A\\|_2\\|A^{-1}\\|_2$。",
                "前向误差界：$\\frac{\\|\\Delta x\\|}{\\|x\\|}\\le\\kappa(A)\\frac{\\delta A+\\delta b}{1-\\kappa\\,\\delta A}$；本图取 $\\delta A=0$。",
                "$\\kappa_1,\\kappa_2,\\kappa_\\infty,\\kappa_F$ 可以相差一个 $n$ 因子，所以读条件数时必须同时看范数。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "summation_order",
            "title": "加法顺序为什么影响结果：朴素、Kahan、成对与排序",
            "description": (
                "同样一组数，换个加法顺序，结果可能不同。上图依次列出大项与小项；"
                "下图比较朴素求和、Kahan 补偿求和、成对求和与排序求和的误差。"
                "Kahan 把每一步的舍入损失记下来再加回去，所以通常更准；但把它分段并行、"
                "段间再用朴素加法合并，误差会重新变大。把小项调小、项数调多，差异最明显。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "big",
                    "label": "大项 big",
                    "type": "float",
                    "min": 1.0,
                    "max": 1e16,
                    "step": 1.0,
                    "default": 1e16,
                    "critical": [1.0 / U],
                    "criticalNote": (
                        "big > 1/u ≈ 9.0e15 时，1.0 这个「小项」已经落在 big 的半个 ulp 之内，"
                        "朴素求和会把它整项丢掉；这正是 1e16+1+1+… 的经典场景。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "small",
                    "label": "小项 small",
                    "type": "float",
                    "min": 1e-18,
                    "max": 1.0,
                    "step": 1e-3,
                    "default": 1.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_terms",
                    "label": "项数 n",
                    "type": "int",
                    "min": 8,
                    "max": 4096,
                    "default": 512,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "compensated",
                    "label": "补偿求和的归约方式",
                    "type": "enum",
                    "options": ["serial", "shuffled", "parallel8", "parallel64"],
                    "default": "serial",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 4096.0, "dt": 16.0, "label": "部分和长度 k"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "terms",
                    "autoscaled": True,
                    "title": "被相加的项（横轴 = 项序号 i；最大项在前）",
                    "ylabel": "xᵢ",
                },
                {
                    "id": "error",
                    "autoscaled": True,
                    "title": "各求和顺序的绝对误差与两条理论上界",
                    "ylabel": "绝对误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "其一：Kahan 求和不是“永远准确”，它只是补偿掉一部分舍入损失；"
                "误差仍会随项数增长，只是常数更小。其二：把 Kahan 分段并行、段间再用朴素加法合并，"
                "会重新丢掉小项，误差可能退回朴素求和的量级。其三：对数轴上恰好为零的误差会被地板抬起；"
                "判断方法是否更好，要看曲线相对理论上界的位置，而不是它离图底多近。"
            ),
            "theory": [
                "朴素从左到右相加：$|\\Delta|\\le n\\,u\\sum_i|x_i|$，误差随项数线性增长。",
                "Kahan 补偿：$|\\Delta|\\le\\left(2u+O(nu^2)\\right)\\sum_i|x_i|$；界与 $n$、与顺序都无关，但不能分段后再朴素合并。",
                "成对求和：$|\\Delta|\\le u\\lceil\\log_2 n\\rceil\\sum_i|x_i|$；排序后再朴素求和有同量级的界。",
                "并行分段补偿：误差退化为 $P\\,u\\sum|\\text{段和}|$，段与段之间那一次朴素加法是瓶颈。",
            ],
        },
    ],
}


# =========================================================================== #
# 计算入口
# =========================================================================== #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "float_zoom":
        return _float_zoom(params, t)
    if viz_id == "cancellation":
        return _cancellation(params, t)
    if viz_id == "cond_ellipsoid":
        return _cond_ellipsoid(params, t)
    if viz_id == "summation_order":
        return _summation_order(params, t)
    raise KeyError(f"章节 1.1 没有可视化 {viz_id!r}")


def _norm_key(norm: str) -> Any:
    return {"1": 1, "2": 2, "inf": "inf", "fro": "fro"}[str(norm)]


# =========================================================================== #
# viz 1：数轴放大镜
# =========================================================================== #
#: 观察位置 → (前端文案用的种类, 窗口中心)。两个中心都故意取十进制字面量：
#: ``0.25`` 恰是 2 的幂，``0.3`` 是 ``0.1+0.2`` 的数学值；后者最近的 double 并不是它本身。
_FLOAT_FOCUS: dict[str, tuple[str, float]] = {
    "0.3（加法）": ("sum", 0.3),
    "0.25（2 的幂）": ("boundary", 0.25),
}


def _float_zoom(params: Mapping[str, Any], t: float) -> Frame:
    focus = str(params["focus"])
    if focus not in _FLOAT_FOCUS:
        raise ValueError(f"未知观察位置 {focus!r}；可用：{', '.join(_FLOAT_FOCUS)}")
    kind, center = _FLOAT_FOCUS[focus]
    radius = max(3, min(32, int(params["window_ulps"])))
    sp_center = fp.spacing(center)

    # 横坐标不是"第几个 double"，而是**真实距离 / ulp(中心)**。于是 0.25 左側
    # （间距只有一半）的圆点会以 0.5 的间隔出现，右侧以 1 的间隔出现——格点疏密
    # 是直接量出来的，不靠两条"间距比"曲线去解释。
    candidates = fp.neighborhood(center, half=2 * radius + 4)
    rel = (candidates - center) / sp_center
    mask = np.abs(rel) <= radius + 1e-12
    positions = np.asarray(rel[mask], dtype=float)
    values = np.asarray(candidates[mask], dtype=float)

    prev_value = fp.next_float(center, direction=-1)
    next_value = fp.next_float(center, direction=1)
    spacing_center = sp_center
    spacing_below = fp.spacing(prev_value)
    spacing_above = fp.spacing(center)
    below_units = (center - prev_value) / sp_center
    above_units = (next_value - center) / sp_center

    special: float | None = None
    special_pos: float | None = None
    if kind == "sum":
        special = float(0.1) + float(0.2)
        special_pos = (special - center) / sp_center

    residual_0102_03 = float((float(0.1) + float(0.2)) - 0.3)

    # ---- 时间轴：0.1 累加 k 次的残差（精确值用有理数算） -------------------- #
    k = _step_of(t, ACC_COUNT)
    raw = fp.accumulation_residuals(0.1, ACC_COUNT, reference=(1, 10))
    raw_sum = fp.accumulation_residuals(0.1, ACC_COUNT)
    sums = fp.accumulation_values(0.1, ACC_COUNT)
    exact_k = np.arange(ACC_COUNT + 1, dtype=float) / 10.0
    residual_floor = (U / 2.0) * np.maximum(np.abs(exact_k), 0.1)
    residual = np.maximum(raw, residual_floor)
    residual_sum_only = np.maximum(raw_sum, residual_floor)
    n_exact_zero = int(np.count_nonzero(raw == 0.0))
    n_exact_zero_sum = int(np.count_nonzero(raw_sum == 0.0))

    fr = Frame(chapter="1.1", viz="float_zoom", kind="frame")
    fr.add("offset", positions, unit="ulp", label="距窗口中心的距离（单位：窗口中心的 ulp）")
    fr.add("value", values, unit="1", label="可表示的 double（精确值见悬停）")
    fr.add("center_offset", np.array([0.0]), unit="ulp", label="窗口中心")
    fr.add("center_y", np.array([0.5]), unit="1", label="窗口中心")
    if special is not None and special_pos is not None and abs(special_pos) <= radius + 1e-12:
        fr.add("special_offset", np.array([float(special_pos)]), unit="ulp", label="0.1+0.2 的实际值")
        fr.add("special_y", np.array([0.5]), unit="1", label="0.1+0.2 的实际值")
    fr.add("k_grid", np.arange(ACC_COUNT + 1, dtype=float), unit="1", label="累加次数 k")
    fr.add("residual", residual, unit="1", label="与数学值 k/10 的残差（精确 0 以 u/2·|k/10| 显示）")
    fr.add("residual_sum_only", residual_sum_only, unit="1", label="纯求和误差：与 k·fl(0.1) 的残差")
    fr.add("residual_floor", residual_floor, unit="1", label="对数轴的显示地板")
    fr.add("sum_value", sums, unit="1", label="fl(Σ0.1) 的部分和")
    fr.add("residual_mark", np.array([float(residual[k])]), unit="1", label="当前位置的残差")
    fr.add("k_mark", np.array([float(k)]), unit="1", label="当前位置的累加次数")
    fr.set_meta(
        center=center,
        focus_kind=0.0 if kind == "sum" else 1.0,
        window_ulps=float(radius),
        n_points=int(values.size),
        spacing_center=spacing_center,
        spacing_below=spacing_below,
        spacing_above=spacing_above,
        spacing_below_units=float(below_units),
        spacing_above_units=float(above_units),
        spacing_ratio=float(spacing_above / spacing_below),
        special_ulps=special_pos,
        residual_0102_03=residual_0102_03 if kind == "sum" else None,
        residual_0102_03_ulps=float(fp.residual_ulps(0.1 + 0.2, 0.3)) if kind == "sum" else None,
        residual_floor=float(residual_floor[k]),
        residual_sum_only_mark=float(residual_sum_only[k]),
        residual_floor_min=float(np.min(residual_floor)),
        residual_exact_zero_steps=n_exact_zero,
        residual_sum_only_exact_zero_steps=n_exact_zero_sum,
        k=k,
        n_acc_max=ACC_COUNT,
    )
    fr.set_strings(
        focus=kind,
        note=_zoom_note(kind, center, spacing_center, spacing_below, spacing_above, k, float(raw[k])),
        center_label="0.3 的最近 double" if kind == "sum" else "2 的幂边界 0.25",
        special_label="0.1+0.2 的实际值（大 1 个 ulp）" if kind == "sum" else "",
        exact_center=fp.exact_decimal(center, digits=24),
        exact_special=fp.exact_decimal(special, digits=24) if special is not None else "",
        spacing_note=(
            "横轴每一格是 ulp(窗口中心) 的长度；在同一个 [2^e,2^(e+1)) 内所有相邻 double 的距离相同。"
            if kind == "sum"
            else "0.25 是 2 的幂：它左侧的相邻间距是右侧的一半，所以左半边圆点更密；横轴仍以 ulp(0.25) 为单位。"
        ),
        zero_note=(
            f"下图是对数轴：残差恰好为 0 的步数（{n_exact_zero} 步）被地板 u/2·|k/10| 抬起显示，"
            "不能读成「误差不为零」。"
        ),
    )
    return fr


def _zoom_note(
    kind: str,
    center: float,
    sp: float,
    sp_below: float,
    sp_above: float,
    k: int,
    residual_k: float,
) -> str:
    tail = f"下图当前 k={k} 时，0.1 的累加结果与 k/10 相差 {residual_k:.3e}。"
    if kind == "boundary":
        return (
            f"0.25 是 2 的幂：左侧 ulp={sp_below:.3e}，右侧 ulp={sp_above:.3e}，"
            f"右侧间距是左侧的 {sp_above / sp_below:.2f} 倍。{tail}"
        )
    return (
        f"窗口中心 0.3 附近每个 double 相距 {sp:.3e}；橙色括号标出相邻两个 double 的一格距离。"
        f"0.1+0.2 存成中心右侧相邻的那个 double，两者正好相差 1 个 ulp。{tail}"
    )


# =========================================================================== #
# viz 2：灾难性抵消
# =========================================================================== #
def _input_uncertainty(input_digits: int) -> float:
    r"""``d`` 位十进制有效数字对应的相对不确定度 :math:`\delta_{\rm in}=\tfrac12 10^{1-d}`。"""
    d = int(input_digits)
    if d < 1:
        raise ValueError(f"input_digits 至少为 1，得到 {input_digits!r}")
    return 0.5 * 10.0 ** (1 - d)


def _cancellation(params: Mapping[str, Any], t: float) -> Frame:
    a = float(params["a"])
    b = float(params["b"])
    input_digits = int(params["input_digits"])
    ulps = int(params["perturb_ulps"])
    degenerate = False
    if a == b:
        b = fp.next_float(a, direction=-1)
        degenerate = True
    d = a - b
    sign = 1.0 if d > 0.0 else -1.0
    mag = max(abs(a), abs(b))
    gap = abs(d)
    delta_in = _input_uncertainty(input_digits)
    loss_now = float(np.log10(mag / gap)) if gap > 0.0 else LOSS_MAX

    # ---- 损失位数 t 的扫描：输入按最坏方向各推 ulps 个 ulp，实测放大比 ------ #
    # 从 0 开始：a、b 相距较远时当前点可能落在 t≈0.1，若曲线从 1 才开始，
    # 标记就会跑到线段左侧（用户看到的“点跑出线外”）。0..16 覆盖参数的全部可能取值。
    t_grid = np.linspace(0.0, LOSS_MAX, 33)
    digits_grid = np.zeros_like(t_grid)
    error_grid = np.zeros_like(t_grid)
    theory_grid = np.zeros_like(t_grid)
    round_grid = np.zeros_like(t_grid)
    for i, tt in enumerate(t_grid):
        gap_t = mag * 10.0 ** (-float(tt))
        b_t = a - sign * gap_t
        digits_grid[i] = fp.remaining_digits(gap_t, mag, u=delta_in)
        error_grid[i] = fp.perturbation_amplification(a, b_t, ulps=max(ulps, 1))
        theory_grid[i] = delta_in * (abs(a) + abs(b_t)) / gap_t
        round_grid[i] = fp.subtraction_relative_error(a, b_t)

    digits_mark = fp.remaining_digits(gap, mag, u=delta_in)
    error_mark = fp.perturbation_amplification(a, b, ulps=ulps) if ulps > 0 else 0.0
    round_mark = fp.subtraction_relative_error(a, b)
    error_floor = ERROR_FLOOR
    n_exact_zero = int(np.count_nonzero(round_grid == 0.0))

    fr = Frame(chapter="1.1", viz="cancellation", kind="frame")
    fr.add("loss_grid", t_grid, unit="1", label="损失位数 t = log10(M/|a−b|)")
    fr.add("digits_remaining", digits_grid, unit="1", label="剩余有效位数（先验估计）")
    fr.add("digits_mark", np.array([digits_mark]), unit="1", label="当前剩余有效位数")
    fr.add("error", np.maximum(error_grid, error_floor), unit="1", label="实测相对误差（0 以 u/2 显示）")
    fr.add("error_theory", np.maximum(theory_grid, error_floor), unit="1", label="理论上界 δ_in(|a|+|b|)/|a−b|")
    fr.add("roundoff", np.maximum(round_grid, error_floor), unit="1", label="减法本身的舍入误差（0 以 u/2 显示）")
    fr.add("error_mark", np.array([max(error_mark, error_floor)]), unit="1", label="当前实测相对误差")
    fr.add("loss_mark", np.array([loss_now]), unit="1", label="当前损失位数")
    fr.add("roundoff_mark", np.array([max(round_mark, error_floor)]), unit="1", label="当前减法舍入误差")
    fr.add("a_value", np.array([a]), unit="1", label="被减数 a")
    fr.add("b_value", np.array([b]), unit="1", label="减数 b")
    fr.add("gap_value", np.array([gap]), unit="1", label="|a−b|")
    fr.set_meta(
        a=a,
        b=b,
        gap=gap,
        loss_digits=_scalar(loss_now),
        input_digits=input_digits,
        delta_in=delta_in,
        digits_remaining=digits_mark,
        digits_lost=float(LOSS_MAX - digits_mark),
        cond_subtraction=_scalar(fp.cancellation_condition(a, b)),
        sterbenz_exact=bool(fp.sterbenz_exact(a, b)),
        subtraction_relative_error=round_mark,
        perturb_ulps=ulps,
        perturbation_amplification=_scalar(error_mark),
        error_bound=_scalar(delta_in * (abs(a) + abs(b)) / gap),
        error_floor=error_floor,
        roundoff_exact_zero_steps=n_exact_zero,
        degenerate=degenerate,
        overflow=bool(not np.isfinite(error_mark) or error_mark > 1.0),
    )
    fr.set_strings(
        note=_cancel_note(a, b, gap, digits_mark, error_mark, round_mark, delta_in, degenerate),
        exact_a=fp.exact_decimal(a, digits=25),
        exact_b=fp.exact_decimal(b, digits=25),
        sterbenz_note=(
            "Sterbenz 引理：b/2 ≤ a ≤ 2b 时 a−b 在浮点里**精确**。所以「灾难性抵消」"
            "说的不是减法算错了，而是输入的相对误差 δ_in 被 (|a|+|b|)/|a−b| 放大。"
        ),
    )
    return fr


def _cancel_note(
    a: float, b: float, gap: float, digits: float, err: float, round_err: float, delta_in: float, degenerate: bool
) -> str:
    head = "两个数完全相等，已用相邻的一个 ulp 代替 b。" if degenerate else ""
    return (
        f"{head}两数之差 {gap:.6e}，输入的相对不确定度 {delta_in:.2e}，"
        f"先验判断还能剩 {digits:.2f} 位有效数字，扰动后的实测相对误差 {err:.3e}，"
        f"而减法本身产生的舍入误差只有 {round_err:.1e}"
    )


# =========================================================================== #
# viz 3：条件数椭球
# =========================================================================== #
def _ellipsoid_experiment(
    family: str,
    log_kappa: float,
    shear: float,
    delta_b: float,
    b_angle: float,
    norm_key: Any,
) -> tuple[float, float]:
    """在目标 :math:`\\log_{10}\\kappa` 处实测一次右端项扰动的放大比。"""
    kappa_target = 10.0 ** float(log_kappa)
    A = cond.condition_matrix(family, kappa_target, shear=shear)
    th = np.deg2rad(b_angle)
    b = np.array([np.cos(th), np.sin(th)])
    db = float(delta_b) * float(np.sqrt(np.sum(b * b))) * np.array([-np.sin(th), np.cos(th)])
    try:
        amp = cond.rhs_amplification_measured(A, b, db, p=norm_key)
    except (ZeroDivisionError, ValueError):
        amp = float("nan")
    return kappa_target, amp


def _cond_ellipsoid(params: Mapping[str, Any], t: float) -> Frame:
    family = str(params["family"])
    shear = float(params["shear"])
    delta_b = float(params["delta_b"])
    b_angle = float(params["b_angle"])
    norm_label = str(params["norm"])
    norm_key = _norm_key(norm_label)
    log_kappa = max(0.0, min(16.0, float(t)))
    kappa_target = 10.0**log_kappa
    A = cond.condition_matrix(family, kappa_target, shear=shear)

    theta, ix, iy = cond.ellipse_image(A, n_theta=361)
    _, cx, cy = cond.ellipse_image(np.eye(2), n_theta=361)
    th = np.deg2rad(b_angle)
    b = np.array([np.cos(th), np.sin(th)])
    db = delta_b * np.array([-np.sin(th), np.cos(th)])
    try:
        x_exact = exact_solve(A, b)
    except ZeroDivisionError:
        x_exact = np.array([np.nan, np.nan])
    res = lu_partial(A)
    x_num = solve_lu(res, b) if not res.singular else np.array([np.nan, np.nan])
    try:
        x_pert = cond.perturbed_solution(A, b, db)
    except ZeroDivisionError:
        x_pert = np.array([np.nan, np.nan])

    denom = float(np.max(np.abs(x_exact))) if np.all(np.isfinite(x_exact)) else 0.0
    fwd = float(np.max(np.abs(x_num - x_exact))) / denom if denom > 0.0 else float("nan")
    try:
        amp_now = cond.rhs_amplification_measured(A, b, db, p=norm_key)
    except (ZeroDivisionError, ValueError):
        amp_now = float("nan")

    kappas = {
        "1": cond.condition_1(A),
        "2": cond.condition_2(A),
        "inf": cond.condition_inf(A),
        "fro": cond.condition_fro(A),
    }
    kappa_measured = kappas[norm_label]
    smin, smax = cond.ellipse_semi_axes(A)
    det_a = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])

    # ---- 副轴曲线：沿 log10 κ 的实测放大比 vs 理论上界 κ -------------------- #
    grid = np.linspace(0.0, 16.0, 33)
    error = np.zeros_like(grid)
    bound = np.zeros_like(grid)
    for i, lg in enumerate(grid):
        k_t, amp = _ellipsoid_experiment(family, float(lg), shear, delta_b, b_angle, norm_key)
        error[i] = amp
        bound[i] = k_t
    error = np.where(np.isfinite(error) & (error > 0.0), error, ERROR_FLOOR)
    j = int(np.argmin(np.abs(grid - log_kappa)))

    fr = Frame(chapter="1.1", viz="cond_ellipsoid", kind="frame")
    fr.add("theta", theta, unit="rad", label="参数角")
    fr.add("circle_x", cx, unit="1", label="单位圆")
    fr.add("circle_y", cy, unit="1", label="单位圆")
    fr.add("image_x", ix, unit="1", label="A 的像（椭圆）")
    fr.add("image_y", iy, unit="1", label="A 的像（椭圆）")
    fr.add("b_vec", np.array([0.0, b[0], 0.0, b[1]]), unit="1", label="b")
    fr.add("db_vec", np.array([0.0, db[0], 0.0, db[1]]), unit="1", label="δb")
    fr.add("x_exact", np.array([0.0, x_exact[0], 0.0, x_exact[1]]), unit="1", label="精确解 x")
    fr.add("x_num", np.array([0.0, x_num[0], 0.0, x_num[1]]), unit="1", label="数值解 x̂")
    fr.add("x_pert", np.array([0.0, x_pert[0], 0.0, x_pert[1]]), unit="1", label="扰动后的解 x̂′")
    fr.add("kappa_grid", grid, unit="1", label="log10 κ（目标）")
    fr.add("error", np.maximum(error, ERROR_FLOOR), unit="1", label="实测放大比（0 以 u/2 显示）")
    fr.add("error_bound", bound, unit="1", label="理论上界 κ")
    fr.add("error_mark", np.array([float(max(error[j], ERROR_FLOOR))]), unit="1", label="当前位置的实测放大比")
    fr.add("kappa_mark", np.array([float(grid[j])]), unit="1", label="当前位置的 log10 κ")
    fr.set_meta(
        shear=shear,
        log10_kappa=log_kappa,
        kappa_target=kappa_target,
        kappa_measured=kappa_measured,
        kappa_1=kappas["1"],
        kappa_2=kappas["2"],
        kappa_inf=kappas["inf"],
        kappa_fro=kappas["fro"],
        sigma_min=smin,
        sigma_max=smax,
        det_a=det_a,
        delta_b=delta_b,
        b_angle=b_angle,
        amplification_measured=_scalar(amp_now),
        amplification_over_kappa=_scalar(amp_now / kappa_measured) if kappa_measured > 0 else None,
        forward_error=_scalar(fwd),
        backward_error=_scalar(backward_error(A, x_num, b)) if np.all(np.isfinite(x_num)) else None,
        residual=_scalar(residual_norm(A, x_num, b)) if np.all(np.isfinite(x_num)) else None,
        bound_rhs=_scalar(kappa_measured * delta_b),
        kappa_u=_scalar(kappa_measured * U),
        n_theta=361,
        error_floor=ERROR_FLOOR,
        overflow=bool(np.max(np.abs(np.concatenate([ix, iy]))) > ELLIPSE_HALF_WIDTH),
    )
    fr.set_strings(
        family=family,
        norm=norm_label,
        note=_ellipse_note(family, norm_label, kappa_measured, kappa_target, amp_now, float(kappa_measured * U)),
        norm_note=(
            f"本帧的 κ 用 <{norm_label}> 范数：κ₁={kappas['1']:.3g}、κ₂={kappas['2']:.3g}、"
            f"κ∞={kappas['inf']:.3g}、κ_F={kappas['fro']:.3g}——不同范数可以差一个 n 因子，"
            "图上任何「条件数」都要连范数一起读。"
        ),
        floor_note="副轴是对数轴：实测放大比小于 u/2 时按地板 u/2 显示（不会出现 log(0)）。",
    )
    return fr


def _ellipse_note(
    family: str, norm_label: str, kappa: float, target: float, amp: float, kappa_u: float
) -> str:
    if kappa_u >= 1.0:
        return (
            f"κu = {kappa_u:.2e} ≥ 1：A 在 float64 里已经和奇异矩阵无法区分，"
            "此时椭圆与实测曲线都只是舍入的产物，不要再读具体数字。"
        )
    return (
        f"{family} 族：目标 κ={target:.3g}，实测 κ_{norm_label}={kappa:.3g}，"
        f"当前方向上的实测放大比 {amp:.3g}（理论上界 κ={kappa:.3g}，实测/上界 = {amp / kappa:.2f}）"
    )


# =========================================================================== #
# viz 4：求和顺序
# =========================================================================== #
def _sum_values(big: float, small: float, n_terms: int) -> list[float]:
    """构造项序列：``big`` 在最前，其后是 ``n-1`` 个 ``small``。"""
    n = max(2, int(n_terms))
    return [float(big)] + [float(small)] * (n - 1)


def _summation_order(params: Mapping[str, Any], t: float) -> Frame:
    big = float(params["big"])
    small = float(params["small"])
    n_terms = int(params["n_terms"])
    scheme = str(params["compensated"])
    values = _sum_values(big, small, n_terms)
    n = len(values)
    traj = fp.sum_trajectory(values)
    k = _at_least_one(t, n)
    idx = int(np.argmin(np.abs(traj.index - float(k))))
    floor = traj.floor
    # 选中的补偿归约方式：serial 直接用轨迹里那条（流式、免费），其余在前缀上重算。
    if scheme == "serial":
        err_scheme = traj.error_kahan
    else:
        _, err_scheme = fp.partial_sum_errors(values, method=scheme, checkpoints=list(traj.index))

    n_head = min(n, 64)
    fr = Frame(chapter="1.1", viz="summation_order", kind="frame")
    fr.add("terms_k", np.arange(n_head, dtype=float), unit="1", label="项序号 i")
    fr.add("terms", np.asarray(values[:n_head], dtype=float), unit="1", label="项序列（前 64 项）")
    fr.add("k_grid", traj.index, unit="1", label="部分和长度 k")
    fr.add("error_naive", np.maximum(traj.error_naive, floor), unit="1", label="朴素求和误差（0 以地板显示）")
    fr.add(
        "error_kahan",
        np.maximum(err_scheme, floor),
        unit="1",
        label=f"补偿求和误差（{scheme}，0 以地板显示）",
    )
    fr.add(
        "error_kahan_serial",
        np.maximum(traj.error_kahan, floor),
        unit="1",
        label="串行 Kahan 误差（参考，0 以地板显示）",
    )
    fr.add("error_pairwise", np.maximum(traj.error_pairwise, floor), unit="1", label="成对求和误差（0 以地板显示）")
    fr.add("error_sorted", np.maximum(traj.error_sorted, floor), unit="1", label="排序后求和误差（0 以地板显示）")
    fr.add("error_bound_naive", traj.error_bound_naive, unit="1", label="理论界 n·u·Σ|x|")
    fr.add("error_bound_kahan", traj.error_bound_kahan, unit="1", label="理论界 (2u+n·u²)·Σ|x|")
    fr.add("error_naive_mark", np.array([float(max(traj.error_naive[idx], floor))]), unit="1", label="当前朴素误差")
    fr.add("error_kahan_mark", np.array([float(max(err_scheme[idx], floor))]), unit="1", label="当前补偿误差")
    fr.add(
        "error_pairwise_mark",
        np.array([float(max(traj.error_pairwise[idx], floor))]),
        unit="1",
        label="当前成对误差",
    )
    fr.add("k_mark", np.array([float(k)]), unit="1", label="当前部分和长度")
    fr.set_meta(
        big=big,
        small=small,
        n_terms=n,
        k=k,
        exact_sum=float(traj.exact[idx]),
        sum_naive=float(traj.naive[idx]),
        sum_kahan=float(err_scheme[idx]),
        sum_pairwise=float(traj.pairwise[idx]),
        sum_sorted=float(traj.sorted[idx]),
        error_naive=float(traj.error_naive[idx]),
        error_kahan=float(err_scheme[idx]),
        error_kahan_serial=float(traj.error_kahan[idx]),
        error_pairwise=float(traj.error_pairwise[idx]),
        error_sorted=float(traj.error_sorted[idx]),
        bound_naive=float(traj.error_bound_naive[idx]),
        bound_kahan=float(traj.error_bound_kahan[idx]),
        kahan_over_naive=(
            float(err_scheme[idx] / traj.error_naive[idx]) if traj.error_naive[idx] > 0.0 else 1.0
        ),
        parallel_over_serial=(
            float(err_scheme[idx] / traj.error_kahan[idx]) if traj.error_kahan[idx] > 0.0 else 1.0
        ),
        terms_abs_sum=float(sum(abs(v) for v in values[:k])),
        error_floor=float(floor),
        error_exact_zero_points=traj.n_exact_zero,
        unit_roundoff=U,
        overflow=bool(not np.all(np.isfinite(traj.error_naive))),
    )
    fr.set_strings(
        scheme=scheme,
        note=_sum_note(big, small, n, k, traj, idx, scheme, float(err_scheme[idx])),
        floor_note=(
            f"副轴是对数轴：误差恰好为 0 的采样点（{traj.n_exact_zero} 个，按朴素/Kahan/成对三法合计）"
            f"按地板 {floor:.3e} 显示，不能读成「误差是地板那么大」。"
        ),
        scheme_note=(
            "serial=串行 Kahan；shuffled=把项做确定性置换后再 Kahan；"
            "parallel8/parallel64=分成 8/64 段、段内 Kahan、段间朴素合并。"
            "置换不改变误差量级（界对任意顺序都成立），分段合并才会——这正是「Kahan 不能直接并行」的含义。"
        ),
    )
    return fr


def _sum_note(big: float, small: float, n: int, k: int, traj: Any, idx: int, scheme: str, err_scheme: float) -> str:
    ratio = float(err_scheme / traj.error_naive[idx]) if traj.error_naive[idx] > 0.0 else 1.0
    par = float(err_scheme / traj.error_kahan[idx]) if traj.error_kahan[idx] > 0.0 else 1.0
    return (
        f"big={big:.3g}, small={small:.3g}, n={n}, 补偿方式 {scheme}：k={k} 时朴素误差 "
        f"{traj.error_naive[idx]:.3e}、补偿 {err_scheme:.3e}（相对朴素 {ratio:.3g}）、"
        f"串行 Kahan {traj.error_kahan[idx]:.3e}（本次归约 / 串行 = {par:.3g}）、"
        f"成对 {traj.error_pairwise[idx]:.3e}"
    )
