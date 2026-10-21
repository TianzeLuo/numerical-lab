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


def _fmt_kappa(value: float) -> str:
    return f"{value:.3g}" if np.isfinite(value) else "无有限值"


def _safe_vector_solve(solve: Any) -> tuple[np.ndarray, float, str]:
    """解一个 2×2 线性系统；失败时返回占位向量 + defined=0 + 人类可读原因。

    不能把 ``NaN`` 直接写进数组：NLF 的数组段会原样编码 NaN，前端拿到的是一条
    "凭空消失"的曲线。这里把"没有有限解"变成**显式状态**，由 model 跳过箭头并写进图注。
    """
    try:
        v = np.asarray(solve(), dtype=float).reshape(2)
    except (ZeroDivisionError, ValueError, FloatingPointError) as exc:
        return np.zeros(2), 0.0, str(exc) or "线性求解失败"
    if not np.all(np.isfinite(v)):
        return np.zeros(2), 0.0, "解的分量不是有限数（尺度超出 double）"
    return v, 1.0, ""


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
                    "help": "决定把数轴放大到哪一段。「0.3（加法）」解释 $0.1+0.2$ 为什么落在 0.3 旁边；「0.25（2 的幂）」看跨越 2 的幂时格点间距如何变化。切换后蓝点、红点与间距括号都会重新定位。",
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
                    "help": "以窗口中心的一个 ulp 为尺子，左右各显示多少个格点。调大能看到更多相邻 double；调小则把 $0.1+0.2$ 与 0.3 的差异放大到像素级。它只改变观察窗口宽度，不改变真实数据。",
                },
            ],
            "time": {"mode": "scrub", "tMax": 64.0, "dt": 1.0, "label": "$0.1$ 的累加次数 $k$", "initial": 64.0},
            "requiredAux": ["residual"],
            "axes": [
                {
                    "id": "ladder",
                    "autoscaled": False,
                    "title": "浮点格点：横轴是与窗口中心的距离（单位 $\\mathrm{ulp}$），每个圆点是一个可表示的 double",
                    "ylabel": "浮点格点（示意行，无数值）",
                },
                {
                    "id": "residual",
                    "autoscaled": True,
                    "title": "$\\mathrm{fl}(0.1)$ 累加 $k$ 次后，结果与 $k/10$ 的差（对数轴）",
                    "ylabel": "$|s_k-k/10|$",
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
                "**先把数轴变成离散的。** 二进制的 double 用 1 位符号、11 位指数和 52 位尾数表示一个数。把指数固定在一个区间 $[2^e,2^{e+1})$ 内，尾数每加 1，数值就前进 $2^{e-52}$；这正是相邻两个可表示数的距离，记作 $\\mathrm{ulp}(x)$。所以浮点数不是“实数的一个近似值”，而是间距由指数决定的一串格点。",
                "**舍入就是选最近的格点。** 定义 $\\mathrm{fl}(x)$ 为离实数 $x$ 最近的 double（恰好在中点时按偶尾数取舍）。最坏情况下绝对误差不超过半个格点：$|\\mathrm{fl}(x)-x|\\le\\tfrac12\\mathrm{ulp}(x)$。把它除以 $|x|$，得到相对误差界 $u=2^{-53}\\approx1.11\\times10^{-16}$，这就是后文反复出现的单位舍入。",
                "**0.1 和 0.2 从一开始就不在格点上。** 十进制 $0.1$ 的二进制展开无限循环，double 只能保留前 53 位，因此 $\\mathrm{fl}(0.1)$ 与 $0.1$ 相差约 $5.55\\times10^{-18}$；$0.2$ 同理。两个近似值相加后再舍入一次，得到 $0.30000000000000004$；而 $0.3$ 的最近 double 是 $0.2999999999999999889\\ldots$，两者恰好相差 $1$ 个 ulp。红点和蓝点不是算错了，而是两条不同的舍入路径。",
                "**跨过 2 的幂，格点会变密或变疏。** 在 $[2^e,2^{e+1})$ 内间距固定为 $2^{e-52}$；一到 $2^{e+1}$，指数加一、间距翻倍。反过来看 $0.25=2^{-2}$：它左侧属于 $[2^{-3},2^{-2})$，间距是 $2^{-55}$；右侧是 $2^{-54}$，所以左半边的点更密。把观察位置切到 0.25，两个括号会分别量出左右两个不同的 ulp。",
                "**累加误差为什么会越走越远。** 设第 $k$ 步的部分和为 $s_k$，数学真值为 $k/10$。每一步只保证 $|\\mathrm{fl}(s_{k-1}+0.1)-(s_{k-1}+0.1)|\\le u|s_{k-1}+0.1|$。把各步误差相加，最坏情况是 $|s_k-k/10|\\le k\\,u/2$；实际符号有正有负，通常小于这个线性界，但趋势仍随 $k$ 增长。下图把恰好为零的残差抬到地板 $u/2$，那只表示“在 log 轴上画得出来”，不代表误差等于地板。",
                "**读图时不要越过数据下结论。** 圆点只展示窗口内的几十个 double，窗口外还有无数个；横轴的单位是“中心处的一个 ulp”，跨到 2 的幂另一侧后，同样的横向距离对应的绝对差可能已经变了。蓝点、红点和间距括号都能悬停读出完整的十进制值——先核对它们，再谈“0.1+0.2 为什么不是 0.3”。",
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
                    "help": "被减数。它与 $b$ 越接近，相减后剩下的有效位数越少。滑轨用于快速扫；要精确复现实验，点右侧数字框输入（例如 $1.000001$）。",
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
                    "help": "减数。把它调到与 $a$ 只差几个 ulp，就能看到输入误差被放大成百分之几十甚至更大。两数完全相等时程序会换成相邻的 double 并明确标注。",
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
                    "help": "假设每个输入在十进制里有 $d$ 位可信数字，对应的相对不确定度约为 $\\tfrac12 10^{1-d}$。$d$ 越小，输入本身越不可信，相减后爆炸得越早。$16$ 附近已接近 double 能可靠表达的上限 $\\log_{10}(2^{53})\\approx15.95$。",
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
                    "help": "实测放大比时，把 $a$ 向上、$b$ 向下各推开多少个 ulp。$0$ 表示不扰动（只显示先验估计）；调大后，输入里那点被放大的误差会变成看得见的相对误差。",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "digits",
                    "autoscaled": False,
                    "title": "还剩多少位十进制有效数字（横轴 = 抵消掉多少位，记作 $t$）",
                    "ylabel": "剩余位数",
                },
                {
                    "id": "error",
                    "autoscaled": True,
                    "title": "实测相对误差与理论上界（横轴 = 损失位数 $t$）",
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
                "**给输入一个可比较的误差模型。** 如果十进制输入保留 $d$ 位有效数字，它的相对不确定度约为 $\\delta_{\\mathrm{in}}=\\tfrac12 10^{1-d}$。例如 $d=16$ 时 $\\delta_{\\mathrm{in}}=5\\times10^{-16}$，与 double 的单位舍入同量级。实验把 $a$、$b$ 都看作“真值 ± 相对误差 $\\delta_{\\mathrm{in}}$”，而不是假装它们精确。",
                "**减法怎样放大误差。** 设绝对误差满足 $|\\delta a|\\le\\delta_{\\mathrm{in}}|a|$、$|\\delta b|\\le\\delta_{\\mathrm{in}}|b|$，则 $a-b$ 的绝对误差最多是这两个误差之和。除以结果本身，得到相对误差界 $\\frac{|\\delta(a-b)|}{|a-b|}\\le\\delta_{\\mathrm{in}}\\frac{|a|+|b|}{|a-b|}.$ 因子 $(|a|+|b|)/|a-b|$ 就是减法的条件数：两数越接近，分母越小，输入的相对误差被放大得越厉害。",
                "**减法本身通常是精确的。** Sterbenz 引理说：当 $b/2\\le a\\le 2b$ 时，浮点差 $\\mathrm{fl}(a-b)$ 精确等于 $a-b$（两个相邻格点之间的差仍然落在格点上）。所以图中那条“减法自身舍入误差”恒为 0 不是程序偷懒；爆炸的是输入原本携带的误差，不是减号。",
                "**损失多少位，可以数出来。** 记 $M=\\max(|a|,|b|)$、$t=\\log_{10}(M/|a-b|)$，则 $|a-b|=M\\cdot10^{-t}$。代入上面的界，相对误差约被放大 $10^t$ 倍；如果输入还有 $d$ 位可信，相减后大约只剩 $d-t$ 位。这就是上图那条斜率 $-1$ 的直线：每抵消一个十进制位，少一位有效数字。$t$ 接近 $\\log_{10}(2^{53})\\approx15.95$ 时，连 double 自身都分不清了。",
                "**实测是怎样做的。** 程序把 $a$ 向上、$b$ 向下各推 $k$ 个 ulp（最坏扰动方向），精确计算新的差与原差的相对变化，得到“实测放大比”；再乘上输入的相对不确定度 $\\delta_{\\mathrm{in}}$，就是下图红线的相对误差。理论上界用连续模型 $\\delta_{\\mathrm{in}}(|a|+|b|)/|a-b|$ 画成橙色虚线。实测点可以略高于或低于上界，因为 ulp 是离散的，而连续模型是平滑近似。",
                "**读图时注意两件事。** 第一，横轴最右端的目标差可能小于一个 ulp，此时程序改用“最近但确实不同”的 double 计算，并在帧 note 里注明，不能把末端读成“差真的这么小”。第二，把 $b$ 设成与 $a$ 完全相等时，程序会取相邻的一个 ulp 代替 $b$，否则 $\\log(0)$ 没有定义；这一点也会明确标注。",
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
                    "help": "选择矩阵长什么样：shear 先剪切再拉伸；diagonal 只沿坐标轴拉伸；rotation 只旋转（$\\kappa=1$）；graded 让两行几乎相同；triangular 是上三角。族不同，目标 $\\kappa$ 与实测 $\\kappa$ 会不同，椭圆形状也不同。",
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
                    "help": "只对 shear 族生效。$s$ 控制剪切矩阵 $S$（把点 $(x_1,x_2)$ 变成 $(x_1+s x_2,\\,x_2)$） 的强度：$s=0$ 时拉伸轴与坐标轴对齐；$s$ 越大，椭圆越倾斜，但行列式（面积缩放）不变。",
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
                    "help": "右端项 $b$ 的相对扰动大小。它决定数值解 $\\hat{x}$ 与扰动解 $\\hat{x}'$ 差多少。低于 $u\\approx1.11\\times10^{-16}$ 的扰动在 double 里无法表示，实测放大比会只剩舍入噪声。",
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
                    "help": "右端项 $b$ 在平面里的方向（从 $x_1$ 轴逆时针量）。最坏方向的放大比接近 $\\kappa$；某些方向远低于 $\\kappa$，这正说明 $\\kappa$ 是“所有方向的上界”，不是每个方向的预测。",
                },
                {
                    "name": "norm",
                    "label": "范数",
                    "type": "enum",
                    "options": ["1", "2", "inf", "fro"],
                    "default": "2",
                    "critical": [],
                    "criticalNote": "",
                    "help": "选择用哪种矩阵范数计算条件数。$1$、$2$、$\\infty$、F 范数之间可能差一个 $n$ 的因子；图上的椭圆半轴比始终对应 2-范数 $\\kappa_2$，其他范数只出现在数值与上界里。",
                },
            ],
            "time": {"mode": "scrub", "tMax": 16.0, "dt": 0.25, "label": "$\\log_{10}\\kappa$（目标）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "geometry",
                    "autoscaled": False,
                    "title": "单位圆与 $A$ 作用后的椭圆",
                    "ylabel": "$x_2$",
                },
                {
                    "id": "error",
                    "autoscaled": True,
                    "title": "实测放大比与最坏情况上界 $\\kappa(A)$",
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
                "**条件数衡量的是一道“题”有多敏感。** 对线性方程组 $Ax=b$，如果右端项有很小的扰动 $\\delta b$，解会变成 $x+\\delta x$，且 $A\\,\\delta x=\\delta b$。两边取范数：$\\|\\delta x\\|\\le\\|A^{-1}\\|\\,\\|\\delta b\\|$。再把 $\\|b\\|\\le\\|A\\|\\,\\|x\\|$ 用上，就得到相对误差界 $\\frac{\\|\\delta x\\|}{\\|x\\|}\\le\\kappa(A)\\,\\frac{\\|\\delta b\\|}{\\|b\\|},\\qquad \\kappa(A)=\\|A\\|\\,\\|A^{-1}\\|.$ 这个 $\\kappa$ 与求解算法无关，只由矩阵本身决定。",
                "**为什么单位圆会变成椭圆。** 所有长度为 1 的向量组成单位圆。矩阵 $A$ 把它线性地压扁、拉伸，像是一条椭圆；椭圆的半轴长正好是 $A$ 的两个奇异值 $\\sigma_{\\max}\\ge\\sigma_{\\min}$，方向是左右奇异向量。半轴比 $\\sigma_{\\max}/\\sigma_{\\min}$ 就是 2-范数条件数 $\\kappa_2(A)$。椭圆越扁，某些方向上的解对输入越敏感。",
                "**上界与某个具体方向不是一回事。** 上面的界对所有 $\\delta b$ 方向都成立，但最坏方向才能取到等号。图上的“实测放大比”把当前方向的相对解变化除以相对 $b$ 扰动：$\\mathrm{amp}=\\frac{\\|\\delta x\\|/\\|x\\|}{\\|\\delta b\\|/\\|b\\|}.$ 它一般低于 $\\kappa$；只有 $\\delta b$ 沿着最小奇异值方向时，才接近 $\\kappa$。看到实测值远小于上界，不代表上界错了。",
                "**矩阵族与剪切参数。** 五个族用不同方式构造出目标条件数：shear 先剪切再各向异性拉伸；diagonal 只沿坐标轴拉伸；rotation 是正交矩阵，$\\kappa_2=1$；graded 让两行几乎相同；triangular 是上三角。shear 族的矩阵写作 $A=R^{\\mathrm T}\\,\\mathrm{diag}(1,1/\\kappa)\\,R\\,S$，其中 $S$ 把点 $(x_1,x_2)$ 变成 $(x_1+s x_2,x_2)$；$s$ 改变椭圆朝向和形状，但 $\\det A=1/\\kappa$ 与 $s$ 无关。",
                "**目标 $\\kappa$ 与实测 $\\kappa$ 为什么不同。** 参数只规定“想要多大的条件数”，具体矩阵族实现出来的谱条件数会有偏差；图上同时标出目标值和实测值，读数时必须用实测值。不同范数 $\\kappa_1,\\kappa_2,\\kappa_\\infty,\\kappa_F$ 还能差一个与维数 $n$ 有关的常数因子，所以“条件数”必须连同范数一起读。",
                "**数值上什么时候不能再信这张图。** 当 $\\kappa u\\ge1$ 时，$A$ 在 double 里已经和奇异矩阵无法区分；此时解向量可能超出画面，程序把箭头按方向裁到 $\\pm3$ 显示，并在溢出说明里给出真实端点。看不到完整箭头不等于它不存在，更不代表解“很温和”。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "summation_order",
            "title": "加法顺序为什么影响结果：朴素、Kahan、成对与排序",
            "description": (
                "同样一组数，换个加法顺序，结果可能不同。上图依次列出大项与小项；"
                "下图比较朴素求和、Kahan 补偿求和、成对求和与排序求和的误差。"
                "默认的 big 是 $2^{53}-1$（刚好低于 1/u），第一次加 1 到达 $2^{53}$，之后每个 1 都落在半个 ulp 内、被整项丢掉；"
                "Kahan 把每一步的舍入损失记下来再加回去，所以通常更准；但把它分段并行、"
                "段间再用朴素加法合并，误差会重新变大。把 big 调到 $2^{53}$ 以上，红色临界提示会出现，差异也更极端。"
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
                    "default": 1.0 / U - 1.0,
                    "critical": [1.0 / U],
                    "criticalNote": (
                        "big ≥ 1/u = 2^53 ≈ 9.0e15 时，1.0 这个「小项」已经落在 big 的半个 ulp 之内，"
                        "朴素求和会把它整项丢掉。默认值取 2^53−1：第一次加 1 恰好到边界，之后每个 1 都被丢掉，"
                        "既演示了问题，又不会一打开就亮红色越界提示。"
                    ),
                    "beyond": "above",
                    "help": "最大的一项。默认取 $2^{53}-1$：第一次加 1 就到达 $2^{53}$，之后每个 1 都小于半个 ulp，朴素加法会把它整项吞掉。超过 $2^{53}$ 后红色临界提示会亮起。",
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
                    "help": "其余 $n-1$ 项的大小。把它调得比 big 的 ulp 还小，朴素求和与补偿求和的差别最明显；调大后所有方法都会更接近精确值。",
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
                    "help": "一共加多少项（第一项是 big，其余是 small）。项数越多，朴素求和的累计舍入损失越大；Kahan、成对求和与排序求和的误差增长慢得多。",
                },
                {
                    "name": "compensated",
                    "label": "补偿求和的归约方式",
                    "type": "enum",
                    "options": ["serial", "shuffled", "parallel8", "parallel64"],
                    "default": "serial",
                    "critical": [],
                    "criticalNote": "",
                    "help": "选择补偿求和的归约方式：serial 是完整串行 Kahan；shuffled 先做确定性置换；parallel8/64 分成 8/64 段、段内 Kahan、段间朴素合并。对比可看到“Kahan 能不能并行”的关键差别。",
                },
            ],
            "time": {"mode": "scrub", "tMax": 4096.0, "dt": 16.0, "label": "部分和长度 $k$", "initial": 4096.0},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "terms",
                    "autoscaled": True,
                    "title": "被相加的项（横轴 = 项序号 $i$；最大项在前）",
                    "ylabel": "$x_i$",
                },
                {
                    "id": "error",
                    "autoscaled": True,
                    "title": "各求和顺序的绝对误差与两条理论上界",
                    "ylabel": "$|s_k-k/10|$",
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
                "**浮点加法只保证一个小相对误差。** 对任意两个浮点数 $x,y$，实际计算满足 $\\mathrm{fl}(x+y)=(x+y)(1+\\varepsilon)$，其中 $|\\varepsilon|\\le u$。从左到右朴素相加 $n$ 项时，每一步的误差都会进入后面的部分和；最坏情况累加起来，得到 $|\\Delta_{\\mathrm{naive}}|\\le (n-1)u\\sum_i|x_i|+O(u^2).$ 误差随项数线性增长，而且大项会把小项直接吞掉。",
                "**Kahan 补偿做了两件事。** 它用 $c$ 记住“刚才这一步舍入丢掉了多少”，下一次加法前先把 $c$ 从输入里减掉，再把新的损失写回 $c$：$y=x_k-c,\\quad t=s+y,\\quad c=(t-s)-y,\\quad s=t.$ 因为 $t-s$ 精确等于实际发生的舍入量，$c$ 就积累着尚未加回的损失。于是误差界变成 $|\\Delta_{\\mathrm{Kahan}}|\\le 2u\\sum_i|x_i|+O(nu^2)$，几乎与 $n$ 无关。",
                "**成对求和靠的是树的高度。** 把 $n$ 项两两相加，再对得到的 $n/2$ 个和两两相加，递归进行。每个元素只经过约 $\\lceil\\log_2 n\\rceil$ 次加法，所以 $|\\Delta_{\\mathrm{pairwise}}|\\le u\\lceil\\log_2 n\\rceil\\sum_i|x_i|+O(u^2).$ 排序后再朴素相加也有同量级的效果：先把量级相近的项放在一起，避免“大数反复吞小数”。",
                "**Kahan 不是“永远准确”。** 它仍然会积累误差，只是常数更小；当某一项本身小于当前部分和的半个 ulp 时，它也救不回来。更重要的是，Kahan 的误差界依赖“从头到尾按顺序传播补偿量”这个结构，不能随意拆开。",
                "**为什么分段并行会退回朴素量级。** 把 $n$ 项分成 $P$ 段，每段内部用 Kahan，段和之间再用普通加法合并。段内误差仍小，但合并 $P$ 个段和时，小段会被大段吞掉；误差变成 $O(Pu\\sum|s_j|)$，也就是“段数 × 朴素合并的放大”。这不是实现细节，而是补偿算法与并行归约的结构冲突。",
                "**图上到底量的是什么。** 下图画的是每种方法的部分和与高精度参考值之差的绝对值，附两条理论上界 $n u\\sum|x_i|$ 与 $(2u+nu^2)\\sum|x_i|$。默认 big 取 $2^{53}-1$，第一次加 1 到达 $2^{53}$，之后每个 1 都小于 half-ulp，朴素求和把它们整项丢掉——这就是两条曲线拉开的原因。对数轴上恰好为零的误差被抬到地板显示，不能读成“误差等于地板”。",
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
    clamped_steps = 0
    for i, tt in enumerate(t_grid):
        gap_t = mag * 10.0 ** (-float(tt))
        b_t = a - sign * gap_t
        # 当目标差小于 a 的半个 ulp 时，`a - gap_t` 会四舍五入回 a 本身，
        # 后面的相对误差就是 0/0。这里改用「离 a 最近、但确实不同的 double」，
        # 并让理论曲线也用这个真实可表示的差——不假装能表示更小的间隔。
        if b_t == a:
            b_t = fp.next_float(a, direction=-sign)
            gap_t = abs(a - b_t)
            clamped_steps += 1
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
        grid_clamped_steps=clamped_steps,
        overflow=bool(not np.isfinite(error_mark) or error_mark > 1.0),
    )
    fr.set_strings(
        note=_cancel_note(a, b, gap, digits_mark, error_mark, round_mark, delta_in, degenerate)
        + (
            f"；曲线最右端有 {clamped_steps} 个采样点的目标差已经小于 double 能分辨的间隔，"
            "这些点按当前数字间实际最小的可表示差计算，不能读成「差真的那么小」。"
            if clamped_steps > 0
            else ""
        ),
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
    # 奇异或数值上解不动时，数组里**不能放 NaN/∞**（PROTOCOLS §1.3：裸 NaN 会让
    # 整帧被拒收）。占位 0 + 一维 defined 掩码，前端据此跳过对应箭头并在图注里点名。
    x_exact, x_exact_defined, exact_error = _safe_vector_solve(lambda: exact_solve(A, b))
    res = lu_partial(A)
    if res.singular:
        x_num, x_num_defined, num_error = np.zeros(2), 0.0, "LU 分解发现矩阵奇异"
    else:
        x_num, x_num_defined, num_error = _safe_vector_solve(lambda: solve_lu(res, b))
    x_pert, x_pert_defined, pert_error = _safe_vector_solve(lambda: cond.perturbed_solution(A, b, db))
    undefined_names = [
        name
        for name, flag in (
            ("精确解 $x$", x_exact_defined),
            ("数值解 $\\hat{x}$", x_num_defined),
            ("扰动解 $\\hat{x}'$", x_pert_defined),
        )
        if not flag
    ]

    denom = float(np.max(np.abs(x_exact))) if x_exact_defined else 0.0
    fwd = (
        float(np.max(np.abs(x_num - x_exact))) / denom
        if denom > 0.0 and x_num_defined
        else float("nan")
    )
    try:
        amp_now = cond.rhs_amplification_measured(A, b, db, p=norm_key)
    except (ZeroDivisionError, ValueError):
        amp_now = float("nan")

    # 条件数函数在 A 数值奇异时会抛错（graded 族在 κ≳1e16 时真的会把矩阵压成 [[1,1],[1,1]]）。
    # 这不是程序错误，而是本图要展示的结论；把"没有有限条件数"写进 note/None，而不是让请求 500。
    def _cond_or_nan(fn: Any) -> float:
        try:
            value = float(fn(A))
        except (ZeroDivisionError, ValueError, FloatingPointError):
            return float("nan")
        return value if np.isfinite(value) else float("nan")

    kappas = {
        "1": _cond_or_nan(cond.condition_1),
        "2": _cond_or_nan(cond.condition_2),
        "inf": _cond_or_nan(cond.condition_inf),
        "fro": _cond_or_nan(cond.condition_fro),
    }
    kappa_measured = kappas[norm_label]
    kappa_u = kappa_measured * U if np.isfinite(kappa_measured) else float("nan")
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
    fr.add("x_exact_defined", np.array([x_exact_defined]), unit="1", label="精确解是否存在（1/0）")
    fr.add("x_num_defined", np.array([x_num_defined]), unit="1", label="数值解是否存在（1/0）")
    fr.add("x_pert_defined", np.array([x_pert_defined]), unit="1", label="扰动解是否存在（1/0）")
    fr.add("matrix_A", A.reshape(-1), unit="1", label="当前矩阵 A（行优先 2×2）")
    fr.add("kappa_grid", grid, unit="1", label="log10 κ（目标）")
    fr.add("error", np.maximum(error, ERROR_FLOOR), unit="1", label="实测放大比（0 以 u/2 显示）")
    fr.add("error_bound", bound, unit="1", label="理论上界 κ")
    fr.add("error_mark", np.array([float(max(error[j], ERROR_FLOOR))]), unit="1", label="当前位置的实测放大比")
    fr.add("kappa_mark", np.array([float(grid[j])]), unit="1", label="当前位置的 log10 κ")
    fr.set_meta(
        shear=shear,
        log10_kappa=log_kappa,
        kappa_target=kappa_target,
        kappa_measured=_scalar(kappa_measured),
        kappa_1=_scalar(kappas["1"]),
        kappa_2=_scalar(kappas["2"]),
        kappa_inf=_scalar(kappas["inf"]),
        kappa_fro=_scalar(kappas["fro"]),
        sigma_min=smin,
        sigma_max=smax,
        det_a=det_a,
        delta_b=delta_b,
        b_angle=b_angle,
        amplification_measured=_scalar(amp_now),
        amplification_over_kappa=(
            _scalar(amp_now / kappa_measured) if np.isfinite(kappa_measured) and kappa_measured > 0 else None
        ),
        forward_error=_scalar(fwd),
        backward_error=_scalar(backward_error(A, x_num, b)) if x_num_defined else None,
        residual=_scalar(residual_norm(A, x_num, b)) if x_num_defined else None,
        x_exact_defined=x_exact_defined,
        x_num_defined=x_num_defined,
        x_pert_defined=x_pert_defined,
        matrix_a00=float(A[0, 0]),
        matrix_a01=float(A[0, 1]),
        matrix_a10=float(A[1, 0]),
        matrix_a11=float(A[1, 1]),
        bound_rhs=_scalar(kappa_measured * delta_b) if np.isfinite(kappa_measured) else None,
        kappa_u=_scalar(kappa_u),
        n_theta=361,
        error_floor=ERROR_FLOOR,
        overflow=bool(np.max(np.abs(np.concatenate([ix, iy]))) > ELLIPSE_HALF_WIDTH),
    )
    fr.set_strings(
        family=family,
        norm=norm_label,
        note=_ellipse_note(family, norm_label, kappa_measured, kappa_target, amp_now, kappa_u).rstrip("。")
        + f"；当前矩阵 A=[{A[0, 0]:.3g} {A[0, 1]:.3g}; {A[1, 0]:.3g} {A[1, 1]:.3g}]"
        + (
            "；" + "、".join(undefined_names) + " 没有有限解（矩阵在 double 里奇异或超出表示范围），对应的箭头不画，其余量照常显示。"
            if undefined_names
            else ""
        ),
        norm_note=(
            f"本帧的 κ 用 <{norm_label}> 范数：κ₁={_fmt_kappa(kappas['1'])}、"
            f"κ₂={_fmt_kappa(kappas['2'])}、κ∞={_fmt_kappa(kappas['inf'])}、"
            f"κ_F={_fmt_kappa(kappas['fro'])}——不同范数可以差一个 n 因子，"
            "图上任何「条件数」都要连范数一起读。"
        ),
        floor_note="副轴是对数轴：实测放大比小于 u/2 时按地板 u/2 显示（不会出现 log(0)）。",
    )
    return fr


def _ellipse_note(
    family: str, norm_label: str, kappa: float, target: float, amp: float, kappa_u: float
) -> str:
    if not np.isfinite(kappa):
        return (
            f"{family} 族：目标 κ={target:.3g}。在 double 里 A 已经被舍入成奇异矩阵"
            "（例如两行完全重合），条件数没有有限值；此时不谈“放大多少倍”，只能看到它已经不可解。"
        )
    if kappa_u >= 1.0:
        return (
            f"κu = {kappa_u:.2e} ≥ 1：A 在 float64 里已经和奇异矩阵无法区分，"
            "此时椭圆与实测曲线都只是舍入的产物，不要再读具体数字。"
        )
    if not np.isfinite(amp):
        return (
            f"{family} 族：目标 κ={target:.3g}，实测 κ_{norm_label}={kappa:.3g}；"
            "这个方向上扰动被奇异方向吞掉，放大比没有有限值。"
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

    def err_text(value: float) -> str:
        return f"{value:.3e}" if value != 0.0 else "0"

    def exact_suffix(value: float) -> str:
        return "；double 里精确为 0" if value == 0.0 else ""

    return (
        f"big={big:.3g}, small={small:.3g}, n={n}, 补偿方式 {scheme}：k={k} 时朴素误差 "
        f"{err_text(traj.error_naive[idx])}、补偿 {err_text(err_scheme)}（相对朴素 {ratio:.3g}{exact_suffix(err_scheme)}）、"
        f"串行 Kahan {err_text(traj.error_kahan[idx])}（本次归约 / 串行 = {par:.3g}{exact_suffix(traj.error_kahan[idx])}）、"
        f"成对 {err_text(traj.error_pairwise[idx])}{exact_suffix(traj.error_pairwise[idx])}"
    )
