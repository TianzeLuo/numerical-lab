"""1.1 章：浮点数、误差传播与问题条件。

四个可视化沿一条主线排开：**浮点网格长什么样 → 误差怎么被放大 → 问题有多难 → 算法能救多少**。

1. ``float_zoom`` —— 数轴放大镜。把数轴放大到 :math:`u` 量级，直接看浮点网格**不均匀**
   （跨过 2 的幂时间距减半），并标出 ``0.1+0.2`` 的真实落点与 ``0.1+0.2-0.3`` 的残差。
   时间轴是"``0.1`` 累加 k 次"的残差演化——残差恰好为 0 的点用 :math:`u/2` 量级的地板显示。
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


def _decimal_center(mantissa: float, exp10: int) -> float:
    """按**十进制字面量**取最近的 double。

    为什么不写 ``mantissa * 10.0 ** exp10``：``3.0 * 0.1`` 会先算出
    ``0.30000000000000004``，于是"放大中心 = 0.3"这件事本身就是错的，
    ``0.1+0.2`` 的落点会正好落在中心上，图上什么都看不出来。
    """
    return float(f"{float(mantissa)!r}e{int(exp10)}")


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
            "title": "0.1+0.2 为什么不等于 0.3：放大数轴看舍入",
            "description": (
                "计算机能表示的数并不连续。把数轴放大到相邻两个 double 的距离，就能看到这一点："
                "这段距离叫一个 ulp（unit in the last place）。上图把相邻浮点数的间距画成阶梯，"
                "实线以当前参数为中心，虚线跨过最近的 2 的幂；每跨过一次 2 的幂，间距就减半。"
                "红点是 0.1+0.2 的实际落点与数学上的 0.3，两者相差一个 ulp。"
                "下图让 0.1 累加 k 次，看结果与 k/10 的差怎样随 k 增大——每一次舍入都会留下一点误差。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "center_log10",
                    "label": "放大中心（10 的幂）",
                    "type": "int",
                    "min": -16,
                    "max": 16,
                    "default": -1,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "mantissa",
                    "label": "放大中心的尾数",
                    "type": "float",
                    "min": 1.0,
                    "max": 9.9,
                    "step": 0.1,
                    "default": 3.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "zoom_log10",
                    "label": "放大倍数 log10（显示的相邻浮点数个数）",
                    "type": "float",
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "default": 1.3,
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
                    "title": "窗口内相邻浮点数的间距，以及跨过 2 的幂时的变化",
                    "ylabel": "间距（ulp）",
                },
                {
                    "id": "residual",
                    "autoscaled": True,
                    "title": "0.1 累加 k 次与精确值 k/10 的差",
                    "ylabel": "残差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "其一：纵轴画的是阶梯，不是连续曲线。每跨过 2 的幂，相邻浮点数的间距减半；"
                "放大不够时它会看起来像一条水平线，别因此认为浮点网格是均匀的。"
                "其二：0.1+0.2 与 0.3 差一个 ulp，是二进制表示和舍入的必然结果，不是加法器出错；"
                "0.1 本身就不能被 double 精确表示。其三：下面一张用对数轴显示残差，"
                "恰好为 0 的步会被地板抬起，不能把它读成“误差不为零”。"
            ),
            "theory": [
                "fl(x) = x(1+δ)，|δ| ≤ u = 2^-53；ulp(x) 在每个 binade 边界减半（网格不均匀）",
                "0.1+0.2 = 0.30000000000000004，与 0.3 恰差 1 个 ulp；0.1+0.2−0.3 = 5.55e-17",
                "累加 k 次 0.1 的残差 ~ k·u/2，误差随项数线性增长（Wilkinson 求和界）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "cancellation",
            "title": "两个相近数相减：有效数字为什么会消失",
            "description": (
                "把 a 和 b 调得越来越近，再做 a−b。两个数本身都只有约 16 位十进制有效数字；"
                "相减时前面相同的位互相抵消，剩下的结果很短，输入里很小的误差在结果中就会占很大比例。"
                "上图数一数还剩多少位可信；下图把放大后的相对误差与理论上界画在一起。"
                "注意：减法本身通常是精确的，消失的是输入本来就有的精度。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "a",
                    "label": "被减数 a",
                    "type": "float",
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.001,
                    "default": 1.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "b",
                    "label": "减数 b（拖到与 a 靠近）",
                    "type": "float",
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.001,
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
                        "d = 15.95 = log10(2^53) 是 binary64 能可靠表达的十进制位数上限；"
                        "把 d 调到它以下，图上的「剩余位数」与理论上界会同步变差——"
                        "这就是把「机器单位舍入 u」与「输入有 d 位有效数字」混为一谈的代价。"
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
            "time": {"mode": "scrub", "tMax": 16.0, "dt": 0.25, "label": "损失位数 t = log10(M/|a−b|)"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "digits",
                    "autoscaled": False,
                    "title": "相减后还剩多少位十进制有效数字（横轴 = 损失位数 t）",
                    "ylabel": "剩余位数",
                },
                {
                    "id": "error",
                    "autoscaled": True,
                    "title": "结果的相对误差与理论上界（横轴 = 损失位数 t）",
                    "ylabel": "相对误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "其一：相减后误差变大，不代表减法算错了。两个相近数相减在浮点里通常精确，"
                "真正被放大的是输入本来就带的误差。其二：上图是“还剩几位有效数字”的先验估计，"
                "不是这一次计算的实测误差；具体误差看下图。其三：把 b 拖到与 a 完全相等时，"
                "程序会换到相邻的 ulp 以避免取对数；这时的点是退化情况，不能当作一般结论。"
            ),
            "theory": [
                "减法的条件数 cond(a−b) = (|a|+|b|)/|a−b|；相对误差 ≤ cond·δ_in",
                "Sterbenz 引理：b/2 ≤ a ≤ 2b 时 fl(a−b) 精确，误差全部来自输入",
                "剩余有效位数 ≈ log10(|a−b|/(δ_in·max(|a|,|b|)))，每抵消 10 倍少一位",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "cond_ellipsoid",
            "title": "条件数 κ(A)：输入的小误差会被放大多少倍",
            "description": (
                "解方程 Ax=b 时，条件数 κ(A) 衡量“b 的小扰动会把解 x 放大多少”。"
                "画面中单位圆经过 A 变成椭圆，长轴与短轴之比就是 κ₂；κ 越大，椭圆越扁，"
                "解对输入方向越敏感。拖动时间轴改变 κ，观察右端项被扰动后解点的跳动；"
                "下图比较当前方向上的实测放大倍数与最坏情况的上界 κ。"
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
                "其一：κ(A) 是最坏情况的上界，不一定在某个具体方向上达到；"
                "实测放大比低于 κ 是正常的。其二：κ 很大时，单位圆会被拉成很扁的椭圆，"
                "图中可能冲出显示范围；看不到不等于不存在，状态标签会说明。"
                "其三：当 κ 接近 1/u≈9×10¹⁵ 时，A 在 double 中已经和奇异矩阵难以区分，"
                "后面的曲线主要由舍入噪声决定。"
            ),
            "theory": [
                "单位球在 A 下的像半轴 = σmin/σmax，半轴之比 = κ₂(A) = ‖A‖₂‖A⁻¹‖₂",
                "前向误差界 ‖Δx‖/‖x‖ ≤ κ(A)·(δA+δb)/(1−κ·δA)，本图 δA = 0",
                "κ₁/κ₂/κ∞/κ_F 可以相差一个 n 因子，图上必须标注用的是哪个范数",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "summation_order",
            "title": "加法顺序为什么影响结果：朴素、Kahan、成对与排序",
            "description": (
                "同样一组数，换个加法顺序，结果可能不同。上图是 big、small、small… 的项序列；"
                "下图比较朴素求和、Kahan 补偿求和、成对求和与排序求和的误差。"
                "Kahan 把每一步的舍入损失记下来再加回去，所以通常更准；但把它分段并行、"
                "段间再用朴素加法合并，误差会重新变大。把 small 调小、项数调多，差异最明显。"
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
                "朴素左折叠：|Δ| ≤ n·u·Σ|x_i|（误差随项数线性增长）",
                "Kahan 补偿：|Δ| ≤ (2u + O(nu²))·Σ|x_i|（与 n、与顺序都无关，但不能分段后朴素合并）",
                "成对求和：|Δ| ≤ u·⌈log2 n⌉·Σ|x_i|；排序后朴素求和有同量级的界",
                "并行分段补偿：误差退化为 P·u·Σ|段和|，段间那一次朴素加法是瓶颈",
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
def _float_zoom(params: Mapping[str, Any], t: float) -> Frame:
    center = _decimal_center(float(params["mantissa"]), int(params["center_log10"]))
    zoom_log10 = float(params["zoom_log10"])
    half = max(1, min(2048, int(round(0.5 * (10.0**zoom_log10)))))
    pts = fp.neighborhood(center, half=half)
    offsets = np.asarray([fp.signed_ulp_offset(center, float(v)) for v in pts], dtype=float)
    sp_center = fp.spacing(center)
    spacing_ratio = fp.spacings(pts) / sp_center
    # 网格不均匀在"2 的幂"处最明显（跨过它间距减半），所以除了用户选的中心，
    # 再给一条以最近的 2 的幂为中心的梯子：两条梯子横轴都是"距各自中心的 ulp 偏移"。
    boundary = 2.0 ** float(np.floor(np.log2(abs(center))))
    bpts = fp.neighborhood(boundary, half=half)
    boffsets = np.asarray([fp.signed_ulp_offset(boundary, float(v)) for v in bpts], dtype=float)
    sp_boundary = fp.spacing(boundary)
    bspacing_ratio = fp.spacings(bpts) / sp_boundary

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

    # ---- 0.1+0.2 的落点：它离 0.3 恰好 1 个 ulp ----------------------------- #
    landing = float(0.1) + float(0.2)
    ideal = 0.3
    lo = float(pts[0])
    hi = float(pts[-1])
    in_window = bool(lo <= landing <= hi)
    ideal_in_window = bool(lo <= ideal <= hi)
    # 窗口外的落点**没有** ulp 偏移可写：这里写 NaN 会被帧契约拒绝（数组也必须有限，
    # 而且前端拿到 NaN 会静默丢段）。所以数组里写 0.0 当占位符，同时把
    # `landing_in_window`/`ideal_in_window` 放进 meta、在 strings 里说明"0 是占位符"。
    landing_off_meta = float(fp.signed_ulp_offset(center, landing)) if in_window else None
    ideal_off_meta = float(fp.signed_ulp_offset(center, ideal)) if ideal_in_window else None
    landing_off = 0.0 if landing_off_meta is None else landing_off_meta
    ideal_off = 0.0 if ideal_off_meta is None else ideal_off_meta
    landing_label = "0.1+0.2 的落点" if in_window else "0.1+0.2 的落点（窗口外：占位 0）"
    ideal_label = "数学上的 0.3" if ideal_in_window else "数学上的 0.3（窗口外：占位 0）"
    residual_0102_03 = float((float(0.1) + float(0.2)) - 0.3)

    # 1e16 + 1：必须显式说明"图上看起来没变"不是画错
    big = float(1e16)
    big_plus_one = big + 1.0
    big_unchanged = bool(big_plus_one == big)

    fr = Frame(chapter="1.1", viz="float_zoom", kind="frame")
    fr.add("offset", offsets, unit="ulp", label="距中心的 ulp 偏移")
    fr.add("value", pts, unit="1", label="可表示数（真实 double）")
    fr.add("spacing_ratio", spacing_ratio, unit="1", label="间距 / ulp(center)")
    fr.add("boundary_offset", boffsets, unit="ulp", label="距最近的 2 的幂的 ulp 偏移")
    fr.add("boundary_value", bpts, unit="1", label="2 的幂附近的可表示数")
    fr.add("boundary_spacing_ratio", bspacing_ratio, unit="1", label="间距 / ulp(2 的幂)：跨过它减半")
    fr.add("center_offset", np.array([0.0]), unit="ulp", label="放大中心")
    fr.add("center_y", np.array([0.0]), unit="1", label="放大中心")
    fr.add("landing_offset", np.array([landing_off]), unit="ulp", label=landing_label)
    fr.add("landing_y", np.array([1.0]), unit="1", label="0.1+0.2 的落点")
    fr.add("ideal_offset", np.array([ideal_off]), unit="ulp", label=ideal_label)
    fr.add("ideal_y", np.array([1.0]), unit="1", label="数学上的 0.3")
    fr.add("k_grid", np.arange(ACC_COUNT + 1, dtype=float), unit="1", label="累加次数 k")
    fr.add("residual", residual, unit="1", label="与数学值 k/10 的残差（精确 0 以 u/2·|k/10| 显示）")
    fr.add("residual_sum_only", residual_sum_only, unit="1", label="纯求和误差：与 k·fl(0.1) 的残差")
    fr.add("residual_floor", residual_floor, unit="1", label="对数轴的显示地板")
    fr.add("sum_value", sums, unit="1", label="fl(Σ0.1) 的部分和")
    fr.add("residual_mark", np.array([float(residual[k])]), unit="1", label="当前位置的残差")
    fr.add("k_mark", np.array([float(k)]), unit="1", label="当前位置的累加次数")
    fr.set_meta(
        center=center,
        center_log10=int(params["center_log10"]),
        mantissa=float(params["mantissa"]),
        zoom_log10=zoom_log10,
        n_points=int(pts.size),
        spacing_center=sp_center,
        spacing_min=float(np.min(spacing_ratio)),
        spacing_max=float(np.max(spacing_ratio)),
        boundary=boundary,
        boundary_spacing_min=float(np.min(bspacing_ratio)),
        boundary_spacing_max=float(np.max(bspacing_ratio)),
        boundary_ulps=half,
        window_lo=lo,
        window_hi=hi,
        window_ulps=half,
        landing_ulps=landing_off_meta,
        landing_in_window=in_window,
        ideal_ulps=ideal_off_meta,
        ideal_in_window=ideal_in_window,
        marker_placeholder=0.0,
        residual_0102_03=residual_0102_03,
        residual_0102_03_ulps=float(fp.residual_ulps(landing, ideal)),
        residual_floor=float(residual_floor[k]),
        residual_sum_only_mark=float(residual_sum_only[k]),
        residual_floor_min=float(np.min(residual_floor)),
        residual_exact_zero_steps=n_exact_zero,
        residual_sum_only_exact_zero_steps=n_exact_zero_sum,
        k=k,
        n_acc_max=ACC_COUNT,
        big_plus_one=big_plus_one,
        big_plus_one_unchanged=big_unchanged,
        overflow=bool(not in_window),
    )
    fr.set_strings(
        note=_zoom_note(center, sp_center, int(pts.size), landing_off, in_window, k, float(raw[k])),
        exact_center=fp.exact_decimal(center, digits=24),
        exact_landing=fp.exact_decimal(landing, digits=24),
        big_note=(
            "1e16 + 1 = 1e16：ulp(1e16) = 2，加 1 恰好落在两个可表示数的中点，"
            "ties-to-even 把它舍回偶数格点。图上「看起来没变」不是画错——把放大中心移到 1e16 就能看见间距变成 2。"
        ),
        zero_note=(
            f"副轴是对数轴：残差恰好为 0 的步数（{n_exact_zero} 步）被地板 u/2·|k/10| 抬起显示，"
            "不能读成「误差不为零」。"
        ),
        marker_note=(
            "落点标记（landing_offset / ideal_offset）在**窗口外**时数组里写 0，"
            "同时 meta.landing_in_window / ideal_in_window 为 false——0 只是占位符，"
            "不要读成「落点就在放大中心」。帧里不写 NaN：NaN 会在前端静默丢段。"
        ),
    )
    return fr


def _zoom_note(
    center: float, sp: float, n_points: int, landing_off: float, in_window: bool, k: int, raw_k: float
) -> str:
    if not in_window:
        return (
            f"0.1+0.2 的落点不在当前窗口内（窗口只有 {n_points} 个可表示数）——"
            "「看不见」不等于「不在那里」，把放大中心调到 0.3 再看。"
        )
    if abs(landing_off) < 0.5:
        return f"0.1+0.2 正好落在放大中心上（与中心相差 {abs(landing_off):.0f} 个 ulp）"
    return (
        f"0.1+0.2 落在中心之上 {landing_off:.0f} 个 ulp；ulp(center)={sp:.3e}，"
        f"窗口共 {n_points} 个可表示数；k={k} 时 0.1 累加的精确残差是 {raw_k:.3e}"
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
    t_grid = np.linspace(1.0, LOSS_MAX, 31)
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
    head = "a 与 b 完全相等，已用相邻的 1 个 ulp 代替 b。" if degenerate else ""
    return (
        f"{head}a−b = {gap:.6e}，输入不确定度 δ_in = {delta_in:.2e}，"
        f"先验剩余有效位 {digits:.2f} 位，实测放大后相对误差 {err:.3e}，"
        f"而减法本身的舍入误差是 {round_err:.1e}"
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
