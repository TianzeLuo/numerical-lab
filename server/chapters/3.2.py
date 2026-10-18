"""3.2 章：稳定性与刚性方程（5 个可视化）。

帧只做搬运：所有数学量都来自 :mod:`numcore.ode.stability` 与 :mod:`numcore.ode.implicit`，
本文件不写数学。五个可视化的分工是"从域到代价"：

1. ``stability_region`` —— **招牌**：复平面上画出 :math:`|R(z)|=1` 的边界（精确参数化，
   不是网格等值线），叠加当前 :math:`z=\\lambda h` 落点与整条 :math:`\\lambda h` 扫描射线；
   时间轴是 :math:`\\log_{10}h`，拖到射线穿出稳定域之外，**真跑出来的解曲线**同步发散。
2. ``stiff_dilemma`` —— 刚性困境：解是 :math:`\\cos t` 那样平缓的曲线，显式方法却被
   :math:`h\\le c/|\\lambda|` 卡住；时间轴是真实时间 :math:`t`，副轴给显式方法的误差。
3. ``stability_surface`` —— :math:`|R|` 的模曲面（二维数组直接给前端）+ :math:`|R|=1` 等高线，
   参数是方法与阶数；副轴给"边界自洽残差" :math:`\\bigl||R(z_\\theta)|-1\\bigr|`。
4. ``explicit_vs_implicit`` —— 三向对比（步长 / 误差 / 计算量）：时间轴是真实时间，
   四根柱子随 :math:`t` 长大，看得见"隐式每步更贵、总账更便宜"。
5. ``bdf_angle`` —— BDF 的 A(:math:`\\alpha`)-稳定角：时间轴是**测试射线角** :math:`\\alpha`，
   拖阶数 :math:`k=1\\ldots6` 看角度从 90° 收缩到 17.84°。

帧契约（与 1.2 章一致）：``meta`` 只放有限标量（``allow_nan=False``，非有限值一律
换成哨兵 + 显式布尔标志），字符串走 ``strings``，数组走 ``add``。时间轴的语义逐个写清：
viz 1 是 :math:`\\log_{10}h` 扫描、viz 2/4 是真实时间、viz 5 是角度扫描、viz 3 无时间轴。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from numcore.ode import implicit as imp
from numcore.ode import stability as stab
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 单位舍入，对数轴地板与"精确 0"的显示约定都用它（与 1.2 章同口径）。
_ROUNDOFF = float(np.finfo(float).eps) / 2.0

#: 复平面固定窗口：显式方法的稳定域整体落在这里，隐式方法的"域外部分"也看得见。
_WIN_RE = (-8.0, 4.0)
_WIN_IM = (-6.0, 6.0)

#: 步长扫描范围（viz 1 的时间轴）：:math:`\\log_{10}h\\in[-3,1]`。
_LOG10_H_LO, _LOG10_H_HI = -3.0, 1.0

#: 显式解的发散显示上限（"软着陆"）：超过它就截断显示，真值写在 meta 里。
_DISPLAY_CEIL = 1.0e6


def _fin(x: Any, default: float = 0.0) -> float:
    """把非有限标量换成哨兵：``.nlf`` 的 header 用 ``allow_nan=False``，NaN/inf 直接写不进去。"""
    v = float(x)
    return v if math.isfinite(v) else float(default)


def _floor(arr: Any, floor: float = _ROUNDOFF) -> np.ndarray:
    """对数轴的「软着陆」：把 0（以及非有限值）抬到明确的地板上并如实报告。"""
    a = np.asarray(arr, dtype=float).copy()
    a[~np.isfinite(a)] = floor
    return np.maximum(a, floor)


#: 极点（|R|=+∞）在帧里的占位值：与显示上限同值——"超出色标"就该看着超出色标，
#: 而不是被换成 0（那会被读成"非常稳定"，正好读反）。
_POLE_FILL = _DISPLAY_CEIL

#: 帧里"非有限值怎么办"的统一口径，写进每帧的 ``strings.nonfinite_policy``，
#: 并被 ``tests/3.2`` 的 B 类断言逐帧核对。
_NONFINITE_POLICY = (
    "本帧不含任何非有限值：|R|=+∞ 的极点位置用 strings.pole_fill（={fill:g}）占位并配 *_pole 掩码数组；"
    "折线不用 NaN 分隔（改为回到起点再画下一段，NaN 会让前端静默丢段）；"
    "meta 一律是有限标量（.nlf 头用 allow_nan=False 编码，塞一个 NaN 整帧就写不出去），"
    "数组段虽然是裸 float64、NaN 编码得进去，但帧契约要求每个数组都有限"
)


def _finite_or(arr: Any, fill: float = _POLE_FILL) -> tuple[np.ndarray, np.ndarray, int]:
    """把非有限元显式换成 ``fill``，返回 ``(干净数组, 极点掩码, 个数)``。

    为什么必须显式处理：``.nlf`` 的 header 用 ``allow_nan=False`` 编码，meta 里一个 NaN
    就让**整帧**写不出去；数组里的 ±Inf 到了前端会变成"整片空白"或"一根竖线"，
    而 NaN 在折线里意味着"断开"。换值的同时给出掩码，是为了让"看不见"与"不存在"
    分开（防误导规则 4）：这些格子不是值很小，是**那里没有有限值**。
    """
    a = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(a)
    n = int(np.count_nonzero(bad))
    if n:
        a = np.where(bad, float(fill), a)
    return a, bad.astype(float), n


def _step_of(t: float, n_steps: float) -> int:
    """把连续时间轴四舍五入到整数步并夹住（与 1.2 章 ``_step_of`` 同口径）。"""
    k = int(math.floor(float(t) + 0.5))
    return max(0, min(k, int(n_steps)))


def _clip_scan(t: float, lo: float, hi: float) -> float:
    """把时间轴夹到扫描区间 [lo, hi]（「拖到头就不动了」，不是卡住）。"""
    return max(float(lo), min(float(hi), float(t)))


def _stab_key(method: str) -> tuple[str, int]:
    """把 viz 的方法名翻成 :mod:`numcore.ode.stability` 的 ``(方法, 阶)``。"""
    return {
        "forward_euler": ("forward_euler", 1),
        "rk4": ("rk4", 4),
        "implicit_euler": ("implicit_euler", 1),
        "trapezoidal": ("trapezoidal", 2),
        "bdf2": ("bdf", 2),
    }[method]


def _integ_key(method: str) -> tuple[str, int]:
    """把 viz 的方法名翻成 :mod:`numcore.ode.implicit` 的 ``(方法, 阶)``。"""
    return {
        "forward_euler": ("forward_euler", 1),
        "rk4": ("rk4", 4),
        "implicit_euler": ("implicit_euler", 1),
        "trapezoidal": ("trapezoidal", 2),
        "bdf2": ("bdf2", 2),
    }[method]


def _method_name(method: str) -> str:
    return {
        "forward_euler": "前向 Euler",
        "rk4": "经典 RK4",
        "implicit_euler": "隐式 Euler",
        "trapezoidal": "梯形法",
        "bdf2": "BDF2",
    }.get(method, method)


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
MANIFEST: dict[str, Any] = {
    "chapter": "3.2",
    "title": "稳定性与刚性方程",
    "part": "III",
    "summary": (
        "线性测试方程 y'=λy 把「一个方法稳不稳」压缩成复平面上的一个区域；"
        "把 λh 拖出这个区域，数值解真的会炸。刚性问题的全部戏剧性就在"
        "「解很平缓、步长却由 λ_max 决定」这一句话里。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "stability_region",
            "title": "稳定域交互图：λh 落点与解的真实发散",
            "description": (
                "复平面上固定坐标画出当前方法的绝对稳定域边界 |R(z)|=1（精确参数化），"
                "灰蓝热力图是 |R(z)|。橙点/橙线是当前 λh 与整条 h 扫描射线；"
                "时间轴就是 log10 h：向右拖 h 变大，射线上的 λh 点向外走，"
                "一旦穿出边界，主图右侧那条**真算出来的解曲线**立刻开始发散。"
                "副轴同时给 |R(λh)| 与数值解的误差。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "method",
                    "label": "方法",
                    "type": "enum",
                    "options": ["forward_euler", "rk4", "implicit_euler", "trapezoidal", "bdf2"],
                    "default": "rk4",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "lam_re",
                    "label": "λ 的实部",
                    "type": "float",
                    "min": -60.0,
                    "max": 10.0,
                    "step": 0.5,
                    "default": -6.0,
                    "critical": [0.0],
                    "criticalNote": (
                        "Re λ = 0 是「真解本身既不衰减也不增长」的分界：越过它以后真解在增长，"
                        "数值解的上升不再等于方法不稳定——判据要分开看。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "lam_im",
                    "label": "λ 的虚部",
                    "type": "float",
                    "min": -40.0,
                    "max": 40.0,
                    "step": 0.5,
                    "default": 12.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "t_end",
                    "label": "积分到 t",
                    "type": "float",
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "default": 4.0,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 4.0, "dt": 0.05, "label": "log10 h（步长扫描）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "复平面 z = λh（固定窗口，域外不裁剪）",
                    "ylabel": "Im z",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "|R(λh)|（右轴）与数值解误差（左轴对数）",
                    "ylabel": "|R| 与误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，这张图**故意固定坐标**（Re∈[−8,4]、Im∈[−6,6]）：改成自适应，"
                "显式方法那个有界的稳定域和隐式方法那个无限大的稳定域就会被画成差不多大小，"
                "「谁更稳」当场消失。第二，λh 落在域外**不等于**数值解一定错：λ 的实部为正时"
                "真解本身在增长，曲线上升是事实而非算法故障；判据是 |R|>1 与误差两条一起看。"
                "第三，热力图的网格分辨率会漏掉细窄的稳定域（BDF 的高阶域像一条细带），"
                "所以边界线是**精确参数化**画出来的，不是从网格上爬出来的等高线。"
            ),
            "theory": [
                "绝对稳定域 S = {z ∈ C : |R(z)| ≤ 1}；z = λh 落在 S 内则该模态在数值上不放大",
                "显式 RK 的 R(z) = 1 + Σ_{j≥1}(bᵀA^{j−1}e)zʲ 由 Butcher 表算出；隐式 Euler 是 1/(1−z)，梯形法是 (1+z/2)/(1−z/2)",
                "|R(z)| = 1 的边界对多步法就是 z(θ) = ρ(e^{iθ})/σ(e^{iθ})（本章直接参数化，不用等值线追踪）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "stiff_dilemma",
            "title": "刚性困境：解那么平缓，步长却被 λ_max 掐住",
            "description": (
                "解 y' = −λ(y − cos t) − sin t 的精确解是 cos t（快模态系数恰好为 0），"
                "平缓得不能再平缓；可显式方法只能走 h ≤ c/λ，副轴的误差曲线就是它的代价。"
                "主图画精确解、显式解与隐式解三条曲线，并用密集/稀疏的步点标出两者各自的步长；"
                "时间轴是真实时间 t。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "stiffness_ratio",
                    "label": "刚性比 λ_max/λ_min",
                    "type": "float",
                    "min": 20.0,
                    "max": 1000000.0,
                    "step": 20.0,
                    "default": 1000.0,
                    "critical": [3183.0],
                    "criticalNote": (
                        "临界值来自「评估预算」：t_end = 2π 上要 10^4 次函数求值，显式方法每步至少"
                        "一次求值，于是 r ≤ 2·10^4/(2π) ≈ 3183。越过它，显式方法在同样的预算下"
                        "连一个周期都走不完（副轴的步数/求值次数会把这件事摆出来）。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "explicit_method",
                    "label": "显式方法",
                    "type": "enum",
                    "options": ["forward_euler", "rk4"],
                    "default": "forward_euler",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "implicit_method",
                    "label": "隐式方法",
                    "type": "enum",
                    "options": ["implicit_euler", "trapezoidal", "bdf2"],
                    "default": "trapezoidal",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "implicit_steps",
                    "label": "隐式步数（一个周期内）",
                    "type": "int",
                    "min": 20,
                    "max": 400,
                    "default": 100,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 6.2831853, "dt": 0.062831853, "label": "时间 t（一个周期）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "解曲线（固定纵轴 ±2）与两种方法的步点",
                    "ylabel": "y(t)",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "显式方法误差 |y_num − cos t| 与步长比",
                    "ylabel": "误差 / 步长",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "最大的骗局是「看起来还行」：显式方法在前几十步里误差只有 1e-3 量级，"
                "曲线也贴着 cos t，直到某一步突然整体抬起来——那是快模态被 |R|>1 放大到了 O(1)。"
                "所以判断「能不能走」要看**步数/求值次数**，不能看「当前误差还挺小」。"
                "第二个坑是把刚性比当成「解有多陡」：本例的解永远平缓（|y''|≤1），刚性说的是"
                "**解里存在一个极快衰减的模态**，它未必在解上看得见——把初始条件改一点点"
                "（y(0)=1+c₀），那个 O(1/λ) 宽的边界层就会露出来。第三个坑是把 λ 增大当成"
                "「更难算」：λ 只改步长天花板，解的形态一个字都没变。"
            ),
            "theory": [
                "显式方法在负实轴上的步长天花板 h ≤ c/|λ|：forward Euler 的 c = 2，RK4 的 c = 2.785293…（x³+4x²+12x+24=0 的实根）",
                "刚性比 r = λ_max/λ_min：稳定性限制步长 ~ c/λ_max，精度限制步长 ~ O(1)（由解的光滑尺度决定）",
                "总代价 = 步数 × 每步代价：隐式每步要解线性系统（Newton + LU），但步数可以少几个数量级",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "stability_surface",
            "title": "|R(z)| 模曲面与 |R|=1 等高线",
            "description": (
                "复平面网格上的 |R(z)| 直接作为二维数组给前端上色（灰蓝=衰减、暖色=放大），"
                "并叠加精确参数化的 |R(z)|=1 边界。参数是方法与阶数：BDF 拖到 k≥3 会看到"
                "稳定域从「整个左半平面」缩成一条贴着实轴的窄带。副轴给边界的自洽残差"
                "||R(z_θ)|−1|（应该是 1e-15 量级）与两条独立计算路径之差。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "method",
                    "label": "方法族",
                    "type": "enum",
                    "options": ["forward_euler", "explicit_taylor", "rk4", "implicit_euler", "trapezoidal", "bdf"],
                    "default": "bdf",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "order",
                    "label": "阶数 k（BDF 与显式 Taylor 族使用）",
                    "type": "int",
                    "min": 1,
                    "max": 6,
                    "default": 2,
                    "critical": [3],
                    "criticalNote": (
                        "Dahlquist 第二障碍：A-稳定的线性多步法阶数 ≤ 2，所以 BDF 在 k=3 处"
                        "第一次失去 A-稳定性（稳定角从 90° 掉到 86.03°）。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "half_width",
                    "label": "窗口半宽",
                    "type": "float",
                    "min": 2.0,
                    "max": 10.0,
                    "step": 0.5,
                    "default": 6.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "grid_n",
                    "label": "网格点数（每边）",
                    "type": "int",
                    "min": 31,
                    "max": 161,
                    "default": 101,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "|R(z)|（固定窗口）与 |R|=1 等高线",
                    "ylabel": "Im z",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "边界残差 ||R|−1|（实线）与双路径之差（虚线）",
                    "ylabel": "残差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "上色用的是 |R| 本身，于是「整片暖色」未必代表算法坏了——z 在右半平面时真解本来"
                "就在涨。另一条更隐蔽：网格分辨率决定了细窄稳定域的可见性，BDF-6 的域在窗口里"
                "只剩一条带，31×31 的网格会直接把它抹掉；所以等高线给的是**精确参数化**的边界，"
                "热力图只负责给颜色。第三，色标的上限会截断：|R| 在窗口角上可以到 1e4，"
                "固定 vmin/vmax 会让「域外发散」看上去只是「稍微偏红」，读数字（副轴）才算数。"
            ),
            "theory": [
                "|R(z)| 的等值线 |R|=c 与稳定域 S={|R|≤1}：单步法直接用 R，多步法用根轨迹的最大根模",
                "多步法的边界 z(θ)=ρ(e^{iθ})/σ(e^{iθ}) 与根轨迹 {r : ρ(r)−zσ(r)=0, |r|=1} 是同一件事的两面",
                "阶数与稳定域的关系：p 级 p 阶显式 RK 的 R 是截断指数 R_p(z)=Σ_{j≤p}zʲ/j!，阶越高域越大但有界；隐式方法可以无界",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "explicit_vs_implicit",
            "title": "显式 vs 隐式：步长 / 误差 / 计算量三向对比",
            "description": (
                "同一个刚性受迫问题上跑两种方法：主轴画解曲线（固定纵轴，溢出显式标记），"
                "副轴（对数）画两者的误差；下方四根柱子给出「到目前为止」的步数、函数求值次数、"
                "LU 分解次数与 Newton 迭代次数。时间轴是真实时间 t：拖着它走，"
                "柱子和误差一起长大，看得见隐式方法每步更贵、总账却便宜。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "tol",
                    "label": "误差容差（决定隐式步长）",
                    "type": "float",
                    "min": 1e-09,
                    "max": 1e-02,
                    "step": 1e-06,
                    "default": 1e-05,
                    "critical": [1e-06],
                    "criticalNote": (
                        "这个数值不是普适临界值，而是「演示尺度」：容差小到 1e-6 以后，"
                        "隐式方法的精度受限步长会缩到与显式方法的稳定受限步长同量级，"
                        "两条代价曲线在图上交叉——这正是「什么时候该用隐式」的转折点。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "stiffness_ratio",
                    "label": "刚性比 λ_max/λ_min",
                    "type": "float",
                    "min": 20.0,
                    "max": 100000.0,
                    "step": 20.0,
                    "default": 1000.0,
                    "critical": [3183.0],
                    "criticalNote": "与 stiff_dilemma 同一把尺子：显式步数 ≈ r·t_end/c，r 越大步数线性爆炸。",
                    "beyond": "above",
                },
                {
                    "name": "explicit_method",
                    "label": "显式方法",
                    "type": "enum",
                    "options": ["forward_euler", "rk4"],
                    "default": "forward_euler",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "implicit_method",
                    "label": "隐式方法",
                    "type": "enum",
                    "options": ["implicit_euler", "trapezoidal", "bdf2"],
                    "default": "trapezoidal",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 6.2831853, "dt": 0.062831853, "label": "时间 t（一个周期）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "解曲线（固定纵轴 ±2）",
                    "ylabel": "y(t)",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "显式/隐式误差（对数）与代价柱状（各族独立刻度）",
                    "ylabel": "误差 / 次数",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "「隐式更贵」只对**每一步**成立：每步一次 LU（O(m³)）+ 若干次 Newton 迭代，"
                "单步代价可能是显式的几十倍；但刚性比一上来，显式步数可以多出三四个数量级，"
                "总账立刻反转。反过来「隐式更快」也不普适：容差很松、问题不刚性时，"
                "隐式那几十倍的单步代价收不回来。第二，误差曲线只画到**当前** t："
                "显式方法在爆炸前误差一直很小，「看起来更准」是假象。第三，柱子的刻度分组是"
                "独立的（步数 vs 求值次数差一个量级），不看轴就会得出「两边差不多」的结论。"
            ),
            "theory": [
                "每步代价：显式 ≈ 1 次函数求值（RK4 是 4 次）；隐式 ≈ Newton 迭代数 × (函数求值 + LU)",
                "总代价 = 步数 × 每步代价；刚性比越大，显式的步数优势越明显地被稳定性限制吃掉",
                "精度受限步长 h_acc 与稳定受限步长 h_stab = c/|λ|：谁小谁说话，交叉点就是「该换方法」的地方",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "bdf_angle",
            "title": "BDF 的 A(α)-稳定角：阶数越高，楔形越窄",
            "description": (
                "把 BDF-k（k=1…6）的稳定域边界画在一起（浅色家族），当前 k 高亮。"
                "青色楔形是以负实轴为轴、半角为 α 的区域；时间轴就是测试角 α："
                "向右拖，楔形张到边界就说明 α 已经超过该阶的 A(α) 角。"
                "副轴给测试射线上的 |R| 超界量 max(0,|R|−1)（对数），以及各阶实测角的对照柱。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "k",
                    "label": "BDF 阶数 k",
                    "type": "int",
                    "min": 1,
                    "max": 6,
                    "default": 3,
                    "critical": [3],
                    "criticalNote": (
                        "k ≥ 3 不再是 A-稳定（Dahlquist 第二障碍：A-稳定的线性多步法阶 ≤ 2）。"
                        "k ≥ 7 更彻底——BDF-k 零稳定 ⟺ k ≤ 6，第 7 阶连收敛性都没有，"
                        "所以本图只到 6。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "show_family",
                    "label": "显示 k=1…6 全家族",
                    "type": "bool",
                    "default": True,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 90.0, "dt": 1.0, "label": "测试角 α（度）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "稳定域边界家族与 A(α) 楔形（固定窗口）",
                    "ylabel": "Im z",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "测试射线上的超界量 max(0,|R|−1) 与各阶实测角",
                    "ylabel": "超界量 / 角度",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，A(α) 角是**含**在稳定域里的最大楔形，不是「边界在左半平面的角度」："
                "BDF-2 的边界整条都在右半闭平面，角度是 90°（A-稳定）；BDF-3 只探进去一点点，"
                "角度就掉到 86.03°——用眼睛比「探进去多少」会严重高估。第二，α 拖到 90° 时"
                "楔形等于整个左半平面，此时「射线没有超界」并不说明方法 A-稳定，"
                "因为 A-稳定要求的是**整个**左半平面（连续统），不是一条射线。"
                "第三，k=1 与 k=2 的边界看起来很不一样（一个是圆、一个是外凸环），"
                "但两者都 A-稳定——稳定域的形状与「稳不稳」不是一回事。"
            ),
            "theory": [
                "A(α)-稳定：楔形 |arg(−z)| ≤ α 整体含于稳定域；A-稳定就是 α = 90° 的特例",
                "BDF-k 的 A(α) 角理论值：90°, 90°, 86.03°, 73.35°, 51.84°, 17.84°（k=1…6）",
                "Dahlquist 第二障碍：A-稳定的线性多步法阶 ≤ 2；BDF-k 零稳定 ⟺ k ≤ 6",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 计算入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "stability_region":
        return _stability_region(params, t)
    if viz_id == "stiff_dilemma":
        return _stiff_dilemma(params, t)
    if viz_id == "stability_surface":
        return _stability_surface(params, t)
    if viz_id == "explicit_vs_implicit":
        return _explicit_vs_implicit(params, t)
    if viz_id == "bdf_angle":
        return _bdf_angle(params, t)
    raise KeyError(f"章节 3.2 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# 公共小工具
# --------------------------------------------------------------------------- #
def _abs_r_grid(method: str, order: int, re_lo: float, re_hi: float, im_lo: float, im_hi: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """复平面网格上的 |R(z)|（二维数组，行主序，形状 ``(ny, nx)``）与 z 网格。"""
    re = np.linspace(re_lo, re_hi, int(nx))
    im = np.linspace(im_lo, im_hi, int(ny))
    zz = re[None, :] + 1j * im[:, None]
    return np.asarray(stab.stability_modulus(method, order, zz), dtype=float), zz


def _explicit_ceiling(method: str, order: int, lam: float) -> tuple[float, float]:
    """显式方法的步长天花板 ``(实测, 理论)``；隐式方法给 ``(inf, inf)``。"""
    return (
        imp.max_stable_step(method, order, -abs(lam)),
        imp.explicit_step_ceiling_theory(method, order, -abs(lam)),
    )


def _two_d_linear_problem(lam_re: float, lam_im: float) -> tuple[np.ndarray, np.ndarray, Any]:
    r"""把复特征值 :math:`\lambda=a+ib` 写成实 :math:`2\times2` 系统 :math:`[[a,-b],[b,a]]`。

    这样"数值解"是实向量（前端能画），而它的每步放大因子**恰好**是复的 :math:`|R(\lambda h)|`
    ——既不是近似也不是另算一套。
    """
    A = np.array([[lam_re, -lam_im], [lam_im, lam_re]], dtype=float)
    y0 = np.array([1.0, 0.0], dtype=float)

    def exact(tt: np.ndarray) -> np.ndarray:
        tt = np.asarray(tt, dtype=float)
        e = np.exp(lam_re * tt)
        return np.stack([e * np.cos(lam_im * tt), e * np.sin(lam_im * tt)], axis=-1)

    return A, y0, exact


# --------------------------------------------------------------------------- #
# viz 1：稳定域交互图
# --------------------------------------------------------------------------- #
def _stability_region(params: Mapping[str, Any], t: float) -> Frame:
    method = str(params["method"])
    lam_re = float(params["lam_re"])
    lam_im = float(params["lam_im"])
    t_end = float(params["t_end"])
    lam = complex(lam_re, lam_im)
    stab_name, stab_order = _stab_key(method)
    integ_name, integ_order = _integ_key(method)

    log10_h = _clip_scan(t, _LOG10_H_LO, _LOG10_H_HI)
    h = 10.0**log10_h

    nx, ny = 121, 101
    abs_r_raw, zz = _abs_r_grid(stab_name, stab_order, _WIN_RE[0], _WIN_RE[1], _WIN_IM[0], _WIN_IM[1], nx, ny)
    # BDF-k 的极点在 z=α_k（k=1→1、k=2→1.5、k=3→11/6…）：网格点正好压上去时 |R|=+∞。
    # 帧里换成占位值 + 掩码，绝不把 Inf 写进去。
    abs_r, pole_mask, n_pole = _finite_or(abs_r_raw)
    finite_cells = abs_r_raw[np.isfinite(abs_r_raw)]
    abs_r_max_finite = float(np.max(finite_cells)) if finite_cells.size else 0.0
    boundary = stab.stability_boundary(stab_name, stab_order, n_theta=721)
    # 边界的自洽残差：||R(z_θ)|−1|，应该贴在舍入底上（它是"边界画对了"的证据）
    residual_boundary = _floor(np.abs(np.abs(stab.stability_modulus(stab_name, stab_order, boundary)) - 1.0))

    # λh 扫描射线：h 从 10^lo 到 10^hi；点落在窗口外时如实裁剪 + 溢出标记
    h_scan = np.exp(np.linspace(math.log(10.0**_LOG10_H_LO), math.log(10.0**_LOG10_H_HI), 241))
    z_scan = lam * h_scan
    abs_r_scan, ray_pole_mask, n_ray_pole = _finite_or(
        np.asarray(stab.stability_modulus(stab_name, stab_order, z_scan), dtype=float)
    )
    ray_inside = (np.real(z_scan) >= _WIN_RE[0]) & (np.real(z_scan) <= _WIN_RE[1]) & (np.imag(z_scan) >= _WIN_IM[0]) & (np.imag(z_scan) <= _WIN_IM[1])

    current_z = lam * h
    abs_r_current = float(np.max(stab.stability_modulus(stab_name, stab_order, current_z)))
    current_is_pole = not math.isfinite(abs_r_current)
    point_visible = bool(
        _WIN_RE[0] <= current_z.real <= _WIN_RE[1] and _WIN_IM[0] <= current_z.imag <= _WIN_IM[1]
    )

    # 沿射线的"出域步长"：最小的 h 使 |R(λh)|>1（隐式方法在指向左半平面的射线上可能没有）
    h_exit = 0.0
    h_exit_unbounded = True
    for hh in h_scan:
        if float(np.max(stab.stability_modulus(stab_name, stab_order, lam * hh))) > 1.0:
            lo, hi = 0.0, float(hh)
            for _ in range(70):
                mid = 0.5 * (lo + hi)
                if float(np.max(stab.stability_modulus(stab_name, stab_order, lam * mid))) > 1.0:
                    hi = mid
                else:
                    lo = mid
            h_exit = 0.5 * (lo + hi)
            h_exit_unbounded = False
            break

    # 真跑一遍：复特征值写成实 2×2 系统
    A, y0, exact_fn = _two_d_linear_problem(lam_re, lam_im)
    n_plan = int(math.ceil(t_end / h)) if h > 0 else 0
    sol = imp.integrate_linear_system(integ_name, A, (0.0, t_end), y0, h, order=integ_order, max_steps=20000)
    tt = np.asarray(sol.t, dtype=float)
    y_num = np.asarray(sol.y, dtype=float)
    y_ex = np.asarray(exact_fn(tt), dtype=float)
    err = _floor(np.max(np.abs(y_num - y_ex), axis=1))
    y_mag = np.max(np.abs(y_num), axis=1)
    y_mag_display = np.minimum(y_mag, _DISPLAY_CEIL)

    fr = Frame(chapter="3.2", viz="stability_region", kind="frame")
    fr.add("abs_R", abs_r, unit="1", label="|R(z)|（二维网格，行主序；极点处 = strings.pole_fill）")
    fr.add("abs_R_pole", pole_mask, unit="1", label="极点掩码：1 表示该格 |R|=+∞（用 pole_fill 占位）")
    fr.add("extent", np.array([_WIN_RE[0], _WIN_RE[1], _WIN_IM[0], _WIN_IM[1]], dtype=float), unit="1", label="窗口 x0,x1,y0,y1")
    fr.add("levels", np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=float), unit="1", label="|R| 等值线层级")
    fr.add("boundary_x", np.real(boundary), unit="1", label="稳定域边界 Re z")
    fr.add("boundary_y", np.imag(boundary), unit="1", label="稳定域边界 Im z")
    fr.add("residual", residual_boundary, unit="1", label="边界自洽残差 ||R|−1|")
    fr.add("ray_x", np.real(z_scan), unit="1", label="λh 扫描射线 Re z")
    fr.add("ray_y", np.imag(z_scan), unit="1", label="λh 扫描射线 Im z")
    fr.add("ray_inside", ray_inside.astype(float), unit="1", label="射线点在窗口内标记")
    fr.add("abs_R_ray", abs_r_scan, unit="1", label="沿射线的 |R(λh)|（极点处 = strings.pole_fill）")
    fr.add("abs_R_ray_pole", ray_pole_mask, unit="1", label="射线上的极点掩码（1 = |R|=+∞）")
    fr.add("z_point_x", np.array([current_z.real]), unit="1", label="当前 λh 的 Re")
    fr.add("z_point_y", np.array([current_z.imag]), unit="1", label="当前 λh 的 Im")
    fr.add("t_solution", tt, unit="s", label="时间")
    fr.add("y1_num", y_num[:, 0], unit="1", label="数值解 y₁（实 2×2 系统的第一分量）")
    fr.add("y2_num", y_num[:, 1], unit="1", label="数值解 y₂")
    fr.add("y1_exact", y_ex[:, 0], unit="1", label="精确解 y₁")
    fr.add("y2_exact", y_ex[:, 1], unit="1", label="精确解 y₂")
    fr.add("y_magnitude", y_mag_display, unit="1", label="|y|（发散时截断显示）")
    fr.add("error", err, unit="1", label="数值解误差 max|y_num−y_exact|（0 以 u/2 显示）")
    fr.add("error_mark", np.array([err[-1]], dtype=float), unit="1", label="当前时刻误差")
    fr.set_meta(
        log10_h=_fin(log10_h),
        h=_fin(h),
        h_exit=_fin(h_exit),
        h_exit_unbounded=bool(h_exit_unbounded),
        abs_R_current=_fin(abs_r_current, _POLE_FILL),
        stable=bool(abs_r_current <= 1.0),
        lam_re=_fin(lam_re),
        lam_im=_fin(lam_im),
        z_re=_fin(current_z.real),
        z_im=_fin(current_z.imag),
        t_end=_fin(t_end),
        n_steps=sol.n_steps,
        n_steps_planned=n_plan,
        n_f_evals=sol.n_f_evals,
        n_lu=sol.n_lu,
        dtrial=t_end / max(n_plan, 1),
        error_final=_fin(err[-1]),
        error_max=_fin(np.max(err)),
        max_abs_y=_fin(sol.max_abs_y),
        display_ceiling=_DISPLAY_CEIL,
        blowup=bool(sol.blowup),
        t_blowup=_fin(sol.t_blowup),
        point_visible=bool(point_visible),
        residual_max=_fin(np.max(residual_boundary)),
        roundoff_floor=_ROUNDOFF,
        overflow=bool(sol.blowup or not point_visible),
        a_stable=bool(stab.stability_report(stab_name, stab_order).a_stable),
        abs_R_max_finite=_fin(abs_r_max_finite),
        n_pole_cells=n_pole,
        n_pole_ray=n_ray_pole,
        pole_current=bool(current_is_pole),
        pole_fill=_POLE_FILL,
    )
    fr.set_strings(
        method=method,
        method_name=_method_name(method),
        status="blowup" if sol.blowup else ("out_of_region" if abs_r_current > 1.0 else "stable"),
        note=_region_note(method, abs_r_current, h_exit, h_exit_unbounded, sol.blowup, log10_h),
        display_policy="|y| 超过 1e6 后截断显示；meta.max_abs_y 永远是未截断的真值",
        nonfinite_policy=_NONFINITE_POLICY.format(fill=_POLE_FILL),
        pole_fill=f"{_POLE_FILL:g}",
        pole_note=(
            f"|R|=+∞ 只出现在 G(θ) 的极点（BDF-k 在 z=α_k：1, 1.5, 11/6, 25/12, 137/60, 49/20）："
            f"那里 r^k 的系数退化、一个根跑到无穷远。本帧有 {n_pole} 个网格点、{n_ray_pole} 个射线点落在极点上，"
            f"已用 pole_fill={_POLE_FILL:g} 占位并由 abs_R_pole / abs_R_ray_pole 掩码标出；"
            f"meta.abs_R_max_finite={abs_r_max_finite:.4g} 是不含极点的真最大值，"
            f"当前 λh 命中极点={bool(current_is_pole)}"
        ),
    )
    return fr


def _region_note(method: str, abs_r: float, h_exit: float, unbounded: bool, blowup: bool, log10_h: float) -> str:
    if not math.isfinite(abs_r):
        return (
            f"{_method_name(method)} 的当前 λh 正好落在**极点**上（r^k 的系数退化、一个根跑到无穷远）："
            f"|R|=+∞，这一步没有有限放大因子。帧里用 pole_fill={_POLE_FILL:g} 占位并置 meta.pole_current=True"
        )
    if blowup:
        return f"{_method_name(method)} 在 log10 h={log10_h:.2f} 上已经发散（|R|={abs_r:.3g}）"
    if not unbounded:
        return f"{_method_name(method)}：h > {h_exit:.4g} 就出稳定域；当前 |R|={abs_r:.4g}"
    return f"{_method_name(method)}：当前方向上没有出域步长（|R|={abs_r:.4g}），整条射线都在域内"


# --------------------------------------------------------------------------- #
# viz 2：刚性困境
# --------------------------------------------------------------------------- #
def _stiff_dilemma(params: Mapping[str, Any], t: float) -> Frame:
    ratio = float(params["stiffness_ratio"])
    explicit = str(params["explicit_method"])
    implicit = str(params["implicit_method"])
    n_implicit = int(params["implicit_steps"])
    lam = -abs(ratio)

    t_end = 2.0 * math.pi
    t_now = _clip_scan(t, 0.0, t_end)

    expl_name, expl_order = _integ_key(explicit)
    impl_name, impl_order = _integ_key(implicit)
    h_ceiling, h_ceiling_theory = _explicit_ceiling(expl_name, expl_order, lam)
    n_plan = int(math.ceil(t_end / h_ceiling)) if math.isfinite(h_ceiling) and h_ceiling > 0 else 0

    rhs = imp.stiff_cos_rhs(lam)
    jac = imp.stiff_cos_jac(lam)
    sol_e = imp.integrate(
        expl_name,
        rhs,
        None,
        (0.0, t_end),
        np.array([1.0]),
        h_ceiling,
        max_steps=20000,
        exact=imp.stiff_cos_exact,
    )
    h_imp = t_end / max(n_implicit, 1)
    sol_i = imp.integrate(
        impl_name,
        rhs,
        jac,
        (0.0, t_end),
        np.array([1.0]),
        h_imp,
        max_steps=20000,
        exact=imp.stiff_cos_exact,
    )

    t_ex = np.linspace(0.0, t_end, 721)
    y_ex = np.asarray(imp.stiff_cos_exact(t_ex), dtype=float)
    te = np.asarray(sol_e.t, dtype=float)
    ye = np.asarray(sol_e.y, dtype=float).reshape(-1)
    ti = np.asarray(sol_i.t, dtype=float)
    yi = np.asarray(sol_i.y, dtype=float).reshape(-1)
    err_e = _floor(np.abs(ye - np.asarray(imp.stiff_cos_exact(te), dtype=float)))

    # 步点（画在主图上：显式密、隐式疏——"被迫走小步"一眼可见）
    n_mark_e = min(int(te.size), 4000)
    mark_e = te[np.linspace(0, te.size - 1, n_mark_e).astype(int)] if te.size else np.zeros(0)
    mark_i = ti

    # 曲率对照：精确解的 |y''|（这里恒 ≤ 1）与显式方法被迫的步长之比
    curvature = np.abs(np.cos(t_ex))

    fr = Frame(chapter="3.2", viz="stiff_dilemma", kind="frame")
    fr.add("t_exact", t_ex, unit="s", label="精确解时间")
    fr.add("y_exact", y_ex, unit="1", label="精确解 cos t")
    fr.add("t_explicit", te, unit="s", label="显式方法时间")
    fr.add("y_explicit", np.minimum(ye, _DISPLAY_CEIL), unit="1", label="显式解（截断显示）")
    fr.add("t_implicit", ti, unit="s", label="隐式方法时间")
    fr.add("y_implicit", np.minimum(yi, _DISPLAY_CEIL), unit="1", label="隐式解")
    fr.add("error", err_e, unit="1", label="显式方法误差（0 以 u/2 显示）")
    fr.add("error_mark", np.array([float(np.interp(t_now, te, err_e))]) if te.size else np.array([_ROUNDOFF]), unit="1", label="当前时刻显式误差")
    fr.add("error_implicit", _floor(np.abs(yi - np.asarray(imp.stiff_cos_exact(ti), dtype=float))), unit="1", label="隐式方法误差")
    fr.add("step_marks_explicit", mark_e, unit="s", label="显式方法的步点（密集）")
    fr.add("step_marks_implicit", mark_i, unit="s", label="隐式方法的步点（稀疏）")
    fr.add("curvature", curvature, unit="1", label="精确解曲率 |y''|")
    fr.add("metric_x", np.arange(4.0), unit="1", label="代价指标序号：步数/求值/LU/Newton")
    fr.add("metric_explicit", np.array([float(sol_e.n_steps), float(sol_e.n_f_evals), float(sol_e.n_lu), float(sol_e.n_newton_iters)]), unit="1", label="显式方法代价")
    fr.add("metric_implicit", np.array([float(sol_i.n_steps), float(sol_i.n_f_evals), float(sol_i.n_lu), float(sol_i.n_newton_iters)]), unit="1", label="隐式方法代价")
    fr.add("step_x", np.arange(2.0), unit="1", label="步长对比序号")
    fr.add("step_values", np.array([_fin(h_ceiling, t_end), h_imp], dtype=float), unit="1", label="两种方法的步长")
    fr.add("step_budget_mark", np.array([float(n_plan)], dtype=float), unit="1", label="显式方法走完一个周期所需步数")
    fr.set_meta(
        stiffness_ratio=_fin(ratio),
        lam=_fin(lam),
        t_now=_fin(t_now),
        t_end=_fin(t_end),
        h_ceiling=_fin(h_ceiling, t_end),
        h_ceiling_theory=_fin(h_ceiling_theory, t_end),
        h_implicit=_fin(h_imp),
        n_steps_explicit=sol_e.n_steps,
        n_steps_explicit_planned=n_plan,
        n_steps_implicit=sol_i.n_steps,
        steps_ratio=_fin(float(sol_e.n_steps) / max(sol_i.n_steps, 1)),
        f_evals_explicit=sol_e.n_f_evals,
        f_evals_implicit=sol_i.n_f_evals,
        lu_implicit=sol_i.n_lu,
        newton_implicit=sol_i.n_newton_iters,
        error_explicit_final=_fin(sol_e.error),
        error_implicit_final=_fin(sol_i.error),
        error_explicit_max=_fin(np.max(err_e)),
        max_abs_y_explicit=_fin(sol_e.max_abs_y),
        blowup=bool(sol_e.blowup),
        t_blowup=_fin(sol_e.t_blowup),
        truncated=bool(sol_e.n_steps >= 20000),
        display_ceiling=_DISPLAY_CEIL,
        curvature_max=_fin(np.max(curvature)),
        roundoff_floor=_ROUNDOFF,
        overflow=bool(sol_e.blowup or sol_e.n_steps >= 20000),
    )
    fr.set_strings(
        explicit_method=explicit,
        implicit_method=implicit,
        note=(
            f"λ={lam:.0f}：显式方法必须走 h≤{h_ceiling:.3g}（理论 {h_ceiling_theory:.3g}），"
            f"一个周期 {n_plan} 步；隐式方法用 h={h_imp:.3g}，{sol_i.n_steps} 步。"
            f"显式总求值 {sol_e.n_f_evals} 次，隐式 {sol_i.n_f_evals} 次 + {sol_i.n_lu} 次 LU。"
        ),
        display_policy="|y| 超过 1e6 后截断显示；meta.max_abs_y_explicit 是未截断真值",
        nonfinite_policy=_NONFINITE_POLICY.format(fill=_POLE_FILL),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 3：|R| 模曲面
# --------------------------------------------------------------------------- #
def _stability_surface(params: Mapping[str, Any], t: float) -> Frame:
    family = str(params["method"])
    order = int(params["order"])
    half = float(params["half_width"])
    n = int(params["grid_n"])

    if family == "bdf":
        stab_name, stab_order, order_effective = "bdf", order, order
    elif family == "explicit_taylor":
        stab_name, stab_order, order_effective = "explicit_taylor", order, order
    else:
        stab_name, stab_order = family, {"forward_euler": 1, "rk4": 4, "implicit_euler": 1, "trapezoidal": 2}[family]
        order_effective = stab_order

    abs_r_raw, zz = _abs_r_grid(stab_name, stab_order, -half, half, -half, half, n, n)
    other_raw = np.asarray(stab.modulus_second_route(stab_name, stab_order, zz), dtype=float)
    # 极点处两条路径都返回 +∞：差没有定义（inf−inf=NaN 还会招来 invalid 哨声）。
    # 做法是先在**有限子集**上做差（极点两侧都换成 0），再用掩码把那些格子从误差统计里摘掉。
    route_finite = np.isfinite(abs_r_raw) & np.isfinite(other_raw)
    abs_r, pole_mask, n_pole = _finite_or(abs_r_raw)
    finite_cells = abs_r_raw[np.isfinite(abs_r_raw)]
    abs_r_max_finite = float(np.max(finite_cells)) if finite_cells.size else 0.0
    abs_r_min_finite = float(np.min(finite_cells)) if finite_cells.size else 0.0
    diff = np.abs(
        np.where(np.isfinite(abs_r_raw), abs_r_raw, 0.0) - np.where(np.isfinite(other_raw), other_raw, 0.0)
    )
    # 同一位置换过值的两路都记 0：mask 之外的点不会被读到
    error_grid = _floor(np.where(route_finite, diff, 0.0))
    if not np.all(route_finite):
        error_grid = np.where(route_finite, error_grid, _ROUNDOFF)

    boundary = stab.stability_boundary(stab_name, stab_order, n_theta=2049)
    res_boundary = _floor(np.abs(np.abs(stab.stability_modulus(stab_name, stab_order, boundary)) - 1.0))
    rep = stab.stability_report(stab_name, stab_order)

    edge, edge_pole, n_edge_pole = _finite_or(
        np.asarray(
            np.abs(stab.stability_modulus(stab_name, stab_order, np.array([-half + 0j, half + 0j, -half * 1j, half * 1j]))),
            dtype=float,
        )
    )

    fr = Frame(chapter="3.2", viz="stability_surface", kind="frame")
    fr.add("abs_R", abs_r, unit="1", label="|R(z)|（二维网格，行主序；极点处 = strings.pole_fill）")
    fr.add("abs_R_pole", pole_mask, unit="1", label="极点掩码：1 表示该格 |R|=+∞（用 pole_fill 占位）")
    fr.add("extent", np.array([-half, half, -half, half], dtype=float), unit="1", label="窗口 x0,x1,y0,y1")
    fr.add("levels", np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0], dtype=float), unit="1", label="|R| 等值线层级")
    fr.add("boundary_x", np.real(boundary), unit="1", label="|R|=1 边界 Re z")
    fr.add("boundary_y", np.imag(boundary), unit="1", label="|R|=1 边界 Im z")
    fr.add("residual", res_boundary, unit="1", label="边界自洽残差 ||R(z_θ)|−1|")
    fr.add("error", error_grid, unit="1", label="两条独立路径之差 |主路径−闭式|（逐网格点）")
    fr.add("error_mark", np.array([float(np.max(error_grid))]), unit="1", label="网格上的最大路径差")
    fr.add("edge_values", edge, unit="1", label="窗口四角的 |R|（色标极值参考；极点处 = pole_fill）")
    fr.add("edge_pole", edge_pole, unit="1", label="窗口四角命中极点的掩码")
    fr.add("angle_k", np.arange(1.0, 7.0), unit="1", label="BDF 阶数")
    fr.add("angle_measured", np.array([stab.bdf_stability_angle_deg(k)[0] for k in range(1, 7)]), unit="deg", label="实测 A(α) 角")
    fr.add("angle_theory", np.array([stab.BDF_ANGLE_THEORY_DEG[k] for k in range(1, 7)]), unit="deg", label="理论 A(α) 角")
    fr.set_meta(
        order_effective=order_effective,
        order_param=order,
        half_width=_fin(half),
        grid_n=n,
        a_stable=bool(rep.a_stable),
        l_stable=bool(rep.l_stable),
        angle_deg=_fin(rep.angle_deg),
        stiff_limit=_fin(rep.stiff_limit),
        endpoint_finite=bool(math.isfinite(rep.negative_real_endpoint)),
        negative_real_endpoint=_fin(rep.negative_real_endpoint, 0.0),
        max_on_imaginary_axis=_fin(rep.max_on_imaginary_axis),
        abs_R_max=_fin(abs_r_max_finite),
        abs_R_min=_fin(abs_r_min_finite),
        n_inside=int(np.count_nonzero(abs_r <= 1.0)),
        n_grid=int(abs_r.size),
        inside_fraction=_fin(float(np.count_nonzero(abs_r <= 1.0)) / max(abs_r.size, 1)),
        residual_max=_fin(np.max(res_boundary)),
        route_diff_max=_fin(np.max(error_grid)),
        edge_max=_fin(np.max(edge)),
        roundoff_floor=_ROUNDOFF,
        vmin=0.0,
        vmax=_fin(min(abs_r_max_finite, 4.0)),
        overflow=bool(not np.all(np.isfinite(abs_r_raw))),
        n_pole_cells=n_pole,
        n_pole_edge=n_edge_pole,
        n_route_mismatch=int(np.count_nonzero(~route_finite)),
        pole_fill=_POLE_FILL,
    )
    fr.set_strings(
        method=family,
        method_name=_method_name(family) if family in ("forward_euler", "rk4", "implicit_euler", "trapezoidal") else family,
        route=rep.route,
        note=(
            f"{family}(k={order_effective})：A-稳定={bool(rep.a_stable)}，A(α) 角 {rep.angle_deg:.4g}°，"
            f"盒内 {int(np.count_nonzero(abs_r <= 1.0))}/{abs_r.size} 个网格点落在稳定域内；"
            f"边界残差 {float(np.max(res_boundary)):.2e}"
        ),
        colour_note="暖色只表示 |R|>1；z 在右半平面时真解本来就在增长，别把两者混为一谈",
        pole_fill=f"{_POLE_FILL:g}",
        pole_note=(
            f"|R|=+∞ 的位置是极点（BDF-k 的 z=α_k：1, 1.5, 11/6, 25/12, 137/60, 49/20）："
            f"多项式降次、一个根跑到无穷远，不是「值很大」。本帧 {n_pole}/{abs_r.size} 个格子是极点，"
            f"已用 pole_fill={_POLE_FILL:g} 占位并由 abs_R_pole 标出；"
            f"两条独立路径在这些格子上都是 +∞，差无定义，故 error 网格在那里记地板值；"
            f"meta.n_route_mismatch={int(np.count_nonzero(~route_finite))} 是「两路不同时有限」的格子数"
            f"（应当等于极点数，不等就说明两条路径本身有问题）"
        ),
        nonfinite_policy=_NONFINITE_POLICY.format(fill=_POLE_FILL),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：显式 vs 隐式
# --------------------------------------------------------------------------- #
def _explicit_vs_implicit(params: Mapping[str, Any], t: float) -> Frame:
    tol = float(params["tol"])
    ratio = float(params["stiffness_ratio"])
    explicit = str(params["explicit_method"])
    implicit = str(params["implicit_method"])
    lam = -abs(ratio)
    t_end = 2.0 * math.pi
    t_now = _clip_scan(t, 0.0, t_end)

    expl_name, expl_order = _integ_key(explicit)
    impl_name, impl_order = _integ_key(implicit)
    h_ceiling, h_ceiling_theory = _explicit_ceiling(expl_name, expl_order, lam)
    n_stages = 4 if expl_name == "rk4" else 1

    rhs = imp.stiff_cos_rhs(lam)
    jac = imp.stiff_cos_jac(lam)
    y0 = np.array([1.0])
    h_exp = min(h_ceiling, t_end / 4.0) if math.isfinite(h_ceiling) else t_end / 64.0
    sol_e = imp.integrate(expl_name, rhs, None, (0.0, t_end), y0, h_exp, max_steps=20000, exact=imp.stiff_cos_exact)

    # 隐式一侧的步长由**实测代价梯**决定：梯子上的每一行都是一次真积分
    ladder = imp.cost_ladder(impl_name, impl_order, rhs, jac, (0.0, t_end), y0, k_range=(3, 11), exact=imp.stiff_cos_exact)
    pick = imp.step_for_tolerance(ladder, tol)
    h_imp = float(pick["h"])
    used_fallback = False
    if h_imp <= 0.0:
        h_imp = t_end / float(2**11)  # 容差比梯子最细的一级还紧：退到梯底并在 strings 里说明
        used_fallback = True
    sol_i = imp.integrate(impl_name, rhs, jac, (0.0, t_end), y0, h_imp, max_steps=20000, exact=imp.stiff_cos_exact)

    # 截到当前时刻（时间轴是真实时间：拖着走，代价与误差一起长大）
    def _upto(sol: imp.ODESolution, tmax: float) -> tuple[np.ndarray, np.ndarray]:
        tt = np.asarray(sol.t, dtype=float)
        yy = np.asarray(sol.y, dtype=float).reshape(tt.size, -1)
        m = tt <= tmax + 1e-15
        if not bool(np.any(m)):
            m = np.zeros_like(tt, dtype=bool)
            m[0] = True
        return tt[m], yy[m]

    te, ye = _upto(sol_e, t_now)
    ti, yi = _upto(sol_i, t_now)
    y_ex = np.asarray(imp.stiff_cos_exact(te), dtype=float)
    err_e = _floor(np.abs(ye[:, 0] - y_ex))
    err_i = _floor(np.abs(yi[:, 0] - np.asarray(imp.stiff_cos_exact(ti), dtype=float)))

    # 代价柱：显式方法没有 LU 也没有 Newton，那两根柱子就是 0（不是"省略"）
    n_step_e = float(max(te.size - 1, 0))
    n_step_i = float(max(ti.size - 1, 0))
    f_exp_now = n_step_e * float(n_stages)
    scale_i = n_step_i / max(sol_i.n_steps, 1)
    metrics_exp = np.array([n_step_e, f_exp_now, 0.0, 0.0], dtype=float)
    metrics_imp = np.array(
        [n_step_i, float(sol_i.n_f_evals) * scale_i, float(sol_i.n_lu) * scale_i, float(sol_i.n_newton_iters) * scale_i],
        dtype=float,
    )

    # 代价—容差曲线：x 是梯子上每一步"实测达到的容差"，y 是那一步的函数求值次数
    cost_x = np.log10(np.maximum(ladder["max_error"], _ROUNDOFF))
    cost_imp = np.asarray(ladder["n_f_evals"], dtype=float)
    cost_exp = np.full(cost_x.shape, n_stages * t_end / max(h_exp, 1e-300), dtype=float)
    cross_lo, cross_hi = _crossing(cost_x, cost_imp, float(cost_exp[0]))

    fr = Frame(chapter="3.2", viz="explicit_vs_implicit", kind="frame")
    fr.add("t_explicit", te, unit="s", label="显式时间")
    fr.add("y_explicit", np.clip(ye[:, 0], -_DISPLAY_CEIL, _DISPLAY_CEIL), unit="1", label="显式解（超出显示上限即截断）")
    fr.add("t_implicit", ti, unit="s", label="隐式时间")
    fr.add("y_implicit", yi[:, 0], unit="1", label="隐式解")
    fr.add("t_exact", te, unit="s", label="精确解时间")
    fr.add("y_exact", y_ex, unit="1", label="精确解 cos t")
    fr.add("error", err_e, unit="1", label="显式误差（对数轴，0 以 u/2 显示）")
    fr.add("error_implicit", err_i, unit="1", label="隐式误差")
    fr.add("error_mark", np.array([err_e[-1] if err_e.size else _ROUNDOFF]), unit="1", label="当前时刻显式误差")
    fr.add("error_implicit_mark", np.array([err_i[-1] if err_i.size else _ROUNDOFF]), unit="1", label="当前时刻隐式误差")
    fr.add("metric_x", np.arange(4.0), unit="1", label="代价指标：步数/函数求值/LU/Newton")
    fr.add("metric_explicit", metrics_exp, unit="1", label="显式方法代价（截到当前时刻；LU/Newton 恒为 0）")
    fr.add("metric_implicit", metrics_imp, unit="1", label="隐式方法代价")
    fr.add("step_x", np.arange(2.0), unit="1", label="步长对比")
    fr.add("step_values", np.array([h_exp, h_imp], dtype=float), unit="1", label="两种方法实际使用的步长")
    fr.add("tolerance_mark", np.array([math.log10(max(tol, _ROUNDOFF))], dtype=float), unit="1", label="当前容差（log10）")
    fr.add("cost_x", cost_x, unit="1", label="代价曲线横轴：实测达到的容差 log10")
    fr.add("cost_explicit", cost_exp, unit="1", label="显式方法的函数求值次数（与容差无关：受稳定性限制）")
    fr.add("cost_implicit", cost_imp, unit="1", label="隐式方法的函数求值次数（每一级都是真跑出来的）")
    fr.add("cost_cross_x", np.asarray(cross_lo, dtype=float), unit="1", label="代价交叉点 log10 容差（无交叉时为空）")
    fr.add("cost_cross_y", np.asarray(cross_hi, dtype=float), unit="1", label="代价交叉点处的求值次数")
    fr.set_meta(
        tol=_fin(tol),
        stiffness_ratio=_fin(ratio),
        lam=_fin(lam),
        t_now=_fin(t_now),
        t_end=_fin(t_end),
        h_explicit=_fin(h_exp),
        h_explicit_theory=_fin(h_ceiling_theory, h_exp),
        h_implicit=_fin(h_imp),
        h_ratio=_fin(h_imp / h_exp) if h_exp > 0 else 0.0,
        n_steps_explicit=int(n_step_e),
        n_steps_implicit=int(n_step_i),
        steps_ratio=_fin(n_step_e / max(n_step_i, 1.0)),
        f_evals_explicit=_fin(f_exp_now),
        f_evals_implicit=_fin(metrics_imp[1]),
        lu_implicit=_fin(metrics_imp[2]),
        newton_implicit=_fin(metrics_imp[3]),
        error_explicit=_fin(err_e[-1] if err_e.size else 0.0),
        error_implicit=_fin(err_i[-1] if err_i.size else 0.0),
        error_explicit_span=_fin(sol_e.max_error),
        error_implicit_span=_fin(sol_i.max_error),
        max_abs_y_explicit=_fin(sol_e.max_abs_y),
        implicit_error_at_planned_step=_fin(pick["max_error"]),
        startup_steps=1 if impl_name == "bdf2" else 0,
        tolerance_fallback=used_fallback,
        cost_cross_log10_tol=_fin(cross_lo[0] if cross_lo.size else 0.0),
        cost_cross_found=bool(cross_lo.size > 0),
        cost_implicit_at_tol=_fin(float(np.interp(math.log10(max(tol, _ROUNDOFF)), cost_x, cost_imp))),
        cost_explicit_const=_fin(float(cost_exp[0])),
        ladder_k_min=float(ladder["k"][0]),
        ladder_k_max=float(ladder["k"][-1]),
        blowup=bool(sol_e.blowup),
        t_blowup=_fin(sol_e.t_blowup),
        truncated=bool(sol_e.n_steps >= 20000),
        display_ceiling=_DISPLAY_CEIL,
        roundoff_floor=_ROUNDOFF,
        overflow=bool(sol_e.blowup or sol_e.n_steps >= 20000),
    )
    fr.set_strings(
        explicit_method=explicit,
        implicit_method=implicit,
        note=(
            f"显式 h={h_exp:.3g}（稳定天花板 {h_ceiling:.3g}）× {n_step_e:.0f} 步 = {f_exp_now:.0f} 次求值，"
            f"误差与容差**无关**（{sol_e.max_error:.2e}，受稳定性限制）；"
            f"隐式 h={h_imp:.3g} × {n_step_i:.0f} 步 = {metrics_imp[1]:.0f} 次求值 + {metrics_imp[2]:.0f} 次 LU"
            f"（容差 {tol:.1e}，整段误差 {sol_i.max_error:.2e}）"
        ),
        crossover_note=(
            "两条代价曲线在 log10(容差)="
            + (f"{cross_lo[0]:.2f}" if cross_lo.size else "梯子范围内没有交叉")
            + " 附近交叉：容差比它更松时隐式便宜，更紧时显式的稳定性限制反而成了优势"
        ),
        display_policy="|y| 超过 1e6 后截断显示；meta.max_abs_y_explicit 是未截断真值",
        nonfinite_policy=_NONFINITE_POLICY.format(fill=_POLE_FILL),
    )
    return fr


def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """线性插值出 ``y`` 穿过水平线 ``level`` 的交点（``x`` 单调递增；无交点返回空数组）。"""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    out_x: list[float] = []
    out_y: list[float] = []
    for i in range(1, xs.size):
        a, b = ys[i - 1] - level, ys[i] - level
        if a == 0.0 or (a < 0.0) != (b < 0.0):
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if y1 != y0:
                t = (level - y0) / (y1 - y0)
                out_x.append(float(x0 + t * (x1 - x0)))
                out_y.append(float(level))
    if not out_x:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(out_x)
    return np.asarray(out_x, dtype=float)[order], np.asarray(out_y, dtype=float)[order]


# --------------------------------------------------------------------------- #
# viz 5：BDF 稳定角
# --------------------------------------------------------------------------- #
def _bdf_angle(params: Mapping[str, Any], t: float) -> Frame:
    k = int(params["k"])
    show_family = bool(params["show_family"])
    alpha = _clip_scan(t, 0.0, 90.0)

    boundary = stab.stability_boundary("bdf", k, n_theta=401)
    res_boundary = _floor(np.abs(np.abs(stab.stability_modulus("bdf", k, boundary)) - 1.0))
    angles = np.array([stab.bdf_stability_angle_deg(j)[0] for j in range(1, 7)], dtype=float)
    theory = np.array([stab.BDF_ANGLE_THEORY_DEG[j] for j in range(1, 7)], dtype=float)

    fam_x: list[float] = []
    fam_y: list[float] = []
    fam_k: list[float] = []
    if show_family:
        for j in range(1, 7):
            zj = stab.stability_boundary("bdf", j, n_theta=241)
            fam_x.extend(np.real(zj).tolist())
            fam_y.extend(np.imag(zj).tolist())
            fam_k.extend([float(j)] * zj.size)
    family_x = np.asarray(fam_x, dtype=float)
    family_y = np.asarray(fam_y, dtype=float)
    family_k = np.asarray(fam_k, dtype=float)

    # 测试射线 z = −r e^{∓iα}：α 是楔形半角
    rad = np.linspace(0.0, 6.0, 241)
    th = math.radians(alpha)
    ray_x = -rad * math.cos(th)
    ray_y = rad * math.sin(th)
    z_ray = ray_x + 1j * ray_y
    mods = np.asarray(stab.stability_modulus("bdf", k, z_ray), dtype=float)
    violation = _floor(np.maximum(mods - 1.0, 0.0))
    violated = bool(np.any(mods > 1.0 + 1e-12))
    # 楔形（两条边界射线 + 原点）。**不用 NaN 做折线分隔符**：`.nlf` 的数组段是裸 float64，
    # NaN 能编码进去，但前端拿到一条含 NaN 的折线时会静默丢掉那一段（不报错），
    # 而全仓帧契约测试要求每个数组都有限。这里改成"画完一条边回到原点再画另一条"，
    # 图形完全一样（V 形的拐点就是原点），但数据里一个非有限值都没有。
    wedge_x = np.array([0.0, -6.0 * math.cos(th), 0.0, -6.0 * math.cos(th)], dtype=float)
    wedge_y = np.array([0.0, 6.0 * math.sin(th), 0.0, -6.0 * math.sin(th)], dtype=float)

    angle_k = angles[k - 1]
    fr = Frame(chapter="3.2", viz="bdf_angle", kind="frame")
    fr.add("boundary_x", np.real(boundary), unit="1", label=f"BDF-{k} 稳定域边界 Re z")
    fr.add("boundary_y", np.imag(boundary), unit="1", label=f"BDF-{k} 稳定域边界 Im z")
    fr.add("family_x", family_x, unit="1", label="k=1…6 边界家族 Re z（顺序拼接，用 family_k 分段）")
    fr.add("family_y", family_y, unit="1", label="k=1…6 边界家族 Im z（顺序拼接，用 family_k 分段）")
    fr.add("family_k", family_k, unit="1", label="家族中每个点所属的阶数 k")
    fr.add("wedge_x", wedge_x, unit="1", label="测试楔形 Re z（0→上边→0→下边，无分隔符）")
    fr.add("wedge_y", wedge_y, unit="1", label="测试楔形 Im z（0→上边→0→下边，无分隔符）")
    fr.add("ray_x", ray_x, unit="1", label="测试射线 Re z")
    fr.add("ray_y", ray_y, unit="1", label="测试射线 Im z")
    fr.add("error", violation, unit="1", label="测试射线上的超界量 max(0,|R|−1)（0 以 u/2 显示）")
    fr.add("residual", res_boundary, unit="1", label="边界自洽残差 ||R(z_θ)|−1|")
    fr.add("angle_k", np.arange(1.0, 7.0), unit="1", label="BDF 阶数")
    fr.add("angle_measured", angles, unit="deg", label="各阶实测 A(α) 角")
    fr.add("angle_theory", theory, unit="deg", label="各阶理论 A(α) 角")
    fr.add("angle_mark_x", np.array([float(k)]), unit="1", label="当前阶数")
    fr.add("angle_mark_y", np.array([angle_k]), unit="deg", label="当前阶的实测角")
    fr.set_meta(
        k=k,
        k_max=6,
        alpha_probe_deg=_fin(alpha),
        angle_current_deg=_fin(angle_k),
        angle_theory_deg=_fin(theory[k - 1]),
        angle_error_deg=_fin(abs(angle_k - theory[k - 1])),
        a_stable=bool(angle_k >= 90.0 - 1e-6),
        wedge_violated=violated,
        violation_max=_fin(np.max(violation)),
        violation_floor=_ROUNDOFF,
        alpha_within_measured=bool(alpha <= angle_k + 1e-9),
        residual_max=_fin(np.max(res_boundary)),
        n_theta_boundary=int(boundary.size),
        family_size=int(family_k.size),
        show_family=show_family,
        angle_k1=_fin(angles[0]),
        angle_k3=_fin(angles[2]),
        angle_k6=_fin(angles[5]),
        angles_monotone=bool(np.all(np.diff(angles) <= 1e-9)),
        z_axis_limit=6.0,
        overflow=False,
    )
    fr.set_strings(
        note=(
            f"BDF-{k}：A(α) 角实测 {angle_k:.4f}°（理论 {theory[k-1]:.2f}°），"
            f"当前测试角 α={alpha:.1f}° → {'楔形已越界' if violated else '楔形仍在域内'}；"
            f"k≥3 不再 A-稳定，k≥7 连零稳定都没有"
        ),
        angle_meaning="A(α) 角 = 以负实轴为轴、整体含于稳定域的最大楔形半角；A-稳定即 90°",
        display_policy="通不过的射线用超界量 max(0,|R|−1) 表示：它为 0 只说明这一条射线没出界，不等于 A-稳定",
        nonfinite_policy=_NONFINITE_POLICY.format(fill=_POLE_FILL),
        wedge_note=(
            "楔形用「0→上边→回到 0→下边」四点折线画（V 的拐点就是原点），**没有任何 NaN 分隔符**："
            "含 NaN 的折线在前端会静默丢段，而全仓帧契约测试要求每个数组都有限。"
            "本帧另一个事实：楔形整体在闭左半平面，BDF-k 的极点 z=α_k 全是正实数，永远不在楔形里，"
            "所以本帧的 |R| 与超界量全部有限"
        ),
    )
    return fr
