"""5.1 章：无约束优化与拟牛顿（帧装配层）。

四个可视化各自回答一个**会被讲错**的问题：

1. ``descent_race`` —— 同一张等高线上，GD 的锯齿、动量的绕圈、BFGS 的拐弯、牛顿的直取。
   参数 ``step_scale`` 是有理论临界值的：它是 :math:`\\alpha / \\alpha^\\star`，
   :math:`\\alpha^\\star=2/(L+\\mu)`；而二次型上 :math:`\\alpha>2/L` 就发散，
   即 ``step_scale`` 超过 :math:`1+1/\\kappa`（κ=100 时是 1.01）。
2. ``convergence_rates`` —— 线性 / 超线性 / 二次三种形态在半对数图上的形状差异，
   以及**实测**的二次收敛常数 :math:`e_{k+1}/e_k^2` 与进入二次区的迭代序号。
   ``alpha_cap`` 是"离开二次区"的开关：卡住步长，二次收敛当场退化成线性。
3. ``saddle_escape`` —— 鞍点上"随机扰动跑掉了"到底说明了什么：负曲率方向、
   负特征值大小、以及**沿该方向**逃逸所需的最小扰动量级（二分实测）。
4. ``trust_region_radius`` —— 半径 :math:`\\Delta_k` 的自适应：被拒绝的步、:math:`\\rho_k`
   与 :math:`\\Delta` 的收缩/扩张，以及 Steihaug-CG 在不定 Hessian 上走边界的那一条分支。

帧纪律（与 0.0/1.2 章一致）：``meta`` 只放 number/bool/None，字符串一律走 ``strings``；
没有有限值就写 ``None`` 并在 ``strings`` 里说明原因；对数轴上的 0 一律给地板值并写明地板
（``error_floor``）；``server/`` 里不写数学，所有量都来自 :mod:`numcore.inverse.optimize`。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from numcore.inverse.optimize import (
    anisotropic_quadratic,
    bfgs,
    convergence_order,
    contour_grid,
    escape_threshold,
    gd,
    gd_iterations,
    min_eigenpair,
    momentum,
    momentum_parameters,
    newton,
    quartic_quadratic,
    quadratic_window,
    rosenbrock_objective,
    saddle_quadratic,
    trust_region,
)
from numcore.utils import rng
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 误差类曲线的对数轴地板（写进 ``strings``；同时给出"真正到 0"的点数）
ERROR_FLOOR = 1e-17
#: 比值类曲线（二次比值 / 线性比值）的地板
RATIO_FLOOR = 1e-30

_MANIFEST_PITFALLS = (
    "半对数图上的「直线」只说明误差按固定因子缩小（线性收敛），"
    "曲线突然变陡才是超线性/二次；把两种形状混着读会把 BFGS 说成二次收敛。"
    "牛顿法的漂亮直线段只出现在解附近且整步被接受时——把初值拖远、或把 $\\alpha_{\\mathrm{cap}}$ 拖到 $1$ 以下，"
    "同一张图立刻变回线性，而图上没有任何东西会提醒你「二次收敛的前提刚刚被破坏」。"
)


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _finite_or_none(x: Any) -> float | None:
    """``NaN/±Inf`` → ``None``（PROTOCOLS §1.3：JSON 头里没有 NaN/Infinity）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _floored(values: Any, floor: float = ERROR_FLOOR) -> np.ndarray:
    """把非负曲线抬到对数轴地板上（0 在对数轴画不出来）。"""
    v = np.asarray(values, dtype=float)
    return np.maximum(v, float(floor))


def _pad_tail(values: Any, n: int, fill: float | None = None) -> np.ndarray:
    """把曲线补到长度 ``n``（**尾部重复末点**，PROTOCOLS 允许的补位方式）。

    ``values`` 为空时用 ``fill``（默认 0.0）填满——空数组代表"这个方法一步都没走成"，
    这时曲线是平的，而不是被截断的假象。
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return np.full(int(n), 0.0 if fill is None else float(fill))
    if v.size >= int(n):
        return v[: int(n)]
    return np.concatenate([v, np.full(int(n) - v.size, float(v[-1]) if fill is None else float(fill))])


def _finite_array(values: Any, *, fill: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """把数组里的 ``NaN/±Inf`` 换成占位值并给出掩码（PROTOCOLS §1.3.1 三件套）。

    真值不会被改：``mask`` 为 0 的位置才是占位值，帧的 ``strings`` 里必须说明它是占位符。
    """
    v = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(v)
    return np.where(ok, v, float(fill)), ok


def _pad_mask(values: Any, n: int) -> np.ndarray:
    """布尔掩码补到长度 ``n``（尾部填 0：补位处**不是**真数据）。"""
    v = np.asarray(values).astype(bool).ravel()
    if v.size >= int(n):
        return v[: int(n)]
    return np.concatenate([v, np.zeros(int(n) - v.size, dtype=bool)])


def _step_index(t: float, n_max: int) -> int:
    """把 scrubbing 的 ``t`` 变成整数步（夹在 ``[0, n_max]``）。"""
    return int(min(max(int(round(float(t))), 0), int(n_max)))


def _slice_traj(xs: np.ndarray, ys: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """取轨迹的前 ``k+1`` 个点（至少 1 个点，避免空数组）。"""
    m = min(int(k) + 1, xs.size)
    m = max(m, 1)
    return xs[:m], ys[:m]


def _random_unit(seed: int) -> np.ndarray:
    """确定性的随机单位向量（P4：随机数走 ``numcore.utils.rng``）。"""
    v = np.asarray(rng(seed).standard_normal(2), dtype=float)
    return v / float(np.sqrt(float(np.dot(v, v))))


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
MANIFEST: dict[str, Any] = {
    "chapter": "5.1",
    "title": "无约束优化与拟牛顿",
    "part": "V",
    "summary": (
        "梯度下降的条件数依赖、牛顿法的二次收敛（以及它在什么条件下**不**成立）、"
        "BFGS/L-BFGS 的割线方程与正定性、非线性共轭梯度、信赖域半径自适应、鞍点与负曲率。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "descent_race",
            "title": "下降路径赛跑：锯齿、动量、割线与直取",
            "description": (
                "同一张等高线上四个方法同时出发：GD（定步长）沿最陡方向来回锯齿，"
                "重球动量绕开锯齿，BFGS 用割线信息拐弯，牛顿法一步取直。"
                "时间轴是迭代步，副轴是到最优解的距离（半对数）。"
                "把 $\\kappa$ 拖大看锯齿变密，把 step_scale 拖过 $1$ 看 GD 直接发散。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "kappa",
                    "label": "条件数 κ = λmax/λmin",
                    "type": "float",
                    "min": 2.0,
                    "max": 10000.0,
                    "step": 1.0,
                    "default": 100.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "angle_deg",
                    "label": "长轴与 x 轴夹角（度）",
                    "type": "float",
                    "min": 0.0,
                    "max": 90.0,
                    "step": 1.0,
                    "default": 35.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "step_scale",
                    "label": "GD 步长 / 最优步长 α* = 2/(L+μ)",
                    "type": "float",
                    "min": 0.1,
                    "max": 1.6,
                    "step": 0.01,
                    "default": 1.0,
                    "critical": [1.0],
                    "criticalNote": (
                        "$1.0$ 就是二次型上的最优定步长 $\\alpha^*=2/(L+\\mu)$（误差按 $(\\kappa-1)/(\\kappa+1)$ 收缩）。"
                        "越过它，GD 仍然收敛但更慢；真正发散线是 $\\alpha=2/L$，即 "
                        "$\\text{step\\_scale}=(L+\\mu)/L=1+1/\\kappa$——$\\kappa=100$ 时只有 $1.01$，"
                        "所以「稍微大于 $1$」就已经在发散区。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "max_iter",
                    "label": "最大迭代步数",
                    "type": "int",
                    "min": 10,
                    "max": 400,
                    "default": 60,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 60.0, "dt": 1.0, "fps": 30, "label": "迭代 k"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "等高线 f(x)=const 与四条下降路径", "ylabel": "x_2"},
                {"id": "aux", "autoscaled": True, "title": "到最优解的距离 ‖x_k - x*‖", "ylabel": "‖x_k - x*‖", "scale": "log"},
            ],
            "pitfalls": _MANIFEST_PITFALLS,
            "theory": [
                "定步长 GD 的误差压缩因子 $\\rho=\\max|1-\\alpha\\lambda_i|$，最优 $\\alpha^*=2/(L+\\mu)$ 时 $\\rho=(\\kappa-1)/(\\kappa+1)$",
                "重球动量的最优压缩因子是 $(\\sqrt\\kappa-1)/(\\sqrt\\kappa+1)$，迭代数约为 GD 的 $1/\\sqrt\\kappa$",
                "牛顿法在二次型上一步到达驻点（$H$ 正定时就是极小点）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "convergence_rates",
            "title": "收敛率图：线性、超线性与二次在半对数图上的形状",
            "description": (
                "同一个各向异性四次型上跑 GD / 动量 / BFGS / 牛顿，画 $\\|x_k-x^*\\|$ 随 $k$ 的曲线，"
                "并单独给出实测的二次比值 $e_{k+1}/e_k^2$ 与线性比值 $e_{k+1}/e_k$。"
                "牛顿段的直线是线性收敛、向下弯的段才是二次；$\\alpha_{\\mathrm{cap}}$ 一拖到 $1$ 以下，"
                "二次段当场消失、比值稳定在 $1-\\alpha_{\\mathrm{cap}}$。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "kappa",
                    "label": "条件数 κ（二次部分）",
                    "type": "float",
                    "min": 2.0,
                    "max": 1000.0,
                    "step": 1.0,
                    "default": 100.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "start",
                    "label": "初值尺度（x0 = (s, s)）",
                    "type": "float",
                    "min": 0.02,
                    "max": 2.0,
                    "step": 0.02,
                    "default": 1.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "beta",
                    "label": "四次项系数 β（β=0 退化成二次型）",
                    "type": "float",
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "default": 0.05,
                    "critical": [0.0],
                    "criticalNote": (
                        "$\\beta=0$ 时问题退化为二次型，牛顿法一步到底——"
                        "此时图上根本没有「二次收敛的尾巴」可看，"
                        "meta.newton_quadratic_constant 会写 null 并说明原因。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "alpha_cap",
                    "label": "牛顿步长上限 α_cap",
                    "type": "float",
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "default": 1.0,
                    "critical": [1.0],
                    "criticalNote": (
                        "二次收敛要求解附近**整步** $\\alpha_k=1$ 被接受。$\\alpha_{\\mathrm{cap}}<1$ 时每步只能走 "
                        "$\\alpha_{\\mathrm{cap}}$ 倍牛顿步，实测退化为线性收敛，因子 $\\approx1-\\alpha_{\\mathrm{cap}}$"
                        "（$\\alpha_{\\mathrm{cap}}=0.5$ 实测 $0.5000$，$\\alpha_{\\mathrm{cap}}=0.25$ 实测 $0.7511$）。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "max_iter",
                    "label": "最大迭代步数",
                    "type": "int",
                    "min": 5,
                    "max": 80,
                    "default": 30,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "convergence", "autoscaled": True, "title": "到最优解的距离", "ylabel": "‖x_k - x*‖", "scale": "log"},
                {"id": "ratio", "autoscaled": True, "title": "实测比值（二次 / 线性）", "ylabel": "比值", "scale": "log"},
            ],
            "pitfalls": (
                "「曲线陡」不等于二次收敛：把纵轴换成对数、横轴仍是迭代数时，"
                "任何超线性收敛都会看起来像一条竖直的线。判据只有 $e_{k+1}/e_k^2$ 是否**趋于常数**，"
                "而且窗口右端必须砍掉舍入地板——实测在 $e\\approx3\\times10^{-29}$ 之后这个比值会反弹成 $1.74$，"
                "把地板当成「超二次收敛」是最典型的假证据。"
                "另外 $\\beta=0$ 时牛顿一步到底，图上只剩一个点，什么都说明不了。"
            ),
            "theory": [
                "线性收敛：$e_{k+1}\\le\\rho e_k$；二次收敛：$e_{k+1}\\le C e_k^2$（$C$ 与 $k$ 无关）",
                "牛顿法在解附近且 $H$ 正定、整步被接受时 $e_{k+1}\\approx C e_k^2$",
                "四次型 $f=\\frac{1}{2}x^T A x+(\\beta/4)\\|x\\|^4$ 的最优解精确为原点，Hessian 处处正定",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "saddle_escape",
            "title": "鞍点逃逸：负曲率方向与扰动幅度",
            "description": (
                "鞍点 $f=\\frac{1}{2}(\\lambda_-x_1^2+\\lambda_+x_2^2)$（$\\lambda_-<0$，负曲率方向是 $x_1$）上，GD 从稳定流形"
                "出发会**收敛到鞍点**并停住；扰动只有在负曲率方向 $u_{\\min}$ 上有分量时才跑得掉。"
                "图中给出扰动轨迹、沿 $u_{\\min}$ / 随机方向 / 正交方向的实测逃逸阈值："
                "随机方向的阈值恰好是 $u_{\\min}$ 方向的 $1/|\\langle u_{\\min},d\\rangle|$ 倍。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "neg",
                    "label": "负特征值 λ-",
                    "type": "float",
                    "min": -2.0,
                    "max": -0.001,
                    "step": 0.001,
                    "default": -0.02,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "delta",
                    "label": "扰动量级 δ",
                    "type": "float",
                    "min": 1e-09,
                    "max": 0.1,
                    "step": 1e-09,
                    "default": 0.02,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "direction",
                    "label": "扰动方向",
                    "type": "enum",
                    "options": ["neg_curv", "random", "orthogonal"],
                    "default": "neg_curv",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "max_iter",
                    "label": "GD 步数",
                    "type": "int",
                    "min": 20,
                    "max": 400,
                    "default": 200,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 200.0, "dt": 1.0, "fps": 30, "label": "GD 步 k"},
            "requiredAux": ["residual"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "鞍点等高线（双曲线族）与扰动轨迹", "ylabel": "x_2"},
                {"id": "aux", "autoscaled": True, "title": "梯度范数与到驻点距离", "ylabel": "‖∇f‖ / ‖x - x*‖", "scale": "log"},
            ],
            "pitfalls": (
                "「加了随机扰动就逃出来了」不是方法有效的证据：逃逸与否只取决于扰动在"
                "负曲率方向上的投影。实测沿 $u_{\\min}$ 的阈值与解析式 $R(1+|\\lambda_-|\\alpha)^{-N}$ 一致；"
                "随机方向的阈值是它的 $1/|\\langle u_{\\min},d\\rangle|$ 倍。把扰动换成与 $u_{\\min}$ 正交的方向，"
                "阈值在放大到 $10^{12}$ 之后仍然找不到——图上看到的「没跑掉」才是真相，"
                "而 meta 里那个 null 必须如实说明它是「没测到」而不是 $0$。"
            ),
            "theory": [
                "梯度下降在鞍点附近只会沿正曲率方向收缩，负曲率分量按 $(1+|\\lambda_-|\\alpha)^k$ 增长",
                "逃逸阈值 $\\delta^*=R(1+|\\lambda_-|\\alpha)^{-N}$（$R$ 邻域半径、$N$ 步数），与实测逐位吻合",
                "随机方向的有效性由 $|\\langle u_{\\min},d\\rangle|$ 决定，阈值与之成反比",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "trust_region_radius",
            "title": "信赖域半径自适应：接受、拒绝与边界",
            "description": (
                "Rosenbrock 函数上逐步解信赖域子问题（Steihaug-CG），按 $\\rho_k=(\\text{实际下降})/(\\text{模型下降})$"
                "接受或拒绝这一步，并把 $\\Delta_k$ 收缩 ($\\times\\gamma_1$) 或扩张 ($\\times\\gamma_2$)。"
                "图上同时给出半径历史、$\\rho_k$、以及半径为 $\\Delta_k$ 的圆；"
                "在 Hessian 不定的起点上还能看到子问题走负曲率方向撞到边界。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "t2",
                    "label": "扩张阈值 t2（ρ>t2 且撞边界才扩张）",
                    "type": "float",
                    "min": 0.05,
                    "max": 1.5,
                    "step": 0.05,
                    "default": 0.75,
                    "critical": [0.25, 1.0],
                    "criticalNote": (
                        "收缩阈值 $t_1=0.25$ 与扩张阈值必须满足 $0<t_1<t_2<1$："
                        "$t_2\\le t_1$ 时「收缩」与「扩张」的判据互相打架（同一个 $\\rho$ 两头都触发）；"
                        "$t_2\\ge1$ 时 $\\rho_k$ 几乎不可能超过它（$\\rho\\le1$ 是二次模型的性质），半径再也不扩张。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "delta0",
                    "label": "初始半径 Δ0",
                    "type": "float",
                    "min": 0.05,
                    "max": 10.0,
                    "step": 0.05,
                    "default": 1.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "start",
                    "label": "起点",
                    "type": "enum",
                    "options": ["valley", "indefinite", "high"],
                    "default": "valley",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "max_iter",
                    "label": "最大迭代步数",
                    "type": "int",
                    "min": 5,
                    "max": 40,
                    "default": 24,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 24.0, "dt": 1.0, "fps": 30, "label": "迭代 k"},
            "requiredAux": ["residual"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "Rosenbrock 等高线与信赖域圆", "ylabel": "y"},
                {"id": "aux", "autoscaled": True, "title": "半径 Δ_k、比值 ρ_k 与梯度范数", "ylabel": "Δ_k / ρ_k / ‖∇f‖", "scale": "log"},
            ],
            "pitfalls": (
                "「半径在自适应」这句话本身不是证据：$\\rho_k>0.75$ 只在步**撞到边界**时才扩张，"
                "而二次模型上 $\\rho$ 恒等于 $1$——用二次型看这个可视化，半径只会单调长大，"
                "一次拒绝都看不到。实测 Rosenbrock 从 $(-1.2,1)$ 出发 22 步里有 3 次拒绝、"
                "$\\rho$ 出现 $-4.25$，半径在 $0.125$ 与 $1$ 之间来回；这才是自适应在工作的样子。"
                "另外 $\\rho$ 为负时若还接受这一步，$f$ 会**上升**——$\\eta$ 阈值就是拦它的那道闸。"
            ),
            "theory": [
                "$\\rho_k=\\frac{f(x_k)-f(x_k+p_k)}{m_k(0)-m_k(p_k)}$，$\\rho_k>\\eta$ 才接受",
                "$\\rho<t_1=0.25$ 收缩半径、$\\rho>t_2$ 且 $\\|p\\|=\\Delta$ 扩张半径（$0<t_1<t_2<1$）",
                "Steihaug-CG 遇到负曲率方向时沿该方向走到信赖域边界",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "descent_race":
        return _descent_race_frame(params, t)
    if viz_id == "convergence_rates":
        return _convergence_rates_frame(params, t)
    if viz_id == "saddle_escape":
        return _saddle_escape_frame(params, t)
    if viz_id == "trust_region_radius":
        return _trust_region_frame(params, t)
    raise KeyError(f"章节 5.1 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# viz 1：下降路径赛跑
# --------------------------------------------------------------------------- #
_RACERS = ("gd", "momentum", "bfgs", "newton")


def _run_racer(name: str, Q: Any, x0: np.ndarray, alpha: float, max_iter: int) -> Any:
    if name == "gd":
        return gd(Q, x0, step=alpha, max_iter=max_iter, tol=1e-14)
    if name == "momentum":
        beta_star, alpha_star = momentum_parameters(float(Q.kappa))
        return momentum(Q, x0, step=alpha_star, beta=beta_star, max_iter=max_iter, tol=1e-14)
    if name == "bfgs":
        return bfgs(Q, x0, max_iter=max_iter, tol=1e-14)
    if name == "newton":
        return newton(Q, x0, max_iter=max_iter, tol=1e-14)
    raise KeyError(name)


def _descent_race_frame(params: Mapping[str, Any], t: float) -> Frame:
    kappa = float(params["kappa"])
    angle = float(params["angle_deg"])
    step_scale = float(params["step_scale"])
    max_iter = int(params["max_iter"])
    Q = anisotropic_quadratic(kappa, angle_deg=angle)
    x0 = np.array([1.0, 1.0], dtype=float)
    alpha = step_scale * float(Q.optimal_step())
    k = _step_index(t, max_iter)

    traces = {name: _run_racer(name, Q, x0, alpha, max_iter) for name in _RACERS}

    n_grid = 61
    X, Y, Z = contour_grid(Q, (-2.2, 2.2), (-2.2, 2.2), n=n_grid)

    fr = Frame(chapter="5.1", viz="descent_race", kind="frame")
    fr.add("contour_x", X[0, :], unit="", label="x_1")
    fr.add("contour_y", Y[:, 0], unit="", label="x_2")
    fr.add("contour_z", Z, unit="", label="f(x)（等高线用）")
    fr.add("iterations", np.arange(max_iter + 1, dtype=float), unit="", label="迭代 k")

    meta: dict[str, Any] = {}
    strings: dict[str, str] = {}
    n_exact = 0
    overflow = False
    for name in _RACERS:
        tr = traces[name]
        fr.add(f"path_x_{name}", tr.xs[:, 0], unit="", label=f"{name}: x_1")
        fr.add(f"path_y_{name}", tr.xs[:, 1], unit="", label=f"{name}: x_2")
        fr.add(f"error_{name}", _pad_tail(_floored(tr.dist), max_iter + 1), unit="", label=f"{name}: ‖x_k - x*‖")
        meta[f"n_iter_{name}"] = int(tr.n_iter)
        meta[f"diverged_{name}"] = bool(tr.stop == "diverged")
        strings[f"stop_{name}"] = tr.stop
        n_exact += int(np.count_nonzero(tr.dist <= 0.0))
        if tr.stop == "diverged" or float(np.max(np.abs(tr.xs))) > 2.2:
            overflow = True

    meta.update(
        kappa_measured=float(Q.kappa),
        mu=float(Q.mu),
        L=float(Q.L),
        alpha_used=alpha,
        alpha_over_stable=float(alpha * Q.L / 2.0),
        alpha_star=float(Q.optimal_step()),
        k_actual=int(k),
        kappa_theory=None if _finite_or_none(gd_iterations(kappa, 1e-8)) is None else int(round(gd_iterations(kappa, 1e-8))),
        n_exact_end=int(n_exact),
        error_floor=ERROR_FLOOR,
        overflow=bool(overflow),
    )
    fr.set_meta(**meta)
    fr.set_strings(
        **strings,
        status="ok",
        setup=(
            "GD 用 α = step_scale·α*；重球动量用经典最优 (β*,α*) = "
            f"({momentum_parameters(float(Q.kappa))[0]:.4f}, {momentum_parameters(float(Q.kappa))[1]:.3e})；"
            "BFGS 与牛顿自带线搜索。四个方法的步长口径不同，比较的是「路径形状」而不是「同一个 α」。"
        ),
        error_floor=f"误差曲线在地板 {ERROR_FLOOR:g} 处截平；meta.n_exact_end 给出真正到达 0 的方法-步数。",
        abort_note=(
            "meta.diverged_* 为 true 表示该方法在 max_iter 之前就跑到非有限值/超出发散判据，"
            "它的 error 曲线用尾部重复末点补位（不是「后来一直在那儿」）。"
        ),
        overflow_note="meta.overflow=true：至少一条路径走出了 ±2.2 的显示窗口，看不见不等于不存在。",
        data_source="backend",
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 2：收敛率图
# --------------------------------------------------------------------------- #
def _convergence_rates_frame(params: Mapping[str, Any], t: float) -> Frame:
    del t  # 时间轴 mode="none"：这一页的横轴是迭代数，不是时刻
    kappa = float(params["kappa"])
    start = float(params["start"])
    beta = float(params["beta"])
    alpha_cap = float(params["alpha_cap"])
    max_iter = int(params["max_iter"])

    F = quartic_quadratic(kappa, angle_deg=35.0, beta=beta)
    x0 = np.array([start, start], dtype=float)
    step = 1.0 / kappa  # 二次部分的 1/L：近原点处稳定
    beta_star, alpha_star = momentum_parameters(kappa)

    traces = {
        "gd": gd(F, x0, step=step, max_iter=max_iter, tol=1e-16),
        "momentum": momentum(F, x0, step=alpha_star, beta=beta_star, max_iter=max_iter, tol=1e-16),
        "bfgs": bfgs(F, x0, max_iter=max_iter, tol=1e-16),
        # 牛顿用极紧的容差：让轨迹**穿过**二次区一直走到舍入地板，
        # 这样 q_k 的爆炸（每步 ×u 的线性下降被误读成"超二次"）才会出现在图上。
        "newton": newton(F, x0, alpha_cap=alpha_cap, max_iter=max_iter, tol=1e-300),
    }

    fr = Frame(chapter="5.1", viz="convergence_rates", kind="frame")
    fr.add("iterations", np.arange(max_iter + 1, dtype=float), unit="", label="迭代 k")
    for name in ("gd", "momentum", "bfgs", "newton"):
        fr.add(f"error_{name}", _pad_tail(_floored(traces[name].dist), max_iter + 1), unit="", label=f"{name}: ‖x_k - x*‖")

    # 二次比值 q_k = e_{k+1}/e_k^2 与线性比值 r_k = e_{k+1}/e_k（都带有效性掩码）
    q_newton, q_valid = _ratio_curve(traces["newton"].dist, quadratic=True)
    r_gd, r_valid = _ratio_curve(traces["gd"].dist, quadratic=False)
    q_curve = _pad_tail(q_newton, max_iter, fill=0.0)
    r_curve = _pad_tail(r_gd, max_iter, fill=0.0)
    fr.add("q_newton", _floored(q_curve, RATIO_FLOOR), unit="", label="牛顿: e_{k+1}/e_k^2")
    fr.add("r_gd", _floored(r_curve, RATIO_FLOOR), unit="", label="GD: e_{k+1}/e_k")
    fr.add("q_newton_valid", _pad_mask(q_valid, max_iter), unit="", label="q 有效掩码")
    fr.add("r_gd_valid", _pad_mask(r_valid, max_iter), unit="", label="r 有效掩码")

    e_newton = traces["newton"].dist
    enter_iter, q_const, win = quadratic_window(e_newton)
    floor_start = _floor_start(e_newton)
    floor_rate = _tail_rate(e_newton)
    order_tail = convergence_order(e_newton)
    first_ratio = float(e_newton[1] / e_newton[0] ** 2) if e_newton.size > 1 and e_newton[0] > 0 else float("nan")

    fr.set_meta(
        kappa_measured=float(kappa),
        beta=beta,
        alpha_cap=alpha_cap,
        n_iter_newton=int(traces["newton"].n_iter),
        newton_enter_iter=None if enter_iter is None else int(enter_iter),
        newton_quadratic_constant=_finite_or_none(q_const),
        newton_first_ratio=_finite_or_none(first_ratio),
        newton_all_full_steps=bool(np.all(traces["newton"].alphas == 1.0)) if traces["newton"].alphas.size else False,
        newton_order_tail=_finite_or_none(order_tail),
        newton_floor_start=None if floor_start is None else int(floor_start),
        newton_floor_rate=_finite_or_none(floor_rate),
        newton_window_size=int(win.size),
        gd_linear_rate_measured=_finite_or_none(_tail_rate(traces["gd"].dist)),
        gd_linear_rate_theory=_finite_or_none(1.0 - step * float(anisotropic_quadratic(kappa).mu)),
        n_iter_gd=int(traces["gd"].n_iter),
        n_iter_bfgs=int(traces["bfgs"].n_iter),
        error_floor=ERROR_FLOOR,
        ratio_floor=RATIO_FLOOR,
        ratio_display_ceiling=1e10,
        overflow=False,
    )
    fr.set_strings(
        status="ok",
        stop_newton=traces["newton"].stop,
        stop_gd=traces["gd"].stop,
        stop_bfgs=traces["bfgs"].stop,
        stop_momentum=traces["momentum"].stop,
        quadratic_window=(
            "meta.newton_enter_iter 是实测的「进入二次区」序号：从它开始 "
            "e_{k+1}/e_k^2 才稳定在常数附近；在那之前牛顿步不是整步、比值没有意义。"
        ),
        floor_note=(
            f"误差曲线地板 {ERROR_FLOOR:g}，比值曲线地板 {RATIO_FLOOR:g}；"
            "掩码数组 q_newton_valid / r_gd_valid 标出哪些步的比值真的有定义"
            "（误差已经到 0 时比值无定义，写地板值只是为了让对数轴画得出来）。"
        ),
        exact_note=(
            "二次常数只在「还在二次区、又还没到舍入地板」的那几步上可测；"
            "β=0（退化成二次型）时牛顿一步到底、或用 alpha_cap<1 卡住步长时，"
            "meta.newton_quadratic_constant 与 newton_enter_iter 都写 null——**没有被测出来**，不是 0。"
        ),
        floor_break_note=(
            "meta.newton_floor_start 之后 q 会爆炸（实测到 1e122 量级）：那不是超二次收敛，"
            "而是到了舍入地板之后误差按每步 ×u 线性下降，q_k=e_{k+1}/e_k^2 自然发散；"
            "meta.newton_floor_rate 给出地板上实测的线性因子（应接近 u≈1.1e-16）。"
        ),
        data_source="backend",
    )
    return fr


def _ratio_curve(dist: np.ndarray, *, quadratic: bool) -> tuple[np.ndarray, np.ndarray]:
    """逐步的二次比值 :math:`e_{k+1}/e_k^2` 或线性比值 :math:`e_{k+1}/e_k`，附有效性掩码。"""
    e = np.asarray(dist, dtype=float).ravel()
    out: list[float] = []
    valid: list[bool] = []
    for k in range(e.size - 1):
        if e[k] > 0.0 and e[k + 1] > 0.0:
            out.append(float(e[k + 1] / (e[k] ** 2 if quadratic else e[k])))
            valid.append(True)
        else:
            out.append(0.0)
            valid.append(False)
    return np.asarray(out, dtype=float), np.asarray(valid, dtype=bool)


def _floor_start(dist: np.ndarray) -> int | None:
    r"""第一个 :math:`q_k=e_{k+1}/e_k^2>1` 的步号（舍入地板开始污染二次比值的点）。

    :math:`q_k>1` 意味着"这一步的下降比二次模型还快"，在双精度下只有一种来源：
    分母 :math:`e_k` 已经小到算不准，误差转而按每步 :math:`\times u` 线性下降。
    """
    e = np.asarray(dist, dtype=float).ravel()
    for k in range(e.size - 1):
        if e[k] > 0.0 and e[k + 1] > 0.0 and e[k + 1] / e[k] ** 2 > 1.0:
            return int(k)
    return None


def _tail_rate(dist: np.ndarray) -> float:
    """线性收敛因子（尾部几何平均）：只取仍在舍入地板之上的那一段。"""
    e = np.asarray(dist, dtype=float).ravel()
    e = e[(e > 1e-13)]
    if e.size < 3:
        return float("nan")
    ratios = e[1:] / e[:-1]
    ratios = ratios[ratios > 0.0]
    if ratios.size == 0:
        return float("nan")
    return float(np.exp(float(np.mean(np.log(ratios[-min(5, ratios.size) :])))))


# --------------------------------------------------------------------------- #
# viz 3：鞍点逃逸
# --------------------------------------------------------------------------- #
def _direction_for(kind: str, u_min: np.ndarray) -> np.ndarray:
    if kind == "neg_curv":
        return np.asarray(u_min, dtype=float)
    if kind == "random":
        return _random_unit(20260501)
    if kind == "orthogonal":
        return np.array([-u_min[1], u_min[0]], dtype=float)
    raise KeyError(kind)


def _saddle_escape_frame(params: Mapping[str, Any], t: float) -> Frame:
    neg = float(params["neg"])
    delta = float(params["delta"])
    direction = str(params["direction"])
    max_iter = int(params["max_iter"])
    S = saddle_quadratic(neg=neg, pos=1.0)
    lam_min, u_min = min_eigenpair(S.hess(np.zeros(2)))
    u_min = u_min / float(np.sqrt(float(np.dot(u_min, u_min))))
    d = _direction_for(direction, u_min)
    d = d / float(np.sqrt(float(np.dot(d, d))))
    overlap = abs(float(np.dot(u_min, d)))
    k = _step_index(t, max_iter)

    step = 1.0  # 1/L（L=1：正曲率方向的稳定上限），也保证负曲率方向按 1+|λ-| 增长
    traj = gd(S, delta * d, step=step, max_iter=max_iter, tol=1e-300)
    stable = gd(S, np.array([0.0, 1e-3], dtype=float), step=step, max_iter=max_iter, tol=1e-300)

    n_grid = 61
    X, Y, Z = contour_grid(S, (-2.5, 2.5), (-2.5, 2.5), n=n_grid)

    fr = Frame(chapter="5.1", viz="saddle_escape", kind="frame")
    fr.add("contour_x", X[0, :], unit="", label="x_1")
    fr.add("contour_y", Y[:, 0], unit="", label="x_2")
    fr.add("contour_z", Z, unit="", label="f(x)（双曲线族）")
    tx, ty = _slice_traj(traj.xs[:, 0], traj.xs[:, 1], k)
    fr.add("traj_x", tx, unit="", label="扰动轨迹 x_1")
    fr.add("traj_y", ty, unit="", label="扰动轨迹 x_2")
    fr.add("residual", _floored(traj.gnorm[: tx.size]), unit="", label="‖∇f‖（扰动轨迹）")
    fr.add("error_to_saddle", _floored(traj.dist[: tx.size]), unit="", label="‖x - x*‖（扰动轨迹）")
    fr.add("stable_x", stable.xs[:, 0], unit="", label="稳定流形上的 GD: x_1")
    fr.add("stable_y", stable.xs[:, 1], unit="", label="稳定流形上的 GD: x_2")

    # 阈值只看 (λ-, α, N, R)，与扰动幅度 δ、scrub 时刻都无关。N 必须与时间轴上的
    # 最大步数一致：否则会出现「δ 已经越过 N=96 的阈值、当前距离也超过 1，但按 96 步
    # 判据却写尚未逃逸」的自相矛盾。二维 GD 的代价很小，直接用 max_iter 做二分。
    n_esc = int(max(1, max_iter))
    esc = escape_threshold(S, np.zeros(2), u_min, step=step, max_iter=n_esc, radius=1.0, tol=1e-4, bisect=40, max_expand=40)
    esc_rand = escape_threshold(S, np.zeros(2), _random_unit(20260501), step=step, max_iter=n_esc, radius=1.0, tol=1e-4, bisect=40, max_expand=40)

    fr.set_meta(
        lambda_min=float(lam_min),
        u_min_x=float(u_min[0]),
        u_min_y=float(u_min[1]),
        direction_overlap=float(overlap),
        delta=delta,
        step=step,
        k_actual=int(k),
        traj_final_distance=_finite_or_none(traj.dist[tx.size - 1]),
        traj_final_residual=_finite_or_none(traj.gnorm[tx.size - 1]),
        stable_final_distance=_finite_or_none(stable.dist[-1]),
        threshold_neg_curv=_finite_or_none(esc.threshold),
        threshold_random=_finite_or_none(esc_rand.threshold),
        threshold_ratio=_finite_or_none(esc_rand.threshold / esc.threshold) if esc.threshold > 0 else None,
        escaped_random=bool(esc_rand.escaped),
        escaped_neg_curv=bool(esc.escaped),
        escape_steps=n_esc,
        theory_threshold=_finite_or_none((1.0 + abs(neg) * step) ** (-n_esc)),
        error_floor=ERROR_FLOOR,
        overflow=bool(float(np.max(np.abs(traj.xs))) > 2.5 or float(np.max(np.abs(stable.xs))) > 2.5),
    )
    fr.set_strings(
        status="ok",
        eigen_note=(
            "meta.lambda_min 与 (u_min_x, u_min_y) 是**实测**的（对 Hessian 做 Jacobi 特征分解），"
            "不是把参数 neg 抄回来；两者在 λ-<0 时逐位一致。"
        ),
        threshold_note=(
            "实测逃逸阈值 = 二分出来的最小扰动量级 δ*（判据：meta.escape_steps 步 GD 之后离驻点 ≥ 1，"
            "二分相对容差 1e-4）。"
            "沿 u_min 的阈值与解析式 R·(1+|λ-|α)^(-N) 吻合；随机方向的阈值是它的 "
            "1/|<u_min,d>| 倍——这就是「随机扰动跑掉了」的全部内容。"
        ),
        null_note=(
            "meta.threshold_* 为 null 表示「放大到 1e12 也没找到逃逸」"
            "（方向与负曲率正交、或步数不足以放大），不是阈值为 0。"
        ),
        floor_note=f"residual / error_to_saddle 的对数地板是 {ERROR_FLOOR:g}（梯度为 0 时写地板值）。",
        overflow_note="meta.overflow=true：轨迹冲出了 ±2.5 的显示窗口（鞍点的负曲率方向本来就不收敛）。",
        data_source="backend",
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：信赖域半径自适应
# --------------------------------------------------------------------------- #
_TR_STARTS: dict[str, tuple[float, float]] = {
    "valley": (-1.2, 1.0),
    "indefinite": (0.5, 3.0),
    "high": (-3.0, 10.0),
}


def _trust_region_frame(params: Mapping[str, Any], t: float) -> Frame:
    t2 = float(params["t2"])
    delta0 = float(params["delta0"])
    start = str(params["start"])
    max_iter = int(params["max_iter"])
    if start not in _TR_STARTS:
        raise KeyError(f"未知起点 {start!r}（valley / indefinite / high）")
    R = rosenbrock_objective()
    x0 = np.array(_TR_STARTS[start], dtype=float)
    tr = trust_region(R, x0, delta0=delta0, t2=t2, max_iter=max_iter, tol=1e-14)
    k = _step_index(t, max_iter)

    # 61×61 时 Rosenbrock 窄谷里的等高线会在屏幕上碎成锯齿；显示底图加密到 121。
    n_grid = 121
    X, Y, Z = contour_grid(R, (-3.5, 3.5), (-1.5, 12.0), n=n_grid)

    fr = Frame(chapter="5.1", viz="trust_region_radius", kind="frame")
    fr.add("contour_x", X[0, :], unit="", label="x")
    fr.add("contour_y", Y[:, 0], unit="", label="y")
    fr.add("contour_z", Z, unit="", label="f(x,y)（Rosenbrock）")
    radius_clean, _ = _finite_array(tr.radius, fill=0.0)
    tx, ty = _slice_traj(tr.xs[:, 0], tr.xs[:, 1], k)
    fr.add("traj_x", tx, unit="", label="迭代点 x")
    fr.add("traj_y", ty, unit="", label="迭代点 y")
    fr.add("iterations", np.arange(max_iter + 1, dtype=float), unit="", label="迭代 k")
    fr.add("gradient_residual", _pad_tail(_floored(tr.gnorm), max_iter + 1), unit="", label="‖∇f‖")
    fr.add("radius_history", _pad_tail(_floored(radius_clean, 1e-14), max_iter + 1), unit="", label="信赖域半径 Δ_k")
    ratio_series, ratio_ok = _finite_array(tr.ratio, fill=0.0)
    fr.add("ratio_history", _pad_tail(ratio_series, max_iter, fill=0.0), unit="", label="比值 ρ_k")
    fr.add("ratio_valid", _pad_mask(ratio_ok, max_iter), unit="", label="ρ_k 有限掩码（0 是占位符）")
    fr.add("accepted", _pad_tail(np.asarray(tr.accepted, dtype=float), max_iter, fill=0.0), unit="", label="接受掩码 1/0")
    fr.add("neg_curv", _pad_tail(tr.extra("neg_curv"), max_iter, fill=0.0), unit="", label="负曲率分支掩码")

    radius_now = float(radius_clean[k]) if radius_clean.size > k else delta0
    # k 步已经走完：第 k 步的结果是 ratio[k-1]；k=0 时还没有任何一步发生，
    # 这里必须写 null（“没有测到”），不能拿第一步的未来值冒充当前读数。
    ratio_now = float(ratio_series[k - 1]) if k >= 1 and ratio_series.size >= k else None
    fr.set_meta(
        k_actual=int(k),
        t2=t2,
        delta0=delta0,
        radius_current=_finite_or_none(radius_now),
        ratio_current=_finite_or_none(ratio_now),
        accepted_current=bool(bool(tr.accepted[k - 1])) if k >= 1 and tr.accepted.size >= k else False,
        n_iter=int(tr.n_iter),
        n_accepted=int(np.count_nonzero(tr.accepted)),
        n_rejected=int(tr.accepted.size - np.count_nonzero(tr.accepted)),
        radius_min=_finite_or_none(np.min(radius_clean)) if radius_clean.size else None,
        radius_max=_finite_or_none(np.max(radius_clean)) if radius_clean.size else None,
        ratio_min=_finite_or_none(np.min(ratio_series)) if ratio_series.size else None,
        n_negative_curv=int(np.count_nonzero(tr.extra("neg_curv"))),
        final_dist=_finite_or_none(tr.dist[-1]),
        radius_floor=1e-14,
        error_floor=ERROR_FLOOR,
        overflow=bool(float(np.max(np.abs(tr.xs))) > 3.5 or float(np.max(np.abs(tr.xs[:, 1]))) > 12.0),
    )
    fr.set_strings(
        status="ok",
        stop=tr.stop,
        radius_note=(
            "半径只在「ρ>t2 且步撞到边界」时扩张、在「ρ<t1=0.25」时按 γ₁=0.25 收缩；"
            "被拒绝的步不进轨迹，但 ratio_history / accepted 里都有记录。"
        ),
        ratio_note=(
            "ρ 可以远大于 1（模型低估了实际下降）也可以为负（这一步会让 f 上升）；"
            "ratio_history 里为 0 的位置是占位符（模型下降量 ≤0 时比值无定义，或那一步还没发生），"
            "必须配合 ratio_valid 与 accepted 两个掩码读。"
        ),
        neg_curv_note=(
            "meta.n_negative_curv 是 Steihaug-CG 走「负曲率 → 沿该方向打到边界」分支的次数；"
            "起点 indefinite (0.5, 3) 上 Rosenbrock 的 Hessian 行列式为负，这个分支才会触发。"
        ),
        floor_note=f"半径与梯度范数的对数地板 {1e-14:g} / {ERROR_FLOOR:g}；半径不会被压到 0。",
        overflow_note="meta.overflow=true：迭代点（或初值）越出了显示窗口。",
        data_source="backend",
    )
    return fr
