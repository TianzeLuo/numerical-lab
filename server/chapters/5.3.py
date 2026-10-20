"""5.3 章：伴随方法与灵敏度分析（帧装配层）。

四个可视化各自回答一个**会被讲错**的问题：

1. ``gradient_field`` —— 参数平面上的灵敏度方向。等高线上叠**伴随梯度**箭头，
   再用一个子网格把伴随梯度与中心差分逐点对拍（相对误差必须 < 1e-6）。
   共振脊在 :math:`p_1=\\omega^2` 附近，阻尼把它压平、三次项把它掰弯。
2. ``influence_field`` —— 伴随变量的"影响场"。它随**反向时间** :math:`\\tau=T-t`
   扩散：宽度按 :math:`\\sigma_0^2+2\\alpha\\tau` 长，**质心不动**（实测漂移 1e-16），
   扩散到 Dirichlet 壁之后宽度律失效（掩码标出）。
3. ``gradient_check`` —— 梯度检验的 V 形曲线：同一个 :math:`\\varepsilon` 扫描上
   截断段（斜率 ≈ 差分阶）与舍入段（斜率 ≈ -1）、**实测最优** :math:`\\varepsilon`、
   以及它与 :math:`u^{1/3}` / :math:`u^{1/2}` 的比值；副轴放"伴随 1 次 vs 有限差分
   :math:`2k` 次"的代价三点（:math:`k=2/8/32`）。
4. ``forward_adjoint_playback`` —— 正向轨迹与反向伴随轨迹叠加播放。播放头处给出
   :math:`\\mu(t)` 的**脉冲响应**含义，并用有限差分验证 :math:`\\Delta J\\approx\\mu^T\\delta`
   （残差进 ``sensitivity_residual`` 数组，这是这一页的"必备对照量"）。

帧纪律（与 0.0/1.2/5.1 章一致）：``meta`` 只放 number/bool/None；字符串走 ``strings``；
没有有限值写 ``None`` 并在 ``strings`` 里说明；数组里的非有限值一律"占位值 + 掩码 +
文字说明"三件套；``server/`` 里不写数学，所有量都来自 :mod:`numcore.inverse.adjoint` 与
:mod:`numcore.inverse.sensitivity`。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from numcore.inverse.adjoint import (
    GradientCheck,
    HeatAdjointProblem,
    UNIT_ROUNDOFF,
    continuous_adjoint,
    discrete_adjoint,
    gradient_check_scan,
    relative_error,
    rk_forward,
    rk_forward_from,
)
from numcore.inverse.sensitivity import (
    cost_scaling_study,
    duffing_model,
    duffing_slice_model,
    gradient_field,
    heat_boundary_layer_ratio,
    influence_width_law,
    log_scaled_length,
    sensitivity_directions,
    trajectory_norms,
)
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 对数轴上的地板（误差、比值曲线；0 在对数轴画不出来，见 PROTOCOLS §1.3.1）
ERROR_FLOOR = 1e-18
#: 相对误差曲线的地板（log10 之后）
LOG10_FLOOR = -18.0
#: 影响场的显示地板（占位值：影响场按峰值归一化后小于它的位置画不出来）
FIELD_FLOOR = 0.0


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


def _finite_array(values: Any, *, fill: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """把数组里的 ``NaN/±Inf`` 换成占位值并给出掩码（PROTOCOLS §1.3.1 三件套）。"""
    v = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(v)
    return np.where(ok, v, float(fill)), ok


def _floored(values: Any, floor: float = ERROR_FLOOR) -> np.ndarray:
    """把非负曲线抬到对数轴地板上。"""
    return np.maximum(np.asarray(values, dtype=float), float(floor))


def _fraction(t: float) -> float:
    """把 scrub 的 ``t`` 夹到 ``[0, 1]``（时间轴的语义是"总时长的比例"）。"""
    v = float(t)
    if not math.isfinite(v):
        return 0.0
    return min(max(v, 0.0), 1.0)


def _add_checked(fr: Frame, name: str, values: Any, *, unit: str = "", label: str = "", fill: float = 0.0) -> bool:
    """加一条数组；出现非有限值时**同时**加掩码数组，并返回"是否有占位值"。"""
    v, ok = _finite_array(values, fill=fill)
    fr.add(name, v, unit=unit, label=label)
    if not bool(np.all(ok)):
        fr.add(f"{name}_valid", ok.astype(float), unit="", label=f"{name} 有效掩码（0=占位值）")
        return True
    return False


def _model_common(params: Mapping[str, Any]) -> tuple[Any, np.ndarray, float, int]:
    """构造两参数切片模型并取出 ``(model, p, T, n_steps)``。"""
    cubic = float(params["cubic"])
    p = np.array([float(params["stiffness"]), float(params["damping"])], dtype=float)
    return duffing_slice_model(cubic), p, float(params["horizon"]), int(params["n_steps"])


#: 梯度检验用的固定工作点（三参数 Duffing：p1=1.0 略偏共振、p2=0.1、p3=0.05）。
#: 只有 ``param_index`` 指向的那个分量被 eps 扫描检验，另外两个保持在这里。
CHECK_POINT = (1.0, 0.1, 0.05)
#: 梯度检验的固定积分时长（与 V 形扫描的步数 N 一起决定 h = T/N）
CHECK_HORIZON = 8.0


def _three_param(params: Mapping[str, Any]) -> tuple[Any, np.ndarray, float, int]:
    """构造三参数 Duffing 模型（V 形曲线与方向分析用它）：工作点固定在 :data:`CHECK_POINT`。"""
    model = duffing_model()
    p = np.array(CHECK_POINT, dtype=float)
    return model, p, float(CHECK_HORIZON), int(params["n_steps"])


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
_MANIFEST_PITFALLS_FIELD = (
    "把箭头的长度当成'灵敏度大小'会直接读反：这里画的是归一化方向，长度按对数压缩，"
    "靠近共振脊时梯度范数跨好几个量级，若不压缩就会只剩一根箭头。"
    "更要命的是'等高线密 = 灵敏度大'：等高线的疏密是 $\\|\\nabla J\\|$ 的倒数，但方向箭头的含义是"
    "'往哪儿调参数'，两者只有在参数同量纲时才可比；$p_1$ 与 $p_2$ 量纲不同（刚度 vs 阻尼），"
    "跨参数的比较必须换成绩效弹性 $\\frac{\\partial J}{\\partial p_j}\\frac{p_j}{J}$。"
)

_MANIFEST_PITFALLS_INFLUENCE = (
    "把'影响场在反向扩散'读成'峰值在往回跑'是错的：纯扩散下伴随场的质心严格不动"
    "（实测漂移 1e-16），长的是**宽度**。另一个坑是把扩散到边界之后的宽度继续套"
    "$\\sigma^2=\\sigma_0^2+2\\alpha\\tau$：Dirichlet 壁会吸收影响场，实测宽度比连续律小 1%"
    "（$\\tau=0.2$、$\\sigma_0=0.05$），图上用掩码标出被污染的行，读到掩码以外就该停。"
)

_MANIFEST_PITFALLS_CHECK = (
    "V 形曲线的两段不能只看形状：截断段与舍入段的斜率必须**分别**拟合出来"
    "（实测中心差分 +1.99 / -1.00，单侧差分 +0.99 / -0.99），"
    "而'最优 $\\varepsilon$'必须落在扫描区间内部才有意义——落在端点说明窗口没盖住 V 形，"
    "此时 meta 里的最优 $\\varepsilon$ 与斜率都该当 null 读。"
    "'$\\varepsilon$ 越小越准'在这里被直接否掉：实测最优点在 $\\varepsilon\\approx10^{-6}$（中心差分），"
    "再小三个数量级误差反而涨 100 倍。"
)

_MANIFEST_PITFALLS_PLAYBACK = (
    "把伴随轨迹与正向轨迹画在同一张相平面上很好看，但两者活在**不同的空间**里："
    "$y$ 是状态，$\\mu$ 是余状态（对偶空间）。有意义的定量陈述只有一条："
    "$\\mu(t)$ 的第 $i$ 个分量是 $J$ 对 $y_i(t)$ 的敏感度，即 $\\Delta J\\approx\\mu(t)^T\\delta$。"
    "另一处容易读错的是'伴随往回走会发散'：那是正向收缩的镜像（实测 $\\|\\mu(0)\\|/\\|\\mu(T)\\|=34$ 倍），"
    "不是数值不稳定——把步长加密它不变，把阻尼调大它变小。"
)


MANIFEST: dict[str, Any] = {
    "chapter": "5.3",
    "title": "伴随方法与灵敏度分析",
    "part": "V",
    "summary": (
        "离散伴随与连续伴随的分界、梯度检验的 V 形曲线（截断 vs 舍入）、"
        "伴随代价与参数个数无关的实测账、以及伴随变量作为影响场/灵敏度方向的读法。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "gradient_field",
            "title": "参数平面上的梯度场：灵敏度方向与共振脊",
            "description": (
                "受迫 Duffing 振子在 $(p_1,p_2)$ =（线性刚度, 阻尼）平面上的终端能量 $J=\\|y(T)\\|^2/2$ 等高线，"
                "叠上逐点的**伴随梯度方向**（归一化箭头）与中心差分方向；"
                "第三张图给出两者的 $\\log_{10}$ 相对误差。小振幅下共振脊在 $p_1=\\omega^2=1$ 附近，"
                "阻尼把它压平，三次项把它掰弯。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "stiffness",
                    "label": "线性刚度 p1（共振脊在 p1 = omega^2）",
                    "type": "float",
                    "min": 0.6,
                    "max": 1.6,
                    "step": 0.02,
                    "default": 1.0,
                    "critical": [1.0],
                    "criticalNote": (
                        "小振幅、小阻尼时受迫振子的响应峰在 $p_1=\\omega^2=1$（激励频率 $\\omega=1$）。"
                        "跨过它，终端能量的量级会变几个数量级——截面上的等高线会从'一条脊'变成'一片坡'。"
                    ),
                    "beyond": "magnitude",
                },
                {
                    "name": "damping",
                    "label": "阻尼 p2",
                    "type": "float",
                    "min": 0.03,
                    "max": 0.4,
                    "step": 0.005,
                    "default": 0.08,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "cubic",
                    "label": "三次项 p3（p3=0 退化成线性振子）",
                    "type": "float",
                    "min": 0.0,
                    "max": 0.4,
                    "step": 0.01,
                    "default": 0.05,
                    "critical": [0.0],
                    "criticalNote": (
                        "$p_3=0$ 时振子线性，共振处的响应幅值 $\\sim F/(2\\zeta)$ 可以很大，"
                        "而且**连续伴随与离散伴随的差会退到舍入级**（非线性正是两者分开的来源）。"
                        "越过这条线往右，脊被非线性掰弯、梯度方向在脊两侧翻转。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "horizon",
                    "label": "积分时长 T",
                    "type": "float",
                    "min": 10.0,
                    "max": 60.0,
                    "step": 1.0,
                    "default": 30.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_steps",
                    "label": "RK4 步数 N",
                    "type": "int",
                    "min": 64,
                    "max": 320,
                    "step": 8,
                    "default": 144,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "arrows",
                    "label": "每边箭头数",
                    "type": "int",
                    "min": 3,
                    "max": 6,
                    "step": 1,
                    "default": 4,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "plane", "autoscaled": False, "title": "参数平面 (p1, p2) 上的终端能量与伴随梯度方向", "ylabel": "p2"},
                {"id": "error", "autoscaled": True, "title": "伴随梯度 vs 中心差分：log10 相对误差（必须 < -6）", "ylabel": "log10 相对误差"},
            ],
            "pitfalls": _MANIFEST_PITFALLS_FIELD,
            "theory": [
                "伴随梯度是离散目标函数 $J_h$ 的精确梯度：与中心差分的差只受 $\\varepsilon$ 的截断/舍入权衡限制",
                "终端能量的灵敏度方向就是伴随变量在 $t=T$ 处的值：$\\mu(T)=\\left(\\frac{\\partial J}{\\partial y(T)}\\right)^T=y(T)$",
                "受迫振子的响应峰在 $p_1=\\omega^2$ 附近，阻尼把峰值压到 $\\sim F/(2\\zeta)$，非线性项使峰位随幅值移动",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "influence_field",
            "title": "伴随变量的影响场：随反向时间扩散",
            "description": (
                "一维热方程 $u_t=\\alpha u_{xx}$ 加源项，目标 $J=w^T u(T)$（$w$ 是以 obs_center 为中心的钟形观测）。"
                "伴随变量 $\\mu$ 从 $\\mu(T)=w$ 出发沿反向时间 $\\tau=T-t$ 演化："
                "主图给出 $\\mu(x,\\tau)$ 的场，副图给出实测宽度 $\\sqrt{\\mathrm{Var}}$ 与连续律 "
                "$\\sqrt{\\sigma_0^2+2\\alpha\\tau}$ 的对照、以及质心漂移。"
                "扩散到 Dirichlet 壁之后宽度律失效，被污染的行用掩码标出。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "alpha",
                    "label": "扩散系数 alpha",
                    "type": "float",
                    "min": 0.02,
                    "max": 0.15,
                    "step": 0.005,
                    "default": 0.05,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "horizon",
                    "label": "总时长 T",
                    "type": "float",
                    "min": 0.1,
                    "max": 0.4,
                    "step": 0.01,
                    "default": 0.2,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "obs_width",
                    "label": "观测宽度 sigma_0",
                    "type": "float",
                    "min": 0.02,
                    "max": 0.1,
                    "step": 0.005,
                    "default": 0.05,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "obs_center",
                    "label": "观测中心 x_obs",
                    "type": "float",
                    "min": 0.25,
                    "max": 0.75,
                    "step": 0.01,
                    "default": 0.5,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_steps",
                    "label": "时间步数 N",
                    "type": "int",
                    "min": 200,
                    "max": 1600,
                    "step": 100,
                    "default": 800,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "scheme",
                    "label": "时间格式",
                    "type": "enum",
                    "options": ["cn", "btcs", "ftcs"],
                    "default": "cn",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.02, "fps": 30, "label": "反向时间比例 tau/T"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "field", "autoscaled": False, "title": "影响场 mu(x, tau)（按 tau=0 的峰值归一化）", "ylabel": "反向时间 tau"},
                {"id": "width", "autoscaled": True, "title": "实测宽度与连续律的偏差、质心漂移", "ylabel": "|宽度比 - 1| / |质心漂移|"},
            ],
            "pitfalls": _MANIFEST_PITFALLS_INFLUENCE,
            "theory": [
                "正向抛物型问题的伴随是**反向**抛物型：把 $\\tau=T-t$ 倒过来就是正向热方程，所以它是适定的",
                "高斯初值在纯扩散下保持高斯：$\\mathrm{Var}(\\tau)=\\sigma_0^2+2\\alpha\\tau$，质心不动",
                "离散伴随要解 $M^T$（Dirichlet 的恒等行让 $M$ 不对称）；用 $M$ 代替 $M^T$ 会在边界层出错",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "gradient_check",
            "title": "梯度检验的 V 形曲线：截断误差 vs 舍入误差",
            "description": (
                "同一个 $\\varepsilon$ 扫描上画两条曲线：中心差分与单侧差分的相对误差。"
                "左边下降段是截断误差（中心差分 $\\sim\\varepsilon^2$、单侧 $\\sim\\varepsilon$），右边上升段是舍入误差（$\\sim u/\\varepsilon$），"
                "谷底就是**实测最优** $\\varepsilon$。副轴给出代价三点：参数个数 $k=2/8/32$ 时"
                "伴随一次梯度与有限差分 $2k$ 次正演的求值次数/求解次数对比。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "eps_min",
                    "label": "扫描下界 eps",
                    "type": "float",
                    "min": 1e-16,
                    "max": 1e-4,
                    "step": 1e-16,
                    "default": 1e-16,
                    "critical": [1e-4],
                    "criticalNote": (
                        "下界抬到 $10^{-4}$ 就把整个窗口挪到 V 形的**右侧**（只剩截断段）："
                        "误差随 $\\varepsilon$ 单调上升，'最优 $\\varepsilon$'落在端点上——那时 meta 里的 eps_opt_* 必须是 null、"
                        "*_interior 必须是 false。窗口没盖住谷底就没有'最优'可言。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "eps_max",
                    "label": "扫描上界 eps",
                    "type": "float",
                    "min": 1e-4,
                    "max": 0.1,
                    "step": 1e-4,
                    "default": 0.1,
                    "critical": [1e-6],
                    "criticalNote": (
                        "扫到 $10^{-6}$ 量级（中心差分的理论最优 $\\varepsilon\\sim u^{1/3}=4.8\\times10^{-6}$）才盖住 V 形的谷底；"
                        "上界小于它时曲线只有下降段，meta 里的最优 $\\varepsilon$ 会落在端点上，"
                        "那种情形下'更小更准'是**扫描窗口的假象**而不是结论。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "n_eps",
                    "label": "扫描点数",
                    "type": "int",
                    "min": 9,
                    "max": 41,
                    "step": 1,
                    "default": 25,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "param_index",
                    "label": "被检验的参数下标（0=p1, 1=p2, 2=p3）",
                    "type": "int",
                    "min": 0,
                    "max": 2,
                    "step": 1,
                    "default": 0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_steps",
                    "label": "RK 步数 N",
                    "type": "int",
                    "min": 32,
                    "max": 256,
                    "step": 8,
                    "default": 96,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "method",
                    "label": "时间格式",
                    "type": "enum",
                    "options": ["rk4", "heun", "euler"],
                    "default": "rk4",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "vcurve", "autoscaled": True, "title": "梯度检验：相对误差 vs eps（双对数）", "ylabel": "|g_fd - g_adj| 相对误差", "scale": "log"},
                {"id": "cost", "autoscaled": True, "title": "代价：伴随 1 次 vs 有限差分 2k 次的求值次数", "ylabel": "求值次数 / 求解次数", "scale": "log"},
            ],
            "pitfalls": _MANIFEST_PITFALLS_CHECK,
            "theory": [
                "中心差分的截断误差 $\\sim\\varepsilon^2$、舍入误差 $\\sim u/\\varepsilon$，最优 $\\varepsilon\\sim u^{1/3}$；单侧差分对应 $u^{1/2}$",
                "离散伴随一次求解给出全部 $k$ 个偏导：代价与参数个数无关（有限差分要 $2k$ 次正演）",
                "伴随法不免费：需要整条正向轨迹（或分块检查点重算），存储/重算的账要单独记",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "forward_adjoint_playback",
            "title": "正向轨迹与反向伴随轨迹叠加播放",
            "description": (
                "相平面上同屏播放两条轨迹：正向 $y(t)$ 从 0 走到播放头，"
                "伴随 $\\mu(t)$ 从 $T$ 倒着走回播放头，两者在播放头相遇。"
                "播放头处给出 $\\mu$ 的方向（脉冲响应：$\\Delta J\\approx\\mu^T\\delta$）并用有限差分验证，"
                "残差写进 sensitivity_residual。副轴画 $\\|y\\|$ 与 $\\|\\mu\\|$ 随时间的走向。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "stiffness",
                    "label": "线性刚度 p1",
                    "type": "float",
                    "min": 0.6,
                    "max": 1.6,
                    "step": 0.02,
                    "default": 1.0,
                    "critical": [1.0],
                    "criticalNote": (
                        "$p_1=\\omega^2=1$ 是共振脊：跨过它，正向轨迹的振幅与伴随轨迹的走向同时换相，"
                        "播放头处的灵敏度箭头会翻向。"
                    ),
                    "beyond": "magnitude",
                },
                {
                    "name": "damping",
                    "label": "阻尼 p2（可以为负：负阻尼 = 泵浦）",
                    "type": "float",
                    "min": -0.1,
                    "max": 0.4,
                    "step": 0.005,
                    "default": 0.08,
                    "critical": [0.0],
                    "criticalNote": (
                        "$p_2=0$ 是**流向分界**：$p_2>0$ 时正向流收缩，伴随往回走也跟着衰减"
                        "（实测 $\\|\\mu(0)\\|/\\|\\mu(T)\\|=0.25$）；$p_2<0$ 时正向流扩张，"
                        "伴随往回走**放大**（同一个比值 $>1$）。'伴随往回走会放大'这句话只在一半的参数区间成立，"
                        "过零时它掉头——播放头上 $\\|\\mu\\|$ 的斜率会整体换号。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "cubic",
                    "label": "三次项 p3",
                    "type": "float",
                    "min": 0.0,
                    "max": 0.4,
                    "step": 0.01,
                    "default": 0.05,
                    "critical": [0.0],
                    "criticalNote": (
                        "$p_3=0$ 时振子线性：正向轨迹退化成单频衰减，反向伴随也退化成同频衰减，"
                        "两者在相平面上变成一对相似椭圆；$p_3>0$ 之后出现倍频与相移，"
                        "伴随轨迹的形状不再与正向轨迹相似（这正是连续/离散伴随分开的来源）。"
                    ),
                },
                {
                    "name": "horizon",
                    "label": "积分时长 T",
                    "type": "float",
                    "min": 10.0,
                    "max": 60.0,
                    "step": 1.0,
                    "default": 30.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_steps",
                    "label": "RK4 步数 N",
                    "type": "int",
                    "min": 64,
                    "max": 320,
                    "step": 8,
                    "default": 192,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.02, "fps": 30, "label": "播放头比例 t/T"},
            "requiredAux": ["residual"],
            "axes": [
                {"id": "phase", "autoscaled": True, "title": "相平面：正向轨迹（到头）与反向伴随轨迹（从尾）", "ylabel": "y2 / mu2"},
                {"id": "norms", "autoscaled": True, "title": "|y(t)| 与 |mu(t)|（半对数）", "ylabel": "范数", "scale": "log"},
            ],
            "pitfalls": _MANIFEST_PITFALLS_PLAYBACK,
            "theory": [
                "伴随变量 $\\mu(t)$ 是终端目标对状态的敏感度：$\\Delta J\\approx\\mu(t)^T\\delta$（一阶）",
                "离散伴随反向递推 $\\mu_n=\\left(\\frac{\\partial\\Phi_n}{\\partial y_n}\\right)^T\\mu_{n+1}$，起点 $\\mu_N=\\left(\\frac{\\partial g}{\\partial y_N}\\right)^T$",
                "正向收缩越快，反向伴随放大越猛：$\\|\\mu(0)\\|/\\|\\mu(T)\\|$ 是正向压缩率的倒数",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "gradient_field":
        return _gradient_field_frame(params, t)
    if viz_id == "influence_field":
        return _influence_field_frame(params, t)
    if viz_id == "gradient_check":
        return _gradient_check_frame(params, t)
    if viz_id == "forward_adjoint_playback":
        return _playback_frame(params, t)
    raise KeyError(f"章节 5.3 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# viz 1：参数平面上的梯度场
# --------------------------------------------------------------------------- #
def _gradient_field_frame(params: Mapping[str, Any], t: float) -> Frame:
    del t  # mode="none"
    model, _, T, n_steps = _model_common(params)
    n_arrow = int(params["arrows"])
    p1_lo, p1_hi = 0.6, 1.6
    p2_lo, p2_hi = 0.03, 0.4
    gf = gradient_field(
        model,
        p1_range=(p1_lo, p1_hi),
        p2_range=(p2_lo, p2_hi),
        n1=11,
        n2=7,
        arrow_n1=n_arrow,
        arrow_n2=n_arrow,
        T=T,
        n_steps=n_steps,
    )

    sd = sensitivity_directions(
        model,
        np.array([float(params["stiffness"]), float(params["damping"])], dtype=float),
        T=T,
        n_steps=n_steps,
    )

    fr = Frame(chapter="5.3", viz="gradient_field", kind="frame")
    placed = _add_checked(fr, "objective", gf.objective, unit="", label="终端能量 J = |y(T)|^2/2")
    fr.add("elasticity", sd.elasticity, unit="", label="绩效弹性 (dJ/dp_j)(p_j/J)（滑块工作点）")
    fr.add("gradient_center", sd.grad_adjoint, unit="", label="滑块工作点上的伴随梯度")
    fr.add("grid_p1", gf.p1, unit="", label="p1 网格")
    fr.add("grid_p2", gf.p2, unit="", label="p2 网格")
    fr.add("arrow_p1", gf.arrow_p1, unit="", label="箭头位置 p1")
    fr.add("arrow_p2", gf.arrow_p2, unit="", label="箭头位置 p2")
    fr.add("adjoint_dir1", gf.dir1, unit="", label="伴随梯度方向（p1 分量，单位向量）")
    fr.add("adjoint_dir2", gf.dir2, unit="", label="伴随梯度方向（p2 分量，单位向量）")
    fr.add("fd_dir1", gf.fd_dir1, unit="", label="中心差分方向（p1 分量）")
    fr.add("fd_dir2", gf.fd_dir2, unit="", label="中心差分方向（p2 分量）")
    fr.add("arrow_length", log_scaled_length(gf.grad_norm), unit="", label="箭头显示长度（梯度范数的对数压缩，0..1）")
    fr.add("grad_norm", gf.grad_norm, unit="", label="伴随梯度范数")
    err_placed = _add_checked(fr, "rel_error_log10", gf.log10_rel_err, unit="", label="log10(伴随 vs 中心差分 相对误差)")
    fr.add("rel_error", gf.rel_err, unit="", label="伴随 vs 中心差分 相对误差")

    fr.set_meta(
        n_arrows=int(gf.arrow_p1.size),
        n_grid_p1=int(gf.p1.size),
        n_grid_p2=int(gf.p2.size),
        max_rel_error=_finite_or_none(gf.max_rel_err()),
        median_rel_error=_finite_or_none(gf.median_rel_err()),
        min_rel_error=_finite_or_none(gf.min_rel_err()),
        grad_norm_max=_finite_or_none(float(np.max(gf.grad_norm)) if gf.grad_norm.size else None),
        grad_norm_min=_finite_or_none(float(np.min(gf.grad_norm)) if gf.grad_norm.size else None),
        objective_min=_finite_or_none(float(np.min(gf.objective)) if gf.objective.size else None),
        objective_max=_finite_or_none(float(np.max(gf.objective)) if gf.objective.size else None),
        resonance_p1=1.0,
        n_grid_solves=int(gf.n_grid_solves),
        n_arrow_solves=int(gf.n_arrow_solves),
        rhs_evals=int(gf.work.n_rhs),
        n_steps=int(n_steps),
        horizon=T,
        cubic=float(params["cubic"]),
        damping_center=float(params["damping"]),
        stiffness_center=float(params["stiffness"]),
        unit_roundoff=UNIT_ROUNDOFF,
        center_angle_deg=_finite_or_none(sd.angle_deg),
        center_rel_error=_finite_or_none(sd.rel_err),
        dominant_raw=int(np.argmax(np.abs(sd.grad_adjoint))) if sd.grad_adjoint.size else None,
        dominant_elasticity=int(sd.dominant()),
        elasticity_1=_finite_or_none(sd.elasticity[0]) if sd.elasticity.size else None,
        elasticity_2=_finite_or_none(sd.elasticity[1]) if sd.elasticity.size > 1 else None,
        overflow=False,
    )
    fr.set_strings(
        status="ok",
        arrow_note=(
            "箭头是**归一化方向**（单位向量），长度按梯度范数的 log10 压到 0..1（meta.grad_norm_max/min 给真实范围）。"
            "别把长度当灵敏度大小读：范数跨量级时它只剩形状信息。"
        ),
        error_note=(
            "rel_error_log10 是伴随梯度与中心差分梯度（eps=1e-6）的逐点相对误差；"
            f"实测中位数 {gf.median_rel_err():.2e}、最大 {gf.max_rel_err():.2e}。"
            "它必须 < 1e-6——这是'伴随梯度确实是 J_h 的精确梯度'的直接证据，"
            "而不是'看起来方向差不多'。"
        ),
        elasticity_note=(
            "跨参数比较灵敏度要用弹性 (dJ/dp_j)(p_j/J)：p1 与 p2 的量纲不同，"
            "直接比梯度分量等于拿刚度和阻尼比大小。"
            f"当前工作点上原始梯度 {np.array2string(np.asarray(sd.grad_adjoint), precision=3)} 的第 "
            f"{int(np.argmax(np.abs(sd.grad_adjoint)))} 个分量最大，而弹性 "
            f"{np.array2string(np.asarray(sd.elasticity), precision=3)} 的第 {int(sd.dominant())} 个分量最大——"
            "同一个点上'最灵敏的参数'有两个答案，差的就是量纲。"
        ),
        placeholder_note=(
            "objective 与 rel_error_log10 的非有限位置（如果有）已换成占位值 0.0 并给出 *_valid 掩码："
            "对数轴上的 0 读不出来，必须先看掩码。"
        ),
        params_note="p1/p2 的中心值由滑块给出（用于标注），网格覆盖 [0.6,1.6] x [0.03,0.4] 的固定窗口。",
        filled_note=(
            "objective / rel_error_log10 里若出现非有限值，已换占位值并给出 *_valid 掩码"
            f"（本帧占位标记：{'有' if (placed or err_placed) else '无'}）。"
        ),
        data_source="backend",
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 2：影响场
# --------------------------------------------------------------------------- #
_SCHEME_THETA = {"cn": 0.5, "btcs": 1.0, "ftcs": 0.0}


def _influence_field_frame(params: Mapping[str, Any], t: float) -> Frame:
    theta = _SCHEME_THETA[str(params["scheme"])]
    prob = HeatAdjointProblem(
        nx=41,
        length=1.0,
        alpha=float(params["alpha"]),
        horizon=float(params["horizon"]),
        n_steps=int(params["n_steps"]),
        centers=(0.35, 0.65),
        widths=(0.08, 0.08),
        obs_center=float(params["obs_center"]),
        obs_width=float(params["obs_width"]),
        theta=theta,
    )
    study = influence_width_law(prob, np.array([1.0, 0.7]), n_sample=61)
    frac = _fraction(t)
    tau_now = frac * float(params["horizon"])
    # tau 网格是"从 T 往回"的降序：找到当前 tau 最近的一行
    idx_now = int(np.argmin(np.abs(study.tau - tau_now)))
    blayer = heat_boundary_layer_ratio(
        nx=41,
        alpha=float(params["alpha"]),
        horizon=float(params["horizon"]),
        obs_center=float(params["obs_center"]),
        obs_width=float(params["obs_width"]),
        theta=theta,
    )

    fr = Frame(chapter="5.3", viz="influence_field", kind="frame")
    x = prob.x()
    fr.add("position", x, unit="", label="x")
    fr.add("tau", study.tau, unit="", label="反向时间 tau = T - t")
    fr.add("influence", study.mu, unit="", label="伴随影响场 mu(x, tau)")
    fr.add("width_measured", study.width_measured, unit="", label="实测宽度 sqrt(Var(mu))")
    fr.add("width_theory", study.width_continuous, unit="", label="连续律 sqrt(sigma_0^2 + 2*alpha*tau)")
    _add_checked(
        fr,
        "width_error",
        np.abs(study.width_measured / np.maximum(study.width_continuous, 1e-300) - 1.0),
        unit="",
        label="|实测宽度/连续律 - 1|",
    )
    fr.add("center_error", study.center_err, unit="", label="质心漂移 xbar(tau) - x_obs")
    fr.add("boundary_flag", study.boundary_contaminated.astype(float), unit="", label="1 = 该行已被 Dirichlet 壁污染")
    fr.add("influence_now", study.mu[idx_now], unit="", label="当前 tau 的影响场")
    _add_checked(
        fr,
        "semidiscrete_abs_error",
        np.abs(study.mu[idx_now] - _semidiscrete_now(prob, tau_now)),
        unit="",
        label="|当前影响场 - 半离散精确解|",
    )

    fr.set_meta(
        tau_current=float(tau_now),
        tau_max=float(params["horizon"]),
        tau_fraction=float(frac),
        width_current=_finite_or_none(study.width_measured[idx_now]),
        width_theory_current=_finite_or_none(study.width_continuous[idx_now]),
        width_ratio_at_max_tau=_finite_or_none(study.width_ratio_at_max_tau),
        width_ratio_clean_max=_finite_or_none(study.width_ratio_clean_max),
        var_slope_measured=_finite_or_none(study.var_slope_measured),
        var_slope_theory=_finite_or_none(study.var_slope_theory),
        var_slope_ratio=_finite_or_none(study.var_slope_measured / study.var_slope_theory),
        center_drift_max=_finite_or_none(study.center_drift_max),
        sigma0=_finite_or_none(math.sqrt(study.sigma0_sq)),
        obs_center=float(params["obs_center"]),
        semidiscrete_rel_error_max=_finite_or_none(study.semidiscrete_rel_err_max),
        semidiscrete_window_lo=_finite_or_none(study.semidiscrete_window[0]),
        semidiscrete_window_hi=_finite_or_none(study.semidiscrete_window[1]),
        boundary_rows=int(study.n_boundary_rows),
        n_tau_rows=int(study.tau.size),
        dx=study.dx,
        dt=study.dt,
        r=study.r,
        theta=theta,
        n_steps=int(params["n_steps"]),
        alpha=float(params["alpha"]),
        grad_source_1=_finite_or_none(study.grad[0]),
        grad_source_2=_finite_or_none(study.grad[1]),
        grad_fd_rel_error=_finite_or_none(study.grad_rel_err),
        n_fd_solves=int(2 * study.grad.size),
        boundary_layer_full=_finite_or_none(blayer["full"]),
        boundary_layer_interior=_finite_or_none(blayer["interior"]),
        boundary_layer_ratio=_finite_or_none(blayer["ratio"]),
        overflow=False,
    )
    fr.set_strings(
        status="ok",
        tau_note=(
            "时间轴是**反向时间比例** tau/T：tau=0 是终端观测时刻（影响场就是观测权重 w），"
            "tau=T 回到初始时刻。让 t 走一遍，看到的是敏感度分布从'一条窄峰'摊成'一片'。"
        ),
        width_note=(
            "宽度律 Var(tau) = sigma_0^2 + 2*alpha*tau 是**连续**律；实测宽度平方的斜率与 2*alpha 的比"
            f"写在 meta.var_slope_ratio（本帧 {study.var_slope_measured / study.var_slope_theory:.6f}）。"
            f"干净窗口内实测宽度与连续律的最大相对偏差 {study.width_ratio_clean_max:.3e}，"
            f"拖到 tau=T 才偏到 {abs(study.width_ratio_at_max_tau - 1.0):.3e}"
            f"（{study.width_measured[0]:.6f} vs {study.width_continuous[0]:.6f}）——"
            "所以'连续律在长 tau 处失效'的**量级**必须报出来，不能说成'完全不对'。"
        ),
        center_note=(
            f"'影响场在扩散'说的是**宽度**在长，不是峰值在跑：纯扩散下质心严格不动，"
            f"本帧实测最大漂移 {study.center_drift_max:.3e}（x_obs={float(params['obs_center']):.3f}）。"
        ),
        boundary_note=(
            f"boundary_flag=1 的行（本帧 {study.n_boundary_rows}/{study.tau.size} 行）表示影响场已经碰到 Dirichlet 壁："
            "那里宽度被壁吸收压小，连续律不再成立，图上必须截断而不是照画。"
        ),
        semidiscrete_note=(
            "semidiscrete_abs_error 是与**半离散精确解**（离散正弦模叠加，没有空间截断误差）的逐点偏差；"
            f"内部节点上的最大相对偏差 {study.semidiscrete_rel_err_max:.3e}，"
            f"对照窗口 tau ∈ [{study.semidiscrete_window[0]:.4f}, {study.semidiscrete_window[1]:.4f}]。"
        ),
        boundary_layer_note=(
            "端点那一层单独记一笔：Dirichlet 用的是恒等行，M 因此不对称，伴随要解 M^T。"
            f"含端点/去端点的相对偏差比 = {blayer['ratio']:.3f}（{blayer['full']:.3e} / {blayer['interior']:.3e}）——"
            "端点层是一阶的，内部是二阶的，两个数必须分开报。"
        ),
        grad_note=(
            "源项幅度 (p1, p2) 的伴随梯度与中心差分的相对误差 "
            f"{study.grad_rel_err:.3e}（同样的梯度用有限差分要 {2 * study.grad.size} 次正演）。"
        ),
        placeholder_note=(
            "influence 与 width_error 里的非有限位置已换占位值 0.0 并给 *_valid 掩码；"
            "对数值 0 读不出来，先看掩码。"
        ),
        data_source="backend",
    )
    return fr


def _semidiscrete_now(prob: HeatAdjointProblem, tau: float) -> np.ndarray:
    """当前 tau 的半离散精确影响场（帧里作为对照曲线）。"""
    from numcore.inverse.adjoint import dense_sine_coefficients, semidiscrete_adjoint_field

    x = prob.x()
    _, coeffs = dense_sine_coefficients(x, prob.observation(), float(prob.length))
    return semidiscrete_adjoint_field(
        x, coeffs, float(tau), nx=int(prob.nx), alpha=float(prob.alpha), length=float(prob.length)
    )


# --------------------------------------------------------------------------- #
# viz 3：梯度检验 V 形曲线 + 代价
# --------------------------------------------------------------------------- #
def _gradient_check_frame(params: Mapping[str, Any], t: float) -> Frame:
    del t  # mode="none"
    model, p, T, n_steps = _three_param(params)
    k_index = int(params["param_index"])
    eps_lo = float(params["eps_min"])
    eps_hi = float(params["eps_max"])
    if not eps_hi > eps_lo:
        eps_hi = eps_lo * 10.0
    eps_values = np.logspace(math.log10(eps_lo), math.log10(eps_hi), int(params["n_eps"]))
    check: GradientCheck = gradient_check_scan(
        model,
        p,
        T=T,
        n_steps=n_steps,
        method=str(params["method"]),
        eps_values=eps_values,
        schemes=("central", "forward"),
    )
    rows = cost_scaling_study(param_counts=(2, 8, 32), T=4.0, n_steps=96, method="rk4")

    fr = Frame(chapter="5.3", viz="gradient_check", kind="frame")
    fr.add("eps", check.eps, unit="", label="差分步长 eps（按 max(1,|p_j|) 缩放）")
    err_c = check.err["central"]
    err_f = check.err["forward"]
    _add_checked(fr, "error_central", _floored(err_c), unit="", label="中心差分：|g_fd - g_adj| 相对误差")
    _add_checked(fr, "error_forward", _floored(err_f), unit="", label="单侧差分：|g_fd - g_adj| 相对误差")
    eps = check.eps
    finite_c = err_c[np.isfinite(err_c)]
    anchor = float(np.max(finite_c)) if finite_c.size else 1e-30
    fr.add(
        "truncation_reference",
        _floored(eps * eps / max(eps_hi * eps_hi, 1e-300) * max(anchor, 1e-30), ERROR_FLOOR),
        unit="",
        label="参考斜率 eps^2（锚在实测最大值）",
    )
    fr.add("roundoff_reference", _floored(UNIT_ROUNDOFF / eps, ERROR_FLOOR), unit="", label="参考斜率 u/eps")
    fr.add("cost_n_params", np.array([r.n_params for r in rows], dtype=float), unit="", label="参数个数 k")
    fr.add("cost_adjoint_evals", np.array([r.adjoint_rhs + r.adjoint_jac for r in rows], dtype=float), unit="", label="伴随：残差+Jacobian 求值次数")
    fr.add("cost_fd_evals", np.array([r.fd_rhs for r in rows], dtype=float), unit="", label="中心差分：残差求值次数")
    fr.add("cost_adjoint_solves", np.array([r.adjoint_solves for r in rows], dtype=float), unit="", label="伴随：正演求解次数")
    fr.add("cost_fd_solves", np.array([r.fd_solves for r in rows], dtype=float), unit="", label="中心差分：正演求解次数")
    fr.add("cost_storage_floats", np.array([r.adjoint_floats for r in rows], dtype=float), unit="", label="伴随：存储浮点数（全存轨迹）")
    fr.add("cost_checkpoint_floats", np.array([r.checkpoint_floats for r in rows], dtype=float), unit="", label="分块检查点：存储浮点数")
    fr.add("cost_checkpoint_evals", np.array([r.checkpoint_rhs for r in rows], dtype=float), unit="", label="分块检查点：残差求值次数")

    ck = check
    opt_c = ck.best_eps("central")
    opt_f = ck.best_eps("forward")
    (lo_c, hi_c) = (ck.best_index.get("central", -1), ck.best_index.get("forward", -1))
    interior_c = 0 < lo_c < eps.size - 1
    interior_f = 0 < hi_c < eps.size - 1
    fr.set_meta(
        eps_min=float(eps[0]),
        eps_max=float(eps[-1]),
        n_eps=int(eps.size),
        param_index=k_index,
        k_scanned=_finite_or_none(p[k_index]),
        adjoint_grad_scanned=_finite_or_none(ck.adjoint_grad[k_index]),
        fd_grad_at_best=_finite_or_none(ck.fd_best["central"][k_index]) if ck.fd_best.get("central") is not None else None,
        eps_opt_central=_finite_or_none(opt_c) if interior_c else None,
        eps_opt_forward=_finite_or_none(opt_f) if interior_f else None,
        eps_opt_central_interior=bool(interior_c),
        eps_opt_forward_interior=bool(interior_f),
        error_best_central=_finite_or_none(ck.best_err("central")),
        error_best_forward=_finite_or_none(ck.best_err("forward")),
        slope_down_central=_finite_or_none(ck.slope_down.get("central")),
        slope_up_central=_finite_or_none(ck.slope_up.get("central")),
        slope_down_forward=_finite_or_none(ck.slope_down.get("forward")),
        slope_up_forward=_finite_or_none(ck.slope_up.get("forward")),
        eps_opt_central_over_u13=_finite_or_none(ck.optimum_ratio("central", "1/3")) if interior_c else None,
        eps_opt_forward_over_sqrtu=_finite_or_none(ck.optimum_ratio("forward", "1/2")) if interior_f else None,
        unit_roundoff=UNIT_ROUNDOFF,
        u_pow_third=_finite_or_none(UNIT_ROUNDOFF ** (1.0 / 3.0)),
        u_pow_half=_finite_or_none(UNIT_ROUNDOFF ** 0.5),
        n_steps=int(n_steps),
        horizon=T,
        fd_rhs_evals=int(check.work["central"].n_rhs),
        adjoint_fd_solves=int(check.work["central"].n_forward_solves),
        cost_ratio_k2=_finite_or_none(rows[0].fd_over_adjoint_evals()),
        cost_ratio_k8=_finite_or_none(rows[1].fd_over_adjoint_evals()),
        cost_ratio_k32=_finite_or_none(rows[2].fd_over_adjoint_evals()),
        cost_solves_k2=_finite_or_none(float(rows[0].fd_solves) / float(rows[0].adjoint_solves)),
        cost_solves_k8=_finite_or_none(float(rows[1].fd_solves) / float(rows[1].adjoint_solves)),
        cost_solves_k32=_finite_or_none(float(rows[2].fd_solves) / float(rows[2].adjoint_solves)),
        checkpoint_storage_reduction=_finite_or_none(rows[2].storage_reduction()),
        checkpoint_grad_rel_error=_finite_or_none(max(r.grad_rel_err_checkpoint for r in rows)),
        overflow=False,
    )
    fr.set_strings(
        status="ok",
        v_shape=(
            "V 形：左边下降段是**截断误差**（中心差分 ~ eps^2、单侧 ~ eps），"
            "右边上升段是**舍入误差**（~ u/eps）。谷底才是可用的 eps。"
        ),
        eps_generic_note=(
            "'eps 越小越准'是错的：实测最优点 中心差分 eps*="
            f"{opt_c if opt_c is not None else float('nan'):.3e}（相对 u^(1/3) 的比值 "
            f"{ck.optimum_ratio('central', '1/3') if interior_c else float('nan'):.3f}），"
            f"单侧差分 eps*={opt_f if opt_f is not None else float('nan'):.3e}"
            f"（相对 u^(1/2) 的比值 {ck.optimum_ratio('forward', '1/2') if interior_f else float('nan'):.3f}）。"
            "两个最优点的位置差三个数量级，正好对应 u^(1/3) 与 u^(1/2)。"
        ),
        slope_note=(
            "两段斜率是**分左右两段**拟合的（最优点左侧 / 右侧，各至少 2 点）："
            f"中心差分 {ck.slope_down.get('central')} / {ck.slope_up.get('central')}，"
            f"单侧差分 {ck.slope_down.get('forward')} / {ck.slope_up.get('forward')}。"
            "如果最优 eps 落在扫描端点上（meta 里 *_interior=false），这两个斜率与最优 eps 都不该当结论。"
        ),
        cost_note=(
            "伴随一次梯度只解一遍正向 + 一遍反向：meta.cost_solves_k32 给出 k=32 时的正演次数比"
            f"（实测 {float(rows[2].fd_solves) / float(rows[2].adjoint_solves):.1f} 倍）。"
            "但伴随**不免费**：要存整条正向轨迹（cost_storage_floats），"
            "或者用分块检查点把它换成重算（cost_checkpoint_floats 与 cost_checkpoint_evals）。"
        ),
        scanned_note=(
            f"被检验的参数是下标 {k_index}（工作点 p={tuple(CHECK_POINT)}，扫描的是 p[{k_index}]）。"
            "换参数下标最优点位置会移动——单一参数上的 V 形不能外推到整条梯度。"
        ),
        data_source="backend",
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：正反轨迹叠加播放
# --------------------------------------------------------------------------- #
def _playback_frame(params: Mapping[str, Any], t: float) -> Frame:
    model, p, T, n_steps = _model_common(params)
    trace = rk_forward(model, p, T=T, n_steps=n_steps, method="rk4", store="full")
    adj = discrete_adjoint(model, p, trace)
    cont = continuous_adjoint(model, p, trace)
    frac = _fraction(t)
    k_play = int(min(max(int(round(frac * n_steps)), 0), n_steps))
    t_play = float(trace.times[k_play])
    y_play = trace.states[k_play].copy()
    mu_play = adj.mu[k_play].copy()

    # 脉冲响应检验：在第 k_play 步扰动状态，看看 ΔJ 是否等于 mu^T δ
    delta = 1e-6 * max(1.0, float(np.max(np.abs(y_play))))
    resid = np.zeros(model.dim, dtype=float)
    fd_sens = np.zeros(model.dim, dtype=float)
    for i in range(model.dim):
        e = np.zeros(model.dim, dtype=float)
        e[i] = 1.0
        if k_play >= n_steps:
            # 播放头就在终端：扰动后的目标值直接由 g 给出（不必再积分，t0=T 会退化）
            jp = float(model.cost(y_play + delta * e))
            jm = float(model.cost(y_play - delta * e))
        else:
            jp = rk_forward_from(model, p, y_play + delta * e, t0=t_play, T=T, n_steps=n_steps - k_play).cost
            jm = rk_forward_from(model, p, y_play - delta * e, t0=t_play, T=T, n_steps=n_steps - k_play).cost
        fd_sens[i] = (jp - jm) / (2.0 * delta)
        resid[i] = abs(fd_sens[i] - mu_play[i]) / max(1.0, abs(fd_sens[i]), abs(mu_play[i]))

    fr = Frame(chapter="5.3", viz="forward_adjoint_playback", kind="frame")
    fr.add("time", trace.times, unit="", label="t")
    fr.add("forward_y1", trace.states[:, 0], unit="", label="正向 y1(t)")
    fr.add("forward_y2", trace.states[:, 1], unit="", label="正向 y2(t)")
    fr.add("adjoint_mu1", adj.mu[:, 0], unit="", label="伴随 mu1(t)（对 y1(t) 的敏感度）")
    fr.add("adjoint_mu2", adj.mu[:, 1], unit="", label="伴随 mu2(t)（对 y2(t) 的敏感度）")
    fr.add("forward_norm", _floored(trajectory_norms(trace.states)), unit="", label="|y(t)|")
    fr.add("adjoint_norm", trajectory_norms(adj.mu), unit="", label="|mu(t)|")
    fr.add("sensitivity_residual", resid, unit="", label="脉冲响应残差 |Delta J/delta - mu_i| 的相对值")
    fr.add("sensitivity_fd", fd_sens, unit="", label="有限差分实测的 dJ/dy_i(t_play)")
    fr.add("sensitivity_adjoint", mu_play, unit="", label="伴随给出的 dJ/dy_i(t_play)")
    _add_checked(
        fr,
        "continuous_gap",
        np.abs(adj.grad - cont.grad),
        unit="",
        label="|离散伴随梯度 - 连续伴随梯度|（逐参数）",
    )

    growth = adj.growth_ratio()
    fr.set_meta(
        play_fraction=float(frac),
        play_index=int(k_play),
        t_play=_finite_or_none(t_play),
        tau_play=_finite_or_none(T - t_play),
        n_steps=int(n_steps),
        horizon=T,
        norm_y_play=_finite_or_none(float(trajectory_norms(trace.states)[k_play])),
        norm_mu_play=_finite_or_none(float(trajectory_norms(adj.mu)[k_play])),
        adjoint_growth_ratio=_finite_or_none(growth),
        norm_mu_min=_finite_or_none(float(np.min(trajectory_norms(adj.mu)))),
        norm_mu_max=_finite_or_none(float(np.max(trajectory_norms(adj.mu)))),
        sensitivity_residual_max=_finite_or_none(float(np.max(resid))),
        grad_stiffness=_finite_or_none(adj.grad[0]),
        grad_damping=_finite_or_none(adj.grad[1]),
        continuous_gap_max=_finite_or_none(relative_error(adj.grad, cont.grad)),
        n_rhs_forward=int(trace.work.n_rhs),
        n_floats_stored=int(trace.work.n_floats_stored),
        overflow=False,
    )
    fr.set_strings(
        status="ok",
        playback_note=(
            "正向轨迹画到播放头 t，伴随轨迹从 T 倒着画回 t，两者在播放头相遇。"
            "正向那一半是**已经发生**的，伴随那一半是**还没发生**的敏感度回流。"
        ),
        dual_note=(
            "两条轨迹画在同一个相平面上只是**视觉**安排：y 是状态、mu 是余状态（对偶空间）。"
            "有意义的定量陈述是 sensitivity_residual——"
            f"本帧实测 Delta J/delta 与 mu_i 的最大相对残差 {float(np.max(resid)):.3e}，"
            "它把'mu(t) = dJ/dy(t)'这句话钉在了地上。"
        ),
        growth_note=(
            f"本帧 |mu(0)|/|mu(T)| = {growth:.4f}："
            + (
                "小于 1，说明**正向流在收缩**、伴随往回走也衰减（p2>0 的耗散区）。"
                if growth < 1.0
                else "大于 1，说明**正向流在扩张**、伴随往回走放大（负阻尼/泵浦区）。"
            )
            + "这个比值就是正向压缩率的倒数（转置不改变奇异值），它随 p2 过零换向，"
            "不是数值不稳定——加密步长它不变。"
        ),
        gap_note=(
            "continuous_gap 是离散伴随与连续伴随（同一张表、同一网格）的逐参数差："
            f"本帧相对差 {relative_error(adj.grad, cont.grad):.3e}。"
            "线性问题上两者会重合到舍入级，非线性（p3>0）才会裂开——这正是本章要分开报的那件事。"
        ),
        floor_note=f"|y| 曲线在 {ERROR_FLOOR:g} 处截平（半对数轴上的 0 画不出来）。",
        data_source="backend",
    )
    return fr
