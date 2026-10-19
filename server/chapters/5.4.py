"""5.4 章：不确定性量化与全局敏感性（帧装配层）。

四个可视化各自回答一个**会被讲错**的问题：

1. ``point_fill`` —— 点集填充对决：MC / Sobol' / 秩一格点 / 张量格点在同一张单位正方形里
   同时铺开，右下角是双对数的"误差-N"曲线与 :math:`1/\\sqrt N` 参考线。
   ``randomize`` 开关是本页的核心：**关掉它，Sobol' 与格点就没有误差棒**——
   不是"误差棒很小"，而是"这个量不存在"。帧里用 ``band_valid=0`` + 占位值 + 文字说明
   如实标注（PROTOCOLS §1.3.1 三件套），并且 ``meta.slope_*`` 是被**实测**出来的斜率
   （MC 理论 −1/2，QMC 实测值原样登记，不写理论值）。
2. ``output_hist`` —— 输出分布直方图随输入分布实时变化：直方图、参考密度、|密度差|
   曲线、Q-Q 对照，加上 bin 数 vs 误差的 U 形曲线。参考线是"精确卷积"还是"CLT 正态近似"
   必须写清楚（``strings.ref_kind``）——两者混着看会把 CLT 的偏差算到抽样误差头上。
3. ``sobol_bars`` —— 一阶 vs 总效应柱状图，拖 ``rho`` 看交互效应浮现。
   :math:`\\sum_iS_i\\le1\\le\\sum_iS_{Ti}` 是**判据**：帧里给出实测和与 bootstrap 区间，
   副轴画同一个函数在不同 :math:`N` 下的估计漂移（估计器有偏的直接证据）；
   ``rho`` 一离开 0，相关输入下的两种分解（条件方差口径 vs 乘积测度口径）同时画出来。
4. ``pce_spectrum`` —— 谱系数衰减 + 截断阶数 vs 误差（含过拟合区）+ 外推误差。
   训练误差与**留出集**误差是两条线：前者到机器零不代表代理模型好。

帧纪律（与 0.0/1.2/5.1 章一致）：``meta`` 只放 number/bool/None，字符串一律走 ``strings``；
没有有限值就写 ``None`` 并在 ``strings`` 里说明原因；对数轴上的 0 一律给地板并写明地板
（``error_floor``）；数组里**不许有** NaN/Inf——"这里没有有限值"用占位值 + 掩码 + 文字。
``server/`` 里不写数学：所有量都来自 :mod:`numcore.inverse.uq`（``mc`` / ``qmc`` /
``sobol_seq`` / ``sensitivity`` / ``pce``）。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from numcore.inverse.uq import mc, pce, qmc, sensitivity, sobol_seq
from numcore.utils import rng, seed_sequence
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 误差类曲线的对数轴地板（写进 ``strings``）：log10(1e-17) = -17，
#: 而 5.4 章所有"真误差"都在 1e-16 以上，只有"精确为 0"（多项式恰好被精确投影）需要占位。
ERROR_FLOOR = 1e-17
#: 双对数曲线的 N 采样点（幂次序列，便于读斜率）
N_CURVE = (64, 128, 256, 512, 1024, 2048, 4096)
#: 每个 N 上的随机化复制数（随机化 QMC 的误差棒就是从这里来的）
N_REPLICATES = 24

_MANIFEST_PITFALLS = (
    "双对数图上那条「误差随 N 下降的直线」不是收敛率的证明：斜率取决于被积函数的光滑性，"
    "把 integrand 从 smooth 切到 offset_box（边界不在 2^-m 网格上的间断），Sobol' 的优势会明显缩水；"
    "而 discontinuous 是反过来的陷阱：它的边界正好落在 2^-1 上，网**精确**积分它，"
    "误差恒为 0（帧里 error_valid 全 0、斜率报 None）——那不是「快得离谱」。"
    "更要紧的是右图那根误差棒——它只有在 randomize=True 时才存在；"
    "关掉随机化后 band_valid 全为 0，图上的带宽是占位值，不是「误差很小」。"
)

MANIFEST: dict[str, Any] = {
    "chapter": "5.4",
    "title": "不确定性量化与全局敏感性",
    "part": "V",
    "summary": (
        "蒙特卡洛的 1/sqrt(N) 与置信区间、拟蒙特卡洛与随机化误差估计、Sobol' 方差分解"
        "（含相关输入下的不唯一性）、多项式混沌代理模型：每个结论都配上它的实测数字与失效边界。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "point_fill",
            "title": "点集填充对决：MC / Sobol / 格点 / 张量格点",
            "description": (
                "单位正方形里同时铺开 MC、Sobol'、秩一格点三种点集（张量格点只出现在误差曲线里），"
                "时间轴是已铺开的点数 k；右下角是同一被积函数上四种方法的误差-N 双对数曲线，"
                "参考线是 sigma/sqrt(N)。开关 randomize 决定误差棒是否存在："
                "关掉它，确定性点集的“复制标准差”恒为 0。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "method",
                    "label": "主方法（曲线与误差棒）",
                    "type": "enum",
                    "options": ["mc", "sobol", "lattice", "grid"],
                    "default": "sobol",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "randomize",
                    "label": "随机化（scrambling / 随机移位）",
                    "type": "bool",
                    "default": True,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "integrand",
                    "label": "被积函数",
                    "type": "enum",
                    "options": ["smooth", "tensor", "kink", "discontinuous", "offset_box"],
                    "default": "smooth",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_max",
                    "label": "展开上限 N_max（时间轴右端）",
                    "type": "int",
                    "min": 256,
                    "max": 4096,
                    "step": 256,
                    "default": 2048,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 2048.0, "dt": 64.0, "label": "已铺开的点数 k"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "点集（单位正方形；蓝=MC / 橙=Sobol / 绿=格点）",
                    "ylabel": "x_2",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "log10 误差 vs N 与 1/sqrt(N) 参考线",
                    "ylabel": "log10|I_hat - I| 或 log10 RMSE",
                },
            ],
            "pitfalls": (
                "左图「点看起来更均匀」不等于积分更准：均匀性要用偏差 D*_N 量，而收敛率要用误差-N "
                "斜率量，两者在本页都有数可读。右下角的误差棒只在 randomize=True 时存在——"
                "关掉随机化后确定性点集的“复制”逐位相同，标准差恒等于 0，"
                "而真实误差仍可能是 1e-3 量级；把 0 读成「误差很小」是本页最容易犯的错。"
                "另外参考线 sigma/sqrt(N) 只对 MC 是理论曲线，对 QMC 它只是一条标度参照。"
            ),
            "theory": [
                "MC 的均方根误差 = sigma/sqrt(N)（sigma^2 = Var f），双对数斜率 -1/2",
                "Koksma-Hlawka：|I_hat - I| <= V(f) D*_N；Sobol' 的 D*_N = O((log N)^d/N)",
                "随机化 QMC：数字移位/线性置乱保持网性质，R 个复制的样本标准差给出可用的误差估计",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "output_hist",
            "title": "输出分布直方图：输入分布与模型一起拖",
            "description": (
                "从输入分布（均匀/正态/Beta/指数）抽 n 个样本，过前向模型得到输出样本，"
                "画直方图密度并与参考密度逐点比较；副轴是 |密度差| 与 Q-Q 对照。"
                "参考密度能写出闭式时用精确卷积，否则用同均值同方差的正态（CLT）近似——"
                "是哪一种由 strings.ref_kind 说明。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "dist",
                    "label": "输入分布",
                    "type": "enum",
                    "options": ["uniform", "normal", "beta", "exponential"],
                    "default": "uniform",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "model",
                    "label": "前向模型",
                    "type": "enum",
                    "options": ["linear", "quadratic", "stiff"],
                    "default": "quadratic",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "spread",
                    "label": "输入标准差（不确定性大小）",
                    "type": "float",
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "default": 1.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n",
                    "label": "样本数 N",
                    "type": "int",
                    "min": 2000,
                    "max": 100000,
                    "step": 2000,
                    "default": 20000,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_bins",
                    "label": "直方图 bin 数 B",
                    "type": "int",
                    "min": 10,
                    "max": 80,
                    "step": 2,
                    "default": 40,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": True,
                    "title": "输出密度（直方图 vs 参考密度）",
                    "ylabel": "p_Y(y)",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "log10|密度差|（逐 bin）",
                    "ylabel": "log10|p_hat - p_ref|",
                },
            ],
            "pitfalls": (
                "直方图的形状由 bin 数决定：bin 太少看不出结构（偏差 O(h^2)），"
                "bin 太多每个 bin 只剩几个样本（方差 O(1/sqrt(Nh))）——副轴那条 U 形曲线就是"
                "这个权衡的实测。第二条更隐蔽：参考密度是正态近似时，图上「密度差」里同时包含"
                "抽样误差与 CLT 自己的偏差，把整条差曲线都说成蒙特卡洛误差是错的；"
                "右偏模型（quadratic/stiff）上 KS 距离实测是正态模型的好几倍。"
            ),
            "theory": [
                "核/直方图密度估计的偏差-方差权衡：MISE ~ h^4 + 1/(Nh)，最优 h ~ N^(-1/5)",
                "中心极限定理：Y 的标准化分布在 n 大时趋近正态，但收敛速度由偏度与峰度决定",
                "全方差公式：Var(Y) = E[Var(Y|X)] + Var(E[Y|X])（第 3 页的分解就从这里来）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "sobol_bars",
            "title": "Sobol 指数柱状图：一阶 vs 总效应，拖 rho 看相关效应",
            "description": (
                "pick-freeze（Saltelli/Jansen）估计一阶与总效应指数并给 bootstrap 区间；"
                "副轴是同一函数在 N = 2^6..2^12 上的估计漂移（估计量本身有偏的证据）。"
                "rho != 0 且模型支持相关输入时，同时给出条件方差口径与乘积测度口径两套分解。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "model",
                    "label": "模型",
                    "type": "enum",
                    "options": ["ishigami", "g_function", "bilinear", "quadratic_corr"],
                    "default": "g_function",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n",
                    "label": "pick-freeze 基样本数 N",
                    "type": "int",
                    "min": 256,
                    "max": 4096,
                    "step": 256,
                    "default": 1024,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "rho",
                    "label": "输入相关系数 rho",
                    "type": "float",
                    "min": -0.8,
                    "max": 0.8,
                    "step": 0.05,
                    "default": 0.0,
                    "critical": [0.0],
                    "criticalNote": (
                        "rho = 0 是方差分解唯一性的边界：输入独立时 sum S_i <= 1 <= sum S_Ti 且分解唯一；"
                        "rho != 0 时“一阶指数”至少有两种口径（条件方差 / 乘积测度），"
                        "sum S_i 可以大于 1（实测 rho=0.5 的双线性模型 sum = 1.49）。"
                    ),
                    "beyond": "magnitude",
                },
                {
                    "name": "estimator",
                    "label": "估计器",
                    "type": "enum",
                    "options": ["saltelli", "jansen"],
                    "default": "saltelli",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": True,
                    "title": "Sobol 指数（一阶 S_i 与总效应 S_Ti，含 bootstrap 区间）",
                    "ylabel": "S_i / S_Ti",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "估计器漂移：sum S_i(N) 与 sum S_Ti(N)",
                    "ylabel": "指数之和",
                },
            ],
            "pitfalls": (
                "sum S_i <= 1 <= sum S_Ti 是判据而不是恒等式：有限样本下估计量有偏，"
                "副轴那条漂移曲线就是偏差本身（Ishigami 在 N=64 与 N=4096 上的 sum S_i 相差 0.29）。"
                "更危险的是相关输入：rho != 0 时方差分解不再唯一，条件方差口径的 sum S_i "
                "可以到 1.5 甚至更高，而这并不违反任何定理——违反的是「输入独立」这个前提。"
                "把bar的高度当成「重要性」之前，先确认你要的是哪一种口径。"
            ),
            "theory": [
                "Sobol-Hoeffding 分解：独立输入下 Var(Y) = sum_alpha V_alpha，S_i = V_i/V <= 1 <= S_Ti",
                "pick-freeze 估计量（Saltelli 2010 / Jansen 1999）的偏差是 O(1/N)，不是无偏的",
                "相关输入下 Var(E[Y|X_i])/V 之和可以大于 1：全方差公式只对单个变量成立",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "pce_spectrum",
            "title": "多项式混沌：谱系数衰减、截断阶数 vs 误差、外推",
            "description": (
                "左：逐阶系数的 RMS（对数轴）与指数衰减参考线；右：截断阶数 p 从 1 到 12 时"
                "训练误差与留出集误差两条曲线（含欠定过拟合区），以及外推倍率扫描曲线。"
                "fit=projection 时用 Gauss 求积，fit=regression 时用最小二乘（样本不够会过拟合）。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "target",
                    "label": "目标函数",
                    "type": "enum",
                    "options": ["poly4", "exp", "runge", "abs"],
                    "default": "exp",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "basis",
                    "label": "正交基",
                    "type": "enum",
                    "options": ["legendre", "hermite"],
                    "default": "legendre",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "fit",
                    "label": "系数求解方式",
                    "type": "enum",
                    "options": ["regression", "projection"],
                    "default": "regression",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "order",
                    "label": "截断阶数 p",
                    "type": "int",
                    "min": 1,
                    "max": 12,
                    "step": 1,
                    "default": 6,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_train",
                    "label": "训练样本数",
                    "type": "int",
                    "min": 8,
                    "max": 256,
                    "step": 8,
                    "default": 64,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "extrap",
                    "label": "外推倍率（1 = 不外推）",
                    "type": "float",
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "default": 1.0,
                    "critical": [1.0],
                    "criticalNote": (
                        "1.0 是训练区间的边界：小于等于它时报告的是同一测度上的留出误差；"
                        "大于它就是在训练支撑集之外评估，多项式混沌对此**没有任何保证**"
                        "（实测 exp 外推 2 倍时误差放大 709 倍，Runge 放大 1.5e4 倍）。"
                    ),
                    "beyond": "above",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": True,
                    "title": "log10 逐阶系数 RMS 与衰减参考线",
                    "ylabel": "log10 rms_|alpha|=k |c_alpha|",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "log10 误差 vs 截断阶数（训练 / 留出）",
                    "ylabel": "log10 相对 L2 误差",
                },
            ],
            "pitfalls": (
                "训练误差到 1e-16 什么也不能说明：n_train=8、p=10 时训练误差 4e-16 而留出误差 0.45"
                "（相差 1e15 倍），这就是过拟合区的样子——它只出现在「基规模超过样本数」之后，"
                "而那一区在图上并不显眼。第二条：外推。多项式混沌在训练区间之外没有收敛保证，"
                "exp 外推 2 倍误差放大 709 倍、Runge 放大 1.5e4 倍；"
                "只有像 poly4 那样被精确表示的函数才能外推，而那不是代理模型的功劳。"
            ),
            "theory": [
                "gPC 投影：c_alpha = E[f phi_alpha]，正交归一基下 E[f]=c_0、Var[f]=sum_{alpha>0} c_alpha^2",
                "解析函数在 Legendre 基下的系数按几何速率衰减；有折点（|x|）时只按 p^(-3/2) 衰减",
                "Gauss 求积的精确性：n_q = p+1 个节点对次数 <= p 的多项式给出精确投影系数",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "point_fill":
        return _point_fill(params, t)
    if viz_id == "output_hist":
        return _output_hist(params, t)
    if viz_id == "sobol_bars":
        return _sobol_bars(params, t)
    if viz_id == "pce_spectrum":
        return _pce_spectrum(params, t)
    raise KeyError(f"章节 5.4 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _finite_or_none(value: Any) -> float | None:
    """把 NaN/±Inf 换成 ``None``：NLF 头用 ``allow_nan=False`` 编码，塞进去整帧都会失败。"""
    if value is None:
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _log10_floor(values: Any, floor: float = ERROR_FLOOR) -> tuple[np.ndarray, np.ndarray]:
    """对数轴曲线：返回 ``(log10(地板截断后的值), 有效掩码)``。

    0 或非有限值一律写成地板（否则 log10 出 -inf，数组契约当场失败），
    掩码标出"这个点不是真值"。调用方必须把掩码一起写进帧。
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    valid = np.isfinite(v) & (v > 0.0)
    safe = np.where(valid, v, floor)
    safe = np.maximum(safe, floor)
    return np.log10(safe), valid.astype(np.float64)


def _step_of(t: float, n_max: int, *, floor: int = 1) -> int:
    """把时间轴四舍五入到整数点数并夹到 ``[floor, n_max]``。"""
    k = int(math.floor(float(t) + 0.5))
    return max(int(floor), min(k, int(n_max)))


def _slope_or_none(n: Any, err: Any, *, head: int | None = None) -> float | None:
    """双对数斜率；**严格为正**的点不足 3 个时返回 ``None``（不在地板上编斜率）。

    这一条是被 Sobol' 网上的间断函数教出来的：那种点集对 :math:`1[x<0.5]` 型被积函数
    给出**精确为 0** 的误差，误差曲线整条贴在地板上，而对数-对数拟合照样能吐出
    -0.495 这种"很像 -1/2"的数。斜率是实测出来的，就不能在地板上实测。
    """
    v = np.asarray(err, dtype=np.float64).ravel()
    if int(np.count_nonzero(np.isfinite(v) & (v > 0.0))) < 3:
        return None
    try:
        return float(qmc.loglog_slope(n, err, min_points=3, head=head))
    except (ValueError, ZeroDivisionError):
        return None


# --------------------------------------------------------------------------- #
# viz 1：点集填充对决
# --------------------------------------------------------------------------- #
#: 误差曲线按 (integrand, randomize) 缓存：拖动时间轴与切方法时不该重算 24x7 次模型求值
_CURVE_CACHE: dict[tuple[str, bool], dict[str, Any]] = {}
#: 点集缓存：按 (method, randomize, integrand) 存 n_max=4096 的点（拖动时间轴时切片）
_POINT_CACHE: dict[tuple[str, bool, str], np.ndarray] = {}


def _curve_for(integrand: str, randomize: bool) -> dict[str, Any]:
    """四种方法在同一被积函数上的 RMSE-N 曲线（带缓存；这是本页最重的一块计算）。"""
    key = (str(integrand), bool(randomize))
    hit = _CURVE_CACHE.get(key)
    if hit is not None:
        return hit
    integ = qmc.make_integrand(integrand, 2)
    ns = np.asarray(N_CURVE, dtype=np.int64)
    curves: dict[str, np.ndarray] = {}
    bands: dict[str, np.ndarray] = {}
    for method in ("mc", "sobol", "lattice", "grid"):
        use_rand = bool(randomize) and method != "grid"
        rmse = np.empty(ns.size, dtype=np.float64)
        se = np.empty(ns.size, dtype=np.float64)
        for i, n in enumerate(ns):
            if method == "grid":
                side = max(2, int(round(math.sqrt(float(n)))))
                pts = sobol_seq.tensor_grid(side, 2)
                vals = integ.func(pts)
                rmse[i] = abs(float(np.mean(vals)) - integ.exact)
                se[i] = 0.0
                continue
            rep = qmc.qmc_error_report(
                integ.func,
                int(n),
                2,
                true_value=integ.exact,
                method=method,
                replicates=N_REPLICATES,
                seed=20261018,
                randomize=use_rand,
            )
            rmse[i] = rep.rmse_single if use_rand else rep.abs_error
            se[i] = rep.stderr_of_mean if use_rand else 0.0
        curves[method] = rmse
        bands[method] = se
    # sigma/sqrt(N) 参考线：sigma 用 2e4 个 MC 样本实测（不写"理论 sigma"）
    g = rng(4242)
    probe = g.random((20000, 2))
    sigma = float(np.std(integ.func(probe), ddof=1))
    ref = sigma / np.sqrt(ns.astype(np.float64))
    out = {
        "integrand": integ,
        "ns": ns,
        "curves": curves,
        "bands": bands,
        "sigma": sigma,
        "ref": ref,
        "randomized": {"mc": True, "sobol": bool(randomize), "lattice": bool(randomize), "grid": False},
    }
    if len(_CURVE_CACHE) < 16:  # 有界缓存：拖动时不重算，但也不把服务进程吃干
        _CURVE_CACHE[key] = out
    return out


def _points_cached(method: str, randomize: bool, integrand: str, n: int) -> np.ndarray:
    """取（必要时生成）某方法的 ``n`` 个点；Sobol'/格点按 ``randomize`` 决定是否随机化。"""
    key = (str(method), bool(randomize), str(integrand))
    pts = _POINT_CACHE.get(key)
    if pts is None or pts.shape[0] < n:
        use_rand = bool(randomize) and method != "grid"
        pts = sobol_seq.point_set(method, 4096, 2, seed=20261018, randomize=use_rand)
        if len(_POINT_CACHE) < 16:
            _POINT_CACHE[key] = pts
    return pts[:n]


def _point_fill(params: Mapping[str, Any], t: float) -> Frame:
    method = str(params["method"])
    randomize = bool(params["randomize"])
    integrand = str(params["integrand"])
    n_max = int(params["n_max"])
    data = _curve_for(integrand, randomize)
    ns = data["ns"]
    integ = data["integrand"]
    k = _step_of(t, n_max)

    curves = data["curves"]
    bands = data["bands"]
    current = curves[method]
    band = bands[method]
    rand_ok = method != "grid" and (randomize or method == "mc")
    err_log, err_valid = _log10_floor(current)
    band_lo = np.maximum(current - band, ERROR_FLOOR)
    band_hi = np.maximum(current + band, ERROR_FLOOR)

    fr = Frame(chapter="5.4", viz="point_fill", kind="frame")
    for name, m in (("mc", "mc"), ("sobol", "sobol"), ("lattice", "lattice")):
        pts = _points_cached(m, randomize, integrand, k)
        fr.add(f"{name}_x", pts[:, 0], unit="1", label=f"{name} 点的 x_1")
        fr.add(f"{name}_y", pts[:, 1], unit="1", label=f"{name} 点的 x_2")
    fr.add("n_axis", ns.astype(np.float64), unit="1", label="样本数 N")
    fr.add("error", err_log, unit="1", label=f"log10({method} 的误差)")
    fr.add("error_valid", err_valid, unit="1", label="error 有效掩码（0 = 触地板/未定义）")
    fr.add("error_mc", _log10_floor(curves["mc"])[0], unit="1", label="log10(MC 误差)")
    fr.add("error_sobol", _log10_floor(curves["sobol"])[0], unit="1", label="log10(Sobol 误差)")
    fr.add("error_lattice", _log10_floor(curves["lattice"])[0], unit="1", label="log10(格点误差)")
    fr.add("error_grid", _log10_floor(curves["grid"])[0], unit="1", label="log10(张量格点误差)")
    fr.add("error_ref", _log10_floor(data["ref"])[0], unit="1", label="log10(sigma/sqrt(N)) 参考线")
    fr.add("band_lo", _log10_floor(band_lo)[0], unit="1", label="log10(误差带下沿)")
    fr.add("band_hi", _log10_floor(band_hi)[0], unit="1", label="log10(误差带上沿)")
    fr.add(
        "band_valid",
        np.full(ns.size, 1.0 if rand_ok else 0.0),
        unit="1",
        label="误差带有效掩码（0 = 确定性点集，没有误差估计）",
    )

    slope_mc = _slope_or_none(ns, curves["mc"], head=6)
    slope_qmc = _slope_or_none(ns, curves["sobol"], head=6)
    slope_lat = _slope_or_none(ns, curves["lattice"], head=6)
    slope_grid = _slope_or_none(ns, curves["grid"], head=6)
    slope_cur = {"mc": slope_mc, "sobol": slope_qmc, "lattice": slope_lat, "grid": slope_grid}[method]
    ratio = float(current[-1] / curves["mc"][-1]) if curves["mc"][-1] > 0 else float("nan")

    fr.set_meta(
        n_shown=k,
        n_max=n_max,
        n_curve=int(ns.size),
        replicates=N_REPLICATES,
        band_available=bool(rand_ok),
        slope_current=_finite_or_none(slope_cur),
        slope_mc=_finite_or_none(slope_mc),
        slope_sobol=_finite_or_none(slope_qmc),
        slope_lattice=_finite_or_none(slope_lat),
        slope_grid=_finite_or_none(slope_grid),
        slope_reference=-0.5,
        error_last=_finite_or_none(float(current[-1])),
        error_mc_last=_finite_or_none(float(curves["mc"][-1])),
        error_ratio_vs_mc=_finite_or_none(ratio),
        sigma_measured=data["sigma"],
        # 规则 1/4：画出来的东西越界才置真。主坐标轴是单位正方形（不自适应），
        # 三个点集按构造都在 [0,1) 内；QMC 的确定性点集也不会越界。
        overflow=bool(
            not np.all(np.isfinite(curves["mc"]))
            or (method != "grid" and not np.all(np.isfinite(curves[method])))
        ),
    )
    fr.set_strings(
        method=method,
        method_name={
            "mc": "蒙特卡洛（i.i.d. 均匀）",
            "sobol": "Sobol' 数字网",
            "lattice": "秩一格点（Korobov）",
            "grid": "张量格点（每维 sqrt(N) 个点）",
        }[method],
        integrand=integrand,
        integrand_label=integ.label,
        smoothness=integ.smoothness,
        note=(
            f"{integ.label}：解析积分 = {integ.exact:.6f}；实测斜率 "
            f"MC {_fmt(slope_mc)} / Sobol {_fmt(slope_qmc)} / 格点 {_fmt(slope_lat)} / "
            f"张量格点 {_fmt(slope_grid)}（参考线 -0.500）；"
            + (
                f"N={int(ns[-1])} 时 {method} 的误差是 MC 的 {ratio:.3g} 倍"
                if ratio > 0.0
                else f"N={int(ns[-1])} 时 {method} 的误差精确为 0（点集与间断/边界对齐，见 error_valid）"
            )
        ),
        band_note=(
            "误差带 = R=24 个随机化复制的均值标准误（对数轴上的上下沿）；"
            "randomize=False 时确定性点集的 24 个复制逐位相同、标准差恒为 0，"
            "band_valid 全为 0、band_lo/band_hi 是占位值——它表示「这个量不存在」，"
            "不是「误差很小」。"
        ),
        grid_note=(
            "张量格点只能取 N = k^2 个点（这里是 k = round(sqrt(N))），"
            "所以它的曲线横坐标是名义 N；它的误差在 d=2 上很好，"
            "但点数随维数按 k^d 增长——d=10 时同样 4096 个点每维只有 2 个取值。"
        ),
        nonfinite_policy=(
            "帧里没有 NaN/Inf：对数轴上的 0 与未定义的量写成地板 "
            f"{ERROR_FLOOR:.0e} 并配 *_valid 掩码，读图前先看掩码。"
        ),
        error_floor=f"{ERROR_FLOOR:.0e}",
    )
    return fr


def _fmt(value: float | None) -> str:
    """把可能为 ``None`` 的斜率格式化成 ``strings`` 里能读的样子（``None`` 一律写"不可测"）。"""
    return "不可测" if value is None else f"{value:+.3f}"


# --------------------------------------------------------------------------- #
# viz 2：输出分布直方图
# --------------------------------------------------------------------------- #
_HIST_CACHE: dict[tuple[str, str, float, int, int], dict[str, Any]] = {}


def _hist_data(dist: str, model: str, spread: float, n: int, n_bins: int) -> dict[str, Any]:
    """抽样 + 直方图 + 参考密度 + Q-Q + bin 权衡曲线（带缓存：参数不变时不重算）。"""
    key = (str(dist), str(model), float(spread), int(n), int(n_bins))
    hit = _HIST_CACHE.get(key)
    if hit is not None:
        return hit
    d2 = mc.input_distribution(dist)
    x = np.empty((int(n), 2), dtype=np.float64)
    seeds = seed_sequence(20261018, 2)
    for j in range(2):
        x[:, j] = d2.samples(int(n), spread=float(spread), seed=int(seeds[j]))
    y = mc.forward_model(model, x)
    lo = float(np.min(y)) - 0.05 * (float(np.max(y)) - float(np.min(y)) + 1e-12)
    hi = float(np.max(y)) + 0.05 * (float(np.max(y)) - float(np.min(y)) + 1e-12)
    centers, dens, counts = mc.histogram_density(y, n_bins=int(n_bins), lo=lo, hi=hi)
    moments = mc.sample_moments(y)

    exact = mc.exact_sum_density(dist, float(spread), centers) if model == "linear" else None
    if exact is not None:
        ref = exact
        ref_kind = "exact"
    else:
        s = moments["std"]
        z = (centers - moments["mean"]) / s if s > 0 else np.zeros_like(centers)
        ref = np.exp(-0.5 * z * z) / (s * math.sqrt(2.0 * math.pi)) if s > 0 else np.zeros_like(centers)
        ref_kind = "clt"

    # Q-Q：经验分位数（等概率网格） vs 参考分位数
    qs = (np.arange(201, dtype=np.float64) + 0.5) / 201.0
    qq_emp = np.quantile(y, qs)
    if ref_kind == "exact":
        qq_ref = np.quantile(np.asarray(y, dtype=np.float64), qs)  # 占位：下面会被精确分布覆盖
        # 精确分布的分位数用二分求根（单调 CDF）
        grid = np.linspace(lo, hi, 4001)
        dens_grid = mc.exact_sum_density(dist, float(spread), grid)
        cdf = np.cumsum(dens_grid) * (grid[1] - grid[0]) if dens_grid is not None else None
        if cdf is not None and cdf[-1] > 0:
            cdf = cdf / cdf[-1]
            qq_ref = np.interp(qs, cdf, grid)
    else:
        s = moments["std"]
        qq_ref = moments["mean"] + s * mc.norm_ppf(qs)

    # bin 数 vs 误差（U 形）：用同一分布的独立复制，参考线用同均值同方差正态
    def _ref_fn(c: np.ndarray) -> np.ndarray:
        s = moments["std"]
        z = (c - moments["mean"]) / s if s > 0 else np.zeros_like(c)
        return np.exp(-0.5 * z * z) / (s * math.sqrt(2.0 * math.pi)) if s > 0 else np.zeros_like(c)

    def _sampler(sd: Any) -> np.ndarray:
        ss = seed_sequence(int(sd), 2)
        xx = np.empty((20000, 2), dtype=np.float64)
        for j in range(2):
            xx[:, j] = d2.samples(20000, spread=float(spread), seed=int(ss[j]))
        return mc.forward_model(model, xx)

    bins_list = [8, 12, 16, 24, 32, 48, 64, 96]
    try:
        bins_curve, bins_err = mc.bin_width_tradeoff(_ref_fn, _sampler, n=int(n), bins_list=bins_list, replicates=4, seed=99)
    except (ValueError, ZeroDivisionError):
        bins_curve, bins_err = np.asarray(bins_list, dtype=np.int64), np.zeros(len(bins_list))
    out = {
        "dist": d2,
        "moments": moments,
        "centers": centers,
        "density": dens,
        "counts": counts,
        "ref": np.asarray(ref, dtype=np.float64),
        "ref_kind": ref_kind,
        "qq_emp": np.asarray(qq_emp, dtype=np.float64),
        "qq_ref": np.asarray(qq_ref, dtype=np.float64),
        "bins_curve": np.asarray(bins_curve, dtype=np.float64),
        "bins_err": np.asarray(bins_err, dtype=np.float64),
        "ks_normal": mc.ks_to_normal(y),
        "ks_ref": _ks_to_ref(y, dist, float(spread), moments, ref_kind == "exact"),
        "within_1p96": float(np.mean(np.abs(y - moments["mean"]) <= 1.959963984540054 * moments["std"])),
    }
    if len(_HIST_CACHE) < 32:
        _HIST_CACHE[key] = out
    return out


def _ks_to_ref(y: np.ndarray, dist: str, spread: float, moments: Mapping[str, float], exact: bool) -> float:
    """经验 CDF 与参考 CDF 的 KS 距离。

    参考 CDF 在一张**细网格**（4001 点）上由密度累加得到，而不是在 40 个 bin 中心上——
    后者会让"参考 CDF"自己带上 O(h) 的离散误差，于是连"参考恰好就是真密度"的情形
    （正态输入 + 线性模型）都会报出 0.05 的 KS 距离，比正态近似的 0.005 还大。
    本模块第一版正是如此，把两个数的含义直接对调了。
    """
    ys = np.sort(np.asarray(y, dtype=np.float64))
    s = float(moments["std"])
    lo = min(float(ys[0]), float(moments["mean"]) - 8.0 * s)
    hi = max(float(ys[-1]), float(moments["mean"]) + 8.0 * s)
    grid = np.linspace(lo, hi, 4001)
    if exact:
        dens = mc.exact_sum_density(dist, float(spread), grid)
        if dens is None:
            exact = False
    if not exact:
        z = (grid - float(moments["mean"])) / s if s > 0 else np.zeros_like(grid)
        dens = np.exp(-0.5 * z * z) / (s * math.sqrt(2.0 * math.pi)) if s > 0 else np.zeros_like(grid)
    dens = np.asarray(dens, dtype=np.float64)
    cdf = np.cumsum(dens) * (grid[1] - grid[0])
    if cdf[-1] <= 0:
        return float("nan")
    cdf = cdf / cdf[-1]
    ref_cdf = np.interp(ys, grid, cdf, left=0.0, right=1.0)
    emp = np.arange(1, ys.size + 1, dtype=np.float64) / ys.size
    return float(max(np.max(np.abs(emp - ref_cdf)), np.max(np.abs(ref_cdf - (emp - 1.0 / ys.size)))))


def _output_hist(params: Mapping[str, Any], t: float) -> Frame:
    dist = str(params["dist"])
    model = str(params["model"])
    spread = float(params["spread"])
    n = int(params["n"])
    n_bins = int(params["n_bins"])
    data = _hist_data(dist, model, spread, n, n_bins)
    m = data["moments"]
    dens = np.asarray(data["density"], dtype=np.float64)
    ref = np.asarray(data["ref"], dtype=np.float64)
    diff = np.abs(dens - ref)

    fr = Frame(chapter="5.4", viz="output_hist", kind="frame")
    fr.add("bin_center", np.asarray(data["centers"], dtype=np.float64), unit="1", label="输出 y")
    fr.add("density", dens, unit="1", label="直方图密度")
    fr.add("density_ref", ref, unit="1", label="参考密度")
    fr.add("error", _log10_floor(diff)[0], unit="1", label="log10|密度 - 参考密度|")
    fr.add("error_valid", _log10_floor(diff)[1], unit="1", label="error 有效掩码")
    fr.add("qq_ref", np.asarray(data["qq_ref"], dtype=np.float64), unit="1", label="参考分位数")
    fr.add("qq_emp", np.asarray(data["qq_emp"], dtype=np.float64), unit="1", label="经验分位数")
    fr.add("bins_curve", np.asarray(data["bins_curve"], dtype=np.float64), unit="1", label="bin 数 B")
    fr.add("bins_error", _log10_floor(data["bins_err"])[0], unit="1", label="log10(密度 RMS 误差)")
    fr.add("counts", np.asarray(data["counts"], dtype=np.float64), unit="1", label="各 bin 计数")
    fr.set_meta(
        n=int(n),
        n_bins=int(n_bins),
        bin_width=_finite_or_none(float(data["centers"][1] - data["centers"][0])) if data["centers"].size > 1 else None,
        mean=m["mean"],
        std=m["std"],
        skewness=m["skewness"],
        excess_kurtosis=m["excess_kurtosis"],
        ks_to_normal=data["ks_normal"],
        ks_to_ref=data["ks_ref"],
        within_1p96_sigma=data["within_1p96"],
        ref_is_exact=bool(data["ref_kind"] == "exact"),
        spread=spread,
        overflow=bool(not np.all(np.isfinite(dens)) or not np.all(np.isfinite(ref))),
    )
    fr.set_strings(
        dist=dist,
        dist_label=data["dist"].label,
        model=model,
        model_label=mc.model_label(model),
        ref_kind=(
            "精确卷积（独立同分布之和的闭式密度）"
            if data["ref_kind"] == "exact"
            else "CLT 近似，不是真值（同均值同方差正态）"
        ),
        note=(
            f"输出：均值 {m['mean']:+.4f}、标准差 {m['std']:.4f}、偏度 {m['skewness']:+.3f}、"
            f"超额峰度 {m['excess_kurtosis']:+.3f}；KS(经验 vs 参考) = {data['ks_ref']:.4f}，"
            f"KS(经验 vs 正态) = {data['ks_normal']:.4f}；落在 mean+-1.96s 内的比例 "
            f"{data['within_1p96']:.4f}（正态参考值 0.95）"
        ),
        clt_note=(
            "参考线是同均值同方差的正态密度时，|密度差| 里同时含**抽样误差**与 **CLT 偏差**："
            "偏度越大，KS 距离越大，而增大 N 只能压掉前者。"
            if data["ref_kind"] != "exact"
            else "参考线是精确卷积密度：此时的 |密度差| 才是纯粹的密度估计误差。"
        ),
        bins_note=(
            "副轴上的 bins_error 是同一分布、不同 bin 数下的密度 RMS 误差（对正态参考线），"
            "它有一个 U 形最优：bin 少 → 偏差 O(h^2)，bin 多 → 方差 O(1/sqrt(Nh))。"
        ),
        nonfinite_policy=(
            "对数轴上的 0（某个 bin 与参考密度完全重合）写成地板 "
            f"{ERROR_FLOOR:.0e} 并配 error_valid 掩码。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 3：Sobol 指数柱状图
# --------------------------------------------------------------------------- #
_SENS_CACHE: dict[tuple[str, int, float, str], dict[str, Any]] = {}
_CORR_N = 40000


def _sens_data(model: str, n: int, rho: float, estimator: str) -> dict[str, Any]:
    """pick-freeze 指数 + bootstrap 区间 + 估计器漂移 + （可选）相关输入的两套分解。"""
    key = (str(model), int(n), float(rho), str(estimator))
    hit = _SENS_CACHE.get(key)
    if hit is not None:
        return hit
    mdl = sensitivity.make_model(model)
    ind = sensitivity.pick_freeze_indices(model, int(n), seed=20261018, bootstrap=200)
    ns, drift_first, drift_total = sensitivity.estimator_drift(
        model, (64, 128, 256, 512, 1024, 2048, 4096), seed=20261018
    )
    corr_applicable = mdl.domain.startswith("normal")
    extra: dict[str, Any] = {"corr_applicable": bool(corr_applicable)}
    if corr_applicable and abs(float(rho)) > 1e-12:
        x = sensitivity.correlated_normal_samples(float(rho), _CORR_N, 2, seed=7)
        y = np.asarray(mdl.func(x), dtype=np.float64)
        st = sensitivity.structural_indices(x, y, n_bins=64, rho=float(rho))
        sh = sensitivity.shuffled_indices(mdl, x, seed=11)
        extra.update(
            {
                "between": np.asarray(st.between, dtype=np.float64),
                "within": np.asarray(st.within, dtype=np.float64),
                "overlap_excess": float(st.overlap_excess),
                "indep_first": np.asarray(sh.first, dtype=np.float64),
                "indep_sum": float(sh.sum_first),
                "rho_measured": _measured_corr(x),
            }
        )
    out = {
        "model": mdl,
        "ind": ind,
        "drift_ns": ns,
        "drift_first": drift_first,
        "drift_total": drift_total,
        **extra,
    }
    if len(_SENS_CACHE) < 24:
        _SENS_CACHE[key] = out
    return out


def _measured_corr(x: np.ndarray) -> float:
    """样本相关系数（自研：P3 禁 ``np.corrcoef``）。"""
    a = np.asarray(x, dtype=np.float64)
    if a.shape[1] < 2:
        return float("nan")
    u = a[:, 0] - float(np.mean(a[:, 0]))
    v = a[:, 1] - float(np.mean(a[:, 1]))
    den = math.sqrt(float(np.sum(u * u)) * float(np.sum(v * v)))
    return float(np.sum(u * v) / den) if den > 0 else float("nan")


def _sobol_bars(params: Mapping[str, Any], t: float) -> Frame:
    model = str(params["model"])
    n = int(params["n"])
    rho = float(params["rho"])
    estimator = str(params["estimator"])
    data = _sens_data(model, n, rho, estimator)
    mdl = data["model"]
    ind = data["ind"]
    d = ind.dim
    use_alt = estimator == "jansen"
    first = np.asarray(ind.first_alt if use_alt else ind.first, dtype=np.float64)
    total = np.asarray(ind.total_alt if use_alt else ind.total, dtype=np.float64)
    err_first = np.asarray(ind.first_halfwidth, dtype=np.float64)
    err_total = np.asarray(ind.total_halfwidth, dtype=np.float64)
    x_pos = np.arange(d, dtype=np.float64)

    corr_on = bool(data.get("corr_applicable")) and abs(rho) > 1e-12
    if corr_on:
        indep = np.asarray(data["indep_first"], dtype=np.float64)
        between = np.asarray(data["between"], dtype=np.float64)
    else:
        indep = np.zeros(d, dtype=np.float64)
        between = np.zeros(d, dtype=np.float64)

    drift_sum_first = np.sum(data["drift_first"], axis=1)
    drift_sum_total = np.sum(data["drift_total"], axis=1)

    fr = Frame(chapter="5.4", viz="sobol_bars", kind="frame")
    fr.add("index_x", x_pos, unit="1", label="输入维序号 i")
    fr.add("first", first, unit="1", label="一阶指数 S_i")
    fr.add("total", total, unit="1", label="总效应指数 S_Ti")
    fr.add("error_first", err_first, unit="1", label="S_i 的 bootstrap 半宽")
    fr.add("error_total", err_total, unit="1", label="S_Ti 的 bootstrap 半宽")
    fr.add("first_other", np.asarray(ind.first if use_alt else ind.first_alt, dtype=np.float64), unit="1", label="另一估计器的一阶指数")
    fr.add("total_other", np.asarray(ind.total if use_alt else ind.total_alt, dtype=np.float64), unit="1", label="另一估计器的总效应指数")
    fr.add("first_indep", indep, unit="1", label="乘积测度口径的一阶指数（相关输入时）")
    fr.add("between_cond", between, unit="1", label="条件方差口径 Var(E[Y|Xi])/V")
    fr.add("corr_valid", np.full(d, 1.0 if corr_on else 0.0), unit="1", label="相关分解有效掩码")
    fr.add("drift_n", np.asarray(data["drift_ns"], dtype=np.float64), unit="1", label="漂移曲线的 N")
    fr.add("drift_sum_first", drift_sum_first, unit="1", label="sum S_i(N)")
    fr.add("drift_sum_total", drift_sum_total, unit="1", label="sum S_Ti(N)")
    fr.add("drift_analytic_first", np.full(drift_sum_first.size, sum(mdl.first) if mdl.has_analytic else drift_sum_first[-1]), unit="1", label="sum S_i 的解析值（无解析值时取 N 最大处）")
    fr.add("drift_analytic_total", np.full(drift_sum_total.size, sum(mdl.total) if mdl.has_analytic else drift_sum_total[-1]), unit="1", label="sum S_Ti 的解析值")

    sum_first = float(np.sum(first))
    sum_total = float(np.sum(total))
    fr.set_meta(
        dim=d,
        n=int(n),
        rho=rho,
        n_evals=int(ind.n_evals),
        sum_first=sum_first,
        sum_total=sum_total,
        sum_first_halfwidth=float(np.sqrt(float(np.sum(err_first**2)))),
        sum_total_halfwidth=float(np.sqrt(float(np.sum(err_total**2)))),
        interaction_share=float(sum_total - sum_first),
        lower_bound_ok=bool(sum_first <= 1.0 + float(np.sqrt(float(np.sum(err_first**2))))),
        upper_bound_ok=bool(sum_total >= 1.0 - float(np.sqrt(float(np.sum(err_total**2))))),
        analytic_sum_first=_finite_or_none(sum(mdl.first)) if mdl.has_analytic else None,
        analytic_sum_total=_finite_or_none(sum(mdl.total)) if mdl.has_analytic else None,
        drift_span_first=float(np.max(drift_sum_first) - np.min(drift_sum_first)),
        drift_span_total=float(np.max(drift_sum_total) - np.min(drift_sum_total)),
        corr_applicable=bool(data.get("corr_applicable")),
        corr_active=bool(corr_on),
        overlap_excess=_finite_or_none(data.get("overlap_excess")) if corr_on else None,
        rho_measured=_finite_or_none(data.get("rho_measured")) if corr_on else None,
        sum_between=_finite_or_none(float(np.sum(between))) if corr_on else None,
        sum_indep=_finite_or_none(float(np.sum(indep))) if corr_on else None,
        overflow=bool(not np.all(np.isfinite(first)) or not np.all(np.isfinite(total))),
    )
    fr.set_strings(
        model=mdl.name,
        model_label=mdl.label,
        estimator="Saltelli 2010" if not use_alt else "Jansen 1999",
        note=(
            f"{mdl.label}：N={n}，模型求值 {ind.n_evals} 次；实测 sum S_i = {sum_first:+.4f} "
            f"+/- {float(np.sqrt(float(np.sum(err_first**2)))):.4f}，"
            f"sum S_Ti = {sum_total:+.4f} +/- {float(np.sqrt(float(np.sum(err_total**2)))):.4f}，"
            f"交互份额 = {float(sum_total - sum_first):+.4f}；方差 V = {ind.variance:.4f}"
        ),
        analytic_note=(
            f"解析值：sum S_i = {sum(mdl.first):.4f}，sum S_Ti = {sum(mdl.total):.4f}；"
            f"两套估计器（Saltelli/Jansen）在同一帧里都给出来了，差异就是估计器本身的偏差。"
            if mdl.has_analytic
            else "该模型没有闭式指数：本页的所有数字都只是估计值，不许当成真值引用。"
        ),
        drift_note=(
            f"副轴是同一函数、同一随机流下 sum S_i(N) 的漂移：N 从 {int(data['drift_ns'][0])} 到 "
            f"{int(data['drift_ns'][-1])} 变化了 {float(np.max(drift_sum_first) - np.min(drift_sum_first)):.4f}"
            "——这是估计量的偏差，不是抽样噪声（它不会随单次实验的运气改变方向）。"
        ),
        corr_note=(
            f"相关输入（样本相关系数实测 {data.get('rho_measured', float('nan')):+.4f}）："
            f"条件方差口径 sum Var(E[Y|Xi])/V = {float(np.sum(between)):.4f}（超出 1 的部分 "
            f"{float(data.get('overlap_excess', float('nan'))):+.4f} 就是被重复计算的方差），"
            f"乘积测度口径 sum S_i = {float(np.sum(indep)):.4f}。两者都不是「错的」，"
            "但它们是两个不同的问题——方差分解在相关输入下**不唯一**。"
            if corr_on
            else "rho = 0（或该模型的输入域不是正态）：本帧只报独立输入下的分解。"
            + (
                ""
                if data.get("corr_applicable")
                else " 该模型定义在均匀方体上，rho 对它无效——切到 bilinear / quadratic_corr 看相关效应。"
            )
        ),
        decomposition_note=(
            "本页最多同时出现三个都叫「一阶指数」的数，必须分清："
            "(1) first —— 独立输入 pick-freeze（Saltelli/Jansen）估计，提问「若输入独立，S_i 是多少」；"
            "(2) between_cond —— 用**相关样本**算的条件方差口径 Var(E[Y|Xi])/V，提问「在现在的相关结构下，"
            "Xi 单独解释了多少方差」，它的和可以大于 1；"
            "(3) first_indep —— 把相关样本逐列打散（乘积测度）后再跑一次 pick-freeze，"
            "提问「打破相关之后还剩多少」。三者相等当且仅当输入独立。"
        ),
        nonfinite_policy="帧内数组全部有限；相关分解不适用时用 0 + corr_valid 掩码标注，不写 NaN。",
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：多项式混沌
# --------------------------------------------------------------------------- #
_PCE_CACHE: dict[tuple[str, str, str, int, int, float], dict[str, Any]] = {}
#: 外推扫描的倍率（与 ``extrap`` 参数无关，是一条完整的曲线）
_EXTRAP_GRID = (1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.65, 1.8, 2.0)


def _pce_data(target: str, basis: str, fit: str, order: int, n_train: int, extrap: float) -> dict[str, Any]:
    """当前阶数的拟合 + 阶数-误差曲线 + 外推扫描（带缓存）。"""
    key = (str(target), str(basis), str(fit), int(order), int(n_train), float(extrap))
    hit = _PCE_CACHE.get(key)
    if hit is not None:
        return hit
    func = pce.target_function(target)
    current = pce.fit_target(target, order=int(order), kind=basis, method=fit, n_train=int(n_train))
    orders = tuple(range(1, 13))
    curve = pce.error_vs_order(
        func, dim=1, orders=orders, n_train=int(n_train), kind=basis, method=fit, n_test=2000
    )
    xte = pce.design_samples(1, 2000, kind=basis, seed=777)
    extrap_err = np.empty(len(_EXTRAP_GRID), dtype=np.float64)
    for i, f in enumerate(_EXTRAP_GRID):
        if f <= 1.0:
            extrap_err[i] = pce.relative_l2_error(current, func, xte)
        else:
            extrap_err[i] = pce.extrapolation_error(
                func, dim=1, order=int(order), n_train=int(n_train), kind=basis, method=fit, factor=f
            )["extrap"]
    levels, rms, cnt = pce.coefficient_spectrum(current)
    decay_slope = _decay_slope(levels, rms)
    out = {
        "fit": current,
        "levels": levels,
        "rms": rms,
        "counts": cnt,
        "curve": curve,
        "extrap_grid": np.asarray(_EXTRAP_GRID, dtype=np.float64),
        "extrap_err": extrap_err,
        "decay_slope": decay_slope,
        "train_err": pce.relative_l2_error(current, func, pce.design_samples(1, 500, kind=basis, seed=20261018)),
        "outsample_err": pce.relative_l2_error(current, func, xte),
    }
    if len(_PCE_CACHE) < 32:
        _PCE_CACHE[key] = out
    return out


def _decay_slope(levels: np.ndarray, rms: np.ndarray) -> float | None:
    """谱衰减率：对 ``log(rms_k)`` 关于 ``k`` 做一元最小二乘，只在"还在下降"的窗口上做。

    触到机器精度（``rms_k`` 相对最大值小于 1e-14）之后的点一律不参与拟合：
    那里是舍入地板，不是谱。
    """
    lv = np.asarray(levels, dtype=np.float64).ravel()
    rr = np.asarray(rms, dtype=np.float64).ravel()
    if rr.size < 4 or float(np.max(rr)) <= 0.0:
        return None
    keep = (rr > 1e-14 * float(np.max(rr))) & (rr > 0.0) & (lv > 0)
    if int(np.count_nonzero(keep)) < 3:
        return None
    x = lv[keep]
    y = np.log(rr[keep])
    mx = float(np.mean(x))
    my = float(np.mean(y))
    den = float(np.sum((x - mx) ** 2))
    if den <= 0.0:
        return None
    return float(np.sum((x - mx) * (y - my)) / den)


def _pce_spectrum(params: Mapping[str, Any], t: float) -> Frame:
    target = str(params["target"])
    basis = str(params["basis"])
    fit_method = str(params["fit"])
    order = int(params["order"])
    n_train = int(params["n_train"])
    extrap = float(params["extrap"])
    data = _pce_data(target, basis, fit_method, order, n_train, extrap)
    current = data["fit"]
    curve = data["curve"]
    rms = np.asarray(data["rms"], dtype=np.float64)
    levels = np.asarray(data["levels"], dtype=np.float64)

    out_curve = np.asarray(curve["outsample"], dtype=np.float64)
    train_curve = np.asarray(curve["train"], dtype=np.float64)
    err_log, err_valid = _log10_floor(out_curve)
    fr = Frame(chapter="5.4", viz="pce_spectrum", kind="frame")
    fr.add("degree", levels, unit="1", label="总阶数 k = |alpha|")
    fr.add("coeff_rms", _log10_floor(rms)[0], unit="1", label="log10(逐阶系数 RMS)")
    fr.add("coeff_valid", _log10_floor(rms)[1], unit="1", label="coeff_rms 有效掩码")
    fr.add("coeff_count", np.asarray(data["counts"], dtype=np.float64), unit="1", label="该阶的多指标个数")
    fr.add("order_curve", np.asarray(curve["orders"], dtype=np.float64), unit="1", label="截断阶数 p")
    fr.add("error", err_log, unit="1", label="log10(留出集相对 L2 误差)")
    fr.add("error_valid", err_valid, unit="1", label="error 有效掩码")
    fr.add("error_train", _log10_floor(train_curve)[0], unit="1", label="log10(训练集相对 L2 误差)")
    fr.add("n_basis_curve", np.asarray(curve["n_basis"], dtype=np.float64), unit="1", label="基规模")
    fr.add("under_flag", np.asarray(curve["underdetermined"], dtype=np.float64), unit="1", label="欠定（过拟合）标记")
    fr.add("extrap_x", np.asarray(data["extrap_grid"], dtype=np.float64), unit="1", label="外推倍率")
    fr.add("error_extrap", _log10_floor(data["extrap_err"])[0], unit="1", label="log10(外推相对 L2 误差)")
    fr.add("decay_ref", _decay_ref(levels, rms, data["decay_slope"]), unit="1", label="log10(指数衰减参考线)")

    exact_order = None
    for i, e in enumerate(out_curve):
        if e < 1e-12:
            exact_order = int(curve["orders"][i])
            break
    ratio = float(data["extrap_err"][-1] / data["outsample_err"]) if data["outsample_err"] > 0 else float("inf")
    fr.set_meta(
        order=order,
        n_basis=int(current.n_basis),
        n_train=int(n_train),
        underdetermined=bool(current.underdetermined),
        # 用训练集本身（不是另抽的 500 点）——否则"训练误差"会变成又一个留出误差，
        # 过拟合区那两条曲线的对比就消失了
        error_train_current=float(train_curve[int(order) - 1]),
        error_outsample_current=float(data["outsample_err"]),
        error_extrap_current=float(data["extrap_err"][-1]),
        extrap_ratio=_finite_or_none(ratio),
        extrap_factor=extrap,
        decay_slope=_finite_or_none(data["decay_slope"]),
        residual_train_l2=_finite_or_none(float(current.residual_train)),
        exact_at_order=exact_order,
        min_outsample=float(np.min(out_curve)),
        overfit_factor=(
            float(np.max(out_curve) / np.min(out_curve)) if float(np.min(out_curve)) > 1e-12 else None
        ),
        n_underdetermined=int(np.sum(np.asarray(curve["underdetermined"]))),
        overflow=bool(not np.all(np.isfinite(out_curve)) or not np.all(np.isfinite(train_curve))),
    )
    fr.set_strings(
        target=target,
        target_label=pce.target_label(target),
        basis=basis,
        fit=fit_method,
        note=(
            f"{pce.target_label(target)}：当前 p={order}（基规模 {current.n_basis}，"
            f"训练样本 {n_train}）→ 训练集误差 {float(train_curve[int(order) - 1]):.3e}，"
            f"**留出集**误差 {data['outsample_err']:.3e}，"
            f"外推 {data['extrap_grid'][-1]:.2f} 倍时 {data['extrap_err'][-1]:.3e}"
            f"（放大 {ratio:.3g} 倍）"
        ),
        overfit_note=(
            "error（留出集）与 error_train（训练集）必须分开读："
            "训练误差到机器零而留出误差上升的那一段就是过拟合，"
            "它的起点是基规模超过样本数的地方（under_flag = 1）。"
        ),
        extrap_note=(
            "外推曲线是训练支撑集**之外**的误差：多项式混沌对外推没有收敛保证，"
            "只有能被有限阶多项式精确表示的目标（poly4）例外——那是函数本身的性质。"
        ),
        floor_note=f"对数轴地板 {ERROR_FLOOR:.0e}：精确为 0 的误差（多项式被精确投影）写在地板上并配掩码。",
        nonfinite_policy="帧内数组全部有限；触地板的点用 error_valid / coeff_valid 掩码标注。",
    )
    return fr


def _decay_ref(levels: np.ndarray, rms: np.ndarray, slope: float | None) -> np.ndarray:
    """指数衰减参考线 ``log10(c0) + slope*k/ln10``（半对数图上是一条直线）。"""
    lv = np.asarray(levels, dtype=np.float64).ravel()
    rr = np.asarray(rms, dtype=np.float64).ravel()
    if slope is None or rr.size == 0 or float(np.max(rr)) <= 0.0:
        return np.log10(ERROR_FLOOR) * np.ones_like(lv)
    base = float(np.max(rr))
    vals = base * np.exp(slope * lv)
    return np.log10(np.maximum(vals, ERROR_FLOOR))
