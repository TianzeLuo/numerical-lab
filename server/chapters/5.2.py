"""5.2 章：反问题与正则化（帧装配层）。

四个可视化各自回答一个**会被讲错**的问题：

1. ``lcurve_tradeoff`` —— L 曲线与"最优 lambda"：拖 lambda 沿曲线滑动，同时把
   ``lambda_oracle``（知道真解，现实中拿不到）、``lambda_Lcurve``、``lambda_GCV``、
   ``lambda_discrepancy`` 四个数量在同一张图上标出，并把"用错判据误差放大多少倍"
   写进 meta。拐点本身给**三个判据**（最大曲率 / 两段直线拟合交点 / 离散二阶差分），
   它们的 lambda 实测可以差几十倍——"拐点"不是一个客观量。
2. ``filter_shapes`` —— 滤波因子：理想低通（TSVD）与平滑过渡（Tikhonov）在同一条
   σ 轴上并排，Landweber 是第三条；外加"同样自由度下谁更准"的实测。TV 只有**探针**
   读数（它不是谱滤波，换一个探针数就变）。
3. ``image_deconvolution`` —— 图像反卷积 / 简化 CT：拖噪声水平与 lambda（或迭代步），
   看解什么时候被噪声吞掉。Landweber/TV 的半收敛（误差先降后升）逐步可见；
   CT 侧还能看到"角度不够时 A 秩亏，误差地板由**零空间**而不是噪声决定"。
4. ``bias_variance`` —— 偏差-方差分解：偏差^2、方差与它们的和随 lambda 的三条曲线，
   蒙特卡洛实测与解析式并列，最优 lambda 处标出；lambda->0 与 lambda->∞ 两侧各自的失败方式都有数字。

帧纪律（与 1.2/5.1 一致）：``meta`` 只放 number/bool/None，字符串走 ``strings``；
没有有限值写 ``None`` 并说明原因；对数轴上的 0 给地板值 + 掩码 + 文案三件套；
``compute`` 在参数角点与 ``t`` 端点都能构建（奇异/发散变成可显示的状态，不抛异常）。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np

from numcore.inverse.lcurve import (
    criterion_comparison,
    gcv_curve,
    lcurve,
    lambda_oracle,
    problem_lcurve,
)
from numcore.inverse.regularize import (
    bias_variance_analytic,
    bias_variance_montecarlo,
    effective_dof,
    factorize,
    filtered_solution,
    landweber_factors,
    landweber_operator,
    noise_dominated_count,
    phantom_image,
    picard_diagnostics,
    radon_matrix,
    relative_error_norm,
    separable_apply_factors,
    separable_blur_adjoint,
    separable_blur_forward,
    separable_filter_factors,
    separable_tikhonov,
    separable_tsvd,
    spatial_blur_matrix,
    test_problem,
    tikhonov_factors,
    tikhonov,
    tsvd_factors,
    tv_filter_probe,
    tv_solve,
    unregularized_amplification,
    vector_norm,
)
from numcore.utils import rng as _rng
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 对数轴地板（0 在对数轴上画不出来）
LOG_FLOOR = 1e-17
#: 误差类曲线的显示上限（发散时写它并配掩码；**不是**真值）
ERROR_CEILING = 1e6
#: 无效标记的占位（数组里不许有 NaN，用 -99 表示"这一项没有有限值"，配掩码读）
INVALID = -99.0


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _finite_or_none(x: Any) -> float | None:
    """``NaN/±Inf`` -> ``None``（PROTOCOLS §1.3：JSON 头里没有 NaN/Infinity）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _floored(values: Any, floor: float = LOG_FLOOR) -> np.ndarray:
    """把非负曲线抬到对数轴地板上（0 在对数轴画不出来）。"""
    return np.maximum(np.asarray(values, dtype=float), float(floor))


def _clip_ceiling(values: Any, ceiling: float = ERROR_CEILING) -> tuple[np.ndarray, np.ndarray]:
    """把发散的量夹到显示上限，并给出"被夹过"的掩码（占位值 + 掩码 + 文案）。"""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v) & (v <= float(ceiling))
    return np.where(ok, v, float(ceiling)), ok


def _pad_tail(values: Any, n: int, fill: float = 0.0) -> np.ndarray:
    """把曲线补到长度 ``n``（尾部重复末点；空数组用 ``fill`` 填满）。"""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return np.full(int(n), float(fill))
    if v.size >= int(n):
        return v[: int(n)]
    return np.concatenate([v, np.full(int(n) - v.size, float(v[-1]) if fill is None else float(fill))])


def _pad_mask(values: Any, n: int) -> np.ndarray:
    """布尔掩码补到长度 ``n``（补位处不是真数据 -> 0）。"""
    v = np.asarray(values).astype(bool).ravel()
    if v.size >= int(n):
        return v[: int(n)]
    return np.concatenate([v, np.zeros(int(n) - v.size, dtype=bool)])


def _step_index(t: float, n_max: int) -> int:
    """把 scrubbing 的 ``t`` 变成整数步（夹在 ``[0, n_max]``）。"""
    return int(min(max(int(round(float(t))), 0), int(n_max)))


def _pow10(exp: Any) -> float:
    """``10^exp``（把滑块上的指数变成量级；指数本身进 meta 便于读数）。"""
    return float(10.0 ** float(exp))


def _loglog_slope(x: Any, y: Any, *, skip: int = 0) -> float:
    """双对数斜率（自研最小二乘；P3 禁现成相关系数）。"""
    xv = np.asarray(x, dtype=float).ravel()[int(skip) :]
    yv = np.asarray(y, dtype=float).ravel()[int(skip) :]
    good = np.isfinite(xv) & np.isfinite(yv) & (xv > 0.0) & (yv > 0.0)
    xv, yv = np.log(xv[good]), np.log(yv[good])
    if xv.size < 2:
        return float("nan")
    xm, ym = float(np.mean(xv)), float(np.mean(yv))
    sxx = float(np.sum((xv - xm) ** 2))
    if sxx == 0.0:
        return float("nan")
    return float(np.sum((xv - xm) * (yv - ym)) / sxx)


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
_MANIFEST_PITFALLS_LCURVE = (
    "「最优 lambda」必须说清是按**什么准则**最优：这一页同时给出 oracle（知道真解，"
    "现实中拿不到）、L 曲线拐点、GCV 与偏差原则四个数，实测（κ=1e6、噪声 1e-3）"
    "L 曲线比 oracle 小 22 倍、GCV 大 4.6 倍，误差分别放大 21 倍与 6.7 倍——"
    "只看一条曲线上的拐点，看到的既不是真误差最小点，也不是同一个数。"
    "更糟的是「拐点」本身没有唯一判据：最大曲率、两段直线拟合交点、离散二阶差分"
    "三个判据给出的 lambda 实测可以差 5--1000 倍（18 组配置测下来最小 5 倍、"
    "最大 1200 倍）；两段直线拟合若不加「每段至少 25% 的点 + 交点在包围盒内」"
    "这两条约束，实测在 18 组里有 9 组会跳到另一个答案（最大偏 126 倍）——"
    "它不是坏在端点，是根本不给同一个数。"
)

_MANIFEST_PITFALLS_FILTER = (
    "把 TV 画成第四条滤波曲线就是在撒谎：全变差不是谱滤波，没有滤波因子。"
    "这里的 TV 曲线是**探针读数**——用 b = u_i 逐个探出来的有效因子，实测把探针换成 "
    "b = (u_1+u_n)/sqrt(2)，同一个下标上的读数就变（差 0.4 量级），非对角泄漏也不为 0。"
    "另一条：ω 拖过 2/σ_1^2 之后 Landweber 当场发散，滤波因子跑出 [0,1] 区间，"
    "图上那一段是被夹到边界并打上掩码的占位值，不是「滤波器变平了」。"
    "最后，「同样自由度下谁更准」不是「形状不同」的同义词：实测 TSVD 在中低自由度"
    "往往更准，而 Tikhonov 在自由度接近满秩时反超——这条必须看两条误差曲线，不能看形状。"
)

_MANIFEST_PITFALLS_IMAGE = (
    "「正则化让图更清楚」不是结论：lambda 太小解被噪声吞掉（κ=1e6 时噪声 1e-3 被放大到"
    "1e3 量级），lambda 太大图像被抹平到只剩均值；两条失败路径在图上都看得见。"
    "Landweber/TV 还多一条半收敛：误差先降后升，最优迭代步 k* 只能靠早停拿到，"
    "而最终的残差比 k* 处更小——**残差小不等于解准**。"
    "CT 侧相反地会骗人一次：角度够多时这个尺寸的投影矩阵条件数只有 250（良态），"
    "而角度减到 8 条时 A 秩亏，误差地板由零空间决定——此时把噪声拖到 0，"
    "误差仍然停在 0.10，那不是噪声问题，是算子本身丢了信息。"
)

_MANIFEST_PITFALLS_BIAS = (
    "「偏差与方差此消彼长」这句话本身是对的，但它常被读成「正则化让结果更准」——"
    "不是：lambda->∞ 时方差趋于 0 而偏差^2趋于 ||x*||^2，总误差回到 100%；"
    "lambda->0 时偏差趋于 0 而方差按 lambda^(-4) 只在地板之外才开始下降，总误差被噪声接管。"
    "所以「总误差最小」只在**中间**存在，而且那个点是按真解定义的 oracle——"
    "曲线上的最低点不是你能算出来的量。实测（κ=1e4、噪声 1e-3）解析与蒙特卡洛"
    "两条总误差曲线的最小点相差不到 2 个网格点，但用 L 曲线拐点会偏出 10 倍以上。"
)

MANIFEST: dict[str, Any] = {
    "chapter": "5.2",
    "title": "反问题与正则化",
    "part": "V",
    "summary": (
        "病态性与奇异值衰减、Tikhonov 与谱滤波因子、L 曲线与广义交叉验证、TSVD、"
        "迭代正则化与早停、全变差、偏差-方差分解——以及「最优 lambda 是按什么准则最优」"
        "这一句必须被实测数字钉死的话。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "lcurve_tradeoff",
            "title": "L 曲线交互：拖 lambda 滑动，四个判据同时标出",
            "description": (
                "在 (log||x_lambda||, log||A x_lambda - b||/||b||) 平面上画出 L 曲线，"
                "拖动 lambda 时解与残差同步移动；同一张图上标出 oracle（知道真解）、"
                "L 曲线三个拐点判据、GCV、偏差原则各自挑出的 lambda，并把"
                "「用错判据误差放大多少倍」写进 meta。副轴是解的真误差随 lambda 的曲线。"
                "把 Picard 档位从 satisfied 拖到 violated，可以看到正则化本身什么时候失效。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "log10_kappa",
                    "label": "lg(kappa) 条件数指数",
                    "type": "int",
                    "min": 2,
                    "max": 8,
                    "default": 6,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "log10_noise",
                    "label": "lg(噪声相对水平)",
                    "type": "int",
                    "min": -8,
                    "max": -1,
                    "default": -3,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "log10_lam",
                    "label": "lg(lambda) 当前正则参数",
                    "type": "float",
                    "min": -6.0,
                    "max": 4.0,
                    "step": 0.1,
                    "default": -2.0,
                    "critical": [0.0],
                    "criticalNote": (
                        "lambda = 1 = sigma_1（谱的上端）时所有滤波因子都不超过 1/2，"
                        "解被系统性压小；越过它继续加大 lambda，滤波因子按 1/lambda^2 收缩，"
                        "解趋向 0、误差重新升到 ||x*||——这一侧已经不是「在滤噪」了。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "picard",
                    "label": "离散 Picard 条件",
                    "type": "enum",
                    "options": ["satisfied", "borderline", "violated"],
                    "default": "satisfied",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "spectrum",
                    "label": "奇异值谱形状",
                    "type": "enum",
                    "options": ["decay", "cliff", "plateau", "two_clusters"],
                    "default": "decay",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "lcurve",
                    "autoscaled": True,
                    "title": "L 曲线（横=log10||x||，纵=log10 残差）",
                    "ylabel": "log10 残差",
                },
                {
                    "id": "error",
                    "autoscaled": True,
                    "title": "真误差随 λ（对数轴）",
                    "ylabel": "相对误差",
                    "scale": "log",
                },
                {
                    "id": "picard",
                    "autoscaled": True,
                    "title": "Picard：系数 vs σ",
                    "ylabel": "log|u_i^T b|",
                    "scale": "log",
                },
            ],
            "pitfalls": _MANIFEST_PITFALLS_LCURVE,
            "theory": [
                "Tikhonov 的滤波因子 f_i = s_i^2/(s_i^2+lambda^2)：lambda 是平方参数，"
                "半功率点恰在 s_i = lambda",
                "L 曲线的拐点是启发式判据；离散 Picard 条件（|u_i^T b| 比 s_i 衰减更快）"
                "才是「截断有意义」的前提",
                "GCV(lambda) = ||A x_lambda - b||^2 / (m - sum_i f_i)^2 只用 A 与 b，"
                "不需要噪声水平；偏差原则需要噪声水平，但只说「残差降到噪声尺度」",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "filter_shapes",
            "title": "滤波因子曲线：理想低通、平滑过渡与「同自由度谁更准」",
            "description": (
                "同一条奇异值轴上画三条滤波因子：TSVD（理想低通，1 到 0 的台阶）、"
                "Tikhonov（平滑过渡，过渡带宽度由 lambda 控制）、Landweber（由迭代步数"
                "控制的渐进过渡）。TV 只能给探针读数（它不是谱滤波）。"
                "副轴把「同样自由度下谁更准」画出来：三条误差-自由度曲线 + 匹配点。"
                "把 omega_rel 拖过 2，Landweber 当场发散（滤波因子跑出 [0,1]）。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "log10_kappa",
                    "label": "lg(kappa) 条件数指数",
                    "type": "int",
                    "min": 2,
                    "max": 10,
                    "default": 6,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "log10_noise",
                    "label": "lg(噪声相对水平)",
                    "type": "int",
                    "min": -8,
                    "max": -2,
                    "default": -4,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "log10_lam",
                    "label": "lg(lambda) Tikhonov 参数",
                    "type": "float",
                    "min": -5.0,
                    "max": 3.0,
                    "step": 0.1,
                    "default": -2.0,
                    "critical": [0.0],
                    "criticalNote": (
                        "lambda = sigma_1 是平滑过渡带刚好压住整个谱的位置："
                        "越过它以后每一条 f_i 都小于 1/2，曲线整体贴到 0，"
                        "「滤波形状」的差别被压平（这时解已接近 0）。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "k_tsvd",
                    "label": "TSVD 截断阶 k",
                    "type": "int",
                    "min": 1,
                    "max": 20,
                    "default": 8,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_iter",
                    "label": "Landweber 迭代步数",
                    "type": "int",
                    "min": 1,
                    "max": 200,
                    "default": 30,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "omega_rel",
                    "label": "步长 omega / (1/sigma_1^2)",
                    "type": "float",
                    "min": 0.1,
                    "max": 2.5,
                    "step": 0.05,
                    "default": 1.0,
                    "critical": [1.0, 2.0],
                    "criticalNote": (
                        "omega_rel = 1 是最快收敛点（omega = 1/sigma_1^2，最慢模态一步压到 0）；"
                        "omega_rel = 2 是**发散线**：omega >= 2/sigma_1^2 时 |1 - omega s_1^2| >= 1，"
                        "迭代值指数增长，滤波因子跑出 [0,1]——图上那一段是夹住的占位值。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "tv_probe",
                    "label": "TV 探针",
                    "type": "enum",
                    "options": ["single", "mixed"],
                    "default": "single",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "filters",
                    "autoscaled": False,
                    "title": "滤波因子 f_i 随模态下标 i（奇异值递减）",
                    "ylabel": "f_i（0 到 1）",
                },
                {
                    "id": "dof_error",
                    "autoscaled": True,
                    "title": "误差随有效自由度（同样自由度下谁更准）",
                    "ylabel": "相对误差",
                    "scale": "log",
                },
            ],
            "pitfalls": _MANIFEST_PITFALLS_FILTER,
            "theory": [
                "TSVD 滤波因子是阶跃 1{i<=k}：理想低通，截止处没有过渡带；"
                "Tikhonov 是 s^2/(s^2+lambda^2)：过渡带宽约一个 lambda 的十倍程",
                "Landweber 的滤波因子 1-(1-omega s^2)^k 在 0 < omega < 2/sigma_1^2 时"
                "单调上升到 1，omega 越大越快、越过 2/sigma_1^2 发散",
                "有效自由度 sum_i f_i：TSVD 是 k，Tikhonov 由 lambda 决定——"
                "「同样自由度」是两种方法唯一公平的比较轴",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "image_deconvolution",
            "title": "图像反卷积 / 简化 CT：噪声什么时候把解吞掉",
            "description": (
                "24x24 幻影经可分离高斯模糊 + 噪声，四类方法重建：Tikhonov、TSVD、"
                "Landweber、TV。拖噪声水平看解什么时候被噪声吞掉，拖迭代步看半收敛"
                "（误差先降后升）；把算子换成简化平行束 CT，还能看到角度不够时 "
                "A 秩亏、误差地板由零空间决定。副轴是误差随迭代步 / 截断阶的历史。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "operator",
                    "label": "正算子",
                    "type": "enum",
                    "options": ["blur", "ct"],
                    "default": "blur",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "blur_sigma",
                    "label": "模糊宽度 sigma（blur 用）",
                    "type": "float",
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.1,
                    "default": 2.0,
                    "critical": [2.4],
                    "criticalNote": (
                        "sigma 越过 2.4 左右时一维模糊矩阵的最小奇异值跌破 1e-8，"
                        "二维条件数越过 1e16——噪声 1e-3 之下**没有任何 lambda** 能把"
                        "最小模态救回来，图上只剩平滑区域。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "n_angles",
                    "label": "CT 角度数（ct 用）",
                    "type": "int",
                    "min": 6,
                    "max": 16,
                    "default": 8,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "log10_noise",
                    "label": "lg(噪声相对水平)",
                    "type": "int",
                    "min": -8,
                    "max": -1,
                    "default": -3,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "log10_lam",
                    "label": "lg(lambda) 正则/截断阈值",
                    "type": "float",
                    "min": -6.0,
                    "max": 1.0,
                    "step": 0.1,
                    "default": -3.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "method",
                    "label": "重建方法",
                    "type": "enum",
                    "options": ["tikhonov", "tsvd", "landweber", "tv"],
                    "default": "tikhonov",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_iter",
                    "label": "迭代步数（landweber / tv）",
                    "type": "int",
                    "min": 5,
                    "max": 150,
                    "default": 60,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 60.0, "dt": 1.0, "fps": 30, "label": "迭代 / 截断阶 k"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "image",
                    "autoscaled": False,
                    "title": "四联热图：真解 / 数据 / 重建 / 误差（独立色标）",
                    "ylabel": "像素",
                },
                {
                    "id": "history",
                    "autoscaled": True,
                    "title": "误差随迭代步（或截断阶 k）",
                    "ylabel": "相对误差",
                    "scale": "log",
                },
                {
                    "id": "lambda_sweep",
                    "autoscaled": True,
                    "title": "误差随 λ（两端失败）",
                    "ylabel": "相对误差",
                    "scale": "log",
                },
            ],
            "pitfalls": _MANIFEST_PITFALLS_IMAGE,
            "theory": [
                "可分离模糊的二维奇异值是 s_i s_j：二维条件数是一维条件数的平方，"
                "这正是「图像反卷积比一维难得多」的定量来源",
                "Landweber 的迭代解等于滤波因子 1-(1-omega s_i^2)^k 的谱滤波："
                "k 起正则参数的作用，early stopping 就是选择 k",
                "TV 的先验是有界变差（分片常数），因此它在分片常数的幻影上比 L2 先验"
                "更合适；但目标函数非光滑，解不对应任何滤波因子",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "bias_variance",
            "title": "偏差-方差分解：lambda 两侧各自的失败方式",
            "description": (
                "lambda 网格上画三条曲线：偏差^2、方差、以及它们的和（期望总误差）；"
                "解析式与蒙特卡洛实测并列（同一批噪声样本）。最优 lambda 处标出，"
                "并给出 lambda->0（噪声接管）与 lambda->∞（解被压成 0）两侧的失败数字，"
                "以及方差在两侧的对数-对数斜率（理论 0 与 -4）。"
                "把 Picard 档位拖到 violated，可以看到「没有中间最小值」是什么样子。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "log10_kappa",
                    "label": "lg(kappa) 条件数指数",
                    "type": "int",
                    "min": 2,
                    "max": 8,
                    "default": 4,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "log10_noise",
                    "label": "lg(噪声相对水平)",
                    "type": "int",
                    "min": -6,
                    "max": -2,
                    "default": -3,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "log10_lam",
                    "label": "lg(lambda) 当前点",
                    "type": "float",
                    "min": -4.0,
                    "max": 4.0,
                    "step": 0.1,
                    "default": -2.0,
                    "critical": [0.0],
                    "criticalNote": (
                        "lambda = sigma_1 之后偏差^2已经吃掉总误差的一半以上"
                        "（解被压向 0），继续往右是「方差很小、但答案根本不是解」的区域。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "n_trials",
                    "label": "蒙特卡洛样本数",
                    "type": "int",
                    "min": 8,
                    "max": 256,
                    "default": 64,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "picard",
                    "label": "离散 Picard 条件",
                    "type": "enum",
                    "options": ["satisfied", "violated"],
                    "default": "satisfied",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "decomposition",
                    "autoscaled": True,
                    "title": "偏差²、方差与总误差随 λ（对数轴）",
                    "ylabel": "平方误差量级",
                    "scale": "log",
                },
                {
                    "id": "variance_slope",
                    "autoscaled": True,
                    "title": "方差的局部斜率（应当看到 0 与 -4 两段）",
                    "ylabel": "d log10(方差) / d log10(lambda)",
                },
            ],
            "pitfalls": _MANIFEST_PITFALLS_BIAS,
            "theory": [
                "E||x_lambda - x*||^2 = ||E x_lambda - x*||^2 + tr(Cov x_lambda)，"
                "两项分别是偏差^2与方差",
                "白噪声总范数 delta 时 Var = (delta^2/m) sum_i (f_i/s_i)^2；"
                "lambda >> sigma_1 时 f_i ~ s_i^2/lambda^2，方差按 lambda^(-4) 下降",
                "最优 lambda 由 d/dlambda (偏差^2+方差) = 0 给出，只依赖真解与噪声水平，"
                "两者在现实中都拿不到——这就是 oracle 与 GCV/L 曲线的差",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "lcurve_tradeoff":
        return _lcurve_frame(params, t)
    if viz_id == "filter_shapes":
        return _filter_frame(params, t)
    if viz_id == "image_deconvolution":
        return _image_frame(params, t)
    if viz_id == "bias_variance":
        return _bias_variance_frame(params, t)
    raise KeyError(f"章节 5.2 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# viz 1：L 曲线交互
# --------------------------------------------------------------------------- #
_LCRITERIA = ("oracle", "lcurve", "gcv", "discrepancy")


def _lcurve_frame(params: Mapping[str, Any], t: float) -> Frame:
    del t  # 时间轴 mode="none"：这一页的横轴是 L 曲线，不是时刻
    m, n = 32, 20
    kappa = _pow10(params["log10_kappa"])
    noise_rel = _pow10(params["log10_noise"])
    picard = str(params["picard"])
    spectrum = str(params["spectrum"])
    lam_cur = _pow10(params["log10_lam"])

    prob = test_problem(m, n, kappa, spectrum=spectrum, picard=picard, noise_rel=noise_rel)
    lc = problem_lcurve(prob, n_lam=61)
    choice = criterion_comparison(
        prob.U, prob.s, prob.V, prob.b,
        x_true=prob.x_true, noise_abs=prob.noise_abs, lam=lc.lam, m=m,
    )
    pic = picard_diagnostics(prob.U, prob.s, prob.b, noise_abs=prob.noise_abs)
    sol = tikhonov(prob.U, prob.s, prob.V, prob.b, lam_cur, x_true=prob.x_true)
    gcv = gcv_curve(prob.U, prob.s, prob.b, lc.lam, m=m)

    fr = Frame(chapter="5.2", viz="lcurve_tradeoff", kind="frame")
    fr.add("log10_lam", np.log10(lc.lam), unit="", label="lg(lambda)")
    fr.add("log10_x_norm", lc.log10_x_norm, unit="", label="lg||x_lambda||")
    fr.add("log10_residual", _floored(lc.residual, LOG_FLOOR), unit="", label="lg(||A x_lambda - b||/||b||)")
    fr.add("error_curve", _floored(lc.error, LOG_FLOOR), unit="", label="相对真误差（oracle 曲线）")
    fr.add("curvature", _floored(lc.curvature, LOG_FLOOR), unit="", label="对数平面上的离散曲率")
    fr.add("dof", lc.dof, unit="", label="有效自由度 sum f_i")
    fr.add("gcv", _floored(gcv, LOG_FLOOR), unit="", label="GCV(lambda)")

    # 四个判据的标记（数组里不许有 NaN：无效项写 INVALID 并用掩码标出）
    crit_lam = np.array(
        [
            math.log10(choice.lam_oracle),
            math.log10(choice.lam_lcurve),
            math.log10(choice.lam_gcv),
            math.log10(choice.lam_discrepancy) if np.isfinite(choice.lam_discrepancy) else INVALID,
        ],
        dtype=float,
    )
    crit_err = np.array(
        [
            choice.error_oracle,
            choice.error_lcurve,
            choice.error_gcv,
            choice.error_discrepancy if np.isfinite(choice.error_discrepancy) else INVALID,
        ],
        dtype=float,
    )
    crit_valid = np.array([True, True, True, bool(np.isfinite(choice.lam_discrepancy))])
    fr.add("criterion_log10_lam", crit_lam, unit="", label="四个判据的 lg(lambda)（oracle/Lcurve/GCV/偏差原则）")
    fr.add("criterion_error", np.where(crit_valid, _floored(crit_err, LOG_FLOOR), LOG_FLOOR), unit="", label="四个判据处的真误差")
    fr.add("criterion_valid", crit_valid.astype(float), unit="", label="判据有效掩码（INVALID 是占位符）")

    corners = choice.lam_corners
    fr.add("corner_log10_lam", np.log10(np.asarray(corners, dtype=float)), unit="", label="三个拐点判据的 lg(lambda)")

    pic_sigma = prob.s
    fr.add("picard_sigma", _floored(pic_sigma, LOG_FLOOR), unit="", label="奇异值 sigma_i")
    fr.add("picard_coeff", _floored(pic.coeff, LOG_FLOOR), unit="", label="|u_i^T b|")
    fr.add("picard_ratio", _floored(pic.ratio, LOG_FLOOR), unit="", label="|u_i^T b| / sigma_i")
    fr.add("picard_below_noise", (pic.coeff < prob.noise_abs).astype(float), unit="", label="系数已落到噪声之下的掩码")
    fr.add("solution_true", prob.x_true, unit="", label="真解 x*")
    fr.add("solution_lam", sol.x, unit="", label="当前 lambda 的解")

    oracle_at_edge = bool(
        choice.lam_oracle <= lc.lam[0] * (1.0 + 1e-9) or choice.lam_oracle >= lc.lam[-1] * (1.0 - 1e-9)
    )
    amp_worst = unregularized_amplification(
        prob.s, prob.noise_abs, m=m, x_norm=prob.x_norm, worst_case=True
    )
    amp_rms = unregularized_amplification(
        prob.s, prob.noise_abs, m=m, x_norm=prob.x_norm, worst_case=False
    )
    fr.set_meta(
        kappa_measured=float(prob.detail["kappa_measured"]),
        noise_rel=float(prob.noise_rel),
        noise_abs=float(prob.noise_abs),
        b_norm=float(prob.b_norm),
        x_norm=float(prob.x_norm),
        n_modes=int(n),
        rank=int(prob.rank),
        lam_current=float(lam_cur),
        log10_lam_current=float(params["log10_lam"]),
        dof_current=float(sol.dof),
        error_current=float(sol.error_rel),
        residual_current=float(sol.residual_rel),
        log10_lam_oracle=float(math.log10(choice.lam_oracle)),
        log10_lam_lcurve=float(math.log10(choice.lam_lcurve)),
        log10_lam_gcv=float(math.log10(choice.lam_gcv)),
        log10_lam_discrepancy=_finite_or_none(
            math.log10(choice.lam_discrepancy) if np.isfinite(choice.lam_discrepancy) else float("nan")
        ),
        error_oracle=float(choice.error_oracle),
        error_lcurve=float(choice.error_lcurve),
        error_gcv=float(choice.error_gcv),
        error_discrepancy=_finite_or_none(choice.error_discrepancy),
        inflation_lcurve=float(choice.inflation_lcurve),
        inflation_gcv=float(choice.inflation_gcv),
        inflation_discrepancy=_finite_or_none(choice.inflation_discrepancy),
        inflation_unregularized=float(choice.detail["inflation_unregularized"]),
        error_unregularized=float(choice.detail["error_unregularized"]),
        ratio_lam_lcurve_over_oracle=float(choice.lam_lcurve / choice.lam_oracle),
        ratio_lam_gcv_over_oracle=float(choice.lam_gcv / choice.lam_oracle),
        amplification_worst=float(amp_worst),
        amplification_rms=float(amp_rms),
        oracle_at_grid_edge=bool(oracle_at_edge),
        discrepancy_found=bool(np.isfinite(choice.lam_discrepancy)),
        corner_index_spread=int(choice.detail["index_spread"]),
        corner_two_line_over_curvature=float(
            corners[1] / corners[0] if corners[0] > 0.0 else float("nan")
        ),
        corner_second_difference_over_curvature=float(
            corners[2] / corners[0] if corners[0] > 0.0 else float("nan")
        ),
        picard_slope_signal=_finite_or_none(pic.slope_coeff_signal),
        picard_slope_ratio_signal=_finite_or_none(pic.slope_ratio_signal),
        picard_slope_all=_finite_or_none(pic.slope_coeff),
        picard_correlation=_finite_or_none(pic.correlation),
        picard_n_below_noise=int(pic.n_below_noise),
        picard_first_below_noise=int(pic.first_below_noise),
        n_lam=int(lc.lam.size),
        floor=LOG_FLOOR,
        overflow=False,
    )
    fr.set_strings(
        status="ok",
        picard_class=picard,
        spectrum=spectrum,
        criterion_order=(
            "criterion_log10_lam / criterion_error 的顺序固定为 "
            "oracle（知道真解）、lcurve（最大曲率拐点）、gcv、discrepancy（偏差原则）；"
            "discrepancy 那一项在整条网格上都够不到噪声水平时写占位值 "
            f"{INVALID:g} 并把 criterion_valid 置 0——那是「没测到」，不是 0。"
        ),
        oracle_note=(
            "meta.log10_lam_oracle 用了现实中拿不到的 x*；它的唯一作用是当基准，"
            "量出其余判据的代价（meta.inflation_*）。meta.oracle_at_grid_edge 为 true 时"
            "oracle 落在 lambda 网格端点，比值类读数没有意义（Picard 不成立时就会这样）。"
        ),
        corner_note=(
            "L 曲线的拐点不是良定义的：corner_log10_lam 是按曲率、两段直线拟合交点、"
            "离散二阶差分三个判据分别测出来的，meta.corner_index_spread 是它们在网格上的"
            "下标跨度。无约束的两段直线拟合实测在 9/18 组配置上跳到别的答案"
            "（最大 126 倍），所以帧里的交点判据带「每段至少 25% 的点 + 交点在包围盒内」"
            "两条约束。"
        ),
        picard_note=(
            "picard_sigma / picard_coeff / picard_ratio 是离散 Picard 条件的实测对照："
            "Picard 成立当且仅当 log|u_i^T b| 对 log sigma_i 的斜率 > 1"
            "（即比值的斜率 < 0）。meta.picard_slope_signal 只在系数高于噪声的那一段上拟合，"
            "meta.picard_slope_all 是整段拟合——噪声接管之后整段斜率必然变平，"
            "两个数分开报就是为了不把「噪声接管」误读成「Picard 不成立」。"
        ),
        floor_note=(
            f"对数轴曲线地板 {LOG_FLOOR:g}；curvature 也抬到地板上（曲率为 0 的点在对数轴上"
            "画不出来）。picard_below_noise 标出哪些模态的系数已经落到噪声之下。"
        ),
        null_note=(
            "meta.log10_lam_discrepancy / error_discrepancy / inflation_discrepancy 为 null "
            "表示偏差原则在整条 lambda 网格上没有交点（残差始终低于目标），不是「最优 lambda = 0」。"
        ),
        svd_note=(
            "SVD 走 1.3 章的 thin 口径（A = U diag(s) V^T、s 降序）；factorize 带收敛与"
            "重建体检，单边 Jacobi 停滞时自动回退到双边/Golub-Kahan（见 numcore.inverse.regularize）。"
        ),
        data_source="backend",
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 2：滤波因子
# --------------------------------------------------------------------------- #
def _filter_frame(params: Mapping[str, Any], t: float) -> Frame:
    del t
    m, n = 32, 20
    kappa = _pow10(params["log10_kappa"])
    noise_rel = _pow10(params["log10_noise"])
    lam = _pow10(params["log10_lam"])
    k_tsvd = int(params["k_tsvd"])
    n_iter = int(params["n_iter"])
    omega_rel = float(params["omega_rel"])
    tv_probe = str(params["tv_probe"])

    prob = test_problem(m, n, kappa, picard="satisfied", noise_rel=noise_rel)
    s = prob.s
    omega = omega_rel / float(s[0] ** 2)

    f_tikh = tikhonov_factors(s, lam)
    f_tsvd = tsvd_factors(s, k_tsvd)
    f_lw_raw = landweber_factors(s, n_iter, omega)
    # 掩码必须由**原始**因子决定：先夹再判会把越界信息丢掉（开发时踩过这个坑，
    # 于是 omega_rel=2.5 的帧里 landweber_diverges=True 而掩码全是 1）。
    lw_ok = np.isfinite(f_lw_raw) & (f_lw_raw >= 0.0) & (f_lw_raw <= 1.0)
    f_lw = np.where(lw_ok, f_lw_raw, np.clip(f_lw_raw, 0.0, 1.0))
    f_lw = np.where(np.isfinite(f_lw), f_lw, 0.0)
    diverges = bool(np.any(~lw_ok))

    # TV 探针：逐个模态用 b = u_i 测"有效滤波因子"，再给一个混合探针做对照
    probe = tv_filter_probe(
        prob.A, prob.U, s, prob.V, lam, probe_index=list(range(n)), n_iter=120
    )
    f_tv = np.where(np.isfinite(probe["filter_diag"]), probe["filter_diag"], 0.0)
    f_tv_mix = np.where(np.isfinite(probe["filter_diag_mixed"]), probe["filter_diag_mixed"], 0.0)
    probe_shift = float(np.max(np.abs(f_tv_mix - f_tv))) if f_tv.size else 0.0

    # 三个方法的解与真误差（同一个 b、同一个噪声实现）
    sol_t = tikhonov(prob.U, s, prob.V, prob.b, lam, x_true=prob.x_true)
    sol_s = filtered_solution(prob.U, s, prob.V, prob.b, f_tsvd, method="tsvd", n_iter=k_tsvd, x_true=prob.x_true)
    f_lw_solve = np.clip(f_lw_raw, 0.0, 1.0)
    sol_l = filtered_solution(prob.U, s, prob.V, prob.b, f_lw_solve, method="landweber", n_iter=n_iter, x_true=prob.x_true)

    # 误差-自由度曲线：三条方法在同一条自由度轴上的公平比较
    dof_tsvd = np.zeros(n, dtype=float)
    err_tsvd = np.zeros(n, dtype=float)
    for k in range(1, n + 1):
        sol = filtered_solution(prob.U, s, prob.V, prob.b, tsvd_factors(s, k), method="tsvd", x_true=prob.x_true)
        dof_tsvd[k - 1] = float(k)
        err_tsvd[k - 1] = sol.error_rel

    n_lam = 61
    lam_sweep = np.logspace(
        np.log10(max(s[-1], np.finfo(float).tiny)) - 1.0, np.log10(float(s[0])) + 1.0, n_lam
    )
    dof_tikh = np.zeros(n_lam, dtype=float)
    err_tikh = np.zeros(n_lam, dtype=float)
    lam_log = np.zeros(n_lam, dtype=float)
    for i, l in enumerate(lam_sweep):
        f = tikhonov_factors(s, float(l))
        sol = filtered_solution(prob.U, s, prob.V, prob.b, f, method="tikhonov", x_true=prob.x_true)
        dof_tikh[i] = float(np.sum(f))
        err_tikh[i] = sol.error_rel
        lam_log[i] = float(math.log10(l))

    n_lw = max(n_iter, 1)
    dof_lw = np.zeros(n_lw, dtype=float)
    err_lw = np.zeros(n_lw, dtype=float)
    for k in range(1, n_lw + 1):
        f = np.clip(landweber_factors(s, k, omega), 0.0, 1.0)
        sol = filtered_solution(prob.U, s, prob.V, prob.b, f, method="landweber", x_true=prob.x_true)
        dof_lw[k - 1] = float(np.sum(f))
        err_lw[k - 1] = sol.error_rel

    # 同样自由度下谁更准：把 TSVD 的 k 当基准自由度
    target = float(k_tsvd)
    i_t = int(np.argmin(np.abs(dof_tikh - target)))
    i_l = int(np.argmin(np.abs(dof_lw - target)))
    errs_matched = {
        "tsvd": float(err_tsvd[max(k_tsvd - 1, 0)]),
        "tikhonov": float(err_tikh[i_t]),
        "landweber": float(err_lw[i_l]),
    }
    winner = min(errs_matched, key=lambda kk: errs_matched[kk])

    fr = Frame(chapter="5.2", viz="filter_shapes", kind="frame")
    fr.add("index", np.arange(1, n + 1, dtype=float), unit="", label="模态下标 i")
    fr.add("log10_sigma", np.log10(_floored(s, LOG_FLOOR)), unit="", label="lg(sigma_i)")
    fr.add("filter_tsvd", f_tsvd, unit="", label=f"TSVD 滤波因子（k={k_tsvd}）")
    fr.add("filter_tikhonov", f_tikh, unit="", label=f"Tikhonov 滤波因子（lambda={lam:g}）")
    fr.add("filter_landweber", f_lw, unit="", label=f"Landweber 滤波因子（k={n_iter}）")
    fr.add("filter_landweber_valid", lw_ok.astype(float), unit="", label="Landweber 掩码（0 = 因子越界，值被夹到 0/1）")
    fr.add("filter_tv_probe", f_tv, unit="", label="TV 探针读数（b = u_i，**不是**滤波因子）")
    fr.add("filter_tv_probe_mixed", f_tv_mix, unit="", label="TV 混合探针（b = (u_1+u_n)/sqrt2）")
    fr.add("dof_tsvd_curve", dof_tsvd, unit="", label="TSVD：自由度")
    fr.add("error_tsvd_curve", _floored(err_tsvd, LOG_FLOOR), unit="", label="TSVD：真误差")
    fr.add("dof_tikhonov_curve", dof_tikh, unit="", label="Tikhonov：自由度")
    fr.add("error_tikhonov_curve", _floored(err_tikh, LOG_FLOOR), unit="", label="Tikhonov：真误差")
    fr.add("log10_lam_curve", lam_log, unit="", label="Tikhonov：lg(lambda)")
    fr.add("dof_landweber_curve", dof_lw, unit="", label="Landweber：自由度")
    fr.add("error_landweber_curve", _floored(err_lw, LOG_FLOOR), unit="", label="Landweber：真误差")
    fr.add("error_matched", np.asarray([errs_matched["tsvd"], errs_matched["tikhonov"], errs_matched["landweber"]]), unit="", label="同自由度下的误差（TSVD/Tikhonov/Landweber）")

    fr.set_meta(
        kappa_measured=float(prob.detail["kappa_measured"]),
        noise_rel=float(prob.noise_rel),
        lam=float(lam),
        k_tsvd=int(k_tsvd),
        n_iter=int(n_iter),
        omega=float(omega),
        omega_rel=float(omega_rel),
        omega_divergence_threshold_rel=2.0,
        landweber_diverges=bool(diverges),
        landweber_max_abs_factor=_finite_or_none(np.max(np.abs(f_lw_raw))) if f_lw_raw.size else None,
        dof_tikhonov=float(effective_dof(f_tikh)),
        dof_tsvd=float(effective_dof(f_tsvd)),
        dof_landweber=float(effective_dof(np.clip(f_lw_raw, 0.0, 1.0))),
        error_tikhonov=float(sol_t.error_rel),
        error_tsvd=float(sol_s.error_rel),
        error_landweber=float(sol_l.error_rel),
        matched_dof=float(target),
        error_matched_tsvd=float(errs_matched["tsvd"]),
        error_matched_tikhonov=float(errs_matched["tikhonov"]),
        error_matched_landweber=float(errs_matched["landweber"]),
        matched_landweber_iter=float(i_l + 1),
        matched_tikhonov_log10_lam=float(lam_log[i_t]),
        tv_probe_dependence=float(probe_shift),
        tv_probe_max_leakage=(
            _finite_or_none(np.nanmax(probe["leakage"]))
            if np.any(np.isfinite(probe["leakage"]))
            else None
        ),
        half_power_index_tikhonov=float(np.argmin(np.abs(f_tikh - 0.5)) + 1),
        half_power_index_landweber=float(np.argmin(np.abs(f_lw - 0.5)) + 1),
        n_modes=int(n),
        floor=LOG_FLOOR,
        overflow=False,
    )
    fr.set_strings(
        status="ok",
        winner=winner,
        winner_note=(
            "meta.matched_dof 是三条误差曲线的公共横坐标（取 TSVD 的 k）；"
            "在这一点上最准的方法写进 strings.winner，三个数在 meta.error_matched_* 里。"
            "「同样自由度」是滤波因子形状之外**必须**补上的一刀：形状不同不等于精度不同。"
        ),
        tv_note=(
            f"TV 不是谱滤波：filter_tv_probe 是用 b = u_i 逐个探出来的有效读数，"
            f"filter_tv_probe_mixed 换成 b = (u_1+u_n)/sqrt(2) 之后同一个下标的读数最大改变 "
            f"{probe_shift:.3f}（meta.tv_probe_dependence），非对角泄漏 "
            f"{float(np.nanmax(probe['leakage'])):.3f}（meta.tv_probe_max_leakage）。"
            "把这条曲线当成 Tikhonov 那样的滤波器就是错的。"
        ),
        omega_note=(
            "omega 是 Landweber 的步长，omega_rel = omega·sigma_1^2：1 是最快收敛点、"
            "2 是发散线。meta.landweber_diverges 为 true 时滤波因子已经跑出 [0,1]"
            "（meta.landweber_max_abs_factor 给出最大绝对值，实测 omega_rel=2.5 时约 1.9e5）；"
            "filter_landweber 里那一段是被夹到边界、并用 filter_landweber_valid 掩码标成 0 的占位值。"
        ),
        dof_note=(
            "有效自由度 sum_i f_i：TSVD 就是 k，Tikhonov 由 lambda 决定，Landweber 由迭代数决定。"
            "误差曲线误差的横轴就是这个量——不是 lambda、不是 k，否则三条曲线根本不可比。"
        ),
        floor_note=f"误差曲线与 lg(sigma) 的地板 {LOG_FLOOR:g}（误差为 0 的点写地板值）。",
        data_source="backend",
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 3：图像反卷积 / 简化 CT
# --------------------------------------------------------------------------- #
def _operator_setup(op: str, blur_sigma: float, n_angles: int) -> dict[str, Any]:
    """按算子类型准备：图像尺寸、正/伴随作用、SVD、条件数。

    两条路的规模不同（反卷积 24x24、CT 10x10），因为 CT 的投影矩阵是**显式**存下来的，
    尺寸直接决定每帧代价；两者都写进 meta 与 strings，不做"看起来一样大"的假象。
    """
    if op == "blur":
        n = 24
        img = phantom_image(n)
        B = spatial_blur_matrix(n, float(blur_sigma))
        fit = factorize(B)
        kappa1 = float(fit.s[0] / fit.s[-1]) if fit.s[-1] > 0.0 else float("inf")

        def forward(Z: np.ndarray) -> np.ndarray:
            return separable_blur_forward(Z, B)

        def adjoint(Z: np.ndarray) -> np.ndarray:
            return separable_blur_adjoint(Z, B)

        return {
            "n": n, "img": img, "fit": fit, "kappa": kappa1 * kappa1, "kappa1d": kappa1,
            "forward": forward, "adjoint": adjoint, "norm_A": float(fit.s[0] ** 2),
            "shape": (n, n), "null_dim": 0, "operator": op,
        }
    if op == "ct":
        n = 10
        na = int(n_angles)
        img = phantom_image(n)
        A = radon_matrix(n, na, n_bins=na + 2)
        fit = factorize(A)
        kappa = float(fit.s[0] / fit.s[-1]) if fit.s[-1] > 0.0 else float("inf")
        rank = int(np.count_nonzero(fit.s > fit.s[0] * 1e-12))

        def forward(Z: np.ndarray) -> np.ndarray:
            return A @ np.asarray(Z, dtype=float).ravel()

        def adjoint(Z: np.ndarray) -> np.ndarray:
            return (A.T @ np.asarray(Z, dtype=float).ravel()).reshape(n, n)

        return {
            "n": n, "img": img, "fit": fit, "kappa": kappa, "kappa1d": float("nan"),
            "forward": forward, "adjoint": adjoint, "norm_A": float(fit.s[0]),
            "shape": (n, n), "null_dim": int(n * n - rank), "operator": op, "A": A,
        }
    raise KeyError(f"未知的算子 {op!r}（blur / ct）")


def _image_frame(params: Mapping[str, Any], t: float) -> Frame:
    op = str(params["operator"])
    blur_sigma = float(params["blur_sigma"])
    n_angles = int(params["n_angles"])
    noise_rel = _pow10(params["log10_noise"])
    lam = _pow10(params["log10_lam"])
    method = str(params["method"])
    n_iter = int(params["n_iter"])
    k_now = _step_index(t, n_iter)

    setup = _operator_setup(op, blur_sigma, n_angles)
    n = int(setup["n"])
    img = np.asarray(setup["img"], dtype=float)
    fit = setup["fit"]
    forward: Callable[[np.ndarray], np.ndarray] = setup["forward"]
    adjoint: Callable[[np.ndarray], np.ndarray] = setup["adjoint"]

    data_clean = forward(img)
    gen = _rng(20330502)
    e = gen.standard_normal(np.asarray(data_clean).shape)
    e = e / (vector_norm(e) or 1.0) * float(noise_rel) * (vector_norm(data_clean) or 1.0)
    data = data_clean + e

    s = fit.s
    noise_abs = float(vector_norm(e))
    # 可分离模糊的**二维**奇异值是 s_i s_j（24x24 = 576 个模态），"被噪声接管"要按二维算
    s_modes = np.outer(s, s).ravel() if op == "blur" else np.asarray(s, dtype=float)
    n_below = noise_dominated_count(s_modes, noise_abs, b_norm=float(vector_norm(data)))

    # --- 当前解 ---
    if method == "tikhonov":
        if op == "blur":
            recon = separable_tikhonov(data, fit.U, fit.s, fit.V, lam)
        else:
            factors = tikhonov_factors(s, lam)
            recon = (np.asarray(fit.V, dtype=float) @ (
                factors * (np.asarray(fit.U, dtype=float).T @ np.asarray(data, dtype=float).ravel())
                / np.where(s > 0.0, s, 1.0)
            )).reshape(n, n)
        hist_kind = "truncation_order"
    elif method == "tsvd":
        if op == "blur":
            recon = separable_tsvd(data, fit.U, fit.s, fit.V, lam)
        else:
            factors = (s > float(lam)).astype(float)
            recon = (np.asarray(fit.V, dtype=float) @ (
                factors * (np.asarray(fit.U, dtype=float).T @ np.asarray(data, dtype=float).ravel())
                / np.where(s > 0.0, s, 1.0)
            )).reshape(n, n)
        hist_kind = "truncation_order"
    elif method == "landweber":
        run = landweber_operator(
            forward, adjoint, data, np.zeros(setup["shape"], dtype=float),
            n_iter=k_now, omega=1.0 / float(setup["norm_A"]) ** 2, x_true=img,
        )
        recon = run.x
        hist_kind = "iteration"
    elif method == "tv":
        run = tv_solve(
            forward, adjoint, data, np.zeros(setup["shape"], dtype=float), lam,
            n_iter=k_now, norm_A=float(setup["norm_A"]), x_true=img,
        )
        recon = run.x
        hist_kind = "iteration"
    else:
        raise KeyError(f"未知的方法 {method!r}")

    # --- 误差随迭代步（landweber/tv）或截断阶 k（谱方法）的历史 ---
    k_axis = np.arange(n_iter + 1, dtype=float)
    if method in ("landweber", "tv"):
        err_hist = _pad_tail(run.error_rel, n_iter + 1, fill=1.0)
        res_hist = _pad_tail(run.residual_rel if method == "landweber" else np.zeros(0), n_iter + 1, fill=0.0)
    else:
        # 谱方法的"截断阶 k 族"：一维方向保留前 k 个模态（二维取张量积 s_i s_j > 0 的前 k 阶），
        # 对可分离模糊用 separable_apply_factors，对 CT 用显式因子——两条路是同一个数学对象
        nm = int(min(int(s.size), n_iter))
        idx = np.arange(int(s.size))
        err_trunc = np.zeros(nm + 1, dtype=float)
        err_trunc[0] = relative_error_norm(np.zeros(n * n), img.ravel())
        for k in range(1, nm + 1):
            mask = (idx < k).astype(float)
            if op == "blur":
                xk = separable_apply_factors(data, fit.U, fit.s, fit.V, np.outer(mask, mask))
            else:
                f = mask
                xk = (np.asarray(fit.V, dtype=float) @ (
                    f * (np.asarray(fit.U, dtype=float).T @ np.asarray(data, dtype=float).ravel())
                    / np.where(s > 0.0, s, 1.0)
                )).reshape(n, n)
            err_trunc[k] = relative_error_norm(xk, img)
        err_hist = _pad_tail(err_trunc, n_iter + 1, fill=float(err_trunc[-1]))
        res_hist = np.zeros(n_iter + 1, dtype=float)

    # --- lambda 扫描（误差何时被噪声吞掉）---
    if method in ("tikhonov", "tsvd"):
        lam_sweep = np.logspace(-6.0, 1.0, 41)
        err_sweep = np.zeros(lam_sweep.size, dtype=float)
        for i, l in enumerate(lam_sweep):
            if op == "blur":
                xl = separable_tikhonov(data, fit.U, fit.s, fit.V, float(l)) if method == "tikhonov" else separable_tsvd(data, fit.U, fit.s, fit.V, float(l))
            else:
                f = tikhonov_factors(s, float(l)) if method == "tikhonov" else (s > float(l)).astype(float)
                xl = (np.asarray(fit.V, dtype=float) @ (
                    f * (np.asarray(fit.U, dtype=float).T @ np.asarray(data, dtype=float).ravel())
                    / np.where(s > 0.0, s, 1.0)
                )).reshape(n, n)
            err_sweep[i] = relative_error_norm(xl, img)
    else:
        lam_sweep = np.zeros(0, dtype=float)
        err_sweep = np.zeros(0, dtype=float)

    fr = Frame(chapter="5.2", viz="image_deconvolution", kind="frame")
    fr.add("image_true", img, unit="", label="幻影 x*")
    fr.add("image_data", np.asarray(data, dtype=float), unit="", label="模糊 + 噪声的数据 b")
    fr.add("image_recon", np.asarray(recon, dtype=float), unit="", label=f"{method} 重建")
    fr.add("image_abs_error", np.abs(np.asarray(recon, dtype=float) - img), unit="", label="|重建 - 真解|")
    fr.add("image_residual", np.asarray(data, dtype=float) - forward(np.asarray(recon, dtype=float)), unit="", label="数据残差")
    fr.add("history_k", k_axis, unit="", label="迭代步 / 截断阶 k")
    fr.add("error_history", _floored(err_hist, LOG_FLOOR), unit="", label="相对误差历史")
    fr.add("residual_history", _floored(res_hist, LOG_FLOOR), unit="", label="相对残差历史（迭代法）")
    fr.add("log10_sigma", np.log10(_floored(s, LOG_FLOOR)), unit="", label="lg(sigma_i)")
    if lam_sweep.size:
        fr.add("log10_lam_sweep", np.log10(lam_sweep), unit="", label="lg(lambda) 扫描")
        fr.add("error_lam_sweep", _floored(err_sweep, LOG_FLOOR), unit="", label="误差随 lambda 的扫描")

    err_now = relative_error_norm(recon, img)
    hist_best = int(np.argmin(err_hist)) if err_hist.size else 0
    if lam_sweep.size:
        i_best = int(np.argmin(err_sweep))
        lam_best: float | None = float(lam_sweep[i_best])
        err_best: float | None = float(err_sweep[i_best])
    else:
        lam_best, err_best = None, None
    fr.set_meta(
        n_image=int(n),
        blur_sigma=float(blur_sigma),
        n_angles=int(n_angles),
        kappa=_finite_or_none(setup["kappa"]),
        kappa_1d=_finite_or_none(setup["kappa1d"]),
        n_modes=int(s_modes.size),
        rank_measured=int(np.count_nonzero(s_modes > s_modes[0] * 1e-12)),
        null_space_dim=int(setup["null_dim"]),
        noise_rel=float(noise_rel),
        noise_abs=float(noise_abs),
        data_norm=float(vector_norm(data)),
        lam=float(lam),
        log10_lam=float(params["log10_lam"]),
        k_current=int(k_now),
        error_rel_current=float(err_now),
        residual_rel_current=float(vector_norm(np.asarray(data, dtype=float) - forward(np.asarray(recon, dtype=float))) / (vector_norm(data) or 1.0)),
        error_rel_best_lambda=_finite_or_none(err_best),
        lam_best=_finite_or_none(lam_best),
        noise_dominated_modes=int(n_below),
        error_rel_at_k_best=_finite_or_none(err_hist[hist_best]) if err_hist.size else None,
        k_best=int(hist_best),
        semi_convergence=bool(method in ("landweber", "tv") and 0 < hist_best < err_hist.size - 1),
        svd_fallback=float(fit.detail.get("fallback", 0.0)),
        tv_objective_final=_finite_or_none(run.objective[-1]) if method == "tv" else None,
        floor=LOG_FLOOR,
        overflow=False,
    )
    fr.set_strings(
        status="ok",
        operator=op,
        method=method,
        history_kind=hist_kind,
        history_note=(
            "history_kind=iteration（landweber/tv）时横轴是迭代步，误差先降后升就是**半收敛**，"
            "meta.k_best 给出实测最优的早停步；history_kind=truncation_order（谱方法）时"
            "横轴是截断阶 k，画的是同一个算子的谱截断族的误差。两种横轴含义不同，"
            "图注必须跟着 history_kind 走。"
        ),
        noise_note=(
            "meta.noise_dominated_modes 是奇异值已经落到 noise_abs/||b|| 之下的模态个数："
            "从那里开始，数据里剩下的全是噪声，任何正则化都只能把它们压小、不能恢复。"
        ),
        method_note=(
            "method=tikhonov/tsvd 时 lambda 分别是滤波参数与截断阈值（同一个滑块，口径写在 "
            "numcore.inverse.regularize 的模块 docstring 里）；landweber/tv 用迭代步数控制"
            "正则化强度，lambda 只对 tv 起作用。"
        ),
        null_space_note=(
            "简化 CT 的角度够多时这个尺寸的投影矩阵是**良态**的（κ 只有几百），"
            "角度减到 8 条时 A 秩亏：meta.null_space_dim > 0 表示误差地板由零空间决定，"
            "把噪声拖到 0 误差也不会降——那不是噪声问题。"
        ),
        floor_note=f"误差历史与 lg(sigma) 的地板 {LOG_FLOOR:g}；误差为 0 的点写地板值。",
        svd_note="factorize 带收敛/重建体检（单边 Jacobi 在重奇异值上会停滞），meta.svd_fallback=1 表示回退过。",
        data_source="backend",
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：偏差-方差分解
# --------------------------------------------------------------------------- #
def _bias_variance_frame(params: Mapping[str, Any], t: float) -> Frame:
    del t
    m, n = 32, 20
    kappa = _pow10(params["log10_kappa"])
    noise_rel = _pow10(params["log10_noise"])
    lam_cur = _pow10(params["log10_lam"])
    n_trials = int(params["n_trials"])
    picard = str(params["picard"])

    prob = test_problem(m, n, kappa, picard=picard, noise_rel=noise_rel)
    lam = np.logspace(-4.0, 4.0, 61)
    ana = bias_variance_analytic(prob.U, prob.s, prob.V, prob.b_exact, prob.x_true, prob.noise_abs, lam)
    mc = bias_variance_montecarlo(
        prob.U, prob.s, prob.V, prob.b_exact, prob.x_true, prob.noise_abs, lam,
        n_trials=n_trials, seed=20261018,
    )
    lam_oracle = lambda_oracle(prob.U, prob.s, prob.V, prob.b, prob.x_true, lam)
    sol_cur = tikhonov(prob.U, prob.s, prob.V, prob.b, lam_cur, x_true=prob.x_true)

    # 方差两侧的对数-对数斜率（理论 0 与 -4）：只在地板之上拟合
    lo_mask = lam <= float(prob.s[-1])
    hi_mask = lam >= float(prob.s[0]) * 10.0
    slope_lo = _loglog_slope(lam[lo_mask], ana.variance[lo_mask], skip=0)
    slope_hi = _loglog_slope(lam[hi_mask], ana.variance[hi_mask], skip=0)
    # 逐点局部斜率（给副轴），用中心差分。方差与 MSE 是两条不同的曲线：
    # 副轴的标题承诺的是**方差**的斜率（理论 0 与 -4 两段），不能拿 MSE 的斜率冒充。
    xl = np.log10(lam)

    def _local_log_slope(y: np.ndarray) -> np.ndarray:
        yl = np.log10(np.maximum(np.asarray(y, dtype=float), np.finfo(float).tiny))
        out = np.zeros(lam.size, dtype=float)
        for i in range(1, lam.size - 1):
            out[i] = (yl[i + 1] - yl[i - 1]) / (xl[i + 1] - xl[i - 1])
        out[0], out[-1] = out[1], out[-2]
        return out

    loc = _local_log_slope(ana.mse)
    loc_var = _local_log_slope(ana.variance)

    fr = Frame(chapter="5.2", viz="bias_variance", kind="frame")
    fr.add("log10_lam", np.log10(lam), unit="", label="lg(lambda)")
    fr.add("bias2_analytic", _floored(ana.bias2, LOG_FLOOR), unit="", label="偏差^2（解析）")
    fr.add("variance_analytic", _floored(ana.variance, LOG_FLOOR), unit="", label="方差（解析）")
    fr.add("mse_analytic", _floored(ana.mse, LOG_FLOOR), unit="", label="偏差^2+方差（解析）")
    fr.add("bias2_mc", _floored(mc.bias2_mc, LOG_FLOOR), unit="", label="偏差^2（蒙特卡洛实测）")
    fr.add("variance_mc", _floored(mc.variance_mc, LOG_FLOOR), unit="", label="方差（蒙特卡洛实测）")
    fr.add("mse_mc", _floored(mc.mse_mc, LOG_FLOOR), unit="", label="总误差（蒙特卡洛实测）")
    fr.add("mse_mc_std", _floored(mc.mse_mc_std, LOG_FLOOR), unit="", label="总误差的标准误")
    fr.add("error_rel_analytic", _floored(np.sqrt(ana.mse), LOG_FLOOR), unit="", label="期望相对误差（解析）")
    fr.add("error_rel_mc", _floored(np.sqrt(mc.mse_mc), LOG_FLOOR), unit="", label="实测相对误差（蒙特卡洛）")
    fr.add("dof", ana.dof, unit="", label="有效自由度")
    fr.add("local_slope", loc, unit="", label="d lg(mse) / d lg(lambda)")
    fr.add("local_slope_variance", loc_var, unit="", label="d lg(variance) / d lg(lambda)")

    err_lo = float(np.sqrt(ana.mse[0]))
    err_hi = float(np.sqrt(ana.mse[-1]))
    fr.set_meta(
        kappa_measured=float(prob.detail["kappa_measured"]),
        noise_rel=float(prob.noise_rel),
        noise_abs=float(prob.noise_abs),
        n_trials=int(n_trials),
        lam_current=float(lam_cur),
        error_current=float(sol_cur.error_rel),
        dof_current=float(sol_cur.dof),
        log10_lam_oracle=float(math.log10(lam_oracle)),
        log10_lam_opt_total=float(math.log10(ana.lam_argmin)),
        log10_lam_opt_mc=float(math.log10(mc.lam_argmin)),
        mse_min=float(ana.mse_min),
        error_min=float(np.sqrt(ana.mse_min)),
        error_at_lam_min=err_lo,
        error_at_lam_max=err_hi,
        variance_slope_low=_finite_or_none(slope_lo),
        variance_slope_high=_finite_or_none(slope_hi),
        bias2_at_lam_max=float(ana.bias2[-1]),
        variance_at_lam_min=float(ana.variance[0]),
        variance_over_mse_at_lam_min=float(ana.variance[0] / max(ana.mse[0], np.finfo(float).tiny)),
        bias2_over_mse_at_lam_max=float(ana.bias2[-1] / max(ana.mse[-1], np.finfo(float).tiny)),
        mc_vs_analytic_max_rel=_finite_or_none(
            np.max(np.abs(mc.mse_mc - ana.mse) / np.maximum(ana.mse, np.finfo(float).tiny))
        ),
        mc_bias_max_abs_over_mse=_finite_or_none(
            np.max(np.abs(mc.bias2_mc - ana.bias2)) / max(ana.mse_min, np.finfo(float).tiny)
        ),
        mc_mse_max_rel_near_min=_finite_or_none(
            np.max(
                np.abs(mc.mse_mc - ana.mse)[ana.mse <= 100.0 * ana.mse_min]
                / np.maximum(ana.mse[ana.mse <= 100.0 * ana.mse_min], np.finfo(float).tiny)
            )
        ),
        mc_variance_max_rel=_finite_or_none(
            np.max(np.abs(mc.variance_mc - ana.variance) / np.maximum(ana.variance, np.finfo(float).tiny))
        ),
        floor=LOG_FLOOR,
        overflow=False,
    )
    fr.set_strings(
        status="ok",
        picard_class=picard,
        decomposition_note=(
            "bias2/variance/mse 三条解析曲线与 bias2_mc/variance_mc/mse_mc 三条蒙特卡洛曲线"
            "是同一批噪声样本上的两个口径：偏差只与无噪的 b_exact 有关，"
            "把带噪的 b 代进偏差里量到的是噪声不是偏差。"
            "meta.mc_vs_analytic_max_rel / mc_variance_max_rel 是最大相对差，"
            "meta.mc_mse_max_rel_near_min 只在小 lambda 最小点附近 100 倍以内比较；"
            "偏差^2 在 lambda -> 0 时本身趋于 0，用相对差比会得到 1e6 量级的假差异"
            "（实测就是把一个 1e-9 的真值拿去当分母），所以偏差用**绝对差对总误差的比**"
            "（meta.mc_bias_max_abs_over_mse）来量。"
        ),
        oracle_note=(
            "meta.log10_lam_oracle 与 log10_lam_opt_total 都是**知道真解**才能算的量；"
            "曲线上的最低点不是可以从数据里算出来的东西——这正是 GCV/L 曲线存在的理由。"
        ),
        tail_note=(
            "方差在两侧的对数-对数斜率：小 lambda 段（lambda <= sigma_n）理论 0（方差饱和到"
            " (delta^2/m) sum 1/s_i^2），大 lambda 段（lambda >= 10 sigma_1）理论 -4"
            "（f_i ~ s_i^2/lambda^2），实测写在 meta.variance_slope_low / _high。"
        ),
        edge_note=(
            "meta.error_at_lam_min / error_at_lam_max 是两侧的失败方式："
            "lambda->0 时误差由噪声接管、lambda->∞ 时误差回到 ||x*||（解被压成 0）。"
            "两个数都接近 1 甚至更大时，正则在中间那一段才有意义。"
        ),
        floor_note=f"所有平方量曲线抬到地板 {LOG_FLOOR:g}（偏差^2可以为 0，对数轴上画不出来）。",
        data_source="backend",
    )
    return fr
