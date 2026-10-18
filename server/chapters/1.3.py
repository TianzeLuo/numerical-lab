"""1.3 章：最小二乘与正交化（QR / SVD / 正规方程）。

四个可视化的分工是“从几何到误差、再到不确定度”：

1. ``projection_3d`` —— 几何图像：``b`` 投影到 ``A`` 的列空间，残差 ``r`` 与列空间**垂直**。
   副轴给出 ``‖Aᵀr‖/(‖A‖_F‖r‖)`` 的实测值——“肉眼可见的直角”必须有一个数字背书。
   主场景的时间轴是**分解动画进度**（``b = p_t + r_t``），不是物理时间。
2. ``normal_vs_qr`` —— 误差图像：``kappa`` 从 ``1e2`` 扫到 ``1e12``（对数轴），
   正规方程的误差 ``~kappa^2 u`` 与 QR/SVD 的 ``~kappa u`` 在双对数图上分道扬镳；
   正交化方法只把误差放大一次，这就是本章的全部理由。时间轴是当前扫描位置。
3. ``svd_truncation`` —— 谱与截断：拖截断阶 ``k`` 看解范数与残差的权衡（L 曲线），
   同时把奇异值谱、Picard 量 ``|u_iᵀb|/sigma_i`` 与最优 ``k`` 摆在一起。
   时间轴就是截断阶。
4. ``uncertainty_fan`` —— 不确定度：**等残差**的谷底上解不是唯一的；解集合的椭圆的扁度
   就是 ``kappa``。时间轴是噪声方向角，扇面随时间张开。

约定（与 1.2 章一致）：``meta`` 只放标量，字符串走 ``strings``。凡是要上对数轴的量
（误差、残差、正交性、奇异值）都取过地板 ``u`` 或 ``tiny``，并在 ``meta`` 里给出地板值——
``log(0)`` 不是“很小”，是“没有定义”。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from numcore.linalg.lstsq import (
    condition_estimate_2,
    condition_sweep,
    least_squares_normal,
    least_squares_qr,
    least_squares_svd,
    noise_fan,
    projection_blend,
    projection_matrix_apply,
    rotation_matrix_3d,
    synthetic_lstsq_problem,
    synthetic_truncation_problem,
    truncation_sweep,
    uncertainty_ellipse,
)
from numcore.linalg.qr import householder_qr
from numcore.linalg.svd import jacobi_svd, numerical_rank
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

_U = float(np.finfo(float).eps) / 2.0  # 单位舍入 u = 2^-53
_LOG_FLOOR = 1.0e-18  # 对数轴地板：比 u 还小两个数量级，足够容纳“精确 0"
_MAX_STEPS = 64  # 时间轴上限（= svd_truncation 的 n 上限 / 各 viz 的步数上限）

MANIFEST: dict[str, Any] = {
    "chapter": "1.3",
    "title": "最小二乘与正交化",
    "part": "I",
    "summary": (
        "投影几何、正交变换、奇异值谱与不确定度：把「解一个超定方程组」拆成"
        "「垂直」「放大一次还是平方次」「截断到哪里」三件事，每件事都给出实测数字。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "projection_3d",
            "title": "三维投影：残差为什么必须垂直于列空间",
            "description": (
                "A 的两列张成 R³ 里的一个平面，b 投影上去得到 p，r = b − p。"
                "时间轴把 b = p + r 逐步拆开（t=0 只看得到 b，t=1 两个分量完全分开）；"
                "副轴给出三种解法实测的 ‖Aᵀr‖/(‖A‖_F‖r‖)——直角不能只靠眼睛看。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "b_theta",
                    "label": "b 的方位角 θ（度）",
                    "type": "float",
                    "min": 0.0,
                    "max": 360.0,
                    "step": 5.0,
                    "default": 40.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "b_phi",
                    "label": "b 的天顶角 φ（度，从 +z 量起）",
                    "type": "float",
                    "min": 0.0,
                    "max": 180.0,
                    "step": 5.0,
                    "default": 55.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "log10_kappa",
                    "label": "A 的条件数 log10 κ",
                    "type": "float",
                    "min": 0.0,
                    "max": 12.0,
                    "step": 0.25,
                    "default": 3.0,
                    "critical": [15.95],
                    "criticalNote": (
                        "κ·u = 1 的界线（log10(1/u) ≈ 15.95）：超过它以后，A 的小奇异方向在"
                        "float64 里已经不存在，列空间在数值上“扁”成一条线，画出来的平面是幻觉。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "method",
                    "label": "求解路线（用于副轴对照）",
                    "type": "enum",
                    "options": ["qr", "normal", "svd"],
                    "default": "qr",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.05, "label": "分解动画 t"},
            "requiredAux": ["residual", "error"],
            "axes": [
                {
                    "id": "scene",
                    "autoscaled": False,
                    "title": "三维场景（三轴等比例、固定 ±1.8）",
                    "ylabel": "坐标",
                    "scale": "linear",
                },
                {
                    "id": "orthogonality",
                    "autoscaled": True,
                    "title": "三种解法的正交性缺陷 ‖Aᵀr̂‖/(‖A‖_F‖r̂‖)",
                    "ylabel": "无量纲",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "陷阱一：**坐标轴不等比例时直角会看起来不是直角**。三维投影图的默认视图会把三个方向"
                "各自拉伸，于是“r 与平面垂直”这件事在屏幕上完全看不出来——所以本帧把主轴声明为"
                "autoscaled=false（固定 ±1.8、等比例），视图之间不能各自缩放。"
                "陷阱二：**残差小不等于解准**。把 κ 拖到 1e8 以上，配一个几乎落在列空间里的 b，"
                "残差会小到 1e-16，但此时 A 的两个奇异方向已经挤在一起，两种完全不同的 x 给出同一个 p；"
                "副轴上 QR 与正规方程的正交性缺陷都贴着地板，而它们背后对应的解可以差到 O(1)——"
                "只看这张图会得出“这个解很可靠”的错误结论，必须配合 uncertainty_fan 一起看。"
            ),
            "theory": [
                "投影的刻画：p = A x̂ 是 b 在 col(A) 上的正交投影 ⟺ Aᵀ(b − p) = 0（正规方程）",
                "残差与解的唯一性：A 满列秩时 x̂ 唯一；秩亏时所有解只差一个 null(A) 里的向量，"
                "而残差完全相同（这正是 uncertainty_fan 画的谷底）",
                "正交变换不改 2-范数与内积：Householder/Givens 只做旋转反射，因此 ‖Qᵀr‖ = ‖r‖，"
                "QR 解最小二乘不会引入放大",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "normal_vs_qr",
            "title": "精度对决：正规方程的 κ² 与 QR 的 κ",
            "description": (
                "同一批构造问题（κ 从 1e2 到 1e12）上跑三条路线：正规方程、QR、SVD，"
                "双对数画前向误差并叠加 κu 与 κ²u 两条理论参考线（含实测斜率）。"
                "副轴是相对残差——它一直贴着 u，与误差无关，这正是“残差小”骗人的地方。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "exp_max",
                    "label": "扫描上限 log10 κ_max",
                    "type": "float",
                    "min": 4.0,
                    "max": 14.0,
                    "step": 0.5,
                    "default": 12.0,
                    "critical": [15.95],
                    "criticalNote": (
                        "log10(1/u) ≈ 15.95 是“双精度里还能表示 1/κ 量级扰动”的尽头："
                        "κ 超过它以后任何解法的前向误差都会饱和到 O(1)，曲线不再是幂律。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "n",
                    "label": "未知数个数 n",
                    "type": "int",
                    "min": 4,
                    "max": 20,
                    "default": 10,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "m_factor",
                    "label": "超定倍数 m/n",
                    "type": "float",
                    "min": 1.5,
                    "max": 4.0,
                    "step": 0.5,
                    "default": 2.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "refinement",
                    "label": "正规方程的迭代精化步数",
                    "type": "enum",
                    "options": ["none", "one", "two"],
                    "default": "one",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 14.0, "dt": 0.25, "label": "当前 log10 κ"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "errors",
                    "autoscaled": False,
                    "title": "前向误差 vs κ（双对数、固定轴 1e-17..1e2）",
                    "ylabel": "‖x̂−x*‖/‖x*‖",
                    "scale": "log",
                },
                {
                    "id": "residual",
                    "autoscaled": True,
                    "title": "相对残差 ‖Ax̂−b‖/‖b‖（对数）",
                    "ylabel": "相对残差",
                    "scale": "log",
                },
                {
                    "id": "divergence",
                    "autoscaled": False,
                    "title": "分道扬镳：误差之比 正规方程 / QR（对数）",
                    "ylabel": "比值",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "陷阱一：**自适应坐标会把两条曲线拉平**。误差从 1e-14 走到 1e0，如果主轴的纵坐标"
                "跟着数据自动缩放，κ² 与 κ 两条曲线在屏幕上都会变成“两条平行直线”，"
                "看不出“正规方程在某个 κ 之后彻底崩掉”。本帧把主轴固定为 1e-17..1e2 并把范围写进"
                "meta，配合 overflow 标记——图上看不见的那一段，必须由数字说出来。"
                "陷阱二：**“残差更小”被当成“解更准”**。副轴上三条路线的相对残差都贴着 1e-16 的地板，"
                "看不出任何差别，可同一时刻正规方程的前向误差已经比 QR 大 10 个数量级"
                "（κ=1e8 时实测比值 ~1e8）。另外注意：迭代精化能把正规方程救回来一段，"
                "但它修的是“解方程”的误差，修不了 AᵀA 本身的病态，κ 越过 1/u 之后一样失效。"
            ),
            "theory": [
                "正规方程：kappa(AᵀA) = kappa(A)^2，前向误差 ~ kappa^2 u（与实测斜率 2 对应）",
                "QR 最小二乘：Householder 后向稳定，前向误差 ~ kappa u + kappa^2 u·eta（eta 是残差因子；"
                "相容问题时 eta ≈ 0，于是只剩 kappa u）",
                "残差恒等式：Aᵀr̂ = 0 在任何 κ 下都成立（实测 ‖Aᵀr̂‖ ≈ n u ‖A‖‖b‖），"
                "所以“残差与列空间垂直”和“解的相对误差”是两件独立的事",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "svd_truncation",
            "title": "奇异值谱与截断：L 曲线、Picard 条件与最优 k",
            "description": (
                "拖动截断阶 k（时间轴）看三件事同步变化：解范数 ‖x_k‖、残差 ‖Ax_k−b‖、"
                "以及只有已知真解时才算得出来的真实误差。主轴是 L 曲线（解范数 vs 残差，双对数），"
                "副轴是奇异值谱与 Picard 量 |u_iᵀb|/σ_i。标出 L 曲线拐点、Picard 转折点与真正最优的 k。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "spectrum",
                    "label": "奇异值谱形状",
                    "type": "enum",
                    "options": ["decay", "cliff", "plateau", "two_clusters"],
                    "default": "decay",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "log10_kappa",
                    "label": "谱的条件数 log10 κ",
                    "type": "float",
                    "min": 2.0,
                    "max": 12.0,
                    "step": 0.5,
                    "default": 8.0,
                    "critical": [15.95],
                    "criticalNote": (
                        "κ > 1/u（log10 ≈ 15.95）以后，最小的奇异值在 float64 里无法与 0 区分，"
                        "谱的尾部整个是表示误差——此时“要不要截断”已经不是算法选择而是物理限制。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "noise_exp",
                    "label": "数据噪声 log10(‖δb‖/‖b‖)",
                    "type": "float",
                    "min": -14.0,
                    "max": -2.0,
                    "step": 0.5,
                    "default": -6.0,
                    "critical": [-15.95],
                    "criticalNote": (
                        "噪声小于 u ≈ 1.1e-16 时（log10 ≈ −15.95）它在 double 里根本表示不出来："
                        "再往下拖，图上的变化全是舍入噪声，不是数据噪声。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "n",
                    "label": "未知数个数 n（= 截断阶上限）",
                    "type": "int",
                    "min": 6,
                    "max": 16,
                    "default": 12,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "oversample",
                    "label": "超定倍数 m/n",
                    "type": "int",
                    "min": 1,
                    "max": 4,
                    "default": 3,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 16.0, "dt": 1.0, "label": "截断阶 k"},
            "requiredAux": ["residual", "error"],
            "axes": [
                {
                    "id": "lcurve",
                    "autoscaled": True,
                    "title": "L 曲线：‖x_k‖ vs ‖Ax_k−b‖/‖b‖（双对数）",
                    "ylabel": "解范数 ‖x_k‖",
                    "scale": "log",
                },
                {
                    "id": "spectrum",
                    "autoscaled": True,
                    "title": "奇异值谱与 Picard 量（对数）",
                    "ylabel": "σ_i 与 |u_iᵀb| 与 |u_iᵀb|/σ_i",
                    "scale": "log",
                },
                {
                    "id": "tradeoff",
                    "autoscaled": True,
                    "title": "三个量随截断阶 k 的变化（对数）",
                    "ylabel": "‖x_k‖ / 残差 / 相对误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "陷阱一：**截断阈值取在奇异值悬崖上，解范数会突变**。把谱形状切到 cliff："
                "奇异值前半段全是 1、后半段突然掉到 1/κ，此时 k 差一个下标，‖x_k‖ 可以跳几个数量级，"
                "L 曲线上出现一段近乎竖直的线——“最优 k”在悬崖上根本不可辨。"
                "陷阱二：**L 曲线的拐点不等于最优截断**。本帧同时给出 k_corner（离散曲率最大处）、"
                "k_picard（Picard 比 |u_iᵀb|/σ_i 的转折点）与 k_opt（真实误差最小处，"
                "只有知道真解才算得出来）：四种种子里它们经常互不相等。另外，"
                "副轴上 Picard 比在小奇异值方向**重新抬头**是“噪声接管”的signature，"
                "不是“这个方向有信息”。最后：残差曲线随 k 单调下降、解范数单调上升，"
                "两条单调曲线本身不告诉你该在哪里停——停在“残差已经很小”的地方通常停得太晚。"
            ),
            "theory": [
                "截断 SVD 解 x_k = Σ_{i≤k}(u_iᵀb/σ_i)v_i：‖x_k‖ 随 k 单调不减、‖Ax_k−b‖ 随 k 单调不增"
                "（B 类不变量，本章测试钉住）",
                "Picard 条件：可解问题要求 |u_iᵀb| 比 σ_i 衰减得更快；噪声让 |u_iᵀb| 在尾部变成常数，"
                "于是 |u_iᵀb|/σ_i ~ 1/σ_i 爆炸——这就是必须截断的全部理由",
                "Eckart–Young–Mirsky：秩 k 截断是所有秩 ≤ k 矩阵中 2-范数与 Frobenius 范数意义下的最优逼近，"
                "所以“截断”不是权宜之计而是最优投影",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "uncertainty_fan",
            "title": "解的不确定扇面：等残差的谷底有多长",
            "description": (
                "在一个 2×2 相容问题上把 b 沿噪声方向转一圈：每个扰动解都有**相同**的残差"
                "（= 噪声水平），但解本身在一条很长的椭圆上跑。主场景画等残差椭圆族、"
                "噪声扇面与当前方向；副轴给出实测不确定度随 κ 的增长与 κ·δ 参考线。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "log10_kappa",
                    "label": "A 的条件数 log10 κ",
                    "type": "float",
                    "min": 0.0,
                    "max": 12.0,
                    "step": 0.25,
                    "default": 6.0,
                    "critical": [15.95],
                    "criticalNote": (
                        "κ·u = 1 的界线：越过它以后 A 在 double 里已经奇异，"
                        "等残差椭圆退化成一条无法分辨的线段，扇面宽度不再有数值意义。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "noise_exp",
                    "label": "噪声水平 log10(‖δb‖/‖b‖)",
                    "type": "float",
                    "min": -14.0,
                    "max": -2.0,
                    "step": 0.5,
                    "default": -8.0,
                    "critical": [-15.95],
                    "criticalNote": (
                        "低于 u（log10 ≈ −15.95）的噪声在 float64 里不可表示："
                        "扇面宽度会退化成舍入噪声的形状，与物理无关。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "rho_exp",
                    "label": "等残差水平 log10(ρ/‖b‖)",
                    "type": "float",
                    "min": -12.0,
                    "max": -1.0,
                    "step": 0.5,
                    "default": -3.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_levels",
                    "label": "等残差椭圆条数",
                    "type": "int",
                    "min": 2,
                    "max": 6,
                    "default": 4,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_samples",
                    "label": "噪声方向采样数",
                    "type": "int",
                    "min": 8,
                    "max": 120,
                    "default": 48,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 360.0, "dt": 5.0, "label": "噪声方向角 θ（度）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "plane",
                    "autoscaled": False,
                    "title": "解平面（固定等比例坐标，等残差椭圆族）",
                    "ylabel": "x₂",
                    "scale": "linear",
                },
                {
                    "id": "growth",
                    "autoscaled": True,
                    "title": "实测不确定度 vs κ（双对数）与 κ·δ 参考线",
                    "ylabel": "‖δx‖/‖x*‖",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "陷阱一：**残差完全相同，解却可以差 κ 倍**。扇面上每一条射线对应的解都有同一个残差"
                "（副轴那条水平线就是实测：所有方向的 ‖Ax̂−b‖/‖b‖ 都等于噪声水平），"
                "所以“用残差挑解”在这里完全失效；解的差别只能靠 ‖δx‖/‖x*‖ 看出来。"
                "严格地说，这份“相同”在浮点里只能稳到 κu‖x‖ 的舍入水平：默认参数下残差的变化实测"
                "1.7e-10，比噪声水平 1e-8 还小 60 倍，而同一时刻解已经动了 2.8e-3——"
                "残差对解的移动几乎是瞎的（meta 里同时给出 residual_spread 与 residual_roundoff）。"
                "陷阱二：**自适应坐标会把针画成圆**。等残差椭圆的长短轴之比就是 κ；"
                "如果让主场景自适应缩放（两个坐标轴各自拉伸），κ=1e6 的针会被画成一个圆，"
                "整个可视化的结论当场反转——所以本帧把主轴声明为 autoscaled=false 并给出固定范围，"
                "同时用 meta 报出真实的长短轴之比。另外提醒：图上画的椭圆是**线性化**的结果"
                "（A 不变、只有 b 扰动）；如果 A 本身有误差，放大倍数还要乘上 A 的条件数，"
                "两个误差源的贡献不能相加了事。还有一个更隐蔽的：解本身自带 κu 的相对舍入，"
                "κ 拖到 1e10 时 κu ≈ 1.1e-6 已经**大于**默认噪声 1e-8，此时扇面量的是舍入而不是噪声"
                "（实测最短半轴从 0.28δ 涨到 8.7δ）——meta 里的 resolvable 就是这条判据，"
                "它为 false 时这张图不该被用来谈“噪声放大”。"
            ),
            "theory": [
                "等残差集：‖A(x−x_LS)‖² = ρ² − ‖r_min‖²，是一族半轴为 sqrt(ρ²−‖r_min‖²)/σ_i、"
                "方向为右奇异向量 v_i 的椭圆；长短轴之比 = κ",
                "扰动界：‖δx‖/‖x‖ ≤ κ(A)·‖δb‖/‖b‖，且当 δb 恰好落在 u_min 方向时取到等号",
                "相容问题的扇面：b + δb 的最小二乘解 = x* + A⁺δb，所有解共享同一个残差 ‖δb‖"
                "（浮点下稳到 κu‖x‖ 的舍入水平）",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 计算入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    "“”章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。“”"
    if viz_id == "projection_3d":
        return _projection_3d(params, t)
    if viz_id == "normal_vs_qr":
        return _normal_vs_qr(params, t)
    if viz_id == "svd_truncation":
        return _svd_truncation(params, t)
    if viz_id == "uncertainty_fan":
        return _uncertainty_fan(params, t)
    raise KeyError(f"章节 1.3 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _floor(values: Any, floor: float = _LOG_FLOOR) -> np.ndarray:
    "“”把数组抬到对数地板上（``log(0)`` 不是“很小”，是“没有定义”）。“”"
    return np.maximum(np.asarray(values, dtype=float), float(floor))


def _scalar(value: Any, floor: float = 0.0) -> float:
    "“”转成**有限**的 float：NaN/Inf 会直接让 NLF 头编码失败，也会骗过前端的坐标轴。“”"
    v = float(value)
    if not np.isfinite(v):
        return float(floor)
    return v


def _step_of(t: float, n_steps: int) -> int:
    "“”把连续时间轴四舍五入到整数步并夹到 ``[0, n_steps]``。“”"
    k = int(np.floor(float(t) + 0.5))
    return max(0, min(k, max(0, int(n_steps))))


def _method_label(method: str) -> str:
    return {"normal": "正规方程 AᵀA", "normal_ir": "正规方程 + 精化", "qr": "QR（Householder）", "svd": "SVD 截断"}.get(
        method, method
    )


_SAFE_EXCEPTIONS = (ZeroDivisionError, ValueError, OverflowError)


def _finite_or_none(value: Any) -> float | None:
    """把 NaN/±Inf 换成 ``None``：NLF 头用 ``allow_nan=False`` 编码，塞进去整帧就废了。

    与 1.2 章 ``_finite_or_none`` 同一套处理：**"没有有限值"是一个要显示的状态**，
    不是一个要崩掉的理由。
    """
    v = float(value)
    return v if np.isfinite(v) else None


def _solve_route(name: str, A: np.ndarray, b: np.ndarray) -> Any:
    """按名字解一次最小二乘（三个路线各自独立，便于单条兜底）。"""
    if name == "normal":
        return least_squares_normal(A, b)
    if name == "qr":
        return least_squares_qr(A, b)
    return least_squares_svd(A, b)


def _solve_routes(A: np.ndarray, b: np.ndarray, order: list[str]) -> tuple[dict[str, Any], dict[str, str]]:
    """逐条路线求解；某条路线**真的**没有有限解时把它记成 ``None`` + 失败类型。

    κ 大到 ``AᵀA`` 的浮点主元恰好为 0 时，正规方程这条路线不是"算得不准"，
    而是 LU 回代直接抛 ``ZeroDivisionError``（``numcore/linalg/lu.py``）。
    这类失败必须能在帧里表达：解写成 ``None``、``meta`` 里标 ``singular``、
    ``strings`` 里说明为什么，帧照常编码、照常画（画"这里没有有限解"）。
    """
    sols: dict[str, Any] = {}
    failed: dict[str, str] = {}
    for name in order:
        try:
            sols[name] = _solve_route(name, A, b)
        except _SAFE_EXCEPTIONS as exc:
            sols[name] = None
            failed[name] = f"{type(exc).__name__}: {exc}"
    return sols, failed


# --------------------------------------------------------------------------- #
# viz 1：三维投影
# --------------------------------------------------------------------------- #
def _projection_3d(params: Mapping[str, Any], t: float) -> Frame:
    kappa = 10.0 ** float(params["log10_kappa"])
    theta = float(params["b_theta"])
    phi = float(params["b_phi"])
    method = str(params["method"])

    A, _b_dummy, _x_dummy = synthetic_lstsq_problem(3, 2, kappa, seed=20261018)
    b = rotation_matrix_3d(theta, phi)
    p = projection_matrix_apply(A, b)
    r = b - p
    b_norm = _scalar(np.sqrt(np.dot(b, b)), 1.0)
    r_norm = _scalar(np.sqrt(np.dot(r, r)))
    # b 与列空间的夹角：sin = ‖r‖/‖b‖
    angle_plane = float(np.rad2deg(np.arcsin(min(max(r_norm / b_norm, 0.0), 1.0))))

    order = ["normal", "qr", "svd"]
    sols, failed = _solve_routes(A, b, order)
    defect = np.full(len(order), _LOG_FLOOR, dtype=float)
    proj_error = np.full(len(order), _LOG_FLOOR, dtype=float)
    residual_norms = np.full(len(order), _LOG_FLOOR, dtype=float)
    method_singular = np.zeros(len(order), dtype=float)  # 1 = 这条路线在浮点下没有有限解
    a_fro = _scalar(np.sqrt(np.sum(A * A)), 1.0)
    for i, name in enumerate(order):
        sol = sols[name]
        if sol is None:
            # 地板值 + 显式标记：前端能画"这里没有有限解"，而不是把一个假的 0 当真值
            method_singular[i] = 1.0
            continue
        x_hat = sol.x
        p_hat = A @ x_hat
        r_hat = b - p_hat
        nr = _scalar(np.sqrt(np.dot(r_hat, r_hat)))
        defect[i] = float(np.sqrt(np.dot(A.T @ r_hat, A.T @ r_hat))) / (a_fro * nr) if nr > 0.0 else 0.0
        proj_error[i] = _scalar(np.sqrt(np.dot(p_hat - p, p_hat - p))) / b_norm
        residual_norms[i] = nr
    defect = _floor(defect)
    proj_error = _floor(proj_error)
    residual_norms = _floor(residual_norms)
    ok_methods = [name for name in order if sols[name] is not None]

    # 列空间的正交基（用自研 Householder QR），画一个方形平面片
    q = householder_qr(A)
    q1 = q.Q[:, 0]
    q2 = q.Q[:, 1] if q.Q.shape[1] > 1 else np.zeros(3)
    grid = np.linspace(-1.5, 1.5, 11)
    gs, gt = np.meshgrid(grid, grid)
    plane = np.zeros((3, grid.size, grid.size), dtype=float)
    for j in range(grid.size):
        for i in range(grid.size):
            plane[:, j, i] = gs[j, i] * q1 + gt[j, i] * q2

    anim = projection_blend(A, b, t)
    sigma = jacobi_svd(A).s
    fr = Frame()
    fr.add("col_a1", A[:, 0], label="A 的第 1 列")
    fr.add("col_a2", A[:, 1], label="A 的第 2 列")
    fr.add("b_vec", b, label="b")
    fr.add("p_vec", p, label="投影 p = A A⁺ b")
    fr.add("r_vec", r, label="残差 r = b − p")
    fr.add("p_anim", anim["p_t"], label="动画：当前 p_t")
    fr.add("r_anim", anim["r_t"], label="动画：当前 r_t")
    fr.add("plane_x", plane[0], label="列空间网格 x")
    fr.add("plane_y", plane[1], label="列空间网格 y")
    fr.add("plane_z", plane[2], label="列空间网格 z")
    # 名字要名副其实：``residual`` 放真残差（requiredAux 要的就是它），
    # 正交性缺陷另起一个名字，别让前端把"缺陷"画成"残差"
    fr.add("residual", residual_norms, label="三种解法的残差范数 ‖b − A x̂‖")
    fr.add("orthogonality_defect", defect, label="三种解法的正交性缺陷 ‖Aᵀr̂‖/(‖A‖_F‖r̂‖)")
    fr.add("error", proj_error, label="三种解法的投影误差")
    fr.add("method_singular", method_singular, label="该路线是否数值奇异（1 = 没有有限解，数组值为地板）")
    qr_sol = sols["qr"]
    fr.add(
        "orthogonality_abs_qr",
        np.array([qr_sol.normal_defect if qr_sol is not None else _LOG_FLOOR]),
        label="‖Aᵀr̂‖（绝对，QR；QR 也奇异时为地板）",
    )
    sel = order.index(method)
    selected_ok = sols[method] is not None
    note_parts = []
    if failed:
        for name in order:
            if name in failed:
                note_parts.append(f"{_method_label(name)}（κ={kappa:.0e}）无有限解：{failed[name]}")
        note_parts.append(
            f"这些路线的三个数组值留在地板 {_LOG_FLOOR:.0e}，并用 method_singular=1 标出；"
            "其余路线照常给出实测值"
        )
    strings = {
        "selected_method": method,
        "method_order": "normal / qr / svd",
        "status": (
            "数值奇异：" + "；".join(f"{_method_label(n)}" for n in order if n in failed) + " 没有有限解"
            if failed
            else ""
        ),
        "note": "；".join(note_parts),
        **{f"method_{i}": _method_label(name) for i, name in enumerate(order)},
    }
    fr.set_strings(**strings)
    fr.set_meta(
        kappa_target=kappa,
        kappa_measured=_scalar(condition_estimate_2(A)),
        sigma_max=_scalar(sigma[0]),
        sigma_min=_scalar(sigma[-1]),
        angle_b_plane_deg=angle_plane,
        residual_norm=r_norm,
        projection_norm=_scalar(np.sqrt(np.dot(p, p))),
        b_norm=b_norm,
        # 奇异的那条路线**没有有限值**：meta 写 None（JSON 里就是 null），
        # 字符串里说清为什么；绝不写 NaN/Inf（Frame 会直接拒绝）
        defect_selected=_scalar(defect[sel]) if selected_ok else None,
        defect_max=_scalar(np.max(defect)) if ok_methods else None,
        defect_floor=_LOG_FLOOR,
        proxied_axis_range=1.8,
        orthogonality_abs=_scalar(sols[method].normal_defect) if selected_ok else None,
        rel_error_selected=_scalar(proj_error[sel]) if selected_ok else None,
        singular=bool(failed),
        n_singular_methods=float(len(failed)),
        selected_singular=bool(not selected_ok),
        t_blend=float(min(max(t, 0.0), 1.0)),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 2：正规方程 vs QR
# --------------------------------------------------------------------------- #
def _normal_vs_qr(params: Mapping[str, Any], t: float) -> Frame:
    exp_max = float(params["exp_max"])
    n = int(params["n"])
    m = max(n + 1, int(round(n * float(params["m_factor"]))))
    refine = {"none": 0, "one": 1, "two": 2}.get(str(params["refinement"]), 0)
    n_points = 17
    sweep = condition_sweep(
        exp_min=2.0,
        exp_max=exp_max,
        n_points=n_points,
        m=m,
        n=n,
        refinement=refine,
        n_seeds=2,
    )
    kappas = sweep["kappa"]
    log10_kappa = sweep["log10_kappa"]
    err_normal = sweep["error_normal"]
    err_qr = sweep["error_qr"]
    err_svd = sweep["error_svd"]
    err_ir = sweep["error_normal_ir"]
    res_normal = sweep["residual_normal"]
    res_qr = sweep["residual_qr"]
    ratio = _floor(err_normal / np.maximum(err_qr, _LOG_FLOOR))

    # 当前扫描位置（时间轴 = log10 κ）
    target = float(min(max(t, float(np.min(log10_kappa))), float(np.max(log10_kappa))))
    idx = int(np.argmin(np.abs(log10_kappa - target)))
    kappa_now = float(kappas[idx])
    has_ir = refine > 0

    fr = Frame()
    fr.add("kappa_grid", log10_kappa, label="log10 κ 网格")
    fr.add("error_normal", err_normal, label="正规方程前向误差")
    fr.add("error_qr", err_qr, label="QR 前向误差")
    fr.add("error_svd", err_svd, label="SVD 前向误差")
    if has_ir:
        fr.add("error_normal_ir", err_ir, label="正规方程 + 迭代精化")
    fr.add("residual_normal", res_normal, label="正规方程相对残差")
    fr.add("residual_qr", res_qr, label="QR 相对残差")
    fr.add("theory_kappa2", sweep["theory_kappa2"], label="理论参考线 κ²u")
    fr.add("theory_kappa1", sweep["theory_kappa1"], label="理论参考线 κu")
    fr.add("divergence", ratio, label="误差之比（正规方程/QR）")
    fr.add("kappa_mark", np.array([kappa_now]), label="当前 κ")
    fr.add("error_mark", np.array([err_normal[idx], err_qr[idx], err_svd[idx]]), label="当前误差")
    if has_ir:
        fr.add("error_mark_ir", np.array([err_ir[idx]]), label="当前精化误差")
    fr.add("residual_mark", np.array([res_normal[idx], res_qr[idx]]), label="当前残差")
    fr.add("defect_normal", sweep["defect_normal"], label="‖Aᵀr̂‖/(n u ‖A‖_F(‖b‖+‖A‖_F‖x̂‖))（正规方程）")
    fr.add("defect_qr", sweep["defect_qr"], label="‖Aᵀr̂‖/(n u ‖A‖_F(‖b‖+‖A‖_F‖x̂‖))（QR）")
    # 每个 κ 上有多少个种子让这条路线**解不出来**（0 = 全部解出来）。
    # 曲线的上限值 1.0 是占位：它表示"没有有限解"，不是"误差恰好 100%"，
    # 所以另给一条 0/1 标记，前端据此在曲线上画空心点或断线（防误导规则 1/4）。
    sing_normal = np.minimum(sweep["singular_normal"], 1.0)
    sing_qr = np.minimum(sweep["singular_qr"], 1.0)
    fr.add("singular_normal", sing_normal, label="正规方程是否解不出来（1 = 该 κ 上没有有限解）")
    fr.add("singular_qr", sing_qr, label="QR 是否解不出来（1 = 该 κ 上没有有限解）")
    n_sing_normal = int(np.count_nonzero(sweep["singular_normal"]))
    n_sing_qr = int(np.count_nonzero(sweep["singular_qr"]))
    note = "两条误差曲线在未饱和窗口内才服从幂律；饱和段不参与斜率拟合"
    if n_sing_normal:
        note += (
            f"；正规方程在 {n_sing_normal}/{n_points} 个 κ 上**没有有限解**"
            f"（AᵀA 的 LU 主元恰好为 0，κ ≳ 1e{np.log10(float(np.min(kappas[sing_normal > 0.0]))):.0f}），"
            f"这些点的误差是占位上限 {sweep['error_ceiling']:.0f}、已排除在斜率拟合之外，"
            "singular_normal 标出它们"
        )
    if n_sing_qr:
        note += f"；QR 在 {n_sing_qr}/{n_points} 个 κ 上也没有有限解"
    fr.set_strings(
        refinement=str(params["refinement"]),
        status=(
            f"数值奇异：正规方程在 {n_sing_normal} 个扫描点上无有限解" if n_sing_normal else ""
        ),
        note=note,
    )
    fr.set_meta(
        kappa_current=kappa_now,
        log10_kappa_current=float(log10_kappa[idx]),
        error_normal_current=_scalar(err_normal[idx]),
        error_qr_current=_scalar(err_qr[idx]),
        error_svd_current=_scalar(err_svd[idx]),
        error_ir_current=_scalar(err_ir[idx]) if has_ir else None,
        error_ratio_current=_scalar(ratio[idx]),
        slope_normal=_scalar(sweep["slopes"].get("normal", float("nan"))),
        slope_qr=_scalar(sweep["slopes"].get("qr", float("nan"))),
        slope_svd=_scalar(sweep["slopes"].get("svd", float("nan"))),
        slope_normal_ir=_scalar(sweep["slopes"].get("normal_ir", float("nan"))) if has_ir else None,
        slope_window_normal=_scalar(sweep["slope_window"].get("normal", float("nan"))),
        slope_window_qr=_scalar(sweep["slope_window"].get("qr", float("nan"))),
        error_floor=sweep["error_floor"],
        residual_floor=_LOG_FLOOR,
        n_points=float(n_points),
        m=float(m),
        n=float(n),
        n_seeds=sweep["n_seeds"],
        refinement_steps=float(refine),
        axis_x_min=2.0,
        axis_x_max=exp_max,
        axis_y_min=-17.0,
        axis_y_max=2.0,
        kappa_max_used=_scalar(np.max(sweep["kappa_measured"])),
        kappa_min_used=_scalar(np.min(sweep["kappa_measured"])),
        defect_max=_scalar(max(np.max(sweep["defect_normal"]), np.max(sweep["defect_qr"]))),
        index=float(idx),
        # 解不出来的点数（0 = 全部解出来）；误差上限值也在 meta 里说清楚
        n_singular_normal=float(n_sing_normal),
        n_singular_qr=float(n_sing_qr),
        singular_any=bool(n_sing_normal or n_sing_qr),
        error_ceiling=sweep["error_ceiling"],
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 3：奇异值谱与截断
# --------------------------------------------------------------------------- #
def _svd_truncation(params: Mapping[str, Any], t: float) -> Frame:
    spectrum = str(params["spectrum"])
    kappa = 10.0 ** float(params["log10_kappa"])
    noise_rel = 10.0 ** float(params["noise_exp"])
    n = int(params["n"])
    m = max(n + 1, n * int(params["oversample"]))

    A, b, x_true, sigma_true = synthetic_truncation_problem(
        m, n, kappa, spectrum=spectrum, decay=2.0, noise_rel=noise_rel, seed=20261018
    )
    sweep = truncation_sweep(A, b, x_true=x_true)
    k_max = int(sweep["k_max"])
    k_now = _step_of(t, k_max)

    x_norm = sweep["x_norm"]
    residual = sweep["residual"]
    error = sweep["error"]
    k_grid = sweep["k"]

    # 奇异值悬崖：相邻 σ 比值的最大跳变位置（截断阈值落在它上面就会突变）
    sigma = sweep["sigma"]
    ratios = sigma[:-1] / np.maximum(sigma[1:], np.finfo(float).tiny)
    cliff_index = int(np.argmax(ratios)) + 1 if ratios.size else 0
    # ‖x_k‖ 的相邻跳变：从 k=1 开始比（k=0 的解恒为 0，比值没有意义，硬除会溢出成 Inf）
    xnorm = np.maximum(x_norm, 1.0e-300)
    jumps = xnorm[2:] / xnorm[1:-1] if xnorm.size > 2 else np.ones(1)
    xnorm_jump = min(float(np.max(jumps)), 1.0e30)

    fr = Frame()
    fr.add("k_grid", k_grid, label="截断阶 k")
    fr.add("sigma", _floor(sigma), label="奇异值 σ_i")
    fr.add("coeff", _floor(sweep["coeff"]), label="|u_iᵀb|")
    fr.add("picard", _floor(sweep["picard"]), label="Picard 比 |u_iᵀb|/σ_i")
    fr.add("x_norm", _floor(x_norm), label="解范数 ‖x_k‖")
    fr.add("residual", _floor(residual), label="相对残差 ‖Ax_k−b‖/‖b‖")
    fr.add("error", _floor(np.nan_to_num(error, nan=_LOG_FLOOR)), label="相对误差 ‖x_k−x*‖/‖x*‖")
    fr.add("lcurve_x", _floor(x_norm), label="L 曲线横坐标（解范数）")
    fr.add("lcurve_y", _floor(residual), label="L 曲线纵坐标（残差）")
    fr.add("k_mark", np.array([float(k_now)]), label="当前截断阶")
    fr.add("x_norm_mark", np.array([_scalar(x_norm[k_now], _LOG_FLOOR)]), label="当前解范数")
    fr.add("residual_mark", np.array([_scalar(residual[k_now], _LOG_FLOOR)]), label="当前残差")
    fr.add("error_mark", np.array([_scalar(error[k_now], _LOG_FLOOR)]), label="当前误差")
    fr.add("k_corner_mark", np.array([sweep["k_corner"]]), label="L 曲线拐点 k")
    fr.add("k_opt_mark", np.array([sweep["k_opt_error"]]), label="真实最优 k")
    fr.add("k_picard_mark", np.array([sweep["k_picard"]]), label="Picard 转折点 k")
    fr.add("noise_level", np.array([max(noise_rel, _LOG_FLOOR)]), label="噪声水平")
    fr.set_strings(spectrum=spectrum, note="k_opt 只有知道真解才算得出来；L 曲线拐点只是启发式")
    fr.set_meta(
        k_current=float(k_now),
        k_max=float(k_max),
        k_opt_error=_scalar(sweep["k_opt_error"]),
        k_corner=float(sweep["k_corner"]),
        k_picard=_scalar(sweep["k_picard"]),
        cliff_index=float(cliff_index),
        x_norm_jump=xnorm_jump,
        kappa_target=kappa,
        kappa_measured=_scalar(sweep["condition"]),
        sigma_max=_scalar(sigma[0]),
        sigma_min=_scalar(sigma[-1]),
        rank=float(sweep["rank"]),
        noise_rel=noise_rel,
        noise_exp=float(params["noise_exp"]),
        n=float(n),
        m=float(m),
        x_norm_current=_scalar(x_norm[k_now], _LOG_FLOOR),
        residual_current=_scalar(residual[k_now], _LOG_FLOOR),
        error_current=_scalar(error[k_now], _LOG_FLOOR),
        error_floor=_LOG_FLOOR,
        picard_min=_scalar(np.min(sweep["picard"])),
        sigma_floor=_scalar(np.min(sigma)),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：解的不确定扇面
# --------------------------------------------------------------------------- #
def _uncertainty_fan(params: Mapping[str, Any], t: float) -> Frame:
    kappa = 10.0 ** float(params["log10_kappa"])
    noise_rel = 10.0 ** float(params["noise_exp"])
    rho_rel = 10.0 ** float(params["rho_exp"])
    n_levels = int(params["n_levels"])
    n_samples = int(params["n_samples"])

    A, b, x_true = synthetic_lstsq_problem(2, 2, kappa, seed=20261018)
    fit = jacobi_svd(A)
    rank = numerical_rank(fit.s, 2, 2)

    # 等残差椭圆族（同一“谷底”的不同高度）
    factors = np.logspace(-1.0, 1.0, n_levels)
    curves_x = np.zeros((n_levels, 181), dtype=float)
    curves_y = np.zeros((n_levels, 181), dtype=float)
    level_rho = np.zeros(n_levels, dtype=float)
    for i, f in enumerate(factors):
        ell = uncertainty_ellipse(A, b, rho_rel=float(rho_rel * f), n_theta=181)
        curves_x[i, :] = ell["ellipse_x"]
        curves_y[i, :] = ell["ellipse_y"]
        level_rho[i] = float(rho_rel * f)

    fan = noise_fan(A, b, noise_rel=noise_rel, n_angles=n_samples)
    error = _floor(fan["error"])
    residual = _floor(fan["residual"])
    theta = np.rad2deg(fan["theta"])

    # 当前方向（时间轴 = 噪声方向角）
    theta_now = float(np.mod(float(t), 360.0))
    idx = int(np.argmin(np.abs(theta - theta_now)))
    x_now = np.array([fan["sol_x"][idx], fan["sol_y"][idx]])
    n_visible = max(1, int(round((theta_now / 360.0) * n_samples)) + 1)
    center = fan["center"]

    # 副轴：不确定度随 κ 的增长（实测）与 κ·δ 参考线
    kappa_grid = np.logspace(0.0, 12.0, 13)
    error_kappa = np.zeros(kappa_grid.size, dtype=float)
    for i, kap in enumerate(kappa_grid):
        Ai, bi, _xi = synthetic_lstsq_problem(2, 2, float(kap), seed=20261018)
        f = noise_fan(Ai, bi, noise_rel=noise_rel, n_angles=16)
        error_kappa[i] = float(np.max(f["error"]))
    theory = np.maximum(noise_rel * kappa_grid, _LOG_FLOOR)
    semi = float(fit.s[0]) / max(float(fit.s[-1]), np.finfo(float).tiny)

    fr = Frame()
    fr.add("level_rho", _floor(level_rho), label="等残差水平 ρ/‖b‖")
    fr.add("level_x", curves_x, label="等残差椭圆 x（每条一行）")
    fr.add("level_y", curves_y, label="等残差椭圆 y")
    fr.add("fan_x", fan["sol_x"], label="噪声扇面解 x")
    fr.add("fan_y", fan["sol_y"], label="噪声扇面解 y")
    fr.add("error", error, label="各方向的相对不确定度 ‖δx‖/‖x*‖")
    fr.add("residual", residual, label="各方向的相对残差（应当完全相同）")
    fr.add("theta_deg", theta, label="各方向的角度")
    fr.add("sol_now", x_now, label="当前方向的解")
    fr.add("center", center, label="x*（谷底中心）")
    fr.add("v1_dir", fit.V[:, 0], label="右奇异向量 v₁")
    fr.add("v2_dir", fit.V[:, -1], label="右奇异向量 v₂")
    fr.add("sigma", _floor(fit.s), label="奇异值")
    fr.add("semi_axes", np.array([semi, 1.0]), label="等残差椭圆长短轴之比（第二个归一为 1）")
    fr.add("kappa_grid", kappa_grid, label="κ 网格")
    fr.add("error_kappa", _floor(error_kappa), label="实测最大不确定度 vs κ")
    fr.add("theory_kappa", theory, label="参考线 κ·δ")
    fr.set_strings(
        rank_note=f"数值秩 {rank}/{2}",
        note="所有扇面解的残差相同（差别只在 κu‖x‖ 的舍入水平）；解本身却差了 κ 倍",
    )
    x_star_norm = _scalar(np.sqrt(np.dot(center, center)), 1.0)
    fr.set_meta(
        kappa_target=kappa,
        kappa_measured=_scalar(condition_estimate_2(A)),
        sigma_max=_scalar(fit.s[0]),
        sigma_min=_scalar(fit.s[-1]),
        noise_rel=noise_rel,
        rho_rel=rho_rel,
        amplification_max=_scalar(np.max(error)),
        amplification_min=_scalar(np.min(error)),
        amplification_theory=_scalar(noise_rel * kappa),
        residual_max=_scalar(np.max(residual)),
        residual_min=_scalar(np.min(residual)),
        residual_spread=_scalar(float(np.max(residual) - np.min(residual))),
        # 残差"常量性"的舍入地板：x 本身带 κu 的相对误差，于是 ‖Ax−b‖ 只能稳到 κu‖x‖
        residual_roundoff=_scalar(condition_estimate_2(A) * _U * x_star_norm),
        residual_flatness=_scalar(
            float(np.max(residual) - np.min(residual)) / max(float(np.mean(residual)), _LOG_FLOOR)
        ),
        # 扇面在测噪声还是在测舍入？判据：解自带的相对舍入 κu 必须远小于噪声
        roundoff_rel=_scalar(condition_estimate_2(A) * _U),
        resolvable=bool(noise_rel > 10.0 * condition_estimate_2(A) * _U),
        theta_current=theta_now,
        n_visible=float(min(n_visible, n_samples)),
        n_samples=float(n_samples),
        n_levels=float(n_levels),
        semi_axis_ratio=semi,
        px_range=2.5,
        error_floor=_LOG_FLOOR,
        x_norm=x_star_norm,
        fan_width_deg=_scalar(float(np.rad2deg(np.arctan2(np.max(error), 1.0)))),
    )
    return fr
