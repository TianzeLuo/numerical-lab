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
                "$A$ 的两列张成 $\\mathbb{R}^{3}$ 里的一个平面，$b$ 投影上去得到 $p$，$r=b-p$。"
                "时间轴把 $b=p+r$ 逐步拆开（$t=0$ 只看得到 $b$，$t=1$ 两个分量完全分开）；"
                "副轴给出三种解法实测的 $\\|A^{\\mathrm{T}}r\\|/(\\|A\\|_{\\mathrm{F}}\\|r\\|)$——直角不能只靠眼睛看。"
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
                    "help": "右端项 $b$ 在水平面内的方位角。改变它会让 $b$ 扫过不同的方向；再配合天顶角，可以把 $b$ 转到几乎落在列空间里，或者垂直地指向平面。",
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
                    "help": "右端项 $b$ 与 $+z$ 轴的夹角（天顶角）。$0^\\circ$ 指向正上方，$90^\\circ$ 落在水平面，$180^\\circ$ 指向正下方；调整它可以直接看到夹角如何影响残差。",
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
                        "$\\kappa u=1$ 的界线（$\\log_{10}(1/u)\\approx 15.95$）：超过它以后，$A$ 的小奇异方向在"
                        "float64 里已经不存在，列空间在数值上“扁”成一条线，画出来的平面是幻觉。"
                    ),
                    "beyond": "above",
                    "help": "把矩阵 $A$ 的条件数设为 $10^{\\text{这个值}}$。数值越大，两列越接近共线，列空间在数值上越“扁”；副轴会显示三条求解路线各自的残差和正交性缺陷。",
                },
                {
                    "name": "method",
                    "label": "求解路线（用于副轴对照）",
                    "type": "enum",
                    "options": ["qr", "normal", "svd"],
                    "default": "qr",
                    "critical": [],
                    "criticalNote": "",
                    "help": "主场景里选中的求解路线，只影响突出显示的数值解与状态，不改数据；副轴始终同时比较正规方程、QR 与 SVD（截断 SVD）三条路线，便于直接对照。",
                },
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.05, "label": "分解动画 t", "initial": 1.0},
            "requiredAux": ["residual", "error"],
            "axes": [
                {
                    "id": "scene",
                    "autoscaled": False,
                    "title": "三维场景（三轴等比例、固定 $\\pm1.8$）",
                    "ylabel": "坐标",
                    "scale": "linear",
                },
                {
                    "id": "orthogonality",
                    "autoscaled": True,
                    "title": "三种解法的正交性缺陷 $\\|A^{\\mathrm{T}}\\hat r\\|/(\\|A\\|_F\\|\\hat r\\|)$",
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
                "**先把“无解”变成一个优化问题。** 当 $A\\in\\mathbb{R}^{m\\times n}$ 的行数多于列数时，$Ax=b$ 一般没有精确解。最小二乘退而求其次，寻找让残差平方和最小的 $x$：$x_{\\mathrm{LS}}=\\arg\\min_x\\|Ax-b\\|_2^2.$ 这个目标函数是 $x$ 的二次函数，因此它的最优性可以用一阶条件完整刻画。",
                "**正规方程是求导得到的，不是背下来的。** 对任意方向 $z$ 展开：$\\|A(x+z)-b\\|^2=\\|Ax-b\\|^2+2z^{\\mathrm T}A^{\\mathrm T}(Ax-b)+\\|Az\\|^2.$ 如果 $x$ 已经最小，那么沿 $z$ 和 $-z$ 走都不能让目标下降，线性项必须为零，即 $A^{\\mathrm T}(Ax-b)=0$。反过来，这一项为零时右端只剩 $\\|Az\\|^2\\ge0$，所以 $x$ 确实最小。",
                "**几何上，残差必须垂直于列空间。** 记 $p=Ax$、$r=b-p$。正规方程就是 $A^{\\mathrm T}r=0$，也就是 $r$ 与 $A$ 的每一列都正交。于是 $b$ 被拆成“列空间里的投影 $p$”加“垂直于列空间的残差 $r$”，并且有勾股恒等式 $\\|b\\|^2=\\|p\\|^2+\\|r\\|^2$。三维动画在 $t=0$ 只显示 $b$，在 $t=1$ 把 $p$ 与 $r$ 完全分开，就是为了让这件事看得见。",
                "**满列秩保证解唯一，秩亏则不是。** 如果 $A$ 的列线性无关，$A^{\\mathrm T}A$ 可逆，正规方程给出唯一的 $x_{\\mathrm{LS}}$。如果列相关，所有解只差一个零空间里的向量，而残差完全相同；这时“最小二乘解”还需要额外约定（例如最小范数解）。图中的条件数 $\\kappa$ 越大，列越接近共线，唯一性在数值上越脆弱。",
                "**三种解法在浮点里并不等价。** 直接解正规方程 $A^{\\mathrm T}Ax=A^{\\mathrm T}b$ 会把条件数平方成 $\\kappa^2$；Householder QR 只做正交变换，不改变 2-范数，通常把误差拉回 $\\kappa u$ 量级；SVD 则提供最稳健的分解，但代价更高。副轴的正交性缺陷 $\\|A^{\\mathrm T}\\hat r\\|/(\\|A\\|_F\\|\\hat r\\|)$ 在三种路线下分别实测，用来暴露“残差看起来垂直”与“解是否可信”是两件事。",
                "**参数怎样改变画面。** 方位角 $\\theta$ 与天顶角 $\\phi$ 控制 $b$ 指向哪里：$b$ 越接近列空间，残差越小；越接近法方向，残差越大。$\\log_{10}\\kappa$ 控制 $A$ 有多病态：越大，列空间在数值上越扁，三种解法的差异越明显。主场景固定为等比例 $\\pm1.8$ 的坐标，不允许各轴单独拉伸，否则直角会被视觉上扭弯。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "normal_vs_qr",
            "title": "精度对决：正规方程的 κ² 与 QR 的 κ",
            "description": (
                "同一批构造问题（$\\kappa$ 从 $10^{2}$ 到 $10^{12}$）上跑三条路线：正规方程、QR、SVD，"
                "双对数画前向误差并叠加 $\\kappa u$ 与 $\\kappa^{2}u$ 两条理论参考线（含实测斜率）。"
                "副轴是相对残差——它一直贴着 $u$，与误差无关，这正是“残差小”骗人的地方。"
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
                        "$\\log_{10}(1/u)\\approx 15.95$ 是“双精度里还能表示 $1/\\kappa$ 量级扰动”的尽头："
                        "$\\kappa$ 超过它以后任何解法的前向误差都会饱和到 $O(1)$，曲线不再是幂律。"
                    ),
                    "beyond": "above",
                    "help": "扫描的条件数上限（对数），从 $10^2$ 一直跑到 $10^{\\text{这个值}}$。调大能看见正规方程更早饱和；曲线是否还服从幂律，要看副轴与斜率窗口。",
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
                    "help": "未知数个数。$n$ 越大，构造问题越复杂，不同范数间的 $n$ 因子差异也越明显；计算量随之增加。",
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
                    "help": "方程个数与未知数个数之比 $m/n$。必须大于 $1$ 才是超定问题；比值越大，残差方向越多，但最小二乘解的正交性结论不变。",
                },
                {
                    "name": "refinement",
                    "label": "正规方程的迭代精化步数",
                    "type": "enum",
                    "options": ["none", "one", "two"],
                    "default": "one",
                    "critical": [],
                    "criticalNote": "",
                    "help": "对正规方程做几步迭代精化。精化能减小“解方程”的舍入误差，却修不了 $A^{T}A$ 本身把条件数平方的事实；$\\kappa$ 接近 $1/u$ 后它也会失效。",
                },
            ],
            "time": {"mode": "scrub", "tMax": 14.0, "dt": 0.25, "label": "当前 $\\log_{10}\\kappa$", "initial": 8.4},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "errors",
                    "autoscaled": False,
                    "title": "前向误差 vs $\\kappa$（双对数、固定轴 $10^{-17}$ 到 $10^2$）",
                    "ylabel": "$\\|\\hat x-x^*\\|/\\|x^*\\|$",
                    "scale": "log",
                },
                {
                    "id": "residual",
                    "autoscaled": True,
                    "title": "相对残差 $\\|A\\hat x-b\\|/\\|b\\|$（对数）",
                    "ylabel": "相对残差",
                    "scale": "log",
                },
                {
                    "id": "divergence",
                    "autoscaled": False,
                    "title": "分道扬镳：误差之比（正规方程 / QR，对数）",
                    "ylabel": "比值",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "陷阱一：**自适应坐标会把两条曲线拉平**。误差从 $10^{-14}$ 走到 $10^{0}$，如果主轴的纵坐标"
                "跟着数据自动缩放，$\\kappa^{2}$ 与 $\\kappa$ 两条曲线在屏幕上都会变成“两条平行直线”，"
                "看不出“正规方程在某个 $\\kappa$ 之后彻底崩掉”。本帧把主轴固定为 $10^{-17}$ 到 $10^{2}$ 并把范围写进"
                "meta，配合 overflow 标记——图上看不见的那一段，必须由数字说出来。"
                "陷阱二：**“残差更小”被当成“解更准”**。副轴上三条路线的相对残差都贴着 $10^{-16}$ 的地板，"
                "看不出任何差别，可同一时刻正规方程的前向误差已经比 QR 大 $10$ 个数量级"
                "（$\\kappa=10^{8}$ 时实测比值 $\\sim10^{8}$）。另外注意：迭代精化能把正规方程救回来一段，"
                "但它修的是“解方程”的误差，修不了 $A^{\\mathrm{T}}A$ 本身的病态，$\\kappa$ 越过 $1/u$ 之后一样失效。"
            ),
            "theory": [
                "**正规方程把条件数平方了。** 若 $A=U\\Sigma V^{\\mathrm T}$，则 $A^{\\mathrm T}A=V\\Sigma^2V^{\\mathrm T}$，奇异值从 $\\sigma_i$ 变成 $\\sigma_i^2$，所以 $\\kappa_2(A^{\\mathrm T}A)=\\kappa_2(A)^2.$ 这意味着 $\\kappa=10^8$ 的问题，在正规方程里看起来像 $\\kappa=10^{16}$ 的问题，已经接近 double 的分辨极限。",
                "**误差界随之分道扬镳。** 用正规方程求解时，前向相对误差大致按 $\\kappa^2u$ 增长；QR 只做正交变换，前向误差约为 $\\kappa u+\\kappa^2u\\,\\eta$，其中 $\\eta$ 是残差因子，相容问题时 $\\eta\\approx0$，于是只剩 $\\kappa u$。双对数图上这两条斜率分别为 2 和 1 的直线就是这件事的可视化。",
                "**QR 为什么能不碰 $A^{\\mathrm T}A$。** Householder 变换用一串正交矩阵 $Q$ 把 $A$ 化成上三角 $R$：$A=QR$。正交变换满足 $\\|Qz\\|=\\|z\\|$，不会放大误差；最小二乘问题变成解 $R x=Q^{\\mathrm T}b$，只需回代一个三角系统。SVD 更进一步，用两组正交基和奇异值直接写出伪逆，但计算量更大。",
                "**残差小不等于解准。** 对任何满列秩的 $A$，QR 解都让 $A^{\\mathrm T}\\hat r$ 接近机器精度；残差范数也可能一直贴着 $u$。可是当 $\\kappa$ 很大时，$A$ 的两个奇异方向几乎重合，两个完全不同的 $x$ 都能给出几乎相同的 $b$。副轴上的残差曲线与主图的前向误差曲线必须分开读。",
                "**迭代精化修什么、不修什么。** 精化用当前解计算残差，再解一次修正方程，可以抵消一部分“求解线性系统”产生的舍入误差；但它无法恢复 $A^{\\mathrm T}A$ 已经丢掉的奇异方向。越过 $\\kappa\\approx1/u$ 之后，无论多少步精化，误差都会饱和到 $O(1)$。",
                "**扫描图的正确读法。** 主轴固定为 $10^{-17}$ 到 $10^{2}$，曲线越过上界时由溢出说明点名；斜率只在未饱和窗口内拟合。图上的实测斜率是量出来的，不是看出来的；$\\kappa$ 用 2-范数实测，目标值只用于构造矩阵。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "svd_truncation",
            "title": "奇异值谱与截断：L 曲线、Picard 条件与最优 k",
            "description": (
                "拖动截断阶 $k$（时间轴）看三件事同步变化：解范数 $\\|x_k\\|$、残差 $\\|Ax_k-b\\|$、"
                "以及只有已知真解时才算得出来的真实误差。主轴是 L 曲线（解范数 vs 残差，双对数），"
                "副轴是奇异值谱与 Picard 量 $|u_i^{\\mathrm{T}}b|/\\sigma_i$。标出 L 曲线拐点、Picard 转折点与真正最优的 $k$。"
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
                    "help": "选择奇异值的分布形状。衰减谱、断崖谱、平台谱与双簇谱的 Picard 曲线完全不同，因此最优截断阶也不同；切换后请同时看 L 曲线拐点与真实误差。",
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
                        "$\\kappa>1/u$（$\\log_{10}\\approx 15.95$）以后，最小的奇异值在 float64 里无法与 $0$ 区分，"
                        "谱的尾部整个是表示误差——此时“要不要截断”已经不是算法选择而是物理限制。"
                    ),
                    "beyond": "above",
                    "help": "把矩阵 $A$ 的条件数设为 $10^{\\text{这个值}}$。数值越大，两个奇异方向越接近，解平面上的等残差椭圆越细长；副轴会显示实测不确定度随 $\\kappa$ 的增长。",
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
                        "噪声小于 $u\\approx 1.1\\times10^{-16}$ 时（$\\log_{10}\\approx -15.95$）它在 double 里根本表示不出来："
                        "再往下拖，图上的变化全是舍入噪声，不是数据噪声。"
                    ),
                    "beyond": "below",
                    "help": "右端项里噪声的相对大小（以 10 为底）。噪声越小，尾部奇异方向还可以多信几个；噪声越大，越早截断。低于 $u$ 时噪声在 double 里表示不出来。",
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
                    "help": "未知数个数。$n$ 越大，奇异值谱越长，可截断的阶数越多；构造问题的计算量与曲线上的点数也随之增加。",
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
                    "help": "方程数 $m$ 与未知数 $n$ 的比值。超定程度越高，残差自由度越多，Picard 量的统计意义也越稳定；不建议低于 $1$。",
                },
            ],
            "time": {"mode": "scrub", "tMax": 16.0, "dt": 1.0, "label": "截断阶 $k$", "initial": 6.4},
            "requiredAux": ["residual", "error"],
            "axes": [
                {
                    "id": "lcurve",
                    "autoscaled": True,
                    "title": "L 曲线：$\\|x_k\\|$ 与 $\\|Ax_k-b\\|/\\|b\\|$（双对数）",
                    "ylabel": "$\\|x_k\\|$",
                    "scale": "log",
                },
                {
                    "id": "spectrum",
                    "autoscaled": True,
                    "title": "奇异值谱与 Picard 量 $|u_i^{\\mathrm{T}}b|/\\sigma_i$（对数）",
                    "ylabel": "$\\sigma_i$、$|u_i^{\\mathrm{T}}b|$、$|u_i^{\\mathrm{T}}b|/\\sigma_i$",
                    "scale": "log",
                },
                {
                    "id": "tradeoff",
                    "autoscaled": True,
                    "title": "三个量随截断阶 k 的变化（对数）",
                    "ylabel": "$\\|x_k\\|$ / 残差 / 相对误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "陷阱一：**截断阈值取在奇异值悬崖上，解范数会突变**。把谱形状切到 cliff："
                "奇异值前半段全是 $1$、后半段突然掉到 $1/\\kappa$，此时 $k$ 差一个下标，$\\|x_k\\|$ 可以跳几个数量级，"
                "L 曲线上出现一段近乎竖直的线——“最优 $k$”在悬崖上根本不可辨。"
                "陷阱二：**L 曲线的拐点不等于最优截断**。本帧同时给出 k_corner（离散曲率最大处）、"
                "k_picard（Picard 比 $|u_i^{\\mathrm{T}}b|/\\sigma_i$ 的转折点）与 k_opt（真实误差最小处，"
                "只有知道真解才算得出来）：四种种子里它们经常互不相等。另外，"
                "副轴上 Picard 比在小奇异值方向**重新抬头**是“噪声接管”的signature，"
                "不是“这个方向有信息”。最后：残差曲线随 $k$ 单调下降、解范数单调上升，"
                "两条单调曲线本身不告诉你该在哪里停——停在“残差已经很小”的地方通常停得太晚。"
            ),
            "theory": [
                "**SVD 把问题拆成互不干扰的方向。** 任意矩阵可写成 $A=\\sum_i\\sigma_i u_i v_i^{\\mathrm T}$，其中 $u_i$ 与 $v_i$ 是正交的左右奇异向量，$\\sigma_1\\ge\\cdots\\ge\\sigma_n\\ge0$。最小二乘解可以展开为 $x=\\sum_i\\frac{u_i^{\\mathrm T}b}{\\sigma_i}v_i.$ 每个方向的贡献是“投影系数除以奇异值”。",
                "**小奇异值会放大噪声。** 如果 $b$ 里混有噪声，$u_i^{\\mathrm T}b$ 在尾部不会继续衰减，而 $1/\\sigma_i$ 越来越大，解会被尾部方向拖走。Picard 条件要求 $|u_i^{\\mathrm T}b|$ 比 $\\sigma_i$ 衰减得更快；噪声破坏了这个条件，所以必须丢掉一部分小奇异方向。",
                "**截断 SVD 就是只保留前 $k$ 个方向。** 定义 $x_k=\\sum_{i\\le k}(u_i^{\\mathrm T}b/\\sigma_i)v_i$。$k$ 增大时残差 $\\|Ax_k-b\\|$ 下降，但解范数 $\\|x_k\\|$ 与噪声放大同时上升；把 $(\\|x_k\\|,\\|Ax_k-b\\|)$ 画在双对数坐标上，就是 L 曲线。拐角处是“残差已经够小、解还没被噪声接管”的折中点。",
                "**为什么截断不是权宜之计。** Eckart–Young–Mirsky 定理说：在所有秩不超过 $k$ 的矩阵中，截断 SVD $A_k=\\sum_{i\\le k}\\sigma_i u_i v_i^{\\mathrm T}$ 在 2-范数和 Frobenius 范数下是最优逼近。也就是说，丢掉尾部奇异方向并不是随便砍一刀，而是在给定秩的约束下做最优投影。",
                "**谱形状决定截断难度。** decay 谱平滑衰减，Picard 曲线也平滑，拐点不明显；cliff 谱有一个明显断崖，最优 $k$ 很接近断崖位置；plateau 谱在中段平坦，难以区分信号与噪声；two_clusters 谱会给出两个候选截断阶。噪声越小，可保留的阶数越多。",
                "**图上三条曲线怎样对应。** 主图是 L 曲线，副图是奇异值谱与 Picard 量；真实误差只有在知道真解时才算得出，用来检验“L 曲线拐点”和“最优 $k$”是否重合。$k=0$ 的解恒为 0，在对数 L 曲线上没有坐标，因此被排除并说明——这是显示约定，不是丢数据。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "uncertainty_fan",
            "title": "解的不确定扇面：等残差的谷底有多长",
            "description": (
                "在一个 $2\\times2$ 相容问题上把 $b$ 沿噪声方向转一圈：每个扰动解都有**相同**的残差"
                "（= 噪声水平），但解本身在一条很长的椭圆上跑。主场景画等残差椭圆族、"
                "噪声扇面与当前方向；副轴给出实测不确定度随 $\\kappa$ 的增长与 $\\kappa\\cdot\\delta$ 参考线。"
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
                        "$\\kappa\\cdot u=1$ 的界线：越过它以后 $A$ 在 double 里已经奇异，"
                        "等残差椭圆退化成一条无法分辨的线段，扇面宽度不再有数值意义。"
                    ),
                    "beyond": "above",
                    "help": "把矩阵 $A$ 的条件数设为 $10^{\\text{这个值}}$。数值越大，两个奇异方向越接近，解平面上的等残差椭圆越细长；副轴会显示实测不确定度随 $\\kappa$ 的增长。",
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
                        "低于 $u$（$\\log_{10}\\approx -15.95$）的噪声在 float64 里不可表示："
                        "扇面宽度会退化成舍入噪声的形状，与物理无关。"
                    ),
                    "beyond": "below",
                    "help": "右端项里噪声的相对大小（以 10 为底）。噪声越小，解还能多信几个小奇异方向；噪声越大，尾部方向越不可信。低于 $u\\approx1.1\\times10^{-16}$ 的噪声在 double 里表示不出来。",
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
                    "help": "解平面上等残差曲线的水平，相对于 $\\|b\\|$。它决定“谷底”的宽度；调大时等残差椭圆变大，更容易超出固定视窗（此时用中性提示说明裁剪，不是数据错误）。",
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
                    "help": "在解平面上画几条等残差椭圆。条数增加会覆盖更多解的可能范围，但画面也更拥挤；每一层对应同一个残差水平。",
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
                    "help": "沿噪声方向采多少个角度。采样越密，扇面边界越平滑，也越容易看见最坏方向；它不改变真实的不确定度。",
                },
            ],
            "time": {"mode": "scrub", "tMax": 360.0, "dt": 5.0, "label": "噪声方向角 θ（度）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "plane",
                    "autoscaled": False,
                    "title": "解平面（固定等比例坐标，等残差椭圆族）",
                    "ylabel": "$x_2$",
                    "scale": "linear",
                },
                {
                    "id": "growth",
                    "autoscaled": True,
                    "title": "实测不确定度 vs $\\kappa$（双对数）与 $\\kappa\\cdot\\delta$ 参考线",
                    "ylabel": "$\\|\\delta x\\|/\\|x^*\\|$",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "陷阱一：**残差完全相同，解却可以差 $\\kappa$ 倍**。扇面上每一条射线对应的解都有同一个残差"
                "（副轴那条水平线就是实测：所有方向的 $\\|A\\hat{x}-b\\|/\\|b\\|$ 都等于噪声水平），"
                "所以“用残差挑解”在这里完全失效；解的差别只能靠 $\\|\\delta x\\|/\\|x^{*}\\|$ 看出来。"
                "严格地说，这份“相同”在浮点里只能稳到 $\\kappa u\\|x\\|$ 的舍入水平：默认参数下残差的变化实测"
                "$1.7\\times10^{-10}$，比噪声水平 $10^{-8}$ 还小 $60$ 倍，而同一时刻解已经动了 $2.8\\times10^{-3}$——"
                "残差对解的移动几乎是瞎的（meta 里同时给出 residual_spread 与 residual_roundoff）。"
                "陷阱二：**自适应坐标会把针画成圆**。等残差椭圆的长短轴之比就是 $\\kappa$；"
                "如果让主场景自适应缩放（两个坐标轴各自拉伸），$\\kappa=10^{6}$ 的针会被画成一个圆，"
                "整个可视化的结论当场反转——所以本帧把主轴声明为 autoscaled=false 并给出固定范围，"
                "同时用 meta 报出真实的长短轴之比。另外提醒：图上画的椭圆是**线性化**的结果"
                "（$A$ 不变、只有 $b$ 扰动）；如果 $A$ 本身有误差，放大倍数还要乘上 $A$ 的条件数，"
                "两个误差源的贡献不能相加了事。还有一个更隐蔽的：解本身自带 $\\kappa u$ 的相对舍入，"
                "$\\kappa$ 拖到 $10^{10}$ 时 $\\kappa u\\approx 1.1\\times10^{-6}$ 已经**大于**默认噪声 $10^{-8}$，此时扇面量的是舍入而不是噪声"
                "（实测最短半轴从 $0.28\\delta$ 涨到 $8.7\\delta$）——meta 里的 resolvable 就是这条判据，"
                "它为 false 时这张图不该被用来谈“噪声放大”。"
            ),
            "theory": [
                "**相容问题里，扰动解是一条直线加出来的。** 设 $A$ 满列秩、$b$ 在列空间内，最小二乘残差为零。给 $b$ 加上噪声 $\\delta b$，新解为 $x=x^*+A^+\\delta b$，其中 $A^+=V\\Sigma^{-1}U^{\\mathrm T}$ 是伪逆。解的移动完全由 $\\delta b$ 在右奇异方向上的分量决定。",
                "**等残差集是一个椭圆。** 对固定的噪声水平 $\\rho=\\|\\delta b\\|$，所有满足 $\\|A(x-x^*)\\|=\\rho$ 的 $x$ 组成一个椭圆。把 $A$ 按奇异值展开，椭圆沿 $v_i$ 方向的半轴是 $\\rho/\\sigma_i$；最长半轴与最短半轴之比正好是 $\\kappa$。所以“等残差”并不意味着“等解”，解可以在椭圆上跑很远。",
                "**前向误差界。** 由 $\\delta x=A^+\\delta b$ 得 $\\|\\delta x\\|\\le\\|A^+\\|\\,\\|\\delta b\\|$；又因为 $\\|x^*\\|$ 与 $\\|b\\|$ 通过 $A$ 联系，最终得到 $\\frac{\\|\\delta x\\|}{\\|x^*\\|}\\le\\kappa(A)\\,\\frac{\\|\\delta b\\|}{\\|b\\|}.$ 当 $\\delta b$ 沿着 $u_{\\min}$ 方向时取等号。图中副轴的 $\\kappa\\cdot\\delta$ 参考线就是这个上界。",
                "**残差对解的移动几乎“失明”。** 每个扰动解都有相同的残差范数 $\\rho$，所以只看残差无法区分它们；但解的变化已经达到 $\\kappa\\rho$ 量级。实验同时给出残差的变化范围与解的变化范围，用来展示这种不对称：残差看起来一动不动，解却已经移动了很多。",
                "**线性化与数值极限。** 图上的椭圆是线性化结果：只扰动 $b$、不扰动 $A$。如果 $A$ 自身有误差，放大倍数还要再乘上 $A$ 的条件数。另一个极限是 $\\kappa u$：当它大于噪声水平时，扇面量到的其实是舍入误差，不是物理噪声；帧里的 resolvable 就是这条判据，为假时不应把扇面宽度当成噪声放大。",
                "**固定视窗与裁剪。** 解平面使用等比例固定窗口，才能让椭圆看起来仍然是椭圆；如果让两个坐标轴各自自适应，$\\kappa=10^6$ 的细椭圆会被拉成一个圆，结论直接反转。代价是椭圆常常超出窗口，超出部分被裁掉；这是**显示选择**，用中性的“视窗裁剪”说明，不是数据坏了。",
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
    else:
        note_parts.append(
            f"三种路线都给出了有限解；当前选中{_method_label(method)}，"
            f"投影误差 {proj_error[sel]:.3e}，残差 {residual_norms[sel]:.3e}"
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
    sing_svd = np.minimum(sweep["singular_svd"], 1.0)
    fr.add("singular_svd", sing_svd, label="SVD 是否解不出来（1 = 该 κ 上没有有限解）")
    if has_ir:
        sing_ir = np.minimum(sweep["singular_normal_ir"], 1.0)
        fr.add("singular_normal_ir", sing_ir, label="精化路线是否解不出来（1 = 该 κ 上没有有限解）")
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
