"""1.4 章：特征值、伪谱与非正规性（五个可视化的帧装配）。

这一章有五页，分工是"从最温和的迭代到最凶险的几何"：

1. ``power_iteration`` —— 幂法对齐过程。参数里有 **|λ₂/λ₁|**（收敛率，有理论临界值 1）
   与 **非正规强度 shear**。时间轴是**迭代步 $k$**；副轴（对数）画残差与理论斜率
   :math:`|\\lambda_2/\\lambda_1|^k`。文字结论（``status``）会说清 converge / plateau。
2. ``qr_schur`` —— QR 迭代的次对角元衰减热力图（:math:`\\log_{10}|h_{i+1,i}|`）
   + Ritz 值收敛曲线。位移策略 ``none/rayleigh/wilkinson`` 与矩阵族
   （对称 / 非对称 / 循环 / Grcar）是关键对照组：**无位移遇到等模特征值不收敛**。
3. ``pseudospectrum`` —— 本章招牌。复平面网格上的 :math:`\\log_{10}\\sigma_{\\min}(A-zI)`
   场 + 各 :math:`\\varepsilon` 的伪谱"云雾"；副轴画
   :math:`\\rho_\\varepsilon/\\rho(A)` 随 :math:`\\varepsilon` 的曲线
   （正规矩阵是水平线，Jordan 块是按 :math:`\\varepsilon^{1/n}` 的幂律）。
4. ``transient_growth`` —— :math:`\\|A^k\\|_2` 先增后减 vs 谱半径预测的单调衰减；
   副轴给出 Kreiss 常数（= 峰值）与伪谱下界 :math:`(\\rho_\\varepsilon-1)/\\varepsilon`。
5. ``gmres_pseudospec`` —— 为 1.5 章铺路：GMRES 残差平台与伪谱形状的对应。
   时间轴 = **GMRES 迭代步**，副轴是残差历史（对数）。

约定（与 0.0/1.2 章一致）：``meta`` 只放数字/布尔/``None``，字符串一律走 ``strings``；
"时间轴 = 离散步"的可视化把 ``t`` 四舍五入到整数步并夹住，真实用到的步号写进 ``meta.step``。
**本章没有一个 viz 在 server 里做数学**：所有量都来自 :mod:`numcore.linalg.eigen` 与
:mod:`numcore.linalg.pseudospec`。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from numcore.linalg.eigen import (
    arnoldi,
    dense_eigenvalues,
    eigen_residual,
    example_eigen_matrix,
    grcar_matrix,
    hessenberg_reduce,
    jordan_block,
    laplacian_1d,
    power_iteration,
    qr_iteration,
    ratio_matrix,
    shear_matrix,
    spectral_radius_estimate,
)
from numcore.linalg.pseudospec import (
    GMRESResult,
    example_pseudospectrum_matrix,
    gmres,
    gmres_arnoldi_residual_bound,
    kreiss_from_pseudospectrum,
    pseudospectrum_grid,
    sigma_min,
    transient_growth,
)
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 伪谱默认的 ε 序列（跨 10 个量级：从"几乎就是谱"到"整片云雾"）
_EPSILONS: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)

#: 伪谱网格的固定窗口（规则 1/4：**不许**自适应）
_PS_WINDOW = 1.5

MANIFEST: dict[str, Any] = {
    "chapter": "1.4",
    "title": "特征值、伪谱与非正规性",
    "part": "I",
    "summary": (
        "幂法、QR 迭代、Lanczos/Arnoldi 的收敛率，以及非正规矩阵的伪谱与瞬态增长："
        "把「特征值」这一个词拆成谱、伪谱、Kreiss 常数三件事，并让每件都能被测量。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "power_iteration",
            "title": "幂法对齐：向量序列与 Rayleigh 商收敛",
            "description": (
                "单位圆上反复乘 $A$：向量一步步对齐主特征方向。副轴（对数）画残差 "
                "$\\|Ax-\\rho(x)x\\|$ 与两条理论参考线（$|\\lambda_2/\\lambda_1|^{k}$ 与它的平方），"
                "看「特征值比特征向量快一倍」以及 shear 造成的瞬态。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "ratio",
                    "label": "第二特征值比 $|\\lambda_2/\\lambda_1|$",
                    "type": "float",
                    "min": 0.05,
                    "max": 0.99,
                    "step": 0.01,
                    "default": 0.6,
                    "critical": [1.0],
                    "criticalNote": (
                        "幂法收敛定理要求 $|\\lambda_2/\\lambda_1|<1$：**等于 $1$ 时完全不收敛**"
                        "（两个等模特征值，向量在它们张成的平面里打转）；"
                        "越接近 $1$，幂法越慢（迭代数 $\\approx\\log(\\mathrm{tol})/\\log(\\mathrm{ratio})$）。"
                        "滑块上限只到 $0.99$，是因为再往上 $400$ 步也看不出收敛——"
                        "那正是这张图要演的 plateau。"
                    ),
                    "beyond": "above",
                    "help": (
                        "第二特征值与主特征值的模之比 $|\\lambda_2/\\lambda_1|$，没有单位，滑杆范围 0.05 到 0.99。"
                        "它越接近 1，每一步能压掉的误差比例越少，收敛越慢；拖小时残差曲线几乎笔直下坠，"
                        "拖到 0.99 会停在平台上。读图时看副轴的实测残差是否与两条参考线平行，"
                        "以及主轴上的向量是否已经停在一个方向。它与矩阵族联动：diagonal 族直接按这个比值构造，"
                        "cyclic 族用它定旋转块的半径，shear 只改非正规程度、不改这个比值。"
                    ),
                },
                {
                    "name": "shear",
                    "label": "非正规强度 shear",
                    "type": "float",
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "default": 0.0,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "非正规强度，没有单位，默认 0 表示纯对角矩阵。调大后矩阵右上角多出一条剪切耦合："
                        "特征值不变，但开头几十步残差会先抬高再下降，主轴上的向量也可能先被拉向坐标轴；"
                        "调小则逐步回到单调衰减。读图时看副轴残差开头有没有「鼓包」，"
                        "以及鼓包之后是否仍与理论斜率平行。它只对 sheared 族真正生效，"
                        "diagonal 与 cyclic 族会忽略它；调它不会改变 ratio 对应的渐近收敛率。"
                    ),
                },
                {
                    "name": "family",
                    "label": "矩阵族",
                    "type": "enum",
                    "options": ["diagonal", "sheared", "cyclic"],
                    "default": "diagonal",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "选择矩阵的构造方式，决定主特征值与特征向量的几何形状。diagonal 是纯对角矩阵，"
                        "特征值按 $\\mathrm{ratio}^{j}$ 递减；sheared 在对角线上方加剪切，谱不变但特征向量几乎平行；"
                        "cyclic 是二维旋转块，给出一对复共轭特征值。换族会立刻改变主轴的轨迹和副轴的残差形状，"
                        "读图时先看 status 里写的是收敛、平台还是复主特征值。它与 ratio、shear 联动："
                        "diagonal 不看 shear，cyclic 只用 ratio 定半径。"
                    ),
                },
            ],
            "time": {"mode": "scrub", "tMax": 120.0, "dt": 1.0, "label": "迭代步 $k$"},
            "requiredAux": ["residual"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "单位圆与迭代向量（固定坐标 $\\pm1.05$）",
                    "ylabel": "$y$",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "残差与理论收敛率",
                    "ylabel": "$\\|Ax-\\rho(x)x\\|$ 与 $|\\lambda_2/\\lambda_1|^{k}$",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "三条最容易读错的地方。其一：**残差下降 $\\ne$ 特征向量准**——Rayleigh 商是 $O(\\theta^{2})$ "
                "而向量只有 $O(\\theta)$，所以副轴的曲线会比向量早很多步就贴到 $10^{-16}$；"
                "看图上的向量序列时请把 $k$ 拉到收敛前那一段。其二：shear 调大后开头几十步"
                "**看起来像不收敛**（残差先涨后落），这是非正规矩阵的瞬态，不是定理失效——"
                "渐近收敛率仍然是 $|\\lambda_2/\\lambda_1|$（两条参考线的斜率）。其三：主轴的坐标轴固定在 $\\pm 1.05$，"
                "**不会**自适应；若把 $|\\lambda_2/\\lambda_1|$ 拖到 $0.99$，曲线会停在某个平台上，"
                "这张图会诚实地告诉你「没收敛」，而不会把平台画成收敛。"
            ),
            "theory": [
                "**把迭代写成特征方向上的展开。** 设 $A$ 可对角化，特征值按模排序 "
                "$|\\lambda_1|>|\\lambda_2|\\ge\\ldots\\ge|\\lambda_n|$，对应特征向量为 $v_1,\\ldots,v_n$。"
                "把初始向量写成 $x_0=\\alpha_1v_1+\\ldots+\\alpha_nv_n$，并假设 $\\alpha_1\\ne0$。"
                "归一化幂法每一步只做 $x_{k+1}=Ax_k/\\|Ax_k\\|_2$，所以方向完全由 $A^kx_0$ 决定；"
                "下面的所有结论都从这个展开式出发，不需要别的假设。图上 $k=0$ 的点就是 $x_0$，"
                "之后每一段折线表示连续两步之间向量的移动。",
                "**提出公因子后，只剩比值在衰减。** 提取 $\\lambda_1^k$ 得 "
                "$A^kx_0=\\lambda_1^k\\left(\\alpha_1v_1+\\sum_{i\\ge2}\\alpha_i(\\lambda_i/\\lambda_1)^kv_i\\right)$。"
                "括号里第二项按 $|\\lambda_2/\\lambda_1|^k$ 缩小，第一项不变，所以归一化后 $x_k$ 与 $v_1$ 的夹角"
                "以 $|\\lambda_2/\\lambda_1|^k$ 的速度趋于零。严格地说，特征向量矩阵 $V$ 的条件数 $\\kappa(V)$ "
                "会把误差放大成常数倍：$\\mathrm{dist}(x_k,\\operatorname{span}\\{v_1\\})\\le"
                "\\kappa(V)|\\lambda_2/\\lambda_1|^k\\|x_0^{(2:n)}\\|/\\|x_0^{(1)}\\|$。"
                "这解释了为什么 $|\\lambda_2/\\lambda_1|$ 拖到 $0.99$ 时，120 步仍看不出收敛。",
                "**Rayleigh 商把误差平方了一次。** 定义 $\\rho(x)=x^{\\mathrm{T}}Ax/(x^{\\mathrm{T}}x)$。"
                "当 $A$ 正规、$x=v_1+\\delta w$（$w$ 与 $v_1$ 正交、$\\|w\\|=1$）时，展开得 "
                "$\\rho(x)=\\lambda_1+\\delta^2w^{\\mathrm{T}}Aw+O(\\delta^3)$：一次项因正交性消失，误差是 $O(\\delta^2)$。"
                "若 $x_k$ 的方向误差是 $O(r^k)$，那么 $\\rho(x_k)$ 的误差就是 $O(r^{2k})$，其中 $r=|\\lambda_2/\\lambda_1|$。"
                "这就是副轴上两条参考线斜率差一倍的原因：长虚线 $r^k$ 对应残差，短虚线 $r^{2k}$ 对应 Rayleigh 商误差。",
                "**残差衡量的是矩阵作用后还剩多少。** 对任意单位向量 $x$，$\\|Ax-\\rho(x)x\\|_2$ 描述 $x$ 离特征方向有多远。"
                "把 $x$ 分解成 $v_1$ 方向与其余方向，第一项满足 $Av_1-\\lambda_1v_1=0$，剩下的是 $O(r^k)\\|A\\|$ 量级，"
                "所以实测残差曲线应当与 $|\\lambda_2/\\lambda_1|^k$ 平行。注意残差小和向量准不是同一件事："
                "残差量的是 $A$ 作用后的偏差，而方向误差可以大得多；当残差贴到 $10^{-16}$ 时，"
                "向量可能还有 $10^{-8}$ 量级的角度误差。",
                "**非正规矩阵改的是常数，不是斜率。** 在右上角加剪切项后，$A$ 不再正规，$V$ 的条件数变大，"
                "展开式里的常数 $\\kappa(V)$ 可以很大；于是开头几十步残差会先升后降，主轴上向量也可能先被拉向某个坐标方向。"
                "但渐近斜率仍然是 $|\\lambda_2/\\lambda_1|$，因为极限只由模最大的两个特征值决定。"
                "把 shear 调到 $5$ 时，副轴前段的「鼓包」会明显变大，后段仍与长虚线平行；"
                "这不是收敛定理失效，而是常数项在有限步内的可见后果。图上 $k=0$ 到 $k\\approx20$ 这一段最容易被误读成「发散了」。",
                "**复共轭对让实向量打转。** 选 cyclic 族时 $A$ 是二维旋转块，特征值为 $r e^{\\pm i\\theta}$，"
                "两个模相等，因此 $|\\lambda_2/\\lambda_1|=1$，定理的前提被破坏。实向量每一步旋转固定角度，"
                "永远不落在一个实特征方向上；副轴残差停在平台，主轴的点沿圆周滑动。此时不是「迭代坏了」，"
                "而是没有唯一的实主方向。读图时看 status 是否写着复主特征值、夹角是否保持常数；"
                "把 family 换回 diagonal 才会重新出现单调收敛。这个对照也解释了为什么 ratio 的上限设在 $0.99$，而不是 $1$。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "qr_schur",
            "title": "QR 迭代：次对角元衰减热力图与 Ritz 值",
            "description": (
                "热力图是 $\\log_{10}|h_{i+1,i}|$（次对角元），横轴迭代步、纵轴下标 $i$："
                "格点变黑就是那一条次对角元衰减到零（deflation）。"
                "副轴画 Ritz 值与真谱的距离（对数），三条曲线对应三种位移策略。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "family",
                    "label": "矩阵族",
                    "type": "enum",
                    "options": ["symmetric", "nonsymmetric", "cyclic", "grcar"],
                    "default": "symmetric",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "选择被迭代的矩阵族，决定谱的形状和热力图的难易。symmetric 是一维 Laplacian，"
                        "特征值全是实数且分得很开；nonsymmetric 在对角之外加了剪切，谱可能是复数；"
                        "cyclic 由若干二维旋转块组成，特征值成对等模；grcar 是上 Hessenberg 矩阵，"
                        "特征值模长几乎相等。换族要同时看热力图哪一列先变黑和副轴的谱距离，"
                        "grcar 通常最需要位移。它与 n、shift 联动：阶数越大、谱越挤，无位移 QR 越难收敛。"
                    ),
                },
                {
                    "name": "shift",
                    "label": "位移策略",
                    "type": "enum",
                    "options": ["none", "rayleigh", "wilkinson"],
                    "default": "wilkinson",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "选择 QR 迭代的位移策略，没有单位。none 不做位移，次对角元按相邻特征值的模比衰减，"
                        "遇到等模特征值就停住；rayleigh 用右下角元素作位移，对对称矩阵通常三次收敛；"
                        "wilkinson 用右下 $2\\times2$ 块的两个特征值中较近的那个，能处理复共轭对并给出块收敛。"
                        "读图时对照三种策略的谱距离曲线与热力图的变黑时刻。它与 family 联动："
                        "grcar 和 cyclic 上位移带来的差别最大。"
                    ),
                },
                {
                    "name": "n",
                    "label": "阶数 n",
                    "type": "int",
                    "min": 4,
                    "max": 16,
                    "default": 8,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "矩阵阶数，取整数 4 到 16，没有单位。调大后热力图多出几行次对角元、每步代价按 $n^3$ 增长，"
                        "谱更容易挤在一起，迭代步数通常变多；调小则行数少、曲线短，单个特征值的偶然误差更显眼。"
                        "读图时先数纵轴行数（它等于 $n-1$），再看块结构是否只剩一个 $2\\times2$ 块。"
                        "它与 family、shift 联动：同样的位移在更大、更挤的谱上未必够用。"
                    ),
                },
            ],
            "time": {"mode": "scrub", "tMax": 60.0, "dt": 1.0, "label": "QR 迭代步"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "$\\log_{10}|h_{i+1,i}|$（行 = $i$，列 = 迭代步）",
                    "ylabel": "次对角元下标 $i$",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "Ritz 值到真谱的距离",
                    "ylabel": "$\\max_i\\min_j|\\theta_i-\\lambda_j|$",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "这张图最会骗人的地方是**纵轴只有 $n-1$ 行**：一条次对角元「变黑」代表那一小段"
                "三角块已经锁定了一个特征值，而不是「特征值算出来了」——上面还没收敛的行"
                "照样可以让 Ritz 值看起来不错。第二条：把 shift 切到 none 再看复共轭特征值"
                "（cyclic 族）：热力图**整片都不变黑**，可副轴的误差并不会爆炸——"
                "「不收敛」不等于「算错」，它只是停在一个 $2\\times2$ 块上（图上用块收敛标记指出）。"
                "第三条：热力图用 $\\log_{10}$ 上色，$10^{-2}$ 和 $10^{-16}$ 只是深浅不同；"
                "判断「真的收敛了」请看 meta 里的 converged 与 block_sizes，别只看颜色。"
            ),
            "theory": [
                "**先把矩阵化成 Hessenberg 形。** QR 迭代从 $A_0=A$ 开始，每步做分解 $A_k=Q_kR_k$，"
                "再令 $A_{k+1}=R_kQ_k$。因为 $Q_k$ 正交，$A_{k+1}=Q_k^{\\mathrm{T}}A_kQ_k$ 是相似变换，"
                "特征值一位都不变。实际计算先做一次 Householder 化，把 $A$ 变成上 Hessenberg 形"
                "（次对角以下全为零），这一步代价 $O(n^3)$；之后每步只需 $O(n^2)$。图中热力图的纵轴"
                "正是 Hessenberg 形的次对角元下标，横轴是迭代步，颜色是 $\\log_{10}|h_{i+1,i}|$。",
                "**无位移 QR 的收敛率来自幂法。** 把各步的 $Q$ 乘起来，可以得到 $A^k=Q^{(k)}R^{(k)}$，"
                "并且 $A_k=Q^{(k)\\mathrm{T}}AQ^{(k)}$。当 $|\\lambda_1|>\\ldots>|\\lambda_n|$ 时，"
                "$A^k$ 的列按 $|\\lambda_i/\\lambda_1|^k$ 分层，$Q^{(k)}$ 的前 $i$ 列张成的子空间趋于前 $i$ 个"
                "特征向量张成的子空间。于是 $A_k$ 的第 $(i+1,i)$ 个次对角元按 $|\\lambda_{i+1}/\\lambda_i|^k$ 衰减。"
                "这条推导也说明了失效条件：只要有两个特征值模相等，对应的比值等于 $1$，那个次对角元就永远不衰减。",
                "**位移把「比值」换成「到位移的距离」。** 取位移 $\\mu_k$，对 $A_k-\\mu_kI$ 做 QR 分解，"
                "再令 $A_{k+1}=R_kQ_k+\\mu_kI$。这等价于在 $A_k$ 上做一次反迭代：收敛快慢由 "
                "$|\\lambda_{\\mathrm{near}}-\\mu_k|/|\\lambda_{\\mathrm{next}}-\\mu_k|$ 决定，"
                "离 $\\mu_k$ 最近的特征值被迅速放大。Rayleigh 位移取 $\\mu_k=(A_k)_{nn}$，"
                "当它接近某个特征值时误差按平方缩小；对称矩阵上还可证明三次收敛。Wilkinson 位移取右下 "
                "$2\\times2$ 块的两个特征值中离 $(A_k)_{nn}$ 较近者，因此在复共轭对附近也能稳定工作。",
                "**复共轭对只能停在 $2\\times2$ 块上。** 实矩阵的复特征值成对出现，实 QR 迭代无法把一对"
                "复共轭数拆成两个实数对角元；它们会留在一个 $2\\times2$ 块里，块下面的次对角元趋于零，"
                "块内部的两个次对角元不趋于零。这就是「块收敛」：热力图上整列变黑，只有 $2\\times2$ 块对应的"
                "位置还亮着。状态里写 block-converge 时不要读成「卡住了」；真正的收敛判据是块外的次对角元足够小，"
                "而不是颜色全黑。",
                "**数值极限与代价。** 收敛判据通常写成 "
                "$|h_{i+1,i}|\\le\\mathrm{tol}(|h_{ii}|+|h_{i+1,i+1}|)$，因为它要衡量的是相对大小；"
                "本图用 $\\log_{10}$ 上色，$10^{-16}$ 以下统一夹到色阶底，所以「全黑」只说明到了机器精度，"
                "不说明特征值算得准。每次位移都要解一个 $2\\times2$ 小问题并做一次 QR，位移选得不好会让收敛变慢，"
                "但不会改变谱。Hessenberg 化是一次性的，迭代阶段每步 $O(n^2)$；这也是大矩阵不直接对稠密矩阵做 QR 的原因。",
                "**读图时把三条信息分开。** 热力图回答「哪一条次对角元在什么时候衰减」，副轴回答「当前 Ritz 值离真谱多远」，"
                "状态文字回答「是否整体收敛」。三者不能互相替代：一条次对角元变黑只锁定了一个特征值，"
                "其他行可能还没动；副轴的距离可能因为某个 Ritz 值已经对准而先降。把 family 换成 grcar、"
                "shift 换成 none 再看一次，能直接看到整片热力图不变黑而谱距离停止下降——"
                "这正是无位移 QR 在等模谱上的典型失败。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "pseudospectrum",
            "title": "$\\varepsilon$-伪谱云雾：$\\sigma_{\\min}(A-zI)$ 的等高线",
            "description": (
                "复平面网格上画 $\\log_{10}\\sigma_{\\min}(A-zI)$；每个 $\\varepsilon$ 给一条 marching-squares 轮廓。"
                "主轴**固定**在 ${[-1.5,1.5]}^{2}$（不随 $\\varepsilon$ 自适应）；副轴画 $\\rho_\\varepsilon/\\rho(A)$ 随 $\\varepsilon$ 的曲线"
                "（正规矩阵水平，Jordan 块是 $\\varepsilon^{1/n}$ 幂律）。灰点标出真谱。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "family",
                    "label": "矩阵族",
                    "type": "enum",
                    "options": ["normal", "jordan", "shifted_jordan", "shear", "grcar", "companion"],
                    "default": "jordan",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "选择矩阵族，决定伪谱云雾的形状。normal 是正规矩阵，伪谱严格是若干 "
                        "$\\varepsilon$-圆盘的并；jordan 是单个 Jordan 块，特征值只有一个却拖出一大片花瓣；"
                        "shifted_jordan 把花瓣平移到复平面另一处；shear 的谱全在 $1$ 附近但非正规性最强；"
                        "grcar 的谱聚成一簇、伪谱向右铺开；companion 的根分得很散、伪谱却连成一片。"
                        "换族后先看灰点（真谱）与云雾的偏离方向，再看副轴比值是否水平。它与 n、epsilon 联动："
                        "n 决定 Jordan 块花瓣的幂次，epsilon 决定看哪一层轮廓。"
                    ),
                },
                {
                    "name": "n",
                    "label": "阶数 n",
                    "type": "int",
                    "min": 4,
                    "max": 14,
                    "default": 10,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "矩阵阶数，取整数 4 到 14，没有单位。对 jordan 族，伪谱半径按 $\\varepsilon^{1/n}$ 增长，"
                        "n 越大花瓣伸得越远；对 normal 族，n 只增加圆盘的个数，不改变单个圆盘的大小；"
                        "对 grcar 等族，n 越大谱越挤、云雾越宽。调大 n 时网格还是固定的 $32\\times32$，"
                        "窄结构可能采不到，图上会出现「抓不到」的空白。它与 family、epsilon 联动，"
                        "读图时看副轴幂律斜率是否接近 $1/n$。"
                    ),
                },
                {
                    "name": "epsilon",
                    "label": "当前 $\\varepsilon$（轮廓高亮）",
                    "type": "float",
                    "min": 1e-8,
                    "max": 1e-1,
                    "step": 1e-8,
                    "default": 1e-4,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "当前高亮的伪谱水平，范围 $10^{-8}$ 到 $10^{-1}$，没有单位。它选中哪一条轮廓线、"
                        "副轴游标停在哪里：调小后轮廓收紧、贴近真谱，但 $\\sigma_{\\min}$ 更容易掉进舍入噪声；"
                        "调大后云雾整片扩张，比值上升。读图时看高亮轮廓与灰点的距离，"
                        "以及副轴曲线在该 $\\varepsilon$ 处是否还在可信区。它与 family、n 联动："
                        "正规族在任何 $\\varepsilon$ 下都该是圆盘并，Jordan 族的半径按 $\\varepsilon^{1/n}$ 变。"
                    ),
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "复平面（固定窗口）",
                    "ylabel": "$\\mathrm{Im}(z)$",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "伪谱半径与谱半径之比",
                    "ylabel": "$\\rho_\\varepsilon/\\rho(A)$",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "伪谱图有四层骗法。第一：**坐标轴必须是固定的**——一旦自适应，"
                "「云雾扩散」这个现象会被抹平成「图还是那么大」（这也是 axes.autoscaled=false "
                "写在契约里的原因）。第二：$\\varepsilon$ 小的那几层（$\\le 10^{-7}$）在 float64 里**测不准**："
                "$\\sigma_{\\min}$ 用 $M^{\\mathrm{H}}M$ 的谱算，条件数被平方，$\\kappa(M)\\ge 10^{8}$ 之后 $\\sigma_{\\min}$ 就掉进舍入噪声，"
                "此时图上那片「云雾」有一半是数值噪声——帧里的 trusted 掩码把不可信格点标出来，"
                "**看到没上色的区域不要当成「不在伪谱里」**。第三：正规矩阵（normal 族）"
                "的伪谱严格等于 $\\varepsilon$-圆盘的并，所以 $\\rho_\\varepsilon/\\rho(A)$ 是一条水平线；如果你在那里看到"
                "「扩散」，那是网格/估计误差，不是物理。第四：companion 族的根分得很开，"
                "伪谱却把它们连成一片——别把「谱点很分散」读成「问题良态」。"
            ),
            "theory": [
                "**伪谱的定义与三种等价说法。** 对 $n$ 阶复矩阵 $A$，定义 "
                "$\\Lambda_\\varepsilon(A)=\\{z:\\sigma_{\\min}(A-zI)\\le\\varepsilon\\}$。"
                "$\\sigma_{\\min}(A-zI)$ 是 $A-zI$ 到奇异矩阵的距离，所以 $z$ 落在伪谱里，"
                "等价于存在一个范数不超过 $\\varepsilon$ 的扰动 $\\delta A$，使 $z$ 成为 $A+\\delta A$ 的特征值。"
                "也等价于预解式范数满足 $\\|(zI-A)^{-1}\\|\\ge\\varepsilon^{-1}$。这三个说法是同一件事的三种用法："
                "定义最好算，扰动说法最好解释，预解式说法最适合推界。",
                "**正规矩阵的伪谱就是圆盘的并。** 若 $A=U\\Lambda U^{\\mathrm{H}}$ 且 $U$ 酉，"
                "则 $A-zI=U(\\Lambda-zI)U^{\\mathrm{H}}$ 仍正规，奇异值就是 $|\\lambda_i-z|$，"
                "于是 $\\sigma_{\\min}(A-zI)=\\min_i|\\lambda_i-z|$。因此 $\\Lambda_\\varepsilon(A)$ "
                "正好是每个特征值周围半径 $\\varepsilon$ 的圆盘的并，伪谱半径 "
                "$\\rho_\\varepsilon=\\rho(A)+\\varepsilon$，副轴上的比值 $(\\rho+\\varepsilon)/\\rho$ 是一条水平线。"
                "图中 normal 族如果看到云雾「扩散」得不均匀，那是网格或估计误差，不是数学。",
                "**Jordan 块给出 $\\varepsilon^{1/n}$ 的幂律。** 取 $J_n(0)$，它只有一个特征值 $0$。"
                "对 $z\\ne0$，方程 $(J-zI)y$ 的分量是 $-zy_i+y_{i+1}$；选 "
                "$y=(1,z,\\ldots,z^{n-1})$ 时，除最后一个分量外全部抵消，只剩 $|z|^n$。"
                "归一化后 $\\sigma_{\\min}(J-zI)\\approx|z|^n/\\sqrt{1+|z|^2+\\ldots+|z|^{2(n-1)}}$，"
                "在 $|z|$ 很小时近似 $|z|^n$。令它等于 $\\varepsilon$ 得 "
                "$\\rho_\\varepsilon\\approx\\varepsilon^{1/n}$：$n=10$ 时 $\\varepsilon$ 缩小 $10^{10}$ 倍，"
                "半径才缩小 $10$ 倍。这就是图上花瓣又大又圆的原因。",
                "**非正规矩阵把谱「抹开」。** 对可对角化的 $A=V\\Lambda V^{-1}$，若 $\\mu$ 是 $A+\\delta A$ 的特征值，"
                "则 $\\mu$ 到谱的距离不超过 $\\kappa(V)\\|\\delta A\\|$。正规矩阵 $\\kappa(V)=1$，扰动只把谱移动 $\\varepsilon$；"
                "非正规矩阵 $\\kappa(V)$ 可以是指数级大，$\\varepsilon$ 的扰动就能把谱推到很远。伪谱把这个上界变成了"
                "精确的等值线：shear 与 grcar 的云雾之所以连成一片，不是因为特征值算错了，而是特征向量几乎平行，"
                "预解式在离谱很远的地方仍然很大。",
                "**数值上要区分信号与舍入。** 本图用 $M^{\\mathrm{H}}M$ 的谱来算 $\\sigma_{\\min}$，"
                "条件数被平方，误差大致按 $u\\kappa(M)^2$ 增长；当 $\\sigma_{\\min}$ 低于 $\\sqrt{u}\\|A\\|$ 时，"
                "算出来的值已经主要是噪声。图上的灰格就是这类点，它们不代表「不在伪谱里」，只代表「这里量不准」。"
                "另一个极限是网格：$32\\times32$ 的步长固定，$\\varepsilon$ 很小时轮廓可能落在两个格点之间，"
                "看起来像空的。读图先看可信比例与网格间距，再看形状。",
                "**参数与曲线的对应。** 主轴固定为 $[-1.5,1.5]^2$，所以云雾的扩张不会被坐标轴缩放掩盖；"
                "灰点是真谱，彩色线是不同 $\\varepsilon$ 的边界，高亮那条由 epsilon 参数选中。"
                "副轴画 $\\rho_\\varepsilon/\\rho(A)$：normal 族是水平线，Jordan 族是斜率为 $1/n$ 的直线，"
                "非正规族介于两者之间。把 n 调大、epsilon 调小，可以看到花瓣变大、轮廓向谱收缩；"
                "但别忘了同时看可信格点比例——小 $\\varepsilon$ 的那几层最容易变成噪声。"
                "常见误读是把等高线密度当成「边界更胖」；每条线各自对应一个 $\\varepsilon$，线之间没有插值意义。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "transient_growth",
            "title": "瞬态增长：$\\|A^k\\|$ 先涨后落，谱半径单调衰减",
            "description": (
                "副轴（对数）画实测 $\\|A^{k}\\|_2$ 与谱半径预测 $\\rho(A)^{k}$；主轴画两者的比值（增长倍数）。"
                "参数里 nonnormal 就是剪切强度：谱半径严格小于 $1$，$\\|A^{k}\\|$ 却能先涨三个量级。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "family",
                    "label": "矩阵族",
                    "type": "enum",
                    "options": ["shear", "jordan_shifted", "grcar", "normal"],
                    "default": "shear",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "选择矩阵族，决定瞬态增长有多强。shear 是单位阵加上三角剪切，所有特征值都等于 $1$，"
                        "谱半径说「不增不减」，但范数能先涨很多；jordan_shifted 把 Jordan 块平移到 $0.75$，"
                        "既有衰减又有巨大峰值；grcar 是上 Hessenberg 矩阵，谱半径小于 $1$ 却能在有限步内放大；"
                        "normal 是对角矩阵，几乎没有瞬态。换族后先看主轴的增长倍数峰值和副轴实测曲线是否先升后降。"
                        "它与 nonnormal、n、k_max 联动：剪切越强、阶数越高，峰值越大也越靠后。"
                    ),
                },
                {
                    "name": "nonnormal",
                    "label": "非正规强度 s（剪切/耦合）",
                    "type": "float",
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "default": 1.0,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "非正规强度 $s$，没有单位，范围 0 到 3。调大后剪切或耦合项变强，谱半径基本不变，"
                        "但有限步的范数峰值按 $s$ 的幂次迅速变大，峰值位置也会移动；调到 0 时矩阵接近正规，"
                        "实测曲线基本贴着谱半径预测。读图时看主轴增长倍数的峰顶、副轴两条曲线的最大间距，"
                        "以及峰值标记对应的 $k$。它与 n 联动得最强：$n$ 阶剪切矩阵的峰值约按 $s^{n-1}$ 增长，"
                        "阶数越大越不能把 $s$ 当小量。"
                    ),
                },
                {
                    "name": "n",
                    "label": "阶数 n",
                    "type": "int",
                    "min": 4,
                    "max": 14,
                    "default": 10,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "矩阵阶数，取整数 4 到 14，没有单位。阶数越大，剪切或 Jordan 耦合能累积的步数越多，"
                        "瞬态峰值通常越高、出现得越晚；阶数小则峰值温和，曲线很快进入衰减。调它还会改变谱的间距与"
                        "条件数，因此副轴的渐近斜率也可能变。读图时把峰值位置与 k_max 对比，"
                        "并看主轴峰值是否已经越过右边界。它与 nonnormal、k_max 联动：大 $n$ 配小 k_max 时"
                        "最容易被截断在上升段。"
                    ),
                },
                {
                    "name": "k_max",
                    "label": "时间上限 $k_{\\max}$",
                    "type": "int",
                    "min": 10,
                    "max": 80,
                    "default": 40,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "计算到第几步的上限，取整数 10 到 80，没有单位。调大后能看到完整的「先涨后落」，"
                        "峰值标记落在曲线内部，右端斜率逐渐接近 $\\log_{10}\\rho(A)$；调小则只看到上升段，"
                        "峰值被右边界截断，图上像是「还在涨」。读图时看峰值标记是否贴着右端、"
                        "状态文字是否提示把上限拖大。它与 family、nonnormal 联动：非正规性越强、阶数越大，"
                        "需要越大的上限才看得到峰顶。"
                    ),
                },
            ],
            "time": {"mode": "scrub", "tMax": 40.0, "dt": 1.0, "label": "幂次 $k$"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": True,
                    "title": "增长倍数 $\\|A^{k}\\|_2/\\rho(A)^{k}$",
                    "ylabel": "倍数",
                    "scale": "log",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "$\\|A^{k}\\|_2$ 与 $\\rho(A)^{k}$",
                    "ylabel": "$\\|\\cdot\\|_2$",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "最重要的一条：**谱半径小于 $1$ 不保证 $\\|A^{k}\\|$ 单调下降**。把 nonnormal 拖到 $1$ 以上，"
                "副轴上那条「预测」直线一路向下，而实测曲线先冲高再回落——图上那个峰值就是"
                "Kreiss 常数，它可以比 $1$ 大好几个量级。第二条：峰值位置与大小**不是**由特征值决定的："
                "shear 族的谱全是 $1$（谱半径 $1$），峰值却随 $n$ 与 $s$ 指数增长；"
                "所以「特征值都在单位圆内 $\\Rightarrow$ 稳定」这句话在非正规问题上只是「最终稳定」。"
                "第三条：k_max 太小会**看不见**峰值（峰值可能出现在 $k\\approx n/2$ 之后），"
                "meta 里的 peak_index 如果贴着 k_max，说明还没到头，请把 k_max 拖大再看。"
                "第四条：$\\|A^{k}\\|$ 是用幂法估的，$\\kappa$ 大时它只是量级可信（实测与 SVD 差 $<10^{-15}$ 相对，"
                "但那是在 $n=10$ 的规模上；更大规模请以参考层为准）。"
            ),
            "theory": [
                "**谱半径只描述 $k\\to\\infty$。** 设 $A$ 的特征值为 $\\lambda_1,\\ldots,\\lambda_n$，"
                "谱半径 $\\rho(A)=\\max_i|\\lambda_i|$。由 Jordan 标准形，$A^k$ 的每个元素都是形如 "
                "$\\frac{k!}{j!(k-j)!}\\lambda_i^{k-j}$ 的项之和，因此对任意 $\\varepsilon>0$ 存在常数 $C$，"
                "使 $\\|A^k\\|_2\\le C(\\rho(A)+\\varepsilon)^k$。这就是 Gelfand 公式 "
                "$\\rho(A)=\\lim_{k\\to\\infty}\\|A^k\\|^{1/k}$ 的定量版本。它只保证足够大的 $k$ 之后曲线被指数包住，"
                "对 $k=5$ 或 $k=20$ 没有任何约束。",
                "**Jordan 块把「足够大」推得很远。** 对 $J_n(0)$，当 $k\\ge n$ 时 $J_n(0)^k=0$，谱半径是 $0$，"
                "范数却在 $k<n$ 时等于 $1$。更一般地，$J_n(\\lambda)=\\lambda I+N$，其中 $N$ 是幂零上三角，"
                "二项式展开给出 $(\\lambda I+N)^k=\\sum_{j=0}^{n-1}\\frac{k!}{j!(k-j)!}\\lambda^{k-j}N^j$。"
                "当 $|\\lambda|<1$ 时，$\\lambda^{k-j}$ 负责最终衰减，而二项式系数在早期随 $k$ 多项式增长；"
                "两者竞争的结果就是先涨后落。峰值大约出现在 $k\\approx(n-1)/(1-|\\lambda|)$ 附近，"
                "阶数越高、衰减越慢，峰值越靠后也越高。",
                "**Kreiss 常数把峰值量化。** 定义 $K(A)=\\sup_{k\\ge0}\\|A^k\\|_2$。由预解式展开 "
                "$(zI-A)^{-1}=\\sum_{k\\ge0}A^k/z^{k+1}$（$|z|>\\|A\\|$），可得 "
                "$\\|(zI-A)^{-1}\\|\\le K(A)/(|z|-1)$，于是 "
                "$K(A)\\ge\\sup_{|z|>1}(|z|-1)\\|(zI-A)^{-1}\\|$。反方向是 Kreiss 矩阵定理的深刻部分，"
                "结论是两边相等。图上的峰值就是 $K(A)$ 的实测值；它可以是 $10^{10}$ 量级，而谱半径仍然等于 $1$。"
                "这也说明「谱半径小于 $1$ 就稳定」只在 $k\\to\\infty$ 的意义上成立。",
                "**伪谱给出可操作的判据。** 若伪谱 $\\Lambda_\\varepsilon(A)$ 整个落在单位圆内，"
                "即 $\\rho_\\varepsilon(A)\\le1$，则对一切 $k$ 有 "
                "$\\|A^k\\|_2\\le\\rho_\\varepsilon(A)^{k+1}/\\varepsilon\\le1/\\varepsilon$，"
                "瞬态被 $1/\\varepsilon$ 控制；反之，若 $\\|A^k\\|$ 先涨到 $M$，则必有 "
                "$\\rho_\\varepsilon(A)>1$ 对某个 $\\varepsilon$ 成立。把 $\\varepsilon$ 取得很小，"
                "$\\rho_\\varepsilon-1$ 就是伪谱鼓出单位圆的距离。图中 shear 族的谱全在 $1$ 上，但伪谱向右鼓出，"
                "所以范数能先涨后落；normal 族的伪谱就是圆盘并，鼓不出单位圆，曲线单调。",
                "**参数怎样改变曲线。** family 决定矩阵的谱与非正规结构：shear 的谱全等于 $1$，"
                "所以谱预测是水平线；jordan_shifted 与 grcar 的谱半径小于 $1$，谱预测向下，而实测先升。"
                "nonnormal 是剪切强度，调大后峰值按幂次增长；n 是阶数，阶数越高，二项式系数累积得越多，"
                "峰值越靠后；k_max 是观察窗口，太小就看不到峰顶。读图时把主轴增长倍数、副轴实测与预测、"
                "误差场的差值三处对齐：它们从三个角度描述同一件事。",
                "**常见误读与数值边界。** 第一，把 $\\rho(A)<1$ 读成「每一步都缩小」：图上的实测曲线会在"
                "开头几十步上升，这是非正规性，不是实现错误。第二，把峰值标记贴着右端当成真峰值："
                "那说明 k_max 不够大，峰值可能还在后面。第三，把 $\\|A^k\\|$ 的峰值当成「最终会发散」："
                "只要 $\\rho(A)<1$，$k$ 足够大后一定衰减，峰值只是有限步现象。第四，阶数很大时范数计算本身"
                "可能溢出，图上会用状态文字说明；这时应减小 k_max 或阶数，而不是相信被截断的曲线。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "gmres_pseudospec",
            "title": "GMRES 残差平台与伪谱形状（1.5 章前哨）",
            "description": (
                "同一矩阵上：左轴是伪谱云雾与其 $\\varepsilon$ 轮廓（固定窗口），右轴是 GMRES 相对残差历史"
                "（对数）以及「只依赖谱」的多项式下降界。时间轴 = GMRES 迭代步。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "family",
                    "label": "矩阵族",
                    "type": "enum",
                    "options": ["grcar", "shear", "jordan_shifted", "normal"],
                    "default": "grcar",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "选择矩阵族，决定伪谱形状与 GMRES 平台的高度。grcar 是上 Hessenberg 矩阵，"
                        "谱聚成一簇、伪谱很大，最容易出现平台；shear 的谱全在 $1$ 附近，非正规性最强；"
                        "jordan_shifted 是亏损矩阵，伪谱围绕一个点铺开；normal 是对角矩阵，伪谱就是圆盘并，"
                        "平台基本消失。换族后先看主轴云雾包不包住原点，再看副轴残差与「只看谱」的界差多远。"
                        "它与 rhs、n、epsilon 联动。"
                    ),
                },
                {
                    "name": "rhs",
                    "label": "右端项 b",
                    "type": "enum",
                    "options": ["ones", "ramp", "alternating", "e1"],
                    "default": "ones",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "右端项 $b$ 的取法，决定初始残差落在哪些特征方向上。ones 是常数向量，"
                        "常常漏掉奇对称模态；ramp 是线性渐变；alternating 是正负交替的高频向量；"
                        "e1 是单位脉冲，激发最宽。换 $b$ 会改变平台的起点、高度和掉落时刻，"
                        "所以只测一个 $b$ 就下结论很危险。读图时比较不同 $b$ 下副轴残差曲线的形状，"
                        "并看它是否仍在「只看谱」的界下方。它与 family、n 联动。"
                    ),
                },
                {
                    "name": "n",
                    "label": "阶数 n",
                    "type": "int",
                    "min": 6,
                    "max": 14,
                    "default": 10,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "矩阵阶数，取整数 6 到 14，没有单位。GMRES 最多跑 $\\min(24,n)$ 步，"
                        "所以 n 同时决定问题的维数和可用的迭代步数；n 越大，谱越宽、伪谱越大，"
                        "平台更容易出现且持续更久。调小时维数低，GMRES 可能几步就到机器精度，"
                        "平台还没来得及形成。读图时看 Krylov 维数与总步数，以及残差最后是否落到地板。"
                        "它与 family、rhs 联动。"
                    ),
                },
                {
                    "name": "epsilon",
                    "label": "轮廓 $\\varepsilon$",
                    "type": "float",
                    "min": 1e-6,
                    "max": 1e-1,
                    "step": 1e-6,
                    "default": 1e-3,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "当前高亮的伪谱水平，范围 $10^{-6}$ 到 $10^{-1}$，没有单位。它决定主图上哪一条轮廓被加粗，"
                        "以及伪谱半径标记落在哪里；调小后轮廓贴近谱，但 $28\\times28$ 的粗网格常常抓不到，"
                        "会出现空白；调大后轮廓向外扩，更容易看见。读图时看高亮轮廓是否包住原点，"
                        "以及状态文字里的伪谱半径。它与 family、n 联动：非正规性越强、阶数越大，"
                        "同一 $\\varepsilon$ 下的轮廓越大。"
                    ),
                },
            ],
            "time": {"mode": "scrub", "tMax": 24.0, "dt": 1.0, "label": "GMRES 迭代步"},
            "requiredAux": ["residual"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "伪谱（固定窗口）与谱",
                    "ylabel": "$\\mathrm{Im}(z)$",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "GMRES 相对残差（对数）",
                    "ylabel": "$\\|r_k\\|/\\|r_0\\|$",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "GMRES 的残差**单调不增**（它每步都在扩大的 Krylov 子空间里取最小残差），"
                "所以副轴上任何一次上翘都说明实现/读图错了。真正会骗人的是「平台」："
                "残差在 $10^{-1}$ 附近卡住十几步再突然掉下去，看起来像收敛停滞，"
                "其实是特征向量矩阵病态（非正规）导致「迹在谱上的多项式」这个界失效——"
                "meta 里的 plateau 标志与「只依赖谱」的下降界之间的**巨大落差**才是要看的量。"
                "另一条：把 family 换成 normal 后，平台消失、下降界与实测几乎重合；"
                "同一张图上对比这两族，比任何文字都清楚。还有：这里的 $b$ 会影响平台高度与位置，"
                "所以要换 $b$ 看稳定性——只测一个 $b$ 就下结论是这类实验最常见的错误。"
            ),
            "theory": [
                "**GMRES 在 Krylov 子空间里挑最小残差。** 设 $x_0$ 是初值、$r_0=b-Ax_0$，"
                "第 $k$ 个 Krylov 子空间是 "
                "$\\mathcal{K}_k=\\operatorname{span}\\{r_0,Ar_0,\\ldots,A^{k-1}r_0\\}$。"
                "GMRES 令 $x_k=x_0+z$、$z$ 属于 $\\mathcal{K}_k$，并使 $\\|b-Ax_k\\|_2$ 最小。"
                "由于 $x_k-x_0=q(A)r_0$（$q$ 的次数不超过 $k-1$），残差可以写成 "
                "$r_k=(I-Aq(A))r_0=p(A)r_0$，其中 $p(0)=1$、次数不超过 $k$。于是 "
                "$\\|r_k\\|_2=\\min_{p(0)=1}\\|p(A)r_0\\|_2$，这就是 GMRES 的多项式刻画。",
                "**Arnoldi 把最小化变成最小二乘。** 从 $r_0/\\|r_0\\|$ 出发做 Arnoldi，得到列正交的 $V_k$ "
                "与上 Hessenberg 矩阵 $\\bar H_k$，满足 $AV_k=V_{k+1}\\bar H_k$。把 $x=x_0+V_ky$ 代入，"
                "利用 $r_0=\\beta v_1$ 得 $\\|b-Ax\\|_2=\\|\\beta e_1-\\bar H_ky\\|_2$。"
                "右边是一个 $(k+1)\\times k$ 的最小二乘问题，用 Givens 旋转把 $\\bar H_k$ 化成上三角即可递推求解；"
                "每步代价 $O(k^2n)$，存储 $O(kn)$。副轴上那条实测残差就是递推得到的 $\\|r_k\\|$。",
                "**只看谱的界从哪里来。** 若 $A$ 可对角化，$A=V\\Lambda V^{-1}$，则 $p(A)=Vp(\\Lambda)V^{-1}$，"
                "于是 $\\|p(A)r_0\\|_2\\le\\kappa_2(V)\\max_{\\lambda}|p(\\lambda)|\\|r_0\\|_2$"
                "（最大值取在谱上）。对两边取最小，得到 "
                "$\\|r_k\\|/\\|r_0\\|\\le\\kappa_2(V)\\min_{p(0)=1}\\max_{\\lambda}|p(\\lambda)|$。"
                "当 $A$ 正规时 $V$ 可以取酉矩阵，$\\kappa_2(V)=1$，这个界完全由谱决定；"
                "非正规时 $\\kappa_2(V)$ 可以是指数级大，界就失去预测力。",
                "**伪谱解释平台。** 预解式与伪谱的关系是 $\\sigma_{\\min}(A-zI)\\le\\varepsilon$ 等价于 "
                "$\\|(zI-A)^{-1}\\|\\ge\\varepsilon^{-1}$。如果伪谱 $\\Lambda_\\varepsilon(A)$ 把原点包在里面，"
                "那么任何满足 $p(0)=1$ 的多项式都不可能在整片伪谱上很小：它在 $0$ 处必须取 $1$，"
                "而伪谱在 $0$ 附近给了它很大的「活动空间」。定量地说，GMRES 的残差可以用伪谱上的最大值来控制，"
                "伪谱越大、越靠近原点，平台越高、持续越久。grcar 与 shear 族的平台就是这样来的；"
                "normal 族的伪谱只是圆盘并，通常不包住原点，所以界与实测贴合。",
                "**参数与三条曲线的对应。** family 改变矩阵，从而改变谱与伪谱；rhs 改变 $r_0$，"
                "从而改变多项式在谱上的取值方式；n 改变维数与最多步数；epsilon 只改变主图高亮的轮廓，"
                "不改变 GMRES 本身。副轴有三条线：实线是实测残差，长虚线是只依赖谱的界，点线是对数地板。"
                "实测残差单调不增，而「只看谱」的界可以随 $k$ 上升甚至大于 $1$；两者之间的落差就是非正规性的代价。"
                "把 family 从 grcar 换到 normal，落差会明显缩小。",
                "**读图与数值极限。** 第一，残差曲线若出现上翘，那是实现或读图错误，因为 GMRES 在精确算术下"
                "单调不增；舍入可能造成极小的回升，但不会改变趋势。第二，平台不是「收敛停滞」，"
                "而是多项式暂时找不到同时在谱上小、又满足 $p(0)=1$ 的解；继续迭代往往会突然掉落。"
                "第三，谱点落在固定窗口外时会被裁剪，图上用状态文字报数，不要据此认为谱只有窗口内那些。"
                "第四，$\\varepsilon$ 很小时轮廓可能为空，这是网格分辨率问题，不是伪谱为空。",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 计算入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "power_iteration":
        return _power_iteration(params, t)
    if viz_id == "qr_schur":
        return _qr_schur(params, t)
    if viz_id == "pseudospectrum":
        return _pseudospectrum(params, t)
    if viz_id == "transient_growth":
        return _transient_growth(params, t)
    if viz_id == "gmres_pseudospec":
        return _gmres_pseudospec(params, t)
    raise KeyError(f"章节 1.4 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _step_of(t: float, n_steps: int) -> int:
    """把连续时间轴四舍五入到整数步并夹到 ``[0, n_steps]``（照 1.2 章的做法）。"""
    k = int(np.floor(float(t) + 0.5))
    return max(0, min(k, max(0, n_steps)))


def _sigma_safe(x: float, floor: float = 1e-300) -> float:
    """把 0 抬到一个小地板，便于 log 轴渲染（同时如实报出地板值）。"""
    return float(max(abs(x), floor))


def _power_matrix(family: str, n: int, ratio: float, shear: float) -> np.ndarray:
    """幂法那一页的矩阵族（全部确定性构造）。"""
    if family == "cyclic":
        return example_eigen_matrix("cyclic", 2, ratio=ratio)
    if family == "sheared":
        return ratio_matrix(n, ratio, shear=max(shear, 1e-3))
    return ratio_matrix(n, ratio, shear=0.0)


# --------------------------------------------------------------------------- #
# viz 1：幂法
# --------------------------------------------------------------------------- #
def _power_iteration(params: Mapping[str, Any], t: float) -> Frame:
    family = str(params["family"])
    ratio = float(params["ratio"])
    shear = float(params["shear"])
    n = 2 if family == "cyclic" else 8
    A = _power_matrix(family, n, ratio, shear)

    res = power_iteration(A, max_iter=400, tol=1e-14, keep=True)
    n_steps = res.n_iter
    k = _step_of(t, n_steps)
    vec = res.vectors[: max(1, k + 1)]
    resid = res.residual[: max(1, k + 1)]
    rq = res.rayleigh[: max(1, k + 1)]

    # 单位圆（主轴固定 ±1.05，规则 1/4）
    theta = np.linspace(0.0, 2.0 * np.pi, 181)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    # 向量轨迹按 (x, y) 交替存一维数组：前两个分量就是二维情形的全部
    head = vec[:, :2] if vec.shape[1] >= 2 else np.concatenate([vec, np.zeros_like(vec)], axis=1)

    ks = np.arange(k + 1, dtype=float)
    theory = np.asarray([ratio**int(j) for j in range(k + 1)], dtype=float)
    theory2 = theory**2

    lam = np.asarray(dense_eigenvalues(A), dtype=complex)
    mags = np.sort(np.abs(lam))[::-1]
    lam1 = float(mags[0]) if mags.size else 0.0
    lam2 = float(mags[1]) if mags.size > 1 else 0.0

    fr = Frame(chapter="1.4", viz="power_iteration", kind="frame")
    fr.add("circle_x", circle_x, unit="1", label="单位圆")
    fr.add("circle_y", circle_y, unit="1", label="单位圆")
    fr.add("vector_x", head[:, 0].astype(float), unit="1", label="迭代向量 $x$ 分量")
    fr.add("vector_y", head[:, 1].astype(float), unit="1", label="迭代向量 $y$ 分量")
    fr.add("step_grid", ks, unit="1", label="迭代步")
    fr.add("rayleigh", rq, unit="1", label="Rayleigh 商 $\\rho(x_k)$")
    fr.add("residual", np.maximum(resid, 1e-300), unit="1", label="残差 $\\|Ax-\\rho x\\|$（地板 $10^{-300}$）")
    fr.add("theory_rate", np.maximum(theory, 1e-300), unit="1", label="理论斜率 $|\\lambda_2/\\lambda_1|^k$")
    fr.add("theory_rate_sq", np.maximum(theory2, 1e-300), unit="1", label="理论斜率（Rayleigh 商，平方）")
    fr.add("error", np.array([_sigma_safe(float(resid[-1]))]), unit="1", label="当前残差（当前步）")
    fr.set_meta(
        family_key=0 if family == "diagonal" else (1 if family == "sheared" else 2),
        n=n,
        step=k,
        n_steps=n_steps,
        ratio=ratio,
        shear=shear,
        lambda1=lam1,
        lambda2=lam2,
        ratio_measured=lam2 / lam1 if lam1 > 0.0 else float("nan"),
        rate_measured=float(np.nanmean(res.ratios[-6:])) if res.ratios.size else float("nan"),
        converged=bool(res.converged),
        complex_pair=bool(res.is_complex_pair),
        angle_step=float(res.angle_step[k]) if res.angle_step.size > k else 0.0,
        rayleigh_k=float(rq[-1]),
        rayleigh_error=abs(float(rq[-1]) - lam1),
        residual_k=float(resid[-1]),
        residual_min=float(np.min(resid)),
        residual_max=float(np.max(resid)),
        transient_ratio=float(np.max(resid) / max(np.min(resid), 1e-300)),
        overflow=False,
    )
    fr.set_strings(
        family=family,
        status=_power_status(res, ratio, family),
        rate_note=f"理论 $|\\lambda_2/\\lambda_1|={ratio:.3f}$，实测残差比 {float(np.nanmean(res.ratios[-6:])) if res.ratios.size else float('nan'):.3f}",
    )
    return fr


def _power_status(res: Any, ratio: float, family: str) -> str:
    """幂法这一页必须给出的「converge / plateau」文字结论（进度条以外的判据）。"""
    if res.is_complex_pair:
        return f"plateau：复主特征值，实向量在二维不变子空间里旋转（夹角平台 {float(np.mean(res.angle_step[-5:])):.3f} rad），幂法不收敛"
    if res.converged:
        return f"converge：{res.n_iter} 步后残差 {res.residual[-1]:.2e}（$|\\lambda_2/\\lambda_1|={ratio:.3f}$）"
    return (
        f"plateau：{res.n_iter} 步仍未到 1e-14（残差 {res.residual[-1]:.2e}）；"
        f"$|\\lambda_2/\\lambda_1|={ratio:.3f}$ 时，按理论估计所需的迭代数已经超过本轮上限"
    )


# --------------------------------------------------------------------------- #
# viz 2：QR 迭代
# --------------------------------------------------------------------------- #
def _qr_matrix(family: str, n: int) -> np.ndarray:
    if family == "symmetric":
        return laplacian_1d(max(2, n))
    if family == "nonsymmetric":
        return ratio_matrix(max(2, n), 0.45, shear=0.6)
    if family == "cyclic":
        base = np.zeros((n, n), dtype=float)
        th = 2.0 * np.pi / 5.0
        r = 0.9
        for i in range(0, n - 1, 2):
            base[i, i] = r * np.cos(th)
            base[i, i + 1] = -r * np.sin(th)
            base[i + 1, i] = r * np.sin(th)
            base[i + 1, i + 1] = r * np.cos(th)
        if n % 2:
            base[n - 1, n - 1] = 0.5
        return base
    if family == "grcar":
        return grcar_matrix(max(4, n))
    raise KeyError(f"未知矩阵族 {family!r}")


def _qr_schur(params: Mapping[str, Any], t: float) -> Frame:
    family = str(params["family"])
    shift = str(params["shift"])
    n = int(params["n"])
    A = _qr_matrix(family, n)
    lam_true = np.asarray(dense_eigenvalues(A), dtype=complex)

    max_iter = 60
    res = qr_iteration(A, shift=shift, max_iter=max_iter, tol=1e-14)
    hist = res.history  # (iters+1, n-1) 的 |次对角元|
    ritz = res.ritz  # (iters+1, n) 的特征值估计（可能长度不一）
    n_steps = max(0, hist.shape[0] - 1)
    k = _step_of(t, n_steps)
    kk = min(k, hist.shape[0] - 1)

    # 热力图：log10|次对角元|（行 = 下标 i，列 = 迭代步）——渲染层按矩阵铺色
    hw = np.log10(np.maximum(hist.T, 1e-300))
    # 当前列（第 k 步）的次对角元，供渲染层画"扫描线"
    cur = np.log10(np.maximum(hist[kk], 1e-300))

    # Ritz 值到真谱的距离（每步的实测误差）
    errs = np.zeros(hist.shape[0], dtype=float)
    for j in range(hist.shape[0]):
        vals = np.asarray(ritz[j], dtype=complex).reshape(-1)
        if vals.size == 0:
            errs[j] = np.nan
            continue
        d = np.abs(vals[:, None] - lam_true[None, :])
        errs[j] = float(np.max(np.min(d, axis=1)))
    cur_err = float(errs[kk])

    # 无位移的理论衰减率：|λ_{i+1}/λ_i|（模降序）
    mags = np.sort(np.abs(lam_true))[::-1]
    gap = float(mags[1] / mags[0]) if mags.size > 1 and mags[0] > 0.0 else float("nan")

    fr = Frame(chapter="1.4", viz="qr_schur", kind="frame")
    fr.add("subdiag_heat", hw, unit="1", label="次对角元的 $\\log_{10}|\\cdot|$（行=下标，列=迭代步）")
    fr.add("subdiag_current", cur, unit="1", label="当前步的次对角元")
    fr.add("subdiag_final", np.log10(np.maximum(hist[-1], 1e-300)), unit="1", label="末步的次对角元")
    fr.add("error", np.maximum(errs, 1e-300), unit="1", label="Ritz 值到真谱的最大距离")
    fr.add("error_mark", np.array([_sigma_safe(cur_err)]), unit="1", label="当前步的谱距离")
    fr.add("step_grid", np.arange(hist.shape[0], dtype=float), unit="1", label="迭代步")
    fr.add("theory_rate", np.asarray([gap ** j if np.isfinite(gap) else 1e-300 for j in range(hist.shape[0])], dtype=float), unit="1", label="理论衰减 $|\\lambda_2/\\lambda_1|^k$")
    fr.set_meta(
        n=n,
        step=kk,
        n_steps=n_steps,
        shift_key=0 if shift == "none" else (1 if shift == "rayleigh" else 2),
        family_key=("symmetric", "nonsymmetric", "cyclic", "grcar").index(family),
        n_iter=int(res.n_iter),
        converged=bool(res.converged),
        n_deflations=int(res.n_deflations),
        n_blocks=int(len(res.block_sizes)),
        spectral_gap=gap,
        error_current=cur_err,
        error_final=float(errs[-1]),
        subdiag_final_max=float(np.max(hist[-1])) if hist.size else 0.0,
        subdiag_initial_max=float(np.max(hist[0])) if hist.size else 0.0,
        is_complex_spectrum=bool(np.any(np.abs(np.imag(lam_true)) > 1e-10)),
        overflow=False,
    )
    fr.set_strings(
        family=family,
        shift=shift,
        status=_qr_status(res, shift, family, cur_err, gap),
        note=(
            f"迭代 {res.n_iter} 步、deflation {res.n_deflations} 次、"
            f"块结构 {list(res.block_sizes) if res.block_sizes else '（未完全收敛）'}"
        ),
    )
    return fr


def _qr_status(res: Any, shift: str, family: str, cur_err: float, gap: float) -> str:
    """QR 这一页的文字结论（converged / block / stall 三态）。"""
    if res.converged and res.block_sizes and any(b == 2 for b in res.block_sizes):
        return f"block-converge：复共轭对停在 2×2 块上（Wilkinson 位移的正常行为），谱距离 {cur_err:.2e}"
    if res.converged:
        return f"converge：{res.n_iter} 步、{res.n_deflations} 次 deflation，谱距离 {cur_err:.2e}"
    if shift == "none":
        return (
            f"stall：无位移 QR 在等模特征值上不收敛（当前谱距离 {cur_err:.2e}）——"
            f"这不是实现问题，是 |λ_i| 相等时次对角元不衰减"
        )
    return f"slow：{res.n_iter} 步内未完全收敛（谱距离 {cur_err:.2e}，族 {family}）"


# --------------------------------------------------------------------------- #
# viz 3：伪谱
# --------------------------------------------------------------------------- #
def _ps_matrix(family: str, n: int) -> np.ndarray:
    """伪谱这一页的矩阵族；``normal`` 与 ``jordan`` 是解析对照组。"""
    name = "shifted_jordan" if family == "shifted_jordan" else family
    return example_pseudospectrum_matrix(name, n)


def _pseudospectrum(params: Mapping[str, Any], t: float) -> Frame:
    family = str(params["family"])
    n = int(params["n"])
    eps_sel = float(params["epsilon"])
    A = _ps_matrix(family, n)

    eps = tuple(sorted(set(list(_EPSILONS) + [eps_sel])))
    grid = pseudospectrum_grid(A, nx=32, ny=32, window=_PS_WINDOW, epsilons=eps, iters=20)
    eps_arr = np.asarray(grid.epsilons, dtype=float)

    lam = np.asarray(dense_eigenvalues(A), dtype=complex)
    rho = float(np.max(np.abs(lam))) if lam.size else 0.0
    rho_ratio = np.asarray(
        [r / rho if rho > 0.0 else 0.0 for r in grid.radius], dtype=float
    )
    # 正规矩阵的解析对照：ρ_ε = ρ + ε ⇒ 比值应当是一条水平线
    exact_ratio = np.asarray([(rho + e) / rho if rho > 0.0 else 0.0 for e in eps_arr], dtype=float)
    # 估计误差：用无条件上界 |t_nn − z| 与谱距离给出"这个 σ_min 有多可信"的场
    est_err = np.maximum(grid.sigma - grid.distance, 0.0)

    sel = int(np.argmin(np.abs(eps_arr - eps_sel)))
    log_sigma = np.log10(np.maximum(grid.sigma, 1e-300))

    fr = Frame(chapter="1.4", viz="pseudospectrum", kind="frame")
    fr.add("grid_x", grid.xs, unit="1", label="$\\mathrm{Re}\\,z$")
    fr.add("grid_y", grid.ys, unit="1", label="$\\mathrm{Im}\\,z$")
    fr.add("log10_sigma", log_sigma, unit="1", label="$\\log_{10}\\sigma_{\\min}(A-zI)$")
    fr.add("inside", grid.inside.astype(float), unit="1", label="伪谱掩码（逐 $\\varepsilon$）")
    # 可信掩码：σ_min 还没掉进 sqrt(u)·‖A‖ 的舍入地板（用 Frobenius 量级做尺度）
    sigma_scale = float(np.max(grid.sigma)) if grid.sigma.size else 1.0
    floor = np.sqrt(np.finfo(float).eps) * max(sigma_scale, 1e-300)
    trusted = np.asarray(
        [[1.0 if (s > floor or d <= floor) else 0.0 for s, d in zip(row_s, row_d)]
         for row_s, row_d in zip(grid.sigma, grid.distance)],
        dtype=float,
    )
    fr.add("trusted", trusted, unit="1", label="可信格点掩码（1 = 没掉进舍入噪声）")
    fr.add("distance", grid.distance, unit="1", label="到谱的距离 dist(z, Λ(A))")
    fr.add("epsilon_grid", eps_arr, unit="1", label="$\\varepsilon$ 序列")
    fr.add("radius", grid.radius, unit="1", label="实测伪谱半径 $\\rho_\\varepsilon$")
    fr.add("radius_ratio", np.maximum(rho_ratio, 1e-300), unit="1", label="$\\rho_\\varepsilon/\\rho(A)$（实测）")
    fr.add("radius_ratio_exact", np.maximum(exact_ratio, 1e-300), unit="1", label="正规矩阵解析 $\\rho_\\varepsilon/\\rho(A)$")
    fr.add("spectrum_x", np.real(lam).astype(float), unit="1", label="谱（$\\mathrm{Re}$）")
    fr.add("spectrum_y", np.imag(lam).astype(float), unit="1", label="谱（$\\mathrm{Im}$）")
    fr.add("error", np.maximum(est_err.ravel(), 1e-300), unit="1", label="$\\sigma_{\\min}$ 估计的可信度余量 $\\sigma_{\\min}-\\mathrm{dist}$")
    fr.add("epsilon_mark", np.array([eps_arr[sel]]), unit="1", label="当前 $\\varepsilon$")
    fr.add("radius_mark", np.array([float(np.maximum(rho_ratio[sel], 1e-300))]), unit="1", label="当前 $\\rho_\\varepsilon/\\rho(A)$")
    fr.set_meta(
        n=n,
        family_key=("normal", "jordan", "shifted_jordan", "shear", "grcar", "companion").index(family),
        rho=rho,
        epsilon=eps_sel,
        epsilon_index=sel,
        n_eps=int(eps_arr.size),
        radius_current=float(grid.radius[sel]),
        radius_ratio_current=float(rho_ratio[sel]),
        radius_ratio_min=float(np.min(rho_ratio)) if rho_ratio.size else 0.0,
        radius_ratio_max=float(np.max(rho_ratio)) if rho_ratio.size else 0.0,
        sigma_min=float(np.min(grid.sigma)),
        sigma_max=float(np.max(grid.sigma)),
        kappa_max=float(grid.kappa_max),
        trusted_fraction=float(grid.converged_fraction),
        n_noise_points=int(grid.n_points - grid.n_converged),
        n_points=int(grid.n_points),
        window=_PS_WINDOW,
        kreiss_lower_bound=kreiss_from_pseudospectrum(eps_arr, grid.radius),
        overflow=False,
    )
    fr.set_strings(
        family=family,
        status=_ps_status(family, rho, grid, sel),
        method="hermitian-jacobi（σ_min 走 MᴴM 的 Hermite Jacobi；不掉进噪声地板的格点标 trusted=1）",
    )
    return fr


def _ps_status(family: str, rho: float, grid: Any, sel: int) -> str:
    frac = grid.converged_fraction
    if family == "normal":
        return (
            f"正规对照：ρ_ε/ρ(A) 应当恒为 1+ε/ρ（实测 {grid.radius[sel] / rho if rho > 0 else 0.0:.4f}），"
            "云雾就是 ε-圆盘的并"
        )
    return (
        f"ρ_ε(ε={grid.epsilons[sel]:.0e}) = {grid.radius[sel]:.4f}，ρ(A) = {rho:.4f}，"
        f"可信格点 {frac:.1%}（其余 {grid.n_points - grid.n_converged} 个已到舍入地板）"
    )


# --------------------------------------------------------------------------- #
# viz 4：瞬态增长
# --------------------------------------------------------------------------- #
def _transient_growth(params: Mapping[str, Any], t: float) -> Frame:
    family = str(params["family"])
    s = float(params["nonnormal"])
    n = int(params["n"])
    k_max = int(params["k_max"])
    # shear 族用 s 当剪切强度；其余族把 s 折算成"额外耦合"（加到对角上的尺度保持谱半径 < 1）
    if family == "shear":
        A = shear_matrix(n, s)
    elif family == "jordan_shifted":
        A = jordan_block(n, 0.75) + s * np.triu(np.ones((n, n)), 1)
    elif family == "grcar":
        A = 0.75 * grcar_matrix(n) + s * np.eye(n)
    else:
        A = np.diag(np.linspace(0.95, 0.3, n)) + s * np.triu(np.ones((n, n)), 1) * 0.1

    tr = transient_growth(A, k_max=k_max)
    k = _step_of(t, k_max)
    ks = np.arange(k + 1, dtype=float)
    norms = np.maximum(tr.norms[: k + 1], 1e-300)
    pred = np.maximum(tr.spectral_prediction[: k + 1], 1e-300)
    ratio = norms / pred

    fr = Frame(chapter="1.4", viz="transient_growth", kind="frame")
    fr.add("step_grid", ks, unit="1", label="幂次 $k$")
    fr.add("norms", norms, unit="1", label="实测 $\\|A^k\\|_2$")
    fr.add("spectral_prediction", pred, unit="1", label="谱半径预测 $\\rho(A)^k$")
    fr.add("growth_ratio", ratio, unit="1", label="增长倍数 $\\|A^k\\|/\\rho(A)^k$")
    fr.add("error", np.abs(norms - pred), unit="1", label="实测与谱预测之差（误差场）")
    fr.add("error_mark", np.array([_sigma_safe(abs(float(tr.norms[k]) - float(tr.spectral_prediction[k])))]), unit="1", label="当前步之差")
    fr.set_meta(
        n=n,
        family_key=("shear", "jordan_shifted", "grcar", "normal").index(family),
        step=k,
        k_max=k_max,
        nonnormal=s,
        spectral_radius=tr.spectral_radius,
        peak=tr.peak,
        peak_index=tr.peak_index,
        growth_ratio=tr.growth_ratio,
        kreiss=tr.kreiss,
        asymptotic_factor=tr.asymptotic_factor,
        norm_current=float(tr.norms[k]),
        pred_current=float(tr.spectral_prediction[k]),
        ratio_current=float(tr.norms[k] / tr.spectral_prediction[k]) if tr.spectral_prediction[k] > 0 else 0.0,
        non_normal=bool(tr.non_normal),
        peak_hit_max=bool(tr.peak_index >= k_max),
        overflow=False,
    )
    fr.set_strings(
        family=family,
        status=_transient_status(tr, k_max),
        note=f"Kreiss 常数（实测峰值）= {tr.kreiss:.4e}，出现在 k = {tr.peak_index}",
    )
    return fr


def _transient_status(tr: Any, k_max: int) -> str:
    if tr.peak_index >= k_max:
        return f"peak 贴着 k_max={k_max}：峰值可能还在后面，请把 k_max 拖大（当前 {tr.peak:.3e}）"
    if tr.non_normal:
        return (
            f"非正规：ρ(A) = {tr.spectral_radius:.3f} < 1，可 ‖A^k‖ 在 k = {tr.peak_index} "
            f"冲到 {tr.peak:.3e}（是谱预测的 {tr.growth_ratio:.1e} 倍）"
        )
    return f"接近正规：峰值 {tr.peak:.3f} ≈ 1，谱半径预测 {tr.spectral_radius:.3f} 基本说了算"


# --------------------------------------------------------------------------- #
# viz 5：GMRES 与伪谱
# --------------------------------------------------------------------------- #
def _gmres_matrix(family: str, n: int) -> np.ndarray:
    if family == "grcar":
        return grcar_matrix(max(4, n))
    if family == "shear":
        return shear_matrix(max(2, n), 1.0) * 0.6
    if family == "jordan_shifted":
        return jordan_block(max(2, n), 0.5)
    return np.diag(np.linspace(1.0, 0.4, max(2, n)))


def _gmres_rhs(kind: str, n: int) -> np.ndarray:
    if kind == "ones":
        return np.ones(n)
    if kind == "ramp":
        return np.linspace(1.0, 2.0, n)
    if kind == "alternating":
        return np.asarray([1.0 if i % 2 == 0 else -1.0 for i in range(n)], dtype=float)
    if kind == "e1":
        e = np.zeros(n, dtype=float)
        e[0] = 1.0
        return e
    raise KeyError(f"未知右端项 {kind!r}")


def _gmres_pseudospec(params: Mapping[str, Any], t: float) -> Frame:
    family = str(params["family"])
    rhs_kind = str(params["rhs"])
    n = int(params["n"])
    eps_sel = float(params["epsilon"])
    A = _gmres_matrix(family, n)
    b = _gmres_rhs(rhs_kind, n)

    eps = tuple(sorted(set([1e-2, 1e-3, 1e-4, 1e-5, eps_sel])))
    grid = pseudospectrum_grid(A, nx=28, ny=28, window=1.5, epsilons=eps, iters=16)
    eps_arr = np.asarray(grid.epsilons, dtype=float)
    sel = int(np.argmin(np.abs(eps_arr - eps_sel)))

    gm: GMRESResult = gmres(A, b, max_iter=min(24, n))
    n_steps = gm.residual.size - 1
    k = _step_of(t, n_steps)
    resid = np.maximum(gm.residual[: k + 1], 1e-300)

    # σ_max 的粗估（用每个格点的上界最大值），以及"只依赖谱"的多项式下降界
    bound = gmres_arnoldi_residual_bound(A, b, max(1, k))
    lam = np.asarray(dense_eigenvalues(A), dtype=complex)

    fr = Frame(chapter="1.4", viz="gmres_pseudospec", kind="frame")
    fr.add("grid_x", grid.xs, unit="1", label="$\\mathrm{Re}\\,z$")
    fr.add("grid_y", grid.ys, unit="1", label="$\\mathrm{Im}\\,z$")
    fr.add("log10_sigma", np.log10(np.maximum(grid.sigma, 1e-300)), unit="1", label="$\\log_{10}\\sigma_{\\min}(A-zI)$")
    fr.add("inside", grid.inside.astype(float), unit="1", label="伪谱掩码（逐 $\\varepsilon$）")
    fr.add("epsilon_grid", eps_arr, unit="1", label="$\\varepsilon$ 序列")
    fr.add("radius", grid.radius, unit="1", label="伪谱半径 $\\rho_\\varepsilon$")
    fr.add("spectrum_x", np.real(lam).astype(float), unit="1", label="谱（$\\mathrm{Re}$）")
    fr.add("spectrum_y", np.imag(lam).astype(float), unit="1", label="谱（$\\mathrm{Im}$）")
    fr.add("residual", resid, unit="1", label="GMRES 相对残差（对数轴）")
    fr.add("residual_floor", np.full(k + 1, 1e-300, dtype=float), unit="1", label="对数轴地板")
    fr.add("spectral_bound", np.full(k + 1, max(bound, 1e-300), dtype=float), unit="1", label="只依赖谱的多项式下降界")
    fr.add("error", np.array([_sigma_safe(float(gm.residual[k]))]), unit="1", label="当前残差")
    fr.add("error_mark", np.array([_sigma_safe(float(gm.residual[k]))]), unit="1", label="当前残差")
    fr.set_meta(
        n=n,
        step=k,
        n_steps=n_steps,
        family_key=("grcar", "shear", "jordan_shifted", "normal").index(family),
        rhs_key=("ones", "ramp", "alternating", "e1").index(rhs_kind),
        epsilon=eps_sel,
        epsilon_index=sel,
        converged=bool(gm.converged),
        breakdown=bool(gm.breakdown),
        plateau=bool(gm.plateau),
        krylov_dim=int(gm.krylov_dim),
        residual_current=float(gm.residual[k]),
        residual_min=float(np.min(gm.residual)),
        residual_final=float(gm.residual[n_steps]),
        max_drop_ratio=float(np.min(gm.residual[1 : n_steps + 1] / np.maximum(gm.residual[:n_steps], 1e-300))) if n_steps else 1.0,
        spectral_bound=float(bound),
        bound_gap=float(gm.residual[k] / bound) if bound > 0.0 else float("inf"),
        radius_current=float(grid.radius[sel]),
        spectral_radius=spectral_radius_estimate(A, lam),
        trusted_fraction=float(grid.converged_fraction),
        overflow=False,
    )
    fr.set_strings(
        family=family,
        rhs=rhs_kind,
        status=_gmres_status(gm, bound),
        note=f"谱半径 {spectral_radius_estimate(A, lam):.3f}，伪谱半径 ρ_ε = {grid.radius[sel]:.3f}（ε={eps_sel:.0e}）",
    )
    return fr


def _gmres_status(gm: GMRESResult, bound: float) -> str:
    if gm.plateau:
        return (
            f"plateau：残差连续多步几乎不降（当前 {gm.residual[gm.n_iter]:.2e}），"
            f"而只依赖谱的下降界是 {bound:.2e}——差距就是特征向量矩阵的条件数"
        )
    if gm.converged:
        return f"converge：{gm.n_iter} 步到 {gm.residual[gm.n_iter]:.2e}（单调下降，符合 GMRES 理论）"
    return f"slow：{gm.n_iter} 步后残差 {gm.residual[gm.n_iter]:.2e}，还没到 1e-12"
