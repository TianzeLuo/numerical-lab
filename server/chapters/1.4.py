"""1.4 章：特征值、伪谱与非正规性（五个可视化的帧装配）。

这一章有五页，分工是"从最温和的迭代到最凶险的几何"：

1. ``power_iteration`` —— 幂法对齐过程。参数里有 **|λ₂/λ₁|**（收敛率，有理论临界值 1）
   与 **非正规强度 shear**。时间轴是**迭代步 k**；副轴（对数）画残差与理论斜率
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
                    "label": "第二特征值比 |λ₂/λ₁|",
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
                },
                {
                    "name": "family",
                    "label": "矩阵族",
                    "type": "enum",
                    "options": ["diagonal", "sheared", "cyclic"],
                    "default": "diagonal",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 120.0, "dt": 1.0, "label": "迭代步 k"},
            "requiredAux": ["residual"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "单位圆与迭代向量（固定坐标 ±1.05）",
                    "ylabel": "y",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "残差与理论收敛率",
                    "ylabel": "‖Ax−ρ(x)x‖ 与 |λ₂/λ₁|^k",
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
                "幂法收敛定理：$\\|x_k-\\pm v_1\\|=O(|\\lambda_2/\\lambda_1|^{k})$，$|\\rho(x_k)-\\lambda_1|=O(|\\lambda_2/\\lambda_1|^{2k})$",
                "$\\xi_k$ 的残差满足 $\\|Ax-\\rho(x)x\\|=O(|\\lambda_2/\\lambda_1|^{k})\\cdot\\|A\\|$（Rayleigh 商是特征值的二次逼近）",
                "复主特征值时实向量序列不收敛：它在 $\\operatorname{span}\\{v,\\bar{v}\\}$ 里以固定角度旋转（cyclic 族可复现）",
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
                },
                {
                    "name": "shift",
                    "label": "位移策略",
                    "type": "enum",
                    "options": ["none", "rayleigh", "wilkinson"],
                    "default": "wilkinson",
                    "critical": [],
                    "criticalNote": "",
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
                },
            ],
            "time": {"mode": "scrub", "tMax": 60.0, "dt": 1.0, "label": "QR 迭代步"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "log10|次对角元|（行 = i，列 = 迭代步）",
                    "ylabel": "次对角元下标 i",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "Ritz 值到真谱的距离",
                    "ylabel": "max_i min_j |θ_i − λ_j|",
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
                "无位移 QR 的次对角元按 $|\\lambda_{i+1}/\\lambda_i|$ 衰减；等模特征值（复对、$\\pm\\lambda$）时它不衰减",
                "带位移 QR：Rayleigh 位移对对称矩阵三次收敛，Wilkinson 位移对非对称矩阵给块收敛",
                "Hessenberg 化把每一步的代价从 $O(n^{3})$ 降到 $O(n^{2})$，且不改变谱（相似变换）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "pseudospectrum",
            "title": "ε-伪谱云雾：σ_min(A−zI) 的等高线",
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
                },
                {
                    "name": "epsilon",
                    "label": "当前 ε（轮廓高亮）",
                    "type": "float",
                    "min": 1e-8,
                    "max": 1e-1,
                    "step": 1e-8,
                    "default": 1e-4,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "复平面（固定窗口）",
                    "ylabel": "Im z",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "伪谱半径与谱半径之比",
                    "ylabel": "ρ_ε / ρ(A)",
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
                "$\\Lambda_\\varepsilon(A)=\\{z:\\sigma_{\\min}(A-zI)\\le\\varepsilon\\}$，即所有 $\\|\\delta A\\|\\le\\varepsilon$ 的扰动矩阵谱的并集，"
                "也即 $\\{z:\\|{(zI-A)}^{-1}\\|\\ge\\varepsilon^{-1}\\}$",
                "正规矩阵：$\\Lambda_\\varepsilon$ 是各圆盘 $\\{|z-\\lambda_i|\\le\\varepsilon\\}$ 的并；Jordan 块 $J_n(0)$：$\\rho_\\varepsilon=\\varepsilon^{1/n}$（花瓣形）",
                "特征值扰动界：$\\Lambda(A+\\delta A)\\subseteq\\Lambda_\\varepsilon(A)$（$\\|\\delta A\\|_2\\le\\varepsilon$）；伪谱半径 $\\ge\\varepsilon+\\rho(A)$ 的凸包结论",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "transient_growth",
            "title": "瞬态增长：‖A^k‖ 先涨后落 vs 谱半径的单调衰减",
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
                },
                {
                    "name": "k_max",
                    "label": "时间上限 k_max",
                    "type": "int",
                    "min": 10,
                    "max": 80,
                    "default": 40,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 40.0, "dt": 1.0, "label": "幂次 k"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": True,
                    "title": "增长倍数 ‖A^k‖ / ρ(A)^k",
                    "ylabel": "倍数",
                    "scale": "log",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "‖A^k‖₂ 与 ρ(A)^k",
                    "ylabel": "2-范数",
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
                "Kreiss 矩阵定理：$K(A)=\\sup_k\\|A^{k}\\|_2=\\sup_{|z|>1}(|z|-1)\\|{(zI-A)}^{-1}\\|_2$（有限矩阵上确界可达）",
                "预解式界：$\\rho_\\varepsilon(A)<1$ 时 $\\|A^{k}\\|_2\\le(1+\\varepsilon)^{k+1}\\rho_\\varepsilon(A)/\\varepsilon$（伪谱一旦鼓出单位圆就有增长）",
                "Jordan 块 $J_n(0)$：$\\rho(A)=0$ 但 $\\|J_n^{k}\\|_2=1$（$1\\le k\\le n-1$）——谱半径对瞬态的预测完全失效",
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
                },
                {
                    "name": "rhs",
                    "label": "右端项 b",
                    "type": "enum",
                    "options": ["ones", "ramp", "alternating", "e1"],
                    "default": "ones",
                    "critical": [],
                    "criticalNote": "",
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
                },
                {
                    "name": "epsilon",
                    "label": "轮廓 ε",
                    "type": "float",
                    "min": 1e-6,
                    "max": 1e-1,
                    "step": 1e-6,
                    "default": 1e-3,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 24.0, "dt": 1.0, "label": "GMRES 迭代步"},
            "requiredAux": ["residual"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "伪谱（固定窗口）与谱",
                    "ylabel": "Im z",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "GMRES 相对残差（对数）",
                    "ylabel": "‖r_k‖/‖r_0‖",
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
                "GMRES 残差单调：$\\|r_k\\|_2=\\min_{p\\in P_k,\\,p(0)=1}\\|p(A)r_0\\|_2$，随 $k$ 不增",
                "收敛界：$\\|r_k\\|/\\|r_0\\|\\le\\kappa_2(V)\\cdot\\min_p\\max_{\\lambda\\in\\Lambda}|p(\\lambda)|$（$V$ 是特征向量矩阵）",
                "非正规矩阵：伪谱大 $\\Rightarrow$ 预解式大 $\\Rightarrow$ GMRES/迭代法的暂态平台；1.5 章用它解释预条件",
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
    fr.add("vector_x", head[:, 0].astype(float), unit="1", label="迭代向量 x 分量")
    fr.add("vector_y", head[:, 1].astype(float), unit="1", label="迭代向量 y 分量")
    fr.add("step_grid", ks, unit="1", label="迭代步")
    fr.add("rayleigh", rq, unit="1", label="Rayleigh 商 ρ(x_k)")
    fr.add("residual", np.maximum(resid, 1e-300), unit="1", label="残差 ‖Ax−ρx‖（地板 1e-300）")
    fr.add("theory_rate", np.maximum(theory, 1e-300), unit="1", label="理论斜率 |λ₂/λ₁|^k")
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
        rate_note=f"理论 |λ₂/λ₁| = {ratio:.3f}，实测残差比 {float(np.nanmean(res.ratios[-6:])) if res.ratios.size else float('nan'):.3f}",
    )
    return fr


def _power_status(res: Any, ratio: float, family: str) -> str:
    """幂法这一页必须给出的「converge / plateau」文字结论（进度条以外的判据）。"""
    if res.is_complex_pair:
        return f"plateau：复主特征值，实向量在二维不变子空间里旋转（夹角平台 {float(np.mean(res.angle_step[-5:])):.3f} rad），幂法不收敛"
    if res.converged:
        return f"converge：{res.n_iter} 步后残差 {res.residual[-1]:.2e}（|λ₂/λ₁| = {ratio:.3f}）"
    return (
        f"plateau：{res.n_iter} 步仍未到 1e-14（残差 {res.residual[-1]:.2e}）；"
        f"|λ₂/λ₁| = {ratio:.3f} 时理论迭代数 ≈ log(tol)/log(ratio) 已经超过上限"
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
    fr.add("subdiag_heat", hw, unit="1", label="log10|次对角元|（行=下标，列=迭代步）")
    fr.add("subdiag_current", cur, unit="1", label="当前步的次对角元")
    fr.add("subdiag_final", np.log10(np.maximum(hist[-1], 1e-300)), unit="1", label="末步的次对角元")
    fr.add("error", np.maximum(errs, 1e-300), unit="1", label="Ritz 值到真谱的最大距离")
    fr.add("error_mark", np.array([_sigma_safe(cur_err)]), unit="1", label="当前步的谱距离")
    fr.add("step_grid", np.arange(hist.shape[0], dtype=float), unit="1", label="迭代步")
    fr.add("theory_rate", np.asarray([gap ** j if np.isfinite(gap) else 1e-300 for j in range(hist.shape[0])], dtype=float), unit="1", label="理论衰减 |λ₂/λ₁|^k")
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
    fr.add("grid_x", grid.xs, unit="1", label="Re z")
    fr.add("grid_y", grid.ys, unit="1", label="Im z")
    fr.add("log10_sigma", log_sigma, unit="1", label="log10 σ_min(A−zI)")
    fr.add("inside", grid.inside.astype(float), unit="1", label="伪谱掩码（逐 ε）")
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
    fr.add("epsilon_grid", eps_arr, unit="1", label="ε 序列")
    fr.add("radius", grid.radius, unit="1", label="实测伪谱半径 ρ_ε")
    fr.add("radius_ratio", np.maximum(rho_ratio, 1e-300), unit="1", label="ρ_ε / ρ(A)（实测）")
    fr.add("radius_ratio_exact", np.maximum(exact_ratio, 1e-300), unit="1", label="正规矩阵解析 ρ_ε/ρ(A)")
    fr.add("spectrum_x", np.real(lam).astype(float), unit="1", label="谱（Re）")
    fr.add("spectrum_y", np.imag(lam).astype(float), unit="1", label="谱（Im）")
    fr.add("error", np.maximum(est_err.ravel(), 1e-300), unit="1", label="σ_min 估计的可信度余量 σ_min−dist")
    fr.add("epsilon_mark", np.array([eps_arr[sel]]), unit="1", label="当前 ε")
    fr.add("radius_mark", np.array([float(np.maximum(rho_ratio[sel], 1e-300))]), unit="1", label="当前 ρ_ε/ρ(A)")
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
    fr.add("step_grid", ks, unit="1", label="幂次 k")
    fr.add("norms", norms, unit="1", label="实测 ‖A^k‖₂")
    fr.add("spectral_prediction", pred, unit="1", label="谱半径预测 ρ(A)^k")
    fr.add("growth_ratio", ratio, unit="1", label="增长倍数 ‖A^k‖/ρ(A)^k")
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
    fr.add("grid_x", grid.xs, unit="1", label="Re z")
    fr.add("grid_y", grid.ys, unit="1", label="Im z")
    fr.add("log10_sigma", np.log10(np.maximum(grid.sigma, 1e-300)), unit="1", label="log10 σ_min(A−zI)")
    fr.add("inside", grid.inside.astype(float), unit="1", label="伪谱掩码（逐 ε）")
    fr.add("epsilon_grid", eps_arr, unit="1", label="ε 序列")
    fr.add("radius", grid.radius, unit="1", label="伪谱半径 ρ_ε")
    fr.add("spectrum_x", np.real(lam).astype(float), unit="1", label="谱（Re）")
    fr.add("spectrum_y", np.imag(lam).astype(float), unit="1", label="谱（Im）")
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
