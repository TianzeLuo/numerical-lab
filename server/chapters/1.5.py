"""1.5 章：稀疏迭代法与预条件。

四个可视化的分工是"从轨迹、到模态、到子空间、到预条件"：

1. ``iteration_race`` —— 二维椭圆等高线上的**轨迹赛跑**：Jacobi / GS / SOR / CG / GMRES
   五条轨迹同图，时间轴是迭代步。参数里 ``omega`` 与 ``kappa`` 都有理论临界值：
   :math:`\\omega=1` 退化为 GS、:math:`0<\\omega<2` 是收敛的必要条件，
   而 :math:`\\kappa` 决定椭圆有多扁（也就决定 Jacobi 要爬多久）。
2. ``error_spectrum`` —— 误差在**解析正弦模态**上的分量条形图随迭代衰减：
   谁先衰减完一目了然；每个模态旁边给出理论衰减因子与实测比值。
3. ``krylov_growth`` —— Krylov 子空间逐步扩张（2D/3D 里画基向量与解轨迹），
   并把**残差正交性**的实测矩阵摊开：CG 是 :math:`r_i^{\\mathsf T}r_j\\approx0`（:math:`i\\ne j`），
   GMRES 是 :math:`r_i^{\\mathsf T}Av_j\\approx0`（:math:`j<i`）——两条不是一回事。
4. ``precond_effect`` —— 预条件前后（none / jacobi / ssor / ilu0）的谱分布散点 + 收敛曲线，
   同时给出**实测**条件数、由迭代数反推的等效条件数与 ILU(0) 丢弃掉的填充次数。

约定（与 0.0/1.2 章一致）：``meta`` 只放数字/布尔/``None``，字符串一律走 ``strings``；
时间轴对"离散步骤"型可视化就是**步骤编号**（``t`` 四舍五入到整数步并夹住），
``meta.step`` 给出真正用的那一步。本文件只做搬运与装配：所有数值都来自
:mod:`numcore.linalg.iterative` / :mod:`krylov` / :mod:`precond`。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from numcore.linalg.iterative import (
    IterationResult,
    convergence_factor,
    jacobi,
    gauss_seidel,
    laplacian_1d,
    laplacian_2d,
    mode_components,
    mode_decay_factors,
    power_method,
    rotated_spd_2x2,
    sine_eigenvectors_1d,
    sor,
    sor_spectral_radius,
    iteration_matrix,
)
from numcore.linalg.krylov import (
    conjugate_gradient,
    gmres,
)
from numcore.linalg.lu import lu_decompose, lu_solve
from numcore.linalg.precond import (
    ILU0Precond,
    JacobiPrecond,
    SSORPrecond,
    build_precond,
    kappa_from_factor,
    precond_spectrum,
)
from numcore.linalg.sparse import CSR
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 迭代步时间轴上限（所有"逐步"型 viz 共用同一根轴的长度）
_MAX_STEPS = 60
#: 谱散点最多画多少个 Ritz 值（Lanczos 步数；内部 Ritz 值收敛慢，见 pitfalls）
_MAX_RITZ = 80


MANIFEST: dict[str, Any] = {
    "chapter": "1.5",
    "title": "稀疏迭代法与预条件",
    "part": "I",
    "summary": (
        "迭代法的收敛由迭代矩阵的谱半径决定；Krylov 方法把问题投影到逐步扩张的子空间上，"
        "CG 极小化 $A$-范数误差、GMRES 极小化残差；预条件改的是谱分布，而不只是一个 $\\kappa$ 数字。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "iteration_race",
            "title": "迭代路径赛跑：椭圆等高线上的五条轨迹",
            "description": (
                "二次型 ${(x-x^{*})}^{\\mathrm{T}}A(x-x^{*})$ 的等高线是椭圆（扁度 $=\\kappa$）；Jacobi / GS / SOR / CG / GMRES "
                "五条迭代轨迹同图竞速，粗点标出当前方法在第 $k$ 步的位置。副轴是各方法的相对残差曲线。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "method",
                    "label": "当前高亮的方法",
                    "type": "enum",
                    "options": ["jacobi", "gs", "sor", "cg", "gmres"],
                    "default": "sor",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "omega",
                    "label": "SOR 松弛因子 ω",
                    "type": "float",
                    "min": 0.5,
                    "max": 1.99,
                    "step": 0.01,
                    "default": 1.3,
                    "critical": [1.0],
                    "criticalNote": (
                        "$\\omega=1$ 时 SOR 退化为 Gauss–Seidel；收敛的必要条件是 $0<\\omega<2$（$\\omega\\ge 2$ 时 "
                        "$\\rho(T_\\omega)\\ge 1$，迭代发散）。最优值 $\\omega_{\\text{opt}}=2/(1+\\sqrt{1-\\mu^{2}})$ 由当前矩阵的 Jacobi "
                        "谱半径 $\\mu$ 决定，帧里的 meta.omega_opt 给出它的真实取值。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "kappa",
                    "label": "条件数 κ（目标）",
                    "type": "float",
                    "min": 1.5,
                    "max": 200.0,
                    "step": 0.5,
                    "default": 20.0,
                    "critical": [1.0],
                    "criticalNote": (
                        "$\\kappa=\\lambda_{\\max}/\\lambda_{\\min}=1$ 时等高线是正圆，任何方法一步到解；$\\kappa$ 越大椭圆越扁，"
                        "沿长轴方向的误差衰减越慢——Jacobi 的谱半径 $\\mu$ 与 SOR 的最优 $\\omega$ 都由它决定。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "angle",
                    "label": "特征向量方向 θ（度）",
                    "type": "float",
                    "min": 0.0,
                    "max": 180.0,
                    "step": 5.0,
                    "default": 35.0,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": float(_MAX_STEPS), "dt": 1.0, "label": "迭代步 k"},
            "requiredAux": ["residual"],
            "axes": [
                {
                    "id": "plane",
                    "autoscaled": False,
                    "title": "等高线与迭代轨迹（物理坐标，固定轴）",
                    "ylabel": "x2",
                },
                {
                    "id": "convergence",
                    "autoscaled": True,
                    "title": "各方法的相对残差 ‖r_k‖/‖r_0‖",
                    "ylabel": "相对残差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "这张图有两个经典骗局。第一，**固定坐标轴**是刻意的：$\\kappa=200$ 时椭圆扁得几乎是一条线，"
                "如果让坐标轴自适应，椭圆会被拉成正圆、「条件数」这个几何事实当场消失（这就是规则 1 "
                "要求逐轴声明 autoscaled 的原因）。第二，轨迹在方法收敛之后是**重复点**（帧里没有 NaN，"
                "免得前端拒绝整张图），所以右下角那团「不动了」的点要配合 meta.race_len 读——它才告诉你"
                "每条轨迹真正在第几步停住。另外 CG/GMRES 在 $n=2$ 的平面上至多两步就到解，"
                "它们的轨迹几乎是直线段；把它们的「快」当成「Krylov 方法在所有问题上都快」是另一个错误"
                "（$\\kappa$ 大到 CG 的最坏界 $2{((\\sqrt{\\kappa}-1)/(\\sqrt{\\kappa}+1))}^{k}$ 生效时，CG 也会一步一挪）。"
            ),
            "theory": [
                "定常迭代收敛 $\\Leftrightarrow$ $\\rho(T)<1$，且渐近收敛因子就是 $\\rho(T)$（误差 $e_k=T^{k}e_0$）",
                "SOR 在一致次序矩阵上的解析谱半径：$\\omega\\le\\omega_{\\text{opt}}$ 时为 ${((\\omega\\mu+\\sqrt{\\omega^{2}\\mu^{2}-4(\\omega-1)})/2)}^{2}$，$\\omega\\ge\\omega_{\\text{opt}}$ 时为 $\\omega-1$",
                "CG 在 $n$ 维空间里至多 $n$ 步给出精确解（精确算术）；GMRES(m) 同理，$m\\ge n$ 时一步周期收敛",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "error_spectrum",
            "title": "误差的特征分量分解：谁先衰减完",
            "description": (
                "一维 Laplacian 的解析正弦模态基下，误差分量 $|v_j^{\\mathrm{T}}e_k|$ 随迭代步的条形演化；"
                "每个模态旁标注理论衰减因子 $|\\lambda_j(T)|$ 与实测比值。副轴给出误差 2-范数与相对残差。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "n",
                    "label": "网格规模 n（模态数）",
                    "type": "int",
                    "min": 8,
                    "max": 64,
                    "default": 32,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "method",
                    "label": "迭代法",
                    "type": "enum",
                    "options": ["jacobi", "gs", "sor"],
                    "default": "sor",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "omega",
                    "label": "SOR 松弛因子 ω",
                    "type": "float",
                    "min": 0.5,
                    "max": 1.99,
                    "step": 0.01,
                    "default": 1.6,
                    "critical": [1.0],
                    "criticalNote": (
                        "$\\omega=1$ 退化为 Gauss–Seidel；$n=32$ 时 $\\mu=\\cos(\\pi/33)\\approx 0.99547$，"
                        "$\\omega_{\\text{opt}}=2/(1+\\sqrt{1-\\mu^{2}})\\approx 1.8236$，此时 $\\rho=\\omega_{\\text{opt}}-1\\approx 0.8236$。"
                        "拖过 $\\omega_{\\text{opt}}$ 之后主特征值变成复共轭对，误差会一边旋转一边衰减。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "init",
                    "label": "初始误差的模态构成",
                    "type": "enum",
                    "options": ["all_modes", "low_mode", "high_mode"],
                    "default": "all_modes",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": float(_MAX_STEPS), "dt": 1.0, "label": "迭代步 k"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "modes",
                    "autoscaled": False,
                    "title": "各模态的误差分量（对数色阶条形）",
                    "ylabel": "模态 j",
                },
                {
                    "id": "decay",
                    "autoscaled": True,
                    "title": "误差与残差（对数）",
                    "ylabel": "‖e_k‖/‖e_0‖ 与 ‖r_k‖/‖r_0‖",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "最容易被这张图骗到的地方是「模态分量按理论因子单调衰减」。对 **Jacobi** 它确实成立"
                "（$T_J=I-D^{-1}A$ 与 $A$ 同特征向量，$c_j(k)=\\mu_j^{k}c_j(0)$ 精确成立，实测偏差只有 $10^{-16}$）；"
                "但对 **GS/SOR**，$A$ 的特征基不是 $T$ 的特征基，投影系数会**先混后衰减**：早期几步低频模态"
                "甚至会短暂变大（非正规性的可见后果），只有渐近段的斜率才等于理论因子。第二，分量幅值掉到"
                "$10^{-16}$ 以下就是舍入噪声，帧里用 meta.component_floor 明确标了这个地板——把地板以下的条形"
                "当成「误差真的降到 $10^{-300}$」是自欺。第三，SOR 的最优 $\\omega$ 让 ${|\\lambda|}_{\\max}$ 最小，但**不是**让每个"
                "模态都最小：某些高频模态在 $\\omega$ 略小于 $\\omega_{\\text{opt}}$ 时反而衰减更快，只有最慢的那个模态（通常是 $j=1$）"
                "决定整体收敛。"
            ),
            "theory": [
                "误差传播 $e_{k+1}=T e_k$，故 $e_k=T^{k}e_0$；$\\rho(T)<1$ 是收敛的充要条件",
                "一维 Laplacian：$\\lambda_j(A)=4\\sin^{2}(j\\pi/(2(n+1)))$，$\\mu_j(T_J)=\\cos(j\\pi/(n+1))$，$T_{\\mathrm{GS}}$ 的谱是 $\\mu_j^{2}$",
                "SOR 的逐模态因子由 Young 关系给出（$\\omega$ 越过 $\\omega_{\\text{opt}}$ 后为复共轭对，模长 $\\sqrt{\\omega-1}$）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "krylov_growth",
            "title": "Krylov 子空间逐步扩张与残差正交性",
            "description": (
                "在 2D/3D 平面上画出 Krylov 基 $v_0,\\ldots,v_k$、解轨迹 $x_k$ 与被扩张的子空间；"
                "右侧给出残差正交性的实测矩阵（CG 用 $|r_i^{\\mathrm{T}}r_j|$，GMRES 用 $|r_i^{\\mathrm{T}}Av_j|$）与 Arnoldi 的 $H$。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "dim",
                    "label": "空间维数 n",
                    "type": "int",
                    "min": 2,
                    "max": 3,
                    "default": 3,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "method",
                    "label": "Krylov 方法",
                    "type": "enum",
                    "options": ["cg", "gmres"],
                    "default": "gmres",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "stretch",
                    "label": "对角反差 s（A = D S D）",
                    "type": "float",
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.1,
                    "default": 2.0,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 3.0, "dt": 1.0, "label": "Krylov 步 k"},
            "requiredAux": ["residual"],
            "axes": [
                {
                    "id": "space",
                    "autoscaled": True,
                    "title": "Krylov 基与解轨迹（2D/3D）",
                    "ylabel": "x2",
                },
                {
                    "id": "orth",
                    "autoscaled": False,
                    "title": "残差正交性实测矩阵",
                    "ylabel": "步 i / 基 j",
                },
            ],
            "pitfalls": (
                "两条正交性极易混为一谈：CG 的残差满足 $r_k\\perp K_k$，所以 $|r_i^{\\mathrm{T}}r_j|$（$i\\ne j$）应当接近 $0$；"
                "GMRES 的残差满足的是 $r_k\\perp A K_k$，即 $|r_i^{\\mathrm{T}}Av_j|\\approx 0$，而 $|r_i^{\\mathrm{T}}r_j|$ **并不**小"
                "（GMRES 只最小化残差，不保证残差两两正交）。把 GMRES 的 $r_i^{\\mathrm{T}}r_j$ 当作「应该为 $0$」去检查，"
                "会得出「GMRES 实现错了」的错误结论。第二，正交性只在残差还高于舍入底时成立：实测最后一步"
                "残差落到 $10^{-17}$ 之后，$|r_i^{\\mathrm{T}}r_j|$ 的归一化值会跳到 $0.04$ 甚至 $0.2$——那不是算法失效，"
                "是分子分母都只剩噪声。第三，坐标轴自适应会把「子空间已经张开到整个 $\\mathbb{R}^{n}$」画成"
                "「向量还在变长」，要读 meta.subspace_dim 与 $H$ 的次对角元（它变成 $0$ 才是子空间不变）。"
            ),
            "theory": [
                "$K_k(A,r_0)=\\operatorname{span}\\{r_0,Ar_0,\\ldots,A^{k-1}r_0\\}$；Arnoldi 给出 $AV_m=V_{m+1}\\bar{H}_m$",
                "CG：$r_k\\perp K_k$ 且 $p_i^{\\mathrm{T}}Ap_j=0$（$i\\ne j$）；GMRES：$r_k\\perp A K_k$（最小残差的法方程）",
                "精确算术下 Krylov 方法至多 $n$ 步终止（$H$ 的次对角元变 $0$ 即 happy breakdown）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "precond_effect",
            "title": "预条件前后：谱分布与收敛曲线联动",
            "description": (
                "左：$M^{-1}A$ 的 Ritz 谱散点（预条件前 vs 后）；右：PCG 的相对残差曲线与"
                "CG 收敛界参考线。角标实时给出实测 $\\kappa$、由迭代数反推的等效 $\\kappa$ 与 ILU(0) 丢弃的填充数。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "precond",
                    "label": "预条件子",
                    "type": "enum",
                    "options": ["none", "jacobi", "ssor", "ilu0"],
                    "default": "ssor",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "grid",
                    "label": "二维网格 k（n = k²）",
                    "type": "int",
                    "min": 4,
                    "max": 14,
                    "default": 8,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "omega",
                    "label": "SSOR 松弛因子 ω",
                    "type": "float",
                    "min": 0.5,
                    "max": 1.9,
                    "step": 0.05,
                    "default": 1.2,
                    "critical": [1.0],
                    "criticalNote": (
                        "$\\omega=1$ 时 SSOR 退化为对称 Gauss–Seidel（SSGS）；SSOR 预条件子要求 "
                        "$0<\\omega<2$ 才能保持对称正定，否则 PCG 的前提就不成立。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "solver",
                    "label": "迭代解法",
                    "type": "enum",
                    "options": ["cg", "gmres"],
                    "default": "cg",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "spectrum",
                    "autoscaled": False,
                    "title": "M⁻¹A 的谱（Ritz 值，按序号排列）",
                    "ylabel": "λ(M⁻¹A)",
                },
                {
                    "id": "convergence",
                    "autoscaled": True,
                    "title": "相对残差与 CG 收敛界（对数）",
                    "ylabel": "‖r_k‖/‖r_0‖",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "最常见的误读是「$\\kappa$ 降了 $\\sqrt{\\kappa}$ 倍，迭代数就该降 $\\sqrt{\\kappa}$ 倍」。实测（二维 $8\\times8$ Laplacian，"
                "$\\kappa_2=32.163$）：SSOR（$\\omega=1.2$）把 $\\kappa$ 压到 $3.586$（降了 $8.97$ 倍），迭代数只从 $23$ 降到 $14$；"
                "一维 $n=32$ 的 $\\kappa=440.7$，但由实测收敛因子 $0.5549$ 反推的等效 $\\kappa$ 只有 $12.2$——CG 的实际"
                "迭代数由**整个谱分布**决定，谱聚成几簇时远快于最坏界 $2{((\\sqrt{\\kappa}-1)/(\\sqrt{\\kappa}+1))}^{k}$（图上那条参考线"
                "就是这层「悲观」的实测证据）。第二条：Jacobi 预条件在一维 Laplacian 上**什么都没做**"
                "（对角是 $2I$，乘常数不改变任何谱比，实测 $\\kappa$ 仍是 $440.69$）——「预条件」不等于「一定有效」。"
                "第三条最隐蔽：这里的谱是 Lanczos 的 Ritz 值，只有**两端**收敛快，内部 Ritz 值收敛很慢，"
                "拿少量步数的谱去宣称「整个谱都量到了」是假的；ILU(0) 在一维三对角上更是直接退化成精确 LU"
                "（$\\kappa=1.00$、PCG 一步收敛），这个「完美」完全不能推广到二维"
                "（那里丢了 $98$ 次填充、$\\kappa$ 只到 $3.684$，GMRES $23$ 步只降到 $13$ 步）。"
            ),
            "theory": [
                "预条件 CG 解 $M^{-1}Ax=M^{-1}b$；$M$ 对称正定时 $M^{-1}A$ 关于内积 ${\\langle x,y\\rangle}_M=x^{\\mathrm{T}}My$ 自伴（不是 $M^{-1}$-内积），谱全实",
                "CG 收敛因子 $\\rho=(\\sqrt{\\kappa}-1)/(\\sqrt{\\kappa}+1)$ 是最坏界；实测因子反推的等效 $\\kappa={((1+\\rho)/(1-\\rho))}^{2}$",
                "SSOR：$M=(D+\\omega L)D^{-1}(D+\\omega U)/(\\omega(2-\\omega))$；ILU(0)：$M=\\tilde{L}\\tilde{U}$，结构 $=A$ 的结构（零填充）",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 计算入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "iteration_race":
        return _iteration_race(params, t)
    if viz_id == "error_spectrum":
        return _error_spectrum(params, t)
    if viz_id == "krylov_growth":
        return _krylov_growth(params, t)
    if viz_id == "precond_effect":
        return _precond_effect(params, t)
    raise KeyError(f"章节 1.5 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _step_of(t: float, n_steps: int) -> int:
    """把连续时间轴四舍五入到整数迭代步并夹到 ``[0, n_steps]``。"""
    k = int(np.floor(float(t) + 0.5))
    return max(0, min(k, max(0, n_steps)))


def _solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """直接解（走自研 LU；server 里不写数学，只调用 numcore）。"""
    res = lu_decompose(A, pivot="partial")
    if res.singular:
        raise ValueError("矩阵奇异：无法给出精确解")
    return lu_solve(res, b)


def _method_name(key: str) -> str:
    return {
        "jacobi": "Jacobi",
        "gs": "Gauss–Seidel",
        "sor": "SOR",
        "cg": "CG（共轭梯度）",
        "gmres": "GMRES(2)",
        "none": "无预条件",
        "ssor": "SSOR",
        "ilu0": "ILU(0)",
    }.get(key, key)


def _tail_factor(residuals: np.ndarray, *, skip: int = 2, tail: int = 8) -> float:
    """残差尾部的实测收敛因子；残差已经落到 0 或不足两个点时返回 NaN。"""
    r = np.asarray(residuals, dtype=float).ravel()[skip:]
    r = r[np.isfinite(r) & (r > 0.0)]
    if r.size < 3:
        return float("nan")
    try:
        return convergence_factor(r, skip=0, tail=min(tail, r.size - 1))
    except ValueError:
        return float("nan")


def _finite_or_none(x: float) -> float | None:
    """meta 里只允许有限数（JSON 没有 NaN/Infinity）：非有限一律写成 None + strings 说明。"""
    v = float(x)
    return v if np.isfinite(v) else None


def _pad(seq: Sequence[float], length: int) -> np.ndarray:
    """把轨迹补齐到统一步数：**重复最后一个点**（不是 NaN，避免非有限值污染整帧）。"""
    arr = np.asarray(seq, dtype=float).ravel()
    if arr.size >= length:
        return arr[:length]
    if arr.size == 0:
        return np.zeros(length, dtype=float)
    return np.concatenate([arr, np.full(length - arr.size, arr[-1], dtype=float)])


# --------------------------------------------------------------------------- #
# viz 1：迭代路径赛跑
# --------------------------------------------------------------------------- #
def _contours(A: np.ndarray, x_star: np.ndarray, c0: float, n_theta: int = 181) -> tuple[np.ndarray, np.ndarray]:
    """二次型 (x−x*)ᵀA(x−x*) = c 的椭圆等高线（每个 c 一行）。"""
    levels = np.array([0.5, 0.2, 0.05, 0.01, 0.002, 0.0004], dtype=float) * c0
    th = np.linspace(0.0, 2.0 * np.pi, n_theta)
    u = np.stack([np.cos(th), np.sin(th)], axis=1)  # (n_theta, 2)
    au = u @ A.T  # (n_theta, 2)
    quad = np.sum(u * au, axis=1)  # uᵀAu
    safe = np.where(quad > 0.0, quad, 1.0)
    xs = np.zeros((levels.size, n_theta), dtype=float)
    ys = np.zeros((levels.size, n_theta), dtype=float)
    for i, c in enumerate(levels):
        r = np.sqrt(c / safe)
        xs[i] = x_star[0] + r * u[:, 0]
        ys[i] = x_star[1] + r * u[:, 1]
    return xs, ys


def _iteration_race(params: Mapping[str, Any], t: float) -> Frame:
    method = str(params["method"])
    omega = float(params["omega"])
    kappa = float(params["kappa"])
    angle = float(params["angle"])
    A = rotated_spd_2x2(kappa, angle)
    b = np.asarray([1.0, 0.5], dtype=float)
    x_star = _solve_lu(A, b)
    x0 = np.zeros(2, dtype=float)
    n_steps = _MAX_STEPS
    k = _step_of(t, n_steps)

    names = ("jacobi", "gs", "sor", "cg", "gmres")
    race_x = np.zeros((len(names), n_steps + 1), dtype=float)
    race_y = np.zeros((len(names), n_steps + 1), dtype=float)
    race_res = np.zeros((len(names), n_steps + 1), dtype=float)
    race_len = np.zeros(len(names), dtype=float)
    converged = np.zeros(len(names), dtype=float)
    factors = np.zeros(len(names), dtype=float)

    for i, nm in enumerate(names):
        seq_x, seq_y, seq_r, n_real, ok = _race_trajectory(nm, A, b, x0, kappa, omega, n_steps)
        race_x[i] = _pad(seq_x, n_steps + 1)
        race_y[i] = _pad(seq_y, n_steps + 1)
        race_res[i] = _pad(seq_r, n_steps + 1)
        race_len[i] = float(n_real)
        converged[i] = 1.0 if ok else 0.0
        factors[i] = _tail_factor(np.asarray(seq_r, dtype=float))

    idx = names.index(method)
    mu = _jacobi_mu_2x2(A)
    omega_opt = float(2.0 / (1.0 + np.sqrt(max(1.0 - mu * mu, 0.0))))
    rho_sor = sor_spectral_radius(mu, omega) if mu < 1.0 else float("nan")
    c0 = float((x0 - x_star) @ (A @ (x0 - x_star)))
    cx, cy = _contours(A, x_star, c0)

    fr = Frame(chapter="1.5", viz="iteration_race", kind="frame")
    fr.add("contour_x", cx, unit="1", label="等高线 x（每行一个水平）")
    fr.add("contour_y", cy, unit="1", label="等高线 y")
    fr.add("race_x", race_x, unit="1", label="五条轨迹的 x（行 = 方法）")
    fr.add("race_y", race_y, unit="1", label="五条轨迹的 y")
    fr.add("race_residual", race_res, unit="1", label="五条轨迹的相对残差")
    fr.add("race_len", race_len, unit="1", label="各方法真正跑了几步（之后是重复点）")
    fr.add("traj_mark_x", np.array([race_x[idx, k]], dtype=float), unit="1", label="当前方法的第 k 步 x")
    fr.add("traj_mark_y", np.array([race_y[idx, k]], dtype=float), unit="1", label="当前方法的第 k 步 y")
    fr.add("residual", race_res[idx][: k + 1], unit="1", label="当前方法的相对残差（到第 k 步）")
    fr.add("residual_mark", np.array([race_res[idx, k]], dtype=float), unit="1", label="第 k 步的相对残差")
    fr.add("step_grid", np.arange(n_steps + 1, dtype=float), unit="1", label="迭代步")
    fr.add("x_star", x_star, unit="1", label="精确解 x*")
    fr.add("b_vec", np.asarray([0.0, b[0], 0.0, b[1]], dtype=float), unit="1", label="b（两段折线）")
    fr.set_meta(
        method_index=idx,
        n_methods=len(names),
        step=k,
        n_steps=n_steps,
        kappa=kappa,
        kappa_actual=_kappa_2x2(A),
        mu_jacobi=mu,
        omega=omega,
        omega_opt=omega_opt,
        rho_jacobi=mu,
        rho_gs=mu * mu,
        rho_sor=rho_sor,
        rho_sor_measured=_finite_or_none(_rho_measured_sor(A, omega)),
        threshold_omega=2.0,
        residual=float(race_res[idx, k]),
        factor_measured=_finite_or_none(factors[idx]),
        n_iter_current=float(race_len[idx]),
        converged_current=bool(converged[idx]),
        cg_iters=float(race_len[names.index("cg")]),
        gmres_iters=float(race_len[names.index("gmres")]),
        overflow=bool(np.max(np.abs(np.concatenate([cx.ravel(), cy.ravel()]))) > 1e6),
    )
    fr.set_strings(
        method=method,
        method_name=_method_name(method),
        note=(
            f"{_method_name(method)}：第 {k} 步相对残差 {float(race_res[idx, k]):.3e}；"
            f"μ={mu:.4f}，ω_opt={omega_opt:.4f}，ρ(SOR,ω)={rho_sor:.4f}"
            if np.isfinite(rho_sor)
            else f"{_method_name(method)}：第 {k} 步相对残差 {float(race_res[idx, k]):.3e}"
        ),
        race_order=" / ".join(_method_name(nm) for nm in names),
    )
    return fr


def _race_trajectory(
    method: str,
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    kappa: float,
    omega: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """跑一条轨迹，返回 ``(x 序列, y 序列, 相对残差序列, 真实步数, 是否收敛)``。"""
    if method in ("jacobi", "gs", "sor"):
        res: IterationResult = (
            jacobi(A, b, x0=x0, maxiter=n_steps, tol=1e-12)
            if method == "jacobi"
            else gauss_seidel(A, b, x0=x0, maxiter=n_steps, tol=1e-12)
            if method == "gs"
            else sor(A, b, x0=x0, omega=omega, maxiter=n_steps, tol=1e-12)
        )
        it = res.iterates
        return it[:, 0], it[:, 1], res.residuals, int(res.n_iter), bool(res.converged)
    if method == "cg":
        cg = conjugate_gradient(A, b, x0=x0, tol=1e-12, maxiter=n_steps, track_orthogonality=False)
        return cg.iterates[:, 0], cg.iterates[:, 1], cg.residuals, int(cg.n_iter), bool(cg.converged)
    gm = gmres(A, b, m=2, tol=1e-12, maxiter=n_steps, x0=x0, track_orthogonality=False)
    return gm.iterates[:, 0], gm.iterates[:, 1], gm.residuals, int(gm.n_iter), bool(gm.converged)


def _jacobi_mu_2x2(A: np.ndarray) -> float:
    r"""2×2 矩阵上 Jacobi 迭代矩阵的解析谱半径 :math:`\mu=\sqrt{|a_{12}a_{21}|/(a_{11}a_{22})}`。"""
    return float(np.sqrt(abs(A[0, 1] * A[1, 0]) / abs(A[0, 0] * A[1, 1])))


def _kappa_2x2(A: np.ndarray) -> float:
    """2×2 对称正定矩阵的条件数（用迹与行列式的闭式，不走任何现成求解器）。"""
    tr = float(A[0, 0] + A[1, 1])
    det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    disc = max(tr * tr - 4.0 * det, 0.0)
    root = float(np.sqrt(disc))
    lam_hi = 0.5 * (tr + root)
    lam_lo = 0.5 * (tr - root)
    return lam_hi / lam_lo if lam_lo > 0.0 else float("inf")


def _rho_measured_sor(A: np.ndarray, omega: float) -> float:
    """SOR 迭代矩阵谱半径的手写幂法实测值（与 Young 解析式对照，与当前高亮的方法无关）。"""
    return power_method(iteration_matrix(A, method="sor", omega=omega)).value


# --------------------------------------------------------------------------- #
# viz 2：误差的特征分量分解
# --------------------------------------------------------------------------- #
def _initial_error(n: int, init: str) -> np.ndarray:
    """构造初始误差向量（在正弦模态基下的系数）。"""
    V = sine_eigenvectors_1d(n)
    if init == "low_mode":
        c = np.zeros(n, dtype=float)
        c[0] = 1.0
    elif init == "high_mode":
        c = np.zeros(n, dtype=float)
        c[-1] = 1.0
    else:
        c = np.ones(n, dtype=float)
    return V @ c


def _error_spectrum(params: Mapping[str, Any], t: float) -> Frame:
    n = int(params["n"])
    method = str(params["method"])
    omega = float(params["omega"])
    init = str(params["init"])
    A = laplacian_1d(n)
    b = np.linspace(1.0, 2.0, n)
    x_star = _solve_lu(A, b)
    e0 = _initial_error(n, init)
    x0 = x_star + e0
    n_steps = _MAX_STEPS
    k = _step_of(t, n_steps)

    if method == "jacobi":
        res = jacobi(A, b, x0=x0, maxiter=n_steps, tol=0.0, x_exact=x_star)
    elif method == "gs":
        res = gauss_seidel(A, b, x0=x0, maxiter=n_steps, tol=0.0, x_exact=x_star)
    else:
        res = sor(A, b, x0=x0, omega=omega, maxiter=n_steps, tol=0.0, x_exact=x_star)

    errs = res.iterates - x_star[None, :]
    C = np.abs(mode_components(errs))  # (k+1, n)
    n_show = min(n, 16)
    comp = C[:, :n_show].T  # (n_show, k+1)
    rates_theory = mode_decay_factors(n, method=method, omega=omega)[:n_show]
    rates_measured_raw = _measured_mode_rates(comp)
    rates_valid = np.isfinite(rates_measured_raw)
    rates_measured = np.where(rates_valid, rates_measured_raw, -1.0)
    floor = 1e-16
    comp_log = np.log10(np.maximum(comp, floor))

    e_norm = np.zeros(k + 1, dtype=float)
    nb = float(np.sqrt(b @ b)) or 1.0
    for i in range(k + 1):
        e_norm[i] = float(np.sqrt(errs[i] @ errs[i]))
    rel_err = e_norm / (e_norm[0] if e_norm[0] > 0.0 else 1.0)
    rho_theory = _rho_theory_1d(n, method, omega)
    rho_meas = power_method(iteration_matrix(A, method=method, omega=omega)).value

    fr = Frame(chapter="1.5", viz="error_spectrum", kind="frame")
    fr.add("components", comp, unit="1", label="各模态误差分量 |v_jᵀe_k|（行 = 模态 j）")
    fr.add("components_log10", comp_log, unit="1", label="分量的 log10（地板 1e-16）")
    fr.add("components_mark", comp[:, k].copy(), unit="1", label="第 k 步的分量")
    fr.add("mode_rates_theory", rates_theory, unit="1", label="理论衰减因子 |λ_j(T)|")
    fr.add("mode_rates_measured", rates_measured, unit="1", label="实测比值（尾部几何平均；-1 = 未测到）")
    fr.add("mode_rates_measured_valid", rates_valid.astype(float), unit="1", label="实测比值是否有效（1/0）")
    fr.add("modes_index", np.arange(n_show, dtype=float), unit="1", label="模态序号 j−1")
    fr.add("error", rel_err[: k + 1], unit="1", label="相对误差 2-范数 ‖e_k‖/‖e_0‖")
    fr.add("residual", res.residuals[: k + 1], unit="1", label="相对残差 ‖r_k‖/‖r_0‖")
    fr.add("error_A", _a_norm_history(A, errs, e_norm[0])[: k + 1], unit="1", label="相对 A-范数误差")
    fr.add("residual_mark", np.array([res.residuals[k]], dtype=float), unit="1", label="第 k 步的相对残差")
    fr.add("error_mark", np.array([rel_err[k]], dtype=float), unit="1", label="第 k 步的相对误差")
    fr.add("step_grid", np.arange(k + 1, dtype=float), unit="1", label="迭代步")
    fr.set_meta(
        n=n,
        method_index=["jacobi", "gs", "sor"].index(method),
        step=k,
        n_steps=n_steps,
        n_modes_shown=n_show,
        omega=omega,
        component_floor=floor,
        rho_theory=float(rho_theory),
        rho_measured=float(rho_meas),
        factor_measured=_finite_or_none(_tail_factor(res.residuals)),
        mu_jacobi=float(np.cos(np.pi / (n + 1))),
        omega_opt=float(2.0 / (1.0 + np.sqrt(max(1.0 - np.cos(np.pi / (n + 1)) ** 2, 0.0)))),
        residual=float(res.residuals[k]),
        error_current=float(rel_err[k]),
        error_final=float(rel_err[k]),
        slowest_mode=int(np.argmax(rates_theory)) if rates_theory.size else 0,
        slowest_rate=_finite_or_none(np.max(rates_theory)) if rates_theory.size else None,
        n_modes_measured=int(np.count_nonzero(rates_valid)),
        overflow=False,
    )
    fr.set_strings(
        method=method,
        method_name=_method_name(method),
        init=init,
        init_name={
            "all_modes": "全模态等幅",
            "low_mode": "只激发最低模态 v_1",
            "high_mode": "只激发最高模态 v_n",
        }.get(init, init),
        note=(
            f"第 {k} 步：最慢模态 j={int(np.argmax(rates_theory)) + 1} 的理论因子 "
            f"{float(np.max(rates_theory)):.6f}，ρ(T) 实测 {float(rho_meas):.6f}，"
            f"残差 {float(res.residuals[k]):.2e}"
        ),
    )
    return fr


def _rho_theory_1d(n: int, method: str, omega: float) -> float:
    """一维 Laplacian 上迭代矩阵谱半径的解析值（Jacobi: μ₁；GS: μ₁²；SOR: Young 关系）。"""
    mu = float(np.cos(np.pi / (n + 1)))
    if method == "jacobi":
        return mu
    if method == "gs":
        return mu * mu
    return float(sor_spectral_radius(mu, omega))


def _measured_mode_rates(comp: np.ndarray) -> np.ndarray:
    """每个模态的实测衰减因子：在其幅值高于 1e-14 的窗口里取比值的几何平均。"""
    m = comp.shape[0]
    out = np.full(m, np.nan, dtype=float)
    for j in range(m):
        row = comp[j]
        mask = row > 1e-14
        vals = row[mask]
        if vals.size >= 3:
            ratios = vals[1:] / vals[:-1]
            out[j] = float(np.exp(np.mean(np.log(ratios))))
    return out


def _a_norm_history(A: np.ndarray, errs: np.ndarray, e0: float) -> np.ndarray:
    """相对 A-范数误差序列（CG 单调下降的就是它；这里是定常迭代的对照量）。"""
    out = np.zeros(errs.shape[0], dtype=float)
    for i in range(errs.shape[0]):
        e = errs[i]
        v = float(e @ (A @ e))
        out[i] = float(np.sqrt(max(v, 0.0))) / (e0 if e0 > 0.0 else 1.0)
    return out


# --------------------------------------------------------------------------- #
# viz 3：Krylov 子空间逐步扩张
# --------------------------------------------------------------------------- #
def _krylov_matrix(dim: int, stretch: float) -> np.ndarray:
    """D S D：S 是小规模 Laplacian，D = diag(1, …, s) 的几何反差对角（对称，故仍 SPD）。"""
    S = laplacian_1d(dim)
    d = np.linspace(1.0, float(stretch), dim) if dim > 1 else np.ones(1)
    return (d[:, None] * S) * d[None, :]


def _krylov_growth(params: Mapping[str, Any], t: float) -> Frame:
    dim = int(params["dim"])
    method = str(params["method"])
    stretch = float(params["stretch"])
    A = _krylov_matrix(dim, stretch)
    b = np.linspace(1.0, 2.0, dim)
    x_star = _solve_lu(A, b)
    k = _step_of(t, dim)

    if method == "cg":
        run = conjugate_gradient(A, b, tol=0.0, maxiter=dim, x_exact=x_star, track_basis=True)
        basis = _cg_basis(run.r_basis, dim, run.abs_residuals)
        orth = run.residual_orthogonality
        orth_name = "|r_iᵀr_j| / (‖r_i‖‖r_j‖)"
        hess = np.zeros((0, 0), dtype=float)
        resid = run.residuals
        iter_x = run.iterates
        basis_orth = None
        happy = False
        orth_self = _orthogonality_offdiag(run.residual_orthogonality, run.abs_residuals)
    else:
        run_g = gmres(A, b, m=dim, tol=0.0, maxiter=dim, x_exact=x_star, track_orthogonality=True)
        basis = run_g.basis
        orth = run_g.residual_vs_basis
        orth_name = "|r_iᵀ(Av_j)| / (‖r_i‖‖Av_j‖)"
        hess = run_g.H
        resid = run_g.residuals
        iter_x = run_g.iterates
        basis_orth = float(run_g.basis_orthogonality)
        happy = bool(run_g.happy_breakdown)
        n_orth = orth.shape[0]
        orth_self = _orthogonality_offdiag(orth, run_g.abs_residuals)

    n_cols = basis.shape[1] if basis.size else 0
    cols = min(n_cols, k + 1)
    basis_show = basis[:, :cols].T if cols else np.zeros((0, dim), dtype=float)  # (列, 维数)
    iter_show = iter_x[: min(k + 1, iter_x.shape[0])]

    fr = Frame(chapter="1.5", viz="krylov_growth", kind="frame")
    fr.add("basis", basis_show, unit="1", label="Krylov 基向量（行 = v_j）")
    fr.add("basis_grid", np.arange(basis_show.shape[0], dtype=float), unit="1", label="基向量序号")
    fr.add("iterates", iter_show, unit="1", label="解轨迹 x_k")
    fr.add("x_star", x_star, unit="1", label="精确解 x*")
    fr.add("orthogonality", orth, unit="1", label=orth_name)
    fr.add("residual", resid[: min(k + 1, resid.size)], unit="1", label="相对残差")
    fr.add("residual_mark", np.array([resid[min(k, resid.size - 1)]], dtype=float), unit="1", label="第 k 步残差")
    fr.add("step_grid", np.arange(min(k + 1, resid.size), dtype=float), unit="1", label="Krylov 步")
    fr.add("hessenberg", hess, unit="1", label="Arnoldi 的 H（GMRES 才有）")
    fr.set_meta(
        dim=dim,
        method_index=["cg", "gmres"].index(method),
        step=k,
        n_steps=dim,
        stretch=stretch,
        n_basis=int(basis_show.shape[0]),
        subspace_dim=float(min(k + 1, dim)),
        kappa_measured=_kappa_small(A),
        orthogonality_max=float(orth_self),
        basis_orthogonality=basis_orth,
        arnoldi_orthogonality=basis_orth,
        residual=float(resid[min(k, resid.size - 1)]),
        residual_final=float(resid[-1]),
        n_iter=int(resid.size - 1),
        happy_breakdown=happy,
        overflow=False,
    )
    fr.set_strings(
        method=method,
        method_name="CG（共轭梯度）" if method == "cg" else "GMRES(3)",
        orthogonality_name=orth_name,
        note=(
            f"第 {k} 步：Krylov 子空间维数 {min(k + 1, dim)}/{dim}，"
            f"相对残差 {float(resid[min(k, resid.size - 1)]):.3e}，"
            f"正交性矩阵的下三角最大值 {orth_self:.2e}"
        ),
    )
    return fr


def _orthogonality_offdiag(G: np.ndarray, abs_residuals: np.ndarray) -> float:
    """正交性矩阵下三角的最大值，**屏蔽掉落在舍入底之下的那些行**。

    残差小于 sqrt(u)·‖r_0‖ 之后，r_i 只是噪声，任何归一化内积都会给出 O(1) 的假值；
    把这种行算进来会让"正交性"这个指标彻底失去意义（实测能从 1e-16 跳到 0.7）。
    """
    n = G.shape[0]
    r = np.asarray(abs_residuals, dtype=float).ravel()
    floor = np.sqrt(np.finfo(float).eps) * (r[0] if r.size and r[0] > 0.0 else 1.0)
    keep = [i for i in range(min(n, r.size)) if r[i] > floor]
    best = 0.0
    for i in keep:
        for j in range(min(i, n)):
            if j in keep or r[j] > floor:
                best = max(best, float(abs(G[i, j])))
    return best


def _cg_basis(r_basis: np.ndarray, dim: int, abs_residuals: np.ndarray) -> np.ndarray:
    """CG 的"Krylov 基"用归一化残差序列充当（r_0..r_k 张成的正是 K_{k+1}）。

    落在舍入底之下的残差不进基：它只是噪声方向，画出来会让人以为"子空间还在扩张"。
    """
    if r_basis.size == 0:
        return np.zeros((dim, 0), dtype=float)
    norms = np.sqrt(np.sum(r_basis * r_basis, axis=1))
    floor = np.sqrt(np.finfo(float).eps) * (norms[0] if norms.size and norms[0] > 0.0 else 1.0)
    keep = int(np.count_nonzero(norms > floor))
    keep = max(keep, 1)
    rows = r_basis[:keep] / norms[:keep, None]
    return rows.T


def _kappa_small(A: np.ndarray) -> float:
    """小矩阵条件数的谱实测（2×2 用闭式，3×3 用迹/行列式的特征多项式闭式不合适，改用幂法两端）。"""
    n = A.shape[0]
    if n == 2:
        return _kappa_2x2(A)
    # 3×3：Lanczos 的 Ritz 两端（m=n 时就是精确谱；幂法只能给 |λmax|，量不到 λmin）
    sp = precond_spectrum(A, m=n)
    return float(sp.kappa_estimate)


# --------------------------------------------------------------------------- #
# viz 4：预条件前后
# --------------------------------------------------------------------------- #
def _precond_effect(params: Mapping[str, Any], t: float) -> Frame:
    name = str(params["precond"])
    grid = int(params["grid"])
    omega = float(params["omega"])
    solver = str(params["solver"])
    A_dense = laplacian_2d(grid)
    A = CSR.from_dense(A_dense)
    n = A.shape[0]
    b = np.linspace(1.0, 2.0, n)
    x_star = _solve_lu(A_dense, b)
    M = build_precond(name, A_dense, omega=omega)
    m_ritz = min(n, _MAX_RITZ)
    sp_pre = precond_spectrum(A, m=m_ritz)
    sp_post = precond_spectrum(A, precond=M, m=m_ritz) if M is not None else sp_pre

    if solver == "cg":
        run = conjugate_gradient(A, b, precond=M, tol=1e-10, maxiter=4 * n + 10, x_exact=x_star, track_orthogonality=False)
        resid = run.residuals
        errs = run.errors_2
    else:
        run_g = gmres(A, b, precond=M, m=min(n, 40), tol=1e-10, maxiter=4 * n + 10, x_exact=x_star, track_orthogonality=False)
        resid = run_g.residuals
        errs = run_g.errors_2

    kappa_pre = float(sp_pre.kappa_estimate)
    kappa_post = float(sp_post.kappa_estimate)
    factor = _tail_factor(resid)
    kappa_eff = kappa_from_factor(factor) if np.isfinite(factor) and 0.0 <= factor < 1.0 else float("nan")
    bound = np.array(
        [_cg_bound(kappa_post, kk) for kk in range(resid.size)],
        dtype=float,
    )
    e0 = errs[0] if errs.size and errs[0] > 0.0 else 1.0
    rel_err = errs / e0

    fr = Frame(chapter="1.5", viz="precond_effect", kind="frame")
    fr.add("ritz_pre", sp_pre.ritz, unit="1", label="A 的 Ritz 值（预条件前）")
    fr.add("ritz_post", sp_post.ritz, unit="1", label="M⁻¹A 的 Ritz 值（预条件后）")
    fr.add("ritz_index_pre", np.arange(sp_pre.ritz.size, dtype=float), unit="1", label="谱序号（前）")
    fr.add("ritz_index_post", np.arange(sp_post.ritz.size, dtype=float), unit="1", label="谱序号（后）")
    fr.add("convergence", resid, unit="1", label="相对残差")
    fr.add("residual", resid, unit="1", label="相对残差（requiredAux）")
    fr.add("error", rel_err, unit="1", label="相对误差 ‖x_k−x*‖/‖x_0−x*‖")
    fr.add("bound", bound, unit="1", label="CG 收敛界 2((√κ−1)/(√κ+1))^k（用实测 κ_post）")
    fr.add("step_grid", np.arange(resid.size, dtype=float), unit="1", label="迭代步")
    fr.add("residual_mark", np.array([resid[-1]], dtype=float), unit="1", label="末步残差")
    fr.set_meta(
        precond_index=["none", "jacobi", "ssor", "ilu0"].index(name),
        grid=grid,
        n=n,
        omega=omega if name == "ssor" else 0.0,
        solver_index=["cg", "gmres"].index(solver),
        kappa_pre=kappa_pre,
        kappa_post=kappa_post,
        kappa_ratio=kappa_pre / kappa_post if kappa_post > 0.0 else float("inf"),
        kappa_effective=_finite_or_none(kappa_eff),
        n_iter=int(resid.size - 1),
        converged=bool(resid[-1] <= 1e-10),
        factor_measured=_finite_or_none(factor),
        lambda_min_pre=sp_pre.lambda_min,
        lambda_max_pre=sp_pre.lambda_max,
        lambda_min_post=sp_post.lambda_min,
        lambda_max_post=sp_post.lambda_max,
        n_ritz_pre=int(sp_pre.ritz.size),
        n_ritz_post=int(sp_post.ritz.size),
        spectrum_steps=int(sp_pre.n_steps),
        spectrum_orthogonality=float(sp_pre.orthogonality),
        # -1 = "这个预条件子没有丢弃概念"（Jacobi/SSOR），**不是**"丢弃了 -1 个填充"。
        # 哨兵必须能自解释：所以同时给出语义字符串与显式布尔，前端不必猜 -1 的含义。
        ilu_dropped=int(M.dropped) if isinstance(M, ILU0Precond) else -1,
        ilu_dropped_applicable=bool(isinstance(M, ILU0Precond)),
        ilu_dropped_sentinel=-1,
        precond_nnz=int(M.nnz) if M is not None else 0,
        matrix_nnz=int(A.nnz),
        residual=float(resid[-1]),
        error_final=float(rel_err[-1]) if np.isfinite(rel_err[-1]) else None,
        overflow=False,
    )
    fr.set_strings(
        precond=name,
        precond_name=_method_name(name),
        solver=solver,
        note=_precond_note(name, kappa_pre, kappa_post, int(resid.size - 1), kappa_eff, M),
        sentinel_note=(
            "meta.ilu_dropped = -1 表示「这个预条件子没有丢弃概念」（Jacobi/SSOR），"
            "不是「丢弃了 −1 个填充」；用 meta.ilu_dropped_applicable 判是否有意义。"
        ),
    )
    return fr


def _cg_bound(kappa: float, k: int) -> float:
    """CG 收敛界 2((√κ−1)/(√κ+1))^k；κ 非有限时给 1（画不出下降就画平的参考线）。"""
    if not np.isfinite(kappa) or kappa < 1.0:
        return 1.0
    s = float(np.sqrt(kappa))
    return float(min(2.0 * ((s - 1.0) / (s + 1.0)) ** int(k), 1.0))


def _precond_note(name: str, kappa_pre: float, kappa_post: float, n_iter: int, kappa_eff: float, M: Any) -> str:
    extra = ""
    if isinstance(M, ILU0Precond):
        extra = f"，ILU(0) 丢弃 {M.dropped} 次填充更新"
    if name == "jacobi":
        extra = "，Jacobi 只改对角尺度"
    eff = f"，等效 κ（由迭代数反推）{kappa_eff:.2f}" if np.isfinite(kappa_eff) else ""
    return (
        f"{_method_name(name)}：实测 κ {kappa_pre:.2f} → {kappa_post:.2f}"
        f"（降 {kappa_pre / kappa_post if kappa_post > 0 else float('inf'):.2f} 倍），"
        f"迭代 {n_iter} 步{eff}{extra}"
    )
