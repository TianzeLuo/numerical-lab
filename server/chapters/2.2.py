r"""2.2 章：样条与分段逼近。

四个可视化分工如下（全部数据来自 :mod:`numcore.interp.spline`，本文件只做搬运与装配）：

1. ``control_points`` —— 拖动数据点看曲线与**曲率梳**实时变化；``bc`` 切换自然/夹持/
   not-a-knot 三种边界，``tension`` 把弯矩整体乘 :math:`(1-\tau)`。
   :math:`\tau=1` 时曲线退化成折线，副轴的 :math:`|\Delta S'|` 会从 :math:`10^{-16}`
   跳到 :math:`10^{-1}` —— "张力"不是"更平滑"，是把 :math:`C^1` 交给折线。
2. ``smoothness_c0c1c2`` —— 折线 / PCHIP（:math:`C^1`）/ 样条（:math:`C^2`）三条曲线，
   二阶导**在节点两侧分别采样**（:math:`x_i\pm\epsilon`，含端点断开），于是
   :math:`C^1` 的阶梯跳变是竖直段而不是被连成斜坡；同一批数据再给一份"默认等距采样"
   的二阶导作为对照，让"跳变被采样抹掉"这件事看得见。
3. ``bspline_local`` —— B 样条的支撑区间与**节点插入动画**：改一个控制点只动
   :math:`k+1` 个区间（``changed`` 数组在区间外恒为 0），插入节点前后曲线逐点不变
   （``error`` 贴 :math:`u`），基函数在支撑外恒为 0（``residual``）。
4. ``spline_vs_poly`` —— 同一组节点上的高阶多项式与自然样条：多项式误差指数爆炸、
   样条 :math:`O(h^4)` 下降，误差轴取 log 并显式给出**舍入地板**；主轴**固定**并把
   溢出如实报告（规则 1：不许靠自动裁剪把发散藏起来）。

约定（与 0.0/1.2 章一致）：``meta`` 只放 number/bool/null，字符串一律走 ``set_strings``；
``time.mode="none"`` 的 viz 不声明任何时间控件；每个 viz 的 ``requiredAux`` 里那个
``error``/``residual`` 必须是帧里**真实存在**的数组名。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from numcore.interp.spline import (
    UNIT_ROUNDOFF,
    basis_support,
    bspline_basis_matrix,
    chebyshev_nodes,
    convex_hull_gap,
    cubic_spline,
    cubic_from_moments,
    curvature_comb,
    curvature_of,
    de_boor,
    insertion_sequence,
    interpolating_polynomial_energy,
    lagrange_value,
    lebesgue_constant,
    pchip,
    polyline,
    sample_function,
    spline_system,
    uniform_clamped_knots,
    uniform_nodes,
)
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 对数轴上"0 画不出来"的统一地板：半个单位舍入（与 1.2 章同一口径）
FLOOR = UNIT_ROUNDOFF
#: 用于"节点两侧"采样的默认相对偏移（相对区间长度）
NODE_EPS_REL = 0.02
_BC_NAME = {"natural": "自然（M0=Mn=0）", "clamped": "夹持（给定端点斜率）", "not_a_knot": "not-a-knot"}
#: 枚举的**数值**编号：meta 只允许标量数字，字符串一律走 set_strings（PROTOCOLS §1.1）
_BC_INDEX = {"natural": 0, "clamped": 1, "not_a_knot": 2}
_DATASET_INDEX = {"runge": 0, "bump": 1, "steps": 2, "monotone": 3}
_NODES_INDEX = {"uniform": 0, "chebyshev": 1}
_FUNCTION_INDEX = {"runge": 0, "sine": 1, "bump": 2, "cusp": 3}
_DATASET_NAME = {"runge": "Runge 1/(1+25x^2)", "bump": "高斯包 exp(-16x^2)", "steps": "台阶数据（阶跃）", "monotone": "单调数据"}
_NODES_NAME = {"uniform": "等距节点", "chebyshev": "Chebyshev--Lobatto 节点"}

MANIFEST: dict[str, Any] = {
    "chapter": "2.2",
    "title": "样条与分段逼近",
    "part": "II",
    "summary": (
        "三次样条的 $C^2$ 条件是一个三对角系统；B 样条的局部支撑让「改一个控制点只动 $k+1$ 段」；"
        "而「样条比多项式稳」必须在同一组数据上量出来——四个可视化把所有结论都变成可测量的数字。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "control_points",
            "title": "控制点、张力与曲率梳",
            "description": (
                "拖动 6 个数据点的纵坐标，曲线与曲率梳实时更新；bc 切换自然/夹持/not-a-knot，"
                "tension 把弯矩乘 $(1-\\tau)$。副轴给出节点处的一阶导跳变 $|dS'|$ 与二阶导跳变 $|dS''|$，"
                "主轴固定坐标以便横向比较。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "y0",
                    "label": "第 1 点 y",
                    "type": "float",
                    "min": -3.0,
                    "max": 3.0,
                    "step": 0.05,
                    "default": 0.4,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "第 1 个数据点的纵坐标（无单位，范围 -3 到 3）。调大或调小会抬升或压低左端锚点，"
                        "曲线在左端附近和曲率梳随之改变；它还会影响自然与夹持边界下的端点斜率。"
                        "看主图左端、跳变面板与曲率面板，和相邻的 y1、边界条件 bc 一起读。"
                    ),
                },
                {
                    "name": "y1",
                    "label": "第 2 点 y",
                    "type": "float",
                    "min": -3.0,
                    "max": 3.0,
                    "step": 0.05,
                    "default": 1.8,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "第 2 个数据点的纵坐标（无单位，范围 -3 到 3）。它是左端第二个锚点，调大时"
                        "曲线在左半段整体上抬、相邻两段的斜率变陡；调小则压出一个局部低谷。"
                        "看主图左中段与曲率梳方向，并注意它与 y0、y2 的相对高低；边界条件会改变端点附近的影响范围。"
                    ),
                },
                {
                    "name": "y2",
                    "label": "第 3 点 y",
                    "type": "float",
                    "min": -3.0,
                    "max": 3.0,
                    "step": 0.05,
                    "default": -0.6,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "第 3 个数据点的纵坐标（无单位，范围 -3 到 3）。它控制曲线中左部的拐点位置："
                        "调高会让这一段向上拱，调低会形成更深的谷；曲率梳在中段的变化最明显。"
                        "看主图中间偏左的弯曲方向与曲率面板的符号，和 y1、y3 的高度差一起判断。"
                    ),
                },
                {
                    "name": "y3",
                    "label": "第 4 点 y",
                    "type": "float",
                    "min": -3.0,
                    "max": 3.0,
                    "step": 0.05,
                    "default": 1.4,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "第 4 个数据点的纵坐标（无单位，范围 -3 到 3）。它与 y2、y4 一起决定中右段的"
                        "起伏：调大时曲线在中部抬高，调小则把峰压平；跳变面板只在内点读一阶导差，"
                        "所以能看出这里是否出现尖角。看主图中右段、曲率梳长度和能量数值。"
                    ),
                },
                {
                    "name": "y4",
                    "label": "第 5 点 y",
                    "type": "float",
                    "min": -3.0,
                    "max": 3.0,
                    "step": 0.05,
                    "default": -1.2,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "第 5 个数据点的纵坐标（无单位，范围 -3 到 3）。它压住右半段的低点，调低会让"
                        "曲线在右端附近出现更深的谷，端点斜率也随之变化；调高则把低谷填平。"
                        "看主图右半段与曲率梳方向，和 y3、y5 的相对位置一起读。"
                    ),
                },
                {
                    "name": "y5",
                    "label": "第 6 点 y",
                    "type": "float",
                    "min": -3.0,
                    "max": 3.0,
                    "step": 0.05,
                    "default": 0.2,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "第 6 个数据点的纵坐标（无单位，范围 -3 到 3）。它是右端锚点，调大或调小会"
                        "整体抬升或压低曲线尾部，并改变右端点斜率；自然边界还会让端部二阶导固定为零。"
                        "看主图右端与曲率面板，和 y4、bc 一起判断端点外推是否可信。"
                    ),
                },
                {
                    "name": "bc",
                    "label": "边界条件",
                    "type": "enum",
                    "options": ["natural", "clamped", "not_a_knot"],
                    "default": "natural",
                    "critical": [],
                    "criticalNote": "",
                    "help": "样条的边界条件：natural 让两端二阶导为零；clamped 指定端点斜率；not-a-knot 让第三阶导在第二和倒数第二个节点连续。不同选择只改变两端的曲线，内部节点仍被插值。",
                },
                {
                    "name": "tension",
                    "label": "张力 tau（弯矩乘 1-tau）",
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.02,
                    "default": 0.0,
                    "critical": [1.0],
                    "criticalNote": (
                        "$\\tau=1$ 时弯矩全为 $0$，曲线退化成折线：仍然过所有数据点，但 $S'$ 在每个内点"
                        "跳变（$C^1$ 断裂）。$0<\\tau<1$ 是「弯矩被整体缩小」的中间态——它同样破坏 $C^1$，"
                        "所以别把张力当成「更平滑」的旋钮，它换的是「更平」，代价是节点处的折角。"
                    ),
                    "beyond": "above",
                    "help": "张力参数 $\\tau$：弯矩乘以 $1-\\tau$ 后再解。$\\tau=0$ 是标准三次样条；$\\tau$ 增大时刻线被拉直、曲率变小，$\\tau\\to1$ 接近分段线性。它不改变插值条件。",
                },
                {
                    "name": "comb",
                    "label": "曲率梳尺度",
                    "type": "enum",
                    "options": ["global", "local"],
                    "default": "global",
                    "critical": [],
                    "criticalNote": "",
                    "help": "曲率梳的尺度：global 用统一的放大倍数，便于比较整条曲线的曲率大小；local 按每段最大值归一化，突出形状变化但会掩盖绝对量级。",
                },
            ],
            "time": {"mode": "scrub", "tMax": 5.0, "dt": 0.05, "label": "探针位置 $x$"},
            "requiredAux": ["residual", "curvature"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "曲线、数据点与曲率梳（固定坐标）", "ylabel": "$y$"},
                {"id": "jump", "autoscaled": True, "title": "节点处的跳变", "ylabel": "$|\\mathrm{d}S'|,\\ |\\mathrm{d}S''|$", "scale": "log"},
                {"id": "curv", "autoscaled": True, "title": "曲率 kappa 与能量", "ylabel": "kappa"},
            ],
            "pitfalls": (
                "三条坑。第一，曲率梳的尺度：global 让所有梳齿按同一比例画，长度可比；"
                "local 把每条梳齿都归一成一样长，于是「哪里更弯」完全读不出来——只剩方向，"
                "切过去以后若还按长度下结论就是被骗。第二，tau 滑块看着像「阻尼/平滑」，"
                "其实它把弯矩乘 $(1-\\tau)$：$\\tau=0$ 时 $|dS'|$ 是 $6.7\\times10^{-16}$（机器精度），$\\tau=0.5$ 跳到 $2.30$，"
                "$\\tau=1$ 直接变成折线（$|dS'|=4.60$）；而折线的二阶导图是一条 $0$ 线，看起来「最光滑」，那是假象。"
                "第三，边界条件不是无关紧要的细节：同一组数据下端点斜率 $S'(x_0)$ 在 "
                "natural/clamped/not-a-knot 下是 $-1.27/-1.00/-1.92$（差 $1.9$ 倍），"
                "曲率能量是 $1.61/1.86/3.09$——差异全在端点上，而端点正是外推最敏感的地方。"
            ),
            "theory": [
                "**三次样条是分段三次、整体 C²。** 把区间分成 $n$ 段，每段写成一个三次多项式。要求函数值在节点连续、一阶导和二阶导也连续，就得到 $4n$ 个未知系数与 $4n-2$ 个方程；剩下两个自由度由边界条件补上。",
                "**用二阶导做未知量最方便。** 记 $M_i=s''(x_i)$。在每个小区间上，$s$ 是三次多项式，由两端的 $M_i,M_{i+1}$ 和函数值唯一确定；一阶导连续给出关于 $M$ 的三对角方程 $\\frac{h_{i-1}}{6}M_{i-1}+\\frac{h_{i-1}+h_i}{3}M_i+\\frac{h_i}{6}M_{i+1}=\\frac{y_{i+1}-y_i}{h_i}-\\frac{y_i-y_{i-1}}{h_{i-1}}.$ 解一个三对角系统即可。",
                "**边界条件决定端部。** natural 令 $M_0=M_n=0$；clamped 指定 $s'(x_0),s'(x_n)$；not-a-knot 要求第一、第二段是同一个三次多项式（三阶导在第二个节点连续）。三种选择只改变靠近端部的两段。",
                "**张力参数是对方程的修改。** 把弯矩项乘以 $1-\\tau$ 后再解，$\\tau=0$ 是标准样条；$\\tau$ 越大，解的曲率被压得越小，$\\tau\\to1$ 时接近折线。它不破坏插值条件，但会改变光滑性和形状。",
                "**曲率梳怎样读。** 曲线在每个节点的法向偏移量与曲率成正比；梳子穿过曲线表示曲率变号（拐点）。global 尺度下比较绝对曲率，local 尺度下比较形状；不要把两种尺度的数值直接对比。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "smoothness_c0c1c2",
            "title": "C0 / C1 / C2：二阶导的阶梯跳变",
            "description": (
                "同一组数据上画折线（$C^0$）、PCHIP（$C^1$）、三次样条（$C^2$）；二阶导在节点两侧 "
                "$x_i\\pm\\epsilon$ 各采一点，于是 $C^1$ 的跳变是竖直段。另给一条「默认等距采样」的二阶导作对照。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "dataset",
                    "label": "数据",
                    "type": "enum",
                    "options": ["runge", "bump", "steps", "monotone"],
                    "default": "steps",
                    "critical": [],
                    "criticalNote": "",
                    "help": "被插值的数据：runge 是光滑但会龙格的函数，bump 有局部隆起，steps 有跳跃，monotone 单调。数据越不光滑，高阶样条的振荡/过冲越明显。",
                },
                {
                    "name": "nodes",
                    "label": "节点分布",
                    "type": "enum",
                    "options": ["uniform", "chebyshev"],
                    "default": "uniform",
                    "critical": [],
                    "criticalNote": "",
                    "help": "节点分布：uniform 等距，chebyshev 在两端加密。对光滑函数，Chebyshev 节点能压低 Runge 型振荡；对台阶，加密也救不了跳跃本身。",
                },
                {"name": "n", "label": "节点数 $n$", "type": "int", "min": 5, "max": 15, "default": 9, "critical": [], "criticalNote": "", "help": "节点数。增加 $n$ 让分段更细，样条误差按 $h^4$ 下降；但分段线性的转折点也更多，C0 的“折角”仍存在。"},
                {
                    "name": "eps",
                    "label": "节点两侧采样偏移 eps",
                    "type": "float",
                    "min": 0.001,
                    "max": 0.2,
                    "step": 0.001,
                    "default": 0.03,
                    "critical": [],
                    "criticalNote": "",
                    "help": "节点数。增加 $n$ 让分段更细，样条误差按 $h^4$ 下降；但分段线性的转折点也更多，C0 的“折角”仍存在。",
                },
                {
                    "name": "bc",
                    "label": "C2 成员的边界条件",
                    "type": "enum",
                    "options": ["natural", "clamped", "not_a_knot"],
                    "default": "natural",
                    "critical": [],
                    "criticalNote": "",
                    "help": "C2 样条的边界条件（仅三次样条使用）。natural、clamped、not-a-knot 会影响端部几段的曲率；PCHIP 是局部 C1 方法，不受这个选项影响。",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["residual", "error"],
            "axes": [
                {"id": "main", "autoscaled": True, "title": "三条插值曲线", "ylabel": "$y$"},
                {"id": "d2", "autoscaled": True, "title": "二阶导 $S''$（节点两侧分别采样）", "ylabel": "$S''$"},
                {"id": "jump", "autoscaled": True, "title": "节点处的跳变量级", "ylabel": "|jump|", "scale": "log"},
            ],
            "pitfalls": (
                "默认等距采样是这张图最大的敌人：跳变恰好落在两个采样点之间时，二阶导会被"
                "连成一段陡坡，「不连续」看起来只是「变化快」。第二，折线的二阶导处处为 $0$，"
                "画出来是一条完美光滑的 $0$ 线——它连 $C^1$ 都不是，跳变藏在一阶导里，"
                "只看二阶导就会把最差的那条曲线当成最光滑的。第三，跳变的量级与节点分布有关"
                "（同一个台阶数据、$n=9$：等距节点上 PCHIP 的 $|dS''|=64$，Chebyshev 节点上 $85.1$），"
                "但它们**永远不可能**掉到 $u$ 量级——那是 $C^2$ 才有的性质，也是这张图要你分辨的那条线。"
            ),
            "theory": [
                "**光滑性由拼接条件决定。** 分段线性在节点处只有函数值连续，所以属于 $C^0$；每段斜率可以突变。PCHIP 的每段是三次，且节点一阶导连续，所以是 $C^1$；三次样条再加二阶导连续，达到 $C^2$。",
                "**PCHIP 的斜率怎么选。** 对每个节点用相邻差商做加权调和平均；当数据局部非单调时把斜率压到零，保证分段单调。这就是它不会在单调数据上产生过冲的原因，但代价是只有 $C^1$。",
                "**三次样条要解全局三对角系统。** 二阶导连续给出关于 $M_i$ 的方程，边界条件补两个端点，解出所有 $M_i$ 后再逐段构造三次多项式。因此移动一个节点会影响全曲线（虽然影响随距离衰减），不像 PCHIP 那样完全局部。",
                "**跳跃与尖点不能靠插值消除。** 台阶数据在节点两侧的差商不收敛到同一个导数；$\\varepsilon$ 是测量这个跳跃的采样偏移。加密网格只会让过渡区变窄，不会让 $C^1$ 方法真的穿过跳跃。",
                "**怎样读比较图。** 先看节点处的跳变（C0 有、C1/C2 无），再看二阶导的跳变（C2 无、C1 有）。曲线的振荡幅值则取决于数据光滑性和节点分布，不能只看“次数高”。此外，节点加密会缩小过渡区，但不会改变跳跃两侧的真实差值。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "bspline_local",
            "title": "B 样条：局部支撑与节点插入",
            "description": (
                "均匀夹紧节点上的 B 样条：画出基函数、控制多边形与曲线；拖动 move_index 对应的"
                "控制点，changed 数组只在 $k+1$ 个区间内非零。时间轴是节点插入次数，"
                "每插一次曲线不变（error 贴 $u$）、控制网更贴曲线。"
            ),
            "dataSource": "backend",
            "params": [
                {"name": "n_ctrl", "label": "控制点个数", "type": "int", "min": 6, "max": 12, "default": 8, "critical": [], "criticalNote": "", "help": "控制点个数。B 样条曲线由控制多边形控制，增加控制点通常让曲线更灵活；但次数固定时，可用的节点区间和局部形状也随之改变。"},
                {"name": "degree", "label": "次数 k", "type": "int", "min": 2, "max": 4, "default": 3, "critical": [], "criticalNote": "", "help": "B 样条次数 $k$。次数越高，曲线越光滑（$C^{k-1}$），每个控制点影响的区间也越宽；次数过高会带来更多振荡。"},
                {"name": "move_index", "label": "移动第几个控制点", "type": "int", "min": 0, "max": 11, "default": 2, "critical": [], "criticalNote": "", "help": "移动第几个控制点。B 样条具有局部支集：移动一个控制点只会改变它附近的 $k+1$ 段，远处的曲线严格不变。"},
                {"name": "move_delta", "label": "位移 delta", "type": "float", "min": -2.0, "max": 2.0, "step": 0.05, "default": 1.2, "critical": [], "criticalNote": "", "help": "控制点的位移量。位移越大，局部曲线被拉得越远；把 delta 设回 0 可以对照原始曲线，确认影响范围。"},
                {"name": "insert_at", "label": "插入参数 u", "type": "float", "min": 0.05, "max": 0.95, "step": 0.01, "default": 0.5, "critical": [], "criticalNote": "", "help": "在参数 $u$ 处插入一个节点。节点插入不改变曲线形状，只增加控制点、提高局部自由度；这是 CAD 中做局部编辑的标准操作。"},
            ],
            "time": {"mode": "scrub", "tMax": 4.0, "dt": 1.0, "label": "节点插入次数"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "曲线、控制多边形与基函数（固定参数域）", "ylabel": "$y$"},
                {"id": "invariance", "autoscaled": True, "title": "插入前后的逐点差", "ylabel": "|dC|", "scale": "log"},
                {"id": "support", "autoscaled": True, "title": "支撑外的基函数值（应恒为 0）", "ylabel": "$\\max|B_i|$（支撑外）", "scale": "log"},
            ],
            "pitfalls": (
                "第一，B 样条的控制点**不是曲线上的点**：曲线在控制多边形的凸包内，"
                "上图的控制多边形会明显偏离曲线，把控制点当「采样点」读会全错。"
                "第二，局部性有明确的边界：改第 $i$ 个控制点只影响 $[t_i,t_{i+k+1}]$，"
                "但这个区间有 $k+1$ 段宽，$k=3$ 时是 $4$ 段——「改一个点只影响一点点」在控制点很少时"
                "可能已经覆盖半条曲线，changed 数组的宽度才是真答案。"
                "第三，节点插入不改曲线（error 贴 $u$），它改的是**表示**：控制多边形越来越贴曲线，"
                "但自由度也越来越多；别把「插了节点」当成「曲线变了」。"
            ),
            "theory": [
                "**B 样条由递归定义。** 给定节点序列 $u_0\\le\\dots\\le u_{m}$，0 次基函数是区间指示函数；$k$ 次基函数由 Cox–de Boor 递推：$N_{i,k}(u)=\\frac{u-u_i}{u_{i+k}-u_i}N_{i,k-1}(u)+\\frac{u_{i+k+1}-u}{u_{i+k+1}-u_{i+1}}N_{i+1,k-1}(u)$。递推立刻给出局部支集：$N_{i,k}$ 只在 $[u_i,u_{i+k+1}]$ 上非零。",
                "**局部支集意味着局部修改。** 移动控制点 $P_i$ 只改变乘了 $N_{i,k}$ 的那一段，也就是相邻的 $k+1$ 个节点区间；其余基函数在该点为零，所以曲线严格不变。这正是 B 样条比全局多项式更适合 CAD 编辑的原因。",
                "**节点插入不改变曲线。** 把一个节点插入 $u$，同时按 de Boor 算法调整控制点，曲线作为集合完全不变，但控制多边形更贴近曲线、局部自由度增加。反复插入就是“节点细分”。",
                "**次数与光滑性。** $k$ 次 B 样条在单节点处至少 $C^{k-1}$；重节点会降低连续性（重数 $r$ 降到 $C^{k-r}$）。次数越高，支集越宽、越光滑，也更容易产生摆动。",
                "**参数怎样读。** n_ctrl 控制自由度，degree 控制光滑性与支集宽度，move_index/delta 演示局部性，insert_at 演示形状不变的节点插入。看曲线变化范围而不是只看控制点的走向。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "spline_vs_poly",
            "title": "样条 vs 高阶多项式：同一组数据的稳定性",
            "description": (
                "同一组节点上比较自然样条与 Lagrange 插值多项式：主轴固定坐标并报告溢出，"
                "副轴（log）给出两条误差曲线与**舍入地板**；再给出误差随 n 增长的整条历史。"
            ),
            "dataSource": "backend",
            "params": [
                {"name": "n", "label": "节点数 $n$", "type": "int", "min": 4, "max": 20, "default": 9, "critical": [], "criticalNote": "", "help": "插值节点数。对多项式，$n$ 增大不保证更好（Runge 现象）；对三次样条，$n$ 增大通常让误差按 $h^4$ 下降，直到舍入地板。"},
                {
                    "name": "function",
                    "label": "被插函数",
                    "type": "enum",
                    "options": ["runge", "sine", "bump", "cusp"],
                    "default": "runge",
                    "critical": [],
                    "criticalNote": "",
                    "help": "插值节点数。对多项式，$n$ 增大不保证更好（Runge 现象）；对三次样条，$n$ 增大通常让误差按 $h^4$ 下降，直到舍入地板。",
                },
                {
                    "name": "nodes",
                    "label": "节点分布",
                    "type": "enum",
                    "options": ["uniform", "chebyshev"],
                    "default": "uniform",
                    "critical": [],
                    "criticalNote": "",
                    "help": "节点分布：等距节点在高 $n$ 时会让多项式在两端剧烈振荡；Chebyshev 节点把节点向两端聚集，显著压低 Lebesgue 常数和最大误差。",
                },
                {
                    "name": "bc",
                    "label": "样条边界条件",
                    "type": "enum",
                    "options": ["natural", "not_a_knot"],
                    "default": "natural",
                    "critical": [],
                    "criticalNote": "",
                    "help": "三次样条的边界条件：natural 端部二阶导为零，not-a-knot 让端部两段共用一段三次多项式。不同选择对内部影响小，对端部误差有可见差异。",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "函数、样条与多项式（固定坐标，溢出显式报告）", "ylabel": "$y$"},
                {"id": "error", "autoscaled": True, "title": "两条误差曲线与地板", "ylabel": "|误差|", "scale": "log"},
                {"id": "growth", "autoscaled": True, "title": "误差随 $n$ 的增长", "ylabel": "$\\max|e|$", "scale": "log"},
            ],
            "pitfalls": (
                "第一，多项式在**节点上**是精确的（残差贴 $u$），误差全部长在节点之间——"
                "只看节点值会得出「多项式更准」的反结论；副轴的 residual 就是这个陷阱的量化。"
                "第二，主轴固定坐标 + 溢出徽标：$n$ 大时多项式会冲出画面几十倍，"
                "自动缩放会把爆炸画成「形状差不多」，那才是真的骗人。"
                "第三，误差轴在细端会撞上舍入地板（约 $10^{-16}$ 量级），"
                "地板以下的「下降」是假的——本章显式画出地板参考线。"
            ),
            "theory": [
                "**多项式插值误差有明确公式。** 对 $n+1$ 个节点，$f(x)-p_n(x)=\\frac{f^{(n+1)}(\\xi)}{(n+1)!}\\prod_i(x-x_i)$。误差由两部分相乘：函数的高阶导数和节点多项式 $\\omega(x)=\\prod(x-x_i)$。",
                "**Runge 现象来自节点多项式。** 等距节点时，$\\omega(x)$ 在区间两端增长极快；$n$ 增大，端部误差反而增大。Chebyshev 节点让 $\\omega$ 的最大值最小化，所以同样的 $n$ 下误差小得多。",
                "**三次样条的误差是局部的。** 在每段上，$|f-s|\\le\\frac{5}{384}h^4\\|f^{(4)}\\|_\\infty$（自然样条端部常数略有不同）。$h$ 减小一半，误差约降 $16$ 倍；它不会像高次多项式那样在端部失控。",
                "**光滑性决定阶数。** 对 $C^4$ 函数，三次样条是四阶；若函数只有 $C^2$ 或带尖点，四阶界不再成立，实际收敛阶会下降。比较方法时必须同时看被插函数的光滑性。",
                "**边界条件与节点分布。** natural/clamped/not-a-knot 影响端部误差；等距/Chebyshev 影响节点多项式。读图时把“方法”和“节点分布”分开看：样条对节点分布不敏感，多项式对两者都敏感。",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 计算入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "control_points":
        return _control_points(params, t)
    if viz_id == "smoothness_c0c1c2":
        return _smoothness(params, t)
    if viz_id == "bspline_local":
        return _bspline_local(params, t)
    if viz_id == "spline_vs_poly":
        return _spline_vs_poly(params, t)
    raise KeyError(f"章节 2.2 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _floor_abs(values: np.ndarray) -> np.ndarray:
    """把绝对值抬到地板之上，好让 log 轴画得出"精确 0"。"""
    return np.maximum(np.abs(values), FLOOR)


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}g}"


def _finite(value: float) -> float | None:
    """``inf``/``nan`` 不能进 meta（JSON 没有这两个字面量）：换成 ``null``。

    什么时候会出现非有限值：``tau=1`` 时能量恰好为 0，比值就是 :math:`1/0`。
    这不是"算错了"，而是那个比值在该点**没有定义**——如实写 null，并在 strings 里说明。
    """
    v = float(value)
    return v if bool(np.isfinite(v)) else None


# --------------------------------------------------------------------------- #
# viz 1：控制点、张力与曲率梳
# --------------------------------------------------------------------------- #
def _control_points(params: Mapping[str, Any], t: float) -> Frame:
    bc = str(params["bc"])
    tau = float(params["tension"])
    comb_mode = str(params["comb"])
    n_pts = 6
    nodes = np.arange(n_pts, dtype=float)
    data = np.array([float(params[f"y{i}"]) for i in range(n_pts)], dtype=float)

    base = cubic_spline(nodes, data, bc)
    # 张力：弯矩整体乘 (1-tau)。曲线仍然过所有数据点，但 S' 开始跳变。
    moments = (1.0 - tau) * np.asarray(base.moments, dtype=float)
    curve = cubic_from_moments(nodes, data, moments, label=f"spline[{bc},tau={tau:g}]", bc=bc)

    n_grid = 301
    xg = np.linspace(nodes[0], nodes[-1], n_grid)
    yg = curve.evaluate(xg, 0)
    kg = curvature_of(curve, xg)
    base_pts, tip_pts = curvature_comb(curve, xg, scale=comb_mode, length=0.45)
    _, tip_local = curvature_comb(curve, xg, scale="local", length=0.45)

    # 节点两侧的跳变：用 PiecewiseCubic.jumps（内点处"右侧极限 - 左侧极限"）
    jump_d1 = _floor_abs(curve.jumps("d1"))
    jump_d2 = _floor_abs(curve.jumps("d2"))
    nodal_res = _floor_abs(curve.evaluate(nodes, 0) - data)

    # 探针：时间轴就是它在 x 方向上的位置
    probe_x = float(np.clip(t, nodes[0], nodes[-1]))
    probe = np.array([probe_x, float(curve.evaluate(np.array([probe_x]), 0)[0])])
    tangent = np.array([1.0, float(curve.evaluate(np.array([probe_x]), 1)[0])])
    tangent = tangent / np.sqrt(float(np.sum(tangent * tangent)))

    sysm = spline_system(nodes, data, bc)
    energy = curve.energy()
    energy_lin = polyline(nodes, data).energy()
    energy_pchip = pchip(nodes, data).energy()

    fr = Frame(chapter="2.2", viz="control_points", kind="frame")
    fr.add("x_grid", xg, unit="1", label="采样横坐标")
    fr.add("curve", yg, unit="1", label="样条曲线 S(x)")
    fr.add("slope", curve.evaluate(xg, 1), unit="1", label="S'(x)")
    fr.add("curvature", kg, unit="1", label="曲率 kappa(x)")
    fr.add("node_x", nodes, unit="1", label="数据点 x")
    fr.add("node_y", data, unit="1", label="数据点 y")
    fr.add("comb_base", base_pts, unit="1", label="曲率梳根部 (n,2)")
    fr.add("comb_tip", tip_pts, unit="1", label="曲率梳齿尖（当前尺度）(n,2)")
    fr.add("comb_tip_local", tip_local, unit="1", label="曲率梳齿尖（local 归一化，仅供对照）(n,2)")
    fr.add("jump_nodes", nodes[1:-1], unit="1", label="内点 x")
    fr.add("jump_d1", jump_d1, unit="1", label="|S'(x+) - S'(x-)|")
    fr.add("jump_d2", jump_d2, unit="1", label="|S''(x+) - S''(x-)|")
    fr.add("residual", jump_d1, unit="1", label="C1 残差（节点处一阶导跳变，0 以 u 显示）")
    fr.add("error", nodal_res, unit="1", label="节点处的插值残差（0 以 u 显示）")
    fr.add("probe", probe, unit="1", label="探针位置 (x,y)")
    fr.add("probe_tangent", tangent, unit="1", label="探针处单位切向")
    fr.add("system_diag", sysm.diag, unit="1", label="三对角系统对角元")
    fr.add("system_rhs", sysm.rhs, unit="1", label="三对角系统右端项")
    fr.set_meta(
        n_nodes=n_pts,
        n_intervals=n_pts - 1,
        bc_index=_BC_INDEX.get(bc, -1),
        tension=tau,
        comb_scale_index=0 if comb_mode == "global" else 1,
        energy=energy,
        energy_polyline=energy_lin,
        energy_pchip=energy_pchip,
        energy_ratio_pchip=_finite(energy_pchip / energy) if energy > 0.0 else None,
        energy_over_polyline=_finite(energy / energy_lin) if energy_lin > 0.0 else None,
        curvature_max=_finite(float(np.max(np.abs(kg)))),
        curvature_at_probe=_finite(float(curvature_of(curve, np.array([probe_x]))[0])),
        jump_d1_max=float(np.max(jump_d1)) if jump_d1.size else 0.0,
        jump_d2_max=float(np.max(jump_d2)) if jump_d2.size else 0.0,
        residual=float(np.max(jump_d1)) if jump_d1.size else 0.0,
        nodal_error_max=float(np.max(nodal_res)),
        system_size=int(sysm.diag.size),
        t_actual=probe_x,
        overflow=False,
    )
    fr.set_strings(
        bc_name=_BC_NAME.get(bc, bc),
        comb_name="统一尺度（长度可比）" if comb_mode == "global" else "逐点归一化（只剩方向）",
        note=_control_note(bc, tau, energy, jump_d1, kg),
    )
    return fr


def _control_note(bc: str, tau: float, energy: float, jump_d1: np.ndarray, kg: np.ndarray) -> str:
    jmax = float(np.max(jump_d1)) if jump_d1.size else 0.0
    if tau >= 1.0:
        return "tau=1：弯矩为 0，曲线就是折线——C1 已经断了，曲率能量恒为 0（比值因此无定义）"
    if jmax > 1e-8:
        return f"tau={tau:g}：节点处 |dS'| 已达 {_fmt(jmax)}，曲线只是 C0（折角可见）"
    return f"{_BC_NAME.get(bc, bc)}：|dS'| ~ {_fmt(jmax)}（机器精度），曲率峰值 {_fmt(float(np.max(np.abs(kg))))}，能量 {_fmt(energy)}"


# --------------------------------------------------------------------------- #
# viz 2：C0 / C1 / C2 与二阶导的阶梯跳变
# --------------------------------------------------------------------------- #
def _dataset_nodes(name: str, n: int, kind: str) -> np.ndarray:
    if kind == "chebyshev":
        return chebyshev_nodes(n, -1.0, 1.0)
    return uniform_nodes(n, -1.0, 1.0)


def _dataset_values(name: str, nodes: np.ndarray) -> np.ndarray:
    if name == "steps":
        # 台阶数据：在 [-1,1] 上取 k 个平台，平台之间是真正的跃变（最考验 C1/C2 的形状）
        n = nodes.size
        k = max(2, (n - 1) // 2)
        idx = np.minimum((np.arange(n) * k) // max(n - 1, 1), k - 1)
        levels = np.linspace(-1.0, 1.0, k)
        return levels[idx].astype(float)
    return sample_function(name, nodes)


def _smoothness(params: Mapping[str, Any], t: float) -> Frame:
    dataset = str(params["dataset"])
    node_kind = str(params["nodes"])
    n = int(params["n"])
    eps_rel = float(params["eps"])
    bc = str(params["bc"])

    nodes = _dataset_nodes(dataset, n, node_kind)
    data = _dataset_values(dataset, nodes)
    lin = polyline(nodes, data)
    pc = pchip(nodes, data)
    spl = cubic_spline(nodes, data, bc)

    # 节点两侧采样：每个内点取 x_i-eps 与 x_i+eps，画出来就是竖直跳变
    span = float(nodes[-1] - nodes[0])
    eps = eps_rel * span / max(n - 1, 1)
    x_step = np.concatenate([[nodes[0]], np.ravel(np.stack([nodes[1:-1] - eps, nodes[1:-1] + eps], axis=1)), [nodes[-1]]])
    eps_used = float(eps)

    # "默认等距采样"：同一批函数、同一段区间，只是采样点没有对准节点
    x_naive = np.linspace(nodes[0], nodes[-1], 200)

    jump_nodes = nodes[1:-1]
    j1_lin = _floor_abs(lin.jumps("d1"))
    j2_pc = _floor_abs(pc.jumps("d2"))
    j2_spl = _floor_abs(spl.jumps("d2"))

    e_lin = _floor_abs(lin.evaluate(nodes) - data)
    e_pc = _floor_abs(pc.evaluate(nodes) - data)
    e_spl = _floor_abs(spl.evaluate(nodes) - data)

    fr = Frame(chapter="2.2", viz="smoothness_c0c1c2", kind="frame")
    fr.add("x_step", x_step, unit="1", label="节点两侧加密采样点")
    fr.add("x_naive", x_naive, unit="1", label="默认等距采样点")
    fr.add("y_lin", lin.evaluate(x_step), unit="1", label="C0 折线")
    fr.add("y_pchip", pc.evaluate(x_step), unit="1", label="C1 PCHIP")
    fr.add("y_spline", spl.evaluate(x_step), unit="1", label="C2 三次样条")
    fr.add("d2_lin", lin.evaluate(x_step, 2), unit="1", label="折线的 S''（两侧都是 0）")
    fr.add("d2_pchip_step", pc.evaluate(x_step, 2), unit="1", label="PCHIP 的 S''（节点两侧分采）")
    fr.add("d2_spline_step", spl.evaluate(x_step, 2), unit="1", label="样条的 S''（节点两侧分采）")
    fr.add("d2_pchip_naive", pc.evaluate(x_naive, 2), unit="1", label="PCHIP 的 S''（默认等距采样：跳变被连成斜坡）")
    fr.add("d2_spline_naive", spl.evaluate(x_naive, 2), unit="1", label="样条的 S''（默认等距采样）")
    fr.add("node_x", nodes, unit="1", label="节点")
    fr.add("jump_nodes", jump_nodes, unit="1", label="内点 x")
    fr.add("jump_d1_lin", j1_lin, unit="1", label="折线 |dS'|（C0 的破绽在这里）")
    fr.add("jump_d2_pchip", j2_pc, unit="1", label="PCHIP |dS''|（C1 的破绽在这里）")
    fr.add("jump_d2_spline", j2_spl, unit="1", label="样条 |dS''|（应当 ~u）")
    fr.add("residual", j2_pc, unit="1", label="C2 残差量：PCHIP 的二阶导跳变（0 以 u 显示）")
    fr.add("error", e_spl, unit="1", label="样条在节点处的插值残差（0 以 u 显示）")
    fr.set_meta(
        n=n,
        dataset_index=_DATASET_INDEX.get(dataset, -1),
        nodes_index=_NODES_INDEX.get(node_kind, -1),
        bc_index=_BC_INDEX.get(bc, -1),
        eps=eps_used,
        eps_rel=eps_rel,
        n_step_samples=int(x_step.size),
        jump_pchip_max=float(np.max(j2_pc)) if j2_pc.size else 0.0,
        jump_spline_max=float(np.max(j2_spl)) if j2_spl.size else 0.0,
        jump_lin_slope_max=float(np.max(j1_lin)) if j1_lin.size else 0.0,
        residual=float(np.max(j2_pc)) if j2_pc.size else 0.0,
        error_spline_max=float(np.max(e_spl)),
        error_pchip_max=float(np.max(e_pc)),
        error_lin_max=float(np.max(e_lin)),
        energy_lin=lin.energy(),
        energy_pchip=pc.energy(),
        energy_spline=spl.energy(),
        naive_samples=int(x_naive.size),
        t_actual=0.0,
        overflow=False,
    )
    fr.set_strings(
        dataset_name=f"{_DATASET_NAME.get(dataset, dataset)} / {_NODES_NAME.get(node_kind, node_kind)}",
        bc_name=_BC_NAME.get(bc, bc),
        note=_smooth_note(dataset, node_kind, eps_used, j2_pc, j2_spl, j1_lin),
    )
    return fr


def _smooth_note(
    dataset: str, node_kind: str, eps: float, j2_pc: np.ndarray, j2_spl: np.ndarray, j1_lin: np.ndarray
) -> str:
    return (
        f"{dataset}/{node_kind}：eps={_fmt(eps)}，PCHIP 的 |dS''| 最大 "
        f"{_fmt(float(np.max(j2_pc)))}，样条 {_fmt(float(np.max(j2_spl)))}，"
        f"折线的 |dS'| 最大 {_fmt(float(np.max(j1_lin)))}"
    )


# --------------------------------------------------------------------------- #
# viz 3：B 样条局部支撑与节点插入
# --------------------------------------------------------------------------- #
def _bspline_ctrl_profile(n_ctrl: int) -> np.ndarray:
    """确定性的控制点纵坐标（不用随机数：帧必须逐位可复现）。"""
    s = np.linspace(0.0, 1.0, n_ctrl)
    return np.sin(2.0 * np.pi * s) + 0.35 * np.cos(4.0 * np.pi * s)


def _bspline_local(params: Mapping[str, Any], t: float) -> Frame:
    n_ctrl = int(params["n_ctrl"])
    degree = int(params["degree"])
    move_index = min(int(params["move_index"]), n_ctrl - 1)
    move_delta = float(params["move_delta"])
    insert_at = float(params["insert_at"])
    insert_at = min(max(insert_at, 0.02), 0.98)

    knots0 = uniform_clamped_knots(n_ctrl, degree, t0=0.0, t1=1.0)
    ctrl_x = np.linspace(0.0, 1.0, n_ctrl)
    ctrl_y = _bspline_ctrl_profile(n_ctrl)
    ctrl_y_moved = ctrl_y.copy()
    ctrl_y_moved[move_index] += move_delta

    n_ins = int(np.clip(np.floor(float(t) + 0.5), 0.0, 4.0))
    seq = insertion_sequence(knots0, ctrl_y, degree, insert_at, n_ins)
    knots, ctrl = seq[-1]

    n_grid = 361
    xg = np.linspace(0.0, 1.0, n_grid)
    curve_orig = de_boor(knots, ctrl, degree, xg)
    # 移动过的控制点走**同样的**插入序列（两条曲线必须用同一套节点向量才可比较）
    seq_moved = insertion_sequence(knots0, ctrl_y_moved, degree, insert_at, n_ins)
    ctrl_moved = seq_moved[-1][1]
    curve_moved = de_boor(knots, ctrl_moved, degree, xg)
    curve_base = de_boor(knots0, ctrl_y, degree, xg)

    changed = np.abs(curve_moved - curve_orig)
    invariance = _floor_abs(curve_orig - curve_base)

    basis = bspline_basis_matrix(knots, degree, xg)
    n_basis = basis.shape[0]
    support_res = np.zeros(n_basis, dtype=float)
    sup_lo = np.zeros(n_basis, dtype=float)
    sup_hi = np.zeros(n_basis, dtype=float)
    for i in range(n_basis):
        lo, hi = basis_support(knots, i, degree)
        sup_lo[i] = lo
        sup_hi[i] = hi
        outside = (xg < lo - 1e-15) | (xg > hi + 1e-15)
        support_res[i] = float(np.max(np.abs(basis[i][outside]))) if bool(np.any(outside)) else 0.0
    support_res = _floor_abs(support_res)

    span_lo, span_hi = basis_support(knots, move_index, degree)
    # 理论上是 k+1 个结点区间；夹紧端点上会有退化区间（长度 0），所以如实数一遍**非退化**区间
    spans_actual = 0
    for j in range(knots.size - 1):
        if knots[j] < knots[j + 1] and knots[j] >= span_lo - 1e-15 and knots[j + 1] <= span_hi + 1e-15:
            spans_actual += 1
    changed_measure = float(np.sum(changed > 1e-12)) / float(n_grid)
    hull_gap = convex_hull_gap(knots, ctrl_moved, degree, xg)

    insert_marker = np.array([insert_at, float(de_boor(knots, ctrl, degree, np.array([insert_at]))[0])])

    fr = Frame(chapter="2.2", viz="bspline_local", kind="frame")
    fr.add("ctrl_x", ctrl_x, unit="1", label="控制点 x")
    fr.add("ctrl_y", ctrl_y, unit="1", label="控制点 y")
    fr.add("ctrl_y_moved", ctrl_y_moved, unit="1", label="移动后的控制点 y")
    fr.add("knots", knots, unit="1", label="当前节点向量")
    fr.add("knots_base", knots0, unit="1", label="初始节点向量")
    fr.add("x_grid", xg, unit="1", label="采样参数")
    fr.add("curve_orig", curve_orig, unit="1", label="当前曲线（未移动控制点）")
    fr.add("curve_moved", curve_moved, unit="1", label="移动控制点后的曲线")
    fr.add("curve_base", curve_base, unit="1", label="未插入节点时的曲线")
    fr.add("basis", basis, unit="1", label="基函数矩阵 (n_basis, n_grid)")
    fr.add("basis_index", np.arange(n_basis, dtype=float), unit="1", label="基函数下标")
    fr.add("support_left", sup_lo, unit="1", label="支撑左端")
    fr.add("support_right", sup_hi, unit="1", label="支撑右端")
    fr.add("changed", changed, unit="1", label="|改控制点前后的曲线差|（区间外应为 0）")
    fr.add("changed_mask", (changed > 1e-12).astype(float), unit="1", label="受影响的采样点标记")
    fr.add("error", invariance, unit="1", label="节点插入前后逐点差（0 以 u 显示）")
    fr.add("residual", support_res, unit="1", label="支撑外的 max|B_i|（恒为 0，以 u 显示）")
    fr.add("affected_span", np.array([span_lo, span_hi]), unit="1", label="受影响参数区间")
    fr.add("insert_marker", insert_marker, unit="1", label="插入点位置 (u, C(u))")
    fr.set_meta(
        n_ctrl=n_ctrl,
        degree=degree,
        n_basis=n_basis,
        n_knots=int(knots.size),
        n_knots_base=int(knots0.size),
        insert_count=n_ins,
        insert_at=insert_at,
        move_index=move_index,
        move_delta=move_delta,
        span_left=span_lo,
        span_right=span_hi,
        affected_spans=spans_actual,
        affected_spans_max=degree + 1,
        changed_measure=changed_measure,
        changed_max=float(np.max(changed)),
        invariance_error_max=float(np.max(invariance)),
        support_violation_max=float(np.max(support_res)),
        partition_unity_error=_finite(float(np.max(np.abs(basis.sum(axis=0) - 1.0)))),
        hull_gap=_finite(hull_gap),
        error=float(np.max(invariance)),
        residual=float(np.max(support_res)),
        t_actual=float(n_ins),
        overflow=False,
    )
    fr.set_strings(
        degree_name=f"{degree} 次（k+1 = {degree + 1} 段受影响）",
        note=(
            f"插入 {n_ins} 次：|dC| ≤ {_fmt(float(np.max(invariance)))}；"
            f"改第 {move_index} 个控制点影响 [{_fmt(span_lo)}, {_fmt(span_hi)}]"
            f"（{spans_actual} 个非退化区间，上界 k+1={degree + 1}；占参数域 {_fmt(changed_measure * 100.0)}%）"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：样条 vs 高阶多项式
# --------------------------------------------------------------------------- #
def _spline_vs_poly(params: Mapping[str, Any], t: float) -> Frame:
    n = int(params["n"])
    name = str(params["function"])
    node_kind = str(params["nodes"])
    bc = str(params["bc"])

    nodes = uniform_nodes(n, -1.0, 1.0) if node_kind == "uniform" else chebyshev_nodes(n, -1.0, 1.0)
    data = sample_function(name, nodes)
    xg = np.linspace(-1.0, 1.0, 401)
    exact = sample_function(name, xg)
    poly = lagrange_value(nodes, data, xg)
    spl = cubic_spline(nodes, data, bc).evaluate(xg)

    scale = max(float(np.max(np.abs(exact))), 1e-300)
    floor = FLOOR * scale
    err_poly = np.maximum(np.abs(poly - exact), floor)
    err_spl = np.maximum(np.abs(spl - exact), floor)
    floor_line = np.full(xg.size, floor)
    nodal_res = _floor_abs(lagrange_value(nodes, data, nodes) - data)

    # 误差随 n 增长的历史（n 从 4 到当前值）：两条曲线用**同一套**增长逻辑算出来
    ns = np.arange(4, n + 1, dtype=float)
    hist_poly = np.zeros(ns.size, dtype=float)
    hist_spl = np.zeros(ns.size, dtype=float)
    for i, nn in enumerate(ns):
        nd = uniform_nodes(int(nn), -1.0, 1.0) if node_kind == "uniform" else chebyshev_nodes(int(nn), -1.0, 1.0)
        dd = sample_function(name, nd)
        xx = np.linspace(-1.0, 1.0, 201)
        ex = sample_function(name, xx)
        hist_poly[i] = max(float(np.max(np.abs(lagrange_value(nd, dd, xx) - ex))), floor)
        hist_spl[i] = max(float(np.max(np.abs(cubic_spline(nd, dd, bc).evaluate(xx) - ex))), floor)

    max_poly = float(np.max(np.abs(poly)))
    overflow = bool(max_poly > 3.0 or float(np.min(poly)) < -3.0)
    leb = lebesgue_constant(nodes, n_grid=401)
    energy_spl = cubic_spline(nodes, data, bc).energy()
    energy_poly = interpolating_polynomial_energy(nodes, data, n_quad=max(n + 2, 16))

    fr = Frame(chapter="2.2", viz="spline_vs_poly", kind="frame")
    fr.add("x_grid", xg, unit="1", label="采样横坐标")
    fr.add("y_exact", exact, unit="1", label="被插函数 f(x)")
    fr.add("y_spline", spl, unit="1", label="样条插值")
    fr.add("y_poly", poly, unit="1", label="Lagrange 插值多项式")
    fr.add("node_x", nodes, unit="1", label="节点")
    fr.add("node_y", data, unit="1", label="节点数据")
    fr.add("error_spline", err_spl, unit="1", label="|样条误差|（地板以下按地板显示）")
    fr.add("error_poly", err_poly, unit="1", label="|多项式误差|（地板以下按地板显示）")
    fr.add("error_floor_line", floor_line, unit="1", label="舍入地板参考线")
    fr.add("error", np.array([float(np.max(err_spl)), float(np.max(err_poly))]), unit="1", label="两条最大误差")
    fr.add("residual", nodal_res, unit="1", label="多项式在节点处的残差（0 以 u 显示）")
    fr.add("n_history", ns, unit="1", label="节点数 $n$")
    fr.add("error_poly_history", hist_poly, unit="1", label="多项式最大误差随 n")
    fr.add("error_spline_history", hist_spl, unit="1", label="样条最大误差随 n")
    fr.set_meta(
        n=n,
        function_index=_FUNCTION_INDEX.get(name, -1),
        nodes_index=_NODES_INDEX.get(node_kind, -1),
        bc_index=_BC_INDEX.get(bc, -1),
        error_floor=floor,
        error_spline=float(np.max(err_spl)),
        error_poly=float(np.max(err_poly)),
        error_ratio=_finite(float(np.max(err_poly) / np.max(err_spl))),
        nodal_residual_max=float(np.max(nodal_res)),
        lebesgue_constant=leb,
        poly_max_abs=_finite(max_poly),
        poly_min=_finite(float(np.min(poly))),
        energy_spline=energy_spl,
        energy_poly=energy_poly,
        energy_ratio=_finite(energy_poly / energy_spl) if energy_spl > 0.0 else None,
        n_history_points=int(ns.size),
        error=float(np.max(err_spl)),
        residual=float(np.max(nodal_res)),
        overflow=overflow,
        t_actual=0.0,
    )
    fr.set_strings(
        function_name=f"{_DATASET_NAME.get(name, name)} / {_NODES_NAME.get(node_kind, node_kind)}",
        bc_name=_BC_NAME.get(bc, bc),
        note=(
            f"n={n}：样条误差 {_fmt(float(np.max(err_spl)))}，多项式误差 {_fmt(float(np.max(err_poly)))}"
            f"（差 {_fmt(float(np.max(err_poly) / np.max(err_spl)))} 倍）；Lebesgue 常数 {_fmt(leb)}；"
            f"多项式峰值 |p|max={_fmt(max_poly)}"
            + ("（已冲出画面，见溢出徽标）" if overflow else "")
        ),
    )
    return fr
