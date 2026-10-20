r"""2.4 章：数值积分（求积公式 / 误差阶 / 外推 / 奇异性）。

四个可视化各守一条**可测量的断言**，而不是让图"看起来对"：

1. ``area_convergence`` —— 面积逼近与误差阶：左轴画被积函数与当前面板的分段逼近
   （面板数与时间轴联动），右轴画 |Q-I| 对面板数的双对数曲线，
   同时叠加**实测拟合斜率**与**理论阶参考线**（防误导规则 2）。
   ``method`` 里的 ``adaptive`` 是同一张图的另一种看法：递归细分把面板放到"难"的地方，
   误差面板改成"自适应估计误差 vs 真误差"——两者的差就是本章第三条陷阱。
2. ``gauss_vs_equispaced`` —— Gauss 节点 vs 等距节点：节点位置、权重、以及
   **双向的多项式精确度证据**（k ≤ 2n−1 在舍入地板、k = 2n 当场失效）。
   第三条曲线是 :math:`\Gamma=\sum|w|/\sum w` 随点数的变化：等距规则在 m=9 处
   首次出现负权重并开始放大输入误差，Gauss 恒为 1。
3. ``romberg_table`` —— Romberg 三角：每一列的实测收敛阶对理论阶 :math:`2j+2`，
   以及外推的加速比与"哪一层开始只剩噪声"。
4. ``singular_endpoint`` —— 端点奇异性：tanh--sinh 的（超）指数收敛 vs
   换元之后等距规则的代数收敛，并给出**地板的理论值**：未被采样的尾巴
   :math:`\int_0^{x_{\min}}f`（:math:`x_{\min}\approx u/2`）。
   实测 :math:`\int_0^1x^{-1/2}` 类被积函数的地板 ≈ :math:`\sqrt{2u}=1.49\times10^{-8}`，
   而不是 :math:`u`；把节点数从 129 加到 513 一动不动。

约定：``meta`` 只放**有限标量**（没有有限值写 ``None`` 并在 ``strings`` 里说明原因；
NLF 头用 ``allow_nan=False`` 编码，写 NaN 会让整帧失败）。所有 log 轴的误差曲线
都按舍入地板截断，并把截断点数写进 meta。``requiredAux`` 里的名字必须真的以
``error`` 开头（全仓帧契约测试按前缀匹配）。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from numcore.quad.adaptive import adaptive_simpson
from numcore.quad.condition import weight_diagnostics
from numcore.quad.gauss import (
    exact_monomial,
    gauss_legendre_rule,
    moment_errors,
    moment_failure_factor,
)
from numcore.quad.integrals import Integrand, integrand, substituted_error_curve
from numcore.quad.newton_cotes import (
    composite_eval,
    composite_rule,
    effective_sub_counts,
    error_curve,
    fitted_constant,
    newton_cotes_weights,
    rule_law,
)
from numcore.quad.rates import fit_loglog, fit_semilog, local_order, theory_line_power
from numcore.quad.romberg import (
    RombergResult,
    column_orders,
    extrapolation_coefficients,
    romberg_table,
)
from numcore.quad.singular import tanh_sinh_rule
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 单位舍入 :math:`u=2^{-53}`：所有误差曲线的截断地板。
U = 2.0**-53

#: 画曲线用的稠密采样点数。
_GRID = 401

#: 面板内插值的采样密度（每个面板画多少段）。
_PER_PANEL = 16

#: tanh--sinh 的网格步长选项（字符串是 manifest 的枚举值，浮点才是数学）。
_H_CHOICES: dict[str, float] = {"1/8": 2.0**-3, "1/16": 2.0**-4, "1/32": 2.0**-5}

#: 自适应模式下的相对容差（乘 |I|）。
_ADAPTIVE_RTOL = 1e-6


MANIFEST: dict[str, Any] = {
    "chapter": "2.4",
    "title": "数值积分",
    "part": "II",
    "summary": (
        "把 $\\int f$ 算出来，并且知道它有多准：Newton–Cotes 族的误差阶、Romberg 外推、"
        "Gauss–Legendre 的 $2n-1$ 次精确度、端点奇异与振荡的处理，"
        "以及三件最容易骗人的事——权重会放大输入误差、非光滑函数上量到的是光滑度的阶、"
        "误差估计器估的是它自己的误差。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "area_convergence",
            "title": "面积逼近：阶、面板与自适应细分",
            "description": (
                "左：被积函数与当前 $k$ 个面板上的分段逼近（时间轴 = 已揭示的面板数），"
                "竖线是面板边界、点是求值节点。右：$|Q-I|$ 对面板数的双对数曲线，"
                "同时画实测拟合斜率与理论阶参考线（规则 2）。"
                "把 method 切到 adaptive：左图变成递归细分后的区间划分，"
                "右图变成「自适应估计误差」与「真误差」两条曲线——在窄峰函数上它们能差 $800$ 倍。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "method",
                    "label": "求积方式",
                    "type": "enum",
                    "options": ["trapezoid", "simpson", "simpson38", "adaptive"],
                    "default": "trapezoid",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "func",
                    "label": "被积函数 f",
                    "type": "enum",
                    "options": ["exp", "runge", "sqrt", "cusp", "spike"],
                    "default": "exp",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n",
                    "label": "面板数上限 n（adaptive 模式下是区间预算）",
                    "type": "int",
                    "min": 4,
                    "max": 512,
                    "default": 64,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 64.0, "dt": 1.0, "label": "已揭示的面板数 k"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "f 与分段逼近", "ylabel": "f(x)"},
                {
                    "id": "err",
                    "autoscaled": True,
                    "title": "|Q−I| vs 面板数（双对数）",
                    "ylabel": "误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "① 双对数图上量到的斜率不一定是求积公式的阶：$f=\\sqrt{x}$（端点导数无界）时，""梯形与 Simpson 都会塌到 $\\min(p,\\alpha+1)=1.5$ 阶——量到的是被积函数光滑度的阶。""② $f=|x|$ 折点落在节点上时梯形法逐段精确，误差直接掉到 $10^{-16}$，""看起来像「阶无穷大」，其实是折点与节点重合的巧合（偶数面板时必然发生）。""③ adaptive 模式下右图的「估计误差」是嵌套法则的差，不是真误差：""窄峰函数上初始三点全落在峰外，估计值 $4.3\\times10^{-5}$ 而真误差 $3.5\\times10^{-2}$。"
            ),
            "theory": [
                "复合误差主项 $E=Q-I=C(b-a)h^pf^{(p)}(\\xi)$：$C_{\\mathrm{trap}}=1/12$、$C_{\\mathrm{simp}}=1/180$、$C_{3/8}=1/80$（笔记定理 2；凸函数被高估，所以是正号）。",
                "非光滑端点 $f\\sim x^\\alpha$ $\\Rightarrow$ 实测阶 $=\\min(p,\\alpha+1)$（笔记定理 7）。",
                "自适应 Simpson 的局部估计误差 $\\hat{h}E=(S_{\\mathrm{fine}}-S_{\\mathrm{coarse}})/15$，仅当 $h^4$ 项主导时才等于真误差（笔记定理 9）。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "gauss_vs_equispaced",
            "title": "Gauss 节点 vs 等距节点：精确度的两个方向",
            "description": (
                "左上：两种公式的节点位置与权重（Gauss 节点在端点加密，权重全正）。"
                "左下：多项式精确度测试——$k\\le2n-1$ 的矩误差在舍入水平，$k=2n$ 当场跳起来，"
                "竖线标出 $2n-1$ 这条边界。右：两种规则的条件数 $\\Gamma=\\sum|w|/\\sum w$ 随点数的变化，"
                "等距规则在 $m=9$ 处首次出现负权重、$\\Gamma$ 随即离开 $1$。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "n",
                    "label": "节点数 n",
                    "type": "int",
                    "min": 2,
                    "max": 20,
                    "default": 8,
                    "critical": [9.0],
                    "criticalNote": (
                        "这条线只对**等距**规则成立：闭 Newton–Cotes 的权重在 $m=9$ 个点处首次变负，"
                        "实测 $\\Gamma=\\sum|w|/\\sum w$ 从 $1.000$ 跳到 $1.451$，$m=13$ 时 $7.53$、$m=19$ 时 $175.46$。"
                        "注意 $m=10$ 是反例（权重全正、$\\Gamma=1.000$），所以判据是 $\\Gamma$ 不是 $m$；"
                        "Gauss–Legendre 的权重恒为正、$\\Gamma\\equiv1$，不受这条线影响。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "func",
                    "label": "收敛对比用的被积函数",
                    "type": "enum",
                    "options": ["exp", "runge", "sqrt"],
                    "default": "exp",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 42.0, "dt": 1.0, "label": "测试到 k 次多项式"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "nodes", "autoscaled": False, "title": "节点与权重", "ylabel": "w_k"},
                {
                    "id": "exact",
                    "autoscaled": True,
                    "title": "矩误差 |Q(x^k)−∫x^k|",
                    "ylabel": "误差",
                    "scale": "log",
                },
                {
                    "id": "cond",
                    "autoscaled": True,
                    "title": "条件数 Γ=Σ|w|/Σw",
                    "ylabel": "Γ",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "① 「Gauss 很准」这句话本身没有上界：必须同时给出 $k=2n$ 处**当场失效**的证据，""否则「精确到 $2n-1$ 次」与「精确到 $100$ 次」在图上无法区分。""② 等距高阶公式的失败不在截断误差而在条件：$m=19$ 时 $\\Gamma=175.5$，""函数值里 $10^{-8}$ 的相对误差会被放大成积分里 $1.8\\times10^{-6}$ 的误差——""「函数值算得准」不等于「积分算得准」。""③ 权重和 $\\sum w=b-a$ 只说明常数被积对；$m=20$ 的 $\\sum w$ 实测偏差 $3.1\\times10^{-8}$，""而这已经是浮点权重本身的误差，不是公式的精度。"
            ),
            "theory": [
                "$n$ 点 Gauss–Legendre 的代数精度**恰好**是 $2n-1$（笔记定理 4）。",
                "等距闭 Newton–Cotes 的权重在 $m\\ge9$ 起出现负值，$\\sum|w|/\\sum w$ 随 $m$ 迅速增长（笔记定理 6）。",
                "求积的条件数 $\\Gamma=\\sum|w|/|\\sum w|$：输入相对误差放大 $\\Gamma$ 倍（笔记定理 6 推论）。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "romberg_table",
            "title": "Romberg 三角：每一列的阶与外推的代价",
            "description": (
                "左：各列的误差对节点数的双对数曲线（第 $0$ 列是梯形，第 $j$ 列消掉到 $h^{2j}$ 的项），"
                "点线是对角线（每一层能拿到的最好估计）。右：每一列的**实测**阶与理论阶 $2j+2$，"
                "以及外推的加速比。时间轴 = 已算到的层数。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "func",
                    "label": "被积函数 f",
                    "type": "enum",
                    "options": ["exp", "runge", "cusp"],
                    "default": "exp",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "levels",
                    "label": "最大层数 m",
                    "type": "int",
                    "min": 2,
                    "max": 10,
                    "default": 8,
                    "critical": [6.0],
                    "criticalNote": (
                        "第 $6$ 层是对角线触到舍入地板的那一层：光滑函数上实测 "
                        "$|T_{55}-I|\\approx2.2\\times10^{-16}$、$|T_{66}-I|\\approx2.2\\times10^{-16}$，再往上加层只会把输入噪声"
                        "放大到 $A_j\\approx1.969$ 倍而没有截断误差可消。$f=|x|$ 时这条线不适用——"
                        "折点在节点上，梯形法从第 $2$ 层起就精确了。"
                    ),
                    "beyond": "above",
                },
            ],
            "time": {"mode": "scrub", "tMax": 10.0, "dt": 1.0, "label": "已算到的层数 k"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "conv",
                    "autoscaled": False,
                    "title": "各列误差 vs 节点数（双对数）",
                    "ylabel": "|T_{k,j}−I|",
                    "scale": "log",
                },
                {
                    "id": "order",
                    "autoscaled": True,
                    "title": "每一列的实测阶 vs 列号",
                    "ylabel": "阶 p_j",
                },
            ],
            "pitfalls": (
                "① 只看最后一列会以为「Romberg 一直在加速」：列号越大，该列**能用的行**越少，"
                "第 $j$ 列只有 $m-j$ 个点，第 $j$ 列的实测阶是从这几个点拟合的，"
                "$m-j<3$ 时根本没有阶可言（帧里如实给 None 而不是 $0$）。"
                "② 不裁掉细端会得到假阶：第 $3$ 列不裁尾拟合 $6.96$、裁掉触地板的 $2$ 个点得 $7.97$"
                "（理论 $8$）——「误差阶」这句话必须带上窗口。"
                "③ 外推不是免费的：组合系数的绝对值之和 $A_j$ 从 $1$ 涨到 $1.9692603$（有界），"
                "舍入噪声被放大近 $2$ 倍；$A_j$ 收敛不等于可以无限外推。"
            ),
            "theory": [
                "Euler–Maclaurin：梯形误差 $T(h)=I+\\tau_1h^2+\\tau_2h^4+\\ldots$，$\\tau_i$ 与 $h$ 无关（笔记定理 5）。",
                "Romberg 第 $j$ 列的误差是 $O(h^{2j+2})$，第 $1$ 列就是 Simpson（笔记定理 5 推论）。",
                "外推系数由递推给出，$\\sum|c_i|=1$、$\\sum|c_i|\\to1.9692603$（笔记定理 5 引理）。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "singular_endpoint",
            "title": "端点奇异：双指数变换 vs 换元",
            "description": (
                "左：误差对节点数的半对数曲线——tanh–sinh 直接作用于奇异被积函数，"
                "对照曲线是「先换元 $x=t^2$ 再用等距规则」（代数收敛）。"
                "横线是地板的理论值：未被采样尾巴 $\\int_0^{x_{\\min}}f$ 的大小。"
                "右：最靠近奇异端点的节点 $x_{\\min}$ 与它带来的尾巴估计，"
                "以及饱和节点数（$|x|$ 在 double 里舍入到 $1$、权重下溢到 $0$ 的节点）。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "func",
                    "label": "带端点奇异的被积函数",
                    "type": "enum",
                    "options": ["inv_sqrt", "log", "sqrt"],
                    "default": "inv_sqrt",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "h",
                    "label": "tanh–sinh 网格步长 h",
                    "type": "enum",
                    "options": ["1/8", "1/16", "1/32"],
                    "default": "1/16",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n",
                    "label": "节点数上限 n（取奇数）",
                    "type": "int",
                    "min": 9,
                    "max": 513,
                    "default": 129,
                    "critical": [103.0],
                    "criticalNote": (
                        "饱和起点（$h=1/16$ 时实测）：第 $103$ 个节点起，两端的节点在 double 里"
                        "已经舍入到 $\\pm1$、权重下溢到 $0$，再加节点不会提高精度。"
                        "$h$ 减半会让它按 $t^*/h$ 翻倍：$h=1/32$ 时是 $205$（$t^*=3.17264\\ldots$，"
                        "即满足 $1-\\tanh(z)<u$ 的临界横坐标）。"
                    ),
                    "beyond": "above",
                },
            ],
            "time": {"mode": "scrub", "tMax": 256.0, "dt": 2.0, "label": "每侧节点数 m"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "conv",
                    "autoscaled": True,
                    "title": "误差 vs 节点数（半对数）",
                    "ylabel": "|Q−I|",
                    "scale": "log",
                },
                {
                    "id": "tail",
                    "autoscaled": True,
                    "title": "未采样尾巴与饱和节点",
                    "ylabel": "∫₀^{x_min}f",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "① 原始等距规则在 $(1+x)/\\sqrt{x}$ 上**没有定义**（$x=0$ 处函数值是 inf），"
                "这不是「误差大」而是公式里出现了 inf——帧里如实标 finite=false 与 inf 个数。"
                "② 「tanh–sinh 是指数收敛」不等于「能到机器精度」：实测它停在 "
                "$\\sqrt{2u}\\approx1.49\\times10^{-8}$（$h=1/16$、inv_sqrt 型），因为最靠近端点的节点只能是 "
                "$x_{\\min}\\approx u/2$，尾巴 $\\int_0^{x_{\\min}}x^{-1/2}dx=2\\sqrt{x_{\\min}}$ 永远采不到；"
                "把节点从 $129$ 加到 $513$ 误差一动不动。"
                "③ $h$ 与节点数是两个旋钮：$h$ 太大时误差在很小的节点数上就停住"
                "（$h=1/8$ 时 $65$ 个节点就到 $1.1\\times10^{-8}$），$h$ 太小则前段收敛慢得多"
                "（$h=1/32$ 在 $129$ 个节点时还有 $6.1\\times10^{-3}$）。"
            ),
            "theory": [
                "双指数变换 $x=\\tanh\\left(\\frac{\\pi}{2}\\sinh t\\right)$ 把端点奇异变成 $t$ 方向的双指数衰减（笔记定理 8）。",
                "换元 $x=t^p$ 把 $x^\\alpha$ 型奇异变成光滑被积函数，代价是必须先知道 $\\alpha$（笔记定理 7 推论）。",
                "误差地板 = 未采样尾巴 $\\int_0^{x_{\\min}}f$，$x_{\\min}\\approx u/2$ $\\Rightarrow$ inv_sqrt 型地板 $\\sqrt{2u}$（笔记定理 8 推论）。",
            ],
        },
    ],
}


def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "area_convergence":
        return _area_convergence(params, t)
    if viz_id == "gauss_vs_equispaced":
        return _gauss_vs_equispaced(params, t)
    if viz_id == "romberg_table":
        return _romberg_frame(params, t)
    if viz_id == "singular_endpoint":
        return _singular_endpoint(params, t)
    raise KeyError(f"章节 2.4 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _f(value: Any) -> float | None:
    """有限数 → float；NaN/±Inf → ``None``（NLF 头用 ``allow_nan=False``，写 NaN 会让整帧失败）。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clamp_step(t: float, lo: int, hi: int) -> int:
    """把连续时间轴夹到整数步 ``[lo, hi]``（NaN 当作 lo，hi<lo 时取 lo）。"""
    if not math.isfinite(float(t)):
        return int(lo)
    if hi < lo:
        return int(lo)
    return max(int(lo), min(int(hi), int(round(float(t)))))


def _clip_floor(values: Any, floor: float) -> tuple[np.ndarray, int]:
    """把误差曲线按 ``floor`` 截断，返回 ``(截断后的曲线, 被截断的点数)``。

    log 轴画不出 0：直接画会丢点或得到 −∞，于是"误差=0"看起来像"完美收敛"。
    截断到地板并把地板值与截断点数写进 meta，才是一句可核对的话。
    """
    v = np.asarray(values, dtype=float).ravel()
    with np.errstate(invalid="ignore"):
        bad = int(np.count_nonzero(~(v > floor)))
    out = np.where(np.isfinite(v), v, floor)
    return np.maximum(out, floor), bad


def _dense_grid(item: Integrand, n: int = _GRID) -> np.ndarray:
    return np.linspace(item.a, item.b, int(n))


def _panel_interpolant(
    method: str,
    nodes: np.ndarray,
    values: np.ndarray,
    *,
    per_panel: int = _PER_PANEL,
) -> tuple[np.ndarray, np.ndarray]:
    """把"当前面板上的分段逼近"画出来：梯形用线性插值，Simpson/3-8 用面板内多项式。

    这不是"再算一遍积分"，只是把公式**隐含**的分段插值多项式在面板内采样：
    梯形法隐含线性、Simpson 隐含二次、Simpson-3/8 隐含三次（都过节点）。
    """
    law = rule_law(method)
    step = {"trapezoid": 1, "simpson": 2, "simpson38": 3}[law.name]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    m = int(per_panel)
    for j in range(0, nodes.size - 1, step):
        xa, xb = float(nodes[j]), float(nodes[j + step])
        local_x = np.linspace(xa, xb, m)
        local_nodes = nodes[j : j + step + 1]
        local_vals = values[j : j + step + 1]
        # 面板内 Lagrange 插值（点少，直接求和；不用被禁的成品多项式求值）
        acc = np.zeros_like(local_x)
        for i in range(local_nodes.size):
            basis = np.ones_like(local_x)
            for k in range(local_nodes.size):
                if k != i:
                    basis = basis * (local_x - local_nodes[k]) / (local_nodes[i] - local_nodes[k])
            acc = acc + local_vals[i] * basis
        xs.append(local_x)
        ys.append(acc)
    if not xs:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    return np.concatenate(xs), np.concatenate(ys)


def _panel_grid(rule: str, n_max: int) -> np.ndarray:
    """误差曲线用的面板数网格：在该规则的合法取值里按几何增长取到 ``n_max``。

    网格里放的是**生效值**（Simpson 取偶、3/8 取 3 的倍数），所以拟合用的是真实的
    :math:`h`。上限很小时（例如 n_max=4 对 3/8 规则）只有一两个合法规模，
    这时曲线仍然给出来，但**不做拟合**——"点不够就不报阶"，而不是硬凑一条直线。
    """
    law = rule_law(rule)
    top = max(law.min_sub, int(n_max))
    raw = np.unique(np.round(np.geomspace(float(law.min_sub), float(top), 26)).astype(int))
    eff = np.unique(np.asarray([law.needed_subintervals(int(v)) for v in raw], dtype=int))
    eff = eff[eff <= top]
    if eff.size == 0:
        eff = np.asarray([min(law.needed_subintervals(top), top)], dtype=int)
    return eff


def _fit_window(
    errs: Any,
    floor: float,
    *,
    max_points: int = 12,
    min_points: int = 3,
    jitter: float = 1.25,
) -> tuple[int, int] | None:
    """挑出**可以做幂律拟合**的窗口 ``(start, stop)``；挑不出就返回 ``None``。

    判据有三条，缺一不可：

    1. 细端要去掉触舍入地板的那一段（``err <= 4*floor``）——地板不是 0，它只是不再下降；
    2. 窗口内误差必须大致单调下降（允许 ``jitter`` 倍的非单调抖动），
       否则"斜率"没有意义；
    3. 至少 ``min_points`` 个点，且最多取 ``max_points`` 个（只用最靠渐近区的一段）。

    返回 ``None`` 是**结论**而不是失败：``f=|x|``（折点在节点上、误差恒为 0）与
    Runge 函数（等距梯形在小 n 上忽上忽下，实测 :math:`R^2` 只有 0.06~0.89）
    都属于"这一段没有可信的阶"，帧里如实留白并在 strings 里说明原因。
    """
    e = np.asarray(errs, dtype=float).ravel()
    stop = -1
    for i in range(e.size - 1, -1, -1):
        if math.isfinite(float(e[i])) and e[i] > 4.0 * floor:
            stop = i
            break
    if stop < min_points - 1:
        return None
    start = stop
    while start > 0 and (stop - start + 1) < max_points and e[start] <= e[start - 1] * jitter:
        start -= 1
    if stop - start + 1 < min_points:
        return None
    return int(start), int(stop)


def _fit_power(x: Any, errs: Any, floor: float, **kw: Any) -> tuple[Any, float | None]:
    """在 :func:`_fit_window` 给的窗口里做幂律拟合；窗口不可信或 :math:`R^2<0.98` 就返回 ``None``。

    返回 ``(fit, slope)``：``fit`` 可能是 ``None``（调用方据此留白），``slope`` 是原始斜率（负值）。
    """
    w = _fit_window(errs, floor, **kw)
    if w is None:
        return None, None
    start, stop = w
    xs = np.asarray(x, dtype=float).ravel()[start : stop + 1]
    ys = np.asarray(errs, dtype=float).ravel()[start : stop + 1]
    if xs.size < 3 or np.any(xs <= 0.0) or np.any(ys <= 0.0):
        return None, None
    fit = fit_loglog(xs, ys)
    if not math.isfinite(fit.r2) or fit.r2 < 0.98:
        return None, None
    return (fit, start, stop), float(fit.slope)


def _adaptive_frame(item: Integrand, budget: int, t: float) -> Frame:
    """adaptive 模式：递归细分 + 「估计误差 vs 真误差」（两条都不是对方的替身）。"""
    tol = _ADAPTIVE_RTOL * max(1.0, abs(item.exact))
    res = adaptive_simpson(
        item.f,
        item.a,
        item.b,
        tol,
        max_depth=12,
        exact=item.exact,
        max_intervals=max(1, int(budget)),
    )
    k = _clamp_step(t, 1, max(1, res.n_intervals))
    x = _dense_grid(item)
    est = res.est[:k]
    tr = res.true_err_cum[:k] if res.true_err_cum is not None else np.zeros(k, dtype=float)
    est_abs = np.abs(est)
    scale = max(abs(item.exact), 1.0)
    est_c, bad_est = _clip_floor(est_abs, U * scale)
    tr_c, bad_tr = _clip_floor(np.abs(tr), U * scale)
    levels, counts = _depth_histogram(res.depth[:k])
    lv_eval, evals_by_depth = _depth_evals(res.depth[:k], res.evals[:k])
    worst = res.worst_intervals(min(5, k))

    fr = Frame(chapter="2.4", viz="area_convergence")
    fr.add("x", x, label="采样点 x")
    fr.add("f", item.values(x), label="被积函数 f")
    fr.add("interval_lo", res.lo[:k], label="已接受区间的左端")
    fr.add("interval_hi", res.hi[:k], label="已接受区间的右端")
    fr.add("interval_depth", res.depth[:k], label="区间深度")
    fr.add("error_estimate_curve", est_c, label="自适应估计误差 |Ê|（累计序）")
    fr.add("error_true_curve", tr_c, label="累计真误差 |Q_k−I|")
    fr.add("evals_cum", res.evals_cum[:k], label="累计函数求值次数")
    fr.add("partial", res.partial[:k], label="累计积分值")
    fr.add("depth_level", levels, label="深度")
    fr.add("depth_count", counts, label="该深度的区间数")
    fr.add("depth_evals", evals_by_depth, label="该深度吃掉的求值次数")
    fr.add("worst_index", worst.astype(float), label="局部误差最大的区间（下标）")
    fr.add("worst_est", np.abs(res.est[worst]), label="最大局部估计误差")
    fr.set_meta(
        step=float(k),
        is_adaptive=True,
        n_intervals_total=float(res.n_intervals),
        n_evals_total=float(res.n_evals),
        max_depth_used=float(res.max_depth_used),
        tol=float(tol),
        error_estimate_total=_f(res.error_estimate),
        error_true_total=_f(res.true_error),
        estimator_over_true=_f(
            res.error_estimate / res.true_error if (res.true_error or 0.0) > 0.0 else None
        ),
        value=_f(res.value),
        exact=float(item.exact),
        converged=bool(res.converged),
        error_floor=float(U * scale),
        clipped_points=float(bad_est + bad_tr),
        budget=float(budget),
    )
    fr.set_strings(
        method="adaptive",
        method_label="自适应 Simpson（嵌套法则 + 容差密度）",
        func=item.name,
        func_label=item.label,
        estimator_note=(
            "右图的估计误差是嵌套法则之差 (S_fine−S_coarse)/15，它估的是**自己那个模型**的误差："
            "光滑函数上偏保守（实测可达真误差的 1e4 倍），窄峰函数上偏乐观"
            "（实测 4.3e-5 对 3.5e-2）。两条曲线必须一起看。"
        ),
        floor_note="误差曲线按 u·max(|I|,1) 截断；截断点数见 meta.clipped_points。",
        worst_note="meta.worst 只给局部估计最大的几个区间：估计大 ≠ 真误差大，这是同一枚硬币的两面。",
    )
    return fr


def _depth_histogram(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(depth, dtype=int)
    if d.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    levels = np.arange(0, int(d.max()) + 1)
    counts = np.asarray([float(np.count_nonzero(d == k)) for k in levels], dtype=float)
    return levels.astype(float), counts


def _depth_evals(depth: np.ndarray, evals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(depth, dtype=int)
    e = np.asarray(evals, dtype=float)
    if d.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    levels = np.arange(0, int(d.max()) + 1)
    tot = np.asarray([float(np.sum(e[d == k])) for k in levels], dtype=float)
    return levels.astype(float), tot


# --------------------------------------------------------------------------- #
# 1. 面积逼近
# --------------------------------------------------------------------------- #
def _area_convergence(params: Mapping[str, Any], t: float) -> Frame:
    method = str(params["method"])
    func = str(params["func"])
    n_max = int(params["n"])
    item = integrand(func)
    if method == "adaptive":
        return _adaptive_frame(item, n_max, t)

    law = rule_law(method)
    k = _clamp_step(t, 1, n_max)
    ev = composite_eval(item.f, item.a, item.b, k, rule=method)
    k_eff = ev.n_sub
    ax, ay = _panel_interpolant(law.name, ev.nodes, ev.values)
    x = _dense_grid(item)
    fv = item.values(x)

    grid = _panel_grid(method, n_max)
    eff = effective_sub_counts(method, grid)
    raw = error_curve(item.exact, item.f, item.a, item.b, grid, rule=method)
    scale = max(abs(item.exact), 1.0)
    floor = U * scale * max(1.0, float(np.max(np.abs(ev.values))) / scale)
    err_c, clipped = _clip_floor(raw, floor)
    fit_pack, slope_measured = _fit_power(grid, err_c, floor)
    if fit_pack is not None:
        fit, w_start, w_stop = fit_pack
        r2_meta: float | None = float(fit.r2)
        x0, y0 = float(grid[w_start]), float(err_c[w_start])
        fit_line = theory_line_power(grid, slope_measured, x0, y0)
        theory_line = theory_line_power(grid, -float(law.error_power), x0, y0)
        fit_lo, fit_hi = float(grid[w_start]), float(grid[w_stop])
    else:
        fit_line = np.full(grid.size, floor)
        theory_line = np.full(grid.size, floor)
        r2_meta = None
        slope_measured = None
        fit_lo, fit_hi = None, None

    diag = weight_diagnostics(ev.weights)
    order_cap = item.order_cap
    # catalog 的 order_cap 就是"阶的上限"（= 奇异指数 α+1），实测阶的预测是 min(p, order_cap)：
    # √x（α=0.5 ⇒ cap=1.5）在梯形与 Simpson 上都实测 ≈1.49，正好是 min(2,1.5)=min(4,1.5)=1.5。
    predicted = float(min(law.error_power, order_cap)) if math.isfinite(order_cap) else float(law.error_power)
    local = local_order(grid.astype(float), err_c) if grid.size >= 2 else np.zeros(0)

    fr = Frame(chapter="2.4", viz="area_convergence")
    fr.add("x", x, label="采样点 x")
    fr.add("f", fv, label="被积函数 f")
    fr.add("approx_x", ax, label=f"{law.label} 的分段逼近（当前 {k_eff} 个面板）")
    fr.add("approx_y", ay, label="分段逼近的值")
    fr.add("node_x", ev.nodes, label="求值节点")
    fr.add("node_y", ev.values, label="节点上的函数值")
    fr.add("panel_edge", ev.nodes, label="面板边界")
    fr.add("panel_count", grid.astype(float), label="误差曲线的面板数 n")
    fr.add("error_curve", err_c, label="实测误差 |Q_n−I|")
    fr.add("error_evals", (eff + 1).astype(float), label="对应的函数求值次数")
    fr.add("fit_curve", fit_line, label="实测拟合线（幂律）")
    fr.add("theory_curve", theory_line, label="理论参考线 n^(−p)")
    fr.add("local_order", local, label="逐段局部阶")
    fr.add("weight", ev.weights, label="当前规则的权重")
    fr.set_meta(
        step=float(k),
        is_adaptive=False,
        panels=int(k_eff),
        panels_requested=int(k),
        n_max=float(n_max),
        rule_order=float(law.error_power),
        value=_f(ev.value),
        exact=float(item.exact),
        error_current=_f(abs(ev.value - item.exact) if ev.finite else None),
        finite=bool(ev.finite),
        n_evals=int(ev.n_evals),
        slope_measured=slope_measured,
        slope_theory=-float(law.error_power),
        slope_theory_smoothness=_f(-predicted),
        fit_r2=r2_meta,
        fit_start=fit_lo,
        fit_end=fit_hi,
        error_floor=float(floor),
        clipped_points=float(clipped),
        order_cap=_f(order_cap),
        sum_w=_f(diag.sum_w),
        abs_sum_w=_f(diag.abs_sum_w),
        gamma=_f(diag.condition),
        n_negative_weights=float(diag.n_negative),
        fitted_constant=_f(fitted_constant(err_c, item.a, item.b, grid, power=law.error_power)),
    )
    fr.set_strings(
        method=law.name,
        method_label=law.label,
        func=item.name,
        func_label=item.label,
        smoothness=item.smoothness,
        order_note=(
            "理论参考线的斜率是公式的阶 p=%d；实测斜率只在渐近窗口内成立"
            "（窗口写进 meta.fit_start/fit_end）。" % law.error_power
        ),
        smoothness_note=(
            f"光滑性上限给出实测阶的预测 min(p, α+1)={predicted:.2f}（α+1={order_cap:g}）"
            + ("（被积函数光滑，不受限）" if not math.isfinite(order_cap) else "：量到的是被积函数光滑度的阶")
        ),
        floor_note=(
            "误差曲线在 %.2e 处截断（u·max|f| 量级）：log 轴画不出 0，"
            "把地板当「完美收敛」是本章最常见的误读。" % floor
        ),
        fit_note=(
            "没有实测斜率 = 这一段没有可信的幂律（窗口触地板、误差不单调或 R²<0.98），"
            "帧里留白而不是硬凑一条直线。"
            if slope_measured is None
            else f"实测斜率取自 n∈[{fit_lo:g},{fit_hi:g}]、R²={r2_meta:.6f} 的窗口。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# 2. Gauss vs 等距
# --------------------------------------------------------------------------- #
def _gauss_vs_equispaced(params: Mapping[str, Any], t: float) -> Frame:
    r"""节点位置、多项式精确度的双向证据、条件数，以及同一函数上的收敛对比。

    这里所有曲线都在**被积函数自己的区间** :math:`[a,b]` 上算：代数精度
    :math:`2n-1` 与区间无关，但节点位置与收敛曲线必须和 :math:`f` 的区间一致，
    否则"Gauss 节点更靠端点"会被误读成"节点位置任意"。
    """
    n = int(params["n"])
    func = str(params["func"])
    item = integrand(func)
    a, b = item.a, item.b
    rule = gauss_legendre_rule(n, a, b)
    equi_nodes = np.linspace(a, b, n)
    equi_w = newton_cotes_weights(n, a, b)
    limit = 2 * n - 1
    k_max = 2 * n + 2
    k = _clamp_step(t, 0, k_max)
    degree = np.arange(0, k_max + 1, dtype=float)
    # 精确度测试放在**标准区间 [−1,1]** 上：代数精度是区间无关的性质，但在 [0,1] 上
    # x^{2n} 的系数是 (1/2)^{2n}，n=20 时误差被压到 1e-16 的地板之下，
    # "k=2n 当场失效"这半边证据就凭空消失了（实测 n=20 时因子 0.52 < 1）。
    mo, mb = -1.0, 1.0
    me = np.linspace(mo, mb, n)
    mw = newton_cotes_weights(n, mo, mb)
    err_g = moment_errors(n, k_max=k_max, a=mo, b=mb)
    err_e = np.asarray(
        [
            abs(float(np.sum(mw * me**j)) - exact_monomial(mo, mb, j))
            for j in range(k_max + 1)
        ],
        dtype=float,
    )
    big = max(abs(exact_monomial(mo, mb, j)) for j in range(k_max + 1))
    floor = U * max(1.0, big)
    err_g_c, bad_g = _clip_floor(err_g, floor)
    err_e_c, bad_e = _clip_floor(err_e, floor)

    m_grid = np.arange(2, 21, dtype=float)
    diags = [weight_diagnostics(newton_cotes_weights(int(m), a, b)) for m in m_grid]
    cond_e = np.asarray([d.condition for d in diags], dtype=float)
    cond_g = np.ones(m_grid.size, dtype=float)
    neg_e = np.asarray([float(d.n_negative) for d in diags], dtype=float)
    min_e = np.asarray([d.min_w for d in diags], dtype=float)

    # 同一次求值预算下的收敛对比：两边都用 nn 个节点（Gauss 用 nn 点公式，
    # 等距用 nn-1 个梯形面板 —— 两者都要 nn 次函数求值）。
    nn = np.arange(2, n + 1, dtype=int)
    conv_g = np.empty(nn.size, dtype=float)
    conv_e = np.empty(nn.size, dtype=float)
    for i, m in enumerate(nn):
        r = gauss_legendre_rule(int(m), a, b)
        conv_g[i] = abs(r.integrate_values(item.values(r.nodes)) - item.exact)
        conv_e[i] = abs(composite_rule(item.f, a, b, int(m) - 1, rule="trapezoid") - item.exact)
    conv_g_c, bad_cg = _clip_floor(conv_g, floor)
    conv_e_c, bad_ce = _clip_floor(conv_e, floor)
    rate_g = None
    order_e = None
    if nn.size >= 4:
        # 窗口：跳过预渐近的粗端，并排除触地板的细端。
        # 注意两条曲线要用**不同的律**描述：Gauss 在解析函数上是几何收敛
        # （半对数直线），等距梯形是代数收敛（双对数直线）。硬把 Gauss 塞进幂律
        # 拟合会得到一个 21.7 这样的"阶"——那是把一个超指数曲线硬说成幂律的产物。
        wg = _fit_window(conv_g_c, floor)
        if wg is not None:
            rate_g = _f(float(fit_semilog(nn[wg[0] : wg[1] + 1].astype(float), conv_g_c[wg[0] : wg[1] + 1]).slope))
        # 等距那一边的步长是 h=(b−a)/(节点数−1)：用节点数当横坐标会把阶拟合歪
        # （实测 2.76 而不是 2.00），这是本章实现期真踩过的第二个坑。
        h_all = (b - a) / (nn.astype(float) - 1.0)
        pack_e, slope_e = _fit_power(h_all, conv_e_c, floor)
        if pack_e is not None:
            order_e = _f(-slope_e)

    dg = weight_diagnostics(rule.weights)
    de = weight_diagnostics(equi_w)

    def _pick(errs: np.ndarray, idx: int) -> float | None:
        return _f(errs[idx]) if 0 <= idx < errs.size else None

    fr = Frame(chapter="2.4", viz="gauss_vs_equispaced")
    fr.add("node_gauss", rule.nodes, label="Gauss–Legendre 节点")
    fr.add("weight_gauss", rule.weights, label="Gauss 权重（全正）")
    fr.add("node_equi", equi_nodes, label="等距节点")
    fr.add("weight_equi", equi_w, label="等距 Newton–Cotes 权重（m≥9 起有负值）")
    fr.add("degree", degree, label="多项式次数 k")
    fr.add("error_moment_gauss", err_g_c, label="Gauss 的矩误差 |Q−∫x^k|")
    fr.add("error_moment_equi", err_e_c, label="等距规则的矩误差")
    fr.add("m_grid", m_grid, label="点数 m")
    fr.add("cond_equi", cond_e, label="等距规则的 Γ=Σ|w|/Σw")
    fr.add("cond_gauss", cond_g, label="Gauss 的 Γ（恒为 1）")
    fr.add("neg_count_equi", neg_e, label="等距规则的负权重个数")
    fr.add("min_weight_equi", min_e, label="等距规则的最小权重")
    fr.add("conv_n", nn.astype(float), label="节点数（两边相同）")
    fr.add("error_conv_gauss", conv_g_c, label="Gauss 的误差（同预算）")
    fr.add("error_conv_equi", conv_e_c, label="等距梯形的误差（同预算）")
    fr.set_meta(
        step=float(k),
        n=float(n),
        interval_a=float(a),
        interval_b=float(b),
        moment_interval=1.0,
        degree_exact_limit=float(limit),
        degree_2n=float(2 * n),
        error_at_limit=_pick(err_g, limit),
        error_beyond_limit=_pick(err_g, 2 * n),
        failure_factor=_f(moment_failure_factor(n, a=mo, b=mb)),
        error_equi_at_limit=_pick(err_e, limit),
        error_equi_beyond=_pick(err_e, 2 * n),
        sum_w_gauss=_f(dg.sum_w),
        sum_w_equi=_f(de.sum_w),
        gamma_gauss=_f(dg.condition),
        gamma_equi_n=_f(de.condition),
        n_negative_gauss=float(dg.n_negative),
        n_negative_equi=float(de.n_negative),
        min_weight_gauss=_f(dg.min_w),
        min_weight_equi=_f(de.min_w),
        error_floor=float(floor),
        clipped_points=float(bad_g + bad_e + bad_cg + bad_ce),
        conv_rate_gauss=rate_g,
        conv_order_equi=order_e,
        error_beyond_relative=_f(
            abs(err_g[2 * n]) / abs(exact_monomial(mo, mb, 2 * n)) if 2 * n < err_g.size else None
        ),
        knots_m9_gamma=_f(weight_diagnostics(newton_cotes_weights(9, a, b)).condition),
        knots_m19_gamma=_f(weight_diagnostics(newton_cotes_weights(19, a, b)).condition),
    )
    fr.critical["degree_exact_limit"] = float(limit)
    fr.critical["equi_first_negative"] = 9.0
    fr.set_strings(
        func=item.name,
        func_label=item.label,
        exactness_note=(
            "k ≤ 2n−1 的矩误差在舍入地板、k = 2n 当场失效——两个方向都画出来，"
            "「精确到 2n−1 次」才有上界。矩误差取标准区间 [−1,1]："
            "在 [0,1] 上 x^{2n} 的系数会小到把失效证据压进地板（实测 n=20 时因子只有 0.52）。"
        ),
        condition_note=(
            "Γ=Σ|w|/Σw 是函数值相对误差到积分相对误差的放大因子；"
            "Gauss 权重全正 ⇒ Γ≡1；等距规则 m=9 起出现负权重，Γ 离开 1。"
        ),
        budget_note="收敛对比两边用**相同的节点数**（Gauss 的 n 点公式 vs 等距的 n−1 个梯形面板）。",
        law_note=(
            "两条误差曲线用两种律描述：Gauss 是半对数直线（几何收敛，meta.conv_rate_gauss 是每节点速率），"
            "等距梯形是双对数直线（代数收敛，meta.conv_order_equi 是阶）。"
            "把 Gauss 硬塞进幂律拟合会得到 21.7 这种没有含义的「阶」。"
        ),
        sum_note="Σw=b−a 只说明常数被积对，与高阶精确度无关。",
        floor_note="矩误差曲线按 u·max|∫x^k| 量级的地板截断（log 轴画不出 0）。",
    )
    return fr


# --------------------------------------------------------------------------- #
# 3. Romberg
# --------------------------------------------------------------------------- #
def _romberg_frame(params: Mapping[str, Any], t: float) -> Frame:
    func = str(params["func"])
    levels = int(params["levels"])
    item = integrand(func)
    res = romberg_table(item.f, item.a, item.b, levels, exact=item.exact)
    k = _clamp_step(t, 1, levels)

    js: list[float] = []
    ks: list[float] = []
    vals: list[float] = []
    errs: list[float] = []
    for row in range(k):
        for col in range(row + 1):
            js.append(float(col))
            ks.append(float(row))
            val = float(res.table[row, col])
            vals.append(val)
            errs.append(abs(val - item.exact))
    err_arr = np.asarray(errs, dtype=float)
    scale = max(abs(item.exact), 1.0)
    floor = U * scale
    err_c, clipped = _clip_floor(err_arr, floor)

    diag = np.abs(res.diagonal()[:k] - item.exact)
    diag_c, bad_d = _clip_floor(diag, floor)
    # 列阶/加速比/地板层都必须用**已经揭示的那一部分**表来算：
    # t=0 时表里只有第 0 行，任何一列都凑不出 3 个点，于是帧里如实没有阶。
    revealed = RombergResult(
        level=res.level[:k],
        n_sub=res.n_sub[:k],
        evals=res.evals[:k],
        table=res.table[:k, :k].copy(),
        exact=item.exact,
    )
    orders = column_orders(revealed, item.exact, trim_tail=0)
    order_j = np.arange(orders.size, dtype=float)
    good = np.isfinite(orders)
    gains_j, gains_k, gains = _gains(revealed, item.exact, k)

    sample = extrapolation_coefficients(max(0, k - 1))
    order_cols = {f"order_col{j}": _f(orders[j]) if j < orders.size else None for j in range(4)}

    fr = Frame(chapter="2.4", viz="romberg_table")
    fr.add("column_j", np.asarray(js, dtype=float), label="列号 j")
    fr.add("column_k", np.asarray(ks, dtype=float), label="行号 k（层）")
    fr.add("column_value", np.asarray(vals, dtype=float), label="T_{k,j}")
    fr.add("error_table", err_c, label="|T_{k,j}−I|（长表，(j,k) 序）")
    fr.add("level", res.level[:k], label="层 k")
    fr.add("n_sub", res.n_sub[:k], label="子区间数 2^k")
    fr.add("evals", res.evals[:k], label="累计函数求值次数")
    fr.add("diag_error", diag_c, label="对角线误差 |T_{k,k}−I|")
    fr.add("order_j", order_j[good], label="有阶可言的列号")
    fr.add("order_measured", orders[good], label="该列的实测阶")
    fr.add("order_theory", 2.0 * order_j[good] + 2.0, label="该列的理论阶 2j+2")
    fr.add("gain_j", gains_j, label="加速比所在列 j")
    fr.add("gain_k", gains_k, label="加速比所在行 k")
    fr.add("gain", gains, label="|T_{k,j}−I|/|T_{k,j+1}−I|")
    fr.add("coef", sample, label=f"第 {max(0, k - 1)} 列的外推系数")
    fr.add("coef_index", np.arange(sample.size, dtype=float), label="系数编号")
    fr.set_meta(
        step=float(k),
        levels=float(levels),
        func_smooth=bool(func != "cusp"),
        value=_f(float(res.table[k - 1, k - 1])),
        exact=float(item.exact),
        diag_error_current=_f(float(diag_c[k - 1])),
        diag_error_prev=_f(float(diag_c[k - 2])) if k >= 2 else None,
        error_floor=float(floor),
        clipped_points=float(clipped + bad_d),
        coefficient_abs_sum=_f(float(np.sum(np.abs(sample)))),
        coefficient_sum=_f(float(np.sum(sample))),
        n_columns=int(k),
        **order_cols,
    )
    fr.critical["diagonal_plateau_level"] = float(_floor_level(revealed, item.exact))
    fr.set_strings(
        func=item.name,
        func_label=item.label,
        column_note=(
            "第 j 列只有 m−j 个可用点：列号越大、可用行越少，"
            "m−j<3 的列在帧里被剔除（没有阶可言，不编一个 0）。"
        ),
        window_note=(
            "每列的拟合窗口都剔掉触到舍入地板的行（不剔时第 3 列实测 6.96、剔掉得 7.97，理论 8）："
            "「误差阶」这句话必须带上窗口。"
        ),
        floor_note=(
            "对角线在 %.2e 处触地板：外推系数绝对值之和 A_j 有界（→1.9692603），"
            "所以舍入噪声只被放大近 2 倍，但截断误差已经小到无意义。" % floor
        ),
        nonsmooth_note=(
            "f=|x| 的折点落在偶数层的节点上，梯形法逐段精确，"
            "整张表从第 2 层起全是舍入噪声——这时「列阶」没有意义。"
            if func == "cusp"
            else "被积函数足够光滑，Romberg 的列阶应当逐列 +2。"
        ),
    )
    return fr


def _gains(
    res: Any,
    exact: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """同一行上相邻两列的误差比（只取已揭示的层）。"""
    js: list[float] = []
    ks: list[float] = []
    gs: list[float] = []
    for row in range(k):
        for col in range(row):
            e0 = abs(float(res.table[row, col]) - exact)
            e1 = abs(float(res.table[row, col + 1]) - exact)
            js.append(float(col))
            ks.append(float(row))
            gs.append(e0 / e1 if e1 > 0.0 else 0.0)
    return np.asarray(js, dtype=float), np.asarray(ks, dtype=float), np.asarray(gs, dtype=float)


def _floor_level(res: Any, exact: float, *, factor: float = 4.0) -> int:
    """对角线不再显著改善的那一层（实测）：第一个"没比上一层好 factor 倍"的层号。

    注意它有两层含义，帧里靠 strings 说清楚：光滑函数上是**触到舍入地板**，
    非光滑函数（Runge、|x|）上则是"外推已经没有截断误差可消"，此时地板还远。
    """
    d = np.abs(res.diagonal() - float(exact))
    for k in range(1, d.size):
        if not (d[k] < d[k - 1] / factor):
            return int(k)
    return int(d.size)


# --------------------------------------------------------------------------- #
# 4. 端点奇异
# --------------------------------------------------------------------------- #
def _singular_endpoint(params: Mapping[str, Any], t: float) -> Frame:
    func = str(params["func"])
    h_key = str(params["h"])
    h = _H_CHOICES.get(h_key, 2.0**-4)
    n_max = int(params["n"])
    item = integrand(func)
    m_max = max(1, (n_max - 1) // 2)
    m = _clamp_step(t, 1, m_max)

    rule = tanh_sinh_rule(m, h=h, a=item.a, b=item.b)
    val = rule.integrate_values(item.values(rule.nodes))
    err = abs(val - item.exact)

    ms = np.arange(1, m_max + 1)
    ts_err = np.empty(ms.size, dtype=float)
    ts_sat = np.empty(ms.size, dtype=float)
    ts_xmin = np.empty(ms.size, dtype=float)
    ts_tail = np.empty(ms.size, dtype=float)
    for i, mm in enumerate(ms):
        r = tanh_sinh_rule(int(mm), h=h, a=item.a, b=item.b)
        v = r.integrate_values(item.values(r.nodes))
        ts_err[i] = abs(v - item.exact)
        ts_sat[i] = float(r.n_saturated)
        # x_min 必须取**未饱和**的最近节点：饱和节点的坐标已经舍入到 0（或 1），
        # 它们的"尾巴"是 0，会把地板机制整个盖住（实测踩过：直接用 min(nodes) 时
        # 大 m 上 x_min=0、tail=0，图上地板线凭空消失）。
        live = ~r.saturated
        xmin = float(np.min(r.nodes[live])) if bool(np.any(live)) else float(np.min(r.nodes))
        ts_xmin[i] = xmin
        tail = item.tail_integral(xmin)
        ts_tail[i] = abs(tail) if tail is not None else 0.0

    nodes = (2 * ms + 1).astype(float)
    scale = max(abs(item.exact), 1.0)
    floor = U * scale
    ts_c, bad_ts = _clip_floor(ts_err, floor)
    tail_c, _ = _clip_floor(np.maximum(ts_tail, floor), floor)
    reveal = nodes <= (2 * m + 1)

    sub_counts = 2 * ms
    sub_available = item.substitution is not None and math.isfinite(float(item.substitution.domain[1]))
    if sub_available:
        sub_c, bad_sub = _clip_floor(substituted_error_curve(item, sub_counts, rule="trapezoid"), floor)
        _, slope_sub = _fit_power(sub_counts.astype(float), sub_c, floor)
        sub_order = _f(-slope_sub) if slope_sub is not None else None
    else:
        # 没有可用的有限区间换元时**留白**（整条曲线按地板给出、meta 里明说 sub_available=false），
        # 不编一条曲线：NaN 被地板截断之后看起来和"正常的曲线"一模一样。
        sub_c = np.full(sub_counts.size, floor, dtype=float)
        bad_sub = 0
        sub_order = None

    tail_theory = item.tail_integral(U / 2.0)
    raw = composite_eval(item.f, item.a, item.b, 8, rule="trapezoid")
    n_bad_raw = int(np.count_nonzero(~np.isfinite(raw.values)))
    n_rev = int(np.count_nonzero(reveal))
    sl = None
    if n_rev >= 4:
        wts = _fit_window(ts_c[reveal], floor)
        if wts is not None:
            sl = fit_semilog(nodes[reveal][wts[0] : wts[1] + 1], ts_c[reveal][wts[0] : wts[1] + 1])

    fr = Frame(chapter="2.4", viz="singular_endpoint")
    fr.add("ts_m", ms.astype(float), label="每侧节点数 m")
    fr.add("ts_nodes", nodes, label="总节点数 2m+1")
    fr.add("error_ts", np.where(reveal, ts_c, floor), label="tanh–sinh 误差")
    fr.add("ts_saturated", ts_sat, label="饱和节点数（权重下溢到 0）")
    fr.add("ts_xmin", np.maximum(ts_xmin, floor), label="最靠近奇异端点的节点 x_min")
    fr.add("ts_tail", tail_c, label="未采样尾巴 |∫₀^{x_min} f|")
    fr.add("sub_nodes", (sub_counts + 1).astype(float), label="换元后等距规则的节点数")
    fr.add("error_sub", sub_c, label="换元 + 等距规则的误差")
    fr.add("floor_nodes", nodes, label="地板参考线的横坐标（节点数）")
    fr.add("floor_value", np.full(nodes.size, abs(tail_theory) if tail_theory is not None else floor),
           label="地板理论值 |∫₀^{u/2} f|")
    xd = _dense_grid(item)
    fd = item.values(xd)
    defined = np.isfinite(fd)
    # 显示用截断：x^{-1/2} 在 x=0 处是 inf（数组里不许有 inf），
    # 截断到 x=u/2 处的函数值，并把未定义点数与掩码如实交给前端（断线而不是画一条假的斜线）。
    cap = float(item.values(np.asarray([U / 2.0], dtype=float))[0])
    if not math.isfinite(cap):
        cap = float(np.max(fd[defined])) if bool(np.any(defined)) else 1.0
    f_disp = np.where(defined, fd, cap)
    fr.add("x", xd, label="被积函数采样点")
    fr.add("f", f_disp, label=f"被积函数（未定义点截断到 u/2 处的值 {cap:.3g}）")
    fr.add("f_defined", defined.astype(float), label="该点函数值是否有定义（1/0）")
    fr.set_meta(
        step=float(m),
        nodes=int(rule.n_nodes),
        h=float(h),
        error_current=_f(err),
        n_saturated=int(rule.n_saturated),
        x_min=_f(xmin),
        tail_estimate=_f(item.tail_integral(xmin)),
        tail_ratio=_f(err / abs(tail_theory)) if (tail_theory or 0.0) != 0.0 else None,
        tail_theory=_f(tail_theory),
        error_floor=float(floor),
        clipped_points=float(bad_ts + bad_sub),
        sub_order=_f(sub_order),
        ts_slope_final=_f(sl.slope) if sl is not None else None,
        ts_slope_r2=_f(sl.r2) if sl is not None else None,
        raw_equispaced_finite=bool(raw.finite),
        raw_equispaced_nonfinite=float(n_bad_raw),
        raw_equispaced_value=_f(raw.value),
        raw_equispaced_error=_f(abs(raw.value - item.exact) if raw.finite else None),
        constant_node_count=float(2 * ((n_max - 1) // 2) + 1),
        sub_available=bool(sub_available),
        f_display_cap=_f(cap),
        f_undefined_points=float(int(np.count_nonzero(~defined))),
    )
    fr.critical["saturation_nodes"] = float(_saturation_nodes(h))
    fr.critical["u_over_2"] = float(U / 2.0)
    fr.set_strings(
        func=item.name,
        func_label=item.label,
        h_label=f"h = {h_key} = {h:.6g}",
        smoothness=item.smoothness,
        floor_note=(
            "tanh–sinh 的地板不是 u 而是未被采样的尾巴 ∫₀^{x_min}f："
            "x_min≈u/2，x^{-1/2} 型给出 2√x_min≈√(2u)≈1.5e-8，"
            "对数型给出 x_min|ln x_min|≈2e-15。"
        ),
        saturation_note=(
            "饱和节点是 |tanh((π/2)sinh t)| 在 double 里舍入到 1 的那些点："
            "它们的权重下溢到 0，贡献必须按 0 计（0·inf 会得到 NaN）。"
        ),
        display_note=(
            "被积函数曲线在未定义点（x=0）处截断到 u/2 处的函数值，"
            "并用 f_defined 掩码标出：画一条连着 inf 的假曲线比留个断口更误导。"
        ),
        raw_note=(
            "等距规则在原始被积函数上**没有定义**（x=0 处 f 无界）："
            f"实测 {n_bad_raw} 个节点值非有限；换元之后才重新可用。"
        ),
        sub_note=(
            "对照曲线用 x=t² 换元后的被积函数，同一套梯形法逐次加密。"
            if sub_available
            else "这一格没有可用的有限区间换元，对照曲线留白（meta.sub_available=false）。"
        ),
    )
    return fr


def _saturation_nodes(h: float) -> int:
    """在步长 ``h`` 下，第一个出现饱和节点的总节点数（解析临界 t* 除以 h 再取整）。"""
    from numcore.quad.singular import saturation_abscissa

    m_star = int(math.ceil(saturation_abscissa() / float(h) - 1e-12))
    return 2 * m_star + 1

