r"""3.1 章：单步法、阶条件与自适应步长（4 个可视化）。

帧只做搬运：所有数学量都来自 :mod:`numcore.ode.rk`（方法族、阶条件、误差阶梯、稠密输出）
与 :mod:`numcore.ode.stepsize`（控制器、自适应积分、工作-精度前沿），本文件不写数学。
四个可视化的分工是"从阶 → 步长 → 条件 → 代价"：

1. ``order_convergence`` —— **招牌**：主图是**真跑出来**的解曲线与步点，副图是同一串
   :math:`h` 上的四条误差曲线（整段最大 / 终点 / 局部 / 步内插值）与理论斜率参考线。
   时间轴是真实时间 :math:`t`：向右拖，积分跟着往前推，节点密度就是当前 :math:`h`。
   端点误差与最大误差的关系（T=10 时差 800 倍）在这里可以当场读出来。
2. ``adaptive_step`` —— 自适应步长序列：每次**试步**的 :math:`h`、归一化误差估计
   :math:`\varepsilon`（接受阈值 1）、以及真局部误差。时间轴是"试步进度"（0–100 = 整段积分），
   被拒绝的步用掩码区分：看得见"先拒绝再缩小"的锯齿，也看得见估计器滞后。
3. ``order_conditions`` —— 有根树与阶条件：17 棵树画在单位方框里，每棵树的
   :math:`\Phi(\tau)` 与 :math:`1/\gamma(\tau)` 的残差画在副轴（对数）。时间轴是条件序号。
   换方法/换 :math:`b_{\hat p}`，可以看着残差从 :math:`10^{-17}` 跳到 :math:`10^{-1}`。
4. ``work_precision`` —— 工作-精度前沿（无时间轴）：左图 x=请求容差、y=实到误差（对数），
   右图 x=工作量、y=实到误差。左图上四条曲线**平行但有偏移**，这就是"同一个 tol
   不是同一个工作点"；右图才是可以公平比较的地方。

帧契约（与 1.2 / 3.2 章一致）：``meta`` 只放有限标量（``allow_nan=False``，非有限值一律
换成哨兵 + 显式掩码），字符串走 ``strings``，数组走 ``add``。每个时间轴的语义写在
``set_strings(time_semantics=...)`` 里。
"""

from __future__ import annotations

import functools
import math
from typing import Any, Mapping

import numpy as np

from numcore.ode import rk
from numcore.ode import stepsize as ss
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 单位舍入（= eps/2）：对数轴地板与"精确 0"的显示约定都用它。
_ROUNDOFF = float(np.finfo(float).eps) / 2.0

#: 误差轴的地板：比这更小的值一律显示为地板值，并如实写进 meta。
_ERR_FLOOR = 1.0e-17

#: 真局部误差/工作量的显示上限（"软着陆"）：超过它截断显示，真值留在 meta。
_BIG_FILL = 1.0e12

_NONFINITE_POLICY = (
    "本帧不含任何非有限值：误差比对数的地板是 strings.err_floor（={floor:g}），"
    "真局部误差超过显示上限的位置用 strings.big_fill（={big:g}）占位并配 *_clipped 掩码；"
    "折线不用 NaN 分隔（NaN 会让前端静默丢段）；meta 一律是有限标量"
    "（.nlf 头用 allow_nan=False 编码，塞一个 NaN 整帧就写不出去）"
)

#: 时间轴语义逐个写清（PROTOCOLS：时间轴的含义必须在帧里说得出）。
_TIME_SEMANTICS = {
    "order_convergence": "真实时间 t：0 → 当前 t，解曲线按当前 h 步进到 t",
    "adaptive_step": "试步进度（0–100）：0 = 还没开始，100 = 整段积分走完；第 k 次试步见 meta.attempt_index",
    "order_conditions": "阶条件序号：第 k 条（按有根树枚举顺序），tMax=17 是 5 阶树的条数",
    "work_precision": "无时间轴（一次性扫描）",
}


def _fin(x: Any, default: float = 0.0) -> float:
    """把非有限标量换成哨兵（``.nlf`` 头用 ``allow_nan=False``，NaN/inf 写不进去）。"""
    v = float(x)
    return v if math.isfinite(v) else float(default)


def _floor(arr: Any, floor: float = _ERR_FLOOR) -> np.ndarray:
    """对数轴的「软着陆」：把 0 与非有限值抬到地板上并如实报告。"""
    a = np.asarray(arr, dtype=float).copy()
    a[~np.isfinite(a)] = floor
    return np.maximum(a, floor)


def _clip_big(arr: Any, ceil: float = _BIG_FILL) -> tuple[np.ndarray, np.ndarray]:
    """把超过显示上限（或非有限）的元换成 ``ceil``，返回 ``(数组, 截断掩码)``。"""
    a = np.asarray(arr, dtype=float).copy()
    bad = ~np.isfinite(a) | (np.abs(a) > ceil)
    return np.where(bad, ceil, a), bad.astype(float)


def _asym_slope(h: Any, err: Any, theory: float) -> tuple[float, int, int]:
    """按本章统一的窗口规则拟合斜率，返回 ``(斜率, 丢头点数, 丢尾点数)``。

    窗口规则在 :func:`numcore.ode.rk.asymptotic_window` 里，测试、断言表与这张图共用同一套，
    谁都没机会"为了让斜率好看"临时挪窗口。点不够时返回 ``nan``（帧里再换成哨兵）。
    """
    try:
        hw, ew, k, m = rk.asymptotic_window(h, err, theory)
    except ValueError:
        return float("nan"), 0, 0
    return float(rk.loglog_slope(hw, ew)), int(k), int(m)


def _step_of(t: float, n_steps: float) -> int:
    """把连续时间轴四舍五入到整数步并夹住。"""
    k = int(math.floor(float(t) + 0.5))
    return max(0, min(k, int(n_steps)))


# --------------------------------------------------------------------------- #
# 缓存：scrub 帧会被连续拖，重算阶梯/前沿会很贵（viz1 一次 ~0.3 s，viz4 ~0.5 s）
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=64)
def _ladder(problem: str, method: str) -> rk.Ladder:
    return rk.error_ladder(problem, method)


@functools.lru_cache(maxsize=64)
def _traj(problem: str, method: str, h_key: float) -> rk.Trajectory:
    prob = rk.problem(problem)
    return rk.integrate_fixed(method, prob.rhs, (0.0, prob.t_end), prob.y0_array(), float(h_key), exact=prob.exact)


@functools.lru_cache(maxsize=32)
def _adapt(problem: str, method: str, tol: float, law: str, extrap: bool, t_end: float) -> ss.AdaptiveRun:
    prob = rk.problem(problem)
    return ss.integrate_adaptive(
        method,
        prob.rhs,
        (0.0, t_end),
        prob.y0_array(),
        float(tol),
        law=ss.ControlLaw(name=law),
        local_extrapolation=bool(extrap),
        exact=prob.exact,
        true_local_error=True,
        n_sub=6,
        max_steps=20000,
    )


@functools.lru_cache(maxsize=64)
def _tree_pack(max_order: int) -> dict[str, np.ndarray]:
    """把所有 ≤max_order 阶的有根树排成网格，返回节点/边/中心三组数组。"""
    trees = rk.rooted_trees(int(max_order))
    n = len(trees)
    ncol = int(math.ceil(math.sqrt(n)))
    nrow = int(math.ceil(n / ncol))
    node_x: list[float] = []
    node_y: list[float] = []
    node_tree: list[float] = []
    edge_x0: list[float] = []
    edge_y0: list[float] = []
    edge_x1: list[float] = []
    edge_y1: list[float] = []
    edge_tree: list[float] = []
    cx: list[float] = []
    cy: list[float] = []
    for i, tree in enumerate(trees):
        x, y, parent = rk.tree_layout(tree)
        col = i % ncol
        row = i // ncol
        w = 1.0 / ncol
        hgt = 1.0 / nrow
        pad = 0.12
        sx = x * (w * (1.0 - 2 * pad)) + col * w + w * pad
        sy = y * (hgt * (1.0 - 2 * pad)) + (nrow - 1 - row) * hgt + hgt * pad
        node_x.extend(sx.tolist())
        node_y.extend(sy.tolist())
        node_tree.extend([float(i)] * sx.size)
        for j in range(1, sx.size):
            p = int(parent[j])
            edge_x0.append(float(sx[p]))
            edge_y0.append(float(sy[p]))
            edge_x1.append(float(sx[j]))
            edge_y1.append(float(sy[j]))
            edge_tree.append(float(i))
        cx.append(float(np.mean(sx)))
        cy.append(float(np.mean(sy)))
    return {
        "node_x": np.asarray(node_x),
        "node_y": np.asarray(node_y),
        "node_tree": np.asarray(node_tree),
        "edge_x0": np.asarray(edge_x0),
        "edge_y0": np.asarray(edge_y0),
        "edge_x1": np.asarray(edge_x1),
        "edge_y1": np.asarray(edge_y1),
        "edge_tree": np.asarray(edge_tree),
        "tree_cx": np.asarray(cx),
        "tree_cy": np.asarray(cy),
    }


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
MANIFEST: dict[str, Any] = {
    "chapter": "3.1",
    "title": "单步法、阶条件与自适应步长",
    "part": "III",
    "summary": (
        "阶条件把「一个方法几阶」变成一组可以逐条验算的和：每棵有根树一条，"
        "Φ(τ)=1/γ(τ)。满足到第 p 棵，误差就按 h^p 走——这一章把两条证据链摆在一起："
        "树上算出来的阶，和真跑积分量出来的斜率。自适应步长则把这套阶变成工程："
        "估计器估的是哪个解，决定了「tol 是不是精度」。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "order_convergence",
            "title": "阶：误差-步长双对数阶梯与真跑出来的解",
            "description": (
                "主图是真算出来的解曲线与步点（时间轴就是真实时间 t，向右拖积分往前推，"
                "节点密度就是当前步长 h）。副图是同一串 h 上的四条误差曲线：整段最大误差、"
                "终点误差、首步真局部误差、步内插值（稠密输出）误差，外加两条理论斜率参考线。"
                "当前 h 用竖线标出，它的四个误差值同时高亮——端点和最大误差的关系当场可见。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "method",
                    "label": "方法",
                    "type": "enum",
                    "options": list(rk.METHODS),
                    "default": "rk4",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "problem",
                    "label": "算例",
                    "type": "enum",
                    "options": ["exp_decay", "exp_decay_long", "logistic", "oscillator"],
                    "default": "exp_decay",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "h",
                    "label": "步长 h",
                    "type": "float",
                    "min": 0.00390625,
                    "max": 0.5,
                    "step": 0.00390625,
                    "default": 0.0625,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 10.0, "dt": 0.05, "label": "t（真实时间，积分推进）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "solution",
                    "autoscaled": True,
                    "title": "解曲线（当前 h 的步点画在曲线上）",
                    "ylabel": "y",
                },
                {
                    "id": "error",
                    "autoscaled": False,
                    "title": "误差 vs 步长（双对数，固定窗口）",
                    "ylabel": "误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，副图是**固定窗口**的双对数图：改成自适应，四条曲线的相对位置会被拉平，"
                "「稠密输出只有 4 阶」「端点误差小三个数量级」这两个结论会当场消失。"
                "第二，斜率参考线是按**声明的阶**画的，不是按拟合出来的斜率画的；拟合窗口若含"
                "预渐近的粗网格点，量出来的斜率会偏大（rkf45 在 exp_decay 上全窗口是 4.16），"
                "所以窗口的丢弃点数写在 meta 里。第三，误差比对数的地板是 1e-17：到了地板"
                "就不是「误差很小」，而是「双精度已经量不出来了」，两条曲线在地板处贴合是假象。"
            ),
            "theory": [
                "全局误差 ~ h^p、局部误差 ~ h^{p+1}；p 由阶条件 bᵀA^{j−1}e 决定",
                "稠密输出（三次 Hermite）的步内插值误差 ~ h^min(p,4)：p≥5 时它**低于**方法阶",
                "终点误差 |y_N−y(T)| 与整段最大误差 max_t|y_h−y(t)| 是两个不同的量，长时程上差三个数量级",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "adaptive_step",
            "title": "自适应步长序列：估计器、接受阈值与真误差",
            "description": (
                "每次试步画一根柱子：h（对数）、归一化误差估计 ε（对数，阈值 1 是接受线）、"
                "以及同一尺度下的**真局部误差**。被拒绝的步用掩码标出。"
                "时间轴是试步进度：向右拖，序列一点点长出来。刚性算例上会看到步长被稳定极限"
                "钉在 3.3e-3 附近，而不是被精度要求放开到 2.4e-1。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "method",
                    "label": "嵌入对",
                    "type": "enum",
                    "options": list(rk.PAIR_METHODS),
                    "default": "dopri5",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "problem",
                    "label": "算例",
                    "type": "enum",
                    "options": ["exp_decay", "logistic", "oscillator", "stiff_cos"],
                    "default": "exp_decay",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "tol",
                    "label": "容差 tol",
                    "type": "float",
                    "min": 1e-09,
                    "max": 1e-03,
                    "step": 1e-09,
                    "default": 1e-06,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "law",
                    "label": "控制器",
                    "type": "enum",
                    "options": ["pi", "i"],
                    "default": "pi",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "extrap",
                    "label": "局部外推（推进高阶解）",
                    "type": "bool",
                    "default": True,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 100.0, "dt": 1.0, "label": "试步进度（0–100 = 整段积分）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "solution",
                    "autoscaled": True,
                    "title": "解曲线（自适应节点）",
                    "ylabel": "y",
                },
                {
                    "id": "hseq",
                    "autoscaled": False,
                    "title": "试步的步长 h（对数）",
                    "ylabel": "h",
                    "scale": "log",
                },
                {
                    "id": "errseq",
                    "autoscaled": False,
                    "title": "归一化误差：估计 ε 与真误差（阈值 1）",
                    "ylabel": "ε",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，varepsilon 是**归一化**的：ε=1 恰好等于容差，所以柱子穿过 1 就必然被拒绝，"
                "这不是抖动而是判据本身。第二，被拒绝的步也占用了一次函数求值（记账在 meta.n_f_evals 里），"
                "所以「步数少」不等于「便宜」。第三，真局部误差是拿同方法 6 个子步的参考解量的，"
                "它自己也有误差；两者在地板附近贴合不说明估计器准，只说明都到地板了。"
                "第四，刚性算例上估计器**没有**说谎（实测比值 0.5–1.0），真正失效的是"
                "「tol 控制精度」这句话：步长被稳定极限接管。"
            ),
            "theory": [
                "h_new = h·safe·(tol/ê)^{1/(p̂+1)}（I 控制器）；PI 再乘历史项 ε_{n−1}^{0.75β}",
                "误差估计 ê = ‖y_{p} − y_{p̂}‖ 的加权 RMS，尺度 s = atol + rtol·max(|y_new|,|y_old|)",
                "嵌入对的估计量与被推进解同阶时（rkf45）比值 O(1)；差一阶时比值 ~ 1/h（悲观）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "order_conditions",
            "title": "阶条件：每一棵有根树一条方程",
            "description": (
                "副图列出全部 ≤max_order 阶的有根树（坐标为固定单位框，树间按网格排布），"
                "主图给出每棵树的阶 ρ、密度 γ 与残差 |1−γΦ|（对数轴）。"
                "时间轴是条件序号：向右拖逐条点亮。切到 b_hat（嵌入低阶权）可以看到"
                "同一个方法的高阶解与低阶解各自满足到第几条——这正是嵌入对误差估计的来源。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "method",
                    "label": "方法",
                    "type": "enum",
                    "options": list(rk.METHODS),
                    "default": "rk4",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "max_order",
                    "label": "最高树阶",
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "default": 4,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "use_hat",
                    "label": "用 b_hat（低阶嵌入权）",
                    "type": "bool",
                    "default": False,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 17.0, "dt": 1.0, "label": "阶条件序号（按树枚举顺序）"},
            "requiredAux": ["residual"],
            "axes": [
                {
                    "id": "trees",
                    "autoscaled": False,
                    "title": "有根树（单位框，按阶数枚举）",
                    "ylabel": "",
                },
                {
                    "id": "residual",
                    "autoscaled": False,
                    "title": "残差 |1−γΦ|（对数，地板 1e-17）",
                    "ylabel": "残差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，残差轴的地板是 1e-17：到了地板不是「残差小」，而是「双精度算不出更小」，"
                "满足与不满足之间隔着 16 个数量级，图上看不见中间地带。第二，树与树之间没有强弱之分——"
                "第 3 条不满足与第 5 条不满足都让方法掉到同一阶，别把「残差更大」读成「更靠后」。"
                "第三，坐标框是固定的：树的数量随 max_order 变（1,1,2,4,9 棵），"
                "自适应会让 5 阶的 17 棵挤成一团，反而看不出结构。第四，b_hat 的阶通常比 b 低一阶，"
                "残差从第一棵就开始大——那**不是** bug，是嵌入对的定义。"
            ),
            "theory": [
                "Φ(τ) = Σ_i b_i ∏_j (A^{ρ_j}e)_{i}^{σ_j}；γ(τ) = ρ(τ)·∏ γ(τ_j)",
                "方法 p 阶 ⟺ 对全部 ρ(τ) ≤ p 的树有 Φ(τ) = 1/γ(τ)",
                "嵌入对 (b, b̂)：推进 b（高阶），用 b−b̂ 的差做误差估计；推进 b̂ 就是「不开局部外推」",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "work_precision",
            "title": "工作-精度前沿：同一个 tol 不是同一个工作点",
            "description": (
                "左图 x=请求容差 tol、y=实到最大误差（对数），四条自适应曲线外加固定步长 RK4 的曲线，"
                "对角线是 err=tol 参考线；右图 x=每单位时间的函数求值次数、y=实到误差（真正的代价前沿）。"
                "当前 tol 用竖线标出：同一个 tol 下四条曲线的高度差就是「实到误差差 10 倍」。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "problem",
                    "label": "算例",
                    "type": "enum",
                    "options": ["exp_decay", "logistic", "oscillator"],
                    "default": "exp_decay",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "tol",
                    "label": "当前容差 tol",
                    "type": "float",
                    "min": 1e-08,
                    "max": 1e-04,
                    "step": 1e-08,
                    "default": 1e-06,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "extrap",
                    "label": "局部外推（推进高阶解）",
                    "type": "bool",
                    "default": True,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "vs_tol",
                    "autoscaled": False,
                    "title": "请求容差 → 实到误差（对角线 = 说到做到）",
                    "ylabel": "实到最大误差",
                    "scale": "log",
                },
                {
                    "id": "frontier",
                    "autoscaled": False,
                    "title": "工作量 → 实到误差（这才是可比的前沿）",
                    "ylabel": "实到最大误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，左图的四条曲线**不会**重合：同一个 tol 下 heun 与 dopri5 实到的误差能差 10 倍，"
                "所以「同样的容差下谁更省」这个问题本身没有定义——要比就固定在精度上比（右图）。"
                "第二，工作量单位取的是「每单位时间的函数求值次数」：换成「步数」会让高阶方法凭空占便宜，"
                "换成「总求值」又会被积分区间长度带跑。第三，固定步长 RK4 的曲线是用误差阶梯反查出来的"
                "（插值），不是真跑出来的；反查只做插值、不外推，超出实测区间会直接报错。"
                "第四，右图的下包络才是「前沿」：任何一条曲线都不该被读成「这个方法的性能」，"
                "它只是这个算例上的一条切面。"
            ),
            "theory": [
                "工作-精度：固定精度看工作量，或固定工作量看精度；两条路在同一组数据上必须自洽",
                "tol→误差 的幂次：开局部外推是 1，不开是 p̂/p（p̂ = 被推进解的阶）",
                "每步代价：FSAL 方法第一步 s 次求值、其后 s−1 次；稠密输出对 FSAL 是 0 次额外求值",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    if viz_id == "order_convergence":
        return _order_convergence(params, t)
    if viz_id == "adaptive_step":
        return _adaptive_step(params, t)
    if viz_id == "order_conditions":
        return _order_conditions(params, t)
    if viz_id == "work_precision":
        return _work_precision(params, t)
    raise ValueError(f"未知 viz_id {viz_id!r}（3.1 章有 4 个可视化）")


# --------------------------------------------------------------------------- #
# viz 1：阶
# --------------------------------------------------------------------------- #
def _order_convergence(params: Mapping[str, Any], t: float) -> Frame:
    method = str(params["method"])
    problem = str(params["problem"])
    h = float(params["h"])
    prob = rk.problem(problem)
    t_end = float(prob.t_end)
    tab = rk.tableau(method)

    lad = _ladder(problem, method)
    traj = _traj(problem, method, h)
    tt = np.asarray(traj.t, dtype=float)
    yy = np.asarray(traj.y, dtype=float).reshape(tt.size, -1)
    exact_all = np.asarray(prob.exact(tt), dtype=float).reshape(tt.size, -1)
    k = _step_of(t, tt.size - 1)

    fr = Frame()
    fr.add("t_curve", tt[: k + 1], unit="s", label="积分节点 t")
    fr.add("y_curve", yy[: k + 1, 0], label="数值解")
    fr.add("y_exact_curve", exact_all[: k + 1, 0], label="精确解")
    err_curve = rk.pointwise_error(yy[: k + 1], exact_all[: k + 1])
    fr.add("error_curve", _floor(err_curve), unit="1", label="逐步误差 |y_h−y|（对数轴，地板 1e-17）")

    hh = np.asarray(lad.h, dtype=float)
    eg = _floor(lad.error_global)
    ee = _floor(lad.error_endpoint)
    el = _floor(lad.error_local)
    ed = _floor(lad.error_dense)
    fr.add("h_ladder", hh, unit="s", label="步长阶梯 h")
    fr.add("error_global", eg, unit="1", label="整段最大误差 max_t|y_h−y|")
    fr.add("error_endpoint", ee, unit="1", label="终点误差 |y_h(T)−y(T)|")
    fr.add("error_local", el, unit="1", label="首步真局部误差（6 子步参考）")
    fr.add("error_dense", ed, unit="1", label="步内插值误差（稠密输出，不含端点）")
    # 理论参考线：过当前 h 的点，斜率 −p（全局）与 −(p+1)（局部）
    i_h = int(np.argmin(np.abs(hh - h)))
    ref_g = eg[i_h] * (hh / hh[i_h]) ** (-float(tab.order))
    ref_l = el[i_h] * (hh / hh[i_h]) ** (-float(tab.order + 1))
    fr.add("theory_global", _floor(ref_g), unit="1", label=f"理论斜率 −{tab.order}（过当前 h）")
    fr.add("theory_local", _floor(ref_l), unit="1", label=f"理论斜率 −{tab.order + 1}（过当前 h）")
    fr.add("mark_h", np.array([hh[i_h]]), unit="s", label="当前 h")
    fr.add("mark_error_global", np.array([eg[i_h]]), unit="1", label="当前 h 的整段最大误差")
    fr.add("mark_error_endpoint", np.array([ee[i_h]]), unit="1", label="当前 h 的终点误差")

    slope_g, drop_h, drop_t = _asym_slope(lad.h, lad.error_global, float(tab.order))
    slope_l, _, _ = _asym_slope(lad.h, lad.error_local, float(tab.order + 1))
    slope_d, _, _ = _asym_slope(lad.h, lad.error_dense, float(min(tab.order, 4)))
    ratio_end = float(eg[i_h] / ee[i_h])
    fr.set_meta(
        declared_order=tab.order,
        order_hat=tab.order_hat if tab.b_hat is not None else 0,
        is_pair=bool(tab.b_hat is not None),
        fsal=bool(tab.fsal),
        stages=tab.stages,
        h=h,
        t_now=float(tt[k]),
        t_end=t_end,
        n_steps_done=k,
        n_steps_total=int(tt.size - 1),
        max_error_so_far=_fin(err_curve.max()),
        endpoint_error_so_far=_fin(err_curve[-1]),
        ratio_max_over_endpoint=_fin(ratio_end),
        measured_slope_global=_fin(slope_g),
        measured_slope_local=_fin(slope_l),
        measured_slope_dense=_fin(slope_d),
        slope_drop_head=drop_h,
        slope_drop_tail=drop_t,
        work_per_time=_fin(float(lad.work[i_h])),
        dense_extra_per_step=0.0 if tab.fsal else 1.0,
        err_floor=_ERR_FLOOR,
    )
    fr.set_strings(
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, big=_BIG_FILL),
        time_semantics=_TIME_SEMANTICS["order_convergence"],
        method_label=tab.label,
        problem_label=prob.label,
        at_h=f"h={h:g}：整段最大 {eg[i_h]:.3e}、终点 {ee[i_h]:.3e}（比 {ratio_end:.1f}）、局部 {el[i_h]:.3e}、插值 {ed[i_h]:.3e}",
        slope_note=f"拟合窗口丢弃：头 {drop_h} 点、尾 {drop_t} 点（规则见 rk.asymptotic_window）",
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 2：自适应步长
# --------------------------------------------------------------------------- #
def _adaptive_step(params: Mapping[str, Any], t: float) -> Frame:
    method = str(params["method"])
    problem = str(params["problem"])
    tol = float(params["tol"])
    law = str(params["law"])
    extrap = bool(params["extrap"])
    prob = rk.problem(problem)
    t_end = float(prob.t_end)
    run = _adapt(problem, method, tol, law, extrap, t_end)

    n_att = int(run.h_attempt.size)
    k = _step_of(float(t) / 100.0 * max(n_att - 1, 0), max(n_att - 1, 0))
    sl = slice(0, k + 1)
    hh = np.asarray(run.h_attempt, dtype=float)[sl]
    eps = np.asarray(run.error_attempt, dtype=float)[sl]
    tru = np.asarray(run.error_true_attempt, dtype=float)[sl]
    acc = np.asarray(run.accepted, dtype=float)[sl]
    t_att = np.asarray(run.t_attempt, dtype=float)[sl]

    tru_disp, tru_clip = _clip_big(tru)
    # 解曲线：只画到第 k 次试步对应的时刻
    t_left = float(t_att[-1]) if t_att.size else 0.0
    ts = np.asarray(run.t, dtype=float)
    ys = np.asarray(run.y, dtype=float).reshape(ts.size, -1)
    n_acc_sofar = int(np.count_nonzero(acc))
    idx = min(n_acc_sofar, ts.size - 1)
    exact_sol = np.asarray(prob.exact(ts[: idx + 1]), dtype=float).reshape(idx + 1, -1)

    fr = Frame()
    fr.add("t_sol", ts[: idx + 1], unit="s", label="被接受的节点 t")
    fr.add("y_sol", ys[: idx + 1, 0], label="自适应数值解")
    fr.add("y_exact_sol", exact_sol[:, 0], label="精确解")
    fr.add("attempt_index", np.arange(k + 1, dtype=float), label="试步序号")
    fr.add("t_attempt", t_att, unit="s", label="试步起点的 t")
    fr.add("h_attempt", _floor(hh, 1e-16), unit="s", label="本次试步的 h（对数）")
    fr.add("error_estimate", _floor(eps, 1e-16), unit="1", label="归一化误差估计 ε（1 = 容差）")
    fr.add("error_true", _floor(tru_disp, 1e-16), unit="1", label="同一尺度下的真局部误差（6 子步参考）")
    fr.add("error_true_clipped", tru_clip, unit="1", label="1 = 真误差超过显示上限（按上限占位）")
    fr.add("accepted", acc, unit="1", label="1 = 该步被接受，0 = 被拒绝")
    fr.add("threshold", np.ones(k + 1), unit="1", label="接受阈值 ε=1")

    eps_acc = eps[acc > 0.5]
    ratio_med = float("nan")
    if eps_acc.size and np.any(np.isfinite(tru) & (tru > 0)):
        m = (acc > 0.5) & np.isfinite(tru) & (tru > 0)
        if np.any(m):
            ratio_med = float(np.median(eps[m] / tru[m]))
    h_acc = np.asarray(run.h_accepted(), dtype=float) if run.n_accepted else np.array([float("nan")])
    h_star = rk.critical_step_for_decay(method, -1000.0) if problem == "stiff_cos" else float("nan")
    fr.set_meta(
        tol=tol,
        law_is_pi=bool(law == "pi"),
        local_extrapolation=extrap,
        order_propagated=run.order_propagated,
        order_estimate=run.order_estimate,
        n_attempts=n_att,
        attempt_index=k,
        progress_percent=float(t) if math.isfinite(float(t)) else 0.0,
        n_accepted=run.n_accepted,
        n_rejected=run.n_rejected,
        accept_ratio=_fin(run.accept_ratio()),
        n_f_evals=run.n_f_evals,
        work_per_time=_fin(run.work_per_time()),
        max_eps=_fin(np.max(eps)) if eps.size else 0.0,
        max_error=_fin(run.max_error),
        endpoint_error=_fin(run.endpoint_error),
        err_over_tol=_fin(run.max_error / tol),
        h_min=_fin(np.min(h_acc)) if h_acc.size else 0.0,
        h_max=_fin(np.max(h_acc)) if h_acc.size else 0.0,
        h_median=_fin(np.median(h_acc)) if h_acc.size else 0.0,
        h_stability_limit=_fin(h_star),
        h_over_stability=_fin(float(np.median(h_acc)) / h_star) if math.isfinite(h_star) else 0.0,
        estimator_ratio_median=_fin(ratio_med, -1.0),
        t_left=t_left,
        ok=bool(run.ok),
        blowup=bool(run.blowup),
        t_blowup=_fin(run.t_blowup),
        err_floor=_ERR_FLOOR,
    )
    fr.set_strings(
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, big=_BIG_FILL),
        time_semantics=_TIME_SEMANTICS["adaptive_step"],
        method=method,
        problem=problem,
        method_label=rk.tableau(method).pair_label,
        problem_label=prob.label,
        ext_note=(
            f"推进的是 {'高阶解 b（局部外推开）' if extrap else '低阶解 b_hat（局部外推关）'}："
            f"实到误差/请求容差 = {_fin(run.max_error / tol):.2f}"
        ),
        stiff_note=(
            f"稳定极限 h*={h_star:.3e}（|R(x*)|=1 除以 |λ|），实测步长中位数是它的 "
            f"{_fin(float(np.median(h_acc)) / h_star):.2f} 倍"
            if problem == "stiff_cos"
            else "非刚性算例：步长由精度要求决定，不受稳定极限约束"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 3：阶条件
# --------------------------------------------------------------------------- #
def _order_conditions(params: Mapping[str, Any], t: float) -> Frame:
    method = str(params["method"])
    max_order = int(params["max_order"])
    use_hat = bool(params["use_hat"])
    tab = rk.tableau(method)
    # 非嵌入对方法（euler/midpoint/rk3/rk4）没有 b_hat：勾选框不该把整帧打掉，
    # 回退到 b 并在 meta/strings 里**明确**写出来（静默回退才是防误导规则 4 的靶子）。
    use_hat_eff = bool(use_hat and tab.b_hat is not None)
    oc = rk.order_conditions(method, max_order=max_order, use_hat=use_hat_eff)

    pack = _tree_pack(max_order)
    n_cond = len(oc.labels)
    k = _step_of(t, n_cond - 1)
    lit = (np.arange(n_cond) <= k).astype(float)
    resid = np.abs(np.asarray(oc.residuals, dtype=float))

    fr = Frame()
    fr.add("node_x", pack["node_x"], label="树节点 x（单位框）")
    fr.add("node_y", pack["node_y"], label="树节点 y（单位框）")
    fr.add("node_tree", pack["node_tree"], label="节点所属树的序号")
    fr.add("edge_x0", pack["edge_x0"], label="边的父节点 x")
    fr.add("edge_y0", pack["edge_y0"], label="边的父节点 y")
    fr.add("edge_x1", pack["edge_x1"], label="边的子节点 x")
    fr.add("edge_y1", pack["edge_y1"], label="边的子节点 y")
    fr.add("edge_tree", pack["edge_tree"], label="边所属树的序号")
    fr.add("tree_cx", pack["tree_cx"], label="树中心 x（放标签）")
    fr.add("tree_cy", pack["tree_cy"], label="树中心 y")
    fr.add("cond_index", np.arange(n_cond, dtype=float), label="条件序号")
    fr.add("cond_tree", np.arange(n_cond, dtype=float), label="对应树的序号")
    fr.add("cond_order", np.asarray(oc.orders, dtype=float), label="树的阶 ρ")
    fr.add("cond_density", np.asarray(oc.densities, dtype=float), label="密度 γ")
    fr.add("cond_phi", np.asarray(oc.values, dtype=float), label="初等权 Φ")
    fr.add("cond_residual", _floor(resid), label="残差 |1−γΦ|（对数，地板 1e-17）")
    fr.add("cond_pass", np.asarray(oc.passing_mask(), dtype=float), label="1 = 该条满足")
    fr.add("cond_lit", lit, label="1 = 已扫到（时间轴）")

    worst = int(np.argmax(resid)) if n_cond else 0
    fr.set_meta(
        declared_order=oc.declared_order,
        measured_order=oc.measured_order,
        order_hat=tab.order_hat if tab.b_hat is not None else 0,
        b_hat_available=bool(tab.b_hat is not None),
        using_hat=use_hat_eff,
        hat_fallback=bool(use_hat and not use_hat_eff),
        n_conditions=n_cond,
        n_failed=oc.n_failed,
        n_passed=n_cond - oc.n_failed,
        max_order=max_order,
        cond_index=k,
        worst_residual=_fin(resid[worst]) if n_cond else 0.0,
        worst_index=worst,
        tol_used=1e-10,
        err_floor=_ERR_FLOOR,
    )
    fr.set_strings(
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, big=_BIG_FILL),
        time_semantics=_TIME_SEMANTICS["order_conditions"],
        method_label=tab.label,
        weight_note=(
            "用的是 b_hat（嵌入对里的低阶权）"
            if use_hat_eff
            else (
                "用的是 b（被推进的权）"
                + ("；该方法没有 b_hat，勾选框已回退到 b" if use_hat else "")
            )
        ),
        # 树标签用 | 连接：前端按 | 切开即可与 cond_* 数组一一对应
        tree_labels="|".join(rk.tree_label(x) for x in rk.rooted_trees(max_order)),
        worst_label=oc.labels[worst] if n_cond else "",
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：工作-精度前沿
# --------------------------------------------------------------------------- #
_WP_TOLS = (1e-04, 1e-05, 1e-06, 1e-07, 1e-08)
_WP_METHODS = ("heun", "bs23", "rkf45", "dopri5")


@functools.lru_cache(maxsize=16)
def _wp(problem: str, extrap: bool) -> dict[str, ss.WorkPrecision]:
    return {
        m: ss.work_precision(m, problem, _WP_TOLS, local_extrapolation=bool(extrap))
        for m in _WP_METHODS
    }


@functools.lru_cache(maxsize=16)
def _fixed_wp(problem: str) -> rk.Ladder:
    return rk.error_ladder(problem, "rk4", hs=[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])


def _work_precision(params: Mapping[str, Any], t: float) -> Frame:
    problem = str(params["problem"])
    tol = float(params["tol"])
    extrap = bool(params["extrap"])
    wps = _wp(problem, extrap)
    lad = _fixed_wp(problem)
    tols = np.asarray(_WP_TOLS, dtype=float)

    fr = Frame()
    fr.add("tol_grid", tols, unit="1", label="请求容差 tol（对数）")
    fr.add("tol_line", tols, unit="1", label="参考线 y=tol")
    for m in _WP_METHODS:
        wp = wps[m]
        fr.add(f"error_{m}", _floor(wp.error_max), unit="1", label=f"{m} 实到最大误差")
        fr.add(f"work_{m}", np.maximum(np.asarray(wp.work, dtype=float), 1e-3), unit="1", label=f"{m} 工作量（每单位时间求值次数）")
    fr.add("error_fixed_rk4", _floor(lad.error_global), unit="1", label="固定步长 RK4 的误差（由阶梯反查）")
    fr.add("work_fixed_rk4", np.asarray(lad.work, dtype=float), unit="1", label="固定步长 RK4 的工作量")

    # 当前 tol 处：四个方法的实到误差与工作量（线性插值到 tol 上，只在实测区间内）
    cur_err = {}
    cur_work = {}
    for m in _WP_METHODS:
        wp = wps[m]
        cur_err[m] = float(wp.error_max[int(np.argmin(np.abs(tols - tol)))])
        cur_work[m] = float(wp.work[int(np.argmin(np.abs(tols - tol)))])
    fr.add("mark_tol", np.array([tol]), unit="1", label="当前 tol")
    fr.add("mark_error", np.asarray([cur_err[m] for m in _WP_METHODS]), unit="1", label="当前 tol 处各方法的实到误差")
    fr.add("mark_work", np.asarray([cur_work[m] for m in _WP_METHODS]), unit="1", label="当前 tol 处各方法的工作量")

    errs = np.asarray([cur_err[m] for m in _WP_METHODS], dtype=float)
    works = np.asarray([cur_work[m] for m in _WP_METHODS], dtype=float)
    target = float(np.max(errs))
    # 反查只做插值、不外推：目标精度落在某个方法的实测区间之外时**拒绝作答**（-1 + 掩码），
    # 而不是硬编一个数出来。实测触发场景：extrap=False 时 heun 在 tol=1e-6 上实到 3.9e-4，
    # dopri5 的实测区间上界只有 1.1e-5，要它「达到 3.9e-4」等于要它用比 1e-4 更松的容差。
    at_target = np.full(len(_WP_METHODS), -1.0)
    at_target_valid = np.zeros(len(_WP_METHODS))
    for i, m in enumerate(_WP_METHODS):
        try:
            at_target[i] = float(wps[m].work_at(target))
            at_target_valid[i] = 1.0
        except ValueError:
            at_target[i] = -1.0
    ok = at_target_valid > 0.5
    at_ok = at_target[ok] if np.any(ok) else np.asarray([float("nan")])
    fr.add("work_at_target", at_target, unit="1", label="达到同一精度所需工作量（-1 = 超出实测区间，拒绝外推）")
    fr.add("work_at_target_valid", at_target_valid, unit="1", label="1 = 该方法的反查点在实测区间内")
    fr.set_meta(
        tol=tol,
        local_extrapolation=extrap,
        n_methods=len(_WP_METHODS),
        error_spread_at_tol=_fin(float(np.max(errs) / np.min(errs))),
        work_spread_at_tol=_fin(float(np.max(works) / np.min(works))),
        target_error=target,
        best_error=_fin(float(np.min(errs))),
        worst_error=_fin(float(np.max(errs))),
        cheapest_work=_fin(float(np.min(works))),
        dearest_work=_fin(float(np.max(works))),
        work_at_target_min=_fin(float(np.min(at_ok))),
        work_at_target_max=_fin(float(np.max(at_ok))),
        work_at_target_spread=_fin(float(np.max(at_ok) / np.min(at_ok))),
        n_work_at_target_unavailable=int(np.count_nonzero(~ok)),
        err_floor=_ERR_FLOOR,
    )
    fr.set_strings(
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, big=_BIG_FILL),
        time_semantics=_TIME_SEMANTICS["work_precision"],
        problem=problem,
        method_order="|".join(_WP_METHODS),
        at_tol="；".join(f"{m}: 误差/请求={cur_err[m] / tol:.2f}、工作量={cur_work[m]:.1f}" for m in _WP_METHODS),
        at_target="；".join(
            (f"{m}: {at_target[i]:.1f}" if ok[i] else f"{m}: 超出实测区间（拒绝外推）")
            for i, m in enumerate(_WP_METHODS)
        ),
        reading_note=(
            "左图看「说到做不到」：曲线在对角线上方 = 实到误差大于请求容差；"
            "右图看代价：x 越小、y 越低越好，四条曲线的下包络才是前沿"
        ),
        target_note=(
            f"工作量反查的目标精度 = 当前 tol 下最差方法的实到误差（{target:.3e}，即四个方法都达到的精度）；"
            "某个方法的反查值为 -1 表示它的实测容差区间没覆盖这个精度（对它来说这个精度太松），"
            "反查只做插值、拒绝外推"
        ),
    )
    return fr
