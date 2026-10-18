"""1.2 章：线性方程组直接法（LU / Cholesky / 稀疏排序）。

四个可视化的分工是"从数值到结构、再到策略"：

1. ``lu_heatmap`` —— 消元过程本身：每一步更新的子矩阵 + 增长因子与残差的历史曲线。
   参数里那个 ``dominance`` 是有理论临界值的：**严格对角占优（>1）是"不选主元也稳定"的
   充分条件**，滑块拖到 1 以下就能看见无主元消元开始爆炸（并且会撞上零主元）。
2. ``unit_ball`` —— 几何图像：单位圆在 :math:`A`、:math:`L`、:math:`U` 下分别变成什么形状，
   面积就是行列式；时间轴是 :math:`\\log_{10}\\kappa`，一路拖到 :math:`\\kappa u \\sim 1`，
   看得见"几何上是同一个分解，数值上解已经失真"。
3. ``spy_ordering`` —— 稀疏结构：消元即弦化，填充边**逐步长出来**；比较自然序 / RCM / 最小度。
4. ``pivot_strategy`` —— 三种主元策略在同一矩阵上的增长因子与前向/后向误差（柱状图）。

约定（与 0.0 章一致）：``meta`` 只放数字/布尔/``None``，字符串一律走 ``strings``；
时间轴对"离散步骤"型可视化就是**步骤编号**（``t`` 会被四舍五入到整数步并夹住），
``meta.step`` 给出真实用的那一步。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from numcore.linalg.chol import cholesky
from numcore.linalg.lu import (
    EXAMPLE_MATRICES,
    ROUNDOFF_HINT,
    backward_error,
    ball_matrix,
    condition_2x2,
    condition_estimate_1,
    dominant_matrix,
    exact_solve,
    example_matrix,
    growth_factor,
    leading_block_residuals,
    lu_decompose,
    lu_solve,
    residual_norm,
    singular_values_2x2,
    unit_ball_image,
)
from numcore.linalg.ordering import minimum_degree, natural_order, ordering_report, rcm
from numcore.linalg.sparse import symbolic_lu
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 时间轴上限（= 最大 n-1，见 ``lu_heatmap`` 的 n 上限 / ``spy_ordering`` 的 grid 上限）
_MAX_STEPS = 195

MANIFEST: dict[str, Any] = {
    "chapter": "1.2",
    "title": "线性方程组直接法",
    "part": "I",
    "summary": (
        "消元过程、单位球的形变、稀疏填充与主元策略：把「解 Ax=b」拆成几何、结构与"
        "舍入误差三件事，并让每件事都能被测量。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "lu_heatmap",
            "title": "消元过程热力图：增长因子与残差",
            "description": (
                "矩阵按 |·| 取 log10 上色；每推进一步，被消元的列变成乘子、右下角的子矩阵被更新。"
                "副轴同时画增长因子 ρ(k) 与部分因子的残差 ‖PA−LU‖/‖A‖。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "n",
                    "label": "矩阵规模 n",
                    "type": "int",
                    "min": 4,
                    "max": 16,
                    "default": 8,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "dominance",
                    "label": "对角占优比 |a_ii| / Σ|a_ij|",
                    "type": "float",
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.05,
                    "default": 1.6,
                    "critical": [1.0],
                    "criticalNote": (
                        "严格对角占优（比值 > 1）是「不选主元也稳定」的**充分条件**：此时 A 非奇异"
                        "且所有顺序主子式非零，乘子 |m_ik| < 1。比值 ≤ 1 时不保证出事，但已经进入"
                        "「可能遇到小主元」的区域——把它拖到 1 以下看 ρ 何时开始爆。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "pivot",
                    "label": "主元策略",
                    "type": "enum",
                    "options": ["partial", "none", "complete"],
                    "default": "partial",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 15.0, "dt": 1.0, "label": "消元步 k"},
            "requiredAux": ["residual"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "log10|M|（当前工作矩阵）",
                    "ylabel": "行 i",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "增长因子 ρ(k) 与残差",
                    "ylabel": "ρ 与 ‖PA−LU‖/‖A‖",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "热力图的颜色是 **log10|元素|**，所以「一片蓝色」并不代表矩阵接近零——1e-16 和 1e-2 "
                "看起来只是深浅不同。更重要的两条：其一，增长因子 ρ 大**不等于解不准**，它只把后向"
                "误差的界放大，真正的误差还要乘条件数 κ（去看 unit_ball 那一页）；其二，把 dominance "
                "调到 1 以下时图**可能**依然很漂亮，直到某一步主元小到 1e-18 才突然全糊——这就是"
                "「不稳定」的伪装：它不是慢慢变差，而是到某一步才崩。"
            ),
            "theory": [
                "PA = LU，部分主元保证 |m_ik| ≤ 1，增长因子 ρ = max|u_ij| / max|a_ij| ≤ 2^(n−1)",
                "后向误差界：‖δA‖_∞ ≤ c·n·ρ·u·‖A‖_∞，u = 2.2e-16 是单位舍入",
                "无主元消元可行的充要条件：所有顺序主子式非零；严格对角占优是它的充分条件",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "unit_ball",
            "title": "单位球的形变：行列式、条件数与解",
            "description": (
                "单位圆在 A（灰）、L（蓝）、U（橙）下分别变成什么形状；面积比值给出 |det|。"
                "时间轴是 log10 κ，副轴给出前向误差随 κ 的实测曲线与当前位置。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "shear",
                    "label": "剪切强度 s（A 的右上元）",
                    "type": "float",
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "default": 0.8,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "pivot",
                    "label": "主元策略",
                    "type": "enum",
                    "options": ["partial", "none"],
                    "default": "partial",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "b_angle",
                    "label": "右端项方向 b（度）",
                    "type": "float",
                    "min": 0.0,
                    "max": 360.0,
                    "step": 5.0,
                    "default": 30.0,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 17.0, "dt": 0.25, "label": "log10 κ"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "单位圆与它的三个像（固定坐标）",
                    "ylabel": "y",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "前向误差 ‖x̂−x‖/‖x‖ 随 κ",
                    "ylabel": "相对误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "三件事最容易被这张图骗到。第一：坐标轴**固定**在 ±3，κ 大时椭圆会直接冲出画面——"
                "看不到不等于不存在，溢出徽标会告诉你它去哪儿了；反过来若把轴设成自适应，"
                "一个 κ=1e16 的细长椭圆会被拉成和单位圆一样「圆」，条件数就被视觉抹平了。"
                "第二：单位圆像的面积是 |det A|，而 det 只告诉你「体积」，不告诉你「形状」——"
                "det = 1 的剪切矩阵把圆压成平行四边形，条件数并不小；反过来 det ≈ 0 的矩阵当然病态，"
                "但 det 很大的矩阵也可能病态。判据是奇异值之比，不是行列式。第三：想「量」出 σmin "
                "必须小心——在圆周上采样取最小值会**系统性高估** σmin（采样点越少越离谱，副轴那条"
                "误差曲线就是它的收敛速度），而 κ 超过 1/u ≈ 4.5e15 之后 A 在 float64 里已经**没有**"
                "可表示的小奇异值了：这时图上画的细椭圆本身就是幻觉，任何声称量到 κ=1e17 的数字都在撒谎。"
            ),
            "theory": [
                "x = A⁻¹b 的前向误差界：‖x̂−x‖/‖x‖ ≲ κ(A)·u（u = 2.2e-16），κ = σmax/σmin",
                "面积：|det A| = |det L|·|det U| = ∏|u_ii|（L 是单位下三角，det L = 1）",
                "1e-18 型小主元：无主元消元的乘子 1e18，误差界 ~1e2 倍地失真；部分主元把它换走即可",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "spy_ordering",
            "title": "稀疏 spy plot：填充是怎么长出来的",
            "description": (
                "二维五点 Laplacian；灰点是原矩阵的非零元，红点是**已经产生**的填充，"
                "浅灰是本次排序最终还会产生的填充。副轴给出三种排序的 LU 后向残差（应当都在 1e-16）。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "grid",
                    "label": "网格 k（矩阵规模 n=k²）",
                    "type": "int",
                    "min": 3,
                    "max": 14,
                    "default": 8,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "ordering",
                    "label": "消元顺序",
                    "type": "enum",
                    "options": ["natural", "rcm", "amd"],
                    "default": "rcm",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 195.0, "dt": 1.0, "label": "消元步 k"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "矩阵结构（行 0 在上）",
                    "ylabel": "行 i",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "各排序的 LU 后向残差",
                    "ylabel": "‖PA−LU‖_∞ / ‖A‖_∞",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "spy plot 只画**结构**，不画数值：一个位置上只要有非零元就是一个点，1e-30 和 1e30 长得"
                "一模一样。所以「填充少」不等于「算得快」——还要看每列的实际非零个数分布与访存模式；"
                "反过来，「填充多」也不等于「更不准」，排序只改变运算量与访存，不改变后向稳定性"
                "（副轴那条 ~1e-16 的水平线就是这件事的实测证据）。另外：最小度优化的是填充，"
                "它会把带宽搞得很差（自己拖到最后看带宽/剖面数字），而 RCM 优化带宽、填充只顺带改善；"
                "「哪个排序更好」取决于你要优化的是内存、是浮点运算量，还是并行度。"
            ),
            "theory": [
                "消元即弦化：消去 v 时把它的邻居连成团，新边 = 填充；填充只由结构与顺序决定，与数值无关",
                "nnz(L+U) = 2n + 2·Σ_k |later(k)|（结构对称、不选主元时精确）",
                "最小度的贪心界：nnz(L+U) 与消元过程中的最大团（树宽上界 +1）相关；RCM 最小化带宽与剖面",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "pivot_strategy",
            "title": "主元策略对比：增长因子与前向/后向误差",
            "description": (
                "同一个矩阵上跑无主元 / 部分主元 / 完全主元：主轴是 log10 ρ（柱状），"
                "副轴（对数）是解的前向误差与后向误差。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "matrix",
                    "label": "测试矩阵",
                    "type": "enum",
                    "options": [
                        "tiny_pivot",
                        "wilkinson",
                        "hilbert",
                        "spd",
                        "diag_dominant",
                        "banded",
                    ],
                    "default": "hilbert",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n",
                    "label": "规模 n（tiny_pivot 固定 2×2）",
                    "type": "int",
                    "min": 4,
                    "max": 40,
                    "default": 8,
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
                    "title": "log10 ρ（增长因子）",
                    "ylabel": "log10 ρ",
                },
                {
                    "id": "aux",
                    "autoscaled": True,
                    "title": "前向 / 后向误差（对数）",
                    "ylabel": "相对误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "这张图最容易得出错误结论：「部分主元 ρ 最小所以最好」。三根柱子里 ρ 只反映**后向**"
                "稳定性，而解准不准由 κ(A)·ρ·u 决定——Hilbert 矩阵上三种策略的后向误差都是 1e-16，"
                "可前向误差能差到 1e-7，因为 κ 有 1e7。反过来 tiny_pivot 上无主元 ρ 高达 1e18 而"
                "部分主元只有 1：这时 ρ 的差距**确实**直接体现成解的错误。还有一条：完全主元为了把 ρ "
                "压到最小，要多花 O(n²) 次比较与数据交换，在缓存友好的现代机器上通常**更慢**，"
                "所以工业实现普遍只用部分主元。"
            ),
            "theory": [
                "部分主元的增长因子上界 2^(n−1)，Wilkinson 矩阵取到等号（ρ = 2^(n−1)）",
                "后向误差 η = ‖Ax̂−b‖_∞/(‖A‖_∞‖x̂‖_∞+‖b‖_∞) ≈ u，与 κ 无关",
                "前向误差界 ‖x̂−x‖_∞/‖x‖_∞ ≲ κ_∞(A)·η；Cholesky 无主元也稳定（ρ ≤ 1）",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 计算入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "lu_heatmap":
        return _lu_heatmap(params, t)
    if viz_id == "unit_ball":
        return _unit_ball(params, t)
    if viz_id == "spy_ordering":
        return _spy_ordering(params, t)
    if viz_id == "pivot_strategy":
        return _pivot_strategy(params, t)
    raise KeyError(f"章节 1.2 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _finite_or_none(value: Any) -> float | None:
    """把 NaN/±Inf 换成 ``None``：NLF 头用 ``allow_nan=False`` 编码，塞进去会直接抛错。

    为什么不让它静默变成一个"很大/很小的数"：前端会把它当真实测量值画出来。
    "没有有限值"与"值很大"在图上必须是两件事。
    """
    v = float(value)
    return v if np.isfinite(v) else None


def _pivot_name(pivot: str) -> str:
    return {"partial": "部分主元", "none": "无主元", "complete": "完全主元"}.get(pivot, pivot)


def _step_of(t: float, n_steps: int) -> int:
    """把连续的时间轴四舍五入到整数消元步并夹到 [0, n_steps]。"""
    k = int(np.floor(float(t) + 0.5))
    return max(0, min(k, max(0, n_steps)))


# --------------------------------------------------------------------------- #
# viz 1：消元过程热力图
# --------------------------------------------------------------------------- #
def _lu_heatmap(params: Mapping[str, Any], t: float) -> Frame:
    n = int(params["n"])
    dominance = float(params["dominance"])
    pivot = str(params["pivot"])
    A = dominant_matrix(n, dominance)
    res = lu_decompose(A, pivot=pivot, history=True)
    n_steps = max(0, res.n - 1)
    k = _step_of(t, n_steps)
    hist = res.history if res.history else (A,)
    idx = min(k + 1, len(hist) - 1)
    M = np.array(hist[idx], dtype=float, copy=True)

    # 指示图：0 = 还没被触及（原始值），1 = 已经写回乘子的下三角列，2 = 当前主元行/列。
    # 渲染层据此画"扫过哪里"的散点与分隔线；数值本身在 matrix 里。
    indicator = np.zeros((n, n), dtype=float)
    for col in range(min(k, n)):
        rows = np.arange(col + 1, n)
        indicator[rows, col] = np.where(np.abs(M[rows, col]) > 0.0, 1.0, 0.0)
    if k < n:
        indicator[k, :] = np.maximum(indicator[k, :], 2.0)
        indicator[:, k] = np.maximum(indicator[:, k], 2.0)

    # 历史曲线：每一步的 ρ 与前导块残差（两者都是**逐步实测**，不是解析式）。
    # 残差经常**恰好**是 0（前导块上用同一套运算重建，舍入互相抵消），而 log 轴画不了 0；
    # 因此给它一个明确的地板 u/2，并在 meta 里如实报告"有几步是精确 0"。
    ROUNDOFF = ROUNDOFF_HINT  # u/2：log 轴上的显示地板（定义在 numcore.linalg.lu）
    rho_hist = np.zeros(len(hist), dtype=float)
    res_hist = np.zeros(len(hist), dtype=float)
    scale_a = float(np.max(np.abs(A))) if A.size else 0.0
    # 逐步残差走 numcore 的单一真源实现（需要 history=True；ref_history 与它同步记录）。
    # 定义与"为什么不能用最终置换"写在 numcore.linalg.lu.leading_block_residuals 的 docstring 里。
    steps = leading_block_residuals(res, relative=True)
    n_exact_zero = 0
    for j, snap in enumerate(hist):
        rho_hist[j] = growth_factor(A, np.triu(np.asarray(snap, dtype=float)))
        raw = steps[j] * (scale_a if scale_a > 0.0 else 1.0)
        if raw == 0.0:
            n_exact_zero += 1
        res_hist[j] = max(raw, ROUNDOFF)
    step_grid = np.arange(len(hist), dtype=float)

    fr = Frame(chapter="1.2", viz="lu_heatmap", kind="frame")
    fr.add("matrix", M, unit="1", label="当前工作矩阵（乘子在下三角）")
    fr.add("indicator", indicator, unit="1", label="区域标记 0/1/2")
    fr.add("rho_history", rho_hist, unit="1", label="增长因子 ρ(k)")
    fr.add("residual_history", res_hist, unit="1", label="前导块残差（0 以 u/2 显示）")
    # MANIFEST 里 requiredAux 声明的是 "residual"：那个名字必须是**帧里真实存在**的数组，
    # 否则前端的"必备对照量"面板会静默空着。这里给一个 1 元的当前值，方便模型直接标点。
    fr.add("residual", np.array([res_hist[idx]]), unit="1", label="当前步的前导块残差")
    fr.add("step_grid", step_grid, unit="1", label="消元步")
    fr.set_meta(
        n=n,
        step=k,
        n_steps=n_steps,
        dominance=dominance,
        rho=float(rho_hist[idx]),
        rho_bound=float(2.0 ** (n - 1)),
        residual=float(res_hist[idx]),
        residual_final=float(res_hist[-1]),
        singular=bool(res.singular),
        residual_exact_zero_steps=n_exact_zero,
        residual_floor=ROUNDOFF,
        history_index=int(idx),
        n_pivots=res.n_pivots,
        row_swaps=len(res.row_swaps),
        ops=res.ops,
        pivot_max_abs=float(np.max(np.abs(np.diag(res.U)))) if n else 0.0,
        pivot_min_abs=float(np.min(np.abs(np.diag(res.U)))) if n else 0.0,
    )
    fr.set_strings(pivot=_pivot_name(pivot), pivot_key=pivot, note=_lu_note(res, dominance, n))
    return fr


def _lu_note(res: Any, dominance: float, n: int) -> str:
    if res.singular:
        return f"遇到零主元：矩阵在给定策略下奇异（dominance={dominance:.2f}）"
    if dominance <= 1.0 and res.rho > 10.0:
        return f"非对角占优（比值 {dominance:.2f} ≤ 1）：增长因子已经到 {res.rho:.3g}"
    if res.rho > 10.0:
        return f"增长因子 {res.rho:.3g}（上界 2^(n−1)={2.0 ** (n - 1):.3g}）"
    return f"增长因子 {res.rho:.3g}：乘子都不大，消元很温和"


# --------------------------------------------------------------------------- #
# viz 2：单位球的形变
# --------------------------------------------------------------------------- #
def _forward_error_curve(
    shear: float, pivot: str, b_angle: float, n_grid: int = 61
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 :math:`\\log_{10}\\kappa \\in [0, 17]` 上实测前向误差曲线。

    每个 κ 都真跑一次 LU + 回代，与 :func:`numcore.linalg.lu.exact_solve` 的有理解对比。
    这是"实测值"，不是把 :math:`\\kappa u` 画出来——两者的差别正是断言要看的东西。

    返回 ``(exps, errs, valid)``：κ 大到让 A 在浮点下**真的奇异**时（``pivot="none"`` 时
    从很小的 κ 就开始），那里没有误差可测——``valid[i] = False``，``errs[i]`` 写 0 当占位符。
    **不写 NaN**：数组里出现 NaN 会被帧契约拒绝，前端拿到也会静默丢段；
    占位符 + 掩码 + 文案才是可读的"这里没有有限值"。
    """
    exps = np.linspace(0.0, 17.0, n_grid)
    errs = np.zeros(n_grid, dtype=float)
    valid = np.zeros(n_grid, dtype=float)
    th = np.deg2rad(b_angle)
    b = np.array([np.cos(th), np.sin(th)])
    for i, e in enumerate(exps):
        A = ball_matrix(shear, 10.0 ** float(e))
        try:
            x_exact = exact_solve(A, b)
            res = lu_decompose(A, pivot="partial" if pivot == "partial" else "none")
            x = lu_solve(res, b)
        except (ZeroDivisionError, ValueError):
            valid[i] = 0.0
            continue
        denom = float(np.max(np.abs(x_exact)))
        if denom <= 0.0:
            valid[i] = 0.0
            continue
        errs[i] = float(np.max(np.abs(x - x_exact))) / denom
        valid[i] = 1.0
    return exps, errs, valid


def _unit_ball(params: Mapping[str, Any], t: float) -> Frame:
    shear = float(params["shear"])
    pivot = str(params["pivot"])
    b_angle = float(params["b_angle"])
    log_kappa = max(0.0, min(17.0, float(t)))
    kappa = 10.0 ** log_kappa
    A = ball_matrix(shear, kappa)

    theta, ax, ay = unit_ball_image(A, n_theta=361)
    lres = lu_decompose(A, pivot="partial" if pivot == "partial" else "none")
    _, lx, ly = unit_ball_image(lres.L, n_theta=361)
    _, ux, uy = unit_ball_image(lres.U, n_theta=361)
    # 单位圆本身（灰色参照）
    _, cx, cy = unit_ball_image(np.eye(2), n_theta=361)

    th = np.deg2rad(b_angle)
    b = np.array([np.cos(th), np.sin(th)])
    x_exact = exact_solve(A, b)
    # κ ≳ 1/u 时 A 在浮点下**真的奇异**（U 出现零对角元），回代会抛 ZeroDivisionError。
    # 这不是程序出错，而是"这个问题在这个精度下不可解"——必须把它当成一种**要显示的
    # 状态**，而不是让整帧 500（真踩过：shear∈{0,2} 且 t≥16.5 时时间轴右端 8 个点直接崩）。
    singular = False
    try:
        x_num = lu_solve(lres, b)
    except (ZeroDivisionError, ValueError):
        singular = True
        # 奇异时 x̂ 没有有限值：数组里写 0 当占位符（写 NaN 会被帧契约拒绝），
        # 真实状态由 meta.singular 与 strings.status 给出。
        x_num = np.zeros(2, dtype=float)
    denom = float(np.max(np.abs(x_exact))) if np.max(np.abs(x_exact)) > 0.0 else 1.0
    fwd_err = float("nan") if singular else float(np.max(np.abs(x_num - x_exact))) / denom

    det_a = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])  # 2×2 行列式：自研，不调 np.linalg（P2）
    det_l = float(lres.L[0, 0] * lres.L[1, 1])
    det_u = float(lres.U[0, 0] * lres.U[1, 1])
    det_err = abs(det_a - det_l * det_u)

    exps, errs, err_valid = _forward_error_curve(shear, pivot, b_angle)
    # 当前位置在曲线上的标记（找最近的格点）
    j = int(np.argmin(np.abs(exps - log_kappa)))
    n_err_invalid = int(np.count_nonzero(err_valid == 0.0))
    first_invalid = float(exps[int(np.argmax(err_valid == 0.0))]) if n_err_invalid else None
    err_mark_valid = bool(err_valid[j] > 0.0)

    fr = Frame(chapter="1.2", viz="unit_ball", kind="frame")
    fr.add("theta", theta, unit="rad", label="参数角")
    fr.add("circle_x", cx, unit="1", label="单位圆")
    fr.add("circle_y", cy, unit="1", label="单位圆")
    fr.add("image_x", ax, unit="1", label="A 的像")
    fr.add("image_y", ay, unit="1", label="A 的像")
    fr.add("l_x", lx, unit="1", label="L 的像")
    fr.add("l_y", ly, unit="1", label="L 的像")
    fr.add("u_x", ux, unit="1", label="U 的像")
    fr.add("u_y", uy, unit="1", label="U 的像")
    fr.add("b_vec", np.array([0.0, b[0], 0.0, b[1]]), unit="1", label="b（两段折线）")
    fr.add("x_num", np.array([0.0, x_num[0], 0.0, x_num[1]]), unit="1", label="数值解 x̂")
    fr.add("x_exact", np.array([0.0, x_exact[0], 0.0, x_exact[1]]), unit="1", label="精确解 x")
    fr.add("kappa_grid", exps, unit="1", label="log10 κ")
    fr.add(
        "error",
        errs,
        unit="1",
        label="前向误差（实测）" if n_err_invalid == 0 else "前向误差（实测；无解处占位 0，见 error_valid）",
    )
    fr.add("error_valid", err_valid, unit="1", label="1=该 κ 有有限误差，0=浮点下不可解（占位 0）")
    fr.add(
        "error_mark",
        np.array([errs[j]], dtype=float),
        unit="1",
        label="当前位置的误差" if err_mark_valid else "当前位置的误差（不可解：占位 0）",
    )
    fr.add("x_num_valid", np.array([1.0 - float(singular)]), unit="1", label="1=x̂ 有有限值，0=奇异（占位 0）")
    fr.add("kappa_mark", np.array([exps[j]], dtype=float), unit="1", label="当前位置的 log10 κ")
    smin, smax = singular_values_2x2(A)
    fr.set_meta(
        shear=shear,
        log10_kappa=log_kappa,
        n_error_invalid=n_err_invalid,
        error_invalid_from_log10_kappa=first_invalid,
        error_mark_valid=err_mark_valid,
        x_num_valid=not singular,
        array_placeholder=0.0,
        kappa=kappa,
        # κ = σmax/σmin 在 σmin = 0 时是**无穷**：JSON 里没有 Infinity（allow_nan=False），
        # 硬塞 float("inf") 会让整帧编码失败。用 null 表示"无有限值"，文案里说清为什么。
        kappa_sigma=smax / smin if smin > 0.0 else None,
        singular=singular,
        lu_singular=bool(lres.singular),
        sigma_min=smin,
        sigma_max=smax,
        kappa_measured=_kappa_measured(A),
        det_a=det_a,
        det_l=det_l,
        det_u=det_u,
        det_error=det_err,
        # 奇异时这些量**没有有限值**：JSON 没有 NaN，写 NaN 会让整帧编码失败
        # （`Frame.validate()` 现在会当场拒绝）。用 null 表示"无有限值"，
        # 状态字符串里说清为什么；前端 `metaNumber` 拿到 undefined 会自行省略这一行。
        forward_error=_finite_or_none(fwd_err),
        backward_error=None if singular else _finite_or_none(backward_error(A, x_num, b)),
        residual=None if singular else _finite_or_none(residual_norm(A, x_num, b)),
        n_theta=361,
        # 规则 1/4：越界必须报。这里把**画出来的每一条曲线与向量**都算进去——
        # 单位圆的像在构造下恒在 ±3 内（σmax ≤ 2.24），真正冲出画面的是 x̂/x（可达 1e15）。
        # 注意用 np.any(...) 而不是 np.max(...) > 3：奇异时 x_num 是 NaN，np.max 会
        # 把整条比较传播成 False（NaN > 3 为假），于是"最该报警的那一帧"反而不报。
        overflow=bool(np.any(np.abs(np.concatenate([ax, ay, x_num, x_exact, b])) > 3.0)),
        x_num_absmax=None if singular else _finite_or_none(np.max(np.abs(x_num))),
        x_exact_absmax=float(np.max(np.abs(x_exact))),
    )
    fr.set_strings(
        pivot=_pivot_name(pivot),
        pivot_key=pivot,
        status="数值奇异：κ > 1/u，A 的浮点分解出现零主元" if singular else "",
        note=_ball_note(log_kappa, fwd_err, det_err, singular=singular),
        nonfinite_policy=(
            "这一帧里**没有 NaN**（帧契约要求所有数组有限）："
            "「这里没有有限值」一律写成占位符 0 + 掩码 + meta 标记，"
            "读图时必须先看掩码。"
        ),
        mask_note=(
            "error_valid[i]=0 表示该 κ 在浮点下不可解（LU 出现零主元），error[i] 的 0 是占位符不是误差；"
            f"实测前 {n_err_invalid} 个格点不可解"
            + (f"（从 log10κ = {first_invalid:.1f} 起）" if first_invalid is not None else "")
            + "。x_num_valid=0 时 x̂ 的两个分量同样是占位 0，不要读成「解在原点」。"
        ),
    )
    return fr


def _kappa_measured(A: np.ndarray, n_theta: int = 4001) -> float:
    """采样圆周量出 :math:`\\sigma_{\\max}/\\sigma_{\\min}`（不调用 SVD）。"""
    _, x, y = unit_ball_image(A, n_theta=n_theta)
    r = np.sqrt(x * x + y * y)
    lo = float(np.min(r))
    hi = float(np.max(r))
    return hi / lo if lo > 0.0 else float("inf")


def _ball_note(log_kappa: float, fwd_err: float, det_err: float, *, singular: bool = False) -> str:
    if singular:
        return (
            f"κ = 1e{log_kappa:.1f} > 1/u ≈ 4.5e15：A 在 float64 下已奇异，"
            "数值解不存在（精确解仍由有理数算出，可用来对照）"
        )
    if log_kappa > 16.0:
        return f"κu ≫ 1：解已经没有任何有效位（实测前向误差 {fwd_err:.1e}）"
    if fwd_err > 1e-8:
        return f"前向误差 {fwd_err:.1e}：与 κu ≈ {10 ** log_kappa * 2.2e-16:.1e} 同量级"
    return f"前向误差 {fwd_err:.1e}，分解的几何误差 |det A − det L·det U| = {det_err:.1e}"


# --------------------------------------------------------------------------- #
# viz 3：稀疏 spy plot 与排序
# --------------------------------------------------------------------------- #
def _laplacian_2d(k: int) -> np.ndarray:
    """k×k 网格上的五点 Laplacian（Dirichlet 边界，对角 4、邻接 −1）。"""
    n = k * k
    A = np.zeros((n, n), dtype=float)
    for i in range(k):
        for j in range(k):
            p = i * k + j
            A[p, p] = 4.0
            if i > 0:
                A[p, p - k] = -1.0
            if i < k - 1:
                A[p, p + k] = -1.0
            if j > 0:
                A[p, p - 1] = -1.0
            if j < k - 1:
                A[p, p + 1] = -1.0
    return A


def _order_for(A: np.ndarray, ordering: str) -> np.ndarray:
    if ordering == "natural":
        return natural_order(A)
    if ordering == "rcm":
        return rcm(A)
    if ordering == "amd":
        return minimum_degree(A)
    raise KeyError(f"未知排序 {ordering!r}")


#: 后向残差与排序无关，但每个 (grid, ordering) 都要真跑一次 LU——按 grid 记住结果
_RESIDUAL_CACHE: dict[int, tuple[list[str], np.ndarray]] = {}


def _residuals_by_ordering(A: np.ndarray, grid: int) -> tuple[list[str], np.ndarray]:
    names = ["natural", "rcm", "amd"]
    cached = _RESIDUAL_CACHE.get(grid)
    if cached is not None:
        return cached
    vals = np.zeros(len(names), dtype=float)
    scale = float(np.max(np.abs(A))) if A.size else 1.0
    for i, name in enumerate(names):
        perm = _order_for(A, name)
        Ap = A[np.ix_(perm, perm)]
        res = lu_decompose(Ap, pivot="partial")
        num = residual_norm(Ap, lu_solve(res, np.ones(A.shape[0])), np.ones(A.shape[0]))
        vals[i] = num / scale if scale > 0.0 else num
    out = (names, vals)
    if len(_RESIDUAL_CACHE) < 32:  # 有界缓存：只是别在拖动时重算
        _RESIDUAL_CACHE[grid] = out
    return out


def _spy_ordering(params: Mapping[str, Any], t: float) -> Frame:
    grid = int(params["grid"])
    ordering = str(params["ordering"])
    n = grid * grid
    A = _laplacian_2d(grid)
    perm = _order_for(A, ordering)
    Ap = A[np.ix_(perm, perm)]
    order_pos = np.empty(n, dtype=int)
    order_pos[perm] = np.arange(n, dtype=int)
    sy = symbolic_lu(A, perm)
    k = _step_of(t, n - 1)

    # 消元到第 k 步为止的填充集合：逐点重放符号消元的"每一步新增了哪些边"
    fill_by_step = _fill_by_step(Ap)
    done = np.concatenate([np.zeros(0, dtype=float)] + [fill_by_step[j] for j in range(min(k + 1, len(fill_by_step)))]) if fill_by_step else np.zeros(0)
    pending_list = [fill_by_step[j] for j in range(min(k + 1, len(fill_by_step)), len(fill_by_step))]
    pending = np.concatenate(pending_list) if pending_list else np.zeros(0)

    orig_rows, orig_cols = np.nonzero(np.abs(Ap) > 0)
    if done.size:
        done_rows, done_cols = done[0::2], done[1::2]
    else:
        done_rows = np.zeros(0, dtype=float)
        done_cols = np.zeros(0, dtype=float)
    if pending.size:
        pend_rows, pend_cols = pending[0::2], pending[1::2]
    else:
        pend_rows = np.zeros(0, dtype=float)
        pend_cols = np.zeros(0, dtype=float)

    names, resid = _residuals_by_ordering(A, grid)
    report = ordering_report(A, perm, ordering)

    fr = Frame(chapter="1.2", viz="spy_ordering", kind="frame")
    fr.add("orig_rows", orig_rows.astype(float), unit="1", label="原矩阵非零（行）")
    fr.add("orig_cols", orig_cols.astype(float), unit="1", label="原矩阵非零（列）")
    fr.add("fill_rows", done_rows.astype(float), unit="1", label="已产生填充（行）")
    fr.add("fill_cols", done_cols.astype(float), unit="1", label="已产生填充（列）")
    fr.add("pending_rows", pend_rows.astype(float), unit="1", label="后续填充（行）")
    fr.add("pending_cols", pend_cols.astype(float), unit="1", label="后续填充（列）")
    fr.add("strategies_x", np.arange(len(names), dtype=float), unit="1", label="排序序号")
    fr.add("error", resid, unit="1", label="各排序的 LU 后向残差")
    fr.add("error_mark", np.array([resid[names.index(ordering)]], dtype=float), unit="1", label="当前排序的残差")
    fr.add("error_mark_x", np.array([float(names.index(ordering))], dtype=float), unit="1", label="当前排序序号")
    fr.set_meta(
        grid=grid,
        n=n,
        step=k,
        n_steps=n - 1,
        nnz_original=int(orig_rows.size),
        nnz_total=sy.nnz_lu,
        fill_edges_total=sy.fill_edges,
        fill_edges_so_far=int(done_rows.size),
        fill_ratio=float(sy.nnz_lu) / float(orig_rows.size) if orig_rows.size else 0.0,
        bandwidth=report.bandwidth,
        profile=report.profile,
        max_clique=sy.max_clique,
        treewidth_bound=sy.treewidth_upper_bound,
        residual_current=float(resid[names.index(ordering)]),
        overflow=False,
    )
    fr.set_strings(
        ordering=ordering,
        ordering_name=_ordering_name(ordering),
        note=_spy_note(ordering, sy, report),
    )
    return fr


def _fill_by_step(Ap: np.ndarray) -> list[np.ndarray]:
    """逐步重放符号消元，返回"第 j 步新增的填充边"坐标（行,列 交替存放）。"""
    n = Ap.shape[0]
    adj = (np.abs(Ap) > 0).copy()
    np.fill_diagonal(adj, True)
    adj = adj | adj.T
    np.fill_diagonal(adj, False)
    alive = np.ones(n, dtype=bool)
    out: list[np.ndarray] = []
    for k in range(n):
        nbrs = np.flatnonzero(adj[k] & alive)
        nbrs = nbrs[nbrs != k]
        new: list[float] = []
        for a_i in range(nbrs.size):
            for b_i in range(a_i + 1, nbrs.size):
                u, v = int(nbrs[a_i]), int(nbrs[b_i])
                if not adj[u, v]:
                    adj[u, v] = True
                    adj[v, u] = True
                    new.extend([float(u), float(v)])
        out.append(np.asarray(new, dtype=float))
        alive[k] = False
        adj[k, :] = False
        adj[:, k] = False
    return out


def _ordering_name(ordering: str) -> str:
    return {"natural": "自然序", "rcm": "RCM（逆 Cuthill–McKee）", "amd": "最小度（AMD 简化版）"}.get(
        ordering, ordering
    )


def _spy_note(ordering: str, sy: Any, report: Any) -> str:
    return (
        f"{_ordering_name(ordering)}：nnz(L+U)={sy.nnz_lu}，填充边 {sy.fill_edges}，"
        f"带宽 {report.bandwidth}，剖面 {report.profile}，最大团 {sy.max_clique}（树宽上界 {sy.treewidth_upper_bound}）"
    )


# --------------------------------------------------------------------------- #
# viz 4：主元策略对比
# --------------------------------------------------------------------------- #
def _strategy_stats(A: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """对三种策略分别算 ρ、前向误差、后向误差；失败（奇异）记 NaN 并在 meta 里点出来。"""
    out: dict[str, Any] = {"rho": [], "fwd": [], "bwd": [], "ok": []}
    try:
        x_exact = exact_solve(A, b)
    except ZeroDivisionError:
        x_exact = None
    for strategy in ("none", "partial", "complete"):
        try:
            res = lu_decompose(A, pivot=strategy)
            x = lu_solve(res, b)
            out["rho"].append(float(res.rho))
            out["bwd"].append(float(backward_error(A, x, b)))
            if x_exact is None:
                out["fwd"].append(float("nan"))
            else:
                denom = float(np.max(np.abs(x_exact)))
                out["fwd"].append(float(np.max(np.abs(x - x_exact))) / denom if denom > 0.0 else float("nan"))
            out["ok"].append(not res.singular)
        except (ZeroDivisionError, ValueError):
            out["rho"].append(float("nan"))
            out["fwd"].append(float("nan"))
            out["bwd"].append(float("nan"))
            out["ok"].append(False)
    return out


def _pivot_strategy(params: Mapping[str, Any], t: float) -> Frame:
    name = str(params["matrix"])
    n = int(params["n"])
    A = example_matrix(name, 2 if name == "tiny_pivot" else n)
    m = A.shape[0]
    # 右端项取"一般位置"的 b：这是有讲究的。若取 b = 全 1，tiny_pivot 的精确解恰好是 [0,1]，
    # 无主元消元居然算对了（前向误差 0）——反例就失效了。b 的分量必须落在"会被小主元
    # 放大的方向"上，才能看见不稳定：见 chapters/1.2/pitfalls.md 第 3 条。
    b = np.linspace(1.0, 2.0, m)
    stats = _strategy_stats(A, b)
    rho = np.asarray(stats["rho"], dtype=float)
    fwd = np.asarray(stats["fwd"], dtype=float)
    bwd = np.asarray(stats["bwd"], dtype=float)
    with np.errstate(divide="ignore"):
        rho_log = np.where(rho > 0, np.log10(rho), np.nan)

    x_pos = np.arange(3, dtype=float)
    fr = Frame(chapter="1.2", viz="pivot_strategy", kind="frame")
    fr.add("strategies_x", x_pos, unit="1", label="策略序号（无 / 部分 / 完全）")
    fr.add("rho_log10", rho_log, unit="1", label="log10 增长因子")
    fr.add("rho", rho, unit="1", label="增长因子 ρ")
    fr.add("error", fwd, unit="1", label="前向误差")
    fr.add("error_backward", bwd, unit="1", label="后向误差")
    fr.set_meta(
        n=m,
        matrix_size=m,
        kappa_1=condition_estimate_1(A) if m <= 64 else float("nan"),
        rho_bound=float(2.0 ** (m - 1)) if m > 0 else 0.0,
        # 键名必须与数值同义。曾经的 `rho_max` 装的是 max(log10 ρ)（20 倍偏差量级），
        # 前端只能自己在文案里写"max log10 ρ"来兜着——那是把注释当修正用。现在：
        # rho_max = 真正的最大增长因子；rho_log10_max = 它的对数（图上的柱高）。
        # 三者都可能"没有任何有限值"（三种策略全奇异）：JSON 没有 NaN/Inf，
        # 所以用 None 表示"无有限值"，前端 metaNumber 会拿到 undefined 并自行降级。
        rho_max=_finite_or_none(np.nanmax(rho)) if np.any(np.isfinite(rho)) else None,
        rho_log10_max=_finite_or_none(np.nanmax(rho_log)) if np.any(np.isfinite(rho_log)) else None,
        forward_max=_finite_or_none(np.nanmax(fwd)) if np.any(np.isfinite(fwd)) else None,
        backward_max=_finite_or_none(np.nanmax(bwd)) if np.any(np.isfinite(bwd)) else None,
        n_singular=int(sum(1 for ok in stats["ok"] if not ok)),
        cholesky_ok=bool(cholesky(0.5 * (A + A.T)).ok) if m <= 64 else False,
        unit_roundoff=float(np.finfo(float).eps / 2.0),
        overflow=bool(not np.all(np.isfinite(rho_log))),
    )
    fr.set_strings(
        matrix=name,
        matrix_name=_matrix_name(name),
        strategy="none / partial / complete",
        note=_pivot_note(name, stats),
    )
    return fr


def _matrix_name(name: str) -> str:
    return {
        "tiny_pivot": "小主元反例 [[1e-18,1],[1,1]]",
        "wilkinson": "Wilkinson 增长矩阵（ρ = 2^(n−1)）",
        "hilbert": "Hilbert 矩阵（κ 指数增长）",
        "spd": "构造性对称正定（BᵀB + nI）",
        "diag_dominant": "严格对角占优",
        "banded": "三对角（带状）",
    }.get(name, EXAMPLE_MATRICES.get(name, name))


def _pivot_note(name: str, stats: Mapping[str, Any]) -> str:
    rho = [r for r in stats["rho"]]
    n_sing = sum(1 for ok in stats["ok"] if not ok)
    if n_sing:
        return f"{n_sing} 种策略在这矩阵上直接失败（零主元），柱子里留空"
    return (
        f"ρ: 无主元 {rho[0]:.3g} / 部分 {rho[1]:.3g} / 完全 {rho[2]:.3g}；"
        f"后向误差 {stats['bwd'][1]:.1e}（部分主元）"
    )
