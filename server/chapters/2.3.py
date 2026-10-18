"""2.3 章：正交多项式与函数逼近（投影几何 / 收敛率 / 交错点 / 零点演化）。

四个可视化各守一个**可测量的断言**，而不是让图“看起来对”：

1. ``projection`` —— 函数空间投影：把 :math:`f` 与它的两项投影画在一起
   （当前 k 维 vs 目标 n 维），并把**系数向量**与**Gram 矩阵**一起端出来。
   参数 ``basis`` 有意给出 ``monomial`` 这一项：三个基张成**同一个子空间**，
   投影（作为函数）相同，但正规方程的条件数从 1 涨到 :math:`10^{17}`——
   “换个基就不一样了”这句话是本章最容易骗人的地方。
2. ``convergence_rates`` —— 解析（Bernstein 椭圆有限 → 几何收敛）、尖点（代数
   :math:`n^{-3/2}`）、跳点（代数 :math:`n^{-1/2}`）三条误差曲线同图，
   每条都同时画**实测拟合线**与**理论参考线**（防误导规则 2），并把 :math:`R^2`
   与拟合窗口写进 meta；误差取 :math:`L^2` 投影误差，非光滑点处**切段求积**。
3. ``equioscillation`` —— Remez 交换算法的逐轮误差曲线 + 交错点闪烁标记，
   并叠加同阶 :math:`L^2` 投影的误差曲线做对照（“最小二乘”不是“最小最大”）。
4. ``zeros_evolution`` —— 零点分布随阶数的演化（分岔图风格）：
   同一阶的 Chebyshev 零点（解析闭式）仿射映射后作为对照，外加理论密度曲线，
   让“零点不是等距的”“两族的分布只差一点点”这两件事都能被看见。

约定：``meta`` 只放**有限标量**（没有有限值就写 ``None``，并在 ``strings`` 里说明；
NLF 头用 ``allow_nan=False`` 编码，写 NaN 会让整帧编码失败）。所有 log 轴的误差曲线
都先按 :data:`numcore.interp.orthopoly.ROUNDOFF` 截断，并把截断点数写进 meta。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np

from numcore.interp.approx import (
    BASES,
    clenshaw,
    composite_gauss_legendre,
    fit_loglog_slope,
    fit_semilog_slope,
    l2_projection,
    projection_error_curve,
    remez,
    test_function,
    theory_algebraic_rate,
)
from numcore.interp.orthopoly import (
    ROUNDOFF,
    chebyshev_roots,
    newton_roots_family,
)
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 曲线采样点数（够画平滑曲线，又不至于让帧过大）。
_GRID = 401

#: 时间轴上限（各 viz 自己再按参数夹一次）。
_MAX_STEPS = 40


MANIFEST: dict[str, Any] = {
    "chapter": "2.3",
    "title": "正交多项式与函数逼近",
    "part": "II",
    "summary": (
        "正交多项式的构造与零点、最小二乘投影的几何、Remez 最优一致逼近，"
        "以及“误差衰减率由光滑性决定”这句话的实测版本："
        "每条收敛曲线都同时给出实测斜率与理论斜率。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "projection",
            "title": "函数空间投影：把函数画成向量",
            "description": (
                "把 f 与它在前 k 个基函数张成的子空间上的投影画在一起（实线=当前 k 维，"
                "虚线=目标 n 维），下方同时给出系数向量 c 与误差 f − P_k f。"
                "切换基底时子空间不变、投影不变，但 Gram 矩阵的条件数与系数向量会完全变样。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "basis",
                    "label": "基底",
                    "type": "enum",
                    "options": ["legendre", "chebyshev", "monomial"],
                    "default": "legendre",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "func",
                    "label": "被逼近的函数 f",
                    "type": "enum",
                    "options": ["pole", "entire", "cusp", "jump"],
                    "default": "pole",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n",
                    "label": "目标维数 n",
                    "type": "int",
                    "min": 2,
                    "max": 24,
                    "default": 12,
                    "critical": [22.0, 23.0, 24.0],
                    "criticalNote": (
                        "这条线只对**单项式基**成立：Gram 矩阵的条件数 κ₁(G) 随 n 指数增长，"
                        "实测 n=22 时 κ≈4.3e15、n=24 时 κ≈2.0e17，在 n≈23 处越过 1/u≈9.0e15。"
                        "越过它以后解的相对误差可以到 O(1)，但正规方程残差依旧很小——"
                        "残差小绝不等于解对。Legendre 基（κ=1）与 Chebyshev 基不存在这条线。"
                    ),
                    "beyond": "above",
                },
            ],
            "time": {"mode": "scrub", "tMax": 24.0, "dt": 1.0, "label": "已展开的基函数个数 k"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "f 与两种投影", "ylabel": "y"},
                {"id": "coef", "autoscaled": True, "title": "投影系数（函数在这组基下的坐标）", "ylabel": "c_k"},
                {"id": "err", "autoscaled": True, "title": "误差 f − P_k f", "ylabel": "误差"},
            ],
            "pitfalls": (
                "① 把 L² 最佳逼近当成一致逼近：正交投影最小化的是加权平方误差，"
                "它在个别点上的偏差可以远大于 Remez 结果（本页误差曲线的尖峰就是证据）。"
                "② 以为“换个基结果就不同”：三个基底张成同一个子空间，投影作为函数是同一个，"
                "差别只在系数与条件数；反过来，把“残差小”当成“系数准”也会翻车——"
                "单项式基在 n≥23 时 Gram 的条件数越过 1/u，系数已经全是噪声而残差仍然很小。"
            ),
            "theory": [
                "投影的几何：Gc=b，G_ij=⟨φ_i,φ_j⟩，c 是 f 在正交基下的坐标（笔记定理 6）。",
                "正交基下 G=I，正规方程的相对误差 ~u·κ(G)；非正交基把 κ 平方放大（笔记定理 6 推论）。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "convergence_rates",
            "title": "收敛率：解析 / 尖点 / 跳点",
            "description": (
                "三条 L² 投影误差曲线画在同一张双对数图上：解析函数（Runge，Bernstein 椭圆有限）"
                "呈几何衰减，|x| 尖点呈 n^(−3/2)，sign(x) 跳点呈 n^(−1/2)。"
                "每条曲线都同时给出实测拟合线与理论参考线，并把斜率、R²、拟合窗口写进 meta。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "scale",
                    "label": "解析函数参数 s（1/(1+s x²)）",
                    "type": "float",
                    "min": 4.0,
                    "max": 200.0,
                    "step": 1.0,
                    "default": 25.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_max",
                    "label": "最大维数",
                    "type": "int",
                    "min": 16,
                    "max": 56,
                    "default": 40,
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
                    "title": "L² 误差 vs 维数（双对数）",
                    "ylabel": "‖f − P_n f‖",
                    "scale": "log",
                },
                {
                    "id": "rate",
                    "autoscaled": True,
                    "title": "实测/理论参考线（拟合窗口内）",
                    "ylabel": "误差（对数轴）",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "① “误差在 log 轴上看起来是直线”不等于指数收敛：整函数（如 eˣ）的误差在"
                "半对数图上会越走越陡，只有当 f 在某个 Bernstein 椭圆 E_ρ 内解析时才有"
                "有限的 ρ，理论斜率才等于 −ln ρ。② 用 Gauss 求积算 |x−a| 或 sign(x−a) 的"
                "内积却不在折点处切段，求积误差（~1e-3）会直接盖住逼近误差，"
                "于是你量到的是求积的错；③ 偶函数的奇次系数恒为 0，误差曲线会呈阶梯状，"
                "拿它去拟合斜率会得到偏小的 α。"
            ),
            "theory": [
                "Jackson 型估计：f 有 ν 阶连续导数且 f^(ν) 有界变差 ⇒ 系数 O(k^(−ν−1))、L² 误差 O(n^(−ν−1/2))（笔记定理 5）。",
                "Bernstein 椭圆：f 在 E_ρ 内解析 ⇒ 误差 O(ρ^(−n))，斜率 −lnρ（笔记定理 5）。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "equioscillation",
            "title": "交错点：Remez 的等幅振荡",
            "description": (
                "Remez 交换算法逐轮的误差曲线（时间轴 = 迭代轮次）与当前交错点标记；"
                "虚线是同一阶的 L² 投影误差，用来看“最小二乘”与“最小最大”差在哪里。"
                "收敛后交错点个数必须等于 n+2，且各点振幅比 ~1。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "func",
                    "label": "被逼近的函数",
                    "type": "enum",
                    "options": ["pole", "entire", "cusp"],
                    "default": "pole",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "deg",
                    "label": "逼近多项式次数 n",
                    "type": "int",
                    "min": 2,
                    "max": 16,
                    "default": 8,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 24.0, "dt": 1.0, "label": "Remez 迭代轮次 k"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "误差曲线 e(x) = f − p", "ylabel": "e(x)"},
                {"id": "alt", "autoscaled": True, "title": "交错点上的 |e|", "ylabel": "|e|", "scale": "log"},
            ],
            "pitfalls": (
                "① 初始交错点选得不好会不收敛：Remez 的收敛依赖“新点集落在旧点集的交错结构里”，"
                "把初值取成等距点或随机点，误差曲线会出现多于 n+2 个极值，换点规则随即振荡；"
                "② 不能用“交错点集合不再变化”当停机判据——浮点噪声下点集会在两三个状态间循环，"
                "其中既有最优解也有误差大一倍的解，必须用 de la Vallée Poussin 下界追平实测"
                "最大误差来判定；③ 交错点个数 = n+2 是定理，但如果误差已经掉到舍入地板"
                "（例如 eˣ 在 n≥12 时），极值全是噪声，这时“数出 n+2 个点”毫无意义。"
            ),
            "theory": [
                "Chebyshev 交错定理：p 是最佳一致逼近 ⟺ f−p 在 n+2 个点上符号交替、等幅（笔记定理 4）。",
                "de la Vallée Poussin 下界：任一 n+2 点交错集的 min|e| ≤ ‖f−p*‖∞（笔记定理 4 推论）。",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "zeros_evolution",
            "title": "正交多项式零点的演化",
            "description": (
                "每个阶数画出一族正交多项式的全部零点（分岔图风格），轴上叠加理论密度曲线；"
                "同一阶的 Chebyshev 零点（解析闭式）仿射映射后作为对照，"
                "下面板画每个根的 Newton 根误差 |p/p′|（log 轴，会触到舍入地板）。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "family",
                    "label": "正交多项式族",
                    "type": "enum",
                    "options": ["legendre", "chebyshev", "laguerre"],
                    "default": "legendre",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_max",
                    "label": "最高阶数",
                    "type": "int",
                    "min": 4,
                    "max": 40,
                    "default": 24,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 40.0, "dt": 1.0, "label": "最高阶数 n"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "零点位置 vs 阶数", "ylabel": "零点 x"},
                {"id": "resid", "autoscaled": True, "title": "Newton 根误差 |p_n/p_n′|", "ylabel": "根误差", "scale": "log"},
            ],
            "pitfalls": (
                "① 正交多项式的零点不是等距的：Legendre 与 Chebyshev 的零点都服从反正弦律"
                "（靠近端点加密），Laguerre 的零点在 (0,∞) 上向大 x 端逐渐稀疏；"
                "把零点画成等距会立刻与图上的 gap 曲线矛盾。"
                "② 别用 |p_n(x_k)| 当误差：Laguerre 的 |L_n| 在 n=40 时能到 1e14，"
                "残差 1e14 对应的根误差其实只有 1e-14；要么用与多项式尺度无关的"
                "Newton 修正量 |p/p′|，要么就明确说明残差是相对什么尺度。"
            ),
            "theory": [
                "零点性质：正交多项式的 n 个零点全在正交区间内、互异、且与低一阶的零点交错（笔记定理 1 推论）。",
                "Gauss 求积节点=正交多项式零点（笔记定理 2）；经验分布收敛到平衡测度（Legendre/Chebyshev 为反正弦律）。",
            ],
        },
    ],
}


def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "projection":
        return _projection(params, t)
    if viz_id == "convergence_rates":
        return _convergence_rates(params, t)
    if viz_id == "equioscillation":
        return _equioscillation(params, t)
    if viz_id == "zeros_evolution":
        return _zeros_evolution(params, t)
    raise KeyError(f"章节 2.3 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _f(value: Any) -> float | None:
    """有限数 → float；NaN/±Inf → ``None``（NLF 头用 ``allow_nan=False``，写 NaN 会让整帧炸）。"""
    v = float(value)
    return v if math.isfinite(v) else None


def _clamp_step(t: float, lo: int, hi: int) -> int:
    """把连续时间轴夹到整数步 ``[lo, hi]``（NaN 当作 lo）。"""
    if not math.isfinite(float(t)):
        return lo
    return max(lo, min(hi, int(round(float(t)))))


def _clip_floor(values: Any, floor: float = ROUNDOFF) -> tuple[np.ndarray, int]:
    """把误差曲线按 ``floor`` 截断，返回 ``(截断后的曲线, 被截断的点数)``。

    log 轴画不出 0：直接画会得到 −∞ 或干脆丢点，于是“误差=0”看起来像“完美收敛”。
    截断到舍入地板并在 meta 里写明地板值与截断点数，才是一句可核对的话。
    """
    v = np.asarray(values, dtype=float)
    bad = int(np.count_nonzero(~(v > floor)))
    return np.maximum(np.where(np.isfinite(v), v, floor), floor), bad


def _grid(a: float = -1.0, b: float = 1.0, n: int = _GRID) -> np.ndarray:
    return np.linspace(a, b, n)


def _breakpoints(func: str) -> tuple[float, ...]:
    """非光滑点：|x| 与 sign(x) 在 0 处折/跳，求积必须在 0 切段。"""
    return (0.0,) if func in ("cusp", "jump") else ()


def _eval_basis_sum(basis: str, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """在采样点 x 上按系数重建投影（与 :func:`l2_projection` 的节点无关）。"""
    from numcore.interp.approx import basis_values

    if coeffs.size == 0:
        return np.zeros_like(x)
    return basis_values(basis, int(coeffs.size), x).T @ coeffs


# --------------------------------------------------------------------------- #
# 1. 投影
# --------------------------------------------------------------------------- #
def _projection(params: Mapping[str, Any], t: float) -> Frame:
    basis = str(params["basis"])
    func = str(params["func"])
    n = int(params["n"])
    k = _clamp_step(t, 0, n)
    f = test_function(func, x0=0.0)
    bps = _breakpoints(func)
    x = _grid()
    fv = np.asarray(f(x), dtype=float)

    target = l2_projection(f, n, basis=basis, breakpoints=bps)
    proj_target = _eval_basis_sum(basis, target.coeffs, x)
    if k >= 1:
        cur = l2_projection(f, k, basis=basis, breakpoints=bps)
        proj = _eval_basis_sum(basis, cur.coeffs, x)
        coef = cur.coeffs
        l2_k = cur.l2_error
        linf_k = float(np.max(np.abs(fv - proj)))
    else:
        proj = np.zeros_like(x)
        coef = np.zeros(0, dtype=float)
        # k=0：投影是零函数，L² 误差就是 f 自己的范数（复合 Gauss，非光滑点切段）
        qx, qw = composite_gauss_legendre(-1.0, 1.0, panels=1, nodes_per_panel=80, breakpoints=bps)
        l2_k = float(math.sqrt(max(0.0, float(np.sum(qw * np.asarray(f(qx), dtype=float) ** 2)))))
        linf_k = float(np.max(np.abs(fv)))

    err = fv - proj
    err_t = fv - proj_target
    coef_pad = np.zeros(n, dtype=float)
    coef_pad[: coef.size] = coef

    fr = Frame(chapter="2.3", viz="projection")
    fr.add("x", x, unit="", label="采样点 x")
    fr.add("f", fv, label="目标函数 f")
    fr.add("proj", proj, label=f"当前 {k} 维投影")
    fr.add("proj_target", proj_target, label=f"目标 {n} 维投影")
    fr.add("error", err, label="误差 f − P_k f")
    fr.add("error_target", err_t, label="误差 f − P_n f")
    fr.add("coef", coef_pad, label="系数向量（该基底下）")
    fr.add("gram", target.gram, label="Gram 矩阵 G")
    fr.set_meta(
        step=float(k),
        n_terms=float(n),
        n_current=float(k),
        gram_cond=_f(target.gram_cond),
        gram_cond_times_u=_f(target.gram_cond * ROUNDOFF),
        normal_residual=_f(target.normal_residual),
        l2_error_current=_f(l2_k),
        l2_error_target=_f(target.l2_error),
        linf_error_current=_f(linf_k),
        linf_error_target=_f(float(np.max(np.abs(err_t)))),
        condition_exceeds_roundoff=bool(target.gram_cond * ROUNDOFF > 1.0),
        error_floor=ROUNDOFF,
    )
    fr.set_strings(
        basis=basis,
        basis_label={"legendre": "归一化 Legendre（对 Lebesgue 测度正交）", "chebyshev": "归一化 Chebyshev", "monomial": "单项式 1,x,x²,…"}[basis],
        func=func,
        note=(
            "三个基张成同一个子空间：投影作为函数相同，差别在系数与条件数。"
            if basis != "legendre"
            else "Legendre 基对 Lebesgue 测度正交归一，Gram 矩阵≈I，正规方程条件数=1。"
        ),
        singular_note=(
            "κ(G)·u > 1：正规方程的系数已不可信（残差仍然很小）"
            if target.gram_cond * ROUNDOFF > 1.0
            else "κ(G)·u ≤ 1：系数在舍入意义下可信"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# 2. 收敛率
# --------------------------------------------------------------------------- #
def _convergence_rates(params: Mapping[str, Any], t: float) -> Frame:
    scale = float(params["scale"])
    n_max = int(params["n_max"])
    ns = np.arange(2, n_max + 1)

    f_an = test_function("pole", scale=scale)
    f_cusp = test_function("cusp", x0=0.0)
    f_jump = test_function("jump", x0=0.0)
    e_an = projection_error_curve(f_an, ns)
    e_cusp = projection_error_curve(f_cusp, ns, breakpoints=(0.0,))
    e_jump = projection_error_curve(f_jump, ns, breakpoints=(0.0,))

    floor = ROUNDOFF
    e_an_c, bad_an = _clip_floor(e_an, floor)
    e_cusp_c, bad_cusp = _clip_floor(e_cusp, floor)
    e_jump_c, bad_jump = _clip_floor(e_jump, floor)

    # --- 解析情形：几何收敛，实测斜率用半对数拟合，理论斜率 = −ln ρ ---
    y_sing = 1.0 / math.sqrt(scale)
    rho = y_sing + math.sqrt(1.0 + y_sing * y_sing)
    rate_an_theory = -math.log(rho)
    skip_an = 1  # n=2 还在预渐近区
    rate_an, r2_an = fit_semilog_slope(ns, e_an_c, skip=skip_an)

    # --- 尖点 / 跳点：代数收敛，实测斜率用双对数拟合 ---
    skip_cusp = 6  # n≥8 才进入渐近区（实测：全窗口 −1.56，n≥8 起 −1.51）
    a_cusp, r2_cusp = fit_loglog_slope(ns, e_cusp_c, skip=skip_cusp)
    a_cusp, r2_cusp = -a_cusp, r2_cusp
    a_jump, r2_jump = fit_loglog_slope(ns, e_jump_c)
    a_jump, r2_jump = -a_jump, r2_jump
    alpha_cusp_theory = theory_algebraic_rate(1.0)
    alpha_jump_theory = theory_algebraic_rate(0.0)

    # 参考线都锚在各自拟合窗口的**第一个实测点**上（常数项不可知，斜率才是理论断言）
    def _anchor(ns_arr: np.ndarray, errs: np.ndarray, skip: int) -> tuple[float, float]:
        return float(ns_arr[skip]), float(errs[skip])

    n0a, e0a = _anchor(ns, e_an_c, skip_an)
    n0c, e0c = _anchor(ns, e_cusp_c, skip_cusp)
    n0j, e0j = _anchor(ns, e_jump_c, 0)

    theory_an = e0a * np.exp(rate_an_theory * (ns - n0a))
    fit_an = e0a * np.exp(rate_an * (ns - n0a))
    theory_cusp = e0c * (ns / n0c) ** (-alpha_cusp_theory)
    fit_cusp = e0c * (ns / n0c) ** (-a_cusp)
    theory_jump = e0j * (ns / n0j) ** (-alpha_jump_theory)
    fit_jump = e0j * (ns / n0j) ** (-a_jump)

    fr = Frame(chapter="2.3", viz="convergence_rates")
    fr.add("n", ns.astype(float), label="投影维数 n")
    fr.add("error", np.vstack([e_an_c, e_cusp_c, e_jump_c]), label="三条误差曲线（解析/尖点/跳点）")
    fr.add("error_analytic", e_an_c, label="解析函数 1/(1+s x²)")
    fr.add("error_cusp", e_cusp_c, label="尖点 |x|")
    fr.add("error_jump", e_jump_c, label="跳点 sign(x)")
    fr.add("theory_analytic", theory_an, label="理论参考线：ρ^(−n)")
    fr.add("theory_cusp", theory_cusp, label="理论参考线：n^(−3/2)")
    fr.add("theory_jump", theory_jump, label="理论参考线：n^(−1/2)")
    fr.add("fit_analytic", fit_an, label="实测拟合线（半对数斜率）")
    fr.add("fit_cusp", fit_cusp, label="实测拟合线（双对数斜率）")
    fr.add("fit_jump", fit_jump, label="实测拟合线（双对数斜率）")
    fr.set_meta(
        rate_analytic_measured=_f(rate_an),
        rate_analytic_theory=_f(rate_an_theory),
        rate_analytic_r2=_f(r2_an),
        rate_analytic_rel_error=_f(abs(rate_an - rate_an_theory) / abs(rate_an_theory)),
        rho=_f(rho),
        alpha_cusp_measured=_f(a_cusp),
        alpha_cusp_theory=_f(alpha_cusp_theory),
        alpha_cusp_r2=_f(r2_cusp),
        alpha_jump_measured=_f(a_jump),
        alpha_jump_theory=_f(alpha_jump_theory),
        alpha_jump_r2=_f(r2_jump),
        fit_window_analytic_start=float(n0a),
        fit_window_cusp_start=float(n0c),
        fit_window_jump_start=float(n0j),
        error_floor=floor,
        clipped_analytic=float(bad_an),
        clipped_cusp=float(bad_cusp),
        clipped_jump=float(bad_jump),
    )
    fr.set_strings(
        analytic="1/(1+s x²)，s=%.3g" % scale,
        regime_analytic="geometric（Bernstein 椭圆参数 ρ=%.6f，理论斜率 −lnρ=%.6f）" % (rho, rate_an_theory),
        regime_cusp="algebraic α=3/2（f′ 有跳变）",
        regime_jump="algebraic α=1/2（f 本身有跳变）",
        floor_note=(
            "误差曲线已按 u=2^-53 截断；截断点数见 meta.clipped_*。"
            "log 轴不能画 0：把 0 当成“完美收敛”是本章最常见的误读。"
        ),
        fit_note=(
            "实测斜率与理论斜率的对照都锚在拟合窗口的第一个实测点上——"
            "常数项不可知，斜率才是理论断言。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# 3. 交错点
# --------------------------------------------------------------------------- #
def _equioscillation(params: Mapping[str, Any], t: float) -> Frame:
    func = str(params["func"])
    deg = int(params["deg"])
    x0 = 0.0
    f = test_function(func, x0=x0)
    res = remez(f, deg, n_grid=1601)
    n_hist = len(res.coeff_history)
    k = _clamp_step(t, 1, n_hist)

    x = _grid()
    fv = np.asarray(f(x), dtype=float)
    coeffs_cur = res.coeff_history[k - 1]
    err_cur = fv - clenshaw(coeffs_cur, x)
    alt_x_cur = res.alt_history[k - 1]
    alt_e_cur = np.asarray(f(alt_x_cur), dtype=float) - clenshaw(coeffs_cur, alt_x_cur)

    err_fin = fv - clenshaw(res.coeffs, x)
    alt_e_fin = np.asarray(f(res.alt_x), dtype=float) - clenshaw(res.coeffs, res.alt_x)

    # 对照：同一阶的 L² 投影（最小二乘 ≠ 最小最大）
    l2 = l2_projection(f, deg + 1, breakpoints=_breakpoints(func))
    err_l2 = fv - _eval_basis_sum("legendre", l2.coeffs, x)
    max_l2 = float(np.max(np.abs(err_l2)))

    err_cur_c, bad_cur = _clip_floor(np.abs(err_cur))
    err_fin_c, bad_fin = _clip_floor(np.abs(err_fin))
    err_l2_c, bad_l2 = _clip_floor(np.abs(err_l2))
    ratio_cur = _ratio(alt_e_cur)
    ratio_fin = _ratio(alt_e_fin)

    fr = Frame(chapter="2.3", viz="equioscillation")
    fr.add("x", x, label="采样点 x")
    fr.add("error", err_cur, label=f"第 {k} 轮的误差曲线（带符号）")
    fr.add("error_final", err_fin, label="收敛后的误差曲线（带符号）")
    fr.add("abs_error", np.vstack([err_cur_c, err_fin_c, err_l2_c]), label="|误差|（当前/最终/L² 对照）")
    fr.add("alt_x", alt_x_cur, label=f"第 {k} 轮交错点")
    fr.add("alt_error", alt_e_cur, label=f"第 {k} 轮交错点上的误差（带符号）")
    fr.add("alt_x_final", res.alt_x, label="最终交错点")
    fr.add("alt_error_final", alt_e_fin, label="最终交错点上的误差（带符号）")
    fr.add("error_l2", err_l2, label="同阶 L² 投影的误差（对照）")
    fr.set_meta(
        step=float(k),
        n_steps=float(n_hist),
        deg=float(deg),
        n_alt_expected=float(deg + 2),
        n_alt_current=float(alt_x_cur.size),
        n_alt_final=float(res.n_alt),
        amplitude_ratio_current=_f(ratio_cur),
        amplitude_ratio_final=_f(ratio_fin),
        alt_signs_alternate=bool(res.sign_alternates),
        max_error_final=_f(res.max_error),
        max_error_l2=_f(max_l2),
        l2_over_minimax=_f(max_l2 / res.max_error if res.max_error > 0 else float("inf")),
        iterations=float(res.iterations),
        converged=bool(res.converged),
        error_floor=ROUNDOFF,
        clipped_points=float(bad_cur + bad_fin + bad_l2),
    )
    fr.set_strings(
        func=func,
        objective="minimax（‖f−p‖∞）",
        contrast="同阶 L² 投影（‖f−p‖₂）",
        certificate_note=(
            "停机判据是 de la Vallée Poussin 下界 min|e| ≥ (1−ε)·max|e|，"
            "不是“交错点不再移动”（浮点噪声下点集会循环）。"
        ),
        floor_note="误差已取绝对值并按 u=2^-53 截断；掉到地板后交错点全是噪声。",
        converged_note=(
            "交错点集稳定且被最优性证书确认"
            if res.converged
            else "本次未拿到证书（误差已触地板，或候选极值不足 deg+2 个）——返回的是历史最优迭代"
        ),
    )
    return fr


def _ratio(values: np.ndarray) -> float:
    a = np.abs(np.asarray(values, dtype=float))
    if a.size == 0 or float(np.min(a)) == 0.0:
        return float("inf")
    return float(np.max(a) / np.min(a))


# --------------------------------------------------------------------------- #
# 4. 零点演化
# --------------------------------------------------------------------------- #
def _density_curve(family: str, lo: float, hi: float, n_cur: int, m: int = 241) -> tuple[np.ndarray, np.ndarray]:
    """理论零点密度（平衡测度）：Legendre/Chebyshev 为反正弦律，Laguerre 为 Marchenko–Pastur 律。"""
    if family in ("legendre", "chebyshev"):
        t = np.linspace(1e-6, math.pi - 1e-6, m)
        x = -np.cos(t)
        y = 1.0 / (math.pi * np.sqrt(np.clip(1.0 - x * x, 1e-12, None)))
        return x, y
    # Laguerre：把 x 缩放到 [0,4]，密度 sqrt((4−X)/X)/(2π)（n→∞ 的平衡测度）
    upper = max(1.0, float(hi))
    xx = np.linspace(1e-4, 4.0 - 1e-4, m) * (upper / 4.0)
    big_x = xx / (upper / 4.0)
    y = np.sqrt(np.clip(4.0 - big_x, 0.0, None) / np.clip(big_x, 1e-12, None)) / (2.0 * math.pi)
    return xx, y / (upper / 4.0)


def _zeros_evolution(params: Mapping[str, Any], t: float) -> Frame:
    family = str(params["family"])
    n_max = int(params["n_max"])
    n_cur = _clamp_step(t, 1, n_max)

    deg_list: list[int] = []
    root_list: list[float] = []
    err_list: list[float] = []
    iter_list: list[float] = []
    contrast: list[float] = []
    cur_roots = np.zeros(0, dtype=float)
    cur_gaps = np.zeros(0, dtype=float)
    cur_iters = 0.0
    for n in range(1, n_cur + 1):
        rs = newton_roots_family(family, n)
        deg_list.extend([float(n)] * n)
        root_list.extend(float(v) for v in rs.roots)
        err_list.extend(float(v) for v in (rs.root_error if rs.root_error is not None else rs.residual))
        iter_list.extend(float(v) for v in rs.iterations)
        # 对照：把同一阶的 Chebyshev 零点仿射映射到本族该阶的根区间
        ch = chebyshev_roots(n)
        lo_r, hi_r = float(np.min(rs.roots)), float(np.max(rs.roots))
        contrast.extend(float(v) for v in (0.5 * (hi_r + lo_r) + 0.5 * (hi_r - lo_r) * ch))
        if n == n_cur:
            cur_roots = rs.roots
            cur_gaps = np.diff(rs.roots)
            cur_iters = float(rs.max_iterations)

    deg_arr = np.array(deg_list, dtype=float)
    root_arr = np.array(root_list, dtype=float)
    err_arr, bad = _clip_floor(np.array(err_list, dtype=float))
    contrast_arr = np.array(contrast, dtype=float)
    lo = float(np.min(root_arr))
    hi = float(np.max(root_arr))
    dens_x, dens_y = _density_curve(family, lo, hi, n_cur)

    gap_ratio = float(np.max(cur_gaps) / np.min(cur_gaps)) if cur_gaps.size else float("nan")
    fr = Frame(chapter="2.3", viz="zeros_evolution")
    fr.add("degree", deg_arr, label="阶数 n")
    fr.add("root", root_arr, label="零点位置")
    fr.add("error", err_arr, label="Newton 根误差 |p_n/p_n′|")
    fr.add("root_contrast", contrast_arr, label="对照：同阶 Chebyshev 零点（仿射映射）")
    fr.add("gap", cur_gaps, label="当前阶的相邻根间距")
    fr.add("gap_index", np.arange(1, cur_gaps.size + 1, dtype=float), label="间距编号")
    fr.add("density_x", dens_x, label="理论密度曲线 x")
    fr.add("density_y", dens_y, label="理论密度（反正弦律 / Marpchenko–Pastur 律）")
    fr.add("root_current", cur_roots, label="当前阶的全部零点")
    fr.set_meta(
        step=float(n_cur),
        n_current=float(n_cur),
        n_roots_current=float(cur_roots.size),
        max_root_error=_f(np.max(err_arr)) if err_arr.size else None,
        min_root_error=_f(np.min(err_arr)) if err_arr.size else None,
        root_error_floor=ROUNDOFF,
        clipped_points=float(bad),
        newton_iter_max=cur_iters,
        newton_iter_mean=_f(float(np.mean(iter_list))),
        gap_ratio_current=_f(gap_ratio),
        gap_min_current=_f(float(np.min(cur_gaps))) if cur_gaps.size else None,
        gap_max_current=_f(float(np.max(cur_gaps))) if cur_gaps.size else None,
        root_span_lo=_f(lo),
        root_span_hi=_f(hi),
    )
    fr.set_strings(
        family=family,
        family_label={
            "legendre": "Legendre P_n（权 1，区间 [−1,1]）",
            "chebyshev": "Chebyshev T_n（权 1/√(1−x²)，零点 cos((2k−1)π/2n)）",
            "laguerre": "Laguerre L_n（权 e^(−x)，区间 [0,∞)）",
        }[family],
        spacing_note=(
            "零点不是等距的：间距在区间中部最大、靠近端点最小（Laguerre 则向大 x 端变稀）。"
            "meta.gap_ratio_current = max gap / min gap 就是这条的实测值。"
        ),
        error_note=(
            "误差用 Newton 修正量 |p_n/p_n′|（与多项式自身尺度无关）。"
            "Laguerre 的 |L_n| 可以到 1e14，用它当误差会得出荒谬的结论。"
        ),
        density_note="理论密度是 n→∞ 的平衡测度，只在中等以上阶数与零点直方图吻合。",
    )
    return fr


# 供测试引用（不在 manifest 里出现的内部工具）
_INTERNAL: dict[str, Callable[..., Any] | tuple[str, ...]] = {
    "BASES": BASES,
    "clip_floor": _clip_floor,
}
