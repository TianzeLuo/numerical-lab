r"""3.2 章：隐式方法（隐式 Euler / 梯形 / BDF2）、Newton 迭代与刚性算例。

每一步都要解一个（一般非线性的）方程

.. math::
    F(y)=0,\qquad
    F(y)=\begin{cases}
      y-y_n-h f(t_{n+1},y) & \text{隐式 Euler}\\
      y-y_n-\tfrac h2\bigl(f(t_n,y_n)+f(t_{n+1},y)\bigr) & \text{梯形法}\\
      \tfrac32 y-2y_n+\tfrac12 y_{n-1}-h f(t_{n+1},y) & \text{BDF2}
    \end{cases}

用 Newton 迭代解，线性化后的 :math:`m\times m` 系统
:math:`J\,\delta=-F` **一律走自研 LU**（:mod:`numcore.linalg.lu` 的
``lu_decompose`` / ``solve_lu``），不使用任何现成求解器（P2/P3）。
Newton 不收敛时**缩步长重试**（``allow_step_reduction``），而不是硬走——
这条策略在 ``pitfalls.md`` 里有实测数字。

显式一侧只保留最小实现：表驱动的显式 RK 单步（Butcher 表来自
:mod:`numcore.ode.stability`），通用显式方法族在 3.1 章的 ``numcore/ode/rk.py``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from numcore.linalg.lu import lu_decompose, solve_lu
from numcore.ode.stability import METHODS as STABILITY_METHODS
from numcore.ode.stability import rk_tableau, stability_modulus

__all__ = [
    "cost_ladder",
    "step_for_tolerance",
    "NewtonReport",
    "ODESolution",
    "EXPLICIT_METHODS",
    "IMPLICIT_METHODS",
    "newton_solve",
    "explicit_rk_step",
    "implicit_euler_step",
    "trapezoidal_step",
    "bdf2_step",
    "integrate",
    "amplification",
    "max_stable_step",
    "stability_limited_steps",
    "explicit_step_ceiling_theory",
    "largest_step_meeting_tolerance",
    "linear_rhs",
    "linear_jac",
    "linear_exact",
    "stiff_cos_rhs",
    "stiff_cos_jac",
    "stiff_cos_exact",
    "stiff_cos_solution_full",
    "stiff_diffusion_matrix",
    "stiff_spectrum_bound",
    "integrate_linear_system",
]

#: 显式方法（稳定域有界，刚性问题的步长天花板 ~ 常数/|λ|）。
EXPLICIT_METHODS: tuple[str, ...] = ("forward_euler", "midpoint", "heun", "rk4")

#: 隐式方法（本章实现的三族）。
IMPLICIT_METHODS: tuple[str, ...] = ("implicit_euler", "trapezoidal", "bdf2")

#: 显式方法在负实轴上的稳定边界常数（:math:`h\le c/|\lambda|`）：
#: forward Euler 是 2，RK4 是 2.785293563405\ldots（:math:`x^3+4x^2+12x+24=0` 的实根）。
_EXPLICIT_CEILING: dict[str, float] = {
    "forward_euler": 2.0,
    "midpoint": 2.0,
    "heun": 2.0,
    # x³ + 4x² + 12x + 24 = 0 的实根（把 R₄(x) = 1 约掉显式因子 x 得到的三次方程）
    "rk4": 2.785293563405282,
}

#: 数值爆炸的判据：|y| 超过它就认为"已经炸了"（把 inf/NaN 挡在帧外，见 pitfalls）。
BLOWUP = 1.0e12


# --------------------------------------------------------------------------- #
# 1. Newton 迭代（线性化 → 自研 LU）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NewtonReport:
    """一次 Newton 求解的可观测物（迭代次数、残余、LU 次数、回溯次数）。"""

    iterations: int = 0
    converged: bool = False
    residual: float = float("inf")
    residual_history: tuple[float, ...] = ()
    n_lu: int = 0
    n_backtracks: int = 0
    n_f_evals: int = 0
    n_jac_evals: int = 0
    singular: bool = False
    residual_scale: float = 1.0


def _max_abs(v: np.ndarray) -> float:
    a = np.abs(np.asarray(v, dtype=float))
    return float(np.max(a)) if a.size else 0.0


def newton_solve(
    residual: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    y0: Any,
    *,
    tol: float = 1e-12,
    max_iter: int = 25,
    damping: bool = True,
    backtrack_max: int = 10,
    atol: float = 0.0,
) -> tuple[np.ndarray, NewtonReport]:
    r""":math:`F(y)=0` 的 Newton 迭代：:math:`J(y)\,\delta=-F(y)`，:math:`y\leftarrow y+s\delta`。

    ``damping=True`` 时做**残差回溯**：若整步不下降就把 :math:`s` 减半（最多 ``backtrack_max`` 次），
    仍不下降就停手并报告不收敛（调用方据此缩步长，而不是接受一个假解）。
    线性问题（本章的刚性算例）一次迭代即到舍入底。

    **收敛判据是相对判据**：:math:`\|F(y)\|_\infty\le\mathrm{tol}\cdot\max(\|y\|_\infty,\|y_0\|_\infty)+\mathrm{atol}`。
    这一条是被实测逼出来的：解衰减到 :math:`10^{-11}` 量级时，绝对判据
    :math:`\|F\|\le10^{-11}` 会在**一次迭代都不做**的情况下宣布收敛，而那个
    "残差"已经是当前解的 50%（见 ``chapters/3.2/pitfalls.md``）。相对判据让
    Newton 在缩小的解上继续做一次迭代，结果就能回到舍入底。
    """
    y = np.array(y0, dtype=float, copy=True)
    r = np.asarray(residual(y), dtype=float)
    rn = _max_abs(r)
    y_ref = max(_max_abs(y), 1.0e-300)  # 步内参考尺度：判据相对化，避免"缩小的解上提前收敛"
    hist = [rn]
    n_f = 1
    n_lu = 0
    n_jac = 0
    n_bt = 0
    it = 0
    converged = rn <= tol * y_ref + atol
    singular = False
    while (not converged) and it < int(max_iter):
        it += 1
        J = np.asarray(jacobian(y), dtype=float)
        n_jac += 1
        lu = lu_decompose(J, pivot="partial")
        n_lu += 1
        if lu.singular or bool(np.any(~np.isfinite(lu.U))):
            singular = True
            break
        try:
            delta = solve_lu(lu, -r)
        except ZeroDivisionError:
            # ``lu_decompose`` 的奇异判据是**相对**尺度的：J 整行全零时尺度也是 0，
            # 它不一定报 singular，而 ``solve_lu`` 回代时会当场抛。Newton 求解器
            # 把这两种情况一视同仁地当成"这一步解不动"（调用方据此缩步长），
            # 绝不能让异常穿出去——刚性算例里 J 在半路上变奇异是家常便饭。
            singular = True
            break
        if not bool(np.all(np.isfinite(delta))):
            singular = True
            break
        step = 1.0
        accepted = False
        y_try = y
        r_try = r
        rn_try = rn
        for _ in range(max(1, int(backtrack_max)) if damping else 1):
            y_try = y + step * delta
            r_try = np.asarray(residual(y_try), dtype=float)
            n_f += 1
            rn_try = _max_abs(r_try)
            if rn_try < rn or not damping:
                accepted = True
                break
            step *= 0.5
            n_bt += 1
        if not accepted:
            break
        y, r, rn = y_try, r_try, rn_try
        hist.append(rn)
        y_ref = max(y_ref, _max_abs(y), 1.0e-300)
        converged = rn <= tol * y_ref + atol
    return y, NewtonReport(
        iterations=it,
        converged=bool(converged),
        residual=float(rn),
        residual_scale=float(y_ref),
        residual_history=tuple(float(v) for v in hist),
        n_lu=n_lu,
        n_backtracks=n_bt,
        n_f_evals=n_f,
        n_jac_evals=n_jac,
        singular=singular,
    )


# --------------------------------------------------------------------------- #
# 2. 单步：显式（表驱动）与三个隐式
# --------------------------------------------------------------------------- #
def explicit_rk_step(tableau: str, f: Callable[[float, np.ndarray], np.ndarray], t: float, y: Any, h: float) -> tuple[np.ndarray, int]:
    r"""一次显式 RK 步：:math:`k_i=f\bigl(t+c_ih,\;y+h\sum_j a_{ij}k_j\bigr)`，:math:`y^+=y+h\sum_i b_ik_i`。

    返回 ``(y_new, 函数求值次数)``；Butcher 表由 :func:`numcore.ode.stability.rk_tableau` 给出，
    因此"稳定多项式"与"实际走的步"用的是**同一张表**（不会出现图与算不一致）。
    """
    A, b = rk_tableau(tableau)
    s = int(b.size)
    c = A @ np.ones(s)
    m = np.asarray(y, dtype=float).size
    ks = np.zeros((s, m), dtype=float)
    yy = np.asarray(y, dtype=float)
    for i in range(s):
        stage = yy.copy()
        if i:
            stage = stage + h * (A[i, :i] @ ks[:i])
        ks[i] = np.asarray(f(t + float(c[i]) * h, stage), dtype=float)
    return yy + h * (b @ ks), s


def implicit_euler_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: Any,
    h: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 25,
) -> tuple[np.ndarray, NewtonReport]:
    r"""隐式 Euler：解 :math:`y^+-y-hf(t+h,y^+)=0`。"""
    yn = np.asarray(y, dtype=float)
    m = yn.size

    def residual(z: np.ndarray) -> np.ndarray:
        return z - yn - h * np.asarray(f(t + h, z), dtype=float)

    def jacobian(z: np.ndarray) -> np.ndarray:
        return np.eye(m) - h * np.asarray(jac(t + h, z), dtype=float)

    return newton_solve(residual, jacobian, yn, tol=tol, max_iter=max_iter)


def trapezoidal_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: Any,
    h: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 25,
) -> tuple[np.ndarray, NewtonReport]:
    r"""梯形法：解 :math:`y^+-y-\tfrac h2\bigl(f(t,y)+f(t+h,y^+)\bigr)=0`（:math:`f(t,y)` 只算一次）。"""
    yn = np.asarray(y, dtype=float)
    m = yn.size
    fn = np.asarray(f(t, yn), dtype=float)
    extra = 1  # 上面那次 f(t,y_n)

    def residual(z: np.ndarray) -> np.ndarray:
        return z - yn - 0.5 * h * (fn + np.asarray(f(t + h, z), dtype=float))

    def jacobian(z: np.ndarray) -> np.ndarray:
        return np.eye(m) - 0.5 * h * np.asarray(jac(t + h, z), dtype=float)

    z, rep = newton_solve(residual, jacobian, yn, tol=tol, max_iter=max_iter)
    return z, NewtonReport(
        iterations=rep.iterations,
        converged=rep.converged,
        residual=rep.residual,
        residual_history=rep.residual_history,
        n_lu=rep.n_lu,
        n_backtracks=rep.n_backtracks,
        n_f_evals=rep.n_f_evals + extra,
        n_jac_evals=rep.n_jac_evals,
        singular=rep.singular,
        residual_scale=rep.residual_scale,
    )


def bdf2_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: Any,
    y_prev: Any,
    h: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 25,
) -> tuple[np.ndarray, NewtonReport]:
    r"""BDF2：解 :math:`\tfrac32y^+-2y_n+\tfrac12y_{n-1}-hf(t+h,y^+)=0`。

    注意它需要**两个**历史值：第一步没有 :math:`y_{n-1}`，必须由别的方法启动
    （:func:`integrate` 缺省用隐式 Euler——刚性问题上前向 Euler 启动会把快模态放大，
    见 ``chapters/3.2/pitfalls.md``）。
    """
    yn = np.asarray(y, dtype=float)
    yp = np.asarray(y_prev, dtype=float)
    m = yn.size

    def residual(z: np.ndarray) -> np.ndarray:
        return 1.5 * z - 2.0 * yn + 0.5 * yp - h * np.asarray(f(t + h, z), dtype=float)

    def jacobian(z: np.ndarray) -> np.ndarray:
        return 1.5 * np.eye(m) - h * np.asarray(jac(t + h, z), dtype=float)

    return newton_solve(residual, jacobian, yn, tol=tol, max_iter=max_iter)


# --------------------------------------------------------------------------- #
# 3. 积分驱动
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ODESolution:
    """一次积分的全部可观测物（步数、函数求值、Jacobi 次数、LU 次数、Newton 迭代）。"""

    method: str
    t: np.ndarray
    y: np.ndarray
    h: float
    n_steps: int = 0
    n_f_evals: int = 0
    n_jac_evals: int = 0
    n_lu: int = 0
    n_newton_iters: int = 0
    max_newton_residual: float = 0.0
    n_step_reductions: int = 0
    n_newton_failures: int = 0
    ok: bool = True
    blowup: bool = False
    t_blowup: float = float("nan")
    max_abs_y: float = 0.0
    startup: str = ""
    error: float = float("nan")
    max_error: float = float("nan")
    notes: tuple[str, ...] = field(default=())

    def final(self) -> np.ndarray:
        """最后一个时刻的解向量（一维数组）。"""
        arr = np.asarray(self.y, dtype=float)
        return arr[-1] if arr.ndim > 1 else np.asarray([arr[-1]], dtype=float)

    def max_abs(self) -> float:
        """整条数值解上 :math:`\\max|y|`（爆炸检测用）。"""
        return float(self.max_abs_y)

    def cost(self) -> dict[str, float]:
        """计算量摘要（"隐式每步更贵"要用它来算总账）。"""
        return {
            "n_steps": float(self.n_steps),
            "n_f_evals": float(self.n_f_evals),
            "n_jac_evals": float(self.n_jac_evals),
            "n_lu": float(self.n_lu),
            "n_newton_iters": float(self.n_newton_iters),
        }


def integrate(
    method: str,
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray] | None,
    t_span: tuple[float, float],
    y0: Any,
    h: float,
    *,
    order: int = 4,
    tol: float = 1e-11,
    max_iter: int = 25,
    max_steps: int = 200000,
    allow_step_reduction: bool = True,
    startup: str = "implicit_euler",
    exact: Callable[[float], Any] | None = None,
) -> ODESolution:
    r"""定步长积分（显式与隐式共用一套计数器），返回 :class:`ODESolution`。

    设计上的三个"不撒谎"约定：

    1. **爆炸要显式**：:math:`|y|>` :data:`BLOWUP` 时立即停止，``blowup=True`` 并给出
       ``t_blowup``，而不是让 ``inf``/``nan`` 流进帧里（前端画不了非有限点）。
    2. **Newton 不收敛要缩步长**：``allow_step_reduction=True`` 时把 :math:`h` 减半重试
       （最多 12 次），并累计 ``n_step_reductions``；否则报告 ``ok=False``。
    3. **BDF2 的启动步要说出来**：``startup`` 取 ``implicit_euler``（缺省）或 ``forward_euler``，
       写进返回值供断言与陷阱分析对照。

    ``order`` 只影响 :math:`p` 级 :math:`p` 阶显式 Taylor/RK 族（``explicit_taylor``，
    见 :func:`numcore.ode.stability.stability_modulus`），本章的演示方法都用不到它。
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not t1 > t0:
        raise ValueError(f"需要 t1 > t0，得到 {t_span}")
    if not h > 0.0:
        raise ValueError(f"步长必须为正，得到 {h}")
    y = np.asarray(y0, dtype=float).reshape(-1)
    m = int(y.size)
    if jac is None and method in IMPLICIT_METHODS:
        raise ValueError(f"{method} 是隐式方法，必须给 Jacobi")

    ts: list[float] = [t0]
    ys: list[np.ndarray] = [y.copy()]
    n_f = 0
    n_jac = 0
    n_lu = 0
    n_it = 0
    n_red = 0
    n_fail = 0
    worst_res = 0.0  # 归一化残差 max|F|/max(|y|,|y0|)：与解的量级无关，才可跨问题比较
    max_abs_y = float(np.max(np.abs(y))) if m else 0.0
    blowup = False
    t_blow = float("nan")
    ok = True
    notes: list[str] = []
    hh = float(h)
    t = t0
    steps = 0
    started = False  # BDF2 是否已经走过启动步

    while t < t1 - 1e-15 and steps < int(max_steps):
        hh = min(hh, t1 - t)
        attempt = 0
        while True:
            attempt += 1
            if method in EXPLICIT_METHODS:
                y_new, k = explicit_rk_step(method, f, t, y, hh)
                n_f += k
                rep = None
            elif method == "implicit_euler":
                y_new, rep = implicit_euler_step(f, jac, t, y, hh, tol=tol, max_iter=max_iter)  # type: ignore[arg-type]
            elif method == "trapezoidal":
                y_new, rep = trapezoidal_step(f, jac, t, y, hh, tol=tol, max_iter=max_iter)  # type: ignore[arg-type]
            elif method == "bdf2":
                if not started:
                    if startup == "forward_euler":
                        y_new, k = explicit_rk_step("forward_euler", f, t, y, hh)
                        n_f += k
                        rep = None
                    else:
                        y_new, rep = implicit_euler_step(f, jac, t, y, hh, tol=tol, max_iter=max_iter)  # type: ignore[arg-type]
                else:
                    # 两个历史值：y 是当前值，ys[-2] 才是 y_{n-1}（ys[-1] 就是 y 自己）
                    y_new, rep = bdf2_step(f, jac, t, y, ys[-2], hh, tol=tol, max_iter=max_iter)  # type: ignore[arg-type]
            else:
                raise ValueError(f"未知方法 {method!r}（可用：{sorted(set(EXPLICIT_METHODS) | set(IMPLICIT_METHODS))}）")
            if rep is not None:
                n_lu += rep.n_lu
                n_it += rep.iterations
                n_jac += rep.n_jac_evals
                n_f += rep.n_f_evals
                worst_res = max(worst_res, rep.residual / max(rep.residual_scale, 1e-300))
                if (not rep.converged) and allow_step_reduction and attempt <= 12 and hh > 1e-15:
                    hh *= 0.5
                    n_red += 1
                    continue
                if not rep.converged:
                    n_fail += 1
                    ok = False
                    notes.append(f"t={t:.6g} 处 Newton 未收敛（残余 {rep.residual:.3e}），已按失败记")
            break
        finite = bool(np.all(np.isfinite(y_new)))
        mag = float(np.max(np.abs(y_new))) if y_new.size else 0.0
        if (not finite) or mag > BLOWUP:
            blowup = True
            t_blow = t + hh
            ok = False
            notes.append(f"t≈{t_blow:.6g} 处数值解爆炸（|y|>{BLOWUP:g}）")
            break
        t = t + hh
        y = y_new
        max_abs_y = max(max_abs_y, mag)
        ts.append(t)
        ys.append(y.copy())
        steps += 1
        started = True
        if steps >= int(max_steps) and t < t1 - 1e-15:
            ok = False
            notes.append(f"达到最大步数 {max_steps}，尚未走到 t={t1:g}（当前 t={t:.6g}）")

    y_arr = np.asarray(ys, dtype=float)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    err = float("nan")
    err_max = float("nan")
    if exact is not None and not blowup:
        try:
            ye = np.asarray(exact(t), dtype=float).reshape(-1)
            err = float(np.max(np.abs(y - ye)))
            # 端点误差会随解振荡（本例的误差在一个周期里能差四个数量级），
            # 所以"这条解有多准"一律看整段上的最大误差——它是稳健的那个量。
            err_hist = np.max(np.abs(y_arr - np.asarray(exact(np.asarray(ts, dtype=float)), dtype=float).reshape(y_arr.shape)), axis=1)
            err_max = float(np.max(err_hist))
        except (ValueError, TypeError):  # pragma: no cover - 调用方给错 exact 时不该炸掉整条流
            err = float("nan")
            err_max = float("nan")
    return ODESolution(
        method=method,
        t=np.asarray(ts, dtype=float),
        y=y_arr,
        h=float(h),
        n_steps=steps,
        n_f_evals=n_f,
        n_jac_evals=n_jac,
        n_lu=n_lu,
        n_newton_iters=n_it,
        max_newton_residual=float(worst_res),
        n_step_reductions=n_red,
        n_newton_failures=n_fail,
        ok=ok,
        blowup=blowup,
        t_blowup=t_blow,
        max_abs_y=max_abs_y,
        startup=startup if method == "bdf2" else "",
        error=err,
        max_error=err_max,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# 4. 放大因子与实测步长天花板
# --------------------------------------------------------------------------- #
def amplification(method: str, order: int, lam: complex, h: float) -> float:
    r"""线性测试方程上的**一步放大因子** :math:`|R(\lambda h)|`（多步法取最大根模）。"""
    return float(np.max(stability_modulus(method, order, complex(lam) * float(h))))


def _bisect_true_then_false(pred: Callable[[float], bool], lo: float, hi: float, *, iters: int = 80) -> float:
    r"""二分 ``pred(lo)=True, pred(hi)=False`` 的临界点（自写，不用 verify 的框架）。"""
    a, b = float(lo), float(hi)
    if not pred(a):
        raise ValueError("左端点处谓词应为真")
    if pred(b):
        raise ValueError("右端点处谓词应为假（请扩大区间）")
    for _ in range(int(iters)):
        mid = 0.5 * (a + b)
        if pred(mid):
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def max_stable_step(method: str, order: int, lam: complex, *, hi_factor: float = 20.0, iters: int = 90) -> float:
    r"""**实测**步长天花板 :math:`h_*=\sup\{h:|R(\lambda h)|\le1\}`（二分，非抄公式）。

    forward Euler 应当给出 :math:`2/|\lambda|`、RK4 给出 :math:`2.785293\ldots/|\lambda|`；
    隐式方法**没有**天花板（返回 ``inf``：稳定域包含整个左半平面）。
    """
    lm = complex(lam)
    if lm == 0:
        return math.inf
    scale = 1.0 / abs(lm)
    if method in IMPLICIT_METHODS or method == "bdf":
        return math.inf
    return _bisect_true_then_false(lambda h: amplification(method, order, lm, h) <= 1.0, 0.0, hi_factor * scale, iters=iters)


def stability_limited_steps(method: str, order: int, lam: complex, t_end: float, *, hi_factor: float = 20.0) -> int:
    r"""走到 ``t_end`` 至少要多少步（用**实测**的稳定天花板，不用理论常数）。"""
    hs = max_stable_step(method, order, lam, hi_factor=hi_factor)
    if not math.isfinite(hs) or hs <= 0.0:
        return 0  # 稳定性不构成限制（隐式方法）
    return int(math.ceil(float(t_end) / hs))


def explicit_step_ceiling_theory(method: str, order: int, lam: complex) -> float:
    r"""显式方法步长天花板的**理论值** :math:`c/|\lambda|`（c 见 :data:`_EXPLICIT_CEILING`）。

    :math:`c` 是稳定函数在负实轴上的边界点：forward Euler 的 :math:`R(z)=1+z` 给出 :math:`c=2`；
    RK4 的 :math:`c` 是 :math:`x^3+4x^2+12x+24=0` 的实根 :math:`2.785293\ldots`。
    隐式方法返回 ``inf``。
    """
    if method in IMPLICIT_METHODS or method == "bdf":
        return math.inf
    key = method if method != "explicit_taylor" else f"explicit_taylor_{int(order)}"
    c = _EXPLICIT_CEILING.get(key, _EXPLICIT_CEILING.get(method, math.nan))
    if not math.isfinite(c):
        return math.nan
    return c / abs(complex(lam))


# --------------------------------------------------------------------------- #
# 5. 刚性算例
# --------------------------------------------------------------------------- #
def _real_if_possible(z: complex) -> float | complex:
    """实参数不该被提升成复数（否则 Newton 的残差会带 1e-17 的虚部噪声）。"""
    return float(z.real) if z.imag == 0.0 else z


def linear_rhs(lam: complex) -> Callable[[float, np.ndarray], np.ndarray]:
    """标量线性问题 :math:`y'=\\lambda y` 的右端。"""
    lm = _real_if_possible(complex(lam))

    def f(t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG001 - 接口一致性
        return lm * np.asarray(y, dtype=float)

    return f


def linear_jac(lam: complex) -> Callable[[float, np.ndarray], np.ndarray]:
    r"""标量线性问题的 Jacobi：:math:`1\times1` 矩阵 :math:`[\lambda]`。"""
    lm = _real_if_possible(complex(lam))

    def jac(t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG001
        m = np.asarray(y, dtype=float).size
        return lm * np.eye(m)

    return jac


def linear_exact(lam: complex, y0: float = 1.0) -> Callable[[float], np.ndarray]:
    r"""标量线性问题的精确解 :math:`y(t)=y_0e^{\lambda t}`。"""
    lm = complex(lam)

    def exact(t: float) -> np.ndarray:
        return np.asarray([(y0 * np.exp(lm * t)).real], dtype=float)

    return exact


def stiff_cos_rhs(lam: float) -> Callable[[float, np.ndarray], np.ndarray]:
    r"""刚性受迫问题 :math:`y'=\lambda(y-\cos t)-\sin t`，其中 :math:`\lambda=-r<0`、:math:`r\gg1`。

    **参数约定**：``lam`` 是（负的）特征值本身，不是它的绝对值。:
    :func:`stiff_cos_exact` 给出的精确解是 :math:`y=\cos t`（要求 :math:`y(0)=1`）——
    快模态的系数恰好为 0，所以"刚性"在这条解曲线上**看不见**，只在步长上限里发作。
    """
    lm = float(lam)

    def f(t: float, y: np.ndarray) -> np.ndarray:
        yy = np.asarray(y, dtype=float)
        return lm * (yy - math.cos(t)) - math.sin(t)

    return f


def stiff_cos_jac(lam: float) -> Callable[[float, np.ndarray], np.ndarray]:
    r"""上式的 Jacobi：常数矩阵 :math:`\lambda I`（:math:`\lambda<0`）。"""
    lm = float(lam)

    def jac(t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG001
        m = np.asarray(y, dtype=float).size
        return lm * np.eye(m)

    return jac


def stiff_cos_exact(t: Any) -> np.ndarray:
    r""":math:`y(0)=1` 时刚性受迫问题的精确解：:math:`y(t)=\cos t`（快模态系数恰好为 0）。"""
    tt = np.asarray(t, dtype=float)
    return np.cos(tt)


def stiff_cos_solution_full(t: Any, lam: float, c0: float = 0.0) -> np.ndarray:
    r"""一般初值 :math:`y(0)=1+c_0` 的精确解 :math:`y=\cos t+c_0e^{-\lambda t}`。

    :math:`c_0\ne0` 时解里有一个 :math:`O(1/\lambda)` 宽的边界层——这层"几乎看不见"的
    快模态正是显式方法的步长天花板来源。
    """
    tt = np.asarray(t, dtype=float)
    return np.cos(tt) + float(c0) * np.exp(-float(lam) * tt)


def stiff_spectrum_bound(n: int) -> float:
    r"""一维扩散半离散矩阵 :math:`A=(n+1)^2\\,\\mathrm{tridiag}(1,-2,1)` 的谱半径上界 :math:`4(n+1)^2`。

    为什么是上界：:math:`\lambda_j=-4(n+1)^2\sin^2\!\bigl(\frac{j\pi}{2(n+2)}\bigr)`，
    最大模在 :math:`j\to n+1` 处趋近 :math:`4(n+1)^2`。
    """
    return 4.0 * (int(n) + 1) ** 2


def stiff_diffusion_matrix(n: int) -> np.ndarray:
    """一维扩散（Dirichlet）半离散矩阵：三对角，谱半径 ~ 4(n+1)^2（刚性比为 O(n²)）。"""
    nn = int(n)
    if nn < 1:
        raise ValueError(f"n 至少为 1，得到 {n}")
    A = np.zeros((nn, nn), dtype=float)
    d = float((nn + 1) ** 2)
    for i in range(nn):
        A[i, i] = -2.0 * d
        if i > 0:
            A[i, i - 1] = d
        if i < nn - 1:
            A[i, i + 1] = d
    return A


def integrate_linear_system(
    method: str,
    A: Any,
    t_span: tuple[float, float],
    y0: Any,
    h: float,
    *,
    order: int = 4,
    tol: float = 1e-12,
    max_steps: int = 200000,
) -> ODESolution:
    r"""线性系统 :math:`y'=Ay` 的定步长积分。

    隐式方法的 Newton 迭代是**精确一步收敛**的（问题线性），每步一次 LU——
    这条路径专门用来验证"自研 LU 真的在解线性系统"（多大规模的矩阵都走同一个入口）。
    """
    Am = np.asarray(A, dtype=float)
    mm = int(Am.shape[0])
    if Am.shape != (mm, mm):
        raise ValueError("A 必须是方阵")

    def f(t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return Am @ np.asarray(y, dtype=float)

    def jac(t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return Am

    return integrate(method, f, jac, t_span, np.asarray(y0, dtype=float).reshape(-1), h, order=order, tol=tol, max_steps=max_steps)


# --------------------------------------------------------------------------- #
# 6. 精度受限步长（实测）——刚性比阈值断言要用它
# --------------------------------------------------------------------------- #
def largest_step_meeting_tolerance(
    method: str,
    order: int,
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray] | None,
    t_span: tuple[float, float],
    y0: Any,
    tol: float,
    *,
    k_range: tuple[int, int] = (3, 13),
    max_steps: int = 200000,
    startup: str = "implicit_euler",
    exact: Callable[[float], Any] | None = None,
) -> tuple[float, float]:
    r"""在几何梯 :math:`h_k=(t_1-t_0)/2^k` 上**实测**满足误差容差的最大步长，返回 ``(h, 整段最大误差)``。

    判据用的是 ``max_error``（整段上的最大误差）而不是端点误差：本例的误差随解振荡，
    端点误差能在相邻两个梯级之间跳四个数量级（见 ``pitfalls.md``）。
    从 :math:`k=k_{\min}`（最大的 h）往上扫，返回**第一个**满足 :math:`\mathrm{err}\le\mathrm{tol}`
    的梯级——即梯子上最大的可用步长。这里用到了"误差随 h 单调下降"这个**渐近性质**
    （收敛方法的常规行为，不是定理）；如果它在某个问题上不成立，返回的只是"梯子上最早的可行点"，
    调用方（章节与陷阱分析）看的也是实测误差本身。
    返回 ``h=0`` 表示梯子上没有一步能满足容差（步长已经小到没意义）。
    这是「精度受限步长」的实现：刚性比阈值断言比较的正是它与「稳定受限步长」谁更小。
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    span = t1 - t0
    k_lo, k_hi = int(k_range[0]), int(k_range[1])
    last_err = float("inf")
    for k in range(max(1, k_lo), k_hi + 1):
        h = span / float(2**k)
        sol = integrate(method, f, jac, (t0, t1), y0, h, order=order, max_steps=max_steps, startup=startup, exact=exact)
        if (not sol.ok) or sol.blowup:
            continue
        if math.isfinite(sol.max_error):
            last_err = float(sol.max_error)
            if sol.max_error <= tol:
                return h, last_err
        if h * float(2**k) < span * 1e-12:  # pragma: no cover - 步长已到舍入尺度
            break
    return 0.0, last_err


# --------------------------------------------------------------------------- #
# 7. 代价梯：一次扫描给出"精度受限步长 vs 容差"的整条曲线
# --------------------------------------------------------------------------- #
def cost_ladder(
    method: str,
    order: int,
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray] | None,
    t_span: tuple[float, float],
    y0: Any,
    *,
    k_range: tuple[int, int] = (3, 11),
    max_steps: int = 200000,
    startup: str = "implicit_euler",
    exact: Callable[[float], Any] | None = None,
) -> dict[str, np.ndarray]:
    r"""在几何梯 :math:`h_k=(t_1-t_0)/2^k` 上把方法**真跑一遍**，返回整条代价曲线。

    每一行都是一次独立的积分（不是插值、不是公式）：``max_error`` 是该步长下整段解的
    最大误差，``n_f_evals`` / ``n_lu`` / ``n_newton_iters`` 是同一次积分的真实计数。
    有了它，"精度受限步长" :math:`h_{acc}(\mathrm{tol})` 与"总代价谁大"都变成一次查表，
    阈值断言才可能在二分的每一次迭代里都真算一遍（见 :func:`step_for_tolerance`）。
    """
    ks = list(range(max(1, int(k_range[0])), int(k_range[1]) + 1))
    span = float(t_span[1]) - float(t_span[0])
    out: dict[str, list[float]] = {k: [] for k in ("k", "h", "max_error", "error", "n_steps", "n_f_evals", "n_lu", "n_newton_iters", "ok")}
    for k in ks:
        h = span / float(2**k)
        sol = integrate(method, f, jac, t_span, y0, h, order=order, max_steps=max_steps, startup=startup, exact=exact)
        out["k"].append(float(k))
        out["h"].append(float(h))
        out["max_error"].append(_finite_or_inf(sol.max_error))
        out["error"].append(_finite_or_inf(sol.error))
        out["n_steps"].append(float(sol.n_steps))
        out["n_f_evals"].append(float(sol.n_f_evals))
        out["n_lu"].append(float(sol.n_lu))
        out["n_newton_iters"].append(float(sol.n_newton_iters))
        out["ok"].append(1.0 if (sol.ok and not sol.blowup) else 0.0)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _finite_or_inf(x: float) -> float:
    """梯子里的非有限误差统一记成 ``inf``（"这一步长根本不能用"），便于查表比较。"""
    v = float(x)
    return v if math.isfinite(v) else math.inf


def step_for_tolerance(ladder: Mapping[str, Any], tol: float) -> dict[str, float]:
    r"""从 :func:`cost_ladder` 的表里取"满足 :math:`\mathrm{max\_error}\le\mathrm{tol}` 的最大步长"。

    梯子上没有可用步长时返回 ``h=0``、``max_error=inf``。这里对 h 的单调性依赖与
    :func:`largest_step_meeting_tolerance` 相同（收敛方法的渐近行为），但**数据**是实测的。
    """
    h = np.asarray(ladder["h"], dtype=float)
    err = np.asarray(ladder["max_error"], dtype=float)
    ok = np.asarray(ladder["ok"], dtype=float) > 0.5
    good = ok & np.isfinite(err) & (err <= float(tol))
    if not bool(np.any(good)):
        return {"h": 0.0, "max_error": math.inf, "n_f_evals": math.inf, "n_lu": math.inf, "n_newton_iters": math.inf, "n_steps": math.inf, "k": math.inf}
    i = int(np.argmax(good))  # 梯子按 h 递减排列，第一个"好"的即最大步长
    return {
        "h": float(h[i]),
        "max_error": float(err[i]),
        "n_f_evals": float(ladder["n_f_evals"][i]),
        "n_lu": float(ladder["n_lu"][i]),
        "n_newton_iters": float(ladder["n_newton_iters"][i]),
        "n_steps": float(ladder["n_steps"][i]),
        "k": float(ladder["k"][i]),
    }
