"""精确解与高精度解 —— "真值"的另一个来源（有理数 / 符号 / 任意精度）。

三类武器：

* :mod:`fractions`（标准库）：**精确有理算术**。用来证明"某个算法在无舍入时是精确的"，
  例如 Cholesky 残差、Gauss 求积对 2n−1 次多项式的精确性、小整数矩阵的 LU。
* :mod:`sympy`：符号推导与**解析积分/极限**，给求积与逼近章节当靶子。
* :mod:`mpmath`：任意精度浮点，用 50–200 位十进制精度算出"接近真值"的参考量，
  用于研究误差随 h 的真实阶数（不会被 binary64 的舍入地板干扰）。

精度声明纪律：使用本模块的测试**必须**在 docstring 里写出所用的精度（``dps``/有理数/符号），
否则"和精确解一致"这句话没有意义。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable, Sequence

import mpmath as mp
import numpy as np
import sympy as sp

__all__ = [
    "exact_solve",
    "exact_det",
    "exact_inverse",
    "exact_rref",
    "exact_residual",
    "mp_solve",
    "mp_inverse",
    "mp_det",
    "mp_eigvals",
    "mp_eigsy",
    "mp_expm",
    "mp_quad",
    "mp_quad_np",
    "mp_sum",
    "mp_diff",
    "mp_ode",
    "mp_nstr",
    "sym_solve",
    "sym_det",
    "sym_integrate",
    "sym_diff",
    "sym_series",
    "sym_lambdify",
    "sym_simplify",
    "sym_limit",
]


# --------------------------------------------------------------------------- #
# 精确有理算术（fractions）
# --------------------------------------------------------------------------- #
def _to_frac_matrix(A: Any) -> list[list[Fraction]]:
    M = np.asarray(A)
    return [[Fraction(int(v)) if float(v).is_integer() else Fraction(float(v)).limit_denominator(10**12) for v in row] for row in M]


def exact_solve(A: Any, b: Any) -> list[Fraction]:
    """用**有理数**高斯消元（部分主元）精确求解 ``Ax = b``。

    输入必须是有理数（整数或有限小数）；浮点输入会被 ``limit_denominator`` 转成近似有理数，
    此时"精确"指的是**转换后的有理系统**精确，而不是原浮点系统——请自行留意。
    """
    M = _to_frac_matrix(A)
    n = len(M)
    rhs = [Fraction(float(v)).limit_denominator(10**12) for v in np.asarray(b).ravel()]
    if len(rhs) != n:
        raise ValueError("A 与 b 尺寸不符")
    aug = [row + [rhs[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if aug[piv][col] == 0:
            raise ValueError(f"矩阵奇异（第 {col} 列无主元），无法精确求解")
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [a - f * bc for a, bc in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def exact_det(A: Any) -> Fraction:
    """有理数 LU 计算行列式（精确）。"""
    M = _to_frac_matrix(A)
    n = len(M)
    if any(len(r) != n for r in M):
        raise ValueError("A 必须是方阵")
    sign = Fraction(1)
    det = Fraction(1)
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            sign = -sign
        det *= M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            M[r] = [a - f * bc for a, bc in zip(M[r], M[col])]
    return sign * det


def exact_inverse(A: Any) -> list[list[Fraction]]:
    """有理数 Gauss–Jordan 精确求逆。"""
    M = _to_frac_matrix(A)
    n = len(M)
    aug = [row + [Fraction(1) if i == j else Fraction(0) for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        piv = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if piv is None:
            raise ValueError("矩阵奇异，无法求逆")
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [a - f * bc for a, bc in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def exact_rref(A: Any) -> list[list[Fraction]]:
    """精确行最简形（rank 与零空间分析用）。"""
    M = _to_frac_matrix(A)
    rows, cols = len(M), len(M[0]) if M else 0
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [v / pv for v in M[r]]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * bc for a, bc in zip(M[i], M[r])]
        r += 1
        if r == rows:
            break
    return M


def exact_residual(A: Any, x: Sequence[Any], b: Any) -> list[Fraction]:
    """精确残差 ``b − Ax``（用有理数算，因此它真的是 0 或不是 0，不掺浮点噪声）。"""
    M = _to_frac_matrix(A)
    xv = [Fraction(v) if isinstance(v, Fraction) else Fraction(float(v)).limit_denominator(10**12) for v in x]
    bv = [Fraction(float(v)).limit_denominator(10**12) for v in np.asarray(b).ravel()]
    out = []
    for i, row in enumerate(M):
        s = sum((a * xi for a, xi in zip(row, xv)), Fraction(0))
        out.append(bv[i] - s)
    return out


# --------------------------------------------------------------------------- #
# 任意精度（mpmath）
# --------------------------------------------------------------------------- #
def mp_solve(A: Any, b: Any, *, dps: int = 50) -> Any:
    """``dps`` 位十进制精度下求解 ``Ax = b``（``mpmath.lu_solve``）。"""
    with mp.workdps(dps):
        M = mp.matrix([[mp.mpf(str(v)) for v in row] for row in np.asarray(A, dtype=float)])
        rhs = mp.matrix([mp.mpf(str(v)) for v in np.asarray(b, dtype=float).ravel()])
        return mp.lu_solve(M, rhs)


def mp_inverse(A: Any, *, dps: int = 50) -> Any:
    """任意精度求逆。"""
    with mp.workdps(dps):
        return mp.inverse(mp.matrix([[mp.mpf(str(v)) for v in row] for row in np.asarray(A, dtype=float)]))


def mp_det(A: Any, *, dps: int = 50) -> Any:
    """任意精度行列式。"""
    with mp.workdps(dps):
        return mp.det(mp.matrix([[mp.mpf(str(v)) for v in row] for row in np.asarray(A, dtype=float)]))


def mp_eigvals(A: Any, *, dps: int = 50) -> list[Any]:
    """任意精度特征值（``mpmath.eig``，QR 迭代）。"""
    with mp.workdps(dps):
        E, _ = mp.eig(mp.matrix([[mp.mpf(str(v)) for v in row] for row in np.asarray(A, dtype=float)]))
        return [E[i] for i in range(E.rows)]


def mp_eigsy(A: Any, *, dps: int = 50) -> list[Any]:
    """对称矩阵的任意精度特征值（``mpmath.eigsy``）。"""
    with mp.workdps(dps):
        E, _ = mp.eigsy(mp.matrix([[mp.mpf(str(v)) for v in row] for row in np.asarray(A, dtype=float)]))
        return [E[i] for i in range(E.rows)]


def mp_expm(A: Any, *, dps: int = 50) -> Any:
    """任意精度矩阵指数（``mpmath.expm``）。"""
    with mp.workdps(dps):
        return mp.expm(mp.matrix([[mp.mpf(str(v)) for v in row] for row in np.asarray(A, dtype=float)]))


def mp_quad(f: Callable[[Any], Any], a: float, b: float, *, dps: int = 50) -> Any:
    """任意精度定积分（``mpmath.quad``）：**被积函数必须接受并返回 mpf**。

    例：``mp_quad(lambda t: mp.e**(-t*t), 0, 1)``。
    若你手上是 numpy 风格的 float 函数，请用 :func:`mp_quad_np`（精度上限 ~1e-16）。
    """
    with mp.workdps(dps):
        return mp.quad(f, [mp.mpf(str(a)), mp.mpf(str(b))])


def mp_quad_np(f: Callable[[float], float], a: float, b: float, *, dps: int = 40) -> float:
    """任意精度求积，但被积函数只在 double 下求值（实用型：精度上限 ≈ 1e-16）。

    用于"已知被积函数只有 double 精度实现"时，仍能避免**求积公式自身**带来的误差。
    """
    with mp.workdps(dps):
        val = mp.quad(lambda t: mp.mpf(float(f(float(t)))), [mp.mpf(str(a)), mp.mpf(str(b))])
        return float(val)


def mp_sum(values: Sequence[Any], *, dps: int = 50) -> Any:
    """任意精度求和（用于展示"先把和算准，再谈误差"）。"""
    with mp.workdps(dps):
        return mp.fsum([mp.mpf(str(v)) for v in values])


def mp_diff(f: Callable[[Any], Any], x: float, *, n: int = 1, dps: int = 50) -> Any:
    """任意精度数值微分（``mpmath.diff``，h 自动选取）。"""
    with mp.workdps(dps):
        return mp.diff(lambda t: mp.mpf(f(t)), mp.mpf(str(x)), n)


def mp_ode(f: Callable[[Any, Any], Any], t0: float, y0: Any, t1: float, *, dps: int = 40) -> Any:
    """任意精度 ODE 解（Taylor 方法，``mpmath.odefun``）。``f(t, y) -> y'``，``y`` 可为向量。"""
    with mp.workdps(dps):
        y0m = mp.matrix([mp.mpf(str(v)) for v in np.asarray(y0, dtype=float).ravel()])
        fun = mp.odefun(lambda t, y: mp.matrix([mp.mpf(str(v)) for v in np.asarray(f(t, np.asarray(y.tolist(), dtype=float)), dtype=float).ravel()]), mp.mpf(str(t0)), y0m)
        return fun(mp.mpf(str(t1)))


def mp_nstr(x: Any, *, n: int = 30) -> str:
    """把 mpmath 数打印成 ``n`` 位有效数字的字符串。"""
    return mp.nstr(x, n)


# --------------------------------------------------------------------------- #
# 符号（sympy）
# --------------------------------------------------------------------------- #
def sym_solve(eqns: Any, syms: Any) -> Any:
    """符号求解方程组（``sympy.solve``）。"""
    return sp.solve(eqns, syms)


def sym_det(M: Any) -> Any:
    """符号行列式。"""
    return sp.Matrix(M).det()


def sym_integrate(expr: Any, var: Any, a: Any = None, b: Any = None) -> Any:
    """符号积分（定积分给 ``a, b``）——求积章节的"真值"来源。"""
    v = sp.Symbol(var) if isinstance(var, str) else var
    e = sp.sympify(expr) if isinstance(expr, str) else expr
    return sp.integrate(e, v) if a is None else sp.integrate(e, (v, a, b))


def sym_diff(expr: Any, var: Any, n: int = 1) -> Any:
    """符号求导。"""
    v = sp.Symbol(var) if isinstance(var, str) else var
    e = sp.sympify(expr) if isinstance(expr, str) else expr
    return sp.diff(e, v, n)


def sym_series(expr: Any, var: Any, x0: Any = 0, n: int = 6) -> Any:
    """泰勒展开（截断到 ``x^n``）——用于比较数值格式的截断误差项。"""
    v = sp.Symbol(var) if isinstance(var, str) else var
    e = sp.sympify(expr) if isinstance(expr, str) else expr
    return sp.series(e, v, x0, n).removeO()


def sym_lambdify(expr: Any, var: Any) -> Callable[[Any], Any]:
    """符号表达式 → 可调用函数（数值实验用）。"""
    v = sp.Symbol(var) if isinstance(var, str) else var
    e = sp.sympify(expr) if isinstance(expr, str) else expr
    f = sp.lambdify(v, e, modules=["numpy"])
    return f


def sym_simplify(expr: Any) -> Any:
    """符号化简。"""
    return sp.simplify(expr)


def sym_limit(expr: Any, var: Any, x0: Any) -> Any:
    """符号极限——用来给"收敛阶"这类结论做解析背书。"""
    v = sp.Symbol(var) if isinstance(var, str) else var
    e = sp.sympify(expr) if isinstance(expr, str) else expr
    return sp.limit(e, v, x0)
