"""成熟实现的包装层：给自研实现当靶子。

约定
----
* 本模块的函数名统一以语义命名（``solve`` / ``eig`` / ``quad`` …），与本仓库自研 API 分开命名空间；
* 每个函数 docstring 里注明**底层库**与**精度等级**，因为"和 SciPy 一致"这句话必须能追溯到具体算法；
* 不在本层做任何"教学化包装"——它是靶子，不是教材。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.linalg as _nla
import scipy.integrate as _sint
import scipy.interpolate as _sinterp
import scipy.linalg as _sla
import scipy.optimize as _sopt
import scipy.sparse as _ssparse
import scipy.sparse.linalg as _sspla
import scipy.special as _sspec
import scipy.stats as _sstats

__all__ = [
    # 稠密线代
    "solve", "solve_triangular", "inv", "det", "lu", "cholesky", "qr", "svd", "eig", "eigh",
    "lstsq", "pinv", "cond", "norm", "matrix_rank", "expm", "logm", "sqrtm", "schur", "hessenberg",
    # 稀疏
    "sparse_csr", "sparse_nnz", "sparse_lu_nnz", "sparse_spsolve", "sparse_eigs",
    # 变换
    "fft", "ifft", "rfft", "dct",
    # 求积与 ODE
    "quad", "quad_inf", "romb", "solve_ivp", "ode_dense",
    # 插值与逼近
    "cubic_spline", "pchip", "barycentric", "bspline", "remez_like",
    # 优化与统计
    "minimize", "linprog", "root", "brentq", "ks_2samp", "norm_ppf",
    # 特殊函数
    "erf", "gamma", "beta", "jv", "yn", "legendre_p", "chebyt", "hermite",
]


# --------------------------------------------------------------------------- #
# 稠密线代（numpy.linalg / scipy.linalg）
# --------------------------------------------------------------------------- #
def solve(A: Any, b: Any) -> np.ndarray:
    """``A x = b`` 的 LU 解（``numpy.linalg.solve``，LAPACK dgesv）。"""
    return _nla.solve(np.asarray(A, dtype=float), np.asarray(b, dtype=float))


def solve_triangular(A: Any, b: Any, *, lower: bool = True, unit_diagonal: bool = False) -> np.ndarray:
    """三角回代（``scipy.linalg.solve_triangular``，LAPACK dtrtrs）。"""
    return _sla.solve_triangular(
        np.asarray(A, dtype=float), np.asarray(b, dtype=float), lower=lower, unit_diagonal=unit_diagonal
    )


def inv(A: Any) -> np.ndarray:
    """矩阵求逆（``numpy.linalg.inv``）。"""
    return _nla.inv(np.asarray(A, dtype=float))


def det(A: Any) -> float:
    """行列式（``scipy.linalg.det``，含 LU 分解）。"""
    return float(_sla.det(np.asarray(A, dtype=float)))


def lu(A: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``scipy.linalg.lu``：返回 ``(P, L, U, A0)``，满足 ``A0 = P @ L @ U``。"""
    return _sla.lu(np.asarray(A, dtype=float))


def cholesky(A: Any, *, lower: bool = True) -> np.ndarray:
    """Cholesky 因子（``numpy.linalg.cholesky``，LAPACK dpotrf）。"""
    L = _nla.cholesky(np.asarray(A, dtype=float))
    return L if lower else L.T


def qr(A: Any, *, mode: str = "complete") -> tuple[np.ndarray, np.ndarray]:
    """QR 分解（``numpy.linalg.qr``，Householder）。"""
    return _nla.qr(np.asarray(A, dtype=float), mode=mode)


def svd(A: Any, *, full_matrices: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD（``numpy.linalg.svd``，LAPACK dgesdd）。"""
    return _nla.svd(np.asarray(A, dtype=float), full_matrices=full_matrices)


def eig(A: Any) -> tuple[np.ndarray, np.ndarray]:
    """一般矩阵特征分解（``numpy.linalg.eig``，LAPACK dgeev）。"""
    return _nla.eig(np.asarray(A, dtype=float))


def eigvals(A: Any) -> np.ndarray:
    """特征值（``numpy.linalg.eigvals``）。"""
    return _nla.eigvals(np.asarray(A, dtype=float))


def eigh(A: Any) -> tuple[np.ndarray, np.ndarray]:
    """对称矩阵特征分解（``numpy.linalg.eigh``，LAPACK dsyevd）。"""
    return _nla.eigh(np.asarray(A, dtype=float))


def lstsq(A: Any, b: Any, *, rcond: float | None = None) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """最小二乘（``numpy.linalg.lstsq``，Gelssd/QR）。"""
    return _nla.lstsq(np.asarray(A, dtype=float), np.asarray(b, dtype=float), rcond=rcond)


def pinv(A: Any, *, rcond: float = 1e-15) -> np.ndarray:
    """Moore–Penrose 伪逆（``numpy.linalg.pinv``）。"""
    return _nla.pinv(np.asarray(A, dtype=float), rcond=rcond)


def cond(A: Any, *, p: float | str | None = None) -> float:
    """条件数（``numpy.linalg.cond``）；缺省 2-范数条件数 ``σ_max/σ_min``。"""
    return float(_nla.cond(np.asarray(A, dtype=float), p=p))


def norm(x: Any, *, ord: float | str | None = None) -> float:
    """向量/矩阵范数（``numpy.linalg.norm``）。"""
    return float(_nla.norm(np.asarray(x, dtype=float), ord=ord))


def matrix_rank(A: Any, *, tol: float | None = None) -> int:
    """数值秩（``numpy.linalg.matrix_rank``，基于 SVD）。"""
    return int(_nla.matrix_rank(np.asarray(A, dtype=float), tol=tol))


def expm(A: Any) -> np.ndarray:
    """矩阵指数（``scipy.linalg.expm``，Padé + scaling-squaring）。"""
    return _sla.expm(np.asarray(A, dtype=float))


def logm(A: Any) -> np.ndarray:
    """矩阵对数（``scipy.linalg.logm``）。"""
    return _sla.logm(np.asarray(A, dtype=float))


def sqrtm(A: Any) -> np.ndarray:
    """矩阵平方根（``scipy.linalg.sqrtm``）。"""
    return _sla.sqrtm(np.asarray(A, dtype=float))


def schur(A: Any, *, output: str = "real") -> tuple[np.ndarray, np.ndarray]:
    """Schur 分解（``scipy.linalg.schur``）。"""
    return _sla.schur(np.asarray(A, dtype=float), output=output)


def hessenberg(A: Any, *, calc_q: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Hessenberg 化（``scipy.linalg.hessenberg``）。"""
    return _sla.hessenberg(np.asarray(A, dtype=float), calc_q=calc_q)


# --------------------------------------------------------------------------- #
# 稀疏
# --------------------------------------------------------------------------- #
def sparse_csr(A: Any) -> Any:
    """转 CSR（``scipy.sparse.csr_matrix``）。"""
    return _ssparse.csr_matrix(np.asarray(A, dtype=float))


def sparse_nnz(A: Any) -> int:
    """非零元个数（严格 ``!= 0``）。"""
    return int(_ssparse.csr_matrix(np.asarray(A, dtype=float)).nnz)


def sparse_lu_nnz(A: Any, *, permc_spec: str = "COLAMD") -> int:
    """``splu`` 分解后 ``L+U`` 的非零元个数（1.2 章填充对比的参照）。"""
    lu_obj = _sspla.splu(_ssparse.csc_matrix(np.asarray(A, dtype=float)), permc_spec=permc_spec)
    return int(lu_obj.L.nnz + lu_obj.U.nnz)


def sparse_spsolve(A: Any, b: Any, *, permc_spec: str = "COLAMD") -> np.ndarray:
    """稀疏直接解（``scipy.sparse.linalg.spsolve``）。"""
    return _sspla.spsolve(_ssparse.csc_matrix(np.asarray(A, dtype=float)), np.asarray(b, dtype=float))


def sparse_eigs(A: Any, k: int = 6, *, which: str = "LM") -> tuple[np.ndarray, np.ndarray]:
    """稀疏特征值（``scipy.sparse.linalg.eigs``，ARPACK）。"""
    vals, vecs = _sspla.eigs(_ssparse.csc_matrix(np.asarray(A, dtype=float)), k=k, which=which)
    return vals, vecs


# --------------------------------------------------------------------------- #
# 变换
# --------------------------------------------------------------------------- #
def fft(x: Any) -> np.ndarray:
    """DFT（``numpy.fft.fft``，pocketfft）。"""
    return np.fft.fft(np.asarray(x, dtype=complex))


def ifft(x: Any) -> np.ndarray:
    """逆 DFT（``numpy.fft.ifft``）。"""
    return np.fft.ifft(np.asarray(x, dtype=complex))


def rfft(x: Any) -> np.ndarray:
    """实序列 DFT（``numpy.fft.rfft``）。"""
    return np.fft.rfft(np.asarray(x, dtype=float))


def dct(x: Any, *, type_: int = 2) -> np.ndarray:
    """DCT（``scipy.fft.dct``）。"""
    return _sfft.dct(np.asarray(x, dtype=float), type=type_)


# --------------------------------------------------------------------------- #
# 求积与 ODE
# --------------------------------------------------------------------------- #
def quad(f: Callable[[float], float], a: float, b: float, *, epsabs: float = 1e-13, epsrel: float = 1e-13) -> tuple[float, float, int]:
    """自适应 Gauss–Kronrod 求积（``scipy.integrate.quad``，QUADPACK dqagse）。

    返回 ``(积分值, 误差估计, 函数求值次数)``。
    """
    val, err, info = _sint.quad(f, a, b, epsabs=epsabs, epsrel=epsrel, limit=500, full_output=1)
    return float(val), float(err), int(info["neval"])


def quad_inf(f: Callable[[float], float], a: float, *, sign: int = 1) -> tuple[float, float]:
    """半无穷区间积分（``scipy.integrate.quad`` 的 ``(-inf, a]`` / ``[a, inf)``）。"""
    bounds = (-np.inf, a) if sign < 0 else (a, np.inf)
    val, err = _sint.quad(f, bounds[0], bounds[1], epsabs=1e-13, epsrel=1e-13, limit=500)[:2]
    return float(val), float(err)


def romb(y: Any, dx: float = 1.0) -> float:
    """Romberg 积分（``scipy.integrate.romb``，要求等距样本且 n = 2^k + 1）。"""
    return float(_sint.romb(np.asarray(y, dtype=float), dx=dx))


def solve_ivp(
    f: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: Any,
    *,
    method: str = "DOP853",
    rtol: float = 1e-12,
    atol: float = 1e-14,
    t_eval: Any = None,
) -> Any:
    """高精度 ODE 参考解（``scipy.integrate.solve_ivp``）。

    默认 ``DOP853``（8 阶显式 RK）+ ``rtol=1e-12``：比任何教学方法的精度高 6 个数量级以上，
    足以当作"真解"。返回 ``OdeSolution`` 对象（``.t`` / ``.y`` / ``.sol``）。
    """
    return _sint.solve_ivp(f, t_span, np.asarray(y0, dtype=float), method=method, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=True)


def ode_dense(sol: Any) -> Callable[[float], np.ndarray]:
    """把 ``solve_ivp`` 结果变成可任意取值的函数 ``t -> y(t)``。"""
    return lambda t: np.asarray(sol.sol(t), dtype=float)


# --------------------------------------------------------------------------- #
# 插值与逼近
# --------------------------------------------------------------------------- #
def cubic_spline(x: Any, y: Any, *, bc_type: str = "natural", xq: Any = None) -> Any:
    """三次样条（``scipy.interpolate.CubicSpline``）；``xq=None`` 时只返回构造好的对象。"""
    cs = _sinterp.CubicSpline(np.asarray(x, dtype=float), np.asarray(y, dtype=float), bc_type=bc_type)
    return cs if xq is None else np.asarray(cs(np.asarray(xq, dtype=float)))


def pchip(x: Any, y: Any, xq: Any = None) -> Any:
    """保形分段三次（``scipy.interpolate.PchipInterpolator``）。"""
    p = _sinterp.PchipInterpolator(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return p if xq is None else np.asarray(p(np.asarray(xq, dtype=float)))


def barycentric(x: Any, y: Any, xq: Any) -> np.ndarray:
    """重心插值（``scipy.interpolate.BarycentricInterpolator``）。"""
    b = _sinterp.BarycentricInterpolator(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.asarray(b(np.asarray(xq, dtype=float)))


def bspline(x: Any, y: Any, *, k: int = 3, xq: Any = None) -> Any:
    """B 样条（``scipy.interpolate.make_interp_spline``）。"""
    sp = _sinterp.make_interp_spline(np.asarray(x, dtype=float), np.asarray(y, dtype=float), k=k)
    return sp if xq is None else np.asarray(sp(np.asarray(xq, dtype=float)))


def remez_like(f: Callable[[float], float], a: float, b: float, deg: int) -> np.ndarray:
    """最佳一致逼近的参照：用 Chebyshev 节点插值近似 Remez 结果（非真 Remez，仅作粗对拍）。

    真 Remez 交换算法由 2.3 章自研实现；这里给出"Chebyshev 插值"这一已知良好的下界参照。
    """
    k = np.arange(deg + 1)
    nodes = np.cos(np.pi * k / deg)
    xs = 0.5 * (a + b) + 0.5 * (b - a) * nodes
    return _sinterp.BarycentricInterpolator(xs, np.asarray([f(float(t)) for t in xs])).coef


# --------------------------------------------------------------------------- #
# 优化与统计
# --------------------------------------------------------------------------- #
def minimize(f: Callable[..., float], x0: Any, *, method: str = "BFGS", jac: Any = None, tol: float = 1e-14) -> Any:
    """无约束最小化（``scipy.optimize.minimize``）。"""
    return _sopt.minimize(f, np.asarray(x0, dtype=float), method=method, jac=jac, tol=tol)


def linprog(c: Any, *, A_ub: Any = None, b_ub: Any = None, bounds: Any = None) -> Any:
    """线性规划（``scipy.optimize.linprog``，HiGHS）。"""
    return _sopt.linprog(np.asarray(c, dtype=float), A_ub=A_ub, b_ub=b_ub, bounds=bounds)


def root(f: Callable[[Any], Any], x0: Any, *, method: str = "hybr", jac: Any = None) -> Any:
    """非线性方程组（``scipy.optimize.root``）。"""
    return _sopt.root(f, np.asarray(x0, dtype=float), method=method, jac=jac)


def brentq(f: Callable[[float], float], a: float, b: float, *, xtol: float = 1e-15) -> float:
    """一维求根（``scipy.optimize.brentq``）。"""
    return float(_sopt.brentq(f, a, b, xtol=xtol))


def ks_2samp(x: Any, y: Any) -> tuple[float, float]:
    """两样本 Kolmogorov–Smirnov（``scipy.stats.ks_2samp``）：返回 ``(统计量, p 值)``。"""
    r = _sstats.ks_2samp(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return float(r.statistic), float(r.pvalue)


def norm_ppf(p: float) -> float:
    """标准正态分位数（``scipy.stats.norm.ppf``，Moro/AS241 算法）。"""
    return float(_sstats.norm.ppf(p))


# --------------------------------------------------------------------------- #
# 特殊函数
# --------------------------------------------------------------------------- #
def erf(x: Any) -> np.ndarray:
    """误差函数（``scipy.special.erf``）。"""
    return _sspec.erf(np.asarray(x, dtype=float))


def gamma(x: Any) -> np.ndarray:
    """Γ 函数（``scipy.special.gamma``）。"""
    return _sspec.gamma(np.asarray(x, dtype=float))


def beta(a: float, b: float) -> float:
    """B 函数（``scipy.special.beta``）。"""
    return float(_sspec.beta(a, b))


def jv(v: float, x: Any) -> np.ndarray:
    """第一类 Bessel 函数（``scipy.special.jv``）。"""
    return _sspec.jv(v, np.asarray(x, dtype=float))


def yn(n: int, x: Any) -> np.ndarray:
    """第二类 Bessel 函数（``scipy.special.yn``）。"""
    return _sspec.yn(n, np.asarray(x, dtype=float))


def legendre_p(n: int, x: Any) -> np.ndarray:
    """Legendre 多项式（``scipy.special.eval_legendre``）。"""
    return _sspec.eval_legendre(n, np.asarray(x, dtype=float))


def chebyt(n: int, x: Any) -> np.ndarray:
    """第一类 Chebyshev 多项式（``scipy.special.eval_chebyt``）。"""
    return _sspec.eval_chebyt(n, np.asarray(x, dtype=float))


def hermite(n: int, x: Any) -> np.ndarray:
    """物理学家 Hermite 多项式（``scipy.special.eval_hermite``）。"""
    return _sspec.eval_hermite(n, np.asarray(x, dtype=float))


def _import_scipy_fft() -> Any:  # pragma: no cover - 延迟导入，避免无关场景的加载开销
    import scipy.fft as sfft

    return sfft


_sfft = _import_scipy_fft()