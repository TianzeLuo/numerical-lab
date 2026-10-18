r"""稀疏迭代法：Jacobi / Gauss--Seidel / SOR 的迭代矩阵、谱半径与特征分量衰减。

数学定义
--------
把 :math:`A\in\mathbb{R}^{n\times n}` 分裂为

.. math::
    A = D - L - U,\qquad
    D = \operatorname{diag}(A),\quad
    L = -\operatorname{tril}(A,-1),\quad
    U = -\operatorname{triu}(A,1),

（对 M 矩阵这种写法让 :math:`L,U\ge 0`）。解 :math:`Ax=b` 的三族定常迭代都是
"把 :math:`A` 的一部分挪到右边"：

.. math::
    x^{(k+1)} = T x^{(k)} + c,\qquad
    T_J = D^{-1}(L+U),\quad
    T_{\mathrm{GS}} = (D-L)^{-1}U,\quad
    T_\omega = (D-\omega L)^{-1}\bigl((1-\omega)D+\omega U\bigr).

:math:`T` 叫**迭代矩阵**，误差满足 :math:`e^{(k)} = T^k e^{(0)}`，于是

.. math::
    \text{对任意 } x^{(0)} \text{ 收敛}
    \iff \rho(T) = \max_i|\lambda_i(T)| < 1,

而渐近收敛因子就是 :math:`\rho(T)`：:math:`\|e^{(k)}\|\approx C\rho(T)^k`。

本模块提供三件东西：

1. **迭代矩阵与谱半径**：:func:`iteration_matrix` 给出 :math:`T`（求逆一律走
   :mod:`numcore.linalg.lu` 的自研 LU，不调用 ``numpy.linalg``）；
   :func:`power_method` 用**手写幂法**估 :math:`|\lambda_1|`——注意主特征值可能是
   **复共轭对**（SOR 过松弛越过最优 :math:`\omega` 时就是这样），此时范数增长比会
   振荡不收敛，所以幂法同时维护"增长比的几何平均"作为复对的兜底估计。
2. **迭代求解器**：:func:`jacobi` / :func:`gauss_seidel` / :func:`sor`
   （统一入口 :func:`stationary_solve`），记录每一步的迭代向量、相对残差与误差。
3. **误差的特征分量分解**：:func:`mode_components` 把误差投影到模型问题
   （:func:`laplacian_1d`）的解析正弦模态上，:func:`mode_decay_factors`
   给出每个模态的理论衰减因子——"谁先衰减完"就是这些数字的大小顺序。

解析基准（A 类断言的靶子）
--------------------------
对 :math:`A=\operatorname{tridiag}(-1,2,-1)`（一维 Dirichlet Laplacian）：
:math:`\lambda_j(A)=4\sin^2\!\bigl(\tfrac{j\pi}{2(n+1)}\bigr)`，
:math:`v_j\propto\sin\bigl(\tfrac{ij\pi}{n+1}\bigr)`，于是
:math:`T_J = I-\tfrac12 A` 与 :math:`A` 同特征向量、
:math:`\mu_j(T_J)=\cos\bigl(\tfrac{j\pi}{n+1}\bigr)`，
:math:`\rho(T_J)=\cos\bigl(\tfrac{\pi}{n+1}\bigr)`。
对一致次序矩阵，Young 关系把 Jacobi 的 :math:`\mu` 映射到 SOR 的谱半径
（:func:`sor_spectral_radius`、:func:`young_optimal_omega`）。

实现约束：只用 numpy 的容器与逐元素原语（P3），随机数走 :func:`numcore.utils.rng`（P4），
本模块是纯函数（P6）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from numcore.linalg.lu import lu_decompose, lu_solve
from numcore.utils import rng

__all__ = [
    "Method",
    "PowerResult",
    "IterationResult",
    "laplacian_1d",
    "laplacian_2d",
    "sine_eigenvalues_1d",
    "sine_eigenvectors_1d",
    "jacobi_eigenvalues_1d",
    "rotated_spd_2x2",
    "jacobi_iteration_matrix",
    "gs_iteration_matrix",
    "sor_iteration_matrix",
    "iteration_matrix",
    "sor_spectral_radius",
    "young_optimal_omega",
    "power_method",
    "spectral_radius",
    "convergence_factor",
    "solve_direct",
    "stationary_solve",
    "jacobi",
    "gauss_seidel",
    "sor",
    "sor_rho_scan",
    "mode_components",
    "mode_decay_factors",
]

Method = Literal["jacobi", "gs", "sor"]

#: 幂法里"单向量瑞利商已经锁定实特征对"的判据
_RAYLEIGH_TOL = 1e-9
#: 复共轭主特征值对：增长比的几何平均窗口
_GEOM_WINDOW = 64


# --------------------------------------------------------------------------- #
# 模型问题与解析谱
# --------------------------------------------------------------------------- #
def laplacian_1d(n: int) -> np.ndarray:
    r"""一维 Dirichlet Laplacian：:math:`\operatorname{tridiag}(-1,2,-1)`（对称正定）。"""
    if int(n) < 1:
        raise ValueError(f"n 至少为 1，得到 {n}")
    m = int(n)
    A = np.zeros((m, m), dtype=float)
    idx = np.arange(m)
    A[idx, idx] = 2.0
    if m > 1:
        A[idx[:-1], idx[:-1] + 1] = -1.0
        A[idx[:-1] + 1, idx[:-1]] = -1.0
    return A


def laplacian_2d(k: int) -> np.ndarray:
    """k x k 网格上的二维 Dirichlet Laplacian（九点模板的对角 4、上下左右 -1）。"""
    if int(k) < 1:
        raise ValueError(f"k 至少为 1，得到 {k}")
    kk = int(k)
    n = kk * kk
    A = np.zeros((n, n), dtype=float)
    for i in range(kk):
        for j in range(kk):
            p = i * kk + j
            A[p, p] = 4.0
            if i > 0:
                A[p, p - kk] = -1.0
            if i < kk - 1:
                A[p, p + kk] = -1.0
            if j > 0:
                A[p, p - 1] = -1.0
            if j < kk - 1:
                A[p, p + 1] = -1.0
    return A


def sine_eigenvalues_1d(n: int) -> np.ndarray:
    r"""一维 Laplacian 的解析特征值 :math:`4\sin^2\!\bigl(j\pi/(2(n+1))\bigr)`，j=1..n。"""
    m = int(n)
    j = np.arange(1, m + 1, dtype=float)
    s = np.sin(j * np.pi / (2.0 * (m + 1)))
    return 4.0 * s * s


def sine_eigenvectors_1d(n: int) -> np.ndarray:
    r"""一维 Laplacian 的解析特征向量矩阵（第 j 列 = :math:`v_j`，2-范数归一）。

    :math:`v_j(i)=\sqrt{2/(n+1)}\,\sin\bigl(ij\pi/(n+1)\bigr)`。
    """
    m = int(n)
    i = np.arange(1, m + 1, dtype=float)[:, None]
    j = np.arange(1, m + 1, dtype=float)[None, :]
    V = np.sin(i * j * np.pi / (m + 1)) * np.sqrt(2.0 / (m + 1))
    return V


def jacobi_eigenvalues_1d(n: int) -> np.ndarray:
    r"""一维 Laplacian 上 Jacobi 迭代矩阵的**解析**谱：:math:`\mu_j=\cos(j\pi/(n+1))`。

    推导：:math:`D=2I`，:math:`T_J = D^{-1}(L+U) = I-\tfrac12 A`，与 :math:`A` 同特征向量。
    """
    m = int(n)
    j = np.arange(1, m + 1, dtype=float)
    return np.cos(j * np.pi / (m + 1))


def rotated_spd_2x2(kappa: float, angle_deg: float = 0.0) -> np.ndarray:
    r"""二维椭圆模型：:math:`A=R_\theta\operatorname{diag}(1,\kappa)R_\theta^{\mathsf T}`。

    特征值是 :math:`1` 与 :math:`\kappa`（故 :math:`\kappa_2(A)=\kappa` 精确成立），
    特征向量被旋转 :math:`\theta`。它的 Jacobi 迭代矩阵是 :math:`\begin{pmatrix}0&*\\ *&0\end{pmatrix}`，
    谱半径为 :math:`\mu=\sqrt{|a_{12}a_{21}|/(a_{11}a_{22})}`（解析可算），
    于是 SOR 的最优 :math:`\omega` 也有闭式——这正是 ``iteration_race`` 那一页要展示的几何图像：

    .. math::
        a_{11}=\cos^2\theta+\kappa\sin^2\theta,\quad
        a_{22}=\sin^2\theta+\kappa\cos^2\theta,\quad
        a_{12}=a_{21}=(1-\kappa)\sin\theta\cos\theta .
    """
    kk = float(kappa)
    if kk < 1.0:
        raise ValueError(f"条件数不能小于 1，得到 {kappa}")
    th = np.deg2rad(float(angle_deg))
    c = float(np.cos(th))
    s = float(np.sin(th))
    a11 = c * c + kk * s * s
    a22 = s * s + kk * c * c
    a12 = (1.0 - kk) * s * c
    out = np.array([[a11, a12], [a12, a22]], dtype=float)
    return out


# --------------------------------------------------------------------------- #
# 分裂与迭代矩阵
# --------------------------------------------------------------------------- #
def _as_square(A: Any) -> np.ndarray:
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    return arr


def _norm2(x: np.ndarray) -> float:
    return float(np.sqrt(float(x @ x)))


def _splitting(A: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(D, L, U)``，满足 ``A = D - L - U``（L、U 严格三角且取正号）。"""
    arr = _as_square(A)
    D = np.diag(np.diag(arr))
    L = -np.tril(arr, -1)
    U = -np.triu(arr, 1)
    return D, L, U


def _inverse_via_lu(M: np.ndarray) -> np.ndarray:
    """``M^{-1}``：逐列调用自研 LU 回代（解线性系统一律走 :mod:`numcore.linalg.lu`）。"""
    n = M.shape[0]
    res = lu_decompose(M, pivot="partial")
    if res.singular:
        raise ValueError("迭代矩阵的某个三角因子奇异（对角元为 0），无法构造迭代矩阵")
    out = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        out[:, j] = lu_solve(res, e)
    return out


def jacobi_iteration_matrix(A: Any) -> np.ndarray:
    r"""Jacobi 迭代矩阵 :math:`T_J=D^{-1}(L+U)`。"""
    D, L, U = _splitting(A)
    d = np.diag(D)
    if np.any(d == 0.0):
        raise ValueError("对角元含 0：Jacobi 迭代矩阵无定义（请先做置换或改用其它分裂）")
    return (L + U) / d[:, None]


def gs_iteration_matrix(A: Any) -> np.ndarray:
    r"""Gauss--Seidel 迭代矩阵 :math:`T_{\mathrm{GS}}=(D-L)^{-1}U`。"""
    D, L, U = _splitting(A)
    return _inverse_via_lu(D - L) @ U


def sor_iteration_matrix(A: Any, omega: float = 1.0) -> np.ndarray:
    r"""SOR 迭代矩阵 :math:`T_\omega=(D-\omega L)^{-1}((1-\omega)D+\omega U)`。

    :math:`\omega=1` 时退化为 Gauss--Seidel；收敛的必要条件是 :math:`0<\omega<2`。
    """
    w = float(omega)
    if not (w > 0.0):
        raise ValueError(f"松弛因子必须为正，得到 {omega}")
    D, L, U = _splitting(A)
    return _inverse_via_lu(D - w * L) @ ((1.0 - w) * D + w * U)


def iteration_matrix(A: Any, *, method: Method = "jacobi", omega: float = 1.0) -> np.ndarray:
    """迭代矩阵统一入口：``method`` 取 ``"jacobi"`` / ``"gs"`` / ``"sor"``。"""
    if method == "jacobi":
        return jacobi_iteration_matrix(A)
    if method == "gs":
        return gs_iteration_matrix(A)
    if method == "sor":
        return sor_iteration_matrix(A, omega)
    raise ValueError(f"未知迭代法 {method!r}")


# --------------------------------------------------------------------------- #
# Young 关系：SOR 的解析谱半径与最优 omega
# --------------------------------------------------------------------------- #
def sor_spectral_radius(mu: float, omega: float) -> float:
    r"""一致次序矩阵上 SOR 的**解析**谱半径（Young 关系）。

    设 Jacobi 迭代矩阵的谱半径为 :math:`\mu\in(0,1)`，则

    .. math::
        \rho(T_\omega)=
        \begin{cases}
        \Bigl(\tfrac{\omega\mu}{2}+\sqrt{\tfrac{\omega^2\mu^2}{4}-\omega+1}\Bigr)^2, & \omega\le\omega_{\rm opt},\\[2mm]
        \omega-1, & \omega\ge\omega_{\rm opt},
        \end{cases}
        \qquad
        \omega_{\rm opt}=\frac{2}{1+\sqrt{1-\mu^2}} .

    :math:`\omega>\omega_{\rm opt}` 时两根为**复共轭对**，模长都是
    :math:`\sqrt{\omega-1}`，这就是"幂法的增长比在复对下不收敛"的来由
    （见 :func:`power_method`）。
    """
    m = float(mu)
    w = float(omega)
    if not (0.0 <= m < 1.0):
        raise ValueError(f"需要 0 <= mu < 1，得到 {mu}")
    half = 0.5 * w * m
    disc = half * half - (w - 1.0)
    if disc < 0.0:
        return float(w - 1.0)
    root = half + float(np.sqrt(disc))
    return float(root * root)


def young_optimal_omega(mu: float) -> float:
    r"""SOR 最优松弛因子 :math:`\omega_{\rm opt}=2/(1+\sqrt{1-\mu^2})`，此时 :math:`\rho=\omega_{\rm opt}-1`。"""
    m = float(mu)
    if not (0.0 <= m < 1.0):
        raise ValueError(f"需要 0 <= mu < 1，得到 {mu}")
    return float(2.0 / (1.0 + np.sqrt(1.0 - m * m)))


# --------------------------------------------------------------------------- #
# 幂法（手写）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PowerResult:
    r"""幂法结果。``value`` 是谱半径估计 :math:`|\lambda_1|`。"""

    value: float
    signed: float
    vector: np.ndarray
    history: np.ndarray
    n_iter: int
    converged: bool
    residual: float
    mode: str

    def to_json(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "signed": self.signed,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "residual": self.residual,
            "mode": self.mode,
        }


def power_method(
    T: Any,
    *,
    seed: int = 20240105,
    tol: float = 1e-13,
    maxiter: int = 2000,
    x0: Any = None,
) -> PowerResult:
    r"""手写幂法估计 :math:`\rho(T)=|\lambda_1(T)|`。

    迭代 :math:`x_{k+1}=Tx_k/\|Tx_k\|_2`，并同时维护两个估计：

    * **瑞利商** :math:`q_k=x_k^{\mathsf T}Tx_k`（:math:`x_k^{\mathsf T}x_k=1`）——
      实主特征值下它二次收敛，特征残差 :math:`\|Tx_k-q_kx_k\|/\|Tx_k\|` 落到
      ``_RAYLEIGH_TOL`` 以下就采信它；
    * **增长比的几何平均** :math:`\bigl(\prod_{k-W<i\le k}\|Tx_i\|\bigr)^{1/W}`——
      主特征值是**复共轭对**时（SOR 的 :math:`\omega>\omega_{\rm opt}`）范数增长比
      只振荡不收敛，但它的几何平均仍收敛到 :math:`|\lambda_1|`。

    这条"复对兜底"不是装饰：``tests/1.5`` 里对
    :math:`\omega>\omega_{\rm opt}` 的实测谱半径必须等于解析值 :math:`\omega-1`。
    """
    M = _as_square(T)
    n = M.shape[0]
    if x0 is None:
        x = rng(seed).standard_normal(n)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1).copy()
    nx = _norm2(x)
    if nx == 0.0:
        raise ValueError("初始向量不能是零向量")
    x = x / nx

    history: list[float] = []
    prev_gm = float("nan")
    value = 0.0
    signed = 0.0
    res = float("inf")
    n_iter = 0
    converged = False
    mode = "maxiter"

    for k in range(int(maxiter)):
        y = M @ x
        ny = _norm2(y)
        if ny == 0.0:
            return PowerResult(
                value=0.0,
                signed=0.0,
                vector=x,
                history=np.asarray(history, dtype=float),
                n_iter=k,
                converged=True,
                residual=0.0,
                mode="zero",
            )
        history.append(ny)
        x = y / ny
        n_iter = k + 1

        Tx = M @ x
        ntx = _norm2(Tx)
        q = float(x @ Tx)
        signed = q
        res = _norm2(Tx - q * x) / ntx if ntx > 0.0 else 0.0
        if res <= _RAYLEIGH_TOL:
            value = abs(q)
            mode = "rayleigh"
            converged = True
            break

        if len(history) >= 8:
            w = min(len(history), _GEOM_WINDOW)
            gm = float(np.exp(np.mean(np.log(np.asarray(history[-w:], dtype=float)))))
            if np.isfinite(prev_gm) and abs(gm - prev_gm) <= tol * max(gm, 1e-300):
                value = gm
                mode = "geometric"
                converged = True
                break
            prev_gm = gm
            value = gm

    return PowerResult(
        value=float(value),
        signed=float(signed),
        vector=x,
        history=np.asarray(history, dtype=float),
        n_iter=n_iter,
        converged=converged,
        residual=float(res),
        mode=mode,
    )


def spectral_radius(T: Any, **kw: Any) -> float:
    """谱半径 :math:`\\rho(T)`（幂法实测）；其余参数透传给 :func:`power_method`。"""
    return power_method(T, **kw).value


# --------------------------------------------------------------------------- #
# 定常迭代求解器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IterationResult:
    """定常迭代法的完整轨迹（可视化与断言共用同一份数据）。"""

    method: str
    omega: float
    x: np.ndarray
    iterates: np.ndarray
    residuals: np.ndarray
    errors: np.ndarray
    matrix: np.ndarray
    rho: float
    factor_measured: float
    n_iter: int
    converged: bool
    power: PowerResult

    @property
    def n_steps(self) -> int:
        return int(self.iterates.shape[0]) - 1


def convergence_factor(residuals: Any, *, skip: int = 2, tail: int | None = None) -> float:
    r"""由残差序列实测渐近收敛因子 :math:`\hat\rho`：尾部比值的几何平均。

    ``skip`` 丢掉初始瞬态（前 ``skip`` 个残差），``tail`` 只用最后若干个比值。
    与 :mod:`numcore.verify.rates.estimate_linear_factor` 同一口径，但本模块不依赖
    测试基础设施。
    """
    r = np.asarray(residuals, dtype=float).ravel()
    if skip:
        r = r[int(skip) :]
    r = r[np.isfinite(r) & (r > 0.0)]
    if r.size < 2:
        raise ValueError(f"有效残差不足（需 >= 2 个正数），得到 {r.size} 个")
    ratios = r[1:] / r[:-1]
    if tail is not None:
        ratios = ratios[-int(tail) :]
    return float(np.exp(np.mean(np.log(ratios))))


def solve_direct(A: Any, b: Any) -> np.ndarray:
    """直接解 :math:`Ax=b`：部分主元 LU + 回代（P2/P3：**不**调用 ``numpy.linalg``）。"""
    res = lu_decompose(A, pivot="partial")
    if res.singular:
        raise ValueError("矩阵奇异：直接解不可用")
    return lu_solve(res, np.asarray(b, dtype=float).reshape(-1))


def _sweep(A: np.ndarray, b: np.ndarray, x: np.ndarray, *, omega: float, gs: bool) -> None:
    r"""就地做一次 Gauss--Seidel（``gs=True``）或 SOR 前向扫描。

    SOR 的更新式 :math:`x_i \leftarrow (1-\omega)x_i
    +\omega\bigl(b_i-\sum_{j\ne i}a_{ij}x_j\bigr)/a_{ii}`，
    :math:`\omega=1` 即 Gauss--Seidel。这里的手写扫描就是"算法本身"，
    与 :func:`iteration_matrix` 里用 LU 显式求逆是同一数学对象的两条实现路径
    （``tests/1.5`` 会验证二者一致）。
    """
    n = x.size
    for i in range(n):
        row = A[i]
        s = float(row @ x) - float(row[i]) * float(x[i])
        new = (b[i] - s) / row[i]
        x[i] = new if gs else (1.0 - omega) * x[i] + omega * new


def stationary_solve(
    A: Any,
    b: Any,
    *,
    method: Method = "jacobi",
    omega: float = 1.0,
    x0: Any = None,
    maxiter: int = 200,
    tol: float = 1e-12,
    x_exact: Any = None,
) -> IterationResult:
    r"""定常迭代统一入口：Jacobi / Gauss--Seidel / SOR，返回完整轨迹。

    收敛判据用**相对残差** :math:`\|b-Ax^{(k)}\|_2/\|b\|_2\le\text{tol}`；
    ``x_exact`` 给出时同时记录误差 2-范数 :math:`\|x^{(k)}-x^\*\|_2`。
    """
    arr = _as_square(A)
    n = arr.shape[0]
    rhs = np.asarray(b, dtype=float).reshape(-1)
    if rhs.size != n:
        raise ValueError(f"b 的长度应为 {n}，得到 {rhs.size}")
    x = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    if x.size != n:
        raise ValueError(f"x0 的长度应为 {n}，得到 {x.size}")
    d = np.diag(arr)
    if method == "jacobi" and np.any(d == 0.0):
        raise ValueError("Jacobi 需要对角元全非零")
    if method not in ("jacobi", "gs", "sor"):
        raise ValueError(f"未知迭代法 {method!r}")
    w = float(omega) if method == "sor" else 1.0

    nb = _norm2(rhs) or 1.0
    xs = np.asarray(x_exact, dtype=float).reshape(-1) if x_exact is not None else None
    iterates = [x.copy()]
    residuals = [_norm2(rhs - arr @ x) / nb]
    errors = [_norm2(x - xs) if xs is not None else float("nan")]

    for _ in range(int(maxiter)):
        if method == "jacobi":
            r = rhs - arr @ x
            x = x + r / d
        else:
            _sweep(arr, rhs, x, omega=w, gs=(method == "gs"))
        iterates.append(x.copy())
        residuals.append(_norm2(rhs - arr @ x) / nb)
        errors.append(_norm2(x - xs) if xs is not None else float("nan"))
        if residuals[-1] <= tol:
            break

    res_arr = np.asarray(residuals, dtype=float)
    T = iteration_matrix(arr, method=method, omega=w)
    pow_res = power_method(T)
    try:
        fac = convergence_factor(res_arr, skip=2)
    except ValueError:
        fac = float("nan")
    return IterationResult(
        method=method,
        omega=w,
        x=x,
        iterates=np.asarray(iterates, dtype=float),
        residuals=res_arr,
        errors=np.asarray(errors, dtype=float),
        matrix=T,
        rho=pow_res.value,
        factor_measured=fac,
        n_iter=int(res_arr.size - 1),
        converged=bool(res_arr[-1] <= tol),
        power=pow_res,
    )


def jacobi(A: Any, b: Any, **kw: Any) -> IterationResult:
    """Jacobi 迭代（``x^{(k+1)} = x^{(k)} + D^{-1}(b-Ax^{(k)})``）。"""
    return stationary_solve(A, b, method="jacobi", **kw)


def gauss_seidel(A: Any, b: Any, **kw: Any) -> IterationResult:
    """Gauss--Seidel 迭代（前向扫描，就地更新）。"""
    return stationary_solve(A, b, method="gs", **kw)


def sor(A: Any, b: Any, *, omega: float = 1.0, **kw: Any) -> IterationResult:
    """SOR 迭代；``omega=1`` 即 Gauss--Seidel，:math:`\\omega_{\\rm opt}` 处谱半径最小。"""
    return stationary_solve(A, b, method="sor", omega=omega, **kw)


def sor_rho_scan(
    A: Any,
    omegas: Sequence[float] | Any,
    *,
    seed: int = 20240105,
    tol: float = 1e-13,
    maxiter: int = 2000,
) -> np.ndarray:
    """对一串 :math:`\\omega` 逐个用幂法实测 :math:`\\rho(T_\\omega)`（C 类断言的原料）。"""
    ws = np.asarray(list(omegas) if not isinstance(omegas, np.ndarray) else omegas, dtype=float).ravel()
    out = np.zeros(ws.size, dtype=float)
    for i, w in enumerate(ws):
        T = sor_iteration_matrix(A, float(w))
        out[i] = power_method(T, seed=seed, tol=tol, maxiter=maxiter).value
    return out


# --------------------------------------------------------------------------- #
# 误差的特征分量分解
# --------------------------------------------------------------------------- #
def mode_components(errors: Any, *, basis: Any = None) -> np.ndarray:
    r"""把误差序列投影到正弦模态基上：:math:`c_j^{(k)}=v_j^{\mathsf T}e^{(k)}`。

    ``errors`` 形状 ``(k+1, n)``；``basis`` 缺省用 :func:`sine_eigenvectors_1d` 的解析基
    （一维 Dirichlet Laplacian 的特征向量，正交归一）。

    对 **Jacobi**（:math:`T_J=I-\tfrac12A`）该分解是精确的：:math:`c_j^{(k)}=\mu_j^k c_j^{(0)}`，
    因为 :math:`T_J` 与 :math:`A` 同特征向量。对 GS/SOR，:math:`A` 的特征基**不是**
    :math:`T` 的特征基，投影系数只会**渐近**按 :func:`mode_decay_factors` 预测的速率衰减，
    早期步会出现模态混合——这正是可视化里"低模态不听话"的来源，也是非正规性的可见后果。
    """
    E = np.asarray(errors, dtype=float)
    if E.ndim != 2:
        raise ValueError(f"errors 需要 (k+1, n) 形状，得到 {E.shape}")
    V = sine_eigenvectors_1d(E.shape[1]) if basis is None else np.asarray(basis, dtype=float)
    return E @ V


def mode_decay_factors(
    n: int,
    *,
    method: Method = "jacobi",
    omega: float = 1.0,
) -> np.ndarray:
    r"""模型问题（一维 Laplacian）上每个模态的**理论**衰减因子 :math:`|\lambda_j(T)|`。

    * Jacobi：:math:`|\mu_j|`，:math:`\mu_j=\cos(j\pi/(n+1))`；
    * GS：:math:`\mu_j^2`（Young 关系，一致次序矩阵上 :math:`\mu` 映射到 :math:`\mu^2`）；
    * SOR：:func:`sor_spectral_radius` 逐模态作用。

    返回长度 ``n`` 的数组，第 ``j-1`` 项对应模态 :math:`v_j`；:math:`|\\mu_j|` 随 ``j``
    单调递减，所以**索引越小越难衰减**（:math:`\\mu_1` 就是决定收敛快慢的那个模态），
    列顺序与 :func:`mode_components` 的输出列一一对应。
    """
    mu = jacobi_eigenvalues_1d(n)
    if method == "jacobi":
        f = np.abs(mu)
    elif method == "gs":
        f = mu * mu
    elif method == "sor":
        f = np.array([sor_spectral_radius(abs(float(m)), float(omega)) for m in mu], dtype=float)
    else:
        raise ValueError(f"未知迭代法 {method!r}")
    return f
