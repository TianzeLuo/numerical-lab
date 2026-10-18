r"""$\varepsilon$-伪谱、Kreiss 常数与瞬态增长：非正规矩阵到底有多"不规矩"。

数学定义
--------
对 :math:`A \in \mathbb{C}^{n\times n}` 与 :math:`\varepsilon > 0`，:math:`\varepsilon`-伪谱是

.. math::
    \Lambda_\varepsilon(A) \;=\; \{\, z \in \mathbb{C} \;:\; \sigma_{\min}(A - zI) \le \varepsilon \,\}
    \;=\; \bigcup_{\|\delta A\|_2 \le \varepsilon} \Lambda(A + \delta A)
    \;=\; \{\, z : \|(zI-A)^{-1}\|_2 \ge \varepsilon^{-1} \,\},

这三个定义**等价**（定理 3，证明见 ``chapters/1.4/notes.tex``）。第三个式子说明伪谱是
**预解式范数的等高线**；第二个式子说明它是"所有 :math:`\varepsilon` 级扰动下的谱的并"。

对正规矩阵 :math:`\Lambda_\varepsilon` 恰好是 :math:`\varepsilon`-圆盘的并；对非正规矩阵
它可以比谱大出**指数级**（Jordan 块 :math:`J_n(0)` 的伪谱半径是 :math:`\varepsilon^{1/n}`，
:math:`\varepsilon = 10^{-16}` 时半径 :math:`10^{-16/n}`——:math:`n=16` 时是 0.1！）。

三个量
------
=========================  ==========================================================
:math:`\rho_\varepsilon(A)`  :math:`\max\{|z| : z \in \Lambda_\varepsilon\}`
:math:`K(A)`                 Kreiss 常数 :math:`\sup_{k\ge 0}\|A^k\|_2`（有限矩阵上确界可达）
:math:`\max_k\|A^k\|_2`      瞬态增长峰值（与 :math:`K` 相等，见定理 4）
=========================  ==========================================================

:math:`K(A)` 与伪谱的关系（Kreiss 矩阵定理）：

.. math::
    K(A) \;=\; \sup_{\varepsilon>0}\frac{\rho_\varepsilon(A)-1}{\varepsilon}
          \;=\; \sup_{|z|>1}(|z|-1)\,\|(zI-A)^{-1}\|_2 ,

而瞬态增长由预解式界住：:math:`\|A^k\|_2 \le \dfrac{(1+\varepsilon)^{k+1}}{\varepsilon}\rho_\varepsilon(A)`
当 :math:`\rho_\varepsilon(A) < 1`。**这条不等式解释了一个反直觉现象**：
:math:`\rho(A) < 1`（谱半径小于 1，长期必然衰减）却可以有 :math:`\|A^k\|` 先增长几个量级——
只要伪谱鼓出单位圆外。

$\sigma_{\min}$ 怎么算（本章的核心工程问题）
-------------------------------------------
:math:`\sigma_{\min}` 的定义要用 SVD，而 SVD 在本仓库是**被禁的成品算法**（P2/P3：``svd`` 在
:data:`tools.check_policy.P3_ATTRS` 里）。所以这里走**手算路线**：

.. math::
    \sigma_{\min}(M) = \frac{1}{\|M^{-1}\|_2}
    \quad\Longrightarrow\quad
    \sigma_{\min}^2(M) = \frac{1}{\lambda_{\max}\!\big((M^{\mathsf H}M)^{-1}\big)} .

于是只要 (a) 用**自研 LU**（:mod:`numcore.linalg.lu`）解 :math:`M x = v`，
(b) 对 :math:`(M^{\mathsf H}M)^{-1}` 做**幂法**（:mod:`numcore.linalg.eigen`），
就能得到 :math:`\sigma_{\min}` 的估计。每步的 Rayleigh 商给出**下界**
（Rayleigh 商是 :math:`\lambda_{\max}` 的下界，取倒数则是 :math:`\sigma_{\min}^2` 的上界），
迭代收敛后它是精确值。

.. warning::
    **这是估计，不是定理。** 收敛因子是 :math:`(\sigma_2/\sigma_1)^2`（:math:`\sigma_1 \le \sigma_2`
    是两个最小的奇异值），所以**奇异值成簇时收敛慢**；矩阵越病态（:math:`\kappa` 越大），
    需要的迭代次数越多、相对误差越大。本模块的 :func:`sigma_min` 返回
    ``converged`` 标志与 ``n_iter``，章节帧把"有多少格点没收敛"如实报给前端。
    量级经验：:math:`n \le 16`、``iters=30`` 时相对误差通常在 :math:`10^{-8}` 以内；
    但 :math:`\sigma_{\min} \lesssim u\|A\|` 的格点上**任何**算法都不可信
    （那里 :math:`A-zI` 在浮点意义下已奇异）。

主要接口
--------
* :func:`sigma_min` / :func:`sigma_min_grid` —— :math:`\sigma_{\min}(A-zI)` 的单点与网格值
* :func:`pseudospectrum_grid` —— 伪谱"云雾"（含 :math:`\rho_\varepsilon` 与等高线）
* :func:`marching_squares` —— 从网格提取等值线（给前端画轮廓，避免前端自造数值）
* :func:`kreiss_constant` / :func:`transient_growth` —— Kreiss 常数与 :math:`\|A^k\|_2`
* :func:`gmres` —— 手写 GMRES（Arnoldi + Givens），为 1.5 章铺路
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ComplexLU",
    "complex_lu",
    "complex_solve",
    "SigmaMinResult",
    "sigma_min",
    "sigma_min_grid",
    "spectral_distance_grid",
    "pseudospectrum_grid",
    "PseudospectrumGrid",
    "marching_squares",
    "pseudospectral_radius",
    "pseudospectrum_curve",
    "kreiss_constant",
    "kreiss_from_pseudospectrum",
    "transient_growth",
    "TransientResult",
    "matrix_powers",
    "operator_2norm",
    "GMRESResult",
    "gmres",
    "gmres_arnoldi_residual_bound",
    "jordan_boundary_radius",
    "grrcar_like_matrix",
    "EXAMPLE_PSEUDOSPECTRUM_MATRICES",
    "example_pseudospectrum_matrix",
]


# --------------------------------------------------------------------------- #
# 复数稠密 LU（自研：复数在 P2/P3 下同样不许调用 numpy.linalg）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ComplexLU:
    """复数矩阵 :math:`M = LU`（部分主元）的紧凑布局。

    ``M`` 的行置换记录在 ``perm`` 里（``LU = P M``），``L`` 单位下三角、``U`` 上三角。
    """

    LU: np.ndarray
    perm: np.ndarray
    n: int
    n_swaps: int
    singular: bool

    def L(self) -> np.ndarray:
        return np.tril(self.LU, -1) + np.eye(self.n, dtype=complex)

    def U(self) -> np.ndarray:
        return np.triu(self.LU)


def complex_lu(A: Any) -> ComplexLU:
    r"""复数稠密 LU 分解（部分主元），:math:`PA = LU`。

    为什么不用 :func:`numcore.linalg.lu.lu_decompose`：那个函数的回代与列置换是为实数
    设计的（``np.zeros(n, dtype=float)``），把复数塞进去会被静默截断成实部。
    复数路径在伪谱里必须存在——:math:`A - zI` 对复的 :math:`z` 就是复矩阵。
    """
    arr = np.asarray(A, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    n = arr.shape[0]
    if n == 0:
        raise ValueError("空矩阵没有 LU 分解")
    M = np.array(arr, dtype=complex, copy=True)
    perm = np.arange(n)
    n_swaps = 0
    scale = float(np.max(np.abs(M))) if M.size else 0.0
    tol = 1e-14 * max(1.0, scale)
    singular = False
    for k in range(n):
        piv = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[piv, k]) <= tol:
            singular = True
        if piv != k:
            M[[k, piv], :] = M[[piv, k], :]
            perm[[k, piv]] = perm[[piv, k]]
            n_swaps += 1
        akk = M[k, k]
        if akk != 0.0:
            rows = np.arange(k + 1, n)
            m = M[rows, k] / akk
            M[rows, k] = m
            M[rows, k + 1 :] -= np.outer(m, M[k, k + 1 :])
    return ComplexLU(LU=M, perm=perm, n=n, n_swaps=n_swaps, singular=singular)


def complex_solve(fac: ComplexLU, b: Any) -> np.ndarray:
    r"""解 :math:`Mx=b`（前代 + 回代），要求 :math:`M` 已经 :func:`complex_lu` 分解。"""
    n = fac.n
    bv = np.asarray(b, dtype=complex).reshape(-1)
    if bv.size != n:
        raise ValueError(f"b 的长度应为 {n}，得到 {bv.size}")
    y = bv[fac.perm].astype(complex)
    L = np.tril(fac.LU, -1)
    U = np.triu(fac.LU)
    for i in range(n):
        if i:
            y[i] -= L[i, :i] @ y[:i]
    x = np.zeros(n, dtype=complex)
    for i in range(n - 1, -1, -1):
        acc = y[i] - U[i, i + 1 :] @ x[i + 1 :]
        diag = U[i, i]
        if diag == 0.0:
            raise ZeroDivisionError(f"U 的第 {i} 个对角元为 0：矩阵奇异")
        x[i] = acc / diag
    return x


# --------------------------------------------------------------------------- #
# sigma_min：1/||M^{-1}||_2 的幂法估计
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SigmaMinResult:
    r""":math:`\sigma_{\min}(M)` 的估计与诊断。

    字段含义：

    * ``value`` —— 估计值；``converged``/``n_iter`` 说明它是否已稳定；
    * ``at_noise_floor`` —— **重要**：:math:`\sigma_{\min}` 已经掉到
      :math:`\sqrt{u}\,\|M\|\approx10^{-8}\|M\|` 的噪声地板以下
      （拿 :math:`M^{\mathsf H}M` 的谱去分辨 :math:`\sigma_{\min}` 时，
      条件数被平方，:math:`\kappa(M)\gtrsim10^{8}` 就再也分不出来了）。
      这类格点的数字**不可信**，前端必须单独标注而不是照常上色；
    * ``bound_upper`` —— 无条件上界 :math:`|t_{nn}-z|`（:math:`t_{nn}` 是 Schur 形末对角元），
      它立刻给出伪谱的**外近似**：:math:`|t_{nn}-z|>\varepsilon\Rightarrow z\notin\Lambda_\varepsilon`；
    * ``bound_lower`` —— 由 :math:`\prod_i\sigma_i=|\det M|` 与
      :math:`\sigma_i\le\|M\|_F` 得到的 :math:`|\det M|/\|M\|_F^{\,n-1}`
      （**无条件**：把除最小之外的全部奇异值都用 Frobenius 范数顶上去）。
      :math:`\|A\|_F` 在 :math:`10^2` 量级的这些问题上几乎免费，
      而它比"几何平均"版本紧得多（实测在小主元附近能差几个量级）。
    """

    value: float
    converged: bool
    n_iter: int
    residual: float
    kappa_estimate: float
    bound_lower: float
    bound_upper: float
    singular: bool
    at_noise_floor: bool = False


def _frobenius(M: np.ndarray) -> float:
    r"""Frobenius 范数 :math:`\|M\|_F = \sqrt{\sum_{ij}|m_{ij}|^2}`（先缩放再开方）。"""
    if M.size == 0:
        return 0.0
    m = float(np.max(np.abs(M)))
    if m == 0.0:
        return 0.0
    return m * float(np.sqrt(np.sum((np.abs(M) / m) ** 2)))


def sigma_min(
    A: Any,
    z: complex = 0.0,
    *,
    max_sweeps: int = 40,
    method: str = "hermitian",
    iters: int = 60,
    tol: float = 1e-12,
    z0: Any | None = None,
) -> SigmaMinResult:
    r""":math:`\sigma_{\min}(A-zI)` —— **不用 SVD**（``svd`` 是 P2/P3 禁用的成品算法），两条自研路线。

    ``method="hermitian"``（缺省）
        对 :math:`M^{\mathsf H}M`（:math:`M=A-zI`）做 **Hermite Jacobi** 特征分解
        （:func:`numcore.linalg.eigen.hermitian_jacobi_eigh`），
        :math:`\sigma_{\min}=\sqrt{\lambda_{\min}(M^{\mathsf H}M)}`。
        循环 Jacobi 的收敛是**二次**的，所以 6–10 轮扫描就到机器精度；
        **代价**是正规方程把条件数平方（:math:`\kappa(M^{\mathsf H}M)=\kappa(M)^2`），
        于是 :math:`\sigma_{\min}\lesssim\sqrt{u}\,\|M\|` 之后就进了舍入噪声——
        ``at_noise_floor`` 把这件事如实标出来。

    ``method="power"``（教学对照：把"估计"两个字演出来）
        :math:`\sigma_{\min}=1/\|M^{-1}\|_2`，对 :math:`(M^{\mathsf H}M)^{-1}` 做幂法
        （每步两次自研三角回代）。Rayleigh 商是 :math:`\lambda_{\max}` 的**下界**，
        所以每一步给出的 :math:`\sigma` 都是**上界**，取迭代过程中的最小值即单调改进。
        矩阵病态时它比 Hermite 路线更差，但它演示了"逆迭代"这条标准路线的代价。

    两个无条件界：:math:`|\det M|^{1/n}\le\sigma_{\min}(M)\le|t_{nn}|`。

    :param z0: 幂法路线的起点（缺省取确定性向量；同一网格上相邻点起点相同 ⇒ 场光滑）。
    """
    arr = np.asarray(A, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    n = arr.shape[0]
    M = arr - complex(z) * np.eye(n, dtype=complex)
    fro = _frobenius(M)
    fac = complex_lu(M)
    diag = np.zeros(n) if fac.singular else np.abs(np.diag(fac.U()))
    # 下界：|det M| / ||M||_F^{n-1}（det 由 LU 对角元乘积给出，符号不影响 |det|）
    logdet = float(np.sum(np.log(np.maximum(diag, 1e-300))))
    bound_lower = float(np.exp(logdet - (n - 1) * np.log(max(fro, 1e-300)))) if n else 0.0
    from numcore.linalg.eigen import qr_iteration

    schur = qr_iteration(M, shift="wilkinson", max_iter=400)
    bound_upper = float(abs(complex(np.asarray(schur.H)[-1, -1]))) if n else 0.0

    if method == "hermitian":
        from numcore.linalg.eigen import hermitian_jacobi_eigh

        G = M.conj().T @ M
        w, V = hermitian_jacobi_eigh(G, max_sweeps=max_sweeps)
        smin = float(np.sqrt(max(w[0], 0.0))) if w.size else 0.0
        smax = float(np.sqrt(max(w[-1], 0.0))) if w.size else 0.0
        floor = float(np.sqrt(np.finfo(float).eps))
        at_floor = bool(smax > 0.0 and smin <= floor * smax)
        v = V[:, 0] if V.size else np.zeros(n, dtype=complex)
        lam = smin * smin
        resid = float(np.sqrt(np.sum(np.abs(G @ v - lam * v) ** 2)) / max(lam, 1e-300)) if smin > 0.0 else 0.0
        return SigmaMinResult(
            value=smin,
            converged=not at_floor,
            n_iter=int(max_sweeps),
            residual=resid,
            kappa_estimate=(smax / smin) if smin > 0.0 else float("inf"),
            bound_lower=bound_lower,
            bound_upper=bound_upper,
            singular=bool(fac.singular),
            at_noise_floor=at_floor,
        )

    if method != "power":
        raise ValueError(f"未知 method {method!r}（可用 'hermitian' / 'power'）")
    if fac.singular:
        return SigmaMinResult(
            value=0.0, converged=True, n_iter=0, residual=0.0, kappa_estimate=float("inf"),
            bound_lower=bound_lower, bound_upper=bound_upper, singular=True, at_noise_floor=False,
        )
    if z0 is None:
        v = np.ones(n, dtype=complex) + 0.5 * np.exp(2j * np.pi * np.arange(n) / n)
    else:
        v = np.asarray(z0, dtype=complex).reshape(-1)
    nv = float(np.sqrt(np.sum(np.abs(v) ** 2)))
    v = v / nv if nv > 0.0 else v
    theta = 0.0
    theta_prev = 0.0
    converged = False
    used = 0
    best = float("inf")
    for k in range(max(1, int(iters))):
        u = complex_solve(_adjoint_factor(fac), v)
        x = complex_solve(fac, u)
        theta = float(np.real(np.vdot(v, x)))
        nx = float(np.sqrt(np.sum(np.abs(x) ** 2)))
        if nx == 0.0 or not np.isfinite(nx) or theta <= 0.0:
            break
        best = min(best, 1.0 / float(np.sqrt(theta)))
        v = x / nx
        used = k + 1
        if k > 0 and abs(theta - theta_prev) <= tol * max(1.0, abs(theta)):
            converged = True
            break
        theta_prev = theta
    value = float(best) if np.isfinite(best) else float("inf")
    return SigmaMinResult(
        value=value,
        converged=bool(converged),
        n_iter=used,
        residual=0.0,
        kappa_estimate=(fro / value) if (value > 0.0 and np.isfinite(value)) else float("inf"),
        bound_lower=bound_lower,
        bound_upper=bound_upper,
        singular=bool(fac.singular),
        at_noise_floor=bool(np.isfinite(value) and value <= np.sqrt(np.finfo(float).eps) * fro),
    )


def _adjoint_factor(fac: ComplexLU) -> ComplexLU:
    r"""把 :math:`M` 的 LU 因子转成 :math:`M^{\mathsf H}` 的 LU 因子。

    :math:`M = P^{\mathsf T}LU` ⇒ :math:`M^{\mathsf H} = U^{\mathsf H}L^{\mathsf H}P`，
    所以 :math:`M^{\mathsf H}` 的"U 因子"是 :math:`L^{\mathsf H}`（单位下三角的共轭转置
    = 单位上三角），"L 因子"是 :math:`U^{\mathsf H}`（下三角）。置换在求解时用
    ``perm`` 的逆序处理（见 :func:`complex_solve_adjoint`）。
    """
    Lu = fac.LU
    L = np.tril(Lu, -1) + np.eye(fac.n, dtype=complex)
    U = np.triu(Lu)
    # M^H 的紧凑布局：下三角放 U^H 的严格下三角（= conj(U) 的严格上三角的转置），
    # 上三角放 L^H。这样 complex_solve 的"先前代再回代"依旧成立。
    return ComplexLU(LU=U.conj().T, perm=np.arange(fac.n), n=fac.n, n_swaps=0, singular=fac.singular)


def complex_solve_adjoint(fac: ComplexLU, b: Any) -> np.ndarray:
    r"""解 :math:`M^{\mathsf H}x=b`（用 :func:`_adjoint_factor` 造出的因子 + 反序置换）。"""
    y = np.asarray(b, dtype=complex).reshape(-1)[np.argsort(fac.perm)]
    af = _adjoint_factor(fac)
    return complex_solve(af, y)


def sigma_min_grid(
    A: Any,
    xs: Any,
    ys: Any,
    *,
    max_sweeps: int = 40,
    iters: int = 26,
    tol: float = 1e-11,
    method: str = "hermitian",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""在网格 :math:`z=x+iy` 上逐点估 :math:`\sigma_{\min}(A-zI)`。

    返回 ``(sigma, trusted, upper)``：

    * ``sigma`` —— :math:`\sigma_{\min}` 的估计（:math:`n_y\times n_x`，行 = y）；
    * ``trusted`` —— 0/1 图：该格点是否**可信**（没掉到 :math:`\sqrt{u}` 的噪声地板以下）；
    * ``upper`` —— 逐点无条件上界 :math:`|t_{nn}-z|`（Schur 形末对角元），
      它给出伪谱的**外近似**：:math:`\sigma_{\min}>\varepsilon\Rightarrow z\notin\Lambda_\varepsilon`。

    为什么前端必须拿到 ``trusted``：:math:`\kappa(M)\gtrsim10^{8}` 的格点上
    :math:`\sigma_{\min}` 的**任何**算法都算不准（信息已经在浮点里丢了），
    把不可信的格点和可信的一起上色，就是让图撒谎（POLICY 规则 1/3）。

    第三个参数 ``method`` 与 :func:`sigma_min` 同义（``"hermitian"`` 缺省 /
    ``"power"`` 教学对照）；网格法默认走 Hermite 路线，因为它在整张网格上是
    **确定性且光滑**的——幂法在相邻格点上的迭代步数不同，会让场出现"锯齿"，
    而伪谱的等高线恰恰对场的抖动极其敏感（``chapters/1.4/pitfalls.md`` 第 5 条）。

    实现：:math:`A` 的**复 Schur 形**先算一次（:func:`numcore.linalg.eigen.qr_iteration`），
    之后每点只处理三角阵 :math:`T-zI`；:math:`\sigma_{\min}` 在酉相似下不变，
    所以与直接对 :math:`A-zI` 算完全一致（``tests/1.4`` 里有对拍断言）。
    """
    from numcore.linalg.eigen import hermitian_jacobi_eigh, qr_iteration

    A0 = np.asarray(A, dtype=complex)
    if A0.ndim != 2 or A0.shape[0] != A0.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {A0.shape}")
    if method not in ("hermitian", "power"):
        raise ValueError(f"未知 method {method!r}（可用 'hermitian' / 'power'）")
    xv = np.asarray(xs, dtype=float).reshape(-1)
    yv = np.asarray(ys, dtype=float).reshape(-1)
    n = A0.shape[0]
    schur = qr_iteration(A0, shift="wilkinson", max_iter=400)
    T = np.asarray(schur.H, dtype=complex)
    sig = np.zeros((yv.size, xv.size), dtype=float)
    trusted = np.zeros((yv.size, xv.size), dtype=float)
    upper = np.zeros((yv.size, xv.size), dtype=float)
    floor = float(np.sqrt(np.finfo(float).eps))
    for j, y in enumerate(yv):
        for i, x in enumerate(xv):
            z = complex(x, y)
            M = T - z * np.eye(n, dtype=complex)
            fac = complex_lu(M)
            diag = np.abs(np.diag(fac.U())) if not fac.singular else np.zeros(n)
            upper[j, i] = float(diag[-1]) if n else 0.0
            if fac.singular:
                # ★ 格点**正好落在特征值上**时 M = T - zI 精确奇异，此时 σ_min = 0 是
                #   **最可信**的值（不是噪声）。早先把这里标成 trusted=0，结果伪谱在谱点处
                #   出现"洞"——恰好是本该最亮的地方。这是本仓库踩过的一个真 bug。
                sig[j, i] = 0.0
                trusted[j, i] = 1.0
                continue
            if method == "power":
                val, conv, _used = _sigma_min_from_factor(fac, fro=_frobenius(M), iters=iters, tol=tol)
                sig[j, i] = val
                trusted[j, i] = 1.0 if (conv and np.isfinite(val)) else 0.0
                continue
            G = M.conj().T @ M
            w, _V = hermitian_jacobi_eigh(G, max_sweeps=max_sweeps)
            smin = float(np.sqrt(max(w[0], 0.0))) if w.size else 0.0
            smax = float(np.sqrt(max(w[-1], 0.0))) if w.size else 0.0
            sig[j, i] = smin
            trusted[j, i] = 0.0 if (smax > 0.0 and smin <= floor * smax) else 1.0
    return sig, trusted, upper


def _sigma_min_from_factor(fac: ComplexLU, *, fro: float, iters: int, tol: float) -> tuple[float, bool, int]:
    r"""已分解好的 :math:`M`：跑幂法估 :math:`\sigma_{\min}`，返回 ``(值, 是否收敛, 步数)``。"""
    n = fac.n
    v = np.ones(n, dtype=complex) + 0.5 * np.exp(2j * np.pi * np.arange(n) / n)
    v = v / float(np.sqrt(np.sum(np.abs(v) ** 2)))
    theta = 0.0
    theta_prev = 0.0
    converged = False
    used = 0
    for k in range(max(1, iters)):
        u = complex_solve(_adjoint_factor(fac), v)
        x = complex_solve(fac, u)
        theta = float(np.real(np.vdot(v, x)))
        nx = float(np.sqrt(np.sum(np.abs(x) ** 2)))
        if nx == 0.0 or not np.isfinite(nx):
            break
        v = x / nx
        used = k + 1
        if k > 0 and abs(theta - theta_prev) <= tol * max(1.0, abs(theta)):
            converged = True
            break
        theta_prev = theta
    val = 1.0 / np.sqrt(theta) if theta > 0.0 else float("inf")
    return float(val), bool(converged), used


def spectral_distance_grid(A: Any, xs: Any, ys: Any, eigenvalues: Any | None = None) -> np.ndarray:
    r"""逐点到谱的最小距离 :math:`\operatorname{dist}(z,\Lambda(A))`。

    它给出伪谱的**必要条件**：:math:`\sigma_{\min}(A-zI) \le \operatorname{dist}(z,\Lambda(A))`
    （取 :math:`z` 最近的特征值 :math:`\lambda`，则 :math:`A-zI` 有特征值 :math:`\lambda-z`，
    而 :math:`\sigma_{\min}\le|\lambda-z|`）。所以 :math:`\Lambda_\varepsilon` **必然**含在
    "以每个特征值为心、半径 :math:`\varepsilon` 的圆盘之并"里——超出这个并集的部分
    就是非正规性"凭空造出来"的谱。
    """
    from numcore.linalg.eigen import dense_eigenvalues

    A0 = np.asarray(A, dtype=complex)
    lam = np.asarray(dense_eigenvalues(A0), dtype=complex).reshape(-1) if eigenvalues is None else np.asarray(eigenvalues, dtype=complex).reshape(-1)
    xv = np.asarray(xs, dtype=float).reshape(-1)
    yv = np.asarray(ys, dtype=float).reshape(-1)
    out = np.zeros((yv.size, xv.size), dtype=float)
    for j, y in enumerate(yv):
        for i, x in enumerate(xv):
            out[j, i] = float(np.min(np.abs(lam - complex(x, y))))
    return out


# --------------------------------------------------------------------------- #
# 伪谱网格与等值线
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PseudospectrumGrid:
    r"""一次伪谱扫描的全部产物。

    ``sigma`` 是 :math:`\\sigma_{\min}(A-zI)` 的网格值（行 = y，列 = x），
    ``inside`` 是逐 :math:`\varepsilon` 的 0/1 掩码（:math:`\\sigma \le \\varepsilon`），
    ``radius`` 是对应的伪谱半径 :math:`\rho_\varepsilon`（**网格分辨率下的实测值**，
    :math:`\rho_\varepsilon` 越小越受网格限制，见 :func:`pseudospectral_radius` 的说明）。
    """

    xs: np.ndarray
    ys: np.ndarray
    sigma: np.ndarray
    distance: np.ndarray
    epsilons: np.ndarray
    inside: np.ndarray
    radius: np.ndarray
    n_converged: int
    n_points: int
    kappa_max: float

    @property
    def converged_fraction(self) -> float:
        return float(self.n_converged) / float(self.n_points) if self.n_points else 0.0


def pseudospectrum_grid(
    A: Any,
    *,
    nx: int = 48,
    ny: int = 48,
    window: float = 1.5,
    epsilons: Any = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12),
    iters: int = 26,
    tol: float = 1e-11,
    xs: Any | None = None,
    ys: Any | None = None,
) -> PseudospectrumGrid:
    r"""在 :math:`[-w,w]^2`（默认 :math:`w=1.5`）上算 :math:`\sigma_{\min}(A-zI)` 与各 :math:`\varepsilon` 的伪谱。

    "云雾"就是 :math:`\log_{10}\sigma_{\min}` 的场：:math:`\Lambda_\varepsilon` 是它的
    :math:`\log_{10}\varepsilon` 等值线内的区域。默认的 :math:`\varepsilon` 序列跨 12 个量级，
    是为了让"正规矩阵纹丝不动 / 非正规矩阵整片亮起"这个对比在滑杆上真的能演出来。

    ``window`` 是**固定**的（不随 :math:`\varepsilon` 自适应）：自适应坐标会把
    "云雾扩散"这类几何现象抹平成"图还是那么大"（POLICY 规则 1/4）。

    ``xs``/``ys`` 允许调用方直接给网格坐标（缺省时用 ``window``/``nx``/``ny`` 生成均匀网格）；
    两条路径都必须给同一组 :math:`(x,y)`，因为 :math:`\sigma_{\min}` 场与
    :func:`pseudospectral_radius`、:func:`marching_squares` 共享这套坐标。
    """
    A0 = np.asarray(A, dtype=complex)
    eps = np.asarray(sorted(set(float(e) for e in np.asarray(epsilons, dtype=float).reshape(-1))), dtype=float)
    if eps.size == 0 or float(np.min(eps)) <= 0.0:
        raise ValueError("epsilons 必须是非空的正数序列")
    w = float(window)
    if w <= 0.0:
        raise ValueError(f"window 必须为正，得到 {w}")
    if xs is not None or ys is not None:
        # 显式网格（测试用：可非均匀、可比 window 更细）。两侧必须同时给。
        if xs is None or ys is None:
            raise ValueError("xs 与 ys 必须同时给出")
        xs = np.asarray(xs, dtype=float).reshape(-1)
        ys = np.asarray(ys, dtype=float).reshape(-1)
        if xs.size < 2 or ys.size < 2:
            raise ValueError("xs/ys 至少各要两个点")
    else:
        xs = np.linspace(-w, w, int(nx))
        ys = np.linspace(-w, w, int(ny))
    sig, trusted, _upper = sigma_min_grid(A0, xs, ys, iters=iters, tol=tol)
    dist = spectral_distance_grid(A0, xs, ys)
    inside = np.zeros((eps.size,) + sig.shape, dtype=float)
    radius = np.zeros(eps.size, dtype=float)
    for k, e in enumerate(eps):
        mask = ((sig <= e) & (trusted > 0.5)).astype(float)
        inside[k] = mask
        radius[k] = pseudospectral_radius(mask, xs, ys)
    # kappa 的最大值：用 sigma 的最小正格点估计（不含 0）
    pos = sig[sig > 0.0]
    kappa_max = float(_frobenius(A0) / float(np.min(pos))) if pos.size else float("inf")
    return PseudospectrumGrid(
        xs=xs,
        ys=ys,
        sigma=sig,
        distance=dist,
        epsilons=eps,
        inside=inside,
        radius=radius,
        n_converged=int(np.sum(trusted > 0.5)),
        n_points=int(sig.size),
        kappa_max=kappa_max,
    )


def pseudospectral_radius(mask: Any, xs: Any, ys: Any) -> float:
    r"""伪谱半径 :math:`\rho_\varepsilon = \max\{|z| : \sigma_{\min}(A-zI)\le\varepsilon\}`（离散实测）。

    **它是网格分辨率的函数**：若真实伪谱半径小于两个格点的间距，实测值会被**系统性低估**到
    0 或一个格点距。所以断言里要么用足够细的网格，要么用解析例子
    （Jordan 块：:math:`\rho_\varepsilon = \varepsilon^{1/n}`，见 :func:`jordan_boundary_radius`）。
    """
    m = np.asarray(mask, dtype=float)
    xv = np.asarray(xs, dtype=float).reshape(-1)
    yv = np.asarray(ys, dtype=float).reshape(-1)
    if m.shape != (yv.size, xv.size):
        raise ValueError(f"mask 形状 {m.shape} 与网格 ({yv.size},{xv.size}) 不符")
    if not np.any(m > 0.5):
        return 0.0
    R = np.sqrt(xv.reshape(1, -1) ** 2 + yv.reshape(-1, 1) ** 2)
    return float(np.max(R[m > 0.5]))


def marching_squares(values: Any, xs: Any, ys: Any, level: float) -> list[np.ndarray]:
    r"""从网格提取等值线（marching squares），返回线段列表（每条形状 ``(2,2)``，:math:`(x,y)`）。

    为什么在后端做：前端**不许自造数值**（POLICY 规则 5、P8/P9）。轮廓线的位置是
    "数值结论"的一部分（它决定读者看到伪谱边界在哪），必须由 :mod:`numcore` 给出。
    每段线段由线性插值决定，所以误差是 :math:`O(h^2)`（:math:`h` 是网格间距）——
    在 :math:`\sigma_{\min}` 变化剧烈的角点（Jordan 块的"花瓣尖"）会偏，
    这是网格法的固有代价，不是 bug。

    实现取标准 16 种情形的**线性插值**版本：对每个网格小方格，检查四个角是否越过
    ``level``，在跨越的边上按线性插值放点，再按情形连成 0、1 或 2 条线段。
    """
    v = np.asarray(values, dtype=float)
    xv = np.asarray(xs, dtype=float).reshape(-1)
    yv = np.asarray(ys, dtype=float).reshape(-1)
    if v.shape != (yv.size, xv.size):
        raise ValueError(f"values 形状 {v.shape} 与网格 ({yv.size},{xv.size}) 不符")
    segs: list[np.ndarray] = []
    for j in range(yv.size - 1):
        for i in range(xv.size - 1):
            # 四个角（逆时针：左下、右下、右上、左上）
            corners = np.array(
                [
                    [xv[i], yv[j], v[j, i]],
                    [xv[i + 1], yv[j], v[j, i + 1]],
                    [xv[i + 1], yv[j + 1], v[j + 1, i + 1]],
                    [xv[i], yv[j + 1], v[j + 1, i]],
                ],
                dtype=float,
            )
            pts: list[tuple[float, float]] = []
            edges = ((0, 1), (1, 2), (2, 3), (3, 0))
            for a, b in edges:
                va, vb = corners[a, 2] - level, corners[b, 2] - level
                if va == 0.0 and vb == 0.0:
                    continue
                if (va <= 0.0 < vb) or (vb <= 0.0 < va) or va == 0.0 or vb == 0.0:
                    t = 0.0 if va == 0.0 else (1.0 if vb == 0.0 else va / (va - vb))
                    t = min(1.0, max(0.0, t))
                    pts.append(
                        (
                            float(corners[a, 0] + t * (corners[b, 0] - corners[a, 0])),
                            float(corners[a, 1] + t * (corners[b, 1] - corners[a, 1])),
                        )
                    )
            if len(pts) == 2:
                segs.append(np.asarray(pts, dtype=float))
            elif len(pts) == 4:  # 鞍点：按中心值决定配对方式
                center = float(np.mean(corners[:, 2]))
                if center >= level:
                    pairs = ((0, 1), (2, 3))
                else:
                    pairs = ((0, 3), (1, 2))
                for a, b in pairs:
                    segs.append(np.asarray([pts[a], pts[b]], dtype=float))
    return segs


def pseudospectrum_curve(A: Any, epsilon: float, *, nx: int = 96, ny: int = 96, window: float = 1.5, iters: int = 30) -> np.ndarray:
    r"""单个 :math:`\varepsilon` 的伪谱边界线段（``marching_squares`` 的便捷封装）。

    返回形状 ``(k,2,2)`` 的数组（k 条线段）；前端直接画，不需要自己找边界。
    """
    g = pseudospectrum_grid(A, nx=nx, ny=ny, window=window, epsilons=(epsilon,), iters=iters)
    segs = marching_squares(np.log10(np.maximum(g.sigma, 1e-300)), g.xs, g.ys, float(np.log10(epsilon)))
    if not segs:
        return np.zeros((0, 2, 2), dtype=float)
    return np.asarray(segs, dtype=float)


# --------------------------------------------------------------------------- #
# Kreiss 常数与瞬态增长
# --------------------------------------------------------------------------- #
def matrix_powers(A: Any, kmax: int) -> np.ndarray:
    r"""矩阵幂序列 :math:`A^0, A^1, \dots, A^{k_{\max}}`（返回形状 ``(kmax+1, n, n)``）。

    只用 ``@`` 连乘（自研；``numpy.linalg.matrix_power`` 与 ``expm`` 都是被禁的成品）。
    :math:`A^0 = I`。
    """
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    k = int(kmax)
    if k < 0:
        raise ValueError(f"kmax 必须非负，得到 {k}")
    n = arr.shape[0]
    out = np.zeros((k + 1, n, n), dtype=float)
    out[0] = np.eye(n)
    for i in range(1, k + 1):
        out[i] = arr @ out[i - 1]
    return out


def operator_2norm(A: Any, *, iters: int = 120, tol: float = 1e-13, seed: int = 5) -> float:
    r""":math:`\|A\|_2` 的幂法估计（对 :math:`A^{\mathsf T}A` 做幂法；不用 SVD）。

    与 :func:`numcore.linalg.eigen.matrix_2norm` 是同一个算法，这里保留一个薄封装是因为
    ``transient_growth`` 要在 :math:`10^2` 量级的矩阵上调用它——参数需要按规模调。
    """
    from numcore.linalg.eigen import matrix_2norm

    return matrix_2norm(A, iters=iters, tol=tol, seed=seed)


@dataclass(frozen=True)
class TransientResult:
    r"""瞬态增长的完整诊断。

    ``norms`` 是 :math:`\|A^k\|_2`（:math:`k=0..k_{\max}`，**实测**），
    ``peak``/``peak_index`` 是它的最大值与位置，
    ``spectral_prediction`` 是"只看谱半径"的预测 :math:`\rho(A)^k`，
    ``kreiss`` 与 ``peak`` 在理论上相等（Kreiss 矩阵定理的有限矩阵版本），
    ``growth_ratio`` :math:`= \text{peak}/\rho(A)^{\text{peak\_index}}` 是"谱预测被低估的倍数"。
    """

    norms: np.ndarray
    spectral_prediction: np.ndarray
    peak: float
    peak_index: int
    spectral_radius: float
    growth_ratio: float
    kreiss: float
    asymptotic_factor: float
    k_max: int
    non_normal: bool


def transient_growth(
    A: Any,
    *,
    k_max: int = 60,
    iters: int = 80,
    seed: int = 5,
) -> TransientResult:
    r"""瞬态增长：算 :math:`\|A^k\|_2`（:math:`k=0..k_{\max}`）并与谱半径预测对比。

    **核心现象**：:math:`\rho(A) < 1`（谱半径小于 1）只能保证 :math:`\|A^k\|\to 0`，
    它**不排除先增长几个量级**。增长的量级由 Kreiss 常数
    :math:`K(A)=\sup_k\|A^k\|_2` 控制，而 :math:`K` 又由伪谱决定：

    .. math::
        K(A)\ \ge\ \frac{\rho_\varepsilon(A)-1}{\varepsilon}\quad(\rho_\varepsilon>1),
        \qquad
        \|A^k\|_2 \le \frac{(1+\varepsilon)^{k+1}}{\varepsilon}\,\rho_\varepsilon(A) .

    对 Jordan 块 :math:`J_n(0)`（谱半径 0！）有 :math:`\|J_n^k\|_2 = 1`（:math:`1\le k\le n-1`）
    甚至 :math:`\|J_n^{n-1}\|_2 = 1`，而剪切矩阵 :math:`I+sN` 的峰值是
    :math:`\sum_j\binom{k}{j}s^j` 的最大值——随手就能到 :math:`10^3` 以上。

    ``asymptotic_factor`` 是尾部 :math:`\|A^k\|` 的实测几何收敛因子
    （应当趋于 :math:`\rho(A)`；非正规矩阵上它**看起来**很长时间都不收敛）。
    """
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    kmax = int(k_max)
    powers = matrix_powers(arr, kmax)
    norms = np.asarray([operator_2norm(powers[i], iters=iters, seed=seed) for i in range(kmax + 1)], dtype=float)
    from numcore.linalg.eigen import dense_eigenvalues

    lam = np.asarray(dense_eigenvalues(arr), dtype=complex)
    rho = float(np.max(np.abs(lam))) if lam.size else 0.0
    ks = np.arange(kmax + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = np.where(rho > 0.0, rho**ks, np.where(ks == 0, 1.0, 0.0))
    peak_index = int(np.argmax(norms))
    peak = float(norms[peak_index])
    growth = float(peak / pred[peak_index]) if pred[peak_index] > 0.0 else float("inf")
    # 尾部收敛因子：||A^k|| 的相邻比值（k 足够大以后）
    tail = norms[max(1, kmax // 2) :]
    pos = tail[tail > 0.0]
    factor = float(np.exp(np.mean(np.log(pos[1:] / pos[:-1])))) if pos.size >= 3 else float("nan")
    return TransientResult(
        norms=norms,
        spectral_prediction=pred,
        peak=peak,
        peak_index=peak_index,
        spectral_radius=rho,
        growth_ratio=growth,
        kreiss=peak,
        asymptotic_factor=factor,
        k_max=kmax,
        non_normal=bool(peak > 1.0 + 1e-12 and rho < 1.0),
    )


def kreiss_constant(
    A: Any,
    *,
    k_max: int = 60,
    iters: int = 80,
    seed: int = 5,
) -> float:
    r"""Kreiss 常数 :math:`K(A) = \sup_{k\ge 0}\|A^k\|_2`（本实现取 :math:`k\le k_{\max}` 的实测最大值）。

    有限矩阵的 :math:`\sup` 在有限个 :math:`k` 上取到（:math:`\|A^k\|` 最终按
    :math:`\rho(A)^k`（可能乘多项式因子）衰减，且前 :math:`n` 步已经决定了全部增长），
    所以 :math:`k_{\max}=60` 对本章的矩阵已经足够——但这是**经验**，不是定理，
    所以 :func:`transient_growth` 把 ``k_max`` 一并报出来。
    """
    return transient_growth(A, k_max=k_max, iters=iters, seed=seed).kreiss


def kreiss_from_pseudospectrum(epsilons: Any, radii: Any, *, shift: float = 1.0) -> float:
    r"""由伪谱半径实测 Kreiss 下界：:math:`\max_\varepsilon(\rho_\varepsilon-1)/\varepsilon`。

    这就是 Kreiss 矩阵定理里 :math:`K(A) \ge \sup_\varepsilon(\rho_\varepsilon(A)-1)/\varepsilon`
    的离散版本（:math:`\rho_\varepsilon > 1` 的 :math:`\varepsilon` 才参与）。
    它是**下界**：网格分辨率不够时 :math:`\rho_\varepsilon` 被低估，下界也随之偏低——
    断言里必须写清这一点，不许把下界当等式用。
    """
    e = np.asarray(epsilons, dtype=float).reshape(-1)
    r = np.asarray(radii, dtype=float).reshape(-1)
    if e.shape != r.shape:
        raise ValueError(f"epsilons 与 radii 形状不一致：{e.shape} vs {r.shape}")
    m = (e > 0.0) & (r > shift)
    if not np.any(m):
        return 0.0
    return float(np.max((r[m] - shift) / e[m]))


# --------------------------------------------------------------------------- #
# GMRES（手写：Arnoldi + Givens）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GMRESResult:
    r"""GMRES 的一次运行：残差历史 + 收敛诊断。

    ``residual`` 是相对残差 :math:`\|r_k\|_2/\|r_0\|_2`（长度 ``max_iter+1``，
    ``residual[0] = 1``）；``arnoldi_bound`` 是 Arnoldi 给出的**理论下降界**
    :math:`\min_{p\in P_k,p(0)=1}\|p(A)r_0\| \le \kappa(V)\min_{z\in\Lambda}\max|p(z)|`；
    ``plateau`` 是残差"卡住"的判定（连续 5 步下降不到 1%）。
    """

    residual: np.ndarray
    x: np.ndarray
    n_iter: int
    converged: bool
    restart: int
    breakdown: bool
    plateau: bool
    h_next: float
    krylov_dim: int


def gmres(
    A: Any,
    b: Any,
    *,
    max_iter: int | None = None,
    restart: int | None = None,
    tol: float = 1e-12,
) -> GMRESResult:
    r"""手写 GMRES（Arnoldi + Givens 最小二乘）：解 :math:`Ax=b`，记录相对残差历史。

    算法（Saad, *Iterative Methods for Sparse Linear Systems*, §6.5）：

    1. **Arnoldi**：构造标准正交基 :math:`V_k` 与上 Hessenberg :math:`\bar H_k`，
       :math:`AV_k = V_{k+1}\bar H_k`；
    2. **Givens**：把 :math:`\bar H_k` 逐步化成上三角 :math:`R`（同时把
       :math:`\|r_0\|e_1` 右端一起变换）——于是第 :math:`k` 步的残差可以**不真算** :math:`x_k` 就拿到：
       :math:`\|r_k\|_2 = |\gamma_{k+1}|`（最后一个变换后的分量）；
    3. **回代**：解 :math:`R_ky_k = \gamma_{1:k}`，:math:`x_k = V_ky_k`。

    与伪谱的联系（本章 ``gmres_pseudospec`` 一页要演的东西）：GMRES 的下降率由
    **特征值分布 + 特征向量矩阵的条件数**决定，

    .. math::
        \frac{\|r_k\|_2}{\|r_0\|_2} \;\le\; \kappa_2(V)\,\min_{p\in P_k,\,p(0)=1}\ \max_{\lambda\in\Lambda(A)}|p(\lambda)| ,

    所以"迹在谱上的多项式"看起来该把残差压下去，但 :math:`\kappa_2(V)` 一大（非正规）
    这个界就形同虚设——残差会在 1e-2 附近**卡住很多步**再掉下去（``plateau`` 字段）。

    实矩阵专用（Givens 旋转是实的）；复矩阵请先把方程拆成实部/虚部——本章的
    ``gmres_pseudospec`` 只用实矩阵。
    """
    arr = np.asarray(A, dtype=float)
    n = arr.shape[0]
    bv = np.asarray(b, dtype=float).reshape(-1)
    if bv.size != n:
        raise ValueError(f"b 的长度应为 {n}，得到 {bv.size}")
    max_iter = n if max_iter is None else int(max_iter)
    restart = max_iter if restart is None else int(restart)
    if max_iter < 1:
        raise ValueError(f"max_iter 至少为 1，得到 {max_iter}")
    beta = float(np.sqrt(np.sum(bv**2)))
    residual = np.zeros(max_iter + 1, dtype=float)
    residual[0] = 1.0
    if beta == 0.0:  # b = 0：解就是 0
        return GMRESResult(
            residual=residual, x=np.zeros(n), n_iter=0, converged=True,
            restart=restart, breakdown=False, plateau=False, h_next=0.0, krylov_dim=0,
        )
    V = np.zeros((n, max_iter + 1), dtype=float)
    H = np.zeros((max_iter + 1, max_iter), dtype=float)
    cs = np.zeros(max_iter, dtype=float)
    sn = np.zeros(max_iter, dtype=float)
    g = np.zeros(max_iter + 1, dtype=float)
    g[0] = beta
    V[:, 0] = bv / beta
    x = np.zeros(n, dtype=float)
    converged = False
    breakdown = False
    used = 0
    krylov = 0
    R = np.zeros((max_iter, max_iter), dtype=float)
    for k in range(max_iter):
        used = k + 1
        # ---- Arnoldi 一步（改进 Gram–Schmidt + 一次重正交化）----
        w = arr @ V[:, k]
        for _ in range(2):
            for i in range(k + 1):
                h = float(V[:, i] @ w)
                H[i, k] += h
                w = w - h * V[:, i]
        h_next = float(np.sqrt(np.sum(w**2)))
        H[k + 1, k] = h_next
        krylov = k + 1
        # ---- 对第 k 列做 Givens 消元（把 H[k+1,k] 干掉）----
        for i in range(k):
            h_i = cs[i] * H[i, k] + sn[i] * H[i + 1, k]
            h_ip1 = -sn[i] * H[i, k] + cs[i] * H[i + 1, k]
            H[i, k] = h_i
            H[i + 1, k] = h_ip1
        r = float(np.hypot(H[k, k], H[k + 1, k]))
        if r == 0.0:
            cs[k], sn[k] = 1.0, 0.0
        else:
            cs[k] = H[k, k] / r
            sn[k] = H[k + 1, k] / r
        H[k, k] = r
        H[k + 1, k] = 0.0
        g[k + 1] = -sn[k] * g[k]
        g[k] = cs[k] * g[k]
        R[: k + 1, k] = H[: k + 1, k]
        residual[k + 1] = abs(g[k + 1]) / beta
        if h_next <= 1e-14 * max(1.0, float(np.max(np.abs(H)))):
            breakdown = True  # 不变子空间：解已在其中，精确
            used = k + 1
            y = _back_substitute(R[: k + 1, : k + 1], g[: k + 1])
            x = V[:, : k + 1] @ y
            converged = True
            break
        V[:, k + 1] = w / h_next
        if residual[k + 1] <= tol:
            converged = True
            y = _back_substitute(R[: k + 1, : k + 1], g[: k + 1])
            x = V[:, : k + 1] @ y
            break
    plateau = _detect_plateau(residual[: used + 1])
    return GMRESResult(
        residual=residual,
        x=np.asarray(x),
        n_iter=used if converged else max_iter,
        converged=converged,
        restart=restart,
        breakdown=breakdown,
        plateau=plateau,
        h_next=float(H[min(used, max_iter), min(used - 1, max_iter - 1)]),
        krylov_dim=krylov,
    )


def _back_substitute(R: np.ndarray, g: np.ndarray) -> np.ndarray:
    r"""上三角回代 :math:`Ry=g`（GMRES 的最后一步）。"""
    m = R.shape[0]
    y = np.zeros(m, dtype=float)
    for i in range(m - 1, -1, -1):
        diag = R[i, i]
        if diag == 0.0:
            continue
        y[i] = (g[i] - R[i, i + 1 :] @ y[i + 1 :]) / diag
    return y


def _detect_plateau(residual: np.ndarray, *, window: int = 6, ratio: float = 0.99) -> bool:
    r"""残差是否"卡住"：连续 ``window`` 步的相对下降都小于 ``1-ratio``。"""
    r = np.asarray(residual, dtype=float)
    if r.size < window + 1:
        return False
    tail = r[-(window + 1) :]
    if tail[-1] <= 0.0:
        return False
    drops = tail[:-1] / np.maximum(tail[1:], 1e-300)
    return bool(np.all(drops <= 1.0 / ratio)) and bool(tail[-1] > 1e-14)


def gmres_arnoldi_residual_bound(A: Any, r0: Any, k: int, *, grid: int = 200, window: float = 2.0) -> float:
    r"""GMRES 的"多项式逼近 + 特征向量条件数"上界的一个**可计算代理**。

    真实上界 :math:`\kappa_2(V)\min_p\max_\lambda|p(\lambda)|` 里的 :math:`\kappa_2(V)`
    需要特征向量矩阵（本章不追求它的高精度），所以这里退一步给出**只依赖谱**的那一半：

    .. math::
        \min_{p\in P_k,p(0)=1}\ \max_{z\in\Omega}|p(z)|
        \quad\text{的实测值（在包含谱的窗口 }\Omega\text{ 上采样）},

    其中 :math:`p` 取"把 :math:`\Omega` 上的 Chebyshev 点当零点、再除以 :math:`p(0)` 归一化"
    的构造多项式——它不最优，但**是显式可算的**，足以说明"多项式视角给出的下降有多快"。
    返回值 :math:`\le 1`；非正规矩阵上实测残差会**远高于**它，这个差距就是 :math:`\kappa_2(V)`。
    """
    from numcore.linalg.eigen import dense_eigenvalues

    A0 = np.asarray(A, dtype=float)
    lam = np.asarray(dense_eigenvalues(A0), dtype=complex)
    c = complex(np.mean(lam)) if lam.size else 0.0
    rad = float(np.max(np.abs(lam - c))) if lam.size else 1.0
    rad = rad if rad > 0.0 else 1.0
    kk = max(1, int(k))
    # Chebyshev 节点（第一类零点）作为 p 的根
    j = np.arange(1, kk + 1, dtype=float)
    roots = c + rad * window * np.cos((2.0 * j - 1.0) * np.pi / (2.0 * kk))
    # 网格上的 max|p|
    xs = np.linspace(c.real - window * rad, c.real + window * rad, int(grid))
    ys = np.linspace(c.imag - window * rad, c.imag + window * rad, int(grid))
    XX, YY = np.meshgrid(xs, ys)
    Z = (XX + 1j * YY).ravel()
    P = np.ones(Z.size, dtype=complex)
    for r in roots:
        P = P * (Z - r)
    p0 = np.prod(-roots)  # p(0)（根的乘积，符号已含）
    if p0 == 0.0:
        return 1.0
    return float(np.max(np.abs(P / p0)))


# --------------------------------------------------------------------------- #
# 解析参照与演示矩阵族
# --------------------------------------------------------------------------- #
def jordan_boundary_radius(n: int, epsilon: float) -> float:
    r"""Jordan 块 :math:`J_n(0)` 的 :math:`\varepsilon`-伪谱半径**解析值** :math:`\varepsilon^{1/n}`。

    .. math::
        \sigma_{\min}(J_n(0)-zI) = |z|\ \text{（当 } |z| \text{ 足够小）}
        \quad\Longrightarrow\quad
        \rho_\varepsilon = \varepsilon^{1/n} .

    严格的结论（Trefethen & Embree, *Spectra and Pseudospectra*, §4）：对 :math:`|z|\le 1`，

    .. math::
        \sigma_{\min}(J_n(0)-zI) = |z|^n \quad(\text{更精确地，}\ \sigma_{\min} = |z|^n/\sqrt{1+|z|^2+\cdots})

    工程上够用的形式是 :math:`\sigma_{\min}\approx|z|^n`：:math:`n=16`、:math:`\varepsilon=10^{-16}`
    时 :math:`\rho_\varepsilon = 0.1`——**16 个特征值全在原点，伪谱却铺到半径 0.1**。
    """
    if n < 1:
        raise ValueError(f"n 必须为正，得到 {n}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon 必须为正，得到 {epsilon}")
    return float(epsilon ** (1.0 / n))


def grrcar_like_matrix(n: int, coupling: float = 1.0) -> np.ndarray:
    r"""非正规演示矩阵：:math:`A = \mathrm{tridiag}(0, 1, 0) + c\,e_1e_n^{\mathsf T}`。

    它是 :math:`n\times n` 的单 Jordan 块（当 :math:`c\neq 0` 时**循环**，
    特征值全为 0 的**单个** Jordan 块），当 :math:`c=0` 时退化为幂零移位矩阵。
    伪谱是半径 :math:`\varepsilon^{1/n}` 的圆盘（到 :math:`O(\varepsilon)` 的修正），
    是本章断言里**唯一有闭式**的非正规例子。
    """
    A = np.zeros((n, n), dtype=float)
    idx = np.arange(n - 1)
    A[idx, idx + 1] = 1.0
    if coupling != 0.0:
        A[n - 1, 0] = float(coupling)
    return A


#: 伪谱矩阵动物园：每个都对应一句"别相信谱"
EXAMPLE_PSEUDOSPECTRUM_MATRICES: dict[str, str] = {
    "normal": "对角正规矩阵：伪谱 = ε-圆盘的并，谱一丝不动（对照组）",
    "jordan": "Jordan 块 J_n(0)：伪谱半径 ε^(1/n)，特征值全在原点却铺满一片",
    "shifted_jordan": "J_n(1)：伪谱跨过单位圆 → ρ(A)=1 却有瞬态增长",
    "shear": "剪切矩阵 I+sN：全部特征值 = 1，‖A^k‖ 峰值却可以到 1e3 以上",
    "grcar": "Grcar 矩阵：谱很规矩，伪谱大得离谱（Trefethen 的经典反例）",
    "companion": "伴随矩阵：多项式根对扰动极其敏感（Wilkinson 多项式）",
    "rotation": "旋转块：复共轭特征值，幂法不收敛的活标本",
}


def example_pseudospectrum_matrix(name: str, n: int = 12, *, scale: float = 1.0, shear: float = 1.0) -> np.ndarray:
    """返回 :data:`EXAMPLE_PSEUDOSPECTRUM_MATRICES` 里的矩阵（确定性构造，可复现）。"""
    from numcore.linalg.eigen import companion_matrix, grcar_matrix, jordan_block, shear_matrix

    if name == "normal":
        idx = np.arange(n)
        return np.diag(scale * (idx - (n - 1) / 2.0) / max(1.0, (n - 1) / 2.0))
    if name == "jordan":
        return jordan_block(max(2, n), 0.0)
    if name == "shifted_jordan":
        return jordan_block(max(2, n), 1.0)
    if name == "shear":
        return shear_matrix(max(2, n), shear)
    if name == "grcar":
        return grcar_matrix(max(4, n))
    if name == "companion":
        # ∏_{k=1}^{n}(z−k)：n 个根 1..n，降幂系数 [1, ..., c_0]（长度 n+1）
        # → n×n 伴随矩阵。根分得很开，但伪谱照样把它们连成一片，
        # 这正是伴随矩阵"根对扰动敏感"的来源。
        deg = max(2, n)
        coeffs = np.array([1.0])
        for k in range(1, deg + 1):
            nxt = np.zeros(coeffs.size + 1, dtype=float)
            nxt[: coeffs.size] += coeffs
            nxt[1:] -= k * coeffs
            coeffs = nxt
        return companion_matrix(coeffs)
    if name == "rotation":
        th = 2.0 * np.pi / 5.0
        return scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    raise KeyError(f"未知示例矩阵 {name!r}（可用 {sorted(EXAMPLE_PSEUDOSPECTRUM_MATRICES)}）")
