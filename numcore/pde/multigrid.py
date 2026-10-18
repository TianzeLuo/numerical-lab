r"""4.5 章（多重网格与快速求解）的数值内核。

模型问题是**一维 Dirichlet Poisson 方程**

.. math::
    -u''(x)=f(x),\quad x\in(0,1),\qquad u(0)=u(1)=0,

在网格 :math:`x_i=ih`、:math:`h=1/(n+1)` 上离散为 :math:`A_hu_h=f_h`，其中

.. math::
    A_h=h^{-2}\operatorname{tridiag}(-1,2,-1)\in\mathbb R^{n\times n}

（与 1.5 章的 :func:`numcore.linalg.iterative.laplacian_1d` 同一个算子，只是乘上
:math:`h^{-2}`）。本章要回答的不是"怎么解这一个系统"，而是
**为什么多重网格的收敛因子与网格尺寸无关**、以及这句话在什么条件下会失效。

四条实现纪律，各自对应一个最容易骗人的地方：

1. **转移算子必须成对出现**。线性插值 :math:`P`（粗 :math:`\to` 细）与满权限制
   :math:`R`（细 :math:`\to` 粗）在本章的取向下满足 :math:`R=\tfrac12P^{\mathsf T}`
   （:func:`restriction_matrix` 的 ``factor=0.5``）；缩放常数与粗层算子是一个**配对**：
   取 :math:`R=\tfrac12P^{\mathsf T}` 配 :math:`A_H` 时 Galerkin 恒等式
   :math:`RA_hP=A_H` 精确成立（:func:`galerkin_defect` 实测），
   而把它换成 :math:`R=P^{\mathsf T}` 就是把粗层校正放大一倍——**会退化成不收敛**
   （见 :func:`scaling_probe`）。
2. **平滑是"逐频率"的性质**，不是"迭代收敛得快"。加权 Jacobi 的迭代矩阵在正弦基下是对角的，
   每个模态的放大率可以逐点实测（:func:`mode_factors_measured`）；
   低频的放大率是 :math:`1-O(h^2)`——**几乎不动**才是正常的
   （:func:`mode_factors_theory` 给出对照公式）。
3. **循环因子要一边跑一边量**。:func:`cycle_factor` 用幂迭代在**算子作用**上量
   :math:`\rho(E_{\rm cyc})`，:func:`factor_scan` 把它沿 :math:`h` 序列扫一遍并给出
   均值/标准差/漂移；代价用 :func:`mg_apply` 的计数器**实测**（不是估的）。
   把这两件事一起看，才谈得上"网格无关"。
4. **非线性必须走 FAS**。线性多重网格直接套到 :math:`-u''+u^3=f` 上会收敛到一个
   **不是解**的假解：它的线性残差可以很小而真残差停在 :math:`10^{-2}` 量级
   （:func:`measure_fas_factor` 的 ``nonlinear=False`` 分支就是这条反例的实测入口）。

本模块只依赖 :mod:`numpy` 原语、1.5 章的迭代法工具（解析谱与幂法）与
:mod:`numcore.linalg.lu`（非线性参考解用的牛顿法），不碰任何现成求解器（P2/P3）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from numcore.linalg.iterative import (
    gs_iteration_matrix,
    jacobi_iteration_matrix,
    laplacian_1d,
    mode_decay_factors,
    power_method,
    sine_eigenvectors_1d,
    sor_iteration_matrix,
)
from numcore.linalg.lu import lu_decompose, lu_solve
from numcore.utils import rng

__all__ = [
    # 网格与算子
    "Grid",
    "grid_sizes",
    "hierarchy",
    "poisson_1d",
    "restriction_matrix",
    "prolongation_matrix",
    "restrict",
    "prolong",
    "galerkin_defect",
    "coarse_scaling_ratio",
    "scaling_probe",
    "ScalingProbe",
    # 平滑子
    "jacobi_sweep",
    "gs_sweep",
    "smooth",
    "smoother_matrix",
    "mode_factors_theory",
    "mode_factors_measured",
    "smoothing_factor",
    # 循环
    "MGHierarchy",
    "TRACE_PHASES",
    "mg_hierarchy",
    "mg_apply",
    "two_grid_matrix",
    "coarse_correction_matrix",
    "window_factor",
    "two_grid_factor",
    "cycle_factor",
    "measure_cycle_factor",
    "CycleResult",
    "factor_scan",
    "FactorScan",
    "MGPrecond",
    # 非线性
    "nonlinear_operator",
    "nonlinear_residual",
    "nonlinear_smooth",
    "fas_cycle",
    "measure_fas_factor",
    "FasResult",
    "exact_sine_source",
    "discrete_newton",
    "NewtonResult",
]

#: 一维最优平滑权重：加权 Jacobi 的 :math:`\omega=2/3` 把高频放大率压到 :math:`1/3`。
OMEGA_SMOOTH_1D = 2.0 / 3.0

#: 循环内部默认的粗层扫描数（把最粗层"解到机器精度"，让循环成为一个确定的算子）。
COARSE_SWEEPS = 24

#: 标量三次方程 :math:`h^2u^3+2u=s` 的二分步数（52 步后区间小于 2^-52，已是机器精度）。
_CUBIC_BISECT = 52

_SMOOTHER_KINDS = ("jacobi", "gs", "sor")

#: :func:`mg_apply` 的 ``trace`` 事件相位编码（层级图按它分组）。
#: 每个事件是 ``(层号, 相位, 该层当前向量的 2-范数, 该层内点数)``。
TRACE_PHASES = {
    0: "enter",
    1: "pre_smooth",
    2: "restrict",
    3: "coarse",
    4: "correct",
    5: "post_smooth",
    6: "coarsest",
}


# --------------------------------------------------------------------------- #
# 1. 网格与算子
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid:
    """一层网格：层号（0 = 最细）、内点数 :math:`n`、步长 :math:`h` 与算子 :math:`A_h`。"""

    level: int
    n: int
    h: float
    A: np.ndarray


def poisson_1d(n: int) -> np.ndarray:
    r"""一维 Dirichlet Poisson 算子 :math:`A_h=h^{-2}\operatorname{tridiag}(-1,2,-1)`。

    :math:`h=1/(n+1)`，内点 :math:`x_i=ih`；与 1.5 章的 ``laplacian_1d`` 只差一个
    :math:`h^{-2}` 因子，乘上它是为了让**粗层算子就是粗网格上的同一个算子**
    （而不是差一个 :math:`1/4` 的缩放）。
    """
    m = int(n)
    if m < 1:
        raise ValueError(f"内点数至少为 1，得到 {n}")
    h = 1.0 / (m + 1)
    return laplacian_1d(m) / (h * h)


def grid_sizes(n_fine: int, levels: int) -> tuple[int, ...]:
    r"""嵌套网格序列的内点数：:math:`n_l=(n_0+1)/2^l-1`（第 0 层最细）。

    要求 :math:`n_0+1` 能被 :math:`2^{\text{levels}-1}` 整除、且最深一层至少 1 个内点，
    否则粗网格点不会落在细网格点上，"限制/延拓"就不是嵌套的（这一步会当场报错，
    而不是安静地插值出一个错的粗层问题）。
    """
    m = int(n_fine)
    lev = int(levels)
    if m < 1:
        raise ValueError(f"最细层内点数至少为 1，得到 {n_fine}")
    if lev < 1:
        raise ValueError(f"层数至少为 1，得到 {levels}")
    if (m + 1) % (2 ** (lev - 1)) != 0:
        raise ValueError(
            f"n+1 = {m + 1} 不能被 2^{lev - 1} = {2 ** (lev - 1)} 整除："
            "粗网格点落不到细网格点上（嵌套网格要求 n = 2^k - 1 这一类尺寸）"
        )
    coarsest = (m + 1) // (2 ** (lev - 1)) - 1
    if coarsest < 1:
        raise ValueError(f"{lev} 层太深：最粗层内点数会变成 {coarsest} < 1")
    return tuple((m + 1) // (2**l) - 1 for l in range(lev))


def hierarchy(n_fine: int, levels: int) -> tuple[Grid, ...]:
    """由细到粗的网格序列（含每层的算子 :math:`A_h`）。"""
    sizes = grid_sizes(n_fine, levels)
    return tuple(
        Grid(level=l, n=s, h=1.0 / (s + 1), A=poisson_1d(s)) for l, s in enumerate(sizes)
    )


# --------------------------------------------------------------------------- #
# 2. 限制与延拓
# --------------------------------------------------------------------------- #
def prolongation_matrix(n_coarse: int) -> np.ndarray:
    r"""线性插值算子 :math:`P`：形状 :math:`(2n_c+1)\times n_c`。

    第 :math:`j` 个粗内点落在细网格的**奇数**下标 :math:`2j+1` 上（:math:`H=2h`），
    偶数下标取两端粗点的平均（缺的那一端是边界上的 0）：

    .. math::
        (Pu)_{2j+1}=u_j,\qquad (Pu)_{2j}=\tfrac12(u_{j-1}+u_j),\qquad
        (Pu)_{2j+2}=\tfrac12(u_j+u_{j+1}),

    越界的 :math:`u_{-1}=u_{n_c}=0`。这样构造出来的 :math:`P` 与
    :func:`restriction_matrix` 的 ``factor=0.5`` 严格互为伴随的两倍，
    见 :func:`galerkin_defect`。
    """
    nc = int(n_coarse)
    if nc < 1:
        raise ValueError(f"粗层内点数至少为 1，得到 {n_coarse}")
    nf = 2 * nc + 1
    P = np.zeros((nf, nc), dtype=float)
    idx = np.arange(nc)
    P[2 * idx + 1, idx] = 1.0
    j = np.arange(nc + 1)
    P[2 * j, np.maximum(j - 1, 0)] = 0.5  # 偶数细点: 左邻居（j=0 时缺的那端是边界 0）
    inner = j < nc
    P[2 * j[inner], j[inner]] = 0.5  # 偶数细点: 右邻居（j=n_c 时缺的那端是边界 0）
    return P


def restriction_matrix(
    n_coarse: int, *, factor: float = 0.5, kind: str = "full_weighting"
) -> np.ndarray:
    r"""满权限制算子 :math:`R=\text{factor}\cdot P^{\mathsf T}`，形状 :math:`n_c\times(2n_c+1)`。

    ``factor=0.5`` 时 :math:`R` 就是权重 :math:`\tfrac14[1,2,1]` 的满权限制
    （:func:`restrict` 的 ``kind="full_weighting"``），并且是本章唯一"正确"的缩放：
    它让 :math:`RA_hP=A_H` 精确成立。``factor`` 是给反例用的旋钮
    （:func:`scaling_probe` 会把它拨到 0.25 / 1.0 / 2.0 看退化）。

    ``kind="injection"`` 时只取 :math:`r_{2j+1}`（<1> 在每个粗点上），
    于是 :math:`R\ne cP^{\mathsf T}`：两网格算子不再对称，因子会变差——
    这也是一个可以量的反例。
    """
    nc = int(n_coarse)
    if kind == "injection":
        nf = 2 * nc + 1
        R = np.zeros((nc, nf), dtype=float)
        R[np.arange(nc), 2 * np.arange(nc) + 1] = 1.0
        return R
    if kind != "full_weighting":
        raise ValueError(f"未知限制方式 {kind!r}（取 full_weighting / injection）")
    P = prolongation_matrix(nc)
    return float(factor) * P.T


def restrict(r_fine: Any, *, factor: float = 0.5, kind: str = "full_weighting") -> np.ndarray:
    r"""细层残差 :math:`\to` 粗层：满权限制（默认）或注入（``kind="injection"``）。

    满权限制 :math:`(Rr)_j=\tfrac14 r_{2j}+\tfrac12r_{2j+1}+\tfrac14r_{2j+2}`
    ——三个细点都在下标范围内（粗内点一定是细内点，两端各有邻居），
    所以**不需要**在这里补边界零。注入只取 :math:`r_{2j+1}`：
    它更便宜，但丢掉 :math:`\tfrac14(r_{2j}+r_{2j+2})` 那一项，
    于是 :math:`R\ne c P^{\mathsf T}`，两网格算子不再对称（1.5 章的非正规性那一套在这里重现）。
    """
    r = np.asarray(r_fine, dtype=float).reshape(-1)
    nf = r.size
    if (nf - 1) % 2 != 0:
        raise ValueError(f"细层长度必须为奇数（嵌套网格），得到 {nf}")
    nc = (nf - 1) // 2
    if kind == "injection":
        return r[1::2].copy()
    if kind != "full_weighting":
        raise ValueError(f"未知限制方式 {kind!r}（取 full_weighting / injection）")
    out = np.zeros(nc, dtype=float)
    idx = np.arange(nc)
    out = 0.25 * r[2 * idx] + 0.5 * r[2 * idx + 1] + 0.25 * r[2 * idx + 2]
    if factor != 1.0:
        out = float(factor) * out
    return out


def prolong(e_coarse: Any) -> np.ndarray:
    """粗层校正 :math:`\to` 细层：:math:`P e_c`（线性插值，边界自动取 0）。"""
    ec = np.asarray(e_coarse, dtype=float).reshape(-1)
    return prolongation_matrix(ec.size) @ ec


def galerkin_defect(n_fine: int, *, levels: int = 2, factor: float = 0.5) -> float:
    r"""实测 Galerkin 缺陷 :math:`\|RA_hP-A_H\|_\infty/\|A_H\|_\infty`。

    :math:`A_H` 取**粗网格上的重离散算子**（即 :func:`poisson_1d`）。
    一维线性插值 + 满权限制（``factor=0.5``）下它精确成立（缺陷 0，实测 ~1e-16 量级）；
    换成 ``factor=1.0``（丢掉 :math:`\tfrac12`）缺陷立刻跳到 1.0 ——
    这正是"缩放常数与粗层算子必须配对"的机器证据。
    配对关系的另一半由 :func:`coarse_scaling_ratio` 给出：:math:`P^{\mathsf T}A_hP=2A_H`。
    """
    grids = hierarchy(int(n_fine), int(levels))
    if len(grids) < 2:
        raise ValueError("Galerkin 恒等式至少需要 2 层")
    fine, coarse = grids[0], grids[1]
    R = restriction_matrix(coarse.n, factor=float(factor))
    P = prolongation_matrix(coarse.n)
    lhs = R @ (fine.A @ P)
    denom = float(np.max(np.abs(coarse.A)))
    return float(np.max(np.abs(lhs - coarse.A)) / denom) if denom > 0 else 0.0


def coarse_scaling_ratio(n_fine: int, *, levels: int = 2) -> float:
    r"""实测 :math:`\|P^{\mathsf T}A_hP\|_\infty/\|A_H\|_\infty`：一维里恒等于 2。

    它解释了为什么满权限制必须带 :math:`\tfrac12`：Galerkin 粗层算子
    :math:`RA_hP` 要等于重离散算子 :math:`A_H`，而 :math:`P^{\mathsf T}A_hP=2A_H`，
    所以 :math:`R=\tfrac12P^{\mathsf T}` 不是"习惯写法"，是被恒等式定死的常数。
    """
    grids = hierarchy(int(n_fine), int(levels))
    if len(grids) < 2:
        raise ValueError("至少需要 2 层")
    fine, coarse = grids[0], grids[1]
    P = prolongation_matrix(coarse.n)
    lhs = P.T @ (fine.A @ P)
    denom = float(np.max(np.abs(coarse.A)))
    return float(np.max(np.abs(lhs)) / denom) if denom > 0 else 0.0


@dataclass(frozen=True)
class ScalingProbe:
    """一批缩放常数下的两网格因子实测（反例证据，见 :func:`scaling_probe`）。"""

    factors: np.ndarray
    cycle_factors: np.ndarray
    coarse_scale: np.ndarray
    galerkin_defect: np.ndarray
    n_fine: int
    levels: int

    @property
    def worst(self) -> float:
        return float(np.max(self.cycle_factors))

    def diverging(self) -> np.ndarray:
        return self.cycle_factors > 1.0


def scaling_probe(
    n_fine: int = 63,
    *,
    levels: int = 4,
    factors: Any = (0.25, 0.5, 1.0, 2.0),
    kind: str = "jacobi",
    omega: float = OMEGA_SMOOTH_1D,
    nu1: int = 1,
    nu2: int = 1,
    coarse_sweeps: int = COARSE_SWEEPS,
    coarse_scale: float = 1.0,
    restrict_kind: str = "full_weighting",
    iters: int = 24,
    seed: int = 45045,
) -> ScalingProbe:
    r"""把限制算子的缩放常数 :math:`c` 拨到不同值，实测 V-cycle 因子。

    ``c=0.5`` 是本章的"正确"取值（:math:`R=\tfrac12P^{\mathsf T}` 配重离散粗层算子）。
    其它取值要么欠校正（:math:`c<0.5`，因子变大）、要么过校正（:math:`c>0.5`，
    粗层校正被放大 :math:`2c` 倍，因子可以越过 1，循环**发散**）。
    ``coarse_scale`` 用来看"配对"的另一半：把粗层算子整体乘一个常数，
    只有 :math:`c\cdot\text{coarse\_scale}=0.5` 这一条线上才是 Galerkin 配对。
    """
    fs = np.asarray(list(factors), dtype=float)
    out = np.zeros(fs.size, dtype=float)
    defects = np.zeros(fs.size, dtype=float)
    for i, c in enumerate(fs):
        mg = mg_hierarchy(
            int(n_fine),
            levels=int(levels),
            kind=kind,
            omega=float(omega),
            nu1=int(nu1),
            nu2=int(nu2),
            coarse_sweeps=int(coarse_sweeps),
            cycle="v",
            restrict_factor=float(c),
            restrict_kind=str(restrict_kind),
            coarse_scale=float(coarse_scale),
        )
        out[i] = cycle_factor(mg, iters=int(iters), seed=int(seed)).factor
        defects[i] = galerkin_defect(
            int(n_fine), levels=max(2, int(levels)), factor=float(c) * float(coarse_scale)
        )
    return ScalingProbe(
        factors=fs,
        cycle_factors=out,
        coarse_scale=np.full(fs.size, float(coarse_scale)),
        galerkin_defect=defects,
        n_fine=int(n_fine),
        levels=int(levels),
    )


# --------------------------------------------------------------------------- #
# 3. 平滑子
# --------------------------------------------------------------------------- #
def jacobi_sweep(A: Any, b: Any, x: Any, *, omega: float = 1.0) -> np.ndarray:
    r"""一次加权 Jacobi 扫描：:math:`x\leftarrow x+\omega D^{-1}(b-Ax)`。

    :math:`\omega=1` 是普通 Jacobi；一维 Poisson 的最优平滑权重是 :math:`\omega=2/3`
    （高频放大率 :math:`\max(1-\omega,|1-2\omega|)=1/3`）。
    """
    M = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float).reshape(-1)
    y = np.asarray(x, dtype=float).reshape(-1).copy()
    d = np.diag(M)
    if np.any(d == 0.0):
        raise ValueError("Jacobi 需要对角元全非零")
    return y + float(omega) * (rhs - M @ y) / d


def gs_sweep(A: Any, b: Any, x: Any, *, omega: float = 1.0, sweeps: int = 1) -> np.ndarray:
    r"""前向 Gauss--Seidel / SOR 扫描（就地更新，与 1.5 章 ``_sweep`` 同一算式）。

    :math:`\omega=1` 时逐位等于 Gauss--Seidel（``(1-1)*x + 1*new`` 在浮点下精确等于 ``new``），
    这一点由 ``tests/4.5`` 与 1.5 章的 ``stationary_solve`` 对拍。
    """
    M = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float).reshape(-1)
    y = np.asarray(x, dtype=float).reshape(-1).copy()
    n = y.size
    for _ in range(max(1, int(sweeps))):
        for i in range(n):
            row = M[i]
            s = float(row @ y) - float(row[i]) * float(y[i])
            new = (rhs[i] - s) / row[i]
            y[i] = new if omega == 1.0 else (1.0 - float(omega)) * y[i] + float(omega) * new
    return y


def smooth(
    A: Any,
    b: Any,
    x: Any,
    *,
    kind: str = "jacobi",
    omega: float = OMEGA_SMOOTH_1D,
    sweeps: int = 1,
) -> np.ndarray:
    """统一平滑入口：``jacobi``（用 ``omega``）/ ``sor``（用 ``omega``，1 即 GS）/ ``gs``。"""
    k = str(kind)
    if k not in _SMOOTHER_KINDS:
        raise ValueError(f"未知平滑子 {kind!r}（取 {'/'.join(_SMOOTHER_KINDS)}）")
    y = np.asarray(x, dtype=float).reshape(-1).copy()
    for _ in range(max(1, int(sweeps))):
        if k == "jacobi":
            y = jacobi_sweep(A, b, y, omega=float(omega))
        elif k == "gs":
            y = gs_sweep(A, b, y, omega=1.0)
        else:
            y = gs_sweep(A, b, y, omega=float(omega))
    return y


def smoother_matrix(A: Any, *, kind: str = "jacobi", omega: float = OMEGA_SMOOTH_1D) -> np.ndarray:
    r"""平滑迭代矩阵 :math:`T`：:math:`e_{k+1}=Te_k`。

    * 加权 Jacobi：:math:`T=I-\omega D^{-1}A`；
    * GS / SOR：:math:`T=(D-\omega L)^{-1}\bigl((1-\omega)D+\omega U\bigr)`，
      :math:`A=D-L-U`（``L``、``U`` 取正号），与 1.5 章的
      :func:`~numcore.linalg.iterative.sor_iteration_matrix` 是同一个对象（测试里对拍）。
    """
    M = np.asarray(A, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {M.shape}")
    k = str(kind)
    if k == "jacobi":
        if float(omega) == 1.0:
            return jacobi_iteration_matrix(M)
        d = np.diag(M)
        if np.any(d == 0.0):
            raise ValueError("Jacobi 需要对角元全非零")
        return np.eye(M.shape[0]) - float(omega) * (M / d[:, None])
    if k == "gs":
        return gs_iteration_matrix(M)
    if k == "sor":
        return sor_iteration_matrix(M, float(omega))
    raise ValueError(f"未知平滑子 {kind!r}")


def mode_factors_theory(n: int, *, kind: str = "jacobi", omega: float = OMEGA_SMOOTH_1D) -> np.ndarray:
    r"""模型问题上**逐模态**的理论放大率 :math:`|\lambda_k(T)|`，:math:`k=1\ldots n`。

    一维 Poisson 上 :math:`D^{-1}A` 与 :math:`A` 同特征向量，特征值
    :math:`1-\cos\theta_k`（:math:`\theta_k=k\pi/(n+1)`），于是

    .. math::
        \text{jacobi}:\ |1-\omega(1-\cos\theta_k)|
        =\Bigl|1-\tfrac43\sin^2\tfrac{\theta_k}{2}\Bigr|\quad(\omega=\tfrac23),

    :math:`\omega=2/3` 时它就是讲义里那条 :math:`|1-\tfrac43\sin^2(\pi k/N)|`。
    GS 用 Young 关系 :math:`\lambda(T_{\rm GS})=\mu_k^2=\cos^2\theta_k`
    （**渐近**值：:math:`T_{\rm GS}` 与 :math:`A` 不同特征向量，初期步会混模态，
    见 :func:`mode_factors_measured` 的第二组返回值）。
    """
    m = int(n)
    if m < 1:
        raise ValueError(f"内点数至少为 1，得到 {n}")
    theta = np.arange(1, m + 1, dtype=float) * np.pi / (m + 1)
    k = str(kind)
    if k == "jacobi":
        return np.abs(1.0 - float(omega) * (1.0 - np.cos(theta)))
    if k == "gs":
        return np.cos(theta) ** 2
    if k == "sor":
        return np.asarray(mode_decay_factors(m, method="sor", omega=float(omega)), dtype=float)
    raise ValueError(f"未知平滑子 {kind!r}")


def mode_factors_measured(
    A: Any,
    *,
    kind: str = "jacobi",
    omega: float = OMEGA_SMOOTH_1D,
) -> tuple[np.ndarray, np.ndarray]:
    r"""**逐点实测**每个正弦模态被一次平滑放大的倍数，返回 ``(范数比, 瑞利商比)``。

    * 范数比 :math:`\|Tv_k\|_2/\|v_k\|_2`：这个模态真的被缩了多少（"高频消失得多快"）；
    * 瑞利商比 :math:`|v_k^{\mathsf T}Tv_k|/(v_k^{\mathsf T}v_k)`：模态能量的一步放大率。

    加权 Jacobi 的 :math:`T` 与 :math:`A` 同特征向量，两者**逐位相同**且等于
    :func:`mode_factors_theory`；GS/SOR 的 :math:`T` 不是正规矩阵、也不与 :math:`A`
    同特征向量，两者会分开——分开多少就是"低频不动"这句话能信到什么程度的实测边界。
    """
    M = np.asarray(A, dtype=float)
    n = M.shape[0]
    T = smoother_matrix(M, kind=kind, omega=omega)
    V = sine_eigenvectors_1d(n)
    num = np.zeros(n, dtype=float)
    ray = np.zeros(n, dtype=float)
    for j in range(n):
        v = V[:, j]
        Tv = T @ v
        num[j] = float(np.sqrt(float(Tv @ Tv)))
        ray[j] = abs(float(v @ Tv))
    return num, ray


def smoothing_factor(
    n: int,
    *,
    kind: str = "jacobi",
    omega: float = OMEGA_SMOOTH_1D,
    measured: bool = True,
) -> float:
    r"""平滑因子 :math:`\mu_{\rm smooth}=\max_{\theta_k\ge\pi/2}|\lambda_k(T)|`。

    ``measured=True``（默认）用 :func:`mode_factors_measured` 的范数比在**高频半带**上取最大，
    ``measured=False`` 用 :func:`mode_factors_theory`。一维 :math:`\omega=2/3` 的加权 Jacobi
    理论值是 :math:`1/3`（在 :math:`\theta=\pi/2` 与 :math:`\theta=\pi` 两处同时取到）；
    GS 的**渐近**值要小得多，但初期步的实测范数比会告诉你另一回事（见 ``tests/4.5``）。
    """
    m = int(n)
    theta = np.arange(1, m + 1, dtype=float) * np.pi / (m + 1)
    high = theta >= np.pi / 2.0
    if not np.any(high):
        high = np.ones(m, dtype=bool)
    if measured:
        num, _ = mode_factors_measured(poisson_1d(m), kind=kind, omega=omega)
        return float(np.max(num[high]))
    return float(np.max(mode_factors_theory(m, kind=kind, omega=omega)[high]))


# --------------------------------------------------------------------------- #
# 4. 层级、两网格与 V / W 循环
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MGHierarchy:
    """一套多重网格的静态数据：网格、转移算子、平滑参数与循环类型。"""

    grids: tuple[Grid, ...]
    restrictions: tuple[np.ndarray, ...]
    prolongations: tuple[np.ndarray, ...]
    kind: str
    omega: float
    nu1: int
    nu2: int
    coarse_sweeps: int
    cycle: str
    restrict_factor: float
    restrict_kind: str
    coarse_scale: float

    @property
    def n_fine(self) -> int:
        return self.grids[0].n

    @property
    def levels(self) -> int:
        return len(self.grids)

    @property
    def h(self) -> float:
        return self.grids[0].h

    @property
    def gamma(self) -> int:
        return 2 if self.cycle == "w" else 1


def mg_hierarchy(
    n_fine: int,
    *,
    levels: int = 4,
    kind: str = "jacobi",
    omega: float = OMEGA_SMOOTH_1D,
    nu1: int = 1,
    nu2: int = 1,
    coarse_sweeps: int = COARSE_SWEEPS,
    cycle: str = "v",
    restrict_factor: float = 0.5,
    restrict_kind: str = "full_weighting",
    coarse_scale: float = 1.0,
) -> MGHierarchy:
    """构造一套多重网格（V 或 W 循环，前/后各 ``nu1``/``nu2`` 次平滑）。"""
    grids = hierarchy(int(n_fine), int(levels))
    # 转移算子按**粗层**尺寸构造：第 l 项把第 l+1 层送到第 l 层
    Rs = tuple(
        restriction_matrix(g.n, factor=float(restrict_factor), kind=str(restrict_kind))
        for g in grids[1:]
    )
    Ps = tuple(prolongation_matrix(g.n) for g in grids[1:])
    if float(coarse_scale) != 1.0:
        grids = (grids[0],) + tuple(
            Grid(level=g.level, n=g.n, h=g.h, A=float(coarse_scale) * g.A) for g in grids[1:]
        )
    cyc = str(cycle).lower()
    if cyc not in ("v", "w"):
        raise ValueError(f"循环类型取 v / w，得到 {cycle!r}")
    if str(kind) not in _SMOOTHER_KINDS:
        raise ValueError(f"未知平滑子 {kind!r}")
    return MGHierarchy(
        grids=grids,
        restrictions=Rs,
        prolongations=Ps,
        kind=str(kind),
        omega=float(omega),
        nu1=max(0, int(nu1)),
        nu2=max(0, int(nu2)),
        coarse_sweeps=max(1, int(coarse_sweeps)),
        cycle=cyc,
        restrict_factor=float(restrict_factor),
        restrict_kind=str(restrict_kind),
        coarse_scale=float(coarse_scale),
    )


def mg_apply(
    mg: MGHierarchy,
    x: Any,
    b: Any = None,
    *,
    counter: list[float] | None = None,
    trace: list[tuple[int, int, float, int]] | None = None,
) -> np.ndarray:
    r"""对 :math:`A_hx=b` 做一次 V/W 循环（``b=None`` 即误差方程 :math:`A_he=0`）。

    ``counter`` 给出一份**实测代价账**：``counter[l]`` 累加第 :math:`l` 层上做过的
    点扫描次数（按 :math:`n_l/n_0` 折算成"最细层等价点数"）。
    代价是"数出来的"，不是估出来的——W-cycle 与 V-cycle 的每步代价比就靠它。

    ``trace`` 非空时按时间顺序追加 ``(层号, 相位, 该层范数, 该层内点数)``，
    相位编码见 :data:`TRACE_PHASES`——层级图那一页要的就是"误差在什么时候
    被搬到哪一层、又搬回来"的真实顺序，而不是模型自己编的顺序。
    """
    y = np.asarray(x, dtype=float).reshape(-1).copy()
    rhs = np.zeros_like(y) if b is None else np.asarray(b, dtype=float).reshape(-1).copy()
    if rhs.size != y.size:
        raise ValueError(f"右端长度 {rhs.size} 与初值长度 {y.size} 不一致")
    return _cycle(mg, 0, y, rhs, counter, trace)


def _note(
    trace: list[tuple[int, int, float, int]] | None,
    level: int,
    phase: int,
    vec: np.ndarray,
) -> None:
    if trace is not None:
        trace.append((int(level), int(phase), float(np.sqrt(float(vec @ vec))), int(vec.size)))


def _cycle(
    mg: MGHierarchy,
    level: int,
    x: np.ndarray,
    b: np.ndarray,
    counter: list[float] | None,
    trace: list[tuple[int, int, float, int]] | None = None,
) -> np.ndarray:
    g = mg.grids[level]
    _note(trace, level, 0, x)
    if level == mg.levels - 1:
        _count(counter, level, mg.coarse_sweeps, g.n, mg.n_fine)
        x = smooth(g.A, b, x, kind=mg.kind, omega=mg.omega, sweeps=mg.coarse_sweeps)
        _note(trace, level, 6, x)
        return x
    if mg.nu1:
        _count(counter, level, mg.nu1, g.n, mg.n_fine)
        x = smooth(g.A, b, x, kind=mg.kind, omega=mg.omega, sweeps=mg.nu1)
    _note(trace, level, 1, x)
    r = b - g.A @ x
    rc = mg.restrictions[level] @ r
    _note(trace, level + 1, 2, rc)
    ec = np.zeros_like(rc)
    for _ in range(mg.gamma):
        ec = _cycle(mg, level + 1, ec, rc, counter, trace)
    _note(trace, level + 1, 3, ec)
    x = x + mg.prolongations[level] @ ec
    _note(trace, level, 4, x)
    if mg.nu2:
        _count(counter, level, mg.nu2, g.n, mg.n_fine)
        x = smooth(g.A, b, x, kind=mg.kind, omega=mg.omega, sweeps=mg.nu2)
    _note(trace, level, 5, x)
    return x


def _count(counter: list[float] | None, level: int, sweeps: int, n: int, n_fine: int) -> None:
    if counter is not None:
        counter[level] += float(sweeps) * float(n) / float(n_fine)


def two_grid_matrix(mg: MGHierarchy, level: int = 0) -> np.ndarray:
    r"""两网格误差传播算子的**矩阵**：

    .. math::
        E_{\rm TG}=S^{\nu_2}\bigl(I-PA_H^{-1}RA_h\bigr)S^{\nu_1},

    粗层用**精确解**（:math:`A_H^{-1}` 由追赶法逐列算出，粗层算子三对角）。
    它是 :func:`mg_apply` 在"粗层无穷次扫描"极限下的样子，用来分开看
    "转移算子对不对"与"粗层解得够不够准"这两件事。
    """
    g = mg.grids[level]
    gc = mg.grids[level + 1]
    R = mg.restrictions[level]
    P = mg.prolongations[level]
    inv = _inverse_tridiag(gc.A)
    E = np.eye(g.n) - P @ (inv @ (R @ g.A))
    S1 = _smoother_power(g.A, mg.kind, mg.omega, mg.nu1)
    S2 = _smoother_power(g.A, mg.kind, mg.omega, mg.nu2)
    return S2 @ (E @ S1)


def _smoother_power(A: np.ndarray, kind: str, omega: float, sweeps: int) -> np.ndarray:
    T = smoother_matrix(A, kind=kind, omega=omega)
    out = np.eye(A.shape[0])
    for _ in range(max(0, int(sweeps))):
        out = T @ out
    return out


def _inverse_tridiag(A: np.ndarray) -> np.ndarray:
    """三对角矩阵的逆（逐列追赶法；只用于两网格算子的矩阵形式）。"""
    n = A.shape[0]
    lower = np.concatenate(([0.0], np.diag(A, -1)))
    diag = np.diag(A).copy()
    upper = np.concatenate((np.diag(A, 1), [0.0]))
    out = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        out[:, j] = _thomas(lower, diag, upper, e)
    return out


def _thomas(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """追赶法（与 2.2 章 ``tridiagonal_apply`` 同一算式，这里只解三对角粗层系统）。"""
    n = diag.size
    cp = np.zeros(n, dtype=float)
    fp = np.zeros(n, dtype=float)
    piv = float(diag[0])
    if piv == 0.0:
        raise ZeroDivisionError("追赶法遇到零主元（第 0 行）")
    cp[0] = float(upper[0]) / piv
    fp[0] = float(rhs[0]) / piv
    for i in range(1, n):
        piv = float(diag[i]) - float(lower[i]) * cp[i - 1]
        if piv == 0.0:
            raise ZeroDivisionError(f"追赶法遇到零主元（第 {i} 行）")
        cp[i] = float(upper[i]) / piv
        fp[i] = (float(rhs[i]) - float(lower[i]) * fp[i - 1]) / piv
    out = np.empty(n, dtype=float)
    out[-1] = fp[-1]
    for i in range(n - 2, -1, -1):
        out[i] = fp[i] - cp[i] * out[i + 1]
    return out


def coarse_correction_matrix(mg: MGHierarchy, level: int = 0) -> np.ndarray:
    r"""粗网格校正算子 :math:`I-PA_H^{-1}RA_h`（**不含**平滑）。

    与 :func:`two_grid_matrix` 的区别就是没有前/后平滑：它的逐模态范数在高频端
    可以大于 1（高频本来就该由平滑子负责），所以单看它不收敛很正常。
    「粗网格管低频、平滑子管高频」这句话的定量版本就是这条曲线与
    :func:`mode_factors_measured` 的互补形状。
    """
    if mg.levels < 2:
        raise ValueError("粗网格校正至少需要两层")
    fine = mg.grids[level]
    coarse = mg.grids[level + 1]
    inv = _inverse_tridiag(coarse.A)
    eye = np.eye(fine.n, dtype=float)
    return eye - mg.prolongations[level] @ (inv @ (mg.restrictions[level] @ fine.A))


def two_grid_factor(mg: MGHierarchy, *, maxiter: int = 2000, tol: float = 1e-12) -> float:
    """两网格因子的实测（对 :func:`two_grid_matrix` 跑幂法，与 1.5 章同一把尺子）。"""
    E = two_grid_matrix(mg)
    return float(power_method(E, maxiter=int(maxiter), tol=float(tol)).value)


@dataclass(frozen=True)
class CycleResult:
    r"""一次循环因子实测的全套账：因子、逐周期放大率、**数出来的**代价。

    ``ratios[k]`` 是第 :math:`k+1` 个周期作用在**单位**误差向量上的一步放大率
    :math:`\|E_{\\rm cyc}y_k\|_2`；``norms`` 是它的累计乘积（相对初始的下降）。
    这两个别混：前者的尾部几何平均才是 :math:`\rho(E_{\\rm cyc})`，
    拿后者相邻项相除只会得到 ≈1 的假因子（本项目真踩过，见 ``tests/4.5`` 的注释）。
    """

    factor: float
    ratios: np.ndarray
    norms: np.ndarray
    work: float
    work_by_level: np.ndarray
    iters_to_tol: int
    tol: float
    n_fine: int
    levels: int
    cycle: str
    kind: str
    omega: float

    @property
    def work_per_cycle(self) -> float:
        """一个周期的实测代价（最细层等价点扫描数）。"""
        return self.work

    def reduction_per_work(self) -> float:
        """每个"最细层等价点扫描"换来的范数下降倍数（因子越小、代价越大越吃亏）。"""
        w = max(self.work, 1e-300)
        return float(self.factor ** (1.0 / w)) if self.factor > 0 else 0.0


def cycle_factor(
    mg: MGHierarchy,
    *,
    iters: int = 32,
    tol: float = 1e-8,
    seed: int = 45045,
    tail: int = 8,
    x0: Any = None,
) -> CycleResult:
    r"""幂迭代实测**循环因子** :math:`\rho(E_{\rm cyc})` 与达到 ``tol`` 所需周期数。

    起点是 :func:`numcore.utils.rng` 的确定性随机向量（P4），每个周期做一次
    :func:`mg_apply`（误差方程，右端 0）后立刻归一化——于是每次量到的范数
    **就是**那个周期的一步放大率，尾部若干项的几何平均即 :math:`\rho(E_{\rm cyc})`。
    ``iters_to_tol`` 是累计下降真的越过 ``tol`` 的那一步（不是纸上除对数）；
    代价是计数器实测的**每周期**最细层等价点扫描数。
    """
    m = int(mg.n_fine)
    x = rng(int(seed)).standard_normal(m) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
    if x.size != m:
        raise ValueError(f"初值长度应为 {m}，得到 {x.size}")
    x = x / float(np.sqrt(float(x @ x)))
    counter = [0.0] * mg.levels
    steps: list[float] = []
    cum = 1.0
    iters_to_tol = -1
    n_iter = 0
    for k in range(int(iters)):
        x = mg_apply(mg, x, None, counter=counter)
        nx = float(np.sqrt(float(x @ x)))
        steps.append(nx)
        cum *= nx
        n_iter = k + 1
        if iters_to_tol < 0 and cum <= float(tol):
            iters_to_tol = k + 1
        if nx == 0.0:
            break
        x = x / nx
    arr = np.asarray(steps, dtype=float)
    use = arr[-int(tail) :] if tail and arr.size else arr
    pos = use[use > 0.0]
    factor = float(np.exp(np.mean(np.log(pos)))) if pos.size else 0.0
    if iters_to_tol < 0:
        iters_to_tol = int(n_iter)
    work = float(np.sum(counter)) / max(1, int(n_iter))
    return CycleResult(
        factor=factor,
        ratios=arr,
        norms=np.cumprod(arr) if arr.size else arr,
        work=work,
        work_by_level=np.asarray(counter, dtype=float) / max(1, int(n_iter)),
        iters_to_tol=int(iters_to_tol),
        tol=float(tol),
        n_fine=m,
        levels=mg.levels,
        cycle=mg.cycle,
        kind=mg.kind,
        omega=mg.omega,
    )


def measure_cycle_factor(
    n_fine: int,
    *,
    levels: int = 4,
    cycle: str = "v",
    kind: str = "jacobi",
    omega: float = OMEGA_SMOOTH_1D,
    nu1: int = 1,
    nu2: int = 1,
    coarse_sweeps: int = COARSE_SWEEPS,
    restrict_factor: float = 0.5,
    restrict_kind: str = "full_weighting",
    iters: int = 32,
    tol: float = 1e-8,
    seed: int = 45045,
    tail: int = 8,
) -> CycleResult:
    """:func:`cycle_factor` 的便捷包装（按参数直接建层级）。"""
    mg = mg_hierarchy(
        int(n_fine),
        levels=int(levels),
        kind=kind,
        omega=float(omega),
        nu1=int(nu1),
        nu2=int(nu2),
        coarse_sweeps=int(coarse_sweeps),
        cycle=cycle,
        restrict_factor=float(restrict_factor),
        restrict_kind=str(restrict_kind),
    )
    return cycle_factor(mg, iters=int(iters), tol=float(tol), seed=int(seed), tail=int(tail))


@dataclass(frozen=True)
class FactorScan:
    """沿 :math:`h` 序列的因子扫描：网格无关性的全部实测账。"""

    levels: np.ndarray
    n_fine: np.ndarray
    h: np.ndarray
    factor_mg: np.ndarray
    factor_two_grid: np.ndarray
    factor_smoother: np.ndarray
    smoother_mode: np.ndarray
    work_mg: np.ndarray
    iters_mg: np.ndarray
    iters_smoother: np.ndarray
    kind: str
    omega: float
    cycle: str
    nu1: int
    nu2: int

    @property
    def mg_factor_mean(self) -> float:
        return float(np.mean(self.factor_mg))

    @property
    def mg_factor_std(self) -> float:
        """跨 :math:`h` 的标准差——"网格无关"这句话的分母。"""
        return float(np.std(self.factor_mg))

    @property
    def mg_factor_drift(self) -> float:
        """最细与最粗网格上的因子之差（漂移），比标准差更能暴露单调恶化。"""
        return float(self.factor_mg[-1] - self.factor_mg[0])

    @property
    def smoother_trend(self) -> float:
        r"""单层平滑子因子到 :math:`1` 的距离按 :math:`h^2` 收缩的实测斜率。

        取 :math:`-\log(1-\rho)` 对 :math:`\log h` 的斜率：理论上是 2（因为
        :math:`1-\cos\theta_1\approx\pi^2h^2/2`）——"单层迭代与网格有关"的定量说法。
        """
        y = -np.log(np.maximum(1.0 - self.factor_smoother, 1e-300))
        x = np.log(self.h)
        xm, ym = float(np.mean(x)), float(np.mean(y))
        den = float(np.sum((x - xm) ** 2))
        return float(np.sum((x - xm) * (y - ym)) / den) if den > 0 else float("nan")

    def iterations_for(self, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
        """由实测因子换算"降到 ``tol`` 需要多少次"（平滑子是换算值，循环是实测值）。"""
        need = float(np.log(1.0 / float(tol)))
        mg = np.array([np.log(need) / np.log(1.0 / f) if 0 < f < 1 else np.inf for f in self.factor_mg])
        sm = np.array(
            [np.log(need) / np.log(1.0 / f) if 0 < f < 1 else np.inf for f in self.factor_smoother]
        )
        return np.ceil(mg), np.ceil(sm)


def factor_scan(
    levels_seq: Any = (3, 4, 5, 6),
    *,
    cycle: str = "v",
    kind: str = "jacobi",
    omega: float = OMEGA_SMOOTH_1D,
    nu1: int = 1,
    nu2: int = 1,
    coarse_sweeps: int = COARSE_SWEEPS,
    iters: int = 24,
    seed: int = 45045,
    smoother_maxiter: int = 1500,
    two_grid_max_levels: int = 6,
) -> FactorScan:
    r"""沿 :math:`n_l=2^l-1` 的网格序列实测四件事，作为"网格无关"的证据链。

    1. ``factor_mg``：V/W 循环因子（:func:`cycle_factor`，作用式幂迭代）；
    2. ``factor_two_grid``：两网格因子（粗层精确解，:func:`two_grid_factor`）；
    3. ``factor_smoother``：**单层**平滑子作为求解器的收敛因子 :math:`\rho(T)`（1.5 章的幂法）；
    4. ``work_mg`` / ``iters_mg``：一次循环的实测代价与真的数出来的周期数。

    ``two_grid_max_levels`` 之外不再算两网格矩阵（最细层超过 ~127 时逐列追赶法的账不划算，
    而且那时的结论已经由 ``factor_mg`` 给出）。
    """
    lv = np.asarray([int(x) for x in levels_seq], dtype=int)
    if lv.size == 0:
        raise ValueError("至少给一个层数")
    ns = 2**lv - 1
    f_mg = np.zeros(lv.size, dtype=float)
    f_tg = np.full(lv.size, np.nan, dtype=float)
    f_sm = np.zeros(lv.size, dtype=float)
    modes: list[str] = []
    work = np.zeros(lv.size, dtype=float)
    it_mg = np.zeros(lv.size, dtype=int)
    for i, (L, n) in enumerate(zip(lv, ns)):
        res = measure_cycle_factor(
            int(n),
            levels=int(L),
            cycle=cycle,
            kind=kind,
            omega=omega,
            nu1=nu1,
            nu2=nu2,
            coarse_sweeps=coarse_sweeps,
            iters=int(iters),
            seed=int(seed),
        )
        f_mg[i] = res.factor
        work[i] = res.work
        it_mg[i] = res.iters_to_tol
        if L <= int(two_grid_max_levels):
            mg = mg_hierarchy(
                int(n),
                levels=int(L),
                kind=kind,
                omega=omega,
                nu1=nu1,
                nu2=nu2,
                coarse_sweeps=coarse_sweeps,
                cycle="v",
            )
            f_tg[i] = two_grid_factor(mg)
        pw = power_method(
            smoother_matrix(poisson_1d(int(n)), kind=kind, omega=omega),
            maxiter=int(smoother_maxiter),
        )
        f_sm[i] = float(pw.value)
        modes.append(str(pw.mode))
    need = float(np.log(1.0 / 1e-8))
    it_sm = np.array(
        [np.ceil(need / np.log(1.0 / f)) if 0 < f < 1 else np.inf for f in f_sm], dtype=float
    )
    return FactorScan(
        levels=lv,
        n_fine=ns,
        h=1.0 / (ns + 1),
        factor_mg=f_mg,
        factor_two_grid=f_tg,
        factor_smoother=f_sm,
        smoother_mode=np.asarray(modes, dtype=object),
        work_mg=work,
        iters_mg=it_mg.astype(float),
        iters_smoother=it_sm,
        kind=str(kind),
        omega=float(omega),
        cycle=str(cycle),
        nu1=int(nu1),
        nu2=int(nu2),
    )


class MGPrecond:
    r"""把一次多重网格循环当成预条件子 :math:`M^{-1}\approx A^{-1}`（供 1.5 章的 PCG 调用）。

    只提供 ``apply(r)``（PCG 需要的接口）：``apply(r)`` = 以 :math:`r` 为右端、零初值做一次
    V-cycle。它**不是**精确逆，也不提供 ``matvec_M``——所以 :math:`M` 本身没有定义，
    :func:`numcore.linalg.precond.precond_spectrum` 那类需要 :math:`M` 的诊断在这里会明确报错，
    而不是拿一个假的 :math:`M` 去算内积。
    """

    __slots__ = ("_mg",)

    def __init__(
        self,
        n_fine: int,
        *,
        levels: int = 4,
        cycle: str = "v",
        kind: str = "jacobi",
        omega: float = OMEGA_SMOOTH_1D,
        nu1: int = 1,
        nu2: int = 1,
        coarse_sweeps: int = COARSE_SWEEPS,
    ) -> None:
        self._mg = mg_hierarchy(
            int(n_fine),
            levels=int(levels),
            kind=kind,
            omega=float(omega),
            nu1=int(nu1),
            nu2=int(nu2),
            coarse_sweeps=int(coarse_sweeps),
            cycle=cycle,
        )

    @property
    def hierarchy(self) -> MGHierarchy:
        return self._mg

    def apply(self, r: Any) -> np.ndarray:
        """一次 V-cycle，零初值：:math:`z\approx A^{-1}r`。"""
        rr = np.asarray(r, dtype=float).reshape(-1)
        if rr.size != self._mg.n_fine:
            raise ValueError(f"右端长度应为 {self._mg.n_fine}，得到 {rr.size}")
        return mg_apply(self._mg, np.zeros_like(rr), rr)


# --------------------------------------------------------------------------- #
# 5. 非线性：FAS
# --------------------------------------------------------------------------- #
def nonlinear_operator(u: Any, h: float) -> np.ndarray:
    r"""非线性算子 :math:`A_h(u)=h^{-2}\operatorname{tridiag}(-1,2,-1)u+u^3`（齐次 Dirichlet）。"""
    v = np.asarray(u, dtype=float).reshape(-1)
    if v.size < 1:
        raise ValueError("u 不能为空")
    T = laplacian_1d(v.size)
    return (T @ v) / (float(h) * float(h)) + v**3


def nonlinear_residual(u: Any, h: float, f: Any) -> np.ndarray:
    r"""真（非线性）残差 :math:`A_h(u)-f`：FAS 与"假装线性"两种做法的分水岭。"""
    return nonlinear_operator(u, h) - np.asarray(f, dtype=float).reshape(-1)


def _cubic_root_vec(h2: float, s: np.ndarray) -> np.ndarray:
    r"""逐点解 :math:`h^2u^3+2u=s`（向量化二分，52 步到机器精度）。

    函数 :math:`g(u)=h^2u^3+2u` 严格单调（:math:`g'=3h^2u^2+2>0`），
    且 :math:`u` 必在 :math:`0` 与 :math:`s` 之间（:math:`u>0\Rightarrow 2u<s`，
    :math:`u<0\Rightarrow 2u>s`），所以 :math:`[\min(0,s),\max(0,s)]` 一定是包围区间——
    不需要牛顿法的收敛性假设，也不会有"迭代没收敛但没人发现"的缝。
    """
    lo = np.minimum(0.0, s)
    hi = np.maximum(0.0, s)
    for _ in range(_CUBIC_BISECT):
        mid = 0.5 * (lo + hi)
        val = h2 * mid**3 + 2.0 * mid - s
        neg = val < 0.0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    return 0.5 * (lo + hi)


def _cubic_root_scalar(h2: float, s: float) -> float:
    lo = 0.0 if s > 0.0 else float(s)
    hi = float(s) if s > 0.0 else 0.0
    for _ in range(_CUBIC_BISECT):
        mid = 0.5 * (lo + hi)
        val = h2 * mid**3 + 2.0 * mid - s
        if val < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def nonlinear_smooth(
    u: Any,
    h: float,
    f: Any,
    *,
    kind: str = "jacobi",
    omega: float = OMEGA_SMOOTH_1D,
    sweeps: int = 1,
    nonlinear: bool = True,
) -> np.ndarray:
    r"""非线性平滑子：逐点解 :math:`h^2u_i^3+2u_i=u_{i-1}+u_{i+1}+h^2f_i`。

    ``kind="jacobi"`` 用上一轮的全部邻居（完全向量化，帧里默认走这条）；
    ``kind="gs"`` 用已经更新的左邻居（前向扫描，Python 循环，只在测试的小网格上用）。
    ``nonlinear=False`` 把三次项从**平滑子**里去掉——这是反例：它会收敛到线性问题的解。
    """
    y = np.asarray(u, dtype=float).reshape(-1).copy()
    rhs = np.asarray(f, dtype=float).reshape(-1)
    if rhs.size != y.size:
        raise ValueError(f"f 的长度应为 {y.size}，得到 {rhs.size}")
    n = y.size
    h2 = float(h) * float(h)
    if str(kind) not in _SMOOTHER_KINDS:
        raise ValueError(f"未知平滑子 {kind!r}")
    for _ in range(max(1, int(sweeps))):
        if kind == "jacobi":
            left = np.concatenate(([0.0], y[:-1]))
            right = np.concatenate((y[1:], [0.0]))
            s = left + right + h2 * rhs
            new = _cubic_root_vec(h2, s) if nonlinear else 0.5 * s
            y = y + float(omega) * (new - y)
        else:
            w = 1.0 if kind == "gs" else float(omega)
            for i in range(n):
                left = y[i - 1] if i > 0 else 0.0
                right = y[i + 1] if i < n - 1 else 0.0
                s = left + right + h2 * rhs[i]
                new = _cubic_root_scalar(h2, s) if nonlinear else 0.5 * s
                y[i] = new if w == 1.0 else (1.0 - w) * y[i] + w * new
    return y


def fas_cycle(
    mg: MGHierarchy,
    u: Any,
    f: Any,
    *,
    nonlinear: bool = True,
    counter: list[float] | None = None,
) -> np.ndarray:
    r"""一次 FAS（Full Approximation Scheme）V/W 循环：:math:`-u''+u^3=f`。

    * 细层：先平滑 :math:`\nu_1` 次，算真残差 :math:`r_h=f_h-A_h(u_h)`，限制到粗层；
    * 粗层右端 :math:`f_H=Rr_h+A_H(Ru_h)`（FAS 的招牌：**粗层解的是解本身，不是误差**），
      初值取 :math:`Ru_h`；
    * 回到细层：:math:`u_h\leftarrow u_h+P\bigl(u_H-Ru_h\bigr)`，再平滑 :math:`\nu_2` 次。

    ``nonlinear=False`` 时残差与粗层算子都退化成线性的（把 :math:`u^3` 丢掉）——
    这时循环会**收敛到线性问题的解**，而真残差停在 :math:`10^{-2}` 量级，
    见 :func:`measure_fas_factor` 的对照组。
    """
    y = np.asarray(u, dtype=float).reshape(-1).copy()
    rhs = np.asarray(f, dtype=float).reshape(-1).copy()
    if rhs.size != y.size:
        raise ValueError(f"f 的长度应为 {y.size}，得到 {rhs.size}")
    return _fas(mg, 0, y, rhs, bool(nonlinear), counter)


def _nl_op(u: np.ndarray, h: float, nonlinear: bool) -> np.ndarray:
    if nonlinear:
        return nonlinear_operator(u, h)
    T = laplacian_1d(u.size)
    return (T @ u) / (h * h)


def _fas(
    mg: MGHierarchy,
    level: int,
    u: np.ndarray,
    f: np.ndarray,
    nonlinear: bool,
    counter: list[float] | None,
) -> np.ndarray:
    g = mg.grids[level]
    if level == mg.levels - 1:
        _count(counter, level, mg.coarse_sweeps, g.n, mg.n_fine)
        return nonlinear_smooth(
            u, g.h, f, kind=mg.kind, omega=mg.omega, sweeps=mg.coarse_sweeps, nonlinear=nonlinear
        )
    if mg.nu1:
        _count(counter, level, mg.nu1, g.n, mg.n_fine)
        u = nonlinear_smooth(u, g.h, f, kind=mg.kind, omega=mg.omega, sweeps=mg.nu1, nonlinear=nonlinear)
    r = f - _nl_op(u, g.h, nonlinear)
    R = mg.restrictions[level]
    P = mg.prolongations[level]
    gc = mg.grids[level + 1]
    rc = R @ r
    uc = R @ u
    fc = rc + _nl_op(uc, gc.h, nonlinear)
    for _ in range(mg.gamma):
        uc = _fas(mg, level + 1, uc, fc, nonlinear, counter)
    u = u + P @ (uc - (R @ u))
    if mg.nu2:
        _count(counter, level, mg.nu2, g.n, mg.n_fine)
        u = nonlinear_smooth(u, g.h, f, kind=mg.kind, omega=mg.omega, sweeps=mg.nu2, nonlinear=nonlinear)
    return u


def window_factor(
    arr: Any,
    *,
    floor_ratio: float = 1e-8,
    skip: int = 1,
    tail: int = 4,
) -> tuple[float, tuple[int, int]]:
    r"""在**触底之前**的渐近段取比值几何平均，返回 ``(因子, 窗口)``。

    残差序列的尾部会撞上舍入地板（例如 FAS 把 :math:`\|F\|` 压到 :math:`10^{-11}` 以后，
    相邻比值就变成 1 附近的噪声）。直接拿"最后几个比值"求几何平均会得到一个
    **假的**大因子（本模块实测过 0.86 这种数），所以这里先按 ``floor_ratio * arr[0]``
    把触底段切掉，再跳过 ``skip`` 个初始瞬态，只留最后 ``tail`` 个可用比值。
    返回的窗口 ``(k0, k1)`` 是真正用到的循环区间，供帧与断言如实标注。
    """
    a = np.asarray(arr, dtype=float).ravel()
    if a.size < 2:
        return float("nan"), (0, 0)
    floor = float(floor_ratio) * float(a[0])
    usable = [
        k
        for k in range(a.size - 1)
        if a[k] > floor and a[k + 1] > floor and a[k] > 0.0 and a[k + 1] > 0.0
    ]
    usable = usable[int(skip) :]
    if not usable:
        usable = list(range(max(0, a.size - 1 - int(tail)), a.size - 1))
    window = usable[-int(tail) :]
    ratios = a[[k + 1 for k in window]] / a[window]
    pos = ratios[ratios > 0.0]
    if pos.size == 0:
        return float("nan"), (int(window[0]), int(window[-1]) + 1)
    return float(np.exp(np.mean(np.log(pos)))), (int(window[0]), int(window[-1]) + 1)


@dataclass(frozen=True)
class FasResult:
    """FAS 实测账：真残差序列、渐近因子、代价，以及（给了参考解时的）误差。"""

    residuals: np.ndarray
    ratios: np.ndarray
    factor: float
    cycles: int
    work: float
    n_fine: int
    levels: int
    u: np.ndarray
    nonlinear: bool
    exact_error: float
    linear_residual: float
    window: tuple[int, int]
    cycles_to_tol: int

    @property
    def false_solution_gap(self) -> float:
        """真残差与线性残差之比：线性残差小得多的那个假解有多假。"""
        if self.linear_residual <= 0.0:
            return float("inf")
        return float(self.residuals[-1] / self.linear_residual)


def measure_fas_factor(
    n_fine: int = 63,
    *,
    levels: int = 4,
    nu1: int = 1,
    nu2: int = 1,
    cycles: int = 12,
    coarse_sweeps: int = COARSE_SWEEPS,
    kind: str = "jacobi",
    omega: float = OMEGA_SMOOTH_1D,
    nonlinear: bool = True,
    rhs_scale: float = 1.0,
    tail: int = 4,
    check: bool = True,
) -> FasResult:
    r"""实测 :math:`-u''+u^3=f` 上 FAS 的渐近收敛因子（真残差的尾部几何平均）。

    精确解取 :math:`u^*(x)=\sin(\pi x)`，源项 :math:`f=\pi^2\sin(\pi x)+\sin^3(\pi x)`
    （:func:`exact_sine_source`）；``check=True`` 时用自研牛顿法（:func:`discrete_newton`）
    解出**离散**参考解，报告 :math:`\|u-u_h^*\|_\infty`。
    ``nonlinear=False`` 是不做 FAS 的反例：残差按线性算子算，收敛到线性解。
    """
    mg = mg_hierarchy(
        int(n_fine),
        levels=int(levels),
        kind=kind,
        omega=float(omega),
        nu1=int(nu1),
        nu2=int(nu2),
        coarse_sweeps=int(coarse_sweeps),
        cycle="v",
    )
    g = mg.grids[0]
    x = np.arange(1, g.n + 1, dtype=float) * g.h
    u_exact, f = exact_sine_source(x, scale=float(rhs_scale))
    u = np.zeros(g.n, dtype=float)
    counter = [0.0] * mg.levels
    res = [float(np.sqrt(float(nonlinear_residual(u, g.h, f) @ nonlinear_residual(u, g.h, f))))]
    lin = [float(np.sqrt(float((f - _nl_op(u, g.h, False)) @ (f - _nl_op(u, g.h, False)))))]
    for _ in range(int(cycles)):
        u = fas_cycle(mg, u, f, nonlinear=nonlinear, counter=counter)
        rr = nonlinear_residual(u, g.h, f)
        ll = f - _nl_op(u, g.h, False)
        res.append(float(np.sqrt(float(rr @ rr))))
        lin.append(float(np.sqrt(float(ll @ ll))))
    arr = np.asarray(res, dtype=float)
    ratios = arr[1:] / np.maximum(arr[:-1], 1e-300)
    factor, window = window_factor(arr, skip=1, tail=int(tail))
    cum = arr / max(arr[0], 1e-300)
    below = np.nonzero(cum <= 1e-8)[0]
    cycles_to_tol = int(below[0]) if below.size else int(cycles)
    err = float("nan")
    if check:
        ref = discrete_newton(g.h, f)
        err = float(np.max(np.abs(u - ref.u))) if u.size else 0.0
    return FasResult(
        residuals=arr,
        ratios=ratios,
        factor=factor,
        cycles=int(cycles),
        work=float(np.sum(counter)) / max(1, int(cycles)),
        n_fine=g.n,
        levels=mg.levels,
        u=u,
        nonlinear=bool(nonlinear),
        exact_error=err,
        linear_residual=float(lin[-1]),
        window=window,
        cycles_to_tol=cycles_to_tol,
    )


def exact_sine_source(x: Any, *, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""精确解 :math:`u^*(x)=\sin(\pi x)` 与对应源项 :math:`f=\pi^2\sin(\pi x)+\sin^3(\pi x)`。

    这里 :math:`-u^{*\prime\prime}=\pi^2\sin(\pi x)`，所以 :math:`f` 让
    :math:`-u''+u^3=f` 的**连续**解正好是 :math:`u^*`（离散解与它差 :math:`O(h^2)`）。
    """
    xx = np.asarray(x, dtype=float).reshape(-1)
    s = np.sin(np.pi * xx)
    return float(scale) * s, float(scale) * (np.pi**2 * s + s**3)


@dataclass(frozen=True)
class NewtonResult:
    """自研牛顿法解离散非线性系统的结果（FAS 的独立参考解）。"""

    u: np.ndarray
    n_iter: int
    residual: float
    converged: bool


def discrete_newton(
    h: float,
    f: Any,
    *,
    u0: Any = None,
    tol: float = 1e-13,
    maxiter: int = 60,
) -> NewtonResult:
    r"""牛顿法解离散系统 :math:`A_h(u)=f`：:math:`J(u)\delta=-F(u)`，

    :math:`J=h^{-2}\operatorname{tridiag}(-1,2,-1)+3\operatorname{diag}(u^2)`（三对角、对角占优），
    每个牛顿步用自研 LU（P2：不碰 ``numpy.linalg``）。停机用**相对**残差
    :math:`\|F(u)\|_2\le\text{tol}\cdot\max(\|f\|_2,1)`。
    """
    rhs = np.asarray(f, dtype=float).reshape(-1)
    n = rhs.size
    u = np.zeros(n, dtype=float) if u0 is None else np.asarray(u0, dtype=float).reshape(-1).copy()
    T = laplacian_1d(n)
    hh = float(h) * float(h)
    scale = max(float(np.sqrt(float(rhs @ rhs))), 1.0)
    res = float("inf")
    conv = False
    k = 0
    for k in range(1, int(maxiter) + 1):
        F = (T @ u) / hh + u**3 - rhs
        res = float(np.sqrt(float(F @ F)))
        if res <= float(tol) * scale:
            conv = True
            break
        J = T / hh + np.diag(3.0 * u * u)
        dec = lu_decompose(J, pivot="partial")
        if dec.singular:
            raise ValueError("牛顿法的雅可比奇异：初值可能落在 u 的对称坏点上")
        u = u - lu_solve(dec, F)
    return NewtonResult(u=u, n_iter=int(k), residual=float(res), converged=bool(conv))
