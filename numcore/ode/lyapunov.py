r"""3.5 章（下半）：**Lyapunov 指数**、初值敏感性与孪生轨迹分离的实测口径。

数学对象
--------

对光滑流 :math:`\dot y=f(y)`，沿轨道 :math:`y(t)` 的**变分方程**是

.. math::
   \dot M=Df(y(t))\,M,\qquad M(0)=I,

:math:`M(t)` 把初始切向量线性地搬到 :math:`t` 时刻。Lyapunov 指数是增长率
:math:`\lambda_i=\lim_{t\to\infty}\frac1t\log\sigma_i(M(t))`（:math:`\sigma_i` 是奇异值）。
直接算 :math:`\sigma_i` 会在混沌系统上指数溢出，所以数值上走 **Benettin 的 QR 重正交化**：
每走 :math:`k` 步把 :math:`M` 做一次 QR 分解，累加 :math:`\log|R_{ii}|`，再令 :math:`M\leftarrow Q`：

.. math::
   \lambda_i\approx\frac{1}{N_k\,k\,h}\sum_{j=1}^{N_k}\log|R^{(j)}_{ii}| .

本模块的 QR 是**自写的修正 Gram--Schmidt**（P2/P3：不碰任何现成分解），并且
:func:`gram_schmidt` 返回正交性缺陷 :math:`\max|Q^{\mathsf T}Q-I|`——
"重正交化到底有没有把基拉正交"是实测的，不是假设的。

对映射 :math:`x_{n+1}=g(x_n)`，切映射按链式法则连乘 :math:`M_{n+1}=Dg(x_n)M_n`，
同一套 QR 累加直接适用；离散系统的"散度"是 :math:`\log|\det Dg|`，
于是 Hénon 映射给出一条**精确**的恒等式 :math:`\lambda_1+\lambda_2=\log|b|`
（:math:`\det Dg=-b` 处处常数）——它是本章最干净的绝对标尺。

口径（写进帧与断言）
--------------------

1. **暂态丢弃**：轨道要先落到吸引子上（本模块统一用 ``transient`` 个时间单位/步数）；
2. **步长**：定步长显式 RK，``h`` 必须做**减半实测**（:func:`step_size_stability`）；
3. **重正交化间隔**：``qr_interval`` 只影响浮点误差与代价，不影响数学；本模块给出
   两个间隔下的差值作为证据；
4. **有限时间估计有涨落**：单条轨道的 :math:`\lambda_1(T)` 是一个**分布**，不是常数。
   本章一律同时给出"点估计 + 跨初值的标准差"，并把公认值 0.9056 放在旁边对照。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from numcore.ode import dynamics as dyn
from numcore.ode import rk as rkmod

__all__ = [
    "BENETTIN_REFERENCE_LORENZ",
    "BENETTIN_REFERENCE_HENON",
    "BENETTIN_REFERENCE_LOGISTIC4",
    "GramSchmidt",
    "BenettinResult",
    "MapSpectrum",
    "SeparationResult",
    "gram_schmidt",
    "orthonormality_defect",
    "benettin_flow",
    "benettin_map",
    "logistic_lyapunov",
    "lyapunov_curve",
    "logistic_analytic_lyapunov",
    "twin_separation",
    "fit_growth_rate",
    "fit_window",
    "saturation_distance",
    "resolvable_time",
    "step_size_stability",
    "kaplan_yorke_dimension",
    "divergence_expectation",
]

#: 公认值（文献）：Lorenz σ=10、ρ=28、β=8/3 的最大 Lyapunov 指数。
#: 0.9056 出自 Sprott《Chaos and Time-Series Analysis》(2003) 的表 5.1；
#: 高精度复算（Viswanath 1998 的周期轨道法）给 0.905632…。本章所有对拍都用 0.9056。
BENETTIN_REFERENCE_LORENZ = 0.9056

#: Hénon 映射 a=1.4、b=0.3 的指数（文献 0.41922 / −1.62319，两者之和 = log|b|）。
BENETTIN_REFERENCE_HENON = (0.41922, -1.62319)

#: logistic 映射 r=4 的 Lyapunov 指数**解析值** ln2（共轭到 Bernoulli 移位）。
BENETTIN_REFERENCE_LOGISTIC4 = math.log(2.0)


# --------------------------------------------------------------------------- #
# 1. 自写 QR（修正 Gram--Schmidt + 二次重正交化）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GramSchmidt:
    """一次 Gram--Schmidt 的结果：正交基 ``Q``、上三角 ``R`` 与实测的正交性缺陷。"""

    Q: np.ndarray
    R: np.ndarray
    defect: float


def orthonormality_defect(Q: Any) -> float:
    r""":math:`\max_{ij}|(Q^{\mathsf T}Q-I)_{ij}|`（列正交性的直接读数，不引任何范数函数）。"""
    A = np.asarray(Q, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"需要二维数组，得到 {A.shape}")
    G = A.T @ A
    eye = np.eye(A.shape[1])
    return float(np.max(np.abs(G - eye)))


def gram_schmidt(A: Any, *, reortho: int = 2) -> GramSchmidt:
    r"""列向量的修正 Gram--Schmidt（MGS）正交化，可重复一轮（"twice is enough"）。

    MGS 与经典 GS 的差别在于内积是**逐个**从当前向量里减掉的——经典 GS 在
    :math:`\kappa` 大的基上会丢掉正交性（1.3 章的老朋友），MGS 的误差是
    :math:`O(\kappa u)`，再补一轮重正交化后基本贴住 :math:`u`。
    这里返回的 ``defect`` 就是这件事的实测值（Benettin 累加的是 :math:`\log|R_{ii}|`，
    正交性坏了等于把指数算错）。
    """
    M = np.asarray(A, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"需要二维矩阵，得到 {M.shape}")
    n, m = M.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))
    for j in range(m):
        v = M[:, j].copy()
        for _ in range(max(1, int(reortho))):
            for i in range(j):
                r = float(Q[:, i] @ v)
                R[i, j] += r
                v = v - r * Q[:, i]
        nrm = float(np.sqrt(v @ v))
        R[j, j] = nrm
        if nrm == 0.0:
            Q[:, j] = 0.0
        else:
            Q[:, j] = v / nrm
    return GramSchmidt(Q=Q, R=R, defect=orthonormality_defect(Q) if m > 1 else 0.0)


# --------------------------------------------------------------------------- #
# 2. 流的 Lyapunov 谱（Benettin QR）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BenettinResult:
    """一次 Benettin QR 的全部可观测物。"""

    exponents: np.ndarray
    t: np.ndarray
    running: np.ndarray
    defects: np.ndarray
    n_qr: int
    qr_interval: int
    h: float
    t_end: float
    n_f_evals: int
    y_final: np.ndarray
    blowup: bool

    def sum_exponents(self) -> float:
        return float(np.sum(self.exponents))

    def defect_max(self) -> float:
        return float(np.max(self.defects)) if self.defects.size else 0.0


def benettin_flow(
    system: dyn.FlowSystem,
    y0: Any,
    *,
    h: float = 0.01,
    t_end: float = 100.0,
    qr_interval: int = 10,
    transient: float = 0.0,
    method: str = "rk4",
    reortho: int = 2,
    max_steps: int = 400000,
) -> BenettinResult:
    r"""流的 Lyapunov 谱：变分方程 + 每 ``qr_interval`` 步一次 QR 重正交化。

    返回的 ``running`` 是**累计估计曲线** :math:`\lambda_i(t)`（形状 ``(n_qr, dim)``），
    它的用途不是好看，而是让"这个数收敛了吗"变成一个可以看的问题：混沌系统上
    有限时间估计的涨落常常比我们要分辨的差别还大。

    ``transient`` 是**先单独丢掉**的时间（不参与累加）：从任意初值出发的轨道要先落到
    吸引子上，把过渡段算进指数会得到一个既不是吸引子也不是初值的数。
    """
    dim = system.dim
    if not h > 0.0:
        raise ValueError("步长必须为正")
    if qr_interval < 1:
        raise ValueError("qr_interval 至少为 1")
    y = np.asarray(y0, dtype=float).reshape(-1)
    n_f = 0
    t = 0.0
    n_tr = int(round(float(transient) / h))
    for _ in range(n_tr):
        step = rkmod.rk_step(method, system.rhs, t, y, h)
        y = step.y
        n_f += step.n_f_evals
        t += h

    def aug(tt: float, Y: Any) -> np.ndarray:
        flat = np.asarray(Y, dtype=float).reshape(1, dim + dim * dim)
        yy = flat[:, :dim]
        M = flat[:, dim:].reshape(1, dim, dim)
        dy = np.asarray(system.rhs(tt, yy), dtype=float).reshape(1, dim)
        J = np.asarray(system.jac(tt, yy), dtype=float).reshape(1, dim, dim)
        dM = np.einsum("nij,njk->nik", J, M)
        return np.concatenate([dy, dM.reshape(1, dim * dim)], axis=1).reshape(-1)

    state = np.concatenate([y, np.eye(dim).reshape(-1)])
    sums = np.zeros(dim)
    defects: list[float] = []
    times: list[float] = []
    runs: list[np.ndarray] = []
    n_qr = 0
    n_steps = 0
    blowup = False
    while t < float(transient) + float(t_end) - 1e-15 and n_steps <= int(max_steps):
        step = rkmod.rk_step(method, aug, t, state, h)
        state = step.y
        n_f += step.n_f_evals
        t += h
        n_steps += 1
        if not np.all(np.isfinite(state)) or float(np.max(np.abs(state))) > dyn.BLOWUP:
            blowup = True
            break
        if n_steps % int(qr_interval) == 0:
            M = state[dim:].reshape(dim, dim)
            gs = gram_schmidt(M, reortho=reortho)
            diag = np.abs(np.diag(gs.R))
            diag = np.where(diag > 0.0, diag, np.finfo(float).tiny)
            sums = sums + np.log(diag)
            state = np.concatenate([state[:dim], gs.Q.reshape(-1)])
            n_qr += 1
            defects.append(gs.defect)
            times.append(t)
            runs.append(sums / (n_qr * qr_interval * h))
    span = n_qr * qr_interval * h
    lam = sums / span if span > 0.0 else np.full(dim, np.nan)
    return BenettinResult(
        exponents=np.asarray(lam, dtype=float),
        t=np.asarray(times, dtype=float),
        running=np.asarray(runs, dtype=float).reshape(-1, dim),
        defects=np.asarray(defects, dtype=float),
        n_qr=int(n_qr),
        qr_interval=int(qr_interval),
        h=float(h),
        t_end=float(t_end),
        n_f_evals=int(n_f),
        y_final=np.asarray(state[:dim], dtype=float),
        blowup=bool(blowup),
    )


# --------------------------------------------------------------------------- #
# 3. 映射的 Lyapunov 谱
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MapSpectrum:
    """映射的 Lyapunov 谱（同一套 QR 累加，作用在切映射的连乘上）。"""

    exponents: np.ndarray
    running: np.ndarray
    defects: np.ndarray
    n_iter: int
    n_qr: int
    qr_interval: int
    x_final: np.ndarray
    x_samples: np.ndarray


def benettin_map(
    g: dyn.MapRHS,
    dg: dyn.MapRHS,
    x0: Any,
    *,
    n_iter: int = 200000,
    transient: int = 20000,
    qr_interval: int = 1,
    reortho: int = 2,
    keep_samples: int = 0,
) -> MapSpectrum:
    r"""映射的 Lyapunov 谱：:math:`M\leftarrow Dg(x_n)M`，每 ``qr_interval`` 步 QR 一次。

    一维映射（logistic）退化成一个数：:math:`\lambda=\lim\frac1N\sum\log|g'(x_n)|`——
    此时 QR 没有意义（"正交化"一个标量就是取绝对值），但接口保持一致，帧那一侧不用分叉。
    """
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    for _ in range(int(transient)):
        x = np.asarray(g(x), dtype=float).reshape(-1)
    m = int(x.size)
    M = np.eye(m)
    sums = np.zeros(m)
    defects: list[float] = []
    runs: list[np.ndarray] = []
    n_qr = 0
    samples: list[np.ndarray] = []
    every = max(1, int(n_iter) // max(1, int(keep_samples))) if keep_samples else 0
    for k in range(int(n_iter)):
        J = np.asarray(dg(x), dtype=float).reshape(m, m)
        M = J @ M
        x = np.asarray(g(x), dtype=float).reshape(-1)
        if keep_samples and every and k % every == 0:
            samples.append(x.copy())
        if (k + 1) % int(qr_interval) == 0:
            if m == 1:
                diag = np.array([abs(M[0, 0])])
                Q = np.array([[1.0 if M[0, 0] >= 0.0 else -1.0]])
                defect = 0.0
            else:
                gs = gram_schmidt(M, reortho=reortho)
                diag = np.abs(np.diag(gs.R))
                Q = gs.Q
                defect = gs.defect
            diag = np.where(diag > 0.0, diag, np.finfo(float).tiny)
            sums = sums + np.log(diag)
            M = Q
            n_qr += 1
            defects.append(defect)
            runs.append(sums / (n_qr * qr_interval))
    span = n_qr * qr_interval
    lam = sums / span if span else np.full(m, np.nan)
    order = np.argsort(-np.asarray(lam, dtype=float))
    return MapSpectrum(
        exponents=np.asarray(lam, dtype=float)[order],
        running=np.asarray(runs, dtype=float).reshape(-1, m)[:, order],
        defects=np.asarray(defects, dtype=float),
        n_iter=int(n_iter),
        n_qr=int(n_qr),
        qr_interval=int(qr_interval),
        x_final=x,
        x_samples=np.asarray(samples, dtype=float).reshape(-1, m),
    )


# --------------------------------------------------------------------------- #
# 4. logistic 映射的指数（逐元素版 + 解析值）
# --------------------------------------------------------------------------- #
def logistic_lyapunov(
    r: float,
    *,
    n_iter: int = 200000,
    transient: int = 20000,
    x0: float = 0.4,
    n_block: int = 1,
) -> tuple[float, float]:
    r"""logistic 映射的 :math:`\lambda(r)=\frac1N\sum\log|r(1-2x_n)|`。

    返回 ``(点估计, 分块标准误)``：把 ``n_iter`` 分成 ``n_block`` 段分别求均值，
    用段间标准差除以 :math:`\sqrt{n_{block}}` 作为标准误——**有限时间涨落**必须
    和点估计一起给，否则 :math:`r=4` 上"和 ln2 差 1e-5"这种话没有意义。
    """
    x = float(x0)
    for _ in range(int(transient)):
        x = r * x * (1.0 - x)
    n_block = max(1, int(n_block))
    per = max(1, int(n_iter) // n_block)
    means = np.zeros(n_block)
    for b in range(n_block):
        acc = 0.0
        for _ in range(per):
            x = r * x * (1.0 - x)
            d = abs(r * (1.0 - 2.0 * x))
            acc += math.log(d) if d > 0.0 else -math.inf
        means[b] = acc / per
    est = float(np.mean(means))
    sem = float(np.std(means, ddof=1) / math.sqrt(n_block)) if n_block > 1 else 0.0
    return est, sem


def logistic_analytic_lyapunov(r: float) -> float:
    r"""三类**解析** Lyapunov 指数，用来给数值估计定标（周期窗口内是精确值）。

    * :math:`1<r<3`：不动点 :math:`x_*=1-1/r`，:math:`g'(x_*)=2-r`，
      :math:`\lambda=\log|2-r|`（:math:`r=2.5` 给 :math:`\log 0.5=-0.693147\ldots`；
      :math:`r=2` 时 :math:`g'=0`，:math:`\lambda=-\infty`）；
    * :math:`3<r<1+\sqrt6`：稳定 2 周期的乘子有闭式 :math:`M=4+2r-r^2`，
      :math:`\lambda=\tfrac12\log|M|`（:math:`r=3.2` 给 :math:`M=0.16`、
      :math:`\lambda=-0.9162907\ldots`）；
    * :math:`r=4`：共轭到 Bernoulli 移位，:math:`\lambda=\log 2`。

    别的参数**不给**解析值（4 周期以上要解高次方程，本章不假装有闭式）。
    """
    if r == 4.0:
        return math.log(2.0)
    if 1.0 < r < 3.0:
        d = abs(2.0 - r)
        return math.log(d) if d > 0.0 else -math.inf
    if 3.0 < r < 1.0 + math.sqrt(6.0):
        m = 4.0 + 2.0 * r - r * r
        return math.log(abs(m)) / 2.0 if m != 0.0 else -math.inf
    raise ValueError("只对 1<r<3、3<r<1+√6 与 r=4 给解析值（更高周期窗口没有闭式）")


def lyapunov_curve(
    r_grid: Any,
    *,
    n_iter: int = 40000,
    transient: int = 2000,
    x0: float = 0.4,
    n_block: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    r"""整条 :math:`\lambda(r)` 曲线（对参数网格**向量化**推进）+ 分块标准误。

    实现是"参数批"版：轨道状态是一个长度 ``n_r`` 的向量，一次迭代同时推进**所有**参数——
    于是 700 个参数 × 4 万次迭代只是一次 Python 循环（实测 ~0.15 s），
    而不是 2800 万次标量调用。这个技巧是分岔图与 λ 曲线能在帧里实时算出来的原因。
    """
    R = np.asarray(r_grid, dtype=float).reshape(-1)
    x = np.full(R.size, float(x0))
    for _ in range(int(transient)):
        x = R * x * (1.0 - x)
    n_block = max(1, int(n_block))
    per = max(1, int(n_iter) // n_block)
    means = np.zeros((n_block, R.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        for b in range(n_block):
            acc = np.zeros(R.size)
            for _ in range(per):
                x = R * x * (1.0 - x)
                d = np.abs(R * (1.0 - 2.0 * x))
                acc += np.log(np.where(d > 0.0, d, np.finfo(float).tiny))
            means[b] = acc / per
    est = np.mean(means, axis=0)
    sem = np.std(means, axis=0, ddof=1) / math.sqrt(n_block) if n_block > 1 else np.zeros(R.size)
    return est, sem


# --------------------------------------------------------------------------- #
# 5. 孪生轨迹分离（蝴蝶效应）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeparationResult:
    """一批孪生轨迹（初值相差 ``eps``）的分离曲线。"""

    t: np.ndarray
    d: np.ndarray
    d_mean: np.ndarray
    base: np.ndarray
    eps: float
    h: float
    dim: int
    #: 全部孪生轨迹本身：形状 ``(n_t, 2*n_pairs, dim)``，前一半是 base、后一半是扰动过的。
    #: 留着它是为了"距离曲线"与"相空间里的那条轨道"来自**同一次积分**——
    #: 分开跑两次会在图上留下两套舍入误差（本章的诚实要求之一）。
    Y: np.ndarray | None = None

    @property
    def n_pairs(self) -> int:
        return int(self.d.shape[1])

    def saturation(self, from_fraction: float = 0.8) -> float:
        """饱和距离：尾段（默认后 20%）的平均分离（吸引子直径的读数）。"""
        return saturation_distance(self.t, self.d_mean, from_fraction=from_fraction)


def twin_separation(
    system: dyn.FlowSystem,
    base: Any,
    *,
    eps: float = 1e-8,
    axis: int = 0,
    h: float = 0.01,
    t_end: float = 30.0,
    method: str = "rk4",
    stride: int = 1,
) -> SeparationResult:
    r"""孪生实验：每对初值只差第 ``axis`` 个分量的 ``eps``，同方法同步长一起推进。

    **为什么必须用同一批初值跑两个步长**：比较"h 与 h/2 下的分离曲线"时，
    如果两次的初值各自用不同步长"落到吸引子上"，比较的其实是吸引子上的两个不同点，
    有限时间指数的涨落（实测跨初值标准差 ±0.045）会把步长效应整个淹掉。
    本函数接受**已经落好的** ``base``，调用方对同一个 ``base`` 跑两次即可
    （:func:`step_size_stability` 就是这么做的）。

    初值差只加在一个分量上（默认 :math:`x`）：差异向量越"一般"越好，
    但为了可复现，绝不掷骰子——方向写死在参数里。
    """
    B = np.asarray(base, dtype=float)
    if B.ndim != 2 or B.shape[1] != system.dim:
        raise ValueError(f"base 形状应为 (n, {system.dim})，得到 {B.shape}")
    pert = B.copy()
    pert[:, int(axis)] += float(eps)
    Y0 = np.concatenate([B, pert], axis=0)
    run = dyn.integrate_ensemble(system, Y0, (0.0, float(t_end)), float(h), method=method, stride=stride)
    n = B.shape[0]
    a = run.Y[:, :n, :]
    b = run.Y[:, n:, :]
    d = np.sqrt(np.sum((a - b) ** 2, axis=2))
    return SeparationResult(
        t=run.t,
        d=d,
        d_mean=np.mean(d, axis=1),
        base=B,
        eps=float(eps),
        h=float(h),
        dim=system.dim,
        Y=run.Y,
    )


def fit_window(t: Any, d: Any, lo: float, hi: float) -> np.ndarray:
    """分离曲线的拟合掩码：``lo ≤ d ≤ hi`` 且有限（窗口是**断言内容的一部分**）。"""
    tt = np.asarray(t, dtype=float).reshape(-1)
    dd = np.asarray(d, dtype=float).reshape(-1)
    return np.isfinite(tt) & np.isfinite(dd) & (dd >= float(lo)) & (dd <= float(hi)) & (dd > 0.0)


def fit_growth_rate(t: Any, d: Any, *, lo: float = 1e-6, hi: float = 1.0) -> tuple[float, int, float]:
    r"""在 :math:`\log d` 上做最小二乘，返回 ``(斜率, 用点数, R²)``。

    自写最小二乘（P3：不碰任何现成拟合），与 :func:`numcore.verify.fit_order` 同一口径：
    :math:`p=\operatorname{cov}(x,y)/\operatorname{var}(x)`。
    返回的斜率单位是 1/时间，物理意义是**这一段有限时间上的指数增长率**——
    它不是 Lyapunov 指数，除非窗口足够长（这件事由跨初值涨落决定，见 :func:`twin_separation`）。
    """
    m = fit_window(t, d, lo, hi)
    n = int(np.count_nonzero(m))
    if n < 3:
        return float("nan"), n, float("nan")
    x = np.asarray(t, dtype=float).reshape(-1)[m]
    y = np.log(np.asarray(d, dtype=float).reshape(-1)[m])
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= 0.0:
        return float("nan"), n, float("nan")
    sxy = float(np.sum((x - xm) * (y - ym)))
    slope = sxy / sxx
    resid = y - (ym + slope * (x - xm))
    ss_tot = float(np.sum((y - ym) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0.0 else 1.0
    return slope, n, r2


def saturation_distance(t: Any, d: Any, *, from_fraction: float = 0.8) -> float:
    """尾段平均分离（默认后 20% 时间）：轨道已被吸引子"忘掉"初值差之后的平均距离。"""
    tt = np.asarray(t, dtype=float).reshape(-1)
    dd = np.asarray(d, dtype=float).reshape(-1)
    if tt.size == 0:
        return float("nan")
    cut = float(tt[0]) + float(from_fraction) * (float(tt[-1]) - float(tt[0]))
    m = tt >= cut
    if not np.any(m):
        return float("nan")
    return float(np.mean(dd[m]))


def resolvable_time(d0: float, lam: float, delta: float) -> float:
    r"""从 :math:`d_0` 长到 :math:`\delta` 需要多久：:math:`t=\log(\delta/d_0)/\lambda`。

    这就是"1e-8 的初值差要多久才**看得出来**"的解析答案：分母用哪个 :math:`\lambda`
    是这件事的全部争议（用公认值 0.9056 得到的是一条理想线，用实测有限时间增长率
    得到的往往更短）。
    """
    if not d0 > 0.0 or not delta > 0.0 or lam <= 0.0:
        return float("nan")
    return math.log(float(delta) / float(d0)) / float(lam)


def step_size_stability(
    system: dyn.FlowSystem,
    base: Any,
    *,
    eps: float = 1e-8,
    h: float = 0.01,
    t_end: float = 30.0,
    lo: float = 1e-6,
    hi: float = 1.0,
) -> dict[str, float]:
    r"""步长减半的实测证据：同一个 ``base``、同一个窗口，两条分离曲线的差。

    返回点估计与三项差值：

    * ``rate_coarse`` / ``rate_fine``：:math:`h` 与 :math:`2h`（即"细"是相对量）两条曲线的
      拟合增长率；
    * ``rel_curve_diff``：两条平均分离曲线在 :math:`d\ge10^{-6}` 段的最大相对差；
    * ``rate_diff``：两个增长率的差。

    阈值不写在函数里：**判据留给断言**（"差小于多少才算稳定"是本章要讨论的事，
    不是这个函数该替你决定的事）。
    """
    fine = twin_separation(system, base, eps=eps, h=float(h), t_end=t_end)
    coarse = twin_separation(system, base, eps=eps, h=float(h) * 2.0, t_end=t_end)
    rf, _, _ = fit_growth_rate(fine.t, fine.d_mean, lo=lo, hi=hi)
    rc, _, _ = fit_growth_rate(coarse.t, coarse.d_mean, lo=lo, hi=hi)
    # 两条曲线的采样点数不同（步长差一倍），逐点比较必须先把细的那条插到粗的格点上。
    # 这里用**索引对齐**（细的每 k 个点对应粗的一个点）+ 端点核对，而不是重新积分：
    # t 的累积在两条路径上可能差 1e-15，但状态差按设计本来就存在（这正是要比较的东西）。
    ratio = max(1, int(round((fine.t.size - 1) / max(coarse.t.size - 1, 1))))
    fine_at = fine.d_mean[::ratio][: coarse.t.size]
    m = fit_window(coarse.t, coarse.d_mean, lo, hi) & fit_window(coarse.t, fine_at, lo, hi)
    rel = float("nan")
    if np.any(m):
        a = fine_at[m]
        b = coarse.d_mean[m]
        rel = float(np.max(np.abs(a - b) / np.maximum(b, np.finfo(float).tiny)))
    return {
        "rate_fine": rf,
        "rate_coarse": rc,
        "rate_diff": abs(rf - rc),
        "rel_curve_diff": rel,
        "n_fine": int(fine.t.size),
        "n_coarse": int(coarse.t.size),
        "align_ratio": int(ratio),
        "t_gap": float(abs(fine.t[::ratio][coarse.t.size - 1] - coarse.t[-1])),
        "saturation": float(fine.saturation()),
    }


# --------------------------------------------------------------------------- #
# 6. 配角的两个标尺
# --------------------------------------------------------------------------- #
def kaplan_yorke_dimension(exponents: Any) -> float:
    r"""Kaplan--Yorke 维数 :math:`D_{KY}=j+\dfrac{\sum_{i\le j}\lambda_i}{|\lambda_{j+1}|}`。

    :math:`j` 是"最大的使前缀和非负"的个数。三维耗散流里它给出吸引子的分形维数
    估计（Lorenz 的公认值约 2.06）；它是**从指数推出来的**，所以不能当作独立证据，
    只能当作一致性检查。
    """
    lam = np.sort(np.asarray(exponents, dtype=float))[::-1]
    csum = np.cumsum(lam)
    j = int(np.count_nonzero(csum >= 0.0))
    if j == 0:
        return 0.0
    if j >= lam.size:
        return float(lam.size)
    denom = abs(float(lam[j]))
    return float(j) + float(csum[j - 1]) / denom if denom > 0.0 else float(lam.size)


def divergence_expectation(system: dyn.FlowSystem, samples: Any) -> float:
    r"""散度 :math:`\nabla\cdot f` 在样本点上的平均（= 全指数和的期望）。

    对 Lorenz 它是常数 :math:`-(\sigma+1+\beta)`（解析）；对 Rössler 是
    :math:`a+x-c`（依赖状态）。**"指数和 = 散度平均"是本章唯一的绝对标尺**：
    它不依赖任何文献值。
    """
    Y = np.asarray(samples, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    n, dim = Y.shape
    h = 1e-6
    div = np.zeros(n)
    for k in range(dim):
        e = np.zeros(dim)
        e[k] = h
        fp = np.asarray(system.rhs(0.0, Y + e), dtype=float).reshape(n, dim)[:, k]
        fm = np.asarray(system.rhs(0.0, Y - e), dtype=float).reshape(n, dim)[:, k]
        div += (fp - fm) / (2.0 * h)
    return float(np.mean(div))
