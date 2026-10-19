r"""6.1 章内核之一：二维 Gray--Scott 反应扩散系统（周期边界、显式 FTCS）。

数学对象
--------
单位正方形 :math:`\Omega=[0,1)^2` 上的 Gray--Scott 系统

.. math::
    u_t = D_u\,\Delta u - u v^2 + F(1-u), \qquad
    v_t = D_v\,\Delta v + u v^2 - (F+k)\,v,

其中 ``feed`` :math:`=F`（补给率）、``kill`` :math:`=k`（额外清除率）。它**不守恒**：
周期域上扩散项求和恒为 0，于是

.. math::
    \frac{\mathrm{d}}{\mathrm{d}t}\sum_{ij} u_{ij}
      = \sum_{ij}\bigl[F(1-u_{ij}) - u_{ij}v_{ij}^2\bigr]
      = \text{:func:`reaction_balance`},

这是离散意义下**精确**的恒等式（不是近似），:math:`\sum v` 同理没有这样的一行
——:math:`v` 的反应项 :math:`\sum(uv^2-(F+k)v)` 与 :math:`u` 的**不**相消，
只有"两者之和"才守恒，而这一章只把 :math:`\sum u` 当观测量（见 :data:`RDResult.mass_history`）。

两个稳态，以及"经典参数根本不是图灵失稳"（本章第一个坑）
----------------------------------------------------------
1. **平凡态** :math:`(u_*,v_*)=(1,0)` 对一切 :math:`F,k` 都存在，且

   .. math::
       J_\text{triv}=\begin{pmatrix}-F & 0\\ 0 & -(F+k)\end{pmatrix}

   是**对角**的。加上扩散仍然是 :math:`\mathrm{diag}(-F-D_u\varphi^2,\,-(F+k)-D_v\varphi^2)`，
   特征值全负、对一切 :math:`\varphi` 成立：**平凡态永远不会扩散失稳**。
   实测（本模块 :func:`most_unstable_mode`）：经典斑点参数
   :math:`(D_u,D_v,F,k)=(2\times10^{-5},\,1\times10^{-5},\,0.037,\,0.060)`
   给出 :math:`\varphi^*=0`、:math:`\lambda^*=-0.037<0`。也就是说，
   教科书上那些漂亮的 Gray--Scott 斑图**不是图灵分岔**，而是激励介质的自复制瞬态；
   把"斑点预设"直接当成图灵靶子，是这一章最容易写出来的假证据。
2. **非平凡稳态**存在当且仅当 :math:`4(F+k)^2<F`：

   .. math::
       u_\pm=\frac{1\mp\sqrt{1-4(F+k)^2/F}}{2},\qquad v_\pm=\frac{F+k}{u_\pm},

   两支的稳定性**相反**：:math:`\det J=(F+k)(v^2-F)`，实测 :math:`u_-` 那支
   :math:`\det J>0`（可取作图灵底态），:math:`u_+` 那支恒有 :math:`\det J<0`（鞍点）。
   "随手取 :math:`+\sqrt{\cdot}`"就把靶子换成了鞍点：底态一错，下面整条色散关系全错。
   本模块固定取**稳定支** :math:`u_-`（:func:`turing_base_state` 返回它；
   非平凡稳态不存在时退回平凡态）。

线性化与解析色散关系（A 类断言用的靶子）
----------------------------------------
在底态 :math:`(u_*,v_*)` 处令 :math:`\Delta\to-\varphi^2`（连续波数 :math:`\varphi`），
得 :math:`2\times2` 矩阵 :math:`M(\varphi)=J-\varphi^2\,\mathrm{diag}(D_u,D_v)`，

.. math::
    \operatorname{tr}M = f_u+g_v-(D_u+D_v)\varphi^2,\qquad
    \det M = D_uD_v\varphi^4-(D_ug_v+D_vf_u)\varphi^2+\det J,

:func:`turing_growth_rate` 返回 :math:`\lambda_{\max}(\varphi)=\Re` 的最大特征值：

.. math::
    \lambda_{\max}=
    \begin{cases}
      \dfrac{\operatorname{tr}M+\sqrt{\operatorname{tr}^2M-4\det M}}{2}, & \operatorname{tr}^2M\ge 4\det M,\\[2mm]
      \dfrac{\operatorname{tr}M}{2}, & \text{否则（复共轭）}.
    \end{cases}

图灵失稳的判别式是 :math:`\min_{\varphi\ge0}\det M<0`（此时必为实特征值）。
:math:`\det M` 是 :math:`z=\varphi^2` 的凸二次式，顶点 :math:`z_*=\frac{D_ug_v+D_vf_u}{2D_uD_v}`；
而 :math:`\det<0` 时 :math:`\lambda_{\max}` 对 :math:`\det` **单调递减**
（:math:`\partial\lambda/\partial\det=-1/\sqrt{\operatorname{tr}^2-4\det}<0`），
所以图灵区里 :math:`\varphi^*=\sqrt{z_*}` **是闭式的**，:func:`most_unstable_mode`
在连续波数上就取这个闭式（不是扫网格点），非图灵区才退回"粗扫 + 黄金分割"。

这一章最容易骗人的地方（数字都是实测，见 ``tests/6.1``）
----------------------------------------------------------
1. **"经典斑图 = 图灵斑图"是错的**（上面第 1 条）：:math:`\lambda^*=-0.037<0`。
2. **:math:`\lambda^*>0` 也不等于图灵**：底态本身不稳定（:math:`v_*^2<k`）时
   :math:`\lambda_{\max}(0)=\operatorname{tr}J/2>0`，最大值在 :math:`\varphi=0`
   ——那是整片同相振荡，不是空间图案。实测反例：
   :math:`(D_u,D_v,F,k)=(10^{-5},10^{-5},0.02,0.05)` 给
   :math:`\varphi^*=0`、:math:`\lambda^*=+0.0117>0`，而时间推进里
   :math:`u` 的空间振幅**不增长**（图案振幅 2.0e-16 量级）。
   所以判据必须是"底态无扩散稳定 **且** :math:`\lambda^*>0`"两条一起。
3. **单位正方形上"离散波数"是 n 无关的**：域长固定为 1 时允许的波数是
   :math:`\varphi_m=2\pi m`（:math:`m` 为整数），加密网格**只抬高截止**
   :math:`\varphi_{\max}=\pi n`，不改变波数栅格。于是
   :func:`dominant_wavelength` 对同一物理图案在不同 n 上给出**同一个**峰值波长
   （实测 n=64/96/128 都是 0.1）——"加密网格能提高波长精度"这句话在这里是假的，
   真正按 :math:`O(\Delta x^2)` 收敛的是**增长率**（五点模板的有效波数
   :math:`\varphi_\text{eff}=\frac{2}{\Delta x}\sin\frac{\varphi\Delta x}{2}`）。
   用 n=32/64/128 实测模态增长率误差对解析值：2.78e-2 → 7.44e-3 → 1.91e-3，
   幂律拟合阶 1.93（理论 2）。
4. **显式稳定性上限只与"最大的那个扩散系数"有关**，与 :math:`F,k` 无关：
   :math:`\Delta t\le\Delta x^2/(4\max(D_u,D_v))`（二维五点模板的放大因子下界
   :math:`1-8D\Delta t/\Delta x^2\ge-1`）。:func:`diffusion_dt_limit` 默认再乘
   安全系数 0.5；二分实测的失稳阈值与 :math:`\Delta x^2/(4D_u)` 相对差 < 1e-6
   （见 :func:`gray_scott_solve` 的 ``clamp`` 与 ``dt_limit``：越界一律**如实报告**，
   绝不静默发散，也不让 NaN 泄漏出去）。
5. **局部极值计数只在饱和图案上有意义**：线性增长期 :math:`v\sim10^{-6}` 时
   8 邻域局部极大值满地都是（实测线性期 :func:`count_spots` 明显多于饱和期）。

数值约定
--------
* 域取单位正方形，网格 :math:`\Delta x=1/n`（:func:`laplacian_periodic` 返回**未除**
  :math:`\Delta x^2` 的五点模板和，除 :math:`\Delta x^2` 由 :func:`gray_scott_rhs` 负责）；
* 周期边界、显式 FTCS（前向 Euler + 五点模板），教学上要让"显式稳定性上限"看得见；
* 纯函数、无 IO、无全局可变状态（P6）；随机性只经 :func:`numcore.utils.rng`（P4）；
* 二维谱只用自研 FFT :func:`numcore.transform.fft.fft`（P2：不碰 numpy 的 fft/linalg 子模块，
  P3：不用任何成品算法）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from numcore.transform.fft import fft
from numcore.utils import rng

__all__ = [
    "RD_PRESETS",
    "RDParams",
    "RDResult",
    "laplacian_periodic",
    "gray_scott_rhs",
    "gray_scott_step",
    "gray_scott_solve",
    "diffusion_dt_limit",
    "initial_condition",
    "turing_base_state",
    "turing_growth_rate",
    "turing_dispersion",
    "most_unstable_mode",
    "count_spots",
    "dominant_wavelength",
    "pattern_metrics",
    "total_u",
    "total_v",
    "reaction_balance",
]

#: 预设参数：名字 -> ``(Du, Dv, F, k)``。
#:
#: ``spots`` / ``stripes`` / ``waves`` 是 Gray--Scott 文献里的经典自复制区参数
#: （:math:`D_u:D_v=2:1`，量级 :math:`10^{-5}`，在 :math:`\Delta x=1/128` 上
#: 图案波长 ~10 个网格）；``uniform`` 是**纯扩散无图灵**的对照组
#: （:math:`F=k=0`，平凡态 Jacobian 为零矩阵，:math:`\lambda_{\max}(\varphi)=-D\varphi^2\le0`）；
#: ``turing`` 是唯一**真正满足图灵条件**的一组（非平凡稳态存在、无扩散稳定、
#: 有扩散失稳，:math:`\varphi^*\approx62.8`、:math:`\lambda^*\approx1.3\times10^{-2}`、波长 ~0.1）。
#:
#: 前三个预设的解析色散关系是**负**的（:math:`-0.037`、:math:`-0.035`、:math:`-0.014`）：
#: 它们的图案来自激励介质的自复制动力学，不是图灵分岔——这不是 bug，是本模块
#: 要教的第一个反例（见模块 docstring 第 1 条）。
RD_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "spots": (2.0e-5, 1.0e-5, 0.035, 0.060),
    "stripes": (2.0e-5, 1.0e-5, 0.030, 0.055),
    "waves": (2.0e-5, 1.0e-5, 0.014, 0.054),
    "uniform": (2.0e-5, 2.0e-5, 0.0, 0.0),
    "turing": (8.0e-5, 8.0e-6, 0.030, 0.045),
}


@dataclass(frozen=True)
class RDParams:
    """一次 Gray--Scott 仿真的全部输入（``n`` 为每边格点数，域恒为单位正方形）。"""

    n: int = 128
    du: float = 2.0e-5
    dv: float = 1.0e-5
    feed: float = 0.035
    kill: float = 0.060
    dt: float = 0.25
    steps: int = 2000


@dataclass
class RDResult:
    """一次仿真的全部产出；振幅/斑点数/质量历史都是**观测量**，不是参数。"""

    u: np.ndarray
    v: np.ndarray
    n: int
    dx: float
    dt: float
    steps: int
    time: float
    amplitude: float
    spot_count: int
    total_u: float
    total_v: float
    mass_history: np.ndarray
    vmax_history: np.ndarray
    #: 本次实际用的 ``dt`` 是否被 :func:`diffusion_dt_limit` 夹过（越界必须如实报告）
    dt_clamped: bool = False
    #: 本次 ``(n, du, dv)`` 下的显式稳定性上限（含默认安全系数 0.5）
    dt_limit: float = 0.0
    #: 全程是否有限（夹过 dt 后应为 True；NaN/Inf 不允许静默溜出去）
    finite: bool = True
    #: 夹 dt 前的请求值，便于"我请求了 0.9、实际跑了 0.38"这类对账
    dt_requested: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 空间算子与 RHS
# --------------------------------------------------------------------------- #
def _laplacian_into(lap: np.ndarray, a: np.ndarray) -> None:
    """把五点模板的周期和写进 ``lap``（未除 :math:`\\Delta x^2`）。"""
    np.multiply(a, -4.0, out=lap)
    lap[1:, :] += a[:-1, :]
    lap[0, :] += a[-1, :]
    lap[:-1, :] += a[1:, :]
    lap[-1, :] += a[0, :]
    lap[:, 1:] += a[:, :-1]
    lap[:, 0] += a[:, -1]
    lap[:, :-1] += a[:, 1:]
    lap[:, -1] += a[:, 0]


def laplacian_periodic(field: Any) -> np.ndarray:
    r""":math:`(\mathcal{L}u)_{ij}=u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{ij}`（周期、未除 :math:`\Delta x^2`）。

    返回的是**分子**：真正的 Laplace 算子是 :math:`\mathcal{L}/\Delta x^2`。
    周期边界下每行/每列都恰好被加两次、减两次，所以
    :math:`\sum_{ij}(\mathcal{L}u)_{ij}=0` **精确成立**（浮点意义下到舍入，
    见 ``tests/6.1`` 的 B 类断言）——这就是"质量只能被反应项改变"的全部理由。
    """
    a = np.asarray(field, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"需要二维场，得到 ndim={a.ndim}")
    if min(a.shape) < 3:
        raise ValueError(f"周期五点模板至少要 3×3，得到 {a.shape}")
    lap = np.empty_like(a)
    _laplacian_into(lap, a)
    return lap


def gray_scott_rhs(
    u: Any,
    v: Any,
    du: float,
    dv: float,
    feed: float,
    kill: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Gray--Scott 的右端 :math:`(u_t,v_t)`（周期五点模板，含 :math:`1/\Delta x^2`）。

    .. math::
        u_t = \frac{D_u}{\Delta x^2}\mathcal{L}u - uv^2 + F(1-u),\qquad
        v_t = \frac{D_v}{\Delta x^2}\mathcal{L}v + uv^2 - (F+k)v .
    """
    a = np.asarray(u, dtype=float)
    b = np.asarray(v, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"u 与 v 形状必须一致，得到 {a.shape} 与 {b.shape}")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError(f"dx 必须是正有限数，得到 {dx}")
    inv = 1.0 / (dx * dx)
    uv2 = a * b * b
    du_lap = laplacian_periodic(a) * (du * inv)
    dv_lap = laplacian_periodic(b) * (dv * inv)
    return du_lap - uv2 + feed * (1.0 - a), dv_lap + uv2 - (feed + kill) * b


def gray_scott_step(
    u: Any,
    v: Any,
    du: float,
    dv: float,
    feed: float,
    kill: float,
    dx: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""显式 FTCS 推进一步：:math:`u^{n+1}=u^n+\Delta t\,u_t(u^n,v^n)`（反应项同时用旧值）。

    这是教学上"看得见稳定性上限"的格式：:math:`\Delta t>\Delta x^2/(4\max(D_u,D_v))`
    时高频模态的放大因子 :math:`|1-8D\Delta t/\Delta x^2|>1`，一步就放大、很快溢出
    （:func:`diffusion_dt_limit` 是它的上界；:func:`gray_scott_solve` 默认夹住它）。
    """
    fu, fv = gray_scott_rhs(u, v, du, dv, feed, kill, dx)
    return np.asarray(u, dtype=float) + dt * fu, np.asarray(v, dtype=float) + dt * fv


# --------------------------------------------------------------------------- #
# 时间步上限与主求解器
# --------------------------------------------------------------------------- #
def diffusion_dt_limit(n: int, du: float, dv: float, *, safety: float = 0.5) -> float:
    r"""二维五点模板 + 前向 Euler 的显式稳定性上限 :math:`\Delta t\le\frac{\Delta x^2}{4\max(D_u,D_v)}`。

    推导：五点 Laplace 的 Fourier 符号取值 :math:`[-8/\Delta x^2,0]`，放大因子
    :math:`1-8D\Delta t/\Delta x^2` 必须落在 :math:`[-1,1]`，故
    :math:`\Delta t\le\Delta x^2/(4D)`。``safety`` 是再乘的安全系数
    （默认 0.5，即只用到上限的一半）；``safety=1.0`` 是理论边界本身，
    ``tests/6.1`` 用二分实测的阈值与它对比。:math:`D_u=D_v=0`（无扩散）时返回 ``inf``。
    """
    if n < 3:
        raise ValueError(f"n 至少为 3，得到 {n}")
    if du < 0.0 or dv < 0.0:
        raise ValueError(f"扩散系数必须非负，得到 du={du}, dv={dv}")
    if not 0.0 < safety <= 1.0:
        raise ValueError(f"safety 必须落在 (0,1]，得到 {safety}")
    dmax = max(float(du), float(dv))
    if dmax == 0.0:
        return float("inf")
    dx = 1.0 / float(n)
    return float(safety) * dx * dx / (4.0 * dmax)


def initial_condition(
    n: int,
    *,
    kind: str = "spots",
    seed: int = 20240601,
) -> tuple[np.ndarray, np.ndarray]:
    r"""初始条件 :math:`(u^0,v^0)`；``kind`` 见下表。

    ==========  ==================================================================
    ``kind``     内容
    ==========  ==================================================================
    ``spots``    中心方块 :math:`u=0.5,v=0.25`（经典 Sims 种子）+ 打破对称的抖动
    ``stripes``  横贯的条带 :math:`u=0.5,v=0.25` + 抖动
    ``waves``    两个错开的方块（一上一下），让波前有地方跑
    ``ring``     中心圆环
    ``uniform``  :math:`u\equiv1,\ v\equiv0`（平凡态，什么都不长）
    ``random``   :math:`u=1-0.5\xi,\ v=0.25\xi`，:math:`\xi\sim U[0,1)`（全局噪声种子）
    ==========  ==================================================================

    抖动由 :func:`numcore.utils.rng` 产生（P4：确定性种子）。**抖动不是装饰**：
    完美对称的种子方块在周期方域上会一直保持四重对称，图案看起来"假"
    （见 ``pitfalls``）。
    """
    if n < 4:
        raise ValueError(f"n 至少为 4，得到 {n}")
    u = np.ones((n, n), dtype=float)
    v = np.zeros((n, n), dtype=float)
    gen = rng(int(seed))
    if kind == "uniform":
        return u, v
    if kind == "random":
        xi = gen.random((n, n))
        return 1.0 - 0.5 * xi, 0.25 * xi
    if kind == "spots":
        s = max(4, n // 8)
        lo = (n - s) // 2
        mask = np.zeros((n, n), dtype=bool)
        mask[lo : lo + s, lo : lo + s] = True
    elif kind == "stripes":
        s = max(4, n // 8)
        lo = (n - s) // 2
        mask = np.zeros((n, n), dtype=bool)
        mask[lo : lo + s, :] = True
    elif kind == "waves":
        s = max(3, n // 16)
        mask = np.zeros((n, n), dtype=bool)
        mask[n // 4 : n // 4 + s, n // 4 : n // 4 + s] = True
        mask[3 * n // 4 : 3 * n // 4 + s, 3 * n // 4 : 3 * n // 4 + s] = True
    elif kind == "ring":
        x = (np.arange(n, dtype=float) + 0.5) / n - 0.5
        r = np.sqrt(x[:, None] ** 2 + x[None, :] ** 2)
        mask = (r > 0.18) & (r < 0.18 + max(2.0 / n, 0.02))
    else:
        raise ValueError(f"未知的 kind={kind!r}；可选：spots/stripes/waves/ring/uniform/random")
    jitter = 1.0e-2 * gen.random((n, n))
    u = np.where(mask, 0.5 + jitter, u)
    v = np.where(mask, 0.25 + jitter, v)
    return u, v


def gray_scott_solve(
    params: RDParams,
    *,
    u0: Any = None,
    v0: Any = None,
    seed: int = 20240601,
    clamp: bool = True,
) -> RDResult:
    r"""按 :class:`RDParams` 推进 ``steps`` 步，返回 :class:`RDResult`（每步都记质量与 :math:`\max v`）。

    ``dt`` 越界（:math:`>\Delta t_{\lim}` = :func:`diffusion_dt_limit`）时**绝不静默发散**：

    * ``clamp=True``（默认）：夹到 :math:`\Delta t_{\lim}`，结果里
      ``dt_clamped=True``、``dt_limit``、``dt_requested`` 三个字段如实对上账；
    * ``clamp=False``：直接 ``ValueError``（"我要的就是越界"不是合法请求）。

    ``u0/v0`` 缺省时用 :func:`initial_condition` 的经典种子（``seed`` 只影响抖动与
    ``kind="random"``）。时间推进走预分配缓冲区的原地版本，``n=128, steps=2000``
    实测 0.3 s 量级（帧里可以放心调用）。
    """
    n = int(params.n)
    if n < 4:
        raise ValueError(f"n 至少为 4，得到 {n}")
    steps = int(params.steps)
    if steps < 0:
        raise ValueError(f"steps 必须非负，得到 {steps}")
    du, dv = float(params.du), float(params.dv)
    feed, kill = float(params.feed), float(params.kill)
    if du < 0.0 or dv < 0.0:
        raise ValueError(f"扩散系数必须非负，得到 du={du}, dv={dv}")
    dx = 1.0 / n
    limit = diffusion_dt_limit(n, du, dv)
    dt = float(params.dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt 必须是正有限数，得到 {dt}")
    clamped = False
    if dt > limit:
        if not clamp:
            raise ValueError(
                f"dt={dt:g} 超过显式稳定性上限 dt_limit={limit:g}"
                f"（n={n}, max(Du,Dv)={max(du, dv):g}）——clamp=False 时拒绝静默发散"
            )
        dt = limit
        clamped = True
    if u0 is None or v0 is None:
        u_init, v_init = initial_condition(n, kind="spots", seed=seed)
    else:
        u_init = np.array(u0, dtype=float, copy=True)
        v_init = np.array(v0, dtype=float, copy=True)
    u = np.ascontiguousarray(u_init, dtype=float)
    v = np.ascontiguousarray(v_init, dtype=float)
    if u.shape != (n, n) or v.shape != (n, n):
        raise ValueError(f"u0/v0 形状必须是 ({n},{n})，得到 {u.shape} 与 {v.shape}")

    mass = np.empty(steps + 1, dtype=float)
    vmax = np.empty(steps + 1, dtype=float)
    mass[0] = float(u.sum())
    vmax[0] = float(v.max())
    # 预分配的工作区：lap_u/lap_v 放五点模板，f/g 放反应项与增量。
    lap_u = np.empty_like(u)
    lap_v = np.empty_like(v)
    uv2 = np.empty_like(u)
    f = np.empty_like(u)
    g = np.empty_like(v)
    inv_dx2 = 1.0 / (dx * dx)
    kd = du * inv_dx2
    kd_v = dv * inv_dx2
    dt_f = float(dt)
    for s in range(steps):
        _laplacian_into(lap_u, u)
        _laplacian_into(lap_v, v)
        np.multiply(u, v, out=uv2)
        np.multiply(uv2, v, out=uv2)  # uv^2
        np.multiply(u, -feed, out=f)
        f += feed
        f -= uv2
        np.multiply(lap_u, kd, out=lap_u)
        lap_u += f
        np.multiply(lap_u, dt_f, out=f)
        u += f
        np.multiply(v, -(feed + kill), out=g)
        g += uv2
        np.multiply(lap_v, kd_v, out=lap_v)
        lap_v += g
        np.multiply(lap_v, dt_f, out=g)
        v += g
        mass[s + 1] = float(u.sum())
        vmax[s + 1] = float(v.max())
    finite = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v)))
    amp = float(u.max() - u.min()) if finite else float("nan")
    return RDResult(
        u=u,
        v=v,
        n=n,
        dx=dx,
        dt=float(dt),
        steps=steps,
        time=float(steps) * float(dt),
        amplitude=amp,
        spot_count=count_spots(v) if finite else 0,
        total_u=float(u.sum()),
        total_v=float(v.sum()),
        mass_history=mass,
        vmax_history=vmax,
        dt_clamped=clamped,
        dt_limit=float(limit),
        finite=finite,
        dt_requested=float(params.dt),
        extra={"feed": feed, "kill": kill, "du": du, "dv": dv},
    )


# --------------------------------------------------------------------------- #
# 解析色散关系（A 类靶子）
# --------------------------------------------------------------------------- #
def turing_base_state(feed: float, kill: float) -> tuple[float, float] | None:
    r"""非平凡稳态的**稳定支** :math:`(u_-,v_-)`；不存在（:math:`4(F+k)^2\ge F`）时返回 ``None``。

    返回的是 :math:`u_-=\frac{1-\sqrt{1-4(F+k)^2/F}}{2}`（小 :math:`u`、大 :math:`v` 的那支）：
    只有它满足 :math:`\det J=(F+k)(v^2-F)>0`；另一支 :math:`u_+` 恒为 :math:`\det J<0` 的鞍点。
    """
    f = float(feed)
    k = float(kill)
    if f <= 0.0:
        return None
    disc = 1.0 - 4.0 * (f + k) ** 2 / f
    if disc <= 0.0:
        return None
    u = 0.5 * (1.0 - np.sqrt(disc))
    if u <= 0.0:
        return None
    return float(u), float((f + k) / u)


def _reaction_jacobian(feed: float, kill: float) -> tuple[float, float, float, float]:
    r""":math:`J=(f_u,f_v,g_u,g_v)`，在稳定非平凡稳态上（不存在则退回平凡态 :math:`(1,0)`）。

    稳态上 :math:`uv=F+k`，于是 :math:`f_v=-2(F+k)`、:math:`g_v=2uv-(F+k)=F+k`
    ——这两个等式是下面所有闭式的来源。
    """
    base = turing_base_state(feed, kill)
    if base is None:
        u, v = 1.0, 0.0
    else:
        u, v = base
    return (-v * v - feed, -2.0 * u * v, v * v, 2.0 * u * v - (feed + kill))


def _det_from_z(
    z: Any,
    fu: float,
    fv: float,
    gu: float,
    gv: float,
    du: float,
    dv: float,
) -> np.ndarray:
    r""":math:`\det M(z)=D_uD_vz^2-(D_ug_v+D_vf_u)z+\det J`（:math:`z=\varphi^2`，向量化）。"""
    zz = np.asarray(z, dtype=float)
    return (fu - du * zz) * (gv - dv * zz) - fv * gu


def _rate_from_z(
    z: Any,
    fu: float,
    fv: float,
    gu: float,
    gv: float,
    du: float,
    dv: float,
) -> np.ndarray:
    r""":math:`\lambda_{\max}` 作为 :math:`z=\varphi^2` 的函数（向量化）。"""
    zz = np.asarray(z, dtype=float)
    tr = (fu + gv) - (du + dv) * zz
    det = _det_from_z(zz, fu, fv, gu, gv, du, dv)
    disc = tr * tr - 4.0 * det
    root = np.sqrt(np.where(disc > 0.0, disc, 0.0))
    return np.where(disc > 0.0, 0.5 * (tr + root), 0.5 * tr)


def turing_growth_rate(phi: Any, du: float, dv: float, feed: float, kill: float) -> Any:
    r"""解析色散关系 :math:`\lambda_{\max}(\varphi)`（:math:`\varphi` 为**连续**波数，弧度/单位长）。

    在稳定底态（:func:`turing_base_state`，不存在时退回平凡态 :math:`(1,0)`）线性化，
    把 :math:`\Delta` 换成 :math:`-\varphi^2`，取 :math:`M(\varphi)=J-\varphi^2\mathrm{diag}(D_u,D_v)`
    的最大特征值实部。**这是 A 类断言的解析靶子**：它与"真的把非线性求解器推进几步、
    量模态振幅增长"必须给同一个数（差一个 :math:`O(\Delta x^2)` 的离散化项，
    见模块 docstring 第 3 条）。

    标量输入返回 ``float``，数组输入返回同形状数组。
    """
    fu, fv, gu, gv = _reaction_jacobian(feed, kill)
    out = _rate_from_z(np.asarray(phi, dtype=float) ** 2, fu, fv, gu, gv, float(du), float(dv))
    if np.ndim(phi) == 0:
        return float(out)
    return out


def _dispersion_phi_max(du: float, dv: float, feed: float, kill: float) -> float:
    """色散曲线的绘图上界：覆盖最不稳定带并留出衰减段。"""
    fu, fv, gu, gv = _reaction_jacobian(feed, kill)
    dmin = min(float(du), float(dv))
    if dmin <= 0.0:
        return 1.0
    scale = max(abs(fu) + abs(gv), abs(fu * gv - fv * gu) ** 0.5, float(feed) + float(kill))
    if scale <= 0.0:
        scale = 1.0
    return float(4.0 * np.sqrt(scale / dmin))


def turing_dispersion(
    du: float,
    dv: float,
    feed: float,
    kill: float,
    n_modes: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    r"""色散曲线的采样：``(phi, rate)``，:math:`\varphi\in[0,\varphi_{\max}]` 等距 ``n_modes`` 点。

    上界取 :math:`\varphi_{\max}=4\sqrt{\text{scale}/\min(D_u,D_v)}`，
    ``scale`` 取 :math:`|f_u|+|g_v|`、:math:`\sqrt{|\det J|}`、:math:`F+k` 的最大值
    （纯扩散退化时取 1）——足够覆盖最不稳定带与它的衰减尾。
    """
    if int(n_modes) < 8:
        raise ValueError(f"n_modes 至少为 8，得到 {n_modes}")
    phi = np.linspace(0.0, _dispersion_phi_max(du, dv, feed, kill), int(n_modes))
    return phi, np.asarray(turing_growth_rate(phi, du, dv, feed, kill), dtype=float)


def _quadratic_roots_real(a: float, b: float, c: float) -> list[float]:
    r"""实系数二次式 :math:`az^2+bz+c=0` 的实根（自己写，P3 不许用成品求根）。"""
    if a == 0.0:
        if b == 0.0:
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    root = np.sqrt(disc)
    # 数值稳定写法：先算同号那一个根，再用韦达定理得另一个（避免灾难性相消）。
    q = -0.5 * (b + np.copysign(root, b))
    out = []
    if q != 0.0:
        out.append(q / a)
        out.append(c / q)
    else:  # pragma: no cover - b=0 且 disc=0 的退化情形
        out.append(0.0)
    return [float(z) for z in out if np.isfinite(z)]


def most_unstable_mode(
    du: float,
    dv: float,
    feed: float,
    kill: float,
    n_modes: int = 512,
) -> tuple[float, float]:
    r"""连续波数上最大化 :math:`\lambda_{\max}(\varphi)`，返回 :math:`(\varphi^*,\lambda^*)`。

    最大点**不是** :math:`\det M` 的顶点（这是一个漂亮的陷阱：:math:`\det<0` 时
    :math:`\partial\lambda/\partial\det<0`，但沿 :math:`z` 走时 :math:`\operatorname{tr}M`
    同时在下降，两项互相竞争）。真正的驻点条件是 :math:`D'(z)=s`，其中

    .. math::
        D(z)=\sqrt{\operatorname{tr}^2M-4\det M}=\sqrt{\alpha z^2+\beta z+\gamma},\quad
        \alpha=(D_u+D_v)^2-4D_uD_v,\ \ \beta=-2\bigl((D_u+D_v)(f_u+g_v)-2(D_ug_v+D_vf_u)\bigr),

    平方后得到一个二次方程（:func:`_quadratic_roots_real` 自己解），加上
    :math:`z=0` 与"复特征值分支在 :math:`z=0` 取最大"这两条，就得到**全局**最大点。
    实测（``turing`` 预设）解析驻点给 :math:`\varphi^*=53.900578`、:math:`\lambda^*=0.0150170`，
    而 :math:`\det` 的顶点 :math:`\sqrt{z_v}=62.7495` 给的是 :math:`0.0133925`——
    **低了 12%**；只信顶点会得到一个系统性偏大 16% 的波长。

    :math:`D_uD_v=0` 等退化情形退回"``n_modes`` 点粗扫 + 黄金分割细化"，
    仍然在**连续**波数上取最大，不是取网格点。
    """
    if int(n_modes) < 8:
        raise ValueError(f"n_modes 至少为 8，得到 {n_modes}")
    du_f, dv_f = float(du), float(dv)
    if du_f < 0.0 or dv_f < 0.0:
        raise ValueError(f"扩散系数必须非负，得到 du={du}, dv={dv}")
    fu, fv, gu, gv = _reaction_jacobian(feed, kill)
    det0 = fu * gv - fv * gu
    a2 = du_f * gv + dv_f * fu  # det(z) = Du*Dv*z^2 - a2*z + det0
    s = du_f + dv_f
    t0 = fu + gv
    p = du_f * dv_f
    z_hi = max(0.0, (t0 / s) if s > 0.0 else 0.0)
    if p > 0.0:
        disc_det = a2 * a2 - 4.0 * p * det0
        if disc_det > 0.0:
            z_hi = max(z_hi, (a2 + np.sqrt(disc_det)) / (2.0 * p))
        z_hi = max(z_hi, a2 / (2.0 * p))
    if z_hi <= 0.0:
        return 0.0, float(_rate_from_z(0.0, fu, fv, gu, gv, du_f, dv_f))

    candidate = [0.0]
    if p > 0.0:
        alpha = s * s - 4.0 * p
        beta = -2.0 * (s * t0 - 2.0 * a2)
        gamma = t0 * t0 - 4.0 * det0
        qc = (s * s * gamma - 0.25 * beta * beta) / (4.0 * p)
        for z in _quadratic_roots_real(alpha, beta, qc):
            if z < 0.0 or z > z_hi:
                continue
            # 平方引入的伪根：必须回到 D'(z)=s 的**未平方**形式（符号条件）。
            if alpha * z + 0.5 * beta < 0.0:
                continue
            if alpha * z * z + beta * z + gamma < 0.0:
                continue
            candidate.append(z)
    # 兜底：粗扫定位 + 黄金分割细化（防驻点方程被舍入弄丢，也覆盖退化情形）。
    grid = np.linspace(0.0, z_hi, int(n_modes))
    vals = np.asarray(_rate_from_z(grid, fu, fv, gu, gv, du_f, dv_f), dtype=float)
    i = int(np.argmax(vals))
    z_gold, _rate_gold = _golden_max(
        float(grid[max(i - 1, 0)]), float(grid[min(i + 1, grid.size - 1)]), fu, fv, gu, gv, du_f, dv_f
    )
    candidate.append(z_gold)
    candidate.append(float(grid[i]))
    rates = np.asarray(
        [_rate_from_z(z, fu, fv, gu, gv, du_f, dv_f) for z in candidate], dtype=float
    )
    j = int(np.argmax(rates))
    z_final = float(candidate[j])
    return float(np.sqrt(max(z_final, 0.0))), float(rates[j])


def _golden_max(
    lo: float,
    hi: float,
    fu: float,
    fv: float,
    gu: float,
    gv: float,
    du: float,
    dv: float,
    *,
    iters: int = 160,
) -> tuple[float, float]:
    """黄金分割在 [lo,hi] 上最大化 :math:`\\lambda_{\\max}`（单峰区间内）。"""
    gr = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = float(lo), float(hi)
    if not b > a:
        return a, float(_rate_from_z(a, fu, fv, gu, gv, du, dv))
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = float(_rate_from_z(c, fu, fv, gu, gv, du, dv))
    fd = float(_rate_from_z(d, fu, fv, gu, gv, du, dv))
    for _ in range(int(iters)):
        if b - a <= 1.0e-16 * max(1.0, abs(b)):
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = float(_rate_from_z(c, fu, fv, gu, gv, du, dv))
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = float(_rate_from_z(d, fu, fv, gu, gv, du, dv))
    if fc > fd:
        return c, fc
    return d, fd


# --------------------------------------------------------------------------- #
# 图案观测量
# --------------------------------------------------------------------------- #
def count_spots(v: Any, *, rel_level: float = 0.2) -> int:
    r"""严格 8 邻域（周期）局部极大值的个数，且极大值须 :math:`>0.2\max v`。

    这就是"斑点计数"：Gray--Scott 的一个斑点在 :math:`v` 上是一个孤立的峰。
    阈值是标定出来的，不是凑的：饱和图案里 :math:`\max v\approx0.3\text{--}0.4`，
    阈值 0.06--0.08 足以滤掉饱和平台上的舍入涟漪；而**线性增长期**
    :math:`v\sim10^{-6}`，一切都是"极大值"，计数没有意义（见 ``pitfalls``）。
    """
    a = np.asarray(v, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"需要二维场，得到 ndim={a.ndim}")
    if a.size == 0:
        return 0
    peak = float(a.max())
    if peak <= 0.0:
        return 0
    flag = a > (float(rel_level) * peak)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            flag &= a > np.roll(np.roll(a, di, axis=0), dj, axis=1)
    return int(np.count_nonzero(flag))


def _power_spectrum_2d(a: np.ndarray) -> np.ndarray:
    r"""二维功率谱 :math:`|F|^2`（两趟自研一维 FFT，P2：不碰 numpy 的 fft 子模块）。"""
    n0, n1 = a.shape
    spec = np.empty((n0, n1), dtype=complex)
    for i in range(n0):
        spec[i, :] = fft(a[i, :])
    col = np.empty(n0, dtype=complex)
    for j in range(n1):
        col[:] = spec[:, j]
        spec[:, j] = fft(col)
    return spec.real * spec.real + spec.imag * spec.imag


def dominant_wavelength(u: Any, dx: float = 1.0) -> float:
    r"""主波长 :math:`\lambda=2\pi/\kappa^*`（单位与 ``dx`` 一致）。

    做法：去均值 → 二维功率谱（自研 FFT）→ 按**整数波数壳** :math:`|k|=\sqrt{k_1^2+k_2^2}`
    做径向平均（单位正方形上允许的波数恰好是 :math:`\varphi=2\pi|k|`，
    所以整数壳就是**精确**的栅格，不是分箱近似）→ 取功率最大的壳 →
    再用壳内功率加权的平均半径把峰值细化到亚壳精度 →
    :math:`\kappa^*=r^*/(n\Delta x)`、:math:`\lambda=2\pi/\kappa^*=n\Delta x/r^*`。

    谱全零（平坦场）时返回 ``inf``：**那是哨兵不是数字**，
    调用方必须按 POLICY §5 写成 ``None`` + ``strings`` 说明，不要塞进帧元数据。
    注意：单位正方形上波数栅格与 :math:`n` 无关，加密网格**不会**改善这个估计
    （模块 docstring 第 3 条）。
    """
    a = np.asarray(u, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"需要二维场，得到 ndim={a.ndim}")
    n0, n1 = a.shape
    if n0 != n1:
        raise ValueError(f"径向平均要求方阵，得到 {a.shape}")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError(f"dx 必须是正有限数，得到 {dx}")
    n = n0
    b = a - float(a.mean())
    power = _power_spectrum_2d(b)
    k = np.arange(n)
    kf = np.where(k <= n // 2, k, k - n).astype(float)
    radius = np.sqrt(kf[:, None] ** 2 + kf[None, :] ** 2)
    shell = np.rint(radius).astype(int)
    shell_max = int(shell.max())
    totals = np.zeros(shell_max + 1, dtype=float)
    for s in range(1, shell_max + 1):
        totals[s] = float(power[shell == s].sum())
    best = int(np.argmax(totals[1:])) + 1
    if totals[best] <= 0.0:
        return float("inf")
    mask = shell == best
    w = power[mask]
    r_mean = float(np.sum(w * radius[mask]) / np.sum(w))
    if r_mean <= 0.0:
        return float("inf")
    return float(n * float(dx) / r_mean)


def total_u(u: Any) -> float:
    r""":math:`\sum_{ij}u_{ij}`（周期域上只有反应项能改变它）。"""
    return float(np.sum(np.asarray(u, dtype=float)))


def total_v(v: Any) -> float:
    r""":math:`\sum_{ij}v_{ij}`（不守恒：:math:`\sum v` 没有像 :math:`\sum u` 那样的闭式右端）。"""
    return float(np.sum(np.asarray(v, dtype=float)))


def reaction_balance(u: Any, v: Any, feed: float, kill: float) -> float:
    r""":math:`\sum_{ij}\bigl[F(1-u_{ij})-u_{ij}v_{ij}^2\bigr]`——:math:`\frac{\mathrm{d}}{\mathrm{d}t}\sum u` 的**精确**离散右端。

    它只用反应项，与 :math:`D_u,D_v,\Delta x` 无关：因为
    :math:`\sum_{ij}(\mathcal{L}u)_{ij}\equiv0`（周期边界，每个点被加两次减两次）。
    前向 Euler 下这条恒等式是**一步精确**的（到舍入）：
    :math:`\bigl(\sum u^{n+1}-\sum u^n\bigr)/\Delta t=\text{reaction\_balance}(u^n,v^n)`。
    ``kill`` 只出现在 :math:`v` 的方程里，这里**不用**它——签名保留它是为了和别的
    RHS 函数对称，误以为它该出现在这里是最常见的手滑。
    """
    a = np.asarray(u, dtype=float)
    b = np.asarray(v, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"u 与 v 形状必须一致，得到 {a.shape} 与 {b.shape}")
    return float(np.sum(float(feed) * (1.0 - a) - a * b * b))


def pattern_metrics(u: Any, v: Any, dx: float = 1.0) -> dict[str, float]:
    r"""图案观测量（前端只需要这一个字典）：振幅、斑点、均值、主波长、极值。

    返回键：``amplitude``（:math:`\max u-\min u`）、``spot_count``、``mean_u``、
    ``mean_v``、``dominant_wavelength``、``total_u``、``total_v``、
    ``u_min``/``u_max``/``v_min``/``v_max``。
    """
    a = np.asarray(u, dtype=float)
    b = np.asarray(v, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"u 与 v 形状必须一致，得到 {a.shape} 与 {b.shape}")
    return {
        "amplitude": float(a.max() - a.min()),
        "spot_count": float(count_spots(b)),
        "mean_u": float(a.mean()),
        "mean_v": float(b.mean()),
        "dominant_wavelength": dominant_wavelength(a, dx),
        "total_u": float(a.sum()),
        "total_v": float(b.sum()),
        "u_min": float(a.min()),
        "u_max": float(a.max()),
        "v_min": float(b.min()),
        "v_max": float(b.max()),
    }
