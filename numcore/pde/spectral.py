r"""4.6 章数值内核：Fourier 谱方法、Chebyshev 谱方法与伪谱去混叠。

这一章要回答的问题只有一个：**「谱方法」这三个字到底承诺了什么，又没承诺什么**。
内核按四组对象组织，每组都配一条可实测的判据（而不是一句形容词）：

1. **Fourier 侧**（:func:`fourier_grid` … :func:`fourier_derivative`）——
   周期区间上的模态表示。对**三角多项式**（有限个模态）谱微分精确到舍入：
   实测误差是 :math:`c\,K^{p}\,u\,\|u\|`，不是「很小」；对带跳变的函数，
   跳变点附近的导数误差**不随 N 下降**（Gibbs 只被压窄、不被压平）。
2. **伪谱乘积与去混叠**（:func:`dealias_product` / :func:`product_naive` /
   :func:`alias_error`）——3/2 规则把乘积的谱精确算出来（到舍入），
   不去混叠则把 :math:`|k|>N/2` 的模态折回低波数，污染量由 :func:`alias_error` 量化。
   三者（朴素 / 去混叠 / 高分辨率参照）必须同时在场，否则「去混叠有效」没有对照物。
3. **谱 vs 有限差分**（:func:`advection_spectral` / :func:`advection_fd` /
   :func:`advection_error_curve`）——同一个线性平流方程、同一个初值、同样**精确的时间推进**
   （两边都只留空间误差），一条指数下降、一条代数下降。判据是「误差随 N 下降快于任何
   代数幂」：实测每加倍一次误差比本身还在变小，而有限差分的比值恒定（2 阶是 1/4）。
4. **Chebyshev 侧**（:func:`chebyshev_coeffs` … :func:`chebyshev_convergence`）——
   非周期区间上的谱表示。光滑解指数收敛，**C^0 尖角解退化为代数**：
   :func:`chebyshev_convergence` 直接给出误差-自由度曲线，尖角一侧的斜率是代数幂。

另配 :class:`BurgersResult` 与 :func:`burgers_solve`：一维周期 Burgers 的伪谱求解
（可选 3/2 去混叠），输出能谱、能流与**能量收支残差**——级联与去混叠的账要能被复算。

约定
----
* 波数顺序与 :func:`numpy.fft.fftfreq` 同一口径：``[0,1,...,n/2-1,-n/2,...,-1]``；
  变换用自研的 :func:`numcore.transform.fft.fft`（POLICY P2 禁止 ``numpy.fft``）；
* 系数归一化：``u_j = sum_k a_k exp(2*pi*i*k*x_j/L)``，故 ``a = fft(u)/n``；
* 一切函数都是纯函数：无 IO、无 ``print``、无全局可变状态（POLICY P6）；
* 对数坐标上的 0 由**调用方**给地板值，内核只保证返回有限数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from numcore.transform.fft import fft, ifft, next_power_of_two

__all__ = [
    "TWO_PI",
    # 网格 / 波数 / 变换 / 微分
    "fourier_grid",
    "fourier_wavenumbers",
    "fourier_coeffs",
    "fourier_synthesize",
    "fourier_derivative",
    "wavenumber_spectrum",
    "two_thirds_mask",
    # 测试剖面
    "poisson_profile",
    "poly_bump_profile",
    "square_wave_profile",
    "sawtooth_profile",
    "tent_profile",
    "fourier_profile",
    "synthetic_field",
    "fourier_truncation_error",
    "fourier_convergence",
    "profile_decay_exponent",
    "modal_decay_fit",
    "gibbs_overshoot",
    "cheb_profile",
    # 伪谱乘积 / 去混叠 / 滤波
    "extended_size",
    "pad_spectrum",
    "truncate_spectrum",
    "product_naive",
    "product_naive_hat",
    "product_reference",
    "product_reference_hat",
    "dealias_product",
    "dealias_product_hat",
    "alias_error",
    "nonlinear_energy_residual",
    "filter_spectrum",
    "spectral_filter",
    # 线性平流：谱 vs 有限差分
    "modified_wavenumber",
    "advection_exact",
    "advection_spectral",
    "advection_fd",
    "advection_error_curve",
    # 拟合工具
    "local_slopes",
    "doubling_ratios",
    "fit_power_law",
    # Chebyshev
    "chebyshev_grid",
    "chebyshev_grid_ascending",
    "chebyshev_coeffs",
    "chebyshev_eval",
    "chebyshev_differentiation_matrix",
    "chebyshev_derivative",
    "chebyshev_truncation_error",
    "chebyshev_convergence",
    # Burgers
    "BurgersResult",
    "burgers_solve",
    "energy_spectrum",
    "spectral_flux",
    "cascade_window",
]

#: 2*pi 的显式写法（避免各函数里各写一遍）
TWO_PI = 2.0 * float(np.pi)


# --------------------------------------------------------------------------- #
# 1. Fourier 侧：网格、波数、系数、谱微分
# --------------------------------------------------------------------------- #
def fourier_grid(n: int, length: float = 1.0, *, x0: float = 0.0) -> np.ndarray:
    r"""周期网格 :math:`x_j = x_0 + j L/n`（``j = 0..n-1``，**不含右端点**）。

    右端点与左端点重合是周期网格的定义，不是遗漏；把它画进去会让卷积与差分的
    下标全部错一位（2.5 章的卷绕也是同一件事）。
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n 必须 >= 1，得到 {n}")
    return x0 + float(length) * np.arange(n, dtype=float) / float(n)


def fourier_wavenumbers(n: int) -> np.ndarray:
    """标准 FFT 顺序的整数波数 ``[0,1,...,n/2-1,-n/2,...,-1]``（偶数 n）。

    这是 :func:`numcore.transform.fft.fft` 输出的第 k 个分量所对应的**有符号**波数；
    混叠、折叠、去混叠全部依赖它，所以它是内核里最该被对拍的一个数组。
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n 必须 >= 1，得到 {n}")
    return np.concatenate(
        [np.arange(0, (n + 1) // 2, dtype=float), np.arange(-(n // 2), 0, dtype=float)]
    )


def fourier_coeffs(u: Any, *, length: float = 1.0) -> np.ndarray:  # noqa: ARG001 - 语义参数
    r"""模态系数 :math:`a_k = \frac1n\sum_j u_j e^{-2\pi i jk/n}`（``fft(u)/n``）。

    ``length`` 只影响物理波数 :math:`2\pi k/L` 的解释，不影响这里的系数本身；
    保留它是为了让调用点的意图可读（并给 :func:`fourier_derivative` 共用签名）。
    """
    arr = np.asarray(u)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size < 1:
        raise ValueError("u 不能为空")
    return fft(arr) / float(arr.size)


def fourier_synthesize(a: Any) -> np.ndarray:
    r"""由系数还原网格值 :math:`u_j = \sum_k a_k e^{2\pi i jk/n}`（``ifft(a)*n``）。

    返回**复数**数组：实数据的虚部是舍入噪声（量级 :math:`u\|u\|`），
    取实部是调用方的责任——这样「哪个环节丢了虚部」不会被静默吞掉。
    """
    arr = np.asarray(a)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size < 1:
        raise ValueError("a 不能为空")
    return ifft(arr) * float(arr.size)


def fourier_derivative(u: Any, order: int = 1, *, length: float = 1.0) -> np.ndarray:
    r"""周期谱微分 :math:`u^{(p)}`：系数乘 :math:`(2\pi i k/L)^p` 再合成。

    对**三角多项式**（谱支撑在 :math:`|k|\le K`）这是精确的：误差只有舍入，
    量级 :math:`c\,(2\pi K/L)^p\,u\,\|u\|`——即误差本身随阶数 :math:`p` 与带宽 :math:`K`
    增长，不是「零」。对非光滑数据，跳变附近的误差**不随 N 消失**（见章内断言）。
    """
    p = int(order)
    if p < 0:
        raise ValueError(f"order 必须 >= 0，得到 {p}")
    arr = np.asarray(u)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if p == 0:
        return np.real(arr.astype(float)) if not np.iscomplexobj(arr) else arr
    a = fourier_coeffs(arr, length=length)
    k = fourier_wavenumbers(a.size)
    lam = 2j * np.pi * k / float(length)
    return np.real(fourier_synthesize(a * lam**p))


def wavenumber_spectrum(u: Any, *, length: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """返回按物理波数**升序**排好的 ``(k, |a_k|)``（双边谱，实数据关于 0 对称）。"""
    a = fourier_coeffs(u, length=length)
    k = fourier_wavenumbers(a.size) * (TWO_PI / float(length))
    order = np.argsort(k)
    return k[order], np.abs(a)[order]


def two_thirds_mask(n: int) -> np.ndarray:
    r"""2/3 规则的保留掩码：``|k| <= n/3`` 为真（标准 FFT 顺序）。

    3/2 规则与 2/3 规则是同一件事的两面：把谱补零到 :math:`3N/2` 算乘积，
    等价于**先**把 :math:`|k|>N/3` 的模态丢掉（否则乘积的带宽 :math:`2K` 会超过
    :math:`(3N/2)/2`，补零也救不回来）。
    """
    n = int(n)
    k = np.abs(fourier_wavenumbers(n))
    return k <= (float(n) / 3.0)


# --------------------------------------------------------------------------- #
# 2. 测试剖面（解析真值，用于模态衰减与收敛率）
# --------------------------------------------------------------------------- #
def poisson_profile(x: Any, *, length: float = 1.0, rho: float = 0.6, x0: float = 0.0) -> np.ndarray:
    r"""Poisson 核 :math:`\frac{1-\rho^2}{1-2\rho\cos\theta+\rho^2}`，:math:`\theta=2\pi(x-x_0)/L`。

    它的 Fourier 系数**恰好**是 :math:`a_k=\rho^{|k|}`（含 :math:`a_0=1`）：
    这是本章唯一一个「解析系数已知」的光滑剖面，所以「指数衰减」不需要拟合就能对拍。
    :math:`\rho<1` 保证解析（带宽无限、系数按几何级数下降）。
    """
    rho = float(rho)
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho 必须在 (0,1)，得到 {rho}")
    th = TWO_PI * (np.asarray(x, dtype=float) - float(x0)) / float(length)
    return (1.0 - rho * rho) / (1.0 - 2.0 * rho * np.cos(th) + rho * rho)


def poly_bump_profile(x: Any, *, length: float = 1.0, order: int = 2, x0: float = 0.0) -> np.ndarray:
    r"""多项式凸包 :math:`(1-(2s)^2)^{\text{order}}`（:math:`|s|\le1/2`，否则 0），
    :math:`s` 是 :math:`(x-x_0)/L` 折回 :math:`[-1/2,1/2]` 的坐标。

    在端点处只有 :math:`\text{order}-1` 阶导数连续，第 :math:`\text{order}` 阶导数跳变，
    于是 :math:`a_k\sim k^{-(\text{order}+1)}`——
    这是「有间断（或导数间断）时代数衰减」的可调旋钮。
    """
    p = int(order)
    if p < 1:
        raise ValueError(f"order 必须 >= 1，得到 {p}")
    s = (np.asarray(x, dtype=float) - float(x0)) / float(length)
    s = s - np.floor(s + 0.5)  # 折回 [-1/2, 1/2)
    base = np.maximum(1.0 - (2.0 * s) ** 2, 0.0)
    return base**p


def square_wave_profile(x: Any, *, length: float = 1.0, x0: float = 0.0) -> np.ndarray:
    r"""周期方波：:math:`+1` 在 :math:`(x_0, x_0+L/2)`，否则 :math:`-1`。

    函数本身跳变 ⇒ :math:`a_k\sim 1/k`，且 N 增大时 Gibbs 振荡的**幅度**不变、
    只被压窄——「系数衰减代数化」和「振荡不消失」是同一件事的两个说法。

    **采样陷阱**：跳变正好落在节点上（:math:`x_0` 是格点）时，偶次谐波不是 0
    而是常数 :math:`1/N`（实测 N=64 时每个偶次谐波都等于 0.03125），
    于是双对数拟合会被这排常数点拖成 :math:`p=-0.848,\ R^2=0.346`。
    要测「跳变 ⇒ :math:`k^{-1}`」得用 :func:`sawtooth_profile`（非零模态铺满）。
    """
    s = (np.asarray(x, dtype=float) - float(x0)) / float(length)
    s = s - np.floor(s)
    return np.where((s > 0.0) & (s < 0.5), 1.0, -1.0)


def tent_profile(x: Any, *, length: float = 1.0, x0: float = 0.0) -> np.ndarray:
    r"""周期尖顶 :math:`1-2|s|`（:math:`|s|\le1/2`）：C^0 尖角 ⇒ :math:`a_k\sim k^{-2}`。

    它和 :func:`sawtooth_profile` 的区别正好是「折角」与「跳变」的区别，
    两条衰减曲线在双对数图上的斜率差 1；而它的导数谱才是有界的
    （方波的导数谱在跳变附近按 :math:`N` 增长，见 :func:`gibbs_overshoot` 的说明）。
    """
    s = (np.asarray(x, dtype=float) - float(x0)) / float(length)
    s = s - np.floor(s + 0.5)
    return np.maximum(1.0 - 2.0 * np.abs(s), 0.0)


def sawtooth_profile(x: Any, *, length: float = 1.0, x0: float = 0.0) -> np.ndarray:
    r"""周期锯齿 :math:`1-2s`（:math:`s\in[0,1)`）：**处处**只有一次跳变 ⇒ 全部模态
    :math:`a_k\sim 1/k`（不邻零）。方波的偶次谐波恒为 0，「拟合斜率」会被那些零点
    带偏（实测 :math:`R^2=0.048`）——两者要分开：衰减指数用锯齿测，Gibbs 过冲用方波测。
    """
    s = (np.asarray(x, dtype=float) - float(x0)) / float(length)
    s = s - np.floor(s)
    return 1.0 - 2.0 * s


def fourier_profile(kind: str, x: Any, *, length: float = 1.0, rho: float = 0.6, order: int = 2) -> np.ndarray:
    """周期剖面分发：``analytic``（Poisson）/ ``corner``（尖顶）/ ``jump``（锯齿）/ ``square`` / ``bump``。"""
    key = str(kind)
    if key == "analytic":
        return poisson_profile(x, length=length, rho=rho)
    if key == "corner":
        return tent_profile(x, length=length)
    if key == "jump":
        return sawtooth_profile(x, length=length)
    if key == "square":
        return square_wave_profile(x, length=length)
    if key == "bump":
        return poly_bump_profile(x, length=length, order=order)
    raise KeyError(f"未知剖面 {kind!r}（可用：analytic/corner/jump/square/bump）")


def synthetic_field(
    n: int,
    decay: float,
    *,
    seed: int = 20261018,
    length: float = 1.0,
) -> np.ndarray:
    r"""按指定谱指数造一个**实**的确定性测试场：:math:`|a_k|=|k|^{-\text{decay}}`（k≠0），相位由 ``seed`` 定。

    为什么要随机相位而不是固定的二次相位（:math:`e^{i\alpha k^2}`）：混叠误差的大小
    **强烈依赖相位结构**。实测同一张 :math:`k^{-1}` 谱，二次相位下能量恒等式残差
    :math:`1.2\times10^{-17}`、线性相位下 :math:`2.0\times10^{-1}`——
    只测一组"好看"的相位，就会把「混叠不动能量账」写成结论。
    相位由 :func:`numcore.utils.rng` 生成（POLICY P4），同一 ``seed`` 逐位可复现。
    """
    from numcore.utils import rng

    n = int(n)
    decay = float(decay)
    k = fourier_wavenumbers(n)
    gen = rng(int(seed))
    amp = np.where(np.abs(k) >= 1.0, np.abs(np.where(np.abs(k) >= 1.0, k, 1.0)) ** (-decay), 0.0)
    phase = gen.uniform(0.0, TWO_PI, size=n)
    a = amp * np.exp(1j * phase)
    u = np.real(fourier_synthesize(a))
    scale = float(np.max(np.abs(u)))
    return u / scale if scale > 0.0 else u


def fourier_truncation_error(
    kind: str,
    n: int,
    *,
    rho: float = 0.6,
    order: int = 2,
    n_eval: int | None = None,
    length: float = 1.0,
) -> float:
    r"""Fourier **插值/截断**误差：``N`` 点采样 -> DFT -> 前 ``N/2`` 个模态重建到稠密网格。

    节点上误差恒为 0（插值性质），所以必须在**节点之间**量——与
    :func:`chebyshev_truncation_error` 的口径一致。这就是「模态衰减」与
    「逼近误差」之间那条被 :math:`a_k` 的图掩盖的桥：
    :math:`\|f-S_Nf\|_\infty` 由**尾巴**决定，光滑时指数小、间断时只按代数小。
    """
    n = int(n)
    x_nodes = fourier_grid(n, length=length)
    values = fourier_profile(kind, x_nodes, length=length, rho=rho, order=order)
    a = fourier_coeffs(values)
    if n_eval is None:
        n_eval = max(8 * n + 1, 257)
    x_eval = np.linspace(0.0, float(length), int(n_eval), endpoint=False)
    half = n // 2
    ksel = np.concatenate([np.arange(0, half + 1, dtype=float), np.arange(-half + 1, 0, dtype=float)])
    idx = np.concatenate([np.arange(0, half + 1), np.arange(n - half + 1, n)])
    with np.errstate(over="ignore", invalid="ignore"):
        basis = np.exp(2j * np.pi * np.outer(x_eval, ksel) / float(length))
    approx = np.real(basis @ a[idx])
    exact = fourier_profile(kind, x_eval, length=length, rho=rho, order=order)
    return float(np.max(np.abs(approx - exact)))


def fourier_convergence(
    kind: str,
    ns: Any,
    *,
    rho: float = 0.6,
    order: int = 2,
    n_eval: int | None = None,
    length: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """误差-自由度曲线 ``(ns, errs)``（Fourier 版，口径同 :func:`chebyshev_convergence`）。"""
    n_arr = np.asarray([int(v) for v in ns], dtype=float)
    errs = np.asarray(
        [
            fourier_truncation_error(kind, int(v), rho=rho, order=int(order), n_eval=n_eval, length=length)
            for v in n_arr
        ],
        dtype=float,
    )
    return n_arr, errs


def profile_decay_exponent(kind: str, *, order: int = 2) -> float | None:
    r"""理论衰减指数 :math:`p`（:math:`a_k\sim k^{-p}`）；解析剖面返回 ``None``。

    ``None`` 的含义是「不存在有限的代数指数」——指数衰减比任何 :math:`k^{-p}` 都快，
    这正是本章要用实测区分的东西。``square`` 给的是**包络**指数（奇次谐波）。
    """
    key = str(kind)
    if key in ("jump", "square"):
        return 1.0
    if key == "corner":
        return 2.0
    if key == "bump":
        return float(int(order) + 1)
    if key == "analytic":
        return None
    raise KeyError(f"未知剖面 {kind!r}")


def modal_decay_fit(k: Any, amp: Any, lo: float, hi: float, *, min_rel: float = 1e-12) -> dict[str, Any]:
    r"""模态衰减曲线的窗口拟合（只在**非零**模态上做双对数最小二乘）。

    为什么不能把整段系数直接丢进最小二乘：采样方波的偶次谐波是常数 :math:`1/N`
    （Chebyshev 侧 Runge 函数的奇次系数才是**恰好** 0），
    那些点在对数轴上要么被丢掉、要么把 :math:`R^2` 拉到 0.35（实测），
    于是「斜率 = -1」这条结论会被数据本身的结构掩盖。这里按 ``min_rel``
    （相对最大系数）先筛掉零模态，再把窗口与用到的点数一起报出来。
    """
    kv = np.asarray(k, dtype=float).reshape(-1)
    av = np.abs(np.asarray(amp, dtype=float).reshape(-1))
    if kv.size != av.size:
        raise ValueError("k 与 amp 必须等长")
    scale = float(np.max(av)) if av.size else 0.0
    thr = max(float(min_rel) * scale, 0.0)
    sel = (kv >= float(lo)) & (kv <= float(hi)) & (av > thr)
    fit = fit_power_law(kv[sel], av[sel])
    fit["n_zero_dropped"] = int(np.count_nonzero((kv >= float(lo)) & (kv <= float(hi)) & (av <= thr)))
    return fit


def gibbs_overshoot(n_modes: int, *, jump: float = 2.0) -> float:
    r"""方波 Fourier 部分和（截到 :math:`K` 个奇次谐波）的相对过冲。

    .. math::

        S_K(x)=\frac{4}{\pi}\sum_{m=0}^{K}\frac{\sin((2m+1)x)}{2m+1},
        \qquad \max_x S_K \to \frac{2}{\pi}\mathrm{Si}(\pi)=1.1789797\ldots

    过冲（相对跳变 :math:`2`）收敛到 :math:`0.0894898\ldots\times2`，
    **不随 K 消失**：K 增大只是把振荡压窄（宽度 :math:`\sim 1/K`）。
    这正是「间断解一侧是代数衰减 + Gibbs」的可执行版本。
    """
    k = int(n_modes)
    if k < 1:
        raise ValueError(f"n_modes 必须 >= 1，得到 {k}")
    m = np.arange(k + 1, dtype=float)
    n_scan = max(4001, 40 * k + 1)
    xs = np.pi * (np.arange(n_scan, dtype=float) + 0.5) / float(n_scan)
    s = np.zeros_like(xs)
    for mi in m:
        s += np.sin((2.0 * mi + 1.0) * xs) / (2.0 * mi + 1.0)
    return float(np.max((4.0 / np.pi) * s) / (0.5 * float(jump)) - 1.0)


def cheb_profile(kind: str, x: Any) -> np.ndarray:
    r"""Chebyshev 侧的测试函数（区间 :math:`[-1,1]`）。

    * ``runge`` —— :math:`1/(1+25x^2)`：解析但极点离区间很近（Bernstein 椭圆参数
      :math:`\rho=3+2\sqrt2\approx5.83`），系数按几何级数下降 ⇒ **指数收敛**；
    * ``exp`` —— :math:`e^{x}`：整函数，收敛更快；
    * ``abs`` —— :math:`|x|`：C^0 尖角 ⇒ 代数收敛；
    * ``abs3`` —— :math:`|x|^3`：C^2 尖角 ⇒ 更慢的代数收敛（指数 |a_k| ~ k^{-4}）；
    * ``sign`` —— :math:`\mathrm{sign}(x)`：跳变 ⇒ :math:`|a_k|\sim k^{-1}`。
    """
    key = str(kind)
    xs = np.asarray(x, dtype=float)
    if key == "runge":
        return 1.0 / (1.0 + 25.0 * xs * xs)
    if key == "exp":
        return np.exp(xs)
    if key == "abs":
        return np.abs(xs)
    if key == "abs3":
        return np.abs(xs) ** 3
    if key == "sign":
        return np.sign(xs)
    raise KeyError(f"未知 Chebyshev 剖面 {kind!r}")


# --------------------------------------------------------------------------- #
# 3. 伪谱乘积 / 3-2 规则 / 滤波
# --------------------------------------------------------------------------- #
def extended_size(n: int, factor: float = 1.5, *, exact: bool = False) -> int:
    r"""把网格从 :math:`N` 扩到不小于 :math:`\text{factor}\cdot N` 的**2 的幂**。

    为什么不是刚好 :math:`3N/2`：数学上只要 :math:`M\ge4N/3`（保证乘积带宽
    :math:`2K\le M/2`）就够，而自研 FFT 只有 2 的幂走 radix-2 快路径，
    非 2 的幂会掉进 Bluestein——实测 192 点 5.8 ms vs 256 点 0.36 ms（16 倍），
    一帧 Burgers 要几千次变换，这个差别是「能不能出帧」的差别。
    :math:`M\ge3N/2` 仍然被满足，所以「3/2 规则」的名字与结论都不变。

    ``exact=True`` 时返回不小于 ``factor*n`` 的最小偶数（教科书里那个精确的
    :math:`3N/2`，例如 N=64 给 96）——它慢（非 2 的幂走 Bluestein），
    但断言里要用它把「3/2 规则必须配 2/3 截断」这句话量出来。
    """
    n = int(n)
    factor = float(factor)
    if factor <= 1.0:
        raise ValueError(f"factor 必须 > 1，得到 {factor}")
    m = int(np.ceil(factor * n))
    if exact:
        return m + (m % 2)
    return int(next_power_of_two(m))


def pad_spectrum(a: Any, m: int) -> np.ndarray:
    r"""把 ``n`` 点谱零填充到 ``m`` 点谱（保持标准 FFT 顺序与系数归一化）。

    偶数 ``n`` 的 Nyquist 系数 :math:`a_{N/2}` 在物理上同时代表 :math:`\pm N/2`
    两个模态，所以**对半分**到两端；直接塞给一端会引入 :math:`O(a_{N/2})` 的误差，
    在去混叠的对拍里就是「误差怎么都下不到 1e-16」的元凶。
    """
    arr = np.asarray(a)
    n = int(arr.size)
    m = int(m)
    if m < n:
        raise ValueError(f"扩展长度 m={m} 必须 >= 原长度 n={n}")
    out = np.zeros(m, dtype=complex)
    k = fourier_wavenumbers(n)
    out[np.mod(k.astype(np.int64), m)] = arr.astype(complex)
    if n % 2 == 0 and n >= 2:
        half = 0.5 * arr[n // 2]
        out[n // 2] = half
        out[(m - n // 2) % m] = half
    return out


def truncate_spectrum(A: Any, n: int) -> np.ndarray:
    r"""把 ``m`` 点谱截回 ``n`` 点谱（:func:`pad_spectrum` 的逆：Nyquist 两端**相加**）。"""
    arr = np.asarray(A)
    m = int(arr.size)
    n = int(n)
    if n > m:
        raise ValueError(f"截断长度 n={n} 必须 <= 原长度 m={m}")
    k = fourier_wavenumbers(n)
    out = arr[np.mod(k.astype(np.int64), m)].astype(complex)
    if n % 2 == 0 and n >= 2:
        out[n // 2] = arr[n // 2] + arr[(m - n // 2) % m]
    return out


def product_naive(u: Any, v: Any, *, length: float = 1.0) -> np.ndarray:  # noqa: ARG001
    r"""朴素伪谱乘积：直接在 ``N`` 点上相乘再变换（**有混叠**）。

    等价于把乘积的谱按 :math:`k \bmod N` 折叠：:math:`|k|>N/2` 的模态会**折回**低波数，
    低波数上的读数因此被污染，而图上看不出任何异常。
    """
    a = np.asarray(u, dtype=float).reshape(-1)
    b = np.asarray(v, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"u 与 v 长度必须相同，得到 {a.size} 与 {b.size}")
    return fft(a * b) / float(a.size)


def product_reference(
    u: Any,
    v: Any,
    *,
    length: float = 1.0,
    factor: float = 6.0,
    exact: bool = False,
) -> np.ndarray:
    r"""高分辨率参照乘积（:math:`M\ge6N`）：对带宽 :math:`\le N/3` 的输入是**精确**的。

    它是本章的「真值」来源。用 3/2 规则的输出当参照是不行的——两者共享同一套
    补零/截断下标约定，同错同对（2.5 章蝶形残差踩过同一个坑）。
    """
    a = np.asarray(u, dtype=float).reshape(-1)
    b = np.asarray(v, dtype=float).reshape(-1)
    n = a.size
    if b.size != n:
        raise ValueError(f"u 与 v 长度必须相同，得到 {n} 与 {b.size}")
    m = extended_size(n, factor, exact=exact)
    A = pad_spectrum(fourier_coeffs(a, length=length), m)
    B = pad_spectrum(fourier_coeffs(b, length=length), m)
    w = np.real(fourier_synthesize(A)) * np.real(fourier_synthesize(B))
    return truncate_spectrum(fft(w) / float(m), n)


def dealias_product(
    u: Any,
    v: Any,
    *,
    length: float = 1.0,
    factor: float = 1.5,
    keep_two_thirds: bool = True,
    exact: bool = False,
) -> np.ndarray:
    r"""3/2 规则去混叠乘积的**系数**（标准 FFT 顺序，长度 ``N``）。

    步骤：系数补零到 :math:`M=3N/2` → 在 :math:`M` 网格上逐点相乘 → 变换回来 → 截回 ``N``。
    ``keep_two_thirds=True`` 时先把 :math:`|k|>N/3` 的模态清零（2/3 规则）：
    只有这样才能保证积的带宽 :math:`2K\le M/2=3N/4`，否则折回依然存在——
    **3/2 规则与 2/3 规则不是两个可以随便挑的选项，是同一条不等式的两半**。
    """
    a = np.asarray(u, dtype=float).reshape(-1)
    b = np.asarray(v, dtype=float).reshape(-1)
    n = a.size
    if b.size != n:
        raise ValueError(f"u 与 v 长度必须相同，得到 {n} 与 {b.size}")
    A = fourier_coeffs(a, length=length)
    B = fourier_coeffs(b, length=length)
    if keep_two_thirds:
        mask = two_thirds_mask(n)
        A = np.where(mask, A, 0.0)
        B = np.where(mask, B, 0.0)
    m = extended_size(n, factor, exact=exact)
    Am = pad_spectrum(A, m)
    Bm = pad_spectrum(B, m)
    w = np.real(fourier_synthesize(Am)) * np.real(fourier_synthesize(Bm))
    return truncate_spectrum(fft(w) / float(m), n)


def product_naive_hat(a: Any, b: Any, *, length: float = 1.0) -> np.ndarray:  # noqa: ARG001
    r"""系数空间版的朴素伪谱乘积：进/出都是系数，省掉一次正变换。

    ``w = fft(ifft(a)*ifft(b))/n``——与 :func:`product_naive` 逐位相同，
    但调用方（时间推进）手里本来就只有系数，少一次 n 点正变换。
    """
    av = np.asarray(a, dtype=complex).reshape(-1)
    bv = np.asarray(b, dtype=complex).reshape(-1)
    if av.size != bv.size:
        raise ValueError("a 与 b 长度必须相同")
    n = av.size
    u = np.real(fourier_synthesize(av))
    v = np.real(fourier_synthesize(bv))
    return fft(u * v) / float(n)


def dealias_product_hat(
    a: Any,
    b: Any,
    *,
    length: float = 1.0,  # noqa: ARG001 - 与网格版保持同一签名
    factor: float = 1.5,
    keep_two_thirds: bool = True,
    exact: bool = False,
) -> np.ndarray:
    r"""系数空间版的 3/2 规则乘积（与 :func:`dealias_product` 同一个数学对象）。

    时间推进里每个 RK 级都要算一次非线性项，网格版要多一次 :math:`N` 点正变换；
    系数版直接补零到 :math:`M` 再逆变换，实测每一步省 1/3 的时间。
    """
    av = np.asarray(a, dtype=complex).reshape(-1)
    bv = np.asarray(b, dtype=complex).reshape(-1)
    if av.size != bv.size:
        raise ValueError("a 与 b 长度必须相同")
    n = av.size
    if keep_two_thirds:
        mask = two_thirds_mask(n)
        av = np.where(mask, av, 0.0)
        bv = np.where(mask, bv, 0.0)
    m = extended_size(n, factor, exact=exact)
    Am = pad_spectrum(av, m)
    Bm = pad_spectrum(bv, m)
    w = np.real(fourier_synthesize(Am)) * np.real(fourier_synthesize(Bm))
    return truncate_spectrum(fft(w) / float(m), n)


def product_reference_hat(a: Any, b: Any, *, factor: float = 6.0) -> np.ndarray:
    """系数空间版的高分辨率参照乘积（:func:`product_reference` 的系数入口）。"""
    m = extended_size(np.asarray(a).size, factor)
    Am = pad_spectrum(a, m)
    Bm = pad_spectrum(b, m)
    w = np.real(fourier_synthesize(Am)) * np.real(fourier_synthesize(Bm))
    return truncate_spectrum(fft(w) / float(m), np.asarray(a).size)


def alias_error(
    u: Any,
    v: Any,
    *,
    length: float = 1.0,
    factor: float = 6.0,
) -> dict[str, Any]:
    r"""3/2 规则的**四项账**：混叠、截断、补零、以及两条路线的总差。

    四条参照物必须分开（混成一条就说不清 3/2 规则到底做了什么），
    记 :math:`S_f=\max_k|W^{\text{ref,full}}_k|`、:math:`S_t=\max_k|W^{\text{ref,trunc}}_k|`：

    ``alias_rel`` / ``alias_low_rel``
        朴素乘积（不补零、不截断）与**全带**参照的偏差：全带 / 保留带
        :math:`|k|\le N/3`。这才是「混叠污染了多少」，它落在**所有**波数上；
    ``truncation_rel``
        2/3 截断参照与全带参照的偏差（:math:`/S_f`）：2/3 规则**主动丢掉**的东西；
    ``dealias_rel``
        ``2/3+3/2`` 与截断参照的偏差（:math:`/S_t`）：去混叠对**截断后的场**是精确的；
    ``pad_only_rel``
        **只补零不截断**（精确 :math:`M=3N/2`）与全带参照的偏差（:math:`/S_f`）。
        数学上保留带 :math:`|k|<N/2` 内的折返来自乘积模 :math:`>M/2`，
        最小落点是 :math:`M-N\ge N/2`——所以只补零**已经**把混叠压到 :math:`|k|=N/2`
        那一条边界模态上（实测 6e-7，朴素的 1/1000），
    ``naive_vs_dealias_rel``
        朴素路线与 ``2/3+3/2`` 路线的**总差**（:math:`/S_t`）：使用者真正在选的两种算法之差，
        里面同时含混叠与截断两项，所以它**不能**被单独读成"混叠误差"；
    ``folded_energy_fraction``
        全带乘积谱里落在 :math:`|k|>N/2` 的能量占比——就是会折回低波数的那部分；
    ``truncated_input_energy_fraction``
        输入谱里 :math:`|k|>N/3` 的能量占比（2/3 规则要从输入里删掉的部分）；
    ``spectrum`` / ``k``
        :math:`|W^{\text{naive}}_k-W^{\text{ref,full}}_k|`（标准 FFT 顺序）与有符号波数，
        供帧直接画"混叠落在哪"。
    """
    a = np.asarray(u, dtype=float).reshape(-1)
    b = np.asarray(v, dtype=float).reshape(-1)
    n = a.size
    if b.size != n:
        raise ValueError(f"u 与 v 长度必须相同，得到 {n} 与 {b.size}")
    A = fourier_coeffs(a, length=length)
    B = fourier_coeffs(b, length=length)
    mask = two_thirds_mask(n)
    a_t = np.real(fourier_synthesize(np.where(mask, A, 0.0)))
    b_t = np.real(fourier_synthesize(np.where(mask, B, 0.0)))

    ref_full = product_reference(a, b, length=length, factor=factor)
    ref_trunc = product_reference(a_t, b_t, length=length, factor=factor)
    naive = product_naive(a, b, length=length)
    deal = dealias_product(a, b, length=length)
    pad_only = dealias_product(a, b, length=length, keep_two_thirds=False, exact=True)

    s_full = float(np.max(np.abs(ref_full))) or 1.0
    s_trunc = float(np.max(np.abs(ref_trunc))) or 1.0
    k = fourier_wavenumbers(n)
    k_in = np.abs(k) <= (float(n) / 3.0)
    d_alias = np.abs(naive - ref_full)

    # 折叠能量：把**未截断**的乘积谱搬到 3N 网格上读，|k| > N/2 的部分就是折回源
    m3 = extended_size(n, 3.0)
    A3 = pad_spectrum(A, m3)
    B3 = pad_spectrum(B, m3)
    w3 = np.real(fourier_synthesize(A3)) * np.real(fourier_synthesize(B3))
    W3 = fft(w3) / float(m3)
    k3 = np.abs(fourier_wavenumbers(m3))
    e_tot = float(np.sum(np.abs(W3) ** 2))
    e_high = float(np.sum(np.abs(W3[k3 > float(n) / 2.0]) ** 2))
    e_in = float(np.sum(np.abs(A) ** 2))
    e_in_high = float(np.sum(np.abs(A[~mask]) ** 2))
    return {
        "alias_rel": float(np.max(d_alias)) / s_full,
        "alias_low_rel": float(np.max(d_alias[k_in])) / s_full,
        "truncation_rel": float(np.max(np.abs(ref_trunc - ref_full))) / s_full,
        "dealias_rel": float(np.max(np.abs(deal - ref_trunc))) / s_trunc,
        "pad_only_rel": float(np.max(np.abs(pad_only - ref_full))) / s_full,
        "naive_vs_dealias_rel": float(np.max(np.abs(naive - deal))) / s_trunc,
        "folded_energy_fraction": (e_high / e_tot) if e_tot > 0.0 else 0.0,
        "truncated_input_energy_fraction": (e_in_high / e_in) if e_in > 0.0 else 0.0,
        "spectrum": d_alias,
        "k": k,
    }


def nonlinear_energy_residual(
    a: Any,
    w: Any,
    *,
    length: float = 1.0,
    band: int | None = None,
) -> float:
    r"""Galerkine 能量恒等式的残差（相对量）。

    对 :math:`u_t = -\frac12 (u^2)_x`，能量 :math:`E=\sum_k|a_k|^2` 的变化率是

    .. math::

        \dot E = \sum_k \kappa_k\,\mathrm{Im}\big(\overline{a_k}\,w_k\big),
        \qquad \kappa_k=\frac{2\pi k}{L},\quad w=\mathcal F(u^2).

    只要 :math:`w` 是**精确**的卷积、且求和范围与 :math:`u` 的支撑一致，
    右端按三波对称化 :math:`k+p+q=0` 恒等于 0。混叠破坏的正是这个对称性，
    于是「残差」把「去混叠到底改善了哪条守恒律」变成一个可以直接读的数。
    ``band`` 限制求和范围（传 ``N//3`` 即 2/3 规则的保留带）：
    **求和范围必须与 :math:`u` 的支撑一致**，否则三波对称化不完整，
    残差里会混进「本来就没在带内」的模态（这不是混叠，是记账错误）。
    归一化用 Cauchy--Schwarz 尺度 :math:`\sum_k|\kappa_k||a_k||w_k|`。
    """
    av = np.asarray(a, dtype=complex).reshape(-1)
    wv = np.asarray(w, dtype=complex).reshape(-1)
    if av.size != wv.size:
        raise ValueError(f"a 与 w 长度必须相同，得到 {av.size} 与 {wv.size}")
    kappa = TWO_PI * fourier_wavenumbers(av.size) / float(length)
    sel = np.ones(av.size, dtype=bool)
    if band is not None:
        sel = np.abs(fourier_wavenumbers(av.size)) <= float(band)
    resid = float(np.sum((kappa * np.imag(np.conj(av) * wv))[sel]))
    scale = float(np.sum((np.abs(kappa) * np.abs(av) * np.abs(wv))[sel]))
    if scale <= 0.0:
        return 0.0
    return abs(resid) / scale


def filter_spectrum(
    n: int,
    *,
    strength: float = 8.0,
    order: int = 8,
    cutoff: float = 2.0 / 3.0,
) -> np.ndarray:
    r"""指数滤波器 :math:`\sigma(k)=\exp(-\alpha(|k|/k_c)^{p})`，:math:`k_c=\text{cutoff}\cdot N/2`。

    **滤波不是去混叠**：去混叠把非线性项算对（不动解本身），滤波则**主动**把高波数
    的系数压小——它换来光滑，代价是引入与 :math:`\alpha` 同阶的耗散误差。
    两者的共同点是都能让「高频不听话」的现象消失，所以必须分开报数。
    """
    n = int(n)
    k = np.abs(fourier_wavenumbers(n))
    kc = max(float(cutoff) * float(n) / 2.0, 1.0)
    return np.exp(-float(strength) * (k / kc) ** int(order))


def spectral_filter(
    u: Any,
    *,
    strength: float = 8.0,
    order: int = 8,
    cutoff: float = 2.0 / 3.0,
    length: float = 1.0,
) -> np.ndarray:
    """对网格值施加指数滤波，返回滤波后的**网格值**（实部）。"""
    arr = np.asarray(u, dtype=float).reshape(-1)
    a = fourier_coeffs(arr, length=length)
    return np.real(fourier_synthesize(a * filter_spectrum(arr.size, strength=strength, order=order, cutoff=cutoff)))


# --------------------------------------------------------------------------- #
# 4. 线性平流：谱方法 vs 有限差分（空间误差隔离）
# --------------------------------------------------------------------------- #
def modified_wavenumber(kappa: Any, dx: float, *, order: int = 2) -> np.ndarray:
    r"""中心差分的**修正波数** :math:`\tilde\kappa(\kappa)`。

    * 2 阶：:math:`\tilde\kappa=\sin(\kappa\Delta x)/\Delta x`；
    * 4 阶：:math:`\tilde\kappa=(8\sin(\kappa\Delta x)-\sin(2\kappa\Delta x))/(6\Delta x)`。

    它与 :math:`\kappa` 的差就是半离散格式的**色散误差**：
    :math:`\tilde\kappa-\kappa=-\kappa^3\Delta x^2/6+O(\Delta x^4)`（2 阶）
    和 :math:`+\kappa^5\Delta x^4/30`（4 阶）——这就是「代数收敛」的来源。
    """
    kap = np.asarray(kappa, dtype=float)
    dx = float(dx)
    p = int(order)
    if p == 2:
        return np.sin(kap * dx) / dx
    if p == 4:
        return (8.0 * np.sin(kap * dx) - np.sin(2.0 * kap * dx)) / (6.0 * dx)
    raise ValueError(f"只实现了 2 阶与 4 阶，得到 order={order}")


def advection_exact(
    x: Any,
    t: float,
    *,
    c: float = 1.0,
    length: float = 1.0,
    rho: float = 0.6,
) -> np.ndarray:
    r"""周期平流 :math:`u_t+c u_x=0` 的精确解 :math:`u_0(x-ct)`（初值取 Poisson 剖面）。"""
    return poisson_profile(np.asarray(x, dtype=float) - float(c) * float(t), length=length, rho=rho)


def advection_spectral(
    n: int,
    t: float,
    *,
    c: float = 1.0,
    length: float = 1.0,
    rho: float = 0.6,
    u0: Any | None = None,
) -> np.ndarray:
    r"""Fourier 谱半离散解：:math:`\hat u_k(t)=e^{-i c\kappa_k t}\hat u_k(0)`。

    常系数线性问题的谱半离散系统是**对角**的，这个指数就是它的精确解
    （不是时间推进格式）——所以测到的误差里一点时间误差都没有，
    全部是初值在网格上的**频带截断**。它也解释了「谱误差与 t 无关」这一条：
    相位是精确的，误差只是 :math:`|k|>N/2` 的尾巴。
    """
    x = fourier_grid(n, length=length)
    u0v = advection_exact(x, 0.0, c=c, length=length, rho=rho) if u0 is None else np.asarray(u0, dtype=float)
    a = fourier_coeffs(u0v, length=length)
    kappa = TWO_PI * fourier_wavenumbers(int(n)) / float(length)
    return np.real(fourier_synthesize(a * np.exp(-1j * float(c) * kappa * float(t))))


def advection_fd(
    n: int,
    t: float,
    *,
    c: float = 1.0,
    length: float = 1.0,
    rho: float = 0.6,
    order: int = 2,
    u0: Any | None = None,
) -> np.ndarray:
    r"""中心差分半离散解（同样**精确时间推进**，用修正波数当符号）。

    :math:`\hat u_k(t)=e^{-ic\tilde\kappa_k t}\hat u_k(0)`：把时间误差拿掉以后，
    剩下的就是格式的色散误差；否则 2 阶时间格式会把 4 阶空间的斜率糊成 2 阶
    （「量错了对象」的典型）。
    """
    n = int(n)
    x = fourier_grid(n, length=length)
    u0v = advection_exact(x, 0.0, c=c, length=length, rho=rho) if u0 is None else np.asarray(u0, dtype=float)
    a = fourier_coeffs(u0v, length=length)
    dx = float(length) / float(n)
    kappa = TWO_PI * fourier_wavenumbers(n) / float(length)
    kt = modified_wavenumber(kappa, dx, order=order)
    return np.real(fourier_synthesize(a * np.exp(-1j * float(c) * kt * float(t))))


def advection_error_curve(
    ns: Sequence[int],
    t: float = 1.0,
    *,
    c: float = 1.0,
    length: float = 1.0,
    rho: float = 0.6,
    orders: Sequence[int] = (2, 4),
) -> dict[str, Any]:
    r"""误差-自由度曲线：同一 PDE、同一初值、同样精确的时间推进。

    返回 ``n``（数组）、``spectral``、``fd{order}`` 与 ``exact_scale``；
    误差是**相对** :math:`\ell_\infty`：:math:`\max_j|u^{num}_j-u^{exact}_j|/\max_j|u^{exact}_j|`。
    分辨率的「自由度」对谱方法是模态数 :math:`N`，对差分是格点数 :math:`N`——
    两边同一个横轴，比较才不会偷换概念。
    """
    n_arr = np.asarray([int(v) for v in ns], dtype=float)
    spec = np.zeros(n_arr.size)
    fd: dict[int, np.ndarray] = {int(p): np.zeros(n_arr.size) for p in orders}
    for i, n in enumerate(n_arr.astype(int)):
        x = fourier_grid(n, length=length)
        ref = advection_exact(x, t, c=c, length=length, rho=rho)
        scale = max(float(np.max(np.abs(ref))), 1e-300)
        spec[i] = float(np.max(np.abs(advection_spectral(n, t, c=c, length=length, rho=rho) - ref))) / scale
        for p in fd:
            fd[p][i] = float(np.max(np.abs(advection_fd(n, t, c=c, length=length, rho=rho, order=p) - ref))) / scale
    out: dict[str, Any] = {"n": n_arr, "spectral": spec, "exact_scale": 1.0}
    for p, arr in fd.items():
        out[f"fd{p}"] = arr
    return out


# --------------------------------------------------------------------------- #
# 5. 拟合工具（对数-对数斜率 / 每加倍比值）
# --------------------------------------------------------------------------- #
def local_slopes(x: Any, y: Any) -> np.ndarray:
    r"""相邻点之间的双对数斜率 :math:`\Delta\log y/\Delta\log x`。

    「斜率本身在变」是指数收敛的可执行判据：代数收敛的斜率是常数，
    指数收敛的斜率**每一步都更负**（对数-对数图上曲线向下弯）。
    """
    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if xv.size != yv.size or xv.size < 2:
        raise ValueError("x 与 y 必须等长且至少两个点")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diff(np.log(yv)) / np.diff(np.log(xv))


def doubling_ratios(x: Any, y: Any, *, factor: float = 2.0) -> np.ndarray:
    r"""相邻倍点之间 :math:`y` 的**比值**（代数收敛是常数，指数收敛越变越小）。

    为了在非等比的网格上也成立，这里按 :math:`\log x` 的对数间距把比值归一化到
    「每乘 factor 倍」：:math:`r_i=(y_{i+1}/y_i)^{1/s_i}`，:math:`s_i=\log(x_{i+1}/x_i)/\log(\text{factor})`。
    """
    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if xv.size != yv.size or xv.size < 2:
        raise ValueError("x 与 y 必须等长且至少两个点")
    s = np.log(xv[1:] / xv[:-1]) / np.log(float(factor))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (yv[1:] / yv[:-1]) ** (1.0 / s)


def fit_power_law(x: Any, y: Any, *, lo: float | None = None, hi: float | None = None) -> dict[str, Any]:
    r"""窗口内双对数最小二乘 :math:`y\approx C x^{p}`。

    返回 ``p`` / ``log10_C`` / ``r2`` / ``n_used`` / ``window``。
    只喂幂律关系（PROTOCOLS §8.4）：指数收敛的曲线也能被硬拟出一个 :math:`p`，
    所以本章一律同时报 :math:`R^2` 与**斜率是否在变**，不靠单个 :math:`p` 下结论。
    """
    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if xv.size != yv.size:
        raise ValueError("x 与 y 必须等长")
    sel = np.ones(xv.size, dtype=bool)
    if lo is not None:
        sel &= xv >= float(lo)
    if hi is not None:
        sel &= xv <= float(hi)
    sel &= yv > 0.0
    xs = np.log(xv[sel])
    ys = np.log(yv[sel])
    n_used = int(xs.size)
    if n_used < 2:
        return {"p": float("nan"), "log10_C": float("nan"), "r2": float("nan"), "n_used": n_used,
                "window": (lo, hi)}
    mx = float(np.mean(xs))
    my = float(np.mean(ys))
    sxx = float(np.sum((xs - mx) ** 2))
    sxy = float(np.sum((xs - mx) * (ys - my)))
    slope = sxy / sxx if sxx > 0.0 else 0.0
    inter = my - slope * mx
    pred = slope * xs + inter
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - my) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"p": float(slope), "log10_C": float(inter / np.log(10.0)), "r2": float(r2),
            "n_used": n_used, "window": (lo, hi)}


# --------------------------------------------------------------------------- #
# 6. Chebyshev 侧
# --------------------------------------------------------------------------- #
def chebyshev_grid(n: int) -> np.ndarray:
    r"""Chebyshev--Gauss--Lobatto 点 :math:`x_j=\cos(\pi j/n)`，``j=0..n``，从 +1 降到 -1。

    含端点（这是谱方法要的：边界条件挂在端点上），与 2.1/2.3 章的高斯点
    :math:`\cos((2j+1)\pi/(2n))`（不含端点）是**两套不同的点**，混用会让端点处的
    导数与边界条件全错位。
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n 必须 >= 1，得到 {n}")
    return np.cos(np.pi * np.arange(n + 1, dtype=float) / float(n))


def chebyshev_grid_ascending(n: int) -> np.ndarray:
    """升序版本的 :func:`chebyshev_grid`（绘图与插值误差扫描用）。"""
    return chebyshev_grid(n)[::-1].copy()


def chebyshev_coeffs(u: Any) -> np.ndarray:
    r"""Chebyshev 插值系数（DCT-I 的定义式）：
    :math:`c_k=\frac{2}{n}\frac{1}{1+\delta_{k0}+\delta_{kn}}
    \sum_{j=0}^{n}{}'' u_j\cos(\pi jk/n)`。

    实现是显式的余弦矩阵乘法（:math:`O(n^2)`，本章 :math:`n\le256`）：
    不引入任何「成品 DCT」，并且端点权重 :math:`{}''` 写在一处，
    免得「系数差 2 倍」这类错误靠调容差蒙过去。
    """
    arr = np.asarray(u, dtype=float).reshape(-1)
    n = arr.size - 1
    if n < 1:
        raise ValueError("至少要 2 个结点")
    j = np.arange(n + 1, dtype=float)
    w = np.ones(n + 1)
    w[0] = 0.5
    w[n] = 0.5
    cosmat = np.cos(np.pi * np.outer(j, j) / float(n))  # [k, j]
    c = (2.0 / float(n)) * (cosmat @ (w * arr))
    c[0] *= 0.5
    c[n] *= 0.5
    return c


def chebyshev_eval(coeffs: Any, x: Any) -> np.ndarray:
    r"""Clenshaw 递推求 :math:`\sum_k c_k T_k(x)`（对 :math:`|x|\le1` 稳定）。"""
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    xs = np.asarray(x, dtype=float)
    if c.size == 0:
        raise ValueError("系数不能为空")
    b1 = np.zeros_like(xs)
    b2 = np.zeros_like(xs)
    for j in range(c.size - 1, 0, -1):
        b1, b2 = 2.0 * xs * b1 - b2 + c[j], b1
    return xs * b1 - b2 + c[0]


def chebyshev_differentiation_matrix(n: int) -> np.ndarray:
    r"""Chebyshev 搭配点微分矩阵（Trefethen 的 ``cheb.m`` 公式）。

    :math:`D_{ij}` 的定义是「插值多项式的导数在结点上的值 = D 乘结点值」。
    端点行对应的是**没有加边界条件**的导数：在边值问题里这些行要被边界条件替换，
    这一条是「Chebyshev 谱方法怎么处理边界」的口径（本章只做微分与插值，
    不重复 3.4 章的 BVP 求解器）。
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n 必须 >= 1，得到 {n}")
    x = chebyshev_grid(n)
    c = np.ones(n + 1)
    c[0] = 2.0
    c[n] = 2.0
    c = c * ((-1.0) ** np.arange(n + 1, dtype=float))
    # dX[i, j] = x_i - x_j（与 Trefethen 的 X-X' 同序：写反会把整张矩阵的符号翻过来，
    # 而 D@x = -1 这种结果在只看「误差很小」的测试里完全看不出来）
    dX = x[:, None] - x[None, :]
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    return D - np.diag(D.sum(axis=1))


def chebyshev_derivative(u: Any) -> np.ndarray:
    """Chebyshev 谱微分：``D @ u``（结点值进、导数的结点值出）。"""
    arr = np.asarray(u, dtype=float).reshape(-1)
    n = arr.size - 1
    if n < 1:
        raise ValueError("至少要 2 个结点值")
    return chebyshev_differentiation_matrix(n).dot(arr)


def chebyshev_truncation_error(kind: str, n: int, *, n_eval: int = 2001) -> float:
    r"""Chebyshev 插值（截断）误差的 :math:`\ell_\infty` 实测值。

    在 :math:`2n_{\text{eval}}+1` 个升序稠密点上比较「插值多项式」与「解析函数」，
    所以它量的是**逼近误差**，与微分矩阵的舍入无关。
    """
    xs = np.cos(np.pi * np.arange(n_eval, -1, -1, dtype=float) / float(n_eval))
    values = cheb_profile(kind, chebyshev_grid(n))
    c = chebyshev_coeffs(values)
    approx = chebyshev_eval(c, xs)
    return float(np.max(np.abs(approx - cheb_profile(kind, xs))))


def chebyshev_convergence(
    kind: str,
    ns: Sequence[int],
    *,
    n_eval: int = 2001,
) -> tuple[np.ndarray, np.ndarray]:
    """误差-自由度曲线：``(ns, errs)``。解析函数指数下降、尖角函数代数下降。"""
    n_arr = np.asarray([int(v) for v in ns], dtype=float)
    errs = np.asarray([chebyshev_truncation_error(kind, int(v), n_eval=n_eval) for v in n_arr], dtype=float)
    return n_arr, errs


# --------------------------------------------------------------------------- #
# 7. 一维周期 Burgers（伪谱 + 可选去混叠）
# --------------------------------------------------------------------------- #
def _burgers_initial(kind: str, x: np.ndarray, *, length: float, seed: int) -> np.ndarray:
    from numcore.utils import rng  # 局部 import：只有这里需要随机性（POLICY P4）

    key = str(kind)
    if key == "sine":
        return np.sin(TWO_PI * x / length)
    if key == "bump":
        return np.exp(-((x - 0.5 * length) / (0.12 * length)) ** 2) - 0.2
    if key == "random":
        g = rng(int(seed))
        u = np.zeros_like(x)
        for kk in range(1, 9):
            u = u + (g.standard_normal() * np.cos(TWO_PI * kk * x / length)
                     + g.standard_normal() * np.sin(TWO_PI * kk * x / length)) / float(kk)
        return u / max(float(np.max(np.abs(u))), 1e-300)
    if key == "two_mode":
        return np.sin(TWO_PI * x / length) + 0.5 * np.sin(2.0 * TWO_PI * x / length + 0.7)
    raise KeyError(f"未知初值 {kind!r}（可用：sine/bump/random/two_mode）")


@dataclass(frozen=True)
class BurgersResult:
    r"""一次 Burgers 求解的全部可观测物（纯数据，不持有求解器状态）。

    * ``u`` —— 快照解，形状 ``(n_snap, n)``；
    * ``spectra`` —— 单边能谱 :math:`E_k`，形状 ``(n_snap, n//2+1)``，``sum_k E_k = mean(u^2)``；
    * ``budget_residual`` —— 能量收支相对残差
      :math:`(E(t)-E(0)+2\nu\int_0^t\Omega)/(E(0)+2\nu\int_0^t\Omega)`；
      精确解恒为 0，实测值同时含时间离散误差与混叠误差；
    * ``flux`` —— 当前时刻的能流 :math:`\Pi(k)`（对当前步的非线性项算），
      惯性区里它应当贴住耗散率 :math:`\varepsilon=2\nu\Omega`。
    """

    n: int
    nu: float
    length: float
    initial: str
    dealias: bool
    dt: float
    n_steps: int
    times: np.ndarray
    u: np.ndarray
    spectra: np.ndarray
    k: np.ndarray
    energy: np.ndarray
    enstrophy: np.ndarray
    dissipation: np.ndarray
    budget_times: np.ndarray
    budget_residual: np.ndarray
    flux: np.ndarray
    flux_time: float
    dt_cfl: float


def energy_spectrum(u: Any, *, length: float = TWO_PI) -> tuple[np.ndarray, np.ndarray]:
    r"""单边能谱 :math:`E_k=\sum_{|p|=k}|a_p|^2`（``k = 0..N/2``）。

    归一化按 Parseval：:math:`\sum_k E_k=\frac1L\int u^2`（离散上是 ``mean(u^2)``）。
    把 :math:`\pm k` 合并成一条曲线是**惯例**，但读图时必须记住
    「一个 :math:`E_k` 代表两个模态」，否则「能谱下降了」的账会差一倍。
    """
    arr = np.asarray(u, dtype=float).reshape(-1)
    a = fourier_coeffs(arr, length=length)
    n = arr.size
    half = n // 2
    k = np.arange(half + 1, dtype=float)
    e = np.zeros(half + 1)
    e[0] = float(np.abs(a[0]) ** 2)
    if half >= 1:
        idx = np.arange(1, half + 1)
        e[idx] = np.abs(a[idx]) ** 2 + np.abs(a[n - idx]) ** 2
    if n % 2 == 0:
        e[half] = float(np.abs(a[half]) ** 2)
    return k, e


def spectral_flux(a: Any, w: Any, *, length: float = TWO_PI) -> tuple[np.ndarray, np.ndarray]:
    r"""能流 :math:`\Pi(k)=-2\sum_{|p|\le k}T_p`，:math:`T_p=\mathrm{Re}(\overline{a_p}N_p)`，
    :math:`N_p=-\frac{i}{2}\kappa_p w_p`。

    符号约定：:math:`\Pi(k)>0` 表示能量由低波数**流向**高波数（正向级联）。
    惯性区里 :math:`\Pi(k)\approx\varepsilon=2\nu\Omega`，在 :math:`k_{\max}` 处回落到 0
    （非线性项在去混叠下守恒总能量）。两头都是可以实测的：中间的平台与末端的回落。
    """
    av = np.asarray(a, dtype=complex).reshape(-1)
    wv = np.asarray(w, dtype=complex).reshape(-1)
    if av.size != wv.size:
        raise ValueError("a 与 w 长度必须相同")
    n = av.size
    kappa = TWO_PI * fourier_wavenumbers(n) / float(length)
    transfer = np.real(np.conj(av) * (-0.5j * kappa * wv))
    kpos = np.abs(fourier_wavenumbers(n))
    half = n // 2
    ks = np.arange(half + 1, dtype=float)
    flux = np.zeros(half + 1)
    for i in range(half + 1):
        sel = kpos <= (float(i) + 1e-12)
        flux[i] = -2.0 * float(np.sum(transfer[sel]))
    return ks, flux


def cascade_window(nu: float, t: float, n: int, *, lo_factor: float = 3.0, hi_factor: float = 1.5) -> tuple[int, int]:
    r"""级联惯性区的建议拟合窗口 :math:`[k_{\text{lo}},k_{\text{hi}}]`。

    :math:`k_{\text{lo}}=3`（注入尺度之上），
    :math:`k_{\text{hi}}=\min(1.5\,k_d,\ N/3)`，:math:`k_d\approx(\nu t)^{-1/2}`。
    上界取**两个**限制里更小的那个：黏性截断之上没有惯性区，而 :math:`N/3` 之上
    是 2/3 规则动过的模态——拿它们拟合斜率等于把去混叠的边界当物理。
    **窗口是断言的组成部分**（PROTOCOLS §8.4）：换窗口斜率就会变，必须写进 theory。
    """
    kd = 1.0 / max(float(np.sqrt(max(float(nu) * max(float(t), 1e-12), 1e-300))), 1e-300)
    k_lo = max(int(lo_factor), 2)
    k_hi = int(min(hi_factor * kd, float(n) / 3.0, float(n) / 2.0))
    k_hi = max(k_hi, k_lo + 1)
    k_hi = min(k_hi, n // 2)
    return k_lo, k_hi


def burgers_solve(
    n: int,
    nu: float,
    t_end: float,
    *,
    length: float = TWO_PI,
    initial: str = "sine",
    dealias: bool = True,
    cfl: float = 0.35,
    dt: float | None = None,
    u0: Any | None = None,
    n_snapshots: int = 9,
    seed: int = 20261018,
    max_steps: int = 40000,
) -> BurgersResult:
    r"""一维周期 Burgers :math:`u_t+uu_x=\nu u_{xx}` 的**积分因子 RK4** + 伪谱求解。

    为什么不是普通 RK4：显式格式对黏性项的稳定限制是 :math:`\Delta t\lesssim
    2.78/(\nu k_{\max}^2)`。取 :math:`N=256,\nu=0.02` 时它是 :math:`6\times10^{-4}`，
    而 CFL 只要求 :math:`8.6\times10^{-3}`——**步长被黏性项卡死 14 倍**，一帧就要上千步。
    把线性项 :math:`-\nu k^2` 用积分因子 :math:`e^{-\nu k^2\Delta t}` 精确吸收后，
    稳定限制只剩对流 CFL：``dt = cfl*dx/max|u|``（实测同一算例 656 -> 175 步，
    4 阶精度不变，见 ``tests/4.6`` 里对 DOP853 的收敛阶梯）。

    非线性项用 :func:`dealias_product_hat`（``dealias=True``，2/3 截断 + 3/2 补零）
    或 :func:`product_naive_hat`（``dealias=False``）；能量收支
    :math:`E(t)+2\nu\int\Omega=E(0)` 在整段积分里逐点记账。
    **注意口径**：在分辨良好时混叠对总能量账的影响很小（实测相对差 1e-4 量级，
    被时间离散误差盖住），它真正污染的是非线性项在保留带里的取值；
    :attr:`BurgersResult.budget_residual` 是那段账的残差，不是「去混叠的证明」。
    """
    n = int(n)
    nu = float(nu)
    t_end = float(t_end)
    if n < 8:
        raise ValueError(f"n 必须 >= 8，得到 {n}")
    if nu <= 0.0:
        raise ValueError(f"nu 必须 > 0，得到 {nu}")
    if t_end < 0.0:
        raise ValueError(f"t_end 必须 >= 0，得到 {t_end}")
    x = fourier_grid(n, length=length)
    kappa = TWO_PI * fourier_wavenumbers(n) / float(length)
    k2 = kappa * kappa

    u0v = _burgers_initial(initial, x, length=length, seed=seed) if u0 is None else np.asarray(u0, dtype=float).reshape(-1)
    if u0v.size != n:
        raise ValueError(f"u0 长度必须为 {n}，得到 {u0v.size}")

    def nonlinear(a_hat: np.ndarray) -> np.ndarray:
        # 系数空间入口：一次非线性求值 = 2 次 M 点（2 的幂）FFT，而不是 3 次含 n 点 Bluestein
        if dealias:
            w = dealias_product_hat(a_hat, a_hat, keep_two_thirds=True)
        else:
            w = product_naive_hat(a_hat, a_hat)
        return -0.5j * kappa * w

    a = fourier_coeffs(u0v, length=length)
    e0 = float(np.sum(np.abs(a) ** 2))
    umax = max(float(np.max(np.abs(u0v))), 1e-12)
    dx = float(length) / float(n)
    k_max = float(np.max(np.abs(kappa)))
    # 积分因子吸收线性项后，黏性不再限制步长；只留对流 CFL（并有 RK4 的安全系数 cfl）
    dt_cfl = float(cfl) * dx / umax
    dt_use = dt_cfl if dt is None else float(dt)
    if t_end == 0.0:
        n_steps = 0
        dt_use = 0.0
    else:
        n_steps = int(np.ceil(t_end / max(dt_use, 1e-300)))
        if n_steps > int(max_steps):
            n_steps = int(max_steps)
        dt_use = t_end / float(n_steps)

    n_snap = max(int(n_snapshots), 1)
    snap_times = np.linspace(0.0, t_end, n_snap) if n_snap > 1 else np.asarray([0.0])
    u_snap = np.zeros((n_snap, n))
    spec_snap = np.zeros((n_snap, n // 2 + 1))
    energy = np.zeros(n_snap)
    enstrophy = np.zeros(n_snap)
    k_axis = np.arange(n // 2 + 1, dtype=float)
    resid_times = np.zeros(n_steps + 1)
    resid_vals = np.zeros(n_steps + 1)

    def record(idx: int, t_now: float) -> None:
        uv = np.real(fourier_synthesize(a))
        _, ek = energy_spectrum(uv, length=length)
        u_snap[idx] = uv
        spec_snap[idx] = ek
        energy[idx] = float(np.sum(ek))
        enstrophy[idx] = float(np.sum(k2 * np.abs(a) ** 2))

    record(0, 0.0)
    cum_diss = 0.0
    resid_times[0] = 0.0
    denom0 = max(e0, 1e-300)
    resid_vals[0] = 0.0
    t_now = 0.0
    snap_i = 1
    omega_prev = float(np.sum(k2 * np.abs(a) ** 2))
    for step in range(n_steps):
        # 积分因子 RK4：E1 = exp(-nu k^2 dt)、Eh = exp(-nu k^2 dt/2)，线性部分**精确**
        e1 = np.exp(-nu * k2 * dt_use)
        eh = np.exp(-0.5 * nu * k2 * dt_use)
        n1 = nonlinear(a)
        n2 = nonlinear(eh * (a + 0.5 * dt_use * n1))
        n3 = nonlinear(eh * a + 0.5 * dt_use * n2)
        n4 = nonlinear(e1 * a + dt_use * eh * n3)
        a = e1 * a + (dt_use / 6.0) * (e1 * n1 + 2.0 * eh * (n2 + n3) + n4)
        t_now += dt_use
        omega_new = float(np.sum(k2 * np.abs(a) ** 2))
        cum_diss += 0.5 * (omega_prev + omega_new) * dt_use
        omega_prev = omega_new
        e_now = float(np.sum(np.abs(a) ** 2))
        resid_times[step + 1] = t_now
        resid_vals[step + 1] = abs(e_now - e0 + 2.0 * nu * cum_diss) / max(denom0, 2.0 * nu * cum_diss, 1e-300)
        while snap_i < n_snap and t_now >= snap_times[snap_i] - 1e-12:
            record(snap_i, t_now)
            snap_i += 1
    if snap_i < n_snap:
        while snap_i < n_snap:
            record(snap_i, t_now)
            snap_i += 1

    # 当前时刻的能流：用与推进同一条非线性项（去混叠与否一致），避免"画的和算的不是一回事"
    u_now = np.real(fourier_synthesize(a))
    w_now = (dealias_product(u_now, u_now, length=length, keep_two_thirds=True)
             if dealias else product_naive(u_now, u_now, length=length))
    ks, flux = spectral_flux(a, w_now, length=length)
    dissipation = 2.0 * nu * np.array([float(np.sum(k2 * np.abs(fourier_coeffs(u_snap[i], length=length)) ** 2))
                                       for i in range(n_snap)])
    return BurgersResult(
        n=n,
        nu=nu,
        length=float(length),
        initial=str(initial),
        dealias=bool(dealias),
        dt=float(dt_use),
        n_steps=int(n_steps),
        times=snap_times,
        u=u_snap,
        spectra=spec_snap,
        k=k_axis,
        energy=energy,
        enstrophy=enstrophy,
        dissipation=dissipation,
        budget_times=resid_times,
        budget_residual=resid_vals,
        flux=flux,
        flux_time=float(t_now),
        dt_cfl=float(dt_cfl),
    )
