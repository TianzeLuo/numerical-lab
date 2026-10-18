r"""2.5 章的数值内核：离散 Fourier 变换（DFT）与快速算法。

数学对象
--------
设 ``x[0..N-1]`` 是区间 :math:`[0,T)` 上以 :math:`\Delta t = T/N` 采样的复序列，
则它的 **DFT** 与 **逆 DFT** 定义为

.. math::

    X_k = \sum_{n=0}^{N-1} x_n\, e^{-2\pi i nk/N}, \qquad
    x_n = \frac{1}{N}\sum_{k=0}^{N-1} X_k\, e^{+2\pi i nk/N},

其中频率栅格为 :math:`f_k = k/T = k/(N\Delta t)`（Hz），角频率 :math:`\omega_k = 2\pi f_k`。
本模块只用 numpy 的容器与逐元素原语实现这些式子：

* :func:`dft_direct` / :func:`idft_direct` —— :math:`O(N^2)` 的定义式与 DFT 矩阵形式；
* :func:`radix2_fft` —— **自研** radix-2 Cooley-Tukey（递归版，非 2 的幂按 ``pad`` 决定
  补零或改走 Bluestein）；
* :func:`radix2_fft_iterative` —— 同一定义的迭代版（位反转置换 + 逐级蝶形），
  与 :func:`butterfly_schedule` 共用同一套「级 → 块 → 蝶形对」的索引约定；
* :func:`bluestein_fft` —— 任意长度（含质数长度）的 chirp-z 卷积实现；
* :func:`rfft` —— 实序列 FFT，返回 :math:`N/2+1` 根**单边**谱线；
* :func:`circular_convolution` / :func:`linear_convolution` / :func:`fft_convolve` ——
  循环卷积、线性卷积与「频域相乘」三者的对照（卷积定理）；
* 窗函数族与旁瓣、主瓣宽度、分辨率阈值等频谱分析量。

约定
----
* 归一化：``ifft(fft(x)) == x``（逆变换带 :math:`1/N`），与教材惯例一致；
* 幅度谱：:func:`amplitude_spectrum` 返回**单边**幅度（内部谱线 ×2，DC/Nyquist 除外），
  因此正弦的峰值就是它的幅度（不是 :math:`A/2`）；
* 无随机性、无 IO、无全局状态（POLICY P6）：所有函数只吃数组、只吐数组。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    # DFT 与快速算法
    "dft_matrix",
    "dft_direct",
    "idft_direct",
    "fft",
    "ifft",
    "radix2_fft",
    "radix2_fft_iterative",
    "bluestein_fft",
    "rfft",
    "irfft",
    "zero_pad",
    "next_power_of_two",
    "bluestein_length",
    # 分解结构
    "FFTPlan",
    "plan_fft",
    "ButterflyStep",
    "Radix2Stage",
    "butterfly_schedule",
    "butterfly_diagram",
    "butterfly_only_values",
    "twiddle",
    "bit_reverse_indices",
    "dif_frequency_order",
    "butterfly_sweep",
    "level_residuals",
    # 频谱
    "frequencies",
    "amplitude_spectrum",
    "power_spectrum",
    "phase_spectrum",
    "db",
    "dB",
    "parabolic_peak",
    "peak_frequency",
    "main_lobe",
    "peak_sidelobe_analytic",
    "main_lobe_scan_first_null",
    "n_local_maxima",
    "separable",
    "min_separable_separation",
    "rayleigh_resolution",
    "frequency_resolution",
    # 窗
    "WINDOW_NAMES",
    "window",
    "window_spectrum",
    "window_coherent_gain",
    "window_energy",
    "coherent_sum",
    "peak_sidelobe_level",
    "sidelobe_envelope",
    "WINDOW_SIDELOBE_DB",
    "WINDOW_MAINLOBE_BINS",
    "WINDOW_ENVELOPE_POWER",
    # 卷积与能量
    "circular_convolution",
    "linear_convolution",
    "fft_convolve",
    "frequency_product",
    "convolution_theorem_error",
    "wrap_around_error",
    "time_energy",
    "frequency_energy",
    "parseval_relative_error",
    "circular_derivative",
    # 信号构造与估计
    "sinusoid_sum",
    "two_tone",
    "square_wave",
    "shift_sample",
    "alias_index",
    "alias_frequency",
    "alias_candidates",
    "goertzel",
    "dft_vs_fft_error",
]


# --------------------------------------------------------------------------- #
# 基础：频率栅格与 DFT
# --------------------------------------------------------------------------- #
def next_power_of_two(n: int) -> int:
    """不小于 ``n`` 的最小 2 的幂（``n ≤ 1`` 时返回 1）。"""
    m = 1
    while m < int(n):
        m <<= 1
    return m


def bluestein_length(n: int) -> int:
    """Bluestein 用的线性卷积长度（2 的幂且 ≥ 2n−1）。"""
    if n < 1:
        raise ValueError(f"序列长度必须为正，得到 {n}")
    return next_power_of_two(2 * int(n) - 1)


def frequencies(n: int, *, dt: float = 1.0, sort: bool = False) -> np.ndarray:
    """DFT 频率栅格 :math:`f_k = k/(N\\Delta t)`（单位 Hz，未做 fftshift）。

    ``sort=True`` 时给出 ``[-Nyquist, Nyquist)`` 的升序栅格（负频率在前）。
    """
    if n < 1:
        raise ValueError(f"序列长度必须为正，得到 {n}")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"采样间隔 dt 必须为正的有限数，得到 {dt}")
    k = np.arange(int(n), dtype=float)
    if sort:
        k = np.where(k < (n + 1) // 2, k, k - n)
        k = np.sort(k)
    return k / (int(n) * dt)


def dft_matrix(n: int) -> np.ndarray:
    """DFT 矩阵 :math:`W_{kn} = e^{-2\\pi i kn/N}`（形状 ``(n, n)``，复数 128 位）。

    实现上**先在整数里把 ``k·n`` 对 ``N`` 取模**再算三角函数：``k·n`` 最大到
    :math:`(N-1)^2`，而它的绝对大小毫无意义（只差 :math:`2\\pi` 的整数倍）。
    先乘后算在 ``N`` 稍大时就会掉精度——2.5 章实测 ``N = 200`` 时最坏逐点差已达
    :math:`10^{4}` 量级（见 ``pitfalls``），而纸面上两者是同一个式子。
    """
    if n < 1:
        raise ValueError(f"序列长度必须为正，得到 {n}")
    k = np.arange(int(n), dtype=np.int64).reshape(-1, 1)
    idx = np.arange(int(n), dtype=np.int64).reshape(1, -1)
    red = (k * idx) % int(n)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(-2j * np.pi * red.astype(float) / float(n))


def dft_direct(x: Any) -> np.ndarray:
    """DFT 定义式（:math:`O(N^2)`，用于对拍与教学，不做任何快速分解）。"""
    a = np.asarray(x, dtype=complex).ravel()
    n = a.size
    if n == 0:
        raise ValueError("空序列没有 DFT")
    return dft_matrix(n) @ a


def idft_direct(X: Any) -> np.ndarray:
    """逆 DFT 定义式（:math:`O(N^2)`，带 :math:`1/N` 归一化）。"""
    a = np.asarray(X, dtype=complex).ravel()
    n = a.size
    if n == 0:
        raise ValueError("空序列没有逆 DFT")
    return np.conj(dft_matrix(n)) @ a / float(n)


def zero_pad(x: Any, n: int) -> np.ndarray:
    """在**尾部**补零到长度 ``n``（``n`` 小于原长则截断）；补零只加密频率栅格，
    不增加任何信息（见 :func:`min_separable_separation` 与 ``pitfalls``）。"""
    a = np.asarray(x, dtype=complex).ravel()
    n = int(n)
    if n < 1:
        raise ValueError(f"目标长度必须为正，得到 {n}")
    out = np.zeros(n, dtype=complex)
    m = min(n, a.size)
    out[:m] = a[:m]
    return out


# --------------------------------------------------------------------------- #
# radix-2 Cooley-Tukey：递归 / 迭代 / 分解结构
# --------------------------------------------------------------------------- #
def twiddle(n: int, *, inverse: bool = False) -> np.ndarray:
    """旋转因子表 :math:`W_N^k = e^{\\mp 2\\pi i k/N}`，``k = 0..N-1``。"""
    if n < 1:
        raise ValueError(f"长度必须为正，得到 {n}")
    k = np.arange(int(n), dtype=float)
    sign = 1.0 if inverse else -1.0
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(sign * 2j * np.pi * k / float(n))


def radix2_fft(x: Any, *, pad: bool = False, inverse: bool = False) -> np.ndarray:
    """自研 radix-2 Cooley-Tukey FFT（递归分治）。

    :math:`X_k = E_k + W_N^k O_k`、:math:`X_{k+N/2} = E_k - W_N^k O_k`，
    其中 :math:`E`、:math:`O` 分别是偶、奇下标的 :math:`N/2` 点 DFT。

    参数
    ----
    pad
        ``False``（缺省）：长度不是 2 的幂时改走 :func:`bluestein_fft`（**精确**，不引入补零）；
        ``True``：补零到下一个 2 的幂再分治。两者结果都等于定义式的 DFT（同一批采样点），
        但补零路径要多算那些零的变换，是纯浪费——这正是 ``pitfalls`` 里「补零不提高分辨率」
        在算法层的同一条事实。
    """
    a = np.asarray(x, dtype=complex).ravel()
    n = a.size
    if n == 0:
        raise ValueError("空序列没有 DFT")
    if n & (n - 1):
        if pad:
            return radix2_fft(zero_pad(a, next_power_of_two(n)), inverse=inverse)
        return bluestein_fft(a, inverse=inverse)
    if n == 1:
        return a.copy()
    half = n // 2
    even = radix2_fft(a[0::2], inverse=inverse)
    odd = radix2_fft(a[1::2], inverse=inverse)
    w = twiddle(n, inverse=inverse)[:half]
    t = w * odd
    out = np.empty(n, dtype=complex)
    out[:half] = even + t
    out[half:] = even - t
    return out / float(n) if inverse else out


def bit_reverse_indices(n: int) -> np.ndarray:
    """位反转置换索引（长度必须是 2 的幂）：``i -> reverse_bits(i, log2 n)``。"""
    if n < 1 or (n & (n - 1)):
        raise ValueError(f"位反转置换要求长度是 2 的幂，得到 {n}")
    bits = int(round(np.log2(n)))
    idx = np.arange(n, dtype=np.int64)
    out = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        out = (out << 1) | ((idx >> b) & 1)
    return out


def radix2_fft_iterative(x: Any, *, pad: bool = False, inverse: bool = False) -> np.ndarray:
    """radix-2 FFT 的**迭代**版：位反转置换 + :math:`\\log_2 N` 级蝶形。

    与 :func:`radix2_fft`（递归版）是同一个数学对象的两种求值顺序，逐点差别应在舍入级；
    ``tests/2.5`` 用「不同长度（2 的幂 / 非 2 的幂 / 质数）」把两条路径都钉住。
    """
    a = np.asarray(x, dtype=complex).ravel()
    n = a.size
    if n == 0:
        raise ValueError("空序列没有 DFT")
    if n & (n - 1):
        if pad:
            return radix2_fft_iterative(zero_pad(a, next_power_of_two(n)), inverse=inverse)
        return bluestein_fft(a, inverse=inverse)
    if n == 1:
        return a.copy()
    out = a[bit_reverse_indices(n)].astype(complex, copy=True)
    length = 2
    while length <= n:
        half = length // 2
        w = twiddle(length, inverse=inverse)[:half]
        for start in range(0, n, length):
            lo = out[start : start + half]
            hi = out[start + half : start + length]
            t = w * hi
            # 先算出两侧再写回：``lo`` 是**视图**，先写 ``lo + t`` 会污染紧接着要用的 ``lo - t``
            # （这个别名坑实测让 N=2 的结果从 [3, -1] 变成 [3, +1]，见 pitfalls）。
            new_lo = lo + t
            new_hi = lo - t
            out[start : start + half] = new_lo
            out[start + half : start + length] = new_hi
        length <<= 1
    # 逆变换的 1/N：**必须**在这里施加，否则「正变换→乘→逆变换」整条链会整体放大 N 倍
    # （Bluestein 只是把这条链用在自己内部，于是它的误差恰好是 m 倍——2.5 章真踩过）。
    return out / float(n) if inverse else out


def bluestein_fft(x: Any, *, inverse: bool = False) -> np.ndarray:
    """Bluestein（chirp-z）算法：任意长度 DFT，含质数长度。

    用 :math:`2nk = n^2 + k^2 - (n-k)^2` 把 DFT 写成卷积

    .. math::

        X_k = b_k^* \\sum_{n} \\big(x_n b_n^*\\big)\\, b_{k-n},
        \\qquad b_j = e^{\\mp \\pi i j^2/N},

    卷积长度取 :math:`M \\ge 2N-1` 的 2 的幂，用 :func:`fft` 的前向 + 逆变换实现（结果取前 N 项）。
    """
    a = np.asarray(x, dtype=complex).ravel()
    n = a.size
    if n == 0:
        raise ValueError("空序列没有 DFT")
    if n & (n - 1) == 0:
        return radix2_fft(a, inverse=inverse)
    sign = 1.0 if inverse else -1.0
    m = bluestein_length(n)
    j = np.arange(n, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        b = np.exp(sign * 1j * np.pi * (j * j) / float(n))
    a_pad = np.zeros(m, dtype=complex)
    a_pad[:n] = a * b
    b_pad = np.zeros(m, dtype=complex)
    b_pad[:n] = np.conj(b)
    # 推导里的 b_{l-n}（l = 0..n-2）落在下标 m-n+1+l，等价于「最后 n-1 个位置」。
    # 写成 [m-n+1:] 会得到 n-1 项却少占一格（错位一个下标），实测让 3 点 DFT 差 8 倍。
    b_pad[m - (n - 1) :] = np.conj(b[1:])[::-1]
    conv = radix2_fft(a_pad) * radix2_fft(b_pad)
    y = fft(conv, inverse=True)
    return y[:n] * b


def fft(x: Any, *, inverse: bool = False) -> np.ndarray:
    """DFT 的统一入口：2 的幂走 radix-2（迭代版），否则走 Bluestein。

    这条「混合基/补零分支」正是只测 :math:`N = 2^k` 会漏掉的地方。
    """
    a = np.asarray(x, dtype=complex).ravel()
    n = a.size
    if n == 0:
        raise ValueError("空序列没有 DFT")
    if n & (n - 1) == 0:
        return radix2_fft_iterative(a, inverse=inverse)
    # radix-2 分支自己带 1/N，Bluestein 分支返回的是**未归一化**的逆变换，必须在这里补上：
    # 否则 ``ifft`` 在非 2 的幂长度上整体放大 N 倍（实测 n=3 差 3 倍、n=1000 差 1000 倍，
    # 而 n=2^k 完全正常——这正是「只测 2 的幂会漏掉」的现场）。
    out = bluestein_fft(a, inverse=inverse)
    return out / float(n) if inverse else out


def ifft(X: Any) -> np.ndarray:
    """逆 DFT（:math:`1/N` 归一化）：``ifft(fft(x)) == x``。"""
    return fft(X, inverse=True)


def rfft(x: Any) -> np.ndarray:
    """实序列的单边谱（长度 ``N//2 + 1``）：返回 :math:`X_0..X_{\\lfloor N/2\\rfloor}`。

    实序列满足共轭对称 :math:`X_{N-k} = \\overline{X_k}`，所以另一半是冗余的
    ——但**能量只算一半是不够的**，见 :func:`frequency_energy` 的 ``one_sided`` 口径。
    """
    a = np.asarray(x, dtype=float).ravel()
    n = a.size
    if n == 0:
        raise ValueError("空序列没有 DFT")
    return fft(a.astype(complex))[: n // 2 + 1]


def irfft(X: Any, n: int | None = None, *, odd: bool = False) -> np.ndarray:
    """由单边谱还原实序列（先按共轭对称补全再逆变换，结果取实部）。

    ``odd=True`` 表示原序列长度为奇数（此时单边谱共 ``(n+1)/2`` 根，
    没有 Nyquist 项，补全时尾部不丢谱线）；``odd=False``（缺省）按偶数长度处理。
    """
    a = np.asarray(X, dtype=complex).ravel()
    if a.size == 0:
        raise ValueError("空谱没有逆变换")
    if odd:
        nn = 2 * a.size - 1
    else:
        nn = 2 * (a.size - 1)
    m = int(n) if n is not None else nn
    if m < 1:
        raise ValueError(f"目标长度必须为正，得到 {m}")
    # 先在**单边谱自己的长度** nn 上补全共轭对称，再按目标长度 m 截断/补零。
    # 反过来（按 m 决定拷贝多少根）会在 m < nn 时把谱线的共轭对上错位置：
    # 实测 n=8 的往返误差 1.16e-2——量级不大，很容易被当成"容许的数值误差"。
    full = np.zeros(nn, dtype=complex)
    # 拷贝根数 = nn//2 + 1（DC 到 Nyquist **含**）；写成 (nn+1)//2 会在偶数长度上
    # 悄悄丢掉 Nyquist 那一根，往返误差 ~1e-2 看着像"数值误差"
    half = min(a.size, nn // 2 + 1)
    full[:half] = a[:half]
    for k in range(1, half):
        if k < nn - k:
            full[nn - k] = np.conj(a[k])
    if m != nn:
        out = np.zeros(m, dtype=complex)
        take = min(m, nn)
        out[:take] = full[:take]
        full = out
    return np.real(ifft(full))


@dataclass(frozen=True)
class FFTPlan:
    """一次 FFT 的**算法选择**（不是数值结果）：长度、路线、级数与旋转因子个数。"""

    n: int
    algorithm: str
    padded_length: int
    n_stages: int
    n_butterflies: int
    n_twiddles: int
    note: str = ""


def plan_fft(n: int, *, pad: bool = False) -> FFTPlan:
    """给出 :func:`fft`/``numcore`` 会为长度 ``n`` 选择的算法与代价。

    这既是文档也是断言对象：``n = 2^k`` 走 radix-2 :math:`O(N\\log N)`，
    其余长度走 Bluestein（同样 :math:`O(N\\log N)`，但要 3 次 2 的幂 FFT + 补零到
    :math:`M \\ge 2N-1`）；``pad=True`` 时非 2 的幂被强行补零到 :math:`2^{\\lceil\\log_2 N\\rceil}`。
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"序列长度必须为正，得到 {n}")
    if n & (n - 1) == 0:
        stages = int(round(np.log2(n)))
        return FFTPlan(n=n, algorithm="radix2-iterative", padded_length=n, n_stages=stages,
                       n_butterflies=n * stages // 2, n_twiddles=n - 1,
                       note="长度是 2 的幂：位反转 + 逐级蝶形")
    if pad:
        m = next_power_of_two(n)
        stages = int(round(np.log2(m)))
        return FFTPlan(n=n, algorithm="zero-pad+radix2", padded_length=m, n_stages=stages,
                       n_butterflies=m * stages // 2, n_twiddles=m - 1,
                       note=f"补零到 {m}：结果仍是同一批采样点的 DFT，多算的全是零")
    m = bluestein_length(n)
    stages = int(round(np.log2(m)))
    return FFTPlan(n=n, algorithm="bluestein", padded_length=m, n_stages=stages,
                   n_butterflies=3 * m * stages // 2, n_twiddles=m - 1,
                   note="任意长度：chirp 卷积，3 次 2 的幂 FFT")


@dataclass(frozen=True)
class ButterflyStep:
    """一个蝶形：``(i, j)`` 两个位置，旋转因子 ``w``，级号 ``level`` 与块号 ``block``。"""

    level: int
    block: int
    i: int
    j: int
    w: complex


@dataclass(frozen=True)
class Radix2Stage:
    """一级蝶形运算的**输入/输出快照**（``input[level]`` → ``output[level]``）。

    ``output`` 是该级结束时的数组：``y[i] = u + w*v``、``y[j] = u - w*v``，
    其中 ``u = input[i]``、``v = input[j]``、``w`` 是该蝶形的旋转因子，
    **不含**级末的 ``1/2`` 归一化（归一化只在 :func:`radix2_decomposition` 的
    ``partial`` 序列里施加，用于「部分变换 == 递归子变换」这条精确恒等式）。
    """

    level: int
    block_size: int
    input: np.ndarray
    output: np.ndarray
    steps: tuple[ButterflyStep, ...]


def butterfly_schedule(n: int, x: Any | None = None) -> tuple[Radix2Stage, ...]:
    """radix-2 **DIF**（频域抽取）的蝶形时间表：每级的输入快照、输出快照与蝶形清单。

    为什么用 DIF 而不是 DIT：DIF 的**块是连续的**（DIT 的块在两级之间是「隔点」取的），
    于是「第 ``level`` 级结束时每个连续块恰好是那一块的 DFT」可以逐块直接检验。
    分级约定（``level = 0`` 对应整段，``level = log2 N − 1`` 对应块长 2）：

    .. math::

        u = y_{\\text{level}}[b+m],\\quad v = y_{\\text{level}}[b+B/2+m],\\quad
        w = e^{-2\\pi i m/B},\\quad B = 2^{\\log_2 N-\\text{level}},
        y_{\\text{level}+1}[b+m] = u + v,\\quad
        y_{\\text{level}+1}[b+B/2+m] = (u - v)\\,w,

    ``m = 0..B/2-1``、``b`` 遍历块起点。走完 :math:`\\log_2 N` 级后输出是**位反序**的，
    再做一次 :func:`bit_reverse_indices` 置换即得 :math:`X_k`
    （``tests/2.5`` 用「DIF 输出 == DFT 的位反序」把这条钉住）。
    """
    if n < 2 or (n & (n - 1)):
        raise ValueError(f"蝶形分级要求 2 的幂且 ≥2，得到 {n}")
    levels = int(round(np.log2(n)))
    cur = np.arange(n, dtype=float).astype(complex) if x is None else np.asarray(x, dtype=complex).ravel()
    if cur.size != n:
        raise ValueError(f"输入长度 {cur.size} 与 n={n} 不一致")
    stages: list[Radix2Stage] = []
    for level in range(levels):
        block = n >> level
        half = block // 2
        w = twiddle(block)[:half]
        steps: list[ButterflyStep] = []
        new = cur.copy()
        for b in range(0, n, block):
            for m in range(half):
                i = b + m
                j = b + half + m
                steps.append(ButterflyStep(level=level, block=b // block, i=i, j=j, w=complex(w[m])))
            u = cur[b : b + half]
            v = cur[b + half : b + block]
            new[b : b + half] = u + v
            new[b + half : b + block] = (u - v) * w
        stages.append(Radix2Stage(level=level, block_size=block, input=cur.copy(), output=new.copy(),
                                  steps=tuple(steps)))
        cur = new
    return tuple(stages)


def butterfly_only_values(x: Any) -> np.ndarray:
    """只回逐级快照矩阵（``(levels+1, n)``：第 0 行是输入，最后一行是**位反序**的谱）。"""
    a = np.asarray(x, dtype=complex).ravel()
    n = a.size
    if n < 2 or (n & (n - 1)):
        raise ValueError(f"逐级快照要求 2 的幂且 ≥2，得到 {n}")
    stages = butterfly_schedule(n, a)
    values = np.zeros((len(stages) + 1, n), dtype=complex)
    values[0] = a
    for i, stage in enumerate(stages):
        values[i + 1] = stage.output
    return values


def dif_frequency_order(n: int) -> np.ndarray:
    """DIF 末行的**频率下标指纹**：末行第 ``i`` 位的值就是它承载的 DFT 下标 ``k``。

    做法是把复指数 :math:`x_m = e^{+2\\pi i k m/N}` 依次送入蝶形：它的 DFT 是位于
    ``k`` 的单点脉冲，于是末行里幅度最大的那个位置就对应 ``k``。用**复指数**而不是单位脉冲
    :math:`e_0`：脉冲的 :math:`X_k \\equiv 1` 处处同相，从相位里读不出任何位置信息
    （实测角度全为 0，指纹退化成全零表——这正是「差点写成假证据」的第 4 例）。
    """
    if n < 2 or (n & (n - 1)):
        raise ValueError(f"频率指纹要求 2 的幂且 ≥2，得到 {n}")
    levels = int(round(np.log2(n)))
    values = butterfly_only_values(np.zeros(n, dtype=complex))
    del values
    order = np.zeros(n, dtype=np.int64)
    m = np.arange(n, dtype=float)
    for k in range(n):
        x = np.exp(2j * np.pi * k * m / n)
        last = butterfly_only_values(x)[levels]
        order[int(np.argmax(np.abs(last)))] = k
    if len(set(order.tolist())) != n:
        raise ValueError("频率指纹不是置换，蝶形结构可疑")
    return order


def butterfly_sweep(x: Any, *, keep_levels: bool = True) -> np.ndarray:
    """radix-2 DIF 的最朴素写法（逐点二重循环），作为 :func:`level_residuals` 的独立参照。

    与 :func:`butterfly_schedule` 的向量化实现是**同一算法、两种写法**：
    ``MUST`` 逐点相同到舍入级。它的价值在于「残差为 0」这句话有明确定义——
    不是因为对上了另一个同样可疑的公式，而是对上了一段能逐行读懂的朴素循环。
    """
    a = np.asarray(x, dtype=complex).ravel()
    n = a.size
    if n < 2 or (n & (n - 1)):
        raise ValueError(f"朴素 DIF 要求 2 的幂且 ≥2，得到 {n}")
    levels = int(round(np.log2(n)))
    snaps = np.zeros((levels + 1, n), dtype=complex) if keep_levels else np.zeros((1, n), dtype=complex)
    cur = a.astype(complex, copy=True)
    if keep_levels:
        snaps[0] = cur
    for level in range(levels):
        B = n >> level
        half = B // 2
        nxt = np.empty(n, dtype=complex)
        for start in range(0, n, B):
            for m in range(half):
                u = cur[start + m]
                v = cur[start + half + m]
                w = np.exp(-2j * np.pi * m / B)
                nxt[start + m] = u + v
                nxt[start + half + m] = (u - v) * w
        cur = nxt
        if keep_levels:
            snaps[level + 1] = cur
    return snaps if keep_levels else cur.reshape(1, n)


def level_residuals(x: Any) -> np.ndarray:
    """逐级残差：``max|向量化蝶形快照 − 朴素 DIF 快照|``（长度 ``log2 N``）。

    这是**实现一致性**检查（两种写法逐点相同），不是数学恒等式检查；
    数学正确性由 :func:`dft_direct` 与 :func:`fft` 的对拍负责（``tests/2.5`` 里两条都在）。
    之所以要写清楚这一点：本节有一版试图用「连续区间 DFT」「全局位反序置换」当参照，
    残差稳定停在 :math:`O(1)`（实测 n=8 时 2.76 / 4.60），看起来像蝶形写错了，
    其实只是参照物与蝶形的**下标约定**不是同一个（详见 ``pitfalls``）。
    """
    a = np.asarray(x, dtype=complex).ravel()
    n = a.size
    if n < 2 or (n & (n - 1)):
        raise ValueError(f"逐级残差要求 2 的幂且 ≥2，得到 {n}")
    fast = butterfly_only_values(a)
    plain = butterfly_sweep(a)
    return np.max(np.abs(fast - plain), axis=1)[1:]


def butterfly_diagram(x: Any) -> dict[str, Any]:
    """把 :func:`butterfly_schedule` 作用到真实输入上，返回逐级快照、逐级残差与位反序谱。

    返回字典含 ``values``（``(levels+1, n)``，第 0 行输入、末行位反序输出）、``stages``
    （每级蝶形清单）、``partial``（``(levels, n)``：每级的「块内 DFT」部分变换）、
    ``residuals``（逐级残差）与 ``spectrum``（位反序置换后的 :math:`X_k`）。
    """
    a = np.asarray(x, dtype=complex).ravel()
    n = a.size
    if n < 2 or (n & (n - 1)):
        raise ValueError(f"蝶形图要求 2 的幂且 ≥2，得到 {n}")
    levels = int(round(np.log2(n)))
    stages = butterfly_schedule(n, a)
    values = butterfly_only_values(a)
    partial = np.zeros((levels, n), dtype=complex)
    for level in range(levels):
        block = n >> level
        partial[level] = values[level + 1] / float(block)
    scrambled = values[levels]
    spectrum = scrambled[bit_reverse_indices(n)]
    return {
        "n": n,
        "levels": levels,
        "values": values,
        "stages": stages,
        "partial": partial,
        "residuals": level_residuals(a),
        "scrambled": scrambled,
        "spectrum": spectrum,
    }



def dft_vs_fft_error(x: Any) -> float:
    """``max|FFT(x) − DFT(x)|``（自研 radix-2/Bluestein 与 :math:`O(N^2)` 定义式的逐点差）。"""
    a = np.asarray(x, dtype=complex).ravel()
    return float(np.max(np.abs(fft(a) - dft_direct(a)))) if a.size else 0.0


# --------------------------------------------------------------------------- #
# 频谱量
# --------------------------------------------------------------------------- #
def amplitude_spectrum(x: Any, *, dt: float = 1.0, one_sided: bool = True,
                       ref_length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """单边幅度谱 ``(|X_k| × 2/N)``（DC 与 Nyquist 不乘 2）与对应频率。

    这样**正弦的峰值就是它的幅度**；忘记那个 2 会把所有幅度系统性低估一半，
    忘记 DC/Nyquist 不乘 2 又会把它们高估一倍（两处都在 ``pitfalls`` 里）。

    ``ref_length`` 是**归一化长度**，默认取 ``len(x)``。补零后必须显式给出**原始长度**：
    FFT 的长度变成 ``N·pad``，若用补零后的长度归一化，峰值会整整齐齐地掉到
    ``A/pad``（实测 pad=2 时峰 0.5、pad=4 时 0.25）——每个峰都掉了 ``pad`` 倍，
    看起来仍然"像一个正常的谱"，只是幅度全错。
    """
    a = np.asarray(x, dtype=float).ravel()
    n = a.size
    if n == 0:
        raise ValueError("空序列没有频谱")
    norm = float(n if ref_length is None else int(ref_length))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"归一化长度必须为正，得到 {ref_length}")
    spec = rfft(a) if one_sided else fft(a.astype(complex))
    amp = np.abs(spec) * (2.0 / norm)
    if one_sided:
        amp[0] = np.abs(spec[0]) / norm
        if n % 2 == 0:
            amp[-1] = np.abs(spec[-1]) / norm
        freq = frequencies(n, dt=dt)[: n // 2 + 1]
    else:
        freq = frequencies(n, dt=dt)
        amp[0] = np.abs(spec[0]) / norm
    return freq, amp


def power_spectrum(x: Any, *, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """单边功率谱 :math:`|X_k|^2`（不做 :math:`1/N` 缩放，能量口径见 :func:`frequency_energy`）。"""
    freq, amp = amplitude_spectrum(x, dt=dt)
    return freq, amp * amp


def phase_spectrum(x: Any, *, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """单边相位谱（弧度，范围 :math:`(-\\pi,\\pi]`）。"""
    a = np.asarray(x, dtype=float).ravel()
    n = a.size
    if n == 0:
        raise ValueError("空序列没有频谱")
    spec = rfft(a)
    return frequencies(n, dt=dt)[: n // 2 + 1], np.angle(spec)


def db(values: Any, *, floor_db: float = -300.0, ref: float | None = None) -> np.ndarray:
    """转 dB：``20 log10(|v| / ref)``；0 与负值截到 ``floor_db``（对数轴上必须给地板）。"""
    v = np.abs(np.asarray(values, dtype=float))
    r = float(np.max(v)) if ref is None else float(ref)
    if r <= 0.0:
        return np.full(v.shape, float(floor_db))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 20.0 * np.log10(np.where(v > 0.0, v / r, np.nan))
    return np.where(np.isfinite(out), np.maximum(out, float(floor_db)), float(floor_db))


def dB(values: Any, **kwargs: Any) -> np.ndarray:
    """:func:`db` 的别名（教材里两种写法都常见）。"""
    return db(values, **kwargs)


def parabolic_peak(y: Any, i: int) -> tuple[float, float, float]:
    """抛物线插值定位峰值：返回 ``(偏移量, 插值后的值, 曲率)``。

    偏移量 :math:`\\delta = \\tfrac{1}{2}\\cdot\\tfrac{y_{i-1}-y_{i+1}}{y_{i-1}-2y_i+y_{i+1}}`
    落在 ``[-0.5, 0.5]``（栅格单位）；用 **dB 值** 插值比用幅度值更接近真实峰位
    （在主瓣顶部，幅度是二次型、dB 才近似抛物线）。
    """
    v = np.asarray(y, dtype=float).ravel()
    i = int(i)
    if i <= 0 or i >= v.size - 1:
        return 0.0, float(v[i]), 0.0
    y0, y1, y2 = float(v[i - 1]), float(v[i]), float(v[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return 0.0, y1, 0.0
    delta = 0.5 * (y0 - y2) / denom
    return float(delta), float(y1 - 0.25 * (y0 - y2) * delta), float(denom)


def peak_frequency(x: Any, *, dt: float = 1.0, refine: bool = True) -> tuple[float, int]:
    """最大谱线及其抛物线插值后的频率：返回 ``(f_peak, k_argmax)``。"""
    freq, amp = amplitude_spectrum(x, dt=dt)
    k = int(np.argmax(amp))
    if not refine:
        return float(freq[k]), k
    delta, _, _ = parabolic_peak(db(amp, ref=float(np.max(amp)) if np.max(amp) > 0 else 1.0), k)
    df = float(freq[1] - freq[0]) if freq.size > 1 else 0.0
    return float(freq[k] + delta * df), k


def main_lobe(w: Any, *, nfft: int | None = None, drop_db: float = 3.0,
              window_name: str = "rect") -> dict[str, float]:
    """窗的主瓣结构：第一零点、−3 dB 全宽、峰值旁瓣电平（都在**窗自身**的栅格 1/N 上量）。

    三个量分别这样量：

    * **第一零点**：窗谱（补零后的光滑曲线）中主瓣外的第一个局部极小点，用抛物线插值取到亚栅格精度；
    * **−3 dB 宽度**：峰值两侧首次跌破 −3 dB 的位置做线性插值（栅格不够密时直接读栅格会偏大）；
    * **峰值旁瓣**：第一零点之外的最大值（相对主瓣峰值，dB）。

    为什么用 :func:`window` 的**解析谱**而不是 ``|FFT|``：窗长只有 32–64 点时，
    ``|FFT|`` 的栅格太粗，−3 dB 宽度会被量化到整格（rect 报 14/32 ≈ 0.44 而不是 0.89），
    只有把 :math:`|W(f)|` 解析延拓出来才能与 Harris 表里的值对齐。
    """
    ww = np.asarray(w, dtype=float).ravel()
    n = ww.size
    if n < 4:
        raise ValueError(f"窗长必须 ≥4，得到 {n}")
    m = int(nfft) if nfft is not None else int(n)
    if m < 4 * n:
        m = 4 * n
    # 扫得够密才能看见第一零点：栅格太粗时 sinc 在 fN=1 的极小会被两个点「夹过去」，
    # 判据 ``amp[i] <= amp[i±1]`` 一个都不成立，first_null 静默变成 NaN。
    # 扫到 1.5：只扫到 1.0 时第一零点正好落在**末点**上，没有右邻点可比较，
    # 判据失效、first_null 静默变成 NaN（实测 rect 就是这样）。
    freqs = np.arange(4097, dtype=float) / 1024.0  # 归一化频率 f·N ∈ [0, 4]
    if window_name not in WINDOW_NAMES:
        raise ValueError(f"未知窗名 {window_name!r}（可用：{', '.join(WINDOW_NAMES)}）")
    amp = _window_spectrum_analytic(window_name, freqs)
    amp = amp / max(float(amp[0]), 1e-300)
    dbv = 20.0 * np.log10(np.maximum(amp, 1e-300))
    # 第一零点（主瓣外的第一个局部极小）
    first_null = float("nan")
    for i in range(2, freqs.size - 1):  # 从 2 起：i=1 时邻点含 f=0（主瓣峰），会立刻误判
        if amp[i] <= amp[i - 1] and amp[i] <= amp[i + 1]:
            d, _, _ = parabolic_peak(amp, i)
            first_null = float(freqs[i] + d * (freqs[1] - freqs[0]))
            break
    # −3 dB 半宽（线性插值）
    half = float("nan")
    for i in range(1, freqs.size):
        if dbv[i] <= -abs(drop_db):
            f0, f1 = dbv[i - 1], dbv[i]
            t = 0.0 if f1 == f0 else (f0 + abs(drop_db)) / (f0 - f1)
            half = float(freqs[i - 1] + t * (freqs[1] - freqs[0]))
            break
    ps = peak_sidelobe_analytic(window_name)
    return {
        "first_null": first_null,
        "first_null_bins": first_null,
        "half_width": half,
        "width_bins": 2.0 * half,
        "peak_sidelobe_db": ps,
        "nfft": float(m),
    }


def peak_sidelobe_analytic(window_name: str, *, span: float = 20.0, n_scan: int = 60000) -> float:
    """窗的**解析**峰值旁瓣电平（dB，相对主瓣峰值）。

    做法：在 :math:`fN \\in (0, \\text{span}]` 上密扫 :math:`|W(f)|`，去掉主瓣（到第一个
    局部极小为止）后取最大值。与 Harris 表值的一致性由 ``tests/2.5`` 实测（容差 0.6 dB）。
    """
    key = str(window_name).lower()
    if key not in WINDOW_NAMES:
        raise ValueError(f"未知窗名 {key!r}（可用：{', '.join(WINDOW_NAMES)}）")
    f = np.linspace(1.0 / n_scan, float(span), int(n_scan))
    amp = _window_spectrum_analytic(key, f)
    amp = amp / max(float(_window_spectrum_analytic(key, np.zeros(1))[0]), 1e-300)
    null = main_lobe_scan_first_null(f, amp)
    tail = amp[f > null]
    if tail.size == 0:
        return float("nan")
    return float(20.0 * np.log10(max(float(np.max(tail)), 1e-300)))


def window_spectrum(name: str, f_n: Any) -> np.ndarray:
    r"""窗的**连续**频谱 :math:`|W(f)|`（归一化到峰值 1），自变量是 ``f*N``。

    四点窗（rect/hann/hamming/blackman）的解析式，用 Dirichlet 核的 ``sinc`` 写成。
    它是画旁瓣理论线、以及把 ``|FFT|`` 的实测值对到一条光滑曲线上的基准：
    栅格上的 ``|FFT(w)|`` 只会**采样**这条曲线，采样点永远恰好落在零点之外
    （所以旁瓣读数会因为"没采到谷底"而偏高几个 dB）。
    """
    amp = _window_spectrum_analytic(str(name), np.asarray(f_n, dtype=float))
    peak = float(np.max(_window_spectrum_analytic(str(name), np.zeros(1))))
    return amp / max(peak, 1e-300)


def _window_spectrum_analytic(name: str, f: np.ndarray) -> np.ndarray:
    """连续频谱 :math:`|W(f)|`（归一化频率 :math:`fN`；Dirichlet 核写成 ``sinc``）。"""
    x = np.asarray(f, dtype=float)
    if name == "rect":
        return np.abs(np.sinc(x))
    if name == "hann":
        return np.abs(0.5 * np.sinc(x) + 0.25 * np.sinc(x - 1.0) + 0.25 * np.sinc(x + 1.0))
    if name == "hamming":
        return np.abs(0.54 * np.sinc(x) + 0.23 * np.sinc(x - 1.0) + 0.23 * np.sinc(x + 1.0))
    return np.abs(0.42 * np.sinc(x) + 0.25 * np.sinc(x - 1.0) + 0.25 * np.sinc(x + 1.0)
                  + 0.04 * np.sinc(x - 2.0) + 0.04 * np.sinc(x + 2.0))


def main_lobe_scan_first_null(f: np.ndarray, amp: np.ndarray) -> float:
    """扫出主瓣外的第一个局部极小位置（归一化频率），供旁瓣扫描用。"""
    for i in range(1, f.size - 1):
        if amp[i] <= amp[i - 1] and amp[i] <= amp[i + 1]:
            return float(f[i])
    return float(f[-1])


def n_local_maxima(y: Any, *, rel_height: float = 0.1) -> int:
    """幅度序列里高度 ≥ ``rel_height×max`` 的局部极大值个数（峰计数）。"""
    v = np.asarray(y, dtype=float).ravel()
    if v.size < 3:
        return int(v.size)
    peak = float(np.max(v))
    if peak <= 0.0:
        return 0
    core = v[1:-1]
    mask = (core >= v[:-2]) & (core >= v[2:]) & (core >= rel_height * peak)
    return int(np.count_nonzero(mask))


def separable(x: Any, *, dt: float = 1.0, rel_height: float = 0.1, min_prominence: float = 0.0,
              valley_ratio: float = 0.7, w: Any = None) -> bool:
    """两个近邻谱峰能否分开（Rayleigh 型判据的数值版本）。

    判据三条**同时**成立才算分得开：

    1. 至少两个局部极大，且都高于 ``rel_height × 峰值``（挡掉主瓣肩部被舍入噪声拱出的伪峰）；
    2. 相邻两峰之间确有谷；
    3. 谷底不高于矮峰的 ``valley_ratio`` 倍（默认 0.7）。

    第 3 条不是装饰：主瓣顶部本来就"又平又毛"，两个 0.94 与 0.92 高的伪峰中间只掉 1%
    也会被"两峰一谷"数出来——那正是把「其实早就分开了」和「其实没分开」搞反的地方。
    0.7 这个数是拿 :func:`min_separable_separation` 标定的：它让四个窗在 1.0/T 处
    第一次判真（与 Rayleigh 判据一致），换成分辨率更高的判据会得到 2.0/T 这种偏保守的值。

    ``min_prominence`` 保留为兼容口径：设成非零时额外要求谷比矮峰低这么多（等价于
    更小的 ``valley_ratio``）。
    """
    a = np.asarray(x, dtype=float).ravel()
    if w is not None:
        a = a * np.asarray(w, dtype=float).ravel()
    freq, amp = amplitude_spectrum(a, dt=dt)
    del freq
    peak = float(np.max(amp)) if amp.size else 0.0
    if peak <= 0.0 or amp.size < 5:
        return False
    idx = [i for i in range(1, amp.size - 1)
           if amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1] and amp[i] >= rel_height * peak]
    if len(idx) < 2:
        return False
    thr = float(valley_ratio)
    if min_prominence > 0.0:
        thr = min(thr, 1.0 - float(min_prominence))
    for a_i, b_i in zip(idx[:-1], idx[1:]):
        valley = float(np.min(amp[a_i : b_i + 1]))
        if valley <= thr * min(amp[a_i], amp[b_i]):
            return True
    return False


def min_separable_separation(t_max: float = 1.0, *, dt: float | None = None, n: int = 256,
                             window_name: str = "rect", rel_height: float = 0.1,
                             valley_ratio: float = 0.7, refine: bool = False,
                             max_sep: float = 10.0) -> float:
    """实测「两个等幅近邻音的最小可分辨间隔」，单位是 :math:`1/T`（:math:`T` = 观测时长）。

    两个音固定在中频 :math:`f_0 = 3/T`，只动间隔，**按整数格扫**而不是二分：
    谓词 :func:`separable` 对间隔既不单调也不连续——间隔落在半格/四分之一格上时
    （如 :math:`5.4/T`）两音都偏离栅格、被"扇"进同一个主瓣，谓词会翻假，
    二分会收敛到一个假的穿越点（实测 1.73/T，看着还挺像 1/T——这就是它会骗人的地方）。

    扫描带来一个**真实且必须写出来的结论**：扫到 1.0/T 时，四个窗都判"能分开"。
    这不是"分辨率与窗无关"，而是判据在这一格上还太宽松——Hann 的真正代价出现在
    :math:`f_2 = f_1 + 1.5/T` 这类两音都落在半格上的位置（实测谷/峰 = 0.479，
    按 0.7 的判据直接判"分不开"，矩形窗同一位置是 0.251）。换句话说：
    **整数格上量出来的分辨率会把窗的代价藏起来**，这也是本章 pitfalls 里的一条。
    """
    t_obs = float(t_max)
    if t_obs <= 0.0:
        raise ValueError(f"观测时长必须为正，得到 {t_max}")
    dt_eff = float(dt) if dt is not None else t_obs / float(n)
    n_pt = int(n)
    f_mid = 3.0 / t_obs

    def predicate(sep: float) -> bool:
        half = 0.5 * float(sep) / t_obs
        x = two_tone(1.0, f_mid - half, 1.0, f_mid + half, t_max=t_obs, n=n_pt)
        if window_name != "rect":
            x = x * window(window_name, x.size)
        return separable(x, dt=dt_eff, rel_height=rel_height, valley_ratio=valley_ratio)

    k_max = max(int(max_sep), 2)
    lo = float(k_max)
    for k in range(1, k_max + 1):
        if predicate(float(k)):
            lo = float(k)
            break
    if not bool(refine) or lo <= 1.0:
        return lo
    hi = lo
    lo = hi - 1.0
    for _ in range(20):  # 在 (lo, hi] 内二分，保持 predicate(hi) 为真
        mid = 0.5 * (lo + hi)
        if predicate(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


def rayleigh_resolution(t_obs: float, *, kappa: float = 1.0) -> float:
    """Rayleigh 判据 :math:`\\Delta f_{\\min} = \\kappa/T`（矩形窗 :math:`\\kappa = 1`）。"""
    t_obs = float(t_obs)
    if not np.isfinite(t_obs) or t_obs <= 0.0:
        raise ValueError(f"观测时长必须为正的有限数，得到 {t_obs}")
    return float(kappa) / t_obs


def frequency_resolution(n: int, *, dt: float = 1.0) -> float:
    """**栅格间距** :math:`\\Delta f = 1/(N\\Delta t)`：它随补零变小，但这不是分辨率。"""
    if n < 1:
        raise ValueError(f"点数必须为正，得到 {n}")
    return 1.0 / (int(n) * float(dt))


# --------------------------------------------------------------------------- #
# 窗函数
# --------------------------------------------------------------------------- #
#: 支持的窗名（顺序即「旁瓣抑制能力」递增的顺序）
WINDOW_NAMES: tuple[str, ...] = ("rect", "hann", "hamming", "blackman")

#: 各窗的**峰值旁瓣电平**理论值（dB，相对主瓣峰值；来源：Harris 1978 表 1）
WINDOW_SIDELOBE_DB: dict[str, float] = {
    "rect": -13.26,
    "hann": -31.47,
    "hamming": -42.68,
    "blackman": -58.11,
}

#: 各窗的 −3 dB 主瓣宽度（单位：栅格 Δf = 1/(NΔt)；来源：Harris 1978 表 1）
WINDOW_MAINLOBE_BINS: dict[str, float] = {
    "rect": 0.89,
    "hann": 1.44,
    "hamming": 1.30,
    "blackman": 1.68,
}

#: 各窗旁瓣渐近包络的幂次（:math:`|W(f)| \\sim f^{-p}`，``-p`` 即双对数斜率）
WINDOW_ENVELOPE_POWER: dict[str, float] = {
    "rect": 1.0,
    "hann": 3.0,
    "hamming": 1.0,
    "blackman": 3.0,
}


def window(name: str, n: int) -> np.ndarray:
    """长度 ``n`` 的窗（对称型，系数按教材通则）：

    * ``rect``：:math:`w \\equiv 1`；
    * ``hann``：:math:`0.5 - 0.5\\cos(2\\pi m/(N-1))`；
    * ``hamming``：:math:`0.54 - 0.46\\cos(2\\pi m/(N-1))`；
    * ``blackman``：:math:`0.42 - 0.5\\cos(2\\pi m/(N-1)) + 0.08\\cos(4\\pi m/(N-1))`。

    指数 :math:`\\alpha` 只对广义余弦窗生效（如 :math:`\\cos^2` 窗 = Hann 的另一种写法）。
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"窗长必须 ≥2，得到 {n}")
    name = str(name).lower()
    if name not in WINDOW_NAMES:
        raise ValueError(f"未知窗名 {name!r}（可用：{', '.join(WINDOW_NAMES)}）")
    m = np.arange(n, dtype=float)
    arg = 2.0 * np.pi * m / float(n - 1)
    if name == "rect":
        return np.ones(n, dtype=float)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(arg)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(arg)
    return 0.42 - 0.5 * np.cos(arg) + 0.08 * np.cos(2.0 * arg)


def window_coherent_gain(w: Any) -> float:
    """相干增益 :math:`CG = \\frac{1}{N}\\sum w_m`（幅度读数的系统偏差）。"""
    a = np.asarray(w, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("空窗没有增益")
    return float(np.sum(a)) / float(a.size)


def window_energy(w: Any) -> float:
    """窗能量 :math:`\\sum w_m^2`（噪声功率口径的等价噪声带宽用它）。"""
    a = np.asarray(w, dtype=float).ravel()
    return float(np.sum(a * a))


def coherent_sum(w: Any, *, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """移位窗之和 :math:`\\sum_p w_{m-pN/2}`（单位：hop = N/2 的 50% 重叠）。

    常数重叠相加（COLA）成立的窗，这条曲线在拼接区间内是常数：
    Hamming 与 Hann 满足 :math:`\\sum_p w_{m-pN/2} = \\text{const}`，
    矩形窗不满足（边界处掉一半）——这就是「窗不只是为了压旁瓣」的那一面。
    """
    a = np.asarray(w, dtype=float).ravel()
    n = a.size
    if n < 4 or n % 2:
        raise ValueError(f"窗长必须是 ≥4 的偶数，得到 {n}")
    hop = n // 2
    total = np.zeros(3 * n, dtype=float)
    for p in range(3):
        total[p * hop : p * hop + n] += a
    x = np.arange(3 * n, dtype=float) * float(dt)
    return x, total


def peak_sidelobe_level(name: str) -> dict[str, float]:
    """窗的峰值旁瓣电平与主瓣指标（一条龙，供帧与断言共用）。

    返回 ``peak_db``、``mainlobe_bins``（−3 dB 全宽，单位栅格 1/N）、
    ``first_null_bins``、``coherent_gain``、``energy``。
    """
    key = str(name).lower()
    if key not in WINDOW_NAMES:
        raise ValueError(f"未知窗名 {key!r}（可用：{', '.join(WINDOW_NAMES)}）")
    ml = main_lobe(window(key, 64), window_name=key)
    w = window(key, 64)
    return {
        "peak_db": float(ml["peak_sidelobe_db"]),
        "mainlobe_bins": float(ml["width_bins"]),
        "first_null_bins": float(ml["first_null_bins"]),
        "coherent_gain": window_coherent_gain(w),
        "energy": window_energy(w),
    }


def sidelobe_envelope(name: str, freq: Any, *, f_ref: float) -> tuple[np.ndarray, float]:
    """窗旁瓣的**渐近包络** :math:`|W(f)| \\approx (f/f_{\\text{ref}})^{-p}`（dB 域是直线）。

    返回 ``(包络值, 幂次 p)``；``p`` 取自 :data:`WINDOW_ENVELOPE_POWER`
    （矩形/汉明 :math:`p=1`，Hann/Blackman :math:`p=3`）——汉明的旁瓣不随 :math:`f`
    继续下降，它是「压低了但不再降」的那一类，这也是为什么它与 Hann 不能互换着说。
    """
    key = str(name).lower()
    if key not in WINDOW_ENVELOPE_POWER:
        raise ValueError(f"未知窗名 {key!r}（可用：{', '.join(WINDOW_NAMES)}）")
    p = float(WINDOW_ENVELOPE_POWER[key])
    f = np.asarray(freq, dtype=float)
    ref = float(f_ref)
    if ref <= 0.0:
        raise ValueError(f"参考频率必须为正，得到 {ref}")
    with np.errstate(divide="ignore", invalid="ignore"):
        env = np.where(f > 0.0, (np.abs(f) / ref) ** (-p), np.nan)
    return env, p


# --------------------------------------------------------------------------- #
# 卷积、能量与谱运算
# --------------------------------------------------------------------------- #
def circular_convolution(a: Any, b: Any) -> np.ndarray:
    """长度 :math:`N` 的**循环**卷积 :math:`(a \\circledast b)_m = \\sum_p a_p b_{(m-p)\\bmod N}`。

    自己写（不用任何库的卷积）：外层对 :math:`p` 循环，内层用 ``np.roll`` 的切片等价物
    ——这里用显式的模索引矩阵乘法实现，:math:`O(N^2)`，恰好也是「循环」二字的字面翻译。
    """
    x = np.asarray(a, dtype=complex).ravel()
    y = np.asarray(b, dtype=complex).ravel()
    n = x.size
    if n == 0 or y.size != n:
        raise ValueError(f"循环卷积要求等长非空序列，得到 {x.size} 与 {y.size}")
    idx = (np.arange(n, dtype=np.int64)[:, None] - np.arange(n, dtype=np.int64)[None, :]) % n
    return np.sum(x[None, :] * y[idx], axis=1)


def linear_convolution(a: Any, b: Any) -> np.ndarray:
    """线性卷积 :math:`(a * b)_m = \\sum_p a_p b_{m-p}`（长度 :math:`N+M-1`）。"""
    x = np.asarray(a, dtype=complex).ravel()
    y = np.asarray(b, dtype=complex).ravel()
    n, m = x.size, y.size
    if n == 0 or m == 0:
        raise ValueError("线性卷积要求非空序列")
    out = np.zeros(n + m - 1, dtype=complex)
    for p in range(n):
        out[p : p + m] += x[p] * y
    return out


def frequency_product(a: Any, b: Any) -> np.ndarray:
    """频域逐点相乘 :math:`A_k B_k`（两个等长序列的 DFT 之积）。"""
    x = np.asarray(a, dtype=complex).ravel()
    y = np.asarray(b, dtype=complex).ravel()
    if x.size != y.size:
        raise ValueError(f"频域相乘要求等长，得到 {x.size} 与 {y.size}")
    return fft(x) * fft(y)


def fft_convolve(a: Any, b: Any, *, mode: str = "circular") -> np.ndarray:
    """用「正变换 → 逐点相乘 → 逆变换」做卷积（**不许**调用任何库的卷积）。

    ``mode="circular"``：长度取 ``max(N, M)`` 的循环卷积（会卷绕）；
    ``mode="linear"``：补零到 :math:`\\ge N+M-1`（2 的幂），结果的线性卷积部分。
    """
    x = np.asarray(a, dtype=complex).ravel()
    y = np.asarray(b, dtype=complex).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("卷积要求非空序列")
    if mode == "circular":
        n = max(x.size, y.size)
        return ifft(frequency_product(zero_pad(x, n), zero_pad(y, n)))
    if mode != "linear":
        raise ValueError(f"mode 只能是 'circular' 或 'linear'，得到 {mode!r}")
    need = x.size + y.size - 1
    n = next_power_of_two(need)
    out = ifft(frequency_product(zero_pad(x, n), zero_pad(y, n)))
    return out[:need]


def convolution_theorem_error(a: Any, b: Any) -> float:
    """循环卷积定理的逐点残差 ``max|a ⊛ b − IDFT(DFT(a)·DFT(b))|``。"""
    x = np.asarray(a, dtype=complex).ravel()
    y = np.asarray(b, dtype=complex).ravel()
    n = max(x.size, y.size)
    ref = circular_convolution(zero_pad(x, n), zero_pad(y, n))
    return float(np.max(np.abs(ref - fft_convolve(x, y, mode="circular"))))


def wrap_around_error(a: Any, b: Any) -> dict[str, float]:
    """循环卷积与线性卷积的差异：``max`` 绝对误差、出现在哪些位置、相对规模。

    当 :math:`N < N_a+N_b-1` 时，线性卷积的尾部会**卷绕**回来叠加到前面几项上；
    返回 ``first_index``（第一处差异）与 ``tail_energy_ratio``，用于把「卷绕伪装成滤波结果」
    这件事量化——只看前几项会觉得"滤得挺好"，差异全藏在后面。
    """
    x = np.asarray(a, dtype=complex).ravel()
    y = np.asarray(b, dtype=complex).ravel()
    lin = linear_convolution(x, y)
    n = max(x.size, y.size)
    cyc = fft_convolve(x, y, mode="circular")
    k = min(lin.size, cyc.size)
    diff = np.abs(cyc[:k] - lin[:k])
    scale = max(float(np.max(np.abs(lin))), 1e-300)
    nz = np.flatnonzero(diff > 1e-12 * scale)
    tail_energy = float(np.sum(np.abs(lin[n:]) ** 2)) if lin.size > n else 0.0
    return {
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "rel_max": float(np.max(diff) / scale) if diff.size else 0.0,
        "first_index": float(nz[0]) if nz.size else -1.0,
        "n_diff": float(nz.size),
        "tail_energy_ratio": float(tail_energy / max(float(np.sum(np.abs(lin) ** 2)), 1e-300)),
    }


def time_energy(x: Any, *, dt: float = 1.0) -> float:
    """时域能量 :math:`\\Delta t\\sum_n |x_n|^2`（:math:`\\Delta t` 把离散和变成积分近似）。"""
    a = np.asarray(x, dtype=complex).ravel()
    return float(np.sum(np.abs(a) ** 2) * float(dt))


def frequency_energy(X: Any, *, dt: float = 1.0, one_sided: bool = False, n: int | None = None) -> float:
    """频域能量（Parseval 的另一边）。

    全谱口径：:math:`\\Delta f \\sum_k |X_k|^2`，其中 :math:`\\Delta f = 1/(N\\Delta t)`
    是频率栅格间距，它等于 :math:`\\Delta t\\sum_n|x_n|^2`。

    ``one_sided=True`` 时 ``X`` 是 :func:`rfft` 的单边谱（``N/2+1`` 根），权重应为
    「内部谱线 ×2、DC 与（N 为偶数时的）Nyquist ×1」。两处容易错**且都错得很隐蔽**：

    * 若用单边谱的**长度**去算 :math:`\\Delta f`，间距会偏大近一倍——实测 ``N = 101``
      的能量只剩 **96.5%**，看起来像「Parseval 不成立」；
    * 奇数 ``N`` 没有 Nyquist 项，单边谱长度是 ``(N+1)/2``，此时**无法**从长度反推 N，
      必须显式给 ``n``（否则抛错，而不是悄悄用一个错的 ``N``）。
    """
    a = np.asarray(X, dtype=complex).ravel()
    if a.size == 0:
        raise ValueError("空谱没有能量")
    if one_sided:
        if int(n) if n is not None else 0:
            n_full = int(n)  # type: ignore[arg-type]
        elif a.size >= 2:
            n_full = 2 * (a.size - 1)
            if a[a.size // 2].imag != 0.0 or a[-1].imag != 0.0:
                # Nyquist 项必须为实数：虚部非零说明这不像是偶数长度的单边谱
                n_full = 2 * a.size - 1
        else:
            raise ValueError("单边谱至少要有 2 根谱线")
        if n_full % 2 == 0 and 2 * (a.size - 1) != n_full:
            raise ValueError(f"单边谱长度 {a.size} 与 n={n_full} 不匹配（应为 {n_full // 2 + 1}）")
    else:
        n_full = int(n) if n is not None else a.size
    df = 1.0 / (n_full * float(dt))
    if one_sided:
        power = 2.0 * float(np.sum(np.abs(a) ** 2)) - float(np.abs(a[0]) ** 2)
        if n_full % 2 == 0:
            power -= float(np.abs(a[-1]) ** 2)
    else:
        power = float(np.sum(np.abs(a) ** 2))
    return power * df


def parseval_relative_error(x: Any, *, dt: float = 1.0, one_sided: bool = False) -> float:
    """Parseval 相对误差 :math:`|E_{\\text{freq}} - E_{\\text{time}}| / E_{\\text{time}}`。"""
    a = np.asarray(x, dtype=complex).ravel()
    spec = rfft(np.real(a)) if one_sided else fft(a)
    e_t = time_energy(a, dt=dt)
    e_f = frequency_energy(spec, dt=dt, one_sided=one_sided, n=a.size)
    return float(abs(e_f - e_t) / max(e_t, 1e-300))


def circular_derivative(x: Any) -> np.ndarray:
    """循环一阶差分 :math:`(Dx)_n = x_{n+1} - x_n`（下标模 N）：它的 DFT 是 :math:`(e^{2\\pi ik/N}-1)X_k`。"""
    a = np.asarray(x, dtype=complex).ravel()
    if a.size == 0:
        raise ValueError("空序列没有差分")
    return np.roll(a, -1) - a


# --------------------------------------------------------------------------- #
# 信号构造、混叠与单频估计
# --------------------------------------------------------------------------- #
def sinusoid_sum(amps: Sequence[float], freqs: Sequence[float], phases: Sequence[float] | None = None,
                 *, fs: float = 1.0, n: int = 256, t0: float = 0.0) -> np.ndarray:
    """多音信号 :math:`x(t) = \\sum_p A_p\\cos(2\\pi f_p t + \\varphi_p)`，``t = t0 + m/fs``。

    **不做混叠检查**：``f > fs/2`` 的音会被采样过程折叠（这正是要演示的东西），
    要知道它折到哪去请调 :func:`alias_frequency`。
    """
    a = np.asarray(list(amps), dtype=float)
    f = np.asarray(list(freqs), dtype=float)
    if a.size != f.size or a.size == 0:
        raise ValueError(f"幅度与频率个数必须一致且非空，得到 {a.size} 与 {f.size}")
    p = np.zeros(a.size, dtype=float) if phases is None else np.asarray(list(phases), dtype=float)
    if p.size != a.size:
        raise ValueError(f"相位个数必须与幅度一致，得到 {p.size} 与 {a.size}")
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"采样率必须为正的有限数，得到 {fs}")
    t = float(t0) + np.arange(int(n), dtype=float) / fs
    out = np.zeros(int(n), dtype=float)
    for ai, fi, pi in zip(a, f, p):
        out += ai * np.cos(2.0 * np.pi * fi * t + pi)
    return out


def two_tone(a1: float, f1: float, a2: float, f2: float, *, t_max: float = 1.0,
             n: int = 512, pad: int = 1) -> np.ndarray:
    """两个等/不等幅余弦的和，采样 ``n`` 点、观测时长 ``t_max``；``pad`` 只补零不改采样点。"""
    fs = int(n) / float(t_max)
    x = sinusoid_sum([a1, a2], [f1, f2], fs=fs, n=int(n))
    return zero_pad(x, int(n) * int(pad)).real


def square_wave(n: int, *, n_harmonics: int = 0, duty: float = 0.5) -> np.ndarray:
    """周期方波（默认 ±1、占空比 1/2）的采样；``n_harmonics>0`` 时返回其截断 Fourier 级数。

    跳变处的 Gibbs 过冲随项数**不消失**（约 8.95% 的跳变量），
    但过冲的**宽度**以 :math:`O(1/N)` 收缩——两件事必须分开说。
    """
    n = int(n)
    if n < 8:
        raise ValueError(f"点数必须 ≥8，得到 {n}")
    m = np.arange(n, dtype=float)
    if n_harmonics and int(n_harmonics) > 0:
        out = np.zeros(n, dtype=float)
        for h in range(1, int(n_harmonics) + 1, 2):
            out += (4.0 / (np.pi * h)) * np.sin(2.0 * np.pi * h * m / float(n))
        return out
    phase = np.mod(m / float(n), 1.0)
    return np.where(phase < float(duty), 1.0, -1.0)


def shift_sample(x: Any, p: int) -> np.ndarray:
    """循环移位 :math:`x_{n-p}`（``p`` 为正表示向右移），用于相位与位移定理。"""
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("空序列不能移位")
    return np.roll(a, int(p))


def alias_index(k: int, n: int) -> int:
    """频率栅格下标 ``k`` 的混叠代表元（落在 ``0..N//2``，即 ``[0, fs/2]``）。"""
    n = int(n)
    if n < 1:
        raise ValueError(f"点数必须为正，得到 {n}")
    kk = int(k) % n
    return int(min(kk, n - kk))


def alias_frequency(f: float, fs: float) -> float:
    """把频率 ``f`` 折叠到 ``[0, fs/2]``：采样率 ``fs`` 下它**看起来**是哪个频率。"""
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"采样率必须为正的有限数，得到 {fs}")
    f = float(f) % fs
    return float(min(f, fs - f))


def alias_candidates(f_obs: float, fs: float, *, f_max: float) -> np.ndarray:
    """所有能伪装成 ``f_obs`` 的真实频率（降序）：:math:`m f_s \\pm f_{\\text{obs}} \\le f_{\\max}`。"""
    fs = float(fs)
    f_obs = float(f_obs)
    f_max = float(f_max)
    if fs <= 0.0 or f_max < 0.0:
        raise ValueError("采样率必须为正、频率上界必须非负")
    out: list[float] = []
    m = 0
    while True:
        base = m * fs
        for cand in (base + f_obs, base - f_obs if m > 0 else -1.0):
            if 0.0 <= cand <= f_max:
                out.append(float(cand))
        if base - f_obs > f_max:
            break
        m += 1
        if m > 10000:  # pragma: no cover - 防御性上限
            break
    uniq = sorted({round(v, 12) for v in out}, reverse=True)
    return np.asarray(uniq, dtype=float)


def goertzel(x: Any, k: int, n: int | None = None) -> complex:
    """Goertzel 算法：只算第 ``k`` 根谱线 :math:`X_k`（:math:`O(N)`，单频检测用）。

    递推 :math:`s_m = x_m + 2\\cos(2\\pi k/N)\\,s_{m-1} - s_{m-2}`，最后
    :math:`X_k = s_{N-1} - e^{-2\\pi i k/N} s_{N-2}`。
    """
    a = np.asarray(x, dtype=float).ravel()
    m = a.size if n is None else int(n)
    if m < 1 or m > a.size:
        raise ValueError(f"点数必须满足 1 ≤ n ≤ 序列长度，得到 n={m}, 长度={a.size}")
    w = 2.0 * np.pi * int(k) / float(m)
    coeff = 2.0 * np.cos(w)
    s1 = 0.0
    s2 = 0.0
    for i in range(m):
        s0 = a[i] + coeff * s1 - s2
        s2 = s1
        s1 = s0
    # 收尾：X_k = s_{N-1}·e^{+iω} − s_{N-2}，ω = 2πk/N。
    # 共轭方向别猜：``s1 − e^{±iω}s2`` 与 ``s1·e^{−iω} − s2`` 这三个候选
    # 在**冲激输入**与 **k=0 / k=N/2** 上都给出正确答案，只有拿随机信号 + 一般 k
    # 才分得开（实测随机信号 k=1 时三者的差别是实部/虚部的符号）。见 pitfalls。
    return complex(s1 * np.exp(1j * w) - s2)
