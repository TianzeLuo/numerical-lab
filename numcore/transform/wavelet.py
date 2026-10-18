r"""2.6 章的数值内核：小波与多分辨分析（multiresolution analysis, MRA）。

数学对象
--------
设 :math:`\varphi` 是正交尺度函数、:math:`\psi` 是相伴的小波，二尺度方程给出

.. math::

    \varphi(t)=\sqrt2\sum_j h_j\,\varphi(2t-j),\qquad
    \psi(t)=\sqrt2\sum_j g_j\,\varphi(2t-j),\qquad g_j=(-1)^j h_{L-1-j},

其中 :math:`h` 是长度 :math:`L=2N` 的**尺度滤波器**。本模块在离散层实现的对象是：

* :func:`dwt` / :func:`idwt` —— 有限长信号的多层离散小波变换与重构。
  分析式 :math:`a_k=\sum_j h_j x[2k+j]`、:math:`d_k=\sum_j g_j x[2k+j]`；
  合成式 :math:`x[2k+j]\mathrel{+}= h_j a_k+g_j d_k`（同一组 :math:`h,g`——
  这是本模块的下标约定下的伴随对，见 :func:`dwt_step` 的说明）。
* :func:`swt` / :func:`iswt` —— 平稳（非抽取）小波变换，冗余因子 :math:`J+1`，
  紧框架界 :math:`J+1`。
* :func:`mra_bands` —— 多分辨分析的**逐层重构**：把 :math:`x` 拆成
  :math:`x=x_J+\sum_{j=1}^{J}w_j`，:math:`w_j` 是第 :math:`j` 层细节单独重构出来的分量。
* :func:`vanishing_moment_probe` —— 消失矩阶数的**双向**证据：:math:`N` 阶消失矩
  把 :math:`\deg p<N` 的多项式在**内部**系数上抹平到 :math:`u` 量级，
  而 :math:`\deg p=N` 不会。
* 阈值算子与去噪：:func:`soft_threshold` / :func:`hard_threshold`、
  :func:`universal_threshold`、:func:`estimate_sigma_mad`、:func:`bias_variance_curve`。

边界延拓（本章最容易偷偷骗人的地方）
------------------------------------
三种模式 :data:`BOUNDARY_MODES` 全部实现为同一句话：

    先按规则把 :math:`x` **延拓**成 :math:`x_{\text{ext}}`（长度 :math:`n_{\text{ext}}`
    是 :math:`2^{J}` 的倍数），再做**严格周期**的正交变换，最后把结果裁回原长度。

因为周期变换在 :math:`\mathbb R^{n_{\text{ext}}}` 上是正交的（:func:`dwt_matrix`
的对拍与 :func:`energy_ratio` 的实测都在钉这一点），三种模式的**完美重构都是精确成立的**
（到舍入级）；差别在于「系数是什么」而不是「能不能还原」：

* ``periodic``：不延拓，:math:`n_{\text{ext}}=n`，分析与合成都是正交变换
  —— 能量比**恰好** :math:`1`；
* ``zero``：两侧各补 :math:`L-1` 个零；延拓算子 :math:`E` 满足 :math:`E^{\mathsf T}E=I`
  （零不贡献能量），所以能量比仍是 :math:`1`，但 :math:`n_{\text{ext}}>n`，表示是冗余的；
* ``symmetric``：半样本对称反射；反射会**复制**样本，:math:`E^{\mathsf T}E\neq I`
  —— 能量比实测 :math:`>1`（db4、:math:`n=256`、:math:`J=4` 时约 :math:`1.06`）。
  这就是「能量守恒要写清是哪个能量」在代码里的样子。

反过来，**不做延拓**（一点余量都不留）时，边界处的系数会被环绕项污染；影响范围随层数
按 :math:`2^{j}` 增长（:func:`boundary_pollution` 量的是这个）。

约定
----
* 长度：``periodic`` 要求 :math:`2^{J}\mid n`；其余两种模式对任意 :math:`n\ge L` 都成立；
* 下标：系数 :math:`a_k` 的支撑在原始样本上是 :math:`[2k,\,2k+L-1]`（延拓坐标）；
* 纯函数、无 IO、无随机源（随机数一律走 :func:`numcore.utils.rng`，只在
  :func:`bias_variance_curve` 里按显式 ``seed`` 取用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from numcore.transform.fft import fft
from numcore.utils import rng

__all__ = [
    # 滤波器
    "WAVELET_NAMES",
    "SCALING_TABLE",
    "scaling_filter",
    "wavelet_filter",
    "filter_length",
    "vanishing_moments",
    "filter_response",
    "orthonormality_defect",
    "vanishing_moment_defect",
    "perfect_reconstruction_defect",
    # 边界
    "BOUNDARY_MODES",
    "guard_length",
    "extension_length",
    "extend_signal",
    "reflect_index",
    # 变换
    "WaveletDecomposition",
    "dwt_step",
    "idwt_step",
    "dwt",
    "idwt",
    "dwt_matrix",
    "orthogonality_defect",
    "truncate_critical",
    "reconstruction_error",
    "coefficient_lengths",
    "coefficient_energy",
    "energy_ratio",
    "level_energies",
    "mra_bands",
    "mra_partial",
    "boundary_pollution",
    # 消失矩
    "TEST_SIGNAL_NAMES",
    "test_signal",
    "polynomial_signal",
    "interior_slice",
    "detail_scale_ratio",
    "vanishing_moment_probe",
    # 平稳小波
    "StationaryDecomposition",
    "swt",
    "iswt",
    "scalogram",
    "scale_time_boxes",
    "translation_change",
    "swt_frame_symbol",
    # 阈值与去噪
    "soft_threshold",
    "hard_threshold",
    "threshold_decomposition",
    "denoise",
    "universal_threshold",
    "estimate_sigma_mad",
    "bias_variance_curve",
    "denoise_mse",
    "optimal_threshold",
    "snr_db",
    # 固定分辨率对照
    "stft_magnitude",
]

#: 支持的族：``haar``（= db1）与 db2..db8。滤波器的支撑长度 :math:`L=2N`。
WAVELET_NAMES: tuple[str, ...] = ("haar", "db2", "db3", "db4", "db5", "db6", "db7", "db8")

#: 尺度滤波器系数 :math:`h_0..h_{L-1}`（Daubechies 1992, Table 6.1 的标准 dbN 表，
#: 以双精度落表；**正确性由定义式在断言里复核**——正交性、归一化、:math:`N` 阶消失矩，
#: 见 :func:`orthonormality_defect` / :func:`vanishing_moment_defect` 与
#: ``tests/2.6/test_wavelet_core.py::test_daubechies_filters_satisfy_defining_properties``）。
SCALING_TABLE: dict[str, tuple[float, ...]] = {
    "haar": (0.7071067811865476, 0.7071067811865476),
    "db2": (0.48296291314453416, 0.83651630373780794, 0.22414386804201339,
            -0.12940952255126037),
    "db3": (0.33267055295008263, 0.80689150931109255, 0.45987750211849154,
            -0.13501102001025458, -0.085441273882026658, 0.035226291885709533),
    "db4": (0.23037781330889651, 0.71484657055291567, 0.63088076792985892,
            -0.027983769416859854, -0.18703481171909309, 0.030841381835560764,
            0.032883011666885197, -0.010597401785069032),
    "db5": (0.16010239797419293, 0.60382926979718965, 0.72430852843777294,
            0.13842814590132074, -0.24229488706638203, -0.032244869584638375,
            0.077571493840045719, -0.0062414902127982744, -0.012580751999081999,
            0.0033357252854737712),
    "db6": (0.11154074335010947, 0.49462389039845306, 0.75113390802109536,
            0.31525035170919763, -0.22626469396543983, -0.12976686756726194,
            0.097501605587323043, 0.027522865530305727, -0.03158203931748603,
            0.00055384220116149613, 0.0047772575109455108, -0.0010773010853084796),
    "db7": (0.077852054085009184, 0.39653931948191729, 0.72913209084623509,
            0.46978228740519312, -0.14390600392856498, -0.22403618499387498,
            0.071309219266830259, 0.080612609151083078, -0.038029936935014413,
            -0.016574541630666881, 0.01255099855609984, 0.00042957797292136651,
            -0.0018016407040474908, 0.00035371379997452024),
    "db8": (0.054415842243104008, 0.31287159091429995, 0.67563073629728976,
            0.58535468365420673, -0.015829105256349306, -0.28401554296154691,
            0.00047248457391328279, 0.12874742662047847, -0.017369301001807547,
            -0.044088253930794755, 0.013981027917398282, 0.0087460940474057766,
            -0.0048703529934515741, -0.00039174037337694705, 0.00067544940645056933,
            -0.00011747678412476953),
}

#: 边界延拓方式。三者在 :func:`dwt` 里都走「延拓 + 严格周期正交变换 + 裁回」。
BOUNDARY_MODES: tuple[str, ...] = ("periodic", "symmetric", "zero")

#: MAD 估计量的正态一致常数：:math:`\Phi^{-1}(3/4)`，:math:`\hat\sigma=\mathrm{MAD}/c`。
_MAD_CONST: float = 0.6744897501960817


def _check_wavelet(name: str) -> str:
    key = str(name).lower()
    if key not in SCALING_TABLE:
        raise ValueError(f"未知小波 {name!r}，可用：{list(WAVELET_NAMES)}")
    return key


def _check_mode(mode: str) -> str:
    key = str(mode).lower()
    if key not in BOUNDARY_MODES:
        raise ValueError(f"未知边界模式 {mode!r}，可用：{list(BOUNDARY_MODES)}")
    return key


# --------------------------------------------------------------------------- #
# 滤波器
# --------------------------------------------------------------------------- #
def scaling_filter(name: str = "db4") -> np.ndarray:
    r"""尺度滤波器 :math:`h`（长度 :math:`L=2N`，:math:`\sum_j h_j=\sqrt2`）。

    小波滤波器由 :func:`wavelet_filter` 给出：:math:`g_j=(-1)^j h_{L-1-j}`。
    """
    key = _check_wavelet(name)
    return np.array(SCALING_TABLE[key], dtype=float)


def wavelet_filter(name: str = "db4") -> np.ndarray:
    r"""小波（高通）滤波器 :math:`g_j=(-1)^j h_{L-1-j}`。

    这一条**就是**二尺度方程里 :math:`\psi` 的离散化身；:math:`N` 阶消失矩
    :math:`\sum_j j^m g_j=0\ (m<N)` 是本章全部"逼近阶"结论的来源。
    """
    h = scaling_filter(name)
    L = h.size
    j = np.arange(L, dtype=float)
    return ((-1.0) ** j) * h[L - 1 - np.arange(L)]


def filter_length(name: str = "db4") -> int:
    r"""滤波器长度 :math:`L=2N`（支撑的样本数）。"""
    return int(scaling_filter(name).size)


def vanishing_moments(name: str = "db4") -> int:
    r"""消失矩阶数 :math:`N`：dbN 为 :math:`N`，Haar 为 :math:`1`。

    等价说法（本章用它）：:math:`\deg p<N` 的多项式落在 :math:`V_J` 里，
    于是它的**细节系数恒为 0**（到舍入级）。
    """
    key = _check_wavelet(name)
    return 1 if key == "haar" else int(key[2:])


def filter_response(h: Any, omega: Any) -> np.ndarray:
    r"""滤波器频率响应 :math:`H(\omega)=\sum_j h_j e^{-i\omega j}`（逐点）。"""
    a = np.asarray(h, dtype=float).ravel()
    w = np.asarray(omega, dtype=float)
    j = np.arange(a.size, dtype=float)
    return np.exp(-1j * np.outer(w, j)) @ a


def orthonormality_defect(h: Any) -> float:
    r"""正交性缺陷 :math:`\max_k\left|\sum_j h_j h_{j-2k}-\delta_k\right|`。

    正交小波的**定义性质**：自相关在偶数位移上只剩 :math:`\delta`。
    """
    a = np.asarray(h, dtype=float).ravel()
    L = a.size
    worst = 0.0
    for k in range(-(L // 2 - 1), L // 2):
        lo, hi = max(0, 2 * k), min(L, L + 2 * k)
        s = float(np.dot(a[lo - 2 * k:hi - 2 * k], a[lo:hi])) if hi > lo else 0.0
        target = 1.0 if k == 0 else 0.0
        worst = max(worst, abs(s - target))
    return float(worst)


def vanishing_moment_defect(h: Any, order: int) -> float:
    r"""第 ``order`` 阶消失矩的**相对**缺陷
    :math:`\left|\sum_j(-1)^j j^{m} h_j\right|/\sum_j j^{m}|h_j|`。

    必须相对化：:math:`j^m` 本身可以到 :math:`10^{8}` 量级（db8、m=7），
    绝对残差 :math:`10^{-10}` 看着很大，其实是 :math:`u` 级。
    """
    a = np.asarray(h, dtype=float).ravel()
    j = np.arange(a.size, dtype=float)
    m = int(order)
    w = j ** m
    num = abs(float(np.sum(((-1.0) ** j) * w * a)))
    den = float(np.sum(w * np.abs(a)))
    return float(num / den) if den > 0.0 else 0.0


def perfect_reconstruction_defect(h: Any, n_omega: int = 512) -> float:
    r"""完美重构条件 :math:`|H(\omega)|^2+|H(\omega+\pi)|^2=2` 的最大偏差。

    这是「滤波器组与完美重构」在频域的样子：它是 :func:`orthonormality_defect`
    的等价命题（两者都在断言里出现，互为独立路径）。
    """
    a = np.asarray(h, dtype=float).ravel()
    w = np.linspace(0.0, np.pi, int(n_omega), endpoint=False)
    H1 = filter_response(a, w)
    H2 = filter_response(a, w + np.pi)
    return float(np.max(np.abs(np.abs(H1) ** 2 + np.abs(H2) ** 2 - 2.0)))


# --------------------------------------------------------------------------- #
# 边界延拓
# --------------------------------------------------------------------------- #
def guard_length(name: str = "db4", mode: str = "symmetric") -> int:
    r"""延拓余量：``periodic`` 为 0，其余模式为 :math:`L-1`（滤波器读不到原数据之外）。"""
    key = _check_mode(mode)
    return 0 if key == "periodic" else filter_length(name) - 1


def extension_length(n: int, level: int, name: str = "db4", mode: str = "symmetric") -> int:
    r"""延拓后的长度 :math:`n_{\text{ext}}`：不小于 :math:`n+2g` 且被 :math:`2^{J}` 整除。

    ``periodic`` 直接返回 :math:`n`（并要求 :math:`2^{J}\mid n`，否则抛 ``ValueError``）。
    """
    n = int(n)
    level = int(level)
    if n < 1:
        raise ValueError(f"信号长度必须为正，得到 {n}")
    if level < 0:
        raise ValueError(f"层数必须非负，得到 {level}")
    key = _check_mode(mode)
    block = 2 ** level
    if key == "periodic":
        if n % block:
            raise ValueError(
                f"periodic 模式要求 2^{level} = {block} 整除信号长度，得到 n={n}；"
                f"请改用 symmetric/zero（它们用余量把长度补到 {block} 的倍数）")
        return n
    g = guard_length(name, key)
    # 左侧余量取 block 的整数倍：这样「延拓坐标 → 原坐标」在第 j 层恰好是除以 2^j
    # （否则 off/2^j 不是整数，环绕与延拓两种口径的系数没法逐点对齐比较）。
    off = -(-g // block) * block
    need = n + 2 * off
    return int(-(-need // block) * block)


def reflect_index(i: Any, n: int) -> np.ndarray:
    r"""半样本对称反射：把任意整数下标折回 :math:`[0,n)`（周期 :math:`2n`）。

    ``..., x2, x1, x0 | x0, x1, x2, ..., x_{n-1} | x_{n-1}, x_{n-2}, ...``
    也就是 :math:`x[-1]=x[0]`、:math:`x[n]=x[n-1]`（**不重复**样本的写法是整样本
    对称，它不给出正交小波的完美重构；这里用的是半样本版本）。
    """
    idx = np.asarray(i, dtype=np.int64)
    p = 2 * int(n)
    t = np.mod(idx, p)
    return np.where(t < n, t, p - 1 - t).astype(np.int64)


def extend_signal(x: Any, name: str = "db4", level: int = 1,
                  mode: str = "symmetric") -> tuple[np.ndarray, int, int]:
    r"""按边界模式把 :math:`x` 延拓成 :math:`x_{\text{ext}}`。

    返回 ``(x_ext, offset, n_orig)``：``x_ext[offset:offset+n_orig] == x``。
    ``periodic`` 的 ``offset`` 为 0（不延拓）。
    """
    a = np.asarray(x, dtype=float).ravel()
    n = int(a.size)
    key = _check_mode(mode)
    n_ext = extension_length(n, level, name, key)
    if key == "periodic":
        return a.copy(), 0, n
    block = 2 ** max(int(level), 0)
    off = -(-guard_length(name, key) // block) * block
    pos = np.arange(n_ext, dtype=np.int64) - off
    if key == "zero":
        out = np.zeros(n_ext, dtype=float)
        inside = (pos >= 0) & (pos < n)
        out[inside] = a[pos[inside]]
    else:  # symmetric
        out = a[reflect_index(pos, n)]
    return out, int(off), n


# --------------------------------------------------------------------------- #
# 变换
# --------------------------------------------------------------------------- #
@dataclass
class WaveletDecomposition:
    r"""一层分析的结果：各层近似系数（长度递减）与细节系数。

    ``approx[0]`` 是原信号（:math:`x_{\text{ext}}`），``approx[j]`` 是第 :math:`j`
    层近似，``detail[j]`` 是第 :math:`j` 层细节（下标从 1 开始数层）。
    """

    name: str
    mode: str
    n: int
    level: int
    offset: int
    approx: list[np.ndarray] = field(default_factory=list)
    detail: list[np.ndarray] = field(default_factory=list)

    @property
    def n_ext(self) -> int:
        r"""延拓后的长度 :math:`n_{\text{ext}}`（周期模式等于原长）。"""
        return int(self.approx[0].size)

    @property
    def redundancy(self) -> float:
        r"""冗余度 :math:`n_{\text{ext}}/n`（周期模式为 1，其余 :math:`>1`）。"""
        return float(self.n_ext) / float(self.n)

    def coefficients(self) -> np.ndarray:
        r"""把所有系数摊平成一维（:math:`a_J` 在前，随后 :math:`d_J,\dots,d_1`）。"""
        parts = [self.approx[-1]] + [self.detail[j] for j in range(self.level, 0, -1)]
        return np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts])

    def with_bands(self, approx: Sequence[Any], details: Sequence[Any]) -> "WaveletDecomposition":
        r"""用新的各层近似与细节数组造一个同构分解（供裁剪 / 阈值 / 逐层重构复用）。

        ``approx`` / ``details`` 都必须从第 0 项（= 延拓后的原信号）开始、长度
        ``level+1``——下标 :math:`j` 就是层号，这样 :func:`idwt` 不必猜。
        """
        if len(approx) != self.level + 1 or len(details) != self.level + 1:
            raise ValueError(f"approx/details 各需 {self.level + 1} 项")
        return WaveletDecomposition(
            name=self.name, mode=self.mode, n=self.n, level=self.level, offset=self.offset,
            approx=[np.asarray(a, dtype=float) for a in approx],
            detail=[np.asarray(d, dtype=float) for d in details],
        )


def dwt_step(x_ext: Any, h: Any, g: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""一层的分析与合成：

    .. math::

        a_k=\sum_j h_j\,x[2k+j],\qquad d_k=\sum_j g_j\,x[2k+j],

    下标约定是 :math:`a_k` 贴着 :math:`x[2k]`（Haar 时 :math:`a_k=(x_{2k}+x_{2k+1})/\sqrt2`，
    与教材的「块平均」一致）。在这个约定下合成滤波器**不需要再翻转**：
    :math:`x[2k+j]\mathrel{+}=h_j a_k+g_j d_k` 就是分析的伴随+逆
    （:func:`idwt_step`），这是把「滤波器组」写成矩阵后可以直接验证的事实。
    """
    a = np.asarray(x_ext, dtype=float).ravel()
    n = int(a.size)
    if n % 2:
        raise ValueError(f"单层变换要求偶数长度，得到 {n}")
    hh = np.asarray(h, dtype=float).ravel()
    gg = np.asarray(g, dtype=float).ravel()
    if hh.size != gg.size:
        raise ValueError("尺度与小波滤波器长度必须相同")
    m = n // 2
    idx = np.mod(2 * np.arange(m, dtype=np.int64)[:, None]
                 + np.arange(hh.size, dtype=np.int64)[None, :], n)
    return a[idx] @ hh, a[idx] @ gg


def idwt_step(a: Any, d: Any, h: Any, g: Any, n_out: int) -> np.ndarray:
    r"""一层合成：:math:`x[2k+j]\mathrel{+}=h_j a_k+g_j d_k`（模 :math:`n_{\text{out}}`）。

    对固定的 :math:`j`，下标 :math:`(2k+j)\bmod n_{\text{out}}` 两两不同
    （:math:`n_{\text{out}}` 为偶数时 :math:`2` 在 :math:`\mathbb Z_{n_{\text{out}}}`
    里不是单射，但 :math:`k` 只跑到 :math:`n_{\text{out}}/2-1`，恰好取遍一个陪集）
    ——所以可以直接用花式下标累加，不需要 ``np.add.at`` 那种慢路径。
    """
    aa = np.asarray(a, dtype=float).ravel()
    dd = np.asarray(d, dtype=float).ravel()
    if aa.size != dd.size:
        raise ValueError(f"近似与细节系数长度必须相同：{aa.size} vs {dd.size}")
    hh = np.asarray(h, dtype=float).ravel()
    gg = np.asarray(g, dtype=float).ravel()
    n_out = int(n_out)
    if 2 * aa.size != n_out:
        raise ValueError(f"系数个数 {aa.size} 与输出长度 {n_out} 不匹配（应为一半）")
    m = aa.size
    idx = np.mod(2 * np.arange(m, dtype=np.int64)[:, None]
                 + np.arange(hh.size, dtype=np.int64)[None, :], n_out)
    vals = aa[:, None] * hh[None, :] + dd[:, None] * gg[None, :]
    out = np.zeros(n_out, dtype=float)
    for j in range(hh.size):
        out[idx[:, j]] += vals[:, j]
    return out


def dwt(x: Any, name: str = "db4", level: int = 1,
        mode: str = "symmetric") -> WaveletDecomposition:
    r"""多层离散小波变换（DWT）：延拓 → 严格周期正交变换 → 逐层减半。

    ``level=0`` 时返回只含原信号的退化分解（用于统一的逐层图）。
    """
    key = _check_wavelet(name)
    md = _check_mode(mode)
    level = int(level)
    x_ext, off, n = extend_signal(x, key, max(level, 0), md)
    h = scaling_filter(key)
    g = wavelet_filter(key)
    approx = [x_ext]
    detail: list[np.ndarray] = [x_ext]
    cur = x_ext
    for _ in range(level):
        a, d = dwt_step(cur, h, g)
        approx.append(a)
        detail.append(d)
        cur = a
    return WaveletDecomposition(name=key, mode=md, n=n, level=level, offset=off,
                                approx=approx, detail=detail)


def idwt(decomp: WaveletDecomposition, *, levels: int | None = None) -> np.ndarray:
    r"""重构（逆 DWT）：从最粗的近似与各层细节回到原长度信号。

    ``levels=k`` 时只从第 :math:`k` 层往下合成（:math:`k<J` 时其余细节被丢掉——
    这正是逐层图里「看到第几层」的语义）。

    退化情形（:func:`truncate_critical` 那种"一点延拓余量都不留"的分解）会连长度都对不上：
    这时把缺口补 0 并照常返回——**这本身就是证据**，那种做法不是"差几个舍入位"，
    而是换了一个变换（见 ``pitfalls``）。
    """
    key = _check_wavelet(decomp.name)
    h = scaling_filter(key)
    g = wavelet_filter(key)
    top = decomp.level if levels is None else max(0, min(int(levels), decomp.level))
    cur = np.asarray(decomp.approx[top], dtype=float)
    for j in range(top, 0, -1):
        cur = idwt_step(cur, np.asarray(decomp.detail[j], dtype=float), h, g, 2 * cur.size)
    off = decomp.offset
    tail = cur[off:off + decomp.n]
    if tail.size < decomp.n:
        tail = np.concatenate([tail, np.zeros(decomp.n - tail.size)])
    return tail


def mra_bands(decomp: WaveletDecomposition) -> np.ndarray:
    r"""逐层重构的**分量**：返回形状 ``(J+1, n)`` 的数组。

    第 0 行是最粗近似单独重构出来的 :math:`x_J`，第 :math:`j` 行（:math:`j\ge1`）
    是第 :math:`j` 层细节单独重构出来的 :math:`w_j`；按行求和逐位等于原信号
    （:math:`x=x_J+\sum_j w_j`）。这条恒等式是「多分辨分析」在代码里的样子。
    """
    key = _check_wavelet(decomp.name)
    h = scaling_filter(key)
    g = wavelet_filter(key)
    J = decomp.level
    zero_like = [np.zeros_like(np.asarray(a, dtype=float)) for a in decomp.approx]
    bands = np.zeros((J + 1, decomp.n), dtype=float)
    # 最粗近似：把细节全置零后合成
    cur = np.asarray(decomp.approx[J], dtype=float)
    for j in range(J, 0, -1):
        cur = idwt_step(cur, np.zeros_like(np.asarray(decomp.detail[j], dtype=float)), h, g, 2 * cur.size)
    bands[0] = cur[decomp.offset:decomp.offset + decomp.n]
    # 第 j 层细节：从第 j 层往上只带这一层的细节
    for j in range(1, J + 1):
        v = idwt_step(np.zeros_like(np.asarray(decomp.approx[j], dtype=float)),
                      np.asarray(decomp.detail[j], dtype=float), h, g,
                      2 * int(np.asarray(decomp.approx[j]).size))
        for k in range(j - 1, 0, -1):
            v = idwt_step(v, np.zeros_like(np.asarray(decomp.detail[k], dtype=float)), h, g,
                          2 * v.size)
        bands[j] = v[decomp.offset:decomp.offset + decomp.n]
    return bands


def mra_partial(decomp: WaveletDecomposition) -> np.ndarray:
    r"""逐层重构的**累加**：形状 ``(J+1, n)``，第 :math:`j` 行 = :math:`x_J+\sum_{i>J-j}w_i`。

    最后一行逐位等于原信号（容差 :math:`\\sim u`）；中间行是「只保留最粗的
    :math:`j` 层」的近似，图上能看到它一层层贴上原曲线。
    """
    bands = mra_bands(decomp)
    out = np.zeros_like(bands)
    running = bands[0].copy()
    out[0] = running
    J = decomp.level
    for j in range(1, J + 1):
        running = running + bands[J - j + 1]
        out[j] = running
    return out


def dwt_matrix(n: int, name: str = "db4", mode: str = "symmetric",
               level: int = 1) -> np.ndarray:
    r"""把分析写成**显式矩阵**：形状 ``(n_coeff, n)``，第 :math:`i` 列是 :math:`e_i` 的系数。

    它是对拍用的第二条独立路径：递归算法 vs 矩阵（逐列跑同一个 :func:`dwt`，
    再与直接构造的矩阵比较）。周期模式下它是方阵且**正交**
    （:func:`orthogonality_defect` 应贴 :math:`u`），其余模式是冗余表示。
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"信号长度必须为正，得到 {n}")
    cols = []
    for i in range(n):
        e = np.zeros(n, dtype=float)
        e[i] = 1.0
        cols.append(dwt(e, name, level, mode).coefficients())
    return np.stack(cols, axis=1)


def orthogonality_defect(M: Any) -> float:
    r""":math:`\max|MM^{\mathsf T}-I|` 与 :math:`\max|M^{\mathsf T}M-I|` 的较大者。

    正交变换两个方向都该贴 :math:`u`；冗余表示只有 :math:`M^{\mathsf T}M=I`
    （等距嵌入）成立。
    """
    A = np.asarray(M, dtype=float)
    if A.ndim != 2:
        raise ValueError("需要二维矩阵")
    eye_c = np.eye(A.shape[0], dtype=float)
    eye_r = np.eye(A.shape[1], dtype=float)
    left = float(np.max(np.abs(A @ A.T - eye_c)))
    right = float(np.max(np.abs(A.T @ A - eye_r)))
    return max(left, right)


def truncate_critical(decomp: WaveletDecomposition) -> WaveletDecomposition:
    r"""把每层系数裁到**临界采样**的个数 :math:`\lceil n/2^{j}\rceil`。

    这是「不留延拓余量」的反面教材：裁掉的正是延拓撑出来的那些系数，
    重构误差立刻从 :math:`u` 级跳到 :math:`O(1)`——边界处理骗人的地方就在这里。
    """
    if decomp.n % (2 ** decomp.level):
        raise ValueError(
            f"临界采样裁剪要求 2^{decomp.level} 整除 n={decomp.n}，"
            f"否则各层几何长度对不上（这正是「临界采样」的记账代价）")
    details = []
    approx = list(decomp.approx)
    for j in range(1, decomp.level + 1):
        m = -(-decomp.n // (2 ** j))
        details.append(np.asarray(decomp.detail[j], dtype=float)[:m])
    m_top = -(-decomp.n // (2 ** decomp.level))
    approx[decomp.level] = np.asarray(decomp.approx[decomp.level], dtype=float)[:m_top]
    return WaveletDecomposition(name=decomp.name, mode=decomp.mode, n=decomp.n,
                                level=decomp.level, offset=decomp.offset,
                                approx=approx, detail=[decomp.approx[0]] + details)


def reconstruction_error(x: Any, name: str = "db4", level: int = 1,
                         mode: str = "symmetric") -> float:
    r"""往返误差 :math:`\max_i|\hat x_i-x_i|/\max_i|x_i|`（相对，免量纲）。"""
    a = np.asarray(x, dtype=float).ravel()
    back = idwt(dwt(a, name, level, mode))
    scale = max(float(np.max(np.abs(a))), 1e-300)
    return float(np.max(np.abs(back - a))) / scale


def coefficient_lengths(decomp: WaveletDecomposition) -> np.ndarray:
    r"""各层系数个数：``[|a_J|, |d_J|, ..., |d_1|]``（画分解树时要用）。"""
    return np.array([decomp.approx[-1].size] + [decomp.detail[j].size
                                                for j in range(decomp.level, 0, -1)], dtype=float)


def coefficient_energy(decomp: WaveletDecomposition) -> float:
    r"""系数能量 :math:`\sum|c|^2`（Parseval 口径的左边）。"""
    c = decomp.coefficients()
    return float(np.dot(c, c))


def energy_ratio(decomp: WaveletDecomposition) -> float:
    r"""能量比 :math:`\|\text{coeff}\|_2^2/\|x\|_2^2`（分母是**原信号**，不是延拓后的）。

    正交小波 + 周期延拓时**恰好**为 1；``zero`` 补的是零、延拓不添能量，也是 1；
    ``symmetric`` 的反射**复制**了样本（:math:`n_{\text{ext}}>n`），于是系数能量
    等于延拓信号的能量、比原信号多出被复制的那部分——实测 db4/n=256/J=4 约 1.12
    （正好是冗余度 :math:`n_{\text{ext}}/n` 的量级）。
    """
    x = np.asarray(decomp.approx[0], dtype=float)
    orig = x[decomp.offset:decomp.offset + decomp.n]
    return coefficient_energy(decomp) / float(np.dot(orig, orig))


def level_energies(decomp: WaveletDecomposition) -> np.ndarray:
    r"""各层占比：``[||a_J||^2, ||d_J||^2, ..., ||d_1||^2] / ||x||^2``（逐层能量分布）。

    分母同样是**原信号**能量；三种模式的行和因此分别是 1、1、:math:`>1`
    （与 :func:`energy_ratio` 同口径）。
    """
    x = np.asarray(decomp.approx[0], dtype=float)
    orig = x[decomp.offset:decomp.offset + decomp.n]
    tot = float(np.dot(orig, orig))
    parts = [float(np.dot(decomp.approx[-1], decomp.approx[-1]))]
    parts += [float(np.dot(decomp.detail[j], decomp.detail[j]))
              for j in range(decomp.level, 0, -1)]
    return np.array(parts, dtype=float) / tot


def boundary_pollution(x: Any, name: str = "db4", level: int = 2,
                       mode: str = "periodic", tol: float = 1e-10) -> dict[str, Any]:
    r"""不做延拓（直接环绕）时，边界系数被污染的量级与**影响范围**。

    做法：把同一个信号用「有余量的延拓」（:func:`dwt` 的正常路径）与「一点余量都不留」
    （:math:`n_{\text{ext}}=n` 的纯周期变换）各算一遍，逐层统计
    :math:`|d^{\text{wrap}}_j-d^{\text{guard}}_j|`。返回字典含每层的最大偏差、
    「超过 ``tol`` 的系数个数」以及这些系数对应的**物理长度**。

    实测（db4、n=512、J=4 的低频双音信号）：偏差最大的系数个数几乎不随层数变
    （3/5/6/6 个），但它们的**物理长度** = 个数 :math:`\\times 2^{j}` 按 :math:`2^{j}`
    增长（6 → 20 → 48 → 96 个样本），偏差幅度也从 0.051 涨到 3.827
    ——「不做延拓时边界会污染整层」指的就是后者。
    """
    key = _check_wavelet(name)
    md = _check_mode(mode)
    lvl = max(1, int(level))
    a = np.asarray(x, dtype=float).ravel()
    guarded = dwt(a, key, lvl, "symmetric" if md == "periodic" else md)
    n = a.size
    block = 2 ** lvl
    n_per = (n // block) * block
    if n_per < 2 * filter_length(key):
        raise ValueError("信号太短，无法比较环绕与延拓")
    wrap = dwt(a[:n_per], key, lvl, "periodic")
    worst = np.zeros(lvl + 1, dtype=float)
    counts = np.zeros(lvl + 1, dtype=float)
    spans = np.zeros(lvl + 1, dtype=float)
    for j in range(1, lvl + 1):
        m = wrap.detail[j].size
        # 延拓坐标 → 原坐标平移 offset，再对齐两种口径的公共下标
        g_arr = np.asarray(guarded.detail[j], dtype=float)
        off = guarded.offset
        start = int(np.ceil(off / (2 ** j)))
        gsel = g_arr[start:start + m]
        k = int(min(m, gsel.size))
        if k == 0:
            continue
        diff = np.abs(np.asarray(wrap.detail[j], dtype=float)[:k] - gsel[:k])
        worst[j] = float(np.max(diff)) if diff.size else 0.0
        counts[j] = float(np.count_nonzero(diff > tol))
        spans[j] = counts[j] * float(2 ** j)
    return {"level_max": worst, "n_polluted": counts, "physical_span": spans,
            "n": float(n), "n_used": float(n_per), "tol": float(tol)}


# --------------------------------------------------------------------------- #
# 消失矩
# --------------------------------------------------------------------------- #
#: 标准测试信号（Donoho & Johnstone 1994 的四个 + 两个给时间-尺度图用的）：
#: ``bumps`` / ``blocks`` / ``heavisine`` / ``doppler`` 是稀疏表示文献里的基准信号，
#: ``tone`` / ``chirp`` 用来对照 STFT 的固定分辨率。全部是 :math:`t` 的确定性函数。
TEST_SIGNAL_NAMES: tuple[str, ...] = ("bumps", "blocks", "heavisine", "doppler", "tone", "chirp")


def test_signal(kind: str = "bumps", n: int = 256, *, t0: float = 0.0,
                t1: float = 1.0) -> np.ndarray:
    r"""基准测试信号在 :math:`[t_0,t_1)` 上的 ``n`` 个等距样本（**确定性**，不用随机源）。

    * ``bumps``：:math:`\sum_k h_k/(1+|(t-t_k)/w_k|)^4`（宽窄不一的脉冲）；
    * ``blocks``：分段常数（跳变）；
    * ``heavisine``：:math:`\tfrac4\pi\sin 4\pi t-\operatorname{sign}(t-\tfrac12)`（低频+一个跳变）；
    * ``doppler``：:math:`\sqrt{t(1-t)}\sin\!\bigl(2\pi\cdot 1.05/(t+0.05)\bigr)`（变频）；
    * ``tone``：两个整数周期数的正弦（:math:`5` 与 :math:`17` 个周期）；
    * ``chirp``：线性调频 :math:`\sin(2\pi(6t+18t^2))`。
    """
    key = str(kind).lower()
    if key not in TEST_SIGNAL_NAMES:
        raise ValueError(f"未知测试信号 {kind!r}，可用：{list(TEST_SIGNAL_NAMES)}")
    m = int(n)
    if m < 8:
        raise ValueError(f"信号长度至少 8，得到 {m}")
    t = np.linspace(float(t0), float(t1), m, endpoint=False)
    if key == "bumps":
        pos = (0.10, 0.22, 0.35, 0.48, 0.62, 0.78, 0.90)
        amp = (6.0, -5.0, 4.0, -3.5, 3.0, -2.5, 2.0)
        wid = (0.020, 0.015, 0.030, 0.012, 0.025, 0.018, 0.010)
        out = np.zeros(m, dtype=float)
        for p, a, wv in zip(pos, amp, wid):
            out = out + a / (1.0 + np.abs((t - p) / wv)) ** 4
        return out
    if key == "blocks":
        edges = (0.10, 0.20, 0.32, 0.45, 0.55, 0.68, 0.80, 0.92)
        hei = (3.0, -4.0, 5.0, -2.0, 4.0, -6.0, 3.0, -1.0)
        out = np.full(m, 2.0, dtype=float)
        for i, (e0, e1) in enumerate(zip(edges[:-1], edges[1:])):
            out = np.where((t >= e0) & (t < e1), hei[i], out)
        return out
    if key == "heavisine":
        return (4.0 / np.pi) * np.sin(4.0 * np.pi * t) - np.where(t < 0.5, 1.0, -1.0)
    if key == "doppler":
        return np.sqrt(np.maximum(t * (1.0 - t), 0.0)) * np.sin(
            2.0 * np.pi * 1.05 / (t + 0.05))
    if key == "tone":
        return np.sin(2.0 * np.pi * 5.0 * t) + 0.5 * np.sin(2.0 * np.pi * 17.0 * t)
    return np.sin(2.0 * np.pi * (6.0 * t + 18.0 * t * t))


def polynomial_signal(n: int, degree: int, *, t0: float = -1.0, t1: float = 1.0) -> np.ndarray:
    r"""多项式样本 :math:`x_i=t_i^{k}`，:math:`t_i` 在 :math:`[t_0,t_1)` 上等距。

    默认 :math:`[-1,1)`：:math:`|x|\le1`，于是「细节系数 vs 近似系数的量级比」
    可以直接读，不会有额外的尺度因子混进来。
    """
    m = int(degree)
    if m < 0:
        raise ValueError(f"次数必须非负，得到 {m}")
    t = np.linspace(float(t0), float(t1), int(n), endpoint=False)
    return t ** m


def interior_slice(n_coeff: int, level: int, offset: int, n: int,
                   name: str = "db4") -> slice:
    r"""某一层里**不碰延拓区**的系数下标区间（内部系数）。

    第 :math:`j` 层系数 :math:`k` 的分析支撑在延拓坐标上是
    :math:`[2^{j}k,\ 2^{j}(k+L-1)]`；要求它整段落在 :math:`[g,\,g+n)` 内。
    :math:`\deg p<N` 的多项式只有在这些系数上才**必须**是 0
    ——边界附近它们本来就被延拓（零或反射）破坏成非多项式。
    """
    L = filter_length(name)
    step = 2 ** int(level)
    lo = int(-(-offset // step))
    hi = int((offset + n - 1) // step) - (L - 1)
    hi = max(hi, lo)
    return slice(max(lo, 0), min(hi, int(n_coeff)))


def detail_scale_ratio(decomp: WaveletDecomposition, level: int,
                       *, name: str | None = None) -> dict[str, float]:
    r"""第 ``level`` 层细节与同层近似的**量级比**（内部系数口径）。

    ``ratio = max|d| / max|a|``；``u`` 级说明多项式被抹平，:math:`O(1)` 说明没有。
    分母取同层近似而不是原信号：深层的系数整体在变小，用原信号做分母会把
    「消失矩」读成「系数变小了」。
    """
    key = _check_wavelet(name or decomp.name)
    j = int(level)
    if j < 1 or j > decomp.level:
        raise ValueError(f"层号必须在 1..{decomp.level}，得到 {j}")
    sl = interior_slice(decomp.detail[j].size, j, decomp.offset, decomp.n, key)
    d = np.asarray(decomp.detail[j], dtype=float)[sl]
    a = np.asarray(decomp.approx[j], dtype=float)[sl]
    dm = float(np.max(np.abs(d))) if d.size else 0.0
    am = float(np.max(np.abs(a))) if a.size else 0.0
    return {"detail_max": dm, "approx_max": am, "ratio": dm / am if am > 0 else 0.0,
            "n_interior": float(d.size)}


def vanishing_moment_probe(name: str = "db4", degree: int = 2, n: int = 256,
                           level: int = 3, mode: str = "zero") -> dict[str, Any]:
    r"""消失矩的**双向**证据：返回 ``degree`` 次多项式在各层的内部系数与边界系数。

    ``zero`` 模式默认（多项式被零延拓，内部的"多项式性"最干净）；
    返回 ``interior_ratio`` / ``boundary_ratio`` 两条曲线（逐层）与
    :math:`\max_j` 的读数——:math:`\deg p<N` 时前者 :math:`\sim u`，
    :math:`\deg p\ge N` 时两者都是 :math:`O(1)`。
    """
    key = _check_wavelet(name)
    J = max(1, int(level))
    x = polynomial_signal(n, degree)
    dec = dwt(x, key, J, mode)
    scale = float(np.max(np.abs(x)))
    interior = np.zeros(J, dtype=float)
    boundary = np.zeros(J, dtype=float)
    n_int = np.zeros(J, dtype=float)
    for j in range(1, J + 1):
        d = np.asarray(dec.detail[j], dtype=float)
        sl = interior_slice(d.size, j, dec.offset, dec.n, key)
        a = np.asarray(dec.approx[j], dtype=float)
        am = float(np.max(np.abs(a[sl]))) if a[sl].size else 1.0
        interior[j - 1] = (float(np.max(np.abs(d[sl]))) / am) if d[sl].size else 0.0
        n_int[j - 1] = float(d[sl].size)
        mask = np.ones(d.size, dtype=bool)
        mask[sl] = False
        boundary[j - 1] = (float(np.max(np.abs(d[mask]))) / scale) if np.any(mask) else 0.0
    N = vanishing_moments(key)
    have = n_int > 0
    return {
        "name": key, "degree": int(degree), "n": int(n), "level": J, "mode": mode,
        "moments": N, "interior_ratio": interior, "boundary_ratio": boundary,
        "n_interior": n_int, "n_levels_with_interior": float(np.count_nonzero(have)),
        "interior_max": float(np.max(interior[have])) if np.any(have) else 0.0,
        "boundary_max": float(np.max(boundary)),
        "killed": bool(degree < N and np.any(have)
                       and float(np.max(interior[have])) < 1e-12),
    }


# --------------------------------------------------------------------------- #
# 平稳小波变换（SWT，非抽取）
# --------------------------------------------------------------------------- #
@dataclass
class StationaryDecomposition:
    r"""非抽取（平稳）小波变换的结果：每层系数都与原信号**等长**。

    ``approx[j]`` / ``detail[j]`` 均为长度 :math:`n` 的数组，:math:`j=1..J`；
    ``approx[0]`` 是原信号，冗余因子 :math:`J+1`。

    **不要**把 :math:`J+1` 当成「紧框架界」：本实现（a trous 递归版）的框架算子
    是频率相关的，:math:`\sum|c|^2/\|x\|^2` 实测随信号频率在 :math:`2` 与
    :math:`2^{J}` 之间变化（低频信号取到上端，白噪声取到 :math:`\approx 2^{J-1}+1` 附近），
    只有 :math:`J=1` 层是真正的紧框架（比值恒为 2，见 :func:`swt_frame_symbol`）。
    """

    name: str
    n: int
    level: int
    approx: list[np.ndarray] = field(default_factory=list)
    detail: list[np.ndarray] = field(default_factory=list)

    @property
    def redundancy(self) -> float:
        return float(self.level + 1)

    def coefficients(self) -> np.ndarray:
        parts = [self.approx[self.level]] + [self.detail[j]
                                             for j in range(self.level, 0, -1)]
        return np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts])


def _swt_step(x: Any, h: Any, g: Any, dilation: int) -> tuple[np.ndarray, np.ndarray]:
    r"""a trous 一层：滤波器每两个抽头之间插 :math:`2^{j-1}-1` 个零，不抽取。"""
    a = np.asarray(x, dtype=float).ravel()
    n = a.size
    d = int(dilation)
    m = h.size
    idx = np.mod(np.arange(n, dtype=np.int64)[:, None]
                 + d * np.arange(m, dtype=np.int64)[None, :], n)
    return a[idx] @ h, a[idx] @ g


def _iswt_step(a: Any, d: Any, h: Any, g: Any, dilation: int) -> np.ndarray:
    r"""a trous 一层合成：:math:`x[i]=\tfrac12\sum_j h_j a[i-dj]+\tfrac12\sum_j g_j d[i-dj]`。

    那个 :math:`\tfrac12` 来自完美重构条件 :math:`|H(\omega)|^2+|G(\omega)|^2=2`
    （对正交滤波器**逐点**成立），不是「除以冗余因子」——SWT 的框架算子并不等于
    冗余因子（见 :class:`StationaryDecomposition` 与 :func:`swt_frame_symbol`）。
    """
    aa = np.asarray(a, dtype=float).ravel()
    dd = np.asarray(d, dtype=float).ravel()
    n = aa.size
    dlt = int(dilation)
    m = h.size
    idx = np.mod(np.arange(n, dtype=np.int64)[:, None]
                 - dlt * np.arange(m, dtype=np.int64)[None, :], n)
    return 0.5 * (aa[idx] @ h + dd[idx] @ g)


def swt(x: Any, name: str = "db4", level: int = 1) -> StationaryDecomposition:
    r"""平稳小波变换（undecimated / stationary WT），周期边界。

    与 :func:`dwt` 的唯一区别是**不抽取**：每层输出都与输入等长，
    因此没有"平移一格系数就全变"的下采样相位问题（见 :func:`translation_change`）。
    """
    key = _check_wavelet(name)
    a = np.asarray(x, dtype=float).ravel()
    J = max(1, int(level))
    h = scaling_filter(key)
    g = wavelet_filter(key)
    approx = [a.copy()]
    detail: list[np.ndarray] = [a.copy()]
    cur = a.copy()
    for j in range(1, J + 1):
        cur, d = _swt_step(cur, h, g, 2 ** (j - 1))
        approx.append(cur)
        detail.append(d)
    return StationaryDecomposition(name=key, n=int(a.size), level=J,
                                   approx=approx, detail=detail)


def iswt(decomp: StationaryDecomposition) -> np.ndarray:
    r""":func:`swt` 的逆：逐层从最粗往回合成，得到原信号（到舍入级）。"""
    key = _check_wavelet(decomp.name)
    h = scaling_filter(key)
    g = wavelet_filter(key)
    cur = np.asarray(decomp.approx[decomp.level], dtype=float)
    for j in range(decomp.level, 0, -1):
        cur = _iswt_step(cur, np.asarray(decomp.detail[j], dtype=float), h, g, 2 ** (j - 1))
    return cur


def scalogram(decomp: StationaryDecomposition) -> np.ndarray:
    r"""时间-尺度（尺度图）矩阵：形状 ``(J+1, n)``，行序为 :math:`[a_J,d_J,\dots,d_1]`。

    每行都是**全长**的，所以它可以直接当热力图；与 STFT 的固定分辨率并排时，
    第 :math:`j` 行的时间支撑是 :math:`(L-1)2^{j-1}+1` 个样本（见 :func:`scale_time_boxes`）。
    """
    rows = [np.asarray(decomp.approx[decomp.level], dtype=float)]
    rows += [np.asarray(decomp.detail[j], dtype=float)
             for j in range(decomp.level, 0, -1)]
    return np.stack(rows, axis=0)


def scale_time_boxes(name: str = "db4", level: int = 4,
                     n: int = 256, n_window: int = 16) -> dict[str, np.ndarray]:
    r"""多分辨 vs 固定分辨的「分辨率元」尺寸。

    小波第 :math:`j` 层：时间支撑 :math:`(L-1)2^{j-1}+1` 个样本
    （有效宽度约 :math:`2^{j}`），频带 :math:`[\pi/2^{j},\pi/2^{j-1}]`
    ——面积近似恒定（不确定性原理的离散版）。
    STFT：时间窗宽 ``n_window`` 个样本、频带 :math:`2\pi/n\_window` 固定不变。
    """
    key = _check_wavelet(name)
    L = filter_length(key)
    J = max(1, int(level))
    j = np.arange(1, J + 1, dtype=float)
    width = (L - 1) * (2.0 ** (j - 1)) + 1.0
    n_win = int(n_window)
    return {
        "wavelet_scale": j,
        "wavelet_width": width,
        "wavelet_width_norm": width / float(n),
        "wavelet_band": 1.0 / (2.0 ** j),
        "stft_width": np.full(J, float(n_win)),
        "stft_width_norm": np.full(J, float(n_win) / float(n)),
        "stft_band": np.full(J, 2.0 / float(n_win)),
    }


def translation_change(x: Any, name: str = "db4", level: int = 2,
                       kind: str = "dwt") -> dict[str, float]:
    r"""平移一个样本后系数变化多少（平移不变性的**可执行判据**）。

    信号 :math:`x_{+1}[i]=x[i-1]`（循环平移）。判据是**逐带**比较：

    * ``kind="swt"``：要求 :math:`c_j(x_{+1})` 恰好等于 :math:`c_j(x)` 循环平移一格
      ——不抽取的变换对平移是等变的，实测变化**恰好为 0**（逐位相同：同一批浮点加法
      作用在平移后的输入上）；
    * ``kind="dwt"``：DWT 每层抽取 :math:`2^{j}`，平移一格会让相位整体错位，
      系数并不只是"挪个位置"；这里取「允许每带整体挪一格」之后仍然残留的最大失配。

    返回 ``{"change", "scale", "relative", ...}``；``scale`` 是系数本身的量级
    （第 1 层细节的最大绝对值），**必须相对化**才有意义：0.28 的绝对变化
    在 scale=0.17 时是 160%，在 scale=100 时什么都不是。
    """
    key = _check_wavelet(name)
    a = np.asarray(x, dtype=float).ravel()
    n = a.size
    shifted = np.roll(a, 1)
    if str(kind).lower() == "swt":
        s0 = swt(a, key, level)
        s1 = swt(shifted, key, level)
        worst = 0.0
        for j in range(0, level + 1):
            c0 = np.asarray(s0.detail[j] if j else s0.approx[0], dtype=float)
            c1 = np.asarray(s1.detail[j] if j else s1.approx[0], dtype=float)
            worst = max(worst, float(np.max(np.abs(np.roll(c0, 1) - c1))))
        scale = float(np.max(np.abs(np.asarray(s0.detail[1], dtype=float))))
        return {"change": float(worst), "scale": scale,
                "relative": float(worst / scale) if scale > 0 else 0.0,
                "shift": 1.0, "redundancy": float(level + 1)}
    if n % (2 ** level):
        raise ValueError(f"DWT 平移判据要求 2^{level} 整除 n={n}（否则连系数个数都在变）")
    d0 = dwt(a, key, level, "periodic")
    d1 = dwt(shifted, key, level, "periodic")
    worst = 0.0
    for j in range(1, level + 1):
        c0 = np.asarray(d0.detail[j], dtype=float)
        c1 = np.asarray(d1.detail[j], dtype=float)
        s = 2 ** (j - 1) % c0.size
        worst = max(worst, float(np.max(np.abs(np.roll(c0, s) - c1))))
    scale = float(np.max(np.abs(np.asarray(d0.detail[1], dtype=float))))
    return {"change": float(worst), "scale": scale,
            "relative": float(worst / scale) if scale > 0 else 0.0,
            "shift": 1.0, "redundancy": 1.0}


def swt_frame_symbol(name: str = "db4", level: int = 4, omega: Any = None) -> np.ndarray:
    r"""SWT 的**框架算子符号** :math:`S(\omega)`（把「能量比」写成频率的函数）。

    逐层展开 :math:`a_j=H_j a_{j-1}`、:math:`d_j=G_j a_{j-1}`，
    :math:`|H_j(\omega)|^2+|G_j(\omega)|^2=2`（完美重构条件），于是

    .. math::

        S(\omega)=2+\sum_{i=1}^{J-1}\prod_{k=1}^{i}|H(2^{k-1}\omega)|^2 ,

    它**不是常数**：低频处 :math:`|H|\to\sqrt2`，:math:`S\to 2^{J}`；
    :math:`\omega=\pi` 处 :math:`|H|=0`，:math:`S\to 2`。所以「SWT 是界为
    :math:`J+1` 的紧框架」这句话对递归 a trous 版本**不成立**（只有 :math:`J=1` 成立）。
    """
    key = _check_wavelet(name)
    h = scaling_filter(key)
    J = max(1, int(level))
    w = np.linspace(0.0, np.pi, 512, endpoint=False) if omega is None else np.asarray(omega, dtype=float)
    S = np.full(w.shape, 2.0)
    prod = np.ones(w.shape)
    for i in range(1, J):
        prod = prod * np.abs(filter_response(h, (2.0 ** (i - 1)) * w)) ** 2
        S = S + prod
    return S



# --------------------------------------------------------------------------- #
# 阈值与去噪
# --------------------------------------------------------------------------- #
def soft_threshold(c: Any, lam: float) -> np.ndarray:
    r"""软阈值 :math:`S_\lambda(c)=\operatorname{sign}(c)\max(|c|-\lambda,0)`。

    它是 :math:`\tfrac12(c-\text{obs})^2+\lambda|\text{obs}|` 的邻近算子，
    所以对**大系数也有 :math:`\lambda` 的系统性收缩**——偏差就是这么来的。
    """
    a = np.asarray(c, dtype=float)
    t = float(lam)
    if t < 0.0:
        raise ValueError(f"阈值必须非负，得到 {t}")
    return np.sign(a) * np.maximum(np.abs(a) - t, 0.0)


def hard_threshold(c: Any, lam: float) -> np.ndarray:
    r"""硬阈值 :math:`H_\lambda(c)=c\cdot\mathbf 1_{|c|>\lambda}`。

    无偏（大系数原样保留）但**不连续**：输入动 :math:`\epsilon`、
    输出可能跳 :math:`\lambda`，方差就是这么来的。
    """
    a = np.asarray(c, dtype=float)
    t = float(lam)
    if t < 0.0:
        raise ValueError(f"阈值必须非负，得到 {t}")
    return np.where(np.abs(a) > t, a, 0.0)


def threshold_decomposition(decomp: WaveletDecomposition, lam: float,
                            rule: str = "soft") -> WaveletDecomposition:
    r"""对**细节**系数逐层阈值（最粗近似原样保留）。"""
    rule = str(rule).lower()
    if rule not in ("soft", "hard"):
        raise ValueError(f"未知阈值规则 {rule!r}，可用：['soft', 'hard']")
    f = soft_threshold if rule == "soft" else hard_threshold
    details = [decomp.detail[0]] + [f(np.asarray(decomp.detail[j], dtype=float), lam)
                                    for j in range(1, decomp.level + 1)]
    return WaveletDecomposition(name=decomp.name, mode=decomp.mode, n=decomp.n,
                                level=decomp.level, offset=decomp.offset,
                                approx=list(decomp.approx), detail=details)


def denoise(x: Any, name: str = "db4", level: int = 3, mode: str = "symmetric",
            lam: float = 0.0, rule: str = "soft") -> np.ndarray:
    r"""小波阈值去噪：``idwt(threshold(dwt(x)))``。

    ``lam=0`` 时就是恒等（往返误差 :math:`u` 级），这是"阈值曲线"的零点锚。
    """
    return idwt(threshold_decomposition(dwt(x, name, level, mode), lam, rule))


def universal_threshold(sigma: float, n: int) -> float:
    r"""通用阈值（VisuShrink）:math:`\lambda=\sigma\sqrt{2\ln n}`。

    它保证「白噪声的系数最大值不超过 :math:`\lambda`」的概率趋于 1，
    因此**偏保守**（把真实信号的系数也一起杀掉一部分）——实测最优点在它的一半附近。
    """
    s = float(sigma)
    m = int(n)
    if s < 0.0:
        raise ValueError(f"噪声水平必须非负，得到 {s}")
    if m < 1:
        raise ValueError(f"长度必须为正，得到 {m}")
    return s * float(np.sqrt(2.0 * np.log(float(m))))


def estimate_sigma_mad(detail: Any) -> float:
    r"""尺度 1 细节系数上的 MAD 噪声估计 :math:`\hat\sigma=\mathrm{med}(|d|)/0.6745`。

    它对信号是稳健的（少数大系数拉不动中位数），代价是小样本方差大、
    而且**只在最细一层**才近似无偏（:math:`d_1` 个数最多）。
    """
    d = np.asarray(detail, dtype=float).ravel()
    if d.size == 0:
        raise ValueError("细节系数为空，无法估计噪声水平")
    return float(np.median(np.abs(d)) / _MAD_CONST)


def bias_variance_curve(x_clean: Any, sigma: float, lambdas: Any, *,
                        name: str = "db4", level: int = 3, mode: str = "symmetric",
                        rule: str = "soft", n_trials: int = 64,
                        seed: int = 20261018) -> dict[str, np.ndarray]:
    r"""阈值去噪的偏差-方差-均方误差曲线（蒙特卡洛，公共随机数）。

    对每个 :math:`\lambda` 与每个噪声实现 :math:`x^{(t)}=x+\sigma\xi^{(t)}`：

    .. math::

        \text{bias}^2(\lambda)=\Bigl\|\overline{\hat x(\lambda)}-x\Bigr\|_2^2/n,\quad
        \text{var}(\lambda)=\overline{\bigl\|\hat x(\lambda)-\overline{\hat x}\bigr\|_2^2}/n,

    而 :math:`\mathrm{MSE}=\text{bias}^2+\text{var}`（逐点分解，样本均值口径）。
    噪声实现**在所有阈值间共用**（common random numbers）：否则两个 :math:`\lambda`
    的差会被"换了噪声"淹没，曲线看起来到处是锯齿，读不出此消彼长。
    """
    a = np.asarray(x_clean, dtype=float).ravel()
    n = a.size
    lams = np.asarray(lambdas, dtype=float).ravel()
    T = int(n_trials)
    g = rng(int(seed))
    noise = g.standard_normal((T, n))
    est = np.zeros((lams.size, T, n), dtype=float)
    for i, lam in enumerate(lams):
        for t in range(T):
            est[i, t] = denoise(a + float(sigma) * noise[t], name, level, mode,
                                float(lam), rule)
    mean = est.mean(axis=1)
    bias2 = np.mean((mean - a[None, :]) ** 2, axis=1)
    var = np.mean(est.var(axis=1), axis=1)
    mse = np.mean((est - a[None, None, :]) ** 2, axis=(1, 2))
    return {"lambda": lams, "bias2": bias2, "var": var, "mse": mse,
            "bias2_plus_var": bias2 + var, "sigma": np.full(lams.size, float(sigma)),
            "n": np.full(lams.size, float(n)), "n_trials": np.full(lams.size, float(T))}


def denoise_mse(x_clean: Any, x_noisy: Any, *, name: str = "db4", level: int = 3,
                mode: str = "symmetric", lam: float = 0.0, rule: str = "soft") -> float:
    r"""单次实现的均方误差 :math:`\\frac1n\\|\\hat x-x\\|_2^2`（含 :math:`\\lambda=0` 的对照）。"""
    a = np.asarray(x_clean, dtype=float).ravel()
    b = np.asarray(x_noisy, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("干净信号与含噪信号长度必须相同")
    return float(np.mean((denoise(b, name, level, mode, lam, rule) - a) ** 2))


def optimal_threshold(lambdas: Any, curve: Any) -> dict[str, float]:
    r"""曲线上最小的那个点（三点抛物线细化，给出比栅格更准的位置）。

    栅格扫描给出的最优点被栅格间距量化；抛物线细化是**插值**，
    与 2.5 章「峰值插值精度远好于栅格」是同一件事。
    """
    lam = np.asarray(lambdas, dtype=float).ravel()
    val = np.asarray(curve, dtype=float).ravel()
    if lam.size == 0 or lam.size != val.size:
        raise ValueError("阈值与曲线长度必须一致且非空")
    i = int(np.argmin(val))
    refined = float(lam[i])
    if 0 < i < lam.size - 1:
        x0, x1, x2 = lam[i - 1], lam[i], lam[i + 1]
        y0, y1, y2 = val[i - 1], val[i], val[i + 1]
        den = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(den) > 0:
            A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
            B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
            if A > 0.0:
                cand = -B / (2.0 * A)
                if x0 <= cand <= x2:
                    refined = float(cand)
    return {"index": float(i), "lambda": refined, "lambda_grid": float(lam[i]),
            "mse": float(val[i])}


def snr_db(x_clean: Any, x_test: Any) -> float:
    r"""信噪比 :math:`10\log_{10}(\|x\|^2/\|x_{\text{test}}-x\|^2)`（dB）。"""
    a = np.asarray(x_clean, dtype=float).ravel()
    b = np.asarray(x_test, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("两个信号长度必须相同")
    err = float(np.dot(b - a, b - a))
    if err <= 0.0:
        return float("inf")
    return 10.0 * float(np.log10(float(np.dot(a, a)) / err))


# --------------------------------------------------------------------------- #
# 固定分辨率对照（STFT）
# --------------------------------------------------------------------------- #
def stft_magnitude(x: Any, n_window: int = 16, hop: int = 4,
                   window_name: str = "hann") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""短时傅里叶幅度谱（固定窗长），作为时间-尺度图的**固定分辨率对照**。

    返回 ``(t_index, freq_norm, mag)``：``t_index`` 是窗中心在原信号里的下标，
    ``freq_norm`` 是 :math:`[0,1)` 上的归一化频率（周期/样本），``mag[k, i]`` 是幅度。
    窗长固定意味着时间分辨率与频率分辨率**同时**被钉死——这正是小波要对比的东西。
    这里只调用 2.5 章 :func:`numcore.transform.fft.fft` 的公开 API（同包、已声明的依赖）。
    """
    a = np.asarray(x, dtype=float).ravel()
    n = a.size
    w = int(n_window)
    step = int(hop)
    if w < 2 or w > n:
        raise ValueError(f"窗长必须在 2..{n} 之间，得到 {w}")
    if step < 1:
        raise ValueError(f"跳步必须为正，得到 {step}")
    key = str(window_name).lower()
    idx = np.arange(w, dtype=float)
    if key == "hann":
        win = 0.5 - 0.5 * np.cos(2.0 * np.pi * idx / float(w))
    elif key == "rect":
        win = np.ones(w, dtype=float)
    else:
        raise ValueError(f"未知窗 {window_name!r}，可用：['hann', 'rect']")
    centers = np.arange(w // 2, n - (w - 1) // 2, step, dtype=int)
    mag = np.zeros((w // 2 + 1, centers.size), dtype=float)
    t_index = np.zeros(centers.size, dtype=float)
    for i, c in enumerate(centers):
        seg = np.zeros(w, dtype=float)
        lo, hi = c - w // 2, c - w // 2 + w
        s0, s1 = max(lo, 0), min(hi, n)
        seg[s0 - lo:s1 - lo] = a[s0:s1]
        spec = np.abs(fft(seg * win))
        mag[:, i] = spec[:w // 2 + 1]
        t_index[i] = float(c)
    freq = np.arange(w // 2 + 1, dtype=float) / float(w)
    return t_index, freq, mag
