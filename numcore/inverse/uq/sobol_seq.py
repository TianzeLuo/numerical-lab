r"""低差异点集本体：Sobol' 数字网、Halton、秩一格点、张量格点，以及它们的随机化。

为什么这一层必须自己写（而不是调一个现成实现）
------------------------------------------------
本项目的 P3 门禁禁止 ``numpy`` 的成品算法；更要紧的是，5.4 章要断言的**恰恰是**
"低差异序列的误差估计从哪里来"这件事。如果序列本身来自黑盒，那么
"不置乱就没有误差估计"这句话就只是一个传说；在这里它是一条能被读出来的代码事实：
:func:`sobol_points` 的 ``scramble="none"`` 分支返回的是**一个确定的点集**，
对同一个 ``n`` 重复调用逐位相同——于是任何"复制实验的方差"都恒等于 0，
而真实误差并不为 0。:mod:`numcore.inverse.uq.qmc` 与 ``tests/5.4`` 把这句话量化。

Sobol' 序列的构造（本实现的全部数学）
-------------------------------------
在 GF(2) 上取原始多项式 :math:`p_j(x)=x^{s_j}+a_1x^{s_j-1}+\cdots+a_{s_j-1}x+1`，
与初始方向数 :math:`m_{j,1},\dots,m_{j,s_j}`（要求 :math:`m_{j,k}` 为奇数且
:math:`m_{j,k}<2^k`）。方向向量由递推给出

.. math::
    v_{j,k}=m_{j,k}2^{L-k}\ (k\le s_j),\qquad
    v_{j,k}=v_{j,k-s_j}\oplus(v_{j,k-s_j}\!\gg\! s_j)
      \oplus\bigoplus_{i=1}^{s_j-1} a_i\,\bigl(v_{j,k-i}\bigr)\ (k>s_j),

第 ``n`` 个点取 :math:`x_{j,n}=\bigoplus_{b} \beta_b(n)\,v_{j,b+1}`（:math:`\beta_b` 是 ``n`` 的二进制位）。
这个"数字网排序"与前 2^m 个点的 Gray 码排序给出**同一个点集**（Gray 码在
:math:`\{0,1\}^m` 上是双射，见 :func:`sobol_points` 的 docstring），而它更容易检查：
前 2^m 个点恰好是前 m 个方向向量的全部异或组合。

由 :math:`m_{j,k}` 为奇数知方向向量矩阵是"对角线上全 1"的下三角矩阵，故它在
:math:`\{0,1\}^m` 上是**可逆线性映射**：前 :math:`2^m` 个点在每一维上的投影是
:math:`\{0,2^{-m},\dots,1-2^{-m}\}` 的一个排列——**逐位均匀分层**，这是
``tests/5.4`` 里那条 B 类不变量的全部内容，它对任意合法方向数都成立（与"取到好的
:math:`t` 值"是两件事，后者只能实测，见 :func:`sobol_t_value`）。

随机化
------
* ``scramble="shift"``：Cranley–Patterson 式的**数字移位**（按位异或一个随机 L 位掩码）。
  它保持网性质、给出无偏估计，是最省事的随机化。
* ``scramble="linear"``：Matoušek 的**线性置乱**——把方向向量矩阵左乘一个随机的
  "下三角、对角全 1"的可逆矩阵（等价于对每个输出位做嵌套置换的线性版本），
  同样保持网性质，但复制之间的差别比单纯移位大得多，误差估计更稳。
* ``scramble="none"``：确定性点集。**它没有误差估计**——这不是实现缺陷，是定义使然。

出处：Sobol' (1967)；Bratley & Fox (1988), *Algorithm 659*；Joe & Kuo (2008)；
Matoušek (1998) 的线性置乱；Dick & Pillichshammer, *Digital Nets and Sequences* (2010)。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from numcore.utils import rng

__all__ = [
    "L_BITS",
    "PRIMITIVE_TABLE",
    "primitive_polynomials",
    "initial_direction_numbers",
    "direction_vectors",
    "sobol_points",
    "sobol_t_value",
    "sobol_net_defect",
    "van_der_corput",
    "radical_inverse",
    "halton_points",
    "lattice_points",
    "korobov_generator",
    "tensor_grid",
    "point_set",
    "is_power_of_two",
]

#: 方向向量的位宽。32 位 = 分辨率 2^-32 ≈ 2.3e-10，对 N ≤ 2^20 的帧足够，
#: 且所有点都能在 float64 里精确表示（二进制小数）。
L_BITS = 32

#: 前 12 个维度用的原始多项式与初始方向数：``(degree, a_bits, m_1..m_s)``。
#:
#: 这是 Bratley–Fox / Joe–Kuo 系经典表的前 13 维（dim 1 是 van der Corput，不需要多项式）。
#: ``a_bits`` 编码 :math:`x^s+a_1x^{s-1}+\cdots+1` 的中间系数（``a`` 的第 :math:`s-1-i` 位是
#: :math:`a_i`）。表本身**不保证**好看：``tests/5.4`` 用 :func:`sobol_t_value` 实测它的
#: :math:`t` 值并如实登记——"我抄了一张表"不是证据，"这张表实测 :math:`t=...`" 才是。
PRIMITIVE_TABLE: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (1, 0, (1,)),  # dim 2
    (2, 1, (1, 3)),  # dim 3
    (3, 1, (1, 3, 1)),  # dim 4
    (3, 2, (1, 1, 1)),  # dim 5
    (4, 1, (1, 1, 3, 3)),  # dim 6
    (4, 4, (1, 3, 5, 13)),  # dim 7
    (5, 2, (1, 1, 5, 5, 17)),  # dim 8
    (5, 4, (1, 1, 5, 5, 5)),  # dim 9
    (5, 7, (1, 1, 7, 11, 19)),  # dim 10
    (5, 11, (1, 1, 5, 1, 1)),  # dim 11
    (5, 13, (1, 1, 1, 3, 11)),  # dim 12
    (5, 14, (1, 3, 5, 5, 7)),  # dim 13
)


# --------------------------------------------------------------------------- #
# GF(2) 多项式与原始多项式（表用完了自己造，保证 dim 任意）
# --------------------------------------------------------------------------- #
def _gf2_mul_mod(a: int, b: int, mod: int) -> int:
    """GF(2)[x]/(mod) 上的乘法（整数按位表示多项式，位 i 是 x^i 的系数）。"""
    out = 0
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        a <<= 1
        if a & (1 << (mod.bit_length() - 1)):
            a ^= mod
    return out


def _is_primitive(poly: int) -> bool:
    """判定 GF(2) 上多项式 ``poly`` 是否原始（阶恰为 ``2^s - 1``）。

    只用于 :func:`primitive_polynomials`，而它只在 ``dim`` 超出内置表时才被调用，
    所以这里的朴素实现（逐次平方 + 素因子检验）足够：``s ≤ 6`` 时 ``2^s - 1 ≤ 63``。
    """
    s = poly.bit_length() - 1
    if s < 1:
        return False
    order = (1 << s) - 1
    # 素因子分解（order 很小）
    factors: list[int] = []
    rest = order
    d = 2
    while d * d <= rest:
        if rest % d == 0:
            factors.append(d)
            while rest % d == 0:
                rest //= d
        d += 1
    if rest > 1:
        factors.append(rest)
    x = 2  # 多项式 x
    acc = x
    for _ in range(order - 1):
        acc = _gf2_mul_mod(acc, x, poly)
    if acc != 1:
        return False
    for q in factors:
        acc = x
        for _ in range(order // q - 1):
            acc = _gf2_mul_mod(acc, x, poly)
        if acc == 1:
            return False
    return True


def primitive_polynomials(count: int) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """给出 ``count`` 个 ``(degree, a_bits, m_init)`` 三元组（与 :data:`PRIMITIVE_TABLE` 同格式）。

    前 ``len(PRIMITIVE_TABLE)`` 条直接取内置表；不够的部分按度数、再按编码升序搜索原始
    多项式，初始方向数取 :math:`m_k=2k-1`（奇数且 :math:`<2^k`，合法但**不保证** :math:`t` 小——
    这一点由 :func:`sobol_t_value` 实测，而不是由注释保证）。
    """
    if count <= len(PRIMITIVE_TABLE):
        return PRIMITIVE_TABLE[:count]
    out = list(PRIMITIVE_TABLE)
    s = 6
    while len(out) < count:
        cands = [p for p in range((1 << s) | 1, 1 << (s + 1), 2) if _is_primitive(p)]
        for poly in cands:
            if len(out) >= count:
                break
            a = (poly >> 1) & ((1 << (s - 1)) - 1)
            m = tuple(2 * (k + 1) - 1 for k in range(s))
            out.append((s, a, m))
        s += 1
    return tuple(out[:count])


def initial_direction_numbers(degree: int, a_bits: int, m_init: Sequence[int], n_bits: int = L_BITS) -> np.ndarray:
    """由 ``(degree, a_bits, m_init)`` 用 Sobol' 递推生成 ``n_bits`` 个方向向量（uint32）。"""
    s = int(degree)
    v = np.zeros(n_bits, dtype=np.uint32)
    for k in range(min(s, n_bits)):
        v[k] = np.uint32(int(m_init[k]) << (n_bits - 1 - k))
    for k in range(s, n_bits):
        acc = int(v[k - s]) ^ (int(v[k - s]) >> s)
        for i in range(1, s):
            if (a_bits >> (s - 1 - i)) & 1:
                acc ^= int(v[k - i])
        v[k] = np.uint32(acc)
    return v


def direction_vectors(dim: int, *, n_bits: int = L_BITS) -> np.ndarray:
    """``(dim, n_bits)`` 的 uint32 方向向量矩阵；第 0 维是 van der Corput（``v_k = 2^{L-k}``）。"""
    if dim < 1:
        raise ValueError("dim 必须 >= 1")
    out = np.zeros((dim, n_bits), dtype=np.uint32)
    for k in range(n_bits):
        out[0, k] = np.uint32(1 << (n_bits - 1 - k))
    table = primitive_polynomials(max(0, dim - 1))
    for j in range(1, dim):
        s, a, m = table[j - 1]
        out[j] = initial_direction_numbers(s, a, m, n_bits=n_bits)
    return out


# --------------------------------------------------------------------------- #
# 数字网：核心生成器
# --------------------------------------------------------------------------- #
def _digital_net(n: int, v: np.ndarray, shift: np.ndarray | None, n_bits: int) -> np.ndarray:
    """由方向向量生成前 ``n`` 个点（uint32 位模式）。"""
    dim = v.shape[0]
    x = np.zeros((n, dim), dtype=np.uint32)
    idx = np.arange(n, dtype=np.uint32)
    m = max(1, int(n - 1).bit_length()) if n > 1 else 1
    m = min(m, n_bits)
    for b in range(m):
        mask = ((idx >> b) & 1).astype(bool)
        if not mask.any():
            break
        x[mask] ^= v[:, b]
    if shift is not None:
        x ^= shift[None, :]
    return x


def _bits_to_unit(x: np.ndarray, n_bits: int) -> np.ndarray:
    return x.astype(np.float64) * float(2.0 ** (-n_bits))


def _linear_scramble(v: np.ndarray, generator: Any, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Matoušek 线性置乱：**所有维共用**一个随机可逆矩阵 + 各维独立数字移位。

    矩阵 :math:`M` 必须满足两条，缺一条网性质就没了——而且**一维分层照样完美**，
    只看边缘分布根本发现不了：

    1. **方向**：输出第 :math:`b'` 位（:math:`b'=31` 是最高位、也是最粗的数字）
       只许由输入第 :math:`b\ge b'` 位（更粗或同位）组合而成。反过来（用更细的位去改粗位）
       会把"前 :math:`m_1` 位"搅进块外的信息，2 维每格计数从"恰好 1"变成 0 或 4。
       本模块第一版两个方向都试过、但当时忘了钉对角线，于是两个方向都失败；
       真正修好需要"方向对吗 + 对角全 1"同时成立。
    2. **对角全 1**：否则 :math:`M` 几乎必然是奇异的。

    为什么"所有维共用一个矩阵"：:math:`(t,m,s)` 网的性质是某些行子矩阵可逆；
    共用 :math:`M` 时这些子矩阵被同一个块下三角（对角 1）矩阵左乘，可逆性不变。
    ``tests/5.4`` 用 GF(2) 秩判据逐格验证：每个 :math:`m_1+m_2=m` 的切法
    都必须给出恰好一个点。
    """
    dim = v.shape[0]
    r = generator.integers(0, 2, size=(n_bits, n_bits)).astype(np.uint32)
    sb = generator.integers(0, 2, size=(dim, n_bits)).astype(np.uint32)
    cols = np.zeros(n_bits, dtype=np.uint32)
    for bp in range(n_bits):
        bitset = 1 << bp  # 对角全 1
        for b in range(bp):
            if r[b, bp]:
                bitset |= 1 << b
        cols[bp] = np.uint32(bitset)
    w = np.zeros_like(v)
    shift = np.zeros(dim, dtype=np.uint32)
    for j in range(dim):
        s = 0
        for b in range(n_bits):
            if sb[j, b]:
                s |= 1 << b
        shift[j] = np.uint32(s)
        for bp in range(n_bits):
            mask = ((v[j] >> np.uint32(bp)) & np.uint32(1)).astype(bool)
            if mask.any():
                w[j, mask] ^= cols[bp]
    return w, shift


def _digital_shift(dim: int, generator: Any, n_bits: int) -> np.ndarray:
    bits = generator.integers(0, 2, size=(dim, n_bits)).astype(np.uint32)
    out = np.zeros(dim, dtype=np.uint32)
    for j in range(dim):
        s = 0
        for b in range(n_bits):
            if bits[j, b]:
                s |= 1 << b
        out[j] = np.uint32(s)
    return out


def _as_generator(seed: Any) -> Any:
    """把种子或现成的随机源统一成"有 ``integers``/``random`` 的对象"。

    这里用鸭子类型而不是 ``isinstance``：P4 要求 ``numcore/`` 下只经
    :func:`numcore.utils.rng` 拿随机数，连那家库的生成器类型名都不许出现
    （检查按子串匹配，写在注释里也算违规）。
    """
    if hasattr(seed, "integers") and hasattr(seed, "random"):
        return seed
    return rng(seed)


def sobol_points(
    n: int,
    dim: int,
    *,
    seed: Any = None,
    scramble: str = "none",
    n_bits: int = L_BITS,
) -> np.ndarray:
    r"""前 ``n`` 个 Sobol' 点，形状 ``(n, dim)``，落在 :math:`[0,1)^d`。

    排序约定（写清楚，因为"第 n 个点"在不同实现里不一样）：本实现用**数字网排序**

    .. math:: x_{j,n}=\bigoplus_{b}\beta_b(n)\,v_{j,b+1},\qquad \beta_b(n)=\text{n 的第 }b\text{ 位}.

    当 :math:`n=2^m` 时，这个点集与经典的 Gray 码排序点集**逐点相同**（Gray 码
    :math:`g(n)=n\oplus(n\gg1)` 在 :math:`\{0,1\}^m` 上是双射，且两点集都等于
    前 m 个方向向量的全部异或组合）——所以"前 N 个点的偏差"这个量不受排序约定影响。
    :math:`n` 不是 2 的幂时两者不同，此时本实现就是"数字网的前 n 个点"。

    ``scramble``：

    * ``"none"``：确定性。**同一 (n, dim) 反复调用逐位相同**——没有误差估计。
    * ``"shift"``：数字移位（保持网性质）。
    * ``"linear"``：线性置乱（保持网性质，复制间差异更大）。
    """
    if n < 1 or dim < 1:
        raise ValueError("n 与 dim 必须 >= 1")
    v = direction_vectors(dim, n_bits=n_bits)
    shift: np.ndarray | None = None
    if scramble == "none":
        shift = None
    elif scramble == "shift":
        shift = _digital_shift(dim, _as_generator(seed), n_bits)
    elif scramble == "linear":
        v, shift = _linear_scramble(v, _as_generator(seed), n_bits)
    else:
        raise ValueError(f"未知 scramble={scramble!r}（可用 none / shift / linear）")
    return _bits_to_unit(_digital_net(n, v, shift, n_bits), n_bits)


def sobol_net_defect(n: int, dim: int, *, level: int, scramble: str = "none", seed: Any = None) -> int:
    r"""``(t, m, s)`` 网性质的**缺陷计数**：``level`` 层基本区间里点数不为 ``n/2^level`` 的个数。

    ``n`` 必须是 ``2^m``（``m ≥ level``）。返回 0 表示前 ``n`` 个点在 ``level`` 层上恰好均匀。
    定义按数字网的标准写法：把 :math:`[0,1)^d` 沿每维切成 :math:`2^{k_j}` 段，
    :math:`\sum k_j=\text{level}` 的每个盒子应当含 :math:`n/2^{\text{level}}` 个点。
    """
    if not is_power_of_two(n):
        raise ValueError("net_defect 只对 n = 2^m 定义")
    m = int(np.log2(n))
    if level > m:
        raise ValueError("需要 level <= log2(n)")
    pts = sobol_points(n, dim, seed=seed, scramble=scramble)
    bad = 0
    # 枚举所有 sum(k_j) = level 的切分；每个盒子应当恰好含 n / 2^level 个点
    for ks in _compositions(level, dim):
        code = np.zeros(n, dtype=np.int64)
        for j, kj in enumerate(ks):
            stride = 1 << kj
            cj = np.minimum((pts[:, j] * stride).astype(np.int64), stride - 1)
            code = code * stride + cj
        uniq, cnt = np.unique(code, return_counts=True)
        expect = n >> level
        bad += int(np.count_nonzero(cnt != expect)) + int((1 << level) - uniq.size)
    return int(bad)


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    """所有 ``sum(k_j)=total``、``k_j >= 0`` 的切分。"""
    if parts == 1:
        return [(total,)]
    out: list[tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            out.append((first,) + rest)
    return out


def sobol_t_value(n: int, dim: int, *, level: int | None = None, scramble: str = "none", seed: Any = None) -> int:
    """实测的 ``t`` 值：使前 ``n=2^m`` 个点成为 ``(t, m, dim)`` 网的最小 ``t``。

    定义：前 :math:`2^m` 个点是 :math:`(t,m,s)`-网，当且仅当每个体积为
    :math:`2^{t-m}` 的基本区间恰好含 :math:`2^t` 个点。``t`` 由方向数决定，
    **不是**"抄了 Sobol' 三个字就自动等于 0**——这里量出来是多少就记多少。
    """
    if not is_power_of_two(n):
        raise ValueError("t 值只对 n = 2^m 定义")
    m = int(np.log2(n))
    lev = m if level is None else int(level)
    pts = sobol_points(n, dim, seed=seed, scramble=scramble)
    for t in range(0, m + 1):
        k = m - t  # 检查 level = k 的所有切分
        ok = True
        for ks in _compositions(k, dim):
            code = np.zeros(n, dtype=np.int64)
            for j, kj in enumerate(ks):
                cj = np.minimum((pts[:, j] * (1 << kj)).astype(np.int64), (1 << kj) - 1)
                code = code * max(1, 1 << kj) + cj
            uniq, cnt = np.unique(code, return_counts=True)
            expect = 1 << t
            if uniq.size != (1 << k) or np.any(cnt != expect):
                ok = False
                break
        if ok:
            return t
    return m


# --------------------------------------------------------------------------- #
# Halton / van der Corput
# --------------------------------------------------------------------------- #
def radical_inverse(index: Any, base: int) -> np.ndarray:
    """radical inverse :math:`\\phi_b(n)`：把 ``n`` 在 ``base`` 下的数字反序写在小数点后。"""
    return _radical_inverse_masked(index, base, mask=None)


def _radical_inverse_masked(index: Any, base: int, mask: int | None) -> np.ndarray:
    """radical inverse 的"数字移位"版本：第 k 位加 ``(mask >> k) & 1`` 再模 ``base``。"""
    idx = np.asarray(index, dtype=np.int64).ravel().copy()
    if base < 2:
        raise ValueError("base 必须 >= 2")
    out = np.zeros(idx.size, dtype=np.float64)
    scale = 1.0 / base
    k = 0
    while np.any(idx > 0):
        d = idx % base
        if mask is not None:
            d = (d + ((mask >> k) & 1)) % base
        out += d * scale
        idx //= base
        scale /= base
        k += 1
    return out.reshape(np.asarray(index).shape)


def van_der_corput(n: int, base: int = 2, *, start: int = 0) -> np.ndarray:
    """``n`` 个 van der Corput 点（``base=2`` 时就是 Sobol' 的第 1 维）。"""
    return radical_inverse(np.arange(start, start + n), base)


def _primes(count: int) -> list[int]:
    out: list[int] = []
    cand = 2
    while len(out) < count:
        if all(cand % p for p in out if p * p <= cand):
            out.append(cand)
        cand += 1
    return out


def halton_points(n: int, dim: int, *, seed: Any = None, scramble: str = "none", start: int = 0) -> np.ndarray:
    """Halton 点集：第 ``j`` 维用第 ``j`` 个素数为基的 radical inverse。

    ``scramble="shift"`` 用"数字移位"（逐位异或随机掩码，即 :math:`\\phi` 的
    :math:`\\beta`-adic 加性移位），是 Halton 上最省事的随机化。
    """
    if n < 1 or dim < 1:
        raise ValueError("n 与 dim 必须 >= 1")
    bases = _primes(dim)
    gen = _as_generator(seed) if scramble == "shift" else None
    out = np.zeros((n, dim), dtype=np.float64)
    if scramble not in ("none", "shift"):
        raise ValueError(f"未知 scramble={scramble!r}（可用 none / shift）")
    for j, b in enumerate(bases):
        mask = None
        if gen is not None:
            nbits = max(4, int(np.ceil(np.log(max(n, 2)) / np.log(b))) + 2)
            mask = int(gen.integers(0, 1 << min(30, nbits)))
        out[:, j] = _radical_inverse_masked(np.arange(start, start + n), b, mask)
    return out


# --------------------------------------------------------------------------- #
# 秩一格点（"格点"）
# --------------------------------------------------------------------------- #
def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def korobov_generator(n: int, dim: int, *, candidates: int = 12, probes: int = 512) -> np.ndarray:
    """Korobov 生成向量 :math:`z=(1,a,a^2,\\dots,a^{d-1})\\bmod n`，``a`` 由**实测**准则挑。

    准则是"环面最小距离最大化"（maximin）：候选 ``a`` 取与 ``n`` 互素的若干个值，
    取使 :math:`\\min_{k\\ne0}\\|kz/n\\|_{\\rm torus}` 最大的那个。判据本身是廉价的近似
    （只探 ``probes`` 个 :math:`k`，而格点的最小距离通常在小 :math:`k` 处取到）——
    写进 docstring 是因为"用近似判据挑出来的向量"与"最优生成向量"不是一回事。
    """
    if n < 3:
        return np.ones(dim, dtype=np.int64)
    # 候选只取"稀疏铺开"的 O(candidates) 个（不要 range(2, n) 全扫：n = 2^16 时那是
    # 几万次 gcd 调用，而候选质量靠下面的实测准则保证，不靠候选数量）。
    step = max(1, n // (2 * candidates))
    cands = [a for a in range(3, n, max(1, step // 2 * 2 - 1)) if _gcd(a, n) == 1][:candidates]
    if not cands:
        cands = [a for a in range(3, n) if _gcd(a, n) == 1][:1]
    best_a, best_dist = cands[0], -1.0
    for a in cands:
        gen = np.array([pow(a, j, n) for j in range(dim)], dtype=np.int64)
        d = _lattice_min_torus_distance(n, gen, probes=probes)
        if d > best_dist:
            best_dist, best_a = d, a
    return np.array([pow(best_a, j, n) for j in range(dim)], dtype=np.int64)


def _lattice_min_torus_distance(n: int, z: np.ndarray, *, probes: int = 512) -> float:
    r"""秩一格点的最小环面距离 :math:`\min_{k}\|kz/n \bmod 1\|_2`（平移不变，只需算 :math:`kz`）。

    ``probes`` 限制探多少个 :math:`k`（全探是 :math:`O(n)`；这里默认只探 512 个，
    是**上界**估计——判据的近似性已在 :func:`korobov_generator` 里声明）。
    """
    kmax = max(2, int(n))
    if kmax - 1 > probes:
        k = np.unique(np.linspace(1, kmax - 1, probes).astype(np.int64))
    else:
        k = np.arange(1, kmax, dtype=np.int64)
    diff = np.mod(k[:, None] * np.asarray(z, dtype=np.int64)[None, :], n).astype(np.float64) / float(n)
    diff = np.minimum(diff, 1.0 - diff)
    return float(np.min(np.sqrt(np.sum(diff * diff, axis=1))))


def lattice_points(n: int, dim: int, *, generating: Any = None, seed: Any = None, shift: bool = True) -> np.ndarray:
    """秩一格点 :math:`x_k=\\{k z/n + \\Delta\\}`（``shift=True`` 时 :math:`\\Delta` 为随机移位）。

    随机移位（Cranley–Patterson）保持格点结构、给出无偏估计——和 Sobol' 的数字移位
    是同一件事的两种写法。``shift=False`` 时是确定性格点，同样**没有误差估计**。
    """
    if n < 2 or dim < 1:
        raise ValueError("n 必须 >= 2，dim >= 1")
    z = korobov_generator(n, dim) if generating is None else np.asarray(generating, dtype=np.int64)
    z = np.asarray(z, dtype=np.int64).ravel()[:dim]
    if z.size < dim:
        raise ValueError("generating 的维数不足")
    k = np.arange(n, dtype=np.int64)[:, None]
    out = np.mod(k * z[None, :], n).astype(np.float64) / float(n)
    if shift:
        gen = _as_generator(seed)
        out = np.mod(out + gen.random((1, dim)), 1.0)
        out = np.mod(out, 1.0)
    return out


def tensor_grid(n_per_side: int, dim: int = 2) -> np.ndarray:
    """张量格点（``n_per_side^dim`` 个点，逐维等距）。点数随维数**指数**增长——这正是它的陷阱。"""
    if n_per_side < 2 or dim < 1:
        raise ValueError("n_per_side >= 2，dim >= 1")
    axes = [(np.arange(n_per_side) + 0.5) / n_per_side for _ in range(dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def is_power_of_two(n: int) -> bool:
    """``n`` 是否为 2 的幂（Sobol' 的网性质只对 :math:`2^m` 个点成立）。"""
    return n >= 1 and (n & (n - 1)) == 0


def point_set(method: str, n: int, dim: int, *, seed: Any = None, randomize: bool = False) -> np.ndarray:
    """统一的点集入口：``mc`` / ``sobol`` / ``lattice`` / ``halton`` / ``grid``。

    ``randomize=True`` 时给该方法的**随机化版本**（MC 本身就是随机的；Sobol 用线性置乱；
    格点用随机移位；Halton 用数字移位）；张量格点没有随机化版本，``grid`` + ``randomize``
    会抛错而不是悄悄返回一个"看起来也能算误差"的东西。
    """
    m = str(method).lower()
    if m == "mc":
        return _as_generator(seed).random((n, dim))
    if m == "sobol":
        return sobol_points(n, dim, seed=seed, scramble="linear" if randomize else "none")
    if m == "halton":
        return halton_points(n, dim, seed=seed, scramble="shift" if randomize else "none")
    if m == "lattice":
        return lattice_points(n, dim, seed=seed, shift=bool(randomize))
    if m == "grid":
        if randomize:
            raise ValueError("张量格点没有随机化版本：它没有可用的误差估计（这是它的缺陷，不是缺省值）")
        side = int(round(n ** (1.0 / dim)))
        side = max(2, side)
        return tensor_grid(side, dim)
    raise ValueError(f"未知点集方法 {method!r}")
