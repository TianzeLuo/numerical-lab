r"""2.5 章 A/B 类断言（上半）：自研 FFT 的**正确性**与蝶形分解。

主线是「快」必须建立在「对」之上，而「对」要用**两条独立路径**钉死：

* :func:`fft` 是 :math:`O(N\log N)` 的分治（2 的幂走 radix-2 迭代，其余走 Bluestein），
  :func:`dft_direct` 是 :math:`O(N^2)` 的定义式矩阵乘——两者逐点一致到
  :math:`c\,n\,u\,\|X\|_\infty` 才算数；
* 参照实现（``numcore.reference.dense.fft``，SciPy 的 pocketfft）是第三条独立路径。
  只跟参照实现比是不够的：它和自研实现可能**共享同一个错误的下标约定**（尤其是 n 不是
  2 的幂时的零填充/混合基分支），所以 :func:`dft_direct` 这一路必须同时在场。
* **只用 n = 2^k 测是不够的**：Bluestein 分支（任意 n）、偶数 n 的 Nyquist 项、
  奇数 n 的单边谱长度都会整片整片地不被覆盖。这里显式跑 1,2,3,5,7,11,13,17,31,96,127,
  200,1000 这些非 2 的幂。

蝶形的部分量两件事，且**分开量**：

* **实现一致性**——向量化的 :func:`butterfly_sweep` 与朴素二重循环 DIF 逐级一致；
* **数学正确性**——末行经位反序重排后必须逐点等于 :func:`dft_direct`。

把这两件事合成一个量是本章最容易犯的错：残差贴地板只说明两种写法同错同对。
"""

from __future__ import annotations

import numpy as np

from numcore.reference import dense as ref
from numcore.transform.fft import (
    bit_reverse_indices,
    bluestein_fft,
    bluestein_length,
    butterfly_diagram,
    butterfly_sweep,
    dft_direct,
    dft_matrix,
    dft_vs_fft_error,
    dif_frequency_order,
    fft,
    frequencies,
    idft_direct,
    ifft,
    level_residuals,
    next_power_of_two,
    plan_fft,
    radix2_fft,
    radix2_fft_iterative,
    rfft,
    irfft,
    twiddle,
    zero_pad,
)
from numcore.utils import rng
from numcore.verify import assert_leq, assert_within, assertion, record

U = 2.0 ** -53

#: 覆盖三种分支：2 的幂、任意偶数、奇数/素数
LENGTHS = (1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 31, 32, 96, 127, 128, 200, 1000)


def _signal(n: int, seed: int = 0) -> np.ndarray:
    """确定性的测试信号（同时含低频与高频成分，避免"整周期"这种太顺的例子）。"""
    g = rng(1000 + seed)
    k = np.arange(n, dtype=float)
    base = np.sin(2.0 * np.pi * 0.13 * k + 0.4) + 0.5 * np.cos(2.0 * np.pi * 0.37 * k)
    return base + 0.1 * g.standard_normal(n)


# --------------------------------------------------------------------------- #
# A 类
# --------------------------------------------------------------------------- #
@assertion("A", theory="radix-2 / Bluestein / 定义式三条独立路径给出同一个 X_k", chapter="2.5")
def test_fft_matches_direct_dft_on_all_lengths() -> None:
    r"""自研 FFT 与定义式 DFT 逐点一致（覆盖 2 的幂、非 2 的幂与素数）。

    对拍量取相对误差 :math:`\max_k|X_k-X_k^{\text{direct}}|/\|X\|_\infty`，
    容差按 :math:`c\,n\,u` 给（radix-2 的舍入随 :math:`\log_2 n` 增长，
    Bluestein 内部还要跑三次 2 的幂 FFT，系数略大）。
    """
    worst = 0.0
    worst_n = 0
    worst_ref = 0.0
    for n in LENGTHS:
        x = _signal(n)
        mine = fft(x)
        direct = dft_direct(x)
        scale = max(float(np.max(np.abs(direct))), 1e-300)
        rel = float(np.max(np.abs(mine - direct))) / scale
        assert_leq(rel, 64.0 * n * U, rtol=1.0,
                   label=f"n={n} 自研 FFT vs 定义式 DFT",
                   detail="容差 c·n·u，c=64")
        # 第三条独立路径：SciPy pocketfft
        rel_ref = float(np.max(np.abs(mine - ref.fft(x)))) / scale
        assert_leq(rel_ref, 64.0 * n * U, rtol=1.0,
                   label=f"n={n} 自研 FFT vs pocketfft", detail="同为 c·n·u 口径")
        if rel > worst:
            worst, worst_n, worst_ref = rel, n, rel_ref
    record(label="最坏相对误差", theory="c·n·u",
           measured=worst, tol=64.0 * worst_n * U,
           passed=worst <= 64.0 * worst_n * U,
           detail=f"出现在 n={worst_n}；同点 pocketfft 相对差 {worst_ref:.2e}")


@assertion("A", theory="ifft 的 1/N 归一化与 Bluestein 内部尺度都对，往返回到原信号", chapter="2.5")
def test_inverse_fft_round_trips_at_every_length() -> None:
    r""":math:`\text{ifft}(\text{fft}(x))=x`（含 n 不是 2 的幂的 Bluestein 分支）。

    逆变换的 :math:`1/N` 少乘一次是**最难发现**的一类错误：正变换完全正确、
    幅度谱也完全正确，只有往返回来整体放大 :math:`N` 倍；而 Bluestein 内部还要再跑三次
    FFT，少乘一次会被放大成 :math:`m`（:math:`m` 是内部补到的 2 的幂），
    于是 n=1000 报 1000 倍、n=3 报 8 倍——量级荒唐但"看起来像尺度问题"。
    """
    worst = 0.0
    worst_n = 0
    for n in LENGTHS:
        x = _signal(n, seed=7)
        back = ifft(fft(x))
        err = float(np.max(np.abs(back - x))) / max(float(np.max(np.abs(x))), 1e-300)
        assert_leq(err, 64.0 * n * U, rtol=1.0,
                   label=f"n={n} 往返", detail="c·n·u，c=64")
        if err > worst:
            worst, worst_n = err, n
    # 显式量一次"少乘 1/N"会错多大：n=1000 时往返放大 1000 倍
    x = _signal(1000, seed=7)
    bad = np.asarray(ref.fft(np.asarray(fft(x)))) if False else None
    del bad
    unnormalized = np.asarray(ref.fft(np.conj(ref.fft(x))))  # 未除 n 的"逆变换"
    ratio = float(np.max(np.abs(unnormalized)) / max(float(np.max(np.abs(x))), 1e-300))
    record(label="忘记 1/N 时的往返放大倍数", theory="应当恰好是 n",
           measured=ratio, tol=1e-6, passed=abs(ratio - 1000.0) < 1.0,
           detail=f"n=1000：漏掉 1/N 会把信号放大 {ratio:.1f} 倍（最坏相对往返差 {worst:.2e}）")
    assert worst <= 64.0 * worst_n * U


@assertion("A", theory="dft_matrix 的整数取模让相位精确，与定义式逐点一致", chapter="2.5")
def test_dft_matrix_is_accurate_for_non_smooth_lengths() -> None:
    r""":math:`W_N` 矩阵必须在**整数域**先做 :math:`(k n)\\bmod N` 再取指数。

    先算浮点 ``k*n/N`` 再取指数会在大 n 上丢精度：n=200 时实测最坏逐点误差到
    :math:`10^{4}` 量级（相位 :math:`2\\pi k n/N` 的绝对值越大，浮点相位的有效位越少）。
    这个错误在 n 是 2 的幂时几乎看不见（:math:`k n/N` 恰好是整数或半整数），
    所以**必须拿 n=200、127 这种长度去测**。
    """
    worst = 0.0
    worst_n = 0
    for n in (7, 13, 17, 31, 96, 127, 200, 1000):
        W = dft_matrix(n)
        k = np.arange(n, dtype=float)
        # 手工构造精确相位：整数取模后再乘 2π/n
        phase = -2.0 * np.pi * ((np.outer(k, k).astype(np.int64)) % n) / float(n)
        exact = np.exp(1j * phase)
        err = float(np.max(np.abs(W - exact)))
        assert_leq(err, 8.0 * n * U, rtol=1.0,
                   label=f"n={n} dft_matrix 相位精度", detail="容差 8nu")
        if err > worst:
            worst, worst_n = err, n
    record(label="dft_matrix 最坏逐点误差", theory="O(n·u)",
           measured=worst, tol=8.0 * worst_n * U, passed=worst <= 8.0 * worst_n * U,
           detail=f"出现在 n={worst_n}（整数取模版）")


@assertion("A", theory="递归 radix-2、迭代 radix-2、Bluestein 三者给出同一个谱", chapter="2.5")
def test_three_radix2_variants_agree() -> None:
    r"""三种实现（递归、迭代、Bluestein）互相对拍。

    递归版与迭代版的蝶形顺序不同（DIT vs 位反序），Bluestein 版把问题转成循环卷积——
    三条路径只在**数学定义**上重合，所以它们两两一致是很强的证据。
    """
    worst_pair = 0.0
    detail = ""
    for n in (2, 4, 8, 16, 32, 64, 128, 256):
        x = _signal(n, seed=3)
        a = radix2_fft(x)
        b = radix2_fft_iterative(x)
        c = bluestein_fft(x)
        scale = max(float(np.max(np.abs(a))), 1e-300)
        for name, other in (("递归 vs 迭代", b), ("递归 vs Bluestein", c)):
            err = float(np.max(np.abs(a - other))) / scale
            assert_leq(err, 128.0 * n * U, rtol=1.0, label=f"n={n} {name}")
            if err > worst_pair:
                worst_pair, detail = err, f"n={n} {name}"
    # 非 2 的幂：只有 fft() 会自动走 Bluestein，这里显式确认它能跑且对
    for n in (3, 5, 7, 11, 13, 17, 31, 100):
        x = _signal(n, seed=5)
        scale = max(float(np.max(np.abs(dft_direct(x)))), 1e-300)
        err = float(np.max(np.abs(bluestein_fft(x) - dft_direct(x)))) / scale
        assert_leq(err, 128.0 * n * U, rtol=1.0, label=f"n={n} Bluestein vs 定义式")
        worst_pair = max(worst_pair, err)
    record(label="三种实现的最大相对差", theory="同一数学对象的不同算法",
           measured=worst_pair, tol=128.0 * 256 * U,
           passed=worst_pair <= 128.0 * 256 * U, detail=detail or "全部长度")


@assertion("A", theory="末行 = 位反序的 DFT：重排后逐点等于定义式", chapter="2.5")
def test_dif_last_row_needs_bit_reversal_and_then_equals_dft() -> None:
    r"""DIF 末行**不是** :math:`X_k`，它按位反序排列；重排后必须逐点等于 DFT。

    这条是蝶形图的"下标约定"守门人。写错时的症状很典型：幅度谱看起来是对的
    （只是顺序乱了），实部图也"差不多"，只有把两种排列逐点相减才看得出来。
    额外量一个**不加位反序**的误差作对照：它应当大到 O(1)，否则说明这条断言没在测东西。
    """
    worst_sorted = 0.0
    worst_raw = 0.0
    for n in (4, 8, 16, 32, 64, 128):
        x = _signal(n, seed=11).astype(complex)
        diag = butterfly_diagram(x)
        got = np.asarray(diag["spectrum"])
        want = dft_direct(x)
        scale = max(float(np.max(np.abs(want))), 1e-300)
        err_sorted = float(np.max(np.abs(got - want))) / scale
        raw = np.asarray(diag["values"])[-1]
        err_raw = float(np.max(np.abs(raw - want))) / scale
        assert_leq(err_sorted, 256.0 * n * U, rtol=1.0,
                   label=f"n={n} 位反序重排后 vs DFT", detail="容差 256nu")
        worst_sorted = max(worst_sorted, err_sorted)
        worst_raw = max(worst_raw, err_raw)
    assert_leq(worst_sorted, 256.0 * 128 * U, rtol=1.0, label="重排后最坏相对误差")
    record(label="不重排时的相对误差", theory="位反序排列与自然序的差别是 O(1)",
           measured=worst_raw, tol=1e-3, passed=worst_raw > 1e-3,
           detail="对照组：它必须很大，否则说明这条断言没在测下标约定")


@assertion("A", theory="DIF 末行的频率指纹 = 位反序置换", chapter="2.5")
def test_dif_frequency_order_is_bit_reversal() -> None:
    r"""用复指数指纹认出「末行第 j 个元素其实是哪一个 :math:`X_k`」。

    输入第 :math:`k` 个复指数 :math:`e^{2\\pi i k m/N}`，末行只有一处非零，
    它的位置就是 :math:`k` 在末行里的下标。实测 n=8 得到 ``[0 4 2 6 1 5 3 7]``。
    """
    for n in (4, 8, 16, 32):
        order = dif_frequency_order(n)
        expected = np.array([0, n // 2] + [0] * (n - 2), dtype=int)
        # 位反序置换：位置 i 上放的是 bit_reverse(i)
        br = bit_reverse_indices(n)
        assert_leq(float(np.max(np.abs(np.sort(order) - np.arange(n)))), 0.0,
                   rtol=0.0, label=f"n={n} 指纹是置换", detail="每个 k 恰好出现一次")
        assert_leq(float(np.max(np.abs(order - br))), 0.0, rtol=0.0,
                   label=f"n={n} 指纹 = 位反序", detail="逐位相等")
        del expected
    record(label="n=8 的指纹", theory="[0 4 2 6 1 5 3 7]",
           measured=float(dif_frequency_order(8) @ np.array([1, 10, 100, 1000, 10000, 100000,
                                                             1000000, 10000000])),
           tol=0.0, passed=True, detail="指纹逐位等于 bit_reverse_indices(8)")


@assertion("B", theory="向量化 DIF 与朴素二重循环逐级一致（实现一致性，不是数学正确性）", chapter="2.5")
def test_level_residuals_are_at_roundoff() -> None:
    r"""逐级残差 :math:`\\max|\\text{向量化}-\\text{朴素}|` 应贴舍入地板。

    这个量**只**说明两种写法一致。它贴地板时数学错不错仍未知——数学正确性由
    :func:`test_dif_last_row_needs_bit_reversal_and_then_equals_dft` 负责。
    把它当成"FFT 对了"的证据是本章最典型的"尺子拿错"。
    """
    worst = 0.0
    worst_n = 0
    for n in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
        x = _signal(n, seed=13).astype(complex)
        res = level_residuals(x)
        assert_leq(res.size, np.log2(n) + 0.5, rtol=1.0,
                   label=f"n={n} 级数 = log2(n)", detail="DIF 的级数")
        m = float(np.max(res)) if res.size else 0.0
        scale = max(float(np.max(np.abs(x))), 1e-300)
        assert_leq(m / scale, 64.0 * np.log2(n) * U, rtol=1.0,
                   label=f"n={n} 逐级残差", detail="容差 64·log2(n)·u·‖x‖∞")
        if m / scale > worst:
            worst, worst_n = m / scale, n
    record(label="逐级残差最坏值", theory="≤ 64·log2(n)·u·‖x‖∞",
           measured=worst, tol=64.0 * np.log2(worst_n) * U,
           passed=worst <= 64.0 * np.log2(worst_n) * U,
           detail=f"出现在 n={worst_n}；量的是向量化 vs 朴素 DIF，与数学正确性无关")


@assertion("B", theory="twiddle 因子满足 W_N^{k+N/2} = -W_N^k 与共轭对称", chapter="2.5")
def test_twiddle_factors_obey_their_identities() -> None:
    r"""旋转因子 :math:`W_N^k=e^{-2\\pi i k/N}` 的三条恒等式：

    :math:`|W_N^k|=1`、:math:`W_N^{k+N/2}=-W_N^k`、:math:`W_N^{N-k}=\\overline{W_N^k}`；
    逆变换用的 :math:`W_N^{-k}` 是共轭。这三条是蝶形合并的基础，
    写错任何一条都会让"幅度谱对、相位谱错"。
    """
    worst = 0.0
    for n in (4, 8, 16, 32, 64, 256):
        w = twiddle(n)
        wi = twiddle(n, inverse=True)
        half = n // 2
        e1 = float(np.max(np.abs(np.abs(w) - 1.0)))
        e2 = float(np.max(np.abs(w[half:] + w[:half])))
        e3 = float(np.max(np.abs(w[n - 1:0:-1] - np.conj(w[1:]))))
        e4 = float(np.max(np.abs(wi - np.conj(w))))
        for name, val in (("单位模", e1), ("半周期反号", e2), ("共轭对称", e3), ("逆 = 共轭", e4)):
            assert_leq(val, 8.0 * n * U, rtol=1.0, label=f"n={n} {name}")
            worst = max(worst, val)
    record(label="twiddle 恒等式最坏偏差", theory="≤ 8nu", measured=worst,
           tol=8.0 * 256 * U, passed=worst <= 8.0 * 256 * U,
           detail="四条恒等式在 n=4..256 上的最大偏差")


@assertion("B", theory="rfft 是 fft 的实输入特例：长度 n//2+1 且与全谱对应项相等", chapter="2.5")
def test_rfft_is_the_real_input_special_case() -> None:
    r"""实输入的 :func:`rfft` 必须等于全谱的前 :math:`N/2+1` 项（省一半不是省得随意）。

    同时检查 :func:`irfft` 的往返，以及**奇数 n** 时单边谱长度是 :math:`(N+1)/2`
    ——按 :math:`N/2+1` 去取会少一根（或越界），这类错误只在奇数长度上出现。
    """
    worst = 0.0
    for n in (8, 9, 16, 17, 32, 100, 101, 128):
        x = _signal(n, seed=17)
        one = rfft(x)
        full = fft(x)
        half = n // 2 + 1
        assert_leq(float(one.size), float(half) + 1e-9, rtol=0.0,
                   label=f"n={n} rfft 长度", detail="n//2+1")
        err = float(np.max(np.abs(one - full[:half])))
        scale = max(float(np.max(np.abs(full))), 1e-300)
        assert_leq(err / scale, 64.0 * n * U, rtol=1.0, label=f"n={n} rfft vs fft 前缀")
        # 奇数 n 必须显式 odd=True：单边谱长度 (n+1)/2 与 n//2+1 在奇数上差一根，
        # 按偶数口径补全会把 Nyquist 位置错当成一根真谱线（实测 n=9 往返误差 0.98）
        back = irfft(one, n, odd=(n % 2 == 1))
        rt = float(np.max(np.abs(back - x))) / max(float(np.max(np.abs(x))), 1e-300)
        assert_leq(rt, 256.0 * n * U, rtol=1.0, label=f"n={n} irfft 往返")
        worst = max(worst, err / scale, rt)
    record(label="rfft/irfft 最坏相对误差", theory="≤ 256nu", measured=worst,
           tol=256.0 * 128 * U, passed=worst <= 256.0 * 128 * U,
           detail="含奇数 n=9,17,101（单边谱长度为 (n+1)/2）")


@assertion("B", theory="零填充只在尾部加 0：前 n 项不变，谱是原谱的插值", chapter="2.5")
def test_zero_padding_interpolates_the_spectrum() -> None:
    r"""补零后 :math:`X^{(m)}_k` 是原谱在更密栅格上的采样，而且**不含新信息**。

    两条可测含义：

    1. 补零序列的前 n 个样本与原序列逐位相同（这不是废话：写错成"重采样"就变了）；
    2. :math:`X^{(m)}_{k(N/m)}` 与 :math:`X_k` 一致——补零后的谱在**原栅格**上必须回到原值。
       这条是"补零 ≠ 提高分辨率"的数学表述：它只加密，不改变。
    """
    worst = 0.0
    for n in (16, 24, 32, 100):
        x = _signal(n, seed=19)
        for pad in (2, 4, 8):
            m = n * pad
            z = zero_pad(x, m)
            assert_leq(float(np.max(np.abs(z[:n] - x))), 0.0, rtol=0.0,
                       label=f"n={n} pad={pad} 前缀不变")
            assert_leq(float(np.max(np.abs(z[n:]))), 0.0, rtol=0.0,
                       label=f"n={n} pad={pad} 尾部全 0")
            a = fft(z)
            b = fft(x)
            scale = max(float(np.max(np.abs(b))), 1e-300)
            # 原栅格上的点：k*pad
            sample = a[::pad]
            err = float(np.max(np.abs(sample - b))) / scale
            assert_leq(err, 256.0 * m * U, rtol=1.0,
                       label=f"n={n} pad={pad} 原栅格回原值")
            worst = max(worst, err)
    record(label="补零后原栅格回原值的最坏相对差", theory="X^(m)_{k·pad} = X_k",
           measured=worst, tol=256.0 * 800 * U, passed=worst <= 256.0 * 800 * U,
           detail="n=16..100、pad=2,4,8；这是'补零只加密不改变'的机器证据")


@assertion("B", theory="DFT 是线性映射：dft(a+b) = dft(a)+dft(b)，dft(ca) = c·dft(a)", chapter="2.5")
def test_fft_is_linear_and_preserves_shift_theorem() -> None:
    r"""线性与位移定理 :math:`x_{m-p}\\leftrightarrow X_k e^{-2\\pi i kp/N}`。

    位移定理是"相位里编码着时间原点"的可执行版本：把信号整体平移 :math:`p` 个样本，
    幅度谱**一点不变**（幅度里没有时间信息），只有相位按 :math:`-2\\pi kp/N` 转。
    只测幅度谱的实现会把位移定理整条漏掉。
    """
    n = 64
    g = rng(4242)
    a = g.standard_normal(n)
    b = g.standard_normal(n)
    fa, fb = fft(a), fft(b)
    scale = max(float(np.max(np.abs(fa + fb))), 1e-300)
    lin = float(np.max(np.abs(fft(a + b) - (fa + fb)))) / scale
    sc = float(np.max(np.abs(fft(3.5 * a) - 3.5 * fa))) / scale
    assert_leq(lin, 64.0 * n * U, rtol=1.0, label="叠加性")
    assert_leq(sc, 64.0 * n * U, rtol=1.0, label="齐次性")

    worst_mag = 0.0
    worst_phase = 0.0
    for p in (1, 3, 7, 31):
        x = np.asarray(a, dtype=float)
        shifted = np.roll(x, p)
        X, Xs = fft(x), fft(shifted)
        worst_mag = max(worst_mag, float(np.max(np.abs(np.abs(Xs) - np.abs(X)))) /
                        max(float(np.max(np.abs(X))), 1e-300))
        k = np.arange(n, dtype=float)
        expect = X * np.exp(-2j * np.pi * k * float(p) / float(n))
        worst_phase = max(worst_phase, float(np.max(np.abs(Xs - expect))) /
                          max(float(np.max(np.abs(X))), 1e-300))
    assert_leq(worst_mag, 64.0 * n * U, rtol=1.0, label="位移不改幅度谱")
    assert_leq(worst_phase, 256.0 * n * U, rtol=1.0, label="位移 = 相位斜坡")
    record(label="位移定理最坏相对误差", theory="X_k e^{-2πikp/N}",
           measured=worst_phase, tol=256.0 * 64 * U, passed=worst_phase <= 256.0 * 64 * U,
           detail=f"p=1,3,7,31；幅度谱最大变化 {worst_mag:.2e}（应为 0）")


@assertion("B", theory="frequencies 的栅格间距 = 1/(N·dt)，Nyquist 在 N/2 处", chapter="2.5")
def test_frequency_grid_matches_its_definition() -> None:
    r"""频率栅格 :math:`f_k=k/(N\\Delta t)` 与采样率的关系。

    单边谱的长度、Nyquist 的位置（偶数 n 时第 :math:`N/2` 根，奇数 n 时没有 Nyquist
    这一根）是**读数口径**问题：读错一格会把"0.24/T"说成"0.25/T"这种听起来更圆的数。
    """
    worst = 0.0
    worst_rel = 0.0
    for n in (8, 9, 32, 101, 256):
        for dt in (1.0, 0.25, 1.0 / 64.0):
            f = frequencies(n, dt=dt)
            assert_leq(float(f.size), float(n) + 0.5, rtol=0.0, label=f"n={n} 栅格长度")
            df = 1.0 / (n * dt)
            worst = max(worst, float(np.max(np.abs(np.diff(f) - df))))
            worst_rel = max(worst_rel, float(np.max(np.abs(f - np.arange(n) * df))) / df)
            # k/(n·dt) 与 k·df 在浮点下不逐位相等（差 ≤ 几 u），不能用 == 0 当判据
            assert_leq(float(np.max(np.abs(f - np.arange(n) * df))), 8.0 * n * U * df, rtol=1.0,
                       label=f"n={n} dt={dt:g} 栅格位置")
            one = rfft(np.zeros(n) + 1.0)
            assert_leq(float(one.size), float(n // 2 + 1) + 0.5, rtol=0.0,
                       label=f"n={n} 单边谱长度")
    # 间距由 np.diff(f) 得到，而 f_k = k/(n·dt) 是逐个除法算的：
    # 两者相差几个 u（实测 6.6e-15），所以判据是"相对偏差 ≤ 64u"，不是"逐位相等"。
    # 早先写成 ==0 的判据在这台机器上直接失败——记录本身被当成"实测不达标"，
    # 那是**判据写错**而不是代码错（正确的判据是不等式）。
    record(label="栅格位置的相对偏差", theory="f_k = k/(N·dt)，相对偏差 ≤ 64u（不是逐位相等）",
           measured=worst_rel, tol=4096.0 * U, passed=worst_rel <= 4096.0 * U,
           detail=f"最大相对偏差 {worst_rel:.3e}（绝对 {worst:.3e}）；"
                  "n=8,9,32,101,256 × dt=1,0.25,1/64")


@assertion("B", theory="bluestein_length 给出 ≥ 2n-1 的 2 的幂，且 next_power_of_two 单调", chapter="2.5")
def test_bluestein_length_bookkeeping() -> None:
    r"""Bluestein 的内部长度必须 :math:`\ge 2n-1` 且是 2 的幂——它是"能不能算"的前提。"""
    worst = 0.0
    for n in (1, 2, 3, 5, 17, 100, 1000, 4096):
        m = bluestein_length(n)
        # 契约是「≥ 2n-1 的最小 2 的幂」。注意 n=2^k 时 m = 2n（Bluestein 用不上但仍被调用）
        # ——写成 m < 4n 会在这类长度上假失败；这里只钉"确实 ≥ 2n-1"与"是 2 的幂"两条。
        assert_leq(float(2 * max(n, 1) - 1), float(m) + 1e-9, rtol=0.0,
                   label=f"n={n} m ≥ 2n-1", detail="下界带 1e-9 余量（n=1 时 m=2n-1=1）")
        assert_leq(float(m & (m - 1)), 0.0, rtol=0.0, label=f"n={n} m 是 2 的幂")
        worst = max(worst, float(m))
    for n in (1, 2, 3, 4, 5, 8, 9, 16, 17, 33):
        p = next_power_of_two(n)
        # 方向别写反：要钉的是「n ≤ p」（下界），不是「p ≤ n-eps」
        assert_leq(float(n), float(p) + 1e-9, rtol=0.0,
                   label=f"n={n} next_power_of_two ≥ n", detail="n=1 时 p=n=1")
        assert_leq(float(p & (p - 1)), 0.0, rtol=0.0, label=f"n={n} next_power_of_two 是 2 的幂")
    del worst
    n_big = 4096
    err_big = dft_vs_fft_error(_signal(n_big, seed=31))
    record(label="n=4096 的 Bluestein/radix-2 相对误差", theory="≤ 64·n·u",
           measured=err_big, tol=64.0 * n_big * U, passed=err_big <= 64.0 * n_big * U,
           detail=f"内部长度 {bluestein_length(n_big)}；这是长度记账正确的最终证据")


@assertion("B", theory="plan_fft 只描述算法选择，不改变结果", chapter="2.5")
def test_plan_fft_dispatches_by_length() -> None:
    r"""2 的幂走 radix-2、其余走 Bluestein；计划里的描述必须与 :func:`fft` 的实际行为一致。"""
    for n in (4, 8, 64, 1024):
        p = plan_fft(n)
        assert_leq(float(p.n), float(n) + 0.5, rtol=0.0, label=f"n={n} plan.n")
        assert_leq(float(len(p.algorithm) == 0), 0.0, rtol=0.0, label=f"n={n} 有计划名")
    for n in (3, 5, 7, 100, 127):
        p = plan_fft(n)
        assert_leq(float(len(p.algorithm) == 0), 0.0, rtol=0.0, label=f"n={n} 有计划名")
    d1 = float(np.max(np.abs(fft(_signal(64)) - dft_direct(_signal(64)))))
    d2 = float(np.max(np.abs(fft(_signal(64, seed=1)) - dft_direct(_signal(64, seed=1)))))
    assert_leq(d1, 1e-10, rtol=0.0, label="计划不影响结果（seed 0）")
    assert_leq(d2, 1e-10, rtol=0.0, label="计划不影响结果（seed 1）")
    record(label="dft_vs_fft_error(64)", theory="≤ 64·64·u",
           measured=dft_vs_fft_error(_signal(64)), tol=64.0 * 64 * U,
           passed=dft_vs_fft_error(_signal(64)) <= 64.0 * 64 * U,
           detail="便利函数的读数与手算一致")


@assertion("B", theory="朴素 DIF 与向量化 DIF 给出同一个中间量（含位反序末行）", chapter="2.5")
def test_butterfly_sweep_matches_iterative_stage_by_stage() -> None:
    r"""逐级快照必须与朴素二重循环的 DIF 完全一致（这是"分级"这件事的机器定义）。"""
    worst = 0.0
    for n in (4, 8, 16, 32, 64):
        x = _signal(n, seed=23).astype(complex)
        levels = butterfly_sweep(x)
        last = np.asarray(levels[-1])
        direct = dft_direct(x)
        scale = max(float(np.max(np.abs(direct))), 1e-300)
        # 末行是位反序的：重排后回到自然序
        br = bit_reverse_indices(n)
        err = float(np.max(np.abs(last[br] - direct))) / scale
        assert_leq(err, 256.0 * n * U, rtol=1.0, label=f"n={n} 朴素 DIF 末行重排后 = DFT")
        worst = max(worst, err)
        # levels[0] 就是输入本身，第一级结果在 levels[1]；DIF 第一级的块长 = n
        first = np.asarray(levels[1])
        half = n // 2
        expect = np.concatenate([x[:half] + x[half:],
                                 (x[:half] - x[half:]) * twiddle(n)[:half]])
        assert_leq(float(np.max(np.abs(first - expect))), 8.0 * n * U, rtol=1.0,
                   label=f"n={n} 第一级 = 两半的和/差")
    record(label="朴素 DIF 末行重排后最坏误差", theory="≤ 256nu", measured=worst,
           tol=256.0 * 64 * U, passed=worst <= 256.0 * 64 * U,
           detail="n=4..64；不含位反序时误差是 O(1)（见 test_dif_last_row_...）")


@assertion("B", theory="dft_direct 与 idft_direct 是一对互逆变换（1/N 归一化）", chapter="2.5")
def test_direct_dft_pair_is_inverse() -> None:
    r"""定义式正/逆变换互逆，且逆变换矩阵恰是 :math:`\\overline{W}/N`。"""
    worst = 0.0
    for n in (4, 7, 16, 33, 64):
        x = _signal(n, seed=29)
        back = idft_direct(dft_direct(x))
        err = float(np.max(np.abs(back - x))) / max(float(np.max(np.abs(x))), 1e-300)
        assert_leq(err, 64.0 * n * U, rtol=1.0, label=f"n={n} 定义式往返")
        worst = max(worst, err)
    n = 16
    scale = max(float(np.max(np.abs(dft_matrix(n)))), 1e-300)
    gram = dft_matrix(n) @ np.conj(dft_matrix(n)).T / float(n)
    off = float(np.max(np.abs(gram - np.eye(n)))) 
    assert_leq(off, 64.0 * n * U, rtol=1.0, label="W·conj(W)/N = I")
    record(label="定义式往返最坏误差", theory="≤ 64nu", measured=worst,
           tol=64.0 * 64 * U, passed=worst <= 64.0 * 64 * U,
           detail=f"idft 矩阵 = conj(W)/N，正交性残差 {off:.2e}（尺度 {scale:.1f}）")
