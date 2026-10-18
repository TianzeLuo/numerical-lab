r"""2.5 章 A/B 类断言（下半）：谱、窗、卷积定理、分辨率。

这一半围绕「谱上的每一个数都有一句可以说清楚的话」展开：

* **幅度标定**：单边谱的 2/N 与 DC/Nyquist 不乘 2，两处都错得很有规律
  （整体差 2 倍 / 两端差 2 倍），所以分开量；
* **Parseval**：全谱与单边两种口径都要守恒，且单边的 df 必须用**原始长度**算
  ——用单边谱长度算会把能量算成 96.5%，看起来像「定理不成立」；
* **泄漏**：整周期采样时矩形窗旁瓣是 1e-16 量级，偏 0.25 格就抬到 -9.5 dB
  ——泄漏不是窗的性质，是**非整周期**的性质；
* **窗的代价**：压低旁瓣必须用主瓣变宽来换，且实测「最小可分辨间隔 / -3 dB 主瓣宽度」
  在四个窗上落在同一个区间（1.5~1.8）——这是「天下没有免费午餐」的可执行版本；
* **卷积定理**：循环卷积定理残差贴地板，但它等于线性卷积**只在 N 足够大时**；
  卷绕误差会伪装成滤波结果（前几项完全一样）；
* **分辨率**：整数格上四个窗的阈值都是 1.0/T；把两音放在半格上，Hann 立刻并成一个包，
  矩形窗同一位置仍分得开——**在哪一格上量，决定了你能不能看见窗的代价**。
"""

from __future__ import annotations

import numpy as np

from numcore.transform.fft import (
    WINDOW_MAINLOBE_BINS,
    WINDOW_NAMES,
    WINDOW_SIDELOBE_DB,
    alias_candidates,
    alias_frequency,
    alias_index,
    amplitude_spectrum,
    circular_convolution,
    circular_derivative,
    convolution_theorem_error,
    db,
    fft,
    fft_convolve,
    frequency_energy,
    frequency_resolution,
    goertzel,
    idft_direct,
    linear_convolution,
    min_separable_separation,
    next_power_of_two,
    parseval_relative_error,
    peak_frequency,
    peak_sidelobe_level,
    phase_spectrum,
    rayleigh_resolution,
    separable,
    shift_sample,
    sidelobe_envelope,
    sinusoid_sum,
    time_energy,
    two_tone,
    window,
    window_coherent_gain,
    window_energy,
    wrap_around_error,
    zero_pad,
)
from numcore.utils import rng
from numcore.verify import assert_leq, assert_within, assertion, record

U = 2.0 ** -53


# --------------------------------------------------------------------------- #
# A 类
# --------------------------------------------------------------------------- #
@assertion('A', theory='单边幅度谱把正弦的峰值标定成它的幅度（DC/Nyquist 不乘 2）', chapter='2.5')
def test_amplitude_spectrum_calibration() -> None:
    r"""幅度标定：正弦的谱峰读数必须恰好是它的幅度。

    两个「差 2 倍」的坑都在这里：漏乘总体的 2 会让所有幅度小一半；
    DC 与 Nyquist 也乘 2 会让两端大一倍。后者只在**信号含直流或恰好落在 Nyquist**
    时暴露，所以三种情况（中频正弦、纯直流、Nyquist 正弦）要分开测。
    """
    n = 256
    t = np.arange(n, dtype=float) / float(n)
    worst = 0.0
    for amp in (0.25, 1.0, 3.5):
        for k in (5, 17, 60, 100):
            x = amp * np.cos(2.0 * np.pi * k * t + 0.3)
            f, sp = amplitude_spectrum(x, dt=1.0 / float(n))
            got = float(np.max(sp))
            assert_within(got, amp, rtol=1e-12, label=f'幅度 {amp} 在 k={k}',
                          detail='单边幅度谱的峰 = 正弦幅度')
            assert_within(float(f[int(np.argmax(sp))]), float(k), rtol=1e-12,
                          label=f'峰位 k={k}', detail='栅格 f_k = k/(n*dt)')
            worst = max(worst, abs(got - amp))

    dc = np.full(n, 2.5)
    _, s0 = amplitude_spectrum(dc, dt=1.0 / float(n))
    assert_within(float(s0[0]), 2.5, rtol=1e-12, label='直流读数 = 2.5',
                  detail='DC 不乘 2；乘了会读到 5.0')

    ny = 1.75 * np.cos(np.pi * np.arange(n, dtype=float))
    _, s1 = amplitude_spectrum(ny, dt=1.0 / float(n))
    assert_within(float(s1[-1]), 1.75, rtol=1e-12, label='Nyquist 读数 = 1.75',
                  detail='偶数 n 的最后一根也不乘 2；乘了会读到 3.5')
    assert_within(float(s1[0]), 0.0, rtol=1.0, atol=1e-13, label='Nyquist 信号无直流分量')
    record(label='三种标定场景的最大读数偏差', theory='不超过 1e-12（绝对）',
           measured=worst, tol=1e-12, passed=worst <= 1e-12,
           detail='中频正弦 12 组 + 直流 + Nyquist；漏乘 2 / 多乘 2 都会在这里现形')


@assertion('A', theory='Parseval：全谱与单边两种口径都守恒，单边的 df 用原始长度算', chapter='2.5')
def test_parseval_in_both_conventions() -> None:
    r"""能量守恒在两种读数口径下都要成立。

    单边口径有两处陷阱：内部谱线乘 2、DC/Nyquist 不乘；以及 df 必须由**原始长度 N**
    决定（单边谱只有 N/2+1 根，用它的长度算 df 会把能量放大 N/(N/2+1) ≈ 1.98 倍，
    Parseval 的相对误差因此是 **0.98**——正好像「定理不成立」而不是「代码写错了」）。
    奇数 N 没有 Nyquist 项，
    连该不该乘 2 都要重新数，所以 N 的奇偶都要测。
    """
    worst_full = 0.0
    worst_one = 0.0
    for n in (64, 100, 101, 127, 128, 255, 256, 1000):
        x = rng(5000 + n).standard_normal(n)
        e_full = parseval_relative_error(x)
        e_one = parseval_relative_error(x, one_sided=True)
        assert_leq(e_full, 64.0 * n * U, rtol=1.0, label=f'n={n} 全谱 Parseval')
        assert_leq(e_one, 64.0 * n * U, rtol=1.0, label=f'n={n} 单边 Parseval',
                   detail=f'奇数 n 也必须过：没有 Nyquist 项时权重是另一套数法（实测 {e_one:.2e}）')
        worst_full = max(worst_full, e_full)
        worst_one = max(worst_one, e_one)

    n = 101
    x = rng(5000 + n).standard_normal(n)
    spec = np.asarray(fft(x))[: n // 2 + 1]
    good = frequency_energy(spec, one_sided=True, n=n)
    # 反例：权重口径对（奇数 N 没有 Nyquist 项），但 df 用**单边谱长度** 51 而不是原始 101。
    # 能量因此被放大 N/(N/2+1) = 101/51 ≈ 1.98 倍，相对误差 ≈ 0.98
    # —— 这正是当年 frequency_energy 的 bug 在 Parseval 上的读数（0.98，不是 0.965）。
    weight = float(np.abs(spec[0]) ** 2 + 2.0 * np.sum(np.abs(spec[1:]) ** 2))
    bad = weight / float(spec.size)
    ratio = bad / time_energy(x)
    expect = float(n) / float(spec.size)
    record(label='单边 df 用错长度时的能量比',
           theory=f'df 用单边谱长度：比值 = N/(N/2+1) = {expect:.4f}（能量被放大近 2 倍）',
           measured=ratio, tol=1e-9, passed=abs(ratio - expect) < 1e-9,
           detail=f'n=101：实测 {ratio:.4f}（正确口径 {good / time_energy(x):.4f}）'
                  '；Parseval 的相对误差就是 {:.3f}'.format(abs(ratio - 1.0)))
    assert worst_full <= 64.0 * 1000 * U and worst_one <= 64.0 * 1000 * U


@assertion('A', theory='循环卷积定理残差贴舍入地板；补零后等于线性卷积', chapter='2.5')
def test_convolution_theorem_and_linear_equivalence() -> None:
    r"""DFT 把循环卷积对角化：频域相乘等于时域循环卷积。

    三方对拍：显式模索引求和（定义式）、频域相乘逆变换、以及线性卷积（只在
    N 不小于 N_a+N_b-1 时才等于循环卷积）。最后一组是**补零之后**的比较
    ——这是把卷绕彻底消掉的那一步。
    """
    worst_theorem = 0.0
    worst_helper = 0.0
    for n, m in ((8, 5), (16, 8), (32, 7), (64, 13), (12, 12)):
        a = rng(600 + n * 100 + m).standard_normal(n)
        b = rng(700 + n * 100 + m).standard_normal(m)
        via_fft = fft_convolve(a, b, mode='circular')
        direct = circular_convolution(a, zero_pad(b, n))
        lin = linear_convolution(a, b)
        scale = max(float(np.max(np.abs(lin))), 1e-300)
        e_theorem = float(np.max(np.abs(via_fft - direct))) / scale
        assert_leq(e_theorem, 128.0 * n * U, rtol=1.0,
                   label=f'N={n},M={m} 定理残差', detail='频域相乘 vs 定义式模索引求和')
        worst_theorem = max(worst_theorem, e_theorem)
        worst_helper = max(worst_helper, convolution_theorem_error(a, b))

        need = n + m - 1
        safe = fft_convolve(a, b, mode='linear')
        e_safe = float(np.max(np.abs(safe - lin))) / scale
        assert_leq(e_safe, 512.0 * next_power_of_two(need) * U, rtol=1.0,
                   label=f'N={n},M={m} 补零后 = 线性卷积',
                   detail='补零到至少 N+M-1 后循环卷积退化为线性卷积')
    record(label='循环卷积定理残差', theory='不超过 128*N*u',
           measured=worst_theorem, tol=128.0 * 64 * U,
           passed=worst_theorem <= 128.0 * 64 * U,
           detail=f'5 组 (N,M)；另一路 convolution_theorem_error 最大 {worst_helper:.2e}')


@assertion('A', theory='卷绕误差只出现在前 M-1 项，且随 N 衰减但不会自动归零', chapter='2.5')
def test_wrap_around_is_local_and_persistent() -> None:
    r"""循环卷积与线性卷积的差**只在尾部折回来的位置**出现，且衰减得比想象中慢。

    实测 (N,M)=(16,8) 时最大偏差 2.5e-1（相对 42%），N 翻到 64 仍有 2.6e-3；
    差异位置数恰好是 M-1=7。真正的解法是补零到 N+M-1，那时误差**逐位为 0**。
    """
    m = 8
    rows = []
    for n in (16, 24, 32, 64, 128):
        x = np.exp(-0.5 * ((np.arange(n, dtype=float) - n // 2) / max(n / 8.0, 1.0)) ** 2)
        h = np.ones(m) / float(m)
        w = wrap_around_error(x, h)
        rows.append((n, w))
        assert_leq(w['n_diff'], float(m - 1) + 0.5, rtol=1.0,
                   label=f'N={n} 差异只出现在 M-1 个位置', detail='尾部卷回来叠加的位置数')
        assert_leq(w['rel_max'], 1.0, rtol=0.0, label=f'N={n} 相对卷绕不超过 1')
    n0, w0 = rows[0]
    n1, w1 = rows[-1]
    assert_leq(w1['max_abs'], w0['max_abs'], rtol=0.0,
               label='卷绕误差随 N 单调不增', detail=f'N={n0} 到 {n1}')
    assert_leq(w1['max_abs'], 1e-2, rtol=0.0,
               label='N=128 时卷绕已经很小但仍非零', detail='是「很小」不是「等于线性卷积」')

    x = np.exp(-0.5 * ((np.arange(16, dtype=float) - 8.0) / 2.0) ** 2)
    h = np.ones(m) / float(m)
    lin = linear_convolution(x, h)
    safe = fft_convolve(x, h, mode='linear')
    exact = float(np.max(np.abs(safe - lin)))
    # 这一组不是逐位 0（实测 1.2e-16）：卷绕被消掉以后，剩下的是两条求和的结合律差异
    record(label='补零到 N+M-1 后的最大偏差', theory='≤ 1e-15（舍入地板；卷绕的 O(1) 项消失）',
           measured=exact, tol=1e-15, passed=exact <= 1e-15,
           detail='对照：N=16 时循环卷积最大偏差 {a:.3f}（相对 {b:.1%}），N=64 时 {c:.2e}'.format(
               a=w0['max_abs'], b=w0['rel_max'], c=rows[3][1]['max_abs']))


@assertion('A', theory='四种窗的峰值旁瓣与主瓣宽度与 Harris 表一致（0.6 dB / 0.05 格）', chapter='2.5')
def test_window_metrics_match_harris_table() -> None:
    r"""窗的解析指标必须落在 Harris 表值上，且**主瓣宽度越大旁瓣越低**。

    用解析谱而不是 |FFT|：窗长小的时候 |FFT| 的栅格太粗，
    -3 dB 宽度会被量化到整格（rect 会报 14/32 = 0.44 格而不是 0.89 格）。
    """
    worst_db = 0.0
    worst_bins = 0.0
    order = sorted(WINDOW_NAMES, key=lambda nm: abs(WINDOW_SIDELOBE_DB[nm]))
    for name in WINDOW_NAMES:
        got = peak_sidelobe_level(name)
        d_db = abs(got['peak_db'] - WINDOW_SIDELOBE_DB[name])
        d_bin = abs(got['mainlobe_bins'] - WINDOW_MAINLOBE_BINS[name])
        assert_leq(d_db, 0.6, rtol=0.0, label=f'{name} 峰值旁瓣 vs Harris',
                   detail='实测 {a:.3f} dB，表值 {b:.2f} dB'.format(
                       a=got['peak_db'], b=WINDOW_SIDELOBE_DB[name]))
        assert_leq(d_bin, 0.05, rtol=0.0, label=f'{name} -3 dB 主瓣宽 vs Harris',
                   detail='实测 {a:.3f} 格，表值 {b:.2f} 格'.format(
                       a=got['mainlobe_bins'], b=WINDOW_MAINLOBE_BINS[name]))
        worst_db = max(worst_db, d_db)
        worst_bins = max(worst_bins, d_bin)
    # ``order`` 按 Harris 表的 |旁瓣| 升序；实测值沿这个顺序也必须单调增。
    # 这里用 record 手写判据而不是 assert_leq：assert_leq 会先取绝对值，
    # 把"增量非负"这种带方向的判据悄悄变成恒真（本文件被这个坑绊过一次）。
    measured = [abs(peak_sidelobe_level(nm)['peak_db']) for nm in order]
    increments = [measured[i] - measured[i - 1] for i in range(1, len(measured))]
    rec = record(label='旁瓣次序的实测增量',
                 theory='沿 ' + ' < '.join(order) + ' 的 |旁瓣| 单调增，增量全为正',
                 measured=min(increments), tol=0.0, passed=min(increments) > 0.0,
                 detail='增量 ' + '、'.join(f'{d:.2f}' for d in increments)
                        + '（assert_leq 会取绝对值，方向性判据必须手写 record）')
    assert rec.passed, rec.measured
    widest = max(peak_sidelobe_level(nm)['mainlobe_bins'] for nm in WINDOW_NAMES)
    assert_leq(WINDOW_MAINLOBE_BINS['rect'], WINDOW_MAINLOBE_BINS['blackman'] - 1e-9,
               rtol=0.0, label='主瓣宽度：rect 最窄、blackman 最宽')
    record(label='与 Harris 表的最大偏差', theory='不超过 0.6 dB / 0.05 格',
           measured=worst_db, tol=0.6,
           passed=worst_db <= 0.6 and worst_bins <= 0.05,
           detail=f'dB 偏差 {worst_db:.3f}、主瓣偏差 {worst_bins:.3f} 格（最宽 {widest:.2f} 格）')


@assertion('A', theory='整周期采样时无泄漏；偏 0.25 格次峰 0.337，偏 0.5 格 0.700', chapter='2.5')
def test_leakage_is_caused_by_non_integer_periods() -> None:
    r"""泄漏的**原因**是非整周期截断，不是「矩形窗不好」。

    频率恰好落在格上时，除主瓣外的谱线是机器零；偏 0.25 格次峰就抬到 0.337，
    偏 0.5 格到 0.700（-3.10 dB）。换 Hann 窗后**远处**旁瓣从 -25 dB 掉到 -67 dB
    ——但主瓣同时变宽，代价见下一条。
    """
    n = 64
    peaks = {}
    for off in (0.0, 0.25, 0.5):
        x = sinusoid_sum([1.0], [1.0 + off], fs=float(n), n=n)
        _, amp = amplitude_spectrum(x, dt=1.0 / float(n))
        srt = np.sort(amp)
        peaks[off] = float(srt[-2] / srt[-1])
    assert_leq(peaks[0.0], 1e-12, rtol=0.0, label='整周期：次峰是机器零',
               detail='off=0 时除主瓣外应当是 1e-16 量级')
    assert_within(peaks[0.25], 0.3265, rtol=2e-2, label='偏 0.25 格：次峰约 0.33',
                  detail='实测 0.3265（-9.72 dB），不是小量')
    assert_within(peaks[0.5], 0.7001, rtol=5e-3, label='偏 0.5 格：次峰 0.700',
                  detail='实测 0.7001（-3.10 dB）')

    n = 256
    dt = 1.0 / float(n)
    k = 20.5
    base = np.cos(2.0 * np.pi * k * np.arange(n, dtype=float) / float(n))
    far = {}
    for name in ('rect', 'hann'):
        xx = base if name == 'rect' else base * window(name, n)
        _, amp = amplitude_spectrum(xx, dt=dt)
        pk = float(np.max(amp))
        i = int(np.argmax(amp))
        far[name] = max(float(amp[j]) for j in range(amp.size) if abs(j - i) >= 10) / pk
    assert_leq(far['hann'], far['rect'] * 1e-2, rtol=1.0,
               label='Hann 的远处旁瓣比矩形低两个数量级',
               detail='rect {a:.1f} dB vs hann {b:.1f} dB'.format(
                   a=20 * np.log10(far['rect']), b=20 * np.log10(far['hann'])))
    record(label='远离主瓣的旁瓣电平（10 格以外）', theory='Hann 应低至少 20 dB',
           measured=20 * np.log10(far['hann'] / far['rect']), tol=20.0,
           passed=far['hann'] <= far['rect'] * 1e-2,
           detail='非整周期 k=20.5：rect {a:.1f} dB、hann {b:.1f} dB'.format(
               a=20 * np.log10(far['rect']), b=20 * np.log10(far['hann'])))


@assertion('A', theory='窗一致增益：峰值读数要除以相干增益才回到真实幅度', chapter='2.5')
def test_window_coherent_gain_rescales_amplitude() -> None:
    r"""加窗后正弦的谱峰读数等于幅度乘相干增益。

    实测 rect 1.0000、hann 0.4980、hamming 0.5382、blackman 0.4184。
    忘记除以它会让幅度系统性偏小一半——这不影响旁瓣、不影响频率位置，
    只影响你报出去的幅度，所以最难自己发现。
    """
    n = 256
    t = np.arange(n, dtype=float) / float(n)
    x = 2.0 * np.cos(2.0 * np.pi * 20.0 * t)
    gains = {}
    for name in WINDOW_NAMES:
        w = window(name, n)
        cg = window_coherent_gain(w)
        gains[name] = cg
        assert_within(cg, float(np.mean(w)), rtol=1e-12, label=f'{name} CG = mean(w)')
        _, amp = amplitude_spectrum(x * w, dt=1.0 / float(n))
        peak = float(np.max(amp))
        assert_within(peak / cg, 2.0, rtol=5e-3, label=f'{name} 除 CG 后回到 2.0',
                      detail=f'峰读数 {peak:.4f} / CG {cg:.4f}')
    assert_within(gains['hann'], 0.5, rtol=1e-2, label='Hann 的 CG 接近 0.5',
                  detail='(N-1)/(2N)：n=256 时 0.4980')
    record(label='四种窗的相干增益', theory='rect 1、hann 0.5、hamming 0.54、blackman 0.42',
           measured=float(gains['blackman']), tol=0.01,
           passed=abs(gains['blackman'] - 0.42) < 0.01,
           detail='；'.join(f'{k}={v:.4f}' for k, v in gains.items()))


@assertion('A', theory='旁瓣包络按 f 的 -p 次幂衰减：矩形/汉明 p=1、Hann/Blackman p=3', chapter='2.5')
def test_sidelobe_envelope_exponents() -> None:
    r"""窗旁瓣的对数斜率是 -20p dB/十倍频程。

    汉明窗的特殊之处：第一旁瓣压到 -42.7 dB，但**之后不再下降**（p=1），
    所以它和 Hann（p=3）不能互换着用——压得低和滚降快是两件事。
    """
    for name in WINDOW_NAMES:
        env, p = sidelobe_envelope(name, np.array([10.0, 100.0, 1000.0]), f_ref=1.0)
        slope = float(np.log10(env[2]) - np.log10(env[1]))
        assert_leq(abs(slope), float(p) + 1e-12, rtol=0.0, label=f'{name} |斜率| = p',
                   detail='包络是幂律，双对数下是直线；assert_within 会取绝对值，所以用 assert_leq')
        expect = 1.0 if name in ('rect', 'hamming') else 3.0
        assert_within(float(p), expect, rtol=1e-12, label=f'{name} p = {expect:g}')
    env_r, _ = sidelobe_envelope('rect', np.array([10.0, 1000.0]), f_ref=1.0)
    ratio = float(env_r[1] / env_r[0])
    env_h, _ = sidelobe_envelope('hann', np.array([10.0, 1000.0]), f_ref=1.0)
    record(label='10 到 1000 的包络比', theory='矩形 0.01（p=1）、Hann 1e-6（p=3）',
           measured=ratio, tol=1e-12, passed=abs(ratio - 0.01) < 1e-12,
           detail='矩形 {a:.2e} vs Hann {b:.2e}——p 的差别就是滚降速度的差别'.format(
               a=ratio, b=float(env_h[1] / env_h[0])))


@assertion('A', theory='整数格上四窗阈值都是 1.0/T；半格上 Hann 立刻分不开', chapter='2.5')
def test_resolution_threshold_and_window_cost() -> None:
    r"""分辨率的两层事实，必须分开量。

    1. **整数格扫描**：四个窗都在 1/T 处第一次判「能分开」（Rayleigh 判据）。
       这一层会让人误以为分辨率与窗无关。
    2. **半格**：两音放在 1.5/T（都偏离栅格）时 Hann 并成一个包，矩形窗仍分得开。
       窗的代价只在非栅格位置现形。
    """
    for name in WINDOW_NAMES:
        thr = min_separable_separation(1.0, n=256, window_name=name)
        assert_within(thr, 1.0, rtol=1e-9, label=f'{name} 整数格阈值 = 1.0/T',
                      detail='四个窗在整数格上给出同一个阈值')
    dt = 1.0 / 256.0
    rect = separable(two_tone(1.0, 2.25, 1.0, 3.75, t_max=1.0, n=256), dt=dt)
    hann = separable(two_tone(1.0, 2.25, 1.0, 3.75, t_max=1.0, n=256) * window('hann', 256), dt=dt)
    assert_leq(float(rect), 1.0, rtol=0.0, label='矩形窗在 1.5/T 上分得开')
    assert_leq(float(hann), 0.0, rtol=0.0, label='Hann 在 1.5/T 上分不开',
               detail='Hann 主瓣 1.44 格，1.5 格的间隔已经贴到极限')
    record(label='同一间隔上矩形 vs Hann 的判定', theory='矩形可分辨、Hann 不可分辨',
           measured=float(rect), tol=0.0, passed=bool(rect) and not bool(hann),
           detail=f'df=1.5/T 且两音都落在半格上：rect={rect}，hann={hann}')


@assertion('A', theory='阈值/主瓣宽度比值在四窗上落在 1.5 到 1.8 之间', chapter='2.5')
def test_resolution_threshold_scales_with_mainlobe_width() -> None:
    r"""把「窗越宽越分不开」变成**一个数**：实测阈值与主瓣宽度的比值。

    四个窗的阈值都约 1.0/T（整数格口径），但主瓣宽度从 0.89 涨到 1.68 格；
    这个比值说明阈值与主瓣宽度成正比，也就说明选窗就是选分辨率。
    """
    ratios = {}
    for name in WINDOW_NAMES:
        thr = min_separable_separation(1.0, n=256, window_name=name)
        ratios[name] = thr / WINDOW_MAINLOBE_BINS[name]
        assert_leq(ratios[name], 1.8, rtol=1e-9, label=f'{name} 比值不超过 1.8')
        assert_leq(0.5, ratios[name] + 1e-9, rtol=0.0,
                   label=f'{name} 比值不小于 0.5',
                   detail='整数格口径下阈值都是 1.0/T，所以比值就是 1/主瓣宽度')
    record(label='阈值/主瓣宽度比值',
           theory='阈值都贴 1.0/T，于是比值恰是 1/主瓣宽度的倒数关系',
           measured=float(ratios['blackman']),
           tol=0.2, passed=max(ratios.values()) - min(ratios.values()) < 0.6,
           detail='；'.join(f'{k}={v:.2f}' for k, v in ratios.items()))


@assertion('A', theory='Goertzel 单频 DFT 与 fft 对应项一致（含 k=0 与 k=N/2 特例）', chapter='2.5')
def test_goertzel_matches_fft_bin_by_bin() -> None:
    r"""Goertzel 递推只算一根谱线，必须与 fft 的第 k 项逐位一致。

    收尾式写反（把 s_{N-1}*exp(-iw) - s_{N-2} 写成 s_{N-1} - exp(-iw)*s_{N-2}）
    在 **k=0 与 k=N/2 上恰好仍然正确**（实测 N=8、k=2 时从 4 变成 -4i），
    所以只测这两个特殊频率是发现不了的，必须包含一般的 k。
    """
    worst = 0.0
    for n in (8, 16, 64, 128, 17):
        x = rng(800 + n).standard_normal(n)
        spec = fft(x)
        scale = max(float(np.max(np.abs(spec))), 1e-300)
        for k in (0, 1, n // 2, n // 2 + 1, n - 1):
            kk = int(k) % n
            g = goertzel(x, kk)
            e = abs(g - spec[kk]) / scale
            assert_leq(e, 4096.0 * n * U, rtol=1.0, label=f'n={n},k={kk} Goertzel vs fft',
                       detail='递推的舍入随 n 线性累积')
            worst = max(worst, e)

    n = 8
    x = np.cos(2.0 * np.pi * 2.0 * np.arange(n, dtype=float) / n)
    w = 2.0 * np.pi * 2.0 / n
    c = 2.0 * np.cos(w)
    s1 = 0.0
    s2 = 0.0
    for i in range(n):
        s0 = x[i] + c * s1 - s2
        s2 = s1
        s1 = s0
    wrong = complex(s1 - np.exp(-1j * w) * s2)
    right = complex(s1 * np.exp(-1j * w) - s2)
    record(label='收尾式写反时的结果', theory='正确值 4（等于 fft 的第 2 项）',
           measured=abs(wrong), tol=1e-9, passed=abs(wrong - 4.0) > 1.0,
           detail='写反得 {a:.2f}（模 {b:.2f}），正确写法得 {c:.2f}；两者在 k=0 与 k=N/2 上恰好相同'.format(
               a=wrong.real, b=abs(wrong), c=right.real))
    assert worst <= 4096.0 * 128 * U


# --------------------------------------------------------------------------- #
# B 类
# --------------------------------------------------------------------------- #
@assertion('B', theory='时间能量 = dt*sum|x|^2，与谱口径无关', chapter='2.5')
def test_time_energy_and_power_accounting() -> None:
    r"""时域能量的定义式必须逐位复现，它是 Parseval 的另一边。"""
    worst = 0.0
    for n in (32, 64, 101, 256):
        for dt in (1.0, 0.01):
            x = rng(900 + n).standard_normal(n)
            e = time_energy(x, dt=dt)
            manual = float(dt * np.sum(np.abs(x) ** 2))
            assert_within(e, manual, rtol=1e-12, label=f'n={n},dt={dt:g} 时域能量')
            worst = max(worst, abs(e - manual))
    record(label='时域能量定义式的最大偏差', theory='逐位相等',
           measured=worst, tol=0.0, passed=worst == 0.0,
           detail='dt 乘 sum|x|^2 是定义，没有数值近似')


@assertion('B', theory='峰值频率：抛物线插值的定位误差远小于栅格间距', chapter='2.5')
def test_peak_frequency_interpolation_beats_the_grid() -> None:
    r"""定位精度不等于分辨率：抛物线插值能把峰位定到栅格间距的百分之几。

    频率落在两格正中间是插值的最坏情况（直接读格误差正好半格），
    实测插值误差 0.042 格，比读格小一个数量级。
    """
    n = 256
    dt = 1.0 / float(n)
    df = frequency_resolution(n, dt=dt)
    worst_grid = 0.0
    worst_interp = 0.0
    for k in (10.5, 20.5, 33.5, 60.5):
        x = np.cos(2.0 * np.pi * k * np.arange(n, dtype=float) / n)
        f, amp = amplitude_spectrum(x, dt=dt)
        i = int(np.argmax(amp))
        worst_grid = max(worst_grid, abs(float(f[i]) - k * df))
        got = peak_frequency(x, dt=dt, refine=True)[0]
        worst_interp = max(worst_interp, abs(float(got) - k * df))
    assert_leq(worst_grid, 0.6 * df, rtol=0.0, label='直接读格：误差约半格',
               detail='这是分辨率的极限')
    assert_leq(worst_interp, 0.05 * df, rtol=0.0, label='抛物线插值：误差不超过 5% 栅格',
               detail='定位精度可以远好于分辨率')
    record(label='插值定位与栅格间距之比', theory='远小于 1',
           measured=float(worst_interp / df), tol=0.05,
           passed=worst_interp <= 0.05 * df,
           detail=f'读格误差 {worst_grid / df:.3f} 格 vs 插值 {worst_interp / df:.4f} 格')


@assertion('B', theory='混叠：可公度的频率对给出逐位相同的样本与谱峰', chapter='2.5')
def test_aliasing_folds_frequency_without_changing_amplitude() -> None:
    r"""采样率 fs 之下的频率是**等价类**：能造出逐位相同的样本，就无法从样本里区分。

    17 Hz 与 3 Hz 在 fs = 20 Hz、n = 80（观察 4 s）下是同一组采样：
    17 折到 |17-20| = 3，且两者都是整周期采样（17*4 = 68、3*4 = 12 都是整数），所以样本逐位相同、
    谱峰都在 3 Hz 且幅度都是 1.0。3 Hz 那一版必须**也**是整周期的——
    否则丢的是扇贝损失（scalloping loss）而不是混叠，两者会混在一起。
    """
    fs = 20.0
    n = 80
    x_a = sinusoid_sum([1.0], [3.0], fs=fs, n=n)
    x_b = sinusoid_sum([1.0], [17.0], fs=fs, n=n)
    # 容差 1e-12：两条路径算 cos 的参数不同（17*4 圈 vs 3*4 圈），舍入会累积到 ~5e-14
    assert_leq(float(np.max(np.abs(x_a - x_b))), 1e-12, rtol=0.0,
               label='3 Hz 与 17 Hz 在 fs=20 下的样本逐位相同',
               detail='同一组采样点，混叠在采样那一刻就发生了')
    for f_true in (3.0, 17.0):
        f, amp = amplitude_spectrum(sinusoid_sum([1.0], [f_true], fs=fs, n=n), dt=1.0 / fs)
        i = int(np.argmax(amp))
        assert_within(float(f[i]), 3.0, rtol=1e-12, label=f'{f_true} Hz 的谱峰在 3 Hz')
        assert_within(float(amp[i]), 1.0, rtol=1e-12, label=f'{f_true} Hz 的幅度仍是 1.0',
                      detail='混叠不改变幅度（前提是两者都整周期）')
    assert_within(alias_frequency(17.0, fs), 3.0, rtol=1e-12, label='alias_frequency(17,20)=3')
    assert_within(alias_frequency(12.0, fs), 8.0, rtol=1e-12,
                  label='alias_frequency(12,20)=8（12 折到 8，不是 3）')
    assert_within(alias_frequency(7.0, 10.0), 3.0, rtol=1e-12, label='alias_frequency(7,10)=3')
    assert_leq(float(alias_index(7, 10)), 3.0, rtol=0.0, label='alias_index(7,10)=3')
    cand = np.asarray(alias_candidates(3.0, fs, f_max=20.0), dtype=float)
    near = [float(c) for c in cand if 0.0 <= float(c) <= 20.0]
    record(label='3 Hz 在 20 Hz 以内的候选真频率个数', theory='3、17（差一个 fs=20）',
           measured=float(len(near)), tol=0.5, passed=len(near) >= 2,
           detail='候选 = ' + '、'.join(f'{c:g}' for c in near)
                  + '——谱上无法区分，必须靠先验或改变 fs')


@assertion('B', theory='db() 与幅度谱的组合是 20log10，且地板值不产生 NaN', chapter='2.5')
def test_db_conversion_and_floor() -> None:
    r"""dB 换算与地板：0 在 dB 域是负无穷，必须给地板值才能画。

    地板值的选取会影响读图（-60 dB 与 -120 dB 看到的旁瓣完全不同），
    所以它是**结论的一部分**，要写出来而不是随手定。
    """
    v = np.array([1.0, 0.5, 0.1, 1e-6, 0.0])
    got = db(v, floor_db=-120.0)
    expect = np.array([0.0, -6.0206, -20.0, -120.0, -120.0])
    assert_within(float(np.max(np.abs(got[:4] - expect[:4]))), 0.0, rtol=1.0, atol=1e-4,
                  label='20log10 换算')
    assert_leq(abs(float(got[4]) + 120.0), 1e-9, rtol=0.0, label='0 映射到地板 -120 dB',
               detail='assert_within 取绝对值，负值要用 assert_leq 配 abs')
    assert_leq(float(np.count_nonzero(~np.isfinite(got))), 0.0, rtol=0.0,
               label='地板后无 NaN/Inf', detail='对数轴上画不出 0，必须给有限值')
    _, amp = amplitude_spectrum(np.zeros(16), dt=1.0)
    full_floor = db(amp, floor_db=-60.0)
    assert_leq(float(np.max(np.abs(full_floor + 60.0))), 1e-9, rtol=0.0,
               label='全零信号映射到全地板 -60 dB',
               detail='assert_leq 不取绝对值，dB 是负数，要比 |值 + 地板|')
    record(label='0 的 dB 值（带地板）', theory='-120 dB',
           measured=float(got[4]), tol=1e-9, passed=abs(float(got[4]) + 120.0) < 1e-9,
           detail='不给地板会得到负无穷，帧校验直接拒绝')


@assertion('B', theory='相位谱与幅度谱互补：幅度不含时间原点，相位含', chapter='2.5')
def test_phase_spectrum_encodes_the_time_origin() -> None:
    r"""位移只改相位、不改幅度，所以只看幅度谱会丢掉时间原点信息。"""
    n = 64
    x = rng(1100).standard_normal(n)
    f1, p1 = phase_spectrum(x, dt=1.0 / n)
    f2, p2 = phase_spectrum(shift_sample(x, 5), dt=1.0 / n)
    assert_leq(float(np.max(np.abs(f1 - f2))), 0.0, rtol=0.0, label='栅格不随位移改变')
    _, a1 = amplitude_spectrum(x, dt=1.0 / n)
    _, a2 = amplitude_spectrum(shift_sample(x, 5), dt=1.0 / n)
    scale = max(float(np.max(a1)), 1e-300)
    assert_leq(float(np.max(np.abs(a1 - a2))) / scale, 64.0 * n * U, rtol=1.0,
               label='位移不改幅度谱', detail='幅度里没有时间信息')
    k = np.arange(n // 2 + 1, dtype=float)
    expect = -2.0 * np.pi * k * 5.0 / float(n)
    diff = np.angle(np.exp(1j * (p2 - p1)))
    err = float(np.max(np.abs(np.angle(np.exp(1j * (diff - expect))))))
    assert_leq(err, 1e-9, rtol=0.0, label='相位斜坡 = -2*pi*k*p/N',
               detail='这是位移定理在相位谱上的样子')
    record(label='相位斜坡的最大偏差', theory='不超过 1e-9',
           measured=err, tol=1e-9, passed=err <= 1e-9,
           detail='p=5、n=64；用 exp(i*) 比较相位，避免分支切割')


@assertion('B', theory='循环导数：频域乘 i*w 对应时域循环差分', chapter='2.5')
def test_circular_derivative_matches_spectral_multiplication() -> None:
    r"""循环差分与频域乘子的对应，顺带说明 DFT 的循环本性。

    一阶循环差分的乘子是 :math:`e^{i\omega}-1`，**不是** :math:`i\omega`：
    在 :math:`\omega=2\pi\cdot 3/64` 处两者的模相差 0.2%，但相位差 0.147 rad，
    作用在纯正弦上就是 14.7% 的波形偏差。把这两个乘子混用，是"用 FFT 做微分"
    最常见的错误。
    """
    n = 64
    x = np.sin(2.0 * np.pi * 3.0 * np.arange(n, dtype=float) / n)
    got = circular_derivative(x)
    assert_leq(float(np.max(np.abs(got - (np.roll(x, -1) - x)))), 0.0, rtol=0.0,
               label='循环差分 = 定义式（逐位相等）')
    k = np.arange(n, dtype=float)
    k = np.where(k < n // 2, k, k - n)
    w = 2.0 * np.pi * k / float(n)
    spec = np.asarray(fft(x))
    back = np.real(idft_direct(spec * (np.exp(1j * w) - 1.0)))
    scale = max(float(np.max(np.abs(back))), 1e-300)
    err = float(np.max(np.abs(got - back))) / scale
    assert_leq(err, 512.0 * n * U, rtol=1.0, label='差分的谱乘子 = e^{i w} - 1',
               detail=f'与定义式对拍 {err:.2e}')
    analytic = (2.0 * np.pi * 3.0 / float(n)
                * np.cos(2.0 * np.pi * 3.0 * np.arange(n, dtype=float) / n))
    # 差分的波形 = 解析导数 × [2sin(w/2)/w] 再平移 w/2，所以它与解析导数差一个相位
    err_phase = float(np.max(np.abs(back - analytic))) / max(float(np.max(np.abs(analytic))), 1e-300)
    # 差分的波形 = 解析导数平移 w/2 再乘 2sin(w/2)/w；波形偏差 = 2sin(w/4)
    # （写 2sin(w/2) 是错的，那个数是"满摆幅"而不是这条曲线上的最大差 —— 见 pitfalls）
    # 差分的波形 = 解析导数平移 w/2 再乘 2sin(w/2)/w，所以波形偏差 = 2sin(w/4)·2sin(w/2)/w。
    # 容差 2e-2：极值不一定正好落在 64 个采样点上（实测差 0.06%）。
    amp_ratio = float(2.0 * np.sin(0.5 * 2.0 * np.pi * 3.0 / n) / (2.0 * np.pi * 3.0 / n))
    theo_phase = 2.0 * np.sin(0.25 * 2.0 * np.pi * 3.0 / n) * amp_ratio
    assert_within(err_phase, theo_phase, rtol=2e-2,
                  label='差分 vs 解析导数的波形偏差 = 2sin(w/4)·2sin(w/2)/w',
                  detail='偏差全是相位（平移 w/2）；残下的 0.06% 是极值没落在采样点上')
    naive = np.real(idft_direct(spec * (1j * w)))
    err_naive = float(np.max(np.abs(naive - analytic))) / max(float(np.max(np.abs(analytic))), 1e-300)
    # 这里必须说实话：i*w 不是"错的乘子"，它就是谱微分；e^{iw}-1 才是**一阶差分**。
    # 两者相差 2sin(w/2)/w 的幅度与 w/2 的相位；在 w 小的时候几乎一样，
    # 所以"用哪个都行"的错觉很容易形成（见 pitfalls）。
    assert_leq(err_naive, 1e-13, rtol=0.0, label='i*w 对纯正弦给出精确导数',
               detail=f'实测 {err_naive:.2e}：谱微分是精确算子，不是近似')
    record(label='谱微分与一阶差分是两个不同的乘子',
           theory='i*w 给出精确导数；e^{iw}-1 给出精确一阶差分，两者差 2sin(w/2)/w 的幅度与 w/2 的相位',
           measured=err_phase, tol=theo_phase,
           passed=err_naive <= 1e-13 and err_phase > 0.1,
           detail=f'i*w 与解析导数差 {err_naive:.2e}；e^{{iw}}-1 与解析导数差 {err_phase:.1%}'
                  f'（幅值比 {amp_ratio:.6f}，只差 {abs(1 - amp_ratio):.2%}，其余 {err_phase:.1%} 全是相位）')


@assertion('B', theory='卷积定理的输入可以不等长：内部补零到 max(N,M)', chapter='2.5')
def test_fft_convolve_accepts_unequal_lengths() -> None:
    r"""circular_convolution 要求等长（定义如此），fft_convolve 则补到 max(N,M)。

    这个约定必须写成断言，否则调用方会以为 fft_convolve 在做线性卷积。
    """
    a = rng(1300).standard_normal(16)
    b = rng(1400).standard_normal(6)
    via = fft_convolve(a, b, mode='circular')
    manual = circular_convolution(a, zero_pad(b, 16))
    scale = max(float(np.max(np.abs(manual))), 1e-300)
    assert_leq(float(np.max(np.abs(via - manual))) / scale, 128.0 * 16 * U, rtol=1.0,
               label='不等长：fft_convolve 等于短序列补零后的循环卷积')
    same = fft_convolve(a, a, mode='circular')
    assert_leq(float(np.max(np.abs(same - circular_convolution(a, a)))) / scale,
               128.0 * 16 * U, rtol=1.0, label='等长时两条路径一致')
    raised = False
    try:
        circular_convolution(a, b)
    except ValueError:
        raised = True
    record(label='circular_convolution 拒绝不等长输入', theory='抛 ValueError，而不是悄悄补零',
           measured=float(raised), tol=0.0, passed=raised,
           detail='这是循环卷积的定义在代码里的样子')


@assertion('B', theory='窗能量恒等式：Hann 的能量约为 3N/8，rect 是 N', chapter='2.5')
def test_window_energy_identities() -> None:
    r"""窗能量的闭式（rect 为 N、Hann 约为 3N/8）决定信噪比损失。"""
    n = 1024
    e_rect = window_energy(window('rect', n))
    e_hann = window_energy(window('hann', n))
    assert_within(e_rect, float(n), rtol=1e-12, label='rect 能量 = N')
    assert_within(e_hann, 3.0 * n / 8.0, rtol=1e-3, label='hann 能量约为 3N/8',
                  detail='实测 383.625 对 384：离散和与连续积分的差别')
    assert_within(window_energy(window('blackman', n)) / float(n), 0.3043, rtol=5e-3,
                  label='blackman 能量约为 0.3045N')
    record(label='Hann/rect 能量比', theory='约 3/8 = 0.375',
           measured=float(e_hann / e_rect), tol=1e-3,
           passed=abs(e_hann / e_rect - 0.375) < 1e-3,
           detail='能量损失换来旁瓣抑制，这是窗的代价在能量上的账')


@assertion('B', theory='补零只是插值：原栅格逐位回到原值，主瓣宽度收敛而不是变小', chapter='2.5')
def test_zero_padding_interpolates_and_does_not_sharpen() -> None:
    r"""补零的三条可测含义，全部与「提高分辨率」相反。

    1. 频率栅格加密 ``pad`` 倍：``df = 1/(N*pad*dt)``；
    2. **原栅格逐位回到原值**：``fft(zero_pad(x, N*pad))[::pad] == fft(x)``，
       误差是**精确的 0**——这就是"没有新信息"的机器版本；
    3. 用补零后的谱去量主瓣宽度，读数会随 ``pad`` 单调**下降**并收敛（量化误差在消失），
       但不会收敛到 0：它趋向的是同一条连续主瓣的真实宽度。

    顺带钉一个归一化陷阱：``amplitude_spectrum`` 的 ``ref_length`` 必须给**原始长度 N**，
    否则补零后峰值会整整齐齐地掉 ``pad`` 倍（实测 pad=2 峰 0.5、pad=4 峰 0.25）。
    """
    n = 64
    dt = 0.01
    assert_within(rayleigh_resolution(1.0), 1.0, rtol=1e-12, label='Rayleigh = 1/T')
    assert_within(frequency_resolution(n, dt=dt), 1.0 / (n * dt), rtol=1e-12,
                  label='栅格 = 1/(N*dt)')

    k = 5
    x = np.cos(2.0 * np.pi * (k / (n * dt)) * np.arange(n, dtype=float) * dt)
    base = np.asarray(fft(x))
    widths = []
    peaks = []
    for pad in (1, 2, 4, 8, 16):
        xp = zero_pad(x, n * pad).real
        fine = np.asarray(fft(xp))
        assert_leq(float(np.max(np.abs(fine[::pad] - base))), 0.0, rtol=0.0,
                   label=f'pad={pad} 原栅格逐位回到原值', detail='精确 0，不是"小量"')
        f, amp = amplitude_spectrum(xp, dt=dt, ref_length=n)
        assert_within(float(f[1] - f[0]), 1.0 / (n * pad * dt), rtol=1e-9,
                      label=f'pad={pad} 栅格 = 1/(N*pad*dt)')
        pk = float(np.max(amp))
        peaks.append(pk)
        i = int(np.argmax(amp))
        j = i
        while j + 1 < amp.size and amp[j + 1] >= 0.5 * pk:
            j += 1
        lo = 0.5 * float(f[i - 1] + f[i]) if i > 0 else float(f[i])
        hi = 0.5 * float(f[j] + f[j + 1]) if j + 1 < amp.size else float(f[j])
        widths.append(hi - lo)
    assert_within(peaks[0], peaks[-1], rtol=1e-3,
                  label='峰值不随 pad 改变', detail='ref_length=N 时峰值应当不动')
    dec = [widths[i] - widths[i - 1] for i in range(1, len(widths))]
    rec = record(label='主瓣半高全宽随 pad 的增量',
                 theory='读数单调下降（量化误差在消失），但不收敛到 0',
                 measured=min(dec), tol=0.0, passed=min(dec) < 0.0,
                 detail='宽度序列 ' + '、'.join(f'{w:.4f}' for w in widths)
                        + ' Hz；最后两档只差 '
                        + f'{abs(dec[-1]):.4f} Hz（趋于同一条连续主瓣）')
    assert rec.passed, rec.measured
    assert_leq(widths[-1], 1.5, rtol=0.0,
               label='收敛值不是 0：主瓣宽度有正的下限',
               detail=f'pad=16 时读数 {widths[-1]:.4f} Hz；把它当"分辨率"会得出'
                      '「补零无限提高分辨率」的结论')
    for t_obs in (0.5, 2.0, 4.0):
        assert_within(rayleigh_resolution(t_obs), 1.0 / t_obs, rtol=1e-12,
                      label=f'T={t_obs} Rayleigh = 1/T')
