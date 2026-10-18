r"""2.5 章：傅里叶分析与 FFT。

五个可视化分工如下（所有数值来自 :mod:`numcore.transform.fft`，本文件只做搬运与装配）：

1. ``signal_spectrum`` —— 时域/频域双视图。可拖基频与前三次谐波幅度；``n`` 是**观测点数**，
   ``pad`` 是**补零倍数**。两者必须分开讲：``pad`` 只把频率栅格加密（弯出更光滑的谱线），
   ``n`` 才决定主瓣宽度 :math:`\\approx 1/T`。副轴给 ``|FFT − DFT|``，它是「这条谱线算得对不对」
   的证据，而不是「分辨率高不高」的证据。
2. ``leak_window`` —— 泄漏与窗：矩形 / Hann / Hamming / Blackman 的窗形与 dB 谱并排，
   叠三条参考线：实测峰值旁瓣、Harris 表值、渐近包络斜率 :math:`f^{-p}`。
   拖 ``offset`` 看**非整周期采样**如何把能量从主瓣漏到旁瓣。
3. ``butterfly_stages`` —— radix-2 的**分级**蝶形：``t`` 逐级点亮（``time.mode="scrub"``），
   实部/虚部两条轨迹 + 逐级残差（向量化实现 vs 朴素二重循环，应贴舍入地板）。
4. ``conv_theorem`` —— 卷积定理：时域卷积与「正变换→逐点相乘→逆变换」两条曲线叠在一起，
   再单独画它们与线性卷积的差：``N`` 不够大时尾部**卷绕**回来，前几项看着「滤得挺好」。
5. ``resolution`` —— 分辨率与不确定性：两个近邻音能否分开 vs 观测长度 ``T``；
   ``pad`` 只加密栅格，栅格扫描实测的最小可分辨间隔贴住 Rayleigh 判据 :math:`1/T`；
   四个窗在**整数格**上给出同一个阈值，窗的代价只在半格位置上现形（见 pitfalls）。

约定（与 0.0/1.2/2.2 章一致）：``meta`` 只放 number/bool/null，字符串一律走 ``set_strings``；
对数轴上画不出的 0 给地板值 ``FLOOR`` 并在 ``strings`` 里说明。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from numcore.transform.fft import (
    WINDOW_ENVELOPE_POWER,
    WINDOW_MAINLOBE_BINS,
    WINDOW_NAMES,
    WINDOW_SIDELOBE_DB,
    amplitude_spectrum,
    butterfly_diagram,
    circular_convolution,
    convolution_theorem_error,
    db,
    dft_direct,
    fft,
    fft_convolve,
    frequency_resolution,
    linear_convolution,
    min_separable_separation,
    peak_sidelobe_level,
    plan_fft,
    rayleigh_resolution,
    separable,
    sidelobe_envelope,
    sinusoid_sum,
    square_wave,
    two_tone,
    window,
    window_spectrum,
    wrap_around_error,
    zero_pad,
)
from server.nlf import Frame

__all__ = ['MANIFEST', 'compute']

#: 对数轴上「0 画不出来」的统一地板：半个单位舍入（与 1.2/2.2 章同一口径）
FLOOR = 1e-18

#: 每帧允许的最大点数（补零后的 FFT 预算）
MAX_N = 8192


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
MANIFEST: dict[str, Any] = {
    'chapter': '2.5',
    'title': '傅里叶分析与 FFT',
    'part': 'II',
    'summary': 'DFT 的定义与快速算法、泄漏与窗、分辨率与不确定性、循环卷积的卷绕。',
    'vizzes': [
        {
            'id': 'signal_spectrum',
            'title': '信号 ↔ 频谱联动',
            'description': '拖基频与各次谐波幅度，看时域波形与单边幅度谱同步变化；pad 只加密栅格。',
            'dataSource': 'backend',
            'params': [
                {'name': 'n', 'label': '观测点数 N', 'type': 'int', 'min': 32, 'max': 2048,
                 'default': 256, 'critical': [64.0],
                 'criticalNote': 'N·dt = T 决定主瓣宽度 ≈ 1/T = df 栅格；N 再小则基频与二次谐波会并进同一个主瓣。'},
                {'name': 'pad', 'label': '补零倍数', 'type': 'int', 'min': 1, 'max': 8,
                 'default': 1, 'critical': [1.0],
                 'criticalNote': 'pad > 1 只把栅格加密到 df/pad（谱线更光滑）；分辨率仍由 T = N·dt 决定，越过它不会让两个近邻音分开。'},
                {'name': 'f0', 'label': '基频 f0（周期数/T）', 'type': 'float', 'min': 0.5, 'max': 16.0,
                 'step': 0.1, 'default': 3.0, 'critical': [], 'criticalNote': ''},
                {'name': 'a1', 'label': '基波幅度', 'type': 'float', 'min': 0.0, 'max': 1.0,
                 'step': 0.05, 'default': 1.0, 'critical': [], 'criticalNote': ''},
                {'name': 'a2', 'label': '二次谐波幅度', 'type': 'float', 'min': 0.0, 'max': 1.0,
                 'step': 0.05, 'default': 0.5, 'critical': [], 'criticalNote': ''},
                {'name': 'a3', 'label': '三次谐波幅度', 'type': 'float', 'min': 0.0, 'max': 1.0,
                 'step': 0.05, 'default': 0.25, 'critical': [], 'criticalNote': ''},
                {'name': 'phase', 'label': '二次谐波相位 (rad)', 'type': 'float', 'min': -3.14159,
                 'max': 3.14159, 'step': 0.1, 'default': 0.0, 'critical': [], 'criticalNote': ''},
            ],
            'time': {'mode': 'none'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'time', 'autoscaled': False, 'title': '时域 x(t)', 'ylabel': 'x',
                 'scale': 'linear'},
                {'id': 'spec', 'autoscaled': False, 'title': '单边幅度谱（相对峰值 dB）',
                 'ylabel': 'dB', 'scale': 'linear'},
                {'id': 'error', 'autoscaled': True, 'title': '|FFT − DFT|（自研对拍）',
                 'ylabel': 'max|delta|', 'scale': 'log'},
            ],
            'pitfalls': '把 pad 从 1 拖到 8，谱线会明显变光滑、峰也显得更尖，很容易读成「分辨率提高了」；其实主瓣宽度只由 T = N·dt 决定，pad 只是把同一条连续谱采得更密。真正的判据是「两个近邻音之间的谷有没有低于两峰」——pad 变大时这个谷的深度不变。',
            'theory': ['DFT 定义 X_k = sum_n x_n exp(-2*pi*i*n*k/N)',
                       '频率栅格 df = 1/(N*dt) = 1/T',
                       'Rayleigh 判据 df_min ≈ 1/T',
                       'FFT 与 O(N^2) DFT 逐点一致到舍入级'],
        },
        {
            'id': 'leak_window',
            'title': '泄漏与窗函数旁瓣',
            'description': '四种窗的窗形与 dB 谱；参考线给出实测旁瓣、Harris 表值与渐近包络斜率。',
            'dataSource': 'backend',
            'params': [
                {'name': 'n', 'label': '窗长 N', 'type': 'int', 'min': 32, 'max': 512,
                 'default': 64, 'critical': [], 'criticalNote': ''},
                {'name': 'offset', 'label': '频率偏移（栅格）', 'type': 'float', 'min': 0.0, 'max': 1.0,
                 'step': 0.05, 'default': 0.25, 'critical': [0.5],
                 'criticalNote': '整周期采样（offset = 0）时矩形窗无泄漏；offset 不为 0 就是非整周期，矩形窗旁瓣立刻抬到 -13 dB 量级，越过 0.5 后相邻两条谱线的幅度开始互相颠倒。'},
                {'name': 'pad', 'label': '补零倍数', 'type': 'int', 'min': 1, 'max': 8,
                 'default': 4, 'critical': [], 'criticalNote': ''},
                {'name': 'window', 'label': '窗', 'type': 'enum',
                 'options': ['rect', 'hann', 'hamming', 'blackman'], 'default': 'hann',
                 'critical': [], 'criticalNote': ''},
            ],
            'time': {'mode': 'none'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'shape', 'autoscaled': False, 'title': '四种窗的时域形状', 'ylabel': 'w[m]',
                 'scale': 'linear'},
                {'id': 'spec', 'autoscaled': False, 'title': '加窗后的 dB 谱（相对峰值）',
                 'ylabel': 'dB', 'scale': 'linear'},
                {'id': 'error', 'autoscaled': True, 'title': '解析窗谱 vs |FFT| 的相对差',
                 'ylabel': 'rel', 'scale': 'log'},
            ],
            'pitfalls': '把窗从矩形换成 Hann，旁瓣从 -13 dB 掉到 -31 dB，图上看是干净多了；但主瓣宽度同时从 0.89 格涨到 1.44 格——两个本来刚好分得开的近邻音会并成一个包。只报旁瓣改善是片面的：Hann 的相干增益只有 0.4922，白噪声下幅度读数要除以它才对。',
            'theory': ['矩形窗峰值旁瓣 -13.26 dB，Hann -31.47 dB（Harris 1978 表 1）',
                       'Hann 主瓣 -3 dB 宽 1.44 格 vs 矩形 0.89 格',
                       '旁瓣渐近包络 |W(f)| ~ f^(-p)：矩形/汉明 p=1、Hann/Blackman p=3'],
        },
        {
            'id': 'butterfly_stages',
            'title': 'radix-2 蝶形分级',
            'description': '逐级点亮蝶形运算：每级的实部/虚部轨迹与逐级残差，最后一级经位反序得 X_k。',
            'dataSource': 'backend',
            'params': [
                {'name': 'n', 'label': '点数 N（2 的幂）', 'type': 'int', 'min': 4, 'max': 512,
                 'default': 32, 'critical': [], 'criticalNote': ''},
                {'name': 'signal', 'label': '输入信号', 'type': 'enum',
                 'options': ['impulse', 'cosine', 'square', 'random'], 'default': 'cosine',
                 'critical': [], 'criticalNote': ''},
                {'name': 'k', 'label': '余弦频率格 k', 'type': 'int', 'min': 1, 'max': 31,
                 'default': 3, 'critical': [], 'criticalNote': ''},
            ],
            'time': {'mode': 'scrub', 'tMax': 1.0, 'dt': 0.05, 'fps': 30, 'label': '级'},
            'requiredAux': ['residual'],
            'axes': [
                {'id': 'trace', 'autoscaled': True, 'title': '蝶形轨迹（当前级的值）',
                 'ylabel': '值', 'scale': 'linear'},
                {'id': 'residual', 'autoscaled': True, 'title': '逐级残差（向量化 vs 朴素 DIF）',
                 'ylabel': 'max|delta|', 'scale': 'log'},
                {'id': 'spectrum', 'autoscaled': False, 'title': '位反序重排后的 |X_k|',
                 'ylabel': '|X|', 'scale': 'linear'},
            ],
            'pitfalls': '蝶形图只画连接关系时，任何一张图看起来都对——真正会错的是下标约定：DIT 与 DIF 的块一个隔点、一个连续，位序也有两种写法。本项目用「末行经位反序重排后必须逐点等于 DFT(x)」钉死它，而不是靠图好看。逐级残差量的是「向量化实现 vs 朴素二重循环」，贴地板只说明两种写法一致，不说明数学对——数学对由 DFT 对拍负责。',
            'theory': ['Cooley-Tukey 分治：X_k = E_k + W_N^k O_k',
                       'N 点 DFT 的蝶形数为 (N/2)*log2(N)',
                       'DIF 末行 = DFT 的位反序置换', '逐级残差贴舍入地板'],
        },
        {
            'id': 'conv_theorem',
            'title': '卷积定理与循环卷绕',
            'description': '时域循环卷积与频域逐点相乘两条曲线；另给它们与线性卷积的差，看尾部卷绕回来。',
            'dataSource': 'backend',
            'params': [
                {'name': 'n', 'label': '信号长度 N', 'type': 'int', 'min': 8, 'max': 128,
                 'default': 16, 'critical': [24.0],
                 'criticalNote': 'N 不小于 N_a+M-1 时循环卷积才等于线性卷积；默认滤波器长 M = 8，所以 N < 15 一定有卷绕，N = 24 已在安全区（参考线给的是含余量的值）。'},
                {'name': 'm', 'label': '滤波器长度 M', 'type': 'int', 'min': 2, 'max': 32,
                 'default': 8, 'critical': [], 'criticalNote': ''},
                {'name': 'kernel', 'label': '滤波器', 'type': 'enum',
                 'options': ['box', 'exp', 'alternating'], 'default': 'box',
                 'critical': [], 'criticalNote': ''},
                {'name': 'signal', 'label': '信号', 'type': 'enum',
                 'options': ['gaussian', 'rect', 'cosine'], 'default': 'gaussian',
                 'critical': [], 'criticalNote': ''},
            ],
            'time': {'mode': 'none'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'curves', 'autoscaled': True, 'title': '两条卷积曲线', 'ylabel': '值',
                 'scale': 'linear'},
                {'id': 'error', 'autoscaled': False, 'title': '循环 − 线性（差异全在卷绕处）',
                 'ylabel': 'delta', 'scale': 'linear'},
                {'id': 'safer', 'autoscaled': False, 'title': '补零到 N+M-1 后的差异',
                 'ylabel': 'delta', 'scale': 'linear'},
            ],
            'pitfalls': 'N 不够大时，循环卷积与线性卷积在前几项上完全一样（卷绕发生在尾部、折回来叠在前面），所以只画滤波结果会看着非常正常；必须把「循环 − 线性」的差单独画出来。另一个陷阱：把补零后的结果当成「更精确的卷积」——补零只让循环卷积等于线性卷积，不改变任何一个乘加。',
            'theory': ['循环卷积定理：DFT(a ⊛ b) = DFT(a)·DFT(b)',
                       'N 不小于 N_a+N_b-1 时循环卷积 = 线性卷积',
                       '频域相乘 = 时域循环卷积（不是线性卷积）'],
        },
        {
            'id': 'resolution',
            'title': '分辨率与不确定性',
            'description': '两个近邻音能否分开 vs 观测长度 T；二分实测最小可分辨间隔并与 Rayleigh 判据比对。',
            'dataSource': 'backend',
            'params': [
                {'name': 'n', 'label': '观测点数 N', 'type': 'int', 'min': 32, 'max': 4096,
                 'default': 256, 'critical': [], 'criticalNote': ''},
                {'name': 'pad', 'label': '补零倍数', 'type': 'int', 'min': 1, 'max': 8,
                 'default': 1, 'critical': [1.0],
                 'criticalNote': 'pad 只加密栅格：二分实测的最小可分辨间隔随 pad 只动百分之几，而随 T = N·dt 成反比变化——这就是「补零不等于提高分辨率」的可执行判据。'},
                {'name': 'df', 'label': '两音间隔（周期数/T）', 'type': 'float', 'min': 0.2,
                 'max': 8.0, 'step': 0.05, 'default': 2.0, 'critical': [1.0, 1.44],
                 'criticalNote': '整数格扫描下四个窗的实测边界都是 1.0/T（= Rayleigh 判据；critical 1.0）'
                                 '——因为两音都落在栅格上时，判据只看主瓣宽度与间隔的关系。'
                                 'Hann 的代价要在**半格**位置上才现形：两音取 2.25/T 与 3.75/T 时矩形窗仍可分，'
                                 'Hann 并成一个包（主瓣 1.44 格 > 1.5 格间隔已经贴极限）。'},
                {'name': 'amp2', 'label': '第二音幅度比', 'type': 'float', 'min': 0.05, 'max': 1.0,
                 'step': 0.05, 'default': 1.0, 'critical': [], 'criticalNote': ''},
                {'name': 'window', 'label': '窗', 'type': 'enum',
                 'options': ['rect', 'hann', 'hamming', 'blackman'], 'default': 'rect',
                 'critical': [], 'criticalNote': ''},
            ],
            'time': {'mode': 'none'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'spec', 'autoscaled': False, 'title': '两音谱（dB，相对峰值）', 'ylabel': 'dB',
                 'scale': 'linear'},
                {'id': 'threshold', 'autoscaled': False, 'title': '能否分辨 vs 观测长度',
                 'ylabel': '可分辨（1/0）', 'scale': 'linear'},
                {'id': 'error', 'autoscaled': True, 'title': '|FFT − DFT|', 'ylabel': 'max|delta|',
                 'scale': 'log'},
            ],
            'pitfalls': '把 N 从 64 拖到 4096，峰越来越瘦，很容易说「分辨率提高了」；本页固定 T = N·dt = 1，所以拖 N 只改变采样密度。判据必须用「两峰之间的谷是否低于矮峰」来判，用「峰有多尖」判会得出相反结论。另外峰值位置用抛物线插值可以做到远好于 df 的精度——定位精度与分辨率不是一回事。',
            'theory': ['Rayleigh 判据 df_min ≈ 1/T（整数格扫描下四种窗同值）',
                       '窗的代价只在非栅格（半格）位置上现形：Hann 在 1.5/T 上分不开',
                       '频率栅格 df = 1/(N*dt)（pad 只加密它）',
                       '不确定原理 df*T 不小于 c（矩形窗 c ≈ 0.89）',
                       '峰值插值精度远小于栅格间距'],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 帧装配
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已经过 :mod:`server.frames` 规范化与区间校验。"""
    if viz_id == 'signal_spectrum':
        return _signal_spectrum(params, t)
    if viz_id == 'leak_window':
        return _leak_window(params, t)
    if viz_id == 'butterfly_stages':
        return _butterfly_stages(params, t)
    if viz_id == 'conv_theorem':
        return _conv_theorem(params, t)
    if viz_id == 'resolution':
        return _resolution(params, t)
    raise KeyError(f'章节 2.5 没有可视化 {viz_id!r}')


def _signal_spectrum(params: Mapping[str, Any], t: float) -> Frame:
    n = int(min(int(params['n']), MAX_N))
    pad = int(params['pad'])
    f0 = float(params['f0'])
    amps = [float(params['a1']), float(params['a2']), float(params['a3'])]
    phase2 = float(params['phase'])
    freqs = [f0, 2.0 * f0, 3.0 * f0]
    x = sinusoid_sum(amps, freqs, [0.0, phase2, 0.0], fs=float(n), n=n)
    n_pad = n * pad
    freq, amp = amplitude_spectrum(zero_pad(x, n_pad).real, dt=1.0 / float(n),
                                   ref_length=n)
    amp_db = db(amp, floor_db=-60.0)
    err = float(np.max(np.abs(fft(x) - dft_direct(x))))
    plan = plan_fft(n)
    limit = float(n) / 2.0  # Nyquist（以周期数/T 为单位）
    aliased = [f for f in freqs if f > limit]

    fr = Frame(chapter='2.5', viz='signal_spectrum', kind='frame')
    fr.add('x', x, unit='', label='时域信号 x(t)')
    fr.add('t_axis', np.arange(n, dtype=float) / float(n), unit='T', label='时间（周期单位）')
    fr.add('spectrum_db', amp_db, unit='dB', label='单边幅度谱（相对峰值）')
    fr.add('freq', freq, unit='1/T', label='频率（周期数/T）')
    fr.add('error', np.full(n, max(err, FLOOR)), unit='',
           label='|FFT-DFT|（逐点对照，已铺满 N 点）')
    fr.set_meta(n=n, pad=pad, n_pad=n_pad, df=frequency_resolution(n_pad),
                df_true=frequency_resolution(n), mainlobe_width=frequency_resolution(n),
                f0=f0, nyquist=limit, max_error=err, n_aliased=float(len(aliased)),
                t_actual=float(t))
    fr.set_strings(status='ok',
                   nonfinite_policy='对数轴上 0 画不出来：谱线取 -60 dB 地板，error 取 1e-18 地板',
                   algorithm=plan.algorithm,
                   note='pad 只加密频率栅格（df = 1/(N*pad)），分辨率仍由 T = N*dt 决定',
                   alias_note=('有谐波超过 Nyquist ' + '、'.join(f'{f:.2f}' for f in aliased)
                               + '，它在谱上会被折回低频——那是混叠，不是新的频率成分')
                   if aliased else '全部谐波都在 Nyquist 之下')
    return fr


def _leak_window(params: Mapping[str, Any], t: float) -> Frame:
    n = int(params['n'])
    offset = float(params['offset'])
    pad = int(params['pad'])
    key = str(params['window'])

    shapes = np.zeros((len(WINDOW_NAMES), n), dtype=float)
    for i, name in enumerate(WINDOW_NAMES):
        shapes[i] = window(name, n)

    # 非整周期取样：频率落在两格之间，泄漏最明显
    x = sinusoid_sum([1.0], [(1.0 + offset)], fs=float(n), n=n)
    xw = x * window(key, n)
    n_pad = n * pad
    dt = 1.0 / float(n)
    freq, amp = amplitude_spectrum(zero_pad(xw, n_pad).real, dt=dt, ref_length=n)
    amp_db = db(amp, floor_db=-120.0)

    # 频率轴的读数是「每记录周期数」= f·T，也就是窗谱的自然自变量 f·N。
    # T = N·dt 必须显式算出来：早先这里写成 freq*n（n 是点数而不是时长），
    # 于是理论旁瓣线被整体压低了 (1/64) 的量级（矩形窗低了 36 dB），
    # 而且看起来只是"理论线画得保守"（见 pitfalls）。
    f_norm = freq * (float(n) * dt)
    metrics = {name: peak_sidelobe_level(name) for name in WINDOW_NAMES}
    # 理论旁瓣线：锚在**最高那条旁瓣**上（位置取解析谱在第一个零点之后的最大值点），
    # 再按 f^(-p) 拉直线。纯 (f/f_ref)^(-p)（f_ref=1）在 fN≈1 处会高出真值 10 dB 以上；
    # 锚点必须来自窗自己，否则"理论线"只是形状对、位置错。
    f_scan = np.arange(0.25, 40.0, 0.005)
    a_scan = window_spectrum(key, f_scan)
    f0 = float(metrics[key]['first_null_bins'])
    sel = f_scan >= f0
    tail, f_tail = a_scan[sel], f_scan[sel]
    j_peak = int(np.argmax(tail))
    f_anchor = float(f_tail[j_peak])
    a_anchor = float(tail[j_peak])
    p_env = float(WINDOW_ENVELOPE_POWER[key])
    with np.errstate(divide='ignore', invalid='ignore'):
        env = np.where(f_norm > 0.0, a_anchor * (np.abs(f_norm) / f_anchor) ** (-p_env), np.nan)
    # f = 0（DC）处包络发散：对数轴上画不出来，压到峰值上并在 strings 里说明
    env = np.where(np.isfinite(env), env, 1.0)
    # 误差曲线必须是**同类对同类**：|FFT(窗)| 与窗的解析连续谱 window_spectrum(key, f*N)。
    # 早先这里拿加窗信号的谱去减窗的谱，两条曲线根本不是一回事，offset=0 时误差恒为 1.0
    # ——那个数看着像"误差"，其实与实现对错无关（见 pitfalls）。
    plain = np.abs(fft(zero_pad(window(key, n), n_pad)))
    plain = plain[: freq.size] / max(float(np.max(plain)), 1e-300)
    analytic = window_spectrum(key, f_norm)
    rel_err = float(np.max(np.abs(plain - analytic)))

    fr = Frame(chapter='2.5', viz='leak_window', kind='frame')
    fr.add('window_shapes', shapes, unit='',
           label='四种窗（行序 rect/hann/hamming/blackman）')
    fr.add('window_shape', window(key, n), unit='', label=f'当前窗 {key}')
    fr.add('x_windowed', xw, unit='', label='加窗后的信号')
    fr.add('spectrum_db', amp_db, unit='dB', label='加窗后的 dB 谱（相对峰值）')
    fr.add('freq', freq, unit='1/T', label='频率（周期数/T）')
    fr.add('envelope', np.maximum(env, FLOOR), unit='',
           label=f'渐近包络 f^(-{p_env:g})（相对 f=1/T）')
    fr.add('error', np.full(n, max(rel_err, FLOOR)), unit='',
           label='|FFT(窗)| vs 解析连续谱的最大偏差（铺满 N 点）')
    fr.set_meta(n=n, pad=pad, offset=offset, peak_sidelobe_db=metrics[key]['peak_db'],
                harris_sidelobe_db=WINDOW_SIDELOBE_DB[key],
                mainlobe_bins=metrics[key]['mainlobe_bins'],
                harris_mainlobe_bins=WINDOW_MAINLOBE_BINS[key],
                coherent_gain=metrics[key]['coherent_gain'],
                envelope_power=p_env, envelope_anchor=f_anchor,
                envelope_anchor_db=20.0 * float(np.log10(max(a_anchor, FLOOR))),
                rel_error=rel_err, t_actual=float(t))
    fr.set_strings(status='ok',
                   nonfinite_policy='dB 谱地板 -120 dB（Blackman 旁瓣可低于 -90 dB）',
                   window=key,
                   note=('旁瓣越低主瓣越宽：{k} 峰值旁瓣 {p:.2f} dB、-3 dB 主瓣 {w:.2f} 格'
                         '（Harris 表值 {ph:.2f} dB / {wh:.2f} 格）').format(
                       k=key, p=metrics[key]['peak_db'], w=metrics[key]['mainlobe_bins'],
                       ph=WINDOW_SIDELOBE_DB[key], wh=WINDOW_MAINLOBE_BINS[key]),
                   cola_note='Hann/Hamming 在 50% 重叠下满足常数重叠相加（COLA），矩形窗不满足')
    return fr


def _butterfly_stages(params: Mapping[str, Any], t: float) -> Frame:
    n = int(params['n'])
    if n & (n - 1) or n < 4:
        n = 4
    signal = str(params['signal'])
    k = int(min(int(params['k']), max(n // 2 - 1, 1)))
    if signal == 'impulse':
        x = np.zeros(n, dtype=float)
        x[0] = 1.0
    elif signal == 'cosine':
        x = np.cos(2.0 * np.pi * k * np.arange(n, dtype=float) / float(n))
    elif signal == 'square':
        x = square_wave(n)
    else:
        # 确定性的伪随机：固定公式，不引入随机源（P4 也要求随机数一律走 numcore.utils）
        idx = np.arange(n, dtype=float)
        x = np.sin(idx * 12.9898) * np.cos(idx * 4.1414)

    diag = butterfly_diagram(x.astype(complex))
    levels = int(diag['levels'])
    values = np.asarray(diag['values'])
    residuals = np.asarray(diag['residuals'], dtype=float)
    spectrum = np.asarray(diag['spectrum'], dtype=complex)

    # t 属于 [0, tMax] → 当前级（0 = 输入，levels = 末行）
    step = int(round(float(t) * levels))
    step = max(0, min(levels, step))
    current = values[step]
    mag = np.abs(spectrum)
    ref = float(np.max(mag))
    err = float(np.max(np.abs(spectrum - dft_direct(x.astype(complex)))))

    fr = Frame(chapter='2.5', viz='butterfly_stages', kind='frame')
    # 帧的数组必须是实值（打包时 astype(float64) 会静默丢掉虚部并触发 ComplexWarning）
    fr.add('stage_values_real', np.real(values), unit='',
           label='逐级快照实部（行 = 级，末行位反序）')
    fr.add('stage_values_imag', np.imag(values), unit='', label='逐级快照虚部')
    fr.add('current_real', np.real(current), unit='', label=f'第 {step} 级的实部')
    fr.add('current_imag', np.imag(current), unit='', label=f'第 {step} 级的虚部')
    fr.add('residual', np.maximum(residuals, FLOOR), unit='',
           label='逐级残差（向量化 vs 朴素 DIF）')
    fr.add('spectrum_mag', mag, unit='', label='位反序重排后的 |X_k|')
    fr.add('error', np.full(n, max(err, FLOOR)), unit='', label='|FFT-DFT|（全谱，铺满 N 点）')
    fr.set_meta(n=n, levels=levels, n_butterflies=float(n * levels // 2), stage=step,
                max_level_residual=float(np.max(residuals)) if residuals.size else 0.0,
                spectrum_error=err, spectrum_peak=ref, k=k,
                t_actual=step / float(levels) if levels else 0.0)
    fr.set_strings(status='ok', signal=signal, algorithm='radix2-DIF',
                   nonfinite_policy='残差与误差取 1e-18 地板（贴舍入地板时对数轴画不出来）',
                   note='末行是位反序的，经 bit_reverse_indices 置换后必须逐点等于 DFT(x)；'
                        '逐级残差量的是向量化实现与朴素二重循环的一致性，不是数学正确性')
    return fr


def _conv_theorem(params: Mapping[str, Any], t: float) -> Frame:
    n = int(params['n'])
    m = int(min(int(params['m']), n))
    kern = str(params['kernel'])
    sig = str(params['signal'])
    idx = np.arange(n, dtype=float)
    center = float(n) // 2
    if sig == 'gaussian':
        x = np.exp(-0.5 * ((idx - center) / max(n / 8.0, 1.0)) ** 2)
    elif sig == 'rect':
        x = ((idx > center - max(n // 4, 1)) & (idx < center + max(n // 4, 1))).astype(float)
    else:
        x = np.cos(2.0 * np.pi * 3.0 * idx / float(n))
    jdx = np.arange(m, dtype=float)
    if kern == 'box':
        h = np.ones(m, dtype=float) / float(m)
    elif kern == 'exp':
        h = np.exp(-jdx / max(m / 4.0, 1.0))
        h = h / float(np.sum(h))
    else:
        h = ((-1.0) ** jdx) / float(m)

    via_fft = np.real(fft_convolve(x, h, mode='circular'))
    # circular_convolution 要求等长（它的定义就是长度 N 的模 N 求和），h 补零到 N
    direct = np.real(circular_convolution(x, zero_pad(h, n)))
    lin = np.real(linear_convolution(x, h))
    need = n + m - 1
    safe = np.real(fft_convolve(x, h, mode='linear'))
    k = min(lin.size, via_fft.size)
    diff_cyclic = via_fft[:k] - lin[:k]
    diff_safe = safe[:k] - lin[:k]
    theo = convolution_theorem_error(x, h)
    wrap = wrap_around_error(x, h)
    direct_err = float(np.max(np.abs(direct - via_fft)))

    fr = Frame(chapter='2.5', viz='conv_theorem', kind='frame')
    fr.add('signal', x, unit='', label='信号 x')
    fr.add('kernel', np.resize(h, n), unit='', label='滤波器 h（补零到 N）')
    fr.add('circular', via_fft, unit='', label='循环卷积（频域相乘）')
    fr.add('linear', np.resize(lin, n), unit='', label='线性卷积（前 N 项）')
    fr.add('difference', np.resize(diff_cyclic, n), unit='', label='循环 - 线性（卷绕误差）')
    fr.add('safe_difference', np.resize(diff_safe, n), unit='', label='补零到 N+M-1 后的差')
    fr.add('error', np.maximum(np.abs(np.resize(diff_cyclic, n)), FLOOR), unit='',
           label='|循环 - 线性|（卷绕幅度）')
    max_wrap = float(np.max(np.abs(diff_cyclic))) if diff_cyclic.size else 0.0
    fr.set_meta(n=n, m=m, need=need, max_wrap=max_wrap,
                first_wrap_index=wrap['first_index'], wrap_rel=wrap['rel_max'],
                safe_error=float(np.max(np.abs(diff_safe))) if diff_safe.size else 0.0,
                theorem_error=theo, direct_vs_fft=direct_err, t_actual=float(t))
    fr.set_strings(status='ok', kernel=kern, signal=sig,
                   nonfinite_policy='差异为 0 时误差取 1e-18 地板',
                   note=(f'N = {n} < N_a+M-1 = {need}：尾部卷绕回来叠在前面，最大偏差 '
                         f'{max_wrap:.3f}（相对 {wrap["rel_max"]:.1%}）')
                   if n < need else
                   f'N = {n} 不小于 N_a+M-1 = {need}：循环卷积已经等于线性卷积')
    return fr


def _resolution(params: Mapping[str, Any], t: float) -> Frame:
    n = int(min(int(params['n']), MAX_N))
    pad = int(params['pad'])
    df_want = float(params['df'])
    amp2 = float(params['amp2'])
    key = str(params['window'])
    t_max = 1.0
    f1 = 2.0
    f2 = f1 + df_want
    x = two_tone(1.0, f1, amp2, f2, t_max=t_max, n=n)
    xw = x * window(key, n)
    n_pad = n * pad
    freq, amp = amplitude_spectrum(zero_pad(xw, n_pad).real, dt=t_max / float(n),
                                   ref_length=n)
    amp_db = db(amp, floor_db=-80.0)
    ok = separable(xw, dt=t_max / float(n), rel_height=0.1, valley_ratio=0.7)

    # 能否分辨 vs 观测点数（本页固定 T = 1，所以点数 = 采样密度）
    spans = np.unique(np.clip(
        np.array([32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]),
        None, MAX_N).astype(int))
    ratio = np.zeros(spans.size, dtype=float)
    for i, nn in enumerate(spans):
        xx = two_tone(1.0, f1, amp2, f2, t_max=t_max, n=int(nn)) * window(key, int(nn))
        ratio[i] = 1.0 if separable(xx, dt=t_max / float(nn), valley_ratio=0.7) else 0.0

    # 实测最小可分辨间隔：用核里的**栅格扫描**（谓词对间隔不单调，半格上会假穿越；
    # 二分实测 1.73/T 这种"看着挺像真值"的假数就是这么来的）。与 pad 无关。
    sep_measured = min_separable_separation(t_max, n=n, window_name=key)
    rayleigh = rayleigh_resolution(t_max)
    # 经验阈值 = 实测最小可分辨间隔；与 Harris 主瓣宽度成正比（比例常数 ~1.7，见 notes 定理）
    mainlobe = WINDOW_MAINLOBE_BINS[key]
    theo = sep_measured / mainlobe if mainlobe > 0 else 0.0
    err = float(np.max(np.abs(fft(x) - dft_direct(x))))
    floor = max(err, FLOOR)

    fr = Frame(chapter='2.5', viz='resolution', kind='frame')
    fr.add('spectrum_db', amp_db, unit='dB', label='两音 dB 谱（相对峰值）')
    fr.add('freq', freq, unit='1/T', label='频率（周期数/T）')
    fr.add('span_ratio', ratio, unit='', label='能否分辨（1 = 能）')
    fr.add('span_n', spans.astype(float), unit='点', label='观测点数（T 固定为 1）')
    fr.add('error', np.full(n, floor), unit='', label='|FFT-DFT|（铺满 N 点）')
    fr.set_meta(n=n, pad=pad, df=df_want, amp2=amp2, separable=bool(ok),
                sep_measured=sep_measured, rayleigh=rayleigh, mainlobe_bins=mainlobe,
                sep_over_mainlobe=theo,
                grid_df=frequency_resolution(n_pad), true_df=frequency_resolution(n),
                max_error=err, t_actual=float(t))
    fr.set_strings(status='ok', window=key,
                   nonfinite_policy='dB 谱地板 -80 dB；误差取 1e-18 地板',
                   note=(f'当前设置：df = {df_want:.3f}/T，实测最小可分辨 {sep_measured:.3f}/T，'
                         f'Rayleigh 判据 {rayleigh:.3f}/T，本窗 −3 dB 主瓣 {mainlobe:.2f} 格 '
                         f'（比 {theo:.2f}）；补零只把栅格从 {frequency_resolution(n):.4f} '
                         f'加密到 {frequency_resolution(n_pad):.4f}'),
                   resolution_note='可分辨判据 = 谱中出现两个局部极大且两峰之间有低于矮峰的谷，'
                                   '不是峰够不够尖')
    return fr
