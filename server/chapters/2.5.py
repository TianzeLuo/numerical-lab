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
                 'criticalNote': '$N\\cdot dt=T$ 决定主瓣宽度 $\\approx1/T=df$ 栅格；$N$ 再小则基频与二次谐波会并进同一个主瓣。',
                 'help': '观测点数，也就是一个记录周期 $T=N\\cdot dt$ 里采到的样本数（本页取 $dt=1/N$，时间轴按周期计）。调大 $N$ 会把同样长的记录采得更密，主瓣变窄、基频与二次谐波更容易分开；调到 32 附近两个峰就会并进同一个包。读图看谱面板的峰宽与两峰之间的谷深，并注意它与补零倍数 pad 是两件不同的事。'},
                {'name': 'pad', 'label': '补零倍数', 'type': 'int', 'min': 1, 'max': 8,
                 'default': 1, 'critical': [1.0],
                 'criticalNote': '$pad>1$ 只把栅格加密到 $df/pad$（谱线更光滑）；分辨率仍由 $T=N\\cdot dt$ 决定，越过它不会让两个近邻音分开。',
                 'help': '补零倍数：在 $N$ 个真实样本之后补上 $(pad-1)N$ 个零再做变换，频率栅格随之加密到 $1/(N\\cdot pad)$。调大它谱线更光滑、峰看起来更尖，但真实分辨率只由观测时长 $T$ 决定，并不会因此提高。读图对照谱面板上的散点（原始栅格）与连续曲线：曲线没动，变的只是采样密度；它要与观测点数 $N$ 分开调节。'},
                {'name': 'f0', 'label': '基频 f0（周期数/T）', 'type': 'float', 'min': 0.5, 'max': 16.0,
                 'step': 0.1, 'default': 3.0, 'critical': [], 'criticalNote': '',
                 'help': '基频，单位是「每记录周期的周期数」，也就是 $f_0/T$。调大让三条谐波整体右移、靠近 Nyquist 上限 $N/2$；超过它以后高频谐波会折回低频，谱上多出并不存在的峰。读图看谱面板三条竖直参考线与谱峰是否对齐，以及状态里的混叠提示。它和三个幅度参数一起决定时域波形的疏密。'},
                {'name': 'a1', 'label': '基波幅度', 'type': 'float', 'min': 0.0, 'max': 1.0,
                 'step': 0.05, 'default': 1.0, 'critical': [], 'criticalNote': '',
                 'help': '基波（一次谐波）的幅度，无量纲，范围 0 到 1。它给整张谱定下高度基准：调到 0 时基波峰消失，只剩二次、三次谐波；调大则基波峰升高，其余谐波在「相对峰值 dB」里被压得更低。读图看谱面板最左侧那根最高峰，以及时域波形的整体起伏。它和二次、三次谐波幅度共同决定波形是否对称。'},
                {'name': 'a2', 'label': '二次谐波幅度', 'type': 'float', 'min': 0.0, 'max': 1.0,
                 'step': 0.05, 'default': 0.5, 'critical': [], 'criticalNote': '',
                 'help': '二次谐波的幅度，无量纲，范围 0 到 1。调大会让频率为 $2f_0$ 的那根谱线升高、时域波形出现更明显的双峰结构；调到 0 就只剩基波与三次谐波。读图看谱面板第二根谱线的高度，以及它与基波峰的相对差。它要和二次谐波相位一起调，才能看出相位只改波形、不改幅度谱。'},
                {'name': 'a3', 'label': '三次谐波幅度', 'type': 'float', 'min': 0.0, 'max': 1.0,
                 'step': 0.05, 'default': 0.25, 'critical': [], 'criticalNote': '',
                 'help': '三次谐波的幅度，无量纲，范围 0 到 1。调大让频率为 $3f_0$ 的谱线升高，波形边缘更方；调到 0 则三次谐波完全消失。读图看谱面板第三根谱线，若它已接近 Nyquist 就会被折回，位置不再对应 $3f_0$。它和基频 $f_0$ 联动最紧：$f_0$ 越大，三次谐波越早碰到上限。'},
                {'name': 'phase', 'label': '二次谐波相位 (rad)', 'type': 'float', 'min': -3.14159,
                 'max': 3.14159, 'step': 0.1, 'default': 0.0, 'critical': [], 'criticalNote': '',
                 'help': '二次谐波的初相位，单位是弧度，范围 $-\\pi$ 到 $\\pi$。它不改变单边幅度谱的高度，只改变二次谐波与基波在时间上的错位：调大调小能看到波峰变得不对称，甚至出现双峰。读图看时域波形的形状，谱面板几乎不动——这正是「幅度谱丢掉相位信息」的直观演示。它和二次谐波幅度一起决定波形的细节。'},
            ],
            'time': {'mode': 'none'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'time', 'autoscaled': False, 'title': '时域 $x(t)$', 'ylabel': '$x$',
                 'scale': 'linear'},
                {'id': 'spec', 'autoscaled': False, 'title': '单边幅度谱（相对峰值 $\\mathrm{dB}$）',
                 'ylabel': '$\\mathrm{dB}$', 'scale': 'linear'},
                {'id': 'error', 'autoscaled': True, 'title': '$|\\mathrm{FFT}-\\mathrm{DFT}|$（自研对拍）',
                 'ylabel': '$\\max|\\delta|$', 'scale': 'log'},
            ],
            'pitfalls': '把 pad 从 $1$ 拖到 $8$，谱线会明显变光滑、峰也显得更尖，很容易读成「分辨率提高了」；其实主瓣宽度只由 $T=N\\cdot dt$ 决定，pad 只是把同一条连续谱采得更密。真正的判据是「两个近邻音之间的谷有没有低于两峰」——pad 变大时这个谷的深度不变。',
            'theory': [
                '**先把连续信号变成有限序列。** 设采样间隔为 $dt$、采样点数为 $N$，记录时长就是 $T=N\\cdot dt$，样本记作 $x_n=x(n\\,dt)$，$n=0,\\ldots,N-1$。计算机只看见这 $N$ 个数，任何频率结论都带着「只观察了 $T$ 这么久」的前提；频率轴上的单位 $1/T$ 就是从这件事来的，而不是从采样率来的。这也解释了为什么后面所有关于「分辨率」的说法，都只在固定 $T$ 的前提下比较才有意义。',
                '**DFT 把这串数拆成 $N$ 个复指数。** 定义 $X_k=\\sum_{n=0}^{N-1}x_n e^{-2\\pi i nk/N}$，$k=0,\\ldots,N-1$。指数因子对 $n$ 以 $N$ 为周期，这 $N$ 个向量在 $0\\le n<N$ 上两两正交，所以反变换是 $x_n=\\frac1N\\sum_{k=0}^{N-1}X_k e^{2\\pi i nk/N}$：把正变换代进去，只有下标对齐的那一项求和得到 $N$，其余全被正交性消掉。正交性是这个定义能成立的全部理由，也是后面「栅格只能是 $1/T$ 的整数倍」的来源。',
                '**频率栅格由观测时长决定。** 第 $k$ 根谱线对应 $f_k=k/(N\\,dt)=k/T$，相邻两根相差 $\\Delta f=1/T$。这个间距不是人为选的：只有取 $1/T$ 的整数倍，$N$ 个复指数才在有限样本上两两正交。于是「观测越久、栅格越细」是定义层面的结论；把采样取得更密只是让同一个时长里的样本更多，不会改变 $1/T$。读图时主瓣宽度、峰间距都该用 $1/T$ 当尺子，而不是用谱线条数当尺子。',
                '**补零只是把同一条曲线采得更密。** 把 $x$ 补零到 $N^{\\prime}=pN$，补零后的谱满足 $X^{Z}_{pk}=X_k$：原栅格上的值逐点不变，新增的点只是在算 $\\sum_n x_n e^{-2\\pi i fn}$ 这条连续曲线在更细频率上的取值。所以谱线会变光滑、峰看起来更尖，但两个近邻音之间的谷深不会改变——补零不产生新信息，只改变你量它的尺子。图里把「原始栅格」的散点叠在连续谱上，就是为了让这件事可以直接看见。',
                '**截断把单根谱线抹成 Dirichlet 核。** 若输入是 $e^{2\\pi i\\kappa n/N}$、$\\kappa$ 不一定是整数，则 $X_k$ 是等比级数，可写成 $e^{i\\pi(\\kappa-k)(N-1)/N}\\cdot\\frac{\\sin\\pi(\\kappa-k)}{N\\sin(\\pi(\\kappa-k)/N)}$。$\\kappa$ 为整数时分子为零，只剩一根谱线；否则能量按这个核向两侧泄漏，主瓣宽度约 $1/T$。这就是「非整周期采样会泄漏」的来源，与窗挑得好不好无关；窗只能改变泄漏的形状，不能改变它为什么发生。',
                '**两个音能否分开，看主瓣宽度。** 两个等幅单音相隔 $1/T$ 时，两个 Dirichlet 核在中点恰好反号相消，谱上出现真正的零；间隔小于 $1/T$ 时两核在中点同号，谷消失，两个峰并成一个包。这就是 Rayleigh 判据 $\\Delta f_{\\min}\\approx1/T$。它把分辨率的上限锁在 $T$ 上：补零只能把谷画得更细，不能让谷变深；把 $N$ 调大但保持 $T$ 不变，也只是把同一段记录采得更密而已。',
                '**读数、误差与混叠要分开看。** 对实正弦，单边幅度谱在峰处的读数是 $2|X_k|/N$，直流与 Nyquist 处不乘 $2$；FFT 与 $O(N^2)$ 定义式的差只在舍入量级，副轴画的就是这个差，它证明「谱算得对」，不证明「两个音分得开」。另外，频率超过 $N/(2T)$ 的谐波会折回低频，谱上多出的峰是混叠，不是新的频率成分——这时该减小 $f_0$ 或提高采样密度，而不是去调补零。',
                '**把参数逐个对应到图上的读数。** 观测点数 $N$ 决定主瓣宽度与两峰能否分开，补零倍数 $pad$ 只决定谱线密度，基频 $f_0$ 决定三根谱线的位置，三个幅度决定它们的高度，二次谐波相位只改时域形状。读图时先看谱面板的三条竖线是否压住峰，再看两峰之间的谷有没有低于矮峰，最后看副轴的 $|\\mathrm{FFT}-\\mathrm{DFT}|$ 是否在舍入量级。把 $pad$ 拖大而谷深不变，就是「补零不等于分辨率」最直接的证据。',
            ],
        },
        {
            'id': 'leak_window',
            'title': '泄漏与窗函数旁瓣',
            'description': '四种窗的窗形与 dB 谱；参考线给出实测旁瓣、Harris 表值与渐近包络斜率。',
            'dataSource': 'backend',
            'params': [
                {'name': 'n', 'label': '窗长 N', 'type': 'int', 'min': 32, 'max': 512,
                 'default': 64, 'critical': [], 'criticalNote': '',
                 'help': '窗长，也就是每次截取多少个样本做变换，范围 32 到 512。调大让频率栅格 $1/(N\\cdot dt)$ 变细、主瓣变窄，但离散窗的旁瓣结构也随之变化；调小则谱线变粗、旁瓣更容易糊在一起。读图看 dB 谱面板的峰宽与旁瓣起伏，以及窗形面板两端是否收到 0。它和补零倍数一起决定谱线密度，但只有窗长影响真实的连续谱形状。'},
                {'name': 'offset', 'label': '频率偏移（栅格）', 'type': 'float', 'min': 0.0, 'max': 1.0,
                 'step': 0.05, 'default': 0.25, 'critical': [0.5],
                 'criticalNote': '整周期采样（$offset=0$）时矩形窗无泄漏；$offset$ 不为 $0$ 就是非整周期，矩形窗旁瓣立刻抬到 $-13$ dB 量级，越过 $0.5$ 后相邻两条谱线的幅度开始互相颠倒。',
                 'help': '信号频率相对频率栅格的小数偏移，单位是「格」。$0$ 表示整周期采样，矩形窗几乎不漏能量；调到 $0.25$ 或 $0.5$ 就是非整周期截断，能量立刻漏到旁瓣。读图看 dB 谱的次峰高度与谱线根数：偏移越大，相邻两根谱线的幅度越接近。它和窗的选择直接联动——换窗能压低旁瓣，却压不掉偏移本身。'},
                {'name': 'pad', 'label': '补零倍数', 'type': 'int', 'min': 1, 'max': 8,
                 'default': 4, 'critical': [], 'criticalNote': '',
                 'help': '补零倍数：把加窗后的 $N$ 个样本补零到 $N\\cdot pad$ 再做变换，只加密频率栅格。调大后同一条窗谱被采得更密、曲线更光滑，但旁瓣高度与主瓣宽度这些连续谱性质一点没变。读图时把连续曲线与谱线的离散点对照着看，并注意理论包络线锚在窗自己的最高旁瓣上。它不改变泄漏，只改变你看到泄漏的精细程度。'},
                {'name': 'window', 'label': '窗', 'type': 'enum',
                 'options': ['rect', 'hann', 'hamming', 'blackman'], 'default': 'hann',
                 'critical': [], 'criticalNote': '',
                 'help': '选择窗函数：矩形、Hann、Hamming 或 Blackman。矩形窗主瓣最窄但旁瓣只到 $-13$ dB；Hann 把旁瓣压到 $-31$ dB 左右，代价是主瓣变宽约一倍；Hamming 旁瓣更低，但端点不为零、衰减更慢；Blackman 旁瓣最低、主瓣最宽。读图看峰值旁瓣、主瓣宽度与包络斜率三个数一起变，再回到时域窗形看两端是否平滑。'},
            ],
            'time': {'mode': 'none'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'shape', 'autoscaled': False, 'title': '四种窗的时域形状', 'ylabel': '$w[m]$',
                 'scale': 'linear'},
                {'id': 'spec', 'autoscaled': False, 'title': '加窗后的 $\\mathrm{dB}$ 谱（相对峰值）',
                 'ylabel': '$\\mathrm{dB}$', 'scale': 'linear'},
                {'id': 'error', 'autoscaled': True, 'title': '解析窗谱 vs $|\\mathrm{FFT}|$ 的相对差',
                 'ylabel': '$\\mathrm{rel}$', 'scale': 'log'},
            ],
            'pitfalls': '把窗从矩形换成 Hann，旁瓣从 $-13$ dB 掉到 $-31$ dB，图上看是干净多了；但主瓣宽度同时从 $0.89$ 格涨到 $1.44$ 格——两个本来刚好分得开的近邻音会并成一个包。只报旁瓣改善是片面的：Hann 的相干增益只有 $0.4922$，白噪声下幅度读数要除以它才对。',
            'theory': [
                '**截断就是乘一个矩形窗。** 观测只保留 $0\\le n<N$ 的样本，等价于把无限长信号乘以 $w_n=1$（窗内）与 $w_n=0$（窗外）。时域相乘在频域变成卷积：观测到的谱不是原信号的一根线，而是原谱与窗谱的卷积。矩形窗的窗谱是 Dirichlet 核，旁瓣衰减很慢，所以即使信号本身只有一个频率，截断也会把能量摊到许多谱线上。',
                '**单音经矩形窗后是 Dirichlet 核。** 取 $x_n=e^{2\\pi i\\kappa n/N}$，$\\kappa$ 不一定是整数，则 $X_k$ 是公比为 $e^{2\\pi i(\\kappa-k)/N}$ 的等比级数，求和得 $X_k=e^{i\\pi(\\kappa-k)(N-1)/N}\\cdot\\frac{\\sin\\pi(\\kappa-k)}{N\\sin(\\pi(\\kappa-k)/N)}$。$\\kappa$ 为整数时分子为零，除 $k=\\kappa$ 外全为零；$\\kappa$ 偏离整数多少，旁瓣就抬多高。图里把 $offset$ 从 $0$ 拖到 $0.5$，看的正是分子从零逐渐变大的过程。',
                '**一般窗的谱等于把原谱再卷一次。** 加窗后 $x_n w_n$ 的变换是 $\\sum_n x_n w_n e^{-2\\pi i nk/N}$。对单音来说，峰值仍落在 $\\kappa$ 附近，但周围的形状完全由窗谱 $W(f)$ 决定：主瓣宽度是 $W$ 的主瓣宽度，旁瓣高度是 $W$ 的旁瓣高度。于是选窗就是选「用多宽的峰去换多低的旁瓣」，频谱图上看到的每一处起伏都能在窗谱里找到对应。',
                '**旁瓣与主瓣是一笔交换。** 矩形窗峰值旁瓣约 $-13.26$ dB、$-3$ dB 主瓣宽 $0.89$ 格；Hann 是 $-31.47$ dB、$1.44$ 格；Hamming 是 $-42.68$ dB、$1.30$ 格；Blackman 是 $-58.11$ dB、$1.68$ 格。压旁瓣的办法是让窗两端平滑地收到零，代价是有效观测变短、主瓣变宽。只报旁瓣改善会得出「Blackman 永远最好」的结论，而它把两个近邻音并成一个包的风险也最大。',
                '**端点光滑度决定滚降速度。** 对窗谱分部积分：每积分一次就把一个 $1/f$ 提到外面，同时把导数移到窗上。若窗在端点有跳变，只积一次就得到 $|W(f)|\\sim f^{-1}$；若窗及其一阶导在端点都为零，则要积三次才出现边界项，得到 $f^{-3}$。所以 Hamming 的第一旁瓣虽低到 $-42.7$ dB，端点却有 $0.08$ 的跳变，远处仍按 $f^{-1}$ 滚降；Hann 的第一旁瓣只有 $-31.5$ dB，远处却按 $f^{-3}$ 滚降。十倍频程上两者包络比分别是 $10^{-2}$ 与 $10^{-6}$，这就是「旁瓣低不等于衰减快」。',
                '**加窗还会改幅度读数。** 对 $x_n=A\\cos(2\\pi k_0n/N)$，加窗后峰值处的变换值是 $\\frac A2\\sum_n w_n$ 加上镜像项，于是单边幅度谱读到 $A\\cdot\\frac1N\\sum_nw_n$，这个因子叫相干增益。矩形窗是 $1$，Hann 约 $0.5$，Hamming 约 $0.54$，Blackman 约 $0.42$；噪声功率则按 $\\frac1N\\sum_nw_n^2$ 缩小。图里的 dB 谱是相对峰值归一化的，所以增益被约掉；一旦要报绝对幅度，必须把读数除以相干增益。',
                '**偏移与补零落在图上的不同位置。** $offset=0$ 时矩形窗的次峰是机器零；偏 $0.25$ 格次峰约 $-9.7$ dB；偏 $0.5$ 格两根谱线几乎等高（约 $-3.1$ dB），此时「峰位」已经不再唯一。补零只把同一条连续窗谱采得更密，不改变这些电平。理论包络线必须锚在窗自己的最高旁瓣上，而不是第一个零点：矩形窗的锚点在 $f\\cdot T\\approx1.43$，Hamming 在 $\\approx4.50$，把两者混为一谈会让整条参考线平移。',
                '**选窗时至少报三个数。** 峰值旁瓣回答「弱信号会不会被强信号的旁瓣盖住」，主瓣宽度回答「两个近邻音还能不能分开」，相干增益回答「幅度读数要乘回多少」。只看第一个数会选出 Blackman，只看第二个会永远选矩形窗；三个数一起看，才知道在具体任务里哪一种代价可以接受。图里的峰值旁瓣水平线、主瓣宽度文字和包络幂次就是这三笔账，副轴再给出解析窗谱与 $|\\mathrm{FFT}|$ 的偏差，说明曲线不是手工画出来的。',
            ],
        },
        {
            'id': 'butterfly_stages',
            'title': 'radix-2 蝶形分级',
            'description': '逐级点亮蝶形运算：每级的实部/虚部轨迹与逐级残差，最后一级经位反序得 $X_k$。',
            'dataSource': 'backend',
            'params': [
                {'name': 'n', 'label': '点数 N（2 的幂）', 'type': 'int', 'min': 4, 'max': 512,
                 'default': 32, 'critical': [], 'criticalNote': '',
                 'help': '变换的点数，必须是 2 的幂，范围 4 到 512。调大让级数按 $\\log_2 N$ 增加、蝶形数按 $(N/2)\\log_2 N$ 增加，轨迹图上的级数和残差曲线都会变长；调小则几级就到底。读图看残差面板的横轴级号与底部横条的级数、蝶形数。它同时限制频率格 $k$ 的可用范围：$k$ 不能超过 $N/2-1$，超过的部分不会生效。'},
                {'name': 'signal', 'label': '输入信号', 'type': 'enum',
                 'options': ['impulse', 'cosine', 'square', 'random'], 'default': 'cosine',
                 'critical': [], 'criticalNote': '',
                 'help': '输入信号类型：冲激、余弦、方波或伪随机序列。冲激的谱是常数，看不出位反序对不对；余弦只在频率格 $k$ 处有峰，最适合检查重排；方波有一串奇次谐波；伪随机序列让所有频率格都有值。读图看右上角位反序谱的峰位与形状。它和频率格 $k$ 联动：只有余弦才把 $k$ 的意义直接画在谱上。'},
                {'name': 'k', 'label': '余弦频率格 k', 'type': 'int', 'min': 1, 'max': 31,
                 'default': 3, 'critical': [], 'criticalNote': '',
                 'help': '余弦输入的频率格，单位是「每 $N$ 个样本的周期数」，范围 1 到 31。调大让余弦振荡变快、谱峰右移；超过 $N/2-1$ 的部分不会生效，实际会停在可用的上限。读图看位反序谱里竖线标出的 $k$ 与最高柱是否重合——不重合就说明位反序或下标约定有问题。它只对余弦信号有意义，换成冲激或方波后这个参数不再决定谱峰位置。'},
            ],
            'time': {'mode': 'scrub', 'tMax': 1.0, 'dt': 0.05, 'fps': 30, 'label': '级'},
            'requiredAux': ['residual'],
            'axes': [
                {'id': 'trace', 'autoscaled': True, 'title': '蝶形轨迹（当前级的值）',
                 'ylabel': '值', 'scale': 'linear'},
                {'id': 'residual', 'autoscaled': True, 'title': '逐级残差（向量化 vs 朴素 DIF）',
                 'ylabel': '$\\max|\\delta|$', 'scale': 'log'},
                {'id': 'spectrum', 'autoscaled': False, 'title': '位反序重排后的 $|X_k|$',
                 'ylabel': '$|X|$', 'scale': 'linear'},
            ],
            'pitfalls': '蝶形图只画连接关系时，任何一张图看起来都对——真正会错的是下标约定：DIT 与 DIF 的块一个隔点、一个连续，位序也有两种写法。本项目用「末行经位反序重排后必须逐点等于 DFT(x)」钉死它，而不是靠图好看。逐级残差量的是「向量化实现 vs 朴素二重循环」，贴地板只说明两种写法一致，不说明数学对——数学对由 DFT 对拍负责。',
            'theory': [
                '**DFT 的定义先摆出来。** $N$ 点序列的变换是 $X_k=\\sum_{n=0}^{N-1}x_n W_N^{nk}$，其中 $W_N=e^{-2\\pi i/N}$。直接按定义算，每个 $X_k$ 要做 $N$ 次复乘，全部 $N$ 根谱线就是 $N^2$ 次。快速算法的全部目标，是把这 $N^2$ 次乘加重新组织成可以重复利用的块，而不是换一个公式。',
                '**按奇偶拆分得到蝶形。** 设 $N$ 为偶数，把 $x$ 分成偶下标 $E_j=x_{2j}$ 与奇下标 $O_j=x_{2j+1}$，各自做 $N/2$ 点 DFT 得 $E_k$、$O_k$。因为 $W_N^2=W_{N/2}$，有 $X_k=E_k+W_N^kO_k$，且 $W_N^{k+N/2}=-W_N^k$，所以 $X_{k+N/2}=E_k-W_N^kO_k$，$k=0,\\ldots,N/2-1$。一对输出只用一次复乘 $W_N^kO_k$ 和两次复加，这就是一个蝶形。注意旋转因子只与级号和块内位置有关、与信号无关，所以同一个 $W$ 能在整级重复使用；这份重复正是省下乘法的来源。',
                '**递归展开给出运算量。** 记 $T(N)$ 为复乘次数，则 $T(N)=2T(N/2)+N/2$，$T(1)=0$。反复展开 $\\log_2N$ 次得 $T(N)=\\frac N2\\log_2N$：每一级有 $N/2$ 个蝶形，共 $\\log_2N$ 级。$N=32$ 时是 $5$ 级、$80$ 个蝶形；$N=8$ 时是 $3$ 级、$12$ 个；若按定义式硬算，$N=32$ 需要 $1024$ 次复乘，差了一个数量级。图底部横条报的级数与蝶形数就是这两个数，滑杆每走一段点亮一级。',
                '**DIF 的下标顺序是位反序。** 换一种拆法：先把前后半段相加、相减，$u_m=x_m+x_{m+N/2}$、$v_m=(x_m-x_{m+N/2})W_N^m$，再对两段各自递归。这样每一次递归决定的是频率下标 $k$ 的最低位，递归到底后数组第 $j$ 项存的是 $X_{\\mathrm{rev}(j)}$，其中 $\\mathrm{rev}$ 把 $r=\\log_2N$ 位二进制反过来。$N=8$ 时顺序是 $0,4,2,6,1,5,3,7$；图上右面板就是重排后的 $|X_k|$，所以余弦的峰必须落回输入的 $k$。',
                '**图上每一级在看什么。** 滑杆把 $t\\in[0,1]$ 映射到级号 $s=\\mathrm{round}(t\\log_2N)$：$s=0$ 是输入，$s=\\log_2N$ 是末行。左面板画当前级的实部与虚部，虚线是已经走过的级；它们不是「信号随时间变化」，而是同一批数在逐级被重新组合。级数增加时，数值的动态范围先略微变大、再回到与输入同量级，这反映的是中间和与差，而不是新的物理量。',
                '**逐级残差量的是实现一致性。** 同一批中间值由两套独立写法算出：一套向量化 DIF，一套逐蝶形的朴素二重循环。逐级比较得到的差在 $N=4,8$ 时为 $0$，$N$ 大时也只到 $10^{-14}$ 量级。它说明两份代码走的是同一条路径、舍入方式接近，却不说明这条路数学上正确——两份一模一样的错代码残差同样是零。数学正确性由另一条线负责：末行重排后与 $O(N^2)$ 定义式逐点对拍，误差也在舍入量级。',
                '**最容易踩的三个误读。** 其一，拿冲激去检验位反序：冲激的谱是常数，对任何置换都不敏感，怎么看都「对」；要改用余弦或复指数。其二，把残差贴地板当成「FFT 精度很高」：它只比较两份实现，与真值无关。其三，忘了 $k$ 受 $N$ 限制：$N=8$ 时只有 $k=1,2,3$ 有意义，更大的 $k$ 画出来的峰位与竖线对不上，不是算法错，而是参数被夹住了。',
                '**DIT 与 DIF 的块结构不同。** 按时间抽取（DIT）在输入侧按奇偶分块，蝶形的两个输入来自不同子问题；按频率抽取（DIF）在输出侧按前后半段分块，蝶形先做和与差，再乘旋转因子。两种写法给出同一组 $X_k$，只是中间数组的排列不同；本项目用 DIF，所以末行是位反序。读图时如果把 DIT 的连线套到 DIF 的数值上，数字可能仍然对，但每一级「第几个位置是什么」全错——这正是只看连线不验下标会踩的坑。',
            ],
        },
        {
            'id': 'conv_theorem',
            'title': '卷积定理与循环卷绕',
            'description': '时域循环卷积与频域逐点相乘两条曲线；另给它们与线性卷积的差，看尾部卷绕回来。',
            'dataSource': 'backend',
            'params': [
                {'name': 'n', 'label': '信号长度 N', 'type': 'int', 'min': 8, 'max': 128,
                 'default': 16, 'critical': [24.0],
                 'criticalNote': '$N$ 不小于 $N_a+M-1$ 时循环卷积才等于线性卷积；默认滤波器长 $M=8$，所以 $N<15$ 一定有卷绕，$N=24$ 已在安全区（参考线给的是含余量的值）。',
                 'help': '信号长度，也是做循环卷积的长度，范围 8 到 128。它必须不小于 $N_a+M-1$ 才能让循环卷积等于线性卷积；调小会让尾部卷绕回头部，差异集中在前几项。读图看「循环 $-$ 线性」面板的差异条数与「补零后」面板是否贴到零线。它和滤波器长度 $M$ 一起决定安全条件，单把 $N$ 调大并不够。'},
                {'name': 'm', 'label': '滤波器长度 M', 'type': 'int', 'min': 2, 'max': 32,
                 'default': 8, 'critical': [], 'criticalNote': '',
                 'help': '滤波器长度（抽头数），范围 2 到 32。调大让平滑或差分的作用范围变宽、线性卷积的尾部变长，卷绕影响的项数也从 $M-1$ 项增加；调小则滤波器更短、卷绕更少。读图看上方两条卷积曲线的差别，以及差异面板里非零条的个数是否等于 $M-1$。它和信号长度 $N$ 联动决定是否安全。'},
                {'name': 'kernel', 'label': '滤波器', 'type': 'enum',
                 'options': ['box', 'exp', 'alternating'], 'default': 'box',
                 'critical': [], 'criticalNote': '',
                 'help': '滤波器类型：盒式平均、指数衰减或正负交替。盒式与指数核都是低通，输出更平滑；交替核是高通，会把慢变化压掉、留下快速起伏。读图看上方循环卷积曲线相对输入信号是被磨平还是被尖锐化，再看差异面板的幅度。它不改变卷绕发生的条件，只改变卷积结果的形状与大小。'},
                {'name': 'signal', 'label': '信号', 'type': 'enum',
                 'options': ['gaussian', 'rect', 'cosine'], 'default': 'gaussian',
                 'critical': [], 'criticalNote': '',
                 'help': '输入信号：高斯脉冲、矩形块或余弦。高斯脉冲的支撑较宽，卷积后拖尾长，最容易看到卷绕叠回；矩形块边缘陡，能看清滤波器的振铃；余弦是窄带信号，卷积后幅度被滤波器的频响调制。读图看上方三条曲线在头部的差异。它和滤波器类型一起决定卷绕误差落在哪几项、有多大。'},
            ],
            'time': {'mode': 'none'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'curves', 'autoscaled': True, 'title': '两条卷积曲线', 'ylabel': '值',
                 'scale': 'linear'},
                {'id': 'error', 'autoscaled': False, 'title': '循环 $-$ 线性（差异全在卷绕处）',
                 'ylabel': '$\\delta$', 'scale': 'linear'},
                {'id': 'safer', 'autoscaled': False, 'title': '补零到 $N+M-1$ 后的差异',
                 'ylabel': '$\\delta$', 'scale': 'linear'},
            ],
            'pitfalls': '$N$ 不够大时，循环卷积与线性卷积在前几项上完全一样（卷绕发生在尾部、折回来叠在前面），所以只画滤波结果会看着非常正常；必须把「循环 − 线性」的差单独画出来。另一个陷阱：把补零后的结果当成「更精确的卷积」——补零只让循环卷积等于线性卷积，不改变任何一个乘加。',
            'theory': [
                '**两种卷积的定义不同。** 线性卷积把滤波器滑过整条信号：记结果为 $y$，则 $y_n=\\sum_m a_{n-m}b_m$，输出长度是 $N_a+N_b-1$，尾部不会被折回。循环卷积则先把下标对 $N$ 取模：记结果为 $c$，则 $c_n=\\sum_{m=0}^{N-1}a_{(n-m)\\ \\mathrm{mod}\\ N}b_m$，长度固定为 $N$，越界的部分从另一端绕回来。图上的两条曲线分别就是这两个定义算出来的。',
                '**循环卷积定理是恒等式。** 对循环卷积做 DFT：$\\sum_n\\sum_m a_{(n-m)\\ \\mathrm{mod}\\ N}b_mW_N^{nk}$，交换求和次序并令 $l=(n-m)\\ \\mathrm{mod}\\ N$。因为 $W_N^{kN}=1$，取模可以去掉，内层正好是 $A_k$，外层是 $\\sum_m b_mW_N^{mk}=B_k$。于是 $\\mathrm{DFT}(c)_k=A_kB_k$：频域逐点相乘与时域循环卷积永远相等，不需要任何附加条件。',
                '**要等于线性卷积，必须装得下。** 线性卷积的支撑是 $0\\le n\\le N_a+N_b-2$，共 $N_a+N_b-1$ 项。若循环长度 $N\\ge N_a+N_b-1$，这段支撑整个落在 $[0,N-1]$ 内，取模下标与真实下标相同，两个求和逐项一致，循环卷积与线性卷积完全相等。若 $N$ 更小，超出 $N-1$ 的尾部会折回前面，差异因此出现在头部，而不是均匀分布。',
                '**卷绕只落在前 $M-1$ 项。** 设信号长 $N$、滤波器长 $M$。当下标满足 $n-m<0$ 时取模会加一个 $N$，把线性卷积的尾部搬到了 $n$ 处；这要求 $n<m\\le M-1$，即 $0\\le n\\le M-2$。所以差异的项数恰好是 $M-1$，而且全在最前面——恰好是你最可能仔细看的那几项；滤波器越长，头部被污染的范围越宽。图里把「差异位置数」从数组里数出来，而不是抄一个常数，就是为了让这条结论可核对。',
                '**补零到 $N+M-1$ 才是质变。** 把 $a$ 与 $b$ 都补零到 $N^{\\prime}=N_a+N_b-1$ 再做循环卷积，长度足够装下全部线性卷积，取模不再发生，两者逐点相等。注意补零没有改变任何一个乘加，也没有提高精度：它只是把「卷绕」这个副作用关掉。实测 $N=16$、$M=8$ 时最大偏差 $0.2508$（相对 $42.1\\%$），补零到 $23$ 后降到 $10^{-16}$ 的舍入地板。',
                '**「N 大一点」只是量变。** 固定 $M=8$、盒式核，实测偏差随 $N$ 变化：$N=16$ 时 $0.2508$，$N=24$ 时 $6.19\\times10^{-2}$，$N=32$ 时 $2.07\\times10^{-2}$，$N=64$ 时 $2.63\\times10^{-3}$。它确实在变小，但不会自己变成零，因为只要 $N<N_a+M-1$，尾部总有一部分折回来。图把「循环 $-$ 线性」与「补零后」两块并排，就是要让「变小」与「变没」的差别一眼可见。',
                '**读图时还要注意归一化。** 盒式核每个抽头是 $1/M$，直流增益为 $1$，所以平滑后的曲线高度与输入同量级；指数核也做了同样的归一化；交替核的正负号让直流成分互相抵消，输出是高频残差，幅度自然小得多。比较两条卷积曲线时要看形状与位置，不要只看高度。补零后的差异面板纵轴只有 $10^{-16}$ 量级，它在图上贴住零线，这不是「更精确」，而是「卷绕被消掉了」。',
                '**频域视角：每个频率格各乘一个数。** 循环卷积矩阵的特征向量正是复指数，特征值是滤波器的 DFT：$c$ 的第 $k$ 个频率成分等于 $a$ 的第 $k$ 个成分乘 $H_k$。所以盒式核压低高频、交替核把直流特征值变成零、指数核让高频缓慢衰减。图上的循环卷积曲线可以直接读成「逐格缩放后再拼回去」。这条视角也解释了为什么 $N$ 必须装得下线性卷积：循环卷积只认 $N$ 个频率格，装不下的尾部只能挤进已有的格子里，表现出来就是卷绕。',
            ],
        },
        {
            'id': 'resolution',
            'title': '分辨率与不确定性',
            'description': (
                '两个近邻音能否分开，取决于观测时长 $T$，补零只让频率栅格更细。'
                '左上画出加窗后的两条谱线与实测间隔尺；左下扫过不同的观测点数，标出“分开 / 分不开”的边界；'
                '右下是 FFT 与直接 DFT 的对拍误差水平——它只说明谱算对了，不代表两个音能分辨。'
            ),
            'dataSource': 'backend',
            'params': [
                {'name': 'n', 'label': '观测点数 N', 'type': 'int', 'min': 32, 'max': 4096,
                 'default': 256, 'critical': [], 'criticalNote': '',
                 'help': '观测点数，范围 32 到 4096。本页固定观测时长 $T=1$，所以调大 $N$ 只是把同一段记录采得更密、频率栅格更细；峰看起来更瘦，但两个音能否分开仍由 $T$ 决定。读图看「能否分辨 vs 观测点数」的阶梯：$N$ 足够大以后判据不再变化。它和补零倍数 pad 一样只改栅格，不改真实分辨率。'},
                {'name': 'pad', 'label': '补零倍数', 'type': 'int', 'min': 1, 'max': 8,
                 'default': 1, 'critical': [1.0],
                 'criticalNote': 'pad 只加密栅格：二分实测的最小可分辨间隔随 pad 只动百分之几，而随 $T=N\\cdot dt$ 成反比变化——这就是「补零不等于提高分辨率」的可执行判据。',
                 'help': '补零倍数：把加窗后的样本补零到 $N\\cdot pad$ 再变换。调大只把频率栅格从 $1/N$ 加密到 $1/(N\\cdot pad)$，谱线更密、峰位读数更细，但两音之间的谷深不变。读图对比补零前后的谱线与实测最小可分辨间隔：后者几乎不动。它常被误当成「提高分辨率」，真正的办法是加长观测时长 $T$。'},
                {'name': 'df', 'label': '两音间隔（周期数/T）', 'type': 'float', 'min': 0.2,
                 'max': 8.0, 'step': 0.05, 'default': 2.0, 'critical': [1.0, 1.44],
                 'criticalNote': '整数格扫描下四个窗的实测边界都是 $1.0/T$（= Rayleigh 判据；critical $1.0$）'
                                 '——因为两音都落在栅格上时，判据只看主瓣宽度与间隔的关系。'
                                 'Hann 的代价要在**半格**位置上才现形：两音取 $2.25/T$ 与 $3.75/T$ 时矩形窗仍可分，'
                                 'Hann 并成一个包（主瓣 $1.44$ 格 $>1.5$ 格间隔已经贴极限）。',
                 'help': '两个音的频率间隔，单位是「每观测时长的周期数」，即 $\\Delta f=\\kappa/T$。调小让两峰靠近，调到 $1/T$ 以下就会并成一个包；调大则两峰分得很开。读图看两个峰之间是否出现低于矮峰的谷，以及三条分辨率尺的相对长度。它和窗的选择联动：主瓣越宽的窗，越容易在半格位置上并包。'},
                {'name': 'amp2', 'label': '第二音幅度比', 'type': 'float', 'min': 0.05, 'max': 1.0,
                 'step': 0.05, 'default': 1.0, 'critical': [], 'criticalNote': '',
                 'help': '第二个音的幅度比，范围 0.05 到 1。调小让第二个峰变矮，谷相对矮峰更容易满足判据；调到很小时第二个峰可能淹没在第一个峰的裙边里，看起来像分不开。读图看两峰高度差与谷的位置：判据要求谷低于较矮的那个峰。它和频率间隔一起决定「看起来能分」还是「真的能分」。'},
                {'name': 'window', 'label': '窗', 'type': 'enum',
                 'options': ['rect', 'hann', 'hamming', 'blackman'], 'default': 'rect',
                 'critical': [], 'criticalNote': '',
                 'help': '窗函数：矩形、Hann、Hamming 或 Blackman。矩形窗主瓣最窄，两个音在整数格上最容易分开；Hann 等窗压低旁瓣、却加宽主瓣，在半格位置上反而更容易并包。读图看两音谱的峰宽与谷深，再对照实测最小可分辨间隔。它和频率间隔联动：间隔越接近主瓣宽度，换窗的后果越明显。'},
            ],
            'time': {'mode': 'none'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'spec', 'autoscaled': False, 'title': '两音谱（$\\mathrm{dB}$，相对峰值）', 'ylabel': '$\\mathrm{dB}$',
                 'scale': 'linear'},
                {'id': 'threshold', 'autoscaled': False, 'title': '能否分辨 vs 观测长度',
                 'ylabel': '可分辨（$1/0$）', 'scale': 'linear'},
                {'id': 'error', 'autoscaled': True, 'title': '当前 $N$ 的 $|\\mathrm{FFT}-\\mathrm{DFT}|$ 水平（对数）', 'ylabel': '$\\max|\\delta|$',
                 'scale': 'log'},
            ],
            'pitfalls': '把 $N$ 从 $64$ 拖到 $4096$，峰越来越瘦，很容易说「分辨率提高了」；本页固定 $T=N\\cdot dt=1$，所以拖 $N$ 只改变采样密度。判据必须用「两峰之间的谷是否低于矮峰」来判，用「峰有多尖」判会得出相反结论。另外峰值位置用抛物线插值可以做到远好于 $df$ 的精度——定位精度与分辨率不是一回事。',
            'theory': [
                '**两个音的谱是两个 Dirichlet 核之和。** 设两个等幅单音落在 $\\kappa_1/T$ 与 $\\kappa_2/T$。矩形窗截断后，单音的连续谱是 $D_N(x)=\\frac{\\sin\\pi x}{N\\sin(\\pi x/N)}$，两个音就是 $D_N(x-\\kappa_1)+D_N(x-\\kappa_2)$ 再乘相位因子。这个函数在图上就是两条主瓣加一串旁瓣；能否分辨，取决于这两个核在中间叠出的是峰还是谷。',
                '**中点为什么是一个谷。** 令 $d=(\\kappa_2-\\kappa_1)/2$，在正中点 $x=(\\kappa_1+\\kappa_2)/2$ 处两核取值 $D_N(-d)$ 与 $D_N(d)$。$D_N$ 是奇函数，所以两者互为相反数，和恒为零：中点永远是严格的零点。间隔越宽，两个零点离得越开；间隔越小，零点越靠近主瓣，最后被主瓣吞掉。',
                '**Rayleigh 判据为什么是 $1/T$。** 主瓣的零点在间隔 $1/T$ 处，即连续频率差 $1/T$。当两音相差正好一格（$\\kappa_2-\\kappa_1=1$）时，一个音的主瓣零点正好落在另一个音的峰上，中点仍是零，两峰可辨；差小于一格时，两核在中点同号叠加，谷消失，两峰并成一个包。于是最小可分辨间隔约为 $1/T$，量纲上它就是观测时长的倒数。',
                '**窗只改常数，不改这条量纲。** 窗的主瓣更宽，两个音需要隔得更远才分得开：Hann 的 $-3$ dB 主瓣是 $1.44$ 格，Blackman 是 $1.68$ 格。可是在整数格扫描下，四个窗的实测阈值都落在 $1.000/T$——因为两音都落在栅格上时，判据只看「峰之间有没有零」。窗的代价要到半格位置才显形：两音取 $2.25/T$ 与 $3.75/T$，间隔 $1.5$ 格，矩形窗仍能分开，Hann 却并成一个包。',
                '**补零只是把同一条谱线采密。** 补零到 $N\\cdot pad$ 后，频率栅格从 $1/N$ 变成 $1/(N\\cdot pad)$，谱线数量增加，但连续谱还是原来那条。如果两个音在补零前并包，补零后仍然并包——你只是用更细的尺子量同一个包。图里把实测最小可分辨间隔与 Rayleigh 判据并排，又单独标出栅格间距，就是为了让「尺子变细」与「东西变大」不再混为一谈。',
                '**时间与频率不能同时任意准。** 观测越短，主瓣越宽，频率越糊；观测越长，频率越准，但时间定位越差。矩形窗的乘积 $\\Delta f\\cdot T\\approx0.89$，Hann 约 $1.44$，这就是不确定性关系在窗上的具体数值。它解释了为什么「提高频率分辨率」的代价总是「变慢」：想分辨更近的两个音，只能观察更久，不能靠补零或换算法绕过。反过来说，如果想准确定位一个短脉冲发生在哪一刻，就必须接受它很宽的频谱。',
                '**读图要看谷，不要看峰有多尖。** 图上两个局部极大加上它们之间的谷，才是判据的三要素；谷低于较矮的峰才算分得开。峰值位置可以用抛物线插值做到远小于栅格间距的精度，但那是「定位精度」，不是「分辨率」：一个包里的两个音，插值也只会给你一个位置。把 $N$ 调大时峰变瘦只是栅格变密，谷的深度不变，这就是本页最想让人看见的反例。',
                '**参数各自改哪一块画面。** 观测点数 $N$ 改变采样密度，因而改变栅格与谱线数量；补零倍数 $pad$ 只改栅格；两音间隔 $df$ 改两个核的距离；幅度比 $amp2$ 改两个峰的高低；窗改主瓣宽度与旁瓣。读图时先看两条峰之间的谷，再看三条分辨率尺（当前间隔、实测阈值、Rayleigh 判据）谁长谁短，最后看「能否分辨」阶梯与副轴的 $|\\mathrm{FFT}-\\mathrm{DFT}|$。副轴只在证明谱算得对，它和能不能分辨没有关系。',
            ],
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
