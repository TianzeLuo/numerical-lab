r"""2.6 章：小波与多分辨分析。

四个可视化分工（所有数值来自 :mod:`numcore.transform.wavelet`，本文件只做搬运与装配）：

1. ``decomp_tree`` —— 逐层分解树：每层的近似/细节系数分支 + **逐层重构**的累加曲线。
   ``t`` 是层号（``time.mode="scrub"``，从 0 = 只留最粗近似走到 J = 全部还原）。
   副轴的 ``error`` 是「当前这层重构 - 原信号」，最后一段必须落到舍入地板。
2. ``time_scale_map`` —— 时间-尺度系数热力图（平稳小波 SWT，每层都与信号等长）
   与 **STFT 的固定分辨率**并排；另给两种「分辨率元」的宽×高数字，说明面积近似守恒、
   形状完全相反。``error`` 是 SWT 往返误差。
3. ``denoise_threshold`` —— 拖阈值看去噪结果与 **偏差^2/方差/MSE 三条曲线**：
   偏差^2随 :math:`\lambda` 单调增、方差单调减，MSE 在中间取得最小值；
   通用阈值 :math:`\sigma\sqrt{2\ln n}` 的位置在图上用 ``meta`` 报出（实测它偏大）。
4. ``vanishing_moments`` —— 消失矩：:math:`\deg p<N` 的多项式被细节系数抹平到
   :math:`u` 量级，而 :math:`\deg p=N` 不会；同时给出**内部 vs 边界**两组读数，
   说明"边界处理"才是本章最容易骗人的地方。

约定（与 0.0/1.2/2.5 章一致）：``meta`` 只放 number/bool/null，字符串一律走
``set_strings``；对数轴上画不出的 0 给地板 ``FLOOR`` 并配 ``*_valid`` 掩码与
``strings.mask_note``（PROTOCOLS §1.3.1 三件套）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from numcore.transform.wavelet import (
    TEST_SIGNAL_NAMES,
    WAVELET_NAMES,
    bias_variance_curve,
    boundary_pollution,
    coefficient_lengths,
    denoise,
    detail_scale_ratio,
    dwt,
    estimate_sigma_mad,
    iswt,
    level_energies,
    mra_partial,
    optimal_threshold,
    polynomial_signal,
    reconstruction_error,
    scale_time_boxes,
    scalogram,
    snr_db,
    stft_magnitude,
    swt,
    swt_frame_symbol,
    test_signal,
    translation_change,
    universal_threshold,
    vanishing_moment_probe,
    vanishing_moments,
)
from server.nlf import Frame

__all__ = ['MANIFEST', 'compute']

#: 对数轴上「0 画不出来」的统一地板（与 1.2/2.2/2.5 章同一口径）。
FLOOR = 1e-18

#: 双精度单位舍入 :math:`u=2^{-53}`：消失矩演示里的参考线，也是"它不是 0"的量级。
U = 2.0 ** -53

#: 单帧允许的最大信号长度（每帧 ~1 秒的预算）。
MAX_N = 1024

#: 可用的小波族（manifest 的 enum 与这里必须一致）。
WAVELET_OPTIONS = ('haar', 'db2', 'db4', 'db6', 'db8')

#: 边界延拓模式的显示名（enum 值仍是 numcore 的关键字）。
MODE_OPTIONS = ('symmetric', 'zero', 'periodic')

#: 去噪演示用的信号。
DENOISE_SIGNALS = ('bumps', 'blocks', 'tone')

#: 时间-尺度图用的信号。
SCALE_SIGNALS = ('chirp', 'doppler', 'tone', 'blocks')


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
MANIFEST: dict[str, Any] = {
    'chapter': '2.6',
    'title': '小波与多分辨分析',
    'part': 'II',
    'summary': '多分辨分析的嵌套子空间、滤波器组的完美重构、消失矩与逼近阶、阈值去噪的偏差-方差权衡。',
    'vizzes': [
        {
            'id': 'decomp_tree',
            'title': '逐层分解树与逐层重构',
            'description': '近似/细节系数分支逐层展开，右图是只保留前几层时的重构曲线与它的误差。',
            'dataSource': 'backend',
            'params': [
                {'name': 'signal', 'label': '信号', 'type': 'enum',
                 'options': list(TEST_SIGNAL_NAMES), 'default': 'bumps',
                 'critical': [], 'criticalNote': ''},
                {'name': 'n', 'label': '样本数 n', 'type': 'int', 'min': 64, 'max': 1024,
                 'default': 256, 'critical': [256.0],
                 'criticalNote': 'n 决定层数的天花板：第 j 层的系数个数约为 n/2^j，'
                                 '层数越过 log2(n) 就没有系数可分了。'},
                {'name': 'wavelet', 'label': '小波', 'type': 'enum',
                 'options': list(WAVELET_OPTIONS), 'default': 'db4',
                 'critical': [], 'criticalNote': ''},
                {'name': 'level', 'label': '分解层数 J', 'type': 'int', 'min': 1, 'max': 6,
                 'default': 4, 'critical': [4.0],
                 'criticalNote': '最粗近似的系数个数 ≈ n/2^J；J=4（n=256、db4）时只剩 18 个，'
                                 '已与滤波器长度 L=8 同量级——再往下分，"近似"几乎全是延拓的产物。'},
                {'name': 'mode', 'label': '边界延拓', 'type': 'enum',
                 'options': list(MODE_OPTIONS), 'default': 'symmetric',
                 'critical': [], 'criticalNote': ''},
            ],
            'time': {'mode': 'scrub', 'tMax': 1.0, 'dt': 0.1, 'fps': 30, 'label': '层'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'tree', 'autoscaled': False, 'title': '系数树（近似 / 细节，按层）',
                 'ylabel': '层基线 + 系数（显示坐标）', 'scale': 'linear'},
                {'id': 'recon', 'autoscaled': False, 'title': '逐层重构（当前层）',
                 'ylabel': 'x', 'scale': 'linear'},
                {'id': 'error', 'autoscaled': True, 'title': '重构误差 |x_hat - x|',
                 'ylabel': 'max|delta|', 'scale': 'log'},
            ],
            'pitfalls': '系数树的横轴是"系数下标"，不是时间：第 j 层的一个系数对应 2^j 个样本，'
                        '把各层的分支画成等宽会把最粗那层放大 2^J 倍。另一个更常见的误读是'
                        '"细节系数小 = 这层没信息"——细节系数小也可能只是这一层被边界延拓主导：'
                        '本页每帧都报出边界系数占比，读图前先看它。',
            'theory': ['多分辨分析：V_J 嵌套、L2 = clos(∪ V_j) 且 ∩ V_j = {0}',
                       '二尺度方程 phi(t) = sqrt2 sum h_j phi(2t-j)',
                       '完美重构 x = x_J + sum_j w_j（到舍入级）',
                       '系数个数按 2^j 递减，能量按 |coeff|^2 记账'],
        },
        {
            'id': 'time_scale_map',
            'title': '时间-尺度图 vs STFT',
            'description': '平稳小波（SWT）系数热力图与固定窗长 STFT 并排，附两种分辨率元的宽高数字。',
            'dataSource': 'backend',
            'params': [
                {'name': 'signal', 'label': '信号', 'type': 'enum',
                 'options': list(SCALE_SIGNALS), 'default': 'chirp',
                 'critical': [], 'criticalNote': ''},
                {'name': 'n', 'label': '样本数 n', 'type': 'int', 'min': 128, 'max': 512,
                 'default': 256, 'critical': [], 'criticalNote': ''},
                {'name': 'wavelet', 'label': '小波', 'type': 'enum',
                 'options': ['haar', 'db4', 'db8'], 'default': 'db4',
                 'critical': [], 'criticalNote': ''},
                {'name': 'level', 'label': '尺度层数 J', 'type': 'int', 'min': 2, 'max': 5,
                 'default': 4, 'critical': [5.0],
                 'criticalNote': 'J 层之后每层的时间支撑翻倍：J=5（n=256）时最粗那层只剩 8 个'
                                 '独立系数，热力图上会出现明显的横向条纹（那不是信号的纹理）。'},
                {'name': 'stft_window', 'label': 'STFT 窗长', 'type': 'int', 'min': 8,
                 'max': 64, 'default': 32, 'critical': [16.0],
                 'criticalNote': '窗长 16 的频率栅格 Δf = 1/16 与 db4 第 2 层的带宽 1/4 同量级；'
                                 '窗长越大时间分辨率越差，而小波是"低频处自动变宽"。'},
            ],
            'time': {'mode': 'none'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'scale', 'autoscaled': False, 'title': 'SWT 系数（尺度 × 时间）',
                 'ylabel': '尺度层', 'scale': 'linear'},
                {'id': 'stft', 'autoscaled': False, 'title': 'STFT 幅度（频率 × 时间，固定窗）',
                 'ylabel': '频率格', 'scale': 'linear'},
                {'id': 'error', 'autoscaled': True, 'title': 'SWT 往返误差',
                 'ylabel': 'max|delta|', 'scale': 'log'},
            ],
            'pitfalls': '把尺度图的纵轴读成"频率"会低估低频的时间定位能力：第 j 层带宽是 1/2^j，'
                        '但它的时间支撑也按 2^j 变宽，两个方向的乘积（分辨率元面积）几乎不变。'
                        '另外 SWT 的能量比不是常数（本页报出实测值与框架符号的预测值）：'
                        '拿 |系数|^2 的总和当"能量"会随信号频率漂移。',
            'theory': ['SWT 不抽取：每层系数与信号等长，平移一格系数逐位平移',
                       '分辨率元：时间支撑 ~ 2^j、带宽 ~ 1/2^j，面积近似守恒',
                       'STFT 固定窗：时间与频率分辨率同时被钉死',
                       '框架算子符号 S(omega) = 2 + sum_i prod_k |H(2^{k-1}omega)|^2'],
        },
        {
            'id': 'denoise_threshold',
            'title': '阈值去噪：偏差-方差',
            'description': '拖阈值看信号、误差与三条曲线（偏差^2、方差、MSE）；同一 SNR 下软/硬阈值对照。',
            'dataSource': 'backend',
            'params': [
                {'name': 'signal', 'label': '干净信号', 'type': 'enum',
                 'options': list(DENOISE_SIGNALS), 'default': 'bumps',
                 'critical': [], 'criticalNote': ''},
                {'name': 'n', 'label': '样本数 n', 'type': 'int', 'min': 128, 'max': 512,
                 'default': 256, 'critical': [], 'criticalNote': ''},
                {'name': 'sigma', 'label': '噪声水平 sigma', 'type': 'float', 'min': 0.05,
                 'max': 0.6, 'step': 0.01, 'default': 0.3, 'critical': [],
                 'criticalNote': ''},
                {'name': 'lam', 'label': '阈值 lambda', 'type': 'float', 'min': 0.0,
                 'max': 1.6, 'step': 0.01, 'default': 0.28, 'critical': [0.28, 1.0],
                 'criticalNote': '0.28 是默认设置（bumps、sigma=0.3、db4、J=4、软阈值）下'
                                 '蒙特卡洛实测的 MSE 最优点；1.00 是通用阈值 sigma*sqrt(2 ln n)，'
                                 '越过它偏差项开始主导（实测 MSE 甚至超过不做去噪）。'},
                {'name': 'rule', 'label': '阈值规则', 'type': 'enum',
                 'options': ['soft', 'hard'], 'default': 'soft',
                 'critical': [], 'criticalNote': ''},
                {'name': 'wavelet', 'label': '小波', 'type': 'enum',
                 'options': ['haar', 'db4', 'db8'], 'default': 'db4',
                 'critical': [], 'criticalNote': ''},
                {'name': 'level', 'label': '分解层数 J', 'type': 'int', 'min': 1, 'max': 6,
                 'default': 4, 'critical': [], 'criticalNote': ''},
            ],
            'time': {'mode': 'none'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'signals', 'autoscaled': False, 'title': '干净 / 含噪 / 去噪',
                 'ylabel': 'x', 'scale': 'linear'},
                {'id': 'curves', 'autoscaled': True, 'title': '偏差^2、方差、MSE vs lambda',
                 'ylabel': 'MSE', 'scale': 'linear'},
                {'id': 'error', 'autoscaled': True, 'title': '去噪残差 |x_hat - x|',
                 'ylabel': 'max|delta|', 'scale': 'log'},
            ],
            'pitfalls': '把"看起来更平滑"当成更好：硬阈值留下的 Gibbs 式振铃在图上更"干净"，'
                        '但同一 SNR 下它的 MSE 常高于软阈值（本页两条曲线直接对照）。'
                        '另一个陷阱是通用阈值 sigma*sqrt(2 ln n)：它保证的是"不留下噪声"'
                        '而不是"最小 MSE"，实测最优点只有它的一半不到，越过它之后 MSE 反而'
                        '超过完全不去噪。',
            'theory': ['软阈值是大系数也有偏的收缩：bias = lambda（大系数处）',
                       '硬阈值无偏但不连续，方差更大',
                       'MSE(lambda) = bias^2(lambda) + var(lambda)',
                       'MAD 估计 sigma_hat = med(|d_1|)/0.6745'],
        },
        {
            'id': 'vanishing_moments',
            'title': '消失矩：多项式被细节系数抹平',
            'description': 'k 次多项式的细节系数（内部 vs 边界）与"内部量级比 / u"随 k 的跳变。',
            'dataSource': 'backend',
            'params': [
                {'name': 'wavelet', 'label': '小波', 'type': 'enum',
                 'options': list(WAVELET_OPTIONS), 'default': 'db4',
                 'critical': [], 'criticalNote': ''},
                {'name': 'degree', 'label': '多项式次数 k', 'type': 'int', 'min': 0, 'max': 6,
                 'default': 3, 'critical': [4.0],
                 'criticalNote': 'db4 的消失矩阶数 N=4：k <= 3 时内部细节系数贴 u（实测 1e-16），'
                                 'k = 4 时跳到 1e-5 量级——跨过 4 就是"抹平"与"没抹平"的分界。'},
                {'name': 'n', 'label': '样本数 n', 'type': 'int', 'min': 64, 'max': 512,
                 'default': 256, 'critical': [], 'criticalNote': ''},
                {'name': 'level', 'label': '层数 J', 'type': 'int', 'min': 1, 'max': 5,
                 'default': 3, 'critical': [], 'criticalNote': ''},
                {'name': 'mode', 'label': '边界延拓', 'type': 'enum',
                 'options': ['zero', 'symmetric', 'periodic'], 'default': 'zero',
                 'critical': [], 'criticalNote': ''},
            ],
            'time': {'mode': 'none'},
            'requiredAux': ['error'],
            'axes': [
                {'id': 'poly', 'autoscaled': False, 'title': '多项式信号 t^k 与它的细节系数',
                 'ylabel': 'x', 'scale': 'linear'},
                {'id': 'ratio', 'autoscaled': True, 'title': '内部量级比 max|d|/max|a|（对数轴）',
                 'ylabel': 'ratio', 'scale': 'log'},
                {'id': 'error', 'autoscaled': True, 'title': '该模式的往返误差',
                 'ylabel': 'max|delta|', 'scale': 'log'},
            ],
            'pitfalls': '把"细节系数是 0"当成严格成立：实测内部系数是 u 量级（约 1e-16）而不是 0，'
                        '对数轴上 0 只能画地板值。更危险的是只看最大值：多项式的边界系数本来就'
                        '是 O(1)（零延拓把多项式截断了），所以"细节系数最大的是边界那几个"'
                        '并不说明消失矩失效——本页把内部与边界分开报。',
            'theory': ['N 阶消失矩：sum_j j^m g_j = 0（m < N）',
                       'deg p < N 的多项式落在 V_J 里，细节系数为 0（到舍入级）',
                       'deg p >= N 时内部细节系数为 O(1)，与 u 相差若干数量级',
                       '内部/边界必须分开量：延拓方式决定边界系数的大小'],
        },
    ],
}


# --------------------------------------------------------------------------- #
# 帧装配
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已经过 :mod:`server.frames` 规范化与区间校验。"""
    if viz_id == 'decomp_tree':
        return _decomp_tree(params, t)
    if viz_id == 'time_scale_map':
        return _time_scale_map(params, t)
    if viz_id == 'denoise_threshold':
        return _denoise_threshold(params, t)
    if viz_id == 'vanishing_moments':
        return _vanishing_moments(params, t)
    raise KeyError(f'章节 2.6 没有可视化 {viz_id!r}')


def _floor(arr: Any, floor: float = FLOOR) -> np.ndarray:
    """把非正的量压到地板（对数轴上画不出 0；配套 ``*_valid`` 掩码说明哪些是占位）。"""
    return np.maximum(np.asarray(arr, dtype=float), floor)


def _stack_padded(rows: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    """把长度不一的各层系数铺成矩形（空位写 0），并返回各层真实长度。

    空位是**占位 0**、不是系数——每帧都会带 ``*_length`` 数组与 ``strings`` 说明，
    否则"树上有个 0"会被读成"这一层系数是 0"。
    """
    out = np.zeros((len(rows), width), dtype=float)
    lens = np.zeros(len(rows), dtype=float)
    for i, r in enumerate(rows):
        a = np.asarray(r, dtype=float).ravel()
        k = int(min(a.size, width))
        out[i, :k] = a[:k]
        lens[i] = float(a.size)
    return out, lens


def _decomp_tree(params: Mapping[str, Any], t: float) -> Frame:
    n = int(min(int(params['n']), MAX_N))
    name = str(params['wavelet'])
    level = int(params['level'])
    mode = str(params['mode'])
    sig = str(params['signal'])
    x = test_signal(sig, n)
    dec = dwt(x, name, level, mode)

    # 各层系数（第 0 行 = 延拓后的原信号，第 j 行 = 第 j 层）铺成矩形
    width = dec.n_ext
    approx_rows = [np.asarray(a, dtype=float) for a in dec.approx]
    detail_rows = [np.zeros(width, dtype=float)] + [np.asarray(d, dtype=float)
                                                     for d in dec.detail[1:]]
    approx_tree, approx_len = _stack_padded(approx_rows, width)
    detail_tree, detail_len = _stack_padded(detail_rows, width)

    partial = mra_partial(dec)
    stage = int(round(float(t) * level))
    stage = max(0, min(level, stage))
    recon = partial[stage]
    err = float(np.max(np.abs(recon - x)))
    # 「完美重构」有两条实现路径，误差都在 u 量级但**不是同一个数**，两个都要报：
    # pr_error  = idwt（直接逆变换）      mra_error = 逐带求和 x_J + sum w_j
    pr_err = reconstruction_error(x, name, level, mode)
    mra_err = float(np.max(np.abs(partial[-1] - x)))
    scale = max(float(np.max(np.abs(x))), 1e-300)

    # 边界系数占比：各层里"支撑碰到延拓区"的系数占总细节能量的比例（第 1 层最大）
    share = level_energies(dec)
    boundary_share = 0.0
    if dec.offset > 0 and level >= 1:
        d1 = np.asarray(dec.detail[1], dtype=float)
        cut = int(np.ceil(dec.offset / 2.0))
        if 0 < cut < d1.size // 2:
            edge = np.concatenate([d1[:cut], d1[d1.size - cut:]])
            tot = float(np.dot(d1, d1))
            boundary_share = float(np.dot(edge, edge) / tot) if tot > 0 else 0.0

    fr = Frame(chapter='2.6', viz='decomp_tree', kind='frame')
    fr.add('signal', x, unit='', label=f'原信号 {sig}')
    fr.add('approx_tree', approx_tree, unit='',
           label='各层近似系数（行 = 层 0..J，空位为占位 0）')
    fr.add('detail_tree', detail_tree, unit='',
           label='各层细节系数（行 = 层 0..J，第 0 行为 0）')
    fr.add('coeff_length', np.concatenate([approx_len, detail_len]), unit='',
           label='各行真实长度（前 J+1 个是近似、后 J+1 个是细节）')
    fr.add('recon', partial, unit='', label='逐层重构（行 = 已保留的层数）')
    fr.add('error', np.full(n, max(err, FLOOR)), unit='',
           label='当前层重构误差（铺满 n 点，供对数轴）')
    fr.add('energy_share', share, unit='', label='各层能量占比 [a_J, d_J, ..., d_1]')
    fr.add('coeff_count', coefficient_lengths(dec), unit='点',
           label='各层系数个数 [|a_J|, |d_J|, ..., |d_1|]')
    fr.set_meta(n=n, level=level, stage=stage, n_ext=dec.n_ext,
                redundancy=dec.redundancy, pr_error=pr_err, mra_error=mra_err,
                stage_error=err, rel_stage_error=err / scale,
                boundary_share=boundary_share,
                energy_top=float(share[0]), energy_detail_sum=float(share[1:].sum()),
                max_detail=float(np.max(np.abs(np.asarray(dec.detail[1], dtype=float)))),
                t_actual=float(stage) / float(level) if level else 0.0)
    fr.set_strings(status='ok', signal=sig, wavelet=name, mode=mode,
                   boundary_mode={
                       'periodic': '周期延拓：不补样本，变换是 n 维正交变换，能量比恰好 1',
                       'symmetric': '半样本对称反射：反射复制样本，系数能量比 > 1（冗余度 n_ext/n）',
                       'zero': '零延拓：补零不添能量（能量比 1），但边界系数 O(1)',
                   }[mode],
                   nonfinite_policy='对数轴上的 0 用 1e-18 地板；树里的空位是占位 0，真实长度看 coeff_length',
                   mask_note='approx_tree/detail_tree 的空位是补零对齐用的，不是系数值',
                   note=(f'{mode} 延拓后 n_ext = {dec.n_ext}（冗余 {dec.redundancy:.3f}）；'
                         f'J={level} 层往返误差：idwt 路径 {pr_err:.2e}、逐带求和路径 '
                         f'{mra_err:.2e}（都在 u 量级，但不是同一个数）；'
                         f'第 1 层边界系数占细节能量 {boundary_share:.1%}'))
    return fr


def _time_scale_map(params: Mapping[str, Any], t: float) -> Frame:
    n = int(min(int(params['n']), MAX_N))
    name = str(params['wavelet'])
    level = int(params['level'])
    win = int(params['stft_window'])
    sig = str(params['signal'])
    x = test_signal(sig, n)

    sta = swt(x, name, level)
    scal = scalogram(sta)
    back = iswt(sta)
    pr_err = float(np.max(np.abs(back - x))) / max(float(np.max(np.abs(x))), 1e-300)

    hop = max(1, win // 4)
    t_idx, freq, mag = stft_magnitude(x, win, hop)
    # 固定分辨率对照：STFT 的窗长/带宽恒定；小波逐层变宽变窄
    boxes = scale_time_boxes(name, level, n, win)

    # SWT 的能量比（信号相关）与框架符号预测的平均值：两个数必须一起报
    coeff = sta.coefficients()
    ratio = float(np.dot(coeff, coeff) / float(np.dot(x, x)))
    symbol_mean = float(np.mean(swt_frame_symbol(name, level)))

    fr = Frame(chapter='2.6', viz='time_scale_map', kind='frame')
    fr.add('signal', x, unit='', label=f'信号 {sig}')
    fr.add('scalogram', scal, unit='', label='SWT 系数（行 = [a_J, d_J, ..., d_1]，每行全长）')
    fr.add('stft_map', mag, unit='', label=f'STFT 幅度（窗长 {win}，跳步 {hop}）')
    fr.add('stft_time', t_idx, unit='点', label='STFT 窗中心下标')
    fr.add('stft_freq', freq, unit='周期/样本', label='STFT 归一化频率')
    fr.add('wav_width', boxes['wavelet_width_norm'], unit='', label='小波各层时间支撑（占全长比例）')
    fr.add('wav_band', boxes['wavelet_band'], unit='', label='小波各层归一化带宽')
    fr.add('stft_width', boxes['stft_width_norm'], unit='', label='STFT 时间窗宽（占全长比例，恒定）')
    fr.add('stft_band', boxes['stft_band'], unit='', label='STFT 带宽（恒定）')
    fr.add('cell_area', boxes['wavelet_width_norm'] * boxes['wavelet_band'], unit='',
           label='小波分辨率元面积（时间 × 带宽，近似守恒）')
    fr.add('error', np.full(n, max(pr_err, FLOOR)), unit='',
           label='SWT 往返误差（相对，铺满 n 点）')
    fr.set_meta(n=n, level=level, n_bands=level + 1, stft_window=win, stft_hop=hop,
                n_stft_time=float(t_idx.size), n_stft_freq=float(freq.size),
                pr_error=pr_err, swt_energy_ratio=ratio, swt_symbol_mean=symbol_mean,
                swt_redundancy=sta.redundancy,
                cell_area_min=float(np.min(boxes['wavelet_width_norm'] * boxes['wavelet_band'])),
                cell_area_max=float(np.max(boxes['wavelet_width_norm'] * boxes['wavelet_band'])),
                t_actual=float(t))
    shift = translation_change(x, name, min(level, 4), 'dwt')
    fr.set_strings(status='ok', signal=sig, wavelet=name,
                   nonfinite_policy='往返误差取 1e-18 地板（贴舍入地板时对数轴画不出来）',
                   note=(f'SWT 系数能量比实测 {ratio:.4f}，框架符号平均 {symbol_mean:.4f}'
                         f'（不是常数：低频取到 2^J，白噪声约为 2^(J-1)+1）；'
                         f'往返误差 {pr_err:.2e}'),
                   shift_note=(f'同一信号平移一格：DWT 第 1 层系数相对变化 '
                               f'{shift["relative"]:.2f}（尺度 {shift["scale"]:.3f}），'
                               f'SWT 系数逐位平移、变化恰好 0'))
    return fr


def _denoise_threshold(params: Mapping[str, Any], t: float) -> Frame:
    n = int(min(int(params['n']), MAX_N))
    name = str(params['wavelet'])
    level = int(params['level'])
    sig = str(params['signal'])
    sigma = float(params['sigma'])
    lam = float(params['lam'])
    rule = str(params['rule'])
    x = test_signal(sig, n)

    # 固定噪声实现：拖动 lambda 时看到的曲线变化全部来自阈值，不来自换噪声
    noise = np.sin(np.arange(n, dtype=float) * 12.9898) * np.cos(
        np.arange(n, dtype=float) * 4.1414)
    noise = noise / max(float(np.std(noise)), 1e-300)
    noisy = x + sigma * noise
    clean_dec = dwt(noisy, name, level, 'symmetric')
    sigma_hat = estimate_sigma_mad(clean_dec.detail[1])

    grid = np.linspace(0.0, 1.6, 33)
    curve = bias_variance_curve(x, sigma, grid, name=name, level=level,
                                mode='symmetric', rule=rule, n_trials=48,
                                seed=20261018)
    opt = optimal_threshold(curve['lambda'], curve['mse'])
    uni = universal_threshold(sigma, n)
    i_here = int(np.argmin(np.abs(grid - lam)))
    i_uni = int(np.argmin(np.abs(grid - uni)))
    den = denoise(noisy, name, level, 'symmetric', lam, rule)
    err = float(np.max(np.abs(den - x)))
    scale = max(float(np.max(np.abs(x))), 1e-300)
    mse_noisy = float(np.mean((noisy - x) ** 2))
    # 两个 MSE 必须分开报：画出来的那条信号的单次实现读数，
    # 与蒙特卡洛曲线在最近栅格点上的读数（前者有实现噪声、后者有栅格误差）。
    mse_here = float(np.mean((den - x) ** 2))
    mse_curve_here = float(curve['mse'][i_here])

    fr = Frame(chapter='2.6', viz='denoise_threshold', kind='frame')
    fr.add('clean', x, unit='', label=f'干净信号 {sig}')
    fr.add('noisy', noisy, unit='', label=f'含噪信号（sigma={sigma:g}）')
    fr.add('denoised', den, unit='', label=f'{rule} 阈值去噪（lambda={lam:g}）')
    fr.add('error', np.full(n, max(err, FLOOR)), unit='',
           label='去噪残差（铺满 n 点，供对数轴）')
    fr.add('lambda_grid', grid, unit='', label='阈值栅格')
    fr.add('mse_curve', _floor(curve['mse']), unit='', label='MSE(lambda)（蒙特卡洛）')
    fr.add('bias2_curve', _floor(curve['bias2']), unit='', label='偏差^2(lambda)')
    fr.add('var_curve', _floor(curve['var']), unit='', label='方差(lambda)')
    fr.add('mse_noisy_line', np.full(grid.size, max(mse_noisy, FLOOR)), unit='',
           label='完全不去噪的 MSE（参考线）')
    fr.add('universal_line', np.full(grid.size, max(uni, FLOOR)), unit='',
           label='通用阈值 sigma*sqrt(2 ln n) 的位置（横轴参考）')
    fr.set_meta(n=n, level=level, sigma=sigma, sigma_hat=sigma_hat,
                sigma_hat_bias=sigma_hat / sigma - 1.0 if sigma > 0 else 0.0,
                lam=lam, mse=mse_here, mse_curve_here=mse_curve_here,
                lam_grid_here=float(grid[i_here]), mse_noisy=mse_noisy,
                mse_opt=opt['mse'], lam_opt=opt['lambda'], lam_grid=opt['lambda_grid'],
                universal=uni, universal_over_opt=uni / opt['lambda'] if opt['lambda'] > 0 else 0.0,
                mse_at_universal=float(curve['mse'][i_uni]),
                bias2_curve_here=float(curve['bias2'][i_here]),
                var_curve_here=float(curve['var'][i_here]),
                snr_in=snr_db(x, noisy), snr_out=snr_db(x, den),
                rel_error=err / scale, n_trials=float(curve['n_trials'][0]),
                t_actual=float(t))
    fr.set_strings(status='ok', signal=sig, wavelet=name, rule=rule,
                   nonfinite_policy='三条曲线取 1e-18 地板（对数轴/零值），MSE 的零点已在 meta 里另报',
                   note=(f'实测最优阈值 {opt["lambda"]:.3f}（栅格 {opt["lambda_grid"]:.3f}），'
                         f'通用阈值 {uni:.3f} 是它的 {uni / opt["lambda"]:.2f} 倍；'
                         f'通用阈值处的 MSE {curve["mse"][i_uni]:.5f} 对比完全不去噪 '
                         f'{mse_noisy:.5f}'),
                   rule_note=('软阈值：大系数也被收缩 lambda，偏差随 lambda 单调增、'
                              '方差单调减（实测逐点成立）'
                              if rule == 'soft' else
                              '硬阈值：大系数原样保留（无偏）但不连续，'
                              '偏差^2 与方差都只是"趋势上"此消彼长（实测有非单调段）'),
                   sigma_note=(f'MAD 估计 sigma_hat = {sigma_hat:.4f}，'
                               f'相对偏差 {100.0 * (sigma_hat / sigma - 1.0):+.2f}%'
                               f'（单次实现；该估计量的标准差约为 sigma 的 9%）'))
    return fr


def _vanishing_moments(params: Mapping[str, Any], t: float) -> Frame:
    n = int(min(int(params['n']), MAX_N))
    name = str(params['wavelet'])
    degree = int(params['degree'])
    level = int(params['level'])
    mode = str(params['mode'])
    N = vanishing_moments(name)
    x = polynomial_signal(n, degree)
    dec = dwt(x, name, level, mode)

    width = dec.n_ext
    detail_rows = [np.zeros(width, dtype=float)] + [np.asarray(d, dtype=float)
                                                     for d in dec.detail[1:]]
    detail_tree, detail_len = _stack_padded(detail_rows, width)

    probe = vanishing_moment_probe(name, degree, n, level, mode)
    degrees = np.arange(0, N + 3, dtype=float)
    interior = np.zeros(degrees.size, dtype=float)
    boundary = np.zeros(degrees.size, dtype=float)
    for i, k in enumerate(degrees):
        p = vanishing_moment_probe(name, int(k), n, level, mode)
        interior[i] = p['interior_max']
        boundary[i] = p['boundary_max']
    # 内部比值可能是**严格 0**（整段抵消）：对数轴上给地板 + 掩码（三件套）
    valid = (interior > 0.0).astype(float)
    interior_plot = np.where(valid > 0.5, interior, FLOOR)
    n_floored = int(np.count_nonzero(valid < 0.5))

    pr_err = reconstruction_error(x, name, level, mode)
    ratio_u = probe['interior_max'] / U
    cur = detail_scale_ratio(dec, 1, name=name)
    pr = boundary_pollution(x, name, min(level, 4), 'periodic' if n % (2 ** min(level, 4)) == 0
                            else 'zero')

    fr = Frame(chapter='2.6', viz='vanishing_moments', kind='frame')
    fr.add('signal', x, unit='', label=f'多项式 t^{degree}')
    fr.add('detail_tree', detail_tree, unit='',
           label='各层细节系数（行 = 层 0..J，空位为占位 0）')
    fr.add('detail_length', detail_len, unit='', label='各层细节系数的真实长度')
    fr.add('degree_axis', degrees, unit='', label='多项式次数 k')
    fr.add('interior_ratio', interior_plot, unit='',
           label='内部系数 max|d|/max|a|（0 已压到 1e-18 地板）')
    fr.add('interior_valid', valid, unit='',
           label='1 = 该点是实测值，0 = 实测严格为 0、图上画的是地板')
    fr.add('boundary_ratio', boundary, unit='', label='边界系数 max|d|/max|x|（同层）')
    fr.add('u_line', np.full(degrees.size, U), unit='',
           label='单位舍入 u = 2^-53（参考线：消失矩成立时内部比值应贴住它）')
    fr.add('error', np.full(n, max(pr_err, FLOOR)), unit='',
           label=f'{mode} 模式的往返误差（铺满 n 点）')
    fr.add('level_ratio', probe['interior_ratio'], unit='',
           label='当前次数下逐层内部量级比')
    fr.add('level_boundary', probe['boundary_ratio'], unit='',
           label='当前次数下逐层边界量级比')
    fr.set_meta(n=n, degree=degree, moments=N, level=level,
                interior_max=probe['interior_max'], boundary_max=probe['boundary_max'],
                ratio_over_u=ratio_u, killed=bool(degree < N and probe['interior_max'] < 1e-12),
                level1_ratio=cur['ratio'], n_interior=cur['n_interior'],
                n_ratio_floored=float(n_floored), u=U, pr_error=pr_err,
                pollution_level1=float(pr['level_max'][1]) if level >= 1 else 0.0,
                pollution_span1=float(pr['physical_span'][1]) if level >= 1 else 0.0,
                t_actual=float(t))
    fr.set_strings(status='ok', wavelet=name, mode=mode,
                   nonfinite_policy=('内部比值可能严格为 0（多项式被精确抵消）：对数轴上写 1e-18 地板，'
                                     '并用 interior_valid 掩码标注'),
                   mask_note='interior_valid = 0 的点不是"很小"，而是实测 0；读图先看掩码',
                   note=(f'{name} 的消失矩阶数 N={N}：k={degree} 时内部比值 '
                         f'{probe["interior_max"]:.2e}（= {ratio_u:.1f} u），'
                         f'边界比值 {probe["boundary_max"]:.2e}；'
                         f'k<{N} 贴 u、k>={N} 跳到 1e-5 量级'),
                   boundary_note=(f'同一条多项式上，"不做延拓"（纯周期环绕）与延拓口径的第 1 层'
                                  f'系数最大偏差 {pr["level_max"][1]:.4f}，'
                                  f'影响 {pr["physical_span"][1]:.0f} 个样本'))
    return fr
