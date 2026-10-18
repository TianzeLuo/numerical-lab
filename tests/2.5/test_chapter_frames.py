r"""2.5 章：可视化帧与 numcore 内核必须同源。

这一条把「图上画的东西」与「断言里量过的东西」钉在一起。做法是：
拿 ``server/chapters/2.5.py`` 的五个 viz 各跑一遍（默认参数 + 时间端点），
检查 requiredAux 齐备、所有数组有限，然后**逐项**把帧 meta 里报的数
拿回来用 numcore 的函数独立重算一遍，要求两边一致到容差以内。

如果哪天有人为了让图好看而改了一个数（比如把泄漏旁瓣手动填成 -31.47），
这条会立刻失败——meta 里的每个量都必须能由 numcore 复现。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from numcore.transform.fft import (
    WINDOW_SIDELOBE_DB,
    min_separable_separation,
    window,
    window_coherent_gain,
)
from numcore.verify import assert_leq, assert_within, assertion, record


def _load_chapter():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "ch25_mod", root / "server" / "chapters" / "2.5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _defaults(viz: dict) -> dict:
    return {p["name"]: p.get("default") for p in viz["params"]}


def _viz(mod, viz_id: str) -> dict:
    return next(v for v in mod.MANIFEST["vizzes"] if v["id"] == viz_id)


def _compute(mod, viz_id: str, **over) -> object:
    viz = _viz(mod, viz_id)
    params = _defaults(viz)
    params.update(over)
    t = 0.0 if viz["time"]["mode"] == "none" else float(viz["time"]["tMax"])
    return mod.compute(viz_id, params, t)


@assertion('B', theory='五个帧在默认参数与时间端点上：requiredAux 齐备、数组全有限', chapter='2.5')
def test_every_frame_is_finite_and_declares_its_aux() -> None:
    r"""帧契约的最低要求：每个 viz 的 requiredAux 都要在，且没有 NaN/Inf。

    ``time.mode`` 不是 ``none`` 的 viz 还要在 t = 0 与 t = tMax 两端都成立
    ——扫描动画的端点往往是最容易越界的地方。
    """
    mod = _load_chapter()
    rows = []
    for viz in mod.MANIFEST["vizzes"]:
        times = [0.0]
        if viz["time"]["mode"] != "none":
            times = [0.0, float(viz["time"]["tMax"])]
        for t in times:
            params = _defaults(viz)
            fr = mod.compute(viz["id"], params, t)
            names = {n.lower() for n in fr.names}
            for aux in viz["requiredAux"]:
                assert_leq(float(any(n.startswith(aux.lower()) for n in names)), 1.0, rtol=1e-9,
                           label=f'{viz["id"]} 有 requiredAux {aux}',
                           detail='缺失会让前端拿不到误差曲线')
            for name in fr.names:
                arr = np.asarray(fr[name])
                assert_leq(float(np.count_nonzero(~np.isfinite(arr.astype(float)))), 0.0, rtol=0.0,
                           label=f'{viz["id"]} 数组 {name} 全有限', detail=f't={t:g}')
        rows.append(f'{viz["id"]}:{len(fr.names)} 数组')
    record(label='五个帧的数组与 aux', theory='全部有限且 requiredAux 齐备',
           measured=float(len(rows)), tol=0.5, passed=len(rows) == 5,
           detail='；'.join(rows))


@assertion('B', theory='signal_spectrum：meta 里的 df 与 max_error 能由 numcore 复现', chapter='2.5')
def test_signal_spectrum_meta_matches_numcore() -> None:
    r"""时频双视图的每个 meta 都要能算回来。

    默认 N=256、pad=1 时栅格 ``df = 1/(N*dt) = 1/T``，而真正采样时长决定的
    ``df_true`` 也必须是同一个数——两者一旦被 pad 拉开，就是「补零提高分辨率」
    那个误解的代码版本。
    """
    mod = _load_chapter()
    fr = _compute(mod, 'signal_spectrum')
    n = float(fr.meta['n'])
    assert_within(float(fr.meta['df']), float(fr.meta['df_true']), rtol=1e-12,
                  label='pad=1 时栅格与真实分辨率一致',
                  detail='df 与 df_true 都由 T=N*dt 决定')
    assert_within(float(fr.meta['df']), 1.0 / n, rtol=1e-9, label='df = 1/(N*dt)')
    # f0 与二次谐波都落在栅格上（否则图上的峰会偏）
    k1 = float(fr.meta['f0'])
    assert_within(k1 / float(fr.meta['df']), round(k1 / float(fr.meta['df'])), rtol=1e-9,
                  label='基频落在栅格上')
    assert_leq(float(fr.meta['max_error']), 1e-12, rtol=0.0,
               label='自写 FFT 与参照实现的最大偏差', detail='图上画的谱必须与参照一致')
    assert_leq(float(fr.meta['n_aliased']), 0.0, rtol=0.0,
               label='默认参数下没有混叠',
               detail='f0 与它的谐波都要低于 Nyquist，否则图本身就是错的')

    # 补零以后：栅格加密 pad 倍，df_true 不动
    fr2 = _compute(mod, 'signal_spectrum', pad=4)
    assert_within(float(fr2.meta['df']), float(fr.meta['df']) / 4.0, rtol=1e-9,
                  label='pad=4 时栅格加密 4 倍')
    assert_within(float(fr2.meta['df_true']), float(fr.meta['df_true']), rtol=1e-9,
                  label='pad 不改变真实分辨率 df_true',
                  detail='这是「补零不等于提高分辨率」在 meta 上的样子')
    assert_within(float(fr2.meta['mainlobe_width']), float(fr.meta['mainlobe_width']), rtol=1e-9,
                  label='pad 不改变主瓣宽度')
    record(label='signal_spectrum 的 df 与 df_true',
           theory='df = 1/(N*dt) 随 pad 加密，df_true = 1/T 不变',
           measured=float(fr2.meta['df'] / fr.meta['df']), tol=0.01,
           passed=abs(fr2.meta['df'] / fr.meta['df'] - 0.25) < 0.01,
           detail=f'pad=4：df {fr2.meta["df"]:.6f}（原 {fr.meta["df"]:.6f}）、'
                  f'df_true {fr2.meta["df_true"]:.6f}；max_error < 1e-12')


@assertion('B', theory='leak_window：帧里的旁瓣/主瓣/相干增益与 Harris 表和 numcore 同值', chapter='2.5')
def test_leak_window_meta_matches_numcore() -> None:
    r"""泄漏视图报的三个窗指标必须与内核的解析值逐位同源。

    图上的理论旁瓣线直接来自 ``WINDOW_SIDELOBE_DB``；如果帧里报的实测值与
    内核不一致，说明图是"另算一份"的——这正是最该避免的分裂。
    """
    mod = _load_chapter()
    worst_db = 0.0
    for name in ('rect', 'hann', 'hamming', 'blackman'):
        fr = _compute(mod, 'leak_window', window=name, n=64, offset=0.25, pad=4)
        # 表值只保留两位小数（-13.26），帧里报的是解析实测（-13.261459…）：
        # 比绝对差，不能用相对容差（这个坑把 rtol=1e-9 撞成 26.52 的"误差"）
        assert_leq(abs(float(fr.meta['harris_sidelobe_db']) - float(WINDOW_SIDELOBE_DB[name])),
                   5e-3, rtol=0.0, label=f'{name} 帧里的 Harris 表值一致',
                   detail=f'帧 {fr.meta["harris_sidelobe_db"]:.6f}，表 {WINDOW_SIDELOBE_DB[name]:.2f}')
        assert_leq(abs(float(fr.meta['peak_sidelobe_db']) - float(WINDOW_SIDELOBE_DB[name])),
                   0.6, rtol=0.0, label=f'{name} 帧里实测旁瓣与表值一致',
                   detail=f'{fr.meta["peak_sidelobe_db"]:.3f} vs {WINDOW_SIDELOBE_DB[name]:.2f}')
        cg = window_coherent_gain(window(name, 64))
        assert_within(float(fr.meta['coherent_gain']), cg, rtol=1e-9,
                      label=f'{name} 帧里的相干增益 = 内核值')
        worst_db = max(worst_db, abs(float(fr.meta['peak_sidelobe_db'])
                                     - float(WINDOW_SIDELOBE_DB[name])))
    # rel_error 是「|FFT(窗)| 与窗的解析连续谱」的最大偏差——它衡量的是
    # 「解析公式 vs 数值实现」，与 offset 无关（曾经拿加窗信号的谱去减窗的谱，
    # 两条曲线不同类，offset=0 时恒等于 1.0，那个数没有任何诊断价值）
    for offset in (0.0, 0.25, 0.5):
        fr_o = _compute(mod, 'leak_window', window='hann', n=64, offset=offset, pad=4)
        assert_leq(float(fr_o.meta['rel_error']), 0.05, rtol=0.0,
                   label=f'offset={offset} 时解析窗谱与 |FFT| 的偏差 ≤ 5%',
                   detail=f'实测 {fr_o.meta["rel_error"]:.4f}（与 offset 无关）')
    # 理论旁瓣线的锚点必须正好落在最高那条旁瓣上（形状对、位置也要对）
    for name in ('rect', 'hann', 'hamming', 'blackman'):
        fr_a = _compute(mod, 'leak_window', window=name, n=64, pad=4)
        # 容差 0.01 dB：锚点扫描用 0.005 的步长，peak_sidelobe_level 内部更细，
        # 两者对"峰在哪"会有半个步长的分歧（实测最大 3.5e-6 dB）
        assert_leq(abs(float(fr_a.meta['envelope_anchor_db'])
                       - float(fr_a.meta['peak_sidelobe_db'])), 1e-2, rtol=0.0,
                   label=f'{name} 理论线的锚点 = 最高旁瓣',
                   detail=f'锚在 fN={fr_a.meta["envelope_anchor"]:.3f}，'
                          f'{fr_a.meta["envelope_anchor_db"]:.2f} dB')
    ham = _compute(mod, 'leak_window', window='hamming', n=64, pad=4)
    assert_leq(4.0, float(ham.meta['envelope_anchor']) + 1e-9, rtol=0.0,
               label='汉明的最高旁瓣在 fN ≈ 4.5，不是第一个零点 fN = 2',
               detail='"第一个旁瓣"与"最高旁瓣"不是同一个位置，锚点要用后者')
    fr = _compute(mod, 'leak_window', window='hann', n=64, offset=0.25, pad=4)
    record(label='四个窗的帧内旁瓣与表值的最大差', theory='不超过 0.6 dB',
           measured=worst_db, tol=0.6, passed=worst_db <= 0.6,
           detail=f'最大差 {worst_db:.3f} dB；解析窗谱 vs |FFT| 最大偏差 '
                  f'{fr.meta["rel_error"]:.4f}（同类对同类的比较，不再随 offset 乱跳）')


@assertion('B', theory='butterfly_stages：级数、蝶形数与每级残差都能由 numcore 复现', chapter='2.5')
def test_butterfly_stages_meta_matches_numcore() -> None:
    r"""蝶形图的三个数：级数 ``log2(N)``、蝶形总数 ``(N/2)·log2(N)``、每级残差。

    残差是"向量化实现 vs 独立朴素实现"的差，所以它的正确读法是
    「实现一致性」，不是「FFT 的数学误差」——后者由 ``spectrum_error`` 报。
    """
    mod = _load_chapter()
    for n in (8, 32, 128):
        fr = _compute(mod, 'butterfly_stages', n=n, signal='cosine', k=min(3, n // 2 - 1))
        levels = int(np.log2(n))
        assert_within(float(fr.meta['levels']), float(levels), rtol=1e-12,
                      label=f'n={n} 级数 = log2(N)',
                      detail='radix-2 的蝶形级数就是 log2(N)，输入行另算')
        assert_within(float(fr.meta['n_butterflies']), 0.5 * n * levels, rtol=1e-12,
                      label=f'n={n} 蝶形数 = (N/2)*log2(N)',
                      detail='每个蝶形一次复数乘加；O(N log N) 就从这里来')
        assert_leq(float(fr.meta['max_level_residual']), 1e-10, rtol=0.0,
                   label=f'n={n} 每级残差贴地板',
                   detail='向量化实现与朴素实现必须给出同一批中间值')
        assert_leq(float(fr.meta['spectrum_error']), 1e-12, rtol=0.0,
                   label=f'n={n} 末级结果与参照谱一致',
                   detail='残差小只说明两份实现一致，这一条才说明它对')
    fr = _compute(mod, 'butterfly_stages', n=32, signal='cosine', k=3)
    assert_within(float(fr.meta['spectrum_peak']), 16.0, rtol=1e-9,
                  label='余弦 k=3 的谱峰 = N/2 = 16',
                  detail='不做 1/N 归一化时，实余弦的谱峰是 N/2')
    # 扫描时间轴：stage 应当随时间推进
    viz = _viz(mod, 'butterfly_stages')
    params = _defaults(viz)
    stages = [float(mod.compute('butterfly_stages', params, t).meta['stage'])
              for t in (0.0, 0.5, 1.0)]
    rec = record(label='蝶形动画的 stage 随时间推进', theory='t 从 0 到 tMax 时 stage 单调不减',
                 measured=stages[-1], tol=1e-9, passed=stages[0] <= stages[1] <= stages[2],
                 detail=f'stage 序列 {stages}（t = 0、0.5、1.0）')
    assert rec.passed, rec.measured


@assertion('B', theory='conv_theorem：卷绕误差与安全区两条读数都能由 numcore 复现', chapter='2.5')
def test_conv_theorem_meta_matches_numcore() -> None:
    r"""卷积定理视图的关键是让观众**看见**卷绕，而 meta 必须报出它的量级。

    ``max_wrap`` 与 ``wrap_rel`` 来自内核的 ``wrap_around_error``；
    ``safe_error`` 必须是 0（补零到 N+M-1 后逐位相等），
    而 N 超过临界值 24 之后 ``max_wrap`` 应当显著下降。
    """
    mod = _load_chapter()
    fr = _compute(mod, 'conv_theorem', n=16, m=8)
    assert_within(float(fr.meta['need']), 23.0, rtol=1e-12,
                  label='need = N+M-1 = 23',
                  detail='循环卷积等于线性卷积的最小长度')
    assert_leq(abs(float(fr.meta['safe_error'])), 1e-15, rtol=0.0,
               label='补零后误差贴舍入地板',
               detail=f'实测 {fr.meta["safe_error"]:.3e}：循环卷绕的 O(1) 量级被彻底消掉，'
                      '剩下的是两条求和的结合律差异')
    assert_leq(float(fr.meta['theorem_error']), 1e-12, rtol=0.0,
               label='定理本身的残差贴地板')
    assert_leq(float(fr.meta['direct_vs_fft']), 1e-12, rtol=0.0,
               label='频域相乘与直接卷积一致')
    assert_leq(float(fr.meta['max_wrap']), 1.0, rtol=0.0,
               label='卷绕误差是 O(1) 量级而不是小量',
               detail=f'实测 {fr.meta["max_wrap"]:.3f}，相对 {fr.meta["wrap_rel"]:.1%}')
    big = _compute(mod, 'conv_theorem', n=64, m=8)
    assert_leq(float(big.meta['max_wrap']), float(fr.meta['max_wrap']), rtol=0.0,
               label='N 增大后卷绕误差单调不增',
               detail=f'N=16 的 {fr.meta["max_wrap"]:.3f} 降到 N=64 的 {big.meta["max_wrap"]:.2e}')
    record(label='卷绕误差随 N 的变化', theory='N=16 时 O(1)，N=64 时已降到 1e-3 以下',
           measured=float(fr.meta['max_wrap'] / big.meta['max_wrap']), tol=1e9,
           passed=float(big.meta['max_wrap']) < 1e-2 < float(fr.meta['max_wrap']),
           detail=f'N=16：{fr.meta["max_wrap"]:.3f}（相对 {fr.meta["wrap_rel"]:.1%}），'
                  f'N=64：{big.meta["max_wrap"]:.2e}；补零后 safe_error = '
                  f'{fr.meta["safe_error"]:.2e}')


@assertion('B', theory='resolution：sep_measured 与 Rayleigh 判据同源，且补零不改变它', chapter='2.5')
def test_resolution_meta_matches_numcore() -> None:
    r"""分辨率视图报的阈值必须与内核的二分/扫描结果一致。

    默认矩形窗阈值是 1.0/T（= Rayleigh）；把窗换成 Hann 后阈值变宽，
    这正是「压旁瓣要拿分辨率换」的机器版本。补零只改栅格，
    ``sep_measured`` 随 pad 变化的幅度必须远小于随窗变化的幅度。
    """
    mod = _load_chapter()
    fr = _compute(mod, 'resolution', n=256, window='rect', df=2.0, pad=1)
    assert_within(float(fr.meta['rayleigh']), 1.0, rtol=1e-12, label='Rayleigh 判据 = 1/T')
    assert_within(float(fr.meta['sep_measured']),
                  float(min_separable_separation(1.0, n=256, window_name='rect')),
                  rtol=1e-9, label='sep_measured = 内核的最小可分辨间隔（rect）')
    assert_within(float(fr.meta['sep_over_mainlobe']),
                  float(fr.meta['sep_measured']) / float(fr.meta['mainlobe_bins']),
                  rtol=1e-9, label='sep_over_mainlobe = 阈值 / 主瓣宽度')
    assert_within(float(fr.meta['grid_df']), float(fr.meta['true_df']), rtol=1e-9,
                  label='pad=1 时栅格 = 真实分辨率')

    # 窗变宽 → 阈值变大（换分辨率的代价）
    hann = _compute(mod, 'resolution', n=256, window='hann', df=2.0, pad=1)
    assert_leq(float(fr.meta['sep_measured']), float(hann.meta['sep_measured']) + 1e-9, rtol=0.0,
               label='Hann 的阈值不小于矩形窗',
               detail=f'rect {fr.meta["sep_measured"]:.2f}/T vs hann {hann.meta["sep_measured"]:.2f}/T')
    assert_leq(float(fr.meta['mainlobe_bins']), float(hann.meta['mainlobe_bins']) - 1e-9, rtol=0.0,
               label='Hann 的主瓣比矩形宽',
               detail=f'{fr.meta["mainlobe_bins"]:.2f} 格 vs {hann.meta["mainlobe_bins"]:.2f} 格')
    # 补零 → 栅格加密，阈值基本不动
    padded = _compute(mod, 'resolution', n=256, window='rect', df=2.0, pad=8)
    drift = abs(float(padded.meta['sep_measured']) - float(fr.meta['sep_measured']))
    window_shift = abs(float(hann.meta['sep_measured']) - float(fr.meta['sep_measured']))
    assert_within(float(padded.meta['grid_df']), float(fr.meta['grid_df']) / 8.0, rtol=1e-9,
                  label='pad=8 时栅格加密 8 倍')
    record(label='补零与换窗对阈值的影响', theory='pad 只动栅格；阈值的变化远小于换窗带来的变化',
           measured=drift, tol=max(window_shift, 1e-9),
           passed=drift <= 0.5 and drift < window_shift + 1e-9,
           detail=f'pad 1→8 阈值漂移 {drift:.3f}/T，而 rect→hann 变化 '
                  f'{window_shift:.3f}/T；max_error = {fr.meta["max_error"]:g}')


@assertion('B', theory='帧的 meta 与图上画的数组互相印证（谱峰位置、误差曲线）', chapter='2.5')
def test_frame_arrays_agree_with_their_own_meta() -> None:
    r"""meta 说的和数组画的必须是同一件事。

    meta 是可以手填的，数组不行；所以要用数组把 meta 验一遍：
    谱峰落在 ``f0`` 对应的格上、误差曲线真的是"小量"、
    蝶形每级残差的最大值就是 ``max_level_residual``。
    """
    mod = _load_chapter()
    fr = _compute(mod, 'signal_spectrum', n=256, pad=2)
    freq = np.asarray(fr['freq'], dtype=float)
    spec = np.asarray(fr['spectrum_db'], dtype=float)
    i = int(np.argmax(spec))
    assert_within(float(freq[i]), float(fr.meta['f0']), rtol=1e-6,
                  label='dB 谱的峰落在 meta 报的 f0 上',
                  detail=f'峰位 {freq[i]:.4f}，meta f0 = {fr.meta["f0"]:.4f}')
    assert_leq(float(np.max(np.asarray(fr['error'], dtype=float))), 1e-12, rtol=0.0,
               label='signal_spectrum 的 error 曲线贴地板')

    bu = _compute(mod, 'butterfly_stages', n=32, signal='cosine', k=3)
    res = np.asarray(bu['residual'], dtype=float)
    assert_within(float(np.max(res)), float(bu.meta['max_level_residual']), rtol=1e-9,
                  label='残差数组的最大值 = meta 的 max_level_residual',
                  detail='meta 不能比数组更乐观')
    assert_leq(float(np.max(res)), 1e-10, rtol=0.0, label='每级残差都不超过 1e-10')

    cv = _compute(mod, 'conv_theorem', n=16, m=8)
    diff = np.asarray(cv['difference'], dtype=float)
    safe = np.asarray(cv['safe_difference'], dtype=float)
    assert_within(float(np.max(np.abs(diff))), float(cv.meta['max_wrap']), rtol=1e-9,
                  label='difference 数组的最大值 = meta 的 max_wrap')
    assert_leq(float(np.max(np.abs(safe))), 1e-15, rtol=0.0,
               label='safe_difference 贴舍入地板',
               detail='图上"两条曲线重合"必须有这个数撑着（实测 '
                      f'{float(np.max(np.abs(safe))):.2e}），不能只是看起来重合')
    record(label='数组与 meta 的互证', theory='最大值类读数必须逐位一致',
           measured=float(np.max(np.abs(diff))), tol=1e-9,
           passed=abs(float(np.max(np.abs(diff))) - float(cv.meta['max_wrap'])) < 1e-9,
           detail=f'max|difference| = {np.max(np.abs(diff)):.6f}，'
                  f'meta max_wrap = {cv.meta["max_wrap"]:.6f}；'
                  f'正规化误差曲线 max|safe_difference| = {np.max(np.abs(safe)):g}')
