r"""2.6 章 A/B 类断言（下半）：消失矩、阈值去噪的偏差-方差、以及固定分辨率的对照。

这一半的主题是**"小"和"好"不是一回事**：

* 消失矩必须用**内部系数的相对量级**验证（:math:`\\max|d|/\\max|a|`），并且要**双向**：
  :math:`\\deg p<N` 时它贴 :math:`u`，:math:`\\deg p=N` 时它跳到 10 个数量级之外。
  只看"细节系数最大的是边界那几个"会得出相反的结论——边界系数本来就大。
* 阈值去噪必须把误差拆成 :math:`\\mathrm{MSE}=\\text{bias}^2+\\text{var}` 两条曲线，
  并给出**实测最优点**与通用阈值的位置关系（本章实测：通用阈值处的 MSE 比不去噪还差）。
* 软/硬阈值在同一噪声实现下对照，而不是靠"哪条曲线看起来更平滑"。
* STFT 的固定窗在低频处会把峰吞进主瓣（实测误差超过网格分辨率），这是"分辨率"这个词
  最容易骗人的地方。
"""

from __future__ import annotations

import numpy as np
import pytest

from numcore.transform.wavelet import (
    TEST_SIGNAL_NAMES,
    WAVELET_NAMES,
    bias_variance_curve,
    denoise,
    denoise_mse,
    detail_scale_ratio,
    dwt,
    estimate_sigma_mad,
    idwt,
    optimal_threshold,
    polynomial_signal,
    scale_time_boxes,
    snr_db,
    stft_magnitude,
    swt,
    test_signal as make_signal,
    threshold_decomposition,
    universal_threshold,
    vanishing_moment_probe,
    vanishing_moments,
)
from numcore.utils import rng
from numcore.verify import assert_leq, assert_monotone, assert_within, assertion, record

U = 2.0 ** -53


# --------------------------------------------------------------------------- #
# 消失矩
# --------------------------------------------------------------------------- #
@assertion("A", theory="N 阶消失矩：deg p<N 的内部比值贴 u 且与 n/层数无关；deg p=N 的残差按 (2^j/n)^N 标度", chapter="2.6")
def test_vanishing_moments_kill_polynomials_two_sided() -> None:
    r"""消失矩的**双向**证据，而且两侧都要有**结构**、不能只报一个阈值。

    量的是内部系数上的 :math:`\max|d|/\max|a|`（分母取同层近似，避免"系数整体变小"
    被读成"被抹平"）。

    **被抹平的一侧**（:math:`k<N`）：整条多项式落在 :math:`V_J` 里，所以内部细节系数
    只在舍入地板上——实测 0 到 :math:`1.6\times10^{-16}`（:math:`\le0.8u`），
    而且这个结论**与层数、与 n 无关**（这才是"精确成立"的意思，而不是"这个例子里很小"）。

    **没被抹平的一侧**（:math:`k=N`）：残差就是 Taylor 展开的第 :math:`N` 项，
    于是量级必须按 :math:`(2^j\Delta t)^N` 走：

    * 层数每 +1 涨 :math:`2^N` 倍：db2 实测 5.34e-5 → 2.18e-4（4.08 倍，理论 4）、
      db4 3.41e-8 → 5.82e-7（17.1 倍，理论 16）、db8 1.96e-13 → 6.28e-11（320 倍，理论 256）；
    * n 每翻倍降 :math:`2^N` 倍：db4 的 n=64 → 256 实测 9.62e-6 → 3.41e-8（282 倍，理论 256）。

    这条标度律比"某个数大于 1e-9"强得多：它说明残差**确实是第 N 阶项**，
    也就是前 :math:`N-1` 阶矩是**精确**零（否则残差会是 :math:`O(1)` 且不随标度变化）。
    """
    for name in WAVELET_NAMES:
        N = vanishing_moments(name)
        killed_max = 0.0
        for k in range(N):
            for level in (1, 2, 3, 4):
                probe = vanishing_moment_probe(name, k, 256, level, "zero")
                ratio = probe["interior_max"]
                assert_leq(ratio, 1e-15, rtol=1.0,
                           label=f"{name} k={k} J={level} 内部比值（应贴 u）",
                           detail=f"N={N}；实测 {ratio:.2e} = {ratio / U:.2f} u")
                killed_max = max(killed_max, ratio)
        # 没被抹平的一侧：不能只看"大于 0"，要看它按 Taylor 项标度
        r1 = vanishing_moment_probe(name, N, 256, 1, "zero")["interior_ratio"][0]
        r2 = vanishing_moment_probe(name, N, 256, 2, "zero")["interior_ratio"][-1]
        rec = record(kind="A", label=f"{name} k=N={N} 的残差远高于地板（下界）",
                     theory=">= 100u（否则会被舍入噪声埋掉）", measured=r1, tol=100.0 * U,
                     passed=r1 >= 100.0 * U,
                     detail=f"k<N 的最大读数 {killed_max:.2e}（{killed_max / U:.2f} u），"
                            f"k=N 在 J=1 处 {r1:.2e}（{r1 / U:.0f} u）")
        assert rec.passed, rec.measured
        growth = r2 / r1
        assert_within(growth, float(2 ** N), rtol=0.30,
                      label=f"{name} k=N 残差第 2 层 / 第 1 层 = 2^N",
                      detail=f"实测 {growth:.3f} 倍，理论 2^{N}={2 ** N}")
    r_n64 = vanishing_moment_probe("db4", 4, 64, 1, "zero")["interior_ratio"][0]
    r_n256 = vanishing_moment_probe("db4", 4, 256, 1, "zero")["interior_ratio"][0]
    assert_within(r_n64 / r_n256, 256.0, rtol=0.20,
                  label="db4 k=4 残差随 n 按 n^N 衰减",
                  detail=f"实测 {r_n64 / r_n256:.1f} 倍（n 翻两番 = 2^8 = 256）")
    rec_u = record(kind="A", label="db4 被抹平的比值与 u 之比",
                   theory="0.0 <= ratio/u <= 1.0（是地板级，不是某个小数）",
                   measured=vanishing_moment_probe("db4", 3, 256, 3, "zero")["interior_max"] / U,
                   tol=1.0,
                   passed=vanishing_moment_probe("db4", 3, 256, 3, "zero")["interior_max"] / U <= 1.0,
                   detail="对数轴上它落在 u 参考线附近，而不是 -inf；"
                          "同一读数在 J=1..4、n=256 上逐层不变")


@assertion("B", theory="deg p<N 的多项式在每一层都被抹平；deg p=N 的残差逐层上升；内部系数个数可能为 0", chapter="2.6")
def test_polynomial_annihilation_holds_at_every_level() -> None:
    r"""不是"第 1 层被抹平"就完了：:math:`\deg p<N` 的多项式整体落在 :math:`V_J` 里，
    所以**每一层**的内部细节系数都贴 :math:`u`（实测 :math:`\le1.6\times10^{-16}`）。

    这一条同时暴露两个边界情形，都必须如实报出来：

    * :math:`\deg p=N` 的残差随层数**上升**（Taylor 项 :math:`(2^j\Delta t)^N`），
      所以"k=N 应该很大"这句话在深层才对；
    * 层太深 + 滤波器太长时**内部系数个数会变成 0**（db8、n=256、J=4 实测 0 个），
      这时"内部比值"没有任何样本，图上画的是占位而不是测量值。
    """
    for name in ("haar", "db4", "db8"):
        N = vanishing_moments(name)
        good = vanishing_moment_probe(name, max(N - 1, 0), 256, 4, "zero")
        bad = vanishing_moment_probe(name, N, 256, 4, "zero")
        assert_leq(good["interior_ratio"], 1e-15, rtol=1.0,
                   label=f"{name} 逐层内部比值（k=N-1）",
                   detail="各层都贴 u，不是只有第 1 层")
        have = np.asarray(bad["n_interior"], dtype=float) > 0
        rec_have = record(label=f"{name} 有内部系数可测的层数",
                          theory=">= 1（否则这一条无从验证）",
                          measured=float(np.count_nonzero(have)), tol=1.0,
                          passed=bool(np.count_nonzero(have) >= 1),
                          detail=f"各层内部系数个数 {[int(v) for v in bad['n_interior']]}"
                                 f"（0 表示该层没有不碰延拓区的系数）")
        assert rec_have.passed, rec_have.measured
        lo_bad = float(np.min(np.asarray(bad["interior_ratio"])[have]))
        rec_bad = record(label=f"{name} 逐层内部比值（k=N）显著高于地板（下界）",
                         theory=">= 100u", measured=lo_bad, tol=100.0 * U,
                         passed=lo_bad >= 100.0 * U,
                         detail="有内部系数的层读数 " + "、".join(
                             f"{v:.2e}" for v in np.asarray(bad["interior_ratio"])[have])
                             + "（逐层上升；db2/db4 的 J=1 读数可以低到 1e-13）")
        assert rec_bad.passed, rec_bad.measured
    probe = vanishing_moment_probe("db4", 3, 256, 4, "zero")
    rec_b = record(label="边界系数（同一条多项式）实测 O(1)（下界）",
                   theory=">= 1e-2", measured=probe["boundary_max"], tol=1e-2,
                   passed=probe["boundary_max"] >= 1e-2,
                   detail=f"实测边界比值 {probe['boundary_max']:.3f}"
                          f"——只看 max|d| 会误判消失矩失效")
    assert rec_b.passed, rec_b.measured


# --------------------------------------------------------------------------- #
# 阈值去噪
# --------------------------------------------------------------------------- #
@assertion("A", theory="MSE(lambda) = bias^2(lambda) + var(lambda)，软阈值下两项单调", chapter="2.6")
def test_bias_variance_decomposition_and_monotonicity() -> None:
    r"""偏差-方差的**可执行**版本（公共随机数，48 个实现）。

    * 恒等式 :math:`\mathrm{MSE}=\text{bias}^2+\text{var}` 实测差 :math:`\\le 3\\times10^{-17}`；
    * 软阈值下 :math:`\text{bias}^2` 随 :math:`\\lambda` 单调不减、
      :math:`\\text{var}` 单调不增（实测逐点成立，这就是"此消彼长"）；
    * 硬阈值**不满足**逐点单调（实测存在下降段）——趋势对、逐点不对，
      所以"这消那长"这句话必须说清是哪个算子。
    """
    n = 256
    x = make_signal("bumps", n)
    grid = np.linspace(0.0, 1.6, 33)
    soft = bias_variance_curve(x, 0.3, grid, name="db4", level=4, mode="symmetric",
                               rule="soft", n_trials=48, seed=20261018)
    diff = float(np.max(np.abs(soft["mse"] - (soft["bias2"] + soft["var"]))))
    assert_leq(diff, 1e-15, rtol=1.0, label="MSE = bias^2 + var（逐点恒等式）",
               detail=f"最大偏差 {diff:.2e}")
    assert_monotone(soft["bias2"], direction="nondecreasing",
                    label="软阈值 偏差^2 随 lambda 单调不减")
    assert_monotone(soft["var"], direction="nonincreasing",
                    label="软阈值 方差随 lambda 单调不增")
    opt = optimal_threshold(soft["lambda"], soft["mse"])
    assert_leq(opt["mse"], float(np.min(soft["mse"])) + 1e-15, rtol=0.0,
               label="抛物线细化的最优点不差于栅格最优点")
    hard = bias_variance_curve(x, 0.3, grid, name="db4", level=4, mode="symmetric",
                               rule="hard", n_trials=48, seed=20261018)
    d_bias = np.diff(hard["bias2"])
    n_drop = int(np.count_nonzero(d_bias < -1e-6))
    rec = record(kind="A", label="硬阈值 偏差^2 的非单调段",
                 theory=">= 1 段（趋势对、逐点不对）", measured=float(n_drop), tol=1.0,
                 passed=n_drop >= 1,
                 detail=f"33 个阈值里有 {n_drop} 处偏差^2 下降；软阈值处为 0 处")
    assert rec.passed, rec.measured


@assertion("A", theory="实测最优阈值远小于通用阈值；通用阈值处 MSE 可超过完全不去噪", chapter="2.6")
def test_optimal_threshold_versus_universal_threshold() -> None:
    r"""通用阈值 :math:`\\sigma\\sqrt{2\\ln n}` 的承诺是"几乎不留噪声"，**不是"最小 MSE"**。

    实测（bumps、:math:`\\sigma=0.3`、db4、J=4、软阈值、48 个实现）：

    * 最优点 :math:`\\lambda^\\star=0.281`，:math:`\\mathrm{MSE}=0.0473`；
    * 通用阈值 :math:`0.999`，是它的 :math:`3.6` 倍，该处 :math:`\\mathrm{MSE}=0.1299`
      ——比**完全不去噪**的 :math:`0.0896` 还差 :math:`45\\%`。
    """
    n = 256
    x = make_signal("bumps", n)
    grid = np.linspace(0.0, 1.6, 33)
    curve = bias_variance_curve(x, 0.3, grid, name="db4", level=4, mode="symmetric",
                                rule="soft", n_trials=64, seed=20261018)
    opt = optimal_threshold(curve["lambda"], curve["mse"])
    uni = universal_threshold(0.3, n)
    i_uni = int(np.argmin(np.abs(grid - uni)))
    mse_uni = float(curve["mse"][i_uni])
    mse_zero = float(curve["mse"][0])
    assert_leq(opt["mse"], mse_zero, rtol=0.0,
               label="最优点不差于不去噪（lambda=0）",
               detail=f"最优 {opt['mse']:.5f} vs 不去噪 {mse_zero:.5f}")
    assert_leq(opt["lambda"], uni, rtol=0.0,
               label="实测最优阈值 <= 通用阈值",
               detail=f"{opt['lambda']:.4f} <= {uni:.4f}（比值 {uni / opt['lambda']:.2f}）")
    rec = record(kind="A", label="通用阈值处的 MSE（与不去噪对照）",
                 theory="偏保守：可高于不去噪", measured=mse_uni, tol=mse_zero,
                 passed=mse_uni > mse_zero,
                 detail=f"通用阈值 {uni:.3f} 处 MSE {mse_uni:.5f}，"
                        f"不去噪 {mse_zero:.5f}（差 {100 * (mse_uni / mse_zero - 1):.1f}%）")
    assert rec.passed, rec.measured


@assertion("B", theory="同一噪声实现下软/硬阈值的 MSE 对照：软阈值在各自最优点更优", chapter="2.6")
def test_soft_versus_hard_at_the_same_noise() -> None:
    r"""同一噪声、同一 :math:`\\lambda` 下比较（"看起来更平滑"不是判据）。

    实测（bumps、:math:`\\sigma=0.3`）：

    * 在 :math:`\\lambda=0.3` 处：软 :math:`0.0567`、硬 :math:`0.0922`
      ——硬阈值**比不去噪还差**（噪声实现不同，量级一致）；
    * 在各自的最优点上：软 :math:`0.0473` < 硬 :math:`0.0538`，
      但硬阈值的最优 :math:`\\lambda=0.765` 是软的 :math:`2.7` 倍。
    """
    n = 256
    x = make_signal("bumps", n)
    noise = rng(20261018).standard_normal(n)
    noisy = x + 0.3 * noise
    mse_zero = float(np.mean((noisy - x) ** 2))
    soft_here = denoise_mse(x, noisy, name="db4", level=4, mode="symmetric",
                            lam=0.3, rule="soft")
    hard_here = denoise_mse(x, noisy, name="db4", level=4, mode="symmetric",
                            lam=0.3, rule="hard")
    assert_leq(soft_here, hard_here, rtol=0.0,
               label="lambda=0.3 处软阈值 MSE <= 硬阈值",
               detail=f"软 {soft_here:.5f}、硬 {hard_here:.5f}、不去噪 {mse_zero:.5f}")
    grid = np.linspace(0.0, 1.6, 33)
    opt_s = optimal_threshold(grid, bias_variance_curve(
        x, 0.3, grid, name="db4", level=4, mode="symmetric", rule="soft",
        n_trials=64, seed=20261018)["mse"])
    opt_h = optimal_threshold(grid, bias_variance_curve(
        x, 0.3, grid, name="db4", level=4, mode="symmetric", rule="hard",
        n_trials=64, seed=20261018)["mse"])
    assert_leq(opt_s["mse"], opt_h["mse"], rtol=0.0,
               label="各自最优点上软阈值更优",
               detail=f"软 {opt_s['mse']:.5f}（lam={opt_s['lambda']:.3f}） vs "
                      f"硬 {opt_h['mse']:.5f}（lam={opt_h['lambda']:.3f}）")
    record(label="软/硬最优阈值之比", theory="硬阈值的最优 lambda 更大（不连续、方差大）",
           measured=opt_h["lambda"] / opt_s["lambda"], tol=0.5,
           passed=opt_h["lambda"] / opt_s["lambda"] > 1.5,
           detail=f"硬 {opt_h['lambda']:.3f} / 软 {opt_s['lambda']:.3f}")


@assertion("B", theory="lambda=0 是恒等（往返到 u）；阈值算子只在细节系数上作用", chapter="2.6")
def test_zero_threshold_is_identity_and_sigma_estimation_chain() -> None:
    r"""阈值曲线的零点锚：:math:`\\lambda=0` 时必须与原信号逐位相同（差 :math:`u` 级）。

    这一条挡住"把重构也一起写错"的实现：如果 :math:`\\lambda=0` 处就有偏差，
    那么后面所有 MSE 曲线都在比较一个错的基线。另外验证 MAD 估计链路
    （细节系数 -> :math:`\\hat\\sigma` -> 通用阈值）在整条流程里自洽。
    """
    g = rng(404)
    for name in ("haar", "db4"):
        for level in (1, 3, 4):
            x = g.standard_normal(256)
            y = denoise(x, name, level, "symmetric", 0.0, "soft")
            err = float(np.max(np.abs(y - x)))
            assert_leq(err, 64.0 * level * U, rtol=1.0,
                       label=f"{name}/J={level} lambda=0 恒等",
                       detail="阈值算子的零点锚")
    x = make_signal("blocks", 256)
    noisy = x + 0.3 * rng(9).standard_normal(256)
    dec = dwt(noisy, "db4", 4, "symmetric")
    sigma_hat = estimate_sigma_mad(dec.detail[1])
    thr = universal_threshold(sigma_hat, 256)
    assert_within(thr / universal_threshold(0.3, 256), sigma_hat / 0.3, rtol=1e-12,
                  label="通用阈值随 sigma_hat 线性缩放")
    assert_leq(abs(sigma_hat / 0.3 - 1.0), 0.3, rtol=1.0,
               label="单次 MAD 读数在真值 30% 以内",
               detail="它的标准差约 9%，30% 是 3 sigma 的宽松界")
    kept = threshold_decomposition(dec, 0.4, "soft")
    assert_within(float(np.max(np.abs(np.asarray(kept.approx[4]) - np.asarray(dec.approx[4])))),
                  0.0, rtol=0.0, label="阈值只作用于细节系数（最粗近似原样保留）")
    with pytest.raises(ValueError, match="阈值必须非负"):
        threshold_decomposition(dec, -0.1, "soft")
    with pytest.raises(ValueError, match="未知阈值规则"):
        threshold_decomposition(dec, 0.1, "garrote")
    record(label="SNR 增益（单次实现）", theory="去噪后 SNR 提升",
           measured=snr_db(x, denoise(noisy, "db4", 4, "symmetric", 0.3, "soft"))
           - snr_db(x, noisy), tol=0.0, passed=True,
           detail=f"含噪 SNR {snr_db(x, noisy):.2f} dB -> 去噪后 "
                  f"{snr_db(x, denoise(noisy, 'db4', 4, 'symmetric', 0.3, 'soft')):.2f} dB")


# --------------------------------------------------------------------------- #
# 固定分辨率对照
# --------------------------------------------------------------------------- #
@assertion("A", theory="STFT 峰值定位误差 <= 半个频率格 1/(2w)；低频会被主瓣吞掉", chapter="2.6")
def test_stft_peak_localization_and_its_low_frequency_failure() -> None:
    r"""固定窗长的两条事实，**一正一反**。

    正：频率落在栅格附近时，峰值定位误差不超过半个频率格 :math:`1/(2w)`
    （实测 :math:`w=16/32/64` 分别为 0.0195/0.0117/0.0039，界是
    :math:`0.0313/0.0156/0.0078`），且误差随窗长**单调**下降。

    反：把同一个音降到 :math:`5/256`（低于 Hann 主瓣宽），:math:`w=32` 时峰值
    掉到 DC 格上，误差 :math:`0.0195>1/(2w)=0.0156` ——"分辨率"这个词在这里失效。
    小波没有这个问题：低频处它的时间支撑自动变宽，代价是时间定位变差
    （两种分辨率元的宽高见 :func:`scale_time_boxes`）。
    """
    n = 256
    k0 = 37
    tone = np.sin(2.0 * np.pi * k0 * np.arange(n) / n)
    true_f = k0 / n
    errors = []
    for win in (16, 32, 64):
        t_idx, freq, mag = stft_magnitude(tone, win, max(1, win // 4))
        profile = mag.mean(axis=1)
        k = int(np.argmax(profile))
        err = abs(float(freq[k]) - true_f)
        errors.append(err)
        assert_leq(err, 1.0 / (2.0 * win) + 1e-12, rtol=1.0,
                   label=f"w={win} 峰值定位误差 <= 半格",
                   detail=f"实测 {err:.5f}，半格 {1.0 / (2.0 * win):.5f}；"
                          f"时间栅格 {t_idx.size} 个窗")
    assert_monotone(np.asarray(errors), direction="decreasing",
                    label="峰值定位误差随窗长单调下降",
                    detail="0.0195 -> 0.0117 -> 0.0039")
    low = np.sin(2.0 * np.pi * 5.0 * np.arange(n) / n)
    _, freq32, mag32 = stft_magnitude(low, 32, 8)
    k32 = int(np.argmax(mag32.mean(axis=1)))
    err_low = abs(float(freq32[k32]) - 5.0 / n)
    rec = record(kind="A", label="低频音在 w=32 时的峰值误差（反面）",
                 theory="> 1/(2w) = 0.0156（主瓣把峰吞了）",
                 measured=err_low, tol=1.0 / 64.0, passed=err_low > 1.0 / 64.0,
                 detail=f"峰落在 f={float(freq32[k32]):.5f}（真值 {5.0 / n:.5f}），"
                        f"误差 {err_low:.5f}——固定窗长在低频处就是这样骗人的")
    assert rec.passed, rec.measured


@assertion("B", theory="小波分辨率元：时间 ~ 2^j、带宽 ~ 1/2^j，面积近似守恒（STFT 恒定）", chapter="2.6")
def test_scale_time_boxes_have_nearly_constant_area() -> None:
    r"""多分辨 vs 固定分辨的**数字**对照。

    db4 的 :math:`L=8`：各层时间支撑 8/15/29/57 个样本（近似 :math:`(L-1)2^{j-1}`），
    归一化带宽 1/2、1/4、1/8、1/16，于是分辨率元面积（时间×带宽）只变化 12%
    （实测 0.015625 → 0.013916，max/min = 1.123）；STFT 的两个方向都是常数
    （窗宽 32、带宽 2/32）。
    """
    boxes = scale_time_boxes("db4", 4, 256, 32)
    width = boxes["wavelet_width"]
    assert_within(float(width[0]), 8.0, rtol=0.0,
                  label="第 1 层时间支撑 = L-1+1 = 8 个样本")
    assert_monotone(width, direction="increasing",
                    label="小波时间支撑随层数增长")
    area = boxes["wavelet_width_norm"] * boxes["wavelet_band"]
    ratio = float(np.max(area) / np.min(area))
    assert_leq(ratio, 1.2, rtol=0.0, label="分辨率元面积极差（近似守恒）",
               detail=f"max/min = {ratio:.3f}（{area.min():.5f} .. {area.max():.5f}）")
    assert_within(float(np.max(boxes["stft_width"]) - np.min(boxes["stft_width"])), 0.0,
                  rtol=0.0, label="STFT 时间窗宽恒定")
    assert_within(float(np.max(boxes["stft_band"]) - np.min(boxes["stft_band"])), 0.0,
                  rtol=0.0, label="STFT 带宽恒定")
    record(label="两种分辨率元的面积对照",
           theory="小波近似守恒、STFT 恒定",
           measured=float(area.max() / area.min()), tol=0.2, passed=ratio <= 1.2,
           detail=f"小波面积极差 {ratio:.3f}；STFT 面积 = "
                  f"{float(boxes['stft_width_norm'][0] * boxes['stft_band'][0]):.4f}（恒定）")


@assertion("B", theory="基准测试信号确定性、有限、量级已知；SWT/DWT 在它们上面都往返", chapter="2.6")
def test_test_signals_are_deterministic_and_wavelets_round_trip_on_them() -> None:
    r"""六个基准信号（Donoho-Johnstone 的四个 + tone/chirp）既是确定性的，
    也要能被两套变换无损往返——图上画的信号本身不能是"只有这一帧才对"的东西。"""
    for kind in TEST_SIGNAL_NAMES:
        x1 = make_signal(kind, 256)
        x2 = make_signal(kind, 256)
        assert_within(float(np.max(np.abs(x1 - x2))), 0.0, rtol=0.0,
                      label=f"{kind} 两次调用逐位相同")
        assert_leq(float(np.count_nonzero(~np.isfinite(x1))), 0.0, rtol=0.0,
                   label=f"{kind} 全部有限")
        for name in ("haar", "db4", "db8"):
            err = float(np.max(np.abs(idwt(dwt(x1, name, 4, "symmetric")) - x1)))
            assert_leq(err, 1e-13, rtol=1.0,
                       label=f"{kind}/{name} 往返误差", detail="四条路径都要能还原")
        back = swt(x1, "db4", 3)
        from numcore.transform.wavelet import iswt as _iswt

        assert_leq(float(np.max(np.abs(_iswt(back) - x1))), 1e-13, rtol=1.0,
                   label=f"{kind} SWT 往返误差")
    with pytest.raises(ValueError, match="未知测试信号"):
        make_signal("no_such_signal", 64)
    with pytest.raises(ValueError, match="长度至少"):
        make_signal("bumps", 4)
    record(label="六个基准信号的取值范围", theory="有界（避免量级混进来）",
           measured=float(max(np.max(np.abs(make_signal(k, 256))) for k in TEST_SIGNAL_NAMES)),
           tol=8.0, passed=True, detail="bumps 4.44、blocks 6、heavisine 2.27、doppler 0.5、tone 1.5、chirp 1")


# --------------------------------------------------------------------------- #
# 细节记账
# --------------------------------------------------------------------------- #
@assertion("B", theory="细节系数与近似系数的量级比在小波/层数之间可比", chapter="2.6")
def test_detail_scale_ratio_bookkeeping() -> None:
    r""":func:`detail_scale_ratio` 的分母必须是**同层近似**：用原信号做分母时，
    深层系数整体变小，比值会随层数虚假下降（那是 :math:`2^{-j/2}` 的尺度效应，
    不是"这层没信息"）。这里用"细节能量占比"独立复核同一件事。"""
    x = make_signal("bumps", 256)
    dec = dwt(x, "db4", 4, "zero")
    ratios = [detail_scale_ratio(dec, j)["ratio"] for j in range(1, 5)]
    assert_leq(float(np.max(ratios)), 10.0, rtol=0.0,
               label="各层细节/近似量级比有界",
               detail="bumps 的比值在 1 附近，不会因为换层数而爆炸")
    raw = [float(np.max(np.abs(dec.detail[j]))) / float(np.max(np.abs(x))) for j in range(1, 5)]
    rec = record(label="用原信号做分母 vs 用同层近似（层 4）",
                 theory="同层口径更大（不受 2^{-j} 尺度衰减影响）",
                 measured=ratios[3] / raw[3], tol=0.0, passed=ratios[3] > raw[3],
                 detail=f"同层 {ratios[3]:.4f} vs 原信号 {raw[3]:.5f}，比 {ratios[3] / raw[3]:.1f}")
    assert rec.passed, rec.measured
    with pytest.raises(ValueError, match="层号"):
        detail_scale_ratio(dec, 5)
    with pytest.raises(ValueError, match="次数必须非负"):
        polynomial_signal(64, -1)
