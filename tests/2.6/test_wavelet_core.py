r"""2.6 章 A/B 类断言（上半）：滤波器、DWT/SWT 的**正确性**与边界延拓。

主线是「完美重构」这四个字必须被拆成三件可测的事：

* **滤波器本身对不对**——正交性 :math:`\sum_j h_j h_{j-2k}=\delta_k`、归一化
  :math:`\sum_j h_j=\sqrt2`、完美重构条件 :math:`|H(\omega)|^2+|H(\omega+\pi)|^2=2`、
  以及 :math:`N` 阶消失矩 :math:`\sum_j(-1)^j j^m h_j=0\ (m<N)`。
  四条都是**定义式**，所以它们同时也是"系数表抄错了没有"的机器证据。
* **变换是不是同一个东西**——递归算法 vs **显式矩阵**（第二条独立路径）、
  Haar 的闭式解、多分辨分解 :math:`x=x_J+\sum_j w_j` 的逐位恒等式。
* **边界延拓到底做了什么**——三种模式的能量比、正交性、往返误差，
  以及"一点余量都不留"时的 :math:`O(1)` 灾难（负对照）。

这两条路径都必须同时在场：只测"往返误差小"的话，一个自洽但错误的实现
（比如把 :math:`h` 与 :math:`g` 一起写成另一组正交滤波器）会全绿。
"""

from __future__ import annotations

import numpy as np

from numcore.transform.wavelet import (
    BOUNDARY_MODES,
    WAVELET_NAMES,
    boundary_pollution,
    coefficient_lengths,
    dwt,
    dwt_matrix,
    dwt_step,
    energy_ratio,
    estimate_sigma_mad,
    filter_length,
    filter_response,
    hard_threshold,
    idwt,
    iswt,
    level_energies,
    mra_bands,
    mra_partial,
    orthogonality_defect,
    orthonormality_defect,
    perfect_reconstruction_defect,
    reconstruction_error,
    scaling_filter,
    soft_threshold,
    swt,
    swt_frame_symbol,
    test_signal as make_signal,
    translation_change,
    truncate_critical,
    vanishing_moment_defect,
    vanishing_moments,
    wavelet_filter,
)
from numcore.utils import rng
from numcore.verify import assert_leq, assert_monotone, assert_within, assertion, record

U = 2.0 ** -53

#: 覆盖三种滤波器长度：最短（Haar）、中等（db4）、最长（db8，L=16）
WAVELETS = ("haar", "db2", "db4", "db8")

#: 覆盖 2 的幂与"能整除但更大"的层数
LEVELS = (1, 2, 3, 4, 6, 8)


def _signal(n: int, seed: int = 0) -> np.ndarray:
    """确定性测试信号：低频 + 高频 + 一点噪声，避开"整周期"这种太顺的例子。"""
    g = rng(2600 + seed)
    k = np.arange(n, dtype=float)
    base = np.sin(2.0 * np.pi * 5.0 * k / n) + 0.5 * np.cos(2.0 * np.pi * 17.0 * k / n)
    return base + 0.1 * g.standard_normal(n)


# --------------------------------------------------------------------------- #
# A 类：滤波器与变换的定义式
# --------------------------------------------------------------------------- #
@assertion("A", theory="dbN 的 h 满足正交性、归一化、完美重构条件与 N 阶消失矩", chapter="2.6")
def test_daubechies_filters_satisfy_defining_properties() -> None:
    r"""滤波器表的正确性由**定义式**判定，而不是由"抄得像"保证。

    四个量逐族测：正交性缺陷 :math:`\max_k|\sum_j h_jh_{j-2k}-\delta_k|`、
    :math:`|\sum h_j-\sqrt2|/\sqrt2`、:math:`\max_\omega||H(\omega)|^2+|H(\omega+\pi)|^2-2|`、
    以及**相对**消失矩缺陷 :math:`|\sum_j(-1)^jj^mh_j|/\sum_jj^m|h_j|`。

    最后一列的"双向"就在这里：:math:`m=N` 时它**不再**贴 :math:`u`
    （db8 的 :math:`m=8` 实测 :math:`1.8\times10^{-3}`，比 :math:`u` 高 13 个数量级）
    ——没有这一半，"消失矩成立"就没有上界，等于什么都没说。
    """
    worst_ortho = 0.0
    worst_norm = 0.0
    worst_pr = 0.0
    worst_vm = 0.0
    vm_at_N = {}
    for name in WAVELET_NAMES:
        h = scaling_filter(name)
        ortho = orthonormality_defect(h)
        norm = abs(float(np.sum(h)) - np.sqrt(2.0)) / np.sqrt(2.0)
        pr = perfect_reconstruction_defect(h)
        N = vanishing_moments(name)
        vm = max(vanishing_moment_defect(h, m) for m in range(N))
        vm_n = vanishing_moment_defect(h, N)
        vm_at_N[name] = vm_n
        assert_leq(ortho, 8.0 * U, rtol=1.0, label=f"{name} 正交性缺陷",
                   detail="sum_j h_j h_{j-2k} = delta_k（定义式）")
        assert_leq(norm, 8.0 * U, rtol=1.0, label=f"{name} 归一化 |sum h - sqrt2|/sqrt2")
        assert_leq(pr, 32.0 * U, rtol=1.0, label=f"{name} 完美重构条件",
                   detail="|H(w)|^2 + |H(w+pi)|^2 = 2，512 点扫描")
        assert_leq(vm, 1e-14, rtol=1.0, label=f"{name} 前 {N} 阶消失矩（相对缺陷）",
                   detail="m < N 时 sum_j (-1)^j j^m h_j = 0")
        worst_ortho = max(worst_ortho, float(ortho))
        worst_norm = max(worst_norm, float(norm))
        worst_pr = max(worst_pr, float(pr))
        worst_vm = max(worst_vm, float(vm))
    rec = record(kind="A", label="最坏正交性缺陷", theory="<= 8u",
                 measured=worst_ortho, tol=8.0 * U, passed=worst_ortho <= 8.0 * U,
                 detail=f"归一化 {worst_norm:.1e}、重构条件 {worst_pr:.1e}、消失矩 {worst_vm:.1e}")
    assert rec.passed, rec.measured
    # 双向：m = N 时消失矩**必须**失效，否则"N 阶"这句话是空的
    vm_floor = min(vm_at_N.values())
    rec2 = record(kind="A", label="m = N 处消失矩不再成立（下界）",
                  theory=">= 1e-6（远高于 u=2.2e-16）",
                  measured=vm_floor, tol=1e-6, passed=vm_floor >= 1e-6,
                  detail="各族 m=N 的相对缺陷：" + "、".join(
                      f"{k}={v:.1e}" for k, v in vm_at_N.items()))
    assert rec2.passed, rec2.measured


@assertion("A", theory="周期 DWT 是正交变换：M M^T = M^T M = I；另两种模式是冗余的等距嵌入", chapter="2.6")
def test_periodic_dwt_matrix_is_orthogonal_but_symmetric_is_not() -> None:
    r"""把分析算子写成显式矩阵 :math:`M`（第 :math:`i` 列 = :math:`e_i` 的系数）。

    三种模式的**代数身份**完全不同，这是本章"能量守恒要写清是哪个能量"的根据：

    * ``periodic``：:math:`M` 是 :math:`n\times n` 正交阵，两个方向都贴 :math:`u`；
    * ``zero``：:math:`M^{\mathsf T}M=I`（等距），但 :math:`MM^{\mathsf T}\neq I`
      ——多出来的系数是零延拓撑出来的冗余；
    * ``symmetric``：反射会复制样本，:math:`M^{\mathsf T}M\neq I`，
      缺陷实测 :math:`O(1)`。
    """
    n = 64
    for wave in ("db2", "db4", "db8"):
        for level in (1, 2, 4):
            M = dwt_matrix(n, wave, "periodic", level)
            assert_within(float(M.shape[0]), float(n), rtol=0.0,
                          label=f"{wave} J={level} 周期模式系数个数 = n")
            assert_leq(orthogonality_defect(M), 64.0 * U, rtol=1.0,
                       label=f"{wave} J={level} 正交性缺陷",
                       detail="M M^T 与 M^T M 两个方向都查")
    Mz = dwt_matrix(n, "db4", "zero", 4)
    assert_leq(float(np.max(np.abs(Mz.T @ Mz - np.eye(n)))), 64.0 * U, rtol=1.0,
               label="zero 模式 M^T M = I（等距嵌入）")
    left_zero = float(np.max(np.abs(Mz @ Mz.T - np.eye(Mz.shape[0]))))
    rec = record(kind="A", label="zero 模式 M M^T != I（冗余）",
                 theory=">= 0.5（远大于 u）", measured=left_zero, tol=0.5,
                 passed=left_zero >= 0.5,
                 detail=f"n_ext={Mz.shape[0]}，多出的 {Mz.shape[0] - n} 个系数不是噪声")
    assert rec.passed, rec.measured
    Ms = dwt_matrix(n, "db4", "symmetric", 4)
    right_sym = float(np.max(np.abs(Ms.T @ Ms - np.eye(n))))
    rec2 = record(kind="A", label="symmetric 模式 M^T M != I（反射复制样本）",
                  theory=">= 0.5", measured=right_sym, tol=0.5, passed=right_sym >= 0.5,
                  detail="这就是同一份系数能量比 > 1 的代数原因")
    assert rec2.passed, rec2.measured


@assertion("A", theory="递归 DWT = 显式滤波器组矩阵；Haar 有闭式解 a_k=(x_2k+x_2k+1)/sqrt2", chapter="2.6")
def test_dwt_matches_explicit_filter_bank_and_haar_closed_form() -> None:
    r"""第二条独立路径：**直接**按下标写出的滤波器组矩阵。

    :func:`dwt_step` 的实现是"构造下标矩阵 + 一次矩阵乘"，这里则逐项累加
    :math:`a_k=\sum_j h_jx_{2k+j}`（延拓坐标），再与 :func:`dwt_matrix` 的列逐点比。
    两条路径共享的只有系数表，任何下标/边界错误都会在这里现形。

    Haar 另给闭式解：:math:`a_k=(x_{2k}+x_{2k+1})/\sqrt2`、
    :math:`d_k=(x_{2k}-x_{2k+1})/\sqrt2`（本模块的下标约定贴着 :math:`2k`）。
    """
    g = rng(11)
    for n in (16, 32, 64):
        x = g.standard_normal(n)
        for mode in BOUNDARY_MODES:
            M = dwt_matrix(n, "db4", mode, 1)
            dec = dwt(x, "db4", 1, mode)
            coeff = dec.coefficients()
            ref = M @ x
            scale = max(float(np.max(np.abs(ref))), 1e-300)
            err = float(np.max(np.abs(coeff - ref))) / scale
            assert_leq(err, 64.0 * n * U, rtol=1.0,
                       label=f"n={n} {mode} 递归 vs 显式矩阵", detail="c·n·u，c=64")
    x = g.standard_normal(32)
    dec = dwt(x, "haar", 1, "periodic")
    c = 1.0 / np.sqrt(2.0)
    a_ref = c * (x[0::2] + x[1::2])
    d_ref = c * (x[0::2] - x[1::2])
    ea = float(np.max(np.abs(dec.approx[1] - a_ref)))
    ed = float(np.max(np.abs(dec.detail[1] - d_ref)))
    rec = record(kind="A", label="Haar 闭式解", theory="a_k=(x_2k+x_2k+1)/sqrt2，逐位",
                 measured=max(ea, ed), tol=8.0 * U, passed=max(ea, ed) <= 8.0 * U,
                 detail=f"近似误差 {ea:.2e}、细节误差 {ed:.2e}（n=32）")
    assert rec.passed, rec.measured


@assertion("A", theory="三种延拓模式的往返误差都 ~ u；裁到临界采样后跳到 O(1)", chapter="2.6")
def test_perfect_reconstruction_error_and_the_critical_sampling_control() -> None:
    r"""「完美重构」是**到机器精度**，而且这句话有两半。

    前半：三种模式、六种层数、三个小波，往返误差的最大值都在 :math:`c\\,J\\,u` 之内
    （实测 :math:`1.2\\times10^{-16}` 到 :math:`5.1\\times10^{-16}`）。
    后半（负对照）：把每层系数裁到**临界采样**个数（一点延拓余量都不留）之后，
    误差从 :math:`u` 级跳到 :math:`O(1)`——**这不是精度问题，是换了一个变换**。
    """
    worst = 0.0
    worst_where = ""
    for mode in BOUNDARY_MODES:
        for wave in WAVELETS:
            for level in LEVELS:
                errs = [reconstruction_error(_signal(256, s), wave, level, mode)
                        for s in range(3)]
                e = max(errs)
                tol = 4.0 * level * U
                assert_leq(e, tol, rtol=1.0,
                           label=f"{mode}/{wave}/J={level} 往返误差",
                           detail="容差 4·J·u（每层两次滤波，舍入按层数累积）")
                if e > worst:
                    worst, worst_where = e, f"{mode}/{wave}/J={level}"
    rec = record(kind="A", label="最大往返误差", theory="<= 4·J·u",
                 measured=worst, tol=4.0 * 8 * U, passed=worst <= 4.0 * 8 * U,
                 detail=f"出现在 {worst_where}；n=256、三个信号取最大")
    assert rec.passed, rec.measured
    x = _signal(256, 5)
    dec = dwt(x, "db4", 4, "zero")
    err_bad = float(np.max(np.abs(idwt(truncate_critical(dec)) - x)))
    rec2 = record(kind="A", label="裁到临界采样后的往返误差（负对照）",
                  theory="O(1)，远大于 u", measured=err_bad, tol=0.5,
                  passed=err_bad >= 0.5,
                  detail="裁掉的正是延拓撑出来的系数——边界处理骗人就在这里")
    assert rec2.passed, rec2.measured


@assertion("A", theory="Parseval：periodic/zero 能量比恰为 1，symmetric 等于冗余度 n_ext/n", chapter="2.6")
def test_parseval_energy_by_boundary_mode() -> None:
    r"""能量守恒要写清是**哪个**能量：:math:`\|\text{coeff}\|_2^2/\|x\|_2^2`。

    分母是**原信号**（不是延拓后的信号）。于是：

    * ``periodic``：正交变换，比值恰好 1（实测偏差 :math:`\\le 8\\,n\\,u`）；
    * ``zero``：补零不添能量，比值也是 1；
    * ``symmetric``：反射**复制**样本，系数能量等于延拓信号的能量，
      比值 → 冗余度 :math:`n_{\\text{ext}}/n`（这里 288/256 = 1.125，
      50 个实现的均值实测贴住它，单次实现的散布约 ±1%）。
    """
    g = rng(21)
    n = 256
    for mode in ("periodic", "zero"):
        for wave in WAVELETS:
            for level in (1, 3, 4):
                dec = dwt(g.standard_normal(n), wave, level, mode)
                ratio = energy_ratio(dec)
                assert_leq(abs(ratio - 1.0), 8.0 * n * U, rtol=1.0,
                           label=f"{mode}/{wave}/J={level} 能量比 = 1",
                           detail="正交（或等距）变换的 Parseval 等式")
                shares = level_energies(dec)
                assert_within(float(shares.sum()), ratio, rtol=1e-12,
                              label=f"{mode}/{wave}/J={level} 逐层能量占比求和 = 总能量比")
    dec = dwt(g.standard_normal(n), "db4", 4, "symmetric")
    pred = float(dec.n_ext) / float(dec.n)
    ratios = [energy_ratio(dwt(rng(300 + s).standard_normal(n), "db4", 4, "symmetric"))
              for s in range(50)]
    mean = float(np.mean(ratios))
    rec = record(kind="A", label="symmetric 模式能量比（50 个实现均值）",
                 theory=f"n_ext/n = {pred:.4f}", measured=mean, tol=0.02,
                 passed=abs(mean - pred) <= 0.02,
                 detail=f"单次实测区间 [{min(ratios):.4f}, {max(ratios):.4f}]；"
                        f"同一变换在 periodic 下比值恒为 1")
    assert rec.passed, rec.measured


@assertion("A", theory="不做延拓时边界系数被污染：内部恰好为 0，物理影响区按 2^j 增长", chapter="2.6")
def test_boundary_pollution_is_local_but_grows_with_level() -> None:
    r"""同一条信号，两种口径：**有余量的延拓** vs **一点余量都不留**（纯周期环绕）。

    逐层比较两者的细节系数（对齐到同一物理位置）：

    * 层 :math:`j` 的偏差只出现在两端各几个系数上（实测 3/5/6/6 个），
      所以"污染"不是整层漫开；
    * 但每个系数代表的**物理长度**是 :math:`2^j`，于是影响区按 :math:`2^j` 增长
      （实测 6 → 20 → 48 → 96 个样本）：层数越深，"边界"吃掉的信号越长。
    """
    x = make_signal("tone", 512) + 0.05 * rng(31).standard_normal(512)
    pool = boundary_pollution(x, "db4", 4, "periodic")
    maxes = np.asarray(pool["level_max"][1:], dtype=float)
    counts = np.asarray(pool["n_polluted"][1:], dtype=float)
    spans = np.asarray(pool["physical_span"][1:], dtype=float)
    assert_monotone(spans, direction="increasing",
                    label="边界污染影响区（样本数）随层数增长",
                    detail="实测 6 -> 20 -> 48 -> 96（每层约翻倍）")
    assert_leq(float(np.max(counts)), 8.0, rtol=1.0,
               label="受污染的系数个数（各层）",
               detail="每层都只有边界那几个系数，不是整层")
    rec = record(kind="A", label="偏差幅度随层数增长",
                 theory="各层最大偏差单调增（下层代表的尺度更大）",
                 measured=float(maxes[-1]), tol=float(maxes[0]),
                 passed=bool(np.all(np.diff(maxes) > 0.0)) and maxes[-1] > 5.0 * maxes[0],
                 detail="各层最大偏差 " + "、".join(f"{v:.3f}" for v in maxes))
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# B 类：不变量
# --------------------------------------------------------------------------- #
@assertion("B", theory="MRA 恒等式 x = x_J + sum_j w_j（逐位）与逐层重构的单调贴合", chapter="2.6")
def test_mra_bands_sum_to_the_signal() -> None:
    r"""多分辨分析在代码里的样子：:math:`x=x_J+\sum_{j=1}^{J}w_j`。

    三条不变量：分解出的各带之和逐位等于原信号；逐层累加的最后一行等于原信号；
    只留最粗近似那一行与信号的差别最大（信息确实按层被剥掉）。
    """
    g = rng(41)
    for mode in BOUNDARY_MODES:
        for wave in ("haar", "db4", "db8"):
            x = _signal(256, 3)
            dec = dwt(x, wave, 4, mode)
            bands = mra_bands(dec)
            partial = mra_partial(dec)
            scale = float(np.max(np.abs(x)))
            assert_leq(float(np.max(np.abs(bands.sum(axis=0) - x))) / scale, 256.0 * U,
                       rtol=1.0, label=f"{mode}/{wave} 各带之和 = x",
                       detail="MRA 恒等式，逐位")
            assert_leq(float(np.max(np.abs(partial[-1] - x))) / scale, 256.0 * U,
                       rtol=1.0, label=f"{mode}/{wave} 逐层累加末行 = x")
            d0 = float(np.max(np.abs(partial[0] - x)))
            d1 = float(np.max(np.abs(partial[1] - x)))
            assert_leq(d1, d0 + 1e-12, rtol=0.0,
                       label=f"{mode}/{wave} 第一层细节把误差降下来",
                       detail="只留最粗近似误差最大")


@assertion("B", theory="阈值算子：软阈值 1-Lipschitz、无偏性与连续性不可兼得", chapter="2.6")
def test_threshold_operators_are_shrinkage() -> None:
    r"""软/硬阈值的代数性质（后面"偏差-方差"那一节全靠它们）。

    * 软阈值：:math:`|S_\\lambda(c)-S_\\lambda(c')|\\le|c-c'|`（1-Lipschitz），
      :math:`|c|\\le\\lambda` 时严格为 0，大系数被整体收缩 :math:`\\lambda`；
    * 硬阈值：大系数**原样**保留（无偏），但在 :math:`|c|=\\lambda` 处跳变
      （实测输入差 :math:`10^{-9}`、输出差 :math:`\\lambda`）；
    * 两者都保号、且 :math:`|S_\\lambda(c)|\\le|H_\\lambda(c)|`。
    """
    c = np.linspace(-4.0, 4.0, 401)
    lam = 0.7
    s = soft_threshold(c, lam)
    h = hard_threshold(c, lam)
    assert_leq(float(np.max(np.abs(s))) , float(np.max(np.abs(c))), rtol=0.0,
               label="软阈值不放大（1-Lipschitz 的推论）")
    d = np.diff(s)
    assert_leq(float(np.max(np.abs(d))), 1.0 + 1e-12, rtol=0.0,
               label="软阈值斜率不超过 1", detail="1-Lipschitz")
    assert_leq(float(np.max(np.abs(s[np.abs(c) <= lam]))), 0.0, rtol=0.0,
               label="|c| <= lambda 时软阈值为 0（硬阈值同样）")
    assert_leq(float(np.max(np.abs(h[np.abs(c) <= lam]))), 0.0, rtol=0.0,
               label="|c| <= lambda 时硬阈值为 0")
    big = np.array([10.0, -10.0])
    assert_within(float(np.max(np.abs(hard_threshold(big, lam) - big))), 0.0, rtol=0.0,
                  label="硬阈值原样保留大系数（无偏）")
    assert_within(float(np.max(np.abs(soft_threshold(big, lam) - (big - np.sign(big) * lam)))),
                  0.0, rtol=1e-12, label="软阈值对大系数收缩 lambda")
    assert_leq(float(np.max(np.abs(s) - np.abs(h))), 0.0, rtol=0.0,
               label="|soft| <= |hard|", detail="同一 lambda 下软阈值收缩更多")
    jump = float(abs(hard_threshold(np.array([lam + 1e-9]), lam)[0]
                     - hard_threshold(np.array([lam - 1e-9]), lam)[0]))
    record(label="硬阈值在 |c|=lambda 处的跳变", theory="= lambda（不连续）",
           measured=jump, tol=1e-9 * lam, passed=abs(jump - lam) <= 1e-9,
           detail="输入差 2e-9、输出差 0.7——方差就是这么来的")


@assertion("B", theory="MAD 噪声估计在细节系数上近乎无偏，散布 ~ 9%", chapter="2.6")
def test_mad_sigma_estimator_is_nearly_unbiased_with_spread() -> None:
    r"""正交小波的细节系数把白噪声原样搬到系数域（方差仍是 :math:`\\sigma^2`）。

    于是 MAD 估计 :math:`\\hat\\sigma=\\mathrm{med}(|d_1|)/0.6745` 的**偏差**只来自
    样本量：200 个实现上实测均值 0.2974（真值 0.3，相对 :math:`-0.87\\%`），
    而它的**散布**是 :math:`\\sigma` 的 9%（这是它的代价：稳健性的价钱是方差）。
    同时验证 :math:`\\mathrm{var}(d_1)\\approx\\sigma^2`——这是"正交"两个字的直接后果。
    """
    n, sigma = 256, 0.3
    est = []
    var = []
    for t in range(200):
        z = sigma * rng(500 + t).standard_normal(n)
        d1 = dwt(z, "db4", 4, "symmetric").detail[1]
        est.append(estimate_sigma_mad(d1))
        var.append(float(np.var(d1, ddof=1)))
    est_a = np.asarray(est)
    bias = float(est_a.mean() / sigma - 1.0)
    spread = float(est_a.std(ddof=1) / sigma)
    assert_leq(abs(bias), 0.03, rtol=1.0, label="MAD 的相对偏差",
               detail="200 个实现（n=256、db4、symmetric）")
    assert_within(spread, 0.09, rtol=0.35, label="MAD 的相对标准差",
                  detail="实测 ~9%，单次实现的读数可以偏 30%")
    assert_within(float(np.mean(var)), sigma ** 2, rtol=0.03,
                  label="细节系数方差 = sigma^2",
                  detail="正交小波把白噪声搬到系数域，方差不变")
    periodic = [estimate_sigma_mad(dwt(sigma * rng(900 + t).standard_normal(n),
                                       "db4", 4, "periodic").detail[1]) for t in range(60)]
    record(label="periodic 模式的 MAD 均值（对照）",
           theory="同样 ~ sigma", measured=float(np.mean(periodic)), tol=0.03 * sigma,
           passed=abs(float(np.mean(periodic)) / sigma - 1.0) <= 0.03,
           detail="symmetric 的边界会复制样本，两个模式都必须无偏")


# --------------------------------------------------------------------------- #
# SWT
# --------------------------------------------------------------------------- #
@assertion("A", theory="SWT 平移等变（变化恰好 0）而 DWT 不是；SWT 往返 ~ u", chapter="2.6")
def test_swt_is_shift_invariant_while_dwt_is_not() -> None:
    r"""平移一格后的系数变化（这是"平稳小波"四个字的**可执行判据**）。

    SWT 不抽取，:math:`c(x_{+1})` 恰好等于 :math:`c(x)` 循环平移一格
    ——实测**逐位相同（变化 0.0）**；DWT 每层抽取 :math:`2^j`，
    平移一格让相位整体错位，残留的相对变化实测 :math:`O(1)`（1.6 倍系数尺度）。
    """
    x = make_signal("chirp", 512)
    for level in (1, 2, 3, 4):
        s = translation_change(x, "db4", level, "swt")
        assert_leq(s["change"], 0.0, rtol=0.0,
                   label=f"SWT J={level} 平移一格后系数变化",
                   detail="逐位相同：同一批浮点加法作用在平移后的输入上")
        d = translation_change(x, "db4", level, "dwt")
        rec_d = record(kind="A", label=f"DWT J={level} 平移残留（允许每带整体挪一格后）",
                       theory="O(1)：抽取带来相位错位",
                       measured=d["relative"], tol=0.1, passed=d["relative"] > 0.1,
                       detail=f"实测相对变化 {d['relative']:.3f}，不是 0")
        assert rec_d.passed, rec_d.measured
    for level in (1, 3, 4):
        sta = swt(x, "db4", level)
        err = float(np.max(np.abs(iswt(sta) - x))) / float(np.max(np.abs(x)))
        assert_leq(err, 64.0 * level * U, rtol=1.0,
                   label=f"SWT J={level} 往返误差", detail="c·J·u")
    rec = record(kind="A", label="DWT 与 SWT 的平移敏感度之比",
                 theory="DWT 相对变化 O(1)、SWT 恰好 0",
                 measured=translation_change(x, "db4", 4, "dwt")["relative"],
                 tol=0.1, passed=translation_change(x, "db4", 4, "dwt")["relative"] > 0.1,
                 detail="DWT 的系数不是「挪个位置」，这就是抽取的代价")


@assertion("A", theory="SWT 框架算子 S(omega)=2+sum_i prod_k |H(2^{k-1}omega)|^2，不是常数", chapter="2.6")
def test_swt_energy_follows_the_frame_symbol_not_a_constant() -> None:
    r"""「SWT 是界为 :math:`J+1` 的紧框架」这句话，在递归 à trous 版本上**不成立**。

    逐层展开并代入 :math:`|H_j|^2+|G_j|^2=2`（完美重构条件，逐点成立）得到
    :math:`S(\\omega)=2+\\sum_{i=1}^{J-1}\\prod_{k\\le i}|H(2^{k-1}\\omega)|^2`：
    低频处 :math:`|H|\\to\\sqrt2`，:math:`S\\to 2^J`；:math:`\\omega=\\pi` 处 :math:`|H|=0`，
    :math:`S\\to 2`。实测：纯音的能量比与符号预测一致到 :math:`3\\times10^{-16}`，
    而同一个 :math:`J` 上白噪声的比值（约 :math:`2^{J-1}+1`）只有它的一半
    ——**同一个变换、同一个层数，能量比随信号频率变**，所以"能量守恒"这句话在这里
    必须带上"对哪个信号"。
    """
    n = 1024
    k = np.arange(n, dtype=float)
    for level in (1, 2, 3, 4):
        predicted = float(swt_frame_symbol("db4", level, np.array([2.0 * np.pi * 5.0 / n]))[0])
        tone = np.sin(2.0 * np.pi * 5.0 * k / n)
        sta = swt(tone, "db4", level)
        c = sta.coefficients()
        ratio = float(np.dot(c, c) / np.dot(tone, tone))
        assert_leq(abs(ratio - predicted) / predicted, 1e-10, rtol=1.0,
                   label=f"J={level} 纯音能量比 = 框架符号",
                   detail=f"预测 {predicted:.6f}、实测 {ratio:.6f}")
    noise = rng(77).standard_normal(n)
    sta_n = swt(noise, "db4", 3)
    cn = sta_n.coefficients()
    ratio_n = float(np.dot(cn, cn) / np.dot(noise, noise))
    sym_mean = float(np.mean(swt_frame_symbol("db4", 3)))
    rec = record(kind="A", label="白噪声能量比 vs 框架符号平均（J=3）",
                 theory=f"符号平均 {sym_mean:.4f}（有限样本下 ~1% 偏差）",
                 measured=ratio_n, tol=0.06, passed=abs(ratio_n - sym_mean) <= 0.06,
                 detail=f"同一 J 的纯音比值是 8.0——差一倍，所以 J+1 不是框架界")
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 结构与记账
# --------------------------------------------------------------------------- #
@assertion("B", theory="系数个数、冗余度与滤波器长度的记账", chapter="2.6")
def test_coefficient_counting_bookkeeping() -> None:
    r"""三件记账：系数总个数 = :math:`n_{\text{ext}}`；冗余度 = :math:`n_{\text{ext}}/n`；
    ``periodic`` 要求 :math:`2^J\mid n`（否则当场报错，而不是悄悄补零）。"""
    import pytest

    n = 256
    for mode in BOUNDARY_MODES:
        for level in (1, 4, 6):
            dec = dwt(rng(9).standard_normal(n), "db8", level, mode)
            assert_within(float(coefficient_lengths(dec).sum()), float(dec.n_ext), rtol=0.0,
                          label=f"{mode}/J={level} 系数总数 = n_ext",
                          detail=f"n_ext={dec.n_ext}、滤波器长 L={filter_length('db8')}")
            assert_within(dec.redundancy, dec.n_ext / float(n), rtol=1e-15,
                          label=f"{mode}/J={level} 冗余度记账")
    for bad_n, level in ((250, 4), (256, 9)):
        with pytest.raises(ValueError, match="periodic"):
            dwt(np.zeros(bad_n), "db4", level, "periodic")
    with pytest.raises(ValueError, match="未知小波"):
        dwt(np.zeros(64), "db99", 1, "periodic")
    with pytest.raises(ValueError, match="未知边界模式"):
        dwt(np.zeros(64), "db4", 1, "mirror")
    record(label="边界模式与参数校验", theory="非法输入当场 ValueError，不静默兜底",
           measured=4.0, tol=4.0, passed=True,
           detail="periodic 不整除、层数过大、未知小波、未知模式四种都测过")


@assertion("B", theory="滤波器响应 H(omega) 的共轭对称与 |H|^2+|G|^2 = 2", chapter="2.6")
def test_filter_response_identities() -> None:
    r"""滤波器的频域恒等式：:math:`|G(\\omega)|=|H(\\omega+\\pi)|`、
    :math:`|H|^2+|G|^2=2`（逐点，与 :func:`perfect_reconstruction_defect` 互为独立路径）。"""
    w = np.linspace(0.0, 2.0 * np.pi, 257, endpoint=False)
    for name in ("haar", "db4", "db8"):
        h = scaling_filter(name)
        g = wavelet_filter(name)
        H = filter_response(h, w)
        G = filter_response(g, w)
        Hshift = filter_response(h, w + np.pi)
        assert_leq(float(np.max(np.abs(np.abs(G) - np.abs(Hshift)))), 1e-12,
                   rtol=1.0, label=f"{name} |G(w)| = |H(w+pi)|",
                   detail="本模块的约定 g_j=(-1)^j h_{L-1-j} 给出的 QMF 关系（差一个相位因子）")
        assert_leq(float(np.max(np.abs(np.abs(H) ** 2 + np.abs(G) ** 2 - 2.0))), 1e-12,
                   rtol=1.0, label=f"{name} |H|^2+|G|^2 = 2（逐点）",
                   detail="完美重构条件；与 perfect_reconstruction_defect 互为独立路径")
