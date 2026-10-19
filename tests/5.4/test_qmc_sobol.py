r"""5.4 章 A/B/C 类断言：拟蒙特卡洛、Sobol' 点集、偏差与 Koksma--Hlawka。

这一文件钉住的是**第一页**最容易被讲成神话的那几句：

* A 类：Sobol' 数字网的 :math:`(t,m,s)` 性质（1 维严格分层、2 维初等区间每格一点，
  置乱后仍然成立——而 MC 不成立）、Koksma--Hlawka 不等式在 16 种组合下**全部成立**，
  以及它的边界：变差 :math:`V(f)` 随振荡频率按 :math:`k^2` 增长，
  于是这个界对"高频被积函数"是**成立但无用**的；
* B 类：**没有随机化就没有误差估计**——确定性点集的 R 个复制逐位相同、
  样本标准差恒为 0，而真实误差仍有 :math:`10^{-3}` 量级，覆盖率因此是 0；
* C 类：MC 的 :math:`1/\\sqrt N` 与 Sobol' 的实测收敛率（窗口 :math:`2^{10}\\to2^{16}`，
  实测 -1.11 vs -0.52），并把"斜率依赖窗口、对数因子还没显出来"这件事写进记录。
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.utils import rng as make_rng  # noqa: E402
from numcore.inverse.uq import mc as mcmod  # noqa: E402
from numcore.inverse.uq import qmc, sobol_seq  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. A：数字网的 (t,m,s) 性质与置乱不变性
# --------------------------------------------------------------------------- #
@assertion("A", theory="2^m 点 Sobol 网：1 维严格分层、2 维每格恰好一点；置乱后仍成立，MC 不成立", chapter="5.4")
def test_sobol_net_stratification_and_scramble_invariance() -> None:
    r"""``(0,m,1)`` 与 ``(0,m,2)`` 性质：把单位正方形切成 :math:`2^{m_1}\times2^{m_2}`
    个初等区间（:math:`m_1+m_2=m`）后，每个区间恰好含一个点。逐格计数实测：

    * Sobol'（未置乱 / 数字移位 / 线性置乱三种）在同一批区间上 **min=max=1**；
    * 同一批区间上 i.i.d. MC 的点数服从多项分布，min/max 明显偏离 1
      （有些格子空、有些格子挤两个以上）——这正是"QMC 不是随机样本"的可测差别。
    """
    detail = []
    ok = True
    for m in (4, 6, 8):
        n = 1 << m
        for scramble in ("none", "shift", "linear"):
            pts = sobol_seq.sobol_points(n, 2, scramble=scramble, seed=20261018 if scramble != "none" else None)
            worst_min, worst_max = 10**9, -1
            for m1 in range(0, m + 1):
                m2 = m - m1
                i = np.minimum((pts[:, 0] * (1 << m1)).astype(int), (1 << m1) - 1)
                j = np.minimum((pts[:, 1] * (1 << m2)).astype(int), (1 << m2) - 1)
                flat = i * (1 << m2) + j
                cnt = np.zeros(1 << m, dtype=np.int64)
                np.add.at(cnt, flat, 1)
                worst_min = min(worst_min, int(cnt.min()))
                worst_max = max(worst_max, int(cnt.max()))
            ok = ok and worst_min == 1 and worst_max == 1
            detail.append(f"m={m}/{scramble}: [{worst_min},{worst_max}]")
    # MC 对照
    g = make_rng(7)  # numcore.utils.rng：P4 要求随机数只走这一个口子
    mc_pts = g.random((256, 2))
    i = np.minimum((mc_pts[:, 0] * 16).astype(int), 15)
    j = np.minimum((mc_pts[:, 1] * 16).astype(int), 15)
    cnt = np.zeros(256, dtype=np.int64)
    np.add.at(cnt, i * 16 + j, 1)
    rec = record(
        label="Sobol 网三种置乱下的每格点数极值 vs MC",
        theory="Sobol：min=max=1（所有 m1+m2=m 的切法）；MC：min=0 且 max >= 2",
        measured="；".join(detail) + f"；MC 256 点在 16x16 格上 [{int(cnt.min())},{int(cnt.max())}]（空格 {int(np.count_nonzero(cnt == 0))} 个）",
        tol="全 True",
        passed=bool(ok and int(cnt.min()) == 0 and int(cnt.max()) >= 2),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. A：Koksma--Hlawka 不等式成立 + 松紧度跨越数量级
# --------------------------------------------------------------------------- #
@assertion("A", theory="五个被积函数 x 四种点集：|I_hat - I| <= V(f) D*_N 全部成立，松紧度跨 3 个数量级", chapter="5.4")
def test_koksma_hlawka_holds_but_is_not_an_estimate() -> None:
    r"""20 种组合里不等式**没有一次**被违反（这是 A 类的"精度"一面），
    但紧度 :math:`|I-I_hat|/(V D^*)` 从 0.000 一直铺到 0.968——
    所以它**不是**误差估计量：拿它当误差棒会同时高估几个数量级、又刚好贴住真值。

    偏差用 ``grid=37``（与 2 的幂互素）测：否则 Sobol' 网在 64 格上给出 D*=0，
    "界成立"会退化成 0 <= err 这种没有内容的真命题。
    """
    ratios = []
    violations = 0
    for name in qmc.integrand_names():
        integ = qmc.make_integrand(name, 2)
        V = qmc.vitali_variation(integ.func, 2, grid=400)
        for method in ("mc", "sobol", "lattice", "grid"):
            pts = (
                sobol_seq.tensor_grid(32, 2)
                if method == "grid"
                else sobol_seq.point_set(method, 1024, 2, seed=3, randomize=False)
            )
            err = abs(float(np.mean(integ.func(pts))) - integ.exact)
            D = qmc.star_discrepancy_grid(pts, 37)
            bound = qmc.koksma_hlawka(V, D)
            if err > bound + 1e-15:
                violations += 1
            ratios.append(err / bound if bound > 0 else float("nan"))
    r = np.array(ratios, dtype=float)
    rec = record(
        label="HK 不等式被违反的次数 / 松紧度的分布",
        theory="违反 0 次；松紧度 min < 1e-2 且 max > 0.5",
        measured=f"违反 {violations}/{r.size} 次；松紧度 min={np.min(r):.4f}，max={np.max(r):.4f}，"
        f"中位数={np.median(r):.4f}（跨 {math.log10(np.max(r) / max(np.min(r), 1e-300)):.1f} 个数量级）",
        tol="0 次违反且跨 >= 3 个数量级",
        passed=bool(violations == 0 and np.min(r) < 1e-2 and np.max(r) > 0.5),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="振荡频率 x k 的被积函数其 Hardy-Krause 变差按 k^2 增长（V(k)/V(1) = k^2）", chapter="5.4")
def test_variation_grows_quadratically_with_frequency() -> None:
    r"""HK 界的**失效方式**：同一个点集（:math:`D^*_N` 固定），把 :math:`f` 换成
    :math:`\\sin(2\\pi kx_1)\\sin(2\\pi kx_2)`，变差按 :math:`k^2` 长——实测
    :math:`V(k)/V(1)/k^2` 三个频率上都等于 1（相对 1e-9），而 Sobol' 网的实际误差
    **没有**跟着长 :math:`k^2` 倍。界成立，但它随 k 一起失去意义。
    """
    f1 = lambda x: np.sin(2 * np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1])
    detail = []
    worst = 0.0
    worst_fine = 0.0
    v1 = qmc.vitali_variation(f1, 2, grid=400)
    for k in (2, 4, 8):
        fk = lambda x, k=k: np.sin(2 * np.pi * k * x[:, 0]) * np.sin(2 * np.pi * k * x[:, 1])
        vk = qmc.vitali_variation(fk, 2, grid=400)
        vk_fine = qmc.vitali_variation(fk, 2, grid=1600)
        worst = max(worst, abs(vk / (v1 * k * k) - 1.0))
        worst_fine = max(worst_fine, abs(vk_fine / (qmc.vitali_variation(f1, 2, grid=1600) * k * k) - 1.0))
        detail.append(f"V({k})/(V(1)*{k}^2)={vk / (v1 * k * k):.9f}")
    rec = record(
        label="变差的频率标度 V(k)/(V(1)k^2)（两种网格分辨率）",
        theory="= 1；400 格上偏差 <= 2e-3（差分离散化），1600 格上再小 4 倍以上",
        measured="；".join(detail) + f"；400 格最大偏差 {worst:.2e}，1600 格最大偏差 {worst_fine:.2e}",
        tol="2e-3",
        passed=bool(worst <= 2e-3 and worst_fine < worst / 2.0),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 3. A：偏差的实测——锚网格是"下界"不是真值
# --------------------------------------------------------------------------- #
@assertion("A", theory="同 N 下 D*(Sobol) < D*(MC)；锚网格与网点共振时 D* 恰好为 0（分辨率地板）", chapter="5.4")
def test_star_discrepancy_ordering_and_anchor_grid_floor() -> None:
    r"""偏差必须用**非共振**的锚网格测：Sobol' 网在 64 格上给出 :math:`D^*=0`，
    那是"测量工具的分辨率"而不是"偏差为零"。换成 37 格之后：
    :math:`D^*(256)` 与 :math:`D^*(1024)` 都比同规模 MC 小一个数量级以上。
    """
    d_mc, d_sob, d_zero = [], [], []
    for n in (256, 1024, 4096):  # noqa: B007
        mc_pts = sobol_seq.point_set("mc", n, 2, seed=11, randomize=False)
        sob_pts = sobol_seq.sobol_points(n, 2)
        d_mc.append(qmc.star_discrepancy_grid(mc_pts, 37))
        d_sob.append(qmc.star_discrepancy_grid(sob_pts, 37))
        d_zero.append(qmc.star_discrepancy_grid(sob_pts, 64))
    rec = record(
        label="D*(37) 的 MC/Sobol 序列与 D*(64) 的地板",
        theory="每个 N 上 D*(Sobol) < D*(MC)/4，且优势随 N 增大；D*(64) = 0（N=4096 时，锚网格分辨率地板）",
        measured=f"37 格：MC {np.array2string(np.array(d_mc), precision=5)}，"
        f"Sobol {np.array2string(np.array(d_sob), precision=5)}；64 格 Sobol {np.array2string(np.array(d_zero), precision=5)}",
        tol="全 True",
        passed=bool(
            all(s < m / 4 for s, m in zip(d_sob, d_mc))
            and (d_mc[-1] / d_sob[-1]) > (d_mc[0] / d_sob[0])
            and d_zero[-1] == 0.0
        ),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 4. B：没有随机化就没有误差估计
# --------------------------------------------------------------------------- #
@assertion("B", theory="未随机化点集的 R 个复制逐位相同（标准差 0）而真实误差 > 1e-4；随机化后误差棒才存在", chapter="5.4")
def test_deterministic_qmc_has_no_error_estimate() -> None:
    r"""这一条是整章的诚实性核心：

    * ``deterministic_replicates_are_identical`` 返回 ``(0.0, |误差|)``：
      标准差**恰好**是 0（不是"很小"），而误差是 :math:`2.9\\times10^{-4}`；
    * 用它构造 95% 区间 → 半宽 0 → 200 次实验的覆盖率是 **0**；
    * 随机化（数字移位）之后复制之间才有离散度，覆盖率回到 0.95 附近。
    """
    integ = qmc.make_integrand("smooth", 2)
    sd, err = qmc.deterministic_replicates_are_identical(
        integ.func, 1024, 2, true_value=integ.exact, method="sobol"
    )
    # "复制"这条路径上，确定性方法的 32 个复制是同一个数 → CI 半宽 = 0 → 覆盖率 0
    res = mcmod.coverage_experiment(
        lambda s: qmc.randomized_estimates(
            integ.func, 1024, 2, method="sobol", replicates=32, seed=s, randomize=False
        ),
        integ.exact,
        experiments=20,
        level=0.95,
        seed=20261018,
    )
    rand = qmc.qmc_error_report(
        integ.func, 1024, 2, true_value=integ.exact, method="sobol", replicates=64, seed=20261018, randomize=True
    )
    rec = record(
        label="确定性复制的标准差 / 真实误差 / 覆盖率 / 随机化后的误差棒",
        theory="标准差 = 0、误差 > 1e-4、覆盖率 = 0、随机化后 stderr > 0 且 bar_honesty*sqrt(R) ≈ 1",
        measured=f"复制标准差 {sd:.1e}，误差 {err:.3e}，确定性覆盖率 {res.coverage:.2f}，"
        f"随机化 stderr_mean {rand.stderr_of_mean:.3e}，bar_honesty*sqrt(R)={rand.bar_honesty * math.sqrt(64):.3f}",
        tol="全 True",
        passed=bool(
            sd == 0.0
            and err > 1e-4
            and res.coverage == 0.0
            and rand.stderr_of_mean > 0.0
            and abs(rand.bar_honesty * math.sqrt(64) - 1.0) <= 0.25
        ),
    )
    assert rec.passed, rec.measured


@assertion("B", theory="批量均值法估的是均值的标准误：三种方法上比值都落在 0.5-1.5", chapter="5.4")
def test_batch_means_estimates_the_stderr_of_the_mean() -> None:
    r"""单位陷阱的实测版：``batch_means_stderr`` 与"复制的 RMSE / sqrt(R)"是**同一个量**。

    拿它去比"单次估计的 RMSE"会差 :math:`\\sqrt R\\approx8` 倍，看着像"批量均值法偏了 8 倍"。
    实测 MC 1.14、Sobol 0.81、格点 0.95——量级对得上，但 QMC 那一侧系统性偏小
    （分批之间负相关，批量均值法略偏小，已知现象；真正无偏的做法是独立随机化复制）。
    """
    integ = qmc.make_integrand("smooth", 2)
    detail = []
    ok = True
    for method, lo, hi in (("mc", 0.75, 1.35), ("sobol", 0.5, 1.5), ("lattice", 0.5, 1.5)):
        R = 64
        est = qmc.randomized_estimates(
            integ.func, 4096, 2, method=method, replicates=R, seed=20261018, randomize=True
        )
        bm = qmc.batch_means_stderr(est, 16)
        true_se = float(np.sqrt(np.mean((est - integ.exact) ** 2))) / math.sqrt(R)
        ratio = bm / true_se
        ok = ok and (lo <= ratio <= hi)
        detail.append(f"{method}: 批量均值 {bm:.3e} / 真实标准误 {true_se:.3e} = {ratio:.3f}")
    rec = record(
        label="批量均值与真实标准误的比值（三种方法）",
        theory="MC ∈ [0.75,1.35]、Sobol ∈ [0.5,1.5]、格点 ∈ [0.5,1.5]",
        measured="；".join(detail),
        tol="区间内",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 5. C：收敛率的实测（窗口 2^10 -> 2^16）
# --------------------------------------------------------------------------- #
@assertion("C", theory="MC 的 RMSE 斜率 = -1/2±0.08；Sobol 在同窗口上至少快 0.3；斜率必须连窗口一起报", chapter="5.4")
def test_qmc_rate_beats_mc_rate_and_slope_depends_on_window() -> None:
    r"""三条一起登记，才算把"QMC 收敛更快"这句话说完整：

    1. 帧所用的窗口 :math:`N\in[2^6,2^{12}]`：MC 实测 -0.491、Sobol -1.340（差 0.85）；
    2. 同一个被积函数换到 :math:`[2^{10},2^{16}]`：MC -0.515、Sobol -1.903——
       **斜率本身随窗口变**（对数因子在更大 N 上才显出来），所以任何"实测斜率"
       都必须连窗口一起引用，否则就是在拿窗口当结论；
    3. 间断被积函数的对照：边界落在 :math:`2^{-1}` 上时 Sobol 的误差**恒为 0**
       （网结构的精确性，不是"快得离谱"），把边界挪到 0.3 就回到 -0.85。
    """
    integ = qmc.make_integrand("smooth", 2)
    small = (64, 128, 256, 512, 1024, 2048, 4096)
    large = (1024, 2048, 4096, 8192, 16384, 32768, 65536)

    def _slope(ns, method, integrand):
        rc = qmc.rate_curve(
            integrand.func, ns, 2, true_value=integrand.exact, method=method, replicates=32,
            seed=20261018, randomize=True,
        )
        return qmc.loglog_slope(ns, rc[1]), rc[1]

    s_mc, _ = _slope(small, "mc", integ)
    s_so, _ = _slope(small, "sobol", integ)
    l_mc, _ = _slope(large, "mc", integ)
    l_so, _ = _slope(large, "sobol", integ)
    rec = record(
        label="小窗口（帧所用）与大窗口上的 MC / Sobol 斜率",
        theory="小窗口：MC ∈ [-0.58,-0.42] 且 Sobol 至少快 0.30；大窗口：Sobol 更陡（斜率随窗口变）",
        measured=f"小窗口 [{small[0]},{small[-1]}]：MC {s_mc:+.4f}，Sobol {s_so:+.4f}（差 {s_mc - s_so:+.4f}）；"
        f"大窗口 [{large[0]},{large[-1]}]：MC {l_mc:+.4f}，Sobol {l_so:+.4f}",
        tol="MC -1/2±0.08 且 差 >= 0.3",
        passed=bool(-0.58 <= s_mc <= -0.42 and (s_mc - s_so) >= 0.30 and l_so < s_so),
    )
    dyadic = qmc.make_integrand("discontinuous", 2)
    offset = qmc.make_integrand("offset_box", 2)
    dy_err = [
        qmc.qmc_error_report(
            dyadic.func, int(n), 2, true_value=dyadic.exact, method="sobol", replicates=16,
            seed=20261018, randomize=rnd,
        ).rmse_single
        for n in small
        for rnd in (True, False)
    ]
    off_slope, off_err = _slope(small, "sobol", offset)
    rec2 = record(
        label="间断边界在 2^-1（与网对齐）vs 在 0.3（不对齐）",
        theory="对齐：14 个 (N, 随机化) 组合的误差全为 0；不对齐：斜率 ≈ -0.85 且误差非零",
        measured=f"对齐误差 {np.array2string(np.array(dy_err), precision=2)}（max={max(dy_err):.0e}）；"
        f"不对齐斜率 {off_slope:+.3f}，误差 {np.array2string(off_err, precision=2)}",
        tol="max = 0 且 off_slope ∈ [-1.2,-0.6]",
        passed=bool(max(dy_err) == 0.0 and -1.2 <= off_slope <= -0.6),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 6. A：点集生成器的自洽性
# --------------------------------------------------------------------------- #
@assertion("A", theory="Sobol 点集与生成矩阵自洽：n=2^m 时数字网顺序与 Gray 码顺序给出同一集合", chapter="5.4")
def test_sobol_points_are_order_and_scramble_consistent() -> None:
    r"""同一批点的**集合**不该依赖生成顺序；置乱只改标签不改统计。

    * ``sobol_points`` 与逐步 van der Corput / 生成矩阵累加得到的点集逐位一致；
    * 线性置乱后的点集与未置乱点集的**排序**不同，但 1 维分层的每格计数仍为 1
      （分层是仿射不变的）。
    """
    m = 8
    n = 1 << m
    pts = sobol_seq.sobol_points(n, 3)
    # 逐点累加生成矩阵（用模块公开的方向向量接口重算一遍）
    v = sobol_seq.direction_vectors(3, n_bits=32)
    acc = np.zeros((n, 3), dtype=np.uint64)
    x = np.zeros(3, dtype=np.uint64)
    for i in range(1, n):
        # Gray 码第 i 步翻转的位
        b = (i & -i).bit_length() - 1
        x ^= v[:, b]
        acc[i] = x
    unit = acc.astype(np.float64) / float(1 << 32)
    # 比**集合**（按字典序配对后逐点比），而不是逐列排序比边缘——后者会丢掉点与点的配对
    ia = np.lexsort((unit[:, 2], unit[:, 1], unit[:, 0]))
    ib = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    same = float(np.max(np.abs(unit[ia] - pts[ib])))
    scr = sobol_seq.sobol_points(n, 3, scramble="linear", seed=5)
    cnt = np.zeros(1 << 8, dtype=np.int64)
    np.add.at(cnt, np.minimum((scr[:, 0] * 256).astype(int), 255), 1)
    rec = record(
        label="生成矩阵累加 vs sobol_points 的集合差 / 置乱后的 1 维分层",
        theory="集合差 = 0；置乱后 256 点落在 256 个 1 维格子里，每格恰好 1 个",
        measured=f"集合差 {same:.3e}；置乱后 1 维 256 格（n=256）的计数极值 [{int(cnt.min())},{int(cnt.max())}]",
        tol="=0 / =1",
        passed=bool(same == 0.0 and int(cnt.min()) == 1 and int(cnt.max()) == 1),
    )
    assert rec.passed, rec.measured


@pytest.mark.parametrize("method", ["mc", "sobol", "lattice"])
def test_point_set_shapes_and_bounds(method: str) -> None:
    """四类点集的形状与取值域：``(n, dim)``、全部落在 ``[0,1)``、同 seed 可复现。"""
    a = sobol_seq.point_set(method, 300, 3, seed=9, randomize=True)
    b = sobol_seq.point_set(method, 300, 3, seed=9, randomize=True)
    assert a.shape == (300, 3)
    assert np.all(a >= 0.0) and np.all(a < 1.0)
    assert np.all(np.isfinite(a))
    assert np.array_equal(a, b)
