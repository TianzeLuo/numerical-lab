"""5.4 章 A/B/C 类断言：蒙特卡洛估计量、置信区间覆盖、直方图与 1/sqrt(N)。

这一文件钉住的是**第一页**（``output_hist`` 与 MC 基线）背后的数学：

* A 类：``norm_ppf`` 与 :mod:`numcore.verify.stats` 的对拍（两条独立实现）、
  四种输入分布的前两阶矩、精确卷积密度与经验分布的 L1 距离，
  以及**边界**——指数分布之和没有本模块给出的闭式密度时必须返回 ``None``，
  而不是编一个"看起来对"的密度；
* B 类：直方图密度积分为 1、bin 数-误差曲线的 U 形、区间覆盖率的频率解释；
* C 类：MC 的 :math:`1/\\sqrt N` 标度（用 ``h = 1/N`` 喂给 :func:`assert_order`，
  于是理论斜率是 +1/2）与 200 次重复实验实测的 95% 覆盖率。
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

from numcore.inverse.uq import mc, qmc  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402
from numcore.verify import fit_order  # noqa: E402
from numcore.verify.stats import norm_ppf as ref_norm_ppf  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. A：正态分位数两条独立实现 + 往返
# --------------------------------------------------------------------------- #
@assertion("A", theory="自研 norm_ppf 与 verify.stats.norm_ppf 逐点一致（|d| <= 1e-9）", chapter="5.4")
def test_norm_ppf_cross_check_and_roundtrip() -> None:
    r"""两条**独立**实现的 Acklam 有理逼近：一条在本模块（UQ 用），一条在 ``numcore.verify.stats``
    （断言框架用）——它们给出同一个数（差 0），且与 :func:`norm_cdf` 往返一致到
    :math:`2.5\times10^{-10}`（这个量级就是 Acklam 逼近自身 1.15e-9 的相对误差，
    不是实现差异；所以往返容差取 1e-8，而不是"拧到刚好通过"）。

    边界：``p <= 0`` / ``p >= 1`` 时返回 ``±inf``（而不是抛异常或静默截断）——
    调用方拿它算置信区间时必须自己挡掉，这一条同时被记录。
    """
    ps = np.array([1e-12, 1e-6, 1e-3, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999, 1 - 1e-9])
    mine = mc.norm_ppf(ps)
    ref = np.array([ref_norm_ppf(float(p)) for p in ps])
    worst = float(np.max(np.abs(mine - ref)))
    # 往返：cdf(ppf(p)) - p
    rt = float(np.max(np.abs(mc.norm_cdf(mine) - ps)))
    lo, hi = float(mc.norm_ppf(0.0)), float(mc.norm_ppf(1.0))
    rec = record(
        label="norm_ppf 与 verify.stats 的最大偏差 + cdf 往返 + 端点行为",
        theory="偏差 = 0（同一有理逼近）、往返 <= 1e-8、端点为 ±inf",
        measured=f"偏差 {worst:.3e}，往返 {rt:.3e}，ppf(0)={lo}，ppf(1)={hi}",
        tol="=0 / <=1e-8 / ±inf",
        passed=bool(worst == 0.0 and rt <= 1e-8 and math.isinf(lo) and math.isinf(hi)),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. A：四种输入分布的矩（含"分布撑"边界）
# --------------------------------------------------------------------------- #
@assertion("A", theory="四种输入分布的样本均值→0、标准差→spread（相对 2%），撑与定义一致", chapter="5.4")
def test_input_distribution_moments_and_support() -> None:
    r"""``uniform/normal/beta/exponential`` 都被**中心化**到均值 0、缩放到标准差 = ``spread``。

    实测 4e5 个样本的前两阶矩与解析值比（相对 2%）；同时检查支撑集：
    正态/指数/均匀的样本撑必须是"整条实轴 / 半轴 / 有界区间"三种不同形态——
    这是"输入分布决定输出分布"的第一层，也是直方图右尾的来源。
    """
    n = 400000
    worst_mu = worst_sd = 0.0
    detail = []
    for name in mc.input_distribution_names():
        d = mc.input_distribution(name)
        z = d.samples(n, spread=1.0, seed=20261018)
        mu = float(np.mean(z))
        sd = float(np.std(z, ddof=1))
        worst_mu = max(worst_mu, abs(mu))
        worst_sd = max(worst_sd, abs(sd - 1.0))
        detail.append(f"{name}: mean={mu:+.4f} std={sd:.4f} range=[{z.min():.2f},{z.max():.2f}]")
    z = mc.sample_inputs("beta", 200000, 2, spread=1.0, seed=7)
    beta_skew = mc.sample_moments(z[:, 0])["skewness"]
    rec = record(
        label="四种分布的一阶/二阶矩与支撑集",
        theory="|mean| <= 0.02、|std-1| <= 0.02",
        measured="；".join(detail) + f"；beta 偏度 {beta_skew:.3f}（Beta(2,5) 解析偏度 0.596）",
        tol="2%",
        passed=bool(worst_mu <= 0.02 and worst_sd <= 0.02 and beta_skew > 0.4),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 3. A：精确卷积密度 vs 经验分布，以及"没有闭式就返回 None"
# --------------------------------------------------------------------------- #
@assertion("A", theory="均匀输入之和的精确三角密度与经验直方图的 L1 距离 < 0.02；无闭式时返回 None", chapter="5.4")
def test_exact_sum_density_matches_empirical_and_declines_when_unknown() -> None:
    r"""双向 A 类：

    * 能写出闭式的（两个独立均匀之和 → 三角分布）必须**对得上**经验直方图；
    * 写不出闭式的（指数之和）必须返回 ``None``——这条比第一条重要：
      上一版这里会给出一条"看起来像密度"的曲线，而它并不归一。
    """
    n, bins = 200000, 60
    x = mc.sample_inputs("uniform", n, 2, spread=1.0, seed=11)
    y = x[:, 0] + x[:, 1]
    lo, hi = float(y.min()), float(y.max())
    centers, dens, _ = mc.histogram_density(y, n_bins=bins, lo=lo, hi=hi)
    ref = mc.exact_sum_density("uniform", 1.0, centers)
    assert ref is not None
    l1 = float(np.sum(np.abs(dens - ref)) * (centers[1] - centers[0]))
    # 归一性：密度积分必须为 1（否则"精确密度"这个说法就站不住）
    mass = float(np.sum(ref) * (centers[1] - centers[0]))
    none_for_exp = mc.exact_sum_density("exponential", 1.0, centers) is None
    rec = record(
        label="均匀和的经验 L1 距离 / 精确密度积分 / 指数和的返回值",
        theory="L1 < 0.02、积分 = 1（误差 <= 1e-4 是 60 格黎曼和的求积误差）、指数返回 None",
        measured=f"L1={l1:.5f}，积分={mass:.6f}，指数分布的返回值={mc.exact_sum_density('exponential', 1.0, centers)!r}",
        tol="0.02 / 1±1e-4 / None",
        passed=bool(l1 < 0.02 and abs(mass - 1.0) < 1e-4 and none_for_exp),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 4. B：直方图密度的归一性与 bin 权衡的 U 形
# --------------------------------------------------------------------------- #
@assertion("B", theory="直方图密度积分为 1；bin 数-误差曲线在两端都变差（内部有极小）", chapter="5.4")
def test_histogram_normalization_and_bin_tradeoff() -> None:
    r"""``sum(density * h) = 1``（相对 1e-12），且 bin 权衡曲线呈 U 形：
    B=8 与 B=96 处的误差都**大于**中间的极小值——这正是"bin 越多越好"的反例。
    """
    y = mc.sample_inputs("normal", 50000, 2, spread=1.0, seed=5).sum(axis=1)
    centers, dens, counts = mc.histogram_density(y, n_bins=50)
    h = float(centers[1] - centers[0])
    mass = float(np.sum(dens) * h)
    n_in = int(np.sum(counts))
    bins, err = mc.bin_width_tradeoff(
        lambda c: np.exp(-0.5 * ((c - float(np.mean(y))) / float(np.std(y))) ** 2)
        / (float(np.std(y)) * math.sqrt(2 * math.pi)),
        lambda sd: mc.sample_inputs("normal", 20000, 2, spread=1.0, seed=sd).sum(axis=1),
        n=20000,
        bins_list=(8, 12, 16, 24, 32, 48, 64, 96),
        replicates=4,
        seed=99,
    )
    i_min = int(np.argmin(err))
    u_shape = bool(err[0] > err[i_min] and err[-1] > err[i_min])
    rec = record(
        label="直方图密度积分 / 计入样本数 / bin 误差曲线形状",
        theory="积分 = 1（1e-12）、计数和 = N、两端误差 > 内部极小",
        measured=f"积分={mass:.12f}，计数 {n_in}/50000，极小在 B={int(bins[i_min])}（误差 {err[i_min]:.4f}），"
        f"B=8 → {err[0]:.4f}，B=96 → {err[-1]:.4f}",
        tol="全 True",
        passed=bool(abs(mass - 1.0) < 1e-12 and n_in == 50000 and u_shape),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 5. C：MC 的 1/sqrt(N) 标度
# --------------------------------------------------------------------------- #
@assertion("C", theory="MC 的 RMSE = sigma/sqrt(N)，双对数斜率 = 1/2（以 h=1/N 为自变量）", chapter="5.4")
def test_mc_rate_is_one_over_sqrt_n() -> None:
    r"""实测斜率必须落在 :math:`1/2\\pm0.08`，且与 ``loglog_slope`` 的定义一致。

    R=64 个独立复制、8 个 N（2^6..2^13）、``head=6``（切掉最细端，那里 RMSE 估计
    自身的统计误差相对最大）。同一批数据用 :func:`numcore.verify.fit_order` 与
    本模块的 :func:`numcore.inverse.uq.qmc.loglog_slope` 两条路径拟合，结果必须一致。
    """
    integ = qmc.make_integrand("smooth", 2)
    ns = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192])
    rmse = np.array(
        [
            qmc.qmc_error_report(
                integ.func, int(n), 2, true_value=integ.exact, method="mc",
                replicates=64, seed=20261018, randomize=True,
            ).rmse_single
            for n in ns
        ]
    )
    order = fit_order(1.0 / ns, rmse, head=6)
    mine = qmc.loglog_slope(ns, rmse, head=6)
    rec = record(
        label="MC 的双对数斜率（框架 fit_order 与本模块 loglog_slope）",
        theory="0.5 ± 0.08，且两条路径一致",
        measured=f"fit_order {order.slope:.4f}（R²={order.r2:.6f}，n_used={order.n_used}），"
        f"loglog_slope {-mine:.4f}，误差序列 {np.array2string(rmse, precision=5)}",
        tol="±0.08",
        passed=bool(abs(order.slope - 0.5) <= 0.08 and abs(order.slope + mine) <= 1e-12),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 6. C：95% 置信区间的实测覆盖率与它的偏差来源
# --------------------------------------------------------------------------- #
@assertion("C", theory="95% CI 的实测覆盖率：N=100 时重尾输出低于正态输出，N=1000 时两者都回到 0.95", chapter="5.4")
def test_confidence_interval_coverage_versus_sample_size() -> None:
    r"""``mean +- 1.96 s/sqrt(N)`` 的覆盖率必须被**实测**，而且要把"什么时候不够"一起报出来。

    同一个区间公式，两种输出：

    * 线性 + 正态输入（输出本身就是正态）→ 均值**精确**正态，任何 N 都该覆盖 95%；
    * 二次 + Beta(2,5) 输入（输出偏度 +2.6、超额峰度 +11）→ 均值只是**近似**正态，
      小 N 下覆盖率实测偏低，N 变大后才回到 0.95。

    这里刻意把"覆盖率不足"写成**随 N 消失**的现象，而不是一句"重尾会失效"：
    300 次实验的二项标准误约 0.013，两个覆盖率之差必须放在这个尺度上读。
    """
    lin = mc.input_distribution("normal")
    beta = mc.input_distribution("beta")

    def _run(dist, model, n, true_value):
        return mc.coverage_experiment(
            lambda sd: mc.forward_model(
                model,
                np.column_stack([dist.samples(n, spread=1.0, seed=sd), dist.samples(n, spread=1.0, seed=int(sd) + 1)]),
            ),
            true_value,
            experiments=300,
            level=0.95,
            seed=20261018,
        )

    small_lin = _run(lin, "linear", 100, 0.0)
    small_skew = _run(beta, "quadratic", 100, 1.0)
    big_lin = _run(lin, "linear", 1000, 0.0)
    big_skew = _run(beta, "quadratic", 1000, 1.0)
    gap = small_lin.coverage - small_skew.coverage
    rec = record(
        label="N=100 时两种输出的 95% 覆盖率与其差",
        theory="正态输出 ∈ [0.90,0.99]；重尾输出低于它（差距在二项标准误 0.013 的 2 倍量级）",
        measured=f"正态 {small_lin.coverage:.4f}（命中 {small_lin.hits}/300）vs 重尾 {small_skew.coverage:.4f}"
        f"（命中 {small_skew.hits}/300），差 {gap:+.4f}（se≈{small_lin.binomial_se:.4f}），"
        f"半宽 {small_lin.mean_halfwidth:.4f} vs {small_skew.mean_halfwidth:.4f}",
        tol="0.90-0.99 且 差 > 0",
        passed=bool(
            0.90 <= small_lin.coverage <= 0.99
            and small_skew.coverage < small_lin.coverage
            and abs(small_lin.coverage - 0.95) <= 3 * small_lin.binomial_se
        ),
    )
    rec2 = record(
        label="N=1000 时同一个重尾输出的覆盖率（差距是否消失）",
        theory="两者都回到 0.95 附近（|覆盖率 - 0.95| <= 3 se）",
        measured=f"正态 {big_lin.coverage:.4f}，重尾 {big_skew.coverage:.4f}，"
        f"|重尾-0.95|={abs(big_skew.coverage - 0.95):.4f}（3se={3 * big_skew.binomial_se:.4f}）",
        tol="|Δ| <= 3 se",
        passed=bool(
            abs(big_lin.coverage - 0.95) <= 3 * big_lin.binomial_se
            and abs(big_skew.coverage - 0.95) <= 3 * big_skew.binomial_se
        ),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 7. B：蜜度估计的 KS 距离随 N 收缩、且对"参考线选错"敏感
# --------------------------------------------------------------------------- #
@assertion("B", theory="KS(经验, 精确参考) 随 N 按 ~N^(-1/2) 收缩；对 CLT 参考线的 KS 不会随之消失", chapter="5.4")
def test_ks_distance_shrinks_for_correct_reference_only() -> None:
    r"""把两件事分开：抽样误差（KS ~ N^{-1/2}）与参考线本身的偏差。

    均匀输入 + 线性模型 → 输出是三角分布，用**精确**三角密度当参考时 KS 随 N 收缩；
    用**同均值同方差正态**当参考时 KS 停在一个正的地板上（它是 CLT 的偏差，不是噪声）。
    """
    ks_exact = []
    ks_clt = []
    ns = (2000, 8000, 32000)
    for n in ns:
        y = mc.sample_inputs("uniform", n, 2, spread=1.0, seed=4242).sum(axis=1)
        m = mc.sample_moments(y)
        s = m["std"]
        grid = np.linspace(m["mean"] - 6 * s, m["mean"] + 6 * s, 2001)
        dens = mc.exact_sum_density("uniform", 1.0, grid)
        cdf = np.cumsum(dens) * (grid[1] - grid[0])
        cdf = cdf / cdf[-1]
        ys = np.sort(y)
        cdf_y = np.interp(ys, grid, cdf, left=0.0, right=1.0)
        emp = np.arange(1, ys.size + 1) / ys.size
        ks_exact.append(float(np.max(np.abs(emp - cdf_y))))
        ks_clt.append(mc.ks_distance(y, m["mean"] + s * mc.norm_ppf(
            (np.arange(1, y.size + 1) - 0.5) / y.size
        )))
    npp = np.array(ks_exact, dtype=float)
    rec = record(
        label="精确参考下的 KS 序列 / CLT 参考下的 KS 序列",
        theory="精确参考：KS(32000) < KS(2000)；CLT 参考：KS 不随 N 消失（地板 > 5e-3）",
        measured=f"精确 {np.array2string(npp, precision=5)}（N={ns}），"
        f"CLT {np.array2string(np.array(ks_clt), precision=5)}",
        tol="单调下降 / 地板 > 5e-3",
        passed=bool(npp[-1] < npp[0] and np.array(ks_clt)[-1] > 5e-3 and np.array(ks_clt)[-1] > npp[-1]),
    )
    assert rec.passed, rec.measured


@pytest.mark.parametrize("name", ["uniform", "normal", "beta", "exponential"])
def test_sample_inputs_shapes_and_seed_reproducibility(name: str) -> None:
    """``sample_inputs`` 的形状/可复现性：同 seed 逐位相同、不同 seed 不同。"""
    a = mc.sample_inputs(name, 1024, 2, spread=1.0, seed=3)
    b = mc.sample_inputs(name, 1024, 2, spread=1.0, seed=3)
    c = mc.sample_inputs(name, 1024, 2, spread=1.0, seed=4)
    assert a.shape == (1024, 2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(np.isfinite(a))
