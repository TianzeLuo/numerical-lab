r"""蒙特卡洛：抽样、样本矩、置信区间、覆盖率，以及一个**自研**的直方图密度。

这一层的立场
------------
"蒙特卡洛的误差是 :math:`\sigma/\sqrt N`"是全书最容易被当成保证的标度关系。
本模块把它拆成三个可以分别检验的量：

1. **标度**：:math:`\sigma/\sqrt N` 是标准差随 :math:`N` 的**幂律**（:func:`mean_stderr_curve`
   给出实测斜率，理论 :math:`-1/2`）。
2. **区间**：:func:`mean_confidence_interval` 给出 95% 区间；
   :func:`coverage_experiment` 用 R 次独立实验数出真值落在区间里的**频率**。
   频率不到 0.95 时，问题不在"随机数不好"，而在下面第 3 条。
3. **偏差**：区间用的是样本标准差 :math:`s`（不是真 :math:`\sigma`）、样本本身可能相关、
   分布可能重尾——三者都会让覆盖率低于名义值。每条都有实测（``tests/5.4``）。

P3 禁止 ``np.histogram`` / ``np.bincount`` / ``np.digitize``，所以直方图自己写：
:func:`histogram_density` 用 :func:`numpy.searchsorted` + 差分计数（O(N log B)），
并把"bin 宽度"这个**人为选择**连同它对误差的影响一起暴露出来
（:func:`bin_width_tradeoff` 给出 U 形误差曲线，最优 bin 数不是"越多越好"）。

正态分位数用 Acklam 有理逼近**自己实现**（:func:`norm_ppf`）：内核不 import
:mod:`numcore.verify`（分层规矩），`tests/5.4` 拿两边逐位对拍，所以"图上的分位数"
与"断言里的分位数"仍然是同一个数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from numcore.utils import rng as make_rng, seed_sequence

__all__ = [
    "InputDistribution",
    "INPUT_DISTRIBUTIONS",
    "input_distribution",
    "input_distribution_names",
    "MODELS",
    "model_names",
    "forward_model",
    "model_label",
    "standard_normal",
    "normal_samples",
    "exponential_samples",
    "beta_order_statistic",
    "sample_inputs",
    "norm_cdf",
    "norm_ppf",
    "mean_confidence_interval",
    "coverage_experiment",
    "CoverageResult",
    "mean_stderr_curve",
    "sample_moments",
    "histogram_density",
    "bin_width_tradeoff",
    "ks_distance",
    "ks_to_normal",
    "exact_sum_density",
]

#: Acklam 逆正态 CDF 系数（相对误差 < 1.15e-9；与 :mod:`numcore.verify.stats` 各自独立实现）
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


# --------------------------------------------------------------------------- #
# 输入分布（viz 2 的"输入分布"参数）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InputDistribution:
    """一维输入分布：名字、标签、理论均值/方差、以及**抽样函数**。

    ``spread`` 参数把分布缩放到指定的标准差（``spread=1`` 时标准差为 1），
    于是"拖大 spread"改变的是**不确定性的大小**，而不是分布形状——两件事必须能分开拖。
    """

    name: str
    label: str
    note: str
    mean: float
    std_unscaled: float
    sampler: Callable[[int, Any], np.ndarray]

    def samples(self, n: int, *, spread: float = 1.0, seed: Any = None) -> np.ndarray:
        z = self.sampler(int(n), seed)
        return float(spread) * z / self.std_unscaled


def _sampler_normal(n: int, seed: Any) -> np.ndarray:
    # Box-Muller：两个均匀数造一个正态数（自研，不用 numpy 的现成分布）
    g = make_rng(seed)
    u1 = g.random(n)
    u2 = g.random(n)
    r = np.sqrt(-2.0 * np.log(np.maximum(u1, np.finfo(float).tiny)))
    return r * np.cos(2.0 * np.pi * u2)


def _sampler_uniform(n: int, seed: Any) -> np.ndarray:
    g = make_rng(seed)
    return np.sqrt(3.0) * (2.0 * g.random(n) - 1.0)  # 标准差恰为 1


def _sampler_beta(n: int, seed: Any, a: int = 2, b: int = 5) -> np.ndarray:
    # Beta(a, b)（整数参数）= a+b-1 个均匀数的第 a 个次序统计量；再标准化到均值 0、标准差 1
    g = make_rng(seed)
    total = a + b - 1
    u = np.sort(g.random((n, total)), axis=1)
    x = u[:, a - 1]
    mu = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return (x - mu) / math.sqrt(var)


def _sampler_exponential(n: int, seed: Any) -> np.ndarray:
    # 逆变换法：X = -ln(1-U) ~ Exp(1)，再标准化
    g = make_rng(seed)
    u = g.random(n)
    return np.log(1.0 / np.maximum(1.0 - u, np.finfo(float).tiny)) - 1.0


INPUT_DISTRIBUTIONS: dict[str, InputDistribution] = {
    "uniform": InputDistribution(
        "uniform",
        "均匀 U(-sqrt(3), sqrt(3))",
        "有界、对称；输出的极端值只能来自输入的极端值，尾部很薄",
        0.0,
        1.0,
        _sampler_uniform,
    ),
    "normal": InputDistribution(
        "normal",
        "正态 N(0, 1)",
        "对称、轻尾；蒙特卡洛的教科书场景，CLT 参考线在这里最准",
        0.0,
        1.0,
        _sampler_normal,
    ),
    "beta": InputDistribution(
        "beta",
        "Beta(2, 5) 标准化",
        "有界但**强右偏**；直方图与正态参考线的差里有相当一部分是 CLT 自己的偏差",
        0.0,
        1.0,
        _sampler_beta,
    ),
    "exponential": InputDistribution(
        "exponential",
        "指数 Exp(1) 中心化",
        "无界、右偏；样本方差本身就不稳定，置信区间的覆盖率会掉",
        0.0,
        1.0,
        _sampler_exponential,
    ),
}


def input_distribution_names() -> tuple[str, ...]:
    """可用输入分布的名字（顺序稳定）。"""
    return tuple(INPUT_DISTRIBUTIONS)


def input_distribution(name: str) -> InputDistribution:
    """按名字取输入分布。"""
    key = str(name)
    if key not in INPUT_DISTRIBUTIONS:
        raise KeyError(f"未知输入分布 {name!r}（可用：{sorted(INPUT_DISTRIBUTIONS)}）")
    return INPUT_DISTRIBUTIONS[key]


# --------------------------------------------------------------------------- #
# 前向模型（viz 2 的"模型"参数）
# --------------------------------------------------------------------------- #
def _model_linear(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + x[:, 1]


def _model_quadratic(x: np.ndarray) -> np.ndarray:
    return x[:, 0] ** 2 + 0.5 * x[:, 1]


def _model_stiff(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x[:, 0], -6.0, 6.0)) + 0.5 * x[:, 1]


MODELS: dict[str, tuple[str, str, Callable[[np.ndarray], np.ndarray]]] = {
    "linear": (
        "线性 Y = X1 + X2",
        "两个独立输入之和；均匀输入时输出密度是精确的三角形（可解析对拍）",
        _model_linear,
    ),
    "quadratic": (
        "二次 Y = X1^2 + 0.5 X2",
        "对称输入也会给出**右偏**输出（平方项）；正态参考线在这里开始骗人",
        _model_quadratic,
    ),
    "stiff": (
        "强非线性 Y = exp(X1) + 0.5 X2",
        "输出近似对数正态：重尾 + 强右偏，是「1/sqrt(N) 不够用」的最短路径",
        _model_stiff,
    ),
}


def model_names() -> tuple[str, ...]:
    """可用前向模型的名字（顺序稳定）。"""
    return tuple(MODELS)


def model_label(name: str) -> str:
    """模型的中文标签（进帧的 ``strings``）。"""
    key = str(name)
    if key not in MODELS:
        raise KeyError(f"未知模型 {name!r}（可用：{sorted(MODELS)}）")
    return MODELS[key][0]


def forward_model(name: str, x: np.ndarray) -> np.ndarray:
    """把 ``(n, 2)`` 的输入样本映成 ``(n,)`` 的输出样本。"""
    key = str(name)
    if key not in MODELS:
        raise KeyError(f"未知模型 {name!r}（可用：{sorted(MODELS)}）")
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("forward_model 需要 (n, 2) 的输入")
    return np.asarray(MODELS[key][2](arr), dtype=np.float64)


# --------------------------------------------------------------------------- #
# 抽样
# --------------------------------------------------------------------------- #
def standard_normal(n: int, *, seed: Any = None) -> np.ndarray:
    """``n`` 个标准正态样本（Box–Muller，自研）。"""
    return _sampler_normal(int(n), seed)


def normal_samples(n: int, *, mean: float = 0.0, std: float = 1.0, dim: int = 1, seed: Any = None) -> np.ndarray:
    """``(n, dim)`` 的正态样本。"""
    g = make_rng(seed)
    z = np.empty((int(n), int(dim)), dtype=np.float64)
    for j in range(int(dim)):
        u1 = g.random(int(n))
        u2 = g.random(int(n))
        r = np.sqrt(-2.0 * np.log(np.maximum(u1, np.finfo(float).tiny)))
        z[:, j] = r * np.cos(2.0 * np.pi * u2)
    return float(mean) + float(std) * z


def exponential_samples(n: int, *, rate: float = 1.0, seed: Any = None) -> np.ndarray:
    """``n`` 个指数样本（逆变换法，自研）。"""
    g = make_rng(seed)
    u = g.random(int(n))
    return -np.log(np.maximum(1.0 - u, np.finfo(float).tiny)) / float(rate)


def beta_order_statistic(n: int, *, a: int = 2, b: int = 5, seed: Any = None) -> np.ndarray:
    """``Beta(a, b)``（整数参数）用**次序统计量**精确抽样：取 ``a+b-1`` 个均匀数的第 ``a`` 小。

    这是精确的（不是近似）：:math:`U_{(a)}\\sim\\mathrm{Beta}(a,b)` 当 :math:`U_{(a)}` 是
    :math:`a+b-1` 个 i.i.d. 均匀数的第 :math:`a` 个次序统计量。
    """
    if a < 1 or b < 1:
        raise ValueError("a, b 必须是正整数")
    return _sampler_beta(int(n), seed, int(a), int(b))


def sample_inputs(name: str, n: int, dim: int = 2, *, spread: float = 1.0, seed: Any = None) -> np.ndarray:
    """``(n, dim)`` 的输入样本：每一维都取自同一个输入分布（互相独立）。"""
    dist = input_distribution(name)
    # 每一维用**不同**的整数种子（``seed_sequence`` 派生）：把 Generator 直接传下去
    # 会在 ``make_rng`` 里炸（本模块第一版如此，而调用方看起来完全正常）
    seeds = seed_sequence(20261018 if seed is None else int(seed), int(dim))
    out = np.empty((int(n), int(dim)), dtype=np.float64)
    for j in range(int(dim)):
        out[:, j] = dist.samples(int(n), spread=float(spread), seed=int(seeds[j]))
    return out


# --------------------------------------------------------------------------- #
# 正态分布函数
# --------------------------------------------------------------------------- #
def norm_cdf(x: Any) -> np.ndarray:
    """标准正态 CDF（``math.erf``，精确到机器精度）。"""
    arr = np.asarray(x, dtype=np.float64)
    flat = np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in arr.ravel()])
    return flat.reshape(arr.shape)


def norm_ppf(p: Any) -> np.ndarray:
    """标准正态分位数（Acklam 有理逼近，|绝对误差| < 1.2e-9）。"""
    q = np.asarray(p, dtype=np.float64)
    out = np.empty_like(q)
    for i, pv in enumerate(q.ravel()):
        if not math.isfinite(float(pv)) or pv < 0.0 or pv > 1.0:
            out.ravel()[i] = float("nan")
        elif pv == 0.0:
            out.ravel()[i] = -math.inf
        elif pv == 1.0:
            out.ravel()[i] = math.inf
        elif pv < _P_LOW:
            t = math.sqrt(-2.0 * math.log(pv))
            num = ((((_C[0] * t + _C[1]) * t + _C[2]) * t + _C[3]) * t + _C[4]) * t + _C[5]
            den = (((_D[0] * t + _D[1]) * t + _D[2]) * t + _D[3]) * t + 1.0
            out.ravel()[i] = num / den
        elif pv <= _P_HIGH:
            t = pv - 0.5
            r = t * t
            num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * t
            den = ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
            out.ravel()[i] = num / den
        else:
            t = math.sqrt(-2.0 * math.log(1.0 - pv))
            num = ((((_C[0] * t + _C[1]) * t + _C[2]) * t + _C[3]) * t + _C[4]) * t + _C[5]
            den = (((_D[0] * t + _D[1]) * t + _D[2]) * t + _D[3]) * t + 1.0
            out.ravel()[i] = -num / den
    return out


# --------------------------------------------------------------------------- #
# 区间、覆盖率、标度
# --------------------------------------------------------------------------- #
def mean_confidence_interval(samples: Any, *, level: float = 0.95) -> tuple[float, float]:
    r"""样本均值的置信区间 ``(均值, 半宽)``：:math:`\bar x\pm z_{1-\alpha/2}\,s/\sqrt n`。

    用的是**样本**标准差 :math:`s` 而不是真 :math:`\sigma`，且用的是**正态**分位数：
    ``n`` 很小时这两件事都会让真实覆盖率低于名义值（``tests/5.4`` 里 n=5 的实测覆盖率
    明显低于 0.95）。这不是实现的毛病，是"用 :math:`z` 代替 :math:`t`"的代价——
    写在这里，免得有人以为覆盖率总是 0.95。
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        raise ValueError("至少需要 2 个样本")
    s = float(np.std(x, ddof=1))
    half = float(norm_ppf(0.5 + 0.5 * float(level))) * s / math.sqrt(n)
    return float(np.mean(x)), half


@dataclass(frozen=True)
class CoverageResult:
    """R 次独立实验里，"真值落在 95% 区间内"的频率与它的二项误差。"""

    experiments: int
    hits: int
    coverage: float
    nominal: float
    level: float
    mean_halfwidth: float
    binomial_se: float

    @property
    def gap(self) -> float:
        """实测覆盖率与名义水平的差（小于 0 = 系统性低估误差）。"""
        return float(self.coverage - self.nominal)

    @property
    def z_score(self) -> float:
        """把差换算成"几个二项标准误"——判断这点差距是不是抽样噪声。"""
        return float(self.gap / self.binomial_se) if self.binomial_se > 0 else float("nan")


def coverage_experiment(
    sampler: Callable[[Any], np.ndarray],
    true_value: float,
    *,
    experiments: int = 200,
    level: float = 0.95,
    seed: Any = None,
) -> CoverageResult:
    r"""跑 ``experiments`` 次独立实验，数真值落在 :math:`\bar x\pm z s/\sqrt n` 里的频率。

    ``sampler(seed) -> samples`` 由调用方给出，于是"i.i.d. 正态"、
    "重尾"、"QMC 点集"三种情形可以走同一条代码路径，差异只在样本本身。
    二项标准误取 :math:`\sqrt{p(1-p)/R}`（用实测 :math:`p`），用于判断差距是否显著。
    """
    r = int(experiments)
    if r < 2:
        raise ValueError("experiments 至少为 2")
    parent = 20261018 if seed is None else seed
    hits = 0
    widths = np.empty(r, dtype=np.float64)
    for i in range(r):
        x = np.asarray(sampler(int(parent) + 7919 * i), dtype=np.float64).ravel()
        mu, half = mean_confidence_interval(x, level=level)
        widths[i] = half
        if mu - half <= float(true_value) <= mu + half:
            hits += 1
    p = hits / float(r)
    return CoverageResult(
        experiments=r,
        hits=int(hits),
        coverage=float(p),
        nominal=float(level),
        level=float(level),
        mean_halfwidth=float(np.mean(widths)),
        binomial_se=float(math.sqrt(max(p * (1.0 - p), 1e-12) / r)),
    )


def mean_stderr_curve(
    sampler: Callable[[Any], np.ndarray],
    n_list: Sequence[int],
    *,
    replicates: int = 200,
    seed: Any = None,
) -> np.ndarray:
    """每个 ``n`` 上"样本均值的标准差"（跨 ``replicates`` 次实验实测）——1/sqrt(N) 的实测曲线。"""
    ns = list(n_list)
    out = np.empty(len(ns), dtype=np.float64)
    parent = 20261018 if seed is None else seed
    for j, n in enumerate(ns):
        means = np.empty(int(replicates), dtype=np.float64)
        for i in range(int(replicates)):
            means[i] = float(np.mean(sampler(int(parent) + 104729 * i)))
        out[j] = float(np.std(means, ddof=1))
    return out


def sample_moments(samples: Any) -> dict[str, float]:
    """样本矩：均值、方差、偏度、超额峰度（都自研，不调 ``scipy.stats``）。"""
    x = np.asarray(samples, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        raise ValueError("至少需要 2 个样本")
    mu = float(np.mean(x))
    d = x - mu
    m2 = float(np.mean(d * d))
    if m2 <= 0.0:
        return {"mean": mu, "var": 0.0, "std": 0.0, "skewness": 0.0, "excess_kurtosis": 0.0, "n": float(n)}
    m3 = float(np.mean(d**3))
    m4 = float(np.mean(d**4))
    return {
        "mean": mu,
        "var": m2,
        "std": math.sqrt(m2),
        "skewness": m3 / m2**1.5,
        "excess_kurtosis": m4 / (m2 * m2) - 3.0,
        "n": float(n),
    }


# --------------------------------------------------------------------------- #
# 直方图（自研）与它的偏差-方差权衡
# --------------------------------------------------------------------------- #
def histogram_density(samples: Any, *, n_bins: int = 40, lo: float | None = None, hi: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """等宽直方图密度：返回 ``(bin 中心, 密度, 计数)``。

    密度 = 计数 / (N · 宽度)，于是 :math:`\\sum_j \\text{密度}_j\\cdot h = 1`（B 类不变量）。
    边界外的样本被**计入最近的一个 bin**（而不是丢掉）：丢样本会让"密度积分为 1"
    这条不变量悄悄失效，而在图上完全看不出来。``lo``/``hi`` 缺省取样本极值。
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    n = x.size
    b = int(n_bins)
    if n < 2 or b < 1:
        raise ValueError("需要至少 2 个样本与至少 1 个 bin")
    a = float(np.min(x)) if lo is None else float(lo)
    c = float(np.max(x)) if hi is None else float(hi)
    if not c > a:
        c = a + 1.0
    edges = np.linspace(a, c, b + 1)
    width = (c - a) / b
    pos = np.searchsorted(edges, x, side="right") - 1
    pos = np.clip(pos, 0, b - 1)
    counts = np.zeros(b, dtype=np.float64)
    np.add.at(counts, pos, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (float(n) * width)
    return centers, density, counts


def bin_width_tradeoff(
    ref_density: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[Any], np.ndarray],
    *,
    n: int,
    bins_list: Sequence[int],
    replicates: int = 8,
    seed: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""bin 数 vs 密度误差（RMS）的 U 形曲线：返回 ``(bins, rmse)``。

    误差有两个来源，方向相反：bin 少 → **偏差** :math:`O(h^2)`；
    bin 多 → **方差** :math:`O(1/\\sqrt{Nh})`。于是存在一个最优 bin 数，
    它随 :math:`N` 变——"把 bin 调细看得更清楚"是错的。
    """
    bins = np.asarray(list(bins_list), dtype=np.int64)
    out = np.empty(bins.size, dtype=np.float64)
    parent = 20261018 if seed is None else seed
    for j, b in enumerate(bins):
        errs = np.empty(int(replicates), dtype=np.float64)
        for i in range(int(replicates)):
            x = np.asarray(sampler(int(parent) + 15485863 * i), dtype=np.float64).ravel()
            centers, dens, _ = histogram_density(x, n_bins=int(b))
            errs[i] = math.sqrt(float(np.mean((dens - ref_density(centers)) ** 2)))
        out[j] = float(np.mean(errs))
    return bins, out


def ks_distance(a: Any, b: Any) -> float:
    """两个样本的 Kolmogorov–Smirnov 距离（自研：合并排序 + 经验 CDF 差的最大值）。"""
    x = np.sort(np.asarray(a, dtype=np.float64).ravel())
    y = np.sort(np.asarray(b, dtype=np.float64).ravel())
    if x.size < 1 or y.size < 1:
        raise ValueError("样本不能为空")
    grid = np.sort(np.concatenate([x, y]))
    cx = np.searchsorted(x, grid, side="right") / float(x.size)
    cy = np.searchsorted(y, grid, side="right") / float(y.size)
    return float(np.max(np.abs(cx - cy)))


def ks_to_normal(samples: Any) -> float:
    r"""样本到"同均值同标准差正态分布"的 KS 距离 :math:`\sup_x|F_N(x)-\Phi((x-\mu)/s)|`。

    这是 **CLT 参考线是否可信** 的直接度量：对称轻尾输出上它 ~ :math:`1/\\sqrt N`，
    强右偏输出上它是前者的好几倍（``tests/5.4`` 有实测对照）。
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        raise ValueError("至少需要 2 个样本")
    mu = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s <= 0.0:
        return 0.0
    z = np.sort((x - mu) / s)
    cdf = norm_cdf(z)
    upper = np.arange(1, n + 1, dtype=np.float64) / n - cdf
    lower = cdf - np.arange(0, n, dtype=np.float64) / n
    return float(max(np.max(upper), np.max(lower)))


def exact_sum_density(dist: str, spread: float, grid: Any) -> np.ndarray | None:
    r"""两个独立同分布输入之和的**精确**密度（只在能写出闭式时返回，否则 ``None``）。

    * ``uniform``（标准化到标准差 1，即 :math:`U(-\\sqrt3,\\sqrt3)`）：两个之和是三角形密度；
    * ``normal``：:math:`N(0,2\sigma^2)` 的密度。

    其余分布（Beta、指数）没有这个闭式，函数**返回 None**，调用方必须改用别的参考线
    并在 ``strings`` 里说明——"编一条参考线"比"没有参考线"更糟。
    """
    g = np.asarray(grid, dtype=np.float64)
    s = float(spread)
    if dist == "normal":
        sd = math.sqrt(2.0) * s
        return np.exp(-0.5 * (g / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))
    if dist == "uniform":
        a = math.sqrt(3.0) * s
        out = np.zeros_like(g)
        left = (g + 2.0 * a) / (2.0 * a) ** 2
        right = (2.0 * a - g) / (2.0 * a) ** 2
        out = np.where((g > -2.0 * a) & (g <= 0.0), left, out)
        out = np.where((g > 0.0) & (g < 2.0 * a), right, out)
        return out
    return None
