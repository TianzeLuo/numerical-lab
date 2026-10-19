r"""拟蒙特卡洛：随机化 QMC 的误差估计、星偏差、Koksma–Hlawka 界，以及"确定性点集没有误差估计"的实测。

这一层要守住的一句话
--------------------
**MC 的误差估计来自随机性本身；QMC 的误差估计只能来自"人为加回去的随机性"。**

:func:`qmc_error_report` 因此对同一个点集做两种事，并把两者都返回：

* ``randomize=True``：用 R 个独立的随机化（数字移位 / 线性置乱 / 格点随机移位）得到 R 个
  无偏估计，用它们的样本标准差给出误差棒。此时 ``stderr_available=True``。
* ``randomize=False``：R 个"复制"是**同一个点集**，样本标准差**逐位等于 0**，
  而真误差并不为 0。此时 ``stderr_available=False``、``stderr=0``，调用方必须
  把 0 当成"没有这个数"而不是"误差是 0"（帧里对应"占位值 + 掩码 + 文字"三件套）。

这不是文档里的告诫，而是 :func:`deterministic_replicates_are_identical` 里被断言的事实。

偏差与界的口径
--------------
* :func:`star_discrepancy_grid` 在 :math:`G^d` 网格上取 :math:`\\max_t|F_N(t)-\\prod t_j|`。
  它是真 :math:`D_N^*` 的**下界估计**（网格越细越接近），并且只对 :math:`\\prod` 型的
  锚点成立——写进名字里，避免被当成精确值。
* :func:`l2_star_discrepancy` 用 Warnock 公式算 :math:`L_2` 星偏差（这是真值，不是下界，
  代价是 :math:`O(N^2)`）。
* :func:`koksma_hlawka` 给出 :math:`|\\hat I-I|\\le V(f)D_N^*`；:func:`vitali_variation`
  用网格差分估 :math:`V(f)=\\int|\\partial^d f|`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from numcore.inverse.uq import sobol_seq
from numcore.utils import seed_sequence

__all__ = [
    "Integrand",
    "ErrorReport",
    "INTEGRANDS",
    "integrand_names",
    "make_integrand",
    "star_discrepancy_grid",
    "l2_star_discrepancy",
    "vitali_variation",
    "koksma_hlawka",
    "qmc_mean",
    "randomized_estimates",
    "qmc_error_report",
    "deterministic_replicates_are_identical",
    "rate_curve",
    "loglog_slope",
    "batch_means_stderr",
]

#: 网格求偏差时的缺省分辨率（2 的幂：锚点正好落在二进制小数上）
DEFAULT_GRID = 64


# --------------------------------------------------------------------------- #
# 被积函数：每个都有解析积分值（没有真值就没法谈误差）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Integrand:
    """一个可积函数 + 它的**解析**积分值 + 光滑性标签。

    ``smoothness`` 是这一章的关键元数据：QMC 的收敛率优势依赖于函数的光滑性，
    把"间断函数上 QMC 不占优"如实标出来，比统一宣称"QMC 更快"要诚实。
    """

    name: str
    dim: int
    exact: float
    smoothness: str  # "analytic" / "lipschitz" / "discontinuous"
    label: str
    note: str
    func: Callable[[np.ndarray], np.ndarray] = field(repr=False, compare=False)


def _f_smooth(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.sum(x, axis=1))


def _f_tensor(x: np.ndarray) -> np.ndarray:
    return np.prod(1.0 + 0.9 * np.sin(2.0 * np.pi * x), axis=1)


def _f_kink(x: np.ndarray) -> np.ndarray:
    # 折点故意取在 0.3（不是 2 的负幂）：若取 0.5，Sobol' 网的对称性会让这个函数
    # 被**精确**积分（实测误差 0），于是"QMC 在折点上退化"这条结论反而看不见了。
    return np.prod(np.abs(x - 0.3), axis=1)


def _f_discontinuous(x: np.ndarray) -> np.ndarray:
    return (np.all(x < 0.5, axis=1)).astype(np.float64)


def _f_offset_box(x: np.ndarray) -> np.ndarray:
    # 与 discontinuous 只差一个边界位置：0.3 不是 2 的负幂，网的结构不再与它对齐
    return (np.all(x < 0.3, axis=1)).astype(np.float64)


def _exact_reciprocal_sum_2d() -> float:
    r""":math:`\int_{[0,1]^2}\mathrm{d}x\,\mathrm{d}y/(1+x+y)` 的解析值 = :math:`3\ln3-4\ln2`。

    内层 :math:`\int_0^1\mathrm dy/(b+y)=\ln\frac{b+1}{b}`；再用
    :math:`\int_0^1\ln(b+y)\,\mathrm dy=(b+1)\ln(b+1)-b\ln b-1`。
    """
    return 3.0 * math.log(3.0) - 4.0 * math.log(2.0)


_INTEGRAND_SPECS: dict[str, tuple[float, str, str, str, Callable[[np.ndarray], np.ndarray]]] = {
    "smooth": (
        _exact_reciprocal_sum_2d(),
        "analytic",
        "光滑（非张量积）：1 / (1 + sum_j x_j)",
        "解析积分 = 3 ln3 - 4 ln2 = 0.523248；不是张量积，是 QMC 优势的典型场景",
        _f_smooth,
    ),
    "tensor": (
        1.0,
        "analytic",
        "张量积光滑：prod_j (1 + 0.9 sin(2 pi x_j))",
        "解析积分 = 1（每维均值都是 1）；谱光滑且有效维数 = 1，Sobol' 在这里快得离谱",
        _f_tensor,
    ),
    "kink": (
        0.29 * 0.29,
        "lipschitz",
        "折线：prod_j |x_j - 0.3|（折点不在任何 2^-m 节点上）",
        "解析积分 = 0.29^2 = 0.0841；折点使 Sobol' 的斜率退化到 1/2 附近",
        _f_kink,
    ),
    "discontinuous": (
        0.25,
        "discontinuous",
        "间断（边界在 2^-1 上）：1[x_1<0.5 且 x_2<0.5]",
        "解析积分 = 1/4。**陷阱**：0.5 是 2 的负幂，Sobol' 网（即使置乱）对它是**精确**积分——"
        "实测误差恒为 0。这不是「QMC 快得离谱」，而是这个函数的边界恰好落在网的初等区间上；"
        "把边界挪到 0.3（offset_box）就恢复正常。",
        _f_discontinuous,
    ),
    "offset_box": (
        0.09,
        "discontinuous",
        "间断（边界在 0.3，不在 2^-m 网格上）：1[x_1<0.3 且 x_2<0.3]",
        "解析积分 = 0.3^2 = 0.09；与 discontinuous 只差边界位置，但网的精确性消失，"
        "QMC 的斜率回到有限值",
        _f_offset_box,
    ),
}

#: 全部被积函数的名字（帧的参数选项就从这里来）
INTEGRANDS: tuple[str, ...] = tuple(_INTEGRAND_SPECS)


def integrand_names() -> tuple[str, ...]:
    """可用的被积函数名（顺序稳定）。"""
    return INTEGRANDS


def make_integrand(name: str, dim: int = 2) -> Integrand:
    """按名字造一个 :class:`Integrand`（只知道解析值的函数才允许进来）。

    两个可分离的被积函数在 :math:`d` 维下的解析值是单维值的 :math:`d` 次方：
    :math:`\\int_0^1|x-0.3|\\,\\mathrm dx=0.29`（:math:`=a^2/2+(1-a)^2/2`，:math:`a=0.3`），
    :math:`\\int_0^1\\mathbf 1[x<1/2]\\,\\mathrm dx=1/2`。
    """
    key = str(name)
    if key not in _INTEGRAND_SPECS:
        raise KeyError(f"未知被积函数 {name!r}（可用：{sorted(_INTEGRAND_SPECS)}）")
    base_exact, smooth, label, note, fn = _INTEGRAND_SPECS[key]
    exact = base_exact
    if key == "kink":
        exact = 0.29**dim
    if key == "discontinuous":
        exact = 0.5**dim
    if key == "offset_box":
        exact = 0.3**dim
    return Integrand(name=key, dim=int(dim), exact=float(exact), smoothness=smooth, label=label, note=note, func=fn)


# --------------------------------------------------------------------------- #
# 偏差
# --------------------------------------------------------------------------- #
def star_discrepancy_grid(points: Any, grid: int = DEFAULT_GRID) -> float:
    r"""网格估计的星偏差 :math:`\hat D_N^*=\max_t|F_N(t)-\prod_j t_j|`（``t`` 取 :math:`G^d` 网格）。

    **这是下界**：真 :math:`D_N^*` 的最大值可能落在网格之间。函数名与 ``strings`` 里都写明
    "grid"，就是为了不让这个数被当成精确偏差。取值点还包括 :math:`t_j=1`（此时
    :math:`|1-1|=0`，无贡献）与网格的每个角点。
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("points 必须是 (n, dim)")
    n, d = p.shape
    g = int(grid)
    if g < 2:
        raise ValueError("grid 必须 >= 2")
    idx = np.minimum((p * g).astype(np.int64), g - 1)
    counts = np.zeros((g,) * d, dtype=np.float64)
    np.add.at(counts, tuple(idx[:, j] for j in range(d)), 1.0)
    cum = counts
    for ax in range(d):
        cum = np.cumsum(cum, axis=ax)
    fn = cum / float(n)
    axis = (np.arange(1, g + 1, dtype=np.float64) / float(g))
    if d == 1:
        prod = axis
    else:
        mesh = np.meshgrid(*([axis] * d), indexing="ij")
        prod = np.ones_like(mesh[0])
        for m in mesh:
            prod = prod * m
    return float(np.max(np.abs(fn - prod)))


def l2_star_discrepancy(points: Any, *, chunk: int = 512) -> float:
    r"""Warnock 公式给出的 :math:`L_2` 星偏差平方：

    .. math::
        D_{N,2}^{*2}=\frac1{3^d}-\frac{2^{1-d}}{N}\sum_i\prod_j(1-x_{ij}^2)
        +\frac1{N^2}\sum_{i,k}\prod_j\bigl(1-\max(x_{ij},x_{kj})\bigr).

    与网格版不同，这是**真值**（对 :math:`L_2` 范数而言），代价是 :math:`O(N^2)`；
    分块计算以免 :math:`N=4096` 时一次开出 134 MB 的中间矩阵。
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("points 必须是 (n, dim)")
    n, d = p.shape
    term1 = 3.0 ** (-d)
    term2 = (2.0 ** (1 - d) / n) * float(np.sum(np.prod(1.0 - p * p, axis=1)))
    total = 0.0
    for start in range(0, n, chunk):
        block = p[start : start + chunk]
        mx = np.maximum(block[:, None, :], p[None, :, :])  # (b, n, d)
        total += float(np.sum(np.prod(1.0 - mx, axis=2)))
    term3 = total / (n * n)
    return float(max(0.0, term1 - term2 + term3))


def vitali_variation(func: Callable[[np.ndarray], np.ndarray], dim: int = 2, *, grid: int = 400) -> float:
    r"""Vitali 意义下的变分 :math:`V(f)=\int_{[0,1]^d}|\partial^d f/\partial x_1\cdots\partial x_d|`（网格差分估计）。

    一维时就是 :math:`\int|f'|`（全变差）；二维时用混合二阶差分。这是**估计**：
    网格细化会单调逼近真值，但折点落在网格上时会偏低。
    """
    g = int(grid)
    axis = (np.arange(g, dtype=np.float64) + 0.5) / g
    mesh = np.meshgrid(*([axis] * int(dim)), indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    vals = np.asarray(func(pts), dtype=np.float64).reshape((g,) * int(dim))
    h = 1.0 / g
    acc = vals
    for ax in range(int(dim)):
        acc = np.diff(acc, axis=ax) / h
    return float(np.mean(np.abs(acc)))


def koksma_hlawka(variation: float, discrepancy: float) -> float:
    r"""Koksma–Hlawka 界 :math:`|\hat I-I|\le V(f)\,D_N^*`（在本仓里它是一个**可检验的上界**）。"""
    return float(abs(variation) * abs(discrepancy))


# --------------------------------------------------------------------------- #
# 估计与误差
# --------------------------------------------------------------------------- #
def qmc_mean(values: Any) -> float:
    """等权平均（QMC 与 MC 的估计量在形式上完全一样，区别只在点集）。"""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 1:
        raise ValueError("至少需要一个函数值")
    return float(np.mean(v))


def batch_means_stderr(values: Any, n_batches: int = 16) -> float:
    r"""批量均值的标准误：把序列切成 ``n_batches`` 段，段均值的标准差除以 :math:`\sqrt{B}`。

    这是"不假装样本独立"的误差估计：QMC 的样本**本来就不独立**，
    用 :math:`s/\sqrt N` 会系统性低估（``tests/5.4`` 里有实测的覆盖率对照）。
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    b = int(n_batches)
    if b < 2 or v.size < 2 * b:
        raise ValueError("需要至少 2*b 个样本")
    m = (v.size // b) * b
    chunks = v[:m].reshape(b, -1).mean(axis=1)
    return float(np.std(chunks, ddof=1) / math.sqrt(b))


def randomized_estimates(
    func: Callable[[np.ndarray], np.ndarray],
    n: int,
    dim: int,
    *,
    method: str = "sobol",
    replicates: int = 32,
    seed: Any = None,
    randomize: bool = True,
) -> np.ndarray:
    """R 个独立随机化下的估计量（``randomize=False`` 时 R 个估计**逐位相同**）。

    随机化的种子由 :func:`numcore.utils.rng` 的父种子派生，保证可复现。
    """
    r = int(replicates)
    if r < 2:
        raise ValueError("replicates 必须 >= 2")
    parent = 20261018 if seed is None else seed
    subs = seed_sequence(int(parent), r)
    out = np.empty(r, dtype=np.float64)
    for i in range(r):
        pts = sobol_seq.point_set(method, n, dim, seed=int(subs[i]), randomize=randomize)
        out[i] = qmc_mean(func(pts))
    return out


@dataclass(frozen=True)
class ErrorReport:
    """一次 QMC/MC 实验的全部可读量（三个"误差"必须分开，混起来就是这一章最大的骗局）。

    ==========================  ==================================================================
    ``abs_error``               这一次**均值**离真值多远（要真值才算得出，图上画不出来）
    ``rmse_single``             单次随机化的均方根误差 :math:`\\sqrt{\\overline{(\\hat I_r-I)^2}}`
                                —— "误差-N 曲线"的纵轴就是它，也是 :math:`1/\\sqrt N` 对标量的那一个
    ``stderr_of_mean``          **算法自己**报出来的误差棒（均值标准误 :math:`s/\\sqrt R`）
    ==========================  ==================================================================

    ``stderr_available`` 是核心诚实字段：``randomize=False`` 时它必须是 ``False``，
    此时 ``stderr_of_mean = 0`` 只表示"没有这个数"，不表示"误差是 0"。帧里据此写掩码。
    """

    method: str
    n: int
    dim: int
    replicates: int
    randomize: bool
    mean: float
    true_value: float
    abs_error: float
    stderr_of_mean: float
    stderr_available: bool
    rmse_single: float
    ci95: tuple[float, float]

    @property
    def covered(self) -> bool:
        """真值是否落在复制给出的 95% 区间内（只有 ``stderr_available`` 时才有意义）。"""
        return bool(self.ci95[0] <= self.true_value <= self.ci95[1])

    @property
    def bar_honesty(self) -> float:
        r"""误差棒 / 实际离散度 = :math:`(s/\sqrt R)/\mathrm{RMSE}_{\\rm single}`。

        诚实时它应当 ≈ :math:`1/\sqrt R`（因为两者差一个 :math:`\sqrt R`）；
        明显偏离就说明复制之间不独立或者估计量有偏。无误差棒时返回 ``nan``。
        """
        if not self.stderr_available or self.rmse_single <= 0.0:
            return float("nan")
        return float(self.stderr_of_mean / self.rmse_single)


def qmc_error_report(
    func: Callable[[np.ndarray], np.ndarray],
    n: int,
    dim: int,
    *,
    true_value: float,
    method: str = "sobol",
    replicates: int = 32,
    seed: Any = None,
    randomize: bool = True,
) -> ErrorReport:
    """跑 R 个复制，把"真误差 / 单次 RMSE / 自报误差棒"三个量一次算清。"""
    est = randomized_estimates(
        func, n, dim, method=method, replicates=replicates, seed=seed, randomize=randomize
    )
    mean = float(np.mean(est))
    sd = float(np.std(est, ddof=1))
    half = 1.959963984540054 * sd / math.sqrt(est.size)
    avail = bool(randomize and sd > 0.0)
    return ErrorReport(
        method=str(method),
        n=int(n),
        dim=int(dim),
        replicates=int(est.size),
        randomize=bool(randomize),
        mean=mean,
        true_value=float(true_value),
        abs_error=abs(mean - float(true_value)),
        stderr_of_mean=float(sd / math.sqrt(est.size)),
        stderr_available=avail,
        rmse_single=float(np.sqrt(np.mean((est - float(true_value)) ** 2))),
        ci95=(mean - half, mean + half),
    )


def deterministic_replicates_are_identical(
    func: Callable[[np.ndarray], np.ndarray],
    n: int,
    dim: int,
    *,
    true_value: float,
    method: str = "sobol",
) -> tuple[float, float]:
    """返回 ``(复制的样本标准差, 真实误差)``：前者**恒为 0**，后者一般不为 0。

    这一对数是本章"不置乱就没有误差估计"的全部证据：把 0 当成"误差很小"是错的，
    它是"这个量不存在"。张量格点与未置乱的 Sobol'/格点都属于这一类。
    """
    est = randomized_estimates(func, n, dim, method=method, replicates=4, seed=7, randomize=False)
    sd = float(np.std(est, ddof=1))
    return sd, abs(float(est[0]) - float(true_value))


def loglog_slope(x: Any, y: Any, *, min_points: int = 3, head: int | None = None) -> float:
    r"""双对数最小二乘斜率：``y ~ C x^p`` 的 ``p``（``x`` 必须递增、``y`` 必须为正）。

    与 :func:`numcore.verify.fit_order` 同口径（先取对数再做一元最小二乘），
    但**不 import 它**（内核不依赖裁判）；``tests/5.4`` 拿两边逐位对拍。
    ``head`` 只保留最前面 ``head`` 个点——对付"触到渐近地板后斜率被拉平"的正确工具。
    """
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    if xx.size != yy.size:
        raise ValueError("x 与 y 长度必须相同")
    if head is not None:
        xx, yy = xx[: int(head)], yy[: int(head)]
    ok = (xx > 0.0) & (yy > 0.0) & np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[ok], yy[ok]
    if xx.size < max(2, int(min_points)):
        raise ValueError(f"有效点不足：{xx.size} < {max(2, int(min_points))}")
    lx = np.log(xx)
    ly = np.log(yy)
    mx = float(np.mean(lx))
    my = float(np.mean(ly))
    denom = float(np.sum((lx - mx) ** 2))
    if denom == 0.0:
        raise ValueError("x 全相同，斜率无定义")
    return float(np.sum((lx - mx) * (ly - my)) / denom)


def rate_curve(
    func: Callable[[np.ndarray], np.ndarray],
    n_list: Sequence[int],
    dim: int,
    *,
    true_value: float,
    method: str = "sobol",
    replicates: int = 32,
    seed: Any = None,
    randomize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """误差-工作量曲线：返回 ``(n, rmse_single, stderr_of_mean)``。

    ``rmse_single`` 是**单次随机化**的均方根误差（每个 N 都用同一套 R 个派生种子，
    所以不同 N 之间的曲线是可比的、可复现的）；``randomize=False`` 时它退化为
    "这一条确定性点集的绝对误差"（不是误差棒，只是历史），而 ``stderr_of_mean`` 为 0。
    """
    ns = np.asarray(list(n_list), dtype=np.int64)
    rmse = np.empty(ns.size, dtype=np.float64)
    se = np.empty(ns.size, dtype=np.float64)
    for i, n in enumerate(ns):
        rep = qmc_error_report(
            func,
            int(n),
            dim,
            true_value=true_value,
            method=method,
            replicates=replicates,
            seed=seed,
            randomize=randomize,
        )
        rmse[i] = rep.rmse_single if randomize else rep.abs_error
        se[i] = rep.stderr_of_mean if randomize else 0.0
    return ns, rmse, se
