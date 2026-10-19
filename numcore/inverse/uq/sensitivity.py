r"""全局敏感性：Sobol' 指数、它的估计器、它的置信区间，以及**它什么时候不成立**。

三条纪律（这一章最容易被讲错的地方）
------------------------------------
1. **:math:`\sum_i S_i\le 1\le\sum_i S_{Ti}` 是判据，不是恒等式。**
   等式只在"输入独立 + 估计器无偏 + 样本无穷"三者同时成立时趋近。
   本模块 :class:`SobolIndices` 同时给出和、bootstrap 置信区间与**估计器漂移**
   （:func:`estimator_drift`：同一个函数在 :math:`N=2^6\dots2^{13}` 上的估计值并不相同）。
2. **相关输入下方差分解不唯一。** 独立性是 Sobol' 分解的**前提**，不是技术细节。
   :func:`structural_indices` 给"条件方差"口径（:math:`\mathrm{Var}(E[Y|X_i])/V`），
   :func:`shuffled_indices` 给"打散相关性后"的口径；两者在 :math:`\rho=0.5` 的双线性模型上
   相差 0.25，而前者之和可以到 1.5（:func:`overlap_excess` 就是这个超出量）。
3. **估计器不止一个。** Saltelli(2010) 与 Jansen(1999) 的一阶/总效应公式在小样本下的偏差
   方向不同（:class:`SobolIndices` 两套都给），"我算出来的 S_i"必须先说清是哪一个。

实现约束：相关样本的 Cholesky 走 :func:`numcore.linalg.chol.cholesky`（P2/P3：不许
``numpy.linalg``）；随机数一律 :func:`numcore.utils.rng`（P4）；本模块无 IO、无全局状态。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from numcore.linalg.chol import cholesky
from numcore.utils import rng as make_rng, seed_sequence

__all__ = [
    "Model",
    "model_names",
    "make_model",
    "SobolIndices",
    "pick_freeze_indices",
    "estimator_drift",
    "structural_indices",
    "StructuralIndices",
    "shuffled_indices",
    "correlated_normal_samples",
    "equicorrelation_matrix",
    "bootstrap_ci",
    "ishigami_analytic",
    "sobol_g_analytic",
]


# --------------------------------------------------------------------------- #
# 模型（每个都带解析指数，否则"测出来的 S_i 对不对"无从谈起）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Model:
    """一个测试模型：样本域、维数、函数，以及（可选的）解析 Sobol' 指数。"""

    name: str
    dim: int
    label: str
    note: str
    func: Callable[[np.ndarray], np.ndarray]
    first: tuple[float, ...] = ()
    total: tuple[float, ...] = ()
    variance: float = float("nan")
    interaction_share: float = 0.0
    domain: str = "unit"

    @property
    def has_analytic(self) -> bool:
        """解析指数是否可用（不可用时任何"对拍"都不许做）。"""
        return len(self.first) == self.dim and len(self.total) == self.dim


def ishigami_analytic(a: float = 7.0, b: float = 0.1) -> dict[str, Any]:
    r""":math:`f=\sin x_1+a\sin^2x_2+bx_3^4\sin x_1`（:math:`x_i\sim U(-\pi,\pi)`）的解析分解。

    .. math::
        V=\frac{a^2}8+\frac{b\pi^4}5+\frac{b^2\pi^8}{18}+\frac12,\quad
        V_1=\frac12+\frac{b\pi^4}5+\frac{b^2\pi^8}{50},\quad V_2=\frac{a^2}8,\quad
        V_{13}=b^2\pi^8\Bigl(\frac1{18}-\frac1{50}\Bigr),

    :math:`S_1=V_1/V,\ S_2=V_2/V,\ S_3=0,\ S_{13}=V_{13}/V`，且
    :math:`S_{T1}=S_1+S_{13},\ S_{T3}=S_{13}`。三者和恰为 1（Ishigami 没有更高阶交互）。
    """
    pi4 = math.pi**4
    pi8 = math.pi**8
    var = a * a / 8.0 + b * pi4 / 5.0 + b * b * pi8 / 18.0 + 0.5
    v1 = 0.5 + b * pi4 / 5.0 + b * b * pi8 / 50.0
    v2 = a * a / 8.0
    v13 = b * b * pi8 * (1.0 / 18.0 - 1.0 / 50.0)
    s1, s2, s13 = v1 / var, v2 / var, v13 / var
    return {
        "variance": var,
        "first": (s1, s2, 0.0),
        "total": (s1 + s13, s2, s13),
        "closed": {"V1": v1, "V2": v2, "V13": v13},
        "interaction_share": s13,
    }


def sobol_g_analytic(a: Sequence[float]) -> dict[str, Any]:
    r"""Sobol' g-函数 :math:`g=\prod_j\frac{|4x_j-2|+a_j}{1+a_j}`（:math:`x_j\sim U(0,1)`）的解析分解。

    :math:`\sigma_j^2=\frac{1/3}{(1+a_j)^2}`，
    :math:`V=\prod_j(1+\sigma_j^2)-1`，:math:`V_j=\sigma_j^2`（因为 :math:`E[g_j]=1`，
    所以 :math:`\mathrm{Var}(E[Y|x_j])=\mathrm{Var}(g_j)`），
    :math:`V_{Tj}=V-\bigl[\prod_{k\ne j}(1+\sigma_k^2)-1\bigr]`。

    **差点写成假证据的地方**：本模块第一版把一阶方差写成
    :math:`V_j=\sigma_j^2\prod_{k\ne j}(1+\sigma_k^2)`（乘积形式看起来很"对称"），
    于是 :math:`a=(0,1,4.5)` 给出 :math:`\sum_jS_j=1.0716>1`——**一阶指数之和大于 1
    是数学上不可能的**，而这个错值当时"看起来"完全正常。正确的写法是
    :math:`V_j=\sigma_j^2`：乘积形式属于**未中心化**的 g-函数约定，
    在 :math:`E[g_j]=1` 的约定下会多乘一个 :math:`\prod_{k\ne j}(1+\sigma_k^2)`。
    :math:`a=(0,1,4.5)` 时 :math:`\sum_jS_j\approx0.93<1`——**这就是"和小于 1"的来源**：
    4.5% 的方差藏在二阶及以上的交互里。
    """
    aa = np.asarray(list(a), dtype=np.float64)
    sig2 = (1.0 / 3.0) / (1.0 + aa) ** 2
    full = float(np.prod(1.0 + sig2))
    var = full - 1.0
    first = []
    total = []
    for j in range(aa.size):
        rest = float(np.prod(np.delete(1.0 + sig2, j)))
        first.append(sig2[j] / var)
        total.append(1.0 - (rest - 1.0) / var)
    return {
        "variance": var,
        "first": tuple(first),
        "total": tuple(total),
        "sigma2": tuple(float(x) for x in sig2),
        "interaction_share": float(sum(total) - sum(first)),
    }


def _fn_ishigami(x: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def _fn_g(x: np.ndarray, aa: tuple[float, ...] = (0.0, 1.0, 4.5)) -> np.ndarray:
    a = np.asarray(aa, dtype=np.float64)[None, :]
    return np.prod((np.abs(4.0 * x - 2.0) + a) / (1.0 + a), axis=1)


def _fn_bilinear(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + x[:, 1]


def _fn_correlated_quadratic(x: np.ndarray) -> np.ndarray:
    return x[:, 0] ** 2 + x[:, 1]


_MODELS: dict[str, Model] = {}


def _register(m: Model) -> None:
    _MODELS[m.name] = m


_ish = ishigami_analytic()
_register(
    Model(
        "ishigami",
        3,
        "Ishigami：sin x1 + 7 sin^2 x2 + 0.1 x3^4 sin x1",
        "x ~ U(-pi, pi)^3；解析 S = (0.3139, 0.4424, 0.0)，只有一阶与 S13，和恰为 1",
        _fn_ishigami,
        first=_ish["first"],
        total=_ish["total"],
        variance=_ish["variance"],
        interaction_share=_ish["interaction_share"],
        domain="uniform(-pi,pi)",
    )
)
_g = sobol_g_analytic((0.0, 1.0, 4.5))
_register(
    Model(
        "g_function",
        3,
        "Sobol g-函数（a = 0, 1, 4.5）",
        "x ~ U(0,1)^3；解析 sum S_i = 0.9273 < 1，4.5% 的方差在交互项里",
        _fn_g,
        first=_g["first"],
        total=_g["total"],
        variance=_g["variance"],
        interaction_share=_g["interaction_share"],
        domain="uniform(0,1)",
    )
)
_register(
    Model(
        "bilinear",
        2,
        "双线性：x1 + x2（相关输入用）",
        "独立时 S = (0.5, 0.5)；rho = 0.5 时条件方差口径给出 (0.75, 0.75)，和 1.5 > 1",
        _fn_bilinear,
        first=(0.5, 0.5),
        total=(0.5, 0.5),
        variance=2.0,
        interaction_share=0.0,
        domain="normal(0,1)",
    )
)
_register(
    Model(
        "quadratic_corr",
        2,
        "相关二次：x1^2 + x2",
        "x ~ N(0, Sigma)，rho 可拖；非线性 + 相关，两种分解的差随 rho 增大",
        _fn_correlated_quadratic,
        domain="normal(0,1)",
    )
)


def model_names() -> tuple[str, ...]:
    """可用模型名（顺序稳定）。"""
    return tuple(_MODELS)


def make_model(name: str) -> Model:
    """按名字取模型。"""
    key = str(name)
    if key not in _MODELS:
        raise KeyError(f"未知模型 {name!r}（可用：{sorted(_MODELS)}）")
    return _MODELS[key]


# --------------------------------------------------------------------------- #
# pick-freeze：估计量与置信区间
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SobolIndices:
    """一次 pick-freeze 实验的全部可读量（两套估计器 + bootstrap 区间）。"""

    model: str
    n: int
    dim: int
    variance: float
    first: np.ndarray
    first_alt: np.ndarray
    total: np.ndarray
    total_alt: np.ndarray
    first_halfwidth: np.ndarray
    total_halfwidth: np.ndarray
    n_evals: int
    rho: float = 0.0

    @property
    def sum_first(self) -> float:
        """一阶指数之和（独立输入下应当 ≤ 1；相关输入下**可以**大于 1）。"""
        return float(np.sum(self.first))

    @property
    def sum_total(self) -> float:
        """总效应指数之和（独立输入下应当 ≥ 1）。"""
        return float(np.sum(self.total))

    @property
    def interaction_share(self) -> float:
        r"""交互份额 :math:`\sum_iS_{Ti}-\sum_iS_i`（独立输入下 ∈ [0, 1]）。"""
        return float(np.sum(self.total) - np.sum(self.first))

    @property
    def sum_first_halfwidth(self) -> float:
        """一阶之和的 bootstrap 半宽（各项独立 bootstrap 的半宽按平方和合成）。"""
        return float(math.sqrt(float(np.sum(self.first_halfwidth**2))))

    @property
    def sum_total_halfwidth(self) -> float:
        """总效应之和的 bootstrap 半宽。"""
        return float(math.sqrt(float(np.sum(self.total_halfwidth**2))))


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], np.ndarray] | None = None,
    replicates: int = 200,
    level: float = 0.95,
    seed: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""bootstrap 分位区间：对 ``values`` 的**行**重采样，返回 ``(下界, 上界)``。

    ``values`` 形状 ``(n, k)``（k 个要同时估计的量），``statistic`` 缺省取按行均值。
    行重采样是必须的：一阶与总效应的估计量共用同一批 :math:`f(A),f(B),f(AB_i)`，
    逐列独立重采样会**低估**区间宽度（把相关性当成噪声丢掉了）。
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = arr.shape[0]
    b = int(replicates)
    if n < 2 or b < 10:
        raise ValueError("至少 2 个样本、10 次 bootstrap")
    g = make_rng(seed)
    idx = g.integers(0, n, size=(b, n))
    draws = arr[idx]  # (b, n, k)
    stat = np.mean if statistic is None else statistic
    reps = stat(draws, axis=1) if statistic is None else stat(draws)
    reps = np.asarray(reps, dtype=np.float64)
    if reps.ndim == 1:
        reps = reps[:, None]
    lo = np.percentile(reps, 100.0 * (1.0 - level) / 2.0, axis=0)
    hi = np.percentile(reps, 100.0 * (1.0 + level) / 2.0, axis=0)
    return lo, hi


def _pick_freeze_core(
    func: Callable[[np.ndarray], np.ndarray], a: np.ndarray, b: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """核心估计：返回 ``(V, S_i, S_i_alt, S_Ti, S_Ti_alt)``。"""
    n, d = a.shape
    fa = np.asarray(func(a), dtype=np.float64).ravel()
    fb = np.asarray(func(b), dtype=np.float64).ravel()
    pooled = np.concatenate([fa, fb])
    var = float(np.var(pooled, ddof=1))
    first = np.empty(d, dtype=np.float64)
    first_alt = np.empty(d, dtype=np.float64)
    total = np.empty(d, dtype=np.float64)
    total_alt = np.empty(d, dtype=np.float64)
    for i in range(d):
        ab = a.copy()
        ab[:, i] = b[:, i]
        fab = np.asarray(func(ab), dtype=np.float64).ravel()
        # Saltelli(2010) 一阶：mean(f_B (f_AB - f_A)) / V
        first[i] = float(np.mean(fb * (fab - fa))) / var
        # Sobol'(1993)/Jansen 一阶：mean(f_A (f_AB - f_B)) / V
        first_alt[i] = float(np.mean(fa * (fab - fb))) / var
        # Jansen(1999) 总效应：mean((f_A - f_AB)^2) / (2V)
        total[i] = float(np.mean((fa - fab) ** 2)) / (2.0 * var)
        # Saltelli(2010) 总效应：1 - mean(f_B (f_AB - f_A)) / V ... 用对称形式以免 1-S 放大小量
        total_alt[i] = float(np.mean((fb - fab) ** 2)) / (2.0 * var)
    return var, first, first_alt, total, total_alt


def _draw_domain(m: Model, n: int, d: int, s1: int, s2: int, scale: float) -> np.ndarray:
    r"""按模型**自己的输入测度**抽 ``n x d`` 个样本。

    这一步看起来只是"换个分布抽数"，其实是整段估计的立足点：pick-freeze 估计的是
    模型在**给定输入测度**下的方差分解，把 :math:`N(0,1)` 的输入用 :math:`U(0,1)`
    抽出来，算出的数就不是那个模型的指数了（本模块第一版如此：双线性模型的方差
    实测 0.1677，而它在正态输入下的真值是 2——数字小得"挺合理"，所以很容易漏掉）。
    """
    u1 = make_rng(int(s1)).random((int(n), int(d)))
    if m.domain.startswith("uniform(-pi,pi)"):
        return (2.0 * u1 - 1.0) * math.pi * float(scale)
    if m.domain.startswith("normal"):
        u2 = make_rng(int(s2)).random((int(n), int(d)))
        r = np.sqrt(-2.0 * np.log(np.maximum(u1, np.finfo(float).tiny)))
        return float(scale) * r * np.cos(2.0 * np.pi * u2)
    return u1 * float(scale)


def pick_freeze_indices(
    model: str | Model,
    n: int,
    *,
    dim: int | None = None,
    seed: Any = None,
    bootstrap: int = 200,
    a_scale: float = 1.0,
) -> SobolIndices:
    r"""pick-freeze（Saltelli 2010 / Jansen 1999）估计 Sobol' 指数，并给 bootstrap 区间。

    需要 :math:`n(d+2)` 次模型求值（:math:`A,B` 各一次 + 每个维度的 :math:`AB_i`），
    这是这一章"计算量随维数线性增长"的根据。``a_scale`` 把样本域从单位超立方体
    线性搬到模型自己的域（Ishigami 用 :math:`U(-\pi,\pi)`）。
    """
    m = make_model(model) if isinstance(model, str) else model
    d = m.dim if dim is None else int(dim)
    nn = int(n)
    parent = 20261018 if seed is None else seed
    subs = seed_sequence(int(parent), 4 + d)
    a = _draw_domain(m, nn, d, int(subs[0]), int(subs[1]), float(a_scale))
    b = _draw_domain(m, nn, d, int(subs[2]), int(subs[3]), float(a_scale))

    var, first, first_alt, total, total_alt = _pick_freeze_core(m.func, a, b)

    # bootstrap：对行重采样，两个估计器共用同一批重采样索引
    reps_first = np.empty((int(bootstrap), d), dtype=np.float64)
    reps_total = np.empty((int(bootstrap), d), dtype=np.float64)
    fa = np.asarray(m.func(a), dtype=np.float64).ravel()
    fb = np.asarray(m.func(b), dtype=np.float64).ravel()
    fabs_ = []
    for i in range(d):
        ab = a.copy()
        ab[:, i] = b[:, i]
        fabs_.append(np.asarray(m.func(ab), dtype=np.float64).ravel())
    gb = make_rng(int(subs[0]) + 99991)
    for r in range(int(bootstrap)):
        idx = gb.integers(0, nn, size=nn)
        fai, fbi = fa[idx], fb[idx]
        pooled = np.concatenate([fai, fbi])
        v = float(np.var(pooled, ddof=1))
        for i in range(d):
            fab = fabs_[i][idx]
            reps_first[r, i] = float(np.mean(fbi * (fab - fai))) / v
            reps_total[r, i] = float(np.mean((fai - fab) ** 2)) / (2.0 * v)
    fh = np.percentile(reps_first, 97.5, axis=0) - np.percentile(reps_first, 2.5, axis=0)
    th = np.percentile(reps_total, 97.5, axis=0) - np.percentile(reps_total, 2.5, axis=0)
    return SobolIndices(
        model=m.name,
        n=nn,
        dim=d,
        variance=var,
        first=first,
        first_alt=first_alt,
        total=total,
        total_alt=total_alt,
        first_halfwidth=fh / 2.0,
        total_halfwidth=th / 2.0,
        n_evals=nn * (d + 2),
    )


def estimator_drift(
    model: str | Model,
    n_list: Sequence[int],
    *,
    seed: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """估计器漂移：同一模型、同一随机流下，``S_i`` 与 ``S_{Ti}`` 随 ``N`` 的实测变化。

    做法是"同一批样本的前 N 行"——这恰好是 pick-freeze 在小 N 下的**偏差**（不是噪声）：
    估计量 :math:`\\frac1n\\sum f_B(f_{AB}-f_A)` 的期望并不等于 :math:`S_iV`，
    偏差是 :math:`O(1/n)` 量级。返回 ``(n_list, first_mean, total_mean)``：
    每行是"各维一阶指数"在该 N 上的取值。
    """
    m = make_model(model) if isinstance(model, str) else model
    d = m.dim
    nmax = int(max(n_list))
    parent = 20261018 if seed is None else seed
    subs = seed_sequence(int(parent), 2 + d)
    a_raw = make_rng(int(subs[0])).random((nmax, d))
    b_raw = make_rng(int(subs[1])).random((nmax, d))
    if m.domain.startswith("uniform(-pi,pi)"):
        a = (2.0 * a_raw - 1.0) * math.pi
        b = (2.0 * b_raw - 1.0) * math.pi
    else:
        a, b = a_raw, b_raw
    ns = np.asarray(list(n_list), dtype=np.int64)
    first = np.empty((ns.size, d), dtype=np.float64)
    total = np.empty((ns.size, d), dtype=np.float64)
    for j, n in enumerate(ns):
        var, f1, _, t1, _ = _pick_freeze_core(m.func, a[: int(n)], b[: int(n)])
        first[j] = f1
        total[j] = t1
    return ns, first, total


# --------------------------------------------------------------------------- #
# 相关输入：两种分解
# --------------------------------------------------------------------------- #
def equicorrelation_matrix(dim: int, rho: float) -> np.ndarray:
    """:math:`\\Sigma=(1-\\rho)I+\\rho\\mathbf 1\\mathbf 1^T`（等相关系数矩阵，需 :math:`\\rho>-1/(d-1)`）。"""
    d = int(dim)
    r = float(rho)
    if d < 1 or r <= -1.0 / max(1, d - 1) or r >= 1.0:
        raise ValueError(f"rho={rho} 超出 ({-1.0 / max(1, d - 1):.4f}, 1) 的可分解范围")
    return (1.0 - r) * np.eye(d) + r * np.ones((d, d))


def correlated_normal_samples(
    rho: float, n: int, dim: int = 2, *, seed: Any = None, mean: float = 0.0
) -> np.ndarray:
    r"""等相关正态样本 :math:`X\\sim N(0,\\Sigma)`：用 :func:`numcore.linalg.chol.cholesky` 的因子。

    自己写 Cholesky 是 P2/P3 的要求，但这里同时是**数学上必须的**：
    :math:`X=LZ` 把独立标准正态变成相关正态，正是"相关性从哪来"的答案。
    """
    d = int(dim)
    s = equicorrelation_matrix(d, rho)
    res = cholesky(s)
    if not res.ok:
        raise ValueError(f"相关系数矩阵在 rho={rho} 处不是正定（Cholesky 失败）")
    g = make_rng(seed)
    z = np.empty((int(n), d), dtype=np.float64)
    for j in range(d):
        u1 = g.random(int(n))
        u2 = g.random(int(n))
        r = np.sqrt(-2.0 * np.log(np.maximum(u1, np.finfo(float).tiny)))
        z[:, j] = r * np.cos(2.0 * np.pi * u2)
    return float(mean) + z @ res.L.T


@dataclass(frozen=True)
class StructuralIndices:
    """"条件方差"口径的分解（相关输入下这才是良定义的量）。

    ``between[i] = Var(E[Y|X_i])/V``、``within[i] = E[Var(Y|X_i)]/V``，逐维
    ``between + within = 1``（全方差公式），但**跨维** :math:`\\sum_i between_i`
    可以大于 1——这就是"方差分解不唯一"的算术根源。
    """

    dim: int
    variance: float
    between: np.ndarray
    within: np.ndarray
    n_bins: int
    rho: float

    @property
    def overlap_excess(self) -> float:
        r""":math:`\sum_i \\mathrm{Var}(E[Y|X_i])/V - 1`：相关输入下"被重复计算"的方差份额。"""
        return float(np.sum(self.between) - 1.0)


def structural_indices(
    x: np.ndarray, y: np.ndarray, *, n_bins: int = 32, rho: float = float("nan")
) -> StructuralIndices:
    r"""给定数据的条件方差分解（given-data / "binned" 估计）。

    把 :math:`X_i` 排序后切成 ``n_bins`` 个等频箱，箱内均值即 :math:`\hat E[Y|X_i\in B]`，
    于是 :math:`\widehat{\mathrm{Var}}(E[Y|X_i])` 是箱间加权方差，
    :math:`\widehat{E}[\mathrm{Var}(Y|X_i)]` 是箱内方差按样本数加权。
    bin 数是一个**人为选择**：它同时控制偏差（箱太宽）与噪声（箱太窄），
    所以这个估计器在报告里必须带上 ``n_bins``。
    """
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64).ravel()
    if xx.ndim != 2 or xx.shape[0] != yy.size:
        raise ValueError("需要 (n, d) 的输入与长度 n 的输出")
    n, d = xx.shape
    b = int(n_bins)
    if b < 2 or n < 2 * b:
        raise ValueError("n_bins 至少 2，且 n >= 2*n_bins")
    var = float(np.var(yy, ddof=1))
    between = np.empty(d, dtype=np.float64)
    within = np.empty(d, dtype=np.float64)
    for i in range(d):
        order = np.argsort(xx[:, i], kind="stable")
        ys = yy[order]
        edges = np.linspace(0, n, b + 1).astype(np.int64)
        m_within = 0.0
        m_between = 0.0
        for j in range(b):
            seg = ys[edges[j] : edges[j + 1]]
            if seg.size == 0:
                continue
            w = seg.size / float(n)
            mu = float(np.mean(seg))
            m_between += w * (mu - float(np.mean(yy))) ** 2
            m_within += w * float(np.var(seg, ddof=0))
        between[i] = m_between / var if var > 0 else 0.0
        within[i] = m_within / var if var > 0 else 0.0
    return StructuralIndices(dim=d, variance=var, between=between, within=within, n_bins=b, rho=float(rho))


def shuffled_indices(model: str | Model, x: np.ndarray, *, seed: Any = None) -> SobolIndices:
    r"""把相关样本"逐列独立打散"后再跑一次 pick-freeze：得到**独立性假设下**的指数。

    打散（每一维独立置换）保持边缘分布、破坏维间相关，于是得到的是**乘积测度**下的样本。
    两个样本 :math:`A^{ind},B^{ind}` 都取自乘积测度，估计器与分母 :math:`V` 也随之落在
    乘积测度上——这才是"Sobol' 分解本来要求的那个问题"。它回答"如果输入本来独立，
    一阶指数是多少"；:func:`structural_indices` 回答"在现在的相关结构下，:math:`X_i`
    单独解释了多少方差"。两者在文献里都被叫做"一阶 Sobol' 指数"——这就是不唯一性的形态。

    **差点写成假证据的地方**：第一版只打散了一侧（A 用相关样本、B 用打散样本），
    分母于是混用了两个测度的方差，双线性模型在 rho=0.5 上给出 0.398 而不是 0.5。
    这个数"看起来像 0.5 附近"，但它既不是相关口径也不是独立口径——两头不靠。
    """
    m = make_model(model) if isinstance(model, str) else model
    xx = np.asarray(x, dtype=np.float64)
    n, d = xx.shape
    g = make_rng(20261018 if seed is None else seed)
    a_ind = np.empty_like(xx)
    b_ind = np.empty_like(xx)
    for j in range(d):
        a_ind[:, j] = xx[g.permutation(n), j]
        b_ind[:, j] = xx[g.permutation(n), j]
    var, first, first_alt, total, total_alt = _pick_freeze_core(m.func, a_ind, b_ind)
    zeros = np.zeros(d, dtype=np.float64)
    return SobolIndices(
        model=m.name,
        n=n,
        dim=d,
        variance=var,
        first=first,
        first_alt=first_alt,
        total=total,
        total_alt=total_alt,
        first_halfwidth=zeros.copy(),
        total_halfwidth=zeros.copy(),
        n_evals=n * (d + 2),
        rho=0.0,
    )
