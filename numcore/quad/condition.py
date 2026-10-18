r"""2.4 章 —— 求积的**条件**：权重和、负权重，以及输入误差被放大多少倍。

求积公式是一个加权和 :math:`Q=\sum_k w_kf(x_k)`。它的"条件数"不是 :math:`\kappa(A)`
那种矩阵条件数，而是同样朴素的一个比值：

.. math::
    \Gamma=\frac{\sum_k|w_k|}{\left|\sum_k w_k\right|}=\frac{\sum_k|w_k|}{b-a}.

理由是一行不等式：若每个函数值带误差 :math:`|\delta_k|\le\varepsilon|f(x_k)|`，则

.. math::
    |Q(f+\delta)-Q(f)|\le\sum_k|w_k|\,|\delta_k|\le\varepsilon\,|f|_\infty\sum_k|w_k|
    =\varepsilon\,|f|_\infty\,(b-a)\,\Gamma .

于是 :math:`\Gamma` 是"函数值的相对误差"到"积分相对误差"的放大因子：

* **Gauss--Legendre 的权重全为正**，且 :math:`\sum w_k=b-a`，所以 :math:`\Gamma=1`
  （实测 1.000000，机器精度）——权重取正不是美学，它是"函数值准 ⇒ 积分准"的**充分条件**；
* **等距高阶 Newton--Cotes 的权重会变负**：闭公式在 :math:`m=9` 个点处首次出现负权重
  （实测 :math:`\Gamma=1.4512`），:math:`m=13` 起跳到 7.53，:math:`m=19` 时 175.5。
  这时"把函数值算准到 1e-8"只能保证积分准到 :math:`1.75\times10^{-6}`。

注意 :math:`m=10` 是一个**反例**：它的权重全是正的（实测最小权重 +0.0241、:math:`\Gamma=1.000000`），
所以"等距阶数一高就不稳"这句话不能按 :math:`m` 一刀切。判据是 :math:`\Gamma` 这个实测值，
不是 :math:`m` 的大小——这条正是本章"用一个数代替一句话"的样板。

本模块只做算术，不做求积：它吃的是任何一个规则给出的权重向量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from numcore.utils import rng

__all__ = [
    "WeightDiagnostics",
    "amplification_experiment",
    "input_error_bound",
    "roundoff_floor",
    "weight_diagnostics",
]


@dataclass(frozen=True)
class WeightDiagnostics:
    r"""一组求积权重的诊断量（都可由权重向量直接复算，不留"约等于"）。"""

    n_weights: int
    sum_w: float
    abs_sum_w: float
    condition: float
    min_w: float
    max_w: float
    n_negative: int

    @property
    def all_positive(self) -> bool:
        return self.n_negative == 0

    def summary(self) -> str:
        return (
            f"Σw={self.sum_w:.12g}，Σ|w|={self.abs_sum_w:.12g}，Γ={self.condition:.6f}，"
            f"min w={self.min_w:+.6g}，负权重 {self.n_negative}/{self.n_weights} 个"
        )


def weight_diagnostics(weights: Any) -> WeightDiagnostics:
    r"""权重向量的 :math:`\sum w`、:math:`\sum|w|`、:math:`\Gamma`、最小/最大权重与负权重个数。

    :math:`\Gamma` 用 :math:`\sum|w|/|\sum w|` 定义；:math:`\sum w=0`（例如把
    :math:`[-1,1]` 上的奇函数权重配成零和）时 :math:`\Gamma` 无定义，返回 ``inf``——
    这是"该公式连常数都积不出来"的如实反映，不是除以零的事故。
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        raise ValueError("权重向量不能为空")
    sw = float(np.sum(w))
    asw = float(np.sum(np.abs(w)))
    if sw == 0.0:
        gamma = float("inf")
    else:
        gamma = asw / abs(sw)
    return WeightDiagnostics(
        n_weights=int(w.size),
        sum_w=sw,
        abs_sum_w=asw,
        condition=float(gamma),
        min_w=float(np.min(w)),
        max_w=float(np.max(w)),
        n_negative=int(np.count_nonzero(w < 0.0)),
    )


def input_error_bound(weights: Any, rel_error: float, f_scale: float = 1.0) -> float:
    r"""函数值带相对误差 ``rel_error`` 时积分结果的**最坏**绝对误差上界。

    即 :math:`\varepsilon\,|f|_\infty\sum_k|w_k|`。它不是"典型误差"而是上界：
    误差同号时才取到，但**负权重**让这个上界与最优情况差出 :math:`\Gamma` 倍，
    这正是"看起来很准"与"其实不准"之间的那条缝。
    """
    diag = weight_diagnostics(weights)
    return float(abs(rel_error) * abs(float(f_scale)) * diag.abs_sum_w)


def roundoff_floor(weights: Any, f_scale: float = 1.0, *, u: float = 2.0**-53) -> float:
    r"""该规则的**舍入地板** :math:`\approx u\,|f|_\infty\sum_k|w_k|`。

    它解释了误差曲线尾部为什么变平：再加密网格，公式的截断误差在减小，
    而 :math:`\sum|w_k|` 带来的求和噪声不动（甚至因为项数变多而变大）。
    把尾部当"已经收敛"是本章最贵的误读——地板不是 0，它只是不再下降。
    """
    diag = weight_diagnostics(weights)
    return float(u * abs(float(f_scale)) * diag.abs_sum_w)


def amplification_experiment(
    weights: Any,
    values: Any,
    *,
    rel_error: float = 1e-8,
    seed: int = 20241024,
    trials: int = 64,
) -> dict[str, float]:
    r"""**实测**放大因子：给函数值加相对扰动，看积分值动了多少。

    返回 ``{"worst_ratio", "mean_ratio", "adversarial_ratio", "bound", "condition", ...}``，其中
    ``worst_ratio = max_t |ΔQ_t| / (rel_error·|f|_∞)`` 是 64 次确定性随机符号试验里的最大值，
    ``adversarial_ratio = Σ|w_k v_k| / (rel_error·|f|_∞)`` 是**精确**的最坏值（符号取成让每项同号）。
    它与理论界
    :math:`\sum|w_k|` 的关系是本章 A/B 类断言的落点：

    * 正权重规则（Gauss）：``worst_ratio`` 顶到 :math:`\sum|w_k|=b-a` 附近，理论界**紧**；
    * 负权重规则（9 点等距）：同一个函数、同一个 ``rel_error``，``worst_ratio``
      被抬高 :math:`\Gamma` 倍——**不是函数值算坏了，是公式把误差放大了**。

    扰动用 :func:`numcore.utils.rng` 的确定性种子（P4），所以这个"实验"可复现。
    """
    w = np.asarray(weights, dtype=float).ravel()
    v = np.asarray(values, dtype=float).ravel()
    if w.size != v.size:
        raise ValueError(f"权重个数 {w.size} 与函数值个数 {v.size} 不一致")
    if not np.all(np.isfinite(v)):
        raise ValueError("函数值里有非有限数：该规则在这个被积函数上无定义，谈不上放大因子")
    diag = weight_diagnostics(w)
    f_inf = float(np.max(np.abs(v)))
    denom = abs(float(rel_error)) * f_inf
    if denom == 0.0:
        raise ValueError("相对扰动与函数值都为零：放大因子无定义")
    # 第 k 项的扰动绝对大小（相对误差 ε ⇒ 绝对值 ε|v_k|），乘上权重就是它对积分误差的贡献。
    contrib = w * (abs(float(rel_error)) * np.abs(v))
    # **精确**最坏情况：符号取成与 w_k·v_k 同号 ⇒ ΔQ = Σ|w_k v_k|（不需要随机试验）。
    adversarial = float(np.sum(np.abs(contrib)) / denom)
    gen = rng(int(seed))
    worst = 0.0
    total = 0.0
    for _ in range(max(1, int(trials))):
        sign = np.where(gen.random(w.size) < 0.5, -1.0, 1.0)
        # 注意这里算的是 ΔQ = Σ w_k·δv_k 本身：减去 base 会把"变化量"换成"两个 1.7 量级
        # 的数相减"，量出来的放大因子直接错成 1e8（实现期真踩过这个坑，
        # 被"放大比必须等于 Γ"的断言当场抓住）。
        delta = abs(float(np.sum(sign * contrib)))
        worst = max(worst, delta)
        total += delta
    return {
        "worst_ratio": float(worst / denom),
        "mean_ratio": float(total / max(1, int(trials)) / denom),
        "adversarial_ratio": adversarial,
        "bound": float(diag.abs_sum_w),
        "condition": float(diag.condition),
        "f_inf": f_inf,
        "sum_w": float(diag.sum_w),
    }
