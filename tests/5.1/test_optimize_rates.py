"""5.1 章 C 类断言（收敛率与阈值）：实测因子、二分出来的临界步长与预算阈值。

这一文件的三条主线：

1. **实测收敛因子 vs 理论**：GD 的 :math:`\\rho=(\\kappa-1)/(\\kappa+1)` 逐位吻合；
   重球动量的 :math:`(\\sqrt\\kappa-1)/(\\sqrt\\kappa+1)` 也吻合，但**迭代数之比不是**
   :math:`\\sqrt\\kappa`（实测只有 0.74–0.81 倍）——速率是渐近的，迭代数含瞬态。
2. **临界步长二分实测**：GD 的"还收敛 / 发散"分界点 = :math:`2/L`。
   这里不抄 :math:`2/L`，而是让脚本自己找到它，再与理论值对账。
3. **条件数预算阈值**：固定 2000 步与 :math:`\\varepsilon=10^{-8}`，
   二分找出"刚好还能收敛"的 κ，与 :math:`2N/\\ln(1/\\varepsilon)` 对账。
"""

from __future__ import annotations

import numpy as np

from numcore.inverse.optimize import (
    anisotropic_quadratic,
    bfgs,
    euclidean_norm,
    gd,
    gd_rate,
    lbfgs,
    momentum,
    momentum_rate,
    quadratic_from_spectrum,
)
from numcore.verify import (
    assert_linear_convergence,
    assert_threshold,
    assertion,
    estimate_linear_factor,
    find_threshold,
    record,
)


# --------------------------------------------------------------------------- #
# 1. GD 的线性收敛因子
# --------------------------------------------------------------------------- #
@assertion("C", theory="GD 的梯度范数按 ρ=(κ-1)/(κ+1) 线性收敛（最优步长）", chapter="5.1")
def test_gd_linear_convergence_factor_matches_theory() -> None:
    r"""实测因子与 :math:`(\\kappa-1)/(\\kappa+1)` 的相对差（三个 κ）。

    ``assert_linear_convergence`` 登记 C 类记录；另登记一条"实测/理论"的比值，
    免得"容差 20%"把结论说软了：实测差在 1e-6 量级。
    """
    worst = 0.0
    detail = []
    measured_rates = []
    for kap in (100.0, 1000.0, 10000.0):
        Q = anisotropic_quadratic(kap, angle_deg=35.0)
        tr = gd(Q, np.array([1.0, 1.0]), max_iter=2000000, tol=1e-10)
        rho = assert_linear_convergence(
            tr.gnorm,
            rate=gd_rate(kap),
            rtol=0.01,
            skip=5,
            tail=50,
            label=f"GD 收敛因子（κ={kap:g}）",
            detail=f"tail=50，{tr.n_iter} 步到 ‖g‖/‖g0‖={tr.gnorm[-1] / tr.gnorm[0]:.1e}",
        )
        measured_rates.append(rho)
        rel = abs(rho - gd_rate(kap)) / gd_rate(kap)
        worst = max(worst, rel)
        detail.append(f"κ={kap:g}: 实测 {rho:.6f} / 理论 {gd_rate(kap):.6f}")
    rec = record(
        label="实测收敛因子与 (κ-1)/(κ+1) 的最大相对差",
        theory="≤ 1e-5",
        measured=f"{worst:.3e}（" + "；".join(detail) + "）",
        tol="1e-5",
        passed=bool(worst <= 1e-5 and len(measured_rates) == 3),
        kind="C",
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. 重球动量：速率对得上，迭代数对不上
# --------------------------------------------------------------------------- #
@assertion("C", theory="重球动量的速率 =(√κ-1)/(√κ+1)；迭代数之比 ≠ √κ（含瞬态）", chapter="5.1")
def test_momentum_rate_matches_theory_but_iteration_ratio_does_not() -> None:
    r"""速率的相对差 ≤ 2%；但 :math:`N_{gd}/N_{mom}` 只有 :math:`0.74\\!-\\!0.81\\sqrt\\kappa`。

    "动量把迭代数降到 :math:`\\sqrt\\kappa` 分之一"是渐近速率的推论，
    实测的**迭代数**比它差 20–26%（初始瞬态 + 梯度范数判据的振幅效应）。
    这条记录的作用就是让这句话在纸面上保持诚实。
    """
    worst_rate = 0.0
    ratios = []
    detail = []
    for kap in (100.0, 1000.0, 10000.0):
        Q = anisotropic_quadratic(kap, angle_deg=35.0)
        tg = gd(Q, np.array([1.0, 1.0]), max_iter=2000000, tol=1e-10)
        tm = momentum(Q, np.array([1.0, 1.0]), max_iter=2000000, tol=1e-10)
        rho = estimate_linear_factor(tm.gnorm, skip=5, tail=50)
        rel = abs(rho - momentum_rate(kap)) / momentum_rate(kap)
        worst_rate = max(worst_rate, rel)
        ratios.append(tg.n_iter / tm.n_iter / np.sqrt(kap))
        detail.append(f"κ={kap:g}: ρ 实测 {rho:.6f}/理论 {momentum_rate(kap):.6f}，N 比 {tg.n_iter / tm.n_iter:.2f}")
    rec = record(
        label="动量实测速率与 (√κ-1)/(√κ+1) 的最大相对差",
        theory="≤ 0.02",
        measured=f"{worst_rate:.4f}（" + "；".join(detail) + "）",
        tol="0.02",
        passed=bool(worst_rate <= 0.02),
        kind="C",
    )
    rec2 = record(
        label="N_gd/N_momentum ÷ √κ",
        theory="实测 0.70–0.85（**不是** 1：速率是渐近的）",
        measured=f"{np.array2string(np.asarray(ratios), precision=4)}",
        tol="0.70–0.85",
        passed=bool(bool(np.all(np.asarray(ratios) >= 0.70)) and bool(np.all(np.asarray(ratios) <= 0.85))),
        kind="C",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 3. 临界步长：二分实测 vs 2/L
# --------------------------------------------------------------------------- #
@assertion("C", theory="GD 在 α ≤ 2/L 时仍收敛（N 步内降 10 倍），α>2/L 发散", chapter="5.1")
def test_gd_step_threshold_is_bisected_not_copied() -> None:
    r"""二分找出"还能把距离降 10 倍"的最大步长，与解析阈值对账（相对 5e-3）。

    解析阈值不是 :math:`2/L` 本身：判据是"**N 步内**降 10 倍"，
    所以分界点在 :math:`\\rho^N=0.1` 处，即
    :math:`\\alpha^\\star_{\\text{thr}}=(1+0.1^{1/N})/L`
    ——这个修正项写进 ``theory``，不是事后调的容差。
    """
    kappa = 100.0
    Q = anisotropic_quadratic(kappa, angle_deg=35.0)
    alpha_star = float(Q.optimal_step())
    n_steps = 2000
    x0 = np.array([1.0, 1.0])

    def reduces(delta: float) -> bool:
        tr = gd(Q, x0, step=float(delta) * alpha_star, max_iter=n_steps, tol=1e-300)
        return euclidean_norm(tr.x_final) <= 0.1 * euclidean_norm(x0)

    hits = {"n": 0}

    def predicate(scale: float) -> bool:
        hits["n"] += 1
        return reduces(scale)

    res = find_threshold(predicate, 0.5, 1.5, tol=1e-9, note=f"{n_steps} 步内降 10 倍")
    theory = (1.0 + 0.1 ** (1.0 / n_steps)) * (Q.L + Q.mu) / (2.0 * Q.L)
    assert_threshold(res, theory, rtol=5e-3, label="最大可收敛步长 / α*", detail=f"{n_steps} 步、降 10 倍判据")
    rec = record(
        label="临界步长与 2/L 的比值",
        theory="1（α_thr → 2/L，见 theory 里的 0.1^(1/N) 修正）",
        measured=f"α_thr/α*={res.threshold:.9f}，α*/(2/L)={alpha_star * Q.L / 2.0:.9f}",
        tol="相对 5e-3",
        passed=bool(abs(res.threshold * alpha_star * Q.L / 2.0 - 1.0) <= 5e-3),
        kind="C",
    )
    rec2 = record(
        label="二分区间宽度 / 谓词调用次数",
        theory="区间宽度 ≤ 1e-9（相对 1）",
        measured=f"{res.width:.3e}（{res.iterations} 次二分，{res.calls} 次谓词调用）",
        tol="1e-9",
        passed=bool(res.width <= 1e-9),
        kind="C",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 4. 迭代预算：二分找出"刚好还能收敛"的 κ
# --------------------------------------------------------------------------- #
@assertion("C", theory="N 步预算下 GD 能收敛的最大 κ ≈ 2N/ln(1/ε)", chapter="5.1")
def test_iteration_budget_condition_number_threshold() -> None:
    r"""固定 N=2000、ε=1e-8，二分找出 κ 的临界值，与 :math:`2N/\\ln(1/\\varepsilon)` 对账。

    "迭代数与条件数成正比"这句话的**可用形式**就是这个阈值：
    :math:`\\kappa_{\\max}\\approx 2N/\\ln(1/\\varepsilon)`，即"给我 N 步，我能处理多病态的问题"。
    """
    n_budget = 2000
    eps = 1e-8

    def within_budget(kap: float) -> bool:
        Q = quadratic_from_spectrum([1.0, float(kap)], seed=11, x_star=np.zeros(2))
        tr = gd(Q, np.array([1.0, 1.0]), max_iter=n_budget, tol=eps, tol_kind="grad")
        return tr.stop == "grad"

    res = find_threshold(within_budget, 1e-2, 1e5, tol=1e-9, note=f"N={n_budget}, ε={eps:g}")
    theory = 2.0 * n_budget / np.log(1.0 / eps)
    assert_threshold(res, theory, rtol=5e-3, label="κ_max（N 步预算）", detail="判据：stop=='grad'")
    rec = record(
        label="κ_max 实测 / 2N·ln(1/ε)^(-1)",
        theory="1（相对 5e-3）",
        measured=f"{res.threshold:.4f} / {theory:.4f} = {res.threshold / theory:.6f}",
        tol="5e-3",
        passed=bool(abs(res.threshold / theory - 1.0) <= 5e-3),
        kind="C",
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 5. L-BFGS 的记忆 m
# --------------------------------------------------------------------------- #
@assertion("C", theory="L-BFGS 迭代数随记忆 m 单调下降；m ≥ n 时接近完整 BFGS", chapter="5.1")
def test_lbfgs_memory_reduces_iteration_count() -> None:
    r"""m = 1,2,3,5,8,12,20 的实测迭代数：216 → 128 → 40 → 28 → 23 → 22 → 22，完整 BFGS 7 步。

    单调性用"不增"而不是"严格递减"来判（m=3 与 m=4 会打平），
    否则这条断言会变成对浮点噪声的断言。
    """
    Q = quadratic_from_spectrum([1.0, 10.0, 100.0, 1000.0], seed=7, x_star=np.zeros(4))
    x0 = np.array([2.0, -1.0, 0.5, 3.0])
    memories = (1, 2, 3, 5, 8, 12, 20)
    counts = []
    for m in memories:
        tr = lbfgs(Q, x0, memory=m, max_iter=5000, tol=1e-8, tol_kind="grad")
        counts.append(int(tr.n_iter))
    full = int(bfgs(Q, x0, max_iter=5000, tol=1e-8, tol_kind="grad").n_iter)
    non_increasing = all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
    rec = record(
        label="L-BFGS 迭代数随 m 的序列",
        theory="不增（m 越大保留的曲率信息越多）",
        measured=f"m={list(memories)} → {counts}（完整 BFGS {full} 步）",
        tol="单调不增",
        passed=bool(non_increasing),
        kind="C",
    )
    rec2 = record(
        label="N(m=1)/N(m=20) 与 N(m=20)/N(BFGS)",
        theory="≥ 5 与 ≤ 6（记忆够用时接近完整 BFGS）",
        measured=f"{counts[0] / counts[-1]:.2f} 与 {counts[-1] / full:.2f}",
        tol="≥5 且 ≤6",
        passed=bool(counts[0] / counts[-1] >= 5.0 and counts[-1] / full <= 6.0),
        kind="C",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"
