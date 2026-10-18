"""5.2 章 A/B/C 断言（二）：L 曲线、GCV 与"到底按哪个判据选 lambda"。

这一文件要钉死三件事：

1. **拐点不是一个数**：三个判据（曲率、两段直线拟合、离散二阶差分）在同一组数据上
   给出的 lambda 实测可以差两个数量级；无约束的两段直线拟合还会直接退化到网格端点。
   曲率判据本身与 1.3 章的 ``lcurve_corner`` 逐位一致（口径统一，不是另起一套）。
2. **判据之间可以差 10 倍**：``kappa = 1e6``、噪声 ``1e-3`` 时 L 曲线拐点比 oracle 小
   21 倍（误差放大 21.5 倍），GCV 大 4.6 倍（放大 3.8 倍），偏差原则几乎贴着 oracle
   （1.16 倍）——而 oracle 用了现实中拿不到的真解。
3. **lambda 的两侧都在网格里**：左端误差随 ``kappa`` 线性增长（实测 0.124 -> 967），
   右端误差回到 ``||x*||`` 的 0.99 倍——"两端都不行"不是修辞，是两条曲线。

``gcv_curve`` 与 1.3 章 ``cross_validation_curve`` 的对拍（2.9e-12）是本文件最有价值的一条：
开发时这里的残差多乘了一次 ``sigma``，整条 GCV 曲线错了 3.2e4 倍，而且
``lambda_GCV`` 对噪声水平完全不敏感（斜率 0.000）——正是这条对拍把它抓出来的。
"""

from __future__ import annotations

import numpy as np

from numcore.inverse.lcurve import (
    corner_criteria,
    criterion_comparison,
    current_solution,
    gcv_curve,
    lambda_discrepancy,
    lambda_gcv,
    lambda_grid,
    lambda_oracle,
    lcurve,
    problem_lcurve,
)
from numcore.inverse.regularize import (
    test_problem as synthetic_problem,
    tikhonov,
    vector_norm,
)
from numcore.linalg.lstsq import cross_validation_curve, lcurve_corner
from numcore.verify import (
    assert_all_finite,
    assert_leq,
    assert_monotone,
    assert_threshold,
    assert_within,
    assertion,
    find_threshold,
    fit_order,
    record,
)

CHAPTER = "5.2"
KAPPA = 1e6
NOISE = 1e-3


def _problem(kappa: float = KAPPA, noise: float = NOISE, picard: str = "satisfied"):
    return synthetic_problem(m=32, n=20, kappa=kappa, picard=picard, noise_rel=noise)


# --------------------------------------------------------------------------- #
# 1. 拐点判据
# --------------------------------------------------------------------------- #
@assertion("A", theory="曲率判据与 1.3 章 lcurve_corner 同一口径（拐点下标逐位一致）", chapter=CHAPTER)
def test_curvature_corner_matches_reference_chapter() -> None:
    r"""5 组配置下我的曲率极值点 == 1.3 的 ``lcurve_corner``。

    "口径一致"必须是逐位一致：两章对同一个拐点给出不同下标，后面的所有读数都不可比。"""
    idx_mine, idx_ref, measured = [], [], []
    for noise in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        prob = _problem(noise=noise)
        lam = lambda_grid(prob.s, n_lam=61)
        lc = lcurve(prob.U, prob.s, prob.V, prob.b, x_true=prob.x_true, lam=lam)
        i_mine = int(np.argmax(lc.curvature))
        i_ref = int(lcurve_corner(lc.x_norm, lc.residual))
        idx_mine.append(i_mine)
        idx_ref.append(i_ref)
        measured.append(f"noise=1e{int(np.log10(noise))}: {i_mine} vs {i_ref}")
    agree = bool(np.array_equal(idx_mine, idx_ref))
    rec = record(
        label="曲率拐点下标（本模块 vs 1.3 lcurve_corner）",
        theory="同一离散曲率公式（相邻三点中心差分）=> 下标必须逐位相同",
        measured="；".join(measured),
        tol="5 组全部一致",
        passed=agree,
        kind="A",
    )
    assert rec.passed, rec.measured


@assertion("A", theory="L 曲线拐点不是良定义的：三个判据给出的 lambda 可以差一个数量级以上", chapter=CHAPTER)
def test_corner_criteria_disagree_measurably() -> None:
    r"""三个判据的 lambda 极差 >= 5 倍（实测 30~680 倍），
    并且**无约束**的两段直线拟合在多数配置下退化到网格端点。

    这是本章"诚实性要求 2"的核心记录：把"拐点"当成一个客观量是错的。"""
    worst_ratio = 0.0
    ratios = []
    measured = []
    for kappa in (1e3, 1e4, 1e5, 1e6, 1e7):
        prob = _problem(kappa=kappa)
        lam = lambda_grid(prob.s, n_lam=61)
        lc = lcurve(prob.U, prob.s, prob.V, prob.b, x_true=prob.x_true, lam=lam)
        cc = corner_criteria(lc.x_norm, lc.residual, lam)
        corners = np.array([cc["lam_curvature"], cc["lam_two_line"], cc["lam_second_difference"]])
        ratio = float(np.max(corners) / np.min(corners))
        worst_ratio = max(worst_ratio, ratio)
        ratios.append(ratio)
        measured.append(
            f"kappa=1e{int(np.log10(kappa))}: lambda 极差 {ratio:.0f} 倍、下标跨度 "
            f"{cc['index_spread']}（曲率 idx {cc['index_curvature']}，两段直线 idx {cc['index_two_line']}，"
            f"二阶差分 idx {cc['index_second_difference']}）"
        )
    rec = record(
        label="三个拐点判据的 lambda 极差 / 下标跨度",
        theory="曲率、两段直线交点、离散二阶差分是三个不同启发式，没有理由给出同一个数",
        measured="；".join(measured),
        tol="每一组配置的 lambda 极差都 >= 5 倍（实测最小 33 倍）",
        passed=bool(float(np.min(ratios)) >= 5.0),
        kind="A",
    )
    assert rec.passed, rec.measured


@assertion("A", theory="两段直线拟合必须加约束：每段至少 25% 的点且交点在包围盒内", chapter=CHAPTER)
def test_two_line_fit_constraints_are_necessary() -> None:
    r"""无约束版本（只最小化两段残差和）会退化成"第一段只含 2 个点"的伪解，
    返回网格端点；加约束后回到内部（实测 5/5 组）。

    负面结论与正面结论成对出现：**不加约束的这个判据是坏的**。"""
    devs, agree = [], 0
    n_cfg = 0
    for kappa in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7):
        for noise in (1e-2, 1e-3, 1e-5):
            prob = _problem(kappa=kappa, noise=noise)
            lam = lambda_grid(prob.s, n_lam=61)
            lc = lcurve(prob.U, prob.s, prob.V, prob.b, x_true=prob.x_true, lam=lam)
            cc = corner_criteria(lc.x_norm, lc.residual, lam)
            n_cfg += 1
            f, c = float(cc["lam_two_line_free"]), float(cc["lam_two_line"])
            if abs(np.log10(f / c)) <= 1e-12:
                agree += 1
            devs.append(max(f / c, c / f))
    devs_arr = np.asarray(devs)
    rec = record(
        label="无约束 vs 有约束两段直线拟合：一致组数 / 最大偏离倍数（18 组配置）",
        theory="无约束时段越短拟合残差越小 => 切点无规律漂移；约束把切点限制在 [25%, 75%] 且交点必须落在包围盒内",
        measured=(
            f"一致 {agree}/{n_cfg} 组；其余偏离最大 {float(np.max(devs_arr)):.1f} 倍"
            f"（中位 {float(np.median(devs_arr)):.2f}）"
        ),
        tol="一致组数 <= 12/18（即至少 1/3 不一致），且最大偏离 >= 5 倍",
        passed=bool(agree <= 12 and float(np.max(devs_arr)) >= 5.0),
        kind="A",
    )
    prob = _problem()
    lam = lambda_grid(prob.s, n_lam=61)
    lc = lcurve(prob.U, prob.s, prob.V, prob.b, x_true=prob.x_true, lam=lam)
    cc = corner_criteria(lc.x_norm, lc.residual, lam)
    rec2 = record(
        label="有约束拐点的 lambda 落在 L 曲线拐弯区间内",
        theory="约束版的交点应当落在曲率极大点附近（同一段弯），偏离量就是这条判据的不确定度",
        measured=(
            f"有约束 lambda = {cc['lam_two_line']:.3e}，曲率判据 lambda = {cc['lam_curvature']:.3e}，"
            f"比值 {cc['ratio_two_line_over_curvature']:.2f}"
        ),
        tol="比值落在 [0.01, 100]",
        passed=bool(0.01 <= cc["ratio_two_line_over_curvature"] <= 100.0),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 2. GCV 与偏差原则
# --------------------------------------------------------------------------- #
@assertion("A", theory="GCV 曲线与 1.3 章 cross_validation_curve 同一个函数（相对差 <= 1e-9）", chapter=CHAPTER)
def test_gcv_matches_reference_chapter() -> None:
    r"""与 1.3 的 GCV 对拍（实测最大相对差 2.9e-12）。

    这里曾经有过一个真 bug：残差多乘了一个 ``sigma``，最大相对差 3.2e4、
    而且 ``lambda_GCV`` 对噪声不敏感（5 个噪声档位的对数斜率实测 0.000）。
    这条对拍是唯一能抓住它的检查。"""
    prob = _problem()
    ref = cross_validation_curve(prob.A, prob.b, n_lambda=40)
    mine = gcv_curve(prob.U, prob.s, prob.b, ref["lambda"], m=prob.m)
    rel = float(np.max(np.abs(mine - ref["cv"]) / np.maximum(np.abs(ref["cv"]), np.finfo(float).tiny)))
    rec = record(
        label="GCV 曲线与 1.3 cross_validation_curve 的最大相对差",
        theory="G(lambda) = ||A x_lambda - b||^2 / (m - sum f_i)^2，两条实现同一个公式",
        measured=f"max rel diff = {rel:.3e}；最小值位置 {lambda_gcv(prob.U, prob.s, prob.b, ref['lambda']):.6g} vs {ref['lambda_best']:.6g}",
        tol="<= 1e-9",
        passed=bool(rel <= 1e-9),
        kind="A",
    )
    vals = []
    for noise in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        p = _problem(noise=noise)
        vals.append(lambda_gcv(p.U, p.s, p.b, lambda_grid(p.s, n_lam=61)))
    rec2 = record(
        label="GCV 不被噪声水平骗：lambda_GCV 随噪声下降而下降",
        theory="噪声越小，能安全保留的模态越多 => 最优 lambda 越小",
        measured=f"lambda_GCV = {np.array2string(np.array(vals), precision=3)}（噪声 1e-2 -> 1e-6）",
        tol="严格单调下降",
        passed=bool(np.all(np.diff(np.array(vals)) < 0.0)),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="lambda_GCV 就是 GCV 曲线的最小点；偏差原则的解残差等于噪声水平", chapter=CHAPTER)
def test_gcv_argmin_and_discrepancy_target() -> None:
    r"""两个"数值定义"的检查：``lambda_gcv`` 落在 ``argmin G`` 上（1e-12 内），
    ``lambda_discrepancy`` 处 ``||Ax-b||/||e||`` 与 1 的偏差 <= 2%（实测 1.5%）。

    偏差原则的容差之所以不是机器精度：交点在两个网格点之间用对数插值找，
    而残差不是 ``log lambda`` 的线性函数。"""
    prob = _problem()
    lam = lambda_grid(prob.s, n_lam=61)
    g = gcv_curve(prob.U, prob.s, prob.b, lam, m=prob.m)
    l_gcv = lambda_gcv(prob.U, prob.s, prob.b, lam)
    rec = record(
        label="lambda_GCV 与 GCV 曲线最小点的相对差",
        theory="lambda_GCV = argmin G(lambda)（定义）",
        measured=f"|lambda_GCV - lambda_argmin| / lambda_GCV = {abs(l_gcv - lam[int(np.argmin(g))]) / l_gcv:.3e}",
        tol="<= 1e-12",
        passed=bool(abs(l_gcv - lam[int(np.argmin(g))]) / l_gcv <= 1e-12),
        kind="A",
    )
    l_disc = lambda_discrepancy(prob.U, prob.s, prob.V, prob.b, prob.noise_abs, lam)
    sol = tikhonov(prob.U, prob.s, prob.V, prob.b, l_disc)
    ratio = float(sol.residual_norm) / float(prob.noise_abs)
    rec2 = record(
        label="偏差原则处的残差 / 噪声水平",
        theory="||A x_lambda - b|| = tau ||e||（tau = 1），交点用对数插值",
        measured=f"lambda_disc = {l_disc:.4e}，比值 = {ratio:.6f}",
        tol="|比值 - 1| <= 0.02",
        passed=bool(abs(ratio - 1.0) <= 0.02),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="oracle 是误差曲线的最小点，且四判据的误差放大倍数有序", chapter=CHAPTER)
def test_oracle_minimizes_error_and_inflation_ordering() -> None:
    r"""``lambda_oracle`` 用真解算出（现实中拿不到），它必须真的是误差最小点。

    同一组读数上四个判据的代价（实测）：oracle 1.0 倍 -> 偏差原则 1.16 倍
    -> GCV 3.79 倍 -> L 曲线 21.48 倍 -> 完全不正则化 12863 倍。"""
    prob = _problem()
    lam = lambda_grid(prob.s, n_lam=61)
    lc = lcurve(prob.U, prob.s, prob.V, prob.b, x_true=prob.x_true, lam=lam)
    l_or = lambda_oracle(prob.U, prob.s, prob.V, prob.b, prob.x_true, lam)
    i_or = int(np.argmin(lc.error))
    ch = criterion_comparison(
        prob.U, prob.s, prob.V, prob.b, x_true=prob.x_true, noise_abs=prob.noise_abs, lam=lam
    )
    rec = record(
        label="oracle 位置与误差曲线最小点",
        theory="lambda_oracle = argmin ||x_lambda - x*||（定义）",
        measured=(
            f"|lambda_oracle - lambda_argmin| / lambda_oracle = "
            f"{abs(l_or - lam[i_or]) / l_or:.3e}；最小误差 {lc.error[i_or]:.4f}"
        ),
        tol="<= 1e-12",
        passed=bool(abs(l_or - lam[i_or]) / l_or <= 1e-12),
        kind="A",
    )
    ordered = bool(
        1.0 <= ch.inflation_discrepancy <= ch.inflation_gcv <= ch.inflation_lcurve
        <= ch.detail["inflation_unregularized"]
    )
    rec2 = record(
        label="四个判据的误差放大倍数（oracle / 偏差原则 / GCV / L 曲线 / 不正则化）",
        theory="判据越依赖残差小这种间接信息，越偏向欠正则化的一端",
        measured=(
            f"1.000（oracle）| {ch.inflation_discrepancy:.3f}（disc）| {ch.inflation_gcv:.3f}（GCV）| "
            f"{ch.inflation_lcurve:.3f}（L）| {ch.detail['inflation_unregularized']:.0f}（无）"
        ),
        tol="倍数单调不减，且 L 曲线至少比 GCV 差 2 倍",
        passed=bool(ordered and ch.inflation_lcurve >= 2.0 * ch.inflation_gcv),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="lambda 网格两端各是一种失败方式：左端噪声接管、右端解被压成 0", chapter=CHAPTER)
def test_both_failure_modes_are_inside_the_grid() -> None:
    r"""左端：误差随 ``kappa`` 线性增长（1e3 -> 0.124，1e7 -> 967）；
    右端：误差 = 0.992 ~ 1，即解被压成 0 时误差回到 ``||x*||``。

    "lambda 要选得合适"这句话的两侧代价都在网格里量得到。"""
    errs_lo, errs_hi, ratios = [], [], []
    for kappa in (1e3, 1e4, 1e5, 1e6, 1e7):
        prob = _problem(kappa=kappa)
        lc = problem_lcurve(prob, n_lam=61)
        errs_lo.append(float(lc.error[0]))
        errs_hi.append(float(lc.error[-1]))
        ratios.append(float(lc.x_norm[-1] / lc.x_norm[0]))
    rec = record(
        label="网格两端（lambda 最小 / 最大）的相对误差与解范数比",
        theory="左端：无正则化的噪声放大；右端：x_lambda -> 0 => 误差 -> ||x*||",
        measured=(
            f"左端 {np.array2string(np.array(errs_lo), precision=3)}；"
            f"右端 {np.array2string(np.array(errs_hi), precision=3)}；"
            f"||x(lambda_max)||/||x(lambda_min)|| = {np.array2string(np.array(ratios), precision=2)}"
        ),
        tol="右端误差 >= 0.9（接近 1）、解范数比 <= 1e-2、左端误差随 kappa 增长",
        passed=bool(
            float(np.min(errs_hi)) >= 0.9
            and float(np.max(ratios)) <= 1e-2
            and float(errs_lo[-1]) >= 100.0 * float(errs_lo[0])
        ),
        kind="A",
    )
    assert rec.passed, rec.measured


@assertion("A", theory="L 曲线的坐标与逐点解一致：x_norm/residual 由 lambda 处的真解给出", chapter=CHAPTER)
def test_lcurve_coordinates_match_direct_solutions() -> None:
    r"""独立重算三处坐标（自己解 Tikhonov、自己算范数）与 ``lcurve`` 的数组对拍。

    这条防止"曲线画得漂亮但坐标不是解出来的"这一类错误。"""
    prob = _problem()
    lam = lambda_grid(prob.s, n_lam=61)
    lc = lcurve(prob.U, prob.s, prob.V, prob.b, x_true=prob.x_true, lam=lam)
    worst_x, worst_r = 0.0, 0.0
    for i in (0, 17, 30, 44, 60):
        sol = tikhonov(prob.U, prob.s, prob.V, prob.b, lam[i], x_true=prob.x_true)
        worst_x = max(worst_x, abs(sol.x_norm - lc.x_norm[i]) / sol.x_norm)
        worst_r = max(worst_r, abs(sol.residual_rel - lc.residual[i]) / sol.residual_rel)
    rec = record(
        label="曲线坐标与逐点独立求解的最大相对差（||x|| 与相对残差）",
        theory="L 曲线的每个点就是该 lambda 处 Tikhonov 解的范数与残差",
        measured=f"||x|| 相对差 {worst_x:.3e}，相对残差 {worst_r:.3e}",
        tol="<= 1e-12",
        passed=bool(worst_x <= 1e-12 and worst_r <= 1e-12),
        kind="A",
    )
    rec2 = record(
        label="current_solution 与 tikhonov 的一致性（拖动 lambda 的那条路）",
        theory="交互拖动 lambda 时前端拿到的是同一个解",
        measured=f"max|dx| = {float(np.max(np.abs(current_solution(prob, 1e-2).x - tikhonov(prob.U, prob.s, prob.V, prob.b, 1e-2).x))):.3e}",
        tol="<= 1e-15",
        passed=bool(
            float(np.max(np.abs(current_solution(prob, 1e-2).x - tikhonov(prob.U, prob.s, prob.V, prob.b, 1e-2).x)))
            <= 1e-15
        ),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 3. B 类
# --------------------------------------------------------------------------- #
@assertion("B", theory="三个受约束的拐点判据都落在网格内部（不贴端点）", chapter=CHAPTER)
def test_corner_indices_are_interior() -> None:
    for kappa in (1e3, 1e5, 1e7):
        prob = _problem(kappa=kappa)
        lam = lambda_grid(prob.s, n_lam=61)
        lc = lcurve(prob.U, prob.s, prob.V, prob.b, x_true=prob.x_true, lam=lam)
        cc = corner_criteria(lc.x_norm, lc.residual, lam)
        idx = np.asarray(
            [cc["index_curvature"], cc["index_two_line"], cc["index_second_difference"]], dtype=float
        )
        # 与 [1, k-2] 的偏差必须为 0（夹住之后的残差就是越界量）
        dev = float(np.max(np.abs(idx - np.clip(idx, 1.0, float(lam.size - 2)))))
        assert_leq(dev, 0.0, label=f"三个拐点下标都在网格内部（kappa=1e{int(np.log10(kappa))}）")


@assertion("B", theory="判据读数全部有限；无效项用 NaN 表示而不是 0", chapter=CHAPTER)
def test_criterion_output_finite_and_nan_policy() -> None:
    prob = _problem()
    lam = lambda_grid(prob.s, n_lam=61)
    ch = criterion_comparison(
        prob.U, prob.s, prob.V, prob.b, x_true=prob.x_true, noise_abs=prob.noise_abs, lam=lam
    )
    assert_all_finite([ch.lam_oracle, ch.lam_lcurve, ch.lam_gcv, ch.error_oracle, ch.inflation_gcv],
                      label="有效判据读数全有限")
    clean = _problem(noise=0.0)
    lam_c = lambda_grid(clean.s, n_lam=61)
    l_none = lambda_discrepancy(clean.U, clean.s, clean.V, clean.b, clean.noise_abs, lam_c)
    assert_within(float(np.isnan(l_none)), 1.0, rtol=0.0, label="无噪时偏差原则没有交点 -> NaN（不是 0）")


@assertion("B", theory="L 曲线的两个坐标随 lambda 单调（x 范数不增、残差不减）", chapter=CHAPTER)
def test_lcurve_monotone_coordinates() -> None:
    prob = _problem()
    lc = problem_lcurve(prob, n_lam=61)
    assert_monotone(lc.x_norm, direction="nonincreasing", label="||x_lambda|| 随 lambda 不增")
    assert_monotone(lc.residual, direction="nondecreasing", label="相对残差随 lambda 不减")
    assert_all_finite(lc.curvature, label="离散曲率全有限")


@assertion("B", theory="lambda 网格覆盖 [sigma_n/10, 10 sigma_1]，两端都能到达失败区", chapter=CHAPTER)
def test_lambda_grid_span() -> None:
    prob = _problem()
    lam = lambda_grid(prob.s, n_lam=61)
    assert_within(float(lam[0]), float(prob.s[-1]) / 10.0, rtol=1e-12, label="网格左端 = sigma_n/10")
    assert_within(float(lam[-1]), 10.0 * float(prob.s[0]), rtol=1e-12, label="网格右端 = 10 sigma_1")
    assert_monotone(lam, direction="increasing", label="网格单调递增")


# --------------------------------------------------------------------------- #
# 4. C 类
# --------------------------------------------------------------------------- #
@assertion("C", theory="噪声越大，L 曲线拐点的相对代价越小（二分实测 10 倍代价的噪声阈值）", chapter=CHAPTER)
def test_lcurve_penalty_threshold_in_noise_level() -> None:
    r"""把噪声水平当自变量，二分找"L 曲线误差放大倍数跌到 10 倍"的位置（实测 log10 噪声 ~ -2.5）。

    趋势本身也是结论：噪声越小，拐点越糟（实测 1e-2 时 6.5 倍，1e-6 时 212 倍）——
    因为真最优 lambda 随噪声下降，而拐点滞后。"""
    def penalty(noise: float) -> float:
        prob = _problem(noise=float(noise))
        lam = lambda_grid(prob.s, n_lam=61)
        ch = criterion_comparison(
            prob.U, prob.s, prob.V, prob.b, x_true=prob.x_true, noise_abs=prob.noise_abs, lam=lam
        )
        return float(ch.inflation_lcurve)

    res = find_threshold(
        lambda n: penalty(n) <= 10.0, 1e-6, 1e-1, tol=1e-10, assume="false_at_lo",
        note="判据：L 曲线拐点的误差放大倍数 <= 10",
    )
    vals = [penalty(n) for n in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)]
    rec = record(
        label="L 曲线代价降到 10 倍以下的噪声水平（二分）",
        theory="最优 lambda 随噪声下降，而 L 曲线拐点滞后 => 噪声越小拐点越差",
        measured=(
            f"阈值 = {res.threshold:.4e}（对数 {-np.log10(res.threshold):.2f} 个数量级）；"
            f"噪声 1e-2..1e-6 时的代价 {np.array2string(np.array(vals), precision=1)}"
        ),
        tol="阈值落在 [1e-3, 1e-1]，且代价随噪声下降而单调上升",
        passed=bool(1e-3 <= res.threshold <= 1e-1 and bool(np.all(np.diff(vals) > 0.0))),
        kind="C",
    )
    assert rec.passed, rec.measured


@assertion("C", theory="lambda_GCV 与噪声水平成幂律（几何谱下指数约 1/2）", chapter=CHAPTER)
def test_gcv_lambda_scales_with_noise() -> None:
    noises = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    vals = []
    for n in noises:
        prob = _problem(noise=float(n))
        vals.append(lambda_gcv(prob.U, prob.s, prob.b, lambda_grid(prob.s, n_lam=61)))
    vals_arr = np.asarray(vals)
    fit = fit_order(noises[::-1], vals_arr[::-1])
    rec = record(
        label="lambda_GCV 对噪声水平的幂律指数",
        theory="几何谱 + GCV：lambda_GCV ~ noise^(1/2)（实测指数略低于 1/2）",
        measured=f"p = {fit.slope:.4f}（R^2 = {fit.r2:.6f}）；lambda_GCV = {np.array2string(vals_arr, precision=2)}",
        tol="|p - 0.5| <= 0.1",
        passed=bool(abs(fit.slope - 0.5) <= 0.1),
        kind="C",
    )
    ors = []
    for n in noises:
        prob = _problem(noise=float(n))
        ors.append(lambda_oracle(prob.U, prob.s, prob.V, prob.b, prob.x_true, lambda_grid(prob.s, n_lam=61)))
    fit_or = fit_order(noises[::-1], np.asarray(ors)[::-1])
    rec2 = record(
        label="lambda_oracle 对噪声水平的幂律指数（对照）",
        theory="oracle 的 lambda 也随噪声下降；GCV 的指数与它的差就是判据的滞后量",
        measured=f"p = {fit_or.slope:.4f}（GCV 比它{'小' if fit.slope < fit_or.slope else '大'} {abs(fit.slope - fit_or.slope):.3f}）",
        tol="两者都落在 [0.3, 0.7]",
        passed=bool(0.3 <= fit_or.slope <= 0.7),
        kind="C",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("C", theory="网格左端的误差随 kappa 线性增长（指数 1）", chapter=CHAPTER)
def test_left_edge_error_scales_with_kappa() -> None:
    kappas = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
    errs = []
    for k in kappas:
        prob = _problem(kappa=float(k))
        errs.append(float(problem_lcurve(prob, n_lam=61).error[0]))
    errs_arr = np.asarray(errs)
    fit = fit_order(kappas[::-1], errs_arr[::-1])
    rec = record(
        label="网格左端误差对 kappa 的幂律指数",
        theory="不做正则化时误差 ~ ||e||/(sigma_n ||x*||) ~ kappa",
        measured=f"p = {fit.slope:.4f}（R^2 = {fit.r2:.6f}）；误差 {np.array2string(errs_arr, precision=3)}",
        tol="|p - 1| <= 0.05",
        passed=bool(abs(fit.slope - 1.0) <= 0.05),
        kind="C",
    )
    assert rec.passed, rec.measured
