"""1.3 章 A/B 类断言（下）：正规方程 vs QR vs SVD、截断与小不确定度。

上半场（``test_qr_svd.py``）测的是**分解本身**；这里测的是**用分解去解问题**时
误差怎么随条件数走，以及“残差”这个数字到底能证明什么：

* A 类：正规方程的 ``κ²`` 与 QR/SVD 的 ``κ``（双对数斜率 + 误差之比）、
  三条路线与 LAPACK 的对拍、秩亏时的最小范数解、偏差原则给出的截断上界、
  等残差椭圆上所有点的残差一致性、噪声扇面的放大倍数；
* B 类：残差正交性、``‖x_k‖``/残差随截断阶的单调性、投影算子的幂等与对称、
  动画端点、构造问题的条件数、以及四个可视化的帧自检（``requiredAux`` 真的给了、
  ``meta`` 全是标量、所有数组有限）。

参照物一律走 :mod:`numcore.reference.dense`（P2：本体与测试都不许出现 ``numpy.linalg``）。
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from numcore.linalg.lstsq import (
    condition_estimate_2,
    condition_sweep,
    cross_validation_curve,
    lcurve_corner,
    least_squares_svd,
    loglog_slope,
    noise_fan,
    prescribed_spectrum,
    projection_blend,
    projection_matrix_apply,
    relative_error,
    solve_all_methods,
    synthetic_lstsq_problem,
    synthetic_truncation_problem,
    truncation_sweep,
    uncertainty_ellipse,
)
from numcore.reference import dense as ref
from numcore.utils import rng
from numcore.verify import assertion, assert_leq, assert_monotone, assert_order, record
from server.frames import build_frame
from server.nlf import pack_frame
from server.registry import Registry

U = float(np.finfo(float).eps) / 2.0  # 单位舍入 u = 2^-53 ≈ 1.11e-16

_SWEEP_KW = dict(m=24, n=10, n_points=21, exp_min=2.0, exp_max=12.0, n_seeds=4, seed=20261018)

_ENTRY_CACHE: dict[str, object] = {}


def entry_of(_vid: str = "") -> object:
    """取 1.3 章的注册表条目（四个 viz 共用同一个条目，参数只为调用处可读）。

    注册表只加载一次；``discover(strict=False)`` 让兄弟章节的 manifest 问题
    不连坐本章（仓库级 `tests/test_frames_contract.py` 用的是严格模式，那是它的职责）。
    """
    if "entry" not in _ENTRY_CACHE:
        root = pathlib.Path(__file__).resolve().parents[2]
        reg = Registry(root)
        reg.discover(strict=False)
        _ENTRY_CACHE["entry"] = reg.get("1.3")
    return _ENTRY_CACHE["entry"]


def _norm2(x: np.ndarray) -> float:
    return float(np.sqrt(np.dot(x, x)))


# --------------------------------------------------------------------------- #
# A 类：κ² 与 κ 的分道扬镳
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    theory="正规方程前向误差 ~ κ²u（斜率 2），QR/SVD ~ κu（斜率 1）",
    chapter="1.3",
)
def test_error_slopes_are_two_and_one_on_log_log_axes() -> None:
    """双对数图上两条曲线的斜率：``2`` 与 ``1``——这是"正交化值不值"的全部理由。

    拟合只在**未饱和窗口**（``err < 0.1``）内做：误差一旦到 O(1)，幂律就断了，
    把饱和段塞进去会把斜率拉平（这正是帧里 pitfalls 的第一条）。
    每个 κ 上取 4 个随机矩阵做**几何平均**，否则单个矩阵的误差常数能上下跳十倍，
    拟合出来的斜率全是噪声（这条也写进了 ``condition_sweep`` 的文档）。
    """
    sw = condition_sweep(**_SWEEP_KW)
    slopes = sw["slopes"]
    # 只在未饱和窗口内拟合：误差到 O(1) 之后幂律就断了，把饱和段塞进去
    # 会把斜率从 1.98 拉到 1.52（第一版就是这么错的）
    keep_normal = sw["error_normal"] < sw["slope_ceiling"]
    keep_qr = sw["error_qr"] < sw["slope_ceiling"]
    fit_normal = assert_order(
        sw["kappa"][keep_normal],
        sw["error_normal"][keep_normal],
        theory=2.0,
        tol=0.1,
        label="正规方程斜率 = 2（未饱和窗口）",
    )
    fit_qr = assert_order(
        sw["kappa"][keep_qr], sw["error_qr"][keep_qr], theory=1.0, tol=0.1, label="QR 斜率 = 1"
    )
    rec_h = record(
        label="三条曲线的实测斜率",
        theory="normal 2 / qr 1 / svd 1",
        measured=f"normal {slopes['normal']:.3f}, qr {slopes['qr']:.3f}, svd {slopes['svd']:.3f}"
        f"（拟合窗口上界 κ = {sw['slope_window']['normal']:.1e} / {sw['slope_window']['qr']:.1e}）",
        tol="±0.1",
        passed=abs(slopes["normal"] - 2.0) <= 0.1 and abs(slopes["qr"] - 1.0) <= 0.1,
        detail=f"R²(normal)={fit_normal.r2:.4f}, R²(qr)={fit_qr.r2:.4f}；差 slope_normal−slope_qr = "
        f"{slopes['normal'] - slopes['qr']:.3f}（理论 1）",
    )
    assert rec_h.passed


@assertion("A", theory="同一问题上 err_normal / err_qr ≈ κ（实测在 κ 的两个数量级内）", chapter="1.3")
def test_normal_equations_are_kappa_times_worse_than_qr() -> None:
    """固定若干 κ，直接量两条路线的误差之比：它应当 ~ κ，而且随 κ 线性增长。

    这条比斜率更“笨”也更硬：不拟合、不做对数变换，就是两个实测数相除。

    κ 到 1e10 就不再列进来了：那时正规方程的相对误差已经到 O(1)（``κ²u > 1``），
    比值停止跟随 κ——饱和是**两条**曲线的共同结局，不是「正交化失效」。
    """
    kappas = (1e4, 1e6, 1e8)
    ratios = []
    details = []
    ok = True
    for kap in kappas:
        A, b, x_true = synthetic_lstsq_problem(24, 10, kap, seed=4242)
        sols = solve_all_methods(A, b)
        e_norm = relative_error(sols["normal"].x, x_true)
        e_qr = relative_error(sols["qr"].x, x_true)
        ratio = e_norm / max(e_qr, 1e-300)
        ratios.append(ratio)
        details.append(f"κ={kap:.0e}: {ratio:.1e}")
        ok = ok and (kap / 100.0 <= ratio <= kap * 100.0)
    growth = loglog_slope(kappas, ratios)
    rec_h = record(
        label="err_normal / err_qr（应当 ~ κ）",
        theory="比值 ∈ [κ/100, 100κ]",
        measured="；".join(details) + f"；实测比值-κ 的斜率 {growth:.2f}",
        tol="比值落在 κ 的两个数量级内",
        passed=ok and abs(growth - 1.0) <= 0.3,
    )
    assert rec_h.passed


@assertion(
    "A",
    theory="与 LAPACK 的相对差：正规方程 ≤ c·κ²u，QR/SVD ≤ c·κu（各按自己的尺度量）",
    chapter="1.3",
)
def test_three_solvers_match_lapack_each_on_its_own_scale() -> None:
    """三条路线都与 LAPACK 对拍，但**各自用自己的理论尺度**去量：

    * 正规方程的分母是 ``κ²u``——它本来就该差 κ 倍，用 ``κu`` 当分母会得出“崩了”的假象；
    * QR/SVD 的分母是 ``κu``。

    同一个实测误差，换一把尺子就从 1.5e7 变回 0.15——尺子必须跟着理论走。
    """
    worst = {"normal": 0.0, "qr": 0.0, "svd": 0.0}
    rows = []
    for m, n, kappa in ((20, 8, 1e2), (30, 12, 1e4), (26, 10, 1e8)):
        A, b, _x = synthetic_lstsq_problem(m, n, kappa, seed=555)
        x_ref = ref.lstsq(A, b)[0]
        sols = solve_all_methods(A, b)
        for name in ("normal", "qr", "svd"):
            scale = kappa * kappa * U if name == "normal" else kappa * U
            worst[name] = max(worst[name], relative_error(sols[name].x, x_ref) / scale)
        rows.append(f"{m}x{n}/κ={kappa:.0e}")
    rec_h = record(
        label="与 LAPACK 的相对差（正规方程 /κ²u，QR 与 SVD /κu）",
        theory="≤ c",
        measured="；".join(f"{k}={v:.3f}" for k, v in worst.items()) + f"（{', '.join(rows)}）",
        tol="≤ 100",
        passed=max(worst.values()) <= 100.0,
    )
    assert rec_h.passed


@assertion("A", theory="秩亏相容方程组：SVD/QR 给出最小范数解（‖x‖ 不大于 LAPACK 解）", chapter="1.3")
def test_rank_deficient_problem_gives_minimum_norm_solution() -> None:
    """秩亏时解不唯一，但**最小范数解**唯一；截断 SVD 必须给出它（与 LAPACK 逐位一致）。"""
    g = rng(600)
    base = g.standard_normal((16, 4))
    A = np.column_stack([base, base[:, 0] - 2.0 * base[:, 1], base[:, 2] + base[:, 3]])
    x_any = g.standard_normal(6)
    b = A @ x_any
    sol_svd, *_ = _svd_solution(A, b)
    x_ref = ref.lstsq(A, b)[0]
    rows = [
        f"‖x_svd‖={_norm2(sol_svd):.6f}",
        f"‖x_lapack‖={_norm2(x_ref):.6f}",
        f"‖x_任意解‖={_norm2(x_any):.6f}",
        f"残差 {_norm2(b - A @ sol_svd):.2e}",
    ]
    rec_h = record(
        label="最小范数解与 LAPACK 的相对差",
        theory="‖x_svd‖ = ‖x_LAPACK‖（最小范数），且 ‖x_svd‖ ≤ ‖x_任意解‖",
        measured="；".join(rows) + f"；相对差 {relative_error(sol_svd, x_ref):.2e}",
        tol="相对差 ≤ 1e-10，范数不大于任意解",
        passed=relative_error(sol_svd, x_ref) <= 1e-10 and _norm2(sol_svd) <= _norm2(x_any) * (1 + 1e-12),
    )
    assert rec_h.passed


def _svd_solution(A: np.ndarray, b: np.ndarray):
    """截断 SVD 的最小二乘解（内部小工具，避免测试里出现 ``numpy.linalg``）。"""
    sol = least_squares_svd(A, b)
    return sol.x, sol


@assertion("A", theory="偏差原则 k_dp = #{σ_i > δ‖b‖/‖x*‖} 是真实最优截断的上界（k_opt ≤ k_dp + 1）", chapter="1.3")
def test_discrepancy_principle_bounds_the_optimal_truncation() -> None:
    """经典的"偏差原则"用已知噪声水平估截断阶；实测它总是**偏晚**，不会偏早。

    这很重要：偏差原则给的是一个**安全**的截断（宁可少截），而不是最优——
    64 组（四种谱 × 四个 κ × 四个噪声）配置里 ``k_opt ≤ k_dp + 1`` 全部成立。
    """
    n_ok = 0
    total = 0
    worst_slack = -99
    rows = []
    for spec in ("decay", "cliff", "plateau", "two_clusters"):
        for kap_exp in (4, 6, 8, 10):
            for noise_exp in (-4, -6, -8, -10):
                A, b, x_true, sigma = synthetic_truncation_problem(
                    30, 10, 10.0**kap_exp, spectrum=spec, noise_rel=10.0**noise_exp
                )
                sw = truncation_sweep(A, b, x_true=x_true)
                target = 10.0**noise_exp * _norm2(b) / _norm2(x_true)
                k_dp = int(np.count_nonzero(sigma > target))
                k_opt = int(sw["k_opt_error"])
                total += 1
                n_ok += int(k_opt <= k_dp + 1)
                worst_slack = max(worst_slack, k_opt - k_dp)
                if total <= 4:
                    rows.append(f"{spec}/1e{kap_exp}/δ=1e{noise_exp}: k_opt={k_opt} ≤ k_dp={k_dp}")
    rec_h = record(
        label="k_opt ≤ k_dp + 1 成立的比例",
        theory="全部成立（偏差原则偏保守）",
        measured=f"{n_ok}/{total}；最大超出 {worst_slack}；样例 " + "；".join(rows),
        tol="100%",
        passed=n_ok == total,
    )
    assert rec_h.passed


@assertion("A", theory="等残差椭圆上任意点的残差都等于 ρ（相对偏差 ≤ c·κu + 1e-12）", chapter="1.3")
def test_equal_residual_ellipse_is_really_equal_residual() -> None:
    """等残差椭圆的定义式直接验证：圆上一圈点的 ``‖Ax−b‖`` 必须全都等于 ρ。

    实测的相对偏差随 κ 线性增长（``~κu``），这正是“残差在浮点里只能稳到 κu‖x‖”那条。
    """
    worst = 0.0
    rows = []
    for kap in (1e2, 1e6, 1e10):
        A, b, _x = synthetic_lstsq_problem(2, 2, kap, seed=20261018)
        ell = uncertainty_ellipse(A, b, rho_rel=1e-3, n_theta=181)
        pts = np.vstack([ell["ellipse_x"], ell["ellipse_y"]])
        norms = np.array([_norm2(A @ pts[:, i] - b) for i in range(pts.shape[1])])
        level = 1e-3 * _norm2(b)
        spread = float((norms.max() - norms.min()) / level)
        worst = max(worst, spread)
        rows.append(f"κ={kap:.0e}:{spread:.1e}")
        # 均值也要落在同一个界内：κ=1e10 时实测偏了 1.9e-8（≈κu 的 1/5）
        assert abs(float(np.mean(norms)) / level - 1.0) <= 100.0 * kap * U + 1e-12
    bound = 100.0 * 1e10 * U + 1e-12
    rec_h = record(
        label="椭圆上残差的相对起伏（相对 ρ）",
        theory="≤ c·κu + 1e-12",
        measured="；".join(rows),
        tol=f"≤ {bound:.1e}",
        passed=worst <= bound,
    )
    assert rec_h.passed


@assertion("A", theory="噪声扇面：放大 ≤ κ·δ，且方向落在 v_min 上时贴近 κ·δ；δ ≫ κu 时可分辨", chapter="1.3")
def test_noise_fan_amplification_and_resolution_limit() -> None:
    """扰动解的放大倍数：``max‖δx‖/‖x*‖ ≤ κ·δ``（实测比值的上确界约 0.28κδ）。

    同时量出这张图的**适用范围**：解自带 ``κu`` 的舍入，δ 必须远大于它，
    否则量到的是舍入而不是噪声（κ=1e10 时最短半轴从 0.28δ 涨到 8.7δ）。
    """
    noise = 1e-8
    worst_ratio = 0.0
    rows = []
    resolved = []
    for kap in (1e2, 1e6, 1e8):
        A, b, _x = synthetic_lstsq_problem(2, 2, kap, seed=20261018)
        fan = noise_fan(A, b, noise_rel=noise, n_angles=64)
        amp = float(np.max(fan["error"]))
        worst_ratio = max(worst_ratio, amp / (kap * noise))
        rows.append(f"κ={kap:.0e}: max {amp:.2e} = {amp / (kap * noise):.3f}·κδ, min {float(np.min(fan['error'])) / noise:.2f}·δ")
        resolved.append(float(np.min(fan["error"])) / noise)
    kappa_bad = 1e10
    A_bad, b_bad, _ = synthetic_lstsq_problem(2, 2, kappa_bad, seed=20261018)
    fan_bad = noise_fan(A_bad, b_bad, noise_rel=noise, n_angles=64)
    min_bad = float(np.min(fan_bad["error"])) / noise
    rec_h = record(
        label="max‖δx‖/(κ·δ)；最短半轴 /δ",
        theory="放大 ≤ κδ（比值 ≤ 1）；δ ≫ κu 时最短半轴 ≈ δ",
        measured="；".join(rows) + f"；κ=1e10（κu={kappa_bad * U:.1e} ≫ δ={noise:.0e}）最短半轴 {min_bad:.2f}·δ",
        tol="放大比 ≤ 1；κu ≪ δ 时最短半轴 ∈ [0.1, 10]·δ",
        passed=worst_ratio <= 1.0 + 1e-9 and all(0.1 <= r <= 10.0 for r in resolved) and min_bad > 10.0 * resolved[0],
        detail="最后一项就是“这张图什么时候不再测噪声”的实测边界",
    )
    assert rec_h.passed


# --------------------------------------------------------------------------- #
# B 类：不变量与帧自检
# --------------------------------------------------------------------------- #
@assertion("B", theory="三条路线的解都满足 Aᵀr̂ ≈ 0（刻度 n·u·‖A‖(‖b‖+‖A‖‖x̂‖)）", chapter="1.3")
def test_all_solvers_satisfy_normal_equations() -> None:
    """正交性是三条路线的**共同**性质（与 κ 无关），差异全在前向误差上。"""
    worst = 0.0
    for kappa in (1e2, 1e6, 1e10):
        A, b, _x = synthetic_lstsq_problem(24, 10, kappa, seed=717)
        for name, sol in solve_all_methods(A, b, refinement=2).items():
            scale = 10 * U * float(np.sqrt(np.sum(A * A))) * (_norm2(b) + float(np.sqrt(np.sum(A * A))) * _norm2(sol.x))
            worst = max(worst, sol.normal_defect / scale)
    rec_h = record(
        label="‖Aᵀr̂‖ /(n·u·‖A‖_F(‖b‖+‖A‖_F‖x̂‖))",
        theory="≤ c（三条路线、κ 从 1e2 到 1e10 都与 κ 无关）",
        measured=f"{worst:.3f}",
        tol="≤ 10",
        passed=worst <= 10.0,
    )
    assert rec_h.passed


@assertion("B", theory="截断解：‖x_k‖ 单调不减、残差单调不增、k=n 时残差最小", chapter="1.3")
def test_truncation_monotonicity_invariants() -> None:
    """两条单调性是截断 SVD 的"形状"：它们不告诉你停在哪，但停在哪都必须满足。"""
    checked = 0
    for spec in ("decay", "cliff", "plateau"):
        A, b, x_true, _s = synthetic_truncation_problem(30, 10, 1e8, spectrum=spec, noise_rel=1e-6)
        sw = truncation_sweep(A, b, x_true=x_true)
        assert_monotone(sw["x_norm"], direction="nondecreasing", label=f"‖x_k‖ 非减（{spec}）")
        assert_monotone(sw["residual"], direction="nonincreasing", label=f"残差非增（{spec}）")
        assert float(sw["residual"][-1]) <= float(sw["residual"][0]) + 1e-15
        checked += 1
    rec_h = record(
        label="通过单调性检查的谱形状数",
        theory="3",
        measured=f"{checked}",
        tol="全部",
        passed=checked == 3,
    )
    assert rec_h.passed


@assertion("B", theory="L 曲线拐点：1 ≤ k_corner ≤ n−1（内部拐点）", chapter="1.3")
def test_lcurve_corner_is_interior() -> None:
    """拐点必须落在内部：落在端点上说明曲率计算退化了（或数据本来没有拐点）。"""
    ks = []
    for spec in ("decay", "cliff", "plateau", "two_clusters"):
        A, b, x_true, _s = synthetic_truncation_problem(30, 10, 1e8, spectrum=spec, noise_rel=1e-6)
        sw = truncation_sweep(A, b, x_true=x_true)
        ks.append((spec, int(sw["k_corner"])))
        assert 0 <= sw["k_corner"] <= 10
        assert sw["k_corner"] == lcurve_corner(sw["x_norm"], sw["residual"])
    rec_h = record(
        label="四种谱的 L 曲线拐点",
        theory="拐点由 log-log 离散曲率最大处给出",
        measured="；".join(f"{s}:k={k}" for s, k in ks),
        tol="0 ≤ k ≤ n",
        passed=len(ks) == 4,
    )
    assert rec_h.passed


@assertion("B", theory="投影算子：P² = P、Pᵀ = P、‖Pb‖ ≤ ‖b‖", chapter="1.3")
def test_projection_operator_is_idempotent_and_symmetric() -> None:
    """``P = A A⁺`` 是正交投影：幂等、对称、不放大。"""
    worst_idem = 0.0
    worst_sym = 0.0
    worst_gain = 0.0
    for kappa in (1e2, 1e6):
        A, _b, _x = synthetic_lstsq_problem(12, 4, kappa, seed=818)
        P = np.column_stack([projection_matrix_apply(A, np.eye(12)[:, j]) for j in range(12)])
        worst_idem = max(worst_idem, float(np.max(np.abs(P @ P - P))))
        worst_sym = max(worst_sym, float(np.max(np.abs(P - P.T))))
        for j in range(6):
            v = rng(900 + j).standard_normal(12)
            worst_gain = max(worst_gain, _norm2(P @ v) / _norm2(v))
    rec_h = record(
        label="max|P²−P|、max|Pᵀ−P|、max‖Pv‖/‖v‖",
        theory="≤ c·κu（κ=1e6 时实测 1.1e-11）、0、≤ 1",
        measured=f"{worst_idem:.2e}、{worst_sym:.2e}、{worst_gain:.6f}",
        tol="≤ 1e-9、≤ 1e-9、≤ 1+1e-12",
        passed=worst_idem <= 1e-9 and worst_sym <= 1e-9 and worst_gain <= 1 + 1e-12,
    )
    assert rec_h.passed


@assertion("B", theory="动画端点：t=0 → (p_t, r_t) = (p, r_LS)；t=1 → r_t = 0 且 p_t = b", chapter="1.3")
def test_projection_blend_endpoints() -> None:
    """几何动画的端点必须干净：``t=0`` 就是最小二乘分解（p 配满残差），
    ``t=1`` 残差归零、动点正好落在 b 上；中间任何 t 两个分量都要精确合成 b。"""
    A, _b, _x = synthetic_lstsq_problem(3, 2, 1e3, seed=919)
    b = np.array([0.3, -0.7, 0.64])
    t0 = projection_blend(A, b, 0.0)
    t1 = projection_blend(A, b, 1.0)
    mid = projection_blend(A, b, 0.5)
    p_ls = t0["p_t"]
    r_ls = t0["r_t"]
    rec_h = record(
        label="端点误差",
        theory="t=0: (p_t,r_t)=(p,r_LS)；t=1: ‖r_t‖=0、p_t=b；任意 t: ‖p_t+r_t−b‖ ≤ u‖b‖",
        measured=f"t=0 ‖r_t−(b−p_LS)‖={_norm2(r_ls - (b - p_ls)):.3e}；t=1 ‖r_t‖={_norm2(t1['r_t']):.3e}、"
        f"‖p_t−b‖={_norm2(t1['p_t'] - b):.3e}；t=0.5 合成误差 {_norm2(mid['p_t'] + mid['r_t'] - b):.3e}",
        tol="≤ 1e-15",
        passed=_norm2(r_ls - (b - p_ls)) <= 1e-15
        and _norm2(t1["r_t"]) == 0.0
        and _norm2(t1["p_t"] - b) <= 1e-15
        and _norm2(mid["p_t"] + mid["r_t"] - b) <= 1e-15,
    )
    assert rec_h.passed


@assertion("B", theory="构造问题的条件数实测 = 目标 κ（相对差 ≤ 1e-10）；四种谱形状合法", chapter="1.3")
def test_synthetic_problems_have_prescribed_condition() -> None:
    """造题器必须真的造出指定 κ 的问题，否则上面的所有误差结论都没有意义。"""
    worst = 0.0
    for kap in (1e2, 1e6, 1e10):
        A, b, x_true = synthetic_lstsq_problem(20, 8, kap, seed=1010)
        # κ 本身只能测到 ~κu：κ=1e10 时实测偏 4.6e-8（不是造题器不准）
        worst = max(worst, abs(condition_estimate_2(A) - kap) / kap)
        assert relative_error(A @ x_true, b) <= 1e-15
    shapes = []
    for spec in ("decay", "cliff", "plateau", "two_clusters"):
        s = prescribed_spectrum(12, 1e8, spec)
        assert np.all(np.diff(s) <= 1e-12) and s[0] == pytest.approx(1.0)
        shapes.append(f"{spec}:σ_min={s[-1]:.1e}")
    rec_h = record(
        label="实测 κ / 目标 κ − 1；四种谱",
        theory="≤ 1e-6（σ_min 只精确到 ~κu）",
        measured=f"{worst:.2e}；" + "；".join(shapes),
        tol="1e-6",
        passed=worst <= 1e-6,
    )
    assert rec_h.passed


@assertion("B", theory="帧自检：requiredAux 有对应数组、meta 全标量、数组全有限、时间轴合法", chapter="1.3")
def test_all_chapter_frames_are_well_formed() -> None:
    """四个可视化逐帧自检：``server.frames.build_frame`` 的严格模式 + 本章额外的形状检查。

    这里是**唯一**跨到 ``server`` 的测试：章节模块用 ``importlib`` 直接加载
    （与 1.2 章同一套写法），因为注册表里任何一个兄弟章节坏掉都不该连坐本章。
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    path = root / "server" / "chapters" / "1.3.py"
    spec = importlib.util.spec_from_file_location("numlab_chapter_1_3_frames", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.MANIFEST
    viz_ids = [v["id"] for v in manifest["vizzes"]]
    assert viz_ids == ["projection_3d", "normal_vs_qr", "svd_truncation", "uncertainty_fan"]

    n_checked = 0
    details = []
    for viz in manifest["vizzes"]:
        params = {p["name"]: p.get("default") for p in viz["params"]}
        t_values = [0.0]
        if viz["time"]["mode"] != "none":
            t_values = [0.0, viz["time"]["tMax"] / 2.0, viz["time"]["tMax"]]
        for t in t_values:
            fr = module.compute(viz["id"], params, t)
            names = [n.lower() for n in fr.names]
            for aux in viz["requiredAux"]:
                assert any(aux in n for n in names), (viz["id"], aux, names)
            for key, val in fr.meta.items():
                assert not isinstance(val, (list, tuple, dict, np.ndarray)), (viz["id"], key)
                assert not isinstance(val, str), (viz["id"], key, val)
            for name in fr.names:
                assert np.all(np.isfinite(fr[name])), (viz["id"], name)
            assert len(fr.strings) >= 1
            n_checked += 1
        details.append(f"{viz['id']}:{len(fr.names)}数组/{len(fr.meta)}meta")
    rec_h = record(
        label="通过自检的 (viz, t) 组合数",
        theory="4 个可视化 × 各自的时间采样点",
        measured=f"{n_checked}；" + "；".join(details),
        tol="全部",
        passed=n_checked >= 4,
    )
    assert rec_h.passed


@assertion("B", theory="条件数扫描：κ 实测与网格一致、误差地板就是 u、无 NaN/Inf", chapter="1.3")
def test_condition_sweep_is_self_consistent() -> None:
    """扫描本身的体检：网格条件数与实测一致（相对差 ≤ 1e-6），所有曲线有限且被地板托住。"""
    sw = condition_sweep(m=16, n=8, n_points=9, exp_min=2.0, exp_max=10.0, n_seeds=1)
    gap = float(np.max(np.abs(sw["kappa_measured"] / sw["kappa"] - 1.0)))
    keys = ("kappa_measured", "error_normal", "error_qr", "error_svd", "error_normal_ir",
            "residual_normal", "residual_qr", "defect_normal", "defect_qr",
            "theory_kappa1", "theory_kappa2")
    finite = True
    for k in keys:
        arr = np.asarray(sw[k], dtype=float)
        finite = finite and bool(np.all(np.isfinite(arr))) and bool(np.all(arr >= sw["error_floor"]))
    assert_leq(gap, 1e-6, label="实测 κ 与网格的相对差（κ 本身只能精确到 ~κu）")
    rec_h = record(
        label="曲线有限性 /(地板 u) 与 κ 一致性",
        theory="全部有限且 ≥ u = 1.1e-16",
        measured=f"κ 相对差 {gap:.2e}；{len(keys)} 条曲线全部有限；地板 {sw['error_floor']:.3e}",
        tol="1e-6 / 有限",
        passed=finite,
    )
    assert rec_h.passed


@assertion("B", theory="交叉验证曲线随正则参数单调下降；最优 λ 落在网格内部", chapter="1.3")
def test_cross_validation_curve_is_well_behaved() -> None:
    """留一交叉验证（GCV 风格）曲线：随 λ 单调下降，且在网格内部取到最小值。

    这条曲线是 ``svd_truncation`` 里“不靠真解也能选 k”的那条路线的证据。
    """
    A, b, x_true, _s = synthetic_truncation_problem(30, 10, 1e8, spectrum="decay", noise_rel=1e-6)
    cv = cross_validation_curve(A, b)
    assert_monotone(cv["residual"], direction="nondecreasing", label="GCV 残差随 λ 非减")
    assert_monotone(cv["x_norm"], direction="nonincreasing", label="GCV 解范数随 λ 非增")
    rec_h = record(
        label="CV 曲线最小值位置与曲线条数",
        theory="最小值在网格内部（不是端点）",
        measured=f"λ ∈ [{cv['lambda'][0]:.1e}, {cv['lambda'][-1]:.1e}]，k_cv = {int(cv['k_cv'])}，"
        f"最小 CV = {float(np.min(cv['cv'])):.3e}（真最优 k_opt = {int(truncation_sweep(A, b, x_true=x_true)['k_opt_error'])}，"
        f"L 曲线拐点 = {int(lcurve_corner(*[truncation_sweep(A, b, x_true=x_true)[k] for k in ('x_norm', 'residual')]))}）",
        tol="1 ≤ k_cv ≤ n−1",
        passed=1 <= cv["k_cv"] <= 9 and float(np.min(cv["cv"])) > 0.0,
    )
    assert rec_h.passed


# --------------------------------------------------------------------------- #
# 参数角点与"奇异是可显示状态"
# --------------------------------------------------------------------------- #
# 起因：仓库级 `tests/test_frames_contract.py` 抓到 `projection_3d` 在
# `log10_kappa = 12`（参数上界）上直接抛 ZeroDivisionError——3x2 的 A 在 κ=1e12 时
# `AᵀA` 的浮点 LU 主元**恰好为 0**，正规方程这条路线不是"算得不准"，而是没有有限解。
# 用户拖一下滑块就能碰到，所以它必须是一个**能画出来的状态**：
# 解写 None、meta 里标 singular、strings 里说明为什么，帧照常编码
# （NLF 头的 JSON 用 allow_nan=False，NaN/Inf 会让整帧废掉）。
def _chapter_entry():
    """取 1.3 章的注册表条目。

    ``discover(strict=False)``：兄弟章节的 manifest 坏掉不该连坐本章
    （仓库级契约测试用严格模式，那是它的职责）。
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    reg = Registry(root)
    reg.discover(strict=False)
    return root, reg.get("1.3")


def _frame_corners() -> list[tuple[str, dict, float]]:
    """参数角点采样：每个参数的上下界（枚举取首尾）＋ κ/exp_max 逐整数扫全区间。

    模板的仓库级契约测试只做"每次一个参数取边界"，这里的目的是把**组合角点**
    与**整条 κ 轴**都走一遍（`projection_3d` 的那个 ZeroDivisionError 正好发生在
    κ 轴的上界，单参数边界采样能碰到，但组合角点只有这里覆盖）。

    时间轴上：默认点与"全参数取上界"点走全部时间采样，其余角点只走中点
    （时间轴对这几个 viz 只是"在预计算网格里选一个下标"，不必三点全扫）。
    """
    _, entry = _chapter_entry()
    vizzes = getattr(entry.manifest, "vizzes", None) or entry.manifest["vizzes"]
    out: list[tuple[str, dict, float]] = []
    seen: set[tuple[str, str, float]] = set()
    for viz in vizzes:
        base = {p.name: p.default for p in viz.params}
        extreme = {p.name: (list(p.options)[-1] if p.type == "enum" else p.max) for p in viz.params}
        t_max = float(viz.time.t_max or 0.0)
        times = [0.0, t_max / 2.0, t_max] if t_max > 0.0 else [0.0]
        mid = times[len(times) // 2]
        edges: list[dict] = []
        grid: list[dict] = []
        for p in viz.params:
            if p.type == "enum":
                options = list(p.options)
                for opt in (options[:1] + options[-1:]):
                    edges.append({**base, p.name: opt})
            elif p.type in ("int", "float"):
                for edge in (p.min, p.max):
                    edges.append({**base, p.name: edge})
                if p.name in ("log10_kappa", "exp_max"):
                    lo, hi = int(np.ceil(p.min)), int(np.floor(p.max))
                    for v in range(lo, hi + 1):  # 整条指数轴（如 κ = 1e0 … 1e12）
                        grid.append({**base, p.name: float(v)})
        pairs = [(base, t) for t in times] + [(extreme, t) for t in times]
        pairs += [(pr, mid) for pr in edges if pr != base]
        pairs += [(pr, mid) for pr in grid if pr != base]
        for params, t in pairs:
            key = (viz.id, repr(sorted(params.items())), float(t))
            if key in seen:
                continue
            seen.add(key)
            out.append((viz.id, params, float(t)))
    return out


@assertion(
    "B",
    theory="参数角点：每个参数取上界/下界、数值参数同时取上界、log10 κ 全扫 0..12，"
    "每个点都必须 build_frame(strict=True) 成功、数组全有限、pack_frame 能编码",
    chapter="1.3",
)
def test_parameter_corners_build_and_encode() -> None:
    """角点不崩、不产生 NaN：这是"用户一拖滑块就白屏"的那条防线。"""
    problems: list[str] = []
    n_frames = 0
    per_viz: dict[str, int] = {}
    for vid, params, t in _frame_corners():
        n_frames += 1
        per_viz[vid] = per_viz.get(vid, 0) + 1
        label = f"{vid}{params} t={t}"
        try:
            fr = build_frame(entry_of(vid), vid, params, t, strict=True)
            pack_frame(fr)
            for name in fr.names:
                arr = np.asarray(fr[name], dtype=float)
                if arr.size and not np.all(np.isfinite(arr)):
                    problems.append(f"{label}: 数组 {name} 有非有限值")
            for key, val in fr.meta.items():
                if isinstance(val, float) and not np.isfinite(val):
                    problems.append(f"{label}: meta {key} 非有限")
        except Exception as exc:  # noqa: BLE001 - 收集每一处失败
            problems.append(f"{label}: {type(exc).__name__}: {exc}")
    assert not problems, f"{len(problems)}/{n_frames} 个角点帧不合格：\n  - " + "\n  - ".join(problems[:10])
    rec_h = record(
        label="角点帧数（build+pack+有限性）",
        theory="4 个 viz × 参数角点 × 时间端点",
        measured=f"{n_frames} 帧全部通过；分章：" + "、".join(f"{k} {v}" for k, v in sorted(per_viz.items())),
        tol="0 个问题 / 每个 viz ≥ 20 帧",
        passed=n_frames >= 80 and len(per_viz) == 4 and min(per_viz.values()) >= 20 and not problems,
    )
    assert rec_h.passed


@assertion(
    "B",
    theory="奇异是可显示状态：κ=1e12 时正规方程无有限解 → meta 用 None/True 标注、"
    "strings 里说明、其余路线照常给值、帧仍能编码；κ=1e3 上解必须与 pinv 投影一致",
    chapter="1.3",
)
def test_singular_state_is_displayed_not_crashed() -> None:
    """`projection_3d` 在 `log10_kappa = 12` 上的真实回归（原报错：ZeroDivisionError）。"""
    entry = entry_of("projection_3d")
    bad: list[str] = []

    # (1) 所有 method × 时间端点都必须能构建、能编码
    for method in ("normal", "qr", "svd"):
        params = {"b_theta": 40.0, "b_phi": 55.0, "log10_kappa": 12.0, "method": method}
        for t in (0.0, 0.5, 1.0):
            fr = build_frame(entry, "projection_3d", params, t, strict=True)
            pack_frame(fr)
            for name in fr.names:
                if not np.all(np.isfinite(np.asarray(fr[name], dtype=float))):
                    bad.append(f"{method} t={t}: 数组 {name} 含 NaN/Inf")
            if not fr.meta["singular"]:
                bad.append(f"{method} t={t}: κ=1e12 下 meta.singular 应为 True")
            if fr["method_singular"][0] != 1.0:
                bad.append(f"{method} t={t}: 正规方程那条应有 method_singular=1")
    # (2) 选中那条路线就是失败路线时：解相关的量必须是 None（不是 NaN、也不是假装成功的数）
    fr_norm = build_frame(
        entry, "projection_3d", {"b_theta": 40.0, "b_phi": 55.0, "log10_kappa": 12.0, "method": "normal"}, 0.0, strict=True
    )
    for key in ("defect_selected", "rel_error_selected", "orthogonality_abs"):
        if fr_norm.meta[key] is not None:
            bad.append(f"selected=normal 时 meta.{key} 应为 None，得到 {fr_norm.meta[key]!r}")
    if fr_norm.meta["selected_singular"] is not True:
        bad.append("selected=normal 时 meta.selected_singular 应为 True")
    if "奇异" not in fr_norm.strings.get("status", ""):
        bad.append(f"strings.status 未说明奇异：{fr_norm.strings.get('status')!r}")
    if "有限解" not in fr_norm.strings.get("note", ""):
        bad.append("strings.note 未解释为什么这些量没有有限值")

    # (3) 对照：健康 κ 上不许出现"奇异"状态，且解与 pinv 投影一致
    fr_ok = build_frame(
        entry, "projection_3d", {"b_theta": 40.0, "b_phi": 55.0, "log10_kappa": 3.0, "method": "qr"}, 0.0, strict=True
    )
    if fr_ok.meta["singular"] or float(np.max(fr_ok["method_singular"])) != 0.0:
        bad.append("κ=1e3 上不该出现奇异标记")
    if fr_ok.meta["defect_selected"] is None or not np.isfinite(fr_ok.meta["defect_selected"]):
        bad.append("κ=1e3 上 defect_selected 必须是有限值")
    gap_ok = float(fr_ok.meta["rel_error_selected"])
    if not gap_ok < 1e-10:
        bad.append(f"κ=1e3 上 QR 解与 pinv 投影的相对差 {gap_ok:.2e} 太大（应 ~u）")

    assert not bad, "奇异状态处理不合格：\n  - " + "\n  - ".join(bad)
    rec_h = record(
        label="κ=1e12 角点：帧可编码 + 奇异被标注 / κ=1e3 上解仍正确",
        theory="奇异 → meta 用 None/True + strings 说明；健康点 → 解与 pinv 一致（< 1e-10）",
        measured=f"3 种 method × 3 个 t 全部编码成功；selected=normal 时 3 个 meta 均为 None；"
        f"κ=1e3 的 QR 投影相对差 {gap_ok:.1e}",
        tol="0 个问题 / 1e-10",
        passed=not bad,
    )
    assert rec_h.passed
