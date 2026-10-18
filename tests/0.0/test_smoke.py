"""0.0 章：三类数值断言（其它 27 章的样板）。

* **A 类（解析对拍）**：实测收敛阶必须等于理论阶 ±0.1；
* **B 类（不变量）**：守恒量漂移不超过机器精度界；
* **C 类（收敛率/阈值）**：稳定阈值必须**二分实测**，再与理论值比对。

特别值得看的是 C 类的第二条：有限网格上的实测阈值是 ``1/(2sin²(θ_max/2)) = 0.503097…``，
**不是**书上那句"r ≤ 1/2"——两者只在连续极限下相等。照着书抄常数就会把这条断言写成
"自己验证自己"，因此这里连"向 1/2 收敛"这件事也一起测了。
"""

from __future__ import annotations

import numpy as np
import pytest

from numcore.smoke import (
    damped_oscillator,
    ftcs_amplification,
    ftcs_critical_r,
    ftcs_critical_r_limit,
    ftcs_growth,
    ftcs_stable,
    heat_exact_sine,
    heat_ftcs,
    heat_run_meta,
    initial_profile,
    oscillator_energy,
    oscillator_ode_residual,
)
from numcore.verify import (
    assertion,
    record,
    assert_conserved,
    assert_leq,
    assert_order,
    assert_threshold,
    diagnose_order,
    find_threshold,
    fit_order,
)


# --------------------------------------------------------------------------- #
# A 类：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="空间二阶精度", chapter="0.0")
def test_ftcs_spatial_order_is_two() -> None:
    """固定 r，加密网格：L2 误差应当以 O(Δx²) 下降（理论阶 2）。"""
    nxs = [11, 21, 41, 81]
    errs = []
    for nx in nxs:
        run = heat_ftcs(nx=nx, alpha=1.0, t_end=0.05, r=0.4, k=1)
        errs.append(run.l2_error)
    fit = assert_order([1.0 / nx for nx in nxs], errs, theory=2.0, tol=0.1, label="FTCS 空间收敛阶")
    diag = diagnose_order([1.0 / nxs_ for nxs_ in nxs], errs)
    assert not diag.floored, f"误差贴到舍入底了，窗口选取有问题：{diag}"
    assert fit.r2 > 0.99


@assertion("A", theory="中心差分残差 O(h²)", chapter="0.0")
def test_oscillator_residual_scales_as_h_squared() -> None:
    """解析解代回方程的中心差分残差应当以 O(h²) 缩小（这一步同时验证 x 与 v 自洽）。"""
    # 窗口选择有讲究：h ≲ 1e-3 之后残差里"两个大项相减"的舍入开始主导（r/h² 漂移），
    # 于是必须用 head= 只保留粗端。把"用了 head 才拿到正确阶数"这件事本身也测进去。
    t = np.array([1.0])
    all_h = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3, 5e-4, 2.5e-4])
    all_res = np.array([float(np.max(oscillator_ode_residual(t, omega=1.3, zeta=0.2, h=h))) for h in all_h])
    diag = diagnose_order(all_h, all_res)
    assert diag.noisy or diag.floored, f"这段窗口本应显示窗口效应，否则测试失去意义：{diag}"
    assert_order(all_h[:4], all_res[:4], theory=2.0, tol=0.05, label="残差的差分截断阶（head=4）")
    assert abs(fit_order(all_h, all_res, head=4).slope - 2.0) < 0.05


@assertion("A", theory="欠阻尼包络 e^(-ζωt)", chapter="0.0")
def test_underdamped_envelope_matches_theory() -> None:
    """解析解的局部极大值应当落在理论包络上（比值 → 1）。"""
    omega, zeta = 2.0, 0.1
    tt = np.linspace(0.0, 10.0, 20001)
    x, _ = damped_oscillator(tt, omega=omega, zeta=zeta)
    # 取相邻极值点：x' = 0 的位置用符号变化定位
    v = np.gradient(x, tt)
    idx = np.where(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
    assert idx.size >= 3
    peaks = np.abs(x[idx])
    envelope = np.abs(np.exp(-zeta * omega * tt[idx]))
    ratio = peaks / envelope
    worst = float(np.max(np.abs(ratio - 1.0)))
    assert np.all(ratio <= 1.0 + 1e-6)
    assert worst < 0.02
    record(
        label="极值点 / 理论包络 − 1",
        theory=0.0,
        measured=worst,
        tol="< 0.02",
        passed=True,
        detail=f"{peaks.size} 个极值点，最大相对偏差 {worst:.3e}",
        kind="A",
    )


@assertion("A", theory="临界阻尼是欠阻尼的极限", chapter="0.0")
def test_critical_damping_is_the_limit_of_underdamped() -> None:
    """ζ → 1⁻ 的欠阻尼解应当收敛到临界阻尼分支（同一点逐值比较）。"""
    tt = np.linspace(0.0, 5.0, 101)
    x_crit, _ = damped_oscillator(tt, omega=1.0, zeta=1.0)
    worst = 0.0
    for eps in (1e-4, 1e-6):
        x_near, _ = damped_oscillator(tt, omega=1.0, zeta=1.0 - eps)
        worst = max(worst, float(np.max(np.abs(x_near - x_crit))) / eps)
    assert worst < 50.0
    record(
        label="‖x(ζ=1−ε) − x(ζ=1)‖∞ / ε",
        theory="≤ 50（ε→0 时应当有界）",
        measured=worst,
        tol="< 50",
        passed=True,
        detail=f"ε = 1e-4 与 1e-6，最大比值 {worst:.4f}",
        kind="A",
    )


# --------------------------------------------------------------------------- #
# B 类：不变量
# --------------------------------------------------------------------------- #
@assertion("B", theory="ζ=0 时比能量严格守恒", chapter="0.0")
def test_energy_conserved_without_damping() -> None:
    tt = np.linspace(0.0, 50.0, 4001)
    x, v = damped_oscillator(tt, omega=1.0, zeta=0.0)
    assert_conserved(oscillator_energy(x, v, omega=1.0), quantity="比能量", label="无阻尼能量守恒")


@assertion("B", theory="离散极值原理：r ≤ r* 时 max|u| 不增", chapter="0.0")
def test_discrete_maximum_principle() -> None:
    """三角帽初值下，稳定区间内解的无穷范数不应超过初值范数。"""
    run = heat_ftcs(nx=41, t_end=0.2, r=0.4, initial="hat")
    assert_leq(float(np.max(np.abs(run.u))), 1.0, label="离散极值原理", rtol=1e-12)


@assertion("B", theory="初值与边界条件", chapter="0.0")
def test_heat_boundary_and_initial_conditions() -> None:
    run = heat_ftcs(nx=31, t_end=0.05, r=0.3, initial="sine")
    assert run.u[0] == 0.0 and run.u[-1] == 0.0
    assert_leq(float(np.max(np.abs(run.u))), 1.0, label="解不超过初值", rtol=1e-12)


@assertion("B", theory="对称性：单模态解关于 x=1/2 对称", chapter="0.0")
def test_heat_solution_stays_symmetric() -> None:
    """u(x,t) = −u(1−x) 是错的（那是奇对称）；sin(πx) 关于 x=1/2 **对称**，格式必须保持它。"""
    run = heat_ftcs(nx=41, t_end=0.05, r=0.4, initial="sine")
    assert_leq(float(np.max(np.abs(run.u - run.u[::-1]))), 1e-15, label="关于 x=1/2 的对称性", rtol=1e-12)


# --------------------------------------------------------------------------- #
# C 类：阈值（必须实测）
# --------------------------------------------------------------------------- #
@assertion("C", theory="有限网格阈值 1/(2sin²(θmax/2))", chapter="0.0")
def test_ftcs_stability_threshold_measured() -> None:
    """对归一化幂迭代实测的增长比做二分，得到阈值后与**有限网格**理论值比对。"""

    def unstable(r: float) -> bool:
        return ftcs_growth(r, nx=21) > 1.0 + 1e-6

    result = find_threshold(unstable, 0.4, 0.6, assume="false_at_lo", tol=1e-7, note="幂迭代测 |λ|max")
    assert_threshold(result, ftcs_critical_r(21), rtol=5e-4, label="FTCS 实测稳定阈值（nx=21）")
    assert result.iterations < 60


@assertion("C", theory="阈值随 nx 收敛到 1/2", chapter="0.0")
def test_ftcs_threshold_converges_to_continuum_limit() -> None:
    """实测阈值与 1/2 的偏差应当以 O(Δx²) 缩小——这才是"r ≤ 1/2"的正确用法。"""
    nxs = [11, 21, 41, 81]
    gaps = []
    for nx in nxs:
        def unstable(r: float, nx: int = nx) -> bool:
            # n_steps=None：让每次测量自己迭代到收敛——阈值附近 |λ₁| 与 |λ₂| 只差 1e-7，
            # 固定步数的幂迭代根本分辨不出来（这是本文件里最容易踩的坑）。
            return ftcs_growth(r, nx=nx) > 1.0 + 1e-9

        result = find_threshold(unstable, 0.4, 0.7, assume="false_at_lo", tol=1e-8)
        gaps.append(abs(result.threshold - ftcs_critical_r_limit()))
    fit = assert_order(
        [1.0 / nx for nx in nxs],
        gaps,
        theory=2.0,
        tol=0.2,
        label="阈值偏差的收敛阶",
    )
    assert fit.slope > 1.5


@assertion("C", theory="解析判据与实测判据一致", chapter="0.0")
def test_analytic_and_measured_stability_agree() -> None:
    """在阈值两侧各取若干点，解析判据 ``ftcs_stable`` 与实测 ``|λ|max`` 必须同号。"""
    nx = 21
    r_star = ftcs_critical_r(nx)
    fracs = (0.7, 0.9, 0.99, 1.01, 1.1, 1.3)
    agree = 0
    detail = []
    for frac in fracs:
        r = r_star * frac
        analytic = ftcs_stable(r, nx)
        measured = ftcs_growth(r, nx=nx) <= 1.0 + 1e-9
        detail.append(f"r/r*={frac}: 解析={analytic} 实测={measured}")
        assert measured == analytic, f"r={r}：解析={analytic} 实测={measured}"
        agree += 1
    record(
        label="解析判据与实测判据一致点数",
        theory=len(fracs),
        measured=agree,
        tol="全等",
        passed=True,
        detail="；".join(detail),
        kind="C",
    )


# --------------------------------------------------------------------------- #
# 数值内核自身的边界
# --------------------------------------------------------------------------- #
def test_amplification_formula_matches_power_iteration_on_single_mode() -> None:
    """单模态一步放大：直接套公式与真的走一步必须一致。"""
    nx = 31
    r = 0.35
    x = np.linspace(0.0, 1.0, nx)
    for mode in (1, 3, 7):
        u = initial_profile(x, "sine", mode)
        w = u.copy()
        w[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        w[0] = w[-1] = 0.0
        got = float(w[mode]) / float(u[mode])
        want = ftcs_amplification(r, mode, nx)
        assert abs(got - want) < 1e-12


def test_boundary_case_guards() -> None:
    with pytest.raises(ValueError, match="nx 至少"):
        heat_ftcs(nx=4)
    with pytest.raises(ValueError, match="alpha"):
        heat_ftcs(nx=11, alpha=0.0)
    with pytest.raises(ValueError, match="r 必须为正"):
        heat_ftcs(nx=11, r=0.0)
    with pytest.raises(ValueError, match="未知初值"):
        initial_profile(np.linspace(0, 1, 5), "gauss")
    with pytest.raises(ValueError, match="omega"):
        damped_oscillator([0.0], omega=0.0)
    with pytest.raises(ValueError, match="zeta"):
        damped_oscillator([0.0], zeta=-0.1)
    with pytest.raises(ValueError, match="mode"):
        ftcs_amplification(0.4, 0, 11)
    with pytest.raises(ValueError, match="nx 至少"):
        ftcs_growth(0.4, nx=3)


def test_heat_run_meta_is_scalar_only() -> None:
    run = heat_ftcs(nx=21, t_end=0.02, r=0.4)
    meta = heat_run_meta(run)
    assert set(meta) == {"dt", "n_steps", "r", "max_error", "l2_error"}
    for key, val in meta.items():
        assert isinstance(val, (int, float)) and not isinstance(val, bool), key
    assert run.n_steps == int(np.ceil(0.02 / run.dt))
    assert run.max_error >= run.l2_error > 0.0


def test_heat_exact_sine_decays_by_known_factor() -> None:
    x = np.linspace(0.0, 1.0, 51)
    u0 = heat_exact_sine(x, 0.0)
    u1 = heat_exact_sine(x, 0.1)
    factor = float(u1[25] / u0[25])
    assert abs(factor - float(np.exp(-(np.pi**2) * 0.1))) < 1e-12


def test_finite_grid_threshold_is_above_one_half_and_decreasing() -> None:
    vals = [ftcs_critical_r(nx) for nx in (11, 21, 41, 81)]
    assert all(v > 0.5 for v in vals)
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert abs(ftcs_critical_r(4001) - 0.5) < 1e-6


def test_overdamped_branch_has_no_oscillation() -> None:
    tt = np.linspace(0.0, 10.0, 501)
    x, _ = damped_oscillator(tt, omega=1.0, zeta=1.5)
    assert np.all(np.diff(np.sign(x)) == 0)
    assert abs(float(x[-1])) < abs(float(x[0]))


def test_fit_order_on_analytic_power_law() -> None:
    """顺手证明对拍工具本身可信：人造 e = 3h² 必须拟合出斜率 2。"""
    h = np.array([0.1, 0.05, 0.025, 0.0125])
    fit = fit_order(h, 3.0 * h**2)
    assert abs(fit.slope - 2.0) < 1e-9


def test_fit_order_head_option_escapes_roundoff_floor() -> None:
    """``head=`` 只保留粗网格端（与 TS 的 fitOrder 逐字段对齐）。

    这条测试记录了一个真实陷阱：诊断信息曾经建议"用 tail= 截掉细网格端"，
    但 ``tail=`` 保留的正是**细端**——照着那条建议做会恰好保留被舍入地板污染的点。
    现在 Python 与 TypeScript 两侧都提供 ``head=``，并且诊断信息指向它。
    """
    h = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    err = np.array([1.0, 0.25, 0.0625, 0.015625, 1e-3, 1e-3])
    full = fit_order(h, err)
    head = fit_order(h, err, head=4)
    tail = fit_order(h, err, tail=3)
    assert abs(full.slope - 2.0) > 0.05  # 全窗口被地板带偏
    assert full.r2 < 0.99  # R² 会把这个偏差暴露出来（斜率本身可能"看着还行"）
    assert abs(head.slope - 2.0) < 1e-9 and head.r2 > 0.999999  # 只保粗端 → 精确 2
    assert tail.r2 < 0.9  # 只保细端 → 窗口里几乎全是地板点，拟合不可信
    assert abs(fit_order(h, err, skip=1, head=3).slope - 2.0) < 1e-9  # skip 与 head 可叠加
    # 诊断信息必须把用户指向 head=，而不是 tail=
    diag = diagnose_order(h, err)
    assert diag.floored, diag
    assert "head=" in diag.note()
    assert "tail= 保留的是细端" in diag.note()
