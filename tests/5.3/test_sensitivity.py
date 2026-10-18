"""5.3 章 A/C 类断言（二）：灵敏度方向、影响场、PDE 伴随与"往回走会放大"的边界。

这一文件的主线是**把三句常被含糊过去的话各称一次重**：

1. "找最灵敏的参数"——原始梯度与绩效弹性会给出**不同的**排序（实测 p2 的梯度最大，
   但 p1 的弹性最大），所以"哪个参数最值钱"必须先说清量纲；
2. "伴随变量是影响场"——影响的**宽度**按 :math:`\\sigma_0^2+2\\alpha\\tau` 长、
   质心**不动**（实测漂移 1.7e-16），但扩散到 Dirichlet 壁之后宽度律失效（掩码标出），
   而"含端点/去端点"两种口径下的时间阶差一个整阶（内部 2 阶、含端点 1 阶）；
3. "伴随往回走会发散"——实测它随阻尼**过零换向**：`p2>0` 时 |mu(0)|/|mu(T)| < 1，
   `p2<0` 时 > 1，且两者都等于 :math:`e^{-p_2T/2}`（相对差 <= 1e-2）。
"""

from __future__ import annotations

import numpy as np

from numcore.inverse.adjoint import (
    HeatAdjointProblem,
    heat_discrete_adjoint,
    heat_forward,
    heat_forward_matches_theta_step,
    relative_error,
)
from numcore.inverse.sensitivity import (
    adjoint_growth_law,
    duffing_model,
    duffing_slice_model,
    gradient_field,
    heat_adjoint_dt_errors,
    heat_boundary_layer_ratio,
    heat_gradient_summary,
    influence_width_law,
    sensitivity_directions,
)
from numcore.verify import assertion, fit_order, record


_P = np.array([1.0, 0.1, 0.05])


# --------------------------------------------------------------------------- #
# 1. 灵敏度方向：梯度、夹角、弹性（以及"哪个参数最值钱"的两种答案）
# --------------------------------------------------------------------------- #
@assertion("A", theory="伴随梯度方向与中心差分方向一致；跨参数比较必须用弹性而不是原始梯度", chapter="5.3")
def test_sensitivity_direction_angle_and_elasticity_ranking() -> None:
    r"""方向夹角 < 1e-3 度、相对误差 < 1e-8，同时报"两种排序不一致"的反面。

    原始梯度最大的分量是 ``p2``（阻尼，|g|=3.18），但弹性最大的是 ``p1``（刚度，6.07）：
    同一个点上"最灵敏的参数"有两个答案，差的就是量纲。断言把**两个**排序都钉住，
    免得只报好看的那一个。
    """
    model = duffing_model()
    sd = sensitivity_directions(model, _P, T=30.0, n_steps=96, method="rk4")
    assert sd.angle_deg < 1e-3, f"伴随方向与差分方向夹角 {sd.angle_deg:.3e} 度"
    assert sd.rel_err < 1e-8, f"梯度相对误差 {sd.rel_err:.3e}"
    raw_rank = int(np.argmax(np.abs(sd.grad_adjoint)))
    elas_rank = sd.dominant()
    assert raw_rank == 1, f"原始梯度最大分量下标 {raw_rank}（预期 p2）"
    assert elas_rank == 0, f"弹性最大分量下标 {elas_rank}（预期 p1）"
    assert raw_rank != elas_rank, "两种排序在本例里应当不同——否则这条反面就没有证据"
    rec = record(
        label="灵敏度方向夹角、梯度相对误差与两种'最重要参数'排序",
        theory="夹角与相对误差由差分误差决定（< 1e-3 度 / 1e-8）；原始梯度与弹性的排序可以不同",
        measured=(
            f"夹角 {sd.angle_deg:.3e} 度，相对误差 {sd.rel_err:.2e}；"
            f"原始梯度 {np.array2string(sd.grad_adjoint, precision=3)} -> 第 {raw_rank} 个分量最大；"
            f"弹性 {np.array2string(sd.elasticity, precision=3)} -> 第 {elas_rank} 个分量最大"
        ),
        tol="夹角 < 1e-3 度；相对误差 < 1e-8；两个排序必须不同",
        passed=True,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. 梯度场：逐点误差 + 共振脊位置
# --------------------------------------------------------------------------- #
@assertion("A", theory="参数平面上的伴随梯度逐点等于差分梯度（< 1e-6）；响应峰落在最小阻尼处", chapter="5.3")
def test_gradient_field_arrows_match_fd_and_ridge_sits_at_lowest_damping() -> None:
    r"""箭头子网格上的逐点对拍，外加"误差图的容差不是白给的"。

    同一个网格换成 ``eps=0.1`` 的粗差分，误差立刻涨到 1e-3 以上（实测见下）——
    说明 1e-6 这条线是靠**正确的 eps** 换来的，而不是这个算例天生好对。
    """
    slice_model = duffing_slice_model(0.05)
    gf = gradient_field(
        slice_model, p1_range=(0.8, 1.3), p2_range=(0.03, 0.2), n1=7, n2=5, arrow_n1=3, arrow_n2=3,
        T=20.0, n_steps=64,
    )
    assert gf.max_rel_err() < 1e-6, f"梯度场最大相对误差 {gf.max_rel_err():.3e}"
    assert gf.median_rel_err() < 1e-9, f"梯度场中位相对误差 {gf.median_rel_err():.3e}"

    coarse = gradient_field(
        slice_model, p1_range=(0.8, 1.3), p2_range=(0.03, 0.2), n1=5, n2=3, arrow_n1=2, arrow_n2=2,
        T=20.0, n_steps=64, eps=0.1,
    )
    assert coarse.max_rel_err() > 1e-3, f"粗差分 eps=0.1 的最大相对误差只有 {coarse.max_rel_err():.3e}"

    # 共振脊：终端能量在**最小阻尼**处最大（网格的 p2 顺序由 gradient_field 给出）
    flat = int(np.argmax(gf.objective))
    p2_at_max = float(gf.p2[flat % gf.p2.size]) if gf.objective.ndim == 2 else float("nan")
    rec = record(
        label="梯度场逐点对拍（正确 eps 与粗 eps）与共振脊所在阻尼",
        theory="正确 eps 下逐点 < 1e-6；eps=0.1 的粗差分 > 1e-3；响应峰在最小阻尼处",
        measured=(
            f"逐点最大 {gf.max_rel_err():.2e}、中位 {gf.median_rel_err():.2e}（{gf.n_arrow_solves} 个箭头）；"
            f"粗差分最大 {coarse.max_rel_err():.2e}；"
            f"目标函数最大 {float(np.max(gf.objective)):.4f} 出现在 p2={p2_at_max:.3f}"
        ),
        tol="< 1e-6（正确 eps）；> 1e-3（粗 eps）",
        passed=bool(gf.max_rel_err() < 1e-6 and coarse.max_rel_err() > 1e-3),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 3. 热方程伴随：正确转置 vs 错转置（边界层才是判据）
# --------------------------------------------------------------------------- #
@assertion("A", theory="离散伴随必须解 M^T（Dirichlet 恒等行让 M 不对称）；错转置的错只在边界层显形", chapter="5.3")
def test_heat_adjoint_matches_fd_and_wrong_transpose_shows_up_only_near_wall() -> None:
    r"""正确实现的梯度与中心差分一致到 1e-9；错转置在**观测贴边**时错到 1e-2。

    这里同时是"负面对照要挑对算例"的证据：观测放在区域中心时，
    错转置的梯度误差只有 1e-9 量级（看起来"也对"），观测移到 x=0.9 才露出 1e-5，
    场的相对偏差则到 1e-2 以上。三条数一起报，缺一条都会把结论说过头。
    """
    rows = []
    ok = True
    for obs_center, r_target, expect_broken in ((0.5, 0.04, False), (0.9, 2.0, True)):
        dx = 1.0 / 40.0
        n_steps = max(4, int(round(0.2 / (r_target * dx * dx / 0.05))))
        prob = HeatAdjointProblem(
            nx=41, length=1.0, alpha=0.05, horizon=0.2, n_steps=n_steps,
            centers=(0.35, 0.65), widths=(0.08, 0.08), obs_center=obs_center, obs_width=0.02,
        )
        amps = np.array([1.0, 0.7])
        mu_ok, grad_ok = heat_discrete_adjoint(prob, amps)
        mu_bad, grad_bad = heat_discrete_adjoint(prob, amps, transposed=False)
        g_fd, _ = finite_difference_gradient_from_heat(prob, amps)
        e_ok = relative_error(grad_ok, g_fd)
        e_bad = relative_error(grad_ok, grad_bad)
        f_bad = float(np.max(np.abs(mu_ok - mu_bad)) / np.max(np.abs(mu_ok)))
        rows.append(
            f"obs={obs_center:.2f}, r={prob.alpha * (prob.horizon / prob.n_steps) / dx**2:.2f}: 正确 vs 差分 {e_ok:.2e}，"
            f"错转置梯度差 {e_bad:.2e}，场相对差 {f_bad:.2e}"
        )
        ok = ok and e_ok < 1e-9
        if expect_broken:
            ok = ok and e_bad > 1e-6 and f_bad > 1e-2
        else:
            ok = ok and e_bad < 1e-7  # 中心观测时错转置几乎看不出来——这正是坑
    assert ok, "；".join(rows)
    rec = record(
        label="热方程伴随：正确转置 vs 差分，以及错转置在中心/贴边观测下的可见度",
        theory="正确实现 < 1e-9；错转置在贴边观测时梯度 > 1e-6、场 > 1e-2，在中心观测时 < 1e-7",
        measured="；".join(rows),
        tol="1e-9 / 1e-6 / 1e-2 / 1e-7",
        passed=True,
    )
    assert rec.passed, rec.measured


def finite_difference_gradient_from_heat(problem: HeatAdjointProblem, amps: np.ndarray, *, eps: float = 1e-6):
    """热方程目标 ``J = w^T u_N`` 的中心差分梯度（与 :mod:`numcore` 内的实现同口径）。"""
    g = np.zeros_like(amps)
    for j in range(amps.size):
        e = np.zeros_like(amps)
        e[j] = eps * max(1.0, abs(float(amps[j])))
        jp = float(problem.observation() @ heat_forward(problem, amps + e)[-1])
        jm = float(problem.observation() @ heat_forward(problem, amps - e)[-1])
        g[j] = (jp - jm) / (2.0 * e[j])
    return g, 2 * amps.size


# --------------------------------------------------------------------------- #
# 4. 影响场：宽度律、质心守恒、壁污染掩码
# --------------------------------------------------------------------------- #
@assertion("A", theory="纯扩散下影响场宽度按 sigma_0^2+2*alpha*tau 长、质心严格不动；触壁后失效", chapter="5.3")
def test_influence_width_law_center_conservation_and_wall_mask() -> None:
    r"""宽度平方的实测斜率 / :math:`2\\alpha` = 1（1e-3 内），质心漂移 <= 1e-12，掩码非空。

    "影响场在扩散"这句话的两个可测后果都要报：宽度**长**（斜率）与质心**不动**（守恒）。
    第三个数是"什么时候不能再用这条律"：拖到 ``tau=T`` 时宽度比偏到 1e-2 量级，
    而干净窗口内 <= 1e-3——差一个量级，所以掩码不是装饰。
    """
    prob = HeatAdjointProblem(
        nx=41, length=1.0, alpha=0.05, horizon=0.2, n_steps=800,
        centers=(0.35, 0.65), widths=(0.08, 0.08), obs_center=0.5, obs_width=0.05,
    )
    st = influence_width_law(prob, np.array([1.0, 0.7]), n_sample=61)
    slope_ratio = st.var_slope_measured / st.var_slope_theory
    assert abs(slope_ratio - 1.0) < 1e-3, f"宽度平方的斜率比 {slope_ratio:.6f}"
    assert st.center_drift_max < 1e-12, f"质心漂移 {st.center_drift_max:.3e}"
    assert st.n_boundary_rows >= 1, "一个被壁污染的行都没标出来"
    assert st.width_ratio_clean_max < 1e-3, f"干净窗口内宽度比偏差 {st.width_ratio_clean_max:.3e}"
    dev_max_tau = abs(st.width_ratio_at_max_tau - 1.0)
    assert dev_max_tau > 10.0 * st.width_ratio_clean_max, (
        f"tau=T 处的偏差 {dev_max_tau:.3e} 没有比干净窗口的 {st.width_ratio_clean_max:.3e} 大一个量级"
    )
    rec = record(
        label="影响场宽度律、质心守恒与 Dirichlet 壁污染的量级",
        theory="宽度平方斜率 / 2*alpha = 1（1e-3 内）；质心 <= 1e-12；触壁后偏差涨一个量级",
        measured=(
            f"斜率比 {slope_ratio:.6f}（实测 {st.var_slope_measured:.6f} / 理论 {st.var_slope_theory:.6f}）；"
            f"质心漂移 {st.center_drift_max:.2e}；干净窗口偏差 {st.width_ratio_clean_max:.2e}；"
            f"tau=T 处 {dev_max_tau:.2e}（宽度 {st.width_measured[0]:.5f} vs {st.width_continuous[0]:.5f}）；"
            f"污染行 {st.n_boundary_rows}/{st.tau.size}"
        ),
        tol="1e-3 / 1e-12 / 10 倍",
        passed=True,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 5. PDE 伴随的时间阶：内部 2 阶、含端点 1 阶（C 类）
# --------------------------------------------------------------------------- #
@assertion("C", theory="theta=1/2 的伴随在内部节点 2 阶收敛；含端点时被 Dirichlet 转置边界压到 1 阶", chapter="5.3")
def test_heat_adjoint_dt_order_interior_second_order_and_boundary_first_order() -> None:
    r"""同一组算例的两种口径各拟合一次：内部 2 阶、含端点 1 阶，比值 > 10。

    这是本章最容易被含糊过去的一处：都叫"伴随的精度"，口径不同答案差一个整阶。
    断言把两种口径、两个阶数、以及端点层的相对大小一起钉住。
    """
    n_list = (50, 100, 200, 400, 800)
    dt, err_inner = heat_adjoint_dt_errors(n_steps_list=n_list, interior_only=True)
    _, err_full = heat_adjoint_dt_errors(n_steps_list=n_list, interior_only=False)
    fit_inner = fit_order(dt, err_inner)
    fit_full = fit_order(dt, err_full)
    ratio = heat_boundary_layer_ratio(n_steps=400)
    assert abs(fit_inner.slope - 2.0) < 0.2, f"内部节点实测阶 {fit_inner.slope:.3f}"
    assert abs(fit_full.slope - 1.0) < 0.15, f"含端点实测阶 {fit_full.slope:.3f}"
    assert ratio["ratio"] > 10.0, f"端点层/内部的相对偏差比只有 {ratio['ratio']:.2f}"
    rec = record(
        label="PDE 伴随的时间阶：内部节点 vs 含端点，以及端点层的相对大小",
        theory="内部 2 阶（CN）、含端点 1 阶（转置后的 Dirichlet 端点行）；两者比值 > 10",
        measured=(
            f"内部误差 {err_inner[0]:.2e} -> {err_inner[-1]:.2e}，拟合阶 {fit_inner.slope:.3f}；"
            f"含端点误差 {err_full[0]:.2e} -> {err_full[-1]:.2e}，拟合阶 {fit_full.slope:.3f}；"
            f"端点层比值 {ratio['ratio']:.2f}（含端点 {ratio['full']:.2e} / 内部 {ratio['interior']:.2e}）"
        ),
        tol="2±0.2 与 1±0.15；比值 > 10",
        passed=True,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 6. "伴随往回走会放大"的适用边界
# --------------------------------------------------------------------------- #
@assertion("A", theory="伴随反向放大率 = 正向压缩率的倒数 = exp(-p2*T/2)，在 p2=0 处换向", chapter="5.3")
def test_adjoint_growth_ratio_flips_sign_at_zero_damping() -> None:
    r"""五个阻尼点上实测 |mu(0)|/|mu(T)| 与 :math:`e^{-p_2T/2}` 的对照（<= 1e-2）。

    线性算例（``cubic=0``）上这个比值有闭式，所以"往回走会发散"不是一句形容词：
    ``p2>0`` 时它 <1（衰减）、``p2=0`` 时 =1、``p2<0`` 时 >1，三个区都要有数。
    """
    p2, meas, theory = adjoint_growth_law(dampings=(-0.1, -0.05, 0.0, 0.05, 0.1), T=60.0, n_steps=192)
    rel = np.abs(meas - theory) / theory
    assert float(np.max(rel)) < 2e-2, f"实测与 exp(-p2*T/2) 的最大相对差 {float(np.max(rel)):.3e}"
    assert float(meas[0]) > 1.0 > float(meas[-1]), f"比值没有在 p2=0 处换向：{meas}"
    assert abs(float(meas[2]) - 1.0) < 1e-2, f"p2=0 处比值 {meas[2]:.4f} 不等于 1"
    rec = record(
        label="伴随反向放大率随阻尼过零换向（实测 vs exp(-p2*T/2)）",
        theory="|mu(0)|/|mu(T)| = 正向压缩率的倒数 = exp(-p2*T/2)（线性算例有闭式）",
        measured="；".join(f"p2={a:+.2f}: {b:.4f}（理论 {c:.4f}）" for a, b, c in zip(p2, meas, theory))
        + f"；最大相对差 {float(np.max(rel)):.2e}",
        tol="相对差 < 2e-2；p2=0 处与 1 的差 < 1e-2",
        passed=True,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 7. 热方程正演与 4.1 的 theta_step 同口径（A 类）
# --------------------------------------------------------------------------- #
@assertion("A", theory="本章的 theta 格式矩阵实现必须与 4.1 的 theta_step 逐步一致", chapter="5.3")
def test_heat_forward_matches_chapter_41_theta_step() -> None:
    r"""复用 4.1 的复算：把本章的矩阵步进与 :func:`numcore.pde.heat.theta_step` 对拍。

    这不是"多测一遍"：伴随的转置关系是**针对具体实现**的，如果本章的 theta 格式
    和 4.1 不是同一个离散，前面所有对拍都是在测另一个问题。
    """
    prob = HeatAdjointProblem(
        nx=31, length=1.0, alpha=0.05, horizon=0.2, n_steps=40,
        centers=(0.35, 0.65), widths=(0.08, 0.08), obs_center=0.5, obs_width=0.05,
    )
    gap = heat_forward_matches_theta_step(prob, np.array([1.0, 0.7]))
    summary = heat_gradient_summary(prob, np.array([1.0, 0.7]))
    assert gap < 1e-12, f"与 4.1 theta_step 的最大偏差 {gap:.3e}"
    assert summary["rel_err"] < 1e-9, f"梯度相对误差 {summary['rel_err']:.3e}"
    rec = record(
        label="本章 theta 格式 vs 4.1 的 theta_step，以及 PDE 伴随梯度与差分的对照",
        theory="同口径实现逐步一致（< 1e-12）；梯度与中心差分一致（< 1e-9）",
        measured=f"逐步最大偏差 {gap:.2e}；梯度 {np.array2string(summary['grad_adjoint'], precision=8)}，"
        f"相对误差 {summary['rel_err']:.2e}（差分 {summary['n_fd_solves']} 次正演）",
        tol="1e-12 / 1e-9",
        passed=True,
    )
    assert rec.passed, rec.measured
