"""3.4 章（上半）：打靶法与有限差分的 A/C 类断言。

这一半只回答两个问题：

* **打靶法什么时候能用**——它的"灵敏度" :math:`S=\\left|\\partial u(b)/\\partial s\\right|`
  由变分方程给出闭式 (:math:`\\sinh(kL)/k`)，是本章唯一可以精确对拍的量；
* **有限差分的阶到底由谁决定**——内部二阶差商人人会写，真正把整体阶拉下来的
  是**边界条件的一阶差商**。所以每条阶断言都同时记**整体阶**与**内部阶**，
  并且写死拟合窗口。

数值上最容易被自己骗过去的三处，都在这里当面量出来：

1. 边界残差有两个口径（离散行残差 vs 把数值解当真函数的真实违反量），
   一阶边界格式下前者 :math:`\\sim10^{-13}`、后者 :math:`\\sim10^{-2}`；
2. "残差小"与"解准"没有关系：线性系统残差 :math:`10^{-13}` 时误差还有 :math:`10^{-2}`；
3. 同一个灵敏度下，**增长**模态能解、**衰减**模态解不了——差别不在灵敏度，
   而在"指数小的那一支要不要靠相减抵消出来"。加密步长救不回来，因为是舍入被放大，
   不是截断误差。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numcore.ode import bvp as bvp
from numcore.verify import (
    assert_leq,
    assert_order,
    assert_threshold,
    assertion,
    find_threshold,
    record,
)

U = float(np.finfo(float).eps) / 2.0


# --------------------------------------------------------------------------- #
# A 类：打靶法的两个数（灵敏度、Newton）
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    "打靶灵敏度由变分方程 z'=Jz、z(a)=(0,1) 积分给出，对 u''=k^2u、u(a)=0 等于闭式 sinh(kL)/k",
    chapter="3.4",
)
def test_shooting_sensitivity_matches_closed_form() -> None:
    cases = [(5.0, 1.0), (10.0, 1.0), (20.0, 1.0)]
    got: list[float] = []
    ref: list[float] = []
    for k, length in cases:
        prob = bvp.problem("layer", k=k, L=length)
        got.append(bvp.shooting_sensitivity(prob, n_steps=2000))
        ref.append(math.sinh(k * length) / k)
    rel = [abs(g - r) / r for g, r in zip(got, ref)]
    worst = max(rel)
    # kL=40 单独看：灵敏度本身已经 ~1e16，变分方程把 1e16 的增长模态积出来，
    # 相对精度自然退化（实测 5e-8）——这条也要如实记，不能只报好看的窗口。
    big = bvp.problem("layer", k=20.0, L=2.0)
    s_big = bvp.shooting_sensitivity(big, n_steps=2000)
    rel_big = abs(s_big - math.sinh(40.0) / 20.0) / (math.sinh(40.0) / 20.0)
    detail = ", ".join(f"k={k:g},L={L:g}: {g:.6e} vs {r:.6e}" for (k, L), g, r in zip(cases, got, ref))
    assert_leq(worst, 1e-8, label="灵敏度 vs sinh(kL)/k（相对，kL≤20）", detail=detail)
    assert_leq(rel_big, 1e-6, label="kL=40 时灵敏度自身的相对偏差")
    record(
        label="变分方程灵敏度 vs 闭式 sinh(kL)/k",
        theory="S = sinh(kL)/k",
        measured=f"kL≤20 最大相对偏差 {worst:.3e}；kL=40 时 S={s_big:.6e}（闭式 {math.sinh(40.0) / 20.0:.6e}，相对 {rel_big:.1e}）",
        tol="kL≤20 相对 1e-8；kL=40 相对 1e-6",
        passed=worst <= 1e-8 and rel_big <= 1e-6,
        kind="A",
        extra={"sensitivity": got, "closed_form": ref, "sensitivity_kL40": s_big, "rel_kL40": rel_big},
    )


@assertion(
    "A",
    "线性两点边值问题的打靶函数 F(s)=u(b;s)-β 对 s 是仿射的，Newton 一步到（第二次求值只是确认）",
    chapter="3.4",
)
def test_linear_shooting_converges_in_one_newton_step() -> None:
    rows: list[str] = []
    worst_slope = 0.0
    worst_err = 0.0
    max_it = 0
    for name, kw in [("smooth", {}), ("neumann", {}), ("layer", {"k": 5.0, "L": 1.0})]:
        prob = bvp.problem(name, **kw)
        sh = bvp.shooting(prob, slope0=1.0, n_steps=400)
        ref = bvp.shooting(prob, slope0=sh.slope_theory * 0.999, n_steps=400, tol=1e-15)
        worst_slope = max(worst_slope, abs(sh.slope - ref.slope) / max(abs(ref.slope), 1e-12))
        worst_err = max(worst_err, sh.error_max)
        max_it = max(max_it, sh.iterations)
        rows.append(f"{name}: s={sh.slope:.6e}，{sh.iterations} 次求值，err={sh.error_max:.2e}")
    assert max_it <= 3, f"线性问题 Newton 应当一步到（≤3 次函数求值），实测最多 {max_it} 次"
    assert_leq(worst_slope, 1e-8, label="线性问题 Newton 收敛后的斜率（与另一次打靶互差，相对）")
    record(
        label="线性问题 Newton 迭代次数",
        theory="1 步（仿射 F）+ 1 次确认求值",
        measured=f"最多 {max_it} 次求值；" + "；".join(rows),
        tol="≤ 3 次求值",
        passed=max_it <= 3 and worst_slope <= 1e-8,
        kind="A",
        extra={"max_f_evals": max_it, "max_error": worst_err},
    )


@assertion(
    "A",
    "非线性打靶（Bratu u''=1.5u^2）的 Newton 残差按二次律下降：|F_{k+1}| ≤ 0.2·|F_k|^2（进入 1e-1 之后）",
    chapter="3.4",
)
def test_nonlinear_shooting_newton_is_quadratic() -> None:
    prob = bvp.problem("bratu")
    sh = bvp.shooting(prob, n_steps=400, max_iter=40)
    hist = [abs(v) for v in sh.history if math.isfinite(v)]
    quad: list[str] = []
    ok = True
    for a, b in zip(hist, hist[1:]):
        # 只查 |F| ≥ 1e-4 的段：再往下打靶函数本身的可分辨下限（~eps·S）就到了，
        # 那时 |F| 的下降由噪声决定，继续要求二次律是苛求。
        if 1e-4 <= a <= 1e-1:
            good = b <= 0.2 * a * a
            ok = ok and good
            quad.append(f"{a:.2e}->{b:.2e}{'' if good else '(超界)'}")
    assert len(hist) >= 4, f"Newton 至少要有 4 次残差记录，实测 {len(hist)}"
    assert ok, "二次收敛判据失败：" + "，".join(quad)
    assert sh.converged and sh.error_max < 1e-9, f"Bratu 打靶应当收敛到 1e-9 以内，实测 err={sh.error_max:.3e}"
    record(
        label="Bratu 打靶 Newton 的二次收敛",
        theory="|F_{k+1}| ≤ 0.2·|F_k|^2（牛顿法的标准二次界，只查 |F|≥1e-4 的段）",
        measured="残差史 " + " -> ".join(f"{v:.2e}" for v in hist),
        tol="进 1e-1 且未到 1e-4 地板时每步满足二次界",
        passed=ok,
        kind="A",
        extra={"iterations": sh.iterations, "error": sh.error_max, "slope": sh.slope},
    )


@assertion(
    "A",
    "同一灵敏度下：增长模态（u(0)=0,u(L)=1）打得准，衰减模态（u(0)=1,u(L)=0）打不准——S 相同而结果差 10 个量级",
    chapter="3.4",
)
def test_growing_and_decaying_layer_modes_differ_at_equal_sensitivity() -> None:
    k, length = 20.0, 2.0
    grow = bvp.shooting(bvp.problem("layer", k=k, L=length), n_steps=2000)
    decay = bvp.shooting(bvp.problem("decay", k=k, L=length), n_steps=2000)
    s_grow, s_decay = grow.sensitivity, decay.sensitivity
    same = abs(s_grow - s_decay) / s_grow
    assert_leq(same, 1e-9, label="两条路线的灵敏度一致（相对）")
    assert grow.error_max < 1e-6, f"增长模态应当解得准，实测 err={grow.error_max:.3e}"
    assert decay.error_max > 1.0, f"衰减模态应当解不出来，实测 err={decay.error_max:.3e}"
    ratio = decay.error_max / max(grow.error_max, 1e-300)
    record(
        label="增长模态 vs 衰减模态（同一 S）",
        theory="S=sinh(kL)/k 相同，但衰减模态要在指数小的分量上做相减",
        measured=f"S={s_grow:.4e}；增长模态 err={grow.error_max:.2e}，衰减模态 err={decay.error_max:.2e}，相差 {ratio:.2e} 倍",
        tol="增长 <1e-6 且 衰减 >1",
        passed=same <= 1e-9 and grow.error_max < 1e-6 and decay.error_max > 1.0,
        kind="A",
        extra={"sensitivity": s_grow, "growing_error": grow.error_max, "decaying_error": decay.error_max},
    )


@assertion(
    "A",
    "衰减模态的失败不是截断误差：把**精确斜率**喂进去再加密步长（h 缩小 32 倍），误差不降",
    chapter="3.4",
)
def test_decaying_mode_failure_is_roundoff_not_truncation() -> None:
    k, length = 20.0, 2.0
    prob = bvp.problem("decay", k=k, L=length)
    slope = -k * math.cosh(k * length) / math.sinh(k * length)
    errs: list[float] = []
    for ns in (250, 500, 1000, 2000, 4000, 8000):
        x, us, _, ok = bvp._integrate_ivp(prob, slope, n_steps=ns, with_variational=False)
        assert ok, f"ns={ns} 的初值推进飞出了 BLOWUP"
        errs.append(float(np.max(np.abs(us - prob.exact(x)))))
    worst = max(errs)
    best = min(errs)
    # 截断误差在 h 缩小 32 倍时应当掉 2^5=32 倍（RK4 是 h^4）；这里连 2 倍都降不下来
    assert best > 0.1, f"精确斜率下加密步长仍应当有大误差，实测最小 {best:.3e}"
    assert worst / best < 100.0, f"误差不随 h 单调下降（best={best:.3e}, worst={worst:.3e}）"
    record(
        label="精确斜率 + 加密步长下的衰减模态误差",
        theory="舍入被 e^{kL} 放大：误差 ~ eps·e^{kL}，与 h 无关",
        measured=f"ns=250..8000（h: 8.0e-3 → 2.5e-4）误差 {min(errs):.2e}~{max(errs):.2e}，kL=40，eps·exp(kL)={U * math.exp(k * length):.1e}",
        tol="误差下界 >0.1 且不随 h 收敛",
        passed=best > 0.1 and worst / best < 100.0,
        kind="A",
        extra={"errors": errs, "step_sizes": [length / n for n in (250, 500, 1000, 2000, 4000, 8000)]},
    )


# --------------------------------------------------------------------------- #
# A 类：有限差分的阶与边界条件
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    "两端 Dirichlet 的有限差分（内部二阶中心差商）整体阶与内部阶都是 2",
    chapter="3.4",
)
def test_fd_dirichlet_is_second_order_globally_and_interiorly() -> None:
    prob = bvp.problem("smooth")
    ns = [32, 64, 128, 256, 512]
    study = bvp.convergence_study(prob, ns, method="fd", bc_order=2)
    g = assert_order(
        1.0 / np.asarray(study["n"], dtype=float),
        np.asarray(study["error"], dtype=float),
        theory=2.0,
        tol=0.05,
        label="Dirichlet 两端：整体阶",
        detail="u''-u=e^x sin(pi x)，n=32..512",
    )
    i = assert_order(
        1.0 / np.asarray(study["n"], dtype=float),
        np.asarray(study["error_interior"], dtype=float),
        theory=2.0,
        tol=0.05,
        label="Dirichlet 两端：内部阶",
        detail="掐掉两端各 25% 的区间，看误差是否被边界局部污染",
    )
    assert_leq(max(study["bc_discrete"]), 1e-11, label="离散边界残差（Dirichlet 被钉死）")
    record(
        label="Dirichlet：整体阶 = 内部阶 = 2",
        theory="二阶中心差商 + 精确 Dirichlet 行",
        measured=f"整体 {g.slope:.4f}，内部 {i.slope:.4f}；误差 {study['error'][0]:.3e} -> {study['error'][-1]:.3e}",
        tol="±0.05",
        passed=abs(g.slope - 2.0) <= 0.05 and abs(i.slope - 2.0) <= 0.05,
        kind="A",
        extra={"p_global": g.slope, "p_interior": i.slope},
    )


@assertion(
    "A",
    "混合边界（u(0)=0, u'(1)=0）的**整体阶由边界差商的阶决定**：一阶边界 => 整体阶 1，二阶边界 => 整体阶 ≈2；内部阶始终是 2 的邻居",
    chapter="3.4",
)
def test_fd_neumann_bc_order_controls_the_global_order() -> None:
    prob = bvp.problem("neumann")
    ns = [128, 256, 512, 1024]
    h = 1.0 / np.asarray(ns, dtype=float)
    out: dict[str, float] = {}
    for order, want in ((1, 1.0), (2, 2.0)):
        study = bvp.convergence_study(prob, ns, method="fd", bc_order=order)
        fit = assert_order(
            h,
            np.asarray(study["error"], dtype=float),
            theory=want,
            tol=0.1,
            label=f"混合边界（边界格式 {order} 阶）：整体阶",
            detail=f"u=sin(pi x/2)，n={ns[0]}..{ns[-1]}，误差不局限在边界附近（内部也被污染）",
        )
        inner = bvp.fit_power(np.asarray(study["n"], dtype=float), np.asarray(study["error_interior"], dtype=float))
        out[f"p_global_{order}"] = fit.slope
        out[f"p_interior_{order}"] = inner["p"]
    # 一阶边界格式：内部阶也掉到 1——这是"内部阶能救回来"这一错觉的反例
    assert out["p_interior_1"] < 1.1, f"一阶边界下内部阶也应当 ~1，实测 {out['p_interior_1']:.3f}"
    assert out["p_interior_2"] > 1.85, f"二阶边界下内部阶应当 ~1.93，实测 {out['p_interior_2']:.3f}"
    record(
        label="混合边界：整体阶与内部阶",
        theory="整体阶 = min(内部阶, 边界格式阶)",
        measured=f"一阶边界：整体 {out['p_global_1']:.4f} / 内部 {out['p_interior_1']:.4f}；"
        f"二阶边界：整体 {out['p_global_2']:.4f} / 内部 {out['p_interior_2']:.4f}",
        tol="±0.1",
        passed=out["p_global_1"] < 1.1 and out["p_interior_1"] < 1.1 and out["p_interior_2"] > 1.85,
        kind="A",
        extra=out,
    )


@assertion(
    "A",
    "边界残差的两个口径：离散行残差恒为舍入级，真实违反量在一阶边界格式下是 O(h)（两者差 11 个量级）",
    chapter="3.4",
)
def test_bc_residual_discrete_versus_true() -> None:
    prob = bvp.problem("neumann")
    rows: list[str] = []
    disc_max = 0.0
    true_min = math.inf
    for n in (64, 128, 256):
        res = bvp.fd_bvp(prob, n, bc_order=1)
        disc_max = max(disc_max, res.bc_discrete)
        true_min = min(true_min, res.bc_true)
        rows.append(f"n={n}: 离散 {res.bc_discrete:.1e} / 真实 {res.bc_true:.1e} / 误差 {res.error_max:.1e}")
    assert_leq(disc_max, 1e-12, label="离散边界残差（格式自己那一行）")
    assert true_min > 1e-4, f"真实边界违反量应当是一阶的 O(h)，实测最小 {true_min:.3e}"
    gap = true_min / max(disc_max, 1e-300)
    record(
        label="边界残差两个口径的差距（一阶边界格式）",
        theory="离散行残差 ~ eps；真实违反量 ~ h·|u''|",
        measured="；".join(rows) + f"；两者相差 {gap:.1e} 倍",
        tol="离散 ≤1e-12 且 真实 >1e-4",
        passed=disc_max <= 1e-12 and true_min > 1e-4,
        kind="A",
        extra={"bc_discrete": disc_max, "bc_true": true_min, "gap": gap},
    )


@assertion(
    "A",
    "有限差分：线性系统残差 ~1e-13 时误差仍有 1e-2（残差小和解准无关），且真实边界残差的衰减阶等于整体阶",
    chapter="3.4",
)
def test_fd_residual_small_but_error_large() -> None:
    prob = bvp.problem("neumann")
    ns = [64, 128, 256, 512]
    study = bvp.convergence_study(prob, ns, method="fd", bc_order=2)
    lin = np.asarray(study["linear_residual"], dtype=float)
    err = np.asarray(study["error"], dtype=float)
    worst_ratio = float(np.min(err / np.maximum(lin, 1e-300)))
    assert_leq(max(lin), 1e-9, label="线性系统残差（LU 回代）")
    assert worst_ratio > 1e4, f"误差/残差 应当至少 1e4 倍，实测最小 {worst_ratio:.1e}"
    fit = assert_order(
        1.0 / np.asarray(ns, dtype=float),
        np.asarray(study["bc_true"], dtype=float),
        theory=3.0,
        tol=0.1,
        label="二阶边界格式下真实边界残差的衰减阶",
        detail="用四阶单侧差商估计 u'(b) 的违反量；实测是 3 阶（h^2 项抵消），如实记录",
    )
    record(
        label="残差与误差的比（有限差分）",
        theory="残差 ~ eps·||A||，误差由截断决定，两者量级无关",
        measured=f"最大线性残差 {max(lin):.2e}，误差 {err[0]:.2e}（n={ns[0]}）-> {err[-1]:.2e}（n={ns[-1]}）；最小比值 {worst_ratio:.1e}",
        tol="残差 ≤1e-9 且 比值 >1e4",
        passed=max(lin) <= 1e-9 and worst_ratio > 1e4,
        kind="A",
        extra={"linear_residual": list(lin), "error": list(err), "bc_true_order": fit.slope},
    )


@assertion(
    "A",
    "有限差分矩阵是稀疏三对角的（每行 ~3 个非零），稠密 LU 的 n^3/3 与三对角追赶法的 5n 差两个量级",
    chapter="3.4",
)
def test_fd_matrix_structure_and_cost() -> None:
    prob = bvp.problem("smooth")
    nnz: list[float] = []
    for n in (32, 128, 512):
        nnz.append(bvp.fd_bvp(prob, n, bc_order=2).nnz_per_row)
    assert_leq(max(nnz), 3.05, label="每行非零元个数（三对角）")
    flops_dense = [n**3 / 3.0 for n in (32, 128, 512)]
    flops_band = [5.0 * n for n in (32, 128, 512)]
    ratios = [d / b for d, b in zip(flops_dense, flops_band)]
    record(
        label="FD 矩阵的稀疏结构与消元代价",
        theory="三对角：每行 3 个非零；稠密 LU ~ n^3/3，追赶法 ~ 5n",
        measured=f"每行非零 {['%.2f' % v for v in nnz]}；n=512 时稠密/带状 = {ratios[-1]:.1f}（n=32: {ratios[0]:.1f}）",
        tol="每行 ≤3.05",
        passed=max(nnz) <= 3.05,
        kind="A",
        extra={"nnz_per_row": nnz, "flops_ratio": ratios},
    )


@assertion(
    "A",
    "同一非线性问题（Bratu）三条独立路线（打靶 / 有限差分+Newton / 谱配置+Newton）都与闭式解一致",
    chapter="3.4",
)
def test_three_routes_agree_on_nonlinear_problem() -> None:
    prob = bvp.problem("bratu")
    sh = bvp.shooting(prob, n_steps=800)
    fd = bvp.fd_bvp(prob, 320)
    co = bvp.cheb_collocation(prob, 32)
    assert sh.converged and fd.converged and co.converged, "三条路线都必须报告收敛"
    assert_leq(sh.error_max, 1e-9, label="打靶 vs 闭式解")
    assert_leq(fd.error_max, 1e-4, label="有限差分（n=320）vs 闭式解")
    assert_leq(co.error_max, 1e-11, label="谱配置（N=32）vs 闭式解")
    # 互差用三角不等式从"各自与闭式解的差"给出上界，避免在等距网格上做全局多项式插值
    bound = fd.error_max + sh.error_max
    record(
        label="Bratu：三条路线的误差与 Newton 迭代数",
        theory="u=4/(1+x)^2",
        measured=f"打靶 {sh.error_max:.2e}（{sh.iterations} 次 F 求值）；FD n=320 {fd.error_max:.2e}（{fd.iterations} 步）；"
        f"配置 N=32 {co.error_max:.2e}（{co.iterations} 步）；互差界 {bound:.2e}",
        tol="打靶 ≤1e-9，FD ≤1e-4，配置 ≤1e-11",
        passed=sh.error_max <= 1e-9 and fd.error_max <= 1e-4 and co.error_max <= 1e-11,
        kind="A",
        extra={
            "shooting_error": sh.error_max,
            "fd_error": fd.error_max,
            "cheb_error": co.error_max,
            "newton_iterations": [sh.iterations, fd.iterations, co.iterations],
        },
    )


@assertion(
    "A",
    "Newton 的收敛判据必须看相对步长：FD 最终残差 ~eps/h^2（n=320 时 3e-11），绝对 1e-13 判据永远达不到",
    chapter="3.4",
)
def test_fd_newton_residual_floor_is_eps_over_h2() -> None:
    prob = bvp.problem("bratu")
    rows: list[str] = []
    ok = True
    for n in (80, 160, 320):
        res = bvp.fd_bvp(prob, n, newton_tol=1e-14)
        floor = U * n * n
        rows.append(f"n={n}: 残差 {res.linear_residual:.2e}（eps/h^2={floor:.1e}）")
        ok = ok and res.converged and res.linear_residual > 1e-13
    assert ok, "绝对值判据不该被满足：残差地板 ~eps/h^2 高于 1e-13：" + "；".join(rows)
    record(
        label="非线性 FD 的 Newton 残差地板",
        theory="线性系统残差地板 ~ eps·||A|| ~ eps/h^2",
        measured="；".join(rows),
        tol="残差 >1e-13（所以判据用相对步长）",
        passed=ok,
        kind="A",
    )


# --------------------------------------------------------------------------- #
# C 类：阈值、收敛率与窗口纪律
# --------------------------------------------------------------------------- #
@assertion(
    "C",
    "衰减模态打靶的失效长度由舍入决定：二分实测 err>0.5 的临界 L* ≈ ln(1/u)/k",
    chapter="3.4",
)
def test_shooting_failure_length_matches_roundoff_estimate() -> None:
    k = 20.0

    def bad(length: float) -> bool:
        prob = bvp.problem("decay", k=k, L=float(length))
        return bvp.shooting(prob, n_steps=2000).error_max > 0.5

    res = find_threshold(bad, 1.0, 3.0, tol=1e-4, assume="false_at_lo", note="err>0.5 视为打靶失效")
    theory = math.log(1.0 / U) / k
    assert_threshold(
        res,
        theory,
        rtol=0.06,
        label="衰减模态打靶失效长度 L*",
        detail=f"k={k:g}，理论值 ln(1/u)/k 用 u=eps/2={U:.3e}",
    )
    record(
        label="失效长度 L* 的两侧",
        theory=f"L* = ln(1/u)/k = {theory:.4f}",
        measured=f"二分 {res.threshold:.4f}（区间 [{res.lo:.4f}, {res.hi:.4f}]，{res.iterations} 次迭代，{res.calls} 次谓词调用）；"
        f"kL* = {k * res.threshold:.2f}",
        tol="相对 6%",
        passed=abs(res.threshold - theory) <= 0.06 * theory,
        kind="C",
        extra={"threshold": res.threshold, "theory": theory, "predicate_calls": res.calls},
    )


@assertion(
    "C",
    "打靶误差随 kL 指数增长：log(err) 对 kL 的斜率 ≈ 1（理论 eps·e^{kL}）",
    chapter="3.4",
)
def test_shooting_error_grows_exponentially_in_kL() -> None:
    kls = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    errs = np.array(
        [bvp.shooting(bvp.problem("decay", k=float(kl), L=1.0), n_steps=2000).error_max for kl in kls],
        dtype=float,
    )
    fit = bvp.fit_exponential(kls, errs)
    slope = float(fit["slope"])
    r2 = float(fit["r2"])
    assert 0.75 <= slope <= 1.25, f"指数律斜率应当 ≈1，实测 {slope:.3f}（R^2={r2:.4f}）"
    assert r2 > 0.90, f"半对数拟合的 R^2 太低：{r2:.4f}"
    record(
        label="衰减模态打靶误差的指数律",
        theory="err ~ C·e^{kL}（舍入被灵敏度放大），半对数斜率 = 1",
        measured=f"kL={kls.tolist()} 对应 err={['%.2e' % v for v in errs]}；拟合斜率 {slope:.4f}，R^2={r2:.5f}",
        tol="斜率 1±0.25，R^2>0.90",
        passed=0.75 <= slope <= 1.25 and r2 > 0.90,
        kind="C",
        extra={"slope": slope, "r2": r2, "kl": kls.tolist(), "error": errs.tolist()},
    )


@assertion(
    "C",
    "有限差分阶的拟合窗口纪律：同一批数据，粗窗口给出 1.86、细窗口给出 1.98——报阶必须写明窗口",
    chapter="3.4",
)
def test_fd_order_depends_on_fit_window() -> None:
    prob = bvp.problem("neumann")
    coarse = bvp.convergence_study(prob, [16, 32, 64, 128], method="fd", bc_order=2)
    fine = bvp.convergence_study(prob, [256, 512, 1024], method="fd", bc_order=2)
    p_coarse = float(bvp.fit_power(np.asarray(coarse["n"], dtype=float), np.asarray(coarse["error"], dtype=float))["p"])
    p_fine = float(bvp.fit_power(np.asarray(fine["n"], dtype=float), np.asarray(fine["error"], dtype=float))["p"])
    assert p_coarse < 1.93, f"粗窗口应当明显低于 2（前渐近），实测 {p_coarse:.4f}"
    assert p_fine > 1.95, f"细窗口应当接近 2，实测 {p_fine:.4f}"
    record(
        label="同一格式两个窗口的实测阶",
        theory="二阶格式的渐近阶 2；前渐近区会偏低",
        measured=f"n=16..128 得 {p_coarse:.4f}；n=256..1024 得 {p_fine:.4f}",
        tol="粗 <1.93 且 细 >1.95",
        passed=p_coarse < 1.93 and p_fine > 1.95,
        kind="C",
        extra={"p_coarse": p_coarse, "p_fine": p_fine},
    )


@assertion(
    "C",
    "有限差分矩阵的条件数随 n 按 n^2 增长（实测指数 2.0），这是二阶差商 1/h^2 的直接后果",
    chapter="3.4",
)
def test_fd_matrix_condition_grows_like_n_squared() -> None:
    prob = bvp.problem("smooth")
    ns = np.array([32.0, 64.0, 128.0, 256.0], dtype=float)
    cond = np.array([bvp.fd_bvp(prob, int(n)).cond_inf for n in ns], dtype=float)
    # fit_power 的模型是 e ~ C·x^{-p}（误差随 N 递减）；条件数是**递增**的，
    # 所以传 1/cond，得到的 p 就是增长指数。
    fit = bvp.fit_power(ns, 1.0 / cond)
    assert 1.85 <= float(fit["p"]) <= 2.15, f"条件数指数应当 ≈2，实测 {fit['p']:.3f}"
    record(
        label="FD 矩阵条件数 ~ n^2",
        theory="cond_inf ~ n^2（差分算子 ~ (1/h^2)·三对角）",
        measured=f"n={ns.tolist()} 对应 cond={['%.3e' % c for c in cond]}；拟合指数 {fit['p']:.4f}（R^2={fit['r2']:.5f}）",
        tol="指数 2±0.15",
        passed=1.85 <= float(fit["p"]) <= 2.15,
        kind="C",
        extra={"exponent": float(fit["p"]), "r2": float(fit["r2"]), "cond": cond.tolist()},
    )
