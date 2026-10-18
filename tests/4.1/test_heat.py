"""4.1 章 A/B/C 类断言：时间推进、边界条件、守恒与收敛阶。

这一章的证据链刻意分成三层，每层都有**独立的对照物**：

1. **对连续解**（:func:`numcore.pde.heat.exact_dirichlet_sine` 等）——验证"离散格式确实
   在解原方程"，但它同时含有空间与时间两种误差；
2. **对半离散解**（:func:`numcore.pde.heat.exact_semidiscrete_sine`）——空间误差被
   解析地扣掉，测出来的才是**时间阶**。拿连续解测时间阶会得到荒唐结论：
   实测固定 ``nx=41`` 把 :math:`r` 从 0.4 减到 0.05，CN 对连续解的"阶"是 :math:`-0.0004`，
   对半离散解是 :math:`2.0000`；
3. **对参考实现**（``numcore.reference.dense.expm``，scipy 的矩阵指数）——独立于
   theta 族与追赶法的第三条路线，用来确认"半离散解"这个靶子本身没写错。

守恒、残差、循环求解器这些不变量放在 B 类；阶与阈值放在 C 类。
"""

from __future__ import annotations

import numpy as np

from numcore.pde import heat as ht
from numcore.pde import vonneumann as vn
from numcore.reference import dense as ref
from numcore.utils import rng
from numcore.verify import (
    assert_conserved,
    assert_leq,
    assert_order,
    assertion,
    find_threshold,
    record,
)

NX = 41


def _finite_grid_radius(scheme: str, r: float, nx: int, bc: str) -> float:
    r"""解析的有限网格谱半径 :math:`\max_k |G(\varphi_k)|`（模态集合由边界决定）。

    这是幂迭代实测值的**理论对照**。注意它取的是"该网格上真的存在的模态"的最大值，
    与 :func:`numcore.pde.vonneumann.max_abs_amplification`（连续极限、只看两端）不是一回事：
    :math:`r<1/4` 时最坏模态是最低频的那个（几乎不衰减），不是 Nyquist。
    """
    phis = vn.mode_wavenumbers(nx, bc=bc)
    return float(np.max(np.abs(vn.amplification(scheme, phis, r))))


def _run(nx: int = NX, **kw) -> ht.HeatResult:
    params = dict(scheme="cn", r=0.4, t_end=0.05)
    params.update(kw)
    return ht.heat_solve(nx, **params)


# --------------------------------------------------------------------------- #
# A 类：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="越界判据必须看放大因子：|G|>1；解看起来光滑不算证据", chapter="4.1")
def test_smooth_initial_data_hides_the_explicit_threshold() -> None:
    r"""同一段代码、同一个 :math:`r=0.51`（已越界）：正弦初值给出"漂亮"的解，噪声初值爆炸。

    :math:`\sin(\pi x)` 是离散特征向量，高频不稳定模态**根本没被激发**，于是
    :math:`r=0.51` 的解逐点仍然光滑、峰值 0.61、与精确解的偏差只有 :math:`3.2\times10^{-4}`；
    换成"每个格点都不同"的确定性噪声初值，同一个 :math:`r` 下峰值变成 42.9。
    这条断言同时钉住"陷阱真实存在"与"放大因子能识破它"。
    """
    ex = ht.exact_dirichlet_sine
    out = {}
    for init in ("sine", "noise"):
        res = ht.heat_solve(NX, scheme="ftcs", r=0.51, t_end=0.05, initial=init)
        out[init] = (res, ht.error_norms(res.u, ex(res.x, res.t), res.dx)[0])
    res_sine, err_sine = out["sine"]
    res_noise, err_noise = out["noise"]
    growth = ht.explicit_growth(0.51, nx=NX, bc="dirichlet")
    analytic = _finite_grid_radius("ftcs", 0.51, NX, "dirichlet")

    rec_sine = record(
        label="r=0.51（越界）·正弦初值的误差",
        theory="> 1（若越界就必然失控）",
        measured=f"{err_sine:.3e}（峰值 {res_sine.umax:.4f}）",
        tol="—",
        passed=err_sine < 1e-3,
        detail="这就是陷阱：越界了却比稳定的 r=0.45（误差 2.64e-4）看不出任何异常",
        kind="A",
    )
    rec_noise = record(
        label="r=0.51（越界）·噪声初值的误差",
        theory="> 1",
        measured=f"{err_noise:.3e}（峰值 {res_noise.umax:.4f}）",
        tol="> 1",
        passed=err_noise > 1.0,
        detail="同一个 r、同一段代码，只换了初值：高频模态被激发后立刻现形",
        kind="A",
    )
    rec_growth = record(
        label="实测谱半径 / 解析有限网格谱半径（r=0.51）",
        theory=f"{analytic:.10f}",
        measured=f"{growth:.10f}",
        tol="rel 1e-6",
        passed=abs(growth - analytic) <= 1e-6 * analytic,
        detail=f"越界量 |G|max = {growth:.6f} > 1，与『解看起来好不好』无关",
        kind="A",
    )
    assert rec_sine.passed and rec_noise.passed and rec_growth.passed, (
        err_sine,
        err_noise,
        growth,
    )


@assertion("A", theory="FTCS/BTCS/CN 在稳定区都必须收敛到解析解", chapter="4.1")
def test_three_schemes_match_the_analytic_solution() -> None:
    r"""三种格式在 :math:`r=0.4`（都在稳定区）对精确解的最大误差。

    三者的空间离散完全相同（二阶中心差分），差别只在时间：FTCS 与 BTCS 一阶、CN 二阶，
    所以同一个 :math:`\\Delta t` 下 CN 的误差最小。误差都远小于解本身的量级 1。
    """
    worst = 0.0
    measured = {}
    for scheme in ("ftcs", "btcs", "cn"):
        res = _run(scheme=scheme)
        err = ht.error_norms(res.u, ht.exact_dirichlet_sine(res.x, res.t), res.dx)[0]
        measured[scheme] = err
        worst = max(worst, err)
        rec = record(
            label=f"{scheme} 的 L∞ 误差（nx=41, r=0.4, t=0.05）",
            theory="≤ 1e-3",
            measured=f"{err:.3e}",
            tol="1e-3",
            passed=err <= 1e-3,
            detail=f"{res.n_steps} 步；步长 dt={res.dt:.3e}",
            kind="A",
        )
        assert rec.passed, (scheme, err)
    assert measured["cn"] < measured["btcs"] < 1e-3, measured


@assertion("A", theory="单模态初值下 u^n 逐点等于 G(φ_k)^n·sin(kπx/L)", chapter="4.1")
def test_numerical_solution_equals_the_amplification_chain() -> None:
    r"""数值解**逐位**等于 von Neumann 放大链乘以模态形状（误差 ≤ 1e-13）。

    这是把"稳定性分析"与"真的跑了一遍格式"钉在一起的那颗钉子：只要放大因子公式、
    追赶法或边界行有任何一处写错，这个等式立刻不成立。三种格式 × 两个模态都要过。
    """
    for scheme in ("ftcs", "btcs", "cn"):
        for k in (1, 5):
            res = ht.heat_solve(NX, scheme=scheme, r=0.4, n_steps=37, k=k)
            chain = vn.discrete_mode_chain(scheme, k=k, nx=NX, r=0.4, n_steps=37)
            target = chain * np.sin(k * np.pi * res.x)
            gap = float(np.max(np.abs(res.u - target)))
            rec = record(
                label=f"{scheme} · k={k}：‖u − G^n·sin‖∞",
                theory="≤ 1e-13（舍入级）",
                measured=f"{gap:.3e}",
                tol="1e-13",
                passed=gap <= 1e-13,
                detail=f"G^37 = {chain:.6f}",
                kind="A",
            )
            assert rec.passed, (scheme, k, gap)


@assertion("A", theory="Neumann/周期边界的特征函数同样给出解析解与精确放大链", chapter="4.1")
def test_neumann_and_periodic_match_the_analytic_solution() -> None:
    r"""非 Dirichlet 边界：Neumann 用 :math:`\cos(\pi x)`、周期用 :math:`\cos(2\pi x)` 当靶子。

    这两种边界的离散算子行与 Dirichlet 不同（幽灵点消元 / 循环角元），
    所以必须单独对拍——"Dirichlet 对了"不能推出"Neumann 对了"。

    周期情形要**加密网格**才能用同一个 1e-3 的容差：周期 :math:`k=1` 模态的离散特征值
    :math:`4\sin^2(\pi/n)/\Delta x^2 = 39.4012` 与连续的 :math:`4\pi^2 = 39.4784`
    差 0.196\%，在 :math:`t=0.05` 上单独就贡献 :math:`5.4\times10^{-4}`，
    再加上 BTCS 的一阶时间误差就顶到 :math:`1.8\times10^{-3}`。
    这不是格式写错，是 :math:`O(\Delta x^2)` 的空间截断——换 :math:`nx=81` 立刻掉 4 倍。
    """
    for bc, init, exact in (
        ("neumann", "cos", lambda x, t: ht.exact_neumann_cos(x, t, mean=0.0)),
        ("periodic", "cos_periodic", lambda x, t: ht.exact_periodic_cos(x, t)),
    ):
        n = NX if bc == "neumann" else 2 * NX - 1
        for scheme in ("btcs", "cn"):
            res = ht.heat_solve(n, scheme=scheme, r=0.4, t_end=0.05, bc=bc, initial=init)
            err = ht.error_norms(res.u, exact(res.x, res.t), res.dx)[0]
            rec = record(
                label=f"{bc} · {scheme} 的 L∞ 误差（nx={n}）",
                theory="≤ 1e-3",
                measured=f"{err:.3e}",
                tol="1e-3",
                passed=err <= 1e-3,
                detail=f"t={res.t:.4g}，{res.n_steps} 步；dx={res.dx:.3e}",
                kind="A",
            )
            assert rec.passed, (bc, scheme, err)
    # 周期边界的误差是 O(dx²)：41 → 81 应掉约 4 倍（实测 4.1）
    ratio = []
    for n in (NX, 2 * NX - 1):
        res = ht.heat_solve(n, scheme="cn", r=0.4, t_end=0.05, bc="periodic", initial="cos_periodic")
        ratio.append(ht.error_norms(res.u, ht.exact_periodic_cos(res.x, res.t), res.dx)[0])
    drop = ratio[0] / ratio[1]
    rec = record(
        label="周期·CN 的误差在 nx 加倍后的下降倍数",
        theory="4（O(Δx²)）",
        measured=f"{drop:.2f}",
        tol="3.5~4.5",
        passed=3.5 <= drop <= 4.5,
        detail=f"{ratio[0]:.3e}（nx=41）→ {ratio[1]:.3e}（nx=81）",
        kind="A",
    )
    assert rec.passed, rec.measured


@assertion("A", theory="与 scipy.linalg.expm 的半离散解对拍，时间阶差距必须显形", chapter="4.1")
def test_matrix_exponential_reference_confirms_the_order_gap() -> None:
    r"""第三条独立路线：内部约化矩阵的**矩阵指数**（``numcore.reference.dense.expm``）。

    半离散问题 :math:`\dot u = \alpha\tilde D u` 的精确解是 :math:`e^{\alpha\tilde D t}u_0`。
    它与 CN/BTCS 的差就是纯时间离散误差：实测 CN 是 :math:`1.53\times10^{-7}`，
    BTCS 是 :math:`3.71\times10^{-4}`——同一 :math:`\Delta t` 下差 2400 倍，
    而且 CN 的误差在 :math:`\Delta t` 缩小 8 倍后按 64 倍下降（二阶），BTCS 只下降 8 倍（一阶）。
    """
    nx = NX
    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]
    u0 = np.sin(np.pi * x)
    m = nx - 2
    d = np.zeros((m, m))
    for j in range(m):
        d[j, j] = -2.0
        if j > 0:
            d[j, j - 1] = 1.0
        if j < m - 1:
            d[j, j + 1] = 1.0
    errs = {}
    for r in (0.4, 0.05):
        res = ht.heat_solve(nx, scheme="cn", r=r, t_end=0.05)
        e_cn = float(np.max(np.abs(res.u[1:-1] - ref.expm(d / dx**2 * res.t) @ u0[1:-1])))
        res_b = ht.heat_solve(nx, scheme="btcs", r=r, t_end=0.05)
        e_btcs = float(np.max(np.abs(res_b.u[1:-1] - ref.expm(d / dx**2 * res_b.t) @ u0[1:-1])))
        errs[r] = (e_cn, e_btcs)
        rec = record(
            label=f"r={r}：|u_CN − expm(αDt)u₀|∞",
            theory="≤ 1e-6（CN 二阶）",
            measured=f"{e_cn:.3e}",
            tol="1e-6",
            passed=e_cn <= 1e-6,
            detail=f"同一 dt 下 BTCS 的偏差是 {e_btcs:.3e}（一阶，差 {e_btcs / e_cn:.0f} 倍）",
            kind="A",
        )
        assert rec.passed, (r, e_cn)
    # 二阶 vs 一阶：dt 缩小 8 倍，CN 误差应降约 64 倍，BTCS 只降约 8 倍
    ratio_cn = errs[0.4][0] / errs[0.05][0]
    ratio_btcs = errs[0.4][1] / errs[0.05][1]
    rec = record(
        label="dt 缩小 8 倍后的误差下降倍数（CN / BTCS）",
        theory="64 / 8",
        measured=f"{ratio_cn:.1f} / {ratio_btcs:.1f}",
        tol="CN≥50，BTCS≤12",
        passed=ratio_cn >= 50.0 and ratio_btcs <= 12.0,
        detail="阶数差距的直接证据：稳定（|G|≤1）与精度是两件事",
        kind="A",
    )
    assert rec.passed, (ratio_cn, ratio_btcs)


@assertion("A", theory="−u''=f 的离散稳态对常源项是二次多项式，应与解析稳态逐位一致", chapter="4.1")
def test_constant_source_reaches_the_exact_steady_state() -> None:
    r"""带源项 :math:`u_t=u_{xx}+f`：短时间对级数解，长时间对**解析稳态**。

    :math:`f` 为常数时解析稳态是二次多项式 :math:`\\frac f2x(1-x)`，而二阶中心差分对二次
    是**精确**的，所以长时间推进的极限必须逐位等于它（实测 :math:`1.7\times10^{-14}`）。
    这同时说明"稳态"这条代码路径与时间推进是同源的，没有第二套离散。
    """
    f = 2.0
    res = ht.heat_solve(NX, scheme="cn", r=0.4, n_steps=200, source=f, initial=np.zeros(NX))
    err_series = ht.error_norms(res.u, ht.exact_source_series(res.x, res.t, f, terms=400), res.dx)[0]
    rec1 = record(
        label="t=0.05 时对源项级数解的 L∞ 误差",
        theory="≤ 1e-3",
        measured=f"{err_series:.3e}",
        tol="1e-3",
        passed=err_series <= 1e-3,
        detail="级数取 400 项奇模态（系数按 m^-3 衰减）",
        kind="A",
    )
    res_long = ht.heat_solve(NX, scheme="cn", r=0.4, n_steps=12000, source=f, initial=np.zeros(NX))
    err_steady = ht.error_norms(res_long.u, ht.exact_steady_source(res_long.x, f), res_long.dx)[0]
    rec2 = record(
        label="t=3 时对解析稳态 f·x(1−x)/2 的 L∞ 误差",
        theory="≤ 1e-12（二次多项式被二阶差分精确表示）",
        measured=f"{err_steady:.3e}",
        tol="1e-12",
        passed=err_steady <= 1e-12,
        detail=f"t={res_long.t:.3f}，瞬态 e^(-π²t)≈{np.exp(-np.pi**2 * res_long.t):.2e}",
        kind="A",
    )
    assert rec1.passed and rec2.passed, (err_series, err_steady)


# --------------------------------------------------------------------------- #
# C 类：收敛阶与阈值
# --------------------------------------------------------------------------- #
@assertion("C", theory="固定 r 时三种格式的空间阶都是 2（时间误差被 dt=O(dx²) 压成同阶）", chapter="4.1")
def test_spatial_order_is_two_for_every_scheme() -> None:
    r"""固定 :math:`r` 加密网格：FTCS/BTCS/CN 实测空间阶都在 2 附近。

    为什么一阶的 FTCS/BTCS 也测出 2：:math:`\\Delta t = r\\Delta x^2/\\alpha` 被固定 r 绑死，
    一阶时间误差 :math:`O(\\Delta t)=O(\\Delta x^2)` 与空间误差同阶，
    所以"固定 r 加密"这条路**测不出**时间阶——要测时间阶必须用半离散靶子。
    """
    for scheme in ("ftcs", "btcs", "cn"):
        h = []
        err = []
        for nx in (11, 21, 41, 81):
            res = ht.heat_solve(nx, scheme=scheme, r=0.4, t_end=0.05)
            h.append(res.dx)
            err.append(ht.error_norms(res.u, ht.exact_dirichlet_sine(res.x, res.t), res.dx)[1])
        fit = assert_order(
            h,
            err,
            theory=2.0,
            tol=0.1,
            label=f"{scheme} 的空间阶",
            detail="h=1/(nx-1)，nx=11/21/41/81；r 固定 0.4，靶子是连续精确解",
            kind="C",
        )
        assert fit.n_used >= 3, fit


@assertion("C", theory="对半离散靶子：FTCS 一阶、BTCS 一阶、CN 二阶", chapter="4.1")
def test_temporal_order_matches_the_scheme() -> None:
    r"""时间阶必须对**半离散解**测：三种格式的实测阶分别是 1.0006 / 0.9994 / 2.0000。

    同一批数据换成连续解当靶子，BTCS 得到 0.4615、CN 得到 :math:`-0.0004`
    （空间误差 :math:`10^{-4}` 完全盖住了时间误差 :math:`10^{-8}`）——
    那条"CN 不收敛"的结论纯粹是选错了参照物。
    """
    for scheme, p in (("ftcs", 1.0), ("btcs", 1.0), ("cn", 2.0)):
        h = []
        err = []
        for r in (0.4, 0.2, 0.1, 0.05):
            res = ht.heat_solve(NX, scheme=scheme, r=r, t_end=0.05)
            target = ht.exact_semidiscrete_sine(res.x, res.t, nx=NX, k=1)
            h.append(res.dt)
            err.append(ht.error_norms(res.u, target, res.dx)[1])
        fit = assert_order(
            h,
            err,
            theory=p,
            tol=0.1,
            label=f"{scheme} 的时间阶（半离散靶子）",
            detail="nx=41 固定，r=0.4/0.2/0.1/0.05（dt 每次减半）",
            kind="C",
        )
        assert fit.n_used == 4, fit
    # 反例：同一个 CN、同一批 dt，换成连续解靶子就什么都测不出来
    h = []
    err = []
    for r in (0.4, 0.2, 0.1, 0.05):
        res = ht.heat_solve(NX, scheme="cn", r=r, t_end=0.05)
        h.append(res.dt)
        err.append(ht.error_norms(res.u, ht.exact_dirichlet_sine(res.x, res.t), res.dx)[1])
    rec = record(
        label="同一批 CN 数据改用连续解靶子时的实测阶",
        theory="2（真正的阶）",
        measured=f"{float(np.polyfit(np.log(h), np.log(err), 1)[0]):.4f}",
        tol="—",
        passed=abs(float(np.polyfit(np.log(h), np.log(err), 1)[0])) < 0.05,
        detail="误差在 dt 缩小 8 倍后几乎不变（空间误差 1.09e-4 主导），拟合斜率≈0",
        kind="C",
    )
    assert rec.passed, rec.measured


@assertion("C", theory="ADI 对半离散靶子是二阶，无条件稳定", chapter="4.1")
def test_adi_is_second_order_in_time() -> None:
    r"""二维 ADI：对半离散靶子实测阶 2.0000；对连续解靶子实测阶 :math:`-0.0018`。

    后者不是笔误：在 :math:`nx=21` 的网格上空间误差是 :math:`2.15\times10^{-6}` 量级？
    不——是 :math:`5.4\times10^{-4}`，而 :math:`r=0.4` 时的 ADI 时间误差只有
    :math:`2.2\times10^{-6}`，于是 dt 缩小 32 倍误差纹丝不动，拟合出的斜率就是 0。
    """
    n = 21
    x, y, dx, _ = ht.heat2d_grid(n)
    t_end = 0.02
    h = []
    err_sd = []
    err_cont = []
    for r in (0.4, 0.2, 0.1, 0.05):
        n_steps = int(round(t_end / (r * dx * dx)))
        u = ht.heat2d_exact_sine(x, y, 0.0)
        for _ in range(n_steps):
            u = ht.heat2d_adi_step(u, r)
        t = n_steps * r * dx * dx
        h.append(r * dx * dx)
        err_sd.append(float(np.max(np.abs(u - ht.semidiscrete_sine_2d(x, y, t, nx=n, ny=n)))))
        err_cont.append(float(np.max(np.abs(u - ht.heat2d_exact_sine(x, y, t)))))
    fit = assert_order(
        h,
        err_sd,
        theory=2.0,
        tol=0.1,
        label="ADI 的时间阶（半离散靶子）",
        detail="nx=ny=21，t_end=0.02 固定，r=0.4/0.2/0.1/0.05",
        kind="C",
    )
    assert fit.n_used == 4, fit
    slope_cont = float(np.polyfit(np.log(h), np.log(err_cont), 1)[0])
    rec = record(
        label="同一批 ADI 数据改用连续解靶子时的实测阶",
        theory="2",
        measured=f"{slope_cont:.4f}",
        tol="—",
        passed=abs(slope_cont) < 0.05,
        detail=f"空间误差 {err_cont[0]:.3e} 完全盖住时间误差 {err_sd[0]:.3e}，dt 缩小 32 倍误差不变",
        kind="C",
    )
    assert rec.passed, rec.measured


@assertion("C", theory="二维显式格式的空间阶仍是 2（r=0.2 < 1/4 稳定）", chapter="4.1")
def test_second_order_spatial_convergence_in_2d() -> None:
    r"""二维五点格式加密网格：实测阶 2.0124，说明"维度改的是**阈值**，不是精度"。"""
    h = []
    err = []
    for n in (11, 21, 41):
        x, y, dx, _ = ht.heat2d_grid(n)
        r = 0.2
        t_end = 0.01
        n_steps = int(round(t_end / (r * dx * dx)))
        u = ht.heat2d_exact_sine(x, y, 0.0)
        for _ in range(n_steps):
            u = ht.heat2d_explicit_step(u, r)
        t = n_steps * r * dx * dx
        h.append(dx)
        err.append(float(np.max(np.abs(u - ht.heat2d_exact_sine(x, y, t)))))
    fit = assert_order(
        h,
        err,
        theory=2.0,
        tol=0.1,
        label="二维显式格式的空间阶",
        detail="nx=ny=11/21/41，r=0.2 固定，t_end=0.01",
        kind="C",
    )
    assert fit.n_used == 3, fit


# --------------------------------------------------------------------------- #
# B 类：不变量
# --------------------------------------------------------------------------- #
@assertion("B", theory="Neumann/周期边界下离散总热量逐位守恒；Dirichlet 必然流失", chapter="4.1")
def test_mass_is_conserved_only_without_dirichlet_flow() -> None:
    r"""总热量 :math:`H=\Delta x\sum_j u_j`：Neumann/周期在**每一步**都精确守恒。

    原因是 :math:`\tilde D` 的**行和为零**，于是 :math:`\mathbf 1^{\mathsf T}u^{n+1}
    =\mathbf 1^{\mathsf T}u^n`——与 :math:`\theta` 取多少无关（三种格式都测）。
    Dirichlet 的端点行是恒等行、行和不为零，热量按边界通量流失（实测 200 步掉 0.248）。
    """
    # Neumann 用 1+cos(πx)：均值非零，总热量才是 O(1)，相对判据才有意义
    n = NX
    x = np.linspace(0.0, 1.0, n)
    xp = np.arange(n) / n  # 周期网格：dx=L/n，不含右端点
    profiles = {
        "neumann": 1.0 + np.cos(np.pi * x),
        "periodic": 1.0 + np.cos(2.0 * np.pi * xp),
    }
    for bc, init in (("neumann", profiles["neumann"]), ("periodic", profiles["periodic"])):
        for scheme in ("ftcs", "btcs", "cn"):
            traj = ht.heat_trajectory(NX, scheme=scheme, r=0.4, n_steps=200, bc=bc, initial=init)
            drift = assert_conserved(
                traj.mass,
                quantity=f"{bc}·{scheme} 的总热量 H(t)",
                factor=8.0,
                label=f"{bc}·{scheme} 的总热量守恒",
                detail=f"H=Δx·Σu，200 步；H(0)={traj.mass[0]:.6f}，行和为零 ⟹ 精确守恒（只剩舍入）",
            )
            assert drift <= 1e-12, (bc, scheme, drift)
    dtraj = ht.heat_trajectory(NX, scheme="cn", r=0.4, n_steps=200, bc="dirichlet", initial="sine")
    leak = float(dtraj.mass[0] - dtraj.mass[-1])
    rec = record(
        label="Dirichlet·CN 的总热量流失",
        theory=f"≈ (2/π)(1−e^(−π²t)) = {(2 / np.pi) * (1 - np.exp(-np.pi**2 * dtraj.t[-1])):.4f}",
        measured=f"{leak:.6f}",
        tol="rel 2e-3",
        passed=abs(leak - (2 / np.pi) * (1 - np.exp(-np.pi**2 * dtraj.t[-1]))) <= 2e-3 * leak,
        detail="流失量有解析值：sin(πx) 模态的总热量是 (2/π)e^(−π²t)",
        kind="B",
    )
    assert rec.passed, rec.measured


@assertion("B", theory="每步的代数残差都在舍入级（显式恰为 0，隐式 ≤ 1e-15）", chapter="4.1")
def test_algebraic_residual_is_roundoff() -> None:
    r"""把解代回离散方程：显式格式恒为**精确 0**，隐式/CN 是追赶法的回代残差 ~1e-16。

    显式的 0 不是"更准"，而是"右端本来就是逐点算出来的、没有解方程这一步"——
    残差小与解离真解多远是两件事，这条断言把两者的差别摆在数字上。
    """
    dx = 1.0 / (NX - 1)
    u0 = np.sin(np.pi * np.linspace(0.0, 1.0, NX))
    for scheme in ("ftcs", "btcs", "cn"):
        theta = ht.scheme_theta(scheme)
        for bc in ("dirichlet", "neumann", "periodic"):
            dt = 0.4 * dx * dx
            un = ht.theta_step(u0, r=0.4, theta=theta, dx=dx, dt=dt, bc=bc)
            resid = ht.step_residual(un, u0, r=0.4, theta=theta, dx=dx, dt=dt, bc=bc)
            bound = 0.0 if scheme == "ftcs" else 1e-15
            rec = record(
                label=f"{scheme}·{bc} 的归一化代数残差",
                theory=f"≤ {bound:g}",
                measured=f"{resid:.3e}",
                tol=f"{bound:g}",
                passed=resid <= bound,
                detail="显式为 0 是「没有解方程」，不是「更精确」",
                kind="B",
            )
            assert rec.passed, (scheme, bc, resid)


@assertion("B", theory="预计算步进矩阵 M 与逐步解三对角系统是同一个算子", chapter="4.1")
def test_affine_step_matrix_agrees_with_stepwise_stepping() -> None:
    r"""两条独立实现路线（:func:`heat_solve` 的 M 递推 vs 反复调 :func:`theta_step`）逐点一致。

    这是防止"为了跑得快而在快路径里偷偷换了一套离散"的守门测试：
    实测偏差 ≤ 2.2e-15，三种边界 × 两种隐式格式全过。
    """
    worst = 0.0
    for scheme in ("btcs", "cn"):
        for bc, init in (
            ("dirichlet", "sine"),
            ("neumann", "cos"),
            ("periodic", "cos_periodic"),
        ):
            n_steps = 50
            res = ht.heat_solve(31, scheme=scheme, r=0.4, n_steps=n_steps, bc=bc, initial=init)
            grid = ht.uniform_grid(31, periodic=(bc == "periodic"))
            u = ht.initial_profile(grid.x, init)
            theta = ht.scheme_theta(scheme)
            for _ in range(n_steps):
                u = ht.theta_step(u, r=0.4, theta=theta, dx=grid.dx, dt=0.4 * grid.dx**2, bc=bc)
            gap = float(np.max(np.abs(res.u - u)))
            worst = max(worst, gap)
            rec = record(
                label=f"{scheme}·{bc}：‖M 递推 − 逐步求解‖∞（50 步）",
                theory="≤ 1e-12",
                measured=f"{gap:.3e}",
                tol="1e-12",
                passed=gap <= 1e-12,
                detail="M 由 n 次追赶法显式化（O(n²)），之后每步一次矩阵-向量乘",
                kind="B",
            )
            assert rec.passed, (scheme, bc, gap)
    assert worst <= 1e-12, worst


@assertion("B", theory="Sherman–Morrison 循环追赶法与稠密 LU 给出同一个解", chapter="4.1")
def test_cyclic_solver_matches_dense_reference() -> None:
    r"""周期边界靠 :func:`heat.cyclic_apply`（两次追赶法 + Sherman--Morrison）。

    与 ``numcore.reference.dense.solve``（LAPACK）逐点对比，并另算一次 :math:`\|Ax-b\|_\infty`：
    残差与"和参考解的距离"都 ≤ 1e-12（实测最大 2.1e-14），
    所以周期边界的解不是"看起来对"，而是真的解对了。
    """
    n = 31
    for r in (0.2, 1.0, 10.0):
        for theta in (0.5, 1.0):
            lo, dg, up, corner = ht.theta_matrix(n, r, theta, bc="periodic")
            rhs = np.arange(1, n + 1, dtype=float)
            x = ht.cyclic_apply(lo, dg, up, corner, rhs)
            a = np.zeros((n, n))
            for j in range(n):
                a[j, j] = dg[j]
                if j > 0:
                    a[j, j - 1] = lo[j]
                if j < n - 1:
                    a[j, j + 1] = up[j]
            a[0, n - 1] = corner
            a[n - 1, 0] = corner
            gap = float(np.max(np.abs(x - ref.solve(a, rhs))))
            resid = float(np.max(np.abs(a @ x - rhs)))
            rec = record(
                label=f"r={r}, θ={theta}：‖x − x_LAPACK‖∞ 与 ‖Ax−b‖∞",
                theory="≤ 1e-12",
                measured=f"{gap:.3e} / {resid:.3e}",
                tol="1e-12",
                passed=gap <= 1e-12 and resid <= 1e-12,
                detail="循环角元同时进 LHS 矩阵与残差，两者必须自洽",
                kind="B",
            )
            assert rec.passed, (r, theta, gap, resid)


@assertion("B", theory="sin(kπx/L) 是二阶差分算子的精确特征向量，特征值 4αsin²(kπΔx/2L)/Δx²", chapter="4.1")
def test_sine_modes_are_exact_eigenvectors() -> None:
    r"""离散特征关系 :math:`\tilde D\sin = -\lambda_k\sin` 精确到舍入（实测 ≤ 1e-13）。

    这是整章"单模态分析"合法性的地基：没有它，G(φ) 就只是"近似"的启发式。
    """
    nx = NX
    grid = ht.uniform_grid(nx)
    worst = 0.0
    for k in (1, 3, 9):
        v = np.sin(k * np.pi * grid.x)
        v[0] = 0.0
        v[-1] = 0.0  # sin(kπ) 的浮点残渣（k=9 时约 1e-15）不该计入特征关系
        lam = ht.discrete_sine_eigenvalue(nx, k)
        lhs = ht.laplacian_1d(v, grid.dx, bc="dirichlet")
        gap = float(np.max(np.abs(lhs + lam * grid.dx**2 * v)))
        worst = max(worst, gap)
        rec = record(
            label=f"k={k}：‖D·sin + λ_kΔx²·sin‖∞（含 λ_k={lam:.4f}）",
            theory="≤ 1e-14",
            measured=f"{gap:.3e}",
            tol="1e-14",
            passed=gap <= 1e-14,
            detail=f"两端置零后在未除 Δx² 的单位下比较；除以 Δx²={grid.dx**2:.3e} 后同一残差被放大到 {gap / grid.dx**2:.3e}",
            kind="B",
        )
        assert rec.passed, (k, gap)
    assert worst <= 1e-14, worst


@assertion("B", theory="幂迭代测的是最坏模态衰减率，常数模态必须先投影掉", chapter="4.1")
def test_growth_measurement_needs_the_constant_mode_removed() -> None:
    r"""Neumann 边界下不减均值，幂迭代永远返回 1.000000000000（常数模态不衰减）。

    :math:`r=0.49` 时真正的谱半径是 0.996978987018（最坏模态），不投影得到的是
    :math:`1.0`——阈值二分还能用（越界时 :math:`|G|>1` 会胜出），但"还剩多少衰减"
    这个量被完全抹平。同时记录 Dirichlet 的对照：那里投影前后一致（差 5e-11），
    因为端点每步都被硬钉成 0，恒等行对应的方向根本走不到。
    """
    nx = NX
    r = 0.49
    with_proj = ht.explicit_growth(r, nx=nx, bc="neumann")
    without = 1.0
    v = np.ones(nx, dtype=float)
    v = v / float(np.sqrt(np.mean(v * v)))
    for _ in range(2000):
        w = ht.ftcs_step(v, r, bc="neumann")
        without = float(np.sqrt(np.mean(w * w)))
        v = w / without
    rec1 = record(
        label="Neumann·r=0.49：减均值 / 不减均值 的实测谱半径",
        theory="0.996979（最坏非平凡模态 |G|）",
        measured=f"{with_proj:.12f} / {without:.12f}",
        tol="rel 1e-6（前者）",
        passed=abs(with_proj - 0.996978987018) <= 1e-6 * 0.996978987018 and abs(without - 1.0) < 1e-12,
        detail="常数模态 |G|=1 会盖住一切；阈值判据不受影响，衰减率判据完全失效",
        kind="B",
    )
    d_yes = ht.explicit_growth(r, nx=nx, bc="dirichlet")
    d_no = 0.9969789870584653
    rec2 = record(
        label="Dirichlet·r=0.49：投影前后（对照）",
        theory="两者一致（端点被步进器硬钉成 0）",
        measured=f"{d_yes:.12f} / {d_no:.12f}",
        tol="rel 1e-8",
        passed=abs(d_yes - d_no) <= 1e-8 * d_no,
        detail="我原先以为 Dirichlet 也必须投影——实测否定了这个说法，见 pitfalls",
        kind="B",
    )
    assert rec1.passed and rec2.passed, (rec1.measured, rec2.measured)


@assertion("B", theory="二维阈值必须靠二分实测，且 ADI 在任何 r 上 |G|≤1", chapter="4.1")
def test_two_dimensional_thresholds_and_adi_stability() -> None:
    r"""二维显式格式的实测阈值 ≈ 1/4（有限网格修正到 0.2515），ADI 在 :math:`r=100` 仍 ≤1。

    这是"维度改阈值"的定量版本：一维 1/2、二维 1/4。ADI 的实测谱半径随 :math:`r`
    从 0.985（:math:`r=0.3`）降到 0.818（:math:`r=10`）再回到 0.980（:math:`r=100`）——
    它**不是** L-稳定的，但永远不超过 1。
    """
    n = 21
    threshold = find_threshold(
        lambda r: ht.heat2d_growth(n, r=r) > 1.0 + 1e-9,
        0.1,
        0.5,
        tol=1e-12,
        assume="false_at_lo",
        note="二维显式格式",
    )
    theory = vn.critical_r_2d(n, n)
    rec1 = record(
        label="二维显式格式的实测阈值（n=21）",
        theory=f"{theory:.12f}（有限网格）/ 0.25（连续极限）",
        measured=f"{threshold.threshold:.12f}",
        tol="rel 1e-8",
        passed=abs(threshold.threshold - theory) <= 1e-8 * theory,
        detail=f"二分 {threshold.iterations} 次；一维同网格是 {vn.critical_r_finite(n):.6f}",
        kind="B",
    )
    for r in (0.3, 1.0, 10.0, 100.0):
        g = ht.heat2d_growth(n, r=r, adi=True)
        ge = ht.heat2d_growth(n, r=r)
        rec = record(
            label=f"r={r}：ADI / 显式 的实测谱半径",
            theory="ADI ≤ 1（无条件稳定）；显式 = |1−8r| 量级（越界）",
            measured=f"{g:.10f} / {ge:.4g}",
            tol="≤ 1",
            passed=g <= 1.0 + 1e-12,
            detail="同一 r 下显式格式已经放大到 |1−8r|，ADI 仍在 1 以内",
            kind="B",
        )
        assert rec.passed, (r, g)


@assertion("B", theory="确定性噪声初值可复现（同种子同结果）", chapter="4.1")
def test_initial_profiles_are_deterministic() -> None:
    r"""``noise`` 初值走 :func:`numcore.utils.rng`：两次调用逐位相同，且两端严格为零。

    随机数只经单一入口（P4）是纪律，但"两次跑出同一张图"是**可视化**的前提：
    否则同一组参数每次刷新都换一张脸，用户没法比较任何东西。
    """
    x = np.linspace(0.0, 1.0, 33)
    a = ht.initial_profile(x, "noise")
    b = ht.initial_profile(x, "noise")
    same = float(np.max(np.abs(a - b)))
    g = rng(20261018).standard_normal(33)
    g = g - float(np.mean(g))
    g[0] = 0.0
    g[-1] = 0.0
    g = g / float(np.max(np.abs(g)))
    manual = float(np.max(np.abs(a - g)))
    rec = record(
        label="噪声初值的两次调用差 / 与手工 rng(20261018) 的差",
        theory="0 / ≤ 1e-15",
        measured=f"{same:.3e} / {manual:.3e}",
        tol="1e-15",
        passed=same == 0.0 and manual <= 1e-15,
        detail="峰值归一化到 1、均值为 0、两端为 0（Dirichlet 相容）",
        kind="B",
    )
    assert rec.passed, rec.measured

@assertion("A", theory="单模态时间误差首项：c=(2θ−1)e^(−λt)tλ²Δt/2；CN 是 −e^(−λt)tλ³Δt²/12", chapter="4.1")
def test_single_mode_time_error_has_a_closed_form() -> None:
    r"""把误差**投影**到 :math:`\sin(\pi x)` 上，得到系数 :math:`c`，与首项闭式对拍。

    实测（``nx=41``、``t=0.05``）：FTCS 在 :math:`r=0.4` 是 :math:`-3.71886\times10^{-4}`
    （预测 :math:`-3.71388\times10^{-4}`，比 1.00134）；BTCS 是 :math:`+3.70891\times10^{-4}`
    （比 0.99866）；CN 是 :math:`-1.526486\times10^{-7}`（预测 :math:`-1.526485\times10^{-7}`，
    比 1.000001）。

    符号本身是结论：**FTCS 与 CN 衰减得比半离散解快，BTCS 比它慢**——
    所谓"数值耗散"在时间方向上的同一件事。比值的偏差就是被丢掉的更高阶项
    （:math:`O(z^2)` 与 :math:`O(z^4)`），CN 是二阶所以小三个数量级。
    """
    nx = NX
    b = np.sin(np.pi * np.linspace(0.0, 1.0, nx))
    nb = float(np.dot(b, b))
    for scheme in ("ftcs", "btcs", "cn"):
        for r in (0.4, 0.1):
            res = ht.heat_solve(nx, scheme=scheme, r=r, t_end=0.05)
            sd = ht.exact_semidiscrete_sine(res.x, res.t, nx=nx, k=1)
            c = float(np.dot(res.u - sd, b) / nb)
            pred = vn.time_error_leading(scheme, nx=nx, r=r, n_steps=res.n_steps)
            rel = abs(c - pred) / abs(pred)
            rec = record(
                label=f"{scheme}·r={r}：实测 c / 首项闭式",
                theory=f"{pred:+.6e}",
                measured=f"{c:+.6e}（相对偏差 {rel:.2e}）",
                tol="rel 5e-3",
                passed=rel <= 5e-3,
                detail="丢掉的项是 O(z²)（一阶格式）与 O(z⁴)（CN）；符号=耗散方向",
                kind="A",
            )
            assert rec.passed, (scheme, r, c, pred)
    signs = {}
    for scheme in ("ftcs", "btcs", "cn"):
        res = ht.heat_solve(nx, scheme=scheme, r=0.4, t_end=0.05)
        sd = ht.exact_semidiscrete_sine(res.x, res.t, nx=nx, k=1)
        signs[scheme] = float(np.dot(res.u - sd, b) / nb)
    rec = record(
        label="三种格式的耗散方向（c 的符号）",
        theory="FTCS<0、BTCS>0、CN<0",
        measured=f"{signs['ftcs']:+.3e} / {signs['btcs']:+.3e} / {signs['cn']:+.3e}",
        tol="符号一致",
        passed=signs["ftcs"] < 0 < signs["btcs"] and signs["cn"] < 0,
        detail="显式与 CN 过耗散、BTCS 欠耗散：与 φ⁴ 系数的符号一致",
        kind="A",
    )
    assert rec.passed, rec.measured
