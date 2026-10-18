"""4.1 章 A/C 类断言：放大因子、稳定域、阈值、数值耗散与"相位"。

这一组测试的主角是 :mod:`numcore.pde.vonneumann`，但每一条都以**真的跑了一遍格式**为证据：

* **放大因子**用两套互相独立的实特征向量去喂推进器——周期网格上的余弦
  （走 Sherman--Morrison 循环追赶法）与 Dirichlet 网格上的正弦（走普通追赶法）。
  两者与解析式 :math:`G_\\theta(\\varphi)` 都必须吻合到舍入级。
* **阈值**一律用二分**测**出来，理论值只用来对拍。实测与理论差 5e-10 的相对量级，
  这不是"凑出来的"：理论式 :math:`r_*=1/(2\\sin^2(\\varphi_{\\max}/2))` 里
  :math:`\\varphi_{\\max}` 取哪个模态由边界（以及网格奇偶性）决定，
  三种边界给出三个不同的数，教科书那个 1/2 只是它们共同的极限。
* **无条件稳定 ≠ 精确**：CN 在 :math:`r=50` 时实测谱半径 0.98（稳定），
  但同一次推进的误差是稳定的 :math:`r=0.4` 的几百倍。
"""

from __future__ import annotations

import numpy as np

from numcore.pde import heat as ht
from numcore.reference import dense as ref_dense
from numcore.pde import vonneumann as vn
from numcore.verify import (
    assert_leq,
    assert_order,
    assert_threshold,
    assertion,
    find_threshold,
    record,
)

NX = 41


def _finite_grid_radius_2d(r: float, nx: int, *, adi: bool = False) -> float:
    r"""解析的二维有限网格谱半径（扫描网格上真实存在的 :math:`(\\varphi_x,\\varphi_y)`）。"""
    phis = vn.mode_wavenumbers(nx, bc="dirichlet")
    px = phis[:, None]
    py = phis[None, :]
    g = vn.adi_amplification_2d(px, py, r) if adi else vn.amplification_2d_ftcs(px, py, r)
    return float(np.max(np.abs(g)))


# --------------------------------------------------------------------------- #
# A 类：解析式 vs 真实推进器
# --------------------------------------------------------------------------- #
@assertion("A", theory="实测放大因子 = G_θ(φ)，两套特征向量路线都吻合到舍入级", chapter="4.1")
def test_measured_amplification_matches_the_analytic_formula() -> None:
    r"""周期余弦与 Dirichlet 正弦两条路线都收敛到同一个 :math:`G_\theta(\varphi)`。

    实测最大相对偏差 :math:`3.7\times10^{-16}`。这条断言一次性钉住三件事：
    放大因子公式、三对角/循环追赶法、以及"单模态确实是特征向量"这个前提。
    """
    worst = 0.0
    worst_desc = ""
    for scheme in ("ftcs", "btcs", "cn"):
        for r in (0.1, 0.3, 0.49):
            for m in (1, 2, 3, 7):
                phi, g = vn.numerical_amplification(scheme, m=m, nx=32, r=r)
                ga = abs(float(vn.amplification(scheme, phi, r)))
                rel = abs(g - ga) / ga if ga > 0 else abs(g)
                if rel > worst:
                    worst, worst_desc = rel, f"周期 m={m}, r={r}, {scheme}"
            for k in (1, 3, 9):
                phi, g = vn.numerical_amplification_sine(scheme, k=k, nx=32, r=r)
                ga = abs(float(vn.amplification(scheme, phi, r)))
                rel = abs(g - ga) / ga if ga > 0 else abs(g)
                if rel > worst:
                    worst, worst_desc = rel, f"Dirichlet k={k}, r={r}, {scheme}"
    rec = record(
        label="实测放大因子与解析式的最大相对偏差",
        theory="≤ 1e-12（舍入级）",
        measured=f"{worst:.3e}",
        tol="1e-12",
        passed=worst <= 1e-12,
        detail=f"最差点：{worst_desc}；覆盖 3 格式 × 3 个 r × 7 个模态",
        kind="A",
    )
    assert rec.passed, rec.measured


@assertion("A", theory="G(0)=1、G_FTCS(π)=1−4r、|G| 单调、极值只在 φ∈{0,π}", chapter="4.1")
def test_amplification_shape_and_endpoints() -> None:
    r"""放大因子的形状：端点值精确、单调、极值在端点——这就是"只看两个数就能判稳定"的依据。

    另外记录符号翻转波数 :math:`\varphi_*=2\\arcsin\\frac1{2\\sqrt r}`：
    :math:`r=0.4` 时 :math:`\\varphi_*=1.8235`，越过它的模态每步翻一次符号。
    """
    phis = np.linspace(0.0, np.pi, 2001)
    for scheme in ("ftcs", "btcs", "cn"):
        for r in (0.2, 0.4):
            g = np.asarray(vn.amplification(scheme, phis, r))
            g0 = float(vn.amplification(scheme, 0.0, r))
            gpi = float(vn.amplification(scheme, np.pi, r))
            mono = bool(np.all(np.diff(g) <= 1e-15))
            ok = abs(g0 - 1.0) <= 1e-15 and mono
            if scheme == "ftcs":
                ok = ok and abs(gpi - (1.0 - 4.0 * r)) <= 1e-15
            rec = record(
                label=f"{scheme}·r={r}：G(0)−1、单调性、G(π)",
                theory=f"G(0)=1，G(π)={1 - 4 * r if scheme == 'ftcs' else float(vn.amplification(scheme, np.pi, r)):.6f}",
                measured=f"{g0 - 1.0:.1e}，单调={mono}，G(π)={gpi:.6f}",
                tol="1e-15",
                passed=ok,
                detail="分子单调减、分母单调增 ⟹ |G| 的极值必在 φ=0 或 φ=π",
                kind="A",
            )
            assert rec.passed, (scheme, r, g0, gpi, mono)
    r = 0.4
    phi_star = vn.sign_flip_wavenumber(r)
    assert phi_star is not None
    g_star = float(vn.amplification("ftcs", phi_star, r))
    assert vn.sign_flip_wavenumber(0.2) is None
    rec = record(
        label="FTCS 的符号翻转波数 φ*（r=0.4）",
        theory=f"2·arcsin(1/(2√r)) = {2 * np.arcsin(1 / (2 * np.sqrt(r))):.10f}",
        measured=f"{phi_star:.10f}，G(φ*)={g_star:.2e}",
        tol="1e-12",
        passed=abs(g_star) <= 1e-14 and vn.sign_flip_wavenumber(0.2) is None,
        detail="r<1/4 时 G 恒正（无符号翻转）；r=0.2 实测返回 None",
        kind="A",
    )
    assert rec.passed, rec.measured


@assertion("A", theory="G(φ)=1−rφ²+(r/12+θr²)φ⁴+O(φ⁶)，系数由 Richardson 外推独立验证", chapter="4.1")
def test_symbol_expansion_coefficients() -> None:
    r"""解析的 :math:`\\varphi^4` 系数与数值 Richardson 外推一致（相对偏差 ≤ 5e-3）。

    这个系数就是"数值耗散"的来源：精确解的对应系数是 :math:`r^2/2`，
    两者之差 :func:`dissipation_coefficient` 决定了模态衰减得比真解快还是慢。
    """
    for scheme in ("ftcs", "btcs", "cn"):
        for r in (0.4, 2.0):
            c4 = vn.symbol_expansion(scheme, r)[2]
            est = []
            for phi in (2e-3, 1e-3):
                g = float(vn.amplification(scheme, phi, r))
                est.append((g - 1.0 + r * phi * phi) / phi**4)
            rich = 2.0 * est[1] - est[0]
            rel = abs(c4 - rich) / abs(c4)
            rec = record(
                label=f"{scheme}·r={r} 的 φ⁴ 系数：解析 vs 外推",
                theory=f"{c4:.8f}",
                measured=f"{rich:.8f}（相对偏差 {rel:.2e}）",
                tol="rel 5e-3",
                passed=rel <= 5e-3,
                detail="φ=2e-3 与 1e-3 两点外推；差值 r/12+r²(θ−1/2) 即耗散误差系数",
                kind="A",
            )
            assert rec.passed, (scheme, r, c4, rich)


# --------------------------------------------------------------------------- #
# C 类：阈值与收敛率
# --------------------------------------------------------------------------- #
@assertion("C", theory="r*=1/(2 sin²(φ_max/2))，φ_max 由边界决定：Dirichlet>1/2，Neumann=1/2", chapter="4.1")
def test_ftcs_thresholds_match_the_finite_grid_formula() -> None:
    r"""三种边界、两种网格的稳定阈值全部二分**实测**，并与有限网格公式对拍。

    实测：``nx=21`` 时 Dirichlet 0.5030969793、Neumann 0.5000000003、
    周期（奇数网格）0.5028079748；理论值分别是 0.5030969793 / 0.5 / 0.5028079748。
    Dirichlet 的阈值**大于** 1/2，因为端点被钉住后 :math:`\\varphi=\\pi` 那个模态
    根本不是容许模态；奇数周期网格也不含 :math:`\\pi`（:math:`m` 与 :math:`n-m`
    是同一个模态）。"r ≤ 1/2" 是 :math:`\\Delta x\\to0` 的极限，不是有限网格上的等式。
    """
    for nx in (20, 21):
        for bc in ("dirichlet", "neumann", "periodic"):
            measured = find_threshold(
                lambda r, bc=bc, nx=nx: ht.explicit_growth(r, nx=nx, bc=bc) > 1.0 + 1e-9,
                0.2,
                0.8,
                tol=1e-12,
                assume="false_at_lo",
                note=f"nx={nx}, {bc}",
            )
            theory = vn.critical_r_finite(nx, bc=bc)
            assert_threshold(
                measured,
                theory,
                rtol=1e-8,
                label=f"{bc}·nx={nx} 的 FTCS 稳定阈值",
                detail=f"理论式 1/(2sin²(φ_max/2))，φ_max={(nx - 2) * np.pi / (nx - 1) if bc == 'dirichlet' else (2 * np.pi * (nx // 2) / nx if bc == 'periodic' else np.pi):.10f}",
            )
            rec = record(
                label=f"{bc}·nx={nx}：实测 − 1/2",
                theory=f"{theory - 0.5:+.10f}",
                measured=f"{measured.threshold - 0.5:+.10f}",
                tol="rel 1e-8",
                passed=abs(measured.threshold - theory) <= 1e-8 * theory,
                detail="教科书那句『r ≤ 1/2』是三条曲线共同的极限，不是每一条本身",
                kind="C",
            )
            assert rec.passed, rec.measured


@assertion("C", theory="Dirichlet 阈值对 1/2 的偏差按 π²/(8(n−1)²) 二阶收敛", chapter="4.1")
def test_dirichlet_threshold_deviation_is_second_order() -> None:
    r""":math:`r_*(nx)-\\frac12 = \\frac{\\pi^2}{8(n-1)^2}(1+O(n^{-2}))`：实测阶 2.000。

    把这句话写清楚很重要：如果只喊"阈值就是 1/2"，那么在 :math:`nx=21` 上取
    :math:`r=0.501` 就该判稳定，实测却已经发散（:math:`r_*=0.503097` 才是真阈值）。
    反过来说，:math:`r=0.5031` 在 :math:`nx=21` 上稳定，在 :math:`nx=81` 上就发散了。
    """
    h = []
    dev = []
    for nx in (11, 21, 41, 81):
        measured = find_threshold(
            lambda r, nx=nx: ht.explicit_growth(r, nx=nx, bc="dirichlet") > 1.0 + 1e-9,
            0.4,
            0.8,
            tol=1e-11,
            assume="false_at_lo",
            note=f"nx={nx}",
        )
        h.append(1.0 / (nx - 1))
        dev.append(measured.threshold - 0.5)
    fit = assert_order(
        h,
        dev,
        theory=2.0,
        tol=0.05,
        label="Dirichlet 阈值偏差 r*−1/2 的收敛阶",
        detail="nx=11/21/41/81；二阶二分到 1e-11，偏差本身是 1e-2…1e-4 量级",
        kind="C",
    )
    const = [d * (1.0 / hh) ** 2 for hh, d in zip(h, dev)]
    rec = record(
        label="偏差常数 (r*−1/2)(n−1)² 的四个取值",
        theory=f"π²/8 = {np.pi**2 / 8:.6f}",
        measured=f"{np.mean(const):.6f}（相对偏差 {abs(np.mean(const) - np.pi**2 / 8) / (np.pi**2 / 8):.2e}）",
        tol="rel 1e-2",
        passed=abs(np.mean(const) - np.pi**2 / 8) <= 1e-2 * np.pi**2 / 8,
        detail=f"R²={fit.r2:.6f}，实测斜率 {fit.slope:.4f}±{fit.stderr:.1e}",
        kind="C",
    )
    assert rec.passed, rec.measured


@assertion("C", theory="θ≥1/2 无条件稳定（r=50/500 谱半径仍 ≤1），但精度随 r 恶化", chapter="4.1")
def test_implicit_schemes_are_unconditionally_stable_but_not_accurate() -> None:
    r"""BTCS/CN 在 :math:`r=50` 与 :math:`r=500` 的实测谱半径都 ≤ 1——"稳定"到此为止。

    但同一次推进的**误差**随 :math:`r` 暴涨：BTCS 在 :math:`r=50`（:math:`\\Delta t=0.031`）
    的 :math:`L_\\infty` 误差是 :math:`r=0.4` 时的几百倍。CN 的谱半径在
    :math:`r\\to\\infty` 时趋近 1（:math:`G(\\pi)\\to-1`），于是"稳定"变成了
    "每步翻符号、永不衰减"——它在 :math:`r=500` 时的实测值是 0.9980。
    """
    for scheme in ("btcs", "cn"):
        for r in (50.0, 500.0):
            g = ht.step_growth(31, scheme=scheme, r=r)
            rec = record(
                label=f"{scheme}·r={r} 的实测谱半径",
                theory="≤ 1（无条件稳定）",
                measured=f"{g:.10f}",
                tol="≤ 1",
                passed=g <= 1.0 + 1e-12,
                detail=f"解析 |G(π)|={abs(float(vn.amplification(scheme, np.pi, r))):.3e}；r→∞ 时 CN→1、BTCS→0",
                kind="C",
            )
            assert rec.passed, (scheme, r, g)
    errs = {}
    for r in (0.4, 50.0):
        res = ht.heat_solve(101, scheme="btcs", r=r, t_end=0.05)
        errs[r] = ht.error_norms(res.u, ht.exact_dirichlet_sine(res.x, res.t), res.dx)[0]
    rec = record(
        label="BTCS 的 L∞ 误差：r=50 与 r=0.4 的比",
        theory="≫ 1（稳定 ≠ 精确）",
        measured=f"{errs[50.0] / errs[0.4]:.1f} 倍",
        tol="≥ 20",
        passed=errs[50.0] / errs[0.4] >= 20.0,
        detail=f"r=0.4 误差 {errs[0.4]:.3e}；r=50 误差 {errs[50.0]:.3e}（Δt 从 4.0e-5 涨到 5.0e-3）",
        kind="C",
    )
    assert rec.passed, rec.measured


@assertion("C", theory="FTCS 的耗散误差系数在 r=1/6 变号（r 大时衰减得比真解快）", chapter="4.1")
def test_dissipation_direction_flips_at_one_sixth() -> None:
    r""":math:`c_4^{\\text{num}}-c_4^{\\text{exact}} = r(\\frac1{12}-\\frac r2)`：二分实测变号点 = 1/6。

    :math:`r<1/6` 时数值衰减比真解**慢**（欠耗散），:math:`r>1/6` 时比真解**快**（过耗散）。
    这不是"格式精度"的问题——同一族格式的错误方向可以随步长反转，
    所以"加密网格一定更接近真解"这句话在固定 :math:`r` 的路径上并不自动成立。
    """
    found = find_threshold(
        lambda r: vn.dissipation_coefficient("ftcs", r) > 0.0,
        0.05,
        0.4,
        tol=1e-13,
        assume="true_at_lo",
        note="FTCS 耗散系数变号点",
    )
    assert_threshold(found, 1.0 / 6.0, rtol=1e-9, label="FTCS 耗散方向反转的 r", detail="解析：r/12−r²/2=0 ⟹ r=1/6")
    for scheme, sign in (("cn", 1.0), ("btcs", 1.0), ("ftcs", -1.0)):
        r = 0.4
        c = vn.dissipation_coefficient(scheme, r)
        rec = record(
            label=f"{scheme}·r={r} 的耗散误差系数（正=衰减比真解慢）",
            theory=f"sign = {sign:+.0f}",
            measured=f"{c:+.6f}",
            tol="符号一致",
            passed=(c > 0.0) == (sign > 0.0),
            detail="CN 是 r/12>0（永远欠耗散）；FTCS 在 r=0.4>1/6 时变号",
            kind="C",
        )
        assert rec.passed, (scheme, r, c)


@assertion("C", theory="变系数：判据要用 max λ_j（充分），用 mean λ_j 会算错 25%", chapter="4.1")
def test_variable_coefficient_threshold_uses_the_maximum_not_the_mean() -> None:
    r"""变系数 :math:`a(x)=1+x`（:math:`\\lambda_j=s\\,a_j/2`）：二分实测阈值 :math:`s_*=0.5423`。

    三种口径给出的答案差得很远：

    * **平均值口径** :math:`\\bar\\lambda=s\\bar a/2`：预测 :math:`s=0.4068`，偏小 25%；
    * **局部冻结系数判据** :math:`\\max_j\\lambda_j\\le\\frac12`：预测 :math:`s=0.5`，
      是**充分条件**但保守 8%（实测到 0.5423 才失稳）；
    * **实测**（幂迭代与稠密特征值两条独立路线）：:math:`s_*=0.5423415824`。

    "用最大系数"是对的方向，"用平均系数"是错的——这条断言把两个数字都摆在报告里。
    """
    nx = 41
    a = 1.0 + np.linspace(0.0, 1.0, nx)
    measured = find_threshold(
        lambda s: ht.explicit_growth(s * a / 2.0, nx=nx) > 1.0 + 1e-9,
        0.5,
        1.5,
        tol=1e-12,
        assume="false_at_lo",
        note="变系数 a(x)=1+x",
    )
    # 独立路线：稠密特征值。必须走 numcore.reference（P2 只允许那里用 numpy.linalg /
    # numpy.fft，测试文件也不例外——check_policy 会逐文件扫）
    def spectral_radius(s: float) -> float:
        lam = s * a / 2.0
        m = np.zeros((nx, nx))
        for j in range(nx):
            m[j, j] = 1.0 - 2.0 * lam[j]
            if j > 0:
                m[j, j - 1] = lam[j]
            if j < nx - 1:
                m[j, j + 1] = lam[j]
        m[0, 0], m[0, 1] = 1.0, 0.0
        m[-1, -1], m[-1, -2] = 1.0, 0.0
        return float(np.max(np.abs(ref_dense.eigvals(m))))

    ref = find_threshold(
        lambda s: spectral_radius(s) > 1.0 + 1e-12,
        0.5,
        1.5,
        tol=1e-12,
        assume="false_at_lo",
        note="稠密特征值口径",
    )
    mean_scale = float(np.mean(a) / np.max(a)) * measured.threshold
    rec1 = record(
        label="变系数实测阈值 s*：幂迭代 vs 稠密特征值",
        theory="两条路线必须一致",
        measured=f"{measured.threshold:.10f} / {ref.threshold:.10f}",
        tol="rel 1e-8",
        passed=abs(measured.threshold - ref.threshold) <= 1e-8 * ref.threshold,
        detail="幂迭代测的是同一个矩阵的谱半径，两条路线互不依赖",
        kind="C",
    )
    rec2 = record(
        label="变系数阈值：max 口径 / mean 口径 / 冻结系数判据",
        theory="max λ ≤ 1/2 是充分条件（保守）；mean λ 口径是错的",
        measured=f"{measured.threshold:.6f} / {mean_scale:.6f} / 0.5",
        tol="mean 口径偏差 ≥ 20%",
        passed=abs(mean_scale - measured.threshold) / measured.threshold >= 0.20,
        detail=f"max 口径比实测保守 {(measured.threshold - 0.5) / measured.threshold:.1%}；mean 口径偏小 {1 - mean_scale / measured.threshold:.1%}",
        kind="C",
    )
    assert rec1.passed and rec2.passed, (rec1.measured, rec2.measured)


# --------------------------------------------------------------------------- #
# B 类：二维与模态集合
# --------------------------------------------------------------------------- #
@assertion("B", theory="二维阈值是 1/4；ADI 的放大因子是两个 (1−z)/(1+z) 之积，模恒 ≤1", chapter="4.1")
def test_two_dimensional_amplification_and_adi_bound() -> None:
    r"""二维显式格式实测谱半径 = 有限网格解析值；ADI 在 :math:`r=100` 仍 ≤ 1。

    顺手记录一个反直觉的数字：ADI 的实测谱半径随 :math:`r` 从 0.985（:math:`r=0.3`）
    **下降**到 0.818（:math:`r=10`）再**回升**到 0.980（:math:`r=100`）——
    因为 :math:`|(1-z)/(1+z)|\\to1`：大 :math:`\\Delta t` 下 ADI 也快变成"不衰减但翻符号"，
    只是永远越不过 1。
    """
    nx = 21
    for r in (0.2, 0.3):
        measured = ht.heat2d_growth(nx, r=r)
        theory = _finite_grid_radius_2d(r, nx)
        rec = record(
            label=f"二维显式·r={r}：实测 / 解析有限网格谱半径",
            theory=f"{theory:.10f}",
            measured=f"{measured:.10f}",
            tol="rel 1e-6",
            passed=abs(measured - theory) <= 1e-6 * theory,
            detail=f"连续极限 |1−8r|={abs(1 - 8 * r):.6f}（有限网格修正后才是 {theory:.6f}）",
            kind="B",
        )
        assert rec.passed, (r, measured, theory)
    for r in (0.3, 1.0, 10.0):
        g = ht.heat2d_growth(nx, r=r, adi=True)
        theory = _finite_grid_radius_2d(r, nx, adi=True)
        rec = record(
            label=f"ADI·r={r}：实测 / 解析有限网格谱半径",
            theory=f"{theory:.10f}",
            measured=f"{g:.10f}",
            tol="rel 1e-6 且 ≤1",
            passed=abs(g - theory) <= 1e-6 * max(theory, 1e-12) and g <= 1.0 + 1e-12,
            detail=f"显式格式在同一 r 上的谱半径是 {ht.heat2d_growth(nx, r=r):.4g}",
            kind="B",
        )
        assert rec.passed, (r, g, theory)
    # 大 r 下幂迭代**低估**谱半径：相邻模态的特征值只差 1.6e-4，4000 步还分不开。
    # 低估不伤"≤1"这个结论，但把实测值当等号用就错了。
    r_big = 100.0
    g_big = ht.heat2d_growth(nx, r=r_big, adi=True)
    th_big = _finite_grid_radius_2d(r_big, nx, adi=True)
    rec = record(
        label=f"ADI·r={r_big}：实测（幂迭代）/ 解析有限网格谱半径",
        theory=f"{th_big:.10f}",
        measured=f"{g_big:.10f}（低 {abs(th_big - g_big) / th_big:.2e}）",
        tol="实测 ≤ 解析 + 1e-12，且 ≤ 1",
        passed=g_big <= th_big + 1e-12 and g_big <= 1.0 + 1e-12,
        detail="z=(r/2)a 达 198.8，前两个特征值之比 1−λ₂/λ₁=1.6e-4，4000 步幂迭代收敛不到 1e-6",
        kind="B",
    )
    assert rec.passed, (g_big, th_big)


@assertion("B", theory="可分辨模态集合：Dirichlet 不含 Nyquist，Neumann/周期（偶数网格）含", chapter="4.1")
def test_resolvable_wavenumber_sets() -> None:
    r"""模态集合的端点决定了有限网格阈值——这是"1/2 不精确"的几何原因。

    Dirichlet 网格上的模态是 :math:`k\\pi/(n-1)`（:math:`k=1..n-2`），
    最大只到 :math:`(n-2)\\pi/(n-1)<\\pi`；Neumann 与偶数周期网格都含 :math:`\\pi`。
    """
    for nx in (20, 21, 41):
        d = vn.mode_wavenumbers(nx, bc="dirichlet")
        nn = vn.mode_wavenumbers(nx, bc="neumann")
        pp = vn.mode_wavenumbers(nx, bc="periodic")
        rec = record(
            label=f"nx={nx} 的模态集合边界",
            theory="Dirichlet: (n−2)π/(n−1) < π；Neumann: π；周期: 2π·floor(n/2)/n",
            measured=f"{d[-1]:.10f} / {nn[-1]:.10f} / {pp[-1]:.10f}",
            tol="1e-12",
            passed=(
                abs(d[-1] - (nx - 2) * np.pi / (nx - 1)) <= 1e-12
                and abs(nn[-1] - np.pi) <= 1e-12
                and abs(pp[-1] - 2.0 * np.pi * (nx // 2) / nx) <= 1e-12
                and (abs(pp[-1] - np.pi) <= 1e-12) == (nx % 2 == 0)
            ),
            detail=f"Dirichlet 模态数 {d.size}，Neumann {nn.size}，周期 {pp.size}",
            kind="B",
        )
        assert rec.passed, (nx, rec.measured)
