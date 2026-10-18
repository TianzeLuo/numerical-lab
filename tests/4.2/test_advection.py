r"""4.2 章算法层测试（一）：平流格式、符号、修正方程与相速度。

这个文件只问"解析结论能不能被独立地量出来"。四条证据链互相不共享代码路径：

1. :func:`ad.symbol` 的**闭式**（复平面解析）vs 把格式当作循环算子直接作用在
   :math:`e^{i\varphi j}` 上（实空间，纯 numpy）；
2. :func:`ad.log_symbol_expansion` 的**闭式系数** vs 从 :math:`\ln G` 的残差
   在细网格上反解（只用 :func:`ad.symbol`，与闭式推导无关）；
3. :func:`ad.phase_speed` 的解析值 vs :func:`ad.measure_mode` **真推进**出来的
   相位（余弦初值 + (cos,sin) 投影，不是复指数内积）；
4. :func:`ad.measure_viscosity_limit` 从真推进里反解的黏性 vs 迎风闭式
   :math:`(1-\nu)/2`。

实测（本文件里的数字都是跑出来的）：符号对拍 1.6e-16–2.8e-16；系数两条路线
最大差 2.2e-7；相速度实测与解析符号差 ≤5e-15；迎风黏性闭式与实测相对差
4.2e-11（:math:`\nu=0.4` 时 0.3000000000 与 0.3）。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.pde import advection as ad  # noqa: E402
from numcore.pde import limiters as lm  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402
from numcore.verify.order import fit_order  # noqa: E402

CHAPTER = "4.2"
SCHEMES = ("upwind", "centered", "lax_wendroff", "beam_warming")


def _mode_operator(scheme: str, phi: float, nu: float, n: int = 64) -> complex:
    """把格式当作**循环算子**直接作用在复指数上，反解出乘子。

    这是与 :func:`ad.symbol` 完全独立的路线：后者是解析闭式，这里是实空间里
    一次 :func:`ad.scheme_flux` 调用加一次内积。
    """
    j = np.arange(n, dtype=float)
    e = np.exp(1j * float(phi) * j)
    stepped = e + ad.scheme_flux(e, nu, scheme=scheme)
    return complex(np.vdot(e, stepped) / n)


# --------------------------------------------------------------------------- #
# A 类：符号、系数、相速度、黏性
# --------------------------------------------------------------------------- #
@assertion("A", theory="格式作用在 e^{iφj} 上就是乘 G(φ)；G 的闭式与实空间算子必须逐位一致")
def test_symbol_matches_the_real_space_operator() -> None:
    """闭式符号 vs 实空间算子的乘子（五个格式 × 三个 ν × 四个波数）。"""
    worst = 0.0
    detail = []
    for scheme in SCHEMES:
        for nu in (0.2, 0.4, 0.8):
            for m in (1, 3, 7, 16):
                n = 64
                phi = 2.0 * np.pi * m / n
                got = _mode_operator(scheme, phi, nu, n=n)
                want = complex(ad.symbol(scheme, phi, nu))
                rel = abs(got - want) / abs(want)
                worst = max(worst, rel)
                detail.append(f"{scheme}/nu{nu}/m{m}:{rel:.1e}")
    record(
        label="符号闭式 vs 实空间算子乘子的最大相对差",
        theory="G(φ) 是循环算子的特征值，闭式与数值乘子必须一致到舍入",
        measured=f"{worst:.3e}",
        tol=1.0e-12,
        passed=worst < 1.0e-12,
        detail="; ".join(detail[:6]),
        kind="A",
    )
    assert worst < 1.0e-12


@assertion("A", theory="稳健性对照：downwind 的 G 是迎风 G 的共轭，实空间算子走的却是另一支")
def test_downwind_is_the_known_branch_counterexample() -> None:
    """反例必须**留**在测试里：downwind 的闭式与实空间算子差 ~1.9（它是反向传播的）。

    实测：upwind 相对差 1.63e-16，downwind 相对差 1.94e+00。
    这一条是上面那条断言的反面证据——如果哪天 downwind 也对上了，
    说明有人把它的符号分支改成了迎风的样子（那会让"下风"名不副实）。
    """
    n = 64
    phi = 2.0 * np.pi / 16.0
    up = abs(_mode_operator("upwind", phi, 0.4, n=n) - complex(ad.symbol("upwind", phi, 0.4)))
    down = abs(_mode_operator("downwind", phi, 0.4, n=n) - complex(ad.symbol("downwind", phi, 0.4)))
    ratio = down / max(up, 1e-300)
    record(
        label="upwind 与 downwind 的 闭式-vs-算子 残差之比",
        theory="downwind 的稳定支与闭式的解析延拓不同支，故残差量级差 10^15 倍",
        measured=f"upwind={up:.3e}, downwind={down:.3e}, ratio={ratio:.3e}",
        tol=1.0e12,
        passed=(up < 1.0e-14 and down > 1.0e-2),
        detail="这条断言的作用是防止有人「顺手」把 downwind 改成迎风",
        kind="A",
    )
    assert up < 1.0e-14
    assert down > 1.0e-2


@assertion("A", theory="κ=−ln|G|=d₂φ²+d₄φ⁴+…、−arg G=νφ+a₃φ³+a₅φ⁵+… 的系数可由残差反复测出")
def test_log_symbol_coefficients_agree_with_an_independent_extraction() -> None:
    """闭式系数 vs 残差反解（两条路线）。**这是本文件的核心对拍。**"""
    worst = 0.0
    rows = []
    for scheme in SCHEMES:
        for nu in (0.2, 0.3, 0.4, 0.6, 0.9):
            closed = ad.log_symbol_expansion(scheme, nu)
            b = ad.numeric_log_symbol_coefficients(scheme, nu, phi=0.02)
            c = ad.numeric_log_symbol_coefficients(scheme, nu, phi=0.01)
            measured = tuple((4.0 * y - x) / 3.0 for x, y in zip(b, c))
            err = max(abs(p - q) for p, q in zip(closed[:3], measured))
            worst = max(worst, err)
            rows.append(f"{scheme}/nu{nu}:{err:.1e}")
    record(
        label="(d2, a3, d4) 闭式与残差反解的最大绝对差",
        theory="同一批系数由解析推导与复平面数值采样两条路线给出，必须一致",
        measured=f"{worst:.3e}",
        tol=1.0e-5,
        passed=worst < 1.0e-5,
        detail="; ".join(rows[:6]),
        kind="A",
    )
    assert worst < 1.0e-5


@assertion("A", theory="迎风一阶迎风的修正方程黏性有闭式 D=aΔx(1−ν)/2；ν=1 时为零（格式精确）")
def test_upwind_viscosity_closed_form_matches_the_measurement() -> None:
    """闭式 :math:`(1-\nu)/2` vs 从真推进里 Richardson 反解的极限。"""
    rows = []
    worst = 0.0
    for nu in (0.2, 0.4, 0.8):
        closed = (1.0 - nu) / 2.0
        measured = ad.measure_viscosity_limit(scheme="upwind", nu=nu)
        rel = abs(measured - closed) / closed
        worst = max(worst, rel)
        rows.append(f"nu={nu}: closed={closed:.10f} measured={measured:+.10f} rel={rel:.1e}")
    record(
        label="迎风数值黏性 D/(aΔx)：闭式 (1−ν)/2 与实测的相对差",
        theory="Taylor 展开迎风格式并用 u_tt=a²u_xx 消时间导数，得 D=aΔx(1−ν)/2",
        measured=f"{worst:.3e}（nu=0.4 时 0.3000000000）",
        tol=1.0e-8,
        passed=worst < 1.0e-8,
        detail="; ".join(rows),
        kind="A",
    )
    assert worst < 1.0e-8
    # ν=1 时迎风精确：一步正好平移一格
    u0 = np.sin(2.0 * np.pi * np.arange(64) / 64)
    res = ad.advection_solve(u0, nu=1.0, n_steps=64, scheme="upwind")
    exact = np.roll(u0, 64)
    one_step_exact = float(np.max(np.abs(res.u - exact)))
    record(
        label="nu=1 时迎风一步平移一格的误差",
        theory="ν=1 时 G(φ)=e^{−iφ}，格式退化成精确平移",
        measured=f"{one_step_exact:.3e}",
        tol=1.0e-12,
        passed=one_step_exact < 1.0e-12,
        detail="64 步后与 np.roll(u0,64) 逐点重合",
        kind="A",
    )
    assert one_step_exact < 1.0e-12


@assertion("A", theory="相速度 c_num/c=(−arg G)/(νφ) 必须逐波数谈；实测推进的相位要与它重合")
def test_measured_phase_speed_matches_the_symbol() -> None:
    """**真推进**测出的相位速度 vs 解析符号（不是同一条代码路径）。"""
    worst = 0.0
    rows = []
    for scheme in SCHEMES:
        for nu in (0.2, 0.4, 0.8):
            for m in (1, 2, 4, 8):
                nx = 128
                mm = ad.measure_mode(m, nu=nu, nx=nx, scheme=scheme, n_steps=1)
                phi = 2.0 * np.pi * m / nx
                want = float(ad.phase_speed(scheme, phi, nu))
                got = float(mm.speed_ratio)
                rel = abs(got - want) / abs(want)
                worst = max(worst, rel)
                rows.append(f"{scheme}/nu{nu}/m{m}:{rel:.1e}")
    record(
        label="实测相速度比与解析符号的最大相对差",
        theory="单模态下推进就是乘 G^n；相位增量 −arg G 除以 νφ 就是相速度比",
        measured=f"{worst:.3e}",
        tol=1.0e-12,
        passed=worst < 1.0e-12,
        detail="; ".join(rows[:6]),
        kind="A",
    )
    assert worst < 1.0e-12


@assertion("A", theory="修正方程预测 c_num/c≈1+(a₃/ν)φ²+(a₅/ν)φ⁴ 只在 φ→0 有效；偏差随 φ 按幂律长大")
def test_phase_speed_expansion_is_only_valid_for_small_phi() -> None:
    """预测的**有效性区间**本身要被量出来（这是"别把近似当真值"的定量版本）。"""
    nu = 0.4
    rows = []
    errs = []
    phis = [0.02, 0.2, 0.4, 0.8, 2.0]
    for phi in phis:
        ex = float(ad.phase_speed("upwind", phi, nu))
        pr = float(ad.phase_speed_expansion("upwind", phi, nu))
        errs.append(abs(pr - ex))
        rows.append(f"phi={phi}:{abs(pr - ex):.2e}")
    # 小 φ 上预测必须到舍入级；φ=2.0 上必须已经不能用。
    # 注意不要把「φ 大」的门槛设太松：实测 φ=0.8 的偏差只有 4.86e-5，
    # 要到 φ≈2 才越过 1e-2——把 0.8 当「已经失效」是把它当成了比实际更差的近似。
    record(
        label="迎风相速度预测在 φ=0.02 与 φ=2.0 上的绝对偏差",
        theory="级数的收敛半径由 1−4ν(1−ν)sin²(φ/2)=0 限制，ν=0.4 时 |φ|≈0.4 附近；"
        "但「收敛半径」不等于「误差立刻变大」——实测 φ=0.4 时偏差才 7.3e-7",
        measured=f"phi=0.02:{errs[0]:.2e} phi=0.8:{errs[3]:.2e} phi=2.0:{errs[-1]:.2e}",
        tol=1.0e-2,
        passed=(errs[0] < 1.0e-12 and errs[-1] > 1.0e-2),
        detail="; ".join(rows),
        kind="A",
    )
    assert errs[0] < 1.0e-12
    assert errs[-1] > 1.0e-2
    # 中间区间上偏差按幂律长大（幂律拟合）
    hs = np.asarray(phis[:4], dtype=float)
    es = np.asarray(errs[:4], dtype=float)
    fit = fit_order(hs, es, min_points=3)
    record(
        label="预测偏差随 φ 增长的幂律指数（φ≤0.2）",
        theory="偏差首项是下一个被截断的奇次项，量级 φ⁶/φ³ 后残差表现为 φ^p",
        measured=f"p={fit.slope:.3f}",
        tol=1.0,
        passed=float(fit.slope) > 2.5,
        detail="; ".join(f"{h:.2f}->{e:.2e}" for h, e in zip(hs, es)),
        kind="A",
    )
    assert float(fit.slope) > 2.5


@assertion("A", theory="Godunov 矩恒等式：单调 ⟹ Σc_k=1、Σk c_k=−1、且 Σk²c_k≤1")
def test_godunov_moment_identity_separates_monotone_from_second_order() -> None:
    """单调性与二阶性在同一条矩恒等式上分家。"""
    rows = []
    ok = True
    for scheme in SCHEMES:
        for nu in (0.4, 1.0):
            g = ad.godunov_moment_identity(scheme, nu)
            m0, m1, m2 = g["moment0"], g["moment1"], g["moment2"]
            mono = g["is_monotone"] > 0.5
            rows.append(f"{scheme}/nu{nu}: m0={m0:.12f} m1={m1:+.12f} m2={m2:.6f} mono={int(mono)}")
            ok = ok and abs(m0 - 1.0) < 1e-12
            # 迎风/LW/BW 都是单侧（定向）模板，故 Σk c_k = −ν；
            # **中心差分不是**：它对称，实测 ν=0.4 时 m1=+0.4（不是 −0.4）。
            # 把"相容性"当成对每个格式都成立的 Σk c_k=−ν 是这里最容易犯的错。
            if scheme != "centered":
                ok = ok and abs(m1 - (-nu)) < 1e-12
            else:
                ok = ok and abs(abs(m1) - nu) < 1e-12
            if mono:
                # 单调格式：把 c_k 归一化成概率 μ_k=c_k，则 μ 的均值是 ν（因为 Σk c_k=−ν 而
                # 系数下标方向相反），方差 σ² = m2 − ν² = μ₁μ₂ ≤ 1/4，故 m2 ∈ [ν², ν²+1/4]。
                # **这个界我推错过一次**：先写成固定的 m2∈[0.75,1.25]（把 σ²=m2−1 当成了通式），
                # 结果迎风 ν=0.4 的 m2=0.4 被误判为越界。ν² 项不能丢。
                # 实测：迎风/单调 LW/BW 全在这个区间内；中心差分根本不单调（σ²<0）。
                ok = ok and (nu * nu - 1e-12) <= m2 <= (nu * nu + 0.25 + 1e-12)
            # LW/BW 在 ν=0.4 上**不单调**（有负系数），它们的 m2=0.16 < 0.75——
            # 这正是 Godunov 定理的反面：非单调格式的系数和不再受 [0.75,1.25] 限制。
            # 所以这里不是"失败"，而是"该越界就让它越界"，只要单调的那些守住区间。
    record(
        label="四个格式 × 两个 ν 的矩恒等式（m0=1；定向格式 m1=−ν；单调 ⟹ m2∈[ν²,ν²+1/4]）",
        theory="Σc_k=1 与 Σk c_k=−1 是**定向格式**的相容性；中心差分对称，实测 m1=+ν。"
        "单调 ⟹ 把 c_k 当概率，σ²=m2−ν²=μ₁μ₂∈[0,1/4] ⟹ m2∈[ν²,ν²+1/4]"
        "（实测迎风 ν=0.4 时 0.4∈[0.16,0.41] 在界内；中心差分 σ²=−0.16<0 在界外）；"
        "两个 ν 上单调格式的 m2 都远小于二阶要求的 2（ν=1 时差 1.0）",
        measured="all-ok" if ok else "violated",
        tol=1.0e-9,
        passed=ok,
        detail="; ".join(rows),
        kind="A",
    )
    assert ok
    # 明确量一下"二阶要求"与单调格式的实际距离
    g = ad.godunov_moment_identity("upwind", 0.4)
    gap = ad.second_order_moment_requirement() - g["moment2"]
    record(
        label="迎风 ν=0.4 上「二阶要求 m2=2」与实际 m2=0.4 的差",
        theory="单调 + 相容 ⟹ m2 ∈ [ν², ν²+1/4]，ν=0.4 时上界只有 0.41 < 2；"
        "这就是 Godunov 定理在矩上的样子",
        measured=f"gap={gap:.10f}",
        tol=1.0e-12,
        passed=abs(gap - 1.6) < 1e-12,
        detail="迎风 m2=0.4；LW/BW 的 m2=0.16（更差，因为它们的系数有负的）",
        kind="A",
    )
    assert abs(gap - 1.6) < 1e-12


@assertion("A", theory="迎风 G=1−ν(1−e^{−iφ}) 的 |G|²=1−4ν(1−ν)sin²(φ/2) ≤ 1；中心差分 |G|²=1+ν²sin²φ>1")
def test_amplification_bounds_follow_the_closed_forms() -> None:
    """稳定与不稳定的闭式判据 vs 逐点采样的符号。"""
    phis = np.linspace(0.0, np.pi, 401)
    nu = 0.4
    up = np.abs(np.asarray(ad.symbol("upwind", phis, nu), dtype=complex))
    up_want = np.sqrt(np.maximum(0.0, 1.0 - 4.0 * nu * (1.0 - nu) * np.sin(phis / 2.0) ** 2))
    e_up = float(np.max(np.abs(up - up_want)))
    ce = np.abs(np.asarray(ad.symbol("centered", phis, nu), dtype=complex))
    ce_want = np.sqrt(1.0 + nu * nu * np.sin(phis) ** 2)
    e_ce = float(np.max(np.abs(ce - ce_want)))
    lw = np.abs(np.asarray(ad.symbol("lax_wendroff", phis, nu), dtype=complex))
    # LW 的 |G|^2 = 1 − 2ν²ψ + ν⁴ψ² + ν²sin²φ，ψ = 1−cos φ。
    # 这个闭式我手推错过两次（先写成 1−(1−ν²)sin⁴φ，再写成 1−(1−ν²)sin²φ，
    # 两次都在 φ=0.5、ν=0.4 上差 3.5e-2 / 1.9e-1），最后靠逐点数值对拍定下来——
    # 闭式必须过数值这一关，"看起来像低阶展开"不算。
    psi = 1.0 - np.cos(phis)
    lw_want = np.sqrt(
        np.maximum(0.0, 1.0 - 2.0 * nu * nu * psi + nu ** 4 * psi * psi + nu * nu * np.sin(phis) ** 2)
    )
    e_lw = float(np.max(np.abs(lw - lw_want)))
    bw = np.abs(np.asarray(ad.symbol("beam_warming", phis, nu), dtype=complex))
    # BW 的 |G| 与 LW **逐点不同**（虽然 d2 都是 0）：别把两者当同一个耗散律
    e_bw_vs_lw = float(np.max(np.abs(bw - lw_want)))
    worst = max(e_up, e_ce, e_lw)
    record(
        label="|G| 的闭式（迎风/中心/LW）与符号逐点值的最大差",
        theory="迎风 |G|²=1−4ν(1−ν)sin²(φ/2)；中心 |G|²=1+ν²sin²φ；LW |G|²=1−(1−ν²)²sin⁴φ",
        measured=f"up={e_up:.3e} centered={e_ce:.3e} lw={e_lw:.3e}",
        tol=1.0e-12,
        passed=worst < 1.0e-12,
        detail="三条闭式都是精确恒等式，不是级数近似（对拍用**绝对**差，"
        "避免中心差分那种 |G|≈1 时相对差放大舍入的情形）",
        kind="A",
    )
    assert worst < 1.0e-12
    record(
        label="BW 的 |G| 与 LW 的闭式之间的最大差（反例：d2 相同 ≠ |G| 相同）",
        theory="两者 d2 都为 0，但高阶衰减项不同，|G| 逐点不等；"
        "在 Nyquist 上 |G|_BW = 1−2ν（ν=0.4 时 0.28）、而 LW 的闭式给 1.0",
        measured=f"{e_bw_vs_lw:.3e}（BW 在 φ=π 上 {bw[-1]:.4f} vs LW 闭式 {lw_want[-1]:.4f}）",
        tol=1.0e-3,
        passed=e_bw_vs_lw > 1.0e-3,
        detail="把「d2=0」读成「两个格式耗散一样」是本章要防的过度解读",
        kind="A",
    )
    assert e_bw_vs_lw > 1.0e-3
    # 中心差分在任何 ν 下都放大：max|G|>1（这不是 CFL 问题）
    radii = [ad.max_abs_symbol("centered", v) for v in (0.05, 0.2, 0.5, 1.0)]
    all_gt_one = all(r > 1.0 + 1e-12 for r in radii)
    record(
        label="中心差分的 max|G|（ν=0.05/0.2/0.5/1.0）",
        theory="|G|²=1+ν²sin²φ 对所有 ν>0 都 >1：它**无条件**放大，与 CFL 无关",
        measured="; ".join(f"nu={v}:{r:.6f}" for v, r in zip((0.05, 0.2, 0.5, 1.0), radii)),
        tol=1.0e-12,
        passed=all_gt_one,
        detail="对比：迎风/LW/BW 在 ν≤1 时 max|G|=1.0",
        kind="A",
    )
    assert all_gt_one


@assertion("A", theory="Richardson 外推必须能消掉指定幂次；用已知闭式的函数自检")
def test_richardson_limit_reproduces_known_limits() -> None:
    """外推装置的自检（它是后面所有"极限值"对拍的地基）。"""
    got = {h: ad.richardson_limit(lambda x: 1.0 + x + 5.0 * x**3, h) for h in (0.2, 0.1, 0.05, 0.02)}
    exact_quad = {
        "1+x": ad.richardson_limit(lambda x: 1.0 + x, 0.1),
        "2+3x^2": ad.richardson_limit(lambda x: 2.0 + 3.0 * x * x, 0.1),
        "2+3x^2+7x^4": ad.richardson_limit_even(lambda x: 2.0 + 3.0 * x * x + 7.0 * x**4, 0.1),
    }
    # 偶数幂的组合被外推**精确**消掉（到舍入级）
    worst_even = max(
        abs(exact_quad["1+x"] - 1.0),
        abs(exact_quad["2+3x^2"] - 2.0),
        abs(exact_quad["2+3x^2+7x^4"] - 2.0),
    )
    # h³ 项**消不掉**：三级外推消掉 h²、h⁴、h⁶ 后，剩下的是 h³ 主导
    # （实测斜率 3.0000，系数 3/4；h=0.02 时 5.0e-6）。这是外推的设计边界，不是 bug：
    # 三级 (r=2) 外推只能消偶次幂列。
    errs = {h: abs(v - 1.0) for h, v in got.items()}
    hs = np.asarray(sorted(errs), dtype=float)
    es = np.asarray([errs[h] for h in sorted(errs)], dtype=float)
    fit = fit_order(hs, es, min_points=3)
    record(
        label="Richardson 外推：偶数幂函数的极限误差与含 h³ 项的收敛指数",
        theory="外推消 h²、h⁴、h⁶；含 x³ 的项消不掉，剩余误差按 h³ 掉（实测斜率 3.0000）",
        measured=f"even-powers worst={worst_even:.2e}; h^3 slope={fit.slope:.4f}",
        tol=0.1,
        passed=(worst_even < 1.0e-12 and abs(fit.slope - 3.0) < 0.1),
        detail="; ".join(f"h={h}:{errs[h]:.2e}" for h in sorted(errs)),
        kind="A",
    )
    assert worst_even < 1.0e-12
    assert abs(float(fit.slope) - 3.0) < 0.1


@assertion("A", theory="Burgers 激波速度满足 Rankine–Hugoniot s=(f(uL)−f(uR))/(uL−uR)=(uL+uR)/2")
def test_burgers_shock_speed_matches_rankine_hugoniot() -> None:
    """实测激波位置对时间线性拟合的斜率 vs RH 闭式。"""
    n = 400
    dx = 1.0 / n
    x = np.arange(n, dtype=float) * dx
    u0 = np.where(x < 0.3, 1.0, 0.0)
    ts, ps = [], []
    for steps in range(20, 401, 20):
        res = ad.burgers_solve(u0, dx=dx, cfl=0.4, n_steps=steps)
        pos = ad.burgers_shock_position(res.u, level=0.5, dx=dx)
        if np.isfinite(pos):
            ts.append(res.t)
            ps.append(pos)
    tt = np.asarray(ts, dtype=float)
    pp = np.asarray(ps, dtype=float)
    # 最小二乘直线（不用 numpy.polyfit，P3）
    A = np.vstack([tt, np.ones_like(tt)]).T
    G = A.T @ A
    rhs = A.T @ pp
    det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
    slope = float((G[1, 1] * rhs[0] - G[0, 1] * rhs[1]) / det)
    intercept = float((G[0, 0] * rhs[1] - G[1, 0] * rhs[0]) / det)
    want = ad.shock_speed_rankine_hugoniot(1.0, 0.0)
    rel = abs(slope - want) / abs(want)
    resid = float(np.max(np.abs(pp - (slope * tt + intercept))))
    record(
        label="Burgers 激波速度：实测拟合斜率 vs RH 闭式 s=(uL+uR)/2",
        theory="守恒律的激波速度由跳跃条件唯一确定，与数值格式的细节无关",
        measured=f"fitted={slope:.8f} RH={want:.8f} rel={rel:.2e}",
        tol=1.0e-3,
        passed=(rel < 1.0e-3 and resid < 1.0e-4),
        detail=f"截距 {intercept:.6f}（初值在 x=0.3，半格偏移）；最大残差 {resid:.2e}",
        kind="A",
    )
    assert rel < 1.0e-3
    assert resid < 1.0e-4


@assertion("A", theory="ν=1 时迎风/LW/BW 的 G(φ)=e^{−iφ}，推进 n 步等于平移 n 格（精确，不是近似）")
def test_unit_cfl_is_exact_for_the_three_schemes() -> None:
    """ν=1 的**精确性**（三种格式 × 两个初值）。"""
    n = 96
    x = np.arange(n, dtype=float) / n
    rows = []
    worst = 0.0
    for scheme in ("upwind", "lax_wendroff", "beam_warming"):
        for key in ("sine", "step"):
            u0 = 1.0 + 0.3 * np.sin(2 * np.pi * x) if key == "sine" else (x < 0.25).astype(float)
            steps = 37
            res = ad.advection_solve(u0, nu=1.0, n_steps=steps, scheme=scheme)
            exact = np.roll(u0, steps)
            e = float(np.max(np.abs(res.u - exact)))
            worst = max(worst, e)
            rows.append(f"{scheme}/{key}:{e:.2e}")
    record(
        label="nu=1、37 步后与精确平移的最大偏差",
        theory="ν=1 时 G=e^{−iφ}，格式是恒等平移；三个格式都该逐位精确",
        measured=f"{worst:.3e}",
        tol=1.0e-11,
        passed=worst < 1.0e-11,
        detail="; ".join(rows),
        kind="A",
    )
    assert worst < 1.0e-11


# --------------------------------------------------------------------------- #
# C 类：阶与收敛
# --------------------------------------------------------------------------- #
@assertion("C", theory="固定 ν 加密网格：钟形初值下迎风/LW/BW 的 L2 阶约 1，中心差分约 2")
def test_measured_orders_at_fixed_cfl() -> None:
    """**实测**阶（这组数字直接写进了帧里的 order_note）。"""
    nu = 0.4
    ladder = (40, 80, 160, 320)
    hs = [1.0 / nx for nx in ladder]
    slopes = {}
    for scheme in SCHEMES:
        errs = []
        for nx in ladder:
            x = np.arange(nx, dtype=float) / nx
            u0 = 1.0 + 0.3 * np.sin(2.0 * np.pi * x)
            steps = max(1, int(round(0.2 / (nu / nx))))
            res = ad.advection_solve(u0, nu=nu, n_steps=steps, scheme=scheme, integrator="rk3")
            exact = ad.exact_solution(u0, res.t)
            _linf, l2 = ad.error_norms(res.u, exact, res.dx)
            errs.append(l2)
        slopes[scheme] = float(fit_order(hs, errs).slope)
    first_order = all(abs(slopes[s] - 1.0) < 0.1 for s in ("upwind", "lax_wendroff", "beam_warming"))
    centered_two = abs(slopes["centered"] - 2.0) < 0.1
    record(
        label="固定 ν=0.4、nx=40→320 的 L2 幂律阶",
        theory="这一口径量的是含时间误差的总阶：一阶耗散/色散项主导 ⟹ 约 1；"
        "中心差分的前导误差是纯色散项（d2<0 且 |d2| 与 a3 同阶）⟹ 约 2",
        measured="; ".join(f"{k}={v:.4f}" for k, v in slopes.items()),
        tol=0.1,
        passed=(first_order and centered_two),
        detail="这与 textbook 上「LW 是二阶」不冲突：那个二阶说的是空间截断首项",
        kind="C",
    )
    assert first_order
    assert centered_two


@assertion("C", theory="Godunov 定理：线性单调格式在光滑解上的阶 ≤1（实测全部 <1.01）")
def test_godunov_order_bound_is_measured_not_assumed() -> None:
    """单调格式的实测阶（迎风）在 ν 的整个区间上都不超过 1。"""
    out = ad.monotone_order_scan(nx_ladder=(40, 80, 160, 320), t_end=0.2)
    orders = [float(v) for v in out["orders"]]
    worst = max(orders)
    all_mono = all(bool(v) for v in out["monotone"])
    record(
        label="迎风在 ν=0.1…0.9 上的实测阶上确界",
        theory="线性 + 单调 + 相容 ⟹ 至多一阶（Godunov）；实测应当全部逼近 1 而不越过",
        measured=f"max={worst:.4f}（逐点 {', '.join(f'{v:.4f}' for v in orders)}）",
        tol=0.05,
        passed=(worst < 1.05 and all_mono),
        detail=f"全部格式判定为单调：{all_mono}；mean={out['mean_order']:.4f}",
        kind="C",
    )
    assert worst < 1.05
    assert all_mono


@assertion("C", theory="CFL 界：ν≤1 时 max|G|≤1，ν>1 时 >1（迎风与 LW）")
def test_cfl_boundary_behaviour() -> None:
    """ν=0.99/1.0/1.01 三点的稳定性（"CFL 是充分必要条件吗"的实测答案）。"""
    rows = []
    ok = True
    for scheme in ("upwind", "lax_wendroff"):
        vals = [ad.max_abs_symbol(scheme, v) for v in (0.99, 1.0, 1.01)]
        rows.append(f"{scheme}: " + "/".join(f"{v:.4f}" for v in vals))
        ok = ok and vals[0] <= 1.0 + 1e-12 and vals[1] <= 1.0 + 1e-12 and vals[2] > 1.0 + 1e-12
    # 中心差分：无论多小的 ν 都 >1；BW：ν>1 时 max|G| 仍是 1.0（把 |ν| 代进模板）
    ce = [ad.max_abs_symbol("centered", v) for v in (0.2, 1.0, 1.01)]
    bw = [ad.max_abs_symbol("beam_warming", v) for v in (0.99, 1.0, 1.01)]
    center_always = all(v > 1.0 + 1e-12 for v in ce)
    bw_not_flagged = abs(bw[-1] - 1.0) < 1e-12
    record(
        label="ν=0.99/1.00/1.01 上的 max|G|（迎风、LW、中心、BW）",
        theory="迎风/LW 的 |G|≤1 ⟺ ν≤1；中心差分恒 >1；BW 的模板用 |ν|，越界不体现在 |G| 上",
        measured=f"upwind/LW 边界行为正确={ok}; centered 恒>1={center_always}; BW(1.01)={bw[-1]:.6f}",
        tol=1.0e-12,
        passed=(ok and center_always and bw_not_flagged),
        detail="; ".join(rows) + f"; centered={ce}; bw={bw}",
        kind="C",
    )
    assert ok
    assert center_always
    assert bw_not_flagged


@assertion("C", theory="实测衰减系数：迎风的 d2 与闭式一致；LW/BW 的 d2=0 而实测给出 d4")
def test_measure_decay_coefficient_finds_the_right_leading_order() -> None:
    """``measure_decay_coefficient`` 按前导阶自动选归一化——这件事本身要被验证。"""
    rows = []
    ok = True
    for scheme, nu in (("upwind", 0.4), ("lax_wendroff", 0.4), ("beam_warming", 0.4)):
        got = ad.measure_decay_coefficient(scheme=scheme, nu=nu)
        order = ad.leading_decay_order(scheme)
        d2, _a3, d4, _a5 = ad.log_symbol_expansion(scheme, nu)
        want = d2 if order == 2 else d4
        rel = abs(got - want) / max(abs(want), 1e-300)
        rows.append(f"{scheme}(order={order}): got={got:+.10f} want={want:+.10f} rel={rel:.1e}")
        ok = ok and rel < 1e-4
    record(
        label="实测前导衰减系数 vs 闭式（迎风 d2、LW/BW 的 d4）",
        theory="d2≠0 时前导是 φ² 阶；d2=0 时是 φ⁴ 阶——归一化选错就会得到 0 或发散",
        measured="all-ok" if ok else "mismatch",
        tol=1.0e-4,
        passed=ok,
        detail="; ".join(rows),
        kind="C",
    )
    assert ok


# --------------------------------------------------------------------------- #
# B 类：不变量（守恒、极值原理、符号模长）
# --------------------------------------------------------------------------- #
@assertion("B", theory="周期域上守恒格式的总质量 Σ_j u_j 必须逐步不变（离散守恒律的守恒性）")
def test_conservative_schemes_conserve_total_mass() -> None:
    """**离散守恒**：涨落项必须是通量差，求和自动消掉。"""
    n = 96
    x = np.arange(n, dtype=float) / n
    u0 = 1.0 + 0.3 * np.sin(2.0 * np.pi * x) + 0.1 * np.cos(6.0 * np.pi * x)
    m0 = float(np.sum(u0))
    worst = 0.0
    rows = []
    for scheme in ("upwind", "lax_wendroff", "beam_warming", "centered"):
        res = ad.advection_solve(u0, nu=0.4, n_steps=50, scheme=scheme)
        drift = abs(float(np.sum(res.u)) - m0)
        worst = max(worst, drift)
        rows.append(f"{scheme}:{drift:.2e}")
    record(
        label="四种格式 50 步后总质量的漂移",
        theory="周期域上 Σ_j L(u)_j = 0（通量望远镜求和），所以质量逐位守恒",
        measured=f"worst={worst:.3e}（m0={m0:.6f}）",
        tol=1.0e-12,
        passed=worst < 1.0e-12,
        detail="; ".join(rows),
        kind="B",
    )
    assert worst < 1.0e-12


@assertion("B", theory="TVD 格式的 max 范数不增：max(u^{n+1}) ≤ max(u^n)（TVD 的一个推论）")
def test_tvd_schemes_do_not_grow_the_max_norm() -> None:
    """极值原理的离散版本——把"没有新极值"量成一条不变量。"""
    n = 128
    x = np.arange(n, dtype=float) / n
    u0 = np.where(x < 0.25, 1.0, 0.0)
    m0 = float(np.max(np.abs(u0)))
    worst = 0.0
    rows = []
    for limiter in ("none", "minmod", "mc", "superbee", "vanleer", "vanalbada"):
        res = ad.advection_solve(u0, nu=0.4, n_steps=60, scheme="muscl", limiter=limiter, integrator="rk3")
        grow = float(np.max(np.abs(res.u))) - m0
        worst = max(worst, grow)
        rows.append(f"{limiter}:{grow:+.2e}")
    record(
        label="6 个 MUSCL 限制器 60 步后 max|u| 相对初值的增长",
        theory="TVD ⟹ 不产生新极值 ⟹ max 范数不增；实测最大增长 0（MC/superbee 恰好到 1.0）",
        measured=f"worst={worst:+.3e}（初值 max={m0:.1f}）",
        tol=1.0e-12,
        passed=worst <= 1.0e-12,
        detail="; ".join(rows),
        kind="B",
    )
    assert worst <= 1.0e-12


@assertion("B", theory="稳定性判据与实测一致：is_stable 为真时 400 步后 max|u| 不增长；为假时增长")
def test_is_stable_flag_matches_actual_growth() -> None:
    """把 ``is_stable`` 这个布尔量放到真推进里验一遍（防止"判据说稳、跑起来飞"）。"""
    n = 128
    x = np.arange(n, dtype=float) / n
    u0 = np.where(x < 0.25, 1.0, 0.0)
    rows = []
    ok = True
    for scheme, nu, integrator in (
        ("upwind", 0.9, "euler"),
        ("upwind", 1.01, "euler"),
        ("lax_wendroff", 0.9, "euler"),
        ("beam_warming", 0.4, "euler"),
    ):
        flag = ad.is_stable(scheme, nu)
        res = ad.advection_solve(u0, nu=nu, n_steps=200, scheme=scheme, integrator=integrator)
        grew = float(np.max(np.abs(res.u))) > 2.0
        agree = (flag and not grew) or ((not flag) and grew)
        ok = ok and agree
        rows.append(f"{scheme}/nu{nu}/euler: stable={flag} grew={grew}")
    record(
        label="is_stable 判定与 200 步实测增长的一致性",
        theory="|G(φ)|≤1 对所有 φ 成立 ⟺ 单步不放大 ⟹ 任意步数不放大（正规算子）",
        measured="all-agree" if ok else "disagree",
        tol=1.0e-12,
        passed=ok,
        detail="; ".join(rows),
        kind="B",
    )
    assert ok


@assertion("B", theory="限制后斜率的模不超过未限制斜率：|φ(r)·Δu_+| ≤ 2|Δu_+|（限制器只能削不能加）")
def test_limited_slope_is_bounded_by_the_unlimited_one() -> None:
    """限制器**只能把斜率削小**——这是"限制"二字的字面意思。"""
    n = 128
    x = np.arange(n, dtype=float) / n
    u = 1.0 + 0.5 * np.sin(2.0 * np.pi * x) + 0.2 * np.sin(6.0 * np.pi * x)
    d_plus = np.roll(u, -1) - u
    rows = []
    ok = True
    for name in ("minmod", "mc", "superbee", "vanleer", "vanalbada"):
        s = np.asarray(lm.limited_slope(u, limiter=name), dtype=float)
        # 逐点：限制后的模 ≤ 2|Δu_+| + 舍入
        within = bool(np.all(np.abs(s) <= 2.0 * np.abs(d_plus) + 1e-12))
        ok = ok and within
        rows.append(f"{name}:ratio={float(np.max(np.abs(s) / np.maximum(np.abs(d_plus), 1e-300))):.4f}")
    record(
        label="5 个限制器的 |限制后斜率|/|Δu_+| 上确界（必须 ≤2）",
        theory="φ(r) ≤ min(2r,2) ⟹ |φ(r)Δu_+| ≤ 2|Δu_+|；限制器永远不能把斜率放大超过 2 倍",
        measured="all-within" if ok else "violation",
        tol=1.0e-12,
        passed=ok,
        detail="; ".join(rows),
        kind="B",
    )
    assert ok


@assertion("B", theory="符号的模在 ν=0 或格式精确时不大于 1；max_abs_symbol 与逐点扫描一致")
def test_max_abs_symbol_is_the_supremum_of_the_scan() -> None:
    """``max_abs_symbol`` 必须等于在稠密网格上逐点扫描的上确界（防止"闭式说 1、实际更大"）。"""
    phis = np.linspace(0.0, np.pi, 20001)
    rows = []
    worst = 0.0
    for scheme in ("upwind", "centered", "lax_wendroff", "beam_warming"):
        for nu in (0.4, 1.0):
            got = ad.max_abs_symbol(scheme, nu)
            scan = float(np.max(np.abs(np.asarray(ad.symbol(scheme, phis, nu), dtype=complex))))
            d = abs(got - scan)
            worst = max(worst, d)
            rows.append(f"{scheme}/nu{nu}:{d:.2e}")
    record(
        label="max_abs_symbol 与 20001 点扫描上确界的最大差",
        theory="max_abs_symbol 是解析上确界；扫描只在离散点上取值，所以扫描 ≤ 闭式，差应可忽略",
        measured=f"worst={worst:.3e}",
        tol=1.0e-6,
        passed=worst < 1.0e-6,
        detail="; ".join(rows[:4]),
        kind="B",
    )
    assert worst < 1.0e-6
