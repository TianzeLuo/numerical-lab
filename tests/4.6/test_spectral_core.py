"""4.6 章 A/B/C 类断言：谱方法的解析对拍、伪谱的三项账、以及实测收敛率。

证据链分三层，每层都有**独立的对照物**：

1. **解析真值**（A 类）：三角多项式的谱微分精确到舍入、Poisson 核的系数恰是
   :math:`\\rho^{|k|}`、单模态乘积的折返落点可以用手算指出来、Chebyshev 微分矩阵作用在
   多项式上精确、Gibbs 过冲收敛到 :math:`\\frac2\\pi\\mathrm{Si}(\\pi)-1`、
   Burgers 与 DOP853 在同一条半离散方程上对拍；
2. **不变量**（B 类）：Parseval、去混叠对带限输入逐位精确、能量恒等式残差的**相位敏感性**、
   能流与能量预算的逐带对账；
3. **实测收敛率**（C 类）：差分 2/4 阶、谱方法每加倍比值**递减**、RK4 时间阶 4、
   Chebyshev 对非光滑数据的退化、Burgers 能谱 :math:`k^{-2}` 段的窗口斜率。

刻意写进断言的"反例"（每条都有实测数）：
* 谱微分对跳变数据**不收敛**（误差随 N 增长）；
* 能量恒等式残差**不能**当去混叠的判据（带限场两者都是舍入级，全带场强烈依赖相位）；
* :math:`k^{-2}` 段在欠分辨时是**网格激波**的谱（能流末端不回落到 0）。
"""

from __future__ import annotations

import numpy as np
import pytest

from numcore.pde import spectral as sp
from numcore.utils import rng as rng_mod
from numcore.reference import dense as ref
from numcore.verify import (
    assert_order,
    assertion,
    fit_order,
    record,
)

#: 单精度之外的双精度地板：对数轴与"是否还在收敛"的判据都用它。
_FLOOR = 1.0e-14


def _leq(
    measured: float,
    bound: float,
    *,
    label: str,
    theory: str | None = None,
    detail: str = "",
    rtol: float = 0.0,
) -> None:
    """``measured <= bound`` 的断言，登记成**当前测试声明的类别**（A/B 各自对拍）。

    ``numcore.verify.assert_leq`` 固定登记 B 类，而本章的 A 类测试需要 A 类记录，
    所以这里自己走一遍 :func:`record`。
    """
    m = float(measured)
    ok = m <= bound * (1.0 + rtol)
    rec = record(
        label=label,
        theory=theory or f"≤ {bound:g}",
        measured=m,
        tol=f"≤ {bound:g}",
        passed=ok,
        detail=detail,
    )
    assert rec.passed, f"{label}: 实测 {m:.6g} 超过上界 {bound:g}"


def _n_list(hi: int) -> list[int]:
    return [n for n in (16, 24, 32, 48, 64, 96, 128, 192, 256) if n <= hi]


# --------------------------------------------------------------------------- #
# A 类：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="三角多项式的谱微分精确到机器精度；对间断数据误差随 N 增长（两向）", chapter="4.6")
def test_fourier_derivative_exact_on_trig_and_diverges_on_jumps() -> None:
    r"""谱微分 :math:`\partial_x` 在三角多项式上是**恒等**的，在跳变数据上不收敛。

    两向都测：光滑侧相对误差应当贴 :math:`\varepsilon/\text{scale}`；
    跳变侧的误差应当**随 N 增长**（Gibbs 层的导数幅度 :math:`\sim N`），
    所以"谱方法总能微分"是错的——判据是数据光滑，不是方法。
    """
    exact_err = []
    for n in (16, 64):
        x = sp.fourier_grid(n, length=sp.TWO_PI)
        u = np.sin(3 * x) + 0.5 * np.cos(7 * x)
        du = sp.fourier_derivative(u, length=sp.TWO_PI)
        truth = 3 * np.cos(3 * x) - 3.5 * np.sin(7 * x)
        err = float(np.max(np.abs(du - truth)) / np.max(np.abs(truth)))
        exact_err.append(err)
        _leq(err, 1e-13, label=f"n={n} 三角多项式谱微分相对误差", theory="≤ 1e-13（舍入级）")
    record(
        label="两种 N 上的谱微分误差都在舍入级",
        theory="≤ 1e-13（误差是 u 量级，不是「很小」）",
        measured=float(max(exact_err)),
        tol="≤ 1e-13",
        passed=max(exact_err) <= 1e-13,
        detail=f"n=16/64 的相对误差 = {np.array2string(np.asarray(exact_err), precision=3)}",
    )
    assert max(exact_err) <= 1e-13

    jump_err = []
    for n in (32, 64, 128):
        x = sp.fourier_grid(n, length=sp.TWO_PI)
        u = sp.fourier_profile("jump", x, length=sp.TWO_PI)
        du = sp.fourier_derivative(u, length=sp.TWO_PI)
        jump_err.append(float(np.max(np.abs(du))))
    growth = jump_err[-1] / jump_err[0]
    record(
        label="跳变剖面谱微分的幅度随 N 增长",
        theory="≈ (N_hi/N_lo)（Gibbs 层导数 ~ N）",
        measured=growth,
        tol="≥ 3",
        passed=growth >= 3.0,
        detail=f"n=32/64/128 的最大 |du/dx| = {np.array2string(np.asarray(jump_err), precision=3)}",
    )
    assert growth >= 3.0


@assertion("A", theory="Poisson 核的 Fourier 系数恰为 rho^|k|（连续意义）；采样会带 rho^(N/2) 的混叠尾巴", chapter="4.6")
def test_poisson_coefficients_are_rho_to_the_k() -> None:
    r"""解析剖面的系数**不需要拟合**：:math:`a_k=\\rho^{|k|}` 是闭式的。

    两向：小 :math:`\\rho` 时计算值贴舍入；:math:`\\rho` 接近 1 时采样网格会在
    :math:`k` 靠近 :math:`N/2` 处引入 :math:`\\rho^{N/2}` 量级的混叠尾巴——
    这是"采样 vs 连续"的差别，不是 bug，必须写进读数里。
    """
    n = 64
    x = sp.fourier_grid(n)
    for rho, tol in ((0.3, 1e-14), (0.6, 1e-6)):
        a = sp.fourier_coeffs(sp.poisson_profile(x, rho=rho))
        k = np.arange(0, 17)
        dev = float(np.max(np.abs(np.abs(a[:17]) - rho**k)))
        _leq(dev, tol, label=f"rho={rho} 的系数偏离 rho^|k|", theory=f"≤ {tol:g}")
    rho = 0.85
    a = sp.fourier_coeffs(sp.poisson_profile(x, rho=rho))
    dev_hi = float(np.max(np.abs(np.abs(a[:17]) - rho ** np.arange(17))))
    record(
        label="rho=0.85 的混叠尾巴",
        theory="rho^(N/2) = 5.5e-3（采样网格的固有偏差）",
        measured=dev_hi,
        tol="1e-4 ~ 1e-2",
        passed=1e-4 < dev_hi < 1e-2,
        detail="同一公式、同一 N：rho 越大尾巴越明显——「系数 = rho^k」只对连续谱成立",
    )
    assert 1e-4 < dev_hi < 1e-2


@assertion("A", theory="朴素乘积的折返落点可手算：cos(K x)^2 的 2K 模态落到 |2K-N|=K", chapter="4.6")
def test_aliasing_lands_on_the_hand_computed_mode() -> None:
    r"""单模态的混叠落点可以**手算**：:math:`K=N/3` 时 :math:`\\cos^2(Kx)` 的 :math:`2K`
    模态折回 :math:`2K-N=-K`——于是朴素乘积在 :math:`-K` 处凭空多出系数 :math:`1/4`。

    去混叠后这个假模态消失（贴舍入）。这是本章唯一一个"用纸笔就能验证"的混叠现场。
    """
    n = 12
    kk = n // 3
    x = sp.fourier_grid(n)
    u = np.cos(sp.TWO_PI * kk * x)
    # product_* 返回的是**系数**（标准 FFT 顺序），直接读即可
    naive = np.asarray(sp.product_naive(u, u))
    deal = np.asarray(sp.dealias_product(u, u))
    true = np.asarray(sp.product_reference(u, u))
    idx = int(np.argmin(np.abs(sp.fourier_wavenumbers(n) + kk)))
    spurious = abs(naive[idx])
    record(
        label="朴素乘积在 k=-N/3 处的假模态",
        theory="0.25（= cos^2 的 2K 分量折回；真值 0）",
        measured=spurious,
        tol="±0.02",
        passed=abs(spurious - 0.25) < 0.02,
        detail=f"n={n}, K={kk}：naive[{idx}]={naive[idx]:.4f}，真值 {true[idx]:.2e}，去混叠后 {deal[idx]:.2e}",
    )
    assert abs(spurious - 0.25) < 0.02, f"naive[{idx}]={naive[idx]}"
    assert abs(deal[idx]) < 1e-14, f"dealias[{idx}]={deal[idx]}"
    assert abs(true[idx]) < 1e-14, f"ref[{idx}]={true[idx]}"


@assertion("A", theory="Chebyshev 微分矩阵对多项式精确；对 |x| 的误差恒为常数（不收敛）", chapter="4.6")
def test_chebyshev_derivative_exact_on_polynomials_and_stuck_on_corners() -> None:
    r"""Chebyshev 侧的两向：多项式导数精确、尖角导数**不收敛**（误差与 n 无关）。

    :math:`|x|` 的导数在 :math:`x=0` 处跳变，误差是 :math:`O(1)`；
    实测在 :math:`n=16..128` 上恒为 1.0000——"谱方法高阶"的先决条件是光滑性。
    """
    for n in (16, 32):
        x = sp.chebyshev_grid(n)
        d1 = sp.chebyshev_derivative(x)
        _leq(float(np.max(np.abs(d1 - 1.0))), 1e-12, label=f"n={n}: D@x = 1", theory="≤ 1e-12")
        d2 = sp.chebyshev_derivative(x * x)
        _leq(float(np.max(np.abs(d2 - 2 * x))), 1e-12, label=f"n={n}: D@x^2 = 2x", theory="≤ 1e-12")
    corner = []
    for n in (16, 32, 64, 128):
        x = sp.chebyshev_grid(n)
        err = float(np.max(np.abs(sp.chebyshev_derivative(np.abs(x)) + np.sign(x))))
        corner.append(err)
    record(
        label="|x| 的 Chebyshev 导数误差不随 n 下降",
        theory="O(1)：实测约 2.39（与 n 无关，跳变量为 2）",
        measured=float(np.max(corner)),
        tol="1 ~ 3 且随 n 的变化 < 20%",
        passed=1.0 < float(np.max(corner)) < 3.0 and (max(corner) - min(corner)) < 0.2 * max(corner),
        detail=f"n=16/32/64/128 的误差 = {np.array2string(np.asarray(corner), precision=4)}",
    )
    assert max(corner) - min(corner) < 0.2 * max(corner)


@assertion("A", theory="Gibbs 过冲收敛到 (2/pi)Si(pi)-1 = 0.1789797，且不随 K 消失", chapter="4.6")
def test_gibbs_overshoot_approaches_the_si_constant() -> None:
    r"""方波部分和的过冲极限是 :math:`\\frac2\\pi\\mathrm{Si}(\\pi)-1`。

    两向：K 增大时值**下降并逼近**该常数（不是趋 0）；K 很小时偏差更大。
    这正是"振荡只被压窄、幅度不变"的定量版。
    """
    # Si(pi) = int_0^pi sin t / t dt：用稠密梯形法自己算（内核不提供特殊函数，也不该提供）
    t = np.linspace(0.0, np.pi, 200001)
    y = np.where(t == 0.0, 1.0, np.sin(t) / np.where(t == 0.0, 1.0, t))
    si = float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(t)))
    limit = 2.0 / np.pi * si - 1.0
    vals = [sp.gibbs_overshoot(K) for K in (4, 64, 1024)]
    record(
        label="Gibbs 过冲随模态数的走向",
        theory=f"(2/pi)Si(pi)-1 = {limit:.7f}",
        measured=float(abs(vals[-1] - limit)),
        tol="≤ 1e-3",
        passed=abs(vals[-1] - limit) <= 1e-3,
        detail=f"K=4/64/1024 -> {np.array2string(np.asarray(vals), precision=6)}；极限 {limit:.7f}",
    )
    assert abs(vals[-1] - limit) <= 1e-3
    assert vals[-1] < vals[0]
    assert vals[-1] > limit - 1e-3


@assertion("A", theory="Burgers 的 RK4 解在 dt->0 时收敛到 DOP853 的参考解", chapter="4.6")
def test_burgers_matches_a_reference_integrator() -> None:
    r"""与 :func:`numcore.reference.dense.solve_ivp`（DOP853、rtol=1e-12）对拍**同一条半离散方程**。

    空间算子两边共用（它的正确性由混叠那几条断言保证），所以这一条测的是**时间积分**。
    两向：小 dt 贴参考、大 dt 明显偏离。
    """
    n, nu, t_end = 64, 0.01, 0.4
    x = sp.fourier_grid(n, length=sp.TWO_PI)
    kappa = sp.fourier_wavenumbers(n)
    a0 = sp.fourier_coeffs(sp._burgers_initial("sine", x, length=sp.TWO_PI, seed=1))  # noqa: SLF001

    def rhs(t, y):  # noqa: ARG001
        a = y[:n] + 1j * y[n:]
        w = sp.dealias_product_hat(a, a, keep_two_thirds=True)
        da = -0.5j * kappa * w - nu * kappa**2 * a
        return np.concatenate([da.real, da.imag])

    sol = ref.solve_ivp(rhs, (0.0, t_end), np.concatenate([a0.real, a0.imag]), rtol=1e-12, atol=1e-14)
    yT = sol.y[:, -1]
    aT = yT[:n] + 1j * yT[n:]
    scale = float(np.max(np.abs(aT)))
    errs = {}
    for dt in (0.04, 0.0025):
        r = sp.burgers_solve(n, nu, t_end, dt=dt, dealias=True, n_snapshots=2)
        aN = sp.fourier_coeffs(r.u[-1])
        errs[dt] = float(np.max(np.abs(aN - aT)) / scale)
    _leq(errs[0.0025], 1e-10, label="dt=0.0025 与 DOP853 的相对偏差", theory="≤ 1e-10")
    record(
        label="时间步从 0.04 缩到 0.0025 的偏差下降",
        theory="≥ 1e4（四阶：16^4）",
        measured=errs[0.04] / errs[0.0025],
        tol="≥ 1e3",
        passed=errs[0.04] / errs[0.0025] >= 1e3,
        detail=f"dt=0.04 -> {errs[0.04]:.3e}，dt=0.0025 -> {errs[0.0025]:.3e}",
    )
    assert errs[0.04] > errs[0.0025] * 1e3


@assertion("A", theory="a_k ~ k^-(p+1) 的代数指数（尖角 p=1 / 跳变 p=0 / 多项式凸包 p=order）", chapter="4.6")
def test_modal_decay_exponents_for_known_smoothness() -> None:
    r"""三类光滑性的模态衰减指数**用闭式剖面**对拍（不是拟合出来的结论）：

    * 锯齿（跳变）：:math:`a_k\\sim k^{-1}`；
    * 尖顶（C^0 折角）：:math:`a_k\\sim k^{-2}`；
    * 多项式凸包 order=2：:math:`a_k\\sim k^{-4}`。
    解析剖面（Poisson）在同样的窗口里**没有**有限的代数指数（局部斜率一路变陡）。
    """
    n = 512
    x = sp.fourier_grid(n)
    k = np.arange(1, n // 2 + 1)
    for kind, theory_p, tol in (("jump", 1.0, 0.05), ("corner", 2.0, 0.05), ("bump", 4.0, 0.05)):
        a = np.abs(sp.fourier_coeffs(sp.fourier_profile(kind, x))[1 : n // 2 + 1])
        fit = sp.modal_decay_fit(k, a, 8.0, 48.0)
        record(
            label=f"{kind} 的衰减指数",
            theory=f"-{theory_p:g}",
            measured=float(fit["p"]),
            tol=f"±{tol:g}",
            passed=abs(float(fit["p"]) + theory_p) <= tol,
            detail=f"窗口 k∈[8,48]，R²={fit['r2']:.6f}，用点 {fit['n_used']}",
        )
        assert abs(float(fit["p"]) + theory_p) <= tol
    a = np.abs(sp.fourier_coeffs(sp.poisson_profile(x, rho=0.6))[1 : n // 2 + 1])
    sl = sp.local_slopes(k, a)
    seg = [float(np.mean(sl[2:8])), float(np.mean(sl[8:16])), float(np.mean(sl[16:32]))]
    record(
        label="解析剖面的局部斜率一路变陡（不存在有限代数指数）",
        theory="每段至少再陡 2（指数衰减的可执行判据）",
        measured=min(seg[0] - seg[1], seg[1] - seg[2]),
        tol="≥ 2（斜率越来越负）",
        passed=(seg[0] - seg[1] >= 2.0) and (seg[1] - seg[2] >= 2.0),
        detail=f"k∈[2,8]/[8,16]/[16,32] 的平均斜率 = {np.array2string(np.asarray(seg), precision=3)}",
    )
    assert seg[0] - seg[1] >= 2.0 and seg[1] - seg[2] >= 2.0


@assertion("A", theory="逼近误差比系数快一阶：||f-S_N f|| ~ N^-(p+1)；指数情形快速撞地板", chapter="4.6")
def test_fourier_truncation_error_exponents() -> None:
    r"""把"系数衰减"翻译成"逼近误差"：尖角 :math:`N^{-1}`、跳变不下降、解析指数下降。

    两向：代数侧实测斜率 ≈ -1（与理论 -1 相符），跳变侧误差在 N 之间**不下降**，
    解析侧在 N=128 就撞上 1e-14 地板。
    """
    ns = np.asarray([16, 32, 64, 128, 256], dtype=float)
    _, e_corner = sp.fourier_convergence("corner", ns)
    _, e_jump = sp.fourier_convergence("jump", ns)
    _, e_smooth = sp.fourier_convergence("analytic", ns, rho=0.6)
    # 约定：阶一律报正数（err ~ N^-p，取 h=1/N 拟合得到 +p）
    fit = fit_order(1.0 / ns, e_corner)
    record(
        label="尖角的逼近误差指数",
        theory="+1（a_k~k^-2 的尾巴积分给出 N^-1）",
        measured=float(fit.slope),
        tol="±0.15",
        passed=abs(fit.slope - 1.0) <= 0.15,
        detail=f"R²={fit.r2:.6f}，N=16..256；误差 = {np.array2string(e_corner, precision=3)}",
    )
    assert abs(fit.slope - 1.0) <= 0.15
    drop = float(e_jump[0] / e_jump[-1])
    record(
        label="跳变的逼近误差不随 N 下降",
        theory="≤ 1.2（Gibbs 把误差钉住）",
        measured=drop,
        tol="≤ 1.2",
        passed=drop <= 1.2,
        detail=f"N=16 -> {e_jump[0]:.4f}，N=256 -> {e_jump[-1]:.4f}（Chebyshev 的 sign 是同一现象）",
    )
    assert drop <= 1.2
    _leq(float(e_smooth[-1]), 1e-13, label="解析剖面 N=256 的误差", theory="≤ 1e-13（地板）")


# --------------------------------------------------------------------------- #
# B 类：不变量
# --------------------------------------------------------------------------- #
@assertion("B", theory="DFT 往返与 Parseval 恒等式（去混叠乘积对带限输入逐位精确）", chapter="4.6")
def test_roundtrip_parseval_and_bandlimited_exactness() -> None:
    r"""三条不变量：变换往返、Parseval、以及"带限输入 + 2/3 截断 ⇒ 去混叠逐位精确"。"""
    x = sp.fourier_grid(96)
    u = sp.synthetic_field(96, 1.0)
    a = sp.fourier_coeffs(u)
    _leq(float(np.max(np.abs(sp.fourier_synthesize(a).real - u))), 1e-13, label="合成-分解往返", theory="≤ 1e-13（舍入级）")
    _, ek = sp.energy_spectrum(u, length=1.0)
    _leq(
        abs(float(np.sum(ek)) - float(np.mean(u * u))),
        1e-14,
        label="Parseval：sum_k E_k = mean(u^2)",
        theory="≤ 1e-14",
    )
    n = 128
    u = sp.synthetic_field(n, 1.0)
    a_t = np.where(sp.two_thirds_mask(n), sp.fourier_coeffs(u), 0.0)
    u_t = np.real(sp.fourier_synthesize(a_t))
    exact = sp.product_reference(u_t, u_t, factor=6.0)
    deal = sp.dealias_product(u_t, u_t)
    rel = float(np.max(np.abs(deal - exact))) / max(float(np.max(np.abs(exact))), 1e-300)
    _leq(rel, 1e-13, label="带限输入的去混叠乘积 vs 高分辨率参照", theory="≤ 1e-13")

    # 同一带限输入，朴素乘积在**保留带内**也是精确的（折返落在 |k|>=N/3 之外）——
    # 这是"能量恒等式不能当判据"的根：折返不去动保留带。
    kk = np.abs(sp.fourier_wavenumbers(n))
    keep = kk <= float(n) / 3.0
    naive = np.asarray(sp.product_naive(u_t, u_t))
    scale = max(float(np.max(np.abs(exact))), 1e-300)
    _leq(
        float(np.max(np.abs(naive - exact)[keep])) / scale,
        1e-13,
        label="带限输入的朴素乘积偏差（保留带内）",
        theory="≤ 1e-13（折返落在保留带外）",
    )
    outside = float(np.max(np.abs(naive - exact)[~keep])) / scale
    record(
        label="带限输入的朴素乘积偏差（保留带外）",
        theory="非零：折返就落在这里",
        measured=outside,
        tol="1e-4 ~ 1e-1",
        passed=outside > 1e-4,
        detail="同一次乘法：保留带内舍入级、带外可观——「混叠有没有害」取决于你读哪个带",
    )
    assert outside > 1e-4


@assertion("B", theory="能量恒等式残差对相位敏感：同一张谱换相位，残差从 1e-17 变到 1e-1", chapter="4.6")
def test_energy_identity_residual_is_phase_sensitive() -> None:
    r"""把一个"看起来很硬"的指标钉成反面教材。

    :math:`R=\\sum_k \\kappa_k\\,\\mathrm{Im}(\\bar a_k w_k)` 在**带限场 + 2/3 截断**下
    对朴素乘积与去混叠乘积都是舍入级（折返落在保留带外）；把场放宽到全带宽后，
    它变成 :math:`10^{-3}\\sim10^{-1}` 量级，而且**同一张谱换相位就能差 4 个数量级**。
    判据只能是"非线性项自身的偏差"，不能是这个残差。
    """
    n = 128
    k = sp.fourier_wavenumbers(n)
    gen = rng_mod(20261018)
    resids = []
    for tag, phase in (
        ("quad", np.exp(1j * 0.7 * k * k)),
        ("lin", np.exp(1j * 1.3 * k)),
        ("rand", np.exp(1j * gen.uniform(0, sp.TWO_PI, n))),
    ):
        amp = np.where(np.abs(k) >= 1, np.abs(np.where(np.abs(k) >= 1, k, 1)) ** -1.0, 0.0)
        a = amp * phase
        u = np.real(sp.fourier_synthesize(a))
        u = u / max(float(np.max(np.abs(u))), 1e-300)
        a = sp.fourier_coeffs(u)
        r = sp.nonlinear_energy_residual(a, sp.product_naive(u, u))
        resids.append(float(r))
        record(
            label=f"全带场的能量残差（{tag} 相位）",
            theory="朴素乘积应当非零，但量级随相位剧烈变化",
            measured=float(r),
            tol="无（只登记）",
            passed=np.isfinite(r),
            detail="同一张 |a_k|=k^-1 谱，只换相位",
        )
    spread = max(resids) / max(min(resids), 1e-300)
    record(
        label="同谱不同相位的残差跨度",
        theory="≥ 1e3（相位决定混叠是否打中能量账）",
        measured=float(spread),
        tol="≥ 1e3",
        passed=spread >= 1e3,
        detail=f"quad={resids[0]:.2e}，lin={resids[1]:.2e}，rand={resids[2]:.2e}",
    )
    assert spread >= 1e3
    a_t = np.where(sp.two_thirds_mask(n), a, 0.0)
    u_t = np.real(sp.fourier_synthesize(a_t))
    r_naive = sp.nonlinear_energy_residual(a_t, sp.product_naive(u_t, u_t))
    r_deal = sp.nonlinear_energy_residual(a_t, sp.dealias_product(u_t, u_t))
    _leq(abs(r_naive), 1e-15, label="带限场：朴素乘积的残差", theory="≤ 1e-15（折返在带外）")
    _leq(abs(r_deal), 1e-15, label="带限场：去混叠乘积的残差", theory="≤ 1e-15")


@assertion("B", theory="能流与逐带能量预算逐点对账：Pi(k) = -(dE_<=k/dt + 2 nu sum k^2 E_k)", chapter="4.6")
def test_spectral_flux_matches_the_energy_budget() -> None:
    r"""能流不是画出来的装饰：它必须与**逐带能量收支**逐点相等（有限差分对账）。"""
    n, nu = 64, 0.01
    r = sp.burgers_solve(n, nu, 0.6, dealias=True, n_snapshots=121, dt=5e-3)
    i = 60
    t = r.times
    dt = float(t[1] - t[0])
    a = sp.fourier_coeffs(r.u[i])
    kappa = sp.fourier_wavenumbers(n)
    kpos = np.abs(kappa)
    w = sp.dealias_product(r.u[i], r.u[i], keep_two_thirds=True)
    _, flux = sp.spectral_flux(a, w, length=sp.TWO_PI)
    e_band = np.cumsum(r.spectra[i])
    edot = (np.cumsum(r.spectra[i + 1]) - np.cumsum(r.spectra[i - 1])) / (2.0 * dt)
    diss_all = 2.0 * nu * kappa**2 * np.abs(a) ** 2
    worst = 0.0
    for kk in (1, 2, 4, 8, 16, 32):
        diss = float(np.sum(diss_all[kpos <= kk]))
        worst = max(worst, abs(float(edot[kk]) + float(flux[kk]) + diss))
    _leq(worst, 1e-4, label="逐带预算的最大偏差（k=1..32）", theory="≤ 1e-4（有限差分 + 账目）")


@assertion("B", theory="辨率够时能量收支闭合；nu 太小则残差单调变差（网格激波）", chapter="4.6")
def test_burgers_budget_residual_grows_as_viscosity_drops() -> None:
    r"""收支残差随 :math:`\\nu` **单调变差**：这不是时间步的问题，是分辨的问题。"""
    nus = (0.03, 0.02, 0.01, 0.005)
    res = [float(np.max(sp.burgers_solve(64, nu, 1.2, dealias=True, n_snapshots=2).budget_residual)) for nu in nus]
    monotone = all(res[i] < res[i + 1] for i in range(len(res) - 1))
    record(
        label="nu 从 0.03 降到 0.005 的收支残差",
        theory="单调变差（欠分辨时 2/3 截断成为能量汇）",
        measured=float(res[-1]),
        tol="≤ 0.5（n=64 最深的一档）",
        passed=monotone and res[-1] <= 0.5,
        detail=f"nu={nus} -> {np.array2string(np.asarray(res), precision=5)}",
    )
    assert monotone and res[-1] <= 0.5


@assertion("B", theory="平流精确解是位移；c=1、t=1 时退化（所有格式都精确是陷阱）", chapter="4.6")
def test_advection_exact_is_a_shift_and_the_period_trap() -> None:
    r"""``c=1`` 在 :math:`t=1` 处位移正好一个整周期：三种格式的误差都退化到舍入级。

    这不是"格式变准了"，是**测试选错了时刻**。取 :math:`c=0.85` 才能量出空间误差。
    """
    n = 64
    x = sp.fourier_grid(n)
    u0 = sp.poisson_profile(x, rho=0.6)
    exact1 = sp.advection_exact(x, 1.0, c=1.0)
    _leq(float(np.max(np.abs(exact1 - u0))), 1e-14, label="c=1,t=1 的精确解 = 初值", theory="≤ 1e-14")
    trap = sp.advection_error_curve([16, 32, 64, 128], 1.0, c=1.0, rho=0.6)
    _leq(
        float(np.max(trap["spectral"])),
        1e-13,
        label="c=1,t=1 时谱方法的误差",
        theory="≤ 1e-13（与 N 无关的退化）",
    )
    real = sp.advection_error_curve([16, 32, 64, 128], 1.0, c=0.85, rho=0.6)
    assert real["spectral"][0] > 1e-4, "c=0.85 时应当量到真实的空间误差"


@assertion("B", theory="Chebyshev 网格/系数/D 矩阵的结构不变量", chapter="4.6")
def test_chebyshev_structure_invariants() -> None:
    r"""LGL 网格端点、偶/奇函数的系数奇偶性、微分矩阵的行和为零。"""
    n = 32
    x = sp.chebyshev_grid(n)
    _leq(abs(float(x[0]) - 1.0), 1e-15, label="LGL 左端点 = +1", theory="≤ 1e-15")
    _leq(abs(float(x[-1]) + 1.0), 1e-15, label="LGL 右端点 = -1", theory="≤ 1e-15")
    d = sp.chebyshev_differentiation_matrix(n)
    _leq(float(np.max(np.abs(np.sum(d, axis=1)))), 1e-12, label="D 的行和 = 0（常数导数为 0）", theory="≤ 1e-12")
    c_even = sp.chebyshev_coeffs(sp.cheb_profile("runge", x))
    odd_max = float(np.max(np.abs(c_even[1::2])))
    _leq(odd_max, 1e-15, label="Runge（偶函数）的奇次系数", theory="≤ 1e-15")
    c_odd = sp.chebyshev_coeffs(sp.cheb_profile("abs", x))
    _leq(float(np.max(np.abs(c_odd[1::2]))), 1e-15, label="|x|（偶函数）的奇次系数", theory="≤ 1e-15")


# --------------------------------------------------------------------------- #
# C 类：实测收敛率
# --------------------------------------------------------------------------- #
@assertion("C", theory="二阶/四阶中心差分的收敛阶 2/4，每加倍误差降 1/4、1/16", chapter="4.6")
def test_finite_difference_orders_are_two_and_four() -> None:
    r"""有限差分侧：阶是**常数比值**；窗口取阶梯上半段（小 N 还没进渐近区）。"""
    ns = np.asarray([16, 24, 32, 48, 64, 96, 128, 192, 256], dtype=float)
    curve = sp.advection_error_curve(list(ns.astype(int)), 0.6, c=0.85, rho=0.6, orders=(2, 4))
    e2 = np.asarray(curve["fd2"])
    e4 = np.asarray(curve["fd4"])
    mask = ns >= 128
    assert_order(1.0 / ns[mask], e2[mask], theory=2.0, tol=0.1, kind="C", label="二阶差分实测阶")
    assert_order(1.0 / ns[mask], e4[mask], theory=4.0, tol=0.15, kind="C", label="四阶差分实测阶")
    # 每加倍比值：只取 2 的幂档（doubling_ratios 按步长比归一化，混档位会带来假波动）
    pow2 = np.asarray([0, 2, 4, 6, 8])
    ns2 = ns[pow2]
    rec2 = sp.doubling_ratios(ns2, e2[pow2])
    rec4 = sp.doubling_ratios(ns2, e4[pow2])
    r2 = float(rec2[-1])
    r4 = float(rec4[-1])
    _leq(abs(r2 - 0.25), 0.02, label="二阶每加倍比值（最细两级）", theory="0.25（渐近值，粗网格端偏大）")
    _leq(abs(r4 - 0.0625), 0.006, label="四阶每加倍比值（最细两级）", theory="0.0625")
    record(
        label="差分的每加倍比值从上方单调趋近理论值",
        theory="单调下降且有下界 0.25 / 0.0625（粗网格端还没进渐近区）",
        measured=float(rec2[0]),
        tol="0.4 ~ 0.8",
        passed=bool(np.all(np.diff(rec2) < 0) and np.all(rec2 >= 0.24) and np.all(np.diff(rec4) < 0) and np.all(rec4 >= 0.06)),
        detail=f"二阶 {np.array2string(rec2, precision=4)}；四阶 {np.array2string(rec4, precision=4)}（2 的幂档）",
    )
    pre = ns <= 64
    pre_slope = float(fit_order(1.0 / ns[pre], e4[pre]).slope)
    record(
        label="小 N 段四阶格式的假阶",
        theory="≤ 3.5（还没进渐近区）",
        measured=pre_slope,
        tol="≤ 3.5",
        passed=pre_slope <= 3.5,
        detail=f"N=16/32/64 的误差 = {np.array2string(e4[pre], precision=4)}：窗口是断言的一部分",
    )
    assert pre_slope <= 3.5


@assertion("C", theory="谱误差每加倍比值递减（指数收敛的可执行判据），差分比值恒定", chapter="4.6")
def test_spectral_error_beats_every_power_law() -> None:
    r"""招牌：**比值是不是常数**。差分恒定、谱方法每步再乘一个更小的数。"""
    ns = np.asarray([16, 24, 32, 48, 64, 96], dtype=float)
    curve = sp.advection_error_curve(list(ns.astype(int)), 0.6, c=0.85, rho=0.6, orders=(2, 4))
    e_sp = np.asarray(curve["spectral"])
    e_2 = np.asarray(curve["fd2"])
    ratios_sp = np.asarray(
        [r for r, keep in zip(sp.doubling_ratios(ns, e_sp), (e_sp[:-1] > _FLOOR) & (e_sp[1:] > _FLOOR)) if keep]
    )
    ratios_2 = np.asarray([r for r, keep in zip(sp.doubling_ratios(ns, e_2), (e_2[:-1] > _FLOOR) & (e_2[1:] > _FLOOR)) if keep])
    strict = bool(np.all(np.diff(ratios_sp) < 0.0))
    record(
        label="谱误差每加倍比值严格递减",
        theory="严格递减（指数收敛）",
        measured=float(ratios_sp[-1] / ratios_sp[0]),
        tol="< 1",
        passed=strict and ratios_sp[-1] < ratios_sp[0],
        detail=f"比值序列 = {np.array2string(ratios_sp, precision=5)}",
    )
    assert strict
    record(
        label="谱方法每加倍比值 vs 差分每加倍比值",
        theory="≤ 0.01（差两个数量级以上）",
        measured=float(ratios_sp[-1] / ratios_2[-1]),
        tol="≤ 0.01",
        passed=float(ratios_sp[-1] / ratios_2[-1]) <= 0.01,
        detail=(
            f"谱 = {np.array2string(ratios_sp, precision=4)}，"
            f"二阶差分 = {np.array2string(ratios_2, precision=4)}（差分比值有下界 0.25，谱没有下界）"
        ),
    )
    assert float(ratios_sp[-1] / ratios_2[-1]) <= 0.01
    # 斜率本身在变：窗口越靠右，拟合出的"阶"越陡
    slopes = []
    for lo, hi in ((16, 32), (24, 48), (32, 64)):
        m = (ns >= lo) & (ns <= hi) & (e_sp > _FLOOR)
        slopes.append(float(fit_order(1.0 / ns[m], e_sp[m]).slope) if int(np.count_nonzero(m)) >= 3 else float("nan"))
    record(
        label="谱误差的局部斜率逐段变陡",
        theory="每段至少 +1.5（对数-对数图上向下弯）",
        measured=float(slopes[-1] - slopes[0]),
        tol="≥ 1.5",
        passed=slopes[-1] - slopes[0] >= 1.5,
        detail=f"窗口 [16,32]/[24,48]/[32,64] 的斜率 = {np.array2string(np.asarray(slopes), precision=2)}；误差 = {np.array2string(e_sp, precision=3)}",
    )
    assert slopes[-1] - slopes[0] >= 1.5


@assertion("C", theory="积分因子 RK4 的时间阶 = 4（每减半步长误差降 16 倍）", chapter="4.6")
def test_rk4_temporal_order_is_four() -> None:
    r"""时间阶用**自收敛**量：同一格式 dt 减半，误差比值应当趋近 1/16。"""
    n, nu, t_end = 64, 0.01, 0.4
    fine = sp.burgers_solve(n, nu, t_end, dt=t_end / 640.0, dealias=True, n_snapshots=2)
    a_ref = sp.fourier_coeffs(fine.u[-1])
    scale = float(np.max(np.abs(a_ref)))
    dts = np.asarray([0.04, 0.02, 0.01, 0.005], dtype=float)
    errs = []
    for dt in dts:
        r = sp.burgers_solve(n, nu, t_end, dt=float(dt), dealias=True, n_snapshots=2)
        errs.append(float(np.max(np.abs(sp.fourier_coeffs(r.u[-1]) - a_ref)) / scale))
    errs_arr = np.asarray(errs)
    assert_order(dts, errs_arr, theory=4.0, tol=0.2, kind="C", label="积分因子 RK4 实测时间阶")
    ratios = errs_arr[1:] / errs_arr[:-1]
    record(
        label="每减半步长的误差比值",
        theory="≈ 1/16 = 0.0625",
        measured=float(np.max(np.abs(ratios - 0.0625))),
        tol="≤ 0.01",
        passed=bool(np.max(np.abs(ratios - 0.0625)) <= 0.01),
        detail=f"dt={np.array2string(dts, precision=4)} -> 比值 {np.array2string(ratios, precision=4)}",
    )
    assert np.max(np.abs(ratios - 0.0625)) <= 0.01


@assertion("C", theory="Chebyshev：解析指数、尖角代数（误差 ~ n^-1）、跳变不收敛", chapter="4.6")
def test_chebyshev_convergence_degrades_off_smoothness() -> None:
    r"""换基不换结论：光滑指数、尖角代数、跳变不降——三种光滑性各测一条。"""
    ns = np.asarray([16, 24, 32, 48, 64, 96, 128], dtype=float)
    _, e_runge = sp.chebyshev_convergence("runge", ns)
    _, e_abs = sp.chebyshev_convergence("abs", ns)
    _, e_sign = sp.chebyshev_convergence("sign", ns)
    ratio = np.asarray(
        [r for r, keep in zip(sp.doubling_ratios(ns, e_runge), (e_runge[:-1] > _FLOOR) & (e_runge[1:] > _FLOOR)) if keep]
    )
    strict = bool(ratio.size >= 2 and np.all(np.diff(ratio) < 0.0))
    record(
        label="Runge 的误差每加倍比值递减",
        theory="严格递减（指数）",
        measured=float(ratio[-1] / ratio[0]),
        tol="< 1",
        passed=strict,
        detail=f"比值 = {np.array2string(ratio, precision=5)}",
    )
    assert strict
    fit_abs = fit_order(1.0 / ns, e_abs)
    record(
        label="|x| 的 Chebyshev 误差指数",
        theory="+1（尖角 ⇒ 误差 ~ n^-1）",
        measured=float(fit_abs.slope),
        tol="±0.15",
        passed=abs(fit_abs.slope - 1.0) <= 0.15,
        detail=f"R²={fit_abs.r2:.6f}，误差 = {np.array2string(e_abs, precision=4)}",
    )
    assert abs(fit_abs.slope - 1.0) <= 0.15
    drop = float(e_sign[0] / e_sign[-1])
    record(
        label="sign(x) 的误差不随 n 下降",
        theory="≤ 1.2（跳变）",
        measured=drop,
        tol="≤ 1.2",
        passed=drop <= 1.2,
        detail=f"n=16 -> {e_sign[0]:.4f}，n=128 -> {e_sign[-1]:.4f}",
    )
    assert drop <= 1.2


@assertion("C", theory="Burgers 能谱的 k^-2 段：窗口内实测斜率 ≈ -2（窗口写进 detail）", chapter="4.6")
def test_burgers_cascade_slope_in_the_inertial_window() -> None:
    r"""能谱 :math:`k^{-2}` 段实测：窗口由 :func:`numcore.pde.spectral.cascade_window` 给，
    上界同时受黏性截断 :math:`k_d` 与 2/3 带边缘 :math:`N/3` 限制。

    两向：分辨够时斜率贴 -2；把 :math:`\\nu` 压到欠分辨时**能流末端不再回落到 0**
    （实测 -4.28），"级联"数字变成网格产物。
    """
    res = sp.burgers_solve(128, 0.02, 1.5, dealias=True, n_snapshots=2)
    lo, hi = sp.cascade_window(0.02, 1.5, 128)
    fit = sp.fit_power_law(res.k[lo : hi + 1], res.spectra[-1][lo : hi + 1])
    record(
        label=f"能谱斜率（窗口 [{lo},{hi}]）",
        theory="-2（激波/间断的谱）",
        measured=float(fit["p"]),
        tol="±0.25",
        passed=abs(float(fit["p"]) + 2.0) <= 0.25,
        detail=f"R²={fit['r2']:.6f}，用点 {fit['n_used']}；k_d={1 / np.sqrt(0.02 * 1.5):.2f}",
    )
    assert abs(float(fit["p"]) + 2.0) <= 0.25
    bad = sp.burgers_solve(128, 0.002, 1.5, dealias=True, n_snapshots=2)
    record(
        label="欠分辨（nu=0.002）时能流末端",
        theory="应当 ≈ 0（非线性项守恒总能量）",
        measured=float(bad.flux[-1]),
        tol="≤ -0.5 视为失控",
        passed=float(bad.flux[-1]) <= -0.5,
        detail=f"收支残差 {float(np.max(bad.budget_residual)):.3f}，E 从 {bad.energy[0]:.3f} 涨到 {bad.energy[-1]:.3f}",
    )
    assert float(bad.flux[-1]) <= -0.5


@assertion("C", theory="2/3 截断的代价与混叠误差都随输入谱指数衰减（同一张谱的两种尾巴）", chapter="4.6")
def test_alias_and_truncation_cost_decay_with_input_spectrum() -> None:
    r"""混叠与截断都随输入谱变陡而变小，但**两者之比**稳定在 2~7：

    :math:`\\text{截断}/\\text{混叠}` 实测 2.10（decay=0.5）、3.13（1.0）、5.65（2.0）、6.75（2.5）。
    "去混叠改善精度"必须说清报的是哪一笔账。
    """
    n = 128
    decays = np.asarray([0.5, 1.0, 2.0, 2.5], dtype=float)
    alias, trunc = [], []
    for d in decays:
        u = sp.synthetic_field(n, float(d), seed=20261018)
        r = sp.alias_error(u, u)
        alias.append(float(r["alias_rel"]))
        trunc.append(float(r["truncation_rel"]))
    alias_arr = np.asarray(alias)
    trunc_arr = np.asarray(trunc)
    assert bool(np.all(np.diff(alias_arr) < 0.0)), "混叠误差应当随 decay 单调下降"
    ratio = trunc_arr / alias_arr
    record(
        label="截断代价 / 混叠误差 之比",
        theory="2 ~ 7（截断这一笔更大）",
        measured=float(np.min(ratio)),
        tol="≥ 2",
        passed=float(np.min(ratio)) >= 2.0 and float(np.max(ratio)) <= 7.0,
        detail=f"decay={np.array2string(decays, precision=2)} -> 比值 {np.array2string(ratio, precision=3)}",
    )
    assert float(np.min(ratio)) >= 2.0 and float(np.max(ratio)) <= 7.0
    fit_alias = fit_order(decays, alias_arr)
    record(
        label="混叠误差随 decay 的幂律指数",
        theory="负指数（谱越陡混叠越小）",
        measured=float(fit_alias.slope),
        tol="≤ -2",
        passed=float(fit_alias.slope) <= -2.0,
        detail=f"R²={fit_alias.r2:.6f}；alias = {np.array2string(alias_arr, precision=3)}",
    )
    assert float(fit_alias.slope) <= -2.0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
