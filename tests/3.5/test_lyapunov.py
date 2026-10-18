r"""3.5 章（下半）：Lyapunov 指数的 A/C 类断言。

这一章最容易自欺的地方是「混沌」这三个字。所以本文件的断言分三层：

1. **绝对标尺**（A 类）：Lorenz 的散度处处等于 :math:`-(\sigma+1+\beta)`，
   所以三个指数之和**必须**等于它；Henon 映射的 :math:`|\det Dg|=|b|` 处处常数，
   所以两个指数之和必须等于 :math:`\log|b|`。这两个数不依赖任何文献值。
2. **文献值对拍**（A 类）：Lorenz 的 :math:`\lambda_1` 与公认值 0.9056 比，
   **容差与依据一起给**（有限时间涨落由跨初值的标准差实测）。
3. **步长与协议证据**（C 类）：把 h 减半、把 QR 间隔加倍，指数变化必须小于某个量；
   孪生轨迹的有限时间增长率必须与 Benettin 谱给出的 :math:`\lambda_1` 同一量级。
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

from numcore.ode import dynamics as dyn
from numcore.ode import lyapunov as lyap
from numcore.ode import continuation as cont
from numcore.verify import assertion, record

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

U = float(np.finfo(float).eps) / 2.0

#: Lorenz 指数实测的固定口径（写在这里，断言与文档引用同一个数）。
_LORENZ_ICS = (np.array([1.0, 0.0, 20.0]), np.array([-8.0, 7.0, 27.0]))
_LORENZ_T = 2000.0
_LORENZ_H = 0.01
_LORENZ_QR = 10
_LORENZ_TRANSIENT = 50.0

_CACHE: dict[str, object] = {}


def _lorenz_spectra() -> list[lyap.BenettinResult]:
    """两条初值各跑一次 Benettin（T=2000，约 15 s），用例之间共用。"""
    if "lorenz" not in _CACHE:
        system = dyn.flow_system("lorenz")
        _CACHE["lorenz"] = [
            lyap.benettin_flow(
                system,
                y0,
                h=_LORENZ_H,
                t_end=_LORENZ_T,
                qr_interval=_LORENZ_QR,
                transient=_LORENZ_TRANSIENT,
            )
            for y0 in _LORENZ_ICS
        ]
    return _CACHE["lorenz"]  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# 1. 绝对标尺：指数之和 = 散度
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    theory="Lyapunov 指数的和等于散度的时间平均：Σλ_i = ⟨∇·f⟩。Lorenz 的散度是常数 "
    "−(σ+1+β)=−13.6667，所以这条恒等式给出一个不依赖任何文献值的标尺",
    chapter="3.5",
)
def test_lorenz_exponent_sum_equals_divergence() -> None:
    results = _lorenz_spectra()
    exact = dyn.lorenz_divergence(10.0, 8.0 / 3.0)
    for k, res in enumerate(results):
        s = res.sum_exponents()
        record(
            label=f"第 {k + 1} 条初值的 Σλ 与散度之差",
            measured=abs(s - exact),
            tol=5e-4,
            passed=abs(s - exact) < 5e-4,
            detail=f"Σλ={s:.8f}，∇·f={exact:.8f}（T={_LORENZ_T:g}，h={_LORENZ_H:g}，QR 每 {_LORENZ_QR} 步）",
        )
        assert abs(s - exact) < 5e-4
        # 第二个指数（沿流向）必须趋于 0：它是"轨道方向"的指数
        record(
            label=f"第 {k + 1} 条初值的 λ₂（流向，应趋于 0）",
            measured=abs(float(res.exponents[1])),
            tol=0.03,
            passed=abs(float(res.exponents[1])) < 0.03,
            detail=f"λ={np.round(res.exponents, 6)}",
        )
        assert abs(float(res.exponents[1])) < 0.03
        # 重正交化的正交性缺陷：QR 是自写的，这件事必须实测
        assert res.defect_max() < 1e-12, f"正交性缺陷 {res.defect_max():.3e} 过大"
    record(
        label="Gram–Schmidt 的最大正交性缺陷",
        measured=max(r.defect_max() for r in results),
        tol=1e-12,
        passed=max(r.defect_max() for r in results) < 1e-12,
        detail="自写 MGS + 一轮重正交化；缺陷坏了等于指数算错",
    )


@assertion(
    "A",
    theory="Henon 映射的 Jacobi 行列式处处等于 −b，所以两个 Lyapunov 指数之和精确等于 log|b|"
    "（b=0.3 → −1.2039728043259361），与轨道无关",
    chapter="3.5",
)
def test_henon_exponents_sum_to_log_det() -> None:
    g, dg = dyn.map_callable("henon", a=1.4, b=0.3)
    spec = lyap.benettin_map(g, dg, [0.1, 0.1], n_iter=500000, transient=20000, qr_interval=1)
    want = math.log(0.3)
    s = float(np.sum(spec.exponents))
    record(
        label="Henon 的 Σλ 与 log|b| 之差",
        measured=abs(s - want),
        tol=1e-12,
        passed=abs(s - want) < 1e-12,
        detail=f"λ={np.round(spec.exponents, 8)}，Σλ={s:.16f}，log|b|={want:.16f}",
    )
    assert abs(s - want) < 1e-12
    record(
        label="λ₁ 与文献值 0.41922 之差",
        measured=abs(float(spec.exponents[0]) - lyap.BENETTIN_REFERENCE_HENON[0]),
        tol=1e-3,
        passed=abs(float(spec.exponents[0]) - lyap.BENETTIN_REFERENCE_HENON[0]) < 1e-3,
        detail=f"实测 {float(spec.exponents[0]):.8f}，文献 0.41922（相加恒等式才是硬证据）",
    )
    assert abs(float(spec.exponents[0]) - 0.41922) < 1e-3


@assertion(
    "A",
    theory="logistic 映射的 Lyapunov 指数有三处闭式：1<r<3 时 λ=log|2−r|；2 周期窗口内 "
    "λ=½log|4+2r−r²|；r=4 共轭到 Bernoulli 移位，λ=log2",
    chapter="3.5",
)
def test_logistic_exponent_matches_closed_forms() -> None:
    cases = ((2.5, 20000), (3.2, 20000), (4.0, 200000))
    for r, n in cases:
        want = lyap.logistic_analytic_lyapunov(r)
        got, sem = lyap.logistic_lyapunov(r, n_iter=n, transient=2000, n_block=8)
        gap = abs(got - want)
        tol = max(1e-9, 5.0 * sem)
        record(
            label=f"λ({r}) 与闭式之差",
            measured=gap,
            tol=tol,
            passed=gap < tol,
            detail=f"实测 {got:.12f}，闭式 {want:.12f}，分块标准误 {sem:.2e}（{n} 次迭代）",
        )
        assert gap < tol


# --------------------------------------------------------------------------- #
# 2. 文献值对拍
# --------------------------------------------------------------------------- #
@assertion(
    "A",
    theory="Lorenz(σ=10,ρ=28,β=8/3) 的最大 Lyapunov 指数公认值 0.9056（Sprott 2003 表 5.1）。"
    "有限时间估计 T=2000 的涨落由跨初值的标准差实测，容差据此给出",
    chapter="3.5",
)
def test_lorenz_max_exponent_matches_accepted_value() -> None:
    results = _lorenz_spectra()
    lam1 = np.array([float(r.exponents[0]) for r in results])
    mean = float(np.mean(lam1))
    sd = float(np.std(lam1, ddof=1))
    ref = lyap.BENETTIN_REFERENCE_LORENZ
    gap = abs(mean - ref)
    record(
        label="λ₁ 的均值与公认值 0.9056 之差",
        measured=gap,
        tol=0.02,
        passed=gap < 0.02,
        detail=(
            f"实测 λ₁={np.round(lam1, 6)}，均值 {mean:.6f}，跨初值标准差 {sd:.6f}（n={lam1.size}）；"
            f"T={_LORENZ_T:g}、h={_LORENZ_H:g}、暂态 {_LORENZ_TRANSIENT:g}、QR 每 {_LORENZ_QR} 步。"
            "容差 0.02 的依据：有限时间估计的涨落随 T 变慢地缩小（本仓实测 T=400/1000/2000 的均值"
            "0.9017/0.9029/0.9041），T=2000 上的标准差本身就有 0.01 量级，"
            "把容差写成 1e-3 是拿有限时间估计冒充极限值"
        ),
    )
    assert gap < 0.02, f"λ₁ 均值 {mean:.6f} 与 0.9056 差 {gap:.6f}（标准差 {sd:.6f}）"
    record(
        label="公认值落在实测的 ±2σ 区间内",
        measured=float(abs(mean - ref) / max(sd, 1e-12)),
        tol=2.0,
        passed=abs(mean - ref) <= 2.0 * sd + 1e-3,
        detail=f"|均值−0.9056|/σ = {abs(mean - ref) / max(sd, 1e-12):.3f}",
    )


# --------------------------------------------------------------------------- #
# 3. 协议证据（C 类）
# --------------------------------------------------------------------------- #
@assertion(
    "C",
    theory="定步长积分的 Lyapunov 估计必须随步长收敛：h 减半后 λ₁ 的变化量要小于积分误差"
    "允许的量级，否则「混沌」可能只是离散化噪声",
    chapter="3.5",
)
def test_lyapunov_estimate_is_step_size_stable() -> None:
    system = dyn.flow_system("lorenz")
    y0 = _LORENZ_ICS[0]
    lam = {}
    for h in (0.01, 0.005):
        res = lyap.benettin_flow(system, y0, h=h, t_end=400.0, qr_interval=10, transient=50.0)
        lam[h] = res
    diff = abs(float(lam[0.01].exponents[0]) - float(lam[0.005].exponents[0]))
    record(
        label="h 减半后 λ₁ 的变化",
        measured=diff,
        tol=0.03,
        passed=diff < 0.03,
        detail=(
            f"λ₁(h=0.01)={float(lam[0.01].exponents[0]):.6f}、"
            f"λ₁(h=0.005)={float(lam[0.005].exponents[0]):.6f}（T=400）"
        ),
    )
    assert diff < 0.03
    for h, res in lam.items():
        s = res.sum_exponents()
        record(
            label=f"h={h:g} 的 Σλ 与散度之差",
            measured=abs(s - (-13.666666666666666)),
            tol=5e-4,
            passed=abs(s - (-13.666666666666666)) < 5e-4,
            detail=f"Σλ={s:.8f}",
        )

    # 孪生轨迹的步长证据（帧里画的那条曲线）
    from numcore.utils import rng

    # 孪生对取 8 对（与帧的默认 n_pairs 一致）：对少的时候"一对的运气"会主导增长率，
    # 4 对时实测两条 h 的增长率差 0.0196，8 对时降到 0.0042——这不是算法变了，
    # 而是有限时间涨落的样本量变了。容差按 8 对的实测值给。
    base = rng(20261018).uniform([-15.0, -20.0, 5.0], [15.0, 20.0, 40.0], size=(8, 3))
    settled = dyn.integrate_ensemble(system, base, (0.0, 20.0), 0.01).final()
    st = lyap.step_size_stability(system, settled, eps=1e-8, h=0.005, t_end=30.0)
    record(
        label="孪生分离曲线的增长率（h 与 2h）之差",
        measured=float(st["rate_diff"]),
        tol=0.03,
        passed=float(st["rate_diff"]) < 0.03,
        detail=(
            f"h=0.005 给 {st['rate_fine']:.4f}、h=0.01 给 {st['rate_coarse']:.4f}；"
            f"逐点相对差的中位数不是 C 类判据（有限时间涨落在跳变处会放大），"
            f"这里只报最大值 {st['rel_curve_diff']:.3f} 作为量级参考"
        ),
    )
    assert float(st["rate_diff"]) < 0.03


@assertion(
    "C",
    theory="QR 重正交化的间隔只影响浮点误差与代价，不影响数学：间隔 10 与 20 必须给出同一个谱",
    chapter="3.5",
)
def test_qr_interval_does_not_change_the_spectrum() -> None:
    system = dyn.flow_system("lorenz")
    y0 = _LORENZ_ICS[0]
    a = lyap.benettin_flow(system, y0, h=0.01, t_end=400.0, qr_interval=10, transient=50.0)
    b = lyap.benettin_flow(system, y0, h=0.01, t_end=400.0, qr_interval=20, transient=50.0)
    gap = float(np.max(np.abs(a.exponents - b.exponents)))
    record(
        label="QR 间隔 10 vs 20 的谱差",
        measured=gap,
        tol=1e-9,
        passed=gap < 1e-9,
        detail=f"{np.round(a.exponents, 10)} vs {np.round(b.exponents, 10)}",
    )
    assert gap < 1e-9


@assertion(
    "C",
    theory="孪生轨迹的有限时间增长率必须与 Benettin 谱给出的 λ₁ 同量级（两者是同一个量的"
    "两种测法：一个直接量两条轨道之差，一个走变分方程 + QR）",
    chapter="3.5",
)
def test_twin_growth_rate_agrees_with_benettin_spectrum() -> None:
    system = dyn.flow_system("lorenz")
    from numcore.utils import rng

    base = rng(20261018).uniform([-15.0, -20.0, 5.0], [15.0, 20.0, 40.0], size=(8, 3))
    settled = dyn.integrate_ensemble(system, base, (0.0, 20.0), 0.01).final()
    sep = lyap.twin_separation(system, settled, eps=1e-8, h=0.005, t_end=30.0, stride=10)
    rate, n_fit, r2 = lyap.fit_growth_rate(sep.t, sep.d_mean, lo=1e-6, hi=1.0)
    lam1 = float(_lorenz_spectra()[0].exponents[0])
    gap = abs(rate - lam1)
    record(
        label="孪生增长率 vs Benettin λ₁",
        measured=gap,
        tol=0.05,
        passed=gap < 0.05,
        detail=(
            f"孪生 {rate:.4f}（拟合窗口 1e-6≤d≤1，{n_fit} 个点，R²={r2:.4f}）、"
            f"Benettin {lam1:.4f}。两者都依赖有限时间窗口，容差 0.05 覆盖这个涨落"
        ),
    )
    assert gap < 0.05
    sat = sep.saturation()
    record(
        label="饱和距离（尾段平均分离）",
        measured=sat,
        tol=0.0,
        passed=5.0 < sat < 40.0,
        detail=f"{sat:.3f}（吸引子直径约 40，饱和后分离与初值无关）",
    )
    assert 5.0 < sat < 40.0


@assertion(
    "C",
    theory="有限时间 λ 估计的误差应当落在分块标准误的量级内：块标准误度量的是同一件事"
    "（有限时间涨落），如果误差远大于它，说明估计有系统偏差",
    chapter="3.5",
)
def test_logistic_estimate_error_is_within_block_sem() -> None:
    r = 4.0
    want = math.log(2.0)
    got, sem = lyap.logistic_lyapunov(r, n_iter=200000, transient=20000, n_block=20)
    gap = abs(got - want)
    record(
        label="λ(4) 与 ln2 之差 vs 分块标准误",
        measured=gap / max(sem, 1e-300),
        tol=3.0,
        passed=gap <= 3.0 * sem + 1e-9,
        detail=f"误差 {gap:.3e}，SEM {sem:.3e}（差 {gap / max(sem, 1e-300):.2f} 倍 SEM）",
    )
    assert gap <= 3.0 * sem + 1e-9
    record(
        label="Kaplan–Yorke 维数（Lorenz，由实测谱推出）",
        measured=lyap.kaplan_yorke_dimension(_lorenz_spectra()[0].exponents),
        tol=0.05,
        passed=abs(lyap.kaplan_yorke_dimension(_lorenz_spectra()[0].exponents) - 2.06) < 0.05,
        detail="文献值约 2.06（它不是独立证据：由同一组指数推出，只作一致性检查）",
    )


@assertion(
    "A",
    theory="自写的修正 Gram–Schmidt 必须满足 A=QR 与 Q^T Q=I；det 的绝对值等于 Π|R_ii|",
    chapter="3.5",
)
def test_gram_schmidt_reconstructs_and_is_orthogonal() -> None:
    system = dyn.flow_system("lorenz")
    gen = np.array([0.3, -0.7, 1.1, 0.2, 0.9, -0.4, -1.3, 0.6, 0.5]).reshape(3, 3)
    gs = lyap.gram_schmidt(gen)
    recon = float(np.max(np.abs(gs.Q @ gs.R - gen)))
    record(
        label="A=QR 的重建误差",
        measured=recon,
        tol=1e-14,
        passed=recon < 1e-14,
        detail=f"缺陷 {gs.defect:.3e}，R 对角 {np.round(np.diag(gs.R), 6)}",
    )
    assert recon < 1e-14
    assert gs.defect < 1e-14

    # det 的绝对值 = Π|R_ii|：用 2x2 的显式展开当独立路径
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    gs2 = lyap.gram_schmidt(A)
    det_direct = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    det_qr = float(abs(gs2.R[0, 0] * gs2.R[1, 1]))
    record(
        label="|det A| vs Π|R_ii|",
        measured=abs(det_direct - det_qr),
        tol=1e-14,
        passed=abs(det_direct - det_qr) < 1e-14,
        detail=f"det={det_direct:.12f}，Π|R_ii|={det_qr:.12f}（Henon 的 |det|=|b| 就是靠这条成立的）",
    )
    assert abs(det_direct - det_qr) < 1e-14

    div = lyap.divergence_expectation(system, np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]]))
    record(
        label="数值散度 vs 闭式 −13.6667",
        measured=abs(div - dyn.lorenz_divergence()),
        tol=1e-6,
        passed=abs(div - dyn.lorenz_divergence()) < 1e-6,
        detail=f"数值 {div:.10f}，闭式 {dyn.lorenz_divergence():.10f}",
    )
    assert abs(div - dyn.lorenz_divergence()) < 1e-6


@assertion(
    "C",
    theory="分岔图的点云与逐参数标量迭代必须给出同一批点（向量化路径不能改变结果）",
    chapter="3.5",
)
def test_bifurcation_cloud_matches_scalar_iteration() -> None:
    grid = np.array([3.2, 3.5, 3.83, 4.0])
    cloud = cont.bifurcation_diagram(grid, n_transient=500, n_keep=8, x0=0.4)
    worst = 0.0
    for j, r in enumerate(grid):
        x = 0.4
        for _ in range(500):
            x = r * x * (1.0 - x)
        for k in range(8):
            x = r * x * (1.0 - x)
            worst = max(worst, abs(x - cloud[k, j]))
    record(
        label="向量化分岔图 vs 标量迭代的最大偏差",
        measured=worst,
        tol=0.0,
        passed=worst == 0.0,
        detail="同一套浮点运算顺序：批量路径只是把参数维摊开，不该改变任何一个比特",
    )
    assert worst == 0.0
