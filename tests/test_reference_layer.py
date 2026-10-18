"""参照物自身的测试。

为什么值得给"靶子"写测试：如果参照物是错的，那么用它做对拍的 1000 条断言全都在验证错误的东西。
本文件用**互相独立的三种方法**交叉验证：有理数精确解 / 任意精度 / 符号积分 / SciPy，
任何两两不一致都必须被抓住。
"""

from __future__ import annotations

import math
import os
import pathlib
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

from numcore.reference import analytic as ra
from numcore.reference import dense as rd
from numcore.reference import exact as rx

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 精确有理算术
# --------------------------------------------------------------------------- #
def test_exact_solve_matches_dense_and_fractions() -> None:
    """2×2 整数系统：有理精确解必须能与 scipy 解对上，且精确残差为 0。"""
    from fractions import Fraction

    A = [[2, 1], [1, 3]]
    b = [3, 5]
    x = rx.exact_solve(A, b)
    assert x == [Fraction(4, 5), Fraction(7, 5)], x
    assert np.allclose(np.array([float(v) for v in x]), rd.solve(np.array(A, float), np.array(b, float)))
    assert rx.exact_residual(A, x, b) == [Fraction(0), Fraction(0)]


def test_exact_det_and_inverse_are_exact() -> None:
    """3×3 整数矩阵：行列式与逆都是有理数，逐项精确可比。"""
    from fractions import Fraction

    A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    assert rx.exact_det(A) == Fraction(1)
    inv = rx.exact_inverse(A)
    prod = [[sum(A[i][k] * inv[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    assert prod == [[Fraction(1), Fraction(0), Fraction(0)], [Fraction(0), Fraction(1), Fraction(0)], [Fraction(0), Fraction(0), Fraction(1)]]
    assert rx.exact_det([[1, 2], [2, 4]]) == Fraction(0)


def test_exact_solve_detects_singular() -> None:
    with pytest.raises(ValueError, match="奇异"):
        rx.exact_solve([[1, 2], [2, 4]], [1, 2])


def test_exact_rref_rank() -> None:
    """行最简形：秩为 2 的 3×4 矩阵，最后一行为 0。"""
    R = rx.exact_rref([[1, 2, 3, 4], [2, 4, 6, 8], [1, 0, 1, 0]])
    assert int(np.sum([any(v != 0 for v in row) for row in R])) == 2


# --------------------------------------------------------------------------- #
# 任意精度
# --------------------------------------------------------------------------- #
def test_mp_solve_matches_dense_to_high_precision() -> None:
    """病态 Hilbert 系统：任意精度解与 scipy 解在**良态残差**意义下一致。"""
    n = 6
    A = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    b = np.ones(n)
    x_mp = rx.mp_solve(A, b, dps=60)
    x_np = rd.solve(A, b)
    x_mp_np = np.array([float(x_mp[i]) for i in range(n)])
    assert np.allclose(x_mp_np, x_np, rtol=1e-6, atol=1e-6)
    # 精确解的残差应当小得多（这正是"病态但算法稳定"的体现）
    r_mp = float(np.max(np.abs(A @ x_mp_np - b)))
    r_np = float(np.max(np.abs(A @ x_np - b)))
    assert r_mp < 1e-12 and r_np < 1e-12, (r_mp, r_np)


def test_mp_eigsy_and_det_agree_with_dense() -> None:
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    vals = sorted(float(v) for v in rx.mp_eigsy(A, dps=40))
    ref = sorted(float(v) for v in rd.eigh(A)[0])
    assert np.allclose(vals, ref, rtol=1e-12)
    assert abs(float(rx.mp_det(A, dps=40)) - rd.det(A)) < 1e-12


def test_mp_quad_beats_double_precision() -> None:
    """``∫₀¹ e^{-x²} dx``：50 位与 80 位精度自洽到 1e-45，且与 scipy 在 1e-15 内一致。"""
    import mpmath as mp

    v50 = rx.mp_quad(lambda t: mp.e ** (-(t * t)), 0.0, 1.0, dps=50)
    v80 = rx.mp_quad(lambda t: mp.e ** (-(t * t)), 0.0, 1.0, dps=80)
    assert abs(v50 - v80) < mp.mpf("1e-45"), (v50, v80)
    v_sp = rd.quad(lambda x: float(np.exp(-(x**2))), 0.0, 1.0)[0]
    assert abs(float(v50) - v_sp) < 1e-15
    # 实用型：被积函数只给 double 实现，也能避开求积公式自身的误差
    assert abs(rx.mp_quad_np(lambda x: float(np.exp(-(x**2))), 0.0, 1.0) - v_sp) < 1e-15


# --------------------------------------------------------------------------- #
# 符号
# --------------------------------------------------------------------------- #
def test_symbolic_integral_and_derivative() -> None:
    """符号积分是真值来源：``∫₀¹ x² dx = 1/3``、``d/dx sin(x²) = 2x cos(x²)``。"""
    import sympy as sp

    assert rx.sym_integrate("x**2", "x", 0, 1) == sp.Rational(1, 3)
    assert rx.sym_simplify(rx.sym_diff("sin(x**2)", "x") - sp.sympify("2*x*cos(x**2)")) == 0
    assert float(rx.sym_lambdify("x**3", "x")(2.0)) == 8.0
    assert rx.sym_limit("sin(x)/x", "x", 0) == 1


def test_symbolic_series_gives_truncation_term() -> None:
    """泰勒展开用于背书截断误差：``(1+h)^(1/2) = 1 + h/2 − h²/8 + …``。"""
    import sympy as sp

    h = sp.Symbol("h")
    s = rx.sym_series("sqrt(1+h)", "h", 0, 3)
    assert sp.simplify(s - (1 + h / 2 - h**2 / 8)) == 0


# --------------------------------------------------------------------------- #
# 解析参考问题
# --------------------------------------------------------------------------- #
def test_runge_function_and_lebesgue_constants() -> None:
    """Runge 函数值、等距 vs Chebyshev 的 Lebesgue 常数增长（指数 vs 对数）。"""
    assert abs(float(ra.runge(0.0)) - 1.0) < 1e-15
    assert abs(float(ra.runge(1.0)) - 1.0 / 26.0) < 1e-15
    lam_eq_8 = ra.lebesgue_constant(ra.equispaced_nodes(8))
    lam_ch_8 = ra.lebesgue_constant(ra.chebyshev_nodes(8))
    lam_eq_16 = ra.lebesgue_constant(ra.equispaced_nodes(16))
    lam_ch_16 = ra.lebesgue_constant(ra.chebyshev_nodes(16))
    # 等距：Λ ≈ (2^{n+1})/(e n log n) 量级 → 从 8 到 16 至少涨 10 倍
    assert lam_eq_16 > 10.0 * lam_eq_8, (lam_eq_8, lam_eq_16)
    # Chebyshev：Λ ≈ (2/π)log n + O(1) → 缓慢增长，且始终比等距小
    assert lam_ch_16 < 3.0 * lam_ch_8, (lam_ch_8, lam_ch_16)
    assert lam_ch_16 < lam_eq_16


def test_interp_error_bound_is_an_upper_bound() -> None:
    """插值误差界：``|f − p_n| ≤ max|f^{(n+1)}|/(n+1)! · Π|x − x_i|``。

    取 ``f = sin``（三阶导绝对值 ≤ 1）与 3 个等距节点，逐点验证实测误差不超过解析上界。
    """
    nodes = np.array([-1.0, 0.0, 1.0])
    xs = np.linspace(-1, 1, 401)
    ys = rd.barycentric(nodes, np.sin(nodes), xs)
    err = np.abs(ys - np.sin(xs))
    f3_sup = 1.0  # |sin 的三阶导| ≤ 1
    bounds = np.array([ra.interp_error_bound(f3_sup, nodes, float(x)) for x in xs])
    assert np.all(err <= bounds + 1e-14), float(np.max(err - bounds))
    # 上界也不能松得离谱：取一个非节点处，误差与上界应当同量级
    k = int(np.argmin(np.abs(xs - 0.5)))
    assert err[k] > 1e-3 and bounds[k] < 1.0, (err[k], bounds[k])


def test_gauss_legendre_nodes_integrate_polynomials_exactly() -> None:
    """n 点 Gauss–Legendre 精确积分 2n−1 次多项式（这里用 n=5、次数 9）。"""
    x, w = ra.gauss_legendre_nodes(5)
    for deg in (0, 1, 2, 5, 9):
        exact = 2.0 / (deg + 1) if deg % 2 == 0 else 0.0
        got = float(np.sum(w * x**deg))
        assert abs(got - exact) < 1e-14, (deg, got, exact)
    with pytest.raises(AssertionError):
        assert abs(float(np.sum(w * x**10))) < 1e-14  # 10 次不再精确


def test_pendulum_period_matches_small_angle_limit() -> None:
    """θ₀ → 0 时精确周期 → 2π√(L/g)；θ₀ = π/2 时大 ~18%（教学要点）。"""
    T0 = 2.0 * math.pi * math.sqrt(1.0 / 9.81)
    assert abs(ra.pendulum_period(1e-6) - T0) / T0 < 1e-12
    big = ra.pendulum_period(math.pi / 2)
    assert 0.17 < big / T0 - 1.0 < 0.19, big / T0


def test_kepler_orbit_conserves_energy_and_momentum() -> None:
    """开普勒轨道：解析解必须精确守恒能量与角动量（这是 3.3 章的"真值"）。"""
    ts = np.linspace(0.0, 2.0 * np.pi, 200)
    e = 0.6
    x, y = ra.kepler_orbit(ts, a=1.0, e=e)
    # 数值微分求速度（中心差分，误差 O(h²)）
    h = 1e-6
    xp, yp = ra.kepler_orbit(ts + h, a=1.0, e=e)
    xm, ym = ra.kepler_orbit(ts - h, a=1.0, e=e)
    vx = (xp - xm) / (2 * h)
    vy = (yp - ym) / (2 * h)
    E = ra.kepler_energy(x, y, vx, vy)
    L = ra.kepler_angular_momentum(x, y, vx, vy)
    assert np.max(np.abs(E - E[0])) < 1e-8, np.max(np.abs(E - E[0]))
    assert np.max(np.abs(L - L[0])) < 1e-8
    # 近点/远点距离应满足 r = a(1 ∓ e)
    r = np.sqrt(x**2 + y**2)
    assert abs(float(np.min(r)) - (1 - e)) < 1e-6
    # 采样时刻未必正好落在远点，允许一个采样间隔的误差；同时不得超出理论远点
    assert abs(float(np.max(r)) - (1 + e)) < 1e-3
    assert float(np.max(r)) <= (1 + e) + 1e-12


def test_kepler_equation_residual() -> None:
    for M in (0.1, 1.0, 3.0, 6.0):
        E = ra.kepler_solve(M, 0.7)
        assert abs(E - 0.7 * math.sin(E) - math.fmod(M, 2 * math.pi)) < 1e-13


def test_heat_sine_mode_second_order_consistency() -> None:
    """热方程单模态解：端点恒为 0；且二阶中心差分逼近 ``u_xx`` 的误差以 ``O(Δx²)`` 衰减。

    这条测试本身就是 A 类断言的样板：不去比一个拍脑袋的常数，而是**测误差的阶**。
    """
    assert np.allclose(ra.heat_sine_mode(np.array([0.0, 1.0]), 0.3), 0.0, atol=1e-15)
    t, alpha, k = 0.05, 0.7, 2
    dt = 1e-7

    def discrepancy(n: int) -> float:
        xs = np.linspace(0.0, 1.0, n)
        u = ra.heat_sine_mode(xs, t, alpha=alpha, k=k)
        ut = (ra.heat_sine_mode(xs, t + dt, alpha=alpha, k=k) - ra.heat_sine_mode(xs, t - dt, alpha=alpha, k=k)) / (2 * dt)
        uxx = np.gradient(np.gradient(u, xs), xs)
        interior = slice(3, -3)  # 端点处 np.gradient 退化为一阶单侧差分，不参与比较
        return float(np.max(np.abs(ut[interior] - alpha * uxx[interior])))

    e1, e2 = discrepancy(201), discrepancy(401)
    assert e2 < e1
    assert 3.0 < e1 / e2 < 5.0, (e1, e2)  # 二阶格式：网格加密一倍误差降为 1/4


def test_heat_dirichlet_series_matches_single_mode() -> None:
    """级数解在初值为单模态时必须精确退化为单模态解（系数 b₁=1，其余为 0）。"""
    L = 1.0
    f0 = lambda x: math.sin(math.pi * x / L)  # noqa: E731
    xs = np.linspace(0.0, L, 41)
    exact = ra.heat_sine_mode(xs, 0.1, alpha=0.5, L=L, k=1)
    series = ra.heat_dirichlet_series(xs, 0.1, f0=f0, alpha=0.5, L=L, modes=8)
    assert np.max(np.abs(series - exact)) < 1e-12


def test_advection_periodic_shifts_and_wraps() -> None:
    """周期对流：``u(x,t) = f0(x − ct)``，绕回必须正确。"""
    f0 = lambda z: np.sin(2 * np.pi * z)  # noqa: E731
    xs = np.linspace(0, 1, 101, endpoint=False)
    u = ra.advection_periodic(xs, 0.25, c=1.0, L=1.0, f0=f0)
    shifted = f0(xs - 0.25)
    assert np.allclose(u, shifted, atol=1e-15)
    assert np.allclose(ra.advection_periodic(xs, 1.25, c=1.0, L=1.0, f0=f0), shifted, atol=1e-15)


def test_poisson_sine_exact_solution() -> None:
    """``−u'' = sin(kπx)`` 的精确解在内部点上必须让二阶差分收敛到右端项。"""
    xs = np.linspace(0, 1, 4001)
    u = ra.poisson_1d_sine(xs, k=3)
    uxx = np.gradient(np.gradient(u, xs), xs)
    assert np.max(np.abs(-uxx - np.sin(3 * np.pi * xs))) < 5e-3
    assert abs(float(ra.poisson_1d_sine(np.array([0.5]), k=3)[0]) - (1 / (3 * math.pi)) ** 2 * math.sin(1.5 * math.pi)) < 1e-15


def test_gibbs_overshoot_matches_theory() -> None:
    """Wilbraham–Gibbs：绝对过冲 → 0.1789797（即跳变高度 2 的 8.95%）。"""
    over = ra.gibbs_overshoot(n_terms=999)
    assert 0.176 < over < 0.181, over
    assert 0.088 < over / 2.0 < 0.0905  # 相对于跳变高度的比例 —— "约 9%" 的出处


# --------------------------------------------------------------------------- #
# 参照层的接口契约：SciPy 版本敏感的返回值 / 缺依赖时的行为
# --------------------------------------------------------------------------- #
def test_dense_quad_matches_analytic_integrals() -> None:
    r"""``quad`` 要对得上能解析算出的积分：``∫₀¹ 4/(1+x²) dx = π``、``∫₀¹ x³ dx = 1/4``。

    相对误差门槛取 1e-10——比 QUADPACK 在 ``epsabs=epsrel=1e-13`` 下的实际表现（~1e-16）
    松三个数量级，只钉"参照物没算错"，不钉浮点噪声。顺带钉住返回值形状：
    第三项必须是含 ``neval`` 的字典（``full_output`` 的解析一旦写错，这条就会红）。
    """
    val, err, neval = rd.quad(lambda x: 4.0 / (1.0 + x * x), 0.0, 1.0)
    assert abs(val - math.pi) / math.pi <= 1e-10, (val, err, neval)
    assert err >= 0.0
    assert isinstance(neval, int) and neval > 0

    val3, err3, neval3 = rd.quad(lambda x: x**3, 0.0, 1.0)
    assert abs(val3 - 0.25) / 0.25 <= 1e-10, (val3, err3, neval3)
    assert isinstance(neval3, int) and neval3 > 0


def test_dense_quad_survives_full_output_four_tuple() -> None:
    r"""回归：``ier != 0`` 时 SciPy 的 ``full_output=1`` 返回 **4 元组**，不许解包崩。

    这是 2.2 章绕开 ``numcore.reference.dense.quad`` 的真实原因：原先写死的
    ``val, err, info = _sint.quad(..., full_output=1)`` 碰上
    ``(val, abserr, infodict, msg)`` 就抛 ``ValueError: too many values to unpack (expected 3)``。
    修复后有两条要求：(a) 不崩、照常返回三元组；(b) SciPy 在 ``full_output`` 模式下**不再**
    发 ``IntegrationWarning``，参照层要把这条警告重发出来，否则"真值"可能是个被判定为发散的数。
    """
    hard = lambda x: np.sin(1.0 / x) / x if x > 0 else 0.0  # noqa: E731

    # 先直连 SciPy 立住前提：这条积分确实走 ier != 0 的那条返回路径。
    # 不在这里 import scipy：P1 规定 scipy 只允许出现在 numcore/reference/（tests/ 也在静态扫描范围内）。
    raw = rd._sint.quad(hard, 1e-12, 1.0, epsabs=1e-13, epsrel=1e-13, limit=500, full_output=1)
    assert len(raw) == 4 and isinstance(raw[3], str) and raw[3], "本机 SciPy 没走到 4 元组分支，测试前提失效"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        val, err, neval = rd.quad(hard, 1e-12, 1.0)  # 修复前：这里抛 ValueError
    assert math.isfinite(val) and math.isfinite(err)
    assert isinstance(neval, int) and neval > 0
    assert "IntegrationWarning" in [w.category.__name__ for w in caught], [w.category.__name__ for w in caught]


def test_reference_dense_fails_loudly_without_scipy() -> None:
    """scipy 不可用时参照层必须**明确报错**，而不是悄悄降级成别的算法。

    P1 把 scipy 收敛到 ``numcore/reference/`` 一处，就是为了让"没装 scipy"一次性暴露：
    本包顶层 ``import scipy.integrate as _sint``，缺库即 ``ImportError``，没有自造兜底。
    本机装了 scipy，所以用子进程装一个 import 拦截器来复现"没装"的环境。
    """
    script = textwrap.dedent(
        """
        import sys

        class _NoScipy:
            def find_spec(self, name, path=None, target=None):
                if name == "scipy" or name.startswith("scipy."):
                    raise ImportError("scipy is not installed (blocked by test)")
                return None

        sys.meta_path.insert(0, _NoScipy())
        try:
            import numcore.reference.dense  # noqa: F401
        except ImportError as exc:
            print("IMPORTERROR", exc)
        else:
            print("NOERROR")
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("IMPORTERROR"), proc.stdout
    assert "scipy" in proc.stdout
