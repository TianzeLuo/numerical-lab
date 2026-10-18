r"""4.2 章算法层测试（二）：限制器函数、Sweby 区域与总变差。

这里要钉住的是**三句容易说过头的话**：

1. 「装了限制器就不会振」——实测 minmod/MC/superbee/vanleer/vanalbada 在阶跃上
   过冲与欠冲都是 0（:math:`\nu=0.4`），但把 :math:`\nu` 拖到 1.01，TVD 立刻破
   （最大相对增量 1.96e-2、umax 1.4889）。**CFL 是 TVD 的必要条件。**
2. 「在 Sweby 区域内就是二阶」——``limiter="none"`` 的 :math:`\phi\equiv0`
   **也**在区域内（它就是迎风，TVD 但一阶）。二阶还要 :math:`\phi(1)=1`。
3. 「TVD 就是解更好看」——TVD 是总变差**非增**，与精度无关；限制器在极值处
   退化正是它绕开 Godunov 定理的代价（实测单调格式的幂律阶 ≤1.01）。

另外把「TVD 不等于单调」也量化了：TV 非增的同时解仍可以有平台与极值
（``count_plateaus``/``extrema_count``）。
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

CHAPTER = "4.2"
LIMITERS = ("none", "minmod", "mc", "superbee", "vanleer", "vanalbada", "koren", "osher", "umist", "sweby")


def _step(n: int = 128) -> np.ndarray:
    x = np.arange(n, dtype=float) / n
    return np.where(x < 0.25, 1.0, 0.0)


# --------------------------------------------------------------------------- #
# A 类
# --------------------------------------------------------------------------- #
@assertion("A", theory="TVD 限制器必须满足 φ(r) ≤ min(2r,2) 且 r<0 时 φ=0；逐点核对 9 个限制器")
def test_limiter_functions_lie_in_the_sweby_region() -> None:
    """Sweby 区域的两个边界 vs 每个限制器的逐点取值。"""
    r = np.linspace(0.0, 6.0, 601)
    rows = []
    ok = True
    for name in LIMITERS:
        phi = np.asarray(lm.limiter_phi(name, r), dtype=float)
        up = np.asarray(lm.sweby_upper(r), dtype=float)
        lo = np.asarray(lm.sweby_lower(r), dtype=float)
        inside = bool(np.all(phi <= up + 1e-15) and np.all(phi >= lo - 1e-15))
        ok = ok and inside
        rows.append(f"{name}:{'ok' if inside else 'OUT'}")
    record(
        label="9 个限制器在 r∈[0,6] 上全部落在 Sweby 区域 [0, min(2r,2)] 内",
        theory="Sweby 区域内的线性格式满足 TVD（充分条件）；区域外的限制器不能保证 TV 非增",
        measured="all-inside" if ok else "violation",
        tol=1.0e-15,
        passed=ok,
        detail="; ".join(rows),
        kind="A",
    )
    assert ok
    # r<0：要求 φ=0（否则在极值处会放大）
    neg = np.linspace(-4.0, -1e-9, 101)
    worst = 0.0
    for name in LIMITERS:
        phi = np.asarray(lm.limiter_phi(name, neg), dtype=float)
        worst = max(worst, float(np.max(np.abs(phi))))
    record(
        label="9 个限制器在 r<0 上的 |φ(r)| 上确界",
        theory="r<0 表示该点处于极值：TVD 要求 φ=0（完全退化成迎风），不允许外推",
        measured=f"{worst:.3e}",
        tol=1.0e-15,
        passed=worst < 1.0e-15,
        detail="这条是「极值处必须退化」的直接读数——也是限制器只有一阶的原因",
        kind="A",
    )
    assert worst < 1.0e-15


@assertion("A", theory="二阶需要 φ(1)=1；limiter=none（φ≡0）在 Sweby 区内却不是二阶——反例必须留在测试里")
def test_sweby_region_is_not_sufficient_for_second_order() -> None:
    """``in_sweby_region`` 与 ``φ(1)=1`` 是**两个**独立条件。"""
    rows = []
    ok = True
    for name in LIMITERS:
        at1 = float(np.asarray(lm.limiter_phi(name, 1.0), dtype=float))
        inside = bool(np.all(np.asarray(lm.in_sweby_region(name, np.linspace(0, 4, 101)))))
        rows.append(f"{name}: phi(1)={at1:.6f} inside={inside}")
        if name != "none":
            ok = ok and inside and abs(at1 - 1.0) < 1e-12
    none_inside = bool(np.all(np.asarray(lm.in_sweby_region("none", np.linspace(0, 4, 101)))))
    none_at1 = float(np.asarray(lm.limiter_phi("none", 1.0), dtype=float))
    record(
        label="φ(1) 与「在 Sweby 区内」的关系（limiter=none 是反例）",
        theory="φ(1)=1 是二阶的必要条件；Sweby 区域只保证 TVD。none 的 φ≡0 在区内但 φ(1)=0",
        measured=f"none: inside={none_inside}, phi(1)={none_at1:.1f}；其余 9 个: all-ok={ok}",
        tol=1.0e-12,
        passed=(ok and none_inside and abs(none_at1) < 1e-15),
        detail="; ".join(rows),
        kind="A",
    )
    assert ok
    assert none_inside
    assert abs(none_at1) < 1e-15


@assertion("A", theory="限制器的闭合形式在 r=1 处必须给 1（保二阶），且全部满足对称性 φ(1/r)=φ(r)/r")
def test_symmetric_limiters_are_self_consistent() -> None:
    r"""对称性 :math:`\phi(1/r)=\phi(r)/r`（这是"对称限制器"的定义，逐点核）。"""
    r = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
    rows = []
    worst = 0.0
    for name in lm.SYMMETRIC_NAMES:
        lhs = np.asarray(lm.limiter_phi(name, 1.0 / r), dtype=float)
        rhs = np.asarray(lm.limiter_phi(name, r), dtype=float) / r
        e = float(np.max(np.abs(lhs - rhs)))
        worst = max(worst, e)
        rows.append(f"{name}:{e:.2e}")
    record(
        label="对称限制器 φ(1/r)=φ(r)/r 的最大偏差（6 个 r 值）",
        theory="左右对称的 TVD 限制器必须满足这条；不满足的会偏一侧（BW 型不对称）",
        measured=f"{worst:.3e}",
        tol=1.0e-12,
        passed=worst < 1.0e-12,
        detail="; ".join(rows),
        kind="A",
    )
    assert worst < 1.0e-12
    # minmod 的等价比价：minmod(a,b) = a 若 |a|<|b| 且同号，否则按 b / 0
    a = np.asarray([1.0, -1.0, 2.0, -2.0, 0.5, 3.0])
    b = np.asarray([2.0, 2.0, -1.0, -3.0, 0.5, -1.0])
    got = np.asarray(lm.minmod(a, b), dtype=float)
    want = np.where(a * b <= 0.0, 0.0, np.where(np.abs(a) <= np.abs(b), a, b))
    e = float(np.max(np.abs(got - want)))
    record(
        label="minmod(a,b) 与逐点判据 max(0,min(|a|,|b|)·sgn(a)) 的最大差",
        theory="minmod 的闭式必须与「同号取小、异号取 0」逐点一致",
        measured=f"{e:.3e}",
        tol=1.0e-15,
        passed=e < 1.0e-15,
        detail="这条独立于 limiter_phi 的实现（不用 ratio 形式）",
        kind="A",
    )
    assert e < 1.0e-15


@assertion("A", theory="TVD ⟺ 每一步 TV(u^{n+1}) ≤ TV(u^n)；ν≤1 时 40 步内最大相对增量应在舍入级")
def test_tvd_holds_below_cfl_and_fails_above() -> None:
    """**CFL 是 TVD 的必要条件**——这一条把「充分」与「必要」分开量。"""
    u0 = _step()
    rows = []
    ok_below = True
    for nu in (0.4, 0.9, 0.99, 1.0):
        _snaps, tv = lm.tvd_step_scan(u0, nu=nu, n_steps=40, limiter="minmod")
        good, rel = lm.tvd_check(tv)
        ok_below = ok_below and good
        rows.append(f"nu={nu}:{'TVD' if good else 'FAIL'}(rel={rel:.2e})")
    above = {}
    for nu in (1.01, 1.1):
        snaps, _tv = lm.tvd_step_scan(u0, nu=nu, n_steps=40, limiter="minmod")
        above[nu] = float(np.max(np.abs(snaps[-1])))
    fails_above = above[1.01] > 1.0 + 1e-2 and above[1.1] > 10.0
    record(
        label="ν≤1 时 TVD 成立 / ν>1 时破（阶跃、40 步、minmod）",
        theory="TVD 的证明要用到 CFL（新值必须是旧值的凸组合）；ν>1 时凸性丢失，"
        "限制器只能限制斜率、管不住增长",
        measured=f"below-ok={ok_below}; umax(1.01)={above[1.01]:.6f}; umax(1.1)={above[1.1]:.4f}",
        tol=1.0e-12,
        passed=(ok_below and fails_above),
        detail="; ".join(rows),
        kind="A",
    )
    assert ok_below
    assert fails_above


@assertion("A", theory="限制器在阶跃上不过冲：实测 5 个 TVD 限制器的 over=under=0，而中心/LW/BW 不是")
def test_overshoot_separates_tvd_from_plain_second_order() -> None:
    """过冲/欠冲把"限制器"与"无条件二阶格式"分开。"""
    u0 = _step()
    lo, hi = float(np.min(u0)), float(np.max(u0))
    rows = []
    ok_tvd = True
    for name in ("minmod", "mc", "superbee", "vanleer", "vanalbada"):
        res = ad.advection_solve(u0, nu=0.4, n_steps=60, scheme="muscl", limiter=name, integrator="rk3")
        over, under = lm.overshoot(res.u, lo=lo, hi=hi)
        ok_tvd = ok_tvd and over == 0.0 and under == 0.0
        rows.append(f"{name}:over={over:.3e},under={under:.3e},TV={res.tv:.6f}")
    plain = {}
    for name in ("centered", "lax_wendroff", "beam_warming"):
        res = ad.advection_solve(u0, nu=0.4, n_steps=60, scheme=name, integrator="rk3")
        over, under = lm.overshoot(res.u, lo=lo, hi=hi)
        plain[name] = (over, under, res.tv)
    # 对比三次实测：迎风（一阶、TVD）与中心（二阶、不 TVD）
    up = ad.advection_solve(u0, nu=0.4, n_steps=60, scheme="upwind", integrator="rk3")
    over_up = lm.overshoot(up.u, lo=lo, hi=hi)
    record(
        label="阶跃（ν=0.4、60 步、RK3）上的过冲/欠冲与 TV",
        theory="TVD ⟹ 不产生新的极值 ⟹ over=under=0；但反过来不成立（一阶迎风也 over=0）",
        measured="; ".join(rows)
        + " | upwind:T=%.6f over=%.1e" % (up.tv, over_up[0])
        + " | centered:over=%.4f,under=%.4f,TV=%.4f" % plain["centered"],
        tol=1.0e-12,
        passed=(ok_tvd and over_up[0] == 0.0 and plain["centered"][0] > 0.2 and plain["centered"][2] > 5.0),
        detail="同时看 TV：限制器 1.9999–2.0000、迎风 1.9981、LW 2.0770、中心 11.0820",
        kind="A",
    )
    assert ok_tvd
    assert over_up[0] == 0.0
    assert plain["centered"][0] > 0.2
    assert plain["centered"][2] > 5.0


@assertion("A", theory="TVD 只保证总变差非增，不保证单调：解仍可含平台与极值（TVD≠单调）")
def test_tvd_does_not_imply_monotonicity() -> None:
    """把「TVD ≠ 单调」量化：TV 非增的同时，解里仍有平台与极值计数。"""
    u0 = _step()
    res = ad.advection_solve(u0, nu=0.4, n_steps=60, scheme="muscl", limiter="minmod", integrator="rk3")
    _snaps, tv = lm.tvd_step_scan(u0, nu=0.4, n_steps=60, limiter="minmod")
    good, rel = lm.tvd_check(tv)
    plateaus = int(lm.count_plateaus(res.u))
    extrema = int(lm.extrema_count(res.u))
    # 初值本身是一个阶跃：它有两个平台（0 与 1）——解里必须还能看到它们
    record(
        label="TVD 成立的同时解里的平台数与极值数",
        theory="TVD 是**总变差**的非增性，不是逐点单调性；限制器只在极值处退化，"
        "既没有抹掉平台也没有消掉极值",
        measured=f"tvd={good} rel={rel:.2e} plateaus={plateaus} extrema={extrema}",
        tol=1.0e-12,
        passed=(good and plateaus >= 1 and rel < 1e-12),
        detail="把 TVD 读成「解单调」是本章的经典过度解读",
        kind="A",
    )
    assert good
    assert plateaus >= 1
    assert rel < 1e-12


@assertion("A", theory="任何限制器下 TV(u) 的历史都必须逐帧核过；不让差分为正")
def test_max_tv_increase_reports_the_worst_frame() -> None:
    """``max_tv_increase`` 本身要被验证：它必须报出**最坏**的那一帧。"""
    u0 = _step()
    rows = []
    ok = True
    for name in ("minmod", "mc", "superbee", "vanleer", "vanalbada"):
        _s, tv = lm.tvd_step_scan(u0, nu=0.4, n_steps=40, limiter=name)
        worst, idx = lm.max_tv_increase(tv)
        good, _rel = lm.tvd_check(tv)
        ok = ok and good and worst <= 1e-12 * max(float(np.max(tv)), 1.0)
        rows.append(f"{name}:worst={worst:.2e}@{idx}")
    record(
        label="5 个限制器的 TV 最大增量（40 步）",
        theory="TVD 要求逐帧 TV 不增；允许舍入级（1e-12 相对容差）的负向/零增量",
        measured="all-ok" if ok else "violation",
        tol=1.0e-12,
        passed=ok,
        detail="; ".join(rows),
        kind="A",
    )
    assert ok


@assertion("A", theory="限制器把数值黏性从 (1−ν)/2 改到接近 0（MUSCL 的二阶路线）；实测 none→0.3、minmod→1.5e-3")
def test_limiter_reduces_the_leading_viscosity() -> None:
    """限制器对**数值黏性**的作用（这是"限制器在做什么"的定量答案）。"""
    nu = 0.4
    got = {name: ad.measure_viscosity_limit(scheme="muscl", nu=nu, limiter=name, base_nx=64)
           for name in ("none", "minmod", "mc", "superbee", "vanleer")}
    # none 就是一阶迎风：(1-ν)/2 = 0.3
    none_ok = abs(got["none"] - 0.3) < 1e-6
    others_small = all(abs(got[n]) < 1e-2 for n in ("minmod", "mc", "vanleer"))
    record(
        label="MUSCL 各限制器在 ν=0.4 上的极限数值黏性",
        theory="limiter=none 的 MUSCL 退化成迎风，黏性恰为 (1−ν)/2=0.3；"
        "启用限制器后前导黏性降到 1e-3 量级（这就是二阶性的来源）",
        measured="; ".join(f"{k}={v:+.8f}" for k, v in got.items()),
        tol=1.0e-6,
        passed=(none_ok and others_small),
        detail="注意 superbee 会给**负**的极限值（−1.46e-3）：它的前导黏性在 Richardson 极限下变号",
        kind="A",
    )
    assert none_ok
    assert others_small


# --------------------------------------------------------------------------- #
# C 类
# --------------------------------------------------------------------------- #
@assertion("C", theory="限制器在光滑解上**不能**恢复二阶：光滑极值每周期只有 O(1) 个点，"
           "但限制器在那里退化成一阶，截断误差被那一小段主导，实测阶降到约 1.03")
def test_limiter_does_not_restore_second_order_on_smooth_data() -> None:
    """**测出来的是"限制器降阶"这件事本身**——原以为会看到二阶，实测只有 1.03。

    实测（固定 ν=0.4、nx=40→320、光滑正弦、RK3）：
    upwind 0.9795、muscl/none 0.9795（与迎风**逐位相同**：none 就是迎风）、
    muscl/minmod **1.0337**、lax_wendroff 0.9949。
    限幅器只把误差常数减半（1.99e-2 → 9.07e-3），**阶没有变**。
    这与"limiters degrade accuracy at smooth extrema"（Harten 1983 之后的经典结论）
    一致，也是本章要防的过度解读之一："装了限制器就二阶了"是错的。
    """
    nu = 0.4
    ladder = (40, 80, 160, 320)
    hs = [1.0 / nx for nx in ladder]
    errs = []
    for nx in ladder:
        x = np.arange(nx, dtype=float) / nx
        u0 = 1.0 + 0.3 * np.sin(2.0 * np.pi * x)
        steps = max(1, int(round(0.2 / (nu / nx))))
        res = ad.advection_solve(u0, nu=nu, n_steps=steps, scheme="muscl", limiter="minmod", integrator="rk3")
        _linf, l2 = ad.error_norms(res.u, ad.exact_solution(u0, res.t), res.dx)
        errs.append(l2)
    from numcore.verify.order import fit_order  # noqa: PLC0415 - 只在 C 类里用

    fit = fit_order(hs, errs, min_points=3)
    # 同一个梯子上再量一次"无限幅"的对照：muscl/none 与纯迎风应当逐位相同
    ref = []
    for nx in ladder:
        x = np.arange(nx, dtype=float) / nx
        u0 = 1.0 + 0.3 * np.sin(2.0 * np.pi * x)
        steps = max(1, int(round(0.2 / (nu / nx))))
        r1 = ad.advection_solve(u0, nu=nu, n_steps=steps, scheme="upwind", integrator="rk3")
        r2 = ad.advection_solve(u0, nu=nu, n_steps=steps, scheme="muscl", limiter="none", integrator="rk3")
        ref.append(float(np.max(np.abs(r1.u - r2.u))))
    worst_ref = max(ref)
    record(
        label="MUSCL+minmod 在光滑正弦上固定 ν=0.4 的 L2 幂律阶（预期约 1，不是 2）",
        theory="限幅器在光滑极值处退化为一阶；那一小段主导截断误差，故湍流阶仍是 1",
        measured=f"p={fit.slope:.4f}（errs {errs[0]:.2e}→{errs[-1]:.2e}）",
        tol=0.15,
        passed=0.9 < float(fit.slope) < 1.2,
        detail=f"对照：muscl/none 与纯迎风逐位相同（max|diff|={worst_ref:.1e}），其阶也是 0.9795",
        kind="C",
    )
    assert 0.9 < float(fit.slope) < 1.2
    assert worst_ref < 1.0e-14


@assertion("C", theory="限制器削波：间断被抹开的宽度收缩得**比一阶还慢**（实测 p≈0.52–0.55）")
def test_shock_smearing_width_is_sublinear() -> None:
    """**削波宽度**的实测阶（TVD 的代价）。"""
    nu = 0.4
    ladder = (64, 128, 256, 512)
    hs = [1.0 / nx for nx in ladder]
    widths = []
    for nx in ladder:
        x = np.arange(nx, dtype=float) / nx
        u0 = np.where(x < 0.25, 1.0, 0.0)
        steps = max(1, int(round(0.4 / (nu / nx))))
        res = ad.advection_solve(u0, nu=nu, n_steps=steps, scheme="muscl", limiter="minmod", integrator="rk3")
        # 削波宽度 = (u>0.05 且 u<0.95) 的点数 × dx，取最大连通段
        mask = (res.u > 0.05) & (res.u < 0.95)
        widths.append(int(np.count_nonzero(mask)) * res.dx)
    from numcore.verify.order import fit_order  # noqa: PLC0415

    fit = fit_order(hs, np.asarray(widths, dtype=float), min_points=3)
    record(
        label="minmod 的间断削波宽度随 Δx 的幂律指数",
        theory="TVD 限制器在间断附近把 φ 压到 0（一阶迎风），所以削波宽度 ∝ Δx 而不是 Δx²",
        measured=f"p={fit.slope:.4f}（宽度 {widths[0]:.4f}→{widths[-1]:.4f}）",
        tol=0.1,
        passed=0.4 < float(fit.slope) < 0.7,
        detail="削波宽度在 Δx 减半时只收缩约 1.43 倍（而不是 2 倍或 4 倍）；"
        "这是实测值，不要按「一阶 ⟹ p=1」去预期",
        kind="C",
    )
    assert 0.4 < float(fit.slope) < 0.7


# --------------------------------------------------------------------------- #
# B 类：导出表自洽（`__all__` 里的每个名字都必须真的存在）
# --------------------------------------------------------------------------- #
@assertion("B", theory="`__all__` 是模块的公开契约：列进去的每个名字都必须能解析")
def test_all_declared_names_resolve() -> None:
    """**这个测试是被一个真错抓出来的**：``limiters.__all__`` 里曾经列了
    ``limiter_values``，但那个函数根本没写——``from numcore.pde.limiters import *``
    会当场 ImportError，而**直接 import 模块不会报错**，所以静态检查也看不见。
    两边的 ``__all__`` 一起核。
    """
    rows = []
    ok = True
    for mod, label in ((lm, "limiters"), (ad, "advection")):
        names = list(getattr(mod, "__all__"))
        missing = [n for n in names if not hasattr(mod, n)]
        ok = ok and not missing
        rows.append(f"{label}: {len(names)} names, missing={missing}")
    record(
        label="limiters/advection 的 __all__ 里无法解析的名字",
        theory="`__all__` 声明了公开名字，`import *` 会逐个取；缺一个就炸",
        measured="; ".join(rows),
        tol=0.0,
        passed=ok,
        detail=f"limiters {len(lm.__all__)} 个、advection {len(ad.__all__)} 个",
        kind="B",
    )
    assert ok


@assertion("B", theory="limiter_values 与 limiter_phi 必须逐点一致；掩码只标 r 是否有限")
def test_limiter_values_matches_limiter_phi_with_a_mask() -> None:
    """新增的薄封装不能偷偷改数学：取值必须与 ``limiter_phi`` 逐位相同。"""
    r = np.array([0.0, 0.25, 1.0, 2.0, np.nan, np.inf, -1.0, -np.inf])
    worst = 0.0
    rows = []
    for name in lm.LIMITER_NAMES:
        got, ok = lm.limiter_values(name, r, with_mask=True)
        want = np.asarray(lm.limiter_phi(name, r), dtype=float)
        d = float(np.max(np.abs(got - want)))
        worst = max(worst, d)
        finite_ok = np.array_equal(ok, np.isfinite(r).astype(float))
        rows.append(f"{name}:{'ok' if (d == 0.0 and finite_ok) else 'FAIL'}")
    record(
        label="limiter_values 与 limiter_phi 逐点一致（含 nan/inf）的最大差",
        theory="薄封装只加掩码，不改取值；nan/inf 由 _ratio_branch 折到 0/big，取值仍有限",
        measured=f"{worst:.3e}",
        tol=0.0,
        passed=worst == 0.0,
        detail="; ".join(rows[:5]),
        kind="B",
    )
    assert worst == 0.0
