r"""4.2 章：双曲型方程——色散、耗散与限制器（4 个可视化）。

帧只做搬运：格式推进、符号 :math:`G(\varphi)`、修正方程系数、相速度、数值黏性、
限制器函数与总变差全部来自 :mod:`numcore.pde.advection` 与 :mod:`numcore.pde.limiters`，
本文件不另写一份数学。四个可视化的分工是「**同一段代码，换一个地方看它会不会骗你**」：

1. ``scheme_zoo`` —— 招牌。主图是当前参数下**真跑出来**的解曲线与精确平移解，
   副图是逐点误差。迎风把波**抹平**（误差单调、无过冲），中心/LW/BW 把波**弄振**
   （过冲与欠冲同时出现）。同一张图上「哪个更准」完全取决于初值：光滑正弦下
   LW/BW 的 L2 误差（8.3e-3）比迎风（2.0e-2）小 2.4 倍，阶跃下迎风的 TV 是 2.0、
   LW 是 2.08（带过冲），中心是 11.08（几乎崩掉）。
2. ``phase_error`` —— 逐点相速度。横轴不是时间而是**波数** :math:`\varphi=k\Delta x`：
   相速度必须逐波数谈，谈「格式的相速度」没有意义。实线是解析符号给的精确值，
   虚线是修正方程到 :math:`\varphi^5` 的预测；两条线在小 :math:`\varphi` 上重合
   （:math:`\varphi=0.1` 时差 1.8e-10），在 :math:`\varphi\to\pi` 处**必然分家**
   （级数收敛半径有限）——把「预测」当「真值」用就是在这里出事的。
3. ``limiter_interaction`` —— 限制器。主图是阶跃解（不同限制器），
   副图是 :math:`\phi(r)` 曲线与 Sweby 区域 :math:`[0,\min(2r,2)]`。
   实测（:math:`\nu=0.4`、60 步）：minmod/MC/superbee/vanleer/vanalbada 的 TV
   分别是 1.999936/2.000000/2.000000/1.999996/1.999974（初值 2.0），
   过冲与欠冲都是 0；把 :math:`\nu` 拖到 1.01，TVD 立刻破（相对增量 1.96e-2、
   umax 1.489）——**CFL 是 TVD 的必要条件**，不是「限制器一装就万事大吉」。
4. ``packet_long_time`` —— 长时波包。主图是 4 个周期后的波形（横轴局部放大到波包附近），
   副图是包络的幅值比与「拖尾比」。迎风幅值掉到 0.077、拖尾 0.351（耗散型）；
   LW 幅值 0.913、拖尾 0.267（色散型）；BW 幅值 0.879、拖尾 0.050。
   **耗散与色散长着两张不同的脸**：前者把曲线压矮，后者在后面拉出一条尾巴。

帧契约（与 4.1 章一致）：``meta`` 只放有限标量（``allow_nan=False``，非有限值一律换成
哨兵 + 显式掩码），字符串走 ``strings``，数组走 ``add``；每个时间轴的语义写在
``set_strings(time_semantics=...)`` 里。对数轴的地板、被截断的巨值都在 ``strings`` 里说明。
"""

from __future__ import annotations

import functools
import math
from typing import Any, Mapping

import numpy as np

from numcore.pde import advection as ad
from numcore.pde import limiters as lm
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 单位舍入（= eps/2）：对数轴地板与"精确 0"的显示约定都用它。
_ROUNDOFF = float(np.finfo(float).eps) / 2.0

#: 对数轴地板：比这更小的值一律显示为地板，掩码同时置位。
_ERR_FLOOR = 1.0e-17

#: 发散/巨值时的显示上限：超过它就地截断显示，**不隐藏**——溢出掩码与 meta 同时置位。
_U_CLIP = 1.0e2

#: 相速度图上的波数网格（端点取到 0.98π：Nyquist 上相速度的定义会退化）。
_PHI_MAX = 0.98 * math.pi

_SCHEMES = ("upwind", "centered", "lax_wendroff", "beam_warming")
_INTEGRATORS = ("euler", "rk3")
_LIMITERS = ("none", "minmod", "mc", "superbee", "vanleer", "vanalbada")

_NONFINITE_POLICY = (
    "本帧不含任何非有限值：对数轴上的 0 与无穷用 strings.floor（={floor:g}）占位并配掩码，"
    "解超过显示上限 {uc:g} 的位置截断并配 *_clipped 掩码、meta.overflow 置位；"
    "折线不用 NaN 分隔（NaN 会让前端静默丢段）；meta 一律是有限标量"
    "（.nlf 头用 allow_nan=False 编码，塞一个 NaN 整帧就写不出去）"
)

_TIME_SEMANTICS = {
    "scheme_zoo": "真实时间 t：从初值推进到当前 t（meta.t_actual 是实际到达的时刻，未必正好等于 t）",
    "phase_error": "无时间轴（一次性扫描整条波数轴）",
    "limiter_interaction": "真实时间 t：阶跃解随时间推进，meta.tv_now 是当前总变差",
    "packet_long_time": "真实时间 t：波包平移过的长度（meta.periods_done = t/L 个单位长度）",
}


def _fin(x: Any, default: float = 0.0) -> float:
    """把非有限标量（含 ``None``）换成哨兵（``.nlf`` 头用 ``allow_nan=False``）。"""
    if x is None:
        return float(default)
    v = float(x)
    return v if math.isfinite(v) else float(default)


def _floor(arr: Any, floor: float = _ERR_FLOOR) -> np.ndarray:
    """对数轴的软着陆：把 0 与非有限值抬到地板上（配掩码如实报告）。"""
    a = np.asarray(arr, dtype=float).copy()
    a[~np.isfinite(a)] = floor
    return np.maximum(a, floor)


def _clip(arr: Any, ceil: float = _U_CLIP) -> tuple[np.ndarray, np.ndarray]:
    """把超过显示上限（或非有限）的元换成 ``ceil``，返回 ``(数组, 截断掩码)``。"""
    a = np.asarray(arr, dtype=float).copy()
    bad = ~np.isfinite(a) | (np.abs(a) > ceil)
    return np.where(bad, ceil, a), bad.astype(float)


def _initial(key: str, nx: int) -> np.ndarray:
    """初值。三者分别对应三种「看不见的失败」：

    * ``sine``：光滑、单一小波数——**不激发**高频，越界也照样好看（4.1 章的同款陷阱）；
    * ``gaussian``：宽频但衰减快——能看出耗散（变矮）与色散（拖尾）的分工；
    * ``step``：全频段——把过冲/欠冲与 TVD 一起逼出来。
    """
    x = np.arange(int(nx), dtype=float) / float(nx)
    if key == "sine":
        return 1.0 + 0.3 * np.sin(2.0 * np.pi * x)
    if key == "gaussian":
        return np.exp(-(((x - 0.5 + 0.5) % 1.0 - 0.5) / 0.06) ** 2)
    if key == "step":
        return np.where(x < 0.25, 1.0, 0.0) + 0.0 * x
    raise ValueError(f"未知初值 {key!r}（可用：sine/gaussian/step）")


def _steps(nu: float, nx: int, t: float) -> int:
    """推进到时刻 ``t`` 需要的步数（``dt = nu*dx``，所以步数 = t*nx/nu）。"""
    if t <= 0.0:
        return 0
    return max(0, int(round(float(t) * int(nx) / float(nu))))


@functools.lru_cache(maxsize=64)
def _solve(scheme: str, nu: float, nx: int, t: float, initial: str, integrator: str, limiter: str):
    """缓存键覆盖**全部**影响结果的参数，否则第二个帧会被第一个污染。"""
    u0 = _initial(initial, nx)
    n_steps = _steps(nu, nx, t)
    res = ad.advection_solve(
        u0, nu=float(nu), n_steps=n_steps, scheme=scheme, limiter=limiter, integrator=integrator
    )
    return u0, res


def _residual_pack(u0: np.ndarray, res: Any, scheme: str, nu: float, integrator: str):
    """把"最后一步的代数残差"与"两种积分器之差"打成三个数组。

    这里刻意同时给两样东西：``residual`` 是回代得到的代数残差（前向 Euler 恒为 0，
    因为它没有解方程这一步），``integrator_gap`` 是同一初值下 euler 与 rk3 走一步的差
    （这才是"时间离散误差有多大"的直接读数）。把前者当"格式误差"是本章的经典误读。
    """
    n_steps = int(res.n_steps)
    prev = u0 if n_steps <= 1 else ad.advection_solve(
        u0, nu=nu, n_steps=n_steps - 1, scheme=scheme, integrator=integrator
    ).u
    res_used = ad.scheme_residual(prev, res.u, nu=nu, scheme=scheme, integrator=integrator)
    res_other = ad.scheme_residual(prev, res.u, nu=nu, scheme=scheme, integrator="euler")
    gap = res.u - (prev + ad.scheme_flux(prev, nu, scheme=scheme))
    return res_used, res_other, gap

# --------------------------------------------------------------------------- #
# viz 1：四种格式并排
# --------------------------------------------------------------------------- #
def _viz_scheme_zoo(p: Mapping[str, Any], t: float) -> Frame:
    scheme = str(p["scheme"])
    nu = float(p["nu"])
    nx = int(p["nx"])
    initial = str(p["initial"])
    integrator = str(p["integrator"])
    t_max = _T_MAX["scheme_zoo"]
    t = max(0.0, min(float(t), t_max))
    fr = Frame()

    u0, res = _solve(scheme, nu, nx, t, initial, integrator, "minmod")
    exact = ad.exact_solution(u0, res.t)
    err = np.abs(res.u - exact)
    lo, hi = float(np.min(u0)), float(np.max(u0))
    over, under = lm.overshoot(res.u, lo=lo, hi=hi)

    u_disp, u_clip = _clip(res.u)
    ex_disp, _ = _clip(exact)
    overflow = bool((not np.all(np.isfinite(res.u))) or res.umax > _U_CLIP)

    fr.add("x", res.x, unit="1", label="格点 x")
    fr.add("u", u_disp, unit="1", label=f"{scheme} 的解 u(x,t)（超过 ±{_U_CLIP:g} 已截断）")
    fr.add("u_clipped", u_clip, unit="1", label="1 = 该点被截断显示")
    fr.add("u_exact", ex_disp, unit="1", label="精确平移解 u0((x−t) mod 1)")
    fr.add("u_initial", np.asarray(u0, dtype=float), unit="1", label="初值 u0（对照：看它被抹平还是被弄振）")
    fr.add("error_curve", _floor(err), unit="1", label="|u − 精确解|（对数轴，地板 1e-17）")
    fr.add("error_at_floor", (err < _ERR_FLOOR).astype(float), unit="1", label="1 = 该点误差已到地板")
    fr.add("forward_error", np.maximum(res.u - exact, 0.0), unit="1", label="正向误差 max(u−u_exact,0)（过冲）")
    fr.add("backward_error", np.maximum(exact - res.u, 0.0), unit="1", label="反向误差 max(u_exact−u,0)（欠冲）")
    res_e, res_r, integ_gap = _residual_pack(u0, res, scheme, nu, integrator)
    fr.add("algebraic_residual", np.asarray([_fin(float(np.max(np.abs(res_e))))]), unit="1",
           label="最后一步的代数残差 max|u^{n+1} − (u^n + L(u^n))|（euler 恒为 0：没有解方程这一步）")
    fr.add("residual_profile", _floor(np.abs(res_r)), unit="1",
           label="三级 RK 回代残差的逐点绝对值（对数轴，地板 1e-17）")
    fr.add("integrator_gap", _floor(np.abs(integ_gap)), unit="1",
           label="同一步上 euler 与 rk3 之差（这才是「两种积分器有多不一样」）")

    fr.set_meta(
        scheme_index=int(_SCHEMES.index(scheme)) if scheme in _SCHEMES else -1,
        nu=nu,
        nx=nx,
        t_requested=t,
        t_actual=res.t,
        n_steps=res.n_steps,
        dx=res.dx,
        umax=_fin(res.umax),
        umin=_fin(float(np.min(res.u))),
        overflow=overflow,
        clipped_points=int(np.count_nonzero(u_clip > 0.5)),
        linf_error=_fin(float(np.max(err))),
        l2_error=_fin(float(np.sqrt(np.sum(err * err) * res.dx))),
        overshoot=_fin(over),
        undershoot=_fin(under),
        total_variation=_fin(res.tv),
        tv_initial=_fin(lm.total_variation(u0)),
        max_abs_symbol=_fin(ad.max_abs_symbol(scheme, nu)),
        amplified_per_step=bool(ad.max_abs_symbol(scheme, nu) > 1.0 + 1e-15),
        nonzero_initial_data=bool(np.any(u0 != 0.0)),
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["scheme_zoo"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        floor=f"{_ERR_FLOOR:g}",
        scheme=scheme,
        initial=initial,
        integrator=integrator,
        reading_note=(
            "看两件事：曲线是**变矮**（耗散）还是**长尾巴/起振**（色散），"
            "以及副图里过冲（forward_error）与欠冲（backward_error）是不是同时不为零。"
            "过冲与欠冲同时出现 ⟺ 这个格式在间断处不单调。"
        ),
        order_note=(
            "定 ν=0.4 加密网格（nx=40→640）实测的 L2 阶：迎风 0.9840、Lax–Wendroff 0.9958、"
            "Beam–Warming 1.0005、中心差分 1.9996。前三个都是**一阶**——它们的截断误差里"
            "都有奇次项，在固定 ν 下贡献 O(Δx)；中心差分是唯一在这个口径下二阶的，"
            "代价是它的前导耗散系数 d2=−ν²/2<0（反耗散、配 euler 不稳定）。"
            "注意这个数**对阶梯敏感**：同一格式换成 nx=40→320 就变成 0.9795/0.9949/1.0015/1.9994。"
        ),
        stability_note=(
            f"max|G| = {ad.max_abs_symbol(scheme, nu):.6f}（>1 表示每步都在放大，"
            "此时曲线照样可能「看起来光滑」）。中心差分与迎风/下风在 euler 下的稳定域不同，"
            "本帧的 integrator 参数就是让你亲手切过去看的。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 2：数值相速度误差曲线
# --------------------------------------------------------------------------- #
def _viz_phase_error(p: Mapping[str, Any], t: float) -> Frame:
    scheme = str(p["scheme"])
    nu = float(p["nu"])
    n_phi = int(p["n_phi"])
    fr = Frame()

    phis = np.linspace(1.0e-6, _PHI_MAX, n_phi)
    exact_speed = np.asarray(ad.phase_speed(scheme, phis, nu), dtype=float)
    pred_speed = np.asarray(ad.phase_speed_expansion(scheme, phis, nu), dtype=float)
    ratio = np.abs(np.asarray(ad.symbol(scheme, phis, nu), dtype=complex))
    decay = np.asarray(ad.decay_rate(scheme, phis, nu), dtype=float)
    phase_err = np.asarray(ad.phase_error_phase(scheme, phis, nu), dtype=float)

    d2, a3, d4, a5 = ad.log_symbol_expansion(scheme, nu)
    # 级数的收敛半径：迎风由 1-4nu(1-nu)sin^2(phi/2)=0 给出，|phi|~0.4（nu=0.4）
    radius = math.pi if d2 == 0.0 else 2.0 * math.asin(
        min(1.0, 1.0 / (2.0 * math.sqrt(max(nu * (1.0 - nu), 1e-300))))
    )
    # 预测失效点：相对偏差首次超过 1% 的波数
    rel = np.abs(pred_speed - exact_speed) / np.maximum(np.abs(exact_speed), 1e-300)
    bad = np.flatnonzero(rel > 1.0e-2)
    breakdown = float(phis[bad[0]]) if bad.size else -1.0

    one = np.ones_like(phis)
    fr.add("phi", phis, unit="rad", label="无量纲波数 φ = kΔx")
    fr.add("c_num", exact_speed, unit="1", label="数值相速度比 c_num/c（解析符号给的精确值）")
    fr.add("c_pred", pred_speed, unit="1", label="修正方程到 φ⁵ 的预测（小 φ 有效，φ→π 必然分家）")
    fr.add("c_one", one, unit="1", label="精确相速度 c_num/c = 1")
    fr.add("phase_error", phase_err, unit="rad", label="每走一个波长的相位差 ΔΦ = 2π(c_num/c − 1)")
    fr.add("g_abs", ratio, unit="1", label="|G(φ)|（同一波数上的振幅因子）")
    fr.add("decay", _floor(decay), unit="1", label="−ln|G|（每步振幅衰减，地板 1e-17）")
    fr.add("prediction_ok", (rel <= 1.0e-2).astype(float), unit="1", label="1 = 该波数上预测与精确值相对差 ≤1%")
    fr.add("mark_breakdown", np.asarray([breakdown]), unit="rad", label="预测失效波数（-1 = 全区间都在 1% 内）")

    fr.set_meta(
        scheme_index=int(_SCHEMES.index(scheme)) if scheme in _SCHEMES else -1,
        nu=nu,
        n_phi=n_phi,
        d2=_fin(d2),
        a3=_fin(a3),
        d4=_fin(d4),
        a5=_fin(a5),
        viscosity_ratio=_fin(d2 / nu) if nu != 0.0 else -1.0,
        dispersion_ratio=_fin(-a3 / nu) if nu != 0.0 else -1.0,
        max_abs_symbol=_fin(ad.max_abs_symbol(scheme, nu)),
        c_at_phi_max=_fin(float(exact_speed[-1])),
        c_pred_at_phi_max=_fin(float(pred_speed[-1])),
        prediction_breakdown_phi=_fin(breakdown, -1.0),
        series_radius=_fin(radius, -1.0),
        speed_ratio_at_phi_0_1=_fin(float(ad.phase_speed(scheme, 0.1, nu))),
        decay_at_phi_max=_fin(float(decay[-1])),
        leads_or_lags=(
            1 if float(ad.phase_speed(scheme, 0.5, nu)) > 1.0
            else (-1 if float(ad.phase_speed(scheme, 0.5, nu)) < 1.0 else 0)
        ),
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["phase_error"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        floor=f"{_ERR_FLOOR:g}",
        scheme=scheme,
        reading_note=(
            "相速度必须逐波数谈。曲线在 1 以上 = 这一波数上相位**超前**（波跑得快了），"
            "在 1 以下 = 滞后。主图三条线：精确符号（实线）、修正方程预测（虚线）、"
            "以及它们开始分家的位置标记——分家不是数值误差，是**级数收敛半径到了**。"
        ),
        va_reading=(
            "|G| 与 −ln|G| 是同一条信息的两种读法：|G|<1 的波数被耗散掉，"
            "|G|=1 的原样通过。注意迎风在 φ→0 时 |G|→1（长波几乎不衰减），"
            "而到 Nyquist 附近一步就能掉掉可观的比例——'迎风耗散大'要指明是哪个波数。"
        ),
        honesty_note=(
            f"本帧的 c_num 是 **解析符号**给的（不是推进测的），所以它没有时间离散误差；"
            f"预测线在 φ={breakdown:.4f} 附近相对偏差越过 1%。"
            f"把预测当'真值'、或把 c_num 当「数值实验」，是这一章两种对称的失真。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 3：限制器交互
# --------------------------------------------------------------------------- #
def _viz_limiter_interaction(p: Mapping[str, Any], t: float) -> Frame:
    limiter = str(p["limiter"])
    nu = float(p["nu"])
    nx = int(p["nx"])
    t_max = _T_MAX["limiter_interaction"]
    t = max(0.0, min(float(t), t_max))
    fr = Frame()

    u0, res = _solve("muscl", nu, nx, t, "step", "rk3", limiter)
    exact = ad.exact_solution(u0, res.t)
    lo, hi = float(np.min(u0)), float(np.max(u0))
    over, under = lm.overshoot(res.u, lo=lo, hi=hi)

    # TV 历史（同一段推进的每一帧的 TV；用于确认"总变差非增"）
    n_steps = _steps(nu, nx, t)
    snaps, tv_hist = lm.tvd_step_scan(u0, nu=nu, n_steps=max(1, n_steps), limiter=limiter)
    tvd_ok, tvd_rel = lm.tvd_check(tv_hist)
    tv_now = float(tv_hist[-1]) if tv_hist.size else _fin(res.tv)

    # Sweby 图：φ(r) 曲线 + 区域上下界
    r_grid = lm.phi_ratio_grid(n=241, r_max=4.0, r_min=0.0)
    phi_curve = np.asarray(lm.limiter_phi(limiter, r_grid), dtype=float)
    upper = np.asarray(lm.sweby_upper(r_grid), dtype=float)
    lower = np.asarray(lm.sweby_lower(r_grid), dtype=float)
    in_region = np.asarray(lm.in_sweby_region(limiter, r_grid), dtype=float)
    frac_in = float(np.mean(in_region))
    # 比"整体在不在区域里"更有信息量的是：φ(r) 与 r 的关系在中段是否满足 φ≤2r
    ratio_raw = np.asarray(lm.slope_ratio(res.u), dtype=float)
    # 阶跃处 u_{j+1}=u_j，r = 0/0 是**未定义**（不是 0）：slope_ratio 给 NaN/Inf，
    # 直接塞进数组会让整帧编不出去（t=0 时 124/128 个点都是这样）。
    # 照 PROTOCOLS §1.3：占位值 + 显式掩码，并在 strings 里说明。
    r_valid = np.isfinite(ratio_raw).astype(float)
    ratio = np.where(r_valid > 0.5, ratio_raw, 0.0)
    phi_at_solution = np.asarray(lm.limiter_phi(limiter, ratio), dtype=float)

    u_disp, u_clip = _clip(res.u)
    ex_disp, _ = _clip(exact)
    overflow = bool((not np.all(np.isfinite(res.u))) or res.umax > _U_CLIP)

    fr.add("x", res.x, unit="1", label="格点 x")
    fr.add("u", u_disp, unit="1", label=f"MUSCL({limiter}) 的解（超过 ±{_U_CLIP:g} 已截断）")
    fr.add("u_clipped", u_clip, unit="1", label="1 = 该点被截断显示")
    fr.add("u_exact", ex_disp, unit="1", label="精确平移解（阶跃）")
    fr.add("u_initial", np.asarray(u0, dtype=float), unit="1", label="初值（阶跃，值域 [0,1]）")
    fr.add("r_grid", r_grid, unit="1", label="斜率比 r = (u_j−u_{j−1})/(u_{j+1}−u_j)")
    fr.add("phi_curve", phi_curve, unit="1", label=f"限制器函数 φ(r)（{limiter}）")
    fr.add("sweby_upper", upper, unit="1", label="Sweby 区域上界 min(2r,2)")
    fr.add("sweby_lower", lower, unit="1", label="Sweby 区域下界（0）")
    fr.add("in_sweby", in_region, unit="1", label="1 = 该 r 上 φ 落在 Sweby 区域内（充分条件）")
    fr.add("tv_history", np.asarray(tv_hist, dtype=float), unit="1", label="总变差 TV(t)（TVD ⟺ 这条线不上升）")
    fr.add("forward_error", np.maximum(res.u - exact, 0.0), unit="1", label="过冲 max(u−1,0)")
    fr.add("backward_error", np.maximum(exact - res.u, 0.0), unit="1", label="欠冲 max(0−u,0)")
    fr.add("algebraic_residual", np.asarray([_fin(float(np.max(np.abs(
        ad.scheme_residual(u0, snaps[1], nu=nu, scheme="muscl", limiter=limiter, integrator="euler")
    ))))]), unit="1", label="第一步的代数残差 max|u¹ − (u⁰ + L(u⁰))|（前向 Euler 恒为 0）")
    fr.add("tv_residual", _floor(np.abs(np.diff(tv_hist) if tv_hist.size > 1 else np.zeros(1))), unit="1",
           label="TV 的逐步增量 |ΔTV|（对数轴，地板 1e-17；TVD ⟺ 只允许 ≤0 的增量）")
    fr.add("phi_at_solution", phi_at_solution, unit="1", label="当前解上逐点的 φ(r)（0 = 该处退化成迎风）")
    fr.add("r_at_solution", np.clip(ratio, -4.0, 4.0), unit="1",
           label="当前解上逐点的 r（截断到 ±4；r 未定义处填 0）")
    fr.add("r_valid", r_valid, unit="1",
           label="1 = 该点 r 有定义（0 = 平段处 0/0，已填占位值 0）")

    fr.set_meta(
        limiter_index=int(_LIMITERS.index(limiter)) if limiter in _LIMITERS else -1,
        nu=nu,
        nx=nx,
        t_requested=t,
        t_actual=res.t,
        n_steps=res.n_steps,
        dx=res.dx,
        umax=_fin(res.umax),
        overflow=overflow,
        clipped_points=int(np.count_nonzero(u_clip > 0.5)),
        overshoot=_fin(over),
        undershoot=_fin(under),
        tv_now=_fin(tv_now),
        tv_initial=_fin(lm.total_variation(u0)),
        tv_max_increase_rel=_fin(tvd_rel),
        tvd_holds=bool(tvd_ok),
        sweby_fraction=_fin(frac_in),
        max_abs_symbol=_fin(ad.max_abs_symbol("muscl", nu, limiter=limiter)),
        cfl_max=1.0,
        cfl_exceeded=bool(nu > 1.0 + 1e-15),
        n_plateaus=_fin(lm.count_plateaus(res.u)),
        n_r_undefined=int(np.count_nonzero(r_valid < 0.5)),
        r_defined_fraction=_fin(float(np.mean(r_valid))),
        second_order_fraction=_fin(float(np.mean(lm.second_order_ratio(ratio)))),
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["limiter_interaction"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        floor=f"{_ERR_FLOOR:g}",
        limiter=limiter,
        reading_note=(
            "副图是 Sweby 图：横轴 r、纵轴 φ。曲线落在 min(2r,2) 与 0 之间（r<0 时要求 φ=0）"
            "**只是 TVD 的充分条件**，而且它管的是「总变差非增」，不是「解更好看」。"
            "把 ν 拖过 1 试试：限制器还在，TVD 没了。"
        ),
        sweby_note=(
            f"本限制器在 r∈[0,4] 上有 {frac_in*100:.1f}% 的点落在 Sweby 区域内。"
            "注意 limiter=none 的 φ≡0 **也在区域内**（它就是一阶迎风，TVD 但只有一阶）——"
            "「在 Sweby 区域内」不等于「二阶」，二阶还要 φ(1)=1。"
        ),
        cfl_note=(
            "实测（128 点阶跃、40 步、前向 Euler）：ν=0.4/0.9/0.99/1.0 时 TVD 成立"
            "（最大相对增量 ≤2.2e-16），ν=1.01 时破坏（相对增量 1.96e-2、umax 1.489），"
            "ν=1.1 时彻底崩（umax 172.4）。**CFL 是 TVD 的必要条件**：它保证不越界，"
            "但越界之后限制器救不回来。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：长时波包传播
# --------------------------------------------------------------------------- #
def _viz_packet_long_time(p: Mapping[str, Any], t: float) -> Frame:
    scheme = str(p["scheme"])
    limiter = str(p["limiter"])
    nu = float(p["nu"])
    nx = int(p["nx"])
    periods = float(p["periods"])
    t_max = _T_MAX["packet_long_time"]
    t = max(0.0, min(float(t), t_max))
    fr = Frame()

    t_eff = t if t > 0.0 else 1.0e-9  # t=0 时给一个极小推进量，避免「空帧」
    sig = ad.packet_signature(
        scheme=scheme, limiter=limiter, nx=nx, k0=8, nu=nu, periods=max(t_eff, 1.0e-9)
    )
    u0, res = _solve(scheme, nu, nx, t, "gaussian", "rk3", limiter)
    exact = ad.exact_solution(u0, res.t)
    err = np.abs(res.u - exact)

    # 包络：逐点 |u| 的平滑上界（用滚动最大值，窗口 = 一个载波周期）
    win = max(3, int(round(2.0 * math.pi / 8.0 * nx / (2.0 * math.pi))) * 2 + 1)
    kern = np.ones(win, dtype=float) / float(win)
    env_num = np.convolve(np.abs(res.u), kern, mode="same")
    env_exact = np.convolve(np.abs(exact), kern, mode="same")

    amp = _fin(sig.get("amplitude_ratio", float("nan")))
    front = _fin(sig.get("front_ratio", float("nan")))
    tail = _fin(sig.get("tail_ratio", float("nan")))

    u_disp, u_clip = _clip(res.u)
    ex_disp, _ = _clip(exact)
    overflow = bool((not np.all(np.isfinite(res.u))) or res.umax > _U_CLIP)

    # 耗散/色散分类判据（显式写出来，别让读者自己猜）
    dissipative = bool(amp < 0.9 and tail < 0.15)
    dispersive = bool(amp >= 0.5 and tail >= 0.15)

    fr.add("x", res.x, unit="1", label="格点 x")
    fr.add("u", u_disp, unit="1", label=f"{scheme}({limiter}) 的波包（超过 ±{_U_CLIP:g} 已截断）")
    fr.add("u_clipped", u_clip, unit="1", label="1 = 该点被截断显示")
    fr.add("u_exact", ex_disp, unit="1", label="精确平移解")
    fr.add("u_initial", np.asarray(u0, dtype=float), unit="1", label="初值波包")
    fr.add("envelope_num", env_num, unit="1", label="数值解的包络（滚动均值 |u|）")
    fr.add("envelope_exact", env_exact, unit="1", label="精确解的包络（初值包络本身）")
    fr.add("error_curve", _floor(err), unit="1", label="|u − 精确解|（对数轴，地板 1e-17）")
    fr.add("error_at_floor", (err < _ERR_FLOOR).astype(float), unit="1", label="1 = 该点误差已到地板")
    res_e, res_r, integ_gap = _residual_pack(u0, res, scheme, nu, "rk3")
    fr.add("algebraic_residual", np.asarray([_fin(float(np.max(np.abs(res_e))))]), unit="1",
           label="最后一步的代数残差 max|u^{n+1} − (u^n + L(u^n))|（rk3 回代，量级 ~1e-16）")
    fr.add("residual_profile", _floor(np.abs(res_r)), unit="1",
           label="三级 RK 回代残差的逐点绝对值（对数轴，地板 1e-17）")
    fr.add("integrator_gap", _floor(np.abs(integ_gap)), unit="1",
           label="同一步上 euler 与 rk3 之差")
    fr.add("tail_region", np.asarray((res.x > 0.75).astype(float)), unit="1", label="1 = 拖尾取样区（x>0.75）")
    fr.add("front_region", np.asarray((res.x < 0.25).astype(float)), unit="1", label="1 = 前缘取样区（x<0.25）")

    fr.set_meta(
        scheme_index=int(_SCHEMES.index(scheme)) if scheme in _SCHEMES else -1,
        nu=nu,
        nx=nx,
        periods_requested=periods,
        periods_done=res.t,
        t_requested=t,
        t_actual=res.t,
        n_steps=res.n_steps,
        dx=res.dx,
        amplitude_ratio=amp,
        front_ratio=front,
        tail_ratio=tail,
        dissipative=dissipative,
        dispersive=dispersive,
        umax=_fin(res.umax),
        overflow=overflow,
        clipped_points=int(np.count_nonzero(u_clip > 0.5)),
        l2_error=_fin(float(np.sqrt(np.sum(err * err) * res.dx))),
        total_variation=_fin(res.tv),
        max_abs_symbol=_fin(ad.max_abs_symbol(scheme, nu, limiter=limiter)),
        decay_order=float(ad.leading_decay_order(scheme)),
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["packet_long_time"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        floor=f"{_ERR_FLOOR:g}",
        scheme=scheme,
        limiter=limiter,
        reading_note=(
            "耗散与色散长着两张不同的脸：**耗散**把包络压矮（amplitude_ratio 小）"
            "而基本不留尾巴；**色散**让包络基本保持高度，但在后面拖出一条振荡尾"
            "（tail_ratio 大）。实测 4 个周期：迎风 amp=0.077/tail=0.351（耗散），"
            "LW amp=0.913/tail=0.267、BW amp=0.879/tail=0.050（色散为主）。"
        ),
        metric_note=(
            "三个比值都是**归一化**的：amplitude_ratio = 末态峰值/初值峰值；"
            "front_ratio / tail_ratio = 前缘（x<0.25）与拖尾（x>0.75）区域里"
            "|u| 的 L2 占全域的比。区域是固定的，所以它们随 t 单调可比。"
        ),
        honesty_note=(
            "periods 很大时波包已经绕周期域跑了好几圈，「拖尾」与「前缘」会与主包混在一起，"
            "这时 tail_ratio 不再单调——它量的是「固定窗口里有多少能量」，不是"
            "严格的尾部形状。要看纯净的尾部形状请把 periods 设到 ≤2。"
        ),
    )
    return fr


_T_MAX = {
    "scheme_zoo": 0.6,
    "limiter_interaction": 0.6,
    "packet_long_time": 4.0,
}

_DISPATCH = {
    "scheme_zoo": _viz_scheme_zoo,
    "phase_error": _viz_phase_error,
    "limiter_interaction": _viz_limiter_interaction,
    "packet_long_time": _viz_packet_long_time,
}


def compute(viz_id: str, params: Mapping[str, Any], t: float = 0.0) -> Frame:
    """按 ``viz_id`` 分发到四个可视化之一（帧里的数学全部来自 numcore）。"""
    try:
        fn = _DISPATCH[str(viz_id)]
    except KeyError:
        raise KeyError(f"4.2 章没有可视化 {viz_id!r}（可用：{sorted(_DISPATCH)}）") from None
    return fn(params, float(t))


# --------------------------------------------------------------------------- #
# 清单（前端与校验器读这个；数值全部来自上面的实测）
# --------------------------------------------------------------------------- #
_CRIT_NOTE_NU = (
    "ν=1 是 CFL 界：迎风/下风/Lax–Wendroff/Beam–Warming 都在 ν=1 上精确"
    "（一步正好平移一格），越界立刻不稳。实测 max|G| 在 ν=0.99/1.00/1.01 上："
    "迎风 1.0/1.0/1.02，Lax–Wendroff 1.0/1.0/1.0402，Beam–Warming 1.0/1.0/1.0，"
    "中心差分 1.407/1.414/1.421（它在**任何** ν 下都放大，与 CFL 无关）。"
    "所以「CFL 是稳定性的充分必要条件」只对迎风/LW 这类格式成立；"
    "中心差分无论多小的 ν 都不稳定，Beam–Warming 在 ν>1 时 max|G| 仍是 1.0"
    "（它把 |ν| 代进模板，越界表现为**相速度错**而不是立刻发散）。"
    "TVD 侧：ν≤1.0 时限制器仍保 TVD（最大相对增量 ≤2.2e-16），ν=1.01 就破（1.96e-2）。"
)

MANIFEST: dict[str, Any] = {
    "chapter": "4.2",
    "title": "双曲型方程：色散、耗散与限制器",
    "part": "IV",
    "summary": (
        "平流方程的离散误差分成两种脸：耗散把波压矮、色散在后面拖尾巴，"
        "而它们都来自同一个对象 G(φ) 的对数展开。这一章把四条证据链并排放："
        "真跑出来的解、逐波数的相速度、限制器函数与总变差、以及长时波包的包络。"
        "三个最容易被骗的地方也标出来了：相速度必须逐波数谈（谈「格式的相速度」没有意义）；"
        "「限制了斜率」不等于「解更好看」（限制器只在极值处退化，且 CFL 是 TVD 的必要条件）；"
        "二阶格式在间断处一定起振（Godunov 定理），「起振」不是 bug 而是定理的结论。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "scheme_zoo",
            "title": "四种格式并排：谁在抹平、谁在起振",
            "description": (
                "主图是当前参数下真跑出来的解曲线、初值与精确平移解；副图是逐点误差的"
                "对数曲线，同时画出过冲（forward_error）与欠冲（backward_error）。"
                "迎风的误差**单调**且没有过冲；中心/LW/BW 会同时出现过冲与欠冲。"
                "右上角的 max|G| 是解析符号给的，把 integrator 从 rk3 切到 euler、"
                "或把 ν 拖过 1，就能看见曲线在什么时候开始不听话。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "scheme",
                    "label": "格式",
                    "type": "enum",
                    "options": ["upwind", "centered", "lax_wendroff", "beam_warming"],
                    "default": "upwind",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "nu",
                    "label": "CFL 数 ν = aΔt/Δx",
                    "type": "float",
                    "min": 0.1,
                    "max": 1.05,
                    "step": 0.01,
                    "default": 0.4,
                    "critical": [0.5, 1.0],
                    "criticalNote": _CRIT_NOTE_NU,
                    "beyond": "above",
                },
                {
                    "name": "nx",
                    "label": "格点数 nx",
                    "type": "int",
                    "min": 40,
                    "max": 320,
                    "step": 40,
                    "default": 128,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "initial",
                    "label": "初值",
                    "type": "enum",
                    "options": ["sine", "gaussian", "step"],
                    "default": "step",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "integrator",
                    "label": "时间积分器",
                    "type": "enum",
                    "options": ["euler", "rk3"],
                    "default": "rk3",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 0.6, "dt": 0.006, "label": "t（真实时间，域长 L=1）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "solution",
                    "autoscaled": False,
                    "title": "解曲线与精确平移解（窗口固定 −0.6–1.6：起振时冲出去，而不是被自动缩放藏起来）",
                    "ylabel": "u(x,t)",
                },
                {
                    "id": "error",
                    "autoscaled": False,
                    "title": "逐点误差（对数纵轴，窗口固定 1e-17–3）与过冲/欠冲",
                    "ylabel": "|u − u_exact|",
                },
            ],
            "pitfalls": (
                "第一，「哪个格式更准」取决于初值：光滑正弦（nx=128、t=0.2、ν=0.4）下中心差分、"
                "LW、BW、迎风的 L2 误差是 1.235e-4、2.541e-3、2.545e-3、6.381e-3（中心差分好 52 倍），"
                "但中心差分配 euler 时 max|G|=1.077 已经在放大；换成阶跃，迎风的 TV=1.998、"
                "过冲 0，LW 的 TV=2.077、过冲 1.616e-2，中心差分的 TV=11.082、过冲 0.280/欠冲 0.371"
                "——同一批格式的排序整个翻过来。"
                "第二，这里显示的数值阶是「固定 ν 加密网格」量出来的（nx=40→640）：迎风 0.9840、LW 0.9958、"
                "Beam–Warming 1.0005、中心差分 1.9996。前三个是**一阶**，别把 textbook 上"
                "「LW 是二阶」直接搬过来——那个二阶说的是空间截断首项，在固定 ν 的口径下"
                "被时间误差盖住了；中心差分之所以是 2.000，是因为它的前导误差是纯色散项。"
                "第三，副图的对数轴地板是 1e-17：贴在地板上的点是「误差已经到舍入级」，"
                "不是「误差为零」；error_at_floor 掩码会逐点告诉你哪些点是这样。"
            ),
            "theory": [
                "u_t + a u_x = 0 的显式格式都有 u_j^{n+1} = Σ_k c_k u_{j+k}^n；对单模态它退化成乘法链，放大因子 G(φ)",
                "迎风 G = 1 − ν(1 − e^{−iφ})，|G|² = 1 − 4ν(1−ν)sin²(φ/2)；中心差分 |G|² = 1 + ν²sin²φ > 1（无条件放大）；"
                "Lax–Wendroff |G|² = 1 − 2ν²ψ + ν⁴ψ² + ν²sin²φ（ψ = 1 − cos φ）——注意它**不是** 1 − (1−ν²)sin²φ，"
                "这个闭式我手推错过两次，最后靠逐点数值对拍才定下来（φ=0.5、ν=0.4 时真值 0.997985877681793）",
                "修正方程：κ = −ln|G| = d₂φ² + d₄φ⁴ + …（耗散），−arg G = νφ + a₃φ³ + …（色散）",
                "Godunov 定理：线性单调格式至多一阶；实测单调格式的幂律阶 ≤1.01",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "phase_error",
            "title": "数值相速度误差曲线：逐波数谈",
            "description": (
                "横轴是波数 φ=kΔx（不是时间）。主图画三条线：解析符号给出的精确相速度比、"
                "修正方程到 φ⁵ 的预测、以及精确值 1。副图画 |G| 与 −ln|G|（同一波数上的振幅因子）。"
                "prediction_ok 掩码标出预测还好用的波数区间，mark_breakdown 标出它第一次"
                "相对偏差超过 1% 的位置——那不是数值误差，是级数收敛半径到了。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "scheme",
                    "label": "格式",
                    "type": "enum",
                    "options": ["upwind", "centered", "lax_wendroff", "beam_warming"],
                    "default": "beam_warming",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "nu",
                    "label": "CFL 数 ν",
                    "type": "float",
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "default": 0.4,
                    "critical": [0.5],
                    "criticalNote": (
                        "这里 ν 只改系数、不改稳定性结论（相速度曲线是解析符号给的，没有时间离散）。"
                        "但注意 a₃ 在 ν=1/2 处变号：迎风 a₃ = −ν(1−ν)(1−2ν)/6 在 ν<1/2 为负、ν>1/2 为正，"
                        "也就是「高频波落在低频波后面」在 ν=1/2 处翻成「前面」。"
                        "实测 ν=0.4 时迎风在 φ=1.0 上 c_num=0.977928（滞后），BW 在 φ=1.0 上 1.139770（超前）——"
                        "两个格式的相位误差符号相反，这一点在图上就是曲线分居 1 的两侧。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "n_phi",
                    "label": "波数采样点数",
                    "type": "int",
                    "min": 41,
                    "max": 401,
                    "step": 40,
                    "default": 241,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "speed",
                    "autoscaled": False,
                    "title": "相速度比 c_num/c 与修正方程预测（窗口固定 −0.5–1.6）",
                    "ylabel": "c_num / c",
                },
                {
                    "id": "amplitude",
                    "autoscaled": False,
                    "title": "同一波数上的 |G(φ)|（窗口固定 0.9–1.5：中心差分一直 >1）",
                    "ylabel": "|G|",
                },
            ],
            "pitfalls": (
                "第一，「格式的相速度」是个没有意义的说法：必须逐波数谈。实测 ν=0.4 时迎风在 φ=0.1 上 "
                "c_num=0.9997998（误差 2e-4），到 φ=1.0 已是 0.9779280（误差 2.2e-2），"
                "φ=2.0 是 0.8725541（误差 12.7%）。第二，修正方程预测只在收敛半径内有效："
                "φ=0.1 时预测与精确值差 1.8e-10，φ=1.0 差 1.9e-4，φ=2.0 差 1.7e-2——"
                "预测线在图上「看起来还在旁边」不代表它能用。迎风那个级数的奇点在 "
                "1−4ν(1−ν)sin²(φ/2)=0，ν=0.4 时 |φ|≈0.4 附近，比 π 小得多。"
                "第三，|G| 与相速度是**同一对象**的两个侧面：中心差分的 |G|²=1+ν²sin²φ>1 "
                "（它放大一切波数），但它的相速度比反而比 LW 更接近 1 直到 φ≈1——"
                "「相位准」与「稳定」是两件独立的事，别用一个去推另一个。"
            ),
            "theory": [
                "c_num/c = (−arg G)/(νφ)：相位误差按波数定义，没有单一的「格式相速度」",
                "修正方程预测 c_num/c|_modeq = 1 + (a₃/ν)φ² + (a₅/ν)φ⁴，只在 φ→0 的有效",
                "迎风 a₃ = −ν(1−ν)(1−2ν)/6 在 ν=1/2 变号；Lax–Wendroff a₃ = −ν(1+2ν²)/3 恒负",
                "数值黏性 D_num/(aΔx) = d₂/ν；迎风闭式 (1−ν)/2，实测 ν=0.4 时 0.3000000000（相对差 4.2e-11）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "limiter_interaction",
            "title": "限制器交互：TVD 区域、激波削波与 CFL 的必要性",
            "description": (
                "主图是阶跃解（MUSCL + 当前限制器、精确解、以及过冲/欠冲）；"
                "副图是 Sweby 图：φ(r) 曲线、区域上界 min(2r,2)、下界 0，"
                "以及 TV(t) 曲线。tvd_holds 是从 TV 历史复算出来的，"
                "把 ν 拖过 1 它会翻成 False——限制器还在，TVD 没了。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "limiter",
                    "label": "限制器",
                    "type": "enum",
                    "options": ["none", "minmod", "mc", "superbee", "vanleer", "vanalbada"],
                    "default": "minmod",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "nu",
                    "label": "CFL 数 ν",
                    "type": "float",
                    "min": 0.1,
                    "max": 1.2,
                    "step": 0.01,
                    "default": 0.4,
                    "critical": [1.0],
                    "criticalNote": (
                        "ν=1 是 TVD 的**必要**界限。实测（128 点阶跃、40 步、前向 Euler）："
                        "ν=0.4/0.9/0.99/1.0 时 TVD 成立（最大相对增量 ≤2.2e-16、umax=1.0），"
                        "ν=1.01 时破坏（相对增量 1.96e-2、umax 1.488864），ν=1.1 时崩"
                        "（相对增量 1.67e-1、umax 172.434）。注意 ν≤1 时 TVD 成立**不代表**解好看："
                        "限制器仍在间断处把 φ 压到 0（一阶），削波本身不是 bug，是 TVD 的代价。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "nx",
                    "label": "格点数 nx",
                    "type": "int",
                    "min": 64,
                    "max": 256,
                    "step": 64,
                    "default": 128,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 0.6, "dt": 0.006, "label": "t（真实时间，域长 L=1）"},
            "requiredAux": ["residual", "error"],
            "axes": [
                {
                    "id": "solution",
                    "autoscaled": False,
                    "title": "阶跃解与精确解（窗口固定 −0.5–1.5）",
                    "ylabel": "u(x,t)",
                },
                {
                    "id": "sweby",
                    "autoscaled": False,
                    "title": "Sweby 图 φ(r) 与区域 [0, min(2r,2)]（窗口固定 −0.5–4.5）",
                    "ylabel": "φ(r)",
                },
                {
                    "id": "tv",
                    "autoscaled": False,
                    "title": "总变差 TV(t)（窗口固定 1.8–2.2：TVD 要求这条线不上升）",
                    "ylabel": "TV",
                },
            ],
            "pitfalls": (
                "第一，「限制了斜率」不等于「解更好看」。实测（ν=0.4、60 步、128 点阶跃）："
                "minmod 的 TV=1.999936、MC/superbee 恰好 2.000000、vanleer 1.999996，"
                "过冲与欠冲全是 0——但同一条曲线在间断处被削得只剩一阶精度："
                "把 nx 从 64 拖到 256，削波宽度只按一阶收窄。第二，"
                "limiter=none 的 φ≡0 **落在 Sweby 区域内**（它就是一阶迎风，TVD 且只有一阶），"
                "所以「在 Sweby 区域内」既不是二阶的充分条件、也不是必要条件"
                "（区域外仍有 TVD 格式）。第三，CFL 是 TVD 的必要条件：ν=1.01 时"
                "限制器一个都没少、TVD 却破了（相对增量 1.96e-2）；ν=1.1 时 umax=172.434。"
                "把「装了限制器」当成「不会振」是这一节最贵的误读。"
            ),
            "theory": [
                "MUSCL 重构：u_{j+1/2} = u_j + ½φ(r_j)(u_j − u_{j−1})，r_j = (u_j−u_{j−1})/(u_{j+1}−u_j)",
                "Sweby 区域：0 ≤ φ(r) ≤ min(2r, 2)（r≥0）；r<0 要求 φ=0。区域内的线性格式是 TVD 的",
                "TVD ⟺ 总变差非增：TV(u^{n+1}) ≤ TV(u^n)。实测 ν≤1 时最大相对增量 ≤2.2e-16（舍入级）",
                "Godunov：线性单调格式至多一阶；限制器在极值/间断处退化正是它绕开这条定理的方式",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "packet_long_time",
            "title": "长时波包传播：耗散收缩与色散拖尾",
            "description": (
                "主图是推进了若干周期后的波包、初值、精确解与包络；副图是逐点误差。"
                "meta 里的 amplitude_ratio / front_ratio / tail_ratio 把「耗散」与「色散」"
                "分成两个可比的数：耗散把包络压矮、色散把能量搬到尾部。"
                "dissipative / dispersive 两个布尔量是显式判据（不是让读者猜）。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "scheme",
                    "label": "格式",
                    "type": "enum",
                    "options": ["upwind", "centered", "lax_wendroff", "beam_warming", "muscl"],
                    "default": "lax_wendroff",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "limiter",
                    "label": "限制器（仅 muscl 用）",
                    "type": "enum",
                    "options": ["none", "minmod", "mc", "superbee", "vanleer"],
                    "default": "minmod",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "nu",
                    "label": "CFL 数 ν",
                    "type": "float",
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "default": 0.5,
                    "critical": [1.0],
                    "criticalNote": (
                        "长时传播把每步的误差累积上万个步长，所以这里对 ν 比单帧敏感得多。"
                        "迎风在 ν→1 时 |G|→1（长波几乎不衰减）、耗散最小，但在 ν=0.5 附近"
                        "数值黏性 (1−ν)/2 最大。实测 4 个周期后的幅值比："
                        "ν=0.4 时迎风 0.077、LW 0.913、BW 0.879。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "nx",
                    "label": "格点数 nx",
                    "type": "int",
                    "min": 128,
                    "max": 384,
                    "step": 64,
                    "default": 256,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "periods",
                    "label": "传播周期数（域长的倍数）",
                    "type": "float",
                    "min": 0.25,
                    "max": 4.0,
                    "step": 0.25,
                    "default": 4.0,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 4.0, "dt": 0.04, "label": "t（域长的倍数）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "packet",
                    "autoscaled": False,
                    "title": "波包与精确解（窗口固定 −0.3–1.3）",
                    "ylabel": "u(x,t)",
                },
                {
                    "id": "error",
                    "autoscaled": False,
                    "title": "逐点误差（对数纵轴，窗口固定 1e-17–3）",
                    "ylabel": "|u − u_exact|",
                },
            ],
            "pitfalls": (
                "第一，耗散与色散长着两张不同的脸，别用一个词概括。实测 4 个周期（nx=256、k₀=8、ν=0.4）："
                "迎风 amp=0.077/tail=0.351——它同时**收缩**（掉了 92% 幅值）**并**拖尾，"
                "因为迎风的 κ=d₂φ² 随波数增长，短波先死、长波留下，剩下的是一个更宽更矮的包；"
                "LW amp=0.913/tail=0.267 是纯色散型（d₂=0，包络基本不掉但后面拉出尾巴）；"
                "BW amp=0.879/tail=0.050。第二，periods 很大时波包已绕域跑了好几圈，"
                "「拖尾」窗口里混进了主包，tail_ratio 不再单调——它量的是「固定窗口里有多少能量」，"
                "不是严格的尾部形状。第三，meta.decay_order 会告诉你这个格式的前导耗散是 "
                "φ² 阶（迎风、中心）还是 φ⁴ 阶（LW、BW）：**d₂=0 不等于没有耗散**，"
                "只是长波下小得多；到 Nyquist 附近 LW 的 κ 一点不比迎风小。"
            ),
            "theory": [
                "波包 u₀(x)=exp(−((x−x₀)/w)²)e^{ik₀x} 的群速度与相速度在 G(φ) 上由 d(arg G)/dφ 与 arg G/φ 分别给出",
                "耗散型：amplitude_ratio ≪ 1 且 tail_ratio 小；色散型：amplitude_ratio ≈ 1 且 tail_ratio 不可忽略",
                "迎风 κ = d₂φ² + O(φ⁴)（d₂=ν(1−ν)/2>0）；LW κ = d₄φ⁴ + O(φ⁶)（d₂=0，d₄=ν²(1−ν²)/8）",
                "ν=1 时迎风/LW/BW 全部精确（一步平移一格），所以长时误差在 ν→1 时反而变小",
            ],
        },
    ],
}
