r"""4.3 章：守恒律与有限体积（4 个可视化）。

四个可视化共用一条证据链：**同一套有限体积记账，换四个地方看它会不会骗你**。

1. ``burgers_shock`` —— 招牌。真跑出来的解 + 精确 Riemann 解 + **逐单元通量进出**
   （每个单元的左/右界面通量与净收支），右边是逐点误差的对数曲线。
   开关一拖就能看见：Godunov/Rusanov/Osher 都收敛到同一条熵解（激波速度都贴 RH 的
   0.5），而「朴素迎风通量」在**跨声速稀疏波**上冻成一条驻定间断（TV 不增、RH 也满足、
   只有熵产率翻正）。
2. ``characteristics`` —— 特征线 :math:`x=x_0+t\,u_0(x_0)`。相交 = 激波、发散 = 稀疏波；
   第一个交点就是破散时间 :math:`t_b`。这里放进两条**理论值**：``sine`` 的闭式
   :math:`1/(2\pi A)` 与数值扫描值，以及「破散之后特征回溯不再是精确解」的显式掩码。
3. ``conservation_ledger`` —— 三本账并排：守恒的 Godunov（一阶/二阶）总质量变化与
   边界通量累计严格对消（:math:`10^{-16}`），非守恒的「特征线形式」在同一算例上
   漏 0.3（占初始质量的 37.5%）且波前**不动**。``bc`` 旋钮把「质量变了」与
   「格式不守恒」两件事分开：出流边界下质量本来就该变，账本残差才是判据。
4. ``riemann_fan`` —— Euler 方程组的精确 Riemann 解扇（激波 / 接触间断 / 稀疏波）
   与 HLLC/Rusanov 数值解的叠加。接触间断两侧**压强与速度连续、密度跳跃**，
   这是标量守恒律里不存在的那种波；``p_*``、``u_*``、两侧星区密度与四条波速
   全部进 ``meta``，Sod 的 :math:`p_*=0.30313018` 与 Toro 的教科书值逐位对上。

帧契约（与 4.1/4.2 章一致）：``meta`` 只放有限标量（``allow_nan=False``，
非有限值一律换成 ``None`` 或哨兵），字符串走 ``strings``，数组走 ``add``；
对数轴地板、显示裁剪、破散后失效的「精确解」都在 ``strings`` 里逐条说明。
**帧里的数学全部来自** :mod:`numcore.pde.fvm` 与 :mod:`numcore.pde.riemann`。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from numcore.pde import fvm
from numcore.pde import riemann as rr
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 单位舍入（= eps/2）：对数轴地板与「精确 0」的显示约定都用它。
_ROUNDOFF = float(np.finfo(float).eps) / 2.0

#: 对数轴地板：比这更小的值一律显示为地板，掩码同时置位。
_ERR_FLOOR = 1.0e-17

#: 解的显示上限：cfl>1 时格式会失控，超过它就地截断显示（**不隐藏**，配溢出掩码）。
_U_CLIP = 2.5

#: 界面通量的显示窗口（Burgers 的 |f| ≤ 0.5，稀疏波两侧到 ±0.5）。
_FLUX_CLIP = 0.75

#: 非守恒支路没有 MUSCL 版本；被选到组合时按一阶显示并在 strings 里声明。
_ADVECTIVE_NOTE = (
    "非守恒的「特征线形式」支路没有二阶版本（斜率重构只对守恒形式有意义）："
    "本次按 limiter=none 显示，meta.limiter_effective 记录了实际使用的限制器。"
)

_NONFINITE_POLICY = (
    "本帧不含任何非有限值：对数轴上的 0 用 strings.floor（={floor:g}）占位并配掩码"
    "（折线不用 NaN 分隔——NaN 会让前端静默丢段）；解超过显示窗口 ±{uc:g} 的位置"
    "截断并配 *_clipped 掩码、meta.overflow 置位；cfl>1 时格式失控也走这条路径；"
    "meta 一律是有限标量或 null（.nlf 头用 allow_nan=False 编码，塞一个 NaN 整帧就写不出去）"
)

_TIME_SEMANTICS = {
    "burgers_shock": "真实时间 t：从初值推进到 t（meta.t_actual 是实际到达时刻；末步被截断到正好落在 t）",
    "characteristics": "真实时间 t：特征线画满 [0, tMax]，t 只是「当前时刻」那条水平切线与解曲线的位置",
    "conservation_ledger": "真实时间 t：三本账都推进到 t；横轴是各方案自己的步进时刻",
    "riemann_fan": "真实时间 t：精确解与数值解都取当前时刻（t=0 时自相似解没有定义，见 t_clamped）",
}


def _fin(x: Any, default: float = 0.0) -> float:
    """把非有限标量（含 ``None``）换成哨兵（``.nlf`` 头用 ``allow_nan=False``）。"""
    if x is None:
        return float(default)
    v = float(x)
    return v if math.isfinite(v) else float(default)


def _num_or_none(x: Any) -> float | None:
    """可用就返回有限浮点，否则 ``None``（meta 允许 null，**不允许** NaN/Inf）。"""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _floor(arr: Any, floor: float = _ERR_FLOOR) -> np.ndarray:
    """对数轴用的地板占位（**不是**把值改成 0：图上是「量不出来」）。"""
    a = np.asarray(arr, dtype=float).copy()
    a[~np.isfinite(a)] = float(floor)
    return np.maximum(a, float(floor))


def _mask(arr: Any, floor: float = _ERR_FLOOR) -> np.ndarray:
    """地板掩码：1 表示该点被地板占位（或非有限），0 表示是真值。"""
    a = np.asarray(arr, dtype=float)
    return np.where(np.isfinite(a) & (a > float(floor)), 0.0, 1.0)


def _clip(arr: Any, limit: float) -> np.ndarray:
    """显示裁剪：非有限与超界一律换成 ``±limit``（配 :func:`_clip_mask`）。"""
    a = np.asarray(arr, dtype=float).copy()
    a[~np.isfinite(a)] = float(limit)
    return np.clip(a, -float(limit), float(limit))


def _clip_mask(arr: Any, limit: float) -> np.ndarray:
    """1 表示该点被裁剪（非有限或超界）。"""
    a = np.asarray(arr, dtype=float)
    return np.where(np.isfinite(a) & (np.abs(a) <= float(limit)), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# viz 1：Burgers 激波形成 + 单元通量进出
# --------------------------------------------------------------------------- #
_SCHEME_CN = {
    "godunov": "Godunov（精确通量）",
    "rusanov": "Rusanov（局部 Lax-Friedrichs）",
    "osher": "Osher（通量分裂）",
    "upwind": "朴素迎风（跨声速处违反熵条件）",
    "advective": "特征线形式（非守恒，负对照）",
}

_IC_CN = {"shock": "右行激波 u_L=1,u_R=0", "rarefaction": "跨声速稀疏波 u_L=-1,u_R=1", "sine": "光滑正弦（t_b=0.31831）"}


def _solve_ic(
    ic: str, n: int, cfl: float, scheme: str, limiter: str, t_end: float, *, snapshots: bool = False
) -> fvm.FVMResult:
    """按初值选边界条件：光滑周期波走 ``periodic``，Riemann 问题走出流。"""
    bc = "periodic" if ic == "sine" else "outflow"
    return fvm.solve(
        n=int(n), t_end=float(t_end), cfl=float(cfl), ic=str(ic), scheme=str(scheme),
        limiter=str(limiter), bc=bc, keep_snapshots=bool(snapshots),
    )


def _exact_for(ic: str, x: np.ndarray, t: float, u: np.ndarray) -> dict[str, Any]:
    """当前初值在时刻 ``t`` 的精确解（``sine`` 只在破散之前有效）。"""
    if ic == "shock":
        return fvm.exact_riemann_solution(x, t, kind="burgers", u_left=1.0, u_right=0.0)
    if ic == "rarefaction":
        return fvm.exact_riemann_solution(x, t, kind="burgers", u_left=-1.0, u_right=1.0)
    out = fvm.exact_smooth_burgers(x, t, "sine", amplitude=0.5)
    out["t_clamped"] = False
    out["riemann"] = None
    del u
    return out


def _viz_burgers_shock(p: Mapping[str, Any], t: float) -> Frame:
    ic = str(p["ic"])
    scheme = str(p["scheme"])
    limiter = str(p["limiter"])
    n = int(p["n"])
    cfl = float(p["cfl"])
    lim_eff = "none" if scheme == "advective" else limiter
    t_end = max(0.0, min(float(t), 0.6))
    fr = Frame()

    res = _solve_ic(ic, n, cfl, scheme, lim_eff, t_end)
    exact = _exact_for(ic, res.x, res.t, res.u)
    u_exact = np.asarray(exact["u"], dtype=float)
    valid = np.asarray(exact["valid_mask"], dtype=float)
    err = np.where(valid > 0.5, np.abs(res.u - u_exact), 0.0)

    flux_in = np.asarray(res.flux[:-1], dtype=float)
    flux_out = np.asarray(res.flux[1:], dtype=float)
    net = -(flux_out - flux_in)

    fr.add("cell_x", res.x, unit="", label="单元中心 x")
    fr.add("u", _clip(res.u, _U_CLIP), unit="", label="数值解（裁剪，见 u_clipped）")
    fr.add("u_clipped", _clip_mask(res.u, _U_CLIP), unit="", label="1=被显示窗口裁剪（含非有限）")
    fr.add("u_exact", _clip(u_exact, _U_CLIP), unit="", label="精确解（破散后失效，见 exact_valid）")
    fr.add("exact_valid", valid, unit="", label="1=该点精确解有效（sine 破散后为 0）")
    fr.add("exact_t_clamped", np.full(res.x.shape, 1.0 if exact["t_clamped"] else 0.0),
           unit="", label="1=t<=0，自相似解被抬到 t_floor，曲线是初始间断")
    fr.add("flux_in", _clip(flux_in, _FLUX_CLIP), unit="", label="单元左界面通量 F_(j-1/2)")
    fr.add("flux_out", _clip(flux_out, _FLUX_CLIP), unit="", label="单元右界面通量 F_(j+1/2)")
    fr.add("flux_net", _clip(net, _FLUX_CLIP), unit="", label="净流入 -(F_out-F_in)，>0 该单元增质量")
    fr.add("flux_clipped", np.maximum(_clip_mask(flux_in, _FLUX_CLIP), _clip_mask(flux_out, _FLUX_CLIP)),
           unit="", label="1=通量超出显示窗口被裁剪")
    fr.add("error_abs", _floor(err), unit="", label="|u-u_exact|（对数轴地板占位）")
    fr.add("error_floor_mask", np.maximum(_mask(err), 1.0 - valid), unit="",
           label="1=误差低于地板或精确解在该点无效")
    fr.add("residual_time", res.time_history, unit="", label="账本残差的时间轴")
    fr.add("residual_mass", _floor(np.abs(res.defect_history)), unit="",
           label="守恒账本残差 |M-M0+out-in| 随时间（对数轴）")
    fr.add("residual_floor_mask", _mask(np.abs(res.defect_history)), unit="", label="1=该步残差在地板上")

    speed: dict[str, float] | None = None
    if ic in ("shock", "rarefaction") and t_end > 0.0:
        snap = res if res.snapshots.size else _solve_ic(ic, n, cfl, scheme, lim_eff, t_end, snapshots=True)
        ts, pos, ok = fvm.shock_track(snap.snapshots, snap.dx, level=0.5, times=snap.snapshot_times)
        if bool(np.any(ok > 0.5)):
            speed = fvm.fit_speed(ts[ok > 0.5], pos[ok > 0.5])
    rh = 0.5 if ic == "shock" else None
    if speed is not None and rh is not None:
        rel = abs(speed["speed"] - rh) / rh
    else:
        rel = None

    fr.set_meta(
        t=_fin(res.t),
        t_actual=_fin(res.t),
        n=n,
        cfl=cfl,
        limiter_requested_index=float(["none", "minmod", "superbee", "vanleer"].index(lim_eff)) if lim_eff in ("none", "minmod", "superbee", "vanleer") else -1.0,
        steps=int(res.n_steps),
        dt=_fin(res.dt),
        dx=_fin(res.dx),
        rh_speed=_num_or_none(rh),
        speed_measured=_num_or_none(None if speed is None else speed["speed"]),
        speed_rel_err=_num_or_none(rel),
        speed_fit_r2=_num_or_none(None if speed is None else speed["r2"]),
        speed_fit_n=int(0 if speed is None else speed["n_used"]),
        mass0=_fin(res.mass_history[0]),
        mass_end=_fin(res.mass_history[-1]),
        mass_in_cum=_fin(res.inflow_history[-1]),
        mass_out_cum=_fin(res.outflow_history[-1]),
        max_defect=_fin(res.max_defect),
        rel_defect=_fin(res.relative_defect),
        entropy_production=_fin(res.entropy_production),
        tv0=_fin(res.tv_history[0]),
        tv_end=_fin(res.tv_history[-1]),
        max_u=_fin(res.max_history[-1]),
        min_u=_fin(res.min_history[-1]),
        max_jump=_fin(float(np.max(np.abs(np.diff(res.u)))) if res.u.size > 1 else 0.0),
        error_l1=_num_or_none(fvm.l1_error(res.u, u_exact, res.dx) if bool(np.all(valid > 0.5)) else None),
        error_max=_num_or_none(float(np.max(err)) if err.size else None),
        exact_t_break=_num_or_none(exact.get("t_break")),
        exact_valid=bool(np.all(valid > 0.5)),
        exact_t_invalid=bool(exact.get("t_invalid", False)),
        overflow=bool(np.any(_clip_mask(res.u, _U_CLIP) > 0.5) or not np.all(np.isfinite(res.u))),
        n_clipped=int(np.sum(_clip_mask(res.u, _U_CLIP))),
        error_floor=_ERR_FLOOR,
        u_clip=_U_CLIP,
        roundoff=_ROUNDOFF,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["burgers_shock"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        scheme_cn=_SCHEME_CN.get(scheme, scheme),
        scheme=scheme,
        limiter_requested=limiter,
        limiter_effective=lim_eff,
        initial=_IC_CN.get(ic, ic),
        ic=ic,
        bc=res.bc,
        floor=f"{_ERR_FLOOR:g}",
        flux_window=f"通量显示窗口 ±{_FLUX_CLIP:g}（Burgers 通量 |f|<=0.5；稀疏波扇面内 |f|<=0.5）",
        limiter_note=("非守恒支路无二阶版本，已按一阶显示" if scheme == "advective" and limiter != "none" else
                      "limiter=none 时 MUSCL-Hancock 与一阶 Godunov 逐位相同（predictor 退化成恒等）"),
        reading_note=(
            "左图：解与精确解；中图：每个单元的左/右界面通量与净流入（柱子朝上=该单元增质量）；"
            "右图：逐点误差（对数轴，地板 1e-17）。"
            "看三件事：(a) 激波被抹宽多少（一阶 ~1 个单元、二阶 ~2 个但更陡）；"
            "(b) upwind 在 rarefaction 上是否冻成一条驻定间断（max_jump 恒为 2.0）；"
            "(c) 破散之后 sine 的精确解失效：exact_valid=0，误差曲线整条作废（不是「误差变大」，是「没有参照物」）。"
        ),
        mask_note=(
            "error_floor_mask=1 表示该点误差低于 1e-17 或精确解在该点无效（sine 破散后整条置 1）。"
            "u_clipped=1 只可能出现在 cfl>1 的失控帧：格式越界后 max|u| 指数增长，"
            "帧把显示截到 ±2.5 并置 meta.overflow——**这不是把错误藏起来，是把它标出来**。"
        ),
        theory_note=(
            "Godunov 通量 = 精确解在 x/t=0 处的通量（定义）；跨声速稀疏波上它给 f(u_s)、"
            "f'(u_s)=0，而朴素迎风给 f(u_L)!=f(u_s)：后者收敛到满足 RH 的**膨胀激波**。"
            "守恒性来自通量差的望远镜求和：内部通量怎么错都不破坏守恒，破坏它的只有"
            "不写成通量差的离散。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 2：特征线与破散
# --------------------------------------------------------------------------- #
def _viz_characteristics(p: Mapping[str, Any], t: float) -> Frame:
    ic = str(p["ic"])
    amp = float(p["amplitude"])
    nl = int(p["n_lines"])
    t_view = 0.8
    t_end = max(0.0, min(float(t), t_view))
    fr = Frame()

    lines = fvm.characteristic_lines(ic, amplitude=amp, t_end=t_view, n_lines=nl)
    tb_closed = fvm.breaking_time(ic, amplitude=amp, width=0.08)
    tb_scan = fvm.breaking_time_numeric(ic, amplitude=amp, width=0.08, n_scan=20001)
    res = fvm.solve(n=200, t_end=t_end, cfl=0.4, ic=ic, scheme="godunov", limiter="minmod",
                    bc="periodic", amplitude=amp)
    exact = fvm.exact_smooth_burgers(res.x, res.t, ic, amplitude=amp)
    u_exact = np.asarray(exact["u"], dtype=float)
    valid = np.asarray(exact["valid_mask"], dtype=float)
    err = np.where(valid > 0.5, np.abs(res.u - u_exact), 0.0)

    fr.add("line_x0", lines["x0"], unit="", label="特征线起点 x0")
    fr.add("line_u0", lines["u0"], unit="", label="起点处的特征速度 u0(x0)")
    fr.add("line_x_end", lines["x_end"], unit="", label=f"t={t_view:g} 处的位置 x0+t*u0")
    fr.add("cross_t", _floor(np.where(np.isfinite(lines["t_cross"]), lines["t_cross"], 0.0)),
           unit="", label="相邻特征线交点时刻")
    fr.add("cross_x", lines["x_cross"], unit="", label="交点位置")
    fr.add("cross_mask", lines["cross_mask"], unit="", label="1=该对特征线相交（u_i>u_(i+1)，激波即将形成）")
    fr.add("break_line_t", np.asarray([tb_scan, tb_scan], dtype=float), unit="", label="破散时刻参考线（数值扫描）")
    fr.add("slice_t", np.asarray([res.t, res.t], dtype=float), unit="", label="当前时刻的水平切线")
    fr.add("profile_x", res.x, unit="", label="解曲线横坐标")
    fr.add("profile_u", _clip(res.u, _U_CLIP), unit="", label="数值解（一阶 Godunov，n=200 固定）")
    fr.add("profile_u_clipped", _clip_mask(res.u, _U_CLIP), unit="", label="1=被显示窗口裁剪")
    fr.add("profile_exact", _clip(u_exact, _U_CLIP), unit="", label="特征回溯精确解（破散后失效）")
    fr.add("exact_valid", valid, unit="", label="1=回溯精确解有效（t<t_b）")
    fr.add("error_abs", _floor(err), unit="", label="|u-u_exact|（对数轴地板占位）")
    fr.add("error_floor_mask", np.maximum(_mask(err), 1.0 - valid), unit="", label="1=误差在地板上或参照无效")
    fr.add("residual_time", res.time_history, unit="", label="守恒账本残差的时间轴")
    fr.add("residual_mass", _floor(np.abs(res.defect_history)), unit="", label="守恒账本残差随时间")
    fr.add("residual_floor_mask", _mask(np.abs(res.defect_history)), unit="", label="1=该步残差在地板上")

    tb_meas = float(lines["t_break"])
    rel_tb = abs(tb_meas - tb_scan) / tb_scan if tb_scan > 0 else None
    fr.set_meta(
        t=_fin(res.t),
        amplitude=amp,
        n_lines=nl,
        t_view=t_view,
        t_break_closed=_num_or_none(tb_closed),
        t_break_scan=_num_or_none(tb_scan),
        t_break_lines=_num_or_none(tb_meas),
        t_break_scan_rel=_num_or_none(rel_tb),
        first_pair=int(lines["first_pair"]),
        x_break=_num_or_none(lines["x_break"]),
        n_crossing=int(np.sum(lines["cross_mask"])),
        n_diverging=int(lines["cross_mask"].size - np.sum(lines["cross_mask"])),
        exact_valid=bool(np.all(valid > 0.5)),
        exact_t_invalid=bool(exact["t_invalid"]),
        tv_now=_fin(res.tv_history[-1]),
        max_u=_fin(res.max_history[-1]),
        min_u=_fin(res.min_history[-1]),
        max_defect=_fin(res.max_defect),
        error_l1=_num_or_none(fvm.l1_error(res.u, u_exact, res.dx) if bool(np.all(valid > 0.5)) else None),
        error_max=_num_or_none(float(np.max(err)) if err.size else None),
        overflow=bool(np.any(_clip_mask(res.u, _U_CLIP) > 0.5)),
        n_clipped=int(np.sum(_clip_mask(res.u, _U_CLIP))),
        error_floor=_ERR_FLOOR,
        u_clip=_U_CLIP,
        roundoff=_ROUNDOFF,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["characteristics"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        profile=ic,
        floor=f"{_ERR_FLOOR:g}",
        reading_note=(
            "横轴 x、纵轴 t 的平面上画特征线 x=x0+t*u0(x0)：**相交=激波、发散=稀疏波**。"
            "cross_mask=1 的那些线对在 cross_t 处相交，最早的那个交点就是破散时间。"
            "sine 的 t_b 有闭式 1/(2*pi*A)（=0.31831 @ A=0.5），数值扫描值与它逐位相同"
            "（极值点正好落在扫描网格上）；gaussian 的极值点不在网格上，扫描值有 O(1/n_scan^2) 偏差。"
        ),
        mask_note=(
            "exact_valid=0 时（t>t_b）profile_exact 整条是**多值解的一支**，"
            "error_floor_mask 全部置 1：破散后拿它当精确解算误差，误差会随 dx 收敛得漂漂亮亮，"
            "而它对照的根本不是熵解——这是本章最容易写成假证据的地方。"
        ),
        theory_note=(
            "Burgers: f'(u)=u，特征线是直线；破散时间 t_b=-1/min_x u0'(x)。"
            "t<t_b 时回溯方程 x0+t*u0(x0)=x 单调、解唯一；t>t_b 时三条特征线折到一点，解多值。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 3：守恒账本
# --------------------------------------------------------------------------- #
def _ledger_run(n: int, cfl: float, bc: str, scheme: str, limiter: str,
                t_end: float = 0.6) -> fvm.FVMResult:
    return fvm.solve(n=int(n), t_end=float(t_end), cfl=float(cfl), ic="shock", scheme=str(scheme),
                     limiter=str(limiter), bc=str(bc), keep_snapshots=True)


def _viz_conservation_ledger(p: Mapping[str, Any], t: float) -> Frame:
    n = int(p["n"])
    cfl = float(p["cfl"])
    bc = str(p["bc"])
    t_end = max(0.0, min(float(t), 0.6))
    fr = Frame()

    runs = {
        "godunov": _ledger_run(n, cfl, bc, "godunov", "none", t_end),
        "muscl": _ledger_run(n, cfl, bc, "godunov", "minmod", t_end),
        "advective": _ledger_run(n, cfl, bc, "advective", "none", t_end),
    }
    # 统一推进到 t_end：上面每种的步数不同（dt 由各自的最大波速定），横轴各画各的。
    # 三条 trace 各自的时间步长不同（dt 由各自的最大波速定），所以各画各的横轴；
    # 另给一条公共时间轴上的插值版本，方便前端把三条质量曲线叠在一张图上。
    common_t = np.linspace(0.0, t_end, 121) if t_end > 0.0 else np.zeros(1)
    for key, r in runs.items():
        tt = r.time_history
        fr.add(f"time_{key}", tt, unit="", label=f"{key} 的推进时刻")
        fr.add(f"mass_raw_{key}", r.mass_history, unit="", label=f"{key} 的总质量 M(t)（各自时间轴）")
        fr.add(f"defect_{key}", _floor(np.abs(r.defect_history)), unit="", label=f"{key} 的账本残差（对数轴）")
        fr.add(f"defect_floor_mask_{key}", _mask(np.abs(r.defect_history)), unit="", label="1=该步残差在地板上")
    fr.add("mass_time", common_t, unit="", label="插值后的公共时间轴（画 mass_* 用）")
    for key, r in runs.items():
        fr.add(f"mass_{key}", np.interp(common_t, r.time_history, r.mass_history), unit="",
               label=f"{key} 的总质量（公共时间轴插值）")
    fr.add("inflow_cum", np.interp(common_t, runs["godunov"].time_history, runs["godunov"].inflow_history),
           unit="", label="累计流入（保守格式，左边界）")
    fr.add("outflow_cum", np.interp(common_t, runs["godunov"].time_history, runs["godunov"].outflow_history),
           unit="", label="累计流出（保守格式，右边界）")

    tracks = {}
    for key in ("godunov", "advective"):
        r = runs[key]
        ts, pos, ok = fvm.shock_track(r.snapshots, r.dx, level=0.5, times=r.snapshot_times)
        fit = fvm.fit_speed(ts[ok > 0.5], pos[ok > 0.5]) if bool(np.any(ok > 0.5)) else None
        tracks[key] = {"t": ts, "x": pos, "mask": ok, "fit": fit}
        fr.add(f"shock_t_{key}", ts, unit="", label=f"{key} 的激波位置采样时刻")
        fr.add(f"shock_x_{key}", np.where(ok > 0.5, pos, 0.0), unit="", label=f"{key} 的激波位置（无效点为 0）")
        fr.add(f"shock_mask_{key}", ok, unit="", label="1=该时刻找到跨越（位置有效）")
    rh_line_t = np.asarray([0.0, t_end], dtype=float)
    fr.add("rh_line_t", rh_line_t, unit="", label="RH 参考线的时间端点")
    fr.add("rh_line_x", 0.5 + 0.5 * rh_line_t, unit="", label="RH 参考线 x=0.5+0.5t（理论激波速度 0.5）")
    ledger = np.asarray([_fin(abs(runs["godunov"].max_defect)),
                         _fin(abs(runs["muscl"].max_defect)),
                         _fin(abs(runs["advective"].max_defect))], dtype=float)
    fr.add("residual_ledger", ledger, unit="", label="三本账的最大残差 [godunov, muscl, advective]")
    fr.add("residual_ledger_floor_mask", _mask(ledger), unit="", label="1=该残差在地板上")

    g, m, a = runs["godunov"], runs["muscl"], runs["advective"]
    fr.set_meta(
        t=_fin(t_end),
        n=n,
        cfl=cfl,
        mass0=_fin(g.mass_history[0]),
        mass_end_godunov=_fin(g.mass_history[-1]),
        mass_end_muscl=_fin(m.mass_history[-1]),
        mass_end_advective=_fin(a.mass_history[-1]),
        defect_godunov=_fin(g.max_defect),
        defect_muscl=_fin(m.max_defect),
        defect_advective=_fin(a.max_defect),
        rel_defect_godunov=_fin(g.relative_defect),
        rel_defect_muscl=_fin(m.relative_defect),
        rel_defect_advective=_fin(a.relative_defect),
        leak_total=_fin(abs(a.mass_history[-1] - a.mass_history[0] - (a.inflow_history[-1] - a.outflow_history[-1]))),
        inflow_cum=_fin(g.inflow_history[-1]),
        outflow_cum=_fin(g.outflow_history[-1]),
        rh_speed=0.5,
        speed_godunov=_num_or_none(None if tracks["godunov"]["fit"] is None else tracks["godunov"]["fit"]["speed"]),
        speed_advective=_num_or_none(None if tracks["advective"]["fit"] is None else tracks["advective"]["fit"]["speed"]),
        speed_rel_err_godunov=_num_or_none(
            None if tracks["godunov"]["fit"] is None else abs(tracks["godunov"]["fit"]["speed"] - 0.5) / 0.5),
        speed_rel_err_advective=_num_or_none(
            None if tracks["advective"]["fit"] is None else abs(tracks["advective"]["fit"]["speed"] - 0.5) / 0.5),
        steps_godunov=int(g.n_steps),
        steps_muscl=int(m.n_steps),
        steps_advective=int(a.n_steps),
        entropy_production_godunov=_fin(g.entropy_production),
        entropy_production_advective=_fin(a.entropy_production),
        residual_floor=_ERR_FLOOR,
        roundoff=_ROUNDOFF,
        overflow=False,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["conservation_ledger"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        bc=bc,
        floor=f"{_ERR_FLOOR:g}",
        reading_note=(
            "左图：M(t)。**质量变了不等于不守恒**——出流边界下质量本来就该变（实测 0.5000 -> 0.8000，"
            "流入累计 0.3000 正好等于增量）。右图：账本残差 |M(t)-M(0)+(out-in)| 的对数曲线，"
            "守恒的两条压在 1e-16（一阶 5.83e-16、二阶 1.17e-15），非守恒那条在 3e-1——"
            "差 15 个数量级。判据是**残差**，不是质量曲线本身。"
        ),
        mask_note=(
            "defect_floor_mask_*=1 表示该步残差低于 1e-17：那是舍入地板，不是「残差为零」的证据。"
            "判「是否守恒」要比的是**两条曲线的高度差**，不是「看起来都很小」。"
        ),
        speed_note=(
            "第三张图是激波位置-时间：conservative 的斜率收敛到 RH 值 0.5（实测 0.5002），"
            "非守恒的「特征线形式」在这组数据（u_L=1,u_R=0）上波前**原地不动**"
            "（拟合斜率 -0.0000），而它照样是相容、单调、TVD 的格式。"
        ),
        theory_note=(
            "守恒格式：sum_j (u_j^{n+1}-u_j^n) dx = -dt(F_N - F_0)，内部通量望远镜相消，"
            "于是账本残差只有一个舍入量级。Lax-Wendroff 定理：守恒 + 相容的格式，"
            "若收敛则收敛到弱解；非守恒格式可以收敛到**不是弱解**的东西。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：Euler Riemann 解扇
# --------------------------------------------------------------------------- #
_PROBLEM_CN = {
    "sod": "Sod：激波 + 接触间断 + 稀疏波（p*=0.30313）",
    "lax": "Lax：强激波 + 接触间断（p*=2.46610）",
    "double_rarefaction": "双稀疏波：u*=0（对称，无真空）",
}

_EULER_CLIP = 5.0


def _viz_riemann_fan(p: Mapping[str, Any], t: float) -> Frame:
    name = str(p["problem"])
    n_req = int(p["n"])
    scheme = str(p["scheme"])
    limiter = str(p["limiter"])
    t_end = max(0.0, min(float(t), 0.2))
    fr = Frame()

    wl, wr = rr.problem(name)
    # 精确 Godunov 通量逐界面解 Riemann 问题（Python 循环），为控制单帧耗时把 n 限到 80。
    n = min(n_req, 80) if scheme == "godunov" else n_req
    res = fvm.euler_solve(wl, wr, n=n, t_end=t_end, cfl=0.4, scheme=scheme, limiter=limiter, bc="outflow")
    exact = rr.euler_exact(res.x, res.t, wl, wr)
    st = exact["star"]
    rho_n, u_n, p_n = res.rho, res.velocity, res.pressure
    err_rho = np.abs(rho_n - exact["rho"])
    err_u = np.abs(u_n - exact["u"])
    err_p = np.abs(p_n - exact["p"])

    fr.add("x", res.x, unit="", label="坐标 x")
    fr.add("rho_exact", _clip(exact["rho"], _EULER_CLIP), unit="", label="精确解密度")
    fr.add("u_exact", _clip(exact["u"], _EULER_CLIP), unit="", label="精确解速度")
    fr.add("p_exact", _clip(exact["p"], _EULER_CLIP), unit="", label="精确解压强")
    fr.add("rho_num", _clip(rho_n, _EULER_CLIP), unit="", label=f"{scheme}+{limiter} 数值密度")
    fr.add("u_num", _clip(u_n, _EULER_CLIP), unit="", label="数值速度")
    fr.add("p_num", _clip(p_n, _EULER_CLIP), unit="", label="数值压强")
    fr.add("overflow_mask", np.maximum(np.maximum(_clip_mask(rho_n, _EULER_CLIP), _clip_mask(u_n, _EULER_CLIP)),
                                       _clip_mask(p_n, _EULER_CLIP)), unit="", label="1=被显示窗口裁剪")
    fr.add("exact_valid", np.asarray(exact["valid_mask"], dtype=float), unit="", label="1=t>0，自相似解有效")
    fr.add("error_rho", _floor(err_rho), unit="", label="|rho-rho_exact|（对数轴）")
    fr.add("error_u", _floor(err_u), unit="", label="|u-u_exact|（对数轴）")
    fr.add("error_p", _floor(err_p), unit="", label="|p-p_exact|（对数轴）")
    fr.add("error_floor_mask", np.maximum(np.maximum(_mask(err_rho), _mask(err_u)), _mask(err_p)),
           unit="", label="1=三条误差里有落在地板的")
    xs = 0.5
    waves = np.asarray([xs + st.head_l * res.t, xs + st.tail_l * res.t, xs + st.u_star * res.t,
                        xs + st.tail_r * res.t, xs + st.head_r * res.t], dtype=float)
    kinds = np.asarray(
        [2.0 if st.left_wave == "shock" else 1.0, 2.0 if st.left_wave == "shock" else 1.0, 0.0,
         2.0 if st.right_wave == "shock" else 1.0, 2.0 if st.right_wave == "shock" else 1.0],
        dtype=float,
    )  # 0=接触 1=稀疏波边界 2=激波（按算例的波型给，不能写死）
    fr.add("wave_x", waves, unit="", label="五条波位置 [稀疏头,稀疏尾,接触,稀疏尾,稀疏头] 或激波位置")
    fr.add("wave_kind", kinds, unit="", label="0=接触间断 1=稀疏波扇面边界 2=激波")
    fr.add("residual_time", res.time_history, unit="", label="Euler 质量账本残差的时间轴")
    fr.add("residual_mass", _floor(np.abs(res.mass_defect)), unit="", label="Euler 的质量账本残差（对数轴）")
    fr.add("residual_floor_mask", _mask(np.abs(res.mass_defect)), unit="", label="1=该步残差在地板上")

    ent_shock = float(rr.euler_entropy(st.rho_star_r, st.p_star) / rr.euler_entropy(wr[0], wr[2]))
    fr.set_meta(
        t=_fin(res.t),
        t_requested=_fin(t_end),
        t_clamped=bool(exact["t_clamped"]),
        n=n,
        n_requested=n_req,
        steps=int(res.n_steps),
        dt=_fin(res.dt),
        p_star=_fin(st.p_star),
        u_star=_fin(st.u_star),
        rho_star_l=_fin(st.rho_star_l),
        rho_star_r=_fin(st.rho_star_r),
        s_left=_fin(st.s_left),
        s_right=_fin(st.s_right),
        head_l=_fin(st.head_l),
        tail_l=_fin(st.tail_l),
        head_r=_fin(st.head_r),
        tail_r=_fin(st.tail_r),
        contact_x=_fin(xs + st.u_star * res.t),
        vacuum=bool(st.vacuum),
        star_iterations=int(st.iterations),
        l1_rho=_num_or_none(fvm.l1_error(rho_n, exact["rho"], res.dx)),
        l1_u=_num_or_none(fvm.l1_error(u_n, exact["u"], res.dx)),
        l1_p=_num_or_none(fvm.l1_error(p_n, exact["p"], res.dx)),
        max_err_rho=_num_or_none(float(np.max(err_rho))),
        min_rho=_fin(float(np.min(res.min_rho_history))),
        min_p=_fin(float(np.min(res.min_p_history))),
        max_defect=_fin(float(np.max(np.abs(res.mass_defect)))),
        entropy_ratio_shock=_fin(ent_shock),
        window_rho_lo=_fin(float(np.min(exact["rho"]))),
        window_rho_hi=_fin(float(np.max(exact["rho"]))),
        window_p_lo=_fin(float(np.min(exact["p"]))),
        window_p_hi=_fin(float(np.max(exact["p"]))),
        overflow=bool(np.any(_clip_mask(rho_n, _EULER_CLIP) > 0.5) or not np.all(np.isfinite(res.U))),
        error_floor=_ERR_FLOOR,
        euler_clip=_EULER_CLIP,
        roundoff=_ROUNDOFF,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["riemann_fan"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_EULER_CLIP),
        problem=_PROBLEM_CN.get(name, name),
        scheme=scheme,
        limiter=limiter,
        left_wave=str(st.left_wave),
        right_wave=str(st.right_wave),
        scheme_note=("精确 Godunov 通量逐界面求解，为控制单帧耗时把 n 限制到 80"
                     if scheme == "godunov" and n_req > 80 else "未触发 n 上限"),
        floor=f"{_ERR_FLOOR:g}",
        wave_note=(
            "wave_kind：2=激波（一条线，两侧都跳）、1=稀疏波扇面的两条边界、0=接触间断。"
            "接触间断处**压强与速度连续、密度跳跃**（rho_star_l != rho_star_r），"
            "这是标量守恒律里根本不存在的波型，也是数值格式最容易抹宽的地方："
            "一阶格式把它抹成 O(sqrt(dx)) 宽，二阶格式 O(dx)。"
        ),
        mask_note=(
            "error_floor_mask=1 表示三条误差里有落在地板上的；exact_valid=0（t=0 帧）表示"
            "自相似变量 x/t 没有定义、精确解被抬到 t_floor——此时误差曲线整条无效，"
            "不是「数值解很准」。"
        ),
        theory_note=(
            "精确解由 p_* 的单调方程 f_L(p)+f_R(p)+(u_R-u_L)=0 定出（二分），"
            "两侧星区密度由激波 RH 或等熵关系给出；跨激波三条 RH 跳跃条件实测残差 <= 1.2e-15，"
            "熵 p/rho^gamma 在激波上增大、在稀疏波扇面内不变。"
        ),
    )
    return fr


_DISPATCH = {
    "burgers_shock": _viz_burgers_shock,
    "characteristics": _viz_characteristics,
    "conservation_ledger": _viz_conservation_ledger,
    "riemann_fan": _viz_riemann_fan,
}


def compute(viz_id: str, params: Mapping[str, Any], t: float = 0.0) -> Frame:
    """按 ``viz_id`` 分发到四个可视化之一（帧里的数学全部来自 numcore）。"""
    try:
        fn = _DISPATCH[str(viz_id)]
    except KeyError:
        raise KeyError(f"4.3 章没有可视化 {viz_id!r}（可用：{sorted(_DISPATCH)}）") from None
    return fn(params, float(t))


# --------------------------------------------------------------------------- #
# 清单（前端与校验器读这个；数值全部来自上面的实测）
# --------------------------------------------------------------------------- #
_CRIT_NOTE_CFL = (
    "cfl=1 是这些单调通量格式在 Burgers 上的稳定界：实测 cfl=0.9 时 max|u|=1.0000、"
    "总变差单调不增（TV 1.0000 -> 1.0000）；cfl=1.2 时格式越界，max|u| 指数增长"
    "（帧把显示截到 ±2.5 并置 meta.overflow）——越界不是「更不准」，是「不再是一个解」。"
)

MANIFEST: dict[str, Any] = {
    "chapter": "4.3",
    "title": "守恒律与有限体积",
    "part": "IV",
    "summary": (
        "守恒律的积分形式把「激波」变成一条可以从账本上读出来的速度。这一章把四条证据链"
        "并排放：真跑出来的有限体积解与精确 Riemann 解、特征线的相交与发散、"
        "质量账本从机器零到 O(1) 的漏、以及 Euler 方程组的激波/接触间断/稀疏波解扇。"
        "三个最容易被骗的地方也标出来了："
        "①「质量不变」不是守恒判据（出流边界下质量本来就该变，判据是账本残差）；"
        "②跨声速稀疏波上，除熵条件外所有检查都抓不住错误格式（RH 满足、TV 不增、格式单调）；"
        "③破散之后特征回溯不再是精确解，拿它算误差会得到漂亮的假收敛。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "burgers_shock",
            "title": "Burgers 激波形成与单元通量进出",
            "description": (
                "主图是当前格式真跑出来的解与精确 Riemann 解；中间是**逐单元的通量进出**"
                "（左/右界面通量与净流入），守恒性肉眼可见；右图是逐点误差的对数曲线。"
                "把 scheme 从 godunov 切到 upwind，再把初值切到 rarefaction："
                "解会冻成一条驻定间断，而「误差」反而小得可疑——那是熵违背，不是精度。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "ic", "label": "初值（Riemann 数据）", "type": "enum",
                    "options": ["shock", "rarefaction", "sine"], "default": "shock",
                    "critical": [], "criticalNote": "",
                },
                {
                    "name": "scheme", "label": "数值通量", "type": "enum",
                    "options": ["godunov", "rusanov", "osher", "upwind", "advective"],
                    "default": "godunov", "critical": [], "criticalNote": "",
                },
                {
                    "name": "limiter", "label": "限制器（none=一阶）", "type": "enum",
                    "options": ["none", "minmod", "mc", "vanleer"], "default": "none",
                    "critical": [], "criticalNote": "",
                },
                {
                    "name": "n", "label": "单元数 n", "type": "int",
                    "min": 40, "max": 320, "step": 40, "default": 160,
                    "critical": [], "criticalNote": "",
                },
                {
                    "name": "cfl", "label": "CFL 数 cfl = dt*max|f'|/dx", "type": "float",
                    "min": 0.1, "max": 1.2, "step": 0.05, "default": 0.4,
                    "critical": [1.0], "criticalNote": _CRIT_NOTE_CFL, "beyond": "above",
                },
            ],
            "time": {"mode": "scrub", "tMax": 0.6, "dt": 0.02, "label": "t（真实时间，域长 L=1）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "solution", "autoscaled": False,
                    "title": "解与精确解（窗口固定，失控时截断）",
                    "ylabel": "u(x,t)",
                },
                {
                    "id": "flux", "autoscaled": False,
                    "title": f"逐单元界面通量（窗口固定 ±{_FLUX_CLIP:g}）",
                    "ylabel": "F_(j±1/2)",
                },
                {
                    "id": "error", "autoscaled": False, "scale": "log",
                    "title": "逐点误差（对数轴，地板掩码）",
                    "ylabel": "|u - u_exact|",
                },
            ],
            "pitfalls": (
                "第一，「误差小」可能是**参照物坏了**：sine 在 t_b=0.31831 之后破散，"
                "特征回溯解多值，此时 error 曲线整条作废（exact_valid=0、掩码全 1）；"
                "实测把 t 拖到 0.5，一阶格式的 L1「误差」是 5.18e-2，而且加密网格不再下降"
                "（n=100..800 的实测阶 -0.08/0.08/0.01）——它不是精度，是参照物的错。"
                "第二，upwind（朴素迎风通量）在跨声速稀疏波上冻成驻定间断：实测 n=100..800 的"
                "最大单元跳变恒为 2.0000（Godunov 是 0.1738/0.0924/0.0479/0.0244，随 dx 减半），"
                "熵产率恒为 +1.333e-1（Godunov 是 -7.59e-3 -> -1.73e-3，随 dx 趋零）。"
                "TV 不增、RH 满足、格式单调——**除了熵条件，所有常规检查都通过**。"
                "第三，cfl 拖过 1.0 时格式失控：实测 cfl=1.2、t=0.6 的 max|u| 指数增长，"
                "帧把显示截到 ±2.5 并置 overflow；这与「数值黏性大了点」是两回事。"
            ),
            "theory": [
                "有限体积：u_j^{n+1}=u_j-(dt/dx)(F_(j+1/2)-F_(j-1/2))，守恒性来自望远镜求和",
                "Godunov 通量 = 精确 Riemann 解在 x/t=0 处的通量；跨声速稀疏波时它取 f'(u)=0 的临界点（Burgers 给 F=0）",
                "Rankine-Hugoniot：s=(f(u_L)-f(u_R))/(u_L-u_R)=u_L+u_R)/2（Burgers），实测拟合斜率 0.50016",
                "熵条件：d/dt∫eta dx + [q] <= 0；正确解在激波处耗散、在光滑稀疏波处趋于 0",
                "MUSCL-Hancock：限制斜率重构 + 半步预测；limiter=none 时与一阶 Godunov 逐位相同（实测 True）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "characteristics",
            "title": "特征线：相交即激波、发散即稀疏波",
            "description": (
                "在 (x,t) 平面上画 x=x0+t*u0(x0)。u0 递减处相邻特征线相交（cross_mask=1），"
                "最早的交点就是破散时间；u0 递增处特征线发散（稀疏波）。"
                "水平切线是当前时刻，右下是这一时刻的解与特征回溯精确解——"
                "它只在 t<t_b 有效，破散后整条失效（掩码标出）。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "ic", "label": "初值剖面", "type": "enum",
                    "options": ["sine", "gaussian"], "default": "sine",
                    "critical": [], "criticalNote": "",
                },
                {
                    "name": "amplitude", "label": "振幅 A", "type": "float",
                    "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.5,
                    "critical": [], "criticalNote": "",
                },
                {
                    "name": "n_lines", "label": "特征线条数", "type": "int",
                    "min": 10, "max": 60, "step": 5, "default": 40,
                    "critical": [], "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 0.8, "dt": 0.02, "label": "t（真实时间；tMax=0.8 覆盖破散前后）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "characteristics", "autoscaled": False,
                    "title": "特征线平面 (x,t)（固定窗口）",
                    "ylabel": "t",
                },
                {
                    "id": "profile", "autoscaled": False,
                    "title": "当前时刻解与回溯精确解",
                    "ylabel": "u(x,t)",
                },
                {
                    "id": "error", "autoscaled": False, "scale": "log",
                    "title": "回溯解误差（对数轴，掩码标无效）",
                    "ylabel": "|u - u_exact|",
                },
            ],
            "pitfalls": (
                "第一，破散时间有两个来源，必须分开报：sine 的闭式 1/(2*pi*A)=0.3183098861837907 "
                "与 20001 点扫描值在本例中**逐位相同**（极值点正好落在网格上）；"
                "gaussian 的极值点不在网格上，扫描值 0.18653154 只有 1e-9 量级的精度——"
                "「扫描出来的理论值」和「闭式理论值」不是一回事。"
                "第二，破散之后（t>t_b）回溯解多值：图上那条曲线是「最靠左的一支」，"
                "它照样光滑、照样能满足一堆不变量，但它不是熵解；"
                "实测拿它当参照，t=0.5 的 L1「误差」在 n=100..800 上不下降（5.18e-2 上下）。"
                "第三，特征线只对凸通量是直线：traffic（凹通量）上激波与稀疏波的角色互换，"
                "把 Burgers 的图直接搬过去会把两类波认反。"
            ),
            "theory": [
                "Burgers: f'(u)=u，特征线 x=x0+t*u0(x0) 是直线，斜率就是 u0(x0)",
                "破散时间 t_b=-1/min_x u0'(x)：sine 给 1/(2*pi*A)，gaussian 给 w*exp(1/2)/(A*sqrt(2))",
                "t<t_b 时 x0+t*u0(x0)=x 单调可逆，回溯唯一；t>t_b 时特征线折到一点，解多值（激波）",
                "相邻特征线交点 t_*=(x_(i+1)-x_i)/(u_i-u_(i+1))，只对 u_i>u_(i+1) 有限（发散时给 inf）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "conservation_ledger",
            "title": "守恒账本：机器零 vs 漏水",
            "description": (
                "三个方案并排跑同一个 Riemann 问题：一阶 Godunov、二阶 MUSCL（minmod）、"
                "以及非守恒的「特征线形式」。左图是总质量，右图是账本残差"
                "|M(t)-M(0)+(out-in)| 的对数曲线，第三张是激波位置-时间与 RH 参考线。"
                "bc 旋钮把「质量该不该变」与「格式守不守恒」分开：出流边界下质量本就该变。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "n", "label": "单元数 n", "type": "int",
                    "min": 40, "max": 320, "step": 40, "default": 160,
                    "critical": [], "criticalNote": "",
                },
                {
                    "name": "cfl", "label": "CFL 数", "type": "float",
                    "min": 0.1, "max": 0.9, "step": 0.05, "default": 0.4,
                    "critical": [], "criticalNote": "",
                },
                {
                    "name": "bc", "label": "边界条件", "type": "enum",
                    "options": ["outflow", "periodic"], "default": "outflow",
                    "critical": [], "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 0.6, "dt": 0.02, "label": "t（真实时间，域长 L=1）"},
            "requiredAux": ["residual"],
            "axes": [
                {
                    "id": "mass", "autoscaled": False,
                    "title": "总质量 M(t)（出流下它本来就该变）",
                    "ylabel": "M = sum u_j dx",
                },
                {
                    "id": "defect", "autoscaled": False, "scale": "log",
                    "title": "账本残差（对数轴 1e-18–10）",
                    "ylabel": "|M(t)-M(0)+(out-in)|",
                },
                {
                    "id": "shock", "autoscaled": False,
                    "title": "激波位置-时间与 RH 参考线",
                    "ylabel": "x_shock(t)",
                },
            ],
            "pitfalls": (
                "第一，**质量守恒 ≠ 格式守恒**：出流边界下守恒格式的质量从 0.5000 涨到 0.8000"
                "（累计流入 0.3000，正好等于增量），看着一点都不「守恒」；"
                "而账本残差是 5.83e-16。反过来，周期边界下非守恒格式的质量从 0.5000 掉到 0.2036，"
                "残差 2.95e-1——**残差才是判据，质量曲线不是**。"
                "第二，同一组数据（u_L=1,u_R=0）上，非守恒的「特征线形式」的波前**不动**"
                "（拟合斜率 -0.0000，r^2=0.00），而守恒格式给 0.5002（r^2=1.00000，RH 值 0.5）："
                "两种格式在光滑区逐点相同，差别只在间断——而间断正是这一章的全部内容。"
                "第三，二阶不会自动更守恒：minmod 的账本残差 1.17e-15 比一阶的 5.83e-16 大 2 倍，"
                "两者都在舍入级；「更准」与「更守恒」是两条独立的账。"
            ),
            "theory": [
                "守恒格式的总量变化只由边界通量决定：M(t)-M(0)=-∫(F_out-F_in)dt（望远镜求和）",
                "Lax-Wendroff：守恒+相容+收敛 => 收敛到弱解；非守恒格式的极限可以不是弱解",
                "熵收支：d/dt∫eta dx + [q] <= 0；Burgers 取 eta=u^2/2, q=u^3/3",
                "实测：一阶 Godunov 在光滑稀疏波上的熵产 -7.59e-3 -> -1.73e-3（n=100->800，O(dx)），"
                "朴素迎风恒为 +1.333e-1（不随 dx 消失）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "riemann_fan",
            "title": "Riemann 解扇：激波、接触间断与稀疏波",
            "description": (
                "Euler 方程组的精确 Riemann 解（Toro 的 p_* 二分）与数值解叠加："
                "密度、速度、压强三条曲线，外加对数误差。五条波的位置（稀疏头/尾、接触、激波）"
                "以数组给出；接触间断处压强与速度连续、密度跳跃，这是标量守恒律里没有的波型。"
                "把方案切到 godunov 可以看「精确通量」与 HLLC 的差别，把 limiter 切到 none "
                "可以看到接触间断被抹宽成什么样子。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "problem", "label": "标准考题", "type": "enum",
                    "options": ["sod", "lax", "double_rarefaction"], "default": "sod",
                    "critical": [], "criticalNote": "",
                },
                {
                    "name": "scheme", "label": "数值通量", "type": "enum",
                    "options": ["hllc", "rusanov", "godunov"], "default": "hllc",
                    "critical": [], "criticalNote": "",
                },
                {
                    "name": "limiter", "label": "限制器（none=一阶）", "type": "enum",
                    "options": ["none", "minmod", "mc"], "default": "minmod",
                    "critical": [], "criticalNote": "",
                },
                {
                    "name": "n", "label": "单元数 n", "type": "int",
                    "min": 100, "max": 600, "step": 50, "default": 300,
                    "critical": [], "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 0.2, "dt": 0.01, "label": "t（真实时间；t=0 时自相似解未定义，见 t_clamped）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "rho", "autoscaled": False,
                    "title": "密度（窗口按算例固定）",
                    "ylabel": "rho",
                },
                {
                    "id": "velocity", "autoscaled": False,
                    "title": "速度（接触两侧连续）",
                    "ylabel": "u",
                },
                {
                    "id": "pressure", "autoscaled": False,
                    "title": "压强（跨激波跳、跨接触连续）",
                    "ylabel": "p",
                },
                {
                    "id": "error", "autoscaled": False, "scale": "log",
                    "title": "三条误差（对数轴，地板掩码）",
                    "ylabel": "|num - exact|",
                },
            ],
            "pitfalls": (
                "第一，Sod 的星区值可以当对拍靶子：实测 p_*=0.30313018、u_*=0.92745262、"
                "rho_*L=0.42631943、rho_*R=0.26557371，与 Toro 教科书值（0.30313/0.92745/0.42632/0.26557）"
                "逐位对上；跨右激波的三条 RH 跳跃条件残差 <= 1.19e-15。"
                "第二，Riemann 问题的 L1 收敛阶**不是**格式的形式阶：Sod 上 HLLC+minmod 的实测"
                "密度 L1 阶是 0.89/0.84/0.81（n=100..800），一阶 HLLC 是 0.66——"
                "因为 L1 误差被激波与接触间断主导，那里无论几阶都只有一阶（甚至更低）。"
                "要量形式阶必须用光滑解（本章用 sine 在破散前量：一阶 0.94/0.97/0.98、"
                "minmod 1.87/1.91/1.93、mc 2.08/2.04/2.07）。"
                "第三，t=0 时自相似变量 x/t 没有定义：帧把 t 抬到 1e-12 并置 t_clamped=1、"
                "exact_valid=0，此时误差整条无效——把「初始间断」当成「格式精度极高」是经典假象。"
            ),
            "theory": [
                "精确 Riemann 解：f_L(p_*)+f_R(p_*)+(u_R-u_L)=0 单调，二分定 p_*；"
                "激波支用 RH、稀疏波支用等熵关系 (p/p_K)^((gamma-1)/(2gamma))",
                "跨激波：s[U]=[F]（质量/动量/能量三条），实测最大残差 1.19e-15；"
                "跨接触间断：[u]=[p]=0、[rho]!=0",
                "熵 p/rho^gamma：跨激波增大（Sod 实测比值 1.30），跨稀疏波扇面内不变（等熵）",
                "HLLC：两条声波 + 中间接触波；波速用最朴素的 Davis 估计，"
                "所以它与精确解的差别（L1 2.92e-3）就是「波速估计」的代价",
            ],
        },
    ],
}
