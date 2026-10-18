r"""4.5 章：多重网格与快速求解（4 个可视化）。

帧只做搬运：网格层级、转移算子、平滑子、V/W 循环、FAS 与所有实测因子都来自
:mod:`numcore.pde.multigrid`，模式分解来自 1.5 章的
:func:`numcore.linalg.iterative.mode_components`，CG/MGCG 的迭代数来自
:func:`numcore.linalg.krylov.pcg`。本文件不另写一份数学。

四个可视化的分工是"同一套代码，换一个地方看它会不会骗你"：

1. ``smoothing_spectrum`` —— 招牌。主图是**逐模态**振幅随迭代步的衰减（实测 vs 理论，
   对数轴 + 地板掩码），副图是每个频率上的**一步放大率**与理论曲线、以及 1/3 那条线。
   拖动步数能看到高频几个数量级地消失、而最低模态几乎不动（n=63 时 10 步只掉 0.8%）。
2. ``vcycle_hierarchy`` —— 层级图。误差在粗细网格间的**真实搬运顺序**由
   :func:`numcore.pde.multigrid.mg_apply` 的 ``trace`` 给出（相位编码在
   ``TRACE_PHASES``），配当前周期各层的误差形态与逐周期范数。
3. ``grid_independence`` —— 网格无关性。同一张图上放四条实测曲线：
   单层平滑子的收敛因子（随 h 趋近 1）、两网格因子（恒为 1/9）、V/W 循环因子
   （≈0.19，跨 h 的均值/标准差/漂移都写在 meta 里）、以及 CG/MGCG 的迭代数。
4. ``residual_transfer`` —— 同一残差在细/粗网格上的形态。粗层可见的部分永远是**低频**的
   （θ/π=0.953 的纯高频残差，粗层可见部分的高频能量比是 0.0000），
   而它占原残差的比例只有 4e-4；平滑步数改变的只是高频的**大小**，不是可见性。

帧契约（与 1.2 / 1.5 / 4.1 章一致）：``meta`` 只放有限标量（``allow_nan=False``），
非有限值一律换成哨兵 + 显式掩码（见 ``_NONFINITE_POLICY``），字符串走 ``strings``，
数组走 ``add``；对数轴地板、被截断的量、以及"这个数为什么不是 0"都写在 ``strings`` 里。
"""

from __future__ import annotations

import functools
import math
from typing import Any, Mapping

import numpy as np

from numcore.linalg.iterative import mode_components, sine_eigenvectors_1d
from numcore.linalg.krylov import conjugate_gradient, pcg
from numcore.pde import multigrid as mg
from numcore.utils import rng
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 加权 Jacobi 在一维的最优平滑权重（高频放大率 1/3）。
OMEGA = mg.OMEGA_SMOOTH_1D

#: 对数轴地板：比它更小的幅度/损失写地板值并置掩码。
_FLOOR = 1.0e-16

#: 迭代次数上限哨兵：因子 >=1 时 iterations_for 给 inf，写进帧前必须换成有限值并置掩码。
_ITERS_CAP = 1.0e6

#: "模态存在"的判据：初值系数小于它就算这个模态没被激发（比值无定义，掩码置 0）。
_PRESENT = 1.0e-13

#: 各 viz 的时间轴上限（秒表口径：viz1/viz2 是步数/周期数，viz4 是平滑步数）。
_T_MAX = {"smoothing_spectrum": 30.0, "vcycle_hierarchy": 5.0, "residual_transfer": 6.0}

_TIME_SEMANTICS = {
    "smoothing_spectrum": "迭代步 k：对同一个初值误差做 k 次平滑（k=0 即初值），不是物理时间",
    "vcycle_hierarchy": "周期 c：做 c 个完整的 V/W 循环（c=0 即初值误差）",
    "grid_independence": "无时间轴（一次性扫完整个网格序列）",
    "residual_transfer": "预平滑步数 nu：限制之前先做 nu 次平滑（nu=0 即直接限制未平滑的残差）",
}

_NONFINITE_POLICY = (
    "本帧不含任何非有限值：对数轴上的 0 用 strings.floor（={floor:g}）占位并配 *_at_floor 掩码；"
    "「比值无定义」（该模态初值为 0、或两级因子未测）的条目写 1.0 并配 mode_present / "
    "*_defined 掩码，前端必须按掩码跳过而不是把占位值当真值画；meta 一律是有限标量"
    "（.nlf 头用 allow_nan=False 编码，塞一个 NaN 整帧就写不出去）"
)

#: 每条曲线的口径说明（防止"两条看起来一样的线其实是两个量"）。
_CURVE_NOTES = {
    "factor_measured": (
        "factor_measured 是**实测的一步范数放大率** ||T v_k||/||v_k||（把第 k 个正弦模态"
        "单独喂给平滑子）；factor_theory 是同一个模态在迭代矩阵特征意义下的模长 |lambda_k(T)|。"
        "加权 Jacobi 的 T 与 A 同特征向量，两者逐位相同；SOR 的 T 不是正规矩阵，"
        "两者会分开——分开多少是「低频不动」这句话的可信边界"
    ),
    "loss_vs_theta": (
        "loss_vs_theta = ||r - P R r||/||r||：纯模态 r 中**粗网格看不见**的那部分占比。"
        "低频段是 O(h^2) 的插值误差，接近 Nyquist 时趋于 1（一点不剩）；"
        "coarse_correction_vs_theta 是两网格校正算子的逐模态范数比（不是特征值）"
    ),
    "vcycle_trace": (
        "event_* 是 mg_apply(trace=...) 的**真实**事件序列，相位编码见 TRACE_PHASES："
        "0 进入该层 / 1 前平滑后 / 2 限制后（粗层残差范数）/ 3 粗层解完 / 4 延拓校正后 / "
        "5 后平滑后 / 6 最粗层解完。event_* 描述的是**下一个周期**从当前状态出发会发生什么"
        "（不改变当前状态）；error_profile 是当前状态本身的误差"
    ),
    "grid_independence": (
        "factor_* 全是幂迭代实测值（1.5 章同一把尺子）；iters_* 中 iters_mg_measured 是"
        "**真的数到**范数降到 1e-8 的周期数，iters_smoother 由实测因子换算"
        "（单层平滑子根本不收敛到 1e-8 那么快，直接数要 6e4 步，帧里不跑）；"
        "cg_iters / mgcg_iters 是同一右端两种方法的迭代数（不是时间）"
    ),
}


def _fin(x: Any, default: float = 0.0) -> float:
    """把非有限标量（含 ``None``）换成哨兵：``.nlf`` 头用 ``allow_nan=False``。"""
    if x is None:
        return float(default)
    v = float(x)
    return v if math.isfinite(v) else float(default)


def _floor(arr: Any, floor: float = _FLOOR) -> tuple[np.ndarray, np.ndarray]:
    """对数轴软着陆：返回 ``(抬到地板上的数组, 触底掩码)``。"""
    a = np.asarray(arr, dtype=float).copy()
    # 用 <= 而不是 <：这样"显示数组 <= 地板"与掩码**逐点等价**（可被帧测试复算），
    # 否则恰好等于地板的那个点会出现"掩码 0 但显示贴地板"的错位。
    bad = ~np.isfinite(a) | (a <= floor)
    return np.where(bad, floor, a), bad.astype(float)


def _smoother(p: Mapping[str, Any]) -> tuple[str, float]:
    """帧的 ``smoother`` 枚举 -> 内核的 ``kind``（jacobi / sor）与权重 omega。"""
    kind = str(p.get("smoother", "jacobi"))
    if kind not in ("jacobi", "sor", "gs"):
        raise ValueError(f"未知平滑子 {kind!r}")
    if kind == "gs":
        return "gs", 1.0  # GS 没有权重；带 omega 旋钮的页面（viz1/viz4）才用 sor
    return kind, float(p.get("omega", OMEGA))


def _initial_coefficients(n: int, content: str) -> np.ndarray:
    r"""初值误差在正弦基下的系数 :math:`c_k(0)`（三种内容）。

    ``mixed``：:math:`c_k=1/k`（每个模态都被激发，低频占优）；
    ``low``：只有 :math:`v_1`（"低频原地不动"的对照）；
    ``high``：只有 :math:`v_n`（"高频迅速消失"的对照）。
    """
    c = np.zeros(n, dtype=float)
    if content == "mixed":
        c = 1.0 / np.arange(1, n + 1, dtype=float)
    elif content == "low":
        c[0] = 1.0
    elif content == "high":
        c[-1] = 1.0
    else:
        raise ValueError(f"未知初值内容 {content!r}（取 mixed / low / high）")
    return c


def _norm(v: Any) -> float:
    a = np.asarray(v, dtype=float).reshape(-1)
    return float(np.sqrt(float(a @ a)))


def _sweep_error(A: np.ndarray, e0: np.ndarray, *, kind: str, omega: float, steps: int) -> list[np.ndarray]:
    """从初值误差出发做 ``steps`` 次平滑，返回包含初值在内的轨迹（右端恒为 0）。"""
    seq = [np.asarray(e0, dtype=float).copy()]
    zero = np.zeros_like(seq[0])
    for _ in range(max(0, int(steps))):
        seq.append(mg.smooth(A, zero, seq[-1], kind=kind, omega=omega, sweeps=1))
    return seq


# --------------------------------------------------------------------------- #
# viz 1：频谱衰减（平滑性质）
# --------------------------------------------------------------------------- #
def _viz_smoothing_spectrum(p: Mapping[str, Any], t: float) -> Frame:
    n = int(p["n"])
    kind, omega = _smoother(p)
    content = str(p["content"])
    steps = int(round(max(0.0, min(float(t), _T_MAX["smoothing_spectrum"]))))
    fr = Frame()

    A = mg.poisson_1d(n)
    V = sine_eigenvectors_1d(n)
    theta = np.arange(1, n + 1, dtype=float) * np.pi / (n + 1)
    c0 = _initial_coefficients(n, content)
    e0 = V @ c0
    seq = _sweep_error(A, e0, kind=kind, omega=omega, steps=steps)
    C = mode_components(np.asarray(seq, dtype=float), basis=V)  # (steps+1, n)

    # 一步放大率：把每个模态（正交归一）单独喂给平滑子，量 ||T v_k||/||v_k||
    # 用的是与轨迹同一条 code path 的迭代矩阵（smoother_matrix），不是另写一份公式
    T = mg.smoother_matrix(A, kind=kind, omega=omega)
    factor_measured = np.sqrt(np.sum((T @ V) ** 2, axis=0))
    present = np.abs(C[0]) > _PRESENT
    factor_measured = np.where(present, factor_measured, 1.0)
    factor_theory = np.asarray(mg.mode_factors_theory(n, kind=kind, omega=omega), dtype=float)

    amp0 = np.abs(C[0])
    amp_now = np.abs(C[-1])
    amp_theory = amp0 * factor_theory**steps
    amp0_disp, floor0 = _floor(amp0)
    amp_now_disp, floor_now = _floor(amp_now)
    amp_theory_disp, floor_theory = _floor(amp_theory)
    f_meas_disp, f_floor = _floor(factor_measured)
    f_theory_disp, f_theory_floor = _floor(factor_theory)

    high = theta >= np.pi / 2.0
    hi_present = np.asarray([bool(high[k] and present[k]) for k in range(n)])
    smooth_meas = float(np.max(factor_measured[hi_present])) if bool(np.any(hi_present)) else 1.0
    smooth_theory = float(np.max(factor_theory[high])) if bool(np.any(high)) else 1.0

    low_surv = float(amp_now[0] / max(amp0[0], 1e-300)) if present[0] else 1.0
    hi_idx = int(np.nonzero(present)[0][-1]) if bool(np.any(present)) else n - 1
    hi_surv = float(amp_now[hi_idx] / max(amp0[hi_idx], 1e-300))
    ratio = float(low_surv / max(hi_surv, 1e-300)) if hi_surv > 0 else float("inf")

    norms = np.asarray([_norm(v) for v in seq], dtype=float)
    norms = norms / max(norms[0], 1e-300)
    slowest = float(factor_measured[0] ** steps) if present[0] else float("nan")

    fr.add("mode_index", np.arange(1, n + 1, dtype=float), unit="1", label="模态序号 k")
    fr.add("theta_over_pi", theta / np.pi, unit="1", label="theta_k / pi")
    fr.add("amplitude_initial", amp0_disp, unit="1", label=f"|c_k(0)|（地板 {_FLOOR:g}）")
    fr.add("amplitude_measured", amp_now_disp, unit="1", label=f"第 {steps} 步实测 |c_k|（地板 {_FLOOR:g}）")
    fr.add("amplitude_theory", amp_theory_disp, unit="1", label="理论 |lambda_k|^k |c_k(0)|")
    fr.add("amplitude_at_floor", np.maximum(floor_now, floor_theory), unit="1", label="1 = 该模态已落到地板")
    fr.add("factor_measured", f_meas_disp, unit="1", label="实测一步放大率 ||T v_k||/||v_k||")
    fr.add("factor_theory", f_theory_disp, unit="1", label="理论 |lambda_k(T)|")
    fr.add("factor_at_floor", np.maximum(f_floor, f_theory_floor), unit="1", label="1 = 该频率的因子落到地板")
    fr.add("mode_present", present.astype(float), unit="1", label="1 = 初值里这个模态非零（比值的定义域）")
    fr.add("smoothing_line", np.full(n, 1.0 / 3.0), unit="1", label="最优 Jacobi 的平滑因子参考线 1/3")
    fr.add("half_band_line", np.full(n, 0.5), unit="1", label="高低频分界 theta/pi = 0.5")
    fr.add("error_norm_history", norms, unit="1", label="||e_m||/||e_0||（整段误差，含低频尾巴）")
    fr.add("error_norm_history_slowest", np.asarray([float(factor_measured[0]) ** m for m in range(steps + 1)]),
           unit="1", label="只按最低模态预测的范数下降 lambda_1^m")
    fr.set_meta(
        n=n,
        smoother_index=int(("jacobi", "gs", "sor").index(kind)),
        omega=omega,
        content_index=int(("mixed", "low", "high").index(content)),
        steps=steps,
        t_requested=float(t),
        floor=_FLOOR,
        theory_smoothing_factor=smooth_theory,
        measured_smoothing_factor=smooth_meas,
        smooth_factor_gap=abs(smooth_meas - smooth_theory),
        low_mode_factor=float(factor_measured[0]),
        high_mode_factor=float(factor_measured[hi_idx]),
        low_mode_survival=low_surv,
        high_mode_survival=hi_surv,
        low_over_high_survival=ratio,
        error_norm_ratio=float(norms[-1]),
        slowest_mode_prediction=_fin(slowest, -1.0),
        n_modes_present=int(np.count_nonzero(present)),
        n_modes_at_floor=int(np.count_nonzero(floor_now)),
    )
    fr.set_strings(
        status="ok",
        time_semantics=_TIME_SEMANTICS["smoothing_spectrum"],
        floor=f"对数轴地板 {_FLOOR:g}：更小的幅度写成地板值，amplitude_at_floor 置 1",
        placeholder_note=(
            "factor_measured / factor_theory 在 mode_present=0 的位置写 1.0（该模态初值为 0，"
            "比值无定义）——1.0 是中性占位符，不要读成「完全没衰减」"
        ),
        curve_note=_CURVE_NOTES["factor_measured"],
        smoother_note=(
            "jacobi：T = I - omega D^{-1}A，一维最优 omega=2/3（高频放大率 1/3）；"
            "sor：omega=1 即 Gauss-Seidel。选择 sor 时理论曲线给的是迭代矩阵特征值的模长"
            "（渐近速率），实测曲线仍是单模态一步范数比——两条线分开本身就是结论"
        ),
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_FLOOR),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 2：V-cycle 层级图
# --------------------------------------------------------------------------- #
def _viz_vcycle_hierarchy(p: Mapping[str, Any], t: float) -> Frame:
    levels = int(p["levels"])
    nu = int(p["nu"])
    cycle = str(p["cycle"])
    kind, omega = _smoother(p)
    content = str(p["content"])
    cycles = int(round(max(0.0, min(float(t), _T_MAX["vcycle_hierarchy"]))))
    n = 2**levels - 1
    fr = Frame()

    h = mg.mg_hierarchy(n, levels=levels, kind=kind, omega=omega, nu1=nu, nu2=nu, cycle=cycle)
    V = sine_eigenvectors_1d(n)
    e = V @ _initial_coefficients(n, content)
    e = e / max(_norm(e), 1e-300)

    hist = [1.0]
    resid = [float(_norm(h.grids[0].A @ e))]
    for _ in range(cycles):
        e = mg.mg_apply(h, e, None)
        hist.append(_norm(e))
        resid.append(float(_norm(h.grids[0].A @ e)))

    # 下一个周期的事件序列（真实顺序；不改变当前状态）
    trace: list[tuple[int, int, float, int]] = []
    mg.mg_apply(h, e, None, trace=trace)
    counter = [0.0] * h.levels
    mg.mg_apply(h, e, None, counter=counter)
    ev_order = np.arange(len(trace), dtype=float)
    ev_level = np.asarray([ev[0] for ev in trace], dtype=float)
    ev_phase = np.asarray([ev[1] for ev in trace], dtype=float)
    ev_norm = np.asarray([ev[2] for ev in trace], dtype=float)
    ev_size = np.asarray([ev[3] for ev in trace], dtype=float)
    ev_norm_disp, ev_floor = _floor(ev_norm)

    # 当前误差在各层上的形态（细 -> 粗 逐层限制），拼成一维数组 + 偏移
    profiles: list[np.ndarray] = []
    cur = e
    for lvl in range(levels):
        profiles.append(np.asarray(cur, dtype=float))
        if lvl < levels - 1:
            cur = h.restrictions[lvl] @ cur
    flat = np.concatenate(profiles) if profiles else np.zeros(1)
    sizes = np.asarray([v.size for v in profiles], dtype=float)
    offsets = np.concatenate(([0.0], np.cumsum(sizes)[:-1]))
    level_of = np.concatenate([np.full(v.size, lvl, dtype=float) for lvl, v in enumerate(profiles)])

    err_abs_disp, err_floor = _floor(np.abs(e))
    fac = mg.cycle_factor(h, iters=24, seed=45045)
    work = float(np.sum(counter))
    converging = bool(fac.factor < 1.0)

    fr.add("fine_x", np.arange(1, n + 1, dtype=float) / (n + 1), unit="1", label="细网格内点 x")
    fr.add("error_profile", e, unit="1", label=f"第 {cycles} 个周期后的误差 e(x)")
    fr.add("error_abs", err_abs_disp, unit="1", label=f"|e(x)|（对数轴，地板 {_FLOOR:g}）")
    fr.add("error_at_floor", err_floor, unit="1", label="1 = 该点误差已到地板")
    fr.add("error_norm_history", np.asarray(hist, dtype=float), unit="1", label="逐周期 ||e_c||/||e_0||")
    fr.add("residual_norm_history", np.asarray(resid, dtype=float) / max(resid[0], 1e-300), unit="1",
           label="逐周期 ||A e_c||/||A e_0||（残差口径）")
    fr.add("level_n", np.asarray([g.n for g in h.grids], dtype=float), unit="1", label="每层内点数")
    fr.add("level_h", np.asarray([g.h for g in h.grids], dtype=float), unit="1", label="每层步长 h")
    fr.add("profile_flat", flat, unit="1", label="各层误差形态（细到粗依次拼接，用 profile_offset 切）")
    fr.add("profile_offset", offsets, unit="1", label="第 l 层在 profile_flat 里的起点")
    fr.add("profile_size", sizes, unit="1", label="第 l 层的点数")
    fr.add("profile_level", level_of, unit="1", label="profile_flat 每个点所属层号")
    fr.add("event_order", ev_order, unit="1", label="事件序号（时间顺序）")
    fr.add("event_level", ev_level, unit="1", label="事件发生的层号")
    fr.add("event_phase", ev_phase, unit="1", label="相位编码 0..6（见 strings.phase_legend）")
    fr.add("event_norm", ev_norm_disp, unit="1", label=f"该层当前向量的 2-范数（地板 {_FLOOR:g}）")
    fr.add("event_at_floor", ev_floor, unit="1", label="1 = 该事件范数已到地板")
    fr.add("event_size", ev_size, unit="1", label="该层内点数")
    fr.set_meta(
        levels=levels,
        n_fine=n,
        nu=nu,
        cycle_index=int(("v", "w").index(cycle)),
        smoother_index=int(("jacobi", "gs", "sor").index(kind)),
        omega=omega,
        content_index=int(("low", "high", "mixed").index(content)),
        cycles=cycles,
        t_requested=float(t),
        floor=_FLOOR,
        factor_measured=fac.factor,
        factor_converging=converging,
        cycles_to_1e8=fac.iters_to_tol,
        work_per_cycle=work,
        work_finest_level=float(counter[0]),
        error_norm_now=float(hist[-1]),
        error_norm_ratio_last=float(hist[-1] / max(hist[-2], 1e-300)) if len(hist) > 1 else 1.0,
        n_events=len(trace),
        gamma=h.gamma,
    )
    fr.set_strings(
        status="ok" if converging else "cycle_diverging",
        time_semantics=_TIME_SEMANTICS["vcycle_hierarchy"],
        floor=f"对数轴地板 {_FLOOR:g}：event_norm / error_abs 上更小的值写成地板并置掩码",
        phase_legend="0 enter / 1 pre_smooth / 2 restrict / 3 coarse / 4 correct / 5 post_smooth / 6 coarsest",
        trace_note=_CURVE_NOTES["vcycle_trace"],
        factor_note=(
            f"factor_measured 是对**误差算子**做幂迭代的实测谱半径（24 个周期，尾部几何平均）："
            f"{fac.factor:.4f}。它不是 0——一个固定递归的 V/W 循环是收缩算子，不是精确求解器；"
            f"降到 1e-8 需要约 {fac.iters_to_tol} 个周期（实测）。"
            "若参数把平滑子调得太弱，这个数会越过 1，帧里 status 会写 cycle_diverging"
        ),
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_FLOOR),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 3：收敛因子 vs 网格尺寸（网格无关性）
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=48)
def _scan_cached(levels_max: int, cycle: str, nu: int, kind: str, omega: float) -> mg.FactorScan:
    return mg.factor_scan(
        tuple(range(3, int(levels_max) + 1)),
        cycle=cycle,
        kind=kind,
        omega=omega,
        nu1=int(nu),
        nu2=int(nu),
        two_grid_max_levels=int(levels_max),
    )


@functools.lru_cache(maxsize=48)
def _krylov_cached(levels_max: int, cycle: str, nu: int, kind: str, omega: float) -> tuple[np.ndarray, ...]:
    """同一右端上 CG 与 MGCG 的实测迭代数（tol=1e-8，不是时间）。"""
    ns = tuple(2**lv - 1 for lv in range(3, int(levels_max) + 1))
    cg_it: list[float] = []
    mgcg_it: list[float] = []
    for n in ns:
        levels = int(round(math.log2(n + 1)))
        A = mg.poisson_1d(n)
        x = np.arange(1, n + 1, dtype=float) / (n + 1)
        b = rng(7).standard_normal(n) + 0.3 * np.sin(np.pi * x)
        direct = conjugate_gradient(A, b, tol=1e-8, maxiter=4 * n + 20)
        pre = mg.MGPrecond(n, levels=levels, cycle=cycle, kind=kind, omega=omega, nu1=int(nu), nu2=int(nu))
        acc = pcg(A, b, precond=pre, tol=1e-8, maxiter=200)
        cg_it.append(float(direct.n_iter))
        mgcg_it.append(float(acc.n_iter))
    return np.asarray(cg_it), np.asarray(mgcg_it)


def _viz_grid_independence(p: Mapping[str, Any], t: float) -> Frame:
    levels_max = int(p["levels_max"])
    cycle = str(p["cycle"])
    nu = int(p["nu"])
    kind, omega = _smoother(p)
    tol_exp = int(p["tol_exp"])
    tol = 10.0 ** (-tol_exp)
    fr = Frame()

    scan = _scan_cached(levels_max, cycle, nu, kind, omega)
    cg_it, mgcg_it = _krylov_cached(levels_max, cycle, nu, kind, omega)
    n_seq = np.asarray(scan.n_fine, dtype=float)
    h_seq = np.asarray(scan.h, dtype=float)
    f_mg = np.asarray(scan.factor_mg, dtype=float)
    f_tg = np.asarray(scan.factor_two_grid, dtype=float)
    f_sm = np.asarray(scan.factor_smoother, dtype=float)

    mg_it, sm_it = scan.iterations_for(tol)
    mg_it = np.asarray(mg_it, dtype=float)
    sm_it = np.asarray(sm_it, dtype=float)
    mg_capped = ~np.isfinite(mg_it)
    sm_capped = ~np.isfinite(sm_it)
    mg_it = np.where(mg_capped, _ITERS_CAP, mg_it)
    sm_it = np.where(sm_capped, _ITERS_CAP, sm_it)
    speedup = cg_it / np.maximum(mgcg_it, 1.0)
    cost_mg = mg_it * np.asarray(scan.work_mg, dtype=float)
    cost_sm = sm_it * 1.0  # 单层平滑子一次扫描就是 1 个最细层等价代价

    two_ok = np.isfinite(f_tg)
    tg_spread = float(np.max(f_tg[two_ok]) - np.min(f_tg[two_ok])) if bool(np.any(two_ok)) else 0.0

    fr.add("n_seq", n_seq, unit="1", label="最细层内点数 n = 2^L - 1")
    fr.add("h_seq", h_seq, unit="1", label="最细层步长 h = 1/(n+1)")
    fr.add("level_seq", np.asarray(scan.levels, dtype=float), unit="1", label="层数 L")
    fr.add("residual_factor_mg", f_mg, unit="1", label=f"{cycle.upper()}-cycle 实测收敛因子（残差口径）")
    fr.add("residual_factor_two_grid", np.where(two_ok, f_tg, 0.0), unit="1",
           label="两网格因子（粗层精确解）")
    fr.add("residual_factor_two_grid_defined", two_ok.astype(float), unit="1", label="1 = 该尺寸上测过两网格")
    fr.add("error_factor_smoother", f_sm, unit="1", label="单层平滑子作为求解器的收敛因子 rho(T)")
    fr.add("iters_mg_measured", mg_it, unit="1", label=f"降到 {tol:g} 所需周期数（实测）")
    fr.add("iters_smoother", sm_it, unit="1", label=f"降到 {tol:g} 所需扫描数（由实测因子换算）")
    fr.add("iters_at_cap", (mg_capped | sm_capped).astype(float), unit="1",
           label=f"1 = 因子 >=1（不收敛），次数写成哨兵 {_ITERS_CAP:g}")
    fr.add("work_mg_per_cycle", np.asarray(scan.work_mg, dtype=float), unit="1",
           label="一个循环的实测代价（最细层等价点扫描数）")
    fr.add("work_to_tol_mg", cost_mg, unit="1", label=f"MG 降到 {tol:g} 的总代价")
    fr.add("work_to_tol_smoother", cost_sm, unit="1", label=f"平滑子降到 {tol:g} 的总代价（1 次扫描 = 1 单位）")
    fr.add("cg_iters", cg_it, unit="1", label="CG 迭代数（tol=1e-8，同一右端）")
    fr.add("mgcg_iters", mgcg_it, unit="1", label="MGCG 迭代数（同一个 V-cycle 当预条件子）")
    fr.add("mgcg_speedup", speedup, unit="1", label="CG/MGCG 的迭代数加速比（不是时间比）")
    fr.set_meta(
        levels_max=levels_max,
        n_max=float(n_seq[-1]) if n_seq.size else 0.0,
        cycle_index=int(("v", "w").index(cycle)),
        nu=nu,
        smoother_index=int(("jacobi", "gs", "sor").index(kind)),
        omega=omega,
        tol=tol,
        tol_exp=tol_exp,
        t_requested=float(t),
        mg_factor_mean=scan.mg_factor_mean,
        mg_factor_std=scan.mg_factor_std,
        mg_factor_drift=scan.mg_factor_drift,
        mg_factor_max=float(np.max(f_mg)) if f_mg.size else 0.0,
        mg_factor_min=float(np.min(f_mg)) if f_mg.size else 0.0,
        mg_factor_relative_spread=(
            float((np.max(f_mg) - np.min(f_mg)) / max(np.mean(f_mg), 1e-300)) if f_mg.size else 0.0
        ),
        two_grid_factor_spread=tg_spread,
        smoother_slope_h2=float(scan.smoother_trend),
        smoother_worst=float(np.max(f_sm)) if f_sm.size else 0.0,
        iters_smoother_max=float(np.max(sm_it)) if sm_it.size else 0.0,
        iters_mg_max=float(np.max(mg_it)) if mg_it.size else 0.0,
        cg_iters_max=float(np.max(cg_it)) if cg_it.size else 0.0,
        mgcg_iters_max=float(np.max(mgcg_it)) if mgcg_it.size else 0.0,
        mgcg_iters_std=float(np.std(mgcg_it)) if mgcg_it.size else 0.0,
        speedup_max=float(np.max(speedup)) if speedup.size else 0.0,
        work_to_tol_mg_max=float(np.max(cost_mg)) if cost_mg.size else 0.0,
        work_to_tol_smoother_max=float(np.max(cost_sm)) if cost_sm.size else 0.0,
        converging=bool(np.all(f_mg < 1.0)),
        n_iters_capped=int(np.count_nonzero(mg_capped | sm_capped)),
        iters_cap=_ITERS_CAP,
        smoother_modes_done=int(np.count_nonzero(scan.smoother_mode != "maxiter")),
    )
    fr.set_strings(
        status="ok" if bool(np.all(f_mg < 1.0)) else "cycle_diverging",
        time_semantics=_TIME_SEMANTICS["grid_independence"],
        curve_note=_CURVE_NOTES["grid_independence"],
        grid_independence_note=(
            f"跨 n={int(n_seq[0])}..{int(n_seq[-1])}：{cycle.upper()}-cycle 因子均值 {scan.mg_factor_mean:.4f}、"
            f"标准差 {scan.mg_factor_std:.2e}、漂移（最细-最粗）{scan.mg_factor_drift:+.2e}；"
            f"同一序列上单层平滑子从 {f_sm[0]:.6f} 一路爬到 {f_sm[-1]:.6f}"
        ),
        two_grid_note=(
            "两网格因子（粗层精确解）在一维里恒等于 1/9 = 0.111111（每一层、每一个 n 都逐位相同），"
            "它是 mu_smooth^2 = (1/3)^2 的直接后果；V/W 循环比它差，差的就是「粗层只解了个近似」这一项"
        ),
        smoother_note=(
            "iters_smoother 是**换算值**：单层平滑子要降到 1e-8 需要 1e4~1e5 次扫描，帧里不跑那么久，"
            "用实测因子换算并在 meta 里注明；MG 那一列是真的数出来的周期数"
        ),
        krylov_note=(
            "CG 的迭代数随 n 线性涨（一维 Poisson 的 kappa = O(h^-2)），MGCG 基本不随 n 变——"
            "「多重网格当预条件子」与「多重网格单独迭代」的差别在这里：前者每次迭代多花一个循环，"
            "但迭代数从 O(1/h) 掉到 O(1)，加速比是**迭代数比**，不是时间比"
        ),
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_FLOOR),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 4：同一残差在细 / 粗网格上的形态
# --------------------------------------------------------------------------- #
def _viz_residual_transfer(p: Mapping[str, Any], t: float) -> Frame:
    n = int(p["n"])
    if n % 2 == 0:
        n += 1  # 嵌套网格要求奇数（契约测试会扫到偶数参数值）
    frac = float(p["theta_frac"])
    mix = float(np.clip(float(p["mix"]), 0.0, 1.0))
    restrict_kind = str(p["restrict"])
    kind, omega = _smoother(p)
    nu = int(round(max(0.0, min(float(t), _T_MAX["residual_transfer"]))))
    fr = Frame()

    nc = (n - 1) // 2
    P = mg.prolongation_matrix(nc)
    R = mg.restriction_matrix(nc, kind=restrict_kind)
    V = sine_eigenvectors_1d(n)
    theta = np.arange(1, n + 1, dtype=float) * np.pi / (n + 1)

    k_mode = int(round(frac * (n + 1)))
    k_mode = min(max(k_mode, 1), n)
    low = V[:, 0]
    osc = V[:, k_mode - 1]
    r0 = math.sqrt(max(0.0, 1.0 - mix)) * low + math.sqrt(mix) * osc
    r0 = r0 / max(_norm(r0), 1e-300)

    hist_r = [1.0]
    hist_loss: list[float] = []
    cur = r0.copy()
    for step in range(nu + 1):
        if step:
            cur = mg.smooth(mg.poisson_1d(n), np.zeros(n), cur, kind=kind, omega=omega, sweeps=1)
            hist_r.append(_norm(cur) / max(_norm(r0), 1e-300))
        back = P @ (R @ cur)
        hist_loss.append(_norm(cur - back) / max(_norm(cur), 1e-300))
    fine_now = cur
    coarse_now = R @ fine_now
    back_now = P @ coarse_now
    lost_now = fine_now - back_now

    # 逐频率曲线：可见性、别名、以及"平滑子 × 粗校正"的互补性
    A = mg.poisson_1d(n)
    E2 = mg.coarse_correction_matrix(mg.mg_hierarchy(n, levels=2))
    loss_vs_theta = np.zeros(n, dtype=float)
    coarse_vis_ratio = np.zeros(n, dtype=float)
    coarse_corr = np.zeros(n, dtype=float)
    alias = np.zeros(n, dtype=float)
    for k in range(n):
        v = V[:, k]
        rv = R @ v
        coarse_vis_ratio[k] = _norm(rv)
        loss_vs_theta[k] = _norm(v - P @ rv)
        coarse_corr[k] = _norm(E2 @ v)
        w = 2.0 * theta[k]
        alias[k] = (w - 2.0 * np.pi * round(w / (2.0 * np.pi))) / np.pi
    smoothing_vs_theta = np.asarray(mg.mode_factors_measured(A, kind=kind, omega=omega)[0], dtype=float)
    product = smoothing_vs_theta * coarse_corr

    fine_disp, fine_floor = _floor(np.abs(lost_now))
    loss_disp, loss_floor = _floor(np.asarray(hist_loss, dtype=float))
    r_norm_disp, r_floor = _floor(np.asarray(hist_r, dtype=float))

    # 粗层可见部分里还剩多少"高频"能量（细网格口径）
    def _hf_ratio(vec: np.ndarray) -> float:
        c = V.T @ np.asarray(vec, dtype=float)
        hi = theta >= np.pi / 2.0
        tot = float(np.sum(c * c))
        return float(np.sum(c[hi] ** 2) / tot) if tot > 0.0 else 0.0

    hf_fine = _hf_ratio(fine_now)
    hf_visible = _hf_ratio(back_now)
    loss_now = hist_loss[-1]
    loss0 = hist_loss[0]

    fr.add("fine_x", np.arange(1, n + 1, dtype=float) / (n + 1), unit="1", label="细网格内点 x")
    fr.add("coarse_x", np.arange(1, nc + 1, dtype=float) * (2.0 / (n + 1)), unit="1", label="粗网格内点 x")
    fr.add("residual_fine", fine_now, unit="1", label=f"平滑 {nu} 次后的细网格残差（已归一化初值）")
    fr.add("residual_coarse", coarse_now, unit="1", label="限制到粗网格的残差 R r")
    fr.add("residual_prolonged", back_now, unit="1", label="粗层可见部分 P R r（搬到细网格上比较）")
    fr.add("residual_lost", lost_now, unit="1", label="粗网格看不见的部分 r - P R r")
    fr.add("residual_lost_abs", fine_disp, unit="1", label=f"|r - P R r|（对数轴，地板 {_FLOOR:g}）")
    fr.add("residual_at_floor", fine_floor, unit="1", label="1 = 该点已到地板")
    fr.add("loss_history", loss_disp, unit="1", label="相对损失 ||r - P R r||/||r|| 随 nu 的走向")
    fr.add("loss_axis", np.arange(nu + 1, dtype=float), unit="1", label="预平滑步数 nu")
    fr.add("residual_norm_history", r_norm_disp, unit="1", label="||r_nu||/||r_0||（残差自身在变小）")
    fr.add("residual_norm_at_floor", r_floor, unit="1", label="1 = 该步残差范数已到地板")
    fr.add("theta_over_pi", theta / np.pi, unit="1", label="theta_k / pi（细网格频率）")
    fr.add("loss_vs_theta", loss_vs_theta, unit="1", label="纯模态的不可见比例（与 nu 无关）")
    fr.add("coarse_visible_ratio", coarse_vis_ratio, unit="1", label="||R v_k||/||v_k||（粗层能看到多少）")
    fr.add("alias_vs_theta", alias, unit="1", label="别名后的粗网格频率（单位 pi，负数表示反向）")
    fr.add("smoothing_vs_theta", smoothing_vs_theta, unit="1", label="平滑子的一步放大率（实测）")
    fr.add("coarse_correction_vs_theta", coarse_corr, unit="1", label="两网格校正算子对纯模态的放大率")
    fr.add("product_vs_theta", product, unit="1", label="平滑 x 粗校正（互补性：一个管高频、一个管低频）")
    fr.set_meta(
        n=n,
        n_coarse=nc,
        theta_frac=frac,
        mode_index=k_mode,
        mix=mix,
        nu=nu,
        t_requested=float(t),
        restrict_index=int(("full_weighting", "injection").index(restrict_kind)),
        smoother_index=int(("jacobi", "gs", "sor").index(kind)),
        omega=omega,
        floor=_FLOOR,
        loss_now=loss_now,
        loss_unsmoothed=loss0,
        loss_ratio=float(loss_now / max(loss0, 1e-300)),
        residual_norm_ratio=float(hist_r[-1]),
        hf_ratio_fine=hf_fine,
        hf_ratio_coarse_visible=hf_visible,
        coarse_visible_amplitude=float(_norm(coarse_now)),
        alias_theta_pi=float(alias[k_mode - 1]),
        loss_at_current_theta=float(loss_vs_theta[k_mode - 1]),
        max_loss=float(np.max(loss_vs_theta)),
        max_smoothing=float(np.max(smoothing_vs_theta)),
        max_coarse_correction=float(np.max(coarse_corr)),
        max_product=float(np.max(product)),
        step_factor_high=float(smoothing_vs_theta[k_mode - 1]),
    )
    fr.set_strings(
        status="ok",
        time_semantics=_TIME_SEMANTICS["residual_transfer"],
        floor=f"对数轴地板 {_FLOOR:g}：更小的量写成地板并置 residual_at_floor",
        restrict_note=(
            "full_weighting 用 (r_{2j} + 2 r_{2j+1} + r_{2j+2})/4（= 0.5 P^T，与线性插值互为伴随）；"
            "injection 只取 r_{2j+1}，便宜但丢掉两侧权重，于是 R 不再是 c P^T，"
            "两网格因子从 0.111 退到 0.332（实测）"
        ),
        curve_note=_CURVE_NOTES["loss_vs_theta"],
        aliasing_note=(
            "细网格频率 theta 限制到粗网格后变成 2 theta（按 2 pi 折叠，见 alias_vs_theta）："
            "theta/pi 接近 1 的高频会变成**粗网格上的低频**——粗层不是看不见它，"
            "而是把它认成了另一个（低频）函数"
        ),
        shape_note=(
            f"当前 theta/pi={theta[k_mode - 1] / np.pi:.4f}：细残差的高频能量比 {hf_fine:.4f}，"
            f"而粗层可见部分 P R r 的高频能量比只有 {hf_visible:.4f}——粗层拿到的永远是低频形状；"
            f"它占原残差的比例（coarse_visible_amplitude）只有 {_norm(coarse_now):.4g}"
        ),
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_FLOOR),
    )
    return fr


# --------------------------------------------------------------------------- #
# compute 入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float = 0.0) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化与校验。"""
    if viz_id == "smoothing_spectrum":
        return _viz_smoothing_spectrum(params, t)
    if viz_id == "vcycle_hierarchy":
        return _viz_vcycle_hierarchy(params, t)
    if viz_id == "grid_independence":
        return _viz_grid_independence(params, t)
    if viz_id == "residual_transfer":
        return _viz_residual_transfer(params, t)
    raise KeyError(f"章节 4.5 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
MANIFEST: dict[str, Any] = {
    "chapter": "4.5",
    "title": "多重网格与快速求解",
    "part": "IV",
    "summary": (
        "经典迭代法不是「收敛得慢」，而是**对不同频率收敛得完全不同**："
        "加权 Jacobi（omega=2/3）一步把最高频压到 1/3，却把最低频留在 1-O(h^2)。"
        "多重网格把这件事变成算法：平滑子管高频、粗网格管低频，"
        "于是收敛因子与网格尺寸无关。这一章把四条证据链并排放：逐模态的实测放大率、"
        "V-cycle 的真实搬运顺序、跨 h 的因子扫描（含均值/标准差/漂移）、"
        "以及 MGCG 的迭代数与 CG 的对比。三个最容易骗人的地方都标出来了："
        "V-cycle 因子不是 0、限制算子的缩放常数必须与粗层算子配对（错配直接发散）、"
        "以及把非线性问题当线性解会收敛到一个真残差 O(1) 的假解。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "smoothing_spectrum",
            "title": "频谱衰减：高频迅速消失，低频原地不动",
            "description": (
                "主图（对数轴）是**逐模态**振幅：初值 |c_k(0)|、第 k 步实测 |c_k|、"
                "以及理论 |lambda_k|^k|c_k(0)| 三条曲线叠在一起，拖动迭代步数看高频整片塌下去、"
                "最低模态几乎不动。副图是每个频率上的一步放大率（实测范数比 vs 理论模长）"
                "与 1/3 参考线，高低频分界画在 theta/pi=0.5。"
                "content=mixed/low/high 三个初值让你看清「整体误差范数掉得慢」"
                "完全是被最低模态拖住的。"
            ),
            "dataSource": "backend",
            "params": [
                {"name": "n", "label": "内点数 n", "type": "int", "min": 15, "max": 127, "step": 16,
                 "default": 63, "critical": [], "criticalNote": ""},
                {"name": "smoother", "label": "平滑子", "type": "enum",
                 "options": ["jacobi", "sor"], "default": "jacobi", "critical": [], "criticalNote": ""},
                {"name": "omega", "label": "权重 omega", "type": "float", "min": 0.2, "max": 1.0,
                 "step": 0.0167, "default": 0.6667, "critical": [0.6667],
                 "criticalNote": (
                     "omega=2/3 是加权 Jacobi 在一维的**最优平滑权重**：高频（theta>=pi/2）放大率的"
                     "最大值降到 1/3，且在 theta=pi/2 与 theta=pi 两端同时取到。"
                     "小于 2/3 时高频与低频一起被拖慢；大于 2/3 时最高频的放大率重新变大，"
                     "omega=1（普通 Jacobi）时最高频几乎是 1-O(h^2)（实测 0.9808，n=15）。"
                     "选 sor 时这条线的含义不变，但 SOR 的实测曲线与理论曲线本来就会分开。"
                 ),
                 "beyond": "above"},
                {"name": "content", "label": "初值内容", "type": "enum",
                 "options": ["mixed", "low", "high"], "default": "mixed", "critical": [], "criticalNote": ""},
            ],
            "time": {"mode": "scrub", "tMax": 30.0, "dt": 1.0, "fps": 30, "label": "迭代步 k"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "spectrum", "autoscaled": True, "title": "逐模态振幅（对数轴，地板 1e-16）",
                 "ylabel": "|c_k|", "scale": "log"},
                {"id": "factor", "autoscaled": False,
                 "title": "一步放大率（固定窗口 0-1.1：越过 1 就是放大了）", "ylabel": "|lambda_k|"},
            ],
            "pitfalls": (
                "第一，整体误差范数是**被最低模态拖住**的：n=63 的 mixed 初值下，10 步后高频"
                "只剩 6.7e-8，而最低模态还有 0.992——看总范数会以为「平滑几乎没用」。"
                "第二，副图两条线在 SOR 下不是同一条：实测是单模态一步范数比、理论是迭代矩阵特征值的"
                "模长，SOR 的 T 非正规，n=127、omega=1.2 时理论值 0.999 而实测只有 0.542，"
                "拿理论线当「实际衰减」会高估几个数量级。第三，振幅对数是 1e-16 的地板："
                "曲线平贴地板不是「衰减停了」，而是已经到舍入尺度（amplitude_at_floor 标了）。"
            ),
            "theory": [
                "加权 Jacobi：T = I - omega D^{-1}A，逐模态 lambda_k = 1 - omega(1-cos theta_k)，"
                "theta_k = k pi/(n+1)；一维最优 omega=2/3 时 |lambda_k| = |1 - (4/3) sin^2(theta_k/2)|",
                "平滑因子 mu_smooth = max_{theta>=pi/2} |lambda| = 1/3（omega=2/3 的加权 Jacobi）；"
                "最低模态 |lambda_1| = 1 - (2/3)(1-cos(pi h)) = 1 - pi^2 h^2/3 + O(h^4)",
                "Gauss-Seidel 的迭代矩阵与 A 不同特征向量（非正规）：特征值是 cos^2 theta_k，"
                "而单模态一步放大率的无穷网格符号是 1/sqrt(5-4cos theta)，高频半带最大值 1/sqrt(5)=0.4472",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "vcycle_hierarchy",
            "title": "V-cycle 层级图：误差在粗细网格间的转移与校正",
            "description": (
                "主图是当前周期的误差形态：细网格误差曲线 + 各层（细到粗）误差的限制结果"
                "（profile_flat 配 profile_offset 切片），副图是事件序列——"
                "mg_apply(trace=...) 记录的真实搬运顺序（进入该层 / 前平滑 / 限制 / 粗层解 / "
                "延拓校正 / 后平滑），横轴是事件序号、纵轴是该层当前向量的范数（对数轴）。"
                "拖动周期数看误差一路塌下去，同时看每层的锯齿。"
            ),
            "dataSource": "backend",
            "params": [
                {"name": "levels", "label": "网格层数 L（n=2^L-1）", "type": "int", "min": 3, "max": 6,
                 "step": 1, "default": 4, "critical": [], "criticalNote": ""},
                {"name": "nu", "label": "前/后平滑步数 nu", "type": "int", "min": 1, "max": 3, "step": 1,
                 "default": 1, "critical": [], "criticalNote": ""},
                {"name": "cycle", "label": "循环类型", "type": "enum", "options": ["v", "w"],
                 "default": "v", "critical": [], "criticalNote": ""},
                {"name": "smoother", "label": "平滑子", "type": "enum", "options": ["jacobi", "gs"],
                 "default": "jacobi", "critical": [], "criticalNote": ""},
                {"name": "content", "label": "初值误差内容", "type": "enum",
                 "options": ["low", "high", "mixed"], "default": "mixed", "critical": [], "criticalNote": ""},
            ],
            "time": {"mode": "scrub", "tMax": 5.0, "dt": 1.0, "fps": 30, "label": "周期 c"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {"id": "profile", "autoscaled": True, "title": "误差形态（细网格与各层的限制结果）",
                 "ylabel": "e(x)"},
                {"id": "events", "autoscaled": True, "title": "下一个周期的事件序列（该层向量范数）",
                 "ylabel": "||v_l||", "scale": "log"},
            ],
            "pitfalls": (
                "第一，最粗层的 24 次扫描在代价账里占 4.5/16（n=15 时 1.6 个最细层等价单位），"
                "而它在图上只是最后一个点——「粗层反正便宜」在小网格上是错的。"
                "第二，副图纵轴是**该层当前向量的范数**，粗层的残差范数天然比细层小几个数量级"
                "（限制的权重和是 1 的一半），跨层直接比大小会得出「粗层误差更小」的假结论，"
                "要比就比同一层的锯齿（限制前的残差 vs 校正后的误差）。"
                "第三，V-cycle 因子实测约 0.19（n=15..127 上基本不变），它不是 0："
                "固定递归的循环只是收缩，降到 1e-8 还要 11 个周期。本页的角点都不会发散；"
                "想让因子越过 1，要回到第 1 页把 omega 调到 2/3 之外的弱平滑区，"
                "那时帧里 status 会直接写 cycle_diverging（而不是画一条悄悄变大的曲线）。"
            ),
            "theory": [
                "粗网格校正：x <- x + P A_H^{-1} R (b - A_h x)；两网格算子 "
                "E_TG = S^{nu2}(I - P A_H^{-1} R A_h) S^{nu1}",
                "V-cycle：每层 gamma=1 次递归；W-cycle：gamma=2。代价按层累加，"
                "V 是几何级数（约 (nu1+nu2)(1+1/2+...) n0 个点扫描），W 每层被访问 2^l 次、总代价 O(n0 log n0)",
                "一维里 R A_h P = A_H 精确成立（Galerkin 与重离散重合），"
                "且 P^T A_h P = 2 A_H —— 这就是 R = (1/2)P^T 里那个 1/2 的来源",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "grid_independence",
            "title": "收敛因子 vs 网格尺寸：跨 h 的实测与单层迭代的对比",
            "description": (
                "主图（横轴 n=2^L-1，对数横轴）四条曲线：单层平滑子的收敛因子 rho(T)（随 n 趋近 1）、"
                "两网格因子（恒为 1/9）、当前 V/W 循环的实测因子（≈0.19 且几乎水平）、"
                "以及 MGCG 的迭代数换算曲线。副图是对数纵轴的「降到 1e-8 要多少次」："
                "单层平滑子从几百次涨到几万次，MG 一直是个位数周期；"
                "CG 与 MGCG 的实测迭代数也画在同一张图上。"
            ),
            "dataSource": "backend",
            "params": [
                {"name": "levels_max", "label": "最大层数（n 到 2^L-1）", "type": "int", "min": 4, "max": 7,
                 "step": 1, "default": 7, "critical": [], "criticalNote": ""},
                {"name": "cycle", "label": "循环类型", "type": "enum", "options": ["v", "w"],
                 "default": "v", "critical": [], "criticalNote": ""},
                {"name": "nu", "label": "前/后平滑步数 nu", "type": "int", "min": 1, "max": 3, "step": 1,
                 "default": 1, "critical": [], "criticalNote": ""},
                {"name": "smoother", "label": "平滑子", "type": "enum", "options": ["jacobi", "gs"],
                 "default": "jacobi", "critical": [], "criticalNote": ""},
                {"name": "tol_exp", "label": "目标精度 1e-k", "type": "int", "min": 4, "max": 10, "step": 1,
                 "default": 8, "critical": [], "criticalNote": ""},
            ],
            "time": {"mode": "none"},
            "requiredAux": ["residual", "error"],
            "axes": [
                {"id": "factors", "autoscaled": False,
                 "title": "收敛因子（固定窗口 0-1.05，对数横轴）", "ylabel": "rho", "scale": "linear"},
                {"id": "iterations", "autoscaled": True,
                 "title": "降到目标精度所需次数（对数轴）", "ylabel": "次数", "scale": "log"},
            ],
            "pitfalls": (
                "第一，「网格无关」不等于常数：实测 V-cycle 因子在 n=15..127 上是 0.184/0.189/0.189/0.191，"
                "均值 0.183、标准差 1.1e-2、漂移 +2.9e-2——随层数增加**缓慢变大**，"
                "把「无关」说成「完全一样」是夸大。第二，W-cycle 的因子确实更小（0.114 vs 0.190，"
                "接近两网格的 1/9），但每个周期的实测代价是 18-23 个最细层等价单位、V 只有 4-5 个："
                "降到 1e-8 的总代价 W 反而贵约 4 倍，「因子更小」被「一步更贵」吃掉。"
                "第三，iters_smoother 是由实测因子**换算**的（1e4~1e5 次扫描帧里没真跑），"
                "而 MG 的周期数是真的数出来的；把换算值当实测值引用是最容易犯的偷懒。"
            ),
            "theory": [
                "两网格因子（nu1=nu2=1，omega=2/3）在一维等于 mu_smooth^2 = 1/9；"
                "V/W 循环因子更大，因为粗层只是被近似求解（V 实测约 0.19）",
                "单层平滑子 rho(T_J) = cos(pi h) = 1 - pi^2 h^2/2 + O(h^4)："
                "-log(1-rho) 对 log h 的斜率应为 2（实测 1.889，因为 n 还不够大、还没进渐近区）",
                "CG 迭代数 ~ sqrt(kappa) = O(h^{-1})；MGCG 每次迭代代价 ~ 一个循环，"
                "迭代数 O(1)，所以加速比按迭代数是 O(h^{-1})（实测 n=255 时 36.4 倍）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "residual_transfer",
            "title": "同一残差在细/粗网格上的形态对比",
            "description": (
                "主图把当前细网格残差、粗层可见部分 P R r、以及不可见部分 r - P R r 画在一起"
                "（时间轴是限制前的平滑步数 nu）。副图是对数纵轴的逐频率曲线："
                "loss_vs_theta（纯模态的不可见比例）、coarse_visible_ratio、以及"
                "平滑子 x 粗校正的乘积（互补性：一条在高频掉下去、另一条在高频涨上去）。"
                "把 theta_frac 拖到 0.95、mix 拖到 1，就能看到「粗层可见部分永远是低频、"
                "却只占原残差的 4e-4」这件事。"
            ),
            "dataSource": "backend",
            "params": [
                {"name": "n", "label": "细网格内点数 n（奇数）", "type": "int", "min": 31, "max": 127,
                 "step": 2, "default": 63, "critical": [], "criticalNote": ""},
                {"name": "theta_frac", "label": "高频分量 theta/pi", "type": "float", "min": 0.05, "max": 0.99,
                 "step": 0.01, "default": 0.95, "critical": [0.5],
                 "criticalNote": (
                     "theta/pi=0.5 是粗网格能**忠实表示**的频率上界：低于它的分量限制后仍是自己"
                     "（损失是 O(h^2) 的插值误差），高于它的分量会被别名成 2 theta 折叠后的另一个频率。"
                     "theta/pi 越接近 1，粗层看到的东西越像低频（theta/pi=0.95 时别名到 -0.0938 pi），"
                     "而可见幅度只剩 3.8e-3；到 Nyquist（theta/pi=1）时 R r = 0，损失 1.000000。"
                 ),
                 "beyond": "above"},
                {"name": "mix", "label": "高频能量占比", "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,
                 "default": 0.5, "critical": [], "criticalNote": ""},
                {"name": "restrict", "label": "限制算子", "type": "enum",
                 "options": ["full_weighting", "injection"], "default": "full_weighting",
                 "critical": [], "criticalNote": ""},
                {"name": "smoother", "label": "平滑子", "type": "enum", "options": ["jacobi", "gs"],
                 "default": "jacobi", "critical": [], "criticalNote": ""},
            ],
            "time": {"mode": "scrub", "tMax": 6.0, "dt": 1.0, "fps": 30, "label": "预平滑步数 nu"},
            "requiredAux": ["residual"],
            "axes": [
                {"id": "profiles", "autoscaled": True, "title": "细网格残差与粗层可见部分", "ylabel": "r(x)"},
                {"id": "visibility", "autoscaled": True,
                 "title": "逐频率的不可见比例与互补性（对数轴）", "ylabel": "比值", "scale": "log"},
            ],
            "pitfalls": (
                "第一，「平滑之后残差就变光滑了」这句半真半假：θ/π=0.95 的纯高频残差不平滑多少步，"
                "不可见比例始终是 0.999985（形状不变、只是幅度每步乘 0.326）；"
                "真正变化的是**混合**残差里高频占的能量比，nu=0→6 时不可见比例从 0.640 掉到 0.0018。"
                "第二，粗层拿到的永远是低频形状：θ/π=0.953 的纯高频残差，粗层可见部分 P R r 的"
                "高频能量比是 0.0000，而它只占原残差的 3.8e-3——「粗层看不见高频」不是「它把高频丢了」，"
                "而是「它把高频认成了低频」。第三，injection 比 full weighting 便宜，"
                "但它破坏了 R = c P^T 的伴随关系：两网格因子从 0.111 退到 0.332（实测，n=63）。"
            ),
            "theory": [
                "满权限制 (Rr)_j = (r_{2j} + 2 r_{2j+1} + r_{2j+2})/4 = (1/2)(P^T r)_j，"
                "在 1D 里与线性插值互为伴随并给出 R A_h P = A_H",
                "细网格频率 theta 限制到粗网格后变成 2 theta（按 2 pi 折叠）："
                "theta > pi/2 的分量因此落到粗网格的低频段，这就是「高频伪装成低频」的别名现象",
                "互补性：平滑因子 |1 - (4/3) sin^2(theta/2)| 在高频降到 1/3，"
                "两网格校正对纯模态的放大率在高频升到 1.4（theta/pi=0.953）——"
                "两者相乘在整个频带上不超过 0.459（实测最大值）",
            ],
        },
    ],
}
