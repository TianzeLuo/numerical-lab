"""3.3 章：几何数值积分与长期行为（4 个可视化）。

帧只做搬运：所有数学量都来自 :mod:`numcore.ode.symplectic` 与
:mod:`numcore.ode.invariants`，本文件不写数学。四个可视化围绕**同一句话**展开——
"辛方法的能量误差是**有界振荡**，非辛方法是**线性漂移**"——但每个 viz 量的是它的不同侧面：

1. ``energy_drift_duel`` —— **招牌**：同一条时间轴上跑两个方法，10^6 步（可拖到 10^1…10^6）。
   辛方法一条有界振荡、非辛对照一条（RK4 的）稳定下坡或（显式 Euler 的）指数爆炸；
   副信息是"短期精度反而是 RK4 高"的同屏对照。时间轴 = :math:`\\log_{10}` 观测步数。
2. ``phase_orbit`` —— 相空间轨道的长期形态：辛方法的轨道**落在同一条不变曲线上**
   （偏差 1e-15 量级，与观测时长无关），非辛方法穿过它并持续偏移
   （RK4 在 600 圈里偏到 2.6e-4，前后半段翻倍）。同时给出到**精确轨道**的偏差
   （辛方法是 :math:`O(h^p)` 的有界带宽）与相位漂移——"轨道闭合"的两种读法必须分开。
   时间轴 = 轨道完成度。
3. ``modified_hamiltonian`` —— 修正哈密顿量：真实能量在带里振荡（带宽对 Verlet 恰为
   :math:`h^2/4`、对辛 Euler 恰为 :math:`h/(2-h)`，都有闭式），而由 2×2 映射手算出的
   二次不变量 :math:`H_h` 沿轨道守恒到 1e-14。RK4 的映射 :math:`\\det M=1-h^6/72\\ne1`：
   它**没有**二次不变量，同一公式给出的曲线自己就在漂。时间轴 = :math:`\\log_{10}` 步数。
4. ``long_term_orbit`` —— 行星轨道（Kepler，e=0、0.6、0.9）：拖观测时长看轨道是否失真。
   辛方法的能量误差在带内、角动量守恒到 1e-14、近星点方位几乎不动；
   RK4 的能量误差更小却一直在掉、角动量漂到 1e-7 量级。时间轴 = 初始周期数。

帧契约（与 1.2/3.2 章一致）：``meta`` 只放有限标量（``allow_nan=False``），字符串走
``strings``，数组走 ``add``；越出稳定边界后积分**主动停止**（``numcore`` 的 ``BLOWUP``），
帧里写"占位值 + 掩码 + 文案"三件套，绝不让 ``inf``/``NaN`` 流到前端。
"""

from __future__ import annotations

import functools
import math
from typing import Any, Mapping

import numpy as np

from numcore.ode import invariants as iv
from numcore.ode import symplectic as sp
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 单位舍入（与 1.2/3.2 章同口径）。
_U = float(np.finfo(float).eps) / 2.0

#: 帧里"非有限值怎么办"的统一口径，写进每帧的 ``strings.nonfinite_policy``。
_NONFINITE_POLICY = (
    "本帧的数组全部有限：越过稳定边界后 numcore 的积分在 max|y|>1e12 处主动停止，"
    "帧里从停止点起写**占位值**（重复最后一个可信值）并配 mask_* 掩码数组（1=可信、0=占位），"
    "meta.blowup_* 与 t_blowup_* 说明停在哪一步；meta 一律有限标量"
    "（.nlf 头用 allow_nan=False 编码，一个 NaN 整帧就写不出去），数组段虽然是裸 float64、"
    "NaN 编码得进去，但帧契约要求每个数组都有限"
)

#: 稳定边界的理论值（谐振子，单位质量）：辛 Euler 与 Verlet 都是 :math:`h<2`。
_HARMONIC_LIMIT = 2.0

#: 观测时长上限（viz 1 / viz 3 的时间轴右端）：10^6 步。
_LOG10_STEPS_MAX = 6.0


def _fin(x: Any, default: float = 0.0) -> float:
    """非有限标量换成哨兵：``.nlf`` 头用 ``allow_nan=False``，NaN/inf 直接写不进去。"""
    v = float(x)
    return v if math.isfinite(v) else float(default)


def _finalize(fr: Frame) -> Frame:
    r"""帧出口的统一体检：``meta`` 里出现非有限值时改写成 ``None`` 并在 ``strings`` 里**点名**。

    为什么需要它：本章有一类量天生可能没有有限值——"趋势/带宽"在带宽恰为 0 时
    （隐式中点在谐振子上的能量误差只有舍入级、采样点又恰好全等）是 :math:`0/0`。
    ``Frame.validate()`` 会在**编码那一刻**抛 ``NlfError``，报的是
    "meta['trend_over_band'] = inf 不是有限数"，与"哪个量没定义"毫无关系。
    所以这里统一兜住：写 ``None``（不是 0——0 会被读成"趋势为零"这个**结论**，
    而实际情况是"这个比值没有定义"），并把键名逐个列进 ``strings.none_note``。
    """
    bad = sorted(k for k, v in fr.meta.items() if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)))
    if not bad:
        return fr
    for k in bad:
        fr.meta[k] = None
    note = (
        "以下 meta 没有有限值，按帧契约写 null（不是 0）："
        + "、".join(bad)
        + "。典型来源：「趋势/带宽」在带宽恰为 0（舍入级）时是 0/0——"
        "它表示「这个比值没有定义」，而不是「趋势为零」；"
        "「闭合增长比」在前后半段偏差都是 0 时同理"
    )
    fr.strings["none_note"] = note
    return fr


def _method_label(name: str) -> str:
    return sp.method_label(name)


@functools.lru_cache(maxsize=32)
def _cached_run(system: str, method: str, h: float, n_steps: int, n_sample: int) -> sp.Run:
    """带缓存的长时间积分。

    为什么必须有缓存：时间轴是"观测步数"的 :math:`\\log_{10}`，一次拖动会接连请求
    10^2、10^3…10^6 步的帧；没有缓存就要把 10^6 步重算几十遍。
    积分本身按 ``n_steps``（参数上界）走满，**帧只截取前 ``n_obs`` 步**——
    所以"拖到 10^2 步"看到的是同一条轨迹的前 100 步，而不是另一条轨迹。
    """
    return sp.integrate(system, method, float(h), int(n_steps), n_sample=int(n_sample))


def _slice(run: sp.Run, n_obs: int) -> sp.Run:
    """把 :class:`sp.Run` 截到前 ``n_obs`` 步（采样点按步号切，端点对齐）。"""
    n = max(1, min(int(n_obs), int(run.n_steps)))
    k = int(np.searchsorted(run.idx, n, side="right"))
    k = max(2, min(k, run.idx.size)) if run.idx.size >= 2 else run.idx.size
    i_blow = run.i_blowup if run.blowup else run.idx.size
    blow = bool(run.blowup and i_blow < k)
    return sp.Run(
        system=run.system,
        method=run.method,
        h=run.h,
        n_steps=int(min(run.idx[k - 1], run.n_steps_done if run.blowup else run.idx[k - 1])),
        t=run.t[:k],
        q=run.q[:k],
        p=run.p[:k],
        energy=run.energy[:k],
        energy_error=run.energy_error[:k],
        energy_relative=run.energy_relative,
        momentum=None if run.momentum is None else run.momentum[:k],
        angular_momentum=None if run.angular_momentum is None else run.angular_momentum[:k],
        n_force=run.n_force,
        iterations_total=run.iterations_total,
        iterations_max=run.iterations_max,
        residual_max=run.residual_max,
        converged=run.converged,
        ok=not blow,
        blowup=blow,
        t_blowup=float(run.n_steps_done) * run.h if blow else float("nan"),
        i_blowup=int(i_blow) if blow else int(k),
        fast_path=run.fast_path,
        n_sample=int(k),
        idx=run.idx[:k],
        n_steps_done=int(min(run.n_steps_done, int(run.idx[k - 1]))),
    )


def _auto_run(
    system: str,
    method: str,
    h: float,
    n_steps_full: int,
    n_steps_obs: int,
    n_sample_full: int,
    *,
    min_samples: int = 16,
) -> sp.Run:
    r"""取"前 ``n_steps_obs`` 步"的轨迹，两种情况下**当场补算**：

    1. **采样太稀**：缓存里的整段轨迹按 ``n_steps_full`` 等距采样，时间轴刚拖到最左边时
       切片里可能只剩 2–3 个采样点，而带宽、趋势、闭合度都要点数够；
    2. **爆点太靠前**：``h`` 越过稳定边界后积分在第几十步就停了，整段窗口里只剩一个
       掩码——那时把窗口收到爆点的 4 倍，让"它是怎么炸的"看得见（窗口收到 0.05% 时
       补算的代价是毫秒级）。

    这两种补算都不改变结论（补算走的是同一个积分器、同一个采样口径），只是让帧可读。
    """
    run = _cached_run(system, method, float(h), int(n_steps_full), int(n_sample_full))
    sliced = _slice(run, n_steps_obs)
    if sliced.q.shape[0] >= int(min_samples) and not (sliced.blowup and run.n_steps_done < int(n_steps_obs)):
        return sliced
    if sliced.blowup:
        n = max(16, min(int(n_steps_obs), 4 * int(run.n_steps_done) + 1))
    else:
        n = max(1, int(n_steps_obs))
    return sp.integrate(system, method, float(h), n, n_sample=min(4000, max(int(min_samples), min(n, int(n_sample_full)))))


def _masked(series: Any, i_blow: int, blowup: bool) -> tuple[np.ndarray, np.ndarray, int]:
    r"""占位值 + 掩码：返回 ``(显示数组, 掩码, 被掩掉的个数)``。

    掩码口径：``1`` = 真算出来的值，``0`` = 积分已经停止、这个位置是**重复的末值**。
    占位值取"最后一个可信值"而不是 0：0 在误差图上会被读成"误差消失了"，正好读反
    （3.2 章把 |R|=+∞ 写成 0 时踩过同一个坑）。
    """
    a = np.array(series, dtype=float)
    m = np.ones(a.size, dtype=float)
    if not blowup:
        return a, m, 0
    n_good = max(1, min(int(i_blow), a.size))
    a[n_good:] = a[n_good - 1]
    m[n_good:] = 0.0
    return a, m, int(a.size - n_good)


def _drift_line(t: Any, series: Any) -> np.ndarray:
    """最小二乘趋势线（画在图上的是**同一窗口内**的拟合，不是外推）。"""
    tt = np.asarray(t, dtype=float).ravel()
    vv = np.asarray(series, dtype=float).ravel()
    if tt.size < 3:
        return np.full(tt.size, float(vv[0]) if vv.size else 0.0)
    a = iv.drift_slope(tt, vv)
    b = float(np.mean(vv)) - a * float(np.mean(tt))
    return a * tt + b


def _short_term_error(system: str, method: str, h: float, n_steps: int, *, n_sub: int = 8) -> float:
    r"""**自收敛**短期误差：同一方法在 ``h`` 与 ``h/n_sub`` 上走到**同一物理时刻**的
    :math:`\max\|y_h-y_{h/8}\|_\infty`。

    为什么不用解析解：单摆、LJ 二聚体没有闭式流。自收敛参考解自身的误差是
    :math:`O((h/8)^p)`——对 RK4（p=4）比被测的量小 4096 倍，对 Verlet 小 64 倍，
    够用且可复现；这一约定写进 docstring，免得被读成"精确误差"。
    两个解在**完全相同的采样时刻**比较（``h/8`` 走 8 倍步数，末端时刻对齐）。
    """
    n = max(2, int(n_steps))
    n_s = min(4000, n)
    a = _cached_run(system, method, float(h), n, n_s)
    b = _cached_run(system, method, float(h) / float(n_sub), n * int(n_sub), n_s)
    m = min(a.q.shape[0], b.q.shape[0], a.idx.size, b.idx.size)
    if m < 2:
        return float("nan")
    dq = np.abs(a.q[:m] - b.q[:m])
    dp = np.abs(a.p[:m] - b.p[:m])
    return float(max(np.max(dq), np.max(dp)))


def _short_term_steps(system: str, h: float, n_obs: int) -> int:
    r"""短期窗口的步数：有精确周期的系统取 8 个周期；没有周期的取 2000 步（或观测时长的 1/8）。

    为什么要封顶：LJ 二聚体没有周期概念，"观测时长的 1/8"在 10^6 步的设置下是 125000 步，
    而参考解还要再走 8 倍——那一项会悄悄吃掉 5 秒预算（实测踩到过）。
    封顶到 2000 步后整帧回到 1 秒以内，而"短期精度"本来就是**窗口内**的话，结论不受影响。
    """
    spec = sp.system(system)
    if spec.period:
        return max(16, min(20000, int(round(8.0 * float(spec.period) / float(h)))))
    return max(16, min(2000, max(16, int(n_obs) // 8)))


def _observation_steps(t: float, n_steps: int) -> int:
    """把时间轴（:math:`\\log_{10}` 步数）换成整数步并夹住：``t=0`` 给 1 步。"""
    tt = max(0.0, min(_LOG10_STEPS_MAX, float(t)))
    return max(1, min(int(n_steps), int(round(10.0**tt))))


def _band_pair(sup: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """常值带边（长度与轨迹一致，前端直接画两条水平线）。"""
    return np.full(n, float(sup)), np.full(n, -float(sup))


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
MANIFEST: dict[str, Any] = {
    "chapter": "3.3",
    "title": "几何数值积分与长期行为",
    "part": "III",
    "summary": (
        "哈密顿系统的数值解有两种错法：辛方法的能量误差在一条**带**里来回振荡（不随时间长大），"
        "非辛方法的能量误差**线性漂移**（RK4 单调掉、显式 Euler 指数涨）。"
        "本章把这句话变成三个可测的数：带宽、漂移率、以及漂移率与 h 的幂律。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "energy_drift_duel",
            "title": "能量漂移对决：10⁶ 步的辛 vs 非辛",
            "description": (
                "同一个哈密顿系统、同一个步长，两种方法各跑一步数上限（可拖到 10⁶ 步）："
                "辛方法给一条在带宽内来回振荡的能量误差曲线，非辛对照给一条单调爬升/下坡的。"
                "副信息是**短期精度反转**：同一帧里给出两种方法在 8 个周期内的自收敛误差，"
                "RK4 通常比 Verlet 小两三个量级——非辛方法不是一无是处，它只是不会长期守规矩。"
                "时间轴是 log10(观测步数)：向右拖就是把观测时长拉长，看漂移会不会被拉出来。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "system",
                    "label": "系统",
                    "type": "enum",
                    "options": ["harmonic", "pendulum", "kepler_ecc_0.6", "lj_dimer"],
                    "default": "harmonic",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "method_symplectic",
                    "label": "辛方法",
                    "type": "enum",
                    "options": ["verlet", "symplectic_euler", "leapfrog", "implicit_midpoint"],
                    "default": "verlet",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "method_baseline",
                    "label": "非辛对照",
                    "type": "enum",
                    "options": ["rk4", "explicit_euler"],
                    "default": "rk4",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "h",
                    "label": "步长 h",
                    "type": "float",
                    "min": 0.002,
                    "max": 2.5,
                    "step": 0.002,
                    "default": 0.05,
                    "critical": [_HARMONIC_LIMIT],
                    "criticalNote": (
                        "h=2 是谐振子上辛 Euler 与 Verlet 的**稳定边界**（放大因子 |λ|=1 的分界）："
                        "越过它数值轨道从椭圆变成双曲线，10⁶ 步内必然溢出——帧里会看到曲线在"
                        "max|y|>1e12 处被掩码截断（这就是「有界」这句话的边界）。RK4 的边界是 2.828。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "n_steps",
                    "label": "步数上限",
                    "type": "int",
                    "min": 1000,
                    "max": 1000000,
                    "step": 1000,
                    "default": 1000000,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_sample",
                    "label": "采样点数",
                    "type": "int",
                    "min": 200,
                    "max": 4000,
                    "step": 100,
                    "default": 1500,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 6.0, "dt": 0.1, "fps": 30, "label": "log10(观测步数)"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "title": "能量相对误差（自适应窗口）",
                    "ylabel": "ΔE/E₀",
                    "autoscaled": True,
                },
                {
                    "id": "aux",
                    "title": "带宽 / 趋势 / 短期误差（对数）",
                    "ylabel": "量级",
                    "autoscaled": True,
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "自动纵轴会把 1e-16 的舍入噪声画得和 1e-4 的漂移一样触目惊心——"
                "两条曲线的**量级**必须看 meta 里的 sup 与 drift_slope，而不是看谁「更陡」；"
                "反过来，把纵轴钉死又会让有界振荡看起来像一条直线（带宽是 O(h) 而不是 0）。"
                "另外千万别把「漂移率」读成「误差」：RK4 在 10⁶ 步内的能量误差只有 2.2e-4，"
                "比辛 Euler 的带宽 2.6e-2 小两个量级——它的错在于这条误差**一直在往同一个方向走**，"
                "观测时长再拉长 10 倍它就输了。"
            ),
            "theory": [
                "辛映射保持相面积 ⟹ 后向误差分析给出修正哈密顿量 H_h = H + O(h^p)，能量误差有界",
                "谐振子上 Verlet 的带宽恰为 h²/4、辛 Euler 恰为 [−h/(2+h), h/(2−h)]（闭式）",
                "RK4 的线性化行列式 det M = |R(ih)|² = 1 − h⁶/72 + O(h⁸) ⟹ 能量单调漂移率 h⁵/72",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "phase_orbit",
            "title": "相空间轨道：闭合曲线还是螺旋",
            "description": (
                "相平面上的长时程轨道：灰虚线是精确轨道（圆），橙线是**该方法自己的不变量**"
                "所对应的闭合曲线，蓝线是数值轨道。辛方法的三条线重合成一条（偏差 1e-15 量级）；"
                "RK4 的轨道向内螺旋（数值耗散）、显式 Euler 向外螺旋，都会穿过橙色曲线。"
                "「闭合」在这里量两件事：轨道到不变曲线的偏差（是否落在同一条曲线上）"
                "与轨道到精确轨道的偏差（离真值多远）。时间轴是轨道完成度，拖它看螺旋怎么长出来。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "method",
                    "label": "方法",
                    "type": "enum",
                    "options": [
                        "verlet",
                        "symplectic_euler",
                        "leapfrog",
                        "implicit_midpoint",
                        "rk4",
                        "explicit_euler",
                    ],
                    "default": "rk4",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "h",
                    "label": "步长 h",
                    "type": "float",
                    "min": 0.01,
                    "max": 2.5,
                    "step": 0.01,
                    "default": 0.1,
                    "critical": [_HARMONIC_LIMIT],
                    "criticalNote": (
                        "h=2 处谐振子的辛 Euler / Verlet 放大因子模正好为 1：越过它轨道不再是椭圆，"
                        "而是无界的双曲支——「轨道闭合」这句话在这里当场失效（帧里由掩码如实标出）。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "n_periods",
                    "label": "观测周期数",
                    "type": "int",
                    "min": 5,
                    "max": 2000,
                    "step": 5,
                    "default": 600,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_draw",
                    "label": "轨道绘制点数",
                    "type": "int",
                    "min": 200,
                    "max": 4000,
                    "step": 100,
                    "default": 1500,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.02, "fps": 30, "label": "轨道完成度"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "title": "相平面 (q,p)（固定窗口 ±1.6）",
                    "xlabel": "q",
                    "ylabel": "p",
                    "autoscaled": False,
                },
                {
                    "id": "aux",
                    "title": "到不变曲线/精确轨道的偏差（对数）",
                    "ylabel": "径向相对偏差",
                    "autoscaled": True,
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "最容易被骗的地方是「最近回归距离」：轨道是连续曲线、采样点是离散的，"
                "在 600 圈 6000 个采样点的设置下，Verlet 的「每圈回到出发点多近」会从前半段的 1.9e-2 "
                "长到后半段的 3.4e-1，看起来像轨道散开了——其实那只是采样点没落在回归点上"
                "（相位每圈差 4.2e-4 圈，600 圈累积 0.25 圈）。判据必须换成「到不变曲线的径向偏差」，"
                "同一组数据给出 1.2e-3 且前后半段不变。另外「闭合」不等于「回到出发点」："
                "辛方法的**相位**以 h²/24 的速率一直漂，RK4 的相位反而准 500 倍。"
            ),
            "theory": [
                "辛映射保持二次型不变量：det M = 1 时 Q=[[c,(d−a)/2],[(d−a)/2,−b]] 满足 MᵀQM=Q",
                "辛方法的相位误差率 = h²/24（Verlet，实测 4.17e-4/时间），RK4 = h⁴/120（实测 8.3e-7）",
                "非辛映射 det M = 1 ± O(h^p) ⟹ 轨道按几何级数向内/向外螺旋，径向偏差线性累积",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "modified_hamiltonian",
            "title": "修正哈密顿量：真实能量在带里振荡",
            "description": (
                "同一个谐振子，两条曲线：灰带是真实能量的误差带（带宽有闭式：Verlet 恰为 h²/4、"
                "辛 Euler 恰为 h/(2−h)），绿线是**由 2×2 映射手算出的二次不变量 H_h** 的相对变化——"
                "它在舍入级（1e-14）平着走，说明「辛方法守恒的是另一个哈密顿量」这句话是真的、可算的。"
                "切到 RK4 就会看到绿色曲线自己也开始漂：det M = 1 − h⁶/72 ≠ 1，"
                "它没有二次不变量，这不是「没算准」，是「不存在」。时间轴是 log10(步数)。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "method",
                    "label": "方法",
                    "type": "enum",
                    "options": [
                        "symplectic_euler",
                        "verlet",
                        "leapfrog",
                        "implicit_midpoint",
                        "rk4",
                        "explicit_euler",
                    ],
                    "default": "symplectic_euler",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "h",
                    "label": "步长 h",
                    "type": "float",
                    "min": 0.01,
                    "max": 1.9,
                    "step": 0.01,
                    "default": 0.1,
                    "critical": [_HARMONIC_LIMIT],
                    "criticalNote": (
                        "h=2 是带宽闭式 h/(2−h) 的极点：辛 Euler 的修正能量在 h→2 时退化，"
                        "带宽发散——滑块上限取 1.9 是为了让这条临界线看得见、又不用去处理极点本身。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "n_steps",
                    "label": "步数上限",
                    "type": "int",
                    "min": 1000,
                    "max": 1000000,
                    "step": 1000,
                    "default": 100000,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_sample",
                    "label": "采样点数",
                    "type": "int",
                    "min": 200,
                    "max": 4000,
                    "step": 100,
                    "default": 1500,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 6.0, "dt": 0.1, "fps": 30, "label": "log10(观测步数)"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "title": "真实能量的相对误差与误差带",
                    "ylabel": "ΔE/E₀",
                    "autoscaled": True,
                },
                {
                    "id": "aux",
                    "title": "修正能量与 det M 的偏离（对数）",
                    "ylabel": "相对变化",
                    "autoscaled": True,
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "最容易读错的是「绿线平就说明方法好」：H_h 平只能说明**它是这个方法真正守恒的量**，"
                "而 H_h 离真实 H 的距离正是那条灰带——带宽 2.5e-3（Verlet、h=0.1）意味着"
                "真实能量有 0.25% 的振荡，绿线对此一言不发。反过来，RK4 的绿线漂移是"
                "「不存在二次不变量」的指纹，不要把它读成「修正能量也算错了」。"
                "还有一条：修正哈密顿量对**线性**系统有精确二次不变量，对非线性系统（单摆）"
                "只有渐近级数（一般不收敛），所以这条绿线只在谐振子上是严格结论。"
            ),
            "theory": [
                "后向误差分析：辛方法精确解出的是修正哈密顿量 H_h = H + h H₁ + h² H₂ + … 的流",
                "谐振子上 H_h 是精确的二次型：Verlet 的 Q ∝ diag(1−h²/4, 1)、辛 Euler 的 Q ∝ [[1,−h/2],[−h/2,1]]",
                "对称方法（Verlet / 隐式中点）的修正级数只含 h 的偶次幂 ⟹ 带宽是 O(h²) 而不是 O(h)",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "long_term_orbit",
            "title": "行星轨道的长期演化（e=0 / 0.6 / 0.9）",
            "description": (
                "Kepler 问题（a=1、μ=1）从近星点出发：橙色是精确椭圆，蓝线是数值轨道。"
                "拖时间轴（单位：初始周期）看长期演化会不会失真：辛方法 200 圈以后轨道仍然趴在"
                "同一条椭圆上（能量误差在带内、角动量守恒到 1e-14），RK4 的能量误差虽然小两个量级，"
                "却在持续掉——200 圈后角动量漂了 4.8e-7、近星点方位每圈偏 1.6e-2 rad。"
                "e=0.9 时近星点附近角度变化最快，把 h 拖过临界值就会看到那一小段被「抄近路」。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "case",
                    "label": "轨道",
                    "type": "enum",
                    "options": ["kepler_circular", "kepler_ecc_0.6", "kepler_ecc_0.9"],
                    "default": "kepler_ecc_0.6",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "method",
                    "label": "方法",
                    "type": "enum",
                    "options": ["verlet", "symplectic_euler", "implicit_midpoint", "rk4"],
                    "default": "verlet",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "h",
                    "label": "步长 h",
                    "type": "float",
                    "min": 0.002,
                    "max": 0.05,
                    "step": 0.001,
                    "default": 0.01,
                    "critical": [0.0023],
                    "criticalNote": (
                        "近星点角步长判据 Δθ ≈ h·L/r_p² = 0.1 rad 的位置（e=0.9：h*≈2.3e-3）；"
                        "越过它，近星点那一小段每圈只落几个点，能量误差带会出现肉眼可见的台阶。"
                        "e=0.6 的同一判据给 0.0124，e=0 时角速度均匀、这条线失去意义。"
                    ),
                    "beyond": "above",
                },
            ],
            "time": {"mode": "scrub", "tMax": 200.0, "dt": 4.0, "fps": 30, "label": "观测时长（初始周期数）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "title": "轨道平面 (x,y)（窗口按 e 固定）",
                    "xlabel": "x",
                    "ylabel": "y",
                    "autoscaled": False,
                },
                {
                    "id": "aux",
                    "title": "能量/角动量/径向偏差（对数）",
                    "ylabel": "误差 / 漂移",
                    "autoscaled": True,
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "「RK4 的能量误差更小」是真的（200 圈内 4.9e-7 vs Verlet 的 7.4e-4，小三个量级），"
                "所以只截前几圈看，结论会完全反过来——这正是本章要防的误导："
                "短期精度与长期行为是两件事，判据必须包含「误差是否随时间单调」。"
                "另一条：e=0.9 的近星点速度是远星点的 19 倍，固定步长在那里最粗——"
                "把 h 拖到 0.05 时轨道看起来「还挺像」，但近星点附近已经抄了近路，"
                "能量带会跳一个量级；要看 meta 里的 sup 与 closure，而不是看轨道「像不像」。"
            ),
            "theory": [
                "中心力场中 L = x·p_y − y·p_x 是运动积分；径向 kick 的辛方法逐位保它，RK4 不保",
                "Kepler 精确解：r(ν) = a(1−e²)/(1+e cos ν)、周期 P = 2π√(a³/μ)，本章用自写 Newton 解 Kepler 方程",
                "能量误差有界 ⟹ 轨道半长轴有界；角动量漂移 ⟹ 偏心率与近星点方位漂移",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已经过 :mod:`server.frames` 规范化与区间校验。"""
    if viz_id == "energy_drift_duel":
        return _energy_drift_duel(params, t)
    if viz_id == "phase_orbit":
        return _phase_orbit(params, t)
    if viz_id == "modified_hamiltonian":
        return _modified_hamiltonian(params, t)
    if viz_id == "long_term_orbit":
        return _long_term_orbit(params, t)
    raise KeyError(f"章节 3.3 没有可视化 {viz_id!r}")


def _energy_drift_duel(params: Mapping[str, Any], t: float) -> Frame:
    r"""viz 1：能量漂移对决（时间轴 = :math:`\log_{10}` 观测步数）。"""
    system = str(params["system"])
    m_sym = str(params["method_symplectic"])
    m_base = str(params["method_baseline"])
    h = float(params["h"])
    n_steps_max = int(params["n_steps"])
    n_sample = int(params["n_sample"])
    n_obs = _observation_steps(t, n_steps_max)

    run_s = _auto_run(system, m_sym, h, n_steps_max, n_obs, n_sample)
    run_b = _auto_run(system, m_base, h, n_steps_max, n_obs, n_sample)

    err_s, mask_s, n_mask_s = _masked(run_s.energy_error, run_s.i_blowup, run_s.blowup)
    err_b, mask_b, n_mask_b = _masked(run_b.energy_error, run_b.i_blowup, run_b.blowup)

    band_s = float(np.max(np.abs(err_s[mask_s > 0.0]))) if bool(np.any(mask_s > 0.0)) else 0.0
    band_b = float(np.max(np.abs(err_b[mask_b > 0.0]))) if bool(np.any(mask_b > 0.0)) else 0.0
    line_s = _drift_line(run_s.t, err_s)
    line_b = _drift_line(run_b.t, err_b)
    trend_s, ratio_s = iv.drift_over_band(run_s.t, err_s) if err_s.size >= 3 else (0.0, 0.0)
    trend_b, ratio_b = iv.drift_over_band(run_b.t, err_b) if err_b.size >= 3 else (0.0, 0.0)
    mono_s = iv.monotone_fraction(err_s) if err_s.size >= 2 else 0.0
    mono_b = iv.monotone_fraction(err_b) if err_b.size >= 2 else 0.0
    grow_s = iv.band_growth(err_s) if err_s.size >= 4 else 0.0
    grow_b = iv.band_growth(err_b) if err_b.size >= 4 else 0.0

    hi, lo = _band_pair(band_s, err_s.size)

    # 短期精度：8 个周期（没有周期概念的系统用 1/8 的观测时长）内的自收敛误差
    spec = sp.system(system)
    n_short = _short_term_steps(system, h, n_obs)
    t_short = float(n_short) * h
    short_s = _short_term_error(system, m_sym, h, n_short)
    short_b = _short_term_error(system, m_base, h, n_short)

    fr = Frame(chapter="3.3", viz="energy_drift_duel", kind="frame")
    fr.add("t", run_s.t, unit="", label="观测时间 t")
    fr.add("energy_error_symplectic", err_s, unit="", label=f"{_method_label(m_sym)} 的能量相对误差")
    fr.add("energy_error_baseline", err_b, unit="", label=f"{_method_label(m_base)} 的能量相对误差")
    fr.add("energy_band_hi", hi, unit="", label="辛方法的误差带上边")
    fr.add("energy_band_lo", lo, unit="", label="辛方法的误差带下边")
    fr.add("drift_line_symplectic", line_s, unit="", label="辛方法在本窗口内的最小二乘趋势线")
    fr.add("drift_line_baseline", line_b, unit="", label="非辛对照在本窗口内的最小二乘趋势线")
    fr.add("mask_symplectic", mask_s, unit="", label="1=真算出来的值，0=积分已停止后的占位值")
    fr.add("mask_baseline", mask_b, unit="", label="1=真算出来的值，0=积分已停止后的占位值")

    fr.set_meta(
        t_axis=float(t),
        n_steps_observed=int(run_s.n_steps),
        n_steps_max=int(n_steps_max),
        h=float(h),
        n_sample=int(err_s.size),
        n_periods_observed=_fin(run_s.n_periods() or 0.0),
        band_symplectic=band_s,
        band_baseline=band_b,
        trend_symplectic=float(trend_s),
        trend_baseline=float(trend_b),
        trend_over_band_symplectic=float(ratio_s),
        trend_over_band_baseline=float(ratio_b),
        monotone_fraction_symplectic=float(mono_s),
        monotone_fraction_baseline=float(mono_b),
        band_growth_symplectic=_fin(grow_s),
        band_growth_baseline=_fin(grow_b),
        drift_slope_symplectic=_fin(iv.drift_slope(run_s.t, err_s) if err_s.size >= 3 else 0.0),
        drift_slope_baseline=_fin(iv.drift_slope(run_b.t, err_b) if err_b.size >= 3 else 0.0),
        end_error_symplectic=_fin(run_s.final_energy_error()),
        end_error_baseline=_fin(run_b.final_energy_error()),
        short_term_error_symplectic=_fin(short_s),
        short_term_error_baseline=_fin(short_b),
        short_term_window=float(t_short),
        ratio_band_over_baseline=_fin(band_b / band_s if band_s > 0.0 else 0.0),
        n_force_symplectic=int(run_s.n_force),
        n_force_baseline=int(run_b.n_force),
        blowup_symplectic=bool(run_s.blowup),
        blowup_baseline=bool(run_b.blowup),
        t_blowup_symplectic=_fin(run_s.t_blowup),
        t_blowup_baseline=_fin(run_b.t_blowup),
        n_masked_symplectic=int(n_mask_s),
        n_masked_baseline=int(n_mask_b),
        det_map_symplectic=_fin(sp.det_2x2(sp.linear_map(m_sym, min(h, 1.9)))),
        det_map_baseline=_fin(sp.det_2x2(sp.linear_map(m_base, min(h, 1.9)))),
    )
    fr.set_strings(
        status="ok" if (run_s.ok and run_b.ok) else "partial_blowup",
        system=spec.label,
        method_symplectic=_method_label(m_sym),
        method_baseline=_method_label(m_base),
        time_axis="log10(观测步数)：t=0 给 1 步，t=6 给 10^6 步；数组只覆盖已观测的那一段",
        horizon=(
            f"本帧观测 {run_s.n_steps} 步（h={h:g}，共 {_fin(run_s.n_periods() or 0.0):.4g} 个精确周期）。"
            f"主判据是**后半段/前半段的 sup 之比**：辛方法 {_fin(grow_s):.4f}"
            f"（{_method_label(m_sym)} 带宽 {band_s:.3e}），对照 {_method_label(m_base)} {_fin(grow_b):.4f}"
            f"（带宽 {band_b:.3e}）——有界振荡给 1，线性漂移给 2，指数爆炸给天文数字。"
            f"辅以趋势/带宽（辛 {float(ratio_s):.3f}、对照 {float(ratio_b):.3f}）："
            "纯振荡在只有 N 个周期的窗口上也会有 ≈3/(πN) 的假趋势，所以它只能当辅证"
        ),
        short_term=(
            f"短期精度（窗口 {t_short:.4g}，参考解 = 同一方法 h/8 的自收敛解）："
            f"{_method_label(m_sym)} {_fin(short_s):.3e}，{_method_label(m_base)} {_fin(short_b):.3e}"
            "——非辛方法短期更准是常态，但它不会一直守规矩"
        ),
        drift_law=(
            "谐振子上 RK4 的能量漂移率 = h^5/72（实测 h 从 0.1 到 0.00625 的幂律指数 4.9995）；"
            "辛方法的带宽幂律 = 1（辛 Euler，h/(2−h)）与 2（Verlet，h²/4）"
        ),
        masking=(
            "mask_* 为 0 的位置是积分停止后的**占位值**（重复最后一个可信值），不是真数据；"
            f"本帧被掩掉的采样点：辛方法 {n_mask_s} 个、对照 {n_mask_b} 个"
        ),
        nonfinite_policy=_NONFINITE_POLICY,
    )
    return _finalize(fr)


def _phase_orbit(params: Mapping[str, Any], t: float) -> Frame:
    r"""viz 2：相空间轨道的长期形态（时间轴 = 轨道完成度）。"""
    method = str(params["method"])
    h = float(params["h"])
    n_periods = int(params["n_periods"])
    n_draw = int(params["n_draw"])
    frac = max(0.005, min(1.0, float(t)))

    spec = sp.system("harmonic")
    period = float(spec.period or 2.0 * math.pi)
    n_steps_full = max(16, int(round(n_periods * period / h)))
    n_steps = max(8, int(round(frac * n_steps_full)))
    # 缓存轨迹的采样密度：既要够画（n_draw），又要让**相位**能解开缠绕——
    # 每个采样点转过的角必须 << π（这里按 ~1 rad 取）。不这么做的话，
    # h 小 + 周期多的组合会拿出 stride>π 的轨迹，phase_error_turns 只能抛错。
    t_full = float(n_steps_full) * h
    n_sample_full = min(20000, max(n_draw, 512, int(math.ceil(t_full / 1.0))), max(2, n_steps_full))
    n_sample = min(n_draw, max(64, n_steps))
    run = _auto_run("harmonic", method, h, n_steps_full, n_steps, n_sample_full)
    # 重采样到 n_draw 个点（等步号抽样），让不同参数下的绘制开销一致
    take = np.unique(np.rint(np.linspace(0, run.q.shape[0] - 1, n_sample)).astype(int))
    q = run.q[take, 0]
    p = run.p[take, 0]
    tt = run.t[take]
    e = run.energy_error[take]

    qi = iv.quadratic_invariant(sp.linear_map(method, h))
    inv_q, inv_p = iv.invariant_level_curve(qi.Q, float(q[0]), float(p[0]), 361)
    ex_q, ex_p = spec.exact_curve(361) if spec.exact_curve is not None else (np.zeros(0), np.zeros(0))

    closure_inv = iv.curve_closure(qi.Q, q, p)
    r_exact = np.full(q.size, float(np.hypot(q[0], p[0])))
    closure_exact = iv.reference_closure(q, p, r_exact)
    rad = iv.radius_stats(q, p)
    try:
        phase = iv.phase_error_turns(run.q[:, 0], run.p[:, 0], run.t)
        phase_note_ok = True
    except ValueError:
        phase = float("nan")
        phase_note_ok = False
    ret = iv.return_distances(q, p, tt, period) if tt[-1] > period else None

    fr = Frame(chapter="3.3", viz="phase_orbit", kind="frame")
    fr.add("orbit_q", q, unit="", label="数值轨道的 q")
    fr.add("orbit_p", p, unit="", label="数值轨道的 p")
    fr.add("invariant_q", inv_q, unit="", label="该方法二次不变量的等值线 q")
    fr.add("invariant_p", inv_p, unit="", label="该方法二次不变量的等值线 p")
    fr.add("exact_q", ex_q, unit="", label="精确轨道（圆）的 q")
    fr.add("exact_p", ex_p, unit="", label="精确轨道（圆）的 p")
    fr.add("closure_error", closure_inv.deviation, unit="", label="到不变量曲线的径向相对偏差")
    fr.add("exact_error", closure_exact.deviation, unit="", label="到精确轨道的径向相对偏差")
    fr.add("energy_error", e, unit="", label="能量相对误差")
    fr.add("t_series", tt, unit="", label="采样时刻")

    fr.set_meta(
        t_axis=float(t),
        h=float(h),
        n_steps=int(run.n_steps),
        n_periods_observed=_fin(run.t[-1] / period if run.t.size else 0.0),
        horizon_time=float(run.t[-1] if run.t.size else 0.0),
        n_draw=int(q.size),
        energy_error_sup=float(np.max(np.abs(e))) if e.size else 0.0,
        energy_error_end=_fin(run.final_energy_error()),
        radius_band=float(rad.band),
        radius_drift=float(rad.drift),
        closure_error_sup=float(closure_inv.sup),
        closure_error_growth=_fin(closure_inv.growth),
        closure_at_roundoff=bool(closure_inv.at_roundoff),
        exact_error_sup=float(closure_exact.sup),
        exact_error_growth=_fin(closure_exact.growth),
        phase_error_turns=_fin(phase),
        phase_defined=bool(phase_note_ok),
        naive_return_growth=_fin(ret.growth) if ret is not None else 0.0,
        det_map=_fin(qi.det),
        one_minus_det=_fin(1.0 - qi.det),
        symplectic=bool(qi.symplectic),
        invariant_residual=_fin(qi.residual),
        invariant_mixing=_fin(qi.mixing),
        method_order=int(sp.method_spec(method).order),
        blowup=bool(run.blowup),
        n_masked=int(np.count_nonzero(run.q.shape[0] > run.i_blowup)),
    )
    fr.set_strings(
        status="ok" if run.ok else "partial_blowup",
        method=_method_label(method),
        time_axis="轨道完成度：0 只画出发点，1 画满 n_periods 圈；数组只覆盖已走过的部分",
        closure_note=(
            f"观测 {run.n_steps} 步 = {run.t[-1] / period:.6g} 个精确周期；"
            f"到**不变量曲线**的偏差 sup={closure_inv.sup:.3e}"
            f"（{'舍入级：轨道逐点落在同一条曲线上' if closure_inv.at_roundoff else '随时间长大：轨道在穿过它'}，"
            f"前后半段 {closure_inv.first_half:.3e} → {closure_inv.second_half:.3e}）；"
            f"到**精确圆**的偏差 sup={closure_exact.sup:.3e}"
            f"（前后半段 {closure_exact.first_half:.3e} → {closure_exact.second_half:.3e}）"
        ),
        phase_note=(
            (
                f"相位误差 {phase:+.5f} 圈（累计，先解开缠绕再作差）：辛方法约 h²/24 每单位时间、"
                "RK4 约 h⁴/120——「轨道闭合」说的是点的集合闭合，不是第 n 圈回到出发点"
            )
            if phase_note_ok else
            "相位误差本帧无法唯一确定（采样步长内转过的角 ≥ π，缠绕解不开），"
            "meta.phase_error_turns 写 0 只是占位值，phase_defined=false"
        ),
        invariant_note=(
            f"映射行列式 det M = {qi.det:.12f}（1−det = {1.0 - qi.det:+.3e}），"
            f"二次型残差 ‖MᵀQM−Q‖/‖Q‖ = {qi.residual:.3e}；"
            + ("辛：存在精确二次不变量" if qi.symplectic else "非辛：不存在二次不变量，橙色曲线只是参考")
        ),
        misleading_note=(
            "「每圈最近回归距离」是采样受限的假象（本章 pitfalls 第 2 条）："
            f"同一组数据给出 growth={_fin(ret.growth) if ret is not None else 0.0:.3g}，"
            "而到不变曲线的偏差 growth 才有意义——判据看 closure_error 与 exact_error"
        ),
        nonfinite_policy=_NONFINITE_POLICY,
    )
    return _finalize(fr)


def _modified_hamiltonian(params: Mapping[str, Any], t: float) -> Frame:
    r"""viz 3：修正哈密顿量示意（时间轴 = :math:`\log_{10}` 步数）。"""
    method = str(params["method"])
    h = float(params["h"])
    n_steps_max = int(params["n_steps"])
    n_sample = int(params["n_sample"])
    n_obs = _observation_steps(t, n_steps_max)

    run = _auto_run("harmonic", method, h, n_steps_max, n_obs, n_sample)
    err, mask, n_mask = _masked(run.energy_error, run.i_blowup, run.blowup)
    good = mask > 0.0
    q = run.q[:, 0]
    p = run.p[:, 0]
    qi = iv.quadratic_invariant(sp.linear_map(method, h))
    hh = iv.invariant_series(qi.Q, q, p)
    mod = (hh - hh[0]) / abs(float(hh[0])) if float(hh[0]) != 0.0 else hh - hh[0]
    mod_disp, _mod_mask, _n = _masked(mod, run.i_blowup, run.blowup)

    band = float(np.max(np.abs(err[good]))) if bool(np.any(good)) else 0.0
    mod_sup = float(np.max(np.abs(mod_disp[good]))) if bool(np.any(good)) else 0.0
    hi, lo = _band_pair(band, err.size)
    line = _drift_line(run.t, err)

    # 带宽的**闭式**预测（只在谐振子上成立，帧里如实标注）
    if method == "verlet" or method == "leapfrog":
        band_theory = 0.25 * h * h
    elif method == "symplectic_euler":
        band_theory = h / (2.0 - h) if h < 2.0 else float("nan")
    elif method == "implicit_midpoint":
        band_theory = 0.0
    else:
        band_theory = float("nan")

    fr = Frame(chapter="3.3", viz="modified_hamiltonian", kind="frame")
    fr.add("t", run.t, unit="", label="观测时间 t")
    fr.add("energy_error", err, unit="", label="真实能量的相对误差")
    fr.add("energy_band_hi", hi, unit="", label="误差带上边（实测 sup）")
    fr.add("energy_band_lo", lo, unit="", label="误差带下边（实测 −sup）")
    fr.add("modified_error", mod_disp, unit="", label="修正能量 H_h 的相对变化")
    fr.add("drift_line", line, unit="", label="真实能量误差的最小二乘趋势线")
    fr.add("mask_energy", mask, unit="", label="1=真值，0=积分停止后的占位值")

    fr.set_meta(
        t_axis=float(t),
        h=float(h),
        n_steps=int(run.n_steps),
        n_sample=int(err.size),
        n_periods_observed=_fin(run.n_periods() or 0.0),
        band_measured=band,
        band_theory=_fin(band_theory),
        band_ratio_theory=_fin(band / band_theory if band_theory and math.isfinite(band_theory) and band_theory > 0 else 0.0),
        drift_slope=_fin(iv.drift_slope(run.t, err) if err.size >= 3 else 0.0),
        trend_over_band=_fin(iv.drift_over_band(run.t, err)[1] if err.size >= 3 else 0.0),
        monotone_fraction=_fin(iv.monotone_fraction(err) if err.size >= 3 else 0.0),
        band_growth=_fin(iv.band_growth(err) if err.size >= 4 else 0.0),
        modified_sup=mod_sup,
        modified_at_roundoff=bool(mod_sup <= 1e-12),
        det_map=_fin(qi.det),
        one_minus_det=_fin(1.0 - qi.det),
        det_theory_rk4=_fin(h**6 / 72.0 - h**8 / 576.0) if method == "rk4" else 0.0,
        symplectic=bool(qi.symplectic),
        invariant_residual=_fin(qi.residual),
        invariant_mixing=_fin(qi.mixing),
        iterations_mean=_fin(run.iterations_total / max(1, run.n_steps)),
        blowup=bool(run.blowup),
        n_masked=int(n_mask),
    )
    fr.set_strings(
        status="ok" if run.ok else "partial_blowup",
        method=_method_label(method),
        time_axis="log10(观测步数)：数组只覆盖已观测的那一段",
        band_note=(
            f"观测 {run.n_steps} 步（后半段/前半段 sup 比 = {_fin(iv.band_growth(err) if err.size >= 4 else 0.0):.4f}："
            f"1 = 有界振荡、2 = 线性漂移）；真实能量的实测带宽 {band:.6e}"
            + (f"、闭式预测 {band_theory:.6e}（比值 {band / band_theory:.6f}）" if band_theory and math.isfinite(band_theory) and band_theory > 0 else "")
            + f"；修正能量 H_h 的相对变化 {mod_sup:.3e}"
            + ("（舍入级：它是这个方法真正守恒的量）" if mod_sup <= 1e-12 else "（在漂：该方法没有二次不变量）")
        ),
        invariant_note=(
            f"det M = {qi.det:.12f}（1−det = {1.0 - qi.det:+.3e}，残差 ‖MᵀQM−Q‖/‖Q‖ = {qi.residual:.3e}）"
            + ("；辛 ⟹ H_h 精确守恒" if qi.symplectic else "；非辛 ⟹ 不存在二次不变量，绿色曲线只是同一公式的取数")
        ),
        linearity_note=(
            "修正哈密顿量的精确二次型只在**线性**系统（谐振子）上存在；非线性系统（单摆、Kepler）"
            "的 H_h 是渐近级数，一般不收敛——所以这一页固定用谐振子，"
            "结论外推时只说「辛方法的能量误差有界」，不说「H_h 一定收敛」"
        ),
        masking=(
            f"mask_energy 为 0 的位置是积分停止后的占位值，本帧 {n_mask} 个；"
            "h 接近 2 时谐振子的辛 Euler / Verlet 会被这条稳定边界抓住"
        ),
        nonfinite_policy=_NONFINITE_POLICY,
    )
    return _finalize(fr)


def _long_term_orbit(params: Mapping[str, Any], t: float) -> Frame:
    r"""viz 4：行星轨道的长期演化（时间轴 = 初始周期数）。

    量的是**两件不同的事**，必须分开报：

    * **形状**（半长轴 a、偏心率 e、近星点距离 r_p = a(1−e)）由 :math:`(E,L)` 决定。
      辛方法能量有界 + 角动量逐位守恒 ⟹ 形状**有界**；RK4 的能量单调掉 ⟹ 形状线性失真。
    * **方位**（近星点方向 :math:`\varpi`，由 Laplace--Runge--Lenz 矢量给出）是**相位型**误差：
      任何固定步长方法都会进动，辛性管不了它。把两者混成一个数，就会得到
      "辛方法也会长期失真"的错误结论（e=0.9 时尤其明显：近星点每圈只落几十个点）。
    """
    case = str(params["case"])
    method = str(params["method"])
    h = float(params["h"])
    spec = sp.system(case)
    period = float(spec.period or 2.0 * math.pi)
    _a0, _e0, _w0 = sp.orbital_elements(spec.q0_array(), spec.p0_array())
    ecc0 = float(_e0[0])
    rp0 = float(_a0[0] * (1.0 - ecc0))

    n_orbits_max = 200
    n_orbits = max(1, min(n_orbits_max, int(round(float(t))) if float(t) >= 1.0 else 1))
    n_steps_full = max(16, int(round(n_orbits_max * period / h)))
    n_steps = max(8, int(round(n_orbits * period / h)))

    run = _auto_run(case, method, h, n_steps_full, n_steps, 4000)

    take = np.unique(np.rint(np.linspace(0, run.q.shape[0] - 1, 2000)).astype(int))
    qx = run.q[take, 0]
    qy = run.q[take, 1]
    tt = run.t[take]
    err, mask, n_mask = _masked(run.energy_error, run.i_blowup, run.blowup)
    ang = run.angular_momentum
    if ang is None:
        ang = np.zeros(run.t.size, dtype=float)
    ang_err = (ang - ang[0]) / abs(ang[0]) if abs(float(ang[0])) > 0.0 else ang - ang[0]

    ref_x, ref_y = sp.kepler_orbit_curve(ecc0, 361)
    r = np.hypot(qx, qy)
    a_num, e_num, w_num = sp.orbital_elements(run.q, run.p)
    rp_num = a_num * (1.0 - e_num)
    closure = iv.closure_stats(rp_num / rp0 - 1.0)
    # e=0 的圆轨道上"相对偏心率偏差"是 0/0：此时数组装**绝对**偏心率（并在 strings 里说明）
    ecc_err = e_num / ecc0 - 1.0 if ecc0 > 1e-12 else e_num.copy()
    ecc_err_absolute = bool(ecc0 <= 1e-12)
    varpi_defined = bool(ecc0 > 1e-12)
    if varpi_defined:
        varpi_drift = np.unwrap(w_num) - float(w_num[0])
    else:
        varpi_drift = np.zeros_like(w_num)
    band = float(np.max(np.abs(err[mask > 0.0]))) if bool(np.any(mask > 0.0)) else 0.0
    trend, ratio = iv.drift_over_band(run.t, err) if err.size >= 3 else (0.0, 0.0)
    e_meas = float((float(np.max(r)) - float(np.min(r))) / (float(np.max(r)) + float(np.min(r))))

    fr = Frame(chapter="3.3", viz="long_term_orbit", kind="frame")
    fr.add("traj_x", qx, unit="", label="数值轨道的 x")
    fr.add("traj_y", qy, unit="", label="数值轨道的 y")
    fr.add("ref_x", ref_x, unit="", label="精确椭圆的 x")
    fr.add("ref_y", ref_y, unit="", label="精确椭圆的 y")
    fr.add("t_series", tt, unit="", label="采样时刻")
    fr.add("energy_error", err, unit="", label="能量相对误差")
    fr.add("angmom_error", ang_err, unit="", label="角动量相对漂移")
    fr.add("closure_error", closure.deviation, unit="", label="近星点距离的相对失真 r_p/r_p0 − 1")
    fr.add("ecc_error", ecc_err, unit="", label="偏心率偏差（e>0 时为相对 e/e0 − 1，e=0 时为绝对 e，见 strings）")
    fr.add("varpi_drift", varpi_drift, unit="rad", label="近星点方位的累积进动（e=0 时恒为 0，见 strings）")
    fr.add("mask_energy", mask, unit="", label="1=真值，0=积分停止后的占位值")

    fr.set_meta(
        t_axis=float(t),
        h=float(h),
        eccentricity=float(ecc0),
        n_orbits=int(run.t[-1] / period if run.t.size else 0),
        n_steps=int(run.n_steps),
        n_steps_max=int(n_steps_full),
        horizon_time=float(run.t[-1] if run.t.size else 0.0),
        band_energy=band,
        end_energy_error=_fin(run.final_energy_error()),
        trend_over_band=_fin(ratio),
        drift_slope=_fin(iv.drift_slope(run.t, err) if err.size >= 3 else 0.0),
        angmom_drift=_fin(iv.angular_momentum_drift(ang)),
        closure_sup=float(closure.sup),
        closure_growth=_fin(closure.growth),
        closure_first_half=float(closure.first_half),
        closure_second_half=float(closure.second_half),
        closure_at_roundoff=bool(closure.at_roundoff),
        ecc_error_sup=float(np.max(np.abs(ecc_err))) if ecc_err.size else 0.0,
        ecc_error_end=_fin(ecc_err[-1]) if ecc_err.size else 0.0,
        ecc_measured=e_meas,
        ecc_error_is_absolute=ecc_err_absolute,
        varpi_defined=varpi_defined,
        varpi_drift_total=_fin(float(varpi_drift[-1]) if varpi_drift.size else 0.0),
        varpi_drift_per_orbit=_fin(float(varpi_drift[-1]) / max(1.0, float(run.t[-1] / period)) if varpi_drift.size else 0.0),
        r_peri_measured=float(np.min(r)),
        r_apo_measured=float(np.max(r)),
        angle_step_perihelion=float(h * math.sqrt(1.0 - ecc0 * ecc0) / max(1e-12, (1.0 - ecc0) ** 2)) if ecc0 < 1.0 else 0.0,
        blowup=bool(run.blowup),
        n_masked=int(n_mask),
        n_force=int(run.n_force),
    )
    fr.set_strings(
        status="ok" if run.ok else "partial_blowup",
        case=case,
        method=_method_label(method),
        time_axis="观测时长（初始周期数）：拖到 200 看长期演化；数组只覆盖已走过的部分",
        shape_note=(
            f"**形状**：近星点距离的相对失真 sup={closure.sup:.3e}"
            f"（前半段 {closure.first_half:.3e} → 后半段 {closure.second_half:.3e}，比值 {closure.growth:.3f}）；"
            f"偏心率{'绝对' if ecc_err_absolute else '相对'}偏差 sup="
            f"{float(np.max(np.abs(ecc_err))) if ecc_err.size else 0.0:.3e}、"
            f"末端 {_fin(ecc_err[-1]) if ecc_err.size else 0.0:+.3e}"
            + ("（e=0：相对量是 0/0，这里报的是被激发的绝对偏心率）；" if ecc_err_absolute else "；")
            + f"能量误差带宽 {band:.3e}、总趋势 {float(trend):.3e}（趋势/带宽 {float(ratio):.3f}）"
        ),
        phase_note=(
            "**方位**：" + (
                f"近星点累积进动 {float(varpi_drift[-1]) if varpi_drift.size else 0.0:+.4e} rad，"
                f"合 {float(varpi_drift[-1]) / max(1.0, float(run.t[-1] / period)) if varpi_drift.size else 0.0:+.3e} rad/圈——"
                "这是相位型误差，辛性管不了它（能量有界不等于轨道方位不动）"
                if varpi_defined else
                "e=0 的圆轨道没有近星点（Laplace–Runge–Lenz 矢量恒为 0），本帧 varpi_drift 恒为 0 是**占位值**，"
                "不是「进动为零」的结论"
            )
        ),
        resolution_note=(
            f"近星点角步长 h·L/r_p² = "
            f"{float(h * math.sqrt(1.0 - ecc0 * ecc0) / max(1e-12, (1.0 - ecc0) ** 2)) if ecc0 < 1.0 else 0.0:.4f} rad；"
            f"实测偏心率 {e_meas:.6f}（初值 {ecc0:g}）、近星点 {float(np.min(r)):.6f}、远星点 {float(np.max(r)):.6f}"
        ),
        masking=(
            f"mask_energy 为 0 的位置是积分停止后的占位值（重复末值），本帧 {n_mask} 个；"
            "h 过大时近星点附近会先失稳"
        ),
        nonfinite_policy=_NONFINITE_POLICY,
    )
    return _finalize(fr)
