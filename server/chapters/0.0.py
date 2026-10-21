"""0.0 章：管线自检（模板章）。

这一章不是教学内容，它是**整条流水线的活体样本**：数值内核 → 参数契约 → 帧 → 前端 Scene →
三类断言 → 视觉回归 → 笔记。任何一环坏掉，`make ci` 都会红。
每一章都应当长得和它一样，只是内容换成真正的算法。

两个可视化刻意选了"有闭式解"的问题，这样 A 类断言才有真值可比：

* ``oscillator_regimes``：阻尼振子三种分支（欠阻尼/临界/过阻尼）的**解析解**，
  副轴画"把解代回原方程的残差"（数量级 1e-10，来自中心差分而不是推导自洽）；
* ``heat_ftcs``：显式 FTCS 解一维热传导，副轴画与精确解之差；
  参数 ``r`` 的理论临界值 1/2 会被 C 类断言**二分实测**（不是抄书上那句话）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from numcore.smoke import (
    damped_oscillator,
    heat_ftcs,
    heat_run_meta,
    oscillator_energy,
    oscillator_ode_residual,
)
from server.nlf import Frame

MANIFEST: dict[str, Any] = {
    "chapter": "0.0",
    "title": "管线自检（模板章）",
    "part": "",
    "summary": "用两个有闭式解的问题跑通「数值内核 → 帧 → 场景 → 三类断言 → 视觉基线」全链路。",
    "vizzes": [
        {
            "id": "oscillator_regimes",
            "title": "阻尼振子：三种阻尼分支与残差",
            "description": "$\\zeta<1$ 振荡衰减、$\\zeta=1$ 临界、$\\zeta>1$ 过阻尼；副轴给出解析解代回原方程的残差。",
            "dataSource": "backend",
            "params": [
                {
                    "name": "omega",
                    "label": "固有频率 $\\omega$",
                    "type": "float",
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "default": 1.0,
                    "critical": [],
                    "criticalNote": "",
                    "help": "无阻尼固有频率，单位是弧度/秒。$\\omega$ 越大，振荡越快、周期越短；图上的时间轴按 $\\omega$ 归一化后，不同 $\\omega$ 的曲线会落在相同的周期刻度上。",
                },
                {
                    "name": "zeta",
                    "label": "阻尼比 $\\zeta$",
                    "type": "float",
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "default": 0.05,
                    "critical": [1.0],
                    "criticalNote": "$\\zeta=1$ 是临界阻尼：$\\zeta<1$ 会穿过平衡位置，$\\zeta\\ge1$ 不再穿过；$\\zeta=0$ 时能量才严格守恒。",
                    "beyond": "above",
                    "help": "阻尼比 $\\zeta$。$\\zeta=0$ 时振幅不衰减；$0<\\zeta<1$ 时振荡并逐渐衰减；$\\zeta=1$ 是临界阻尼，刚好不振荡；$\\zeta>1$ 过阻尼，缓慢返回平衡。它决定包络 $e^{-\\zeta\\omega t}$ 的衰减速度。",
                },
                {
                    "name": "n_periods",
                    "label": "显示周期数",
                    "type": "int",
                    "min": 1,
                    "max": 20,
                    "default": 3,
                    "critical": [],
                    "criticalNote": "",
                    "help": "画多少个振荡周期。调大能看到更长时间的衰减和数值误差积累；调小则把前几个周期放大。它只改变观察窗口，不改变解本身。",
                },
            ],
            "time": {"mode": "stream", "tMax": 20.0, "dt": 0.05, "fps": 30, "label": "$t$"},
            "requiredAux": ["residual"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "$x(t)$", "ylabel": "位移 $x$"},
                {"id": "error", "autoscaled": True, "title": "ODE 残差", "ylabel": "$|x''+2\\zeta\\omega x'+\\omega^2x|$", "scale": "log"},
            ],
            "pitfalls": (
                "把纵轴设成自适应时，过阻尼曲线看起来和欠阻尼一样「陡」——因为看到的只是衰减尾巴被拉满；"
                "另外 $\\zeta=0$ 时能量守恒只是「解析解的性质」，任何数值积分都会缓慢漂移，别把这两件事混为一谈。"
            ),
            "theory": [
                "**从方程的解出开始。** 考虑 $x''+2\\zeta\\omega x'+\\omega^2x=0$。设 $x=e^{\\lambda t}$，代入得特征方程 $\\lambda^2+2\\zeta\\omega\\lambda+\\omega^2=0$，所以 $\\lambda_\\pm=\\omega\\left(-\\zeta\\pm\\sqrt{\\zeta^2-1}\\right).$ 两根的实部决定增长或衰减，虚部决定振荡快慢。",
                "**三种阻尼区间来自根的类型。** 当 $\\zeta<1$ 时根是一对共轭复数，解为 $x=e^{-\\zeta\\omega t}\\left(C_1\\cos\\omega_d t+C_2\\sin\\omega_d t\\right)$，其中 $\\omega_d=\\omega\\sqrt{1-\\zeta^2}$；当 $\\zeta=1$ 时是重根，解为 $(C_1+C_2t)e^{-\\omega t}$；当 $\\zeta>1$ 时两个实根都小于零，解是两个衰减指数之和，不再穿过平衡位置。",
                "**包络和周期要分开读。** 欠阻尼解的振荡周期是 $2\\pi/\\omega_d$，不是 $2\\pi/\\omega$；振幅的包络按 $e^{-\\zeta\\omega t}$ 衰减。图上看起来“周期变长”是阻尼把有效频率拖慢了，并不是时间轴拉伸。",
                "**数值误差来自离散梯度。** 帧用中心差分近似 $x'$，残差的主项是 $O(h^2x'''')$；当 $h=10^{-5}$ 时大约落在 $10^{-10}$ 量级，再往下就被双精度的舍入地板挡住。因此残差曲线下降一段后会变平，这不是格式“失去精度”，而是已经接近机器精度。",
                "**怎样用参数做实验。** 固定 $\\omega$ 调 $\\zeta$ 可以看到从等幅振荡、指数衰减到过阻尼的完整过渡；固定 $\\zeta$ 调 $\\omega$ 则整体压缩时间轴。$n_\\text{periods}$ 只决定窗口长度。先看包络，再看每个周期的相位，两者不要混在一起。",
            ],
        },
        {
            "id": "heat_ftcs",
            "title": "一维热传导：显式格式与稳定阈值",
            "description": "FTCS（时间前向、空间中心）解 $u_t=\\alpha u_{xx}$；副轴给出与精确解的差。",
            "dataSource": "backend",
            "params": [
                {
                    "name": "nx",
                    "label": "空间网格 nx",
                    "type": "int",
                    "min": 11,
                    "max": 201,
                    "default": 41,
                    "critical": [],
                    "criticalNote": "",
                    "help": "空间网格点数。$nx$ 越大，$\\Delta x$ 越小，空间截断误差按 $\\Delta x^2$ 下降；但为了保持同一个 $r$，时间步长也要跟着变小。调大后曲线更接近精确解。",
                },
                {
                    "name": "r",
                    "label": "网格比 $r=\\alpha\\Delta t/\\Delta x^2$",
                    "type": "float",
                    "min": 0.05,
                    "max": 0.7,
                    "step": 0.01,
                    "default": 0.4,
                    "critical": [0.5],
                    "criticalNote": "FTCS 的放大因子 $g=1-4r\\sin^2(k\\pi\\Delta x/2)$，$|g|\\le1$ 对全部模态成立当且仅当 $r\\le1/2$。",
                    "beyond": "above",
                    "help": "网格比 $r=\\alpha\\Delta t/\\Delta x^2$。FTCS 要求 $r\\le1/2$；超过它时最快的高频模态会被放大，数值解出现非物理振荡并迅速失控。$r$ 接近 $1/2$ 时误差也偏大。",
                },
                {
                    "name": "t_end",
                    "label": "终止时刻 t",
                    "type": "float",
                    "min": 0.005,
                    "max": 0.2,
                    "step": 0.005,
                    "default": 0.05,
                    "critical": [],
                    "criticalNote": "",
                    "help": "积分到哪个时刻。调大可以看长时间的衰减行为，也会放大不稳定格式的误差；调小则主要观察初始阶段。它不改变格式的稳定性判据。",
                },
            ],
            "time": {"mode": "scrub", "tMax": 0.2, "dt": 0.005, "label": "$t$"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "$u(x,t)$", "ylabel": "温度 $u$"},
                {"id": "error", "autoscaled": True, "title": "误差 $u-u_{\\mathrm{exact}}$", "ylabel": "$|u-u_{\\mathrm{exact}}|$", "scale": "log"},
            ],
            "pitfalls": (
                "r 略大于 1/2 时图在前几十步看起来完全正常，随后突然炸成锯齿——这是「离散不稳定」的典型伪装；"
                "误差副轴用对数刻度，会让「误差已经很小」和「误差还很大」看起来差不多，必须同时读数量级。"
                "另外请求时刻 t 与实际到达时刻相差至多一个 Δt（meta.t_actual 给出真实值），别把它当成时钟。"
            ),
            "theory": [
                "**从热方程到半离散系统。** 一维热方程 $u_t=\\alpha u_{xx}$。对空间用中心差分 $u_{xx}(x_j)\\approx(u_{j+1}-2u_j+u_{j-1})/\\Delta x^2$，截断误差是 $O(\\Delta x^2)$。固定边界后，这变成一个常微分方程组 $u_t=Lu$，其中 $L$ 是三对角矩阵。",
                "**FTCS 的更新与放大因子。** 时间用前向 Euler：$u_j^{n+1}=u_j^n+r(u_{j+1}^n-2u_j^n+u_{j-1}^n)$，$r=\\alpha\\Delta t/\\Delta x^2$。代入 Fourier 模态 $u_j^n=g^n e^{ij\\theta}$，得到 $g(\\theta)=1-4r\\sin^2(\\theta/2).$ 稳定性要求对所有 $\\theta$ 都有 $|g|\\le1$，最坏情况在 $\\sin^2=1$，于是 $|1-4r|\\le1$，即 $0<r\\le1/2$。",
                "**误差阶与时间步的关系。** 前向 Euler 的时间截断误差是 $O(\\Delta t)$，空间是 $O(\\Delta x^2)$。由于稳定性要求 $\\Delta t\\sim\\Delta x^2$，当网格加密时时间误差和空间误差同阶，整体表现为 $O(\\Delta x^2)$；这就是显式格式“看起来是二阶”的原因。",
                "**和精确解对照。** 齐次 Dirichlet 条件下，初值 $\\sin(k\\pi x)$ 的精确解是 $u(x,t)=e^{-\\alpha k^2\\pi^2t}\\sin(k\\pi x)$：每个空间模态独立地指数衰减，波数越大衰减越快。数值解如果出现高频抖动或振幅增长，就说明 $r$ 超过了稳定上限，而不是物理现象。",
                "**参数怎样改变图。** $nx$ 增大让空间更细，但保持 $r$ 会让 $\\Delta t$ 更小；$r$ 是稳定性旋钮，越过 $1/2$ 会失控；$t_\\text{end}$ 决定看多长时间。先看 $r$ 是否在安全区，再讨论网格误差。",
            ],
        },
    ],
}


def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已经过 :mod:`server.frames` 规范化与区间校验。"""
    if viz_id == "oscillator_regimes":
        return _oscillator_frame(params, t)
    if viz_id == "heat_ftcs":
        return _heat_frame(params, t)
    raise KeyError(f"章节 0.0 没有可视化 {viz_id!r}")


def _oscillator_frame(params: Mapping[str, Any], t: float) -> Frame:
    omega = float(params["omega"])
    zeta = float(params["zeta"])
    n_periods = int(params["n_periods"])

    # 时间栅格由"周期数"决定：ζ≥1 时用 1/ω 当特征时间
    period = 2.0 * np.pi / omega if zeta < 1.0 else 1.0 / omega
    t_max = n_periods * period
    n = 801
    tt = np.linspace(0.0, t_max, n)
    x, v = damped_oscillator(tt, omega=omega, zeta=zeta)
    energy = oscillator_energy(x, v, omega=omega)
    residual = oscillator_ode_residual(tt, omega=omega, zeta=zeta)

    fr = Frame(chapter="0.0", viz="oscillator_regimes", kind="frame")
    fr.add("t", tt, unit="s", label="时间")
    fr.add("x", x, unit="1", label="位移")
    fr.add("v", v, unit="1/s", label="速度")
    fr.add("energy", energy, unit="1", label="比能量 $\\tfrac12(v^2+\\omega^2x^2)$")
    fr.add("residual", residual, unit="1", label="ODE 残差")
    e0 = float(energy[0]) if energy.size else 0.0
    fr.set_meta(
        omega=omega,
        zeta=zeta,
        n_periods=n_periods,
        t_span=float(t_max),
        energy_drift=float(np.max(np.abs(energy - e0)) / abs(e0)) if e0 != 0.0 else 0.0,
        residual_max=float(np.max(residual)) if residual.size else 0.0,
    )
    # 字符串一律进 strings：meta 只允许 number/bool/null（PROTOCOLS §1.1）
    regime = "underdamped" if zeta < 1.0 else ("critical" if abs(zeta - 1.0) < 1e-9 else "overdamped")
    fr.set_strings(regime=regime, regime_note=_regime_note(zeta))
    return fr


def _regime_note(zeta: float) -> str:
    if zeta < 1.0:
        return "欠阻尼：解振荡衰减，包络 $e^{-\\zeta\\omega t}$"
    if abs(zeta - 1.0) < 1e-9:
        return "临界阻尼：最快回到平衡且不穿过"
    return "过阻尼：两个实指数衰减，慢的那个主导尾部"


def _heat_frame(params: Mapping[str, Any], t: float) -> Frame:
    nx = int(params["nx"])
    r = float(params["r"])
    t_end = float(params["t_end"])

    run = heat_ftcs(nx=nx, alpha=1.0, t_end=t_end, r=r, k=1)
    fr = Frame(chapter="0.0", viz="heat_ftcs", kind="frame")
    fr.add("x", run.x, unit="1", label="空间坐标")
    fr.add("u", run.u, unit="1", label="数值解")
    fr.add("exact", run.u_exact, unit="1", label="精确解")
    fr.add("error", run.error, unit="1", label="误差")
    meta = heat_run_meta(run)
    meta["t_actual"] = run.n_steps * run.dt
    meta["t_requested"] = float(t_end)
    fr.set_meta(**meta)
    fr.set_strings(stability="stable" if run.r <= 0.5 else "unstable")
    return fr
