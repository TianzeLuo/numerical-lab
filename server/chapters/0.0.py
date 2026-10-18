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
            "description": "ζ<1 振荡衰减、ζ=1 临界、ζ>1 过阻尼；副轴给出解析解代回原方程的残差。",
            "dataSource": "backend",
            "params": [
                {
                    "name": "omega",
                    "label": "固有频率 ω",
                    "type": "float",
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "default": 1.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "zeta",
                    "label": "阻尼比 ζ",
                    "type": "float",
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "default": 0.05,
                    "critical": [1.0],
                    "criticalNote": "ζ=1 是临界阻尼：ζ<1 会穿过平衡位置，ζ≥1 不再穿过；ζ=0 时能量才严格守恒。",
                    "beyond": "above",
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
                },
            ],
            "time": {"mode": "stream", "tMax": 20.0, "dt": 0.05, "fps": 30, "label": "t"},
            "requiredAux": ["residual"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "x(t)", "ylabel": "位移 x"},
                {"id": "error", "autoscaled": True, "title": "ODE 残差", "ylabel": "|x''+2ζωx'+ω²x|", "scale": "log"},
            ],
            "pitfalls": (
                "把纵轴设成自适应时，过阻尼曲线看起来和欠阻尼一样「陡」——因为看到的只是衰减尾巴被拉满；"
                "另外 ζ=0 时能量守恒只是「解析解的性质」，任何数值积分都会缓慢漂移，别把这两件事混为一谈。"
            ),
            "theory": [
                "x'' + 2ζω x' + ω² x = 0 的特征根 λ± = ω(−ζ ± √(ζ²−1))",
                "欠阻尼频率 ω_d = ω√(1−ζ²)，包络按 e^{−ζωt} 衰减",
                "残差的期望量级由中心差分截断误差决定：O(h²·x'''')，h=1e-5 时约 1e-10",
            ],
        },
        {
            "id": "heat_ftcs",
            "title": "一维热传导：显式格式与稳定阈值",
            "description": "FTCS（时间前向、空间中心）解 u_t = αu_xx；副轴给出与精确解的差。",
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
                },
                {
                    "name": "r",
                    "label": "网格比 r = αΔt/Δx²",
                    "type": "float",
                    "min": 0.05,
                    "max": 0.7,
                    "step": 0.01,
                    "default": 0.4,
                    "critical": [0.5],
                    "criticalNote": "FTCS 的放大因子 g=1−4r sin²(kπΔx/2)，|g|≤1 对全部模态成立当且仅当 r ≤ 1/2。",
                    "beyond": "above",
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
                },
            ],
            "time": {"mode": "scrub", "tMax": 0.2, "dt": 0.005, "label": "t"},
            "requiredAux": ["error"],
            "axes": [
                {"id": "main", "autoscaled": False, "title": "u(x,t)", "ylabel": "温度 u"},
                {"id": "error", "autoscaled": True, "title": "误差 u−u_exact", "ylabel": "|误差|", "scale": "log"},
            ],
            "pitfalls": (
                "r 略大于 1/2 时图在前几十步看起来完全正常，随后突然炸成锯齿——这是「离散不稳定」的典型伪装；"
                "误差副轴用对数刻度，会让「误差已经很小」和「误差还很大」看起来差不多，必须同时读数量级。"
                "另外请求时刻 t 与实际到达时刻相差至多一个 Δt（meta.t_actual 给出真实值），别把它当成时钟。"
            ),
            "theory": [
                "FTCS 截断误差：空间 O(Δx²)、时间 O(Δt)，故总误差在 Δt = rΔx²/α 下以 O(Δx²) 收敛",
                "放大因子 g(θ) = 1 − 4r sin²(θ/2)，稳定性要求 max|g| ≤ 1 ⟺ r ≤ 1/2",
                "精确解 u(x,t) = sin(kπx)e^{−αk²π²t}（齐次 Dirichlet 边界）",
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
    fr.add("energy", energy, unit="1", label="比能量 ½(v²+ω²x²)")
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
        return "欠阻尼：解振荡衰减，包络 e^(−ζωt)"
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
