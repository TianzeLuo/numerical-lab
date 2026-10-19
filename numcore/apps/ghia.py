r"""6.1 章基准数据：Ghia--Ghia--Shin (1982) 顶盖驱动方腔中心线剖面。

这一节只做一件事：把**公开基准**原样存进来，让 6.1 章的"与公开算例对齐"有一条
可核对的靶子。它不含任何求解器、不含任何插值算法（插值在 :mod:`numcore.apps.cavity`
里手写——``np.interp`` 是 P3 黑名单成员）。

数据来源与**交叉核对**（三个独立转载，逐位核对过）
--------------------------------------------------
* Ghia, Ghia & Shin, *J. Comput. Phys.* **48** (1982) 387--411, Table I（竖直中心线
  :math:`u`，Re = 100/400/1000）与 Table II（水平中心线 :math:`v`）；
* UPC 学位论文（Daniel Martinez Valdivieso）表 3.1/3.2：一次给出 Re = 100/400/1000/
  3200/5000 的六列，本模块的 100/400/1000 三列与它逐位一致；
* 两个公开代码库的硬编码表（truchas ``test/lid-driven-cavity/reader.py``、
  pysph ``ghia_cavity_data.py``）：Re = 100 的 17 个点与本模块**逐位相同**。

**转载分歧（这就是本章第一条"容易读错的量"）**：Re = 1000 的 :math:`v` 剖面在
:math:`x=0.9063` 处有 ``-0.51550``（Ghia 原表、UPC 表）与 ``-0.51500``（某代码库
转载）两种写法，:math:`x=0.1563` 处有 ``0.37095`` 与 ``0.38095`` 两种写法。
分歧量级 5e-4 ~ 1e-2 —— 比大多数求解器在该网格上的离散误差还小，但足以让"容差
压到 1e-3"的对拍断言变成**假证据**：它量的是抄写差异，不是数值误差。所以
:func:`numcore.apps.cavity.compare_with_benchmark` 的登记容差按网格与格式写实，
并且**不允许**低于 :data:`BENCHMARK_TRANSCRIPTION_SPREAD`。

约定：``u`` 是沿竖直中心线 :math:`x=0.5` 的水平速度分量，自变量是 :math:`y`；
``v`` 是沿水平中心线 :math:`y=0.5` 的竖直速度分量，自变量是 :math:`x`。两者都是
单位顶盖速度 :math:`U=1` 下的无量纲量。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "REYNOLDS_SUPPORTED",
    "BENCHMARK_TRANSCRIPTION_SPREAD",
    "GHIA_VORTEX_CENTERS",
    "VERTICAL_CENTERLINE_Y",
    "HORIZONTAL_CENTERLINE_X",
    "centerline_u",
    "centerline_v",
    "vortex_centers",
    "reference_vorticity_caveat",
]

#: 本模块收录的雷诺数（Ghia 原表的另外两列 Re = 3200/5000 未收录：本章只在 100/1000 上有断言）。
REYNOLDS_SUPPORTED: tuple[int, ...] = (100, 400, 1000)

#: 同一张公开表在不同转载之间的**逐点**最大分歧（实测：Re=1000 的 v 在 x=0.9063 处
#: ``-0.51550`` vs ``-0.51500``，以及 x=0.1563 处 ``0.37095`` vs ``0.38095``）。
#: 任何"与基准对齐"的容差低于它都不可信——那是在断言抄写版本相同。
BENCHMARK_TRANSCRIPTION_SPREAD = 1.0e-2

#: 竖直中心线 :math:`x=0.5` 的采样点（Ghia 原表的节点，从 y=0 到 y=1）。
VERTICAL_CENTERLINE_Y: np.ndarray = np.array(
    [
        0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000,
        0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000,
    ],
    dtype=float,
)

#: 水平中心线 :math:`y=0.5` 的采样点。
HORIZONTAL_CENTERLINE_X: np.ndarray = np.array(
    [
        0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000,
        0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000,
    ],
    dtype=float,
)

#: Re -> 竖直中心线上的 u（与 :data:`VERTICAL_CENTERLINE_Y` 同序）。
_U_TABLE: dict[int, list[float]] = {
    100: [
        0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
        -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
        0.78871, 0.84123, 1.00000,
    ],
    400: [
        0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726,
        -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756,
        0.68439, 0.75837, 1.00000,
    ],
    1000: [
        0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805,
        -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117,
        0.57492, 0.65928, 1.00000,
    ],
}

#: Re -> 水平中心线上的 v（与 :data:`HORIZONTAL_CENTERLINE_X` 同序）。
_V_TABLE: dict[int, list[float]] = {
    100: [
        0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527,
        0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
        -0.05906, 0.00000,
    ],
    400: [
        0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174,
        0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254, -0.15663,
        -0.12146, 0.00000,
    ],
    1000: [
        0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235,
        0.02526, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669,
        -0.21388, 0.00000,
    ],
}

#: Re -> 主涡与二次涡的涡心（原文节点坐标，单位方腔）。
#: **核对状态**：Re = 100 的主涡、Re = 1000 的三个涡心都做过逐位交叉核对；
#: Re = 400 的主涡只在单一来源里出现。断言只用核对过的那两组——没核对过的数
#: 拿来当靶子，量的是抄写运气而不是求解器。
GHIA_VORTEX_CENTERS: dict[int, dict[str, tuple[float, float]]] = {
    100: {"primary": (0.6172, 0.7344)},
    400: {"primary": (0.5547, 0.6055)},
    1000: {
        "primary": (0.5313, 0.5625),
        "bottom_left": (0.0859, 0.0781),
        "bottom_right": (0.8594, 0.1094),
    },
}


def _check_re(re: int) -> int:
    r = int(re)
    if r not in _U_TABLE:
        raise ValueError(f"没有 Re={r} 的公开基准表（收录：{sorted(_U_TABLE)}）")
    return r


def centerline_u(re: int) -> tuple[np.ndarray, np.ndarray]:
    """Re 的竖直中心线剖面 ``(y, u)``（17 个点，原表节点）。"""
    r = _check_re(re)
    return VERTICAL_CENTERLINE_Y.copy(), np.array(_U_TABLE[r], dtype=float)


def centerline_v(re: int) -> tuple[np.ndarray, np.ndarray]:
    """Re 的水平中心线剖面 ``(x, v)``（17 个点，原表节点）。"""
    r = _check_re(re)
    return HORIZONTAL_CENTERLINE_X.copy(), np.array(_V_TABLE[r], dtype=float)


def vortex_centers(re: int) -> dict[str, tuple[float, float]]:
    """Re 的公开涡心位置（单位方腔坐标）。"""
    return dict(GHIA_VORTEX_CENTERS[_check_re(re)])


def reference_vorticity_caveat() -> str:
    """一句话说清"拿基准表的节点值对拍"的适用范围（进帧的 strings）。"""
    return (
        "Ghia 表是 129x129 网格上的节点值：把它当「精确解」用，误差地板就是它自己的"
        f"离散误差与转载分歧（实测最大转载分歧 {BENCHMARK_TRANSCRIPTION_SPREAD:g}）；"
        "比它更细的网格上继续压 RMSE 只是在拟合别人的离散误差"
    )
