r"""``numcore.pde`` —— 4.1 章「抛物型方程与时间推进」的数值内核。

这一包只回答一个问题：**把 :math:`u_t=\alpha u_{xx}` 往前推一步，并且知道这一步有多准、
什么时候会翻车**。"知道"是本章真正的主题——三条最容易骗人的地方各自对应一个模块：

1. **稳定判据不是"解看起来好不好"**：$r\le\frac12$ 只在连续极限下成立，有限网格上的阈值
   由边界条件、网格奇偶性与系数本身决定（:func:`~numcore.pde.vonneumann.critical_r_finite`、
   :func:`~numcore.pde.vonneumann.critical_r_2d`）；而越界的解在纯正弦初值下可以毫无异常
   （:func:`~numcore.pde.heat.heat_solve` 配 :func:`~numcore.pde.heat.initial_profile`）；
2. **稳定 $\ne$ 精确**：无条件稳定的格式在大 :math:`\Delta t` 下"不衰减、每步翻符号"
   （:func:`~numcore.pde.vonneumann.amplification`），误差被截断误差而非稳定性支配；
3. **时间阶必须对半离散解测**：拿连续解当靶子，CN 会被量成 0 阶——
   单模态下数值解精确等于 :math:`G(\varphi_k)^n\sin(k\pi x/L)`
   （:func:`~numcore.pde.heat.exact_semidiscrete_sine`、
   :func:`~numcore.pde.vonneumann.time_error_leading`）。

模块边界（谁负责哪个文件，避免与别的章抢同一个模块）：

========================================  ==========================================================
文件                                      数学对象 / 章内角色
========================================  ==========================================================
:mod:`~numcore.pde.heat`                  **空间离散 + 时间推进**：$\theta$ 族（FTCS/BTCS/CN）、
                                          三对角与循环（Sherman--Morrison）求解、三种边界条件、
                                          常源项与离散稳态、二维显式与 Peaceman--Rachford ADI、
                                          解析解与半离散精确解、误差范数、总热量
:mod:`~numcore.pde.vonneumann`            **谱与判据**：放大因子闭式与符号展开、稳定域与阈值
                                          （含有限网格与二维）、符号翻转波数、耗散系数、
                                          模态分解与逐模态存活率（带"能不能信"的掩码）
========================================  ==========================================================

依赖方向：:mod:`~numcore.pde.vonneumann` 只依赖 :mod:`numpy` 与
:mod:`~numcore.pde.heat`；:mod:`~numcore.pde.heat` 只依赖 :mod:`numpy` 的原语、
:func:`numcore.interp.spline.tridiagonal_apply`（追赶法，1.5 章的成果，不重复实现）
与 :mod:`numcore.utils.rng`（确定性噪声初值）。两者都**不 import**
:mod:`numcore.reference`（P10），参考实现只出现在 ``tests/4.1/`` 里。
"""

from __future__ import annotations

__all__ = [
    "heat",
    "vonneumann",
]
