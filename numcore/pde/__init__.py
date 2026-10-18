r"""``numcore.pde`` —— PDE 各章的数值内核（4.1 抛物型、4.2 双曲型，以及后续章节）。

**4.1 章**回答：把 :math:`u_t=\alpha u_{xx}` 往前推一步，并且知道这一步有多准、
什么时候会翻车。"知道"是那一章真正的主题——三条最容易骗人的地方各自对应一个模块：

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

**4.2 章**回答另一个问题：把 :math:`u_t+au_x=0` 推一步之后，**波变成了什么样**。
答案全部装在同一个对象 :math:`G(\varphi)` 里，但读法有三种：

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

4.2 章的三条读法是：

1. **耗散**看 :math:`\kappa=-\ln|G|=d_2\varphi^2+d_4\varphi^4+\cdots`：迎风
   :math:`d_2=\nu(1-\nu)/2`（实测与闭式相对差 4.2e-11），Lax--Wendroff / Beam--Warming 的
   :math:`d_2=0` 而 :math:`d_4\ne0`——"没有 :math:`\varphi^2` 项"不等于"没有耗散"；
2. **色散**看 :math:`-\arg G=\nu\varphi+a_3\varphi^3+\cdots`：迎风的
   :math:`a_3=-\nu(1-\nu)(1-2\nu)/6` 在 :math:`\nu=1/2` 处**变号**，
   Beam--Warming 的 :math:`a_3` 与迎风**反号**（相位超前）；
3. **谁能救**看限制器：MUSCL 重构 + :math:`\phi(r)` 把极值处的斜率削掉，
   于是总变差非增（TVD）、过冲与欠冲归零（实测 5 个 TVD 限制器在阶跃上 over=under=0），
   代价是光滑极值处降到一阶（实测 MUSCL+minmod 的幂律阶只有 1.0337，不是 2）
   并且间断永远被抹开（削波宽度按 :math:`\Delta x^{0.55}` 收缩，比一阶还慢）。

**这一章最容易骗人的三处**，也各对应一个模块：

* "格式的相速度"是没有意义的说法——相速度必须逐波数谈
  （:func:`~numcore.pde.advection.phase_speed`；实测迎风在 :math:`\varphi=0.1` 上误差
  2e-4、:math:`\varphi=2` 上 12.7%）；
* 修正方程是**截断**级数，只在收敛半径内有效（迎风在 :math:`\nu=0.4` 的奇点在
  :math:`|\varphi|\approx0.4` 附近），把预测当"真值"用是本章最贵的误读；
* "装了限制器就没事了"——CFL 是 TVD 的**必要**条件：实测 :math:`\nu=0.99` 时 TVD 成立
  （最大相对增量 ≤2.2e-16），:math:`\nu=1.01` 就破（1.96e-2、umax 1.4889）。

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
:mod:`~numcore.pde.advection`             **双曲推进与色散/耗散**：五种格式（迎风/下风/中心/
                                          Lax--Wendroff/Beam--Warming/MUSCL）的推进与单步增量、
                                          闭式符号 :math:`G(\varphi)` 与稳定性判据、
                                          修正方程系数（闭式 + 残差反解两条路线）、
                                          逐模态相速度与相位误差、数值黏性的 Richardson 极限、
                                          波包包络签名、L2/L∞ 范数、Burgers 激波与
                                          Rankine--Hugoniot 对照
:mod:`~numcore.pde.limiters`              **限制器与总变差**：10 个 :math:`\phi(r)`
                                          （含 Sweby 区域上下界与逐点归属判定）、
                                          斜率比与限制后斜率、minmod、总变差与 TV 历史、
                                          TVD 判定、逐步 TV 扫描、平台/极值计数、
                                          过冲与欠冲
========================================  ==========================================================

依赖方向：:mod:`~numcore.pde.vonneumann` 只依赖 :mod:`numpy` 与
:mod:`~numcore.pde.heat`；:mod:`~numcore.pde.heat` 只依赖 :mod:`numpy` 的原语、
:func:`numcore.interp.spline.tridiagonal_apply`（追赶法，1.5 章的成果，不重复实现）
与 :mod:`numcore.utils.rng`（确定性噪声初值）。:mod:`~numcore.pde.advection` 依赖
:mod:`numpy` 与 :mod:`~numcore.pde.limiters`（正因如此，4.2 章的两个模块**不**互相
重复实现斜率比与 TV：它们只有一份，在 :mod:`~numcore.pde.limiters` 里）。
三者都**不 import** :mod:`numcore.reference`（P10），参考实现只出现在
``tests/4.1/`` 与 ``tests/4.2/`` 里。
"""

from __future__ import annotations

__all__ = [
    "heat",
    "vonneumann",
    "multigrid",
    "advection",
    "limiters",
]
