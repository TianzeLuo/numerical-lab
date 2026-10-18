"""``numcore.ode`` —— 常微分方程初值问题的数值方法。

本包按"教什么"切分，而不是按"能算什么"：

* :mod:`numcore.ode.stability`（3.2 章）—— 线性测试方程 :math:`y'=\\lambda y` 的**稳定函数**
  :math:`R(z)`、绝对稳定域 :math:`S=\\{z:|R(z)|\\le1\\}`、A/L/A(:math:`\\alpha`)-稳定性判据与
  BDF 的稳定角；外加"稳定域需要的最小显式推进器"（表驱动的显式 RK 单步）。
* :mod:`numcore.ode.implicit`（3.2 章）—— 隐式 Euler / 梯形 / BDF2（Newton 迭代 + 自研 LU
  解线性系统）、刚性算例、显式与隐式的步长天花板对比。
* ``numcore.ode.rk`` / ``numcore.ode.stepsize``（3.1 章）—— **通用**显式 RK 方法族
  （euler/midpoint/heun/rk4/dopri5/bs23）、阶条件、嵌入对与自适应步长控制。
* :mod:`numcore.ode.symplectic` / :mod:`numcore.ode.invariants`（3.3 章）—— **几何**积分：
  辛 Euler / Verlet(leapfrog) / 隐式中点的一步映射与行列式、谐振子能量误差带的**闭式**、
  振荡系统的长时积分（含爆点语义）、Kepler 精确流与轨道要素（Laplace--Runge--Lenz）、
  以及一批"长期行为"的度量：带宽、两半 :math:`\\sup` 之比、趋势/带宽、幂律、闭合度、
  相位误差（先解缠绕）。非辛对照（RK4）直接复用 3.1 章的 ``rk``，不另写一份 tableau。

**边界说清楚**（避免 3.1 与 3.2 两章重复造轮子）：3.1 章负责"有哪些方法、它们的阶是多少、
步长怎么自适应"；3.2 章只负责"这些方法在 :math:`z=\\lambda h` 上是放大还是衰减"。
因此 3.2 章自带的显式实现是**最小集**：只保留稳定函数推导所需的 Butcher 表与一个
表驱动的显式 RK 单步，够用即可，不重复实现 3.1 章的嵌入对与步长控制（那部分在
``numcore.ode.stepsize`` 里）。

* :mod:`numcore.ode.dynamics` / :mod:`numcore.ode.lyapunov` / :mod:`numcore.ode.continuation`
  （3.5 章）—— **动力系统、分岔与混沌**：右端项与 Jacobi 的工厂（Lorenz/Rossler/van der Pol/
  阻尼摆/Hopf 标准型，外加 logistic 与 Henon 两个映射）、平衡点与相型判定（含**鞍焦**
  ``saddle_focus``：混合符号 + 复共轭对）、零倾线（数值追踪与闭式参数化两条路径）、
  Poincare 截面（定向穿越 + 用局部导数的 Newton 精化，残差到 :math:`10^{-12}`）、
  Lyapunov 谱（Benettin 变分方程 + 自写修正 Gram--Schmidt，带正交性缺陷与
  :math:`\\sum\\lambda=\\langle\\nabla\\!\\cdot\\!f\\rangle` 恒等式）、孪生轨道的有限时间增长率与
  可分辨时间、周期轨 Newton 与乘子（网格扫描 + 二分是第二条独立路径）、周期倍增点、
  超稳定参数与 Feigenbaum 常数。轨迹积分一律复用 3.1 章的 ``rk.rk_step``，
  三维投影只在**显示空间**做（原始 :math:`x/y/z` 照给，投影参数写进帧的 strings）。

依赖方向是单向的：3.2 不 import 3.1，3.1 也不需要 3.2（"稳定域"是 3.1 的下游话题）。
这样两章的模块可以各自独立跑断言，不会因为对方还没写完而互相阻塞。
3.3 只依赖 3.1（复用 ``rk.rk_step`` 作非辛对照），不 import 3.2；
反向也没有依赖（3.1/3.2 不知道辛性的存在）。
3.5 同样只依赖 3.1（``rk.rk_step`` / ``integrate_fixed``）；它的三个模块之间是单向的：
``dynamics`` 不 import 另外两个，``lyapunov`` 与 ``continuation`` 都 import ``dynamics``。
3.5 也不 import 3.2/3.3（稳定域与辛性都不是它的话题）。
"""

from __future__ import annotations

__all__: list[str] = []
