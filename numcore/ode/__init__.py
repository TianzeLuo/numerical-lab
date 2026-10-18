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

依赖方向是单向的：3.2 不 import 3.1，3.1 也不需要 3.2（"稳定域"是 3.1 的下游话题）。
这样两章的模块可以各自独立跑断言，不会因为对方还没写完而互相阻塞。
3.3 只依赖 3.1（复用 ``rk.rk_step`` 作非辛对照），不 import 3.2；
反向也没有依赖（3.1/3.2 不知道辛性的存在）。
"""

from __future__ import annotations

__all__: list[str] = []
