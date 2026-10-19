r"""``numcore.inverse.uq`` —— 5.4 章（不确定性量化与全局敏感性）的数值内核。

这一层回答四个互不相同、却经常被同一个词（"不确定性"）糊在一起的问题：

=======================================  ==========================================================
:mod:`numcore.inverse.uq.mc`             蒙特卡洛：抽样、样本均值与标准误、置信区间、
                                         覆盖率实验、直方图密度（P3 禁 ``histogram``，全部自研）、
                                         批量均值误差（不假装样本独立）
:mod:`numcore.inverse.uq.sobol_seq`     低差异序列本体：GF(2) 上的原始多项式与方向向量、
                                         Sobol' 数字网、van der Corput / Halton、
                                         秩一格点、数字移位（digital shift）与线性置乱
:mod:`numcore.inverse.uq.qmc`           拟蒙特卡洛：随机化 QMC 的复制误差估计（**不置乱就没有
                                         可用的误差估计**，这里是量出来的）、星偏差、Koksma–Hlawka
:mod:`numcore.inverse.uq.sensitivity`   Sobol' 指数（pick-freeze / Saltelli / Jansen）、
                                         bootstrap 置信区间、相关输入下的两种方差分解
:mod:`numcore.inverse.uq.pce`           多项式混沌：正交基、多指标、投影与回归系数、
                                         谱衰减、**留出集**误差、外推误差、解析 Sobol' 指数
=======================================  ==========================================================

边界（与 :mod:`numcore.inverse` 的包契约一致，这里再写一遍因为它最容易在赶工时被破）
-----------------------------------------------------------------------------------------
* 依赖方向单向：``numcore.inverse.*`` → ``numcore.linalg.*`` / ``numcore.utils``。本子包
  额外只用 numpy 的**容器与逐元素原语**：``einsum``/``cumsum``/``searchsorted``/``sort`` 之类。
* **不 import** :mod:`numcore.verify`（它是测试基础设施；``numcore/ode/rk.py`` 已经立过这条
  分层规矩）。本层需要的小工具——正态分位数、样本矩、双对数斜率——都在这里重写一遍，
  再由 ``tests/5.4`` 与 :mod:`numcore.verify` 逐位对拍。"图上的斜率"与"断言里的斜率"
  因此仍然是同一个数，但内核不依赖裁判。
* **不 import** :mod:`numcore.quad` / :mod:`numcore.interp`：PCE 的 Gauss–Legendre 节点、
  Legendre/Hermite 递推都在 :mod:`numcore.inverse.uq.pce` 内部实现（见该模块 docstring
  里的理由），``tests/5.4`` 拿 2.4 章的求积去对拍它。
* 随机数一律 :func:`numcore.utils.rng`（P4）；本层无 IO、无 ``print``、无全局可变状态（P6）。
* 帧里出现的一切"没有有限值"的情形（对数轴上的 0、未置乱时的误差带）都由 ``server/``
  那一层写成"占位值 + 掩码 + 文字"（PROTOCOLS §1.3.1）。本层只如实返回 ``nan``/``inf``，
  不做静默替换——把"没有有限值"这件事在**最靠近计算的地方**暴露出来，比在装配层猜要安全。

注：``__all__`` 只登记子模块**名字**，不在此 import 它们（与 :mod:`numcore.linalg` /
:mod:`numcore.inverse` 同一做法：并发写章时别人的半成品不该让整包 import 失败）。
"""

from __future__ import annotations

__all__ = [
    "mc",
    "pce",
    "qmc",
    "sensitivity",
    "sobol_seq",
]
