r"""``numcore.inverse`` —— 反问题、优化与不确定性（Part V）。

**包边界与依赖方向**（这一节是这一层的契约，不是风格建议）：

* 依赖方向**单向**：``numcore.inverse.*`` → ``numcore.linalg.*`` / ``numcore.utils``。
  反问题层可以调用 Part I 的分解、特征值与条件数；反过来**不许**——``numcore.linalg``
  不得 import ``numcore.inverse``（否则 Part I 的章会依赖 Part V，注册表的拓扑序就废了）。
* 只允许 numpy 的**容器与逐元素原语**（``ndarray``、``@``、``einsum``、``cumsum``…）。
  解线性系统一律走 :mod:`numcore.linalg.lu` / :mod:`numcore.linalg.chol`，
  特征分解走 :mod:`numcore.linalg.eigen`：P2/P3 在 ``numcore/`` 下禁止
  ``numpy.linalg`` 与现成求解器属性（``solve``/``inv``/``eig``/``norm``…），
  这里不例外。
* 随机数一律 :func:`numcore.utils.rng`（P4）；本层**没有** IO、``print`` 或全局可变状态（P6）。
* 参照物（SciPy / ``numpy.linalg``）只在 :mod:`numcore.reference` 里，本层**禁止** import（P10）。

已实现的模块（各章自行追加自己那一行，**不要重排别人的条目**）：

===============================  ==========================================================
:mod:`numcore.inverse.optimize`  5.1 章：无约束优化与拟牛顿 —— 梯度下降 / 重球动量 /
                                 牛顿（Armijo、Wolfe 线搜索）/ BFGS / L-BFGS / 非线性共轭
                                 梯度 / 信赖域（Steihaug-CG 子问题）、割线方程残差、
                                 负曲率方向与鞍点逃逸阈值的实测
===============================  ==========================================================

注：这里的 ``__all__`` 只登记子模块**名字**，不在此 import 它们——与
:mod:`numcore.linalg` 同一做法。并发写章时，别人半成品的子模块不该让整包 import 失败。
"""

from __future__ import annotations

__all__ = [
    "optimize",
]
