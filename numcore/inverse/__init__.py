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
:mod:`numcore.inverse.regularize`  5.2 章：反问题与正则化（一）—— Tikhonov / TSVD /
                                 Landweber（谱形式与矩阵迭代两种写法）/ 全变差（原始-对偶）、
                                 滤波因子与有效自由度、离散 Picard 诊断、
                                 偏差-方差分解（解析式与蒙特卡洛）、
                                 可分离二维反卷积与简化 CT 的显式投影矩阵
:mod:`numcore.inverse.lcurve`  5.2 章：反问题与正则化（二）—— L 曲线与**三个**拐点判据
                                 （最大曲率 / 两段直线拟合交点 / 离散二阶差分）、
                                 GCV、偏差原则、oracle，以及"用错判据误差放大多少倍"的
                                 横向比较
:mod:`numcore.inverse.adjoint`  5.3 章：伴随方法与灵敏度分析（一）—— 离散伴随（RK 的
                                 转置表反向递推）/ 连续伴随（同一网格上的反向 ODE）/
                                 分块检查点重算 / 梯度检验的 V 形扫描（两段斜率与实测最优
                                 eps）/ 代价三点账 / 热方程 theta 格式的伴随（M^T 与
                                 端点层）
:mod:`numcore.inverse.sensitivity`  5.3 章：伴随方法与灵敏度分析（二）—— 灵敏度方向与
                                 绩效弹性、参数平面上的梯度场、影响场的宽度律
                                 （sigma_0^2 + 2*alpha*tau）与质心守恒、半离散精确解对照、
                                 伴随反向放大率随阻尼过零换向
:mod:`numcore.inverse.uq`  5.4 章：不确定性量化与全局敏感性 —— 蒙特卡洛与
                                 置信区间覆盖率、低偏差点集（Sobol' 方向向量与
                                 (0,m,2)-网、线性/数字置乱、格点、Halton）、Koksma-Hlawka
                                 上界与变差、Sobol' 方差分解（pick-freeze 与相关输入的
                                 不唯一性）、多项式混沌（Legendre/Hermite 基、投影与
                                 回归、系数谱与 Sobol' 指数、过拟合与外推失效）
===============================  ==========================================================

注：5.3 章的**正演**沿用 Part III/IV 的 :mod:`numcore.ode.rk`（表与 FSAL 记账）与
:mod:`numcore.pde.heat`（theta 格式与离散正弦特征值），本层只加伴随与灵敏度那一段。

注：这里的 ``__all__`` 只登记子模块**名字**，不在此 import 它们——与
:mod:`numcore.linalg` 同一做法。并发写章时，别人半成品的子模块不该让整包 import 失败。
"""

from __future__ import annotations

__all__ = [
    "optimize",
    "regularize",
    "lcurve",
    "adjoint",
    "sensitivity",
    "uq",
]
