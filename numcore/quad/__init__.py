"""``numcore.quad`` —— 2.4 章「数值积分」的数值内核。

这一包只回答一个问题：**把 :math:`\\int_a^b f` 算出来，并且知道这个数有多准**。
"知道有多准"是本章真正的主题——三条最容易骗人的地方各自对应一个模块：

1. **函数值准 ≠ 积分准**：求积是加权和，权重里出现负值（或巨值）就会把输入的
   舍入误差放大 :math:`\\sum|w_k|/|\\sum w_k|` 倍（:mod:`~numcore.quad.condition`）；
2. **量到的阶是"求积的阶"还是"被积函数光滑度的阶"**：被积函数不光滑时，
   复合求积的实际误差指数由 :math:`f` 的奇异指数决定，而不是由公式的代数精度决定
   （:mod:`~numcore.quad.newton_cotes` 的 :func:`~numcore.quad.newton_cotes.error_curve`
   与 :mod:`~numcore.quad.integrals` 的光滑性标签）；
3. **误差估计器估的是自己的误差**：自适应细分用嵌套法则的差值当误差，
   在非光滑函数上这个估计可以偏离真误差几个数量级（:mod:`~numcore.quad.adaptive`）。

模块边界（谁负责哪个文件，避免与别的章抢同一个模块）：

==============================  ==================================================================
文件                            数学对象 / 章内角色
==============================  ==================================================================
:mod:`~numcore.quad.integrals`  **被积函数目录**：带解析原函数、光滑性标签、导数界、换元的测试函数
:mod:`~numcore.quad.newton_cotes` 闭 Newton--Cotes 权重与复合梯形 / Simpson / Simpson-3/8、
                                 误差主项（幂次 + 常数 + 导数阶）与**严格误差括号**
:mod:`~numcore.quad.romberg`    Richardson 外推与 Romberg 三角、每一列的实测阶、外推的舍入放大
:mod:`~numcore.quad.gauss`      Gauss--Legendre（节点来自 2.3 章的 :mod:`numcore.interp.orthopoly`）、
                                复合 Gauss、矩精确度曲线（2n−1 精确 / 2n 失效）
:mod:`~numcore.quad.adaptive`   递归细分 + 嵌套 Simpson 对的误差估计、细分树（哪些区间吃掉求值）
:mod:`~numcore.quad.singular`   双指数（tanh--sinh）变换、幂换元、振荡被积函数的分段处理
:mod:`~numcore.quad.condition`  求积的条件：权重和、负权重、Σ|w| 放大输入误差、舍入地板
:mod:`~numcore.quad.rates`      双对数 / 半对数斜率拟合与收敛表（"实测斜率"的唯一出处）
==============================  ==================================================================

依赖方向：本包只依赖 :mod:`numpy` 的原语、2.1 章的 :mod:`numcore.interp.poly`（多项式乘法与
由根构造首一多项式）与 2.3 章的 :mod:`numcore.interp.orthopoly`（Legendre 零点与 Gauss 权重）。
**Legendre 求根不在这里重新实现**——那是 2.3 章的成果，重复实现只会带来两套会漂移的节点。
"""

from __future__ import annotations

__all__ = [
    "adaptive",
    "condition",
    "gauss",
    "integrals",
    "newton_cotes",
    "rates",
    "romberg",
    "singular",
]
