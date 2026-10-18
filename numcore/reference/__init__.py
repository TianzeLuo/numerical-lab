"""``numcore.reference`` —— 对拍基准层，**全仓库唯一允许 import SciPy / numpy.linalg / numpy.fft 的地方**。

存在的意义（`docs/CHAPTERS.md` §0）：

> 不用现成求解器替代自研实现。SciPy / NumPy / PyTorch 只作为对拍参照物存在。

于是本包把"参照物"集中隔离在一个可被 CI 静态检查锁死的目录里：

* :mod:`numcore.reference.dense` —— 稠密线代 / FFT / 求积 / ODE / 插值 / 优化 / 统计的**成熟实现**包装；
* :mod:`numcore.reference.exact` —— 精确解与高精度解（``fractions`` / ``sympy`` / ``mpmath``）；
* :mod:`numcore.reference.analytic` —— 教材上可写出闭式解的参考问题（Runge、Kepler、热方程本征解……）。

**反向禁令**（由 `tools/check_policy.py` 的 P10 强制）：``numcore`` 的其他模块、
``server``、``viz`` 都**不得** import 本包——参照物只能出现在测试与工具里，
否则"自研实现"就会被悄悄替换成"调用参照物"。
"""

from __future__ import annotations

__all__ = ["dense", "exact", "analytic"]
