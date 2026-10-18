"""numcore —— 交互式数值方法实验室的纯数值核心。

设计约束（见 docs/POLICY.md）：

* 纯函数：无 IO、无全局可变状态、无 print；
* 只依赖 ``numpy`` 作为数组容器与逐元素原语；
* **禁止**使用任何"成品求解器"（``numpy.linalg`` / ``numpy.fft`` / ``scipy`` 等），
  唯一的例外是 :mod:`numcore.reference`——它存在的意义正是给自研实现当对拍靶子。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
