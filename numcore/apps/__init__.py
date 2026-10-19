r"""``numcore.apps`` —— 6.1 章的综合仿真内核（把前面各章的零件装成完整求解器）。

本包是 6.1 章规格指定的内核路径（``docs/chapters.yaml`` 的 ``modules: [numcore/apps]``），
三个应用各占一个文件：

============================  ==========================================================
模块                           数学对象
============================  ==========================================================
:mod:`~numcore.apps.cavity`    二维顶盖驱动方腔（流函数--涡量、DST 直接 Poisson、
                               红黑 SOR 稳态迭代、Ghia 基准对拍）
:mod:`~numcore.apps.reaction_diffusion`  灰-斯科特反应扩散（图灵斑图、色散关系）
:mod:`~numcore.apps.shallow_water`       浅水方程（守恒型有限体积、Ritter 精确解）
:mod:`~numcore.apps.ghia`      Ghia--Ghia--Shin (1982) 公开基准表（数据，不是求解器）
============================  ==========================================================

依赖方向：三个应用互不 import；它们都只依赖 ``numpy`` 原语与**前面章节已提交的公开函数**
（2.5 章 :mod:`numcore.transform.fft`、1.5 章 :mod:`numcore.linalg.iterative`），
不 import :mod:`numcore.reference`（P10）。
"""

from __future__ import annotations

__all__ = [
    "cavity",
    "ghia",
    "reaction_diffusion",
    "shallow_water",
]
