"""``numcore.linalg`` —— 线性代数地基（Part I）。

设计原则（`docs/POLICY.md` P4/P5）：这里只允许 ``numpy`` 的**容器与逐元素原语**
（``ndarray``、切片、ufunc、``sum``/``max`` 等归约），**不允许** ``numpy.linalg``、
``scipy.linalg``、``numpy.fft``、``numpy.random``。要"靶子"就去 :mod:`numcore.reference`。

已实现的模块：

============================  ==========================================================
:mod:`numcore.linalg.lu`      LU 分解（无主元 / 部分主元 / 完全主元）、增长因子、后向误差
:mod:`numcore.linalg.chol`    Cholesky 分解与正定性检查
:mod:`numcore.linalg.sparse`  CSR 容器、结构对称置换、符号消元（填充预测）
:mod:`numcore.linalg.ordering`  RCM / 最小度排序、带宽与剖面、消元即弦化
:mod:`numcore.linalg.cond`    条件数 κ(A)=‖A‖‖A⁻¹‖（1/2/∞/F 范数）、条件数椭球、前向/后向误差界
:mod:`numcore.linalg.iterative` Jacobi / GS / SOR 的迭代矩阵、手写幂法谱半径、模态衰减
:mod:`numcore.linalg.krylov`  CG / PCG、GMRES(m)（Arnoldi + Givens）、Lanczos + Sturm 二分
:mod:`numcore.linalg.precond` Jacobi / SSOR / ILU(0) 预条件子与预条件谱分布
:mod:`numcore.linalg.eigen`   幂法 / 反迭代 / 位移、Hessenberg + QR 迭代、Lanczos、Arnoldi、
                              Rayleigh 商、手写单边 Jacobi SVD、Hermite Jacobi 特征分解、
                              特征值残差、Grcar / Jordan / 伴随 / 三对角等结构化例子矩阵
:mod:`numcore.linalg.pseudospec` ε-伪谱（手写 σ_min 场、marching squares 轮廓）、伪谱半径、
                              Kreiss 常数、瞬态增长 sup_k‖A^k‖、GMRES 与"只看谱"的界
============================  ==========================================================

注：1.1 章新增的 :mod:`numcore.linalg.cond` 只依赖 :mod:`numcore.linalg.lu` 与
:mod:`numcore.utils`（没有引入任何新的外部依赖），命名一律避开被禁属性名
（``cond``/``norm``/``inv``/``det``/``svd``），入口是 ``condition_number`` / ``condition_1`` 等。

1.4 章新增的两个模块同样只用上面的原语：:mod:`~numcore.linalg.eigen` 里的
``one_sided_jacobi_svd`` / ``hermitian_jacobi_eigh`` 是**手写**的 Jacobi 旋转，
:mod:`~numcore.linalg.pseudospec` 里的 ``complex_lu`` 是**手写**的部分选主元复数 LU
——三者的存在理由都是"P3 不许把成品算法当结论"。
"""

from __future__ import annotations

__all__ = [
    "lu",
    "chol",
    "sparse",
    "ordering",
    "cond",
    "iterative",
    "krylov",
    "precond",
    "eigen",
    "pseudospec",
]
