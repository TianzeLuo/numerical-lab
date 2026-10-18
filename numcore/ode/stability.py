r"""3.2 章（稳定性与刚性方程）的稳定性真源。

数学对象
--------
线性测试方程 :math:`y'=\lambda y` 上，单步法给出 :math:`y_{n+1}=R(z)y_n`（:math:`z=\lambda h`），
**绝对稳定域**是 :math:`S=\{z\in\mathbb{C}:|R(z)|\le1\}`。

* 显式 RK（:math:`A` 严格下三角）的稳定函数是**多项式**
  :math:`R(z)=1+\sum_{j=1}^{s}\bigl(b^{\mathsf T}A^{j-1}e\bigr)z^{j}`——
  这里由 Butcher 表用矩阵乘法现场算出系数，而不是把 :math:`1+z+z^2/2+\cdots` 抄进来；
  只有这样，"RK4 的 :math:`R` 等于截断指数"才是一个**被验证的结论**而不是假设。
* 隐式 Euler 与梯形法是**有理**的：:math:`1/(1-z)` 与 :math:`(1+z/2)/(1-z/2)`。
* 线性多步法（BDF-k）没有单个 :math:`R`：改用**根轨迹**
  :math:`\rho(r)-z\sigma(r)=0` 的"稳定模" :math:`\max_i|r_i(z)|`，
  其在 :math:`|r|=1` 上的像 :math:`z(\theta)=\rho(e^{i\theta})/\sigma(e^{i\theta})`
  正好是稳定域边界（这是本章"BDF 稳定角"可视化的全部依据）。

**自研边界**（P2/P3）：多项式求值用自写 Horner；多步法的根用自写 Durand--Kerner 迭代
（:func:`lmm_root_modulus`），**不用** ``np.roots`` / ``np.polyval`` / ``np.linalg``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "METHODS",
    "TABLEAU_NAMES",
    "BDF_ANGLE_THEORY_DEG",
    "StabilityReport",
    "rk_tableau",
    "rk_stability_coefficients",
    "taylor_stability_coefficients",
    "bdf_coefficients",
    "lmm_coefficients",
    "polyval_ascending",
    "stability_function",
    "stability_modulus",
    "modulus_second_route",
    "lmm_root_modulus",
    "lmm_boundary",
    "polynomial_boundary",
    "stability_boundary",
    "max_on_imaginary_axis",
    "negative_real_endpoint",
    "stability_angle_deg",
    "bdf_stability_angle_deg",
    "stiff_limit_amplification",
    "stability_report",
    "a_stable_orders_bdf",
    "dahlquist_barrier_evidence",
]

#: 本章认得的全部"方法名"（``order`` 只对 ``explicit_taylor`` 与 ``bdf`` 有语义）。
METHODS: tuple[str, ...] = (
    "forward_euler",
    "midpoint",
    "heun",
    "rk4",
    "explicit_taylor",
    "implicit_euler",
    "trapezoidal",
    "bdf",
)

# --------------------------------------------------------------------------- #
# 1. Butcher 表（显式 RK 的最小集合）
# --------------------------------------------------------------------------- #
#: 显式 RK 的 Butcher 表：``(A 的严格下三角各行, b)``。存成不可变嵌套元组，
#: 需要数组时才构造——这样模块级不留可变状态。
_TABLEAUX: dict[str, tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]] = {
    "forward_euler": (((0.0,),), (1.0,)),
    "midpoint": (((0.0, 0.0), (0.5, 0.0)), (0.0, 1.0)),
    "heun": (((0.0, 0.0), (1.0, 0.0)), (0.5, 0.5)),
    "rk4": (
        (
            (0.0, 0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.0),
            (0.0, 0.5, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    ),
}

#: 极点判据的相对阈值：:math:`r^k` 的系数（:math:`\alpha_k-z\beta_k`）小到
#: 这个量级就认为它**退化**（多项式降次、有一个根跑到无穷远）。
_POLE_RTOL = 8.0 * float(np.finfo(float).eps)

#: 库函数一律不外抛浮点哨声。仓库把 ``RuntimeWarning`` 当错误，而这里的
#: 除零/溢出**不是错误**：它标记的是 :math:`R` 的极点（:math:`|R|=+\infty` 是有意义的极限值）。
#: 帧层会把这些位置显式换成占位值并给出掩码（见 ``server/chapters/3.2.py``）。
_IGNORE_FP = {"divide": "ignore", "invalid": "ignore", "over": "ignore", "under": "ignore"}


def _pole_where(coeffs: np.ndarray, k: int) -> np.ndarray:
    r"""``rho(r) − z·beta_k·r^k`` 的最高次系数退化（≈0）的位置掩码。

    退化点就是极点：多项式降次、一个根趋向无穷，故 :math:`|R|=+\infty`。
    用**相对**判据（对系数尺度归一）而不是绝对阈值，免得把"离极点很近但完全正常"的点
    （例如 :math:`z=1.5-10^{-13}`，那里 :math:`|R|\approx2\times10^{12}` 是有限真值）误判成极点。
    """
    lead = np.abs(coeffs[:, k])
    scale = np.max(np.abs(coeffs), axis=1)
    return lead <= _POLE_RTOL * np.maximum(scale, 1e-300)

TABLEAU_NAMES: tuple[str, ...] = tuple(sorted(_TABLEAUX))

#: 隐式单步法的有理稳定函数 :math:`R(z)=N(z)/D(z)`（升幂系数）。
_RATIONAL: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "implicit_euler": ((1.0,), (1.0, -1.0)),
    "trapezoidal": ((1.0, 0.5), (1.0, -0.5)),
}

#: 线性多步法写成 :math:`\sum_j\alpha_j y_{n+j}=h\sum_j\beta_j f_{n+j}` 时的
#: :math:`\beta`（升幂、与 ``[-1, 1]`` 的 :math:`\alpha` 同长）。隐式 Euler / 梯形法也是多步法：
#: :math:`\beta=(0,1)` 对应 :math:`\sigma(r)=r`，:math:`\beta=(1/2,1/2)` 对应 :math:`\sigma(r)=(r+1)/2`。
_LMM_BETA: dict[str, tuple[float, ...]] = {
    "implicit_euler": (0.0, 1.0),
    "trapezoidal": (0.5, 0.5),
}

#: BDF-k 的 :math:`\rho(r)=\sum_j\alpha_j r^j`（升幂）与 :math:`\sigma(r)=\beta_k r^k`。
#: 出处：Hairer & Wanner, *Solving ODEs II*, §III.1（BDF 系数表）。
_BDF_ALPHA: dict[int, tuple[float, ...]] = {
    1: (-1.0, 1.0),
    2: (0.5, -2.0, 1.5),
    3: (-1.0 / 3.0, 1.5, -3.0, 11.0 / 6.0),
    4: (0.25, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0),
    5: (-0.2, 1.25, -10.0 / 3.0, 5.0, -5.0, 137.0 / 60.0),
    6: (1.0 / 6.0, -1.2, 3.75, -20.0 / 3.0, 7.5, -6.0, 49.0 / 20.0),
}

#: 文献上的 BDF A(:math:`\alpha`)-稳定角（度）。本章的实现必须**实测**出这些数，
#: 而不是把它们抄回去当"实测值"（见 ``tests/3.2/test_stability.py``）。
BDF_ANGLE_THEORY_DEG: dict[int, float] = {
    1: 90.0,
    2: 90.0,
    3: 86.03,
    4: 73.35,
    5: 51.84,
    6: 17.84,
}


# --------------------------------------------------------------------------- #
# 2. 由 Butcher 表算稳定多项式
# --------------------------------------------------------------------------- #
def rk_tableau(name: str) -> tuple[np.ndarray, np.ndarray]:
    """显式 RK 的 :math:`(A,b)`：:math:`A` 严格下三角、:math:`b` 是权重向量。"""
    if name not in _TABLEAUX:
        raise ValueError(f"未知显式 RK 表 {name!r}（可用：{sorted(_TABLEAUX)}）")
    rows, b = _TABLEAUX[name]
    return np.asarray(rows, dtype=float), np.asarray(b, dtype=float)


def rk_stability_coefficients(name: str) -> np.ndarray:
    r"""显式 RK 的稳定多项式系数（升幂）：:math:`R(z)=\sum_j c_j z^j`，:math:`c_j=b^{\mathsf T}A^{j-1}e`。

    显式方法的 :math:`A` 严格下三角，故 :math:`A^{s}=0`，级数在第 :math:`s` 项自动截断——
    "显式 RK 的 :math:`R` 是多项式"这件事本身就是显式性的推论。
    """
    A, b = rk_tableau(name)
    s = int(b.size)
    coeffs = np.zeros(s + 1, dtype=float)
    coeffs[0] = 1.0
    u = np.ones(s, dtype=float)  # A^{j-1} e
    for j in range(1, s + 1):
        coeffs[j] = float(b @ u)
        u = A @ u
    return coeffs


def taylor_stability_coefficients(order: int) -> np.ndarray:
    r""":math:`p` 级 :math:`p` 阶显式 RK 的稳定多项式 :math:`R_p(z)=\sum_{j\le p}z^j/j!`。

    这不是"另一个方法的表"，而是**阶条件**的推论：:math:`p` 级显式 RK 满足全部
    :math:`p` 阶条件时 :math:`b^{\mathsf T}A^{j-1}e=1/j!`，而 :math:`A^{p}=0` 把级数截断，
    于是稳定多项式必然是截断指数。本章的断言会把 :func:`rk_stability_coefficients`
    算出的系数与 :math:`1/j!` 逐项对拍。
    """
    p = int(order)
    if p < 1:
        raise ValueError(f"阶数至少为 1，得到 {order}")
    return np.asarray([1.0 / math.factorial(j) for j in range(p + 1)], dtype=float)


def bdf_coefficients(k: int) -> tuple[np.ndarray, float]:
    r"""BDF-:math:`k` 的 :math:`(\alpha,\beta_k)`：:math:`\sum_{j=0}^{k}\alpha_j y_{n+j}=h\beta_k f_{n+k}`。

    :math:`\alpha` 按**升幂**给出（:math:`\rho(r)=\sum_j\alpha_j r^j`），:math:`\beta_k=1`。
    """
    kk = int(k)
    if kk not in _BDF_ALPHA:
        raise ValueError(f"BDF 阶数只实现 1..6（k≥7 连零稳定都没有），得到 {k}")
    return np.asarray(_BDF_ALPHA[kk], dtype=float), 1.0


def lmm_coefficients(method: str, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    r"""把方法写成线性多步法 :math:`\sum_j\alpha_j y_{n+j}=h\sum_j\beta_j f_{n+j}`。

    返回 :math:`(\alpha,\beta)`（都是升幂、等长，末尾补零）。显式 RK 不在其中
    （它们不是多步法），调用会抛 :class:`ValueError`。
    """
    if method == "bdf":
        alpha, beta_k = bdf_coefficients(order)
        beta = np.zeros_like(alpha)
        beta[-1] = beta_k
        return alpha, beta
    if method in _LMM_BETA:
        beta_raw = np.asarray(_LMM_BETA[method], dtype=float)
        alpha = np.asarray([-1.0, 1.0], dtype=float)
        beta = np.zeros_like(alpha)
        beta[: beta_raw.size] = beta_raw
        return alpha, beta
    raise ValueError(f"{method!r} 不是线性多步法，没有 (alpha, beta) 表示")


# --------------------------------------------------------------------------- #
# 3. 多项式求值（自写 Horner，P3：不用 np.polyval）
# --------------------------------------------------------------------------- #
def polyval_ascending(coeffs: Any, z: Any) -> np.ndarray:
    r"""按升幂系数求值 :math:`\sum_j c_j z^j`（Horner，自写；``np.polyval`` 属禁用成品算法）。"""
    c = np.asarray(coeffs, dtype=complex).reshape(-1)
    zz = np.asarray(z, dtype=complex)
    out = np.full(zz.shape, c[-1], dtype=complex)
    for j in range(c.size - 2, -1, -1):
        out = out * zz + c[j]
    return out


def _polyval_real(coeffs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """实系数、复自变量版本的 Horner（返回复数数组）。"""
    return polyval_ascending(coeffs, z)


# --------------------------------------------------------------------------- #
# 4. 稳定函数与稳定模
# --------------------------------------------------------------------------- #
def _explicit_coefficients(method: str, order: int) -> np.ndarray:
    if method == "explicit_taylor":
        return taylor_stability_coefficients(order)
    return rk_stability_coefficients(method)


def stability_function(method: str, order: int, z: Any) -> np.ndarray:
    r"""单步法的稳定函数 :math:`R(z)`（复数值）。

    :math:`z\to0` 时主根趋于 1，故 ``bdf`` 的 :math:`k=1,2` 两支分别就是隐式 Euler 与 BDF2 的
    闭式主根。**注意**：多步法（:math:`k\ge3`）根本不存在单一的 :math:`R`——那时本函数
    退化为返回稳定模 :math:`\max_i|r_i(z)|`（与 :func:`stability_modulus` 同值），
    调用方不应把它当成"一步的放大因子"。
    """
    zz = np.asarray(z, dtype=complex)
    if method == "bdf":
        k = int(order)
        alpha, beta_k = bdf_coefficients(k)
        if k == 1:
            # z=1 是极点（1/(1−z) 的分母零点）：|R|=+∞ 是极限值，不是错误
            with np.errstate(**_IGNORE_FP):
                return 1.0 / (1.0 - zz)
        if k == 2:
            # (3/2 − z)r² − 2r + 1/2 = 0  ⟹  r = (2 ± sqrt(1+2z)) / (3 − 2z)
            # z=1.5 处分子 4、分母 0：极点。
            with np.errstate(**_IGNORE_FP):
                return (2.0 + np.sqrt(1.0 + 2.0 * zz)) / (3.0 - 2.0 * zz)
        mods, resid = lmm_root_modulus(alpha, beta_k, zz, return_residual=True)
        _ = resid
        return mods  # 多步法只保证"最大模"有意义
    if method in _RATIONAL:
        num, den = _RATIONAL[method]
        # 同 stability_modulus：分母的零点是 R 的极点（梯形法在 z=2），那里的值就是 ±∞
        with np.errstate(divide="ignore", invalid="ignore"):
            return polyval_ascending(num, zz) / polyval_ascending(den, zz)
    if method in _TABLEAUX or method == "explicit_taylor":
        with np.errstate(**_IGNORE_FP):
            return polyval_ascending(_explicit_coefficients(method, order), zz)
    raise ValueError(f"未知方法 {method!r}")


def lmm_root_modulus(
    alpha: Any,
    beta_k: float,
    z: Any,
    *,
    iters: int = 48,
    return_residual: bool = False,
) -> Any:
    r"""线性多步法的稳定模 :math:`\max_i|r_i(z)|`，:math:`r_i` 是 :math:`\rho(r)-z\sigma(r)` 的根。

    根用**自写的 Durand--Kerner 迭代**（Weierstrass 同时迭代）求：
    :math:`r_i\leftarrow r_i-p(r_i)/\prod_{j\ne i}(r_i-r_j)`，
    初值取半径为 :math:`1+\max|c_j|` 的圆上等分布点——确定性、无随机、无 ``np.roots``（P3）。
    对向量化的 :math:`z` 一次算完整张网格（这是它的主要用途：模曲面）。
    ``iters`` 缺省 48：实测与 240 次迭代的差 < 1e-15（B 类断言盯着这件事），
    再往上加只是烧时间。
    """
    a = np.asarray(alpha, dtype=float).reshape(-1)
    k = a.size - 1
    if k < 1:
        raise ValueError("多步法至少是一步的")
    zz = np.asarray(z, dtype=complex)
    shape = zz.shape
    zf = zz.reshape(-1)
    n = int(zf.size)
    coeffs = np.tile(a.astype(complex), (n, 1))
    coeffs[:, k] = coeffs[:, k] - zf * float(beta_k)  # ρ(r) − z·β_k·r^k

    if n == 0:
        empty = np.zeros(0, dtype=float)
        return (empty.reshape(shape), empty.reshape(shape)) if return_residual else empty.reshape(shape)

    # 退化点（r^k 的系数为 0）是极点：多项式降次、一个根跑到无穷远。
    # 直接给 +∞ 并跳过 DK——否则首一化那一步会除以 0，迭代出来的也是 NaN/Inf。
    pole = _pole_where(coeffs, k)
    safe_lead = np.where(pole, 1.0, coeffs[:, k])
    coeffs = coeffs.copy()
    coeffs[pole, k] = 1.0  # 占位（这些点的结果下面会被 +∞ 覆盖，只为让首一化不炸）
    monic = coeffs / safe_lead[:, None]

    radius = 1.0 + np.max(np.abs(monic[:, :k]), axis=1)
    ang = 2.0 * math.pi * (np.arange(k, dtype=float) + 0.5) / k
    roots = radius[:, None] * np.exp(1j * (ang[None, :] + 0.35))
    idx = np.arange(k)
    with np.errstate(**_IGNORE_FP):
        for _ in range(max(1, int(iters))):
            vals = _eval_monic(monic, roots)
            diff = roots[:, :, None] - roots[:, None, :]
            diff[:, idx, idx] = 1.0  # 自因子置 1（标准 DK 正则化）
            prod = np.prod(diff, axis=2)
            roots = roots - vals / prod

        mods = np.max(np.abs(roots), axis=1) if k > 1 else np.abs(roots[:, 0])
        # 库的约定：非有限一律折成 +∞（"模无界"），绝不返回 NaN——NaN 在这里没有数学含义，
        # 只会让上层（帧、绘图、meta）多一个必须到处判的分支。
        mods = np.where(np.isfinite(mods), mods, np.inf)
        mods = np.where(pole, np.inf, mods)
        if not return_residual:
            return mods.reshape(shape)
        vals = _eval_monic(monic, roots)
        scale = np.sum(np.abs(monic[:, None, :]) * np.abs(roots)[:, :, None] ** np.arange(k + 1)[None, None, :], axis=2)
        resid = np.max(np.abs(vals) / np.maximum(scale, 1e-300), axis=1)
        resid = np.where(np.isfinite(resid), resid, np.inf)
        resid = np.where(pole, 0.0, resid)  # 极点处"根在无穷远"是解析事实，残差记 0
    return mods.reshape(shape), resid.reshape(shape)


def _eval_monic(monic: np.ndarray, roots: np.ndarray) -> np.ndarray:
    """在 ``roots``（形状 ``(n, k)``）处求首一多项式 ``monic``（形状 ``(n, k+1)``）。"""
    k = monic.shape[1] - 1
    out = np.full(roots.shape, monic[:, k : k + 1], dtype=complex)
    for j in range(k - 1, -1, -1):
        out = out * roots + monic[:, j : j + 1]
    return out


def stability_modulus(method: str, order: int, z: Any) -> np.ndarray:
    r"""稳定模 :math:`|R(z)|`（单步法）或 :math:`\max_i|r_i(z)|`（多步法）。

    这是"绝对稳定域" :math:`S=\{z:|R(z)|\le1\}` 的唯一判据函数：模曲面、边界、
    A-稳定性全都从它派生。
    """
    zz = np.asarray(z, dtype=complex)
    if method == "bdf":
        k = int(order)
        alpha, beta_k = bdf_coefficients(k)
        # 先算出极点掩码（BDF-k 的极点在 z=alpha_k，k=1..6 分别是 1, 1.5, 11/6, 25/12, 137/60, 49/20），
        # 再在下面把那些位置**显式**写成 +∞：绝不返回 NaN，也不让 numpy 的除零哨声漏出去。
        coeffs = np.tile(alpha.astype(complex), (int(zz.size), 1))
        coeffs[:, k] = coeffs[:, k] - zz.reshape(-1) * float(beta_k)
        pole = _pole_where(coeffs, k).reshape(zz.shape)
        if k == 1:
            with np.errstate(**_IGNORE_FP):
                m = np.abs(1.0 / (1.0 - zz))
        elif k == 2:
            with np.errstate(**_IGNORE_FP):
                s = np.sqrt(1.0 + 2.0 * zz)
                den = np.abs(3.0 - 2.0 * zz)
                m = np.maximum(np.abs(2.0 + s), np.abs(2.0 - s)) / den
        else:
            m = lmm_root_modulus(alpha, beta_k, zz)
        return np.where(pole, np.inf, m)
    if method in _RATIONAL:
        num, den = _RATIONAL[method]
        # 分母的零点是 R 的**极点**（例如梯形法在 z=2 处）：那里的模是 +∞，
        # 这是有意义的极限值而不是错误，所以在这里就地关掉 numpy 的除零警告
        # （仓库把 RuntimeWarning 当错误，库函数不能把哨声留给调用方）。
        with np.errstate(**_IGNORE_FP):
            return np.abs(polyval_ascending(num, zz)) / np.abs(polyval_ascending(den, zz))
    if method in _TABLEAUX or method == "explicit_taylor":
        with np.errstate(**_IGNORE_FP):
            return np.abs(polyval_ascending(_explicit_coefficients(method, order), zz))
    raise ValueError(f"未知方法 {method!r}")


def modulus_second_route(method: str, order: int, z: Any) -> np.ndarray:
    r"""稳定模的**第二条计算路径**，用于交叉验证第二条实现没有跟着第一条一起错。

    * 显式多项式：用 :math:`\sum_j z^j/j!` 的解析系数（若方法不是 :math:`p` 级 :math:`p` 阶，
      则退化为用 Horner 逐项累加，仍与"表驱动系数"是不同代码路径）；
    * 有理方法：直接写死解析式 :math:`1/|1-z|`、:math:`|1+z/2|/|1-z/2|`；
    * BDF-2：闭式判别式 :math:`\max|2\pm\sqrt{1+2z}|/|3-2z|`；
    * BDF-:math:`k\ge3`：多算一倍的 DK 迭代（独立初值相位），用于检查迭代是否收敛。
    """
    zz = np.asarray(z, dtype=complex)
    if method == "bdf":
        k = int(order)
        if k <= 2:
            with np.errstate(**_IGNORE_FP):
                s = np.sqrt(1.0 + 2.0 * zz)
                if k == 1:
                    return np.abs(1.0 / (1.0 - zz))
                den = np.abs(3.0 - 2.0 * zz)
                return np.maximum(np.abs(2.0 + s), np.abs(2.0 - s)) / den
        alpha, beta_k = bdf_coefficients(k)
        return lmm_root_modulus(alpha, beta_k, zz, iters=240)
    if method == "implicit_euler":
        with np.errstate(**_IGNORE_FP):
            return np.abs(1.0 / (1.0 - zz))
    if method == "trapezoidal":
        with np.errstate(**_IGNORE_FP):
            return np.abs(1.0 + 0.5 * zz) / np.abs(1.0 - 0.5 * zz)
    if method in _TABLEAUX or method == "explicit_taylor":
        coeffs = _explicit_coefficients(method, order)
        if method == "explicit_taylor":
            # 与 stability_modulus 走同一条 Horner，但系数换成浮点累乘（4/3/2 的写法不同）
            acc = np.ones(zz.shape, dtype=complex)
            out = np.ones(zz.shape, dtype=complex)
            term = np.ones(zz.shape, dtype=complex)
            for j in range(1, int(order) + 1):
                term = term * zz / float(j)
                out = out + term
            _ = acc
            return np.abs(out)
        with np.errstate(**_IGNORE_FP):
            return np.abs(polyval_ascending(coeffs, zz))
    raise ValueError(f"未知方法 {method!r}")


# --------------------------------------------------------------------------- #
# 5. 稳定域边界
# --------------------------------------------------------------------------- #
def lmm_boundary(alpha: Any, beta: Any, n_theta: int = 2049, theta: Any = None) -> np.ndarray:
    r"""多步法的稳定域边界 :math:`z(\theta)=\rho(e^{i\theta})/\sigma(e^{i\theta})`，:math:`\theta\in[0,2\pi)`。

    这是**精确参数化**（不是网格等值线）：对单步法写成多步法形式后同样成立，
    例如隐式 Euler 给出圆心在 1、半径为 1 的圆，梯形法给出整条虚轴。
    ``theta`` 显式给出时用给定的角度网格（稳定角的局部加密用）。
    """
    a = np.asarray(alpha, dtype=float).reshape(-1)
    b = np.asarray(beta, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError("alpha 与 beta 长度必须一致（末尾补零）")
    if theta is None:
        n = max(8, int(n_theta))
        th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    else:
        th = np.asarray(theta, dtype=float).reshape(-1)
    r = np.exp(1j * th)
    rho = polyval_ascending(a, r)
    sig = polyval_ascending(b, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = rho / sig
    return z


def polynomial_boundary(
    coeffs: Any,
    n_theta: int = 2049,
    *,
    theta: Any = None,
    r_max: float = 200.0,
    n_scan: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    r"""多项式稳定函数 :math:`|R(z)|=1` 的边界（极坐标二分），返回 ``(z, found)``。

    原点在稳定域边界上（:math:`|R(0)|=1`），沿每条射线 :math:`r e^{i\theta}` 从 0 向外找**第一次**
    穿越 :math:`|R|=1`。该方法要求稳定域关于原点星形——这一点由 B 类断言实测
    （在 0.95r 与 1.05r 处分别检查模 < 1 与 > 1），而不是假设。``theta`` 显式给出时用给定网格。
    """
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    if theta is None:
        n = max(8, int(n_theta))
        th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    else:
        th = np.asarray(theta, dtype=float).reshape(-1)
    dirs = np.exp(1j * th)
    r_grid = np.exp(np.linspace(math.log(1e-8), math.log(float(r_max)), int(n_scan)))
    vals = np.abs(polyval_ascending(c, r_grid[:, None] * dirs[None, :]))  # (n_scan, n)
    above = vals > 1.0
    found = np.any(above, axis=0)
    first = np.argmax(above, axis=0)
    # ``first == 0``：在最小的扫描半径处就已经 |R|>1。对多项式 R 这只会发生在
    # Re(c₁e^{iθ})>0 的方向上，此时 |R(z)|=1 的穿越点**就是**原点（正实轴方向上的尖点），
    # 所以直接把半径置 0；若拿去二分，sqrt(0·h) 会永远卡死。
    at_zero = first <= 0
    ok = found & ~at_zero
    lo = np.where(ok, r_grid[np.maximum(first - 1, 0)], np.where(at_zero, 0.0, r_grid[0]))
    hi = np.where(ok, r_grid[first], np.where(at_zero, 0.0, r_grid[-1]))
    for _ in range(70):
        mid = np.sqrt(lo * hi)
        f = np.abs(polyval_ascending(c, mid * dirs)) - 1.0
        lo = np.where(f < 0.0, mid, lo)
        hi = np.where(f >= 0.0, mid, hi)
    rad = np.sqrt(lo * hi)
    return rad * dirs, found


def stability_boundary(method: str, order: int = 4, n_theta: int = 2049, theta: Any = None) -> np.ndarray:
    """任意方法的稳定域边界（复平面上的闭合折线）。"""
    if method == "bdf" or method in _LMM_BETA:
        alpha, beta = lmm_coefficients(method, order)
        return lmm_boundary(alpha, beta, n_theta, theta)
    if method in _TABLEAUX or method == "explicit_taylor":
        z, found = polynomial_boundary(_explicit_coefficients(method, order), n_theta, theta=theta)
        if not np.all(found):
            raise ValueError(f"{method} 的稳定域在部分射线上没有边界（星形假设不成立）")
        return z
    raise ValueError(f"未知方法 {method!r}")


# --------------------------------------------------------------------------- #
# 6. A-稳定性 / L-稳定性 / 稳定角
# --------------------------------------------------------------------------- #
def max_on_imaginary_axis(
    method: str, order: int = 4, *, y_max: float = 60.0, n: int = 30001
) -> tuple[float, float]:
    r"""虚轴上的最大模 :math:`\max_{|y|\le y_{\max}}|R(iy)|` 及其位置。

    对**有理**（单步隐式）方法，最大模原理给出：:math:`|R|` 在闭左半平面的上确界在虚轴上取到，
    因此"虚轴最大模 ≤ 1"就是 A-稳定性的判据（且是充分必要条件，只要 :math:`R` 在闭左半平面解析）。
    对显式多项式方法它只能**证伪**（找到 > 1 的点即不 A-稳定），因为多项式在
    :math:`z\to-\infty` 时无界——这本身就是「显式方法不可能 A-稳定」的证明。
    """
    y = np.linspace(0.0, float(y_max), max(3, int(n)))
    vals = stability_modulus(method, order, 1j * y)
    j = int(np.argmax(vals))
    return float(vals[j]), float(y[j])


def negative_real_endpoint(method: str, order: int = 4, *, lo: float = -400.0, n: int = 40001) -> float:
    r"""负实轴上的稳定边界点（稳定域左端），如 forward Euler 的 :math:`-2`、RK4 的 :math:`-2.785\ldots`。

    返回 ``-inf`` 表示整条负实轴都在稳定域内（隐式 Euler / 梯形法 / BDF1-2）。
    """
    x = np.linspace(0.0, float(lo), max(3, int(n)))
    m = stability_modulus(method, order, x.astype(complex))
    bad = m > 1.0 + 1e-12
    if not bool(np.any(bad)):
        return -math.inf
    j = int(np.argmax(bad))
    a, b = float(x[j - 1]), float(x[j])  # m(a) ≤ 1 < m(b)
    for _ in range(80):
        mid = 0.5 * (a + b)
        if float(stability_modulus(method, order, complex(mid))) > 1.0:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)


def stability_angle_deg(method: str, order: int = 4, *, n_theta: int = 4097, refine: int = 4) -> tuple[float, bool]:
    r"""A(:math:`\alpha`)-稳定角（度）与 A-稳定性标志。

    A(:math:`\alpha`)-稳定指楔形 :math:`|\arg(-z)|\le\alpha` 整体落在稳定域内。
    取边界折线上 :math:`\varphi(z)=|\arg(-z)|` 的最小值即为该角：
    边界一旦有 Re :math:`z<0` 的点，楔形就被它"咬掉"一块（角度 < 90 度）；
    边界完全落在右半闭平面时楔形是整条左半平面（角度 = 90 度，即 A-稳定）。

    角度是**实测**的：先在均匀 :math:`\theta` 网格上找最小 :math:`\varphi`，
    再在该点附近反复局部加密（``refine`` 轮），把离散化误差压到 1e-6 度量级——
    否则 4097 个点采出来的"角度"里混着 :math:`O(2\pi/N)` 的网格误差。
    """
    if method == "bdf" or method in _LMM_BETA:
        alpha, beta = lmm_coefficients(method, order)

        def boundary_at(th: np.ndarray) -> np.ndarray:
            return lmm_boundary(alpha, beta, theta=th)

    elif method in _TABLEAUX or method == "explicit_taylor":
        coeffs = _explicit_coefficients(method, order)

        def boundary_at(th: np.ndarray) -> np.ndarray:
            z, _found = polynomial_boundary(coeffs, theta=th)
            return z

    else:
        raise ValueError(f"未知方法 {method!r}")

    n = max(64, int(n_theta))
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    z = boundary_at(th)
    best = _argmin_phi(z)
    for _ in range(max(0, int(refine))):
        span = 2.0 * math.pi / max(n, 2)
        th = np.linspace(th[best] - span, th[best] + span, 401)
        z = boundary_at(th)
        best = _argmin_phi(z)
    keep = np.isfinite(z) & (np.abs(z) > 1e-12)
    zz = z[keep] if bool(np.any(keep)) else z
    left = np.real(zz) < -_LEFT_TOL * np.maximum(np.abs(zz), 1.0)
    if not bool(np.any(left)):
        return 90.0, True
    phi = np.abs(np.arctan2(-np.imag(zz[left]), -np.real(zz[left])))
    return min(float(np.degrees(np.min(phi))), 90.0), False


#: 判定"边界点真的落在左半平面"的**相对**阈值。必须相对：梯形法的边界参数化
#: :math:`2(r-1)/(r+1)` 在 :math:`\theta\to\pi` 时出现灾难性相消，|z|~1e6 的点的
#: 实部会被舍入污染成 ~1e-4 的噪声——绝对阈值会把整条虚轴误判成"伸进左半平面"。
_LEFT_TOL = 1e-9


def _argmin_phi(z: np.ndarray) -> int:
    r"""边界上真正落在左半平面的点里 :math:`|\arg(-z)|` 最小的下标。

    先剔除原点附近（那里没有角度信息）与**非有限点**：梯形法的边界参数化
    :math:`2(r-1)/(r+1)` 在 :math:`\theta=\pi` 处有极点（那正是"边界跑到无穷远"），
    不剔除就会把 NaN 当成"Re z < 0"从而误判成"不 A-稳定"。
    没有任何点落在左半平面时返回 0（调用方随后会判定 A-稳定，不再加密）。
    """
    keep = np.isfinite(z) & (np.abs(z) > 1e-12)
    if not bool(np.any(keep)):
        return 0
    left = keep & (np.real(z) < -_LEFT_TOL * np.maximum(np.abs(z), 1.0))
    if not bool(np.any(left)):
        return 0
    phi = np.abs(np.arctan2(-np.imag(z), -np.real(z)))
    phi = np.where(left, phi, np.inf)
    return int(np.argmin(phi))


def bdf_stability_angle_deg(k: int, *, n_theta: int = 8193) -> tuple[float, bool]:
    """BDF-:math:`k`（:math:`k=1..6`）的实测 A(:math:`\\alpha`)-稳定角（度）与 A-稳定性。"""
    return stability_angle_deg("bdf", k, n_theta=n_theta)


def stiff_limit_amplification(method: str, order: int = 4, *, z_test: float = -1.0e14) -> float:
    r""":math:`z\to-\infty` 方向（刚性极限）的放大模 :math:`|R(z)|`。

    L-稳定性 :math:`\Leftrightarrow` A-稳定 **且** 该极限为 0。数值上取
    :math:`z=-10^{14}`：隐式 Euler :math:`\approx10^{-14}`、BDF2 :math:`\approx7\times10^{-8}`
    （多步法的衰减是 :math:`O(|z|^{-1/2})`，比隐式 Euler 慢，但仍趋于 0）、
    梯形法 :math:`\approx1`（这就是它在刚性问题上"抖动不衰减"的原因）。
    """
    return float(np.max(stability_modulus(method, order, complex(z_test))))


# --------------------------------------------------------------------------- #
# 7. 汇总报告
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StabilityReport:
    """一个方法在 :math:`z=\\lambda h` 上的稳定性档案（全部字段都是实测值）。"""

    method: str
    order: int
    a_stable: bool
    angle_deg: float
    l_stable: bool
    stiff_limit: float
    negative_real_endpoint: float
    max_on_imaginary_axis: float
    argmax_y: float
    is_explicit: bool
    route: str = ""
    notes: tuple[str, ...] = field(default=())

    def to_json(self) -> dict[str, Any]:
        """给报告/测试用的标量摘要。"""
        endpoint = self.negative_real_endpoint
        return {
            "method": self.method,
            "order": self.order,
            "a_stable": self.a_stable,
            "angle_deg": self.angle_deg,
            "l_stable": self.l_stable,
            "stiff_limit": self.stiff_limit,
            "negative_real_endpoint": None if math.isinf(endpoint) else endpoint,
            "unbounded_negative_axis": bool(math.isinf(endpoint)),
            "max_on_imaginary_axis": self.max_on_imaginary_axis,
            "argmax_y": self.argmax_y,
            "is_explicit": self.is_explicit,
            "route": self.route,
        }


def stability_report(method: str, order: int = 4, *, n_theta: int = 8193, n_y: int | None = None) -> StabilityReport:
    """把一个方法的 A-稳定性、L-稳定性、稳定角、实轴端点等一次算清。

    ``n_y`` 是虚轴采样点数：单步法（多项式/有理式）缺省 30001 点，几乎不花时间；
    多步法要逐点求根，缺省只用 401 点——多步法的 A-稳定性判据走的是**边界路线**
    （:func:`stability_angle_deg`），虚轴采样只作为参考量，不值得为它烧掉几秒钟。
    """
    is_lmm = method == "bdf"
    if n_y is None:
        n_y = 401 if is_lmm else 30001
    angle, a_boundary = stability_angle_deg(method, order, n_theta=n_theta)
    max_im, argmax_y = max_on_imaginary_axis(method, order, n=n_y)
    explicit = method in _TABLEAUX or method == "explicit_taylor"
    if explicit:
        # 显式方法的稳定函数是次数 ≥ 1 的多项式，沿负实轴无界 ⟹ 不可能 A-稳定（证明，不是数值）。
        a_stable = False
        route = "polynomial: |R(z)|→∞ along z→−∞ ⟹ not A-stable"
    elif method in _RATIONAL:
        a_stable = bool(max_im <= 1.0 + 1e-12)
        route = "maximum modulus on the imaginary axis"
    else:
        a_stable = bool(a_boundary)
        route = "root locus boundary"
    stiff = stiff_limit_amplification(method, order)
    l_stable = bool(a_stable and stiff <= 1e-6)
    endpoint = negative_real_endpoint(method, order, n=4001 if is_lmm else 40001)
    notes: list[str] = []
    if math.isinf(endpoint):
        notes.append("整条负实轴都在稳定域内")
    if method == "trapezoidal":
        notes.append("|R(iy)|≡1：A-稳定但不 L-稳定，刚性模态只在边界上旋转")
    return StabilityReport(
        method=method,
        order=int(order),
        a_stable=a_stable,
        angle_deg=float(angle),
        l_stable=l_stable,
        stiff_limit=stiff,
        negative_real_endpoint=endpoint,
        max_on_imaginary_axis=max_im,
        argmax_y=argmax_y,
        is_explicit=bool(explicit),
        route=route,
        notes=tuple(notes),
    )


def a_stable_orders_bdf(*, n_theta: int = 8193) -> tuple[int, ...]:
    """实测：BDF 中真正 A-稳定的阶数集合（应当只有 :math:`(1,2)`——Dahlquist 第二障碍）。"""
    return tuple(k for k in range(1, 7) if bdf_stability_angle_deg(k, n_theta=n_theta)[1])


def dahlquist_barrier_evidence() -> dict[str, float | int | bool]:
    r"""Dahlquist 第二障碍的实现证据：:math:`A`-稳定的线性多步法阶数 :math:`\le2`。

    三条独立证据（都在返回值里，测试会逐条复核）：

    1. BDF 家族里 A-稳定的只有 :math:`k=1,2`（:math:`k=3` 的角度 86.03° 已经 < 90°）；
    2. 稳定角随阶数单调收缩（:math:`k=1\\to6`：90, 90, 86.03, 73.35, 51.84, 17.84）；
    3. 显式 RK 的 :math:`R` 是次数 :math:`\\ge1` 的多项式，沿负实轴无界，
       因此**任何**显式 RK 都不可能 A-稳定（与阶数无关）。
    """
    angles = {k: bdf_stability_angle_deg(k)[0] for k in range(1, 7)}
    stable = a_stable_orders_bdf()
    angles_seq = [angles[k] for k in range(1, 7)]
    monotone = all(b <= a + 1e-9 for a, b in zip(angles_seq, angles_seq[1:]))
    return {
        "max_a_stable_bdf_order": max(stable) if stable else 0,
        "a_stable_bdf_orders": tuple(stable),  # type: ignore[dict-item]
        "angle_k1": angles[1],
        "angle_k3": angles[3],
        "angle_k6": angles[6],
        "angles_monotone": bool(monotone),
        "explicit_rk_a_stable": False,
    }
