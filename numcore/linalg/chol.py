"""Cholesky 分解：对称正定矩阵的"半个 LU"。

数学定义
--------
:math:`A = A^{\\mathsf T} \\succ 0` 时存在唯一的下三角 :math:`L`（对角元为正）使

.. math::
    A = L L^{\\mathsf T}, \\qquad
    \\ell_{jj} = \\sqrt{a_{jj} - \\sum_{k<j} \\ell_{jk}^2}, \\qquad
    \\ell_{ij} = \\frac{1}{\\ell_{jj}}\\Big(a_{ij} - \\sum_{k<j} \\ell_{ik}\\ell_{jk}\\Big)\\ (i>j).

三个值得记住的结论
------------------
1. **无需选主元**：只要 :math:`A \\succ 0`，被开方的量恒正（:math:`a_{jj} - \\sum \\ell_{jk}^2`
   是 Schur 补的对角元，正定矩阵的 Schur 补仍正定）。所以 Cholesky 天然稳定，
   增长因子 :math:`\\rho = \\max|\\ell_{ij}| / \\sqrt{\\max_i a_{ii}} \\le 1`。
2. **运算量是 LU 的一半**：:math:`\\approx n^3/3` 次乘加，且不需要存 :math:`L` 之外的东西。
3. **"开方失败"就是判定**：出现 :math:`a_{jj} - \\sum \\ell_{jk}^2 \\le 0` 当且仅当
   :math:`A` 非正定（在舍入误差意义下）。这是最便宜的 SPD 判定，比算特征值快得多。

与 LU 的关系：Cholesky 就是"利用对称性的 LU"，:math:`U = D L^{\\mathsf T}`。
本章的对比可视化用它说明"**结构可以换来稳定性**"——LU 需要选主元，Cholesky 不需要。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CholResult",
    "cholesky",
    "chol_solve",
    "chol_reconstruct_error",
    "chol_flops",
    "is_symmetric",
    "is_positive_definite",
    "nearest_spd",
]


@dataclass(frozen=True)
class CholResult:
    """Cholesky 分解结果与诊断量。"""

    L: np.ndarray
    ok: bool
    min_pivot_sq: float
    max_pivot_sq: float
    recon_error: float
    rho: float
    ops: int
    fail_at: int = -1
    """第一个非正主元的位置（``ok=False`` 时有效，否则 -1）。"""

    @property
    def n(self) -> int:
        return int(self.L.shape[0])

    def to_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "ok": self.ok,
            "min_pivot_sq": self.min_pivot_sq,
            "max_pivot_sq": self.max_pivot_sq,
            "recon_error": self.recon_error,
            "rho": self.rho,
            "ops": self.ops,
            "fail_at": self.fail_at,
        }


def _as_square(A: Any) -> np.ndarray:
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("空矩阵没有 Cholesky 分解")
    return np.array(arr, dtype=float, copy=True)


def is_symmetric(A: Any, *, rtol: float = 1e-12) -> bool:
    """相对意义下的对称性检查：:math:`\\|A - A^{\\mathsf T}\\|_\\infty \\le \\mathrm{rtol}\\,\\|A\\|_\\infty`。"""
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False
    asym = float(np.max(np.abs(arr - arr.T))) if arr.size else 0.0
    scale = float(np.max(np.abs(arr))) if arr.size else 0.0
    return asym <= rtol * (scale if scale > 0.0 else 1.0)


def cholesky(A: Any, *, tol: float = 0.0) -> CholResult:
    """自研 Cholesky（按列，右看 / 左看混合的经典写法）。

    ``tol=0`` 时"能否分解"的判据就是精确的 :math:`s > 0`；给正 ``tol`` 会把
    :math:`s \\le \\mathrm{tol}\\cdot\\max_i a_{ii}` 也判为失败——用来演示
    "接近奇异"与"非正定"之间的灰色地带（舍入误差尺度）。
    """
    arr = _as_square(A)
    n = arr.shape[0]
    L = np.zeros((n, n), dtype=float)
    scale = float(np.max(np.abs(np.diag(arr)))) if n else 0.0
    floor = tol * (scale if scale > 0.0 else 1.0)
    ops = 0
    min_p = float("inf")
    max_p = 0.0
    fail_at = -1
    ok = True
    for j in range(n):
        s = float(arr[j, j])
        if j:
            s -= float(L[j, :j] @ L[j, :j])
        if not (s > floor):  # NaN 也会走到这里：非有限输入一律判失败
            ok = False
            fail_at = j
            break
        L[j, j] = np.sqrt(s)
        min_p = min(min_p, s)
        max_p = max(max_p, s)
        if j + 1 < n:
            if j:
                L[j + 1 :, j] = (arr[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
            else:
                L[j + 1 :, j] = arr[j + 1 :, j] / L[j, j]
            ops += (n - j - 1) * (2 * j + 1)
    if not ok:
        # 失败时保留已经算出的前 j 列，便于可视化"崩在哪一步"
        return CholResult(
            L=L,
            ok=False,
            min_pivot_sq=0.0,
            max_pivot_sq=max_p,
            recon_error=float("nan"),
            rho=float("nan"),
            ops=ops,
            fail_at=fail_at,
        )
    recon = chol_reconstruct_error(arr, L)
    diag_scale = np.sqrt(scale) if scale > 0.0 else 1.0
    rho = float(np.max(np.abs(L))) / diag_scale if n else 0.0
    return CholResult(
        L=L,
        ok=True,
        min_pivot_sq=min_p if min_p != float("inf") else 0.0,
        max_pivot_sq=max_p,
        recon_error=recon,
        rho=rho,
        ops=ops,
    )


def chol_reconstruct_error(A: Any, L: Any) -> float:
    """相对重构误差 :math:`\\|A - LL^{\\mathsf T}\\|_\\infty / \\|A\\|_\\infty`。"""
    arr = np.asarray(A, dtype=float)
    Lm = np.asarray(L, dtype=float)
    R = arr - Lm @ Lm.T
    num = float(np.max(np.abs(R))) if R.size else 0.0
    den = float(np.max(np.abs(arr))) if arr.size else 0.0
    return num / den if den > 0.0 else num


def chol_solve(res: CholResult, b: Any) -> np.ndarray:
    """用 :math:`A = LL^{\\mathsf T}` 解方程：前代 :math:`Ly=b`，回代 :math:`L^{\\mathsf T}x=y`。"""
    if not res.ok:
        raise ValueError("分解失败（矩阵非正定），无法求解")
    n = res.n
    bv = np.asarray(b, dtype=float).reshape(-1)
    if bv.size != n:
        raise ValueError(f"b 的长度应为 {n}，得到 {bv.size}")
    y = np.array(bv, dtype=float, copy=True)
    for i in range(n):
        if i:
            y[i] -= float(res.L[i, :i] @ y[:i])
        y[i] /= float(res.L[i, i])
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        acc = y[i]
        if i + 1 < n:
            acc -= float(res.L[i + 1 :, i] @ x[i + 1 :])
        x[i] = acc / float(res.L[i, i])
    return x


def chol_flops(n: int) -> int:
    """Cholesky 的标量运算次数：:math:`\\sum_{j}(n-j-1)(2j+1) \\sim n^3/3`（LU 的一半）。"""
    if n < 1:
        raise ValueError(f"n 必须为正，得到 {n}")
    j = np.arange(0, n - 1, dtype=float)
    return int(np.sum((n - j - 1.0) * (2.0 * j + 1.0)))


def is_positive_definite(A: Any, *, tol: float = 0.0) -> bool:
    """用 Cholesky 判定正定性（最便宜的判据；对称性单独检查，不在这里兜底）。"""
    try:
        return cholesky(A, tol=tol).ok
    except ValueError:
        return False


def nearest_spd(A: Any, eps: float = 1e-3) -> np.ndarray:
    """把任意对称矩阵"推"成正定：:math:`A + \\epsilon\\|A\\| I`（对称即可，不需要特征值）。

    用途：给"非正定 ⇒ Cholesky 失败"造一组参数可控的对照矩阵
    （:math:`\\epsilon` 从负到正扫过去，看它在哪一步开始能分解）。
    """
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"需要方阵，得到形状 {arr.shape}")
    sym = 0.5 * (arr + arr.T)
    scale = float(np.max(np.abs(sym))) if sym.size else 1.0
    return sym + eps * (scale if scale > 0.0 else 1.0) * np.eye(sym.shape[0])
