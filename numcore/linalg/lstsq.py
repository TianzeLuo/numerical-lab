"""1.3 章最小二乘内核：正规方程 / QR / SVD 三条解法，以及它们的误差证据。

数学对象
--------
``min_x ‖A x - b‖_2``（``A`` 为 ``m x n``、``m >= n``）。三条路线给出**同一个**数学解，
但浮点代价完全不同：

======================  ==========================================================
正规方程 ``AᵀA x = Aᵀb``  ``kappa(AᵀA) = kappa(A)^2``：误差被**平方**放大
QR ``R x = (Qᵀb)_{1:n}``  ``kappa`` 只放大一次；Householder 后向稳定
SVD 截断 ``x = V_k Sigma_k^{-1} U_kᵀ b``  秩亏也能给最小范数解；可显式截断噪声
======================  ==========================================================

等式右侧还欠一个说明：**残差 r 与列空间正交**（``Aᵀr = 0``）是三条路线共同的定义方程，
它只保证"垂直"，不保证"解准"——秩亏或病态时必须看 ``‖x - x*‖``（本章 pitfalls 第 4 条）。

出处：Golub & Van Loan, *Matrix Computations* 4th ed., §5.3、§6.1；Higham,
*Accuracy and Stability of Numerical Algorithms* 2nd ed., §20；Hansen,
*Rank-Deficient and Discrete Ill-Posed Problems*, §4（L 曲线与 Picard 条件）。

POLICY 说明（P2/P3/P4/P6）：全部自研；随机矩阵经 :func:`numcore.utils.rng` 取种子；
无 IO、无 print、无全局可变状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from numcore.linalg.lu import lu_partial, solve_lu
from numcore.linalg.qr import (
    householder_qr,
    orthogonality_measure,
    qr_least_squares,
)
from numcore.linalg.svd import (
    SVDFit,
    jacobi_svd,
    numerical_rank,
    pseudoinverse,
    spectral_condition,
    truncated_solution,
)
from numcore.utils import rng

__all__ = [
    "LSSolution",
    "least_squares_normal",
    "least_squares_qr",
    "least_squares_svd",
    "solve_all_methods",
    "relative_error",
    "residual_vector",
    "normal_equation_defect",
    "normal_matrix",
    "condition_of_normal_equations",
    "condition_estimate_2",
    "random_orthogonal",
    "projection_matrix_apply",
    "synthetic_lstsq_problem",
    "prescribed_spectrum",
    "synthetic_truncation_problem",
    "loglog_slope",
    "condition_sweep",
    "truncation_sweep",
    "lcurve_corner",
    "cross_validation_curve",
    "picard_coefficients",
    "uncertainty_ellipse",
    "noise_fan",
    "projection_blend",
    "rotation_matrix_3d",
]

_U = float(np.finfo(float).eps) / 2.0  # 单位舍入 u = 2^-53


# --------------------------------------------------------------------------- #
# 容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LSSolution:
    """一次最小二乘求解的完整记录（解、残差、正交性缺陷、方法名）。"""

    x: np.ndarray
    residual: np.ndarray
    method: str
    condition: float = float("nan")
    rank: int = 0
    normal_defect: float = 0.0
    orthogonality: float = 0.0
    refinement_steps: int = 0

    @property
    def residual_norm(self) -> float:
        return float(np.sqrt(np.dot(self.residual, self.residual)))

    def to_dict(self) -> dict[str, float]:
        return {
            "residual_norm": self.residual_norm,
            "normal_defect": self.normal_defect,
            "orthogonality": self.orthogonality,
            "rank": float(self.rank),
            "condition": self.condition,
        }


def _as_least_squares(A: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    M = np.asarray(A, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"需要二维矩阵，得到 {M.ndim} 维")
    rhs = np.asarray(b, dtype=float).ravel()
    if rhs.size != M.shape[0]:
        raise ValueError(f"b 的长度 {rhs.size} 与 A 的行数 {M.shape[0]} 不一致")
    if M.shape[0] < M.shape[1]:
        raise ValueError(f"最小二乘要求 m >= n（超定/方阵），得到 {M.shape[0]} x {M.shape[1]}")
    return M, rhs


def residual_vector(A: Any, x: Any, b: Any) -> np.ndarray:
    """残差 ``r = b - A x``（向量；范数由调用方决定）。"""
    return np.asarray(b, dtype=float).ravel() - np.asarray(A, dtype=float) @ np.asarray(x, dtype=float).ravel()


def normal_equation_defect(A: Any, x: Any, b: Any) -> float:
    """``‖Aᵀ r‖_2``：最小二乘定义方程的**绝对**缺陷（理论值 0）。"""
    r = residual_vector(A, x, b)
    return float(np.sqrt(np.dot(np.asarray(A, dtype=float).T @ r, np.asarray(A, dtype=float).T @ r)))


def relative_error(x: Any, x_ref: Any) -> float:
    """相对前向误差 ``‖x - x*‖_2 / ‖x*‖_2``（``x*`` 已知时才有意义）。"""
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(x_ref, dtype=float).ravel()
    denom = float(np.sqrt(np.dot(b, b)))
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff))) / denom if denom > 0.0 else float("nan")


def normal_matrix(A: Any) -> np.ndarray:
    """``AᵀA``——**显式**形成它就等于把条件数平方（本模块只在"做错的那条路"上用它）。"""
    M = np.asarray(A, dtype=float)
    return M.T @ M


def condition_estimate_2(A: Any, *, fit: SVDFit | None = None) -> float:
    """``kappa_2(A) = s_1 / s_n``（用自研 Jacobi SVD 实测，不调 ``np.linalg``）。"""
    f = jacobi_svd(A) if fit is None else fit
    return spectral_condition(f.s)


def condition_of_normal_equations(A: Any, *, fit: SVDFit | None = None) -> float:
    """``kappa_2(AᵀA)`` 的实测值（理论上等于 ``kappa_2(A)^2``）。"""
    G = normal_matrix(A)
    return condition_estimate_2(G)


# --------------------------------------------------------------------------- #
# 三条解法
# --------------------------------------------------------------------------- #
def least_squares_normal(A: Any, b: Any, *, refinement: int = 0) -> LSSolution:
    """正规方程法：解 ``(AᵀA) x = Aᵀb``（可选 ``refinement`` 步迭代精化）。

    ``AᵀA`` 的条件数是 ``A`` 的平方，所以即使 LU 本身后向稳定，前向误差也会被
    ``kappa(A)^2 u`` 放大——这正是本章"精度对决"那条分道扬镳的曲线。

    迭代精化（``refinement > 0``）用**同一精度**重算残差 ``r = b - Ax`` 再解
    ``AᵀA d = Aᵀr``：它能修掉"解方程"这一步的误差，但修不掉 ``AᵀA`` 本身的病态，
    实测在 ``kappa > 1/u`` 之后基本无效（pitfalls 第 2 条）。
    """
    M, rhs = _as_least_squares(A, b)
    G = M.T @ M
    c = M.T @ rhs
    res = lu_partial(G)
    if res.singular:
        raise ValueError("AᵀA 奇异（A 秩亏）：请改用截断 SVD")
    x = solve_lu(res, c)
    steps = 0
    for _ in range(max(0, int(refinement))):
        r = rhs - M @ x
        d = solve_lu(res, M.T @ r)
        x = x + d
        steps += 1
    return _package(M, rhs, x, method="normal", rank=M.shape[1], refinement_steps=steps)


def least_squares_qr(A: Any, b: Any, *, tol: float | None = None) -> LSSolution:
    """Householder（列主元）QR 法：``R x = (Qᵀ b)_{1:n}``，误差只被 ``kappa`` 放大一次。"""
    M, rhs = _as_least_squares(A, b)
    x = qr_least_squares(M, rhs, tol=tol)
    return _package(M, rhs, x, method="qr", rank=M.shape[1])


def least_squares_svd(A: Any, b: Any, *, rtol: float | None = None, fit: SVDFit | None = None) -> LSSolution:
    """SVD 法（可截断）：``x = V_r Sigma_r^{-1} U_rᵀ b``，秩亏时给**最小范数**解。"""
    M, rhs = _as_least_squares(A, b)
    f = jacobi_svd(M) if fit is None else fit
    r = numerical_rank(f.s, M.shape[0], M.shape[1], rtol=rtol)
    x = truncated_solution(f.U, f.s, f.V, rhs, r if r > 0 else f.s.size)
    return _package(M, rhs, x, method="svd", rank=r, condition=spectral_condition(f.s))


def _package(
    M: np.ndarray,
    rhs: np.ndarray,
    x: np.ndarray,
    *,
    method: str,
    rank: int,
    condition: float = float("nan"),
    refinement_steps: int = 0,
    extra: Mapping[str, float] | None = None,
) -> LSSolution:
    r = residual_vector(M, x, rhs)
    defect = normal_equation_defect(M, x, rhs)
    orth = orthogonality_measure(M, r)
    return LSSolution(
        x=np.asarray(x, dtype=float),
        residual=r,
        method=method,
        condition=float(condition),
        rank=int(rank),
        normal_defect=defect,
        orthogonality=orth,
        refinement_steps=refinement_steps,
    )


def solve_all_methods(
    A: Any,
    b: Any,
    *,
    refinement: int = 0,
    rtol: float | None = None,
) -> dict[str, LSSolution]:
    """同一问题上跑三条路线（外加可选精化的正规方程），返回 ``方法名 -> 解``。"""
    out = {
        "normal": least_squares_normal(A, b),
        "qr": least_squares_qr(A, b),
        "svd": least_squares_svd(A, b, rtol=rtol),
    }
    if refinement > 0:
        out["normal_ir"] = least_squares_normal(A, b, refinement=refinement)
    return out


# --------------------------------------------------------------------------- #
# 构造"条件数可控"的测试问题
# --------------------------------------------------------------------------- #
def random_orthogonal(m: int, n: int, *, seed: int = 20261018) -> np.ndarray:
    """``m x n`` 列正交矩阵（``m >= n``）：随机矩阵经自研 Householder QR 正交化。"""
    if m < n:
        raise ValueError(f"需要 m >= n，得到 {m} x {n}")
    g = rng(seed)
    Z = g.standard_normal((m, n))
    q = householder_qr(Z)
    return q.Q


def synthetic_lstsq_problem(
    m: int,
    n: int,
    kappa: float,
    *,
    seed: int = 20261018,
    x_kind: str = "ones",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造条件数**恰好**为 ``kappa`` 的相容最小二乘问题：返回 ``(A, b, x*)``，``b = A x*``。

    做法：``A = U diag(sigma) Vᵀ``，其中 ``U``、``V`` 列正交、``sigma`` 从 1 几何递减到
    ``1/kappa``，于是 ``kappa_2(A) = kappa``（精确到舍入）。右端项取 ``b = A x*``：
    解是**已知**的，所以"前向误差"是可以直接测量的量，不需要再找一个参考解
    （避免"用同一套算法算参考解"这种自证）。
    """
    if kappa < 1.0:
        raise ValueError(f"kappa 必须 >= 1，得到 {kappa}")
    U = random_orthogonal(m, n, seed=seed)
    V = random_orthogonal(n, n, seed=seed + 977)
    sig = np.logspace(0.0, -float(np.log10(kappa)), n)
    A = (U * sig) @ V.T
    if x_kind == "ones":
        x_true = np.ones(n, dtype=float)
    elif x_kind == "ramp":
        x_true = np.linspace(1.0, 2.0, n)
    else:
        g = rng(seed + 4241)
        x_true = g.standard_normal(n)
    return A, A @ x_true, x_true


def projection_matrix_apply(A: Any, b: Any, *, fit: SVDFit | None = None) -> np.ndarray:
    """``P b = A A⁺ b``：到 ``col(A)`` 的正交投影（幂等、对称）。"""
    M = np.asarray(A, dtype=float)
    f = jacobi_svd(M) if fit is None else fit
    return M @ (f.pinv() @ np.asarray(b, dtype=float).ravel())


def prescribed_spectrum(n: int, kappa: float, kind: str = "decay") -> np.ndarray:
    """构造长度为 ``n``、``s_1/s_n = kappa`` 的奇异值谱（四种形状，各有各的陷阱）。

    * ``decay``：几何递减——"正常"的离散不适定问题；
    * ``cliff``：前半段全是 1、后半段突然掉到 ``1/kappa``——截断阈值落在**悬崖**上时，
      解范数会突变（阈值差一个下标，解范数可以差几个数量级）；
    * ``plateau``：只有一个很小的奇异值——秩看起来是 ``n-1``，实际是"几乎秩亏"；
    * ``two_clusters``：两簇——重奇异值附近 Jacobi 收敛慢，且截断点不可辨。
    """
    k = float(kappa)
    if kind == "decay":
        return np.logspace(0.0, -np.log10(k), n)
    if kind == "cliff":
        half = max(1, n // 2)
        s = np.ones(n, dtype=float)
        s[half:] = 1.0 / k
        return s
    if kind == "plateau":
        s = np.ones(n, dtype=float)
        s[-1] = 1.0 / k
        return s
    if kind == "two_clusters":
        s = np.ones(n, dtype=float)
        half = max(1, n // 2)
        s[half:] = 1.0 / k
        for i in range(half):
            s[i] = 1.0 - 1e-3 * (i / max(half - 1, 1))
        for i in range(half, n):
            s[i] = (1.0 / k) * (1.0 - 1e-3 * ((i - half) / max(n - half - 1, 1)))
        # 谱按约定递减：两簇各自内部再降一点，否则第二簇是"升"的，
        # 排序假设（截断 SVD、Picard 图、L 曲线全都默认 s 递减）当场失效
        return np.sort(s)[::-1]
    raise ValueError(f"未知的谱形状 {kind!r}（可选 decay/cliff/plateau/two_clusters）")


def synthetic_truncation_problem(
    m: int,
    n: int,
    kappa: float,
    *,
    spectrum: str = "decay",
    decay: float = 2.0,
    noise_rel: float = 0.0,
    seed: int = 20261018,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造"离散不适定"型问题：返回 ``(A, b, x*, sigma)``。

    关键是让 Picard 条件**看起来成立**：取系数 ``alpha_i = sigma_i^decay``（``decay = 2``），
    于是真解系数 ``alpha_i/sigma_i = sigma_i`` 随 ``i`` 衰减——解是"光滑"的。
    加噪之后，小奇异值方向的系数被噪声接管，``|u_iᵀb|/sigma_i`` 重新抬头，
    截断就成了必须的动作（这正是可视化 3 要讲的事）。
    """
    U = random_orthogonal(m, n, seed=seed)
    V = random_orthogonal(n, n, seed=seed + 977)
    sig = prescribed_spectrum(n, kappa, spectrum)
    alpha = sig ** float(decay)
    b_exact = U @ alpha
    x_true = V @ (alpha / sig)
    A = (U * sig) @ V.T
    if noise_rel > 0.0:
        g = rng(seed + 4241)
        b = b_exact + float(noise_rel) * float(np.sqrt(np.dot(b_exact, b_exact))) * g.standard_normal(m)
    else:
        b = b_exact
    return A, b, x_true, sig


def rotation_matrix_3d(theta_deg: float, phi_deg: float) -> np.ndarray:
    """单位球面上方向角 ``(theta, phi)`` 对应的单位向量（``phi`` 从 +z 轴量起）。"""
    th = np.deg2rad(float(theta_deg))
    ph = np.deg2rad(float(phi_deg))
    return np.array([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], dtype=float)


def projection_blend(A: Any, b: Any, t: float) -> dict[str, np.ndarray]:
    """投影分解的动画插值：``b = p_t + r_t``，``p_t = p + t r``、``r_t = (1-t) r``。

    ``t = 0`` 时 ``p_t = b``、``r_t = 0``（只看见 b）；``t = 1`` 时是两个真正的分量。
    中间量是线性插值（纯几何动画），不引入任何新数学。
    """
    M = np.asarray(A, dtype=float)
    bv = np.asarray(b, dtype=float).ravel()
    tt = float(min(max(t, 0.0), 1.0))
    p = projection_matrix_apply(M, bv)
    r = bv - p
    return {"p_t": p + tt * r, "r_t": (1.0 - tt) * r, "b_t": bv}


# --------------------------------------------------------------------------- #
# 双对数斜率（"分道扬镳"的定量证据）
# --------------------------------------------------------------------------- #
def loglog_slope(x: Any, y: Any, *, skip: int = 0, floor: float | None = None) -> float:
    """``log y`` 对 ``log x`` 的最小二乘斜率（实测幂律指数）。

    ``skip`` 丢弃最前面的 ``skip`` 点（小 ``kappa`` 端还在舍入地板附近，不服从幂律）；
    ``floor`` 把 ``y`` 抬到给定下限，避免 ``log 0``（与帧里画图的地板同一口径）。
    """
    xv = np.asarray(x, dtype=float).ravel()[int(skip) :]
    yv = np.asarray(y, dtype=float).ravel()[int(skip) :]
    if floor is not None:
        yv = np.maximum(yv, float(floor))
    good = np.isfinite(xv) & np.isfinite(yv) & (xv > 0.0) & (yv > 0.0)
    xv = np.log(xv[good])
    yv = np.log(yv[good])
    if xv.size < 2:
        return float("nan")
    xm = float(np.mean(xv))
    ym = float(np.mean(yv))
    sxx = float(np.sum((xv - xm) ** 2))
    if sxx == 0.0:
        return float("nan")
    return float(np.sum((xv - xm) * (yv - ym)) / sxx)


# 一条路线在浮点下**解不出来**时抛的异常（例如 AᵀA 的 LU 主元恰好为 0）。
# 这不是"算得不准"，而是"没有有限解"——必须能被扫描器表达，不能把整帧带崩。
SOLVE_FAILURES = (ZeroDivisionError, ValueError, OverflowError)
# 相对误差的上限：一位有效数字都不剩。路线失败时用它占位，并在 singular_* 里标出，
# 绝不写 NaN/Inf（`.nlf` 的 JSON 头用 allow_nan=False，NaN 会让整帧编码失败）。
ERROR_CEILING = 1.0


def _sweep_solve(name: str, A: np.ndarray, b: np.ndarray, *, refinement: int) -> "LSSolution":
    """扫描器用的一次求解（按名字选路线，便于逐条兜底）。"""
    if name == "normal":
        return least_squares_normal(A, b)
    if name == "normal_ir":
        return least_squares_normal(A, b, refinement=refinement)
    if name == "qr":
        return least_squares_qr(A, b)
    return least_squares_svd(A, b)


# --------------------------------------------------------------------------- #
# kappa 扫描：正规方程 vs QR vs SVD
# --------------------------------------------------------------------------- #
def condition_sweep(
    *,
    exp_min: float = 2.0,
    exp_max: float = 12.0,
    n_points: int = 21,
    m: int = 16,
    n: int = 8,
    seed: int = 20261018,
    refinement: int = 0,
    n_seeds: int = 3,
    slope_ceiling: float = 0.1,
) -> dict[str, Any]:
    """沿 ``log10 kappa`` 均匀扫描，实测三条路线的前向误差与残差。

    每个 ``kappa`` 上取 ``n_seeds`` 个不同种子的矩阵，误差用**几何平均**（对数平均）汇总：
    单个随机矩阵的误差常数可以上下跳十倍，不平均的话拟合出来的斜率全是噪声
    （这本身是 pitfalls 里"斜率是量出来的、不是看出来的"一条）。

    斜率只在**未饱和窗口** ``err < slope_ceiling`` 内拟合：误差一旦到 O(1)，
    幂律就断了，把饱和段塞进拟合只会把斜率拉平。窗口上界写进 ``slope_window``，
    供前端在图上标出"这一段才是幂律区"（防误导规则 2）。

    **解不出来的点**：κ 大到 ``AᵀA`` 的浮点主元恰好为 0 时，正规方程这条路线会抛
    ``ZeroDivisionError``（见 :func:`least_squares_normal`）。扫描器逐条路线兜底：
    只要有任何一个种子上这条路线解不出来，该 κ 上的误差就记成上限
    :data:`ERROR_CEILING`（一位有效数字都不剩），残差同样取 1.0，并在
    ``singular_normal`` / ``singular_qr`` / … 里记下失败的种子数。这些点天然落在
    拟合窗口之外（``1.0 > slope_ceiling``），不会被当成幂律的一部分。
    """
    exps = np.linspace(float(exp_min), float(exp_max), int(n_points))
    kappas = 10.0**exps
    n_seeds = max(1, int(n_seeds))
    names = ("normal", "qr", "svd") + (("normal_ir",) if refinement > 0 else ())
    keys = ("normal", "qr", "svd", "normal_ir", "res_normal", "res_qr")
    acc: dict[str, np.ndarray] = {k: np.zeros(n_points, dtype=float) for k in keys}
    acc["def_normal"] = np.zeros(n_points, dtype=float)
    acc["def_qr"] = np.zeros(n_points, dtype=float)
    acc["kappa_measured"] = np.zeros(n_points, dtype=float)
    singular: dict[str, np.ndarray] = {name: np.zeros(n_points, dtype=float) for name in names}
    for i, kap in enumerate(kappas):
        logs: dict[str, list[float]] = {k: [] for k in keys}
        defs: dict[str, list[float]] = {"normal": [], "qr": []}
        for s in range(n_seeds):
            A, b, x_true = synthetic_lstsq_problem(m, n, float(kap), seed=int(seed) + 1000 * s)
            bnorm = float(np.sqrt(np.dot(b, b))) or 1.0
            a_fro = float(np.sqrt(np.sum(A * A)))
            solved: dict[str, Any] = {}
            for name in names:
                try:
                    solved[name] = _sweep_solve(name, A, b, refinement=refinement)
                except SOLVE_FAILURES:
                    singular[name][i] += 1.0
            for name in names:
                sol = solved.get(name)
                if sol is None:
                    continue
                logs[name].append(np.log(max(relative_error(sol.x, x_true), _U)))
            if solved.get("normal") is not None:
                logs["res_normal"].append(np.log(max(solved["normal"].residual_norm / bnorm, _U)))
                # 后向误差的标尺：n u ‖A‖_F (‖b‖ + ‖A‖_F‖x̂‖)。
                # 只用 ‖b‖ 会在病态时量错——那时 ‖x̂‖ 自己就大到 κ‖b‖/‖A‖
                # （实测 κ=1e10 用旧标尺得 5.4e8，用这把得 0.18）。
                x_qr = solved["qr"].x if solved.get("qr") is not None else solved["normal"].x
                scale = n * _U * a_fro * (bnorm + a_fro * float(np.sqrt(np.dot(x_qr, x_qr))))
                defs["normal"].append(solved["normal"].normal_defect / scale)
            if solved.get("qr") is not None:
                logs["res_qr"].append(np.log(max(solved["qr"].residual_norm / bnorm, _U)))
                x_qr = solved["qr"].x
                scale = n * _U * a_fro * (bnorm + a_fro * float(np.sqrt(np.dot(x_qr, x_qr))))
                defs["qr"].append(solved["qr"].normal_defect / scale)
            acc["kappa_measured"][i] = condition_estimate_2(A)
        for k in keys:
            route = k[4:] if k.startswith("res_") else k
            if route not in names:
                continue  # 这条路线本次扫描没跑（例如 refinement=0 时的 normal_ir）
            if (not logs[k]) or singular[route][i] > 0.0:
                # 所有种子都失败，或只要有一个种子失败：这条路线在这个 κ 上不可靠，
                # 记上限（一位有效数字都不剩），别在图上凭空画一个 0
                acc[k][i] = ERROR_CEILING
            else:
                acc[k][i] = float(np.exp(np.mean(logs[k])))
        acc["def_normal"][i] = float(np.max(defs["normal"])) if defs["normal"] else _U
        acc["def_qr"][i] = float(np.max(defs["qr"])) if defs["qr"] else _U
    floor = _U
    slopes: dict[str, float] = {}
    windows: dict[str, float] = {}
    for key in ("normal", "qr", "svd", "normal_ir"):
        series = acc[key]
        if not np.any(series > 0.0):
            continue
        mask = series <= float(slope_ceiling)
        if np.count_nonzero(mask) < 3:
            mask = np.ones_like(series, dtype=bool)
        slopes[key] = loglog_slope(kappas[mask], series[mask], floor=floor)
        windows[key] = float(np.max(kappas[mask])) if np.any(mask) else float("nan")
    return {
        "kappa": kappas,
        "log10_kappa": exps,
        "kappa_measured": acc["kappa_measured"],
        "error_normal": np.maximum(acc["normal"], floor),
        "error_qr": np.maximum(acc["qr"], floor),
        "error_svd": np.maximum(acc["svd"], floor),
        "error_normal_ir": np.maximum(acc["normal_ir"], floor),
        "residual_normal": np.maximum(acc["res_normal"], floor),
        "residual_qr": np.maximum(acc["res_qr"], floor),
        "defect_normal": np.maximum(acc["def_normal"], floor),
        "defect_qr": np.maximum(acc["def_qr"], floor),
        # 每个 κ 上有多少个种子让这条路线**解不出来**（0 = 全部解出来了）。
        # 前端据此在曲线上打"没有有限解"的标记，而不是把上限值当实测点读。
        "singular_normal": singular["normal"],
        "singular_qr": singular["qr"],
        "singular_svd": singular["svd"],
        "singular_normal_ir": singular.get("normal_ir", np.zeros(n_points, dtype=float)),
        "error_ceiling": float(ERROR_CEILING),
        "theory_kappa1": np.maximum(kappas * _U, floor),
        "theory_kappa2": np.maximum(kappas * kappas * _U, floor),
        "slopes": slopes,
        "slope_window": windows,
        "slope_ceiling": float(slope_ceiling),
        "error_floor": floor,
        "n_seeds": float(n_seeds),
        "m": float(m),
        "n": float(n),
        "refinement": float(refinement),
        "n_points": float(n_points),
    }


# --------------------------------------------------------------------------- #
# 截断 SVD：L 曲线与 Picard 条件
# --------------------------------------------------------------------------- #
def picard_coefficients(U: Any, s: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    """Picard 量：``|u_iᵀ b|`` 与比值 ``|u_iᵀ b| / sigma_i``（后者必须**递减**才可解）。"""
    Uu = np.asarray(U, dtype=float)
    sv = np.asarray(s, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    coeff = np.abs(Uu.T @ bb)
    ratio = coeff / np.where(sv > 0.0, sv, np.inf)
    return coeff, ratio


def truncation_sweep(
    A: Any,
    b: Any,
    *,
    x_true: Any | None = None,
    k_max: int | None = None,
    rtol: float | None = None,
) -> dict[str, Any]:
    """对每个截断阶 ``k = 0..k_max`` 实测 ``‖x_k‖``、``‖A x_k - b‖``（与可选的真误差）。

    这正是 L 曲线的两个坐标：**解范数**与**残差**。两者都随 ``k`` 单调
    （前者不减、后者不增），而真误差先降后升——"残差最小"与"解最准"因此是两件事。
    """
    M = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float).ravel()
    m, n = M.shape
    f = jacobi_svd(M, tol=None)
    kk = int(k_max if k_max is not None else n)
    kk = max(0, min(kk, n))
    grid = np.arange(kk + 1, dtype=float)
    x_norm = np.zeros(kk + 1, dtype=float)
    residual = np.zeros(kk + 1, dtype=float)
    error = np.full(kk + 1, np.nan, dtype=float)
    bnorm = float(np.sqrt(np.dot(rhs, rhs))) or 1.0
    for k in range(kk + 1):
        x = truncated_solution(f.U, f.s, f.V, rhs, k)
        x_norm[k] = float(np.sqrt(np.dot(x, x)))
        r = residual_vector(M, x, rhs)
        residual[k] = float(np.sqrt(np.dot(r, r))) / bnorm
        if x_true is not None:
            error[k] = relative_error(x, x_true)
    coeff, ratio = picard_coefficients(f.U, f.s, rhs)
    rank = numerical_rank(f.s, m, n, rtol=rtol)
    return {
        "k": grid,
        "sigma": f.s,
        "coeff": coeff,
        "picard": ratio,
        "x_norm": x_norm,
        "residual": residual,
        "error": error,
        "rank": float(rank),
        "k_opt_error": float(int(np.nanargmin(error))) if x_true is not None and np.any(np.isfinite(error)) else float("nan"),
        "k_corner": float(lcurve_corner(x_norm, residual)),
        "k_picard": float(int(np.argmin(ratio))) if ratio.size else float("nan"),
        "k_max": float(kk),
        "condition": spectral_condition(f.s),
        "n": float(n),
        "m": float(m),
    }


def lcurve_corner(x_norm: Any, residual: Any) -> int:
    """L 曲线的离散拐点：在 ``(log‖x_k‖, log‖A x_k - b‖)`` 上取曲率最大的点。

    曲率用相邻三点的**有限差分**估计（第二类离散曲率），不去拟合任何东西——
    这是"实测拐点"，不是把某个解析阈值抄回来。
    """
    u = np.log(np.maximum(np.asarray(x_norm, dtype=float).ravel(), np.finfo(float).tiny))
    v = np.log(np.maximum(np.asarray(residual, dtype=float).ravel(), np.finfo(float).tiny))
    k = u.size
    if k < 3:
        return 0
    best_i, best_c = 1, -1.0
    for i in range(1, k - 1):
        du = 0.5 * (u[i + 1] - u[i - 1])
        dv = 0.5 * (v[i + 1] - v[i - 1])
        ddu = u[i + 1] - 2.0 * u[i] + u[i - 1]
        ddv = v[i + 1] - 2.0 * v[i] + v[i - 1]
        denom = (du * du + dv * dv) ** 1.5
        if denom <= 0.0:
            continue
        curv = abs(du * ddv - dv * ddu) / denom
        if curv > best_c:
            best_c, best_i = curv, i
    return int(best_i)


# --------------------------------------------------------------------------- #
# 不确定扇面（等残差谷底）
# --------------------------------------------------------------------------- #
def uncertainty_ellipse(
    A: Any,
    b: Any,
    *,
    rho_rel: float = 1e-3,
    n_theta: int = 181,
    center: Any | None = None,
) -> dict[str, np.ndarray]:
    """等残差集合 ``{x : ‖A x - b‖ = rho}`` 在 ``R^2`` 中的椭圆（``A`` 必须两列）。

    推导：``‖Ax-b‖² = ‖A(x-x_LS)‖² + ‖r_min‖²``（``r_min`` 是极小残差，与列空间正交）。
    所以等残差集是 "``A(x-x_LS)`` 是一个半径 ``sqrt(rho²-‖r_min‖²)`` 的圆" 的原像，
    即半轴为 ``sqrt(rho²-‖r_min‖²)/sigma_i``、方向为右奇异向量 ``v_i`` 的椭圆。
    **它的扁度就是条件数**：``kappa`` 越大，同一个残差对应的解集合越像一根针。
    """
    M = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float).ravel()
    if M.shape[1] != 2:
        raise ValueError(f"uncertainty_ellipse 只画两列的情形，得到 {M.shape[1]} 列")
    f = jacobi_svd(M)
    x_ls = truncated_solution(f.U, f.s, f.V, rhs, numerical_rank(f.s, M.shape[0], M.shape[1]))
    x0 = np.asarray(center, dtype=float).ravel() if center is not None else x_ls
    r_min = residual_vector(M, x0, rhs)
    r_min_norm = float(np.sqrt(np.dot(r_min, r_min)))
    bnorm = float(np.sqrt(np.dot(rhs, rhs))) or 1.0
    rho = float(rho_rel) * bnorm
    inside = rho * rho - r_min_norm * r_min_norm
    t = float(np.sqrt(inside)) if inside > 0.0 else 0.0
    th = np.linspace(0.0, 2.0 * np.pi, int(n_theta))
    w = np.vstack([t * np.cos(th), t * np.sin(th)])
    offsets = f.V @ (w / f.s[:, None])
    return {
        "theta": th,
        "ellipse_x": x0[0] + offsets[0, :],
        "ellipse_y": x0[1] + offsets[1, :],
        "center": x0,
        "radius": np.array([t]),
        "residual_level": np.array([rho / bnorm]),
        "semi_axis1": np.array([t / f.s[0]]),
        "semi_axis2": np.array([t / f.s[-1]]),
        "v1": f.V[:, 0],
        "v2": f.V[:, -1],
        "sigma": f.s,
    }


def noise_fan(
    A: Any,
    b: Any,
    *,
    noise_rel: float = 1e-6,
    n_angles: int = 25,
) -> dict[str, np.ndarray]:
    """噪声扇面：让 ``b`` 的扰动方向在 ``(u_1, u_2)`` 平面内转一圈，实测最小二乘解怎么跑。

    相容问题（``b = A x*``）下每个扰动解都有**相同**的残差范数（就是噪声水平），
    但 ``‖x(θ) - x*‖`` 会被 ``1/sigma_i`` 放大——扇面的张开角就是条件数。
    """
    M = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float).ravel()
    f = jacobi_svd(M)
    rank = numerical_rank(f.s, M.shape[0], M.shape[1])
    x_ls = truncated_solution(f.U, f.s, f.V, rhs, rank)
    bnorm = float(np.sqrt(np.dot(rhs, rhs))) or 1.0
    amp = float(noise_rel) * bnorm
    r1 = f.U[:, 0]
    r2 = f.U[:, 1] if f.U.shape[1] > 1 else f.U[:, 0]
    th = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    sol = np.zeros((2, th.size), dtype=float)
    err = np.zeros(th.size, dtype=float)
    res = np.zeros(th.size, dtype=float)
    xnorm = float(np.sqrt(np.dot(x_ls, x_ls))) or 1.0
    pinv = pseudoinverse(f.U, f.s, f.V, rtol=None)
    for i, ang in enumerate(th):
        db = amp * (np.cos(ang) * r1 + np.sin(ang) * r2)
        x = pinv @ (rhs + db)
        sol[:, i] = x
        err[i] = float(np.sqrt(np.dot(x - x_ls, x - x_ls))) / xnorm
        rr = residual_vector(M, x, rhs)
        res[i] = float(np.sqrt(np.dot(rr, rr))) / bnorm
    return {
        "theta": th,
        "sol_x": sol[0, :],
        "sol_y": sol[1, :],
        "center": x_ls,
        "error": np.maximum(err, _U),
        "residual": np.maximum(res, _U),
        "noise_rel": np.array([float(noise_rel)]),
        "sigma": f.s,
        "condition": np.array([spectral_condition(f.s)]),
    }


def cross_validation_curve(
    A: Any,
    b: Any,
    *,
    n_lambda: int = 25,
    fit: SVDFit | None = None,
) -> dict[str, Any]:
    """广义交叉验证（GCV）曲线：**不知道真解**时怎么挑截断阶。

    对 Tikhonov 形式 ``min ‖Ax-b‖² + lambda²‖x‖²``，解是
    ``x_λ = Σ σ_i/(σ_i²+λ²)·(u_iᵀb)·v_i``，GCV 函数为

    ``G(λ) = ‖A x_λ - b‖² / (m - Σ σ_i²/(σ_i²+λ²))²``

    分母是"有效自由度"的补：它惩罚把 λ 调得过小（拟合噪声）的解。
    取 ``G`` 的最小点 ``λ*``，截断阶就取 ``#{σ_i > λ*}``——这条路线只用 A 和 b，
    比 L 曲线的目测拐点可复现（两者不一定一致，见本章 pitfalls）。
    """
    M = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float).ravel()
    m = M.shape[0]
    f = jacobi_svd(M) if fit is None else fit
    s = np.asarray(f.s, dtype=float)
    coeff = f.U.T @ rhs
    lam = np.logspace(np.log10(max(s[-1], np.finfo(float).tiny)) - 1.0, np.log10(s[0]) + 1.0, int(n_lambda))
    cv = np.zeros(lam.size, dtype=float)
    x_norm = np.zeros(lam.size, dtype=float)
    residual = np.zeros(lam.size, dtype=float)
    bnorm = float(np.sqrt(np.dot(rhs, rhs))) or 1.0
    for i, l in enumerate(lam):
        filt = s / (s * s + l * l)
        x = f.V @ (filt * coeff)
        x_norm[i] = float(np.sqrt(np.dot(x, x)))
        r = rhs - M @ x
        residual[i] = float(np.sqrt(np.dot(r, r))) / bnorm
        denom = m - float(np.sum(s * s / (s * s + l * l)))
        cv[i] = (residual[i] * bnorm) ** 2 / max(denom * denom, np.finfo(float).tiny)
    best = int(np.argmin(cv))
    k_grid = np.array([int(np.count_nonzero(s > l)) for l in lam], dtype=float)
    return {
        "lambda": lam,
        "cv": cv,
        "x_norm": _floor_positive(x_norm),
        "residual": _floor_positive(residual),
        "k_grid": k_grid,
        "k_cv": float(k_grid[best]),
        "lambda_best": float(lam[best]),
        "cv_min": float(cv[best]),
        "n": float(s.size),
        "m": float(m),
    }


def _floor_positive(values: Any) -> np.ndarray:
    """给对数轴用的正数地板（GCV 的残差可能小到 0）。"""
    return np.maximum(np.asarray(values, dtype=float).ravel(), _U)
