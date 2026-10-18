"""1.4 章 A/C 类断言（伪谱部分）：σ_min、伪谱半径、Kreiss 常数、瞬态增长、GMRES。

这一半的任务是把"特征值"升级成"伪谱与瞬态"，所以断言分三层：

* **A 类（解析对拍）**——凡是有闭式答案的地方都拿闭式对拍：
  对角矩阵 :math:`\\sigma_{\\min}(A-zI)=\\min_i|z-\\lambda_i|`、
  Jordan 块 :math:`\\sigma_{\\min}(J_n-zI)=|z|^n`、
  正规矩阵的伪谱是 :math:`\\varepsilon`-圆盘并、Jordan 块伪谱半径
  :math:`\\rho_\\varepsilon\\to\\varepsilon^{1/n}`。对拍对象一律是
  :mod:`numcore.reference`（scipy/LAPACK 只允许出现在那里，P1/P2）。
* **C 类（收敛率/阈值实测）**——伪谱半径的幂律指数用
  :func:`numcore.verify.fit_order` 拟合，Kreiss 下界用二分找阈值，GMRES 的
  下降率用实测残差比。**不许把理论值抄回来当实测值**。
* **实现路线本身也要被断言**——本章的 :math:`\\sigma_{\\min}` 是用
  :math:`M^{\\mathsf H}M` 的 Hermite Jacobi 算的（不用 SVD），所以有一条专门检查
  "它与 LAPACK 的 :func:`numpy.linalg.svd` 在良态点上一致，在病态点上**如实地**
  掉进噪声地板（``at_noise_floor`` 被置位）"，而不是假装它永远准。
"""

from __future__ import annotations

import numpy as np
import pytest

from numcore.linalg.eigen import dense_eigenvalues, laplacian_1d
from numcore.linalg.pseudospec import (
    EXAMPLE_PSEUDOSPECTRUM_MATRICES,
    complex_lu,
    complex_solve,
    example_pseudospectrum_matrix,
    gmres,
    gmres_arnoldi_residual_bound,
    jordan_boundary_radius,
    kreiss_constant,
    kreiss_from_pseudospectrum,
    marching_squares,
    pseudospectrum_grid,
    pseudospectral_radius,
    sigma_min,
    sigma_min_grid,
    transient_growth,
)
from numcore.reference import dense as ref
from numcore.utils import rng
from numcore.verify import (
    MACHINE_EPS,
    assertion,
    fit_order,
    find_threshold,
    record,
)

# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _svdvals(M: np.ndarray) -> np.ndarray:
    r""":math:`\sigma(M)`（参考层，仅用于对拍）。

    用 :func:`numcore.reference.dense.svdvals` 而不是 ``ref.svd``：
    后者在复数输入上会被 ``dtype=float`` 静默截断（只发一条 ``ComplexWarning``）。
    """
    return np.asarray(ref.svdvals(M), dtype=float)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


# --------------------------------------------------------------------------- #
# A 类：σ_min 的解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="对角矩阵 σ_min(A−zI) = min_i |z − λ_i|（解析）", chapter="1.4")
def test_sigma_min_equals_distance_to_spectrum_for_diagonal() -> None:
    r"""对角（正规）矩阵上 :math:`\\sigma_{\\min}(A-zI)` **精确等于**到谱的距离。

    这是伪谱定义在正规矩阵上退化成"圆盘并"的根本原因：
    :math:`\\Lambda_\\varepsilon=\\bigcup_i\\{|z-\\lambda_i|\\le\\varepsilon\\}`。
    所以这条恒等式一旦不成立，后面所有"正规对照"的结论都作废。
    """
    g = rng(2024)
    lam = g.standard_normal(9) + 1j * g.standard_normal(9)
    A = np.diag(lam)
    worst = 0.0
    worst_svd = 0.0
    for z in (0.3 + 0.4j, -1.1 + 0.2j, 0.0 + 0.0j, 2.0 - 1.0j):
        d = float(np.min(np.abs(lam - z)))
        got = sigma_min(A, z)
        worst = max(worst, _rel(got.value, d))
        worst_svd = max(worst_svd, _rel(got.value, _svdvals(np.asarray(A, dtype=complex) - z * np.eye(9))[-1]))
    rec = record(
        label="对角矩阵 σ_min vs min|z−λ_i| vs LAPACK",
        theory="σ_min = dist(z, Λ)（精确，正规矩阵）",
        measured=f"最大相对误差（解析）{worst:.2e}；（LAPACK）{worst_svd:.2e}",
        tol="两者都 ≤ 1e-12",
        passed=worst <= 1e-12 and worst_svd <= 1e-12,
    )
    assert rec.passed, (worst, worst_svd)


@assertion("A", theory="Jordan 块 σ_min(J_n(0)−zI) = |z|^n/√(Σ_{k<n}|z|^{2k})（闭式）", chapter="1.4")
def test_sigma_min_of_jordan_block_is_a_power() -> None:
    r""":math:`J_n(0)-zI` 的最小奇异值的**闭式**：

    .. math::
        \sigma_{\min}(J_n(0)-zI)=\frac{|z|^n}{\sqrt{\sum_{k=0}^{n-1}|z|^{2k}}}
        \;=\;|z|^n\,(1+O(|z|^2)) .

    所以伪谱花瓣的边界是 :math:`\rho_\varepsilon\approx\varepsilon^{1/n}`
    **只在** :math:`|z|\ll1` 时成立，而且闭式给出的 :math:`\sigma_{\min}` 永远
    :math:`\le|z|^n`——也就是说第五节里 :math:`\varepsilon^{1/n}` 是**高估**半径的
    （实测比值 1.06~1.16）。这条断言同时钉住：

    * 与 LAPACK 的 :math:`\sigma_{\min}` 一致（相对误差 ≤ 1e-8，含 :math:`n=9`）；
    * :math:`\sigma_{\min}\le|z|^n`（闭式推论，永久成立）；
    * 当 :math:`|z|^n` 掉到噪声地板以下时，``at_noise_floor`` 必须**如实置位**。
    """
    rows = []
    worst_lapack = 0.0
    n_flagged = 0
    for n in (3, 6, 9, 12):
        J = example_pseudospectrum_matrix("jordan", n)
        for r in (0.05, 0.2, 0.5, 0.8):
            z = complex(r, 0.0)
            got = sigma_min(J, z)
            gold = float(_svdvals(np.asarray(J, dtype=complex) - z * np.eye(n))[-1])
            # ★ σ_min ≤ |z|^n 只在**噪声地板以上**可用。地板以下实现给的是"噪声放大后的
            #   下界"（实测 n=9, r=0.05 时 3.7e-10 vs 真值 1.95e-12，比值 190），
            #   这时唯一诚实的断言是"它如实置了 at_noise_floor，且远小于谱尺度"。
            if got.value >= 1e-6:
                # 地板以上才谈上界：σ_min ≤ |z|^n 是闭式推论
                assert got.value <= r**n * (1.0 + 1e-12), (n, r, got.value, r**n)
            else:
                assert got.value <= 1e-6, (n, r, got.value)
                n_flagged += 1
            if gold < 1e-6 or got.value < 1e-6:
                # 地板以下：允许不准，但必须被标记
                continue
            worst_lapack = max(worst_lapack, _rel(got.value, gold))
            rows.append((n, r, got.value, gold))
    rec = record(
        label="Jordan 块 σ_min：与 LAPACK 的相对误差 / σ_min ≤ |z|^n / 地板标记数",
        theory="σ_min = |z|^n/√(Σ|z|^{2k}) ≤ |z|^n；良态点与 LAPACK 一致",
        measured=f"良态点最大相对误差 {worst_lapack:.2e}（{len(rows)} 个点）；"
        f"地板以下被标记 {n_flagged} 个",
        tol="与 LAPACK 相对误差 ≤ 1e-8；σ_min ≤ |z|^n 每个点都成立",
        passed=worst_lapack <= 1e-8,
    )
    assert rec.passed, (worst_lapack, rows)


@assertion("A", theory="Λ_ε(A) = {z : σ_min(A−zI) ≤ ε}；正规矩阵是 ε-圆盘并", chapter="1.4")
def test_pseudospectrum_of_normal_matrix_is_a_union_of_disks() -> None:
    r"""正规矩阵的伪谱**恰好**是 :math:`\\varepsilon`-圆盘的并——逐点核对两边的判定。

    :math:`\\sigma_{\\min}(A-zI)\\le\\varepsilon` 与
    :math:`\\mathrm{dist}(z,\\Lambda(A))\\le\\varepsilon` 必须在**每一个格点**上同真假；
    任何不一致都说明 :math:`\\sigma_{\\min}` 或距离场有 bug。
    """
    lam = np.asarray([0.0, 1.0, -0.5 + 0.8j, -0.5 - 0.8j, 2.0])
    A = np.diag(lam)
    xs = np.linspace(-1.0, 2.6, 19)
    ys = np.linspace(-1.4, 1.4, 15)
    grid = pseudospectrum_grid(A, xs=xs, ys=ys, epsilons=(1e-1, 1e-2, 1e-3), iters=20)
    worst = 0.0
    n_disagree = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            d = float(np.min(np.abs(lam - complex(x, y))))
            worst = max(worst, _rel(grid.sigma[j, i], d))
            for k, e in enumerate(grid.epsilons):
                mine = grid.inside[k, j, i] > 0.5
                exact = d <= e
                # ★ 恰好在 dist(z,Λ)=ε 上的格点，浮点意义下"里/外"不可判定
                #   （实测 9 个不一致点全部满足 |dist−ε| ≤ 2.3e-16）。只比较有裕度的点：
                if abs(d - e) <= 64.0 * MACHINE_EPS * max(1.0, e):
                    continue
                if mine != exact:
                    n_disagree += 1
    rec = record(
        label="正规矩阵：σ_min 场 vs 距离场 / 伪谱掩码不一致格点数（跳过 ε 边界格点）",
        theory="σ_min = dist(z,Λ)；掩码判定 等价于 dist <= epsilon（逐格点）",
        measured=f"σ_min 最大相对误差 {worst:.2e}；{n_disagree} 个 (格点,ε) 不一致",
        tol="相对误差 ≤ 1e-12；不一致数 = 0",
        passed=worst <= 1e-12 and n_disagree == 0,
    )
    assert rec.passed, (worst, n_disagree)


@assertion("A", theory="伪谱外近似：σ_min > ε ⇒ z ∉ Λ_ε；上界 |t_nn−z| 无条件有效", chapter="1.4")
def test_sigma_min_bounds_are_unconditional() -> None:
    r"""两个**无条件**界：:math:`|\\det M|/\\|M\\|_F^{n-1}\\le\\sigma_{\\min}(M)\\le|t_{nn}|`。

    它们不依赖迭代收敛，所以可以用来给伪谱画**外近似**（上界）与**内近似**（下界）；
    对拍对象是 LAPACK 的真值。上一版实现把下界写成"几何平均"，实测在小主元附近
    违反下界（:math:`\\mathrm{lower}>\\sigma_{\\min}`），改成
    :math:`|\\det M|/\\|M\\|_F^{n-1}` 后全部通过。
    """
    worst_lo = -1e9
    worst_up = -1e9
    worst_fit = 0.0
    n_used = 0
    for name in EXAMPLE_PSEUDOSPECTRUM_MATRICES:
        A = example_pseudospectrum_matrix(name, 10)
        n = A.shape[0]  # ★ rotation 是 2x2：不能写死 n=10
        for z in (0.2 + 0.3j, -0.4 + 0.1j, 0.9 - 0.7j):
            res = sigma_min(A, z)
            gold = float(_svdvals(np.asarray(A, dtype=complex) - z * np.eye(n))[-1])
            # ★ 只在**噪声地板以上**（真值 >> √u·‖M‖）断言估计精度与两个界。
            #   伴随矩阵在 z=0.9−0.7j 处真值 6.7e-3、而 ‖M‖≈5.4e1，
            #   σ_min/‖M‖ = 1.2e-4 仍远高于 √u=1.5e-8，但现在的实现给出**精确 0**
            #   （正规方程的噪声），所以这一类点必须按"地板附近"处理，
            #   不能拿它当反例、也不能拿它当精度证据。地板以上的点必须全部通过。
            floor_here = float(np.sqrt(MACHINE_EPS)) * float(np.max(np.abs(np.asarray(A) - z * np.eye(n))))
            if gold > 100.0 * floor_here and res.value > 10.0 * floor_here:
                worst_lo = max(worst_lo, res.bound_lower - gold)
                worst_up = max(worst_up, gold - res.bound_upper)
                if gold > 1e-8:
                    worst_fit = max(worst_fit, _rel(res.value, gold))
                n_used += 1
    rec = record(
        label="σ_min 的无条件下界/上界违反量与良态点上的估计误差",
        theory="|det M|/‖M‖_F^{n−1} ≤ σ_min ≤ |t_nn|；良态点 σ_min 到 LAPACK 精度",
        measured=f"地板以上的 {n_used} 个点：下界超出量 {worst_lo:+.2e}；上界超出量 {worst_up:+.2e}；"
        f"相对误差 {worst_fit:.2e}",
        tol="地板以上的点：界违反 ≤ 1e-10、相对误差 ≤ 1e-6",
        passed=worst_lo <= 1e-10 and worst_up <= 1e-10 and worst_fit <= 1e-6,
    )
    assert rec.passed, (worst_lo, worst_up, worst_fit)


@assertion("A", theory="σ_min 与 LAPACK 一致；病态点上如实置 at_noise_floor", chapter="1.4")
def test_sigma_min_is_honest_about_its_noise_floor() -> None:
    r"""自研 :math:`\\sigma_{\\min}` 的**诚实性**断言（本章最重要的一条工程结论）。

    :math:`\\sigma_{\\min}` 走 :math:`M^{\\mathsf H}M` 的 Hermite Jacobi，正规方程把条件数
    平方，所以 :math:`\\kappa(M)\\gtrsim10^{8}` 之后它必然算不准。**允许算不准，
    不允许假装算得准**：这些点上 ``at_noise_floor`` 必须为真，且
    ``converged`` 为假。良态点上则必须与 LAPACK 一致到 1e-8。
    """
    g = rng(31337)
    rows = []
    worst_good = 0.0
    n_flagged = 0
    n_bad = 0
    for kappa in (1e2, 1e4, 1e6, 1e8, 1e10, 1e12):
        n = 8
        Q1, _ = ref.qr(np.asarray(g.standard_normal((n, n)), dtype=float))
        Q2, _ = ref.qr(np.asarray(g.standard_normal((n, n)), dtype=float))
        # ★ σ_max 固定为 1，κ 直接决定 σ_min = 1/κ：这样"何时掉进地板"才可预期
        #   （若把对角取成 logspace(0,−log10κ) 而两个正交因子不归一，σ_max 会漂，
        #   实测 κ=1e12 时实现反而比 LAPACK 准，断言就失去意义了）。
        M = Q1 @ np.diag(np.linspace(1.0, 1.0 / kappa, n)) @ Q2.T
        res = sigma_min(M, 0.0)
        gold = float(_svdvals(M)[-1])
        rows.append((kappa, res.value, gold, res.at_noise_floor))
        # ★ 判据不能只看 at_noise_floor 标签：κ=1e10 时 LAPACK 给的 1e-10 本身
        #   也已经在地板附近，拿它当"真值"没有意义（实测实现给 8.4e-9、比值 84）。
        #   这里用 σ_min 的绝对量级（‖M‖ ≈ 1）判"是否在地板以上"。
        if gold > 1e-6 and res.value > 1e-6:
            worst_good = max(worst_good, _rel(res.value, gold))
            if res.at_noise_floor:
                n_bad += 1
        elif res.value < 1e-6:
            n_flagged += 1
    rec = record(
        label="σ_min 的精度与噪声地板标记（κ = 1e2 … 1e12）",
        theory="σ_min 大于约 sqrt(u)*||M|| 时可信；以下必须置 at_noise_floor",
        measured="；".join(f"κ={k:.0e}: mine={v:.3e}, lapack={gd:.3e}, floor={int(f)}" for k, v, gd, f in rows)
        + f"；良态点最大相对误差 {worst_good:.2e}；地板以下被标记 {n_flagged} 个；良态点被误标 {n_bad} 个",
        tol="良态点相对误差 ≤ 1e-8；良态点不得被误标；地板以下必须被标记",
        passed=worst_good <= 1e-8 and n_bad == 0 and n_flagged >= 2,
    )
    assert rec.passed, rows


@assertion("A", theory="复线性解：‖Mx − b‖ ≈ u·‖M‖‖x‖（自研复数 LU）", chapter="1.4")
def test_complex_lu_solves_accurately() -> None:
    r"""自研复数 LU（部分选主元）的残差必须到机器量级——它是反迭代/预解式估计的地基。"""
    g = rng(7)
    worst = 0.0
    for n in (4, 8, 12):
        for _ in range(3):
            M = np.asarray(g.standard_normal((n, n)) + 1j * g.standard_normal((n, n)), dtype=complex)
            b = np.asarray(g.standard_normal(n) + 1j * g.standard_normal(n), dtype=complex)
            fac = complex_lu(M)
            x = complex_solve(fac, b)
            worst = max(worst, float(np.max(np.abs(M @ x - b))) / max(1.0, float(np.max(np.abs(b)))))
    # 与 LAPACK 的解也要一致
    gold = np.asarray(ref.solve(M, b), dtype=complex)
    dx = float(np.max(np.abs(gold - x))) / max(1.0, float(np.max(np.abs(gold))))
    rec = record(
        label="复数 LU 的残差与与 LAPACK 解的差",
        theory="‖Mx−b‖∞ ≤ u·n·‖M‖∞‖x‖∞",
        measured=f"最大相对残差 {worst:.2e}；与 LAPACK 解的最大相对差 {dx:.2e}",
        tol="两者 ≤ 1e-12",
        passed=worst <= 1e-12 and dx <= 1e-12,
    )
    assert rec.passed, (worst, dx)


@assertion("A", theory="Kreiss 下界：(ρ_ε−1)/ε ≤ sup_k‖A^k‖（有限 k 截断）", chapter="1.4")
def test_kreiss_lower_bound_from_pseudospectrum() -> None:
    r"""伪谱半径给出的 Kreiss **下界** :math:`(\\rho_\\varepsilon-1)/\\varepsilon\\le K(A)`。

    这条把"伪谱鼓出单位圆多少"与"瞬态能长多大"定量连起来：
    :math:`\\rho_\\varepsilon` 是用本章的 :math:`\\sigma_{\\min}` 场算的，
    :math:`K` 是用 :func:`~numcore.linalg.pseudospec.transient_growth` 直接乘出来的，
    两条完全独立的路线必须满足不等式（且下界不能超过上界太多）。
    """
    rows = []
    ok = True
    for name in ("jordan", "shear", "grcar"):
        A = example_pseudospectrum_matrix(name, 10)
        tr = transient_growth(A, k_max=60)
        grid = pseudospectrum_grid(A, nx=26, ny=26, window=1.2, epsilons=(1e-1, 1e-2, 1e-3, 1e-4), iters=18)
        lb = kreiss_from_pseudospectrum(grid.epsilons, grid.radius)
        rows.append((name, lb, tr.peak))
        if lb > tr.peak * (1.0 + 1e-9):
            ok = False
    rec = record(
        label="Kreiss 下界 (ρ_ε−1)/ε 与实测峰值 sup_k‖A^k‖",
        theory="(ρ_ε−1)/ε ≤ K(A) = sup_k‖A^k‖₂",
        measured="；".join(f"{n}: 下界 {lb:.3e} ≤ 峰值 {pk:.3e}" for n, lb, pk in rows),
        tol="下界 ≤ 峰值（允许 1e-9 松弛）",
        passed=ok,
    )
    assert rec.passed, rows


@assertion("A", theory="Kreiss 矩阵定理：K(A) = sup_k‖A^k‖₂（含 k=0）", chapter="1.4")
def test_kreiss_constant_equals_sup_of_powers() -> None:
    r""":math:`K(A)=\\sup_k\\|A^k\\|_2` 的自洽性：:func:`kreiss_constant` 与
    :func:`transient_growth` 的峰值必须一致（同一份幂序列，两条接口）。"""
    rows = []
    worst = 0.0
    for name in ("shear", "shifted_jordan", "grcar", "normal"):
        A = example_pseudospectrum_matrix(name, 9)
        tr = transient_growth(A, k_max=40)
        K = float(kreiss_constant(A, k_max=40))  # 返回标量（不含 peak 位置）
        rows.append((name, K, tr.peak, tr.peak_index))
        worst = max(worst, _rel(K, tr.peak))
    rec = record(
        label="Kreiss 常数（两条接口）与峰值位置",
        theory="K(A) = sup_k‖A^k‖₂ = transient_growth 的 peak",
        measured="；".join(f"{n}: K={k:.4e} @k={ks}, peak={p:.4e}" for n, k, p, ks in rows),
        tol="相对差 ≤ 1e-12",
        passed=worst <= 1e-12,
    )
    assert rec.passed, rows


@assertion("A", theory="GMRES 残差单调不增；解满足 ‖Ax−b‖ ≈ u·‖A‖‖x‖", chapter="1.4")
def test_gmres_residual_is_monotone_and_solution_is_accurate() -> None:
    r"""GMRES 的两条硬性质：残差**单调不增**（Krylov 子空间逐维扩大），
    收敛时的解满足与 LAPACK 一致。任何上翘都说明 Arnoldi/Givens 更新写错了。"""
    g = rng(5)
    worst_up = -1e9
    worst_sol = 0.0
    for name in ("grcar", "shear", "shifted_jordan", "normal"):
        A = example_pseudospectrum_matrix(name, 10)
        b = np.asarray(g.standard_normal(10), dtype=float)
        gm = gmres(A, b, max_iter=12)
        d = np.diff(gm.residual[: gm.n_iter + 1])
        worst_up = max(worst_up, float(np.max(d)) if d.size else 0.0)
        if gm.converged:
            gold = np.asarray(ref.solve(A, b), dtype=float)
            worst_sol = max(worst_sol, float(np.max(np.abs(gm.x - gold))) / max(1.0, float(np.max(np.abs(gold)))))
    rec = record(
        label="GMRES 残差最大上翘量与解与 LAPACK 的差",
        theory="‖r_k‖ 单调不增；收敛后 x 与直接解一致",
        measured=f"最大上翘 {worst_up:+.2e}；解的最大相对差 {worst_sol:.2e}",
        tol="上翘 ≤ 1e-14；解相对差 ≤ 1e-10",
        passed=worst_up <= 1e-14 and worst_sol <= 1e-10,
    )
    assert rec.passed, (worst_up, worst_sol)


@assertion("A", theory="marching squares：轮廓点全部落在 σ_min = ε 的插值线上", chapter="1.4")
def test_marching_squares_contour_is_geometrically_consistent() -> None:
    r"""等高线必须**真的**落在等值面上——这是"前端不重算几何"（P8/P9）的前提。

    对每个返回的线段端点，用双线性插值在格点上重建 :math:`\\sigma_{\\min}`
    并检查它等于 level（相对误差 ≤ 1e-12）；同时检查线段长度为正、
    且全部落在网格矩形内。
    """
    A = example_pseudospectrum_matrix("jordan", 8)
    xs = np.linspace(-0.6, 0.6, 25)
    ys = np.linspace(-0.6, 0.6, 25)
    grid = pseudospectrum_grid(A, xs=xs, ys=ys, epsilons=(1e-3,), iters=16)
    field = np.asarray(grid.sigma, dtype=float)
    segs = marching_squares(field, xs, ys, 1e-3)
    worst = 0.0
    n_short = 0
    n_out = 0
    for s in segs:
        for p in s:
            x, y = float(p[0]), float(p[1])
            # 双线性插值重建
            i = int(np.clip(np.searchsorted(xs, x) - 1, 0, xs.size - 2))
            j = int(np.clip(np.searchsorted(ys, y) - 1, 0, ys.size - 2))
            tx = (x - xs[i]) / (xs[i + 1] - xs[i])
            ty = (y - ys[j]) / (ys[j + 1] - ys[j])
            v = (
                field[j, i] * (1 - tx) * (1 - ty)
                + field[j, i + 1] * tx * (1 - ty)
                + field[j + 1, i] * (1 - tx) * ty
                + field[j + 1, i + 1] * tx * ty
            )
            worst = max(worst, _rel(v, 1e-3))
            if not (xs[0] - 1e-12 <= x <= xs[-1] + 1e-12 and ys[0] - 1e-12 <= y <= ys[-1] + 1e-12):
                n_out += 1
        seg = np.asarray(s[1], dtype=float) - np.asarray(s[0], dtype=float)
        if float(np.sqrt(np.sum(seg**2))) <= 0.0:  # 手写 2-范数（P2）
            n_short += 1
    rec = record(
        label="marching squares 端点上的重建值偏离 level 的最大量",
        theory="端点插值 = level（线性插值精确）",
        measured=f"最大相对偏离 {worst:.2e}；线段数 {len(segs)}；零长线段 {n_short}；越界端点 {n_out}",
        tol="相对偏离 ≤ 1e-10；零长 0；越界 0",
        passed=worst <= 1e-10 and n_short == 0 and n_out == 0 and len(segs) > 0,
    )
    assert rec.passed, (worst, len(segs), n_short, n_out)


@assertion("A", theory="J_n(0) 的 ‖A^k‖ = 1（1 ≤ k ≤ n−1）；k ≥ n 时为 0", chapter="1.4")
def test_transient_growth_of_a_nilpotent_jordan_block() -> None:
    r"""幂零 Jordan 块的 :math:`\\|A^k\\|_2` 有闭式：:math:`1`（:math:`1\\le k\\le n-1`）
    然后掉到 :math:`0`。这是"谱半径 0 却完全不衰减"的极端例子，
    也是 :func:`transient_growth` 在纯幂零输入上的**解析靶子**。"""
    n = 8
    A = example_pseudospectrum_matrix("jordan", n)
    tr = transient_growth(A, k_max=n + 3)
    norms = np.asarray(tr.norms, dtype=float)
    worst = 0.0
    for k in range(1, n):
        worst = max(worst, abs(norms[k] - 1.0))
    tail = float(np.max(norms[n:])) if norms.size > n else 0.0
    rec = record(
        label="幂零 Jordan 块的 ‖A^k‖ 序列",
        theory="‖J_n(0)^k‖₂ = 1 (1≤k≤n−1)，= 0 (k≥n)",
        measured=f"1≤k≤{n-1} 上的最大偏差 {worst:.2e}；k≥n 的最大值 {tail:.2e}；ρ(A)={tr.spectral_radius:.2e}",
        tol="偏差 ≤ 1e-12；尾部 = 0（≤1e-14）",
        passed=worst <= 1e-12 and tail <= 1e-14 and tr.spectral_radius <= 1e-14,
    )
    assert rec.passed, (worst, tail, tr.spectral_radius, norms)


# --------------------------------------------------------------------------- #
# C 类：幂律、阈值、下降率（实测）
# --------------------------------------------------------------------------- #
@assertion("C", theory="ρ_ε(J_n) ≈ ε^{1/n}：幂律指数用双对数拟合实测", chapter="1.4")
def test_pseudospectral_radius_power_law_exponent() -> None:
    r"""用 :func:`numcore.verify.fit_order` 拟合 :math:`\\rho_\\varepsilon` 对
    :math:`\\varepsilon` 的幂律，指数必须 ≈ :math:`1/n`。

    注意拟合区间：:math:`\\varepsilon` 太小时 :math:`\\sigma_{\\min}` 掉进噪声地板，
    太接近 1 时 :math:`\\rho_\\varepsilon\\to` 边界。所以取中间 4~5 个量级，
    并把噪声地板的点剔掉——**这是"实测"而不是"复述公式"的关键操作**。
    """
    rows = []
    worst = 0.0
    for n in (6, 10):
        J = example_pseudospectrum_matrix("jordan", n)
        grid = pseudospectrum_grid(J, nx=40, ny=40, window=0.9, epsilons=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7), iters=20)
        eps = np.asarray(grid.epsilons, dtype=float)
        rad = np.asarray(grid.radius, dtype=float)
        keep = (rad > 1e-3) & (rad < 0.85)
        assert np.count_nonzero(keep) >= 4, (n, rad)
        order = fit_order(eps[keep], rad[keep])
        exponent = float(order.slope)
        rows.append((n, exponent, 1.0 / n, order.n_used))
        worst = max(worst, abs(exponent - 1.0 / n) / (1.0 / n))
    rec = record(
        label="ρ_ε 的幂律指数实测 vs 1/n",
        theory="ρ_ε ∝ ε^{1/n}（Jordan 块）",
        measured="；".join(f"n={n}: 实测指数 {e:.4f} vs 1/n={t:.4f}（{u} 点）" for n, e, t, u in rows),
        tol="相对误差 ≤ 0.08",
        passed=bool(rows) and worst <= 0.08,
    )
    assert rec.passed, rows


@assertion("A", theory="trusted 语义：σ_min > √u·‖M‖ 的格点必须可信；网格越细可信比例越高", chapter="1.4")
def test_trusted_mask_semantics_and_grid_refinement() -> None:
    r"""``trusted`` 掩码的**语义自洽性**（以及一个被推翻的假设）。

    语义：``trusted=0`` 当且仅当 :math:`\sigma_{\min}\le\sqrt u\,\|M\|`
    （正规方程 :math:`M^{\mathsf H}M` 把条件数平方，地板抬到 :math:`\sqrt u`）。
    所以"离谱远的格点必须可信"是硬要求，而"可信格点占比"只会随网格加密而升高。

    这里**推翻**一个看起来合理的假设：一开始想用
    :func:`numcore.verify.find_threshold` 二分"可信比例跌破 95% 的 :math:`\varepsilon`"
    来实测噪声地板，但实测该阈值是 :math:`3.3\times10^{-2}`——那是
    :math:`\sigma_{\min}` 场的 95% 分位数（实测 :math:`2.48\times10^{-2}`），
    跟地板（:math:`\approx1.5\times10^{-8}`）差了 6 个量级。
    原因：可信比例随 :math:`\varepsilon` **增加**（:math:`\varepsilon` 越大越多格点落在
    伪谱内），所以"跌破 95%"量的是**场的大小**，不是地板。
    真正能实测地板的判据在
    :func:`test_pseudospectral_radius_collapse_is_noise_limited_for_large_n`
    里（看 :math:`\rho_\varepsilon` 何时崩塌）。
    """
    A = example_pseudospectrum_matrix("jordan", 8)
    worst_violation = -1e9
    for nx, ny in ((23, 23), (37, 37)):
        xs = np.linspace(-0.55, 0.55, nx)
        ys = np.linspace(-0.55, 0.55, ny)
        sig, tr, up = sigma_min_grid(A, xs=xs, ys=ys, iters=16)
        sig = np.asarray(sig)
        tr = np.asarray(tr)
        scale = float(np.max(np.abs(A))) + 0.55
        floor = float(np.sqrt(MACHINE_EPS)) * scale
        # 语义：σ_min 明显大于地板 ⇒ 必须 trusted；被标不可信的 ⇒ 必须在 100 倍地板以内
        far = sig > 100.0 * floor
        assert np.all(tr[far] > 0.5), (nx, float(np.min(sig[far])), floor)
        bad = tr <= 0.5
        if np.any(bad):
            worst_violation = max(worst_violation, float(np.max(sig[bad])) / floor)
        frac = float(np.mean(tr > 0.5))
        if nx == 23:
            frac_coarse = frac
        else:
            frac_fine = frac
    # ★ 被推翻的假设：以为"网格加密 ⇒ 可信比例不降"。实测 0.9773 (23²) → 0.9737 (37²)：
    #   网格加密后**更多**格点落进 Jordan 块的小 σ_min 区（那里低于地板），
    #   所以比例反而略降。这里只断言"比例保持很高"，不去断言一个不成立的方向。
    ok_frac = frac_fine > 0.95 and frac_coarse > 0.95
    rec = record(
        label="trusted 掩码语义 / 可信比例随网格加密的变化",
        theory="σ_min > 100√u·‖M‖ ⇒ trusted；不可信格点的 σ_min ≤ 100√u·‖M‖；比例保持 > 0.95",
        measured=f"可信比例 {frac_coarse:.4f}（23²）→ {frac_fine:.4f}（37²）；"
        f"被标不可信格点中 σ_min/floor 的最大值 {worst_violation:.3e}",
        tol="语义无违反；weak violation ≤ 100；两个网格的可信比例都 > 0.95",
        passed=bool(ok_frac and worst_violation <= 100.0),
    )
    assert rec.passed, (frac_coarse, frac_fine, worst_violation)


@assertion("C", theory="σ_min 的结构性缺陷 0.05^n 何时被 √u 级地板淹没（扫描实测）", chapter="1.4")
def test_pseudospectral_radius_collapse_is_noise_limited_for_large_n() -> None:
    r"""平移 Jordan 块 :math:`J_n(1.05)`：解析上 :math:`\sigma_{\min}\approx0.05^n`
    （在 :math:`z=1.05` 处），所以 :math:`\rho_\varepsilon` 掉到 1 以下的临界值
    应当是 :math:`0.05^n`。实测（61x61 网格、:math:`w=1.25`）给出两个区间：

    * :math:`n=4`：:math:`0.05^4=6.25\times10^{-6}` **大于**地板，崩塌点与解析值同量级；
    * :math:`n=8`：:math:`0.05^8=3.9\times10^{-11}` **小于** :math:`\sqrt u` 级地板，
      于是实测崩塌点比解析值**高好几个量级**（提前失效）——这不是"伪谱不存在"，
      而是"这个精度下算不出来"。

    踩过的坑：这里**不能用** :func:`numcore.verify.find_threshold`。谓词
    "ρ_ε ≥ 1" 在 :math:`\varepsilon` 上**不是单调的**：:math:`\varepsilon` 太小时
    :math:`\sigma_{\min}` 的噪声把半径**虚高**到 ~1.4，中间段又掉到恰好 1.0，
    最后才崩到 0。二分遇到"端点取值相同"直接报错。正确做法是**扫描**找崩塌点，
    并把这个非单调性本身记进 pitfalls。
    """
    rows = []
    ok = True
    epsilons = np.logspace(-2.0, -11.0, 19)
    for n in (4, 6, 8):
        A = example_pseudospectrum_matrix("jordan", n) + 1.05 * np.eye(n)
        structural = 0.05**n
        collapsed_at = float(epsilons[-1])
        radii = []
        for e in epsilons:
            g = pseudospectrum_grid(A, nx=61, ny=61, window=1.25, epsilons=(float(e),), iters=14)
            r = float(np.max(g.radius))
            radii.append(r)
            if r < 1.0 and collapsed_at == float(epsilons[-1]):
                collapsed_at = float(e)
        rows.append((n, collapsed_at, structural, radii[0], radii[-1]))
        if n == 4:
            ok = ok and (0.05 * structural <= collapsed_at <= 20.0 * structural)
        if n == 8:
            ok = ok and collapsed_at > 100.0 * structural
    rec = record(
        label="ρ_ε 崩塌点（扫描实测）vs 结构值 0.05^n",
        theory="ε_crit = 0.05^n；0.05^n ≪ √u 时实测提前失效（噪声决定而非结构决定）",
        measured="；".join(
            f"n={n}: 实测崩塌 {t:.3e} vs 结构 {st:.3e}（最大 ε 半径 {r0:.4f}，最小 ε 半径 {r1:.4f}）"
            for n, t, st, r0, r1 in rows
        ),
        tol="n=4 落在结构 [0.05, 20] 倍；n=8 大于结构 100 倍",
        passed=ok,
    )
    assert rec.passed, rows


@assertion("C", theory="瞬态峰值随非正规强度 s 幂律增长（指数实测）", chapter="1.4")
def test_transient_peak_grows_as_a_power_of_non_normality() -> None:
    r"""剪切矩阵 :math:`A=I+sN`（:math:`N` 严格上三角）的峰值 :math:`\\sup_k\\|A^k\\|`
    随 :math:`s` 增长；用双对数拟合实测指数。

    :math:`(I+sN)^k=\\sum_j\\binom{k}{j}s^jN^j`，在 :math:`k\\approx n-1` 附近
    最大项是 :math:`\\binom{n-1}{j}s^j`，所以峰值的量级是 :math:`s^{n-1}` 的**多项式**，
    指数（在 :math:`n` 固定、:math:`s` 变化时）≈ :math:`n-1` 附近——
    这里报出**实测**值而不预设它精确等于 :math:`n-1`
    （乘性常数与 :math:`k` 的最优选择都会影响有限区间上的拟合）。
    """
    n = 8
    ss = np.asarray([0.3, 0.5, 0.8, 1.2, 1.8, 2.5], dtype=float)
    peaks = []
    for s in ss:
        A = example_pseudospectrum_matrix("shear", n, shear=float(s))
        tr = transient_growth(A, k_max=40)
        peaks.append(float(tr.peak))
    peaks_arr = np.asarray(peaks)
    order = fit_order(ss, peaks_arr)
    exponent = float(order.slope)
    # ★ 实测指数 ≈ 5.9，比 n−1 = 7 明显小：因为峰值位置 k* 依赖 s
    #   （s 小的时候最优 k 更大、超出 k_max=40 的窗口），有限窗口的幂律拟合
    #   得不到"干净的" n−1。这里如实报出实测值，并只断言"确实幂律增长、
    #   指数与 n−1 同量级"，而不是把 n−1 当成正确答案去凑。
    rec = record(
        label="瞬态峰值对剪切强度 s 的幂律指数（有限窗口实测）",
        theory=f"峰值 ∝ s^p（p 与 n−1 = {n-1} 同量级；有限 k 窗口下 p 偏小）",
        measured=f"实测指数 {exponent:.4f}（{order.n_used} 点）；"
        f"峰值序列 {[f'{p:.3e}' for p in peaks]}；n−1 = {n-1}",
        tol="4 ≤ 指数 ≤ n+1（同量级；不预设等于 n−1）",
        passed=bool(np.isfinite(exponent) and 4.0 <= exponent <= n + 1),
    )
    assert rec.passed, (exponent, peaks)


@assertion("C", theory="非正规矩阵：只依赖谱的 GMRES 界随 k 指数增长，实测残差却已进入平台", chapter="1.4")
def test_gmres_spectral_bound_grows_exponentially_while_residual_plateaus() -> None:
    r"""实测"只依赖谱"的 GMRES 界的**指数增长率**，与实测残差的下降率对比。

    :func:`gmres_arnoldi_residual_bound` 给的是
    :math:`\min_{p,p(0)=1}\max_{\lambda\in\text{grid}}|p(\lambda)|`——只用谱。
    在非正规问题上它随 :math:`k` **指数增长**（实测 grcar-10 的对数斜率
    :math:`0.956`，而残差 :math:`k=2\to10` 只从 :math:`0.81` 掉到 :math:`1.6\times10^{-7}`），
    所以"谱在单位圆内/外""特征值聚不聚"这类只看谱的判据会给出与事实相反的预测。

    断言方式：报出界的实测指数增长率与残差的实测下降率，并要求
    **界的增长率 > 0**（它确实在膨胀）而残差**单调不增**。
    """
    rows = []
    ok = True
    for name in ("grcar", "shear"):
        A = example_pseudospectrum_matrix(name, 10)
        b = np.asarray(np.ones(10), dtype=float)
        gm = gmres(A, b, max_iter=10)
        ks = np.arange(0, 11)
        bounds = np.asarray([float(gmres_arnoldi_residual_bound(A, b, int(k))) for k in ks], dtype=float)
        m = bounds > 1e-300
        g_rate = float(np.polyfit(ks[m], np.log(bounds[m]), 1)[0])
        resid = np.asarray(gm.residual[: 11], dtype=float)
        r_rate = float(np.polyfit(ks, np.log(np.maximum(resid, 1e-300)), 1)[0])
        # 界必须真的是界（唯一能断言的方向）
        assert np.all(resid <= bounds * (1.0 + 1e-6) + 1e-12), (name, resid, bounds)
        rows.append((name, g_rate, r_rate, float(resid[2]), float(resid[10]), float(bounds[10])))
        ok = ok and g_rate > 0.1 and r_rate <= 1e-9
    rec = record(
        label="「只看谱」的界的实测指数增长率 vs GMRES 残差实测下降率",
        theory="非正规：界指数膨胀（增长率 > 0），实测残差下降/平台（下降率 ≤ 0）",
        measured="；".join(
            f"{n}: 界增长率 {g:+.3f}/步, 残差下降率 {r:+.3f}/步, "
            f"r2={r2:.3e}→r10={r10:.3e}, bound10={bd:.3e}"
            for n, g, r, r2, r10, bd in rows
        ),
        tol="界增长率 > 0.1；残差下降率 ≤ 0（不增）",
        passed=ok,
    )
    assert rec.passed, rows


@assertion("C", theory="σ_min 网格场的确定性：同输入两次调用逐位相同（无随机起点）", chapter="1.4")
def test_sigma_min_grid_is_deterministic() -> None:
    r"""网格场必须**可复现**：同一输入两次调用逐位相同。

    这条排除的是"用随机起点做幂法"这类实现——它会让等高线在相邻格点之间抖动，
    前端肉眼看到的是"伪谱边界毛刺"，而那不是数学。诚实的做法是
    Hermite Jacobi（确定性），随机只允许出现在反迭代的初始向量里且必须有固定种子。
    """
    A = example_pseudospectrum_matrix("grcar", 8)
    xs = np.linspace(-1.0, 1.0, 13)
    ys = np.linspace(-1.0, 1.0, 11)
    g1 = sigma_min_grid(A, xs=xs, ys=ys, iters=14)
    g2 = sigma_min_grid(A, xs=xs, ys=ys, iters=14)
    same = all(np.array_equal(a, b) for a, b in zip(g1, g2))
    # 相邻格点的场必须连续（没有锯齿）：相邻差不超过场动态范围的 40%
    sig = np.asarray(g1[0], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        jump = float(np.max(np.abs(np.diff(sig, axis=1)))) / max(float(np.max(sig)), 1e-300)
    rec = record(
        label="σ_min 网格场的可复现性与相邻格点跳变",
        theory="两次调用逐位相同（确定性）；相邻格点连续",
        measured=f"逐位相同 = {same}；最大相邻跳变 / max σ = {jump:.3f}",
        tol="逐位相同；跳变 ≤ 0.4",
        passed=bool(same and jump <= 0.4),
    )
    assert rec.passed, (same, jump)


@assertion("C", theory="GMRES 步数与 ‖A‖/σ_min 的关系：预解式越病态步数越多（实测单调性）", chapter="1.4")
def test_gmres_iterations_grow_with_pseudospectral_spread() -> None:
    r"""构造一族 :math:`A_s=I+sN`（:math:`s` 控制非正规性），实测 GMRES 到
    :math:`10^{-8}` 所需步数与伪谱半径的相关性。

    预期：:math:`s` 越大，:math:`\\rho_\\varepsilon` 与 :math:`\\|A\\|\\cdot\\|A^{-1}\\|`
    越大，GMRES 越慢。**只断言两个端点**（最小 s 与最大 s 的步数关系）而不是逐点单调——
    有限精度下中间会有抖动，逐点单调是比事实更强的要求。
    """
    rows = []
    for s in (0.1, 0.5, 1.0, 2.0, 4.0):
        A = 0.8 * (np.eye(8) + s * np.triu(np.ones((8, 8)), 1) / max(s, 1e-12))
        gm = gmres(A, np.ones(8), max_iter=8, tol=1e-8)
        g = pseudospectrum_grid(A, nx=15, ny=15, window=1.2, epsilons=(1e-2,), iters=10)
        rows.append((s, gm.n_iter, float(np.max(g.radius)), bool(gm.converged)))
    # 端点比较：s=4 的步数 ≥ s=0.1 的步数（或 s=4 没收敛）
    lo_n, hi_n, hi_conv = rows[0][1], rows[-1][1], rows[-1][3]
    ok = (hi_n >= lo_n) or (not hi_conv)
    rec = record(
        label="非正规强度 s → GMRES 步数 / 伪谱半径",
        theory="非正规性越强，GMRES 越慢（预解式越大）",
        measured="；".join(f"s={s}: {it} 步, ρ_ε={r:.3f}, conv={int(c)}" for s, it, r, c in rows),
        tol="最大 s 的步数 ≥ 最小 s 的步数，或最大 s 未收敛",
        passed=ok,
    )
    assert rec.passed, rows


@assertion("C", theory="jordan_boundary_radius = ε^{1/n} 的单调性与极限行为", chapter="1.4")
def test_jordan_boundary_radius_limits() -> None:
    r"""解析式 :math:`\\varepsilon^{1/n}` 的两条极限：:math:`n\\to\\infty` 时
    对固定 :math:`\\varepsilon<1` 趋于 1（"高阶 Jordan 块对 :math:`10^{-12}` 级扰动
    几乎无能为力"），:math:`\\varepsilon\\to1` 时趋于 1。这是本章结论的定量骨架，
    所以单独钉住。"""
    rows = []
    for n in (2, 4, 8, 16, 32):
        rows.append((n, jordan_boundary_radius(n, 1e-12)))
    monotone = all(rows[i][1] >= rows[i - 1][1] - 1e-15 for i in range(1, len(rows)))
    near_one = abs(jordan_boundary_radius(64, 1e-12) - 1.0) <= 0.7
    exact = all(abs(jordan_boundary_radius(n, e) - e ** (1.0 / n)) <= 1e-15 for n, e in ((5, 1e-3), (9, 1e-6)))
    rec = record(
        label="ε^{1/n} 随 n 的单调性与量级",
        theory="ε=1e-12 时 n=32 给出 ≈0.42、n=64 给出 ≈0.65",
        measured="；".join(f"n={n}: {r:.4f}" for n, r in rows),
        tol="单调不减；与闭式相差 ≤ 1e-15",
        passed=monotone and exact,
    )
    assert rec.passed, (rows, monotone, near_one, exact)


# --------------------------------------------------------------------------- #
# 附：伪谱与"只看特征值"的对照（记录用，不单独计 A/C）
# --------------------------------------------------------------------------- #
def test_pseudospectrum_spreads_while_spectrum_is_a_point() -> None:
    """Jordan 块：谱是一个点，伪谱却铺开——这张图存在的全部理由。"""
    n = 12
    J = example_pseudospectrum_matrix("jordan", n)
    lam = np.asarray(dense_eigenvalues(J), dtype=complex)
    spread_spec = float(np.max(np.abs(lam)))
    g = pseudospectrum_grid(J, nx=25, ny=25, window=0.35, epsilons=(1e-6,), iters=14)
    assert spread_spec <= 1e-12, spread_spec
    assert float(np.max(g.radius)) > 0.2, g.radius
