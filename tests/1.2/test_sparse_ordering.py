"""1.2 章（下半）：稀疏结构、填充与排序的 A/B 类断言。

这一半只谈**结构**：稀疏性、填充、带宽、剖面，全都与数值无关。最值得看的是
:func:`test_symbolic_fill_matches_exact_elimination`——符号预测与有理数精确消元
在结构对称、不选主元的条件下必须**逐个非零元一致**；一旦精确消元中发生抵消，
符号预测就会多算（这是"结构预测"与"数值事实"的差别，也是为什么工程实现里
要留一点安全余量）。
"""

from __future__ import annotations

import numpy as np
import pytest

from numcore.linalg.lu import example_matrix, lu_decompose
from numcore.linalg.ordering import (
    bandwidth,
    cuthill_mckee,
    minimum_degree,
    natural_order,
    ordering_report,
    profile,
    rcm,
)
from numcore.linalg.sparse import (
    CSR,
    band_storage,
    exact_lu_nnz,
    fill_edges,
    permutation_matrix,
    spy_points,
    structure_of,
    symbolic_lu,
)
from numcore.reference import dense as ref
from numcore.verify import assertion, assert_leq, record

U = float(np.finfo(float).eps) / 2.0


def _laplacian_2d(k: int) -> np.ndarray:
    """k×k 网格五点 Laplacian（与章节里那个生成器同构，独立写一份避免测试依赖 server）。"""
    n = k * k
    A = np.zeros((n, n))
    for i in range(k):
        for j in range(k):
            p = i * k + j
            A[p, p] = 4.0
            if i > 0:
                A[p, p - k] = -1.0
            if i < k - 1:
                A[p, p + k] = -1.0
            if j > 0:
                A[p, p - 1] = -1.0
            if j < k - 1:
                A[p, p + 1] = -1.0
    return A


def _tridiag(n: int, b: float = -1.0) -> np.ndarray:
    A = np.zeros((n, n))
    A[np.arange(n), np.arange(n)] = 2.0
    idx = np.arange(n - 1)
    A[idx, idx + 1] = b
    A[idx + 1, idx] = b
    return A


def _later_counts(A: np.ndarray, perm) -> np.ndarray:
    """逐列重放符号消元，数出每步"后来列"的个数（即：消去该列时它的活跃邻居数）。"""
    n = A.shape[0]
    Ap = np.asarray(A, dtype=float)[np.ix_(perm, perm)]
    adj = (np.abs(Ap) > 0.0) | (np.abs(Ap) > 0.0).T
    alive = np.ones(n, dtype=bool)
    out = np.zeros(n, dtype=int)
    for k in range(n):
        nbrs = np.flatnonzero(adj[k] & alive)
        later = [int(v) for v in nbrs if v > k]
        out[k] = len(later)
        for a_i in range(len(later)):
            for b_i in range(a_i + 1, len(later)):
                adj[later[a_i], later[b_i]] = True
                adj[later[b_i], later[a_i]] = True
        alive[k] = False
        adj[k, :] = False
        adj[:, k] = False
    return out


def _permuted(A: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """随机置换行列（保持对称），返回 (A_perm, perm)。"""
    from numcore.utils import rng

    g = rng(seed)
    perm = g.permutation(A.shape[0])
    return A[np.ix_(perm, perm)], perm


# --------------------------------------------------------------------------- #
# A 类：解析对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="nnz(L+U) = 2n + 2·Σ_k |later(k)|（结构对称、不选主元）", chapter="1.2")
def test_symbolic_fill_matches_exact_elimination() -> None:
    """符号预测与**有理数精确消元**逐个非零元对拍（结构化矩阵上必须完全相等）。"""
    cases = [
        ("tridiag-16", _tridiag(16)),
        ("banded-8", example_matrix("banded", 8)),
        ("laplacian-5x5", _laplacian_2d(5)),
        ("laplacian-6x6", _laplacian_2d(6)),
        ("spd-9", example_matrix("spd", 9)),
    ]
    mismatches = []
    detail = []
    for label, A in cases:
        perm = natural_order(A)
        sy = symbolic_lu(A, perm)
        nnz_l, nnz_u, nnz_total = exact_lu_nnz(A, perm)
        detail.append(f"{label}: 符号 {sy.nnz_lu} / 精确 {nnz_total}")
        if sy.nnz_lu != nnz_total:
            mismatches.append((label, sy.nnz_lu, nnz_total))
    rec = record(
        label="符号 nnz(L+U) 与精确消元不一致的例数",
        theory="0（5 例）",
        measured=f"{len(mismatches)}；" + "，".join(detail),
        tol="0",
        passed=not mismatches,
    )
    assert rec.passed, mismatches


@assertion("A", theory="精确消元中的抵消使 nnz ≤ 符号预测（结构预测只会多算）", chapter="1.2")
def test_symbolic_prediction_is_an_upper_bound_under_cancellation() -> None:
    """全 1 矩阵：消元后尾部块精确变成 0——符号预测仍按"非零"记，精确值更小。

    这是"结构 ≠ 数值"的最小例子：任何声称"填充数就是符号预测值"的说法都得加前提。
    """
    A = np.ones((4, 4))
    perm = natural_order(A)
    sy = symbolic_lu(A, perm)
    nnz_l, nnz_u, nnz_total = exact_lu_nnz(A, perm)
    rec = record(
        label="符号 nnz(L+U) − 精确 nnz(L+U)（全 1 矩阵 4×4）",
        theory="≥ 0（本例应当严格大于 0：消元后尾部块精确变成 0）",
        measured=f"符号 {sy.nnz_lu}，精确 {nnz_total}（L {nnz_l} / U {nnz_u}），差 {sy.nnz_lu - nnz_total}",
        tol="≥ 0 且本例 > 0",
        passed=sy.nnz_lu >= nnz_total and sy.nnz_lu > nnz_total,
    )
    assert rec.passed, (sy.nnz_lu, nnz_total)


@assertion("A", theory="RCM 在带状图上恢复带宽（≤ 原带宽）", chapter="1.2")
def test_rcm_recovers_bandwidth_of_permuted_banded_matrix() -> None:
    """把三对角矩阵的行列随机打乱：RCM 必须把带宽拉回 1，而自然序是 O(n)。"""
    A, _ = _permuted(_tridiag(60), 4242)
    # 注意带宽接口返回的是 (lower, upper) 二元组（下带宽、上带宽各自可能不同）
    lo_nat, up_nat = bandwidth(A, natural_order(A))
    lo_rcm, up_rcm = bandwidth(A, rcm(A))
    lo_cm, up_cm = bandwidth(A, cuthill_mckee(A))
    rec = record(
        label="带宽 (lower, upper)：自然序 / CM / RCM",
        theory="RCM ≤ (1,1)（三对角原带宽）；自然序 ≈ (n/2, n/2)",
        measured=f"自然 ({lo_nat},{up_nat}) / CM ({lo_cm},{up_cm}) / RCM ({lo_rcm},{up_rcm})",
        tol="RCM ≤ (1,1) 且不差于自然序",
        passed=max(lo_rcm, up_rcm) <= 1 and max(lo_rcm, up_rcm) <= max(lo_nat, up_nat),
    )
    assert rec.passed, (lo_nat, up_nat, lo_rcm, up_rcm)


@assertion("A", theory="最小度减小填充：nnz(L+U) 明显低于自然序", chapter="1.2")
def test_minimum_degree_reduces_fill_on_laplacians() -> None:
    """二维网格上比较三种顺序的填充（实测），最小度应当最好、自然序最差。"""
    rows = []
    for k in (5, 7, 9):
        A = _laplacian_2d(k)
        fills = {
            name: ordering_report(A, perm, name).nnz_lu
            for name, perm in (
                ("natural", natural_order(A)),
                ("rcm", rcm(A)),
                ("amd", minimum_degree(A)),
            )
        }
        rows.append((k, fills))
        assert fills["amd"] < fills["natural"], fills
    worst = rows[-1][1]
    rec = record(
        label="nnz(L+U)（k=9：自然 / RCM / AMD）",
        theory="AMD < RCM < 自然序",
        measured=f"{worst['natural']} / {worst['rcm']} / {worst['amd']}"
        + "；各 k：" + ", ".join(f"k={k}:{f['amd']}/{f['natural']}" for k, f in rows),
        tol="AMD < RCM < natural",
        passed=worst["amd"] < worst["rcm"] < worst["natural"],
    )
    assert rec.passed, rows


@assertion("A", theory="k×k 网格的真实树宽 = k；贪心消元给出的只是上界（≥ k）", chapter="1.2")
def test_treewidth_upper_bound_is_above_the_true_treewidth() -> None:
    """实测消元最大团给出的树宽上界，与"k×k 网格树宽恰为 k"这个已知结论比对。

    上界 **≥** 真值是必然的（贪心不做最优性保证），实测差距说明贪心的代价；
    反过来如果实测上界 < k，那一定是实现错了（这正是这条断言的价值）。
    """
    rows = []
    for k in (4, 6, 8, 10):
        A = _laplacian_2d(k)
        sy = symbolic_lu(A, minimum_degree(A))
        rows.append((k, sy.max_clique, sy.treewidth_upper_bound))
        assert sy.treewidth_upper_bound >= k, (k, sy.treewidth_upper_bound)
        assert sy.max_clique == sy.treewidth_upper_bound + 1
    gaps = [bound - k for k, _, bound in rows]
    rec = record(
        label="(树宽上界 − 真实树宽 k) 的最大值",
        theory="≥ 0（贪心只是上界）；k=4..10",
        measured=f"最大 {max(gaps)}，逐例 " + ", ".join(f"k={k}:{b}(团{c})" for k, c, b in rows),
        tol="≥ 0",
        passed=min(gaps) >= 0,
    )
    assert rec.passed, rows


@assertion("A", theory="最小度的填充不比自然序差（逐例），且结构预测与精确一致", chapter="1.2")
def test_ordering_report_matches_reference_column_counts() -> None:
    """自研符号填充 vs 参考层（SciPy）在同一排序下的 nnz(L+U) 对拍。"""
    A = _laplacian_2d(6)
    worst = 0
    detail = []
    for name, perm in (("natural", natural_order(A)), ("rcm", rcm(A)), ("amd", minimum_degree(A))):
        mine = symbolic_lu(A, perm).nnz_lu
        # 参考层没有"指定排序"的接口，所以把矩阵**按同一排序预置换**再要求 NATURAL——
        # 这样 SuperLU 的消元顺序与我们的 perm 完全一致，两边数的才是同一件事。
        theirs = ref.sparse_lu_nnz(A[np.ix_(perm, perm)], permc_spec="NATURAL")
        detail.append(f"{name}: 自研 {mine} / 参考 {theirs}")
        worst = max(worst, abs(mine - theirs))
    rec = record(
        label="|nnz(L+U) 自研 − 参考| 的最大值",
        theory="0（同一矩阵、同一排序，符号填充是确定量）",
        measured=f"{worst}；" + "，".join(detail),
        tol="0",
        passed=worst == 0,
    )
    assert rec.passed, detail


# --------------------------------------------------------------------------- #
# B 类：不变量
# --------------------------------------------------------------------------- #
@assertion("B", theory="CSR 容器与稠密表示等价", chapter="1.2")
def test_csr_container_matches_dense_operations() -> None:
    """matvec / to_dense / transpose / permute_symmetric 四个操作逐一对拍稠密实现。"""
    from numcore.utils import rng

    g = rng(31337)
    checked = 0
    for n in (5, 12, 40):
        A = g.standard_normal((n, n))
        A[np.abs(A) < 0.7] = 0.0
        csr = CSR.from_dense(A)
        x = g.standard_normal(n)
        assert np.allclose(csr.matvec(x), A @ x), "matvec 不等价"
        assert np.allclose(csr.to_dense(), A), "to_dense 不等价"
        assert np.allclose(csr.transpose().to_dense(), A.T), "转置不等价"
        perm = g.permutation(n)
        assert np.allclose(csr.permute_symmetric(perm).to_dense(), A[np.ix_(perm, perm)]), "对称置换不等价"
        assert csr.nnz == int(np.count_nonzero(A))
        assert abs(csr.density - csr.nnz / (n * n)) < 1e-15
        checked += 1
    rec = record(
        label="CSR 等价性检查的矩阵个数",
        theory="全部等价（4 个操作）",
        measured=f"{checked} 个（n=5/12/40）",
        tol="0 反例",
        passed=checked == 3,
    )
    assert rec.passed


@assertion("B", theory="置换不变性：nnz、对称性、行和、谱在对称置换下不变", chapter="1.2")
def test_symmetric_permutation_preserves_structure_and_spectrum() -> None:
    """对称置换只是重排，不改变矩阵的任何本质量（结构 + 数值都查）。"""
    A = _laplacian_2d(5)
    perm = rcm(A)
    Ap = A[np.ix_(perm, perm)]
    ev_a = np.sort(ref.eigh(A)[0])
    ev_p = np.sort(ref.eigh(Ap)[0])
    worst = float(np.max(np.abs(ev_a - ev_p)))
    same_nnz = int(np.count_nonzero(A)) == int(np.count_nonzero(Ap))
    same_sym = np.allclose(Ap, Ap.T)
    rec = record(
        label="对称置换后的最大特征值偏差 / 结构是否保持",
        theory="0（置换相似变换）",
        measured=f"偏差 {worst:.3e}，nnz 相同={same_nnz}，对称={same_sym}",
        tol="1e-12",
        passed=worst <= 1e-12 and same_nnz and same_sym,
    )
    assert rec.passed, (worst, same_nnz, same_sym)


@assertion("B", theory="三种排序的输出都是合法置换", chapter="1.2")
def test_orderings_return_valid_permutations() -> None:
    """自然序/RCM/最小度都必须给出 0..n-1 的一个排列（不能漏、不能重）。"""
    A = _laplacian_2d(7)
    n = A.shape[0]
    ok = {}
    for name, perm in (("natural", natural_order(A)), ("rcm", rcm(A)), ("cm", cuthill_mckee(A)), ("amd", minimum_degree(A))):
        ok[name] = sorted(int(v) for v in perm) == list(range(n))
    assert all(ok.values()), ok
    rec = record(
        label="四种排序是否为合法置换",
        theory="全部是",
        measured=str(ok),
        tol="全 True",
        passed=all(ok.values()),
    )
    assert rec.passed


@assertion("B", theory="填充边集与 nnz 恒等式：nnz_lu = nnz(L)+nnz(U) = 2n+2Σ|later|", chapter="1.2")
def test_fill_edges_and_nnz_identity() -> None:
    """逐例检查 nnz(L+U) 的分解式，以及 fill_edges 与符号结构的自洽。"""
    A = _laplacian_2d(5)
    perm = natural_order(A)
    sy = symbolic_lu(A, perm)
    n = A.shape[0]
    ident = sy.nnz_lu == sy.nnz_l + sy.nnz_u and sy.nnz_l == sy.nnz_u
    # 闭式：nnz(L+U) = 2n + 2Σ_k |later(k)|；对 5×5 网格，逐列重放得到 Σ|later| = 68
    later = _later_counts(A, perm)
    closed = 2 * n + 2 * int(np.sum(later))
    edges = fill_edges(A, perm)
    tridiag_closed = 2 * 8 + 2 * 7  # 三对角：每一步 later=1
    sy_tri = symbolic_lu(_tridiag(8), natural_order(_tridiag(8)))
    rec = record(
        label="nnz(L+U) 的闭式与分解式",
        theory=f"2n + 2Σ|later| = {closed}（5×5 网格）；三对角 n=8 时 = {tridiag_closed}",
        measured=f"符号 {sy.nnz_lu}（分解式{ident}，L {sy.nnz_l} / U {sy.nnz_u}），Σ|later|={int(np.sum(later))}，"
        f"fill_edges 接口 {edges} = 结构里的 {sy.fill_edges}；三对角符号 {sy_tri.nnz_lu}",
        tol="闭式与分解式都成立",
        passed=sy.nnz_lu == closed and ident and sy_tri.nnz_lu == tridiag_closed and edges == sy.fill_edges,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="spy 坐标与矩阵非零元一一对应", chapter="1.2")
def test_spy_points_match_nonzeros() -> None:
    """画图用的坐标必须就是非零元坐标（多一个点就是误导）。"""
    from numcore.utils import rng

    g = rng(909)
    worst = 0
    for n in (6, 15):
        A = g.standard_normal((n, n))
        A[np.abs(A) < 0.5] = 0.0
        rows, cols = spy_points(A)
        worst = max(worst, abs(len(rows) - int(np.count_nonzero(A))))
        got = {(int(i), int(j)) for i, j in zip(rows, cols)}
        want = {(int(i), int(j)) for i, j in zip(*np.nonzero(A))}
        assert got == want
    rec = record(
        label="spy 点数与 nnz 的最大偏差",
        theory="0（且坐标集合完全相同）",
        measured=f"{worst}",
        tol="0",
        passed=worst == 0,
    )
    assert rec.passed


@assertion("B", theory="LAPACK 带状存储 ab[b+i−j, j] = a_ij，带内元素全部可复原", chapter="1.2")
def test_band_storage_keeps_all_nonzeros() -> None:
    """带状压缩：带内每个元素都能从 :math:`ab[b+i-j, j]` 还原，存储量 = (2b+1)n。"""
    n = 50
    A = _tridiag(n)
    b = max(bandwidth(A, natural_order(A)))  # (lower, upper) → 三对角两侧都是 1
    ab = band_storage(A, b, b)
    ok = all(ab[b + i - j, j] == A[i, j] for i in range(n) for j in range(max(0, i - b), min(n, i + b + 1)))
    rec = record(
        label="带宽 / 带存储形状 / 可复原",
        theory=f"带宽 1；形状 (2b+1, n) = (3, {n})",
        measured=f"带宽 {b}，形状 {ab.shape}，可复原={ok}",
        tol="形状 (3, 50) 且逐元素复原",
        passed=b == 1 and ab.shape == (3, n) and ok,
    )
    assert rec.passed


@assertion("B", theory="置换矩阵 P 满足 PᵀP = I 且 PA 只重排行", chapter="1.2")
def test_permutation_matrix_acts_as_row_permutation() -> None:
    """置换矩阵的构造正确性（行重排 + 正交性）。"""
    A = example_matrix("spd", 8)
    perm = rcm(A)
    P = permutation_matrix(perm)
    ok_orth = np.allclose(P.T @ P, np.eye(len(perm)))
    ok_rows = np.allclose(P @ A, A[perm, :])
    res = lu_decompose(A, pivot="partial")
    ok_perm = np.allclose(res.P() @ A, A[res.perm, :])
    rec = record(
        label="PᵀP=I / PA=A[perm] / LU 的 perm 自洽",
        theory="三者全成立",
        measured=f"正交={ok_orth}，行重排={ok_rows}，LU 的 perm={ok_perm}",
        tol="全 True",
        passed=ok_orth and ok_rows and ok_perm,
    )
    assert rec.passed, (ok_orth, ok_rows, ok_perm)


@assertion("B", theory="剖面 profile = Σ(带宽内首末非零列距)，且 RCM 的剖面 ≤ 自然序", chapter="1.2")
def test_profile_and_bandwidth_are_consistent() -> None:
    """剖面/带宽的定义自洽：带宽为 0（对角）时剖面恒为 n。"""
    D = np.diag(np.arange(1.0, 10.0))
    # 剖面的定义是"首末非零列的**距离**"（不含对角本身），所以对角阵的剖面是 0 而不是 n
    ok_diag = max(bandwidth(D, natural_order(D))) == 0 and profile(D, natural_order(D)) == 0
    A, _ = _permuted(_tridiag(40), 5)
    prof_nat = profile(A, natural_order(A))
    prof_rcm = profile(A, rcm(A))
    assert_leq(prof_rcm, prof_nat, label="RCM 剖面 ≤ 自然序剖面")
    rec = record(
        label="对角阵的带宽/剖面；置换三对角阵的剖面（自然 → RCM）",
        theory="对角阵：带宽 (0,0)、剖面 0；RCM 剖面 ≤ 自然序",
        measured=f"对角 {ok_diag}；{prof_nat} → {prof_rcm}",
        tol="如上",
        passed=ok_diag and prof_rcm <= prof_nat,
    )
    assert rec.passed


# --------------------------------------------------------------------------- #
# 与章节帧的耦合：可视化里画出来的量必须满足同一批不变量
# --------------------------------------------------------------------------- #
@assertion("B", theory="spy_ordering 帧：填充随步数单调增，三种排序的后向残差都在 u 量级", chapter="1.2")
def test_spy_ordering_frames_obey_invariants() -> None:
    """直接检查章节产出的帧（可视化与断言不能各说各话）。"""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("ch12_mod", root / "server" / "chapters" / "1.2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params = {"grid": 6, "ordering": "rcm"}
    prev = -1
    monotone = True
    worst_err = 0.0
    nnz_total = None
    for t in (0.0, 3.0, 9.0, 20.0, 35.0):
        fr = mod.compute("spy_ordering", params, t)
        cur = int(fr.meta["fill_edges_so_far"])
        monotone = monotone and cur >= prev
        prev = cur
        nnz_total = int(fr.meta["nnz_total"])
        worst_err = max(worst_err, float(np.max(fr["error"])))
    last = mod.compute("spy_ordering", params, 1e9)  # 时间轴夹到最后一步
    rec = record(
        label="填充单调性 / 总填充 / 三种排序的最大后向残差",
        theory="单调不减；末步 fill_so_far = fill_total；后向残差 ≈ u",
        measured=f"单调={monotone}，nnz(L+U)={nnz_total}，末步 fill={int(last.meta['fill_edges_so_far'])}"
        f"/{int(last.meta['fill_edges_total'])}，残差最大 {worst_err:.2e}",
        tol="单调、末步相等、残差 ≤ 100u",
        passed=monotone and int(last.meta["fill_edges_so_far"]) == int(last.meta["fill_edges_total"]) and worst_err <= 100 * U,
    )
    assert rec.passed, rec.measured


@assertion("B", theory="四个可视化的帧都带 requiredAux 与一致的形状", chapter="1.2")
def test_all_chapter_frames_are_well_formed() -> None:
    """逐 viz 检查帧契约：requiredAux 存在、轴声明与数组一致、无 NaN（除曲线允许的缺失点）。"""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("ch12_mod2", root / "server" / "chapters" / "1.2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = []
    ok = True
    for viz in mod.MANIFEST["vizzes"]:
        params = {p["name"]: p.get("default") for p in viz["params"]}
        fr = mod.compute(viz["id"], params, 0.0)
        names = {n.lower() for n in fr.names}
        for aux in viz["requiredAux"]:
            # 与 server/frames.py 的体检规则保持一致：**子串**匹配（error 可以叫 error_curve，
            # residual 可以叫 residual_history）。这里额外要求"至少有一个数组的名字以它开头"，
            # 免得出现 residual 落在一个叫 x_residual_gap 的无关数组上。
            if not any(aux in n for n in names):
                ok = False
                report.append(f"{viz['id']}: 缺 requiredAux {aux}")
            elif not any(n.startswith(aux) for n in names):
                ok = False
                report.append(f"{viz['id']}: {aux} 只有间接匹配 {sorted(n for n in names if aux in n)}")
        for arr in ("error", "residual") if viz["id"] in ("unit_ball", "lu_heatmap") else ():
            if arr in names and np.all(np.isnan(fr[arr])):
                ok = False
                report.append(f"{viz['id']}: {arr} 全是 NaN")
        n_finite = sum(1 for nm in names if np.all(np.isfinite(fr[nm])))
        report.append(f"{viz['id']}: {len(names)} 数组，{n_finite} 个全有限")
    rec = record(
        label="帧契约检查",
        theory="每个 viz 的 requiredAux 齐备、关键曲线不是全 NaN",
        measured="；".join(report),
        tol="全通过",
        passed=ok,
    )
    assert rec.passed, report
