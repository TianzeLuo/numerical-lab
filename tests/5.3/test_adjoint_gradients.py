"""5.3 章 A 类断言（一）：离散伴随 = 目标函数的精确梯度，以及它的两处"边界"。

这一文件要钉死四件事，每一件都带**反面**：

1. **离散伴随与中心差分一致**（相对误差 < 1e-9），而把差分步长放大到 1e-2 时
   同一个比较立刻失效（>= 1e-2）——容差不是随手写的；
2. **V 形曲线两段都要有数**：截断段斜率与舍入段斜率分别拟合，实测最优
   ``eps*`` 落在扫描区间**内部**，且与 ``u^(1/3)``（中心）``u^(1/2)``（单侧）同量级；
   "eps 越小越准"由"再小三个数量级误差涨 100 倍以上"直接否掉；
3. **连续伴随与离散伴随必须分开报**：常数雅可比的线性模型上两者重合到 1e-12
   （这是本章的"零差异边界"），非线性 Duffing 上按时间格式的阶分开（rk4 ~4、heun ~2、euler ~1）；
4. **伴随不免费**：全存轨迹与分块检查点的梯度逐位相同，但存储降一个量级、
   残差求值涨到 2 倍附近；而伴随的代价与参数个数 k **无关**（有限差分是 2k 次正演）。
"""

from __future__ import annotations

import numpy as np

from numcore.inverse.adjoint import (
    UNIT_ROUNDOFF,
    discrete_adjoint,
    discrete_adjoint_checkpointed,
    finite_difference_gradient,
    gradient_check_scan,
    loglog_slope,
    relative_error,
    rk_forward,
)
from numcore.inverse.sensitivity import (
    adjoint_and_continuous,
    cost_scaling_study,
    duffing_model,
    linear_forcing_model,
    scalar_decay_model,
)
from numcore.verify import assertion, fit_order, record


_DUFFING_P = np.array([1.0, 0.1, 0.05])
_DUFFING_T = 30.0
_DUFFING_N = 96


def _duffing_trace(n_steps: int = _DUFFING_N, p=None):
    model = duffing_model()
    pv = _DUFFING_P if p is None else np.asarray(p, dtype=float)
    return model, pv, rk_forward(model, pv, T=_DUFFING_T, n_steps=int(n_steps), method="rk4", store="full")


# --------------------------------------------------------------------------- #
# 1. 离散伴随 vs 中心差分（含"容差为什么是 1e-9"的反面）
# --------------------------------------------------------------------------- #
@assertion("A", theory="离散伴随是 J_h 的精确梯度：与中心差分的差只剩差分自身的截断/舍入误差", chapter="5.3")
def test_discrete_adjoint_matches_finite_difference_and_loses_it_at_big_eps() -> None:
    r"""两个工作点上的 |g_adj - g_fd|/max(1,|g|)：小步长必须 < 1e-9，大步长必须 >= 1e-2。

    "<= 1e-9" 这张容差表不是装饰：同一个比较在 ``eps=0.1`` 处掉到 1e-1 量级
    说明测到的确实是小步长下的**一致性**，而不是"两个数碰巧都很大"。

    解析对照用 :func:`scalar_decay_model`（:math:`y'=-p y`，梯度有闭式）：
    离散伴随与解析梯度只差浮点舍入。
    """
    small = 1e-6  # 实测最优量级（见下一条断言：u^(1/3)=4.8e-6 附近，对数网格上最优点在 1e-6）
    rows = []
    for label, p in (("duffing", _DUFFING_P), ("duffing-near-ridge", np.array([1.02, 0.05, 0.05]))):
        model, pv, trace = _duffing_trace(p=p)
        adj = discrete_adjoint(model, pv, trace)
        g_small, _ = finite_difference_gradient(model, pv, T=_DUFFING_T, n_steps=_DUFFING_N, eps=small)
        g_big, _ = finite_difference_gradient(model, pv, T=_DUFFING_T, n_steps=_DUFFING_N, eps=0.1)
        e_small = relative_error(adj.grad, g_small)
        e_big = relative_error(adj.grad, g_big)
        rows.append(f"{label}: eps=1e-6 -> {e_small:.2e}，eps=0.1 -> {e_big:.2e}")
        assert e_small < 1e-9, f"{label} 小步长相对误差 {e_small:.3e} 超过 1e-9"
        assert e_big > 1e-2, f"{label} 大步长相对误差只有 {e_big:.3e}，容差 1e-9 失去意义"

    # 解析对照（标量衰减模型，两个梯度都有闭式）：
    #   离散：J_h = R(-hp)^N -> dJ_h/dp = -T R^{N-1} R'；连续：dJ/dp = -T e^{-pT}
    # 离散伴随必须**逐位**等于离散闭式（这是"伴随求的是离散问题的梯度"的直接证据），
    # 与连续闭式的差则是 O(h^4) —— 两者都要报，只报一个都会误导。
    model = scalar_decay_model()
    p0 = np.array([0.7])
    gaps = []
    disc_err = 0.0
    for n_steps in (128, 256, 512):
        tr = rk_forward(model, p0, T=2.0, n_steps=n_steps, method="rk4", store="full")
        adj = discrete_adjoint(model, p0, tr)
        h = 2.0 / n_steps
        z = -h * float(p0[0])
        R = 1.0 + z + z * z / 2.0 + z**3 / 6.0 + z**4 / 24.0
        Rp = 1.0 + z + z * z / 2.0 + z**3 / 6.0
        g_disc = -2.0 * R ** (n_steps - 1) * Rp
        g_cont = -2.0 * float(np.exp(-float(p0[0]) * 2.0))
        disc_err = max(disc_err, abs(float(adj.grad[0]) - g_disc) / abs(g_disc))
        gaps.append(abs(float(adj.grad[0]) - g_cont) / abs(g_cont))
    assert disc_err < 1e-12, f"离散伴随 vs 离散闭式的相对差 {disc_err:.3e}"
    ratios = [gaps[i] / gaps[i + 1] for i in range(len(gaps) - 1)]
    assert all(abs(r - 16.0) < 0.5 for r in ratios), f"连续/离散间隙的步长比 {ratios} 不是 16（4 阶）"
    assert gaps[0] > 1e-12, f"N=128 处连续/离散间隙只有 {gaps[0]:.3e}，'两者不同'这句话没有证据"

    rec = record(
        label="离散伴随 vs 中心差分（小步长 / 大步长），以及标量模型的离散/连续闭式对照",
        theory=(
            "eps=1e-6 时 < 1e-9；eps=0.1 时 >= 1e-2（否则容差无意义）；"
            "与离散闭式 < 1e-12（伴随对离散问题精确）；与连续闭式的间隙按 16 倍缩小（4 阶）"
        ),
        measured=(
            "；".join(rows)
            + f"；标量模型离散闭式差 {disc_err:.2e}，连续/离散间隙 "
            + " -> ".join(f"{g:.2e}" for g in gaps)
            + f"（步长比 {', '.join(f'{r:.2f}' for r in ratios)}）"
        ),
        tol="1e-9 / 1e-2 / 1e-12 / 比值 16±0.5",
        passed=True,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. V 形曲线：两段斜率 + 实测最优 eps
# --------------------------------------------------------------------------- #
@assertion("A", theory="中心差分截断 ~ eps^2、舍入 ~ u/eps，最优 eps ~ u^(1/3)；单侧差分对应 u^(1/2)", chapter="5.3")
def test_gradient_check_v_curve_has_both_branches_and_measured_optimum() -> None:
    r"""V 形曲线必须两头都有数：下降段斜率、上升段斜率、谷底位置、以及"更小更差"的反面。

    扫描区间 ``[1e-16, 1e-1]``（对数均匀 25 点），实测最优 ``eps*`` 必须**落在区间内部**；
    与 :math:`u^{1/3}`（中心差分）/ :math:`u^{1/2}`（单侧差分）的比值只承诺"同量级"
    （对数网格间隔 4.2 倍，谷底只能定位到最近的扫描点）。
    """
    model, pv, _ = _duffing_trace()
    eps_values = np.logspace(-16.0, -1.0, 25)
    ck = gradient_check_scan(
        model, pv, T=_DUFFING_T, n_steps=_DUFFING_N, method="rk4",
        eps_values=eps_values, schemes=("central", "forward"),
    )
    n = int(eps_values.size)
    best_c = ck.best_index.get("central", -1)
    best_f = ck.best_index.get("forward", -1)
    assert 0 < best_c < n - 1, f"中心差分最优 eps 落在端点上：index={best_c}"
    assert 0 < best_f < n - 1, f"单侧差分最优 eps 落在端点上：index={best_f}"

    s_down_c = ck.slope_down["central"]
    s_up_c = ck.slope_up["central"]
    s_down_f = ck.slope_down["forward"]
    s_up_f = ck.slope_up["forward"]
    # 语义：slope_down = 最优点**左侧**（舍入主导，误差随 eps 下降）-> -1；
    #       slope_up   = 最优点**右侧**（截断主导，误差随 eps 上升）-> +2（中心）/ +1（单侧）
    assert abs(s_down_c + 1.0) < 0.15, f"中心差分舍入段斜率 {s_down_c} 偏离 -1"
    assert abs(s_up_c - 2.0) < 0.15, f"中心差分截断段斜率 {s_up_c} 偏离 2"
    assert abs(s_down_f + 1.0) < 0.15, f"单侧差分舍入段斜率 {s_down_f} 偏离 -1"
    assert abs(s_up_f - 1.0) < 0.15, f"单侧差分截断段斜率 {s_up_f} 偏离 1"

    ratio_c = ck.optimum_ratio("central", "1/3")
    ratio_f = ck.optimum_ratio("forward", "1/2")
    assert 0.05 < ratio_c < 5.0, f"中心差分 eps*/u^(1/3) = {ratio_c}"
    assert 0.1 < ratio_f < 2.0, f"单侧差分 eps*/u^(1/2) = {ratio_f}"

    # 反面：往更小的 eps 走三个数量级，误差必须显著变差（"eps 越小越准"不成立）
    err_c = ck.err["central"]
    tiny = float(err_c[0])
    best_err = ck.best_err("central")
    assert tiny > 100.0 * best_err, f"eps=1e-16 的误差 {tiny:.3e} 不比最优 {best_err:.3e} 差 100 倍"

    rec = record(
        label="V 形曲线两段斜率、实测最优 eps 与理论比值",
        theory="最优点左侧（舍入）斜率 -1；右侧（截断）斜率 2（中心）/1（单侧）；eps* 在扫描内部且与 u^(1/3)、u^(1/2) 同量级",
        measured=(
            f"中心 eps*={ck.best_eps('central'):.3e}（u^(1/3)={UNIT_ROUNDOFF ** (1 / 3):.3e}，比值 {ratio_c:.3f}），"
            f"左段（舍入）斜率 {s_down_c:.3f}、右段（截断）斜率 {s_up_c:.3f}，谷底误差 {best_err:.2e}；"
            f"单侧 eps*={ck.best_eps('forward'):.3e}（u^(1/2)={UNIT_ROUNDOFF ** 0.5:.3e}，比值 {ratio_f:.3f}），"
            f"左段 {s_down_f:.3f}、右段 {s_up_f:.3f}；eps=1e-16 处误差 {tiny:.2e}"
        ),
        tol="斜率 ±0.15；比值区间 0.05-5 / 0.1-2",
        passed=True,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 3. 连续伴随 vs 离散伴随：零差异边界 + 按阶分开
# --------------------------------------------------------------------------- #
@assertion("A", theory="常数雅可比模型上连续/离散伴随重合；非线性问题上两者差 O(h^p)，p 是时间格式的阶", chapter="5.3")
def test_continuous_vs_discrete_adjoint_split_by_case() -> None:
    r"""先报"两者重合到机器零"的边界，再报"非线性下按阶分开"——两个方向都要有数。

    线性模型 ``f = A y + B p`` 的雅可比是常数，连续伴随与离散伴随解的是同一个方程，
    实测相对差 <= 1e-12（此处只受舍入限制）；Duffing 上按格式阶分开，
    阶数用**渐近窗口**拟合（把尚未进入渐近区的粗网格点丢掉，窗口写进 detail）。
    """
    linear_rows = []
    for k in (2, 8, 32):
        model = linear_forcing_model(k)
        pv = np.linspace(-0.4, 0.4, k)
        d, c = adjoint_and_continuous(model, pv, T=4.0, n_steps=96, method="rk4")
        gap = relative_error(d.grad, c.grad)
        linear_rows.append(f"k={k}: {gap:.2e}")
        assert gap < 1e-12, f"线性模型 k={k} 上连续/离散伴随差 {gap:.3e} > 1e-12"

    nl_rows = []
    nl_orders = {}
    for method, n_list, skip, expect in (
        ("rk4", (32, 64, 128, 256, 512), 1, 4.0),
        ("heun", (64, 128, 256, 512, 1024), 1, 2.0),
        ("euler", (1024, 2048, 4096, 8192, 16384), 1, 1.0),
    ):
        model = duffing_model()
        hs, errs = [], []
        for n_steps in n_list:
            d, c = adjoint_and_continuous(model, _DUFFING_P, T=_DUFFING_T, n_steps=int(n_steps), method=method)
            hs.append(_DUFFING_T / float(n_steps))
            errs.append(relative_error(d.grad, c.grad))
        fit = fit_order(np.asarray(hs), np.asarray(errs), skip=skip)
        nl_orders[method] = fit.slope
        # 拟合窗口内还残留 O(h^(p+1)) 项，局部阶会**系统性地**略高于渐近阶（实测 rk4 4.16、heun 2.01、euler 1.00）
        assert abs(fit.slope - expect) < 0.25, f"{method} 上连续/离散伴随的实测阶 {fit.slope:.3f} 偏离 {expect}"
        nl_rows.append(
            f"{method}: p={fit.slope:.3f}（N={n_list[skip]}-{n_list[-1]}，残差 {errs[0]:.2e} -> {errs[-1]:.2e}）"
        )

    rec = record(
        label="连续伴随 vs 离散伴随：线性重合边界与非线性阶数",
        theory="常数雅可比模型上 < 1e-12；非线性按格式阶（rk4=4、heun=2、euler=1）",
        measured="线性 " + "，".join(linear_rows) + "；非线性 " + "；".join(nl_rows),
        tol="1e-12；阶数 ±0.25（渐近窗口已去掉最粗一点，局部阶系统性偏高）",
        passed=True,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 4. 分块检查点与全存轨迹：梯度相同、存储/重算互换
# --------------------------------------------------------------------------- #
@assertion("A", theory="分块检查点重算给出与全存轨迹逐位相同的梯度，代价是重算而非存储", chapter="5.3")
def test_checkpointed_adjoint_is_identical_and_trades_storage_for_recompute() -> None:
    r"""检查点不是"近似"：梯度必须逐位相同（相对差 0），而存储/求值次数按块大小互换。

    这条断言的意义在于把"伴随法不免费"写成两个可核对的数：``n_floats_stored``
    降多少、``n_rhs`` 涨多少。两者必须同时报，只报其中一个就是挑好看的写。
    """
    model = duffing_model()
    rows = []
    ok = True
    for n_steps in (256, 1024):
        tr = rk_forward(model, _DUFFING_P, T=_DUFFING_T, n_steps=n_steps, method="rk4", store="full")
        full = discrete_adjoint(model, _DUFFING_P, tr)
        ck = discrete_adjoint_checkpointed(model, _DUFFING_P, T=_DUFFING_T, n_steps=n_steps, method="rk4")
        gap = relative_error(full.grad, ck.grad)
        # 口径：全存一侧 = 轨迹（states + 级值 + 节点导数）+ mu 数组；
        #       检查点一侧 = 检查点 + 当前重算块 + mu 数组（它自己的账）。
        full_total = int(tr.work.n_floats_stored) + int(full.work.n_floats_stored)
        ck_total = int(ck.work.n_floats_stored)
        store_ratio = float(ck_total) / float(full_total)
        recompute = float(ck.work.n_rhs) / float(tr.work.n_rhs)
        rows.append(
            f"N={n_steps}: 梯度相对差 {gap:.1e}，存储 {full_total}->{ck_total}（{store_ratio:.3f} 倍），"
            f"重算残差求值 {tr.work.n_rhs}->{ck.work.n_rhs}（{recompute:.3f} 倍于正演）"
        )
        ok = ok and gap == 0.0 and store_ratio < 0.5 and 1.5 < recompute < 2.5
    assert ok, "；".join(rows)
    rec = record(
        label="全存轨迹 vs 分块检查点的梯度一致性与存储/重算账",
        theory="梯度逐位相同（相对差 0）；存储降到 1/sqrt(N) 量级，重算涨到正演的 2 倍附近",
        measured="；".join(rows),
        tol="差值 = 0；存储 < 0.5 倍；重算 1.5-2.5 倍",
        passed=True,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 5. 代价与参数个数无关（C 类）
# --------------------------------------------------------------------------- #
@assertion("C", theory="伴随一次求解给出全部 k 个偏导：代价与 k 无关；有限差分是 2k 次正演", chapter="5.3")
def test_adjoint_cost_is_independent_of_parameter_count() -> None:
    r"""k=2/8/32 的正演次数比必须是 2k（4/16/64），而伴随自身的计数一个都不许变。

    第三个点（k=32）是这条断言的**边界**：正演次数比 64 倍，但伴随的
    ``n_rhs``/``n_jac``/``n_forward_solves`` 必须与 k=2 时逐位相同——
    "与前向同阶"这句话的正确读法是"与 k 无关"，不是"与正演一样便宜"。
    """
    rows = cost_scaling_study(param_counts=(2, 8, 32), dim=4, T=4.0, n_steps=96, method="rk4")
    assert len(rows) == 3
    solve_ratios = []
    for row in rows:
        assert row.grad_rel_err_checkpoint == 0.0, f"检查点梯度与全存不一致：{row.grad_rel_err_checkpoint}"
        solve_ratios.append(float(row.fd_solves) / float(row.adjoint_solves))
    assert solve_ratios == [4.0, 16.0, 64.0], f"正演次数比 {solve_ratios} 不是 2k"
    base = rows[0].adjoint_rhs
    assert all(r.adjoint_rhs == base for r in rows), "伴随残差求值次数随 k 变了"
    assert all(r.adjoint_jac == rows[0].adjoint_jac for r in rows), "伴随 Jacobian 求值次数随 k 变了"
    assert all(r.adjoint_solves == 1 for r in rows), "伴随正演次数不是 1"
    # 但"计算量与前向同阶"不等于"免存储"：存储量随步数线性长
    assert rows[-1].adjoint_floats > 0 and rows[-1].checkpoint_floats < rows[-1].adjoint_floats
    rec = record(
        label="伴随 vs 有限差分的正演次数比（k=2/8/32）与伴随次数的 k 无关性",
        theory="正演次数比 = 2k（4/16/64）；伴随的求值次数与存储量与 k 无关（只与 N、dim 有关）",
        measured="；".join(
            f"k={r.n_params}: 伴随 rhs={r.adjoint_rhs} jac={r.adjoint_jac} 正演={r.adjoint_solves}，"
            f"有限差分 rhs={r.fd_rhs} 正演={r.fd_solves}，比值 {float(r.fd_solves) / float(r.adjoint_solves):.0f}"
            for r in rows
        ),
        tol="比值严格 = 2k；伴随计数逐位相同",
        passed=True,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 6. 斜率拟合的拒答行为：点数不够 / 非正数时必须给 None
# --------------------------------------------------------------------------- #
@assertion("A", theory="拟合工具在证据不足时必须返回 None，不能用单点或含零的序列冒充幂律", chapter="5.3")
def test_loglog_slope_refuses_to_report_without_a_power_law() -> None:
    r"""``loglog_slope`` 的**拒答**行为：真幂律给对数斜率，退化输入给 ``None``。

    V 形曲线的最优 ``eps`` 一旦落在扫描端点，某一侧的可用点就不足 2 个，
    这时 ``slope_up``/``slope_down`` 必须是 ``None`` 而不是"最后两个点连线的斜率"。
    这条断言就是那个机制的检查：非正/非有限值被剔除后若不足两点，直接 ``None``。
    """
    x = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    perfect = 3.0 * x**2
    slope = loglog_slope(x, perfect)
    assert slope is not None and abs(slope - 2.0) < 1e-12, f"幂律斜率 {slope}"
    assert loglog_slope(x[:2], 3.0 * x[:2] ** 2) is not None  # 恰好两点：还能算
    assert loglog_slope(x[:1], 3.0 * x[:1] ** 2) is None, "单点也给出斜率"
    # 非正值被**剔除**（0 在对数轴上不携带信息），剔除后仍 >= 2 点就继续拟合
    dropped = loglog_slope(x, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert dropped is not None and abs(dropped - 0.1982) < 1e-3, f"剔除 0 之后应剩 4 点，实测 {dropped}"
    assert loglog_slope(x, np.full(5, np.nan)) is None, "全 NaN 仍给出斜率"
    only_one_ok = loglog_slope(x, np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert only_one_ok is None, "剔除后只剩一点仍给出斜率"
    masked = loglog_slope(x, perfect, np.array([True, False, False, False, False]))
    assert masked is None, "掩码后只剩一点仍给出斜率"
    rec = record(
        label="loglog_slope：幂律斜率、剔除非正值的行为与三种退化输入的拒答",
        theory="剔除非正/非有限点后有效点 >= 2 才给斜率（0 在 log 轴上无信息），否则 None",
        measured=f"幂律 -> {slope:.12f}；含一个 0 -> {dropped:.4f}（剩 4 点）；全 NaN / 只剩一点 / 掩码后一点 -> None",
        tol="斜率误差 < 1e-12；退化输入必须 None",
        passed=True,
    )
    assert rec.passed, rec.measured
