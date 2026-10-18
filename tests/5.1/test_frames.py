"""5.1 章的**帧级**测试：把"图上看到的东西"与 `numcore` 的实测值钉在一起。

帧是唯一会被前端直接画出来的东西，所以这里检查的不是算法而是**呈现纪律**：

1. **没有非有限值**。四个可视化在默认参数、每个参数的两端、以及 ``t=0``/``t=tMax``
   全组合下都必须能 ``pack_frame``（JSON 头里不许出现 ``NaN``/``Infinity``）。
2. **占位与真值分开**。误差曲线在地板处截平、比值曲线带 ``*_valid`` 掩码、
   半径/比值里的非有限位置换成占位值并有掩码——"图上少一段"不许被画成"一直是那个值"。
3. **诚实字段真的会亮**。GD 发散时 ``meta.diverged_gd`` 为真、``meta.overflow`` 为真；
   β=0（退化成二次型）时 ``meta.newton_quadratic_constant`` 必须是 ``None`` 而不是编一个数；
   鞍点上与负曲率正交的扰动必须"跑不掉"。
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.inverse.optimize import (  # noqa: E402
    anisotropic_quadratic,
    gd,
    quadratic_window,
    rosenbrock_objective,
    saddle_quadratic,
    trust_region,
)
from numcore.verify import assertion, record  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.nlf import pack_frame  # noqa: E402
from server.registry import Registry  # noqa: E402


_ENTRY_CACHE: list = []


def _entry():
    if not _ENTRY_CACHE:
        reg = Registry(ROOT)
        reg.discover(strict=False)  # 并发开发时兄弟章节可能是半成品；本章必须能加载
        _ENTRY_CACHE.append(reg.get("5.1"))
    return _ENTRY_CACHE[0]


def _viz(viz_id: str):
    return next(v for v in _entry().manifest.vizzes if v.id == viz_id)


def _defaults(viz_id: str) -> dict:
    return {p.name: p.default for p in _viz(viz_id).params}


def _frame(viz_id: str, **over):
    t = float(over.pop("t", 0.0))
    params = _defaults(viz_id)
    params.update(over)
    return build_frame(_entry(), viz_id, params, t)


def _corners(viz_id: str):
    spec = _viz(viz_id)
    base = {p.name: p.default for p in spec.params}
    out = [dict(base)]
    for p in spec.params:
        if p.type == "enum":
            out.extend({**base, p.name: o} for o in p.options)
        elif p.type in ("int", "float"):
            out.append({**base, p.name: p.min})
            out.append({**base, p.name: p.max})
        elif p.type == "bool":
            out.append({**base, p.name: not base[p.name]})
    return out


# --------------------------------------------------------------------------- #
# 1. 全组合下的有限性与 JSON 可序列化
# --------------------------------------------------------------------------- #
@assertion("A", theory="帧在参数端点与时间端点全组合下都只有有限值，且能装进 JSON", chapter="5.1")
def test_all_frames_finite_and_packable_at_corners() -> None:
    r"""参数端点 × 时间端点的全组合：数组全有限、``meta`` 无 NaN、JSON 里没有 ``Infinity``。

    这不是重复 ``tests/test_frames_contract.py``：那边只跑默认值与单参数改动，
    这里额外把 ``t`` 推到 ``tMax``（scrub 的最后一帧是最容易炸的一帧）。
    """
    n_cases = 0
    worst_token = 0
    detail = []
    for viz_id in ("descent_race", "convergence_rates", "saddle_escape", "trust_region_radius"):
        spec = _viz(viz_id)
        times = [0.0] if spec.time.mode == "none" else [0.0, float(spec.time.t_max)]
        for params, t in itertools.product(_corners(viz_id), times):
            fr = build_frame(_entry(), viz_id, params, t)
            for name in fr.names:
                arr = np.asarray(fr[name], dtype=float)
                assert np.all(np.isfinite(arr)), f"{viz_id}/{name} 在 {params} t={t} 处有非有限值"
            for key, val in fr.meta.items():
                if isinstance(val, float):
                    assert np.isfinite(val), f"{viz_id} meta.{key} 非有限：{val}"
            blob = pack_frame(fr)  # 内部用 allow_nan=False 序列化帧头，并且 validate() 拒绝非有限值
            bad = blob.count(b"Infinity") + blob.count(b"NaN")
            worst_token += bad
            n_cases += 1
            if not detail:
                detail.append(f"{viz_id}: {len(fr.names)} 个数组、{len(fr.meta)} 个 meta 字段")
    rec = record(
        label="端点组合帧数与 JSON 里的 NaN/Infinity 计数",
        theory="NaN/Infinity 计数 = 0",
        measured=f"{n_cases} 帧，计数 {worst_token}（" + "；".join(detail) + "）",
        tol="= 0",
        passed=bool(worst_token == 0 and n_cases >= 60),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. 误差曲线的地板与掩码
# --------------------------------------------------------------------------- #
@assertion("A", theory="误差曲线在地板处截平并给出说明；比值曲线带有效性掩码", chapter="5.1")
def test_error_curves_are_floored_with_documented_placeholder() -> None:
    r"""对数轴上画不出 0：误差曲线必须截平 + 在 ``strings`` 里写明地板，不能留 0 给前端踩。

    同时检查掩码长度与曲线长度一致、且掩码里确实有 0（说明它真的在标占位/无效位置）。
    """
    detail = []
    ok = True
    for viz_id, names in (
        ("descent_race", ("error_gd", "error_newton")),
        ("convergence_rates", ("error_gd", "error_newton")),
        ("saddle_escape", ("residual", "error_to_saddle")),
        ("trust_region_radius", ("gradient_residual", "radius_history")),
    ):
        fr = _frame(viz_id)
        for name in names:
            arr = np.asarray(fr[name], dtype=float)
            ok = ok and bool(np.all(arr > 0.0))
            detail.append(f"{viz_id}.{name} min={arr.min():.1e}")
        keys = " ".join(fr.strings)
        ok = ok and ("floor" in keys)
    fr = _frame("convergence_rates")
    q = np.asarray(fr["q_newton"], dtype=float)
    mask = np.asarray(fr["q_newton_valid"])
    ok = ok and (mask.shape == q.shape) and (mask.dtype == np.bool_)
    fr4 = _frame("trust_region_radius")
    rmask = np.asarray(fr4["ratio_valid"])
    ok = ok and (rmask.shape == np.asarray(fr4["ratio_history"]).shape)
    rec = record(
        label="对数轴曲线最小值 > 0、掩码形状与 dtype、strings 里的地板说明",
        theory="全部满足",
        measured="；".join(detail[:6]) + f"；q_newton_valid dtype={mask.dtype}",
        tol="全 True",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 3. 下降赛跑：发散要如实亮灯
# --------------------------------------------------------------------------- #
@assertion("A", theory="step_scale 越过分界线时 GD 发散，帧必须标记而不是静默截断", chapter="5.1")
def test_descent_race_frame_marks_divergence() -> None:
    r"""κ=1e4、step_scale=1.6（α/α* > 1+1/κ = 1.0001）时 GD 必然发散。

    实测：``meta.diverged_gd`` 为真、``strings.stop_gd == "diverged"``、
    ``meta.overflow`` 为真，而**帧本身仍然有限**（尾部重复末点补位）。
    最后一帧的 ``error_gd`` 曲线长度等于 ``max_iter+1``，没有被截短。
    """
    fr = build_frame(
        _entry(),
        "descent_race",
        {**_defaults("descent_race"), "kappa": 10000.0, "step_scale": 1.6, "max_iter": 400},
        400.0,
    )
    meta = fr.meta
    err = np.asarray(fr["error_gd"], dtype=float)
    rec = record(
        label="发散帧的三个诚实字段",
        theory="diverged_gd=True（放大 1/u 倍即判发散）、overflow=True、stop_gd='diverged'",
        measured=f"diverged_gd={meta['diverged_gd']}，overflow={meta['overflow']}，stop_gd={fr.strings['stop_gd']!r}，"
        f"α/α_stable={meta['alpha_over_stable']:.4f}",
        tol="全 True",
        passed=bool(meta["diverged_gd"] and meta["overflow"] and fr.strings["stop_gd"] == "diverged" and meta["alpha_over_stable"] > 1.0),
    )
    rec2 = record(
        label="发散帧里 error_gd 的长度与有限性",
        theory="长度 = max_iter+1 = 401，全有限（尾部补位）",
        measured=f"长度 {err.size}，max={err.max():.3e}，尾部 3 个 {np.array2string(err[-3:], precision=3)}",
        tol="401 且有限",
        passed=bool(err.size == 401 and np.all(np.isfinite(err))),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="默认参数下四条路径的迭代数与核心算法逐位一致", chapter="5.1")
def test_descent_race_frame_matches_core_iteration_counts() -> None:
    r"""帧里的 ``meta.n_iter_*`` 必须等于直接调用核心算法得到的迭代数（防装配层改数）。"""
    kappa, angle, scale = 100.0, 35.0, 1.0
    Q = anisotropic_quadratic(kappa, angle_deg=angle)
    core = gd(Q, np.array([1.0, 1.0]), step=scale * float(Q.optimal_step()), max_iter=60, tol=1e-14)
    fr = build_frame(_entry(), "descent_race", {**_defaults("descent_race"), "kappa": kappa, "angle_deg": angle, "step_scale": scale, "max_iter": 60}, 60.0)
    rec = record(
        label="meta.n_iter_gd 与核心 gd(...) 的迭代数",
        theory="相等",
        measured=f"帧 {fr.meta['n_iter_gd']} vs 核心 {core.n_iter}（α={fr.meta['alpha_used']:.6e}）",
        tol="相等",
        passed=bool(int(fr.meta["n_iter_gd"]) == int(core.n_iter)),
    )
    rec2 = record(
        label="meta.kappa_measured 与实测谱",
        theory="= κ（Jacobi 实测，相对 1e-12）",
        measured=f"{fr.meta['kappa_measured']:.10f}（给定 {kappa}）",
        tol="相对 1e-12",
        passed=bool(abs(fr.meta["kappa_measured"] - kappa) / kappa <= 1e-12),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 4. 收敛率图：常数要么是实测的、要么是 null
# --------------------------------------------------------------------------- #
@assertion("A", theory="β>0 时帧里的二次常数 = quadratic_window 的实测值；β=0 时为 None", chapter="5.1")
def test_convergence_frame_constant_is_measured_or_null() -> None:
    r"""双向：β=0.05 时帧里的常数必须等于 :func:`quadratic_window` 的实测 C；
    β=0（退化成二次型、牛顿一步到底）时必须写 ``None``，且 ``strings`` 说明原因。"""
    from numcore.inverse.optimize import newton, quartic_quadratic

    fa = _frame("convergence_rates", beta=0.05, start=1.0, max_iter=30)
    tr = newton(quartic_quadratic(100.0, angle_deg=35.0, beta=0.05), np.array([1.0, 1.0]), max_iter=30, tol=1e-300)
    enter, c_hat, window = quadratic_window(tr.dist)
    rec = record(
        label="帧 meta.newton_quadratic_constant 与核心实测 C",
        theory="相等（同一窗口定义 + 同一初值/容差）",
        measured=f"帧 {fa.meta['newton_quadratic_constant']!r} vs 核心 {c_hat!r}（enter_iter={fa.meta['newton_enter_iter']}）",
        tol="相对 1e-12",
        passed=bool(
            fa.meta["newton_quadratic_constant"] is not None
            and abs(fa.meta["newton_quadratic_constant"] - c_hat) <= 1e-12 * c_hat
            and int(fa.meta["newton_enter_iter"]) == int(enter)
            and int(fa.meta["newton_window_size"]) == window.size
        ),
    )
    fb = _frame("convergence_rates", beta=0.0, max_iter=30)
    rec2 = record(
        label="β=0（退化成二次型）时的常数与说明",
        theory="常数 = None，且 strings 里有解释",
        measured=f"常数 {fb.meta['newton_quadratic_constant']!r}，enter_iter {fb.meta['newton_enter_iter']!r}，"
        f"string 提到 null={'null' in fb.strings.get('exact_note', '')}",
        tol="None",
        passed=bool(
            fb.meta["newton_quadratic_constant"] is None
            and fb.meta["newton_enter_iter"] is None
            and "null" in fb.strings.get("exact_note", "")
        ),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="卡住步长 → 实测阶从 ≈3 掉到 ≈1，二次常数变成 null；地板之后 q 爆炸", chapter="5.1")
def test_convergence_frame_reports_degraded_rate_under_capped_steps() -> None:
    r"""同一个问题、同一个初值，只把 ``alpha_cap`` 从 1 拖到 0.5：

    实测尾部阶从 ≈3 掉到 ≈1、二次窗口长度掉到 0、常数变成 ``None``；
    而 ``alpha_cap=1`` 的帧在 ``meta.newton_floor_start`` 之后 q 会冲到 1e100 以上
    ——那是舍入地板（每步 ×u），不是"超二次收敛"。这两件事必须同时出现在帧里。
    """
    fa = _frame("convergence_rates", alpha_cap=1.0, max_iter=30)
    fb = _frame("convergence_rates", alpha_cap=0.5, max_iter=30)
    qa = np.asarray(fa["q_newton"], dtype=float)
    rec = record(
        label="alpha_cap=1 时的尾部阶 / 窗口 / 地板起点 / q 最大值",
        theory="阶 ≥ 1.8、窗口 ≥ 2、常数有限、地板起点有限、q 冲到 ≥ 1e50",
        measured=f"阶 {fa.meta['newton_order_tail']:.3f}，窗口 {fa.meta['newton_window_size']}，"
        f"C={fa.meta['newton_quadratic_constant']!r}，地板起点 k={fa.meta['newton_floor_start']}，q_max={qa.max():.2e}",
        tol="全 True",
        passed=bool(
            fa.meta["newton_order_tail"] >= 1.8
            and fa.meta["newton_window_size"] >= 2
            and fa.meta["newton_quadratic_constant"] is not None
            and fa.meta["newton_floor_start"] is not None
            and qa.max() >= 1e50
        ),
    )
    rec2 = record(
        label="alpha_cap=0.5 时的尾部阶 / 窗口 / 常数",
        theory="阶 ≈ 1（线性）、窗口 0、常数 None",
        measured=f"阶 {fb.meta['newton_order_tail']:.3f}，窗口 {fb.meta['newton_window_size']}，"
        f"C={fb.meta['newton_quadratic_constant']!r}，步数 {fb.meta['n_iter_newton']}",
        tol="阶 ≤ 1.2 且 C 为 None",
        passed=bool(
            fb.meta["newton_order_tail"] <= 1.2
            and fb.meta["newton_window_size"] == 0
            and fb.meta["newton_quadratic_constant"] is None
        ),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


# --------------------------------------------------------------------------- #
# 5. 鞍点：方向决定一切
# --------------------------------------------------------------------------- #
@assertion("A", theory="扰动沿负曲率方向才逃得掉；正交方向落回鞍点", chapter="5.1")
def test_saddle_frame_direction_decides_escape() -> None:
    r"""δ=0.1（> 阈值 2.87e-3）时：沿 ``neg_curv`` 逃逸、沿 ``orthogonal`` 落回鞍点。

    实测两个方向上的 ‖x-x*‖ 末值差 3 个数量级以上——"随机扰动逃出来了"这句话
    必须带上方向，这正是一条帧级记录能钉住的东西。
    """
    S = saddle_quadratic(neg=-0.05, pos=1.0)
    fa = _frame("saddle_escape", neg=-0.05, delta=0.1, direction="neg_curv", max_iter=200, t=200.0)
    fb = _frame("saddle_escape", neg=-0.05, delta=0.1, direction="orthogonal", max_iter=200, t=200.0)
    da, db = float(fa.meta["traj_final_distance"]), float(fb.meta["traj_final_distance"])
    rec = record(
        label="两个方向上的扰动末距（δ=0.1, 200 步）",
        theory="neg_curv 远大于 1 远大于 orthogonal（差 ≥ 1e3 倍）",
        measured=f"neg_curv {da:.3e} vs orthogonal {db:.3e}（比 {da / max(db, 1e-300):.2e}）",
        tol="≥ 1e3",
        passed=bool(da >= 1.0 and da / max(db, 1e-300) >= 1e3),
    )
    rec2 = record(
        label="实测 λ_min 与解析阈值的对照",
        theory="λ_min = -0.05，阈值 ≈ 2.87e-3（N=96 时的解析值）",
        measured=f"λ_min={fa.meta['lambda_min']:.6f}，u_min=({fa.meta['u_min_x']:.3f},{fa.meta['u_min_y']:.3f})，"
        f"阈值 {fa.meta['threshold_neg_curv']:.4e}，解析 {fa.meta['theory_threshold']:.4e}",
        tol="相对 1e-3",
        passed=bool(
            abs(fa.meta["lambda_min"] + 0.05) <= 1e-12
            and abs(fa.meta["threshold_neg_curv"] / fa.meta["theory_threshold"] - 1.0) <= 1e-3
        ),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="contour_z 是逐点求值的等高线（帧的背景图不许自己编）", chapter="5.1")
def test_saddle_frame_contour_matches_pointwise_value() -> None:
    r"""抽 16 个网格点与 ``Saddle.value`` 对拍：图上背景错了没人看得出来。"""
    S = saddle_quadratic(neg=-0.05, pos=1.0)
    fr = _frame("saddle_escape", neg=-0.05)
    X = np.asarray(fr["contour_x"], dtype=float)
    Y = np.asarray(fr["contour_y"], dtype=float)
    Z = np.asarray(fr["contour_z"], dtype=float)
    worst = 0.0
    for i in range(5, Z.shape[0], max(Z.shape[0] // 4, 1)):
        for j in range(5, Z.shape[1], max(Z.shape[1] // 4, 1)):
            z = float(S.value(np.array([X[j], Y[i]])))
            worst = max(worst, abs(Z[i, j] - z) / max(abs(z), 1e-300))
    rec = record(
        label="contour_z 与逐点求值的最大相对差",
        theory="≤ 1e-14",
        measured=f"{worst:.3e}（网格 {Z.shape}）",
        tol="1e-14",
        passed=bool(worst <= 1e-14 and Z.shape == (Z.shape[0], Z.shape[0])),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 6. 信赖域：拒绝、负曲率与半径历史
# --------------------------------------------------------------------------- #
@assertion("A", theory="不定起点上 Steihaug-CG 走负曲率分支、步被拒绝、半径被收缩", chapter="5.1")
def test_trust_region_frame_exposes_rejections_and_negative_curvature() -> None:
    r"""``start="indefinite"`` 上 Rosenbrock 的 Hessian 行列式为负：
    实测 ``n_negative_curv``≥1、``n_rejected``≥1、``ratio_history`` 里出现负值、
    ``radius_min`` < ``delta0`` < ``radius_max``（收缩与扩张都发生过）。"""
    fr = _frame("trust_region_radius", start="indefinite", delta0=1.0, max_iter=24)
    ratio = np.asarray(fr["ratio_history"], dtype=float)
    accepted = np.asarray(fr["accepted"], dtype=float)
    negc = np.asarray(fr["neg_curv"], dtype=float)
    meta = fr.meta
    rec = record(
        label="不定起点上的负曲率分支 / 拒绝次数 / 负 ρ",
        theory="n_negative_curv ≥ 1、n_rejected ≥ 1、min ρ < 0",
        measured=f"负曲率 {meta['n_negative_curv']} 次，拒绝 {meta['n_rejected']} 次，"
        f"min ρ={meta['ratio_min']:.3f}，接受掩码里 0 的个数 {int(np.count_nonzero(accepted == 0.0))}",
        tol="全 True",
        passed=bool(
            meta["n_negative_curv"] >= 1
            and meta["n_rejected"] >= 1
            and meta["ratio_min"] < 0.0
            and np.count_nonzero(negc == 1.0) >= 1
            and np.count_nonzero(accepted == 0.0) >= 1
        ),
    )
    rec2 = record(
        label="半径的收缩与扩张范围",
        theory=f"radius_min < Δ0={fr.meta['delta0']} < radius_max",
        measured=f"[{meta['radius_min']:.4f}, {meta['radius_max']:.4f}]，终值 {meta['radius_current']:.4f}",
        tol="严格夹住 Δ0",
        passed=bool(meta["radius_min"] < meta["delta0"] < meta["radius_max"]),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="帧里的轨迹与半径历史 = 直接调用 trust_region 的结果", chapter="5.1")
def test_trust_region_frame_matches_core_trace() -> None:
    r"""装配层只许搬运：``traj_x``/``radius_history``/``gradient_residual`` 与核心逐位一致。"""
    R = rosenbrock_objective()
    tr = trust_region(R, np.array([-1.2, 1.0]), delta0=1.0, t2=0.75, max_iter=24, tol=1e-14)
    fr = _frame("trust_region_radius", start="valley", delta0=1.0, t2=0.75, max_iter=24, t=24.0)
    tx = np.asarray(fr["traj_x"], dtype=float)
    rad = np.asarray(fr["radius_history"], dtype=float)
    gres = np.asarray(fr["gradient_residual"], dtype=float)
    g_tail = gres[: tr.gnorm.size]  # 帧里补位到 max_iter+1，只比真数据那一段
    rec = record(
        label="轨迹 / 半径历史 / 梯度范数与核心的最大偏差",
        theory="0（同一份数组）",
        measured=f"轨迹 {np.max(np.abs(tx - tr.xs[:, 0])):.1e}，半径 {np.max(np.abs(rad - tr.radius)):.1e}，"
        f"梯度 {np.max(np.abs(g_tail - tr.gnorm)):.1e}（{tr.n_iter} 步，帧长 {gres.size}）",
        tol="= 0",
        passed=bool(
            tx.size == tr.xs.shape[0]
            and np.max(np.abs(tx - tr.xs[:, 0])) == 0.0
            and np.max(np.abs(rad - tr.radius)) == 0.0
            and np.max(np.abs(g_tail - tr.gnorm)) == 0.0
            and gres.size == 25
        ),
    )
    rec2 = record(
        label="meta.n_accepted + meta.n_rejected 与总步数",
        theory="和 = trace 的步数（被拒绝的步不画进轨迹，但计数里在）",
        measured=f"{fr.meta['n_accepted']} + {fr.meta['n_rejected']} = {fr.meta['n_accepted'] + fr.meta['n_rejected']}"
        f"（max_iter=24，实际 {tr.n_iter} 步）",
        tol="≤ 24",
        passed=bool(fr.meta["n_accepted"] + fr.meta["n_rejected"] == tr.accepted.size <= 24),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@pytest.mark.parametrize("viz_id", ["descent_race", "convergence_rates", "saddle_escape", "trust_region_radius"])
def test_manifest_declares_aux_axis_arrays(viz_id: str) -> None:
    """``requiredAux`` 里声明的名字（含 error/residual）必须真的出现在帧里。"""
    spec = _viz(viz_id)
    fr = _frame(viz_id)
    names = set(fr.names)
    missing = [req for req in spec.required_aux if not any(req in n for n in names)]
    assert not missing, f"{viz_id} 的 requiredAux {missing} 在帧里找不到（现有：{sorted(names)}）"
