"""5.3 章的**帧级**测试：把"图上看到的东西"与 `numcore` 的实测值钉在一起。

帧是唯一会被前端直接画出来的东西，所以这里检查的不是算法而是**呈现纪律**：

1. **没有非有限值**。四个可视化在默认参数、每个参数的两端、以及 ``t=0``/``t=tMax``
   全组合下都必须能 ``pack_frame``（JSON 头里不许出现 ``NaN``/``Infinity``）。
2. **帧不撒谎**。``meta`` 里的关键数字（梯度检验的最优 eps 与两段斜率、影响场的宽度偏差
   与半离散偏差、正向/伴随的放大率与连续-离散间隙）必须与直接调用 ``numcore`` 的
   同参数结果**逐位相同**——不能是"另算一套"。
3. **该给 null 就给 null**。把 eps 扫描窗口整个挪到 V 形的右侧（只剩截断段）时，
   "最优 eps"落在端点上，帧必须报 ``null`` 并把 ``*_interior`` 标成 false，
   而不是把端点当谷底报出去。
4. **占位三件套**。一旦某个数组里出现非有限值，同名的 ``*_valid`` 掩码与
   ``strings`` 里的说明必须同时出现（PROTOCOLS §1.3.1）。
"""

from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.inverse.adjoint import (  # noqa: E402
    HeatAdjointProblem,
    continuous_adjoint,
    discrete_adjoint,
    gradient_check_scan,
    rk_forward,
)
from numcore.inverse.sensitivity import (  # noqa: E402
    duffing_model,
    duffing_slice_model,
    gradient_field,
    influence_width_law,
)
from numcore.verify import assertion, record  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.nlf import pack_frame  # noqa: E402
from server.registry import Registry  # noqa: E402

_VIZZES = ("gradient_field", "influence_field", "gradient_check", "forward_adjoint_playback")
_ENTRY_CACHE: list = []


def _entry():
    if not _ENTRY_CACHE:
        reg = Registry(ROOT)
        reg.discover(strict=False)  # 并发开发时兄弟章节可能是半成品；本章必须能加载
        _ENTRY_CACHE.append(reg.get("5.3"))
    return _ENTRY_CACHE[0]


def _viz(viz_id: str):
    return _entry().manifest.viz(viz_id)


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
@assertion("A", theory="帧在参数端点与时间端点全组合下都只有有限值，且能装进 JSON", chapter="5.3")
def test_all_frames_finite_and_packable_at_corners() -> None:
    r"""参数端点 × 时间端点：数组全有限、``meta`` 无 NaN、JSON 里没有 ``Infinity``。

    比 ``tests/test_frames_contract.py`` 多跑一步：这里把 ``t`` 推到 ``tMax``
    （scrub 的最后一帧最容易越界——本章的 ``t`` 是"总时长的比例"，推到端点会取到
    ``tau=0`` 与 ``k_play=N`` 两个退化位置）。
    """
    n_cases = 0
    n_token = 0
    detail = []
    for viz_id in _VIZZES:
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
            blob = pack_frame(fr)
            n_token += blob.count(b"Infinity") + blob.count(b"NaN")
            n_cases += 1
        detail.append(f"{viz_id}: {len(fr.names)} 数组 / {len(fr.meta)} meta")
    rec = record(
        label="端点组合帧数与 JSON 里的 NaN/Infinity 计数",
        theory="NaN/Infinity 计数 = 0",
        measured=f"{n_cases} 帧，计数 {n_token}（" + "；".join(detail) + "）",
        tol="= 0",
        passed=bool(n_token == 0 and n_cases >= 40),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. 帧里的关键数字必须与 numcore 的直接计算一致
# --------------------------------------------------------------------------- #
@assertion("A", theory="帧 meta 的关键数字必须与直接调用 numcore 的同参数结果一致（帧不撒谎）", chapter="5.3")
def test_frame_meta_matches_direct_numcore_computation() -> None:
    r"""四帧各挑 2-3 个关键数字，与 numcore 同参数重算逐位对拍。

    这条是"图上的数不是手抄的"的机器检查：任何一处被四舍五入、抄错或用旧参数，
    这里都会红。
    """
    rows = []
    # (1) 梯度场：逐点最大误差
    fr = _frame("gradient_field")
    gf = gradient_field(
        duffing_slice_model(float(fr.meta["cubic"])),
        p1_range=(0.6, 1.6), p2_range=(0.03, 0.4), n1=11, n2=7, arrow_n1=4, arrow_n2=4,
        T=float(fr.meta["horizon"]), n_steps=int(fr.meta["n_steps"]),
    )
    assert fr.meta["max_rel_error"] == gf.max_rel_err(), "梯度场最大误差与 numcore 不一致"
    rows.append(f"梯度场 max_rel_error={fr.meta['max_rel_error']:.6e}")

    # (2) 影响场：宽度偏差、半离散偏差、梯度相对误差
    default = _defaults("influence_field")
    prob = HeatAdjointProblem(
        nx=41, length=1.0, alpha=float(default["alpha"]), horizon=float(default["horizon"]),
        n_steps=int(default["n_steps"]), centers=(0.35, 0.65), widths=(0.08, 0.08),
        obs_center=float(default["obs_center"]), obs_width=float(default["obs_width"]), theta=0.5,
    )
    st = influence_width_law(prob, np.array([1.0, 0.7]), n_sample=61)
    fr_inf = _frame("influence_field")
    assert fr_inf.meta["width_ratio_clean_max"] == st.width_ratio_clean_max, "宽度偏差不一致"
    assert fr_inf.meta["semidiscrete_rel_error_max"] == st.semidiscrete_rel_err_max, "半离散偏差不一致"
    assert fr_inf.meta["grad_fd_rel_error"] == st.grad_rel_err, "梯度相对误差不一致"
    rows.append(
        f"影响场 width_ratio_clean_max={fr_inf.meta['width_ratio_clean_max']:.6e}, "
        f"semidiscrete={fr_inf.meta['semidiscrete_rel_error_max']:.6e}"
    )

    # (3) 梯度检验：最优 eps、两段斜率
    chk_defaults = _defaults("gradient_check")
    eps_values = np.logspace(
        np.log10(float(chk_defaults["eps_min"])), np.log10(float(chk_defaults["eps_max"])), int(chk_defaults["n_eps"])
    )
    ck = gradient_check_scan(
        duffing_model(), np.array([1.0, 0.1, 0.05]), T=8.0, n_steps=int(chk_defaults["n_steps"]),
        method=str(chk_defaults["method"]), eps_values=eps_values, schemes=("central", "forward"),
    )
    fr_chk = _frame("gradient_check")
    assert fr_chk.meta["eps_opt_central"] == ck.best_eps("central"), "最优 eps 不一致"
    assert fr_chk.meta["slope_up_central"] == ck.slope_up["central"], "截断段斜率不一致"
    assert fr_chk.meta["slope_down_forward"] == ck.slope_down["forward"], "舍入段斜率不一致"
    rows.append(
        f"梯度检验 eps*={fr_chk.meta['eps_opt_central']:.3e}, 斜率 {fr_chk.meta['slope_down_central']:.4f}/"
        f"{fr_chk.meta['slope_up_central']:.4f}"
    )

    # (4) 播放：放大率与连续-离散间隙
    pb = _defaults("forward_adjoint_playback")
    pv = np.array([float(pb["stiffness"]), float(pb["damping"])], dtype=float)
    model = duffing_slice_model(float(pb["cubic"]))
    trace = rk_forward(model, pv, T=float(pb["horizon"]), n_steps=int(pb["n_steps"]), method="rk4", store="full")
    adj = discrete_adjoint(model, pv, trace)
    cont = continuous_adjoint(model, pv, trace)
    from numcore.inverse.adjoint import relative_error

    fr_pb = _frame("forward_adjoint_playback")
    assert fr_pb.meta["adjoint_growth_ratio"] == adj.growth_ratio(), "放大率不一致"
    assert fr_pb.meta["continuous_gap_max"] == relative_error(adj.grad, cont.grad), "连续-离散间隙不一致"
    rows.append(f"播放 growth={fr_pb.meta['adjoint_growth_ratio']:.6f}, 间隙={fr_pb.meta['continuous_gap_max']:.3e}")

    rec = record(
        label="四帧的关键 meta 与 numcore 直接重算的逐位一致性",
        theory="帧只是 numcore 结果的搬运工：同参数下的数字必须逐位相同",
        measured="；".join(rows),
        tol="逐位相同（==）",
        passed=True,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 3. 最优 eps 落在扫描端点时必须报 null
# --------------------------------------------------------------------------- #
@assertion("A", theory="最优 eps 落在扫描端点上时该报 null：窗口没盖住 V 形就没有'最优'可言", chapter="5.3")
def test_endpoint_optimum_is_reported_as_null_with_interior_flag_false() -> None:
    r"""把窗口整个挪到截断段（``eps_min=1e-6``）后，帧必须报 ``null`` 并标 false。

    这是"宁可少报一个数，也不报一个假的谷底"的机制检查：窗口整体在 V 形右侧
    （``eps_min`` 抬到它的上界 1e-4）时误差随 eps 单调上升，"最优"必然落在左端点——
    那只是窗口的边界，不是曲线的谷底。
    """
    fr = _frame("gradient_check", eps_min=1e-4)
    assert fr.meta["eps_opt_central"] is None, f"端点最优仍被报成 {fr.meta['eps_opt_central']}"
    assert fr.meta["eps_opt_forward"] is None, f"单侧差分的端点最优仍被报成 {fr.meta['eps_opt_forward']}"
    assert fr.meta["eps_opt_central_interior"] is False
    assert fr.meta["eps_opt_forward_interior"] is False
    assert fr.meta["error_best_central"] is not None, "谷底误差本身还是该报（它确实被扫到了）"
    # 反过来：默认窗口必须落在内部并给出数
    fr_ok = _frame("gradient_check")
    assert fr_ok.meta["eps_opt_central_interior"] is True and fr_ok.meta["eps_opt_central"] is not None
    rec = record(
        label="eps 扫描窗口挪到截断段后'最优 eps'的报 null 行为",
        theory="端点最优 -> eps_opt_* = null 且 *_interior = false；内部最优 -> 给出数值",
        measured=(
            f"eps_min=1e-4 时 central/forward 均为 {fr.meta['eps_opt_central']}, "
            f"interior={fr.meta['eps_opt_central_interior']}/{fr.meta['eps_opt_forward_interior']}；"
            f"默认窗口 {fr_ok.meta['eps_opt_central']:.2e}（interior=true）"
        ),
        tol="端点必须 null",
        passed=True,
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 4. 占位三件套与必备对照量
# --------------------------------------------------------------------------- #
@assertion("A", theory="非有限值必须换成占位值并同时给出掩码与文字说明；必备对照量必须在帧里", chapter="5.3")
def test_placeholder_triplet_and_required_aux_present() -> None:
    r"""``requiredAux`` 里的名字必须在帧里存在，且掩码/说明成对出现。

    本章四个可视化的必备对照量分别是：``rel_error_log10``（梯度场）、``width_error``（影响场）、
    ``error_central``/``error_forward``（V 形）、``sensitivity_residual``（播放）。
    最后一条是"图上没有'误差'这一项"的兜底检查。
    """
    rows = []
    for viz_id in _VIZZES:
        spec = _viz(viz_id)
        fr = _frame(viz_id)
        need = [a for a in spec.required_aux if a in ("error", "residual")]
        assert need, f"{viz_id} 的 requiredAux 里没有 error/residual"
        hit = [n for n in fr.names if any(k in n for k in need)]
        assert hit, f"{viz_id} 的帧里没有任何数组名包含 {need}"
        for name in fr.names:
            if name.endswith("_valid"):
                base = name[: -len("_valid")]
                assert base in fr.names, f"{viz_id}: 掩码 {name} 没有对应的数据数组"
                assert float(np.min(fr[name])) == 0.0, f"{viz_id}: 掩码 {name} 里没有 0（说明它没在标占位）"
        assert fr.strings, f"{viz_id} 没有任何 strings 说明"
        rows.append(f"{viz_id}: requiredAux {spec.required_aux} -> {hit}")
    rec = record(
        label="四个可视化的必备对照量与占位掩码的成对性",
        theory="requiredAux 指向的数组必须存在；*_valid 掩码必须与数据同行且含 0",
        measured="；".join(rows),
        tol="全部存在且成对",
        passed=True,
    )
    assert rec.passed, rec.measured
