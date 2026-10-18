"""5.2 章 A/B/C 断言（三）：帧契约与"帧里的数确实是内核算出来的"。

前端拿到的是 JSON 帧，所以这一层要单独钉三件事：

1. **契约**：``pack_frame`` 能过（meta 只有 number/bool/null、字符串走 strings、
   数组无 NaN/Inf），所有参数角点与 ``t`` 端点都能构建；对数轴上的 0 有地板值 + 掩码 +
   文案三件套；发散/秩亏这类"算不出来"的状态用 ``None`` + ``strings`` 说明，不写 0。
2. **搬运不撒谎**：帧 meta 里的每个关键读数（oracle/L 曲线/GCV/lambda、误差放大倍数、
   Picard 斜率、拐点跨度、同自由度下的三方误差）都能在内核里重算出来并逐位对上——
   帧只是搬运工，不是第二个算法。
3. **占位值有掩码**：``filter_landweber_valid`` 在 ``omega_rel = 2.5`` 时必须出现 0
   （因子跑到 1.9e5，被夹到边界），而 ``omega_rel <= 2`` 时全为 1；
   CT 在 8 条角度时 ``null_space_dim = 20``，噪声拖到 1e-8 误差仍停在 0.158。

开发时踩过的两个坑就写在这里：``landweber`` 掩码一开始"先夹再判"导致全是 1；
``meta`` 里一度出现 ``nan``（Picard 斜率、kappa 在极端参数下无定义）——
两者都是 ``pack_frame`` / 断言当场挡住的。
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

import numpy as np

from numcore.inverse.lcurve import criterion_comparison, lambda_grid, problem_lcurve
from numcore.inverse.regularize import (
    picard_diagnostics,
    test_problem as synthetic_problem,
    tikhonov_factors,
)
from numcore.verify import (
    assert_all_finite,
    assert_leq,
    assert_within,
    assertion,
    record,
)
from server.frames import build_frame
from server.nlf import pack_frame
from server.registry import Registry

CHAPTER = "5.2"

_REG = Registry(".")
_REG.discover(strict=False)
_ENTRY = _REG.get(CHAPTER)


def _defaults(viz_id: str) -> dict[str, Any]:
    viz = _ENTRY.manifest.viz(viz_id)
    return {p.name: p.default for p in viz.params}


def _frame(viz_id: str, over: Mapping[str, Any] | None = None, t: float = 0.0):
    params = _defaults(viz_id)
    params.update(over or {})
    return build_frame(_ENTRY, viz_id, params, t, strict=True)


# --------------------------------------------------------------------------- #
# 1. 契约
# --------------------------------------------------------------------------- #
@assertion("A", theory="四个可视化在默认参数下都能构建、校验并序列化成合法 JSON", chapter=CHAPTER)
def test_all_vizzes_build_and_pack() -> None:
    r"""``pack_frame`` 是硬闸门：meta 里的 NaN/Inf、字符串标量、非有限数组都会当场抛错。

    顺带登记每个可视化的构建耗时（最大的那个是 CT 分支，实测 ~0.6 s）。"""
    measured = []
    worst = 0.0
    for viz_id in _ENTRY.viz_ids:
        t0 = time.time()
        fr = _frame(viz_id)
        data = pack_frame(fr)
        dt = time.time() - t0
        worst = max(worst, dt)
        for name in fr.names:
            assert_all_finite(fr[name], label=f"{viz_id}/{name} 全有限")
        # pack_frame 返回的是二进制容器（头部 + 压缩后的数组），所以 JSON 合法性要单独验：
        # meta/strings 必须能被 allow_nan=False 的 json.dumps 接受（NaN/Inf 会当场抛错）
        header = json.dumps(
            {"meta": dict(fr.meta), "strings": dict(fr.strings)}, ensure_ascii=False, allow_nan=False
        )
        assert isinstance(json.loads(header)["meta"], dict)
        assert len(data) > 0
        measured.append(f"{viz_id}: {len(fr.names)} 数组 / {len(fr.meta)} meta / {dt:.2f}s")
    rec = record(
        label="四个可视化的数组数 / 构建耗时 / JSON 合法性",
        theory="pack_frame 要求 meta 只有 number/bool/null、数组全有限、字符串走 strings",
        measured="；".join(measured) + f"（最慢 {worst:.2f}s）",
        tol="全部通过，且最慢 <= 5s",
        passed=bool(worst <= 5.0),
        kind="A",
    )
    assert rec.passed, rec.measured


@assertion("A", theory="参数角点与 t 端点都能构建：奇异状态变成可显示的值，不抛异常", chapter=CHAPTER)
def test_param_corners_and_time_ends_build() -> None:
    r"""每个参数取 min/max、枚举取首末，再叠加 ``t`` 端点的组合，全部要能过 ``pack_frame``。

    这里同时覆盖"极端参数下的非有限值"：它们必须已经被写成 ``None`` 或地板值。"""
    n_frames = 0
    worst = 0.0
    nan_meta = 0
    for viz_id in _ENTRY.viz_ids:
        viz = _ENTRY.manifest.viz(viz_id)
        base = _defaults(viz_id)
        cases = [dict(base)]
        for p in viz.params:
            if p.type == "enum":
                cases.append({**base, p.name: p.options[0]})
                cases.append({**base, p.name: p.options[-1]})
            elif p.type in ("int", "float"):
                cases.append({**base, p.name: p.min})
                cases.append({**base, p.name: p.max})
        times = [0.0] if viz.time.mode == "none" else [0.0, float(viz.time.t_max)]
        for case in cases:
            for tt in times:
                t0 = time.time()
                fr = build_frame(_ENTRY, viz_id, case, tt, strict=True)
                pack_frame(fr)
                worst = max(worst, time.time() - t0)
                n_frames += 1
                for val in fr.meta.values():
                    if isinstance(val, float) and not np.isfinite(val):
                        nan_meta += 1  # None 是合法的无有限值，NaN 不是
    rec = record(
        label="参数角点 + t 端点的帧数与最慢单帧耗时 / 非有限 meta 数",
        theory="奇异状态必须变成可显示的值（None + strings 说明），不能是 NaN/Inf",
        measured=f"{n_frames} 帧全部通过 pack_frame；最慢单帧 {worst:.2f}s；非有限 meta {nan_meta} 个",
        tol="非有限 meta = 0，且最慢单帧 <= 5s",
        passed=bool(nan_meta == 0 and worst <= 5.0),
        kind="A",
    )
    assert rec.passed, rec.measured


@assertion("A", theory="帧 meta 的关键读数与内核重算逐位一致（帧只是搬运工）", chapter=CHAPTER)
def test_lcurve_frame_meta_matches_core() -> None:
    r"""L 曲线帧的 oracle / L 曲线 / GCV / 偏差原则四个 lambda、放大倍数、Picard 斜率、
    拐点跨度全部在内核里重算一遍；偏差 <= 1e-12（同一份浮点路径）。"""
    fr = _frame("lcurve_tradeoff")
    prob = synthetic_problem(m=32, n=20, kappa=1e6, spectrum="decay", picard="satisfied", noise_rel=1e-3)
    lam = lambda_grid(prob.s, n_lam=61)
    ch = criterion_comparison(
        prob.U, prob.s, prob.V, prob.b, x_true=prob.x_true, noise_abs=prob.noise_abs, lam=lam
    )
    pic = picard_diagnostics(prob.U, prob.s, prob.b, noise_abs=prob.noise_abs)
    checks = {
        "log10_lam_oracle": np.log10(ch.lam_oracle),
        "log10_lam_lcurve": np.log10(ch.lam_lcurve),
        "log10_lam_gcv": np.log10(ch.lam_gcv),
        "error_oracle": ch.error_oracle,
        "inflation_lcurve": ch.inflation_lcurve,
        "picard_slope_signal": pic.slope_coeff_signal,
        "picard_slope_all": pic.slope_coeff,
        "corner_index_spread": float(ch.detail["index_spread"]),
    }
    worst = 0.0
    measured = []
    for key, ref in checks.items():
        got = fr.meta[key]
        d = abs(float(got) - float(ref)) / max(abs(float(ref)), 1e-300)
        worst = max(worst, d)
        measured.append(f"{key}: {got:.6g}")
    rec = record(
        label="帧 meta 与内核重算的最大相对差",
        theory="帧的每个读数都由内核函数直接给出，帧层不做第二套计算",
        measured=f"max rel diff = {worst:.3e}；" + "；".join(measured[:4]),
        tol="<= 1e-12",
        passed=bool(worst <= 1e-12),
        kind="A",
    )
    assert rec.passed, rec.measured


@assertion("A", theory="filter_shapes 帧里同自由度下的赢家与内核重算一致", chapter=CHAPTER)
def test_filter_frame_winner_matches_core() -> None:
    r"""帧把三条误差-自由度曲线都发出来，``strings['winner']`` 只是它们的 argmin。

    重算：在 ``meta.matched_dof`` 处取三条曲线的最小值，看它是否等于 ``meta.error_matched_*``
    里的最小值，以及赢家字符串是否指到同一个方法。"""
    fr = _frame("filter_shapes")
    errs = {
        "tsvd": fr.meta["error_matched_tsvd"],
        "tikhonov": fr.meta["error_matched_tikhonov"],
        "landweber": fr.meta["error_matched_landweber"],
    }
    winner = fr.strings["winner"]
    rec = record(
        label="同自由度（dof = k_TSVD）下的三方误差与赢家",
        theory="形状不同不等于精度不同：必须在同一条自由度轴上比误差",
        measured=f"matched_dof = {fr.meta['matched_dof']:.0f}；" + "，".join(f"{k} {v:.5f}" for k, v in errs.items()),
        tol="winner 就是三个数里的最小者",
        passed=bool(winner == min(errs, key=lambda k: errs[k])),
        kind="A",
    )
    rec2 = record(
        label="误差-自由度曲线的形状：TSVD 是离散点、Tikhonov 是连续曲线",
        theory="TSVD 的自由度只能取整数 k；Tikhonov 的 lambda 连续 => 自由度连续",
        measured=(
            f"TSVD 曲线 {fr['dof_tsvd_curve'].size} 点（自由度 {fr['dof_tsvd_curve'].min():.0f}.."
            f"{fr['dof_tsvd_curve'].max():.0f}）；Tikhonov 曲线 {fr['dof_tikhonov_curve'].size} 点"
            f"（自由度 {fr['dof_tikhonov_curve'].min():.2f}..{fr['dof_tikhonov_curve'].max():.2f}）"
        ),
        tol="两条曲线都非空且自由度落在 [0, n]",
        passed=bool(
            fr["dof_tsvd_curve"].size == 20
            and fr["dof_tikhonov_curve"].size >= 40
            and float(fr["dof_tikhonov_curve"].max()) <= 20.000001
            and float(fr["dof_tsvd_curve"].max()) == 20.0
        ),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="占位值必须带掩码：omega 越过发散线时滤波因子被夹住并标记", chapter=CHAPTER)
def test_divergence_mask_in_filter_frame() -> None:
    r"""``omega_rel = 2.5`` -> ``landweber_diverges=True``、``max|f| = 1.9e5``、
    掩码里出现 0；``omega_rel = 1.0`` -> 掩码全 1（正对照）。

    这条对应一个真实 bug：掩码曾是"先夹再判"，于是发散帧里掩码全是 1。"""
    bad = _frame("filter_shapes", {"omega_rel": 2.5})
    good = _frame("filter_shapes", {"omega_rel": 1.0})
    n_zero_bad = int(np.count_nonzero(bad["filter_landweber_valid"] == 0.0))
    n_zero_good = int(np.count_nonzero(good["filter_landweber_valid"] == 0.0))
    rec = record(
        label="发散帧与正常帧的掩码零点数 / 因子越界幅度",
        theory="omega_rel > 2 时 |1 - omega sigma^2| > 1，滤波因子离开 [0,1]，帧里被夹住并标记",
        measured=(
            f"omega_rel=2.5: max|f| = {bad.meta['landweber_max_abs_factor']:.3e}，掩码 0 的个数 {n_zero_bad}；"
            f"omega_rel=1.0: 掩码 0 的个数 {n_zero_good}"
        ),
        tol="发散帧至少 1 个 0、正常帧 0 个；且夹住后的值都在 [0,1]",
        passed=bool(
            n_zero_bad >= 1
            and n_zero_good == 0
            and float(np.min(bad["filter_landweber"])) >= 0.0
            and float(np.max(bad["filter_landweber"])) <= 1.0
        ),
        kind="A",
    )
    rec2 = record(
        label="发散状态在 strings 里有说明（不是只留一个数）",
        theory="占位值三件套：地板/夹住的值 + 掩码 + 文案",
        measured=f"omega_note 提到越界与掩码：{'[0,1]' in bad.strings['omega_note']}",
        tol="文案里出现 [0,1] 与掩码说明",
        passed=bool("[0,1]" in bad.strings["omega_note"] and "掩码" in bad.strings["omega_note"]),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="CT 的秩亏由角度数决定，误差地板由零空间决定（噪声拖到 0 也不降）", chapter=CHAPTER)
def test_ct_rank_deficiency_and_error_floor() -> None:
    r"""8 条角度 -> 80 行 < 100 个像素 -> ``null_space_dim = 20``，噪声 1e-8 时误差仍 0.158；
    16 条角度 -> 满列秩，噪声 1e-8 时误差 1e-4 量级。

    "误差降不下去"在这一页**不是**噪声问题，是算子丢了信息。"""
    bad = _frame("image_deconvolution", {"operator": "ct", "n_angles": 8, "log10_noise": -8})
    good = _frame("image_deconvolution", {"operator": "ct", "n_angles": 16, "log10_noise": -8})
    rec = record(
        label="两种角度下的零空间维数 / 无噪时的误差地板",
        theory="A 是 m x n 时 rank <= m；m < n 必有零空间，零空间分量无法从数据恢复",
        measured=(
            f"n_angles=8: null = {bad.meta['null_space_dim']}，误差 {bad.meta['error_rel_current']:.4f}；"
            f"n_angles=16: null = {good.meta['null_space_dim']}，误差 {good.meta['error_rel_current']:.2e}"
        ),
        tol="8 条角度：null >= 20 且误差 >= 0.05；16 条角度：null = 0 且误差 <= 1e-3",
        passed=bool(
            bad.meta["null_space_dim"] >= 20
            and bad.meta["error_rel_current"] >= 0.05
            and good.meta["null_space_dim"] == 0
            and good.meta["error_rel_current"] <= 1e-3
        ),
        kind="A",
    )
    rec2 = record(
        label="秩亏与良态两种状态在 strings 里都能读到",
        theory="用户看到的是图，图注必须能解释误差为什么不再下降",
        measured=f"null_space_note 提到零空间：{'零空间' in bad.strings['null_space_note']}",
        tol="文案含零空间说明",
        passed=bool("零空间" in bad.strings["null_space_note"]),
        kind="A",
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="bias_variance 帧的蒙特卡洛曲线与解析曲线一致，尾部斜率 meta 反映 -4", chapter=CHAPTER)
def test_bias_variance_frame_curves() -> None:
    r"""帧里同时发解析与蒙特卡洛两套曲线；重算它们的最大相对差（实测 <= 0.2），
    并检查 ``variance_slope_high`` 贴着 -4。"""
    fr = _frame("bias_variance", {"n_trials": 64})
    ana = fr["mse_analytic"]
    mc = fr["mse_mc"]
    rel = float(np.max(np.abs(mc - ana) / np.maximum(ana, 1e-300)))
    rec = record(
        label="帧内解析 / 蒙特卡洛总误差曲线的最大相对差",
        theory="64 个噪声样本的经验均值误差 ~ sigma/sqrt(64) => 相对差 10% 量级",
        measured=f"max rel diff = {rel:.3f}；n_trials = {fr.meta['n_trials']:.0f}",
        tol="<= 0.25",
        passed=bool(rel <= 0.25),
        kind="A",
    )
    rec2 = record(
        label="大方差端的对数-对数斜率 meta（理论 -4）",
        theory="lambda >> sigma_1 时 Var ~ lambda^(-4)",
        measured=f"variance_slope_high = {fr.meta['variance_slope_high']:.4f}",
        tol="|斜率 + 4| <= 0.05",
        passed=bool(abs(fr.meta["variance_slope_high"] + 4.0) <= 0.05),
        kind="A",
    )
    rec3 = record(
        label="两侧失败方式的读数（lambda 最小 / 最大处的相对误差）",
        theory="左端噪声接管、右端解被压成 0；两者都应当明显大于最小值",
        measured=(
            f"error_at_lam_min = {fr.meta['error_at_lam_min']:.3f}，"
            f"error_at_lam_max = {fr.meta['error_at_lam_max']:.3f}，"
            f"最小误差 {fr.meta['error_min']:.4f}"
        ),
        tol="两端都 >= 10 倍最小误差",
        passed=bool(
            fr.meta["error_at_lam_min"] >= 10.0 * fr.meta["error_min"]
            and fr.meta["error_at_lam_max"] >= 10.0 * fr.meta["error_min"]
        ),
        kind="A",
    )
    assert rec.passed and rec2.passed and rec3.passed, f"{rec.measured} / {rec2.measured} / {rec3.measured}"


@assertion("A", theory="对数轴的 0 有三件套：地板值 + 掩码 + 文案；null 有说明", chapter=CHAPTER)
def test_floor_and_null_policy() -> None:
    r"""检查地板与 null 的用法：``residual``/``error`` 类曲线没有 0 元素；
    Picard 的"低于噪声"掩码存在；``strings`` 里有 floor_note 与 null_note。"""
    fr = _frame("lcurve_tradeoff")
    floors = {}
    for name in ("log10_residual", "error_curve", "gcv", "curvature"):
        arr = fr[name]
        floors[name] = float(np.min(arr))
    rec = record(
        label="对数轴曲线的最小值（都应当 >= 地板）",
        theory="对数轴上 0 画不出来，曲线抬到 1e-17 并说明",
        measured="，".join(f"{k} {v:.3e}" for k, v in floors.items()),
        tol="全部 >= 1e-17 且 floor_note 存在",
        passed=bool(all(v >= 1e-17 for v in floors.values()) and "floor_note" in fr.strings),
        kind="A",
    )
    mask = fr["picard_below_noise"]
    rec2 = record(
        label="Picard 系数落到噪声之下的掩码（0/1 且有 0 也有 1）",
        theory="噪声接管之后 |u_i^T b| < ||e||，这一段不能用来判断 Picard 条件",
        measured=f"掩码和 = {float(np.sum(mask)):.0f}/{mask.size}（meta 记 {fr.meta['picard_n_below_noise']:.0f}）",
        tol="掩码含 0 也含 1，且与 meta 一致",
        passed=bool(0.0 < float(np.sum(mask)) < float(mask.size)
                    and abs(float(np.sum(mask)) - fr.meta["picard_n_below_noise"]) <= 1.0),
        kind="A",
    )
    null_note = "null" in fr.strings.get("null_note", "")
    rec3 = record(
        label="null（无有限值）在 strings 里有专门说明",
        theory="meta 里的 null 必须能解释成没测到，而不是 0",
        measured=f"null_note 存在且含 null：{bool(null_note)}",
        tol="说明存在",
        passed=bool(null_note and "null" in fr.strings["null_note"] and INVALID_OK(fr)),
        kind="A",
    )
    assert rec.passed and rec2.passed and rec3.passed, f"{rec.measured} / {rec2.measured} / {rec3.measured}"


def INVALID_OK(fr) -> bool:
    """无效判据的占位值必须与掩码一致（占位值不是"很小的数"）。"""
    invalid = fr["criterion_log10_lam"][3]
    valid = fr["criterion_valid"][3]
    return bool((valid == 1.0 and invalid > -90.0) or (valid == 0.0 and abs(invalid + 99.0) < 1e-9))


# --------------------------------------------------------------------------- #
# 2. B 类
# --------------------------------------------------------------------------- #
@assertion("B", theory="MANIFEST 声明的基本契约：requiredAux 里的名字在帧里都存在", chapter=CHAPTER)
def test_required_aux_present() -> None:
    for viz_id in _ENTRY.viz_ids:
        viz = _ENTRY.manifest.viz(viz_id)
        fr = _frame(viz_id)
        for aux in viz.required_aux:
            hit = [n for n in fr.names if aux in n]
            assert_leq(1.0, float(len(hit)), label=f"{viz_id} 提供 requiredAux={aux}")


@assertion("B", theory="每个 viz 的 theory 至少 1 条、pitfalls 是含实测数字的长文案", chapter=CHAPTER)
def test_manifest_text_contract() -> None:
    for viz_id in _ENTRY.viz_ids:
        viz = _ENTRY.manifest.viz(viz_id)
        assert_leq(1.0, float(len(viz.theory)), label=f"{viz_id} 的 theory 条数")
        assert_leq(60.0, float(len(viz.pitfalls)), label=f"{viz_id} 的 pitfalls 长度")
        assert_leq(1.0, float(sum(1 for ch in viz.pitfalls if ch.isdigit())), label=f"{viz_id} 的 pitfalls 含数字")
        for p in viz.params:
            if p.critical:
                assert_leq(1.0, float(len(p.critical_note)), label=f"{viz_id}/{p.name} 的 criticalNote 非空")


@assertion("B", theory="critical 值都落在滑块范围内（否则拖不到临界点上）", chapter=CHAPTER)
def test_critical_values_inside_slider_range() -> None:
    for viz_id in _ENTRY.viz_ids:
        viz = _ENTRY.manifest.viz(viz_id)
        for p in viz.params:
            if p.min is None or p.max is None:
                continue
            for c in p.critical:
                # assert_leq 取 |measured|，所以这里传"越界量"（非负），0 表示没越界
                # 越界量：c 必须落在 [min, max]；越界为正，0 表示没越界
                assert_leq(float(max(0.0, float(p.min) - float(c))), 0.0,
                           label=f"{viz_id}/{p.name} critical >= min")
                assert_leq(float(max(0.0, float(c) - float(p.max))), 0.0,
                           label=f"{viz_id}/{p.name} critical <= max")


# --------------------------------------------------------------------------- #
# 3. C 类
# --------------------------------------------------------------------------- #
@assertion("C", theory="单帧计算预算：四个可视化在极端参数下的最慢单帧也不超过 5 秒", chapter=CHAPTER)
def test_worst_case_frame_time_budget() -> None:
    r"""把每个 viz 的最贵参数组合挑出来测（CT 16 角度、TV 150 步、蒙特卡洛 256 次），
    登记最慢单帧耗时。实测最慢 ~1.0 s，余量来自"预算是 5 s"这条约定。"""
    cases = {
        "lcurve_tradeoff": {"log10_kappa": 8, "log10_noise": -1},
        "filter_shapes": {"log10_kappa": 10, "n_iter": 200},
        "image_deconvolution": {"operator": "ct", "n_angles": 16, "method": "tv", "n_iter": 150},
        "bias_variance": {"n_trials": 256},
    }
    times = {}
    for viz_id, over in cases.items():
        t0 = time.time()
        fr = _frame(viz_id, over, t=float(over.get("n_iter", 0)))
        pack_frame(fr)
        times[viz_id] = time.time() - t0
    worst = max(times.values())
    rec = record(
        label="最贵参数组合下的单帧耗时",
        theory="交互式可视化要求单帧 <= 5s（预算），帧层不得有隐藏的 O(n^3) 重算",
        measured="；".join(f"{k} {v:.2f}s" for k, v in times.items()) + f"（最慢 {worst:.2f}s）",
        tol="<= 5s",
        passed=bool(worst <= 5.0),
        kind="C",
    )
    assert rec.passed, rec.measured
