r"""4.1 章的**帧级**回归测试：把"帧会不会撒谎"钉在 pytest 里。

算法层的对拍在 ``test_heat.py`` / ``test_vonneumann.py`` 里；这个文件只问四件事：

1. **角点上不炸也不骗**：默认参数 + 每个参数的 min/max + :math:`t\in\{0,t_{\max}\}`
   全部能构建、能编成 ``.nlf``（``allow_nan=False``）、数组与 meta 全有限；
2. **掩码与真值一致**：对数轴地板（``error_at_floor``）、"这个模态初值里没有"
   （``decay_alive``）、"解被截断显示"（``u_clipped``）三个掩码必须逐点对得上原始数组，
   否则图上会出现"贴在地板上的一条线"而账上没人认领；
3. **meta 与自己画的图一致**：``overflow``、``critical``、``conservation_holds``、
   ``measured_order`` 这些数字必须能从帧里的数组复算出来；
4. **可复现**：同一组参数两次构建逐位相同（帧是纯函数，缓存不改变结果）。

数值都是"实测 vs 帧里自己写的数"，不是把算法再算一遍——算法已经在上面的文件里对拍过了。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.verify import assertion, record  # noqa: E402
from server import nlf as nlf_mod  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.registry import Registry  # noqa: E402

CHAPTER = "4.1"
FLOOR = 1.0e-17
CLIP = 1.0e3


def _entry():
    reg = Registry(ROOT)
    reg.discover(strict=False)  # 并发开发时兄弟章节可能是半成品；本章必须能加载
    return reg.get(CHAPTER)


def _defaults(spec) -> dict:
    return {p.name: p.default for p in spec.params}


def _corners(spec) -> list[dict]:
    """默认 + 每个参数的 min/max（枚举取两端）——与全仓契约测试同口径。"""
    base = _defaults(spec)
    out = [dict(base)]
    for p in spec.params:
        if p.type == "enum" and len(p.options) >= 2:
            out.append({**base, p.name: p.options[0]})
            out.append({**base, p.name: p.options[-1]})
        elif p.type in ("int", "float") and p.min is not None and p.max is not None:
            out.append({**base, p.name: p.min})
            out.append({**base, p.name: p.max})
        elif p.type == "bool":
            out.append({**base, p.name: not base[p.name]})
    return out


def _times(spec) -> list[float]:
    if spec.time.mode == "none":
        return [0.0]
    t_max = float(spec.time.t_max or 0.0)
    return [0.0, t_max / 2.0, t_max] if t_max > 0 else [0.0]


@assertion("B", "参数角点上所有数组与 meta 都有限、能编成 .nlf，且掩码逐点自洽", chapter=CHAPTER)
def test_every_frame_corner_stays_finite_and_packs() -> None:
    r"""帧里**不许出现 NaN/Inf**：``.nlf`` 头用 ``allow_nan=False`` 编码，塞一个 NaN 整帧就写不出去。

    除了有限性，还检查三件事：

    * ``requiredAux`` 里的每个名字（``error``/``residual``）在本帧的数组名里能按子串找到
      ——这是 manifest 的硬约定，缺了前端就拿不到副图数据；
    * 对数轴地板的掩码 ``error_at_floor`` 与 ``error_curve`` 一致（地板值 ↔ 掩码 1）；
    * ``decay_alive`` 的 0 个数与 ``meta.n_alive`` 对得上（viz 3 的模态掩码）。
    """
    entry = _entry()
    problems: list[str] = []
    checked = 0
    for spec in entry.manifest.vizzes:
        aux = [str(a).lower() for a in (spec.required_aux or [])]
        for params in _corners(spec):
            for t in _times(spec):
                checked += 1
                label = f"{spec.id} {params} t={t:g}"
                try:
                    frame = build_frame(entry, spec.id, params, t, strict=True)
                    nlf_mod.pack_frame(frame)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{label}: {type(exc).__name__}: {exc}")
                    continue
                names = [s.name for s in frame.specs]
                for key in aux:
                    if not any(key in n for n in names):
                        problems.append(f"{label}: requiredAux {key!r} 在数组名里找不到（{names}）")
                for arr_spec, arr in nlf_mod.iter_arrays(frame):
                    bad = int(np.count_nonzero(~np.isfinite(arr))) if arr.size else 0
                    if bad:
                        problems.append(f"{label}: 数组 {arr_spec.name} 有 {bad}/{arr.size} 个非有限值")
                for key, val in frame.meta.items():
                    if isinstance(val, float) and not np.isfinite(val):
                        problems.append(f"{label}: meta[{key!r}] = {val}")
                if "error_at_floor" in frame and "error_curve" in frame:
                    floor_mask = frame["error_at_floor"] > 0.5
                    at_floor = frame["error_curve"] <= FLOOR * (1.0 + 1e-12)
                    n_mismatch = int(np.count_nonzero(floor_mask != at_floor))
                    if n_mismatch:
                        problems.append(f"{label}: error_at_floor 与 error_curve 有 {n_mismatch} 点不一致")
                if "decay_alive" in frame:
                    n_alive_mask = int(np.count_nonzero(frame["decay_alive"] > 0.5))
                    if n_alive_mask != int(frame.meta.get("n_alive", -1)):
                        problems.append(
                            f"{label}: decay_alive 有 {n_alive_mask} 个 1，meta.n_alive={frame.meta.get('n_alive')}"
                        )
                    n_mask = int(np.count_nonzero(frame["decay_alive"] <= 0.5))
                    if n_mask != int(frame.meta["n_absent"]) + int(frame.meta["n_masked_by_roundoff"]):
                        problems.append(f"{label}: 掩码数 {n_mask} 与 meta 的两种原因之和不一致")
                    masked = frame["decay_alive"] <= 0.5
                    if np.any(frame["decay_measured"][masked] > FLOOR * (1.0 + 1e-12)):
                        problems.append(f"{label}: 被掩码的模态没有显示为地板值")
    if checked < 40:
        problems.append(f"采样点只有 {checked} 个：测试可能没真的跑起来（预期 ≥ 40）")
    rec = record(
        label="帧角点：有限性 + requiredAux + 掩码自洽",
        theory="数组与 meta 全有限、能编 .nlf；requiredAux 名字命中；地板/模态掩码与数据一致",
        measured=f"{checked} 帧、{len(problems)} 条问题",
        tol="0 条问题",
        passed=not problems,
        detail="；".join(problems[:8]) or "全部通过",
        kind="B",
    )
    assert rec.passed, f"{len(problems)} 条问题：\n  - " + "\n  - ".join(problems[:12])


@assertion("B", "critical 标记、overflow 与显示截断掩码都跟着参数走", chapter=CHAPTER)
def test_critical_marker_and_overflow_follow_the_parameters() -> None:
    r"""``r`` 越过 1/2 时 ``meta.critical`` 必须给出 ``{'r': 0.5}``；越界到显示上限时
    ``meta.overflow`` 与 ``u_clipped`` 掩码必须同时置位。

    实测（``nx=101``、``r=0.8``、噪声初值、``t=0.01``）：解被顶到显示上限之上，
    ``overflow=True``、``u_clipped`` 有 $101$ 个点；而 ``r=0.4`` 时两者都是空/假。
    """
    entry = _entry()
    spec = next(s for s in entry.manifest.vizzes if s.id == "stability_threshold")
    params = _defaults(spec)
    t_max = float(spec.time.t_max)
    over = build_frame(entry, "stability_threshold", {**params, "r": 0.8, "initial": "noise", "nx": 101}, t_max)
    under = build_frame(entry, "stability_threshold", {**params, "r": 0.4, "initial": "noise", "nx": 101}, t_max)
    implicit = build_frame(
        entry, "stability_threshold", {**params, "r": 0.6, "scheme": "btcs", "initial": "noise"}, t_max
    )

    n_clipped = int(np.count_nonzero(over["u_clipped"] > 0.5))
    clip_consistent = bool(np.all(over["u"][over["u_clipped"] > 0.5] == CLIP))
    checks = {
        "r=0.8 越界时 critical 应为 {'r': 0.5}": over.critical == {"r": 0.5},
        "r=0.4 未越界时 critical 指向下一条线": under.critical == {"r": 0.5},
        "r=0.8 越界时 overflow 应为真": bool(over.meta["overflow"]) is True,
        "r=0.4 未越界时 overflow 应为假": bool(under.meta["overflow"]) is False,
        "u_clipped 掩码位置的值应等于显示上限": clip_consistent,
        "掩码点数应与 meta.clipped_points 一致": n_clipped == int(over.meta["clipped_points"]),
        "未越界时不应有截断点": int(np.count_nonzero(under["u_clipped"] > 0.5)) == 0,
        "BTCS 在 r=0.6 不该越界（无条件稳定）": bool(implicit.meta["overflow"]) is False,
    }
    failed = [k for k, v in checks.items() if not v]
    rec = record(
        label="critical / overflow / 截断掩码的一致性",
        theory="critical 报「当前位置最该被提及」的那条线（越界报已越过的最大的、否则报最小的）；overflow ⟺ umax>1e3",
        measured=f"越界帧截断 {n_clipped}/101 点、umax={over.meta['umax']:.3e}；"
        f"未越界帧 umax={under.meta['umax']:.3e}、critical={under.critical}；"
        f"BTCS r=0.6 umax={implicit.meta['umax']:.3e}",
        tol="8 项全真",
        passed=not failed,
        detail="；".join(failed) or "全部通过（显式越界即截断、隐式无条件稳定）",
        kind="B",
    )
    assert rec.passed, failed


@assertion("B", "t=0 的帧就是初值：解、总热量漂移、模态存活率都回到定义值", chapter=CHAPTER)
def test_t0_frames_are_the_initial_state() -> None:
    r"""时间轴从 0 开始时，帧必须给出**初值本身**（而不是"推进了一步"）。

    这里踩过一个真实的坑：``heat_solve(t_end=0)`` 会按 ``max(1, ceil(0/dt)))`` 走**一步**，
    时间轴上 ``t=0`` 的帧于是画的是 ``t=dt`` 的解。修法是显式传 ``n_steps=0``。
    实测：viz 1 在 ``t=0`` 的解与 ``sin(πx)`` 逐点相同（差 $0.0$）；
    viz 4 的三条总热量曲线在 ``t=0`` 全为 0（差 0.0）；
    viz 3 的存活率在 ``t=0`` 全为 $1$（这是我的数值解除以自己）。
    """
    entry = _entry()
    problems: list[str] = []
    x = np.linspace(0.0, 1.0, 41)
    v1 = build_frame(entry, "stability_threshold", {"scheme": "cn", "r": 0.4, "nx": 41, "bc": "dirichlet", "initial": "sine"}, 0.0)
    d_sine = float(np.max(np.abs(v1["u"] - np.sin(np.pi * x))))
    if d_sine > 0.0:
        problems.append(f"viz1 t=0 的解与 sin(πx) 差 {d_sine:g}")
    if float(v1.meta["t_actual"]) != 0.0 or int(v1.meta["n_steps"]) != 0:
        problems.append(f"viz1 t=0 的 t_actual={v1.meta['t_actual']}、n_steps={v1.meta['n_steps']}")

    v4 = build_frame(entry, "boundary_conservation", {"bc": "neumann", "scheme": "cn", "r": 0.4, "nx": 41}, 0.0)
    for name in ("mass_shift_dirichlet", "mass_shift_neumann", "mass_shift_periodic"):
        peak = float(np.max(np.abs(v4[name])))
        if peak != 0.0:
            problems.append(f"viz4 t=0 的 {name} 峰值 {peak:g}（应为 0）")
    d_exact = float(np.max(np.abs(v4["u"] - v4["u_exact"])))
    if d_exact != 0.0:
        problems.append(f"viz4 t=0 的解与解析解差 {d_exact:g}")

    v3 = build_frame(entry, "numerical_dissipation", {"scheme": "cn", "r": 0.4, "nx": 41, "n_modes": 8}, 0.0)
    alive = v3["decay_alive"] > 0.5
    if not np.allclose(v3["decay_measured"][alive], 1.0, rtol=0, atol=1e-12):
        problems.append("viz3 t=0 的存活率不为 1（数值解在 t=0 应等于初值）")
    if int(np.count_nonzero(alive)) != 8:
        problems.append(f"viz3 t=0 存活模态 {int(np.count_nonzero(alive))} 个（初值有 8 个模态）")

    rec = record(
        label="t=0 的帧 = 初值（viz1/viz3/viz4）",
        theory="t=0 时 u=u⁰、ΔH=0、|a_k(t)/a_k(0)|=1",
        measured=f"viz1 差 {d_sine:g}、viz4 漂移峰值 0、viz4 与解析解差 {d_exact:g}",
        tol="精确 0（n_steps=0）",
        passed=not problems,
        detail="；".join(problems) or "三个可视化的 t=0 帧都精确等于初值",
        kind="B",
    )
    assert rec.passed, problems


@assertion("B", "viz3 的实测存活率与 |G|^n 在帧内逐点重合（模态分解没算错）", chapter=CHAPTER)
def test_dissipation_identity_holds_inside_the_frame() -> None:
    r"""帧里同时画了"从真的解里分解出来的存活率"与"解析的 $|G|^n$"。

    两条线必须重合到舍入级——``meta.max_amplitude_residual`` 就是这个差的最大值。
    实测（$nx=41$、$r=0.4$、$t=0.05$、8 个模态）$\le 9\times10^{-15}$：
    说明模态分解（正弦族的内积）与放大因子闭式这两条独立路线在帧里对上了。
    如果只画解析曲线、不画实测曲线，这个检查就不存在了——这正是这条断言的意义。
    """
    entry = _entry()
    problems: list[str] = []
    worst = 0.0
    for scheme in ("ftcs", "btcs", "cn"):
        for r in (0.2, 0.4, 0.8):
            frame = build_frame(
                entry, "numerical_dissipation", {"scheme": scheme, "r": r, "nx": 41, "n_modes": 8}, 0.05
            )
            resid = float(np.max(np.abs(frame["amplitude_residual"])))
            worst = max(worst, resid)
            if resid > 1e-10:
                problems.append(f"{scheme} r={r}: max|实测−|G|^n| = {resid:.3e}")
            if float(frame.meta["max_amplitude_residual"]) != resid:
                problems.append(f"{scheme} r={r}: meta.max_amplitude_residual 与数组不一致")
            alive = frame["decay_alive"] > 0.5
            if not np.all(frame["decay_measured"][alive] > 0.0):
                problems.append(f"{scheme} r={r}: 存活模态里出现了非正的存活率")
    rec = record(
        label="viz3：实测存活率 vs |G|^n（帧内自洽）",
        theory="0（两条独立路线：正弦内积分解 vs 放大因子闭式）",
        measured=f"max ≤ {worst:.3e}（3 格式 × 3 个 r）",
        tol="1e-10",
        passed=not problems,
        detail="；".join(problems) or "9 帧全部重合到舍入级",
        kind="B",
    )
    assert rec.passed, problems


@assertion("B", "viz4 的守恒结论与数组一致：Neumann/周期恒为 0，Dirichlet 与解析流失量一致", chapter=CHAPTER)
def test_boundary_frame_conservation_matches_the_arrays() -> None:
    r"""帧里的 ``conservation_holds`` 是个布尔结论，必须能从数组复算出来。

    实测（$nx=41$、$r=0.4$、$t=0.05$）：Neumann 与周期的末刻漂移
    $3.2\times10^{-14}$ / $3.4\times10^{-14}$（舍入级），Dirichlet 掉 $0.247739$
    而解析值 $(2/\pi)(1-e^{-\pi^2t})=0.247975$，比值 $0.9991$。
    """
    entry = _entry()
    problems: list[str] = []
    detail: list[str] = []
    for bc in ("dirichlet", "neumann", "periodic"):
        frame = build_frame(entry, "boundary_conservation", {"bc": bc, "scheme": "cn", "r": 0.4, "nx": 41}, 0.05)
        drift = abs(float(frame.meta["drift_now"]))
        holds = bool(frame.meta["conservation_holds"])
        if bc == "dirichlet":
            if holds:
                problems.append("Dirichlet 不该报 conservation_holds=True")
            if not (0.2 < drift < 0.3):
                problems.append(f"Dirichlet 的漂移 {drift:.6f} 不在 (0.2,0.3)")
            ratio = float(frame.meta["leak_ratio"])
            if abs(ratio - 1.0) > 2e-2:
                problems.append(f"Dirichlet 的流失量/解析值 = {ratio:.4f}（应≈1）")
            detail.append(f"Dirichlet 漂移 {drift:.6f}（解析比 {ratio:.4f}）")
        else:
            if not holds:
                problems.append(f"{bc} 应报 conservation_holds=True")
            if drift > 1e-12:
                problems.append(f"{bc} 的漂移 {drift:.3e} 超过舍入级")
            shift = frame[f"mass_shift_{bc}"]
            if float(np.max(np.abs(shift))) > 1e-12:
                problems.append(f"{bc} 的历史曲线峰值 {float(np.max(np.abs(shift))):.3e} 超过舍入级")
            detail.append(f"{bc} 漂移 {drift:.2e}")
        if bc != "dirichlet" and float(frame.meta["drift_neumann_max"]) > 1e-12:
            problems.append(f"{bc} 帧的 Neumann 历史峰值 {frame.meta['drift_neumann_max']:.3e} 超过舍入级")
    rec = record(
        label="viz4：守恒标志与数组复算一致（三种边界）",
        theory="Neumann/周期 |ΔH| ≤ 1e-12（结构性守恒）；Dirichlet 流失量 = (2/π)(1−e^(−π²t))",
        measured="；".join(detail),
        tol="1e-12 / 比值 2e-2",
        passed=not problems,
        detail="；".join(problems) or "三种边界的标志、数组与解析值三方一致",
        kind="B",
    )
    assert rec.passed, problems


@assertion("B", "viz2 的 measured_order 与理论阶一致；换靶子的对照斜率也如实写进 meta", chapter=CHAPTER)
def test_order_frame_meta_matches_the_theory() -> None:
    r"""``meta.measured_order`` 是帧自己拟合出来的斜率，必须等于格式的理论阶。

    实测：``target=space`` 时三个格式都是 $2.000$（固定 $r$，时间误差被压成同阶）；
    ``target=time`` 时 FTCS/BTCS $1.000$、CN $2.000$；
    而 ``meta.order_vs_continuous``（同一批数据换连续解当靶子）在时间细化时
    CN 是 $-0.0004$、BTCS 是 $0.4615$——这个"错答案"是故意留在帧里的教学内容，
    所以这里只检查它确实**不等于**理论阶（而不是检查它接近理论阶）。
    """
    entry = _entry()
    problems: list[str] = []
    detail: list[str] = []
    for target in ("space", "time"):
        for scheme in ("ftcs", "btcs", "cn"):
            frame = build_frame(
                entry, "convergence_order", {"scheme": scheme, "target": target, "nx": 41, "levels": 4, "t_end": 0.05}, 0.0
            )
            measured = float(frame.meta["measured_order"])
            theory = float(frame.meta["theory_order"])
            cont = float(frame.meta["order_vs_continuous"])
            gap = abs(measured - theory)
            if gap > 0.1:
                problems.append(f"{target}/{scheme}: 实测阶 {measured:.4f} 偏离理论 {theory:g} 超过 0.1")
            if float(frame.meta["order_gap"]) != gap:
                problems.append(f"{target}/{scheme}: meta.order_gap 与 measured/theory 不一致")
            if target == "time":
                if abs(cont - theory) <= 0.1:
                    problems.append(f"time/{scheme}: 对连续解的斜率 {cont:.4f} 不该等于理论阶 {theory:g}")
                if gap > 0.1:
                    problems.append(f"time/{scheme}: 半离散靶子的阶 {measured:.4f} 不对")
            elif cont < 1.5:
                problems.append(f"space/{scheme}: 对连续解的斜率 {cont:.4f} 应≈2")
            detail.append(f"{target}/{scheme}: 半离散靶 {measured:.4f}（理论 {theory:g}）、连续靶 {cont:.4f}")
    ladder = build_frame(
        entry, "convergence_order", {"scheme": "cn", "target": "space", "nx": 41, "levels": 4, "t_end": 0.05}, 0.0
    )
    h = ladder["h_ladder"]
    if not np.all(np.diff(h) < 0):
        problems.append("h_ladder 不是严格递减：fit_order 会拒绝")
    if not np.all(np.diff(ladder["error_ladder"]) < 0):
        problems.append("error_ladder 不是随 h 减小而减小")
    rec = record(
        label="viz2：实测阶 vs 理论阶（含换靶子对照）",
        theory="space 全部 2；time：ftcs/btcs 1、cn 2；连续解靶子在 time 下给不出时间阶",
        measured="；".join(detail),
        tol="±0.1",
        passed=not problems,
        detail="；".join(problems) or "6 条阶梯的斜率与理论阶一致，对照斜率如实为 0 附近",
        kind="B",
    )
    assert rec.passed, problems


@assertion("B", "帧是纯函数：同一组参数两次构建逐位相同，缓存不改变结果", chapter=CHAPTER)
def test_frames_are_pure_functions_of_their_parameters() -> None:
    r"""``numcore`` 不许有全局可变状态（P6），帧层也一样：两次构建必须逐位相同。

    这条测试用来钉住"缓存"这类看起来无害的优化：``server/chapters/4.1.py`` 里
    对推进与幂迭代都加了 ``lru_cache``，如果哪天缓存键漏了一个参数
    （例如参考解的 ``r/16`` 缩放没进键），第二个帧就会被第一个污染。
    实测：4 个可视化 × 各自默认参数，两次构建的最大差 $0.0$。
    """
    entry = _entry()
    problems: list[str] = []
    worst = 0.0
    for spec in entry.manifest.vizzes:
        params = _defaults(spec)
        t = 0.0 if spec.time.mode == "none" else float(spec.time.t_max) / 2.0
        a = build_frame(entry, spec.id, params, t)
        b = build_frame(entry, spec.id, params, t)
        for name in a.names:
            d = float(np.max(np.abs(a[name] - b[name]))) if a[name].size else 0.0
            worst = max(worst, d)
            if d != 0.0:
                problems.append(f"{spec.id}.{name}: 两次构建差 {d:g}")
        if dict(a.meta) != dict(b.meta):
            diff = [k for k in a.meta if a.meta[k] != b.meta[k]]
            problems.append(f"{spec.id}: meta 不同（{diff}）")
    rec = record(
        label="帧的纯函数性（两次构建逐位相同）",
        theory="0.0（无全局状态；缓存键必须覆盖所有影响结果的参数）",
        measured=f"max|Δ| = {worst:g}（4 个可视化）",
        tol="精确 0",
        passed=not problems,
        detail="；".join(problems) or "4 个可视化的数组与 meta 都逐位可复现",
        kind="B",
    )
    assert rec.passed, problems
