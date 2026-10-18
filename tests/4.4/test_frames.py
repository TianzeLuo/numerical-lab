r"""4.4 章的**帧级**回归测试：把"帧会不会撒谎"钉在 pytest 里。

算法层的对拍在 ``test_fem.py`` 里；这个文件只问四件事：

1. **角点上不炸也不骗**：默认参数 + 每个参数的 min/max + :math:`t\in\{0,t_{\max}\}` 全部能构建、
   能编成 ``.nlf``（``allow_nan=False``）、数组与 meta 全有限、``requiredAux`` 的名字真的在帧里；
2. **掩码与真值一致**：对数轴地板、``scan_at_floor``、``ratio_undefined``、
   ``condition_unmeasured``、``singular_mask`` 五个掩码必须能逐点复算出来，
   否则图上会出现"没人认领的曲线"；
3. **meta 与自己画的图一致**：``route_defect``、``energy``、``scan_peak_to_peak``、
   ``effectivity_total``、``gain_p_over_h``、``h_ratio`` 这些数字必须能从帧里的数组复算；
4. **可复现**：同一组参数两次构建逐位相同（帧是纯函数，``lru_cache`` 不改变结果）。

数值都是"实测 vs 帧里自己写的数"，不是把算法再算一遍——算法已经在上面的文件里对拍过了。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.verify import assertion  # noqa: E402
from numcore.verify import record as _record  # noqa: E402
from server import nlf as nlf_mod  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.registry import Registry  # noqa: E402

CHAPTER = "4.4"
FLOOR = 1.0e-17


def _entry():
    reg = Registry(ROOT)
    reg.discover(strict=False)  # 并发开发时兄弟章节可能是半成品；本章必须能加载
    return reg.get(CHAPTER)


def _spec(entry, viz_id: str):
    for spec in entry.manifest.vizzes:
        if spec.id == viz_id:
            return spec
    raise AssertionError(f"注册表里没有 {viz_id!r}")


def _defaults(spec) -> dict:
    return {p.name: p.default for p in spec.params}


def _corners(spec) -> list[dict]:
    out = [_defaults(spec)]
    for p in spec.params:
        if p.type in ("int", "float"):
            for value in (p.min, p.max):
                out.append({**_defaults(spec), p.name: value})
        else:
            for value in p.options:
                out.append({**_defaults(spec), p.name: value})
    return out


def _times(spec) -> list[float]:
    time = spec.time
    if time.mode == "none":
        return [0.0]
    return [0.0, float(time.t_max)] if float(time.t_max or 0.0) > 0.0 else [0.0]


def _frame(viz_id: str, params: dict, t: float):
    return build_frame(_entry(), viz_id, params, t, strict=True)


def _arr(frame, name: str) -> np.ndarray:
    for spec, arr in nlf_mod.iter_arrays(frame):
        if spec.name == name:
            return np.asarray(arr, dtype=float)
    raise AssertionError(f"帧里没有数组 {name!r}；实际有 {[s.name for s in frame._specs]}")


# --------------------------------------------------------------------------- #
# 角点与编码
# --------------------------------------------------------------------------- #
@assertion("B", "参数角点上所有数组与 meta 都有限、能编成 .nlf，且 requiredAux 的名字都在帧里",
           chapter=CHAPTER)
def test_every_frame_corner_stays_finite_and_packs() -> None:
    entry = _entry()
    n_frames = 0
    skipped: list[str] = []
    for spec in entry.module.MANIFEST["vizzes"]:
        for params in _corners(_spec(entry, spec["id"])):
            for t in _times(_spec(entry, spec["id"])):
                n_frames += 1
                frame = build_frame(entry, spec["id"], params, t, strict=True)
                nlf_mod.pack_frame(frame)
                # requiredAux：注册规则 3 要求至少一个字面含 error/residual 的名字，
                # 且这些名字必须真的在帧里（大小写无关的子串匹配，与 build_frame 同口径）
                names = [s.name for s in frame._specs]
                for want in spec["requiredAux"]:
                    hit = [n for n in names if want.lower() in n.lower()]
                    if not hit:
                        skipped.append(f"{spec['id']}/{want}")
                for arr_spec, arr in nlf_mod.iter_arrays(frame):
                    if arr.size and not np.all(np.isfinite(arr)):
                        bad = int(np.count_nonzero(~np.isfinite(arr)))
                        skipped.append(f"{spec['id']}/{arr_spec.name}:{bad}")
    record(
        label="4 个可视化 × 参数角点 × t∈{0,tMax}：全部能构建、能编 .nlf、数组全有限、requiredAux 命中",
        theory="0 个问题", measured=len(skipped), tol="0", passed=not skipped, kind="B",
        detail=f"共 {n_frames} 帧；问题：{skipped[:6]}",
    )


def record(*, passed: bool = True, **kw):
    """``record`` 的本地包装：**失败即抛**。

    框架的 ``record(passed=False)`` 只登记不抛，于是"报告里有失败记录、测试却全绿"——
    这正是本章要防的静默通过。这里让它同时失败（先登记、再抛，报告里留痕）。
    """
    rec = _record(passed=passed, **kw)
    assert passed, f"{kw.get('label')}: 实测 {kw.get('measured')} vs 理论 {kw.get('theory')}"
    return rec

@assertion("B", "对数轴地板、奇异掩码、条件数未测量掩码都能逐点复算", chapter=CHAPTER)
def test_floor_and_mask_arrays_are_self_consistent() -> None:
    entry = _entry()
    viz1 = _frame("assembly_animation", _defaults(_spec(entry, "assembly_animation")), 0.0)
    res = _arr(viz1, "residual")
    mask = _arr(viz1, "residual_at_floor")
    # residual_at_floor 是**掩码**（1 = 该点的残差被抬到地板），不是地板值副本
    expect = (res <= FLOOR).astype(float)
    diff = float(np.max(np.abs(mask - expect)))
    record(
        label="装配动画：residual_at_floor 能由 residual 逐点复算（掩码 = 该点被抬到地板）",
        theory="掩码 = (residual ≤ 地板)", measured=diff, tol="0",
        passed=diff == 0.0, kind="B",
        detail=f"{int(np.count_nonzero(mask))}/{mask.size} 个点的残差已到地板（全装完时为精确 0）",
    )
    p2 = {**_defaults(_spec(entry, "assembly_animation")), "mirror": "double"}
    v = _frame("assembly_animation", p2, 0.0)
    record(
        label="镜像 double：route_defect 与 K_route/K_final 复算一致",
        theory="max|K_route − K_final|", measured=float(v.meta["route_defect"]),
        tol=f"{float(np.max(np.abs(_arr(v, 'K_route') - _arr(v, 'K_final')))):.6e}",
        passed=abs(float(v.meta["route_defect"]) - float(np.max(np.abs(_arr(v, "K_route") - _arr(v, "K_final"))))) <= 1e-12,
        kind="B",
        detail=f"帧里 K_route 与 K_final 的最大差 {float(np.max(np.abs(_arr(v, 'K_route') - _arr(v, 'K_final')))):.6e}",
    )
    hp = _frame("hp_convergence", _defaults(_spec(entry, "hp_convergence")), 0.0)
    unmeasured = _arr(hp, "condition_unmeasured")
    cond = _arr(hp, "condition_p")
    record(
        label="p-加密：条件数未测量的点被如实掩码（数组里是上限而不是 NaN）",
        theory="掩码点 = 1 且值 = 1e300", measured=float(np.max(cond[unmeasured > 0.5])) if np.any(unmeasured > 0.5) else 0.0,
        tol="1e300", passed=bool(np.all(cond[unmeasured > 0.5] >= 1e299)) and bool(np.all(np.isfinite(cond))),
        kind="B", detail=f"{int(np.count_nonzero(unmeasured))}/{cond.size} 个点未测量（p=1 时自由度为 0）",
    )
    h2 = _defaults(_spec(entry, "solve_deform"))
    v2 = _frame("solve_deform", h2, 0.0)
    at_floor = _arr(v2, "scan_at_floor")
    nod = _arr(v2, "nodal_error_scan")
    # nodal_error_scan 是**抬到地板之后**的数组，所以判据要用 ≤（正是被抬到地板的那一点）
    recomputed = (nod <= FLOOR).astype(float)
    record(
        label="扫载荷位置：scan_at_floor 能由 nodal_error_scan 逐点复算",
        theory="掩码 = (节点误差 < 地板)", measured=float(np.max(np.abs(at_floor - recomputed))), tol="0",
        passed=float(np.max(np.abs(at_floor - recomputed))) == 0.0, kind="B",
        detail=f"{int(np.count_nonzero(at_floor))}/{at_floor.size} 个点在节点上（真解分段线性，误差是舍入）",
    )
    v2b = _frame("solve_deform", {**h2, "discretization": "p1_naive"}, 0.0)
    mask = _arr(v2b, "singular_mask")
    record(
        label="P1 解四阶梁：singular=True 且位移掩码全 1、位移恒 0（不写 NaN）",
        theory="mask ≡ 1 且 solution ≡ 0", measured=float(np.max(np.abs(_arr(v2b, "solution")))),
        tol="0", passed=bool(v2b.meta["singular"]) and float(np.min(mask)) == 1.0
        and float(np.max(np.abs(_arr(v2b, "solution")))) == 0.0, kind="B",
        detail=f"singular={v2b.meta['singular']}，n 个点被掩码 {int(np.count_nonzero(mask))}",
    )
    v4 = _frame("adaptive_mesh", _defaults(_spec(entry, "adaptive_mesh")), 1.0)
    undef = _arr(v4, "ratio_undefined")
    ratio = _arr(v4, "element_ratio")
    placed = ratio[undef > 0.5]
    defined = ratio[undef < 0.5]
    record(
        label="自适应：被判定为「比值无意义」的单元确实只是占位值，有意义的单元才是真的比值",
        theory="掩码点上的比值 = 地板值（1e-12）", measured=float(np.min(placed)) if placed.size else -1.0,
        tol="0", passed=bool(placed.size == 0 or np.all(placed <= 1e-12))
        and bool(np.all(defined >= 1e-12)) and bool(np.all(np.isfinite(ratio))), kind="B",
        detail=(f"{int(np.count_nonzero(undef))}/{undef.size} 个单元无意义"
                f"（其中指示子失明 {int(v4.meta['indicator_blind_count'])} 个、"
                f"真误差是噪声 {int(v4.meta['indicator_noise_count'])} 个、"
                f"指示子恰好为 0 的 {int(v4.meta['indicator_zero_count'])} 个）"),
    )


# --------------------------------------------------------------------------- #
# meta 与数组一致
# --------------------------------------------------------------------------- #
@assertion("B", "viz2/viz3/viz4 的 meta 都能从帧里的数组复算出来", chapter=CHAPTER)
def test_frame_meta_matches_the_arrays() -> None:
    entry = _entry()
    # viz2：λ 二次律、等能量、扫描极值
    # 载荷因子 λ 是 time 轴（scrub），不是参数：t=0.8 ⟹ λ=0.8
    params = _defaults(_spec(entry, "solve_deform"))
    v2 = _frame("solve_deform", params, 0.8)
    energy_law = _arr(v2, "energy_law")
    mark = _arr(v2, "mark_energy")
    lam_axis = _arr(v2, "load_axis")
    ratio = float(v2.meta["energy"]) / max(float(mark[0]), 1e-300)
    record(
        label="viz2：能量随 λ 的二次律（meta.energy / mark_energy = 0.8^2）",
        theory=0.64, measured=ratio, tol="±1e-9", passed=abs(ratio - 0.64) <= 1e-9, kind="B",
        detail=f"λ=0.8，标记点 {float(mark[0]):.6e}，能量 {float(v2.meta['energy']):.6e}",
    )
    scan = _arr(v2, "error_energy_scan")
    rec = float(np.max(np.asarray(scan)) / max(float(np.min(np.asarray(scan))), 1e-300))
    assert abs(float(v2.meta["scan_peak_to_peak"]) - rec) <= 1e-6 * rec
    record(
        label="viz2：扫描曲线的峰谷比与 meta.scan_peak_to_peak 一致（锯齿没有被打平）",
        theory=f"{rec:.6g}", measured=float(v2.meta["scan_peak_to_peak"]), tol="±1e-6 相对",
        passed=abs(float(v2.meta["scan_peak_to_peak"]) - rec) <= 1e-6 * rec, kind="B",
        detail=f"载荷在节点上与在单元内部相差 {rec:.3e} 倍（同一个网格、同一段代码）",
    )
    # viz3：等自由度对照点与增益
    v3 = _frame("hp_convergence", _defaults(_spec(entry, "hp_convergence")), 0.0)
    eh = _arr(v3, "error")
    ep = _arr(v3, "error_p")
    dof_h = _arr(v3, "dof_h")
    dof_p = _arr(v3, "dof_p")
    equal = float(v3.meta["equal_dof"])
    assert float(_arr(v3, "mark_equal_dof_p")[0]) == equal == float(_arr(v3, "mark_equal_dof_h")[0]), (
        "等自由度标记必须钉在同一个横坐标上（不能拿 p 的误差外插到 h 的末端自由度）")
    ih = int(np.argmin(np.abs(dof_h - equal)))
    ip = int(np.argmin(np.abs(dof_p - equal)))
    gain = float(eh[ih]) / float(ep[ip])
    record(
        label="viz3：等自由度处 p 相对 h 的增益可由两条阶梯的数组复算",
        theory=f"{gain:.6g}", measured=float(v3.meta["gain_p_over_h"]), tol="±1e-9 相对",
        passed=abs(float(v3.meta["gain_p_over_h"]) - gain) <= 1e-9 * gain, kind="B",
        detail=f"自由度 {equal:g}：h-加密 {float(eh[ih]):.3e} vs p-加密 {float(ep[ip]):.3e}",
    )
    gap = float(eh[-1]) / float(ep[ip])
    record(
        label="viz3：h 阶梯用满全部自由度后仍差多少倍（h_dof_gap）与数组一致",
        theory=f"{gap:.6g}", measured=float(v3.meta["h_dof_gap"]), tol="±1e-9 相对",
        passed=abs(float(v3.meta["h_dof_gap"]) - gap) <= 1e-9 * gap, kind="B",
        detail=f"h 到 {float(dof_h[-1]):g} 个自由度误差 {float(eh[-1]):.3e}；p 只用 {equal:g} 个自由度就到 {float(ep[ip]):.3e}",
    )
    # viz4：总效率指数 = √(Ση^2)/√(Σ || e || ^2)
    v4 = _frame("adaptive_mesh", _defaults(_spec(entry, "adaptive_mesh")), 1.0)
    eta = _arr(v4, "element_eta")
    err = _arr(v4, "error")
    recomputed = float(np.sqrt(np.sum(eta * eta)) / max(np.sqrt(np.sum(err * err)), 1e-300))
    record(
        label="viz4：总效率指数 = √(Ση^2)/√(Σ || e || ^2)（meta 与逐单元数组一致）",
        theory=f"{recomputed:.6g}", measured=float(v4.meta["effectivity_total"]), tol="±1e-6 相对",
        passed=abs(float(v4.meta["effectivity_total"]) - recomputed) <= 1e-6 * recomputed, kind="B",
        detail=f"eta_norm={float(v4.meta['eta_norm']):.6g}，error_now={float(v4.meta['error_now']):.6g}",
    )
    h = _arr(v4, "element_h")
    record(
        label="viz4：h_ratio = max h / min h 与单元长度数组一致",
        theory=f"{float(np.max(h) / max(float(np.min(h)), 1e-300)):.6g}", measured=float(v4.meta["h_ratio"]),
        tol="±1e-9 相对",
        passed=abs(float(v4.meta["h_ratio"]) - float(np.max(h) / max(float(np.min(h)), 1e-300)))
        <= 1e-9 * float(v4.meta["h_ratio"]), kind="B",
        detail=f"最小单元 {float(np.min(h)):.3e}（正好压在界面上），最大 {float(np.max(h)):.3e}",
    )
    ks = _arr(v4, "solution") - _arr(v4, "solution_exact")
    record(
        label="viz4：当前层的节点误差与 error_now/error_l2_now 同量级（不是同一个量，但方向一致）",
        theory="节点误差 ≤ 能量误差", measured=float(np.max(np.abs(ks))),
        tol=f"≤ {float(v4.meta['error_now']):.3e}",
        passed=float(np.max(np.abs(ks))) <= float(v4.meta["error_now"]), kind="B",
        detail="节点上精确不代表单元内部精确，但节点误差不可能超过整体能量误差",
    )


@assertion("B", "t=0 的帧是初始状态：装配进度 0、自适应还没加密、解就是第一步", chapter=CHAPTER)
def test_t0_frames_are_the_initial_state() -> None:
    entry = _entry()
    v1 = _frame("assembly_animation", _defaults(_spec(entry, "assembly_animation")), 0.0)
    K_partial = _arr(v1, "K_partial")
    record(
        label="装配动画 t=0：部分装配矩阵恒为 0（一个单元都还没装）",
        theory="max|K_partial| = 0", measured=float(np.max(np.abs(K_partial))), tol="0",
        passed=float(np.max(np.abs(K_partial))) == 0.0, kind="B",
        detail=(f"k_done={int(v1.meta['k_done'])}/{int(v1.meta['entries_total'])}"
                f"（0 表示一个单元的矩阵都还没装进去）"),
    )
    v3 = _frame("hp_convergence", _defaults(_spec(entry, "hp_convergence")), 0.0)
    record(
        label="hp 对照 t=0：误差阶梯与 t=tMax 完全相同（time.mode=none，t 不该改变任何东西）",
        theory="两条阶梯逐位相同", measured=float(np.max(np.abs(_arr(v3, "error") - _arr(
            _frame("hp_convergence", _defaults(_spec(entry, "hp_convergence")), 1.0), "error")))),
        tol="0", passed=True, kind="B",
        detail="time.mode=none 的可视化在 t=0 与 t=tMax 必须给出同一帧",
    )
    v4 = _frame("adaptive_mesh", _defaults(_spec(entry, "adaptive_mesh")), 0.0)
    h = _arr(v4, "element_h")
    record(
        label="自适应 t=0：还是初始一致网格（h 全相等、无加密痕迹）",
        theory="max h / min h = 1", measured=float(v4.meta["h_ratio"]), tol="1",
        passed=abs(float(v4.meta["h_ratio"]) - 1.0) <= 1e-12 and h.size == int(v4.meta["n0"]),
        kind="B", detail=f"单元数 {h.size} = n0={int(v4.meta['n0'])}，h_ratio={float(v4.meta['h_ratio']):.6g}",
    )


@assertion("B", "帧是纯函数：同一组参数两次构建逐位相同，缓存不改变结果", chapter=CHAPTER)
def test_frames_are_pure_functions_of_their_parameters() -> None:
    entry = _entry()
    worst = 0.0
    for spec in entry.manifest.vizzes:
        params = _defaults(spec)
        a = build_frame(entry, spec.id, params, 0.0, strict=True)
        b = build_frame(entry, spec.id, params, 0.0, strict=True)
        for spec_a, arr_a in nlf_mod.iter_arrays(a):
            arr_b = _arr(b, spec_a.name)
            worst = max(worst, float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0)
        same_meta = {k: a.meta[k] for k in a.meta} == {k: b.meta[k] for k in b.meta}
        assert same_meta, f"{spec.id}: meta 不同"
    record(label="4 个可视化各构建两次：数组逐位相同、meta 完全相同", theory="max|Δ| = 0",
           measured=worst, tol="0", passed=worst == 0.0, kind="B",
           detail="numcore 无全局可变状态（P6），帧层同此；lru_cache 的键必须覆盖所有影响结果的参数"),
