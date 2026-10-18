r"""4.5 章的**帧级**回归测试：把"这一页的图会不会撒谎"钉在 pytest 里。

算法层的对拍在 ``test_multigrid.py``；这个文件只问四件事：

1. **角点上不炸也不骗**：默认参数 + 每个参数的 min/max + 枚举两端 +
   :math:`t\in\{0,t_{\max}/2,t_{\max}\}` 全部能构建、能编成 ``.nlf``
   （``allow_nan=False``）、数组与 meta 全有限，``requiredAux`` 的名字命得中；
2. **掩码与真值逐点一致**：对数轴地板（``amplitude_at_floor`` / ``error_at_floor`` /
   ``event_at_floor`` / ``residual_at_floor``）与"占位值"（模态没被激发时写 1.0 的
   ``factor_measured``、因子 >=1 时写哨兵的 ``iters_*``）必须与原始数组对得上；
3. **meta 与自己画的图一致**：``measured_smoothing_factor``、``mg_factor_mean/std/drift``、
   ``loss_now``、``alias_theta_pi`` 这些数字都能从帧里的数组复算出来；
4. **可复现且时间轴老实**：同一组参数两次构建逐位相同；``t`` 超出 ``tMax`` 与取 ``tMax``
   等价（scrub 是钳制不是外推）；``time.mode="none"`` 的页面不许声明 ``tMax/dt/fps``。

数值都是"实测 vs 帧里自己写的数"，不是把算法再算一遍——算法已经在 ``test_multigrid.py`` 里对拍过了。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.linalg.iterative import sine_eigenvectors_1d  # noqa: E402
from numcore.pde import multigrid as mg  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402
from server import nlf as nlf_mod  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.registry import Registry  # noqa: E402

CHAPTER = "4.5"
FLOOR = 1.0e-16
PRESENT = 1.0e-13
ITERS_CAP = 1.0e6


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
    return [0.0, t_max / 2.0, t_max]


def _viz(entry, viz_id: str):
    return entry.manifest.viz(viz_id)


@assertion("B", "角点：数组与 meta 全有限、能编 .nlf、requiredAux 命中、四类掩码逐点自洽", chapter=CHAPTER)
def test_every_frame_corner_stays_finite_and_packs() -> None:
    r"""帧里不许出现 NaN/Inf：``.nlf`` 头用 ``allow_nan=False`` 编码，塞一个 NaN 整帧就写不出去。

    同时检查"非有限值改造"是否两头对齐：

    * 对数轴地板的三个掩码（振幅 / 逐层误差与事件 / 残差损失）必须与对应数据的
      "落在地板上"逐点一致；
    * viz1 的 ``factor_measured`` 在没有该模态时写 1.0（中性占位）——必须与
      ``mode_present=0`` 的位置**完全重合**，否则读者会把占位值当测量值；
    * viz3 的 ``iters_at_cap`` 必须恰好标出因子 >=1 的那些尺寸（次数写成哨兵 1e6）。
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
                if "amplitude_at_floor" in frame:
                    mask = frame["amplitude_at_floor"] > 0.5
                    at = (frame["amplitude_measured"] <= FLOOR) | (frame["amplitude_theory"] <= FLOOR)
                    if int(np.count_nonzero(mask != at)):
                        problems.append(f"{label}: amplitude_at_floor 与振幅数据不一致")
                if "mode_present" in frame:
                    present = frame["mode_present"] > 0.5
                    if int(np.count_nonzero(frame["amplitude_initial"] > PRESENT)) != int(np.count_nonzero(present)):
                        problems.append(f"{label}: mode_present 与 amplitude_initial 不一致")
                    placeholder = np.abs(frame["factor_measured"] - 1.0) <= 1e-15
                    if np.any(placeholder & present):
                        problems.append(f"{label}: 有模态被激发却写了占位因子 1.0")
                    if int(np.count_nonzero(present)) != int(frame.meta["n_modes_present"]):
                        problems.append(f"{label}: n_modes_present 与 mode_present 不一致")
                for abs_name, floor_name in (("error_abs", "error_at_floor"),
                                            ("event_norm", "event_at_floor"),
                                            ("residual_lost_abs", "residual_at_floor"),
                                            ("residual_norm_history", "residual_norm_at_floor")):
                    if abs_name in frame and floor_name in frame:
                        mask = frame[floor_name] > 0.5
                        at = frame[abs_name] <= FLOOR
                        if int(np.count_nonzero(mask != at)):
                            problems.append(f"{label}: {floor_name} 与 {abs_name} 有 "
                                            f"{int(np.count_nonzero(mask != at))} 点不一致")
                if "iters_at_cap" in frame:
                    mask = frame["iters_at_cap"] > 0.5
                    expect = (frame["iters_mg_measured"] >= ITERS_CAP) | (frame["iters_smoother"] >= ITERS_CAP)
                    if int(np.count_nonzero(mask != expect)):
                        problems.append(f"{label}: iters_at_cap 与迭代数哨兵不一致")
                if "profile_flat" in frame:
                    total = int(np.sum(frame["profile_size"]))
                    if total != int(frame["profile_flat"].size):
                        problems.append(f"{label}: profile_size 之和 {total} != profile_flat 长度")
                    offs = frame["profile_offset"]
                    if np.any(offs < 0) or np.any(offs > frame["profile_flat"].size):
                        problems.append(f"{label}: profile_offset 越界")
                    ph = frame["event_phase"]
                    if np.any(ph < 0) or np.any(ph > 6):
                        problems.append(f"{label}: event_phase 超出 0..6")
                    if int(frame["event_order"].size) != int(frame.meta["n_events"]):
                        problems.append(f"{label}: event_order 长度与 n_events 不一致")
                    if not np.allclose(frame["level_h"] * (frame["level_n"] + 1.0), 1.0):
                        problems.append(f"{label}: level_h 与 level_n 不符合 h=1/(n+1)")
    if checked < 60:
        problems.append(f"采样点只有 {checked} 个：测试可能没真的跑起来（预期 >= 60）")
    rec = record(
        label="帧角点：有限性 + requiredAux + 掩码自洽",
        theory="数组与 meta 全有限、能编 .nlf；requiredAux 命中；地板/占位/哨兵掩码与数据一致",
        measured={"n_frames": checked, "n_problems": len(problems), "first": problems[:4]},
        tol="0 个问题且采样点 >= 60",
        passed=not problems and checked >= 60,
    )
    assert rec.passed, rec.measured


@assertion("B", "meta 里的数字能从帧里的数组复算出来（四个页面各自对账）", chapter=CHAPTER)
def test_meta_matches_frame_arrays() -> None:
    r"""meta 是给人读的"结论"，必须能从同一帧的数组复算——否则图和账会各说各话。

    复算的口径：viz1 的平滑因子 = 高频半带内 ``factor_measured`` 的最大值；
    viz3 的均值/标准差/漂移来自 ``residual_factor_mg``；viz4 的 ``loss_now``
    来自 ``loss_history``、``alias_theta_pi`` 来自 ``alias_vs_theta``、
    ``hf_ratio_coarse_visible`` 用正弦基重新投影 ``residual_prolonged``。
    """
    entry = _entry()
    problems: list[str] = []

    viz1 = _viz(entry, "smoothing_spectrum")
    f1 = build_frame(entry, viz1.id, {**_defaults(viz1), "n": 127}, 10.0, strict=True)
    present = f1["mode_present"] > 0.5
    high = (f1["theta_over_pi"] >= 0.5) & present
    if abs(float(np.max(f1["factor_measured"][high])) - float(f1.meta["measured_smoothing_factor"])) > 1e-12:
        problems.append("viz1: measured_smoothing_factor 与 factor_measured 不一致")
    if abs(float(f1["error_norm_history"][-1]) - float(f1.meta["error_norm_ratio"])) > 1e-12:
        problems.append("viz1: error_norm_ratio 与 error_norm_history[-1] 不一致")
    n_floor = int(np.count_nonzero(f1["amplitude_measured"] <= FLOOR))
    if n_floor != int(f1.meta["n_modes_at_floor"]):
        problems.append(f"viz1: n_modes_at_floor={f1.meta['n_modes_at_floor']} 实际 {n_floor}")
    if abs(float(f1.meta["low_mode_factor"]) - float(f1["factor_measured"][0])) > 1e-12:
        problems.append("viz1: low_mode_factor 与 factor_measured[0] 不一致")

    viz2 = _viz(entry, "vcycle_hierarchy")
    f2 = build_frame(entry, viz2.id, {**_defaults(viz2), "levels": 5, "nu": 2}, 2.0, strict=True)
    if abs(float(f2["error_norm_history"][-1]) - float(f2.meta["error_norm_now"])) > 1e-12:
        problems.append("viz2: error_norm_now 与 error_norm_history[-1] 不一致")
    sizes = f2["level_n"]
    n_lv = int(sizes.size)
    gamma = int(f2.meta["gamma"])
    nu_v = int(f2.meta["nu"])
    closed = sum((gamma**lv) * 2.0 * nu_v * sizes[lv] / sizes[0] for lv in range(n_lv - 1))
    closed += (gamma ** (n_lv - 1)) * float(mg.COARSE_SWEEPS) * sizes[-1] / sizes[0]
    if abs(closed - float(f2.meta["work_per_cycle"])) > 1e-9:
        problems.append(f"viz2: work_per_cycle={f2.meta['work_per_cycle']:.4f} 与闭式 {closed:.4f} 不一致")
    if int(f2.meta["levels"]) != int(f2["level_n"].size):
        problems.append("viz2: levels 与 level_n 长度不一致")
    if float(f2["level_h"][0]) != 1.0 / (float(f2["level_n"][0]) + 1.0):
        problems.append("viz2: 最细层 h 不对")

    viz3 = _viz(entry, "grid_independence")
    f3 = build_frame(entry, viz3.id, _defaults(viz3), 0.0, strict=True)
    fm = f3["residual_factor_mg"]
    if abs(float(np.mean(fm)) - float(f3.meta["mg_factor_mean"])) > 1e-12:
        problems.append("viz3: mg_factor_mean 与 residual_factor_mg 不一致")
    if abs(float(np.std(fm)) - float(f3.meta["mg_factor_std"])) > 1e-12:
        problems.append("viz3: mg_factor_std 与 residual_factor_mg 不一致")
    if abs(float(fm[-1] - fm[0]) - float(f3.meta["mg_factor_drift"])) > 1e-12:
        problems.append("viz3: mg_factor_drift 与 residual_factor_mg 不一致")
    speed = f3["cg_iters"] / np.maximum(f3["mgcg_iters"], 1.0)
    if float(np.max(np.abs(speed - f3["mgcg_speedup"]))) > 1e-12:
        problems.append("viz3: mgcg_speedup 与 cg/mgcg 不一致")
    if abs(float(np.max(f3["work_to_tol_mg"])) - float(f3.meta["work_to_tol_mg_max"])) > 1e-9:
        problems.append("viz3: work_to_tol_mg_max 与数组不一致")

    viz4 = _viz(entry, "residual_transfer")
    f4 = build_frame(entry, viz4.id, {**_defaults(viz4), "mix": 1.0, "theta_frac": 0.95}, 6.0, strict=True)
    if abs(float(f4["loss_history"][-1]) - float(f4.meta["loss_now"])) > 1e-12:
        problems.append("viz4: loss_now 与 loss_history[-1] 不一致")
    if abs(float(f4["alias_vs_theta"][int(f4.meta["mode_index"]) - 1]) - float(f4.meta["alias_theta_pi"])) > 1e-12:
        problems.append("viz4: alias_theta_pi 与 alias_vs_theta 不一致")
    if abs(float(f4["loss_vs_theta"][int(f4.meta["mode_index"]) - 1]) - float(f4.meta["loss_at_current_theta"])) > 1e-12:
        problems.append("viz4: loss_at_current_theta 与 loss_vs_theta 不一致")
    if abs(float(np.sqrt(np.sum(f4["residual_coarse"] ** 2))) - float(f4.meta["coarse_visible_amplitude"])) > 1e-12:
        problems.append("viz4: coarse_visible_amplitude 与 R r 的范数不一致")
    V = sine_eigenvectors_1d(int(f4.meta["n"]))
    theta = f4["theta_over_pi"] * np.pi
    c = V.T @ f4["residual_prolonged"]
    hf = float(np.sum(c[theta >= np.pi / 2.0] ** 2) / max(float(np.sum(c**2)), 1e-300))
    if abs(hf - float(f4.meta["hf_ratio_coarse_visible"])) > 1e-10:
        problems.append(f"viz4: hf_ratio_coarse_visible={f4.meta['hf_ratio_coarse_visible']} 复算 {hf}")

    rec = record(
        label="meta 与数组的自洽性（四页）",
        theory="meta 的每个结论都能从同帧数组复算（平滑因子/地板计数/因子统计/损失与别名）",
        measured={"n_problems": len(problems), "problems": problems},
        tol="0 个不一致",
        passed=not problems,
    )
    assert rec.passed, rec.measured


@assertion("B", "帧是纯函数且时间轴钳制：两次构建逐位相同，t<0 等价 t=0，t>tMax 等价 tMax", chapter=CHAPTER)
def test_frames_are_pure_and_time_is_clamped() -> None:
    r"""帧不许有隐藏状态（缓存不改变结果），scrub 也不许偷偷外推。

    ``t`` 超出 ``tMax`` 时若继续按步数外推，读者会以为"再拖一点就能看到更多"——
    本页明确钳制：t=1e6 与 t=tMax 必须给出同一帧。
    """
    entry = _entry()
    problems: list[str] = []
    for spec in entry.manifest.vizzes:
        params = _defaults(spec)
        t_mid = 0.0 if spec.time.mode == "none" else float(spec.time.t_max) / 2.0
        a = build_frame(entry, spec.id, params, t_mid, strict=True)
        b = build_frame(entry, spec.id, params, t_mid, strict=True)
        for name in a.names:
            if not np.array_equal(a[name], b[name]):
                problems.append(f"{spec.id}: 两次构建的 {name} 不一致（不纯）")
        if spec.time.mode != "none":
            t_max = float(spec.time.t_max)
            hi = build_frame(entry, spec.id, params, t_max, strict=True)
            over = build_frame(entry, spec.id, params, 1.0e6, strict=True)
            zero = build_frame(entry, spec.id, params, -5.0, strict=True)
            zero_ref = build_frame(entry, spec.id, params, 0.0, strict=True)
            for name in hi.names:
                if not np.array_equal(hi[name], over[name]):
                    problems.append(f"{spec.id}: t>tMax 没有钳制到 tMax（{name}）")
                    break
            for name in zero.names:
                if not np.array_equal(zero[name], zero_ref[name]):
                    problems.append(f"{spec.id}: t<0 没有钳制到 0（{name}）")
                    break
        else:
            if spec.time.t_max is not None or spec.time.dt is not None or spec.time.fps is not None:
                problems.append(f"{spec.id}: time.mode=none 却声明了 tMax/dt/fps")
    rec = record(
        label="纯函数与时间钳制",
        theory="同参数两次构建逐位相同；scrub 在 [0,tMax] 外钳制；mode=none 不声明 tMax/dt/fps",
        measured={"n_problems": len(problems), "problems": problems},
        tol="0 个问题",
        passed=not problems,
    )
    assert rec.passed, rec.measured


@assertion("C", "帧里的实测阈值确实被跨过：高频消失 4 个量级、低频几乎不动、MG 比平滑子省 2 个量级", chapter=CHAPTER)
def test_frame_thresholds_and_rates() -> None:
    r"""把这一章的三个"数量级"结论做成可失败的阈值断言（都在默认参数附近的实测值上）。

    * n=127、mixed 初值、10 步：高频模态只剩 <= 1e-4（实测 4.2e-8 量级），
      而最低模态还在 0.998 以上；低频/高频存活比 >= 1e3；
    * n=127 的两网格因子（1/9）比单层平滑子（0.9997）小 8 倍以上，
      降到 1e-8 的总代价差 >= 100 倍；
    * viz4 的纯高频残差无论平滑多少步都不可见（loss >= 0.9），
      而混合残差从 nu=0 到 nu=6 的损失至少掉 10 倍。
    """
    entry = _entry()
    problems: list[str] = []

    viz1 = _viz(entry, "smoothing_spectrum")
    f1 = build_frame(entry, viz1.id, {**_defaults(viz1), "n": 127, "content": "mixed"}, 10.0, strict=True)
    if float(f1.meta["high_mode_survival"]) > 1e-4:
        problems.append(f"viz1: 高频存活 {f1.meta['high_mode_survival']:.3e} 没有掉到 1e-4 以下")
    if float(f1.meta["low_mode_survival"]) < 0.99:
        problems.append(f"viz1: 低频存活 {f1.meta['low_mode_survival']:.6f} 掉得太快")
    if float(f1.meta["low_over_high_survival"]) < 1e3:
        problems.append("viz1: 低/高存活比不足 1e3")

    viz3 = _viz(entry, "grid_independence")
    f3 = build_frame(entry, viz3.id, _defaults(viz3), 0.0, strict=True)
    if not (float(np.max(f3["residual_factor_two_grid"])) <= 0.12 <= 8.0 * float(f3["residual_factor_two_grid"][0])):
        problems.append("viz3: 两网格因子不是 1/9 量级")
    ratio = float(f3.meta["work_to_tol_smoother_max"]) / max(float(f3.meta["work_to_tol_mg_max"]), 1e-300)
    if ratio < 100.0:
        problems.append(f"viz3: 总代价比只有 {ratio:.1f}，没有达到两个数量级")
    if float(f3.meta["mgcg_iters_max"]) > 10.0:
        problems.append(f"viz3: MGCG 迭代数 {f3.meta['mgcg_iters_max']} 超过 10")

    viz4 = _viz(entry, "residual_transfer")
    pure = build_frame(entry, viz4.id, {**_defaults(viz4), "mix": 1.0, "theta_frac": 0.95}, 6.0, strict=True)
    if float(pure.meta["loss_now"]) < 0.9:
        problems.append(f"viz4: 纯高频残差的损失只有 {pure.meta['loss_now']:.4f}")
    mixed0 = build_frame(entry, viz4.id, {**_defaults(viz4), "mix": 0.5}, 0.0, strict=True)
    mixed6 = build_frame(entry, viz4.id, {**_defaults(viz4), "mix": 0.5}, 6.0, strict=True)
    drop = float(mixed0.meta["loss_now"]) / max(float(mixed6.meta["loss_now"]), 1e-300)
    if drop < 10.0:
        problems.append(f"viz4: 混合残差的损失只掉了 {drop:.2f} 倍")
    if float(pure["residual_norm_history"][-1]) > 5e-3:
        problems.append(f"viz4: 6 步后高频残差范数还剩 {pure['residual_norm_history'][-1]:.3e}")

    rec = record(
        label="跨页阈值：高频消失、低频不动、MG 与平滑子的量级差",
        theory="10 步后高频 <=1e-4 而低频 >=0.99；MG 总代价比平滑子 >=100 倍；"
               "纯高频残差的不可见比例 >=0.9 且混合残差损失至少掉 10 倍",
        measured={
            "viz1_high_survival": float(f1.meta["high_mode_survival"]),
            "viz1_low_survival": float(f1.meta["low_mode_survival"]),
            "viz3_two_grid": float(f3["residual_factor_two_grid"][0]),
            "viz3_cost_ratio": ratio,
            "viz3_mgcg_iters": float(f3.meta["mgcg_iters_max"]),
            "viz4_pure_loss": float(pure.meta["loss_now"]),
            "viz4_mixed_drop": drop,
            "viz4_pure_norm": float(pure["residual_norm_history"][-1]),
        },
        tol="全部阈值跨过（见 theory）",
        passed=not problems,
    )
    assert rec.passed, problems or rec.measured


@assertion("B", "manifest 自洽：每条参数边界/默认值/枚举/临界值都落在定义域内，轴与时间轴声明齐全", chapter=CHAPTER)
def test_manifest_is_self_consistent() -> None:
    r"""参数面板不许出现"拖不到"或"拖了就炸"的声明：默认值在区间内、临界值在区间内、
    ``beyond`` 有交代、每个 viz 至少两条轴且 ``autoscaled`` 显式、``pitfalls`` 与 ``theory`` 非空。"""
    entry = _entry()
    problems: list[str] = []
    for spec in entry.manifest.vizzes:
        if len(spec.axes) < 2:
            problems.append(f"{spec.id}: 只有 {len(spec.axes)} 条轴")
        for axis in spec.axes:
            if not isinstance(axis.autoscaled, bool):
                problems.append(f"{spec.id}/{axis.id}: autoscaled 不是显式布尔")
            if axis.scale not in ("linear", "log"):
                problems.append(f"{spec.id}/{axis.id}: 未知 scale {axis.scale!r}")
        if len(spec.pitfalls) < 60:
            problems.append(f"{spec.id}: pitfalls 只有 {len(spec.pitfalls)} 字")
        if not spec.theory:
            problems.append(f"{spec.id}: theory 为空")
        if spec.data_source != "backend":
            problems.append(f"{spec.id}: dataSource={spec.data_source!r}")
        names = [p.name for p in spec.params]
        if len(set(names)) != len(names):
            problems.append(f"{spec.id}: 参数名重复")
        for p in spec.params:
            if p.type == "enum":
                if p.default not in p.options:
                    problems.append(f"{spec.id}/{p.name}: 默认值不在枚举里")
                if not p.options:
                    problems.append(f"{spec.id}/{p.name}: 空枚举")
            elif p.type in ("int", "float"):
                if p.min is None or p.max is None or p.step is None:
                    problems.append(f"{spec.id}/{p.name}: 缺 min/max/step")
                    continue
                if not (p.min <= p.default <= p.max):
                    problems.append(f"{spec.id}/{p.name}: 默认值越界")
                if p.step <= 0:
                    problems.append(f"{spec.id}/{p.name}: step <= 0")
                for c in p.critical:
                    if not (p.min <= c <= p.max):
                        problems.append(f"{spec.id}/{p.name}: 临界值 {c} 不在 [{p.min},{p.max}] 内")
                    if not p.critical_note:
                        problems.append(f"{spec.id}/{p.name}: 临界值 {c} 没有 criticalNote")
            elif p.type == "bool":
                if not isinstance(p.default, bool):
                    problems.append(f"{spec.id}/{p.name}: bool 默认值不是布尔")
            else:
                problems.append(f"{spec.id}/{p.name}: 未知类型 {p.type!r}")
        if spec.time.mode == "none":
            if spec.time.t_max is not None or spec.time.dt is not None or spec.time.fps is not None:
                problems.append(f"{spec.id}: mode=none 却声明了时间轴参数")
        else:
            if not spec.time.t_max or spec.time.t_max <= 0:
                problems.append(f"{spec.id}: 缺 tMax")
            if spec.time.mode not in ("scrub", "stream"):
                problems.append(f"{spec.id}: 未知 time.mode {spec.time.mode!r}")
    rec = record(
        label="manifest 自洽性",
        theory="参数边界/默认值/临界值/枚举自洽，轴 >= 2 且 autoscaled 显式，pitfalls 与 theory 非空",
        measured={"n_problems": len(problems), "problems": problems},
        tol="0 个问题",
        passed=not problems,
    )
    assert rec.passed, rec.measured
