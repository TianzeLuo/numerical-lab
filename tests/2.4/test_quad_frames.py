"""2.4 章帧契约与内容断言：四个可视化必须在参数角落都站得住。

契约部分与全仓的 ``tests/test_frames_contract.py`` 同一口径，但这里跑**本章**的
参数角落（含 ``sqrt``/``cusp``/``spike``/``adaptive`` 这些容易出 NaN 的分支），
并额外要求：

* ``requiredAux`` 的名字要真的有数组**以它开头**（全仓测试按前缀匹配，
  只写成 ``moment_error`` 这种"包含 error"的名字会被判不合格）；
* 每个 viz 的 ``critical`` 都要真的能被滑块拖到（``min ≤ critical ≤ max``）；
* **内容**上也要对得上：帧里报的实测斜率/阶/地板必须与理论一致——
  可视化不是"看起来像"，它是每章最容易被信以为真的那一层。

A 类守"帧里的数字 = 理论数字"，B 类守契约与不变量。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from numcore.verify import assertion, assert_leq, record
from server import nlf as nlf_mod
from server.frames import build_frame
from server.registry import Registry

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = "2.4"


def _entry() -> Any:
    reg = Registry(ROOT)
    reg.discover(strict=False)
    return reg.get(CHAPTER)


def _viz(chapter_entry: Any, viz_id: str) -> dict[str, Any]:
    raw = getattr(chapter_entry.module, "MANIFEST", None) or {}
    for spec in raw.get("vizzes", []):
        if spec["id"] == viz_id:
            return spec
    raise AssertionError(f"章节 {CHAPTER} 里没有 viz {viz_id}")


def _corners(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """默认参数 + 每个参数的两个端点（枚举取首尾选项）。"""
    base = {p["name"]: p["default"] for p in spec.get("params", [])}
    out = [dict(base)]
    for p in spec.get("params", []):
        if p.get("type") == "enum":
            for opt in (p["options"][0], p["options"][-1]):
                out.append({**base, p["name"]: opt})
        elif p.get("type") in ("int", "float"):
            for edge in (p["min"], p["max"]):
                out.append({**base, p["name"]: edge})
        elif p.get("type") == "bool":
            out.append({**base, p["name"]: not base[p["name"]]})
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for c in out:
        key = repr(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _times(spec: dict[str, Any]) -> list[float]:
    tm = spec.get("time") or {}
    if tm.get("mode") == "none":
        return [0.0]
    t_max = float(tm.get("tMax", 0.0) or 0.0)
    return [0.0, t_max] if t_max > 0 else [0.0]


@assertion("B", theory="PROTOCOLS §1 帧契约：requiredAux 齐备、数组全有限、meta 只放有限标量", chapter="2.4")
def test_all_chapter_frames_are_well_formed() -> None:
    """四个 viz 在默认参数与全部参数端点上打包，并检查时间轴真的接上了。"""
    entry = _entry()
    problems: list[str] = []
    n_frames = 0
    for viz_id in ("area_convergence", "gauss_vs_equispaced", "romberg_table", "singular_endpoint"):
        spec = _viz(entry, viz_id)
        for params in _corners(spec):
            for t in _times(spec):
                n_frames += 1
                label = f"{viz_id} params={params} t={t}"
                try:
                    frame = build_frame(entry, viz_id, params, t, strict=True)
                    nlf_mod.pack_frame(frame)
                except Exception as exc:  # noqa: BLE001 - 把每一处失败都收集起来
                    problems.append(f"{label}: {type(exc).__name__}: {exc}")
                    continue
                names = [n.lower() for n in frame.names]
                for aux in spec["requiredAux"]:
                    if not any(n.startswith(aux.lower()) for n in names):
                        problems.append(f"{label}: requiredAux {aux!r} 没有以它开头的数组 {names}")
                for name in frame.names:
                    arr = np.asarray(frame[name], dtype=float)
                    if arr.size and not np.all(np.isfinite(arr)):
                        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
                        problems.append(f"{label}: 数组 {name} 有 {n_bad}/{arr.size} 个非有限值")
                for key, value in frame.meta.items():
                    if isinstance(value, (list, tuple, dict, str)):
                        problems.append(f"{label}: meta[{key}] 是非标量 {value!r}")
                    elif isinstance(value, float) and not math.isfinite(value):
                        problems.append(f"{label}: meta[{key}] 不是有限数 {value!r}")
        # 时间轴：默认参数下 t=0 与 t=tMax 必须真的不一样（按结构比较，不盯某一个数组）
        if (spec.get("time") or {}).get("mode") == "scrub":
            base = {p["name"]: p["default"] for p in spec.get("params", [])}
            a = build_frame(entry, viz_id, base, 0.0, strict=True)
            b = build_frame(entry, viz_id, base, float(spec["time"]["tMax"]), strict=True)
            changed = a.meta.get("step") != b.meta.get("step")
            if not changed:
                for name in a.names:
                    if name not in b.names:
                        changed = True
                        break
                    av, bv = np.asarray(a[name], float), np.asarray(b[name], float)
                    if av.shape != bv.shape or float(np.max(np.abs(av - bv))) > 0.0:
                        changed = True
                        break
            if not changed:
                problems.append(f"{viz_id}: 时间轴没有接上（t=0 与 t={spec['time']['tMax']} 完全相同）")
    record(
        label=f"{n_frames} 帧全部打包成功",
        theory="0 个问题（数组有限、meta 标量有限、requiredAux 前缀匹配、时间轴有效）",
        measured="0 个问题" if not problems else f"{len(problems)} 个问题：" + "；".join(problems[:6]),
        tol="0",
        passed=not problems,
        detail="参数角落含 adaptive/sqrt/cusp/spike 这些最容易出 NaN 的分支",
    )


@assertion("B", theory="manifest 的 critical 值必须落在滑块量程内（防误导规则 6）", chapter="2.4")
def test_critical_values_are_reachable_by_the_sliders() -> None:
    """逐参数检查：数值参数的 critical 都在 [min,max]，枚举参数的 critical 必须为空。"""
    entry = _entry()
    checked = 0
    problems: list[str] = []
    for viz_id in ("area_convergence", "gauss_vs_equispaced", "romberg_table", "singular_endpoint"):
        spec = _viz(entry, viz_id)
        for p in spec["params"]:
            crit = list(p.get("critical") or [])
            if p["type"] == "enum":
                if crit:
                    problems.append(f"{viz_id}.{p['name']}：枚举参数不该有 critical {crit}")
                continue
            for c in crit:
                checked += 1
                if not (float(p["min"]) <= float(c) <= float(p["max"])):
                    problems.append(f"{viz_id}.{p['name']}：critical {c} 不在 [{p['min']}, {p['max']}]")
                if not str(p.get("criticalNote") or "").strip():
                    problems.append(f"{viz_id}.{p['name']}：critical 非空但 criticalNote 为空")
            if crit and float(p["min"]) <= float(p["default"]) <= float(p["max"]):
                pass
            elif not (float(p["min"]) <= float(p["default"]) <= float(p["max"])):
                problems.append(f"{viz_id}.{p['name']}：默认值 {p['default']} 不在量程内")
    assert_leq(len(problems), 0, label=f"{checked} 条 critical 全部可拖到")
    record(
        label="critical 与 criticalNote 成对出现",
        theory="critical 非空 ⇒ criticalNote 非空，且值在滑块量程内",
        measured=f"检查 {checked} 条，问题 {len(problems)} 个" + ("" if not problems else "：" + "；".join(problems)),
        tol="0 个问题",
        passed=not problems,
        detail="拖不到的临界线就是装饰：它永远不会在交互里出现",
    )


@assertion("A", theory="帧里报的实测斜率 = 公式的阶；√x 上 = min(p, α+1)", chapter="2.4")
def test_area_convergence_frame_reports_measured_and_theory_slopes() -> None:
    """同一个 viz：光滑函数上实测 = 理论，√x 上实测 = 光滑性上限。"""
    entry = _entry()
    for method, p in (("trapezoid", 2.0), ("simpson", 4.0), ("simpson38", 4.0)):
        frame = build_frame(entry, "area_convergence", {"method": method, "func": "exp", "n": 384}, 384.0, strict=True)
        # 帧里的 slope_measured 是 fit_loglog 的**原始**斜率（双对数图上的倾斜方向，负值），
        # 而"阶"是这个斜率的相反数：两个量都要在帧里，免得读者对着 −2 说"阶是 −2"。
        measured = frame.meta["slope_measured"]
        assert measured is not None
        order = -float(measured)
        record(
            label=f"{method}：帧里的实测斜率 = 理论阶 {p:g}",
            theory=p,
            measured=f"斜率 {measured:.6f} ⇒ 阶 {order:.6f}",
            tol="±0.1",
            passed=bool(abs(order - p) <= 0.1),
            detail=f"R²={frame.meta['fit_r2']:.8f}，窗口 n∈[{frame.meta['fit_start']:g}, {frame.meta['fit_end']:g}]；"
            f"理论线斜率 meta.slope_theory={frame.meta['slope_theory']:g}",
        )
    frame = build_frame(entry, "area_convergence", {"method": "simpson", "func": "sqrt", "n": 192}, 192.0, strict=True)
    predicted = frame.meta["slope_theory_smoothness"]
    measured = frame.meta["slope_measured"]
    record(
        label="√x 上帧里的实测量 = 光滑性上限（不是 4 阶）",
        theory=f"min(p, α+1) = {predicted:g}",
        measured=measured,
        tol="±0.1",
        passed=bool(abs(measured - predicted) <= 0.1 and abs(measured - 4.0) > 1.0),
        detail="帧把两条线都画出来：理论阶（4）与光滑性预测（1.5）。"
        "只有前者会把「Simpson 在 √x 上是 1.5 阶」这件事整个藏起来",
    )


@assertion("A", theory="Gauss 帧必须同时给出 k=2n−1 精确与 k=2n 失效", chapter="2.4")
def test_gauss_frame_shows_both_directions_of_the_exactness_claim() -> None:
    """两个方向都在 meta 里：一个在地板上，一个高出地板若干数量级。"""
    entry = _entry()
    for n in (2, 4, 8, 12, 20):
        spec = {"n": n, "func": "exp"}
        frame = build_frame(entry, "gauss_vs_equispaced", spec, 0.0, strict=True)
        at_limit = float(frame.meta["error_at_limit"])
        beyond = float(frame.meta["error_beyond_limit"])
        floor = float(frame.meta["error_floor"])
        record(
            label=f"n={n}：k=2n−1 在地板、k=2n 高出 1e3 倍以上",
            theory=f"err(2n−1) ≤ 4u、err(2n) ≥ 1e3·u（帧里的地板 {floor:.3e}）",
            measured=f"err(2n−1)={at_limit:.3e}、err(2n)={beyond:.3e}、比 {beyond / max(at_limit, floor):.3e}",
            tol="at_limit ≤ 4u 且 beyond ≥ 1e3·u",
            passed=bool(at_limit <= 4.0 * floor and beyond >= 1e3 * floor),
            detail=f"失效因子 meta.failure_factor={frame.meta['failure_factor']:.3e}、"
            f"2n 处的相对误差={frame.meta['error_beyond_relative']:.2e}",
        )
    frame = build_frame(entry, "gauss_vs_equispaced", {"n": 8, "func": "exp"}, 0.0, strict=True)
    record(
        label="条件数曲线：等距规则 m=9 起离开 1，Gauss 恒为 1",
        theory="Γ_equi(m=9)=1.4512、Γ_equi(m=19)=175.46、Γ_gauss≡1",
        measured=f"m=9 时 {frame.meta['knots_m9_gamma']:.4f}、m=19 时 {frame.meta['knots_m19_gamma']:.4f}、"
        f"Gauss {frame.meta['gamma_gauss']:.4f}",
        tol="±0.05（m=9）/ ±1（m=19）",
        passed=bool(
            abs(frame.meta["knots_m9_gamma"] - 1.4512) <= 0.05
            and abs(frame.meta["knots_m19_gamma"] - 175.46) <= 1.0
            and abs(frame.meta["gamma_gauss"] - 1.0) <= 1e-12
        ),
        detail="采样点集与 Gauss 节点集在同一张轴上：负权重与 Γ>1 是同一件事的两面",
    )


@assertion("A", theory="Romberg 帧里每一列的实测阶 = 2j+2", chapter="2.4")
def test_romberg_frame_column_orders_match_theory() -> None:
    """帧数组 order_measured 与 order_theory 必须逐项一致（±0.1）。"""
    entry = _entry()
    frame = build_frame(entry, "romberg_table", {"func": "exp", "levels": 10}, 10.0, strict=True)
    measured = np.asarray(frame["order_measured"], dtype=float)
    theory = np.asarray(frame["order_theory"], dtype=float)
    record(
        label="帧里的列阶数组与理论数组逐项一致",
        theory="order_theory = 2j+2（j=0..3）",
        measured="；".join(f"{m:.4f}" for m in measured),
        tol="±0.1",
        passed=bool(measured.size >= 4 and np.all(np.abs(measured - theory) <= 0.1)),
        detail="理论列号 " + "；".join(f"{t:.0f}" for t in theory) + "；"
        "第 j 列的可用行数随 j 减少，行数不够的列在帧里被剔除而不是补 0",
    )
    # 揭示到第 4 层时，只有两列算得出阶——数组长度必须跟着变，而不是填地板值
    partial = build_frame(entry, "romberg_table", {"func": "exp", "levels": 10}, 4.0, strict=True)
    n_partial = int(np.asarray(partial["order_measured"], dtype=float).size)
    record(
        label="早期揭示时列阶数组随之变短（不编造）",
        theory="t=4（4 行）时只有 2 列有 ≥3 个可用点",
        measured=f"t=4 时 {n_partial} 列；t=10 时 {int(measured.size)} 列",
        tol="n_partial == 2",
        passed=bool(n_partial == 2),
        detail="meta.order_col3 在 t=4 时是 None：留白比填一个 0 诚实",
    )


@assertion("B", theory="端点奇异帧：误差停在地板上，地板 = 未采样尾巴", chapter="2.4")
def test_singular_frame_error_plateaus_at_the_unsampled_tail() -> None:
    """节点数从 129 加到 513，帧里的误差一动不动；地板与解析尾巴同量级。"""
    entry = _entry()
    frame_lo = build_frame(entry, "singular_endpoint", {"func": "inv_sqrt", "h": "1/16", "n": 129}, 64.0, strict=True)
    frame_hi = build_frame(entry, "singular_endpoint", {"func": "inv_sqrt", "h": "1/16", "n": 513}, 256.0, strict=True)
    e_lo, e_hi = float(frame_lo.meta["error_current"]), float(frame_hi.meta["error_current"])
    theory = float(frame_hi.meta["tail_theory"])
    record(
        label="节点数 ×4，误差不变（地板）",
        theory="比值 ∈ [0.9, 1.1]（完全不动）",
        measured=f"{e_lo:.6e} → {e_hi:.6e}（比值 {e_hi / e_lo:.6f}）",
        tol="±10%",
        passed=bool(0.9 <= e_hi / e_lo <= 1.1),
        detail=f"地板参考线 tail_theory={theory:.3e}（= ∫_0^{{u/2}}f = √(2u)），"
        f"实测 x_min={frame_hi.meta['x_min']:.3e}、尾巴={frame_hi.meta['tail_estimate']:.3e}",
    )
    assert_leq(
        abs(e_hi / theory),
        5.0,
        rtol=0.0,
        atol=0.0,
        label="实测误差与 √(2u) 同量级（不超过 5 倍）",
        detail=f"误差 {e_hi:.3e}、理论地板 {theory:.3e}、比 {e_hi / theory:.3f}",
    )
    saturated = np.asarray(frame_hi["ts_saturated"], dtype=float)
    nodes = np.asarray(frame_hi["ts_nodes"], dtype=float)
    first_sat = float(nodes[np.argmax(saturated > 0)]) if bool(np.any(saturated > 0)) else -1.0
    record(
        label="饱和节点从 h=1/16 的第 103 个节点开始出现",
        theory="2·ceil(t*/h)+1 = 103（t*=3.172645）",
        measured=f"首个饱和出现在 {first_sat:.0f} 个节点处；"
        f"513 个节点里有 {int(saturated[-1])} 个饱和",
        tol="[101, 105]",
        passed=bool(101.0 <= first_sat <= 105.0),
        detail="饱和节点的权重下溢到 0、坐标舍入到端点：它们必须按 0 计，"
        "否则 0·inf 会让整帧变成 NaN",
    )
    # h 是另一个旋钮：h=1/8 时很小的节点数就到底，h=1/32 时前段收敛慢得多
    fast = build_frame(entry, "singular_endpoint", {"func": "inv_sqrt", "h": "1/8", "n": 129}, 32.0, strict=True)
    slow = build_frame(entry, "singular_endpoint", {"func": "inv_sqrt", "h": "1/32", "n": 129}, 64.0, strict=True)
    record(
        label="h 与节点数是两个独立旋钮",
        theory="h=1/8 时 65 个节点已到底；h=1/32 时同样节点数还差得远",
        measured=f"h=1/8（m=32）err={fast.meta['error_current']:.3e}；"
        f"h=1/32（m=64）err={slow.meta['error_current']:.3e}，饱和 {slow.meta['n_saturated']:.0f} 个",
        tol="fast ≤ 1e-6 且 slow ≥ 1e-4",
        passed=bool(
            float(fast.meta["error_current"]) <= 1e-6 and float(slow.meta["error_current"]) >= 1e-4
        ),
        detail="h 太大：还没细分就饱和；h 太小：前段（|t| 小）的节点太稀，收敛慢得多",
    )
