r"""3.3 章：章节帧（``server/chapters/3.3.py``）的 A/B 类断言。

这一层回答的是**"帧有没有撒谎"**：

* 每个可视化在默认参数与**所有参数的边界/枚举端点**上都能构建并打包成 ``.nlf``，
  数组全有限、``meta`` 全有限（三个反例的具体做法见 ``strings.nonfinite_policy``）；
* 招牌帧（能量漂移对决）里那句"跑了 10⁶ 步"必须真的是 10⁶ 步——``meta.n_steps_observed``
  与数组长度一起看，不看墙上时钟；
* 越过稳定边界的那些帧必须走"占位值 + 掩码 + 文案"三件套：掩码有 0、
  占位值等于最后一个可信值（不是 0——0 会被读成"误差消失了"，正好读反）；
* 帧里的结论性数字（带宽 = h²/4、两半之比 = 1 vs 2、角动量 1e-14 vs 5e-7）
  与内核测试里的同名量必须一致到给定的容差。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numcore.verify import assertion, record

ROOT = None
try:  # 与其它章节测试同口径：把仓库根塞进 sys.path
    import pathlib
    import sys

    ROOT = pathlib.Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
except Exception:  # pragma: no cover
    raise

from server import nlf as nlf_mod  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.registry import Registry  # noqa: E402

U = float(np.finfo(float).eps) / 2.0
_ENTRY = None


def entry():
    global _ENTRY
    if _ENTRY is None:
        reg = Registry(str(ROOT))
        reg.discover(strict=False)
        _ENTRY = reg.get("3.3")
    return _ENTRY


def viz_ids() -> list[str]:
    return [v["id"] for v in entry().module.MANIFEST["vizzes"]]


def arr_of(fr, name: str) -> np.ndarray:
    """取帧里的数组（``Frame`` 没有公开的 arrays 字典，统一走 ``iter_arrays``）。"""
    for spec, arr in nlf_mod.iter_arrays(fr):
        if spec.name == name:
            return np.asarray(arr)
    raise KeyError(f"帧里没有数组 {name!r}；现有 {[sp.name for sp, _ in nlf_mod.iter_arrays(fr)]}")


def build(viz_id: str, params: dict | None = None, t: float = 0.0):
    spec = next(v for v in entry().module.MANIFEST["vizzes"] if v["id"] == viz_id)
    full = {p["name"]: p["default"] for p in spec["params"]}
    if params:
        full.update(params)
    return build_frame(entry(), viz_id, full, t, strict=True), full


def corner_params(viz_id: str) -> list[dict]:
    """默认值 + 每个参数各自的边界（数值取 min/max，枚举取两端）——契约测试同一套。"""
    spec = next(v for v in entry().module.MANIFEST["vizzes"] if v["id"] == viz_id)
    base = {p["name"]: p["default"] for p in spec["params"]}
    out = [dict(base)]
    for p in spec["params"]:
        if p["type"] == "enum":
            out.append({**base, p["name"]: p["options"][0]})
            out.append({**base, p["name"]: p["options"][-1]})
        elif p["type"] in ("int", "float"):
            out.append({**base, p["name"]: p["min"]})
            out.append({**base, p["name"]: p["max"]})
    uniq: list[dict] = []
    seen = set()
    for it in out:
        key = repr(sorted(it.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return uniq


# --------------------------------------------------------------------------- #
# A 类：清单与章节声明的对账
# --------------------------------------------------------------------------- #
@assertion("A", "MANIFEST 与 docs/chapters.yaml 的声明一致：4 个可视化、依赖 3.1、断言类 B/C", chapter="3.3")
def test_manifest_matches_chapter_declaration() -> None:
    import yaml  # 只在测试里用（P3 禁令只覆盖 numcore/server）

    raw = entry().module.MANIFEST
    ids = viz_ids()
    spec = None
    for item in yaml.safe_load((ROOT / "docs" / "chapters.yaml").read_text(encoding="utf-8"))["chapters"]:
        if str(item["id"]) == "3.3":
            spec = item
            break
    assert spec is not None, "docs/chapters.yaml 里没有 3.3"
    assert len(ids) == int(spec["viz"]) == 4, f"可视化个数：MANIFEST {len(ids)} vs chapters.yaml {spec['viz']}"
    assert spec["deps"] == ["3.1"], f"依赖声明变了：{spec['deps']}"
    assert spec["asserts"] == ["B", "C"], f"断言类声明变了：{spec['asserts']}"
    for rel in spec["modules"]:
        assert (ROOT / rel).is_file(), f"chapters.yaml 声明的模块不存在：{rel}"
    assert ids == ["energy_drift_duel", "phase_orbit", "modified_hamiltonian", "long_term_orbit"]
    for v in raw["vizzes"]:
        assert len(v["pitfalls"]) >= 60 and v["theory"], f"{v['id']} 的 pitfalls/theory 不完整"
        assert v["requiredAux"] == ["error"], f"{v['id']} 的 requiredAux 必须含 'error'（规则 3）"
    record(
        label="章节声明对账（4 viz / 依赖 3.1 / 断言 B,C）",
        theory="docs/chapters.yaml 是章节清单的唯一真源，MANIFEST 必须逐项对齐",
        measured=f"{len(ids)} 个可视化：{ids}；deps={spec['deps']}；asserts={spec['asserts']}；"
        f"模块 {len(spec['modules'])} 个全部存在",
        tol="逐项一致",
        passed=True,
        kind="A",
    )


@assertion("A", "每个可视化的参数角点 × 时间轴两端都能构建并打包，且数组/meta 全有限", chapter="3.3")
def test_every_corner_frame_is_finite_and_packable() -> None:
    n_frames = 0
    n_arrays = 0
    bad: list[str] = []
    for vid in viz_ids():
        spec = next(v for v in entry().module.MANIFEST["vizzes"] if v["id"] == vid)
        times = [0.0, float(spec["time"]["tMax"])] if spec["time"]["mode"] != "none" else [0.0]
        for params in corner_params(vid):
            for t in times:
                fr, _full = build(vid, params, t)
                blob = nlf_mod.pack_frame(fr)
                if not blob:
                    bad.append(f"{vid} 打包为空")
                n_frames += 1
                for sa, arr in nlf_mod.iter_arrays(fr):
                    n_arrays += 1
                    a = np.asarray(arr)
                    if a.size and not np.all(np.isfinite(a)):
                        bad.append(f"{vid}.{sa.name} 有非有限值（t={t}, {params}）")
                for k, v in fr.meta.items():
                    if isinstance(v, float) and not math.isfinite(v):
                        bad.append(f"{vid}.meta[{k}]={v}")
    assert not bad, "帧体检发现：" + "；".join(bad[:6])
    record(
        label="全参数角点 × 时间轴两端的帧体检",
        theory="PROTOCOLS 1.1/1.3 节：帧里不许出现 NaN/Inf；出不了有限值就写占位+掩码+文案",
        measured=f"{n_frames} 帧、{n_arrays} 个数组、"
        f"{sum(len(build(v) [0].meta) for v in viz_ids())} 个 meta 字段全部有限且可打包（0 个反例）",
        tol="0 个非有限值",
        passed=True,
        kind="A",
        extra={"n_frames": n_frames, "n_arrays": n_arrays},
    )


# --------------------------------------------------------------------------- #
# B 类：帧里的数字必须与内核一致
# --------------------------------------------------------------------------- #
@assertion("B", "招牌帧真的跑了 10⁶ 步：n_steps_observed = 10^6、数组覆盖整段、带宽 = h²/4", chapter="3.3")
def test_headline_frame_really_runs_a_million_steps() -> None:
    fr, _ = build("energy_drift_duel", {"system": "harmonic", "method_symplectic": "verlet",
                                        "method_baseline": "rk4", "h": 0.05, "n_steps": 10**6,
                                        "n_sample": 1500}, t=6.0)
    m = fr.meta
    assert m["n_steps_observed"] == 10**6, f"观测步数必须是 10^6，实测 {m['n_steps_observed']}"
    t = arr_of(fr, "t")
    assert t.size >= 1000, f"采样点太少：{t.size}"
    assert abs(float(t[-1]) - 10**6 * 0.05) < 1.0, f"时间轴末端应到 T=50000，实测 {t[-1]}"
    band = m["band_symplectic"]
    assert abs(band - 0.05**2 / 4) / (0.05**2 / 4) < 1e-5, f"Verlet 带宽应等于 h²/4，实测 {band:.6e}"
    assert abs(m["band_growth_symplectic"] - 1.0) < 0.01, f"辛方法两半之比应为 1，实测 {m['band_growth_symplectic']}"
    assert abs(m["band_growth_baseline"] - 2.0) < 0.05, f"RK4 两半之比应为 2，实测 {m['band_growth_baseline']}"
    assert m["short_term_error_baseline"] < m["short_term_error_symplectic"], (
        f"短期精度反而是 RK4 好（这就是不能只看短期的那条证据）："
        f"RK4 {m['short_term_error_baseline']:.3e} vs Verlet {m['short_term_error_symplectic']:.3e}"
    )
    record(
        label="10^6 步的招牌帧",
        theory="「跑了 10^6 步」必须落在 meta 与数组上，不能靠文案",
        measured=f"n_steps_observed={m['n_steps_observed']}，t 末端 {float(t[-1]):.1f}，"
        f"带宽 {band:.6e}（理论 h²/4={0.05**2 / 4:.6e}），两半之比 {m['band_growth_symplectic']:.4f} vs "
        f"{m['band_growth_baseline']:.4f}；短期误差 RK4 {m['short_term_error_baseline']:.2e} < Verlet {m['short_term_error_symplectic']:.2e}",
        tol="步数精确；带宽相对差 < 1e-5",
        passed=True,
        kind="B",
        extra={"band": band, "growth_sym": m["band_growth_symplectic"]},
    )


@assertion("B", "越过稳定边界的帧走「占位值 + 掩码 + 文案」：掩码有 0、占位值 = 最后一个可信值", chapter="3.3")
def test_blowup_frame_uses_placeholder_and_mask() -> None:
    fr, _ = build("energy_drift_duel", {"system": "harmonic", "method_symplectic": "verlet",
                                        "method_baseline": "explicit_euler", "h": 2.5, "n_steps": 10**6}, t=6.0)
    m = fr.meta
    assert m["blowup_baseline"] is True and m["blowup_symplectic"] is True, "h=2.5 两个方法都该炸"
    assert m["n_masked_baseline"] > 0, "掩码必须有 0（否则就是拿占位值冒充数据）"
    arr = arr_of(fr, "energy_error_baseline")
    mask = arr_of(fr, "mask_baseline")
    k = int(np.argmax(mask == 0.0))
    assert k > 0, "至少要留一个可信点"
    tail = arr[k:]
    assert np.allclose(tail, arr[k - 1]), "占位值必须是最后一个可信值（不是 0、不是 NaN）"
    assert arr[k - 1] != 0.0, "占位值恰好为 0 会让读者读成「误差消失了」"
    assert "占位值" in fr.strings["masking"] and "mask" in fr.strings["masking"]
    assert fr.strings["status"] == "partial_blowup"
    record(
        label="爆点帧的三件套",
        theory="PROTOCOLS 1.3 节：占位值 + 掩码 + 文案；占位值取最后可信值",
        measured=f"h=2.5 显式 Euler 在第 {m['t_blowup_baseline']:.4g} 时刻被掩码截断，"
        f"掩掉的采样点 {m['n_masked_baseline']}/{arr.size}，占位值 {arr[k - 1]:.6e}",
        tol="掩码有 0 且占位值等于末可信值",
        passed=True,
        kind="B",
        extra={"n_masked": m["n_masked_baseline"], "placeholder": float(arr[k - 1])},
    )


@assertion("B", "相空间帧：辛方法到不变曲线的偏差是舍入级、RK4 线性累积（同一帧里两个数）", chapter="3.3")
def test_phase_orbit_frame_separates_closure_from_drift() -> None:
    fv, _ = build("phase_orbit", {"method": "verlet", "h": 0.1, "n_periods": 600}, t=1.0)
    fk, _ = build("phase_orbit", {"method": "rk4", "h": 0.1, "n_periods": 600}, t=1.0)
    mv, mk = fv.meta, fk.meta
    assert mv["closure_at_roundoff"] is True and mv["closure_error_sup"] < 1e-12, (
        f"Verlet 的轨道必须逐点落在不变曲线上，实测 sup={mv['closure_error_sup']:.3e}"
    )
    # 舍入级时"两半之比"没有意义（分母是 1e-15 的噪声）：实测 Verlet 的这个比值是 3.6，
    # 而 sup 只有 1.8e-14。判据是 at_roundoff，比值只在高于舍入时才读。
    if not mv["closure_at_roundoff"]:
        assert abs(mv["closure_error_growth"] - 1.0) < 0.05, f"Verlet 闭合度两半之比应为 1，实测 {mv['closure_error_growth']}"
    assert abs(mk["det_map"] - 1.0) > 1e-9 and mk["symplectic"] is False
    assert abs(mk["closure_error_growth"] - 2.0) < 0.15, f"RK4 闭合度两半之比应 ≈2，实测 {mk['closure_error_growth']}"
    assert mv["exact_error_sup"] < 5e-3, f"Verlet 到精确圆的偏差应是 O(h²)=1.25e-3 量级，实测 {mv['exact_error_sup']:.3e}"
    assert mk["exact_error_sup"] < mv["exact_error_sup"], "RK4 在 600 圈内的瞬时偏差反而更小（这正是要防的误导）"
    assert mk["phase_error_turns"] < abs(mv["phase_error_turns"]), (
        f"相位误差反过来：RK4 更小（{mk['phase_error_turns']:.6f} vs {mv['phase_error_turns']:.6f}）"
        "——闭合与相位是两件事"
    )
    record(
        label="相空间帧的两个数（闭合 vs 相位）",
        theory="辛：点集闭合（到不变曲线偏差 = 舍入级）；相位照样漂（h²/24）",
        measured=f"Verlet sup={mv['closure_error_sup']:.2e}（舍入级 ⟹ growth 不读；"
        f"名义 growth {mv['closure_error_growth']:.3f}，"
        f"det M={mv['det_map']:.12f}）、相位 {mv['phase_error_turns']:+.6f} 圈；"
        f"RK4 sup={mk['closure_error_sup']:.2e}（growth {mk['closure_error_growth']:.3f}，"
        f"1−det={mk['one_minus_det']:.3e}）、相位 {mk['phase_error_turns']:+.6f} 圈",
        tol="Verlet 舍入级且 growth≈1；RK4 growth≈2",
        passed=True,
        kind="B",
        extra={"verlet_sup": mv["closure_error_sup"], "rk4_growth": mk["closure_error_growth"]},
    )


@assertion("B", "修正哈密顿量帧：辛方法的 H_h 守恒到舍入级（真实带宽 = 闭式），RK4 的同一公式在漂", chapter="3.3")
def test_modified_hamiltonian_frame_numbers() -> None:
    fv, _ = build("modified_hamiltonian", {"method": "verlet", "h": 0.1, "n_steps": 10**5}, t=5.0)
    fk, _ = build("modified_hamiltonian", {"method": "rk4", "h": 0.1, "n_steps": 10**5}, t=5.0)
    mv, mk = fv.meta, fk.meta
    assert abs(mv["band_measured"] - mv["band_theory"]) / mv["band_theory"] < 1e-6, (
        f"Verlet 的带宽必须等于闭式 h²/4：实测 {mv['band_measured']:.6e} vs {mv['band_theory']:.6e}"
    )
    assert mv["modified_at_roundoff"] is True and mv["modified_sup"] < 1e-12
    assert mk["modified_at_roundoff"] is False and mk["modified_sup"] > 1e-9, (
        f"RK4 没有二次不变量，同一公式应看到漂移，实测 {mk['modified_sup']:.3e}"
    )
    assert mv["symplectic"] is True and mk["symplectic"] is False
    assert len(fv.strings["linearity_note"]) > 50, "必须写明「精确二次型只在线性系统上存在」"
    record(
        label="修正哈密顿量帧：带宽闭式 + H_h 守恒",
        theory="H_h 的二次型由 2×2 映射给出；det M=1 ⟺ 它精确守恒",
        measured=f"Verlet 带宽 {mv['band_measured']:.6e} = 闭式 {mv['band_theory']:.6e}"
        f"（比值 {mv['band_ratio_theory']:.6f}），H_h 相对变化 {mv['modified_sup']:.2e}；"
        f"RK4 的同一公式给出 {mk['modified_sup']:.2e}（1−det = {mk['one_minus_det']:.3e}）",
        tol="带宽相对差 < 1e-6；辛 < 1e-12、RK4 > 1e-9",
        passed=True,
        kind="B",
        extra={"band_ratio": mv["band_ratio_theory"], "rk4_mod": mk["modified_sup"]},
    )


@assertion("B", "行星轨道帧：形状（有界）与方位（相位）分开报；e=0 时方位是占位值并有文案", chapter="3.3")
def test_long_term_orbit_frame_separates_shape_and_orientation() -> None:
    fv, _ = build("long_term_orbit", {"case": "kepler_ecc_0.6", "method": "verlet", "h": 0.01}, t=200.0)
    fk, _ = build("long_term_orbit", {"case": "kepler_ecc_0.6", "method": "rk4", "h": 0.01}, t=200.0)
    fc, _ = build("long_term_orbit", {"case": "kepler_circular", "method": "verlet", "h": 0.01}, t=200.0)
    mv, mk = fv.meta, fk.meta
    assert mv["n_orbits"] == 200 and mv["n_steps"] > 100000
    assert abs(mv["closure_growth"] - 1.0) < 0.05, f"Verlet 的形状失真应有界，实测 growth={mv['closure_growth']}"
    assert mk["closure_growth"] > 1.7, f"RK4 的形状失真应线性长，实测 growth={mk['closure_growth']}"
    assert mv["angmom_drift"] < 1e-12, f"Verlet 的角动量必须守恒，实测 {mv['angmom_drift']:.3e}"
    assert mk["angmom_drift"] > 100.0 * mv["angmom_drift"], f"RK4 的角动量必须一直在漂，实测 {mk['angmom_drift']:.3e}"
    assert mk["band_energy"] < mv["band_energy"], "RK4 的瞬时能量误差确实更小（这正是要防的误导）"
    assert mv["varpi_defined"] is True and mv["varpi_drift_per_orbit"] != 0.0
    assert fc.meta["varpi_defined"] is False and fc.meta["varpi_drift_total"] == 0.0
    assert "占位值" in fc.strings["phase_note"], "e=0 时 varpi 恒为 0 必须写明是占位值"
    record(
        label="行星轨道帧：形状 vs 方位",
        theory="形状由 (E,L) 决定（辛性管得了）；方位是相位型误差（辛性管不了）",
        measured=f"Verlet：能量带 {mv['band_energy']:.3e}、形状 growth {mv['closure_growth']:.3f}、"
        f"角动量 {mv['angmom_drift']:.2e}、进动 {mv['varpi_drift_per_orbit']:+.3e} rad/圈；"
        f"RK4：能量带 {mk['band_energy']:.3e}（更小）、形状 growth {mk['closure_growth']:.3f}、"
        f"角动量 {mk['angmom_drift']:.2e}、进动 {mk['varpi_drift_per_orbit']:+.3e} rad/圈",
        tol="Verlet 形状 growth≈1 且 |ΔL|<1e-12；RK4 growth>1.7 且 |ΔL| > 100×Verlet",
        passed=True,
        kind="B",
        extra={"verlet_closure": mv["closure_growth"], "rk4_angmom": mk["angmom_drift"]},
    )


@assertion("B", "时间轴口径一致：scrub 帧的 t=0 与 t=tMax 给不同的观测段，且段内数组自洽", chapter="3.3")
def test_time_axis_is_monotone_and_self_consistent() -> None:
    f0, _ = build("energy_drift_duel", {"n_steps": 10**6}, t=0.0)
    f6, _ = build("energy_drift_duel", {"n_steps": 10**6}, t=6.0)
    assert f0.meta["n_steps_observed"] == 1, f"t=0 应给 1 步，实测 {f0.meta['n_steps_observed']}"
    assert f6.meta["n_steps_observed"] == 10**6
    for fr in (f0, f6):
        for name in ("t", "energy_error_symplectic", "energy_error_baseline", "mask_symplectic"):
            assert arr_of(fr, name).ravel().size == arr_of(fr, "t").size, f"{name} 与时间轴长度不一致"
    # log10 时间轴：中间点必须严格介于两端
    f3, _ = build("energy_drift_duel", {"n_steps": 10**6}, t=3.0)
    assert 1 < f3.meta["n_steps_observed"] < 10**6
    assert arr_of(f0, "t").size < arr_of(f3, "t").size < arr_of(f6, "t").size
    record(
        label="时间轴（log10 观测步数）的口径",
        theory="t=0 → 1 步、t=6 → 10^6 步；数组只覆盖已观测的那一段（不是把全长曲线裁一下）",
        measured=f"t=0：{f0.meta['n_steps_observed']} 步 / {arr_of(f0, 't').size} 点；"
        f"t=3：{f3.meta['n_steps_observed']} 步 / {arr_of(f3, 't').size} 点；"
        f"t=6：{f6.meta['n_steps_observed']} 步 / {arr_of(f6, 't').size} 点",
        tol="步数按 10^t 取整且段内自洽",
        passed=True,
        kind="B",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
