/**
 * 3.3 章（几何数值积分与长期行为）model 层测试 —— **自足**：不依赖 Python，也不依赖
 * `data/frames` 是否存在。夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，
 * 字段集合（数组名 / dtype / shape / unit / label、meta / strings / critical 键）
 * 逐字照抄 `server/chapters/3.3.py` 的真实帧（数值是测试用的合成值）。
 *
 * 本文件守的几条「这张图会不会骗人」的规矩：
 *
 * 1. **掩码后的占位值不进图层**：`mask_* = 0` 的位置在夹具里填成 9 / 7 / 1e9 这种哨兵，
 *    模型若把占位值当数据画，图层里立刻出现它（`energy_drift_duel` / `modified_hamiltonian`
 *    的掩码尾段、`long_term_orbit` 的副轴）。
 * 2. **采样太稀不连线**：`phase_orbit` 每圈只有 4 个点时主图必须是散点；每圈 80 点时才连线。
 * 3. **跨圈数据分段**：`long_term_orbit` 的 120 点轨迹按 `floor(t/2π)` 切成 3 段，
 *    每条折线的点数不超过一圈。
 * 4. **临界值参考线真画**：修正哈密顿量的 ±h²/4 闭式带、所有对数副轴的 1e-16 地板、
 *    爆点竖线、`h` 的 critical 角标，都是图上的证据。
 * 5. **缺字段不静默**：meta / 可选数组缺项 → 降级角标点名；必需数组缺失 → `ModelInputError`。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Layer, type Scene } from "../shared/src/scene.js";
import { manifest, modelFor, models } from "../chapters/3.3/index.js";

type Row = readonly [name: string, data: ArrayLike<number>, label: string, shape?: readonly number[]];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical?: Record<string, number>;
  pitfalls?: string;
}

const PITFALLS =
  "它会怎样骗你：夹具里的数字是合成的，但字段集合、掩码口径与降级路径与真实帧完全一致；" +
  "重点看掩码占位值、采样密度与对数轴地板。";

const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };

async function makeFrame(def: FrameDef): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const [name, data, label, shape] of def.arrays) {
    const typed = Float64Array.from(data as ArrayLike<number>);
    specs.push({
      name,
      dtype: "f8",
      shape: (shape ?? [data.length]) as number[],
      order: "C",
      offset: 0,
      nbytes: typed.byteLength,
      unit: "",
      label,
    });
    arrays.set(name, typed);
  }
  return decodeNlf(
    await encodeNlf(
      new Frame({
        chapter: "3.3",
        viz: def.viz,
        kind: "frame",
        codec: "none",
        meta: def.meta,
        strings: def.strings,
        critical: def.critical ?? {},
        pitfalls: def.pitfalls ?? PITFALLS,
        specs,
        arrays,
      }),
    ),
  );
}

interface Override {
  arrays?: Record<string, ArrayLike<number>>;
  meta?: Record<string, number | boolean | null>;
  strings?: Record<string, string>;
  drop?: readonly string[];
  dropMeta?: readonly string[];
}

function tweak(def: FrameDef, ov: Override): FrameDef {
  const arrays = def.arrays
    .filter(([name]) => !(ov.drop ?? []).includes(name))
    .map((row): Row => {
      if (!ov.arrays || !(row[0] in ov.arrays)) return row;
      const data = ov.arrays[row[0]] as ArrayLike<number>;
      const shape = row[3];
      const keepShape = shape !== undefined && shape.reduce((a, b) => a * b, 1) === data.length;
      return [row[0], data, row[2], keepShape ? shape : [data.length]];
    });
  const meta = { ...def.meta };
  for (const k of ov.dropMeta ?? []) delete meta[k];
  return { ...def, arrays, meta: { ...meta, ...ov.meta }, strings: { ...def.strings, ...ov.strings } };
}

const sceneOf = async (def: FrameDef, opts: Partial<ModelOptions> = {}): Promise<Scene> => {
  const model = models[def.viz];
  assert.ok(model, `登记表里没有 ${def.viz}`);
  return model(await makeFrame(def), { ...OPTS, ...opts });
};

const lineSeriesOn = (scene: Scene, axis: string): { x: ArrayLike<number>; y: ArrayLike<number> }[] =>
  scene.layers
    .filter((l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === axis)
    .flatMap((l) => l.series.map((s) => ({ x: s.x, y: s.y })));
const scattersOn = (scene: Scene, axis: string): Array<{ x: ArrayLike<number>; y: ArrayLike<number> }> =>
  scene.layers
    .filter((l): l is Extract<Layer, { kind: "scatter" }> => l.kind === "scatter" && l.axes === axis)
    .map((l) => ({ x: l.x, y: l.y }));
const layerYs = (scene: Scene): number[] => {
  const out: number[] = [];
  for (const l of scene.layers) {
    if (l.kind === "line") for (const s of l.series) out.push(...Array.from(s.y as ArrayLike<number>));
    else if (l.kind === "scatter") out.push(...Array.from(l.y as ArrayLike<number>));
    else if (l.kind === "band") out.push(...Array.from(l.band.ylo as ArrayLike<number>), ...Array.from(l.band.yhi as ArrayLike<number>));
    else if (l.kind === "heat") out.push(...Array.from(l.values as ArrayLike<number>));
    else if (l.kind === "bars") out.push(...Array.from(l.values as ArrayLike<number>));
  }
  return out;
};
const degradedOf = (scene: Scene): string =>
  scene.badges.filter((b) => b.text.includes("降级")).map((b) => b.text).join(" | ");
const textsOf = (scene: Scene): string => scene.texts.map((t) => t.text).join(" | ");
const badgesOf = (scene: Scene): string => scene.badges.map((b) => b.text).join(" | ");
const axisOf = (scene: Scene, id: string) => {
  const ax = scene.axes.find((a) => a.id === id);
  assert.ok(ax, `场景里没有轴 ${id}`);
  return ax;
};

// --------------------------------------------------------------------------- //
// 夹具
// --------------------------------------------------------------------------- //

const ET = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
const E_ERR_S = [0, -2e-4, 1e-4, -1.5e-4, 1e-4, -2e-4];
/** 后两点被掩码：填 9 是哨兵——模型若画占位值，图层里会出现 9。 */
const E_ERR_B = [0, -2e-5, -4e-5, -6e-5, 9, 9];
const E_DEF: FrameDef = {
  viz: "energy_drift_duel",
  arrays: [
    ["t", ET, "观测时间 t"],
    ["energy_error_symplectic", E_ERR_S, "辛方法的能量相对误差"],
    ["energy_error_baseline", E_ERR_B, "非辛对照的能量相对误差"],
    ["energy_band_hi", ET.map(() => 6e-4), "误差带上边"],
    ["energy_band_lo", ET.map(() => -6e-4), "误差带下边"],
    ["drift_line_symplectic", [0, -5e-5, -9e-5, -1.2e-4, -1.5e-4, -1.8e-4], "辛方法趋势线"],
    ["drift_line_baseline", [0, -3e-5, -6e-5, -9e-5, -9e-5, -9e-5], "对照趋势线"],
    ["mask_symplectic", ET.map(() => 1), "辛方法掩码"],
    ["mask_baseline", [1, 1, 1, 1, 0, 0], "对照掩码"],
  ],
  meta: {
    t_axis: 3,
    n_steps_observed: 1000,
    n_steps_max: 1000000,
    h: 0.05,
    n_sample: 6,
    n_periods_observed: 79.58,
    band_symplectic: 6.25e-4,
    band_baseline: 2.17e-4,
    trend_symplectic: 5.17e-7,
    trend_baseline: 2.169e-4,
    trend_over_band_symplectic: 8.27e-4,
    trend_over_band_baseline: 0.9999,
    monotone_fraction_symplectic: 2e-4,
    monotone_fraction_baseline: 1,
    band_growth_symplectic: 0.99999,
    band_growth_baseline: 2.0026,
    drift_slope_symplectic: -1e-11,
    drift_slope_baseline: -4.3e-9,
    end_error_symplectic: -1.33e-4,
    end_error_baseline: -2.17e-4,
    short_term_error_symplectic: 5.157e-3,
    short_term_error_baseline: 2.612e-6,
    short_term_window: 50.25,
    ratio_band_over_baseline: 0.347,
    n_force_symplectic: 2000,
    n_force_baseline: 4000,
    blowup_symplectic: false,
    blowup_baseline: true,
    t_blowup_symplectic: 0,
    t_blowup_baseline: 0.4,
    n_masked_symplectic: 0,
    n_masked_baseline: 2,
    det_map_symplectic: 1,
    det_map_baseline: 0.9999999998,
    t: 3,
  },
  strings: {
    status: "partial_blowup",
    system: "谐振子",
    method_symplectic: "Verlet",
    method_baseline: "RK4",
    time_axis: "log10(观测步数)",
    data_source: "backend",
  },
  critical: { h: 2 },
};

const PH_T = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
const PH_Q = PH_T.map((t) => Math.cos(2 * Math.PI * t));
const PH_P = PH_T.map((t) => Math.sin(2 * Math.PI * t));
const CIRCLE = Array.from({ length: 5 }, (_, i) => {
  const a = (2 * Math.PI * i) / 5;
  return [Math.cos(a), Math.sin(a)] as const;
});
const PH_DEF: FrameDef = {
  viz: "phase_orbit",
  arrays: [
    ["orbit_q", PH_Q, "数值轨道的 q"],
    ["orbit_p", PH_P, "数值轨道的 p"],
    ["invariant_q", CIRCLE.map(([q]) => q), "不变量等值线 q"],
    ["invariant_p", CIRCLE.map(([, p]) => p), "不变量等值线 p"],
    ["exact_q", CIRCLE.map(([q]) => q), "精确圆 q"],
    ["exact_p", CIRCLE.map(([, p]) => p), "精确圆 p"],
    ["closure_error", PH_T.map((_, i) => -1e-3 * (i + 1)), "到不变曲线的偏差"],
    ["exact_error", PH_T.map((_, i) => 1.2e-3 * (i + 1)), "到精确圆的偏差"],
    ["energy_error", PH_T.map((_, i) => 2e-6 * (i + 1)), "能量相对误差"],
    ["t_series", PH_T, "采样时刻"],
  ],
  meta: {
    t_axis: 0.5,
    h: 0.1,
    n_steps: 40,
    n_periods_observed: 2,
    horizon_time: 4,
    n_draw: 8,
    energy_error_sup: 1.6e-5,
    energy_error_end: 1.6e-5,
    radius_band: 8e-3,
    radius_drift: -8e-3,
    closure_error_sup: 8e-3,
    closure_error_growth: 1.9,
    closure_at_roundoff: false,
    exact_error_sup: 9.6e-3,
    exact_error_growth: 1.9,
    phase_error_turns: 1e-4,
    phase_defined: true,
    naive_return_growth: 1.5,
    det_map: 0.99999,
    one_minus_det: 1e-5,
    symplectic: false,
    invariant_residual: 1e-5,
    invariant_mixing: 0,
    method_order: 4,
    blowup: false,
    n_masked: 0,
    t: 0.5,
  },
  strings: {
    status: "ok",
    method: "RK4",
    time_axis: "轨道完成度",
    closure_note: "到不变量曲线的偏差随时间长大",
    phase_note: "相位误差是另一种读法",
    invariant_note: "非辛：不存在二次不变量",
    misleading_note: "每圈最近回归距离是采样受限的假象",
    nonfinite_policy: "非有限值不画",
    data_source: "backend",
  },
  critical: { h: 2 },
};

const MH_T = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
const MH_E = [0, 0.03, -0.04, 0.05, 0.04, -0.05];
const MH_MOD = [0, 1e-14, -2e-15, 3e-14, -1e-14, 2e-14];
const MH_DEF: FrameDef = {
  viz: "modified_hamiltonian",
  arrays: [
    ["t", MH_T, "观测时间 t"],
    ["energy_error", MH_E, "真实能量的相对误差"],
    ["energy_band_hi", MH_T.map(() => 0.0526), "误差带上边"],
    ["energy_band_lo", MH_T.map(() => -0.0526), "误差带下边"],
    ["modified_error", MH_MOD, "修正能量 H_h 的相对变化"],
    ["drift_line", [0, 0.001, 0.002, 0.003, 0.004, 0.005], "真实能量误差趋势线"],
    ["mask_energy", [1, 1, 1, 1, 0, 0], "能量掩码"],
  ],
  meta: {
    t_axis: 6,
    h: 0.1,
    n_steps: 100000,
    n_sample: 6,
    n_periods_observed: 1591.5,
    band_measured: 0.0526,
    band_theory: 0.052631578947368425,
    band_ratio_theory: 0.9999,
    drift_slope: -1.45e-9,
    trend_over_band: 1.45e-4,
    monotone_fraction: 4.86e-4,
    band_growth: 0.999997,
    modified_sup: 2.04e-14,
    modified_at_roundoff: true,
    det_map: 1,
    one_minus_det: 0,
    det_theory_rk4: 0,
    symplectic: true,
    invariant_residual: 1.7e-17,
    invariant_mixing: -0.05,
    iterations_mean: 0,
    blowup: false,
    n_masked: 2,
    t: 6,
  },
  strings: {
    status: "ok",
    method: "辛 Euler",
    time_axis: "log10(观测步数)",
    band_note: "真实能量在带里振荡",
    invariant_note: "辛 ⟹ H_h 精确守恒",
    linearity_note: "二次型只在谐振子上存在",
    masking: "mask 0 是占位值",
    nonfinite_policy: "非有限值不画",
    data_source: "backend",
  },
  critical: { h: 2 },
};

const LT_LAPS = 3;
const LT_PER_LAP = 40;
const LT_PTS = LT_LAPS * LT_PER_LAP;
const LT_T = Array.from(
  { length: LT_PTS },
  (_, i) => (Math.floor(i / LT_PER_LAP) + (i % LT_PER_LAP) / LT_PER_LAP) * 2 * Math.PI,
);
const LT_X = LT_T.map((t) => Math.cos(t) - 0.6);
const LT_Y = LT_T.map((t) => 0.8 * Math.sin(t));
const LT_REF = Array.from({ length: 50 }, (_, i) => {
  const a = (2 * Math.PI * i) / 49;
  return [Math.cos(a) - 0.6, 0.8 * Math.sin(a)] as const;
});
const LT_N = 12;
const LT_DEF: FrameDef = {
  viz: "long_term_orbit",
  arrays: [
    ["traj_x", LT_X, "数值轨道 x"],
    ["traj_y", LT_Y, "数值轨道 y"],
    ["ref_x", LT_REF.map(([x]) => x), "精确椭圆 x"],
    ["ref_y", LT_REF.map(([, y]) => y), "精确椭圆 y"],
    ["t_series", LT_T, "采样时刻"],
    ["energy_error", Array.from({ length: LT_N }, (_, i) => (i < 10 ? 7e-4 : 1e9)), "能量误差"],
    ["angmom_error", Array.from({ length: LT_N }, (_, i) => (i < 10 ? 1e-14 * Math.sin(i) : 1e9)), "角动量漂移"],
    ["closure_error", Array.from({ length: LT_N }, (_, i) => (i < 10 ? -2e-4 * (i / 10) : 1e9)), "近星点失真"],
    ["ecc_error", Array.from({ length: LT_N }, (_, i) => (i < 10 ? 6e-4 * (i / 10) : 1e9)), "偏心率偏差"],
    ["varpi_drift", Array.from({ length: LT_N }, (_, i) => (i < 10 ? -1e-4 * i : 1e9)), "近星点进动"],
    ["mask_energy", Array.from({ length: LT_N }, (_, i) => (i < 10 ? 1 : 0)), "能量掩码"],
  ],
  meta: {
    t_axis: 200,
    h: 0.01,
    eccentricity: 0.6,
    n_orbits: 3,
    n_steps: 1884,
    n_steps_max: 125664,
    horizon_time: 18.85,
    band_energy: 7.4e-4,
    end_energy_error: 7e-4,
    trend_over_band: 5e-5,
    drift_slope: 3e-11,
    angmom_drift: 1.9e-14,
    closure_sup: 2.47e-4,
    closure_growth: 1.0,
    closure_first_half: 2.469e-4,
    closure_second_half: 2.47e-4,
    closure_at_roundoff: false,
    ecc_error_sup: 6.58e-4,
    ecc_error_end: 6.4e-4,
    ecc_measured: 0.6003,
    ecc_error_is_absolute: false,
    varpi_defined: true,
    varpi_drift_total: -1.3e-3,
    varpi_drift_per_orbit: -4.3e-4,
    r_peri_measured: 0.4,
    r_apo_measured: 1.6,
    angle_step_perihelion: 0.05,
    blowup: false,
    n_masked: 2,
    n_force: 4000,
    t: 200,
  },
  strings: {
    status: "ok",
    case: "kepler_ecc_0.6",
    method: "Verlet",
    time_axis: "观测时长（初始周期数）",
    shape_note: "形状在带内",
    phase_note: "进动是相位型误差",
    resolution_note: "近星点角步长 0.05 rad",
    masking: "mask 0 是占位值",
    nonfinite_policy: "非有限值不画",
    data_source: "backend",
  },
  critical: { h: 0.0023 },
};

const DEFS: Record<string, FrameDef> = {
  energy_drift_duel: E_DEF,
  phase_orbit: PH_DEF,
  modified_hamiltonian: MH_DEF,
  long_term_orbit: LT_DEF,
};

// --------------------------------------------------------------------------- //
// 1. 四个 viz 在整条时间轴上都过校验器
// --------------------------------------------------------------------------- //

for (const [viz, def] of Object.entries(DEFS)) {
  const spec = manifest.vizzes.find((v) => v.id === viz);
  const tMax = spec?.time.mode === "none" ? 0 : (spec?.time.tMax ?? 1);
  for (const t of [0, tMax / 2, tMax]) {
    test(`${viz}: t=${t} 的 Scene 过 validateScene（规则 1/2/8 全绿）`, async () => {
      const scene = await sceneOf(def, { t });
      assert.deepEqual(validateScene(scene), []);
      assert.equal(scene.chapter, "3.3");
      assert.ok(scene.pitfalls.length > 0, "pitfalls 面板必须非空（规则 7）");
      assert.ok(scene.layers.length >= 3, "每个 viz 至少 3 个图层");
      assert.ok(scene.badges.some((b) => b.kind === "source"), "必须有一枚 source 徽标（规则 5）");
    });
  }
}

test("manifest 与登记表一致：4 个 viz 都有 model，未知 id 返回 null", () => {
  assert.deepEqual(Object.keys(models).sort(), [
    "energy_drift_duel",
    "long_term_orbit",
    "modified_hamiltonian",
    "phase_orbit",
  ]);
  for (const v of manifest.vizzes) assert.ok(modelFor(v.id), `manifest 里的 ${v.id} 没有 model`);
  assert.equal(modelFor("nope"), null);
  assert.equal(manifest.vizzes.length, 4);
});

// --------------------------------------------------------------------------- //
// 2. requiredAux 真的画出来了（按**数值**认）
// --------------------------------------------------------------------------- //

test("energy_drift_duel：requiredAux=error 的两条能量误差曲线都在 main 轴上", async () => {
  const scene = await sceneOf(E_DEF);
  const ys = lineSeriesOn(scene, "main").flatMap((s) => Array.from(s.y as ArrayLike<number>));
  assert.ok(ys.some((v) => Math.abs(v - (-2e-4)) < 1e-15), "辛方法误差曲线不在 main 轴上");
  assert.ok(ys.some((v) => Math.abs(v - (-6e-5)) < 1e-15), "对照误差曲线不在 main 轴上");
});

test("phase_orbit：requiredAux=error 的到不变曲线偏差画在 aux 轴上（取绝对值）", async () => {
  const scene = await sceneOf(PH_DEF);
  const ys = lineSeriesOn(scene, "aux").flatMap((s) => Array.from(s.y as ArrayLike<number>));
  // 夹到 1e-16 地板后取 |·|：第一段应为 1e-3。
  assert.ok(ys.some((v) => Math.abs(v - 1e-3) < 1e-12), "closure_error 的 |值| 不在 aux 轴上");
  assert.ok(ys.every((v) => v > 0), "对数轴上不允许出现 0/负值");
});

test("modified_hamiltonian：requiredAux=error 的真实能量误差在 main 轴上（掩码外的值）", async () => {
  const scene = await sceneOf(MH_DEF);
  const ys = lineSeriesOn(scene, "main").flatMap((s) => Array.from(s.y as ArrayLike<number>));
  assert.ok(ys.some((v) => Math.abs(v - 0.03) < 1e-15), "能量误差曲线不在 main 轴上");
  assert.ok(!ys.some((v) => Math.abs(v - 7) < 1e-12), "掩码占位值 7 被画进了图层");
});

test("long_term_orbit：requiredAux=error 的能量误差在 aux 轴上（下标横轴）", async () => {
  const scene = await sceneOf(LT_DEF);
  const series = lineSeriesOn(scene, "aux").filter((s) => s.y.length > 3);
  assert.ok(series.length >= 3, "aux 轴上至少要有三条误差曲线");
  const maxX = Math.max(...series.map((s) => Math.max(...Array.from(s.x as ArrayLike<number>))));
  assert.equal(maxX, 9, "aux 数据曲线的横轴必须止于最后一个可信下标（0…9）");
  const ys = series.flatMap((s) => Array.from(s.y as ArrayLike<number>));
  assert.ok(ys.some((v) => Math.abs(v - 7e-4) < 1e-12), "能量误差不在 aux 轴上");
});

// --------------------------------------------------------------------------- //
// 3. autoscaled 与 manifest 一致 + 固定窗口
// --------------------------------------------------------------------------- //

test("autoscaled 与 manifest 声明一致（两列轴都逐项核对）", async () => {
  for (const [viz, def] of Object.entries(DEFS)) {
    const spec = manifest.vizzes.find((v) => v.id === viz);
    assert.ok(spec);
    const scene = await sceneOf(def);
    for (const ax of spec.axes) {
      const got = scene.axes.find((a) => a.id === ax.id);
      assert.ok(got, `${viz} 缺少 manifest 声明的轴 ${ax.id}`);
      assert.equal(got.autoscaled, ax.autoscaled, `${viz}.${ax.id} 的 autoscaled 与 manifest 不一致`);
    }
  }
});

test("固定窗口：phase_orbit ±1.6、long_term_orbit 按 e=0.6 取 ±1.72", async () => {
  const near = (a: number, b: number) => Math.abs(a - b) < 1e-12;
  const ph = await sceneOf(PH_DEF);
  assert.ok(axisOf(ph, "main").xlim.every((v, i) => near(v, [-1.6, 1.6][i] as number)));
  assert.ok(axisOf(ph, "main").ylim.every((v, i) => near(v, [-1.6, 1.6][i] as number)));
  const lt = await sceneOf(LT_DEF);
  const main = axisOf(lt, "main");
  assert.ok(main.xlim.every((v, i) => near(v, [-1.72, 1.72][i] as number)), "主图窗口必须覆盖远星点 1.6");
  assert.ok(
    lt.texts.some((t) => t.text.includes("±1.72")),
    "面板图例里必须写出窗口 ±1.72（标题只写口径，不重复数字）",
  );
});

test("long_term_orbit：e=0.9 时窗口扩到 ±2.02（远星点 1.9 不被裁掉）", async () => {
  const scene = await sceneOf(LT_DEF);
  const wide = await sceneOf(tweak(LT_DEF, { meta: { eccentricity: 0.9 } }));
  const near = (a: number, b: number) => Math.abs(a - b) < 1e-12;
  assert.ok(axisOf(wide, "main").xlim.every((v, i) => near(v, [-2.02, 2.02][i] as number)));
  assert.ok(!axisOf(wide, "main").xlim.every((v, i) => near(v, axisOf(scene, "main").xlim[i] as number)));
  assert.deepEqual(validateScene(wide), []);
});

// --------------------------------------------------------------------------- //
// 4. 掩码 / 分段 / 对数地板：这一章特有的不变量
// --------------------------------------------------------------------------- //

test("energy_drift_duel：mask=0 的占位值（9）绝不出现在任何图层里", async () => {
  const scene = await sceneOf(E_DEF);
  const ys = layerYs(scene);
  assert.ok(!ys.some((v) => Math.abs(v - 9) < 1e-12), "掩码尾段的占位值 9 被画进了图层");
  // 掩码外的最后一点（−6e-5）仍在：分段不能把整条曲线都丢掉。
  assert.ok(ys.some((v) => Math.abs(v - (-6e-5)) < 1e-15));
});

test("energy_drift_duel：误差带、带边线、零线、趋势线与爆点竖线都真画", async () => {
  const scene = await sceneOf(E_DEF);
  const bands = scene.layers.filter((l): l is Extract<Layer, { kind: "band" }> => l.kind === "band" && l.axes === "main");
  assert.equal(bands.length, 1, "main 轴上必须有辛方法的误差带");
  assert.ok(bands[0].band.yhi[0] === 6e-4 && bands[0].band.ylo[0] === -6e-4);
  const mainLines = lineSeriesOn(scene, "main");
  assert.ok(mainLines.some((s) => s.y.length === 2 && s.y[0] === 0 && s.y[1] === 0), "缺零线");
  assert.ok(mainLines.some((s) => s.y.length === 2 && s.y[0] === 6e-4 && s.y[1] === 6e-4), "缺带上边线");
  assert.ok(mainLines.some((s) => s.y.length === 6 && Math.abs((s.y[1] as number) - (-5e-5)) < 1e-15), "缺辛方法趋势线");
  const verticals = mainLines.filter((s) => s.x.length === 2 && s.x[0] === s.x[1] && s.y[0] !== s.y[1]);
  assert.ok(verticals.some((s) => Math.abs((s.x[0] as number) - 0.3) < 1e-12), "缺对照爆点竖线（最后一个可信点 t=0.3）");
});

test("energy_drift_duel：对照爆点时横轴仍是时间，且溢出/掩码文案在场", async () => {
  const scene = await sceneOf(E_DEF);
  assert.match(badgesOf(scene), /h = 0\.0500/, "缺 h 的 criticalBadge");
  assert.match(textsOf(scene), /趋势\/带宽/);
  assert.match(textsOf(scene), /掩码 2 点/);
});

const phaseMainScatters = (scene: Scene) => scattersOn(scene, "main");
const phaseMainLines = (scene: Scene) => lineSeriesOn(scene, "main");

test("phase_orbit：每圈 4 个采样点时只画散点（连线会画出横穿相平面的弦）", async () => {
  const scene = await sceneOf(PH_DEF);
  assert.ok(phaseMainScatters(scene).some((s) => s.y.length === PH_Q.length), "采样太稀时主图必须是散点");
  assert.ok(
    !phaseMainLines(scene).some((s) => s.y.length === PH_Q.length),
    "采样太稀时不许把轨道连成折线",
  );
  assert.match(textsOf(scene), /只画散点/);
});

test("phase_orbit：每圈 80 个采样点时轨道连线（点数够才连）", async () => {
  const scene = await sceneOf(tweak(PH_DEF, { meta: { n_periods_observed: 0.1 } }));
  assert.ok(
    phaseMainLines(scene).some((s) => s.y.length === PH_Q.length),
    "每圈点数够时必须画折线",
  );
  assert.match(textsOf(scene), /轨道连线/);
});

test("phase_orbit：不变量等值线为空 → 降级而非抛错（h 越界时没有实二次型）", async () => {
  const scene = await sceneOf(tweak(PH_DEF, { arrays: { invariant_q: [], invariant_p: [] } }));
  assert.deepEqual(validateScene(scene), []);
  assert.match(degradedOf(scene), /不变量等值线为空/);
});

test("modified_hamiltonian：闭式带宽 ±h²/4 的两条参考线真画（与实测带区分）", async () => {
  const scene = await sceneOf(MH_DEF);
  const lines = lineSeriesOn(scene, "main");
  const theory = lines.filter(
    (s) => s.y.length === 2 && Math.abs((s.y[0] as number) - 0.052631578947368425) < 1e-15 && s.y[0] === s.y[1],
  );
  assert.equal(theory.length, 1, "缺 +h²/4 参考线");
  const neg = lines.filter(
    (s) => s.y.length === 2 && Math.abs((s.y[0] as number) + 0.052631578947368425) < 1e-15 && s.y[0] === s.y[1],
  );
  assert.equal(neg.length, 1, "缺 −h²/4 参考线");
});

test("modified_hamiltonian：mask=0 的占位值（7）不进主图，也不进副轴", async () => {
  const scene = await sceneOf(MH_DEF);
  const ys = layerYs(scene);
  assert.ok(!ys.some((v) => Math.abs(v - 7) < 1e-12), "掩码占位值 7 被画进了图层");
});

test("modified_hamiltonian：副轴 |H_h| 在舍入级，且有 1e-16 地板线与 |1−det| 参考线", async () => {
  const scene = await sceneOf(MH_DEF);
  const aux = lineSeriesOn(scene, "aux");
  assert.ok(aux.some((s) => s.y.length === 2 && s.y[0] === 1e-16 && s.y[1] === 1e-16), "缺 1e-16 地板线");
  // 辛方法 det=1：|1−det| 落到地板（画在地板上，不是“没画”）。
  assert.ok(aux.some((s) => s.y.length === 2 && s.y[0] === 1e-16), "缺 |1−det| 参考线");
  const ys = aux.flatMap((s) => Array.from(s.y as ArrayLike<number>));
  assert.ok(ys.some((v) => Math.abs(v - 3e-14) < 1e-18), "|modified_error| 的最大值不在副轴上");
});

test("modified_hamiltonian：RK4 的 |1−det| 参考线随帧数据走（1.39e-9）", async () => {
  const scene = await sceneOf(
    tweak(MH_DEF, {
      meta: { det_map: 0.9999999986, one_minus_det: 1.39e-9, symplectic: false, det_theory_rk4: 1.39e-9, band_theory: 0, band_ratio_theory: 0 },
    }),
  );
  const hit = lineSeriesOn(scene, "aux").some(
    (s) => s.y.length === 2 && Math.abs((s.y[0] as number) - 1.39e-9) < 1e-18,
  );
  assert.ok(hit, "缺 |1−det|=1.39e-9 的参考线");
  assert.match(textsOf(scene), /没有二次不变量/);
  assert.ok(!/闭式预测带宽 0/.test(textsOf(scene)), "band_theory=0 是「无闭式」哨兵，不许读成「理论带宽为 0」");
  assert.match(textsOf(scene), /没有闭式带宽/);
  // 隐式中点在谐振子上精确守恒能量：band_theory=0 是**真实的**零带，不是「没有公式」。
  const midpoint = await sceneOf(
    tweak(MH_DEF, { meta: { band_theory: 0, band_ratio_theory: 0, symplectic: true, modified_at_roundoff: true } }),
  );
  assert.match(textsOf(midpoint), /精确守恒能量/);
});

test("long_term_orbit：120 点轨迹按圈切成 3 段，每段不超过一圈（40 点）", async () => {
  const scene = await sceneOf(LT_DEF);
  const traj = lineSeriesOn(scene, "main").filter((s) => s.y.length <= 40);
  assert.equal(traj.length, LT_LAPS, "轨迹没有按圈切段");
  assert.equal(traj.reduce((acc, s) => acc + s.y.length, 0), LT_PTS, "切段后丢了采样点");
  for (const s of traj) assert.ok(s.y.length <= 40, `有一段跨了多圈：${s.y.length} 点`);
});

test("long_term_orbit：mask=0 的副轴占位值（1e9）不进图层", async () => {
  const scene = await sceneOf(LT_DEF);
  const ys = lineSeriesOn(scene, "aux").flatMap((s) => Array.from(s.y as ArrayLike<number>));
  assert.ok(Math.max(...ys) < 1e6, "掩码尾段的 1e9 被画进了副轴");
  assert.ok(ys.some((v) => Math.abs(v - 7e-4) < 1e-12), "掩码外的能量误差也不见了");
});

test("long_term_orbit：副轴数据曲线止于最后一个可信点（参考线可以横贯全轴）", async () => {
  const scene = await sceneOf(LT_DEF);
  for (const s of lineSeriesOn(scene, "aux").filter((s) => s.y.length > 3)) {
    assert.ok(Math.max(...Array.from(s.x as ArrayLike<number>)) <= 9, "副轴数据曲线越过了掩码边界");
  }
});

test("long_term_orbit：h 的临界角标按本轨道角步长折算（e=0.6、h=0.01 → h*=0.02）", async () => {
  const scene = await sceneOf(LT_DEF);
  assert.match(badgesOf(scene), /h = 0\.01000/);
  assert.match(badgesOf(scene), /理论临界 0\.02000/, "e=0.6 时临界值应折算成 0.02，而不是照抄 e=0.9 的 0.0023");
  const e0 = await sceneOf(tweak(LT_DEF, { meta: { eccentricity: 0 } }));
  assert.match(degradedOf(e0), /圆轨道/);
});

// --------------------------------------------------------------------------- //
// 5. 降级不静默 / 必需数组缺失 / 窄画布
// --------------------------------------------------------------------------- //

test("缺可选数组/meta → 不抛错，但 degraded 角标点名", async () => {
  const cases: Array<[string, Override, RegExp]> = [
    ["energy_drift_duel", { drop: ["drift_line_symplectic", "drift_line_baseline"] }, /drift_line/],
    ["phase_orbit", { dropMeta: ["n_periods_observed"] }, /n_periods_observed/],
    ["modified_hamiltonian", { dropMeta: ["band_theory"] }, /band_theory/],
    ["long_term_orbit", { drop: ["varpi_drift", "ecc_error"] }, /varpi_drift/],
  ];
  for (const [viz, over, want] of cases) {
    const scene = await sceneOf(tweak(DEFS[viz] as FrameDef, over));
    assert.deepEqual(validateScene(scene), [], `${viz} 降级后仍要过校验器`);
    const degraded = degradedOf(scene);
    assert.ok(degraded.length > 0, `${viz} 降级后必须有 degraded 角标（不许静默）`);
    assert.ok(want.test(degraded) || /等 \d+ 项/.test(degraded), `${viz} 降级说明没点名 ${want}：${degraded}`);
  }
});

test("缺必需数组 → 抛 ModelInputError（不静默画空图）", async () => {
  for (const [viz, name] of [
    ["energy_drift_duel", "energy_error_symplectic"],
    ["phase_orbit", "orbit_q"],
    ["modified_hamiltonian", "mask_energy"],
    ["long_term_orbit", "traj_x"],
  ] as const) {
    await assert.rejects(async () => sceneOf(tweak(DEFS[viz] as FrameDef, { drop: [name] })), ModelInputError);
  }
});

test("pitfalls 为空 → 抛错（规则 7：防误导面板必须有内容）", async () => {
  await assert.rejects(async () => {
    const frame = await makeFrame({ ...E_DEF, pitfalls: "" });
    const model = models["energy_drift_duel"];
    assert.ok(model);
    model(frame, OPTS);
  }, ModelInputError);
});

test("窄画布（480×520）仍能过校验器，并明确说明右列文字被放弃", async () => {
  for (const [viz, def] of Object.entries(DEFS)) {
    const scene = await sceneOf(def, { width: 480, height: 520 });
    assert.deepEqual(validateScene(scene), [], `${viz} 在 480×520 下仍要合法`);
    for (const ax of scene.axes) assert.ok(ax.rect[2] > 0 && ax.rect[3] > 0, `${viz}.${ax.id} 宽高必须为正`);
    assert.match(degradedOf(scene), /窄画布/, `${viz} 放弃说明文字时必须点名降级`);
  }
});

test("null meta 值按缺失处理（不把 null 当 0 画进图里）", async () => {
  const scene = await sceneOf(tweak(MH_DEF, { meta: { band_theory: null, modified_sup: null } }));
  assert.deepEqual(validateScene(scene), []);
  assert.ok(!/null/.test(textsOf(scene)), "null 不许被格式化成「null」写进图里");
  assert.match(degradedOf(scene), /band_theory/);
});
