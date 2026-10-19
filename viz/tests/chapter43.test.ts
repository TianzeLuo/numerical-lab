/**
 * 4.3 章（守恒律与有限体积）model 层测试 —— **自足**：不依赖 Python，也不依赖
 * `data/frames` 是否存在。夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，
 * 字段集合（数组名 / dtype / shape / label、meta / strings / critical 键）逐字照抄
 * `server/chapters/4.3.py` 的真实帧（数值是测试用的合成值）。
 *
 * 这一页最容易骗人的四件事，逐条钉在测试里：
 *
 * 1. **掩码点是占位值，不是数据**：`error_floor_mask=1` 的点在夹具里填 555/888；
 *    模型一旦把它们画进折线，测试当场抓到（对数轴会溢出，且数组里出现哨兵）。
 * 2. **无效参照不画**：`exact_valid=0`（破散后 / t=0）时，精确解与误差曲线整条不画，
 *    只留文字说明；绝不能拿多值解或未定义的自相似解算“误差”。
 * 3. **无交点的特征线不是“交于 4.5e14”**：`cross_mask=0` 的点携带巨大占位值，
 *    模型只画掩码命中的交点。
 * 4. **波位置要按 kind 上色**：Sod 的左波是稀疏波（灰点），不是激波（红实线）。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Layer, type Scene } from "../shared/src/scene.js";
import { manifest, modelFor, models } from "../chapters/4.3/index.js";

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
  "这张图会怎么骗你：夹具里的数字是合成的，但字段集合、掩码口径与降级路径与真实帧完全一致。";
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
        chapter: "4.3",
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

const sceneOf = async (def: FrameDef, opts: Partial<ModelOptions> = {}): Promise<Scene> => {
  const model = models[def.viz]!;
  return model(await makeFrame(def), { ...OPTS, ...opts });
};

const layerArrays = (scene: Scene): number[] => {
  const out: number[] = [];
  for (const l of scene.layers) {
    if (l.kind === "line") for (const s of l.series) out.push(...Array.from(s.y as ArrayLike<number>));
    else if (l.kind === "scatter") out.push(...Array.from(l.y as ArrayLike<number>));
    else if (l.kind === "heat") out.push(...Array.from(l.values as ArrayLike<number>));
    else if (l.kind === "bars") out.push(...Array.from(l.values as ArrayLike<number>));
  }
  return out;
};
const lineSeriesOn = (scene: Scene, axis: string): { x: ArrayLike<number>; y: ArrayLike<number> }[] =>
  scene.layers
    .filter((l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === axis)
    .flatMap((l) => l.series.map((s) => ({ x: s.x, y: s.y })));
const scatterOn = (scene: Scene, axis: string): { x: ArrayLike<number>; y: ArrayLike<number> }[] =>
  scene.layers
    .filter((l): l is Extract<Layer, { kind: "scatter" }> => l.kind === "scatter" && l.axes === axis)
    .map((l) => ({ x: l.x, y: l.y }));
const allText = (scene: Scene): string =>
  scene.texts.map((t) => t.text).join("\n") + "\n" + scene.badges.map((b) => b.text).join("\n");
const assertValid = (scene: Scene): void => {
  assert.deepEqual(validateScene(scene), []);
};

const BURGERS_STRINGS = {
  time_semantics: "真实时间 t：从初值推进到 t。",
  nonfinite_policy: "对数轴 0 用 1e-17 占位并配掩码。",
  scheme_cn: "Godunov（精确通量）",
  scheme: "godunov",
  limiter_requested: "none",
  limiter_effective: "none",
  initial: "右行激波 u_L=1,u_R=0",
  ic: "shock",
  bc: "outflow",
  floor: "1e-17",
  flux_window: "通量窗口 ±0.75",
  limiter_note: "limiter=none 时 MUSCL 与一阶相同。",
  reading_note: "左图看解，中图看通量进出，右图看误差。",
  mask_note: "掩码=1 的点是地板占位。",
  theory_note: "Godunov 通量取 x/t=0 处的精确解。",
};

const BURGERS_META: Record<string, number | boolean | null> = {
  t: 0.3,
  t_actual: 0.3,
  n: 16,
  cfl: 0.4,
  limiter_requested_index: 0,
  steps: 120,
  dt: 0.0025,
  dx: 0.0625,
  rh_speed: 0.5,
  speed_measured: 0.5001,
  speed_rel_err: 2e-4,
  speed_fit_r2: 0.9999,
  speed_fit_n: 121,
  mass0: 0.5,
  mass_end: 0.65,
  mass_in_cum: 0.15,
  mass_out_cum: 0,
  max_defect: 1e-16,
  rel_defect: 2e-16,
  entropy_production: -0.01,
  tv0: 1,
  tv_end: 1,
  max_u: 1,
  min_u: 0,
  max_jump: 0.5,
  error_l1: 0.003,
  error_max: 0.01,
  exact_t_break: null,
  exact_valid: true,
  exact_t_invalid: false,
  overflow: false,
  n_clipped: 0,
  error_floor: 1e-17,
  u_clip: 2.5,
  roundoff: 1e-16,
};

const BURGERS_N = 16;
const BURGERS_X = Array.from({ length: BURGERS_N }, (_, i) => (i + 0.5) / BURGERS_N);
const BURGERS_U = BURGERS_X.map((x) => (x < 0.65 ? 1 : 0));
const BURGERS_FLUX: number[] = BURGERS_X.map((x) => (x < 0.62 ? 0.25 : x > 0.68 ? 0 : 0.125));
const BURGERS_NET = BURGERS_FLUX.map((_, i) => {
  const fin = BURGERS_FLUX[Math.max(0, i - 1)] as number;
  const fout = BURGERS_FLUX[Math.min(BURGERS_N - 1, i)] as number;
  return -(fout - fin);
});

function burgersDef(overrides: Partial<FrameDef> = {}): FrameDef {
  const errorAbs = BURGERS_X.map((_, i) => (i < 10 ? 0.01 : 555));
  const errorMask = BURGERS_X.map((_, i) => (i < 10 ? 0 : 1));
  return {
    viz: "burgers_shock",
    arrays: [
      ["cell_x", BURGERS_X, "单元中心 x"],
      ["u", BURGERS_U, "数值解"],
      ["u_clipped", BURGERS_X.map(() => 0), "裁剪掩码"],
      ["u_exact", BURGERS_U, "精确解"],
      ["exact_valid", BURGERS_X.map(() => 1), "精确解有效"],
      ["exact_t_clamped", BURGERS_X.map(() => 0), "t=0 极限"],
      ["flux_in", BURGERS_FLUX, "左界面通量"],
      ["flux_out", BURGERS_FLUX, "右界面通量"],
      ["flux_net", BURGERS_NET, "净流入"],
      ["flux_clipped", BURGERS_X.map(() => 0), "通量裁剪"],
      ["error_abs", errorAbs, "逐点误差"],
      ["error_floor_mask", errorMask, "地板/无效掩码"],
      ["residual_time", [0, 0.1, 0.2, 0.3], "账本时间"],
      ["residual_mass", [1e-17, 2e-16, 1e-16, 1e-17], "账本残差"],
      ["residual_floor_mask", [1, 0, 0, 1], "账本地板掩码"],
    ],
    meta: { ...BURGERS_META },
    strings: { ...BURGERS_STRINGS },
    critical: { cfl: 1 },
    ...overrides,
  };
}

// --------------------------------------------------------------------------- //
// burgers_shock
// --------------------------------------------------------------------------- //

test("4.3 章登记表：四个 viz 都有 model，manifest 数量一致", () => {
  assert.deepEqual(Object.keys(models).sort(), [
    "burgers_shock",
    "characteristics",
    "conservation_ledger",
    "riemann_fan",
  ]);
  assert.equal(manifest.vizzes.length, 4);
  for (const v of manifest.vizzes) assert.notEqual(modelFor(v.id), null);
  assert.equal(modelFor("nope"), null);
});

test("burgers：四块轴、固定窗口与自适应残差轴的口径与 manifest 一致", async () => {
  const scene = await sceneOf(burgersDef());
  assertValid(scene);
  assert.deepEqual(scene.axes.map((a) => a.id), ["solution", "flux", "error", "residual"]);
  const byId = new Map(scene.axes.map((a) => [a.id, a]));
  assert.equal(byId.get("solution")?.autoscaled, false);
  assert.equal(byId.get("flux")?.autoscaled, false);
  assert.equal(byId.get("error")?.autoscaled, false);
  assert.equal(byId.get("residual")?.autoscaled, true);
  assert.deepEqual(byId.get("solution")?.ylim, [-1.2, 1.33]);
  assert.deepEqual(byId.get("flux")?.ylim, [-0.75, 0.75]);
  assert.equal(byId.get("error")?.yscale, "log");
  assert.equal(scene.t, 0.3);
});

test("burgers：掩码点（哨兵 555）绝不进误差图层", async () => {
  const scene = await sceneOf(burgersDef());
  const y = lineSeriesOn(scene, "error").flatMap((s) => Array.from(s.y));
  assert.ok(!y.includes(555), "555 是 error_floor_mask=1 的占位值，被画进了误差轴");
  assert.ok(y.some((v) => Math.abs(v - 0.01) < 1e-12), "有效误差点 0.01 应当被画出来");
});

test("burgers：exact_valid=0 时精确解与误差整条不画", async () => {
  const def = burgersDef();
  const arrays = def.arrays.map(([name, data, label]) => {
    if (name === "exact_valid") return [name, BURGERS_X.map(() => 0), label] as Row;
    if (name === "u_exact") return [name, BURGERS_X.map(() => 777), label] as Row;
    if (name === "error_abs") return [name, BURGERS_X.map(() => 888), label] as Row;
    if (name === "error_floor_mask") return [name, BURGERS_X.map(() => 1), label] as Row;
    return [name, data, label] as Row;
  });
  const scene = await sceneOf({ ...def, arrays, meta: { ...BURGERS_META, exact_valid: false, exact_t_break: 0.31831 } });
  assertValid(scene);
  const solY = lineSeriesOn(scene, "solution").flatMap((s) => Array.from(s.y));
  assert.ok(!solY.includes(777), "无效的精确解（哨兵 777）不应进解面板");
  const errY = lineSeriesOn(scene, "error").flatMap((s) => Array.from(s.y));
  assert.ok(!errY.includes(888), "无效误差（哨兵 888）不应进误差轴");
  assert.match(allText(scene), /参照无效/);
});

test("burgers：显示裁剪（u=2.5）触发 overflow 角标，而不是静默丢点", async () => {
  const arrays = burgersDef().arrays.map(([name, data, label]) =>
    name === "u" ? ([name, [...BURGERS_U.slice(0, 15), 2.5], label] as Row) : ([name, data, label] as Row),
  );
  const scene = await sceneOf(burgersDef({ arrays }));
  assert.equal(scene.overflow.flag, true);
  assert.ok(scene.badges.some((b) => b.kind === "overflow"));
});

test("burgers：缺 flux_net 抛 ModelInputError（不画半张图）", async () => {
  const arrays = burgersDef().arrays.filter(([name]) => name !== "flux_net");
  await assert.rejects(() => sceneOf(burgersDef({ arrays })), ModelInputError);
});

test("burgers：480×520 窄画布可渲染并显式降级", async () => {
  const scene = await sceneOf(burgersDef(), { width: 480, height: 520 });
  assertValid(scene);
  assert.match(allText(scene), /降级/);
});

// --------------------------------------------------------------------------- //
// characteristics
// --------------------------------------------------------------------------- //

const CH_STRINGS = {
  time_semantics: "真实时间 t：特征线画满 [0, tMax]。",
  nonfinite_policy: "对数轴 0 用 1e-17 占位。",
  profile: "sine",
  floor: "1e-17",
  reading_note: "相交=激波、发散=稀疏波。",
  mask_note: "exact_valid=0 时回溯解多值。",
  theory_note: "Burgers 的特征线 x=x0+t*u0(x0)。",
};

const CH_META: Record<string, number | boolean | null> = {
  t: 0.3,
  amplitude: 0.5,
  n_lines: 4,
  t_view: 0.8,
  t_break_closed: 0.3183098862,
  t_break_scan: 0.3183098862,
  t_break_lines: 0.32,
  t_break_scan_rel: 1e-4,
  first_pair: 0,
  x_break: 0.45,
  n_crossing: 2,
  n_diverging: 1,
  exact_valid: true,
  exact_t_invalid: false,
  tv_now: 2,
  max_u: 0.5,
  min_u: -0.5,
  max_defect: 1e-16,
  error_l1: 0.001,
  error_max: 0.01,
  overflow: false,
  n_clipped: 0,
  error_floor: 1e-17,
  u_clip: 2.5,
  roundoff: 1e-16,
};

const CH_NEX = 20;
const CH_PX = Array.from({ length: CH_NEX }, (_, i) => i / (CH_NEX - 1));
const CH_PU = CH_PX.map((x) => 0.5 * Math.sin(2 * Math.PI * x));

function characteristicsDef(overrides: Partial<FrameDef> = {}): FrameDef {
  const error = CH_PX.map((_, i) => (i % 3 === 0 ? 555 : 0.01));
  return {
    viz: "characteristics",
    arrays: [
      ["line_x0", [0.1, 0.35, 0.6, 0.85], "特征线起点"],
      ["line_u0", [0.5, 0, -0.5, 0.2], "特征速度"],
      ["line_x_end", [0.5, 0.35, 0.2, 1.01], "终点"],
      ["cross_t", [0.2, 4.5036e14, 0.35], "交点时刻"],
      ["cross_x", [0.45, 1e14, 0.65], "交点位置"],
      ["cross_mask", [1, 0, 1], "交点掩码"],
      ["break_line_t", [0.25, 0.25], "破散参考线"],
      ["slice_t", [0.3, 0.3], "当前切线"],
      ["profile_x", CH_PX, "剖面 x"],
      ["profile_u", CH_PU, "数值剖面"],
      ["profile_u_clipped", CH_PX.map(() => 0), "剖面裁剪"],
      ["profile_exact", CH_PU, "回溯精确解"],
      ["exact_valid", CH_PX.map(() => 1), "精确解有效"],
      ["error_abs", error, "误差"],
      ["error_floor_mask", CH_PX.map((_, i) => (i % 3 === 0 ? 1 : 0)), "地板掩码"],
      ["residual_time", [0, 0.1, 0.2, 0.3], "账本时间"],
      ["residual_mass", [1e-17, 2e-16, 1e-16, 1e-17], "账本残差"],
      ["residual_floor_mask", [1, 0, 0, 1], "账本地板掩码"],
    ],
    meta: { ...CH_META },
    strings: { ...CH_STRINGS },
    ...overrides,
  };
}

test("characteristics：占位交点 4.5e14 与 1e14 绝不被画出来", async () => {
  const scene = await sceneOf(characteristicsDef());
  assertValid(scene);
  const sc = scatterOn(scene, "characteristics").flatMap((s) => [...Array.from(s.x), ...Array.from(s.y)]);
  assert.ok(!sc.some((v) => v > 1e6), `占位交点进了图层：${sc.filter((v) => v > 1e6)}`);
});

test("characteristics：只有掩码命中的交点被画出（2 个交点 + 1 个破散点）", async () => {
  const scene = await sceneOf(characteristicsDef());
  const scatters = scatterOn(scene, "characteristics");
  const pts = scatters.flatMap((s) => Array.from(s.y));
  assert.equal(pts.length, 3, `散点数 ${pts.length}：应为 2 个交点 + 1 个破散点`);
});

test("characteristics：当前 t 与破散 t_b 两条参考线真画", async () => {
  const scene = await sceneOf(characteristicsDef());
  const flat = lineSeriesOn(scene, "characteristics").filter(
    (s) => s.x.length === 2 && (s.x[0] as number) === 0 && (s.x[1] as number) === 1,
  );
  const ys = flat.map((s) => s.y[0] as number);
  assert.ok(ys.some((v) => Math.abs(v - 0.3) < 1e-12), "缺当前 t=0.3 的水平线");
  assert.ok(ys.some((v) => Math.abs(v - 0.32) < 1e-12), "缺破散 t_b 的水平线");
});

test("characteristics：破散后不画精确解与误差（哨兵 777/888）", async () => {
  const arrays = characteristicsDef().arrays.map(([name, data, label]) => {
    if (name === "exact_valid") return [name, CH_PX.map(() => 0), label] as Row;
    if (name === "profile_exact") return [name, CH_PX.map(() => 777), label] as Row;
    if (name === "error_abs") return [name, CH_PX.map(() => 888), label] as Row;
    if (name === "error_floor_mask") return [name, CH_PX.map(() => 1), label] as Row;
    return [name, data, label] as Row;
  });
  const scene = await sceneOf({
    ...characteristicsDef(),
    arrays,
    meta: { ...CH_META, exact_valid: false },
  });
  assertValid(scene);
  const py = lineSeriesOn(scene, "profile").flatMap((s) => Array.from(s.y));
  assert.ok(!py.includes(777));
  const ey = lineSeriesOn(scene, "error").flatMap((s) => Array.from(s.y));
  assert.ok(!ey.includes(888));
  assert.match(allText(scene), /参照无效/);
});

test("characteristics：缺 cross_mask 抛 ModelInputError", async () => {
  const arrays = characteristicsDef().arrays.filter(([name]) => name !== "cross_mask");
  await assert.rejects(() => sceneOf(characteristicsDef({ arrays })), ModelInputError);
});

test("characteristics：480×520 窄画布可渲染并降级", async () => {
  const scene = await sceneOf(characteristicsDef(), { width: 480, height: 520 });
  assertValid(scene);
  assert.match(allText(scene), /降级/);
});

// --------------------------------------------------------------------------- //
// conservation_ledger
// --------------------------------------------------------------------------- //

const LEDGER_STRINGS = {
  time_semantics: "真实时间 t：三本账都推进到 t。",
  nonfinite_policy: "对数轴 0 用 1e-17 占位。",
  bc: "outflow",
  floor: "1e-17",
  reading_note: "质量该不该变由边界决定，残差判格式。",
  mask_note: "地板=舍入，不是残差为零。",
  speed_note: "非守恒的波前原地不动。",
  theory_note: "守恒格式的总量变化只由边界通量决定。",
};

const LEDGER_META: Record<string, number | boolean | null> = {
  t: 0.3,
  n: 16,
  cfl: 0.4,
  mass0: 0.5,
  mass_end_godunov: 0.65,
  mass_end_muscl: 0.65,
  mass_end_advective: 0.5,
  defect_godunov: 1e-16,
  defect_muscl: 1.4e-16,
  defect_advective: 0.15,
  rel_defect_godunov: 2e-16,
  rel_defect_muscl: 2e-16,
  rel_defect_advective: 0.3,
  leak_total: 0.15,
  inflow_cum: 0.15,
  outflow_cum: 0,
  rh_speed: 0.5,
  speed_godunov: 0.5001,
  speed_advective: 0,
  speed_rel_err_godunov: 2e-4,
  speed_rel_err_advective: 1,
  steps_godunov: 120,
  steps_muscl: 120,
  steps_advective: 120,
  entropy_production_godunov: -0.026,
  entropy_production_advective: -0.1,
  residual_floor: 1e-17,
  roundoff: 1e-16,
  overflow: false,
};

function ledgerDef(overrides: Partial<FrameDef> = {}): FrameDef {
  const time = [0, 0.1, 0.2, 0.3];
  return {
    viz: "conservation_ledger",
    arrays: [
      ["time_godunov", time, "G 时间"],
      ["mass_raw_godunov", [0.5, 0.55, 0.6, 0.65], "G 原始质量"],
      ["defect_godunov", [1e-17, 2e-16, 1e-16, 1e-17], "G 残差"],
      ["defect_floor_mask_godunov", [1, 0, 0, 1], "G 地板掩码"],
      ["time_muscl", time, "M 时间"],
      ["mass_raw_muscl", [0.5, 0.55, 0.6, 0.65], "M 原始质量"],
      ["defect_muscl", [1e-17, 2.4e-16, 1e-16, 1e-17], "M 残差"],
      ["defect_floor_mask_muscl", [1, 0, 0, 1], "M 地板掩码"],
      ["time_advective", time, "A 时间"],
      ["mass_raw_advective", [0.5, 0.5, 0.5, 0.5], "A 原始质量"],
      ["defect_advective", [0, 0.05, 0.1, 0.15], "A 残差"],
      ["defect_floor_mask_advective", [1, 0, 0, 0], "A 地板掩码"],
      ["mass_time", time, "公共时间"],
      ["mass_godunov", [0.5, 0.55, 0.6, 0.65], "G 质量"],
      ["mass_muscl", [0.5, 0.55, 0.6, 0.65], "M 质量"],
      ["mass_advective", [0.5, 0.5, 0.5, 0.5], "A 质量"],
      ["inflow_cum", [0, 0.05, 0.1, 0.15], "累计流入"],
      ["outflow_cum", [0, 0, 0, 0], "累计流出"],
      ["shock_t_godunov", time, "G 激波时间"],
      ["shock_x_godunov", [0.5, 0.55, 0.6, 999], "G 激波位置"],
      ["shock_mask_godunov", [1, 1, 1, 0], "G 激波掩码"],
      ["shock_t_advective", time, "A 激波时间"],
      ["shock_x_advective", [0.5, 0.5, 0.5, 0.5], "A 激波位置"],
      ["shock_mask_advective", [1, 1, 1, 1], "A 激波掩码"],
      ["rh_line_t", [0, 0.3], "RH 时间"],
      ["rh_line_x", [0.5, 0.65], "RH 位置"],
      ["residual_ledger", [1e-16, 1.4e-16, 0.15], "残差汇总"],
      ["residual_ledger_floor_mask", [0, 0, 0], "残差汇总掩码"],
    ],
    meta: { ...LEDGER_META },
    strings: { ...LEDGER_STRINGS },
    ...overrides,
  };
}

test("ledger：三块轴、账本残差轴是 log 且固定窗口", async () => {
  const scene = await sceneOf(ledgerDef());
  assertValid(scene);
  assert.deepEqual(scene.axes.map((a) => a.id), ["mass", "shock", "defect"]);
  const defect = scene.axes.find((a) => a.id === "defect")!;
  assert.equal(defect.yscale, "log");
  assert.equal(defect.autoscaled, false);
  assert.deepEqual(defect.ylim, [1e-18, 10]);
});

test("ledger：激波位置数组里 mask=0 的哨兵 999 不进图层", async () => {
  const scene = await sceneOf(ledgerDef());
  const y = lineSeriesOn(scene, "shock").flatMap((s) => Array.from(s.y));
  assert.ok(!y.includes(999), "mask=0 的占位激波位置被画了出来");
});

test("ledger：requiredAux=residual 真的画在 defect 轴上", async () => {
  const scene = await sceneOf(ledgerDef());
  const defectLines = lineSeriesOn(scene, "defect");
  assert.ok(defectLines.length >= 3, `defect 轴只有 ${defectLines.length} 条线，应为三条账本`);
  const text = allText(scene);
  assert.match(text, /bc=outflow/);
  assert.match(text, /M\(0\)/);
});

test("ledger：缺 defect_advective 抛 ModelInputError", async () => {
  const arrays = ledgerDef().arrays.filter(([name]) => name !== "defect_advective");
  await assert.rejects(() => sceneOf(ledgerDef({ arrays })), ModelInputError);
});

test("ledger：480×520 窄画布可渲染并降级", async () => {
  const scene = await sceneOf(ledgerDef(), { width: 480, height: 520 });
  assertValid(scene);
  assert.match(allText(scene), /降级/);
});

// --------------------------------------------------------------------------- //
// riemann_fan
// --------------------------------------------------------------------------- //

const RIEMANN_STRINGS = {
  time_semantics: "真实时间 t：t=0 时自相似解没有定义。",
  nonfinite_policy: "对数轴 0 用 1e-17 占位。",
  problem: "Sod：激波 + 接触间断 + 稀疏波",
  scheme: "hllc",
  limiter: "minmod",
  left_wave: "rarefaction",
  right_wave: "shock",
  scheme_note: "未触发 n 上限",
  floor: "1e-17",
  wave_note: "2=激波 1=稀疏边界 0=接触。",
  mask_note: "t=0 时误差整条无效。",
  theory_note: "p* 由单调方程二分定出。",
};

const RIEMANN_META: Record<string, number | boolean | null> = {
  t: 0.12,
  t_requested: 0.12,
  t_clamped: false,
  n: 20,
  n_requested: 20,
  steps: 50,
  dt: 0.0024,
  p_star: 0.30313018,
  u_star: 0.92745262,
  rho_star_l: 0.42631943,
  rho_star_r: 0.26557371,
  s_left: -0.07,
  s_right: 1.75,
  head_l: -1.18,
  tail_l: -0.07,
  head_r: 1.75,
  tail_r: 1.75,
  contact_x: 0.61,
  vacuum: false,
  star_iterations: 1,
  l1_rho: 0.0026,
  l1_u: 0.0041,
  l1_p: 0.002,
  max_err_rho: 0.0695,
  min_rho: 0.125,
  min_p: 0.1,
  max_defect: 1e-16,
  entropy_ratio_shock: 1.0555,
  window_rho_lo: 0.125,
  window_rho_hi: 1,
  window_p_lo: 0.1,
  window_p_hi: 1,
  overflow: false,
  error_floor: 1e-17,
  euler_clip: 5,
  roundoff: 1e-16,
};

const R_N = 20;
const R_X = Array.from({ length: R_N }, (_, i) => (i + 0.5) / R_N);
const R_RHO = R_X.map((x) => (x < 0.61 ? 0.426 : 0.266));
const R_U = R_X.map((x) => (x < 0.36 || x > 0.71 ? 0 : 0.927));
const R_P = R_X.map((x) => (x < 0.61 ? 0.303 : 0.1));

function riemannDef(overrides: Partial<FrameDef> = {}): FrameDef {
  const err = R_X.map((_, i) => (i % 4 === 0 ? 555 : 0.01));
  const emask = R_X.map((_, i) => (i % 4 === 0 ? 1 : 0));
  return {
    viz: "riemann_fan",
    arrays: [
      ["x", R_X, "坐标"],
      ["rho_exact", R_RHO, "精确密度"],
      ["u_exact", R_U, "精确速度"],
      ["p_exact", R_P, "精确压强"],
      ["rho_num", R_RHO, "数值密度"],
      ["u_num", R_U, "数值速度"],
      ["p_num", R_P, "数值压强"],
      ["overflow_mask", R_X.map(() => 0), "裁剪掩码"],
      ["exact_valid", R_X.map(() => 1), "精确解有效"],
      ["error_rho", err, "rho 误差"],
      ["error_u", err, "u 误差"],
      ["error_p", err, "p 误差"],
      ["error_floor_mask", emask, "地板掩码"],
      ["wave_x", [0.36, 0.49, 0.61, 0.71, 0.71], "波位置"],
      ["wave_kind", [1, 1, 0, 2, 2], "波型"],
      ["residual_time", [0, 0.04, 0.08, 0.12], "账本时间"],
      ["residual_mass", [1e-17, 1e-16, 1e-16, 1e-17], "账本残差"],
      ["residual_floor_mask", [1, 0, 0, 1], "账本地板掩码"],
    ],
    meta: { ...RIEMANN_META },
    strings: { ...RIEMANN_STRINGS },
    ...overrides,
  };
}

test("riemann：五块轴、固定窗口与 log 误差轴", async () => {
  const scene = await sceneOf(riemannDef());
  assertValid(scene);
  assert.deepEqual(scene.axes.map((a) => a.id), ["rho", "velocity", "pressure", "error", "residual"]);
  for (const id of ["rho", "velocity", "pressure", "error"]) {
    assert.equal(scene.axes.find((a) => a.id === id)?.autoscaled, false);
  }
  assert.equal(scene.axes.find((a) => a.id === "residual")?.autoscaled, true);
  assert.deepEqual(scene.axes.find((a) => a.id === "velocity")?.ylim, [-0.6, 1]);
});

test("riemann：2 条重叠激波位线去重后只剩 1 条（共 4 个唯一位置）", async () => {
  const scene = await sceneOf(riemannDef());
  const vertical = lineSeriesOn(scene, "rho").filter(
    (s) => s.x.length === 2 && (s.x[0] as number) === (s.x[1] as number),
  );
  const xs = new Set(vertical.map((s) => (s.x[0] as number).toFixed(9)));
  assert.equal(xs.size, 4, `波位竖线应去重到 4 条，实际 ${[...xs]}`);
});

test("riemann：Sod 稀疏波边界与接触、激波的颜色按 kind 区分", async () => {
  const scene = await sceneOf(riemannDef());
  const vertical = scene.layers
    .filter((l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === "rho")
    .flatMap((l) => l.series)
    .filter((s) => s.x.length === 2 && (s.x[0] as number) === (s.x[1] as number));
  const colors = new Set(vertical.map((s) => s.color));
  assert.equal(colors.size, 3, `三种波型应有三种颜色，实际 ${[...colors]}`);
});

test("riemann：掩码点（哨兵 555）绝不进误差轴", async () => {
  const scene = await sceneOf(riemannDef());
  const y = lineSeriesOn(scene, "error").flatMap((s) => Array.from(s.y));
  assert.ok(!y.includes(555));
});

test("riemann：t=0 时精确解与误差都无效，只画初始间断与文字说明", async () => {
  const arrays = riemannDef().arrays.map(([name, data, label]) => {
    if (name === "exact_valid") return [name, R_X.map(() => 0), label] as Row;
    if (name === "error_rho" || name === "error_u" || name === "error_p") {
      return [name, R_X.map(() => 888), label] as Row;
    }
    if (name === "error_floor_mask") return [name, R_X.map(() => 1), label] as Row;
    return [name, data, label] as Row;
  });
  const scene = await sceneOf({
    ...riemannDef(),
    arrays,
    meta: { ...RIEMANN_META, t: 0, t_requested: 0, t_clamped: true },
  });
  assertValid(scene);
  const ey = lineSeriesOn(scene, "error").flatMap((s) => Array.from(s.y));
  assert.ok(!ey.includes(888), "t=0 的误差整条无效，不应进图层");
  assert.match(allText(scene), /t=0/);
  assert.match(allText(scene), /误差整条无效/);
});

test("riemann：缺 error_p 抛 ModelInputError", async () => {
  const arrays = riemannDef().arrays.filter(([name]) => name !== "error_p");
  await assert.rejects(() => sceneOf(riemannDef({ arrays })), ModelInputError);
});

test("riemann：480×520 窄画布可渲染并降级", async () => {
  const scene = await sceneOf(riemannDef(), { width: 480, height: 520 });
  assertValid(scene);
  assert.match(allText(scene), /降级/);
});

test("四个 viz 在 960×600 的图层数组里没有 NaN/Inf", async () => {
  const defs = [burgersDef(), characteristicsDef(), ledgerDef(), riemannDef()];
  for (const def of defs) {
    const scene = await sceneOf(def);
    for (const v of layerArrays(scene)) assert.ok(Number.isFinite(v), `${def.viz} 图层出现非有限值 ${v}`);
  }
});
