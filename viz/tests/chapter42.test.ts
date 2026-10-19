/**
 * 4.2 章（双曲型方程：色散、耗散与限制器）model 层测试 —— **自足**：不依赖 Python，
 * 也不依赖 `data/frames` 是否存在。夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，
 * 字段集合（数组名 / dtype / shape / unit / label、meta / strings / critical 键）逐字照抄
 * `server/chapters/4.2.py` 的真实帧（数值是测试用的合成值）。
 *
 * 本文件守的是这一章最容易画反、也最容易读反的几件事：
 *
 * 1. **固定窗口是契约**：解/误差/相速度/振幅/Sweby/TV 六个轴都 `autoscaled=false`，
 *    范围写死；换格式、换帧不许让自适应把幅度差拉平。越窗由溢出角标与正文明说。
 * 2. **Sweby 区域是画出来的边界**：绿带的 yhi 必须逐点等于帧的 `sweby_upper`（min(2r,2)），
 *    ylo 等于 `sweby_lower`；r<0 的「φ 必须为 0」单独一条竖带。TVD 区间上界单调不减。
 * 3. **相位误差带符号**：ΔΦ 轴是线性、带零线；ν→0 的帧（c_num≡1）必须正好落在 c=1 上，
 *    而不是被自适应放大成一条「误差曲线」。
 * 4. **耗散 ≠ 色散**：长时波包要有幅值标尺（同一 x 上两条包络的落差）与固定的
 *    前缘/拖尾取样带；`dissipative`/`dispersive` 两个布尔只从 meta 读，不自己分类。
 * 5. **降级不静默**：缺可选数组/meta → degraded 角标点名；缺必需数组 → ModelInputError。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import { manifest, modelFor, models } from "../chapters/4.2/index.js";

type Row = readonly [name: string, data: ArrayLike<number>, label: string, shape?: readonly number[]];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical: Record<string, number>;
  pitfalls?: string;
}

const PITFALLS =
  "它会怎样骗你：第一，固定窗口外不是没有数据，是越窗；第二，0 值不画不是缺失，是该点没有过冲；" +
  "第三，限制器还在不等于 TVD 成立；第四，包络变矮是耗散，拖尾是色散，不要用一个词概括。";

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
      unit: "1",
      label,
    });
    arrays.set(name, typed);
  }
  return decodeNlf(
    await encodeNlf(
      new Frame({
        chapter: "4.2",
        viz: def.viz,
        kind: "frame",
        codec: "none",
        meta: def.meta,
        strings: def.strings,
        critical: def.critical,
        pitfalls: def.pitfalls ?? PITFALLS,
        specs,
        arrays,
      }),
    ),
  );
}

interface Override {
  arrays?: Record<string, ArrayLike<number>>;
  critical?: Record<string, number>;
  meta?: Record<string, number | boolean | null>;
  strings?: Record<string, string>;
  drop?: readonly string[];
  dropMeta?: readonly string[];
  dropStrings?: readonly string[];
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
  const strings = { ...def.strings };
  for (const k of ov.dropStrings ?? []) delete strings[k];
  const critical = { ...def.critical, ...ov.critical };
  return { ...def, arrays, meta: { ...meta, ...ov.meta }, strings: { ...strings, ...ov.strings }, critical };
}

// --------------------------------------------------------------------------- //
// 夹具
// --------------------------------------------------------------------------- //

const N = 16;
const x16 = Array.from({ length: N }, (_, i) => i / N);
const zeros = (n: number): Float64Array => new Float64Array(n);
const ones = (n: number): Float64Array => new Float64Array(n).fill(1);
const stepInit = Array.from({ length: N }, (_, i) => (i < 4 ? 1 : 0));
const stepU = Array.from({ length: N }, (_, i) => (i < 3 ? 1 : i === 3 ? 0.8 : i === 4 ? 0.2 : 0));
const stepErr = stepU.map((v, i) => Math.max(1e-17, Math.abs(v - (stepInit[i] as number))));
const forward = stepU.map((v, i) => Math.max(v - (stepInit[i] as number), 0));
const backward = stepU.map((v, i) => Math.max((stepInit[i] as number) - v, 0));

const SZ_DEF: FrameDef = {
  viz: "scheme_zoo",
  arrays: [
    ["x", x16, "格点 x"],
    ["u", stepU, "数值解"],
    ["u_clipped", zeros(N), "1 = 该点被截断显示"],
    ["u_exact", stepInit, "精确平移解"],
    ["u_initial", stepInit, "初值"],
    ["error_curve", stepErr, "|u − 精确解|（对数轴，地板 1e-17）"],
    ["error_at_floor", zeros(N), "1 = 该点误差已到地板"],
    ["forward_error", forward, "正向误差 max(u−u_exact,0)"],
    ["backward_error", backward, "负向误差 max(u_exact−u,0)"],
    ["algebraic_residual", [1e-4], "最后一步的代数残差"],
    ["residual_profile", Array.from({ length: N }, (_, i) => (i === 0 ? 1e-3 : 1e-17)), "RK 回代残差"],
    ["integrator_gap", Array.from({ length: N }, (_, i) => (i === 0 ? 2e-5 : 1e-17)), "euler 与 rk3 之差"],
  ],
  meta: {
    scheme_index: 0, nu: 0.4, nx: N, t_requested: 0.2, t_actual: 0.2, n_steps: 8, dx: 1 / N,
    umax: 1, umin: 0, overflow: false, clipped_points: 0, linf_error: 0.2, l2_error: 0.1,
    overshoot: 0, undershoot: 0, total_variation: 1.8, tv_initial: 2, max_abs_symbol: 1,
    amplified_per_step: false, nonzero_initial_data: true,
  },
  strings: {
    scheme: "upwind", initial: "step", integrator: "rk3", floor: "1e-17",
    time_semantics: "真实时间 t。", nonfinite_policy: "0 与无穷用 1e-17 地板占位。",
    reading_note: "看变矮还是长尾巴。", data_source: "backend",
  },
  critical: { nu: 0.5 },
};

const P = 11;
const phi = Array.from({ length: P }, (_, i) => 1e-6 + (i / (P - 1)) * 3.0);
const cNum = phi.map((p) => 1 + 0.2 * (p / Math.PI));
const cPred = phi.map((p) => 1 + 0.1 * (p / Math.PI));
const phaseErr = cNum.map((c) => 2 * Math.PI * (c - 1));
const gAbs = phi.map((p) => 0.95 + 0.05 * Math.sin(p));

const PE_DEF: FrameDef = {
  viz: "phase_error",
  arrays: [
    ["phi", phi, "无量纲波数 φ"],
    ["c_num", cNum, "数值相速度比（解析符号）"],
    ["c_pred", cPred, "修正方程到 φ^5 的预测"],
    ["c_one", ones(P), "精确相速度 1"],
    ["phase_error", phaseErr, "每波长相位差 ΔΦ"],
    ["g_abs", gAbs, "同一波数上的 |G|"],
    ["decay", gAbs.map((g) => Math.max(1e-17, Math.abs(1 - g))), "−ln|G|（地板）"],
    ["prediction_ok", phi.map((p) => (p <= 1.2 ? 1 : 0)), "预测相对差 ≤1%"],
    ["mark_breakdown", [1.2], "预测失效波数"],
  ],
  meta: {
    scheme_index: 2, nu: 0.4, n_phi: P, d2: 0.12, a3: -0.008, d4: 0.0044, a5: -0.00075,
    viscosity_ratio: 0.3, dispersion_ratio: 0.02, max_abs_symbol: 1, c_at_phi_max: cNum[P - 1] as number,
    c_pred_at_phi_max: cPred[P - 1] as number, prediction_breakdown_phi: 1.2, series_radius: Math.PI,
    speed_ratio_at_phi_0_1: 0.9998, decay_at_phi_max: 0.05, leads_or_lags: 1,
  },
  strings: {
    scheme: "beam_warming", floor: "1e-17", time_semantics: "无时间轴（一次性扫描）。",
    nonfinite_policy: "0 与无穷用 1e-17 地板占位。", data_source: "backend",
  },
  critical: { nu: 0.5 },
};

const R = 21;
const rGrid = Array.from({ length: R }, (_, i) => (4 * i) / (R - 1));
const upper = rGrid.map((r) => Math.min(2 * r, 2));
const phiCurve = rGrid.map((r) => Math.min(r, 1));
const tvHist = [2, 1.995, 1.99];

const LI_DEF: FrameDef = {
  viz: "limiter_interaction",
  arrays: [
    ["x", x16, "格点 x"],
    ["u", stepU, "MUSCL 的解"],
    ["u_clipped", zeros(N), "1 = 该点被截断显示"],
    ["u_exact", stepInit, "精确平移解"],
    ["u_initial", stepInit, "初值"],
    ["r_grid", rGrid, "斜率比 r"],
    ["phi_curve", phiCurve, "限制器函数 φ(r)"],
    ["sweby_upper", upper, "Sweby 区域上界 min(2r,2)"],
    ["sweby_lower", zeros(R), "Sweby 区域下界 0"],
    ["in_sweby", ones(R), "1 = φ 落在 Sweby 区域内"],
    ["tv_history", tvHist, "总变差 TV(t)"],
    ["forward_error", forward, "正向误差"],
    ["backward_error", backward, "负向误差"],
    ["algebraic_residual", [0], "第一步的代数残差（euler 恒为 0）"],
    ["tv_residual", [1e-17, 1e-17], "TV 的逐步增量 |ΔTV|"],
    ["phi_at_solution", rGrid.slice(0, N).map((r) => Math.min(r, 1)), "当前解上逐点的 φ(r)"],
    ["r_at_solution", rGrid.slice(0, N), "当前解上逐点的 r"],
    ["r_valid", ones(N), "1 = 该点 r 有定义"],
  ],
  meta: {
    limiter_index: 1, nu: 0.4, nx: N, t_requested: 0.3, t_actual: 0.3, n_steps: 2, dx: 1 / N,
    umax: 1, overflow: false, clipped_points: 0, overshoot: 0, undershoot: 0, tv_now: 1.99,
    tv_initial: 2, tv_max_increase_rel: 2.220446e-16, tvd_holds: true, sweby_fraction: 1,
    max_abs_symbol: 1, cfl_max: 1, cfl_exceeded: false, n_plateaus: 1, n_r_undefined: 0,
    r_defined_fraction: 1, second_order_fraction: 0.5,
  },
  strings: {
    limiter: "minmod", floor: "1e-17", time_semantics: "真实时间 t。",
    nonfinite_policy: "r 未定义处填 0 并配掩码。", data_source: "backend",
  },
  critical: { nu: 1.0 },
};

const NP = 32;
const xp = Array.from({ length: NP }, (_, i) => i / NP);
const gauss = (c: number, w: number): number[] =>
  Array.from({ length: NP }, (_, i) => Math.exp(-((((xp[i] as number) - c) / w) ** 2)));
const u0 = gauss(0.5, 0.06);
const envExactP = u0.map((v) => 0.42 * v);
const envNumP = gauss(0.52, 0.07).map((v) => 0.35 * v);
const uP = gauss(0.52, 0.07).map((v) => 0.9 * v);
const errP = uP.map((v, i) => Math.max(1e-17, Math.abs(v - (u0[i] as number))));

const PL_DEF: FrameDef = {
  viz: "packet_long_time",
  arrays: [
    ["x", xp, "格点 x"],
    ["u", uP, "数值波包"],
    ["u_clipped", zeros(NP), "1 = 该点被截断显示"],
    ["u_exact", u0, "精确平移解"],
    ["u_initial", u0, "初值波包"],
    ["envelope_num", envNumP, "数值包络"],
    ["envelope_exact", envExactP, "精确包络"],
    ["error_curve", errP, "|u − 精确解|（对数轴）"],
    ["error_at_floor", zeros(NP), "1 = 该点误差已到地板"],
    ["residual_profile", Array.from({ length: NP }, () => 1e-4), "RK 回代残差"],
    ["integrator_gap", Array.from({ length: NP }, () => 1e-5), "euler 与 rk3 之差"],
    ["tail_region", xp.map((v) => (v > 0.75 ? 1 : 0)), "1 = 拖尾取样区"],
    ["front_region", xp.map((v) => (v < 0.25 ? 1 : 0)), "1 = 前缘取样区"],
  ],
  meta: {
    scheme_index: 2, nu: 0.4, nx: NP, periods_requested: 4, periods_done: 4, t_requested: 4,
    t_actual: 4, n_steps: 1024, dx: 1 / NP, amplitude_ratio: 0.88, front_ratio: 0.01,
    tail_ratio: 0.058, dissipative: true, dispersive: false, umax: 0.9, overflow: false,
    clipped_points: 0, l2_error: 0.1, total_variation: 0.9, max_abs_symbol: 1, decay_order: 4,
  },
  strings: {
    scheme: "lax_wendroff", limiter: "minmod", floor: "1e-17",
    time_semantics: "t 是域长的倍数。", nonfinite_policy: "0 与无穷用 1e-17 地板占位。",
    honesty_note: "periods 很大时主包绕回，tail 不再单调。", data_source: "backend",
  },
  critical: { nu: 1.0 },
};

const DEFS: Record<string, FrameDef> = {
  scheme_zoo: SZ_DEF,
  phase_error: PE_DEF,
  limiter_interaction: LI_DEF,
  packet_long_time: PL_DEF,
};

const sceneOf = async (viz: string, over: Override = {}, opts: Partial<ModelOptions> = {}): Promise<Scene> =>
  models[viz]!(await makeFrame(tweak(DEFS[viz] as FrameDef, over)), { ...OPTS, ...opts });

const axisOf = (scene: Scene, id: string): Axis => {
  const hit = scene.axes.find((a) => a.id === id);
  assert.ok(hit, `场景里没有轴 ${id}`);
  return hit;
};
const layersOn = (scene: Scene, id: string): Layer[] => scene.layers.filter((l) => l.axes === id);
const lineSeriesOn = (scene: Scene, id: string): Array<{ x: ArrayLike<number>; y: ArrayLike<number> }> =>
  layersOn(scene, id).flatMap((l) => (l.kind === "line" ? l.series.map((s) => ({ x: s.x, y: s.y })) : []));
const bandsOn = (scene: Scene, id: string): Array<{ x: ArrayLike<number>; ylo: ArrayLike<number>; yhi: ArrayLike<number> }> =>
  layersOn(scene, id).flatMap((l) => (l.kind === "band" ? [l.band] : []));
const scatterOn = (scene: Scene, id: string): Array<{ x: ArrayLike<number>; y: ArrayLike<number> }> =>
  layersOn(scene, id).flatMap((l) => (l.kind === "scatter" ? [{ x: l.x, y: l.y }] : []));
const textsOf = (scene: Scene): string => scene.texts.map((t) => t.text).join(" | ");
const badgesOf = (scene: Scene): string => scene.badges.map((b) => b.text).join(" | ");
const degradedText = (scene: Scene): string =>
  scene.badges.filter((b) => b.text.includes("降级")).map((b) => b.text).join(" | ");

const near = (a: number, b: number, eps = 1e-12): boolean => Math.abs(a - b) <= eps;

// --------------------------------------------------------------------------- //
// 1. 时间轴端点（与中点）都过校验器
// --------------------------------------------------------------------------- //

for (const [viz, times] of [
  ["scheme_zoo", [0, 0.3, 0.6]],
  ["limiter_interaction", [0, 0.3, 0.6]],
  ["packet_long_time", [0, 2, 4]],
  ["phase_error", [0]],
] as const) {
  for (const t of times) {
    test(`${viz}: t=${t} 的 Scene 过 validateScene（规则 1/2/8 全绿）`, async () => {
      const scene = await sceneOf(viz, {}, { t });
      assert.deepEqual(validateScene(scene), []);
      assert.equal(scene.chapter, "4.2");
      assert.ok(scene.pitfalls.length > 0, "pitfalls 面板必须非空（规则 7）");
      assert.ok(scene.layers.length >= 3, "每个 viz 至少 3 个图层");
      assert.ok(scene.badges.some((b) => b.kind === "source"), "source 角标必须在");
    });
  }
}

test("manifest 与登记表一致：4 个 viz 都有 model，未知 id 返回 null", () => {
  assert.deepEqual(Object.keys(models).sort(), [
    "limiter_interaction",
    "packet_long_time",
    "phase_error",
    "scheme_zoo",
  ]);
  for (const v of manifest.vizzes) assert.ok(modelFor(v.id), `manifest 里的 ${v.id} 没有 model`);
  assert.equal(modelFor("nope"), null);
  assert.equal(manifest.vizzes.length, 4);
});

// --------------------------------------------------------------------------- //
// 2. requiredAux 真的画在对应轴上
// --------------------------------------------------------------------------- //

/** viz → aux 名 → (轴 id, 数组名)。 */
const AUX: Record<string, Record<string, readonly [string, string]>> = {
  scheme_zoo: { error: ["error", "error_curve"], residual: ["residual", "residual_profile"] },
  phase_error: { error: ["phase", "phase_error"] },
  limiter_interaction: { residual: ["residual", "tv_residual"] },
  packet_long_time: { error: ["error", "error_curve"], residual: ["residual", "residual_profile"] },
};

for (const [viz, need] of Object.entries(AUX)) {
  test(`${viz}: requiredAux 的每个名字都有一条真画出来的曲线`, async () => {
    const scene = await sceneOf(viz);
    const frame = await makeFrame(DEFS[viz] as FrameDef);
    for (const [aux, [axisId, arrayName]] of Object.entries(need)) {
      const want = frame.array(arrayName);
      const hit = lineSeriesOn(scene, axisId).some(
        (s) => s.y.length === want.length && Array.from(s.y).every((v, i) => near(v, want[i] as number, 1e-12)),
      );
      assert.ok(hit, `${viz}.${aux}：轴 ${axisId} 上没有画出 ${arrayName} 的数值`);
    }
  });
}

test("limiter_interaction：误差带编码了 forward_error/backward_error（不是只写文字）", async () => {
  const scene = await sceneOf("limiter_interaction");
  const bands = bandsOn(scene, "solution");
  assert.ok(bands.length >= 2, "解面板上缺少正向/负向误差带");
  // 把每条带的每个点映射回 x 下标，检查带的两端确实等于精确解与数值解。
  for (const band of bands) {
    for (let k = 0; k < band.x.length; k += 1) {
      const i = Math.round((band.x[k] as number) * N);
      const lo = band.ylo[k] as number;
      const hi = band.yhi[k] as number;
      const okOver = near(lo, stepInit[i] as number) && near(hi, stepU[i] as number);
      const okUnder = near(lo, stepU[i] as number) && near(hi, stepInit[i] as number);
      assert.ok(okOver || okUnder, `误差带在第 ${i} 点的两端不是 (u, u_exact)`);
      assert.ok((forward[i] as number) > 0 || (backward[i] as number) > 0, "0 误差点不该进带");
    }
  }
});

// --------------------------------------------------------------------------- //
// 3. autoscaled 与 manifest 一致 + 固定窗口确实固定
// --------------------------------------------------------------------------- //

test("autoscaled 与 manifest 一致：所有声明轴都照抄，副轴显式自适应", async () => {
  for (const [viz] of Object.entries(DEFS)) {
    const spec = manifest.vizzes.find((v) => v.id === viz);
    assert.ok(spec);
    const scene = await sceneOf(viz);
    for (const ax of spec.axes) {
      const got = scene.axes.find((a) => a.id === ax.id);
      assert.ok(got, `${viz} 缺 manifest 声明的轴 ${ax.id}`);
      assert.equal(got.autoscaled, ax.autoscaled, `${viz}.${ax.id} 的 autoscaled 与 manifest 不一致`);
    }
    for (const extra of scene.axes.filter((a) => !spec.axes.some((s) => s.id === a.id))) {
      assert.equal(extra.autoscaled, true, `${viz} 的副轴 ${extra.id} 必须显式标注为自适应`);
    }
  }
  assert.deepEqual(axisOf(await sceneOf("scheme_zoo"), "solution").ylim, [-0.6, 1.6]);
  assert.deepEqual(axisOf(await sceneOf("scheme_zoo"), "error").ylim, [1e-17, 3]);
  assert.deepEqual(axisOf(await sceneOf("phase_error"), "speed").ylim, [-0.2, 2.6]);
  assert.deepEqual(axisOf(await sceneOf("phase_error"), "amplitude").ylim, [0.15, 1.15]);
  assert.deepEqual(axisOf(await sceneOf("limiter_interaction"), "sweby").xlim, [-0.5, 4.5]);
  assert.deepEqual(axisOf(await sceneOf("limiter_interaction"), "tv").ylim, [1.8, 2.2]);
  assert.deepEqual(axisOf(await sceneOf("packet_long_time"), "packet").ylim, [-0.3, 1.3]);
  assert.deepEqual(axisOf(await sceneOf("packet_long_time"), "error").ylim, [1e-17, 3]);
});

test("scheme_zoo：换格式不换窗口（防「自适应轴把幅度差拉平」）", async () => {
  const a = await sceneOf("scheme_zoo");
  const b = await sceneOf("scheme_zoo", {
    arrays: {
      u: stepU.map((v) => 3 * v),
      error_curve: stepErr.map((v) => 10 * v),
    },
    strings: { scheme: "lax_wendroff" },
  });
  assert.deepEqual(axisOf(a, "solution").ylim, axisOf(b, "solution").ylim);
  assert.deepEqual(axisOf(a, "error").ylim, axisOf(b, "error").ylim);
  const plotted = lineSeriesOn(b, "solution").find((s) => s.y.length === N && near(s.y[0] as number, 3));
  assert.ok(plotted, "数值解曲线不见了");
  assert.ok(near(plotted.y[3] as number, 2.4), "模型把 u 重新缩放过（必须原样画帧里的值）");
});

// --------------------------------------------------------------------------- //
// 4. 本章特有的画法约束
// --------------------------------------------------------------------------- //

test("scheme_zoo：误差轴固定 1e-17–3；过冲/欠冲只在 >0 处画，0 点不落在地板上", async () => {
  const scene = await sceneOf("scheme_zoo");
  for (const s of lineSeriesOn(scene, "error")) {
    for (const v of Array.from(s.y)) assert.ok((v as number) >= 1e-17, "误差轴上出现了 0（会被读成精确）");
  }
  // 把 forward_error 全置 0：所有正向误差曲线都必须消失（不能画一条地板线）。
  const baseLines = layersOn(scene, "error").filter((l) => l.kind === "line").length;
  const none = await sceneOf("scheme_zoo", { arrays: { forward_error: zeros(N) } });
  assert.deepEqual(validateScene(none), []);
  const noneLines = layersOn(none, "error").filter((l) => l.kind === "line").length;
  assert.equal(noneLines, baseLines - 1, "forward_error 全 0 时仍画了正向误差曲线（0 不是数据）");
  // 全部贴地板时必须有散点标记，而不是静默。
  const floored = await sceneOf("scheme_zoo", { arrays: { error_at_floor: ones(N), error_curve: new Float64Array(N).fill(1e-17) } });
  assert.equal(scatterOn(floored, "error").length, 1, "贴地板的点必须用散点标出");
});

test("phase_error 没有时间轴：请求 t 不进 scene.t", async () => {
  for (const t of [0, 3, 99]) {
    const scene = await sceneOf("phase_error", {}, { t });
    assert.equal(scene.t, 0, "mode=none 的 viz 不许把 opts.t 写进 scene.t");
  }
});

test("phase_error：预测失效区是一条从 breakdown 到 π 的带 + 竖线", async () => {
  const scene = await sceneOf("phase_error");
  const bands = bandsOn(scene, "speed");
  assert.equal(bands.length, 1, "缺少预测失效区带");
  const band = bands[0]!;
  assert.ok(near(band.x[0] as number, 1.2) && near(band.x[1] as number, Math.PI), `带的右边界错了：${Array.from(band.x)}`);
  assert.ok(
    lineSeriesOn(scene, "speed").some(
      (s) => s.x.length === 2 && near(s.x[0] as number, 1.2) && near(s.x[1] as number, 1.2),
    ),
    "缺少 breakdown 竖线",
  );
  // mark_breakdown = -1：不画带，正文点名「全区间 ≤1%」。
  const none = await sceneOf("phase_error", { arrays: { mark_breakdown: [-1] } });
  assert.equal(bandsOn(none, "speed").length, 0);
  assert.match(textsOf(none), /全区间预测相对差 ≤1%/);
});

test("phase_error：ν→0 的帧（c_num≡1）正好落在 c=1 上，相位误差轴仍有零线", async () => {
  const scene = await sceneOf("phase_error", {
    arrays: {
      c_num: ones(P),
      c_pred: ones(P),
      c_one: ones(P),
      phase_error: zeros(P),
      g_abs: ones(P),
      prediction_ok: ones(P),
      mark_breakdown: [-1],
    },
  });
  assert.deepEqual(axisOf(scene, "speed").ylim, [-0.2, 2.6], "速度窗被自适应改掉了");
  const curves = lineSeriesOn(scene, "speed").filter((s) => s.y.length === P);
  const exactLine = curves.find((s) => Array.from(s.y).every((v) => near(v as number, 1)));
  assert.ok(exactLine, "c_num≡1 时找不到落在 1 上的曲线（被重新缩放或丢掉了）");
  const phaseLines = lineSeriesOn(scene, "phase");
  assert.ok(phaseLines.some((s) => s.y.length === 2 && near(s.y[0] as number, 0) && near(s.y[1] as number, 0)), "缺少 ΔΦ=0 零线");
});

test("limiter_interaction：Sweby 绿带逐点等于帧的上下界，r<0 单独一条带", async () => {
  const scene = await sceneOf("limiter_interaction");
  const bands = bandsOn(scene, "sweby");
  assert.ok(bands.length >= 2, "缺少 Sweby 区域带或 r<0 带");
  const region = bands.find((b) => b.x.length === R);
  assert.ok(region, "Sweby 区域带必须覆盖整条 r 网格");
  for (let i = 0; i < R; i += 1) {
    assert.ok(near(region.yhi[i] as number, upper[i] as number), `Sweby 上界在第 ${i} 点不等于 min(2r,2)`);
    assert.ok(near(region.ylo[i] as number, 0), "Sweby 下界不是 0");
  }
  // TVD 区间上界单调不减，且不超过 2。
  for (let i = 1; i < R; i += 1) {
    assert.ok((region.yhi[i] as number) >= (region.yhi[i - 1] as number) - 1e-12, "Sweby 上界出现回落");
    assert.ok((region.yhi[i] as number) <= 2 + 1e-12, "Sweby 上界越过 2");
  }
  const negative = bands.find((b) => b.x.length === 2 && near(b.x[0] as number, -0.5) && near(b.x[1] as number, 0));
  assert.ok(negative, "缺少 r<0（φ 必须为 0）的竖带");
});

test("limiter_interaction：TV 曲线照画帧值，tvd_holds 翻转时文字必须跟着翻", async () => {
  const scene = await sceneOf("limiter_interaction");
  const tvLine = lineSeriesOn(scene, "tv").find((s) => s.y.length === tvHist.length);
  assert.ok(tvLine, "TV(t) 曲线不见了");
  assert.deepEqual(Array.from(tvLine.y), tvHist);
  assert.match(textsOf(scene), /TVD 成立/);
  const broken = await sceneOf("limiter_interaction", {
    meta: { tvd_holds: false, tv_now: 3.2, tv_max_increase_rel: 0.6 },
    arrays: { tv_history: [2, 2.4, 3.2] },
  });
  assert.deepEqual(validateScene(broken), []);
  assert.match(textsOf(broken), /TVD 破坏/);
  assert.match(badgesOf(broken), /TVD 破坏/);
});

test("packet_long_time：同一 x 上有幅值标尺（耗散），前缘/拖尾取样带都在", async () => {
  const scene = await sceneOf("packet_long_time");
  const ruler = lineSeriesOn(scene, "packet").find(
    (s) => s.x.length === 2 && near(s.x[0] as number, s.x[1] as number) && Math.abs((s.y[0] as number) - (s.y[1] as number)) > 1e-9,
  );
  assert.ok(ruler, "缺少包络峰值的竖直标尺（耗散读数）");
  const bands = bandsOn(scene, "packet");
  assert.equal(bands.filter((b) => b.x.length > 1).length, 2, "前缘/拖尾取样带缺一");
  assert.match(textsOf(scene), /耗散型/);
  const dispersive = await sceneOf("packet_long_time", {
    meta: { dissipative: false, dispersive: true, amplitude_ratio: 0.95, tail_ratio: 0.3 },
  });
  assert.match(textsOf(dispersive), /色散型/);
  assert.deepEqual(axisOf(dispersive, "packet").ylim, [-0.3, 1.3], "波包窗被自适应改掉了");
});

// --------------------------------------------------------------------------- //
// 5. 降级不静默 / 缺必需字段抛错
// --------------------------------------------------------------------------- //

test("缺可选数组/可选 meta → 不抛错，但 degraded 角标点名", async () => {
  const cases: Array<[string, Override, RegExp]> = [
    ["scheme_zoo", { drop: ["u_clipped"] }, /u_clipped/],
    ["scheme_zoo", { dropMeta: ["max_abs_symbol"] }, /max_abs_symbol/],
    ["phase_error", { drop: ["prediction_ok"] }, /prediction_ok/],
    ["phase_error", { dropMeta: ["series_radius"] }, /series_radius/],
    ["limiter_interaction", { drop: ["in_sweby"] }, /in_sweby/],
    ["limiter_interaction", { dropMeta: ["tv_max_increase_rel"] }, /tv_max_increase_rel/],
    ["packet_long_time", { drop: ["tail_region"] }, /tail_region/],
    ["packet_long_time", { dropMeta: ["amplitude_ratio"] }, /amplitude_ratio/],
  ];
  for (const [viz, over, want] of cases) {
    const scene = await sceneOf(viz, over);
    assert.deepEqual(validateScene(scene), [], `${viz} 降级后仍要过校验器`);
    const degraded = degradedText(scene);
    assert.ok(degraded.length > 0, `${viz} 降级后必须有 degraded 角标（不许静默）`);
    assert.ok(
      want.test(degraded) || /等 \d+ 项/.test(degraded),
      `${viz} 降级说明里既没点名 ${want}，也没有「等 N 项」：${degraded}`,
    );
  }
});

test("必填数组缺失 → 抛 ModelInputError（不静默画空图）", async () => {
  for (const [viz, name] of [
    ["scheme_zoo", "u_exact"],
    ["phase_error", "c_num"],
    ["limiter_interaction", "sweby_upper"],
    ["packet_long_time", "envelope_num"],
  ] as const) {
    await assert.rejects(async () => sceneOf(viz, { drop: [name] }), ModelInputError);
  }
});

test("pitfalls 为空 → 抛错（规则 7：防误导面板必须有内容）", async () => {
  await assert.rejects(async () => {
    const frame = await makeFrame({ ...SZ_DEF, pitfalls: "" });
    models["scheme_zoo"]!(frame, OPTS);
  }, ModelInputError);
});

test("窄画布 480×520：四个 viz 仍合法，并明确说明长说明被放弃", async () => {
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz, {}, { width: 480, height: 520 });
    assert.deepEqual(validateScene(scene), [], `${viz} 在 480×520 下仍要合法`);
    for (const ax of scene.axes) assert.ok(ax.rect[2] > 0 && ax.rect[3] > 0, `${viz}.${ax.id} 宽高必须为正`);
    assert.ok(degradedText(scene).length > 0, `${viz} 放弃说明文字时必须点名降级`);
  }
});

test("null meta 值按缺失处理（不把 null 当 0 画进图里）", async () => {
  const scene = await sceneOf("scheme_zoo", { meta: { l2_error: null, max_abs_symbol: null } });
  assert.deepEqual(validateScene(scene), []);
  assert.ok(!/L2 误差 null/.test(textsOf(scene)), "null 不许被格式化成 null 写进图里");
  assert.match(degradedText(scene), /max_abs_symbol|l2_error/);
});
