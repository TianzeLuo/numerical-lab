/**
 * 4.6 章（谱方法与伪谱）model 层测试 —— **自足**：不依赖 Python，也不依赖
 * `data/frames` 是否存在。夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，
 * 字段集合（数组名 / dtype / shape / unit / label、meta / strings / critical 键）逐字照抄
 * `server/chapters/4.6.py` 的真实帧（数值是测试用的合成值，量级与真实帧一致）。
 *
 * 本文件守的是这一章最容易骗人的几张图：
 *
 * - **模态衰减**：主图双对数、理论包络同屏；系数必须单调衰减并压到地板；
 *   `local_slope` 是「斜率在变」的证据，窗口固定 -24–0；
 * - **去混叠**：本尊 2N/3 与别名 N/3 必须画在**波数轴**上（不是比值轴）；
 *   2/3+3/2 的扫描线必须压在所有账的最下面（贴舍入）；
 * - **收敛竞赛**：横轴同口径（自由度），差分理论阶 N^-2 / N^-4 必须有参考线；
 *   谱误差撞地板（n_at_floor）之后的每加倍比值不进图层；t=0 退化帧不画假比值；
 * - **Burgers**：能谱 k^-2、能流窗口、激波位置、能量漂移/收支残差是四件不同的事，
 *   分别落在四块面板/副轴上；欠分辨时能流越窗必须由 overflow 点名。
 *
 * 缺字段纪律：必需数组缺失 → `ModelInputError`；可选掩码/meta 缺失 → 降级角标点名。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, toNumeric, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import {
  burgersCascadeModel,
  convergenceRaceModel,
  dealiasRuleModel,
  manifest,
  modalDecayModel,
  modelFor,
  models,
} from "../chapters/4.6/index.js";

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
  "它会怎样骗你：第一，「指数收敛」不能只看拟合 p，要看局部斜率是否在变、每加倍比值是否在变小；" +
  "第二，跳变剖面的误差不随 N 下降，Gibbs 只被压窄、不被压平，加密网格没有指数红利；" +
  "第三，混叠与截断是两笔账，2/3 截断自己丢掉的比它顺手消掉的混叠还大，混在一起读会得出相反结论；" +
  "第四，只补零不截断时残差集中在 Nyquist 边界模态，不是「白补」；" +
  "第五，1D Burgers 的 k^-2 段是激波的谱，不是 3D 湍流的惯性区，Pi≈eps 在这里通常不成立；" +
  "第六，nu 太小时能量会涨，收支残差与高波数占比会告诉你高波数已经由网格决定；" +
  "第七，选错平流时刻（半周期/整周期）会让所有格式的误差退化到舍入级；" +
  "第八，固定窗口之外的点必须由溢出角标点名，不能靠裁剪把越界藏起来。";

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
        chapter: "4.6",
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
  const meta: Record<string, number | boolean | null> = { ...def.meta };
  for (const key of ov.dropMeta ?? []) delete meta[key];
  return { ...def, arrays, meta: { ...meta, ...ov.meta }, strings: { ...def.strings, ...ov.strings } };
}

const arrOf = (frame: Frame, name: string): Float64Array => Float64Array.from(toNumeric(frame.array(name)));

const sceneOf = async (viz: string, over: Override = {}, opts: Partial<ModelOptions> = {}): Promise<Scene> =>
  models[viz as keyof typeof models](await makeFrame(tweak(DEFS[viz] as FrameDef, over)), { ...OPTS, ...opts });

const axisOf = (scene: Scene, id: string): Axis => {
  const hit = scene.axes.find((a) => a.id === id);
  assert.ok(hit, `场景里没有轴 ${id}`);
  return hit;
};
const layersOn = (scene: Scene, id: string): Layer[] => scene.layers.filter((l) => l.axes === id);
const lineSeriesOn = (scene: Scene, id: string): Array<{ x: ArrayLike<number>; y: ArrayLike<number>; color?: string }> =>
  layersOn(scene, id).flatMap((l) =>
    l.kind === "line" ? l.series.map((s) => ({ x: s.x, y: s.y, color: s.color })) : [],
  );
const textsOf = (scene: Scene): string => scene.texts.map((t) => t.text).join(" | ");
const badgesOf = (scene: Scene): string => scene.badges.map((b) => b.text).join(" | ");
const degradedText = (scene: Scene): string =>
  scene.badges.filter((b) => b.text.includes("降级")).map((b) => b.text).join(" | ");
const hasSeries = (seriesList: Array<{ x: ArrayLike<number>; y: ArrayLike<number> }>, x: ArrayLike<number>, y: ArrayLike<number>, tol = 1e-12): boolean =>
  seriesList.some(
    (s) =>
      s.y.length === y.length &&
      s.x.length === x.length &&
      Array.from(s.y).every((v, i) => Math.abs(v - (y[i] as number)) <= tol) &&
      Array.from(s.x).every((v, i) => Math.abs(v - (x[i] as number)) <= tol),
  );

// --------------------------------------------------------------------------- //
// 夹具
// --------------------------------------------------------------------------- //

const K32 = Array.from({ length: 32 }, (_, i) => i + 1);
const AMP = K32.map((k) => (k > 27 ? 1e-18 : 0.36 * Math.pow(0.6, k - 1)));
const THEORY = K32.map((k) => 0.36 * Math.pow(0.6, k - 1));
const ERR_N = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256];

const MD_DEF: FrameDef = {
  viz: "modal_decay",
  arrays: [
    ["mode_k", K32, "模态序号 k"],
    ["coeff_abs", AMP, "|a_k|（地板占位）"],
    ["coeff_theory", THEORY, "理论包络 C·rho^k 或 C·k^-p"],
    ["coeff_floor_mask", K32.map((k) => (k > 27 ? 1 : 0)), "1=该模态低于地板"],
    ["slope_k", K32.slice(0, 31), "局部斜率的横坐标"],
    ["local_slope", K32.slice(0, 31).map((_, i) => -0.7 - (3.2 * i) / 30), "dlog|a_k|/dlog k"],
    ["slope_floor_mask", new Float64Array(31), "1=斜率不可测"],
    ["error_n", ERR_N, "自由度 N"],
    ["error_smooth", ERR_N.map((n) => 0.48 * Math.pow(n / 8, -3)), "光滑剖面逼近误差"],
    ["error_corner", ERR_N.map((n) => 0.05 * Math.pow(n / 8, -1)), "尖角剖面逼近误差"],
    ["error_jump", ERR_N.map((_, i) => 1.956 - 0.0006 * i), "跳变剖面逼近误差"],
    ["error_floor_mask", ERR_N.map((_, i) => (i >= 9 ? 1 : 0)), "1=该 N 有误差在地板"],
  ],
  meta: {
    n: 64,
    rho: 0.6,
    n_modes: 32,
    fit_lo: 4,
    fit_hi: 32,
    fit_n_used: 29,
    fit_p: -7.129,
    fit_r2: 0.9377,
    fit_dropped: 0,
    slope_first: -0.737,
    slope_last: -3.942,
    slope_steepens: true,
    coeff_floor: 1e-18,
    error_floor: 1e-17,
    err_smooth_min: 7.1e-15,
    err_smooth_last: 7.1e-15,
    err_corner_slope: -1.008,
    err_jump_first: 1.956,
    err_jump_last: 1.802,
    err_jump_drop: 1.085,
    n_at_floor: 256,
    ratio_smooth_first: 0.1434,
    ratio_smooth_last: 8.6e-7,
    ratio_smooth_n: 7,
    gibbs_overshoot: null,
    theory_p: null,
    roundoff: 1.1e-16,
  },
  strings: {
    status: "ok",
    data_source: "backend",
    basis: "fourier",
    profile: "解析（Poisson 核 / Runge 函数）",
    floor: "1e-18",
    time_semantics: "无时间轴（一次性扫整条衰减曲线与误差阶梯）",
    reading_note: "局部斜率是否在变才是指数收敛的判据。",
    theory_note: "C^∞ 解析 a_k ~ rho^k；尖角 k^-2；跳变 k^-1。",
  },
  critical: {},
};

const KD = Array.from({ length: 65 }, (_, i) => i);
const PROD_REF = KD.map((k) => 0.274 / Math.pow(1 + k, 1.2));
const SCAN = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5];
const DR_DEF: FrameDef = {
  viz: "dealias_rule",
  arrays: [
    ["k_abs", KD, "|k|（单边）"],
    ["prod_ref", PROD_REF, "全带参照乘积谱"],
    ["prod_cur", PROD_REF.map((v) => 0.99 * v), "当前模式乘积谱"],
    ["prod_floor_mask", new Float64Array(KD.length), "1=两条谱都低于地板"],
    ["alias_error_spectrum", KD.map((k) => (k === 0 ? 1e-18 : 0.0033 / Math.pow(k, 0.8))), "混叠偏差折叠谱"],
    ["alias_error_mask", KD.map((k) => (k === 0 ? 1 : 0)), "1=该模偏差低于地板"],
    ["scan_decay", SCAN, "输入谱指数"],
    ["scan_alias", SCAN.map((_, i) => 0.066 * Math.pow(0.6, i)), "混叠误差"],
    ["scan_truncation", SCAN.map((_, i) => 0.14 * Math.pow(0.6, i)), "2/3 截断代价"],
    ["scan_pad_only", SCAN.map((_, i) => 0.0016 * Math.pow(0.35, i)), "只补零偏差"],
    ["scan_dealias", SCAN.map(() => 1e-16), "2/3+3/2 对截断参照"],
    ["scan_floor_mask", new Float64Array(SCAN.length), "1=有曲线在地板"],
    ["k_keep_line", [128 / 3, 128 / 3], "2/3 带边缘 k=N/3"],
  ],
  meta: {
    n: 128,
    decay: 1.0,
    exact_pad: false,
    m_used: 256,
    k_keep_max: 128 / 3,
    alias_rel: 0.01196,
    alias_low_rel: 0.002666,
    truncation_rel: 0.0374,
    dealias_rel: 1.137e-16,
    pad_only_rel: 7.76e-5,
    naive_vs_dealias_rel: 0.04034,
    cur_rel: 1.137e-16,
    cur_low_rel: 1.137e-16,
    cur_nyquist_rel: 9.73e-18,
    folded_energy_fraction: 4.25e-4,
    truncated_input_energy_fraction: 0.00537,
    energy_resid_naive: 3.42e-18,
    energy_resid_dealias: 8.45e-17,
    energy_resid_scale: 0.2744,
    energy_floor: 1e-18,
    error_floor: 1e-18,
    roundoff: 1.1e-16,
  },
  strings: {
    status: "ok",
    data_source: "backend",
    mode: "dealias",
    floor: "1e-18",
    time_semantics: "无时间轴（对静态测试场做一次乘积）",
    reading_note: "副图是混叠偏差折叠谱，铺满保留带。",
    truncation_note: "截断代价是另一笔账，不能和混叠混读。",
    energy_note: "能量恒等式残差不是去混叠的证据。",
  },
  critical: {},
};

const CR_N = [16, 24, 32, 48, 64, 96, 128, 192, 256];
const RN = [24, 32, 48, 64, 96, 128, 192, 256];
const PX = Array.from({ length: 64 }, (_, i) => i / 64);
const CR_DEF: FrameDef = {
  viz: "convergence_race",
  arrays: [
    ["error_n", CR_N, "自由度 N"],
    ["error_spectral", [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-11, 1e-14, 1e-14, 1e-14], "谱相对误差"],
    ["error_fd2", CR_N.map((n) => 0.5 * Math.pow(n / 16, -2)), "二阶差分相对误差"],
    ["error_fd4", CR_N.map((n) => 0.5 * Math.pow(n / 16, -4)), "四阶差分相对误差"],
    ["error_floor_mask", CR_N.map((_, i) => (i >= 6 ? 1 : 0)), "1=有误差在地板"],
    ["ratio_n", RN, "每加倍比值横坐标"],
    ["ratio_spectral", [0.1, 0.05, 0.01, 0.001, 1e-4, 3.0, 0.0, 1e-9], "谱每加倍比值"],
    ["ratio_fd2", RN.map(() => 0.25), "二阶差分每加倍比值"],
    ["ratio_fd4", RN.map(() => 0.0625), "四阶差分每加倍比值"],
    ["point_x", PX, "网格点 x"],
    ["point_error_spectral", PX.map(() => 3e-15), "逐点误差 谱"],
    ["point_error_fd2", PX.map((x) => 1e-3 * (1 + Math.sin(2 * Math.PI * x))), "逐点误差 FD2"],
    ["point_error_fd4", PX.map((x) => 1e-6 * (1 + Math.cos(2 * Math.PI * x))), "逐点误差 FD4"],
    ["point_floor_mask", PX.map(() => 0), "1=逐点误差在地板"],
  ],
  meta: {
    rho: 0.6,
    n_max: 256,
    t_actual: 0.5,
    c: 0.85,
    shift: 0.425,
    n_points: 9,
    fit_lo: 128,
    fit_hi: 256,
    slope_spectral: null,
    slope_spectral_r2: null,
    slope_spectral_trap: -1.777,
    slope_spectral_trap_r2: 0.1112,
    fit_spectral_lo: null,
    fit_spectral_hi: null,
    fit_spectral_points: 0,
    slope_fd2: -1.9859,
    slope_fd2_r2: 0.99989,
    slope_fd4: -3.9727,
    slope_fd4_r2: 0.99996,
    degenerate: false,
    ratio_spectral_last: 1.35e-4,
    ratio_spectral_first: 0.02974,
    ratio_fd2_last: 0.24449,
    ratio_fd4_last: 0.06145,
    err_spectral_min: 1.2e-15,
    err_spectral_at_max_n: 1.2e-15,
    err_fd2_at_max_n: 0.00966,
    err_fd4_at_max_n: 9.1e-5,
    gain_over_fd2: 7.9e12,
    n_at_floor: 128,
    error_floor: 1e-17,
    roundoff: 1.1e-16,
  },
  strings: {
    status: "ok",
    data_source: "backend",
    floor: "1e-17",
    time_semantics: "真实时间 t：同一条平流解推进到 t",
    reading_note: "比值是不是常数才是阶的判据。",
    shift_note: "c=0.85 避开半周期/整周期。",
    floor_note: "谱曲线撞 1e-13 以下地板后不再下降。",
  },
  critical: {},
};

const KB = Array.from({ length: 65 }, (_, i) => i);
const XB = Array.from({ length: 128 }, (_, i) => (2 * Math.PI * i) / 128);
const U_SHOCK = XB.map((x) => Math.tanh((Math.PI - x) * 8));
const E_T = Array.from({ length: 9 }, (_, i) => (1.5 * i) / 8);
const B_T = Array.from({ length: 30 }, (_, i) => (1.5 * i) / 29);
const BC_DEF: FrameDef = {
  viz: "burgers_cascade",
  arrays: [
    ["k", KB, "波数 k"],
    ["energy_now", KB.map((k) => (k === 0 ? 1e-18 : 0.25 * Math.pow(k, -2))), "当前能谱"],
    ["energy_initial", KB.map((k) => (k === 1 ? 0.5 : 1e-18)), "初值能谱"],
    ["energy_ref_slope", KB.map((k) => (k === 0 ? 1e-18 : 0.5 * Math.pow(k, -2))), "k^-2 参考线"],
    ["energy_floor_mask", KB.map((k) => (k === 0 || k > 40 ? 1 : 0)), "1=能谱低于地板"],
    ["flux", KB.map((k) => 0.18 * Math.exp(-((k - 6) ** 2) / 30) - 0.02 * (k / 64) ** 2), "能流"],
    ["x", XB, "网格点 x"],
    ["u_now", U_SHOCK, "当前解"],
    ["u_initial", XB.map((x) => Math.sin(x)), "初值"],
    ["u_clipped", new Float64Array(XB.length), "1=该点被截断"],
    ["energy_t", E_T, "能谱快照时刻"],
    ["energy_error", E_T.map((t) => Math.max(1e-18, 0.1853 * (t / 1.5) ** 2)), "相对能量漂移"],
    ["energy_error_mask", E_T.map((t) => (t === 0 ? 1 : 0)), "1=漂移低于地板"],
    ["budget_t", B_T, "收支记账时刻"],
    ["budget_residual", B_T.map((t) => Math.max(1e-18, 0.00586 * (t / 1.5) ** 2)), "收支残差"],
    ["budget_residual_mask", B_T.map((t) => (t === 0 ? 1 : 0)), "1=残差低于地板"],
  ],
  meta: {
    n: 128,
    nu: 0.02,
    dealias: true,
    t_actual: 1.5,
    dt: 0.017,
    dt_cfl: 0.01718,
    n_steps: 88,
    k_diss: 5.7735,
    fit_lo: 3,
    fit_hi: 8,
    fit_window_points: 6,
    slope_fit: -2.1994,
    slope_r2: 0.99998,
    slope_theory: -2.0,
    energy_0: 0.5,
    energy_now: 0.4073,
    energy_drift: -0.0927,
    energy_error_max: 0.1853,
    enstrophy: 5.1535,
    dissipation: 0.2061,
    flux_max: 0.17475,
    flux_at_end: -0.01688,
    flux_over_eps: 0.8479,
    budget_error_max: 0.005862,
    frac_energy_above_2_3: 4.93e-4,
    shock_width_ratio: 0.2037,
    max_u: 0.993,
    u_clip: 1.6,
    n_clipped: 0,
    energy_floor: 1e-18,
    error_floor: 1e-17,
    roundoff: 1.1e-16,
  },
  strings: {
    status: "ok",
    data_source: "backend",
    initial: "sine",
    floor: "1e-18",
    time_semantics: "真实时间 t：从初值推进到 t",
    reading_note: "能谱、能流、解曲线取当前时刻。",
    cascade_note: "1D Burgers 的 k^-2 段是激波的谱。",
    energy_note: "收支残差同时受时间离散、截断与欠分辨影响。",
  },
  critical: { nu: 0.015 },
};

const DEFS: Record<string, FrameDef> = {
  modal_decay: MD_DEF,
  dealias_rule: DR_DEF,
  convergence_race: CR_DEF,
  burgers_cascade: BC_DEF,
};

// --------------------------------------------------------------------------- //
// 1. 时间轴端点 + 校验器
// --------------------------------------------------------------------------- //

const MODEL_BY_VIZ = {
  modal_decay: modalDecayModel,
  dealias_rule: dealiasRuleModel,
  convergence_race: convergenceRaceModel,
  burgers_cascade: burgersCascadeModel,
};

test("manifest 与登记表一致：4 个 viz 都有 model，未知 id 返回 null", () => {
  assert.deepEqual(Object.keys(models).sort(), [
    "burgers_cascade",
    "convergence_race",
    "dealias_rule",
    "modal_decay",
  ]);
  assert.equal(manifest.vizzes.length, 4);
  for (const v of manifest.vizzes) assert.ok(modelFor(v.id), `manifest 里的 ${v.id} 没有 model`);
  assert.equal(modelFor("nope"), null);
  for (const [viz, fn] of Object.entries(MODEL_BY_VIZ)) assert.equal(models[viz], fn);
});

for (const viz of ["modal_decay", "dealias_rule"] as const) {
  test(`${viz}: mode=none，t 轴两端都合法且 scene.t 恒为 0`, async () => {
    for (const t of [0, 0.5, 99]) {
      const scene = await sceneOf(viz, {}, { t });
      assert.deepEqual(validateScene(scene), []);
      assert.equal(scene.chapter, "4.6");
      assert.equal(scene.t, 0, "无时间轴的 viz 不许把 opts.t 写进 scene.t");
      assert.ok(scene.pitfalls.length > 0);
      assert.ok(scene.layers.length >= 3);
    }
  });
}

for (const viz of ["convergence_race", "burgers_cascade"] as const) {
  test(`${viz}: 时间轴端点 t=0 / tMax 都过 validateScene`, async () => {
    const tMax = viz === "convergence_race" ? 1.0 : 1.5;
    for (const t of [0, tMax]) {
      const scene = await sceneOf(viz, { meta: { t_actual: t } }, { t });
      assert.deepEqual(validateScene(scene), []);
      assert.ok(Math.abs(scene.t - t) < 1e-12, `${viz} 的 scene.t 必须取 meta.t_actual`);
    }
  });
}

// --------------------------------------------------------------------------- //
// 2. requiredAux 真的画在对应轴上（按数值认，不按 label）
// --------------------------------------------------------------------------- //

test("modal_decay: requiredAux=error 的三条误差曲线都在 error 轴上", async () => {
  const frame = await makeFrame(MD_DEF);
  const scene = await sceneOf("modal_decay");
  const series = lineSeriesOn(scene, "error");
  for (const name of ["error_smooth", "error_corner", "error_jump"]) {
    assert.ok(hasSeries(series, arrOf(frame, "error_n"), arrOf(frame, name)), `error 轴上没有 ${name}`);
  }
});

test("dealias_rule: requiredAux=error 的混叠偏差折叠谱在 alias 轴上", async () => {
  const frame = await makeFrame(DR_DEF);
  const scene = await sceneOf("dealias_rule");
  const want = arrOf(frame, "alias_error_spectrum").slice(1); // k=0 不进对数横轴
  const got = lineSeriesOn(scene, "alias").some(
    (s) => s.y.length === want.length && Array.from(s.y).every((v, i) => Math.abs(v - (want[i] as number)) < 1e-15),
  );
  assert.ok(got, "alias 轴上没有画出 |W_naive-W_ref| 的折叠谱");
});

test("convergence_race: requiredAux=error 的三条误差阶梯都在 error 轴上", async () => {
  const frame = await makeFrame(CR_DEF);
  const scene = await sceneOf("convergence_race");
  const series = lineSeriesOn(scene, "error");
  for (const name of ["error_spectral", "error_fd2", "error_fd4"]) {
    assert.ok(hasSeries(series, arrOf(frame, "error_n"), arrOf(frame, name)), `error 轴上没有 ${name}`);
  }
});

test("burgers_cascade: requiredAux=error/residual 的漂移与收支残差都在 residual 轴上", async () => {
  const frame = await makeFrame(BC_DEF);
  const scene = await sceneOf("burgers_cascade");
  const series = lineSeriesOn(scene, "residual");
  assert.ok(hasSeries(series, arrOf(frame, "energy_t"), arrOf(frame, "energy_error")), "residual 轴上没有 energy_error");
  assert.ok(hasSeries(series, arrOf(frame, "budget_t"), arrOf(frame, "budget_residual")), "residual 轴上没有 budget_residual");
});

// --------------------------------------------------------------------------- //
// 3. 轴与窗口的硬约束
// --------------------------------------------------------------------------- //

test("modal_decay: 主图双对数、三个轴都是固定窗口且 autoscaled=false", async () => {
  for (const t of [0, 0.5]) {
    const scene = await sceneOf("modal_decay", {}, { t });
    const decay = axisOf(scene, "decay");
    assert.equal(decay.xscale, "log");
    assert.equal(decay.yscale, "log");
    assert.deepEqual(decay.ylim, [1e-18, 1]);
    assert.deepEqual(axisOf(scene, "slope").ylim, [-24, 0]);
    assert.deepEqual(axisOf(scene, "error").ylim, [1e-17, 10]);
    for (const ax of scene.axes) assert.equal(ax.autoscaled, false, `${ax.id} 必须固定窗口`);
    assert.ok(lineSeriesOn(scene, "decay").some((s) => s.y.length === 32), "decay 轴上缺少实测系数曲线");
  }
});

test("dealias_rule: 三块面板都是固定窗口的对数轴", async () => {
  const scene = await sceneOf("dealias_rule");
  for (const id of ["spectrum", "alias", "scan"]) {
    const ax = axisOf(scene, id);
    assert.equal(ax.yscale, "log");
    assert.equal(ax.autoscaled, false);
  }
  assert.deepEqual(axisOf(scene, "spectrum").ylim, [1e-18, 10]);
  assert.deepEqual(axisOf(scene, "alias").ylim, [1e-18, 10]);
  assert.deepEqual(axisOf(scene, "scan").ylim, [1e-17, 1]);
});

test("convergence_race: ratio 固定 0–1、error/pointwise 双对数固定 1e-17–1", async () => {
  const scene = await sceneOf("convergence_race");
  const ratio = axisOf(scene, "ratio");
  assert.deepEqual(ratio.ylim, [0, 1]);
  assert.equal(ratio.yscale, "linear");
  assert.equal(axisOf(scene, "error").yscale, "log");
  assert.deepEqual(axisOf(scene, "error").ylim, [1e-17, 1]);
  assert.equal(axisOf(scene, "pointwise").yscale, "log");
  assert.deepEqual(axisOf(scene, "pointwise").ylim, [1e-17, 1]);
  for (const ax of scene.axes) assert.equal(ax.autoscaled, false);
});

test("burgers_cascade: 能谱双对数、能流固定 -0.3–0.3、解固定 ±1.6、残差固定 1e-18–10", async () => {
  const scene = await sceneOf("burgers_cascade");
  assert.equal(axisOf(scene, "spectrum").xscale, "log");
  assert.equal(axisOf(scene, "spectrum").yscale, "log");
  assert.deepEqual(axisOf(scene, "spectrum").ylim, [1e-18, 1]);
  assert.deepEqual(axisOf(scene, "flux").ylim, [-0.3, 0.3]);
  assert.deepEqual(axisOf(scene, "solution").ylim, [-1.6, 1.6]);
  assert.equal(axisOf(scene, "residual").yscale, "log");
  assert.deepEqual(axisOf(scene, "residual").ylim, [1e-18, 10]);
  for (const ax of scene.axes) assert.equal(ax.autoscaled, false);
});

// --------------------------------------------------------------------------- //
// 4. 本章特有的不变式
// --------------------------------------------------------------------------- //

test("modal_decay: 系数单调衰减到地板，理论包络与地板线同屏（规则 2/7）", async () => {
  const scene = await sceneOf("modal_decay");
  const lines = lineSeriesOn(scene, "decay");
  const meas = lines.find(
    (s) =>
      s.y.length === 32 &&
      Math.abs((s.y[0] as number) - 0.36) < 1e-12 &&
      Math.abs((s.y[s.y.length - 1] as number) - 1e-18) < 1e-24,
  );
  assert.ok(meas, "找不到实测系数曲线");
  for (let i = 1; i < meas.y.length; i += 1) {
    assert.ok((meas.y[i] as number) <= (meas.y[i - 1] as number) + 1e-15, `系数在 k=${i + 1} 处反弹`);
  }
  assert.ok(Math.abs((meas.y[meas.y.length - 1] as number) - 1e-18) < 1e-24, "最后一个模态没有压到地板");
  assert.ok(hasSeries(lines, K32, THEORY), "decay 轴上缺少理论包络（参考线）");
  const floorLine = lines.some((s) => s.y.length === 2 && s.y[0] === 1e-18 && s.y[1] === 1e-18);
  assert.ok(floorLine, "decay 轴上缺少 1e-18 地板线");
  assert.match(textsOf(scene), /ρ\^k|rho\^k/);
  assert.match(textsOf(scene), /跳变误差/);
});

test("dealias_rule: 本尊 2N/3 与别名 N/3 都画在波数轴上，且 2N/3 在 Nyquist 之外", async () => {
  const scene = await sceneOf("dealias_rule");
  const spectrum = axisOf(scene, "spectrum");
  const vertical = lineSeriesOn(scene, "spectrum").filter(
    (s) => s.x.length === 2 && s.x[0] === s.x[1] && s.y[0] !== s.y[1],
  );
  const xs = vertical.map((s) => s.x[0] as number);
  assert.ok(xs.some((x) => Math.abs(x - 128 / 3) < 1e-9), "缺少别名落点 N/3 的竖线");
  assert.ok(xs.some((x) => Math.abs(x - (2 * 128) / 3) < 1e-9), "缺少本尊 2N/3 的竖线");
  assert.ok(spectrum.xlim[1] >= (2 * 128) / 3, "x 轴必须延伸到 2N/3 才能标出本尊频率");
  assert.ok(xs.some((x) => Math.abs(x - 64) < 1e-9), "缺少 Nyquist N/2 的分界线");
  assert.match(textsOf(scene), /本尊/);
  assert.match(textsOf(scene), /别名/);
  assert.match(textsOf(scene), /波数/);
});

test("dealias_rule: 2/3+3/2 的扫描线压在所有账的最下面（贴舍入）", async () => {
  const scene = await sceneOf("dealias_rule");
  const scan = lineSeriesOn(scene, "scan");
  const deal = scan.find((s) => s.y.length === SCAN.length && Math.abs((s.y[0] as number) - 1e-16) < 1e-20);
  assert.ok(deal, "scan 轴上没有 2/3+3/2 的曲线");
  const alias = scan.find((s) => Math.abs((s.y[0] as number) - 0.066) < 1e-12);
  const trunc = scan.find((s) => Math.abs((s.y[0] as number) - 0.14) < 1e-12);
  const pad = scan.find((s) => Math.abs((s.y[0] as number) - 0.0016) < 1e-15);
  assert.ok(alias && trunc && pad, "scan 轴上缺混叠/截断/只补零三条账");
  const dealMax = Math.max(...Array.from(deal.y));
  const aliasMin = Math.min(...Array.from(alias.y));
  assert.ok(dealMax * 1e6 < aliasMin, "3/2 的残差应比真混叠小至少 6 个量级");
  assert.ok(Math.max(...Array.from(trunc.y)) > Math.max(...Array.from(alias.y)), "截断代价应大于混叠（两笔账）");
  assert.match(badgesOf(scene), /不是去混叠证据/);
});

test("convergence_race: 谱比值剔除 n_at_floor 之后的段；差分比值是常数", async () => {
  const scene = await sceneOf("convergence_race");
  const ratio = lineSeriesOn(scene, "ratio");
  const spectral = ratio.filter((s) => s.color === "#6b46c1");
  assert.ok(spectral.length > 0, "ratio 轴上没有谱方法的比值曲线");
  for (const s of spectral) {
    for (const x of Array.from(s.x)) {
      assert.ok((x as number) < 128, `谱比值画到了 n_at_floor=128 之后（x=${x}）`);
    }
  }
  const fd2 = ratio.find((s) => s.y.length === RN.length && Math.abs((s.y[0] as number) - 0.25) < 1e-12);
  const fd4 = ratio.find((s) => s.y.length === RN.length && Math.abs((s.y[0] as number) - 0.0625) < 1e-12);
  assert.ok(fd2 && fd4, "差分比值常数线缺失");
  for (const s of [fd2, fd4]) {
    const first = s.y[0] as number;
    for (const v of Array.from(s.y)) assert.ok(Math.abs((v as number) - first) < 1e-12, "差分比值应恒定");
  }
});

test("convergence_race: 差分理论阶参考线的斜率关系（N^-2 / N^-4）", async () => {
  const scene = await sceneOf("convergence_race");
  const lines = lineSeriesOn(scene, "error").filter((s) => s.x.length === 2 && s.x[0] !== s.x[1]);
  const slopeOf = (s: { x: ArrayLike<number>; y: ArrayLike<number> }): number =>
    Math.log((s.y[0] as number) / (s.y[1] as number)) / Math.log((s.x[1] as number) / (s.x[0] as number));
  const fd2Ref = lines.filter((s) => Math.abs(slopeOf(s) - 2) < 1e-6);
  const fd4Ref = lines.filter((s) => Math.abs(slopeOf(s) - 4) < 1e-6);
  assert.ok(fd2Ref.length > 0, "缺少 N^-2 理论阶参考线");
  assert.ok(fd4Ref.length > 0, "缺少 N^-4 理论阶参考线");
  assert.match(textsOf(scene), /FD2 -1\.986|FD2 -1\.99/);
  assert.match(textsOf(scene), /-3\.97/);
});

test("convergence_race: 退化帧（t=0）不画假的每加倍比值，并显式说明", async () => {
  const scene = await sceneOf("convergence_race", { meta: { degenerate: true, t_actual: 0 } });
  assert.deepEqual(validateScene(scene), []);
  const ratio = lineSeriesOn(scene, "ratio");
  // 只剩两条理论水平参考线（0.25 / 0.0625），没有任何数据曲线。
  const dataLines = ratio.filter((s) => s.y.length > 2);
  assert.equal(dataLines.length, 0, "退化帧里 ratio 轴上不应有数据曲线");
  assert.match(textsOf(scene), /退化帧/);
  assert.match(textsOf(scene), /比值无信息/);
});

test("burgers_cascade: 能谱拟合窗口与 k^-2 参考线在场，斜率与理论都写出来", async () => {
  const scene = await sceneOf("burgers_cascade");
  const frame = await makeFrame(BC_DEF);
  const series = lineSeriesOn(scene, "spectrum");
  const wantK = arrOf(frame, "k").slice(1);
  assert.ok(hasSeries(series, wantK, arrOf(frame, "energy_ref_slope").slice(1)), "缺少 k^-2 参考线");
  const vertical = series.filter((s) => s.x.length === 2 && s.x[0] === s.x[1] && s.y[0] !== s.y[1]);
  const vx = vertical.map((s) => s.x[0] as number);
  assert.ok(vx.some((x) => Math.abs(x - 3) < 1e-9) && vx.some((x) => Math.abs(x - 8) < 1e-9), "缺少拟合窗口竖线");
  assert.match(textsOf(scene), /斜率 -2\.2/);
  assert.match(textsOf(scene), /理论 -2/);
  assert.match(textsOf(scene), /R²=0\.99998/);
});

test("burgers_cascade: 激波示踪只在成形后出现；平滑初值不标激波", async () => {
  const sharp = await sceneOf("burgers_cascade");
  const solution = layersOn(sharp, "solution");
  const vertical = solution.filter(
    (l) => l.kind === "line" && l.series.some((s) => s.x.length === 2 && s.x[0] === s.x[1] && s.y[0] !== s.y[1]),
  );
  assert.ok(vertical.length > 0, "成形激波缺少位置竖线");
  assert.match(textsOf(sharp), /激波示踪/);
  assert.match(textsOf(sharp), /最大相邻落差/);

  const smooth = await sceneOf("burgers_cascade", {
    arrays: { u_now: XB.map((x) => Math.sin(x)) },
    meta: { max_u: 1, n_clipped: 0 },
  });
  assert.deepEqual(validateScene(smooth), []);
  const smoothVertical = layersOn(smooth, "solution").filter(
    (l) => l.kind === "line" && l.series.some((s) => s.x.length === 2 && s.x[0] === s.x[1] && s.y[0] !== s.y[1]),
  );
  assert.equal(smoothVertical.length, 0, "平滑正弦不该被标成激波");
  assert.match(textsOf(smooth), /激波未形成/);
});

test("burgers_cascade: 能流越窗（欠分辨）必须进 overflow，不能靠裁剪藏起来", async () => {
  const badFlux = KB.map((k) => (k === 64 ? -1 : 0.1));
  const scene = await sceneOf("burgers_cascade", { arrays: { flux: badFlux }, meta: { flux_at_end: -1 } });
  assert.deepEqual(validateScene(scene), []);
  assert.equal(scene.overflow.flag, true, "flux 低于 -0.3 时 overflow.flag 必须为真");
  assert.match(scene.overflow.note, /flux/);
  assert.ok(badgesOf(scene).includes("溢出"));
  assert.match(textsOf(scene), /欠分辨|低于窗口/);
});

// --------------------------------------------------------------------------- //
// 5. 降级点名 / 缺必需数组抛错
// --------------------------------------------------------------------------- //

test("缺可选掩码/meta → 不抛错，但 degraded 角标点名", async () => {
  const cases: Array<[string, Override, RegExp]> = [
    ["modal_decay", { drop: ["coeff_floor_mask"] }, /coeff_floor_mask/],
    ["modal_decay", { dropMeta: ["fit_p"] }, /fit_p/],
    ["dealias_rule", { drop: ["k_keep_line"] }, /k_keep_line/],
    ["dealias_rule", { drop: ["scan_floor_mask"] }, /scan_floor_mask/],
    ["convergence_race", { dropMeta: ["n_at_floor"] }, /n_at_floor/],
    ["convergence_race", { dropMeta: ["slope_fd2"] }, /slope_fd2/],
    ["burgers_cascade", { drop: ["energy_error_mask"] }, /energy_error_mask/],
    ["burgers_cascade", { drop: ["u_clipped"] }, /u_clipped/],
  ];
  for (const [viz, over, want] of cases) {
    const scene = await sceneOf(viz, over);
    assert.deepEqual(validateScene(scene), [], `${viz} 降级后仍要过校验器`);
    const degraded = degradedText(scene);
    assert.ok(degraded.length > 0, `${viz} 降级后必须有 degraded 角标`);
    assert.ok(
      want.test(degraded) || /等 \d+ 项/.test(degraded),
      `${viz} 降级说明里既没点名 ${want}，也没有「等 N 项」：${degraded}`,
    );
  }
});

test("缺必需数组 → 抛 ModelInputError（不静默画半张图）", async () => {
  for (const [viz, name] of [
    ["modal_decay", "coeff_theory"],
    ["dealias_rule", "scan_alias"],
    ["convergence_race", "ratio_fd2"],
    ["burgers_cascade", "budget_residual"],
  ] as const) {
    await assert.rejects(async () => sceneOf(viz, { drop: [name] }), ModelInputError);
  }
});

test("窄画布 480×520：四个 viz 都合法（说明文字放弃时点名降级）", async () => {
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz, {}, { width: 480, height: 520 });
    assert.deepEqual(validateScene(scene), [], `${viz} 在 480×520 下仍要合法`);
    for (const ax of scene.axes) assert.ok(ax.rect[2] > 0 && ax.rect[3] > 0, `${viz}.${ax.id} 宽高必须为正`);
    assert.match(degradedText(scene), /画布过矮|画布过窄|窄画布/);
  }
});

test("完整帧不降级，且 source/axisMode/status 角标齐全（burgers 还有 critical）", async () => {
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz);
    assert.deepEqual(validateScene(scene), [], `${viz} 完整帧必须合法`);
    assert.equal(degradedText(scene), "", `${viz} 完整帧不该出现降级角标`);
    const kinds = new Set(scene.badges.map((b) => b.kind));
    for (const kind of ["source", "autoscale", "status"] as const) {
      assert.ok(kinds.has(kind), `${viz} 缺少 ${kind} 角标`);
    }
    if (viz === "burgers_cascade") assert.ok(kinds.has("critical"), "burgers 的 nu 临界值角标缺失");
    if (scene.overflow.flag) assert.ok(scene.overflow.note.trim().length > 0, "溢出时必须写 note");
  }
});
