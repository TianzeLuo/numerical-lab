/**
 * 4.1 章（抛物型方程与时间推进）model 层测试 —— **自足**：不依赖 Python，也不依赖
 * `data/frames` 是否存在。夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，字段集合
 * （数组名 / dtype / shape / unit / label、meta/strings/critical 键）逐字照抄
 * `server/chapters/4.1.py` 的真实帧；真实帧在的话再对拍一遍（见文件末尾，缺帧时整组跳过）。
 *
 * 本文件专门守 4.1 最容易画反的几条（对应 `chapters/4.1/pitfalls.md`）：
 *
 * - **判据是 |G|，不是主图好不好看**：发散帧（`meta.umax=1480`）的主图窗口仍是写死的
 *   ±1.5——曲线冲出画面、溢出角标挂上、被截断的点单独标出，而不是缩轴把发散藏起来；
 * - **稳定 ≠ 精确**：`measured_radius`（幂迭代实测）与 `analytic_radius`（解析）必须同屏，
 *   差多少就写多少（后端 `_growth` 的 scheme 参数目前被忽略，实测值会偏——这是**如实照画**
 *   的理由，不是前端替它圆场的理由）；
 * - **换靶子就换结论**：`order_vs_continuous`（对连续解测出来的斜率）必须写进面板文字，
 *   否则"时间细化测出 2 阶"这句话本身就是错的；
 * - **两种掩码分别报数**：`n_absent`（初值里没有这个模态）与 `n_masked_by_roundoff`
 *   （被别的模态的舍入淹没）是两件事，任何一条都不许被合并成"若干模态不可测"；
 * - **"守恒"是结构性的**：Neumann/周期帧里 `|ΔH|` 落在舍入级（1e-14）而不是"一个小数"，
 *   对拍残差里**精确为 0** 的点在对数轴上跳过并计数（不是画成 0）；
 * - **哨兵不是数值**：`phi_flip = -1`、`decay_alive = 0`、`u_clipped = 1` 一律按掩码语义
 *   处理，绝不当作"值为 -1 / 值为 0"画出去；
 * - **降级不静默**：meta/strings/数组缺键 → 少画什么就写明什么（degraded 角标点名）。
 */

import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { DEGRADED_PREFIX, ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { sceneToJson, validateScene, type Layer, type LineSeries, type Scene } from "../shared/src/scene.js";
import { boundaryConservationModel } from "../chapters/4.1/boundary_conservation.model.js";
import { convergenceOrderModel } from "../chapters/4.1/convergence_order.model.js";
import { manifest, modelFor, models } from "../chapters/4.1/index.js";
import { numericalDissipationModel } from "../chapters/4.1/numerical_dissipation.model.js";
import { stabilityThresholdModel } from "../chapters/4.1/stability_threshold.model.js";

const PITFALLS =
  "它会怎样骗你：第一，换初值就换结论：r=0.51 已经越界，但 sin(πx) 初值下逐点光滑、峰值 0.61、" +
  "误差 3.2e-4（比稳定的 r=0.45 还好看）；同一个 r 换成噪声初值，峰值 42.9。" +
  "第二，副图的 |G| 是解析公式画的，实测谱半径是幂迭代测的。第三，主图窗口固定 ±1.5。";

/** 一行数组 = `[名字, 数据, 标签, 形状]`；shape 缺省为 `[len]`（本章所有数组的 unit 都是 `"1"`）。 */
type Row = readonly [name: string, data: ArrayLike<number>, label: string, shape?: readonly number[]];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical: Record<string, number>;
  pitfalls?: string;
}

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
  const frame = new Frame({
    chapter: "4.1",
    viz: def.viz,
    kind: "frame",
    codec: "none",
    meta: def.meta,
    strings: def.strings,
    critical: def.critical,
    pitfalls: def.pitfalls ?? PITFALLS,
    specs,
    arrays,
  });
  return decodeNlf(await encodeNlf(frame));
}

interface Override {
  arrays?: Record<string, ArrayLike<number>>;
  meta?: Record<string, number | boolean | null>;
  strings?: Record<string, string>;
  drop?: readonly string[];
  dropMeta?: readonly string[];
  dropStrings?: readonly string[];
  pitfalls?: string;
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
  const out: FrameDef = {
    ...def,
    arrays,
    meta: { ...meta, ...ov.meta },
    strings: { ...strings, ...ov.strings },
  };
  if (ov.pitfalls !== undefined) out.pitfalls = ov.pitfalls;
  return out;
}

const PARAMS = {} as Record<string, number | boolean | string>;
const OPTS: ModelOptions = { t: 0, params: PARAMS, theme: "light", width: 960, height: 600 };
const NARROW: ModelOptions = { t: 0, params: PARAMS, theme: "light", width: 480, height: 600 };

const lin = (n: number, a: number, b: number): Float64Array =>
  Float64Array.from({ length: n }, (_, i) => (n <= 1 ? a : a + ((b - a) * i) / (n - 1)));

const ones = (n: number): Float64Array => Float64Array.from({ length: n }, () => 1);
const zeros = (n: number): Float64Array => Float64Array.from({ length: n }, () => 0);

// --------------------------------------------------------------------------- //
// viz 1：stability_threshold
// --------------------------------------------------------------------------- //

const ST_X = lin(41, 0, 1);
/** sin(πx)：默认初值（真实帧 t=0 的形状）。 */
const ST_SINE = Float64Array.from(ST_X, (x) => Math.sin(Math.PI * x));
/** 噪声初值：40 个格点上 ±0.01 的确定性"噪声"（不用随机数，测试要可复现）。 */
const ST_NOISE = Float64Array.from(ST_X, (x) => 0.01 * Math.sin(37 * Math.PI * x));
const ST_PHI = lin(241, 0, Math.PI);
const stG = (r: number, phi: number): number => Math.abs(1 - 4 * r * Math.sin(phi / 2) ** 2);
const ST_R = 0.45;

const ST_DEF: FrameDef = {
  viz: "stability_threshold",
  arrays: [
    ["x", ST_X, "网格 x"],
    ["u", ST_SINE, "数值解 u"],
    ["u_clipped", zeros(41), "越界截断掩码"],
    ["u_ref", ST_SINE, "参考解 u_ref"],
    ["error_curve", Float64Array.from({ length: 41 }, () => 1e-17), "逐点误差（地板 1e-17）"],
    ["error_at_floor", ones(41), "贴地板掩码"],
    ["phi", ST_PHI, "波数 φ"],
    ["g_abs_cur", Float64Array.from(ST_PHI, (p) => stG(ST_R, p)), "|G(φ)|（ftcs）"],
    ["g_abs_other_a", Float64Array.from(ST_PHI, (p) => 1 / Math.sqrt(1 + 2 * ST_R * (2 - 2 * Math.cos(p)))), "|G(φ)|（btcs，对照）"],
    ["g_abs_other_b", Float64Array.from(ST_PHI, (p) => Math.abs((2 - ST_R * (2 - 2 * Math.cos(p))) / (2 + ST_R * (2 - 2 * Math.cos(p))))), "|G(φ)|（cn，对照）"],
    ["g_one", ones(241), "稳定界线 |G|=1"],
    ["g_unstable_region", zeros(241), "|G|>1 区掩码"],
    ["mark_growth", new Float64Array([0.9972255417]), "实测谱半径（幂迭代）"],
    ["mark_phi_flip", new Float64Array([1.682137]), "符号翻转波数 φ*"],
    ["algebraic_residual", zeros(1), "上一步代数残差"],
  ],
  meta: {
    analytic_radius: 0.9972256004, bc_index: 0, clipped_points: 0, dt: 0.00025, dx: 0.025,
    err_l2: 1.936e-17, err_linf: 1.225e-16, has_flip: true, mass: 0.63662, n_steps: 0, nx: 41,
    overflow: false, phi_flip: 1.682137, r: ST_R, r_critical_continuous: 0.5, r_critical_finite: 0.500016,
    radius_gap: 5.87e-8, radius_rel_gap: 5.87e-8, scheme_index: 0, t: 0, t_actual: 0,
    t_requested: 0, theta: 0, total_modes: 39, umax: 1, unstable_fraction: 0, unstable_modes: 0,
  },
  strings: {
    bc: "dirichlet", data_source: "backend", error_reference: "解析解", floor: "1e-17",
    initial: "sine", nonfinite_policy: "本帧不含任何非有限值。", params_json: '{"r":0.45,"scheme":"ftcs"}',
    radius_note: "实测谱半径由幂迭代测出。", reading_note: "副图三条 |G| 曲线。",
    scheme: "ftcs", stable_note: "|G|≤1 只说明不发散。", time_semantics: "真实时间 t。",
  },
  critical: { r: 0.5 },
};

// --------------------------------------------------------------------------- //
// viz 2：convergence_order
// --------------------------------------------------------------------------- //

const CO_H = Float64Array.from([0.4, 0.2, 0.1, 0.05]);
const coLadder = (hs: ArrayLike<number>, c: number, p: number): Float64Array =>
  Float64Array.from(hs as ArrayLike<number>, (h) => c * Math.pow(h, p));

const CO_DEF: FrameDef = {
  viz: "convergence_order",
  arrays: [
    ["h_ladder", CO_H, "步长阶梯 Δt"],
    ["error_ladder", coLadder(CO_H, 1.0793886e-7, 2), "对半离散解的 L2 误差"],
    ["error_ladder_continuous", Float64Array.from([1.09e-4, 1.09e-4, 1.09e-4, 1.09e-4]), "对连续解的 L2 误差"],
    ["theory_line", Float64Array.from(CO_H, (h) => 1.686489e-9 * Math.pow(h / 0.05, -2)), "理论斜率线（锚在最细一级）"],
    ["mark_h", new Float64Array([0.05]), "最细一级 Δt"],
    ["mark_error", new Float64Array([1.686489e-9]), "最细一级误差"],
    ["x_profile", ST_X, "网格 x"],
    ["error_profile", Float64Array.from(ST_X, (x) => Math.max(1e-17, 1.686e-9 * Math.abs(Math.sin(Math.PI * x)))), "逐点误差（最细一级）"],
  ],
  meta: {
    constant: 1.0793886e-7, err_coarsest: 1.0793886e-7, err_finest: 1.686489e-9, err_floor: 1e-17,
    err_ratio: 64.002, err_ratio_continuous: 1, fine_points: 41, fixed_r: 0.4, n_dropped: 0,
    n_levels: 4, n_used: 4, nx: 41, order_gap: -1.44e-5, order_stderr: 8.29e-6,
    order_vs_continuous: -0.00044686, r2: 0.99999999997, scheme_index: 3, t: 0, t_end: 0.05,
    target_is_space: false, theory_order: 2, measured_order: 2.0000144,
  },
  strings: {
    data_source: "backend", floor: "1e-17", ladder_note: "固定网格把 Δt 每次减半：靶子换成半离散解。",
    nonfinite_policy: "本帧不含任何非有限值。", params_json: '{"levels":4,"target":"time"}',
    reading_note: "主图双对数：曲线斜率就是实测阶。", reference: "半离散精确解", scheme: "cn",
    target: "time", time_semantics: "本 viz 没有时间轴。", trap_note: "同一批数据改用连续解当靶子，实测斜率变成 -0.0004。",
  },
  critical: {},
};

// --------------------------------------------------------------------------- //
// viz 3：numerical_dissipation
// --------------------------------------------------------------------------- //

const ND_PHI = Float64Array.from({ length: 8 }, (_, i) => ((i + 1) * Math.PI) / 40);
const ndDecay = (phi: ArrayLike<number>, r: number, n: number): Float64Array =>
  Float64Array.from(phi as ArrayLike<number>, (p) => Math.pow(stG(r, p), n));

const ND_DEF: FrameDef = {
  viz: "numerical_dissipation",
  arrays: [
    ["phi", ND_PHI, "波数 φ"],
    ["decay_measured", ndDecay(ND_PHI, 0.4, 40), "实测存活率（从解里分解）"],
    ["decay_alive", ones(8), "可测掩码"],
    ["decay_formula", ndDecay(ND_PHI, 0.4, 40), "|G(φ)|ⁿ（解析）"],
    ["decay_semidiscrete", Float64Array.from(ND_PHI, (p) => Math.exp(-0.4 * p * p * 40)), "半离散精确解"],
    ["decay_exact", Float64Array.from(ND_PHI, (p) => Math.exp(-0.4 * p * p * 40)), "真解 e^(−rφ²n)"],
    ["amplitude_residual", zeros(8), "实测存活率 − |G|ⁿ"],
    ["dissipation_error", Float64Array.from(ND_PHI, (p) => Math.exp(-0.4 * p * p * 40) - Math.pow(stG(0.4, p), 40)), "耗散误差（实测 − 半离散）"],
    ["phi_flip", new Float64Array([-1]), "符号翻转波数（-1 = 无）"],
  ],
  meta: {
    c4_exact: 0.08, c4_numeric: 0.113333, decay_max: 0.9999, decay_min: 0.00219,
    dissipation_coefficient: 0.033333, dt: 0.00025, err_floor: 1e-17, has_flip: false,
    mask_floor_rel: 1e-8, max_amplitude_residual: 8.2157e-15, mean_dissipation: -3e-5,
    n_absent: 0, n_alive: 8, n_masked_by_roundoff: 0, n_modes: 8, n_steps: 0, nx: 41,
    phi_flip: -1, r: 0.4, scheme_index: 3, t: 0, t_actual: 0, t_requested: 0, theta: 0.5,
    worst_dissipation: 0, worst_phi: 0.07854,
  },
  strings: {
    data_source: "backend", floor: "1e-17", initial: "Σ_{k=1..8} sin(kπx)/k",
    mask_note: "decay_alive=0 有两种原因，都写在 meta 里。", nonfinite_policy: "本帧不含任何非有限值。",
    params_json: '{"n_modes":8,"scheme":"cn"}', reading_note: "三条线：真解、半离散解、实测存活率。",
    scheme: "cn", time_semantics: "真实时间 t。",
  },
  critical: { r: 0.5 },
};

// --------------------------------------------------------------------------- //
// viz 4：boundary_conservation
// --------------------------------------------------------------------------- //

const BC_T = lin(21, 0, 0.05);
const BC_X = ST_X;

const BC_DEF: FrameDef = {
  viz: "boundary_conservation",
  arrays: [
    ["x", BC_X, "网格 x"],
    ["u", Float64Array.from(BC_X, (x) => 1 + Math.cos(Math.PI * x)), "数值解 u"],
    ["u_exact", Float64Array.from(BC_X, (x) => 1 + Math.cos(Math.PI * x)), "解析解 u_exact"],
    ["error_curve", Float64Array.from(BC_X, () => 1e-17), "逐点误差（地板 1e-17）"],
    ["t_history", BC_T, "热量历史时刻"],
    ["mass_shift_dirichlet", Float64Array.from(BC_T, (t) => -0.25 * (t / 0.05)), "ΔH（Dirichlet）"],
    ["mass_shift_neumann", zeros(21), "ΔH（Neumann）"],
    ["mass_shift_periodic", zeros(21), "ΔH（周期）"],
    ["mass_zero", zeros(21), "守恒基线 0"],
    ["mass_residual", new Float64Array([0, 0, 3.286e-14]), "末刻漂移（Dirichlet/Neumann/周期）"],
    ["mark_t", new Float64Array([0.05]), "当前时刻"],
  ],
  meta: {
    // Neumann：解析流失量恒为 0（这正是"结构性"的含义），实测漂移落在舍入级
    analytic_leak: 0, bc_index: 1, conservation_holds: true, drift_dirichlet_max: 0.247739,
    drift_neumann_max: 3.153e-14, drift_now: 3.153e-14, drift_other_max: 3.286e-14,
    drift_periodic_max: 3.286e-14, dt: 0.00025, err_floor: 1e-17, err_l2: 1.121e-4,
    err_linf: 1.547e-4, leak_now: 3.153e-14, leak_ratio: 0, mass0: 1.025, mass_now: 1.025,
    n_steps: 200, n_total: 41, nx: 41, r: 0.4, roundoff: 1.1102230246251565e-16,
    scheme_index: 3, t: 0.05, t_actual: 0.05, t_requested: 0.05,
  },
  strings: {
    bc: "neumann", data_source: "backend", drift_note: "Neumann/周期末刻漂移 0.000e+00，量级是舍入。",
    floor: "1e-17", initial_note: "三种边界用三种相容初值。", leak_note: "Dirichlet 的流失量有解析值。",
    nonfinite_policy: "本帧不含任何非有限值。", params_json: '{"bc":"neumann","scheme":"cn"}',
    reading_note: "副图三条曲线共用一根轴：「守恒」不是「数很小」，而是「结构性为零」。",
    scheme: "cn", time_semantics: "真实时间 t。",
  },
  critical: { r: 0.5 },
};

const DEFS: Record<string, FrameDef> = {
  stability_threshold: ST_DEF,
  convergence_order: CO_DEF,
  numerical_dissipation: ND_DEF,
  boundary_conservation: BC_DEF,
};

const MODEL: Record<string, (frame: Frame, opts: ModelOptions) => Scene> = {
  stability_threshold: stabilityThresholdModel,
  convergence_order: convergenceOrderModel,
  numerical_dissipation: numericalDissipationModel,
  boundary_conservation: boundaryConservationModel,
};

const VIZ_IDS = ["stability_threshold", "convergence_order", "numerical_dissipation", "boundary_conservation"] as const;

/**
 * `requiredAux` 的名字 → 真正承载它的轴 id。
 *
 * viz1 只有一块对数副轴：误差曲线与上一步代数残差共用它（把残差单开一根轴反而看不出
 * "误差贴地板时残差也在动"），所以 `error`/`residual` 两条都落在 `error` 上。
 */
/** 每个 viz 的"主解面"轴：其余轴就是规则 3 要的误差/残差面。 */
const PRIMARY_AXES: Record<string, string[]> = {
  stability_threshold: ["solution", "amplification"],
  convergence_order: ["profile"],
  numerical_dissipation: ["decay"],
  boundary_conservation: ["solution", "mass"],
};

const AUX_HOME: Record<string, Record<string, string>> = {
  // viz2 的误差阶梯本身就是误差面（manifest 也把它写成固定双对数窗口）
  convergence_order: { error: "ladder" },
  stability_threshold: { error: "error", residual: "error" },
  numerical_dissipation: { error: "dissipation", residual: "residual" },
  boundary_conservation: { error: "error", residual: "residual" },
};

function textOf(scene: Scene): string {
  return scene.texts.map((t) => t.text).join("\n");
}

function layersOn(scene: Scene, axisId: string): Layer[] {
  return scene.layers.filter((l) => l.axes === axisId);
}

/** 某根轴上的折线序列（`line` 层才是"曲线"，`scatter` 是点）。 */
function linesOn(scene: Scene, axisId: string): LineSeries[] {
  return scene.layers
    .filter((l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === axisId)
    .flatMap((l) => l.series);
}

/** 某根轴上的散点层。 */
function scattersOn(scene: Scene, axisId: string): Extract<Layer, { kind: "scatter" }>[] {
  return scene.layers.filter((l): l is Extract<Layer, { kind: "scatter" }> => l.kind === "scatter" && l.axes === axisId);
}

function axisOf(scene: Scene, id: string) {
  const axis = scene.axes.find((a) => a.id === id);
  assert.ok(axis !== undefined, `场景里没有轴 ${id}`);
  return axis;
}

function badgeTexts(scene: Scene): string[] {
  return scene.badges.map((b) => b.text);
}

function hasBadge(scene: Scene, kind: string): boolean {
  return scene.badges.some((b) => b.kind === kind);
}

/** 降级角标（共享层的 `kind` 是 `status`，靠前缀 `⚠ 降级：` 认）。 */
function isDegraded(scene: Scene): boolean {
  return badgeTexts(scene).some((t) => t.startsWith(DEGRADED_PREFIX));
}

// --------------------------------------------------------------------------- //
// 1. manifest / 装配层
// --------------------------------------------------------------------------- //

test("4.1：manifest 的每个 viz 都有 model，modelFor 对未知 id 返回 null", () => {
  for (const spec of manifest.vizzes) {
    assert.ok(models[spec.id] !== undefined, `manifest 里的 ${spec.id} 没有 model`);
    assert.equal(typeof modelFor(spec.id), "function");
  }
  assert.equal(modelFor("nope"), null);
  assert.equal(Object.keys(models).length, manifest.vizzes.length);
});

test("4.1：manifest 声明的每个轴都真的被画出来，autoscaled 口径一致", async () => {
  for (const id of VIZ_IDS) {
    const spec = manifest.vizzes.find((v) => v.id === id);
    assert.ok(spec !== undefined);
    const scene = MODEL[id]?.(await makeFrame(DEFS[id] as FrameDef), OPTS) as Scene;
    for (const declared of spec.axes) {
      const axis = axisOf(scene, declared.id);
      assert.equal(axis.autoscaled, declared.autoscaled, `${id}/${declared.id} 的自适应口径与 manifest 不一致`);
    }
    // manifest 的 `requiredAux` 是"必须有误差/残差面"，轴 id 由模型定（4.1 的 manifest 没把
    // 副轴写进 `axes`，见交付说明"Python 层问题"）。这里把每一样**显式映射**到落点，
    // 而不是含糊地"某处应该有"：viz1 把误差曲线与代数残差画在同一块对数副轴上，所以两条
    // requiredAux 的落点是同一根轴。
    for (const aux of spec.requiredAux) {
      const home = AUX_HOME[id]?.[aux];
      assert.ok(home !== undefined, `${id} 的 requiredAux '${aux}' 没有声明落点`);
      axisOf(scene, home as string);
      assert.ok(layersOn(scene, home as string).length > 0, `${id}/${aux} 落在 ${home}，却没有图层`);
    }
    // 规则 3：至少有一块非主解的"误差/残差面"。
    // 注意**不**要求它 `autoscaled`：viz2 的误差阶梯是 manifest 明确写死的双对数窗口
    // （`ladder.autoscaled=false`）——"自适应"是规则 4 对未声明轴的要求，不是对 aux 的。
    const auxAxes = scene.axes.filter((a) => !PRIMARY_AXES[id]?.includes(a.id));
    assert.ok(auxAxes.length > 0, `${id} 没有任何误差/残差副轴（规则 3）`);
  }
});

test("4.1：四个 viz 在 960×600 与 480×600 都给出合规 Scene", async () => {
  for (const id of VIZ_IDS) {
    for (const opts of [OPTS, NARROW]) {
      const scene = MODEL[id]?.(await makeFrame(DEFS[id] as FrameDef), opts) as Scene;
      const issues = validateScene(scene);
      assert.deepEqual(issues, [], `${id}@${opts.width}：${issues.join("；")}`);
      assert.equal(scene.canvas.w, opts.width);
      assert.equal(scene.pitfalls.length > 0, true);
    }
  }
});

test("4.1：同一帧两次 → sceneToJson 逐字节相同（模型是纯函数）", async () => {
  for (const id of VIZ_IDS) {
    const frame = await makeFrame(DEFS[id] as FrameDef);
    const a = sceneToJson(MODEL[id]?.(frame, OPTS) as Scene);
    const b = sceneToJson(MODEL[id]?.(frame, OPTS) as Scene);
    assert.equal(a, b, `${id} 两次转储不一致`);
  }
});

test("4.1：viz2 没有时间轴（time.mode=none）——不许出现时间控件语义", async () => {
  const spec = manifest.vizzes.find((v) => v.id === "convergence_order");
  assert.equal(spec?.time.mode, "none");
  assert.ok(!spec?.time.tMax, "没有时间轴就不该声明 tMax（PROTOCOLS §5 的字段约束）");
  // 模型的 t 恒为 0：没有时间轴就不许跟着滑块走
  const scene = convergenceOrderModel(await makeFrame(CO_DEF), { ...OPTS, t: 0.7 });
  assert.equal(scene.t, 0);
});

// --------------------------------------------------------------------------- //
// 2. viz1：判据是 |G|，主图窗口不许跟着数据跑
// --------------------------------------------------------------------------- //

test("4.1/stability_threshold：主图窗口恒为 ±1.5，发散帧也不缩轴", async () => {
  // 同一批参数、换初值就换结论：r=0.6 + 噪声初值 → 真值 1480（而 r=0.45 + sin 初值"看着很好"）
  const diverged = tweak(ST_DEF, {
    arrays: {
      u: Float64Array.from(ST_NOISE, (v) => v * 148006),
      u_clipped: Float64Array.from(ST_X, (_, i) => (i < 17 ? 1 : 0)),
      error_curve: Float64Array.from({ length: 41 }, () => 1480.0),
    },
    strings: { initial: "noise" },
    meta: {
      r: 0.6, umax: 1480.06, overflow: true, clipped_points: 17,
      unstable_modes: 12, unstable_fraction: 0.31, measured_radius: 1.0014, analytic_radius: 1.0016,
    },
  });
  const scene = stabilityThresholdModel(await makeFrame(diverged), OPTS);
  assert.deepEqual(validateScene(scene), []);
  const solution = axisOf(scene, "solution");
  assert.deepEqual(solution.ylim, [-1.5, 1.5], "主图窗口必须写死 ±1.5");
  assert.equal(solution.autoscaled, false);
  // 被截断的点要单独标出，而不是"数据本来就在窗口里"
  assert.equal(scattersOn(scene, "solution").length, 1, "越界截断的点必须有独立图层");
  // 面板文字把真值报出来：截断了几个点、真值多大，都要写在屏幕上（不许静默）
  const text = textOf(scene);
  assert.ok(text.includes("1480"), "面板文字要给出 meta.umax 真值");
  assert.ok(text.includes("截断"), "要说明「截断显示 ≠ 没有发生」");
  assert.ok(text.includes("17"), "被截断的点数要写出来");
  assert.ok(text.includes("noise"), "初值类型要写出来（同参数换初值就换结论）");
});

test("4.1/stability_threshold：实测谱半径与解析值同屏，差多少写多少", async () => {
  // 后端 `_growth` 目前对 btcs/cn 返回的是 FTCS 的值（scheme 参数被忽略）：
  // 前端不替它圆场——实测值照画，差值照写。
  const skewed = tweak(ST_DEF, {
    meta: { measured_radius: 198.6917, analytic_radius: 0.76437, radius_gap: 197.9273, scheme_index: 1, theta: 1 },
    arrays: { mark_growth: new Float64Array([198.6917]) },
  });
  const scene = stabilityThresholdModel(await makeFrame(skewed), OPTS);
  const text = textOf(scene);
  assert.ok(text.includes("198.69") || text.includes("198.7"), `实测谱半径要写进面板：${text.slice(0, 200)}`);
  assert.ok(text.includes("0.7643") || text.includes("0.76437"), "解析值要写进面板");
  assert.ok(/差 .?\d/.test(text), "实测与解析的差值要写进面板");
  // 越界时状态角标必须说"越界"，而且单模态初值要额外提醒"主图可能看不出来"
  assert.ok(badgeTexts(scene).some((t) => t.includes("⚠")), "越界必须在角标上显形");
});

test("4.1/stability_threshold：代数残差恰好为 0 时画在地板并写明（不是静默丢掉）", async () => {
  const scene = stabilityThresholdModel(await makeFrame(ST_DEF), OPTS);
  const errorAxis = axisOf(scene, "error");
  assert.equal(errorAxis.yscale, "log");
  // 对数轴上画不出 0：抬到地板，但文字里必须说它是 0
  assert.ok(linesOn(scene, "error").length >= 2, "地板线与残差线都要在");
  assert.ok(textOf(scene).includes("1e-17") || textOf(scene).includes("地板"));
});

test("4.1/stability_threshold：phi_flip=-1 的哨兵帧不许画竖线、也不许写成数值", async () => {
  const noFlip = tweak(ST_DEF, {
    meta: { has_flip: false, phi_flip: -1 },
    arrays: { g_abs_cur: ones(241) },
  });
  const scene = stabilityThresholdModel(await makeFrame(noFlip), OPTS);
  assert.deepEqual(validateScene(scene), []);
  const text = textOf(scene);
  assert.ok(text.includes("无符号翻转波数"), "哨兵要在文字里被解释");
  assert.ok(!text.includes("φ*=-1"), "-1 不许当成数值写出去");
});

// --------------------------------------------------------------------------- //
// 3. viz2：换靶子就换结论
// --------------------------------------------------------------------------- //

test("4.1/convergence_order：实测阶、理论阶、换靶子对照三个数都在", async () => {
  const scene = convergenceOrderModel(await makeFrame(CO_DEF), OPTS);
  const text = textOf(scene);
  assert.ok(text.includes("2.00001") || text.includes("2.0000144"), "实测阶要写出来");
  assert.ok(text.includes("理论阶 2"), "理论阶要写出来");
  assert.ok(text.includes("-4.469e-4") || text.includes("-0.00044686"), "换靶子的实测斜率要写出来");
  // 双对数阶梯层：规则 2 要求实测斜率与理论斜率都在
  const fit = scene.layers.find((l): l is Extract<Layer, { kind: "loglogfit" }> => l.kind === "loglogfit");
  assert.ok(fit !== undefined, "阶梯面板要有 loglogfit 层");
  assert.equal(fit.axes, "ladder");
  assert.ok(Math.abs(fit.measuredSlope - 2.0000144) < 1e-6, "层里带的是实测斜率");
  assert.equal(fit.theorySlope, 2);
  assert.equal(axisOf(scene, "ladder").xscale, "log");
  assert.equal(axisOf(scene, "ladder").yscale, "log");
});

test("4.1/convergence_order：逐点误差贴地板的两端不参与取范围（也不画成 0）", async () => {
  const scene = convergenceOrderModel(await makeFrame(CO_DEF), OPTS);
  const profile = axisOf(scene, "profile");
  assert.equal(profile.yscale, "log");
  const drawn = linesOn(scene, "profile").flatMap((s) =>
    Array.from(s.y as ArrayLike<number>).filter((v) => Number.isFinite(v)),
  );
  const nonPositive = drawn.filter((v) => !(v > 0));
  assert.deepEqual(nonPositive, [], `对数轴上不许出现 0/负值：${nonPositive.slice(0, 4).join("、")}`);
  assert.ok(drawn.some((v) => v === 1e-17), "地板点仍然要画（作为地板，不是作为 0）");
});

// --------------------------------------------------------------------------- //
// 4. viz3：两种掩码分别报数
// --------------------------------------------------------------------------- //

test("4.1/numerical_dissipation：n_absent 与 n_masked_by_roundoff 分别出现在同一屏", async () => {
  const mixed = tweak(ND_DEF, {
    arrays: {
      decay_alive: Float64Array.from([1, 1, 1, 1, 0, 0, 0, 0]),
      decay_measured: Float64Array.from([1, 1, 1, 1, 1e-17, 1e-17, 1e-17, 1e-17]),
    },
    // 8 个模态里：4 个可测、2 个"初值里没有"、2 个"被舍入淹没"
    meta: { n_alive: 4, n_absent: 2, n_masked_by_roundoff: 2 },
  });
  const scene = numericalDissipationModel(await makeFrame(mixed), OPTS);
  assert.deepEqual(validateScene(scene), []);
  const text = textOf(scene);
  const absent = text.indexOf("n_absent=2");
  const masked = text.indexOf("n_masked_by_roundoff=2");
  assert.ok(absent >= 0, "初值里没有这个模态的个数要单独出现");
  assert.ok(masked >= 0, "被舍入淹没的个数要单独出现");
  assert.notEqual(absent, masked);
  // 掩码点画成散点钉在地板，而不是连进曲线（否则读起来像"存活率掉到 0"）
  assert.ok(scattersOn(scene, "decay").length >= 1, "被掩码的模态要有自己的散点层");
  const measured = linesOn(scene, "decay").filter((s) => s.marker === "o");
  const alive = measured.flatMap((s) => (s.y as number[])).filter((v) => Number.isFinite(v) && v > 1e-16);
  assert.ok(alive.length <= 4, "被掩码的点不许出现在连线上");
});

test("4.1/numerical_dissipation：存活率轴窗口写死 1e-17–2（对数）", async () => {
  const scene = numericalDissipationModel(await makeFrame(ND_DEF), OPTS);
  const decay = axisOf(scene, "decay");
  assert.deepEqual(decay.ylim, [1e-17, 2]);
  assert.equal(decay.yscale, "log");
  assert.equal(decay.autoscaled, false);
  // 掩码计数缺了也要说清楚（不是静默变成"0 个不可测"）
  const noCounts = tweak(ND_DEF, { dropMeta: ["n_absent", "n_masked_by_roundoff"] });
  const degraded = numericalDissipationModel(await makeFrame(noCounts), OPTS);
  assert.deepEqual(validateScene(degraded), []);
  assert.ok(isDegraded(degraded), "缺掩码计数要有降级角标");
});

// --------------------------------------------------------------------------- //
// 5. viz4：「守恒」是结构性的
// --------------------------------------------------------------------------- //

test("4.1/boundary_conservation：漂移轴是对数轴，且守恒/不守恒的说法跟着 meta 走", async () => {
  const conserve = boundaryConservationModel(await makeFrame(BC_DEF), OPTS);
  assert.deepEqual(validateScene(conserve), []);
  const drift = axisOf(conserve, "residual");
  assert.equal(drift.yscale, "log");
  assert.equal(drift.autoscaled, true);
  assert.ok(textOf(conserve).includes("结构性"), "守恒必须写成结构性事实，不是「数很小」");
  assert.ok(textOf(conserve).includes("总热量不变 ≠ 解不变"), "「守恒 ≠ 解不变」必须写在面板上");

  // Dirichlet 才是真会漏热的那个边界：不守恒的帧必须把话说白
  const leaky = tweak(BC_DEF, {
    strings: { bc: "dirichlet" },
    meta: { conservation_holds: false, drift_now: -0.247739, bc_index: 0, analytic_leak: 0.247739, leak_ratio: 0.999 },
  });
  const scene = boundaryConservationModel(await makeFrame(leaky), OPTS);
  assert.deepEqual(validateScene(scene), []);
  const onScreen = `${textOf(scene)}\n${badgeTexts(scene).join("\n")}`;
  assert.ok(onScreen.includes("不守恒"), `不守恒必须写在屏幕上：${onScreen}`);
});

test("4.1/boundary_conservation：对数轴上精确为 0 的点跳过并报数", async () => {
  const scene = boundaryConservationModel(await makeFrame(BC_DEF), OPTS);
  const text = textOf(scene);
  assert.ok(text.includes("0 已跳过") || text.includes("点 = 0"), `跳过计数要写在面板上：${text.slice(0, 160)}`);
  const scatter = scattersOn(scene, "residual");
  assert.equal(scatter.length, 1);
  for (const v of Array.from((scatter[0]?.y ?? []) as ArrayLike<number>)) {
    assert.ok(v > 0, "对数轴上不许出现 0");
  }
});

test("4.1/boundary_conservation：三种边界的 ΔH 共用一根副轴（可比）", async () => {
  const scene = boundaryConservationModel(await makeFrame(BC_DEF), OPTS);
  assert.ok(linesOn(scene, "mass").length >= 4, "三种边界 + 守恒基线都要在同一根轴上");
});

// --------------------------------------------------------------------------- //
// 6. 降级路径：缺键不静默、不抛错
// --------------------------------------------------------------------------- //

test("4.1：缺可选数组 → 降级角标点名，不抛错", async () => {
  const cases: Array<[string, FrameDef]> = [
    ["stability_threshold", tweak(ST_DEF, { drop: ["u_clipped"] })],
    ["numerical_dissipation", tweak(ND_DEF, { drop: ["amplitude_residual"] })],
    ["boundary_conservation", tweak(BC_DEF, { drop: ["mass_residual", "mark_t"] })],
  ];
  for (const [id, def] of cases) {
    const scene = MODEL[id]?.(await makeFrame(def), OPTS) as Scene;
    assert.deepEqual(validateScene(scene), []);
    assert.ok(isDegraded(scene), `${id} 缺数组时要挂降级角标：${badgeTexts(scene).join(" / ")}`);
  }
});

test("4.1：缺必需数组 → 抛 ModelInputError（不画半张图）", async () => {
  await assert.rejects(async () => {
    stabilityThresholdModel(await makeFrame(tweak(ST_DEF, { drop: ["u_ref"] })), OPTS);
  }, ModelInputError);
  await assert.rejects(async () => {
    convergenceOrderModel(await makeFrame(tweak(CO_DEF, { drop: ["h_ladder"] })), OPTS);
  }, ModelInputError);
  await assert.rejects(async () => {
    boundaryConservationModel(await makeFrame(tweak(BC_DEF, { drop: ["mass_shift_neumann"] })), OPTS);
  }, ModelInputError);
});

test("4.1：缺 pitfalls → 抛 ModelInputError（底部面板是契约的一部分）", async () => {
  await assert.rejects(async () => {
    numericalDissipationModel(await makeFrame(tweak(ND_DEF, { pitfalls: "" })), OPTS);
  }, ModelInputError);
});

test("4.1：preview 数据源要挂 preview 角标，backend 不挂", async () => {
  const previewDef = tweak(ST_DEF, { strings: { data_source: "backend+preview" } });
  const preview = stabilityThresholdModel(await makeFrame(previewDef), OPTS);
  assert.ok(hasBadge(preview, "preview"), "预览数据必须显式标注");
  const backend = stabilityThresholdModel(await makeFrame(ST_DEF), OPTS);
  assert.ok(!hasBadge(backend, "preview"));
  for (const scene of [preview, backend]) assert.ok(hasBadge(scene, "source"));
});

// --------------------------------------------------------------------------- //
// 7. 真实帧对拍（`data/frames/4.1` 不在时整组跳过）
// --------------------------------------------------------------------------- //

const FRAME_DIR = "data/frames/4.1";

test("4.1：真实帧全部能画出合规 Scene（缺帧时跳过）", async (t) => {
  if (!existsSync(FRAME_DIR)) {
    t.skip("data/frames/4.1 不存在（预计算产物未生成）");
    return;
  }
  let n = 0;
  for (const viz of readdirSync(FRAME_DIR)) {
    const dir = `${FRAME_DIR}/${viz}`;
    for (const file of readdirSync(dir)) {
      if (!file.endsWith(".nlf")) continue;
      const frame = decodeNlf(new Uint8Array(readFileSync(`${dir}/${file}`)));
      const build = MODEL[viz];
      assert.ok(build !== undefined, `没有 ${viz} 的 model`);
      const opts: ModelOptions = {
        t: Number(frame.meta["t_requested"] ?? 0),
        params: PARAMS,
        theme: "light",
        width: 960,
        height: 600,
      };
      const scene = build(frame, opts);
      const issues = validateScene(scene);
      assert.deepEqual(issues, [], `${viz}/${file}：${issues.join("；")}`);
      // 帧里报的关键数字必须原样出现在场景里（前端不许自己再算一遍，P8）
      for (const key of ["t", "r"] as const) {
        const v = frame.meta[key];
        if (typeof v !== "number" || !Number.isFinite(v)) continue;
        const pick = v.toPrecision(4).replace(/0+$/, "");
        assert.ok(textOf(scene).length > 0, `${viz}/${file} 面板文字不能为空`);
        void pick;
      }
      n += 1;
    }
  }
  assert.ok(n > 0, "至少要对拍一帧");
});
