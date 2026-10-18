/**
 * 3.1 章（单步法、阶条件与自适应步长）model 层测试 —— **自足**：不依赖 Python，也不依赖
 * `data/frames` 是否存在。夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，字段集合
 * （数组名/dtype/shape/unit/label、meta/strings/critical 键）逐字照抄 `server/chapters/3.1.py`
 * 的真实帧；真实帧在的话再对拍一遍（见文件末尾，缺帧时整组跳过）。
 *
 * 本文件专门守 3.1 最容易画反的几条（对应 `chapters/3.1/pitfalls.md`）：
 * - **误差轴是固定双对数窗口**：全都贴地板（1e-17）时窗口也不许塌缩成一条线——否则
 *   "误差都在地板上"看起来和"误差在 1e-3"一模一样；
 * - **是哪一种误差**：整段最大 / 终点 / 首步真局部 / 稠密输出四条曲线必须同时在，
 *   `meta.ratio_max_over_endpoint`（终点比整段小 810–908 倍）必须被写出来；
 * - **拒绝的试步也是钱**：`accepted=0` 的试步单独标记，图上要点明"被拒绝的步也占一次求值"；
 * - **`-1` 哨兵不是数值**：`work_at_target_valid=0` 的位置绝不画成工作量（反查拒绝外推）；
 * - **残差横轴锁在 [0, tMax]**：`max_order` 调小时右边留空——不裁窗口，第 3 条破与第 5 条破
 *   才在图上分得开；
 * - **b_hat 从第一棵就不满足**：`meta.hat_fallback` 为真时必须显式写在图上，不许静默换权重；
 * - **没有时间轴的 viz 不许碰 t**（`work_precision` 的 `time.mode="none"`）；
 * - **降级不静默**：meta/strings/数组缺键 → 少画什么就写明什么（degraded 角标点名）。
 */

import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import { adaptiveStepModel } from "../chapters/3.1/adaptive_step.model.js";
import { manifest, modelFor, models } from "../chapters/3.1/index.js";
import { orderConditionsModel } from "../chapters/3.1/order_conditions.model.js";
import { orderConvergenceModel } from "../chapters/3.1/order_convergence.model.js";
import { workPrecisionModel } from "../chapters/3.1/work_precision.model.js";

const PITFALLS =
  "它会怎样骗你：把「阶条件满足」当成「误差小」——Σb=1+1e-3 时残差正好 −1e-3，可误差只差 1.19 倍；" +
  "把「请求容差」当成「实到精度」——关掉局部外推的 Heun 实际误差是请求值的 389 倍；" +
  "把「步数少」当成「便宜」——被拒绝的试步照样要花一次求值。";

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
    chapter: "3.1",
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
  return { ...def, arrays, meta: { ...meta, ...ov.meta }, strings: { ...strings, ...ov.strings }, pitfalls: ov.pitfalls ?? def.pitfalls };
}

// --------------------------------------------------------------------------- //
// 四个 viz 的夹具（字段集合照抄真实帧，数值取真实帧的头几个值）
// --------------------------------------------------------------------------- //

const lin = (n: number, a: number, b: number): Float64Array =>
  Float64Array.from({ length: n }, (_, i) => (n <= 1 ? a : a + ((b - a) * i) / (n - 1)));

/** 幂律阶梯：`err_i = c · h_i^p`（真实帧的误差阶梯就是这么走的）。 */
const ladder = (hs: ArrayLike<number>, c: number, p: number): Float64Array =>
  Float64Array.from(hs as ArrayLike<number>, (h) => c * Math.pow(h, p));

const OC_H = Float64Array.from([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]);
const OC_DEF: FrameDef = {
  viz: "order_convergence",
  arrays: [
    ["t_curve", new Float64Array([0]), "解曲线时间"],
    ["y_curve", new Float64Array([1]), "数值解"],
    ["y_exact_curve", new Float64Array([1]), "精确解"],
    ["error_curve", new Float64Array([1e-17]), "逐步误差（地板 1e-17）"],
    ["h_ladder", OC_H, "步长阶梯 h"],
    ["error_global", ladder(OC_H, 2.4e-5, 4), "整段最大误差"],
    ["error_endpoint", ladder(OC_H, 1.77e-5, 4), "终点误差"],
    ["error_local", ladder(OC_H, 1.54e-5, 5), "首步真局部误差"],
    ["error_dense", ladder(OC_H, 2.38e-5, 4), "稠密输出误差"],
    ["theory_global", ladder(OC_H, 4.93e-8 / Math.pow(0.0625, 4), 4), "声明阶的理论线（整段）"],
    ["theory_local", ladder(OC_H, 7.86e-9 / Math.pow(0.0625, 5), 5), "声明阶+1 的理论线（局部）"],
    ["mark_h", new Float64Array([0.0625]), "当前 h"],
    ["mark_error_global", new Float64Array([4.92811e-8]), "当前 h 的整段误差"],
    ["mark_error_endpoint", new Float64Array([3.6259e-8]), "当前 h 的终点误差"],
  ],
  meta: {
    declared_order: 4, dense_extra_per_step: 3, endpoint_error_so_far: 3.6259e-8, err_floor: 1e-17,
    fsal: true, h: 0.0625, is_pair: false, max_error_so_far: 0, measured_slope_dense: 4.0483,
    measured_slope_global: 4.0439, measured_slope_local: 4.983, n_steps_done: 0, n_steps_total: 32,
    order_hat: 3, ratio_max_over_endpoint: 810.3, slope_drop_head: 1, slope_drop_tail: 0,
    stages: 4, t: 0, t_end: 2, t_now: 0, work_per_time: 96,
  },
  strings: {
    at_h: "h=0.0625：整段最大误差 4.93e-08，终点 3.63e-08",
    data_source: "backend",
    method_label: "经典 RK4",
    nonfinite_policy: "误差为 0 或非有限时抬到地板 1e-17，并配 *_clipped 掩码。",
    params_json: '{"method":"rk4","problem":"exp_decay","h":0.0625}',
    problem_label: "指数衰减 y' = −y",
    slope_note: "拟合窗口由后端固定：丢弃头 1 点、尾 0 点；不裁窗口时预渐近点会把斜率抬高（rkf45 整窗 4.16）。",
    time_semantics: "真实时间 t：0 → 当前 t，解曲线按当前 h 步进到 t。",
  },
  critical: {},
};

const AS_N = 6;
const AS_T = Float64Array.from([0, 0.02, 0.05, 0.08, 0.2, 0.5]);
const AS_H = Float64Array.from([0.02, 0.03, 0.05, 0.03, 0.1, 0.25]);
const AS_EST = Float64Array.from([1.3e-6, 3.1e-6, 4.9e-5, 2.2e-7, 8.1e-7, 1.4e-6]);
const AS_TRUE = Float64Array.from([1.8e-14, 2.0e-12, 3.3e-9, 1.1e-14, 6.0e-13, 2.4e-12]);
/** 第 3 次试步被拒绝（ε=4.9e-5 / tol=1e-6 ≈ 49 > 1），其余接受。 */
const AS_ACCEPT = Float64Array.from([1, 1, 0, 1, 1, 1]);

const AS_DEF: FrameDef = {
  viz: "adaptive_step",
  arrays: [
    ["t_sol", Float64Array.from([0, 0.02, 0.05, 0.2, 0.5]), "已接受节点时间"],
    ["y_sol", Float64Array.from([1, 0.980199, 0.951229, 0.818731, 0.606531]), "数值解（只画到最后一个接受节点）"],
    ["y_exact_sol", Float64Array.from([1, 0.980199, 0.951229, 0.818731, 0.606531]), "精确解"],
    ["attempt_index", Float64Array.from({ length: AS_N }, (_, i) => i), "试步序号"],
    ["t_attempt", AS_T, "试步时刻"],
    ["h_attempt", AS_H, "试步步长 h"],
    ["error_estimate", AS_EST, "归一化误差估计 ε"],
    ["error_true", AS_TRUE, "真局部误差（同方法细分参考解）"],
    ["error_true_clipped", Float64Array.from([0, 0, 0, 0, 0, 0]), "真误差截断掩码"],
    ["accepted", AS_ACCEPT, "接受掩码（0=被拒绝）"],
    ["threshold", new Float64Array(AS_N).fill(1), "接受阈值（ε=1）"],
  ],
  meta: {
    accept_ratio: 5 / 6, attempt_index: 0, blowup: false, endpoint_error: 2.4e-12, err_floor: 1e-17,
    err_over_tol: 0.1806, estimator_ratio_median: 7.205e7, h_max: 0.25, h_median: 0.09,
    h_min: 0.02, h_over_stability: 0.1, h_stability_limit: 2.5, law_is_pi: true, local_extrapolation: true,
    max_eps: 49, max_error: 3.3e-9, n_accepted: 5, n_attempts: AS_N, n_f_evals: 61, n_rejected: 1,
    ok: true, order_estimate: 4, order_propagated: 5, progress_percent: 6, t: 0, t_blowup: -1, t_left: 0.5,
    tol: 1e-6, work_per_time: 275.5,
  },
  strings: {
    data_source: "backend",
    ext_note: "θ=0.2：h_max=0.25 小于稳定极限 2.5，本例步长由精度决定。",
    method: "dopri5",
    method_label: "Dormand–Prince 5(4)",
    nonfinite_policy: "真误差为 0 或非有限时抬到地板 1e-17，并配 error_true_clipped 掩码。",
    params_json: '{"method":"dopri5","problem":"exp_decay","tol":1e-6,"law":"pi","extrap":true}',
    problem: "exp_decay",
    problem_label: "指数衰减 y' = −y",
    stiff_note: "刚性算例上估计器不会说谎：估计/真误差的比值趋近稳定极限决定的常数。",
    time_semantics: "试步进度（0–100）：0 = 还没开始，100 = 整段积分走完；第 k 次试步见 meta.attempt_index。",
  },
  critical: {},
};

/** 8 棵 ≤4 阶的有根树（真实帧的坐标与密度）。 */
const OT_CX = Float64Array.from([1 / 6, 0.5, 5 / 6, 1 / 6, 0.5, 0.865, 0.5, 1 / 6]);
const OT_CY = Float64Array.from([0.706667, 0.833333, 0.791111, 0.5, 0.436667, 0.468333, 0.28, 0.2]);
const OT_NODE_TREE = Float64Array.from([0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7]);
const OT_EDGE_TREE = Float64Array.from([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7]);

const OT_DEF: FrameDef = {
  viz: "order_conditions",
  arrays: [
    ["node_x", lin(OT_NODE_TREE.length, 1 / 6, 5 / 6), "树节点 x（归一化单位框）"],
    ["node_y", lin(OT_NODE_TREE.length, 0.2, 0.96), "树节点 y"],
    ["node_tree", OT_NODE_TREE, "节点所属树"],
    ["edge_x0", lin(OT_EDGE_TREE.length, 0.5, 1 / 6), "边起点 x"],
    ["edge_y0", lin(OT_EDGE_TREE.length, 0.96, 0.3), "边起点 y"],
    ["edge_x1", lin(OT_EDGE_TREE.length, 0.4, 0.2), "边终点 x"],
    ["edge_y1", lin(OT_EDGE_TREE.length, 0.7, 0.2), "边终点 y"],
    ["edge_tree", OT_EDGE_TREE, "边所属树"],
    ["tree_cx", OT_CX, "树心 x"],
    ["tree_cy", OT_CY, "树心 y"],
    ["cond_index", Float64Array.from({ length: 8 }, (_, i) => i), "条件序号（= 树序号）"],
    ["cond_tree", Float64Array.from({ length: 8 }, (_, i) => i), "该条对应第几棵树"],
    ["cond_order", Float64Array.from([1, 2, 3, 3, 4, 4, 4, 4]), "树的阶 ρ"],
    ["cond_density", Float64Array.from([1, 2, 3, 6, 4, 8, 12, 24]), "密度 γ"],
    ["cond_phi", Float64Array.from([1, 0.5, 1 / 3, 1 / 6, 0.25, 0.125, 1 / 12, 1 / 24]), "初等权 Φ"],
    ["cond_residual", Float64Array.from([1e-17, 1e-17, 1e-17, 1e-17, 1.11e-16, 1e-17, 1e-17, 1e-17]), "残差 |1−γΦ|"],
    ["cond_pass", Float64Array.from([1, 1, 1, 1, 1, 1, 1, 1]), "残差判据（≤1e-10）"],
    ["cond_lit", Float64Array.from([1, 0, 0, 0, 0, 0, 0, 0]), "已扫到的条件"],
  ],
  meta: {
    b_hat_available: false, cond_index: 0, declared_order: 4, err_floor: 1e-17, hat_fallback: false,
    max_order: 4, measured_order: 4, n_conditions: 8, n_failed: 0, n_passed: 8, order_hat: 3, t: 0,
    tol_used: 1e-10, using_hat: false, worst_index: 4, worst_residual: 1.11e-16,
  },
  strings: {
    data_source: "backend",
    method_label: "经典 RK4",
    nonfinite_policy: "残差为 0 或非有限时抬到地板 1e-17；判据 tol_used 见 meta。",
    params_json: '{"method":"rk4","max_order":4,"use_hat":false}',
    time_semantics: "阶条件序号（按树枚举顺序）：k = 已点亮的条数，0-based。",
    tree_labels: "*|[*]|[*,*]|[[*]]|[*,*,*]|[*,[*]]|[[*,*]]|[[[*]]]",
    weight_note: "用的是 b（被推进的权）",
    worst_label: "[*,*,*]",
  },
  critical: {},
};

const WP_TOL = Float64Array.from([1e-4, 1e-5, 1e-6, 1e-7, 1e-8]);
const WP_DEF: FrameDef = {
  viz: "work_precision",
  arrays: [
    ["tol_grid", WP_TOL, "请求容差网格"],
    ["tol_line", WP_TOL, "对角线 y=tol 的横坐标"],
    ["error_heun", Float64Array.from([2.99298e-5, 3.01243e-6, 3.02057e-7, 3.02322e-8, 3.02406e-9]), "Heun 实到误差"],
    ["work_heun", Float64Array.from([84, 263, 828, 2613, 8252]), "Heun 工作量"],
    ["error_bs23", Float64Array.from([1.79464e-4, 1.68999e-5, 1.64324e-6, 1.62952e-7, 1.61528e-8]), "BS23 实到误差"],
    ["work_bs23", Float64Array.from([17, 30.5, 62, 129.5, 280]), "BS23 工作量"],
    ["error_rkf45", Float64Array.from([5.21116e-5, 9.66328e-6, 1.81006e-6, 2.98569e-7, 4.83009e-8]), "RKF45 实到误差"],
    ["work_rkf45", Float64Array.from([21, 24, 33, 45, 66]), "RKF45 工作量"],
    ["error_dopri5", Float64Array.from([1.11411e-5, 1.49648e-6, 1.80572e-7, 1.91842e-8, 2.00797e-9]), "DOPRI5 实到误差"],
    ["work_dopri5", Float64Array.from([18.5, 21.5, 30.5, 39.5, 57.5]), "DOPRI5 工作量"],
    ["error_fixed_rk4", Float64Array.from([2.91403e-4, 1.47582e-5, 8.30751e-7, 4.92811e-8, 3.00081e-9, 1.85123e-10, 1.15e-11]), "固定步长 RK4 误差阶梯"],
    ["work_fixed_rk4", Float64Array.from([26, 36, 56, 96, 176, 336, 656]), "固定步长 RK4 工作量"],
    ["mark_tol", new Float64Array([1e-6]), "当前 tol"],
    ["mark_error", Float64Array.from([3.02057e-7, 1.64324e-6, 1.81006e-6, 1.80572e-7]), "当前 tol 上的实到误差"],
    ["mark_work", Float64Array.from([828, 62, 33, 30.5]), "当前 tol 上的工作量"],
    ["work_at_target", Float64Array.from([339.056, 60.2019, 33, 21.196]), "同一精度反查的工作量"],
    ["work_at_target_valid", Float64Array.from([1, 1, 1, 1]), "反查有效掩码"],
  ],
  meta: {
    best_error: 3.02057e-7, cheapest_work: 30.5, dearest_work: 828, err_floor: 1e-17,
    error_spread_at_tol: 10.02, local_extrapolation: true, n_methods: 4, n_work_at_target_unavailable: 0,
    t: 0, target_error: 1.81e-6, tol: 1e-6, work_at_target_max: 339.056, work_at_target_min: 21.196,
    work_at_target_spread: 15.996, work_spread_at_tol: 27.15, worst_error: 1.81006e-6,
  },
  strings: {
    at_target: "同一精度（1.81e-06）反查：heun 339.1、bs23 60.2、rkf45 33.0、dopri5 21.2",
    at_tol: "tol=1e-06：heun 3.02e-07、bs23 1.64e-06、rkf45 1.81e-06、dopri5 1.81e-07",
    data_source: "backend",
    method_order: "heun|bs23|rkf45|dopri5",
    nonfinite_policy: "工作量或误差为 0/非有限时不画在对数轴上；反查不可用用 -1 + 掩码。",
    params_json: '{"problem":"exp_decay","tol":1e-6,"extrap":true}',
    problem: "exp_decay",
    reading_note: "左图看「说到做不到」：曲线在 y=tol 上方 = 实到误差大于请求容差。右图看代价：x 越小、y 越低越好，四条曲线的下包络才是前沿。",
    target_note: "目标精度 = 当前 tol 下最差方法的实到误差，四个方法都达到它。",
    time_semantics: "本 viz 没有时间轴（time.mode=none）：四个方法在整段区间上各扫一遍容差网格。",
  },
  critical: {},
};

// --------------------------------------------------------------------------- //
// 公共工具
// --------------------------------------------------------------------------- //

const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };
const DEFS: Record<string, FrameDef> = {
  order_convergence: OC_DEF,
  adaptive_step: AS_DEF,
  order_conditions: OT_DEF,
  work_precision: WP_DEF,
};
const MODELS: Record<string, typeof orderConvergenceModel> = {
  order_convergence: orderConvergenceModel,
  adaptive_step: adaptiveStepModel,
  order_conditions: orderConditionsModel,
  work_precision: workPrecisionModel,
};
/** manifest 的 `requiredAux` 名 →（画在哪条轴，取帧里哪个数组）。 */
const AUX_WHERE: Record<string, [string, string]> = {
  order_convergence: ["error", "error_global"],
  adaptive_step: ["errseq", "error_estimate"],
  order_conditions: ["residual", "cond_residual"],
  work_precision: ["vs_tol", "error_dopri5"],
};

const sceneOf = async (viz: string, over: Override = {}, opts: Partial<ModelOptions> = {}): Promise<Scene> =>
  MODELS[viz](await makeFrame(tweak(DEFS[viz] as FrameDef, over)), { ...OPTS, ...opts });
const specOf = (viz: string) => manifest.vizzes.find((v) => v.id === viz) ?? assert.fail(`manifest 缺 ${viz}`);
const textsOf = (s: Scene): string => s.texts.map((t) => t.text).join("\n");
const badgesOf = (s: Scene): string => s.badges.map((b) => `${b.kind}:${b.text}`).join("\n");
const badgeKinds = (s: Scene): string[] => s.badges.map((b) => b.kind);
const axisOf = (s: Scene, id: string): Axis => s.axes.find((a) => a.id === id) ?? assert.fail(`缺少轴 ${id}`);
const layersOn = (s: Scene, id: string): Layer[] => s.layers.filter((l) => l.axes === id);
/** 某个轴上所有被画出来的数值（line/scatter/bars 通吃）。 */
const drawnOn = (s: Scene, id: string): number[] => {
  const out: number[] = [];
  for (const l of layersOn(s, id)) {
    if (l.kind === "line") for (const ser of l.series) out.push(...Array.from(ser.y, Number));
    else if (l.kind === "scatter") out.push(...Array.from(l.y, Number));
    else if (l.kind === "bars") out.push(...Array.from(l.values, Number));
    else if (l.kind === "heat" || l.kind === "contour") out.push(...Array.from(l.values, Number));
  }
  return out;
};
/** 某个轴上画出来的所有折线点（含 x）。 */
const seriesOn = (s: Scene, id: string): { x: number[]; y: number[] }[] => {
  const out: { x: number[]; y: number[] }[] = [];
  for (const l of layersOn(s, id)) {
    if (l.kind === "line") for (const ser of l.series) out.push({ x: Array.from(ser.x, Number), y: Array.from(ser.y, Number) });
  }
  return out;
};
const scatterOn = (s: Scene, id: string): { x: number[]; y: number[] }[] => {
  const out: { x: number[]; y: number[] }[] = [];
  for (const l of layersOn(s, id)) {
    if (l.kind === "scatter") out.push({ x: Array.from(l.x, Number), y: Array.from(l.y, Number) });
  }
  return out;
};
const numsOf = (frame: Frame, name: string): number[] => Array.from(frame.array(name) as ArrayLike<number>, Number);
const close = (a: number, b: number, tol = 1e-9): boolean => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// --------------------------------------------------------------------------- //
// 装配层与公共契约
// --------------------------------------------------------------------------- //

test("3.1：manifest 的四个 viz 都有 model，映射表没有多余键，未知 id 返回 null", () => {
  assert.deepEqual(Object.keys(models).sort(), manifest.vizzes.map((v) => v.id).sort());
  assert.equal(manifest.chapter, "3.1");
  for (const v of manifest.vizzes) assert.equal(modelFor(v.id), models[v.id]);
  assert.equal(modelFor("no_such_viz"), null);
});

test("3.1：四个 viz 都能产出合法 Scene（validateScene 零 issue），轴按 manifest 声明", async () => {
  for (const viz of manifest.vizzes.map((v) => v.id)) {
    const scene = await sceneOf(viz);
    assert.deepEqual(validateScene(scene), [], `${viz} 的 Scene 有校验问题`);
    assert.equal(scene.chapter, "3.1");
    assert.equal(scene.viz, viz);
    assert.ok(scene.pitfalls.length >= 10);
    for (const spec of specOf(viz).axes) {
      const axis = axisOf(scene, spec.id);
      assert.equal(axis.autoscaled, spec.autoscaled, `${viz}/${spec.id} 的 autoscaled 与 manifest 不一致`);
      if (spec.scale === "log") assert.equal(axis.yscale, "log", `${viz}/${spec.id} 应为 log 轴`);
      assert.ok(scene.layers.some((l) => l.axes === spec.id), `${viz}/${spec.id} 上一个图层都没有`);
    }
  }
});

test("3.1：requiredAux 声明的数组真的画在了对应轴上（按数值认，不按 label）", async () => {
  for (const spec of manifest.vizzes) {
    for (const name of spec.requiredAux) {
      const where = AUX_WHERE[spec.id];
      assert.ok(where, `测试缺 ${spec.id} 的 requiredAux 映射`);
      const scene = await sceneOf(spec.id);
      const frame = await makeFrame(DEFS[spec.id] as FrameDef);
      const values = numsOf(frame, where[1]);
      const drawn = drawnOn(scene, where[0]);
      const hit = values.filter((v) => Number.isFinite(v) && drawn.some((d) => close(d, v, 1e-9))).length;
      assert.ok(hit >= 2, `${spec.id} 的 requiredAux ${name}（${where[1]}）没有画在 ${where[0]} 上（命中 ${hit}）`);
    }
  }
});

test("3.1：角标齐全——source/自适应说明必有，溢出与降级按需", async () => {
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz);
    assert.ok(badgeKinds(scene).includes("source"), `${viz} 没有 source 角标`);
    assert.ok(badgeKinds(scene).includes("autoscale"), `${viz} 没有坐标轴模式角标（规则 4）`);
    assert.ok(scene.overflow !== undefined, `${viz} 缺 overflow 字段`);
  }
});

// --------------------------------------------------------------------------- //
// order_convergence：四条误差曲线 + 固定窗口 + 地板
// --------------------------------------------------------------------------- //

test("order_convergence：误差轴是**固定**双对数窗口——误差全贴地板时窗口也不许塌缩", async () => {
  const normal = await sceneOf("order_convergence");
  const flat = await sceneOf("order_convergence", {
    arrays: {
      error_global: new Float64Array(8).fill(1e-17),
      error_endpoint: new Float64Array(8).fill(1e-17),
      error_local: new Float64Array(8).fill(1e-17),
      error_dense: new Float64Array(8).fill(1e-17),
    },
  });
  const a = axisOf(normal, "error");
  const b = axisOf(flat, "error");
  assert.equal(a.autoscaled, false);
  assert.equal(b.autoscaled, false);
  assert.deepEqual(b.ylim, a.ylim, "误差轴随数据自适应了：全都贴地板时窗口会塌成一条线");
  assert.ok(b.ylim[1] / b.ylim[0] >= 1e6, `误差窗口太窄（${b.ylim}）：地板与 1e-3 量级必须同屏`);
});

test("order_convergence：四条误差曲线都在，且文字说清「是哪一种误差」+ 终点/整段之比", async () => {
  const scene = await sceneOf("order_convergence");
  const drawn = drawnOn(scene, "error");
  const frame = await makeFrame(OC_DEF);
  for (const name of ["error_global", "error_endpoint", "error_local", "error_dense"]) {
    const values = numsOf(frame, name).filter((v) => Number.isFinite(v) && v > 0);
    const hit = values.filter((v) => drawn.some((d) => close(d, v, 1e-6))).length;
    assert.ok(hit >= 3, `${name} 没有画在误差轴上（命中 ${hit}）——四条误差曲线必须同时在`);
  }
  const text = textsOf(scene);
  assert.match(text, /整段最大/, "没有说明「整段最大」这一条");
  assert.match(text, /终点/, "没有说明「终点」这一条");
  assert.match(text, /首步真局部|局部/, "没有说明「首步真局部」这一条");
  assert.match(text, /稠密输出|插值/, "没有说明「稠密输出」这一条");
});

test("order_convergence：地板线画在 err_floor，并写明它不是「很小」", async () => {
  const scene = await sceneOf("order_convergence");
  const floor = Number((DEFS.order_convergence as FrameDef).meta["err_floor"]);
  const hasFloor = seriesOn(scene, "error").some((s) => s.y.length >= 2 && s.y.every((v) => close(v, floor, 1e-12)));
  assert.ok(hasFloor, `误差轴上没有 ${floor} 的水平参考线`);
  assert.match(textsOf(scene) + badgesOf(scene), /地板/, "没有文字说明地板是什么");
  assert.match(textsOf(scene) + badgesOf(scene), /不是/, "没有点明「地板不是数值小」：\n" + textsOf(scene));
});

test("order_convergence：理论参考线用**声明阶**，并交代拟合窗口丢弃点数", async () => {
  const scene = await sceneOf("order_convergence");
  const fit = scene.layers.find((l) => l.kind === "loglogfit");
  assert.ok(fit, "误差轴上没有 loglogfit（实测斜率 vs 理论斜率）");
  if (fit.kind === "loglogfit") assert.equal(fit.theorySlope, 4, "理论线没有用声明的 p=4");
  assert.match(textsOf(scene), /丢弃|裁窗口/, "没有交代拟合窗口丢了几个点（不裁窗口 rkf45 会量到 4.16）");
});

test("order_convergence：缺 mark_* / 理论线数组时降级点名，不崩", async () => {
  const scene = await sceneOf("order_convergence", { drop: ["mark_h", "mark_error_global", "theory_local"] });
  assert.deepEqual(validateScene(scene), []);
  assert.ok(badgeKinds(scene).includes("status"), "降级时没有状态/降级角标");
});

// --------------------------------------------------------------------------- //
// adaptive_step：拒绝的步也是钱 + ε 阈值 + 时间语义
// --------------------------------------------------------------------------- //

test("adaptive_step：被拒绝的试步单独标记，且写明「拒绝也要花一次求值」", async () => {
  const scene = await sceneOf("adaptive_step");
  const frame = await makeFrame(AS_DEF);
  const rejected = numsOf(frame, "t_attempt").filter((_, i) => (AS_ACCEPT[i] as number) < 0.5);
  assert.equal(rejected.length, 1, "夹具应有一个被拒绝的试步");
  // 拒绝点必须出现在图上（散点或折线），且不是解曲线的一部分。
  const drawn = [...drawnOn(scene, "hseq"), ...drawnOn(scene, "errseq"), ...seriesOn(scene, "solution").flatMap((s) => s.x)];
  assert.ok(drawn.some((v) => close(v, rejected[0] as number, 1e-9)), "被拒绝的试步时刻没有出现在图上");
  const text = textsOf(scene) + badgesOf(scene);
  assert.match(text, /拒绝/, "没有提到拒绝的试步");
  assert.match(text, /求值|代价|评估/, "没有说明被拒绝的试步也占一次求值");
});

test("adaptive_step：ε 是归一化的（阈值 1），真误差与估计同屏，并交代真误差自己也有误差", async () => {
  const scene = await sceneOf("adaptive_step");
  const text = textsOf(scene);
  assert.match(text, /归一化/, "没有说明 ε 是归一化误差");
  assert.match(text, /tol/, "没有给出 tol 的数值");
  assert.match(text, /真局部误差|真误差/, "没有说明真误差那条线");
  const frame = await makeFrame(AS_DEF);
  const est = numsOf(frame, "error_estimate");
  const errs = drawnOn(scene, "errseq");
  assert.ok(est.filter((v) => errs.some((d) => close(d, v, 1e-6))).length >= 2, "估计 ε 没有画在 errseq 轴上");
});

test("adaptive_step：h 轴与 ε 轴都是固定对数窗口（不随数据自适应）", async () => {
  const tight = await sceneOf("adaptive_step", { arrays: { h_attempt: Float64Array.from([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) } });
  const loose = await sceneOf("adaptive_step");
  assert.equal(axisOf(tight, "hseq").autoscaled, false);
  assert.deepEqual(axisOf(tight, "hseq").ylim, axisOf(loose, "hseq").ylim, "h 轴随数据变了");
  assert.deepEqual(axisOf(tight, "errseq").ylim, axisOf(loose, "errseq").ylim, "ε 轴随数据变了");
});

test("adaptive_step：真实帧的 t_left 是「当前积分时刻」而不是剩余时间——文字里必须这么说", async () => {
  const scene = await sceneOf("adaptive_step");
  const text = textsOf(scene);
  // 后端把 meta.t_left 命名错了；模型不许跟着错，必须点出来。
  assert.match(text, /t_left|积分时刻/, "没有提到 t_left 的真实含义");
  assert.match(text, /不是|misnamed|命名/, "没有点明 t_left 不是「剩余时间」");
});

test("adaptive_step：滑块语义是**试步进度**（0–100），不是真实时间", async () => {
  const spec = specOf("adaptive_step");
  assert.equal(spec.time.mode, "scrub");
  assert.equal(spec.time.tMax, 100, "进度轴的上沿必须是 manifest 的 tMax=100");
  // `scene.t` 取帧自己的进度坐标（meta.t）——它**不是**积分时刻：同一帧里 meta.t_now 才是。
  const scene = await sceneOf("adaptive_step", { meta: { t: 50 } }, { t: 3 });
  assert.equal(scene.t, 50, "scene.t 应等于帧的进度坐标 meta.t");
  const text = textsOf(scene);
  assert.match(text, /试步进度|进度/, "没有说明滑块是进度");
  assert.match(text, /meta\.t_now|积分时刻/, "没有把进度与真实积分时刻分开写");
});

// --------------------------------------------------------------------------- //
// order_conditions：固定序号窗口 + b_hat 回退 + 地板
// --------------------------------------------------------------------------- //

test("order_conditions：残差横轴锁在 [0, tMax]，max_order 变小只是右边留空（不裁窗口）", async () => {
  const full = await sceneOf("order_conditions");
  const small = await sceneOf("order_conditions", {
    arrays: {
      cond_index: Float64Array.from([0, 1]),
      cond_tree: Float64Array.from([0, 1]),
      cond_order: Float64Array.from([1, 2]),
      cond_density: Float64Array.from([1, 2]),
      cond_phi: Float64Array.from([1, 0.5]),
      cond_residual: Float64Array.from([1e-17, 1e-17]),
      cond_pass: Float64Array.from([1, 1]),
      cond_lit: Float64Array.from([1, 0]),
    },
    meta: { max_order: 2, n_conditions: 2, n_passed: 2, n_failed: 0, measured_order: 2, cond_index: 0 },
  });
  assert.deepEqual(axisOf(small, "residual").xlim, axisOf(full, "residual").xlim, "残差横轴随 max_order 裁了窗口");
  assert.deepEqual(axisOf(full, "residual").xlim, [0, 17], "残差横轴应是 [0, 17]（5 阶共 17 棵树）");
});

test("order_conditions：b_hat 回退（hat_fallback）必须显式写在图上，不许静默换权重", async () => {
  const scene = await sceneOf("order_conditions", {
    meta: { hat_fallback: true, using_hat: false, b_hat_available: false, n_failed: 2, worst_residual: 0.5, worst_index: 5 },
    strings: { weight_note: "用的是 b（use_hat 已回退）" },
  });
  const text = textsOf(scene) + badgesOf(scene);
  assert.match(text, /回退/, "勾了 b_hat 却没有 b_hat 时必须显式说明回退");
  assert.deepEqual(validateScene(scene), []);
});

test("order_conditions：不满足的条件按红色单独画，并说明「残差大 ≠ 更靠后」", async () => {
  const scene = await sceneOf("order_conditions", {
    arrays: {
      cond_residual: Float64Array.from([1e-17, 1e-17, 1e-17, 0.5, 1e-17, 0.9, 1e-17, 1e-17]),
      cond_pass: Float64Array.from([1, 1, 1, 0, 1, 0, 1, 1]),
      cond_lit: new Float64Array(8).fill(1),
    },
    meta: { n_failed: 2, n_passed: 6, worst_index: 5, worst_residual: 0.9 },
    strings: { worst_label: "[*,[*]]" },
  });
  assert.deepEqual(validateScene(scene), []);
  const drawn = drawnOn(scene, "residual");
  assert.ok(drawn.some((v) => close(v, 0.5, 1e-9)) && drawn.some((v) => close(v, 0.9, 1e-9)), "不满足的残差没有画出来");
  assert.match(textsOf(scene), /更靠后|枚举顺序|不是重要性/, "没有点明树间没有强弱次序");
  assert.ok(badgeKinds(scene).includes("overflow"), "有条件不满足时应给出状态/溢出角标");
});

test("order_conditions：残差地板线与树标签来自帧（不按网格反推）", async () => {
  const scene = await sceneOf("order_conditions");
  const floor = Number((DEFS.order_conditions as FrameDef).meta["err_floor"]);
  assert.ok(
    seriesOn(scene, "residual").some((s) => s.y.length >= 2 && s.y.every((v) => close(v, floor, 1e-12))),
    "残差轴上没有地板参考线",
  );
  const labels = scene.texts.filter((t) => t.role === "axis").map((t) => t.text);
  assert.equal(labels.length, 8, `树标签个数与树数不一致（${labels.length}）`);
  assert.ok(labels.some((l) => l.includes("(4,24,0.042)") || l.includes("(4,24,0.04")), `树标签没有 ρ/γ/Φ：${labels.slice(0, 3)}`);
  assert.match(textsOf(scene), /tree_labels|结构串|\*/, "没有给出后端给的结构串");
});

test("order_conditions：缺 tree_labels/edge_* 时降级点名（不按网格反推树形）", async () => {
  const scene = await sceneOf("order_conditions", {
    drop: ["edge_x0", "edge_y0", "edge_x1", "edge_y1", "edge_tree"],
    dropStrings: ["tree_labels"],
  });
  assert.deepEqual(validateScene(scene), []);
  assert.match(badgesOf(scene), /降级|缺/, "缺数组/字符串时没有降级角标");
});

// --------------------------------------------------------------------------- //
// work_precision：没有时间轴 + -1 哨兵 + 反查不外推
// --------------------------------------------------------------------------- //

test("work_precision：没有时间轴（time.mode=none）——scene.t 只来自帧 meta", async () => {
  assert.equal(specOf("work_precision").time.mode, "none");
  const scene = await sceneOf("work_precision", { meta: { t: 7 } }, { t: 3 });
  assert.equal(scene.t, 7, "无时间轴的 viz 不许把 opts.t 当时间");
  assert.match(textsOf(scene) + badgesOf(scene), /没有时间轴|无时间轴|time\.mode/, "没有声明「本 viz 没有时间轴」");
});

test("work_precision：-1 哨兵与 valid=0 不许当数值画（反查拒绝外推）", async () => {
  const scene = await sceneOf("work_precision", {
    arrays: {
      work_at_target: Float64Array.from([828, 23.67, -1, -1]),
      work_at_target_valid: Float64Array.from([1, 1, 0, 0]),
    },
    meta: { n_work_at_target_unavailable: 2, work_at_target_spread: 0, target_error: 1.81e-6 },
  });
  assert.deepEqual(validateScene(scene), []);
  for (const s of scatterOn(scene, "frontier")) {
    for (const v of s.x) assert.notEqual(v, -1, "-1 哨兵被当成工作量画出来了");
  }
  // 不可用的点只能贴在轴左端（明确标记），不能被伪装成某个真实工作量。
  const drawn = drawnOn(scene, "frontier");
  assert.ok(!drawn.includes(-1));
  const text = textsOf(scene) + badgesOf(scene);
  assert.match(text, /反查|不外推|超出|不可用/, "没有说明反查拒绝外推");
  assert.match(text, /2/, "没有给出拒绝作答的方法个数");
});

test("work_precision：y=tol 对角线、四条自适应曲线、固定步长 RK4 阶梯都在，且说明下包络", async () => {
  const scene = await sceneOf("work_precision");
  const frame = await makeFrame(WP_DEF);
  const tol = numsOf(frame, "tol_grid");
  // 对角线：起点与终点都在 y=tol 上。
  const diag = seriesOn(scene, "vs_tol").some((s) => s.x.length >= 2 && close(s.x[0] as number, s.y[0] as number, 1e-9));
  assert.ok(diag, "没有 y=tol 的对角线参考线");
  const drawn = drawnOn(scene, "vs_tol");
  for (const m of ["heun", "bs23", "rkf45", "dopri5", "fixed_rk4"]) {
    const key = m === "fixed_rk4" ? "error_fixed_rk4" : `error_${m}`;
    const values = numsOf(frame, key).filter((v) => v > 0);
    const hit = values.filter((v) => drawn.some((d) => close(d, v, 1e-6))).length;
    assert.ok(hit >= 3, `${key} 没有画在 vs_tol 轴上（命中 ${hit}）`);
  }
  const text = textsOf(scene);
  assert.match(text, /下包络|前沿/, "没有说明「下包络才是前沿」");
  assert.match(text, /每单位时间|求值次数/, "没有声明工作量的单位");
  assert.ok(tol.length === 5);
});

test("work_precision：两个轴都是固定对数窗口（没有时间轴，数据一次扫描给出）", async () => {
  const scene = await sceneOf("work_precision");
  assert.equal(axisOf(scene, "vs_tol").autoscaled, false);
  assert.equal(axisOf(scene, "frontier").autoscaled, false);
  assert.equal(axisOf(scene, "vs_tol").xscale, "log");
  assert.equal(axisOf(scene, "frontier").xscale, "log");
  assert.equal(axisOf(scene, "vs_tol").yscale, "log");
});

// --------------------------------------------------------------------------- //
// 降级、结构性错误、纯函数
// --------------------------------------------------------------------------- //

test("3.1：降级不静默——缺 meta/strings 键时点名，且不崩", async () => {
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz, {
      dropMeta: ["err_floor", "t", "declared_order", "max_order"],
      dropStrings: ["time_semantics", "params_json", "method_label"],
    });
    assert.deepEqual(validateScene(scene), [], `${viz} 降级后 Scene 不合法`);
    assert.ok(badgeKinds(scene).includes("status"), `${viz} 降级后没有状态/降级角标`);
  }
});

test("3.1：结构性错误直接抛 ModelInputError（不画半张图）", async () => {
  for (const [viz, arr] of [
    ["order_convergence", "h_ladder"],
    ["adaptive_step", "h_attempt"],
    ["order_conditions", "cond_residual"],
    ["work_precision", "tol_grid"],
  ] as const) {
    await assert.rejects(
      async () => MODELS[viz](await makeFrame(tweak(DEFS[viz] as FrameDef, { drop: [arr] })), OPTS),
      (err: unknown) => (err instanceof ModelInputError ? true : (() => { throw err; })()),
      `${viz} 缺 ${arr} 时应抛 ModelInputError`,
    );
  }
});

test("3.1：模型是纯函数——同一帧两次调用逐字节相同，且不改帧", async () => {
  for (const viz of Object.keys(MODELS)) {
    const frame = await makeFrame(DEFS[viz] as FrameDef);
    const before = JSON.stringify({ meta: frame.meta, strings: frame.strings });
    const a = MODELS[viz](frame, OPTS);
    const b = MODELS[viz](frame, OPTS);
    assert.equal(JSON.stringify(a), JSON.stringify(b), `${viz} 两次调用结果不同`);
    assert.equal(JSON.stringify({ meta: frame.meta, strings: frame.strings }), before, `${viz} 修改了帧`);
  }
});

test("3.1：窄/宽画布都合法（760/860/960/1200 宽；文本不许压到装饰带或互相压住）", async () => {
  for (const viz of Object.keys(MODELS)) {
    for (const width of [760, 860, 960, 1200]) {
      const scene = await sceneOf(viz, {}, { width, height: 600 });
      assert.deepEqual(validateScene(scene), [], `${viz} @${width}×600`);
    }
  }
});

// --------------------------------------------------------------------------- //
// 真实帧对拍（缺 data/frames 时整组跳过）
// --------------------------------------------------------------------------- //

const FRAME_ROOT = "data/frames/3.1";
const real = existsSync(FRAME_ROOT) ? test : test.skip;

function realFrames(viz: string): string[] {
  const dir = `${FRAME_ROOT}/${viz}`;
  return existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith(".nlf")).sort() : [];
}

function realFrame(viz: string, file: string): Frame {
  const buf = readFileSync(`${FRAME_ROOT}/${viz}/${file}`);
  return decodeNlf(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength));
}

real("真实帧：夹具与 build_frame 产物的数组名集合一致（字段不漂）", () => {
  for (const viz of Object.keys(DEFS)) {
    const files = realFrames(viz);
    if (files.length === 0) continue;
    const frame = realFrame(viz, files[0] as string);
    const realNames = new Set(frame.specs.map((s) => s.name));
    const fixtureNames = new Set((DEFS[viz] as FrameDef).arrays.map(([n]) => n));
    assert.deepEqual([...fixtureNames].sort(), [...realNames].sort(), `${viz} 的数组名集合与真实帧不一致`);
  }
});

real("真实帧：meta/strings/critical 的键都被夹具覆盖（多了没关系，少了会漏测）", () => {
  for (const viz of Object.keys(DEFS)) {
    const files = realFrames(viz);
    if (files.length === 0) continue;
    const frame = realFrame(viz, files[0] as string);
    for (const key of Object.keys(frame.meta)) {
      if (key === "t") continue;
      assert.ok(key in (DEFS[viz] as FrameDef).meta, `${viz} 的夹具缺 meta.${key}`);
    }
    for (const key of Object.keys(frame.strings)) {
      if (key === "params_json" || key === "data_source") continue;
      assert.ok(key in (DEFS[viz] as FrameDef).strings, `${viz} 的夹具缺 strings.${key}`);
    }
    for (const key of Object.keys(frame.critical)) {
      assert.ok(key in (DEFS[viz] as FrameDef).critical, `${viz} 的夹具缺 critical.${key}`);
    }
  }
});

real("真实帧：3.1 的每一帧都能画出合法 Scene（含 h 阶梯两端与 order_conditions 的 17 条）", () => {
  let n = 0;
  for (const viz of Object.keys(MODELS)) {
    for (const file of realFrames(viz)) {
      const frame = realFrame(viz, file);
      const t = Number(frame.meta["t"] ?? 0);
      const scene = MODELS[viz](frame, { width: 960, height: 600, t, params: {} });
      assert.deepEqual(validateScene(scene), [], `${viz}/${file}`);
      n += 1;
    }
  }
  assert.ok(n > 0, "一帧都没跑到：data/frames/3.1 是空的？");
});

real("真实帧：自适应/阶条件两帧的关键声明都在图上（时间语义、地板、权重）", () => {
  for (const viz of ["adaptive_step", "order_conditions"]) {
    for (const file of realFrames(viz)) {
      const frame = realFrame(viz, file);
      const scene = MODELS[viz](frame, { width: 960, height: 600, t: Number(frame.meta["t"] ?? 0), params: {} });
      const text = textsOf(scene) + badgesOf(scene);
      assert.match(text, /地板/, `${viz}/${file} 没有提到误差地板`);
      assert.match(text, /tol|容差|阈值|判据/, `${viz}/${file} 没有提到容差`);
    }
  }
});
