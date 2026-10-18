/**
 * 4.5 章（多重网格与快速求解）model 层测试 —— **自足**：不依赖 Python，也不依赖
 * `data/frames` 是否存在。夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，
 * 字段集合（数组名/dtype/shape/unit/label、meta/strings/critical 键）逐字照抄
 * `server/chapters/4.5.py` 的真实帧。
 *
 * 本文件守的是这一章**最容易画反、也最容易读反**的几件事：
 *
 * - **「网格无关」不是「等于常数」**：`factors` 轴窗口固定 `0–1.05` 且 `autoscaled=false`
 *   ——把窗口交给自适应，4 个因子的差别（0.161 vs 0.190）会被拉平，读者就看不见漂移；
 * - **跨层范数不可连线**：`events` 面板上每一条折线都必须**只属于一层**（`event_level`
 *   分段）。连起来画，读者会得出"粗层误差更小"的结论（第 3 个陷阱）；
 * - **纯模态的不可见比例是常数**：`residual_transfer` 的 `loss` 面板必须画出
 *   `meta.loss_at_current_theta` 的水平线，否则"残差变光滑了"会被读成"高频没了"；
 * - **别名频率有负号**：`meta.alias_theta_pi` 只进文本、不进比值轴（量纲不同），
 *   缺了它必须**点名降级**——它是"高频被认成低频"的唯一直接证据；
 * - **`time.mode="none"` 的 viz 不许碰 t**：`grid_independence` 的 `scene.t` 恒为 0；
 * - **临界值参考线（规则 6）**：`ω=2/3` 与 `θ/π=0.5` 两条竖线必须真画出来，
 *   值取自 `frame.critical`（帧为准），并带 `criticalBadge`；
 * - **降级不静默**：meta 缺键 / 掩码数组缺项 → 少画什么就写明什么（degraded 角标点名）。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import { gridIndependenceModel } from "../chapters/4.5/grid_independence.model.js";
import { manifest, modelFor, models } from "../chapters/4.5/index.js";
import { residualTransferModel } from "../chapters/4.5/residual_transfer.model.js";
import { smoothingSpectrumModel } from "../chapters/4.5/smoothing_spectrum.model.js";
import { vcycleHierarchyModel } from "../chapters/4.5/vcycle_hierarchy.model.js";

/** 一行数组 = `[名字, 数据, 标签, 形状]`；shape 缺省为 `[len]`（本章所有数组 unit 都是 `"1"`）。 */
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
  return decodeNlf(
    await encodeNlf(
      new Frame({
        chapter: "4.5",
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

const PITFALLS =
  "它会怎样骗你：第一，「网格无关」不等于常数：因子随层数缓慢变大，把「无关」说成「完全一样」是夸大。" +
  "第二，W-cycle 的因子确实更小，但每个周期的代价贵得多，按总代价算并不更省。" +
  "第三，粗层的范数天然比细层小几个数量级，跨层比大小会得出「粗层误差更小」的假结论。";

const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };

// --------------------------------------------------------------------------- //
// 四个 viz 的夹具（字段集合照抄真实帧）
// --------------------------------------------------------------------------- //

const N = 63;
const lin = (n: number, a: number, b: number): Float64Array =>
  Float64Array.from({ length: n }, (_, i) => (n <= 1 ? a : a + ((b - a) * i) / (n - 1)));
const theta63 = Float64Array.from({ length: N }, (_, i) => (i + 0.5) / N);
/** 平滑因子：低频 ≈1，越过 θ/π=0.5 后降到 1/3（与 jacobi + ω=2/3 的定性一致）。 */
const smoothFactor = (t: number): Float64Array =>
  Float64Array.from(theta63, (th) => (th < 0.5 ? 1 - 0.02 * t * th : 1 / 3 + 1e-16));
/** 振幅：低频几乎不动、高频每步乘平滑因子。 */
const amplitude = (t: number): Float64Array =>
  Float64Array.from(theta63, (th, i) => Math.pow(th < 0.5 ? 0.9992 : 0.3326, t) * (1 - 0.5 * (i / N)));

const SS_DEF: FrameDef = {
  viz: "smoothing_spectrum",
  arrays: [
    ["mode_index", lin(N, 1, N), "模态序号 k"],
    ["theta_over_pi", theta63, "θ/π（模态 k 的归一化频率）"],
    ["amplitude_initial", Float64Array.from(theta63, (_, i) => 1 - 0.5 * (i / N)), "初值 |c_k(0)|"],
    ["amplitude_measured", amplitude(30), "实测 |c_k(T)|"],
    ["amplitude_theory", amplitude(30), "理论 |c_k(T)|"],
    ["amplitude_at_floor", Float64Array.from(theta63, (th) => (th > 0.5 ? 1e-16 : 0)), "已到地板的振幅（占位）"],
    ["factor_measured", smoothFactor(1), "实测一步放大率 |λ_k(T)|^(1/T)"],
    ["factor_theory", smoothFactor(1), "理论一步放大率"],
    ["factor_at_floor", Float64Array.from(theta63, (th) => (th > 0.5 ? 1e-16 : 0)), "地板上的因子（占位）"],
    ["mode_present", Float64Array.from(theta63, () => 1), "该模态是否出现在本次扫描里"],
    ["smoothing_line", Float64Array.from(theta63, () => 1 / 3), "1/3 上限参考线"],
    ["half_band_line", Float64Array.from(theta63, () => 0.5), "θ/π=0.5 分界参考线"],
    ["error_norm_history", Float64Array.from({ length: 31 }, (_, k) => 0.8862 + k / 300), "整体误差范数 ‖e_k‖/‖e_0‖"],
    ["error_norm_history_slowest", Float64Array.from({ length: 31 }, (_, k) => 1 - k / 5000), "最慢模态的预测范数"],
  ],
  meta: {
    content_index: 0, error_norm_ratio: 0.8862, floor: 1e-16, high_mode_factor: 0.3326,
    high_mode_survival: 3.316e-14, low_mode_factor: 0.9992, low_mode_survival: 0.97619,
    low_over_high_survival: 2.944e13, measured_smoothing_factor: 0.3333, n: N, n_modes_at_floor: 8,
    n_modes_present: N, omega: 0.666667, slowest_mode_prediction: 0.9992, smooth_factor_gap: 1.67e-16,
    smoother_index: 0, steps: 30, t_requested: 30, theory_smoothing_factor: 0.3333, t: 30,
  },
  strings: {
    curve_note: "两条曲线在 jacobi 下逐位相同；SOR 下会分开，分开的幅度就是非正规性的可信边界。",
    data_source: "backend", floor: "1e-16", nonfinite_policy: "非有限值不画。",
    params_json: '{"n":63,"omega":0.6667,"smoother":"jacobi","content":"mixed"}',
    placeholder_note: "地板上的因子是 1.0 占位符，不是「没衰减」。",
    smoother_note: "jacobi：实测与理论逐位相同。", status: "ok",
    time_semantics: "k 是迭代步数，不是物理时间。",
  },
  critical: { omega: 0.6667 },
};

const NV = 15;
const LEVELS = [15, 7, 3, 1];
const LEVEL_H = [0.0625, 0.125, 0.25, 0.5];
const PROF_OFFSET = [0, 15, 22, 25];
const profileFlat = Float64Array.from({ length: 26 }, (_, i) => 0.49 * Math.cos((Math.PI * i) / 26 + 0.1));
const xFine = Float64Array.from({ length: NV }, (_, i) => (i + 1) / (NV + 1));
/** 事件序列：(level, phase, norm)——照抄真实帧的量级与顺序。 */
const EVENTS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 0, 1], [0, 1, 0.929], [1, 2, 30.54], [1, 0, 1e-16], [1, 1, 0.159],
  [2, 2, 7.38], [2, 0, 1e-16], [2, 1, 0.154], [3, 2, 1.809], [3, 0, 1e-16],
  [3, 6, 0.226], [3, 3, 0.226], [2, 4, 0.366], [2, 5, 0.391], [2, 3, 0.391],
  [1, 4, 0.588], [1, 5, 0.607], [1, 3, 0.607], [0, 4, 0.122], [0, 5, 0.0935],
];

const VC_DEF: FrameDef = {
  viz: "vcycle_hierarchy",
  arrays: [
    ["fine_x", xFine, "细网格内点坐标"],
    ["error_profile", Float64Array.from(xFine, (x) => 0.49 * Math.sin(Math.PI * x)), "细网格误差 e(x)"],
    ["error_abs", Float64Array.from(xFine, (x) => Math.abs(0.49 * Math.sin(Math.PI * x))), "|e(x)|"],
    ["error_at_floor", new Float64Array(NV), "已到地板的点"],
    ["error_norm_history", Float64Array.from({ length: 6 }, (_, c) => Math.pow(0.1837, c)), "逐周期误差范数"],
    ["residual_norm_history", Float64Array.from({ length: 6 }, (_, c) => 0.3 * Math.pow(0.2, c)), "逐周期残差范数"],
    ["level_n", Float64Array.from(LEVELS), "各层内点数"],
    ["level_h", Float64Array.from(LEVEL_H), "各层步长"],
    ["profile_flat", profileFlat, "各层限制结果的拼接（按 profile_offset 切）"],
    ["profile_offset", Float64Array.from(PROF_OFFSET), "每层在 profile_flat 里的起点"],
    ["profile_size", Float64Array.from([15, 7, 3, 1]), "每层占多少个点"],
    ["profile_level", Float64Array.from([0, 1, 2, 3]), "每段属于哪一层"],
    ["event_order", Float64Array.from(EVENTS, (_, i) => i), "事件序号"],
    ["event_level", Float64Array.from(EVENTS, (e) => e[0]), "事件发生在哪一层"],
    ["event_phase", Float64Array.from(EVENTS, (e) => e[1]), "事件相位编码"],
    ["event_norm", Float64Array.from(EVENTS, (e) => e[2]), "该层向量范数"],
    ["event_at_floor", Float64Array.from(EVENTS, (e) => (e[2] <= 1e-16 ? 1 : 0)), "事件范数已到地板"],
    ["event_size", Float64Array.from(EVENTS, (e) => LEVELS[e[0]] ?? 1), "事件规模（点数 = 该层内点数）"],
  ],
  meta: {
    content_index: 2, cycle_index: 0, cycles: 5, error_norm_now: 5.302e-5, error_norm_ratio_last: 0.181,
    factor_converging: true, factor_measured: 0.1837371257461596, floor: 1e-16, gamma: 1, levels: 4,
    n_events: 20, n_fine: NV, nu: 1, omega: 0.666667, smoother_index: 0, t_requested: 5, work_finest_level: 2,
    work_per_cycle: 4.933333, cycles_to_1e8: 11, t: 5,
  },
  strings: {
    data_source: "backend", factor_note: "因子 <1 表示收敛，但它不是 0。", floor: "1e-16",
    nonfinite_policy: "非有限值不画。",
    params_json: '{"levels":4,"nu":1,"cycle":"v","smoother":"jacobi","content":"mixed"}',
    phase_legend: "0 enter / 1 pre_smooth / 2 restrict / 3 coarse / 4 correct / 5 post_smooth / 6 coarsest",
    status: "ok", time_semantics: "c 是完整循环数，不是物理时间。",
    trace_note: "事件序列来自 mg_apply(trace=True) 的真实搬运顺序。",
  },
  critical: {},
};

const NSEQ = [7, 15, 31, 63, 127];
const GRID_DEF: FrameDef = {
  viz: "grid_independence",
  arrays: [
    ["n_seq", Float64Array.from(NSEQ), "网格规模序列"],
    ["h_seq", Float64Array.from(NSEQ, (n) => 1 / (n + 1)), "步长序列"],
    ["level_seq", Float64Array.from([3, 4, 5, 6, 7]), "层数序列"],
    ["residual_factor_mg", Float64Array.from([0.160938, 0.183737, 0.189351, 0.188658, 0.190168]), "V-cycle 实测残差因子"],
    ["residual_factor_two_grid", Float64Array.from(NSEQ, () => 0.111111), "两网格因子（粗层精确解）"],
    ["residual_factor_two_grid_defined", Float64Array.from(NSEQ, () => 1), "两网格因子是否测过"],
    ["error_factor_smoother", Float64Array.from([0.949254, 0.98719, 0.99679, 0.99919, 0.99971]), "单层平滑子的误差因子"],
    ["iters_mg_measured", Float64Array.from(NSEQ, () => 2), "MG 实测迭代数"],
    ["iters_smoother", Float64Array.from([56, 226, 907, 3600, 10061]), "单层平滑子达到同样精度的换算迭代数"],
    ["iters_at_cap", Float64Array.from(NSEQ, () => 0), "该尺寸的迭代数是否触到上限"],
    ["cg_iters", Float64Array.from(NSEQ, (n) => n), "CG 迭代数"],
    ["mgcg_iters", Float64Array.from([6, 7, 7, 7, 7]), "MGCG 迭代数"],
    ["mgcg_speedup", Float64Array.from([1.167, 2.143, 4.429, 9, 18.143]), "MGCG 相对 CG 的加速比"],
    ["work_mg_per_cycle", Float64Array.from([6.286, 4.933, 4.387, 4.159, 4.063]), "MG 每周期代价"],
    ["work_to_tol_mg", Float64Array.from([12.571, 9.867, 8.774, 8.317, 8.126]), "MG 到 tol 的总代价"],
    ["work_to_tol_smoother", Float64Array.from([56, 226, 907, 3600, 10061]), "平滑子到 tol 的总代价"],
  ],
  meta: {
    cg_iters_max: 127, converging: true, cycle_index: 0, iters_cap: 1e6, iters_mg_max: 2, iters_smoother_max: 10061,
    levels_max: 7, mg_factor_drift: 0.02922987995958487, mg_factor_max: 0.19016815452800603,
    mg_factor_mean: 0.18257187737611352, mg_factor_min: 0.16093827456842116,
    mg_factor_relative_spread: 0.16010067037525633, mg_factor_std: 0.011046771127791241,
    mgcg_iters_max: 7, n_iters_capped: 0, n_max: 127, nu: 1, omega: 0.666667,
    smoother_modes_done: 3, smoother_slope_h2: -1.8891537362805895, smoother_worst: 0.99971,
    speedup_max: 18.142857, tol: 1e-8, two_grid_factor_spread: 4.857e-16, work_to_tol_mg_max: 12.571,
    work_to_tol_smoother_max: 10061,
  },
  strings: {
    curve_note: "iters_smoother 是换算值，不是实测值。", data_source: "backend",
    grid_independence_note: "「网格无关」指因子与 n 基本无关，不是等于同一个常数。",
    krylov_note: "CG/MGCG 按迭代数比，不是按时间比。", nonfinite_policy: "非有限值不画。",
    params_json: '{"levels_max":7,"cycle":"v","nu":1,"smoother":"jacobi","tol_exp":8}',
    status: "ok", two_grid_note: "两网格因子（粗层精确解）在一维里恒为 1/9。",
  },
  critical: {},
};

const NC = 31;
const RES_DEF: FrameDef = {
  viz: "residual_transfer",
  arrays: [
    ["fine_x", Float64Array.from({ length: N }, (_, i) => (i + 1) / (N + 1)), "细网格内点坐标"],
    ["coarse_x", Float64Array.from({ length: NC }, (_, i) => (i + 1) / (NC + 1)), "粗网格内点坐标"],
    ["residual_fine", Float64Array.from({ length: N }, (_, i) => 0.25 * Math.sin((61 * Math.PI * (i + 1)) / (N + 1))), "细网格残差 r"],
    ["residual_coarse", Float64Array.from({ length: NC }, (_, i) => 0.11 * Math.sin((30 * Math.PI * (i + 1)) / (NC + 1))), "限制到粗网格的 R r"],
    ["residual_prolonged", Float64Array.from({ length: N }, (_, i) => 0.12 * Math.sin((61 * Math.PI * (i + 1)) / (N + 1))), "粗层可见部分 P R r"],
    ["residual_lost", Float64Array.from({ length: N }, (_, i) => 0.13 * Math.sin((61 * Math.PI * (i + 1)) / (N + 1))), "粗层看不见的部分 r − P R r"],
    ["residual_lost_abs", Float64Array.from({ length: N }, () => 0.13), "看不见部分的绝对值"],
    ["residual_at_floor", new Float64Array(N), "已到地板的点"],
    ["loss_history", Float64Array.from([0.7071, 0.42, 0.2, 0.09, 0.04, 0.01, 0.001809]), "混合残差的不可见比例"],
    ["loss_axis", Float64Array.from([0, 1, 2, 3, 4, 5, 6]), "预平滑步数 ν"],
    ["residual_norm_history", Float64Array.from([1, 0.98, 0.94, 0.91, 0.87, 0.83, 0.7037]), "残差范数比"],
    ["residual_norm_at_floor", new Float64Array(7), "残差范数已到地板"],
    ["theta_over_pi", theta63, "θ/π"],
    ["loss_vs_theta", Float64Array.from(theta63, (th) => (Math.abs(th - 0.95) < 0.01 ? 0.999985 : 0.9)), "纯模态的不可见比例"],
    ["coarse_visible_ratio", Float64Array.from(theta63, (th) => Math.max(2e-15, 0.5 * Math.cos((Math.PI * th) / 2))), "粗层能看到多少"],
    ["alias_vs_theta", Float64Array.from(theta63, (th) => (th > 0.5 ? -(2 * th - 2) : th)), "别名后的频率（θ/π）"],
    ["smoothing_vs_theta", Float64Array.from(theta63, (th) => (th < 0.5 ? 1 : 1 / 3)), "平滑子放大率"],
    ["coarse_correction_vs_theta", Float64Array.from(theta63, (th) => 0.5 + 0.9 * th), "两网格校正放大率"],
    ["product_vs_theta", Float64Array.from(theta63, (th) => Math.min(0.47, (th < 0.5 ? 1 : 1 / 3) * (0.5 + 0.9 * th))), "平滑 × 粗校正"],
  ],
  meta: {
    alias_theta_pi: -0.09375, coarse_visible_amplitude: 0.4973, floor: 1e-16,
    hf_ratio_coarse_visible: 3.64e-7, hf_ratio_fine: 1.461e-6, loss_at_current_theta: 0.9999852,
    loss_now: 0.001809, loss_ratio: 0.0025587, loss_unsmoothed: 0.7070969559015992,
    max_coarse_correction: 1.4133618, max_loss: 1, max_product: 0.46998564, max_smoothing: 0.999197,
    mix: 0.5, mode_index: 61, n: N, n_coarse: NC, nu: 6, omega: 0.666667, residual_norm_ratio: 0.7037071,
    restrict_index: 0, smoother_index: 0, step_factor_high: 0.32611767, theta_frac: 0.95, t: 6,
    t_requested: 6,
  },
  strings: {
    aliasing_note: "θ/π=0.95 的高频在粗网格上被认成 −0.094π 的低频：这是别名，不是「看不见」。",
    curve_note: "细/粗/延拓三条线要在同一张图上比。", data_source: "backend", floor: "1e-16",
    nonfinite_policy: "非有限值不画。",
    params_json: '{"n":63,"theta_frac":0.95,"mix":0.5,"restrict":"full_weighting","smoother":"jacobi"}',
    restrict_note: "injection 省事，但它破坏 R = c·Pᵀ 的伴随关系。", shape_note: "形状比大小重要。",
    status: "ok", time_semantics: "ν 是预平滑步数，不是物理时间。",
  },
  critical: { theta_frac: 0.5 },
};

const DEFS: Record<string, FrameDef> = {
  smoothing_spectrum: SS_DEF,
  vcycle_hierarchy: VC_DEF,
  grid_independence: GRID_DEF,
  residual_transfer: RES_DEF,
};
const MODELS = {
  smoothing_spectrum: smoothingSpectrumModel,
  vcycle_hierarchy: vcycleHierarchyModel,
  grid_independence: gridIndependenceModel,
  residual_transfer: residualTransferModel,
};

const sceneOf = async (viz: string, over: Override = {}, opts: Partial<ModelOptions> = {}): Promise<Scene> =>
  MODELS[viz as keyof typeof MODELS](await makeFrame(tweak(DEFS[viz] as FrameDef, over)), { ...OPTS, ...opts });

const axisOf = (scene: Scene, id: string): Axis => {
  const hit = scene.axes.find((a) => a.id === id);
  assert.ok(hit, `场景里没有轴 ${id}`);
  return hit;
};
const layersOn = (scene: Scene, id: string): Layer[] => scene.layers.filter((l) => l.axes === id);
const lineSeriesOn = (scene: Scene, id: string): Array<{ x: ArrayLike<number>; y: ArrayLike<number> }> =>
  layersOn(scene, id).flatMap((l) => (l.kind === "line" ? l.series.map((s) => ({ x: s.x, y: s.y })) : []));
const textsOf = (scene: Scene): string => scene.texts.map((t) => t.text).join(" | ");
/** `degradedBadge` 产出的是一条 `status` 角标（带固定前缀），这里按前缀取。 */
const degradedText = (scene: Scene): string =>
  scene.badges.filter((b) => b.text.includes("降级")).map((b) => b.text).join(" | ");
const badgesOf = (scene: Scene): string => scene.badges.map((b) => b.text).join(" | ");

// --------------------------------------------------------------------------- //
// 1. 四个 viz 在整条时间轴上都过校验器
// --------------------------------------------------------------------------- //

for (const [viz, def] of Object.entries(DEFS)) {
  const tMax = manifest.vizzes.find((v) => v.id === viz)?.time.mode === "none" ? 0 : 6;
  for (const t of [0, Math.round(tMax / 2), tMax]) {
    test(`${viz}: t=${t} 的 Scene 过 validateScene（规则 1/2/8 全绿）`, async () => {
      const scene = await sceneOf(viz, {}, { t });
      assert.deepEqual(validateScene(scene), []);
      assert.ok(scene.pitfalls.length > 0, "pitfalls 面板必须非空（规则 7）");
      assert.ok(scene.layers.length >= 3, "每个 viz 至少 3 个图层");
      assert.equal(scene.chapter, "4.5");
      void def;
    });
  }
}

test("manifest 与登记表一致：4 个 viz 都有 model，未知 id 返回 null", () => {
  assert.deepEqual(
    Object.keys(models).sort(),
    ["grid_independence", "residual_transfer", "smoothing_spectrum", "vcycle_hierarchy"],
  );
  for (const v of manifest.vizzes) assert.ok(modelFor(v.id), `manifest 里的 ${v.id} 没有 model`);
  assert.equal(modelFor("nope"), null);
  assert.equal(manifest.vizzes.length, 4);
});

// --------------------------------------------------------------------------- //
// 2. requiredAux 真的画出来了（按**数值**认，不按 label）
// --------------------------------------------------------------------------- //

/** viz → `requiredAux` 名字 → (轴 id, 该名字对应的真实数组名)。 */
const AUX: Record<string, Record<string, readonly [string, string]>> = {
  smoothing_spectrum: { error: ["error", "error_norm_history"] },
  vcycle_hierarchy: { error: ["convergence", "error_norm_history"], residual: ["convergence", "residual_norm_history"] },
  grid_independence: { residual: ["factors", "residual_factor_mg"], error: ["iterations", "iters_smoother"] },
  residual_transfer: { residual: ["profiles", "residual_fine"] },
};

for (const [viz, need] of Object.entries(AUX)) {
  test(`${viz}: requiredAux 的每个名字都有一条真画出来的曲线`, async () => {
    const scene = await sceneOf(viz);
    const frame = await makeFrame(DEFS[viz] as FrameDef);
    for (const [aux, [axisId, arrayName]] of Object.entries(need)) {
      const want = frame.array(arrayName);
      const hit = lineSeriesOn(scene, axisId).some(
        (s) => s.y.length === want.length && Array.from(s.y).every((v, i) => Math.abs(v - (want[i] as number)) < 1e-12),
      );
      assert.ok(hit, `${viz}.${aux}：轴 ${axisId} 上没有画出 ${arrayName} 的数值`);
    }
  });
}

test("autoscaled 与 manifest 一致：固定窗口的两个轴必须是 false", async () => {
  for (const viz of ["smoothing_spectrum", "vcycle_hierarchy", "grid_independence", "residual_transfer"]) {
    const spec = manifest.vizzes.find((v) => v.id === viz);
    assert.ok(spec);
    const scene = await sceneOf(viz);
    for (const ax of spec.axes) {
      const got = scene.axes.find((a) => a.id === ax.id);
      if (!got) continue; // manifest 只列主面板，模型可以补副面板
      assert.equal(got.autoscaled, ax.autoscaled, `${viz}.${ax.id} 的 autoscaled 与 manifest 不一致`);
    }
  }
  const ss = await sceneOf("smoothing_spectrum");
  assert.deepEqual(axisOf(ss, "factor").ylim, [0, 1.1], "一步放大率必须固定 0–1.1");
  const gi = await sceneOf("grid_independence");
  assert.deepEqual(axisOf(gi, "factors").ylim, [0, 1.05], "因子窗口必须固定 0–1.05（自适应会把漂移拉平）");
});

// --------------------------------------------------------------------------- //
// 3. 临界值参考线（规则 6）：ω=2/3 与 θ/π=0.5
// --------------------------------------------------------------------------- //

test("smoothing_spectrum：θ/π=0.5 分界线 + 1/3 上限线 + ω 的 criticalBadge（规则 6）", async () => {
  const scene = await sceneOf("smoothing_spectrum");
  // x 轴是频率 θ/π，ω 不是一个位置：能画的临界**线**是 θ/π=0.5（粗网格能表示的频率上界，
  // 来自帧数组 half_band_line，不按下标猜）与 factor 面板上的 y=1/3（ω*=2/3 时高频放大率的上限）。
  for (const ax of ["spectrum", "factor"]) {
    const vertical = lineSeriesOn(scene, ax).some(
      (s) => s.x.length === 2 && s.x[0] === 0.5 && s.x[1] === 0.5 && s.y[0] !== s.y[1],
    );
    assert.ok(vertical, `${ax} 面板上缺少 θ/π=0.5 的竖直分界线`);
  }
  const bound = lineSeriesOn(scene, "factor").some(
    (s) => s.y.length === 2 && Math.abs((s.y[0] as number) - 1 / 3) < 1e-12 && s.y[0] === s.y[1] && s.x[0] !== s.x[1],
  );
  assert.ok(bound, "factor 面板上缺少 1/3 上限线（ω*=2/3 时高频放大率的上限）");
  // ω 本身进 criticalBadge：帧的 ω 与理论临界 2/3 都要报出来。
  const off = await sceneOf("smoothing_spectrum", { meta: { omega: 0.8 }, critical: { omega: 0.8 } });
  assert.match(badgesOf(off), /ω = 0\.8000/, "criticalBadge 要报出帧给的 ω");
  assert.match(badgesOf(off), /理论临界 0\.6667/, "理论临界值按 manifest 报，两者要能区分");
});

test("residual_transfer：θ/π=0.5 的临界竖线 + criticalBadge（帧 critical 优先）", async () => {
  const scene = await sceneOf("residual_transfer");
  const vertical = lineSeriesOn(scene, "visibility").some(
    (s) => s.x.length === 2 && s.x[0] === 0.5 && s.x[1] === 0.5 && s.y[0] !== s.y[1],
  );
  assert.ok(vertical, "visibility 面板上缺少 θ/π=0.5 的临界竖线");
  assert.match(badgesOf(scene), /理论临界 0\.5000/, "criticalBadge 要报出临界值");
  // 帧的 critical 与 manifest 不一致时以**帧**为准（不照抄 manifest）。
  const moved = await sceneOf("residual_transfer", { critical: { theta_frac: 0.25 } });
  assert.ok(
    lineSeriesOn(moved, "visibility").some((s) => s.x.length === 2 && s.x[0] === 0.25 && s.x[1] === 0.25),
    "帧 critical.theta_frac=0.25 时竖线必须跟着走",
  );
});

// --------------------------------------------------------------------------- //
// 4. 四个陷阱的"画法"约束
// --------------------------------------------------------------------------- //

test("陷阱③：events 面板上的每条折线只属于一层（绝不跨层连线）", async () => {
  const scene = await sceneOf("vcycle_hierarchy");
  const { event_level: lv, event_norm: nrm } = await makeFrame(VC_DEF).then((f) => ({
    event_level: f.array("event_level"),
    event_norm: f.array("event_norm"),
  }));
  // 每层的范数集合（按值认：同一层的点必然取自该层的值集合）。
  const byLevel = new Map<number, Set<number>>();
  for (let i = 0; i < lv.length; i += 1) {
    const k = Math.round(lv[i] as number);
    if (!byLevel.has(k)) byLevel.set(k, new Set());
    (byLevel.get(k) as Set<number>).add(Math.log10(Math.max(nrm[i] as number, 1e-17)));
  }
  const lines = lineSeriesOn(scene, "events");
  assert.ok(lines.length >= 4, "至少每层一条折线");
  for (const s of lines) {
    const logs = Array.from(s.y, (v) => Math.log10(Math.max(v, 1e-17)));
    const owner = [...byLevel.entries()].filter(([, set]) => logs.every((v) => [...set].some((u) => Math.abs(u - v) < 1e-9)));
    assert.equal(owner.length, 1, `有一条折线跨越了多层（或不属于任何一层）：${logs.join(",")}`);
  }
});

test("陷阱④：纯模态的不可见比例画成水平参考线（与混合残差同轴对比）", async () => {
  const scene = await sceneOf("residual_transfer");
  const flat = lineSeriesOn(scene, "loss").some(
    (s) => s.y.length === 2 && Math.abs((s.y[0] as number) - 0.9999852) < 1e-9 && s.y[0] === s.y[1],
  );
  assert.ok(flat, "loss 面板上必须画出纯模态的常数水平线");
  const hist = lineSeriesOn(scene, "loss").some((s) => s.y.length === 7);
  assert.ok(hist, "loss 面板上还要有随 ν 下降的混合残差曲线");
  assert.match(textsOf(scene), /纯高频模态不可见比例 0\.999985/);
  assert.match(textsOf(scene), /把高频认成低频/, "别名后的频率必须写明（负号=方向反转）");
  assert.match(textsOf(scene), /-0\.09375π/);
});

test("陷阱①：因子曲线带均值/标准差带，且把漂移写进文本", async () => {
  const scene = await sceneOf("grid_independence");
  const bands = layersOn(scene, "factors").filter((l) => l.kind === "band");
  assert.equal(bands.length, 1, "factors 面板上要有一条均值±标准差带");
  const band = bands[0];
  assert.ok(band.kind === "band" && band.band.ylo.length >= 2);
  assert.match(textsOf(scene), /均值 0\.1826/);
  assert.match(textsOf(scene), /std 0\.01105/);
  assert.match(textsOf(scene), /漂移 \+0\.02923/);
  assert.match(textsOf(scene), /无关 ≠ 常数/);
});

test("陷阱②：W-cycle 的因子更小 ≠ 更省（代价账必须在场）", async () => {
  const scene = await sceneOf("grid_independence");
  assert.match(textsOf(scene), /W 因子更小 ≠ 更省/);
  const work = lineSeriesOn(scene, "cost");
  assert.ok(work.some((s) => s.y.length === 5), "cost 面板上要有 5 个尺寸的代价曲线");
  assert.match(textsOf(scene), /代价\/周期 6\.286→4\.063/);
});

test("vcycle：代价账按整周期算（不是「粗层几乎免费」）", async () => {
  const scene = await sceneOf("vcycle_hierarchy");
  assert.match(textsOf(scene), /代价账 4\.933 单位\/周期（最细层自身 2）/);
  assert.match(textsOf(scene), /粗层不是免费的/);
  assert.match(textsOf(scene), /事件账 L0 4\/60点　L1 6\/42点/);
  assert.match(textsOf(scene), /跨层比范数是错觉|每层各自连线|点数\/h\/口径都不同/);
});

// --------------------------------------------------------------------------- //
// 5. 时间轴语义与掩码
// --------------------------------------------------------------------------- //

test("grid_independence 没有时间轴：scene.t 恒为 0，且不引用 tMax", async () => {
  for (const t of [0, 3, 99]) {
    const scene = await sceneOf("grid_independence", {}, { t });
    assert.equal(scene.t, 0, "time.mode=none 的 viz 不许把 opts.t 写进 scene.t");
  }
});

test("mode_present / residual_factor_two_grid_defined 掩码：未测的点不画", async () => {
  // 只有一个模态存在（content=low 的真实情形）。
  const one = await sceneOf("smoothing_spectrum", {
    arrays: { mode_present: Float64Array.from({ length: N }, (_, i) => (i === 0 ? 1 : 0)) },
  });
  const xs = lineSeriesOn(one, "spectrum")
    .filter((s) => s.x.length > 2)
    .flatMap((s) => Array.from(s.x));
  assert.ok(Math.max(...xs) < 0.05, "只有 k=1 存在时，振幅曲线的 x 不该越过第一个模态");
  // 两网格因子只在"测过"的位置画：全 0 掩码 → 一条两网格曲线都不该有。
  const none = await sceneOf("grid_independence", {
    arrays: { residual_factor_two_grid_defined: new Float64Array(NSEQ.length) },
  });
  assert.deepEqual(validateScene(none), []);
  const twoGrid = lineSeriesOn(none, "factors").filter((s) => s.x.length === NSEQ.length && Math.abs((s.y[0] as number) - 0.1111) < 1e-3);
  assert.equal(twoGrid.length, 0, "掩码全 0 时不许画两网格曲线（0.0 会被读成「一步收敛」）");
});

test("grid_independence：迭代数触到上限必须显式告警（截断值不是实测值）", async () => {
  const scene = await sceneOf("grid_independence", {
    arrays: { iters_at_cap: Float64Array.from([0, 0, 1, 0, 1]) },
    meta: { n_iters_capped: 2 },
  });
  assert.deepEqual(validateScene(scene), []);
  assert.match(textsOf(scene), /触到迭代上限/);
  assert.match(textsOf(scene), /⚠/);
});

// --------------------------------------------------------------------------- //
// 6. 降级不静默
// --------------------------------------------------------------------------- //

test("缺可选数组/可选 meta → 不抛错，但 degraded 角标点名", async () => {
  const cases: Array<[string, Override, RegExp]> = [
    ["smoothing_spectrum", { drop: ["amplitude_at_floor"] }, /amplitude_at_floor/],
    ["smoothing_spectrum", { dropMeta: ["measured_smoothing_factor"] }, /measured_smoothing_factor|平滑因子/],
    ["vcycle_hierarchy", { drop: ["event_at_floor"] }, /event_at_floor/],
    ["residual_transfer", { drop: ["residual_at_floor"] }, /residual_at_floor/],
    ["residual_transfer", { dropMeta: ["alias_theta_pi"] }, /alias_theta_pi/],
    ["grid_independence", { drop: ["iters_at_cap"] }, /iters_at_cap/],
  ];
  for (const [viz, over, want] of cases) {
    const scene = await sceneOf(viz, over);
    assert.deepEqual(validateScene(scene), [], `${viz} 降级后仍要过校验器`);
    const degraded = degradedText(scene);
    assert.ok(degraded.length > 0, `${viz} 降级后必须有 degraded 角标（不许静默）`);
    // `degradedBadge` 只印前两项，其余压成「等 N 项」——按它的口径认。
    assert.ok(
      want.test(degraded) || /等 \d+ 项/.test(degraded),
      `${viz} 降级说明里既没点名 ${want}，也没有「等 N 项」：${degraded}`,
    );
  }
});

test("必填数组缺失 → 抛 ModelInputError（不静默画空图）", async () => {
  for (const [viz, name] of [
    ["smoothing_spectrum", "theta_over_pi"],
    ["vcycle_hierarchy", "event_level"],
    ["grid_independence", "residual_factor_mg"],
    ["residual_transfer", "loss_vs_theta"],
  ] as const) {
    await assert.rejects(async () => sceneOf(viz, { drop: [name] }), ModelInputError);
  }
});

test("pitfalls 为空 → 抛错（规则 7：防误导面板必须有内容）", async () => {
  await assert.rejects(async () => {
    const frame = await makeFrame({ ...SS_DEF, pitfalls: "" });
    smoothingSpectrumModel(frame, OPTS);
  }, ModelInputError);
});

test("窄画布：420px 仍要合法（走 degraded），再窄就明确报错而不是画一张读不出来的图", async () => {
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz, {}, { width: 420, height: 640 });
    assert.deepEqual(validateScene(scene), [], `${viz} 在 420×640 下仍要合法`);
    for (const ax of scene.axes) assert.ok(ax.rect[2] > 0 && ax.rect[3] > 0, `${viz}.${ax.id} 宽高必须为正`);
    assert.ok(degradedText(scene).length > 0, `${viz} 放弃说明文字时必须点名降级`);
  }
  await assert.rejects(async () => sceneOf("grid_independence", {}, { width: 320, height: 640 }), ModelInputError);
});

test("null meta 值按缺失处理（不把 null 当 0 画进图里）", async () => {
  const scene = await sceneOf("residual_transfer", {
    meta: { coarse_visible_amplitude: null, loss_at_current_theta: null },
  });
  assert.deepEqual(validateScene(scene), []);
  assert.ok(!/粗层可见幅度 null/.test(textsOf(scene)), "null 不许被格式化成「null」写进图里");
  const degraded = degradedText(scene);
  assert.match(degraded, /coarse_visible_amplitude|loss_at_current_theta/);
});
