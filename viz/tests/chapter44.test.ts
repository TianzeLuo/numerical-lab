/**
 * 4.4 章（有限元方法）model 层测试 —— **自足**：不依赖 Python，也不依赖
 * `data/frames` 是否存在。夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，
 * 字段集合（数组名 / dtype / shape / unit / label、meta / strings / critical 键）逐字照抄
 * `server/chapters/4.4.py` 的真实帧（数值是测试用的合成值，量级与真实帧一致）。
 *
 * 本文件守的是这一章最容易骗人的四处：
 *
 * - **装配**：未定型格子的哨兵必须落在 `clipBelow` 之下（灰=无数据），主元比的
 *   1e300 显示上限不得进图层；`symmetry_residual=0` 不是合格证；
 * - **拖动载荷**：`scan_at_floor` 掩码点不进曲线；奇异帧不画位移/扫描，也不写假的 0 误差；
 * - **hp**：地板点被切开；条件数未测的 1e300 不进图层；两个横轴（自由度 / p）都在；
 * - **自适应**：效率指数的占位值 1e-12 永不进图层；地板层不画；同自由度比较用 dof 轴。
 *
 * 缺字段纪律：必需数组缺失 → `ModelInputError`；可选掩码/meta 缺失 → 降级角标点名。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, toNumeric, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { sceneToJson, validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import {
  adaptiveMeshModel,
  assemblyAnimationModel,
  hpConvergenceModel,
  manifest,
  modelFor,
  models,
  solveDeformModel,
} from "../chapters/4.4/index.js";

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
  "它会怎样骗你：第一，对称残差为 0 不是合格证，double 路线仍然对称正定但矩阵是错的；" +
  "第二，节点值准推不出解准，点载荷的误差随载荷位置呈锯齿；" +
  "第三，p-加密的指数收敛只在光滑解上成立，尖角解上退化成代数；" +
  "第四，效率指数是分布不是常数，同自由度比较必须固定自由度；" +
  "第五，掩码点与地板占位值不能画成数据，越窗必须由溢出角标点名。";

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
        chapter: "4.4",
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

const buildScene = async (def: FrameDef, opts: Partial<ModelOptions> = {}): Promise<Scene> =>
  models[def.viz](await makeFrame(def), { ...OPTS, ...opts });

const sceneOf = async (viz: string, over: Override = {}, opts: Partial<ModelOptions> = {}): Promise<Scene> =>
  buildScene(tweak(DEFS[viz] as FrameDef, over), opts);

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
const hasSeries = (
  seriesList: Array<{ x: ArrayLike<number>; y: ArrayLike<number> }>,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  tol = 1e-12,
): boolean =>
  seriesList.some(
    (s) =>
      s.y.length === y.length &&
      s.x.length === x.length &&
      Array.from(s.y).every((v, i) => Math.abs(v - (y[i] as number)) <= tol) &&
      Array.from(s.x).every((v, i) => Math.abs(v - (x[i] as number)) <= tol),
  );

function countNonFiniteDeep(value: unknown): number {
  if (typeof value === "number") return Number.isFinite(value) ? 0 : 1;
  if (typeof value === "bigint") return 0;
  if (ArrayBuffer.isView(value)) {
    const a = value as unknown as ArrayLike<number>;
    let n = 0;
    for (let i = 0; i < a.length; i += 1) if (!Number.isFinite(a[i] as number)) n += 1;
    return n;
  }
  if (Array.isArray(value)) return value.reduce<number>((n, v) => n + countNonFiniteDeep(v), 0);
  if (value !== null && typeof value === "object") {
    return Object.values(value as Record<string, unknown>).reduce<number>((n, v) => n + countNonFiniteDeep(v), 0);
  }
  return 0;
}

function layerNumbers(layer: Layer): number[] {
  switch (layer.kind) {
    case "line":
      return layer.series.flatMap((s) => [...Array.from(s.x), ...Array.from(s.y)]).map(Number);
    case "scatter":
      return [...Array.from(layer.x), ...Array.from(layer.y)].map(Number);
    case "heat":
    case "contour":
      return Array.from(layer.values).map(Number);
    case "bars":
      return [...Array.from(layer.x), ...Array.from(layer.values)].map(Number);
    default:
      return [];
  }
}

function allLayerNumbers(scene: Scene): number[] {
  return scene.layers.flatMap(layerNumbers);
}

// --------------------------------------------------------------------------- //
// 夹具
// --------------------------------------------------------------------------- //

// —— assembly：n=4（P1），t=0.5，已装 2 个单元 ——
const ASM_FINAL = [
  8, -8, 0, 0, 0,
  -8, 16, -8, 0, 0,
  0, -8, 16, -8, 0,
  0, 0, -8, 16, -8,
  0, 0, 0, -8, 8,
];
const ASM_PARTIAL = [
  8, -8, 0, 0, 0,
  -8, 16, -8, 0, 0,
  0, -8, 8, 0, 0,
  0, 0, 0, 0, 0,
  0, 0, 0, 0, 0,
];
const ASM_DONE = [
  0, 0, -1, -1, -1,
  0, 1, 1, -1, -1,
  -1, 1, 2, 2, -1,
  -1, -1, 2, 3, 3,
  -1, -1, -1, 3, 3,
];
const ASM_PENDING = ASM_DONE.map((v) => (v < 0 ? 1 : 0));
const ASM_DEF: FrameDef = {
  viz: "assembly_animation",
  arrays: [
    ["mesh_x", [0, 0.25, 0.5, 0.75, 1], "网格顶点 x"],
    ["mesh_h", [0.25, 0.25, 0.25, 0.25], "单元长度 h_K"],
    ["K_partial", ASM_PARTIAL, "部分 K", [5, 5]],
    ["K_route", ASM_FINAL, "路线 K", [5, 5]],
    ["K_final", ASM_FINAL, "完整 K", [5, 5]],
    ["K_defect", new Array(25).fill(0), "路线 − 完整", [5, 5]],
    ["k_local", [8, -8, -8, 8], "局部 ke", [2, 2]],
    ["local_dofs", [2, 3], "当前单元自由度"],
    ["entry_done_at", ASM_DONE, "定型单元", [5, 5]],
    ["entry_pending", ASM_PENDING, "未定型掩码", [5, 5]],
    ["k_axis", [0, 1, 2, 3, 4], "已装单元数 k"],
    ["residual", [32, 32, 32, 16, 1e-17], "一致性残差"],
    ["residual_at_floor", [0, 0, 0, 0, 1], "残差地板掩码"],
    ["symmetry_residual", [0, 0, 0, 0, 0], "对称残差"],
    ["pivot_condition", [1e300, 1e300, 1e300, 14, 1.75], "主元比"],
    ["pivot_unmeasured", [1, 1, 1, 0, 0], "主元未测掩码"],
    ["u_full", [0, 0.1, 0.2, 0.1, 0], "完整解"],
    ["u_route", [0, 0.1, 0.2, 0.1, 0], "路线解"],
    ["solution_defect", [1e-17, 1e-17, 1e-17, 1e-17, 1e-17], "解差"],
  ],
  meta: {
    n_elem: 4,
    n_dof: 5,
    npe: 2,
    degree_index: 0,
    mirror_index: 0,
    coefficient_index: 0,
    t: 0.5,
    k_done: 2,
    entries_pending: 11,
    route_defect: 0,
    solve_defect: 0,
    lambda_min_final: 1.2,
    lambda_max_final: 18,
    condition_final: 15,
    eigen_exact: true,
    symmetry_max_over_history: 0,
  },
  strings: {
    data_source: "backend",
    floor: "1e-17",
    time_semantics: "装配进度",
    reading_note: "主图看矩阵累积。",
  },
  critical: {},
};

// —— solve_deform：4 单元梁，节点载荷 x0=0.5 ——
const SD_CURVE_X = Array.from({ length: 12 }, (_, i) => i / 11);
const SD_CURVE_Y = SD_CURVE_X.map((x) => 0.1 * x * (1 - x));
const SD_DEF: FrameDef = {
  viz: "solve_deform",
  arrays: [
    ["mesh_xy", [0, 0, 0.25, 0, 0.5, 0, 0.75, 0, 1, 0], "节点 (x,0)", [5, 2]],
    [
      "elements",
      [0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9],
      "单元 → 自由度",
      [4, 4],
    ],
    ["solution_curve", SD_CURVE_X.flatMap((x, i) => [x, SD_CURVE_Y[i] as number]), "插值曲线", [12, 2]],
    ["solution", [0, 0.01875, 0.025, 0.01875, 0], "节点位移"],
    ["solution_exact", [0, 0.01875, 0.025, 0.01875, 0], "精确节点位移"],
    ["rotation", [0.1, 0.05, 0, -0.05, -0.1], "转角"],
    ["error", [1e-17, 3e-15, 4e-15, 3e-15, 1e-17], "节点误差"],
    ["error_energy_scan", [1e-2, 5e-3, 1e-15, 5e-3, 1e-2], "能量误差扫描"],
    ["nodal_error_scan", [1e-16, 2e-16, 1e-17, 2e-16, 1e-16], "节点误差扫描"],
    ["scan_at_floor", [0, 0, 1, 0, 0], "节点误差地板掩码"],
    ["scan_x0", [0.1, 0.3, 0.5, 0.7, 0.9], "载荷位置"],
    ["load_axis", [0, 0.5, 1], "加载因子"],
    ["energy_law", [0, 1e-15, 4e-15], "λ² 能量律"],
    ["mark_x0", [0.5], "当前载荷位置"],
    ["mark_energy", [1e-15], "当前能量误差"],
    ["singular_mask", [0, 0, 0, 0, 0], "奇异掩码"],
  ],
  meta: {
    model_index: 0,
    n_elem: 4,
    n_dof: 10,
    load_factor: 0.5,
    x0: 0.5,
    node_distance: 0,
    load_on_node: true,
    h2: 1e-14,
    energy: 2.5e-15,
    l2: 1e-16,
    h1: 1e-8,
    nodal_error: 4e-15,
    w_max: 0.025,
    theta_max: 0.1,
    condition: 100,
    residual: 1e-16,
    scan_min_error: 1e-15,
    scan_max_error: 1e-2,
    scan_peak_to_peak: 1e13,
    scan_at_floor_count: 1,
    scan_points: 5,
    singular: false,
  },
  strings: {
    data_source: "backend",
    kind: "beam",
    floor: "1e-17",
    fail_reason: "",
    time_semantics: "准静态加载",
    reading_note: "锯齿来自载荷与节点的相对位置。",
  },
  critical: { x0: 0.5 },
};

// —— solve_deform：2×2 四边形膜（额外覆盖 2D 线框路径）——
const MB_NODES = [
  0, 0, 0.5, 0, 1, 0,
  0, 0.5, 0.5, 0.5, 1, 0.5,
  0, 1, 0.5, 1, 1, 1,
];
const MB_DEF: FrameDef = {
  viz: "solve_deform",
  arrays: [
    ["mesh_xy", MB_NODES, "节点 (x,y)", [9, 2]],
    ["elements", [0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7], "单元 → 顶点", [4, 4]],
    ["solution", [0, 0, 0, 0, 0.2, 0, 0, 0, 0], "节点位移"],
    ["solution_exact", [0, 0, 0, 0, 0.2, 0, 0, 0, 0], "精确节点位移"],
    ["error", [1e-17, 1e-17, 1e-17, 1e-17, 4e-2, 1e-17, 1e-17, 1e-17, 1e-17], "节点误差"],
    ["error_element", [0.01, 0.01, 0.01, 0.01], "逐单元误差"],
    ["error_energy_scan", [0.2, 0.3, 0.4, 0.3, 0.2], "能量误差扫描"],
    ["scan_x0", [0.1, 0.3, 0.5, 0.7, 0.9], "载荷位置"],
    ["load_axis", [0, 0.5, 1], "加载因子"],
    ["energy_law", [0, 0.1, 0.4], "λ² 能量律"],
    ["mark_x0", [0.5], "当前载荷中心"],
    ["mark_energy", [0.4], "当前能量误差"],
    ["element_h", [0.7, 0.7, 0.7, 0.7], "单元直径"],
    ["singular_mask", new Array(9).fill(0), "奇异掩码"],
  ],
  meta: {
    model_index: 1,
    n_grid: 2,
    n_node: 9,
    n_elem: 4,
    element_index: 1,
    distortion: 0,
    sigma: 0.2,
    load_factor: 0.5,
    x0: 0.5,
    singular: false,
    energy: 0.1,
    l2: 0.01,
    h1: 0.2,
    nodal_error: 0.04,
    u_max: 0.2,
    jac_ratio: 1,
  },
  strings: {
    data_source: "backend",
    kind: "membrane",
    floor: "1e-17",
    fail_reason: "",
    time_semantics: "准静态加载",
  },
  critical: { x0: 0.5, distortion: 0.16 },
};

// —— hp：pmax=4，hmax=3 ——
const HP_DEF: FrameDef = {
  viz: "hp_convergence",
  arrays: [
    ["dof_h", [3, 7, 15], "h 自由度"],
    ["error", [0.5, 0.25, 0.125], "h 误差"],
    ["dof_p", [0, 1, 2, 3], "p 自由度"],
    ["error_p", [2.2, 0.27, 0.0087, 1e-4], "p 误差"],
    ["error_at_floor", [0, 0, 0], "h 地板掩码"],
    ["error_p_at_floor", [0, 0, 0, 0], "p 地板掩码"],
    ["p_axis", [1, 2, 3, 4], "阶次 p"],
    ["error_vs_p", [2.2, 0.27, 0.0087, 1e-4], "误差 vs p"],
    ["fit_h", [0.5, 0.23, 0.115], "h 拟合"],
    ["fit_p_exp", [2.2, 0.3, 0.04, 0.006], "半对数拟合"],
    ["fit_p_alg", [2.2, 0.03, 0.002, 0.0004], "双对数拟合"],
    ["condition_p", [1e300, 1, 2.4, 5], "条件数"],
    ["condition_unmeasured", [1, 0, 0, 0], "条件数未测掩码"],
    ["focus_dof", [3, 7, 15], "聚焦策略自由度"],
    ["focus_error", [0.5, 0.25, 0.125], "聚焦策略误差"],
    ["mark_equal_dof_p", [3], "等自由度标记"],
    ["mark_equal_dof_h", [3], "等自由度标记"],
    ["mark_h_last", [15], "h 末端"],
    ["error_h_at_mark", [0.25], "等自由度 h 误差"],
    ["error_p_at_mark", [1e-4], "等自由度 p 误差"],
    ["error_h_last_at_mark", [0.125], "h 末端误差"],
  ],
  meta: {
    solution_index: 0,
    strategy_index: 0,
    metric_index: 0,
    pmax: 4,
    hmax: 3,
    n_h: 3,
    n_p: 4,
    dof_h_last: 15,
    dof_p_last: 3,
    err_h_last: 0.125,
    err_p_last: 1e-4,
    slope_h_dof: -1,
    r2_h_dof: 0.999,
    slope_p_semilog: -2,
    r2_p_semilog: 0.95,
    slope_p_loglog: -4,
    r2_p_loglog: 0.8,
    exponential_better: true,
    equal_dof: 3,
    err_h_at_pdof: 0.25,
    err_p_at_hdof: 1e-4,
    gain_p_over_h: 2500,
    h_dof_gap: 1250,
    cond_p_first: -1,
    cond_p_last: 5,
    cond_growth: 5,
    cond_at_pmax_measured: true,
  },
  strings: {
    data_source: "backend",
    floor: "1e-17",
    time_semantics: "无时间轴",
    reading_note: "两个横轴一起看。",
  },
  critical: {},
};

// —— adaptive：3 单元、4 层 ——
const AD_DEF: FrameDef = {
  viz: "adaptive_mesh",
  arrays: [
    ["mesh_x", [0, 0.5, 0.75, 1], "网格顶点"],
    ["elements", [0, 1, 1, 2, 2, 3], "单元 → 顶点", [3, 2]],
    ["element_h", [0.5, 0.25, 0.25], "单元长度"],
    ["element_eta", [0.9, 0.2, 1e-16], "指示子 η_K"],
    ["error", [0.3, 0.1, 1e-6], "真误差"],
    ["element_ratio", [1.1, 2, 1e-12], "效率指数"],
    ["ratio_undefined", [0, 0, 1], "效率指数掩码"],
    ["element_eta_share", [0.9, 0.09, 0.01], "η 份额"],
    ["solution", [0, 0.1, 0.05, 0], "节点解"],
    ["solution_exact", [0, 0.1, 0.05, 0], "精确解"],
    ["level_axis", [0, 1, 2, 3], "加密层数"],
    ["dof_adaptive", [3, 5, 9, 17], "自适应自由度"],
    ["error_adaptive", [0.9, 0.5, 0.2, 0.08], "自适应误差"],
    ["dof_uniform", [3, 5, 9, 17], "一致自由度"],
    ["error_uniform", [0.9, 0.55, 0.25, 0.1], "一致误差"],
    ["effectivity", [1.9, 1.4, 1.7, 1.25], "总效率指数"],
    ["error_at_floor", [0, 0, 0, 0], "误差地板掩码"],
    ["mark_level", [2], "当前层"],
    ["compare_ratio", [1, 0.9, 0.8, 0.75], "同自由度比值"],
    ["compare_unmeasured", [0, 0, 0, 0], "比值未测掩码"],
    ["compare_index", [0, 1, 2, 3], "比较序号"],
  ],
  meta: {
    t: 0.5,
    strategy_index: 0,
    theta: 0.5,
    ratio: 100,
    levels: 3,
    n0: 2,
    level: 2,
    n_elem: 3,
    n_node: 4,
    n_dof: 4,
    n_free: 2,
    min_h: 0.25,
    max_h: 0.5,
    h_ratio: 2,
    eta_sum: 2,
    eta_norm: 1.1,
    eta_max: 0.9,
    eta_share_max: 0.9,
    eta_over_half: 1,
    error_now: 0.2,
    error_l2_now: 0.01,
    effectivity_total: 1.7,
    effectivity_min_defined: 1.1,
    effectivity_max_defined: 2,
    effectivity_spread: 1.8,
    effectivity_median: 1.5,
    effectivity_q25: 1.2,
    effectivity_q75: 1.8,
    effectivity_defined_count: 2,
    ratio_undefined_count: 1,
    indicator_noise_count: 0,
    indicator_blind_count: 1,
    indicator_zero_count: 0,
    error_share_top3: 0.99,
    eta_share_top3: 1,
    undefined_error_share: 0.01,
    blind_error_share: 0.01,
    log_corr: 0.4,
    marked_fraction: 0.5,
    best_ratio: 0.75,
    median_ratio: 0.8,
    ratio_at_equal_dof: 0.8,
    uniform_levels_used: 3,
    uniform_truncated: false,
    uniform_dof_last: 17,
    adaptive_dof_last: 17,
    uniform_error_last: 0.1,
    adaptive_error_last: 0.08,
    kink_element_index: 0,
    kink_error_share: 0.9,
    error_now_equals_eta_norm_ratio: 1.7,
  },
  strings: {
    data_source: "backend",
    floor: "1e-17",
    time_semantics: "加密层数",
    reading_note: "η 是分布。",
  },
  critical: { theta: 1 },
};

const DEFS: Record<string, FrameDef> = {
  assembly_animation: ASM_DEF,
  solve_deform: SD_DEF,
  hp_convergence: HP_DEF,
  adaptive_mesh: AD_DEF,
};

// --------------------------------------------------------------------------- //
// 1. 登记表 / 时间轴 / 校验器
// --------------------------------------------------------------------------- //

test("manifest 与登记表一致：4 个 viz 都有 model，未知 id 返回 null", () => {
  assert.deepEqual(Object.keys(models).sort(), [
    "adaptive_mesh",
    "assembly_animation",
    "hp_convergence",
    "solve_deform",
  ]);
  assert.equal(manifest.vizzes.length, 4);
  for (const v of manifest.vizzes) assert.ok(modelFor(v.id), `manifest 里的 ${v.id} 没有 model`);
  assert.equal(modelFor("nope"), null);
  assert.equal(models.assembly_animation, assemblyAnimationModel);
  assert.equal(models.solve_deform, solveDeformModel);
  assert.equal(models.hp_convergence, hpConvergenceModel);
  assert.equal(models.adaptive_mesh, adaptiveMeshModel);
});

test("hp_convergence: mode=none，scene.t 恒为 0（不把 opts.t 写进去）", async () => {
  for (const t of [0, 0.5, 99]) {
    const scene = await sceneOf("hp_convergence", {}, { t });
    assert.deepEqual(validateScene(scene), []);
    assert.equal(scene.chapter, "4.4");
    assert.equal(scene.viz, "hp_convergence");
    assert.equal(scene.t, 0);
    assert.ok(scene.pitfalls.length > 0);
  }
});

test("scrub vizzes：scene.t 取帧的 meta.t（不是请求值）", async () => {
  const asm = await sceneOf("assembly_animation", {}, { t: 0.9 });
  assert.ok(Math.abs(asm.t - 0.5) < 1e-12, "装配 scene.t 应取 meta.t=0.5");
  const sd = await sceneOf("solve_deform", {}, { t: 0.9 });
  assert.ok(Math.abs(sd.t - 0.5) < 1e-12, "加载 scene.t 应取 load_factor=0.5");
  const ad = await sceneOf("adaptive_mesh", {}, { t: 0.9 });
  assert.ok(Math.abs(ad.t - 0.5) < 1e-12, "自适应 scene.t 应取 meta.t=0.5");
});

// --------------------------------------------------------------------------- //
// 2. requiredAux 真的画在对应轴上（按数值认，不按 label）
// --------------------------------------------------------------------------- //

test("assembly: requiredAux=residual/symmetry_residual 都在 residuals 轴上（P2 型非零对称残差）", async () => {
  const sym = [2e-15, 1e-15, 3e-15, 2e-15, 1e-15];
  const frame = await makeFrame(tweak(ASM_DEF, { arrays: { symmetry_residual: sym } }));
  const scene = models.assembly_animation(frame, OPTS);
  const series = lineSeriesOn(scene, "residuals");
  assert.ok(
    hasSeries(series, Array.from(arrOf(frame, "k_axis")).slice(0, 4), Array.from(arrOf(frame, "residual")).slice(0, 4)),
    "residuals 轴缺 residual（掩码标为地板的末点按约定不进曲线）",
  );
  assert.ok(hasSeries(series, arrOf(frame, "k_axis"), Float64Array.from(sym)), "residuals 轴缺 symmetry_residual");
});

test("solve_deform: requiredAux=error 在 error 轴上、error_energy_scan 在 scan 轴上", async () => {
  const frame = await makeFrame(SD_DEF);
  const scene = models.solve_deform(frame, OPTS);
  assert.ok(hasSeries(lineSeriesOn(scene, "error"), Array.from({ length: 5 }, (_, i) => i), arrOf(frame, "error")));
  assert.ok(hasSeries(lineSeriesOn(scene, "scan"), arrOf(frame, "scan_x0"), arrOf(frame, "error_energy_scan")));
});

test("hp: requiredAux=error 在 ladder 轴上、error_p 在 vs_p 轴上（按 p）", async () => {
  const frame = await makeFrame(HP_DEF);
  const scene = models.hp_convergence(frame, OPTS);
  assert.ok(hasSeries(lineSeriesOn(scene, "ladder"), arrOf(frame, "dof_h"), arrOf(frame, "error")));
  assert.ok(hasSeries(lineSeriesOn(scene, "vs_p"), arrOf(frame, "p_axis"), arrOf(frame, "error_vs_p")));
  // error_p 也应出现在 ladder 上（按自由度，p=1 的 dof=0 落在对数轴外，被排除）
  const pSeries = lineSeriesOn(scene, "ladder").filter(
    (s) => s.y.length === 3 && Math.abs((s.y[0] as number) - 0.27) < 1e-12,
  );
  assert.ok(pSeries.length > 0, "ladder 轴上缺 p 阶梯误差（error_p）");
});

test("adaptive_mesh: requiredAux=element_eta/error 都在 indicator 轴上", async () => {
  const frame = await makeFrame(AD_DEF);
  const scene = models.adaptive_mesh(frame, OPTS);
  const series = lineSeriesOn(scene, "indicator");
  const eta = arrOf(frame, "element_eta");
  const err = arrOf(frame, "error");
  assert.ok(series.some((s) => s.y.length === eta.length && Array.from(s.y).every((v, i) => v === eta[i])), "indicator 缺 element_eta");
  assert.ok(series.some((s) => s.y.length === err.length && Array.from(s.y).every((v, i) => v === err[i])), "indicator 缺 error");
});

// --------------------------------------------------------------------------- //
// 3. 轴契约 / 固定窗口
// --------------------------------------------------------------------------- //

test("每个 manifest 声明轴都用 manifest 标题；autoscaled 标志与 manifest 一致", async () => {
  const manifestOf = (viz: string) => manifest.vizzes.find((v) => v.id === viz);
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz);
    const spec = manifestOf(viz);
    assert.ok(spec);
    for (const ax of spec.axes) {
      const axis = axisOf(scene, ax.id);
      assert.equal(axis.title, ax.title, `${viz}.${ax.id} 的标题应取 manifest`);
      assert.equal(axis.autoscaled, ax.autoscaled, `${viz}.${ax.id} 的 autoscaled 应与 manifest 一致`);
    }
  }
});

test("assembly: 矩阵/残差窗固定，局部矩阵色标自适应", async () => {
  const scene = await sceneOf("assembly_animation");
  const m = axisOf(scene, "matrix");
  assert.equal(m.autoscaled, false);
  assert.deepEqual(m.xlim, [-0.5, 4.5]);
  assert.deepEqual(m.ylim, [-0.5, 4.5]);
  const r = axisOf(scene, "residuals");
  assert.equal(r.autoscaled, false);
  assert.equal(r.yscale, "log");
  assert.equal(r.ylim[0], 1e-17);
  assert.equal(axisOf(scene, "local").autoscaled, true);
  const heat = layersOn(scene, "matrix").find((l) => l.kind === "heat");
  assert.ok(heat, "矩阵面板必须是 heatLayer");
  assert.ok(heat.kind === "heat" && heat.clipBelow !== undefined, "heatLayer 必须有 clipBelow 才能藏住未定型哨兵");
});

test("solve_deform: 主图自动缩放，扫描/误差副轴固定对数窗", async () => {
  const scene = await sceneOf("solve_deform");
  assert.equal(axisOf(scene, "deformed").autoscaled, true);
  for (const id of ["scan", "error"]) {
    const ax = axisOf(scene, id);
    assert.equal(ax.autoscaled, false);
    assert.equal(ax.yscale, "log");
    assert.deepEqual(ax.ylim, [1e-17, 1]);
  }
});

test("hp: ladder 双对数固定 1–300 / 1e-17–10；vs_p 固定；condition 自适应", async () => {
  const scene = await sceneOf("hp_convergence");
  const ladder = axisOf(scene, "ladder");
  assert.equal(ladder.xscale, "log");
  assert.equal(ladder.yscale, "log");
  assert.equal(ladder.autoscaled, false);
  assert.deepEqual(ladder.xlim, [1, 300]);
  assert.deepEqual(ladder.ylim, [1e-17, 10]);
  const vsP = axisOf(scene, "vs_p");
  assert.equal(vsP.autoscaled, false);
  assert.deepEqual(vsP.xlim, [1, 4]);
  assert.deepEqual(vsP.ylim, [1e-17, 10]);
  assert.equal(axisOf(scene, "condition").autoscaled, true);
  assert.equal(axisOf(scene, "condition").yscale, "log");
});

test("adaptive: 网格 0–1、指示子 1e-17–1、历史 1e-17–10 全部固定", async () => {
  const scene = await sceneOf("adaptive_mesh");
  const mesh = axisOf(scene, "mesh");
  assert.deepEqual(mesh.xlim, [0, 1]);
  assert.deepEqual(mesh.ylim, [0, 1]);
  const ind = axisOf(scene, "indicator");
  assert.deepEqual(ind.xlim, [0, 1]);
  assert.ok(ind.ylim[0] === 1e-17 && ind.ylim[1] >= 1);
  assert.equal(ind.yscale, "log");
  const hist = axisOf(scene, "history");
  assert.equal(hist.xscale, "log");
  assert.deepEqual(hist.ylim, [1e-17, 10]);
  for (const ax of scene.axes) assert.equal(ax.autoscaled, false, `${ax.id} 必须固定窗口`);
});

// --------------------------------------------------------------------------- //
// 4. 掩码 / 哨兵 / 地板 —— 本章的核心
// --------------------------------------------------------------------------- //

test("assembly: entry_pending=1 的格子落在 clipBelow 之下（灰=无数据），非未定型格子不受影响", async () => {
  const frame = await makeFrame(ASM_DEF);
  const scene = models.assembly_animation(frame, OPTS);
  const heat = layersOn(scene, "matrix").find((l) => l.kind === "heat");
  assert.ok(heat && heat.kind === "heat");
  assert.ok(heat.clipBelow !== undefined);
  const pending = arrOf(frame, "entry_pending");
  const values = heat.values as Float64Array;
  for (let i = 0; i < pending.length; i += 1) {
    if ((pending[i] as number) > 0.5) {
      assert.ok((values[i] as number) < (heat.clipBelow as number), `未定型格 ${i} 没有落到 clipBelow 之下`);
    } else {
      assert.ok((values[i] as number) >= (heat.clipBelow as number), `已定型格 ${i} 被错误藏掉`);
    }
  }
});

test("assembly: pivot_unmeasured=1 的 1e300 显示上限永不进图层", async () => {
  const scene = await sceneOf("assembly_animation");
  for (const s of lineSeriesOn(scene, "residuals")) {
    for (const v of Array.from(s.y)) assert.ok((v as number) < 1e299, "主元比的显示上限被画成了数据");
  }
});

test("solve_deform: scan_at_floor=1 的节点误差点不进曲线；地板线单独存在", async () => {
  const scene = await sceneOf("solve_deform");
  const scan = lineSeriesOn(scene, "scan");
  const data = scan.filter((s) => s.y.length > 2);
  for (const s of data) {
    for (let i = 0; i < s.y.length; i += 1) {
      assert.ok(!((s.y[i] as number) === 1e-17 && (s.x[i] as number) === 0.5), "地板掩码点 0.5 被画成了数据");
    }
  }
  assert.ok(scan.some((s) => s.y.length === 2 && s.y[0] === 1e-17 && s.y[1] === 1e-17), "缺 1e-17 地板线");
});

test("hp: condition_unmeasured=1 的 1e300 不进图层；error_at_floor=1 的 h 点被切开", async () => {
  const scene = await sceneOf("hp_convergence", { arrays: { error_at_floor: [0, 1, 0] } });
  for (const s of lineSeriesOn(scene, "condition")) {
    for (const v of Array.from(s.y)) assert.ok((v as number) < 1e299, "未测条件数的显示上限被画成了数据");
  }
  const hLines = lineSeriesOn(scene, "ladder").filter((s) => s.color === "#2b6cb0");
  assert.ok(hLines.length > 0);
  for (const s of hLines) {
    for (const x of Array.from(s.x)) assert.ok((x as number) !== 7, "error_at_floor=1 的 h 点 x=7 仍被画进了曲线");
  }
  assert.ok(lineSeriesOn(scene, "ladder").some((s) => s.y.length === 2 && s.y[0] === 1e-17), "缺地板线");
});

test("adaptive: 效率指数占位值 1e-12 永不进任何图层；地板层不画", async () => {
  const scene = await sceneOf("adaptive_mesh", { arrays: { error_at_floor: [0, 0, 1, 0] } });
  for (const v of allLayerNumbers(scene)) {
    assert.ok(!(Math.abs(v - 1e-12) <= 1e-24), "效率指数的占位值 1e-12 被画进了图层");
  }
  const aLines = lineSeriesOn(scene, "history").filter((s) => s.color === "#2b6cb0");
  for (const s of aLines) {
    for (const x of Array.from(s.x)) assert.ok((x as number) !== 9, "error_at_floor=1 的层 dof=9 仍被画进了曲线");
  }
});

test("solve_deform: 奇异帧不画位移/扫描，也不写假的 0 误差", async () => {
  const scene = await sceneOf("solve_deform", {
    arrays: { singular_mask: new Array(5).fill(1) },
    meta: { singular: true },
  });
  assert.deepEqual(validateScene(scene), []);
  assert.match(textsOf(scene), /奇异/);
  assert.match(badgesOf(scene), /奇异/);
  // scan 轴只剩地板线（2 点）；没有 5 点数据曲线
  const scan = lineSeriesOn(scene, "scan");
  assert.equal(scan.filter((s) => s.y.length > 2).length, 0, "奇异帧的扫描曲线应被全部排除");
  // 误差副轴只剩地板线
  const err = lineSeriesOn(scene, "error");
  assert.equal(err.filter((s) => s.y.length > 2).length, 0, "奇异帧的节点误差应被全部排除");
  assert.match(textsOf(scene), /误差范数不可用/);
});

// --------------------------------------------------------------------------- //
// 5. 本章特有的不变式
// --------------------------------------------------------------------------- //

test("assembly: double 路线的 K_defect 非零格用红色散点标出", async () => {
  const defect = new Array(25).fill(0);
  for (const i of [0, 6, 12, 18, 24]) defect[i] = 8;
  const scene = await sceneOf("assembly_animation", {
    arrays: { K_defect: defect },
    meta: { route_defect: 16 },
  });
  const scatter = layersOn(scene, "matrix").filter((l) => l.kind === "scatter");
  assert.ok(scatter.length >= 1, "defect 非零时应有散点层");
  assert.ok(scatter.some((l) => l.kind === "scatter" && l.x.length === 5), "应有 5 个对角线偏差格");
  assert.match(textsOf(scene), /路线偏差|route_defect/);
});

test("solve_deform: membrane 用单元边线画变形网格，并保留未变形参考虚线", async () => {
  const scene = await buildScene(MB_DEF);
  const series = lineSeriesOn(scene, "deformed");
  assert.ok(series.length >= 8, `4 个单元 + 参考网格应有至少 8 条折线，得到 ${series.length}`);
  assert.ok(series.some((s) => s.y.length === 5), "四边形单元边线应闭合（5 点）");
  assert.match(textsOf(scene), /变形网格/);
  // 2D 膜没有 h2，必须用 h1 报误差（不写 H²=?）
  assert.match(textsOf(scene), /H¹=/);
});

test("hp: kink 的同一个 p 阶梯在 vs_p 上画出半对数/双对数两条拟合线", async () => {
  const scene = await sceneOf("hp_convergence", { meta: { solution_index: 1, exponential_better: false } });
  const vsP = lineSeriesOn(scene, "vs_p");
  assert.ok(vsP.some((s) => s.y.length === 4 && Math.abs((s.y[0] as number) - 2.2) < 1e-12), "缺 p 误差曲线");
  assert.ok(hasSeries(vsP, [1, 2, 3, 4], [2.2, 0.3, 0.04, 0.006]), "缺半对数拟合线");
  assert.ok(hasSeries(vsP, [1, 2, 3, 4], [2.2, 0.03, 0.002, 0.0004]), "缺双对数拟合线");
  assert.match(textsOf(scene), /半对数并不更优|代数收敛/);
});

test("adaptive: 当前层竖线落在 dof_adaptive 上，历史轴是同自由度而非层数", async () => {
  const scene = await sceneOf("adaptive_mesh");
  const vertical = lineSeriesOn(scene, "history").filter(
    (s) => s.x.length === 2 && s.x[0] === s.x[1] && s.y[0] !== s.y[1],
  );
  assert.ok(vertical.some((s) => s.x[0] === 9), "缺当前层 dof=9 的竖线");
  assert.match(textsOf(scene), /同自由度/);
  assert.match(textsOf(scene), /log η 与 log 真误差/);
});

test("overflow：扫描误差越窗必须由 applyOverflow 点名（不靠裁剪藏起来）", async () => {
  const bad = [1e-2, 5e-3, 1e3, 5e-3, 1e-2];
  const scene = await sceneOf("solve_deform", { arrays: { error_energy_scan: bad } });
  assert.deepEqual(validateScene(scene), []);
  assert.equal(scene.overflow.flag, true);
  assert.ok(scene.overflow.note.trim().length > 0);
  assert.match(badgesOf(scene), /溢出/);
});

test("完整帧不降级，source/axisMode/status 齐全；critical 只在 manifest 有临界参数时出现", async () => {
  const expectCritical: Record<string, boolean> = {
    assembly_animation: false,
    solve_deform: true,
    hp_convergence: false,
    adaptive_mesh: true,
  };
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz);
    assert.deepEqual(validateScene(scene), [], `${viz} 完整帧必须合法`);
    assert.equal(degradedText(scene), "", `${viz} 完整帧不该出现降级角标`);
    const kinds = new Set(scene.badges.map((b) => b.kind));
    for (const kind of ["source", "autoscale", "status"] as const) {
      assert.ok(kinds.has(kind), `${viz} 缺少 ${kind} 角标`);
    }
    assert.equal(kinds.has("critical"), expectCritical[viz], `${viz} 的 critical 角标不符合 manifest`);
    if (scene.overflow.flag) assert.ok(scene.overflow.note.trim().length > 0);
  }
});

test("critical 角标点名当前参数（solve 的 x0、adaptive 的 θ）", async () => {
  const sd = await sceneOf("solve_deform");
  assert.match(badgesOf(sd), /x0 = 0\.5/);
  const ad = await sceneOf("adaptive_mesh");
  assert.match(badgesOf(ad), /θ = 0\.5/);
  const mem = await buildScene(MB_DEF);
  assert.match(badgesOf(mem), /distortion = 0/);
});

// --------------------------------------------------------------------------- //
// 6. 降级点名 / 缺必需数组抛错 / 窄画布
// --------------------------------------------------------------------------- //

test("缺可选掩码/meta → 不抛错，但 degraded 角标点名", async () => {
  const cases: Array<[string, Override, RegExp]> = [
    ["assembly_animation", { drop: ["pivot_unmeasured"] }, /pivot_unmeasured/],
    ["assembly_animation", { dropMeta: ["t"] }, /t\/t_actual|t_actual/],
    ["solve_deform", { drop: ["scan_at_floor"] }, /scan_at_floor/],
    ["solve_deform", { dropMeta: ["load_factor"] }, /load_factor/],
    ["hp_convergence", { drop: ["error_at_floor"] }, /error_at_floor/],
    ["hp_convergence", { drop: ["fit_p_exp"] }, /fit_p_exp/],
    ["adaptive_mesh", { drop: ["error_at_floor"] }, /error_at_floor/],
    ["adaptive_mesh", { dropMeta: ["level"] }, /level/],
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
    ["assembly_animation", "K_partial"],
    ["solve_deform", "error_energy_scan"],
    ["hp_convergence", "error_p"],
    ["adaptive_mesh", "element_eta"],
  ] as const) {
    await assert.rejects(async () => sceneOf(viz, { drop: [name] }), ModelInputError);
  }
});

test("pitfalls 为空 → 抛 ModelInputError（规则 7：model 不编造）", async () => {
  for (const viz of Object.keys(DEFS)) {
    const def: FrameDef = { ...(DEFS[viz] as FrameDef), pitfalls: "" };
    await assert.rejects(async () => buildScene(def), ModelInputError);
  }
});

test("窄画布 480×520：四个 viz 都合法（说明文字放弃时点名降级）", async () => {
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz, {}, { width: 480, height: 520 });
    assert.deepEqual(validateScene(scene), [], `${viz} 在 480×520 下仍要合法`);
    for (const ax of scene.axes) assert.ok(ax.rect[2] > 0 && ax.rect[3] > 0, `${viz}.${ax.id} 宽高必须为正`);
    assert.match(degradedText(scene), /窄画布|画布过矮|画布过窄/);
  }
});

// --------------------------------------------------------------------------- //
// 7. Scene JSON 不含非有限数（NaN/∞ 一律不进图层与 meta）
// --------------------------------------------------------------------------- //

test("四个 viz 的 Scene 里没有非有限数", async () => {
  for (const viz of Object.keys(DEFS)) {
    for (const opts of [{}, { width: 480, height: 520 }]) {
      const scene = await sceneOf(viz, {}, opts);
      assert.equal(countNonFiniteDeep(scene), 0, `${viz} 的 Scene 含非有限数（${JSON.stringify(opts)}）`);
      const json = sceneToJson(scene);
      assert.ok(json.length > 100, "Scene JSON 不应为空");
    }
  }
});

test("adaptive_mesh: 一致加密提前停止时数组可以短于 level_axis（不拒整张图）", async () => {
  const scene = await sceneOf("adaptive_mesh", {
    arrays: {
      dof_uniform: [3, 5, 9],
      error_uniform: [0.9, 0.55, 0.25],
    },
    meta: { levels: 3, uniform_truncated: true },
  });
  assert.deepEqual(validateScene(scene), []);
  const hist = lineSeriesOn(scene, "history");
  assert.ok(hist.some((s) => s.x.length === 4), "自适应仍有 4 层");
  assert.ok(hist.some((s) => s.x.length === 3), "一致加密提前停止，保留 3 层");
});
