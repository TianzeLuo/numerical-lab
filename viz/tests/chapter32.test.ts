/**
 * 3.2 章（稳定性与刚性方程）model 层测试 —— **自足**：不依赖 Python，也不依赖 `data/frames`
 * 是否存在。夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，字段集合（数组名/dtype/
 * shape/unit/label、meta/strings/critical 键）逐字照抄 `server/chapters/3.2.py` 的真实帧；
 * 真实帧在的话再对拍一遍（见文件末尾，缺帧时整组跳过）。
 *
 * 本文件专门守 3.2 最容易画反的几条：
 * - **真极点不是数值**：BDF-k 在 z=α_k 上是 |R|=+∞，帧用 `pole_fill=1e6` 占位并配 `*_pole`
 *   掩码。极点格必须**单独标记**，而且绝不能拿 1e6 去标"最不稳定"（这一章最贵的一课）；
 * - **`abs_R_current` 的哨兵同样是显示上限**：`pole_current` 为真时，绝不许把占位值当成
 *   "稳定/不稳定"的判据；`point_visible=False` 时主图上根本不该有落点；
 * - **楔形与家族都是顺序拼接的折线**：`wedge_x/y` 是**一条**四点折线（0→上边→回 0→下边），
 *   `family_x/y` 是 k=1…6 六条边界首尾相接——必须按 `family_k` 切段，任何图层都不许出现 NaN；
 * - **requiredAux 真的画出来了**（按**数值**认，不按 label——`LineSeries.label` 会被渲染器
 *   画在曲线末端，本章一律不写长 label）；
 * - **没有时间轴的 viz 不许碰 t**（`stability_surface` 的 `time.mode="none"`）；
 * - **降级不静默**：meta 缺键 / 值是 null → 少画什么就写明什么（degraded 角标点名）。
 */

import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import { bdfAngleModel } from "../chapters/3.2/bdf_angle.model.js";
import { explicitVsImplicitModel } from "../chapters/3.2/explicit_vs_implicit.model.js";
import { manifest, modelFor, models } from "../chapters/3.2/index.js";
import { stabilityRegionModel } from "../chapters/3.2/stability_region.model.js";
import { stabilitySurfaceModel } from "../chapters/3.2/stability_surface.model.js";
import { stiffDilemmaModel } from "../chapters/3.2/stiff_dilemma.model.js";

const PITFALLS =
  "它会怎样骗你：把极点格子上的占位值（1e6）当成「最不稳定」的真值——那里根本没有有限值；" +
  "把「当前误差还挺小」当成「能走下去」——|R|>1 的放大迟早把快模态抬到 O(1)；" +
  "把刚性当成「解很陡」——本问题里解是 cos t，|y''|≤1，陡的是那个看不见的模态。";

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
    chapter: "3.2",
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
      // 覆盖后长度与声明形状不符时，形状跟着数据走（这样才能测"模型自己发现形状不对"）。
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
// 五个 viz 的夹具（字段集合照抄真实帧）
// --------------------------------------------------------------------------- //

const lin = (n: number, a: number, b: number): Float64Array =>
  Float64Array.from({ length: n }, (_, i) => (n <= 1 ? a : a + ((b - a) * i) / (n - 1)));

/** 复平面网格：行主序 `(ny, nx)`，`|R|` = |1+z| 型（在 z=-1 附近衰减、右半平面放大）。 */
function grid(nx: number, ny: number, x0: number, x1: number, y0: number, y1: number): { absR: Float64Array; re: Float64Array; im: Float64Array } {
  const absR = new Float64Array(nx * ny);
  const re = new Float64Array(nx * ny);
  const im = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j += 1) {
    for (let i = 0; i < nx; i += 1) {
      const x = x0 + ((x1 - x0) * i) / (nx - 1);
      const y = y0 + ((y1 - y0) * j) / (ny - 1);
      const k = j * nx + i;
      re[k] = x;
      im[k] = y;
      // 圆 |z+1|=1 内外：域内给出 <1 的值，域外给出 >1 的值。
      absR[k] = Math.abs(1 + 1.2 * x + 0.2 * y);
    }
  }
  return { absR, re, im };
}

const SR_NX = 26;
const SR_NY = 21;
const srGrid = grid(SR_NX, SR_NY, -8, 4, -6, 6);
const SR_BOUNDARY_N = 181;
const SR_RAY_N = 121;
const SR_T = 51;
/** 极点格：网格正好压在一个极点格点上（真实帧里 BDF-2 的 z=1.5 就是这样）。 */
const SR_POLE_INDEX = 12 + 10 * SR_NX;

const SR_DEF: FrameDef = {
  viz: "stability_region",
  arrays: [
    ["abs_R", srGrid.absR, "|R(λh)|（帧网格；极点位置是显示上限占位）", [SR_NY, SR_NX]],
    ["abs_R_pole", srGrid.absR.map((_, i) => (i === SR_POLE_INDEX ? 1 : 0)), "极点掩码（1=该格 |R|=+∞）", [SR_NY, SR_NX]],
    ["extent", [-8, 4, -6, 6], "[x0,x1,y0,y1]"],
    ["levels", [0.25, 0.5, 0.75, 1, 1.5, 2], "|R| 等值线"],
    ["boundary_x", lin(SR_BOUNDARY_N, -2, 0.2), "精确参数化边界 Re z"],
    ["boundary_y", lin(SR_BOUNDARY_N, -1, 1), "精确参数化边界 Im z"],
    ["residual", lin(SR_BOUNDARY_N, 1e-16, 4e-16), "边界自洽残差 ||R(z_θ)|−1|"],
    ["ray_x", lin(SR_RAY_N, 0, -8), "扫描射线 Re z"],
    ["ray_y", lin(SR_RAY_N, 0, 0), "扫描射线 Im z"],
    ["ray_inside", lin(SR_RAY_N, 1, 0), "射线采样点在域内"],
    ["abs_R_ray", lin(SR_RAY_N, 0.2, 3), "射线上的 |R|"],
    ["abs_R_ray_pole", new Float64Array(SR_RAY_N), "射线上的极点掩码"],
    ["z_point_x", [-6], "当前 λh 的 Re"],
    ["z_point_y", [12], "当前 λh 的 Im"],
    ["t_solution", lin(SR_T, 0, 6.2831853), "解的时间轴"],
    ["y1_num", lin(SR_T, 1, 1.5), "数值解 y1（已按显示上限截断）"],
    ["y2_num", lin(SR_T, 0, 0.25), "数值解 y2"],
    ["y1_exact", lin(SR_T, 1, 0.9), "精确解 y1"],
    ["y2_exact", lin(SR_T, 0, 0.1), "精确解 y2"],
    ["y_magnitude", lin(SR_T, 0, 3.5), "max|y|（未截断真值在 meta）"],
    ["error", lin(SR_T, 1e-16, 2e-2), "数值解与精确解之差"],
    ["error_mark", [2e-2], "当前时刻的误差"],
  ],
  meta: {
    a_stable: false, abs_R_current: 1165.0, abs_R_max_finite: 306.113, blowup: true, display_ceiling: 1e6,
    dtrial: 1, error_final: 2e-2, error_max: 2e-2, h: 1, h_exit: 0.978, h_exit_unbounded: 2, lam_im: 12,
    lam_re: -6, log10_h: 0, max_abs_y: 1.571e9, n_f_evals: 3142, n_lu: 0, n_pole_cells: 1, n_pole_ray: 0,
    n_steps: 5, n_steps_planned: 5, overflow: true, point_visible: false, pole_current: false, pole_fill: 1e6,
    residual_max: 4e-16, roundoff_floor: 1.1102230246251565e-16, stable: false, t_blowup: 4, t_end: 6.2831853,
    z_im: 12, z_re: -6, t: 0,
  },
  strings: {
    display_policy: "|R| 超过 1e6 后按显示上限截断；meta.abs_R_max_finite 是窗口内未截断的最大有限值",
    method: "rk4", method_name: "经典 RK4",
    nonfinite_policy: "|R|=+∞ 的极点位置用 pole_fill（=1e+06）占位并配 *_pole 掩码数组；折线不用 NaN 分隔。",
    note: "λ=-6+12i：|R(λh)|=1165>1（域外），数值解在 t=4 处发散。",
    pole_fill: "1e+06", pole_note: "极点格子上没有有限值：占位值只是显示上限，不要当数值读。",
    status: "blowup", data_source: "backend",
    params_json: '{"lam_re":-6.0,"lam_im":12.0,"method":"rk4"}',
  },
  critical: { lam_re: 0 },
};

const SD_EXPLICIT_N = 301;
const SD_IMPLICIT_N = 11;
const SD_EXACT_N = 71;
const SD_T_END = 6.2831853;
const SD_DEF: FrameDef = {
  viz: "stiff_dilemma",
  arrays: [
    ["t_exact", lin(SD_EXACT_N, 0, SD_T_END), "精确解时间轴"],
    ["y_exact", lin(SD_EXACT_N, 1, 1), "cos t"],
    ["t_explicit", lin(SD_EXPLICIT_N, 0, SD_T_END), "显式方法时间轴"],
    ["y_explicit", Float64Array.from(lin(SD_EXPLICIT_N, 1, 1), (v, i) => v + 1e-3 * Math.sin(i)), "显式解 y"],
    ["t_implicit", lin(SD_IMPLICIT_N, 0, SD_T_END), "隐式方法时间轴"],
    ["y_implicit", Float64Array.from(lin(SD_IMPLICIT_N, 1, 1), (v, i) => v - 1e-4 * Math.cos(i)), "隐式解 y"],
    ["error", lin(SD_EXPLICIT_N, 1.11e-16, 2e-6), "显式方法误差 |y_num−cos t|"],
    ["error_mark", [2e-6], "当前时刻的显式误差"],
    ["error_implicit", lin(SD_IMPLICIT_N, 1.11e-16, 3.29e-7), "隐式方法误差"],
    ["step_marks_explicit", lin(SD_EXPLICIT_N, 0, SD_T_END), "显式方法每步的落点（时间）"],
    ["step_marks_implicit", lin(SD_IMPLICIT_N, 0, SD_T_END), "隐式方法每步的落点（时间）"],
    ["curvature", lin(SD_EXACT_N, 1, 1), "|y''|（快模态系数为 0，解很平缓）"],
    ["metric_x", [0, 1, 2, 3], "代价指标下标"],
    ["metric_explicit", [300, 300, 0, 0], "显式：步数/求值/LU/Newton"],
    ["metric_implicit", [10, 20, 10, 10], "隐式：步数/求值/LU/Newton"],
    ["step_x", [0, 1], "两种方法的步长"],
    ["step_values", [0.002, 0.06283185], "步长值"],
    ["step_budget_mark", [300], "步数预算标记"],
  ],
  meta: {
    blowup: false, curvature_max: 1, display_ceiling: 1e6, error_explicit_final: 2e-6, error_explicit_max: 2e-6,
    error_implicit_final: 3.29e-7, f_evals_explicit: 300, f_evals_implicit: 20, h_ceiling: 0.002,
    h_ceiling_theory: 0.002, h_implicit: 0.06283185, lam: -1000, lu_implicit: 10, max_abs_y_explicit: 1.0000013,
    n_steps_explicit: 300, n_steps_explicit_planned: 300, n_steps_implicit: 10, newton_implicit: 10, overflow: false,
    roundoff_floor: 1.1102230246251565e-16, stiffness_ratio: 1000, steps_ratio: 30, t: 0, t_now: 0, t_blowup: 0,
    t_end: SD_T_END, truncated: false,
  },
  strings: {
    data_source: "backend", display_policy: "|y| 超过 1e6 后截断显示；meta.max_abs_y_explicit 是未截断真值",
    explicit_method: "forward_euler", implicit_method: "trapezoidal",
    nonfinite_policy: "本帧不含任何非有限值；折线不用 NaN 分隔。",
    note: "λ=-1000：显式 h≤0.002，一个周期 300 步；隐式 h=0.0628，10 步。",
    params_json: '{"stiffness_ratio":1000.0,"implicit_steps":10}',
  },
  critical: { stiffness_ratio: 3183 },
};

const SS_N = 21;
const ssGrid = grid(SS_N, SS_N, -6, 6, -6, 6);
const SS_BOUNDARY_N = 201;
const SS_DEF: FrameDef = {
  viz: "stability_surface",
  arrays: [
    ["abs_R", ssGrid.absR, "|R(z)|（帧网格）", [SS_N, SS_N]],
    ["abs_R_pole", ssGrid.absR.map((_, i) => (i === 5 + 5 * SS_N ? 1 : 0)), "极点掩码", [SS_N, SS_N]],
    ["extent", [-6, 6, -6, 6], "[x0,x1,y0,y1]"],
    ["levels", [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4], "|R| 等值线"],
    ["boundary_x", lin(SS_BOUNDARY_N, -0.1, 6.6), "精确参数化边界 Re z"],
    ["boundary_y", lin(SS_BOUNDARY_N, -3.9, 3.9), "精确参数化边界 Im z"],
    ["residual", lin(SS_BOUNDARY_N, 1e-16, 4e-16), "边界自洽残差"],
    ["error", lin(SS_N * SS_N, 1e-16, 2.2e-16), "两条独立路径之差（逐格）", [SS_N, SS_N]],
    ["error_mark", [2.2e-16], "双路径之差的最大值"],
    ["edge_values", [0.5, 0.6, 0.62, 0.4], "盒边采样点的 |R|"],
    ["edge_pole", [0, 0, 0, 0], "盒边采样点的极点掩码"],
    ["angle_k", [1, 2, 3, 4, 5, 6], "阶数 k"],
    ["angle_measured", [90, 90, 86.03, 73.35, 51.84, 17.84], "各阶实测 A(α) 角（度）"],
    ["angle_theory", [90, 90, 86.03, 73.35, 51.84, 17.84], "各阶理论角（度）"],
  ],
  meta: {
    abs_R_max: 18.49, abs_R_min: 0.258199, angle_deg: 86.03, a_stable: false, edge_max: 0.622839,
    endpoint_finite: false, grid_n: SS_N, half_width: 6, inside_fraction: 0.904225, l_stable: false,
    max_on_imaginary_axis: 1, negative_real_endpoint: 0, n_grid: SS_N * SS_N, n_inside: 400,
    n_pole_cells: 1, n_pole_edge: 0, n_route_mismatch: 1, order_effective: 3, order_param: 3, overflow: true,
    pole_fill: 1e6, residual_max: 4e-16, roundoff_floor: 1.1102230246251565e-16, route_diff_max: 2.22e-16,
    stiff_limit: 0, t: 0, vmax: 4, vmin: 0,
  },
  strings: {
    colour_note: "|R| 直接上色；色标上限会截断，读数才算数。", data_source: "backend",
    method: "bdf", method_name: "BDF",
    nonfinite_policy: "极点位置用 pole_fill（=1e+06）占位并配 *_pole 掩码数组。",
    note: "BDF-3：稳定域缩成贴着实轴的窄带。", pole_fill: "1e+06",
    pole_note: "极点格子（z=11/6）上没有有限值：占位值只是显示上限，不要当数值读。",
    route: "边界走 z(θ)=ρ(e^{iθ})/σ(e^{iθ}) 参数化，网格走逐格 |R| 求值。",
    params_json: '{"method":"bdf","order":3,"half_width":6.0,"grid_n":21}',
  },
  critical: { order: 3 },
};

const EVI_EXPLICIT_N = 301;
const EVI_IMPLICIT_N = 11;
const EVI_T_END = 6.2831853;
const EVI_DEF: FrameDef = {
  viz: "explicit_vs_implicit",
  arrays: [
    ["t_explicit", lin(EVI_EXPLICIT_N, 0, EVI_T_END), "显式方法时间轴"],
    ["y_explicit", lin(EVI_EXPLICIT_N, 1, 1), "显式解"],
    ["t_implicit", lin(EVI_IMPLICIT_N, 0, EVI_T_END), "隐式方法时间轴"],
    ["y_implicit", lin(EVI_IMPLICIT_N, 1, 1), "隐式解"],
    ["t_exact", lin(EVI_EXPLICIT_N, 0, EVI_T_END), "精确解时间轴"],
    ["y_exact", lin(EVI_EXPLICIT_N, 1, 1), "cos t"],
    ["error", lin(EVI_EXPLICIT_N, 1.11e-16, 2e-6), "显式误差"],
    ["error_implicit", lin(EVI_IMPLICIT_N, 1.11e-16, 3.23e-6), "隐式误差"],
    ["error_mark", [2e-6], "当前时刻显式误差"],
    ["error_implicit_mark", [6.34e-7], "当前时刻隐式误差"],
    ["metric_x", [0, 1, 2, 3], "代价指标下标"],
    ["metric_explicit", [300, 300, 0, 0], "显式：步数/求值/LU/Newton"],
    ["metric_implicit", [10, 20, 10, 10], "隐式：步数/求值/LU/Newton"],
    ["step_x", [0, 1], "两种方法的步长"],
    ["step_values", [0.002, 0.19635], "步长值"],
    ["tolerance_mark", [-5], "log10(容差) 在代价梯子上的位置"],
    ["cost_x", lin(9, -9.105, -4.261), "代价—容差曲线的 log10(容差)"],
    ["cost_explicit", new Float64Array(9).fill(300), "显式代价（与容差无关）"],
    ["cost_implicit", [24, 48, 96, 194, 386, 770, 1536, 3074, 6144], "隐式代价"],
    ["cost_cross_x", [-8.517], "两条代价曲线的交叉点 log10(容差)"],
    ["cost_cross_y", [300], "交叉点代价"],
  ],
  meta: {
    blowup: false, cost_cross_found: true, cost_cross_log10_tol: -8.516676, cost_explicit_const: 300,
    cost_implicit_at_tol: 6144, display_ceiling: 1e6, error_explicit: 2e-6, error_explicit_span: 2e-6,
    error_implicit: 6.34e-7, error_implicit_span: 3.23e-6, f_evals_explicit: 300, f_evals_implicit: 20,
    h_explicit: 0.002, h_explicit_theory: 0.002, h_implicit: 0.19635, h_ratio: 98.17,
    implicit_error_at_planned_step: 3.23e-6, ladder_k_max: 11, ladder_k_min: 3, lam: -1000, lu_implicit: 10,
    max_abs_y_explicit: 1.0000013, n_steps_explicit: 300, n_steps_implicit: 10, newton_implicit: 10,
    overflow: false, roundoff_floor: 1.1102230246251565e-16, startup_steps: 0, steps_ratio: 30,
    stiffness_ratio: 1000, t: 0, t_end: EVI_T_END, t_now: 0, t_blowup: 0, tol: 1e-5, tolerance_fallback: false,
    truncated: false,
  },
  strings: {
    crossover_note: "两条代价曲线在 log10(容差)=-8.52 附近交叉：容差更松时隐式便宜。", data_source: "backend",
    display_policy: "|y| 超过 1e6 后截断显示；meta.max_abs_y_explicit 是未截断真值",
    explicit_method: "forward_euler", implicit_method: "trapezoidal",
    nonfinite_policy: "本帧不含任何非有限值；折线不用 NaN 分隔。",
    note: "显式 h=0.002×300 步，误差与容差无关；隐式 h=0.196×10 步 + 10 次 LU。",
    params_json: '{"stiffness_ratio":1000.0,"tol":1e-05}',
  },
  critical: { stiffness_ratio: 3183, tol: 1e-6 },
};

const BA_THETA_N = 121;
const BA_RAY_N = 61;
const BA_K = [1, 2, 3, 4, 5, 6];
/** 家族：k=1…6 各 41 点，首尾相接（**不含** NaN 分隔符）。 */
const BA_FAMILY_PER_K = 41;
const BA_DEF: FrameDef = {
  viz: "bdf_angle",
  arrays: [
    ["boundary_x", lin(BA_THETA_N, -0.0833, 6.67), "当前 k 的稳定域边界 Re z"],
    ["boundary_y", lin(BA_THETA_N, -3.96, 3.96), "当前 k 的稳定域边界 Im z"],
    [
      "family_x",
      Float64Array.from({ length: BA_FAMILY_PER_K * 6 }, (_, i) => -(i % BA_FAMILY_PER_K) * (1 + Math.floor(i / BA_FAMILY_PER_K)) * 0.2),
      "k=1…6 的边界家族 Re z（按 family_k 分段）",
    ],
    [
      "family_y",
      Float64Array.from({ length: BA_FAMILY_PER_K * 6 }, (_, i) => ((i % BA_FAMILY_PER_K) - 20) * 0.1 * (1 + Math.floor(i / BA_FAMILY_PER_K))),
      "k=1…6 的边界家族 Im z（按 family_k 分段）",
    ],
    ["family_k", Float64Array.from({ length: BA_FAMILY_PER_K * 6 }, (_, i) => Math.floor(i / BA_FAMILY_PER_K) + 1), "该段属于哪一阶"],
    ["wedge_x", [0, -6, 0, -6], "A(α) 楔形：0→上边→回 0→下边（顺序拼接，无 NaN）"],
    ["wedge_y", [0, 0, 0, 0], "A(α) 楔形（α=0 时退化成负实轴）"],
    ["ray_x", lin(BA_RAY_N, 0, -6), "测试射线 Re z"],
    ["ray_y", lin(BA_RAY_N, 0, 0), "测试射线 Im z"],
    ["error", lin(BA_RAY_N, 1.11e-16, 4.44e-16), "射线上的超界量 max(0,|R|−1)"],
    ["residual", lin(BA_THETA_N, 1.11e-16, 4.44e-16), "边界自洽残差"],
    ["angle_k", BA_K, "阶数 k"],
    ["angle_measured", [90, 90, 86.032367, 73.352, 51.84, 17.839778], "各阶实测 A(α) 角（度）"],
    ["angle_theory", [90, 90, 86.03, 73.35, 51.84, 17.84], "各阶理论角（度）"],
    ["angle_mark_x", [3], "当前 k"],
    ["angle_mark_y", [86.032367], "当前 k 的实测角"],
  ],
  meta: {
    a_stable: false, alpha_probe_deg: 0, alpha_within_measured: true, angle_current_deg: 86.032367,
    angle_error_deg: 0.002367, angle_k1: 90, angle_k3: 86.032367, angle_k6: 17.839778, angle_theory_deg: 86.03,
    angles_monotone: true,
    family_size: BA_FAMILY_PER_K * 6, k: 3, k_max: 6, n_theta_boundary: BA_THETA_N, overflow: false,
    residual_max: 4.44e-16, show_family: true, t: 0, violation_floor: 1.1102230246251565e-16,
    violation_max: 4.44e-16, wedge_violated: false, z_axis_limit: 6,
  },
  strings: {
    angle_meaning: "A(α) 角是**含**在稳定域里的最大楔形半角，不是「边界在左半平面的角度」。",
    data_source: "backend", display_policy: "超界量为 0 只说明这一条射线没出界，不等于 A-稳定。",
    nonfinite_policy: "本帧 |R| 与超界量全部有限；极点 z=α_k 是正实数，永远不在楔形里。",
    note: "BDF-3：A(α) 角实测 86.0324°（理论 86.03°），当前测试角 α=0.0° → 楔形仍在域内。",
    params_json: '{"k":3,"show_family":true}',
    wedge_note: "楔形是四点顺序拼接的折线（0→上边→回 0→下边），不含 NaN 分隔符。",
  },
  critical: { k: 3 },
};

// --------------------------------------------------------------------------- //
// 公共工具
// --------------------------------------------------------------------------- //

const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };
const DEFS: Record<string, FrameDef> = {
  stability_region: SR_DEF,
  stiff_dilemma: SD_DEF,
  stability_surface: SS_DEF,
  explicit_vs_implicit: EVI_DEF,
  bdf_angle: BA_DEF,
};
const MODELS: Record<string, typeof stabilityRegionModel> = {
  stability_region: stabilityRegionModel,
  stiff_dilemma: stiffDilemmaModel,
  stability_surface: stabilitySurfaceModel,
  explicit_vs_implicit: explicitVsImplicitModel,
  bdf_angle: bdfAngleModel,
};
/** manifest 的 `requiredAux` 名 →（画在哪条轴，取帧里哪个数组）。 */
const AUX_WHERE: Record<string, [string, string]> = {
  stability_region: ["aux", "error"],
  stiff_dilemma: ["aux", "error"],
  stability_surface: ["aux", "residual"],
  explicit_vs_implicit: ["aux", "error"],
  bdf_angle: ["aux", "error"],
};

const sceneOf = async (viz: string, over: Override = {}, opts: Partial<ModelOptions> = {}): Promise<Scene> =>
  MODELS[viz](await makeFrame(tweak(DEFS[viz] as FrameDef, over)), { ...OPTS, ...opts });
const specOf = (viz: string) => manifest.vizzes.find((v) => v.id === viz) ?? assert.fail(`manifest 缺 ${viz}`);
const textsOf = (s: Scene): string => s.texts.map((t) => t.text).join("\n");
const badgesOf = (s: Scene): string => s.badges.map((b) => `${b.kind}:${b.text}`).join("\n");
const badgeKinds = (s: Scene): string[] => s.badges.map((b) => b.kind);
const axisOf = (s: Scene, id: string): Axis => s.axes.find((a) => a.id === id) ?? assert.fail(`缺少轴 ${id}`);
const layersOn = (s: Scene, id: string): Layer[] => s.layers.filter((l) => l.axes === id);
/** 某个轴上所有被画出来的数值（line/scatter/bars/heat/contour 通吃）。 */
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
/** 某个轴上画出来的所有折线点（含 x），用来检查"没有 NaN"这类结构性约定。 */
const seriesOn = (s: Scene, id: string): { x: number[]; y: number[] }[] => {
  const out: { x: number[]; y: number[] }[] = [];
  for (const l of layersOn(s, id)) {
    if (l.kind === "line") for (const ser of l.series) out.push({ x: Array.from(ser.x, Number), y: Array.from(ser.y, Number) });
  }
  return out;
};
/** 帧里的数组 → `number[]`（`TypedArray` 是联合类型，`Array.from` 会挑不出重载）。 */
const numsOf = (frame: Frame, name: string): number[] => Array.from(frame.array(name) as ArrayLike<number>, Number);
const close = (a: number, b: number, tol = 1e-9): boolean => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// --------------------------------------------------------------------------- //
// 装配层与公共契约
// --------------------------------------------------------------------------- //

test("3.2：manifest 的五个 viz 都有 model，映射表没有多余键，未知 id 返回 null", () => {
  assert.deepEqual(Object.keys(models).sort(), manifest.vizzes.map((v) => v.id).sort());
  assert.equal(manifest.chapter, "3.2");
  for (const v of manifest.vizzes) assert.equal(modelFor(v.id), models[v.id]);
  assert.equal(modelFor("no_such_viz"), null);
});

test("3.2：五个 viz 都能产出合法 Scene（validateScene 零 issue），主/副轴按 manifest 声明", async () => {
  for (const viz of manifest.vizzes.map((v) => v.id)) {
    const scene = await sceneOf(viz);
    assert.deepEqual(validateScene(scene), [], `${viz} 的 Scene 有校验问题`);
    assert.equal(scene.chapter, "3.2");
    assert.equal(scene.viz, viz);
    assert.ok(scene.pitfalls.length >= 10);
    // manifest 声明的轴 id 必须都在（模型可以多给，不能少给）。
    for (const spec of specOf(viz).axes) {
      const axis = axisOf(scene, spec.id);
      assert.equal(axis.autoscaled, spec.autoscaled, `${viz}/${spec.id} 的 autoscaled 与 manifest 不一致`);
      if (spec.scale === "log") assert.equal(axis.yscale, "log", `${viz}/${spec.id} 应为 log 轴`);
      assert.ok(scene.layers.some((l) => l.axes === spec.id), `${viz}/${spec.id} 上一个图层都没有`);
    }
  }
});

test("3.2：requiredAux 声明的数组真的画在了对应轴上（按数值认，不按 label）", async () => {
  for (const spec of manifest.vizzes) {
    for (const name of spec.requiredAux) {
      const where = AUX_WHERE[spec.id];
      assert.ok(where, `测试缺 ${spec.id} 的 requiredAux 映射`);
      if (name === "residual" && spec.id === "stability_surface") {
        // 该 viz 有两个 requiredAux（error=双路径之差、residual=边界残差），都必须在 aux 上。
        const scene = await sceneOf(spec.id);
        const frame = await makeFrame(DEFS[spec.id] as FrameDef);
        const drawn = drawnOn(scene, "aux");
        assert.ok(
          numsOf(frame, "residual").every((v) => drawn.some((d) => close(d, v, 1e-9))),
          "stability_surface 的 residual 没有画在 aux 上",
        );
        assert.ok(
          numsOf(frame, "error").every((v) => drawn.some((d) => close(d, v, 1e-9))),
          "stability_surface 的 error（双路径之差）没有画在 aux 上",
        );
        continue;
      }
      const scene = await sceneOf(spec.id, {}, {});
      const frame = await makeFrame(DEFS[spec.id] as FrameDef);
      const values = numsOf(frame, where[1]);
      const drawn = drawnOn(scene, where[0]);
      // 时间序列可能被裁剪（越界/触地板），所以只要求"至少有 2 个样本原样出现"。
      const hit = values.filter((v) => Number.isFinite(v) && drawn.some((d) => close(d, v, 1e-9))).length;
      assert.ok(hit >= 2, `${spec.id} 的 requiredAux ${name}（${where[1]}）没有画在 ${where[0]} 上（命中 ${hit}）`);
    }
  }
});

test("3.2：critical 角标带帧里的临界值；两个临界值的 viz 两枚都在", async () => {
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz);
    const crit = scene.badges.filter((b) => b.kind === "critical");
    assert.ok(crit.length >= 1, `${viz} 没有 critical 角标`);
    assert.ok(badgeKinds(scene).includes("source"), `${viz} 没有 source 角标`);
    assert.ok(scene.badges.some((b) => b.kind === "autoscale"), `${viz} 没有自适应角标（规则 4）`);
  }
  const evi = await sceneOf("explicit_vs_implicit");
  assert.equal(evi.badges.filter((b) => b.kind === "critical").length, 2, "tol 与刚性比两枚临界角标应同时在");
  assert.match(badgesOf(evi), /0\.000001|1e-0?6/i);
  assert.match(badgesOf(evi), /3183/);
});

// --------------------------------------------------------------------------- //
// 本章的七条防误导
// --------------------------------------------------------------------------- //

test("stability_region：极点格的占位值不被当数值——单独标记 + 文本写明「显示上限」", async () => {
  const scene = await sceneOf("stability_region");
  const heat = layersOn(scene, "main").find((l) => l.kind === "heat") ?? assert.fail("没有热力图");
  assert.equal(heat.kind, "heat");
  // 色标上端是**窗口内最大有限值**（或 |R|=1），绝不是 pole_fill=1e6。
  assert.ok(heat.vmax < 1e3, `热力图 vmax=${heat.vmax} 太接近占位值 1e6`);
  // 极点格取上端饱和，另有单独颜色的散点标记。
  const poleMarks = layersOn(scene, "main").filter((l) => l.kind === "scatter" && (l.label ?? "").includes("极点"));
  assert.equal(poleMarks.length, 1, "极点格应有且仅有一层单独标记");
  const poleMark = poleMarks[0];
  assert.ok(poleMark && poleMark.kind === "scatter");
  assert.notEqual(poleMark.color, heat.cmap, "极点标记的颜色不能是热力图的饱和色");
  assert.match(textsOf(scene), /显示上限|不是真值|不是数值/);
  assert.match(textsOf(scene), /极点/);
});

test("stability_region：点不在窗口内（point_visible=False）时不画落点，且写明原因", async () => {
  const scene = await sceneOf("stability_region");
  const marks = layersOn(scene, "main").filter((l) => l.kind === "scatter" && (l.label ?? "").includes("当前 λh"));
  assert.equal(marks.length, 0, "point_visible=False 时不该有当前 λh 落点");
  // 这条说明写在**正文**里（角标会被 clipBadgeText 截断，写尾部等于没写）。
  assert.match(textsOf(scene), /不在固定窗口|point_visible/);
});

test("stability_region：pole_current=True 时绝不把哨兵值当判据（文本明说 + 不画落点）", async () => {
  const scene = await sceneOf("stability_region", {
    meta: { pole_current: true, point_visible: false, abs_R_current: 1e6, z_re: 1.5, z_im: 0, n_pole_cells: 1 },
  });
  assert.match(textsOf(scene), /真极点|显示上限/);
  assert.match(badgesOf(scene), /显示上限不是数值|真极点/);
  assert.equal(layersOn(scene, "main").filter((l) => l.kind === "scatter" && (l.label ?? "").includes("当前 λh")).length, 0);
});

test("stability_surface：极点格单独标记，掩码点数与 meta 不一致时点名降级", async () => {
  const scene = await sceneOf("stability_surface");
  const poleMarks = layersOn(scene, "main").filter((l) => l.kind === "scatter" && (l.label ?? "").includes("极点"));
  assert.equal(poleMarks.length, 1);
  assert.equal(Array.from(poleMarks[0]?.kind === "scatter" ? poleMarks[0].x : []).length, 1, "夹具里只有 1 个极点格");
  const heat = layersOn(scene, "main").find((l) => l.kind === "heat") ?? assert.fail("没有热力图");
  assert.equal(heat.kind, "heat");
  assert.ok(heat.vmax <= 20, `色标上端应取数组里最大的有限值（约 18.49），得到 ${heat.vmax}`);
  assert.match(textsOf(scene), /显示上限/);

  const bad = await sceneOf("stability_surface", { meta: { n_pole_cells: 7 } });
  assert.match(badgesOf(bad), /n_pole_cells=7/);
});

test("stability_surface：没有时间轴（time.mode=none）——不读 opts.t，也不画时间游标", async () => {
  const a = await sceneOf("stability_surface", {}, { t: 0 });
  const b = await sceneOf("stability_surface", {}, { t: 12345 });
  assert.equal(a.t, b.t, "scene.t 只能来自帧 meta");
  assert.deepEqual(
    a.layers.map((l) => l.kind + ":" + l.axes),
    b.layers.map((l) => l.kind + ":" + l.axes),
  );
  assert.equal(specOf("stability_surface").time.mode, "none");
});

test("bdf_angle：楔形是**一条**四点折线（无 NaN），家族按 family_k 分成六段", async () => {
  const scene = await sceneOf("bdf_angle");
  const lines = seriesOn(scene, "main");
  const wedge = lines.filter((s) => s.x.length === 4);
  assert.equal(wedge.length, 1, "楔形必须是恰好一条四点折线（0→上边→回 0→下边）");
  assert.deepEqual(wedge[0]?.x, [0, -6, 0, -6]);
  assert.deepEqual(wedge[0]?.y, [0, 0, 0, 0]);
  for (const s of lines) {
    assert.ok(s.x.every(Number.isFinite) && s.y.every(Number.isFinite), "任何折线都不许含 NaN（NaN 会让渲染器静默丢段）");
  }
  // 家族：六段各自独立（每次 clip 之后仍是若干段），且当前 k=3 被高亮。
  const familyPieces = layersOn(scene, "main").filter(
    (l) => l.kind === "line" && (l.series[0]?.label ?? "").startsWith("BDF-"),
  );
  assert.ok(familyPieces.length >= 6, `家族应有 ≥6 段，得到 ${familyPieces.length}`);
  const highlighted = familyPieces.filter((l) => l.kind === "line" && (l.series[0]?.width ?? 0) === 2.4);
  assert.ok(highlighted.length >= 1, "当前 k 的边界必须更粗");
});

test("bdf_angle：射线第一次越界点来自 error>0 的下标（不重算 |R|）", async () => {
  const scene = await sceneOf("bdf_angle", { arrays: { error: lin(BA_RAY_N, 0, 0).map((_, i) => (i < 10 ? 0 : 0.05)) } });
  const mark = layersOn(scene, "main").find((l) => l.kind === "scatter" && (l.label ?? "").includes("第一次越界"));
  assert.ok(mark, "缺第一次越界标记");
  if (mark?.kind === "scatter") {
    assert.equal(Number(mark.x[0]), Number(lin(BA_RAY_N, 0, -6)[10]), "越界点应取 error 第一次 >0 的那个下标");
  }
});

test("bdf_angle：超界量全为 0（射线没出界）时没有越界标记，且 status 说清「≠ A-稳定」", async () => {
  const scene = await sceneOf("bdf_angle", { arrays: { error: new Float64Array(BA_RAY_N) } });
  assert.equal(layersOn(scene, "main").filter((l) => l.kind === "scatter" && (l.label ?? "").includes("第一次越界")).length, 0);
  assert.match(badgesOf(scene), /A-稳定|不是 A-稳定/);
  assert.match(textsOf(scene), /不等于 A-稳定|不代表|只说明/);
});

test("stiff_dilemma / explicit_vs_implicit：固定纵轴 ±2，越界样本截断并报账", async () => {
  for (const viz of ["stiff_dilemma", "explicit_vs_implicit"]) {
    const scene = await sceneOf(viz);
    const main = axisOf(scene, "main");
    assert.deepEqual(main.ylim, [-2, 2]);
    assert.equal(main.autoscaled, false);
    assert.match(textsOf(scene), /±2/);
  }
  const bad = await sceneOf("stiff_dilemma", { arrays: { y_explicit: lin(SD_EXPLICIT_N, 1, 5) } });
  assert.match(badgesOf(bad), /overflow|超出|截断/);
});

test("explicit_vs_implicit：代价柱在对数轴上，恒 0 的柱子（显式 LU/Newton）在文本里交代", async () => {
  const scene = await sceneOf("explicit_vs_implicit");
  const cost = axisOf(scene, "cost");
  assert.equal(cost.yscale, "log");
  const bars = layersOn(scene, "cost").filter((l) => l.kind === "bars");
  assert.equal(bars.length, 2, "显式/隐式各一组柱子");
  for (const b of bars) {
    if (b.kind !== "bars") continue;
    assert.equal(b.baseline, cost.ylim[0], "对数轴上柱子的基线必须是轴底（不能是 0）");
  }
  assert.match(textsOf(scene), /0 = 该方法没有这一步|没有 LU\/Newton/);
  assert.match(badgesOf(scene), /高度是 0|高度为 0/);
});

test("3.2：降级不静默——缺 meta/strings 键时点名，且不崩", async () => {
  const cases: [string, Override, RegExp][] = [
    ["stability_region", { meta: { abs_R_max_finite: null, pole_fill: null }, dropStrings: ["pole_note", "note"] }, /⚠ 降级/],
    ["stiff_dilemma", { dropMeta: ["h_ceiling", "steps_ratio"], dropStrings: ["note"] }, /⚠ 降级/],
    ["stability_surface", { dropMeta: ["route_diff_max", "n_route_mismatch"], dropStrings: ["note", "pole_note"] }, /⚠ 降级/],
    ["explicit_vs_implicit", { dropMeta: ["tol"], dropStrings: ["note"] }, /⚠ 降级/],
    ["bdf_angle", { meta: { alpha_probe_deg: null, show_family: 0 }, dropStrings: ["note"] }, /⚠ 降级/],
  ];
  for (const [viz, over, re] of cases) {
    const scene = await sceneOf(viz, over);
    assert.deepEqual(validateScene(scene), [], `${viz} 降级帧的 Scene 不合法`);
    assert.match(badgesOf(scene), re, `${viz} 降级帧没有 degraded 角标`);
  }
});

test("3.2：结构性错误直接抛 ModelInputError（不画半张图）", async () => {
  // abs_R 的形状与长度不符（稳定性曲面）。
  await assert.rejects(
    async () => sceneOf("stability_surface", { arrays: { abs_R: new Float64Array(10) } }),
    (e: unknown) => e instanceof ModelInputError,
  );
  // 楔形长度必须为 4。
  await assert.rejects(
    async () => sceneOf("bdf_angle", { arrays: { wedge_x: [0, -6, 0] } }),
    (e: unknown) => e instanceof ModelInputError,
  );
  // 折线长度不一致。
  await assert.rejects(
    async () => sceneOf("stiff_dilemma", { arrays: { y_explicit: new Float64Array(5) } }),
    (e: unknown) => e instanceof ModelInputError,
  );
  // pitfalls 太短。
  await assert.rejects(
    async () => sceneOf("bdf_angle", { pitfalls: "太短" }),
    (e: unknown) => e instanceof ModelInputError,
  );
});

test("3.2：窄画布退化成单列也合法（640/700/712/960 宽；文本不许压到装饰带或互相压住）", async () => {
  for (const viz of Object.keys(DEFS)) {
    const frame = await makeFrame(DEFS[viz] as FrameDef);
    for (const [w, h] of [
      [640, 600],
      [700, 600],
      [712, 600],
      [960, 600],
    ] as const) {
      const scene = MODELS[viz](frame, { width: w, height: h, t: 0, params: {} });
      assert.deepEqual(validateScene(scene), [], `${viz} 在 ${w}×${h} 下不合法`);
      assert.ok(scene.layers.length > 0);
    }
  }
});

test("3.2：模型是纯函数——同一帧两次调用逐字节相同，且不改帧", async () => {
  for (const viz of Object.keys(DEFS)) {
    const frame = await makeFrame(DEFS[viz] as FrameDef);
    const snapshot = (): string =>
      JSON.stringify(frame.meta) + "|" + (frame.has("boundary_x") ? numsOf(frame, "boundary_x") : numsOf(frame, "error")).join(",");
    const before = snapshot();
    const a = MODELS[viz](frame, OPTS);
    const b = MODELS[viz](frame, OPTS);
    assert.equal(JSON.stringify(a), JSON.stringify(b), `${viz} 两次调用结果不同（模型不是纯函数？）`);
    const after = snapshot();
    assert.equal(after, before, `${viz} 的调用改动了帧`);
  }
});

// --------------------------------------------------------------------------- //
// 真实帧对拍（缺 data/frames 时整组跳过）
// --------------------------------------------------------------------------- //

const FRAME_ROOT = "data/frames/3.2";
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
    for (const spec of frame.specs) {
      const row = (DEFS[viz] as FrameDef).arrays.find(([n]) => n === spec.name);
      if (!row) continue;
      assert.equal(spec.shape.length, row[3]?.length ?? 1, `${viz}.${spec.name} 的维度数与真实帧不一致`);
    }
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

real("真实帧：3.2 的每一帧都能画出合法 Scene（含 BDF 的极点帧与 α=0/90 两端）", () => {
  let n = 0;
  for (const viz of Object.keys(MODELS)) {
    for (const file of realFrames(viz)) {
      const frame = realFrame(viz, file);
      const t = Number(frame.meta["t"] ?? 0);
      const scene = MODELS[viz](frame, { width: 960, height: 600, t, params: {} });
      assert.deepEqual(validateScene(scene), [], `${viz}/${file}`);
      if (viz === "stability_surface") assert.equal(scene.t, Number(frame.meta["t"] ?? 0));
      n += 1;
    }
  }
  assert.ok(n > 0, "一帧都没跑到：data/frames/3.2 是空的？");
});

real("真实帧：极点帧（BDF-2 的 z=1.5）与发散帧都必须把「没有有限值」写出来", () => {
  for (const viz of ["stability_region", "stability_surface"]) {
    for (const file of realFrames(viz)) {
      const frame = realFrame(viz, file);
      const nPole = Number(frame.meta["n_pole_cells"] ?? 0);
      if (!(nPole > 0)) continue;
      const scene = MODELS[viz](frame, { width: 960, height: 600, t: Number(frame.meta["t"] ?? 0), params: {} });
      assert.match(textsOf(scene), /极点/, `${viz}/${file} 有极点却没有说明`);
      assert.match(textsOf(scene) + badgesOf(scene), /显示上限|不是数值/, `${viz}/${file} 没有写明占位值是显示上限`);
      const heat = scene.layers.find((l) => l.kind === "heat");
      if (heat?.kind === "heat") assert.ok(heat.vmax < 1e3, `${viz}/${file} 把显示上限当成了色标上端`);
    }
  }
});
