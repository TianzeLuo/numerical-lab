/**
 * 2.2 章 model 层的测试（`control_points` / `smoothness_c0c1c2` / `bspline_local` / `spline_vs_poly`）。
 *
 * 夹具怎么造：**手搓帧走完整编解码**（`new Frame(...)` → `encodeNlf` → `decodeNlf`），
 * 字段集合（数组名 / dtype / shape / meta 键 / strings 键 / critical 键 / pitfalls 非空）
 * 照 `server/chapters/2.2.py` 的默认帧抄，数值是**测试用合成值**、不是物理计算。
 * 这样 `npm test`（门禁 G5）在没有 Python、没有 `data/frames` 的机器上自足：
 * 不依赖 `.cache/`、不依赖预计算产物，也不会因为没有帧而静默跳过。
 *
 * 降级行为（本文件逐条守）
 * ----------------------
 * | 缺失 | 行为 |
 * |---|---|
 * | 被 model 引用的数组（`x_grid`/`curve`/`error`…） | **抛 `ModelInputError`**，不画半张图 |
 * | 同名数组长度不一致 | **抛 `ModelInputError`**（错位的图比不画更坏） |
 * | `frame.pitfalls` 为空 | **抛 `ModelInputError`**（规则 7，model 不自行编造陷阱） |
 * | `meta.t` / `meta.t_actual` | 回落请求 `t` + 降级角标点名（`scrub` 语义） |
 * | `meta.tension`（唯一的临界值参数） | **不画**临界角标 + 降级角标点名（不猜） |
 * | `strings.note` / `strings.comb_name` | 文本里写明"帧未给出"，不自行编造结论 |
 * | `meta.error_spline_max` 等标量 | 显示 `—`（`fmt(undefined)`），不当 0 用 |
 * | 多项式的 `meta.overflow=1` | 溢出徽标 + `scene.overflow.flag`，主轴**不**自动放大 |
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type Dtype } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Layer, type Scene } from "../shared/src/scene.js";
import { bsplineLocalModel } from "../chapters/2.2/bspline_local.model.js";
import { controlPointsModel } from "../chapters/2.2/control_points.model.js";
import { manifest, modelFor, models } from "../chapters/2.2/index.js";
import { smoothnessModel } from "../chapters/2.2/smoothness_c0c1c2.model.js";
import { splineVsPolyModel } from "../chapters/2.2/spline_vs_poly.model.js";

// --------------------------------------------------------------------------- //
// 夹具
// --------------------------------------------------------------------------- //
function fail(message: string): never {
  throw new Error(message);
}

const VIZ_CONTROL = "control_points";
const VIZ_SMOOTH = "smoothness_c0c1c2";
const VIZ_BSPLINE = "bspline_local";
const VIZ_VSPOLY = "spline_vs_poly";

/** 每个 viz 在 manifest 里的 spec（轴/参数/临界值的真源）。 */
const SPEC = {
  control: manifest.vizzes.find((v) => v.id === VIZ_CONTROL) ?? fail(`manifest 缺少 ${VIZ_CONTROL}`),
  smooth: manifest.vizzes.find((v) => v.id === VIZ_SMOOTH) ?? fail(`manifest 缺少 ${VIZ_SMOOTH}`),
  bspline: manifest.vizzes.find((v) => v.id === VIZ_BSPLINE) ?? fail(`manifest 缺少 ${VIZ_BSPLINE}`),
  vspoly: manifest.vizzes.find((v) => v.id === VIZ_VSPOLY) ?? fail(`manifest 缺少 ${VIZ_VSPOLY}`),
} as const;

/** 陷阱文案：真实帧里的这一段是后端给的（`frame.pitfalls`），模型只负责显示。 */
const PITFALLS =
  "曲率梳的 local 尺度会把每条齿归一成一样长，「哪里更弯」就读不出来了；tension 看着像平滑旋钮，" +
  "其实把弯矩乘 (1-tau)，节点处的一阶导跳变从 u 量级涨到折角量级。";

type Row = ArrayLike<number>;
interface Arr {
  name: string;
  values: Row;
  dtype?: Dtype;
  cols?: number;
  unit?: string;
  label?: string;
}

interface FrameOpts {
  meta?: Record<string, number | boolean | null>;
  strings?: Record<string, string>;
  critical?: Record<string, number | number[]>;
  dropMeta?: string[];
  dropStrings?: string[];
  dropArrays?: string[];
  pitfalls?: string;
}

async function makeFrame(chapter: string, viz: string, arrays: Arr[], opts: FrameOpts = {}): Promise<Frame> {
  const dropArrays = new Set(opts.dropArrays ?? []);
  const kept = arrays.filter((a) => !dropArrays.has(a.name));
  const specs: ArraySpec[] = [];
  const map = new Map<string, Float64Array>();
  let offset = 0;
  for (const a of kept) {
    const values = a.values instanceof Float64Array ? a.values : Float64Array.from(a.values as ArrayLike<number>);
    const cols = a.cols ?? 1;
    const shape = cols > 1 ? [values.length / cols, cols] : [values.length];
    const nbytes = values.length * 8;
    specs.push({
      name: a.name,
      dtype: a.dtype ?? "f8",
      shape,
      order: "C",
      offset,
      nbytes,
      unit: a.unit ?? "",
      label: a.label ?? a.name,
    });
    offset += nbytes;
    map.set(a.name, values);
  }
  const meta: Record<string, number | boolean | null> = { ...(opts.meta ?? {}) };
  for (const key of opts.dropMeta ?? []) delete meta[key];
  const strings: Record<string, string> = { data_source: "backend", params_json: "{}", ...(opts.strings ?? {}) };
  for (const key of opts.dropStrings ?? []) delete strings[key];
  const frame = new Frame({
    chapter,
    viz,
    meta,
    strings,
    critical: opts.critical ?? {},
    pitfalls: opts.pitfalls ?? PITFALLS,
    specs,
    arrays: map,
  });
  const bytes = await encodeNlf(frame);
  return decodeNlf(bytes);
}

function range(n: number, f: (i: number) => number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) out[i] = f(i);
  return out;
}

const X_GRID = range(301, (i) => (5 * i) / 300);
const CURVE = Float64Array.from(X_GRID, (x) => 0.4 + 0.5 * x - 0.18 * x * x + 0.02 * x * x * x);
const SLOPE = Float64Array.from(X_GRID, (x) => 0.5 - 0.36 * x + 0.06 * x * x);
const CURVATURE = Float64Array.from(X_GRID, (x) => -0.36 + 0.12 * x);
const NODE_X = Float64Array.from([0, 1, 2, 3, 4, 5]);
const NODE_Y = Float64Array.from(NODE_X, (x) => 0.4 + 0.5 * x - 0.18 * x * x + 0.02 * x * x * x);
const COMB_BASE = Float64Array.from([...X_GRID].flatMap((x, i) => [x, CURVE[i] as number]));
const COMB_TIP = Float64Array.from([...X_GRID].flatMap((x, i) => [x, (CURVE[i] as number) + 0.05 * (CURVATURE[i] as number)]));
const COMB_TIP_LOCAL = Float64Array.from([...X_GRID].flatMap((x, i) => [x, (CURVE[i] as number) + 0.4]));

/** `control_points` 的默认帧：字段集合与真实帧一致（22 个数组 + 20 个 meta 键）。 */
function controlArrays(): Arr[] {
  return [
    { name: "x_grid", values: X_GRID, unit: "x", label: "采样横坐标" },
    { name: "curve", values: CURVE, label: "样条曲线 S(x)" },
    { name: "slope", values: SLOPE, label: "S'(x)" },
    { name: "curvature", values: CURVATURE, label: "曲率 kappa(x)" },
    { name: "node_x", values: NODE_X, label: "数据点 x" },
    { name: "node_y", values: NODE_Y, label: "数据点 y" },
    { name: "comb_base", values: COMB_BASE, cols: 2, label: "曲率梳根部 (n,2)" },
    { name: "comb_tip", values: COMB_TIP, cols: 2, label: "曲率梳齿尖（当前尺度）(n,2)" },
    { name: "comb_tip_local", values: COMB_TIP_LOCAL, cols: 2, label: "曲率梳齿尖（local 归一化）(n,2)" },
    { name: "jump_nodes", values: Float64Array.from([1, 2, 3, 4]), label: "内点 x" },
    { name: "jump_d1", values: Float64Array.from([1.1e-16, 6.7e-16, 2.2e-16, 1.1e-16]), label: "|S'(x+) − S'(x−)|" },
    { name: "jump_d2", values: Float64Array.from([1.1e-16, 1.1e-16, 2.2e-16, 1.1e-16]), label: "|S''(x+) − S''(x−)|" },
    { name: "residual", values: Float64Array.from([1.1e-16, 6.7e-16, 2.2e-16, 1.1e-16]), label: "C1 残差" },
    { name: "error", values: Float64Array.from([1.1e-16, 3.9e-16, 2.2e-16, 1.1e-16, 3.3e-16, 1.1e-16]), label: "节点处插值残差" },
    { name: "probe", values: Float64Array.from([0, 0.4]), label: "探针位置 (x,y)" },
    { name: "probe_tangent", values: Float64Array.from([0.332, 0.943]), label: "探针处单位切向" },
    { name: "system_diag", values: Float64Array.from([2, 2, 2, 2]), label: "三对角系统对角元" },
    { name: "system_rhs", values: Float64Array.from([-13.8, 2.4, -6.6, 13.2]), label: "三对角系统右端项" },
  ];
}

function controlMeta(extra: Record<string, number | null> = {}): Record<string, number | boolean | null> {
  return {
    t: 0,
    t_actual: 0,
    n_nodes: 6,
    n_intervals: 5,
    system_size: 4,
    bc_index: 0,
    tension: 0,
    comb_scale_index: 0,
    energy: 176.50220095693783,
    energy_polyline: 0,
    energy_pchip: 222.52,
    energy_ratio_pchip: 1.2607,
    energy_over_polyline: null,
    curvature_max: 12.01529149097391,
    curvature_at_probe: 0.4,
    jump_d1_max: 6.661338147750939e-16,
    jump_d2_max: 1.1102230246251565e-16,
    nodal_error_max: 3.885780586188048e-16,
    residual: 6.661338147750939e-16,
    overflow: false,
    ...extra,
  };
}

const CONTROL_STRINGS = {
  bc_name: "自然（M0=Mn=0）",
  comb_name: "统一尺度（长度可比）",
  note: "自然（M0=Mn=0）：|dS'| ~ 6.66e-16（机器精度），曲率峰值 12，能量 177",
};

const controlFrame = (o: FrameOpts = {}): Promise<Frame> =>
  makeFrame("2.2", VIZ_CONTROL, controlArrays(), {
    meta: controlMeta(),
    strings: CONTROL_STRINGS,
    critical: { tension: 1.0 },
    ...o,
  });

// —— smoothness ——
const X_STEP = Float64Array.from([...range(41, (i) => i / 40)].flatMap((x) => [x - 1e-3, x + 1e-3]).filter((x) => x >= 0 && x <= 1));
const Y_LIN = Float64Array.from(X_STEP, (x) => x * x);
const Y_PCHIP = Float64Array.from(X_STEP, (x) => x * x);
const Y_SPLINE = Float64Array.from(X_STEP, (x) => x * x);
const D2_LIN = new Float64Array(X_STEP.length);
const D2_PCHIP_STEP = Float64Array.from(X_STEP, (x) => (x > 0.5 ? 64 : 0));
const D2_SPLINE_STEP = Float64Array.from(X_STEP, () => 2);
const X_NAIVE = range(120, (i) => i / 119);
const D2_PCHIP_NAIVE = Float64Array.from(X_NAIVE, (x) => 64 * x);
const D2_SPLINE_NAIVE = Float64Array.from(X_NAIVE, () => 2);

function smoothArrays(): Arr[] {
  return [
    { name: "x_step", values: X_STEP, label: "节点两侧加密采样" },
    { name: "y_lin", values: Y_LIN, label: "折线" },
    { name: "y_pchip", values: Y_PCHIP, label: "PCHIP" },
    { name: "y_spline", values: Y_SPLINE, label: "样条" },
    { name: "d2_lin", values: D2_LIN, label: "折线 S''" },
    { name: "d2_pchip_step", values: D2_PCHIP_STEP, label: "PCHIP S''（分采）" },
    { name: "d2_spline_step", values: D2_SPLINE_STEP, label: "样条 S''（分采）" },
    { name: "d2_pchip_naive", values: D2_PCHIP_NAIVE, label: "PCHIP S''（等距采样）" },
    { name: "d2_spline_naive", values: D2_SPLINE_NAIVE, label: "样条 S''（等距采样）" },
    { name: "x_naive", values: X_NAIVE, label: "等距采样网格" },
    { name: "node_x", values: Float64Array.from([0, 0.25, 0.5, 0.75, 1]), label: "节点" },
    { name: "jump_nodes", values: Float64Array.from([0.25, 0.5, 0.75]), label: "内点" },
    { name: "jump_d1_lin", values: Float64Array.from([2.67, 2.67, 2.67]), label: "折线 |dS'|" },
    { name: "jump_d2_pchip", values: Float64Array.from([64, 64, 64]), label: "PCHIP |dS''|" },
    { name: "jump_d2_spline", values: Float64Array.from([1.1e-16, 1.1e-16, 1.1e-16]), label: "样条 |dS''|" },
    { name: "residual", values: Float64Array.from([64, 64, 64]), label: "PCHIP |dS''|（要求）" },
    { name: "error", values: Float64Array.from([1.1e-16, 1.1e-16, 1.1e-16]), label: "节点残差" },
  ];
}

const smoothFrame = (o: FrameOpts = {}): Promise<Frame> =>
  makeFrame("2.2", VIZ_SMOOTH, smoothArrays(), {
    meta: {
      t: 0,
      t_actual: 0,
      n: 9,
      eps: 0.03,
      eps_rel: 0.02,
      nodes_index: 0,
      dataset_index: 2,
      bc_index: 0,
      n_step_samples: 41,
      naive_samples: 120,
      jump_pchip_max: 64,
      jump_spline_max: 1.11e-16,
      jump_lin_slope_max: 2.67,
      error_spline_max: 1.1e-16,
      error_pchip_max: 1.1e-16,
      error_lin_max: 1.1e-16,
      energy_lin: 0,
      energy_pchip: 311.099,
      energy_spline: 239.477,
      residual: 64,
      overflow: false,
    },
    strings: {
      dataset_name: "台阶数据（阶跃）",
      bc_name: "自然（M0=Mn=0）",
      note: "n=9：样条 |dS''| 1.1e-16（C²），PCHIP 64（C¹），折线 d2 恒 0 但 dS' 跳 2.67",
    },
    ...o,
  });

// —— bspline ——
const GRID = range(201, (i) => i / 200);
const CTRL_X = Float64Array.from([0, 1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7, 1]);
const CTRL_Y = Float64Array.from([0, 0.8, -0.4, 0.6, -0.9, 0.3, 0.7, 0]);
const CTRL_Y_MOVED = Float64Array.from(CTRL_Y);
CTRL_Y_MOVED[2] = 0.8;
const BASIS_ROWS = 8;
const BASIS = new Float64Array(GRID.length * BASIS_ROWS);
for (let j = 0; j < GRID.length; j += 1) {
  const u = GRID[j] as number;
  for (let i = 0; i < BASIS_ROWS; i += 1) {
    const lo = i / (BASIS_ROWS + 1);
    const hi = (i + 4) / (BASIS_ROWS + 1);
    BASIS[j * BASIS_ROWS + i] = u >= lo && u <= hi ? Math.max(0, 1 - Math.abs((u - (lo + hi) / 2) / ((hi - lo) / 2))) : 0;
  }
}
const CHANGED = Float64Array.from(GRID, (u) => (u >= 0 && u <= 0.6 ? 0.7158 * Math.sin((Math.PI * u) / 0.6) : 0));

function bsplineArrays(): Arr[] {
  const arrs: Arr[] = [
    { name: "ctrl_x", values: CTRL_X, label: "控制点 x" },
    { name: "ctrl_y", values: CTRL_Y, label: "控制点 y" },
    { name: "ctrl_y_moved", values: CTRL_Y_MOVED, label: "移动后的控制点 y" },
    { name: "knots", values: Float64Array.from([0, 0, 0, 0, 1 / 5, 2 / 5, 3 / 5, 4 / 5, 1, 1, 1, 1]), label: "节点向量" },
    { name: "knots_base", values: Float64Array.from([0, 0, 0, 0, 1 / 5, 2 / 5, 3 / 5, 4 / 5, 1, 1, 1, 1]), label: "节点向量（初始）" },
    { name: "x_grid", values: GRID, label: "参数采样" },
    { name: "curve_orig", values: Float64Array.from(GRID, (u) => 0.3 * Math.sin(2 * Math.PI * u)), label: "当前曲线" },
    { name: "curve_moved", values: Float64Array.from(GRID, (u) => 0.3 * Math.sin(2 * Math.PI * u) + 0.1 * u), label: "移动后曲线" },
    { name: "curve_base", values: Float64Array.from(GRID, (u) => 0.3 * Math.sin(2 * Math.PI * u)), label: "基准曲线" },
    { name: "basis", values: BASIS, cols: BASIS_ROWS, dtype: "f8", label: "基函数矩阵 (k+1, m)" },
    { name: "basis_index", values: Float64Array.from(range(8, (i) => i)), label: "基函数下标" },
    { name: "support_left", values: Float64Array.from(range(8, () => 0)), label: "支撑左端" },
    { name: "support_right", values: Float64Array.from(range(8, (i) => Math.min(1, (i + 4) / 9))), label: "支撑右端" },
    { name: "changed", values: CHANGED, label: "改控制点后的 |ΔC|" },
    { name: "changed_mask", values: Float64Array.from(CHANGED, (v) => (v > 0 ? 1 : 0)), label: "受影响掩码" },
    { name: "error", values: Float64Array.from(GRID, () => 1.11e-16), label: "插入前后逐点差" },
    { name: "residual", values: Float64Array.from(range(8, () => 1.11e-16)), label: "支撑外 max|B|" },
    { name: "affected_span", values: Float64Array.from([0, 0.6]), label: "受影响区间" },
    { name: "insert_marker", values: Float64Array.from([0.4, 0.2]), label: "插入点 (u, C(u))" },
  ];
  return arrs;
}

const bsplineFrame = (o: FrameOpts = {}): Promise<Frame> =>
  makeFrame("2.2", VIZ_BSPLINE, bsplineArrays(), {
    meta: {
      t: 0,
      t_actual: 0,
      degree: 3,
      n_ctrl: 8,
      n_basis: 8,
      n_knots: 12,
      n_knots_base: 12,
      insert_count: 0,
      insert_at: 0.4,
      move_index: 2,
      move_delta: 1.2,
      span_left: 0,
      span_right: 0.6,
      affected_spans: 3,
      affected_spans_max: 4,
      changed_measure: 0.596,
      changed_max: 0.7158,
      invariance_error_max: 1.1102e-16,
      support_violation_max: 1.1102e-16,
      partition_unity_error: 1.1102e-16,
      hull_gap: 0,
      residual: 1.1102e-16,
      error: 1.1102e-16,
      overflow: false,
    },
    strings: {
      degree_name: "3 次（k=3）",
      note: "插入节点前后 |ΔC| ≤ 4.4e-16：曲线不变；只改 ≤k 个控制点。",
    },
    ...o,
  });

// —— spline_vs_poly ——
const XS = range(401, (i) => -1 + (2 * i) / 400);
const Y_EXACT = Float64Array.from(XS, (x) => 1 / (1 + 25 * x * x));
const Y_SPL = Float64Array.from(XS, (x, i) => (Y_EXACT[i] as number) + 0.01 * Math.sin(10 * x));
const Y_POLY = Float64Array.from(XS, (x, i) => (Y_EXACT[i] as number) + 1.2 * Math.sin(9 * x) * x * x);

function vsPolyArrays(): Arr[] {
  return [
    { name: "x_grid", values: XS, label: "采样横坐标" },
    { name: "y_exact", values: Y_EXACT, label: "被插函数 f(x)" },
    { name: "y_spline", values: Y_SPL, label: "样条插值" },
    { name: "y_poly", values: Y_POLY, label: "Lagrange 插值多项式" },
    { name: "node_x", values: Float64Array.from(range(9, (i) => -1 + (2 * i) / 8)), label: "节点" },
    { name: "node_y", values: Float64Array.from(range(9, (i) => 1 / (1 + 25 * (-1 + (2 * i) / 8) ** 2))), label: "节点数据" },
    { name: "error_spline", values: Float64Array.from(XS, () => 1.11e-16), label: "|样条误差|" },
    { name: "error_poly", values: Float64Array.from(XS, (x) => 1.11e-16 + 1.045 * Math.abs(x)), label: "|多项式误差|" },
    { name: "error_floor_line", values: Float64Array.from(XS, () => 1.11e-16), label: "舍入地板参考线" },
    { name: "error", values: Float64Array.from([0.0560736, 1.04517]), label: "两条最大误差" },
    { name: "residual", values: Float64Array.from(range(9, () => 1.11e-16)), label: "多项式在节点处的残差" },
    { name: "n_history", values: Float64Array.from([4, 5, 6, 7, 8, 9]), label: "节点数 n" },
    { name: "error_poly_history", values: Float64Array.from([0.247, 0.4, 0.65, 0.9, 1.0, 1.045]), label: "多项式最大误差随 n" },
    { name: "error_spline_history", values: Float64Array.from([0.701, 0.4, 0.2, 0.12, 0.08, 0.056]), label: "样条最大误差随 n" },
  ];
}

const vsPolyFrame = (o: FrameOpts = {}): Promise<Frame> =>
  makeFrame("2.2", VIZ_VSPOLY, vsPolyArrays(), {
    meta: {
      t: 0,
      t_actual: 0,
      n: 9,
      nodes_index: 0,
      bc_index: 0,
      function_index: 0,
      error_spline: 0.0560736,
      error_poly: 1.04517,
      error_ratio: 18.639,
      error_floor: 1.1102e-16,
      nodal_residual_max: 1.11e-16,
      lebesgue_constant: 4.55,
      poly_max_abs: 1.00005,
      poly_min: -1.00005,
      energy_spline: 239.477,
      energy_poly: 30905.8,
      energy_ratio: 129.06,
      n_history_points: 6,
      residual: 1.11e-16,
      error: 1.04517,
      overflow: false,
    },
    strings: {
      function_name: "Runge 1/(1+25x^2) / 等距节点",
      bc_name: "自然（M0=Mn=0）",
      note: "n=9：样条误差 0.0561，多项式误差 1.045（差 18.6 倍）；Lebesgue 常数 4.55",
    },
    ...o,
  });

// --------------------------------------------------------------------------- //
// 共用断言
// --------------------------------------------------------------------------- //
const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };

function assertValid(scene: Scene): void {
  const issues = validateScene(scene);
  assert.deepEqual(issues, [], `Scene 不合规：${issues.join("；")}`);
}

function lineSeries(scene: Scene, axes: string, index = 0): { x: ArrayLike<number>; y: ArrayLike<number>; label?: string }[] {
  const layers = scene.layers.filter(
    (l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === axes,
  );
  const layer = layers[index];
  assert.ok(layer !== undefined, `轴 ${axes} 上应有第 ${index} 个 line 图层`);
  return layer.series;
}

/**
 * 某个轴上**所有** line 图层的 series 拉平。
 *
 * `lineSeries` 只取第 index 个图层——模型把"基函数"、"曲线"、"控制多边形"分成
 * 不同图层（图层数本身就是要断言的东西），所以"在这条轴上找某条曲线"必须跨图层找。
 */
function allSeries(scene: Scene, axes: string): { x: ArrayLike<number>; y: ArrayLike<number>; label?: string }[] {
  return lineLayers(scene, axes).flatMap((l) => l.series);
}

function lineLayers(scene: Scene, axes: string): { series: { x: ArrayLike<number>; y: ArrayLike<number>; label?: string }[] }[] {
  return scene.layers.filter(
    (l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === axes,
  );
}

function layersOn(scene: Scene, axes: string): Layer[] {
  return scene.layers.filter((l) => l.axes === axes);
}

function textsOf(scene: Scene): string {
  return scene.texts.map((t) => t.text).join("\n");
}

function badgesOf(scene: Scene, kind: string): string[] {
  return scene.badges.filter((b) => b.kind === kind).map((b) => b.text);
}

const CONTROL = await controlFrame();
const SMOOTH = await smoothFrame();
const BSPLINE = await bsplineFrame();
const VSPOLY = await vsPolyFrame();

// --------------------------------------------------------------------------- //
// 1. control_points
// --------------------------------------------------------------------------- //
test("2.2/control_points：默认帧产出合法 Scene，章节/viz/来源逐项对", () => {
  const scene = controlPointsModel(CONTROL, { ...OPTS, params: { tension: 0, n_nodes: 6 } });
  assertValid(scene);
  assert.equal(scene.chapter, "2.2");
  assert.equal(scene.viz, VIZ_CONTROL);
  assert.equal(scene.dataSource, "backend");
  assert.equal(scene.title, SPEC.control.title);
  assert.ok(scene.pitfalls.length > 0, "规则 7：陷阱必须随图出现");
});

test("2.2/control_points：四条轴 —— 主轴固定、跳变/曲率/能量自适应（规则 1/4）", () => {
  const scene = controlPointsModel(CONTROL, OPTS);
  assert.deepEqual(scene.axes.map((a) => a.id), ["main", "jump", "curv", "energy"]);
  const [main, jump, curv, energy] = scene.axes;
  assert.equal(main?.autoscaled, false);
  assert.equal(jump?.autoscaled, true);
  assert.equal(jump?.yscale, "log");
  assert.ok((jump?.ylim[0] as number) > 0, "log 轴下界必须 > 0");
  assert.equal(curv?.autoscaled, true);
  assert.equal(curv?.yscale, "linear", "曲率有正负，log 画不出来");
  assert.equal(energy?.autoscaled, true);
  assert.equal(energy?.yscale, "log");
  // manifest 里逐轴的 autoscaled 声明必须与模型一致
  for (const spec of SPEC.control.axes) {
    const axis = scene.axes.find((a) => a.id === spec.id) ?? fail(`缺少轴 ${spec.id}`);
    assert.equal(axis.autoscaled, spec.autoscaled, `轴 ${spec.id} 的 autoscaled 与 manifest 不一致`);
  }
});

test("2.2/control_points：requiredAux 的每个名字都有可见图层（residual / curvature）", () => {
  assert.deepEqual(SPEC.control.requiredAux, ["residual", "curvature"]);
  const scene = controlPointsModel(CONTROL, OPTS);
  const jump = lineSeries(scene, "jump");
  const residualSeries = jump.find((s) => String(s.label ?? "").includes("residual"));
  assert.ok(residualSeries !== undefined, "residual 必须出现在跳变轴的实线里");
  const curv = lineSeries(scene, "curv");
  assert.ok(
    curv.some((s) => String(s.label ?? "").includes("curvature")),
    "curvature 必须出现在曲率轴上",
  );
});

test("2.2/control_points：临界值参考线与 criticalBadge 出现（tension = 1.0，beyond=above）", () => {
  const atCritical = controlPointsModel(CONTROL, { ...OPTS, params: { tension: 1 } });
  const critical = badgesOf(atCritical, "critical");
  assert.equal(critical.length, 1);
  assert.match(critical[0] as string, /tension|τ/i);
});

test("2.2/control_points：τ=1 帧画折线，τ=0 帧不画（判据取自帧的 |dS'|max）", async () => {
  const flat = controlPointsModel(CONTROL, OPTS);
  const labels0 = flat.layers
    .filter((l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === "main")
    .flatMap((l) => l.series.map((s) => String(s.label ?? "")));
  assert.ok(!labels0.some((t) => t.includes("折线")), "τ=0 时不该出现折线层");

  const kinked = await controlFrame({ meta: controlMeta({ tension: 1, jump_d1_max: 4.6, jump_d2_max: 4.6 }) });
  const scene = controlPointsModel(kinked, { ...OPTS, params: { tension: 1 } });
  assertValid(scene);
  assert.ok(
    layersOn(scene, "main").some(
      (l) => l.kind === "line" && l.series.some((s) => String(s.label ?? "").includes("折线")),
    ),
    "τ=1 且帧里 |dS'| 到折角量级时必须画折线",
  );
  assert.match(textsOf(scene), /折线/);
  assert.match(badgesOf(scene, "status").join("；"), /τ=1/);
});

test("2.2/control_points：comb=local 时画归一化齿尖，并在文本里点明读不出「哪里更弯」", async () => {
  const frame = await controlFrame({ meta: controlMeta({ comb_scale_index: 1 }), strings: { ...CONTROL_STRINGS, comb_name: "逐点归一化（只剩方向）" } });
  const scene = controlPointsModel(frame, { ...OPTS, params: { comb: "local" } });
  assertValid(scene);
  const teeth = lineSeries(scene, "main", 1);
  assert.ok(teeth.length > 0, "梳齿层必须在");
  assert.match(textsOf(scene), /一样长|读不出/);
});

test("2.2/control_points：缺 meta.tension → 不画临界角标 + 降级角标点名", async () => {
  const frame = await controlFrame({ dropMeta: ["tension"] });
  const scene = controlPointsModel(frame, OPTS);
  assertValid(scene);
  assert.equal(badgesOf(scene, "critical").length, 0, "不猜临界值");
  assert.match(badgesOf(scene, "status").join("；"), /缺 tension/);
});

test("2.2/control_points：缺 strings.note / comb_name → 文本写明「未给出」，不编造结论", async () => {
  const frame = await controlFrame({ dropStrings: ["note", "comb_name"] });
  const scene = controlPointsModel(frame, OPTS);
  assertValid(scene);
  assert.match(textsOf(scene), /帧未给出 note|comb=\?|缺 comb_name/);
});

test("2.2/control_points：energy_polyline = 0（折线能量恒 0）→ 画在地板上并写成「不在 H² 里」", () => {
  const scene = controlPointsModel(CONTROL, OPTS);
  assert.match(textsOf(scene), /折线能量恒为 0|不在 H²/);
  assert.match(textsOf(scene), /折线 0/);
});

test("2.2/control_points：缺数组 → 抛 ModelInputError", async () => {
  const missing = await controlFrame({ dropArrays: ["curvature"] });
  assert.throws(() => controlPointsModel(missing, OPTS), ModelInputError);
});

test("2.2/control_points：同名数组长度不一致 → 抛 ModelInputError（错位的图比不画更坏）", () => {
  // 零拷贝的帧数组是只读的，所以这里**造一帧**长度不一致的：curve 少一个点。
  const specs: ArraySpec[] = [
    { name: "x_grid", dtype: "f8", shape: [3], order: "C", offset: 0, nbytes: 24, unit: "", label: "x" },
    { name: "curve", dtype: "f8", shape: [2], order: "C", offset: 24, nbytes: 16, unit: "", label: "S" },
  ];
  const frame = new Frame({
    chapter: "2.2",
    viz: VIZ_CONTROL,
    meta: controlMeta(),
    strings: CONTROL_STRINGS,
    critical: { tension: 1 },
    pitfalls: PITFALLS,
    specs,
    arrays: new Map<string, Float64Array>([
      ["x_grid", Float64Array.from([0, 1, 2])],
      ["curve", Float64Array.from([0, 1])],
    ]),
  });
  assert.throws(() => controlPointsModel(frame, OPTS), ModelInputError);
});

test("2.2/control_points：缺 pitfalls → 抛 ModelInputError（规则 7，模型不编造陷阱）", async () => {
  const noPitfalls = await controlFrame({ pitfalls: "" });
  assert.throws(() => controlPointsModel(noPitfalls, OPTS), ModelInputError);
});

test("2.2/control_points：小画布（单列）也能产出合法 Scene（角标不越界）", () => {
  const scene = controlPointsModel(CONTROL, { ...OPTS, width: 420, height: 620 });
  assertValid(scene);
  assert.ok(scene.axes.every((a) => a.rect[0] >= 0 && a.rect[0] + a.rect[2] <= 420), "轴不越出画布");
});

// --------------------------------------------------------------------------- //
// 2. smoothness_c0c1c2
// --------------------------------------------------------------------------- //
test("2.2/smoothness：默认帧合法；time.mode=none → scene.t 恒为 0，不引用 tMax", () => {
  const scene = smoothnessModel(SMOOTH, OPTS);
  assertValid(scene);
  assert.equal(scene.viz, VIZ_SMOOTH);
  assert.equal(scene.t, 0);
  assert.equal(SPEC.smooth.time.mode, "none");
  assert.ok(!Object.prototype.hasOwnProperty.call(SPEC.smooth.time, "tMax"));
});

test("2.2/smoothness：三条轴都自适应（与 manifest 一致）", () => {
  const scene = smoothnessModel(SMOOTH, OPTS);
  assert.deepEqual(scene.axes.map((a) => a.id), ["main", "d2", "jump"]);
  for (const spec of SPEC.smooth.axes) {
    const axis = scene.axes.find((a) => a.id === spec.id) ?? fail(`缺少轴 ${spec.id}`);
    assert.equal(axis.autoscaled, spec.autoscaled);
    assert.equal(axis.autoscaled, true);
  }
  assert.equal(scene.axes[2]?.yscale, "log");
});

test("2.2/smoothness：requiredAux（residual / error）都有可见图层", () => {
  assert.deepEqual(SPEC.smooth.requiredAux, ["residual", "error"]);
  const scene = smoothnessModel(SMOOTH, OPTS);
  const jump = lineSeries(scene, "jump");
  assert.ok(jump.some((s) => String(s.label ?? "").includes("residual")), "residual 在跳变轴上");
  assert.ok(
    jump.some((s) => String(s.label ?? "").includes("jump_d1_lin") || String(s.label ?? "").includes("dS'")),
    "折线的一阶导跳变也要在（error 的量化）",
  );
});

test("2.2/smoothness：同一帧里同时画「节点两侧」与「等距采样」两条 S''（坑必须可读）", () => {
  const scene = smoothnessModel(SMOOTH, OPTS);
  const series = lineSeries(scene, "d2");
  const step = series.filter((s) => !(s.label ?? "").includes("等距采样"));
  const naive = series.filter((s) => (s.label ?? "").includes("等距采样"));
  assert.ok(step.length >= 2, "节点两侧分采的曲线");
  assert.ok(naive.length >= 2, "等距采样对照曲线（PCHIP + 样条）");
  assert.match(textsOf(scene), /跳变连成了斜坡|等距采样/);
  assert.match(textsOf(scene), /S''≡0|连 C¹ 都不是/);
});

test("2.2/smoothness：跳变量级与 u/2 地板参考线同时出现（log 轴上的判据）", () => {
  const scene = smoothnessModel(SMOOTH, OPTS);
  const jump = lineSeries(scene, "jump");
  assert.equal(jump.length, 3, "折线 dS' / PCHIP dS'' / 样条 dS''");
  // 参考线由 `hLine` 单独成层，所以这里必须跨图层找（`lineSeries` 只取第 0 层）。
  assert.ok(
    allSeries(scene, "jump").some((s) => (s.label ?? "").includes("u/2")),
    "u/2 地板参考线必须在跳变轴上（C² 的判据）",
  );
  assert.match(textsOf(scene), /u\/2/);
});

test("2.2/smoothness：缺 meta.nodes_index → 不猜节点分布 + 降级点名", async () => {
  const frame = await smoothFrame({ dropMeta: ["nodes_index"] });
  const scene = smoothnessModel(frame, OPTS);
  assertValid(scene);
  assert.match(badgesOf(scene, "status").join("；") + textsOf(scene), /缺 nodes_index|节点分布/);
});

test("2.2/smoothness：缺 jump 数组 → 抛 ModelInputError（不画半张图）", async () => {
  const frame = await smoothFrame({ dropArrays: ["jump_d2_pchip"] });
  assert.throws(() => smoothnessModel(frame, OPTS), ModelInputError);
});

// --------------------------------------------------------------------------- //
// 3. bspline_local
// --------------------------------------------------------------------------- //
test("2.2/bspline_local：默认帧合法；主轴固定、三条副轴自适应", () => {
  const scene = bsplineLocalModel(BSPLINE, OPTS);
  assertValid(scene);
  assert.deepEqual(scene.axes.map((a) => a.id), ["main", "invariance", "support", "changed"]);
  const [main, invariance, support, changed] = scene.axes;
  assert.equal(main?.autoscaled, false);
  assert.equal(invariance?.autoscaled, true);
  assert.equal(invariance?.yscale, "log");
  assert.equal(support?.autoscaled, true);
  assert.equal(support?.yscale, "log");
  assert.equal(changed?.autoscaled, true);
  assert.equal(changed?.yscale, "linear");
  for (const spec of SPEC.bspline.axes) {
    const axis = scene.axes.find((a) => a.id === spec.id) ?? fail(`缺少轴 ${spec.id}`);
    assert.equal(axis.autoscaled, spec.autoscaled, `轴 ${spec.id} 与 manifest 不一致`);
  }
});

test("2.2/bspline_local：requiredAux（error / residual）都有可见图层", () => {
  assert.deepEqual(SPEC.bspline.requiredAux, ["error", "residual"]);
  const scene = bsplineLocalModel(BSPLINE, OPTS);
  assert.ok(
    lineSeries(scene, "invariance").some((s) => (s.label ?? "").includes("error")),
    "插入前后的逐点差（error）在 invariance 轴上",
  );
  assert.ok(
    lineSeries(scene, "support").some((s) => (s.label ?? "").includes("residual")),
    "支撑外的基函数值（residual）在 support 轴上",
  );
});

test("2.2/bspline_local：控制多边形与曲线是两套图层（控制点不在曲线上）", () => {
  const scene = bsplineLocalModel(BSPLINE, OPTS);
  const series = allSeries(scene, "main");
  assert.ok(series.some((s) => (s.label ?? "").includes("控制多边形")), "控制多边形必须单独画");
  assert.ok(series.some((s) => (s.label ?? "").includes("curve_orig")), "曲线单独画");
  assert.ok(series.some((s) => (s.label ?? "").includes("curve_moved")), "移动后的曲线单独画");
  // "控制多边形"必须与曲线**不同图层**：同一个图层里画，两条线的采样点数会被迫一致。
  assert.ok(
    lineLayers(scene, "main").length >= 2,
    "曲线与控制多边形要分成不同图层（采样点数不同）",
  );
  const pts = layersOn(scene, "main").filter((l): l is Extract<Layer, { kind: "scatter" }> => l.kind === "scatter");
  assert.ok(pts.length >= 1, "被移动的控制点要有散点标记");
});

test("2.2/bspline_local：受影响区间（affected_span）画成带 + 两条竖边界", () => {
  const scene = bsplineLocalModel(BSPLINE, OPTS);
  const band = layersOn(scene, "main").find((l) => l.kind === "band");
  assert.ok(band !== undefined, "受影响参数区间要有 band 层");
  const verticals = allSeries(scene, "main").filter((s) => {
    const x = s.x as ArrayLike<number>;
    const y = s.y as ArrayLike<number>;
    return x.length === 2 && x[0] === x[1] && y[0] !== y[1];
  });
  assert.ok(verticals.length >= 2, "两条支撑边界（t_i 与 t_{i+k+1}）");
  assert.match(textsOf(scene), /受影响/);
});

test("2.2/bspline_local：插入次数被夹住时在文本里如实说明（scrub 语义）", async () => {
  const frame = await bsplineFrame({ meta: { ...(await metaOf(BSPLINE)), insert_count: 4, t: 4 } });
  const scene = bsplineLocalModel(frame, { ...OPTS, t: 7 });
  assertValid(scene);
  assert.match(textsOf(scene), /超出|夹/, "越界的 t 必须写出来，不假装 t 就是次数");
});

test("2.2/bspline_local：缺 meta.move_index / note → 降级但不崩", async () => {
  const frame = await bsplineFrame({ dropMeta: ["move_index", "move_delta"], dropStrings: ["note"] });
  const scene = bsplineLocalModel(frame, OPTS);
  assertValid(scene);
  assert.ok(scene.layers.length > 0);
});

test("2.2/bspline_local：basis 与 (n_basis, x_grid) 不自洽时降级（不画错位的基函数）", async () => {
  const frame = await bsplineFrame({ meta: { ...(await metaOf(BSPLINE)), n_basis: 99 } });
  const scene = bsplineLocalModel(frame, OPTS);
  assertValid(scene);
  assert.match(badgesOf(scene, "status").join("；"), /不自洽|画不出来/);
});

// --------------------------------------------------------------------------- //
// 4. spline_vs_poly
// --------------------------------------------------------------------------- //
test("2.2/spline_vs_poly：默认帧合法；主轴固定、误差/增长轴自适应 log", () => {
  const scene = splineVsPolyModel(VSPOLY, OPTS);
  assertValid(scene);
  assert.deepEqual(scene.axes.map((a) => a.id), ["error", "growth", "main"]);
  const [error, growth, main] = scene.axes;
  assert.equal(error?.yscale, "log");
  assert.equal(error?.autoscaled, true);
  assert.equal(growth?.yscale, "log");
  assert.equal(growth?.xscale, "log");
  assert.equal(growth?.autoscaled, true);
  assert.equal(main?.autoscaled, false);
  for (const spec of SPEC.vspoly.axes) {
    const axis = scene.axes.find((a) => a.id === spec.id) ?? fail(`缺少轴 ${spec.id}`);
    assert.equal(axis.autoscaled, spec.autoscaled);
  }
});

test("2.2/spline_vs_poly：requiredAux（error / residual）都有可见图层", () => {
  assert.deepEqual(SPEC.vspoly.requiredAux, ["error", "residual"]);
  const scene = splineVsPolyModel(VSPOLY, OPTS);
  const err = lineSeries(scene, "error");
  assert.ok(err.some((s) => (s.label ?? "").includes("error_spline")));
  assert.ok(err.some((s) => (s.label ?? "").includes("error_poly")));
  assert.ok(err.some((s) => (s.label ?? "").includes("error_floor_line")), "舍入地板参考线必须在");
  // residual：节点上的多项式残差（"节点上精确"这个陷阱的量化）
  assert.match(textsOf(scene), /节点残差|节点上是精确的/);
  assert.ok(
    scene.layers.some((l) => l.kind === "scatter" && l.axes === "main"),
    "节点散点在主轴上",
  );
});

test("2.2/spline_vs_poly：规则 2 —— loglogfit 同时给实测斜率与理论斜率", () => {
  const scene = splineVsPolyModel(VSPOLY, OPTS);
  const fit = scene.layers.find((l): l is Extract<Layer, { kind: "loglogfit" }> => l.kind === "loglogfit");
  assert.ok(fit !== undefined, "增长轴必须是 loglogfit 层");
  assert.ok(Number.isFinite(fit.measuredSlope), "实测斜率必须有限");
  assert.equal(fit.theorySlope, 4, "样条理论阶 O(h⁴)");
  assert.match(textsOf(scene), /理论/);
});

test("2.2/spline_vs_poly：多项式溢出 → overflow 徽标 + 固定主轴不放大（规则 1/4）", async () => {
  const yPoly = Float64Array.from(XS, (x) => 3 * Math.sin(9 * x));
  const arrays = vsPolyArrays().map((a) => (a.name === "y_poly" ? { ...a, values: yPoly } : a));
  const meta = { ...(await metaOf(VSPOLY)), overflow: true, poly_max_abs: 3 };
  const frame = await makeFrame("2.2", VIZ_VSPOLY, arrays, {
    meta,
    strings: {
      function_name: "Runge 1/(1+25x^2) / 等距节点",
      bc_name: "自然（M0=Mn=0）",
      note: "多项式峰值 3，已冲出画面",
    },
  });
  const scene = splineVsPolyModel(frame, OPTS);
  assertValid(scene);
  assert.equal(scene.overflow.flag, true);
  assert.ok(scene.overflow.note.length > 0, "溢出时 note 必须非空");
  assert.ok(badgesOf(scene, "overflow").length >= 1, "必须挂 overflow 徽标");
  const main = scene.axes.find((a) => a.id === "main") ?? fail("缺少主轴");
  assert.equal(main.autoscaled, false);
  assert.ok((main.ylim[1] as number) < 2, "主轴不允许被多项式撑开");
  // 被裁掉的点不能在图层里留下越界值（否则线条会画到别的面板上）
  const poly = scene.layers
    .filter((l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === "main")
    .flatMap((l) => l.series)
    .find((s) => (s.label ?? "").includes("多项式"));
  assert.ok(poly !== undefined);
  for (const v of Array.from(poly.y as ArrayLike<number>)) {
    assert.ok(v >= (main.ylim[0] as number) - 1e-9 && v <= (main.ylim[1] as number) + 1e-9, "已裁到轴内");
  }
  assert.match(textsOf(scene), /裁到|溢出/);
});

test("2.2/spline_vs_poly：缺 meta.error_floor → 不虚构地板，仍用 u/2 兜底并在降级里点名", async () => {
  const frame = await vsPolyFrame({ dropMeta: ["error_floor", "lebesgue_constant"] });
  const scene = splineVsPolyModel(frame, OPTS);
  assertValid(scene);
  assert.match(textsOf(scene), /地板 —|地板/);
});

test("2.2/spline_vs_poly：缺数组 → 抛 ModelInputError", async () => {
  const frame = await vsPolyFrame({ dropArrays: ["error_floor_line"] });
  assert.throws(() => splineVsPolyModel(frame, OPTS), ModelInputError);
});

// --------------------------------------------------------------------------- //
// 5. index / 注册表
// --------------------------------------------------------------------------- //
test("2.2/index：四个 viz 全部登记，未知 id 返回 null（不抛错）", () => {
  assert.deepEqual(Object.keys(models).sort(), ["bspline_local", "control_points", "smoothness_c0c1c2", "spline_vs_poly"]);
  for (const viz of manifest.vizzes) {
    assert.ok(modelFor(viz.id) !== null, `${viz.id} 必须有 model`);
  }
  assert.equal(modelFor("nope"), null);
  assert.equal(models[VIZ_CONTROL], controlPointsModel);
});

test("2.2/index：manifest 的 viz 数与模型数一致（POLICY：可视化数 = 声明数）", () => {
  assert.equal(manifest.chapter, "2.2");
  assert.equal(manifest.vizzes.length, 4);
  assert.equal(Object.keys(models).length, manifest.vizzes.length);
});

// --------------------------------------------------------------------------- //
// 工具：取一帧的 meta（造变体帧时用）
// --------------------------------------------------------------------------- //
async function metaOf(frame: Frame): Promise<Record<string, number | boolean | null>> {
  return { ...frame.meta };
}
