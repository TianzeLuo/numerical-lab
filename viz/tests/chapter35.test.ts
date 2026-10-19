/**
 * 3.5 章 model 层的测试（`lorenz_butterfly` / `bifurcation_logistic` / `lyapunov_overlay` /
 * `poincare_section` / `phase_portrait`）。
 *
 * 夹具怎么造：**手搓帧**（`new Frame(...)` → `encodeNlf` → `decodeNlf`）。
 * ---------------------------------------------------------------------
 * 数组名 / shape / 单位 / `meta` 键 / `strings` 键 / `critical` 键 / `pitfalls`
 * 都照 `server/chapters/3.5.py` 的真实帧抄（数值是测试用的合成值，**不是**物理计算）。
 * 这样 `npm test` 在没有 Python、没有 numpy 的机器上也自足，不会因为 `.cache/` 没生成
 * 而变红或静默跳过；而后端字段一改名，这里的"字段集合"断言立刻红。
 *
 * 本文件守的几条"这张图会不会骗人"的规矩
 * --------------------------------------
 * 1. **每个声明轴都要有东西画**，而且 `autoscaled` 必须与 `manifest.gen.ts` 一致
 *    （规则 3：自适应轴要显式标注，固定轴不许偷偷自适应）。
 * 2. **`requiredAux` 必须有可见图层**：`error`/`residual` 这类辅助量如果只写在帧里
 *    而不画出来，判据就成了"看不见的证据"。
 * 3. **文字不许被自己挤掉**：`resolveTextOverlaps` 一旦丢字就会在画布顶部留一行
 *    "另有 N 行未显示"——这里断言所有夹具都不出现那行（丢内容比排版难看严重得多）。
 * 4. **缺字段必须显式降级**：只影响标注的 `meta` 缺失 → 降级角标点名；
 *    必需数组缺失/长度不一致 → 抛 `ModelInputError`（画半张图比不画更坏）。
 * 5. **空帧不崩也不装**：`poincare_section` 在 t=0 时截面数组长度是 0，
 *    这时不许画空层、不许把"还没有穿越点"说成"零残差"。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { linspace } from "../shared/src/axes.js";
import { Frame, type ArraySpec, type Dtype, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, badgeColumns, toNumeric, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Layer, type Scene } from "../shared/src/scene.js";
import { manifest, modelFor, models } from "../chapters/3.5/index.js";

// --------------------------------------------------------------------------- //
// 夹具
// --------------------------------------------------------------------------- //

function fail(message: string): never {
  throw new Error(message);
}

const VIZ_IDS = ["lorenz_butterfly", "bifurcation_logistic", "lyapunov_overlay", "poincare_section", "phase_portrait"] as const;
type VizId = (typeof VIZ_IDS)[number];

const PITFALLS = "这张图会怎么骗你：夹具里的数字是合成的，但字段集合、判据口径与降级路径与真实帧完全一致。";

interface ArrayDef {
  name: string;
  data: ArrayLike<number>;
  shape?: number[];
  unit?: string;
  label?: string;
}

interface FrameDef {
  viz: string;
  arrays: ArrayDef[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical: Record<string, number | number[]>;
}

function toTyped(data: ArrayLike<number>, dtype: Dtype): TypedArray {
  switch (dtype) {
    case "f8":
      return Float64Array.from(data as ArrayLike<number>);
    default:
      throw new Error(`夹具只用 f8：${dtype}`);
  }
}

async function makeFrame(def: FrameDef): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const a of def.arrays) {
    const data = toTyped(a.data, "f8");
    specs.push({
      name: a.name,
      dtype: "f8",
      shape: a.shape ?? [a.data.length],
      order: "C",
      offset: 0,
      nbytes: data.byteLength,
      unit: a.unit ?? "",
      label: a.label ?? "",
    });
    arrays.set(a.name, data);
  }
  const frame = new Frame({
    chapter: "3.5",
    viz: def.viz,
    kind: "frame",
    codec: "none",
    meta: def.meta,
    strings: def.strings,
    critical: def.critical,
    pitfalls: PITFALLS,
    specs,
    arrays,
  });
  return decodeNlf(await encodeNlf(frame));
}

/** 列优先铺开一个 `rows × cols` 矩阵（与后端 `frame.add(..., shape=(rows, cols))` 同序）。 */
function matrix(rows: number, cols: number, f: (r: number, c: number) => number): number[] {
  const out: number[] = [];
  for (let r = 0; r < rows; r += 1) for (let c = 0; c < cols; c += 1) out.push(f(r, c));
  return out;
}

function opts(t = 0): ModelOptions {
  return { width: 960, height: 600, t, params: {} };
}

// —— lorenz_butterfly：602 个采样、8 对孪生轨道 —— //
const LZ_N = 60;
const LZ_T = linspace(0, 30, LZ_N);
const LZ_TRAJ = (phase: number, axis: number) => Array.from(LZ_T, (t) => 20 * Math.sin(0.7 * t + phase + axis));
const LZ_SEP = Array.from(LZ_T, (_, i) => 1e-8 * Math.exp(0.8876 * LZ_T[i]!));
// `error_stepsize` 没有自己的时间轴：模型按 `1e-6 ≤ d ≤ 1` 的掩码去 `sep_mean` 上取时刻，
// 点数对不上就（正确地）不画那根曲线——夹具必须复现这个口径，否则测的就不是真行为。
const LZ_WINDOW = LZ_SEP.map((d, i) => (d >= 1e-6 && d <= 1 ? i : -1)).filter((i) => i >= 0);
const LZ_ERR = LZ_WINDOW.map((_, i) => 0.2 + 0.5 * Math.sin(i));

function lorenzDef(): FrameDef {
  return {
    viz: "lorenz_butterfly",
    arrays: [
      ...[0, 1, 2].flatMap((axis) => [
        { name: `traj1_${"xyz"[axis]}`, data: LZ_TRAJ(0, axis), unit: "1", label: "轨道 1" },
        { name: `traj2_${"xyz"[axis]}`, data: LZ_TRAJ(0.001, axis), unit: "1", label: "轨道 2" },
      ]),
      { name: "traj1_u", data: Array.from(LZ_T, (t) => 20 * Math.sin(0.7 * t)), unit: "1", label: "投影 u" },
      { name: "traj1_v", data: Array.from(LZ_T, (t) => 25 + 15 * Math.cos(0.7 * t)), unit: "1", label: "投影 v" },
      { name: "traj2_u", data: Array.from(LZ_T, (t) => 20 * Math.sin(0.7 * t + 1e-3)), unit: "1", label: "投影 u（孪生）" },
      { name: "traj2_v", data: Array.from(LZ_T, (t) => 25 + 15 * Math.cos(0.7 * t + 1e-3)), unit: "1", label: "投影 v（孪生）" },
      { name: "t_traj", data: LZ_T, unit: "s", label: "时间" },
      { name: "sep_t", data: LZ_T, unit: "s", label: "时间" },
      { name: "sep_pairs", data: matrix(LZ_N, 8, (r, c) => LZ_SEP[r]! * (1 + 0.1 * c)), shape: [LZ_N, 8], unit: "1", label: "每一对" },
      { name: "sep_mean", data: LZ_SEP, unit: "1", label: "平均分离" },
      { name: "sep_coarse", data: LZ_SEP.map((v) => v * 1.002), unit: "1", label: "粗步长平均分离" },
      { name: "error_stepsize", data: LZ_ERR, unit: "1", label: "两种步长的相对差" },
      { name: "fit_x", data: [4.9, 20.25], unit: "s", label: "拟合窗口" },
      { name: "fit_y", data: [LZ_SEP[10]!, LZ_SEP[40]!], unit: "1", label: "拟合线" },
      { name: "growth_line_t", data: [0, 30], unit: "s", label: "增长线" },
      { name: "growth_line_d", data: [1e-8, 1e-8 * Math.exp(0.8876 * 30)], unit: "1", label: "增长线" },
      { name: "floor_line_t", data: [0, 30], unit: "s", label: "地板线" },
      { name: "floor_line_d", data: [1e-3, 1e-3], unit: "1", label: "地板线" },
      { name: "float_line_d", data: [4.44e-15, 4.44e-15], unit: "1", label: "float64 极限" },
      { name: "mark1_u", data: [10], unit: "1", label: "当前点" },
      { name: "mark1_v", data: [25], unit: "1", label: "当前点" },
      { name: "mark2_u", data: [10.01], unit: "1", label: "当前点（孪生）" },
      { name: "mark2_v", data: [25.01], unit: "1", label: "当前点（孪生）" },
      { name: "mark_t", data: [30], unit: "s", label: "标记时刻" },
      { name: "mark_sep", data: [13.8653], unit: "1", label: "标记分离" },
      { name: "start_x", data: [1, 2, 3, 4, 5, 6, 7, 8], unit: "1", label: "初值" },
      { name: "start_y", data: [1, 2, 3, 4, 5, 6, 7, 8], unit: "1", label: "初值" },
      { name: "start_z", data: [1, 2, 3, 4, 5, 6, 7, 8], unit: "1", label: "初值" },
      { name: "rate_pairs", data: [0.72, 0.8, 0.85, 0.88, 0.89, 0.9, 0.91, 0.91], unit: "1", label: "每对增长率" },
      { name: "rate_pair_valid", data: [1, 1, 1, 1, 1, 1, 1, 1], unit: "1", label: "有效" },
      { name: "pair_index", data: [0, 1, 2, 3, 4, 5, 6, 7], unit: "1", label: "编号" },
    ],
    meta: {
      sigma: 10,
      rho: 28,
      beta: 2.6667,
      rho_hopf: 24.736842105263154,
      eps: 1e-8,
      log10_eps: -8,
      n_pairs: 8,
      t_now: 30,
      h_fine: 0.005,
      h_coarse: 0.01,
      stride: 10,
      n_samples: 602,
      rate: 0.8876420263737683,
      rate_ok: true,
      rate_coarse: 0.8895928413483786,
      rate_diff: 0.0019508149746102488,
      rate_r2: 0.9707249891867711,
      rate_n_fit: 288,
      lambda_reference: 0.9056,
      saturation: 13.8653,
      t_visible: 12.5,
      t_visible_theory: 12.713,
      display_res: 1e-3,
      rel_step_max: 10.9545,
      rel_step_median: 0.2418,
      fit_lo: 1e-6,
      fit_hi: 1,
      azimuth: -60,
      elevation: 20,
      t: 30,
    },
    strings: {
      projection_note: "正交投影（无透视除法）：远近不改变尺寸。",
      resolution_note: "分辨率：以显示分辨率 1e-3 为准。",
    },
    critical: { rho: 24.736842105263154, log10_eps: -14.3522 },
  };
}

// —— bifurcation_logistic：700 个参数 × 64 个迭代值 —— //
const BIF_NR = 20;
const BIF_KEEP = 8;
const BIF_R = linspace(2.4, 4, BIF_NR);
const BIF_PERIOD = BIF_R.map((r) => (r > 3.6 ? 0 : 2));
const BIF_RES = BIF_PERIOD.map((p, i) => (p > 0 ? 1e-16 : 4e-3 + 1e-3 * (i % 3)));

function bifurcationDef(): FrameDef {
  const bifR: number[] = [];
  const bifX: number[] = [];
  for (let i = 0; i < BIF_NR; i += 1) {
    for (let k = 0; k < BIF_KEEP; k += 1) {
      bifR.push(BIF_R[i]!);
      bifX.push(BIF_PERIOD[i]! > 0 ? 0.5 + 0.01 * k : 0.2 + 0.6 * ((k * 7) % BIF_KEEP) / BIF_KEEP);
    }
  }
  const grid = linspace(0, 1, 21);
  return {
    viz: "bifurcation_logistic",
    arrays: [
      { name: "bif_r", data: bifR, unit: "1", label: "参数" },
      { name: "bif_x", data: bifX, unit: "1", label: "迭代值" },
      { name: "par_grid", data: BIF_R, unit: "1", label: "参数网格" },
      { name: "period_grid", data: BIF_PERIOD, unit: "1", label: "检出周期" },
      { name: "residual_cycle", data: BIF_RES, unit: "1", label: "闭合残差" },
      { name: "residual_min", data: BIF_RES.map((v) => v * 0.5), unit: "1", label: "最小残差" },
      { name: "map_x", data: grid, unit: "1", label: "映射 x" },
      { name: "map_y", data: grid.map((x) => 3.9 * x * (1 - x)), unit: "1", label: "映射 f(x)" },
      { name: "cobweb_x", data: [0.4, 0.9, 0.35, 0.88], unit: "1", label: "蛛网 x" },
      { name: "cobweb_y", data: [0.9, 0.35, 0.88, 0.4], unit: "1", label: "蛛网 y" },
      { name: "cycle_x", data: linspace(0.2, 0.9, BIF_KEEP), unit: "1", label: "周期轨 x" },
      { name: "cycle_y", data: linspace(0.2, 0.9, BIF_KEEP), unit: "1", label: "周期轨 y" },
      { name: "return_map_x", data: grid, unit: "1", label: "对角线 x" },
      { name: "return_map_y", data: grid, unit: "1", label: "对角线 y" },
      { name: "mark_r", data: [3.9], unit: "1", label: "当前参数" },
      { name: "mark_period", data: [0], unit: "1", label: "当前周期" },
      { name: "critical_r", data: [3.0, 3.4495, 3.5441, 3.5644, 3.8284], unit: "1", label: "临界参数" },
    ],
    meta: {
      r_now: 3.9,
      period_now: 0,
      residual_now: 8.961e-3,
      n_r: BIF_NR,
      n_keep: BIF_KEEP,
      n_transient: 800,
      max_period: 32,
      dr: 0.002289,
      n_points: BIF_NR * BIF_KEEP,
      n_chaotic: BIF_PERIOD.filter((p) => p === 0).length,
      n_periodic: BIF_PERIOD.filter((p) => p > 0).length,
      frac_chaotic: 0.4,
      r_accumulation: 3.569945672,
      residual_min: 1e-18,
      residual_max: 7.05e-2,
      t: 0,
    },
    strings: {
      resolution_note: "分辨率不足会把周期窗口画糊：Δr 与窗口宽度同量级时整串倍周期级联被压成一条灰带。",
      period_protocol: "周期检测：|f^p(x)−x| ≤ 1e-9 才算闭合。",
      cobweb_note: "蛛网图从 x0=0.4 迭代 400 次暂态后画 120 步。",
    },
    critical: { r: 3.8284271247461903 },
  };
}

// —— lyapunov_overlay：600 个参数、8 块 —— //
const LY_NR = 20;
const LY_R = linspace(2.4, 4, LY_NR);
const LY_LAMBDA = LY_R.map((r) => (r > 3.5699 ? 0.4 + 0.3 * (r - 3.5699) : -0.6 - 0.2 * (r - 2.4)));

function lyapunovDef(): FrameDef {
  return {
    viz: "lyapunov_overlay",
    arrays: [
      { name: "lyap_r", data: LY_R, unit: "1", label: "参数" },
      { name: "lyap_lambda", data: LY_LAMBDA, unit: "1", label: "λ(r)" },
      { name: "error_sem", data: LY_R.map((_, i) => (i % 5 === 0 ? 1e-18 : 1e-4 * (1 + (i % 4)))), unit: "1", label: "分块标准误" },
      { name: "sem_band_lo", data: LY_LAMBDA.map((v, i) => v - 1e-4 * (1 + (i % 4))), unit: "1", label: "下带" },
      { name: "sem_band_hi", data: LY_LAMBDA.map((v, i) => v + 1e-4 * (1 + (i % 4))), unit: "1", label: "上带" },
      { name: "bif_r", data: LY_R, unit: "1", label: "分岔参数" },
      { name: "bif_x", data: LY_R.map((r) => (r > 3.6 ? 0.5 : 0.4)), unit: "1", label: "分岔迭代值" },
      { name: "zero_x", data: [2.4, 4], unit: "1", label: "λ=0" },
      { name: "zero_y", data: [0, 0], unit: "1", label: "λ=0" },
      { name: "ln2_x", data: [2.4, 4], unit: "1", label: "λ=ln2" },
      { name: "ln2_y", data: [0.6931471805599453, 0.6931471805599453], unit: "1", label: "λ=ln2" },
      { name: "analytic_r", data: [2.5, 3.2, 4], unit: "1", label: "对拍点" },
      { name: "analytic_lambda", data: [-0.6931471805599453, -0.9162907318741602, 0.6931471805599453], unit: "1", label: "解析 λ" },
      { name: "measured_lambda", data: [-0.6931471805599071, -0.9162907318741711, 0.693182745610001], unit: "1", label: "实测 λ" },
      { name: "mark_r", data: [3.9], unit: "1", label: "当前参数" },
      { name: "mark_lambda", data: [0.49875230044794816], unit: "1", label: "当前 λ" },
    ],
    meta: {
      r_probe: 3.9,
      lambda_probe: 0.49875230044794816,
      sem_probe: 1.647e-3,
      period_probe: 0,
      residual_probe: 6.436e-3,
      n_r: LY_NR,
      n_iter: 20000,
      n_transient: 2000,
      n_block: 8,
      dr: 0.002671,
      n_positive: 8,
      frac_positive: 0.4,
      lambda_max: 0.693182745610001,
      lambda_argmax: 4,
      sem_max: 6.38e-3,
      r_accumulation: 3.569945672,
      t: 0,
    },
    strings: {
      estimate_protocol: "丢 2000 暂态、20000 次迭代、8 块、块标准差/√8；标准误度量有限时间涨落而非舍入。",
      analytic_check: "对拍：r=2.5 解析 −0.69314718 / 实测 −0.69314718。",
      r2_special_note: "r=2 是超稳定不动点（λ=−∞），本图从 2.4 起画——是不画，不是截断。",
    },
    critical: { r_probe: 3.8284271247461903 },
  };
}

// —— poincare_section：t=0 空帧 / t=60 有 80 个穿越点 —— //
const PO_N = 40;

function poincareDef(crossings: boolean): FrameDef {
  const n = crossings ? 12 : 0;
  const secT = linspace(0.5, 59.7, n);
  const secX = secT.map((t) => 10 * Math.sin(t / 9));
  const secY = secT.map((t) => 20 * Math.cos(t / 9));
  const secZ = secT.map(() => 27);
  return {
    viz: "poincare_section",
    arrays: [
      { name: "orbit_t", data: linspace(0, crossings ? 60 : 0, PO_N), unit: "s", label: "时间" },
      { name: "orbit_x", data: linspace(-18, 18, PO_N), unit: "1", label: "x" },
      { name: "orbit_y", data: linspace(-24, 24, PO_N), unit: "1", label: "y" },
      { name: "orbit_z", data: linspace(0, 45, PO_N), unit: "1", label: "z" },
      { name: "orbit_u", data: linspace(-25, 25, PO_N), unit: "1", label: "投影 u" },
      { name: "orbit_v", data: linspace(5, 40, PO_N), unit: "1", label: "投影 v" },
      { name: "section_x", data: secX, unit: "1", label: "截面点 x" },
      { name: "section_y", data: secY, unit: "1", label: "截面点 y" },
      { name: "section_z", data: secZ, unit: "1", label: "截面点 z" },
      { name: "section_u", data: secX, unit: "1", label: "截面点投影 u" },
      { name: "section_v", data: secY, unit: "1", label: "截面点投影 v" },
      { name: "section_inplane_x", data: secX, unit: "1", label: "平面内第一轴" },
      { name: "section_inplane_y", data: secY, unit: "1", label: "平面内第二轴" },
      { name: "section_t", data: secT, unit: "s", label: "穿越时刻" },
      { name: "residual_plane", data: secT.map(() => 1e-13), unit: "1", label: "离平面距离" },
      { name: "residual_shift", data: secT.map((t) => 1.9e-4 * (1 + t / 60)), unit: "1", label: "插值→精化位移" },
      { name: "lin_x", data: secX.map((v) => v + 1e-3), unit: "1", label: "线性插值 x" },
      { name: "lin_y", data: secY.map((v) => v + 1e-3), unit: "1", label: "线性插值 y" },
      { name: "lin_z", data: secZ.map(() => 27), unit: "1", label: "线性插值 z" },
      { name: "return_x", data: secX.slice(0, Math.max(0, n - 1)), unit: "1", label: "第 k 次" },
      { name: "return_y", data: secY.slice(1), unit: "1", label: "第 k+1 次" },
      { name: "plane_corner_x", data: crossings ? [-15, 15] : [0, 0], unit: "1", label: "平面框" },
      { name: "plane_corner_y", data: crossings ? [-24, 24] : [0, 0], unit: "1", label: "平面框" },
    ],
    meta: {
      system_code: 0,
      rho: 28,
      sigma: 10,
      plane_index: 2,
      level: 27,
      direction: 1,
      h: 0.01,
      t_now: crossings ? 60 : 0,
      t_end: crossings ? 60 : 0,
      n_cross: n,
      n_cross_max: 80,
      mean_return: crossings ? 0.7487 : -1,
      max_residual: crossings ? 7.28e-12 : -1,
      max_shift: crossings ? 0.0582 : 0,
      has_crossings: crossings,
      blowup: false,
      azimuth: -60,
      elevation: 20,
      t: crossings ? 60 : 0,
    },
    strings: {
      note: "只收由负变正的那次穿越（双向都收会让返回映射变成双值的重影）。",
      protocol: "RK4 定步长 h=0.01；线性插值后用局部导数做 3 次 Newton 精化。",
      projection_note: "正交投影（无透视除法）：远近不改变尺寸。",
    },
    critical: { rho: 24.736842105263158, c: 4.2 },
  };
}

// —— phase_portrait：19×19 方向场 + 8 条轨迹 —— //
const PP_N = 20;
const PP_NTRAJ = 4;

function phaseDef(): FrameDef {
  const grid = linspace(-3.2, 3.2, 19);
  const vfU: number[] = [];
  const vfV: number[] = [];
  for (const y of grid) {
    for (const x of grid) {
      vfU.push(y);
      vfV.push(-x + 0.5 * (1 - x * x) * y);
    }
  }
  const seg = (n: number, k: number) => Array.from({ length: n }, () => k);
  return {
    viz: "phase_portrait",
    arrays: [
      { name: "vf_x", data: matrix(19, 19, (_r, c) => grid[c]!), unit: "1", label: "网格 x" },
      { name: "vf_y", data: matrix(19, 19, (r) => grid[r]!), unit: "1", label: "网格 y" },
      { name: "vf_u", data: vfU, unit: "1", label: "dx/dt" },
      { name: "vf_v", data: vfV, unit: "1", label: "dy/dt" },
      { name: "null_x_x", data: [-3.2, -1, 1, 3.2], unit: "1", label: "dx/dt=0 的 x" },
      { name: "null_x_y", data: [0, 0, 0, 0], unit: "1", label: "dx/dt=0 的 y" },
      { name: "null_x_seg", data: seg(4, 0), unit: "1", label: "分支号" },
      { name: "null_y_x", data: [-1.5, 0, 1.5], unit: "1", label: "dy/dt=0 的 x" },
      { name: "null_y_y", data: [-1.2, 0.4, 1.2], unit: "1", label: "dy/dt=0 的 y" },
      { name: "null_y_seg", data: seg(3, 1), unit: "1", label: "分支号" },
      { name: "analytic_x_x", data: [-3.2, 3.2], unit: "1", label: "闭式" },
      { name: "analytic_x_y", data: [0, 0], unit: "1", label: "闭式" },
      { name: "analytic_x_seg", data: seg(2, 0), unit: "1", label: "分支号" },
      { name: "analytic_y_x", data: [-1.5, 1.5], unit: "1", label: "闭式" },
      { name: "analytic_y_y", data: [-1.2, 1.2], unit: "1", label: "闭式" },
      { name: "analytic_y_seg", data: seg(2, 0), unit: "1", label: "分支号" },
      { name: "residual_null", data: [4.4e-16, 1e-17, 0, 2.2e-16], unit: "1", label: "零倾线残差" },
      { name: "residual_fixed", data: [0], unit: "1", label: "平衡点残差" },
      { name: "fixed_x", data: [0], unit: "1", label: "平衡点 x" },
      { name: "fixed_y", data: [0], unit: "1", label: "平衡点 y" },
      { name: "fixed_kind", data: [5], unit: "1", label: "相型码" },
      { name: "fixed_max_real", data: [0.25], unit: "1", label: "最大实部" },
      { name: "traj_t", data: linspace(0, 12, PP_N), unit: "s", label: "时间" },
      { name: "traj_x", data: matrix(PP_N, PP_NTRAJ, (r, c) => 2 * Math.cos(0.5 * r + c)), shape: [PP_N, PP_NTRAJ], unit: "1", label: "轨迹 x" },
      { name: "traj_y", data: matrix(PP_N, PP_NTRAJ, (r, c) => 2 * Math.sin(0.5 * r + c)), shape: [PP_N, PP_NTRAJ], unit: "1", label: "轨迹 y" },
      { name: "mark_x", data: [1, 0, -1, 0], unit: "1", label: "当前端点" },
      { name: "mark_y", data: [0, 1, 0, -1], unit: "1", label: "当前端点" },
      { name: "start_x", data: [2, 0, -2, 0], unit: "1", label: "初值" },
      { name: "start_y", data: [0, 2, 0, -2], unit: "1", label: "初值" },
      { name: "traj_index", data: [0, 1, 2, 3], unit: "1", label: "轨迹编号" },
    ],
    meta: {
      system_code: 2,
      mu: 0.5,
      c: 0.3,
      mu_critical: 0,
      n_traj: PP_NTRAJ,
      grid_n: 19,
      t_now: 12,
      t_end: 12,
      x_lo: -3.2,
      x_hi: 3.2,
      y_lo: -4.5,
      y_hi: 4.5,
      n_stable: 0,
      n_unstable: 1,
      n_saddle: 0,
      max_residual_null: 4.44e-16,
      max_residual_fixed: 0,
      limit_radius: 1.1,
      limit_radius_theory: null,
      limit_radius_theory_null: true,
      n_null_branches: 2,
      n_analytic_branches: 2,
      n_f_evals: 4,
      blowup: false,
      t: 12,
    },
    strings: {
      kind_codes: "0=稳定结点、1=稳定焦点、2=中心、3=鞍点、4=不稳定结点、5=不稳定焦点、6=临界、7=鞍焦。",
      limit_cycle_note: "van der Pol 的极限环不是圆：轨迹末端平均半径 1.1，幅角与 μ 的极限环半径不同。",
      nullcline_protocol: "零倾线用归一化二维距离筛选成两族，再按分支号断开。",
    },
    critical: { mu: 0 },
  };
}

async function frameFor(viz: VizId, over: { drop?: string[]; dropMeta?: string[]; crossings?: boolean } = {}): Promise<Frame> {
  const def =
    viz === "lorenz_butterfly"
      ? lorenzDef()
      : viz === "bifurcation_logistic"
        ? bifurcationDef()
        : viz === "lyapunov_overlay"
          ? lyapunovDef()
          : viz === "poincare_section"
            ? poincareDef(over.crossings ?? true)
            : phaseDef();
  if (over.drop !== undefined) def.arrays = def.arrays.filter((a) => !over.drop!.includes(a.name));
  if (over.dropMeta !== undefined) for (const k of over.dropMeta) delete def.meta[k];
  return makeFrame(def);
}

async function sceneFor(viz: VizId, over: Parameters<typeof frameFor>[1] = {}, t = 0): Promise<Scene> {
  const model = modelFor(viz) ?? fail(`modelFor(${viz}) 返回 null`);
  return model(await frameFor(viz, over), opts(t));
}

// --------------------------------------------------------------------------- //
// 工具
// --------------------------------------------------------------------------- //

/** 一个图层引用的所有 y 值（不同 kind 的字段名不一样，这里统一取出来）。 */
function yValuesOf(layer: Layer): number[] {
  switch (layer.kind) {
    case "line":
      return layer.series.flatMap((s) => Array.from(s.y as ArrayLike<number>));
    case "band":
      return [...Array.from(layer.band.yhi as ArrayLike<number>), ...Array.from(layer.band.ylo as ArrayLike<number>)];
    case "scatter":
      return Array.from(layer.y as ArrayLike<number>);
    case "heat":
    case "contour":
      return Array.from(layer.values as ArrayLike<number>);
    case "bars":
      return Array.from(layer.values as ArrayLike<number>);
    case "vectorfield":
      return Array.from(layer.v as ArrayLike<number>);
    case "loglogfit":
      return Array.from(layer.err as ArrayLike<number>);
    default:
      return [];
  }
}

function axisLayers(scene: Scene, axis: string): Layer[] {
  return scene.layers.filter((l) => l.axes === axis);
}

/** 某个轴上是否画了"这些值里的至少一个"（用来确认 requiredAux 真的上了图）。 */
function axisDrawsValues(scene: Scene, axis: string, values: ArrayLike<number>): boolean {
  const want = new Set<number>();
  for (let i = 0; i < values.length; i += 1) want.add(values[i] as number);
  return axisLayers(scene, axis).some((l) => yValuesOf(l).some((v) => want.has(v)));
}

function hasDegradedBadge(scene: Scene): boolean {
  return scene.badges.some((b) => b.text.includes("降级"));
}

function droppedTextNotice(scene: Scene): string | undefined {
  return scene.texts.find((t) => t.text.includes("未显示"))?.text;
}

function allText(scene: Scene): string {
  return scene.texts.map((t) => t.text).join("\n");
}

// --------------------------------------------------------------------------- //
// 1. 五个 viz 的公共契约
// --------------------------------------------------------------------------- //

test("3.5：五个 viz 都能出场景，且不出现'另有 N 行未显示'（文字没有被自己挤掉）", async () => {
  for (const viz of VIZ_IDS) {
    const scene = await sceneFor(viz);
    assert.deepEqual(validateScene(scene), [], `${viz} 的 Scene 不合规`);
    assert.equal(scene.chapter, "3.5");
    assert.equal(scene.viz, viz);
    assert.ok(scene.title.length > 0, `${viz} 缺标题`);
    assert.equal(droppedTextNotice(scene), undefined, `${viz} 有文字被挤掉：${droppedTextNotice(scene)}`);
  }
});

test("3.5：每个声明的轴都存在、都有图层、autoscaled 与 manifest 一致（规则 3）", async () => {
  for (const viz of VIZ_IDS) {
    const spec = manifest.vizzes.find((v) => v.id === viz) ?? fail(`manifest 缺 ${viz}`);
    const scene = await sceneFor(viz);
    const byId = new Map(scene.axes.map((a) => [a.id, a]));
    for (const declared of spec.axes) {
      const axis = byId.get(declared.id) ?? fail(`${viz} 少了轴 ${declared.id}`);
      assert.equal(axis.autoscaled, declared.autoscaled, `${viz}/${declared.id} 的 autoscaled 与 manifest 不一致`);
      if (declared.scale === "log") assert.equal(axis.yscale, "log", `${viz}/${declared.id} 该是对数轴`);
      assert.ok(axisLayers(scene, declared.id).length > 0, `${viz}/${declared.id} 上一个图层都没有`);
    }
  }
});

test("3.5：requiredAux 都真的画出来了（辅助量不能只躺在帧里）", async () => {
  const lorenz = await frameFor("lorenz_butterfly");
  const lzScene = await sceneFor("lorenz_butterfly");
  assert.ok(axisDrawsValues(lzScene, "erreldiff", toNumeric(lorenz.array("error_stepsize"))), "lorenz 的 error_stepsize 没上副轴");

  const bif = await frameFor("bifurcation_logistic");
  const bifScene = await sceneFor("bifurcation_logistic");
  assert.ok(axisDrawsValues(bifScene, "residual", toNumeric(bif.array("residual_cycle"))), "bifurcation 的闭合残差没上副轴");

  const ly = await frameFor("lyapunov_overlay");
  const lyScene = await sceneFor("lyapunov_overlay");
  assert.ok(axisDrawsValues(lyScene, "err", toNumeric(ly.array("error_sem"))), "lyapunov 的分块标准误没上副轴");

  const po = await frameFor("poincare_section");
  const poScene = await sceneFor("poincare_section");
  assert.ok(axisDrawsValues(poScene, "residual", toNumeric(po.array("residual_shift"))), "poincare 的定位位移没上残差条");

  const pp = await frameFor("phase_portrait");
  const ppScene = await sceneFor("phase_portrait");
  assert.ok(axisDrawsValues(ppScene, "residual", toNumeric(pp.array("residual_null"))), "phase_portrait 的零倾线残差没上副轴");
});

test("3.5：角标放在合法槽位（两列时列数正确），source/status 都在", async () => {
  for (const viz of VIZ_IDS) {
    const scene = await sceneFor(viz);
    assert.ok(scene.badges.some((b) => b.kind === "source"), `${viz} 缺 source 角标`);
    assert.ok(scene.badges.some((b) => b.kind === "status"), `${viz} 缺 status 角标`);
    const cols = badgeColumns(960);
    const xs = new Set(scene.badges.map((b) => b.x));
    assert.ok(xs.size <= cols, `${viz} 的角标列数 ${xs.size} 超过 ${cols}`);
  }
});

// --------------------------------------------------------------------------- //
// 2. 三条"必须同屏挡住"的误读
// --------------------------------------------------------------------------- //

test("3.5/lorenz：判据常驻 + 两种步长的相对差单独一根轴（量纲不同不许同轴）", async () => {
  const scene = await sceneFor("lorenz_butterfly", {}, 30);
  const text = allText(scene);
  assert.match(text, /不是混沌证据/);
  assert.match(text, /步长对照/);
  assert.match(text, /饱和/);
  const sep = scene.axes.find((a) => a.id === "sep")!;
  const err = scene.axes.find((a) => a.id === "erreldiff") ?? fail("没有 erreldiff 轴");
  assert.notEqual(sep.rect[1], err.rect[1], "相对差与分离距离必须在两根不同的轴上");
  assert.equal(err.autoscaled, true);
  assert.match(allText(scene) + scene.badges.map((b) => b.text).join(""), /Benettin|理论/);
});

test("3.5/bifurcation：容差线 + 灰区计数都在（判据不看图也要看得见）", async () => {
  const frame = await frameFor("bifurcation_logistic");
  const scene = await sceneFor("bifurcation_logistic");
  const res = frame.array("residual_cycle");
  let grey = 0;
  for (let i = 0; i < res.length; i += 1) {
    const v = res[i] as number;
    if (v > 1e-18 && v <= 1e-9) grey += 1;
  }
  const text = allText(scene);
  assert.match(text, /周期检测容差/);
  assert.ok(text.includes(`灰区`), "没提灰区");
  assert.ok(text.includes(`${grey} 个落在地板与容差之间`), `灰区计数与数组数出来的不一致（应为 ${grey}）`);
  assert.ok(axisDrawsValues(scene, "residual", [1e-9]), "容差线 1e-9 没画");
});

test("3.5/lyapunov：λ 轴固定、副轴自适应，且对拍点在同图（解析 vs 实测）", async () => {
  const scene = await sceneFor("lyapunov_overlay");
  const main = scene.axes.find((a) => a.id === "main")!;
  const err = scene.axes.find((a) => a.id === "err")!;
  assert.equal(main.autoscaled, false);
  assert.equal(err.autoscaled, true);
  const text = allText(scene);
  assert.match(text, /解析/);
  assert.match(text, /实测/);
  assert.match(text, /有限时间涨落/);
  assert.ok(scene.badges.some((b) => b.text.includes("autoscaled=false") || b.text.includes("固定")), "固定轴没有显式标注");
});

test("3.5/poincare：线性插值点与 Newton 精化点同图，位移的代价写在残差条上", async () => {
  const scene = await sceneFor("poincare_section");
  const section = axisLayers(scene, "section");
  assert.ok(section.length >= 2, "截面图上必须同时有插值点与精化点");
  assert.ok(section.every((l) => l.kind === "scatter"), "截面图是散点");
  const text = allText(scene);
  assert.match(text, /线性插值/);
  assert.match(text, /Newton/);
  assert.match(text, /不是"零残差"|空 ≠ 0|位移/);
});

test("3.5/phase_portrait：相平面等比例（像素尺度两轴相等）", async () => {
  const scene = await sceneFor("phase_portrait");
  const phase = scene.axes.find((a) => a.id === "phase")!;
  const spanX = phase.xlim[1] - phase.xlim[0];
  const spanY = phase.ylim[1] - phase.ylim[0];
  const ratio = phase.rect[2] / phase.rect[3];
  assert.ok(Math.abs(ratio - spanX / spanY) < 0.02, `相平面不是等比例：像素比 ${ratio.toFixed(3)} vs 世界比 ${(spanX / spanY).toFixed(3)}`);
  assert.match(allText(scene), /极限环半径/);
  assert.match(allText(scene), /零倾线/);
});

// --------------------------------------------------------------------------- //
// 3. 空帧、降级与硬错误
// --------------------------------------------------------------------------- //

test("3.5/poincare：t=0 时截面数组长度是 0——不画空层、也不说成'零残差'", async () => {
  const scene = await sceneFor("poincare_section", { crossings: false }, 0);
  assert.deepEqual(validateScene(scene), []);
  assert.equal(axisLayers(scene, "section").length, 0, "截面图不该有空图层");
  assert.equal(axisLayers(scene, "retmap").length, 0, "返回映射不该有空图层");
  assert.equal(axisLayers(scene, "residual").length, 0, "残差条不该有空图层");
  const text = allText(scene);
  assert.match(text, /n_cross=0/);
  assert.match(text, /不是"零残差"|空 ≠ 0/);
  assert.doesNotMatch(text, /平均返回时间 -1/, "不能把 -1 这种 N/A 哨兵当成读数");
  assert.equal(droppedTextNotice(scene), undefined);
  assert.ok(hasDegradedBadge(scene), "空帧必须显式降级说明");
});

test("3.5：缺 meta（只影响标注）→ 降级角标点名，场景仍然合规", async () => {
  const cases: { viz: VizId; drop: string[] }[] = [
    { viz: "lorenz_butterfly", drop: ["rate", "saturation"] },
    { viz: "bifurcation_logistic", drop: ["max_period", "r_now"] },
    { viz: "lyapunov_overlay", drop: ["r_probe", "lambda_probe"] },
    { viz: "poincare_section", drop: ["n_cross", "h"] },
    { viz: "phase_portrait", drop: ["mu", "limit_radius"] },
  ];
  for (const c of cases) {
    const scene = await sceneFor(c.viz, { dropMeta: c.drop });
    assert.deepEqual(validateScene(scene), [], `${c.viz} 缺 meta 后不合规`);
    assert.ok(hasDegradedBadge(scene), `${c.viz} 缺 ${c.drop.join("/")} 却没有降级角标`);
    const badge = scene.badges.find((b) => b.text.includes("降级"))!;
    assert.ok(badge.text.startsWith("⚠ 降级："), "降级角标必须以统一前缀开头");
    assert.equal(droppedTextNotice(scene), undefined, `${c.viz} 降级后文字被挤掉`);
  }
});

test("3.5：缺临界值 → 回落到 manifest 声明的常量并原样写进角标", async () => {
  const frame = await frameFor("bifurcation_logistic");
  const noCritical = new Frame({
    chapter: frame.chapter,
    viz: frame.viz,
    kind: "frame",
    codec: "none",
    meta: frame.meta,
    strings: frame.strings,
    critical: {},
    pitfalls: frame.pitfalls,
    specs: frame.specs,
    arrays: new Map(frame.names.map((n) => [n, frame.array(n) as TypedArray])),
  });
  const model = models["bifurcation_logistic"] ?? fail("缺 model");
  const scene = model(noCritical, opts());
  const badge = scene.badges.find((b) => b.kind === "critical") ?? fail("缺 critical 角标");
  assert.match(badge.text, /3\.828/, "缺 critical 时应回落到 manifest 的 3.8284");
});

test("3.5：必需数组缺失 / 长度不一致 → 抛 ModelInputError（不画半张图）", async () => {
  await assert.rejects(() => sceneFor("lorenz_butterfly", { drop: ["sep_mean"] }), ModelInputError);
  await assert.rejects(() => sceneFor("bifurcation_logistic", { drop: ["par_grid"] }), ModelInputError);
  await assert.rejects(() => sceneFor("lyapunov_overlay", { drop: ["bif_x"] }), ModelInputError);
  await assert.rejects(() => sceneFor("poincare_section", { drop: ["orbit_z"] }), ModelInputError);
  await assert.rejects(() => sceneFor("phase_portrait", { drop: ["vf_v"] }), ModelInputError);

  // 长度不一致：把 sep_coarse 截短一个点（同一帧内自相矛盾 → 拒绝画错位的图）
  const def = lorenzDef();
  const coarse = def.arrays.find((a) => a.name === "sep_coarse")!;
  coarse.data = (coarse.data as number[]).slice(0, LZ_N - 1);
  const model = models["lorenz_butterfly"] ?? fail("缺 model");
  await assert.rejects(async () => model(await makeFrame(def), opts()), ModelInputError);
});

test("3.5：pitfalls 缺失 → 抛 ModelInputError（规则 7，model 不自行编造）", async () => {
  const def = lyapunovDef();
  const frame = await makeFrame(def);
  const noPitfalls = new Frame({
    chapter: frame.chapter,
    viz: frame.viz,
    kind: "frame",
    codec: "none",
    meta: frame.meta,
    strings: frame.strings,
    critical: frame.critical,
    pitfalls: "",
    specs: frame.specs,
    arrays: new Map(frame.names.map((n) => [n, frame.array(n) as TypedArray])),
  });
  const model = models["lyapunov_overlay"] ?? fail("缺 model");
  assert.throws(() => model(noPitfalls, opts()), ModelInputError);
});

// --------------------------------------------------------------------------- //
// 4. 装配层
// --------------------------------------------------------------------------- //

test("3.5：index.ts 登记了全部 5 个 viz，未知 id 返回 null", () => {
  assert.deepEqual(Object.keys(models).sort(), [...VIZ_IDS].sort());
  for (const viz of VIZ_IDS) assert.notEqual(modelFor(viz), null);
  assert.equal(modelFor("not_a_viz"), null);
  assert.equal(manifest.chapter, "3.5");
  assert.deepEqual(
    manifest.vizzes.map((v) => v.id).sort(),
    [...VIZ_IDS].sort(),
  );
});

test("3.5：model 是纯函数（同一帧两次调用结果一致）", async () => {
  const frame = await frameFor("bifurcation_logistic");
  const model = models["bifurcation_logistic"] ?? fail("缺 model");
  const a = JSON.stringify(model(frame, opts()));
  const b = JSON.stringify(model(frame, opts()));
  assert.equal(a, b);
});
