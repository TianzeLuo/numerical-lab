/**
 * 1.5 章（稀疏迭代法与预条件）model 层的测试。
 *
 * 夹具：**手搓帧**（`new Frame(...)` → `encodeNlf` → `decodeNlf`），字段集合（数组名 / dtype /
 * shape / 顺序 / meta 键 / strings 键 / critical 键）与 `server/chapters/1.5.py` 的真实帧逐项一致，
 * 数值量级也是从真实帧上抄下来的（κ=20 的赛跑、n=32 的模态谱、GMRES(3) 的 4×3 Hessenberg、
 * SSOR(ω=1.2)/8×8 的 32.16 → 3.59）。**本文件不需要 Python / numpy**，`npm test` 在任何机器上都能跑。
 *
 * 这里守的是什么（对应 1.5 最容易被画错的地方）
 * --------------------------------------------
 * | 断言 | 防的是 |
 * |---|---|
 * | 五条轨迹的**点数 = race_len + 1** | 把收敛后的重复末点画成"方法卡住了"（`race_len` 才是真实步数） |
 * | 实测比值里的 `-1` **不进任何图层** | 哨兵被当成 0 画上去，变成"这个模态不衰减"的反结论 |
 * | `mode_rates_measured_valid == 0` 的点缺席 | 掩码被无视，拿哨兵当测量值 |
 * | 空 `hessenberg`（CG 的 0×0）→ 不画 heat + 文本点名 | 把"没有 H"画成"全是 0 的 H" |
 * | `ilu_dropped_applicable=false` → 引帧的 `sentinel_note` | model 自己发明"帧没说明"，或把 -1 当计数 |
 * | `meta.omega = 0`（只有 SSOR 用 ω）→ 文本说不适用 | 把哨兵 0 当成合法的松弛因子画进副标题 |
 * | NaN/∞ 计数进 `overflow.note` + 角标 | 静默丢点 |
 * | 缺必需数组 / pitfalls 太短 → `ModelInputError` | 画半张图、或 model 自己编 pitfalls（规则 7） |
 * | 窄画布 480×600 仍合规；同帧两次 `sceneToJson` 逐字节相同 | 布局硬编码 960 宽 / 结果不确定 |
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { linspace } from "../shared/src/axes.js";
import { Frame, type ArraySpec, type Dtype, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { sceneToJson, validateScene, type Layer, type Scene } from "../shared/src/scene.js";
import { manifest, modelFor, models } from "../chapters/1.5/index.js";

// --------------------------------------------------------------------------- //
// 0. 夹具
// --------------------------------------------------------------------------- //

const VIZ_IDS = ["iteration_race", "error_spectrum", "krylov_growth", "precond_effect"] as const;
type VizId = (typeof VIZ_IDS)[number];

const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };

/** `manifest.requiredAux` 的每个数组名画在哪根轴上（少了就是"辅助量没露面"，规则 1 的红线）。 */
const AUX_AXIS: Record<(typeof VIZ_IDS)[number], Record<string, string>> = {
  iteration_race: { residual: "convergence" },
  error_spectrum: { error: "decay", residual: "decay" },
  krylov_growth: { residual: "resid" },
  precond_effect: { error: "convergence", residual: "convergence" },
};

function fail(message: string): never {
  throw new Error(message);
}

function specOf(viz: string) {
  return manifest.vizzes.find((v) => v.id === viz) ?? fail(`manifest 缺 ${viz}`);
}

interface ArrayDef {
  name: string;
  data: ArrayLike<number>;
  /** 缺省 `[data.length]`；二维数组必须显式给（`components` 是 16×61 这种）。 */
  shape?: number[];
  dtype?: Dtype;
  unit?: string;
  label?: string;
}

interface FrameDef {
  chapter: string;
  viz: string;
  arrays: ArrayDef[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical: Record<string, number | number[]>;
  pitfalls: string;
}

/** 去掉若干键（造"缺字段"的帧；置 `undefined` 会被 `.nlf` 写端拒绝，必须真删）。 */
function omit<T extends object>(rec: T, keys: readonly string[]): T {
  const out = { ...rec } as Record<string, unknown>;
  for (const k of keys) delete out[k];
  return out as T;
}

function toTyped(data: ArrayLike<number>, dtype: Dtype): TypedArray {
  switch (dtype) {
    case "f8":
      return Float64Array.from(data as ArrayLike<number>);
    case "f4":
      return Float32Array.from(data as ArrayLike<number>);
    case "i4":
      return Int32Array.from(data as ArrayLike<number>);
    case "u1":
      return Uint8Array.from(data as ArrayLike<number>);
    default:
      throw new Error(`夹具里未实现的 dtype：${dtype}`);
  }
}

/** 造帧 → 编码成 `.nlf` → 解码回来：夹具本身就是**合法的线上帧**。 */
async function makeFrame(def: FrameDef): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const a of def.arrays) {
    const dtype = a.dtype ?? "f8";
    const data = toTyped(a.data, dtype);
    const shape = a.shape ?? [a.data.length];
    specs.push({
      name: a.name,
      dtype,
      shape,
      order: "C",
      offset: 0,
      nbytes: data.byteLength,
      unit: a.unit ?? "",
      label: a.label ?? "",
    });
    arrays.set(a.name, data);
  }
  const frame = new Frame({
    chapter: def.chapter,
    viz: def.viz,
    kind: "frame",
    codec: "none",
    meta: def.meta,
    strings: def.strings,
    critical: def.critical,
    pitfalls: def.pitfalls,
    specs,
    arrays,
  });
  return decodeNlf(await encodeNlf(frame));
}

interface Over {
  arrays?: Record<string, ArrayLike<number>>;
  /** 覆盖数组数据时同步给 shape（二维数组必须显式给，否则会造出"形状不是二维"的非法帧）。 */
  shapes?: Record<string, number[]>;
  drop?: string[];
  meta?: Record<string, number | boolean | null>;
  dropMeta?: string[];
  strings?: Record<string, string>;
  dropStrings?: string[];
  critical?: Record<string, number | number[]>;
  pitfalls?: string;
}

function mergeArrays(base: ArrayDef[], over: Over): ArrayDef[] {
  const arrays = base
    .filter((a) => !(over.drop ?? []).includes(a.name))
    .map((a) =>
      over.arrays?.[a.name] !== undefined
        ? // 覆盖数据时同步 shape：一维覆盖二维会造出长度对不上的非法帧，那不是这一条要测的东西
          {
            ...a,
            data: over.arrays[a.name] as ArrayLike<number>,
            shape: over.shapes?.[a.name] ?? [(over.arrays[a.name] as ArrayLike<number>).length],
          }
        : a,
    );
  for (const [name, data] of Object.entries(over.arrays ?? {})) {
    if (!arrays.some((a) => a.name === name)) arrays.push({ name, data, unit: "1", label: name, shape: [data.length] });
  }
  return arrays;
}

// —— iteration_race：κ=20、θ=35°、ω=1.3、SOR 高亮；5 方法 × 61 步（收敛后是重复末点） —— //

const RACE_LEN = [60, 60, 57, 2, 2];
const N_STEP_COLS = 61;

/** 一条"先快后慢"的合成残差（单调下降、末端到 1e-13 量级）；不是物理计算。 */
function synthResidualRow(realSteps: number, floor = 1e-13): Float64Array {
  const out = new Float64Array(N_STEP_COLS);
  for (let k = 0; k < N_STEP_COLS; k += 1) {
    const eff = Math.min(k, realSteps);
    const v = 0.95 ** eff;
    out[k] = v <= floor ? floor : v;
  }
  return out;
}

function raceArrays(): ArrayDef[] {
  const grid = linspace(0, 60, N_STEP_COLS);
  // 真帧里 contour_x / contour_y 各是**一个 (6,181) 的二维数组**（6 条等高线）。
  const contourX = new Float64Array(6 * 181);
  const contourY = new Float64Array(6 * 181);
  for (let i = 0; i < 6; i += 1) {
    const a = 0.2 + 0.3 * i;
    for (let j = 0; j < 181; j += 1) {
      const th = (2 * Math.PI * j) / 180;
      contourX[i * 181 + j] = 0.4 + a * Math.cos(th);
      contourY[i * 181 + j] = 0.3 + a * 0.6 * Math.sin(th);
    }
  }
  const raceX = new Float64Array(5 * N_STEP_COLS);
  const raceY = new Float64Array(5 * N_STEP_COLS);
  const raceRes = new Float64Array(5 * N_STEP_COLS);
  for (let i = 0; i < 5; i += 1) {
    const row = synthResidualRow(RACE_LEN[i] as number);
    for (let k = 0; k < N_STEP_COLS; k += 1) {
      const idx = i * N_STEP_COLS + k;
      const t = Math.min(k, RACE_LEN[i] as number) / 60;
      raceX[idx] = 1 - t * (0.5 + 0.1 * i);
      raceY[idx] = 1.5 - t * (1.0 + 0.05 * i);
      raceRes[idx] = row[k] as number;
    }
  }
  return [
    { name: "contour_x", data: contourX, shape: [6, 181], unit: "1", label: "等高线 x" },
    { name: "contour_y", data: contourY, shape: [6, 181], unit: "1", label: "等高线 y" },
    { name: "race_x", data: raceX, shape: [5, N_STEP_COLS], unit: "1", label: "各方法轨迹 x" },
    { name: "race_y", data: raceY, shape: [5, N_STEP_COLS], unit: "1", label: "各方法轨迹 y" },
    { name: "race_residual", data: raceRes, shape: [5, N_STEP_COLS], unit: "1", label: "各方法相对残差" },
    { name: "race_len", data: Float64Array.from(RACE_LEN), unit: "1", label: "各方法真实步数" },
    { name: "traj_mark_x", data: Float64Array.of(0.5), unit: "1", label: "当前方法游标 x" },
    { name: "traj_mark_y", data: Float64Array.of(0.75), unit: "1", label: "当前方法游标 y" },
    { name: "residual", data: Float64Array.of(7.6e-13), unit: "1", label: "当前步相对残差" },
    { name: "residual_mark", data: Float64Array.of(7.6e-13), unit: "1", label: "当前步相对残差（游标）" },
    { name: "step_grid", data: grid, unit: "1", label: "迭代步" },
    { name: "x_star", data: Float64Array.of(0.4, 0.3), unit: "1", label: "精确解" },
    { name: "b_vec", data: Float64Array.of(0, 0, 1, 1.5), unit: "1", label: "右端项线段 [x0,x1,y0,y1]" },
  ];
}

async function raceFrame(over: Over = {}): Promise<Frame> {
  return makeFrame({
    chapter: "1.5",
    viz: "iteration_race",
    arrays: mergeArrays(raceArrays(), over),
    meta: omit(
      {
        angle: 35.0,
        cg_iters: 2,
        converged_current: true,
        factor_measured: 0.601431640625,
        gmres_iters: 2,
        kappa: 20.0,
        kappa_actual: 20.0,
        method_index: 2,
        mu_jacobi: 0.8944271909999159,
        n_iter_current: 57,
        n_methods: 5,
        n_steps: 60,
        omega: 1.3,
        omega_opt: 1.381261251172633,
        overflow: 0,
        residual: 7.6e-13,
        rho_gs: 0.7994134726356794,
        rho_jacobi: 0.8944271909999159,
        rho_sor: 0.6013312521946036,
        rho_sor_measured: 0.601431640625,
        step: 60,
        t: 60.0,
        threshold_omega: 2.0,
        ...over.meta,
      },
      over.dropMeta ?? [],
    ),
    strings: omit(
      {
        data_source: "backend",
        method: "sor",
        method_name: "SOR",
        note: "SOR：第 60 步相对残差 7.599e-13；μ=0.8941，ω_opt=1.3813，ρ(SOR,ω)=0.6013",
        params_json: '{"angle":35.0,"kappa":20.0,"method":"sor","omega":1.3}',
        race_order: "Jacobi / Gauss–Seidel / SOR / CG（共轭梯度） / GMRES(2)",
        ...over.strings,
      },
      over.dropStrings ?? [],
    ),
    critical: { omega: 1.0, kappa: 1.0, ...over.critical },
    pitfalls: over.pitfalls ?? specOf("iteration_race").pitfalls,
  });
}

// —— error_spectrum：n=32、SOR、ω=1.6、全模态等幅；16 模态 × 61 步 —— //

const N_MODES = 16;
const SPECTRUM_STEPS = 60;
/** 实测比值：前 12 个模态测到，后 4 个是 -1 哨兵（窗口内幅值不足 3 点高于 1e-14）。 */
const MEASURED_VALID = 12;

function spectrumArrays(): ArrayDef[] {
  const grid = linspace(0, SPECTRUM_STEPS, SPECTRUM_STEPS + 1);
  const comp = new Float64Array(N_MODES * (SPECTRUM_STEPS + 1));
  const compLog = new Float64Array(N_MODES * (SPECTRUM_STEPS + 1));
  for (let j = 0; j < N_MODES; j += 1) {
    const rate = j === 0 ? 0.9630604037 : 0.5 ** (j / 4);
    // 真帧里初始误差是归一化的（全模态等幅、‖e_0‖=1）：每个模态 1/√16 = 0.25 起步，
    // 否则 k=0 处 16 个模态相加会大于 1，decay 曲线冲出轴（夹具不真，测试就会误报溢出）。
    const v0 = 1 / Math.sqrt(N_MODES);
    for (let k = 0; k <= SPECTRUM_STEPS; k += 1) {
      const v = Math.max(v0 * rate ** k, 1e-16);
      comp[j * (SPECTRUM_STEPS + 1) + k] = v;
      compLog[j * (SPECTRUM_STEPS + 1) + k] = Math.log10(v);
    }
  }
  const ratesTheory = Float64Array.from({ length: N_MODES }, (_, j) => (j === 0 ? 0.9630604037 : 0.5 ** (j / 4)));
  const ratesMeasured = Float64Array.from({ length: N_MODES }, (_, j) => (j < MEASURED_VALID ? 0.9630604042 : -1));
  const ratesValid = Float64Array.from({ length: N_MODES }, (_, j) => (j < MEASURED_VALID ? 1 : 0));
  return [
    { name: "components", data: comp, shape: [N_MODES, SPECTRUM_STEPS + 1], unit: "1", label: "各模态误差分量" },
    { name: "components_log10", data: compLog, shape: [N_MODES, SPECTRUM_STEPS + 1], unit: "1", label: "log10 分量" },
    {
      name: "components_mark",
      data: Float64Array.from({ length: N_MODES }, (_, j) =>
        Math.max((1 / Math.sqrt(N_MODES)) * (j === 0 ? 0.9630604037 : 0.5 ** (j / 4)) ** SPECTRUM_STEPS, 1e-16),
      ),
      unit: "1",
      label: "当前步分量",
    },
    { name: "mode_rates_theory", data: ratesTheory, unit: "1", label: "解析衰减因子" },
    { name: "mode_rates_measured", data: ratesMeasured, unit: "1", label: "实测收敛比值（-1 = 没测到）" },
    { name: "mode_rates_measured_valid", data: ratesValid, unit: "1", label: "实测比值可用掩码" },
    { name: "modes_index", data: Float64Array.from({ length: N_MODES }, (_, j) => j + 1), unit: "1", label: "模态编号" },
    { name: "error", data: Float64Array.of(4.33e-3), unit: "1", label: "相对误差" },
    { name: "residual", data: Float64Array.of(4.33e-3), unit: "1", label: "相对残差" },
    { name: "error_A", data: Float64Array.of(2.1e-3), unit: "1", label: "A-范数误差" },
    { name: "residual_mark", data: Float64Array.of(4.33e-3), unit: "1", label: "游标残差" },
    { name: "error_mark", data: Float64Array.of(4.33e-3), unit: "1", label: "游标误差" },
    { name: "step_grid", data: grid, unit: "1", label: "迭代步" },
  ];
}

async function spectrumFrame(over: Over = {}): Promise<Frame> {
  return makeFrame({
    chapter: "1.5",
    viz: "error_spectrum",
    arrays: mergeArrays(spectrumArrays(), over),
    meta: omit(
      {
        component_floor: 1e-16,
        error_current: 4.33e-3,
        error_final: 4.33e-3,
        factor_measured: 0.9630604042,
        method_index: 2,
        mu_jacobi: 0.8944271909999159,
        n: 32,
        n_modes_measured: MEASURED_VALID,
        n_modes_shown: N_MODES,
        n_steps: SPECTRUM_STEPS,
        omega: 1.6,
        omega_opt: 1.8263549792920066,
        overflow: 0,
        residual: 4.33e-3,
        rho_measured: 0.9630604042,
        rho_theory: 0.9630604037,
        slowest_mode: 0,
        slowest_rate: 0.9630604037,
        step: SPECTRUM_STEPS,
        t: 60.0,
        ...over.meta,
      },
      over.dropMeta ?? [],
    ),
    strings: omit(
      {
        data_source: "backend",
        init: "all_modes",
        init_name: "全模态等幅",
        method: "sor",
        method_name: "SOR",
        note: "第 60 步：最慢模态 j=1 的理论因子 0.963060，ρ(T) 实测 0.963060",
        params_json: '{"init":"all_modes","n":32,"omega":1.6}',
        ...over.strings,
      },
      over.dropStrings ?? [],
    ),
    critical: { omega: 1.0, ...over.critical },
    pitfalls: over.pitfalls ?? specOf("error_spectrum").pitfalls,
  });
}

// —— krylov_growth：dim=3、GMRES(3)、stretch=2；H 是 (n_iter+1)×n_iter —— //

const KRYLOV_NITER = 3;

function krylovArrays(nIter = KRYLOV_NITER): ArrayDef[] {
  const nRow = nIter + 1;
  const basis = new Float64Array(nIter * 3);
  const iterates = new Float64Array(nRow * 3);
  for (let j = 0; j < nIter; j += 1) {
    basis[j * 3] = 0.297 - 0.6 * j;
    basis[j * 3 + 1] = 0.493 + 0.36 * j;
    basis[j * 3 + 2] = -0.5 * j;
  }
  for (let k = 0; k < nRow; k += 1) {
    iterates[k * 3] = 0.13 * k;
    iterates[k * 3 + 1] = 0.22 * Math.min(k, 2);
    iterates[k * 3 + 2] = 0.03 * k;
  }
  const hess = new Float64Array(nRow * nIter);
  for (let i = 0; i < nRow; i += 1) {
    for (let j = 0; j < nIter; j += 1) {
      hess[i * nIter + j] = i === j + 1 ? 1.5 * (nIter - j) : i <= j ? 1.2 - 0.3 * j : 0;
    }
  }
  const orth = new Float64Array(nRow * nRow);
  for (let i = 0; i < nRow; i += 1) {
    for (let j = 0; j < nRow; j += 1) orth[i * nRow + j] = i === j ? 1 : 1e-16;
  }
  return [
    { name: "basis", data: basis, shape: [nIter, 3], unit: "1", label: "Krylov 基" },
    { name: "basis_grid", data: Float64Array.from({ length: nIter }, (_, j) => j), unit: "1", label: "基的步号" },
    { name: "iterates", data: iterates, shape: [nRow, 3], unit: "1", label: "迭代点" },
    { name: "x_star", data: Float64Array.of(0.4054, 0.0325, 0.0225), unit: "1", label: "精确解" },
    { name: "orthogonality", data: orth, shape: [nRow, nRow], unit: "1", label: "残差正交性实测" },
    { name: "residual", data: Float64Array.of(1, 0.717, 0.437, 0), unit: "1", label: "相对残差" },
    { name: "residual_mark", data: Float64Array.of(0), unit: "1", label: "游标残差" },
    { name: "step_grid", data: Float64Array.from({ length: nRow }, (_, k) => k), unit: "1", label: "Krylov 步" },
    { name: "hessenberg", data: hess, shape: [nRow, nIter], unit: "1", label: "Hessenberg" },
  ];
}

async function krylovFrame(over: Over = {}): Promise<Frame> {
  return makeFrame({
    chapter: "1.5",
    viz: "krylov_growth",
    arrays: mergeArrays(krylovArrays(), over),
    meta: omit(
      {
        arnoldi_orthogonality: 2.220446049250313e-16,
        basis_orthogonality: 2.220446049250313e-16,
        dim: 3,
        happy_breakdown: true,
        kappa_measured: 9.900842920608879,
        method_index: 4,
        n_basis: KRYLOV_NITER,
        n_iter: KRYLOV_NITER,
        n_steps: KRYLOV_NITER,
        orthogonality_max: 3.75e-16,
        overflow: 0,
        residual: 0.0,
        residual_final: 0.0,
        step: KRYLOV_NITER,
        stretch: 2.0,
        subspace_dim: KRYLOV_NITER,
        t: 3.0,
        ...over.meta,
      },
      over.dropMeta ?? [],
    ),
    strings: omit(
      {
        data_source: "backend",
        method: "gmres",
        method_name: "GMRES(3)",
        note: "第 3 步：Krylov 子空间维数 3/3，相对残差 0.000e+00，正交性矩阵的下三角最大值 3.75e-16",
        orthogonality_name: "残差正交性实测矩阵",
        params_json: '{"dim":3,"method":"gmres","stretch":2.0}',
        ...over.strings,
      },
      over.dropStrings ?? [],
    ),
    critical: { ...over.critical },
    pitfalls: over.pitfalls ?? specOf("krylov_growth").pitfalls,
  });
}

// —— precond_effect：SSOR(ω=1.2) / 8×8 Laplacian / CG；κ 32.16 → 3.59，14 步 —— //

const PRECOND_STEPS = 15;

function precondArrays(): ArrayDef[] {
  const nPre = 59;
  const nPost = 63;
  const grid = linspace(0, PRECOND_STEPS - 1, PRECOND_STEPS);
  // 谱：前一组爬升到 7.76，后一组压在 0.28…1.0（与真实帧的 κ 一致）
  const pre = Float64Array.from({ length: nPre }, (_, i) => 0.2412 + (7.759 - 0.2412) * (i / (nPre - 1)) ** 1.5);
  const post = Float64Array.from({ length: nPost }, (_, i) => 0.2789 + (1.0 - 0.2789) * (i / (nPost - 1)) ** 1.2);
  const res = Float64Array.from({ length: PRECOND_STEPS }, (_, k) => Math.max(0.123 ** k, 1e-11));
  return [
    { name: "ritz_pre", data: pre, unit: "1", label: "A 的 Ritz 值" },
    { name: "ritz_post", data: post, unit: "1", label: "预条件后 Ritz 值" },
    { name: "ritz_index_pre", data: Float64Array.from({ length: nPre }, (_, i) => i), unit: "1", label: "序号（前）" },
    { name: "ritz_index_post", data: Float64Array.from({ length: nPost }, (_, i) => i), unit: "1", label: "序号（后）" },
    { name: "convergence", data: Float64Array.from({ length: PRECOND_STEPS }, (_, k) => Math.max(0.9 ** k, 1e-11)), unit: "1", label: "收敛历史" },
    { name: "residual", data: res, unit: "1", label: "相对残差" },
    { name: "error", data: Float64Array.from({ length: PRECOND_STEPS }, (_, k) => Math.max(0.8 ** k, 1e-11)), unit: "1", label: "相对误差" },
    { name: "bound", data: Float64Array.from({ length: PRECOND_STEPS }, (_, k) => Math.max(2 * 0.123 ** k, 1e-7)), unit: "1", label: "CG 最坏界" },
    { name: "step_grid", data: grid, unit: "1", label: "迭代步" },
    { name: "residual_mark", data: Float64Array.of(1.5e-11), unit: "1", label: "游标残差" },
  ];
}

async function precondFrame(over: Over = {}): Promise<Frame> {
  return makeFrame({
    chapter: "1.5",
    viz: "precond_effect",
    arrays: mergeArrays(precondArrays(), over),
    meta: omit(
      {
        converged: true,
        error_final: 1.2e-11,
        factor_measured: 0.12301817381199562,
        grid: 8,
        ilu_dropped: -1,
        ilu_dropped_applicable: false,
        ilu_dropped_sentinel: -1,
        kappa_effective: 1.639805663336336,
        kappa_post: 3.5861054271744828,
        kappa_pre: 32.16343747752635,
        kappa_ratio: 8.968905576012384,
        lambda_max_post: 1.0000000000000002,
        lambda_max_pre: 7.759055417544758,
        lambda_min_post: 0.2788657619189254,
        lambda_min_pre: 0.2412261226122554,
        matrix_nnz: 288,
        n: 64,
        n_iter: 14,
        n_ritz_post: 63,
        n_ritz_pre: 59,
        omega: 1.2,
        overflow: 0,
        precond_index: 1,
        precond_nnz: 288,
        residual: 1.5e-11,
        solver_index: 0,
        spectrum_orthogonality: 2.2e-16,
        spectrum_steps: 59,
        t: 0.0,
        ...over.meta,
      },
      over.dropMeta ?? [],
    ),
    strings: omit(
      {
        data_source: "backend",
        note: "SSOR：实测 κ 32.16 → 3.59（降 8.97 倍），迭代 14 步，等效 κ（由迭代数反推）1.64",
        params_json: '{"grid":8,"omega":1.2,"precond":"ssor","solver":"cg"}',
        precond: "ssor",
        precond_name: "SSOR",
        sentinel_note:
          "meta.ilu_dropped = -1 表示「这个预条件子没有丢弃概念」（Jacobi/SSOR），不是「丢弃了 −1 个填充」；用 meta.ilu_dropped_applicable 判是否有意义。",
        solver: "cg",
        ...over.strings,
      },
      over.dropStrings ?? [],
    ),
    critical: { omega: 1.0, ...over.critical },
    pitfalls: over.pitfalls ?? specOf("precond_effect").pitfalls,
  });
}

type FrameOver = (over?: Over) => Promise<Frame>;

const FRAME_FOR: Record<VizId, FrameOver> = {
  iteration_race: raceFrame,
  error_spectrum: spectrumFrame,
  krylov_growth: krylovFrame,
  precond_effect: precondFrame,
};

// —— 常用断言 —— //

function sceneOf(frame: Frame, over: Partial<ModelOptions> = {}): Scene {
  const model = modelFor(frame.viz) ?? fail(`modelFor(${frame.viz}) 返回 null`);
  return model(frame, { ...OPTS, ...over });
}

/** 用**帧自己的 t** 造场景：夹具的 meta.t 与 opts.t 不一致时会多挂一条"夹紧"降级，
 *  掩盖这一条测试真正要看的降级项（各测试只关心一个变量）。 */
function sceneAt(frame: Frame, over: Partial<ModelOptions> = {}): Scene {
  const t = typeof frame.meta["t"] === "number" ? (frame.meta["t"] as number) : 0;
  return sceneOf(frame, { t, ...over });
}

function assertValid(scene: Scene): void {
  const problems = validateScene(scene);
  assert.deepEqual(problems, [], `Scene 应合法，但有问题：${problems.join("；")}`);
}

function axisOf(scene: Scene, id: string) {
  return scene.axes.find((a) => a.id === id) ?? fail(`场景里没有轴 ${id}`);
}

function layersOn(scene: Scene, axisId: string): Layer[] {
  return scene.layers.filter((l) => l.axes === axisId);
}

function allText(scene: Scene): string {
  return scene.texts.map((t) => t.text).join("\n");
}

function badgeText(scene: Scene): string {
  return scene.badges.map((b) => b.text).join("\n");
}

function lineSeriesOf(scene: Scene, axisId: string): { x: number[]; y: number[] }[] {
  const out: { x: number[]; y: number[] }[] = [];
  for (const layer of layersOn(scene, axisId)) {
    if (layer.kind !== "line") continue;
    for (const s of layer.series ?? []) out.push({ x: Array.from(s.x as ArrayLike<number>), y: Array.from(s.y as ArrayLike<number>) });
  }
  return out;
}

// --------------------------------------------------------------------------- //
// 1. manifest ↔ model 表
// --------------------------------------------------------------------------- //

test("1.5 manifest 与模型表一致：4 个 viz 都有 model，未知 id 返回 null", () => {
  assert.deepEqual(
    manifest.vizzes.map((v) => v.id).sort(),
    [...VIZ_IDS].sort(),
    "manifest 的 viz 集合与测试里的清单不一致",
  );
  for (const viz of VIZ_IDS) {
    assert.notEqual(modelFor(viz), null, `${viz} 没有 model`);
    assert.equal(typeof models[viz], "function", `${viz} 的 model 必须是 (frame, opts) => Scene`);
  }
  assert.equal(modelFor("不存在的 viz"), null);
});

// --------------------------------------------------------------------------- //
// 2. 默认帧 → 合规 Scene；轴口径、requiredAux、source/status 角标
// --------------------------------------------------------------------------- //

for (const viz of VIZ_IDS) {
  test(`1.5/${viz}：默认帧 → 合规 Scene，轴线口径与 requiredAux 都对得上`, async () => {
    const spec = specOf(viz);
    const scene = sceneAt(await FRAME_FOR[viz]());
    assertValid(scene);
    assert.equal(scene.chapter, "1.5");
    assert.equal(scene.viz, viz);
    assert.ok(scene.layers.length > 0, "至少要有一个可绘图层");
    assert.ok(scene.pitfalls.length >= 10, "规则 7：pitfalls 必须带进场景");
    assert.equal(scene.dataSource, "backend", "帧写的是 backend，就不该退化成 preview");
    assert.equal(scene.badges.filter((b) => b.kind === "source").length, 1, "有且只有一枚 source 角标（规则 5）");
    assert.ok(scene.badges.some((b) => b.kind === "status"), "必须有 status 角标");
    for (const ax of spec.axes) {
      assert.equal(axisOf(scene, ax.id).autoscaled, ax.autoscaled ?? false, `${viz} 的轴 ${ax.id} autoscaled 与 manifest 不一致`);
    }
    // requiredAux 是**帧里的数组名**（后端保证有），不是轴 id：这里声明它被画在哪根轴上。
    for (const aux of spec.requiredAux ?? []) {
      const axis = AUX_AXIS[viz][aux] ?? fail(`${viz} 的 requiredAux '${aux}' 没有在测试里登记轴`);
      assert.ok(layersOn(scene, axis).length > 0, `${viz} 的 requiredAux '${aux}' 在轴 ${axis} 上一个图层都没有`);
    }
    // 每根轴都要有 title 或 label 之一，否则图上没人知道这根轴是什么
    for (const ax of scene.axes) {
      assert.ok((ax.title ?? "").length > 0 || (ax.ylabel ?? "").length > 0 || (ax.xlabel ?? "").length > 0, `轴 ${ax.id} 没有名字`);
    }
  });
}

// --------------------------------------------------------------------------- //
// 3. 退化输入：缺必需数组 / pitfalls 太短 → ModelInputError
// --------------------------------------------------------------------------- //

const REQUIRED_ARRAY: Record<VizId, string> = {
  iteration_race: "race_x",
  error_spectrum: "components_log10",
  krylov_growth: "orthogonality",
  precond_effect: "ritz_pre",
};

for (const viz of VIZ_IDS) {
  test(`1.5/${viz}：缺必需数组 '${REQUIRED_ARRAY[viz]}' → 抛 ModelInputError（不画半张图）`, async () => {
    const frame = await FRAME_FOR[viz]({ drop: [REQUIRED_ARRAY[viz]] });
    assert.throws(() => sceneOf(frame), (err: unknown) => err instanceof ModelInputError);
  });

  test(`1.5/${viz}：pitfalls 太短 → 抛 ModelInputError（规则 7：model 不自行编造）`, async () => {
    const frame = await FRAME_FOR[viz]({ pitfalls: "太短" });
    assert.throws(() => sceneOf(frame), (err: unknown) => err instanceof ModelInputError);
  });
}

// --------------------------------------------------------------------------- //
// 4. 重复末点：轨迹长度 = race_len + 1（不是 step_grid 的长度）
// --------------------------------------------------------------------------- //

test("1.5/iteration_race：五条轨迹按 race_len 截断，收敛后的重复末点不画", async () => {
  const scene = sceneAt(await raceFrame());
  assertValid(scene);
  const plane = lineSeriesOf(scene, "plane").filter((s) => s.x.length >= 3 && s.x.length <= 61);
  // 5 条轨迹（等高线是 181 点的闭合曲线，用长度 > 3 且 ≤ 61 过滤掉）
  assert.equal(plane.length, 5, `应画出 5 条轨迹，实际 ${plane.length}`);
  const lens = plane.map((s) => s.x.length).sort((a, b) => a - b);
  assert.deepEqual(lens, [3, 3, 58, 61, 61], "轨迹点数必须是 race_len+1（SOR 57 → 58 点）");
  // 收敛后的重复点不该出现：SOR 那一行第 58…61 列在原数组里是同一个点
  const sor = plane.find((s) => s.x.length === 58) ?? fail("找不到 SOR 的轨迹");
  const last = sor.x[sor.x.length - 1] as number;
  assert.notEqual(last, sor.x[sor.x.length - 2] as number, "SOR 末点不能是重复点");
  assert.match(allText(scene), /race_len/, "面板文本必须交代截断口径（race_len）");
});

test("1.5/iteration_race：缺 race_len 不崩，降级角标点名（会画出重复末点）", async () => {
  const scene = sceneAt(await raceFrame({ drop: ["race_len"] }));
  assertValid(scene);
  assert.match(badgeText(scene), /race_len/, "缺 race_len 时降级角标要点名");
  assert.ok(
    allText(scene).includes("race_len"),
    "文本里仍要说明真实步数口径（退化成整段长度）",
  );
});

// --------------------------------------------------------------------------- //
// 5. -1 哨兵：不画、不当 0、文本与角标点名
// --------------------------------------------------------------------------- //

test("1.5/error_spectrum：实测比值的 -1 哨兵不进任何图层，只在文本/角标里报个数", async () => {
  const scene = sceneAt(await spectrumFrame());
  assertValid(scene);
  for (const layer of layersOn(scene, "rates")) {
    if (layer.kind === "scatter") {
      const ys = Array.from(layer.y as ArrayLike<number>);
      assert.equal(ys.includes(-1), false, "rates 面板上出现了 -1：哨兵被当成测量值画了");
      assert.equal(ys.length, MEASURED_VALID, `可画的实测比值应只有 ${MEASURED_VALID} 个（掩码为 0 的缺席）`);
    }
    if (layer.kind === "line") {
      for (const s of layer.series ?? []) {
        assert.equal(Array.from(s.y as ArrayLike<number>).includes(-1), false, "折线里出现了 -1 哨兵");
      }
    }
  }
  assert.match(badgeText(scene), /-1/, "状态角标要点出哨兵个数");
  assert.match(allText(scene), /哨兵/, "面板文本要说明 -1 是哨兵（不是 0）");
});

test("1.5/error_spectrum：缺掩码时仍不把 -1 当数值（安全退化）", async () => {
  const scene = sceneAt(await spectrumFrame({ drop: ["mode_rates_measured_valid"] }));
  assertValid(scene);
  for (const layer of layersOn(scene, "rates")) {
    if (layer.kind === "scatter") {
      assert.equal(Array.from(layer.y as ArrayLike<number>).includes(-1), false);
    }
  }
  assert.match(badgeText(scene), /mode_rates_measured_valid/, "降级角标要点名缺了掩码");
});

// --------------------------------------------------------------------------- //
// 6. 临界值：用帧的 critical，缺帧值时回落 manifest
// --------------------------------------------------------------------------- //

for (const viz of ["iteration_race", "error_spectrum", "precond_effect"] as const) {
  test(`1.5/${viz}：critical 角标用帧值（ω 越过 1.0 时给出双阈值提示）`, async () => {
    const scene = sceneAt(await FRAME_FOR[viz]());
    const crit = scene.badges.filter((b) => b.kind === "critical");
    assert.ok(crit.length >= 1, `${viz} 的 ω 越过了帧 critical.omega=1.0，应当有 critical 角标`);
    assert.match(crit.map((b) => b.text).join(" "), /ω|omega/, "critical 角标要写清是哪个量越界");
  });
}

test("1.5/krylov_growth：帧 critical 为空 → 不发明临界角标", async () => {
  const scene = sceneAt(await krylovFrame());
  assertValid(scene);
  assert.equal(scene.badges.filter((b) => b.kind === "critical").length, 0, "帧没给 critical，就不该画临界线");
});

// --------------------------------------------------------------------------- //
// 7. Krylov：空 Hessenberg（CG 的 0×0）不画 heat，而且要说清"不是缺数据"
// --------------------------------------------------------------------------- //

test("1.5/krylov_growth：hessenberg 是 0×0（CG 三项递推）→ 不画 heat 层 + 文本点名", async () => {
  const frame = await krylovFrame({
    // 真实 CG 帧：hessenberg 是 (0,0)，但 basis 仍是 (3,3) 的正常二维数组
    arrays: { hessenberg: new Float64Array(0) },
    shapes: { hessenberg: [0, 0] },
    meta: { happy_breakdown: false, residual_final: 4.1983027032905e-16, residual: 4.1983027032905e-16 },
    strings: { method: "cg", method_name: "CG", note: "第 3 步：Krylov 子空间维数 3/3，相对残差 4.198e-16" },
  });
  const scene = sceneOf(frame);
  assertValid(scene);
  assert.equal(layersOn(scene, "hess").length, 0, "0×0 的 H 不该画出 heat 层");
  assert.match(allText(scene), /空矩阵|CG/, "文本必须说明是空矩阵/没有 H，而不是缺数据");
});

test("1.5/krylov_growth：H 的形状 = (n_iter+1)×n_iter，heat 的 values 长度对得上", async () => {
  const scene = sceneAt(await krylovFrame());
  assertValid(scene);
  const hess = layersOn(scene, "hess").find((l) => l.kind === "heat") ?? fail("没有 H 的 heat 层");
  const nx = hess.nx as number;
  const ny = hess.ny as number;
  assert.equal(nx, KRYLOV_NITER, "heat 的 nx = H 的列数 = n_iter（3）");
  assert.equal(ny, KRYLOV_NITER + 1, "heat 的 ny = H 的行数 = n_iter+1（4）");
  assert.equal((hess.values as ArrayLike<number>).length, nx * ny);
  assert.equal(axisOf(scene, "hess").autoscaled, false, "矩阵格子是固定轴（0…n_iter 的格心）");
});

// --------------------------------------------------------------------------- //
// 8. 预条件：哨兵口径（ilu_dropped / ω）与"κ 降幅 ≠ 迭代数降幅"
// --------------------------------------------------------------------------- //

test("1.5/precond_effect：ilu_dropped_applicable=false → 引帧的 sentinel_note，不自己发明判据", async () => {
  const scene = sceneAt(await precondFrame());
  assertValid(scene);
  const text = `${allText(scene)}\n${badgeText(scene)}`;
  assert.match(text, /ilu_dropped/, "要点名这个键");
  assert.match(text, /丢弃/, "要引帧 sentinel_note 里的解释（-1 = 没有丢弃概念）");
  assert.doesNotMatch(text, /帧 strings 未说明|未说明/, "帧给了 sentinel_note，不许说「没说明」");
});

test("1.5/precond_effect：ILU(0) 帧（applicable=true）→ 报真实丢弃次数", async () => {
  const scene = sceneOf(
    await precondFrame({
      meta: { ilu_dropped: 98, ilu_dropped_applicable: true, precond_index: 2 },
      strings: { precond: "ilu0", precond_name: "ILU(0)" },
    }),
  );
  assertValid(scene);
  assert.match(allText(scene), /98/, "要把帧给的丢弃次数写出来");
  assert.doesNotMatch(allText(scene), /ilu_dropped[^\n]*不适用/, "applicable=true 时 ilu 行不能说「不适用」");
});

test("1.5/precond_effect：ω=0 是非 SSOR 帧的哨兵 → 明说「不适用」，不画成 ω=0", async () => {
  const scene = sceneOf(
    await precondFrame({
      meta: { omega: 0.0, precond_index: 0 },
      strings: { precond: "none", precond_name: "无预条件" },
    }),
  );
  assertValid(scene);
  const text = allText(scene);
  assert.match(text, /不适用|哨兵/, "ω=0 必须被解释成哨兵或不适用");
  assert.doesNotMatch(text, /ω\s*=\s*0(\.0+)?\s*；?\s*ω=1 退化/, "不能把哨兵 0 当成合法的 ω 写进正文");
});

test("1.5/precond_effect：κ 降幅 != 迭代数降幅 的结论在图上（两个数都来自帧 meta）", async () => {
  const scene = sceneAt(await precondFrame());
  const text = allText(scene);
  assert.match(text, /!=|≠/, "要有「κ 降幅 != 迭代数降幅」这一句");
  assert.match(text, /32\.16|32\.1634/, "κ_pre 必须出现在图上");
  assert.match(text, /3\.586|3\.59/, "κ_post 必须出现在图上");
  assert.match(text, /8\.969/, "降幅倍数要写出来（帧 meta.kappa_ratio）");
});

test("1.5/precond_effect：两组散点都画在 spectrum 轴上（预条件前后可比）", async () => {
  const scene = sceneAt(await precondFrame());
  assertValid(scene);
  const scatter = layersOn(scene, "spectrum").filter((l) => l.kind === "scatter");
  assert.equal(scatter.length, 2, "预条件前 / 后各一组散点");
  const ys = scatter.flatMap((l) => Array.from(l.y as ArrayLike<number>));
  assert.ok(Math.max(...ys) > 7, "A 的谱要爬到 7 以上（真实帧 7.759）");
  assert.ok(ys.filter((v) => v < 1.2).length >= 60, "预条件后的谱要压在 1 附近");
});

// --------------------------------------------------------------------------- //
// 9. 溢出 / 非有限值：计数进 overflow.note + 角标，场景仍合规
// --------------------------------------------------------------------------- //

test("1.5/iteration_race：residual 里插 NaN/∞ → 不静默丢点，overflow 记账并挂角标", async () => {
  const base = raceArrays();
  const res = Float64Array.from(base.find((a) => a.name === "race_residual")?.data as ArrayLike<number>);
  res[3] = Number.NaN;
  res[4] = Number.POSITIVE_INFINITY;
  const scene = sceneAt(await raceFrame({ arrays: { race_residual: res }, shapes: { race_residual: [5, N_STEP_COLS] } }));
  assertValid(scene);
  assert.equal(scene.overflow.flag, true, "有 NaN/∞ 就必须挂溢出角标");
  assert.ok((scene.overflow.note ?? "").length > 0, "溢出必须带 note（不静默）");
  assert.ok(scene.badges.some((b) => b.kind === "overflow"), "要有 overflow 角标");
});

// --------------------------------------------------------------------------- //
// 10. 布局：窄画布仍合规；同一帧两次逐字节相同
// --------------------------------------------------------------------------- //

for (const viz of VIZ_IDS) {
  test(`1.5/${viz}：窄画布 480×600 仍合规；同一帧两次 sceneToJson 逐字节相同`, async () => {
    const frame = await FRAME_FOR[viz]();
    const narrow = sceneOf(frame, { width: 480, height: 600 });
    assertValid(narrow);
    const a = sceneOf(frame);
    const b = sceneOf(frame);
    assert.equal(sceneToJson(a), sceneToJson(b), "同一帧两次构造必须完全一致（不许有随机/时间依赖）");
  });
}

test("1.5：帧 meta 里的 t 与 opts.t 不一致时以帧为准（图上写的时刻不能是请求值）", async () => {
  const frame = await raceFrame();
  const scene = sceneOf(frame, { t: 12.5 });
  assertValid(scene);
  assert.equal(scene.t, 60, "帧 meta.t=60，场景的 t 必须跟帧走");
});
