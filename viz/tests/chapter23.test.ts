/**
 * 2.3 章 model 层的测试（`projection` / `convergence_rates` / `equioscillation` / `zeros_evolution`）。
 *
 * 夹具用哪种方式？—— **手搓帧（唯一夹具）**
 * ----------------------------------------
 * 用 `new Frame(...)` + `encodeNlf` → `decodeNlf` 造出与 `server/chapters/2.3.py` **字段集合
 * 一致**的帧（数组名 / shape / meta 键 / strings 键 / critical 键 / pitfalls 全照抄后端），
 * 于是 `npm test` 在没有 Python、没有 `.cache/` 的机器上自足，也不会静默跳过。
 * 数值是**测试用的合成值**（例如 `error_analytic = 0.33·ρ^(−n)`），但**关系是真的**：
 * `error = f − proj`、`abs_error` 是 3×N 的行主序堆叠、`error_target − error = P_k f − P_n f`，
 * 所以模型里那些"显示空间里的差/绝对值"能测到真东西。
 *
 * 这个文件主要守什么
 * ------------------
 * 1. **契约**：四个 viz 都产出 `validateScene` 通过的 Scene；轴/图层/角标结构与章节约定逐项对。
 * 2. **两条最容易被骗的地方**（见各 model 文件头的说明）在图上**同屏**：
 *    `projection` 的残差 vs `κ(G)u` vs 投影偏差；`equioscillation` 的
 *    `converged` / 振幅比 `null` / `alt_signs_alternate`；
 *    `zeros_evolution` 的"误差量到底是哪一个"；`convergence_rates` 的实测/理论两条线。
 * 3. **不许偷偷补数**：`meta` 缺标量 → 文本写 `?` + 降级角标，绝不写 0；
 *    `meta` 的 `null`（`amplitude_ratio_*`）→ **照原样**写"无有限值（null）"，不当成 1 或 0。
 * 4. **缺数组 / 长度不一致 / pitfalls 太短 → 抛 `ModelInputError`**（画不出来就不画半张图）。
 * 5. **版面自洽**：任何文本块都不许出现"画布放不下"（出现即说明我写的字超出了面板预算），
 *    也不许压住坐标轴装饰带——后者由 `validateScene` 的规则 8 兜住。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { linspace } from "../shared/src/axes.js";
import { Frame, type ArraySpec, type Dtype, type TypedArray } from "../shared/src/frame.js";
import { DEGRADED_PREFIX, ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Layer, type Scene } from "../shared/src/scene.js";
import { convergenceRatesModel } from "../chapters/2.3/convergence_rates.model.js";
import { equioscillationModel } from "../chapters/2.3/equioscillation.model.js";
import { manifest, modelFor, models } from "../chapters/2.3/index.js";
import { projectionModel } from "../chapters/2.3/projection.model.js";
import { zerosEvolutionModel } from "../chapters/2.3/zeros_evolution.model.js";

function fail(message: string): never {
  throw new Error(message);
}

const VIZ_IDS = ["projection", "convergence_rates", "equioscillation", "zeros_evolution"] as const;
function specOf(id: string) {
  return manifest.vizzes.find((v) => v.id === id) ?? fail(`manifest 缺少 ${id}`);
}

// --------------------------------------------------------------------------- //
// 夹具（与 server/chapters/2.3.py 的字段集合一致）
// --------------------------------------------------------------------------- //

const PITFALLS =
  "① 残差小不等于解对：单项式基 n=24 时正规方程残差只有 1e-11，而 κ(G)u≈21.7，系数已全是噪声。" +
  "② 交错点个数 = n+2 是必要条件不是收敛判据：误差掉到舍入地板后极值全是噪声。" +
  "③ 别用 |p_n(x)| 当误差：Laguerre n=40 的 |L_n| 能到 1e14，Newton 修正量只有 1e-14。";

interface ArrayDef {
  name: string;
  data: ArrayLike<number>;
  shape?: number[];
  unit?: string;
  label?: string;
  dtype?: Dtype;
}

interface FrameDef {
  viz: string;
  arrays: ArrayDef[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical: Record<string, number | number[]>;
  pitfalls?: string;
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

function omit<T extends object>(rec: T, keys: readonly string[]): T {
  const out = { ...rec } as Record<string, unknown>;
  for (const k of keys) delete out[k];
  return out as T;
}

/** 造帧 → 编码成 `.nlf` → 解码回来：夹具本身就是**合法的线上帧**，不是内存里的随手对象。 */
async function makeFrame(def: FrameDef): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const a of def.arrays) {
    const dtype = a.dtype ?? "f8";
    const data = toTyped(a.data, dtype);
    specs.push({
      name: a.name,
      dtype,
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
    chapter: "2.3",
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
  drop?: string[];
  shapes?: Record<string, number[]>;
  meta?: Record<string, number | boolean | null>;
  dropMeta?: string[];
  strings?: Record<string, string>;
  dropStrings?: string[];
  critical?: Record<string, number | number[]>;
  pitfalls?: string;
}

function applyOverride(base: ArrayDef[], over: Override): ArrayDef[] {
  const arrays = base
    .filter((a) => !(over.drop ?? []).includes(a.name))
    .map((a) =>
      over.arrays?.[a.name] !== undefined
        ? { ...a, data: over.arrays[a.name] as ArrayLike<number>, shape: over.shapes?.[a.name] ?? a.shape }
        : over.shapes?.[a.name] !== undefined
          ? { ...a, shape: over.shapes[a.name] }
          : a,
    );
  for (const [name, data] of Object.entries(over.arrays ?? {})) {
    if (!arrays.some((a) => a.name === name)) {
      arrays.push({ name, data, shape: over.shapes?.[name] ?? [data.length], unit: "1", label: name });
    }
  }
  return arrays;
}

// ------------------------------ projection ------------------------------- //

const P_X = linspace(-1, 1, 41);
const P_F = Array.from(P_X, (x) => 1 / (1 + 25 * x * x));
/** 12 维 Legendre 投影：用 f 减一条合成误差曲线表示（关系 error = f − proj 必须精确成立）。 */
const P_ERR12 = Array.from(P_X, (x) => 0.05 * Math.cos(3.1 * x) + 0.01 * x);
const P_PROJ12 = Array.from(P_X, (_, i) => (P_F[i] as number) - (P_ERR12[i] as number));
const P_COEF12 = Array.from({ length: 12 }, (_, i) => 0.39 / (1 + i) - 0.12 / (1 + 2 * i));
const P_GRAM12 = (() => {
  const g = new Float64Array(144);
  for (let i = 0; i < 12; i += 1) g[i * 12 + i] = 1;
  return g;
})();

/** 单项式基 n=24、k=23：目标投影与当前投影差一条 1.49e-3 量级的曲线（帧里真的可度量）。 */
const P_DEV = Array.from(P_X, (x) => 1.49e-3 * Math.sin(7 * x));
const P_ERR24 = Array.from(P_X, (x) => 0.0051 * Math.sin(5 * x));
const P_ERR24_TARGET = Array.from(P_X, (_, i) => (P_ERR24[i] as number) - (P_DEV[i] as number));
const P_COEF24 = Array.from({ length: 24 }, (_, i) => (i % 2 === 0 ? 1 : -1) * Math.pow(10, 5.4 - 0.5 * i));

async function projectionFrame(over: Override = {}, variant: "orth" | "mono" = "orth"): Promise<Frame> {
  const orth = variant === "orth";
  const n = orth ? 12 : 24;
  const base: ArrayDef[] = [
    { name: "x", data: P_X, unit: "1", label: "采样点 x" },
    { name: "f", data: P_F, unit: "1", label: "被逼近的函数" },
    { name: "proj", data: orth ? P_PROJ12 : P_ERR24_TARGET.map((e, i) => (P_F[i] as number) - e), unit: "1", label: "当前投影" },
    { name: "proj_target", data: orth ? P_PROJ12 : P_F.map((v, i) => v - (P_ERR24_TARGET[i] as number)), unit: "1", label: "目标投影" },
    { name: "error", data: orth ? P_ERR12 : P_ERR24, unit: "1", label: "当前误差" },
    { name: "error_target", data: orth ? P_ERR12 : P_ERR24_TARGET, unit: "1", label: "目标误差" },
    { name: "coef", data: orth ? P_COEF12 : P_COEF24, unit: "1", label: "投影系数" },
    { name: "gram", data: orth ? P_GRAM12 : new Float64Array(576), shape: [n, n], unit: "1", label: "Gram 矩阵" },
  ];
  const meta: Record<string, number | boolean | null> = orth
    ? {
        t: over.meta?.t !== undefined ? (over.meta.t as number) : 12,
        step: 12,
        n_terms: 12,
        n_current: 12,
        gram_cond: 1,
        gram_cond_times_u: 1.11e-16,
        normal_residual: 1.43e-16,
        l2_error_current: 0.0558234,
        l2_error_target: 0.0558234,
        linf_error_current: 0.100575,
        linf_error_target: 0.100575,
        condition_exceeds_roundoff: false,
        error_floor: 1.1102230246251565e-16,
      }
    : {
        t: over.meta?.t !== undefined ? (over.meta.t as number) : 23,
        step: 23,
        n_terms: 24,
        n_current: 23,
        gram_cond: 1.952e17,
        gram_cond_times_u: 21.668515719301993,
        normal_residual: 1.0438620691522063e-11,
        l2_error_current: 0.00515253,
        l2_error_target: 0.00514984,
        linf_error_current: 0.0109462,
        linf_error_target: 0.0111029,
        condition_exceeds_roundoff: true,
        error_floor: 1.1102230246251565e-16,
      };
  const strings: Record<string, string> = orth
    ? {
        basis: "legendre",
        basis_label: "归一化 Legendre（对 Lebesgue 测度正交）",
        func: "pole",
        note: "Legendre 基对 Lebesgue 测度正交归一，Gram 矩阵≈I，正规方程条件数=1。",
        singular_note: "",
        data_source: "backend",
        params_json: '{"basis":"legendre","func":"pole","n":12}',
      }
    : {
        basis: "monomial",
        basis_label: "单项式 1,x,x²,…（未正交化）",
        func: "pole",
        note: "单项式基在 n 大时迅速变坏。",
        singular_note: "κ(G) 已达 1e17：残差 ‖Gc−b‖ 仍只有 1e-11，但系数 c 已全是噪声。",
        data_source: "backend",
        params_json: '{"basis":"monomial","func":"pole","n":24}',
      };
  return makeFrame({
    viz: "projection",
    arrays: applyOverride(base, over),
    meta: omit({ ...meta, ...(over.meta ?? {}) }, over.dropMeta ?? []),
    strings: omit({ ...strings, ...(over.strings ?? {}) }, over.dropStrings ?? []),
    critical: over.critical ?? (orth ? { n: 22 } : { n: 23 }),
    pitfalls: over.pitfalls,
  });
}

// --------------------------- convergence_rates ---------------------------- //

const C_N = Array.from({ length: 39 }, (_, i) => i + 2);
const C_RHO = 1.219803902718557;
const pow = (base: number, e: number) => Math.pow(base, e);
/** 三条"实测"曲线：合成值，但用的是帧里那几个真实指数（ρ / 1.5106 / 0.4924）。 */
const C_AN = C_N.map((n) => 0.33 * pow(C_RHO, -n) + 1.1e-16);
const C_CUSP = C_N.map((n) => 0.7 * pow(n, -1.5106));
const C_JUMP = C_N.map((n) => 0.72 * pow(n, -0.4924));
/** 理论参考曲线：锚在各自拟合窗口的首个实测点上（与后端做法一致）。 */
function anchored(values: number[], start: number, exponent: number): number[] {
  const i0 = C_N.indexOf(start);
  const v0 = values[i0] as number;
  return C_N.map((n, i) => (i < i0 ? v0 * pow(n / start, exponent) : v0 * pow(n / start, exponent)));
}
const C_THEORY_AN = anchored(C_AN, 3, -Math.log(C_RHO));
const C_THEORY_CUSP = anchored(C_CUSP, 8, -1.5);
const C_THEORY_JUMP = anchored(C_JUMP, 2, -0.5);

async function convergenceFrame(over: Override = {}): Promise<Frame> {
  const base: ArrayDef[] = [
    { name: "n", data: C_N, unit: "1", label: "维数 n" },
    { name: "error", data: [...C_AN, ...C_CUSP, ...C_JUMP], shape: [3, 39], unit: "1", label: "三族 L² 误差（堆叠）" },
    { name: "error_analytic", data: C_AN, unit: "1", label: "解析函数误差" },
    { name: "error_cusp", data: C_CUSP, unit: "1", label: "尖点误差" },
    { name: "error_jump", data: C_JUMP, unit: "1", label: "跳点误差" },
    { name: "theory_analytic", data: C_THEORY_AN, unit: "1", label: "解析理论参考" },
    { name: "theory_cusp", data: C_THEORY_CUSP, unit: "1", label: "尖点理论参考" },
    { name: "theory_jump", data: C_THEORY_JUMP, unit: "1", label: "跳点理论参考" },
    { name: "fit_analytic", data: C_THEORY_AN, unit: "1", label: "解析拟合线" },
    { name: "fit_cusp", data: C_THEORY_CUSP, unit: "1", label: "尖点拟合线" },
    { name: "fit_jump", data: C_THEORY_JUMP, unit: "1", label: "跳点拟合线" },
  ];
  const meta: Record<string, number | boolean | null> = {
    t: 0,
    rate_analytic_measured: -0.198167,
    rate_analytic_theory: -0.19869,
    rate_analytic_r2: 0.99792,
    rate_analytic_rel_error: 0.002632,
    alpha_cusp_measured: 1.510615,
    alpha_cusp_theory: 1.5,
    alpha_cusp_r2: 0.996067,
    alpha_cusp_rel_error: 0.00041,
    alpha_jump_measured: 0.492386,
    alpha_jump_theory: 0.5,
    alpha_jump_r2: 0.993293,
    alpha_jump_rel_error: 0.015228,
    fit_window_analytic_start: 3,
    fit_window_cusp_start: 8,
    fit_window_jump_start: 2,
    rho: C_RHO,
    error_floor: 1.1102230246251565e-16,
    clipped_analytic: 0,
    clipped_cusp: 0,
    clipped_jump: 0,
  };
  const strings: Record<string, string> = {
    analytic: "1/(1+s x²)，s=25",
    regime_analytic: "geometric（Bernstein 椭圆参数 ρ=1.219804，理论斜率 −lnρ=-0.198690）",
    regime_cusp: "algebraic α=3/2（f′ 有跳变）",
    regime_jump: "algebraic α=1/2（f 本身有跳变）",
    floor_note: "误差曲线已按 u=2^-53 截断；截断点数见 meta.clipped_*。log 轴不能画 0。",
    fit_note: "实测斜率与理论斜率的对照都锚在拟合窗口的第一个实测点上——常数项不可知，斜率才是理论断言。",
    data_source: "backend",
    params_json: '{"scale":25,"n_max":40}',
  };
  return makeFrame({
    viz: "convergence_rates",
    arrays: applyOverride(base, over),
    meta: omit({ ...meta, ...(over.meta ?? {}) }, over.dropMeta ?? []),
    strings: omit({ ...strings, ...(over.strings ?? {}) }, over.dropStrings ?? []),
    critical: over.critical ?? {},
    pitfalls: over.pitfalls,
  });
}

// ---------------------------- equioscillation ----------------------------- //

const E_X = linspace(-1, 1, 401);
/** 收敛后的误差曲线：振幅 9.81e-2 的等幅振荡（交错点正好在 ±max|e| 上）。 */
const E_ALT_X = Array.from({ length: 10 }, (_, i) => Math.cos((Math.PI * i) / 9));
const E_SIGN = (x: number) => Math.cos(9 * Math.acos(Math.max(-1, Math.min(1, x))));
const E_FINAL = E_X.map((x) => 0.0980881 * E_SIGN(x));
const E_L2 = E_X.map((x) => 0.149584 * Math.sin(2.2 * x) * (0.6 + 0.4 * x * x));
/** 第 1 轮：误差曲线几乎贴在零上（初始交错点的 |e| 全在舍入地板），振幅比因此无有限值。 */
const E_ROUND0 = E_X.map((x) => 1e-16 * Math.sin(11 * x) + 0);
const E_ALT_ROUND0 = E_ALT_X.map(() => 0);
const E_ALT_FINAL = E_ALT_X.map((x) => 0.0980881 * (x === 0 ? 0.9999 : 1));

async function equioFrame(over: Override = {}, round: 1 | 5 = 5): Promise<Frame> {
  const cur = round === 5
    ? E_X.map((x, i) => (E_FINAL[i] as number) + (x === 0 ? 0 : 0))
    : E_ROUND0;
  const altCur = round === 5 ? E_ALT_FINAL : E_ALT_ROUND0;
  const clip = (v: number) => Math.max(1.1102230246251565e-16, Math.abs(v));
  const base: ArrayDef[] = [
    { name: "x", data: E_X, unit: "1", label: "采样点 x" },
    { name: "error", data: cur, unit: "1", label: "第 k 轮误差（带符号）" },
    { name: "error_final", data: E_FINAL, unit: "1", label: "收敛后误差（带符号）" },
    { name: "abs_error", data: [...cur.map(clip), ...E_FINAL.map(clip), ...E_L2.map(clip)], shape: [3, 401], unit: "1", label: "|误差|（当前/最终/L²）" },
    { name: "alt_x", data: E_ALT_X, unit: "1", label: "第 k 轮交错点" },
    { name: "alt_error", data: altCur, unit: "1", label: "第 k 轮交错点上的误差" },
    { name: "alt_x_final", data: E_ALT_X, unit: "1", label: "最终交错点" },
    { name: "alt_error_final", data: E_ALT_FINAL, unit: "1", label: "最终交错点上的误差" },
    { name: "error_l2", data: E_L2, unit: "1", label: "同阶 L² 投影误差" },
  ];
  const meta: Record<string, number | boolean | null> = {
    t: round,
    step: round,
    n_steps: 5,
    deg: 8,
    n_alt_expected: 10,
    n_alt_current: 10,
    n_alt_final: 10,
    amplitude_ratio_current: round === 5 ? 1.0000000000000033 : null,
    amplitude_ratio_final: 1.0000000000000033,
    alt_signs_alternate: true,
    max_error_final: 0.0980881,
    max_error_l2: 0.149584,
    l2_over_minimax: 1.525,
    iterations: 5,
    converged: true,
    error_floor: 1.1102230246251565e-16,
    clipped_points: round === 5 ? 0 : 4,
  };
  const strings: Record<string, string> = {
    func: "pole",
    objective: "minimax（‖f−p‖∞）",
    contrast: "同阶 L² 投影（‖f−p‖₂）",
    certificate_note: "停机判据是 de la Vallée Poussin 下界 min|e| ≥ (1−ε)·max|e|，不是“交错点不再移动”。",
    floor_note: "误差已取绝对值并按 u=2^-53 截断；掉到地板后交错点全是噪声。",
    converged_note: "交错点集稳定且被最优性证书确认",
    data_source: "backend",
    params_json: '{"func":"pole","deg":8}',
  };
  return makeFrame({
    viz: "equioscillation",
    arrays: applyOverride(base, over),
    meta: omit({ ...meta, ...(over.meta ?? {}) }, over.dropMeta ?? []),
    strings: omit({ ...strings, ...(over.strings ?? {}) }, over.dropStrings ?? []),
    critical: over.critical ?? {},
    pitfalls: over.pitfalls,
  });
}

// ---------------------------- zeros_evolution ----------------------------- //

function zerosArrays(family: "legendre" | "laguerre", nMax: number) {
  const degree: number[] = [];
  const root: number[] = [];
  const error: number[] = [];
  const contrast: number[] = [];
  for (let n = 1; n <= nMax; n += 1) {
    for (let k = 0; k < n; k += 1) {
      const u = (k + 0.5) / n;
      degree.push(n);
      // Legendre 型分布（反正弦律）：端点密；Laguerre 型：向大 x 端稀疏。
      root.push(family === "legendre" ? -Math.cos(Math.PI * u) : -Math.log(1 - u) * (1 + 0.02 * n));
      contrast.push(-Math.cos(Math.PI * u));
      error.push(family === "legendre" ? 1.1102230246251565e-16 : 1e-15 * (1 + (k % 5)));
    }
  }
  // 与后端一致：`gap` 只有**当前阶**（n = nMax）的相邻根间距，`gap_index` = 1…size。
  const gap: number[] = [];
  const gapIndex: number[] = [];
  {
    const roots = root.slice(degree.indexOf(nMax));
    for (let k = 0; k + 1 < roots.length; k += 1) {
      gap.push(Math.abs((roots[k + 1] as number) - (roots[k] as number)));
      gapIndex.push(k + 1);
    }
  }
  const densityX = linspace(family === "legendre" ? -1 : 0.02, family === "legendre" ? 1 : 120, 241);
  const densityY = densityX.map((x) => (family === "legendre" ? 1 / Math.sqrt(Math.max(1e-9, 1 - x * x)) : Math.exp(-x / 2)));
  const rootCurrent = root.slice(degree.lastIndexOf(nMax));
  return { degree, root, error, contrast, gap, gapIndex, densityX, densityY, rootCurrent };
}

async function zerosFrame(over: Override = {}, family: "legendre" | "laguerre" = "laguerre", nMax = 24): Promise<Frame> {
  const a = zerosArrays(family, nMax);
  const base: ArrayDef[] = [
    { name: "degree", data: a.degree, unit: "1", label: "阶数 n" },
    { name: "root", data: a.root, unit: "1", label: "零点 x" },
    { name: "error", data: a.error, unit: "1", label: "Newton 根误差 |p/p′|" },
    { name: "root_contrast", data: a.contrast, unit: "1", label: "Chebyshev 零点（对照）" },
    { name: "gap", data: a.gap, unit: "1", label: "相邻根间距" },
    { name: "gap_index", data: a.gapIndex, unit: "1", label: "间距编号" },
    { name: "density_x", data: a.densityX, unit: "1", label: "密度自变量" },
    { name: "density_y", data: a.densityY, unit: "1", label: "理论密度" },
    { name: "root_current", data: a.rootCurrent, unit: "1", label: "当前阶的零点" },
  ];
  const legendre = family === "legendre";
  const meta: Record<string, number | boolean | null> = {
    t: nMax,
    step: nMax,
    n_current: nMax,
    n_roots_current: nMax,
    max_root_error: legendre ? 1.1102230246251565e-16 : 1.198e-14,
    min_root_error: 1.1102230246251565e-16,
    root_error_floor: 1.1102230246251565e-16,
    clipped_points: legendre ? 300 : 104,
    newton_iter_max: 4,
    newton_iter_mean: 2.257,
    gap_ratio_current: legendre ? 6.262 : 92.339,
    gap_min_current: 0.02046,
    gap_max_current: 0.1281,
    root_span_lo: legendre ? -0.9952 : 0.0357,
    root_span_hi: legendre ? 0.9952 : 142.3,
  };
  const strings: Record<string, string> = {
    family,
    family_label: legendre ? "Legendre P_n（权 1，区间 [−1,1]）" : "Laguerre L_n（权 e^−x，区间 [0,∞)）",
    spacing_note: "零点不是等距的：间距在区间中部最大、靠近端点最小。meta.gap_ratio_current = max gap / min gap 就是实测值。",
    error_note: "误差用 Newton 修正量 |p_n/p_n′|（与多项式自身尺度无关）。Laguerre 的 |L_n| 可以到 1e14。",
    density_note: "理论密度是 n→∞ 的平衡测度，只在中等以上阶数与零点直方图吻合。",
    data_source: "backend",
    params_json: `{"family":"${family}","n_max":${nMax}}`,
  };
  return makeFrame({
    viz: "zeros_evolution",
    arrays: applyOverride(base, over),
    meta: omit({ ...meta, ...(over.meta ?? {}) }, over.dropMeta ?? []),
    strings: omit({ ...strings, ...(over.strings ?? {}) }, over.dropStrings ?? []),
    critical: over.critical ?? {},
    pitfalls: over.pitfalls,
  });
}

// ------------------------------- 小工具 ---------------------------------- //

interface Ctx {
  width: number;
  height: number;
  theme: "light" | "dark";
  t: number;
  history: { t: number; frame: Frame }[];
}

function opts(over: Partial<Ctx> = {}): ModelOptions {
  return {
    t: over.t ?? 0,
    width: over.width ?? 960,
    height: over.height ?? 600,
    params: {},
    theme: over.theme ?? "light",
    history: over.history ?? [],
  };
}

function assertValid(scene: Scene): void {
  const problems = validateScene(scene);
  assert.deepEqual(problems, [], `Scene 应合法，但有问题：${problems.join("；")}`);
}

function layersOf<K extends Layer["kind"]>(scene: Scene, kind: K): Extract<Layer, { kind: K }>[] {
  return scene.layers.filter((l): l is Extract<Layer, { kind: K }> => l.kind === kind);
}

function seriesOfLayer(scene: Scene, axes: string, index: number) {
  const layers = scene.layers.filter(
    (l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === axes,
  );
  assert.ok(layers[index], `轴 ${axes} 上应有第 ${index} 个 line 图层`);
  return layers[index].series;
}

/** 面板文本按绘制顺序拼起来（层与层之间原来是硬换行，这里去掉，便于跨行断言）。 */
function textsOf(scene: Scene): string {
  return scene.texts.map((t) => t.text).join("");
}

/** 再去掉所有空白：`wrapUnits` 会在任意字符处折行，跨行的片段只有这样才比得中。 */
function flatOf(scene: Scene): string {
  return textsOf(scene).replace(/\s+/g, "");
}

function badgesOf(scene: Scene, kind: string): string[] {
  return scene.badges.filter((b) => b.kind === kind).map((b) => b.text);
}

/** 降级角标在实现里是 `badge("status", DEGRADED_PREFIX + …)`，所以按前缀找而不是按 kind 找。 */
function degradedTexts(scene: Scene): string[] {
  return scene.badges.map((b) => b.text).filter((t) => t.startsWith(DEGRADED_PREFIX));
}

function axisOf(scene: Scene, id: string) {
  return scene.axes.find((a) => a.id === id) ?? fail(`场景里没有轴 ${id}`);
}

/** 文本块一律不许被裁（出现"画布放不下"= 我写的字超出了面板预算，是版面 bug）。 */
function assertNoTruncatedText(scene: Scene, where: string): void {
  const bad = scene.texts.filter((t) => t.text.includes("画布放不下"));
  assert.deepEqual(bad.map((t) => t.text), [], `${where}：文本被裁掉了`);
}

const PROJ = await projectionFrame();
const PROJ_MONO = await projectionFrame({}, "mono");
const CONV = await convergenceFrame();
const EQUI0 = await equioFrame({}, 1);
const EQUI5 = await equioFrame({}, 5);
const ZEROS = await zerosFrame({}, "laguerre", 24);
const ZEROS_LEG = await zerosFrame({}, "legendre", 24);

// --------------------------------------------------------------------------- //
// 1. 装配层：manifest ↔ models ↔ modelFor
// --------------------------------------------------------------------------- //

test("index：四个 viz 都在 manifest 里，且 models 与 manifest 的 id 集合逐项相等", () => {
  const ids = manifest.vizzes.map((v) => v.id).sort();
  assert.deepEqual(ids, [...VIZ_IDS].sort());
  assert.deepEqual(Object.keys(models).sort(), [...VIZ_IDS].sort());
});

test("index：modelFor 取到对应 model，未知 id 返回 null（不抛错）", () => {
  assert.equal(modelFor("projection"), projectionModel);
  assert.equal(modelFor("convergence_rates"), convergenceRatesModel);
  assert.equal(modelFor("equioscillation"), equioscillationModel);
  assert.equal(modelFor("zeros_evolution"), zerosEvolutionModel);
  assert.equal(modelFor("nope"), null);
  assert.equal(modelFor("__proto__"), null);
});

test("manifest：2.3 的四个 viz 都声明了 requiredAux=error 与时间轴语义", () => {
  assert.deepEqual(specOf("projection").requiredAux, ["error"]);
  assert.deepEqual(specOf("convergence_rates").requiredAux, ["error"]);
  assert.equal(specOf("convergence_rates").time.mode, "none");
  assert.equal(specOf("projection").time.mode, "scrub");
  assert.equal(specOf("projection").time.label, "已展开的基函数个数 $k$");
  assert.equal(specOf("equioscillation").time.label, "Remez 迭代轮次 $k$");
  assert.equal(specOf("zeros_evolution").time.label, "最高阶数 $n$");
  assert.deepEqual(specOf("projection").params.find((p) => p.name === "n")?.critical, [22, 23, 24]);
});

// --------------------------------------------------------------------------- //
// 2. projection
// --------------------------------------------------------------------------- //

test("projection：默认帧产出合法 Scene，章节/viz/来源/时间逐项对", () => {
  const scene = projectionModel(PROJ, opts());
  assertValid(scene);
  assert.equal(scene.chapter, "2.3");
  assert.equal(scene.viz, "projection");
  assert.equal(scene.dataSource, "backend");
  assert.equal(scene.t, 12);
  assert.ok(scene.title.includes("投影"));
});

test("projection：三个轴 —— 主轴固定线性，系数/误差轴自适应（规则 3）", () => {
  const scene = projectionModel(PROJ, opts());
  assert.equal(axisOf(scene, "main").autoscaled, false);
  assert.equal(axisOf(scene, "coef").autoscaled, true);
  assert.equal(axisOf(scene, "err").autoscaled, true);
  assert.equal(axisOf(scene, "main").yscale, "linear");
});

test("projection：图层顺序 = f / 目标投影 / 当前投影 + 系数柱 + 目标误差 / 当前误差", () => {
  const scene = projectionModel(PROJ, opts());
  const main = layersOf(scene, "line").filter((l) => l.axes === "main");
  assert.equal(main.length, 3);
  assert.deepEqual(Array.from(seriesOfLayer(scene, "main", 0)[0].y), P_F);
  assert.deepEqual(Array.from(seriesOfLayer(scene, "main", 1)[0].y), P_PROJ12);
  assert.deepEqual(Array.from(seriesOfLayer(scene, "main", 2)[0].y), P_PROJ12);
  assert.equal(layersOf(scene, "bars").length, 1);
  assert.equal(layersOf(scene, "line").filter((l) => l.axes === "err").length, 2);
});

test("projection：k=n 时不画差值线，并明说「跨基对照不在本帧」（不拿 0 冒充实测）", () => {
  const scene = projectionModel(PROJ, opts());
  assert.equal(layersOf(scene, "line").filter((l) => l.axes === "err").length, 2);
  const flat = flatOf(scene);
  assert.ok(flat.includes("跨基对照1.17e-3不在本帧"), textsOf(scene));
  assert.ok(flat.includes("偏差恒为0"), textsOf(scene));
});

test("projection：0<k<n 时画出差值线，最大值来自帧数据（显示空间的两条帧曲线之差）", () => {
  const scene = projectionModel(PROJ_MONO, opts());
  assertValid(scene);
  const errLines = layersOf(scene, "line").filter((l) => l.axes === "err");
  assert.equal(errLines.length, 3, "k<n 时应有第三条差值线");
  const dev = seriesOfLayer(scene, "err", 2)[0].y;
  for (let i = 0; i < dev.length; i += 1) {
    const expect = (P_ERR24_TARGET[i] as number) - (P_ERR24[i] as number);
    assert.ok(Math.abs((dev[i] as number) - expect) < 1e-15);
  }
  assert.ok(textsOf(scene).includes("P_k f − P_n f"));
});

test("projection：κ(G)u > 1 时状态角标点名「系数已不可信」，且残差与 κ 同屏（规则 1/6）", () => {
  const scene = projectionModel(PROJ_MONO, opts());
  const flat = flatOf(scene);
  assert.ok(flat.includes("1.04×10⁻¹¹"), textsOf(scene));
  assert.ok(flat.includes("κ(G)u=21.7"), textsOf(scene));
  assert.ok(flat.includes("1.95×10¹⁷"), textsOf(scene));
  const status = badgesOf(scene, "status").join(" ");
  assert.ok(status.includes("已不可信"), status);
  const critical = badgesOf(scene, "critical").join(" ");
  assert.ok(critical.includes("κ(G)u"), critical);
});

test("projection：正交基不冒充越过临界 —— 角标说明该阈值只对单项式基成立", () => {
  const scene = projectionModel(PROJ, opts());
  assert.ok(badgesOf(scene, "critical").join(" ").includes("临界只对单项式"), badgesOf(scene, "critical").join(" "));
  assert.ok(badgesOf(scene, "status").join(" ").includes("可信"));
});

test("projection：k=0 不画空柱，并写明「零函数」与「补零≠系数为 0」", async () => {
  const frame = await projectionFrame({ meta: { t: 0, step: 0, n_current: 0 }, arrays: {} });
  const scene = projectionModel(frame, opts({ t: 0 }));
  assertValid(scene);
  assert.equal(layersOf(scene, "bars").length, 0);
  const text = textsOf(scene);
  assert.ok(text.includes("零函数"), text);
  assert.ok(text.includes("未展开"), text);
});

test("projection：系数柱只画前 k 个（i ≥ k 是后端补零，不当成「系数为 0」）", () => {
  const scene = projectionModel(PROJ, opts());
  const bars = layersOf(scene, "bars")[0];
  assert.equal(bars.x.length, 12);
  assert.equal(bars.values.length, 12);
});

test("projection：字面照登帧的 note 与 singular_note（原样链，不自己判基的好坏）", async () => {
  const scene = projectionModel(PROJ_MONO, opts());
  assert.ok(textsOf(scene).includes("κ(G) 已达 1e17"));
  assert.ok(textsOf(scene).includes("单项式基在 n 大时迅速变坏"));
});

test("projection：时间轴按帧 meta.step 显示；t 越界时写明被夹住（不假装 t 就是索引）", async () => {
  const frame = await projectionFrame({ meta: { t: 24, step: 12 } });
  const scene = projectionModel(frame, opts({ t: 24 }));
  const text = textsOf(scene);
  assert.ok(text.includes("请求 t=24"), text);
  assert.ok(text.includes("不假装 t 就是索引"), text);
});

// --------------------------------------------------------------------------- //
// 3. convergence_rates
// --------------------------------------------------------------------------- //

test("convergence_rates：默认帧产出合法 Scene；主轴双对数且固定（图题写明双对数）", () => {
  const scene = convergenceRatesModel(CONV, opts());
  assertValid(scene);
  const main = axisOf(scene, "main");
  assert.equal(main.xscale, "log");
  assert.equal(main.yscale, "log");
  assert.equal(main.autoscaled, false);
  assert.equal(axisOf(scene, "rate").autoscaled, true);
  assert.ok(main.title.includes("双对数"));
  assert.ok(main.ylim[0] > 0 && main.xlim[0] > 0, "对数轴必须严格为正");
});

test("convergence_rates：主轴图层 = 三条实测 + 三条理论虚线（实测与理论都要画）", () => {
  const scene = convergenceRatesModel(CONV, opts());
  const lines = layersOf(scene, "line").filter((l) => l.axes === "main");
  assert.equal(lines.length, 6);
  assert.deepEqual(Array.from(seriesOfLayer(scene, "main", 0)[0].y), C_AN);
  assert.deepEqual(Array.from(seriesOfLayer(scene, "main", 3)[0].y), C_THEORY_AN);
  const dashed = lines.filter((l) => l.series[0].dash !== undefined);
  assert.equal(dashed.length, 3, "理论参考线必须是虚线");
});

test("convergence_rates：两条幂律曲线用 loglogfit 层画理论直线（规则 2），且两条斜率都有限", () => {
  const scene = convergenceRatesModel(CONV, opts());
  const fits = layersOf(scene, "loglogfit");
  assert.equal(fits.length, 2);
  for (const f of fits) {
    assert.equal(f.axes, "rate");
    assert.ok(Number.isFinite(f.measuredSlope), "实测斜率必须画出来");
    assert.ok(Number.isFinite(f.theorySlope), "理论斜率必须画出来");
    assert.ok(f.h.length >= 2 && f.err.length === f.h.length);
  }
  const jump = fits.find((f) => f.seriesLabel.includes("jump")) ?? fail("缺 jump 的 loglogfit 层");
  assert.ok(Math.abs(jump.measuredSlope + 0.492386) < 1e-9);
  assert.ok(Math.abs(jump.theorySlope + 0.5) < 1e-12);
});

test("convergence_rates：解析情形不用直线参考（ρ^(−n) 在双对数下是弯的），用后端曲线", () => {
  const scene = convergenceRatesModel(CONV, opts());
  const rateLines = layersOf(scene, "line").filter((l) => l.axes === "rate");
  assert.equal(rateLines.length, 2, "解析情形应是 fit_analytic + theory_analytic 两条曲线");
  assert.ok(rateLines[0].series[0].y.length >= 30);
});

test("convergence_rates：窗口起点来自 meta，横轴覆盖最早的窗口（不把晚开窗的点挤出轴外）", () => {
  const scene = convergenceRatesModel(CONV, opts());
  const rate = axisOf(scene, "rate");
  assert.ok(rate.xlim[0] <= 2, `rate 轴左端应 ≤ 最早窗口起点 2，实际 ${rate.xlim[0]}`);
  assert.equal(scene.overflow.flag, false, scene.overflow.note);
  assert.ok(textsOf(scene).includes("解析 n ≥ 3"));
  assert.ok(textsOf(scene).includes("尖点 n ≥ 8"));
});

test("convergence_rates：实测/理论/相对差/R² 四个数同屏（斜率不是自己算的）", () => {
  const text = textsOf(convergenceRatesModel(CONV, opts()));
  assert.ok(text.includes("-0.198167"), text);
  assert.ok(text.includes("-0.19869") || text.includes("0.19869"), text);
  assert.ok(text.includes("0.263%"), text);
  assert.ok(text.includes("0.99792"), text);
  assert.ok(text.includes("1.219804"), text);
});

test("convergence_rates：两条幂律的理论指数逐字来自 strings.regime_*（不重算）", () => {
  const text = textsOf(convergenceRatesModel(CONV, opts()));
  assert.ok(text.includes("algebraic α=3/2（f′ 有跳变）"), text);
  assert.ok(text.includes("algebraic α=1/2（f 本身有跳变）"), text);
});

test("convergence_rates：没触底就不画地板横线，但触底点数照写（规则 4）", () => {
  const scene = convergenceRatesModel(CONV, opts());
  const mainLines = layersOf(scene, "line").filter((l) => l.axes === "main");
  assert.equal(mainLines.length, 6, "没有点触底时不应多出地板横线");
  assert.ok(textsOf(scene).includes("触底点数 解析/尖点/跳点 = 0/0/0"));
});

test("convergence_rates：有点触底时画出地板横线并挂状态角标", async () => {
  const frame = await convergenceFrame({
    arrays: { error_cusp: C_CUSP.map(() => 1.1102230246251565e-16) },
    meta: { clipped_cusp: 7 },
  });
  const scene = convergenceRatesModel(frame, opts());
  assertValid(scene);
  const mainLines = layersOf(scene, "line").filter((l) => l.axes === "main");
  assert.equal(mainLines.length, 7);
  assert.ok(badgesOf(scene, "status").join(" ").includes("触地板"));
});

test("convergence_rates：没有时间轴 —— 不画游标、不引用 tMax", () => {
  const scene = convergenceRatesModel(CONV, opts({ t: 999 }));
  assert.equal(scene.layers.some((l) => l.axes === "__time__"), false);
  assert.ok(!textsOf(scene).includes("时间轴"));
});

test("convergence_rates：字面照登 fit_note / floor_note", () => {
  const text = textsOf(convergenceRatesModel(CONV, opts()));
  assert.ok(text.includes("常数项不可知，斜率才是理论断言"), text);
  assert.ok(text.includes("log 轴不能画 0"), text);
});

// --------------------------------------------------------------------------- //
// 4. equioscillation
// --------------------------------------------------------------------------- //

test("equioscillation：默认帧产出合法 Scene；主轴固定线性、副轴自适应对数", () => {
  const scene = equioscillationModel(EQUI5, opts());
  assertValid(scene);
  assert.equal(axisOf(scene, "main").autoscaled, false);
  assert.equal(axisOf(scene, "alt").autoscaled, true);
  assert.equal(axisOf(scene, "alt").yscale, "log");
  assert.ok(axisOf(scene, "alt").ylim[0] > 0);
});

test("equioscillation：图层 = 等幅带 / 零线 / 收敛曲线 / L² 对照 / 当前曲线 / 交错点 + 副轴三组点", () => {
  const scene = equioscillationModel(EQUI5, opts());
  assert.equal(layersOf(scene, "band").length, 1);
  const mainLines = layersOf(scene, "line").filter((l) => l.axes === "main");
  assert.equal(mainLines.length, 4);
  assert.equal(layersOf(scene, "scatter").filter((l) => l.axes === "main").length, 1);
  assert.equal(layersOf(scene, "scatter").filter((l) => l.axes === "alt").length, 2);
});

test("equioscillation：等幅带宽度取帧的 max_error_final（不是前端算的最大值）", () => {
  const scene = equioscillationModel(EQUI5, opts());
  const band = layersOf(scene, "band")[0];
  assert.ok(Math.abs((band.band.yhi[0] as number) - 0.0980881) < 1e-12);
  assert.ok(Math.abs((band.band.ylo[0] as number) + 0.0980881) < 1e-12);
});

test("equioscillation：L² 对照曲线与 max|e|/L² 两个数同屏（最小二乘≠最小最大）", () => {
  const scene = equioscillationModel(EQUI5, opts());
  assert.deepEqual(Array.from(seriesOfLayer(scene, "main", 2)[0].y), Array.from(E_L2));
  const text = textsOf(scene);
  assert.ok(text.includes("9.81×10⁻²"));
  assert.ok(text.includes("1.5×10⁻¹"));
  assert.ok(text.includes("1.525"));
});

test("equioscillation：converged=false 与 alt_signs_alternate=false 时逐条说明（不隐藏）", async () => {
  const frame = await equioFrame({ meta: { converged: false, alt_signs_alternate: false } }, 5);
  const scene = equioscillationModel(frame, opts());
  assertValid(scene);
  const text = textsOf(scene);
  assert.ok(text.includes("converged=false"), text);
  assert.ok(text.includes("没有拿到"), text);
  assert.ok(text.includes("alt_signs_alternate=false"), text);
  assert.ok(badgesOf(scene, "status").join(" ").includes("未拿到最优性证书"));
});

test("equioscillation：amplitude_ratio=null 时照原样写「无有限值」，不写成 1 也不写成 0", () => {
  const scene = equioscillationModel(EQUI0, opts({ t: 1 }));
  assertValid(scene);
  const text = textsOf(scene);
  assert.ok(text.includes("无有限值"), text);
  const flat = flatOf(scene);
  assert.ok(flat.includes("**不是**1，也**不是**0"), text);
  assert.ok(badgesOf(scene, "status").join(" ").includes("无有限值"));
  // 不能出现 "振幅比 ... = 1" 这种把 null 当成 1 的说法
  assert.equal(/振幅比[^；\n]*当前轮 = 1[，、\s]/.test(text), false);
});

test("equioscillation：|e| ≤ 地板的交错点画在地板上并报数（对数轴画不到 0）", () => {
  const scene = equioscillationModel(EQUI0, opts({ t: 1 }));
  const altScatters = layersOf(scene, "scatter").filter((l) => l.axes === "alt");
  assert.equal(altScatters.length, 2, "地板上的点 + 最终点（没有地板之上的点）");
  assert.ok(textsOf(scene).includes("≤ u"));
});

test("equioscillation：交错点数与 deg+2 同屏，并挂定理角标（规则 6）", () => {
  const scene = equioscillationModel(EQUI5, opts());
  assert.ok(textsOf(scene).includes("定理要求 deg+2 = 10"));
  assert.ok(badgesOf(scene, "critical").join(" ").includes("deg+2 = 10"));
});

test("equioscillation：字面照登 converged_note / certificate_note / floor_note", () => {
  const text = textsOf(equioscillationModel(EQUI5, opts()));
  assert.ok(text.includes("交错点集稳定且被最优性证书确认"), text);
  assert.ok(text.includes("de la Vallée Poussin"), text);
  assert.ok(text.includes("掉到地板后交错点全是噪声"), text);
});

test("equioscillation：时间轴按 meta.step 显示，请求 t 超出 n_steps 时写明被夹住", async () => {
  const scene = equioscillationModel(EQUI5, opts({ t: 24 }));
  const text = textsOf(scene);
  assert.ok(text.includes("k = 5 / 5"), text);
  assert.ok(text.includes("请求 t=24"), text);
});

// --------------------------------------------------------------------------- //
// 5. zeros_evolution
// --------------------------------------------------------------------------- //

test("zeros_evolution：默认帧产出合法 Scene；主轴固定线性、副轴自适应对数", () => {
  const scene = zerosEvolutionModel(ZEROS, opts({ t: 24 }));
  assertValid(scene);
  assert.equal(axisOf(scene, "main").autoscaled, false);
  assert.equal(axisOf(scene, "resid").autoscaled, true);
  assert.equal(axisOf(scene, "resid").yscale, "log");
  assert.ok(axisOf(scene, "resid").ylim[0] > 0);
});

test("zeros_evolution：主面板图层 = 密度 / 间距 / 对照点 / 零点 / 当前阶零点 / 游标", () => {
  const scene = zerosEvolutionModel(ZEROS, opts({ t: 24 }));
  const mainLines = layersOf(scene, "line").filter((l) => l.axes === "main");
  assert.equal(mainLines.length, 3, "密度 + 间距 + 游标");
  assert.equal(layersOf(scene, "scatter").filter((l) => l.axes === "main").length, 3);
});

test("zeros_evolution：密度曲线是横向映射（x 落在横轴左侧 20%，y 是零点位置）", () => {
  const scene = zerosEvolutionModel(ZEROS, opts({ t: 24 }));
  const density = seriesOfLayer(scene, "main", 0)[0];
  const lim = axisOf(scene, "main").xlim;
  const span = lim[1] - lim[0];
  for (const v of Array.from(density.x)) {
    assert.ok(v >= lim[0] - 1e-9 && v <= lim[0] + 0.2 * span + 1e-9, `密度点 x=${v} 不在左侧 20% 内`);
  }
  assert.deepEqual(Array.from(density.y), Array.from(linspace(0.02, 120, 241)));
});

test("zeros_evolution：间距曲线用 gap_index 作横轴（不是阶数），并给出 max/min 比", () => {
  const scene = zerosEvolutionModel(ZEROS, opts({ t: 24 }));
  const gapLayer = layersOf(scene, "line").filter((l) => l.axes === "main")[1];
  assert.equal(gapLayer.series[0].x.length, 23, "gap 只覆盖当前阶的 23 个间距（与后端一致）");
  assert.deepEqual(Array.from(gapLayer.series[0].x), Array.from({ length: 23 }, (_, i) => i + 1));
  assert.ok(textsOf(scene).includes("max/min = 92.34"));
  assert.ok(textsOf(scene).includes("零点不是等距的"));
});

test("zeros_evolution：误差量的身份写在四处（轴名/角标/文本/帧 note）——不是 |p_n|", () => {
  const scene = zerosEvolutionModel(ZEROS, opts({ t: 24 }));
  assert.ok(axisOf(scene, "resid").title.includes("p_n/p_n"), axisOf(scene, "resid").title);
  assert.ok(axisOf(scene, "resid").title.includes("根误差"), axisOf(scene, "resid").title);
  assert.ok(badgesOf(scene, "status").join(" ").includes("|p_n/p_n′|"));
  const text = textsOf(scene);
  assert.ok(text.includes("Newton 修正量"), text);
  assert.ok(flatOf(scene).includes("残差量|p_n(x_k)|不在本帧里"), text);
  assert.ok(text.includes("Laguerre 的 |L_n| 可以到 1e14"), text);
});

test("zeros_evolution：地板横线与触底点数（Laguerre 104 个点落在地板上）", () => {
  const scene = zerosEvolutionModel(ZEROS, opts({ t: 24 }));
  const residLines = layersOf(scene, "line").filter((l) => l.axes === "resid");
  assert.equal(residLines.length, 1, "地板横线");
  assert.ok(textsOf(scene).includes("落在地板（≤ u）上的根误差 104 个"));
});

test("zeros_evolution：n=1 的退化跨度按 ±1 扩展并明说「这是扩展」；没有间距就不画", async () => {
  const frame = await zerosFrame({ meta: { t: 1, step: 1, n_current: 1, n_roots_current: 1, root_span_lo: 0, root_span_hi: 0 } }, "legendre", 1);
  const scene = zerosEvolutionModel(frame, opts({ t: 1 }));
  assertValid(scene);
  const mainLines = layersOf(scene, "line").filter((l) => l.axes === "main");
  assert.equal(mainLines.length, 2, "没有间距曲线，只剩密度与游标");
  const text = textsOf(scene);
  assert.ok(text.includes("退化"), text);
  assert.ok(text.includes("不是实测跨度"), text);
});

test("zeros_evolution：字面照登 error_note / density_note / spacing_note", () => {
  const text = textsOf(zerosEvolutionModel(ZEROS, opts({ t: 24 })));
  assert.ok(text.includes("与多项式自身尺度无关"), text);
  assert.ok(text.includes("平衡测度"), text);
  assert.ok(text.includes("max gap / min gap"), text);
});

test("zeros_evolution：Legendre 全在地板时如实写出（不是「误差为 0」）", () => {
  const scene = zerosEvolutionModel(ZEROS_LEG, opts({ t: 24 }));
  assertValid(scene);
  assert.ok(textsOf(scene).includes("落在地板（≤ u）上的根误差 300 个"));
  assert.ok(textsOf(scene).includes("6.262"));
});

test("zeros_evolution：当前阶的零点/误差单独高亮，游标落在 meta.step 上", () => {
  const scene = zerosEvolutionModel(ZEROS, opts({ t: 24 }));
  const cursor = layersOf(scene, "line").filter((l) => l.axes === "main")[2];
  assert.deepEqual(Array.from(cursor.series[0].x), [24, 24]);
  const residScatters = layersOf(scene, "scatter").filter((l) => l.axes === "resid");
  assert.equal(residScatters.length, 2, "全部根误差 + 当前阶高亮");
});

// --------------------------------------------------------------------------- //
// 6. 规则与降级（四个 viz 共同守的几条）
// --------------------------------------------------------------------------- //

test("规则 5：四个 viz 都挂了 source 角标，且文本里没有第二处手写的来源", () => {
  for (const scene of [
    projectionModel(PROJ, opts()),
    convergenceRatesModel(CONV, opts()),
    equioscillationModel(EQUI5, opts()),
    zerosEvolutionModel(ZEROS, opts({ t: 24 })),
  ]) {
    const badges = badgesOf(scene, "source");
    assert.equal(badges.length, 1, `${scene.viz} 应有且仅有一个 source 角标`);
    assert.ok(badges[0]?.includes("后端"), badges[0]);
  }
});

test("规则 3：每个自适应的副轴都有 axismode/autoscale 角标点名「自适应」", () => {
  for (const scene of [
    projectionModel(PROJ, opts()),
    convergenceRatesModel(CONV, opts()),
    equioscillationModel(EQUI5, opts()),
    zerosEvolutionModel(ZEROS, opts({ t: 24 })),
  ]) {
    const auto = scene.axes.filter((a) => a.autoscaled);
    const notes = [...badgesOf(scene, "autoscale"), ...badgesOf(scene, "axismode")].join(" ");
    assert.ok(auto.length >= 1, `${scene.viz} 应至少有一个自适应轴`);
    assert.ok(notes.includes("自适应"), `${scene.viz} 的自适应轴没有被点名`);
  }
});

test("规则 7：scene.pitfalls 逐字来自 frame.pitfalls（model 不自己编）", () => {
  const scene = projectionModel(PROJ, opts());
  assert.equal(scene.pitfalls, PITFALLS);
});

test("规则 7：帧 pitfalls 太短 → 抛 ModelInputError", async () => {
  const frame = await projectionFrame({ pitfalls: "太短" });
  assert.throws(() => projectionModel(frame, opts()), ModelInputError);
});

test("缺必需数组 → 抛 ModelInputError（画不出来就不画半张图）", async () => {
  const cases: [string, () => Promise<Frame>][] = [
    ["projection/x", () => projectionFrame({ drop: ["x"] })],
    ["projection/error_target", () => projectionFrame({ drop: ["error_target"] })],
    ["convergence_rates/n", () => convergenceFrame({ drop: ["n"] })],
    ["convergence_rates/error_cusp", () => convergenceFrame({ drop: ["error_cusp"] })],
    ["equioscillation/alt_x", () => equioFrame({ drop: ["alt_x"] })],
    ["equioscillation/error_l2", () => equioFrame({ drop: ["error_l2"] })],
    ["zeros_evolution/degree", () => zerosFrame({ drop: ["degree"] })],
    ["zeros_evolution/root_current", () => zerosFrame({ drop: ["root_current"] })],
  ];
  for (const [name, make] of cases) {
    const frame = await make();
    const model = name.startsWith("projection")
      ? projectionModel
      : name.startsWith("convergence")
        ? convergenceRatesModel
        : name.startsWith("equioscillation")
          ? equioscillationModel
          : zerosEvolutionModel;
    assert.throws(() => model(frame, opts({ t: 24 })), ModelInputError, `${name} 缺数组时应抛错`);
  }
});

test("数组长度不一致 → 抛 ModelInputError（错位的图比不画更坏）", async () => {
  const frame = await projectionFrame({ arrays: { error: P_ERR12.slice(0, 30) } });
  assert.throws(() => projectionModel(frame, opts()), ModelInputError);
  const z = await zerosFrame({ arrays: { root: zerosArrays("laguerre", 24).root.slice(0, 10) } });
  assert.throws(() => zerosEvolutionModel(z, opts({ t: 24 })), ModelInputError);
});

test("meta 缺标量 → 文本写「?」+ 降级角标，绝不写 0 顶替", async () => {
  const frame = await projectionFrame({ dropMeta: ["gram_cond", "normal_residual", "l2_error_current"] });
  const scene = projectionModel(frame, opts());
  assertValid(scene);
  const text = textsOf(scene);
  assert.ok(text.includes("κ(G) = ?"), text);
  assert.ok(text.includes("残差 ‖Gc−b‖ = ?"), text);
  assert.equal(text.includes("κ(G) = 0"), false);
  assert.ok(degradedTexts(scene).length >= 1, "应挂降级角标");
  assert.ok(degradedTexts(scene).join(" ").includes("gram_cond"), degradedTexts(scene).join(" "));
});

test("meta 缺 n_terms / step → 降级说明 + 按请求 t 夹取并点明「不一定是帧真正算的那一档」", async () => {
  const frame = await projectionFrame({ dropMeta: ["n_terms", "step", "n_current"] });
  const scene = projectionModel(frame, opts({ t: 5 }));
  assertValid(scene);
  const degraded = degradedTexts(scene).join(" ");
  assert.ok(degraded.includes("n_terms"), degraded);
  assert.ok(flatOf(scene).includes("不一定是帧真正算的那一档"), textsOf(scene));
});

test("meta 的 null（amplitude_ratio_*=null）不算缺失，不算 0，也不进降级列表", () => {
  const scene = equioscillationModel(EQUI0, opts({ t: 1 }));
  const degraded = degradedTexts(scene).join(" ");
  assert.equal(degraded.includes("amplitude_ratio"), false, degraded);
});

test("equioscillation：abs_error 不是 2 维时降级（不画包络），其余照画", async () => {
  const frame = await equioFrame({ shapes: { abs_error: [1203] } });
  const scene = equioscillationModel(frame, opts());
  assertValid(scene);
  assert.equal(layersOf(scene, "line").filter((l) => l.axes === "alt").length, 0, "包络线不画");
  assert.ok(degradedTexts(scene).length >= 1, "应挂降级角标");
  assert.ok(degradedTexts(scene).join(" ").includes("abs_error"), degradedTexts(scene).join(" "));
});

test("rule 8：四张图的文本块都不许被裁（不出现「画布放不下」）", () => {
  const scenes = [
    projectionModel(PROJ, opts()),
    projectionModel(PROJ_MONO, opts()),
    convergenceRatesModel(CONV, opts()),
    equioscillationModel(EQUI5, opts()),
    equioscillationModel(EQUI0, opts({ t: 1 })),
    zerosEvolutionModel(ZEROS, opts({ t: 24 })),
    zerosEvolutionModel(ZEROS_LEG, opts({ t: 24 })),
  ];
  for (const scene of scenes) assertNoTruncatedText(scene, scene.viz);
});

test("数组里的 NaN/∞ 不进图层，但计数进溢出角标（规则 4）", async () => {
  const bad = [...P_F];
  bad[3] = Number.NaN;
  bad[7] = Number.POSITIVE_INFINITY;
  const frame = await projectionFrame({ arrays: { f: bad } });
  const scene = projectionModel(frame, opts());
  assertValid(scene);
  assert.equal(scene.overflow.flag, true);
  assert.ok(badgesOf(scene, "overflow").length === 1);
  assert.ok(scene.overflow.note.includes("非有限") || scene.overflow.note.includes("NaN"), scene.overflow.note);
});

test("窄画布（单列角标）也能产出合法 Scene，文本不越界", () => {
  const scene = projectionModel(PROJ, opts({ width: 760, height: 560 }));
  assertValid(scene);
  const conv = convergenceRatesModel(CONV, opts({ width: 760, height: 560 }));
  assertValid(conv);
  const equi = equioscillationModel(EQUI5, opts({ width: 760, height: 560 }));
  assertValid(equi);
  const zeros = zerosEvolutionModel(ZEROS, opts({ width: 760, height: 560, t: 24 }));
  assertValid(zeros);
});

test("dark 主题只改颜色，不改结构", () => {
  const light = projectionModel(PROJ, opts({ theme: "light" }));
  const dark = projectionModel(PROJ, opts({ theme: "dark" }));
  assert.equal(light.layers.length, dark.layers.length);
  assert.deepEqual(
    light.layers.map((l) => l.kind),
    dark.layers.map((l) => l.kind),
  );
  const firstColor = (scene: Scene) => layersOf(scene, "line")[0]?.series[0].color ?? "";
  assert.notEqual(firstColor(light), firstColor(dark));
});

test("scene.t 一律取帧 meta.t（帧说了算），缺 meta.t 时才回落 opts.t", async () => {
  const withT = projectionModel(PROJ, opts({ t: 3 }));
  assert.equal(withT.t, 12);
  const frame = await projectionFrame({ dropMeta: ["t"] });
  assert.equal(projectionModel(frame, opts({ t: 3 })).t, 3);
});

test("dataSource=backend+preview 时另挂 preview 角标，source 仍在（规则 5）", async () => {
  const frame = await projectionFrame({ strings: { data_source: "backend+preview" } });
  const scene = projectionModel(frame, opts());
  assert.equal(scene.dataSource, "backend+preview");
  assert.equal(badgesOf(scene, "source").length, 1);
  assert.equal(badgesOf(scene, "preview").length, 1);
});
