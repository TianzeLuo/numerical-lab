/**
 * 1.4 章（特征值、伪谱与非正规性）model 层的测试。
 *
 * 夹具：**手搓帧**（`new Frame(...)` → `encodeNlf` → `decodeNlf`），字段集合（数组名 / shape /
 * meta 键 / strings 键 / critical 键）与 `server/chapters/1.4.py` 的真实帧逐项一致，
 * 数值也按真实帧的量级抄（jordan-10 的 σ_min 云雾、shear-10 的 1.7e9 瞬态、
 * grcar-10 的 3.24e4 谱界 vs 7.7e-33 残差、QR 的 −300 地板）。
 * **本文件不需要 Python / numpy**，`npm test` 在任何机器上都能跑。
 *
 * 这里守的是什么（对应 1.4 最容易被画错的地方）
 * --------------------------------------------
 * | 断言 | 防的是 |
 * |---|---|
 * | `trusted=0` 的格点在热力图里是 NaN，且个数 = `n_noise_points` | 把"σ_min 掉进舍入地板"画成"σ_min 很小"（本章第二个大坑） |
 * | 不可信格点另有一层标记 + 文本/`overflow` 都报数 | 掩码点被静默丢掉（规则 1 的红线） |
 * | `log10_sigma = −300` 的格点被夹到色阶下限并计数 | 300 个数量级把整张图的对比度拉平 |
 * | ρ(A)=0 的帧改画绝对半径 ρ_ε，且轴名/角标/文本三处都说明 | "比值的分母是 0"被当成"比值等于地板" |
 * | ρ(A)>0 的帧必须同时画实测比值与解析对照 `radius_ratio_exact` | 规则 2 的第二条线被省掉 |
 * | `error` 场恒为地板时照样画出来 + 文本点明字段没信息 | 悄悄不画 requiredAux，或替后端编一个"误差场" |
 * | 主轴窗口恒为 ±1.05 / ±1.5（`autoscaled=false`） | 自适应把"云雾扩散/向量对齐"抹平成"图还是那么大" |
 * | 越出固定窗口的谱点计数进 `overflow` | 谱点被裁掉却账上无记录 |
 * | `error`/`residual` 里的精确 0 与 1e-300 地板：显式丢弃或贴轴底 + 计数 | 对数轴上把"没有这个值"画成"值很小" |
 * | 比值轴上恒画参考线 y=1（= 帧里的 `spectral_prediction`） | 规则 2 的第二条斜率缺失，ρ(A) 的"预测"看不见 |
 * | `meta.non_normal=False`（shear，ρ=1）不许当成结论 | 用一个布尔量代替看数据 |
 * | 缺必需数组 / pitfalls 太短 → `ModelInputError` | 画半张图、或 model 自己编 pitfalls（规则 7） |
 * | 窄画布 480×600 仍合规；同帧两次 `sceneToJson` 逐字节相同 | 布局硬编码 960 宽 / 结果不确定 |
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { linspace } from "../shared/src/axes.js";
import { Frame, type ArraySpec, type Dtype, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { sceneToJson, validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import { manifest, modelFor, models } from "../chapters/1.4/index.js";

// --------------------------------------------------------------------------- //
// 0. 夹具（与 `viz/tests/chapter15.test.ts` 同一套做法）
// --------------------------------------------------------------------------- //

const VIZ_IDS = ["power_iteration", "qr_schur", "pseudospectrum", "transient_growth", "gmres_pseudospec"] as const;
type VizId = (typeof VIZ_IDS)[number];

const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };

/** `manifest.requiredAux` 的每个数组名画在哪根轴上（少了就是"辅助量没露面"）。 */
const AUX_AXIS: Record<VizId, Record<string, string>> = {
  power_iteration: { residual: "aux" },
  qr_schur: { error: "aux" },
  pseudospectrum: { error: "err" },
  transient_growth: { error: "err" },
  gmres_pseudospec: { residual: "aux" },
};

/** 每个 viz 被 `requireArray` 拦下的第一根数组（删掉它必须抛 `ModelInputError`）。 */
const REQUIRED_ARRAY: Record<VizId, string> = {
  power_iteration: "vector_x",
  qr_schur: "subdiag_heat",
  pseudospectrum: "log10_sigma",
  transient_growth: "growth_ratio",
  gmres_pseudospec: "residual",
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
      unit: a.unit ?? "1",
      label: a.label ?? a.name,
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
  shapes?: Record<string, number[]>;
  drop?: string[];
  meta?: Record<string, number | boolean | null>;
  dropMeta?: string[];
  strings?: Record<string, string>;
  critical?: Record<string, number | number[]>;
  pitfalls?: string;
}

function mergeArrays(base: ArrayDef[], over: Over): ArrayDef[] {
  return base
    .filter((a) => !(over.drop ?? []).includes(a.name))
    .map((a) =>
      over.arrays?.[a.name] !== undefined
        ? {
            ...a,
            data: over.arrays[a.name] as ArrayLike<number>,
            shape: over.shapes?.[a.name] ?? [(over.arrays[a.name] as ArrayLike<number>).length],
          }
        : a,
    );
}

function build(base: FrameDef, over: Over = {}): FrameDef {
  return {
    chapter: base.chapter,
    viz: base.viz,
    arrays: mergeArrays(base.arrays, over),
    meta: { ...omit(base.meta, over.dropMeta ?? []), ...(over.meta ?? {}) },
    strings: { ...base.strings, ...(over.strings ?? {}) },
    critical: { ...base.critical, ...(over.critical ?? {}) },
    pitfalls: over.pitfalls ?? base.pitfalls,
  };
}

// —— 共用 pitfalls（与 `chapters/1.4/pitfalls.md` 同口径的删节版，长度足够过 10 字门槛）—— //
const PIT = {
  power:
    "三条是容易读错的地方。其一：残差下降 ≠ 特征向量准——Rayleigh 商是 O(θ²) 而向量只有 O(θ)。其二：shear 调大后开头几十步看起来不像收敛（残差先涨后落）。其三：主轴的坐标轴固定在 ±1.05，不会自适应。",
  qr: "第一条：纵轴只有 n−1 行，收敛与否读 status/converged，不是读颜色。第二条：谱里有复共轭对时它们只能停在 2×2 块上，别读成卡住了。第三条：色阶固定 [−16, 0]，−300 是地板不是数据。",
  ps: "四条是读错的常见方式。其一：坐标轴固定 ±1.5，自适应会把云雾扩散抹平。其二：σ_min 在非正规矩阵上不可信，掉进舍入地板的格点要被掩掉。其三：正规矩阵的伪谱是 ε-圆盘的并。其四：网格分辨率会系统性低估 ρ_ε。",
  transient:
    "最重要的是三条：谱半径小于 1 不保证 ‖A^k‖ 单调下降；把 nonnormal 拖到 1 以上，副轴上那条预测线会一路向下而实测先冲高再回落；meta 里的 peak_index 贴着 k_max 说明峰值可能还在后面。",
  gmres:
    "两条容易读错：GMRES 的残差单调不增是指每一次都在扩大的 Krylov 空间里取最小残差；只依赖谱的界与实测残差之间的差距巨大，谱说会变糟而实际在变好。固定窗口外的谱点要被如实报数。",
} as const;

// --------------------------------------------------------------------------- //
// 1. 五个 viz 的夹具
// --------------------------------------------------------------------------- //

/** `power_iteration`：单位圆 181 点 + 25 步迭代（末点残差在地板 1e-300 上）。 */
function powerFrame(over: Over & { ratio?: number; converged?: number } = {}): FrameDef {
  const ratio = over.ratio ?? 0.6;
  const steps = 24;
  const circle = linspace(0, 2 * Math.PI, 181);
  const ray = linspace(0, 1, steps + 1);
  const vecX = ray.map((r, j) => Math.cos(0.12 * j) * (0.2 + 0.8 * r));
  const vecY = ray.map((r, j) => Math.sin(0.12 * j) * (0.2 + 0.8 * r));
  const res = Array.from({ length: steps + 1 }, (_, j) => (j === steps ? 1e-300 : ratio ** (2 * j)));
  const base = build(
    {
      chapter: "1.4",
      viz: "power_iteration",
      arrays: [
        { name: "circle_x", data: circle.map((a) => Math.cos(a)), label: "单位圆 x" },
        { name: "circle_y", data: circle.map((a) => Math.sin(a)), label: "单位圆 y" },
        { name: "vector_x", data: vecX, label: "迭代向量投影 x" },
        { name: "vector_y", data: vecY, label: "迭代向量投影 y" },
        { name: "step_grid", data: linspace(0, steps, steps + 1), label: "迭代步" },
        { name: "rayleigh", data: ray.map((r) => 1 - (1 - ratio) * (1 - r)), label: "Rayleigh 商" },
        { name: "residual", data: res, label: "残差" },
        { name: "theory_rate", data: ray.map((_, j) => ratio ** j), label: "理论率" },
        { name: "theory_rate_sq", data: ray.map((_, j) => ratio ** (2 * j)), label: "理论率平方" },
        { name: "error", data: [1e-300], label: "当前残差" },
      ],
      meta: {
        family_key: 0,
        n: 8,
        step: steps,
        n_steps: steps,
        ratio,
        shear: 0,
        lambda1: 1,
        lambda2: ratio,
        ratio_measured: ratio,
        rate_measured: ratio,
        converged: over.converged ?? 1,
        complex_pair: 0,
        angle_step: 0,
        rayleigh_k: 1,
        rayleigh_error: 0.0131,
        residual_k: 6.02e-15,
        residual_min: 6.02e-15,
        residual_max: 0.2731,
        transient_ratio: 4.53e13,
        overflow: false,
        t: steps,
      },
      strings: {
        family: "diagonal",
        status: "converge：24 步后残差 6.02e-15（|λ₂/λ₁| = 0.60）",
        rate_note: "理论 |λ₂/λ₁| = 0.600，实测残差比 0.600",
        data_source: "backend",
        params_json: "{}",
      },
      critical: { ratio: 1.0 },
      pitfalls: PIT.power,
    },
    over,
  );
  return base;
}

/** `qr_schur`：7×12 的次对角元热力图（半数是 −300 地板）+ Ritz 距离（末段贴地板）。 */
function qrFrame(over: Over = {}): FrameDef {
  const n = 8;
  const rows = n - 1;
  const cols = 12;
  const heat = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      // 第 i 条次对角元在第 2i+2 步之后被压到机器精度（真实帧里是 −300）
      heat[i * cols + j] = j >= 2 * i + 2 ? -300 : Math.log10(1 - 0.05 * j);
    }
  }
  const err = Array.from({ length: cols }, (_, j) => (j >= 7 ? 1e-300 : 0.31 * 0.91 ** j));
  return build(
    {
      chapter: "1.4",
      viz: "qr_schur",
      arrays: [
        { name: "subdiag_heat", data: heat, shape: [rows, cols], label: "log10|次对角元|" },
        { name: "subdiag_current", data: heat.slice(rows * (cols - 1)), label: "当前步次对角元" },
        { name: "subdiag_final", data: heat.slice(rows * (cols - 1)), label: "末步次对角元" },
        { name: "error", data: err, label: "Ritz 距离" },
        { name: "error_mark", data: [1e-300], label: "当前谱距离" },
        { name: "step_grid", data: linspace(0, cols - 1, cols), label: "迭代步" },
        { name: "theory_rate", data: Array.from({ length: cols }, (_, j) => 0.91 ** j), label: "理论衰减" },
      ],
      meta: {
        n,
        step: cols - 1,
        n_steps: cols - 1,
        shift_key: 2,
        family_key: 0,
        n_iter: cols - 1,
        converged: true,
        n_deflations: rows,
        n_blocks: rows,
        spectral_gap: 0.91047644569758,
        error_current: 0,
        error_final: 0,
        subdiag_final_max: 0,
        subdiag_initial_max: 1,
        is_complex_spectrum: false,
        overflow: false,
        t: 60,
      },
      strings: {
        family: "symmetric",
        shift: "wilkinson",
        status: "converge：11 步、7 次 deflation，谱距离 0.00e+00",
        note: "迭代 11 步、deflation 7 次、块结构 [1, 1, 1, 1, 1, 1, 1]",
        data_source: "backend",
        params_json: "{}",
      },
      critical: {},
      pitfalls: PIT.qr,
    },
    over,
  );
}

/**
 * `pseudospectrum`：8×8 网格、3 个 ε。`rho=0`（jordan）时比值全是地板 → 模型应改画绝对半径；
 * `rho>0` 时比值可用（走 manifest 的"ρ_ε/ρ(A)"口径）。`untrusted` 个格点掉进舍入地板。
 */
function psFrame(over: Over & { rho?: number; untrusted?: number } = {}): FrameDef {
  const nx = 8;
  const ny = 8;
  const nEps = 3;
  const rho = over.rho ?? 0;
  const untrusted = over.untrusted ?? 3;
  const gx = linspace(-1.5, 1.5, nx);
  const gy = linspace(-1.5, 1.5, ny);
  const logS = new Float64Array(nx * ny);
  const dist = new Float64Array(nx * ny);
  const trusted = new Float64Array(nx * ny).fill(1);
  for (let iy = 0; iy < ny; iy += 1) {
    for (let ix = 0; ix < nx; ix += 1) {
      const k = iy * nx + ix;
      const r = Math.hypot(gx[ix] as number, gy[iy] as number);
      logS[k] = Math.log10(Math.max(r, 1e-9));
      dist[k] = r + 0.05;
    }
  }
  // 中心附近的格点掉进舍入地板（真实帧：jordan-10 有 12/1024 个）
  const center = Math.floor(ny / 2) * nx + Math.floor(nx / 2);
  for (let i = 0; i < untrusted; i += 1) {
    const k = (center + i) % (nx * ny); // 取模：untrusted = nx*ny 时正好铺满（整帧不可信）
    trusted[k] = 0;
    logS[k] = -300;
  }
  const eps = [1e-6, 1e-4, 1e-2];
  const inside = new Float64Array(nEps * nx * ny);
  const radius = [0.06, 0.32, 0.72];
  for (let e = 0; e < nEps; e += 1) {
    for (let k = 0; k < nx * ny; k += 1) {
      const r = Math.hypot(gx[k % nx] as number, gy[Math.floor(k / nx)] as number);
      inside[e * nx * ny + k] = r <= (radius[e] as number) && trusted[k] > 0.5 ? 1 : 0;
    }
  }
  const ratio = radius.map((r) => (rho > 0 ? r / rho : 0));
  const ratioExact = eps.map((e) => (rho > 0 ? (rho + e) / rho : 0));
  const errField = new Float64Array(nx * ny).fill(1e-300);
  errField[10] = 2.2e-16;
  return build(
    {
      chapter: "1.4",
      viz: "pseudospectrum",
      arrays: [
        { name: "grid_x", data: gx, label: "Re z" },
        { name: "grid_y", data: gy, label: "Im z" },
        { name: "log10_sigma", data: logS, shape: [ny, nx], label: "log10 σ_min" },
        { name: "inside", data: inside, shape: [nEps, ny, nx], label: "伪谱掩码" },
        { name: "trusted", data: trusted, shape: [ny, nx], label: "可信掩码" },
        { name: "distance", data: dist, shape: [ny, nx], label: "到谱距离" },
        { name: "epsilon_grid", data: eps, label: "ε 序列" },
        { name: "radius", data: radius, label: "ρ_ε" },
        { name: "radius_ratio", data: ratio.map((v) => Math.max(v, 1e-300)), label: "ρ_ε/ρ(A)" },
        { name: "radius_ratio_exact", data: ratioExact.map((v) => Math.max(v, 1e-300)), label: "解析对照" },
        { name: "spectrum_x", data: [0, 0, 0, 0], label: "谱 Re" },
        { name: "spectrum_y", data: [0, 0, 0, 0], label: "谱 Im" },
        { name: "error", data: errField, label: "σ_min − dist" },
        { name: "epsilon_mark", data: [1e-4], label: "当前 ε" },
        { name: "radius_mark", data: [Math.max(ratio[1] as number, 1e-300)], label: "当前比值" },
      ],
      meta: {
        n: 4,
        family_key: rho > 0 ? 0 : 1,
        rho,
        epsilon: 1e-4,
        epsilon_index: 1,
        n_eps: nEps,
        radius_current: radius[1] as number,
        radius_ratio_current: ratio[1] as number,
        radius_ratio_min: Math.min(...ratio),
        radius_ratio_max: Math.max(...ratio),
        sigma_min: 1e-9,
        sigma_max: 2.12,
        kappa_max: 3.19e9,
        trusted_fraction: (nx * ny - untrusted) / (nx * ny),
        n_noise_points: untrusted,
        n_points: nx * ny,
        window: 1.5,
        kreiss_lower_bound: 0,
        overflow: false,
        t: 0,
      },
      strings: {
        family: "jordan",
        status: `ρ_ε(ε=1e-04) = 0.3200，ρ(A) = ${rho.toFixed(4)}，可信格点 ${(((nx * ny - untrusted) / (nx * ny)) * 100).toFixed(1)}%`,
        method: "hermitian-jacobi（σ_min 走 MᴴM 的 Hermite Jacobi；不掉进噪声地板的格点标 trusted=1）",
        data_source: "backend",
        params_json: "{}",
      },
      critical: {},
      pitfalls: PIT.ps,
    },
    over,
  );
}

/** `transient_growth`：21 个 k，ρ(A)=1，‖A^k‖ 先涨后落，`error[0]` 恰好是 0。 */
function transientFrame(over: Over & { peakAhead?: boolean } = {}): FrameDef {
  const K = 21;
  const norms = Array.from({ length: K }, (_, k) => (k === 0 ? 1 : (k + 1) ** 2));
  const pred = Array.from({ length: K }, () => 1);
  const err = norms.map((v, k) => Math.abs(v - (pred[k] as number)));
  const peak = Math.max(...norms);
  const peakIndex = over.peakAhead ? 40 : norms.indexOf(peak);
  return build(
    {
      chapter: "1.4",
      viz: "transient_growth",
      arrays: [
        { name: "step_grid", data: linspace(0, K - 1, K), label: "幂次 k" },
        { name: "norms", data: norms, label: "‖A^k‖₂" },
        { name: "spectral_prediction", data: pred, label: "ρ(A)^k" },
        { name: "growth_ratio", data: norms.map((v, k) => v / (pred[k] as number)), label: "增长倍数" },
        { name: "error", data: err, label: "误差场" },
        { name: "error_mark", data: [Math.max(err[K - 1] as number, 1e-300)], label: "当前之差" },
      ],
      meta: {
        n: 10,
        family_key: 0,
        step: K - 1,
        k_max: peakIndex === 40 ? 40 : K - 1,
        nonnormal: 1,
        spectral_radius: 1,
        peak,
        peak_index: peakIndex,
        growth_ratio: peak,
        kreiss: peak,
        asymptotic_factor: 1.31137849193891,
        norm_current: norms[K - 1] as number,
        pred_current: 1,
        ratio_current: norms[K - 1] as number,
        non_normal: false,
        peak_hit_max: peakIndex >= K - 1,
        overflow: false,
        t: K - 1,
      },
      strings: {
        family: "shear",
        status: `peak 贴着 k_max=${K - 1}：峰值可能还在后面，请把 k_max 拖大（当前 ${peak.toExponential(3)}）`,
        note: `Kreiss 常数（实测峰值）= ${peak.toExponential(4)}，出现在 k = ${peakIndex}`,
        data_source: "backend",
        params_json: "{}",
      },
      critical: {},
      pitfalls: PIT.transient,
    },
    over,
  );
}

/** `gmres_pseudospec`：8×8 网格、4 个 ε（掩码整片为空）、残差掉到 7.7e-33、谱界 3.24e4。 */
function gmresFrame(over: Over = {}): FrameDef {
  const nx = 8;
  const ny = 8;
  const nEps = 4;
  const K = 11;
  const gx = linspace(-1.5, 1.5, nx);
  const gy = linspace(-1.5, 1.5, ny);
  const logS = new Float64Array(nx * ny);
  for (let iy = 0; iy < ny; iy += 1) {
    for (let ix = 0; ix < nx; ix += 1) {
      logS[iy * nx + ix] = Math.log10(0.15 + 0.02 * ix);
    }
  }
  const eps = [1e-5, 1e-4, 1e-3, 1e-2];
  const residual = Array.from({ length: K }, (_, k) => (k === 0 ? 1 : 10 ** (-3.6 * k)));
  const bound = new Array<number>(K).fill(32379.812843062147);
  return build(
    {
      chapter: "1.4",
      viz: "gmres_pseudospec",
      arrays: [
        { name: "grid_x", data: gx, label: "Re z" },
        { name: "grid_y", data: gy, label: "Im z" },
        { name: "log10_sigma", data: logS, shape: [ny, nx], label: "log10 σ_min" },
        { name: "inside", data: new Float64Array(nEps * nx * ny), shape: [nEps, ny, nx], label: "伪谱掩码" },
        { name: "epsilon_grid", data: eps, label: "ε 序列" },
        { name: "radius", data: new Float64Array(nEps), label: "ρ_ε" },
        // 谱 Im 到 ±2.1：固定窗口 ±1.5 装不下，越界点必须报数
        { name: "spectrum_x", data: [0.45, 0.9, 1.3, 1.78, 1.1], label: "谱 Re" },
        { name: "spectrum_y", data: [-2.1, -1.0, 0.2, 2.1, 0.5], label: "谱 Im" },
        { name: "residual", data: residual, label: "相对残差" },
        { name: "residual_floor", data: new Array<number>(K).fill(1e-300), label: "地板" },
        { name: "spectral_bound", data: bound, label: "谱界" },
        { name: "error", data: [residual[K - 1] as number], label: "当前残差" },
        { name: "error_mark", data: [residual[K - 1] as number], label: "当前残差" },
      ],
      meta: {
        n: 10,
        step: K - 1,
        n_steps: K - 1,
        family_key: 0,
        rhs_key: 0,
        epsilon: 1e-3,
        epsilon_index: 2,
        converged: true,
        breakdown: true,
        plateau: false,
        krylov_dim: 10,
        residual_current: residual[K - 1] as number,
        residual_min: residual[K - 1] as number,
        residual_final: residual[K - 1] as number,
        max_drop_ratio: 3.9e-32,
        spectral_bound: 32379.812843062147,
        bound_gap: (residual[K - 1] as number) / 32379.812843062147,
        radius_current: 0,
        spectral_radius: 2.7532134313442658,
        trusted_fraction: 1,
        overflow: false,
        t: 24,
      },
      strings: {
        family: "grcar",
        rhs: "ones",
        status: "converge：10 步到 7.72e-33（单调下降，符合 GMRES 理论）",
        note: "谱半径 2.753，伪谱半径 ρ_ε = 0.000（ε=1e-03）",
        data_source: "backend",
        params_json: "{}",
      },
      critical: {},
      pitfalls: PIT.gmres,
    },
    over,
  );
}

const FRAME_FOR: Record<VizId, (over?: Over) => FrameDef> = {
  power_iteration: powerFrame,
  qr_schur: qrFrame,
  pseudospectrum: psFrame,
  transient_growth: transientFrame,
  gmres_pseudospec: gmresFrame,
};

// —— 常用断言 —— //

function sceneOf(frame: Frame, over: Partial<ModelOptions> = {}): Scene {
  const model = modelFor(frame.viz) ?? fail(`modelFor(${frame.viz}) 返回 null`);
  return model(frame, { ...OPTS, ...over });
}

/** 用**帧自己的 t** 造场景（夹具的 meta.t 与 opts.t 不一致时会多挂一条降级）。 */
function sceneAt(frame: Frame, over: Partial<ModelOptions> = {}): Scene {
  const t = typeof frame.meta["t"] === "number" ? (frame.meta["t"] as number) : 0;
  return sceneOf(frame, { t, ...over });
}

function assertValid(scene: Scene): void {
  const problems = validateScene(scene);
  assert.deepEqual(problems, [], `Scene 应合法，但有问题：${problems.join("；")}`);
}

function axisOf(scene: Scene, id: string): Axis {
  return scene.axes.find((a) => a.id === id) ?? fail(`场景里没有轴 ${id}`);
}

function layersOn(scene: Scene, axisId: string): Layer[] {
  return scene.layers.filter((l) => l.axes === axisId);
}

function heatOf(scene: Scene, axisId: string): Layer & { kind: "heat" } {
  const l = layersOn(scene, axisId).find((x) => x.kind === "heat");
  if (l === undefined || l.kind !== "heat") fail(`轴 ${axisId} 上没有热力图层`);
  return l;
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
    for (const s of layer.series ?? []) {
      out.push({ x: Array.from(s.x as ArrayLike<number>), y: Array.from(s.y as ArrayLike<number>) });
    }
  }
  return out;
}

/** 热力图层里非有限（NaN = 被掩成"无数据"）的格点数。 */
function nanCells(layer: Layer & { kind: "heat" }): number {
  let n = 0;
  for (let i = 0; i < layer.values.length; i += 1) if (!Number.isFinite(layer.values[i] as number)) n += 1;
  return n;
}

function finiteValues(a: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < a.length; i += 1) if (Number.isFinite(a[i] as number)) out.push(a[i] as number);
  return out;
}

// --------------------------------------------------------------------------- //
// 2. manifest ↔ model 表
// --------------------------------------------------------------------------- //

test("1.4 manifest 与模型表一致：5 个 viz 都有 model，未知 id 返回 null", () => {
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
// 3. 默认帧 → 合规 Scene；轴口径、requiredAux、source/status 角标
// --------------------------------------------------------------------------- //

for (const viz of VIZ_IDS) {
  test(`1.4/${viz}：默认帧 → 合规 Scene，轴线口径与 requiredAux 都对得上`, async () => {
    const spec = specOf(viz);
    const scene = sceneAt(await makeFrame(FRAME_FOR[viz]()));
    assertValid(scene);
    assert.equal(scene.chapter, "1.4");
    assert.equal(scene.viz, viz);
    assert.ok(scene.layers.length > 0, "至少要有一个可绘图层");
    assert.ok(scene.pitfalls.length >= 10, "规则 7：pitfalls 必须带进场景");
    assert.equal(scene.dataSource, "backend", "帧写的是 backend，就不该退化成 preview");
    assert.equal(scene.badges.filter((b) => b.kind === "source").length, 1, "有且只有一枚 source 角标（规则 5）");
    assert.ok(scene.badges.some((b) => b.kind === "status"), "必须有 status 角标");
    for (const ax of spec.axes ?? []) {
      assert.equal(axisOf(scene, ax.id).autoscaled, ax.autoscaled ?? false, `${viz} 的轴 ${ax.id} autoscaled 与 manifest 不一致`);
    }
    // requiredAux 是**帧里的数组名**（后端保证有），这里声明它被画在哪根轴上。
    for (const aux of spec.requiredAux ?? []) {
      const axis = AUX_AXIS[viz][aux] ?? fail(`${viz} 的 requiredAux '${aux}' 没有在测试里登记轴`);
      assert.ok(layersOn(scene, axis).length > 0, `${viz} 的 requiredAux '${aux}' 在轴 ${axis} 上一个图层都没有`);
    }
    for (const ax of scene.axes) {
      assert.ok((ax.title ?? "").length > 0 || (ax.ylabel ?? "").length > 0 || (ax.xlabel ?? "").length > 0, `轴 ${ax.id} 没有名字`);
    }
  });
}

// --------------------------------------------------------------------------- //
// 4. 退化输入：缺必需数组 / pitfalls 太短 → ModelInputError
// --------------------------------------------------------------------------- //

for (const viz of VIZ_IDS) {
  test(`1.4/${viz}：缺必需数组 '${REQUIRED_ARRAY[viz]}' → 抛 ModelInputError（不画半张图）`, async () => {
    const frame = await makeFrame(FRAME_FOR[viz]({ drop: [REQUIRED_ARRAY[viz]] }));
    assert.throws(() => sceneOf(frame), (err: unknown) => err instanceof ModelInputError);
  });

  test(`1.4/${viz}：pitfalls 太短 → 抛 ModelInputError（规则 7：model 不自行编造）`, async () => {
    const frame = await makeFrame(FRAME_FOR[viz]({ pitfalls: "太短" }));
    assert.throws(() => sceneOf(frame), (err: unknown) => err instanceof ModelInputError);
  });
}

// --------------------------------------------------------------------------- //
// 5. 布局：窄画布仍合规；同一帧两次逐字节相同；帧的 t 优先
// --------------------------------------------------------------------------- //

for (const viz of VIZ_IDS) {
  test(`1.4/${viz}：窄画布 480×600 仍合规；同一帧两次 sceneToJson 逐字节相同`, async () => {
    const frame = await makeFrame(FRAME_FOR[viz]());
    const narrow = sceneOf(frame, { width: 480, height: 600, t: 0 });
    assertValid(narrow);
    assert.ok(narrow.layers.length > 0, "窄画布上也得有图层");
    const a = sceneOf(frame);
    const b = sceneOf(frame);
    assert.equal(sceneToJson(a), sceneToJson(b), "同一帧两次构造必须完全一致（不许有随机/时间依赖）");
  });
}

test("1.4：帧 meta 里的 t 与 opts.t 不一致时以帧为准（图上写的时刻不能是请求值）", async () => {
  const frame = await makeFrame(powerFrame());
  const scene = sceneOf(frame, { t: 12.5 });
  assertValid(scene);
  assert.equal(scene.t, 24, "帧 meta.t=24，场景的 t 必须跟帧走");
});

// --------------------------------------------------------------------------- //
// 6. power_iteration：固定窗口、投影的坑、地板残差、临界角标、plateau
// --------------------------------------------------------------------------- //

test("1.4/power_iteration：主轴固定 ±1.05（不自适应）且文本点明「投影」", async () => {
  const scene = sceneAt(await makeFrame(powerFrame()));
  assertValid(scene);
  const main = axisOf(scene, "main");
  assert.deepEqual(main.xlim, [-1.05, 1.05]);
  assert.deepEqual(main.ylim, [-1.05, 1.05]);
  assert.equal(main.autoscaled, false);
  assert.match(allText(scene), /投影/, "必须写明画的是投影（n>2 时点落在圆内）");
  assert.match(allText(scene), /单位圆/, "必须说明虚线圈是单位圆");
});

test("1.4/power_iteration：残差里的地板 1e-300 不进纵向范围，但要计数", async () => {
  const scene = sceneAt(await makeFrame(powerFrame()));
  assertValid(scene);
  const aux = axisOf(scene, "aux");
  assert.ok(aux.ylim[0] > 1e-299, `地板值不该把对数纵轴拉到底（实际 ylim=${JSON.stringify(aux.ylim)}）`);
  assert.match(scene.overflow.note, /地板/, "地板样本必须写进 overflow");
  const ys = lineSeriesOf(scene, "aux").flatMap((s) => s.y);
  for (const y of ys) assert.ok(y > 1e-299, "地板值不该被当成数据画出来");
});

test("1.4/power_iteration：ratio 的临界角标（临界 1.0）必须现身", async () => {
  const scene = sceneAt(await makeFrame(powerFrame()));
  const crit = scene.badges.find((b) => b.kind === "critical") ?? fail("没有 critical 角标");
  assert.match(crit.text, /ratio/);
  assert.match(crit.text, /1\.000/, "必须写出理论临界值 1.0");
  assert.match(crit.text, /0\.600/, "必须写出当前值");
});

test("1.4/power_iteration：未收敛（plateau）时必须说出来，不许把平台画成收敛", async () => {
  const scene = sceneAt(await makeFrame(powerFrame({ converged: 0, strings: { status: "slow：400 步内未收敛（残差停在 1.4e-02）" } })));
  assertValid(scene);
  assert.match(allText(scene) + badgeText(scene), /未收敛/, "未收敛必须显式写在图上");
});

// --------------------------------------------------------------------------- //
// 7. qr_schur：热力图方向/地板夹取、Ritz 距离的地板
// --------------------------------------------------------------------------- //

test("1.4/qr_schur：热力图 nx = 迭代步数、ny = n−1（行 = 下标 i，列 = 迭代步）", async () => {
  const scene = sceneAt(await makeFrame(qrFrame()));
  assertValid(scene);
  const heat = heatOf(scene, "main");
  assert.equal(heat.nx, 12, "x 方向是迭代步（12 步）");
  assert.equal(heat.ny, 7, "y 方向是 n−1 = 7 个次对角元");
  const main = axisOf(scene, "main");
  assert.deepEqual(main.ylim, [-0.5, 6.5], "y 轴只有 n−1 行，格心落在整数上");
});

test("1.4/qr_schur：−300 地板被夹到色阶下限并计数（不许把 300 个数量级铺进色阶）", async () => {
  const scene = sceneAt(await makeFrame(qrFrame()));
  assertValid(scene);
  const heat = heatOf(scene, "main");
  const vals = finiteValues(heat.values);
  assert.ok(
    vals.every((v) => v >= -16),
    "所有格点都必须在色阶下限之上（−300 是地板，已夹到 −16）",
  );
  assert.equal(heat.vmin, -16);
  assert.equal(heat.vmax, 0);
  assert.match(scene.overflow.note, /地板/, "被夹掉的格点数必须写进 overflow");
  assert.match(allText(scene), /地板/, "文本里也要说明灰/暗格点是「已压没了」");
});

test("1.4/qr_schur：Ritz 距离里的 1e-300 换到轴底再画，且计数", async () => {
  const scene = sceneAt(await makeFrame(qrFrame()));
  assertValid(scene);
  const aux = axisOf(scene, "aux");
  const ys = lineSeriesOf(scene, "aux").flatMap((s) => s.y);
  assert.ok(ys.length > 0, "必须画出 Ritz 距离曲线");
  assert.ok(
    ys.every((y) => y >= aux.ylim[0] * (1 - 1e-9) && y > 0),
    "地板值应被换成轴底（对数轴上不能出现 0 或负值）",
  );
  assert.match(scene.overflow.note, /地板/);
  assert.match(allText(scene), /恰好 0|地板/, "要说明这些点其实是「恰好 0」");
  // 复共轭对必须点名（2×2 块不是"卡住了"）
  const complexScene = sceneAt(await makeFrame(qrFrame({ meta: { is_complex_spectrum: true }, strings: { status: "block-converge：复共轭对停在 2×2 块上" } })));
  assert.match(allText(complexScene), /复共轭|2×2/);
});

test("1.4/qr_schur：收敛与否不许靠读颜色——status/converged 必须进文本与角标", async () => {
  const scene = sceneAt(await makeFrame(qrFrame()));
  assert.match(allText(scene), /status/, "文本里必须点名 status/converged");
  assert.match(badgeText(scene), /converged|deflation/);
});

// --------------------------------------------------------------------------- //
// 8. pseudospectrum：掩码、ρ=0 的口径切换、误差场的退化
// --------------------------------------------------------------------------- //

test("1.4/pseudospectrum：trusted=0 的格点在热力图里是 NaN，且个数 = n_noise_points", async () => {
  const scene = sceneAt(await makeFrame(psFrame({ untrusted: 3 })));
  assertValid(scene);
  const heat = heatOf(scene, "main");
  assert.equal(nanCells(heat), 3, "不可信格点必须变成「无数据」（NaN），不是「σ_min 很小」");
  assert.match(scene.overflow.note, /3\/64/, "掩码个数必须进 overflow");
  assert.match(allText(scene), /灰格点|掩/, "文本里要说明灰格点是掩掉的");
  // 不可信格点另有一层标记（看得见"哪里不能读"）
  const marks = layersOn(scene, "main").filter((l) => l.kind === "scatter" && l.x.length === 3);
  assert.equal(marks.length, 1, "不可信格点应该有自己的一层标记（3 个点）");
});

test("1.4/pseudospectrum：ρ(A)=0（jordan）时改画绝对半径，三处都要说明", async () => {
  const scene = sceneAt(await makeFrame(psFrame({ rho: 0 })));
  assertValid(scene);
  const aux = axisOf(scene, "aux");
  assert.match(aux.ylabel, /ρ\(A\)=0|无定义/, "轴名要说明比值无定义");
  assert.match(allText(scene), /分母是 0|没有定义/, "文本要说明为什么换口径");
  assert.match(badgeText(scene), /绝对半径/, "角标要说明副轴换了口径");
  // 画出来的是 radius（0.06/0.32/0.72），不是全是地板的比值
  const measured = lineSeriesOf(scene, "aux").find((ser) => ser.x.length === 3 && ser.y.every((y) => y > 1e-299));
  assert.ok(measured !== undefined, "应当有一条 3 个点的实测半径曲线");
  assert.deepEqual(
    measured.y.map((v) => Number(v.toFixed(6))),
    [0.06, 0.32, 0.72],
    "画出来的必须是绝对半径 ρ_ε（不是全是地板的比值）",
  );
  assert.match(allText(scene), /斜率/, "换口径后要给出实测 log-log 斜率（这是 pitfalls 的定量证据）");
});

test("1.4/pseudospectrum：ρ(A)>0 时走 manifest 的口径，实测与解析对照两条线都在", async () => {
  const scene = sceneAt(await makeFrame(psFrame({ rho: 1.5 })));
  assertValid(scene);
  const aux = axisOf(scene, "aux");
  assert.match(aux.ylabel, /ρ_ε \/ ρ\(A\)/, "轴名必须与 manifest 一致");
  const ys = lineSeriesOf(scene, "aux").flatMap((s) => s.y).filter((y) => y > 1e-299);
  assert.ok(ys.length >= 6, "实测比值 + 解析对照至少各 3 个点");
  assert.match(allText(scene), /解析对照|正规矩阵/, "必须说明第二条线是解析对照（规则 2）");
});

test("1.4/pseudospectrum：等高线逐 ε 画，掩码整片为空的 ε 要报数（不是「伪谱为空」）", async () => {
  const scene = sceneAt(await makeFrame(psFrame()));
  assertValid(scene);
  const contours = layersOn(scene, "main").filter((l) => l.kind === "contour");
  // 8×8 网格（间距 0.43）上 ε=1e-6 的掩码整片为空：只有 2 个 ε 有边界可画
  assert.equal(contours.length, 2, "只有非空掩码才有等高线（这里 2/3 个 ε）");
  assert.match(allText(scene), /1\/3 个 ε/, "空掩码的 ε 个数必须报出来");
  const empty = sceneAt(
    await makeFrame(
      psFrame({
        arrays: { inside: new Float64Array(3 * 64) },
        shapes: { inside: [3, 8, 8] },
      }),
    ),
  );
  assertValid(empty);
  assert.equal(layersOn(empty, "main").filter((l) => l.kind === "contour").length, 0, "整片为空时没有等高线可画");
  assert.match(allText(empty), /整片为空|为空/, "空掩码必须点名（网格分辨率问题）");
});

test("1.4/pseudospectrum：误差场恒为地板时照样画，并说明它没有信息量", async () => {
  const scene = sceneAt(await makeFrame(psFrame()));
  assertValid(scene);
  const errHeat = heatOf(scene, "err");
  assert.equal(errHeat.values.length, 64, "requiredAux=error 必须真的画出来（32×32 的场）");
  assert.match(allText(scene), /恒等式|舍入尺度|地板/, "要说清这个字段为什么贴在地板上");
  assert.match(scene.overflow.note, /error 场/, "地板格点数要进 overflow");
});

test("1.4/pseudospectrum：整帧不可信（companion 那种）仍然是合规场景，且不许静默", async () => {
  const scene = sceneAt(await makeFrame(psFrame({ untrusted: 64, meta: { trusted_fraction: 0, n_noise_points: 64 } })));
  assertValid(scene);
  const heat = heatOf(scene, "main");
  assert.equal(nanCells(heat), 64, "整帧不可信 ⇒ 整块灰格");
  assert.match(scene.overflow.note, /64\/64/, "必须报出「全部不可信」");
  assert.match(badgeText(scene), /64\/64|不可信/, "角标上也要看得见");
});

// --------------------------------------------------------------------------- //
// 9. transient_growth：ρ(A)=1 却涨 9 个量级；精确 0 的丢弃
// --------------------------------------------------------------------------- //

test("1.4/transient_growth：主轴画的是增长倍数，且比值 1.0 的参考线必须在", async () => {
  const scene = sceneAt(await makeFrame(transientFrame()));
  assertValid(scene);
  const series = lineSeriesOf(scene, "main");
  const flat = series.find((s) => s.y.length >= 2 && s.y.every((y) => Math.abs(y - 1) < 1e-12));
  assert.ok(flat !== undefined, "必须有一条恒等于 1 的参考线（= 谱半径的预测）");
  assert.match(allText(scene), /谱半径/, "文本必须把「谱半径说了不算」写出来");
  assert.match(allText(scene), /ρ\(A\)=1/, "要把 ρ(A)=1 与峰值放在一起对照");
});

test("1.4/transient_growth：error 里精确的 0 在对数轴上被丢弃并计数", async () => {
  const scene = sceneAt(await makeFrame(transientFrame()));
  assertValid(scene);
  const ys = lineSeriesOf(scene, "err").flatMap((s) => s.y);
  assert.ok(ys.length > 0, "误差场必须画出来（requiredAux=error）");
  assert.ok(
    ys.every((y) => y > 0),
    "对数轴上不许出现 0（精确 0 必须显式丢弃）",
  );
  assert.match(scene.overflow.note, /恰好是 0|0（对数轴上不存在）/, "被丢掉的 0 要计数");
});

test("1.4/transient_growth：峰值落在本帧之外时不画标记，但要降级说明", async () => {
  const scene = sceneAt(await makeFrame(transientFrame({ peakAhead: true })));
  assertValid(scene);
  assert.match(badgeText(scene), /降级/, "峰值不在本帧范围内必须挂降级角标");
  assert.match(badgeText(scene) + allText(scene), /峰值/, "要说明峰值还没到");
});

test("1.4/transient_growth：meta.non_normal=False（shear，ρ=1）不许被当成结论", async () => {
  const scene = sceneAt(await makeFrame(transientFrame()));
  assertValid(scene);
  assert.match(badgeText(scene) + allText(scene), /non_normal/, "要把这个布尔量的定义说清楚，避免被当成「没有瞬态」");
});

// --------------------------------------------------------------------------- //
// 10. gmres_pseudospec：只依赖谱的界 vs 实测残差；固定窗口外的谱点
// --------------------------------------------------------------------------- //

test("1.4/gmres_pseudospec：副轴必须同时装下 1e-33 的残差与 3.2e4 的谱界", async () => {
  const scene = sceneAt(await makeFrame(gmresFrame()));
  assertValid(scene);
  const aux = axisOf(scene, "aux");
  assert.ok(aux.ylim[1] >= 3.2e4, `纵轴上限要包住谱界（实际 ${aux.ylim[1]}）`);
  assert.ok(aux.ylim[0] <= 1e-30, `纵轴下限要装得下 7.7e-33 的残差（实际 ${aux.ylim[0]}）`);
  assert.match(allText(scene), /只依赖谱/, "必须说明那条虚线是「只依赖谱」的界");
  assert.match(allText(scene), /斜率/, "要给出实测残差斜率（与 pitfalls 的 −3.5/步 对照）");
});

test("1.4/gmres_pseudospec：固定窗口外的谱点要计数进 overflow（窗口不跟着谱缩放）", async () => {
  const scene = sceneAt(await makeFrame(gmresFrame()));
  assertValid(scene);
  assert.equal(axisOf(scene, "main").autoscaled, false, "主轴是固定窗口");
  assert.match(scene.overflow.note, /2\/5|谱点/, "被裁掉的谱点必须报数");
  assert.match(allText(scene), /窗口是固定的|固定窗口/, "要说明窗口不跟着谱缩放");
});

test("1.4/gmres_pseudospec：residual 的地板贴轴底并计数；空掩码点名", async () => {
  const scene = sceneAt(await makeFrame(gmresFrame()));
  assertValid(scene);
  const aux = axisOf(scene, "aux");
  const ys = lineSeriesOf(scene, "aux").flatMap((s) => s.y);
  assert.ok(ys.every((y) => y > 0 && y >= aux.ylim[0] * (1 - 1e-9)), "地板值必须贴轴底，不许出现 0/负值");
  assert.match(allText(scene), /整片为空|为空/, "4 个 ε 的掩码全空必须点名");
  assert.match(allText(scene), /哨兵/, "要说明这一页的掩码来自 −300 哨兵而不是帧字段");
});

test("1.4/gmres_pseudospec：plateau 帧必须点名「不是收敛」", async () => {
  const scene = sceneAt(await makeFrame(gmresFrame({ meta: { plateau: true }, strings: { status: "plateau：残差连续多步几乎不降（当前 7.7e-02）" } })));
  assertValid(scene);
  assert.match(allText(scene) + badgeText(scene), /plateau/, "plateau 必须显式出现");
});

// --------------------------------------------------------------------------- //
// 11. 规则 5：预览帧要另挂 preview 角标；来源不许静默降级
// --------------------------------------------------------------------------- //

test("1.4：data_source=backend+preview 时另挂 preview 角标，source 角标仍在", async () => {
  const frame = await makeFrame(powerFrame({ strings: { data_source: "backend+preview" } }));
  const scene = sceneAt(frame);
  assertValid(scene);
  assert.equal(scene.dataSource, "backend+preview");
  assert.equal(scene.badges.filter((b) => b.kind === "source").length, 1);
  assert.equal(scene.badges.filter((b) => b.kind === "preview").length, 1, "预览必须另挂一枚 preview 角标");
  assert.match(badgeText(scene), /预览/);
});

test("1.4：缺 meta 的关键字段时降级角标要点名（不是静默用请求值）", async () => {
  const frame = await makeFrame(powerFrame({ dropMeta: ["ratio", "step"] }));
  const scene = sceneOf(frame, { t: 0 });
  assertValid(scene);
  const degraded = scene.badges.find((b) => b.text.includes("降级")) ?? fail("缺 meta 时必须挂降级角标");
  assert.match(degraded.text, /ratio|step/, "降级文案要点名缺了什么");
});
