/**
 * 2.2 章四个 model 共用的**显示派生量**：把帧里的数组读成画图要用的形状。
 *
 * 边界（与 `viz/shared/src/model.ts` 的文件头同一条纪律）：
 *  * 这里**不产生任何物理量**——跳变量级、曲率、能量、误差全部来自帧；
 *  * 这里只做"显示几何"与"读帧"：取最大有限值、按地板抬升、把越界点裁到画面内
 *    并**数清楚裁掉了几个**（规则 1：裁了就必须报出来）、以及格式化。
 */

import { niceLimits } from "../../shared/src/axes.js";

/** 半单位舍入 `u/2 = 2^-53`：对数轴上"0 画不出来"的统一地板（与后端 `UNIT_ROUNDOFF` 同口径）。 */
export const FLOOR = Math.pow(2, -53);

/** 一段数值里的最大有限值；一个有限值都没有时返回 `undefined`（不拿 0 顶替）。 */
export function maxFinite(values: ArrayLike<number>): number | undefined {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] as number;
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : undefined;
}

/** 是否全部为有限值（长度 0 视为"全有限"）。 */
export function allFinite(values: ArrayLike<number>): boolean {
  for (let i = 0; i < values.length; i += 1) {
    if (!Number.isFinite(values[i] as number)) return false;
  }
  return true;
}

/** 数值格式化：整数按整数显示，其余按有效位数显示（与 1.2 章同一口径）。 */
export function fmt(value: number | undefined, digits = 4): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  if (Number.isInteger(value)) return String(value);
  const a = Math.abs(value);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) {
    const [m, e] = value.toExponential(digits - 1).split("e");
    return `${m}e${e}`;
  }
  return value.toPrecision(digits);
}

/** 布尔/枚举键的中文表（**只**用于把帧里的 key 翻成 UI 文案，不参与任何判断之外的计算）。 */
export const BC_NAME: Record<string, string> = {
  natural: "自然（M0=Mn=0）",
  clamped: "夹持（给定端点斜率）",
  not_a_knot: "not-a-knot",
};

export const BC_KEY: readonly string[] = ["natural", "clamped", "not_a_knot"];
export const NODES_KEY: readonly string[] = ["uniform", "chebyshev"];

/**
 * 枚举参数的显示名：**帧 `strings` 优先**（后端已经把中文名写好了），
 * 缺了才按 `meta` 里的数值编号查本地表，再缺就如实说"未知"。
 *
 * 为什么不直接用 `opts.params` 里的字符串：帧与请求可能不同步（时间轴/降级路径），
 * 图上的标注必须描述**这一帧**，不是描述请求。
 */
export function enumName(
  fromFrame: string | undefined,
  metaIndex: number | undefined,
  keys: readonly string[],
  names: Record<string, string>,
): { text: string; known: boolean } {
  if (fromFrame !== undefined && fromFrame.trim() !== "") return { text: fromFrame, known: true };
  if (metaIndex !== undefined && Number.isInteger(metaIndex) && metaIndex >= 0 && metaIndex < keys.length) {
    const key = keys[metaIndex] as string;
    return { text: names[key] ?? key, known: true };
  }
  return { text: "", known: false };
}

/** 一串非负量在 log 轴上的取整范围；全为非有限/非正值时给"只画得下地板"的退化范围。 */
export function logRange(
  values: readonly number[],
  fallback: [number, number] = [FLOOR, 1e-2],
  ticks = 7,
): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (!Number.isFinite(v) || v <= 0) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return fallback;
  const lim = niceLimits(lo, hi, ticks, { log: true });
  return lim[0] > 0 && lim[1] > lim[0] ? lim : fallback;
}

/** 线性范围（含 0 打底的情形）；全为非有限值时返回回落范围。 */
export function linRange(values: readonly number[], fallback: [number, number], ticks = 6): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return fallback;
  const lim = niceLimits(lo, hi, ticks);
  return lim[1] > lim[0] ? lim : fallback;
}

/** 固定（非自适应）主轴的范围：**同样**走 1-2-5 取整，只是不随数据变。 */
export function fixedRange(values: readonly number[], pad: number, fallback: [number, number]): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return fallback;
  const lim = niceLimits(lo - pad, hi + pad, 6);
  return lim[1] > lim[0] ? lim : fallback;
}

/** 把一条序列裁进 `[lo, hi]`，并返回**被裁掉了几个点**（规则 1：裁了多少要能报出来）。 */
export function clipSeries(
  y: ArrayLike<number>,
  lo: number,
  hi: number,
): { values: Float64Array; clipped: number; min: number; max: number } {
  const out = new Float64Array(y.length);
  let clipped = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < y.length; i += 1) {
    const v = y[i] as number;
    if (!Number.isFinite(v)) {
      out[i] = NaN;
      continue;
    }
    if (v < min) min = v;
    if (v > max) max = v;
    if (v > hi) {
      out[i] = hi;
      clipped += 1;
    } else if (v < lo) {
      out[i] = lo;
      clipped += 1;
    } else {
      out[i] = v;
    }
  }
  return { values: out, clipped, min, max };
}

/** 一列 `(x, y)` 里 y 超出 `[lo, hi]` 的点数（不修改数据）。 */
export function countOutside(y: ArrayLike<number>, lo: number, hi: number): number {
  let n = 0;
  for (let i = 0; i < y.length; i += 1) {
    const v = y[i] as number;
    if (Number.isFinite(v) && (v < lo || v > hi)) n += 1;
  }
  return n;
}

/** 非有限值个数（与 `geometry.ts::countNonFinite` 同义，这里避免多一次 import）。 */
export function countNonFinite(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if (!Number.isFinite(a[i] as number)) n += 1;
  return n;
}

/** 把多个数组的非有限值个数加起来（`applyOverflow` 的第二个参数）。 */
export function totalNonFinite(arrays: readonly ArrayLike<number>[]): number {
  let n = 0;
  for (const a of arrays) n += countNonFinite(a);
  return n;
}
