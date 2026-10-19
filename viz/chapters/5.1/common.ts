/**
 * 5.1 章（无约束优化与拟牛顿）四个 model 共用的**显示几何**。
 *
 * 这里只放不改变物理口径的纯函数：
 * - 轨迹切片（帧里的数组是完整历史，scrub 到第 k 步时只画前 k+1 点）；
 * - 等高线级别（对数间隔，避免二次型/鞍点的等高线全挤在一侧）；
 * - 方法配色与标签（四张图统一，读图时不用重新认颜色）。
 */

import type { LineSeries } from "../../shared/src/scene.js";
import { series } from "../../shared/src/scene.js";

export const CHAPTER = "5.1";

/** 固定角标槽位：缺一枚时其余不挪位。 */
export const SLOT = {
  source: 0,
  axisMode: 1,
  critical: 2,
  aux: 3,
  status: 4,
  overflow: 5,
  degraded: 6,
} as const;

export const METHOD_ORDER = ["gd", "momentum", "bfgs", "newton"] as const;
export type Method = (typeof METHOD_ORDER)[number];

export const METHOD_LABEL: Record<Method, string> = {
  gd: "梯度下降",
  momentum: "重球动量",
  bfgs: "BFGS",
  newton: "牛顿",
};

/** 四种方法在四张图里的颜色一致：蓝=GD、绿=动量、黄=BFGS、紫=牛顿。 */
export function methodColors(pal: {
  curves: readonly string[];
  residual: string;
}): Record<Method, string> {
  return {
    gd: pal.curves[0] as string,
    momentum: pal.curves[1] as string,
    bfgs: pal.curves[2] as string,
    newton: pal.residual,
  };
}

/** 取数组前 `n` 个元素；`n` 夹到 `[0, length]`，不补位也不外推。 */
export function prefix(a: ArrayLike<number>, n: number): Float64Array {
  const m = Math.max(0, Math.min(a.length, Math.floor(n)));
  const out = new Float64Array(m);
  for (let i = 0; i < m; i += 1) out[i] = a[i] as number;
  return out;
}

/** 0,1,…,n-1（用于没有显式横轴的逐点历史）。 */
export function indexAxis(n: number): Float64Array {
  const out = new Float64Array(Math.max(0, n));
  for (let i = 0; i < out.length; i += 1) out[i] = i;
  return out;
}

/** 最近邻下标（`xs` 不要求单调）；空数组返回 -1。 */
export function nearestIndex(xs: ArrayLike<number>, target: number): number {
  let best = -1;
  let bestD = Number.POSITIVE_INFINITY;
  for (let i = 0; i < xs.length; i += 1) {
    const v = xs[i] as number;
    if (!Number.isFinite(v) || !Number.isFinite(target)) continue;
    const d = Math.abs(v - target);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/** 把 `k` 夹进 `[0, kMax]`；非有限值不当成 0。 */
export function stepIndex(t: number, kMax: number): number {
  if (!Number.isFinite(t)) return 0;
  return Math.max(0, Math.min(kMax, Math.round(t)));
}

/**
 * 正值的对数间隔级别。若最小值非正（例如 f 在原点为 0），从 `positiveFloor` 起。
 *
 * `contourLayer` 的默认级别是线性等距的；二次型的 f 从 0 涨到几百，
 * 线性级别会把原点附近全糊成一条。这里给的是**显示级别**，不参与任何误差计算。
 */
export function logLevels(values: ArrayLike<number>, count = 12, positiveFloor = 1e-4): number[] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] as number;
    if (!Number.isFinite(v)) continue;
    if (v > 0 && v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(hi) || hi <= 0) return [0];
  const start = Math.max(Math.min(Number.isFinite(lo) ? lo : positiveFloor, hi / 100), positiveFloor);
  const levels: number[] = [];
  const top = Math.log10(hi);
  const bottom = Math.log10(Math.max(start, hi * 1e-12));
  const n = Math.max(2, count);
  for (let i = 0; i < n; i += 1) {
    const e = bottom + ((top - bottom) * i) / (n - 1);
    levels.push(10 ** e);
  }
  return levels;
}

/**
 * 等高线级别：值跨零时用对称的线性级别（鞍点的双曲线族不能只画正的一侧），
 * 全正时用对数级别（二次型/Rosenbrock 的量级跨好几阶）。
 */
export function signedLevels(values: ArrayLike<number>, count = 13): number[] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  let hasNeg = false;
  let hasPos = false;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] as number;
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    if (v < 0) hasNeg = true;
    if (v > 0) hasPos = true;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0];
  if (hasNeg && hasPos) {
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    const n = Math.max(3, count);
    const out: number[] = [];
    for (let i = 0; i < n; i += 1) out.push(-m + (2 * m * i) / (n - 1));
    // 去掉恰好 0：退化等高线 f=0 是两条直线，画出来会把鞍点的“十字”误当数据。
    return out.filter((v) => Math.abs(v) > m * 1e-9);
  }
  return logLevels(values, count, Math.max(Math.abs(lo), 1e-4) * 1e-3);
}

/** 由 (x,y) 生成闭合圆的多段线；`n` 固定，纯显示。 */
export function circleSeries(
  cx: number,
  cy: number,
  r: number,
  color: string,
  n = 73,
): LineSeries {
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const rr = Math.max(0, Number.isFinite(r) ? r : 0);
  for (let i = 0; i < n; i += 1) {
    const a = (2 * Math.PI * i) / (n - 1);
    xs[i] = cx + rr * Math.cos(a);
    ys[i] = cy + rr * Math.sin(a);
  }
  return series(xs, ys, color, { width: 1.4 });
}

/** 一段 `[lo, hi]` 的水平参考线（用于理论比值/地板）。 */
export function horizontal(xlim: readonly [number, number], y: number, color: string, width = 1, dash?: number[]): LineSeries {
  return series([xlim[0], xlim[1]], [y, y], color, { width, dash, label: undefined });
}

/**
 * 把规则网格 `(xs × ys, values)` 裁到显示窗 `[xlim]×[ylim]`。
 *
 * 背景等高线常常覆盖比视窗更大的区域；把整张网格交给 `contourLayer` 会让
 * `applyOverflow` 把“底图超出视窗”误报成数据溢出（5.1 第一版基线上真的出现过）。
 * 这里只做显示裁剪：输出网格的边界夹到视窗与数据范围的交集，节点值双线性插值。
 * 不插值而直接取相邻节点会让 extent 比视窗多出半格，规则 1 仍会报假溢出。
 */
export function cropGrid(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  values: ArrayLike<number>,
  xlim: readonly [number, number],
  ylim: readonly [number, number],
): { nx: number; ny: number; x0: number; x1: number; y0: number; y1: number; values: Float64Array } | undefined {
  const nx = xs.length;
  const ny = ys.length;
  if (!(nx > 1) || !(ny > 1) || values.length !== nx * ny) return undefined;
  if ((xs[nx - 1] as number) < xlim[0] || (xs[0] as number) > xlim[1]) return undefined;
  if ((ys[ny - 1] as number) < ylim[0] || (ys[0] as number) > ylim[1]) return undefined;
  const x0 = Math.max(xlim[0], xs[0] as number);
  const x1 = Math.min(xlim[1], xs[nx - 1] as number);
  const y0 = Math.max(ylim[0], ys[0] as number);
  const y1 = Math.min(ylim[1], ys[ny - 1] as number);
  if (!(x1 > x0) || !(y1 > y0)) return undefined;

  const locate = (coords: ArrayLike<number>, v: number): readonly [number, number, number] => {
    const n = coords.length;
    if (v <= (coords[0] as number)) return [0, 0, 0];
    if (v >= (coords[n - 1] as number)) return [n - 1, n - 1, 0];
    let hi = 1;
    while (hi < n - 1 && (coords[hi] as number) < v) hi += 1;
    const lo = hi - 1;
    const span = (coords[hi] as number) - (coords[lo] as number);
    return [lo, hi, span > 0 ? (v - (coords[lo] as number)) / span : 0];
  };
  const sample = (xv: number, yv: number): number => {
    const [i0, i1, tx] = locate(xs, xv);
    const [j0, j1, ty] = locate(ys, yv);
    const v00 = values[j0 * nx + i0] as number;
    const v10 = values[j0 * nx + i1] as number;
    const v01 = values[j1 * nx + i0] as number;
    const v11 = values[j1 * nx + i1] as number;
    const a = v00 + (v10 - v00) * tx;
    const b = v01 + (v11 - v01) * tx;
    return a + (b - a) * ty;
  };

  const dx = ((xs[nx - 1] as number) - (xs[0] as number)) / (nx - 1);
  const dy = ((ys[ny - 1] as number) - (ys[0] as number)) / (ny - 1);
  const w = Math.max(2, Math.min(257, Math.round((x1 - x0) / dx) + 1));
  const h = Math.max(2, Math.min(257, Math.round((y1 - y0) / dy) + 1));
  const out = new Float64Array(w * h);
  for (let j = 0; j < h; j += 1) {
    const yv = y0 + ((y1 - y0) * j) / (h - 1);
    for (let i = 0; i < w; i += 1) {
      const xv = x0 + ((x1 - x0) * i) / (w - 1);
      out[j * w + i] = sample(xv, yv);
    }
  }
  return { nx: w, ny: h, x0, x1, y0, y1, values: out };
}
