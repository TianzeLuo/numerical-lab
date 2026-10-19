/** 5.4 章（UQ 与全局敏感性）四个 model 共用的显示工具。注意：本章误差数组存的是 log10 值。 */

export const CHAPTER = "5.4";

export const SLOT = {
  source: 0,
  axisMode: 1,
  critical: 2,
  status: 3,
  overflow: 4,
  degraded: 5,
} as const;

export function prefix(a: ArrayLike<number>, n: number): Float64Array {
  const m = Math.max(0, Math.min(a.length, Math.floor(n)));
  const out = new Float64Array(m);
  for (let i = 0; i < m; i += 1) out[i] = a[i] as number;
  return out;
}

export function nearestIndex(xs: ArrayLike<number>, target: number): number {
  let best = -1;
  let bestD = Number.POSITIVE_INFINITY;
  for (let i = 0; i < xs.length; i += 1) {
    const v = xs[i] as number;
    if (!Number.isFinite(v)) continue;
    const d = Math.abs(v - target);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/** 把 log10 值换算成“×10^k”的短读法。 */
export function pow10Label(v: number | undefined, digits = 2): string {
  if (v === undefined || !Number.isFinite(v)) return "?";
  return `10^${v.toFixed(digits)}`;
}

/** 在 log10 误差数组里取最小值下标（-Infinity/NaN 跳过）。 */
export function argMinLog(a: ArrayLike<number>): number {
  let best = -1;
  let bestV = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (!Number.isFinite(v)) continue;
    if (v < bestV) {
      bestV = v;
      best = i;
    }
  }
  return best;
}
