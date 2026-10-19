/** 5.3 章（伴随方法与灵敏度）四个 model 共用的显示工具。 */

export const CHAPTER = "5.3";

/**
 * 光栅化器会为每个 heat 轴自动画色标（轴右缘 +46px 起、宽 14px，再往右是刻度）。
 * 热图轴必须从右列文字里让出这段宽度，否则色标与右栏说明叠字（5.3 第一版真踩过）。
 */
export const HEAT_RIGHT_GUTTER = 110;

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

/** 取 `[start, end]` 闭区间（用于反向伴随轨迹：从尾部倒着取）。 */
export function sliceRange(a: ArrayLike<number>, start: number, end: number): Float64Array {
  const lo = Math.max(0, Math.min(a.length, Math.floor(start)));
  const hi = Math.max(lo, Math.min(a.length, Math.floor(end) + 1));
  const out = new Float64Array(hi - lo);
  for (let i = lo; i < hi; i += 1) out[i - lo] = a[i] as number;
  return out;
}

export function finiteExtent(values: readonly ArrayLike<number>[], pad = 0.05): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const arr of values) {
    for (let i = 0; i < arr.length; i += 1) {
      const v = arr[i] as number;
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-1, 1];
  if (hi === lo) {
    const d = Math.max(1e-9, Math.abs(hi) * 0.1);
    return [lo - d, hi + d];
  }
  const d = (hi - lo) * pad;
  return [lo - d, hi + d];
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

/** 用对数压缩把跨量级的箭头长度压到 0.25–1（帧里已给 `arrow_length`，这里只作缺省）。 */
export function logArrowLength(norm: number, lo: number, hi: number): number {
  if (!(norm > 0) || !(hi > lo)) return 0.4;
  const t = (Math.log(norm) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
  return 0.25 + 0.75 * Math.max(0, Math.min(1, t));
}
