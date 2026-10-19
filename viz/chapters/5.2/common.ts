/**
 * 5.2 章（反问题与正则化）四个 model 共用的显示几何与判据配色。
 *
 * 判据顺序在四个 viz 里保持一致：oracle（知道真解，只作基准）→ L 曲线 →
 * GCV → 偏差原则。颜色固定，读图不用重新认。
 */

export const CHAPTER = "5.2";

export const SLOT = {
  source: 0,
  axisMode: 1,
  critical: 2,
  status: 3,
  overflow: 4,
  degraded: 5,
} as const;

export const CRITERIA = ["oracle", "lcurve", "gcv", "discrepancy"] as const;
export type Criterion = (typeof CRITERIA)[number];

export const CRITERION_LABEL: Record<Criterion, string> = {
  oracle: "oracle（用 x*，基准）",
  lcurve: "L 曲线拐点",
  gcv: "GCV 最小",
  discrepancy: "偏差原则",
};

export function criterionColors(pal: {
  exact: string;
  curves: readonly string[];
  residual: string;
}): Record<Criterion, string> {
  return {
    oracle: pal.exact,
    lcurve: pal.curves[2] as string,
    gcv: pal.curves[1] as string,
    discrepancy: pal.residual,
  };
}

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
    if (!Number.isFinite(v) || !Number.isFinite(target)) continue;
    const d = Math.abs(v - target);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/** 把数值指数（如 -3.13）写成 λ=10^-3.13 的短读法。 */
export function lambdaAt(log10Lam: number | undefined): string {
  if (log10Lam === undefined || !Number.isFinite(log10Lam)) return "λ=?";
  return `λ=10^${log10Lam.toFixed(2)}`;
}
