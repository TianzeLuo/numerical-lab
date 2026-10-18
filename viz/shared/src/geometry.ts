/**
 * 几何与数据场工具（DOM 无关，可在 Node 里直接测）。
 *
 * 这里放"画之前要算的东西"：数据↔像素映射、网格场→RGBA、等值线的 marching squares、
 * 以及**溢出检测**——`checkOverflow` 是防误导规则 1 的自动化：
 * 谁被裁掉了、裁掉多少，都必须能被算出来并写进 `Scene.overflow`，而不是靠人眼发现。
 */

import { applyCmap } from "./colorscale.js";
import type { CmapName } from "./colorscale.js";
import { mapToPixel, mapToData } from "./axes.js";
import type { Axis, Scene } from "./scene.js";

// --------------------------------------------------------------------------- //
// 网格与坐标
// --------------------------------------------------------------------------- //
/** 第 `i` 个**节点**的坐标：`nx` 个节点均匀铺满 `[lo, hi]`（含两端）。 */
export function gridCoord(i: number, n: number, lo: number, hi: number): number {
  if (n <= 1) return lo;
  return lo + ((hi - lo) * i) / (n - 1);
}

/** 数据坐标 → 画布像素（取轴矩形内的位置；y 轴向下）。 */
export function dataToPixel(axis: Axis, x: number, y: number): [number, number] {
  const [rx, ry, rw, rh] = axis.rect;
  return [
    mapToPixel(x, axis.xlim[0], axis.xlim[1], rx, rx + rw, axis.xscale),
    mapToPixel(y, axis.ylim[0], axis.ylim[1], ry + rh, ry, axis.yscale),
  ];
}

/** 画布像素 → 数据坐标（交互用：鼠标位置 → 参数值）。 */
export function pixelToData(axis: Axis, px: number, py: number): [number, number] {
  const [rx, ry, rw, rh] = axis.rect;
  return [
    mapToData(px, axis.xlim[0], axis.xlim[1], rx, rx + rw, axis.xscale),
    mapToData(py, axis.ylim[0], axis.ylim[1], ry + rh, ry, axis.yscale),
  ];
}

/** 统计非有限值个数（用于提示"这张图里有没画出来的格子"）。 */
export function countNonFinite(values: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (!Number.isFinite(values[i] as number)) n += 1;
  }
  return n;
}

// --------------------------------------------------------------------------- //
// 热力图
// --------------------------------------------------------------------------- //
/**
 * 网格场 → RGBA 字节（行主序，长度 `4·nx·ny`）。
 *
 * **为什么把"无数据"涂成灰色而不是透明**：白底上透明格子看起来和"零值"一样，
 * 而零值在多数色标里是有颜色的。把 NaN 画成中性灰，是让"这里没有数据"和
 * "这里的值是 0"在视觉上不再混淆——这是本项目最在意的一类细节。
 */
export function buildHeatRgba(
  values: ArrayLike<number>,
  nx: number,
  ny: number,
  cmap: CmapName,
  vmin: number,
  vmax: number,
  opts: { logScale?: boolean; clipBelow?: number; nanColor?: [number, number, number, number] } = {},
): Uint8ClampedArray {
  if (values.length !== nx * ny) {
    throw new RangeError(`buildHeatRgba：values 长度 ${values.length} ≠ nx*ny = ${nx * ny}`);
  }
  const rgba = applyCmap(cmap, values, vmin, vmax, { logScale: opts.logScale, clipBelow: opts.clipBelow });
  const nanColor = opts.nanColor ?? [130, 130, 130, 255];
  for (let p = 0; p < nx * ny; p += 1) {
    if (rgba[4 * p + 3] === 0) {
      rgba[4 * p] = nanColor[0];
      rgba[4 * p + 1] = nanColor[1];
      rgba[4 * p + 2] = nanColor[2];
      rgba[4 * p + 3] = nanColor[3];
    }
  }
  return rgba;
}

/** 网格场里某个值的 RGBA 像素索引（行主序）。 */
export function pixelIndex(i: number, j: number, nx: number): number {
  return j * nx + i;
}

// --------------------------------------------------------------------------- //
// 等值线（marching squares）
// --------------------------------------------------------------------------- //
/**
 * Marching squares：把标量场在给定层级上切成线段。
 *
 * 返回**扁平线段数组** `[x0, y0, x1, y1, …]`，坐标是**网格单位**（列号 i、行号 j，
 * 可以是小数），调用方再用 {@link gridCoord} + {@link dataToPixel} 映射到画布。
 * 用网格单位而不是数据单位，是为了让"重采样分辨率"与"物理坐标"解耦。
 *
 * 鞍点（5/10 两种歧义构型）按**中心值**决定连接方式，避免出现自交的等值线。
 */
export function marchingSquares(values: ArrayLike<number>, nx: number, ny: number, level: number): number[] {
  if (nx < 2 || ny < 2) return [];
  const out: number[] = [];
  const at = (i: number, j: number): number => values[j * nx + i] as number;
  const lerp = (a: number, b: number): number => {
    const d = b - a;
    if (d === 0 || !Number.isFinite(d)) return 0.5;
    const t = (level - a) / d;
    return t < 0 ? 0 : t > 1 ? 1 : t;
  };
  for (let j = 0; j < ny - 1; j += 1) {
    for (let i = 0; i < nx - 1; i += 1) {
      const v00 = at(i, j);
      const v10 = at(i + 1, j);
      const v11 = at(i + 1, j + 1);
      const v01 = at(i, j + 1);
      if (!Number.isFinite(v00) || !Number.isFinite(v10) || !Number.isFinite(v11) || !Number.isFinite(v01)) continue;
      let code = 0;
      if (v00 > level) code |= 1;
      if (v10 > level) code |= 2;
      if (v11 > level) code |= 4;
      if (v01 > level) code |= 8;
      if (code === 0 || code === 15) continue;
      // 四条边上的交点（用网格坐标表示）
      const bottom: [number, number] = [i + lerp(v00, v10), j];
      const right: [number, number] = [i + 1, j + lerp(v10, v11)];
      const top: [number, number] = [i + lerp(v01, v11), j + 1];
      const left: [number, number] = [i, j + lerp(v00, v01)];
      const push = (a: [number, number], b: [number, number]): void => {
        out.push(a[0], a[1], b[0], b[1]);
      };
      switch (code) {
        case 1:
        case 14:
          push(left, bottom);
          break;
        case 2:
        case 13:
          push(bottom, right);
          break;
        case 3:
        case 12:
          push(left, right);
          break;
        case 4:
        case 11:
          push(right, top);
          break;
        case 6:
        case 9:
          push(bottom, top);
          break;
        case 7:
        case 8:
          push(left, top);
          break;
        case 5:
        case 10: {
          // 鞍点：看中心平均值的符号决定拓扑
          const center = (v00 + v10 + v11 + v01) / 4;
          const centerAbove = center > level;
          if ((code === 5) === centerAbove) {
            push(left, top);
            push(bottom, right);
          } else {
            push(left, bottom);
            push(right, top);
          }
          break;
        }
        default:
          break;
      }
    }
  }
  return out;
}

/** 多层等值线：返回每层的层级与线段。 */
export function contourSegments(
  values: ArrayLike<number>,
  nx: number,
  ny: number,
  levels: number[],
): { level: number; segments: number[] }[] {
  return levels.map((level) => ({ level, segments: marchingSquares(values, nx, ny, level) }));
}

// --------------------------------------------------------------------------- //
// 溢出检测（防误导规则 1 的自动化）
// --------------------------------------------------------------------------- //
export interface OverflowReport {
  flag: boolean;
  note: string;
  /** 每个越界图层：图层序号、类型、轴 id、越界方向与相对幅度。 */
  offenders: { index: number; kind: string; axes: string; axis: "x" | "y"; over: number; limit: number }[];
  /** 有多少个非有限值没有被画出来。 */
  nonFinite: number;
}

function axisOf(scene: Scene, id: string): Axis | undefined {
  return scene.axes.find((a) => a.id === id);
}

function noteFor(report: Omit<OverflowReport, "flag" | "note">): string {
  const parts: string[] = [];
  if (report.offenders.length > 0) {
    const first = report.offenders[0]!;
    const extra = report.offenders.length > 1 ? `（共 ${report.offenders.length} 处）` : "";
    parts.push(
      `图层${first.index}(${first.kind}) 的 ${first.axis} 超出轴 ${first.axes} 范围：` +
        `${first.over.toPrecision(3)} 倍于其 ${first.limit.toPrecision(3)} 的跨度${extra}`,
    );
  }
  if (report.nonFinite > 0) parts.push(`${report.nonFinite} 个非有限值未绘制（以灰色表示"无数据"）`);
  return parts.join("；");
}

/**
 * 扫描整个 Scene，找出"被坐标轴裁掉"和"根本没画出来"的数据。
 *
 * 调用方（章节 model）应当把结果写进 `scene.overflow`：这是规则 1 要求的
 * "溢出必须显式标记"。只统计**数据单位**的越界，不做任何插值猜测。
 */
export function checkOverflow(scene: Scene, opts: { tolerance?: number } = {}): OverflowReport {
  const tol = opts.tolerance ?? 1e-12;
  const offenders: OverflowReport["offenders"] = [];
  let nonFinite = 0;
  const consider = (index: number, kind: string, axisId: string, values: { x?: ArrayLike<number>; y?: ArrayLike<number> }): void => {
    const ax = axisOf(scene, axisId);
    if (!ax) return;
    for (const which of ["x", "y"] as const) {
      const data = values[which];
      if (!data) continue;
      const lim = which === "x" ? ax.xlim : ax.ylim;
      const span = Math.abs(lim[1] - lim[0]);
      let worst = 0;
      for (let i = 0; i < data.length; i += 1) {
        const v = data[i] as number;
        if (!Number.isFinite(v)) {
          nonFinite += 1;
          continue;
        }
        const overLo = lim[0] - v;
        const overHi = v - lim[1];
        const over = Math.max(overLo, overHi, 0);
        const rel = span > 0 ? over / span : over > 0 ? Number.POSITIVE_INFINITY : 0;
        if (rel > worst) worst = rel;
      }
      if (worst > tol) offenders.push({ index, kind, axes: axisId, axis: which, over: worst, limit: span });
    }
  };
  scene.layers.forEach((layer, index) => {
    switch (layer.kind) {
      case "line":
        for (const s of layer.series) consider(index, layer.kind, layer.axes, { x: s.x, y: s.y });
        break;
      case "scatter":
        consider(index, layer.kind, layer.axes, { x: layer.x, y: layer.y });
        break;
      case "band":
        consider(index, layer.kind, layer.axes, { x: layer.band.x, y: layer.band.yhi });
        consider(index, layer.kind, layer.axes, { y: layer.band.ylo });
        break;
      case "bars": {
        consider(index, layer.kind, layer.axes, { x: layer.x, y: layer.values });
        if (layer.errors) {
          // 误差棒画出去也算溢出：值 ± 误差都必须落在轴内
          const hi = new Float64Array(layer.values.length);
          const lo = new Float64Array(layer.values.length);
          for (let i = 0; i < layer.values.length; i += 1) {
            const v = layer.values[i] as number;
            const e = Number.isFinite(layer.errors[i] as number) ? (layer.errors[i] as number) : 0;
            hi[i] = v + e;
            lo[i] = v - e;
          }
          consider(index, layer.kind, layer.axes, { y: hi });
          consider(index, layer.kind, layer.axes, { y: lo });
        }
        break;
      }
      case "loglogfit":
        consider(index, layer.kind, layer.axes, { x: layer.h, y: layer.err });
        break;
      case "heat":
      case "contour": {
        const [x0, x1, y0, y1] = layer.extent;
        consider(index, layer.kind, layer.axes, { x: [x0, x1], y: [y0, y1] });
        break;
      }
      case "vectorfield": {
        const [x0, x1, y0, y1] = layer.extent;
        consider(index, layer.kind, layer.axes, { x: [x0, x1], y: [y0, y1] });
        break;
      }
      default:
        break;
    }
  });
  const partial = { offenders, nonFinite };
  return { flag: offenders.length > 0 || nonFinite > 0, note: noteFor(partial), ...partial };
}

/** 把图层数据映射到像素折线（渲染层与光栅器共用；纯函数）。 */
export function layerToPixelPath(axis: Axis, x: ArrayLike<number>, y: ArrayLike<number>): (number | null)[] {
  const out: (number | null)[] = [];
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i += 1) {
    const px = dataToPixel(axis, x[i] as number, y[i] as number);
    if (!Number.isFinite(px[0]) || !Number.isFinite(px[1])) {
      out.push(null, null); // 断开（NaN 处不允许用直线连过去）
      continue;
    }
    out.push(px[0], px[1]);
  }
  return out;
}
