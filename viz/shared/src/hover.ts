/**
 * 图形命中测试：把鼠标/手指的位置换算回数据坐标，找最近的线、点、柱、网格或箭头。
 *
 * 这一层不碰 DOM，方便在 Node 里逐类测试；view 只负责把返回的数值排版成提示框，
 * 并在透明覆盖层上画一个定位圈。命中阈值按**画布逻辑像素**计（不是设备像素）。
 */

import { formatNumber } from "./axes.js";
import { dataToPixel, gridCoord, pixelToData } from "./geometry.js";
import type { Axis, Layer, Scene } from "./scene.js";

export type HoverShape = "point" | "bar" | "cell" | "vector" | "fit";

export interface HoverHit {
  layerIndex: number;
  kind: Layer["kind"];
  axisId: string;
  label: string;
  color: string;
  /** 数据坐标。 */
  x: number;
  y: number;
  /** 画布逻辑像素（用于覆盖层定位）。 */
  px: number;
  py: number;
  xText: string;
  yText: string;
  /** 额外一行说明（热图数值、数值误差、向量模长等）。 */
  detail?: string;
  shape: HoverShape;
  dash: boolean;
  axisRect: [number, number, number, number];
}

export interface HoverOptions {
  /** 命中半径（逻辑像素）。 */
  radius?: number;
  /** 每个轴在 HTML 面板说明里的完整横/纵轴名（画布上可能因空间不足而省略）。 */
  labels?: Readonly<Record<string, { x?: string; y?: string }>>;
  /** 图层没有自己的 label 时，用图例里同色条目的文字兜底（键为小写颜色）。 */
  colorLabels?: Readonly<Record<string, string>>;
}

function finitePair(x: unknown, y: unknown): boolean {
  return typeof x === "number" && typeof y === "number" && Number.isFinite(x) && Number.isFinite(y);
}

function insideAxis(axis: Axis, px: number, py: number, pad = 0): boolean {
  const [rx, ry, rw, rh] = axis.rect;
  return px >= rx - pad && px <= rx + rw + pad && py >= ry - pad && py <= ry + rh + pad;
}

function segDistance(
  px: number,
  py: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): { dist: number; qx: number; qy: number } {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len2 = dx * dx + dy * dy;
  let t = len2 > 0 ? ((px - x1) * dx + (py - y1) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  const qx = x1 + t * dx;
  const qy = y1 + t * dy;
  return { dist: Math.hypot(px - qx, py - qy), qx, qy };
}

function numText(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Object.is(value, -0)) return "0";
  return formatNumber(value, 4);
}

/** 轴名过长时用 x/y 简写，避免提示框被一个长标题撑满。 */
function shortAxisName(label: string, fallback: string): string {
  const text = label.trim();
  if (text === "") return fallback;
  return [...text].length <= 12 ? text : fallback;
}

function candidateHit(
  scene: Scene,
  axis: Axis,
  px: number,
  py: number,
  radius: number,
  labels?: HoverOptions["labels"],
  colorLabels?: HoverOptions["colorLabels"],
): HoverHit | null {
  let best: HoverHit | null = null;
  let bestDist = radius;
  let bestIndex = -1;

  const consider = (hit: HoverHit, dist: number): void => {
    if (dist > bestDist + 1e-9) return;
    // 同距离时后画的图层在上面，应该优先命中。
    if (dist < bestDist - 1e-9 || hit.layerIndex >= bestIndex) {
      best = hit;
      bestDist = dist;
      bestIndex = hit.layerIndex;
    }
  };

  for (let layerIndex = 0; layerIndex < scene.layers.length; layerIndex += 1) {
    const layer = scene.layers[layerIndex] as Layer;
    if (layer.axes !== axis.id) continue;
    const xName = shortAxisName(labels?.[axis.id]?.x ?? axis.xlabel, "横轴");
    const yName = shortAxisName(labels?.[axis.id]?.y ?? axis.ylabel, "纵轴");

    if (layer.kind === "line") {
      for (const s of layer.series) {
        let prev: [number, number] | null = null;
        for (let i = 0; i < Math.min(s.x.length, s.y.length); i += 1) {
          const xv = s.x[i] as number;
          const yv = s.y[i] as number;
          if (!finitePair(xv, yv)) {
            prev = null;
            continue;
          }
          const [qx, qy] = dataToPixel(axis, xv, yv);
          if (!Number.isFinite(qx) || !Number.isFinite(qy)) {
            prev = null;
            continue;
          }
          if (prev !== null) {
            const seg = segDistance(px, py, prev[0], prev[1], qx, qy);
            if (seg.dist <= bestDist && insideAxis(axis, seg.qx, seg.qy)) {
              const [dx, dy] = pixelToData(axis, seg.qx, seg.qy);
              consider(
                {
                  layerIndex,
                  kind: layer.kind,
                  axisId: axis.id,
                  label: s.label?.trim() || colorLabels?.[s.color.toLowerCase()] || "曲线",
                  color: s.color,
                  x: dx,
                  y: dy,
                  px: seg.qx,
                  py: seg.qy,
                  xText: `${xName} = ${numText(dx)}`,
                  yText: `${yName} = ${numText(dy)}`,
                  shape: "point",
                  dash: s.dash !== undefined && s.dash.length > 0,
                  axisRect: axis.rect,
                },
                seg.dist,
              );
            }
          } else if (insideAxis(axis, qx, qy)) {
            const dist = Math.hypot(px - qx, py - qy);
            if (dist <= bestDist) {
              consider(
                {
                  layerIndex,
                  kind: layer.kind,
                  axisId: axis.id,
                  label: s.label?.trim() || colorLabels?.[s.color.toLowerCase()] || "曲线",
                  color: s.color,
                  x: xv,
                  y: yv,
                  px: qx,
                  py: qy,
                  xText: `${xName} = ${numText(xv)}`,
                  yText: `${yName} = ${numText(yv)}`,
                  shape: "point",
                  dash: s.dash !== undefined && s.dash.length > 0,
                  axisRect: axis.rect,
                },
                dist,
              );
            }
          }
          prev = [qx, qy];
        }
      }
      continue;
    }

    if (layer.kind === "scatter") {
      const n = Math.min(layer.x.length, layer.y.length);
      for (let i = 0; i < n; i += 1) {
        const xv = layer.x[i] as number;
        const yv = layer.y[i] as number;
        if (!finitePair(xv, yv)) continue;
        const [qx, qy] = dataToPixel(axis, xv, yv);
        if (!Number.isFinite(qx) || !Number.isFinite(qy) || !insideAxis(axis, qx, qy)) continue;
        const dist = Math.hypot(px - qx, py - qy);
        if (dist > bestDist) continue;
        const note = layer.notes?.[i];
        consider(
          {
            layerIndex,
            kind: layer.kind,
            axisId: axis.id,
            label: layer.label?.trim() || colorLabels?.[layer.color.toLowerCase()] || "数据点",
            color: layer.color,
            x: xv,
            y: yv,
            px: qx,
            py: qy,
            xText: `${xName} = ${numText(xv)}`,
            yText: layer.hideY === true ? "" : `${yName} = ${numText(yv)}`,
            ...(note === undefined || note === "" ? {} : { detail: note }),
            shape: "point",
            dash: false,
            axisRect: axis.rect,
          },
          dist,
        );
      }
      continue;
    }

    if (layer.kind === "bars") {
      const n = Math.min(layer.x.length, layer.values.length);
      const baseline = layer.baseline ?? 0;
      for (let i = 0; i < n; i += 1) {
        const xv = layer.x[i] as number;
        const yv = layer.values[i] as number;
        if (!finitePair(xv, yv)) continue;
        const [cx] = dataToPixel(axis, xv, 0);
        const [halfPx] = dataToPixel(axis, xv + layer.width / 2, 0);
        const [, yPx] = dataToPixel(axis, xv, yv);
        const [, bPx] = dataToPixel(axis, xv, baseline);
        if (!Number.isFinite(cx) || !Number.isFinite(yPx) || !Number.isFinite(bPx)) continue;
        const half = Math.max(3, Math.abs(halfPx - cx));
        const top = Math.min(yPx, bPx);
        const bottom = Math.max(yPx, bPx);
        if (Math.abs(px - cx) > half + 2 || py < top - 2 || py > bottom + 2) continue;
        const err = layer.errors?.[i];
        const detail = err !== undefined && Number.isFinite(err) ? `误差 ±${numText(err)}` : undefined;
        const dist = Math.min(Math.abs(px - cx), 1);
        consider(
          {
            layerIndex,
            kind: layer.kind,
            axisId: axis.id,
            label: layer.label?.trim() || "柱",
            color: layer.color,
            x: xv,
            y: yv,
            px: cx,
            py: Math.min(top, Math.max(yPx, top)),
            xText: `${xName} = ${numText(xv)}`,
            yText: `${yName} = ${numText(yv)}`,
            ...(detail === undefined ? {} : { detail }),
            shape: "bar",
            dash: false,
            axisRect: axis.rect,
          },
          dist,
        );
      }
      continue;
    }

    if (layer.kind === "heat") {
      const [ex0, ex1, ey0, ey1] = layer.extent;
      const [dx, dy] = pixelToData(axis, px, py);
      if (!(dx >= ex0 && dx <= ex1 && dy >= ey0 && dy <= ey1)) continue;
      const fx = layer.nx > 1 ? (dx - ex0) / (ex1 - ex0) : 0;
      const fy = layer.ny > 1 ? (dy - ey0) / (ey1 - ey0) : 0;
      const i = Math.max(0, Math.min(layer.nx - 1, Math.round(fx * (layer.nx - 1))));
      const j = Math.max(0, Math.min(layer.ny - 1, Math.round(fy * (layer.ny - 1))));
      const value = layer.values[j * layer.nx + i];
      if (value === undefined || !Number.isFinite(value)) continue;
      const xv = gridCoord(i, layer.nx, ex0, ex1);
      const yv = gridCoord(j, layer.ny, ey0, ey1);
      const [qx, qy] = dataToPixel(axis, xv, yv);
      if (!Number.isFinite(qx) || !Number.isFinite(qy) || !insideAxis(axis, qx, qy)) continue;
      consider(
        {
          layerIndex,
          kind: layer.kind,
          axisId: axis.id,
          label: layer.label?.trim() || "网格数值",
          color: "#94a3b8",
          x: xv,
          y: yv,
          px: qx,
          py: qy,
          xText: `${xName} = ${numText(xv)}`,
          yText: `${yName} = ${numText(yv)}`,
          detail: `数值 = ${numText(value)}`,
          shape: "cell",
          dash: false,
          axisRect: axis.rect,
        },
        radius,
      );
      continue;
    }

    if (layer.kind === "vectorfield") {
      const [ex0, ex1, ey0, ey1] = layer.extent;
      let nearest: { i: number; j: number; px: number; py: number; dist: number } | null = null;
      for (let j = 0; j < layer.ny; j += layer.stride) {
        for (let i = 0; i < layer.nx; i += layer.stride) {
          const xv = gridCoord(i, layer.nx, ex0, ex1);
          const yv = gridCoord(j, layer.ny, ey0, ey1);
          const [qx, qy] = dataToPixel(axis, xv, yv);
          if (!Number.isFinite(qx) || !Number.isFinite(qy) || !insideAxis(axis, qx, qy)) continue;
          const dist = Math.hypot(px - qx, py - qy);
          if (dist <= bestDist && (nearest === null || dist < nearest.dist)) nearest = { i, j, px: qx, py: qy, dist };
        }
      }
      if (nearest !== null) {
        const k = nearest.j * layer.nx + nearest.i;
        const uv = layer.u[k] as number;
        const vv = layer.v[k] as number;
        if (Number.isFinite(uv) && Number.isFinite(vv)) {
          const [dx, dy] = pixelToData(axis, nearest.px, nearest.py);
          consider(
            {
              layerIndex,
              kind: layer.kind,
              axisId: axis.id,
              label: "向量场",
              color: layer.color,
              x: dx,
              y: dy,
              px: nearest.px,
              py: nearest.py,
              xText: `${xName} = ${numText(dx)}`,
              yText: `${yName} = ${numText(dy)}`,
              detail: `u = ${numText(uv)}，v = ${numText(vv)}，模长 = ${numText(Math.hypot(uv, vv))}`,
              shape: "vector",
              dash: false,
              axisRect: axis.rect,
            },
            nearest.dist,
          );
        }
      }
      continue;
    }

    if (layer.kind === "loglogfit") {
      let prev: [number, number] | null = null;
      for (let i = 0; i < Math.min(layer.h.length, layer.err.length); i += 1) {
        const hv = layer.h[i] as number;
        const ev = layer.err[i] as number;
        if (!finitePair(hv, ev) || !(hv > 0) || !(ev > 0)) {
          prev = null;
          continue;
        }
        const [qx, qy] = dataToPixel(axis, hv, ev);
        if (!Number.isFinite(qx) || !Number.isFinite(qy)) {
          prev = null;
          continue;
        }
        if (prev !== null) {
          const seg = segDistance(px, py, prev[0], prev[1], qx, qy);
          if (seg.dist <= bestDist && insideAxis(axis, seg.qx, seg.qy)) {
            const [dx, dy] = pixelToData(axis, seg.qx, seg.qy);
            consider(
              {
                layerIndex,
                kind: layer.kind,
                axisId: axis.id,
                label: layer.seriesLabel,
                color: layer.color ?? "#2b6cb0",
                x: dx,
                y: dy,
                px: seg.qx,
                py: seg.qy,
                xText: `步长 h = ${numText(dx)}`,
                yText: `误差 = ${numText(dy)}`,
                detail: `实测斜率 ${numText(layer.measuredSlope)}　理论斜率 ${numText(layer.theorySlope)}`,
                shape: "fit",
                dash: false,
                axisRect: axis.rect,
              },
              seg.dist,
            );
          }
        }
        prev = [qx, qy];
      }
      continue;
    }
  }

  return best;
}

/**
 * 命中测试入口。`px/py` 是画布逻辑坐标（左上角为原点，未经 CSS 缩放）。
 */
export function hoverAt(scene: Scene, px: number, py: number, opts: HoverOptions = {}): HoverHit | null {
  const radius = opts.radius ?? 14;
  let best: HoverHit | null = null;
  for (const axis of scene.axes) {
    if (!insideAxis(axis, px, py, 2)) continue;
    const hit = candidateHit(scene, axis, px, py, radius, opts.labels, opts.colorLabels);
    if (hit === null) continue;
    const dist = Math.hypot(hit.px - px, hit.py - py);
    if (best === null) {
      best = hit;
    } else {
      const bestDist = Math.hypot(best.px - px, best.py - py);
      if (dist < bestDist) best = hit;
    }
  }
  return best;
}

/** 给读屏/调试用的短句。 */
export function hoverDescription(hit: HoverHit): string {
  return [hit.label, hit.xText, hit.yText, hit.detail].filter((part) => part !== undefined && part !== "").join("；");
}
