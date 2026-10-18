/**
 * Scene → Canvas 2D 的渲染器。
 *
 * 与 `tools/raster.py`（Python 侧同一套 Scene → PNG）保持"同构"：
 * 同样的分层顺序、同样的裁剪规则、同样的徽标位置。因此视觉回归不需要浏览器。
 *
 * 渲染原则（与防误导规则对应）：
 *  1. 曲线在 NaN 处**断开**，绝不用直线把空洞连起来（`geometry.layerToPixelPath`）；
 *  2. 热力图的"无数据"格子画成中性灰而不是透明（`geometry.buildHeatRgba`）；
 *  3. 徽标永远画在最上层：坐标轴模式、数据来源、溢出、临界值都必须可见；
 *  4. 溢出（被轴裁掉）时不静默：由 `checkOverflow` 的结果驱动徽标文本。
 */

import { cmapTable } from "./colorscale.js";
import { formatNumber, tickLabel } from "./axes.js";
import {
  buildHeatRgba,
  contourSegments,
  dataToPixel,
  gridCoord,
  layerToPixelPath,
} from "./geometry.js";
import type { Axis, Badge, CmapName, Layer, Scene, TextSpec } from "./scene.js";

export interface RenderOptions {
  /** 设备像素比；>1 时渲染器会按 dpr 放大画布并缩放上下文（保证文字清晰）。 */
  dpr?: number;
  fontFamily?: string;
  baseFontSize?: number;
  /** 是否绘制徽标（快照对比时可关掉以隔离图形差异）。 */
  showBadges?: boolean;
  /** 背景色（缺省取 scene.canvas.background）。 */
  background?: string;
  /** 是否画轴标题（缺省 true）。 */
  showTitles?: boolean;
}

interface Ctx {
  ctx: CanvasRenderingContext2D;
  scene: Scene;
  opts: Required<Pick<RenderOptions, "fontFamily" | "baseFontSize" | "showBadges" | "showTitles">>;
}

const DEFAULT_FONT = '13px -apple-system, "Segoe UI", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif';

function fontOf(c: Ctx, size: number, weight = "400"): string {
  void c;
  return `${weight} ${size}px ${DEFAULT_FONT.split(" ").slice(1).join(" ")}`;
}

/** 只保留轴上可见区间的折线（避免把 1e300 的越界点真的画出去）。 */
function clipSeriesToAxis(axis: Axis, path: (number | null)[]): (number | null)[] {
  const [rx, ry, rw, rh] = axis.rect;
  const x0 = rx;
  const x1 = rx + rw;
  const y0 = ry;
  const y1 = ry + rh;
  // 只要坐标是有限数就交给 canvas 的 clip() 处理；这里只过滤明显荒谬的值，
  // 避免某些后端（或巨大坐标）让路径膨胀到不可用。
  const out = path.slice();
  for (let i = 0; i < out.length; i += 2) {
    const px = out[i];
    const py = out[i + 1];
    if (px === null || py === null) continue;
    const far = 1e6;
    if (px < x0 - far || px > x1 + far || py < y0 - far || py > y1 + far) {
      out[i] = null;
      out[i + 1] = null;
    }
  }
  return out;
}

function strokePath(ctx: CanvasRenderingContext2D, path: (number | null)[]): void {
  ctx.beginPath();
  let pen = false;
  for (let i = 0; i < path.length; i += 2) {
    const px = path[i];
    const py = path[i + 1];
    if (px === null || py === null) {
      pen = false;
      continue;
    }
    if (!pen) {
      ctx.moveTo(px, py);
      pen = true;
    } else {
      ctx.lineTo(px, py);
    }
  }
  ctx.stroke();
}

// --------------------------------------------------------------------------- //
// 坐标轴
// --------------------------------------------------------------------------- //
function drawAxis(c: Ctx, axis: Axis): void {
  const { ctx } = c;
  const [rx, ry, rw, rh] = axis.rect;
  const x1 = rx + rw;
  const y1 = ry + rh;
  ctx.save();
  if (axis.grid) {
    ctx.strokeStyle = "#e8e8e8";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const t of axis.xticks) {
      const [px] = dataToPixel(axis, t, axis.ylim[0]);
      if (px >= rx && px <= x1) {
        ctx.moveTo(px, ry);
        ctx.lineTo(px, y1);
      }
    }
    for (const t of axis.yticks) {
      const [, py] = dataToPixel(axis, axis.xlim[0], t);
      if (py >= ry && py <= y1) {
        ctx.moveTo(rx, py);
        ctx.lineTo(x1, py);
      }
    }
    ctx.stroke();
  }
  if (axis.frame) {
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.strokeRect(rx + 0.5, ry + 0.5, rw - 1, rh - 1);
  }
  ctx.fillStyle = "#333";
  ctx.font = fontOf(c, c.opts.baseFontSize - 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of axis.xticks) {
    const [px] = dataToPixel(axis, t, axis.ylim[0]);
    if (px < rx - 1 || px > x1 + 1) continue;
    ctx.fillText(tickLabel(t, { log: axis.xscale === "log" }), px, y1 + 4);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of axis.yticks) {
    const [, py] = dataToPixel(axis, axis.xlim[0], t);
    if (py < ry - 1 || py > y1 + 1) continue;
    ctx.fillText(tickLabel(t, { log: axis.yscale === "log" }), rx - 6, py);
  }
  if (c.opts.showTitles) {
    ctx.fillStyle = "#222";
    ctx.font = fontOf(c, c.opts.baseFontSize, "600");
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    if (axis.xlabel) ctx.fillText(axis.xlabel, rx + rw / 2, y1 + c.opts.baseFontSize + 16);
    if (axis.title) ctx.fillText(axis.title, rx + rw / 2, ry - 6);
    if (axis.ylabel) {
      ctx.save();
      ctx.translate(rx - 42, ry + rh / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = "top";
      ctx.fillText(axis.ylabel, 0, 0);
      ctx.restore();
    }
  }
  ctx.restore();
}

// --------------------------------------------------------------------------- //
// 各图层
// --------------------------------------------------------------------------- //
function drawLineLayer(c: Ctx, axis: Axis, layer: Extract<Layer, { kind: "line" }>): void {
  const { ctx } = c;
  for (const s of layer.series) {
    const path = clipSeriesToAxis(axis, layerToPixelPath(axis, s.x, s.y));
    ctx.save();
    ctx.globalAlpha = s.alpha ?? 1;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width;
    ctx.lineJoin = "round";
    ctx.setLineDash(s.dash ?? []);
    strokePath(ctx, path);
    ctx.setLineDash([]);
    const marker = s.marker ?? "none";
    if (marker !== "none") {
      ctx.fillStyle = s.color;
      ctx.strokeStyle = s.color;
      for (let i = 0; i < path.length; i += 2) {
        const px = path[i];
        const py = path[i + 1];
        if (px === null || py === null) continue;
        if (path[i - 1] === null && i > 1) continue; // 断点后不画标记
        const r = Math.max(2, s.width * 1.6);
        if (marker === "o") {
          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fill();
        } else if (marker === "s") {
          ctx.fillRect(px - r, py - r, 2 * r, 2 * r);
        } else {
          ctx.beginPath();
          ctx.moveTo(px - r, py - r);
          ctx.lineTo(px + r, py + r);
          if (marker === "+") {
            ctx.moveTo(px - r, py + r);
            ctx.lineTo(px + r, py - r);
          }
          ctx.stroke();
        }
      }
    }
    if (s.label) {
      // 曲线标签放在最后一个可见点附近，避免压住图形主体
      const pts: [number, number][] = [];
      for (let i = 0; i < path.length; i += 2) {
        const px = path[i];
        const py = path[i + 1];
        if (px !== null && py !== null) pts.push([px, py]);
      }
      const last = pts[pts.length - 1];
      if (last) {
        ctx.fillStyle = s.color;
        ctx.font = fontOf(c, c.opts.baseFontSize - 1);
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(s.label, last[0] + 4, last[1] - 2);
      }
    }
    ctx.restore();
  }
}

function drawBandLayer(c: Ctx, axis: Axis, layer: Extract<Layer, { kind: "band" }>): void {
  const { ctx } = c;
  const b = layer.band;
  const n = Math.min(b.x.length, b.ylo.length, b.yhi.length);
  ctx.save();
  ctx.globalAlpha = b.alpha;
  ctx.fillStyle = b.color;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < n; i += 1) {
    const [px, py] = dataToPixel(axis, b.x[i] as number, b.yhi[i] as number);
    if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
    if (!started) {
      ctx.moveTo(px, py);
      started = true;
    } else ctx.lineTo(px, py);
  }
  for (let i = n - 1; i >= 0; i -= 1) {
    const [px, py] = dataToPixel(axis, b.x[i] as number, b.ylo[i] as number);
    if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
    ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawScatterLayer(c: Ctx, axis: Axis, layer: Extract<Layer, { kind: "scatter" }>): void {
  const { ctx } = c;
  ctx.save();
  ctx.globalAlpha = layer.alpha ?? 1;
  ctx.fillStyle = layer.color;
  for (let i = 0; i < Math.min(layer.x.length, layer.y.length); i += 1) {
    if (!Number.isFinite(layer.x[i] as number) || !Number.isFinite(layer.y[i] as number)) continue;
    const [px, py] = dataToPixel(axis, layer.x[i] as number, layer.y[i] as number);
    if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
    ctx.beginPath();
    ctx.arc(px, py, layer.r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function withClippedAxis(c: Ctx, axis: Axis, draw: () => void): void {
  const { ctx } = c;
  const [rx, ry, rw, rh] = axis.rect;
  ctx.save();
  ctx.beginPath();
  ctx.rect(rx, ry, rw, rh);
  ctx.clip();
  draw();
  ctx.restore();
}

function drawHeatLayer(c: Ctx, axis: Axis, layer: Extract<Layer, { kind: "heat" }>): void {
  const { ctx } = c;
  const rgba = buildHeatRgba(layer.values, layer.nx, layer.ny, layer.cmap as CmapName, layer.vmin, layer.vmax, {
    logScale: layer.logScale,
    clipBelow: layer.clipBelow,
  });
  const [x0, y0] = dataToPixel(axis, layer.extent[0], layer.extent[3]);
  const [x1, y1] = dataToPixel(axis, layer.extent[1], layer.extent[2]);
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const off = makeOffscreen(layer.nx, layer.ny);
  withClippedAxis(c, axis, () => {
    if (off) {
      // 快路径：原始分辨率像素整体缩放贴图（保持"每格一像素"的锐利）
      off.ctx.putImageData(new ImageData(Uint8ClampedArray.from(rgba), layer.nx, layer.ny), 0, 0);
      const prev = ctx.imageSmoothingEnabled;
      ctx.imageSmoothingEnabled = false;
      // **纵向翻转**：`values` 的第 0 行对应 extent[2]（慢路径用 dataToPixel 把 j=0 画在底部），
      // 而 drawImage 把图像第 0 行放在矩形**顶部**。不翻就会上下颠倒——慢路径与快路径
      // 对同一份数据画出两张镜像图（光栅器 D 在 `tools/raster.py` 里复现过这一点）。
      const left = Math.min(x0, x1);
      const top = Math.min(y0, y1);
      ctx.save();
      ctx.translate(left, top + dy);
      ctx.scale(1, -1);
      ctx.drawImage(off.canvas, 0, 0, dx, dy);
      ctx.restore();
      ctx.imageSmoothingEnabled = prev;
      return;
    }
    // 慢路径（无离屏画布的宿主，如 Node 上的假上下文）：逐格填充，绝不静默跳过
    const cw = layer.nx > 1 ? dx / (layer.nx - 1) : dx;
    const ch = layer.ny > 1 ? dy / (layer.ny - 1) : dy;
    for (let j = 0; j < layer.ny; j += 1) {
      for (let i = 0; i < layer.nx; i += 1) {
        const p = j * layer.nx + i;
        ctx.fillStyle = `rgba(${rgba[4 * p]},${rgba[4 * p + 1]},${rgba[4 * p + 2]},${((rgba[4 * p + 3] as number) / 255).toFixed(3)})`;
        const [px, py] = dataToPixel(
          axis,
          gridCoord(i, layer.nx, layer.extent[0], layer.extent[1]),
          gridCoord(j, layer.ny, layer.extent[2], layer.extent[3]),
        );
        if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
        ctx.fillRect(px - cw / 2, py - ch / 2, cw, ch);
      }
    }
  });
}

/** 离屏画布工厂（浏览器用 canvas，Node 里返回 null 以便纯逻辑测试不依赖 DOM）。 */
function makeOffscreen(w: number, h: number): { canvas: CanvasImageSource; ctx: CanvasRenderingContext2D } | null {
  if (typeof OffscreenCanvas !== "undefined") {
    const cv = new OffscreenCanvas(w, h);
    const ctx2 = cv.getContext("2d") as CanvasRenderingContext2D | null;
    if (!ctx2) return null;
    return { canvas: cv as unknown as CanvasImageSource, ctx: ctx2 };
  }
  if (typeof document !== "undefined") {
    const cv = document.createElement("canvas");
    cv.width = w;
    cv.height = h;
    const ctx2 = cv.getContext("2d");
    if (!ctx2) return null;
    return { canvas: cv, ctx: ctx2 };
  }
  return null;
}

function drawContourLayer(c: Ctx, axis: Axis, layer: Extract<Layer, { kind: "contour" }>): void {
  const { ctx } = c;
  const [ex0, ex1, ey0, ey1] = layer.extent;
  const segs = contourSegments(layer.values, layer.nx, layer.ny, layer.levels);
  withClippedAxis(c, axis, () => {
    ctx.save();
    ctx.strokeStyle = layer.color;
    ctx.lineWidth = layer.width;
    for (const { segments } of segs) {
      ctx.beginPath();
      for (let i = 0; i < segments.length; i += 4) {
        const [px0, py0] = dataToPixel(
          axis,
          gridCoord(segments[i] as number, layer.nx, ex0, ex1),
          gridCoord(segments[i + 1] as number, layer.ny, ey0, ey1),
        );
        const [px1, py1] = dataToPixel(
          axis,
          gridCoord(segments[i + 2] as number, layer.nx, ex0, ex1),
          gridCoord(segments[i + 3] as number, layer.ny, ey0, ey1),
        );
        if (!Number.isFinite(px0) || !Number.isFinite(py0) || !Number.isFinite(px1) || !Number.isFinite(py1)) continue;
        ctx.moveTo(px0, py0);
        ctx.lineTo(px1, py1);
      }
      ctx.stroke();
    }
    ctx.restore();
  });
}

function drawBarsLayer(c: Ctx, axis: Axis, layer: Extract<Layer, { kind: "bars" }>): void {
  const { ctx } = c;
  const base = layer.baseline ?? 0;
  withClippedAxis(c, axis, () => {
    ctx.save();
    ctx.fillStyle = layer.color;
    ctx.strokeStyle = layer.color;
    ctx.lineWidth = 1;
    for (let i = 0; i < Math.min(layer.x.length, layer.values.length); i += 1) {
      const xv = layer.x[i] as number;
      const yv = layer.values[i] as number;
      if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
      // 柱宽以数据单位给出
      const half = layer.width / 2;
      const [pxa, pya] = dataToPixel(axis, xv - half, yv);
      const [pxb, pyb] = dataToPixel(axis, xv + half, base);
      if (!Number.isFinite(pxa) || !Number.isFinite(pya) || !Number.isFinite(pxb) || !Number.isFinite(pyb)) continue;
      ctx.fillRect(Math.min(pxa, pxb), Math.min(pya, pyb), Math.abs(pxb - pxa), Math.abs(pyb - pya));
      const err = layer.errors?.[i];
      if (err !== undefined && Number.isFinite(err) && err > 0) {
        const [xm] = dataToPixel(axis, xv, base);
        const [, yTop] = dataToPixel(axis, xv, yv + err);
        const [, yBot] = dataToPixel(axis, xv, Math.max(base, yv - err));
        ctx.beginPath();
        ctx.moveTo(xm, yTop);
        ctx.lineTo(xm, yBot);
        const cap = Math.max(2, Math.abs(pxb - pxa) / 4);
        ctx.moveTo(xm - cap, yTop);
        ctx.lineTo(xm + cap, yTop);
        ctx.moveTo(xm - cap, yBot);
        ctx.lineTo(xm + cap, yBot);
        ctx.stroke();
      }
    }
    ctx.restore();
  });
}

function drawVectorFieldLayer(c: Ctx, axis: Axis, layer: Extract<Layer, { kind: "vectorfield" }>): void {
  const { ctx } = c;
  const [ex0, ex1, ey0, ey1] = layer.extent;
  withClippedAxis(c, axis, () => {
    ctx.save();
    ctx.strokeStyle = layer.color;
    ctx.fillStyle = layer.color;
    ctx.lineWidth = 1;
    for (let j = 0; j < layer.ny; j += layer.stride) {
      for (let i = 0; i < layer.nx; i += layer.stride) {
        const k = j * layer.nx + i;
        const xv = gridCoord(i, layer.nx, ex0, ex1);
        const yv = gridCoord(j, layer.ny, ey0, ey1);
        const uv = layer.u[k] as number;
        const vv = layer.v[k] as number;
        if (!Number.isFinite(uv) || !Number.isFinite(vv)) continue;
        const [px, py] = dataToPixel(axis, xv, yv);
        const [px2, py2] = dataToPixel(axis, xv + uv * layer.scale, yv + vv * layer.scale);
        if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(px2) || !Number.isFinite(py2)) continue;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px2, py2);
        ctx.stroke();
        const ang = Math.atan2(py2 - py, px2 - px);
        const head = 4;
        ctx.beginPath();
        ctx.moveTo(px2, py2);
        ctx.lineTo(px2 - head * Math.cos(ang - 0.4), py2 - head * Math.sin(ang - 0.4));
        ctx.lineTo(px2 - head * Math.cos(ang + 0.4), py2 - head * Math.sin(ang + 0.4));
        ctx.closePath();
        ctx.fill();
      }
    }
    ctx.restore();
  });
}

function drawLoglogFitLayer(c: Ctx, axis: Axis, layer: Extract<Layer, { kind: "loglogfit" }>): void {
  const { ctx } = c;
  const color = layer.color ?? "#2b6cb0";
  const theoryColor = layer.theoryColor ?? "#c53030";
  // 实测数据
  const path = clipSeriesToAxis(axis, layerToPixelPath(axis, layer.h, layer.err));
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.6;
  strokePath(ctx, path);
  for (let i = 0; i < path.length; i += 2) {
    const px = path[i];
    const py = path[i + 1];
    if (px === null || py === null) continue;
    ctx.beginPath();
    ctx.arc(px, py, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
  // 理论参考线：经过实测曲线的末点，斜率固定为 theorySlope（双对数下是直线）
  let anchor: [number, number] | null = null;
  for (let i = path.length - 2; i >= 0; i -= 2) {
    const px = path[i];
    const py = path[i + 1];
    if (px !== null && py !== null) {
      anchor = [px, py];
      break;
    }
  }
  const [x0, x1] = layer.refX;
  if (anchor && x0 > 0 && x1 > 0) {
    const [ax0] = dataToPixel(axis, x0, layer.err[0] as number);
    const [ax1] = dataToPixel(axis, x1, layer.err[layer.err.length - 1] as number);
    const logSpan = Math.log10(x1) - Math.log10(x0);
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = theoryColor;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    for (let s = 0; s <= 1; s += 0.02) {
      const lx = Math.log10(x0) + s * logSpan;
      const ly = Math.log10(layer.err[layer.err.length - 1] as number) + layer.theorySlope * (lx - Math.log10(x1));
      const [px, py] = dataToPixel(axis, Math.pow(10, lx), Math.pow(10, ly));
      if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      if (s === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    void ax0;
    void ax1;
  }
  // 两条线的文字标注（规则 2：缺一不可）
  const yTop = axis.rect[1] + 14;
  ctx.font = fontOf(c, c.opts.baseFontSize - 1);
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillStyle = color;
  ctx.fillText(layer.seriesLabel, axis.rect[0] + axis.rect[2] - 8, yTop);
  ctx.fillStyle = theoryColor;
  ctx.fillText(layer.theoryLabel, axis.rect[0] + axis.rect[2] - 8, yTop + c.opts.baseFontSize + 2);
  ctx.restore();
}

function drawLayer(c: Ctx, layer: Layer): void {
  const axis = c.scene.axes.find((a) => a.id === layer.axes);
  if (!axis) return;
  switch (layer.kind) {
    case "line":
      drawLineLayer(c, axis, layer);
      break;
    case "band":
      drawBandLayer(c, axis, layer);
      break;
    case "scatter":
      drawScatterLayer(c, axis, layer);
      break;
    case "heat":
      drawHeatLayer(c, axis, layer);
      break;
    case "contour":
      drawContourLayer(c, axis, layer);
      break;
    case "bars":
      drawBarsLayer(c, axis, layer);
      break;
    case "vectorfield":
      drawVectorFieldLayer(c, axis, layer);
      break;
    case "loglogfit":
      drawLoglogFitLayer(c, axis, layer);
      break;
    default:
      break;
  }
}

// --------------------------------------------------------------------------- //
// 文本、色标、徽标
// --------------------------------------------------------------------------- //
function drawText(c: Ctx, t: TextSpec): void {
  const { ctx } = c;
  ctx.save();
  ctx.font = fontOf(c, t.size, t.role === "title" ? "600" : "400");
  ctx.fillStyle = t.color;
  ctx.textAlign = t.align;
  ctx.textBaseline = "top";
  const lines = t.maxWidth ? wrapText(ctx, t.text, t.maxWidth) : [t.text];
  lines.forEach((line, i) => ctx.fillText(line, t.x, t.y + i * (t.size + 4)));
  ctx.restore();
}

function wrapText(ctx: CanvasRenderingContext2D, body: string, maxWidth: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const ch of body) {
    const probe = line + ch;
    if (ctx.measureText(probe).width > maxWidth && line.length > 0) {
      out.push(line);
      line = ch;
    } else {
      line = probe;
    }
  }
  if (line.length > 0) out.push(line);
  return out;
}

/** 色标条（热力图必备，否则读不出数值）。 */
export function drawColorbar(
  ctx: CanvasRenderingContext2D,
  cmap: CmapName,
  vmin: number,
  vmax: number,
  rect: [number, number, number, number],
  opts: { label?: string; logScale?: boolean; font?: string } = {},
): void {
  const [x, y, w, h] = rect;
  const table = cmapTable(cmap);
  const steps = Math.max(2, Math.round(h));
  for (let i = 0; i < steps; i += 1) {
    const t = 1 - i / (steps - 1);
    const idx = Math.min(255, Math.max(0, Math.round(t * 255)));
    const r = table[3 * idx] as number;
    const g = table[3 * idx + 1] as number;
    const b = table[3 * idx + 2] as number;
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, y + (i * h) / steps, w, h / steps + 1);
  }
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  ctx.fillStyle = "#222";
  ctx.font = opts.font ?? DEFAULT_FONT;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const labels = opts.logScale ? [vmax, Math.sqrt(vmin * vmax), vmin] : [vmax, (vmin + vmax) / 2, vmin];
  labels.forEach((v, i) => {
    ctx.fillText(formatNumber(v, 4), x + w + 4, y + (i * h) / (labels.length - 1));
  });
  if (opts.label) {
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(opts.label, x + w / 2, y + h + 4);
  }
}

function drawBadge(c: Ctx, b: Badge): void {
  const { ctx } = c;
  ctx.save();
  ctx.font = fontOf(c, c.opts.baseFontSize - 2, "600");
  const pad = 5;
  const w = ctx.measureText(b.text).width + 2 * pad;
  const h = c.opts.baseFontSize + 4;
  ctx.fillStyle = "rgba(255,255,255,0.86)";
  ctx.fillRect(b.x, b.y, w, h);
  ctx.strokeStyle = b.color ?? "#4a5568";
  ctx.lineWidth = 1;
  ctx.strokeRect(b.x + 0.5, b.y + 0.5, w - 1, h - 1);
  ctx.fillStyle = b.color ?? "#4a5568";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(b.text, b.x + pad, b.y + h / 2);
  ctx.restore();
}

/** 把 pitfalls 面板画在底部（规则 7：它会怎样骗你，必须随图出现）。 */
function drawPitfalls(c: Ctx): void {
  const { ctx, scene } = c;
  const size = c.opts.baseFontSize - 1;
  ctx.save();
  ctx.font = fontOf(c, size);
  const maxW = scene.canvas.w - 24;
  const lines = wrapText(ctx, `它会怎样骗你：${scene.pitfalls}`, maxW);
  const lineH = size + 4;
  const y0 = scene.canvas.h - 10 - lines.length * lineH;
  ctx.fillStyle = "rgba(255,247,237,0.95)";
  ctx.fillRect(8, y0 - 4, scene.canvas.w - 16, lines.length * lineH + 8);
  ctx.strokeStyle = "#dd6b20";
  ctx.strokeRect(8.5, y0 - 3.5, scene.canvas.w - 17, lines.length * lineH + 7);
  ctx.fillStyle = "#7b341e";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  lines.forEach((line, i) => ctx.fillText(line, 14, y0 + i * lineH));
  ctx.restore();
}

// --------------------------------------------------------------------------- //
// 入口
// --------------------------------------------------------------------------- //
/**
 * 渲染整个 Scene。**不**做校验（校验是 `assertValidScene` 的职责），
 * 因为渲染器应当只被喂合法场景；把两者混在一起会让"哪一步出错"变模糊。
 */
export function renderScene(ctx: CanvasRenderingContext2D, scene: Scene, opts: RenderOptions = {}): void {
  const dpr = opts.dpr ?? 1;
  const c: Ctx = {
    ctx,
    scene,
    opts: {
      fontFamily: opts.fontFamily ?? DEFAULT_FONT,
      baseFontSize: opts.baseFontSize ?? 13,
      showBadges: opts.showBadges ?? true,
      showTitles: opts.showTitles ?? true,
    },
  };
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, scene.canvas.w, scene.canvas.h);
  ctx.fillStyle = opts.background ?? scene.canvas.background;
  ctx.fillRect(0, 0, scene.canvas.w, scene.canvas.h);
  if (scene.title) {
    ctx.fillStyle = "#111";
    ctx.font = fontOf(c, c.opts.baseFontSize + 5, "700");
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(scene.title, 12, 8);
  }
  for (const axis of scene.axes) drawAxis(c, axis);
  for (const layer of scene.layers) drawLayer(c, layer);
  for (const t of scene.texts) drawText(c, t);
  if (c.opts.showBadges) for (const b of scene.badges) drawBadge(c, b);
  drawPitfalls(c);
  ctx.restore();
}

/** 便捷入口：直接把 Scene 画到 `<canvas>` 上（自动按 dpr 调整位图尺寸）。 */
export function renderSceneToCanvas(canvas: HTMLCanvasElement, scene: Scene, opts: RenderOptions = {}): void {
  const dpr = opts.dpr ?? (typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1);
  canvas.width = Math.round(scene.canvas.w * dpr);
  canvas.height = Math.round(scene.canvas.h * dpr);
  canvas.style.width = `${scene.canvas.w}px`;
  canvas.style.height = `${scene.canvas.h}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("无法获取 2D 上下文");
  renderScene(ctx, scene, { ...opts, dpr });
}
