/** 6.1 章（二维综合仿真）四个 model 共用的显示工具。 */

import { formatNumber } from "../../shared/src/axes.js";
import { headerBand, palette, type Palette } from "../../shared/src/model.js";
import { pitfallsBand, type Scene, type TextSpec } from "../../shared/src/scene.js";
import { TEXT_ROW_H, textBlock, unitBudget } from "../_common/panel.js";

export const CHAPTER = "6.1";

/**
 * 光栅化器会为每个 heat 轴自动画色标（轴右缘 +46px 起、宽 14px，再往右是刻度）。
 * 热图轴必须从右侧文字/邻轴里让出这段宽度，否则色标与说明叠字。
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

/** 侧栏数字的统一写法；缺失/非有限写 `?`（模型不猜）。 */
export function fmt(v: number | undefined, sig = 4): string {
  return v === undefined || !Number.isFinite(v) ? "?" : formatNumber(v, sig);
}

/** 稳健的 `[min,max]`，空/非有限时给 `fallback`。 */
export function extentOf(values: readonly ArrayLike<number>[], fallback: [number, number] = [0, 1]): [number, number] {
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
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : fallback;
}

/** 对称范围（涡量的发散色标）：`[−m, +m]`，全零时 `[−1, 1]`。 */
export function symmetricExtent(values: readonly ArrayLike<number>[]): [number, number] {
  const [lo, hi] = extentOf(values, [-1, 1]);
  const m = Math.max(Math.abs(lo), Math.abs(hi));
  return m > 0 ? [-m, m] : [-1, 1];
}

/** 按比例加边距；退化时扩到 ±1。 */
export function padRange(lo: number, hi: number, pad = 0.06): [number, number] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-1, 1];
  if (!(hi > lo)) {
    const c = Math.max(1e-9, Math.abs(lo) * 0.1);
    return [lo - c, hi + c];
  }
  const d = (hi - lo) * pad;
  return [lo - d, hi + d];
}

/**
 * 左列「方形热图 + 右侧小字 + 下方副轴」的几何。
 *
 * 色标占 `HEAT_RIGHT_GUTTER`，所以热图不是简单用满左列；方形场（方腔、图灵）保持
 * 近 1:1，避免把单位正方形拖成宽银幕。`minInner` 是热图右侧小字的最小宽度。
 */
export function heatColumn(
  left: number,
  leftW: number,
  topY: number,
  topH: number,
  minInner = 150,
): { field: [number, number, number, number]; innerX: number; innerW: number; innerBottom: number } {
  const maxField = Math.max(120, leftW - HEAT_RIGHT_GUTTER - minInner - 8);
  const fieldW = Math.max(120, Math.min(Math.round(topH * 1.05), maxField));
  const fieldH = topH;
  const innerX = left + fieldW + HEAT_RIGHT_GUTTER + 8;
  const innerW = Math.max(80, leftW - fieldW - HEAT_RIGHT_GUTTER - 8);
  return {
    field: [left, topY, fieldW, fieldH],
    innerX,
    innerW,
    innerBottom: topY + fieldH,
  };
}

export interface SingleLayout {
  main: [number, number, number, number];
  aux: [number, number, number, number];
  left: number;
  topY: number;
  floor: number;
  contentW: number;
}

/** 窄画布的退化版式：主图 + 副图竖直堆叠，右侧说明列取消。 */
export function singleColumn(scene: Scene, width: number, height: number, mainShare = 0.58): SingleLayout {
  const left = 64;
  const right = 24;
  const topY = headerBand(width) + 10;
  const panelTop = pitfallsBand(scene)?.[1] ?? height;
  const floor = panelTop - 44;
  const contentW = Math.max(180, width - left - right);
  const avail = Math.max(180, floor - topY);
  const gap = 56;
  const mainH = Math.max(92, Math.round((avail - gap) * mainShare));
  return {
    main: [left, topY, contentW, mainH],
    aux: [left, topY + mainH + gap, contentW, Math.max(56, avail - mainH - gap)],
    left,
    topY,
    floor,
    contentW,
  };
}

/** 每隔 `stride` 个元素取一个下标（只用于显示采样，不改变数据）。 */
export function decimateIndices(n: number, stride: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i += Math.max(1, Math.floor(stride))) out.push(i);
  return out;
}

/** 一行状态文字，用全角竖线分隔，便于角标裁剪时仍能读出先后。 */
export function statusLine(parts: readonly string[]): string {
  return parts.join("｜");
}

export interface TextGroup {
  lines: readonly string[];
  color: string;
}

/**
 * 右栏说明块：按组依次向下排，组间空一行；超出 floor 时由 `textBlock` 显式截断。
 * 返回实际用到的 y，供调用方（测试/调试）知道底部位置。
 */
export function pushGroups(
  scene: Scene,
  x: number,
  top: number,
  bottom: number,
  width: number,
  groups: readonly TextGroup[],
  pal: Palette,
): number {
  const budget = unitBudget(width);
  let y = top;
  for (const g of groups) {
    if (y >= bottom) break;
    const block: TextSpec[] = textBlock(x, y, bottom, g.lines, g.color, budget, "top");
    scene.texts.push(...block);
    y += block.length * TEXT_ROW_H;
    if (y < bottom) y += TEXT_ROW_H;
  }
  void pal;
  return y;
}

/** 常见配色组，保证四页的图例颜色一致。 */
export function legendColor(pal: Palette, kind: "numeric" | "reference" | "error"): string {
  return kind === "numeric" ? (pal.curves[0] as string) : kind === "reference" ? pal.exact : pal.residual;
}

/** 默认调色板（供模型内联文本颜色；避免每处重复 `palette(opts.theme)`）。 */
export { palette };
