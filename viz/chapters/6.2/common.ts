/** 6.2 章（打包与导出）三个 model 共用的显示工具。 */

import { formatNumber } from "../../shared/src/axes.js";
import { headerBand, palette, type Palette } from "../../shared/src/model.js";
import { pitfallsBand, type Scene, type TextSpec } from "../../shared/src/scene.js";
import { TEXT_ROW_H, textBlock, unitBudget } from "../_common/panel.js";

export const CHAPTER = "6.2";

/** 热图右侧的色标 + 刻度带（与光栅器一致，见 6.1/common.ts）。 */
export const HEAT_RIGHT_GUTTER = 110;

export const SLOT = {
  source: 0,
  axisMode: 1,
  critical: 2,
  status: 3,
  overflow: 4,
  degraded: 5,
} as const;

export const VIZ_TITLES = {
  catalog_nav: "目录与依赖：28 章的构建顺序",
  export_timeline: "视频导出：帧数、时长与两本时间账",
  book_build: "讲义构建：引用、附表与字形审计",
} as const;

export function fmt(v: number | undefined, sig = 4): string {
  return v === undefined || !Number.isFinite(v) ? "?" : formatNumber(v, sig);
}

export function jsonList(text: string | undefined): string[] {
  if (!text) return [];
  try {
    const parsed: unknown = JSON.parse(text);
    if (!Array.isArray(parsed)) return [];
    return parsed.map((v) => String(v));
  } catch {
    return [];
  }
}

export function statusLine(parts: readonly string[]): string {
  return parts.join("｜");
}

export interface NarrowLayout {
  left: number;
  topY: number;
  textTop: number;
  contentW: number;
  main: [number, number, number, number];
  aux?: [number, number, number, number];
  floor: number;
}

/**
 * 窄画布的公共版式：顶部最多 3 行说明带 + 主图 +（可选）副图。
 * 说明带放在轴**外**，因此不会压住刻度/标题；底部给 xlabel 与 pitfalls 留 44px。
 */
export function narrowLayout(
  scene: Scene,
  width: number,
  height: number,
  opts: { textRows?: number; mainShare?: number; aux?: boolean } = {},
): NarrowLayout {
  const left = 62;
  const right = 22;
  const topY = headerBand(width) + 8;
  const panelTop = pitfallsBand(scene)?.[1] ?? height;
  const floor = panelTop - 44;
  const contentW = Math.max(200, width - left - right);
  const rows = Math.max(0, opts.textRows ?? 2);
  const textH = rows > 0 ? rows * TEXT_ROW_H + 6 : 0;
  const gap = 54;
  const hasAux = opts.aux ?? true;
  // 轴标题带在轴矩形上方约 16px；顶部说明带必须让开它，否则规则 8 会判"文字压轴装饰"。
  const titlePad = 18;
  const avail = Math.max(150, floor - (topY + textH + titlePad) - (hasAux ? gap : 0));
  const mainH = Math.max(96, Math.round(avail * (opts.mainShare ?? 0.6)));
  const auxTop = topY + textH + titlePad + mainH + gap;
  return {
    left,
    topY,
    textTop: topY + 4,
    contentW,
    main: [left, topY + textH + titlePad, contentW, mainH],
    aux: hasAux ? [left, auxTop, contentW, Math.max(58, floor - auxTop)] : undefined,
    floor,
  };
}

export interface TextGroup {
  lines: readonly string[];
  color: string;
}

/** 右栏分组文字；超额时由 `textBlock` 显式截断（不画到 pitfalls 里）。 */
export function pushGroups(
  scene: Scene,
  x: number,
  top: number,
  bottom: number,
  width: number,
  groups: readonly TextGroup[],
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
  return y;
}

/** 顶部的紧凑说明带（窄画布用）；返回实际行数。 */
export function compactText(
  scene: Scene,
  x: number,
  y: number,
  width: number,
  lines: readonly string[],
  color: string,
  bottom: number,
): number {
  const specs = textBlock(x, y, bottom, lines, color, unitBudget(width), "top");
  scene.texts.push(...specs);
  return specs.length;
}

/** 六个分部的统一颜色（I..VI；0 表示模板/未分部的灰）。 */
export function partColors(pal: Palette): readonly string[] {
  return [pal.muted, pal.curves[0] as string, pal.curves[1] as string, pal.exact, pal.residual, pal.curves[2] as string, pal.marker];
}

export { palette };
