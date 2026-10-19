/**
 * 4.3 章四个 model 共用的**排版几何**与**波型图例**。
 *
 * 只做三件事：
 * 1. 把左列切成 2/3 块等高面板（同一组参数下逐像素固定，拖动时间不抖）；
 * 2. 在右列叠一块误差轴 + 一块账本残差轴，并给出文字区上边界；
 * 3. 五条波位置去重 + 按波型给颜色/线型（kind 的物理含义来自帧，前端只配色）。
 *
 * 不读物理量、不重算后端已经给出的数（P8）。
 */

import type { Palette } from "../../shared/src/model.js";
import { lineLayer, series, type Scene } from "../../shared/src/scene.js";
import { XL_LABEL_BAND, type ColumnsLayout } from "../_common/panel.js";

/** 角标槽位固定：缺一枚时其余不挪位，连续帧不抖。 */
export const SLOT = {
  source: 0,
  axisMode: 1,
  critical: 2,
  status: 3,
  preview: 4,
  overflow: 5,
  degraded: 6,
} as const;

/** 一块面板矩形 `[x, y, w, h]`（与 `Axis.rect` 同约定）。 */
export type Rect = [number, number, number, number];

/**
 * 左列等分 `count` 块面板（2 块即 columnsLayout 的上/下两块）。
 *
 * 底边钉在 `L.botY + L.botH`（= pitfalls 面板顶 − 28），最后一块的 x 轴名装饰带
 * 也落在这条线以上；块间留 `gap`（缺省 44 = 标题带 18 + 刻度/轴名带 28 − 2）。
 */
export function leftStack(L: ColumnsLayout, count: number, gap = 44): Rect[] {
  if (count !== 2 && count !== 3) throw new Error(`4.3 的 leftStack 只支持 2/3 块，收到 ${count}`);
  if (count === 2) {
    return [
      [L.left, L.topY, L.leftW, L.topH],
      [L.left, L.botY, L.leftW, L.botH],
    ];
  }
  const top = L.topY;
  const bottom = L.botY + L.botH;
  const h = Math.max(60, Math.floor((bottom - top - gap * 2) / 3));
  return [
    [L.left, top, L.leftW, h],
    [L.left, top + h + gap, L.leftW, h],
    [L.left, top + 2 * (h + gap), L.leftW, h],
  ];
}

export interface RightStack {
  error: Rect;
  residual: Rect;
  /** 文字区上边界；`undefined` 表示右列放不下任何文字（调用方必须记降级）。 */
  textTop?: number;
}

/**
 * 右列：误差轴在上、账本残差轴在下、说明文字再往下。
 *
 * 两块轴之间以及文字上方都让出 46px：x 轴名装饰带（28px）在下、下一块轴标题带
 * （18px）在上，28+18=46 才是"标题不压上一块的轴名"的最小间距。
 * 文字区不足 3 行时返回 `textTop: undefined`，由调用方写降级而不是叠字。
 */
export function rightStack(L: ColumnsLayout, errorH = 72, residualH = 52): RightStack {
  const error: Rect = [L.rightX, L.topY, L.rightW, Math.max(60, errorH)];
  // 56px：上一块的 xlabel 带（底 28）与下一块的标题带（顶 18）之间再留 10px 呼吸。
  const residual: Rect = [
    L.rightX,
    error[1] + error[3] + 56,
    L.rightW,
    Math.max(50, residualH),
  ];
  const textTop = residual[1] + residual[3] + XL_LABEL_BAND + 6;
  return { error, residual, textTop: textTop <= L.floor - 3 * 15 ? textTop : undefined };
}

/**
 * 右列加宽：`columnsLayout` 的右列上限是 300px（够放两三行短注），
 * 但 4.3 的右列要同时排两块对数轴和 40+ 单位宽的说明文字。画布够宽时
 * 从左列让出宽度到 340px；窄画布保持共享版式（宁可少写字也不压图）。
 */
export function widenRightColumn(L: ColumnsLayout, width: number, target = 380): ColumnsLayout {
  if (width < 820 || !L.columns) return L;
  const contentWidth = width - L.left - 24;
  // 右列轴的 y 轴名在轴左 56px、刻度再占 35px；列间距必须 ≥100px，
  // 否则右列的刻度会压进左列图里（4.3 的旧基线就是这样）。
  const gap = 100;
  let rightW = Math.max(L.rightW, Math.min(target, contentWidth - gap - 300));
  let leftW = contentWidth - gap - rightW;
  if (leftW < 300) {
    rightW = Math.max(L.rightW, contentWidth - gap - 300);
    leftW = contentWidth - gap - rightW;
  }
  if (leftW < 280 || rightW <= L.rightW) return L;
  return { ...L, leftW, rightW, rightX: L.left + leftW + gap };
}

export interface Wave {
  x: number;
  kind: number;
}

/** 帧里的五条波位置去重（激波的头/尾是同一条线；t=0 时五条全叠在 x=0.5）。 */
export function dedupeWaves(waveX: ArrayLike<number>, waveKind: ArrayLike<number>): Wave[] {
  const out: Wave[] = [];
  const n = Math.min(waveX.length, waveKind.length);
  for (let i = 0; i < n; i += 1) {
    const x = waveX[i];
    if (!Number.isFinite(x)) continue;
    if (out.some((w) => Math.abs(w.x - x) <= 1e-9)) continue;
    out.push({ x, kind: Math.round(waveKind[i] ?? 0) });
  }
  return out;
}

export interface WaveStyle {
  color: string;
  dash: number[];
  label: string;
}

/** 波型配色：激波=红实线，接触=绿虚线，稀疏波边界=灰点线。 */
export function waveStyle(kind: number, pal: Palette): WaveStyle {
  if (kind === 2) return { color: pal.warn, dash: [], label: "激波" };
  if (kind === 0) return { color: pal.curves[1], dash: [6, 3], label: "接触间断" };
  return { color: pal.reference, dash: [2, 3], label: "稀疏波边界" };
}

/** 在一条轴上画波位置竖线（五条波按 kind 上色；同一位置只画一次）。 */
export function drawWaves(
  scene: Scene,
  axisId: string,
  waves: readonly Wave[],
  ylim: readonly [number, number],
  pal: Palette,
  width = 1.1,
): void {
  for (const w of waves) {
    const style = waveStyle(w.kind, pal);
    scene.layers.push(
      lineLayer(axisId, series([w.x, w.x], [ylim[0], ylim[1]], style.color, { width, dash: style.dash })),
    );
  }
}

/** 三个波型图例（固定顺序，不随算例变）。 */
export const WAVE_LEGEND = "红实=激波　绿虚=接触（p,u 连续、rho 跳）　灰点=稀疏波边界";
