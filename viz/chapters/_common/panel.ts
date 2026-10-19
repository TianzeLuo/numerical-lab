/**
 * 3.3 / 3.4 / 4.2–4.6 / 5.1–5.4 / 6.x 各 model 共用的**排版与显示几何**工具。
 *
 * 为什么是共享模块而不是每章复制一份：这一批章节是同一时期、同一套版式落地的，
 * 其中的每条纪律（文本按共享尺子折行、面板底线钉在 pitfalls 面板之上、跨层数据不连线、
 * 掩码占位值不进图层）都曾经以"复制时漏了一行"的方式出过事故。共享一份实现之后，
 * 规则只有一处可以改错，测试也能只钉一处。
 *
 * 这里**只有显示几何**：
 * - 按共享尺子（`textWidthUnits`：CJK=2、ASCII=1）折行；
 * - 把折好的行放进给定净空；
 * - 按数组自身的值切段、按掩码选点；
 * - 由数据算 1-2-5 取整的轴范围（`model.ts` 文件头第 2 条允许的"显示几何"）。
 *
 * 不读帧的物理含义、不重算后端已经给出的任何量（P8）。
 */

import type { AxisSpecJson } from "../../shared/src/manifest.js";
import { BADGE_UNIT_PX, ModelInputError, clamp, headerBand } from "../../shared/src/model.js";
import { pitfallsBand, text, type Scene, type TextSpec } from "../../shared/src/scene.js";

/** 面板内说明文字的字号（与 0.0 / 1.1 / 1.2 / 3.2 / 4.5 章一致）。 */
export const TEXT_SIZE = 11;
/** 行距：`textBoxOf` 按 `size + 4 = 15` 估盒，模型自己排行也必须用同一个数。 */
export const TEXT_ROW_H = TEXT_SIZE + 4;
/** 面板底边之上留出的净空（避开 `bottom+3..bottom+28` 的两条装饰带）。 */
export const SAFE_GAP = 18;
/** 面板下沿到 pitfalls 面板顶边之间必须让出的像素（轴名装饰带 28px）。 */
export const XL_LABEL_BAND = 28;
/** 两列版式能画的最小画布宽度（再窄就挤不出两块宽高为正的图，明确报错）。 */
export const MIN_CANVAS_W = 420;

/** 面板宽度 → 显示宽度单位预算（1 单位 = 7px，与 `textWidthUnits` 同一把尺子）。 */
export function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/** 按显示宽度单位折行（CJK=2、ASCII=1；口径同 `model.ts::textWidthUnits`）。 */
export function wrapUnits(body: string, budget: number): string[] {
  const out: string[] = [];
  let line = "";
  let used = 0;
  const push = (): void => {
    if (line !== "") out.push(line);
    line = "";
    used = 0;
  };
  for (const word of body.split(/(?<=[\s　])/)) {
    const w = textWidthUnitsOf(word);
    if (used > 0 && used + w > budget) push();
    if (w <= budget) {
      line += word;
      used += w;
      continue;
    }
    let piece = "";
    let pieceUsed = 0;
    for (const ch of word) {
      const cw = textWidthUnitsOf(ch);
      if (pieceUsed + cw > budget && piece !== "") {
        out.push(piece);
        piece = "";
        pieceUsed = 0;
      }
      piece += ch;
      pieceUsed += cw;
    }
    line = piece;
    used = pieceUsed;
  }
  push();
  return out;
}

/** 本地等价实现，避免为了一个宽度函数把整个 `scene.ts` 的运行时依赖拖进来（其实无副作用，纯粹省 import）。 */
function textWidthUnitsOf(body: string): number {
  let n = 0;
  for (const ch of body) n += (ch.codePointAt(0) ?? 0) > 0x2e80 ? 2 : 1;
  return n;
}

/**
 * 一段说明文字 → 若干 `text` 层，**一行一个**（`render2d.ts::drawText` 不拆 `\n`）。
 *
 * 放不下时**截断并说明**（最后一行留给"另有 N 行未画"），绝不叠着画：
 * 叠字等于两块都没说清，而规则 8 会当场拒绝。
 */
export function textBlock(
  x: number,
  top: number,
  bottom: number,
  lines: readonly string[],
  color: string,
  budget: number,
  anchor: "top" | "bottom",
): TextSpec[] {
  let pieces = lines.flatMap((body) => wrapUnits(body, budget));
  if (bottom > top) {
    const maxRows = Math.max(1, Math.floor((bottom - top) / TEXT_ROW_H));
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行未画）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - (pieces.length - 1) * TEXT_ROW_H);
  return pieces.map((body, i) =>
    text(x, y + i * TEXT_ROW_H, body, { role: "note", size: TEXT_SIZE, color, maxWidth: budget * BADGE_UNIT_PX }),
  );
}

/**
 * 对数轴范围：把 `[lo, hi]` 扩到"整十次幂"的上下界。
 *
 * 不用共享的 `niceLimits(lo, hi, target, {log:true})`：它在 log 模式下按 tick 目标"漂亮化"，
 * 跨度一大就一次跳十几个数量级（3.2 章实测 `[1.1e-16, 2e-6]` 被扩成 `[1e-21, 0.1]`，
 * 曲线当场被压成一条直线）。这里只要"包住数据 + 最多各让一格"。
 */
export function logRange(lo: number, hi: number): [number, number] {
  const a = Number.isFinite(lo) && lo > 0 ? lo : 1;
  const b = Number.isFinite(hi) && hi > 0 ? hi : 1;
  const loExact = Math.log10(Math.min(a, b));
  const hiExact = Math.log10(Math.max(a, b));
  let l = Math.floor(loExact);
  let h = Math.ceil(hiExact);
  if (loExact - l < 0.05) l -= 1;
  if (h - hiExact < 0.05) h += 1;
  if (h - l < 1) h = l + 1;
  return [10 ** l, 10 ** h];
}

/** 线性轴范围：包住 `[lo, hi]` 并留出 `pad` 比例的边距；退化区间给 ±1 的窗。 */
export function linearRange(lo: number, hi: number, pad = 0.08): [number, number] {
  let a = Number.isFinite(lo) ? lo : 0;
  let b = Number.isFinite(hi) ? hi : 1;
  if (a > b) [a, b] = [b, a];
  if (!(b > a)) {
    const c = Math.abs(a) > 0 ? Math.abs(a) * 0.1 : 1;
    return [a - c, a + c];
  }
  const d = (b - a) * pad;
  return [a - d, b + d];
}

/** manifest 里该轴的标题（文案直接取 manifest，缺省才用回落值——不许模型自己发明轴名）。 */
export function axisTitleOf(axes: readonly AxisSpecJson[] | undefined, id: string, fallback: string): string {
  return axes?.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel（同上）。 */
export function axisYlabelOf(axes: readonly AxisSpecJson[] | undefined, id: string, fallback: string): string {
  return axes?.find((a) => a.id === id)?.ylabel ?? fallback;
}

/** 一段下标区间 `[start, end)`。 */
export interface Run {
  start: number;
  end: number;
}

/**
 * 把"顺序拼接/按层分组"的数组按段落键切成若干段（键值相同且**相邻** → 同一段）。
 *
 * 跨层数据不许连成一条线：把第 0 层的事件与第 1 层的事件连起来，画出的是一段
 * **不存在的搬运**（4.5 章的第 3 个陷阱）。切段之后每一层只有自己的锯齿。
 */
export function runsBy(key: ArrayLike<number>, n = key.length): Run[] {
  const runs: Run[] = [];
  let start = -1;
  let current = Number.NaN;
  for (let i = 0; i < n; i += 1) {
    const v = key[i];
    const usable = Number.isFinite(v);
    if (start >= 0 && (!usable || v !== current)) {
      runs.push({ start, end: i });
      start = -1;
    }
    if (usable && start < 0) {
      start = i;
      current = v;
    }
  }
  if (start >= 0) runs.push({ start, end: n });
  return runs;
}

/** 从 `(x, y)` 里切出 `[run.start, run.end)` 的一段（复制成新数组，避免共享缓冲被下游改）。 */
export function sliceRun(x: ArrayLike<number>, y: ArrayLike<number>, run: Run): { x: Float64Array; y: Float64Array } {
  const n = Math.max(0, run.end - run.start);
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    xs[i] = x[run.start + i];
    ys[i] = y[run.start + i];
  }
  return { x: xs, y: ys };
}

/** 按掩码选出的下标复制成新数组（掩码过滤的曲线一律走它，避免共享缓冲被下游改）。 */
export function selectValues(a: ArrayLike<number>, idx: readonly number[]): Float64Array {
  const out = new Float64Array(idx.length);
  idx.forEach((i, k) => {
    out[k] = a[i] as number;
  });
  return out;
}

/** 把下标数组映射成 `(x, y)` 两列，供 `scatter` 图层使用。 */
export function pick(x: ArrayLike<number>, y: ArrayLike<number>, idx: readonly number[]): { x: Float64Array; y: Float64Array } {
  const xs = new Float64Array(idx.length);
  const ys = new Float64Array(idx.length);
  idx.forEach((i, k) => {
    xs[k] = x[i];
    ys[k] = y[i];
  });
  return { x: xs, y: ys };
}

/** 掩码（`> 0.5` 记命中）的个数——"已到地板 / 已触顶"的**计数**，规则 1 要的是计数不是估计。 */
export function countMask(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] ?? 0) > 0.5) n += 1;
  return n;
}

/** 命中掩码的下标（`> 0.5`），按原顺序。 */
export function maskIndices(a: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < a.length; i += 1) if ((a[i] ?? 0) > 0.5) out.push(i);
  return out;
}

/** 掩码（`> 0.5`）为真的下标里，值也有限且满足 `accept` 的下标。 */
export function maskIndicesWhere(mask: ArrayLike<number>, value: ArrayLike<number>, accept: (v: number) => boolean): number[] {
  const out: number[] = [];
  const n = Math.min(mask.length, value.length);
  for (let i = 0; i < n; i += 1) {
    const v = value[i];
    if ((mask[i] ?? 0) > 0.5 && Number.isFinite(v) && accept(v)) out.push(i);
  }
  return out;
}

/** 有限值的 `[min, max]`；没有有限值返回 undefined。 */
export function finiteMinMax(a: ArrayLike<number> | undefined): [number, number] | undefined {
  if (a === undefined) return undefined;
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) ? [lo, hi] : undefined;
}

/** 有限正值的 `[min, max]`（对数轴的定义域）；没有这样的值返回 undefined。 */
export function positiveMinMax(a: ArrayLike<number> | undefined): [number, number] | undefined {
  if (a === undefined) return undefined;
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v) || v <= 0) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) ? [lo, hi] : undefined;
}

/** 数组里第一个有限值（参考线数组通常是常量列，取一个就够）。 */
export function firstFinite(a: ArrayLike<number> | undefined): number | undefined {
  if (a === undefined) return undefined;
  for (let i = 0; i < a.length; i += 1) if (Number.isFinite(a[i])) return a[i];
  return undefined;
}

/** 第 `row` 行（行主序二维数组）→ `Float64Array`；越界返回 undefined（调用方降级，不猜）。 */
export function rowOfMatrix(
  values: ArrayLike<number>,
  rows: number,
  cols: number,
  row: number,
  width = cols,
): Float64Array | undefined {
  if (!(rows > 0) || !(cols > 0) || row < 0 || row >= rows) return undefined;
  const n = Math.min(cols, width);
  if (values.length < rows * cols) return undefined;
  const out = new Float64Array(n);
  for (let c = 0; c < n; c += 1) out[c] = values[row * cols + c];
  return out;
}

/** 0..n−1 的下标数组（x 轴是"样本下标"时用；不做物理换算）。 */
export function indexAxis(n: number): Float64Array {
  const out = new Float64Array(Math.max(0, n));
  for (let i = 0; i < out.length; i += 1) out[i] = i;
  return out;
}

/** 一段长度为 2 的水平/垂直参考线；`vertical=true` 时把常量放在 x 上。 */
export function refLine(a: number, b: number, value: number, vertical = false): { x: [number, number]; y: [number, number] } {
  return vertical ? { x: [value, value], y: [a, b] } : { x: [a, b], y: [value, value] };
}

// --------------------------------------------------------------------------- //
// 两列版式
// --------------------------------------------------------------------------- //

export interface ColumnsLayout {
  /** 左列（主面板 + 可选的下面板）的 x 与宽。 */
  left: number;
  leftW: number;
  /** 上面板。 */
  topY: number;
  topH: number;
  /** 下面板。 */
  botY: number;
  botH: number;
  /** 右列（副轴 + 文字块）的 x 与宽；单列时等于左列。 */
  rightX: number;
  rightW: number;
  /** 右列副轴的高度（单列时不用）。 */
  auxH: number;
  /** 右列文字区的上边界（副轴 x 轴名装饰带之下）。 */
  textTop: number;
  /** 面板/文字允许的最低底边（= pitfalls 面板顶 − 28）。 */
  floor: number;
  /** 是否真的排成了两列（测试据此断言"窄画布退化"路径）。 */
  columns: boolean;
}

export interface ColumnsOptions {
  topShare?: number;
  auxH?: number;
  gap?: number;
  /** 右列最小宽度（放说明文字）；低于它时退化成单列。缺省 200。 */
  minRight?: number;
  /** 上下面板共用一条 x 轴时把 gap 设小（缺省 46）。 */
  contentLeft?: number;
}

/**
 * 两列版式：左列上下两块面板，右列一块副轴 + 文字。
 *
 * 底线一律取 `pitfallsBand(scene)` 的顶边 − 28（pitfalls 面板由渲染层绘制，
 * 模型看不见它，只能从 scene 的 pitfalls 文本按同一把尺子算出来）。
 * 与帧无关：同一 params 下连续帧的面板位置逐像素相同，动画不跳。
 */
export function columnsLayout(
  scene: Scene,
  width: number,
  height: number,
  opts: ColumnsOptions = {},
): ColumnsLayout {
  if (width < MIN_CANVAS_W) {
    throw new ModelInputError(
      `画布宽 ${width}px：本章的两列布局至少要 ${MIN_CANVAS_W}px（左列图 + 副轴装饰带 + 说明列）`,
    );
  }
  const contentLeft = opts.contentLeft ?? 64;
  const contentWidth = Math.max(160, width - contentLeft - 24);
  const topY = headerBand(width) + 10;
  const panelTop = pitfallsBand(scene)?.[1] ?? height;
  const floor = panelTop - XL_LABEL_BAND;
  const rightW = clamp(contentWidth - 56 - (opts.minRight ?? 200), 132, 300);
  const leftW = Math.max(80, contentWidth - 56 - rightW);
  const rightX = contentLeft + leftW + 56;
  const gap = opts.gap ?? 46;
  const avail = Math.max(150, floor - topY - XL_LABEL_BAND);
  const topShare = opts.topShare ?? 0.6;
  const topH = Math.max(80, Math.round((avail - gap) * topShare));
  const botH = Math.max(60, avail - gap - topH);
  const botY = topY + topH + gap;
  const auxH = Math.max(64, Math.min(opts.auxH ?? 130, topH));
  return {
    left: contentLeft,
    leftW,
    topY,
    topH,
    botY,
    botH,
    rightX,
    rightW,
    auxH,
    textTop: topY + auxH + XL_LABEL_BAND + 8,
    floor,
    columns: rightW >= 200,
  };
}

/**
 * 面板内侧的图例块：在**面板宽度预算**内折行（最多 3 行）。
 *
 * 长图例写一行会横穿到右列副轴的装饰带里（`axisDecoBands` 的 ylabel/刻度带），
 * 两个校验器都会当场报"文本压住坐标轴装饰（规则 8）"。
 */
export function legendBlock(x: number, y: number, panelW: number, body: string, color: string): TextSpec[] {
  const budget = Math.max(8, Math.floor((panelW - 12) / BADGE_UNIT_PX));
  const fitted = textWidthUnitsOf(body) <= budget ? body : `${clipUnits(body, Math.max(4, budget - 1))}…`;
  return textBlock(x, y, y + 3 * TEXT_ROW_H, [fitted], color, budget, "top");
}

/** 按显示宽度单位截断（CJK=2、ASCII=1），**不加省略号**（调用方自己加）。 */
export function clipUnits(body: string, budget: number): string {
  let used = 0;
  let out = "";
  for (const ch of body) {
    const w = textWidthUnitsOf(ch);
    if (used + w > budget) break;
    out += ch;
    used += w;
  }
  return out;
}

/** "第 N 行未画"这类固定短注（模型各处一致，测试按它断言"降级不静默"）。 */
export function clippedNote(dropped: number): string {
  return `…（另有 ${dropped} 行未画）`;
}
