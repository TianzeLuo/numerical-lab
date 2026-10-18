/**
 * 4.5 章四个 model 共用的**排版 / 显示几何**工具（纯几何与纯文本，不含任何物理量）。
 *
 * 为什么要单独一个文件：`textBlock` / `wrapUnits` / `unitBudget` 是"模型自己折行"的同一套实现
 * （`render2d.ts::drawText` 不拆 `\n`，光栅器的续行行距是 `size + 4` 而模型按 15px 排行，
 * 把折行交给渲染层就会让续行压到下一行文字上）。1.1 / 3.2 章的 `_panel.ts` 已经把这条教训
 * 写进注释，本章照它的口径走，不再复制四份。
 *
 * 本章多出来的**唯一**一件几何事：{@link panel45Layout} 把面板底线钉在
 * **底部 pitfalls 面板之上**（`pitfallsBand(scene)` 的顶边 − 28px）。
 *
 * 为什么必须自己算：`stackedPanels` 的缺省 footer 是 `8 + 60 + 34 = 102`，它假设 pitfalls
 * 面板只有 **3 行**（`3×16 + 8 = 56 ≤ 60`）。本章四段 pitfalls 都是 270~360 字，960×600 下
 * 折成 5~6 行（面板高 88~104px），缺省 footer 会让最下面那条轴的轴名（`bottom+16..bottom+28`）
 * 正好落在面板底下——渲染层后画的不透明面板把它盖掉，图上看不见 x 轴名，而**两个校验器都不查**
 * "装饰带 vs pitfalls 面板"（它们只查文本层），所以这件事只能由模型自己守。
 * 见 `docs/AUTHORING.md` §3.2 与 `scene.ts::pitfallsBand`。
 *
 * 这些函数只做三件事：按共享尺子（`textWidthUnits`：CJK=2、ASCII=1）折行、把折好的行
 * 放进给定的净空里、按数组自身的值切段。**不碰帧、不碰轴范围、不算物理量。**
 */

import type { AxisSpecJson } from "../../shared/src/manifest.js";
import { BADGE_UNIT_PX, ModelInputError, clamp, headerBand, textWidthUnits, type Rect } from "../../shared/src/model.js";
import { pitfallsBand, text, type Scene, type TextSpec } from "../../shared/src/scene.js";

/** 面板内说明文字的字号（与 0.0 / 1.1 / 1.2 / 3.2 章一致）。 */
/** 本布局能画的最小画布宽度（再窄就只能报错，不画读不出来的图）。 */
export const MIN_CANVAS_W = 420;

export const TEXT_SIZE = 11;
/** 行距：`textBoxOf` 按 `size + 4 = 15` 估盒，模型自己排行也必须用同一个数。 */
export const TEXT_ROW_H = TEXT_SIZE + 4;
/** 面板底边之上留出的净空（避开 `bottom+3..bottom+28` 的两条装饰带）。 */
export const SAFE_GAP = 18;
/** 面板下沿到 pitfalls 面板顶边之间必须让出的像素（轴名装饰带 28px）。 */
export const XL_LABEL_BAND = 28;

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
    const w = textWidthUnits(word);
    if (used > 0 && used + w > budget) push();
    if (w <= budget) {
      line += word;
      used += w;
      continue;
    }
    let piece = "";
    let pieceUsed = 0;
    for (const ch of word) {
      const cw = textWidthUnits(ch);
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
      // 截断标记要短到不会折行：它一旦折成两行，盒子高度翻倍就会越过面板底线。
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行未画）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - (pieces.length - 1) * TEXT_ROW_H);
  return pieces.map((body, i) => text(x, y + i * TEXT_ROW_H, body, { role: "note", size: TEXT_SIZE, color, maxWidth: budget * 7 }));
}

/**
 * 对数轴范围：把 `[lo, hi]` 扩到"整十次幂"的上下界。
 *
 * 不用共享的 `niceLimits(lo, hi, target, {log:true})`：它在 log 模式下按 tick 目标"漂亮化"，
 * 跨度一大就一次跳十几个数量级（3.2 章实测 `[1.1e-16, 2e-6]` 被扩成 `[1e-21, 0.1]`，
 * 曲线当场被压成一条直线）。这里只要"包住数据 + 最多各让一格"。
 * 传进来的必须是 **> 0 的有限值**（调用方负责过滤 0/负/非有限）。
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
 * `vcycle_hierarchy` 的 `event_level` 就是键：真实事件序列是 0,0,1,1,1,2,2,… 这样
 * 按层成块出现的（`mg_apply(trace=...)` 的递归顺序）。**必须切段的原因**：一条折线把
 * 第 0 层的事件（0.93）与第 1 层的事件（30.5）连起来，画出的是一段**不存在的搬运**——
 * 而那正好是本章第 3 个陷阱（跨层比范数）想让人相信的东西。切段之后每一层只有自己的锯齿。
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

/** 掩码（`> 0.5`）为真的下标里，值也有限的下标。 */
export function maskIndicesWhere(mask: ArrayLike<number>, value: ArrayLike<number>, accept: (v: number) => boolean): number[] {
  const out: number[] = [];
  const n = Math.min(mask.length, value.length);
  for (let i = 0; i < n; i += 1) {
    const v = value[i];
    if ((mask[i] ?? 0) > 0.5 && Number.isFinite(v) && accept(v)) out.push(i);
  }
  return out;
}

// --------------------------------------------------------------------------- //
// 本章版式：两列（左：上下两块面板 / 右：副轴 + 文字）或单列退化
// --------------------------------------------------------------------------- //

export interface Panel45Layout {
  /** 左列两个面板的 x 与宽。 */
  left: number;
  leftW: number;
  /** 上面板（主图 / 因子面板）。 */
  topY: number;
  topH: number;
  /** 下面板（副图 / 迭代数面板）。 */
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

/**
 * 本章的版式（与帧无关：同一 params 下连续帧的面板位置逐像素相同，动画不跳）。
 *
 * * **两列**（`width ≥ 712`）：左列上下两块面板共用同一条 x 轴（上面板清空 `xticks`/`xlabel`，
 *   免得与下面板的轴标题挤在同一条带里）；右列放副轴与文字块。
 * * **单列**（窄画布）：两块面板占满宽度依次排下，文字块放弃（角标仍然在）。
 *
 * 底线一律取 `pitfallsBand(scene)` 的顶边 − 28（见文件头）。
 */
export function panel45Layout(
  scene: Scene,
  width: number,
  height: number,
  opts: { topShare?: number; auxH?: number; gap?: number } = {},
): Panel45Layout {
  // 画布窄到 420 以下时，左列 + 副轴装饰带 + 右列已经挤不出两块**宽高为正**的图：
  // 与其画一张读不出来的图，不如明确报错（模型里的 degraded 分支照常保留）。
  if (width < MIN_CANVAS_W) {
    throw new ModelInputError(`画布宽 ${width}px：4.5 章的布局至少要 ${MIN_CANVAS_W}px（左列图 + 副轴装饰带 + 说明列）`);
  }
  const contentLeft = 64;
  const contentWidth = Math.max(160, width - contentLeft - 24);
  const topY = headerBand(width) + 10;
  const panelTop = pitfallsBand(scene)?.[1] ?? height;
  const floor = panelTop - XL_LABEL_BAND;
  // 副轴自己的 y 轴装饰带占 56px（`axisDecoBands` 的 [x−56, x−4]）：左右两列之间必须留出来，
  // 否则左列面板里的图例/文字会横穿到右轴的刻度带里（校验器会当场报规则 8）。
  const rightW = clamp(contentWidth - 56 - 200, 132, 300);
  const leftW = Math.max(80, contentWidth - 56 - rightW);
  const rightX = contentLeft + leftW + 56;
  // 两块面板的间距：上面板自己的 x 轴刻度带（3..15）与轴名带（16..28），加上下面板的轴标题带
  // （−18..−2），所以**两块面板各有各的 x 轴时**至少要 46px（留 58 更稳）。
  // 上面板清空了 `xticks`/`xlabel`（两块共用一条 x 轴）的模型可以传 30。
  const gap = opts.gap ?? 46;
  // 下面板自己的 x 轴名装饰带（28px）也要算进去。
  const avail = Math.max(150, floor - topY - XL_LABEL_BAND);
  const topShare = opts.topShare ?? 0.6;
  const topH = Math.max(80, Math.round((avail - gap) * topShare));
  const botH = Math.max(60, avail - gap - topH);
  const botY = topY + topH + gap;
  const auxH = Math.max(64, Math.min(opts.auxH ?? 150, topH));
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
    // 右列窄于 200px 时排不下说明文字：模型改走 degraded（宁可不说，也不叠着说）。
    columns: rightW >= 200,
  };
}

/**
 * 面板内侧的图例块：在**面板宽度预算**内自动折行（最多 3 行）。
 *
 * 为什么不能直接 `text(...)` 写一行长图例：模型给不出 `measureText`，一行的盒子宽度按
 * "显示宽度单位 × 7px" 估。图例写长了就会横穿到**右列副轴**的装饰带里
 * （`axisDecoBands` 的 `[x−56, x−4]` ylabel 带与 `[x−41, x−6]` 刻度带）——两个校验器都会
 * 当场报"文本压住坐标轴装饰（规则 8）"。这里按 `panelW` 折行，从构造上保证出不了面板。
 */
export function legendBlock(x: number, y: number, panelW: number, body: string, color: string): TextSpec[] {
  // 这里不能用 `unitBudget`：它有 24 个单位的**下限**（那是给右列文字块的），
  // 窄画布下会把图例放到比面板还宽。按面板宽度直接算。
  const budget = Math.max(8, Math.floor((panelW - 12) / BADGE_UNIT_PX));
  // 预算不够时**硬截断**：`textBoxOf` 对没有空格的整段文字不会折行，
  // 长图例的盒子会直接横穿到右列副轴的装饰带里（窄画布下必然发生）。
  const fitted = textWidthUnits(body) <= budget ? body : `${clipUnits(body, Math.max(4, budget - 1))}…`;
  return textBlock(x, y, y + 3 * TEXT_ROW_H, [fitted], color, budget, "top");
}

/** 按显示宽度单位截断（CJK=2、ASCII=1），**不加省略号**（调用方自己加）。 */
export function clipUnits(body: string, budget: number): string {
  let used = 0;
  let out = "";
  for (const ch of body) {
    const w = textWidthUnits(ch);
    if (used + w > budget) break;
    out += ch;
    used += w;
  }
  return out;
}

/** 面板矩形（`rect` 的第四个数是**高度**，与 `textBoxOf` 同口径）。 */
export function rectOf(x: number, y: number, w: number, h: number): Rect {
  return [x, y, w, h];
}
