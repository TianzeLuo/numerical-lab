/**
 * 3.1 章四个 model 共用的**排版 / 显示几何**工具（纯几何与纯文本，不含任何物理量）。
 *
 * 为什么要单独一个文件：`textBlock` / `wrapUnits` / `unitBudget` 是"模型自己折行"的同一套实现
 * （`render2d.ts::drawText` 不拆 `\n`，光栅器的续行行距是 `size + 4`，而模型按 15px 排行——
 * 把折行交给渲染层就会让续行压到下一行文字上）。1.2 / 3.2 章各有一份同口径的 `_panel.ts`；
 * 本章照抄这套口径而**不去 import 别的章节的私有模块**：章节前端要保持可以单独编译、单独删除。
 *
 * 与 3.2 章那份的差别只有两处，都是本章版式的需要：
 *
 * 1. {@link stackPanels}：本章三个 viz 是"一块主图 + 一到两个副图"的**竖直堆叠**，每块的高度
 *    按权重分配、间距固定 46px。46 这个数不是随手取的：`AXIS_DECO` 给上方面板下沿的装饰带是
 *    `[bottom+3, bottom+15]`（刻度标签）与 `[bottom+16, bottom+28]`（轴名），而下方面板的**轴标题**
 *    占 `[top−18, top−2]`——46 = 28（下装饰带的底）+ 18（标题带的顶）+ 0 余量，两块刚好不撞。
 *    间距固定（与帧无关）是硬要求：动画里面板不许上下跳。
 * 2. {@link clipMark}：把帧给的 `mark_*` 标量/数组夹成"对数轴上画得出来"的点。本章的 `mark_*`
 *    里既有正数也有 `-1` 哨兵（`work_at_target`），**不许把哨兵当数值画**。
 *
 * 这些函数只做三件事：按共享尺子（`textWidthUnits`：CJK=2、ASCII=1）折行、把折好的行放进
 * `[top, bottom]` 的净空里、按数组自身的值做显示裁剪。**不碰帧、不碰轴范围、不算物理量。**
 *
 * `SAFE_GAP` 的口径（规则 8）：一段文字的盒子高度按 `行数 × (size+4)` 估（`scene.ts::textBoxOf`），
 * 所以"贴边放"必须让最后一行的盒子底 ≤ `bottom−3`：本模块留 **18px**。
 */

import type { AxisSpecJson } from "../../shared/src/manifest.js";
import {
  applyOverflow,
  placeBadge,
  textWidthUnits,
  type Rect,
} from "../../shared/src/model.js";
import { overflowBadge, text, type Scene, type TextSpec } from "../../shared/src/scene.js";

/** 面板内说明文字的字号（与 0.0 / 1.1 / 1.2 / 3.2 章一致）。 */
export const TEXT_SIZE = 11;
/** 行距：`textBoxOf` 按 `size + 4 = 15` 估盒，模型自己排行也必须用同一个数。 */
export const TEXT_ROW_H = TEXT_SIZE + 4;
/** 面板底边之上留出的净空（见文件头：避开 `bottom+3..bottom+28` 的两条装饰带）。 */
export const SAFE_GAP = 18;
/** 两块竖直堆叠面板之间的固定间距（= 下方面板标题带的顶，见文件头）。 */
export const PANEL_GAP = 46;

/**
 * 面板宽度 → 显示宽度单位预算（1 单位 = 7px，与 `textWidthUnits` 同一把尺子）。
 *
 * 留 12px 边距再折算：折出来的行一定装得进面板，两个渲染器都不会再自动折行。
 */
export function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/**
 * 对数轴范围：把 `[lo, hi]` 扩到"整十次幂"的上下界（最多各让一格）。
 *
 * 为什么不用共享的 `niceLimits(..., {log: true})`：它的目标是"漂亮刻度"，跨度一大就一次跳十几个
 * 数量级（3.2 章实测 `[1.1e-16, 2e-6]` 被扩成 `[1e-21, 0.1]`）。本章的误差轴要**固定窗口**、
 * 残差轴要看见"地板"与"10^-1"之间那 16 个数量级，范围必须紧贴数据。
 *
 * 传进来的必须是 **> 0 的有限值**（调用方负责过滤 0/负/非有限：对数轴上它们不存在）。
 */
export function logRange(lo: number, hi: number): [number, number] {
  const a = Number.isFinite(lo) && lo > 0 ? lo : 1;
  const b = Number.isFinite(hi) && hi > 0 ? hi : 1;
  const loExact = Math.log10(Math.min(a, b));
  const hiExact = Math.log10(Math.max(a, b));
  let l = Math.floor(loExact);
  let h = Math.ceil(hiExact);
  // 数据贴着上下沿时各让一格，免得曲线正好压在轴线上（视觉上会读成"被截断"）。
  if (loExact - l < 0.05) l -= 1;
  if (h - hiExact < 0.05) h += 1;
  if (h - l < 1) h = l + 1;
  return [10 ** l, 10 ** h];
}

/**
 * 按显示宽度单位折行（CJK=2、ASCII=1；口径同 `model.ts::textWidthUnits`）。
 *
 * 先按空白（含全角空格）切"词"再往行里塞：`h=0.0625`、`4.93e-08` 这类字段不会被从中间劈开
 * （劈开在光栅器里读起来像两个数）。单个词比预算还长时才硬切。
 */
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
 * `anchor`：`"top"` 从 `top` 往下排；`"bottom"` 让整块贴着 `bottom`（盒子底 = 最后一行 y + 15）。
 * 返回数组的长度 = 实际排出的行数（调用方据此把下一块往下推）。
 *
 * 画布小的时候宁可**截断并说明**，也不让文字叠在一起：能放下的行数由 `[top, bottom]` 的净空决定，
 * 最后一行留给"还有 N 行"（截断标记本身必须短到不折行，否则盒子高度翻倍会越过面板底线）。
 */
export function textBlock(
  x: number,
  top: number,
  bottom: number,
  lines: readonly string[],
  color: string,
  budget: number,
  anchor: "top" | "bottom" = "top",
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
    text(x, y + i * TEXT_ROW_H, body, { role: "note", size: TEXT_SIZE, color, maxWidth: budget * 7 }),
  );
}

/** 面板内文字块的上边界（面板矩形 → 第一行 y）。 */
export function textTop(rect: readonly [number, number, number, number]): number {
  return rect[1] + 4;
}

/**
 * 面板内文字块的下边界（面板矩形 → 最后一行允许的 y）。
 *
 * `bottom` 取**面板矩形底 − SAFE_GAP**：矩形底通常就是所叠轴的下沿，再往下是刻度带与轴名带。
 * `withXAxis=false` 表示这块矩形下面**没有**轴（本章的说明块、注释块），此时可以贴到更低。
 */
export function textBottom(rect: readonly [number, number, number, number], withXAxis = true): number {
  return rect[1] + rect[3] - (withXAxis ? SAFE_GAP : 4);
}

/** manifest 里该轴的标题（文案直接取 manifest，缺省才用回落值——不许模型自己发明轴名）。 */
export function axisTitleOf(axes: readonly AxisSpecJson[] | undefined, id: string, fallback: string): string {
  return axes?.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel（同上）。 */
export function axisYlabelOf(axes: readonly AxisSpecJson[] | undefined, id: string, fallback: string): string {
  return axes?.find((a) => a.id === id)?.ylabel ?? fallback;
}

/** 一块竖直堆叠面板的矩形。 */
export interface PanelStack {
  /** 各块的矩形，顺序与 `shares` 一致。 */
  rects: Rect[];
  /** 内容区左边界 / 宽度 / 顶 / 底（透传 `stackedPanels` 的口径）。 */
  left: number;
  contentWidth: number;
  top: number;
  bottom: number;
}

/**
 * 把内容区按 `shares` 权重竖直切成若干块（间距固定 `gap`）。
 *
 * * 权重只决定**剩余空间**的分配；每块先拿到 `minHeights[i]`（缺省 56），
 *   这样窄画布上副图也不会被压成一条缝；
 * * 结果与帧无关：同一画布尺寸下连续帧的矩形逐像素相同（动画不跳）。
 */
export function stackPanels(
  base: { left: number; contentWidth: number; top: number; bottom: number },
  shares: readonly number[],
  opts: { gap?: number; minHeights?: readonly number[] } = {},
): PanelStack {
  const gap = opts.gap ?? PANEL_GAP;
  const n = Math.max(1, shares.length);
  const minHeights = opts.minHeights ?? [];
  const usable = Math.max(120, base.bottom - base.top - gap * (n - 1));
  const total = shares.reduce((s, v) => s + v, 0) || 1;
  // 第一轮：按权重分（不低于各自的 minHeight）。
  const heights = shares.map((s, i) => Math.max(minHeights[i] ?? 56, Math.round((usable * s) / total)));
  // 第二轮：总和超出 usable 时**从最大的一块开始**削（副图不能被主图挤没了，
  // 但主图被压成一条缝同样读不出东西——削最大值是最不坏的取舍）。
  let over = heights.reduce((s, v) => s + v, 0) - usable;
  let guard = 0;
  while (over > 0 && guard < 64) {
    let k = 0;
    for (let i = 1; i < heights.length; i += 1) if (heights[i] > heights[k]) k = i;
    if (heights[k] <= (minHeights[k] ?? 56)) break;
    heights[k] -= 1;
    over -= 1;
    guard += 1;
  }
  const rects: Rect[] = [];
  let cursor = base.top;
  for (let i = 0; i < n; i += 1) {
    rects.push([base.left, cursor, base.contentWidth, heights[i]]);
    cursor += heights[i] + gap;
  }
  return { rects, left: base.left, contentWidth: base.contentWidth, top: base.top, bottom: base.bottom };
}

/**
 * 一块文字的**估算盒**（校验器口径：按 `maxWidth` 折行、行高 `size+4`），供模型自查重叠。
 *
 * 模型拿不到 `measureText`，但 `validateScene` 用的是这把尺子；这里让模型在**排之前**就能
 * 判断"这两个标签会不会撞"，而不是等校验器报违规。
 */
export function textBox(
  t: { x: number; y: number; text: string; size?: number; align?: string; maxWidth?: number },
): { x: number; y: number; w: number; h: number } {
  const size = t.size ?? TEXT_SIZE;
  const body = t.text ?? "";
  const perLine = t.maxWidth && t.maxWidth > 0 ? Math.max(1, Math.floor(t.maxWidth / 7)) : Number.POSITIVE_INFINITY;
  const rows: string[] = [];
  if (Number.isFinite(perLine)) {
    let line = "";
    for (const ch of body) {
      const probe = line + ch;
      if (textWidthUnits(probe) > perLine && line !== "") {
        rows.push(line);
        line = ch;
      } else {
        line = probe;
      }
    }
    if (line !== "") rows.push(line);
  } else {
    rows.push(body);
  }
  const w = Math.max(...rows.map((l) => textWidthUnits(l)), 0) * 7;
  const x = t.align === "right" ? t.x - w : t.align === "center" ? t.x - w / 2 : t.x;
  return { x, y: t.y, w, h: rows.length * (size + 4) };
}

/** 两个估算盒是否"两个方向都超过 2px"地重叠（与两侧校验器同一判据）。 */
export function boxesOverlap(
  a: { x: number; y: number; w: number; h: number },
  b: { x: number; y: number; w: number; h: number },
): boolean {
  const ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  return ox > 2 && oy > 2;
}

/**
 * 把溢出写进 Scene：先跑共享的 `applyOverflow`（图层越界 + 非有限值），
 * 再把**模型这一侧发现的**截断/占位（地板值、`-1` 哨兵、被跳过的点）并进同一条 note。
 *
 * 为什么要合并：本章最贵的两个坑（误差轴地板 1e-17、反查的 `-1` 掩码）都发生在**画之前**——
 * 不写进同一条 note，它们就成了"图上没有、账上也没有"的东西，正是规则 1 要禁止的。
 */
export function applyOverflowNote(
  scene: Scene,
  extra: string,
  rawNonFinite: number,
  opts: { slot: number; width: number },
): void {
  applyOverflow(scene, rawNonFinite, opts);
  const trimmed = extra.trim();
  if (trimmed === "") return;
  // 地板/掩码/固定窗口裁剪是显示约定，不是“数据坏了”；只有真正的非有限值
  // 或没有有限解才升到红色。网页对 warning 显示为“显示范围”，避免默认画面吓人。
  const dataLike = rawNonFinite > 0 || /非有限|NaN|∞|无有限解|未定义/.test(trimmed);
  const tone = dataLike ? "danger" : "warning";
  scene.overflow = {
    flag: true,
    note: scene.overflow.note === "" ? trimmed : `${scene.overflow.note}；${trimmed}`,
    severity: dataLike ? "data" : "clip",
  };
  const existing = scene.badges.find((b) => b.kind === "overflow");
  if (existing !== undefined) {
    existing.tone = tone;
    existing.color = tone === "danger" ? "#c53030" : "#b7791f";
    return;
  }
  scene.badges.push(placeBadge(overflowBadge(trimmed, 8, 8, tone), opts.slot, opts.width));
}

/** 有限值的 `[min, max]`（只扫正有限值）；一个都没有时 undefined。 */
export function positiveMinMax(a: ArrayLike<number>): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v) || v <= 0) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : undefined;
}

/** 有限值的 `[min, max]`（含负值）；一个都没有时 undefined。 */
export function finiteMinMaxOf(a: ArrayLike<number>): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : undefined;
}

/** 严格等于 0 的样本个数（与"非有限"含义不同，本章要在文本里分开报）。 */
export function countExactZero(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if (a[i] === 0) n += 1;
  return n;
}

/** `|v| ≥ value·(1−relTol)` 的计数：显示上限"软着陆"（占位值触顶）的个数。 */
export function countAtCeiling(a: ArrayLike<number>, ceiling: number, relTol = 1e-9): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && Math.abs(v) >= ceiling * (1 - relTol)) n += 1;
  }
  return n;
}

/** 掩码（`> 0.5` 记 1）命中的个数。 */
export function countMask(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] ?? 0) > 0.5) n += 1;
  return n;
}

/** 一段下标区间 `[start, end)`。 */
export interface Run {
  start: number;
  end: number;
}

/**
 * 按段键把"顺序拼接"的数组切成若干段（键值相同且相邻 → 同一段）。
 *
 * `edge_tree` / `node_tree` 就是键：第 i 棵树的节点紧跟第 i+1 棵树的节点。
 * 键非有限的散点单独成段（不并进任何一棵树）。
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

/** 整数刻度 `lo, lo+step, …`（含两端，最多 `maxTicks` 条）——本章横轴都是序号。 */
export function integerTicks(lo: number, hi: number, step: number, maxTicks = 24): number[] {
  const out: number[] = [];
  const s = Math.max(1, Math.round(step));
  for (let v = Math.ceil(lo); v <= Math.floor(hi) && out.length < maxTicks; v += s) out.push(v);
  return out;
}

/**
 * 对数轴上的 stem（竖线）段：每个样本一根 `x → x`、`base → v` 的线段。
 *
 * 为什么要自己拼而不是用 `bars`：本章的副图横轴是**序号**（试步序号、条件序号），
 * 样本可能上千（`stiff_cos` 的 336 次试步），柱子必须"一像素宽"才不会糊成一片；
 * 而 `bars.width` 是数据单位、随轴范围变化，拼线段则与轴范围无关。
 * 返回的是**扁平的两点折线**（`[x,x]`,`[base,v]`），每段一条 —— 渲染层的断点语义是
 * "这里没有数据"，所以段与段之间不连线（不会出现一条假的包络）。
 */
export function stems(
  xs: ArrayLike<number>,
  vs: ArrayLike<number>,
  base: number,
  accept: (v: number, i: number) => boolean = () => true,
): { x: Float64Array; y: Float64Array }[] {
  const out: { x: Float64Array; y: Float64Array }[] = [];
  const n = Math.min(xs.length, vs.length);
  for (let i = 0; i < n; i += 1) {
    const x = xs[i];
    const v = vs[i];
    if (!Number.isFinite(x) || !Number.isFinite(v) || !accept(v, i)) continue;
    out.push({ x: Float64Array.from([x, x]), y: Float64Array.from([base, v]) });
  }
  return out;
}

/**
 * 把帧给的 `mark_*` 值夹成"对数轴上画得出来"的点：非正/非有限 → `undefined`。
 *
 * 本章的哨兵值是 `-1`（`work_at_target` 的"超出实测区间，拒绝外推"）与地板值 `1e-17`：
 * 前者**不是数值**，模型必须按掩码跳过并在文本里说明，绝不能画到图上。
 */
export function positiveOrUndefined(v: number | undefined): number | undefined {
  return v !== undefined && Number.isFinite(v) && v > 0 ? v : undefined;
}

/** 一个面板的绘图区在矩形内缩进（上下留标题带与刻度带）后的范围。 */
export function plotTopOf(rect: readonly [number, number, number, number]): number {
  return rect[1] + 22;
}
