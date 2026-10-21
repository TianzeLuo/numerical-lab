/**
 * 3.2 章五个 model 共用的**排版 / 显示几何**工具（纯几何与纯文本，不含任何物理量）。
 *
 * 为什么要有这个文件：`textBlock` / `wrapUnits` / `unitBudget` 是"模型自己折行"的同一套实现
 * （`render2d.ts::drawText` 不拆 `\n`，光栅器的续行行距是 `size + 4` 而模型按 15px 排行，
 * 把折行交给渲染层就会让续行压到下一行文字上）。1.1 章的 `_panel.ts` 已经把这条教训写进注释，
 * 本章照它的口径走，**不再复制五份**。
 *
 * 另外三个本章专用的纯显示工具：
 *
 * 1. {@link runsBy}：把后端"顺序拼接"的折线按段落键切成若干条独立曲线。
 *    `bdf_angle` 的 `family_x/family_y` 就是 k=1…6 六条边界首尾相接拼成的（没有 NaN 分隔符——
 *    后端注释与 `strings.wedge_note` 都写明了理由），前端必须按 `family_k` 切段，
 *    **不许自己插 NaN**，也不许当成一条折线画（那会画出六条横穿画面的假连线）。
 * 2. {@link clipRuns}：把越出固定纵轴的数据切成"可画的连续段"，并如实报告被裁掉多少。
 *    裁掉的部分不能画成一条贴着上边的水平线——那看起来像"解稳定在上限"，正好读反。
 * 3. {@link countAbove} / {@link countEqual}：热力图色标截断、显示上限触顶的**计数**。
 *    规则 1 要的是"溢出必须显式标记"，计数是标记的依据，不是一个估计。
 *
 * 这些函数只做三件事：按共享尺子（`textWidthUnits`：CJK=2、ASCII=1）折行、把折好的行
 * 放进 `[top, bottom]` 的净空里、按数组自身的值切段。**不碰帧、不碰轴范围、不算物理量。**
 *
 * `SAFE_GAP` 的口径（规则 8）：`axisDecoBands` 给面板下沿的装饰带是
 * `[bottom+3, bottom+15]`（刻度标签）与 `[bottom+16, bottom+28]`（轴名）。
 * 一段文字的盒子高度按 `行数 × (size+4)` 估（`scene.ts::textBoxOf`），
 * 所以"贴边放"必须让最后一行的盒子底 ≤ `bottom−3`：本模块留 **18px**。
 */

import type { AxisSpecJson } from "../../shared/src/manifest.js";
import {
  applyOverflow,
  placeBadge,
  stackedPanels,
  textWidthUnits,
  type Rect,
  type StackedPanels,
} from "../../shared/src/model.js";
import { overflowBadge, text, type TextSpec } from "../../shared/src/scene.js";
import type { Scene } from "../../shared/src/scene.js";

/** 面板内说明文字的字号（与 0.0 / 1.1 / 1.2 章一致）。 */
export const TEXT_SIZE = 11;
/** 行距：`textBoxOf` 按 `size + 4 = 15` 估盒，模型自己排行也必须用同一个数。 */
export const TEXT_ROW_H = TEXT_SIZE + 4;
/** 面板底边之上留出的净空（见文件头：避开 `bottom+3..bottom+28` 的两条装饰带）。 */
export const SAFE_GAP = 18;

/**
 * 面板宽度 → 显示宽度单位预算（1 单位 = 7px，与 `textWidthUnits` 同一把尺子）。
 *
 * 留 12px 边距再折算：折出来的行一定装得进面板，两个渲染器都不会再自动折行。
 */
/**
 * 对数轴范围：把 `[lo, hi]` 扩到"整十次幂"的上下界。
 *
 * 为什么不直接用共享的 `niceLimits(lo, hi, target, {log:true})`：它在 log 模式下会按
 * tick 目标做"漂亮化"，跨度一大就一次跳十几个数量级——实测 `[1.1e-16, 2e-6]` 被它扩成
 * `[1e-21, 0.1]`、`[1.1e-16, 1165]` 扩成 `[1e-21, 1e15]`，曲线当场被压成一条直线。
 * 这里只要"包住数据 + 最多各让一格"，读数仍然落在整十次幂的刻度上。
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

export function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/**
 * 按显示宽度单位折行（CJK=2、ASCII=1；口径同 `model.ts::textWidthUnits`）。
 *
 * 先按空白（含全角空格）切"词"再往行里塞：`k=6`、`|R|=1.33e+07` 这类字段不会被从中间劈开
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
 * 画布小的时候宁可**截断并说明**，也不让文字叠在一起：能放下的行数由 `[top, bottom]` 的
 * 净空决定，最后一行留给"还有 N 行"。
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
    // 行数按 "整行" 算（不 +1）：最后一行的盒子高度是 15px，留一行余量才不会压到
    // 面板下方的刻度带（[bottom+3, bottom+15]）——窄画布上那是真会被校验器抓到的。
    const maxRows = Math.max(1, Math.floor((bottom - top) / TEXT_ROW_H));
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      // 截断标记必须**短到不会折行**：它一旦折成两行，盒子高度翻倍就会越过面板底线。
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行未画）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - (pieces.length - 1) * TEXT_ROW_H);
  return pieces.map((body, i) => text(x, y + i * TEXT_ROW_H, body, { role: "note", size: TEXT_SIZE, color, maxWidth: budget * 7 }));
}

/** 面板内文字块的上边界（面板矩形 → 第一行 y）。 */
export function textTop(rect: readonly [number, number, number, number]): number {
  return rect[1] + 4;
}

/** 面板内文字块的下边界（面板矩形 → 最后一行允许的 y；`SAFE_GAP` 见文件头）。 */
export function textBottom(rect: readonly [number, number, number, number]): number {
  return rect[1] + rect[3] - SAFE_GAP;
}

/** manifest 里该轴的标题（文案直接取 manifest，缺省才用回落值——不许模型自己发明轴名）。 */
export function axisTitleOf(axes: readonly AxisSpecJson[] | undefined, id: string, fallback: string): string {
  return axes?.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel（同上）。 */
export function axisYlabelOf(axes: readonly AxisSpecJson[] | undefined, id: string, fallback: string): string {
  return axes?.find((a) => a.id === id)?.ylabel ?? fallback;
}

/** 有限值的绝对值上界（只是显示范围，不是物理量）；一个有限值都没有时 undefined。 */
export function absMax(a: ArrayLike<number>): number | undefined {
  let hi = 0;
  let seen = false;
  for (let i = 0; i < a.length; i += 1) {
    const v = Math.abs(a[i]);
    if (!Number.isFinite(v)) continue;
    seen = true;
    if (v > hi) hi = v;
  }
  return seen ? hi : undefined;
}

/** 整数刻度（`lo, lo+step, …`，含两端，最多 `maxTicks` 条）。 */
export function integerTicks(lo: number, hi: number, step: number, maxTicks = 12): number[] {
  const out: number[] = [];
  const s = Math.max(1, Math.round(step));
  for (let v = Math.ceil(lo); v <= Math.floor(hi) && out.length < maxTicks; v += s) out.push(v);
  return out;
}

/** 一段下标区间 `[start, end)`。 */
export interface Run {
  start: number;
  end: number;
}

/**
 * 把"顺序拼接"的数组按段落键切成若干段（键值相同且相邻 → 同一段）。
 *
 * `bdf_angle` 的 `family_k` 就是键：k=1 的 241 个点后紧跟 k=2 的 241 个点……
 * 相邻两段的键必然不同（后端逐阶拼接），所以这里只比较前后两个键就够。
 * 键非有限/为 0 的散点单独成段（不并进任何一条边界）。
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
    if (usable) {
      if (start < 0) {
        start = i;
        current = v;
      }
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

/** 最大值所在的下标（只扫有限值；一个都没有时 -1）。用于把帧 `meta` 里的极值标在曲线上。 */
export function argmaxIndex(a: ArrayLike<number>): number {
  let best = -1;
  let bestV = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    if (best < 0 || v > bestV) {
      best = i;
      bestV = v;
    }
  }
  return best;
}

/** 严格大于 `threshold` 的有限值个数（色标截断计数用）。 */
export function countAbove(a: ArrayLike<number>, threshold: number): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > threshold) n += 1;
  }
  return n;
}

/** 命中掩码（`> 0.5` 记 1）的个数。 */
export function countMask(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    if ((a[i] ?? 0) > 0.5) n += 1;
  }
  return n;
}

/** 触顶（`v ≥ value·(1−relTol)` 且有限）的个数：显示上限的"软着陆"计数。 */
export function countAtCeiling(a: ArrayLike<number>, ceiling: number, relTol = 1e-12): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v >= ceiling * (1 - relTol)) n += 1;
  }
  return n;
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

/** 把帧给的一个**标量**铺成长度 `n` 的数组（只用于"哪些点触顶/贴地板"的显示判断）。 */
export function constantArray(n: number, value: number): Float64Array {
  const out = new Float64Array(Math.max(0, n));
  out.fill(value);
  return out;
}

/** 把一段折线按固定纵轴 `[lo, hi]` 裁成"可画的连续段"，并报告被裁掉多少（规则 1）。 */
export interface ClipResult {
  /** 全部落在 `[lo, hi]` 内的连续段（保持原始 x）。 */
  runs: { x: Float64Array; y: Float64Array }[];
  /** 越界的样本个数。 */
  nClipped: number;
  /** 第一个越界样本（x, 被夹到边界的 y）；没有越界时为 undefined。 */
  firstClip?: [number, number];
  /** 被裁掉的样本的时间范围（用于文本说明"哪一段没画"）。 */
  clippedSpan?: [number, number];
}

export function clipRuns(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  lo: number,
  hi: number,
  accept: (v: number) => boolean = () => true,
): ClipResult {
  const n = Math.min(x.length, y.length);
  const runs: ClipResult["runs"] = [];
  let start = -1;
  let nClipped = 0;
  let firstClip: [number, number] | undefined;
  let clipLo = Number.POSITIVE_INFINITY;
  let clipHi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < n; i += 1) {
    const xv = x[i];
    const yv = y[i];
    const inside = Number.isFinite(xv) && Number.isFinite(yv) && yv >= lo && yv <= hi && accept(yv);
    if (inside) {
      if (start < 0) start = i;
      continue;
    }
    if (start >= 0) {
      runs.push(sliceRun(x, y, { start, end: i }));
      start = -1;
    }
    if (Number.isFinite(xv) && Number.isFinite(yv) && (yv < lo || yv > hi)) {
      nClipped += 1;
      if (firstClip === undefined) firstClip = [xv, yv < lo ? lo : hi];
      if (xv < clipLo) clipLo = xv;
      if (xv > clipHi) clipHi = xv;
    }
  }
  if (start >= 0) runs.push(sliceRun(x, y, { start, end: n }));
  return {
    runs,
    nClipped,
    firstClip,
    clippedSpan: nClipped > 0 ? [clipLo, clipHi] : undefined,
  };
}

/**
 * 竖直切分一个矩形（右列放 2~3 块面板时用）。
 *
 * `shares` 是各块的高度占比（会归一化），`gap` 是块间距。**与帧无关**：同一参数下
 * 连续帧的面板位置逐像素相同，动画里不会跳。
 */
export function splitColumn(
  rect: readonly [number, number, number, number],
  shares: readonly number[],
  gap: number,
): [number, number, number, number][] {
  const [x, y, w, h] = rect;
  const total = shares.reduce((s, v) => s + v, 0);
  const usable = h - gap * (shares.length - 1);
  const out: [number, number, number, number][] = [];
  let cursor = y;
  for (const share of shares) {
    const hi = Math.max(40, Math.round((usable * share) / total));
    out.push([x, Math.round(cursor), w, hi]);
    cursor += hi + gap;
  }
  return out;
}

/**
 * 本章的两种版式：**宽画布用两列**（左方形复平面 / 左主图，右列分块），
 * **窄画布退化成单列**（方panel 在上，各分块依次排在下面）。
 *
 * 为什么主面板一定要是正方形：这一章的四个复平面图里，`|R|=1` 的边界形状（圆盘、外凸环、
 * 贴着虚轴的窄带、A(α) 楔形）**就是结论本身**。面板一旦被拉成长方形，"梯形法把整个左半平面
 * 都装进去"与"前向 Euler 只有一个圆盘"看起来就会差不多——这正是 pitfalls 第 2 条的坑。
 *
 * `gutter` 是给**光栅器的自动色标**留的过道（`tools/raster.py::_draw_colorbar` 把色标画在
 * 轴右侧 46px、宽 14px，再加数值标签与居中说明）：1.2 章的 `lu_heatmap` 为它留了 162px，
 * 本章沿用同一口径。窄画布下不留（那时右列已经排到下方去了）。
 *
 * 与帧无关：同一 params 下连续帧的面板位置逐像素相同，动画不会跳。
 */
export interface ChapterLayout {
  base: StackedPanels;
  /** 左边的方形面板（复平面 / 主图）。 */
  main: Rect;
  /** 右列（或窄画布的下一行）分块，按 `rightShares` 切。 */
  right: Rect[];
  /** 是否真的排成了两列（测试据此断言"窄画布退化"路径）。 */
  columns: boolean;
}

export function chapterLayout(
  width: number,
  height: number,
  opts: { rightShares: readonly number[]; gutter?: number; minRight?: number; mainReserve?: number },
): ChapterLayout {
  const base = stackedPanels(width, height);
  // 同一个竖直列里，上方面板的"轴名"装饰带是 [bottom+16, bottom+28]：两块的间距必须 > 28，
  // 否则下方那一块的顶部文字（top+4）会压进它的轴名带里——校验器会当场报"规则 8"。
  const gap = 30;
  const plotH = Math.max(80, base.bottom - base.top);
  const gutter = opts.gutter ?? 0;
  const minRight = opts.minRight ?? 150;
  const reserve = opts.mainReserve ?? 0;
  const wide = Math.max(120, Math.min(plotH - reserve, Math.floor(base.contentWidth * 0.46)));
  if (base.contentWidth - wide - gutter >= minRight) {
    const main: Rect = [base.left, base.top, wide, wide];
    const rightRect: Rect = [base.left + wide + gutter, base.top, base.contentWidth - wide - gutter, plotH];
    return { base, main, right: splitColumn(rightRect, opts.rightShares, gap), columns: true };
  }
  // 窄画布：主面板在上（仍是正方形），各分块排在下面；必要时压缩主面板高度。
  const nRight = Math.max(1, opts.rightShares.length);
  const belowNeed = 62 * nRight + gap;
  const side = Math.max(96, Math.min(base.contentWidth, plotH - belowNeed));
  const main: Rect = [base.left, base.top, side, side];
  const belowTop = base.top + side + gap;
  const belowRect: Rect = [base.left, belowTop, base.contentWidth, Math.max(40, base.bottom - belowTop)];
  return { base, main, right: splitColumn(belowRect, opts.rightShares, gap), columns: false };
}

/**
 * 把溢出写进 Scene：先跑共享的 `applyOverflow`（图层越界 + 非有限值），
 * 再把**模型这一侧发现的**截断/占位（色标上限、显示上限、掩码格子）并进同一条 note。
 *
 * 为什么要合并：`applyOverflow` 只看"画出来的数据有没有越界"，而本章最贵的两个坑
 * （热力图色标截断、`pole_fill` 占位值）都发生在**画之前**——不写进同一条 note，
 * 它们就成了"图上没有、账上也没有"的东西，正是规则 1 要禁止的。
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
