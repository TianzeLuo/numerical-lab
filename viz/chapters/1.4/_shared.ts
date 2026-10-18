/**
 * 1.4 章五个 model 共用的**排版 / 显示几何**工具（纯几何与纯文本，不含任何物理量）。
 *
 * 为什么要有这个文件：`textBlock` / `wrapUnits` / `unitBudget` 是"模型自己折行"的同一套实现
 * （`render2d.ts::drawText` 不拆 `\n`，而光栅器的续行行距是 `size + 4`、模型按 15px 排行，
 * 把折行交给渲染层就会让续行压到下一行文字上）。1.1 / 3.2 章各有一份同样的 `_panel.ts`，
 * 本章照它们的口径走，**不再复制五份**。
 *
 * 另外三件事也在这里统一，因为它们都是"两侧共用一把尺子"的地方（PROTOCOLS §6.1/§6.2）：
 *
 * 1. {@link logRange}：对数轴范围。共享的 `niceLimits(..., {log:true})` 在跨度大时会一次
 *    跳十几个数量级（3.2 章实测 `[1.1e-16, 2e-6] → [1e-21, 0.1]`），本章的残差/范数曲线
 *    动辄跨 15~20 个数量级，必须用"包住数据 + 最多各让一格"的口径。
 * 2. {@link channelRange}：**排除地板哨兵**之后再取范围。本章的帧把"恰好为 0"写成
 *    `1e-300`（后端 `_sigma_safe` / `np.maximum(x, 1e-300)`），若把它当普通值参与取范围，
 *    纵轴会被拉到 300 个数量级、真数据压成一条直线。
 * 3. {@link footerFor}：底部 pitfalls 面板要占多少像素。`stackedPanels` 的缺省 footer 是
 *    102px（按 4 行估的），而 1.4 的 pitfalls 是 278~383 字（光栅器 5 行 = 88px 面板），
 *    照缺省值排就会让最下面那条轴名带被 pitfalls 面板盖住——PNG 基线里是"轴名被色块压掉"。
 *
 * 这些函数只做三件事：按共享尺子（`textWidthUnits`：CJK=2、ASCII=1）折行、把折好的行
 * 放进 `[top, bottom]` 的净空里、按数组自身的值切段/取范围。**不碰帧内容、不算物理量。**
 */

import { formatNumber } from "../../shared/src/axes.js";
import type { AxisSpecJson } from "../../shared/src/manifest.js";
import {
  BADGE_UNIT_PX,
  applyOverflow,
  placeBadge,
  stackedPanels,
  textWidthUnits,
  type Rect,
  type StackedPanels,
} from "../../shared/src/model.js";
import {
  AXIS_DECO,
  BASE_FONT_SIZE,
  overflowBadge,
  pitfallsBand,
  text,
  type TextSpec,
} from "../../shared/src/scene.js";
import type { Scene } from "../../shared/src/scene.js";

/** 面板内说明文字的字号（与 0.0 / 1.1 / 1.2 / 3.2 章一致）。 */
export const TEXT_SIZE = 11;
/** 行距：`textBoxOf` 按 `size + 4 = 15` 估盒，模型自己排行也必须用同一个数。 */
export const TEXT_ROW_H = TEXT_SIZE + 4;
/** 面板底边之上留出的净空（避开 `bottom+3..bottom+28` 的两条装饰带）。 */
export const SAFE_GAP = 18;

/**
 * 帧里的**地板哨兵**：后端把"恰好为 0 / 没有有限值"写成 `1e-300`
 * （`server/chapters/1.4.py::_sigma_safe` 与散落各处的 `np.maximum(x, 1e-300)`）。
 *
 * 为什么模型必须认识这个数：它在对数轴上是 -300，会让 `finitePositiveMinMax` 把轴范围
 * 拉到 300 个数量级；而把它当"真实的小值"画出来又正好读反（"σ_min = 1e-300" 比
 * "σ_min 恰好是 0"更荒谬）。本章的统一口径是：**≤ 1e-299 的值一律读作"在地板上"**
 * （不是 0，而是"这个量在本帧量不出正值"），取范围时排除、画图时贴轴底并报数
 * （`PROTOCOLS` §1.3 的占位值 + 掩码 + 文案三件套里，模型这侧负责后两件）。
 */
export const LOG_FLOOR = 1e-300;

/** 某个值是不是"在地板上"（见 {@link LOG_FLOOR}）。 */
export function atFloor(v: number): boolean {
  return Number.isFinite(v) && v <= 1e-299;
}

/** 面板宽度 → 显示宽度单位预算（1 单位 = 7px，与 `textWidthUnits` 同一把尺子）。 */
export function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / BADGE_UNIT_PX));
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
 * `anchor`：`"top"` 从 `top` 往下排；`"bottom"` 让整块贴着 `bottom`。
 * 返回数组的长度 = 实际排出的行数（调用方据此把下一块往下推）。
 *
 * 画布小的时候宁可**截断并说明**，也不让文字叠在一起：能放下的行数由 `[top, bottom]` 的
 * 净空决定，最后一行留给"还有 N 行未画"。
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

/** 面板内文字块的上边界（面板矩形 → 第一行 y）。 */
export function textTop(rect: readonly [number, number, number, number]): number {
  return rect[1] + 4;
}

/** 面板内文字块的下边界（面板矩形 → 最后一行允许的 y；`SAFE_GAP` 见文件头）。 */
export function textBottom(rect: readonly [number, number, number, number]): number {
  return rect[1] + rect[3] - SAFE_GAP;
}

/** 轴**下方**文字块的起始 y：让开刻度带与轴名带（`bottom+3..bottom+28`）。 */
export function belowAxis(rect: readonly [number, number, number, number]): number {
  return rect[1] + rect[3] + 32;
}

/** manifest 里该轴的标题（文案直接取 manifest，缺省才用回落值——不许模型自己发明轴名）。 */
export function axisTitleOf(axes: readonly AxisSpecJson[] | undefined, id: string, fallback: string): string {
  return axes?.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel（同上）。 */
export function axisYlabelOf(axes: readonly AxisSpecJson[] | undefined, id: string, fallback: string): string {
  return axes?.find((a) => a.id === id)?.ylabel ?? fallback;
}

/**
 * 对数轴范围：把 `[lo, hi]` 扩到"整十次幂"的上下界（最多各让一格）。
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
  if (loExact - l < 0.05) l -= 1;
  if (h - hiExact < 0.05) h += 1;
  if (h - l < 1) h = l + 1;
  return [10 ** l, 10 ** h];
}

/**
 * 一串值取 `[min, max]`，**排除地板哨兵与 ≤0**（见 {@link LOG_FLOOR}）。
 *
 * `salt` 是"无论如何都要包进去"的锚点（例如比值的参照线 1.0）：即使数据全在地板上，
 * 轴也不会退化成 `[1e-300, 1e-300]`。
 */
export function channelRange(
  a: ArrayLike<number>,
  salt: readonly number[] = [],
): { lo: number; hi: number; nFloor: number; nUsable: number } {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  let nFloor = 0;
  let nUsable = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    if (v <= 1e-299) {
      nFloor += 1;
      continue;
    }
    nUsable += 1;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  for (const v of salt) {
    if (!Number.isFinite(v) || v <= 0) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo: 1, hi: 10, nFloor, nUsable };
  return { lo, hi, nFloor, nUsable };
}

/** 命中掩码（`> 0.5` 记 1）的个数。 */
export function countMask(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] ?? 0) > 0.5) n += 1;
  return n;
}

/** 掩码为 0（不可信）的个数。 */
export function countNotMask(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if (!((a[i] ?? 0) > 0.5)) n += 1;
  return n;
}

/** 严格大于 `threshold` 的有限值个数。 */
export function countAbove(a: ArrayLike<number>, threshold: number): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > threshold) n += 1;
  }
  return n;
}

/** 有限值的最大值（一个都没有时 undefined）。 */
export function maxFinite(a: ArrayLike<number>): number | undefined {
  let hi = Number.NEGATIVE_INFINITY;
  let seen = false;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    seen = true;
    if (v > hi) hi = v;
  }
  return seen ? hi : undefined;
}

/** 最大值所在下标（只扫有限值；一个都没有时 -1）。用于把帧 `meta` 里的极值标在曲线上。 */
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

/** 整数刻度（0 起步，按面板宽度取步进；`count` 是样本数）。 */
export function integerTicks(panelWidth: number, count: number): number[] {
  const target = Math.max(3, Math.round(panelWidth / 90));
  const step = Math.max(1, Math.ceil(count / target));
  const out: number[] = [];
  for (let v = 0; v < count; v += step) out.push(v);
  return out;
}

/**
 * 科学计数写法（`1e-300` 这类地板值用它）。
 *
 * `formatNumber` 走的是"有效数字"口径，写 300 个数量级的指数会很长；这里对
 * `|e| > 5` 一律用 `m.mme±E`，方便在文本里对齐比较（例如 `6.02e-15`）。
 */
export function fmtSci(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(v)));
  if (e >= -3 && e <= 5) return formatNumber(v, 4);
  const m = v / 10 ** e;
  return `${m.toFixed(digits)}e${e >= 0 ? "+" : ""}${e}`;
}

/** 一段下标区间 `[start, end)`。 */
export interface Run {
  start: number;
  end: number;
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

/**
 * 半对数斜率（每步的 `log10(y)` 变化率）：最小二乘拟合 `log10 y = a + b·x`。
 *
 * 这是**显示用拟合**（与 1.3 章 `powerLawSlope`、`shared/src/loglogfit.ts` 同一性质），
 * 只用帧里已有的曲线，不产生任何新的物理量；点在 2 个以下或全在地板上时返回 undefined，
 * 调用方据此说"斜率算不出来"而不是编一个数。
 */
export function log10Slope(x: ArrayLike<number>, y: ArrayLike<number>): number | undefined {
  const n = Math.min(x.length, y.length);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const xv = x[i];
    const yv = y[i];
    if (!Number.isFinite(xv) || !Number.isFinite(yv) || yv <= 1e-299) continue;
    xs.push(xv);
    ys.push(Math.log10(yv));
  }
  if (xs.length < 2) return undefined;
  const mx = xs.reduce((s, v) => s + v, 0) / xs.length;
  const my = ys.reduce((s, v) => s + v, 0) / ys.length;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i += 1) {
    const dx = (xs[i] as number) - mx;
    sxx += dx * dx;
    sxy += dx * ((ys[i] as number) - my);
  }
  return sxx > 0 ? sxy / sxx : undefined;
}

/**
 * 底部 pitfalls 面板的位置与"模型该让出多少"。
 *
 * 面板不在 Scene 里——它是渲染层拿 `scene.pitfalls` 现画的（`render2d.ts::drawPitfalls` /
 * `tools/raster.py::_draw_pitfalls`），所以模型只能靠共享的 {@link pitfallsBand} 反推它的盒子。
 * `validateScene` 与 `raster.py` 都会检查"文本层有没有被 pitfalls 面板压住"（规则 7），
 * 于是这两个数就是模型排版时的硬约束：
 *
 * * `footer`：`stackedPanels(..., {footer})` 用，让最下面那条轴的刻度/轴名带（`bottom+3..+28`）
 *   整个落在面板之上，再留 16px 余量（面板行数按 7px/单位估，位图字体的实际折行可能差一行）；
 * * `textLimit`：模型自己写的文本块**不得越过**这条线（越过就是"被面板压住"）。
 *
 * 为什么不能沿用 `stackedPanels` 的缺省 footer 102：那是按 4 行估的，而 1.4 的 pitfalls
 * 是 278~383 字（960 宽下 4~5 行、窄画布上十几行），照缺省值排就会压掉最下面那条轴名。
 */
export interface PitfallGuard {
  /** 底部总留白（交给 `stackedPanels`）。 */
  footer: number;
  /** 模型自写文本的 y 上界（盒子底 ≤ 它）。 */
  textLimit: number;
  /** 面板盒子的顶边 y（诊断/文本用）。 */
  bandTop: number;
}

export function pitfallsGuard(width: number, height: number, pitfalls: string): PitfallGuard {
  const band = pitfallsBand({ canvas: { w: width, h: height }, pitfalls }, BASE_FONT_SIZE);
  const bandTop = band === null ? height - 10 : band[1];
  return { footer: Math.round(height - bandTop + 44), textLimit: Math.round(bandTop - 6), bandTop };
}

/**
 * 面板内文本的显示宽度预算，**扣掉右边那个面板的 y 轴装饰带**。
 *
 * 右邻居的 `yticks` 带是 `[x−41, ·, 35, h]`、`ylabel` 带是 `[x−56, ·, 14, h]`
 * （`scene.ts::axisDecoBands`），它们盖住整根面板的高度；而主面板里/下方的文字
 * 只要横着伸进那 56px，就会被校验器判"文本压住坐标轴装饰"。
 * `chapterLayout` 给的 `gutter` 已经含了这段过道，这里按同一口径把预算收窄 12px 余量。
 */
export function panelBudget(rect: Rect, blockers: readonly Rect[] = []): number {
  let right = rect[0] + rect[2];
  for (const b of blockers) {
    if (b[0] <= rect[0]) continue; // 只算右边的邻居
    // 关键：邻居的**轴装饰带**（y 刻度 + 轴名，`AXIS_DECO.left = 56`）在它的 rect **左边**，
    // 文本压到那儿也算压住坐标轴（`validateScene` 会判违规）。所以要让到 `b[0] − 56`。
    right = Math.min(right, b[0] - AXIS_DECO.left - 10);
  }
  const units = unitBudget(Math.max(120, right - rect[0]));
  // 窄面板要再收一次：`unitBudget` 有 24 单位（168px）的下限，那是给角标留的余量；
  // 面板宽度只有 140px 时，这个下限会把文本挤出面板、压到邻居的 y 轴装饰带上
  // （`validateScene` 的规则 8 会判定违规）。这里按面板实宽算一个更紧的上限。
  const capUnits = Math.max(8, Math.floor((rect[2] - 12) / BADGE_UNIT_PX));
  return Math.min(units, capUnits);
}

/** 竖直切分一个矩形（右列放 2 块面板时用）；与帧无关，动画不会跳。 */
export function splitColumn(rect: Rect, shares: readonly number[], gap: number): Rect[] {
  const [x, y, w, h] = rect;
  const total = shares.reduce((s, v) => s + v, 0);
  const usable = h - gap * (shares.length - 1);
  const out: Rect[] = [];
  let cursor = y;
  for (const share of shares) {
    const hi = Math.max(40, Math.round((usable * share) / total));
    out.push([x, Math.round(cursor), w, hi]);
    cursor += hi + gap;
  }
  return out;
}

export interface ChapterLayout {
  base: StackedPanels;
  /** 左边的方形面板（复平面 / 单位圆）。 */
  main: Rect;
  /** 右列（或窄画布的下一行）分块，按 `rightShares` 切。 */
  right: Rect[];
  /** 是否真的排成了两列（测试据此断言"窄画布退化"路径）。 */
  columns: boolean;
}

/**
 * 「上面一块宽的 + 下面一块宽的」版式（`qr_schur` 的次对角元热力图用）。
 *
 * 为什么这一页不能套"左方形 + 右列"：manifest 给它的轴标题是
 * 「log10|次对角元|（**行 = i，列 = 迭代步**）」——热力图的宽高比必须跟着 `[n−1, iters+1]`
 * 这个矩阵形状走（7 行 × 22~61 列），做成正方形会让每个格点变成 30×5 的长条，
 * 「哪一列（哪一步）整列变暗」就看不出来了。
 *
 * `gutter` 留给光栅器的自动色标（画在轴右侧 46px 处）：**宽面板的色标只能画在右边**，
 * 不留过道就会被画到画布外面裁掉（基线里就没有色标了）。
 */
export interface WideLayout {
  base: StackedPanels;
  /** 上面那块（热力图）。 */
  main: Rect;
  /** 下面那块（曲线）。 */
  aux: Rect;
}

export function wideLayout(
  width: number,
  height: number,
  opts: { footer: number; topShare?: number; topHeight?: number; gutter?: number; gap?: number; minAux?: number },
): WideLayout {
  const base = stackedPanels(width, height, { footer: opts.footer, headerExtra: 14 });
  const plotH = Math.max(120, base.bottom - base.top);
  const gap = opts.gap ?? 34;
  const minAux = opts.minAux ?? 120;
  const want = opts.topHeight ?? Math.round(plotH * (opts.topShare ?? 0.45));
  const topH = Math.max(72, Math.min(want, plotH - minAux - gap));
  const gutter = opts.gutter ?? 0;
  return {
    base,
    main: [base.left, base.top, Math.max(120, base.contentWidth - gutter), topH],
    aux: [base.left, base.top + topH + gap, base.contentWidth, Math.max(60, plotH - topH - gap)],
  };
}

/** 把"在地板上的值"（见 {@link LOG_FLOOR}）换成一个可画的显示值，并数出个数。 */
export function replaceFloors(a: ArrayLike<number>, replacement: number): { values: Float64Array; nFloor: number } {
  const out = new Float64Array(a.length);
  let nFloor = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && v <= 1e-299) {
      out[i] = replacement;
      nFloor += 1;
    } else {
      out[i] = v;
    }
  }
  return { values: out, nFloor };
}

/** 数一数有多少个值 ≤ 0（对数轴上画不出来，必须显式丢弃并报数）。 */
export function countNonPositive(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v <= 0) n += 1;
  }
  return n;
}

/**
 * 本章的两种版式：**宽画布两列**（左方形主图 / 右列分块），**窄画布单列**（主图在上，分块排下面）。
 *
 * `gutter` 是给**光栅器的自动色标**留的过道（`tools/raster.py::_draw_colorbar` 把色标画在
 * 轴右侧 46px、宽 14px，再加数值标签与居中说明）：1.2 章的 `lu_heatmap` 为它留了 162px，
 * 本章沿用同一口径。**只有 `columns = true` 时才留**——窄画布下右列已经排到主图下面去了。
 */
export function chapterLayout(
  width: number,
  height: number,
  opts: {
    footer: number;
    rightShares: readonly number[];
    gutter?: number;
    mainReserve?: number;
    minRight?: number;
    /** 给定则主面板是"定宽整幅高"的（见函数内注释），而不是正方形。 */
    mainWidth?: number;
    /**
     * 顶部在 `headerBand(width)` 之外再让出的高度（缺省 14，比共享的 8 多 6px）。
     *
     * 为什么本章要多让 6px：`BADGE_LAYOUT` 是 `{top:34, rowH:18}`，8 枚角标在 960 宽下排 4 行、
     * 最后一行到 `34+3*18+17 = 105`；而面板标题带从 `panelTop−18` 起算（`AXIS_DECO.top`），
     * 用共享缺省值 `panelTop = headerBand+8 = 122` 时标题带是 `[104, 120]`——和最后一行角标
     * 只差 1px。多让 6px 就到 `[110, 126]`，角标与轴标题不再贴脸。
     */
    headerExtra?: number;
  },
): ChapterLayout {
  const base = stackedPanels(width, height, { footer: opts.footer, headerExtra: opts.headerExtra ?? 14 });
  const plotH = Math.max(80, base.bottom - base.top);
  // 同一列里上方面板的刻度带 + 轴名带占到 `bottom+28`（`AXIS_DECO.bottom`），
  // 下方面板的标题带又从 `top−18` 起算：间距必须 > 46，这里取 38 + 下方面板不画标题。
  const gap = 38;
  const gutter = opts.gutter ?? 0;
  const minRight = opts.minRight ?? 160;
  const reserve = opts.mainReserve ?? 0;
  if (opts.mainWidth !== undefined) {
    // 定宽主面板（整幅高）：给"格点形状决定面板形状"的热力图用（见 `wideLayout` 的说明）。
    const w = Math.max(140, Math.min(opts.mainWidth, base.contentWidth - gutter - minRight));
    const rightRect: Rect = [base.left + w + gutter, base.top, base.contentWidth - w - gutter, plotH];
    return { base, main: [base.left, base.top, w, plotH], right: splitColumn(rightRect, opts.rightShares, gap), columns: true };
  }
  const wide = Math.max(120, Math.min(plotH - reserve, Math.floor(base.contentWidth * 0.5)));
  if (base.contentWidth - wide - gutter >= minRight && wide >= 150) {
    const main: Rect = [base.left, base.top, wide, wide];
    const rightRect: Rect = [base.left + wide + gutter, base.top, base.contentWidth - wide - gutter, plotH];
    return { base, main, right: splitColumn(rightRect, opts.rightShares, gap), columns: true };
  }
  const nRight = Math.max(1, opts.rightShares.length);
  const belowNeed = 64 * nRight + gap;
  const side = Math.max(96, Math.min(base.contentWidth, plotH - belowNeed));
  const main: Rect = [base.left, base.top, side, side];
  const belowTop = base.top + side + gap;
  const belowRect: Rect = [base.left, belowTop, base.contentWidth, Math.max(40, base.bottom - belowTop)];
  return { base, main, right: splitColumn(belowRect, opts.rightShares, gap), columns: false };
}

/**
 * 把溢出写进 Scene：先跑共享的 `applyOverflow`（图层越界 + 非有限值），
 * 再把**模型这一侧发现的**截断/占位（地板哨兵、掩码格点、被丢掉的 0）并进同一条 note。
 *
 * 为什么要合并：`applyOverflow` 只看"画出来的数据有没有越界"，而本章最贵的几个坑
 * （`1e-300` 地板、σ_min 的 `trusted=0` 格点、对数轴上被丢掉的 0）都发生在**画之前**——
 * 不写进同一条 note，它们就成了"图上没有、账上也没有"的东西（规则 1）。
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
  scene.overflow = {
    flag: true,
    note: scene.overflow.note === "" ? trimmed : `${scene.overflow.note}；${trimmed}`,
  };
  if (!scene.badges.some((b) => b.kind === "overflow")) {
    scene.badges.push(placeBadge(overflowBadge(trimmed), opts.slot, opts.width));
  }
}
