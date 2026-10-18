/**
 * 2.1 章四个 viz 共用的**排版与文本**工具（纯几何：只依赖画布尺寸与 manifest 文案，
 * 与帧、与 DOM 都无关）。
 *
 * 为什么单独一个文件而不是各自抄一份：本章四个 viz 的版面完全同构——
 * 「主面板 + 误差/残差面板 + 一个量化面板」三行堆叠 + 每行左上角一个数值块。
 * 抄四份的话，"底部要给 pitfalls 面板让出多少像素"这种数字会在四处漂移，
 * 而规则 8 的报错恰恰只在光栅化那一刻才出现。
 *
 * 三条硬约定（写在代码里，而不是靠注释提醒）：
 *  1. `rowLayout` 的 `bottom` 已经扣掉 pitfalls 面板与最底轴的 xlabel 装饰带；
 *     正文文本只允许落在 `[top, bottom]` 之内（`textBlock` 不接受越界锚点）。
 *  2. 文本块**一行一个 `text` 层**：`render2d.ts::drawText` 不拆 `\n`，而
 *     `tools/raster.py` 会拆——把多行字符串交给两个渲染器等于让它们排版分叉。
 *  3. 折行用共享尺子 `textWidthUnits`（CJK=2、ASCII=1），并且模型这一侧就折好，
 *     免得两个渲染器各自的自动折行把下一行文字压住。
 */

import { formatNumber } from "../../shared/src/axes.js";
import type { Rect } from "../../shared/src/model.js";
import { textWidthUnits } from "../../shared/src/model.js";
import type { TextSpec } from "../../shared/src/scene.js";
import { text } from "../../shared/src/scene.js";

/** 一行文本的显示宽度单位预算（size 11 时 1 单位 ≈ 7px，与 `BADGE_UNIT_PX` 同尺子）。 */
export function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 16) / 7));
}

export interface RowLayout {
  /** 三行面板矩形（顺序 = 自上而下）。 */
  rects: Rect[];
  left: number;
  right: number;
  contentWidth: number;
  top: number;
  bottom: number;
  /** 相邻两行之间的竖直间距（= `gap`）。 */
  gap: number;
}

export interface RowLayoutOptions {
  left?: number;
  right?: number;
  /** 底部留白：pitfalls 面板（规则 7 的画布底部面板）+ 最底轴的 xlabel 装饰带。 */
  footer?: number;
  gap?: number;
  headerExtra?: number;
}

/**
 * 「主 / 误差 / 量化」三行堆叠排版。
 *
 * `headerBand(width)` 之后才是正文（角标区 114px / 单列 186px，见 PROTOCOLS §6.2）；
 * `weights` 是三行的相对高度，实际高度按可用空间等比分配（保证任何画布尺寸下
 * 三个面板都 ≥ 44px，且**与帧无关**——同一 params 的连续帧逐像素同位置，动画不跳）。
 *
 * `footer` 缺省 130px 的来历：`renderScene` 在画布底部画 pitfalls 面板
 * （`size = base − 1 = 12`、行高 16、上下各 4/10px 边距），本章最长的 pitfalls
 * （`barycentric_vs_vandermonde`，约 500 字）在 960 宽下折成 7 行 ≈ 120px；
 * 再加上最底轴的刻度带（15px）与 xlabel 带（12px）。四个 model 各自把**实测行数**
 * 写死在 `footer` 里（常量，不随帧变——否则连续帧的面板会上下跳），
 * 基线 PNG 复核过：底轴 xlabel 与 pitfalls 框之间有 6px 净空。
 *
 * **相邻两行之间至少要 36px**：下面那一行的轴标题占 `[top−18, top−2]`，
 * 上面那一行的刻度标签占 `[bottom+3, bottom+15]`；上面那一行的 `xlabel` 再占
 * `[bottom+16, bottom+28]`，那就需要 46px。所以本章除最底一行外**一律把 `xlabel` 置空**
 * （刻度保留，横轴身份由最底那一行的轴名与各面板标题交代）——36px 因此够用。
 * 这两个数字都不是风格问题：少 1px 就是标题压刻度、渲染出来两层字叠在一起。
 */
export function rowLayout(width: number, height: number, weights: readonly number[], opts: RowLayoutOptions = {}): RowLayout {
  const left = opts.left ?? 64;
  const right = opts.right ?? 26;
  const footer = opts.footer ?? 130;
  const gap = opts.gap ?? 36;
  const rows = Math.max(1, weights.length);
  const top = headerBandOf(width) + (opts.headerExtra ?? 8);
  const bottom = Math.max(top + 3 * 44, height - footer);
  const contentWidth = Math.max(140, width - left - right);
  const usable = Math.max(3 * 44, bottom - top - gap * (rows - 1));
  const total = weights.reduce((a, b) => a + Math.max(0.01, b), 0);
  const rects: Rect[] = [];
  let y = top;
  for (let i = 0; i < rows; i += 1) {
    const h = Math.max(44, Math.round((usable * Math.max(0.01, weights[i] as number)) / total));
    rects.push([left, y, contentWidth, h]);
    y += h + gap;
  }
  return { rects, left, right, contentWidth, top, bottom, gap };
}

/**
 * 顶部角标预留高度（与 `model.ts::headerBand` 同一个口径，这里复制一份是为了让
 * `scaffold.ts` 不依赖帧相关的模块；常量来自 `BADGE_LAYOUT`）。
 */
function headerBandOf(width: number): number {
  const cols = width >= 2 * 344 + 24 ? 2 : 1;
  const rows = Math.ceil(8 / cols);
  return 34 + rows * 18 + 8;
}

/**
 * 一段说明文字 → 若干 `text` 层，**一行一个**。
 *
 * `top` / `bottom` 是文本块的净空（通常是面板内缩几个像素后的 `[y+4, y+h−3]`），
 * **每一行的盒子都必须落在净空之内**——这条约束不是装饰：坐标轴的刻度标签带在
 * `bottom+3 .. bottom+15`，一行文本只要越过 `bottom` 就会压住它，`validateScene`
 * 的规则 8 会当场拒绝整张 Scene（"文本压住坐标轴装饰"）。
 *
 * `anchor="top"` 从 `top` 往下排；`anchor="bottom"` 让最后一行贴 `bottom`。
 * 行数放不下时**截断并写明"另有 N 行"**——叠字等于看不见，宁可少显示几行，
 * 也不让浏览器与光栅器画出互相压住的文本。
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
  const size = 11;
  const rowH = 15;
  let pieces = lines.flatMap((body) => wrapUnits(body, budget));
  if (bottom >= top) {
    // 一行的盒子是 [y, y+rowH]（校验器口径），所以"最后一行不越界"= y_last + rowH ≤ bottom。
    const maxRows = Math.max(1, Math.floor((bottom - top) / rowH));
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  const y =
    anchor === "top"
      ? top
      : Math.min(Math.max(top, bottom - (pieces.length - 1) * rowH), Math.max(top, bottom - rowH));
  return pieces.map((body, i) => text(x, y + i * rowH, body, { role: "note", size, color, maxWidth: budget * 7 }));
}

/**
 * 按显示宽度单位折行（CJK=2、ASCII=1，口径同 `model.ts::textWidthUnits`）。
 *
 * 先按空白（含全角空格）切"词"再往行里塞：`Λ=6.601e6`、`k=12/12` 这类字段不会被从中间
 * 劈开（劈开在光栅器里读起来像两个数）；单个词比预算还长时才硬切。
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

/** 帧 `strings.note` 里的 `**强调**` 是两个渲染器都不认的 Markdown：显示前去掉标记，文字本身照抄。 */
export function plain(body: string | undefined): string | undefined {
  if (body === undefined) return undefined;
  return body.split("**").join("");
}

/** manifest 里某个轴的 `title`（缺省才用回落文案；文案的真源在 manifest）。 */
export function axisTitleOf(
  spec: { axes?: readonly { id: string; title?: string }[] } | undefined,
  axisId: string,
  fallback: string,
): string {
  return spec?.axes?.find((a) => a.id === axisId)?.title ?? fallback;
}

/** manifest 里某个轴的 `ylabel`。 */
export function axisYlabelOf(
  spec: { axes?: readonly { id: string; ylabel?: string }[] } | undefined,
  axisId: string,
  fallback: string,
): string {
  return spec?.axes?.find((a) => a.id === axisId)?.ylabel ?? fallback;
}

/** 数字格式化（与 `axes.ts::formatNumber` 同一实现，这里只为少写一层 import）。 */
export function num(v: number, sig = 4): string {
  return formatNumber(v, sig);
}

/**
 * 在**严格递增**的 `xs` 里找最接近 `value` 的下标；找不到返回 `-1`。
 *
 * 只用于把「当前参数值」对准帧里的采样格点（例如 `meta.n_mark` → `n_axis` 的下标），
 * 不做任何插值：宁可返回 −1 让调用方降级，也不在显示层造一个数。
 */
export function nearestIndex(xs: ArrayLike<number>, value: number): number {
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let i = 0; i < xs.length; i += 1) {
    const x = xs[i] as number;
    if (!Number.isFinite(x)) continue;
    const d = Math.abs(x - value);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

/** 一维数组里最小/最大**有限**值（空数组或全非有限时 undefined）。 */
export function minMax(a: ArrayLike<number>): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : undefined;
}
