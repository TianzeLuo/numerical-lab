/**
 * 4.1 章四个 model 共用的**排版 / 文本 / 取值**工具（纯几何与纯文本，不碰物理量）。
 *
 * 为什么单独一个文件：`textBlock` / `wrapUnits` / `unitBudget` 是"模型自己折行"的同一套实现
 * （`render2d.ts::drawText` 不拆 `\n`，而 `scene.ts::textBoxOf` 按 `maxWidth` 折行估盒——
 * 模型按 15px 排行、校验器按 `size+4` 估盒，两边必须是同一把尺子）。1.2 / 1.5 / 3.1 章
 * 各有一份同口径的私有实现；本章照抄这套口径而**不去 import 别的章节的私有模块**：
 * 章节前端要保持可以单独编译、单独删除。
 *
 * 与 3.1 那版的差别只有一处，是本章版式的硬需要：
 *
 * {@link contentBox} 的底边同时受 `stackedPanels` 的缺省 footer（102px）与**底部 pitfalls
 * 面板的实际顶边**约束。渲染层按 `scene.pitfalls` 现画那块不透明面板（字号 base−1、行高
 * `size+4`、左右各 12px、底边距 10px），字数一多它就长得比 footer 预留的还高：960 宽下
 * 本章四条 pitfalls 都占 4 行（顶边 y=522），而窄画布 480 宽下会涨到 9 行（顶边 y=455）。
 * 照搬 `stackedPanels` 的固定 footer，窄画布上的面板会有一截被那块面板盖住——所以这里
 * 把 `pitfallsBand` 的顶边也算进来，取两者更严的那个。
 *
 * 这些函数只做三件事：按共享尺子（`textWidthUnits`：CJK=2、ASCII=1）折行、把折好的行
 * 放进 `[top, bottom]` 的净空里、从 manifest 里取声明过的轴元信息。**不算任何物理量。**
 */

import type { AxisSpecJson, VizSpecJson } from "../../shared/src/manifest.js";
import type { Frame } from "../../shared/src/frame.js";
import { BADGE_UNIT_PX, headerBand, textWidthUnits, type Rect } from "../../shared/src/model.js";
import { text, type TextSpec } from "../../shared/src/scene.js";

/** 面板内说明文字的字号（与 0.0 / 1.2 / 1.5 / 3.1 章一致）。 */
export const TEXT_SIZE = 11;
/** 行距：`textBoxOf` 按 `size + 4 = 15` 估盒，模型自己排行也必须用同一个数。 */
export const TEXT_ROW_H = TEXT_SIZE + 4;
/**
 * 竖直堆叠的两块面板之间的固定间距。
 *
 * 46 = 上面板两条底部装饰带（刻度 15 + 轴名 12 = 28，见 `AXIS_DECO.bottom`）
 * + 下方面板轴标题带（`top−18..top−2`，16px）+ 2px 余量。本模块的
 * {@link stackRows} 恒让出这一条，所以层与层之间**永远不许**再塞文本。
 */
export const PANEL_GAP = 46;
/** 上方面板没有轴名（只留 15px 刻度带）时可以用的小间距。 */
export const TIGHT_GAP = 22;

/**
 * 「两排面板 + 底部通栏文字」的版式（4.1 章四个 viz 共用）。
 *
 * 为什么是通栏而不是"每块面板下面挂一段"：4.1 章的数字块要同时报 r / θ / dt / n_steps /
 * 实测谱半径 / 解析值 / 模态计数 / 误差……按 326px 的侧栏折行，每行只有 40 个单位
 * （≈20 个汉字），二十几条事实会被折成三十几行，`textBlock` 只能砍掉大半——**被砍掉的
 * 恰好是"实测 vs 解析"这类关键对拍**。通栏一行 124 个单位，同样的内容 4 行就装完了。
 *
 * 版式自下而上算，保证文字一定装得下：文字占 `rows × 15` 贴住 `box.bottom`；下排面板
 * 让开自己的刻度带（15 + 3 + 2 = 20）；上排面板占据剩余高度（下排的轴标题带 `botY−18`
 * 已经由间距让开）。
 */
export interface BandLayout {
  /** 上排面板的顶（= `box.top`）。 */
  topY: number;
  /** 上排面板的高。 */
  topH: number;
  /** 下排面板的顶。 */
  botY: number;
  /** 下排面板的高。 */
  botH: number;
  /** 通栏文字的顶（第一行的盒子顶）。 */
  textTop: number;
  /** 通栏文字实际可用行数。 */
  rows: number;
  /** 每行的单位预算（`unitBudget(box.width)`）。 */
  budget: number;
}

/**
 * 见 {@link BandLayout}。`rows` 是文字块想要的行数；空间不足时按实际可用行数算。
 * `botH = 0` 表示**没有下排面板**，此时上排直接贴着文字的刻度带排（不留轴标题带）。
 */
export function bandLayout(box: ContentBox, opts: { botH: number; rows: number; gap?: number }): BandLayout {
  const gap = opts.gap ?? TIGHT_GAP;
  const textH = Math.max(TEXT_ROW_H, opts.rows * TEXT_ROW_H);
  const textTop = box.bottom - textH;
  if (opts.botH <= 0) {
    const topH = Math.max(70, textTop - 24 - box.top);
    return { topY: box.top, topH, botY: box.top + topH, botH: 0, textTop, rows: Math.max(1, Math.floor(textH / TEXT_ROW_H)), budget: unitBudget(box.width) };
  }
  const botY = textTop - 20 - opts.botH;
  const topH = Math.max(70, botY - gap - box.top);
  return {
    topY: box.top,
    topH,
    botY,
    botH: opts.botH,
    textTop,
    rows: Math.max(1, Math.floor(textH / TEXT_ROW_H)),
    budget: unitBudget(box.width),
  };
}

/** 内容区（四块面板 + 文本块都从这里排）。 */
export interface ContentBox {
  left: number;
  right: number;
  top: number;
  bottom: number;
  /** `right − left`。 */
  width: number;
  /** `bottom − top`。 */
  height: number;
}

/**
 * 内容区：左右留白 64 / 24（与 `stackedPanels` 同口径），顶端让开角标区。
 *
 * 底边取三者更严的那个：`height − 102`（`stackedPanels` 的缺省 footer）、
 * pitfalls 面板顶边 − 8、以及至少留 120px 给面板。`pitfallsTop` 传 `null`
 * 表示这块面板不存在（帧的 pitfalls 为空），此时只用 footer。
 */
export function contentBox(width: number, height: number, pitfallsTop: number | null): ContentBox {
  const left = 64;
  const right = Math.max(left + 120, width - 24);
  const top = headerBand(width) + 8;
  const footerBottom = height - 102;
  const limit = (pitfallsTop ?? height) - 8;
  const bottom = Math.max(top + 120, Math.min(footerBottom, limit));
  return { left, right, top, bottom, width: right - left, height: bottom - top };
}

/**
 * 把内容区按权重切成竖直堆叠的矩形（自下而上留 `PANEL_GAP`）。
 *
 * 权重只是比例；每块都不小于 `minH`，切完仍然贴合 `box.top` / `box.bottom`。
 */
export function stackRows(box: ContentBox, weights: readonly number[], minH = 64, gap = PANEL_GAP): Rect[] {
  const n = weights.length;
  if (n === 0) return [];
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  const usable = Math.max(n * minH, box.height - gap * (n - 1));
  // 先按权重分，再把低于 minH 的抬到 minH（差额按比例从高于 minH 的那几块里扣）。
  let raw = weights.map((w) => (usable * w) / total);
  for (let iter = 0; iter < 4; iter += 1) {
    const deficit = raw.reduce((a, v) => a + Math.max(0, minH - v), 0);
    if (deficit <= 1e-9) break;
    const surplus = raw.reduce((a, v) => a + Math.max(0, v - minH), 0);
    if (surplus <= 1e-9) break;
    raw = raw.map((v) => (v < minH ? minH : v - (Math.max(0, v - minH) / surplus) * deficit));
  }
  const heights = raw.map((v) => Math.max(1, Math.round(v)));
  // 取整漂移落在最大的一块上，保证首尾仍然贴合 box.top / box.bottom。
  let drift = usable - heights.reduce((a, b) => a + b, 0);
  let biggest = 0;
  for (let i = 1; i < n; i += 1) if ((heights[i] as number) > (heights[biggest] as number)) biggest = i;
  heights[biggest] = (heights[biggest] as number) + drift;
  drift = 0;
  const rects: Rect[] = [];
  let y = box.top;
  for (let i = 0; i < n; i += 1) {
    rects.push([box.left, y, box.width, heights[i] as number]);
    y += (heights[i] as number) + gap;
  }
  return rects;
}

/**
 * 左右分栏：左列宽 `share`，右列贴右边界，中间留 `columnGap`。
 *
 * `columnGap = 58` 是右列自己的 y 刻度带（35）+ 轴名带（14）+ 余量，与
 * `columnsLayout` 的缺省值一致。
 */
export function splitColumns(box: ContentBox, share = 0.56, columnGap = 58): { left: ContentBox; right: ContentBox } {
  const leftW = Math.max(160, Math.round(box.width * share));
  const rightX = Math.min(box.left + leftW + columnGap, box.right - 140);
  return {
    left: { ...box, right: box.left + leftW, width: leftW },
    right: { ...box, left: rightX, width: box.right - rightX },
  };
}

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
  // 折行点：空白、顿号/逗号、句读、右括号之后（CJK 长句没有空格，只在空白处折会把
  // 「t_actual=0.010125」这种整体从中间劈开）。
  for (const word of body.split(/(?<=[\s　、，,。；;：:）)])/)) {
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
    line += piece;
    used += pieceUsed;
  }
  push();
  return out;
}

/**
 * 一段说明 → 若干 `text` 层，**一行一个**（渲染层不拆 `\n`）。
 *
 * 先折行再做一次"短行拼接"：把装得下的相邻短句并进同一行，否则一句 20 单位的短句
 * 会独占一行、白白吃掉一行高度。超出 `[top, bottom]` 净空的行**以及被丢弃的行数**
 * 都如实写在最后一行里（防误导：不许静默丢说明）。
 *
 * `anchor="bottom"` 时整块贴着 `bottom` 往上排（用于"先保证底部提示可见"的场合）。
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
  let pieces: string[] = [];
  for (const body of lines) {
    for (const part of wrapUnits(body, budget)) {
      const last = pieces[pieces.length - 1];
      if (last !== undefined && textWidthUnits(last) + 1 + textWidthUnits(part) <= budget) {
        pieces[pieces.length - 1] = `${last}　${part}`;
      } else {
        pieces.push(part);
      }
    }
  }
  if (bottom > top) {
    // 行盒高度按 `size + 4` 估（`textBoxOf` 口径）：最后一行必须整体落在 `bottom` 之上。
    const maxRows = Math.max(1, Math.floor((bottom - top) / TEXT_ROW_H));
    if (pieces.length > maxRows) {
      const kept = Math.max(0, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - (pieces.length - 1) * TEXT_ROW_H);
  return pieces.map((body, i) =>
    text(x, y + i * TEXT_ROW_H, body, { role: "note", size: TEXT_SIZE, color, maxWidth: budget * 7 }),
  );
}

/** 从 manifest 的 viz 里取某个轴的声明（模型的轴 id 与它一一对应）。 */
export function axisSpecOf(spec: VizSpecJson | undefined, id: string): AxisSpecJson | undefined {
  return spec?.axes.find((a) => a.id === id);
}

/**
 * 把帧里给的**长枚举串**裁到单位预算内（只用于显示，不改帧）。
 *
 * 例：4.1 的 `initial=Σ_{k=1..8} sin(kπx)/k（多模态，前 8 个模态…）`、`bc` 的显示名会与
 * 枚举值叠成 `neumann（Neumann（绝缘））`。这类串长度由后端决定，前端排版不能赌它短——
 * 裁掉时补 `…`，并**只裁显示**：帧、meta、数组一律不动（P8）。
 */
export function clip(text: string, maxUnits: number): string {
  if (textWidthUnits(text) <= maxUnits) return text;
  let out = "";
  let used = 0;
  for (const ch of text) {
    const w = textWidthUnits(ch);
    if (used + w > maxUnits - 1) break;
    out += ch;
    used += w;
  }
  return `${out}…`;
}

/** 轴标题：manifest 声明优先，缺了用回落值（不自己编一个像样的标题）。 */
export function axisTitleOf(spec: VizSpecJson | undefined, id: string, fallback: string): string {
  return axisSpecOf(spec, id)?.title ?? fallback;
}

/**
 * 轴标题（**按面板宽度裁过**）。
 *
 * 渲染层把轴标题画在面板左上角、**不裁剪**也不换行（`raster.py` 与 `render2d` 同口径），
 * 所以"manifest 里的那句长标题"在窄面板上会横穿到隔壁面板的标题上——4.1 章的 manifest
 * 轴标题都是完整句子（例：「解曲线（窗口固定 ±1.5：发散时冲出去，而不是被自动缩放藏起来）」），
 * 直接照抄就当场撞车。这里的规矩：**装得下就用声明的那句，装不下就用模型自己的短标题**，
 * 长口径留给面板文字与 pitfalls（信息没丢，只是换了地方）。
 */
export function axisTitleFitted(
  spec: VizSpecJson | undefined,
  id: string,
  fallback: string,
  panelWidthPx: number,
): string {
  const budget = Math.max(8, Math.floor((panelWidthPx - 8) / BADGE_UNIT_PX));
  const declared = axisSpecOf(spec, id)?.title;
  const chosen = declared === undefined || declared === "" ? fallback : declared;
  // 声明装不下就用模型自己的短标题；短标题仍装不下（窄面板 / 窄画布）就裁它——
  // 轴标题是**居中**画的（`raster.py::text`，`rx + rw/2`），裁掉比横穿隔壁面板强。
  if (textWidthUnits(chosen) <= budget) return chosen;
  if (chosen !== fallback && textWidthUnits(fallback) <= budget) return fallback;
  return clip(chosen, budget);
}

/**
 * 曲线标签现在只进网页的 HTML 图例：显示层会把 label 从图层里取走、不在画布末端绘制，
 * 所以不必再按剩余像素决定要不要保留。
 */
export function endLabel(
  _xLast: number,
  _xlim: readonly [number, number],
  _panelW: number,
  label: string,
  _opts: { log?: boolean } = {},
): string {
  return label;
}

/** {@link endLabel} 的数组版：保留签名，统一返回标签。 */
export function endLabelOf(
  _x: ArrayLike<number>,
  _xlim: readonly [number, number],
  _panelW: number,
  label: string,
  _opts: { log?: boolean } = {},
): string {
  return label;
}

/** 轴名（ylabel）：manifest 声明优先。 */
export function axisYlabelOf(spec: VizSpecJson | undefined, id: string, fallback: string): string {
  return axisSpecOf(spec, id)?.ylabel ?? fallback;
}

/**
 * 轴的 `autoscaled` **一律从 manifest 取**（规则 1/4 的契约源在 Python 侧）。
 *
 * 模型里再写一个字面量就等于把同一条声明抄了两遍——两处迟早会漂移，
 * 而"自适应轴没被标注"正是本项目最警惕的那类骗术。
 */
export function axisAutoscaledOf(spec: VizSpecJson | undefined, id: string, fallback: boolean): boolean {
  return axisSpecOf(spec, id)?.autoscaled ?? fallback;
}

/** 一条横贯面板的参考线 `[x0, x1] × [y, y]`（残差水平线、守恒基线、地板线都用它）。 */
export function hLine(x0: number, x1: number, y: number): { x: number[]; y: number[] } {
  return { x: [x0, x1], y: [y, y] };
}

/** 一条竖线 `[x, x] × [y0, y1]`（当前时刻、符号翻转波数都用它）。 */
export function vLine(x: number, y0: number, y1: number): { x: number[]; y: number[] } {
  return { x: [x, x], y: [y0, y1] };
}

/** 逐元素取绝对值（显示几何，不是物理量）。非有限值原样保留，交给分段逻辑断开。 */
export function absolute(a: ArrayLike<number>): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i += 1) out[i] = Math.abs(a[i] as number);
  return out;
}

/**
 * 帧数组自带的 `label`（后端在 `fr.add(..., label=...)` 里写的那句）。
 *
 * 图例文字用它而不是模型自己另写一遍：那张表是**后端**对这条曲线的定义
 * （例：`|G(φ)|（btcs，对照）`），前端重写就等于把契约抄了第二遍。
 */
export function arrayLabelOf(frame: Frame, name: string, fallback: string): string {
  const label = frame.specs.find((s) => s.name === name)?.label;
  return label !== undefined && label !== "" ? label : fallback;
}

/** 有限的 0 的个数（对数轴上"画不出"，与 NaN 不同，要分开报数）。 */
export function countZero(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if (a[i] === 0) n += 1;
  return n;
}

/** 取值 > 0.5 的元素个数（帧里的 0/1 掩码一律这么数）。 */
export function countMask(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] as number) > 0.5) n += 1;
  return n;
}
