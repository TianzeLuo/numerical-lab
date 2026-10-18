/**
 * 2.5 章（傅里叶分析与 FFT）五个 model 共用的排版与读帧工具。
 *
 * 为什么单独一个文件（1.2 章是每个 model 各带一份 `textBlock`/`wrapUnits`）：
 * 本章五个 viz 的排版需求高度一致——两三个面板 + 面板内数值块 + 一处时间轴夹取说明——
 * 共用一份可以让「折行口径 / 角标槽位 / 时间轴语义 / dB 显示变换」不可能在五个文件里各说各话。
 *
 * 纪律（与 `docs/POLICY.md` 对 model 层的要求一致）：
 *  - **纯函数**：无 DOM / 无 IO / 不用 `Date` / 不用 `Math.random`（渲染结果必须可复现）；
 *  - **只读帧**：这里不做任何数值方法的重算（P8）。最"重"的操作是：把帧数组折成
 *    「显示宽度单位」、取有限值的极值、以及在**帧自己给的样本**上做线性插值找穿越点
 *    （画 −3 dB 主瓣宽度要用；插值是显示几何，不是重算物理量）；
 *  - 角标只放"这是什么"（索引级标签），长句子一律进 `textBlock` 或 `frame.pitfalls`。
 */

import { formatNumber, formatSci } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import type { AxisSpecJson, VizSpecJson } from "../../shared/src/manifest.js";
import { clamp, headerBand, metaNumber, stackedPanels, type ModelOptions } from "../../shared/src/model.js";
import { pitfallsBand, text, textWidthUnits, type Scene, type TextSpec } from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

export const CHAPTER = "2.5";

/**
 * 固定角标槽位（见 `model.ts::BADGE_LAYOUT`，两列画布每行 2 枚、共 4 行）。
 * 0 来源 / 1 主轴固定 / 2 副轴自适应 / 3 临界 / 4 状态 / 5 第二临界或预览 /
 * 6 溢出 / 7 降级。槽位固定是为了"少一枚角标时其余不挪位"（时间轴拖动时角标不抖）。
 */
export const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  status: 4,
  critical2: 5,
  overflow: 6,
  degraded: 7,
} as const;

/** 视觉回归/像素比对的默认画布（与 `tools/raster.py`、`dump_scenes.ts` 一致）。 */
export const BASE_FONT = 13;

/** manifest 里这个 viz 的 spec；找不到直接抛（生成物过期是配置错误，不是运行时状态）。 */
export function vizSpec(id: string): VizSpecJson {
  const spec = manifest.vizzes.find((v) => v.id === id);
  if (!spec) throw new Error(`manifest 里没有 ${CHAPTER}/${id}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

/** 标题：文案取 manifest（单一真源），取不到才用调用方的回落值。 */
export function specTitle(id: string, fallback: string): string {
  return vizSpec(id).title || fallback;
}

/** 轴标题（manifest 的 `axes[].title`）。 */
export function axisTitle(id: string, axisId: string, fallback: string): string {
  return vizSpec(id).axes.find((a) => a.id === axisId)?.title ?? fallback;
}

/** 轴名（manifest 的 `axes[].ylabel`）。 */
export function axisYlabel(id: string, axisId: string, fallback: string): string {
  return vizSpec(id).axes.find((a) => a.id === axisId)?.ylabel ?? fallback;
}

/** manifest 声明该轴是否自适应（测试用它钉"角标与 manifest 一致"）。 */
export function axisAutoscaled(id: string, axisId: string): boolean {
  return vizSpec(id).axes.find((a) => a.id === axisId)?.autoscaled ?? false;
}

/** manifest 里某个轴的原始声明（缺了返回 undefined，调用方自己决定回落）。 */
export function axisSpec(id: string, axisId: string): AxisSpecJson | undefined {
  return vizSpec(id).axes.find((a) => a.id === axisId);
}

// --------------------------------------------------------------------------- //
// 排版（显示几何，不是物理）
// --------------------------------------------------------------------------- //

/** 面板宽度 → 显示宽度单位预算（size 11 时 1 单位 ≈ 7px；CJK=2、ASCII=1）。 */
export function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/**
 * 按显示宽度单位折行（CJK=2、ASCII=1；口径同 `scene.ts::textWidthUnits`）。
 *
 * 先按空白（含全角空格）切"词"再往行里塞：`df=0.003906`、`max|Δ|=0.2508` 这类字段
 * 不会被从中间劈开（劈开在光栅器里读起来像两个数）。单个词比预算还长时才硬切。
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
    // 超长词：硬切成若干段
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
 * 一段说明文字 → 若干 `text` 层，**一行一个**。
 *
 * 为什么不交给渲染层的 `maxWidth` 自动折行：两个渲染器折出来的续行间距与模型排的
 * 行距口径不同，续行会压到下一行文字上。这里先折好（每行 ≤ budget 个单位），
 * 两个渲染器都不会再折。
 *
 * `anchor = "bottom"`：整块的**底边**落在 `bottom`；行数超过 `top..bottom` 的净空时
 * **截断并说明**（叠字等于看不见）。
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
  const size = 11;
  const rowH = 15;
  let pieces = lines.flatMap((body) => wrapUnits(body, budget));
  if (bottom > top) {
    // 校验器按 `行数 × (size+4)` 估盒（`scene.ts::textBoxOf`），"放得下"的判据是
    // `top + 行数 × rowH ≤ bottom`（与 2.3 章同一口径）。
    const maxRows = Math.max(1, Math.floor((bottom - top) / rowH));
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - pieces.length * rowH);
  return pieces.map((body, i) => text(x, y + i * rowH, body, { role: "note", size, color, maxWidth: budget * 7 }));
}

/**
 * 与 {@link textBlock} 同一口径，但**贴面板右边界右对齐**（`align: "right"`）。
 *
 * 为什么需要它：谱面板左上角正是主瓣/峰值所在的地方，把说明文字放在那儿会压住曲线；
 * 而同一面板的右上角（高频端）通常已经衰减到地板，是空的。
 * 盒子口径由 `scene.ts::textBoxOf` 处理（`align="right"` 时盒子是 `[x−w, y, w, h]`），
 * 所以规则 8 的文本重叠校验照样管得住它。
 */
export function textBlockRight(
  right: number,
  top: number,
  bottom: number,
  lines: readonly string[],
  color: string,
  budget: number,
  anchor: "top" | "bottom",
): TextSpec[] {
  const size = 11;
  const rowH = 15;
  let pieces = lines.flatMap((body) => wrapUnits(body, budget));
  if (bottom > top) {
    const maxRows = Math.max(1, Math.floor((bottom - top) / rowH));
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - pieces.length * rowH);
  return pieces.map((body, i) => text(right, y + i * rowH, body, { role: "note", size, color, align: "right", maxWidth: budget * 7 }));
}

/** 紧凑数字（4 位有效数字）：面板文本里的默认口径。 */
export function fmt(v: number, sig = 4): string {
  return formatNumber(v, sig);
}

/** 可能缺失的量：`undefined` → `"?"`（**绝不**拿 0 或 NaN 顶替）。 */
export function fmtOpt(v: number | undefined, sig = 4): string {
  return v === undefined ? "?" : formatNumber(v, sig);
}

/** 科学计数法（`1.04e-11` 形式）：量级本身是结论时用它（如舍入地板 vs 卷绕误差）。 */
export function sci(v: number, sig = 3): string {
  return formatSci(v, sig);
}

// --------------------------------------------------------------------------- //
// 显示变换（不是重算物理量）
// --------------------------------------------------------------------------- //

/** 逐元素 `|a|`（对数轴的纵轴要求 > 0，取模是显示变换）。 */
export function absOf(a: ArrayLike<number>): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i += 1) out[i] = Math.abs(a[i] as number);
  return out;
}

/**
 * 幅度 → dB（`20log10`，低于地板压到地板）：**显示变换**，与后端 `numcore.transform.fft.db`
 * 同一口径。帧里的 `envelope` 是幅度（`f^-p` 包络），要画在 dB 轴上必须换一次单位——
 * 这不是"在 TS 里重算数值量"（数一个都没变，只是换了纵轴单位），换完在文本里注明。
 */
export function toDb(amp: ArrayLike<number>, floorDb = -140): Float64Array {
  const out = new Float64Array(amp.length);
  for (let i = 0; i < amp.length; i += 1) {
    const v = amp[i] as number;
    out[i] = Number.isFinite(v) && v > 0 ? Math.max(floorDb, 20 * Math.log10(v)) : floorDb;
  }
  return out;
}

/** 有限值的 `[min, max]`；一个有限值都没有时 `undefined`（调用方给退化范围）。 */
export function minMaxOfAll(arrays: readonly ArrayLike<number>[]): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const a of arrays) {
    for (let i = 0; i < a.length; i += 1) {
      const v = a[i] as number;
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : undefined;
}

/** `[lo, hi]` 里包含 0（差值/柱状图要给 0 留位置）。 */
export function withZero(range: [number, number] | undefined): [number, number] {
  if (range === undefined) return [-1, 1];
  return [Math.min(0, range[0]), Math.max(0, range[1])];
}

/** 从帧数组里切出 `[xlo, xhi]` 这段（x 单调不减）；给图层用，免得越界触发溢出角标。 */
export function sliceRange(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  xlo: number,
  xhi: number,
): { x: Float64Array; y: Float64Array } {
  const n = Math.min(x.length, y.length);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const xv = x[i] as number;
    if (xv < xlo || xv > xhi) continue;
    xs.push(xv);
    ys.push(y[i] as number);
  }
  return { x: Float64Array.from(xs), y: Float64Array.from(ys) };
}

/**
 * 在**帧给的样本**上找 `level` 的第一处穿越点（线性插值），从 `from` 出发按 `step` 走。
 * 找不到（走到头都没穿过去）返回 `undefined`——不猜、不外推。
 */
export function crossAt(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  from: number,
  step: 1 | -1,
  level: number,
): number | undefined {
  const n = Math.min(x.length, y.length);
  let i = from;
  while (i + step >= 0 && i + step < n) {
    const y0 = y[i] as number;
    const y1 = y[i + step] as number;
    if (!Number.isFinite(y0) || !Number.isFinite(y1)) return undefined;
    if (y1 <= level) {
      const x0 = x[i] as number;
      const x1 = x[i + step] as number;
      if (y1 === y0) return x1;
      return x0 + ((level - y0) / (y1 - y0)) * (x1 - x0);
    }
    i += step;
  }
  return undefined;
}

/**
 * 「当前级」：`time.mode = "scrub"` 的那一个 viz（`butterfly_stages`）用它把
 * `t ∈ [0, tMax]` 映射到帧 `meta.stage` 并**如实报出夹取**。
 *
 * 语义（见 `server/chapters/2.5.py::_butterfly_stages`）：`stage = round(t·levels)`，
 * 越界夹到 `[0, levels]`；`meta.t_actual = stage/levels`。所以画面永远以帧为准，
 * 请求 `t` 与实到档位不一致时**明说**，不假装 `t` 就是级号。
 */
export interface StageInfo {
  stage: number;
  levels: number;
  /** 请求 `t` 按 `round(t·log2N)` 折出来的档位（**只是请求**，画面以 `stage` 为准）。 */
  requested: number;
  notes: string[];
}

export function currentStage(frame: Frame, opts: ModelOptions, levels: number): StageInfo {
  const notes: string[] = [];
  const asked = Number.isFinite(opts.t) ? opts.t : 0;
  const hi = Math.max(0, Math.round(levels));
  const fromFrame = metaNumber(frame, "stage");
  const requested = clamp(Math.round(asked * hi), 0, hi);
  if (fromFrame === undefined) {
    const stage = requested;
    notes.push(
      `帧 meta 缺 stage：只能按请求 t=${fmt(asked)} 折算到第 ${stage}/${hi} 级显示——` +
        `这一级不一定是帧真正算的那一级（帧没给，不猜）`,
    );
    return { stage, levels: hi, requested, notes };
  }
  const rounded = Math.round(fromFrame);
  const stage = clamp(rounded, 0, hi);
  if (stage !== rounded) notes.push(`帧 meta.stage=${fmt(fromFrame)} 落在 [0, ${hi}] 之外：画面夹到第 ${stage} 级`);
  const tActual = metaNumber(frame, "t_actual");
  const got = tActual ?? stage / Math.max(1, hi);
  if (Math.abs(asked - got) > 0.5 / Math.max(1, hi)) {
    notes.push(
      `请求 t=${fmt(asked, 3)} → 实到第 ${stage}/${hi} 级（t ↦ round(t·log2N)，帧 meta.t_actual=${fmt(got, 3)}）：` +
        `一个 t 区间对应同一级，画面按实到级画`,
    );
  }
  return { stage, levels: hi, requested, notes };
}

/**
 * 从**帧给的一组数**里挑出落在 1-2-5 序列上的那些（`1/2/5 × 10^k`），用作刻度。
 *
 * 为什么不用 `niceTicks`：`span_n`（32…4096）与 `freq` 这些横轴的采样点本身就是
 * "故意挑的不整值"（48、96、192、384…），`niceTicks` 生成的漂亮刻度会与柱/点错开；
 * 直接标"哪些采样点本身是整值"读者才对得上号。
 */
export function oneTwoFive(values: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] as number;
    if (!Number.isFinite(v) || v <= 0) continue;
    const e = Math.floor(Math.log10(v));
    const mant = v / Math.pow(10, e);
    const tol = 1e-9 * Math.max(1, mant);
    if (Math.abs(mant - 1) <= tol || Math.abs(mant - 2) <= tol || Math.abs(mant - 5) <= tol) out.push(v);
  }
  return [...new Set(out)].sort((a, b) => a - b);
}

/**
 * 「整宽上 + 两个并排下」的三面板排版。
 *
 * `stackedPanels` 只给"上下两块"；本章有两处要把**同类的两个差/两个视角并排**
 * （`conv_theorem` 的「循环−线性」与「补零后」、`resolution` 的「能否分辨」与「对拍误差」）。
 * 右侧面板的 y 刻度带（`left−56..left−4`）必须落在左侧面板的右边界之外，
 * 故 `columnGap = 70 > 56`。
 *
 * 上方与下方之间留 **48px**：上方面板的装饰带占 `bottom+3..15`（刻度）与 `bottom+16..28`
 * （轴名），下方面板的**标题**由渲染层画在自己的 rect 之上（`rect[1]−19 .. rect[1]−6`），
 * 所以 `gap ≥ 28 + 19 = 47` 才不让"上面板的轴名"和"下面板的标题"挤在同一行
 * （2.5 章这里是照着 `resolution` 的实拍基线调出来的：gap=30 时两者在中间撞在一起）。
 */
export interface TopWithTwoBelow {
  top: [number, number, number, number];
  leftBottom: [number, number, number, number];
  rightBottom: [number, number, number, number];
}

export function topWithTwoBelow(
  width: number,
  height: number,
  opts: { topShare?: number; footer?: number } = {},
): TopWithTwoBelow {
  const base = stackedPanels(width, height, { footer: opts.footer });
  const gap = 54;
  const colGap = 70;
  const topH = Math.max(80, Math.round((base.bottom - base.top - gap) * (opts.topShare ?? 0.55)));
  const botH = Math.max(60, base.bottom - base.top - gap - topH);
  const colW = Math.max(120, Math.floor((base.contentWidth - colGap) / 2));
  return {
    top: [base.left, base.top, base.contentWidth, topH],
    leftBottom: [base.left, base.top + topH + gap, colW, botH],
    rightBottom: [base.left + colW + colGap, base.top + topH + gap, colW, botH],
  };
}

/**
 * 底部「关键数字」横条：面板只留**一行**身份标签，长解释全部下移到 pitfalls 面板
 * 之上的这条横条里（2~3 行，整幅宽），这样面板里的曲线不再被文字压住。
 *
 * 为什么必须**先建壳再排版**：这条横条的上界是底部 pitfalls 面板的顶边（规则 7 的
 * 面板由渲染层按 `scene.pitfalls` 现画，高度随文案行数变），只有 `pitfallsBand(scene)`
 * 能算出它。`footer` 回填给 `stackedPanels`，让**最下面那块面板的 x 轴名（28px）**
 * 正好落在横条之上——否则轴名装饰带会伸进横条，规则 8 当场报违规。
 */
export interface BottomStrip {
  /** 横条顶边（第一行文字的 y）。 */
  stripTop: number;
  /** 横条的底边（再往下就是 pitfalls 面板，不许越线）。 */
  limit: number;
  /** 回填给 `stackedPanels` / `columnsLayout` 的底部留白。 */
  footer: number;
}

export function bottomStrip(scene: Scene, width: number, height: number, rows = 3): BottomStrip {
  const panelTop = pitfallsBand(scene)?.[1] ?? height - 60;
  const stripTop = Math.max(headerBand(width) + 150, panelTop - 4 - rows * 15);
  const footer = Math.max(70, height - stripTop + 30);
  return { stripTop, limit: panelTop - 4, footer };
}
