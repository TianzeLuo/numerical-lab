/**
 * 2.4 章（数值积分）四个 model 共用的排版、读帧与显示几何工具。
 *
 * 为什么单独一个文件（与 2.3 章 `common.ts` 同一条理由）：本章四个 viz 的排版需求高度一致
 * ——两列面板（一列主图、一列误差/阶/尾巴）、面板内数值块、时间轴夹取说明、"两条理论线"——
 * 共用一份可以让「折行口径 / 角标槽位 / 时间轴语义 / 理论参考线的锚点约定」不可能在四个
 * 文件里各说各话。
 *
 * 纪律（`docs/POLICY.md` P8/P9 与 `docs/AUTHORING.md` §2.3）：
 *  - **纯函数**：无 DOM / 无 IO / 不用 `Date` / 不用 `Math.random`（渲染必须可复现）；
 *  - **不重算数值量**：所有物理量（误差、阶、条件数、尾巴）都来自帧数组/meta；
 *    这里最"重"的三件事都是**显示几何**：折显示宽度单位、取有限值极值、
 *    按帧给的幂律指数画理论参考线（{@link powerLine}，斜率与锚点都来自帧）。
 *  - 角标只放"这是什么"（索引级标签），长句子一律进 `textBlock` 或 `frame.pitfalls`。
 */

import { formatNumber, formatSci, niceTicks } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import type { AxisSpecJson, VizSpecJson } from "../../shared/src/manifest.js";
import { clamp, headerBand, metaNumber, type ModelOptions, type Rect } from "../../shared/src/model.js";
import { text, textWidthUnits, type TextSpec } from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

export const CHAPTER = "2.4";

/**
 * 固定角标槽位（见 `model.ts::BADGE_LAYOUT`，两列画布每行 2 枚、共 4 行）。
 * 本章四个 viz 共用同一套槽位：0 来源 / 1 主轴固定 / 2 副轴自适应 / 3 临界 /
 * 4 状态 / 5 前端预览（仅 `backend+preview`）/ 6 溢出 / 7 降级。
 * 槽位固定是为了"少一枚角标时其余不挪位"（时间轴拖动时角标不抖）。
 */
export const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  status: 4,
  preview: 5,
  overflow: 6,
  degraded: 7,
} as const;

/** manifest 里这个 viz 的 spec；找不到直接抛（生成物过期是配置错误，不是运行时状态）。 */
export function vizSpec(id: string): VizSpecJson {
  const spec = manifest.vizzes.find((v) => v.id === id);
  if (!spec) throw new Error(`manifest 里没有 ${id}（viz/chapters/2.4/manifest.gen.ts 需要重新生成？）`);
  return spec;
}

/** 标题：文案取 manifest（单一真源），取不到才用调用方的回落值。 */
export function specTitle(id: string, fallback: string): string {
  return vizSpec(id).title || fallback;
}

function axisSpec(id: string, axisId: string): AxisSpecJson | undefined {
  return vizSpec(id).axes.find((a) => a.id === axisId);
}

/** 轴标题（manifest 的 `axes[].title`）。 */
export function axisTitle(id: string, axisId: string, fallback: string): string {
  return axisSpec(id, axisId)?.title ?? fallback;
}

/** 轴名（manifest 的 `axes[].ylabel`）。 */
export function axisYlabel(id: string, axisId: string, fallback: string): string {
  return axisSpec(id, axisId)?.ylabel ?? fallback;
}

/**
 * manifest 的 `axes[].scale` 只描述**纵轴**（`docs/PROTOCOLS.md` §5 的字段表里没有
 * `xscale`）；横轴是不是对数由模型按图的内容决定，并写进轴标题。
 */
export function axisYScale(id: string, axisId: string): "linear" | "log" {
  return axisSpec(id, axisId)?.scale ?? "linear";
}

/** manifest 声明的理论条目原文（面板文本里**照登**，不自己改写成数字）。 */
export function theoryOf(id: string): string[] {
  return vizSpec(id).theory ?? [];
}

/** 参数 spec（用于取 `critical` / `options` / `min` / `max`）。 */
export function paramSpec(id: string, name: string) {
  return vizSpec(id).params.find((p) => p.name === name);
}

/* ------------------------------------------------------------------ *
 * 排版：两列面板
 * ------------------------------------------------------------------ */

/** 两列（左主图 / 右副图）面板布局，两列都占满内容高度。 */
export interface SplitLayout {
  left: Rect;
  right: Rect;
  /** 内容区顶端（= 两列的 `y`）。 */
  top: number;
  /** 内容区底端（再往下是底轴的 xlabel 与 pitfalls 面板）。 */
  bottom: number;
  /** 内容区左边界。 */
  contentLeft: number;
  /** 内容区总宽（左边界到右边界）。 */
  contentWidth: number;
  /** 两列之间的净空（右列的 y 刻度与轴名画在这里，所以 ≥ 56+6）。 */
  gap: number;
}

/**
 * 「左列 + 右列」的排版（0.0/1.2 的 `columnsLayout` 是"左列再上下分层"，
 * 本章的四个 viz 都是**两列各自占满高度**：左列看主图、右列看误差/阶/尾巴）。
 *
 * 全部由画布尺寸算出、与帧无关：同一 params 下连续帧的面板位置逐像素相同。
 */
export function splitColumns(
  width: number,
  height: number,
  opts: { share?: number; gap?: number; footer?: number; headerExtra?: number; left?: number; right?: number } = {},
): SplitLayout {
  const left = opts.left ?? 64;
  const right = opts.right ?? 24;
  const footer = opts.footer ?? 8 + 60 + 34;
  const top = headerBand(width) + (opts.headerExtra ?? 8);
  const bottom = Math.max(top + 140, height - footer);
  const contentWidth = Math.max(200, width - left - right);
  const gap = opts.gap ?? 64;
  const leftW = Math.max(160, Math.round((contentWidth - gap) * (opts.share ?? 0.5)));
  const rightW = Math.max(120, contentWidth - leftW - gap);
  return {
    left: [left, top, leftW, bottom - top],
    right: [left + leftW + gap, top, rightW, bottom - top],
    top,
    bottom,
    contentLeft: left,
    contentWidth,
    gap,
  };
}

/** 面板宽度 → 显示宽度单位预算（size 11 时 1 单位 ≈ 7px；CJK=2、ASCII=1）。 */
export function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/**
 * 按显示宽度单位折行（口径同 `scene.ts::textWidthUnits`，与两个渲染层、校验器同一把尺子）。
 *
 * 先按空白（含全角空格）切词再往行里塞：`4.294e-05`、`min(p,α+1)=1.50` 这类字段不会被
 * 从中间劈开（劈开在光栅器里读起来像两个数）。单个词比预算还长时才硬切。
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
  for (const word of body.split(/(?<=[\s　、，；：/])/)) {
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
        // 硬断行时先退到"数字/小数点内部不断行"的位置之后：`Γ=175.463` 被切成
        // `175` + `.463` 两行是真实的排版事故（滑块宽度变化时真出现过）。
        let cut = piece.length;
        while (cut > 1 && isNumberGlue(piece[cut - 1] as string) && isNumberGlue(ch)) cut -= 1;
        if (cut <= 0 || cut >= piece.length) {
          out.push(piece);
          piece = "";
          pieceUsed = 0;
        } else {
          out.push(piece.slice(0, cut));
          piece = piece.slice(cut);
          pieceUsed = textWidthUnits(piece);
        }
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

/** 数字内部的"胶水"字符：跨过它们断行会把一个数拆成两行。 */
function isNumberGlue(ch: string): boolean {
  return /[0-9.]/.test(ch);
}

/**
 * 一段说明文字 → 若干 `text` 层，**一行一个**。
 *
 * 为什么不交给渲染层的 `maxWidth` 自动折行：两个渲染器折出来的续行间距与模型排的
 * 行距不一致（`render2d.ts::drawText` 与光栅器都用 `size+4`，而这里按 15px 排行），
 * 续行会压到下一行文字上。这里先折好，两个渲染器都不会再折。
 *
 * `anchor = "bottom"`：整块贴着 `bottom` 排；行数超过 `top..bottom` 的净空时**截断并说明**
 * （叠字等于看不见）。校验器按 `行数 × (size+4)` 估盒（`scene.ts::textBoxOf`），
 * 所以块的**底边**落在 `bottom` 才不会探进该面板的 x 刻度带（规则 8）。
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
    const maxRows = Math.max(1, Math.floor((bottom - top) / rowH));
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - pieces.length * rowH);
  return pieces.map((body, i) => text(x, y + i * rowH, body, { role: "note", size, color, maxWidth: budget * 7 }));
}

/* ------------------------------------------------------------------ *
 * 数字与数组
 * ------------------------------------------------------------------ */

/** 紧凑数字（4 位有效数字）：面板文本里的默认口径。 */
export function fmt(v: number, sig = 4): string {
  return formatNumber(v, sig);
}

/** 引用帧文本时去掉 markdown 粗体标记。 */
export function plainNote(body: string | undefined): string | undefined {
  return body === undefined ? undefined : body.replace(/\*\*/g, "");
}

/** 可能缺失的量：`undefined` → `"?"`（**绝不**拿 0 或 NaN 顶替）。 */
export function fmtOpt(v: number | undefined, sig = 4): string {
  return v === undefined ? "?" : formatNumber(v, sig);
}

/** 科学计数法（`1.04×10⁻¹¹` 形式）：量级本身是结论时用它。 */
export function sci(v: number, sig = 3): string {
  return formatSci(v, sig);
}

/** 一组数组里所有有限值的 `[min, max]`；一个都没有时返回 `undefined`（调用方给退化范围）。 */
export function minMaxOfAll(arrays: readonly (ArrayLike<number> | undefined)[]): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const a of arrays) {
    if (a === undefined) continue;
    for (let i = 0; i < a.length; i += 1) {
      const v = a[i] as number;
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : undefined;
}

/** 一组数组里所有**有限且 > 0** 的值的 `[min, max]`（对数轴的定义域）。 */
export function positiveMinMaxOfAll(arrays: readonly (ArrayLike<number> | undefined)[]): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const a of arrays) {
    if (a === undefined) continue;
    for (let i = 0; i < a.length; i += 1) {
      const v = a[i] as number;
      if (!Number.isFinite(v) || v <= 0) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : undefined;
}

/** `min..max` 里包含 0（线性轴上的柱/误差曲线要给 0 留位置）。 */
export function withZero(range: [number, number] | undefined): [number, number] {
  if (range === undefined) return [0, 1];
  return [Math.min(0, range[0]), Math.max(0, range[1])];
}

/** 有限且 ≤ `floor` 的样本数（对数轴上"画不到"的那些点，必须报数）。 */
export function countAtOrBelow(values: ArrayLike<number>, floor: number): number {
  let n = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] as number;
    if (Number.isFinite(v) && v <= floor) n += 1;
  }
  return n;
}

/** 有限值的最大绝对值；没有有限值时 `undefined`（用来报同屏对照的"差有多大"）。 */
export function maxAbs(values: ArrayLike<number>): number | undefined {
  let best = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] as number;
    if (!Number.isFinite(v)) continue;
    const a = Math.abs(v);
    if (a > best) best = a;
  }
  return Number.isFinite(best) ? best : undefined;
}

/** 前 `n` 个元素（复制成 Float64Array）——时间轴"已揭示"的窗口。 */
export function head(values: ArrayLike<number>, n: number): Float64Array {
  const k = Math.max(0, Math.min(values.length, Math.floor(n)));
  const out = new Float64Array(k);
  for (let i = 0; i < k; i += 1) out[i] = values[i] as number;
  return out;
}

/** 把两个数组按有限性成对切出（`segmentedSeries` 之外的另一种"按掩码筛点"）。 */
export function pickPairs(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  ok: (xv: number, yv: number, i: number) => boolean = () => true,
): { x: number[]; y: number[] } {
  const xs: number[] = [];
  const ys: number[] = [];
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i += 1) {
    const xv = x[i] as number;
    const yv = y[i] as number;
    if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
    if (!ok(xv, yv, i)) continue;
    xs.push(xv);
    ys.push(yv);
  }
  return { x: xs, y: ys };
}

/**
 * 幂律参考线 `y = y0·(x/x0)^slope`（**显示参考线**，不是重算数值方法）。
 *
 * 为什么需要它：后端把**公式阶**那条理论线作为数组给了（`theory_curve`），但
 * **光滑性上限 min(p, α+1)** 那条只给了斜率（`meta.slope_theory_smoothness`）——
 * 而本章的第三条陷阱正是"两条理论线必须同时出现"，否则「Simpson 是 4 阶」会被读成
 * 「这里算错了」。斜率与锚点都取自帧，本函数只做逐点取值（`Math.pow`），
 * 不做任何积分/拟合。返回的数组在锚点处恒等于 `y0`。
 */
export function powerLine(x: ArrayLike<number>, x0: number, y0: number, slope: number): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    const xv = x[i] as number;
    out[i] = Number.isFinite(xv) && xv > 0 && x0 > 0 && y0 > 0 ? y0 * Math.pow(xv / x0, slope) : Number.NaN;
  }
  return out;
}

/**
 * 从帧给的一组数里挑出 1-2-5 序列上的那些（`1/2/5 × 10^k`），用作对数横轴的刻度。
 *
 * 为什么不用 `niceTicks`：`n` 只跨一两个十倍程（2…40）时 `niceTicks` 的"十倍程不足"
 * 分支只给得出两三个刻度；而 1-2-5 筛选出来的值**全部来自帧**（不是模型编的坐标）。
 */
export function oneTwoFive(values: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] as number;
    if (!Number.isFinite(v) || v <= 0) continue;
    const mant = v / Math.pow(10, Math.floor(Math.log10(v)));
    const r = Math.round(mant * 1000) / 1000;
    if (Math.abs(r - 1) < 1e-6 || Math.abs(r - 2) < 1e-6 || Math.abs(r - 5) < 1e-6) out.push(v);
  }
  return out;
}

/**
 * 对数横轴的候选刻度：1-2-5 之外再补上首末两端（`n_sub` 是 2 的幂、`nodes` 是奇数，
 * 只靠 1-2-5 会只剩一两个刻度）。**值全部取自帧**，不生成中间值。
 */
export function frameTicks(values: ArrayLike<number>, limit = 8): number[] {
  const kept = oneTwoFive(values);
  const first = values.length > 0 ? (values[0] as number) : Number.NaN;
  const last = values.length > 0 ? (values[values.length - 1] as number) : Number.NaN;
  const out = [...kept];
  for (const v of [first, last]) {
    if (Number.isFinite(v) && v > 0 && !out.includes(v)) out.push(v);
  }
  const sorted = out.filter((v) => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
  return sorted.length >= 2 ? sorted.slice(0, limit) : niceTicks(sorted[0] ?? 1, sorted[sorted.length - 1] ?? 10, 4, { log: true });
}

/** 帧 `strings` 里的说明原文（缺失时返回 `undefined`，调用方负责"缺了什么"的降级文案）。 */
export function noteOf(frame: Frame, key: string): string | undefined {
  const raw = frame.strings[key];
  const body = typeof raw === "string" ? raw.trim() : "";
  return body === "" ? undefined : body;
}

/* ------------------------------------------------------------------ *
 * 时间轴（`scrub`：t 不是时间，是"已揭示到第几步"）
 * ------------------------------------------------------------------ */

/** 时间轴 / 步进信息：**真实档位**与「请求 t 被夹住」的说明（`docs/PROTOCOLS.md` §5）。 */
export interface StepInfo {
  /** 画面真正使用的档位（来自帧 `meta.step`，缺失时才按请求 t 夹取）。 */
  step: number | undefined;
  /** 需要出现在图上的说明（每条都点了名：缺什么 / 夹到了哪里）。 */
  notes: string[];
}

/**
 * 把 `t` 解析成**实际档位**：本章四个 viz 的 `t` 都不是时间，而是"已揭示的面板数 /
 * 多项式次数 / 层数 / 节点数"（后端 `_clamp_step` 已经夹过一次）；模型这一侧只做两件事：
 * 以帧 `meta.step` 为准、并在请求值与实到值不同时**明说**（不假装 t 就是索引）。
 */
export function currentStep(
  frame: Frame,
  opts: ModelOptions,
  cfg: { lo: number; hi: number; label: string },
): StepInfo {
  const notes: string[] = [];
  const asked = Number.isFinite(opts.t) ? opts.t : cfg.lo;
  const fromFrame = metaNumber(frame, "step");
  if (fromFrame === undefined) {
    const step = clamp(Math.round(asked), cfg.lo, cfg.hi);
    notes.push(
      `帧 meta 缺 step：只能按请求 t=${fmt(asked)} 夹到 ${cfg.label}=${step} 显示——` +
        `这一档不一定是帧真正算的那一档（帧没给，不猜）`,
    );
    return { step, notes };
  }
  const rounded = Math.round(fromFrame);
  const step = clamp(rounded, cfg.lo, cfg.hi);
  if (step !== rounded) notes.push(`帧 meta.step=${fmt(fromFrame)} 落在 [${cfg.lo}, ${cfg.hi}] 之外：画面夹到 ${step}`);
  if (Math.abs(asked - step) > 0.5) {
    notes.push(
      `请求 t=${fmt(asked)} ≠ 实到 ${cfg.label}=${fmt(step)}（帧 meta.step）：` +
        `这一档由后端夹取，画面按实到档位画，不假装 t 就是索引`,
    );
  }
  return { step, notes };
}
