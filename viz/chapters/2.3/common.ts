/**
 * 2.3 章（正交多项式与函数逼近）四个 model 共用的排版与读帧工具。
 *
 * 为什么单独一个文件（1.2 章是每个 model 各带一份 `textBlock`/`wrapUnits`）：
 * 本章四个 viz 的排版需求高度一致——两三个面板 + 面板内数值块 + 时间轴夹取说明——
 * 共用一份可以让「折行口径 / 角标槽位 / 时间轴语义」不可能在四个文件里各说各话。
 *
 * 纪律（与 `docs/POLICY.md` 对 model 层的要求一致）：
 *  - **纯函数**：无 DOM / 无 IO / 不用 `Date` / 不用 `Math.random`（渲染结果必须可复现）；
 *  - **只读帧**：这里不做任何物理计算（P8）。最"重"的操作是把帧数组折成
 *    「显示宽度单位」与取有限值的极值——都是显示几何，不是数值方法；
 *  - 角标只放"这是什么"（索引级标签），长句子一律进 `textBlock` 或 `frame.pitfalls`。
 */

import { formatNumber, formatSci } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import type { AxisSpecJson, VizSpecJson } from "../../shared/src/manifest.js";
import { clamp, metaNumber, type ModelOptions } from "../../shared/src/model.js";
import { text, textWidthUnits, type TextSpec } from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

export const CHAPTER = "2.3";

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
  if (!spec) throw new Error(`manifest 里没有 ${id}（viz/chapters/2.3/manifest.gen.ts 需要重新生成？）`);
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

/** 面板宽度 → 显示宽度单位预算（size 11 时 1 单位 ≈ 7px；CJK=2、ASCII=1）。 */
export function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/**
 * 按显示宽度单位折行（口径同 `scene.ts::textWidthUnits`，与两个渲染层、校验器同一把尺子）。
 *
 * 先按空白（含全角空格）切词再往行里塞：`κ(G)u=21.7`、`1.04e-11` 这类字段不会被从中间劈开
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
 * 一段说明文字 → 若干 `text` 层，**一行一个**。
 *
 * 为什么不交给渲染层的 `maxWidth` 自动折行：两个渲染器折出来的续行间距与模型排的
 * 行距不一致（`render2d.ts::drawText` 用 `size+4`，而模型这边的预算按 15px 排），
 * 续行会压到下一行文字上。这里先折好，两个渲染器都不会再折。
 *
 * `anchor = "bottom"`：整块贴着 `bottom` 排（行数由折行结果决定）；行数超过
 * `top..bottom` 的净空时**截断并说明**（叠字等于看不见）。
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
    // 校验器按 `行数 × (size+4)` 估盒（`scene.ts::textBoxOf`），所以"放得下"的判据是
    // `top + 行数 × rowH ≤ bottom`——不是"最后一行的基线还在 bottom 之上"。
    // 早先按后者算，末行会探进该面板的 x 刻度带（规则 8 当场报违规）。
    const maxRows = Math.max(1, Math.floor((bottom - top) / rowH));
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  // 整块的"高"按校验器的口径算（每行 rowH）：anchor="bottom" 时让块的**底边**落在 bottom，
  // 否则最后一行自己的盒子（rowH 高）会探到 bottom 以下、压住该面板的 x 刻度带。
  const y = anchor === "top" ? top : Math.max(top, bottom - pieces.length * rowH);
  return pieces.map((body, i) => text(x, y + i * rowH, body, { role: "note", size, color, maxWidth: budget * 7 }));
}

/** 紧凑数字（4 位有效数字）：面板文本里的默认口径。 */
export function fmt(v: number, sig = 4): string {
  return formatNumber(v, sig);
}

/** 可能缺失的量：`undefined` → `"?"`（**绝不**拿 0 或 NaN 顶替）。 */
export function fmtOpt(v: number | undefined, sig = 4): string {
  return v === undefined ? "?" : formatNumber(v, sig);
}

/** 科学计数法（`1.04×10⁻¹¹` 形式）：量级本身是结论时用它（如残差 vs 偏差）。 */
export function sci(v: number, sig = 3): string {
  return formatSci(v, sig);
}

/** 时间轴 / 步进信息：**真实档位**与「请求 t 被夹住」的说明（`docs/PROTOCOLS.md` §5）。 */
export interface StepInfo {
  /** 画面真正使用的档位（来自帧 `meta.step`，缺失时才按请求 t 夹取）。 */
  step: number | undefined;
  /** 需要出现在图上的说明（每条都点了名：缺什么 / 夹到了哪里）。 */
  notes: string[];
}

/**
 * 把 `t` 解析成**实际档位**：本章三个带时间轴的 viz 的 `t` 不是时间，而是
 * 「已展开的基函数个数 k」（projection）、「Remez 轮次」（equioscillation）、
 * 「最高阶数 n」（zeros_evolution），后端 `_clamp_step` 已经夹过一次；模型这一侧
 * 只做两件事：以帧 `meta.step` 为准、并在请求值与实到值不同时**明说**（不假装 t 就是索引）。
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
        `时间轴拖过 ${cfg.hi} 就被后端夹住，画面按实到档位画，不假装 t 就是索引`,
    );
  }
  return { step, notes };
}

/**
 * 从帧给的一组数里挑出 1-2-5 序列上的那些（`1/2/5 × 10^k`），用作对数横轴的刻度。
 *
 * 为什么不用 `niceTicks`：`n` 只跨 1.4 个十倍程（2…40），`niceTicks` 的"十倍程不足"分支
 * 只给得出 `[5, 10]` 两个刻度；而 1-2-5 筛选出来的 `[2, 5, 10, 20, 40]` 是同一套 1-2-5
 * 约定的更细一层，且**值全部来自帧**（不是模型编的坐标）。
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

/** 一组数组里所有有限值的 `[min, max]`；一个都没有时返回 `undefined`（调用方给退化范围）。 */
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

/** `min..max` 里包含 0（柱状图/误差曲线要给 0 留位置）。 */
export function withZero(range: [number, number] | undefined): [number, number] {
  if (range === undefined) return [0, 1];
  return [Math.min(0, range[0]), Math.max(0, range[1])];
}

/** 逐元素 `a − b`（长度取小者）；只用于**显示空间的对照**，不是重算物理量（P8）。 */
export function difference(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const n = Math.min(a.length, b.length);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) out[i] = (a[i] as number) - (b[i] as number);
  return out;
}

/** 逐元素 `|a|`（对数轴的纵轴要求 > 0，取模是显示变换）。 */
export function absolute(a: ArrayLike<number>): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i += 1) out[i] = Math.abs(a[i] as number);
  return out;
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

/** 有限值的最大绝对值；没有有限值时 `undefined`（用来报同屏对照的"偏差有多大"）。 */
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

/** 帧 `strings` 里的说明原文（缺失时返回 `undefined`，调用方负责"缺了什么"的降级文案）。 */
export function noteOf(frame: Frame, key: string): string | undefined {
  const raw = frame.strings[key];
  const body = typeof raw === "string" ? raw.trim() : "";
  return body === "" ? undefined : body;
}
