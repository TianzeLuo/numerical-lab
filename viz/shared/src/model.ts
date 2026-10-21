/**
 * model 层契约（`docs/PROTOCOLS.md` §6）：`Frame`（后端 `.nlf` 帧）→ `Scene`（纯 JSON 场景）
 * 的唯一类型入口，外加两个 0.0 章模型共用的「读帧 / 降级 / 溢出 / 角标排布」工具。
 *
 * 三条不变量（写进类型与 docstring，而不是靠自觉）：
 *
 * 1. **纯函数、DOM-free**：model 只读 `frame` 与 `ModelOptions`，返回 `Scene`。
 *    像素与 canvas 属 `render2d.ts`，光栅化属 `tools/raster.py`——本文件不碰它们。
 * 2. **前端不算物理**（`docs/POLICY.md` P8/P9）：模型只消费帧里已有的数组与 `meta` 标量。
 *    这里唯一的"计算"是**显示几何**：线性插值定位当前时刻标记、按有限值分段、
 *    由数据算 1-2-5 取整的轴范围。任何物理量（解、残差、误差、稳定性判据）都必须来自后端。
 * 3. **缺字段分两档**（这是本项目"宁可当场炸，也不画一张空图"的落地）：
 *    - **画不出来必需的**（被引用的数组、长度不一致、非空 `frame.pitfalls`）
 *      → 抛 {@link ModelInputError}，绝不画半张图；
 *    - **只影响标注的**（`meta` 标量、`strings` 说明、`frame.critical`）
 *      → 降级：跳过该角标/提示，并追加一条显式文字说明**缺了什么**。
 *      **绝不猜数**：不补默认值、不拿 UI 请求参数冒充实测值（用了会写明"请求值"）。
 */

import type { Frame, TypedArray } from "./frame.js";
import type { Badge, LineSeries, Scene } from "./scene.js";
import {
  BADGE_BOX_H,
  BADGE_PAD_PX,
  BADGE_UNIT_PX,
  badge,
  badgeBoxWidth,
  newScene,
  overflowBadge,
  series,
  text,
  textBoxOf,
  textWidthUnits,
  type TextSpec,
} from "./scene.js";
// 这两个是"文本宽度尺子"的公开入口：多章模型从 `model.ts` 取（历史原因），
// 实现在 `scene.ts`（与校验器同一份）。
export { BADGE_BOX_H, BADGE_PAD_PX, BADGE_UNIT_PX, badgeBoxWidth, textWidthUnits };
import { checkOverflow, type OverflowReport } from "./geometry.js";

// --------------------------------------------------------------------------- //
// 对外契约
// --------------------------------------------------------------------------- //

/** 渲染一个场景所需的全部外部输入（时间轴、画布尺寸、参数、历史帧）。 */
export interface ModelOptions {
  width: number;
  height: number;
  t: number;
  params: Record<string, number | boolean | string>;
  theme?: "light" | "dark";
  /** 时间轴上已经收到的帧（`scrub` 模式下用来拼时空热力图）。 */
  history?: readonly FrameHistoryEntry[];
}

/** 历史帧：`t` 是该帧对应的时间（由时间轴给出，不是帧内插值出来的）。 */
export interface FrameHistoryEntry {
  t: number;
  frame: Frame;
}

/** 一个可视化的 model：`Frame × ModelOptions → Scene`（纯函数，可重复调用）。 */
export type SceneModel = (frame: Frame, opts: ModelOptions) => Scene;

/**
 * 帧不满足模型输入契约时抛出的错误。
 *
 * 只用于"画不出来"的缺失（必需数组、长度不一致、`pitfalls` 为空）；
 * 只影响标注的缺失走降级（见文件头第 3 条），**不抛错**——否则时间轴滑到
 * 一个缺 `t_actual` 的帧上就会整块白屏。
 */
export class ModelInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ModelInputError";
  }
}

// --------------------------------------------------------------------------- //
// 主题
// --------------------------------------------------------------------------- //
/** 模型用到的全部颜色（render2d 的轴/刻度色不在此列：那是渲染层的事）。 */
export interface Palette {
  background: string;
  ink: string;
  muted: string;
  note: string;
  warn: string;
  /** 主面板曲线色（多个序列按顺序取用）。 */
  curves: readonly string[];
  /** 精确解/参考解的对比色。 */
  exact: string;
  /** 残差/误差层颜色。 */
  residual: string;
  /** 当前时刻标记颜色。 */
  marker: string;
  /** 参考线（阈值、meta 标量线）。 */
  reference: string;
  /** 当前时刻游标竖线。 */
  cursor: string;
}

const LIGHT: Palette = {
  background: "#ffffff",
  ink: "#1a202c",
  muted: "#4a5568",
  note: "#2c5282",
  warn: "#c53030",
  curves: ["#2b6cb0", "#2f855a", "#b7791f"],
  exact: "#dd6b20",
  residual: "#6b46c1",
  marker: "#c53030",
  reference: "#718096",
  cursor: "#a0aec0",
};

const DARK: Palette = {
  background: "#111827",
  ink: "#f7fafc",
  muted: "#cbd5e0",
  note: "#90cdf4",
  warn: "#fc8181",
  curves: ["#63b3ed", "#68d391", "#f6ad55"],
  exact: "#f6ad55",
  residual: "#d6bcfa",
  marker: "#fc8181",
  reference: "#a0aec0",
  cursor: "#4a5568",
};

/** 取主题配色；未知/缺省主题按 light 处理（时间轴不该因为主题拼错而白屏）。 */
export function palette(theme: "light" | "dark" = "light"): Palette {
  return theme === "dark" ? DARK : LIGHT;
}

// --------------------------------------------------------------------------- //
// 读帧：必需 vs 可选
// --------------------------------------------------------------------------- //

/**
 * 把 `frame.array()` 的 `TypedArray` 联合收窄成 `ArrayLike<number>`。
 *
 * `.nlf` 允许 `i8`（`BigInt64Array`），而本项目的物理量都是 f8/f4；真遇到 i8 时
 * 逐元素 `Number()` 转换（而不是把它当 number 硬读——那会静默产出 bigint），
 * 转换长度与元素个数一致，故下游的"长度必须一致"校验依旧成立。
 */
export function toNumeric(a: TypedArray): ArrayLike<number> {
  if (a instanceof BigInt64Array) {
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i += 1) out[i] = Number(a[i]);
    return out;
  }
  return a as unknown as ArrayLike<number>;
}

/** 取一个**必需**数组；不存在就抛 {@link ModelInputError}（消息里列出帧里现有的数组名）。 */
export function requireArray(frame: Frame, name: string): ArrayLike<number> {
  if (!frame.has(name)) {
    const have = frame.names.map((n) => `'${n}'`).join(", ");
    throw new ModelInputError(
      `帧 ${frame.chapter}/${frame.viz} 缺少必需数组 '${name}'；现有：[${have}]` +
        `（PROTOCOLS §1.1：数组只能由后端给，model 不得自造）`,
    );
  }
  return toNumeric(frame.array(name));
}

/** 取一个**可选**数组；不存在返回 undefined（用于"有就画、没有就不画"的辅助量）。 */
export function optionalArray(frame: Frame, name: string): ArrayLike<number> | undefined {
  return frame.has(name) ? toNumeric(frame.array(name)) : undefined;
}

/** 两个数组必须等长，否则抛错（x/y 长度不一致画出来是错位的图，不是"缺一点数据"）。 */
export function requireSameLength(a: ArrayLike<number>, b: ArrayLike<number>, what: string): void {
  if (a.length !== b.length) {
    throw new ModelInputError(`${what} 长度不一致：${a.length} vs ${b.length}（拒绝画错位的图）`);
  }
}

/** `meta` 里的有限标量；缺失/`null`/非有限一律 undefined（不猜 0）。 */
export function metaNumber(frame: Frame, key: string): number | undefined {
  const v = frame.scalar(key);
  return v !== undefined && Number.isFinite(v) ? v : undefined;
}

/** `strings` 里的非空字符串；缺失/空白返回 undefined。 */
export function stringOf(frame: Frame, key: string): string | undefined {
  const raw = frame.strings[key];
  return typeof raw === "string" && raw.trim() !== "" ? raw : undefined;
}

/** 数据来源：帧 `strings.data_source` 优先（后端在 `build_frame` 里写入），缺省 `"backend"`。 */
export function dataSourceOf(frame: Frame): string {
  return stringOf(frame, "data_source") ?? "backend";
}

/**
 * 理论临界值：优先帧里的 `frame.critical[name]`（后端由 manifest 的 `governing_critical` 写入），
 * 缺失时回落到调用方给的 manifest 常量。**不猜**：两种来源都会原样进徽标文本。
 */
export function criticalOf(frame: Frame, name: string, fallback: number): number | number[] {
  const raw = frame.critical[name];
  if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  if (Array.isArray(raw) && raw.length > 0 && raw.every((v) => Number.isFinite(v))) return raw;
  return fallback;
}

/** `opts.params` 里的数值（布尔按 1/0，数字串按字面转换；其它一律 undefined）。 */
export function paramNumber(params: ModelOptions["params"], name: string): number | undefined {
  const raw = params?.[name];
  if (typeof raw === "number") return Number.isFinite(raw) ? raw : undefined;
  if (typeof raw === "boolean") return raw ? 1 : 0;
  if (typeof raw === "string" && raw.trim() !== "") {
    const v = Number(raw);
    return Number.isFinite(v) ? v : undefined;
  }
  return undefined;
}

/** 规则 7：`frame.pitfalls` 必须非空且够长（与 `validateScene` 的 10 字门槛一致），否则抛错。 */
export function requirePitfalls(frame: Frame): string {
  const body = typeof frame.pitfalls === "string" ? frame.pitfalls.trim() : "";
  if (body.length < 10) {
    throw new ModelInputError(
      `帧 ${frame.chapter}/${frame.viz} 的 pitfalls 为空或太短（${body.length} 字）：` +
        `规则 7 要求帧自带「它会怎样骗你」，model 不自行编造`,
    );
  }
  return body;
}

/** 参数行（用于面板角落的参数摘要）。 */
export interface ParamLine {
  /** `meta` / `params` 里的键名（后端字段名，两者一致）。 */
  name: string;
  /** 显示标签（如 `ζ`）。 */
  label: string;
  /** 有效数字位数（缺省 4）。 */
  digits?: number;
}

/** 参数摘要：**帧 `meta` 优先**，缺失才回落到 UI 请求参数并显式标注「请求值」。 */
export interface ParamSummary {
  text: string;
  /** 降级说明（每条都会作为显式文字出现在图上）。 */
  degraded: string[];
}

/** 拼一行"ζ=0.05000　ω=1.000"式的参数摘要；每项的来源都会标注在降级列表里。 */
export function paramSummary(frame: Frame, opts: ModelOptions, lines: readonly ParamLine[]): ParamSummary {
  const parts: string[] = [];
  const degraded: string[] = [];
  for (const line of lines) {
    const fromFrame = metaNumber(frame, line.name);
    const value = fromFrame ?? paramNumber(opts.params, line.name);
    if (value === undefined) {
      parts.push(`${line.label}=?`);
      degraded.push(`帧 meta 与请求参数都没有 ${line.name}：留「?」，不猜`);
      continue;
    }
    const shown = Number.isInteger(value) ? String(value) : value.toPrecision(line.digits ?? 4);
    if (fromFrame === undefined) {
      parts.push(`${line.label}=${shown}（请求值）`);
      degraded.push(`帧 meta 缺 ${line.name}：参数摘要用请求值并标注`);
    } else {
      parts.push(`${line.label}=${shown}`);
    }
  }
  return { text: parts.join("　"), degraded };
}

// --------------------------------------------------------------------------- //
// 采样与分段（显示几何，不是物理）
// --------------------------------------------------------------------------- //

/**
 * 在 `(xs, ys)` 上线性插值取样（用于把"当前时刻"标记放到曲线上）。
 *
 * * `xq` 落在 `xs` 覆盖范围之外 → `undefined`（**不外推**：帧外没有数据可画）；
 * * 区间端点非有限 → 返回 `undefined`（缺口处绝不用直线跨过去）；
 * * 一律按给定顺序扫描，不假设 `xs` 单调。
 */
export function sampleLinear(xs: ArrayLike<number>, ys: ArrayLike<number>, xq: number): number | undefined {
  const n = Math.min(xs.length, ys.length);
  if (n === 0 || !Number.isFinite(xq)) return undefined;
  if (n === 1) {
    const y0 = ys[0] as number;
    return xs[0] === xq && Number.isFinite(y0) ? y0 : undefined;
  }
  for (let i = 0; i + 1 < n; i += 1) {
    const a = xs[i] as number;
    const b = xs[i + 1] as number;
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    const lo = a < b ? a : b;
    const hi = a < b ? b : a;
    if (xq < lo || xq > hi) continue;
    const ya = ys[i] as number;
    const yb = ys[i + 1] as number;
    if (!Number.isFinite(ya) || !Number.isFinite(yb)) return undefined;
    if (a === b) return xq === a ? ya : undefined;
    return ya + ((xq - a) / (b - a)) * (yb - ya);
  }
  return undefined;
}

/** 一段连续可绘样本 `[start, end)`。 */
export interface Run {
  start: number;
  end: number;
}

/** 有限且 **> 0** 的值的 `[min, max]`（对数轴的定义域）；没有这样的值时 undefined。 */
export function finitePositiveMinMax(a: ArrayLike<number>): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (!Number.isFinite(v) || v <= 0) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : undefined;
}

/** 把 `v` 夹到 `[lo, hi]`（文本定位用；`lo > hi` 时返回 `lo`）。 */
export function clamp(v: number, lo: number, hi: number): number {
  if (!(hi > lo)) return lo;
  return v < lo ? lo : v > hi ? hi : v;
}
/** 有限值的 `[min, max]`；一个有限值都没有时返回 undefined（调用方须给退化范围）。 */
export function finiteMinMax(a: ArrayLike<number>): [number, number] | undefined {
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

/** 把 `[0, n)` 按 `ok(i)` 切成若干连续段（NaN/∞ 处断开，而不是跨过去连线）。 */
export function finiteRuns(n: number, ok: (index: number) => boolean): Run[] {
  const runs: Run[] = [];
  let start = -1;
  for (let i = 0; i < n; i += 1) {
    if (ok(i)) {
      if (start < 0) start = i;
    } else if (start >= 0) {
      runs.push({ start, end: i });
      start = -1;
    }
  }
  if (start >= 0) runs.push({ start, end: n });
  return runs;
}

/** 线型样式：`series()` 的可选项，外加一个"这个点能不能上对数轴"的判据。 */
export interface SeriesStyle extends Omit<Partial<LineSeries>, "x" | "y"> {
  /** 额外的可绘判据（如对数轴要求 `v > 0`）；缺省只要求有限。 */
  accept?: (value: number, index: number) => boolean;
}

function sliceOf(a: ArrayLike<number>, run: Run): Float64Array {
  const out = new Float64Array(run.end - run.start);
  for (let i = run.start; i < run.end; i += 1) out[i - run.start] = a[i] as number;
  return out;
}

/**
 * 把 `(x, y)` 切成"每段都是有限值"的若干条曲线。
 *
 * **为什么必须分段而不是把 NaN 交给渲染层**：`validateScene` 明确拒绝含非有限纵坐标的
 * line 图层（"请显式裁剪或用 overflow 标记"）；而 `render2d` 的 `strokePath` 遇到
 * 断点会 `moveTo`，所以分段 = 缺口处断开，视觉上就是"这里没有数据"，
 * 既不会被直线跨过去，也不会被静默丢掉。
 */
export function segmentedSeries(x: ArrayLike<number>, y: ArrayLike<number>, color: string, style: SeriesStyle = {}): LineSeries[] {
  const { accept, ...rest } = style;
  const n = Math.min(x.length, y.length);
  const runs = finiteRuns(n, (i) => {
    const xv = x[i] as number;
    const yv = y[i] as number;
    return Number.isFinite(xv) && Number.isFinite(yv) && (accept === undefined || accept(yv, i));
  });
  return runs.map((run, k) =>
    series(sliceOf(x, run), sliceOf(y, run), color, k === 0 ? rest : { ...rest, label: undefined }),
  );
}

// --------------------------------------------------------------------------- //
// 溢出（规则 1）与角标排布
// --------------------------------------------------------------------------- //

/** 一行"有多少非有限值没画出来"的说明（与 `geometry.ts` 的灰色=无数据约定对齐）。 */
export function nonFiniteNote(count: number): string {
  return `${count} 个非有限值（NaN/±∞）未绘制：折线在缺口处断开，不用直线跨过去`;
}

/**
 * 把溢出写进 Scene：`checkOverflow` 负责"图层数据越出轴范围"，
 * `rawNonFinite` 负责"帧里有、但被分段剔除因而 `checkOverflow` 看不见"的非有限值。
 *
 * 两者都必须进 `overflow.note` 并挂上 `overflowBadge`——否则 NaN/∞ 就成了
 * "图上没有、账上也没有"的东西，正是规则 1 要禁止的。
 */
export function applyOverflow(scene: Scene, rawNonFinite: number, opts: { slot: number; width: number }): OverflowReport {
  const report = checkOverflow(scene);
  const parts: string[] = [];
  if (report.offenders.length > 0) parts.push(report.note);
  if (rawNonFinite > 0) parts.push(nonFiniteNote(rawNonFinite));
  const note = parts.join("；") || report.note;
  scene.overflow = { flag: report.flag || rawNonFinite > 0, note, severity: rawNonFinite > 0 ? "data" : "clip" };
  if (scene.overflow.flag) {
    // 非有限值是数据坏了；单纯越出固定视窗是显示选择。两者都要报告，但颜色/措辞不同。
    const tone = rawNonFinite > 0 ? "danger" : "warning";
    scene.badges.push(placeBadge(overflowBadge(note, 8, 44, tone), opts.slot, opts.width));
  }
  return report;
}

/**
 * 角标列布局。
 *
 * * `top = 34`：`renderScene` 把 `scene.title` 画在 (12, 8)，角标整体下移才不会压住标题；
 * * 槽位**固定**（见各模型的 `SLOT_*` 常量）：缺一个角标时其余不挪位，动画不抖；
 * * `maxSlots = 8`（4 行 × 2 列）：长文本角标（溢出/降级）独占一行、右边留空，
 *   不会与邻居的文字叠在一起。
 */
export const BADGE_LAYOUT = {
  x: 8,
  top: 34,
  rowH: 18,
  colW: 344,
  maxSlots: 8,
} as const;

/**
 * 一行角标能容纳的显示宽度单位（`placeBadge` 用它裁剪过长文案）。
 *
 * 两列布局时就是列宽；画布窄到只剩一列时，整幅宽减去左右各 8px 都是可用的。
 */
export function badgeUnitBudget(width: number): number {
  // 两列布局：预算是列宽；单列（窄画布）：预算是"整幅宽 − 左右边距"。
  // 早先单列时错误地取了 `max(colW, width-16)`：200px 宽的导出画布于是按 344px 放行，
  // 角标直接冲出画布（`--width 200` 的 PNG/GIF 导出场景真踩到，规则 8 当场报违规）。
  const usable = badgeColumns(width) >= 2 ? BADGE_LAYOUT.colW : Math.max(60, width - 2 * BADGE_LAYOUT.x);
  return Math.max(8, Math.floor((usable - BADGE_PAD_PX) / BADGE_UNIT_PX));
}

/** 画布够宽时用两列角标（列宽 344 + 列间距 24）。 */
export function badgeColumns(width: number): number {
  return width >= 2 * BADGE_LAYOUT.colW + 24 ? 2 : 1;
}

/** 角标区需要几行（窄画布退化成单列，行数翻倍——排版函数据此让出顶部）。 */
export function badgeRows(width: number): number {
  return Math.ceil(BADGE_LAYOUT.maxSlots / badgeColumns(width));
}

/**
 * 把 builder 造出来的角标放进第 `slot` 个固定槽位。
 *
 * `sourceBadge` / `criticalBadge` / `overflowBadge` 的缺省坐标（8, 8/44/62）会被
 * `renderScene` 画在 (12, 8) 的标题压住，所以这里统一重排。
 */
export function placeBadge<T extends Badge>(b: T, slot: number, width: number): T {
  const cols = badgeColumns(width);
  b.x = BADGE_LAYOUT.x + (slot % cols) * BADGE_LAYOUT.colW;
  b.y = BADGE_LAYOUT.top + Math.floor(slot / cols) * BADGE_LAYOUT.rowH;
  // 槽位是**定宽**的（colW = 344），而角标文本长度由帧里的数值决定（状态/溢出/降级文案
  // 都可能很长）。不定宽裁剪的后果不是"难看一点"，而是**长角标横穿到右列、把邻居压在下面**——
  // 1.2 与 0.0 章真出现过（左侧 65 单位的"副轴：自适应…"盖住右侧的临界值角标），
  // 而 `validateScene` 当时查不出来（它不知道文字宽度）。
  // 现在统一在这里裁到列宽预算：宁可少显示几个字（尾部 `…`），也不许两枚角标叠在一起。
  // 完整文案请放到面板文本 / `pitfalls` 里——角标只是索引，不是正文。
  if (b.detail === undefined) b.detail = b.text;
  b.text = clipBadgeText(b.text, badgeUnitBudget(width));
  return b;
}

/** 顶部预留高度：角标区之下才是面板，两者互不重叠；只由画布宽度决定（与帧无关）。 */
export function headerBand(width: number): number {
  return BADGE_LAYOUT.top + badgeRows(width) * BADGE_LAYOUT.rowH + 8;
}

/** 一个图层都没建起来 ⇒ 帧里没有任何可绘数据；抛错比返回"空图"诚实（规则 1/3）。 */
export function requireDrawableLayers(scene: Scene): void {
  if (scene.layers.length === 0) {
    throw new ModelInputError(`帧 ${scene.chapter}/${scene.viz} 里没有任何可绘的有限值：拒绝返回空场景`);
  }
}

// --------------------------------------------------------------------------- //
// 面板排版（纯几何：与帧无关，动画时不会抖动）
// --------------------------------------------------------------------------- //

/** 像素矩形 `[x, y, w, h]`，与 `Axis.rect` 同一约定。 */
export type Rect = [number, number, number, number];

export interface StackedPanels {
  /** 主面板。 */
  main: Rect;
  /** 误差/残差副轴（规则 3）。 */
  error: Rect;
  /** 内容区左边界（= `main[0]`，供分栏用）。 */
  left: number;
  /** 内容区总宽（左边界到右边界）。 */
  contentWidth: number;
  /** 内容区顶端（= `main[1]`）。 */
  top: number;
  /** 内容区底端（再往下是底轴 xlabel 与 pitfalls 面板，别再放面板）。 */
  bottom: number;
}

export interface StackedPanelOptions {
  /** 左侧留白（y 轴刻度 + ylabel）；缺省 64。 */
  left?: number;
  /** 右侧留白；缺省 24。 */
  right?: number;
  /** 主面板占内容高度的比例；缺省 0.56。 */
  mainShare?: number;
  /** 两个面板之间的间距（放下主面板的 x 刻度）；缺省 22。 */
  gap?: number;
  /**
   * 底部总留白：底轴的 xlabel（render2d 画在轴下 ~29px）+ `renderScene` 的 pitfalls 面板。
   * 缺省 `8 + 60 + 34`——`web` 端画布高度不足时它会被自动压缩。
   */
  footer?: number;
  /** 顶部在 `headerBand(width)` 之外再让出的高度；缺省 8。 */
  headerExtra?: number;
}

/**
 * 「主面板 + 误差面板」竖直堆叠的排版。
 *
 * 全部由画布尺寸算出，**与帧无关**：同一 params 下连续帧的面板位置逐像素相同，
 * 否则动画里曲线会随角标/文本增删上下跳。
 */
export function stackedPanels(width: number, height: number, opts: StackedPanelOptions = {}): StackedPanels {
  const left = opts.left ?? 64;
  const right = opts.right ?? 24;
  const footer = opts.footer ?? 8 + 60 + 34;
  const top = headerBand(width) + (opts.headerExtra ?? 8);
  const contentWidth = Math.max(120, width - left - right);
  const bottom = Math.max(top + 120, height - footer);
  const plotH = bottom - top;
  const gap = opts.gap ?? 22;
  const mainH = Math.max(56, Math.round(plotH * (opts.mainShare ?? 0.56)));
  const errorH = Math.max(40, plotH - mainH - gap);
  return {
    main: [left, top, contentWidth, mainH],
    error: [left, top + mainH + gap, contentWidth, errorH],
    left,
    contentWidth,
    top,
    bottom,
  };
}

export interface ColumnsLayout {
  /** 左列的主面板 / 误差面板。 */
  main: Rect;
  error: Rect;
  /** 右列（时空热力图这类"第二个视角"）。 */
  side: Rect;
}

/**
 * 「左列：主 + 误差；右列：侧面板」的分栏排版（`heat_ftcs` 的时空热力图用）。
 *
 * 右列与左列之间要留出右列自己的 y 轴刻度与 ylabel，故 `columnGap` 缺省 58。
 */
export function columnsLayout(
  width: number,
  height: number,
  opts: StackedPanelOptions & { share?: number; columnGap?: number } = {},
): ColumnsLayout {
  const full = stackedPanels(width, height, opts);
  const share = opts.share ?? 0.58;
  const gap = opts.columnGap ?? 58;
  const leftW = Math.max(160, Math.round(full.contentWidth * share));
  const sideW = Math.max(120, full.contentWidth - leftW - gap);
  const totalH = full.error[1] + full.error[3] - full.top;
  return {
    main: [full.left, full.top, leftW, full.main[3]],
    error: [full.left, full.error[1], leftW, full.error[3]],
    side: [full.left + leftW + gap, full.top, sideW, totalH],
  };
}

// --------------------------------------------------------------------------- //
// Scene 外壳
// --------------------------------------------------------------------------- //
export interface SceneShellInit {
  /** 帧里没写 chapter 时的回落值（章节模型自己的常量）。 */
  chapter: string;
  viz: string;
  title: string;
  /** 该帧对应的时间（调用方已决定：帧 `meta.t` 优先）。 */
  t: number;
  dataSource: string;
  pitfalls: string;
}

/** 起一个已带背景/主题/元信息的空 Scene（后续只用 scene.ts 的 builder 往里加东西）。 */
export function sceneShell(frame: Frame, opts: ModelOptions, init: SceneShellInit): Scene {
  const scene = newScene({
    chapter: frame.chapter || init.chapter,
    viz: frame.viz || init.viz,
    title: init.title,
    width: Math.max(320, Math.round(opts.width)),
    height: Math.max(240, Math.round(opts.height)),
    t: init.t,
    dataSource: init.dataSource,
    pitfalls: init.pitfalls,
  });
  scene.canvas.background = palette(opts.theme).background;
  return scene;
}

/** 画布尺寸（供模型算面板矩形；与 `sceneShell` 用同一套夹取，避免两处不一致）。 */
export function canvasSize(opts: ModelOptions): { width: number; height: number } {
  return { width: Math.max(320, Math.round(opts.width)), height: Math.max(240, Math.round(opts.height)) };
}

/** 降级说明的统一前缀（测试按它断言"缺字段没有静默"）。 */
export const DEGRADED_PREFIX = "⚠ 降级：";

/** 把角标文本裁到 `maxUnits` 显示宽度内（超出用 `…` 收尾），避免角标横穿画布。 */
/**
 * 按显示宽度裁剪可能含 `$...$` 的标题，**绝不把公式定界符或 TeX 命令切成两半**。
 * 装不下的数学段整段丢弃（保留前面的普通文字与 `…`），而不是留下一个未闭合的 `$`。
 */
export function clipRichText(text: string, maxUnits: number): string {
  if (textWidthUnits(text) <= maxUnits) return text;
  const parts = text.split(/(\$[^$]*\$|\\\([\s\S]*?\\\))/g).filter((p) => p !== "");
  let out = "";
  let used = 0;
  for (const part of parts) {
    const w = textWidthUnits(part);
    if (used + w > Math.max(1, maxUnits - 1)) break;
    out += part;
    used += w;
  }
  return `${out.trimEnd()}…`;
}

export function clipBadgeText(text: string, maxUnits = 84): string {
  if (textWidthUnits(text) <= maxUnits) return text;
  let out = "";
  let used = 0;
  for (const ch of text) {
    const w = (ch.codePointAt(0) ?? 0) > 0x2e80 ? 2 : 1;
    if (used + w > maxUnits - 1) break;
    out += ch;
    used += w;
  }
  return `${out}…`;
}

/**
 * 把降级说明收进**一个** status 角标（放第 `slot` 槽）。
 *
 * 为什么不抛错：缺 `meta.t_actual` 这类字段只是"少一条标注"，时间轴滑到这样的帧上
 * 不该整块白屏；但也不能静默——角标里逐条写明**缺了什么、因此没画什么**。
 * 超过 2 条时折叠为前 2 条 + 计数，避免角标横穿整张画布。
 */
export function degradedBadge(items: readonly string[], slot: number, width: number): Badge | undefined {
  if (items.length === 0) return undefined;
  const head = items.slice(0, 2).join("；");
  const rest = items.length > 2 ? `；等 ${items.length} 项` : "";
  return placeBadge(badge("status", clipBadgeText(`${DEGRADED_PREFIX}${head}${rest}`), BADGE_LAYOUT.x, BADGE_LAYOUT.top), slot, width);
}

/**
 * 消解文本层之间的重叠：按 y 序保留先到者，与已保留块重叠的**丢弃**并如实计数。
 *
 * 为什么放在共享层：模型里几块说明通常各自按"自己的行数"排下一块的起点
 * （`top + lines.length * 15`），而**真实占几行**取决于 `maxWidth` 处的折行——
 * 画布变窄或帧里的说明变长时，每块都会多折出一两行，下一块就压上来了。
 * 规则 8 的"文本互相压住"由两侧校验器（`tools/raster.py::_text_deco_issues` 与
 * `scene.ts::validateScene`）当场拒绝，所以模型必须自己保证不重叠。
 *
 * 为什么只丢不移：往下推会推进下一条轴的**装饰带**（刻度与轴名由渲染层自己画，
 * 模型看不见），那是另一条规则 8 违规。诚实口径：宁可少说，不可叠着说、也不可压着轴说。
 * 丢弃数量写在画布顶部一行短注里，不静默（防误导七条第一条）。
 *
 * ``rect`` 的约定与 `textBoxOf` 一致：``[x, y(顶), 宽, 高]``——**第四个是高度不是底边**。
 * 想"不许越过某条底线"时传 ``高 = 底线 − 顶``（这里踩过坑：把底线当高度传，
 * 实际允许的底边比预期低了 24px，底部的诚实提示就又被面板压住了）。
 */
export function resolveTextOverlaps(
  layers: TextSpec[],
  rect: readonly [number, number, number, number],
): TextSpec[] {
  const kept: TextSpec[] = [];
  let dropped = 0;
  const limit = rect[1] + rect[3];
  for (const layer of [...layers].sort((a, b) => (a.y ?? 0) - (b.y ?? 0))) {
    const box = textBoxOf({ ...layer, y: layer.y ?? 0 });
    if (!box.ok || box.y + box.h > limit) {
      dropped += 1;
      continue;
    }
    const hit = kept.some((prev) => {
      const a = textBoxOf({ ...prev, y: prev.y ?? 0 });
      const ox = Math.min(a.x + a.w, box.x + box.w) - Math.max(a.x, box.x);
      const oy = Math.min(a.y + a.h, box.y + box.h) - Math.max(a.y, box.y);
      return ox > 2 && oy > 2;
    });
    if (hit) {
      dropped += 1;
      continue;
    }
    kept.push(layer);
  }
  if (dropped > 0 && rect[1] >= 20) {
    kept.push(
      text(rect[0], 8, `文本放不下：另有 ${dropped} 行未显示（完整口径见章节笔记与角标）`, {
        role: "note",
        size: 11,
        color: "#718096",
      }),
    );
  }
  return kept;
}
