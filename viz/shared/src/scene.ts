/**
 * Scene —— 可视化层与光栅层之间的**唯一中间表示**。
 *
 * 为什么要有这一层（docs/ARCHITECTURE.md §1）：
 *  - 浏览器用 canvas 画它（render2d.ts），Python 用光栅器画它（tools/raster.py）；
 *  - 于是"视觉回归"在没有浏览器的环境里也能做像素级基线；
 *  - 更重要的是：**防误导七条规则被写进了类型与校验器**，而不是写在文档里靠自觉。
 *    例如 `axes.autoscaled` 是必填字段、`overflow.note` 在溢出时必须非空、
 *    `badges` 必须至少含一个 `source` 徽标——忘记标注会直接让 `validateScene` 抛错。
 *
 * 本文件是**纯函数**模块：不触碰 DOM、不做网络请求、不读全局状态。
 */

import { linspace, logspace, niceLimits, niceTicks, type ScaleKind } from "./axes.js";

// --------------------------------------------------------------------------- //
// 类型
// --------------------------------------------------------------------------- //
export interface CanvasSpec {
  w: number;
  h: number;
  background: string;
}

/** 一个坐标轴（含副轴：误差场 / 残差场）。 */
export interface Axis {
  id: string;
  /** 像素矩形 [x, y, w, h]，原点在左上角。 */
  rect: [number, number, number, number];
  xlim: [number, number];
  ylim: [number, number];
  xscale: ScaleKind;
  yscale: ScaleKind;
  xlabel: string;
  ylabel: string;
  title: string;
  /**
   * 规则 1/4：坐标轴是否自适应。**必须显式声明**——调度自适应轴以"藏住发散"
   * 是本项目最警惕的骗术之一。
   */
  autoscaled: boolean;
  xticks: number[];
  yticks: number[];
  grid: boolean;
  frame: boolean;
}

export type CmapName = "viridis" | "magma" | "coolwarm" | "gray" | "rdbu";
export type MarkerKind = "none" | "o" | "s" | "x" | "+";

export interface LineSeries {
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  color: string;
  width: number;
  dash?: number[];
  label?: string;
  marker?: MarkerKind;
  alpha?: number;
}

export interface BandSpec {
  x: ArrayLike<number>;
  ylo: ArrayLike<number>;
  yhi: ArrayLike<number>;
  color: string;
  alpha: number;
  label?: string;
}

export type Layer =
  | { kind: "line"; axes: string; series: LineSeries[] }
  | { kind: "band"; axes: string; band: BandSpec }
  | { kind: "scatter"; axes: string; x: ArrayLike<number>; y: ArrayLike<number>; r: number; color: string; label?: string; alpha?: number }
  | {
      kind: "heat";
      axes: string;
      nx: number;
      ny: number;
      /**
       * 行主序、长度 nx*ny 的**节点采样**：第 j 行第 i 列对应
       * x = extent[0] + (extent[1]−extent[0])·i/(nx−1)，y 同理（**含两端**）。
       * 与 `contour`/`vectorfield` 用同一约定——两种约定混用是画错图的经典原因。
       */
      values: ArrayLike<number>;
      extent: [number, number, number, number];
      cmap: CmapName;
      vmin: number;
      vmax: number;
      logScale: boolean;
      clipBelow?: number;
      label?: string;
    }
  | {
      kind: "contour";
      axes: string;
      nx: number;
      ny: number;
      values: ArrayLike<number>;
      extent: [number, number, number, number];
      levels: number[];
      color: string;
      width: number;
      label?: string;
    }
  | {
      kind: "bars";
      axes: string;
      x: ArrayLike<number>;
      values: ArrayLike<number>;
      width: number;
      color: string;
      errors?: ArrayLike<number>;
      baseline?: number;
      label?: string;
    }
  | {
      kind: "vectorfield";
      axes: string;
      nx: number;
      ny: number;
      u: ArrayLike<number>;
      v: ArrayLike<number>;
      extent: [number, number, number, number];
      stride: number;
      color: string;
      scale: number;
    }
  | {
      /** 规则 2 的强制形态：双对数图上"实测斜率"与"理论斜率"必须同时出现。 */
      kind: "loglogfit";
      axes: string;
      h: ArrayLike<number>;
      err: ArrayLike<number>;
      measuredSlope: number;
      theorySlope: number;
      refX: [number, number];
      seriesLabel: string;
      theoryLabel: string;
      color?: string;
      theoryColor?: string;
    };

export type TextRole = "title" | "label" | "note" | "axis" | "pitfall";

export interface TextSpec {
  x: number;
  y: number;
  text: string;
  size: number;
  color: string;
  align: "left" | "center" | "right";
  role: TextRole;
  maxWidth?: number;
}

export type BadgeKind = "source" | "autoscale" | "overflow" | "preview" | "critical" | "status";

export interface Badge {
  x: number;
  y: number;
  text: string;
  kind: BadgeKind;
  color?: string;
}

export interface Scene {
  version: 1;
  chapter: string;
  viz: string;
  title: string;
  /** 该帧对应的时间（无时间轴的 viz 恒为 0）。 */
  t: number;
  canvas: CanvasSpec;
  axes: Axis[];
  layers: Layer[];
  texts: TextSpec[];
  badges: Badge[];
  /** 规则 1：越界必须显式标记，不得靠裁剪掩盖。 */
  overflow: { flag: boolean; note: string };
  /** 规则 7：它会怎样骗你（非空）。 */
  pitfalls: string;
  /** 规则 5：数据来源（backend / backend+preview / precomputed…）。 */
  dataSource: string;
}

// --------------------------------------------------------------------------- //
// 校验
// --------------------------------------------------------------------------- //
const KNOWN_KINDS = new Set(["line", "band", "scatter", "heat", "contour", "bars", "vectorfield", "loglogfit"]);
const KNOWN_CMAPS = new Set(["viridis", "magma", "coolwarm", "gray", "rdbu"]);
const KNOWN_BADGES = new Set(["source", "autoscale", "overflow", "preview", "critical", "status"]);
const KNOWN_ROLES = new Set(["title", "label", "note", "axis", "pitfall"]);

function len(a: ArrayLike<number> | undefined): number {
  return a === undefined ? 0 : (a as { length: number }).length;
}

function finiteAll(a: ArrayLike<number> | undefined): boolean {
  if (a === undefined) return false;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (!Number.isFinite(v)) return false;
  }
  return true;
}

/**
 * 校验一个 Scene 是否满足契约；返回**问题列表**（空数组表示合法）。
 *
 * 之所以返回列表而不是抛错：调用方（快照测试、光栅器、调试面板）需要一次性看到全部问题。
 * 想直接失败请用 `assertValidScene`。
 */
export function validateScene(scene: unknown): string[] {
  const issues: string[] = [];
  if (typeof scene !== "object" || scene === null) return ["Scene 不是对象"];
  const s = scene as Partial<Scene>;
  if (s.version !== 1) issues.push(`version 必须是 1，得到 ${String(s.version)}`);
  for (const key of ["chapter", "viz", "title", "pitfalls", "dataSource"] as const) {
    if (typeof s[key] !== "string" || (s[key] as string).length === 0) issues.push(`${key} 必须是非空字符串`);
  }
  if (typeof s.t !== "number" || !Number.isFinite(s.t)) issues.push("t 必须是有限数");
  if (!s.canvas || !(s.canvas.w > 0) || !(s.canvas.h > 0)) issues.push("canvas.w/h 必须为正");
  if (!Array.isArray(s.axes) || s.axes.length === 0) issues.push("至少需要一个坐标轴");
  const axisIds = new Set<string>();
  for (const a of s.axes ?? []) {
    if (!a.id) issues.push("轴缺少 id");
    if (axisIds.has(a.id)) issues.push(`轴 id 重复：${a.id}`);
    axisIds.add(a.id);
    if (!Array.isArray(a.rect) || a.rect.length !== 4) issues.push(`轴 ${a.id} 的 rect 必须是 4 元组`);
    else if (!(a.rect[2] > 0 && a.rect[3] > 0)) issues.push(`轴 ${a.id} 的宽高必须为正`);
    for (const key of ["xlim", "ylim"] as const) {
      const lim = a[key];
      if (!Array.isArray(lim) || lim.length !== 2 || !Number.isFinite(lim[0]) || !Number.isFinite(lim[1])) {
        issues.push(`轴 ${a.id} 的 ${key} 必须是两个有限数`);
      } else if (lim[0] === lim[1]) {
        issues.push(`轴 ${a.id} 的 ${key} 退化（上下限相等）——请显式给定范围或标记 autoscaled`);
      } else if (lim[0] > lim[1]) {
        issues.push(`轴 ${a.id} 的 ${key} 顺序颠倒`);
      }
    }
    if (a.xscale === "log" && Array.isArray(a.xlim) && a.xlim[0] <= 0) issues.push(`轴 ${a.id}：对数 x 轴下界必须 > 0`);
    if (a.yscale === "log" && Array.isArray(a.ylim) && a.ylim[0] <= 0) issues.push(`轴 ${a.id}：对数 y 轴下界必须 > 0`);
    if (typeof a.autoscaled !== "boolean") issues.push(`轴 ${a.id} 必须显式声明 autoscaled（规则 1/4）`);
    if (!Array.isArray(a.xticks) || !Array.isArray(a.yticks)) issues.push(`轴 ${a.id} 必须给出 xticks/yticks`);
  }
  if (!Array.isArray(s.layers) || s.layers.length === 0) issues.push("至少需要一个图层");
  const gridLayerIssue = (tag: string, layer: { nx: number; ny: number; extent: [number, number, number, number] }): void => {
    if (!(layer.nx >= 2) || !(layer.ny >= 2)) issues.push(`${tag} 网格至少 2×2（节点采样含两端），得到 ${layer.nx}×${layer.ny}`);
    const e = layer.extent;
    if (!(e[1] > e[0]) || !(e[3] > e[2])) issues.push(`${tag} extent 必须严格递增：${JSON.stringify(e)}`);
  };
  for (const [i, layer] of (s.layers ?? []).entries()) {
    const tag = `图层[${i}](${(layer as { kind?: string }).kind ?? "?"})`;
    if (!KNOWN_KINDS.has((layer as { kind: string }).kind)) {
      issues.push(`${tag} 未知的 kind`);
      continue;
    }
    const ax = (layer as { axes?: string }).axes;
    if (!ax || !axisIds.has(ax)) issues.push(`${tag} 引用了不存在的轴 ${String(ax)}`);
    switch (layer.kind) {
      case "line": {
        if (!Array.isArray(layer.series) || layer.series.length === 0) issues.push(`${tag} 至少一条曲线`);
        for (const ser of layer.series ?? []) {
          if (len(ser.x) !== len(ser.y)) issues.push(`${tag} x/y 长度不一致（${len(ser.x)} vs ${len(ser.y)}）`);
          if (!finiteAll(ser.y)) issues.push(`${tag} 含非有限纵坐标（请显式裁剪或用 overflow 标记）`);
        }
        break;
      }
      case "band": {
        const b = layer.band;
        if (!b) issues.push(`${tag} 缺少 band`);
        else if (len(b.x) !== len(b.ylo) || len(b.x) !== len(b.yhi)) issues.push(`${tag} x/ylo/yhi 长度不一致`);
        break;
      }
      case "scatter": {
        if (len(layer.x) !== len(layer.y)) issues.push(`${tag} x/y 长度不一致`);
        if (!(layer.r > 0)) issues.push(`${tag} 半径必须 > 0`);
        break;
      }
      case "heat": {
        if (len(layer.values) !== layer.nx * layer.ny) {
          issues.push(`${tag} values 长度 ${len(layer.values)} ≠ nx*ny = ${layer.nx * layer.ny}`);
        }
        if (!KNOWN_CMAPS.has(layer.cmap)) issues.push(`${tag} 未知色标 ${String(layer.cmap)}`);
        if (!(layer.vmax > layer.vmin)) issues.push(`${tag} 需要 vmax > vmin`);
        if (layer.logScale && layer.vmin <= 0) issues.push(`${tag} 对数色标要求 vmin > 0`);
        gridLayerIssue(tag, layer);
        break;
      }
      case "contour": {
        if (len(layer.values) !== layer.nx * layer.ny) issues.push(`${tag} values 长度与 nx*ny 不符`);
        if (!Array.isArray(layer.levels) || layer.levels.length === 0) issues.push(`${tag} 至少一个等值线层级`);
        gridLayerIssue(tag, layer);
        break;
      }
      case "bars": {
        if (len(layer.x) !== len(layer.values)) issues.push(`${tag} x/values 长度不一致`);
        if (layer.errors !== undefined && len(layer.errors) !== len(layer.values)) issues.push(`${tag} errors 长度不符`);
        break;
      }
      case "vectorfield": {
        if (len(layer.u) !== layer.nx * layer.ny || len(layer.v) !== layer.nx * layer.ny) {
          issues.push(`${tag} u/v 长度必须等于 nx*ny`);
        }
        if (!(layer.stride >= 1)) issues.push(`${tag} stride 必须 ≥ 1`);
        gridLayerIssue(tag, layer);
        break;
      }
      case "loglogfit": {
        if (len(layer.h) !== len(layer.err)) issues.push(`${tag} h/err 长度不一致`);
        for (const key of ["measuredSlope", "theorySlope"] as const) {
          if (!Number.isFinite(layer[key])) issues.push(`${tag} ${key} 必须是有限数（规则 2 不允许缺省）`);
        }
        if (layer.seriesLabel.length === 0 || layer.theoryLabel.length === 0) {
          issues.push(`${tag} 必须同时标注实测曲线与理论参考线`);
        }
        break;
      }
      default:
        break;
    }
  }
  if (!Array.isArray(s.texts)) issues.push("texts 必须是数组（可以为空）");
  for (const txt of s.texts ?? []) {
    if (!Number.isFinite(txt.x) || !Number.isFinite(txt.y)) issues.push(`文本 ${txt.text.slice(0, 12)}… 坐标非有限`);
    if (!KNOWN_ROLES.has(txt.role)) issues.push(`文本角色未知：${String(txt.role)}`);
  }
  if (!Array.isArray(s.badges)) issues.push("badges 必须是数组");
  else {
    if (!s.badges.some((b) => b.kind === "source")) issues.push("缺少 source 徽标（规则 5：必须标明数据来源）");
    for (const b of s.badges) {
      if (!KNOWN_BADGES.has(b.kind)) issues.push(`徽标类型未知：${String(b.kind)}`);
      if (!b.text) issues.push("徽标文本不得为空");
    }
  }
  if (!s.overflow) issues.push("缺少 overflow 字段（规则 1）");
  else if (s.overflow.flag && s.overflow.note.trim().length === 0) {
    issues.push("overflow.flag 为真时必须给出 overflow.note（规则 1）");
  }
  if (typeof s.pitfalls === "string" && s.pitfalls.trim().length < 10) {
    issues.push("pitfalls 太短（规则 7：必须真的说清它会怎样骗你）");
  }
  return issues;
}

/** 校验失败即抛错（带全部问题清单）。 */
export function assertValidScene(scene: unknown): Scene {
  const issues = validateScene(scene);
  if (issues.length > 0) {
    throw new Error(`Scene 不合法（${issues.length} 处）：\n  - ${issues.join("\n  - ")}`);
  }
  return scene as Scene;
}

// --------------------------------------------------------------------------- //
// 构造器（章节 model 层只用这些，避免手写字段遗漏规则）
// --------------------------------------------------------------------------- //
export interface AxisOptions {
  xlim?: [number, number];
  ylim?: [number, number];
  xscale?: ScaleKind;
  yscale?: ScaleKind;
  xlabel?: string;
  ylabel?: string;
  title?: string;
  autoscaled?: boolean;
  grid?: boolean;
  frame?: boolean;
  xticks?: number[];
  yticks?: number[];
  tickTarget?: number;
}

/** 造一个坐标轴；缺省刻度和范围按 1-2-5 / 对数规则自动给出。 */
export function makeAxis(id: string, rect: [number, number, number, number], opts: AxisOptions = {}): Axis {
  const autoscaled = opts.autoscaled ?? false;
  const xscale: ScaleKind = opts.xscale ?? "linear";
  const yscale: ScaleKind = opts.yscale ?? "linear";
  let xlim = opts.xlim ?? [0, 1];
  let ylim = opts.ylim ?? [0, 1];
  const target = opts.tickTarget ?? Math.max(3, Math.round(rect[2] / 90));
  if (opts.xlim === undefined) {
    xlim = xscale === "log" ? [1e-3, 1] : niceLimits(xlim[0], xlim[1], target);
  }
  if (opts.ylim === undefined) {
    ylim = yscale === "log" ? [1e-16, 1] : niceLimits(ylim[0], ylim[1], target);
  }
  const yTarget = opts.tickTarget ?? Math.max(3, Math.round(rect[3] / 55));
  return {
    id,
    rect,
    xlim,
    ylim,
    xscale,
    yscale,
    xlabel: opts.xlabel ?? "",
    ylabel: opts.ylabel ?? "",
    title: opts.title ?? "",
    autoscaled,
    xticks: opts.xticks ?? niceTicks(xlim[0], xlim[1], target, { log: xscale === "log" }),
    yticks: opts.yticks ?? niceTicks(ylim[0], ylim[1], yTarget, { log: yscale === "log" }),
    grid: opts.grid ?? true,
    frame: opts.frame ?? true,
  };
}

/** 规则 3：把"误差场/残差场"副轴标准化，避免有人忘记配一条。 */
export function makeErrorAxis(
  id: string,
  rect: [number, number, number, number],
  opts: AxisOptions = {},
): Axis {
  return makeAxis(id, rect, {
    title: opts.title ?? "误差 / 残差",
    ylabel: opts.ylabel ?? "‖误差‖",
    yscale: opts.yscale ?? "log",
    autoscaled: opts.autoscaled ?? true,
    ...opts,
  });
}

export function lineLayer(
  axes: string,
  series: LineSeries[] | LineSeries,
): Layer {
  return { kind: "line", axes, series: Array.isArray(series) ? series : [series] };
}

export function series(x: ArrayLike<number>, y: ArrayLike<number>, color: string, opts: Partial<LineSeries> = {}): LineSeries {
  return { x, y, color, width: opts.width ?? 1.6, ...opts };
}

export function bandLayer(axes: string, band: BandSpec): Layer {
  return { kind: "band", axes, band };
}

export function scatterLayer(
  axes: string,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  opts: { r?: number; color?: string; label?: string; alpha?: number } = {},
): Layer {
  return { kind: "scatter", axes, x, y, r: opts.r ?? 2.5, color: opts.color ?? "#4c78a8", label: opts.label, alpha: opts.alpha };
}

export function heatLayer(
  axes: string,
  values: ArrayLike<number>,
  nx: number,
  ny: number,
  extent: [number, number, number, number],
  opts: { cmap?: CmapName; vmin?: number; vmax?: number; logScale?: boolean; clipBelow?: number; label?: string } = {},
): Layer {
  let vmin = opts.vmin;
  let vmax = opts.vmax;
  if (vmin === undefined || vmax === undefined) {
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < values.length; i += 1) {
      const v = values[i] as number;
      if (!Number.isFinite(v)) continue;
      if (opts.logScale && v <= 0) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      lo = 0;
      hi = 1;
    }
    vmin = vmin ?? lo;
    vmax = vmax ?? (hi > lo ? hi : lo + 1);
  }
  if (opts.logScale && vmin <= 0) vmin = Number.MIN_VALUE;
  return {
    kind: "heat",
    axes,
    nx,
    ny,
    values,
    extent,
    cmap: opts.cmap ?? "viridis",
    vmin,
    vmax,
    logScale: opts.logScale ?? false,
    clipBelow: opts.clipBelow,
    label: opts.label,
  };
}

export function contourLayer(
  axes: string,
  values: ArrayLike<number>,
  nx: number,
  ny: number,
  extent: [number, number, number, number],
  opts: { levels?: number[]; color?: string; width?: number; label?: string } = {},
): Layer {
  let levels = opts.levels;
  if (!levels) {
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < values.length; i += 1) {
      const v = values[i] as number;
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!Number.isFinite(lo)) {
      lo = 0;
      hi = 1;
    }
    levels = linspace(lo, hi, 7).slice(1, -1) as unknown as number[];
  }
  return { kind: "contour", axes, nx, ny, values, extent, levels, color: opts.color ?? "#222", width: opts.width ?? 1, label: opts.label };
}

export function barsLayer(
  axes: string,
  x: ArrayLike<number>,
  values: ArrayLike<number>,
  opts: { width?: number; color?: string; errors?: ArrayLike<number>; baseline?: number; label?: string } = {},
): Layer {
  return {
    kind: "bars",
    axes,
    x,
    values,
    width: opts.width ?? 0.8,
    color: opts.color ?? "#4c78a8",
    errors: opts.errors,
    baseline: opts.baseline ?? 0,
    label: opts.label,
  };
}

export function vectorFieldLayer(
  axes: string,
  u: ArrayLike<number>,
  v: ArrayLike<number>,
  nx: number,
  ny: number,
  extent: [number, number, number, number],
  opts: { stride?: number; color?: string; scale?: number } = {},
): Layer {
  return {
    kind: "vectorfield",
    axes,
    nx,
    ny,
    u,
    v,
    extent,
    stride: Math.max(1, Math.round(opts.stride ?? 1)),
    color: opts.color ?? "#666",
    scale: opts.scale ?? 1,
  };
}

/**
 * 规则 2：双对数拟合层。`measuredSlope` 与 `theorySlope` 都是必填——
 * 少了任何一条，这个构造函数就会拒绝工作（而不是悄悄画一条光秃秃的误差曲线）。
 */
export function loglogFitLayer(
  axes: string,
  h: ArrayLike<number>,
  err: ArrayLike<number>,
  measuredSlope: number,
  theorySlope: number,
  opts: { seriesLabel?: string; theoryLabel?: string; color?: string; theoryColor?: string } = {},
): Layer {
  if (!Number.isFinite(measuredSlope) || !Number.isFinite(theorySlope)) {
    throw new Error("loglogFitLayer：实测斜率与理论斜率都必须是有限数（规则 2）");
  }
  let x0 = Number.POSITIVE_INFINITY;
  let x1 = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < h.length; i += 1) {
    const v = h[i] as number;
    if (!Number.isFinite(v) || v <= 0) continue;
    x0 = Math.min(x0, v);
    x1 = Math.max(x1, v);
  }
  if (!Number.isFinite(x0) || !Number.isFinite(x1) || x0 === x1) {
    throw new Error("loglogFitLayer：h 必须包含至少两个不同的正值");
  }
  return {
    kind: "loglogfit",
    axes,
    h,
    err,
    measuredSlope,
    theorySlope,
    refX: [x0, x1],
    seriesLabel: opts.seriesLabel ?? `实测斜率 ${measuredSlope.toFixed(2)}`,
    theoryLabel: opts.theoryLabel ?? `理论斜率 ${theorySlope.toFixed(2)}`,
    color: opts.color,
    theoryColor: opts.theoryColor,
  };
}

// --------------------------------------------------------------------------- //
// 文本与徽标
// --------------------------------------------------------------------------- //
export function text(x: number, y: number, body: string, opts: Partial<TextSpec> = {}): TextSpec {
  return { x, y, text: body, size: opts.size ?? 12, color: opts.color ?? "#222", align: opts.align ?? "left", role: opts.role ?? "note", maxWidth: opts.maxWidth };
}

export function badge(kind: BadgeKind, body: string, x = 8, y = 8): Badge {
  return { kind, text: body, x, y };
}

/** 规则 5 的标准数据来源徽标。 */
export function sourceBadge(dataSource: string, extra = ""): Badge {
  const label = dataSource.startsWith("backend+preview")
    ? `数据：后端帧 + 前端近似预览${extra ? "（" + extra + "）" : ""}`
    : `数据：后端 .nlf 帧${extra ? "（" + extra + "）" : ""}`;
  return { kind: dataSource.startsWith("backend+preview") ? "preview" : "source", text: label, x: 8, y: 8, color: "#2b6cb0" };
}

/** 规则 4 的自适应/固定坐标轴徽标。 */
export function axisModeBadge(autoscaled: boolean, x = 8, y = 26): Badge {
  return {
    kind: "autoscale",
    text: autoscaled ? "坐标轴：自适应（可能掩盖发散）" : "坐标轴：固定",
    x,
    y,
    color: autoscaled ? "#b7791f" : "#4a5568",
  };
}

/** 规则 1 的溢出徽标。 */
export function overflowBadge(note: string, x = 8, y = 44): Badge {
  return { kind: "overflow", text: `溢出：${note}`, x, y, color: "#c53030" };
}

/**
 * 规则 6 的临界值徽标（参数越过理论临界值时必须现身）。
 *
 * `beyond` 必须由调用方明确给出方向——"越界"的方向无法从数值本身推断：
 *  - `"above"`（缺省）：常见于 CFL、Re/Re_c、ρ(J)、Δt/Δt_max 等"不得超过"的量；
 *  - `"below"`：下界型约束（如阻尼比不得小于某值、每波长至少 2 个采样点）；
 *  - `"magnitude"`：比较绝对值（如谱半径）。
 *
 * 给出的 `critical` 可以是一族阈值（如 [0.25, 0.5, 1]）；徽标会标出**当前起支配作用**的那一个。
 */
export function criticalBadge(
  name: string,
  value: number,
  critical: number | number[],
  opts: { unit?: string; beyond?: "above" | "below" | "magnitude" } = {},
): Badge {
  const list = (Array.isArray(critical) ? critical : [critical]).slice().sort((a, b) => a - b);
  if (list.length === 0 || !Number.isFinite(value)) {
    return { kind: "critical", text: `${name} = ${String(value)}（无可比临界值）`, x: 8, y: 62, color: "#718096" };
  }
  const beyond = opts.beyond ?? "above";
  let governing: number;
  let crossed: boolean;
  if (beyond === "above") {
    governing = list.find((c) => value > c) ?? (list[0] as number); // 已被突破的最小阈值（最紧的那条）
    crossed = value > governing;
  } else if (beyond === "below") {
    governing = [...list].reverse().find((c) => value < c) ?? (list[list.length - 1] as number);
    crossed = value < governing;
  } else {
    governing = list.reduce((a, b) => (Math.abs(b) < Math.abs(a) ? b : a), list[0] as number);
    crossed = Math.abs(value) > Math.abs(governing);
  }
  const shown = list.map((c) => (c === governing ? `${c.toPrecision(4)}*` : c.toPrecision(4))).join(" / ");
  return {
    kind: "critical",
    text: `${name} = ${value.toPrecision(4)}${opts.unit ?? ""}　理论临界 ${shown}${crossed ? "　⚠ 已越界" : ""}`,
    x: 8,
    y: 62,
    color: crossed ? "#c53030" : "#2f855a",
  };
}

// --------------------------------------------------------------------------- //
// 便捷构造
// --------------------------------------------------------------------------- //
export interface SceneInit {
  chapter: string;
  viz: string;
  title: string;
  width?: number;
  height?: number;
  t?: number;
  dataSource?: string;
  pitfalls: string;
}

/** 起一个空白 Scene（含背景与默认徽标槽）。 */
export function newScene(init: SceneInit): Scene {
  return {
    version: 1,
    chapter: init.chapter,
    viz: init.viz,
    title: init.title,
    t: init.t ?? 0,
    canvas: { w: init.width ?? 900, h: init.height ?? 520, background: "#ffffff" },
    axes: [],
    layers: [],
    texts: [],
    badges: [],
    overflow: { flag: false, note: "" },
    pitfalls: init.pitfalls,
    dataSource: init.dataSource ?? "backend",
  };
}

/** 稳定序列化（键序固定），供快照与视觉回归使用。 */
export function sceneToJson(scene: Scene): string {
  const norm = (v: unknown): unknown => {
    if (ArrayBuffer.isView(v)) {
      const arr = Array.from(v as unknown as ArrayLike<number>);
      return arr.map((x) => (Object.is(x, -0) ? 0 : Number.isFinite(x) ? Number(x.toPrecision(12)) : String(x)));
    }
    if (Array.isArray(v)) return v.map(norm);
    if (v && typeof v === "object") {
      const out: Record<string, unknown> = {};
      for (const k of Object.keys(v as Record<string, unknown>).sort()) {
        const val = (v as Record<string, unknown>)[k];
        if (val !== undefined) out[k] = norm(val);
      }
      return out;
    }
    if (typeof v === "number") return Number.isFinite(v) ? Number(v.toPrecision(12)) : String(v);
    return v;
  };
  return JSON.stringify(norm(scene), null, 1);
}

export { linspace, logspace, niceTicks, niceLimits };
