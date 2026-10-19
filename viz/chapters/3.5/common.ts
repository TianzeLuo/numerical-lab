/**
 * 3.5 章各 model 共用的**显示几何**工具（纯函数、不碰 DOM、不重算物理量）。
 *
 * 为什么投影放在前端，而不是直接用帧里的 `*_u` / `*_v`
 * --------------------------------------------------
 * 帧里**同时**给了原始 x/y/z 与后端在"它那一刻的角度"下算出的 u/v。角度是参数
 * （manifest 的 `azimuth` / `elevation`），所以拖动角度时新帧还没到的那一瞬间，
 * 前端可以拿同一批原始 x/y/z 在显示空间重投影，图不会卡住不动。投影是**像素摆放**
 * （`model.ts` 文件头第 2 条明确允许的显示几何：线性插值放游标、按有限值分段、
 * 由数据算 1-2-5 取整的轴范围，都属于这一类），不是物理量。
 *
 * 公式与后端 `numcore/ode/dynamics.py::projection_matrix` 逐项相同（帧的
 * `strings.projection` 也把它写成了字面公式）：
 *
 * ```text
 * u = -sin(az)·x + cos(az)·y
 * v = -cos(az)·sin(el)·x - sin(az)·sin(el)·y + cos(el)·z
 * ```
 *
 * 没有透视除法：远近不改变尺寸，深度在二维图上不可读（帧的
 * `strings.projection_note` 就是这么说的，模型必须把这句话带到画面上）。
 * 同一角度下本文件算出的 u/v 与帧里的 `traj*_u/v`、`orbit_u/v`、`section_u/v`
 * 必须逐点一致——`viz/tests/chapter35.test.ts` 用真帧钉住这条对拍。
 */

import type { Frame } from "../../shared/src/frame.js";
import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { metaNumber, paramNumber, type ModelOptions, type Rect } from "../../shared/src/model.js";
import { text, textBoxOf, type TextSpec, type TextRole } from "../../shared/src/scene.js";

/** manifest 里 `azimuth` / `elevation` 的缺省值（回落用，见 {@link projectionOf}）。 */
export const DEFAULT_AZIMUTH = -60;
export const DEFAULT_ELEVATION = 20;

/** 角度的来源：帧 `meta` 优先（那是这一帧真正用的角度），缺了才回落并如实标注。 */
export type ProjectionSource = "meta" | "params" | "fallback";

/** 正交投影：把 (x,y,z) 摆到 (u,v) 两个像素方向的**线性**映射。 */
export interface OrthoProjection {
  azimuthDeg: number;
  elevationDeg: number;
  source: ProjectionSource;
  u(x: number, y: number): number;
  v(x: number, y: number, z: number): number;
}

/** 按角度造一个投影算子（显示几何；公式见文件头）。 */
export function orthoProjection(
  azimuthDeg: number,
  elevationDeg: number,
  source: ProjectionSource = "meta",
): OrthoProjection {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  const ca = Math.cos(az);
  const sa = Math.sin(az);
  const ce = Math.cos(el);
  const se = Math.sin(el);
  return {
    azimuthDeg,
    elevationDeg,
    source,
    u: (x: number, y: number): number => -sa * x + ca * y,
    v: (x: number, y: number, z: number): number => -ca * se * x - sa * se * y + ce * z,
  };
}

/**
 * 取这一帧该用的投影角度。
 *
 * 优先级是 **帧 `meta` > 请求参数 > 常量**，并把"用了请求值"的事实写进 `degraded`
 * （`model.ts` 的纪律：不拿 UI 请求参数冒充实测值，用了就在图上说明）。
 * `azimuth` / `elevation` 是"这一帧画在什么角度下"的标注，不是物理量。
 */
export function projectionOf(frame: Frame, opts: ModelOptions, degraded: string[]): OrthoProjection {
  const azMeta = metaNumber(frame, "azimuth");
  const elMeta = metaNumber(frame, "elevation");
  if (azMeta !== undefined && elMeta !== undefined) {
    return orthoProjection(azMeta, elMeta, "meta");
  }
  const azReq = paramNumber(opts.params, "azimuth");
  const elReq = paramNumber(opts.params, "elevation");
  if (azReq !== undefined && elReq !== undefined) {
    degraded.push(
      `帧 meta 缺 azimuth/elevation（${azMeta === undefined ? "azimuth " : ""}${elMeta === undefined ? "elevation" : ""}空白）：` +
        `投影按请求角 ${azReq}°/${elReq}° 画，并标注为请求值`,
    );
    return orthoProjection(azReq, elReq, "params");
  }
  degraded.push(
    `帧 meta 与请求参数都没有 azimuth/elevation：投影用缺省角 ${DEFAULT_AZIMUTH}°/${DEFAULT_ELEVATION}°（标注为缺省）`,
  );
  return orthoProjection(DEFAULT_AZIMUTH, DEFAULT_ELEVATION, "fallback");
}

/** 三条同长数组的投影结果（非有限值原样保留，交给分段逻辑断开）。 */
export function projectInto(
  proj: OrthoProjection,
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  zs: ArrayLike<number>,
): { u: Float64Array; v: Float64Array } {
  const n = Math.min(xs.length, ys.length, zs.length);
  const u = new Float64Array(n);
  const v = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const x = xs[i] as number;
    const y = ys[i] as number;
    const z = zs[i] as number;
    u[i] = proj.u(x, y);
    v[i] = proj.v(x, y, z);
  }
  return { u, v };
}

/**
 * 取二维数组 `(rows, cols)` 的第 `col` 列（行主序缓冲；`Frame.matrix()` 的口径）。
 *
 * 列不存在或行数与期望不符时返回 `undefined`——调用方据此降级（不猜、不错位）。
 */
export function columnOf(frame: Frame, name: string, col: number, expectRows?: number): Float64Array | undefined {
  if (!frame.has(name)) return undefined;
  const shape = frame.shapeOf(name);
  if (shape.length !== 2) return undefined;
  const rows = shape[0] as number;
  const cols = shape[1] as number;
  if (col < 0 || col >= cols) return undefined;
  if (expectRows !== undefined && rows !== expectRows) return undefined;
  const flat = frame.array(name);
  const out = new Float64Array(rows);
  for (let r = 0; r < rows; r += 1) out[r] = Number(flat[r * cols + col] as number);
  return out;
}

/** 一段连续的可绘下标区间 `[start, end)`。 */
export interface IndexRun {
  start: number;
  end: number;
}

/**
 * 按**分段号**把点切成若干段（`null_x_seg` / `null_y_seg` 这类整数标号）。
 *
 * 为什么必须按标号切：零倾线在渐近线处不连续（van der Pol 的 dy/dt=0 在 x=±1
 * 有竖直渐近线），把两支连成一条会画出"箭头往右、零倾线却说 dx/dt=0"的假曲线——
 * 这正是本章第三个陷阱。标号变化 = 断开。
 */
export function segmentRuns(seg: ArrayLike<number>): IndexRun[] {
  const runs: IndexRun[] = [];
  let start = 0;
  for (let i = 1; i <= seg.length; i += 1) {
    const changed = i === seg.length || !Object.is(seg[i], seg[start]);
    if (changed) {
      if (i > start) runs.push({ start, end: i });
      start = i;
    }
  }
  return runs;
}

/** 等比例的轴范围：`xlim` 与 `ylim` 的**宽度逐位相等**（三维投影必须等比例）。 */
export function equalSpanLimits(
  us: ArrayLike<number>,
  vs: ArrayLike<number>,
  target = 5,
): { xlim: [number, number]; ylim: [number, number] } {
  let u0 = Number.POSITIVE_INFINITY;
  let u1 = Number.NEGATIVE_INFINITY;
  let v0 = Number.POSITIVE_INFINITY;
  let v1 = Number.NEGATIVE_INFINITY;
  const n = Math.min(us.length, vs.length);
  for (let i = 0; i < n; i += 1) {
    const u = us[i] as number;
    const v = vs[i] as number;
    if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
    if (u < u0) u0 = u;
    if (u > u1) u1 = u;
    if (v < v0) v0 = v;
    if (v > v1) v1 = v;
  }
  if (!Number.isFinite(u0) || !Number.isFinite(v0)) {
    return { xlim: [-1, 1], ylim: [-1, 1] };
  }
  const cu = (u0 + u1) / 2;
  const cv = (v0 + v1) / 2;
  const span = Math.max(u1 - u0, v1 - v0, 1e-9) * 1.12;
  const [a, b] = niceLimits(cu - span / 2, cu + span / 2, target);
  const w = b - a;
  return { xlim: [a, b], ylim: [cv - w / 2, cv + w / 2] };
}

/** 等比例轴自己的刻度（两边都用同一个宽度，刻度才对齐）。 */
export function equalSpanTicks(lim: [number, number], target = 5): number[] {
  return niceTicks(lim[0], lim[1], target);
}

/** 三维世界的固定窗口（**与数据无关**，见 {@link projectedBoxLimits}）。 */
export interface WorldBox {
  x: [number, number];
  y: [number, number];
  z: [number, number];
}

/** 把三维窗口的 8 个角投影出来的 `(u, v)`。 */
export function projectBoxCorners(proj: OrthoProjection, box: WorldBox): { u: Float64Array; v: Float64Array } {
  const us = new Float64Array(8);
  const vs = new Float64Array(8);
  let k = 0;
  for (const x of box.x) {
    for (const y of box.y) {
      for (const z of box.z) {
        us[k] = proj.u(x, y);
        vs[k] = proj.v(x, y, z);
        k += 1;
      }
    }
  }
  return { u: us, v: vs };
}

/**
 * **固定**三维窗口的等比例轴范围：投影立方体的 8 个角 → 等跨度 1-2-5 取整。
 *
 * 为什么用"世界的固定盒子"而不是"本帧数据的包围盒"：`manifest` 把这个轴声明成
 * `autoscaled: false`（规则 3），数据包围盒会随参数与角度变，那就成了自适应轴——
 * 于是"换个 ρ 之后吸引子变大"这件事会被悄悄抹平。固定盒子（由 manifest 的参数
 * 范围标定，写在角标/文本里）保证**同一参数下画面不抖、不同参数下大小可比**；
 * 数据超出盒子时 `checkOverflow` 会当场标出溢出（规则 1），不会假装没发生。
 */
export function projectedBoxLimits(
  proj: OrthoProjection,
  box: WorldBox,
  target = 5,
): { xlim: [number, number]; ylim: [number, number]; ticks: number[] } {
  const { u, v } = projectBoxCorners(proj, box);
  const span = equalSpanLimits(u, v, target);
  return { ...span, ticks: equalSpanTicks(span.xlim, target) };
}

// --------------------------------------------------------------------------- //
// 3.5 章的面板排版（在 stackedPanels / columnsLayout 之上加一条"误差窄条"）
// --------------------------------------------------------------------------- //

/** 3.5 章各 model 用的面板矩形集合（单位 px）。 */
export interface Layout35 {
  left: number;
  right: number;
  contentW: number;
  /** 面板区顶端（= `main[1]`）。 */
  top: number;
  /** 面板区底端（窄条**之上**；主/副轴的下装饰带必须落在这里以上）。 */
  bottom: number;
  /** 整块内容区的底端（含窄条；再往下是渲染层现画的 pitfalls 面板）。 */
  bottomFull: number;
  main: Rect;
  error: Rect;
  /** 右列（与左列同高）。 */
  side: Rect;
  /** 底部窄条（`strip: false` 时为 `undefined`）。 */
  strip: Rect | undefined;
}

export interface Layout35Options {
  /** 左列宽度占内容宽的比例（只在 `columns: true` 时有意义）；缺省 0.58。 */
  share?: number;
  /** 右列与左列之间的间距（要放下右列的 y 刻度与轴名）；缺省 58。 */
  columnGap?: number;
  /** 主面板占（面板区高度 − 间距）的比例；缺省 0.56。 */
  mainShare?: number;
  /** 主/副轴之间的间距；缺省 22。放得下主轴的刻度带（15）就够，要放轴名请给 ≥ 47。 */
  gap?: number;
  /** 是否留底部窄条。 */
  strip?: boolean;
  /** 窄条高度占内容高比例；缺省 0.16（夹在 [42, 58]）。 */
  stripShare?: number;
  /** 底部总留白；缺省 `8 + 60 + 34`（与 `stackedPanels` 同口径）。 */
  footer?: number;
  /** 是否分左右两列（`columnsLayout` 的口径）。 */
  columns?: boolean;
}

/**
 * 3.5 章通用的面板排版。
 *
 * 与 `stackedPanels` 的差异只有一条：可以在底部再让出一条**误差/残差窄条**
 * （`lorenz_butterfly` 的两种步长相对差、`poincare_section` 的截面残差），
 * 窄条上方留 30px 给上面板的下装饰带。所有尺寸都由画布算出、与帧无关
 * （连续帧的面板位置逐像素相同，动画里不会跳）。
 */
export function layout35(width: number, height: number, opts: Layout35Options = {}): Layout35 {
  const left = 64;
  const right = 24;
  const footer = opts.footer ?? 8 + 60 + 34;
  // `+18`：轴的标题带画在 `rect[1]−18`，不留这 18px 它会落进角标区（角标最长到 headerBand−9）
  const top = headerBandOf(width) + 8 + 18;
  const contentW = Math.max(120, width - left - right);
  const bottomFull = Math.max(top + 120, height - footer);
  const withStrip = opts.strip ?? false;
  const stripH = withStrip
    ? clampInt(Math.round((bottomFull - top) * (opts.stripShare ?? 0.16)), 42, 58)
    : 0;
  const stripGap = withStrip ? 30 : 0;
  const bottom = bottomFull - stripH - stripGap;
  const plotH = Math.max(80, bottom - top);
  const gap = opts.gap ?? 22;
  const mainShare = opts.mainShare ?? 0.56;

  const fullW = opts.columns ?? false;
  const colGap = opts.columnGap ?? 58;
  const leftW = fullW ? Math.max(160, Math.round(contentW * (opts.share ?? 0.58))) : contentW;
  const sideW = fullW ? Math.max(120, contentW - leftW - colGap) : 0;
  const mainH = Math.max(56, Math.round(plotH * mainShare));
  const errorH = Math.max(40, plotH - mainH - gap);

  return {
    left,
    right,
    contentW,
    top,
    bottom,
    bottomFull,
    main: [left, top, leftW, mainH],
    error: [left, top + mainH + gap, leftW, errorH],
    side: fullW ? [left + leftW + colGap, top, sideW, plotH] : [left, top, contentW, plotH],
    strip: withStrip ? [left, bottomFull - stripH, contentW, stripH] : undefined,
  };
}

/** 与 `model.ts::headerBand` 同口径（本地放一份，免得每个 model 都 import 它）。 */
/**
 * 角标文字的**字数上限**（单位 = 半角；CJK 算 2）。
 *
 * 排版契约按 7px/单位算，但 `tools/raster.py` 画角标用的是 `base_font_size=13`，
 * 那一档 CJK 是 8px、ASCII 是 6px——所以 47 个单位的角标在 PNG 里会被框裁掉半句。
 * 这里按"框宽 ÷ 8"给上限：两列（`badgeColumns=2`）时是 344/8=43，留一点余量取 40。
 */
export function badgeUnitCap(width: number): number {
  const cols = width >= 712 ? 2 : 1;
  return cols >= 2 ? 40 : Math.max(40, Math.floor((width - 24) / 8));
}

export function headerBandOf(width: number): number {
  const rows = width >= 2 * 344 + 24 ? 4 : 8;
  return 34 + rows * 18 + 8;
}

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(v)));
}

/** 文本块折行后的盒高（与渲染层同一把尺子 `textBoxOf`）——自下而上码字时用它让位，别自己叠自己。 */
export function textHeight(body: string, size: number, maxWidth: number): number {
  return textBoxOf({ x: 0, y: 0, text: body, size, maxWidth }).h;
}

/** 一段说明文字（`stackTexts` 的输入）。 */
export interface TextBlock {
  body: string;
  color: string;
  size?: number;
  role?: TextRole;
  /** 与下一块之间的间距；缺省 4。 */
  gap?: number;
}

/**
 * 在 `[x, x+width] × [top, bottom]` 里**逐块量高**地码放说明文字。
 *
 * 为什么不用固定行距：折行数由 `textBoxOf` 决定，而折行数又由文本长度、字号、列宽
 * 一起决定——按 `i * 15` 摆，长句一折行就和下一块叠住，`resolveTextOverlaps` 会
 * 把叠住的那块**丢掉**（只在画布顶部留一行"另有 N 行未显示"）。丢内容比排版难看
 * 严重得多，所以这里先量后摆，放不下的块**显式记账**交给调用方写进降级说明。
 *
 * 返回的每块都保证：不与自己这一列的任何一块相交（`>2px` 才算相交，与校验器同口径）。
 */
export function stackTexts(
  x: number,
  top: number,
  width: number,
  bottom: number,
  blocks: readonly TextBlock[],
): { texts: TextSpec[]; dropped: string[] } {
  const texts: TextSpec[] = [];
  const dropped: string[] = [];
  let cursor = top;
  for (const b of blocks) {
    const size = b.size ?? 10;
    const h = textHeight(b.body, size, width);
    if (cursor + h > bottom) {
      dropped.push(b.body);
      continue;
    }
    texts.push(text(x, cursor, b.body, { role: b.role ?? "note", size, color: b.color, maxWidth: width }));
    cursor += h + (b.gap ?? 4);
  }
  return { texts, dropped };
}

/** 文本块折行后的盒宽。 */
export function textWidth(body: string, size: number, maxWidth: number): number {
  return textBoxOf({ x: 0, y: 0, text: body, size, maxWidth }).w;
}

/** 按 `(worldWidth : worldHeight)` 把矩形收成"等比例"的（像素尺度两轴相等）。 */
export function fitAspect(rect: Rect, worldW: number, worldH: number): Rect {
  if (!(worldW > 0) || !(worldH > 0)) return rect;
  const want = worldW / worldH;
  const have = rect[2] / rect[3];
  if (have > want) {
    const w = Math.max(60, Math.round(rect[3] * want));
    return [rect[0], rect[1], w, rect[3]];
  }
  const h = Math.max(60, Math.round(rect[2] / want));
  return [rect[0], rect[1], rect[2], h];
}
