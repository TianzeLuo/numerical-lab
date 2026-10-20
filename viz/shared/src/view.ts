/**
 * 一个可视化实验的完整视图：图表、参数、时间轴、图例与阅读区。
 *
 * 画布之上只放图形：模型原本塞进 Scene 的标题、说明、徽标与长文由
 * `sceneForDisplay()` 移出画布，改由这里用 HTML 排版。这样视觉回归与网页看到的是
 * 同一张干净图，而网页又不丢掉解释与读数。
 *
 * 纯逻辑仍在别处并有测试（framesource / timeline / controls / share / display）。
 * 本文件只负责把状态接到 DOM 与 canvas 上，不做数值判断。
 */

import type { Frame } from "./frame.js";
import type { ChapterManifestJson, VizSpecJson } from "./manifest.js";
import type { ModelOptions, SceneModel } from "./model.js";
import type { Scene, TextSpec } from "./scene.js";
import { renderScene } from "./render2d.js";
import { sceneForDisplay, type DisplaySceneResult } from "./display.js";
import { hoverAt, type HoverHit } from "./hover.js";
import { keyMetrics } from "./metrics.js";
import { ChapterFrameSource, type SourceStatus } from "./framesource.js";
import { Timeline, type TimelineSnapshot } from "./timeline.js";
import { describeControls, mountControls } from "./controls.js";
import { splitSentences } from "./pitfalls.js";
import { canvasPngBytes, downloadName, sceneToJsonText } from "./export.js";
import { buildHash, decodeHash, type ParamValues, type ShareState } from "./share.js";
import { renderRichText, texToMathML } from "./math.js";
import { el, icon, toast } from "./dom.js";

export interface SiteFormula {
  tex: string;
  caption: string;
}

export interface VizViewOptions {
  manifest: ChapterManifestJson;
  viz: VizSpecJson;
  model: SceneModel;
  /** 挂载点。 */
  host: HTMLElement;
  /** 站点根（`""` = 同源）。 */
  baseUrl?: string;
  width?: number;
  height?: number;
  dpr?: number;
  theme?: "light" | "dark";
  /** 初始参数（缺省从 manifest 的 default 取）。 */
  params?: ParamValues;
  /** 初始时刻（秒）。 */
  t?: number;
  /** 打开时是否直接播放（分享链接里可能带着这个状态）。 */
  playing?: boolean;
  /** 本章的核心公式（由站点从 data/site/formulas.json 注入）。 */
  formulas?: SiteFormula[];
  /** 第几个实验（从 1 开始）；与 total 一起用于标题上方的小字。 */
  ordinal?: number;
  total?: number;
  /** 注入数据源（测试/复用）；缺省内部创建。 */
  source?: ChapterFrameSource;
  /** 状态变化回调（站点用它记录当前视图）。 */
  onState?: (state: ShareState, status: SourceStatus) => void;
}

export const DEFAULT_WIDTH = 960;
export const DEFAULT_HEIGHT = 600;

/** 画面宽度的上下限：太窄时模型排不下，太宽时文字行会失去比例。 */
const LOGICAL_MIN_W = 460;
const LOGICAL_MAX_W = 1180;
/** 模型按 960×600 设计；网页只改变宽度，高度保持这一比例，避免面板被压扁。 */
const LOGICAL_HEIGHT = 600;

interface TimeDom {
  bar: HTMLElement;
  play: HTMLButtonElement;
  seek: HTMLInputElement | null;
  clock: HTMLElement;
  rate: HTMLSelectElement;
  dropped: HTMLElement;
}

function numberText(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Number.isInteger(value)) return String(value);
  return String(Number(value.toPrecision(4)));
}

function trimNumber(value: number): string {
  return numberText(value);
}

/** 去掉 Markdown 记号，用于标题旁的一句话导语。 */
function plainText(text: string): string {
  return text
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\$([^$]*)\$/g, "$1")
    .replace(/\\\(([^)]*)\\\)/g, "$1")
    .replace(/\*\*([^*]*)\*\*/g, "$1")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

/** 去掉正文里对内部规则编号的引用；这些编号对读者没有意义，也会让语气变得机械。 */
function soften(text: string): string {
  return text
    .replace(/[（(]\s*规则\s*[0-9]+(?:\s*\/\s*[0-9]+)?[^）)]*[）)]/g, "")
    .replace(/[（(]\s*(?:PROTOCOLS|POLICY)\s*[^）)]*[）)]/g, "")
    .replace(/规则\s*[0-9]+(?:\s*\/\s*[0-9]+)?/g, "")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\s+([，。；：、）】])/g, "$1")
    .replace(/[（(]\s*[）)]/g, "")
    .trim();
}

function firstSentence(text: string): string {
  const sentences = splitSentences(text);
  return plainText(sentences[0] ?? text);
}

function clip(text: string, max: number): string {
  const body = text.trim();
  if (body.length <= max) return body;
  return `${body.slice(0, max).replace(/[，。；、\s]+$/, "")}…`;
}

function timeModeText(mode: string, label?: string): string {
  if (mode === "stream") return label === undefined || label === "" ? "连续时间演化" : label;
  if (mode === "scrub") return label === undefined || label === "" ? "可逐帧拖动的时间轴" : label;
  return "静态视图";
}

function dataSourceText(source: string): string {
  return source.startsWith("backend+preview")
    ? "后端帧为主，缺帧时由前端在显示空间里做近似插值。"
    : "每一帧都由后端从参数出发算出，浏览器只负责绘制。";
}

interface LegendChip {
  label: string;
  color: string;
  shape: "line" | "scatter" | "band" | "bars" | "heat" | "contour" | "vertical";
  dash: boolean;
  axis?: string;
  gradient?: string;
}

const HINT_COLORS: Readonly<Record<string, string>> = {
  红: "var(--hint-red)",
  蓝: "var(--hint-blue)",
  绿: "var(--hint-green)",
  橙: "var(--hint-orange)",
  紫: "var(--hint-purple)",
  灰: "var(--hint-gray)",
  黑: "var(--hint-ink)",
  青: "var(--hint-teal)",
  黄: "var(--hint-yellow)",
  粉: "var(--hint-pink)",
  白: "var(--hint-white)",
};

/**
 * 有些模型把图例写成画布文字（“蓝=实测误差　红虚线=包络”）。
 * 画布文字被移出后，这里把这些颜色说明还原成 HTML 图例；只在模型没有给
 * 图层 label 时启用，避免同一张图出现两份图例。
 */
function legendFromTexts(texts: TextSpec[]): LegendChip[] {
  const out: LegendChip[] = [];
  const seen = new Set<string>();
  const tokenRe = /^([红蓝绿橙紫灰黑青黄粉白])(?:色)?\s*(实线|虚线|点线|竖线|圆点|实|虚|点|线)?\s*[=＝]\s*(.+)$/;
  const startsToken =
    /(?=(?:[红蓝绿橙紫灰黑青黄粉白](?:色)?\s*(?:实线|虚线|点线|竖线|圆点|实|虚|点|线)?\s*[=＝]))/;
  for (const item of texts) {
    if (item.role === "title") continue;
    for (const chunk of item.text.split(/[\u3000\n]+/)) {
      // 颜色说明常用空格而不是全角空格分隔；在每个“颜色[样式]=”前切开。
      for (const raw of chunk.split(startsToken)) {
        const match = tokenRe.exec(raw.trim());
        if (match === null) continue;
        const colorWord = match[1] ?? "";
        const styleWord = match[2] ?? "";
        const label = (match[3] ?? "").replace(/\*\*|__|`/g, "").replace(/\s+/g, " ").trim().replace(/[。；，,]\s*$/, "");
        if (label === "" || label.length > 64) continue;
        if (/画面放不下|画布放不下|另有/.test(label)) continue;
        const shape: LegendChip["shape"] = styleWord.includes("竖")
          ? "vertical"
          : styleWord.includes("圆点") || styleWord === "点"
            ? "scatter"
            : "line";
        const key = `${shape}\u0000${colorWord}\u0000${label}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({
          label: clip(label, 52),
          color: HINT_COLORS[colorWord] ?? "var(--muted)",
          shape,
          dash: styleWord.includes("虚") || styleWord.includes("点线"),
        });
        if (out.length >= 14) return out;
      }
    }
  }
  return out;
}

const HINT_HEX: Readonly<Record<string, string>> = {
  红: "#c53030",
  蓝: "#2b6cb0",
  绿: "#2f855a",
  橙: "#dd6b20",
  紫: "#805ad5",
  灰: "#718096",
  黑: "#1a202c",
  青: "#319795",
  黄: "#b7791f",
  粉: "#d53f8c",
  白: "#ffffff",
};

const COLOR_FAMILY: Readonly<Record<string, string>> = {
  "#b7791f": "#dd6b20",
  "#d69e2e": "#dd6b20",
};

function colorKey(color: string): string {
  const hint = /^var\(--hint-([^)]+)\)$/.exec(color);
  const raw = hint !== null ? (HINT_HEX[hint[1] ?? ""] ?? color) : color;
  const key = raw.toLowerCase();
  return COLOR_FAMILY[key] ?? key;
}

/**
 * 合并显式图层图例与画面文字里的颜色说明。
 *
 * 模型为了 PNG 的可独立阅读，常把“蓝=实测　红虚线=包络”写进画布文字；网页把文字
 * 移走后，这部分含义不能跟着消失。文字里的线条说明优先补进来，显式标签随后；
 * 同一颜色/形状已经由显式标签覆盖时不重复。
 */
function mergeLegend(explicit: LegendChip[], textual: LegendChip[]): LegendChip[] {
  // 图层已经有名字时，画布文字里的颜色说明通常只是同一批曲线的另一种说法。
  // 两套图例并排会出现“同色、同义、不同写法”的重复项，反而难核对；有显式标签就只信它。
  if (explicit.length > 0) {
    const seenExplicit = new Set<string>();
    const outExplicit: LegendChip[] = [];
    for (const chip of explicit) {
      const key = `${chip.shape}\u0000${colorKey(chip.color)}\u0000${chip.label}`;
      if (seenExplicit.has(key)) continue;
      seenExplicit.add(key);
      outExplicit.push(chip);
    }
    return [...outExplicit.filter((chip) => chip.shape !== "scatter"), ...outExplicit.filter((chip) => chip.shape === "scatter")];
  }
  const out: LegendChip[] = [];
  const seen = new Set<string>();
  const explicitColors = new Set(explicit.map((chip) => colorKey(chip.color)));
  const push = (chip: LegendChip): void => {
    const key = `${chip.shape}\u0000${colorKey(chip.color)}\u0000${chip.label}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(chip);
  };
  // 图层已带标签时，画布文字里的颜色说明通常只是同一批曲线的另一种说法。
  // 同色说明一律不重复补入；只有显式标签完全没覆盖的颜色才用文字补齐。
  for (const chip of textual) {
    if (explicitColors.has(colorKey(chip.color))) continue;
    push(chip);
  }
  for (const chip of explicit) push(chip);
  // 线条在前、散点在后：图例被截断时先保住“这条线是什么颜色”，而不是各方法的当前点。
  return [...out.filter((chip) => chip.shape !== "scatter"), ...out.filter((chip) => chip.shape === "scatter")];
}

/** 把 pitfalls 切成「一条一个提醒」。原文里的“第一/第二/第三”会转成独立段落。 */
function tipChunks(pitfalls: string): string[] {
  const text = soften(pitfalls ?? "").trim();
  if (text === "") return [];
  const marked = text
    .split(/(?=(?:第[一二三四五六七八九十]+|其[一二三四五六七八九十]+)[：:，,])/g)
    .map((part) => part.replace(/^(?:第[一二三四五六七八九十]+|其[一二三四五六七八九十]+)[：:，,]\s*/, "").trim())
    .filter((part) => part.length > 0);
  if (marked.length > 1) {
    // “三条都要命。”这类短促的开场白是段落标题，不是一条可以独立阅读的提醒。
    return marked.filter((part, index) => !(index === 0 && part.length < 20 && !/\d/.test(part)));
  }
  return splitSentences(text)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

export class VizView {
  readonly #opts: VizViewOptions;
  readonly #viz: VizSpecJson;
  readonly #model: SceneModel;
  readonly #canvas: HTMLCanvasElement;
  readonly #ctx: CanvasRenderingContext2D;
  readonly #timeline: Timeline;
  readonly #source: ChapterFrameSource;
  readonly #history: { t: number; frame: Frame }[] = [];
  #params: ParamValues;
  #theme: "light" | "dark";
  #width: number;
  #height: number;
  #frame: Frame | null = null;
  #frameT = 0;
  #scene: Scene | null = null;
  #display: DisplaySceneResult | null = null;
  #status: SourceStatus;
  #raf = 0;
  #disposed = false;
  #autoPlay = false;
  #lastAutoAt = 0;
  #resizeObserver: ResizeObserver | null = null;
  #resizeTimer = 0;
  #lastCssWidth = 0;
  #hadFrame = false;
  #initialT: number | null = null;

  #stage: HTMLElement | null = null;
  #loading: HTMLElement | null = null;
  #stageError: HTMLElement | null = null;
  #statusEl: HTMLElement | null = null;
  #sourceNote: HTMLElement | null = null;
  #metricRow: HTMLElement | null = null;
  #legendRow: HTMLElement | null = null;
  #badgeRow: HTMLElement | null = null;
  #guideList: HTMLElement | null = null;
  #figureCaption: HTMLElement | null = null;
  #time: TimeDom | null = null;
  #resetParams: (() => void) | null = null;
  #hoverCanvas: HTMLCanvasElement | null = null;
  #hoverTip: HTMLElement | null = null;
  #hoverHint: HTMLElement | null = null;
  #hoverRaf = 0;
  #hoverPending: { x: number; y: number } | null = null;
  #touchStart: { id: number; x: number; y: number; moved: boolean } | null = null;
  #hoverHideTimer = 0;
  #lastMetricKey = "";
  #legendColors: Record<string, string> = {};
  #lastLegendKey = "";
  #lastBadgeKey = "";
  #lastGuideKey = "";
  #sourceNoteLevel = "";
  #outsideClick: ((ev: PointerEvent) => void) | null = null;
  #fullscreenHandler: (() => void) | null = null;
  #errorToastAt = 0;

  constructor(opts: VizViewOptions) {
    this.#opts = opts;
    this.#viz = opts.viz;
    this.#model = opts.model;
    this.#theme = opts.theme ?? "light";
    this.#width = opts.width ?? DEFAULT_WIDTH;
    this.#height = opts.height ?? DEFAULT_HEIGHT;
    const normalized = describeControls(opts.viz, opts.params);
    this.#params = {};
    for (const spec of normalized) this.#params[spec.name] = spec.value;

    const canvas = document.createElement("canvas");
    canvas.className = "nl-canvas";
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("这个浏览器不支持 canvas 2D，无法绘制图形。");
    this.#canvas = canvas;
    this.#ctx = ctx;

    const time = opts.viz.time;
    this.#timeline = new Timeline({
      mode: time.mode,
      tMax: this.#timeMax(),
      dt: time.dt ?? 1,
      fps: time.fps ?? 30,
    });
    if (opts.t !== undefined) this.#timeline.seek(opts.t);

    this.#source =
      opts.source ??
      new ChapterFrameSource({
        chapter: opts.manifest.chapter,
        viz: opts.viz.id,
        baseUrl: opts.baseUrl,
        params: this.#params,
        fps: time.fps ?? 30,
        paramTypes: Object.fromEntries(opts.viz.params.map((p) => [p.name, p.type])),
        // scrub 只按需取当前时刻那一帧；订阅时预推整条时间轴会让首屏解码几十帧。
        time: { mode: time.mode, dt: time.dt, tMax: time.tMax, frames: 1 },
      });
    this.#status = this.#source.status;
  }

  get canvas(): HTMLCanvasElement {
    return this.#canvas;
  }

  get scene(): Scene | null {
    return this.#scene;
  }

  get frame(): Frame | null {
    return this.#frame;
  }

  get state(): ShareState {
    return {
      chapter: this.#opts.manifest.chapter,
      viz: this.#viz.id,
      params: this.#params,
      t: this.#timeline.t,
      playing: this.#autoPlay || this.#timeline.playing,
    };
  }

  /** 主题切换：模型用主题取色，所以需要按当前帧重画。 */
  setTheme(theme: "light" | "dark"): void {
    if (theme === this.#theme) return;
    this.#theme = theme;
    this.#renderCurrent();
  }

  /** 建 DOM、绑事件、连数据源。 */
  async mount(): Promise<void> {
    const host = this.#opts.host;
    host.innerHTML = "";
    host.classList.add("nl-viz");

    this.#buildHeader(host);

    const grid = el("div", { className: "nl-viz-grid" });
    const chart = this.#buildChartCard();
    const guide = this.#buildGuideStrip();
    if (this.#legendRow !== null) chart.insertBefore(guide, this.#legendRow);
    else chart.append(guide);
    const side = el("aside", { className: "nl-side" });
    side.append(this.#buildControlsCard());
    const workbench = el("div", { className: "nl-workbench" });
    workbench.append(chart, side);
    grid.append(workbench, this.#buildReading());
    host.append(grid);

    this.#renderStatus();
    this.#renderGuide();
    this.#measure();
    this.#buildTimebar();

    this.#source.on("frame", (frame, t) => this.#onFrame(frame, t));
    this.#source.on("status", (status) => {
      this.#status = status;
      this.#renderStatus();
      this.#emit();
    });
    this.#source.on("error", (err) => this.#onSourceError(err));

    this.#timeline.onChange((snap) => this.#onTick(snap));
    this.#tick();

    this.#resizeObserver = new ResizeObserver(() => this.#scheduleMeasure());
    if (this.#stage !== null) this.#resizeObserver.observe(this.#stage);

    this.#outsideClick = (ev: PointerEvent): void => {
      const target = ev.target as Node | null;
      for (const menu of host.querySelectorAll<HTMLDetailsElement>("details.nl-menu[open]")) {
        if (target === null || !menu.contains(target)) menu.open = false;
      }
    };
    document.addEventListener("pointerdown", this.#outsideClick);
    this.#fullscreenHandler = (): void => this.#syncFullscreenButton();
    document.addEventListener("fullscreenchange", this.#fullscreenHandler);

    this.#initialT = this.#timeline.t;
    await this.#source.connect();
    this.#source.request(this.#timeline.t);
    if (this.#viz.time.mode === "stream") this.#source.startStream();
    if (this.#opts.playing === true) {
      if (this.#viz.time.mode === "stream") this.#timeline.play();
      else if (this.#viz.time.mode === "scrub") this.#startScrubPlay();
    }
    this.#emit();
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    if (this.#raf !== 0) cancelAnimationFrame(this.#raf);
    this.#raf = 0;
    if (this.#hoverRaf !== 0) cancelAnimationFrame(this.#hoverRaf);
    this.#hoverRaf = 0;
    if (this.#hoverHideTimer !== 0) window.clearTimeout(this.#hoverHideTimer);
    this.#hoverHideTimer = 0;
    if (this.#resizeTimer !== 0) window.clearTimeout(this.#resizeTimer);
    this.#resizeObserver?.disconnect();
    this.#resizeObserver = null;
    if (this.#outsideClick !== null) document.removeEventListener("pointerdown", this.#outsideClick);
    if (this.#fullscreenHandler !== null) document.removeEventListener("fullscreenchange", this.#fullscreenHandler);
    this.#source.dispose();
    this.#opts.host.innerHTML = "";
    this.#opts.host.classList.remove("nl-viz");
  }

  // ------------------------------------------------------------------ 结构

  #buildHeader(host: HTMLElement): void {
    const head = el("header", { className: "nl-viz-head" });
    const heading = el("div", { className: "nl-viz-heading" });
    if (this.#opts.ordinal !== undefined) {
      const kicker = el("p", { className: "nl-viz-kicker" });
      kicker.textContent =
        this.#opts.total !== undefined
          ? `实验 ${String(this.#opts.ordinal).padStart(2, "0")} / ${String(this.#opts.total).padStart(2, "0")}`
          : `实验 ${String(this.#opts.ordinal).padStart(2, "0")}`;
      heading.append(kicker);
    }
    const title = el("h2", { className: "nl-viz-title", text: this.#viz.title, attrs: { tabindex: "-1" } });
    heading.append(title);

    const description = this.#viz.description ?? "";
    const leadText = description === "" ? firstSentence(this.#viz.pitfalls) : firstSentence(description);
    if (leadText !== "") {
      const lead = el("p", { className: "nl-viz-lead" });
      lead.textContent = clip(leadText, 118);
      heading.append(lead);
    }

    const actions = el("div", { className: "nl-viz-actions" });
    actions.append(this.#buildMenu());
    head.append(heading, actions);
    host.append(head);
  }

  #buildMenu(): HTMLElement {
    const menu = el("details", { className: "nl-menu" });
    const summary = el("summary", { className: "nl-btn nl-btn-quiet", attrs: { "aria-label": "导出与分享" } });
    summary.append(icon("download"), el("span", { text: "导出与分享" }), icon("chevron-down", "nl-chevron"));
    const panel = el("div", { className: "nl-menu-panel" });

    const png = el("button", { className: "nl-menu-item", attrs: { type: "button" } });
    png.append(icon("download"), el("span", { text: "保存当前画面 PNG" }));
    png.addEventListener("click", () => {
      menu.open = false;
      void this.#exportPng();
    });

    const json = el("button", { className: "nl-menu-item", attrs: { type: "button" } });
    json.append(icon("chart"), el("span", { text: "导出场景数据 JSON" }));
    json.addEventListener("click", () => {
      menu.open = false;
      this.#exportScene();
    });

    const share = el("button", { className: "nl-menu-item", attrs: { type: "button" } });
    share.append(icon("link"), el("span", { text: "复制带状态的分享链接" }));
    share.addEventListener("click", () => {
      menu.open = false;
      void this.#share();
    });

    const note = el("p", { className: "nl-menu-note" });
    note.textContent = "PNG 只包含图形；分享链接会保存参数与时刻。";
    panel.append(png, json, el("div", { className: "nl-menu-sep" }), share, note);
    menu.append(summary, panel);
    return menu;
  }

  #buildChartCard(): HTMLElement {
    const card = el("section", { className: "nl-chart-card nl-panel", attrs: { "aria-label": "交互图形" } });
    const top = el("div", { className: "nl-chart-top" });
    const caption = el("div", { className: "nl-chart-caption" });
    const fig = el("span", { className: "nl-figure-label", text: "图形" });
    this.#figureCaption = el("span", {
      className: "nl-figure-name",
      text: this.#viz.axes[0]?.title ?? this.#viz.title,
    });
    caption.append(fig, this.#figureCaption);

    const tools = el("div", { className: "nl-chart-tools" });
    this.#statusEl = el("span", { className: "nl-live" });
    const full = el("button", {
      className: "nl-icon-btn",
      attrs: { type: "button", "aria-label": "全屏查看图形", title: "全屏查看" },
    });
    full.append(icon("expand"));
    full.addEventListener("click", () => {
      if (document.fullscreenElement === card) void document.exitFullscreen();
      else if (typeof card.requestFullscreen === "function") {
        void card.requestFullscreen().catch(() => toast("这个浏览器不允许全屏显示。", "warn"));
      } else {
        toast("这个浏览器不支持全屏查看。", "warn");
      }
    });
    tools.append(this.#statusEl, full);
    top.append(caption, tools);

    const stage = el("div", { className: "nl-stage" });
    const frame = el("div", { className: "nl-canvas-frame" });
    const hoverCanvas = el("canvas", { className: "nl-hover-layer", attrs: { "aria-hidden": "true" } });
    this.#hoverCanvas = hoverCanvas;
    frame.append(this.#canvas, hoverCanvas);
    const hoverTip = el("div", { className: "nl-hover-tip", attrs: { role: "tooltip" } });
    hoverTip.hidden = true;
    this.#hoverTip = hoverTip;
    const loading = el("div", { className: "nl-stage-loading" });
    const spinner = el("span", { className: "nl-spinner", attrs: { "aria-hidden": "true" } });
    loading.append(spinner, el("span", { text: "正在生成第一帧…" }));
    const error = el("div", { className: "nl-stage-error" });
    error.hidden = true;
    stage.append(frame, loading, error, hoverTip);
    this.#stage = stage;
    this.#loading = loading;
    this.#stageError = error;

    const swipe = el("p", { className: "nl-stage-swipe" });
    swipe.append(
      icon("chevron-left"),
      el("span", { text: "图形较宽，左右滑动查看完整内容" }),
      icon("chevron-right"),
    );

    const hoverHint = el("p", { className: "nl-hover-hint" });
    hoverHint.append(icon("info"), el("span", { text: "悬停或轻点图中的曲线、点、柱，查看对应数值" }));
    hoverHint.hidden = true;
    this.#hoverHint = hoverHint;

    this.#sourceNote = el("p", { className: "nl-source-note" });
    this.#sourceNote.hidden = true;

    this.#metricRow = el("div", { className: "nl-metrics" });
    this.#metricRow.hidden = true;
    this.#legendRow = el("div", { className: "nl-legend" });
    this.#legendRow.hidden = true;
    this.#badgeRow = el("div", { className: "nl-badge-row" });
    this.#badgeRow.hidden = true;

    const timeSlot = el("div", { className: "nl-timebar-slot" });
    timeSlot.dataset["role"] = "timebar-slot";

    card.append(top, stage, swipe, hoverHint, this.#sourceNote, this.#metricRow, this.#legendRow, this.#badgeRow, timeSlot);
    this.#bindHover();
    return card;
  }

  #bindHover(): void {
    const canvas = this.#canvas;
    const stage = this.#stage;
    if (stage === null) return;
    canvas.addEventListener("pointermove", (ev) => {
      if (ev.pointerType === "touch") {
        const start = this.#touchStart;
        if (start === null) return;
        if (Math.hypot(ev.clientX - start.x, ev.clientY - start.y) > 10) {
          start.moved = true;
          this.#hideHover();
        }
        return;
      }
      this.#queueHover(ev.clientX, ev.clientY);
    });
    canvas.addEventListener("pointerleave", (ev) => {
      if (ev.pointerType !== "touch") this.#hideHover();
    });
    canvas.addEventListener("pointerdown", (ev) => {
      if (ev.pointerType !== "touch") return;
      if (this.#hoverHideTimer !== 0) window.clearTimeout(this.#hoverHideTimer);
      this.#touchStart = { id: ev.pointerId, x: ev.clientX, y: ev.clientY, moved: false };
    });
    canvas.addEventListener("pointerup", (ev) => {
      const start = this.#touchStart;
      if (ev.pointerType !== "touch" || start === null) return;
      this.#touchStart = null;
      if (start.moved) return;
      this.#queueHover(ev.clientX, ev.clientY);
      this.#hoverHideTimer = window.setTimeout(() => this.#hideHover(), 6000);
    });
    canvas.addEventListener("pointercancel", () => {
      this.#touchStart = null;
      this.#hideHover();
    });
    stage.addEventListener("scroll", () => this.#hideHover(), { passive: true });
  }

  #queueHover(clientX: number, clientY: number): void {
    this.#hoverPending = { x: clientX, y: clientY };
    if (this.#hoverRaf !== 0) return;
    this.#hoverRaf = requestAnimationFrame(() => {
      this.#hoverRaf = 0;
      const pending = this.#hoverPending;
      this.#hoverPending = null;
      if (pending !== null) this.#showHoverAt(pending.x, pending.y);
    });
  }

  #showHoverAt(clientX: number, clientY: number): void {
    const scene = this.#scene;
    if (scene === null) return;
    const rect = this.#canvas.getBoundingClientRect();
    if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) {
      this.#hideHover();
      return;
    }
    const px = ((clientX - rect.left) / Math.max(1, rect.width)) * scene.canvas.w;
    const py = ((clientY - rect.top) / Math.max(1, rect.height)) * scene.canvas.h;
    const labels: Record<string, { x: string; y: string }> = {};
    for (const axis of this.#display?.axes ?? []) {
      labels[axis.id] = { x: axis.xlabel, y: axis.ylabel };
    }
    const hit = hoverAt(scene, px, py, { labels, colorLabels: this.#legendColors });
    if (hit === null) {
      this.#hideHover();
      return;
    }
    this.#drawHover(hit);
    this.#showHoverTip(hit, clientX, clientY);
  }

  #drawHover(hit: HoverHit): void {
    const overlay = this.#hoverCanvas;
    if (overlay === null || this.#scene === null) return;
    const cssW = Math.max(1, Math.round(this.#canvas.clientWidth));
    const cssH = Math.max(1, Math.round(this.#canvas.clientHeight));
    const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    if (overlay.width !== Math.round(cssW * dpr) || overlay.height !== Math.round(cssH * dpr)) {
      overlay.width = Math.round(cssW * dpr);
      overlay.height = Math.round(cssH * dpr);
    }
    const ctx = overlay.getContext("2d");
    if (ctx === null) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const sx = overlay.width / this.#scene.canvas.w;
    const sy = overlay.height / this.#scene.canvas.h;
    ctx.setTransform(sx, 0, 0, sy, 0, 0);
    const [rx, ry, , rh] = hit.axisRect;
    ctx.save();
    ctx.strokeStyle = this.#theme === "dark" ? "rgba(203,213,224,0.55)" : "rgba(51,51,51,0.42)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(hit.px, ry);
    ctx.lineTo(hit.px, ry + rh);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(hit.px, hit.py, 5.5, 0, Math.PI * 2);
    ctx.strokeStyle = hit.color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(hit.px, hit.py, 2, 0, Math.PI * 2);
    ctx.fillStyle = hit.color;
    ctx.fill();
    ctx.restore();
    void rx;
  }

  #showHoverTip(hit: HoverHit, clientX: number, clientY: number): void {
    const tip = this.#hoverTip;
    const stage = this.#stage;
    if (tip === null || stage === null) return;
    const shape =
      hit.shape === "cell" ? "heat" : hit.shape === "bar" ? "bars" : hit.kind === "scatter" ? "scatter" : "line";
    const head = el("div", { className: "nl-hover-head" });
    const swatch = el("span", {
      className: `nl-swatch nl-swatch-${shape}${hit.dash ? " is-dashed" : ""}`,
    });
    swatch.style.setProperty("--swatch", hit.color);
    head.append(swatch, el("span", { className: "nl-hover-name", text: hit.label }));
    const values = el("p", { className: "nl-hover-values" });
    values.append(el("span", { text: hit.xText }), el("span", { text: hit.yText }));
    const parts: HTMLElement[] = [head, values];
    if (hit.detail !== undefined && hit.detail !== "") {
      parts.push(el("p", { className: "nl-hover-detail", text: hit.detail }));
    }
    tip.replaceChildren(...parts);
    tip.hidden = false;
    const stageRect = stage.getBoundingClientRect();
    const x = clientX - stageRect.left;
    const y = clientY - stageRect.top;
    const gap = 14;
    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    let left = x + gap;
    if (left + tw > stage.clientWidth - 8) left = x - tw - gap;
    left = Math.min(Math.max(8, left), Math.max(8, stage.clientWidth - tw - 8));
    let top = y - th - gap;
    if (top < 8) top = y + gap;
    top = Math.min(Math.max(8, top), Math.max(8, stage.clientHeight - th - 8));
    tip.style.left = `${left.toFixed(1)}px`;
    tip.style.top = `${top.toFixed(1)}px`;
  }

  #hideHover(): void {
    const tip = this.#hoverTip;
    if (tip !== null) tip.hidden = true;
    const overlay = this.#hoverCanvas;
    if (overlay !== null) {
      const ctx = overlay.getContext("2d");
      if (ctx !== null) {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
    }
    if (this.#hoverHideTimer !== 0) {
      window.clearTimeout(this.#hoverHideTimer);
      this.#hoverHideTimer = 0;
    }
  }

  #buildControlsCard(): HTMLElement {
    const card = el("section", { className: "nl-panel nl-controls-card" });
    const head = el("div", { className: "nl-card-head" });
    const titleBox = el("div");
    titleBox.append(
      el("p", { className: "nl-card-kicker", text: "可调参数" }),
      el("h3", { className: "nl-card-title", text: "参数调节" }),
    );
    const actions = el("div", { className: "nl-card-actions" });
    const reset = el("button", {
      className: "nl-reset-btn",
      attrs: { type: "button", title: "把所有参数恢复为默认值" },
    });
    reset.append(icon("reset"), el("span", { text: "恢复默认" }));
    reset.addEventListener("click", () => this.#resetParams?.());
    actions.append(reset);
    head.append(titleBox, actions);
    const slot = el("div", { className: "nl-controls-slot" });
    card.append(head, slot);
    const parent = slot;
    const controls = mountControls(
      this.#viz,
      parent,
      (_name, _value, params) => {
        this.#params = params;
        this.#history.length = 0;
        this.#source.setParams(params);
        this.#timeline.setRange(this.#timeMax());
        this.#source.request(this.#timeline.t);
        this.#emit();
      },
      this.#params,
    );
    this.#resetParams = controls.reset;
    return card;
  }

  #buildGuideStrip(): HTMLElement {
    const card = el("section", { className: "nl-chart-guide" });
    const head = el("div", { className: "nl-chart-guide-head" });
    head.append(
      el("span", { className: "nl-card-kicker", text: "读图" }),
      el("span", { className: "nl-chart-guide-note", text: "每个面板的横轴与纵轴" }),
    );
    this.#guideList = el("ol", { className: "nl-panel-list" });
    card.append(head, this.#guideList);
    return card;
  }

  #buildReading(): HTMLElement {
    const reading = el("section", { className: "nl-reading", attrs: { "aria-label": "实验说明" } });

    const about = el("section", { className: "nl-reading-block" });
    about.append(this.#sectionHead("01", "这张图在讲什么", "先从画面要回答的问题开始"));
    const prose = el("div", { className: "nl-prose" });
    const description = this.#viz.description ?? "";
    if (description.trim() !== "") {
      prose.innerHTML = renderRichText(soften(description));
    } else {
      const p = el("p", { text: firstSentence(this.#viz.pitfalls) });
      prose.append(p);
    }
    about.append(prose);

    const play = this.#playbook();
    if (play.length > 0) {
      const box = el("div", { className: "nl-playbook" });
      const title = el("p", { className: "nl-playbook-title", text: "建议先试这些操作" });
      const list = el("ul");
      for (const item of play) list.append(el("li", { text: item }));
      box.append(title, list);
      about.append(box);
    }
    reading.append(about);

    const theory = this.#viz.theory ?? [];
    if (theory.length > 0 || (this.#opts.formulas ?? []).length > 0) {
      const block = el("section", { className: "nl-reading-block" });
      block.append(this.#sectionHead("02", "原理与推导", "先看两条式子，再看它们对应的实验事实"));
      const formulas = this.#opts.formulas ?? [];
      if (formulas.length > 0) {
        block.append(el("p", { className: "nl-block-label", text: "关键公式" }));
        const grid = el("div", { className: "nl-reading-formulas" });
        for (const item of formulas) {
          const figure = el("figure", { className: "nl-formula" });
          const math = el("div", { className: "nl-formula-math" });
          math.innerHTML = texToMathML(item.tex, { display: true });
          const caption = el("figcaption", { text: item.caption });
          figure.append(math, caption);
          grid.append(figure);
        }
        block.append(grid);
      }
      if (theory.length > 0) {
        block.append(el("p", { className: "nl-block-label", text: "推导要点" }));
        const list = el("ul", { className: "nl-theory-list" });
        for (const line of theory) {
          const item = el("li");
          const body = el("div", { className: "nl-theory-text" });
          body.innerHTML = renderRichText(soften(line));
          item.append(body);
          list.append(item);
        }
        block.append(list);
      }
      reading.append(block);
    }

    const tips = tipChunks(this.#viz.pitfalls);
    if (tips.length > 0) {
      const block = el("section", { className: "nl-reading-block" });
      block.append(this.#sectionHead("03", "读图时，留意这些地方", "不是每个像素都值得相信——先看坐标轴、缩放与采样方式"));
      const shown = tips.slice(0, 3);
      const rest = tips.slice(3);
      block.append(this.#tipsList(shown));
      if (rest.length > 0) {
        const more = el("details", { className: "nl-more" });
        const summary = el("summary");
        summary.textContent = `继续读：还有 ${rest.length} 条提醒`;
        more.append(summary, this.#tipsList(rest));
        block.append(more);
      }
      reading.append(block);
    }

    const tech = el("details", { className: "nl-tech" });
    const summary = el("summary");
    summary.append(icon("info"), el("span", { text: "数据、坐标轴与技术细节" }));
    const body = el("div", { className: "nl-tech-body" });
    const dl = el("dl", { className: "nl-tech-grid" });
    const add = (term: string, value: string): void => {
      dl.append(el("dt", { text: term }), el("dd", { text: value }));
    };
    add("数据来源", dataSourceText(this.#viz.dataSource));
    add("时间轴", timeModeText(this.#viz.time.mode, this.#viz.time.label));
    add("误差 / 残差场", this.#viz.requiredAux.join("、"));
    add("坐标轴", this.#viz.axes.map((axis) => `${axis.title ?? axis.id}（${axis.scale === "log" ? "对数" : "线性"}${axis.autoscaled ? " · 自适应" : ""}）`).join("；"));
    add("分享", "链接里保存章节、实验、参数与时刻；PNG 只保存当前画面。");
    body.append(dl);
    const badgeTexts = (this.#display?.badges ?? [])
      .filter((badge) => badge.kind === "source" || badge.kind === "autoscale")
      .map((badge) => badge.text);
    if (badgeTexts.length > 0) {
      const badges = el("div", { className: "nl-tech-badges" });
      for (const text of badgeTexts) badges.append(el("span", { className: "nl-tech-badge", text }));
      body.append(badges);
    }
    tech.append(summary, body);
    reading.append(tech);

    return reading;
  }

  #sectionHead(index: string, title: string, sub: string): HTMLElement {
    const head = el("div", { className: "nl-section-head" });
    const num = el("span", { className: "nl-section-index", text: index });
    const box = el("div");
    box.append(el("h3", { className: "nl-section-title", text: title }), el("p", { className: "nl-section-sub", text: sub }));
    head.append(num, box);
    return head;
  }

  #tipsList(items: string[]): HTMLElement {
    const list = el("ul", { className: "nl-tips" });
    for (const text of items) {
      const item = el("li");
      const mark = el("span", { className: "nl-tip-mark", attrs: { "aria-hidden": "true" } });
      mark.append(icon("arrow-right"));
      const body = el("div", { className: "nl-tip-text" });
      body.innerHTML = renderRichText(text);
      item.append(mark, body);
      list.append(item);
    }
    return list;
  }

  #playbook(): string[] {
    const out: string[] = [];
    const mode = this.#viz.time.mode;
    if (mode === "scrub") {
      out.push("拖动图下方的滑轨，可以逐帧查看过程；也可以点播放，让时间自己走一遍。");
    } else if (mode === "stream") {
      out.push("点播放开始实时推进；随时可以暂停或改变速度。");
    } else {
      out.push("这是一张静态图。拖动参数，图形会立刻重画。");
    }
    if (this.#viz.params.length > 0) {
      out.push("参数改变后不需要重新加载页面，几块图会一起更新。");
    }
    if (this.#viz.time.tMaxParam !== undefined) {
      const cap = this.#viz.params.find((p) => p.name === this.#viz.time.tMaxParam);
      out.push(`时间轴的上限跟着「${cap?.label ?? this.#viz.time.tMaxParam}」走；把它调大，才能继续往后拖。`);
    }
    if (this.#viz.params.some((p) => (p.critical ?? []).length > 0)) {
      out.push("把参数拖过临界值，对比越界前后曲线形状与误差的变化。");
    }
    if (this.#viz.axes.some((axis) => axis.autoscaled)) {
      out.push("误差副图的纵轴会自动缩放，比较绝对大小时先看刻度。");
    }
    return out.slice(0, 4);
  }

  // ------------------------------------------------------------------ 时间轴

  #buildTimebar(): void {
    const slot = this.#opts.host.querySelector<HTMLElement>('[data-role="timebar-slot"]');
    if (slot === null) return;
    const mode = this.#viz.time.mode;
    if (mode === "none") {
      const note = el("div", { className: "nl-static-strip" });
      note.append(icon("info"), el("span", { text: "静态视图 · 调整参数即可重画" }));
      slot.append(note);
      return;
    }

    const bar = el("div", { className: "nl-timebar" });
    const play = el("button", {
      className: "nl-play-btn",
      attrs: { type: "button", "aria-label": "播放", title: "播放" },
    });
    play.append(icon("play"));
    play.addEventListener("click", () => {
      if (mode === "stream") this.#timeline.toggle();
      else this.#toggleScrubPlay();
    });

    const controls = el("div", { className: "nl-transport-actions" });
    if (mode === "scrub") {
      const back = el("button", {
        className: "nl-icon-btn nl-icon-btn-sm",
        attrs: { type: "button", "aria-label": "上一帧", title: "上一帧" },
      });
      back.append(icon("step-back"));
      back.addEventListener("click", () => {
        this.#stopScrubPlay();
        this.#timeline.step(-1);
      });
      const fwd = el("button", {
        className: "nl-icon-btn nl-icon-btn-sm",
        attrs: { type: "button", "aria-label": "下一帧", title: "下一帧" },
      });
      fwd.append(icon("step-forward"));
      fwd.addEventListener("click", () => {
        this.#stopScrubPlay();
        this.#timeline.step(1);
      });
      controls.append(back, fwd);
    }

    const seek = mode === "stream" ? null : document.createElement("input");
    if (seek !== null) {
      seek.type = "range";
      seek.className = "nl-range nl-seek";
      seek.min = "0";
      seek.max = String(this.#timeline.tMax);
      seek.step = String(this.#viz.time.dt ?? 0.01);
      seek.value = String(this.#timeline.t);
      seek.setAttribute("aria-label", this.#viz.time.label ?? "时间");
      seek.addEventListener("input", () => {
        this.#stopScrubPlay();
        this.#timeline.seek(Number(seek.value));
      });
    }

    const clock = el("div", { className: "nl-clock" });
    const rate = document.createElement("select");
    rate.className = "nl-rate";
    rate.setAttribute("aria-label", "播放速度");
    for (const value of [0.5, 1, 2, 4]) {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = `${value}×`;
      if (value === 1) option.selected = true;
      rate.append(option);
    }
    rate.addEventListener("change", () => this.#timeline.setRate(Number(rate.value)));
    const dropped = el("span", { className: "nl-dropped" });

    if (seek !== null) bar.append(play, seek, clock, rate, dropped);
    else bar.append(play, clock, rate, dropped);
    bar.append(controls);
    slot.append(bar);

    this.#time = { bar, play, seek, clock, rate, dropped };
    this.#renderTimelineBar(this.#timeline.snapshot());
  }

  /** 时间轴上限：声明值，或由 `time.tMaxParam` 指向的参数收紧。 */
  #timeMax(): number {
    const declared = this.#viz.time.tMax ?? 1;
    const name = this.#viz.time.tMaxParam;
    if (name === undefined) return declared;
    const value = this.#params[name];
    return typeof value === "number" && Number.isFinite(value) && value > 0
      ? Math.min(declared, value)
      : declared;
  }

  #renderTimelineBar(snap: TimelineSnapshot): void {
    const dom = this.#time;
    if (dom === null) return;
    const mode = this.#viz.time.mode;
    const playing = mode === "scrub" ? this.#autoPlay : snap.playing;
    dom.play.replaceChildren(icon(playing ? "pause" : "play"));
    const label = playing ? "暂停" : "播放";
    dom.play.setAttribute("aria-label", label);
    dom.play.title = label;
    dom.play.classList.toggle("is-playing", playing);

    if (dom.seek !== null) {
      dom.seek.max = String(this.#timeline.tMax);
      if (document.activeElement !== dom.seek) dom.seek.value = String(snap.t);
      const min = Number(dom.seek.min);
      const max = Number(dom.seek.max);
      const fraction = max > min ? ((snap.t - min) / (max - min)) * 100 : 0;
      dom.seek.style.setProperty("--fill", `${Math.min(100, Math.max(0, fraction)).toFixed(3)}%`);
    }

    const labelName = this.#viz.time.label ?? (mode === "stream" ? "t" : "t");
    const unit = mode === "stream" ? " s" : "";
    const metaK = this.#frame?.meta["k"];
    const useK = /(^|[^A-Za-z])k([^A-Za-z]|$)/.test(labelName) && typeof metaK === "number" && Number.isFinite(metaK);
    const shownValue = useK ? (metaK as number) : snap.t;
    const value = mode === "stream" ? snap.t.toFixed(2) : trimNumber(shownValue);
    dom.clock.replaceChildren();
    const main = el("span", { className: "nl-clock-main", text: `${labelName} = ${value}${unit}` });
    const sub = el("span", {
      className: "nl-clock-sub",
      text: mode === "scrub" ? `第 ${snap.frame + 1} / ${snap.frames} 帧` : playing ? "实时推进" : "已暂停",
    });
    dom.clock.append(main, sub);
    dom.dropped.textContent = snap.dropped > 0 ? `丢帧 ${snap.dropped}` : "";
    dom.rate.value = String(snap.rate);
  }

  #toggleScrubPlay(): void {
    if (this.#autoPlay) {
      this.#stopScrubPlay();
    } else {
      this.#startScrubPlay();
    }
  }

  #startScrubPlay(): void {
    if (this.#viz.time.mode !== "scrub") return;
    if (this.#timeline.t >= this.#timeline.tMax - this.#timeline.dt * 1e-9) {
      this.#timeline.seek(0);
    }
    this.#autoPlay = true;
    this.#lastAutoAt = performance.now();
    this.#renderTimelineBar(this.#timeline.snapshot());
    this.#emit();
  }

  #stopScrubPlay(): void {
    if (!this.#autoPlay) return;
    this.#autoPlay = false;
    this.#renderTimelineBar(this.#timeline.snapshot());
    this.#emit();
  }

  #tick = (): void => {
    if (this.#disposed) return;
    if (this.#viz.time.mode === "stream") {
      if (this.#timeline.playing) this.#timeline.advance();
    } else if (this.#viz.time.mode === "scrub" && this.#autoPlay) {
      const now = performance.now();
      const interval = 1000 / (6 * Math.max(0.5, this.#timeline.rate));
      if (now - this.#lastAutoAt >= interval) {
        this.#lastAutoAt = now;
        if (this.#timeline.t >= this.#timeline.tMax - this.#timeline.dt * 1e-9) {
          this.#stopScrubPlay();
        } else {
          this.#timeline.step(1);
        }
      }
    }
    this.#raf = requestAnimationFrame(this.#tick);
  };

  #onTick(snap: TimelineSnapshot): void {
    this.#source.request(snap.t);
    this.#renderTimelineBar(snap);
    this.#emit();
  }

  // ------------------------------------------------------------------ 数据与绘制

  #onFrame(frame: Frame, t: number): void {
    // 订阅确认会先回一帧 t=0。若页面是从 tMax / 分享链接的 t 进入，
    // 这帧只是握手噪声，画出来会让首屏从 0 闪一下再跳到目标时刻。
    if (this.#initialT !== null) {
      const tolerance = Math.max(this.#timeline.dt * 0.51, 1e-9);
      if (Math.abs(t - this.#initialT) > tolerance) return;
      this.#initialT = null;
    }
    this.#frame = frame;
    this.#frameT = t;
    this.#hadFrame = true;
    this.#history.push({ t, frame });
    if (this.#history.length > 600) this.#history.shift();
    if (this.#loading !== null) this.#loading.hidden = true;
    if (this.#stageError !== null) this.#stageError.hidden = true;
    this.#renderCurrent();
    // 帧里的 meta（例如迭代步 k）可能和请求的 t 不同；帧一到就刷新时钟。
    this.#renderTimelineBar(this.#timeline.snapshot());
  }

  #onSourceError(err: Error): void {
    if (!this.#hadFrame) {
      this.#showStageError("暂时取不到这一帧", err.message);
      return;
    }
    if (this.#sourceNote !== null) {
      this.#sourceNote.hidden = false;
      this.#sourceNote.dataset["tone"] = "warn";
      this.#sourceNote.textContent = "这个参数组合暂时取不到新帧，画面停在上一帧。";
    }
    const now = performance.now();
    if (now - this.#errorToastAt > 6000) {
      this.#errorToastAt = now;
      toast("暂时取不到新帧，画面停在上一帧。", "warn");
    }
  }

  #showStageError(title: string, detail: string): void {
    if (this.#loading !== null) this.#loading.hidden = true;
    const box = this.#stageError;
    if (box === null) return;
    box.hidden = false;
    box.replaceChildren();
    const mark = el("span", { className: "nl-stage-error-icon", attrs: { "aria-hidden": "true" } });
    mark.append(icon("alert"));
    const h = el("strong", { text: title });
    const p = el("p", { text: "检查本地服务是否在运行，然后刷新页面。" });
    const retry = el("button", { className: "nl-btn nl-btn-primary", attrs: { type: "button" } });
    retry.append(icon("reset"), el("span", { text: "刷新页面" }));
    retry.addEventListener("click", () => location.reload());
    const tech = el("details", { className: "nl-error-detail" });
    const summary = el("summary", { text: "技术细节" });
    const pre = el("p", { text: detail });
    tech.append(summary, pre);
    box.append(mark, h, p, retry, tech);
  }

  /** 依据容器宽度选择逻辑画布尺寸：字体随宽度保持正常大小，窄屏按比例缩小。 */
  #logicalSize(cssWidth: number): { width: number; height: number } {
    if (!(cssWidth > 0)) return { width: this.#width, height: this.#height };
    const width = Math.round(Math.min(LOGICAL_MAX_W, Math.max(LOGICAL_MIN_W, cssWidth)));
    return { width, height: LOGICAL_HEIGHT };
  }

  #measure(): void {
    if (this.#disposed || this.#stage === null) return;
    const frame = this.#stage.querySelector<HTMLElement>(".nl-canvas-frame");
    const cssWidth = frame?.clientWidth ?? this.#stage.clientWidth;
    if (!(cssWidth > 0)) return;
    this.#lastCssWidth = cssWidth;
    const size = this.#logicalSize(cssWidth);
    if (size.width === this.#width && size.height === this.#height) return;
    this.#width = size.width;
    this.#height = size.height;
    if (this.#frame !== null) this.#renderCurrent();
  }

  #scheduleMeasure(): void {
    if (this.#disposed) return;
    if (this.#resizeTimer !== 0) window.clearTimeout(this.#resizeTimer);
    this.#resizeTimer = window.setTimeout(() => {
      this.#resizeTimer = 0;
      if (this.#stage === null) return;
      const frame = this.#stage.querySelector<HTMLElement>(".nl-canvas-frame");
      const cssWidth = frame?.clientWidth ?? this.#stage.clientWidth;
      if (!(cssWidth > 0) || Math.abs(cssWidth - this.#lastCssWidth) < 1) return;
      this.#lastCssWidth = cssWidth;
      const size = this.#logicalSize(cssWidth);
      if (size.width === this.#width && size.height === this.#height) return;
      this.#width = size.width;
      this.#height = size.height;
      if (this.#frame !== null) this.#renderCurrent();
    }, 120);
  }

  #renderCurrent(): void {
    const frame = this.#frame;
    if (frame === null) return;
    const opts: ModelOptions = {
      width: this.#width,
      height: this.#height,
      t: this.#frameT,
      params: this.#params,
      theme: this.#theme,
      history: this.#history.slice(-60),
    };
    let scene: Scene;
    try {
      scene = this.#model(frame, opts);
    } catch (err) {
      this.#showStageError("这张图没有画出来", (err as Error).message);
      return;
    }
    const display = sceneForDisplay(scene);
    this.#scene = display.scene;
    this.#display = display;
    this.#hideHover();
    const hoverable = display.scene.layers.some((layer) =>
      layer.kind === "line" || layer.kind === "scatter" || layer.kind === "bars" ||
      layer.kind === "heat" || layer.kind === "vectorfield" || layer.kind === "loglogfit",
    );
    if (this.#hoverHint !== null) this.#hoverHint.hidden = !hoverable;
    this.#canvas.setAttribute("aria-label", `交互图形：${this.#figureCaption?.textContent ?? this.#viz.title}。可以悬停或轻点图中的数据元素查看数值。`);
    this.#draw(display.scene);
    this.#renderMetrics(frame);
    this.#renderLegend(display);
    this.#renderBadges(display);
    this.#renderGuide(display);
    if (this.#figureCaption !== null) {
      const title = display.axes[0]?.title ?? this.#viz.axes[0]?.title ?? this.#viz.title;
      this.#figureCaption.textContent = title;
      this.#figureCaption.title = title;
    }
  }

  #draw(scene: Scene): void {
    const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const w = Math.max(1, Math.round(scene.canvas.w));
    const h = Math.max(1, Math.round(scene.canvas.h));
    const bw = Math.round(w * dpr);
    const bh = Math.round(h * dpr);
    if (this.#canvas.width !== bw) this.#canvas.width = bw;
    if (this.#canvas.height !== bh) this.#canvas.height = bh;
    this.#canvas.style.aspectRatio = `${w} / ${h}`;
    this.#ctx.save();
    this.#ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.#ctx.clearRect(0, 0, bw, bh);
    this.#ctx.restore();
    renderScene(this.#ctx, scene, { dpr, theme: this.#theme });
  }

  #renderMetrics(frame: Frame): void {
    const row = this.#metricRow;
    if (row === null) return;
    const chips = keyMetrics(frame.meta, this.#opts.manifest.chapter, this.#viz.id);
    const key = JSON.stringify(chips);
    if (key === this.#lastMetricKey) return;
    this.#lastMetricKey = key;
    row.replaceChildren();
    if (chips.length === 0) {
      row.hidden = true;
      return;
    }
    row.hidden = false;
    row.append(el("span", { className: "nl-metrics-label", text: "当前帧" }));
    for (const chip of chips) {
      const item = el("span", { className: "nl-metric" });
      item.title = "随参数与时间更新";
      item.append(
        el("span", { className: "nl-metric-name", text: chip.label }),
        el("strong", { className: "nl-metric-value", text: chip.value }),
      );
      if (chip.unit !== "") item.append(el("span", { className: "nl-metric-unit", text: chip.unit }));
      row.append(item);
    }
  }

  #renderLegend(display: DisplaySceneResult): void {
    const row = this.#legendRow;
    if (row === null) return;
    const explicit: LegendChip[] = display.legend.map((item) => {
      const shape: LegendChip["shape"] =
        item.kind === "scatter"
          ? "scatter"
          : item.kind === "band"
            ? "band"
            : item.kind === "bars"
              ? "bars"
              : item.kind === "heat"
                ? "heat"
                : item.kind === "contour"
                  ? "contour"
                  : "line";
      return {
        label: item.label,
        color: item.color,
        shape,
        dash: item.dash === true || item.kind === "contour",
        axis: item.axis,
        ...(item.gradient === undefined ? {} : { gradient: item.gradient }),
      };
    });
    const chips = mergeLegend(explicit, legendFromTexts(display.texts));
    const letters = new Map((this.#display?.axes ?? []).map((axis, index) => [axis.id, String.fromCharCode(65 + index)]));
    const repeated = new Map<string, number>();
    for (const chip of chips) {
      if (chip.axis === undefined) continue;
      const k = `${chip.shape}\u0000${colorKey(chip.color)}\u0000${chip.dash ? "d" : "s"}`;
      repeated.set(k, (repeated.get(k) ?? 0) + 1);
    }
    const colorLabels: Record<string, string> = {};
    for (const chip of chips) {
      if (chip.shape !== "line" && chip.shape !== "scatter") continue;
      const key = colorKey(chip.color);
      if (colorLabels[key] === undefined) colorLabels[key] = chip.label;
      const raw = chip.color.toLowerCase();
      if (raw.startsWith("#") && colorLabels[raw] === undefined) colorLabels[raw] = chip.label;
    }
    this.#legendColors = colorLabels;
    for (const chip of chips) {
      if (chip.axis === undefined) continue;
      const k = `${chip.shape}\u0000${colorKey(chip.color)}\u0000${chip.dash ? "d" : "s"}`;
      const letter = letters.get(chip.axis);
      if ((repeated.get(k) ?? 0) > 1 && letter !== undefined) chip.label = `${chip.label}（${letter}）`;
    }
    const key = JSON.stringify(chips);
    if (key === this.#lastLegendKey) return;
    this.#lastLegendKey = key;
    row.replaceChildren();
    if (chips.length === 0) {
      row.hidden = true;
      return;
    }
    row.hidden = false;
    row.append(el("span", { className: "nl-legend-label", text: "图例" }));
    const limit = 8;
    chips.forEach((item, index) => {
      const chip = el("span", { className: "nl-legend-item" });
      chip.hidden = index >= limit;
      const swatch = el("span", {
        className: `nl-swatch nl-swatch-${item.shape}${item.dash ? " is-dashed" : ""}`,
      });
      if (item.gradient !== undefined) {
        swatch.style.background = item.gradient;
        swatch.classList.add("nl-swatch-gradient");
      } else {
        swatch.style.setProperty("--swatch", item.color);
      }
      const text = el("span", { className: "nl-legend-text", text: item.label });
      text.title = item.label;
      chip.append(swatch, text);
      row.append(chip);
    });
    if (chips.length > limit) {
      const toggle = el("button", {
        className: "nl-legend-toggle",
        attrs: { type: "button", title: "展开全部图例" },
      });
      toggle.textContent = `+${chips.length - limit} 项`;
      toggle.addEventListener("click", () => {
        for (const chip of row.querySelectorAll<HTMLElement>(".nl-legend-item")) chip.hidden = false;
        toggle.remove();
      });
      row.append(toggle);
    }
  }

  #renderBadges(display: DisplaySceneResult): void {
    const row = this.#badgeRow;
    if (row === null) return;
    const badges = display.badges.filter((badge) => badge.kind === "critical" || badge.kind === "overflow" || badge.kind === "preview");
    const key = JSON.stringify(badges.map((badge) => `${badge.kind}:${badge.text}`));
    if (key === this.#lastBadgeKey) return;
    this.#lastBadgeKey = key;
    row.replaceChildren();
    if (badges.length === 0) {
      row.hidden = true;
      return;
    }
    row.hidden = false;
    for (const badge of badges) {
      const dangerous = badge.kind === "overflow" || /越界/.test(badge.text);
      const chip = el("span", { className: `nl-state-chip nl-state-${badge.kind}` });
      chip.dataset["tone"] = dangerous ? "danger" : "note";
      chip.append(icon(dangerous ? "alert" : "info"));
      chip.append(el("span", { text: badge.text }));
      row.append(chip);
    }
  }

  #renderGuide(display?: DisplaySceneResult): void {
    const list = this.#guideList;
    if (list === null) return;
    const axes = display?.axes ?? this.#viz.axes.map((axis) => ({
      id: axis.id,
      title: axis.title ?? "",
      xlabel: "",
      ylabel: axis.ylabel ?? "",
      xscale: "linear",
      yscale: axis.scale ?? "linear",
    }));
    const manifestAxes = new Map(this.#viz.axes.map((axis) => [axis.id, axis]));
    const key = JSON.stringify(axes.map((axis) => [axis.id, axis.title, axis.xlabel, axis.ylabel, axis.xscale, axis.yscale]));
    if (key === this.#lastGuideKey) return;
    this.#lastGuideKey = key;
    list.replaceChildren();
    const letters = ["A", "B", "C", "D", "E", "F"];
    axes.forEach((axis, index) => {
      const item = el("li", { className: "nl-panel-row" });
      const letter = el("span", { className: "nl-panel-letter", text: letters[index] ?? String(index + 1) });
      const body = el("div", { className: "nl-panel-body" });
      const spec = manifestAxes.get(axis.id);
      const title =
        axis.title.trim() !== ""
          ? axis.title
          : axis.ylabel.trim() !== ""
            ? `${axis.ylabel.trim()}（补充视图）`
            : "补充视图";
      const name = el("p", { className: "nl-panel-name", text: title });
      const xText = axis.xlabel.trim() !== "" ? axis.xlabel.trim() : "见面板标题";
      const yText = axis.ylabel.trim() !== "" ? axis.ylabel.trim() : "见面板标题";
      const metaBits = [`横轴：${xText}`, `纵轴：${yText}${axis.yscale === "log" ? "（对数）" : ""}`];
      if (axis.xscale === "log") metaBits.push("横轴取对数");
      if (spec?.autoscaled === true) metaBits.push("自适应");
      const meta = el("p", { className: "nl-panel-meta", text: metaBits.join(" · ") });
      body.append(name, meta);
      item.append(letter, body);
      list.append(item);
    });
    if (axes.length === 0) {
      list.append(el("li", { className: "nl-panel-empty", text: "这张图没有独立的坐标轴。" }));
    }
  }

  #renderStatus(): void {
    const node = this.#statusEl;
    const note = this.#sourceNote;
    if (node === null) return;
    const status = this.#status;
    if (status.level === "ws") {
      node.hidden = false;
      node.dataset["level"] = "ok";
      node.replaceChildren(el("span", { className: "nl-live-dot" }), el("span", { text: "实时计算" }));
      node.title = status.message;
      if (note !== null) note.hidden = true;
      this.#sourceNoteLevel = "";
      return;
    }
    if (status.attempts === 0) {
      node.hidden = true;
      return;
    }
    const map: Record<string, string> = {
      rest: "按需取帧",
      static: "离线帧",
      none: "暂无数据",
    };
    node.hidden = false;
    node.dataset["level"] = status.level === "rest" ? "warn" : status.level === "static" ? "warn" : "error";
    node.replaceChildren(el("span", { className: "nl-live-dot" }), el("span", { text: map[status.level] ?? "数据状态" }));
    node.title = status.message;
    if (note === null) return;
    const level = `${status.level}:${status.attempts}`;
    if (this.#sourceNoteLevel === level) return;
    this.#sourceNoteLevel = level;
    note.hidden = false;
    note.dataset["tone"] = status.level === "none" ? "error" : "warn";
    if (status.level === "static") {
      note.textContent = "当前使用预先算好的帧。部分参数或时刻可能没有快照，画面会停在最近一帧。";
    } else if (status.level === "rest") {
      note.textContent = "实时通道不可用，正在按需向服务器取帧；拖动时可能会有一点延迟。";
    } else {
      note.textContent = "还没有连接到数据服务。检查本地服务是否在运行，然后刷新页面。";
    }
  }

  // ------------------------------------------------------------------ 导出

  async #exportPng(): Promise<void> {
    try {
      const bytes = await canvasPngBytes(this.#canvas);
      const blob = new Blob([bytes.slice().buffer as ArrayBuffer], { type: "image/png" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = downloadName(this.#opts.manifest.chapter, this.#viz.id, this.#params, this.#timeline.t);
      a.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      toast("PNG 已保存。");
    } catch (err) {
      toast(`导出失败：${(err as Error).message}`, "warn");
    }
  }

  #exportScene(): void {
    if (this.#scene === null) {
      toast("还没有可导出的画面。", "warn");
      return;
    }
    const blob = new Blob([sceneToJsonText(this.#scene)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = downloadName(this.#opts.manifest.chapter, this.#viz.id, this.#params, this.#timeline.t).replace(/\.png$/, ".json");
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("场景 JSON 已保存。");
  }

  async #share(): Promise<void> {
    const hash = buildHash(this.state);
    const url = `${location.origin}${location.pathname}${hash}`;
    try {
      if (navigator.clipboard === undefined) throw new Error("clipboard unavailable");
      await navigator.clipboard.writeText(url);
      toast("分享链接已复制。");
    } catch {
      history.replaceState(null, "", hash);
      toast("分享链接已放进地址栏。");
    }
  }

  #syncFullscreenButton(): void {
    const card = this.#opts.host.querySelector<HTMLElement>(".nl-chart-card");
    const button = this.#opts.host.querySelector<HTMLButtonElement>(".nl-chart-tools .nl-icon-btn");
    if (button === null) return;
    const active = document.fullscreenElement !== null && document.fullscreenElement === card;
    button.setAttribute("aria-label", active ? "退出全屏" : "全屏查看图形");
    button.title = active ? "退出全屏" : "全屏查看";
    button.classList.toggle("is-active", active);
  }

  #emit(): void {
    this.#opts.onState?.(this.state, this.#status);
  }
}

/** 便捷入口：建视图并挂载。 */
export async function mountViz(opts: VizViewOptions): Promise<VizView> {
  const view = new VizView(opts);
  await view.mount();
  return view;
}

/** 从 URL hash 恢复初始状态（站点与分享链接共用）。 */
export function stateFromHash(hash: string): ShareState | null {
  return decodeHash(hash);
}
