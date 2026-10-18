/**
 * 可视化视图层：把一个章节可视化"装"进页面（DOM + canvas + 控件 + 时间轴 + 导出）。
 *
 * 分工（这是刻意的，因为环境里没有浏览器可以自动化测试）：
 *
 * * **纯逻辑**在别处，且有测试：`framesource`（取帧与降级）、`timeline`（时间轴）、
 *   `controls`（控件描述与夹取）、`pitfalls`（陷阱面板文案）、`share`（链接状态编码）。
 * * **本文件**只做"把上面那些接到 DOM 上"：建元素、绑事件、调用 `renderScene`。
 *   所以这里不写任何数值逻辑，也不做任何判断性计算——出错的话只可能是"接错线"。
 *
 * 视图的输入是 manifest 里的 `VizSpecJson` 与一个 `SceneModel`，因此**章节无关**：
 * 新章节只要提供 model 与 manifest，这里不用改一行。
 */

import type { Frame } from "./frame.js";
import type { ChapterManifestJson, VizSpecJson } from "./manifest.js";
import type { ModelOptions, SceneModel } from "./model.js";
import { palette } from "./model.js";
import type { Scene } from "./scene.js";
import { renderScene } from "./render2d.js";
import { ChapterFrameSource, type SourceStatus } from "./framesource.js";
import { Timeline, type TimelineSnapshot } from "./timeline.js";
import { mountControls } from "./controls.js";
import { mountPitfalls, pitfallsModel } from "./pitfalls.js";
import { canvasPngBytes, downloadName, sceneToJsonText } from "./export.js";
import { buildHash, decodeHash, encodeState, type ParamValues, type ShareState } from "./share.js";

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
  /** 注入数据源（测试/复用）；缺省内部创建。 */
  source?: ChapterFrameSource;
  /** 状态变化回调（站点用它同步 URL）。 */
  onState?: (state: ShareState, status: SourceStatus) => void;
}

export const DEFAULT_WIDTH = 960;
export const DEFAULT_HEIGHT = 600;

/** 去掉多余尾零的数字写法（只看一眼人读的时间步长，不参与任何计算）。 */
function trimNumber(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(6)));
}

/** 从 manifest 取缺省参数（与 `defaultParams` 同义，这里只做本地化避免额外 import）。 */
function defaultsOf(viz: VizSpecJson): ParamValues {
  const out: ParamValues = {};
  for (const p of viz.params) out[p.name] = p.default;
  return out;
}

export class VizView {
  readonly #opts: VizViewOptions;
  readonly #viz: VizSpecJson;
  readonly #model: SceneModel;
  readonly #width: number;
  readonly #height: number;
  readonly #dpr: number;
  readonly #canvas: HTMLCanvasElement;
  readonly #ctx: CanvasRenderingContext2D;
  readonly #timeline: Timeline;
  readonly #source: ChapterFrameSource;
  readonly #history: { t: number; frame: Frame }[] = [];
  readonly #roots: { timeline: HTMLElement; controls: HTMLElement; pitfalls: HTMLElement; status: HTMLElement };
  #params: ParamValues;
  #frame: Frame | null = null;
  #scene: Scene | null = null;
  #status: SourceStatus;
  #raf = 0;
  #disposed = false;

  constructor(opts: VizViewOptions) {
    this.#opts = opts;
    this.#viz = opts.viz;
    this.#model = opts.model;
    this.#width = opts.width ?? DEFAULT_WIDTH;
    this.#height = opts.height ?? DEFAULT_HEIGHT;
    this.#dpr = opts.dpr ?? (typeof devicePixelRatio === "number" ? Math.min(2, devicePixelRatio) : 1);
    this.#params = opts.params ?? defaultsOf(opts.viz);

    const canvas = document.createElement("canvas");
    canvas.width = Math.round(this.#width * this.#dpr);
    canvas.height = Math.round(this.#height * this.#dpr);
    canvas.className = "nl-canvas";
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("拿不到 2D 上下文：这个环境不支持 canvas");
    this.#canvas = canvas;
    this.#ctx = ctx;

    const time = opts.viz.time;
    this.#timeline = new Timeline({
      mode: time.mode,
      tMax: time.tMax ?? 1,
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
        time: { mode: time.mode, dt: time.dt, tMax: time.tMax, frames: this.#timeline.frameCount },
      });
    this.#status = this.#source.status;

    this.#roots = {
      timeline: document.createElement("div"),
      controls: document.createElement("div"),
      pitfalls: document.createElement("div"),
      status: document.createElement("div"),
    };
  }

  get canvas(): HTMLCanvasElement {
    return this.#canvas;
  }

  get scene(): Scene | null {
    return this.#scene;
  }

  /** 最近一帧原始数据（导出/调试用；渲染一律走 Scene，不要手改数组）。 */
  get frame(): Frame | null {
    return this.#frame;
  }

  get state(): ShareState {
    return {
      chapter: this.#opts.manifest.chapter,
      viz: this.#viz.id,
      params: this.#params,
      t: this.#timeline.t,
      playing: this.#timeline.playing,
    };
  }

  /** 建 DOM、绑事件、连数据源。可重复调用（幂等：先 dispose 再重建）。 */
  async mount(): Promise<void> {
    const host = this.#opts.host;
    host.innerHTML = "";
    host.classList.add("nl-view");

    const head = document.createElement("header");
    head.className = "nl-head";
    const h2 = document.createElement("h2");
    h2.textContent = this.#viz.title;
    const badges = document.createElement("div");
    badges.className = "nl-badges";
    this.#roots.status.className = "nl-status";
    head.append(h2, badges, this.#roots.status);

    const stage = document.createElement("div");
    stage.className = "nl-stage";
    stage.append(this.#canvas);

    const aside = document.createElement("aside");
    aside.className = "nl-side";
    aside.append(this.#roots.controls, this.#roots.pitfalls);

    const bar = document.createElement("div");
    bar.className = "nl-timeline";
    this.#roots.timeline = bar;

    const main = document.createElement("div");
    main.className = "nl-main";
    main.append(stage, bar);
    host.append(head, main, aside);

    this.#buildBadges(badges);
    mountControls(this.#viz, this.#roots.controls, (_name, _value, params) => {
      this.#params = params;
      this.#history.length = 0;
      this.#source.setParams(params);
      this.#source.request(this.#timeline.t);
      this.#emit();
    });
    mountPitfalls(pitfallsModel(this.#viz), this.#roots.pitfalls);
    this.#buildTimelineBar(bar);
    this.#buildExportBar(badges);

    this.#source.on("frame", (frame, t) => this.#onFrame(frame, t));
    this.#source.on("status", (status) => {
      this.#status = status;
      this.#renderStatus();
      this.#emit();
    });
    this.#source.on("error", (err) => {
      this.#roots.status.textContent = `取帧出错：${err.message}`;
      this.#roots.status.dataset["level"] = "error";
    });

    await this.#source.connect();
    this.#source.request(this.#timeline.t);
    if (this.#timeline.mode === "stream") this.#source.startStream();

    this.#timeline.onChange((snap) => this.#onTick(snap));
    this.#tick();
  }

  dispose(): void {
    this.#disposed = true;
    if (this.#raf) cancelAnimationFrame(this.#raf);
    this.#raf = 0;
    this.#source.dispose();
    this.#opts.host.innerHTML = "";
  }

  // ------------------------------------------------------------------ //
  #tick = (): void => {
    if (this.#disposed) return;
    if (this.#timeline.playing || this.#timeline.mode === "stream") this.#timeline.advance();
    this.#raf = requestAnimationFrame(this.#tick);
  };

  #onTick(snap: TimelineSnapshot): void {
    this.#source.request(snap.t);
    this.#renderTimelineBar(snap);
    this.#emit();
  }

  #onFrame(frame: Frame, t: number): void {
    this.#frame = frame;
    this.#history.push({ t, frame });
    if (this.#history.length > 600) this.#history.shift();
    const opts: ModelOptions = {
      width: this.#width,
      height: this.#height,
      t,
      params: this.#params,
      theme: this.#opts.theme ?? "light",
      history: this.#history.slice(-60),
    };
    let scene: Scene;
    try {
      scene = this.#model(frame, opts);
    } catch (err) {
      this.#roots.status.textContent = `模型出错：${(err as Error).message}`;
      this.#roots.status.dataset["level"] = "error";
      return;
    }
    this.#scene = scene;
    const pal = palette(this.#opts.theme ?? "light");
    this.#ctx.save();
    this.#ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.#ctx.clearRect(0, 0, this.#canvas.width, this.#canvas.height);
    this.#ctx.fillStyle = scene.canvas.background || pal.background;
    this.#ctx.fillRect(0, 0, this.#canvas.width, this.#canvas.height);
    this.#ctx.restore();
    renderScene(this.#ctx, scene, { dpr: this.#dpr });
  }

  // ------------------------------------------------------------------ //
  #buildBadges(host: HTMLElement): void {
    const model = pitfallsModel(this.#viz);
    for (const text of model.badges) {
      const b = document.createElement("span");
      b.className = "nl-badge";
      b.textContent = text;
      host.append(b);
    }
  }

  #renderStatus(): void {
    const s = this.#status;
    this.#roots.status.textContent = s.message;
    this.#roots.status.dataset["level"] = s.level;
    this.#roots.status.dataset["degraded"] = s.degraded ? "1" : "0";
    this.#roots.status.title = `数据源：${s.level}（尝试 ${s.attempts} 次）`;
  }

  #renderTimelineBar(snap: TimelineSnapshot): void {
    const bar = this.#roots.timeline;
    const label = bar.querySelector<HTMLElement>("[data-role=clock]");
    if (label) {
      const t = snap.t;
      const dt = this.#timeline.dt;
      label.textContent = `t = ${t.toFixed(t < 1 ? 4 : 3)}（${snap.frame + 1}/${snap.frames} 帧，Δt = ${trimNumber(dt)}）`;
    }
    const slider = bar.querySelector<HTMLInputElement>("[data-role=seek]");
    if (slider && document.activeElement !== slider) slider.value = String(snap.t);
    const play = bar.querySelector<HTMLButtonElement>("[data-role=play]");
    if (play) play.textContent = snap.playing ? "⏸ 暂停" : "▶ 播放";
    const drop = bar.querySelector<HTMLElement>("[data-role=dropped]");
    if (drop) drop.textContent = snap.dropped > 0 ? `丢帧 ${snap.dropped}` : "";
  }

  #buildTimelineBar(bar: HTMLElement): void {
    const time = this.#viz.time;
    const play = document.createElement("button");
    play.dataset["role"] = "play";
    play.textContent = "▶ 播放";
    play.addEventListener("click", () => this.#timeline.toggle());

    const stepBack = document.createElement("button");
    stepBack.textContent = "⏪";
    stepBack.title = "上一帧";
    stepBack.addEventListener("click", () => this.#timeline.step(-1));

    const stepFwd = document.createElement("button");
    stepFwd.textContent = "⏩";
    stepFwd.title = "下一帧";
    stepFwd.addEventListener("click", () => this.#timeline.step(1));

    const seek = document.createElement("input");
    seek.type = "range";
    seek.dataset["role"] = "seek";
    seek.min = "0";
    seek.max = String(time.tMax ?? 1);
    seek.step = String(time.dt ?? 0.01);
    seek.value = String(this.#timeline.t);
    seek.addEventListener("input", () => this.#timeline.seek(Number(seek.value)));

    const clock = document.createElement("span");
    clock.dataset["role"] = "clock";

    const dropped = document.createElement("span");
    dropped.dataset["role"] = "dropped";
    dropped.className = "nl-dropped";

    const rate = document.createElement("select");
    for (const r of [0.25, 0.5, 1, 2, 4]) {
      const o = document.createElement("option");
      o.value = String(r);
      o.textContent = `${r}×`;
      if (r === 1) o.selected = true;
      rate.append(o);
    }
    rate.addEventListener("change", () => this.#timeline.setRate(Number(rate.value)));

    bar.append(play, stepBack, stepFwd, seek, clock, rate, dropped);
    if (time.mode === "none") {
      for (const el of [play, stepBack, stepFwd, seek, rate]) el.setAttribute("disabled", "disabled");
      clock.textContent = "本图无时间轴";
    }
    this.#renderTimelineBar(this.#timeline.snapshot());
  }

  #buildExportBar(badges: HTMLElement): void {
    const png = document.createElement("button");
    png.textContent = "⬇ PNG";
    png.title = "把当前画面存成 PNG（后端不在时唯一可用的导出）";
    png.addEventListener("click", () => {
      void canvasPngBytes(this.#canvas).then((bytes) => {
        const blob = new Blob([bytes.slice().buffer as ArrayBuffer], { type: "image/png" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = downloadName(this.#opts.manifest.chapter, this.#viz.id, this.#params, this.#timeline.t);
        a.click();
        URL.revokeObjectURL(url);
      });
    });

    const json = document.createElement("button");
    json.textContent = "⬇ Scene JSON";
    json.title = "导出当前 Scene（视觉回归的输入就是这个）";
    json.addEventListener("click", () => {
      if (!this.#scene) return;
      const blob = new Blob([sceneToJsonText(this.#scene)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = downloadName(this.#opts.manifest.chapter, this.#viz.id, this.#params, this.#timeline.t).replace(/\.png$/, ".json");
      a.click();
      URL.revokeObjectURL(url);
    });

    const share = document.createElement("button");
    share.textContent = "🔗 复制链接";
    share.addEventListener("click", () => {
      const hash = buildHash(this.state);
      const url = `${location.origin}${location.pathname}${hash}`;
      void navigator.clipboard?.writeText(url).then(
        () => {
          share.textContent = "✅ 已复制";
          setTimeout(() => (share.textContent = "🔗 复制链接"), 1200);
        },
        () => {
          location.hash = hash;
        },
      );
    });

    const copy = document.createElement("button");
    copy.textContent = "⧉ 复制状态";
    copy.title = "把可分享的状态串复制到剪贴板（`v1.` 前缀的 base64url）";
    copy.addEventListener("click", () => {
      void navigator.clipboard?.writeText(encodeState(this.state));
    });

    const box = document.createElement("div");
    box.className = "nl-export";
    box.append(png, json, share, copy);
    badges.append(box);
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
