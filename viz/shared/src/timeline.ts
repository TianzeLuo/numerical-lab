/**
 * 时间轴控制器（纯逻辑 + 可注入时钟，不碰 DOM、不碰网络）。
 *
 * 三种模式（与 manifest 的 `time.mode` 一一对应）：
 *
 * | mode | play/pause/advance | seek/step | 说明 |
 * |---|---|---|---|
 * | `stream` | 有效：按真实时间推进，受 `fps` 限流 | 有效（seek 会重置节流基准） | 时间驱动，UI 每帧调 `advance()` |
 * | `scrub` | **无效**（`play()` 是 no-op） | 有效 | 只由用户拖动/单步决定时刻 |
 * | `none` | 无效 | 无效 | 没有时间轴，`t` 恒为 0 |
 *
 * 两个容易被忽略的点，这里显式处理：
 *
 * 1. **"帧还没到就又该推进"**：真实时间过去了 5 个帧间隔、但渲染只来得及画一帧时，
 *    `advance()` 直接把时间推到最新（不补画中间帧），并把跳过的帧数记进 `droppedFrames`。
 *    补画没有意义——用户看到的应该是"现在"，而不是排队播放一段已经过时的动画。
 * 2. **零头不丢**：不足一个帧间隔的时间留在内部累加器里，下一次 `advance()` 继续算，
 *    所以长期平均速度精确等于 `fps × rate`，不会因为定时器抖动而越走越慢。
 *
 * `now` 可注入：测试里用假时钟把 10 秒的播放压缩成几次函数调用，一秒都不用真等。
 */

import type { TimeMode } from "./manifest.js";

export interface TimelineOptions {
  mode: TimeMode;
  tMax: number;
  dt: number;
  fps: number;
  /** 注入时钟（ms）；缺省 `Date.now`。 */
  now?: () => number;
}

/** 状态快照（`onChange` 的参数，也是 UI 一次性刷新的输入）。 */
export interface TimelineSnapshot {
  t: number;
  playing: boolean;
  rate: number;
  /** 当前帧号（`round(t/dt)`，夹在 `[0, frameCount-1]`）。 */
  frame: number;
  /** 总帧数（`floor(tMax/dt)+1`；无时间轴时为 1）。 */
  frames: number;
  /** 因"来不及"而跳过的帧数。 */
  dropped: number;
  /** 是否已到时间轴末端（到末端会自动暂停）。 */
  ended: boolean;
}

export type TimelineListener = (snapshot: TimelineSnapshot) => void;

const MIN_RATE = 0.1;
const MAX_RATE = 16;

/** 有限正数校验（`dt` / `fps` / `tMax` 的容错入口）。 */
function positiveOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

/**
 * 时间轴控制器。
 *
 * ```ts
 * const tl = new Timeline({ mode: "stream", tMax: 20, dt: 0.05, fps: 30 });
 * tl.onChange(() => request(tl.t));   // UI 侧把 t 交给 ChapterFrameSource
 * tl.play();
 * requestAnimationFrame(function loop() { tl.advance(); requestAnimationFrame(loop); });
 * ```
 */
export class Timeline {
  readonly #mode: TimeMode;
  #tMax: number;
  readonly #dt: number;
  readonly #fps: number;
  readonly #now: () => number;
  #frameCount: number;

  #t = 0;
  #playing = false;
  #rate = 1;
  #acc = 0;
  #lastNow = 0;
  #dropped = 0;
  #ended = false;
  #listeners: TimelineListener[] = [];

  constructor(opts: TimelineOptions) {
    this.#mode = opts.mode;
    this.#dt = positiveOr(opts.dt, 1);
    this.#fps = positiveOr(opts.fps, 30);
    this.#tMax = opts.mode === "none" ? 0 : positiveOr(opts.tMax, 0);
    this.#now = opts.now ?? (() => Date.now());
    this.#frameCount = this.#tMax > 0 ? Math.floor(this.#tMax / this.#dt) + 1 : 1;
  }

  // ------------------------------------------------------------------ 只读状态

  get t(): number {
    return this.#t;
  }

  get playing(): boolean {
    return this.#playing;
  }

  get rate(): number {
    return this.#rate;
  }

  get mode(): TimeMode {
    return this.#mode;
  }

  get dt(): number {
    return this.#dt;
  }

  get tMax(): number {
    return this.#tMax;
  }

  get fps(): number {
    return this.#fps;
  }

  /** 因"来不及"跳过的帧数（可读指标）。 */
  get droppedFrames(): number {
    return this.#dropped;
  }

  /**
   * 当前帧号；`t` 落在两帧之间时取最近的整数帧。
   *
   * `+ dt·1e-9` 是吸收浮点噪声用的：`0.35/0.1` 在 IEEE754 下是 `3.4999999999999996`，
   * 直接 `round` 会掉到第 3 帧（明明离第 4 帧一样近）；而 `0.3/0.1 = 2.9999999999999996`
   * 又必须落在第 3 帧。加一个远小于 dt 的 epsilon 两个都对。
   */
  get frameIndex(): number {
    if (this.#frameCount <= 1) return 0;
    const raw = Math.round((this.#t + this.#dt * 1e-9) / this.#dt);
    return Math.min(this.#frameCount - 1, Math.max(0, raw));
  }

  /** 总帧数。 */
  get frameCount(): number {
    return this.#frameCount;
  }

  /** 是否已到末端。 */
  get ended(): boolean {
    return this.#ended;
  }

  /** 进度 0–1（无时间轴时为 0）。 */
  get progress(): number {
    return this.#tMax > 0 ? Math.min(1, Math.max(0, this.#t / this.#tMax)) : 0;
  }

  /** 当前状态快照。 */
  snapshot(): TimelineSnapshot {
    return {
      t: this.#t,
      playing: this.#playing,
      rate: this.#rate,
      frame: this.frameIndex,
      frames: this.#frameCount,
      dropped: this.#dropped,
      ended: this.#ended,
    };
  }

  onChange(cb: TimelineListener): () => void {
    this.#listeners.push(cb);
    return () => {
      const i = this.#listeners.indexOf(cb);
      if (i >= 0) this.#listeners.splice(i, 1);
    };
  }

  // ------------------------------------------------------------------ 控制

  /** 开始播放。`scrub`/`none` 模式下是 no-op；已在末端时从头开始。 */
  play(): void {
    if (this.#mode !== "stream" || this.#playing) return;
    if (this.#tMax <= 0) return; // 没有可推进的区间
    if (this.#t >= this.#tMax) {
      this.#t = 0;
      this.#ended = false;
    }
    this.#playing = true;
    this.#acc = 0;
    this.#lastNow = this.#now();
    this.#emit();
  }

  pause(): void {
    if (!this.#playing) return;
    this.#playing = false;
    this.#acc = 0;
    this.#emit();
  }

  toggle(): void {
    if (this.#playing) this.pause();
    else this.play();
  }

  /** 跳到某时刻（夹到 `[0, tMax]`）；会重置节流基准，避免跳转后立刻"丢帧"。 */
  seek(t: number): void {
    if (this.#mode === "none") return;
    const next = this.#clamp(typeof t === "number" && Number.isFinite(t) ? t : 0);
    this.#acc = 0;
    this.#lastNow = this.#now();
    this.#ended = this.#tMax > 0 && next >= this.#tMax;
    if (next === this.#t) {
      this.#emit(); // 时刻没变，但播放态可能刚刚被 seek 到末端清掉
      return;
    }
    this.#t = next;
    this.#emit();
  }

  /** 单步 `n` 帧（`n` 可为负）；`scrub`/`stream` 都有效，`none` 无效。 */
  step(n: number): void {
    if (this.#mode === "none") return;
    const count = typeof n === "number" && Number.isFinite(n) ? Math.trunc(n) : 0;
    if (count === 0) return;
    this.seek(this.#t + count * this.#dt);
  }

  /**
   * 调整时间轴上限（例如 `k` 只能走到参数 `n`）。
   *
   * 上限缩小时当前时刻随之夹紧；放大时保持当前位置。播放头停在旧末端的情况
   * 会退出 `ended`，让用户能继续往后拖。
   */
  setRange(tMax: number): void {
    if (this.#mode === "none") return;
    if (typeof tMax !== "number" || !Number.isFinite(tMax) || tMax <= 0) return;
    if (tMax === this.#tMax) return;
    this.#tMax = tMax;
    this.#frameCount = Math.floor(tMax / this.#dt) + 1;
    if (this.#t > tMax) this.#t = tMax;
    this.#ended = this.#t >= this.#tMax;
    this.#acc = 0;
    this.#lastNow = this.#now();
    this.#emit();
  }

  /** 调速（0.1×–16×）；非法值（≤0、NaN）回落到 1×。 */
  setRate(rate: number): void {
    const next =
      typeof rate === "number" && Number.isFinite(rate) && rate > 0
        ? Math.min(MAX_RATE, Math.max(MIN_RATE, rate))
        : 1;
    if (next === this.#rate) return;
    this.#rate = next;
    this.#acc = 0; // 换速度不结算零头，免得"慢→快"时突然跳一大步
    this.#lastNow = this.#now();
    this.#emit();
  }

  /**
   * 按真实时间推进一次（UI 每帧调一次）。
   *
   * 返回**本次推进的帧数**（0 = 还没到下一帧、或没在播放）。跨越多个帧间隔时，
   * 只把 `t` 推到最新并累加 `droppedFrames`——见文件头第 1 条。
   */
  advance(): number {
    if (this.#mode !== "stream" || !this.#playing) return 0;
    const nowMs = this.#now();
    const elapsed = nowMs - this.#lastNow;
    if (!(elapsed > 0)) return 0;
    this.#lastNow = nowMs;
    const frameMs = 1000 / this.#fps;
    this.#acc += elapsed * this.#rate;
    const due = Math.floor(this.#acc / frameMs);
    if (due <= 0) return 0;
    this.#acc -= due * frameMs;

    const target = this.#t + due * this.#dt;
    if (this.#tMax > 0 && target >= this.#tMax) {
      // 到末端：把时间夹到 tMax，并按"还剩几帧"计数（至少 1 帧），然后自动暂停。
      const remaining = Math.max(1, Math.round((this.#tMax - this.#t) / this.#dt));
      this.#dropped += Math.max(0, remaining - 1);
      this.#t = this.#tMax;
      this.#playing = false;
      this.#ended = true;
      this.#acc = 0;
      this.#emit();
      return remaining;
    }
    this.#t = target;
    if (due > 1) this.#dropped += due - 1;
    this.#emit();
    return due;
  }

  /** 归零（停播、t=0、清空丢帧计数）；供"重播"按钮用。 */
  reset(): void {
    this.#playing = false;
    this.#acc = 0;
    this.#lastNow = this.#now();
    this.#dropped = 0;
    this.#ended = false;
    if (this.#t !== 0) this.#t = 0;
    this.#emit();
  }

  // ------------------------------------------------------------------ 内部

  #clamp(t: number): number {
    if (t < 0) return 0;
    if (this.#tMax > 0 && t > this.#tMax) return this.#tMax;
    if (this.#tMax <= 0 && this.#mode === "none") return 0;
    return t;
  }

  #emit(): void {
    const snap = this.snapshot();
    for (const cb of [...this.#listeners]) {
      try {
        cb(snap);
      } catch {
        /* 订阅者异常不影响时间轴 */
      }
    }
  }
}
