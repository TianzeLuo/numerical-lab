/**
 * 数据源与降级链（L1 WS → L2 REST → L3 静态 → none）的测试。
 *
 * 这里没有浏览器、没有后端：WebSocket / fetch / 静态目录 / 时钟 / 定时器**全部是假的**，
 * 所以每一条降级路径都能被精确地、可重复地走一遍——包括"WS 连上了但只推到一半就掉线"
 * 这种在真实环境里很难复现的情况。
 *
 * 帧字节用 `encodeNlf` 自己造（`nlf.ts` 是唯一被允许的"造帧"出口），fake fetch 直接回这些字节，
 * 于是"解码 → 按 meta.t 对齐时间轴"这条链路也一起被验了。
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import { Frame } from "../shared/src/frame.js";
import type { TypedArray } from "../shared/src/frame.js";
import { encodeNlf } from "../shared/src/nlf.js";
import {
  ChapterFrameSource,
  REST_BACKOFF_BASE_MS,
  REST_THROTTLE_MS,
  canonicalParamsJson,
  decodeParams,
  encodeParams,
  formatPyG,
  paramsHash,
  pythonFloatRepr,
  restFrameUrl,
  restFramesUrl,
  sha256Hex,
  staticFrameCandidates,
  staticFrameRelPath,
  wsChapterUrl,
} from "../shared/src/framesource.js";
import type { FrameSourceDeps, SourceLevel, SourceStatus } from "../shared/src/framesource.js";

// --------------------------------------------------------------------------- //
// 造帧
// --------------------------------------------------------------------------- //

/** 造一帧最小可解码的 `.nlf`（2×2 的 f8 数组 + meta.t）。 */
async function frameBytes(t: number, chapter = "0.0", viz = "demo"): Promise<Uint8Array> {
  const values = new Float64Array([1, 2, 3, t]);
  const frame = new Frame({
    chapter,
    viz,
    meta: { t },
    strings: { data_source: "backend" },
    specs: [
      { name: "u", dtype: "f8", shape: [2, 2], order: "C", offset: 0, nbytes: 32, unit: "", label: "u" },
    ],
    arrays: new Map<string, TypedArray>([["u", values]]),
  });
  return encodeNlf(frame);
}

// --------------------------------------------------------------------------- //
// 假依赖
// --------------------------------------------------------------------------- //

/** 假时钟：`now()` 只读它，测试自己决定"过了多久"。 */
class FakeClock {
  ms = 0;
  readonly now = (): number => this.ms;
  advance(ms: number): void {
    this.ms += ms;
  }
}

interface TimerHandle {
  cb: () => void;
  ms: number;
  cancelled: boolean;
}

/** 假定时器：只登记回调，测试显式地"跑"它们（不真等）。 */
class FakeTimers {
  readonly handles: TimerHandle[] = [];
  readonly setTimer = (cb: () => void, ms: number): unknown => {
    const handle: TimerHandle = { cb, ms, cancelled: false };
    this.handles.push(handle);
    return handle;
  };
  readonly clearTimer = (handle: unknown): void => {
    (handle as TimerHandle).cancelled = true;
  };
  get pending(): TimerHandle[] {
    return this.handles.filter((h) => !h.cancelled);
  }
  /** 跑掉"下一个"未取消的定时器（按登记顺序）。 */
  runNext(): boolean {
    const handle = this.pending[0];
    if (handle === undefined) return false;
    handle.cancelled = true;
    handle.cb();
    return true;
  }
  /** 依次跑掉所有未取消的定时器（含它们新登记的），上限防止死循环。 */
  runAll(limit = 50): number {
    let ran = 0;
    while (ran < limit && this.runNext()) ran++;
    return ran;
  }
}

type WsBehavior = "open" | "error" | "silent" | "throw";

/** 假 WebSocket：行为可控，且把收到的控制消息记下来供断言。 */
class FakeWebSocket {
  static behavior: WsBehavior = "open";
  static instances: FakeWebSocket[] = [];
  static reset(): void {
    FakeWebSocket.behavior = "open";
    FakeWebSocket.instances = [];
  }
  static last(): FakeWebSocket {
    const inst = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
    assert.ok(inst, "还没有创建过 WebSocket");
    return inst;
  }

  readonly url: string;
  readonly sent: string[] = [];
  binaryType = "";
  closed = false;
  closeCode: number | undefined;
  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: unknown) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  onclose: ((ev: unknown) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
    if (FakeWebSocket.behavior === "throw") throw new Error("构造失败（假）");
    queueMicrotask(() => {
      if (FakeWebSocket.behavior === "open") this.onopen?.({});
      else if (FakeWebSocket.behavior === "error") this.onerror?.({ code: 1006, reason: "假错误" });
    });
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(code?: number): void {
    this.closed = true;
    this.closeCode = code;
  }

  /** 测试驱动：推一帧二进制（`.nlf` 字节）。 */
  pushFrame(bytes: Uint8Array): void {
    this.onmessage?.({ data: bytes });
  }

  /** 测试驱动：推一条文本控制消息。 */
  pushText(text: string): void {
    this.onmessage?.({ data: text });
  }

  /** 测试驱动：服务端关闭。 */
  drop(code = 1006): void {
    this.onclose?.({ code, reason: "" });
  }

  json(i = 0): Record<string, unknown> {
    const raw = this.sent[i];
    assert.ok(raw !== undefined, `没有第 ${i} 条控制消息（共 ${this.sent.length} 条）`);
    return JSON.parse(raw) as Record<string, unknown>;
  }
}

function okResponse(bytes: Uint8Array): Response {
  return {
    ok: true,
    status: 200,
    arrayBuffer: async () => bytes.slice().buffer,
  } as unknown as Response;
}

function statusResponse(status: number): Response {
  return { ok: false, status, arrayBuffer: async () => new ArrayBuffer(0) } as unknown as Response;
}

/** 假 fetch：记录 URL；handler 决定每次调用的结果（可以返回未决的 Promise 来模拟"在途"）。 */
class FakeFetch {
  readonly calls: string[] = [];
  handler: (url: string, n: number) => Response | Promise<Response> = () => statusResponse(404);
  readonly fetch = async (input: unknown): Promise<Response> => {
    const url = String(input);
    this.calls.push(url);
    return await this.handler(url, this.calls.length);
  };
}

/** 假静态目录：路径 → 字节。 */
class FakeStatic {
  readonly calls: string[] = [];
  readonly files = new Map<string, ArrayBuffer>();
  put(path: string, bytes: Uint8Array): void {
    this.files.set(path, bytes.slice().buffer);
  }
  readonly read = async (path: string): Promise<ArrayBuffer | null> => {
    this.calls.push(path);
    return this.files.get(path) ?? null;
  };
}

const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

interface Harness {
  clock: FakeClock;
  timers: FakeTimers;
  fetcher: FakeFetch;
  statics: FakeStatic;
  deps: FrameSourceDeps;
}

function harness(): Harness {
  const clock = new FakeClock();
  const timers = new FakeTimers();
  const fetcher = new FakeFetch();
  const statics = new FakeStatic();
  const deps: FrameSourceDeps = {
    now: clock.now,
    setTimer: timers.setTimer,
    clearTimer: timers.clearTimer,
    fetch: fetcher.fetch as unknown as typeof fetch,
    WebSocket: FakeWebSocket as unknown as typeof WebSocket,
    readStatic: statics.read,
  };
  return { clock, timers, fetcher, statics, deps };
}

/** 只有"跳过 WS"这一个目的：`ctor` 不是函数 → status 里说"没有 WebSocket"。 */
const NO_WS = { WebSocket: 0 as unknown as typeof WebSocket };

interface SourceOptions {
  params?: Record<string, number | boolean | string>;
  staticDir?: string;
  fps?: number;
  timeoutMs?: number;
  time?: { mode?: "none" | "stream" | "scrub"; dt?: number; tMax?: number; frames?: number };
  paramTypes?: Record<string, "int" | "float" | "bool" | "enum">;
  maxRestFailures?: number;
}

function makeSource(h: Harness, overrides: SourceOptions = {}, depsPatch: FrameSourceDeps = {}): ChapterFrameSource {
  return new ChapterFrameSource({
    chapter: "0.0",
    viz: "demo",
    baseUrl: "",
    params: overrides.params ?? { r: 0.4 },
    fps: overrides.fps ?? 30,
    timeoutMs: overrides.timeoutMs ?? 1000,
    deps: { ...h.deps, ...depsPatch },
    ...(overrides.staticDir === undefined ? {} : { staticDir: overrides.staticDir }),
    ...(overrides.time === undefined ? {} : { time: overrides.time }),
    ...(overrides.paramTypes === undefined ? {} : { paramTypes: overrides.paramTypes }),
    ...(overrides.maxRestFailures === undefined ? {} : { maxRestFailures: overrides.maxRestFailures }),
  });
}

/** 收集事件的最小记录器。 */
function record(src: ChapterFrameSource): {
  frames: Array<{ t: number; viz: string }>;
  statuses: SourceStatus[];
  errors: Error[];
} {
  const out = {
    frames: [] as Array<{ t: number; viz: string }>,
    statuses: [] as SourceStatus[],
    errors: [] as Error[],
  };
  src.on("frame", (frame, t) => out.frames.push({ t, viz: frame.viz }));
  src.on("status", (s) => out.statuses.push(s));
  src.on("error", (e) => out.errors.push(e));
  return out;
}

const levels = (statuses: SourceStatus[]): SourceLevel[] => statuses.map((s) => s.level);

/** 越过 L2 的拖动节流窗口（PROTOCOLS §4：60 ms）——真实拖动时时间本来就在走。 */
function pastThrottle(h: Harness): void {
  h.clock.advance(REST_THROTTLE_MS);
}

// --------------------------------------------------------------------------- //
// L1：WebSocket 成功
// --------------------------------------------------------------------------- //

test("L1：WS 连上后 level=ws、不降级，并发出 subscribe（params 走 base64url）", async () => {
  FakeWebSocket.reset();
  const h = harness();
  const src = makeSource(h, { params: { r: 0.4, nx: 41 } });
  const rec = record(src);
  await src.connect();

  assert.equal(src.level, "ws");
  assert.equal(src.status.degraded, false);
  assert.deepEqual(levels(rec.statuses), ["none", "ws"]);
  assert.match(rec.statuses[1].message, /L1 WebSocket 已连接/);

  const ws = FakeWebSocket.last();
  assert.equal(ws.url, "/ws/chapter/0.0");
  const sub = ws.json();
  assert.equal(sub["type"], "subscribe");
  assert.equal(sub["chapter"], "0.0");
  assert.equal(sub["viz"], "demo");
  assert.deepEqual(sub["params"], { r: 0.4, nx: 41 });
  const encoded = (sub["time"] as Record<string, unknown>)["fps"];
  assert.equal(encoded, 30);
  // 报文里的 params 必须能被自己的解码器还原（PROTOCOLS §3 的传输格式）
  assert.deepEqual(decodeParams(encodeParams({ r: 0.4, nx: 41 })), { r: 0.4, nx: 41 });
  assert.equal(h.fetcher.calls.length, 0, "WS 成功时不该碰 REST");
});

test("L1：页面空闲时定期发 ping，服务端 30s 空闲超时不会把连接踢掉", async () => {
  FakeWebSocket.reset();
  const h = harness();
  const src = makeSource(h);
  await src.connect();
  const ws = FakeWebSocket.last();
  const heartbeats = h.timers.pending.filter((timer) => timer.ms === 10_000);
  assert.equal(heartbeats.length, 1, "连接后应登记一个 10 s 心跳定时器");
  h.clock.advance(10_000);
  assert.equal(h.timers.runNext(), true);
  const last = ws.json(ws.sent.length - 1);
  assert.equal(last["type"], "ping");
  assert.equal(typeof last["t"], "number");
  // 心跳会重新登记自己；连接关闭后必须停掉，不能继续往已关闭的 socket 发
  assert.equal(h.timers.pending.filter((timer) => timer.ms === 10_000).length, 1);
  ws.drop(1006);
  await flush();
  assert.equal(h.timers.pending.filter((timer) => timer.ms === 10_000).length, 0);
  src.dispose();
});

test("L1：二进制帧按 frame.meta.t 对齐时间轴（不按到达顺序）", async () => {
  FakeWebSocket.reset();
  const h = harness();
  const src = makeSource(h);
  const rec = record(src);
  await src.connect();

  FakeWebSocket.last().pushFrame(await frameBytes(0.3));
  await flush();
  assert.equal(rec.frames.length, 1);
  assert.equal(rec.frames[0].t, 0.3);
  assert.equal(src.t, 0.3, "时间轴取 meta.t");
  assert.equal(src.receivedFrames, 1);

  // 乱序到达也以 meta.t 为准
  FakeWebSocket.last().pushFrame(await frameBytes(0.1));
  await flush();
  assert.equal(src.t, 0.1);
});

test("L1：request(t) 发 seek；startStream/stopStream 发 play/pause", async () => {
  FakeWebSocket.reset();
  const h = harness();
  const src = makeSource(h, { time: { mode: "stream", tMax: 1, dt: 0.1 } });
  await src.connect();
  const ws = FakeWebSocket.last();

  src.request(0.5);
  assert.deepEqual(ws.json(1), { type: "seek", t: 0.5 });

  src.startStream();
  assert.deepEqual(ws.json(2), { type: "play", t: 0 });
  src.stopStream();
  assert.deepEqual(ws.json(3), { type: "pause" });

  // L1 下 tick 不自己推进时间（时间由服务端的 meta.t 决定）
  h.clock.advance(1000);
  h.timers.runAll();
  assert.equal(src.t, 0);
});

test("L1：坏帧字节只发 error，不抛、不影响后续帧", async () => {
  FakeWebSocket.reset();
  const h = harness();
  const src = makeSource(h);
  const rec = record(src);
  await src.connect();

  FakeWebSocket.last().pushFrame(Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]));
  await flush();
  assert.equal(rec.errors.length, 1);
  assert.match(rec.errors[0].message, /帧解码失败/);
  assert.equal(src.level, "ws");

  FakeWebSocket.last().pushFrame(await frameBytes(0.2));
  await flush();
  assert.equal(rec.frames.length, 1);
});

test("L1：服务端 error 控制消息转成 error 事件（带 code）", async () => {
  FakeWebSocket.reset();
  const h = harness();
  const src = makeSource(h);
  const rec = record(src);
  await src.connect();
  FakeWebSocket.last().pushText(JSON.stringify({ type: "error", code: 42, message: "参数越界" }));
  assert.equal(rec.errors.length, 1);
  assert.match(rec.errors[0].message, /code=42/);
  assert.match(rec.errors[0].message, /参数越界/);
});

test("L1 掉线：自动降级到 L2（status 里说清是「断开」而不是「连不上」）", async () => {
  FakeWebSocket.reset();
  const h = harness();
  h.fetcher.handler = async () => okResponse(await frameBytes(0.25));
  const src = makeSource(h);
  const rec = record(src);
  await src.connect();
  assert.equal(src.level, "ws");

  FakeWebSocket.last().drop(1006);
  await flush();
  assert.equal(src.level, "rest");
  assert.deepEqual(levels(rec.statuses), ["none", "ws", "none", "rest"]);
  assert.match(rec.statuses[2].message, /WebSocket 断开/);
  assert.equal(rec.frames.length, 1, "降级后立刻补了一帧");
});

test("三层全挂后改参数会立刻重连，不需要刷新页面", async () => {
  FakeWebSocket.reset();
  const h = harness();
  h.fetcher.handler = async () => statusResponse(503);
  const src = makeSource(h);
  const rec = record(src);
  await src.connect();
  assert.equal(src.level, "ws");
  FakeWebSocket.last().drop(1006);
  await flush();
  assert.equal(src.level, "none");
  assert.equal(src.status.attempts, 3);
  // 新参数不再让后端失败：FakeWebSocket 下一次 open 后必须重新订阅并 seek。
  FakeWebSocket.behavior = "open";
  src.setParams({ r: 0.9 });
  await flush();
  assert.equal(src.level, "ws", "改参数应触发重连，而不是永远停在 none");
  const ws = FakeWebSocket.last();
  const subscribe = ws.sent.map((text) => JSON.parse(text) as Record<string, unknown>).find((m) => m["type"] === "subscribe");
  assert.deepEqual(subscribe?.["params"], { r: 0.9 });
  assert.ok(ws.sent.some((text) => text.includes('"type":"seek"')), "恢复后要把当前时刻重新要一遍");
  assert.ok(rec.statuses.some((s) => s.level === "ws" && s.message.includes("已连接")));
});

// --------------------------------------------------------------------------- //
// 降级：WS → REST → static → none
// --------------------------------------------------------------------------- //

test("降级：WS 失败 → REST 成功（level=rest、degraded=true、message 讲清原因）", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async () => okResponse(await frameBytes(0));
  const src = makeSource(h);
  const rec = record(src);
  await src.connect();

  assert.equal(src.level, "rest");
  assert.equal(src.status.degraded, true);
  assert.deepEqual(levels(rec.statuses), ["none", "none", "rest"]);
  assert.match(rec.statuses[1].message, /L1 WebSocket 不可用/);
  assert.match(rec.statuses[1].message, /连接失败/);
  assert.match(rec.statuses[2].message, /L2 REST 单帧可用/);
  assert.equal(h.fetcher.calls.length, 1);
  assert.match(h.fetcher.calls[0], /^\/api\/chapter\/0\.0\/frame\?viz=demo&params=/);
  assert.equal(rec.frames.length, 1, "探测帧直接交付，不白拿");
});

test("降级：WS 超时（假时钟 + 假定时器，不真等）→ REST", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "silent"; // 既不 open 也不 error：只能靠超时
  const h = harness();
  h.fetcher.handler = async () => okResponse(await frameBytes(0));
  const src = makeSource(h, { timeoutMs: 1234 });
  const rec = record(src);

  const pending = src.connect();
  await flush();
  assert.equal(src.level, "none", "还没结果时状态是 none");
  assert.equal(h.timers.pending.length, 1, "登记了一个超时定时器");
  assert.equal(h.timers.pending[0].ms, 1234);
  h.timers.runNext();
  await pending;

  assert.equal(src.level, "rest");
  assert.match(rec.statuses[1].message, /连接超时（1234 ms）/);
});

test("降级：WS 构造抛错 → REST", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "throw";
  const h = harness();
  h.fetcher.handler = async () => okResponse(await frameBytes(0));
  const src = makeSource(h);
  const rec = record(src);
  await src.connect();
  assert.equal(src.level, "rest");
  assert.match(rec.statuses[1].message, /构造失败/);
});

test("降级：运行环境没有 WebSocket → REST（不把「没有实现」当异常）", async () => {
  FakeWebSocket.reset();
  const h = harness();
  h.fetcher.handler = async () => okResponse(await frameBytes(0));
  const src = makeSource(h, {}, NO_WS);
  const rec = record(src);
  await src.connect();
  assert.equal(src.level, "rest");
  assert.match(rec.statuses[1].message, /当前运行环境没有 WebSocket/);
});

test("降级：REST 失败（HTTP 500）→ L3 静态帧命中", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async () => statusResponse(500);
  const rel = staticFrameRelPath("0.0", "demo", { r: 0.4 }, 0);
  h.statics.put(rel, await frameBytes(0));
  const src = makeSource(h);
  const rec = record(src);
  await src.connect();

  assert.equal(src.level, "static");
  assert.deepEqual(levels(rec.statuses), ["none", "none", "none", "static"]);
  assert.match(rec.statuses[2].message, /L2 REST 不可用/);
  assert.match(rec.statuses[2].message, /HTTP 500/);
  assert.match(rec.statuses[3].message, /L3 静态预计算帧/);
  assert.equal(h.statics.calls[0], rel);
  assert.equal(rec.frames.length, 1);
});

test("降级：全失败 → none，message 里三层原因都能看到", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async () => statusResponse(503);
  const src = makeSource(h);
  const rec = record(src);
  await src.connect();

  assert.equal(src.level, "none");
  assert.equal(src.status.degraded, true);
  assert.equal(src.status.attempts, 3);
  assert.deepEqual(levels(rec.statuses), ["none", "none", "none", "none"]);
  const last = rec.statuses[3].message;
  assert.match(last, /三层数据源都不可用/);
  assert.match(last, /L1/);
  assert.match(last, /L2/);
  assert.match(last, /L3/);
  assert.match(last, /检查后端/);
});

test("降级：没有 readStatic（静态目录不可达）→ none，理由具体", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async () => {
    throw new Error("network down");
  };
  const src = makeSource(h, {}, { readStatic: undefined });
  const rec = record(src);
  await src.connect();
  assert.equal(src.level, "none");
  assert.match(rec.statuses[3].message, /没有提供 readStatic/);
});

test("降级：静态帧缺失 → none，message 带上 t 与候选文件名", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async () => statusResponse(404);
  const src = makeSource(h, { time: { mode: "scrub", tMax: 1, dt: 0.25 } });
  const rec = record(src);
  await src.connect();
  assert.equal(src.level, "none");
  assert.match(rec.statuses[3].message, /data\/frames\/0\.0\/demo\//);
});

test("降级：connect() 幂等（连第二次不会重连）", async () => {
  FakeWebSocket.reset();
  const h = harness();
  const src = makeSource(h);
  await src.connect();
  await src.connect();
  assert.equal(FakeWebSocket.instances.length, 1);
});

// --------------------------------------------------------------------------- //
// 请求去重 / 在途缓存 / setParams
// --------------------------------------------------------------------------- //

test("REST：同一个 t 只请求一次（去重）", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async (url) => okResponse(await frameBytes(Number(/t=([\d.]+)/.exec(url)?.[1] ?? 0)));
  const src = makeSource(h);
  await src.connect();
  const before = h.fetcher.calls.length;

  pastThrottle(h);
  src.request(0.5);
  await flush();
  src.request(0.5);
  await flush();
  assert.equal(h.fetcher.calls.length, before + 1, "第二次同 t 被去重");
  assert.equal(src.receivedFrames, 2);
});

test("REST：在途时只保留最新请求，被顶掉的计入 droppedFrames", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  const deferred: Array<(res: Response) => void> = [];
  const bytesFor0 = await frameBytes(0);
  h.fetcher.handler = async (url) => {
    const t = Number(/t=([\d.]+)/.exec(url)?.[1] ?? 0);
    if (h.fetcher.calls.length === 1) return okResponse(bytesFor0); // 探测帧
    return await new Promise<Response>((resolve) => {
      deferred.push((res) => resolve(res));
      void t;
    });
  };
  const src = makeSource(h);
  await src.connect();
  assert.equal(src.level, "rest");

  pastThrottle(h);
  src.request(0.1); // 在途
  await flush();
  src.request(0.2); // 缓存位
  src.request(0.3); // 顶掉 0.2
  assert.equal(src.droppedFrames, 1);

  // 放行第一个在途请求 → 应该立刻发出 t=0.3
  deferred[0](okResponse(await frameBytes(0.1)));
  await flush();
  assert.equal(h.fetcher.calls.length, 3, "探测 + t=0.1 + t=0.3");
  assert.match(h.fetcher.calls[2], /t=0\.3/);
});

test("setParams：立刻按新参数重取当前时刻，并且丢弃旧参数的帧", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  const pendingResponses: Array<(res: Response) => void> = [];
  h.fetcher.handler = async () => {
    if (h.fetcher.calls.length === 1) return okResponse(await frameBytes(0)); // 探测（旧参数）
    return await new Promise<Response>((resolve) => pendingResponses.push(resolve));
  };
  const src = makeSource(h, { params: { r: 0.4 } });
  const rec = record(src);
  await src.connect();
  assert.equal(src.level, "rest");

  pastThrottle(h);
  src.request(0.5); // 用旧参数在途
  await flush();
  src.setParams({ r: 0.9 }); // 换参数：旧请求作废，重取当前时刻 t=0
  await flush();

  const urls = h.fetcher.calls;
  assert.equal(urls.length, 3, "探测 + 旧参数 t=0.5 + 新参数 t=0");
  assert.equal(decodeParams(/params=([^&]+)/.exec(urls[1])?.[1] ?? "")?.r, 0.4);
  assert.equal(decodeParams(/params=([^&]+)/.exec(urls[2])?.[1] ?? "")?.r, 0.9);
  assert.match(urls[2], /t=0(&|$)/);

  // 旧帧到达：必须被丢弃（画出来就是撒谎）
  const framesBefore = rec.frames.length;
  pendingResponses[0](okResponse(await frameBytes(0.5)));
  await flush();
  assert.equal(rec.frames.length, framesBefore);
  assert.equal(src.droppedFrames, 1);
  assert.equal(src.t, 0, "时间轴没有被旧帧带跑");
});

test("setParams：WS 模式只发 params 增量消息，重连由服务端处理", async () => {
  FakeWebSocket.reset();
  const h = harness();
  const src = makeSource(h);
  await src.connect();
  src.setParams({ r: 0.9 });
  assert.deepEqual(FakeWebSocket.last().json(1), { type: "params", params: { r: 0.9 } });
});

// --------------------------------------------------------------------------- //
// 流式推进（注入时钟）
// --------------------------------------------------------------------------- //

test("startStream：按注入时钟与 fps 推进，到 tMax 自动停", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async (url) => okResponse(await frameBytes(Number(/t=([\d.e+-]+)/.exec(url)?.[1] ?? 0)));
  const src = makeSource(h, { fps: 10, time: { mode: "stream", dt: 0.5, tMax: 1 } });
  await src.connect();
  assert.equal(src.level, "rest");
  const before = h.fetcher.calls.length;

  src.startStream();
  // 时钟未动：不能推进
  h.timers.runNext();
  await flush();
  assert.equal(h.fetcher.calls.length, before, "时间没走就不该取帧");

  for (let i = 0; i < 2; i++) {
    h.clock.advance(100);
    h.timers.runNext();
    await flush();
  }
  assert.equal(h.fetcher.calls.length, before + 2);
  assert.match(h.fetcher.calls[before], /t=0\.5/);
  assert.match(h.fetcher.calls[before + 1], /t=1(&|$)/);
  assert.equal(src.t, 1);

  // 再推一次：越过 tMax → 停在末端并停止流
  h.clock.advance(100);
  h.timers.runNext();
  await flush();
  assert.equal(src.t, 1);
  assert.equal(h.fetcher.calls.length, before + 2, "到达末端后不再取帧");
  h.clock.advance(1000);
  assert.equal(h.timers.runAll(), 0, "流已停止，没有待跑的定时器");
});

test("stopStream：之后时钟再走也不推进；dispose 后一切静默", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async (url) => okResponse(await frameBytes(Number(/t=([\d.e+-]+)/.exec(url)?.[1] ?? 0)));
  const src = makeSource(h, { fps: 10, time: { mode: "stream", dt: 0.5, tMax: 10 } });
  await src.connect();
  const rec = record(src);
  const before = h.fetcher.calls.length;

  src.startStream();
  h.clock.advance(100);
  h.timers.runNext();
  await flush();
  assert.equal(h.fetcher.calls.length, before + 1);

  src.stopStream();
  h.clock.advance(1000);
  h.timers.runAll();
  await flush();
  assert.equal(h.fetcher.calls.length, before + 1, "停流后不再取帧");

  const framesBefore = rec.frames.length;
  const statusesBefore = rec.statuses.length;
  src.dispose();
  src.request(5);
  src.startStream();
  h.timers.runAll();
  await flush();
  assert.equal(rec.frames.length, framesBefore);
  assert.equal(rec.statuses.length, statusesBefore);
  assert.equal(rec.errors.length, 0);
});

test("dispose：关掉 WebSocket、清空回调，之后推帧不再触发任何事件", async () => {
  FakeWebSocket.reset();
  const h = harness();
  const src = makeSource(h);
  const rec = record(src);
  await src.connect();
  const ws = FakeWebSocket.last();

  src.dispose();
  assert.equal(ws.closed, true);
  ws.pushFrame(new Uint8Array(0));
  ws.pushText(JSON.stringify({ type: "error", message: "不该被看到" }));
  ws.drop(1006);
  await flush();
  assert.equal(rec.frames.length, 0);
  assert.equal(rec.errors.length, 0);
  assert.deepEqual(levels(rec.statuses), ["none", "ws"]);
  src.dispose(); // 幂等
});

// --------------------------------------------------------------------------- //
// REST 退避
// --------------------------------------------------------------------------- //

test("REST 退避：失败后要先等退避窗口，窗口过了由定时器补发最新请求", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async (url) => {
    if (/t=0(&|$)/.test(url)) return okResponse(await frameBytes(0));
    return statusResponse(500);
  };
  const src = makeSource(h, { time: { mode: "scrub", tMax: 10, dt: 0.5 } });
  const rec = record(src);
  await src.connect();
  assert.equal(src.level, "rest");

  pastThrottle(h);
  src.request(0.5);
  await flush();
  assert.equal(h.fetcher.calls.length, 2);
  assert.equal(rec.errors.length, 1, "失败要发 error（UI 可以提示）");
  const failedAt = h.clock.ms;

  // 退避窗口内：只登记，不真发请求
  src.request(1.0);
  await flush();
  assert.equal(h.fetcher.calls.length, 2, "退避窗口内不发请求");

  h.clock.advance(REST_BACKOFF_BASE_MS);
  assert.equal(h.timers.runNext(), true, "退避定时器在窗口末尾触发");
  await flush();
  assert.equal(h.fetcher.calls.length, 3, "窗口过后补发最新请求");
  assert.match(h.fetcher.calls[2], /t=1(&|$)/);
  assert.ok(h.clock.ms - failedAt >= REST_BACKOFF_BASE_MS);
});

test("REST 连续失败到阈值 → 降级到 L3（不再继续重试 REST）", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async (url) => (/t=0(&|$)/.test(url) ? okResponse(await frameBytes(0)) : statusResponse(500));
  const staticBytes = await frameBytes(0);
  for (const rel of staticFrameCandidates("0.0", "demo", { r: 0.4 }, 0)) h.statics.put(rel, staticBytes);
  const src = makeSource(h, { time: { mode: "scrub", tMax: 10, dt: 0.5 }, maxRestFailures: 2 });
  const rec = record(src);
  await src.connect();

  pastThrottle(h);
  src.request(0.5); // 第 1 次失败（此时没有后续请求 → 不排重试）
  await flush();
  assert.equal(src.level, "rest");
  assert.equal(h.timers.pending.length, 0, "没有待取的时刻就不该空转重试");
  src.request(1.0); // 退避窗口内的新请求：登记为 pending 并排重试
  await flush();
  assert.equal(h.fetcher.calls.length, 2, "退避窗口内不真发请求");
  h.clock.advance(REST_BACKOFF_BASE_MS);
  assert.equal(h.timers.runNext(), true);
  await flush();
  // 第 2 次失败 → 降级：先探静态帧（t=0 命中）
  assert.equal(src.level, "static");
  assert.match(rec.statuses[rec.statuses.length - 1].message, /L3 静态预计算帧/);
  assert.ok(
    rec.statuses.some((s) => /连续 2 次失败/.test(s.message)),
    "降级理由要说清是连续失败",
  );
});

// --------------------------------------------------------------------------- //
// L3 路径与 params 编码（与 Python 逐字节对齐）
// --------------------------------------------------------------------------- //

test("staticFrameRelPath 与 server/frames.py::static_frame_path 逐字节一致", () => {
  const rel = staticFrameRelPath("0.0", "demo", { nx: 41, r: 0.4, t_end: 0.05 }, 0.15);
  assert.equal(rel, "data/frames/0.0/demo/8800ead731cf946f_t0.15.nlf");
  // 独立复算一遍 Python 的公式（node:crypto + 同一份规范 JSON）
  const canonical = '{"nx":41,"r":0.4,"t_end":0.05}';
  assert.equal(canonicalParamsJson({ nx: 41, r: 0.4, t_end: 0.05 }), canonical);
  const digest = createHash("sha256").update(`0.0/demo/${canonical}`).digest("hex").slice(0, 16);
  assert.equal(digest, "8800ead731cf946f");
  assert.equal(rel, `data/frames/0.0/demo/${digest}_t${formatPyG(0.15, 6)}.nlf`);
});

test("paramsHash：Python json.dumps 的数字写法（1.0 与 1 是两个不同的哈希）", () => {
  // 下面四个常量都是 python3 -c 直接算出来的
  assert.equal(paramsHash("0.0", "demo", { nx: 41, r: 0.4, t_end: 0.05 }), "8800ead731cf946f");
  assert.equal(paramsHash("0.0", "demo", { omega: 1.0, zeta: 1.0, n_periods: 3 }, { floatKeys: ["omega", "zeta"] }), "dc12d54ec876b02e");
  assert.equal(paramsHash("0.0", "demo", { n: 24, pivot: "partial", rho: 10.0 }, { floatKeys: ["rho"] }), "aafd92b6a2d1eee2");
  assert.equal(paramsHash("0.0", "demo", { tol: 1e-6, h: 0.05 }), "3ca100cc7aeea752");
  // 同一个整值：int 写法与 float 写法必须分开
  assert.notEqual(
    paramsHash("0.0", "demo", { zeta: 1 }),
    paramsHash("0.0", "demo", { zeta: 1 }, { floatKeys: ["zeta"] }),
  );
  assert.equal(canonicalParamsJson({ zeta: 1 }), '{"zeta":1}');
  assert.equal(canonicalParamsJson({ zeta: 1 }, { floatKeys: ["zeta"] }), '{"zeta":1.0}');
  assert.equal(canonicalParamsJson({ b: true, a: "x", n: 1 }), '{"a":"x","b":true,"n":1}');
});

test("staticFrameCandidates：类型未知时同时给出 int/float 两种文件名", () => {
  const two = staticFrameCandidates("0.0", "demo", { nx: 41, r: 0.4 }, 0);
  assert.equal(two.length, 2);
  assert.notEqual(two[0], two[1]);
  assert.ok(two.every((p) => p.startsWith("data/frames/0.0/demo/")));
  const one = staticFrameCandidates("0.0", "demo", { nx: 41, r: 0.4 }, 0, { floatKeys: ["r"] });
  assert.equal(one.length, 1);
  assert.equal(one[0], two[0], "给定类型时只保留正确的那一个（顺序：先 int 写法）");
  assert.equal(staticFrameCandidates("0.0", "demo", { r: 0.4 }, 0).length, 1, "没有整值参数就只有一个候选");
});

test("formatPyG / pythonFloatRepr 复刻 Python 的 %.6g 与 repr", () => {
  const g6: Array<[number, string]> = [
    [0, "0"],
    [0.05, "0.05"],
    [0.2, "0.2"],
    [20 / 19, "1.05263"],
    [1 / 3, "0.333333"],
    [1e-5, "1e-05"],
    [0.0001, "0.0001"],
    [0.000123456789, "0.000123457"],
    [0.000999999, "0.000999999"],
    [123456.789, "123457"],
    [1234567.89, "1.23457e+06"],
    [100000, "100000"],
    [1e6, "1e+06"],
    [1e15, "1e+15"],
    [1e16, "1e+16"],
    [0.1 + 0.2, "0.3"],
    [-0, "-0"],
    [2, "2"],
  ];
  for (const [value, want] of g6) assert.equal(formatPyG(value, 6), want, `%.6g(${value})`);

  const repr: Array<[number, string]> = [
    [1.0, "1.0"],
    [3.0, "3.0"],
    [0.05, "0.05"],
    [0.1 + 0.2, "0.30000000000000004"],
    [1e-6, "1e-06"],
    [1e-5, "1e-05"],
    [1e15, "1000000000000000.0"],
    [1e16, "1e+16"],
    [1e21, "1e+21"],
    [-0, "-0.0"],
  ];
  for (const [value, want] of repr) assert.equal(pythonFloatRepr(value), want, `repr(${value})`);
});

test("sha256Hex 与 node:crypto 完全一致（含多块与空输入）", () => {
  for (const text of ["", "abc", "0.0/demo/{}", "中文-测试-中文", "x".repeat(1000)]) {
    const bytes = new TextEncoder().encode(text);
    assert.equal(sha256Hex(bytes), createHash("sha256").update(text).digest("hex"), `sha256(${text.slice(0, 8)})`);
  }
});

test("encodeParams/decodeParams：与 Python urlsafe_b64encode 逐字符一致、键排序、坏输入返回 null", () => {
  const params = { nx: 41, r: 0.4, pivot: "partial", use: true };
  const encoded = encodeParams(params);
  // python3: urlsafe_b64encode(json.dumps(p, sort_keys=True, separators=(",",":")).encode()).rstrip("=")
  assert.equal(encoded, "eyJueCI6NDEsInBpdm90IjoicGFydGlhbCIsInIiOjAuNCwidXNlIjp0cnVlfQ");
  assert.deepEqual(decodeParams(encoded), params);
  // 键序不影响结果（确定性 → 可做缓存键）
  assert.equal(encodeParams({ use: true, pivot: "partial", r: 0.4, nx: 41 }), encoded);
  // 非 ASCII 走 UTF-8
  assert.deepEqual(decodeParams(encodeParams({ label: "中文" })), { label: "中文" });

  assert.equal(decodeParams(""), null);
  assert.equal(decodeParams("!!!!"), null);
  assert.equal(decodeParams("eyJhIjpbMSwyXX0"), null, "数组不是合法参数表");
  assert.equal(decodeParams(encodeParams({ a: 1 }).slice(0, 6)), null, "截断的 base64");
  assert.equal(decodeParams(123 as unknown), null);
  // 非有限数不能进 JSON
  assert.throws(() => encodeParams({ bad: Number.NaN }), /有限数/);
  assert.throws(() => canonicalParamsJson({ nested: {} as unknown as number }), /必须是有限数/);
});

test("URL 拼装：ws/http 协议映射、相对路径、批量帧端点", () => {
  assert.equal(wsChapterUrl("", "1.2"), "/ws/chapter/1.2");
  assert.equal(wsChapterUrl("http://localhost:8000", "1.2"), "ws://localhost:8000/ws/chapter/1.2");
  assert.equal(wsChapterUrl("https://lab.example.com/", "0.0"), "wss://lab.example.com/ws/chapter/0.0");
  assert.equal(wsChapterUrl("ws://h:9000", "0.0"), "ws://h:9000/ws/chapter/0.0");
  const frame = restFrameUrl("http://h:8000/", "0.0", "demo", { r: 0.4 }, 0.1);
  assert.equal(
    frame,
    `http://h:8000/api/chapter/0.0/frame?viz=demo&params=${encodeParams({ r: 0.4 })}&t=0.1`,
  );
  assert.match(restFramesUrl("", "0.0", "demo", { r: 0.4 }, 60, 0.5), /\/frames\?viz=demo&params=.*&t=0\.5&n=60$/);
});

test("REST URL 的时间保留精确值：%.6g 不得把 tMax 向上舍入到界外", () => {
  const tMax = 6.2831853;
  const url = restFrameUrl("http://h:8000", "3.2", "stiff_dilemma", { method: "be" }, tMax);
  const t = Number(new URL(url).searchParams.get("t"));
  assert.equal(t, tMax, "线上时间必须与服务端 tMax 逐位一致");
  assert.ok(t <= tMax, "舍入后的 t 不能大于 tMax（3.2 章拖到末尾曾因 6.28319 > 6.2831853 报 400）");
});

test("静态帧命中路径会拼上 staticDir（浏览器里就是 /data 前缀）", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async () => statusResponse(500);
  const rel = staticFrameRelPath("0.0", "demo", { r: 0.4 }, 0);
  h.statics.put(`/data/${rel}`, await frameBytes(0));
  const src = makeSource(h, { staticDir: "/data" });
  await src.connect();
  assert.equal(src.level, "static");
  assert.equal(h.statics.calls[0], `/data/${rel}`);
});

test("REST：在途请求卡住时按 timeoutMs 放行，并且不冻住后续请求", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  const hung: Array<(res: Response) => void> = [];
  h.fetcher.handler = async (url) => {
    if (/t=0(&|$)/.test(url) && h.fetcher.calls.length === 1) return okResponse(await frameBytes(0));
    return await new Promise<Response>((resolve) => hung.push(resolve));
  };
  const src = makeSource(h, { timeoutMs: 500, time: { mode: "scrub", tMax: 10, dt: 0.5 } });
  await src.connect();
  assert.equal(src.level, "rest");

  pastThrottle(h);
  src.request(0.5); // 服务端不响应
  await flush();
  assert.equal(h.fetcher.calls.length, 2);
  src.request(1.0); // 在途缓存位
  assert.equal(h.fetcher.calls.length, 2, "在途时不并发发请求");

  h.clock.advance(500);
  assert.equal(h.timers.runNext(), true, "在途超时定时器触发");
  await flush();
  assert.equal(src.droppedFrames, 1, "卡住的那一帧记成丢弃");
  assert.equal(h.fetcher.calls.length, 3, "放行后立刻补发最新请求");
  assert.match(h.fetcher.calls[2], /t=1(&|$)/);
});

test("off()：注销单个回调，其它订阅者不受影响", async () => {
  FakeWebSocket.reset();
  const h = harness();
  const src = makeSource(h);
  let a = 0;
  let b = 0;
  const cbA = (): void => void a++;
  const cbB = (): void => void b++;
  src.on("frame", cbA);
  src.on("frame", cbB);
  await src.connect();
  src.off("frame", cbA);
  FakeWebSocket.last().pushFrame(await frameBytes(0.2));
  await flush();
  assert.equal(a, 0);
  assert.equal(b, 1);
});

test("level=none 时 request 静默（不刷错误），status 已经把原因写在脸上", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async () => statusResponse(500);
  const src = makeSource(h);
  const rec = record(src);
  await src.connect();
  assert.equal(src.level, "none");
  const errorsBefore = rec.errors.length;
  src.request(0.5);
  src.request(Number.NaN); // 非法 t 仍然要报错（这是调用方的 bug）
  await flush();
  assert.equal(rec.errors.length, errorsBefore + 1);
  assert.match(rec.errors[errorsBefore].message, /有限数/);
});

test("参数类型表（paramTypes）让 L3 只用正确的那一个文件名", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async () => statusResponse(500);
  const rel = staticFrameRelPath("0.0", "demo", { nx: 41, r: 0.4 }, 0, { floatKeys: ["r"] });
  h.statics.put(rel, await frameBytes(0));
  const src = makeSource(h, { params: { nx: 41, r: 0.4 }, paramTypes: { nx: "int", r: "float" } });
  await src.connect();
  assert.equal(src.level, "static");
  assert.deepEqual(h.statics.calls, [rel], "给定类型时只试一个候选");
});

test("L2 拖动节流（PROTOCOLS §4，60 ms）：窗口内的新请求只更新「最新」，到窗口末尾补发", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  h.fetcher.handler = async (url) => okResponse(await frameBytes(Number(/t=([\d.e+-]+)/.exec(url)?.[1] ?? 0)));
  const src = makeSource(h);
  await src.connect();
  assert.equal(src.level, "rest");
  const before = h.fetcher.calls.length;

  // 探测刚结束，此刻就在节流窗口内
  src.request(0.1);
  src.request(0.2);
  src.request(0.3);
  await flush();
  assert.equal(h.fetcher.calls.length, before, "窗口内一个请求都不发");
  assert.equal(src.droppedFrames, 2, "0.1 与 0.2 都被顶掉，各记一笔丢帧");

  h.clock.advance(REST_THROTTLE_MS);
  assert.equal(h.timers.runNext(), true);
  await flush();
  assert.equal(h.fetcher.calls.length, before + 1, "窗口末尾只补发最新那个 t");
  assert.match(h.fetcher.calls[before], /t=0\.3(&|$)/);
});

test("REST 失败过的 t 在退避窗口结束后可以重试（不会被去重缓存永久挡住）", async () => {
  FakeWebSocket.reset();
  FakeWebSocket.behavior = "error";
  const h = harness();
  let failNext = false;
  h.fetcher.handler = async (url) => {
    if (/t=0(&|$)/.test(url) && h.fetcher.calls.length === 1) return okResponse(await frameBytes(0));
    return failNext ? statusResponse(500) : okResponse(await frameBytes(Number(/t=([\d.e+-]+)/.exec(url)?.[1] ?? 0)));
  };
  const src = makeSource(h, { time: { mode: "scrub", tMax: 10, dt: 0.5 } });
  await src.connect();

  pastThrottle(h);
  failNext = true;
  src.request(0.5);
  await flush();
  assert.equal(h.fetcher.calls.length, 2, "第一次真的发了（失败）");

  failNext = false;
  h.clock.advance(REST_BACKOFF_BASE_MS);
  src.request(0.5); // 退回同一个 t
  await flush();
  assert.equal(h.fetcher.calls.length, 3, "失败过的 t 必须能再来一次");
  assert.match(h.fetcher.calls[2], /t=0\.5(&|$)/);
  assert.equal(src.t, 0.5, "重试成功后时间轴跟上");
});
