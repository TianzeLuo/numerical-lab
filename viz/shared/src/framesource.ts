/**
 * 数据源与 **L1 → L2 → L3 降级**（`docs/PROTOCOLS.md` §3、§4）。
 *
 * 三层递降，前端只做渲染与显示空间插值：
 *
 * | 层 | 通道 | 端点 | 失效后的表现 |
 * |---|---|---|---|
 * | L1 | WebSocket 流 | `ws://<host>/ws/chapter/<chapter_id>` | 实时算，所有参数量连续可拖 |
 * | L2 | REST 单帧 | `GET /api/chapter/<id>/frame?viz=..&params=<base64url json>` | 按需取帧（拖动节流 + 失败退避） |
 * | L3 | 静态预计算 | `data/frames/<chapter>/<viz>/<params_hash>_t<t>.nlf` | 只覆盖预计算过的参数/时刻点 |
 *
 * 设计要点（每一条都有对应的测试）：
 *
 * 1. **降级必须说人话**：每次降级都发 `status` 事件，`message` 里带上"为什么"（哪一层、
 *    什么原因、还剩什么可用）。用户看到的不是"加载失败"，而是"L1 不可用（连接超时）：
 *    改用 L2 REST 单帧"。
 * 2. **`params` 一律按 §3 走 `base64url(JSON)`**：`encodeParams` / `decodeParams` 是纯函数；
 *    键排序 + 无填充，同一个参数表永远得到同一串（可做缓存键、可逐字节比较）。
 * 3. **L3 的文件名必须与 Python 落盘一致**：`server/frames.py::static_frame_path` 用
 *    `sha256(f"{chapter}/{viz}/" + json.dumps(params, sort_keys=True, separators=(",",":")))[:16]`
 *    和 `f"{t:.6g}"`。本文件里 `paramsHash` / `formatPyG` / `pythonFloatRepr` 就是为了在
 *    **不依赖 Python** 的前提下逐字节复刻这两个格式（顺带复刻了 Python 的数字写法：
 *    `repr(1e-6) == "1e-06"` 而不是 JS 的 `"0.000001"`）。
 * 4. **时间轴以 `frame.meta.t` 对齐**（§2.2）：到达顺序不参与推时间。
 * 5. **在途只留一个"最新"请求**：拖动时旧请求作废、新请求覆盖缓存位，被作废的帧计入
 *    `droppedFrames`（可读，UI 可以显示"因来不及而丢掉的帧数"）。
 * 6. **时钟/定时器/网络全部可注入**：`deps.now` / `deps.setTimer` / `deps.fetch` /
 *    `deps.WebSocket` / `deps.readStatic`。没有浏览器的测试环境里，`startStream` 靠
 *    假定时器 + 假时钟推进，一秒都不用真等。
 *
 * 本模块**不碰 DOM**（不 import 任何浏览器全局，只在运行时读 `globalThis.fetch` 等）。
 */

import { Frame } from "./frame.js";
import { decodeNlfAsync } from "./nlf.js";
import { base64UrlDecode, base64UrlEncode, utf8Decode, utf8Encode } from "./share.js";
import type { ParamValues } from "./share.js";
import type { ParamType, TimeMode } from "./manifest.js";

/** 数据源层级；`none` 表示三层都不可用（画面保持空白，但 UI 必须显示原因）。 */
export type SourceLevel = "ws" | "rest" | "static" | "none";

/** 当前数据源状态快照（`status` getter 返回副本，外部改不坏内部状态）。 */
export interface SourceStatus {
  level: SourceLevel;
  /** 是否处于降级态（`ws` 以外一律为 `true`）。 */
  degraded: boolean;
  /** 人类可读的"现在用的是哪一层、为什么"；UI 直接显示这一句。 */
  message: string;
  /** 到目前为止尝试过的层数（0 = 刚开始连，1 = 试过 WS，2 = 试过 REST，3 = 试过静态）。 */
  attempts: number;
}

/** 可注入依赖。全部可选：缺省用运行时的 `fetch` / `WebSocket` / `setTimeout`。 */
export interface FrameSourceDeps {
  fetch?: typeof fetch;
  WebSocket?: typeof WebSocket;
  now?: () => number;
  /**
   * 读一个静态帧文件。
   *
   * 收到的是**已经拼好 `staticDir` 的路径**（例如 `data/frames/0.0/heat_ftcs/ab12…_t0.nlf`，
   * 或 `staticDir="/data"` 时的 `/data/frames/…`）；文件不存在时返回 `null`（不要抛）。
   * 浏览器里通常是 `(p) => fetch(p).then(r => r.ok ? r.arrayBuffer() : null)`。
   */
  readStatic?: (path: string) => Promise<ArrayBuffer | null>;
  /** 可注入的定时器（测试用假实现，配合 `now` 精确推进，不必真等）。 */
  setTimer?: (cb: () => void, ms: number) => unknown;
  /** 与 `setTimer` 配套的清理函数。 */
  clearTimer?: (handle: unknown) => void;
}

/** 构造选项。 */
export interface FrameSourceOptions {
  chapter: string;
  viz: string;
  /** 站点根（`""` = 同源相对路径；`http://host:8000` 会自动换成 `ws://`）。 */
  baseUrl?: string;
  params?: ParamValues;
  /** 静态帧目录（缺省 `""`：`readStatic` 直接收到 `data/frames/...` 相对路径）。 */
  staticDir?: string;
  deps?: FrameSourceDeps;
  /** 流式推进帧率（缺省 30）；同时决定 L2/L3 的取帧节奏。 */
  fps?: number;
  /** 连接 / 取帧超时（ms，缺省 5000）。 */
  timeoutMs?: number;
  /**
   * 参数类型表（来自 manifest）。只影响 **L3 文件名的数字写法**：
   * `float` 参数在 Python 侧 `json.dumps` 里是 `1.0`，JS 里 `1`——不告诉它就没法逐字节对齐。
   * 不传时会同时尝试两种写法（最多读 2 个候选文件）。
   */
  paramTypes?: Record<string, ParamType>;
  /** 订阅时上报的时间轴规格（PROTOCOLS §2.1 的 `time` 段）。 */
  time?: { mode?: TimeMode; dt?: number; tMax?: number; frames?: number };
  /** REST 连续失败多少次后降级到 L3（缺省 3）。 */
  maxRestFailures?: number;
}

/** 参数哈希 / 路径的选项。 */
export interface ParamsHashOptions {
  /**
   * 哪些参数按"浮点"写进 JSON（Python 侧 `float` 参数即便整值也写成 `1.0`）。
   * 与 manifest 的 `type: "float"` 对应。
   */
  floatKeys?: readonly string[];
}

/** 事件名 → 回调签名（`on` 的重载用它保持类型安全）。 */
export interface FrameSourceEvents {
  frame: (frame: Frame, t: number) => void;
  status: (status: SourceStatus) => void;
  error: (error: Error) => void;
}

/** REST 取帧的拖动节流（PROTOCOLS §4：「拖动节流 60 ms」）。 */
export const REST_THROTTLE_MS = 60;

/** REST 退避起点与上限。 */
export const REST_BACKOFF_BASE_MS = 200;
export const REST_BACKOFF_MAX_MS = 5000;

/** L3 请求失败后的最小重试间隔（避免同一时刻反复读同一个不存在的文件）。 */
const STATIC_RETRY_MS = 250;

// --------------------------------------------------------------------------- //
// Python 兼容的数字 / JSON 写法
// --------------------------------------------------------------------------- //

/**
 * Python `repr(float)` 的等价物（`json.dumps` 用的就是它）。
 *
 * 与 JS `String(v)` 的差别只在"什么时候用科学计数法"和"整值浮点要不要点零"：
 * `repr(1e-6) == "1e-06"`（JS 给 `"0.000001"`）、`repr(1e15) == "1000000000000000.0"`
 * （JS 给 `"1000000000000000"`）、`repr(1e16) == "1e+16"`。这些差别会直接改变
 * `params_hash`，进而决定 L3 能不能命中，所以必须逐字节对齐。
 */
export function pythonFloatRepr(value: number): string {
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  if (value === 0) return Object.is(value, -0) ? "-0.0" : "0.0";

  const negative = value < 0;
  // toExponential() 不给参数时给"最短且能往返"的数字串——与 Python 的 repr 同源（都是 Grisu/Ryu 一类的算法）。
  const [, lead = "0", rest = "", expText = "+0"] = /^(\d)(?:\.(\d+))?e([+-]\d+)$/.exec(
    Math.abs(value).toExponential(),
  ) ?? [];
  const digits = lead + rest;
  const exp = Number(expText);
  // decpt：小数点在 digits 里的位置（Python 内部表示法）。
  const decpt = exp + 1;
  let body: string;
  if (decpt > -4 && decpt <= 16) {
    if (decpt <= 0) body = `0.${"0".repeat(-decpt)}${digits}`;
    else if (decpt >= digits.length) body = `${digits}${"0".repeat(decpt - digits.length)}.0`;
    else body = `${digits.slice(0, decpt)}.${digits.slice(decpt)}`;
  } else {
    const mantissa = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    body = `${mantissa}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  return negative ? `-${body}` : body;
}

/**
 * Python `format(v, f".{precision}g")` 的等价物（`.nlf` 文件名里的 `t` 就是 `%.6g`）。
 *
 * `%g` 的三条规则：有效数字 `precision` 位；指数在 `[-4, precision)` 内用定点、否则科学计数法；
 * 尾随零全部去掉（`2.0 → "2"`、`0.5 → "0.5"`）。
 */
export function formatPyG(value: number, precision = 6): string {
  if (Number.isNaN(value)) return "nan";
  if (value === Number.POSITIVE_INFINITY) return "inf";
  if (value === Number.NEGATIVE_INFINITY) return "-inf";
  if (value === 0) return Object.is(value, -0) ? "-0" : "0";
  const p = Math.max(1, Math.min(21, Math.trunc(precision)));

  const negative = value < 0;
  const text = Math.abs(value).toPrecision(p); // 恰好 p 位有效数字（可能带 e）
  let digits: string;
  let exp: number;
  const eIndex = text.indexOf("e");
  if (eIndex >= 0) {
    digits = text.slice(0, eIndex).replace(".", "");
    exp = Number(text.slice(eIndex + 1));
  } else {
    const dot = text.indexOf(".");
    const intPart = dot >= 0 ? text.slice(0, dot) : text;
    const fracPart = dot >= 0 ? text.slice(dot + 1) : "";
    const all = intPart + fracPart;
    let lead = 0;
    while (lead < all.length - 1 && all[lead] === "0") lead++;
    digits = all.slice(lead);
    exp = intPart.length - 1 - lead;
  }

  let body: string;
  if (exp >= -4 && exp < p) {
    const decimals = p - 1 - exp;
    if (decimals <= 0) {
      body = digits + "0".repeat(-decimals);
    } else {
      const intPart = exp < 0 ? "0" : digits.slice(0, exp + 1);
      const fracRaw = (exp < 0 ? "0".repeat(-exp - 1) + digits : digits.slice(exp + 1)).padEnd(decimals, "0");
      const frac = fracRaw.replace(/0+$/, "");
      body = frac === "" ? intPart : `${intPart}.${frac}`;
    }
  } else {
    const frac = digits.slice(1).replace(/0+$/, "");
    body = `${digits[0]}${frac === "" ? "" : `.${frac}`}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  return negative ? `-${body}` : body;
}

/** JSON 里的字符串字面量（`ensure_ascii=False`：非 ASCII 原样保留）。 */
function jsonString(text: string): string {
  return JSON.stringify(text) ?? '""';
}

/** 把标量写成 Python `json.dumps` 的样子（数字部分见 `pythonFloatRepr`）。 */
function jsonScalar(value: number | boolean | string, asFloat: boolean, key: string): string {
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return jsonString(value);
  if (!Number.isFinite(value)) {
    throw new TypeError(`params['${key}'] 必须是有限数，得到 ${String(value)}`);
  }
  if (!asFloat && Number.isInteger(value)) {
    // 安全整数直接写；超出安全范围时用 BigInt 补回精确十进制（Python 的 int 没有 2^53 限制）。
    return Number.isSafeInteger(value) ? String(value) : BigInt(value).toString();
  }
  return pythonFloatRepr(value);
}

/**
 * 参数表的**规范 JSON**：键排序、无空格、Python 数字写法。
 *
 * 与 `server/frames.py::params_hash` 里那句
 * `json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))` 等价。
 * 非标量 / 非有限数会抛 `TypeError`——静默丢参数会让 L3 悄悄读到**别的**参数点的帧。
 */
export function canonicalParamsJson(params: ParamValues, opts: ParamsHashOptions = {}): string {
  if (typeof params !== "object" || params === null || Array.isArray(params)) {
    throw new TypeError("params 必须是对象");
  }
  const floatKeys = new Set(opts.floatKeys ?? []);
  const parts: string[] = [];
  for (const key of Object.keys(params).sort()) {
    const value = params[key];
    if (typeof value !== "number" && typeof value !== "boolean" && typeof value !== "string") {
      throw new TypeError(`params['${key}'] 必须是有限数/布尔/字符串，得到 ${JSON.stringify(value)}`);
    }
    parts.push(`${jsonString(key)}:${jsonScalar(value, floatKeys.has(key), key)}`);
  }
  return `{${parts.join(",")}}`;
}

/**
 * 参数指纹：`server/frames.py::params_hash` 的 TS 复刻（前 16 个十六进制字符）。
 * 用于 L3 静态帧文件名，以及导出文件名。
 */
export function paramsHash(
  chapter: string,
  viz: string,
  params: ParamValues,
  opts: ParamsHashOptions = {},
): string {
  return sha256Hex(utf8Encode(`${chapter}/${viz}/${canonicalParamsJson(params, opts)}`)).slice(0, 16);
}

/**
 * L3 静态帧路径（**相对站点根**）：`data/frames/<chapter>/<viz>/<hash>_t<t>.nlf`。
 *
 * 与 `server/frames.py::static_frame_path(root, …)` 去掉 `root` 之后的部分逐字节一致
 * （测试里用 `node:crypto` 复算 Python 公式来守这条）。
 */
export function staticFrameRelPath(
  chapter: string,
  viz: string,
  params: ParamValues,
  t = 0,
  opts: ParamsHashOptions = {},
): string {
  return `data/frames/${chapter}/${viz}/${paramsHash(chapter, viz, params, opts)}_t${formatPyG(t, 6)}.nlf`;
}

/**
 * L3 候选路径。
 *
 * 调用方给了 `paramTypes`（manifest）时只有一个候选；没给时，整值参数可能是 `int` 也可能是
 * `float`（`{"zeta": 1}` vs `{"zeta": 1.0}` 是两个不同的哈希），于是给出两种写法各算一个，
 * 由 `ChapterFrameSource` 依次尝试。正常情况最多 2 个候选、1 次命中。
 */
export function staticFrameCandidates(
  chapter: string,
  viz: string,
  params: ParamValues,
  t = 0,
  opts: ParamsHashOptions = {},
): string[] {
  const keys = Object.keys(params);
  const hasIntegral = keys.some((k) => typeof params[k] === "number" && Number.isInteger(params[k] as number));
  const variants: ParamsHashOptions[] = [{ floatKeys: opts.floatKeys ?? [] }];
  if (opts.floatKeys === undefined && hasIntegral) {
    variants.push({ floatKeys: keys.filter((k) => typeof params[k] === "number") });
  }
  const out: string[] = [];
  for (const variant of variants) {
    const rel = staticFrameRelPath(chapter, viz, params, t, variant);
    if (!out.includes(rel)) out.push(rel);
  }
  return out;
}

// --------------------------------------------------------------------------- //
// SHA-256（自己写：浏览器里 crypto.subtle 只在安全上下文可用，http 内网页面没有它）
// --------------------------------------------------------------------------- //

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/** SHA-256 → 小写十六进制。纯 JS（无 `crypto.subtle`、无第三方），浏览器与 Node 行为一致。 */
export function sha256Hex(data: Uint8Array): string {
  const h = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const len = data.length;
  // 补位：0x80 + 若干 0 + 64 bit 长度，使总长为 64 的倍数。
  const padded = new Uint8Array((((len + 8) >> 6) + 1) << 6);
  padded.set(data);
  padded[len] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, Math.floor(len / 0x20000000), false);
  view.setUint32(padded.length - 4, (len << 3) >>> 0, false);

  const w = new Uint32Array(64);
  for (let offset = 0; offset < padded.length; offset += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(offset + i * 4, false);
    for (let i = 16; i < 64; i++) {
      const x = w[i - 15];
      const y = w[i - 2];
      const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
      const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let a = h[0];
    let b = h[1];
    let c = h[2];
    let d = h[3];
    let e = h[4];
    let f = h[5];
    let g = h[6];
    let hh = h[7];
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const t1 = (hh + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) >>> 0;
      hh = g;
      g = f;
      f = e;
      e = (d + t1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) >>> 0;
    }
    h[0] = (h[0] + a) >>> 0;
    h[1] = (h[1] + b) >>> 0;
    h[2] = (h[2] + c) >>> 0;
    h[3] = (h[3] + d) >>> 0;
    h[4] = (h[4] + e) >>> 0;
    h[5] = (h[5] + f) >>> 0;
    h[6] = (h[6] + g) >>> 0;
    h[7] = (h[7] + hh) >>> 0;
  }
  let out = "";
  for (let i = 0; i < 8; i++) out += h[i].toString(16).padStart(8, "0");
  return out;
}

// --------------------------------------------------------------------------- //
// params 传输编码（PROTOCOLS §3）
// --------------------------------------------------------------------------- //

/** `params` 的线上形式：`base64url(JSON)`，键排序、无填充。 */
export function encodeParams(params: ParamValues, opts: ParamsHashOptions = {}): string {
  return base64UrlEncode(utf8Encode(canonicalParamsJson(params, opts)));
}

/**
 * `params` 的反解；坏 base64 / 坏 JSON / 值类型不对 → `null`（**不抛**，便于回放脏链接）。
 * 只接受 base64url 形式（`params_json` 是调试用的旁路，不在前端实现里）。
 */
export function decodeParams(text: unknown): ParamValues | null {
  if (typeof text !== "string" || text === "") return null;
  const bytes = base64UrlDecode(text);
  if (bytes === null) return null;
  const json = utf8Decode(bytes);
  if (json === null) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
  const out: ParamValues = {};
  for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
    if (typeof value === "string" || typeof value === "boolean") out[key] = value;
    else if (typeof value === "number" && Number.isFinite(value)) out[key] = value;
    else return null;
  }
  return out;
}

// --------------------------------------------------------------------------- //
// URL 拼装
// --------------------------------------------------------------------------- //

function trimBase(baseUrl: string | undefined): string {
  return (baseUrl ?? "").trim().replace(/\/+$/, "");
}

/** `ws://<host>/ws/chapter/<chapter_id>`；`baseUrl` 为 http(s) 时自动换协议。 */
export function wsChapterUrl(baseUrl: string | undefined, chapter: string): string {
  const path = `/ws/chapter/${encodeURIComponent(chapter)}`;
  const base = trimBase(baseUrl);
  if (base === "") return path; // 相对 URL：浏览器按文档基址解析
  const match = /^(https?|wss?):\/\/(.*)$/.exec(base);
  if (match === null) return `${base}${path}`;
  const scheme = match[1] === "https" ? "wss" : match[1] === "http" ? "ws" : match[1];
  return `${scheme}://${match[2]}${path}`;
}

/**
 * L2 单帧 URL：`/api/chapter/<id>/frame?viz=..&params=<base64url json>&t=<精确值>`。
 *
 * ⚠ `&t=` 是**本实现补上的**：PROTOCOLS §3 的表里 `/frame` 没有 `t` 参数，但 `time.mode`
 * 为 `scrub`/`stream` 的 viz 在降级到 L2 之后必须能取指定时刻的帧，否则时间轴就是死的。
 * 时刻**不能**用 `%.6g`：例如时间上限 6.2831853 会被写成 6.28319，反而超出服务端上限
 * （3.2 章真踩过，表现为拖到末尾突然「取不到帧」）。这里用 JS 的最短往返写法，
 * 服务端按 float 解析；L3 文件名仍由 `formatPyG(..., 6)` 单独负责。
 */
export function restFrameUrl(
  baseUrl: string | undefined,
  chapter: string,
  viz: string,
  params: ParamValues,
  t = 0,
  opts: ParamsHashOptions = {},
): string {
  return (
    `${trimBase(baseUrl)}/api/chapter/${encodeURIComponent(chapter)}/frame` +
    `?viz=${encodeURIComponent(viz)}&params=${encodeParams(params, opts)}&t=${encodeURIComponent(String(t))}`
  );
}

/**
 * L2 批量帧 URL：`/api/chapter/<id>/frames?...&n=<n>`。
 *
 * 尚未被 `ChapterFrameSource` 使用：§3 说它回 `multipart/mixed` **或**长度前缀拼接，
 * 而服务端还没定下哪一种（`server/` 里目前没有 HTTP 层）。等格式定下来，L2 的 `startStream`
 * 可以改成一次取 n 帧。先把 URL 拼装和测试留在库里，免得又有人各写一份。
 */
export function restFramesUrl(
  baseUrl: string | undefined,
  chapter: string,
  viz: string,
  params: ParamValues,
  n = 60,
  t = 0,
): string {
  const count = Math.max(1, Math.round(Number.isFinite(n) ? n : 1));
  return (
    `${trimBase(baseUrl)}/api/chapter/${encodeURIComponent(chapter)}/frames` +
    `?viz=${encodeURIComponent(viz)}&params=${encodeParams(params)}&t=${encodeURIComponent(String(t))}&n=${count}`
  );
}

// --------------------------------------------------------------------------- //
// 内部小工具
// --------------------------------------------------------------------------- //

function errText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

const TIMERS = globalThis as unknown as {
  setTimeout: (cb: () => void, ms: number) => unknown;
  clearTimeout: (handle: unknown) => void;
};

/** WebSocket 的最小面（只用到 on* 事件属性与 send/close）。 */
interface WsLike {
  binaryType?: string;
  onopen?: ((ev: unknown) => void) | null;
  onmessage?: ((ev: unknown) => void) | null;
  onerror?: ((ev: unknown) => void) | null;
  onclose?: ((ev: unknown) => void) | null;
  send(data: string): void;
  close(code?: number, reason?: string): void;
}

function closeEventText(ev: unknown): string {
  if (typeof ev !== "object" || ev === null) return "";
  const e = ev as { code?: unknown; reason?: unknown };
  const parts: string[] = [];
  if (typeof e.code === "number" && e.code !== 0) parts.push(`code=${e.code}`);
  if (typeof e.reason === "string" && e.reason !== "") parts.push(e.reason);
  return parts.join(" ");
}

// --------------------------------------------------------------------------- //
// ChapterFrameSource
// --------------------------------------------------------------------------- //

/**
 * 一个可视化（chapter + viz）的帧来源，负责 L1→L2→L3 降级与请求去重。
 *
 * 典型用法：
 *
 * ```ts
 * const src = new ChapterFrameSource({
 *   chapter: "0.0", viz: "heat_ftcs", params: defaultParams(vizSpec), staticDir: "/data",
 *   paramTypes: Object.fromEntries(vizSpec.params.map(p => [p.name, p.type])),
 *   time: vizSpec.time, fps: vizSpec.time.fps ?? 30,
 *   deps: { readStatic: (p) => fetch(p).then(r => (r.ok ? r.arrayBuffer() : null)) },
 * });
 * src.on("frame", (frame, t) => render(frame, t));
 * src.on("status", (s) => showBanner(s.message, s.degraded));
 * await src.connect();
 * ```
 *
 * 事件语义：
 * - `frame`：一帧可用（`t` 取 `frame.meta.t`，缺失时回落到请求的 t）；
 * - `status`：数据源状态变化（每次降级都有事件 + 可读 message）；**不会**在订阅时重放，
 *   晚订阅的 UI 请读 `source.status`；
 * - `error`：单次失败（服务端 error 消息、坏帧、HTTP 失败…）。**降级本身不算错误**：
 *   它是设计好的路径，只走 `status`。
 */
export class ChapterFrameSource {
  readonly #chapter: string;
  readonly #viz: string;
  readonly #baseUrl: string;
  readonly #staticDir: string;
  readonly #deps: FrameSourceDeps;
  readonly #fps: number;
  readonly #timeoutMs: number;
  readonly #mode: TimeMode;
  readonly #dt: number;
  readonly #tMax: number;
  readonly #frames: number | undefined;
  readonly #paramTypes: Record<string, ParamType> | undefined;
  readonly #maxRestFailures: number;
  readonly #now: () => number;

  #params: ParamValues;
  #floatKeys: readonly string[];

  #status: SourceStatus = { level: "none", degraded: true, message: "尚未连接", attempts: 0 };
  #frameCbs: Array<FrameSourceEvents["frame"]> = [];
  #statusCbs: Array<FrameSourceEvents["status"]> = [];
  #errorCbs: Array<FrameSourceEvents["error"]> = [];

  #ws: WsLike | null = null;
  #wsOpen = false;
  #connected = false;
  #disposed = false;

  /** 当前（预测或已确认）时刻。 */
  #t = 0;
  #generation = 0;
  #lastRequestedT = Number.NaN;
  #lastDeliveredT = Number.NaN;
  #pendingT: number | null = null;
  #inFlight = false;
  #inflightGen: number | null = null;

  #dropped = 0;
  #received = 0;

  #restFailures = 0;
  #nextAttemptAt = 0;
  #lastRestAt = Number.NEGATIVE_INFINITY;
  #restFailReason = "未尝试";
  #wsFailReason = "未尝试";
  #staticFailReason = "未尝试";

  #streaming = false;
  #lastTickAt = 0;
  #tickTimer: unknown = null;
  #inflightTimer: unknown = null;
  #retryTimer: unknown = null;
  #heartbeatTimer: unknown = null;

  constructor(opts: FrameSourceOptions) {
    this.#chapter = opts.chapter;
    this.#viz = opts.viz;
    this.#baseUrl = opts.baseUrl ?? "";
    this.#staticDir = (opts.staticDir ?? "").replace(/\/+$/, "");
    this.#deps = opts.deps ?? {};
    this.#fps = Number.isFinite(opts.fps) && (opts.fps as number) > 0 ? (opts.fps as number) : 30;
    this.#timeoutMs = Number.isFinite(opts.timeoutMs) && (opts.timeoutMs as number) > 0 ? (opts.timeoutMs as number) : 5000;
    this.#maxRestFailures = Number.isFinite(opts.maxRestFailures) && (opts.maxRestFailures as number) > 0
      ? Math.trunc(opts.maxRestFailures as number)
      : 3;
    this.#paramTypes = opts.paramTypes;
    this.#params = { ...(opts.params ?? {}) };
    this.#floatKeys = this.#deriveFloatKeys(this.#params);
    this.#mode = opts.time?.mode ?? "stream";
    this.#dt = Number.isFinite(opts.time?.dt) && (opts.time?.dt as number) > 0 ? (opts.time?.dt as number) : 1 / this.#fps;
    const declaredTMax = opts.time?.tMax;
    this.#tMax =
      Number.isFinite(declaredTMax) && (declaredTMax as number) > 0
        ? (declaredTMax as number)
        : this.#mode === "none"
          ? 0
          : Number.POSITIVE_INFINITY;
    this.#frames = opts.time?.frames;
    this.#now = this.#deps.now ?? (() => Date.now());
    this.#t = 0;
  }

  // ------------------------------------------------------------------ 只读状态

  /** 当前状态快照（每次都是副本）。 */
  get status(): SourceStatus {
    return { ...this.#status };
  }

  /** 当前数据源层级。 */
  get level(): SourceLevel {
    return this.#status.level;
  }

  /** 当前时刻（L1 下由 `frame.meta.t` 对齐）。 */
  get t(): number {
    return this.#t;
  }

  /** 因"来不及/被新请求取代/参数已变"而未渲染的帧数（可读指标，UI 可以直接显示）。 */
  get droppedFrames(): number {
    return this.#dropped;
  }

  /** 已成功交付的帧数。 */
  get receivedFrames(): number {
    return this.#received;
  }

  /** 当前参数表（副本）。 */
  get params(): ParamValues {
    return { ...this.#params };
  }

  /** 是否已连过（`connect()` 幂等）。 */
  get connected(): boolean {
    return this.#connected;
  }

  // ------------------------------------------------------------------ 事件

  on<E extends keyof FrameSourceEvents>(ev: E, cb: FrameSourceEvents[E]): void {
    if (ev === "frame") this.#frameCbs.push(cb as FrameSourceEvents["frame"]);
    else if (ev === "status") this.#statusCbs.push(cb as FrameSourceEvents["status"]);
    else this.#errorCbs.push(cb as FrameSourceEvents["error"]);
  }

  /** 注销回调（UI 面板卸载时用；`dispose()` 会一次性清空）。 */
  off<E extends keyof FrameSourceEvents>(ev: E, cb: FrameSourceEvents[E]): void {
    const drop = <T>(list: T[], target: T): void => {
      const i = list.indexOf(target);
      if (i >= 0) list.splice(i, 1);
    };
    if (ev === "frame") drop(this.#frameCbs, cb as FrameSourceEvents["frame"]);
    else if (ev === "status") drop(this.#statusCbs, cb as FrameSourceEvents["status"]);
    else drop(this.#errorCbs, cb as FrameSourceEvents["error"]);
  }

  // ------------------------------------------------------------------ 连接与降级

  /** 建立数据源：WS → REST → static → none。幂等；失败不抛（状态里说明原因）。 */
  async connect(): Promise<void> {
    if (this.#disposed || this.#connected) return;
    this.#connected = true;

    this.#setStatus("none", "正在尝试 L1 WebSocket 实时流…", 0);
    if (await this.#tryConnectWs()) return;

    this.#setStatus("none", `L1 WebSocket 不可用（${this.#wsFailReason}）：改用 L2 REST 单帧…`, 1);
    if (await this.#probeRest()) return;

    this.#setStatus("none", `L2 REST 不可用（${this.#restFailReason}）：改用 L3 静态预计算帧…`, 2);
    if (await this.#probeStatic()) return;

    this.#setStatus(
      "none",
      `三层数据源都不可用（L1：${this.#wsFailReason}；L2：${this.#restFailReason}；L3：${this.#staticFailReason}）` +
        "——画面将保持空白，请检查后端是否已启动",
      3,
    );
  }

  #setStatus(level: SourceLevel, message: string, attempts: number): void {
    const next: SourceStatus = { level, degraded: level !== "ws", message, attempts };
    const same =
      this.#status.level === next.level &&
      this.#status.message === next.message &&
      this.#status.attempts === next.attempts;
    this.#status = next;
    if (same) return;
    for (const cb of [...this.#statusCbs]) {
      try {
        cb({ ...next });
      } catch {
        /* 订阅者自己的异常不该打断数据源 */
      }
    }
  }

  async #tryConnectWs(): Promise<boolean> {
    const ctor = this.#deps.WebSocket ?? (globalThis as unknown as { WebSocket?: typeof WebSocket }).WebSocket;
    if (typeof ctor !== "function") {
      this.#wsFailReason = "当前运行环境没有 WebSocket";
      return false;
    }
    const url = wsChapterUrl(this.#baseUrl, this.#chapter);
    let ws: WsLike;
    try {
      ws = new (ctor as unknown as new (u: string) => WsLike)(url);
    } catch (err) {
      this.#wsFailReason = `构造失败：${errText(err)}`;
      return false;
    }

    let opened = false;
    let failReason = "";
    const outcome = await new Promise<boolean>((resolve) => {
      let settled = false;
      const timeout = this.#setTimer(() => {
        if (settled) return;
        settled = true;
        failReason = `连接超时（${this.#timeoutMs} ms）`;
        try {
          ws.close(1000, "timeout");
        } catch {
          /* 关不掉也没关系，下面已经当它失败了 */
        }
        resolve(false);
      }, this.#timeoutMs);

      const finish = (ok: boolean): void => {
        if (settled) return;
        settled = true;
        this.#clearTimer(timeout);
        resolve(ok);
      };

      ws.onopen = () => {
        opened = true;
        finish(true);
      };
      ws.onerror = (ev) => {
        const detail = closeEventText(ev);
        failReason = detail === "" ? "连接被拒绝或网络不可达" : `连接失败（${detail}）`;
        finish(false);
      };
      ws.onclose = (ev) => {
        if (!opened) {
          const detail = closeEventText(ev);
          failReason = detail === "" ? "连接被关闭" : `连接被关闭（${detail}）`;
          finish(false);
          return;
        }
        this.#onWsClosed(ev);
      };
      ws.onmessage = (ev) => {
        void this.#onWsMessage(ev);
      };
    });

    if (!outcome) {
      this.#wsFailReason = failReason === "" ? "未知原因" : failReason;
      try {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        ws.close(1000, "degrade");
      } catch {
        /* 忽略 */
      }
      return false;
    }

    this.#ws = ws;
    this.#wsOpen = true;
    try {
      ws.binaryType = "arraybuffer"; // 浏览器：让它给 ArrayBuffer 而不是 Blob
    } catch {
      /* 某些实现是只读的，无所谓：下面按需处理 Blob */
    }
    const time: Record<string, unknown> = { mode: this.#mode, fps: this.#fps, dt: this.#dt };
    if (this.#frames !== undefined) time["frames"] = this.#frames;
    this.#send({ type: "subscribe", chapter: this.#chapter, viz: this.#viz, params: this.#params, time });
    this.#setStatus("ws", `L1 WebSocket 已连接（${url}）：实时计算，参数可连续拖动`, 1);
    this.#startHeartbeat();
    return true;
  }

  /**
   * WebSocket 空闲心跳。服务端把「30 s 没有收到任何控制消息」当作超时并关闭连接；
   * 页面停留、只悬停读图时，客户端本来不会再发消息。每 10 s 发一个 `ping`
   * （服务端回 `pong`）把连接保持住，避免读到一半突然掉回 REST。
   */
  #startHeartbeat(): void {
    this.#stopHeartbeat();
    // 固定 10 s：后端空闲阈值是 30 s，这个频率留了 3 倍余量；与单帧请求超时无关。
    const interval = 10_000;
    const beat = (): void => {
      this.#heartbeatTimer = null;
      if (this.#disposed || !this.#wsOpen) return;
      this.#send({ type: "ping", t: this.#now() });
      this.#heartbeatTimer = this.#setTimer(beat, interval);
    };
    this.#heartbeatTimer = this.#setTimer(beat, interval);
  }

  #stopHeartbeat(): void {
    if (this.#heartbeatTimer === null) return;
    this.#clearTimer(this.#heartbeatTimer);
    this.#heartbeatTimer = null;
  }

  #onWsClosed(ev: unknown): void {
    this.#wsOpen = false;
    this.#ws = null;
    this.#stopHeartbeat();
    if (this.#disposed) return;
    const detail = closeEventText(ev);
    const reason = detail === "" ? "连接被关闭" : `连接被关闭（${detail}）`;
    this.#wsFailReason = reason;
    void this.#degradeFromWs(reason);
  }

  /** L1 掉线：先试 L2，再试 L3（与 `connect()` 的后半程同一条路）。 */
  async #degradeFromWs(reason: string): Promise<void> {
    if (this.#disposed) return;
    this.#inFlight = false;
    this.#clearInflightTimer();
    this.#setStatus("none", `L1 WebSocket 断开（${reason}）：改用 L2 REST 单帧…`, 1);
    if (await this.#probeRest()) return;
    this.#setStatus("none", `L2 REST 不可用（${this.#restFailReason}）：改用 L3 静态预计算帧…`, 2);
    if (await this.#probeStatic()) return;
    this.#setStatus(
      "none",
      `三层数据源都不可用（L1：${reason}；L2：${this.#restFailReason}；L3：${this.#staticFailReason}）`,
      3,
    );
  }

  async #probeRest(): Promise<boolean> {
    const url = restFrameUrl(this.#baseUrl, this.#chapter, this.#viz, this.#params, this.#t);
    try {
      const frame = await this.#fetchRestFrame(url);
      this.#restFailures = 0;
      this.#nextAttemptAt = 0;
      this.#lastRestAt = this.#now();
      this.#setStatus("rest", `L2 REST 单帧可用（${url}）：拖动按需取帧，节流 ${REST_THROTTLE_MS} ms`, 2);
      this.#deliver(frame, this.#t, this.#generation);
      return true;
    } catch (err) {
      this.#restFailReason = errText(err);
      return false;
    }
  }

  async #probeStatic(): Promise<boolean> {
    try {
      const hit = await this.#readStaticFrame(this.#t);
      this.#nextAttemptAt = 0;
      this.#setStatus(
        "static",
        `L3 静态预计算帧（${hit.path}）：只能看预计算过的参数与时刻点，参数会被吸附到最近的已算点`,
        3,
      );
      this.#deliver(hit.frame, this.#t, this.#generation);
      return true;
    } catch (err) {
      this.#staticFailReason = errText(err);
      return false;
    }
  }

  /** REST 连续失败到阈值：降级到 L3（这是"降级"不是"报错"，只走 status）。 */
  async #degradeToStatic(reason: string): Promise<void> {
    if (this.#disposed) return;
    this.#setStatus("none", `${reason}：改用 L3 静态预计算帧…`, 2);
    if (await this.#probeStatic()) return;
    this.#setStatus(
      "none",
      `L3 静态帧也不可用（${this.#staticFailReason}）：三层数据源都失败，画面将保持空白`,
      3,
    );
  }

  // ------------------------------------------------------------------ 取帧

  /**
   * 请求某时刻的帧。
   *
   * - 同一个 `t` 只请求一次（`setParams` 会重置这个去重缓存）；
   * - 在途时最多缓存 **1 个**最新请求（旧的那个被丢掉，计入 `droppedFrames`）；
   * - L2 拖动节流 `REST_THROTTLE_MS`（PROTOCOLS §4）：60 ms 内的新请求不立刻发，
   *   只更新"最新想要的那个 t"，到窗口末尾由定时器补发（拖动时后端最多承受 ~16 req/s）；
   * - `level === "none"` 时静默忽略（`status.message` 已经在喊了，错误事件不刷屏）。
   */
  request(t: number): void {
    if (this.#disposed) return;
    if (typeof t !== "number" || !Number.isFinite(t)) {
      this.#emitError(new Error(`request(t) 需要有限数，得到 ${String(t)}`));
      return;
    }
    const target = this.#clampT(t);
    if (this.#inFlight) {
      if (this.#pendingT !== null && this.#pendingT !== target) this.#dropped++;
      this.#pendingT = target;
      return;
    }
    if (target === this.#lastRequestedT) return; // 同 t 去重
    if (this.#status.level === "none") return;

    if (this.#status.level === "rest") {
      const throttledUntil = this.#lastRestAt + REST_THROTTLE_MS;
      if (this.#now() < throttledUntil) {
        if (this.#pendingT !== null && this.#pendingT !== target) this.#dropped++;
        this.#pendingT = target;
        if (throttledUntil > this.#nextAttemptAt) this.#nextAttemptAt = throttledUntil;
        this.#scheduleRetry();
        return;
      }
    }
    this.#issue(target);
  }

  #clampT(t: number): number {
    if (t < 0) return 0;
    if (Number.isFinite(this.#tMax) && t > this.#tMax) return this.#tMax;
    return t;
  }

  #issue(t: number): void {
    this.#lastRequestedT = t;
    const gen = this.#generation;

    // 退避窗口内：不真发请求，只记"最新想要的那个 t"，等定时器到点再发。
    if (this.#status.level !== "ws" && this.#now() < this.#nextAttemptAt) {
      this.#pendingT = t;
      this.#scheduleRetry();
      return;
    }

    this.#inFlight = true;
    this.#inflightGen = gen;
    switch (this.#status.level) {
      case "ws": {
        if (!this.#send({ type: "seek", t })) this.#inFlight = false;
        else this.#armInflightTimeout();
        return;
      }
      case "rest": {
        this.#armInflightTimeout();
        void this.#restAttempt(t, gen);
        return;
      }
      case "static": {
        this.#armInflightTimeout();
        void this.#staticAttempt(t, gen);
        return;
      }
      default:
        this.#inFlight = false;
        return;
    }
  }

  #flushPending(): void {
    if (this.#disposed) return;
    const pending = this.#pendingT;
    this.#pendingT = null;
    if (pending === null) return;
    if (pending === this.#lastDeliveredT) return; // 这个 t 刚拿到，不必再取
    this.#issue(pending);
  }

  #deliver(frame: Frame, fallbackT: number, gen: number): void {
    if (gen !== this.#generation) {
      // 参数已经换了：这一帧是按旧参数算的，画出来就是撒谎。
      this.#dropped++;
      return;
    }
    if (this.#disposed) return;
    this.#clearInflight();
    this.#inFlight = false;
    this.#inflightGen = null;

    const metaT = frame.meta["t"];
    const t = typeof metaT === "number" && Number.isFinite(metaT) ? metaT : fallbackT;
    if (Number.isFinite(t)) {
      this.#t = t;
      this.#lastDeliveredT = t;
    }
    this.#received++;
    const emitted = [...this.#frameCbs];
    for (const cb of emitted) {
      try {
        cb(frame, this.#t);
      } catch (err) {
        this.#emitError(err instanceof Error ? err : new Error(errText(err)));
      }
    }
    this.#flushPending();
  }

  async #fetchRestFrame(url: string): Promise<Frame> {
    const fetchFn = this.#deps.fetch ?? (globalThis as unknown as { fetch?: typeof fetch }).fetch;
    if (typeof fetchFn !== "function") throw new Error("当前运行环境没有 fetch");
    const controller =
      typeof AbortController === "function" ? new AbortController() : null;
    const timer = controller === null
      ? null
      : this.#setTimer(() => {
          try {
            controller.abort();
          } catch {
            /* 忽略 */
          }
        }, this.#timeoutMs);
    try {
      const res = await fetchFn(url, controller === null ? undefined : { signal: controller.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buf = await res.arrayBuffer();
      return await decodeNlfAsync(new Uint8Array(buf));
    } finally {
      if (timer !== null) this.#clearTimer(timer);
    }
  }

  async #restAttempt(t: number, gen: number): Promise<void> {
    this.#lastRestAt = this.#now();
    const url = restFrameUrl(this.#baseUrl, this.#chapter, this.#viz, this.#params, t);
    try {
      const frame = await this.#fetchRestFrame(url);
      this.#restFailures = 0;
      this.#nextAttemptAt = 0;
      this.#deliver(frame, t, gen);
    } catch (err) {
      if (gen !== this.#inflightGen) return; // 已被新参数或新请求取代
      this.#clearInflight();
      this.#inFlight = false;
      this.#inflightGen = null;
      this.#lastRequestedT = Number.NaN; // 失败的这个 t 之后还可以再要（退避窗口会限制频率）
      this.#restFailures++;
      const backoff = Math.min(REST_BACKOFF_MAX_MS, REST_BACKOFF_BASE_MS * 2 ** (this.#restFailures - 1));
      this.#nextAttemptAt = this.#now() + backoff;
      const message = `t=${formatPyG(t, 6)} 取帧失败：${errText(err)}`;
      this.#emitError(new Error(`L2 REST ${message}`));
      if (this.#restFailures >= this.#maxRestFailures) {
        await this.#degradeToStatic(`L2 REST 连续 ${this.#restFailures} 次失败（${errText(err)}）`);
        return;
      }
      this.#setStatus(
        this.#status.level,
        `L2 REST ${message}；已退避 ${backoff} ms（第 ${this.#restFailures}/${this.#maxRestFailures} 次，超过则降级到 L3）`,
        2,
      );
      if (this.#pendingT !== null) this.#scheduleRetry();
    }
  }

  async #readStaticFrame(t: number): Promise<{ frame: Frame; path: string }> {
    const reader = this.#deps.readStatic;
    if (typeof reader !== "function") throw new Error("调用方没有提供 readStatic（静态目录不可达）");
    const rels = staticFrameCandidates(this.#chapter, this.#viz, this.#params, t, {
      floatKeys: this.#paramTypes === undefined ? undefined : this.#floatKeys,
    });
    const reasons: string[] = [];
    let sawMissing = false;
    for (const rel of rels) {
      const path = this.#staticDir === "" ? rel : `${this.#staticDir}/${rel}`;
      let buf: ArrayBuffer | null;
      try {
        buf = await reader(path);
      } catch (err) {
        reasons.push(`${path}：读取失败（${errText(err)}）`);
        continue;
      }
      if (buf === null || buf === undefined) {
        sawMissing = true;
        continue;
      }
      try {
        return { frame: await decodeNlfAsync(new Uint8Array(buf)), path };
      } catch (err) {
        reasons.push(`${path}：解码失败（${errText(err)}）`);
      }
    }
    if (reasons.length === 0 && sawMissing) {
      throw new Error(`没有 t=${formatPyG(t, 6)} 的预计算帧（试过 ${rels.length} 个文件名，例如 ${rels[0] ?? "?"}）`);
    }
    throw new Error(reasons.length > 0 ? reasons.join("；") : "没有可用的静态帧候选路径");
  }

  async #staticAttempt(t: number, gen: number): Promise<void> {
    try {
      const hit = await this.#readStaticFrame(t);
      this.#nextAttemptAt = 0;
      this.#deliver(hit.frame, t, gen);
    } catch (err) {
      if (gen !== this.#inflightGen) return;
      this.#clearInflight();
      this.#inFlight = false;
      this.#inflightGen = null;
      this.#lastRequestedT = Number.NaN;
      this.#nextAttemptAt = this.#now() + STATIC_RETRY_MS;
      this.#emitError(new Error(`L3 静态帧：${errText(err)}`));
      if (this.#status.level === "static") {
        this.#setStatus(
          "static",
          `L3 静态帧缺少 t=${formatPyG(t, 6)}（${errText(err)}）：请换用预计算过的时刻`,
          this.#status.attempts,
        );
      }
      if (this.#pendingT !== null) this.#scheduleRetry();
    }
  }

  #scheduleRetry(): void {
    if (this.#disposed || this.#retryTimer !== null) return;
    const delay = Math.max(1, this.#nextAttemptAt - this.#now());
    this.#retryTimer = this.#setTimer(() => {
      this.#retryTimer = null;
      if (this.#disposed) return;
      const pending = this.#pendingT;
      this.#pendingT = null;
      if (pending === null || this.#inFlight) {
        if (pending !== null) this.#pendingT = pending;
        return;
      }
      this.#issue(pending);
    }, delay);
  }

  // ------------------------------------------------------------------ WebSocket 消息

  async #onWsMessage(ev: unknown): Promise<void> {
    if (this.#disposed) return;
    const data = (ev as { data?: unknown } | null)?.data;
    if (typeof data === "string") {
      this.#onWsControl(data);
      return;
    }
    const bytes = await this.#toBytes(data);
    if (bytes === null) {
      this.#emitError(new Error(`WebSocket 收到无法识别的载荷：${Object.prototype.toString.call(data)}`));
      return;
    }
    let frame: Frame;
    try {
      frame = await decodeNlfAsync(bytes);
    } catch (err) {
      this.#emitError(new Error(`WebSocket 帧解码失败（${bytes.byteLength} 字节）：${errText(err)}`));
      return;
    }
    this.#deliver(frame, this.#lastRequestedT, this.#generation);
  }

  #onWsControl(text: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (err) {
      this.#emitError(new Error(`WebSocket 控制消息不是合法 JSON：${errText(err)}`));
      return;
    }
    if (typeof parsed !== "object" || parsed === null) return;
    const msg = parsed as Record<string, unknown>;
    const type = msg["type"];
    if (type === "error") {
      const code = msg["code"];
      const detail = typeof code === "string" || typeof code === "number" ? `（code=${String(code)}）` : "";
      this.#emitError(new Error(`服务端错误${detail}：${String(msg["message"] ?? "未给出原因")}`));
    } else if (type === "done") {
      this.#setStatus(
        "ws",
        `L1 流已结束（frames_sent=${String(msg["frames_sent"] ?? "?")}）：停留在最后一帧`,
        this.#status.attempts,
      );
    }
    // ready / frames / pong：正常控制消息，数据本身走二进制帧，这里无需处理。
  }

  async #toBytes(data: unknown): Promise<Uint8Array | null> {
    if (data instanceof Uint8Array) return data;
    if (data instanceof ArrayBuffer) return new Uint8Array(data);
    const BlobCtor = (globalThis as unknown as { Blob?: typeof Blob }).Blob;
    if (typeof BlobCtor === "function" && data instanceof BlobCtor) {
      return new Uint8Array(await data.arrayBuffer());
    }
    return null;
  }

  #send(payload: Record<string, unknown>): boolean {
    const ws = this.#ws;
    if (ws === null || !this.#wsOpen) return false;
    try {
      ws.send(JSON.stringify(payload));
      return true;
    } catch (err) {
      this.#emitError(new Error(`WebSocket 发送失败（${String(payload["type"])}）：${errText(err)}`));
      return false;
    }
  }

  // ------------------------------------------------------------------ 参数

  /**
   * 换参数：作废旧的在途请求（旧帧即使到达也会被丢弃并计入 `droppedFrames`），
   * 重置"同 t 去重"缓存，并立刻按新参数重新请求当前时刻。
   */
  setParams(params: ParamValues): void {
    if (this.#disposed) return;
    this.#params = { ...params };
    this.#floatKeys = this.#deriveFloatKeys(this.#params);
    this.#generation++;
    this.#lastRequestedT = Number.NaN;
    this.#lastDeliveredT = Number.NaN;
    this.#pendingT = null;
    this.#inFlight = false;
    this.#inflightGen = null;
    this.#clearInflightTimer();
    this.#restFailures = 0;
    this.#nextAttemptAt = 0;

    if (this.#status.level === "ws") {
      this.#send({ type: "params", params: this.#params });
      return;
    }
    if (this.#status.level === "none") return;
    this.#issue(this.#t);
  }

  #deriveFloatKeys(params: ParamValues): readonly string[] {
    const types = this.#paramTypes;
    if (types === undefined) return [];
    return Object.keys(params).filter((k) => types[k] === "float");
  }

  // ------------------------------------------------------------------ 时间轴推进

  /** 按 `fps` 逐帧推进（L1 只发 `play`，时间由服务端帧的 `meta.t` 对齐）。 */
  startStream(): void {
    if (this.#disposed || this.#streaming) return;
    if (this.#mode === "none") return; // 无时间轴的 viz 不该有播放按钮
    this.#streaming = true;
    this.#lastTickAt = this.#now();
    if (this.#status.level === "ws") this.#send({ type: "play", t: this.#t });
    this.#scheduleTick();
  }

  stopStream(): void {
    const wasStreaming = this.#streaming;
    this.#streaming = false;
    if (this.#tickTimer !== null) {
      this.#clearTimer(this.#tickTimer);
      this.#tickTimer = null;
    }
    if (wasStreaming && this.#status.level === "ws") this.#send({ type: "pause" });
  }

  #scheduleTick(): void {
    if (!this.#streaming || this.#disposed || this.#tickTimer !== null) return;
    const interval = Math.max(1, Math.round(1000 / this.#fps));
    this.#tickTimer = this.#setTimer(() => {
      this.#tickTimer = null;
      if (!this.#streaming || this.#disposed) return;
      this.#tick();
      this.#scheduleTick();
    }, interval);
  }

  #tick(): void {
    const interval = Math.max(1, Math.round(1000 / this.#fps));
    const nowMs = this.#now();
    if (nowMs - this.#lastTickAt < interval - 0.5) return; // 定时器早到了：真实时间才算数
    this.#lastTickAt = nowMs;
    if (this.#status.level === "ws") return; // L1：帧由服务端推送
    const next = this.#t + this.#dt;
    if (next > this.#tMax) {
      this.#t = this.#tMax;
      this.stopStream();
      return;
    }
    this.#t = next;
    this.request(this.#t);
  }

  // ------------------------------------------------------------------ 定时器与清理

  #armInflightTimeout(): void {
    this.#clearInflightTimer();
    this.#inflightTimer = this.#setTimer(() => {
      this.#inflightTimer = null;
      if (this.#disposed || !this.#inFlight) return;
      this.#inFlight = false;
      this.#inflightGen = null;
      this.#lastRequestedT = Number.NaN; // 卡住的那一次不算数，之后可以重新要
      this.#dropped++;
      this.#flushPending();
    }, this.#timeoutMs);
  }

  #clearInflightTimer(): void {
    if (this.#inflightTimer !== null) {
      this.#clearTimer(this.#inflightTimer);
      this.#inflightTimer = null;
    }
  }

  #clearInflight(): void {
    this.#clearInflightTimer();
  }

  #setTimer(cb: () => void, ms: number): unknown {
    const fn = this.#deps.setTimer;
    return fn ? fn(cb, ms) : TIMERS.setTimeout(cb, ms);
  }

  #clearTimer(handle: unknown): void {
    if (handle === null || handle === undefined) return;
    const fn = this.#deps.clearTimer;
    if (fn) fn(handle);
    else TIMERS.clearTimeout(handle);
  }

  #emitError(error: Error): void {
    if (this.#disposed) return;
    for (const cb of [...this.#errorCbs]) {
      try {
        cb(error);
      } catch {
        /* 订阅者异常不影响数据源 */
      }
    }
  }

  /** 断开连接、停掉所有定时器、清空回调；此后**不再触发任何回调**。 */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#streaming = false;
    if (this.#tickTimer !== null) {
      this.#clearTimer(this.#tickTimer);
      this.#tickTimer = null;
    }
    this.#clearInflightTimer();
    this.#stopHeartbeat();
    if (this.#retryTimer !== null) {
      this.#clearTimer(this.#retryTimer);
      this.#retryTimer = null;
    }
    const ws = this.#ws;
    this.#ws = null;
    this.#wsOpen = false;
    if (ws !== null) {
      try {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        ws.close(1000, "dispose");
      } catch {
        /* 忽略 */
      }
    }
    this.#frameCbs = [];
    this.#statusCbs = [];
    this.#errorCbs = [];
  }
}
