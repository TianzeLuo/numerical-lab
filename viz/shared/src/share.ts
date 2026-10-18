/**
 * 链接状态（"把当前这一屏发给别人"）与 base64url 字节层原语。
 *
 * 三件事：
 *
 * 1. **base64url 原语**：`docs/PROTOCOLS.md` §3 规定 `params` 走 `base64url(JSON)`，
 *    分享链接也用它。这里自己写（不依赖 `btoa`/`atob`）：那两个函数只认 Latin-1 字符串，
 *    传中文要先做一层转义，而转义写法在 Node 与浏览器上还有历史差异；直接用字节做基更省心。
 * 2. **`encodeState` / `decodeState`**：章节 + viz + 参数 + 时刻 + 是否播放 ⇄ 一段可粘贴文本。
 * 3. **`buildHash` / `decodeHash`**：上面那段文本 ⇄ `location.hash`。
 *
 * 两条硬约束（都是被测试钉住的）：
 *
 * - **`decodeState` / `decodeHash` 绝不抛**：坏 base64 / 坏 JSON / 版本不认识 / 字段类型不对
 *   一律返回 `null`。一个来路不明的分享链接不该让整页崩掉——最坏情况是"回到默认视图"。
 * - **`encodeState` 确定性**：键排序 + 不带 padding 的 base64url，同一份状态永远得到同一串。
 *   这样"同一视图的链接"可以逐字节比较（测试里就是这么做的），也可以安全地做缓存键。
 *
 * 本模块**不碰 DOM**：`decodeHash` 接受的是"hash 文本"，不是 `location` 对象，
 * 于是它在 Node 里也能被测。
 */

/** 单个参数值（与 `VizSpecJson.params[*].default` 同域）。 */
export type ParamValue = number | boolean | string;

/** 参数表：`base64url(JSON)` 编解码、分享链接、帧请求三处共用的形状。 */
export type ParamValues = Record<string, ParamValue>;

/** 分享链接里携带的完整视图状态。 */
export interface ShareState {
  chapter: string;
  viz: string;
  params: ParamValues;
  /** 时刻；无时间轴的 viz 恒为 0。 */
  t: number;
  /** 是否处于播放态（接收方打开时是否自动播放由 UI 决定）。 */
  playing: boolean;
}

/** 当前状态串版本；不兼容改字段时 +1（老链接会被 `decodeState` 判为 `null`）。 */
export const STATE_VERSION = 1;

/** 版本前缀，形如 `v1.`。 */
export const STATE_PREFIX = `v${STATE_VERSION}.`;

// --------------------------------------------------------------------------- //
// base64 / base64url（纯字节层，无 btoa/atob）
// --------------------------------------------------------------------------- //

const B64_STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_URL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/** 字符 → 6 bit 的反查表（非字母表字符为 -1）。 */
function buildLookup(alphabet: string): Int8Array {
  const table = new Int8Array(128).fill(-1);
  for (let i = 0; i < alphabet.length; i++) table[alphabet.charCodeAt(i)] = i;
  return table;
}

const LOOKUP_STD = buildLookup(B64_STD);
const LOOKUP_URL = buildLookup(B64_URL);

/** 通用编码：`urlSafe` 时用 `-_`；`pad` 决定是否补 `=`（base64url 不补，RFC 4648 §5）。 */
function encodeBase64(bytes: Uint8Array, urlSafe: boolean, pad: boolean): string {
  const alphabet = urlSafe ? B64_URL : B64_STD;
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < bytes.length ? bytes[i + 1] : undefined;
    const b2 = i + 2 < bytes.length ? bytes[i + 2] : undefined;
    out += alphabet[b0 >> 2];
    out += alphabet[((b0 & 0x03) << 4) | ((b1 ?? 0) >> 4)];
    if (b1 === undefined) {
      if (pad) out += "==";
      break;
    }
    out += alphabet[((b1 & 0x0f) << 2) | ((b2 ?? 0) >> 6)];
    if (b2 === undefined) {
      if (pad) out += "=";
      break;
    }
    out += alphabet[b2 & 0x3f];
  }
  return out;
}

/**
 * 通用解码；遇到字母表外的字符、长度非法或填充位不为零时返回 `null`（**不抛**）。
 *
 * "填充位为零"这一条是刻意的：`Zh` 与 `Zg` 都能解出同一个字节，若都接受，
 * 同一份状态就会有多种写法，确定性就没了。
 */
function decodeBase64(text: string, urlSafe: boolean): Uint8Array | null {
  const lookup = urlSafe ? LOOKUP_URL : LOOKUP_STD;
  const cleaned = text.replace(/[\t\n\r ]+/g, "");
  if (cleaned === "") return new Uint8Array(0);
  if (cleaned.length % 4 === 1) return null;
  const out = new Uint8Array(((cleaned.length + 3) >> 2) * 3);
  let acc = 0;
  let accBits = 0;
  let n = 0;
  let seenPad = false;
  for (let i = 0; i < cleaned.length; i++) {
    const ch = cleaned[i];
    if (ch === "=") {
      seenPad = true;
      continue;
    }
    if (seenPad) return null; // 填充后不得再有数据字符
    const code = ch.charCodeAt(0);
    const value = code < 128 ? lookup[code] : -1;
    if (value < 0) return null;
    acc = ((acc << 6) | value) & 0xffffff;
    accBits += 6;
    if (accBits >= 8) {
      accBits -= 8;
      out[n++] = (acc >> accBits) & 0xff;
    }
  }
  // 剩余位必须是零（否则不是规范编码）
  if (accBits > 0 && ((acc & ((1 << accBits) - 1)) !== 0)) return null;
  return out.subarray(0, n);
}

/** 标准 base64（带 `=` 填充）。 */
export function base64Encode(bytes: Uint8Array): string {
  return encodeBase64(bytes, false, true);
}

/** 标准 base64 解码；非法输入返回 `null`。 */
export function base64Decode(text: string): Uint8Array | null {
  return decodeBase64(text, false);
}

/** base64url（`-_`、无填充）——PROTOCOLS §3 的 `params` 传输格式。 */
export function base64UrlEncode(bytes: Uint8Array): string {
  return encodeBase64(bytes, true, false);
}

/** base64url 解码；非法输入返回 `null`（同时接受带 `=` 填充的写法）。 */
export function base64UrlDecode(text: string): Uint8Array | null {
  return decodeBase64(text.replace(/=+$/, ""), true);
}

const UTF8_ENCODER = new TextEncoder();
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });

/** UTF-8 编码（纯函数，供 base64url(JSON) 使用）。 */
export function utf8Encode(text: string): Uint8Array {
  return UTF8_ENCODER.encode(text);
}

/** UTF-8 解码；非法字节序列返回 `null`（`fatal: true`，不产出 U+FFFD 假字符串）。 */
export function utf8Decode(bytes: Uint8Array): string | null {
  try {
    return UTF8_DECODER.decode(bytes);
  } catch {
    return null;
  }
}

// --------------------------------------------------------------------------- //
// 状态编解码
// --------------------------------------------------------------------------- //

/** 确定性 JSON：对象键按码点排序（参数名是 ASCII snake_case，与 Python `sort_keys` 同序）。 */
export function stableJson(value: unknown): string {
  if (value === null || typeof value === "number" || typeof value === "boolean" || typeof value === "string") {
    return JSON.stringify(value) ?? "null";
  }
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).sort();
    const parts: string[] = [];
    for (const k of keys) {
      const v = obj[k];
      if (v === undefined) continue;
      parts.push(`${JSON.stringify(k)}:${stableJson(v)}`);
    }
    return `{${parts.join(",")}}`;
  }
  throw new TypeError(`stableJson 不支持的值：${typeof value}`);
}

function isParamValue(v: unknown): v is ParamValue {
  return typeof v === "string" || typeof v === "boolean" || (typeof v === "number" && Number.isFinite(v));
}

/** 把参数表按稳定顺序排好，顺便校验值域（非标量值直接报错，不静默丢参数）。 */
function normalizeParams(params: ParamValues | undefined): ParamValues {
  const out: ParamValues = {};
  const src = params ?? {};
  if (typeof src !== "object" || Array.isArray(src)) {
    throw new TypeError("encodeState: params 必须是对象");
  }
  for (const key of Object.keys(src).sort()) {
    const value = src[key];
    if (!isParamValue(value)) {
      throw new TypeError(`encodeState: params['${key}'] 必须是有限数/布尔/字符串，得到 ${JSON.stringify(value)}`);
    }
    out[key] = value;
  }
  return out;
}

/** 读取一份 `ShareState`；任何不合规都返回 `null`。 */
function readState(raw: unknown): ShareState | null {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
  const obj = raw as Record<string, unknown>;
  const chapter = obj["chapter"];
  const viz = obj["viz"];
  if (typeof chapter !== "string" || chapter === "") return null;
  if (typeof viz !== "string" || viz === "") return null;

  const params: ParamValues = {};
  const rawParams = obj["params"];
  if (rawParams !== undefined) {
    if (typeof rawParams !== "object" || rawParams === null || Array.isArray(rawParams)) return null;
    for (const [key, value] of Object.entries(rawParams as Record<string, unknown>)) {
      if (!isParamValue(value)) return null; // 类型不对 → 整体判废，不"尽量修"
      params[key] = value;
    }
  }

  const rawT = obj["t"];
  let t = 0;
  if (rawT !== undefined) {
    if (typeof rawT !== "number" || !Number.isFinite(rawT)) return null;
    t = rawT;
  }

  const rawPlaying = obj["playing"];
  let playing = false;
  if (rawPlaying !== undefined) {
    if (typeof rawPlaying !== "boolean") return null;
    playing = rawPlaying;
  }
  return { chapter, viz, params, t, playing };
}

/**
 * 编码分享状态：`v1.` + base64url(JSON)。
 *
 * 键排序 + 无填充 base64url ⇒ 确定性；`params` 里出现非标量值会抛 `TypeError`
 * （生产端出错应当立刻被发现，而不是生成一条"看起来能用"的链接）。
 */
export function encodeState(state: ShareState): string {
  if (typeof state !== "object" || state === null) throw new TypeError("encodeState 需要 ShareState 对象");
  if (typeof state.chapter !== "string" || state.chapter === "") throw new TypeError("encodeState: chapter 必须是非空字符串");
  if (typeof state.viz !== "string" || state.viz === "") throw new TypeError("encodeState: viz 必须是非空字符串");
  const t = state.t;
  if (typeof t !== "number" || !Number.isFinite(t)) throw new TypeError("encodeState: t 必须是有限数");
  const payload = {
    chapter: state.chapter,
    viz: state.viz,
    params: normalizeParams(state.params),
    t,
    playing: state.playing === true,
  };
  return STATE_PREFIX + base64UrlEncode(utf8Encode(stableJson(payload)));
}

/**
 * 解码分享状态。**绝不抛**：`null` 表示"这段文本不是本版本的合法状态"。
 *
 * 容错清单（测试逐条钉住）：坏 base64 / 坏 UTF-8 / 坏 JSON / 不是对象 / 版本不是 `v1` /
 * 缺 chapter 或 viz / `params` 值类型不对 / `t` 不是有限数 / `playing` 不是布尔。
 */
export function decodeState(text: unknown): ShareState | null {
  if (typeof text !== "string") return null;
  const body = text.trim();
  const dot = body.indexOf(".");
  if (dot <= 0) return null;
  const prefix = body.slice(0, dot);
  if (`${prefix}.` !== STATE_PREFIX) return null; // 认不出的版本（含 v2、无前缀）一律 null
  const bytes = base64UrlDecode(body.slice(dot + 1));
  if (bytes === null) return null;
  const json = utf8Decode(bytes);
  if (json === null) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return null;
  }
  return readState(parsed);
}

/** 把状态编成 `location.hash` 形式（含 `#`）。 */
export function buildHash(state: ShareState): string {
  return `#${encodeState(state)}`;
}

/**
 * 从 hash / 整条 URL / 裸状态串里解出状态。
 *
 * 接受三种输入（都是为了调用方少写一层判断）：
 * `"#v1.xxx"`、`"http://host/path#v1.xxx"`、`"v1.xxx"`；
 * 空串、`"#"`、`"http://host/path"`（没有 hash）→ `null`。
 */
export function decodeHash(text: unknown): ShareState | null {
  if (typeof text !== "string") return null;
  const body = text.trim();
  if (body === "") return null;
  const i = body.indexOf("#");
  if (i >= 0) {
    const fragment = body.slice(i + 1);
    if (fragment === "") return null;
    return decodeState(fragment);
  }
  return decodeState(body);
}
