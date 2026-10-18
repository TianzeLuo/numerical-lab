/**
 * `.nlf` 帧容器的字节级编解码（`docs/PROTOCOLS.md` §1，参考实现 `server/nlf.py`）。
 *
 * 布局::
 *
 *     offset 0   : b"NLF1"
 *     offset 4   : uint32LE  header_len
 *     offset 8   : header_len 字节 UTF-8 JSON（无 BOM、无尾随空白）
 *     offset 8+H : payload（各数组按 arrays 顺序紧排，little-endian）
 *
 * 三条不可动摇的规则（与 Python 侧一致）：
 *
 * 1. **header 恒为明文**，只有 payload 允许 zlib/gzip 压缩——`decodeNlf` 因此可以同步完成
 *    `codec === "none"` 的解码，压缩帧必须显式走 `decodeNlfAsync`（需要 DecompressionStream）。
 * 2. **解码端做全量自洽校验**：magic / header_len / version / endian / codec / 每个数组的
 *    dtype·order·shape·nbytes·越界 / 数组区间不重叠 / payload 总长等于各数组之和。
 *    任何一条不满足都抛 `NlfError`，绝不返回"看起来能画"的可疑数据。
 * 3. **只依赖 JS 内建**（DataView / TypedArray / TextDecoder / DecompressionStream /
 *    CompressionStream）：`docs/POLICY.md` P7 要求前端运行期零第三方依赖。
 */

import { DTYPE_ITEMSIZE, Frame, NlfError } from "./frame.js";
import type { ArraySpec, Dtype, FrameCodec, FrameKind, TypedArray } from "./frame.js";

export { NlfError } from "./frame.js";

/** 容器魔数：`b"NLF1"`。 */
export const NLF_MAGIC: Uint8Array = Uint8Array.from([0x4e, 0x4c, 0x46, 0x31]);

/** 容器版本；不兼容升级才改。 */
export const NLF_VERSION = 1;

/** header 上界：正常 header 只有几 KB，超过 32 MiB 必是 bug 或恶意构造。 */
export const MAX_HEADER_LEN = 32 * 1024 * 1024;

/** 写端缺省 generator（Python 侧写 python/numpy 版本；前端没有这些，写工具名即可）。 */
const DEFAULT_GENERATOR: Record<string, string> = { tool: "viz/shared/src/nlf.ts", format: "0.1.0" };

/**
 * 宿主字节序。`.nlf` 固定 little-endian，只有在**大端宿主**（s390x 等）上才必须
 * 走 DataView 逐元素路径；小端宿主可以直接把字节当作 TypedArray 使用（快得多）。
 */
const HOST_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

// --------------------------------------------------------------------------- //
// 小工具
// --------------------------------------------------------------------------- //

/** 把任意抛出物转成可读消息（`NlfError` 消息里要带上底层原因）。 */
function errText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** 错误消息里显示 JSON 值（字符串加引号，便于和"缺字段"区分）。 */
function show(value: unknown): string {
  if (typeof value === "string") return `'${value}'`;
  const json = JSON.stringify(value);
  return json === undefined ? String(value) : json;
}

/** 归一化输入：`ArrayBuffer` → 视图；`Uint8Array`（含 Buffer 子类）原样使用。 */
function toBytes(bytes: Uint8Array | ArrayBuffer): Uint8Array {
  return bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
}

/** dtype 白名单判定（`f2` 不在其中：Python 侧 `DTYPE_CODES` 也没有它）。 */
function isDtype(value: string): value is Dtype {
  return Object.prototype.hasOwnProperty.call(DTYPE_ITEMSIZE, value);
}

/** shape 的元素个数；空 shape 视为 1（0 维标量）。 */
function shapeSize(shape: number[]): number {
  let size = 1;
  for (const d of shape) size *= d;
  return size;
}

/** 拼接读流分片（DecompressionStream 不保证一次给全）。 */
function concatChunks(chunks: Uint8Array[]): Uint8Array {
  if (chunks.length === 1) return chunks[0];
  let total = 0;
  for (const c of chunks) total += c.byteLength;
  const out = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out;
}

// --------------------------------------------------------------------------- //
// 元素读写（dtype × 字节序）
// --------------------------------------------------------------------------- //

/** 从 DataView 读一个元素；`littleEndian` 只对多字节 dtype 有意义。 */
function readElement(view: DataView, byteOffset: number, dtype: Dtype, littleEndian: boolean): number | bigint {
  switch (dtype) {
    case "f8":
      return view.getFloat64(byteOffset, littleEndian);
    case "f4":
      return view.getFloat32(byteOffset, littleEndian);
    case "i8":
      return view.getBigInt64(byteOffset, littleEndian);
    case "i4":
      return view.getInt32(byteOffset, littleEndian);
    case "i2":
      return view.getInt16(byteOffset, littleEndian);
    case "u1":
      return view.getUint8(byteOffset);
    case "u2":
      return view.getUint16(byteOffset, littleEndian);
    case "u4":
      return view.getUint32(byteOffset, littleEndian);
    case "bool":
      // 与 numpy 的 bool 视图一致：非 0 即 true，但对外统一成 0/1。
      return view.getUint8(byteOffset) !== 0 ? 1 : 0;
  }
}

/** 往 DataView 写一个元素；`bool` 只写 0/1。 */
function writeElement(
  view: DataView,
  byteOffset: number,
  dtype: Dtype,
  value: number | bigint,
  littleEndian: boolean,
): void {
  switch (dtype) {
    case "f8":
      view.setFloat64(byteOffset, Number(value), littleEndian);
      return;
    case "f4":
      view.setFloat32(byteOffset, Number(value), littleEndian);
      return;
    case "i8":
      view.setBigInt64(byteOffset, typeof value === "bigint" ? value : BigInt(Math.trunc(value)), littleEndian);
      return;
    case "i4":
      view.setInt32(byteOffset, Number(value), littleEndian);
      return;
    case "i2":
      view.setInt16(byteOffset, Number(value), littleEndian);
      return;
    case "u1":
      view.setUint8(byteOffset, Number(value));
      return;
    case "u2":
      view.setUint16(byteOffset, Number(value), littleEndian);
      return;
    case "u4":
      view.setUint32(byteOffset, Number(value), littleEndian);
      return;
    case "bool":
      view.setUint8(byteOffset, Number(value) !== 0 ? 1 : 0);
      return;
  }
}

/** 统一的按索引写入：TypedArray 联合类型的索引赋值 TS 表达不了，在这里集中转型一次。 */
function setElement(arr: TypedArray, index: number, value: number | bigint): void {
  (arr as unknown as { [i: number]: number | bigint })[index] = value;
}

/** 按 dtype 分配一个空的 C 序数组。 */
function allocate(dtype: Dtype, size: number): TypedArray {
  switch (dtype) {
    case "f8":
      return new Float64Array(size);
    case "f4":
      return new Float32Array(size);
    case "i8":
      return new BigInt64Array(size);
    case "i4":
      return new Int32Array(size);
    case "i2":
      return new Int16Array(size);
    case "u1":
      return new Uint8Array(size);
    case "u2":
      return new Uint16Array(size);
    case "u4":
      return new Uint32Array(size);
    case "bool":
      return new Uint8Array(size);
  }
}

/** 为一整块字节（byteOffset=0 的新副本）建立对应 dtype 的视图；`bool` 直接用 u1 字节。 */
function viewOf(chunk: Uint8Array, dtype: Dtype): TypedArray {
  const buf = chunk.buffer;
  const off = chunk.byteOffset;
  const n = chunk.byteLength;
  switch (dtype) {
    case "f8":
      return new Float64Array(buf, off, n / 8);
    case "f4":
      return new Float32Array(buf, off, n / 4);
    case "i8":
      return new BigInt64Array(buf, off, n / 8);
    case "i4":
      return new Int32Array(buf, off, n / 4);
    case "i2":
      return new Int16Array(buf, off, n / 2);
    case "u1":
    case "bool":
      return chunk;
    case "u2":
      return new Uint16Array(buf, off, n / 2);
    case "u4":
      return new Uint32Array(buf, off, n / 4);
  }
}

/** 把 bool 字节规整成 0/1（numpy 的 bool 视图是"非 0 即真"，我们对齐这个语义）。 */
function normalizeBoolBytes(bytes: Uint8Array): void {
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    if (b !== 0 && b !== 1) bytes[i] = 1;
  }
}

// --------------------------------------------------------------------------- //
// C 序 ↔ F 序 索引映射
// --------------------------------------------------------------------------- //

/**
 * 按 **C 序**（行主序）遍历 `shape` 的每个元素，同时给出它在 **F 序**（列主序）
 * 扁平缓冲里的位置。解码 F 序数组时 `out[c] = src[f]`，编码时反过来。
 *
 * 用进位加法（里程表）而不是先算整个置换表：F 序数组可能很大，多分配一份
 * size 长度的索引表在浏览器里不值得。
 */
function forEachCIndexAndFPosition(shape: number[], visit: (cIndex: number, fPosition: number) => void): void {
  const nd = shape.length;
  if (nd <= 1) {
    const n = nd === 0 ? 1 : shape[0];
    for (let i = 0; i < n; i++) visit(i, i);
    return;
  }
  const fStride = new Array<number>(nd);
  let acc = 1;
  for (let j = 0; j < nd; j++) {
    fStride[j] = acc;
    acc *= shape[j];
  }
  const size = acc;
  const idx = new Array<number>(nd).fill(0);
  for (let c = 0; c < size; c++) {
    let f = 0;
    for (let j = 0; j < nd; j++) f += idx[j] * fStride[j];
    visit(c, f);
    for (let j = nd - 1; j >= 0; j--) {
      idx[j] += 1;
      if (idx[j] < shape[j]) break;
      idx[j] = 0;
    }
  }
}

// --------------------------------------------------------------------------- //
// header 解析
// --------------------------------------------------------------------------- //

interface ParsedNlf {
  header: Record<string, unknown>;
  /** 未解压的 payload 视图（`codec === "none"` 时它就是 payload 本体）。 */
  payload: Uint8Array;
  codec: FrameCodec;
}

/**
 * 校验并解析"magic + header_len + header JSON"这一段。
 *
 * 校验顺序刻意与 `nlf.py::unpack_frame` 一致（先能读出的错先报），
 * 这样两侧对同一个坏文件给出的失败原因相同，对拍时不会出现"都失败但原因不同"。
 */
function parseHeader(data: Uint8Array): ParsedNlf {
  if (data.byteLength < 8) {
    throw new NlfError(`数据太短：${data.byteLength} 字节`);
  }
  for (let i = 0; i < NLF_MAGIC.length; i++) {
    if (data[i] !== NLF_MAGIC[i]) {
      const got = Array.from(data.subarray(0, 4), (b) => b.toString(16).padStart(2, "0")).join(" ");
      throw new NlfError(`magic 不符：0x${got} != 0x4e 0x4c 0x46 0x31（"NLF1"）`);
    }
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const headerLen = view.getUint32(4, true);
  if (headerLen > MAX_HEADER_LEN || 8 + headerLen > data.byteLength) {
    throw new NlfError(`header_len 非法：${headerLen}（总长 ${data.byteLength}）`);
  }

  let text: string;
  try {
    // fatal: true 让非法 UTF-8 直接报错，而不是塞一堆 U+FFFD 后解析出一个"像样"的 JSON。
    text = new TextDecoder("utf-8", { fatal: true }).decode(data.subarray(8, 8 + headerLen));
  } catch (err) {
    throw new NlfError(`header JSON 解析失败：${errText(err)}`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new NlfError(`header JSON 解析失败：${errText(err)}`);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new NlfError("header 必须是 JSON 对象");
  }
  const header = parsed as Record<string, unknown>;

  if (header.version !== NLF_VERSION) {
    throw new NlfError(`版本不支持：${show(header.version)}（本实现支持 ${NLF_VERSION}）`);
  }
  if (header.endian !== "little") {
    throw new NlfError(`endian 非法：${show(header.endian)}`);
  }
  const rawCodec = header.codec === undefined ? "none" : header.codec;
  if (rawCodec !== "none" && rawCodec !== "zlib" && rawCodec !== "gzip") {
    throw new NlfError(`codec 非法：${show(rawCodec)}`);
  }

  return { header, payload: data.subarray(8 + headerLen), codec: rawCodec };
}

// --------------------------------------------------------------------------- //
// header 字段读取（比 Python 更严：宁可报错，也不返回与声明类型不符的值）
// --------------------------------------------------------------------------- //

/** `kind` 只允许协议里的三种取值。 */
function readKind(raw: unknown): FrameKind {
  const value = raw === undefined ? "frame" : raw;
  if (value !== "frame" && value !== "dataset" && value !== "manifest") {
    throw new NlfError(`kind 非法：${show(value)}`);
  }
  return value;
}

/** `chapter` / `viz`：允许字符串或数字（Python 用 `str()` 强转），其余拒绝。 */
function readLabel(raw: unknown, field: string): string {
  if (raw === undefined) return "";
  if (typeof raw === "string") return raw;
  if (typeof raw === "number") return String(raw);
  throw new NlfError(`${field} 必须是字符串：${show(raw)}`);
}

/** `meta` 只允许 number / boolean / null——塞数组会破坏"前端自算物理量无处藏身"的设计。 */
function readMeta(raw: unknown): Record<string, number | boolean | null> {
  if (raw === undefined) return {};
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new NlfError(`meta 必须是 JSON 对象：${show(raw)}`);
  }
  const out: Record<string, number | boolean | null> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (value === null || typeof value === "number" || typeof value === "boolean") {
      out[key] = value;
    } else {
      throw new NlfError(`meta['${key}'] 必须是 number/bool/null；数组请走 arrays（PROTOCOLS §1.1）`);
    }
  }
  return out;
}

/** `strings`：字符串标量；number/boolean 按 Python `str()` 的习惯强转。 */
function readStrings(raw: unknown): Record<string, string> {
  if (raw === undefined) return {};
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new NlfError(`strings 必须是 JSON 对象：${show(raw)}`);
  }
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (typeof value === "string") out[key] = value;
    else if (typeof value === "number" || typeof value === "boolean") out[key] = String(value);
    else throw new NlfError(`strings['${key}'] 必须是字符串标量：${show(value)}`);
  }
  return out;
}

/** `critical`：参数名 → 理论临界值（数值或数值数组）。 */
function readCritical(raw: unknown): Record<string, number | number[]> {
  if (raw === undefined) return {};
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new NlfError(`critical 必须是 JSON 对象：${show(raw)}`);
  }
  const out: Record<string, number | number[]> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (typeof value === "number") {
      out[key] = value;
    } else if (Array.isArray(value) && value.every((x) => typeof x === "number")) {
      out[key] = value as number[];
    } else {
      throw new NlfError(`critical['${key}'] 必须是数值或数值数组：${show(value)}`);
    }
  }
  return out;
}

/** `pitfalls`：防误导文本，必须是字符串。 */
function readPitfalls(raw: unknown): string {
  if (raw === undefined) return "";
  if (typeof raw !== "string") throw new NlfError(`pitfalls 必须是字符串：${show(raw)}`);
  return raw;
}

/** `generator`：写帧者信息，只做透传（键值都按字符串存）。 */
function readGenerator(raw: unknown): Record<string, string> {
  if (raw === undefined) return {};
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new NlfError(`generator 必须是 JSON 对象：${show(raw)}`);
  }
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(raw)) out[key] = typeof value === "string" ? value : String(value);
  return out;
}

/** `annotations`：本实现不解释其内容，只要求是对象数组。 */
function readAnnotations(raw: unknown): Array<Record<string, unknown>> {
  if (raw === undefined) return [];
  if (!Array.isArray(raw)) throw new NlfError(`annotations 必须是数组：${show(raw)}`);
  return raw.map((item) => {
    if (typeof item !== "object" || item === null || Array.isArray(item)) {
      throw new NlfError(`annotations 元素必须是对象：${show(item)}`);
    }
    return item as Record<string, unknown>;
  });
}

// --------------------------------------------------------------------------- //
// 数组元信息校验
// --------------------------------------------------------------------------- //

/** 解析并校验一个 `arrays[i]` 条目（顺序与 `nlf.py` 的检查顺序一致）。 */
function readSpec(raw: unknown): ArraySpec {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new NlfError("arrays 元素必须是对象");
  }
  const item = raw as Record<string, unknown>;
  for (const key of ["name", "dtype", "shape", "offset", "nbytes"]) {
    if (item[key] === undefined) throw new NlfError(`arrays 元素缺少字段：'${key}'`);
  }

  const name = String(item.name);
  const dtype = String(item.dtype);
  if (!isDtype(dtype)) throw new NlfError(`dtype 非法：${show(dtype)}`);

  const rawShape = item.shape;
  if (!Array.isArray(rawShape)) throw new NlfError(`shape 必须是数组：${show(rawShape)}`);
  const shape = rawShape.map((d) => {
    if (typeof d !== "number" || !Number.isSafeInteger(d) || d < 0) {
      throw new NlfError(`shape 元素必须是非负整数：${show(d)}（数组 '${name}'）`);
    }
    return d;
  });

  const rawOrder = item.order === undefined ? "C" : item.order;
  if (rawOrder !== "C" && rawOrder !== "F") throw new NlfError(`order 非法：${show(rawOrder)}`);

  const offset = item.offset;
  const nbytes = item.nbytes;
  if (typeof offset !== "number" || !Number.isSafeInteger(offset)) {
    throw new NlfError(`offset 必须是整数：${show(offset)}（数组 '${name}'）`);
  }
  if (typeof nbytes !== "number" || !Number.isSafeInteger(nbytes)) {
    throw new NlfError(`nbytes 必须是整数：${show(nbytes)}（数组 '${name}'）`);
  }
  if (nbytes !== shapeSize(shape) * DTYPE_ITEMSIZE[dtype]) {
    throw new NlfError(`nbytes 与 shape/dtype 不符：'${name}'`);
  }

  const unit = item.unit === undefined ? "" : String(item.unit);
  const label = item.label === undefined ? "" : String(item.label);
  return { name, dtype, shape, order: rawOrder, offset, nbytes, unit, label };
}

/**
 * 解出一个数组的数据：先整块拷出（顺带解决对齐问题），再按 order 规范化成 C 序。
 *
 * 只有 F 序（且元素数 > 1）或大端宿主才需要逐元素搬；小端 + C 序走零拷贝视图，
 * 这条路径决定了 8 MB 帧的解码是几十毫秒而不是几百毫秒。
 */
function decodeArray(payload: Uint8Array, spec: ArraySpec): TypedArray {
  const chunk = new Uint8Array(spec.nbytes);
  chunk.set(payload.subarray(spec.offset, spec.offset + spec.nbytes));

  const size = shapeSize(spec.shape);
  const needsGather = spec.order === "F" && size > 1;

  if (!needsGather && (HOST_LITTLE_ENDIAN || spec.dtype === "bool")) {
    const arr = viewOf(chunk, spec.dtype);
    if (spec.dtype === "bool") normalizeBoolBytes(chunk);
    return arr;
  }

  const out = allocate(spec.dtype, size);
  const view = new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  const itemsize = DTYPE_ITEMSIZE[spec.dtype];
  if (needsGather) {
    forEachCIndexAndFPosition(spec.shape, (c, f) => {
      setElement(out, c, readElement(view, f * itemsize, spec.dtype, true));
    });
  } else {
    for (let i = 0; i < size; i++) {
      setElement(out, i, readElement(view, i * itemsize, spec.dtype, true));
    }
  }
  return out;
}

/** 由已校验的 header + payload 组装 `Frame`（数组校验与重叠检查都在这里）。 */
function buildFrame(header: Record<string, unknown>, payload: Uint8Array, codec: FrameCodec): Frame {
  const rawArrays = header.arrays === undefined ? [] : header.arrays;
  if (!Array.isArray(rawArrays)) throw new NlfError(`arrays 必须是数组：${show(rawArrays)}`);

  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  const spans: Array<{ start: number; end: number; name: string }> = [];
  let total = 0;

  for (const rawSpec of rawArrays) {
    const spec = readSpec(rawSpec);
    if (spec.offset < 0 || spec.offset + spec.nbytes > payload.byteLength) {
      throw new NlfError(
        `数组越界：'${spec.name}' offset=${spec.offset} nbytes=${spec.nbytes} payload=${payload.byteLength}`,
      );
    }
    arrays.set(spec.name, decodeArray(payload, spec));
    specs.push(spec);
    spans.push({ start: spec.offset, end: spec.offset + spec.nbytes, name: spec.name });
    total += spec.nbytes;
  }

  if (total !== payload.byteLength) {
    throw new NlfError(`payload 长度与数组总和不符：${payload.byteLength} != ${total}`);
  }

  // 区间重叠检查放在最后：即使每个数组都"在界内"，两个数组覆盖同一段字节
  // 也说明写端布局错了，此时返回任何数据都是在替写端掩饰 bug。
  spans.sort((a, b) => a.start - b.start || a.end - b.end || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  for (let i = 1; i < spans.length; i++) {
    if (spans[i - 1].end > spans[i].start) {
      throw new NlfError(`数组区间重叠：'${spans[i - 1].name}' 与 '${spans[i].name}'`);
    }
  }

  return new Frame({
    chapter: readLabel(header.chapter, "chapter"),
    viz: readLabel(header.viz, "viz"),
    kind: readKind(header.kind),
    codec,
    meta: readMeta(header.meta),
    strings: readStrings(header.strings),
    critical: readCritical(header.critical),
    pitfalls: readPitfalls(header.pitfalls),
    generator: readGenerator(header.generator),
    annotations: readAnnotations(header.annotations),
    specs,
    arrays,
  });
}

// --------------------------------------------------------------------------- //
// 压缩 / 解压（仅用 Web 标准流，Node ≥ 18 与所有现代浏览器都有）
// --------------------------------------------------------------------------- //

/** 把 payload 喂给 DecompressionStream 并收齐结果；失败一律包成 `NlfError`。 */
async function inflatePayload(payload: Uint8Array, codec: "zlib" | "gzip"): Promise<Uint8Array> {
  const label = codec === "zlib" ? "zlib" : "gzip";
  if (typeof DecompressionStream !== "function") {
    throw new NlfError(`当前运行时不支持 DecompressionStream，无法解码 ${label} 压缩帧`);
  }
  try {
    const stream = new DecompressionStream(codec === "zlib" ? "deflate" : "gzip");
    // 写端独立 catch：读端会给出真正的解压错误，这里只需避免未处理的 rejection。
    const pump = (async () => {
      const writer = stream.writable.getWriter();
      try {
        // 复制一份：流 API 要求 BufferSource 背后是 ArrayBuffer，而传入的视图
        // 可能挂在 SharedArrayBuffer 上；这里只多一次压缩路径上的拷贝。
        await writer.write(new Uint8Array(payload));
        await writer.close();
      } catch {
        /* 读端会报同一个错误，见下 */
      }
    })();
    const chunks: Uint8Array[] = [];
    const reader = stream.readable.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value !== undefined) chunks.push(value);
    }
    await pump;
    return concatChunks(chunks);
  } catch (err) {
    throw new NlfError(`${label} 解压失败：${errText(err)}`);
  }
}

/** 用 CompressionStream 生成 zlib / gzip 字节流。 */
async function deflatePayload(payload: Uint8Array, codec: "zlib" | "gzip"): Promise<Uint8Array> {
  const label = codec === "zlib" ? "zlib" : "gzip";
  if (typeof CompressionStream !== "function") {
    throw new NlfError(`当前运行时不支持 CompressionStream，无法写出 ${label} 压缩帧`);
  }
  try {
    const stream = new CompressionStream(codec === "zlib" ? "deflate" : "gzip");
    const pump = (async () => {
      const writer = stream.writable.getWriter();
      try {
        // 复制一份：流 API 要求 BufferSource 背后是 ArrayBuffer，而传入的视图
        // 可能挂在 SharedArrayBuffer 上；这里只多一次压缩路径上的拷贝。
        await writer.write(new Uint8Array(payload));
        await writer.close();
      } catch {
        /* 读端会报同一个错误 */
      }
    })();
    const chunks: Uint8Array[] = [];
    const reader = stream.readable.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value !== undefined) chunks.push(value);
    }
    await pump;
    return concatChunks(chunks);
  } catch (err) {
    throw new NlfError(`${label} 压缩失败：${errText(err)}`);
  }
}

// --------------------------------------------------------------------------- //
// 解码
// --------------------------------------------------------------------------- //

/**
 * 同步解码 `codec === "none"` 的帧。
 *
 * 压缩帧会抛 `NlfError`（解压是异步 API）——这是刻意的：静默返回半个帧比报错危险得多。
 * `bytes` 可以是 `Uint8Array` 或 `ArrayBuffer`。
 */
export function decodeNlf(bytes: Uint8Array | ArrayBuffer): Frame {
  const data = toBytes(bytes);
  const parsed = parseHeader(data);
  if (parsed.codec !== "none") {
    throw new NlfError(
      `帧使用 ${parsed.codec} 压缩，请改用 await decodeNlfAsync()（decodeNlf 只处理 codec="none"）`,
    );
  }
  return buildFrame(parsed.header, parsed.payload, parsed.codec);
}

/**
 * 异步解码任意 codec 的帧（zlib / gzip 经 `DecompressionStream`）。
 * 除了解压这一步，其它校验与 `decodeNlf` 完全相同。
 */
export async function decodeNlfAsync(bytes: Uint8Array | ArrayBuffer): Promise<Frame> {
  const data = toBytes(bytes);
  const parsed = parseHeader(data);
  const payload = parsed.codec === "none" ? parsed.payload : await inflatePayload(parsed.payload, parsed.codec);
  return buildFrame(parsed.header, payload, parsed.codec);
}

// --------------------------------------------------------------------------- //
// 编码
// --------------------------------------------------------------------------- //

/** 写帧前自检：与 `nlf.py::Frame::validate` 对齐（写端不该产出解码端会拒绝的东西）。 */
function validateForEncode(frame: Frame): void {
  if (frame.kind !== "frame" && frame.kind !== "dataset" && frame.kind !== "manifest") {
    throw new NlfError(`kind 非法：${show(frame.kind)}`);
  }
  if (frame.codec !== "none" && frame.codec !== "zlib" && frame.codec !== "gzip") {
    throw new NlfError(`codec 非法：${show(frame.codec)}`);
  }
  if (typeof frame.pitfalls !== "string") throw new NlfError("pitfalls 必须是字符串");

  const seen = new Set<string>();
  for (const spec of frame.specs) {
    if (seen.has(spec.name)) throw new NlfError(`数组名重复：'${spec.name}'`);
    seen.add(spec.name);
    if (!isDtype(spec.dtype)) throw new NlfError(`dtype 非法：${show(spec.dtype)}`);
    if (spec.shape.length > 3) throw new NlfError(`shape 维数 > 3：'${spec.name}' [${spec.shape.join(", ")}]`);
    if (spec.order !== "C" && spec.order !== "F") throw new NlfError(`order 非法：${show(spec.order)}`);
    if (spec.nbytes !== shapeSize(spec.shape) * DTYPE_ITEMSIZE[spec.dtype]) {
      throw new NlfError(`nbytes 与 shape/dtype 不符：'${spec.name}'`);
    }
    if (!frame.has(spec.name)) throw new NlfError(`specs 与 arrays 不一致：缺少数组 '${spec.name}'`);
  }
  for (const [key, value] of Object.entries(frame.meta)) {
    if (value === null || typeof value === "number" || typeof value === "boolean") continue;
    throw new NlfError(`meta['${key}'] 非标量`);
  }
  for (const [key, value] of Object.entries(frame.critical)) {
    if (typeof value === "number") continue;
    if (Array.isArray(value) && value.every((x) => typeof x === "number")) continue;
    throw new NlfError(`critical['${key}'] 必须是数值或数值数组`);
  }
}

/** 数组的原始字节（零拷贝视图）。 */
function rawBytesOf(arr: TypedArray): Uint8Array {
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
}

/** bool 数组写出前规整成 0/1（Python 侧 `np.bool_` 早已是 0/1，这里补齐手工造帧的情况）。 */
function rawBytesForWrite(arr: TypedArray, dtype: Dtype): Uint8Array {
  const raw = rawBytesOf(arr);
  if (dtype !== "bool") return raw;
  let dirty = false;
  for (let i = 0; i < raw.length; i++) {
    const b = raw[i];
    if (b !== 0 && b !== 1) {
      dirty = true;
      break;
    }
  }
  if (!dirty) return raw;
  const copy = raw.slice();
  normalizeBoolBytes(copy);
  return copy;
}

/**
 * 把一个数组写进 payload 的指定位置。
 *
 * 写端必须**忠实还原 `order`**：如果内部 C 序数据配上头部 `order:"F"` 直接按 C 序写出，
 * Python 解出来就是转置矩阵（f_order 夹具专门守这条）。
 */
function encodeArray(payload: Uint8Array, payloadOffset: number, spec: ArraySpec, arr: TypedArray): void {
  const itemsize = DTYPE_ITEMSIZE[spec.dtype];
  const size = shapeSize(spec.shape);
  if (arr.length !== size) {
    throw new NlfError(`数组元素个数与 shape 不符：'${spec.name}' ${arr.length} != ${size}`);
  }
  const raw = rawBytesForWrite(arr, spec.dtype);
  const gather = spec.order === "F" && size > 1;

  if (HOST_LITTLE_ENDIAN) {
    if (!gather) {
      payload.set(raw, payloadOffset);
      return;
    }
    // F 序：按元素字节搬到 F 序缓冲再整块拷贝（对任意 itemsize 都成立，且不依赖宿主字节序）。
    const tmp = new Uint8Array(spec.nbytes);
    forEachCIndexAndFPosition(spec.shape, (c, f) => {
      tmp.set(raw.subarray(c * itemsize, c * itemsize + itemsize), f * itemsize);
    });
    payload.set(tmp, payloadOffset);
    return;
  }

  // 大端宿主：源按宿主序解释，目标固定 little-endian。
  const src = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const dst = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  if (gather) {
    forEachCIndexAndFPosition(spec.shape, (c, f) => {
      writeElement(dst, payloadOffset + f * itemsize, spec.dtype, readElement(src, c * itemsize, spec.dtype, false), true);
    });
  } else {
    for (let i = 0; i < size; i++) {
      writeElement(dst, payloadOffset + i * itemsize, spec.dtype, readElement(src, i * itemsize, spec.dtype, false), true);
    }
  }
}

/**
 * 把一帧编码成 `.nlf` 字节串（对拍与测试用；也是唯一被允许的"造帧"出口）。
 *
 * * 各数组的 `offset` **重新计算**（按 `specs` 顺序紧排），不信任 `spec.offset`——
 *   与 `nlf.py::pack_frame` 一致，免得调用方手工算偏移算错还写出一个自洽性错误的文件。
 * * 数值一律 little-endian；`opts.codec` 覆盖帧自身的 codec（与 Python 的 `codec=` 参数一致）。
 * * `opts.codec` 只声明 `"none" | "zlib"`；帧自带 `codec === "gzip"` 时按 gzip 写，
 *   否则"头里写 gzip、字节里是明文"会造出一个谁也算不清的文件。
 */
export async function encodeNlf(frame: Frame, opts?: { codec?: "none" | "zlib" }): Promise<Uint8Array> {
  validateForEncode(frame);
  const codec: FrameCodec = opts?.codec ?? frame.codec;

  let payloadLength = 0;
  for (const spec of frame.specs) payloadLength += spec.nbytes;
  const payload = new Uint8Array(payloadLength);

  const headerSpecs: Array<Record<string, unknown>> = [];
  let offset = 0;
  for (const spec of frame.specs) {
    encodeArray(payload, offset, spec, frame.array(spec.name));
    headerSpecs.push({
      name: spec.name,
      dtype: spec.dtype,
      shape: [...spec.shape],
      order: spec.order,
      offset,
      nbytes: spec.nbytes,
      unit: spec.unit,
      label: spec.label,
    });
    offset += spec.nbytes;
  }

  const header: Record<string, unknown> = {
    version: NLF_VERSION,
    kind: frame.kind,
    chapter: frame.chapter,
    viz: frame.viz,
    codec,
    endian: "little",
    generator: Object.keys(frame.generator).length > 0 ? frame.generator : DEFAULT_GENERATOR,
    meta: frame.meta,
    strings: frame.strings,
    critical: frame.critical,
    pitfalls: frame.pitfalls,
    arrays: headerSpecs,
  };
  if (frame.annotations.length > 0) header.annotations = frame.annotations;

  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  if (headerBytes.byteLength > MAX_HEADER_LEN) {
    throw new NlfError(`header 过大：${headerBytes.byteLength} 字节 > ${MAX_HEADER_LEN}`);
  }

  let body: Uint8Array = payload;
  if (codec === "zlib") body = await deflatePayload(payload, "zlib");
  else if (codec === "gzip") body = await deflatePayload(payload, "gzip");

  const out = new Uint8Array(8 + headerBytes.byteLength + body.byteLength);
  out.set(NLF_MAGIC, 0);
  new DataView(out.buffer).setUint32(4, headerBytes.byteLength, true);
  out.set(headerBytes, 8);
  out.set(body, 8 + headerBytes.byteLength);
  return out;
}

// --------------------------------------------------------------------------- //
// 摘要
// --------------------------------------------------------------------------- //

/**
 * 一帧的人类可读摘要（日志 / 测试失败信息用），与 `nlf.py::frame_summary` 同构：
 * 第一行给"帧是谁、多少数组、多少字节"，随后每个数组一行，最后给 meta。
 */
export function frameSummary(frame: Frame): string {
  const total = frame.specs.reduce((sum, s) => sum + s.nbytes, 0);
  const parts = [
    `<Frame ${frame.chapter}/${frame.viz} kind=${frame.kind} codec=${frame.codec} ` +
      `arrays=${frame.specs.length} bytes=${total}>`,
  ];
  for (const s of frame.specs) {
    // Python 的 str.rstrip() 等价物：unit/label 为空时不要留下尾随空格。
    parts.push(`  · ${s.name}: ${s.dtype}[${s.shape.join(", ")}] order=${s.order} ${s.unit} ${s.label}`.replace(/\s+$/, ""));
  }
  if (Object.keys(frame.meta).length > 0) parts.push(`  meta: ${JSON.stringify(frame.meta)}`);
  return parts.join("\n");
}
