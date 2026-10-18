/**
 * `.nlf` 解码器 / 编码器的验收测试（PROTOCOLS §1 与 §7 的可执行部分）。
 *
 * 这里有三类断言，各自防一种失败模式：
 *
 * 1. **夹具对拍**：`tests/fixtures/nlf/*.nlf` 是 Python 写的，`tests/parity/golden/nlf_*.json`
 *    是 Python 认的标准答案；逐值相等才说明"两个实现没有漂移"（G7 的 TS 侧证据）。
 * 2. **失败路径**：坏字节必须抛 `NlfError`。解码端"宁可报错也不返回可疑数据"这条承诺
 *    只有被逐条钉住才算数——否则它会在某次重构里悄悄退化成"尽量解"。
 * 3. **性能**：8 MB 帧的编解码必须远低于 1.5 s，用来挡住"逐元素 toString / 逐元素字符串拼接"
 *    这类在小夹具上看不出来、一上真数据就崩的实现。
 *
 * 测试里允许用 `node:fs` / `node:test` / `node:assert`；被测代码只用 JS 内建（P7）。
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { DTYPE_ITEMSIZE, Frame, NlfError } from "../shared/src/frame.js";
import type { ArraySpec, Dtype, FrameInit, TypedArray } from "../shared/src/frame.js";
import {
  MAX_HEADER_LEN,
  NLF_MAGIC,
  NLF_VERSION,
  decodeNlf,
  decodeNlfAsync,
  encodeNlf,
  frameSummary,
} from "../shared/src/nlf.js";

// --------------------------------------------------------------------------- //
// 夹具 / 标准答案
// --------------------------------------------------------------------------- //

/** 编译后测试位于 `dist/viz/tests/`，所以 `../../../tests/...` 指向仓库根。 */
const FIXTURE_DIR = new URL("../../../tests/fixtures/nlf/", import.meta.url);
const GOLDEN_DIR = new URL("../../../tests/parity/golden/", import.meta.url);

interface GoldenArray {
  /** numpy 的 dtype 名（float64 / int32 / uint8 / bool…），用来与 dtype 代号互查。 */
  dtype: string;
  shape: number[];
  /** C 序展开的数值（Python 以 17 位有效数字写出，可精确还原成同一个 double）。 */
  values: number[];
}

interface Golden {
  chapter: string;
  viz: string;
  kind: string;
  codec: string;
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical: Record<string, number | number[]>;
  pitfalls: string;
  arrays: Record<string, GoldenArray>;
}

interface FixtureEntry {
  name: string;
  file: string;
  bytes: string;
  arrays: string;
}

const FIXTURES = (JSON.parse(readFileSync(new URL("index.json", FIXTURE_DIR), "utf8")) as {
  fixtures: FixtureEntry[];
}).fixtures;

function readFixtureBytes(file: string): Uint8Array {
  return new Uint8Array(readFileSync(new URL(file, FIXTURE_DIR)));
}

function readGolden(name: string): Golden {
  return JSON.parse(readFileSync(new URL(`nlf_${name}.json`, GOLDEN_DIR), "utf8")) as Golden;
}

async function loadFrame(name: string, golden: Golden): Promise<Frame> {
  const bytes = readFixtureBytes(`${name}.nlf`);
  return golden.codec === "none" ? decodeNlf(bytes) : await decodeNlfAsync(bytes);
}

/** dtype 代号 → 标准答案里的 numpy 名字。 */
const GOLDEN_DTYPE: Record<Dtype, string> = {
  f8: "float64",
  f4: "float32",
  i8: "int64",
  i4: "int32",
  i2: "int16",
  u1: "uint8",
  u2: "uint16",
  u4: "uint32",
  bool: "bool",
};

/** dtype 代号 → 期望的 TypedArray 具体类名（`instanceof` 在跨 realm 时不可靠，比类名）。 */
const DTYPE_CTOR: Record<Dtype, string> = {
  f8: "Float64Array",
  f4: "Float32Array",
  i8: "BigInt64Array",
  i4: "Int32Array",
  i2: "Int16Array",
  u1: "Uint8Array",
  u2: "Uint16Array",
  u4: "Uint32Array",
  bool: "Uint8Array",
};

const DTYPE_OF_GOLDEN = new Map<string, Dtype>(
  Object.entries(GOLDEN_DTYPE).map(([code, numpyName]) => [numpyName, code as Dtype]),
);

/** 把 TypedArray 摊成 number[]（i8 用 Number() 降级，夹具里没有 i8，仅防御）。 */
function collect(arr: TypedArray): number[] {
  const out: number[] = [];
  for (let i = 0; i < arr.length; i++) out.push(Number(arr[i]));
  return out;
}

function assertArrayMatches(frame: Frame, name: string, expected: GoldenArray, where: string): void {
  assert.ok(frame.has(name), `${where}：帧里缺少数组 ${name}`);
  const spec = frame.specs.find((s) => s.name === name);
  assert.ok(spec !== undefined, `${where}：specs 里缺少 ${name}`);
  const arr = frame.array(name);
  const dtype = DTYPE_OF_GOLDEN.get(expected.dtype);
  assert.ok(dtype !== undefined, `${where}：标准答案里出现了未知 dtype ${expected.dtype}`);

  assert.equal(spec.dtype, dtype, `${where}.${name}：dtype 不符`);
  assert.equal(typeof arr, "object", `${where}.${name}：数组应当是可索引对象`);
  assert.equal(arr.constructor.name, DTYPE_CTOR[dtype], `${where}.${name}：运行时类型不符`);
  assert.deepEqual(spec.shape, expected.shape, `${where}.${name}：shape 不符`);
  assert.equal(arr.length, expected.values.length, `${where}.${name}：元素个数不符`);
  if (expected.values.length > 0) {
    assert.equal(typeof arr[0], dtype === "i8" ? "bigint" : "number", `${where}.${name}：元素类型不符`);
  }
  for (let i = 0; i < expected.values.length; i++) {
    const got = arr[i];
    assert.equal(typeof got === "bigint" ? Number(got) : got, expected.values[i], `${where}.${name}[${i}] 数值不符`);
  }
}

// --------------------------------------------------------------------------- //
// 1. 六个夹具逐值对拍
// --------------------------------------------------------------------------- //

for (const fx of FIXTURES) {
  test(`夹具 ${fx.name}：解码结果与标准答案逐值相等`, async () => {
    const golden = readGolden(fx.name);
    assert.equal(readFixtureBytes(fx.file).byteLength, Number(fx.bytes), `${fx.name}：夹具字节数与 index.json 不符`);

    const frame = await loadFrame(fx.name, golden);
    assert.deepEqual(frame.names.slice().sort(), Object.keys(golden.arrays).sort(), `${fx.name}：数组集合不符`);
    assert.deepEqual(frame.names, fx.arrays.split(","), `${fx.name}：数组顺序应当等于 payload 顺序`);

    for (const [name, expected] of Object.entries(golden.arrays)) {
      assertArrayMatches(frame, name, expected, fx.name);
    }
  });

  test(`夹具 ${fx.name}：header 字段与标准答案一致`, async () => {
    const golden = readGolden(fx.name);
    const frame = await loadFrame(fx.name, golden);

    assert.equal(frame.chapter, golden.chapter, `${fx.name}.chapter`);
    assert.equal(frame.viz, golden.viz, `${fx.name}.viz`);
    assert.equal(frame.kind, golden.kind, `${fx.name}.kind`);
    assert.equal(frame.codec, golden.codec, `${fx.name}.codec`);
    assert.equal(frame.pitfalls, golden.pitfalls, `${fx.name}.pitfalls`);
    assert.deepEqual(frame.meta, golden.meta, `${fx.name}.meta`);
    assert.deepEqual(frame.strings, golden.strings, `${fx.name}.strings`);
    assert.deepEqual(frame.critical, golden.critical, `${fx.name}.critical`);
  });
}

// --------------------------------------------------------------------------- //
// 2. codec / order / dtype 的定点检查
// --------------------------------------------------------------------------- //

test("zlib_2d：decodeNlf 必须抛 NlfError，decodeNlfAsync 必须成功", async () => {
  const bytes = readFixtureBytes("zlib_2d.nlf");
  // 压缩帧的 header 是明文，所以同步解码"看得见 codec 字段但解不开 payload"是确定的事实，
  // 这里要求它明确报错，而不是偷偷返回未解压的字节。
  assert.throws(
    () => decodeNlf(bytes),
    (err: unknown) => err instanceof NlfError && /zlib/.test(err.message) && /decodeNlfAsync/.test(err.message),
  );

  const golden = readGolden("zlib_2d");
  const frame = await decodeNlfAsync(bytes);
  assert.equal(frame.codec, "zlib");
  for (const [name, expected] of Object.entries(golden.arrays)) {
    assertArrayMatches(frame, name, expected, "zlib_2d");
  }
  // 解压出来的必须是 48×48 = 2304 个 double，而不是"没解压成功但长度凑巧"的东西。
  assert.equal(frame.specs[0].nbytes, 48 * 48 * 8);
});

test("f_order：列主序在解码时重排为 C 序（AF ≡ AC，且不是转置）", () => {
  const frame = decodeNlf(readFixtureBytes("f_order.nlf"));
  const af = frame.array("AF");
  const ac = frame.array("AC");

  assert.equal(frame.specs[0].order, "F", "specs 必须保留写盘时的 order 字段");
  assert.equal(frame.specs[1].order, "C");
  assert.equal(af.constructor.name, "Float64Array");
  for (let i = 0; i < af.length; i++) assert.equal(af[i], ac[i], `AF[${i}] 与 AC[${i}] 不等`);

  const m = frame.matrix("AF");
  assert.equal(m.rows, 3);
  assert.equal(m.cols, 4);
  // A[i, j] = i + 3j：若忽略 order 直接按字节读，这里会得到转置矩阵。
  assert.deepEqual(collect(m.values), [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]);
});

test("dtypes：i → Int32Array、u → Uint8Array、flag → Uint8Array 且只含 0/1", () => {
  const frame = decodeNlf(readFixtureBytes("dtypes.nlf"));
  const i = frame.array("i");
  const u = frame.array("u");
  const flag = frame.array("flag");

  assert.equal(i.constructor.name, "Int32Array");
  assert.deepEqual(collect(i), [-3, -1, 0, 2, 7]);
  assert.equal(u.constructor.name, "Uint8Array");
  assert.deepEqual(collect(u), [0, 1, 127, 255]);
  assert.equal(flag.constructor.name, "Uint8Array");
  assert.deepEqual(collect(flag), [1, 0, 1, 1]);
  for (let k = 0; k < flag.length; k++) assert.ok(flag[k] === 0 || flag[k] === 1, `flag[${k}] 不是 0/1`);
});

// --------------------------------------------------------------------------- //
// 3. 失败路径：坏字节一律 NlfError
// --------------------------------------------------------------------------- //

const utf8 = new TextEncoder();

/** 拼一个 .nlf；`headerLenOverride` 用来故意写出与 header 实际长度不符的 header_len。 */
function rawWithHeaderText(headerText: string, payload = new Uint8Array(0), headerLenOverride?: number): Uint8Array {
  const hb = utf8.encode(headerText);
  const hlen = headerLenOverride ?? hb.byteLength;
  const out = new Uint8Array(8 + hb.byteLength + payload.byteLength);
  out.set(NLF_MAGIC, 0);
  new DataView(out.buffer).setUint32(4, hlen, true);
  out.set(hb, 8);
  out.set(payload, 8 + hb.byteLength);
  return out;
}

function headerJson(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    version: NLF_VERSION,
    kind: "frame",
    chapter: "0.0",
    viz: "bad",
    codec: "none",
    endian: "little",
    meta: {},
    strings: {},
    critical: {},
    pitfalls: "",
    arrays: [],
    ...over,
  };
}

function specJson(over: Record<string, unknown> = {}): Record<string, unknown> {
  return { name: "a", dtype: "f8", shape: [2], order: "C", offset: 0, nbytes: 16, unit: "", label: "", ...over };
}

const rawNlf = (header: Record<string, unknown>, payload = new Uint8Array(0), headerLenOverride?: number): Uint8Array =>
  rawWithHeaderText(JSON.stringify(header), payload, headerLenOverride);

/** 断言"抛的是 NlfError，且消息里提到某个关键词"——只看类型会漏掉"报错原因张冠李戴"。 */
function nlfError(pattern: RegExp): (err: unknown) => boolean {
  return (err) => err instanceof NlfError && pattern.test(err.message);
}

test("失败路径：空 / 太短的 buffer", () => {
  for (const bytes of [new Uint8Array(0), new Uint8Array(4), new Uint8Array(7)]) {
    assert.throws(() => decodeNlf(bytes), nlfError(/数据太短/));
  }
  assert.throws(() => decodeNlf(new ArrayBuffer(3)), nlfError(/数据太短/));
});

test("失败路径：magic 不符", () => {
  const bad = readFixtureBytes("basic_f8_1d.nlf").slice();
  bad.set(utf8.encode("NLF0"), 0);
  assert.throws(() => decodeNlf(bad), nlfError(/magic/));
});

test("失败路径：header_len 超出总长 / 超过上界", () => {
  assert.throws(() => decodeNlf(rawNlf(headerJson(), new Uint8Array(0), 4096)), nlfError(/header_len/));
  assert.throws(
    () => decodeNlf(rawNlf(headerJson(), new Uint8Array(0), MAX_HEADER_LEN + 1)),
    nlfError(/header_len/),
  );
});

test("失败路径：header JSON 解析失败 / 不是对象", () => {
  assert.throws(() => decodeNlf(rawWithHeaderText("{not json")), nlfError(/JSON 解析失败/));
  assert.throws(() => decodeNlf(rawWithHeaderText("[1,2,3]")), nlfError(/header 必须是 JSON 对象/));
  assert.throws(() => decodeNlf(rawWithHeaderText("42")), nlfError(/header 必须是 JSON 对象/));
});

test("失败路径：version 不是 1", () => {
  assert.throws(() => decodeNlf(rawNlf(headerJson({ version: 2 }))), nlfError(/版本不支持/));
});

test("失败路径：endian 不是 little", () => {
  assert.throws(() => decodeNlf(rawNlf(headerJson({ endian: "big" }))), nlfError(/endian/));
});

test("失败路径：codec 非法", () => {
  assert.throws(() => decodeNlf(rawNlf(headerJson({ codec: "lzma" }))), nlfError(/codec/));
});

test("失败路径：payload 总长与数组 nbytes 之和不符", () => {
  const header = headerJson({ arrays: [specJson()] }); // 声明 16 字节
  assert.throws(() => decodeNlf(rawNlf(header, new Uint8Array(24))), nlfError(/payload 长度/));
});

test("失败路径：两个数组区间重叠", () => {
  const header = headerJson({
    arrays: [specJson({ name: "a", offset: 0, nbytes: 16 }), specJson({ name: "b", offset: 8, nbytes: 16 })],
  });
  assert.throws(() => decodeNlf(rawNlf(header, new Uint8Array(32))), nlfError(/重叠/));
});

test("失败路径：nbytes 与 shape×itemsize 不符", () => {
  const header = headerJson({ arrays: [specJson({ shape: [2], nbytes: 24 })] });
  assert.throws(() => decodeNlf(rawNlf(header, new Uint8Array(24))), nlfError(/nbytes 与 shape/));
});

test("失败路径：dtype 未知（含协议文档提到但参考实现不支持的 f2）", () => {
  for (const dtype of ["f2", "q8", "float64", ""]) {
    const header = headerJson({ arrays: [specJson({ dtype, nbytes: 16 })] });
    assert.throws(() => decodeNlf(rawNlf(header, new Uint8Array(16))), nlfError(/dtype/), `dtype=${dtype}`);
  }
});

test("失败路径：order 非法", () => {
  const header = headerJson({ arrays: [specJson({ order: "R" })] });
  assert.throws(() => decodeNlf(rawNlf(header, new Uint8Array(16))), nlfError(/order/));
});

test("失败路径：offset+nbytes 越界", () => {
  const header = headerJson({ arrays: [specJson({ offset: 8, nbytes: 16 })] });
  assert.throws(() => decodeNlf(rawNlf(header, new Uint8Array(16))), nlfError(/越界/));
});

test("失败路径：arrays 元素缺少字段", () => {
  const missing: Record<string, unknown> = specJson();
  delete missing.nbytes;
  assert.throws(() => decodeNlf(rawNlf(headerJson({ arrays: [missing] }))), nlfError(/缺少字段/));
});

test("失败路径：meta 里塞数组、kind 非法（比 Python 更严，见交付说明）", () => {
  assert.throws(() => decodeNlf(rawNlf(headerJson({ meta: { a: [1, 2] } }))), nlfError(/meta/));
  assert.throws(() => decodeNlf(rawNlf(headerJson({ kind: "weird" }))), nlfError(/kind/));
});

test("失败路径：访问不存在的数组名抛 NlfError", () => {
  const frame = decodeNlf(readFixtureBytes("basic_f8_1d.nlf"));
  assert.throws(
    () => frame.array("nope"),
    (err: unknown) => err instanceof NlfError && /没有数组/.test(err.message) && /'u'/.test(err.message),
  );
  assert.throws(() => frame.shapeOf("nope"), nlfError(/没有数组/));
  assert.throws(() => frame.matrix("u"), nlfError(/只接受 2 维/));
});

// --------------------------------------------------------------------------- //
// 4. 往返
// --------------------------------------------------------------------------- //

function makeSpec(name: string, dtype: Dtype, shape: number[], order: "C" | "F" = "C", unit = "", label = ""): ArraySpec {
  let size = 1;
  for (const d of shape) size *= d;
  return { name, dtype, shape, order, offset: 0, nbytes: size * DTYPE_ITEMSIZE[dtype], unit, label };
}

/** 测试专用的造帧工厂（业务代码 MUST NOT 造帧，见 POLICY P8/P9）。 */
function makeFrame(
  entries: Array<{ name: string; dtype: Dtype; values: TypedArray; shape?: number[]; order?: "C" | "F" }>,
  init: Omit<FrameInit, "specs" | "arrays"> = {},
): Frame {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const e of entries) {
    specs.push(makeSpec(e.name, e.dtype, e.shape ?? [e.values.length], e.order ?? "C"));
    arrays.set(e.name, e.values);
  }
  return new Frame({ ...init, specs, arrays });
}

/** 逐值比较两个数组（类型、长度、每个元素）。 */
function assertSameArray(a: TypedArray, b: TypedArray, where: string): void {
  assert.equal(a.constructor.name, b.constructor.name, `${where}：类型不符`);
  assert.equal(a.length, b.length, `${where}：长度不符`);
  for (let i = 0; i < a.length; i++) assert.equal(b[i], a[i], `${where}[${i}] 不符`);
}

test("往返：六个夹具 encode(none) → decode 后数组与 header 完全一致", async () => {
  for (const fx of FIXTURES) {
    const golden = readGolden(fx.name);
    const frame = await loadFrame(fx.name, golden);
    const back = decodeNlf(await encodeNlf(frame, { codec: "none" }));

    assert.equal(back.codec, "none");
    assert.equal(back.kind, frame.kind);
    assert.equal(back.chapter, frame.chapter);
    assert.equal(back.viz, frame.viz);
    assert.equal(back.pitfalls, frame.pitfalls);
    assert.deepEqual(back.meta, frame.meta, `${fx.name}.meta 往返后不一致`);
    assert.deepEqual(back.strings, frame.strings);
    assert.deepEqual(back.critical, frame.critical);
    assert.deepEqual(back.names, frame.names);
    assert.equal(back.specs.length, frame.specs.length);
    for (const [i, spec] of frame.specs.entries()) {
      // order/unit/label/shape 都不能在往返中丢：order 丢了 F 序数组就会静默变转置。
      assert.equal(back.specs[i].order, spec.order, `${fx.name}.${spec.name}：order 往返丢失`);
      assert.equal(back.specs[i].unit, spec.unit);
      assert.equal(back.specs[i].label, spec.label);
      assert.deepEqual(back.specs[i].shape, spec.shape);
      assertSameArray(frame.array(spec.name), back.array(spec.name), `${fx.name}.${spec.name}`);
    }
  }
});

test("往返：encode(zlib) → decodeNlfAsync 一致，且 decodeNlf 拒绝", async () => {
  const golden = readGolden("multi_2d");
  const frame = await loadFrame("multi_2d", golden);
  const compressed = await encodeNlf(frame, { codec: "zlib" });
  assert.ok(compressed.byteLength > 0);
  assert.throws(() => decodeNlf(compressed), NlfError);

  const back = await decodeNlfAsync(compressed);
  assert.equal(back.codec, "zlib");
  assert.deepEqual(back.meta, frame.meta);
  for (const spec of frame.specs) assertSameArray(frame.array(spec.name), back.array(spec.name), spec.name);
});

test("往返：gzip（由帧自身 codec 决定）也能解回", async () => {
  const frame = makeFrame([{ name: "v", dtype: "f8", values: Float64Array.from([1.5, -2.5, 1e-13]) }], {
    chapter: "0.0",
    viz: "gzip_probe",
    codec: "gzip",
    meta: { t: 0.5 },
  });
  const bytes = await encodeNlf(frame);
  assert.throws(() => decodeNlf(bytes), NlfError);
  const back = await decodeNlfAsync(bytes);
  assert.equal(back.codec, "gzip");
  assert.deepEqual(collect(back.array("v")), [1.5, -2.5, 1e-13]);
});

test("往返：全部 dtype（含 i8 的 BigInt64Array 与 bool 的 0/1）", async () => {
  const frame = makeFrame(
    [
      { name: "f8", dtype: "f8", values: Float64Array.from([1.5, -2.25, 3e-13]) },
      { name: "f4", dtype: "f4", values: Float32Array.from([0.5, -1.5, 3.25]) },
      { name: "i8", dtype: "i8", values: BigInt64Array.from([-9223372036854775808n, 9223372036854775807n, 0n]) },
      { name: "i4", dtype: "i4", values: Int32Array.from([-2147483648, 2147483647]) },
      { name: "i2", dtype: "i2", values: Int16Array.from([-32768, 32767]) },
      { name: "u1", dtype: "u1", values: Uint8Array.from([0, 255]) },
      { name: "u2", dtype: "u2", values: Uint16Array.from([0, 65535]) },
      { name: "u4", dtype: "u4", values: Uint32Array.from([0, 4294967295]) },
      { name: "flag", dtype: "bool", values: Uint8Array.from([0, 1, 1, 0]) },
    ],
    { chapter: "0.0", viz: "dtype_probe", meta: { n: 9, ok: true, none: null } },
  );

  const back = decodeNlf(await encodeNlf(frame, { codec: "none" }));
  assert.deepEqual(back.names, frame.names);
  for (const spec of frame.specs) {
    assert.equal(back.specs.find((s) => s.name === spec.name)?.dtype, spec.dtype);
    assertSameArray(frame.array(spec.name), back.array(spec.name), spec.name);
  }
  assert.equal(back.array("i8").constructor.name, "BigInt64Array");
  assert.equal(back.array("i8")[0], -9223372036854775808n);
  assert.equal(back.scalar("ok"), 1);
  assert.equal(back.scalar("none"), undefined);
  assert.equal(back.scalar("missing", 7), 7);
});

test("往返：F 序数组写盘时按列主序排字节（否则 Python 会读成转置）", async () => {
  // shape [2,3]，A[i,j] = i + 2j → C 序 [1,2,3,4,5,6]，F 序扁平 [1,4,2,5,3,6]。
  const frame = makeFrame(
    [{ name: "AF", dtype: "f8", values: Float64Array.from([1, 2, 3, 4, 5, 6]), shape: [2, 3], order: "F" }],
    { chapter: "0.0", viz: "f_probe" },
  );
  const bytes = await encodeNlf(frame, { codec: "none" });
  const headerLen = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(4, true);
  const payload = new DataView(bytes.buffer, bytes.byteOffset + 8 + headerLen);
  const flatF: number[] = [];
  for (let i = 0; i < 6; i++) flatF.push(payload.getFloat64(i * 8, true));
  assert.deepEqual(flatF, [1, 4, 2, 5, 3, 6]);

  const back = decodeNlf(bytes);
  assert.equal(back.specs[0].order, "F");
  assert.deepEqual(collect(back.array("AF")), [1, 2, 3, 4, 5, 6]);
});

// --------------------------------------------------------------------------- //
// 5. 摘要与访问器
// --------------------------------------------------------------------------- //

test("frameSummary：给出帧头、数组清单与 meta", () => {
  const frame = decodeNlf(readFixtureBytes("basic_f8_1d.nlf"));
  const text = frameSummary(frame);
  assert.match(text, /^<Frame 0\.0\/fake_curve kind=frame codec=none arrays=2 bytes=272>/);
  assert.match(text, /· u: f8\[17\] order=C 1 u\(x\)/);
  assert.match(text, /· x: f8\[17\] order=C 1 x$/m);
  assert.match(text, /meta: /);
});

test("Frame 访问器：matrix / has / shapeOf / scalar", () => {
  const frame = decodeNlf(readFixtureBytes("multi_2d.nlf"));
  assert.equal(frame.scalar("rho"), 1.25);
  assert.equal(frame.scalar("不存在"), undefined);
  assert.equal(frame.scalar("不存在", 7), 7);
  assert.ok(frame.has("A") && !frame.has("nope"));
  assert.deepEqual(frame.shapeOf("b"), [3]);

  const m = frame.matrix("A");
  assert.equal(m.rows, 3);
  assert.equal(m.cols, 4);
  assert.equal(m.values.length, 12);
  assert.equal(m.values[0], 0);
  assert.throws(() => frame.matrix("b"), nlfError(/只接受 2 维/));
});

test("输入形态：ArrayBuffer / 带偏移的子视图 / Node Buffer / 异步 none 都能解码", async () => {
  const bytes = readFixtureBytes("basic_f8_1d.nlf");
  const golden = readGolden("basic_f8_1d");
  const expectAll = (frame: Frame, where: string): void => {
    for (const [name, expected] of Object.entries(golden.arrays)) assertArrayMatches(frame, name, expected, where);
  };

  // 1) fetch(...).arrayBuffer() 的形态。
  expectAll(decodeNlf(bytes.slice().buffer), "ArrayBuffer");

  // 2) WebSocket / TCP 分包缓冲里带偏移的视图（byteOffset ≠ 0 最容易踩对齐坑）。
  const padded = new Uint8Array(bytes.byteLength + 7);
  padded.set(bytes, 5);
  const view = padded.subarray(5, 5 + bytes.byteLength);
  assert.equal(view.byteOffset, 5);
  expectAll(decodeNlf(view), "带偏移视图");

  // 3) readFileSync 原样返回的 Buffer（Buffer 是 Uint8Array 子类，且常挂在共享池上）。
  expectAll(decodeNlf(readFileSync(new URL("basic_f8_1d.nlf", FIXTURE_DIR))), "Buffer");

  // 4) codec=none 也允许走异步入口（framesource 只有一条解码路径时不必分支）。
  expectAll(await decodeNlfAsync(bytes.slice().buffer), "异步 none");
});

test("bool 语义与 numpy 对齐：非 0 字节视为 1", () => {
  const payload = Uint8Array.from([0, 2, 255, 1]);
  const header = headerJson({ arrays: [specJson({ name: "flag", dtype: "bool", shape: [4], nbytes: 4 })] });
  const frame = decodeNlf(rawNlf(header, payload));
  assert.equal(frame.array("flag").constructor.name, "Uint8Array");
  assert.deepEqual(collect(frame.array("flag")), [0, 1, 1, 1]);
});

test("空数组（shape [0]）可以合法往返", async () => {
  const frame = makeFrame([{ name: "empty", dtype: "f8", values: new Float64Array(0), shape: [0] }], {
    chapter: "0.0",
    viz: "empty_probe",
  });
  const back = decodeNlf(await encodeNlf(frame, { codec: "none" }));
  assert.deepEqual(back.shapeOf("empty"), [0]);
  assert.equal(back.array("empty").length, 0);
});

test("encodeNlf：拒绝自相矛盾的帧（specs 与 arrays 不一致）", () => {
  const specs = [makeSpec("missing", "f8", [1])];
  const frame = new Frame({ chapter: "0.0", viz: "broken", specs, arrays: new Map() });
  assert.rejects(() => encodeNlf(frame, { codec: "none" }), NlfError);
});

// --------------------------------------------------------------------------- //
// 6. 大帧性能
// --------------------------------------------------------------------------- //

test("大帧性能：1e6 个 float64（8 MB）编码 + 解码 < 1.5 s", async () => {
  const n = 1_000_000;
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = Math.sin(i * 1e-6) * 1000 + i * 0.25;
  const frame = makeFrame([{ name: "u", dtype: "f8", values, shape: [n] }], { chapter: "0.0", viz: "bench" });

  const t0 = performance.now();
  const bytes = await encodeNlf(frame, { codec: "none" });
  const back = decodeNlf(bytes);
  const elapsed = performance.now() - t0;

  assert.ok(bytes.byteLength >= 8_000_000, `payload 应当至少 8 MB，实际 ${bytes.byteLength}`);
  const arr = back.array("u");
  assert.equal(arr.length, n);
  for (let i = 0; i < n; i += 997) assert.equal(arr[i], values[i], `u[${i}] 不符`);
  assert.equal(arr[n - 1], values[n - 1]);
  assert.ok(elapsed < 1500, `8 MB 帧编码 + 解码耗时 ${elapsed.toFixed(1)} ms，超过 1500 ms 上限`);
});

test("大帧：F 序 1000×1000 float64 重排正确", async () => {
  const rows = 1000;
  const cols = 1000;
  const values = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) values[i * cols + j] = i + j / 1000;
  const frame = makeFrame([{ name: "A", dtype: "f8", values, shape: [rows, cols], order: "F" }], {
    chapter: "0.0",
    viz: "f_bench",
  });

  const back = decodeNlf(await encodeNlf(frame, { codec: "none" }));
  assert.equal(back.specs[0].order, "F");
  const arr = back.array("A");
  assert.equal(arr.length, rows * cols);
  for (const [i, j] of [
    [0, 0],
    [0, 999],
    [999, 0],
    [999, 999],
    [123, 456],
  ]) {
    assert.equal(arr[i * cols + j], values[i * cols + j], `A[${i},${j}] 重排错误`);
  }
});
