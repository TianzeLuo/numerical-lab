/**
 * 分享链接状态与 base64url 原语的测试。
 *
 * 两条被反复钉住的承诺：
 * - **确定性**：同一份状态永远编出同一串（键排序 + 无填充 base64url），20 组随机往返逐字节比较；
 * - **绝不抛**：坏 base64 / 坏 JSON / 版本不认识 / 字段类型不对 → `null`（10 组坏输入）。
 *
 * 顺带钉住 base64 的规范性：不接受"填充位非零"的写法，否则同一份状态会有多种文本形式，
 * 确定性就名存实亡了。
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  STATE_PREFIX,
  base64Decode,
  base64Encode,
  base64UrlDecode,
  base64UrlEncode,
  buildHash,
  decodeHash,
  decodeState,
  encodeState,
  stableJson,
  utf8Decode,
  utf8Encode,
} from "../shared/src/share.js";
import type { ParamValues, ShareState } from "../shared/src/share.js";

/** 确定性 PRNG（mulberry32）：随机往返失败时可以原样复现。 */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --------------------------------------------------------------------------- //
// base64 / base64url
// --------------------------------------------------------------------------- //

test("base64url：往返一致，且不产生 + / = 三个字符", () => {
  const cases: Uint8Array[] = [
    new Uint8Array(0),
    Uint8Array.from([0]),
    Uint8Array.from([0xfb, 0xff, 0xbf]), // 会命中 - 与 _
    utf8Encode("中文 · 参数"),
    Uint8Array.from({ length: 256 }, (_, i) => i),
  ];
  for (const bytes of cases) {
    const text = base64UrlEncode(bytes);
    assert.doesNotMatch(text, /[+/=]/, "base64url 不得出现 + / =");
    assert.deepEqual(base64UrlDecode(text), bytes);
  }
  // 与标准 base64 的关系：字母表替换、去掉填充
  assert.equal(base64Encode(Uint8Array.from([0xfb, 0xff])), "+/8=");
  assert.equal(base64UrlEncode(Uint8Array.from([0xfb, 0xff])), "-_8");
  assert.deepEqual(base64Decode("+/8="), Uint8Array.from([0xfb, 0xff]));
});

test("base64url 解码：非法输入一律 null（含填充位不为零的写法）", () => {
  assert.equal(base64UrlDecode("a"), null, "长度 % 4 == 1");
  assert.equal(base64UrlDecode("****"), null, "字母表外字符");
  assert.equal(base64UrlDecode("AA==AA"), null, "填充后还有数据");
  // "Zh" 与 "Zg" 都能解出 'f'，但只有一个的填充位是零——只接受规范写法
  assert.deepEqual(base64UrlDecode("Zg"), Uint8Array.from([0x66]));
  assert.equal(base64UrlDecode("Zh"), null);
  assert.deepEqual(base64UrlDecode("Zg=="), Uint8Array.from([0x66]), "带填充也接受");
});

test("utf8Decode：非法字节返回 null（不产出 U+FFFD 假字符串）", () => {
  assert.equal(utf8Decode(Uint8Array.from([0xff, 0xfe])), null);
  assert.equal(utf8Decode(utf8Encode("好的")), "好的");
});

// --------------------------------------------------------------------------- //
// 状态编解码
// --------------------------------------------------------------------------- //

const sample: ShareState = {
  chapter: "1.2",
  viz: "lu_heatmap",
  params: { n: 24, pivot: "partial", rho: 10 },
  t: 0.3,
  playing: true,
};

test("encodeState：v1 前缀、确定性、键序无关", () => {
  const text = encodeState(sample);
  assert.ok(text.startsWith(STATE_PREFIX), `应以 ${STATE_PREFIX} 开头，实际 ${text.slice(0, 8)}`);
  assert.equal(encodeState(sample), text, "同一状态必须得到同一串");
  const reordered: ShareState = { ...sample, params: { rho: 10, pivot: "partial", n: 24 } };
  assert.equal(encodeState(reordered), text, "参数键序不影响结果");
  assert.equal(stableJson({ b: 1, a: { d: 2, c: 3 } }), '{"a":{"c":3,"d":2},"b":1}');
});

test("decodeState：encodeState 的逆（含中文、布尔、浮点）", () => {
  const state: ShareState = {
    chapter: "0.0",
    viz: "heat_ftcs",
    params: { nx: 41, r: 0.05, t_end: 0.2, label: "显式格式", use_log: false },
    t: 1e-3,
    playing: false,
  };
  assert.deepEqual(decodeState(encodeState(state)), state);
});

test("decodeState：20 组随机状态往返一致", () => {
  const random = rng(20261018);
  const chapters = ["0.0", "1.2", "3.10", "6.1"];
  const vizzes = ["oscillator_regimes", "heat_ftcs", "lu_heatmap", "stability_domain"];
  const words = ["partial", "rk4", "无", "自适应", "x"];
  for (let i = 0; i < 20; i++) {
    const params: ParamValues = {};
    const nKeys = 1 + Math.floor(random() * 4);
    for (let k = 0; k < nKeys; k++) {
      const key = `p${k}`;
      const roll = random();
      if (roll < 0.34) params[key] = Math.round(random() * 2000) / 8;
      else if (roll < 0.67) params[key] = Math.floor(random() * 100);
      else if (roll < 0.85) params[key] = random() < 0.5;
      else params[key] = `${words[Math.floor(random() * words.length)]}-${i}`;
    }
    const state: ShareState = {
      chapter: chapters[Math.floor(random() * chapters.length)],
      viz: vizzes[Math.floor(random() * vizzes.length)],
      params,
      t: random() * 20,
      playing: random() < 0.5,
    };
    const text = encodeState(state);
    assert.equal(encodeState(state), text, `第 ${i} 组必须确定性`);
    assert.deepEqual(decodeState(text), state, `第 ${i} 组往返`);
  }
});

test("decodeState：10 组坏输入全部返回 null（绝不抛）", () => {
  const good = encodeState(sample);
  const json = utf8Decode(base64UrlDecode(good.slice(STATE_PREFIX.length)) as Uint8Array) as string;
  const tamper = (mutate: (obj: Record<string, unknown>) => void): string => {
    const obj = JSON.parse(json) as Record<string, unknown>;
    mutate(obj);
    return STATE_PREFIX + base64UrlEncode(utf8Encode(JSON.stringify(obj)));
  };
  const bad: unknown[] = [
    "",
    "v1",
    "v1.",
    "v1.!!!!",
    `v1.${base64UrlEncode(utf8Encode("这不是 JSON"))}`,
    `v1.${base64UrlEncode(utf8Encode("[1,2,3]"))}`,
    `v2.${good.slice(STATE_PREFIX.length)}`, // 版本不认识
    `V1.${good.slice(STATE_PREFIX.length)}`, // 前缀大小写不符
    tamper((o) => (o["chapter"] = 12)),
    tamper((o) => (o["params"] = { nested: { a: 1 } })),
    tamper((o) => (o["t"] = "0.3")),
    tamper((o) => (o["playing"] = 1)),
    good.slice(0, good.length - 6), // 截断
  ];
  for (const text of bad) {
    assert.equal(decodeState(text), null, `应判为 null：${String(text).slice(0, 24)}`);
  }
  assert.equal(decodeState(undefined), null);
  assert.equal(decodeState(123), null);
  assert.equal(decodeState(null), null);
  assert.deepEqual(decodeState(good), sample, "对照组：原串仍然可解");
});

test("decodeState：可选字段缺省（params/t/playing 可以不给）", () => {
  const minimal = STATE_PREFIX + base64UrlEncode(utf8Encode('{"chapter":"1.2","viz":"x"}'));
  assert.deepEqual(decodeState(minimal), { chapter: "1.2", viz: "x", params: {}, t: 0, playing: false });
});

test("encodeState：非法输入立刻抛（生产端出错不该生成一条「看起来能用」的链接）", () => {
  assert.throws(() => encodeState({ ...sample, chapter: "" }), /chapter/);
  assert.throws(() => encodeState({ ...sample, viz: "" }), /viz/);
  assert.throws(() => encodeState({ ...sample, t: Number.NaN }), /有限数/);
  assert.throws(
    () => encodeState({ ...sample, params: { bad: {} as unknown as number } }),
    /params\['bad'\]/,
  );
});

// --------------------------------------------------------------------------- //
// hash 形式
// --------------------------------------------------------------------------- //

test("buildHash / decodeHash：往返，且能吃掉整条 URL", () => {
  const hash = buildHash(sample);
  assert.ok(hash.startsWith("#v1."));
  assert.deepEqual(decodeHash(hash), sample);
  assert.deepEqual(decodeHash(hash.slice(1)), sample, "裸状态串也接受");
  assert.deepEqual(decodeHash(`http://localhost:3080/viz/1.2/lu_heatmap${hash}`), sample);
  assert.deepEqual(decodeHash(`http://localhost:3080/viz/1.2/lu_heatmap?x=1${hash}`), sample);
});

test("decodeHash：空串 / 只有 # / 没有 hash 的 URL → null", () => {
  assert.equal(decodeHash(""), null);
  assert.equal(decodeHash("#"), null);
  assert.equal(decodeHash("   "), null);
  assert.equal(decodeHash("http://localhost:3080/viz/1.2"), null);
  assert.equal(decodeHash("#v1.坏东西"), null);
  assert.equal(decodeHash(42), null);
});
