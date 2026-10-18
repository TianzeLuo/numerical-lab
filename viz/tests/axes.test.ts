import { test } from "node:test";
import assert from "node:assert/strict";

import {
  linspace,
  logspace,
  niceTicks,
  niceLimits,
  tickLabel,
  formatNumber,
  formatSci,
  mapToPixel,
  mapToData,
  nextafter_log,
} from "../shared/src/axes.js";

/**
 * 属性测试（不是"跑一遍看看"）：所有断言都对**随机区间 + 固定边界用例**批量成立。
 * 随机数用自带 mulberry32，种子固定 → 失败可复现；不引入任何第三方依赖。
 */
function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 固定边界区间（含跨零、负区间、极窄、极宽）。 */
const FIXED_LINEAR: Array<[number, number, number]> = [
  [0, 1, 7],
  [-1, 1, 7],
  [0, 10, 7],
  [-3.2, 7.8, 7],
  [123, 456, 7],
  [0.37, 0.38, 7],
  [1e-4, 3, 7],
  [0.1 + 0.2, 0.7, 7],
  [-1e6, -1, 6],
  [0, 1e5, 8],
  [1e12, 1e12 + 1, 7],
];

/** 随机线性区间（数量级/偏移/目标刻度数都随机）。 */
function randomLinearRanges(n: number, seed: number): Array<[number, number, number]> {
  const rng = makeRng(seed);
  const out: Array<[number, number, number]> = [];
  for (let k = 0; k < n; k++) {
    const e1 = Math.floor(rng() * 15) - 7;
    const e2 = Math.floor(rng() * 15) - 7;
    const a = (rng() * 9 + 1) * Math.pow(10, e1);
    const b = (rng() * 9 + 1) * Math.pow(10, e2);
    const target = [3, 5, 6, 7, 8, 10][Math.floor(rng() * 6)];
    out.push([Math.min(a, b), Math.max(a, b), target]);
  }
  return out;
}

/** 随机对数区间（跨度 0.5~12 个十倍程）。 */
function randomLogRanges(n: number, seed: number): Array<[number, number, number]> {
  const rng = makeRng(seed);
  const out: Array<[number, number, number]> = [];
  for (let k = 0; k < n; k++) {
    const lo = Math.pow(10, -6 + rng() * 8);
    const hi = lo * Math.pow(10, 0.5 + rng() * 12);
    const target = [6, 7, 8][Math.floor(rng() * 3)];
    out.push([lo, hi, target]);
  }
  return out;
}

/**
 * 步长"好看度"检查：把 `d` 写成 `m × 10^k` 后 `m` 应落在 {1,2,5} 内。
 *
 * 容差有两项来源，都必须说清楚（POLICY §3：容差要能讲清来源）：
 * 1. `1e-9` —— 十进制/二进制换算与端点钳位的常规余量；
 * 2. `(8u + 1e-12)·max|端点| / d` —— "大偏移 + 窄跨度"时的 ulp 粒度，
 *    以及实现文档中写明的 1e-12 相对端点钳位。
 */
function stepLooksNice(d: number, maxAbs: number): boolean {
  const frac = (8 * Number.EPSILON + 1e-12) * maxAbs / d;
  const l10 = Math.log10(d);
  // floor 与 round 都要试：`0.5` 这类"刚低于整十次幂"的值需要 floor，
  // 被端点钳位抬高了一丝的 `1e7` 需要 round
  for (const e of [Math.floor(l10), Math.round(l10)]) {
    const m = d / Math.pow(10, e);
    if ([1, 2, 5].some((x) => Math.abs(m - x) <= x * frac + 1e-9)) return true;
  }
  return false;
}

/** 区间内 1×/2×/5×10^k 格点的个数（测试侧独立实现，用来判定"端点退化"是否确有必要）。 */
function niceLatticeCount(lo: number, hi: number): number {
  let n = 0;
  for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)) + 1; e++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= lo && v <= hi) n++;
    }
  }
  return n;
}

/** 测试侧独立实现的"1-2-5 格点刻度"（纯算术，无端点钳位）。 */
function latticeTicks(lo: number, hi: number, step: number): number[] | null {
  const i0 = Math.ceil(lo / step);
  const i1 = Math.floor(hi / step);
  // 索引超出 2^53 时 i++ 不再推进（会假死循环），极端量级直接放弃该候选
  if (!Number.isFinite(i0) || !Number.isFinite(i1) || Math.abs(i0) > 1e15) return null;
  const n = i1 - i0 + 1;
  if (!(n >= 0) || n > 2e5) return null;
  const out: number[] = [];
  for (let i = i0; i <= i1; i++) {
    const v = i * step;
    if (v >= lo && v <= hi) out.push(v);
  }
  return out;
}

/** 在 1-2-5 格点上能达到的最优覆盖率（用来证明实现没有"随便挑个步长"）。 */
function bestLatticeCoverage(lo: number, hi: number, target: number): number {
  const span = hi - lo;
  const minCount = Math.max(2, Math.ceil(target / 2));
  const maxCount = 2 * target;
  const base = Math.floor(Math.log10(span));
  let best = 0;
  for (let e = base - 4; e <= base + 4; e++) {
    for (const m of [1, 2, 5]) {
      const ticks = latticeTicks(lo, hi, m * Math.pow(10, e));
      if (ticks === null || ticks.length < minCount || ticks.length > maxCount) continue;
      best = Math.max(best, (ticks[ticks.length - 1] - ticks[0]) / span);
    }
  }
  return best;
}

/** 覆盖率 =（末刻度 − 首刻度）/ 区间长度。 */
function coverageOf(ticks: number[], lo: number, hi: number): number {
  if (ticks.length < 2) return 0;
  return (ticks[ticks.length - 1] - ticks[0]) / (hi - lo);
}

/** 区间内整十次幂的个数（测试侧独立实现）。 */
function decadeCount(lo: number, hi: number): number {
  const eLo = Math.ceil(Math.log10(lo) - 1e-12);
  const eHi = Math.floor(Math.log10(hi) + 1e-12);
  return eHi - eLo + 1;
}

/* ------------------------------------------------------------------ *
 * 序列
 * ------------------------------------------------------------------ */

test("axes: linspace/logspace 端点精确、退化点数明确", () => {
  const xs = linspace(0, 1, 11);
  assert.equal(xs.length, 11);
  assert.equal(xs[0], 0);
  assert.equal(xs[10], 1);
  for (let i = 0; i < 11; i++) assert.ok(Math.abs(xs[i] - i / 10) < 1e-15);

  assert.deepEqual(Array.from(linspace(1, 0, 3)), [1, 0.5, 0]); // 允许递减
  assert.deepEqual(Array.from(linspace(3, 3, 4)), [3, 3, 3, 3]); // 退化区间
  assert.deepEqual(Array.from(linspace(0, 1, 1)), [0]); // n = 1 → [a]
  assert.equal(linspace(0, 1, 0).length, 0); // n = 0 → 空

  // logspace 的参数是十进制指数（与 numpy.logspace 同约定）
  const ls = logspace(-2, 2, 5);
  assert.deepEqual(Array.from(ls), [0.01, 0.1, 1, 10, 100]);
  assert.deepEqual(Array.from(logspace(-3, -3, 1)), [0.001]);
});

/* ------------------------------------------------------------------ *
 * 线性刻度
 * ------------------------------------------------------------------ */

test("axes: 线性刻度为 1-2-5 步长、严格落在区间内、数量在 [target/2, 2·target]", () => {
  const cases = [...FIXED_LINEAR, ...randomLinearRanges(400, 0xa11ce)];
  for (const [lo, hi, target] of cases) {
    const ticks = niceTicks(lo, hi, target);
    const tag = `niceTicks(${lo}, ${hi}, ${target})`;
    assert.ok(ticks.length >= 2, `${tag} 至少要有 2 个刻度`);
    for (const v of ticks) {
      assert.ok(v >= lo && v <= hi, `${tag} 刻度 ${v} 越界（规则 1：不许靠裁剪掩盖越界）`);
    }
    for (let i = 0; i + 1 < ticks.length; i++) {
      assert.ok(ticks[i] < ticks[i + 1], `${tag} 刻度必须严格递增`);
    }
    const diffs: number[] = [];
    for (let i = 0; i + 1 < ticks.length; i++) diffs.push(ticks[i + 1] - ticks[i]);
    const maxAbs = Math.max(Math.abs(lo), Math.abs(hi));
    const frac = (8 * Number.EPSILON + 1e-12) * maxAbs / diffs[0];
    assert.ok(
      stepLooksNice(diffs[0], maxAbs),
      `${tag} 步长 ${diffs[0]} 不属于 1-2-5（相对容差 ${frac}）`,
    );
    // 等距（同一容差口径）
    for (const d of diffs) {
      assert.ok(Math.abs(d / diffs[0] - 1) <= frac + 1e-9, `${tag} 步长不等距：${diffs[0]} vs ${d}`);
    }
    const minCount = Math.max(2, Math.ceil(target / 2));
    assert.ok(
      ticks.length >= minCount && ticks.length <= 2 * target,
      `${tag} 刻度数 ${ticks.length} 不在 [${minCount}, ${2 * target}]`,
    );
  }
});

test("axes: 线性刻度覆盖区间（niceLimits 管线压住两端；原始区间不劣于 1-2-5 格点最优）", () => {
  const cases = [...FIXED_LINEAR, ...randomLinearRanges(400, 0xbeef1)];
  for (const [lo, hi, target] of cases) {
    // 管线：先取整轴范围，再取刻度 → 首末刻度必须压住轴端（覆盖 100%）
    const [l, h] = niceLimits(lo, hi, target);
    assert.ok(l <= lo && h >= hi, `niceLimits(${lo}, ${hi}) = [${l}, ${h}] 未包含输入区间`);
    const niced = niceTicks(l, h, target);
    const covNiced = coverageOf(niced, l, h);
    assert.ok(covNiced >= 1 - 1e-9, `niceLimits 管线覆盖不足：${covNiced}`);

    // 原始区间：覆盖率必须达到"1-2-5 格点 + ≤2·target 个刻度"下的最优（或直接 ≥ 90%）。
    // 宽跨度区间在这个格点约束下本就无法覆盖 90%（步长只能取 1-2-5×10^k），
    // 以格点最优为下界才是可实现且非平凡的断言。
    const ticks = niceTicks(lo, hi, target);
    const cov = coverageOf(ticks, lo, hi);
    const best = bestLatticeCoverage(lo, hi, target);
    assert.ok(
      cov >= Math.min(0.9, best) - 1e-9,
      `niceTicks(${lo}, ${hi}, ${target}) 覆盖 ${cov}，格点最优 ${best}`,
    );
  }
});

/* ------------------------------------------------------------------ *
 * 对数刻度
 * ------------------------------------------------------------------ */

test("axes: 对数刻度为 1×/2×/5×10^k，十倍程足够时比值恒为 10 的幂", () => {
  const cases = [...randomLogRanges(400, 0x10910), [1e-3, 1e3, 7] as [number, number, number]];
  for (const [lo, hi, target] of cases) {
    const ticks = niceTicks(lo, hi, target, { log: true });
    const tag = `niceTicks(${lo}, ${hi}, ${target}, log)`;
    assert.ok(ticks.length >= 2, `${tag} 至少要有 2 个刻度`);
    const maxAbs = Math.max(lo, hi);
    // 极窄区间（1-2-5 格点在区间内不足 2 个点）允许退化为端点刻度——必须给出证明
    const endpointFallback = ticks.length === 2 && ticks[0] === lo && ticks[1] === hi;
    if (endpointFallback) {
      assert.ok(
        niceLatticeCount(lo, hi) < 2,
        `${tag} 只有在 1-2-5 格点不足 2 个刻度时才允许退化为端点`,
      );
    } else {
      for (const v of ticks) {
        assert.ok(v >= lo && v <= hi, `${tag} 刻度 ${v} 越界`);
        assert.ok(stepLooksNice(v, maxAbs), `${tag} 刻度 ${v} 不是 1×10^k / 2×10^k / 5×10^k`);
      }
    }
    for (let i = 0; i + 1 < ticks.length; i++) {
      assert.ok(ticks[i] < ticks[i + 1], `${tag} 必须严格递增`);
    }
    assert.ok(ticks.length <= 2 * target, `${tag} 刻度数 ${ticks.length} 超过 2·target`);

    // 十倍程足够时（主刻度走整十次幂分支）相邻比值恒为 10 的幂
    if (decadeCount(lo, hi) >= Math.max(2, Math.ceil(target / 2))) {
      for (let i = 0; i + 1 < ticks.length; i++) {
        const ratio = Math.log10(ticks[i + 1] / ticks[i]);
        assert.ok(
          Math.abs(ratio - Math.round(ratio)) < 1e-9,
          `${tag} 相邻比值 ${ticks[i + 1] / ticks[i]} 不是 10 的幂`,
        );
      }
      // 十倍程跨度极大时主刻度按 10^stride 取样：首末刻度各在端点的一个 stride 内
      const stride = Math.max(1, Math.ceil((decadeCount(lo, hi) - 1) / Math.max(1, target - 1)));
      assert.ok(
        ticks[0] / lo < Math.pow(10, stride) && hi / ticks[ticks.length - 1] < Math.pow(10, stride),
        `${tag} 首末刻度离端点超过 10^${stride}`,
      );
    }
  }
});

test("axes: 对数刻度经 niceLimits 后覆盖 100% 且包含输入区间", () => {
  const cases = [...randomLogRanges(400, 0xbeef2), [1.1e-3, 1.1, 7] as [number, number, number]];
  for (const [lo, hi, target] of cases) {
    const [l, h] = niceLimits(lo, hi, target, { log: true });
    assert.ok(l <= lo && h >= hi, `niceLimits(${lo}, ${hi}, log) = [${l}, ${h}] 未包含输入区间`);
    const ticks = niceTicks(l, h, target, { log: true });
    const cov = (Math.log10(ticks[ticks.length - 1]) - Math.log10(ticks[0])) / (Math.log10(h) - Math.log10(l));
    assert.ok(cov >= 1 - 1e-9, `对数轴覆盖 ${cov}（应压住两端）`);
    assert.ok(ticks.length <= 2 * target, `对数轴刻度数 ${ticks.length} 超过 2·target`);
  }
  // 整十次幂对齐的例子：1e-3 … 1e3 恰好七个刻度
  assert.deepEqual(niceTicks(1e-3, 1e3, 7, { log: true }), [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]);
});

/* ------------------------------------------------------------------ *
 * 数据 ↔ 像素
 * ------------------------------------------------------------------ */

test("axes: mapToPixel 与 mapToData 互为逆（200 组随机，含反向像素轴与对数轴，容差 1e-12）", () => {
  const rng = makeRng(0x5eed);
  for (let k = 0; k < 200; k++) {
    const log = k % 2 === 1;
    const p0 = rng() * 800;
    const p1 = rng() < 0.5 ? p0 - rng() * 600 : p0 + rng() * 600; // 反向像素轴也要成立
    let lo: number;
    let hi: number;
    let v: number;
    if (log) {
      lo = Math.pow(10, -6 + rng() * 5);
      hi = lo * Math.pow(10, 0.5 + rng() * 8);
      v = lo * Math.pow(hi / lo, rng()); // [lo, hi] 内对数均匀取样
    } else {
      lo = (rng() * 2 - 1) * Math.pow(10, Math.floor(rng() * 12) - 6);
      hi = lo + rng() * Math.pow(10, Math.floor(rng() * 12) - 6) + 1e-9;
      v = lo + rng() * (hi - lo);
    }
    const scale = log ? "log" : "linear";
    const tag = `k=${k} scale=${scale} v=${v} [${lo}, ${hi}] px[${p0}, ${p1}]`;
    // 往返误差的诚实口径：1e-12 相对 + 浮点放大项。
    // 线性轴"窄跨度 + 大偏移"时条件数 κ = max|端点|/跨度 会放大往返误差，
    // 放大项 4·u·κ 正是这类病态下的下界（不是给实现放水，而是问题本身的条件数）。
    const kappa = log ? 1 : Math.max(Math.abs(lo), Math.abs(hi)) / (hi - lo);
    const pxSpan = Math.abs(p1 - p0);

    const p = mapToPixel(v, lo, hi, p0, p1, scale);
    const back = mapToData(p, lo, hi, p0, p1, scale);
    const dataTol = 1e-12 * (log ? Math.abs(v) : hi - lo) + 4 * Number.EPSILON * Math.max(Math.abs(lo), Math.abs(hi), Math.abs(v));
    assert.ok(
      Math.abs(back - v) <= dataTol,
      `${tag} mapToData∘mapToPixel 往返误差 ${Math.abs(back - v)} > ${dataTol}`,
    );

    // 反方向：像素 → 数据 → 像素
    const pmid = p1 * 0.5 + p0 * 0.5;
    const vv = mapToData(pmid, lo, hi, p0, p1, scale);
    const pp = mapToPixel(vv, lo, hi, p0, p1, scale);
    const pxTol = (1e-12 + 4 * Number.EPSILON * kappa) * pxSpan;
    assert.ok(
      Math.abs(pp - pmid) <= pxTol,
      `${tag} mapToPixel∘mapToData 往返误差 ${Math.abs(pp - pmid)} > ${pxTol}（κ=${kappa}）`,
    );
  }
  // 端点必须精确落到像素端
  assert.equal(mapToPixel(0, 0, 1, 10, 610), 10);
  assert.equal(mapToPixel(1, 0, 1, 10, 610), 610);
  assert.equal(mapToPixel(1e3, 1e-3, 1e3, 0, 700, "log"), 700);
});

/* ------------------------------------------------------------------ *
 * 格式化
 * ------------------------------------------------------------------ */

test("axes: formatSci 使用上标、formatNumber 在 1e-3..1e5 用定点", () => {
  assert.equal(formatSci(1.2e-7, 2), "1.2×10⁻⁷"); // 契约里的例子
  assert.equal(formatSci(1.2e-7), "1.2×10⁻⁷");
  assert.equal(formatSci(-3.4e8, 2), "-3.4×10⁸");
  assert.equal(formatSci(9.99e-7, 2), "1×10⁻⁶"); // 尾数进位到 10 要归一
  assert.equal(formatSci(0), "0");
  assert.equal(formatSci(NaN), "NaN");
  assert.equal(formatSci(Infinity), "∞");

  const superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻";
  for (const v of [1.2e-7, 5e-3, 6.02e23, -1e-300, 9.87e-15]) {
    const s = formatSci(v);
    assert.ok(
      [...s].some((c) => superscripts.includes(c)),
      `formatSci(${v}) = ${s} 不含上标`,
    );
    assert.ok(s.includes("×10"), `formatSci(${v}) = ${s} 不是 m×10^k 形式`);
  }

  assert.equal(formatNumber(0), "0");
  assert.equal(formatNumber(0.001), "0.001");
  assert.equal(formatNumber(1e5), "100000");
  assert.equal(formatNumber(-0.5), "-0.5");
  assert.equal(formatNumber(123.456), "123.456");
  assert.equal(formatNumber(1.5e-4), "1.5e-4"); // 小于 1e-3 → e 记法
  assert.equal(formatNumber(123456.7), "1.23457e5"); // 大于 1e5 → e 记法

  assert.equal(tickLabel(1e-7, { log: true }), "10⁻⁷");
  assert.equal(tickLabel(1, { log: true }), "1");
  assert.equal(tickLabel(10, { log: true }), "10");
  assert.equal(tickLabel(100, { log: true }), "10²");
  assert.equal(tickLabel(5e-3, { log: true }), "5×10⁻³");
  assert.equal(tickLabel(123.456), "123.456");
});

/* ------------------------------------------------------------------ *
 * niceLimits 与退化 / 非法输入
 * ------------------------------------------------------------------ */

test("axes: niceLimits 恒包含输入区间，pad 只向外扩", () => {
  const cases = [...FIXED_LINEAR, ...randomLinearRanges(200, 0xc0ffee)];
  for (const [lo, hi, target] of cases) {
    const [l, h] = niceLimits(lo, hi, target);
    assert.ok(l <= lo && h >= hi, `niceLimits(${lo}, ${hi}) = [${l}, ${h}] 未包含输入区间`);
    assert.ok(h > l, `niceLimits(${lo}, ${hi}) 退化成了零长区间`);
    // 轴端必须是"好看的数"：能落在同一 1-2-5 步长格点上
    const padded = niceLimits(lo, hi, target, { pad: 0.1 });
    const span = hi - lo;
    assert.ok(
      padded[0] <= lo - 0.1 * span + 1e-12 && padded[1] >= hi + 0.1 * span - 1e-12,
      `pad=0.1 未向外扩：${padded}`,
    );
  }
});

test("axes: 退化与非法输入有明确行为（抛错或退化，均写进 docstring）", () => {
  // 退化区间
  assert.deepEqual(niceTicks(5, 5), [5]);
  assert.deepEqual(niceTicks(5, 5, 7, { log: true }), [5]);
  const dl = niceLimits(5, 5);
  assert.ok(dl[0] < 5 && dl[1] > 5, "退化线性区间应向两侧扩开");
  const dl0 = niceLimits(0, 0);
  assert.ok(dl0[0] < 0 && dl0[1] > 0, "退化到 0 的区间也应扩开");
  assert.deepEqual(niceLimits(1, 1, 7, { log: true }), [0.1, 10]);
  assert.equal(mapToPixel(3, 5, 5, 0, 100), 50); // 退化轴 → 数据画在中点
  assert.equal(mapToData(50, 5, 5, 0, 100), 5);

  // 非法输入
  assert.throws(() => niceTicks(1, 0), RangeError);
  assert.throws(() => niceLimits(1, 0), RangeError);
  assert.throws(() => niceTicks(0, 10, 7, { log: true }), RangeError);
  assert.throws(() => niceTicks(-5, 10, 7, { log: true }), RangeError);
  assert.throws(() => niceLimits(-1, 1, 7, { log: true }), RangeError);
  assert.throws(() => niceTicks(0, 1, 1), RangeError); // target < 2
  assert.throws(() => niceTicks(0, NaN), RangeError);
  assert.throws(() => niceLimits(0, 1, 7, { pad: -0.1 }), RangeError);
  assert.throws(() => mapToPixel(1, 0, 10, 5, 5), RangeError); // 零长像素轴
  assert.throws(() => mapToData(1, 0, 10, 5, 5), RangeError);
  assert.throws(() => mapToPixel(1, -1, 10, 0, 100, "log"), RangeError);
  assert.throws(() => mapToData(1, 0, 10, 0, 100, "log"), RangeError);
  // 对数轴上的非正数据：返回 NaN 交给渲染层跳过，不抛错
  assert.ok(Number.isNaN(mapToPixel(0, 1, 100, 0, 100, "log")));
  assert.ok(Number.isNaN(mapToPixel(-3, 1, 100, 0, 100, "log")));
  assert.ok(Number.isNaN(mapToPixel(NaN, 1, 100, 0, 100, "log")));
});

test("axes: nextafter_log 给出可安全取对数的下一个正数", () => {
  const m = nextafter_log(0);
  assert.ok(m > 0, "0 之后必须是正数");
  assert.ok(Number.isFinite(Math.log10(m)), "log10(下一个正数) 必须有限（不产生 -Infinity）");
  assert.ok(Math.log10(m) < -300, "应贴近最小正数");

  assert.equal(nextafter_log(-1), m);
  assert.equal(nextafter_log(Number.NEGATIVE_INFINITY), m);
  assert.ok(Number.isNaN(nextafter_log(NaN)));
  assert.equal(nextafter_log(Infinity), Infinity);

  for (const v of [1, 2.5, 1e-300, 1e300]) {
    const n = nextafter_log(v);
    assert.ok(n > v, `nextafter_log(${v}) = ${n} 必须严格更大`);
    assert.ok((n - v) / v <= Number.EPSILON, `nextafter_log(${v}) 跳得太远`);
  }
  // 次正规数处的相对间距大于 EPSILON（只能要求严格递增）
  assert.ok(nextafter_log(Number.MIN_VALUE) > Number.MIN_VALUE);
  assert.equal(nextafter_log(1), 1 + Number.EPSILON);
});
