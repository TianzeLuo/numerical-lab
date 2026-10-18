import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CMAP_NAMES,
  cmapTable,
  lookupCmap,
  cmapToCss,
  applyCmap,
  normalizeValue,
  luminance,
  cmapCssGradient,
  type CmapName,
} from "../shared/src/colorscale.js";

/** 各色标首末色（真源：viz/shared/colorscales.json，门禁 G6 保证两侧一致）。 */
const FIRST_LAST: Record<CmapName, [[number, number, number], [number, number, number]]> = {
  viridis: [[68, 1, 84], [253, 231, 37]],
  magma: [[0, 0, 4], [252, 253, 191]],
  coolwarm: [[59, 76, 192], [180, 4, 38]],
  gray: [[0, 0, 0], [255, 255, 255]],
  rdbu: [[103, 0, 31], [5, 48, 97]],
};

test("colorscale: 5 个色标的 LUT 长度为 768，首末色与真源一致", () => {
  assert.deepEqual([...CMAP_NAMES], ["viridis", "magma", "coolwarm", "gray", "rdbu"]);
  for (const name of CMAP_NAMES) {
    const table = cmapTable(name);
    assert.equal(table.length, 768, `${name} LUT 必须 768 = 256×3 字节`);
    assert.ok(table instanceof Uint8Array);
    for (const v of table) assert.ok(v >= 0 && v <= 255, `${name} LUT 取值必须在 0..255`);
    assert.deepEqual(lookupCmap(name, 0), FIRST_LAST[name][0], `${name} 首色不符`);
    assert.deepEqual(lookupCmap(name, 1), FIRST_LAST[name][1], `${name} 末色不符`);
  }
  const v = cmapTable("viridis");
  assert.deepEqual([v[3], v[4], v[5]], [68, 2, 86], "viridis 第 2 个颜色应是 (68,2,86)");
});

test("colorscale: gray 是恒等映射（t·255 四舍五入）", () => {
  for (let i = 0; i <= 255; i++) {
    const [r, g, b] = lookupCmap("gray", i / 255);
    assert.equal(r, i);
    assert.equal(g, i);
    assert.equal(b, i);
  }
  assert.deepEqual(lookupCmap("gray", 0.5), [128, 128, 128]); // round(127.5) = 128
  assert.deepEqual(lookupCmap("gray", 0.001), [0, 0, 0]);
});

test("colorscale: t 越界被截断到 [0,1]，NaN/±Infinity 不产生越界索引", () => {
  for (const name of CMAP_NAMES) {
    assert.deepEqual(lookupCmap(name, -1), lookupCmap(name, 0), `${name} t < 0 应截断到 0`);
    assert.deepEqual(lookupCmap(name, -1e300), lookupCmap(name, 0));
    assert.deepEqual(lookupCmap(name, 2), lookupCmap(name, 1), `${name} t > 1 应截断到 1`);
    assert.deepEqual(lookupCmap(name, 1e300), lookupCmap(name, 1));
    assert.deepEqual(lookupCmap(name, NaN), lookupCmap(name, 0), `${name} NaN 取起点色`);
    assert.deepEqual(lookupCmap(name, Infinity), lookupCmap(name, 1));
    assert.deepEqual(lookupCmap(name, -Infinity), lookupCmap(name, 0));
    assert.equal(cmapToCss(name, 0.5), `rgb(${lookupCmap(name, 0.5).join(",")})`);
  }
  assert.match(cmapToCss("viridis", 0.5), /^rgb\(\d{1,3},\d{1,3},\d{1,3}\)$/);
  assert.throws(() => cmapTable("jet" as CmapName), RangeError);
  assert.throws(() => lookupCmap("nope" as CmapName, 0.5), RangeError);
});

test("colorscale: normalizeValue 线性/对数/退化口径明确", () => {
  // 线性
  assert.equal(normalizeValue(5, 0, 10), 0.5);
  assert.equal(normalizeValue(-5, 0, 10), 0); // 越界截断
  assert.equal(normalizeValue(15, 0, 10), 1);
  assert.equal(normalizeValue(0, 0, 10), 0);
  assert.equal(normalizeValue(10, 0, 10), 1);
  assert.equal(normalizeValue(5, 10, 0), 0.5); // 反向区间也自洽
  assert.equal(normalizeValue(3, 3, 3), 0.5, "常数场映射到色标中点");
  assert.ok(Number.isNaN(normalizeValue(NaN, 0, 1)));
  assert.ok(Number.isNaN(normalizeValue(Infinity, 0, 1)));

  // 对数：t = log10(v/vmin) / log10(vmax/vmin)
  assert.equal(normalizeValue(1, 1, 100, true), 0);
  assert.equal(normalizeValue(10, 1, 100, true), 0.5);
  assert.equal(normalizeValue(100, 1, 100, true), 1);
  assert.equal(normalizeValue(1e-3, 1, 100, true), 0); // 低于下界 → 截断
  assert.equal(normalizeValue(1e6, 1, 100, true), 1);
  assert.ok(Number.isNaN(normalizeValue(0, 1, 100, true)), "对数刻度下 0 无定义");
  assert.ok(Number.isNaN(normalizeValue(-2, 1, 100, true)), "对数刻度下负数无定义");
  assert.throws(() => normalizeValue(1, 0, 100, true), RangeError);
  assert.throws(() => normalizeValue(1, -1, 100, true), RangeError);
  assert.throws(() => normalizeValue(1, 10, 10, true), RangeError);
  assert.throws(() => normalizeValue(1, 0, NaN), RangeError);
});

test("colorscale: applyCmap 输出 RGBA，NaN/±Infinity 与 clipBelow 一律透明", () => {
  const values = [NaN, Infinity, -Infinity, 0, 0.5, 1];
  const rgba = applyCmap("viridis", values, 0, 1);
  assert.equal(rgba.length, values.length * 4, "必须返回 4N 字节");
  assert.ok(rgba instanceof Uint8ClampedArray);
  // 非有限值 → 四字节全 0（完全透明）
  for (const i of [0, 1, 2]) {
    assert.deepEqual([...rgba.slice(4 * i, 4 * i + 4)], [0, 0, 0, 0], `索引 ${i} 应透明`);
  }
  // 正常值 → 不透明且等于查表结果
  for (const i of [3, 4, 5]) {
    const t = values[i];
    const [r, g, b] = lookupCmap("viridis", t);
    assert.deepEqual([...rgba.slice(4 * i, 4 * i + 4)], [r, g, b, 255]);
  }

  // clipBelow：低于阈值一律透明（以数据单位比较）
  const clipped = applyCmap("gray", [0.1, 0.49, 0.5, 0.9], 0, 1, { clipBelow: 0.5 });
  assert.deepEqual([...clipped.filter((_, i) => i % 4 === 3)], [0, 0, 255, 255]);
  assert.equal(clipped.length, 16);

  // 常数场（线性）：取色标中点
  const flat = applyCmap("gray", [7, 7], 7, 7);
  assert.deepEqual([...flat.slice(0, 4)], [128, 128, 128, 255]);

  // 越界值截断到端点色
  const out = applyCmap("gray", [-1, 2], 0, 1);
  assert.deepEqual([...out.slice(0, 4)], [0, 0, 0, 255]);
  assert.deepEqual([...out.slice(4, 8)], [255, 255, 255, 255]);

  assert.equal(applyCmap("gray", [], 0, 1).length, 0);
  assert.throws(() => applyCmap("gray", [1], 0, 1, { clipBelow: NaN }), RangeError);
});

test("colorscale: applyCmap 对数刻度——vmin ≤ 0 或 vmin = vmax 抛错，非正数据透明", () => {
  assert.throws(() => applyCmap("viridis", [1], 0, 10, { logScale: true }), RangeError);
  assert.throws(() => applyCmap("viridis", [1], -1, 10, { logScale: true }), RangeError);
  assert.throws(() => applyCmap("viridis", [1], 1, 0, { logScale: true }), RangeError);
  assert.throws(() => applyCmap("viridis", [1], 5, 5, { logScale: true }), RangeError);

  const rgba = applyCmap("gray", [0, -1, 1, 10, 100], 1, 100, { logScale: true });
  assert.equal(rgba[3], 0, "v = 0 在对数刻度下必须透明");
  assert.equal(rgba[7], 0, "v < 0 在对数刻度下必须透明");
  assert.deepEqual([...rgba.slice(0, 4)], [0, 0, 0, 0]);
  assert.deepEqual([...rgba.slice(4, 8)], [0, 0, 0, 0]);
  // v = vmin → t = 0 → 色标起点（gray = 0），不透明
  assert.deepEqual([...rgba.slice(8, 12)], [0, 0, 0, 255]);
  // v = 10 → t = log10(10/1)/log10(100) = 0.5 → gray = 128
  assert.deepEqual([...rgba.slice(12, 16)], [128, 128, 128, 255]);
  // v = vmax → t = 1 → 色标终点
  assert.deepEqual([...rgba.slice(16, 20)], [255, 255, 255, 255]);
});

test("colorscale: luminance 为 Rec.709 相对亮度（反伽马线性化）", () => {
  assert.equal(luminance(0, 0, 0), 0);
  assert.ok(Math.abs(luminance(255, 255, 255) - 1) < 1e-15);
  for (const name of CMAP_NAMES) {
    for (let i = 0; i <= 16; i++) {
      const y = luminance(...lookupCmap(name, i / 16));
      assert.ok(y >= 0 && y <= 1, `${name} 亮度 ${y} 越界`);
    }
  }
  // 权重大小关系（Rec.709：绿 > 红 > 蓝）
  assert.ok(luminance(0, 255, 0) > luminance(255, 0, 0));
  assert.ok(luminance(255, 0, 0) > luminance(0, 0, 255));
  assert.ok(Number.isNaN(luminance(NaN, 0, 0)));
  // 越界通道先截断
  assert.equal(luminance(300, 300, 300), luminance(255, 255, 255));
});

test("colorscale: viridis 亮度单调不减（8 位 LUT 的量化伪影必须逐条可解释）", () => {
  const ys: number[] = [];
  for (let i = 0; i <= 255; i++) ys.push(luminance(...lookupCmap("viridis", i / 255)));
  let artifacts = 0;
  for (let i = 0; i + 1 <= 255; i++) {
    const d = ys[i + 1] - ys[i];
    if (d >= -1e-9) continue;
    // 唯一的允许来源：色标被量化成 8 位后，恰好一个通道降了 1 LSB（其余通道不变）
    const a = lookupCmap("viridis", i / 255);
    const b = lookupCmap("viridis", (i + 1) / 255);
    const changed = [b[0] - a[0], b[1] - a[1], b[2] - a[2]].filter((x) => x !== 0);
    assert.equal(changed.length, 1, `i=${i} 的亮度下降不是单通道量化伪影：${a} → ${b}`);
    assert.equal(changed[0], -1, `i=${i} 的通道跳变应为 −1 LSB：${a} → ${b}`);
    artifacts++;
  }
  // 实测 2 处（i=95 红通道 −1、i=146 蓝通道 −1）；给一点余量但不允许"到处都在降"
  assert.ok(artifacts <= 3, `量化伪影过多：${artifacts}`);
  assert.ok(ys[255] > ys[0] + 0.5, "整体必须显著变亮");
  // 单调段占绝对多数
  let mono = 0;
  for (let i = 0; i + 1 <= 255; i++) if (ys[i + 1] >= ys[i] - 1e-9) mono++;
  assert.ok(mono >= 253, `单调步数 ${mono}/255 过少`);
});

test("colorscale: cmapCssGradient 给出可用于图例的 CSS 渐变", () => {
  const g = cmapCssGradient("viridis");
  assert.match(g, /^linear-gradient\(to right, rgb\(\d+,\d+,\d+\) 0%, .* rgb\(\d+,\d+,\d+\) 100%\)$/);
  assert.ok(g.includes("0%") && g.includes("100%"));
  assert.equal(g.split("rgb(").length - 1, 16, "缺省 16 个采样点");
  const g3 = cmapCssGradient("magma", 3);
  assert.equal(g3.split("rgb(").length - 1, 3);
  assert.ok(g3.includes("50%"));
  const g2 = cmapCssGradient("gray", 1); // < 2 时按 2 处理
  assert.equal(g2.split("rgb(").length - 1, 2);
  assert.ok(g2.includes("rgb(0,0,0) 0%") && g2.includes("rgb(255,255,255) 100%"));
  assert.throws(() => cmapCssGradient("bad" as CmapName), RangeError);
});
