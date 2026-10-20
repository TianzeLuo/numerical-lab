import assert from "node:assert/strict";
import { test } from "node:test";

import { keyMetrics } from "../shared/src/metrics.js";

test("keyMetrics：1.1 的关键读数按可视化定制，不出现内部字段", () => {
  const chips = keyMetrics(
    {
      n_points: 21,
      residual_0102_03_ulps: 1,
      residual_0102_03: 5.551115123125783e-17,
      k: 64,
      center_log10: -1,
      overflow: false,
    },
    "1.1",
    "float_zoom",
  );
  assert.deepEqual(chips.map((chip) => chip.label), [
    "窗口内浮点数",
    "0.1+0.2 与 0.3 相差",
    "0.1+0.2 的表示误差",
    "累加次数",
  ]);
  assert.equal(chips[0]?.value, "21");
  assert.equal(chips[1]?.value, "1");
  assert.equal(chips[1]?.unit, "ulp");
  assert.equal(chips[3]?.value, "64");
});

test("keyMetrics：同一个 k 在不同实验里有不同含义", () => {
  const sum = keyMetrics({ k: 512, error_naive: 511, error_kahan: 1, bound_naive: 568.4 }, "1.1", "summation_order");
  assert.equal(sum[0]?.label, "当前项数");
  assert.equal(sum[0]?.value, "512");
  assert.equal(sum[1]?.value, "511");
});

test("keyMetrics：通用回退只挑含义明确的量，跳过 t 与内部索引", () => {
  const chips = keyMetrics(
    { t: 3.2, residual: 1e-9, max_error: 0.00042, family_key: 2, azimuth: 30 },
    "9.9",
    "unknown",
  );
  assert.deepEqual(chips.map((chip) => chip.label), ["最大误差", "残差"]);
  assert.equal(chips[0]?.value, "4.2e-4");
  assert.ok(!chips.some((chip) => chip.label.includes("family_key")));
});

test("keyMetrics：没有可展示的量时返回空数组，limit 生效", () => {
  assert.deepEqual(keyMetrics({ t: 1, azimuth: 30 }, "1.1", "float_zoom"), []);
  assert.equal(keyMetrics({ residual: 1, max_error: 2, l2_error: 3, forward_error: 4 }, "x", "y", 2).length, 2);
});
