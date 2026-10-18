/**
 * 参数控件描述的测试（纯函数部分；DOM 部分在无浏览器环境里不测）。
 *
 * 这里的断言都对着"防误导"义务：临界值必须被标出来（规则 6）、自适应坐标轴必须提示
 * （规则 1/4）、`clampParam` 必须挡住越界与类型错误（UI 的最后一道防线）。
 * 数据用真实 manifest（`viz/chapters/0.0/manifest.gen.ts`），免得测试自己编一份"理想 manifest"。
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { manifest } from "../chapters/0.0/manifest.gen.js";
import type { ParamSpecJson, VizSpecJson } from "../shared/src/manifest.js";
import { BEYOND_TEXT, autoscaleNote, clampParam, describeControls, isBeyond } from "../shared/src/controls.js";

const oscillator = manifest.vizzes.find((v) => v.id === "oscillator_regimes") as VizSpecJson;
const heat = manifest.vizzes.find((v) => v.id === "heat_ftcs") as VizSpecJson;

/** 造一个只关心被测字段的 ParamSpecJson。 */
function param(over: Partial<ParamSpecJson> & Pick<ParamSpecJson, "name" | "type">): ParamSpecJson {
  return { label: over.name, default: 0, critical: [], ...over } as ParamSpecJson;
}

// --------------------------------------------------------------------------- //
// describeControls：基本结构
// --------------------------------------------------------------------------- //

test("describeControls：顺序跟着 manifest，值取默认值，字段齐全", () => {
  const specs = describeControls(heat);
  assert.deepEqual(specs.map((s) => s.name), ["nx", "r", "t_end"]);
  const r = specs[1];
  assert.equal(r.label, "网格比 r = αΔt/Δx²");
  assert.equal(r.type, "float");
  assert.equal(r.value, 0.4);
  assert.equal(r.defaultValue, 0.4);
  assert.equal(r.min, 0.05);
  assert.equal(r.max, 0.7);
  assert.equal(r.step, 0.01);
  assert.equal(r.unit, "");
  assert.deepEqual(r.options, []);
  assert.deepEqual(r.critical, [0.5]);
  assert.match(r.criticalNote, /FTCS 的放大因子/);
  assert.ok(r.hints.length > 0, "hints 至少要有临界值说明");
});

test("describeControls：int 参数按整数给值，超范围/NaN/类型错都夹回合法域", () => {
  const specs = describeControls(heat, { nx: 1000, r: Number.NaN, t_end: "0.1" });
  assert.equal(specs[0].value, 201, "nx 上界 201");
  assert.equal(specs[1].value, 0.4, "NaN 回落到默认值");
  assert.equal(specs[2].value, 0.1, "数字字符串按数字处理");

  const low = describeControls(heat, { nx: 0, r: -99 });
  assert.equal(low[0].value, 11);
  assert.equal(low[1].value, 0.05);

  const weird = describeControls(heat, { nx: 41.6, r: "abc" });
  assert.equal(weird[0].value, 42, "int 先取整再夹取");
  assert.equal(weird[1].value, 0.4);
});

// --------------------------------------------------------------------------- //
// 临界值标记（规则 6）
// --------------------------------------------------------------------------- //

test("临界值标记：fraction 落在 [min,max] 上，governing 与 crossed 口径正确", () => {
  const [spec] = describeControls(
    { ...heat, params: [heat.params[1]] },
    { r: 0.4 },
  );
  assert.equal(spec.critical.length, 1);
  const mark = spec.criticalMarks[0];
  assert.equal(mark.value, 0.5);
  assert.equal(mark.label, "0.5");
  assert.equal(mark.governing, true, "只有一个临界值，它就是支配值");
  assert.equal(mark.crossed, false, "0.4 < 0.5");
  assert.ok(mark.fraction !== null);
  assert.ok(Math.abs(mark.fraction - (0.5 - 0.05) / (0.7 - 0.05)) < 1e-12);
  assert.equal(spec.governing, 0.5);
  assert.equal(spec.crossed, false);
});

test("临界值标记：越界时 crossed=true 且 hints 里出现警告", () => {
  const specs = describeControls(heat, { r: 0.6 });
  const r = specs[1];
  assert.equal(r.crossed, true);
  assert.equal(r.criticalMarks[0].crossed, true);
  assert.ok(r.hints.some((h) => h.includes("已越过")), r.hints.join(" / "));
  assert.equal(isBeyond("above", 0.6, 0.5), true);
  assert.equal(isBeyond("below", 0.6, 0.5), false);
  assert.equal(isBeyond("magnitude", -0.6, 0.5), true);
});

test("临界值标记：beyond=below 时方向相反，取已被突破的最大阈值", () => {
  const spec = param({
    name: "samples_per_wavelength",
    type: "int",
    default: 1,
    min: 1,
    max: 20,
    critical: [2, 4, 8],
    criticalNote: "每波长至少 2 个采样点，否则看起来在振荡其实全是锯齿",
    beyond: "below",
  });
  const viz: VizSpecJson = { ...heat, params: [spec] };
  const [c] = describeControls(viz, { samples_per_wavelength: 3 });
  assert.equal(c.governing, 4, "低于 4 但没低于 8 → 支配值是 4");
  assert.equal(c.crossed, true);
  assert.equal(c.beyond, "below");
  assert.equal(BEYOND_TEXT.below, "不得低于");

  const [ok] = describeControls(viz, { samples_per_wavelength: 9 });
  assert.equal(ok.crossed, false, "9 没低于任何阈值");
  assert.equal(ok.governing, 8, "below 方向、都没突破时取最大阈值（最紧的那条）");
});

test("临界值标记：范围缺失时 fraction 为 null（UI 就不画刻度）", () => {
  const spec = param({ name: "x", type: "float", default: 1, critical: [2, 3], criticalNote: "越界会发散" });
  const viz: VizSpecJson = { ...heat, params: [spec] };
  const [c] = describeControls(viz);
  assert.deepEqual(c.criticalMarks.map((m) => m.fraction), [null, null]);
  assert.equal(c.min, null);
  assert.equal(c.max, null);
  assert.equal(c.crossed, false, "1 没超过任何阈值");
});

// --------------------------------------------------------------------------- //
// enum / bool / 自适应提示
// --------------------------------------------------------------------------- //

test("enum 与 bool：options 保留、非法值回落、checkbox 语义正确", () => {
  const params: ParamSpecJson[] = [
    param({ name: "pivot", type: "enum", default: "partial", options: ["none", "partial", "complete"], critical: [] }),
    param({ name: "use_log", type: "bool", default: true, critical: [] }),
  ];
  const viz: VizSpecJson = { ...heat, params };
  const specs = describeControls(viz, { pivot: "不存在的选项", use_log: "false" });
  assert.deepEqual(specs[0].options, ["none", "partial", "complete"]);
  assert.equal(specs[0].value, "partial", "非法选项回落到 default");
  assert.equal(specs[1].value, false, "字符串 'false' 按布尔处理");
  assert.equal(specs[1].governing, null, "bool 没有临界值");
  assert.deepEqual(specs[1].criticalMarks, []);
});

test("自适应坐标轴提示（规则 1/4）：出现在每个控件的 hints 里", () => {
  assert.ok(heat.axes.some((a) => a.autoscaled), "夹具前提：heat_ftcs 有自适应误差副轴");
  const specs = describeControls(heat);
  assert.equal(specs[0].autoscaled, true);
  assert.ok(specs[0].hints.some((h) => h.includes("自适应")), specs[0].hints.join(" / "));
  assert.match(autoscaleNote(heat) ?? "", /error/);

  const fixed: VizSpecJson = { ...heat, axes: heat.axes.map((a) => ({ ...a, autoscaled: false })) };
  assert.equal(autoscaleNote(fixed), null);
  assert.equal(describeControls(fixed)[0].autoscaled, false);
});

test("单位会带上，且真实 manifest 的每个控件都有提示可读", () => {
  const spec = param({ name: "dt_ms", type: "float", default: 1, unit: "ms", min: 0, max: 10, critical: [] });
  const viz: VizSpecJson = { ...heat, params: [spec] };
  assert.ok(describeControls(viz)[0].hints.some((h) => h === "单位：ms"));
  for (const v of manifest.vizzes) {
    for (const c of describeControls(v)) {
      assert.ok(c.hints.length > 0, `${v.id}.${c.name} 必须至少有一条提示`);
      assert.equal(typeof c.crossed, "boolean");
    }
  }
});

// --------------------------------------------------------------------------- //
// clampParam
// --------------------------------------------------------------------------- //

test("clampParam：int/float/enum/bool 的边界与类型归一", () => {
  const int = param({ name: "n", type: "int", default: 24, min: 4, max: 128 });
  assert.equal(clampParam(int, 3.2), 4, "先取整（3）再夹下界（4）");
  assert.equal(clampParam(int, 3.7), 4, "四舍五入到 4，仍在域内");
  assert.equal(clampParam(int, 1e9), 128);
  assert.equal(clampParam(int, "64"), 64, "数字字符串可用");
  assert.equal(clampParam(int, "abc"), 24, "非数字回默认值");
  assert.equal(clampParam(int, Number.POSITIVE_INFINITY), 24);
  assert.equal(clampParam(int, true), 24, "布尔不是数字");

  const float = param({ name: "r", type: "float", default: 0.5, min: 0, max: 1 });
  assert.equal(clampParam(float, 0.25), 0.25);
  assert.equal(clampParam(float, -1), 0);
  assert.equal(clampParam(float, 1e9), 1);
  assert.equal(clampParam(float, null), 0.5);

  const noRange = param({ name: "k", type: "float", default: 7 });
  assert.equal(clampParam(noRange, 1e9), 1e9, "没有 min/max 就不夹");
  assert.equal(clampParam(noRange, Number.NaN), 7, "但 NaN 仍然回默认值");

  const en = param({ name: "m", type: "enum", default: "b", options: ["a", "b"] });
  assert.equal(clampParam(en, "a"), "a");
  assert.equal(clampParam(en, "z"), "b");
  assert.equal(clampParam(en, 123), "b");

  const flag = param({ name: "f", type: "bool", default: false });
  assert.equal(clampParam(flag, true), true);
  assert.equal(clampParam(flag, "true"), true);
  assert.equal(clampParam(flag, "false"), false);
  assert.equal(clampParam(flag, 1), false, "数字不猜，回默认值");
});

test("clampParam：默认值本身越界时也要被夹回域内", () => {
  const bad = param({ name: "x", type: "float", default: 99, min: 0, max: 1 });
  assert.equal(clampParam(bad, "junk"), 1, "默认值 99 先被夹到 1");
});

test("describeControls：默认值缺失（undefined）时不抛", () => {
  const params: ParamSpecJson[] = [
    param({ name: "a", type: "float" }),
    param({ name: "b", type: "int", min: 2, max: 4 }),
    param({ name: "c", type: "enum", options: [] }),
  ];
  const viz: VizSpecJson = { ...heat, params };
  const specs = describeControls(viz);
  assert.equal(specs[0].value, 0);
  assert.equal(specs[1].value, 2, "没有 default 就用 min");
  assert.equal(specs[2].value, "", "空 options 给空串，不抛");
  assert.deepEqual(describeControls(oscillator).map((s) => s.name), ["omega", "zeta", "n_periods"]);
});
