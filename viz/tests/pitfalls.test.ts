/**
 * "它会怎样骗你"面板模型的测试（DOM 部分不测）。
 *
 * 最容易被忽视、也最容易出错的是**切句**：中文技术文本里全是 `r=0.5`、`1/2`、`e^{−αt}`，
 * 按英文句点切会把数字劈成两半。这里用真实 manifest 的 pitfalls 文本当夹具。
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { manifest } from "../chapters/0.0/manifest.gen.js";
import type { VizSpecJson } from "../shared/src/manifest.js";
import { dataSourceBadgeText, pitfallsModel, splitSentences } from "../shared/src/pitfalls.js";

const oscillator = manifest.vizzes.find((v) => v.id === "oscillator_regimes") as VizSpecJson;
const heat = manifest.vizzes.find((v) => v.id === "heat_ftcs") as VizSpecJson;

test("splitSentences：数字里的小数点不切句", () => {
  const text = "r=0.5 是阈值。超过 1.5 倍就会炸；0.05 没事。";
  const out = splitSentences(text);
  assert.deepEqual(out, ["r=0.5 是阈值。", "超过 1.5 倍就会炸；", "0.05 没事。"]);
  assert.equal(splitSentences("e^{−αk²π²t} 里的 3.14 不该断").length, 1);
  // 属性名里的点（meta.t_actual）后面没有空白 → 不是句末
  assert.deepEqual(splitSentences("时刻由 meta.t_actual 给出"), ["时刻由 meta.t_actual 给出"]);
});

test("splitSentences：中文/英文终止符都切，并保留标点", () => {
  const out = splitSentences("第一句！第二句？第三句；第四句. 第五句");
  assert.deepEqual(out, ["第一句！", "第二句？", "第三句；", "第四句.", "第五句"]);
  assert.deepEqual(splitSentences("真的吗？！是的。"), ["真的吗？！", "是的。"]);
});

test("splitSentences：换行当分隔，空行不产出空句", () => {
  const out = splitSentences("第一段\n\n第二段\n\n\n第三段");
  assert.deepEqual(out, ["第一段", "第二段", "第三段"]);
  assert.deepEqual(splitSentences("   \n  "), []);
});

test("pitfallsModel：标题、段落、徽标三段式结构齐全", () => {
  const model = pitfallsModel(heat);
  assert.ok(model.title.includes(heat.title));
  assert.match(model.title, /它会怎样骗你/);
  assert.ok(model.paragraphs.length >= 3, "至少有 pitfalls 句子 + 临界提示 + 数据来源说明");
  assert.ok(model.badges.length >= 3);
  // 第一段就是 pitfalls 的第一个句子（原样，不改写）
  assert.ok(heat.pitfalls.startsWith(model.paragraphs[0]));
  assert.ok(splitSentences(heat.pitfalls).length > 1, "夹具前提：pitfalls 是多句");
});

test("pitfallsModel：临界参数被单独成段，带上 criticalNote 与默认值状态", () => {
  const model = pitfallsModel(heat);
  const criticalLine = model.paragraphs.find((p) => p.includes("理论临界"));
  assert.ok(criticalLine, "必须有一段讲临界值（规则 6）");
  assert.match(criticalLine, /网格比 r/);
  assert.match(criticalLine, /0\.5/);
  assert.match(criticalLine, /FTCS 的放大因子/);
  assert.match(criticalLine, /默认值 0\.4 尚在阈值内/);
  assert.ok(model.badges.includes("临界参数：r"));
});

test("pitfallsModel：默认值已越界时给出 ⚠ 警告", () => {
  const viz: VizSpecJson = {
    ...heat,
    params: heat.params.map((p) => (p.name === "r" ? { ...p, default: 0.6 } : p)),
  };
  const model = pitfallsModel(viz);
  assert.ok(model.paragraphs.some((p) => p.includes("⚠ 默认值 0.6 已超过 0.5")));
});

test("pitfallsModel：自适应坐标轴与误差场都出现在段落与徽标里", () => {
  const model = pitfallsModel(heat);
  assert.ok(model.paragraphs.some((p) => p.includes("自适应")));
  assert.ok(model.badges.includes("坐标轴：自适应（可能掩盖发散）"));
  assert.ok(model.badges.includes("含误差场：error"));
  assert.ok(model.badges.includes(dataSourceBadgeText(heat.dataSource)));
  assert.equal(dataSourceBadgeText("backend"), "数据：后端 .nlf 帧");
  assert.equal(dataSourceBadgeText("backend+preview"), "数据：后端帧 + 前端近似预览");

  const fixed: VizSpecJson = {
    ...heat,
    axes: heat.axes.map((a) => ({ ...a, autoscaled: false })),
  };
  const fixedModel = pitfallsModel(fixed);
  assert.ok(fixedModel.badges.includes("坐标轴：固定"));
  assert.ok(fixedModel.paragraphs.some((p) => p.includes("坐标轴是固定的")));
});

test("pitfallsModel：时间轴信息进徽标；无时间轴则不出现", () => {
  assert.ok(pitfallsModel(heat).badges.includes("时间轴：scrub"));
  const still: VizSpecJson = { ...heat, time: { mode: "none" } };
  assert.ok(!pitfallsModel(still).badges.some((b) => b.startsWith("时间轴")));
});

test("pitfallsModel：pitfalls 为空也不抛，给出明确的兜底段落", () => {
  const viz: VizSpecJson = { ...heat, pitfalls: "" };
  const model = pitfallsModel(viz);
  assert.equal(model.paragraphs.length, 4, "兜底段 + 临界段 + 坐标轴段 + 数据来源段");
  assert.match(model.paragraphs[0], /没有给出 pitfalls/);
  assert.ok(model.badges.length > 0);
});

test("pitfallsModel：真实 manifest 的每个 viz 都能生成非空面板", () => {
  for (const viz of manifest.vizzes) {
    const model = pitfallsModel(viz);
    assert.ok(model.title.length > 0, viz.id);
    assert.ok(model.paragraphs.length >= 3, viz.id);
    assert.ok(model.badges.length >= 3, viz.id);
    assert.ok(model.paragraphs.every((p) => p.trim().length > 0), viz.id);
    assert.ok(model.badges.every((b) => b.trim().length > 0), viz.id);
  }
  const osc = pitfallsModel(oscillator);
  assert.ok(osc.paragraphs.some((p) => p.includes("ζ")), "保留原文里的特殊符号");
});
