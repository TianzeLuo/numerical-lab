/**
 * 6.2 章 model 层测试 —— 自足合成帧。
 *
 * 守的几条：
 * 1. 目录默认只标高亮闭包的章号（28 个标签会互相压）；窄画布仍保留图与副图。
 * 2. 导出页的主图 x 轴是**名义帧时刻**，stall 用红圈标；线层不许再落内联标签。
 * 3. 讲义热图行 0 在底：第一行章的数值必须出现在热图最后一行；失败行强制有行标签。
 * 4. 960 与 640 两种版式都要通过 validateScene（含文本重叠检查）。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Layer, type Scene } from "../shared/src/scene.js";
import {
  bookBuildModel,
  catalogNavModel,
  exportTimelineModel,
  manifest,
  modelFor,
  models,
} from "../chapters/6.2/index.js";

type Row = readonly [name: string, data: ArrayLike<number>, label: string, shape?: readonly number[]];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  pitfalls?: string;
}

const PITFALLS =
  "目录顺序不是构建顺序：章节号只反映写作计划，依赖图才决定先构建谁。" +
  "视频的时间轴有两本账：内容时刻由取样决定，播放时刻被 GIF 的整数厘秒 delay 改写；" +
  "24 fps 会被写成 4 cs，实际播成 25 fps，片尾漂移按帧号线性累积。" +
  "编译通过不等于书对：缺字形会被 xelatex 静默吞掉，附表行数必须与断言报告条数一致，" +
  "注释里的引用与输入不算依赖。热图的 row 0 在 y-min，行标签必须按同一约定翻转。";

const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };

async function makeFrame(def: FrameDef): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const [name, data, label, shape] of def.arrays) {
    const typed = Float64Array.from(data as ArrayLike<number>);
    specs.push({ name, dtype: "f8", shape: (shape ?? [data.length]) as number[], order: "C", offset: 0, nbytes: typed.byteLength, unit: "", label });
    arrays.set(name, typed);
  }
  return decodeNlf(
    await encodeNlf(
      new Frame({
        chapter: "6.2",
        viz: def.viz,
        kind: "frame",
        codec: "none",
        meta: def.meta,
        strings: def.strings,
        critical: {},
        pitfalls: def.pitfalls ?? PITFALLS,
        specs,
        arrays,
      }),
    ),
  );
}

async function sceneOf(viz: string, def: FrameDef, opts: Partial<ModelOptions> = {}): Promise<Scene> {
  const frame = await makeFrame(def);
  const model = models[viz];
  assert.ok(model, `没有 model ${viz}`);
  const scene = model(frame, { ...OPTS, ...opts });
  const problems = validateScene(scene);
  assert.deepEqual(problems, [], `${viz} 场景不合规：${problems.join("；")}`);
  return scene;
}

const layersOn = (scene: Scene, id: string): Layer[] => scene.layers.filter((l) => l.axes === id);
const textOf = (scene: Scene): string => scene.texts.map((t) => t.text).join(" | ");
const badgeOf = (scene: Scene): string => scene.badges.map((b) => b.text).join(" | ");

// --------------------------------------------------------------------------- //
// 合成帧
// --------------------------------------------------------------------------- //
const CATALOG: FrameDef = {
  viz: "catalog_nav",
  arrays: [
    ["node_x", [0, 1, 1, 2, 3, 4], "深度"],
    ["node_y", [0, -0.5, 0.5, 0, 0, 0], "层内序号"],
    ["node_part", [1, 1, 2, 2, 6, 6], "分部"],
    ["node_viz", [3, 4, 4, 4, 4, 3], "viz 数"],
    ["node_rank", [0, 1, 2, 3, 4, 5], "拓扑序号"],
    ["node_deps", [0, 1, 1, 1, 1, 1], "前置数"],
    ["node_highlight", [1, 1, 0, 0, 0, 1], "高亮"],
    ["node_focus", [0, 0, 0, 0, 0, 1], "焦点"],
    ["edge_x0", [0, 0, 0, 1, 2, 3], "边起点 x"],
    ["edge_y0", [0, 0, 0, -0.5, 0.5, 0], "边起点 y"],
    ["edge_x1", [1, 1, 1, 2, 3, 4], "边终点 x"],
    ["edge_y1", [-0.5, 0.5, 0, 0, 0, 0], "边终点 y"],
    ["edge_highlight", [1, 1, 0, 0, 0, 1], "边高亮"],
    ["error", [0, 0, 0, 0], "错误计数"],
    ["error_values", [0, 0, 0, 0], "错误计数"],
  ],
  meta: {
    n_chapters: 6, n_edges: 6, n_layers: 5, max_depth: 4, n_cycles: 0,
    n_unknown: 0, n_self: 0, n_missing_module: 0, n_violations: 0,
    focus_closure: 3, focus_edges: 3, chain_len: 4,
  },
  strings: {
    status: "ok",
    focus: "6.2",
    part: "all",
    chain: "1.1 → 1.2 → 6.2",
    chapter_ids: JSON.stringify(["1.1", "1.2", "2.1", "2.2", "6.1", "6.2"]),
    parts: JSON.stringify(["I", "I", "II", "II", "VI", "VI"]),
    closure_note: "焦点 6.2 的前置闭包含 3 章。",
    layer_note: "5 层、6 条边。",
    problem_note: "四项检查全为 0。",
  },
};

const EXPORT: FrameDef = {
  viz: "export_timeline",
  arrays: [
    ["source_step_t", [0, 0.2, 0.5, 1], "步长起点"],
    ["source_step_h", [0.2, 0.3, 0.5, 3], "步长"],
    ["frame_t", Array.from({ length: 12 }, (_, i) => i / 3), "名义帧时刻"],
    ["content_t", Array.from({ length: 12 }, (_, i) => i / 3 + (i % 2 === 0 ? 0.05 : -0.05)), "内容时刻"],
    ["playback_t", Array.from({ length: 12 }, (_, i) => i * 0.04), "播放时刻"],
    ["content_index", [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3], "源下标"],
    ["timing_error", Array.from({ length: 12 }, (_, i) => (i % 2 === 0 ? 0.05 : -0.05)), "内容误差"],
    ["playback_error", Array.from({ length: 12 }, (_, i) => i * 0.04 - i / 3), "播放漂移"],
    ["state_error", Array.from({ length: 12 }, (_, i) => 0.01 / (i + 1)), "状态误差"],
    ["stall", [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1], "stall"],
    ["error", Array.from({ length: 12 }, (_, i) => (i % 2 === 0 ? 0.05 : -0.05)), "requiredAux"],
  ],
  meta: {
    tol: 1e-5, n_source: 5, n_steps: 4, n_rejected: 0,
    fps: 24, delay_cs: 4, actual_fps: 25,
    h_min: 0.2, h_max: 3, h_ratio: 15, span: 4,
    n_raw: 97, n_frames: 12, truncated: 0, n_truncated: 0,
    nominal_duration: 0.5, playback_duration: 0.48,
    max_timing_error: 0.05, max_playback_error: 0.16, max_state_error: 0.01,
    n_stalls: 7, max_hold: 0.25, n_skipped_source: 0,
  },
  strings: {
    status: "ok",
    sampling: "nearest",
    sampling_note: "最近邻快照。",
    source_note: "自适应源。",
    frame_note: "12 帧；stall 7。",
    gif_note: "delay=4 cs。",
    state_note: "状态误差 0.01。",
    error_note: "内容误差有界，播放漂移线性。",
  },
};

const BOOK: FrameDef = {
  viz: "book_build",
  arrays: [
    ["check_values", [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "检查矩阵"],
    ["chapter_index", [0, 1, 2, 3], "行号"],
    ["error", [0, 1, 0, 0], "问题数"],
    ["error_y", [0, 1, 0, 0], "问题数"],
    ["labels", [5, 6, 7, 8], "labels"],
    ["refs", [2, 3, 4, 5], "refs"],
    ["inputs", [1, 1, 1, 1], "inputs"],
    ["risky", [0, 0, 0, 0], "risky"],
    ["table_rows", [12, 10, 14, 16], "table rows"],
    ["declared", [12, 11, 14, 16], "report count"],
    ["theorems", [2, 3, 1, 2], "theorems"],
    ["chars", [1000, 1200, 900, 1100], "chars"],
  ],
  meta: {
    n_chapters: 4, n_checks: 6, n_issues: 1,
    n_labels: 26, n_refs: 14, n_inputs: 4, n_risky: 0,
    n_unresolved: 0, n_duplicates: 0, n_missing_inputs: 0,
    n_missing_tables: 0, n_table_mismatch: 1, n_table_rows: 52, n_declared: 53, n_theorems: 8,
  },
  strings: {
    status: "degraded",
    part: "all",
    sort: "topo",
    chapter_ids: JSON.stringify(["1.1", "1.2", "1.3", "1.4"]),
    check_ids: JSON.stringify(["notes", "table", "refs", "inputs", "glyph", "pitfalls"]),
    check_labels: JSON.stringify(["笔记", "附表", "引用", "输入", "字形", "陷阱"]),
    summary: "4 章 × 6 项检查；问题格 1 个。",
    issue_note: "未通过：1.2: 附表",
    ref_note: "标签 26、引用 14、输入 4；未定义 0。",
    table_note: "附表行数 52，报告条数 53；不一致 1 章。",
    glyph_note: "字形风险字符 0 个。",
    pitfall_note: "pitfalls 全部 ≥800 字。",
    check_note: "列：笔记存在且含定理 / 附表行数=声明 / 引用闭合 / 输入存在 / 无字形风险 / pitfalls≥800。",
  },
};

// --------------------------------------------------------------------------- //
// 1. 登记表
// --------------------------------------------------------------------------- //
test("manifest 与 model 登记表：三页齐备且 id 固定", () => {
  assert.deepEqual(manifest.vizzes.map((v) => v.id), ["catalog_nav", "export_timeline", "book_build"]);
  for (const id of manifest.vizzes.map((v) => v.id)) {
    assert.equal(typeof models[id], "function");
    assert.equal(modelFor(id), models[id]);
  }
  assert.equal(modelFor("nope"), null);
  assert.equal(catalogNavModel, models["catalog_nav"]);
  assert.equal(exportTimelineModel, models["export_timeline"]);
  assert.equal(bookBuildModel, models["book_build"]);
});

// --------------------------------------------------------------------------- //
// 2. 目录
// --------------------------------------------------------------------------- //
test("catalog_nav：小图全标、边/节点分层、副图与侧栏", async () => {
  const scene = await sceneOf("catalog_nav", CATALOG);
  const labelTexts = scene.texts.filter((t) => t.role === "label").map((t) => t.text);
  // 6 个节点的小图全标；28 章的默认帧只标高亮闭包（视觉基线检查）。
  assert.deepEqual(labelTexts.sort(), ["1.1", "1.2", "2.1", "2.2", "6.1", "6.2"]);
  assert.ok(scene.axes.some((a) => a.id === "main"));
  assert.ok(scene.axes.some((a) => a.id === "error"));
  const kinds = layersOn(scene, "main").map((l) => l.kind);
  assert.ok(kinds.includes("line"));
  assert.ok(kinds.includes("scatter"));
  assert.ok(layersOn(scene, "error").some((l) => l.kind === "bars"));
  assert.match(textOf(scene), /最长关键路径/);
  assert.match(textOf(scene), /闭包/);
  assert.match(badgeOf(scene), /坐标轴：固定/);
  assert.match(badgeOf(scene), /问题 0/);
});

test("catalog_nav：640 窄画布保留图与副图并显式降级", async () => {
  const scene = await sceneOf("catalog_nav", CATALOG, { width: 640 });
  assert.match(badgeOf(scene), /窄画布/);
  assert.ok(scene.axes.some((a) => a.id === "main"));
  assert.ok(scene.axes.some((a) => a.id === "error"));
});

// --------------------------------------------------------------------------- //
// 3. 导出
// --------------------------------------------------------------------------- //
test("export_timeline：主图 x=名义帧时刻，stall 红圈，线层无内联标签", async () => {
  const scene = await sceneOf("export_timeline", EXPORT);
  const mainLayers = layersOn(scene, "main");
  const scatters = mainLayers.filter((l) => l.kind === "scatter") as Extract<Layer, { kind: "scatter" }>[];
  assert.ok(scatters.length >= 2, "应有普通帧与 stall 两层 scatter");
  const stall = scatters.find((l) => l.r === 4);
  assert.ok(stall, "stall 层半径应为 4（红圈）");
  assert.ok(Math.abs((stall.x[0] as number) - (EXPORT.arrays[2]![1] as number[])[1]!) < 1e-12, "stall 的 x 应是名义帧时刻");
  for (const layer of scene.layers) {
    if (layer.kind !== "line") continue;
    for (const s of layer.series) assert.equal(s.label, undefined, "线层不许带内联标签");
  }
  assert.match(textOf(scene), /GIF/);
  assert.match(textOf(scene), /24→25/);
  assert.match(textOf(scene), /片尾漂移/);
  assert.match(textOf(scene), /红圈/);
  assert.match(badgeOf(scene), /25 fps/);
});

test("export_timeline：640 窄画布保留两条误差曲线与说明", async () => {
  const scene = await sceneOf("export_timeline", EXPORT, { width: 640 });
  assert.match(badgeOf(scene), /窄画布/);
  const line = layersOn(scene, "error").find((l) => l.kind === "line") as Extract<Layer, { kind: "line" }>;
  assert.equal(line.series.length, 3);
});

// --------------------------------------------------------------------------- //
// 4. 讲义
// --------------------------------------------------------------------------- //
test("book_build：热图行 0 在底、失败行强制标签、副图问题点分开", async () => {
  const scene = await sceneOf("book_build", BOOK);
  const heat = layersOn(scene, "main").find((l) => l.kind === "heat") as Extract<Layer, { kind: "heat" }>;
  assert.ok(heat);
  assert.equal(heat.nx, 6);
  assert.equal(heat.ny, 4);
  // 第一个章（1.1）全过，应出现在热图的最后一行（y-max）。
  assert.deepEqual(Array.from(heat.values).slice(3 * 6, 4 * 6), [1, 1, 1, 1, 1, 1]);
  // 1.2 的附表格在第二行（从底数第 2 行，即 y=2）。
  assert.equal(heat.values[2 * 6 + 1], 0);
  const labels = scene.texts.filter((t) => t.role === "label").map((t) => t.text);
  assert.ok(labels.includes("1.2"), "失败行必须有标签");
  const err = layersOn(scene, "error").filter((l) => l.kind === "scatter") as Extract<Layer, { kind: "scatter" }>[];
  assert.equal(err.length, 2, "通过/失败两组散点");
  assert.ok(err.some((l) => Array.from(l.y).includes(1)), "问题章的点应在 y=1");
  assert.match(textOf(scene), /未通过：1\.2/);
  assert.match(badgeOf(scene), /问题 1/);
});

test("book_build：640 窄画布省略副图但保留热图与降级说明", async () => {
  const scene = await sceneOf("book_build", BOOK, { width: 640 });
  assert.match(badgeOf(scene), /窄画布/);
  assert.ok(scene.axes.some((a) => a.id === "main"));
  assert.ok(!scene.axes.some((a) => a.id === "error"));
});

// --------------------------------------------------------------------------- //
// 5. 缺数组必须当场炸
// --------------------------------------------------------------------------- //
test("缺少必需数组时抛 ModelInputError", async () => {
  const broken: FrameDef = {
    ...EXPORT,
    arrays: EXPORT.arrays.filter(([name]) => name !== "timing_error" && name !== "error"),
  };
  const frame = await makeFrame(broken);
  assert.throws(() => exportTimelineModel(frame, OPTS), ModelInputError);
});
