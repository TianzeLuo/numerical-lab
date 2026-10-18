/**
 * Scene 契约的测试。
 *
 * 这一层是"防误导七条规则"的落地点：**忘记标注就必须报错**。
 * 因此测试的重点不是"能不能画出来"，而是"少写一个字段会不会被抓住"。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  assertValidScene,
  axisModeBadge,
  badge,
  barsLayer,
  contourLayer,
  criticalBadge,
  heatLayer,
  lineLayer,
  loglogFitLayer,
  makeAxis,
  makeErrorAxis,
  newScene,
  overflowBadge,
  scatterLayer,
  sceneToJson,
  series,
  sourceBadge,
  text,
  validateScene,
  vectorFieldLayer,
  type Axis,
  type Badge,
  type Layer,
  type Scene,
  previewBadge,
} from "../shared/src/scene.js";

/** 造一个"合法但极简"的场景，再让各测试按需破坏它。 */
function baseScene(): Scene {
  const s = newScene({ chapter: "0.0", viz: "demo", title: "演示", pitfalls: "自适应坐标轴会掩盖发散，这里固定纵轴。" });
  s.axes.push(makeAxis("main", [60, 40, 500, 300], { xlim: [0, 1], ylim: [-1, 1], xlabel: "x", ylabel: "u" }));
  s.layers.push(lineLayer("main", series([0, 0.5, 1], [0, 1, 0], "#333")));
  s.badges.push(sourceBadge(s.dataSource));
  return s;
}

/** 深拷贝后再改，避免测试之间互相污染。 */
function clone(s: Scene): Scene {
  return JSON.parse(JSON.stringify(s)) as Scene;
}

// --------------------------------------------------------------------------- //
// 合法场景
// --------------------------------------------------------------------------- //
test("最小合法场景通过校验", () => {
  const s = baseScene();
  s.badges.push(axisModeBadge(false));
  s.texts.push(text(10, 10, "标题", { role: "title" }));
  assert.deepEqual(validateScene(s), []);
  assert.equal(assertValidScene(s).viz, "demo");
});

test("assertValidScene 在非法时抛出并列出全部问题", () => {
  const s = clone(baseScene());
  s.badges = [];
  s.overflow = { flag: true, note: "" };
  s.pitfalls = "太短";
  const err = (() => {
    try {
      assertValidScene(s);
      return null;
    } catch (e) {
      return e as Error;
    }
  })();
  assert.ok(err, "应当抛错");
  assert.match(err.message, /source 徽标/);
  assert.match(err.message, /overflow\.note/);
  assert.match(err.message, /pitfalls 太短/);
});

test("非对象与错误版本被拒绝", () => {
  assert.deepEqual(validateScene(null), ["Scene 不是对象"]);
  const s = clone(baseScene()) as unknown as Record<string, unknown>;
  s["version"] = 2;
  assert.ok(validateScene(s).some((x) => x.includes("version")));
});

// --------------------------------------------------------------------------- //
// 规则 1 / 4：溢出与自适应
// --------------------------------------------------------------------------- //
test("overflow.flag 为真但没有说明 → 报错（规则 1）", () => {
  const s = clone(baseScene());
  s.overflow = { flag: true, note: "   " };
  assert.ok(validateScene(s).some((x) => x.includes("overflow.note")));
  s.overflow = { flag: true, note: "u 超过 1e6，已裁剪到 1e6 显示" };
  assert.deepEqual(validateScene(s), []);
});

test("坐标轴必须显式声明 autoscaled（规则 4）", () => {
  const s = clone(baseScene());
  delete (s.axes[0] as Partial<Axis>).autoscaled;
  assert.ok(validateScene(s).some((x) => x.includes("autoscaled")));
});

test("自适应徽标必须写明风险", () => {
  assert.match(axisModeBadge(true).text, /自适应/);
  assert.match(axisModeBadge(true).text, /掩盖/);
  assert.equal(axisModeBadge(false).text, "坐标轴：固定");
});

test("退化与颠倒的坐标范围被拒绝", () => {
  const s = clone(baseScene());
  s.axes[0]!.xlim = [1, 1];
  s.axes[0]!.ylim = [1, -1];
  const issues = validateScene(s);
  assert.ok(issues.some((x) => x.includes("退化")));
  assert.ok(issues.some((x) => x.includes("顺序颠倒")));
});

test("对数轴下界非正被拒绝", () => {
  const s = clone(baseScene());
  s.axes[0]!.xscale = "log";
  s.axes[0]!.xlim = [0, 1];
  assert.ok(validateScene(s).some((x) => x.includes("对数 x 轴")));
});

// --------------------------------------------------------------------------- //
// 规则 2：双对数拟合必须同时给出实测与理论斜率
// --------------------------------------------------------------------------- //
test("loglogFitLayer 拒绝非有限斜率", () => {
  assert.throws(() => loglogFitLayer("main", [0.1, 0.05], [1e-2, 2.5e-3], Number.NaN, 2), /实测斜率与理论斜率/);
  assert.throws(() => loglogFitLayer("main", [0.1, 0.05], [1e-2, 2.5e-3], 2, Number.POSITIVE_INFINITY), /实测斜率与理论斜率/);
});

test("loglogFitLayer 需要至少两个不同的正 h", () => {
  assert.throws(() => loglogFitLayer("main", [0.1, 0.1], [1e-2, 2e-2], 2, 2), /两个不同/);
});

test("loglogfit 层缺失理论斜率时校验器报错（构造器绕不过去的后门也要堵）", () => {
  const s = clone(baseScene());
  const layer = loglogFitLayer("main", [0.1, 0.05, 0.025], [1e-2, 2.6e-3, 6.6e-4], 1.98, 2) as Layer & { theorySlope?: number };
  s.layers.push(layer);
  assert.deepEqual(validateScene(s).filter((x) => x.includes("loglogfit")), []);
  delete layer.theorySlope;
  assert.ok(validateScene(s).some((x) => x.includes("theorySlope")));
});

test("loglogfit 的默认标签同时点明实测与理论", () => {
  const l = loglogFitLayer("main", [0.1, 0.05], [1e-2, 2.5e-3], 1.98, 2);
  assert.equal(l.kind, "loglogfit");
  if (l.kind === "loglogfit") {
    assert.match(l.seriesLabel, /实测斜率/);
    assert.match(l.theoryLabel, /理论斜率/);
    assert.ok(l.refX[0] > 0 && l.refX[1] > l.refX[0]);
  }
});

// --------------------------------------------------------------------------- //
// 图层字段
// --------------------------------------------------------------------------- //
test("图层引用不存在的轴被拒绝", () => {
  const s = clone(baseScene());
  s.layers.push(scatterLayer("nope", [0], [0]));
  assert.ok(validateScene(s).some((x) => x.includes("不存在的轴")));
});

test("曲线长度不一致与非有限值被拒绝", () => {
  const s = clone(baseScene());
  s.layers.push(lineLayer("main", series([0, 1, 2], [0, Number.NaN], "#000")));
  const issues = validateScene(s);
  assert.ok(issues.some((x) => x.includes("长度不一致")));
  assert.ok(issues.some((x) => x.includes("非有限纵坐标")));
});

test("heat 的 values 长度必须等于 nx*ny，且 vmax>vmin", () => {
  const s = clone(baseScene());
  s.layers.push(heatLayer("main", new Float64Array(8), 4, 2, [0, 1, 0, 1]));
  assert.deepEqual(validateScene(s), []);
  const bad = clone(baseScene());
  bad.layers.push(heatLayer("main", new Float64Array(12), 4, 2, [0, 1, 0, 1]));
  assert.ok(validateScene(bad).some((x) => x.includes("nx*ny")));
});

test("heatLayer 自动取值域；对数色标下 vmin 被抬到正数", () => {
  const vals = new Float64Array([1, 2, 3, 4, 5, 6]);
  const lin = heatLayer("main", vals, 3, 2, [0, 1, 0, 1]);
  assert.deepEqual(validateScene({ ...baseScene(), layers: [lin] }), []);
  if (lin.kind === "heat") {
    assert.equal(lin.vmin, 1);
    assert.equal(lin.vmax, 6);
    assert.equal(lin.cmap, "viridis");
  }
  const log = heatLayer("main", new Float64Array([0, 1, 10]), 3, 1, [0, 1, 0, 1], { logScale: true, vmax: 10 });
  if (log.kind === "heat") assert.ok(log.vmin > 0, "对数色标必须把 vmin 抬到正数");
});

test("未知色标与未知图层类型被拒绝", () => {
  const s = clone(baseScene());
  const l = heatLayer("main", new Float64Array(4), 2, 2, [0, 1, 0, 1]) as Layer & { cmap: string };
  l.cmap = "jet";
  s.layers.push(l);
  assert.ok(validateScene(s).some((x) => x.includes("未知色标")));
  const s2 = clone(baseScene());
  s2.layers.push({ kind: "sparkle", axes: "main" } as unknown as Layer);
  assert.ok(validateScene(s2).some((x) => x.includes("未知的 kind")));
});

test("contour 自动给出内部层级", () => {
  const vals = new Float64Array(25).map((_, i) => i);
  const l = contourLayer("main", vals, 5, 5, [0, 1, 0, 1]);
  if (l.kind === "contour") {
    assert.ok(l.levels.length >= 3);
    assert.ok(l.levels[0]! > 0 && l.levels[l.levels.length - 1]! < 24);
  }
});

test("bars 的误差棒长度必须与数值一致", () => {
  const s = clone(baseScene());
  s.layers.push(barsLayer("main", [1, 2], [0.1, 0.2], { errors: new Float64Array([0.01]) }));
  assert.ok(validateScene(s).some((x) => x.includes("errors 长度")));
});

test("vectorfield 的 u/v 长度必须等于 nx*ny，stride 至少为 1", () => {
  const s = clone(baseScene());
  s.layers.push(vectorFieldLayer("main", new Float64Array(6), new Float64Array(6), 3, 2, [0, 1, 0, 1], { stride: 0 }));
  assert.deepEqual(validateScene(s).filter((x) => x.includes("u/v 长度")), []);
  // 构造器会把非法 stride 夹到 1（前端不该因为一个手滑参数就崩）
  const built = vectorFieldLayer("main", new Float64Array(6), new Float64Array(6), 3, 2, [0, 1, 0, 1], { stride: 0 });
  if (built.kind === "vectorfield") assert.equal(built.stride, 1);
  // 校验器本身仍然要挡住别人手写的非法图层
  const handWritten = clone(baseScene());
  handWritten.layers.push({ kind: "vectorfield", axes: "main", nx: 3, ny: 2, u: new Float64Array(6), v: new Float64Array(6), extent: [0, 1, 0, 1], stride: 0, color: "#000", scale: 1 });
  assert.ok(validateScene(handWritten).some((x) => x.includes("stride")));
});

// --------------------------------------------------------------------------- //
// 文本与徽标
// --------------------------------------------------------------------------- //
test("徽标类型与文本非空被校验", () => {
  const s = clone(baseScene());
  s.badges.push({ kind: "shiny", text: "?" } as unknown as Badge);
  s.badges.push(badge("status", ""));
  const issues = validateScene(s);
  assert.ok(issues.some((x) => x.includes("徽标类型未知")));
  assert.ok(issues.some((x) => x.includes("徽标文本不得为空")));
});

test("文本角色受限，坐标必须有限", () => {
  const s = clone(baseScene());
  s.texts.push({ x: 0, y: 0, text: "x", size: 12, color: "#000", align: "left", role: "wat" as never });
  s.texts.push(text(Number.NaN, 0, "坏坐标"));
  const issues = validateScene(s);
  assert.ok(issues.some((x) => x.includes("角色未知")));
  assert.ok(issues.some((x) => x.includes("坐标非有限")));
});

test("criticalBadge 在越过理论临界值时给出警告（方向由调用方声明）", () => {
  const inside = criticalBadge("ρ", 0.9, 1.0);
  assert.equal(inside.kind, "critical");
  assert.doesNotMatch(inside.text, /越界/);
  assert.match(inside.text, /1\.000\*/); // 起支配作用的那条阈值被标星
  const outside = criticalBadge("Δt", 0.6, [0.5, 0.25]);
  assert.match(outside.text, /理论临界/);
  assert.match(outside.text, /越界/);
  assert.match(outside.text, /0\.2500\*/); // 最紧的那条先被突破
});

test("criticalBadge 支持下界型与绝对值型约束", () => {
  const above = criticalBadge("ν", 0.005, [0.01, 0.1, 1]);
  assert.doesNotMatch(above.text, /越界/);
  const aboveBad = criticalBadge("ν", 0.02, [0.01, 0.1, 1]);
  assert.match(aboveBad.text, /越界/);
  const below = criticalBadge("ζ", 0.05, [0.1, 0.5], { beyond: "below" });
  assert.match(below.text, /越界/);
  assert.match(below.text, /0\.5000\*/); // 下界型：最大的那条是支配约束
  const mag = criticalBadge("ρ(J)", -1.2, 1.0, { beyond: "magnitude" });
  assert.match(mag.text, /越界/);
  const magOk = criticalBadge("ρ(J)", -0.8, 1.0, { beyond: "magnitude" });
  assert.doesNotMatch(magOk.text, /越界/);
  const unknown = criticalBadge("x", Number.NaN, 1);
  assert.match(unknown.text, /无可比临界值/);
});

test("sourceBadge 恒为 source，preview 另有一枚（规则 5）", () => {
  // `kind` 的唯一性是有意的：`validateScene` 要求任何场景都有 `kind="source"`，
  // 早先 `sourceBadge("backend+preview")` 返回 `kind="preview"`，纯预览场景必然非法。
  assert.equal(sourceBadge("backend").kind, "source");
  const prev = sourceBadge("backend+preview", "拖动时");
  assert.equal(prev.kind, "source");
  assert.match(prev.text, /近似预览/);
  assert.match(prev.text, /拖动时/);
  const badge2 = previewBadge("拖动时", 8, 26);
  assert.equal(badge2.kind, "preview");
  assert.match(badge2.text, /^预览：前端近似/);
  assert.match(badge2.text, /拖动时/);
});

test("overflowBadge 文本带上前缀", () => {
  assert.equal(overflowBadge("已裁剪").kind, "overflow");
  assert.match(overflowBadge("已裁剪").text, /^溢出：/);
});

// --------------------------------------------------------------------------- //
// 轴构造
// --------------------------------------------------------------------------- //
test("makeAxis 自动生成刻度，makeErrorAxis 默认对数且自适应", () => {
  const a = makeAxis("m", [0, 0, 540, 320], { xlim: [0, 10], ylim: [0, 100] });
  assert.ok(a.xticks.length >= 3 && a.yticks.length >= 3);
  assert.equal(a.autoscaled, false);
  const e = makeErrorAxis("err", [0, 360, 540, 120]);
  assert.equal(e.yscale, "log");
  assert.equal(e.autoscaled, true);
  assert.match(e.title, /误差/);
});

test("对数轴的默认刻度来自 niceTicks(log)", () => {
  const a = makeAxis("log", [0, 0, 400, 300], { xlim: [1e-4, 1], xscale: "log" });
  assert.ok(a.xticks.length >= 3);
  assert.ok(a.xticks.every((t) => t > 0));
});

// --------------------------------------------------------------------------- //
// 稳定序列化（快照/视觉回归的基础）
// --------------------------------------------------------------------------- //
test("sceneToJson 键序稳定、TypedArray 归一、-0 与 NaN 归一", () => {
  /** 每次新建（而不是 JSON.parse 克隆），保证两侧都持有真正的 TypedArray。 */
  const build = (): Scene => {
    const s = baseScene();
    s.layers.push(lineLayer("main", series(new Float64Array([0, -0, Number.NaN]), new Float64Array([1, 2, 3]), "#111")));
    return s;
  };
  const j1 = sceneToJson(build());
  const j2 = sceneToJson(build());
  assert.equal(j1, j2);
  const parsed = JSON.parse(j1) as { layers: { kind: string; series?: { x: number[] }[] }[] };
  const xArr = parsed.layers[1]!.series![0]!.x;
  assert.equal(xArr[1], 0);
  assert.ok(!Object.is(xArr[1], -0), "-0 必须被归一成 0");
  assert.equal(xArr[2], "NaN"); // 非有限值变成可读字符串，而不是静默的 null
  // 键序稳定：axes 一定出现在 badges 之前（按字典序）
  assert.ok(j1.indexOf('"axes"') < j1.indexOf('"badges"'));
});

test("sceneToJson 对数值做 12 位有效数字截断，避免跨平台末位抖动", () => {
  const s = baseScene();
  s.t = 0.1 + 0.2;
  const json = sceneToJson(s);
  assert.match(json, /"t": 0\.3/);
});
