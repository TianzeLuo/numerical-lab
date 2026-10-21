/**
 * 0.0 章 model 层的测试（`oscillator_regimes` / `heat_ftcs`）。
 *
 * 夹具用哪种方式？—— **两种都用，各司其职**（这是本文件最重要的一个决定）
 * ---------------------------------------------------------------------
 * 1. **手搓帧（主夹具，永远运行）**：用 `new Frame(...)` + `encodeNlf` → `decodeNlf`
 *    造出与 `server/chapters/0.0.py` **字段集合完全一致**的帧（数组名 / dtype / shape /
 *    unit / label / meta 键 / strings 键 / critical 键 / pitfalls 全照抄后端）。
 *    这样 `npm test`（门禁 G5）在没有 Python、没有 numpy 的机器上也自足，
 *    不可能因为 `.cache/` 没生成而变红或静默跳过。
 * 2. **真实帧（对拍夹具，`.cache/frames/*.nlf` 在就跑）**：`python .cache/export_frames.py`
 *    调 `server.frames.build_frame` 写出真实 `.nlf`，测试里 `decodeNlf` 读。
 *    "手搓夹具 ↔ 真实帧"的**字段集合逐项相等**由测试断言（数组名/顺序/dtype/shape/
 *    unit/label，以及 meta/strings/critical 的键集合），所以字段名不可能悄悄漂：
 *    后端一改名，第 44 组测试立刻红，而修复动作就是把 `.cache/` 重新导出并更新手搓夹具。
 *    重新生成（仓库根目录执行，脚本是临时文件、不进 git）：
 *
 *     ```python
 *     # .cache/export_frames.py —— 用 build_frame 造帧、pack_frame 落盘
 *     from server.registry import default_registry
 *     from server.frames import build_frame
 *     from server.nlf import pack_frame
 *     entry = default_registry().get("0.0")
 *     open(".cache/frames/x.nlf", "wb").write(pack_frame(build_frame(entry, "heat_ftcs", {"nx": 41, "r": 0.4, "t_end": 0.05}, 0.05)))
 *     ```
 *
 * `meta` 缺字段的降级行为（明确决定，测试逐条守）
 * ------------------------------------------------
 * | 缺失 | 行为 |
 * |---|---|
 * | 被引用的数组（`t`/`x`/`residual`；`x`/`u`/`exact`/`error`） | **抛 `ModelInputError`**（画不出来，不画半张图） |
 * | 数组长度不一致 | **抛 `ModelInputError`**（错位的图比不画更坏） |
 * | `frame.pitfalls` 为空/过短 | **抛 `ModelInputError`**（规则 7，model 不自行编造） |
 * | 整帧没有任何可绘有限值 | **抛 `ModelInputError`**（拒绝返回空场景） |
 * | `meta.t` | 回落 `opts.t` + 降级角标点名 |
 * | `meta.zeta` / `meta.r` | **不画**临界徽标 + 降级角标点名（不猜数值） |
 * | `meta.t_actual` / `meta.dt` | 不做请求时刻比对 + 降级角标点名（不假设相等） |
 * | `strings.regime_note` / `strings.stability` | 写明"帧未给出"，**不按 ζ / r 自行判断** |
 * | `frame.critical[name]` | 回落到 `manifest.gen.ts` 声明的临界值（1.0 / 0.5） |
 * | `opts.history` 可用帧 < 2 | **不画** heat 层（不是画空层）+ 说明文本 |
 * | 数组里的 NaN/±∞ | 不进 line 层（按有限段断开），计数进 `overflow.note` + 溢出角标 |
 */

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

import { linspace } from "../shared/src/axes.js";
import { countNonFinite } from "../shared/src/geometry.js";
import { Frame, type ArraySpec, type Dtype, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { sceneToJson, validateScene, type Layer, type Scene } from "../shared/src/scene.js";
import { heatFtcsModel } from "../chapters/0.0/heat_ftcs.model.js";
import { manifest, modelFor, models } from "../chapters/0.0/index.js";
import { oscillatorModel } from "../chapters/0.0/oscillator.model.js";

// --------------------------------------------------------------------------- //
// 夹具
// --------------------------------------------------------------------------- //

function fail(message: string): never {
  throw new Error(message);
}

const OSC_SPEC = manifest.vizzes.find((v) => v.id === "oscillator_regimes") ?? fail("manifest 缺少 oscillator_regimes");
const HEAT_SPEC = manifest.vizzes.find((v) => v.id === "heat_ftcs") ?? fail("manifest 缺少 heat_ftcs");

/** 与真实后端帧同形：801 点、t_span = 3 个周期（ω=1）。数值是测试用的合成值，不是物理计算。 */
const OSC_T = linspace(0, 18.84955592153876, 801);
const OSC_X = Array.from(OSC_T, (t) => 1 - t / 20);
/** 噪声形状的残差：首点一个大尖峰（与真实帧一样来自中心差分在 t=0 处的单边差分）。 */
const OSC_RESIDUAL = Array.from(OSC_T, (_, i) => (i === 0 ? 1.5e-5 : 1e-10 * (1 + (i % 7))));
const HEAT_X = linspace(0, 1, 41);
const HEAT_U0 = Array.from(HEAT_X, (x) => 0.5 * Math.sin(Math.PI * x));

interface ArrayDef {
  name: string;
  data: ArrayLike<number>;
  dtype?: Dtype;
  unit?: string;
  label?: string;
}

interface FrameDef {
  chapter: string;
  viz: string;
  arrays: ArrayDef[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical: Record<string, number | number[]>;
  pitfalls: string;
}

/** 去掉若干键（造"缺字段"的帧；置 `undefined` 会被 `.nlf` 写端拒绝，必须真删）。 */
function omit<T extends object>(rec: T, keys: readonly string[]): T {
  const out = { ...rec } as Record<string, unknown>;
  for (const k of keys) delete out[k];
  return out as T;
}

function toTyped(data: ArrayLike<number>, dtype: Dtype): TypedArray {
  switch (dtype) {
    case "f8":
      return Float64Array.from(data as ArrayLike<number>);
    case "f4":
      return Float32Array.from(data as ArrayLike<number>);
    case "i4":
      return Int32Array.from(data as ArrayLike<number>);
    case "u1":
      return Uint8Array.from(data as ArrayLike<number>);
    default:
      throw new Error(`夹具里未实现的 dtype：${dtype}`);
  }
}

/** 造帧 → 编码成 `.nlf` → 解码回来：夹具本身就是**合法的线上帧**，不是内存里的随手对象。 */
async function makeFrame(def: FrameDef): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const a of def.arrays) {
    const dtype = a.dtype ?? "f8";
    const data = toTyped(a.data, dtype);
    specs.push({
      name: a.name,
      dtype,
      shape: [a.data.length],
      order: "C",
      offset: 0,
      nbytes: data.byteLength,
      unit: a.unit ?? "",
      label: a.label ?? "",
    });
    arrays.set(a.name, data);
  }
  const frame = new Frame({
    chapter: def.chapter,
    viz: def.viz,
    kind: "frame",
    codec: "none",
    meta: def.meta,
    strings: def.strings,
    critical: def.critical,
    pitfalls: def.pitfalls,
    specs,
    arrays,
  });
  return decodeNlf(await encodeNlf(frame));
}

interface OscOverride {
  /** 覆盖数组数据（键必须是帧里已有的数组名）。 */
  arrays?: Record<string, ArrayLike<number>>;
  /** 删掉若干数组（造"缺数组"）。 */
  drop?: string[];
  meta?: Record<string, number | boolean | null>;
  dropMeta?: string[];
  strings?: Record<string, string>;
  dropStrings?: string[];
  critical?: Record<string, number | number[]>;
  pitfalls?: string;
}

/** 与 `server/chapters/0.0.py::_oscillator_frame` 字段集合一致的帧（数值是测试用的合成值）。 */
async function oscFrame(over: OscOverride = {}): Promise<Frame> {
  const base: ArrayDef[] = [
    { name: "t", data: OSC_T, unit: "s", label: "时间" },
    { name: "x", data: OSC_X, unit: "1", label: "位移" },
    { name: "v", data: Array.from(OSC_T, () => -0.05), unit: "1/s", label: "速度" },
    { name: "energy", data: Array.from(OSC_T, () => 0.5), unit: "1", label: "比能量 ½(v²+ω²x²)" },
    { name: "residual", data: OSC_RESIDUAL, unit: "1", label: "ODE 残差" },
  ];
  const arrays = base
    .filter((a) => !(over.drop ?? []).includes(a.name))
    .map((a) => (over.arrays?.[a.name] !== undefined ? { ...a, data: over.arrays[a.name] } : a));
  // 覆盖里给了、但基础定义里没有的名字（不删的情况下）也加进去
  for (const [name, data] of Object.entries(over.arrays ?? {})) {
    if (!arrays.some((a) => a.name === name)) arrays.push({ name, data, unit: "1", label: name });
  }
  const meta = omit(
    {
      omega: 1.0,
      zeta: 0.05,
      n_periods: 3,
      t_span: 18.84955592153876,
      energy_drift: 0.0,
      residual_max: 1.5e-5,
      t: 0.5,
      ...over.meta,
    },
    over.dropMeta ?? [],
  );
  return makeFrame({
    chapter: "0.0",
    viz: "oscillator_regimes",
    arrays,
    meta,
    strings: omit(
      {
        regime: "underdamped",
        regime_note: "欠阻尼：解振荡衰减，包络 e^(−ζωt)",
        data_source: "backend",
        params_json: '{"n_periods":3,"omega":1.0,"zeta":0.05}',
        ...over.strings,
      },
      over.dropStrings ?? [],
    ),
    critical: { zeta: 1.0, ...over.critical },
    pitfalls: over.pitfalls ?? OSC_SPEC.pitfalls,
  });
}

interface HeatOverride {
  arrays?: Record<string, ArrayLike<number>>;
  drop?: string[];
  meta?: Record<string, number | boolean | null>;
  dropMeta?: string[];
  strings?: Record<string, string>;
  dropStrings?: string[];
  critical?: Record<string, number | number[]>;
  pitfalls?: string;
  /** 数值解整体缩放（造"r>1/2 后爆炸"的帧）。 */
  scale?: number;
}

/** 与 `server/chapters/0.0.py::_heat_frame` 字段集合一致的帧。 */
async function heatFrame(over: HeatOverride = {}): Promise<Frame> {
  const scale = over.scale ?? 1;
  const u = Array.from(HEAT_U0, (v) => v * scale);
  const exact = Array.from(HEAT_U0, (v) => v * 1.001);
  // 两端 Dirichlet 0：sin(π) 在浮点下是 1.2e-16，这里显式置 0，让"0 点画不出"的边界情形可控
  for (const arr of [u, exact]) {
    arr[0] = 0;
    arr[arr.length - 1] = 0;
  }
  const error = u.map((v, i) => v - (exact[i] as number));
  const base: ArrayDef[] = [
    { name: "x", data: HEAT_X, unit: "1", label: "空间坐标" },
    { name: "u", data: u, unit: "1", label: "数值解" },
    { name: "exact", data: exact, unit: "1", label: "精确解" },
    { name: "error", data: error, unit: "1", label: "误差" },
  ];
  const arrays = base
    .filter((a) => !(over.drop ?? []).includes(a.name))
    .map((a) => (over.arrays?.[a.name] !== undefined ? { ...a, data: over.arrays[a.name] } : a));
  for (const [name, data] of Object.entries(over.arrays ?? {})) {
    if (!arrays.some((a) => a.name === name)) arrays.push({ name, data, unit: "1", label: name });
  }
  const meta = omit(
    {
      dt: 2.5e-4,
      n_steps: 200,
      r: 0.4,
      max_error: 2.17e-4,
      l2_error: 1.52e-4,
      t_actual: 0.05,
      t_requested: 0.05,
      t: 0.05,
      ...over.meta,
    },
    over.dropMeta ?? [],
  );
  return makeFrame({
    chapter: "0.0",
    viz: "heat_ftcs",
    arrays,
    meta,
    strings: omit(
      {
        stability: "stable",
        data_source: "backend",
        params_json: '{"nx":11,"r":0.4,"t_end":0.05}',
        ...over.strings,
      },
      over.dropStrings ?? [],
    ),
    critical: { r: 0.5, ...over.critical },
    pitfalls: over.pitfalls ?? HEAT_SPEC.pitfalls,
  });
}

function oscOpts(over: Partial<ModelOptions> = {}): ModelOptions {
  return { width: 900, height: 520, t: 0.5, params: { omega: 1, zeta: 0.05, n_periods: 3 }, ...over };
}

function heatOpts(over: Partial<ModelOptions> = {}): ModelOptions {
  return { width: 900, height: 520, t: 0.05, params: { nx: 11, r: 0.4, t_end: 0.05 }, ...over };
}

// —— 常用断言 ——

function assertValid(scene: Scene): void {
  assert.deepEqual(validateScene(scene), [], `Scene 应合法，但有问题：${validateScene(scene).join("；")}`);
}

function layersOf(scene: Scene, kind: Layer["kind"]): Layer[] {
  return scene.layers.filter((l) => l.kind === kind);
}

function lineSeries(scene: Scene, axes: string): { x: ArrayLike<number>; y: ArrayLike<number>; color: string; dash?: number[]; label?: string }[] {
  const out: { x: ArrayLike<number>; y: ArrayLike<number>; color: string; dash?: number[]; label?: string }[] = [];
  for (const layer of scene.layers) {
    if (layer.kind === "line" && layer.axes === axes) out.push(...layer.series);
  }
  return out;
}

/** 某个轴上第 `index` 个 line 图层的 series（把"数据层"与"参考线层"分开数）。 */
function seriesOfLayer(scene: Scene, axes: string, index: number): { x: ArrayLike<number>; y: ArrayLike<number>; label?: string }[] {
  const layers = scene.layers.filter((l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === axes);
  assert.ok(layers[index], `轴 ${axes} 上应有第 ${index} 个 line 图层`);
  return layers[index].series;
}

function allSeriesFinite(scene: Scene): boolean {
  return scene.layers
    .filter((l): l is Extract<Layer, { kind: "line" }> => l.kind === "line")
    .every((l) => l.series.every((s) => Array.from(s.y).every((v) => Number.isFinite(v))));
}

function textsOf(scene: Scene): string {
  return scene.texts.map((t) => t.text).join("\n");
}

function badgesOf(scene: Scene, kind: string): string[] {
  return scene.badges.filter((b) => b.kind === kind).map((b) => b.text);
}

const OSC_DEFAULT = await oscFrame();
const HEAT_DEFAULT = await heatFrame();
const heatHistory = async (): Promise<{ t: number; frame: Frame }[]> => [
  { t: 0.03, frame: await heatFrame({ scale: 0.64, meta: { t: 0.03 } }) },
  { t: 0.01, frame: await heatFrame({ scale: 0.8, meta: { t: 0.01 } }) },
  { t: 0.05, frame: HEAT_DEFAULT },
];

// --------------------------------------------------------------------------- //
// 1. oscillator：结构
// --------------------------------------------------------------------------- //

test("oscillator：默认帧产出合法 Scene，章节/viz/来源逐项对", () => {
  const scene = oscillatorModel(OSC_DEFAULT, oscOpts());
  assertValid(scene);
  assert.equal(scene.chapter, "0.0");
  assert.equal(scene.viz, "oscillator_regimes");
  assert.equal(scene.dataSource, "backend");
  assert.equal(scene.t, 0.5);
  assert.equal(scene.title, OSC_SPEC.title);
});

test("oscillator：轴 —— 主面板固定线性、误差面板对数且自适应（规则 1/3/4）", () => {
  const scene = oscillatorModel(OSC_DEFAULT, oscOpts());
  assert.deepEqual(scene.axes.map((a) => a.id), ["main", "error"]);
  const [main, error] = scene.axes;
  assert.equal(main.autoscaled, false);
  assert.equal(main.xscale, "linear");
  assert.equal(main.yscale, "linear");
  assert.equal(error.autoscaled, true);
  assert.equal(error.yscale, "log");
  assert.ok(error.ylim[0] > 0, "对数轴下界必须 > 0");
  assert.equal(error.title, "ODE 残差");
  assert.equal(error.ylabel, "$|x''+2\\zeta\\omega x'+\\omega^2x|$");
});

test("oscillator：图层顺序 = 游标 / 全曲线 / 当前标记 / 残差 / 参考线", () => {
  const scene = oscillatorModel(OSC_DEFAULT, oscOpts());
  assert.deepEqual(
    scene.layers.map((l) => [l.kind, l.axes]),
    [
      ["line", "main"],
      ["line", "main"],
      ["scatter", "main"],
      ["line", "error"],
      ["line", "error"],
    ],
  );
});

test("oscillator：x(t) 全曲线来自帧数组 t/x，点数等于帧内样本数（不抽稀）", () => {
  const scene = oscillatorModel(OSC_DEFAULT, oscOpts());
  const main = lineSeries(scene, "main");
  const curve = main[1];
  assert.equal(curve.x.length, OSC_T.length);
  assert.equal(curve.y.length, OSC_T.length);
  assert.equal(curve.x[0], 0);
  assert.equal(curve.y[0], 1);
  assert.ok(typeof curve.label === "string" && curve.label.length > 0, "曲线要带图例标签");
});

test("oscillator：当前时刻标记落在 meta.t 上，y 为帧内线性插值", async () => {
  const frame = await oscFrame({ arrays: { x: [0, 10, 0], t: [0, 1, 2], residual: [1e-10, 2e-10, 3e-10] }, meta: { t: 1.5 } });
  const scene = oscillatorModel(frame, oscOpts({ t: 1.5 }));
  const scatter = layersOf(scene, "scatter")[0];
  assert.ok(scatter && scatter.kind === "scatter");
  assert.deepEqual(Array.from(scatter.x), [1.5]);
  assert.equal(Array.from(scatter.y)[0], 5, "t=1.5 处线性插值应是 5（不是端点值）");
  assert.ok(textsOf(scene).includes("t = 1.5"), "标记旁应写出当前时刻");

  const linear = oscillatorModel(OSC_DEFAULT, oscOpts({ t: 0.5 }));
  const marker = layersOf(linear, "scatter")[0];
  assert.ok(marker && marker.kind === "scatter");
  assert.ok(Math.abs((Array.from(marker.y)[0] as number) - (1 - 0.5 / 20)) < 1e-12, "线性解上插值应与解析值一致");
});

test("oscillator：帧声明的时刻超出帧时间范围时不画标记，并写明原因", async () => {
  // meta.t = 25 但帧的 t 数组只到 18.85：帧自己说不清"当前"在哪 ⇒ 不外推，只报缺陷
  const frame = await oscFrame({ meta: { t: 25 } });
  const scene = oscillatorModel(frame, oscOpts({ t: 25 }));
  assert.equal(layersOf(scene, "scatter").length, 0, "帧外不插值、不外推");
  assert.equal(lineSeries(scene, "main").length, 1, "主面板只剩全曲线（不画游标）");
  const status = badgesOf(scene, "status").join("；");
  assert.match(status, /不在帧时间范围/);
  assertValid(scene);
});

test("oscillator：当前时刻以帧 meta.t 为准（与请求 t 不一致时也不改用请求值）", () => {
  const scene = oscillatorModel(OSC_DEFAULT, oscOpts({ t: 3 }));
  const marker = layersOf(scene, "scatter")[0];
  assert.ok(marker && marker.kind === "scatter");
  assert.deepEqual(Array.from(marker.x), [0.5], "帧 meta.t=0.5 才是这一帧真正对应的时刻");
});

test("oscillator：regime 说明逐字来自 strings.regime_note（不自己判分支）", async () => {
  for (const zeta of [0.05, 1.0, 1.5]) {
    const note = `分支说明-${zeta}`;
    const frame = await oscFrame({ meta: { zeta }, strings: { regime_note: note } });
    const scene = oscillatorModel(frame, oscOpts());
    assert.ok(textsOf(scene).includes(note), `ζ=${zeta} 时必须原样显示帧给的说明`);
  }
});

test("oscillator：残差逐点画出（未平滑、未截断），NaN 处断开成多段", async () => {
  const scene = oscillatorModel(OSC_DEFAULT, oscOpts());
  const residual = lineSeries(scene, "error")[0];
  assert.equal(residual.y.length, OSC_RESIDUAL.length, "全部样本都要画出来");

  const damaged = Array.from(OSC_RESIDUAL);
  damaged[3] = NaN;
  const frame = await oscFrame({ arrays: { residual: damaged } });
  const scene2 = oscillatorModel(frame, oscOpts());
  const split = seriesOfLayer(scene2, "error", 0);
  assert.equal(split.length, 2, "一个 NaN 应把残差断成两段");
  assert.equal(split[0].y.length, 3);
  assert.equal(split[1].y.length, OSC_RESIDUAL.length - 4);
  assert.ok(allSeriesFinite(scene2), "line 层里不允许残留非有限值");
  assert.ok(seriesOfLayer(scene2, "error", 1).length === 1, "参考线仍在（它有自己的图层）");
});

test("oscillator：残差参考线取帧给的 meta.residual_max，不是前端算的", async () => {
  const scene = oscillatorModel(OSC_DEFAULT, oscOpts());
  const ref = lineSeries(scene, "error")[1];
  assert.deepEqual(Array.from(ref.y), [1.5e-5, 1.5e-5]);
  assert.match(String(ref.label), /residual_max/);

  const noMax = await oscFrame({ dropMeta: ["residual_max"] });
  const scene2 = oscillatorModel(noMax, oscOpts());
  assert.equal(lineSeries(scene2, "error").length, 1, "没有该标量就不画参考线");
});

test("oscillator：误差轴是 makeErrorAxis（对数、自适应、ylim[0]>0）", () => {
  const scene = oscillatorModel(OSC_DEFAULT, oscOpts());
  const error = scene.axes.find((a) => a.id === "error");
  assert.ok(error);
  assert.equal(error.yscale, "log");
  assert.equal(error.autoscaled, true);
  assert.ok(error.ylim[0] > 0 && error.ylim[1] > error.ylim[0]);
  assert.ok(error.yticks.length >= 2);
});

test("oscillator：徽标齐备 —— 来源 / 轴模式 / 误差轴自适应 / ζ 临界（规则 4/5/6）", () => {
  const scene = oscillatorModel(OSC_DEFAULT, oscOpts());
  const source = badgesOf(scene, "source");
  assert.equal(source.length, 1);
  assert.match(source[0], /^数据：后端 \.nlf 帧/);
  assert.deepEqual(badgesOf(scene, "autoscale").length, 2, "主面板固定 + 误差轴自适应各一条");
  assert.ok(badgesOf(scene, "autoscale").some((t) => t.includes("固定")));
  assert.ok(badgesOf(scene, "autoscale").some((t) => t.includes("误差轴")));
  const critical = badgesOf(scene, "critical");
  assert.equal(critical.length, 1);
  assert.match(critical[0], /ζ = 0\.05000/);
  assert.match(critical[0], /理论临界 1\.000/);
  assert.ok(!critical[0].includes("已超过临界"), "ζ=0.05 < 1 不应标越界");
});

test("oscillator：临界值取自帧 critical（与 manifest 声明一致时回落值相同）", async () => {
  const declared = OSC_SPEC.params.find((p) => p.name === "zeta")?.critical;
  assert.deepEqual(declared, [1.0], "回落常量 1.0 必须与 manifest 的声明一致");
  const frame = await oscFrame({ critical: { zeta: 0.75 } });
  const scene = oscillatorModel(frame, oscOpts());
  assert.match(badgesOf(scene, "critical")[0], /理论临界 0\.7500/);
});

test("oscillator：scene.pitfalls 逐字来自 frame.pitfalls（规则 7）", async () => {
  const frame = await oscFrame({ pitfalls: "这是后端帧自带的陷阱说明，前端一个字都不许改。" });
  const scene = oscillatorModel(frame, oscOpts());
  assert.equal(scene.pitfalls, frame.pitfalls);
  assertValid(scene);
});

test("oscillator：dataSource=backend+preview 时另挂 preview 徽标（source 仍在）", async () => {
  const frame = await oscFrame({ strings: { data_source: "backend+preview" } });
  const scene = oscillatorModel(frame, oscOpts());
  assert.equal(scene.dataSource, "backend+preview");
  // kind 的唯一性：source 徽标回答"数据从哪来"（规则 5 的硬要求），preview 徽标回答"这是近似"。
  assert.equal(badgesOf(scene, "source").length, 1, "规则 5：无论是否 preview，都必须有 source 徽标");
  assert.match(badgesOf(scene, "source")[0], /近似预览/);
  assert.match(badgesOf(scene, "preview")[0], /前端近似/);
  assertValid(scene);
});

test("oscillator：dark 主题只改配色，不改结构", () => {
  const light = oscillatorModel(OSC_DEFAULT, oscOpts());
  const dark = oscillatorModel(OSC_DEFAULT, oscOpts({ theme: "dark" }));
  assert.notEqual(dark.canvas.background, light.canvas.background);
  assert.deepEqual(
    dark.layers.map((l) => l.kind),
    light.layers.map((l) => l.kind),
  );
  assert.deepEqual(dark.axes.map((a) => a.id), light.axes.map((a) => a.id));
  assertValid(dark);
});

// --------------------------------------------------------------------------- //
// 2. oscillator：NaN / ∞ / 溢出（规则 1）
// --------------------------------------------------------------------------- //

test("oscillator：NaN/∞ 不进 line 层，但全部进 overflow.note 与溢出角标", async () => {
  const x = Array.from(OSC_X);
  const residual = Array.from(OSC_RESIDUAL);
  const tt = Array.from(OSC_T);
  x[5] = Number.POSITIVE_INFINITY;
  residual[3] = NaN;
  tt[7] = NaN;
  const frame = await oscFrame({ arrays: { x, residual, t: tt } });
  const scene = oscillatorModel(frame, oscOpts());
  assertValid(scene);
  assert.ok(allSeriesFinite(scene), "line 层必须只剩有限值（分段的职责）");
  assert.equal(scene.overflow.flag, true);
  assert.match(scene.overflow.note, /非有限/);
  assert.match(scene.overflow.note, /3 个非有限值/);
  assert.equal(badgesOf(scene, "overflow").length, 1);
  assert.match(badgesOf(scene, "overflow")[0], /非有限/);
  // 曲线在 ∞（x）与 NaN（t）处各断开一次：可绘点数 = 801 − 2
  const curve = seriesOfLayer(scene, "main", 1);
  assert.ok(curve.length >= 2, "断点两侧各成一段");
  const drawn = curve.reduce((n, seg) => n + seg.x.length, 0);
  assert.equal(drawn, OSC_X.length - 2);
});

test("oscillator：数据冲出固定纵轴时 overflow.flag=true 且角标说明越界", async () => {
  // residual_max 被故意给成比数据大两个数量级：参考线会画到轴外，规则 1 必须报出来
  const frame = await oscFrame({ meta: { residual_max: 1e6 } });
  const scene = oscillatorModel(frame, oscOpts());
  assertValid(scene);
  assert.equal(scene.overflow.flag, true);
  assert.match(scene.overflow.note, /超出轴/);
  assert.equal(badgesOf(scene, "overflow").length, 1);
});

test("oscillator：没有溢出时不挂溢出角标", () => {
  const scene = oscillatorModel(OSC_DEFAULT, oscOpts());
  assert.equal(scene.overflow.flag, false);
  assert.equal(scene.overflow.note, "");
  assert.equal(badgesOf(scene, "overflow").length, 0);
});

// --------------------------------------------------------------------------- //
// 3. oscillator：缺字段的降级 / 抛错
// --------------------------------------------------------------------------- //

test("oscillator：缺 meta.zeta → 不画临界徽标 + 降级角标点名（不猜）", async () => {
  const frame = await oscFrame({ dropMeta: ["zeta"] });
  const scene = oscillatorModel(frame, oscOpts());
  assertValid(scene);
  assert.equal(badgesOf(scene, "critical").length, 0);
  assert.match(badgesOf(scene, "status").join("；"), /meta 缺 zeta/);
});

test("oscillator：缺 meta.t → scene.t 回落请求 t + 降级角标", async () => {
  const frame = await oscFrame({ dropMeta: ["t"] });
  const scene = oscillatorModel(frame, oscOpts({ t: 1.25 }));
  assert.equal(scene.t, 1.25);
  assert.match(badgesOf(scene, "status").join("；"), /meta 缺 t/);
  assertValid(scene);
});

test("oscillator：缺 strings.regime_note → 写明「帧未给出」，不按 ζ 猜分支", async () => {
  const frame = await oscFrame({ dropStrings: ["regime_note"] });
  const scene = oscillatorModel(frame, oscOpts());
  assert.match(textsOf(scene), /帧未给出 regime_note/);
  assert.match(badgesOf(scene, "status").join("；"), /regime_note/);
  assertValid(scene);
});

test("oscillator：缺必需数组 residual / x → 抛 ModelInputError", async () => {
  const noResidual = await oscFrame({ drop: ["residual"] });
  assert.throws(
    () => oscillatorModel(noResidual, oscOpts()),
    (err: unknown) => err instanceof ModelInputError && /residual/.test((err as Error).message),
  );
  const noX = await oscFrame({ drop: ["x"] });
  assert.throws(
    () => oscillatorModel(noX, oscOpts()),
    (err: unknown) => err instanceof ModelInputError && /'x'/.test((err as Error).message),
  );
});

test("oscillator：t 与 x 长度不一致 → 抛 ModelInputError（不错位画图）", async () => {
  const frame = await oscFrame({ arrays: { x: [1, 2, 3] } });
  assert.throws(() => oscillatorModel(frame, oscOpts()), ModelInputError);
});

test("oscillator：pitfalls 为空 → 抛 ModelInputError（前端不编陷阱说明）", async () => {
  const frame = await oscFrame({ pitfalls: "" });
  assert.throws(
    () => oscillatorModel(frame, oscOpts()),
    (err: unknown) => err instanceof ModelInputError && /pitfalls/.test((err as Error).message),
  );
});

test("oscillator：整帧无有限值 → 抛 ModelInputError（拒绝返回空场景）", async () => {
  const nan = Array.from(OSC_T, () => NaN);
  const frame = await oscFrame({ arrays: { x: nan, residual: nan } });
  assert.throws(
    () => oscillatorModel(frame, oscOpts()),
    (err: unknown) => err instanceof ModelInputError && /没有任何可绘/.test((err as Error).message),
  );
});

test("oscillator：残差全为 0 → 对数轴退化范围 + 降级说明（不静默）", async () => {
  const frame = await oscFrame({ arrays: { residual: Array.from(OSC_T, () => 0) } });
  const scene = oscillatorModel(frame, oscOpts());
  assertValid(scene);
  const error = scene.axes.find((a) => a.id === "error");
  assert.deepEqual(error?.ylim, [1e-16, 1]);
  assert.match(badgesOf(scene, "status").join("；"), /residual/);
  assert.match(textsOf(scene), /对数轴画不出|非有限/);
});

// --------------------------------------------------------------------------- //
// 4. heat_ftcs：结构
// --------------------------------------------------------------------------- //

test("heat_ftcs：默认帧产出合法 Scene，章节/viz/时间对", () => {
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts());
  assertValid(scene);
  assert.equal(scene.chapter, "0.0");
  assert.equal(scene.viz, "heat_ftcs");
  assert.equal(scene.t, 0.05);
  assert.equal(scene.title, HEAT_SPEC.title);
});

test("heat_ftcs：u 与 exact 同轴，靠颜色/虚线/标签区分（图例分得清）", () => {
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts());
  const main = lineSeries(scene, "main");
  assert.equal(main.length, 2);
  const [u, exact] = main;
  assert.notEqual(u.color, exact.color);
  assert.equal(u.dash, undefined);
  assert.ok(Array.isArray(exact.dash) && exact.dash.length > 0, "精确解用虚线");
  assert.ok(u.label && exact.label && u.label !== exact.label);
});

test("heat_ftcs：误差副轴画 |error| 且为对数轴；=0 的点画不出但报数", () => {
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts());
  const error = scene.axes.find((a) => a.id === "error");
  assert.equal(error?.yscale, "log");
  assert.equal(error?.autoscaled, true);
  assert.ok((error?.ylim[0] ?? 0) > 0);
  const seriesErr = lineSeries(scene, "error")[0];
  assert.equal(seriesErr.y.length, HEAT_X.length - 2, "两端 |error|=0 的点在对数轴上画不出");
  assert.ok(Array.from(seriesErr.y).every((v) => v > 0));
  assert.match(textsOf(scene), /2 点为 0（对数轴画不出）/);
});

test("heat_ftcs：主面板纵轴固定，不随数值解爆炸自适应（规则 1/4）", async () => {
  const exploding = await heatFrame({ scale: 1e6, meta: { r: 0.7 }, strings: { stability: "unstable" } });
  const scene = heatFtcsModel(exploding, heatOpts());
  const main = scene.axes.find((a) => a.id === "main");
  assert.deepEqual(main?.ylim, [-0.05, 1.05]);
  assert.equal(main?.autoscaled, false);
  assert.equal(scene.overflow.flag, true, "冲出画面必须显式标记，而不是缩轴藏起来");
  assert.match(badgesOf(scene, "overflow")[0], /超出轴/);
  assertValid(scene);
});

test("heat_ftcs：history ≥2 帧 → heat 层；nx/ny/extent 与采样约定一致", async () => {
  const history = await heatHistory();
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts({ history }));
  assertValid(scene);
  const heat = layersOf(scene, "heat");
  assert.equal(heat.length, 1);
  assert.ok(heat[0].kind === "heat");
  assert.equal(heat[0].nx, HEAT_X.length);
  assert.equal(heat[0].ny, history.length);
  assert.equal(heat[0].values.length, HEAT_X.length * history.length);
  assert.deepEqual(heat[0].extent, [0, 1, 0.01, 0.05]);
  assert.equal(scene.axes.find((a) => a.id === "xt")?.yscale, "linear");
});

test("heat_ftcs：heat 层行主序（第 j 行 = 第 j 个历史帧的 u）", async () => {
  const history = await heatHistory();
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts({ history }));
  const heat = layersOf(scene, "heat")[0];
  assert.ok(heat.kind === "heat");
  const sorted = [...history].sort((a, b) => a.t - b.t);
  for (let j = 0; j < sorted.length; j += 1) {
    for (let i = 0; i < HEAT_X.length; i += 1) {
      assert.equal(heat.values[j * heat.nx + i], sorted[j].frame.array("u")[i], `第 ${j} 行第 ${i} 列应取该帧的 u`);
    }
  }
});

test("heat_ftcs：history 为空 → 不画 heat 层，并写明为什么", () => {
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts({ history: [] }));
  assertValid(scene);
  assert.equal(layersOf(scene, "heat").length, 0);
  assert.equal(scene.axes.some((a) => a.id === "xt"), false, "不画空轴");
  assert.match(textsOf(scene), /history 为空/);
});

test("heat_ftcs：history 只有 1 帧 → 不画 heat 层", async () => {
  const one = [{ t: 0.03, frame: await heatFrame({ meta: { t: 0.03 } }) }];
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts({ history: one }));
  assertValid(scene);
  assert.equal(layersOf(scene, "heat").length, 0);
  assert.match(textsOf(scene), /不足 2 帧/);
});

test("heat_ftcs：history 空间网格不一致 → 该帧被跳过，不够 2 帧就不画", async () => {
  const other = await heatFrame({ arrays: { x: linspace(0, 2, 11) }, meta: { t: 0.02 } });
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts({ history: [{ t: 0.02, frame: other }] }));
  assertValid(scene);
  assert.equal(layersOf(scene, "heat").length, 0);
  assert.match(textsOf(scene), /不足 2 帧/);
});

test("heat_ftcs：history 时间不递增 → 不画（避免退化 extent）", async () => {
  const a = await heatFrame({ scale: 0.8, meta: { t: 0.05 } });
  const b = await heatFrame({ scale: 0.9, meta: { t: 0.05 } });
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts({ history: [{ t: 0.05, frame: a }, { t: 0.05, frame: b }] }));
  assertValid(scene);
  assert.equal(layersOf(scene, "heat").length, 0);
});

test("heat_ftcs：history 乱序也给对的 extent（内部按 t 排序）", async () => {
  const frames = [
    { t: 0.05, frame: await heatFrame({ scale: 0.6, meta: { t: 0.05 } }) },
    { t: 0.01, frame: await heatFrame({ scale: 1.0, meta: { t: 0.01 } }) },
    { t: 0.03, frame: await heatFrame({ scale: 0.8, meta: { t: 0.03 } }) },
  ];
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts({ history: frames }));
  const heat = layersOf(scene, "heat")[0];
  assert.ok(heat.kind === "heat");
  assert.deepEqual(heat.extent.slice(2), [0.01, 0.05]);
  assert.equal(heat.values[0], frames[1].frame.array("u")[0], "第一行应是 t 最小的那帧");
  assertValid(scene);
});

// --------------------------------------------------------------------------- //
// 5. heat_ftcs：临界值 / 稳定性 / t_actual
// --------------------------------------------------------------------------- //

test("heat_ftcs：r 临界徽标（未越界）+ 稳定性角标来自 strings.stability", () => {
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts());
  const critical = badgesOf(scene, "critical");
  assert.equal(critical.length, 1);
  assert.match(critical[0], /r = 0\.4000/);
  assert.match(critical[0], /理论临界 0\.5000/);
  assert.ok(!critical[0].includes("已超过临界"));
  assert.match(badgesOf(scene, "status").join("；"), /稳定性：stable/);
});

test("heat_ftcs：r 越过 1/2 → 徽标写「已超过临界」，稳定性角标写 unstable", async () => {
  const frame = await heatFrame({ meta: { r: 0.7 }, strings: { stability: "unstable" } });
  const scene = heatFtcsModel(frame, heatOpts({ params: { nx: 11, r: 0.7, t_end: 0.05 } }));
  const critical = badgesOf(scene, "critical")[0];
  assert.match(critical, /r = 0\.7000/);
  assert.match(critical, /已超过临界/);
  assert.match(badgesOf(scene, "status").join("；"), /unstable/);
  assertValid(scene);
});

test("heat_ftcs：|t_actual − t| > dt/2 时用 text 明说（真陷阱）", async () => {
  const frame = await heatFrame({ meta: { t_actual: 0.0503, dt: 2.5e-4, t: 0.05 } });
  const scene = heatFtcsModel(frame, heatOpts({ t: 0.05 }));
  const body = textsOf(scene);
  assert.match(body, /t_actual=0\.0503/);
  assert.match(body, /dt\/2=1\.25e-4/);
  assert.match(body, /不是请求时刻/);
  assertValid(scene);
});

test("heat_ftcs：|t_actual − t| ≤ dt/2 时不出现该提示（只在需要时出现）", async () => {
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts());
  assert.ok(!textsOf(scene).includes("t_actual"), "相差 1e-17 不该报警");
  const frame = await heatFrame({ meta: { t_actual: 0.0501, dt: 2.5e-4 } });
  assert.ok(!textsOf(heatFtcsModel(frame, heatOpts())).includes("t_actual"));
});

test("heat_ftcs：缺 meta.t_actual / dt → 不比对 + 降级角标点名", async () => {
  const frame = await heatFrame({ dropMeta: ["t_actual", "dt"] });
  const scene = heatFtcsModel(frame, heatOpts());
  assertValid(scene);
  assert.ok(!textsOf(scene).includes("t_actual"), "没有真值就不做比对");
  assert.match(badgesOf(scene, "status").join("；"), /t_actual/);
  assert.match(badgesOf(scene, "status").join("；"), /dt/);
});

test("heat_ftcs：缺 meta.r → 不画临界徽标 + 降级说明", async () => {
  const frame = await heatFrame({ dropMeta: ["r"] });
  const scene = heatFtcsModel(frame, heatOpts());
  assertValid(scene);
  assert.equal(badgesOf(scene, "critical").length, 0);
  assert.match(badgesOf(scene, "status").join("；"), /meta 缺 r/);
});

test("heat_ftcs：缺 strings.stability → 写明「帧 strings 缺 stability」，不自判", async () => {
  const frame = await heatFrame({ dropStrings: ["stability"] });
  const scene = heatFtcsModel(frame, heatOpts());
  assert.match(badgesOf(scene, "status").join("；"), /缺 stability/);
  assertValid(scene);
});

// --------------------------------------------------------------------------- //
// 6. heat_ftcs：NaN / ∞ / 缺数组
// --------------------------------------------------------------------------- //

test("heat_ftcs：u/exact 含 NaN/∞ 时不崩、不断言崩，计数进溢出说明", async () => {
  const u = Array.from(HEAT_U0);
  const exact = Array.from(HEAT_U0, (v) => v * 1.001);
  u[4] = NaN;
  exact[6] = Number.POSITIVE_INFINITY;
  const frame = await heatFrame({ arrays: { u, exact } });
  const scene = heatFtcsModel(frame, heatOpts());
  assertValid(scene);
  assert.ok(allSeriesFinite(scene));
  assert.equal(scene.overflow.flag, true);
  assert.match(scene.overflow.note, /2 个非有限值/);
  assert.equal(badgesOf(scene, "overflow").length, 1);
});

test("heat_ftcs：history 帧里带 NaN → 热力图原样保留（灰格），Scene 仍合法", async () => {
  const bad = Array.from(HEAT_U0);
  bad[5] = NaN;
  const frames = [
    { t: 0.01, frame: await heatFrame({ arrays: { u: bad, error: bad }, meta: { t: 0.01 } }) },
    { t: 0.05, frame: HEAT_DEFAULT },
  ];
  const scene = heatFtcsModel(HEAT_DEFAULT, heatOpts({ history: frames }));
  assertValid(scene);
  const heat = layersOf(scene, "heat")[0];
  assert.ok(heat.kind === "heat");
  assert.ok(countNonFinite(heat.values) > 0, "热力图把没数据的格子留给渲染层涂灰");
});

test("heat_ftcs：缺必需数组 error / u → 抛 ModelInputError", async () => {
  const noError = await heatFrame({ drop: ["error"] });
  assert.throws(
    () => heatFtcsModel(noError, heatOpts()),
    (err: unknown) => err instanceof ModelInputError && /error/.test((err as Error).message),
  );
  const noU = await heatFrame({ drop: ["u"] });
  assert.throws(() => heatFtcsModel(noU, heatOpts()), ModelInputError);
});

test("heat_ftcs：pitfalls 为空 → 抛 ModelInputError", async () => {
  const frame = await heatFrame({ pitfalls: "太短" });
  assert.throws(() => heatFtcsModel(frame, heatOpts()), ModelInputError);
});

// --------------------------------------------------------------------------- //
// 7. 装配层：index.ts
// --------------------------------------------------------------------------- //

test("index：models 的键与 manifest 的 viz id 一一对应（不漏不多）", () => {
  assert.deepEqual(Object.keys(models).sort(), manifest.vizzes.map((v) => v.id).sort());
  assert.equal(models["oscillator_regimes"], oscillatorModel);
  assert.equal(models["heat_ftcs"], heatFtcsModel);
});

test("index：modelFor 命中已知 id，未知 id 返回 null（不抛错）", () => {
  assert.equal(modelFor("heat_ftcs"), heatFtcsModel);
  assert.equal(modelFor("oscillator_regimes"), oscillatorModel);
  assert.equal(modelFor("nope"), null);
});

test("两个 model 都是纯函数：同一输入两次调用逐字节相同", async () => {
  const history = await heatHistory();
  const o = oscOpts();
  const h = heatOpts({ history });
  assert.equal(sceneToJson(oscillatorModel(OSC_DEFAULT, o)), sceneToJson(oscillatorModel(OSC_DEFAULT, o)));
  assert.equal(sceneToJson(heatFtcsModel(HEAT_DEFAULT, h)), sceneToJson(heatFtcsModel(HEAT_DEFAULT, h)));
});

test("两个 model 都不依赖 opts.width/height 之外的隐藏状态：尺寸变化只改矩形", () => {
  const big = oscillatorModel(OSC_DEFAULT, oscOpts({ width: 1200, height: 700 }));
  const small = oscillatorModel(OSC_DEFAULT, oscOpts({ width: 640, height: 360 }));
  assert.equal(big.canvas.w, 1200);
  assert.equal(small.canvas.h, 360);
  assert.notDeepEqual(big.axes[0].rect, small.axes[0].rect);
  assertValid(big);
  assertValid(small);
});

// --------------------------------------------------------------------------- //
// 8. 真实后端帧（.cache/frames/*.nlf；由 .cache/export_frames.py 生成）
// --------------------------------------------------------------------------- //

const REAL_DIR = new URL("../../../.cache/frames/", import.meta.url);

/** 真实帧的一行数组元信息（与 `Frame.specs` 同字段）。 */
interface ArrayRow {
  name: string;
  dtype: string;
  shape: number[];
  order: string;
  unit: string;
  label: string;
}

interface RealFieldset {
  chapter: string;
  viz: string;
  kind: string;
  codec: string;
  arrays: ArrayRow[];
  meta: Record<string, unknown>;
  strings: Record<string, string>;
  critical: Record<string, unknown>;
  pitfalls: string;
}

interface RealIndexEntry {
  path: string;
  t: number;
  fieldset: RealFieldset;
}

/** `.cache/frames/index.json`：Python 侧（`build_frame`）写出的字段集合，是"字段名不漂"的真值。 */
function loadRealIndex(): Record<string, RealIndexEntry> | null {
  const url = new URL("index.json", REAL_DIR);
  if (!existsSync(url)) return null;
  return JSON.parse(readFileSync(url, "utf8")) as Record<string, RealIndexEntry>;
}

type Loaded = { frame: Frame } | { error: string };

function loadReal(name: string): Loaded | null {
  const url = new URL(`${name}.nlf`, REAL_DIR);
  if (!existsSync(url)) return null;
  try {
    return { frame: decodeNlf(readFileSync(url)) };
  } catch (err) {
    return { error: err instanceof Error ? `${err.name}: ${err.message}` : String(err) };
  }
}

function frameOrFail(loaded: Loaded | null, name: string): Frame {
  if (loaded === null) throw new Error(`没有 .cache/frames/${name}.nlf`);
  if ("error" in loaded) throw new Error(`解码 ${name} 失败：${loaded.error}`);
  return loaded.frame;
}

const REAL_INDEX = loadRealIndex();
const REAL_OSC = loadReal("oscillator_regimes__default");
const REAL_HEAT = loadReal("heat_ftcs__default");
const REAL_HEAT_UNSTABLE = loadReal("heat_ftcs__unstable");
const realSkip =
  REAL_INDEX === null || REAL_OSC === null || REAL_HEAT === null || REAL_HEAT_UNSTABLE === null
    ? "缺少 .cache/frames/*.nlf（先跑 python .cache/export_frames.py；脚本见本文件 docstring）"
    : undefined;
const realOptions = realSkip === undefined ? {} : { skip: realSkip };

function specRows(frame: Frame): ArrayRow[] {
  return frame.specs.map((s) => ({
    name: s.name,
    dtype: s.dtype,
    shape: [...s.shape],
    order: s.order,
    unit: s.unit,
    label: s.label,
  }));
}

test("真实帧：手搓夹具与 build_frame 产物的字段集合一致（字段名/单位/标签不漂）", realOptions, () => {
  const osc = REAL_INDEX?.["oscillator_regimes__default"].fieldset;
  const heat = REAL_INDEX?.["heat_ftcs__default"].fieldset;
  assert.ok(osc && heat);

  assert.deepEqual(specRows(OSC_DEFAULT), osc.arrays, "oscillator：数组名/dtype/shape/order/unit/label 必须逐项一致");
  assert.deepEqual(specRows(HEAT_DEFAULT), heat.arrays, "heat_ftcs：数组名/dtype/shape/order/unit/label 必须逐项一致");

  // 键集合严格相等：少一个、多一个都算漂（过去 meta.regime 就是这么漂出去的）
  assert.deepEqual(Object.keys(osc.meta).sort(), Object.keys(OSC_DEFAULT.meta).sort(), "oscillator meta 键");
  assert.deepEqual(Object.keys(osc.strings).sort(), Object.keys(OSC_DEFAULT.strings).sort(), "oscillator strings 键");
  assert.deepEqual(Object.keys(osc.critical).sort(), Object.keys(OSC_DEFAULT.critical).sort(), "oscillator critical 键");

  assert.deepEqual(Object.keys(heat.meta).sort(), Object.keys(HEAT_DEFAULT.meta).sort(), "heat meta 键");
  assert.deepEqual(Object.keys(heat.strings).sort(), Object.keys(HEAT_DEFAULT.strings).sort(), "heat strings 键");
  assert.deepEqual(Object.keys(heat.critical).sort(), Object.keys(HEAT_DEFAULT.critical).sort(), "heat critical 键");

  // 规则 7 的文本由 build_frame 从 manifest 写入：夹具抄的就是同一份
  assert.equal(osc.pitfalls, OSC_DEFAULT.pitfalls);
  assert.equal(heat.pitfalls, HEAT_DEFAULT.pitfalls);
});

test("真实帧：oscillator 真实帧可解码并驱动 model（分支说明/临界徽标/残差量级）", realOptions, () => {
  const frame = frameOrFail(REAL_OSC, "oscillator_regimes__default");
  // 契约回归：分支标记必须在 strings 里（曾在 meta 里，导致 TS 解码器拒帧）
  assert.equal(frame.strings["regime"], "underdamped");
  assert.equal("regime" in frame.meta, false, "meta 只允许 number/bool/null（PROTOCOLS §1.1）");

  const scene = oscillatorModel(frame, oscOpts({ t: Number(frame.meta["t"]) }));
  assertValid(scene);
  const error = scene.axes.find((a) => a.id === "error");
  assert.ok(Math.log10((error?.ylim[1] ?? 1) / (error?.ylim[0] ?? 1)) >= 3, "真实残差横跨若干数量级，对数轴要画出来");
  assert.match(badgesOf(scene, "critical")[0], /理论临界 1\.000/);
  assert.equal(frame.strings["regime_note"], "欠阻尼：解振荡衰减，包络 e^(−ζωt)");
  assert.ok(textsOf(scene).includes("underdamped｜欠阻尼：解振荡衰减，包络 e^(−ζωt)"), `分支说明必须来自帧 strings：${textsOf(scene)}`);
  assert.equal(scene.t, 0.5);
});

test("真实帧：heat 默认帧——无 t_actual 提示、稳定性 stable、误差轴跨数量级", realOptions, () => {
  const frame = frameOrFail(REAL_HEAT, "heat_ftcs__default");
  const scene = heatFtcsModel(frame, heatOpts({ t: Number(frame.meta["t"]) }));
  assertValid(scene);
  assert.ok(!textsOf(scene).includes("t_actual"), "真实默认帧的 |t_actual−t| ≈ 1e-17，不该报警");
  assert.match(badgesOf(scene, "status").join("；"), /稳定性：stable/);
  assert.ok(textsOf(scene).includes("|误差| 41 点"), `应报出误差点数与 0 点个数：${textsOf(scene)}`);
});

test("真实帧：heat r=0.7（后端实测不稳定）——溢出 + 已超过临界 + t_actual 提示齐现", realOptions, () => {
  const frame = frameOrFail(REAL_HEAT_UNSTABLE, "heat_ftcs__unstable");
  const scene = heatFtcsModel(frame, heatOpts({ t: 0.05, params: { nx: 41, r: 0.7, t_end: 0.05 } }));
  assertValid(scene);
  assert.equal(scene.overflow.flag, true, "r>1/2 的真实帧会冲出固定纵轴");
  assert.match(badgesOf(scene, "critical")[0], /已超过临界/);
  assert.match(badgesOf(scene, "status").join("；"), /unstable/);
  assert.match(textsOf(scene), /t_actual=0\.05031/);
  assert.match(textsOf(scene), /不是请求时刻/);
});

test("真实帧：两帧真实 heat 数据拼出时空热力图（nx=41、行=时间）", realOptions, () => {
  const a = frameOrFail(REAL_HEAT, "heat_ftcs__default");
  const b = frameOrFail(REAL_HEAT_UNSTABLE, "heat_ftcs__unstable");
  const scene = heatFtcsModel(a, heatOpts({ t: Number(a.meta["t"]), history: [{ t: 0.04, frame: a }, { t: 0.05, frame: b }] }));
  assertValid(scene);
  const heat = layersOf(scene, "heat")[0];
  assert.ok(heat.kind === "heat");
  assert.equal(heat.nx, 41);
  assert.equal(heat.ny, 2);
  assert.deepEqual(heat.extent, [0, 1, 0.04, 0.05]);
  assert.equal(heat.values[heat.nx + 5], b.array("u")[5], "第二行必须来自第二个历史帧");
});
