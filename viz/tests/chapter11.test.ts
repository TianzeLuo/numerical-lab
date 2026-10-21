/**
 * 1.1 章 model 层的测试（`float_zoom` / `cancellation` / `cond_ellipsoid` / `summation_order`）。
 *
 * 夹具口径：**手搓帧**，`new Frame(...)` + `encodeNlf` → `decodeNlf`，字段集合照抄
 * `server/chapters/1.1.py`（数组名 / dtype / unit / label、meta 键、strings 键、critical 键），
 * 数值是测试用的合成值，**不是物理计算**。于是 `npm test`（门禁 G5）在没有 Python、
 * 没有 `.cache/frames` 的机器上也自足：它守"契约与降级行为"，数值真值由后端 tests/1.1 与视觉基线守。
 *
 * 被守的降级行为（每条都有断言）
 * ------------------------------
 * | 缺失 | 行为 |
 * |---|---|
 * | 必需数组（`offset` / `loss_grid` / `image_x` / `k_grid`） | **抛 `ModelInputError`**（不画半张图） |
 * | 同组数组长度不一致 | **抛 `ModelInputError`**（错位的图比不画更坏） |
 * | `frame.pitfalls` 为空或 < 10 字 | **抛 `ModelInputError`**（规则 7：不自行编造"它会怎样骗你"） |
 * | 可选标量数组（`*_mark`） | 不画该标记 + 降级角标点名 |
 * | `meta.k` / `meta.loss_mark` 等 | 不猜数值 + 降级角标点名 |
 * | `strings.note` / `floor_note` 等文案 | 不自行编造 + 降级角标点名 |
 * | 曲线里的 NaN/±∞、冲出 ±3 框的向量 | 不进图层，计数进 `overflow.note` + 溢出角标 |
 * | `critical` 键 | 回落到 `manifest.gen.ts` 声明的临界值（15.95 / 1.1e-16 / 2^53），并挂降级角标 |
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type Dtype, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { pitfallsBand, sceneToJson, textBoxOf, validateScene, type Scene } from "../shared/src/scene.js";
import { manifest, modelFor, models } from "../chapters/1.1/index.js";

// --------------------------------------------------------------------------- //
// 夹具
// --------------------------------------------------------------------------- //

const PITFALLS =
  "它会怎样骗你：其一，副轴是对数轴，误差恰好为 0 的点被地板抬起，不能读成「误差不为零」；" +
  "其二，实测曲线通常低于理论上界，那不是界太松，而是界对所有方向成立、当前方向不在最坏方向上。";
const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };
type MetaValue = number | boolean | null;

interface ArrayDef {
  name: string;
  data: ArrayLike<number>;
  dtype?: Dtype;
  unit?: string;
  label?: string;
}
interface FrameDef {
  arrays: ArrayDef[];
  meta: Record<string, MetaValue>;
  strings: Record<string, string>;
  critical?: Record<string, number | number[]>;
}

/** 数组声明（一行一个；unit/label 只做记录，模型不读它们）。 */
const A = (name: string, data: ArrayLike<number>, unit = "1", label = name): ArrayDef => ({ name, data, unit, label });
const seq = (lo: number, hi: number, n: number): Float64Array =>
  Float64Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / Math.max(1, n - 1));
const fill = (n: number, v: number): Float64Array => new Float64Array(n).fill(v);
const gen = (n: number, f: (i: number) => number): Float64Array => Float64Array.from({ length: n }, (_, i) => f(i));

function toTyped(data: ArrayLike<number>, dtype: Dtype): TypedArray {
  if (dtype === "f8") return Float64Array.from(data as ArrayLike<number>);
  if (dtype === "f4") return Float32Array.from(data as ArrayLike<number>);
  if (dtype === "i4") return Int32Array.from(data as ArrayLike<number>);
  if (dtype === "u1") return Uint8Array.from(data as ArrayLike<number>);
  throw new Error(`夹具里未实现的 dtype：${dtype}`);
}

/** 造帧 → 编码成 `.nlf` → 解码回来：夹具本身就是合法的线上帧，不是内存里的随手对象。 */
async function makeFrame(viz: string, def: FrameDef, pitfalls = PITFALLS): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const a of def.arrays) {
    const dtype = a.dtype ?? "f8";
    const data = toTyped(a.data, dtype);
    specs.push({ name: a.name, dtype, shape: [a.data.length], order: "C", offset: 0, nbytes: data.byteLength, unit: a.unit ?? "", label: a.label ?? "" });
    arrays.set(a.name, data);
  }
  const frame = new Frame({
    chapter: "1.1", viz, kind: "frame", codec: "none",
    meta: def.meta, strings: { data_source: "backend", params_json: "{}", ...def.strings },
    critical: def.critical ?? {}, pitfalls, specs, arrays,
  });
  return decodeNlf(await encodeNlf(frame));
}

interface Override {
  arrays?: Record<string, ArrayLike<number>>;
  drop?: string[];
  meta?: Record<string, MetaValue>;
  dropMeta?: string[];
  dropStrings?: string[];
  strings?: Record<string, string>;
  critical?: Record<string, number | number[]>;
  pitfalls?: string;
}

async function frameFor(viz: string, over: Override = {}): Promise<Frame> {
  const def = FIXTURES[viz];
  assert.ok(def, `没有 ${viz} 的夹具`);
  const drop = new Set(over.drop ?? []);
  const dropMeta = new Set(over.dropMeta ?? []);
  const dropStr = new Set(over.dropStrings ?? []);
  const arrays = def.arrays.filter((a) => !drop.has(a.name)).map((a) => ({ ...a, data: over.arrays?.[a.name] ?? a.data }));
  const meta = Object.fromEntries(Object.entries(def.meta).filter(([k]) => !dropMeta.has(k)));
  const strings = {
    ...Object.fromEntries(Object.entries(def.strings).filter(([k]) => !dropStr.has(k))),
    ...(over.strings ?? {}),
  };
  return makeFrame(viz, { arrays, meta: { ...meta, ...over.meta }, strings, critical: over.critical ?? def.critical }, over.pitfalls);
}

function sceneOf(frame: Frame, over: Partial<ModelOptions> = {}): Scene {
  const model = models[frame.viz];
  assert.ok(model, `models 里没有 ${frame.viz}`);
  const t = frame.meta["t"] === undefined ? 0 : Number(frame.meta["t"]);
  return model(frame, { ...OPTS, ...over, t: over.t ?? t, params: over.params ?? OPTS.params });
}

const axisOf = (scene: Scene, id: string) => scene.axes.find((a) => a.id === id) ?? assert.fail(`场景里没有轴 ${id}`);
const badgesOf = (scene: Scene, kind: string): string[] => scene.badges.filter((b) => b.kind === kind).map((b) => b.text);
/** 降级角标（`degradedBadge` 用 `kind="status"`，文本带 `DEGRADED_PREFIX = "⚠ 降级："`）。 */
const degradedText = (scene: Scene): string =>
  scene.badges.filter((b) => b.kind === "status" && b.text.includes("降级：")).map((b) => b.text).join(" ");

// —— 四个 viz 的字段表（对齐 data/frames/1.1 里的真实帧） ——

const FLOAT_ZOOM: FrameDef = {
  arrays: [
    A("offset", seq(-8, 8, 17), "ulp", "距窗口中心的距离（单位：窗口中心的 ulp）"),
    A("value", gen(17, (i) => 0.3 + (i - 8) * 5.551115123125783e-17), "1", "可表示的 double"),
    A("center_offset", [0], "ulp", "窗口中心"), A("center_y", [0.5], "1", "窗口中心"),
    A("special_offset", [1], "ulp", "0.1+0.2 的实际值"), A("special_y", [0.5], "1", "0.1+0.2 的实际值"),
    A("k_grid", seq(0, 64, 65), "1", "累加次数 k"),
    A("residual", gen(65, (i) => Math.max(5.551115123125783e-18, (i + 1) * 1.05e-16)), "1", "与数学值 k/10 的残差"),
    A("residual_sum_only", gen(65, (i) => Math.max(5.551115123125783e-18, (i + 1) * 1.1e-16)), "1", "与 k·fl(0.1) 的残差"),
    A("residual_floor", fill(65, 5.551115123125783e-18), "1", "对数轴的显示地板"),
    A("sum_value", seq(0, 6.4, 65), "1", "fl(Σ0.1) 的部分和"),
    A("residual_mark", [6.75e-15], "1", "当前位置的残差"), A("k_mark", [64], "1", "当前位置的累加次数"),
  ],
  meta: {
    center: 0.3, focus_kind: 0, k: 64, n_points: 17, residual_0102_03: 5.551115123125783e-17,
    residual_0102_03_ulps: 1, residual_exact_zero_steps: 3, residual_floor: 5.551115123125783e-18,
    residual_floor_min: 5.551115123125783e-18, residual_sum_only_exact_zero_steps: 4,
    residual_sum_only_mark: 7.105427357601002e-15, spacing_center: 5.551115123125783e-17,
    spacing_below: 5.551115123125783e-17, spacing_above: 5.551115123125783e-17,
    spacing_below_units: 1, spacing_above_units: 1, spacing_ratio: 1, special_ulps: 1, t: 64, window_ulps: 8,
  },
  strings: {
    focus: "sum",
    center_label: "0.3 的最近 double",
    special_label: "0.1+0.2 的实际值（比 0.3 大 1 个 ulp）",
    exact_center: "0.2999999999999999888977698",
    exact_special: "0.3000000000000000444089209",
    note: "0.1+0.2 存在 0.3 右边相邻的一个 double 上，两者相差 1 个 ulp。",
    spacing_note: "在同一个二进制区间内，所有相邻 double 的距离相同。",
    zero_note: "副轴是对数轴：残差恰好为 0 的步数被地板抬起显示，不能读成「误差不为零」。",
  },
  critical: {},
};

const N16 = 16;
const CANCELLATION: FrameDef = {
  arrays: [
    A("loss_grid", seq(1, 16, N16), "1", "损失位数 t = log10(M/|a−b|)"),
    A("digits_remaining", gen(N16, (i) => 15.95 - (i + 1)), "1", "剩余有效位数（先验估计）"),
    A("digits_mark", [9.30103], "1", "当前剩余有效位数"),
    A("error", gen(N16, (i) => Math.pow(10, 0.5 * (i + 1) - 16)), "1", "实测相对误差"),
    A("error_theory", gen(N16, (i) => Math.pow(10, 0.5 * (i + 1) - 16.5)), "1", "理论上界 δ_in(|a|+|b|)/|a−b|"),
    A("roundoff", fill(N16, 5.551115123125783e-17), "1", "减法本身的舍入误差"),
    A("error_mark", [3.3306690738754696e-10], "1", "当前实测相对误差"), A("loss_mark", [6], "1", "当前损失位数"),
    A("roundoff_mark", [5.551115123125783e-17], "1", "当前减法舍入误差"),
    A("a_value", [1], "1", "被减数 a"), A("b_value", [0.999999], "1", "减数 b"), A("gap_value", [1e-6], "1", "|a−b|"),
  ],
  meta: {
    a: 1, b: 0.999999, cond_subtraction: 2000000, degenerate: false, delta_in: 5e-16, digits_lost: 6.69897,
    digits_remaining: 9.30103, error_bound: 1e-9, error_floor: 5.551115123125783e-17, gap: 1e-6, input_digits: 16,
    loss_digits: 6.69897, overflow: false, perturb_ulps: 1, perturbation_amplification: 3.3306690738754696e-10,
    roundoff_exact_zero_steps: N16, sterbenz_exact: true, subtraction_relative_error: 0, t: 0,
  },
  strings: {
    exact_a: "1", exact_b: "0.9999989999999999712443355",
    note: "a−b = 1.000000e-06，输入不确定度 δ_in = 5.00e-16，先验剩余有效位 9.30 位，实测放大后相对误差 3.331e-10",
    sterbenz_note: "Sterbenz 引理：b/2 ≤ a ≤ 2b 时 a−b 在浮点里**精确**。所以「性能」说的不是减法露了，而是输入的相对误差 δ_in 被 |a|/(|a−b|) 放大。",
  },
  critical: { input_digits: 15.95 },
};

const TH = 73;
const KN = 9;
const COND_ELLIPSOID: FrameDef = {
  arrays: [
    A("theta", seq(0, 2 * Math.PI, TH), "rad", "参数角"),
    A("circle_x", gen(TH, (i) => Math.cos((2 * Math.PI * i) / (TH - 1))), "1", "单位圆"),
    A("circle_y", gen(TH, (i) => Math.sin((2 * Math.PI * i) / (TH - 1))), "1", "单位圆"),
    A("image_x", gen(TH, (i) => 0.9 * Math.cos((2 * Math.PI * i) / (TH - 1))), "1", "A 的像（椭圆）"),
    A("image_y", gen(TH, (i) => 0.4 * Math.sin((2 * Math.PI * i) / (TH - 1)) + 0.2), "1", "A 的像（椭圆）"),
    A("b_vec", [0.6, 0.9, -0.9, 0.6], "1", "b"), A("db_vec", [1e-12, -1e-12, 0, 0], "1", "δb"),
    A("x_exact", [0.5, 0.5, -0.5, 0.5], "1", "精确解 x"), A("x_num", [0.6, 0.4, -0.4, 0.6], "1", "数值解 x̂"),
    A("x_pert", [1.5, 2, -2, 1.5], "1", "扰动后的解 x̂′"),
    A("kappa_grid", seq(0, 8, KN), "1", "log10 κ（目标）"),
    A("error", gen(KN, (i) => Math.pow(10, 0.3 * i)), "1", "实测放大比"), A("error_bound", gen(KN, (i) => Math.pow(10, i)), "1", "理论上界 κ"),
    A("error_mark", [2.1568], "1", "当前位置的实测放大比"), A("kappa_mark", [0], "1", "当前位置的 log10 κ"),
    A("matrix_A", [1, 0.8, 0, 1], "1", "当前矩阵 A（行优先 2×2）"),
    A("x_exact_defined", [1], "1", "精确解是否存在"), A("x_num_defined", [1], "1", "数值解是否存在"),
    A("x_pert_defined", [1], "1", "扰动解是否存在"),
  ],
  meta: {
    amplification_measured: 2.1568, amplification_over_kappa: 0.9886, b_angle: 30, backward_error: 0, bound_rhs: 2.1816e-12,
    delta_b: 1e-12, det_a: 1, error_floor: 1.1102230246251565e-16, forward_error: 0, kappa_1: 3.24, kappa_2: 2.1816,
    kappa_fro: 2.64, kappa_inf: 3.24, kappa_measured: 2.1816, kappa_target: 1, kappa_u: 2.42e-16, log10_kappa: 0,
    n_theta: TH, overflow: false, residual: 0, shear: 0.8, sigma_max: 1.477, sigma_min: 0.677, t: 0,
  },
  strings: {
    family: "shear", floor_note: "副轴是对数轴：实测放大比小于 u/2 时按地板 u/2 显示（不会出现 log(0)）。", norm: "2",
    norm_note: "本帧的 $\\kappa$ 按 2-范数计算：$\\kappa_1=3.24$、$\\kappa_2=2.18$、$\\kappa_\\infty=3.24$、$\\kappa_{\\mathrm{F}}=2.64$——不同范数可以差一个 $n$ 因子，图上任何「条件数」都要连范数一起读。",
    note: "shear 族：目标 $\\kappa=1$，实测 $\\kappa_2=2.18$，当前方向上的实测放大比 $2.16$（理论上界 $\\kappa=2.18$，实测/上界 $=0.99$）",
  },
  critical: { delta_b: 1.1e-16 },
};

const SK = 29;
const SUMMATION_ORDER: FrameDef = {
  arrays: [
    A("terms_k", seq(0, 63, 64), "1", "项序号 i"),
    A("terms", gen(64, (i) => (i === 0 ? 1e16 : 1)), "1", "项序列（前 64 项）"),
    A("k_grid", seq(1, 512, SK), "1", "部分和长度 k"),
    A("error_naive", seq(0.5551, 511, SK), "1", "朴素求和误差"), A("error_kahan", fill(SK, 0.5551), "1", "补偿求和误差"),
    A("error_kahan_serial", fill(SK, 0.5551), "1", "串行 Kahan 误差（参考）"), A("error_pairwise", fill(SK, 1.5), "1", "成对求和误差"),
    A("error_sorted", fill(SK, 0.5551), "1", "排序后求和误差"), A("error_bound_naive", seq(1.11, 568.4, SK), "1", "理论界 n·u·Σ|x|"),
    A("error_bound_kahan", fill(SK, 2.22), "1", "理论界 (2u+n·u²)·Σ|x|"),
    A("error_naive_mark", [511], "1", "当前朴素误差"), A("error_kahan_mark", [1], "1", "当前补偿误差"),
    A("error_pairwise_mark", [1], "1", "当前成对误差"), A("k_mark", [512], "1", "当前部分和长度"),
  ],
  meta: {
    big: 1e16, bound_kahan: 2.220446049250313, bound_naive: 568.4341886081093, error_exact_zero_points: 35,
    error_floor: 0.5551115123126067, error_kahan: 1, error_kahan_serial: 1, error_naive: 511, error_pairwise: 1,
    error_sorted: 1, exact_sum: 1.0000000000000512e16, k: 512, kahan_over_naive: 0.0019569471624266144, n_terms: 512,
    overflow: false, parallel_over_serial: 1, small: 1, sum_kahan: 1, sum_naive: 1e16, sum_pairwise: 1.000000000000051e16,
    sum_sorted: 1.0000000000000512e16, t: 4096, terms_abs_sum: 1.0000000000000512e16, unit_roundoff: 1.1102230246251565e-16,
  },
  strings: {
    floor_note: "副轴是对数轴：误差恰好为 0 的采样点按地板 5.551e-01 显示，不能读成「误差是地板那么大」。",
    note: "big=1e+16, small=1, n=512, 补偿方式 serial：k=512 时朴素误差 5.110e+02、补偿 1.000e+00（相对朴素 0.00196）",
    scheme: "serial", scheme_note: "serial=串行 Kahan；shuffled=置换后再 Kahan；parallel=分段并行。置换不改变误差量级，分段合并才会。",
  },
  critical: { big: 9007199254740992 },
};

const FIXTURES: Record<string, FrameDef> = {
  float_zoom: FLOAT_ZOOM,
  cancellation: CANCELLATION,
  cond_ellipsoid: COND_ELLIPSOID,
  summation_order: SUMMATION_ORDER,
};
const VIZ_IDS = ["float_zoom", "cancellation", "cond_ellipsoid", "summation_order"] as const;
/** 每个 viz 的一个必需数组（缺了就画不出来）与一个"缺了只降级"的 strings 键。 */
const REQUIRED_ARRAY: Record<string, string> = {
  float_zoom: "offset",
  cancellation: "loss_grid",
  cond_ellipsoid: "image_x",
  summation_order: "k_grid",
};
const CUT_STRING: Record<string, string> = {
  float_zoom: "note",
  cancellation: "note",
  cond_ellipsoid: "floor_note",
  summation_order: "floor_note",
};

// --------------------------------------------------------------------------- //
// 1. manifest 与模型表
// --------------------------------------------------------------------------- //

test("1.1 manifest 与模型表一致：每个 viz 都有 model，未知 id 返回 null", () => {
  assert.equal(manifest.vizzes.length, VIZ_IDS.length);
  for (const viz of manifest.vizzes) {
    assert.ok(models[viz.id], `manifest 里的 ${viz.id} 没有对应 model`);
    assert.equal(modelFor(viz.id) !== null, true);
    for (const aux of viz.requiredAux ?? []) {
      assert.ok(viz.axes.some((a) => a.id === aux), `${viz.id} 的 requiredAux '${aux}' 不在 axes 里`);
    }
  }
  assert.equal(modelFor("不存在的 viz"), null);
});

// --------------------------------------------------------------------------- //
// 2. 默认帧 → 合规 Scene；轴口径与 requiredAux 都对得上
// --------------------------------------------------------------------------- //

for (const viz of VIZ_IDS) {
  test(`1.1/${viz}：默认帧 → 合规 Scene，轴口径与 requiredAux 都对得上`, async () => {
    const spec = manifest.vizzes.find((v) => v.id === viz) ?? assert.fail(`manifest 缺 ${viz}`);
    const scene = sceneOf(await frameFor(viz));
    assert.deepEqual(validateScene(scene), [], `${viz} 的 Scene 不合规`);
    assert.equal(scene.chapter, "1.1");
    assert.equal(scene.viz, viz);
    assert.ok(scene.layers.length > 0, "至少要有一个可绘图层");
    assert.ok(scene.pitfalls.length > 0, "规则 7：pitfalls 必须带进场景");
    assert.equal(badgesOf(scene, "source").length, 1, "必须有且只有一枚 source 角标（规则 5）");
    assert.ok(badgesOf(scene, "status").length >= 1, "必须有 status 角标");
    for (const ax of spec.axes) {
      assert.equal(axisOf(scene, ax.id).autoscaled, ax.autoscaled ?? false, `${viz} 的轴 ${ax.id} autoscaled 与 manifest 不一致`);
    }
    for (const aux of spec.requiredAux ?? []) {
      assert.ok(scene.layers.some((l) => l.axes === aux), `${viz} 的 requiredAux '${aux}' 上一个图层都没有`);
    }
  });
}

test("1.1/float_zoom：960px 说明文字只占右栏，不压绘图区；480px 不铺说明", async () => {
  const frame = await frameFor("float_zoom");
  const wide = sceneOf(frame);
  const ladder = axisOf(wide, "ladder");
  const residual = axisOf(wide, "residual");
  assert.ok(wide.texts.length > 0, "宽画布必须有说明文字");
  const textX = new Set(wide.texts.map((t) => t.x));
  assert.equal(textX.size, 1, "说明文字必须排在同一条右栏里");
  assert.ok((wide.texts[0]?.x ?? 0) > ladder.rect[0] + ladder.rect[2] + 10, "右栏与主图之间要留净空");
  for (const t of wide.texts) {
    const box = textBoxOf({ ...t, x: t.x ?? 0, y: t.y ?? 0 });
    assert.ok(box.ok, `说明行折行失败：${t.text}`);
    for (const axis of [ladder, residual]) {
      const [ax, ay, aw, ah] = axis.rect;
      const ox = Math.min(box.x + box.w, ax + aw) - Math.max(box.x, ax);
      const oy = Math.min(box.y + box.h, ay + ah) - Math.max(box.y, ay);
      assert.ok(ox <= 0 || oy <= 0, `说明压住了 ${axis.id} 绘图区：${t.text}`);
    }
  }
  assert.ok(residual.rect[1] - (ladder.rect[1] + ladder.rect[3]) >= 40, "两块图之间要同时容下刻度带与副轴标题带");

  const narrow = sceneOf(frame, { width: 480, height: 600 });
  assert.deepEqual(validateScene(narrow), [], "窄画布仍要合规");
  assert.equal(narrow.texts.length, 0, "窄画布不铺说明：宁可少说，也不压曲线");

  for (const [label, s] of [["wide", wide], ["narrow", narrow]] as const) {
    const band = pitfallsBand(s);
    assert.ok(band, "底部 pitfalls 面板必须存在");
    for (const axis of s.axes) {
      assert.ok(axis.rect[1] + axis.rect[3] + 28 < band[1], `${label} 的 ${axis.id} 轴与 pitfalls 面板之间放不下 xlabel`);
    }
  }
});

test("1.1/float_zoom：圆点全在横轴内；0.25 边界模式用左右两个括号讲清疏密", async () => {
  const scene = sceneOf(await frameFor("float_zoom"));
  const axis = axisOf(scene, "ladder");
  for (const layer of scene.layers) {
    if (layer.kind !== "scatter" || layer.axes !== "ladder") continue;
    for (let i = 0; i < layer.x.length; i += 1) {
      const x = layer.x[i] as number;
      assert.ok(x >= axis.xlim[0] && x <= axis.xlim[1], `圆点 x=${x} 跑到轴外`);
    }
  }
  assert.equal(scene.overflow.flag, false, "默认窗口不应触发“溢出”");

  const boundary = await frameFor("float_zoom", {
    drop: ["special_offset", "special_y"],
    arrays: {
      offset: [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 1, 2, 3, 4],
      value: seq(0.25 - 8e-17, 0.25 + 8e-17, 13),
    },
    meta: {
      window_ulps: 4,
      focus_kind: 1,
      special_ulps: null,
      spacing_ratio: 2,
      spacing_below: 2.7755575615628914e-17,
      spacing_above: 5.551115123125783e-17,
    },
    strings: { focus: "boundary", center_label: "2 的幂边界 0.25", special_label: "", exact_special: "" },
  });
  const boundaryScene = sceneOf(boundary);
  assert.deepEqual(validateScene(boundaryScene), []);
  const labels = boundaryScene.layers
    .filter((l) => l.kind === "line" && l.axes === "ladder")
    .flatMap((l) => (l.kind === "line" ? l.series.map((s) => s.label ?? "") : []));
  assert.ok(labels.some((text) => text.includes("左侧相邻间距")), "左侧要有一格的间距标注");
  assert.ok(labels.some((text) => text.includes("右侧相邻间距")), "右侧要有一格的间距标注");
  assert.equal(
    boundaryScene.layers.filter((l) => l.kind === "scatter" && l.label?.includes("0.1+0.2")).length,
    0,
    "边界模式不应画 0.1+0.2 的红点",
  );
});

// --------------------------------------------------------------------------- //
// 3. 退化输入：缺必需数组 / 数组不等长 / pitfalls 太短 → ModelInputError
// --------------------------------------------------------------------------- //

for (const viz of VIZ_IDS) {
  test(`1.1/${viz}：缺必需数组 '${REQUIRED_ARRAY[viz]}' → 抛 ModelInputError`, async () => {
    const frame = await frameFor(viz, { drop: [REQUIRED_ARRAY[viz] as string] });
    assert.throws(() => sceneOf(frame), (err: unknown) => err instanceof ModelInputError);
  });
  test(`1.1/${viz}：pitfalls 太短 → 抛 ModelInputError（规则 7：不自行编造）`, async () => {
    const frame = await frameFor(viz, { pitfalls: "太短" });
    assert.throws(() => sceneOf(frame), (err: unknown) => err instanceof ModelInputError);
  });
  test(`1.1/${viz}：缺 strings.${CUT_STRING[viz]} → 不编造 + 降级角标点名`, async () => {
    const scene = sceneOf(await frameFor(viz, { dropStrings: [CUT_STRING[viz] as string] }));
    assert.deepEqual(validateScene(scene), [], `${viz} 降级后 Scene 不合规`);
    assert.match(degradedText(scene), new RegExp(CUT_STRING[viz] as string), `${viz} 的降级角标要点名缺了哪个键`);
  });
}

test("1.1：同组数组长度不一致 → 抛 ModelInputError（错位的图比不画更坏）", async () => {
  const frame = await frameFor("cancellation", { arrays: { digits_remaining: seq(0, 1, 5) } });
  assert.throws(() => sceneOf(frame), (err: unknown) => err instanceof ModelInputError);
});

test("1.1/float_zoom：缺 meta.k → 不猜 k，降级角标点名", async () => {
  const scene = sceneOf(await frameFor("float_zoom", { dropMeta: ["k"] }));
  assert.deepEqual(validateScene(scene), []);
  assert.match(degradedText(scene), /k/);
});

test("1.1/cancellation：缺数组 loss_mark → 游标不装作在 t 上（降级角标点名）", async () => {
  const scene = sceneOf(await frameFor("cancellation", { drop: ["loss_mark"] }));
  assert.deepEqual(validateScene(scene), []);
  assert.match(degradedText(scene), /loss_mark/);
});

test("1.1/cond_ellipsoid：x_pert 里出现 NaN → 不抛错，退化说明进降级角标", async () => {
  const scene = sceneOf(await frameFor("cond_ellipsoid", { arrays: { x_pert: [Number.NaN, 2, -2, 1.5] } }));
  assert.deepEqual(validateScene(scene), []);
  assert.notEqual(degradedText(scene), "");
});

// --------------------------------------------------------------------------- //
// 4. critical 徽标：帧里有值就用帧里的，没有就回落 manifest
// --------------------------------------------------------------------------- //

for (const viz of ["cancellation", "cond_ellipsoid", "summation_order"] as const) {
  test(`1.1/${viz}：critical 徽标用帧值，缺帧值时回落 manifest 的临界值`, async () => {
    const spec = manifest.vizzes.find((v) => v.id === viz) ?? assert.fail(`manifest 缺 ${viz}`);
    const param = spec.params.find((p) => (p.critical ?? []).length > 0) ?? assert.fail(`${viz} 没有声明临界参数`);
    const texts = badgesOf(sceneOf(await frameFor(viz)), "critical");
    assert.equal(texts.length, 1, `${viz} 应当有且只有一枚 critical 角标`);
    assert.ok(texts[0]?.includes(param.name), `critical 角标要点名参数 ${param.name}，实际：${texts[0]}`);
    const fallback = sceneOf(await frameFor(viz, { critical: {} }));
    assert.deepEqual(validateScene(fallback), []);
    assert.equal(badgesOf(fallback, "critical").length, 1, "缺帧值时仍要显示 manifest 的临界值");
    assert.match(degradedText(fallback), new RegExp(param.name), "回落要挂降级角标点名");
  });
}

test("1.1/float_zoom：manifest 没有临界参数 → 不挂 critical 角标（不自己发明临界值）", async () => {
  assert.equal(sceneOf(await frameFor("float_zoom")).badges.filter((b) => b.kind === "critical").length, 0);
});

// --------------------------------------------------------------------------- //
// 5. 非有限值 / 越界：不抛错、不进图层、计数进 overflow
// --------------------------------------------------------------------------- //

test("1.1/summation_order：曲线里插 NaN/∞ → 仍合规，overflow 记录条数并挂角标", async () => {
  const bad = fill(SK, 3);
  bad[3] = Number.NaN;
  bad[7] = Number.POSITIVE_INFINITY;
  const scene = sceneOf(await frameFor("summation_order", { arrays: { error_pairwise: bad } }));
  assert.deepEqual(validateScene(scene), []);
  assert.ok(scene.overflow.note.length > 0, "非有限值必须写进 overflow.note");
  assert.equal(badgesOf(scene, "overflow").length, 1, "非有限值必须挂 overflow 角标");
});

test("1.1/cond_ellipsoid：defined=0 的向量不画（不把占位 0 当解），并写明原因", async () => {
  const scene = sceneOf(
    await frameFor("cond_ellipsoid", {
      arrays: { x_num_defined: [0], x_num: [0, 0, 0, 0] },
      strings: { note: "graded 族在 double 里已奇异，数值解无有限解" },
    }),
  );
  assert.deepEqual(validateScene(scene), []);
  assert.ok(!scene.texts.some((t) => t.text.includes("当前矩阵") === false && t.text.includes("x̂（数值解）")), "未定义的箭头不该有图例");
  assert.match(scene.texts.map((t) => t.text).join("\n"), /a_\{11\}=1/, "当前矩阵必须显示出来");
  const degraded = scene.badges.filter((b) => b.kind === "status").map((b) => b.text).join("\n");
  assert.ok(degraded.length > 0);
});

test("1.1/cond_ellipsoid：向量冲出 ±3 框 → 记录到 overflow，仍合规", async () => {
  const scene = sceneOf(await frameFor("cond_ellipsoid", { arrays: { x_pert: [30, 40, -40, 30] } }));
  assert.deepEqual(validateScene(scene), []);
  assert.ok(scene.overflow.note.length > 0, "被裁掉的向量必须说明");
});

// --------------------------------------------------------------------------- //
// 6. 时间轴：游标只认帧 meta（不在前端重算物理量）
// --------------------------------------------------------------------------- //

test("1.1/float_zoom：k 由 meta.k 决定；帧没变，opts.t 变了图也不许变", async () => {
  const frame = await frameFor("float_zoom");
  assert.equal(sceneToJson(sceneOf(frame, { t: 0 })), sceneToJson(sceneOf(frame, { t: 64 })), "帧没变图就不该变");
  assert.notEqual(
    sceneToJson(sceneOf(await frameFor("float_zoom", { meta: { k: 8 } }))),
    sceneToJson(sceneOf(frame)),
    "meta.k 变了游标必须跟着动",
  );
});

test("1.1/cancellation：游标画在实测损失位数（数组 loss_mark）上，不假装 t 就是它", async () => {
  const scene = sceneOf(await frameFor("cancellation"), { t: 15.75 });
  assert.deepEqual(validateScene(scene), []);
  const texts = scene.texts.map((t) => t.text).join("\n");
  assert.match(texts, /游标/, "必须说明游标口径");
  assert.match(texts, /6/, "游标值要来自帧的 loss_mark=6");
});

test("1.1/summation_order：k 取帧 meta.k=512；t 与 k 不是一个量，必须写在图上", async () => {
  const texts = sceneOf(await frameFor("summation_order")).texts.map((t) => t.text).join("\n");
  assert.match(texts, /k=512/);
  assert.match(texts, /时间轴口径/);
});

// --------------------------------------------------------------------------- //
// 7. 排版与确定性：窄画布不炸、两次序列化逐字节相同
// --------------------------------------------------------------------------- //

for (const viz of VIZ_IDS) {
  test(`1.1/${viz}：窄画布 480×600 仍合规；同一帧两次 → sceneToJson 逐字节相同`, async () => {
    const frame = await frameFor(viz);
    assert.deepEqual(validateScene(sceneOf(frame, { width: 480, height: 600 })), [], `${viz} 窄画布下文字压住了轴线`);
    assert.equal(sceneToJson(sceneOf(frame)), sceneToJson(sceneOf(frame)), "动画里同一帧必须逐字节稳定");
  });
}
