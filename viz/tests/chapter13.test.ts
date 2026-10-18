/**
 * 1.3 章 model 层的测试（`projection_3d` / `normal_vs_qr` / `svd_truncation` / `uncertainty_fan`）。
 *
 * 夹具口径与 1.1 章一致：**手搓帧**，`new Frame(...)` + `encodeNlf` → `decodeNlf`，
 * 字段名/长度按 `server/chapters/1.3.py` 的真实帧裁小（点数少、数值是合成值，**不是物理计算**）。
 * 于是 `npm test` 在没有 Python 的机器上也自足：它守"契约与降级行为"，
 * 数值真值由后端 tests/1.3 与 `tests/baselines/1.3/*.png` 守。
 *
 * 被守的降级行为（每条都有断言）
 * ------------------------------
 * | 缺失 / 情形 | 行为 |
 * |---|---|
 * | 必需数组（含 2-D 的 `level_x`/`level_y`、3-D 的 `plane_*`） | **抛 `ModelInputError`**（不画半张图） |
 * | 同组数组长度不一致（如 `plane_x` 与 `plane_z`） | **抛 `ModelInputError`** |
 * | `frame.pitfalls` 为空或 < 10 字 | **抛 `ModelInputError`**（规则 7：不自行编造） |
 * | 可选 meta（`defect_selected` 等）缺失 | 不猜数值 + `⚠ 降级：` 角标点名 |
 * | 选中的路线**没有有限解**（`selected_singular=1`） | `defect_selected`/`rel_error_selected`/`orthogonality_abs` 为 `null`：整项省略，**不写 0**；文字里写"没有有限解" |
 * | 某条路线 `method_singular=1` | 该行写"**没有有限解**"；缺陷数组里是地板值（占位，不是测量） |
 * | 场景合规 | `validateScene` 无问题；每个 viz 的 `requiredAux` 轴都真有图元 |
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type Dtype, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Scene } from "../shared/src/scene.js";
import { models } from "../chapters/1.3/index.js";
import { manifest as spec } from "../chapters/1.3/manifest.gen.js";

// --------------------------------------------------------------------------- //
// 夹具
// --------------------------------------------------------------------------- //

const PITFALLS =
  "它会怎样骗你：其一，「解不出来」与「误差很大」在误差数组里都是大数，但前者是占位、后者是实测；" +
  "其二，对数轴上的地板值看起来像「很小的误差」，其实那是「没有定义」的占位，不是测量。";
const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };
type MetaValue = number | boolean | null;

interface ArrayDef {
  name: string;
  data: ArrayLike<number>;
}
interface FrameDef {
  arrays: ArrayDef[];
  meta: Record<string, MetaValue>;
  strings: Record<string, string>;
  critical?: Record<string, number | number[]>;
}

const A = (name: string, data: ArrayLike<number>): ArrayDef => ({ name, data });
const fill = (n: number, v: number): Float64Array => new Float64Array(n).fill(v);
const gen = (n: number, f: (i: number) => number): Float64Array => Float64Array.from({ length: n }, (_, i) => f(i));

function toTyped(data: ArrayLike<number>, dtype: Dtype): TypedArray {
  if (dtype === "f8") return Float64Array.from(data as ArrayLike<number>);
  if (dtype === "f4") return Float32Array.from(data as ArrayLike<number>);
  if (dtype === "i4") return Int32Array.from(data as ArrayLike<number>);
  return Uint8Array.from(data as ArrayLike<number>);
}

/** 造帧 → 编码 `.nlf` → 解码回来：夹具本身就是合法的线上帧。 */
async function makeFrame(viz: string, def: FrameDef, pitfalls = PITFALLS): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const a of def.arrays) {
    const data = toTyped(a.data, "f8");
    specs.push({ name: a.name, dtype: "f8", shape: [a.data.length], order: "C", offset: 0, nbytes: data.byteLength, unit: "", label: a.name });
    arrays.set(a.name, data);
  }
  const frame = new Frame({
    chapter: "1.3",
    viz,
    kind: "frame",
    codec: "none",
    meta: def.meta,
    strings: { data_source: "backend", params_json: JSON.stringify({ viz }), note: "夹具 note", ...def.strings },
    critical: def.critical ?? {},
    pitfalls,
    specs,
    arrays,
  });
  return decodeNlf(await encodeNlf(frame));
}

interface Override {
  arrays?: Record<string, ArrayLike<number>>;
  drop?: string[];
  meta?: Record<string, MetaValue>;
  dropMeta?: string[];
  strings?: Record<string, string>;
  dropStrings?: string[];
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
  const strings = { ...Object.fromEntries(Object.entries(def.strings).filter(([k]) => !dropStr.has(k))), ...over.strings };
  return makeFrame(viz, { arrays, meta: { ...meta, ...over.meta }, strings, critical: over.critical ?? def.critical }, over.pitfalls);
}

function sceneOf(frame: Frame, over: Partial<ModelOptions> = {}): Scene {
  const model = models[frame.viz];
  assert.ok(model, `models 里没有 ${frame.viz}`);
  const t = frame.meta["t"] === undefined ? 0 : Number(frame.meta["t"]);
  return model(frame, { ...OPTS, ...over, t: over.t ?? t, params: over.params ?? OPTS.params });
}

const degradedText = (scene: Scene): string =>
  scene.badges.filter((b) => b.text.includes("降级：")).map((b) => b.text).join(" ");
const textOf = (scene: Scene): string => scene.texts.map((t) => t.text).join("\n");
/** 画布上的全部文字：文字层 + 角标（角标也是图上的字，规则 5 允许把细节放角标）。 */
const allText = (scene: Scene): string => `${textOf(scene)}\n${scene.badges.map((b) => b.text).join("\n")}`;
const badgeKinds = (scene: Scene): Set<string> => new Set(scene.badges.map((b) => b.kind));

// --------------------------------------------------------------------------- //
// 四个 viz 的字段表（长度裁小，形状/名字对齐 server/chapters/1.3.py）
// --------------------------------------------------------------------------- //

/** `projection_3d`：平面 3×3、三条路线。κ 用 30（log10 κ ≈ 1.48）。 */
const PROJECTION: FrameDef = {
  arrays: [
    A("col_a1", [0.8, 0.1, -0.4]),
    A("col_a2", [0.2, 0.9, 0.3]),
    A("b_vec", [0.5, 0.5, 0.5]),
    A("p_vec", [0.45, 0.5, 0.5]),
    A("r_vec", [0.05, 0, 0]),
    A("p_anim", [0.45, 0.5, 0.5]),
    A("r_anim", [0, 0, 0]),
    A("plane_x", gen(9, (i) => -1 + (i % 3))),
    A("plane_y", gen(9, (i) => -1 + Math.floor(i / 3))),
    A("plane_z", fill(9, 0)),
    A("residual", [0.377, 0.377, 0.377]),
    A("orthogonality_defect", [2.5e-3, 1.2e-4, 9.1e-5]),
    A("error", [1.1e-4, 7.6e-5, 3.2e-5]),
    A("method_singular", [0, 0, 0]),
  ],
  meta: {
    t: 0,
    t_blend: 0,
    kappa_target: 30,
    kappa_measured: 30,
    sigma_max: 1,
    sigma_min: 0.033,
    b_norm: 0.866,
    projection_norm: 0.85,
    residual_norm: 0.05,
    angle_b_plane_deg: 3.3,
    defect_selected: 1.2e-4,
    defect_max: 2.5e-3,
    defect_floor: 1e-18,
    orthogonality_abs: 1.2e-4,
    rel_error_selected: 7.6e-5,
    n_singular_methods: 0,
    selected_singular: 0,
    singular: 0,
  },
  strings: { selected_method: "qr", method_0: "normal", method_1: "qr", method_2: "svd", status: "夹具 status" },
};

/** `normal_vs_qr`：κ 网格 4 点。 */
const NVQ: FrameDef = {
  arrays: [
    A("kappa_grid", [2, 5, 9, 12]),
    A("error_normal", [1e-13, 1e-8, 1e-2, 1]),
    A("error_qr", [1e-15, 1e-11, 1e-7, 1e-5]),
    A("error_svd", [2e-15, 2e-11, 2e-7, 2e-5]),
    A("error_normal_ir", [1e-15, 1e-11, 1e-6, 1]),
    A("residual_normal", [1e-14, 1e-13, 1e-10, 1e-7]),
    A("residual_qr", [1e-16, 1e-16, 1e-16, 1e-16]),
    A("residual_svd", [2e-16, 2e-16, 2e-16, 2e-16]),
    A("theory_kappa2", [1e-12, 1e-9, 1e-5, 1e-2]),
    A("theory_kappa1", [1e-14, 1e-11, 1e-7, 1e-4]),
    A("divergence", [100, 1e3, 1e5, 1e7]),
    A("kappa_mark", [100]),
    A("error_mark", [2.6e-13, 2.7e-15, 3.9e-15]),
    A("error_mark_ir", [1e-15]),
    A("residual_mark", [9e-15, 2.9e-16]),
  ],
  meta: {
    t: 0,
    n: 10,
    m: 20,
    n_points: 4,
    n_seeds: 2,
    refinement_steps: 1,
    log10_kappa_current: 2,
    kappa_current: 100,
    slope_normal: 1.965,
    slope_qr: 0.925,
    slope_svd: 0.974,
    slope_normal_ir: 1.881,
    slope_window_normal: 4.2e7,
    slope_window_qr: 1e12,
    error_ceiling: 1,
    error_floor: 1e-16,
    ratio_current: 98,
    defect_max: 0.545,
    n_singular_normal: 1,
    n_singular_qr: 0,
    singular: 0,
  },
  strings: { refinement: "one", status: "夹具 status" },
};

/** `svd_truncation`：n=4（sigma 4 个）、k 网格 5 点（含 k=0）。 */
const SVD: FrameDef = {
  arrays: [
    A("k_grid", [0, 1, 2, 3, 4]),
    A("sigma", [1, 0.5, 0.1, 0.01]),
    A("coeff", [0.8, 0.3, 0.05, 0.001]),
    A("picard", [0.8, 0.6, 0.5, 0.2]),
    A("x_norm", [1e-18, 0.9, 1.1, 3.0, 40]),
    A("residual", [1, 0.4, 0.2, 0.01, 1e-18]),
    A("error", [1, 0.5, 0.3, 0.05, 3]),
    A("lcurve_x", [1e-18, 0.9, 1.1, 3.0, 40]),
    A("lcurve_y", [1, 0.4, 0.2, 0.01, 1e-18]),
    A("k_mark", [2]),
    A("noise_level", [1e-6]),
  ],
  meta: {
    t: 0,
    n: 4,
    m: 12,
    rank: 4,
    log10_kappa_target: 8,
    kappa_measured: 1e8,
    sigma_max: 1,
    sigma_min: 1e-8,
    noise_exp: -6,
    noise_rel: 1e-6,
    k_current: 2,
    k_max: 4,
    k_opt: 3,
    k_corner: 2,
    k_picard: 3,
    x_norm_current: 1.1,
    residual_current: 0.2,
    error_current: 0.3,
    x_norm_jump: 13,
    cliff_index: 4,
    picard_min: 0.2,
    error_floor: 1e-18,
  },
  strings: { spectrum: "decay" },
};

/** `uncertainty_fan`：2 条等残差曲线 × 8 点、6 个方向。 */
const FAN: FrameDef = {
  arrays: [
    A("level_x", gen(16, (i) => 1 + (i % 8) * 1e-3)),
    A("level_y", gen(16, (i) => 1 + Math.floor(i / 8) * 1e-3)),
    A("fan_x", gen(6, (i) => 1 + 4e-4 * Math.cos(i))),
    A("fan_y", gen(6, (i) => 1 + 4e-4 * Math.sin(i))),
    A("error", [1e-9, 2e-9, 3e-9, 2e-9, 1e-9, 5e-10]),
    A("residual", fill(6, 1e-8)),
    A("theta_deg", [0, 60, 120, 180, 240, 300]),
    A("sol_now", [1, 1]),
    A("center", [1, 1]),
    A("v1_dir", [0.48, -0.88]),
    A("v2_dir", [-0.88, -0.48]),
    A("sigma", [1, 1e-6]),
    A("semi_axes", [1e6, 1]),
    A("kappa_grid", [1, 100, 1e4, 1e6]),
    A("error_kappa", [1e-8, 1e-6, 1e-4, 3e-3]),
    A("theory_kappa", [0.01, 1, 100, 1e4]),
    A("level_rho", [1e-4, 1e-2]),
  ],
  meta: {
    t: 0,
    kappa_target: 1e6,
    kappa_measured: 1e6,
    sigma_max: 1,
    sigma_min: 1e-6,
    noise_rel: 1e-8,
    rho_rel: 1e-3,
    semi_axis_ratio: 1e6,
    amplification_max: 2.8e-3,
    amplification_theory: 1e-2,
    residual_spread: 1.7e-10,
    residual_roundoff: 1.6e-10,
    residual_flatness: 0.017,
    roundoff_rel: 1.1e-10,
    fan_width_deg: 0.16,
    x_norm: 1.414,
    error_floor: 1e-18,
    resolvable: true,
    n_samples: 6,
    n_levels: 2,
    n_visible: 6,
  },
  strings: { status: "夹具 status" },
};

const FIXTURES: Record<string, FrameDef> = {
  projection_3d: PROJECTION,
  normal_vs_qr: NVQ,
  svd_truncation: SVD,
  uncertainty_fan: FAN,
};

/** manifest 里声明为**自适应**的轴 id（`axisModeBadge` 的文案要与它一致）。 */
const AUTOSCALED: Record<string, string[]> = {
  projection_3d: ["orthogonality"],
  normal_vs_qr: ["residual"],
  svd_truncation: ["lcurve", "spectrum", "tradeoff"],
  uncertainty_fan: ["growth"],
};

// --------------------------------------------------------------------------- //
// 1) 契约：默认帧 → 合规场景 + 必需的可见层与轴
// --------------------------------------------------------------------------- //

for (const viz of Object.keys(FIXTURES)) {
  test(`${viz}：默认帧产出合规场景，轴集合与 manifest 一致`, async () => {
    const scene = sceneOf(await frameFor(viz));
    assert.deepEqual(validateScene(scene), [], `${viz} 的场景必须合规`);

    const declared = (spec.vizzes.find((v) => v.id === viz)?.axes ?? []).map((a) => a.id);
    const drawn = scene.axes.map((a) => a.id);
    for (const id of declared) assert.ok(drawn.includes(id), `${viz} 少了 manifest 声明的轴 ${id}`);
    for (const id of drawn) assert.ok(scene.layers.some((l) => l.axes === id), `${viz} 的轴 ${id} 上没有图元（画了空轴）`);

    // `requiredAux` 是**数据名**（"residual"/"error"）：规则 3 要求这块辅助量在图上看得见。
    // 判据：图层标签或文字层里必须出现对应的中文口径（残差 / 误差、不确定度）。
    const surface = [...scene.layers.map((l) => ("label" in l ? String(l.label ?? "") : "")), ...scene.texts.map((t) => t.text)].join("\n");
    for (const aux of spec.vizzes.find((v) => v.id === viz)?.requiredAux ?? []) {
      const re = aux === "residual" ? /残差/ : /误差|不确定度/;
      assert.match(surface, re, `${viz} 的 requiredAux=${aux} 在画面上看不到（规则 3）`);
    }
  });

  test(`${viz}：自适应轴与 manifest 一致（角标文案跟着变）`, async () => {
    const scene = sceneOf(await frameFor(viz));
    const declared = new Set((spec.vizzes.find((v) => v.id === viz)?.axes ?? []).map((a) => a.id));
    const auto = new Set(AUTOSCALED[viz]);
    for (const axis of scene.axes) {
      // 本章 svd_truncation 多一条 `picard` 轴（后端给了 coeff/picard 数组，manifest 只声明 3 条）：
      // 多出来的轴按设计**一律自适应**（见 model 文件头），这里把它单独守住。
      if (!declared.has(axis.id)) {
        assert.equal(Boolean(axis.autoscaled), true, `${viz}/${axis.id} 是 manifest 之外的轴，必须自适应并自己挂角标`);
        continue;
      }
      assert.equal(Boolean(axis.autoscaled), auto.has(axis.id), `${viz}/${axis.id} 的 autoscaled 与 manifest 不一致`);
    }
    const autoText = scene.badges.map((b) => b.text).join(" ");
    assert.match(autoText, /坐标轴：(固定|自适应)/, `${viz} 缺轴模式角标（规则 4）`);
    if (auto.size > 0) assert.match(autoText, /自适应/, `${viz} 有自适应轴，角标必须说清楚`);
  });

  test(`${viz}：角标齐全（来源／自适应／临界／溢出／状态）`, async () => {
    const scene = sceneOf(await frameFor(viz));
    const kinds = badgeKinds(scene);
    for (const need of ["source", "autoscale", "critical", "status"]) {
      assert.ok(kinds.has(need), `${viz} 缺 ${need} 角标`);
    }
    assert.ok(scene.badges.some((b) => b.kind === "overflow"), `${viz} 缺 overflow 角标`);
    assert.ok(scene.dataSource.length > 0);
  });
}

// --------------------------------------------------------------------------- //
// 2) projection_3d：κ=1e12 的"没有有限解"路径
// --------------------------------------------------------------------------- //

test("projection_3d：选中路线没有有限解 → 整项省略（不写 0）、文字点名、场景仍合规", async () => {
  const frame = await frameFor("projection_3d", {
    arrays: {
      method_singular: [1, 0, 0],
      orthogonality_defect: [1e-18, 1.2e-4, 9.1e-5], // 地板占位
      error: [1e-18, 7.6e-5, 3.2e-5],
    },
    meta: {
      log10_kappa_target: 12,
      kappa_target: 1e12,
      kappa_measured: 1e12,
      sigma_min: 1e-12,
      selected_singular: 1,
      singular: 1,
      n_singular_methods: 1,
      // 三个"选中量"在帧里就是 null（后端口径：没有有限解就不给数）
      defect_selected: null,
      rel_error_selected: null,
      orthogonality_abs: null,
    },
    strings: { selected_method: "normal" },
  });
  const scene = sceneOf(frame);
  assert.deepEqual(validateScene(scene), []);
  const body = textOf(scene);
  assert.match(body, /没有有限解/);
  assert.match(body, /不是测量/); // 地板值是占位
  assert.match(body, /解不出来/, "必须把「解不出来」与「误差很大」分开说");
  // 选中量缺失时**不许**出现 `缺陷=0` 这类假数
  assert.doesNotMatch(body, /缺陷=0(?!\.)/);
  // 全局角标要点名"无有限解"，而不是把它当成一个大误差
  const status = scene.badges.filter((b) => b.text.includes("有限解")).map((b) => b.text).join(" ");
  assert.match(status, /没有有限解|无有限解/);
});

test("projection_3d：singular 点画成空心标记，且与实测点不同（不会被读成实测）", async () => {
  const frame = await frameFor("projection_3d", {
    arrays: { method_singular: [1, 0, 0] },
    meta: { n_singular_methods: 1, singular: 1, selected_singular: 0 },
  });
  const scene = sceneOf(frame);
  const marks: string[] = [];
  for (const l of scene.layers) {
    if (l.axes !== "orthogonality" || l.kind !== "scatter") continue;
    marks.push(`${l.label ?? ""}|alpha=${l.alpha ?? 1}|r=${l.r}`);
  }
  assert.ok(marks.length >= 1, "正交性副轴上应有散点");
  assert.ok(marks.some((m) => /没有有限解/.test(m)), `singular 点必须自带说明，实际：${marks.join(" ")}`);
  assert.ok(marks.some((m) => /alpha=0\.\d/.test(m)), `singular 点必须是半透明的（空心语义），实际：${marks.join(" ")}`);
});

// --------------------------------------------------------------------------- //
// 3) normal_vs_qr：误差上限是**占位**，不是测量
// --------------------------------------------------------------------------- //

test("normal_vs_qr：error_ceiling 只作截断与标注，曲线不留在大误差处", async () => {
  const scene = sceneOf(await frameFor("normal_vs_qr"));
  assert.deepEqual(validateScene(scene), []);
  const errOnly: { label?: string; y: ArrayLike<number> }[] = [];
  const series: { label?: string; y: ArrayLike<number> }[] = [];
  for (const l of scene.layers) {
    if (l.kind !== "line") continue;
    for (const s of l.series) {
      series.push({ label: s.label, y: s.y });
      if (l.axes === "errors") errOnly.push({ label: s.label, y: s.y });
    }
  }
  const ceiling = series.filter((s) => /上限/.test(s.label ?? ""));
  assert.ok(ceiling.length >= 1, "误差上限必须有一条自己的参考线（规则 5）");
  for (const s of ceiling) {
    for (const v of Array.from(s.y)) assert.ok(Number(v) <= 1 + 1e-9, "上限参考线不能画到 1 以上");
  }
  const body = textOf(scene);
  assert.match(body, /占位/, "文字必须说明 1 是占位上限");
  // 正规方程那条曲线在达到上限的 κ 处截断：不该出现 > 1 的实测点
  // 只查 `errors` 轴上的**误差曲线**：比值曲线（误差之比）本来就 >1，它不在这个轴上。
  const normal = errOnly.filter((s) => (s.label ?? "").includes("正规方程"));
  assert.ok(normal.length >= 1, "errors 轴上应有正规方程的误差曲线");
  for (const s of normal) {
    for (const v of Array.from(s.y)) assert.ok(Number(v) <= 1 + 1e-9, `正规方程误差曲线越过了上限：${v}`);
  }
});

test("normal_vs_qr：数组长度不一致 → 抛 ModelInputError", async () => {
  const frame = await frameFor("normal_vs_qr", { arrays: { residual_normal: [1e-14, 1e-13] } });
  assert.throws(() => sceneOf(frame), ModelInputError);
});

// --------------------------------------------------------------------------- //
// 4) svd_truncation / uncertainty_fan：k=0 与地板
// --------------------------------------------------------------------------- //

test("svd_truncation：k=0 的点不进对数轴，但要在文字里交代地板", async () => {
  const scene = sceneOf(await frameFor("svd_truncation"));
  assert.deepEqual(validateScene(scene), []);
  for (const axis of scene.axes) {
    for (const layer of scene.layers) {
      if (layer.axes !== axis.id || layer.kind !== "line") continue;
      for (const s of layer.series) {
        for (const v of Array.from(s.x)) {
          if (axis.id === "lcurve" || axis.id === "spectrum") assert.ok(Number(v) > 0, `${axis.id} 的对数轴上出现了 0/负坐标`);
        }
      }
    }
  }
  assert.match(allText(scene), /地板/, "要说明地板值（后端把 0 抬到地板）");
  assert.match(allText(scene), /k=0/, "要说明 k=0 为什么不在图上（地板占位）");
});

test("uncertainty_fan：扇面与增长轴都在，且给出可分辨性判断", async () => {
  const scene = sceneOf(await frameFor("uncertainty_fan"));
  assert.deepEqual(validateScene(scene), []);
  assert.ok(scene.layers.some((l) => l.axes === "plane" && l.kind === "line"), "解平面上要有扇面");
  const body = textOf(scene);
  assert.match(body, /同一半径|等比例/);
  assert.match(body, /κu|可分辨/);
  // 不可分辨时必须报警，且不能继续谈"噪声放大"
  const bad = sceneOf(
    await frameFor("uncertainty_fan", {
      meta: { resolvable: false, roundoff_rel: 1e-3, noise_rel: 1e-8 },
    }),
  );
  const badBody = allText(bad);
  assert.match(badBody, /不可分辨/);
  assert.match(badBody, /舍入/);
});

// --------------------------------------------------------------------------- //
// 5) 降级路径：缺必需数组 / 缺可选 meta / 缺 pitfalls
// --------------------------------------------------------------------------- //

test("必需数组缺失 → 抛 ModelInputError（不画半张图）", async () => {
  for (const [viz, name] of [
    ["projection_3d", "plane_z"],
    ["normal_vs_qr", "kappa_grid"],
    ["svd_truncation", "sigma"],
    ["uncertainty_fan", "level_y"],
  ] as const) {
    const frame = await frameFor(viz, { drop: [name] });
    assert.throws(() => sceneOf(frame), ModelInputError, `${viz} 缺 ${name} 时应抛错`);
  }
});

test("pitfalls 缺失或过短 → 抛 ModelInputError（规则 7）", async () => {
  const frame = await frameFor("svd_truncation", { pitfalls: "太短" });
  assert.throws(() => sceneOf(frame), ModelInputError);
});

test("可选 meta 缺失 → 不猜数，挂降级角标并写进文字", async () => {
  const frame = await frameFor("projection_3d", {
    dropMeta: ["t_blend", "kappa_target", "kappa_measured", "defect_selected", "rel_error_selected"],
    dropStrings: ["selected_method"],
  });
  const scene = sceneOf(frame);
  assert.deepEqual(validateScene(scene), []);
  const deg = degradedText(scene);
  assert.match(deg, /t_blend/, "缺 meta 必须在降级角标里点名（角标超宽会截断，所以只守第一条）");
  assert.ok(scene.texts.length > 0, "降级时也要有文字层");
  assert.doesNotMatch(textOf(scene), /kappa_target=[\d]/, "缺 meta 时不许凭空写数值");
  assert.doesNotMatch(textOf(scene), /实测 κ=[\d]/, "缺 kappa_measured 时不许写 κ 的数值（不拿 kappa_target 冒充）");
  assert.match(degradedText(scene), /t_blend/);
});

test("数值里出现非有限值：不进图层、计数进 overflow 角标（不静默丢）", async () => {
  const frame = await frameFor("normal_vs_qr", {
    arrays: { error_qr: [1e-15, Number.NaN, 1e-7, Number.POSITIVE_INFINITY] },
  });
  const scene = sceneOf(frame);
  assert.deepEqual(validateScene(scene), []);
  for (const layer of scene.layers) {
    if (layer.kind !== "line") continue;
    for (const s of layer.series) {
      for (const v of Array.from(s.y)) assert.ok(Number.isFinite(v), "图层里不许有 NaN/∞");
    }
  }
  assert.ok(scene.overflow.flag, "有非有限值时 overflow 必须立起来");
  assert.match(scene.overflow.note, /非有限|NaN|∞/);
});

test("帧的 critical 覆盖 manifest 默认值（角标跟着帧走）", async () => {
  const frame = await frameFor("normal_vs_qr", { critical: { exp_max: 11.5 } });
  const scene = sceneOf(frame);
  const badge = scene.badges.find((b) => b.kind === "critical");
  assert.ok(badge, "缺 critical 角标");
  assert.match(badge.text, /11\.5/, `临界值应取帧里的 11.5，实际：${badge.text}`);
});

test("manifest 声明的 viz 与 models 一一对应（漏注册会当场失败）", () => {
  for (const v of spec.vizzes) {
    assert.ok(models[v.id], `manifest 里有 ${v.id}，models 里没有`);
  }
  assert.equal(Object.keys(models).length, spec.vizzes.length);
});
