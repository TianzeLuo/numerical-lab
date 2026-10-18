/**
 * 2.4 章 model 层的测试（`area_convergence` / `gauss_vs_equispaced` / `romberg_table` / `singular_endpoint`）。
 *
 * 夹具用哪种方式？—— **手搓帧（唯一夹具）**
 * ----------------------------------------
 * `new Frame(...)` + `encodeNlf` → `decodeNlf` 造出与 `server/chapters/2.4.py` **字段集合一致**
 * 的帧（数组名 / meta 键 / strings 键 / critical 键 / pitfalls 全照抄后端），于是
 * `npm test` 在没有 Python、没有 `.cache/` 的机器上自足。数值是测试用的合成值，但**关系是真的**：
 * `approx_y` 就是被积函数的分段线性插值、`error_curve = C·n^-2`、`error_moment_gauss` 在
 * `k ≤ 2n−1` 时落在地板而 `k = 2n` 跳起来、`error_table` 是三角长表（行主序
 * `row(row+1)/2 + col`）、`error_ts` 在 `step` 之外是**地板占位值**。
 *
 * 这个文件主要守什么
 * ------------------
 * 1. **契约**：四个 viz 都产出 `validateScene` 通过的 Scene；轴 id / 自适应标记 / 图层
 *    `axes` 归属逐项对得上 manifest。
 * 2. **三处最容易被骗的地方必须同屏**（对应 model 文件头的说明）：
 *    `area_convergence` 的「公式阶」与「光滑性上限」两条理论线 + adaptive 的估计/真误差；
 *    `gauss_vs_equispaced` 的 `k=2n` 失效与 Γ（两条曲线：等距 vs Gauss≡1）；
 *    `romberg_table` 的「实测阶 vs 理论阶」与地板竖线；`singular_endpoint` 的尾巴/地板。
 * 3. **不许偷偷补数**：`meta` 缺标量 → 文本写 `?` + 降级角标（`DEGRADED_PREFIX`），
 *    绝不写 0；`error_ts` 在已揭示窗口之外是占位值 → 不许画出来。
 * 4. **缺数组 / 长度不一致 / pitfalls 太短 → 抛 `ModelInputError`**（画不出来就不画半张图）。
 * 5. **版面自洽**：文本块不许出现"画布放不下"，也不许压住坐标轴装饰带
 *    （后者由 `validateScene` 的规则 8 兜住）。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type Dtype, type TypedArray } from "../shared/src/frame.js";
import { DEGRADED_PREFIX, ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Layer, type Scene } from "../shared/src/scene.js";
import { areaConvergenceModel } from "../chapters/2.4/area_convergence.model.js";
import { gaussVsEquispacedModel } from "../chapters/2.4/gauss_vs_equispaced.model.js";
import { manifest, modelFor, models } from "../chapters/2.4/index.js";
import { rombergTableModel } from "../chapters/2.4/romberg_table.model.js";
import { singularEndpointModel } from "../chapters/2.4/singular_endpoint.model.js";

function fail(message: string): never {
  throw new Error(message);
}

const OPTS: ModelOptions = { theme: "light", t: 0, width: 960, height: 600, params: {} };
const VIZ_IDS = ["area_convergence", "gauss_vs_equispaced", "romberg_table", "singular_endpoint"] as const;

function specOf(id: string) {
  return manifest.vizzes.find((v) => v.id === id) ?? fail(`manifest 缺少 ${id}`);
}

const PITFALLS =
  "① 双对数图上量到的斜率不一定是求积公式的阶：f=√x 时梯形与 Simpson 都塌到 min(p,α+1)=1.5。" +
  "② f=|x| 折点落在节点上时梯形法逐段精确，误差掉到 1e-16，像「阶无穷大」，其实是巧合。" +
  "③ adaptive 的「估计误差」是嵌套法则之差，不是真误差：窄峰上估计 4.3e-5 而真误差 3.5e-2。";

interface ArrayDef {
  name: string;
  data: ArrayLike<number>;
  dtype?: Dtype;
  unit?: string;
  label?: string;
}

interface FrameDef {
  viz: string;
  arrays: ArrayDef[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical: Record<string, number | number[]>;
  pitfalls?: string;
}

interface Override {
  arrays?: Record<string, ArrayLike<number>>;
  drop?: string[];
  meta?: Record<string, number | boolean | null>;
  dropMeta?: string[];
  strings?: Record<string, string>;
  dropStrings?: string[];
  critical?: Record<string, number | number[]>;
  pitfalls?: string;
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

/** 造帧 → 编码成 `.nlf` → 解码回来：夹具本身就是**合法的线上帧**。 */
async function makeFrame(def: FrameDef, over: Override = {}): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  const meta = { ...def.meta };
  for (const k of over.dropMeta ?? []) delete meta[k];
  Object.assign(meta, over.meta ?? {});
  const strings = { ...def.strings, ...(over.strings ?? {}) };
  for (const k of over.dropStrings ?? []) delete strings[k];
  const defs: ArrayDef[] = [
    ...def.arrays
      .filter((a) => !(over.drop ?? []).includes(a.name))
      .map((a) => (over.arrays?.[a.name] !== undefined ? { ...a, data: over.arrays[a.name] as ArrayLike<number> } : a)),
    ...Object.entries(over.arrays ?? {})
      .filter(([name]) => !def.arrays.some((a) => a.name === name))
      .map(([name, data]) => ({ name, data })),
  ];
  for (const a of defs) {
    const dtype = a.dtype ?? "f8";
    const data = toTyped(a.data, dtype);
    specs.push({
      name: a.name,
      dtype,
      shape: [a.data.length],
      order: "C",
      offset: 0,
      nbytes: data.byteLength,
      unit: a.unit ?? "1",
      label: a.label ?? a.name,
    });
    arrays.set(a.name, data);
  }
  const frame = new Frame({
    chapter: "2.4",
    viz: def.viz,
    kind: "frame",
    codec: "none",
    meta,
    strings,
    critical: over.critical ?? def.critical,
    pitfalls: over.pitfalls ?? def.pitfalls ?? PITFALLS,
    specs,
    arrays,
  });
  return decodeNlf(await encodeNlf(frame));
}

function ok(scene: Scene, viz: string): Scene {
  const problems = validateScene(scene);
  assert.deepEqual(problems, [], `${viz} 的 Scene 不合规：${problems.join("；")}`);
  assert.equal(scene.chapter, "2.4");
  assert.equal(scene.viz, viz);
  return scene;
}

function layersOf(scene: Scene, axes: string): Layer[] {
  return scene.layers.filter((l) => (l as { axes?: string }).axes === axes);
}

function badgeTexts(scene: Scene): string {
  return scene.badges.map((b) => b.text).join(" ｜ ");
}

function textBody(scene: Scene): string {
  return scene.texts.map((t) => t.text).join("\n");
}

function hasKind(scene: Scene, axes: string, kind: Layer["kind"]): boolean {
  return layersOf(scene, axes).some((l) => l.kind === kind);
}

/** 去掉所有空白：文本块在预算处折行，断言只关心"这串字在不在屏幕上"。 */
function squash(s: string): string {
  return s.replace(/[\s\u3000]+/g, "");
}

/** 期望文本在不在这个 Scene 的文字里（忽略折行位置）。 */
function has(body: string, plain: string): boolean {
  return squash(body).includes(squash(plain));
}

/** 文本块里出现"画布放不下"= 我写的字超出了面板预算（版面自检）。 */
function assertNoClippedText(scene: Scene, viz: string): void {
  const body = textBody(scene);
  assert.ok(!body.includes("画布放不下"), `${viz} 的文本块超出面板预算：${body}`);
}

// --------------------------------------------------------------------------- //
// area_convergence
// --------------------------------------------------------------------------- //

const AREA_N = [1, 2, 4, 8, 16, 32, 64];
const AREA_X = Array.from({ length: 41 }, (_, i) => i / 40);
const AREA_F = AREA_X.map((x) => Math.exp(x));
/** 分段线性插值（与后端 approx_y 的定义一致：节点之间是直线）。 */
function piecewiseLinear(x: number, k: number): number {
  const h = 1 / k;
  const i = Math.min(k - 1, Math.floor(x / h));
  const left = Math.exp(i * h);
  const right = Math.exp((i + 1) * h);
  return left + ((x - i * h) / h) * (right - left);
}

const AREA_STRINGS = {
  method: "trapezoid",
  method_label: "复合梯形",
  func: "exp",
  func_label: "e^x（解析，所有导数有界）",
  smoothness: "C^∞：所有阶导数在 [0,1] 上有界",
  order_note: "理论参考线的斜率是公式的阶 p=2；实测斜率只在渐近窗口内成立。",
  smoothness_note: "光滑性上限给出实测阶的预测 min(p, α+1)=2.00（α+1=inf）。",
  floor_note: "误差曲线在 3.02e-16 处截断（u·max|f| 量级）：log 轴画不出 0。",
  fit_note: "实测斜率取自 n∈[10,64]、R²=1.000000 的窗口。",
  data_source: "backend",
  params_json: '{"func":"exp","method":"trapezoid","n":64}',
};

const AREA_META = {
  step: 64,
  is_adaptive: 0,
  panels: 64,
  panels_requested: 64,
  n_max: 64,
  rule_order: 2,
  value: 1.7183168,
  exact: 1.7182818,
  error_current: 3.48e-5,
  finite: 1,
  n_evals: 65,
  slope_measured: -1.99992,
  slope_theory: -2,
  slope_theory_smoothness: -2,
  fit_r2: 1,
  fit_start: 10,
  fit_end: 64,
  error_floor: 3.02e-16,
  clipped_points: 0,
  order_cap: null,
  sum_w: 1,
  abs_sum_w: 1,
  gamma: 1,
  n_negative_weights: 0,
  fitted_constant: 0.14317578,
  t: 0,
};

async function areaClassic(over: Override = {}): Promise<Frame> {
  return makeFrame(
    {
      viz: "area_convergence",
      arrays: [
        { name: "x", data: AREA_X, unit: "", label: "采样点" },
        { name: "f", data: AREA_F, unit: "", label: "被积函数" },
        { name: "approx_x", data: AREA_X, unit: "", label: "分段逼近采样点" },
        { name: "approx_y", data: AREA_X.map((x) => piecewiseLinear(x, 64)), unit: "", label: "分段逼近" },
        { name: "node_x", data: [0, 0.25, 0.5, 0.75, 1], unit: "", label: "求值节点" },
        { name: "node_y", data: [1, Math.exp(0.25), Math.exp(0.5), Math.exp(0.75), Math.E], unit: "", label: "节点值" },
        { name: "panel_edge", data: [0, 0.25, 0.5, 0.75, 1], unit: "", label: "面板边界" },
        { name: "panel_count", data: AREA_N, unit: "1", label: "面板数" },
        { name: "error_curve", data: AREA_N.map((n) => 0.14317578 / (n * n)), unit: "1", label: "误差" },
        { name: "error_evals", data: AREA_N.map((n) => n + 1), unit: "1", label: "求值次数" },
        { name: "fit_curve", data: AREA_N.map((n) => 0.14317578 / (n * n)), unit: "1", label: "拟合曲线" },
        { name: "theory_curve", data: AREA_N.map((n) => 0.14317578 / (n * n)), unit: "1", label: "理论曲线" },
        { name: "local_order", data: AREA_N.map(() => 2), unit: "1", label: "局部阶" },
        { name: "weight", data: AREA_N.map(() => 1), unit: "1", label: "权重" },
      ],
      meta: AREA_META,
      strings: AREA_STRINGS,
      critical: {},
    },
    over,
  );
}

/** 窄峰 + 自适应：估计曲线贴地板、真误差停在 3.5e-2（与 pitfalls 第 6 条同一个故事）。 */
async function areaAdaptive(over: Override = {}): Promise<Frame> {
  const evals = [5, 7, 9, 11, 13, 15, 17];
  const est = [1.11e-16, 1.11e-16, 1.02e-11, 4.96e-10, 2.99e-10, 3.45e-9, 5.39e-9];
  const tr = [3.54e-2, 3.54e-2, 3.54e-2, 3.54e-2, 3.54e-2, 4.72e-6, 4.77e-7];
  return makeFrame(
    {
      viz: "area_convergence",
      arrays: [
        { name: "x", data: AREA_X, unit: "", label: "采样点" },
        { name: "f", data: AREA_X.map((x) => Math.exp(-(((x - 0.3) / 0.02) ** 2))), unit: "", label: "被积函数" },
        { name: "interval_lo", data: [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], unit: "", label: "区间左端" },
        { name: "interval_hi", data: [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1], unit: "", label: "区间右端" },
        { name: "interval_depth", data: [1, 1, 2, 3, 4, 5, 6], unit: "1", label: "递归深度" },
        { name: "error_estimate_curve", data: est, unit: "1", label: "估计误差" },
        { name: "error_true_curve", data: tr, unit: "1", label: "真误差" },
        { name: "evals_cum", data: evals, unit: "1", label: "累计求值次数" },
        { name: "partial", data: est.map((e) => 0.035449 + e), unit: "1", label: "部分和" },
        { name: "depth_level", data: [1, 2, 3, 4, 5, 6], unit: "1", label: "深度" },
        { name: "depth_count", data: [3, 2, 1, 1, 1, 1], unit: "1", label: "该深度的区间数" },
        { name: "depth_evals", data: [9, 7, 5, 3, 3, 3], unit: "1", label: "该深度的求值次数" },
        { name: "worst_index", data: [3, 4], unit: "1", label: "最差区间下标" },
        { name: "worst_est", data: [3.45e-9, 5.39e-9], unit: "1", label: "最差区间估计" },
      ],
      meta: {
        is_adaptive: 1,
        step: 7,
        n_intervals_total: 7,
        n_evals_total: 17,
        max_depth_used: 6,
        tol: 1e-6,
        error_estimate_total: 5.39e-9,
        error_true_total: 4.77e-7,
        estimator_over_true: 0.0113,
        value: 0.035449083,
        exact: 0.035449077,
        converged: 1,
        error_floor: 1.11e-16,
        clipped_points: 2,
        budget: 64,
        t: 7,
      },
      strings: { ...AREA_STRINGS, method: "adaptive", method_label: "自适应 Simpson（嵌套法则 + 容差密度）", func: "spike", func_label: "exp(-((x−0.3)/0.02)²)（窄高峰）" },
      critical: {},
    },
    over,
  );
}

test("2.4 area_convergence：经典模式出两张图，两条理论线都画出来", async () => {
  const scene = ok(areaConvergenceModel(await areaClassic(), OPTS), "area_convergence");
  assert.deepEqual(
    scene.axes.map((a) => a.id),
    ["main", "err"],
  );
  assert.equal(scene.axes[0]?.autoscaled, false);
  assert.equal(scene.axes[1]?.autoscaled, true);
  assert.equal(scene.axes[1]?.yscale, "log");
  assert.ok(hasKind(scene, "main", "scatter"), "左列要有求值节点");
  // 规则 2：实测 + 实测斜率 + 理论斜率三层必须在 err 面板上。
  assert.ok(layersOf(scene, "err").some((l) => l.kind === "loglogfit"), "err 面板缺 loglogfit（规则 2）");
  const fit = layersOf(scene, "err").find((l) => l.kind === "loglogfit") as Extract<Layer, { kind: "loglogfit" }>;
  assert.ok(fit.seriesLabel.length > 0 && fit.theoryLabel.length > 0, "loglogfit 的两条线都要有文字标注");
  assert.equal(fit.measuredSlope, -1.99992);
  assert.equal(fit.theorySlope, -2);
  // 光滑性上限那条参考线（斜率来自 meta.slope_theory_smoothness）。
  const body = textBody(scene);
  assert.ok(has(body, "光滑性上限 min(p,α+1)=-2"));
  assert.ok(has(body, "实测斜率 -1.99992"));
  assertNoClippedText(scene, "area_convergence");
});

test("2.4 area_convergence：adaptive 模式把「估计误差」与「真误差」同屏（陷阱 1）", async () => {
  const scene = ok(areaConvergenceModel(await areaAdaptive(), OPTS), "area_convergence");
  const body = textBody(scene);
  assert.match(body, /估计误差估的是嵌套法则/);
  assert.ok(has(body, "估计/真误差 比：最小"), "必须给出实测比值区间（不是只说「估计器不准」）");
  assert.match(body, /不参与比值/, "贴地板的点必须如实说明被排除，否则比值只是浮点地板");
  assert.ok(layersOf(scene, "err").length >= 3, "估计误差、真误差、tol 三条线");
  assert.ok(has(body, "整段收尾：估计 5.39e-9 vs 真 4.77e-7"));
  assertNoClippedText(scene, "area_convergence");
});

test("2.4 area_convergence：缺 slope_measured 时不硬凑斜率线，并降级说明", async () => {
  const scene = ok(areaConvergenceModel(await areaClassic({ dropMeta: ["slope_measured", "slope_theory"], dropStrings: ["fit_note"] }), OPTS), "area_convergence");
  assert.ok(!layersOf(scene, "err").some((l) => l.kind === "loglogfit"), "没有可信斜率就不该有 loglogfit");
  assert.match(badgeTexts(scene), new RegExp(DEGRADED_PREFIX));
  assert.ok(has(textBody(scene), "没有实测斜率"));
});

test("2.4 area_convergence：缺数组/长度不一致/pitfalls 太短 → ModelInputError", async () => {
  await assert.rejects(async () => areaConvergenceModel(await areaClassic({ drop: ["error_curve"] }), OPTS), ModelInputError);
  await assert.rejects(
    async () => areaConvergenceModel(await areaClassic({ arrays: { panel_count: [1, 2, 4] } }), OPTS),
    ModelInputError,
  );
  await assert.rejects(async () => areaConvergenceModel(await areaClassic({ pitfalls: "太短" }), OPTS), ModelInputError);
});

// --------------------------------------------------------------------------- //
// gauss_vs_equispaced
// --------------------------------------------------------------------------- //

const G_DEGREE = Array.from({ length: 19 }, (_, k) => k);
/** Gauss：k ≤ 2n−1（15）在地板；k ≥ 16 跳起来。等距：偶数 k ≥ 8 就失败（奇数次靠对称平凡精确）。 */
const G_FLOOR = 2.22e-16;
const G_ERR_GAUSS = G_DEGREE.map((k) => (k <= 15 ? G_FLOOR : k === 16 ? 4.655e-5 : 1.971e-4));
const G_ERR_EQUI = G_DEGREE.map((k) => (k % 2 === 1 ? G_FLOOR : k >= 8 ? 8.075e-3 * Math.pow(2, k - 8) : G_FLOOR));
const G_M = Array.from({ length: 19 }, (_, i) => i + 2);
const G_COND_EQUI = G_M.map((m) => (m < 9 ? 1 : Math.pow(1.26, m - 9) * 1.4512));
const G_COND_GAUSS = G_M.map(() => 1);

async function gaussFrame(over: Override = {}): Promise<Frame> {
  const n = 8;
  return makeFrame(
    {
      viz: "gauss_vs_equispaced",
      arrays: [
        { name: "node_gauss", data: [0.0199, 0.1017, 0.2372, 0.4083, 0.5917, 0.7628, 0.8983, 0.9801], unit: "", label: "Gauss 节点" },
        { name: "weight_gauss", data: [0.0506, 0.1112, 0.1569, 0.1813, 0.1813, 0.1569, 0.1112, 0.0506], unit: "", label: "Gauss 权重" },
        { name: "node_equi", data: [0, 0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571, 1], unit: "", label: "等距节点" },
        { name: "weight_equi", data: [0.0435, 0.1304, 0.1087, 0.1739, 0.1739, 0.1087, 0.1304, 0.0435], unit: "", label: "等距权重" },
        { name: "degree", data: G_DEGREE, unit: "1", label: "多项式次数 k" },
        { name: "error_moment_gauss", data: G_ERR_GAUSS, unit: "1", label: "Gauss 矩误差" },
        { name: "error_moment_equi", data: G_ERR_EQUI, unit: "1", label: "等距矩误差" },
        { name: "m_grid", data: G_M, unit: "1", label: "点数 m" },
        { name: "cond_equi", data: G_COND_EQUI, unit: "1", label: "等距 Γ" },
        { name: "cond_gauss", data: G_COND_GAUSS, unit: "1", label: "Gauss Γ" },
        { name: "neg_count_equi", data: G_M.map((m) => (m < 9 ? 0 : ((m - 7) >> 1) + 1)), unit: "1", label: "负权重个数" },
        { name: "min_weight_equi", data: G_M.map((m) => (m < 9 ? 0.035 : -0.009 * (m - 8))), unit: "", label: "最小权重" },
        { name: "conv_n", data: [4, 8, 12, 16], unit: "1", label: "节点数" },
        { name: "error_conv_gauss", data: [1e-6, 1e-12, 1e-16, 1e-16], unit: "1", label: "Gauss 误差" },
        { name: "error_conv_equi", data: [1e-2, 1e-3, 4e-4, 2e-4], unit: "1", label: "等距误差" },
      ],
      meta: {
        step: 18,
        n,
        interval_a: 0,
        interval_b: 1,
        degree_exact_limit: 15,
        degree_2n: 16,
        error_at_limit: 0,
        error_beyond_limit: 4.655e-5,
        failure_factor: 2.1e11,
        error_equi_at_limit: 2.9e-16,
        error_equi_beyond: 8.075e-3,
        sum_w_gauss: 1,
        sum_w_equi: 1,
        gamma_gauss: 1,
        gamma_equi_n: 1,
        n_negative_gauss: 0,
        n_negative_equi: 0,
        min_weight_gauss: 0.05061,
        min_weight_equi: 0.03486,
        error_floor: G_FLOOR,
        clipped_points: 4,
        conv_rate_gauss: -6.737,
        conv_order_equi: -1.99977,
        error_beyond_relative: 5.79e-11,
        knots_m9_gamma: 1.4512,
        knots_m19_gamma: 175.463,
        t: 18,
      },
      strings: {
        func: "exp",
        func_label: "e^x（解析，所有导数有界）",
        exactness_note: "k ≤ 2n−1 的矩误差在舍入地板、k = 2n 当场失效——两个方向都画出来。",
        condition_note: "Γ=Σ|w|/Σw 是函数值相对误差到积分相对误差的放大因子；Gauss 全正 ⇒ Γ≡1。",
        budget_note: "收敛对比两边用相同的节点数。",
        law_note: "Gauss 是半对数直线（几何收敛），等距梯形是双对数直线（代数收敛）。",
        sum_note: "Σw=b−a 只说明常数被积对。",
        floor_note: "矩误差曲线按地板截断（log 轴画不出 0）。",
        data_source: "backend",
        params_json: '{"func":"exp","n":8}',
      },
      critical: { n: 9 },
    },
    over,
  );
}

test("2.4 gauss_vs_equispaced：节点/权重、矩误差、Γ 三块都要有，且 2n 处失效同屏", async () => {
  const scene = ok(gaussVsEquispacedModel(await gaussFrame(), OPTS), "gauss_vs_equispaced");
  assert.deepEqual(
    scene.axes.map((a) => a.id),
    ["nodes", "exact", "cond"],
  );
  assert.equal(scene.axes[1]?.yscale, "log");
  assert.equal(scene.axes[2]?.yscale, "log");
  assert.ok(hasKind(scene, "nodes", "scatter"), "左上要有节点/权重散点");
  assert.ok(layersOf(scene, "exact").length >= 3, "Gauss 曲线 + 等距曲线 + 2n−1 竖线");
  assert.ok(layersOf(scene, "cond").length >= 3, "等距 Γ + Gauss Γ≡1 + 临界竖线");
  const body = textBody(scene);
  assert.ok(has(body, "Gauss 精确到 k ≤ 2n−1 = 15（帧实测 0）；k = 2n 失效 4.655e-5（因子 2.1e11）"));
  assert.ok(has(body, "区间不同是刻意的"), "节点区间与矩误差区间不同必须在屏上说明");
  assert.ok(has(body, "负权重 0 / 0"));
  assert.ok(has(body, "⚠ 奇数次的精确是平凡的"));
  assert.ok(has(body, "等距 m=9：Γ=1.4512；m=19：Γ=175.463"));
  assert.ok(has(body, "同一 ε=1×10⁻⁸ 的扰动 → 积分相对误差"), "Γ 的结论必须落到放大多少上");
  assert.match(badgeTexts(scene), /理论临界 9/);
  assertNoClippedText(scene, "gauss_vs_equispaced");
});

test("2.4 gauss_vs_equispaced：n=20 时负权重与 Γ 都上来，文字如实跟上", async () => {
  const frame = await gaussFrame({
    meta: { n: 20, gamma_equi_n: 63.25, n_negative_equi: 2, degree_exact_limit: 39, degree_2n: 40 },
  });
  const scene = ok(gaussVsEquispacedModel(frame, OPTS), "gauss_vs_equispaced");
  assert.ok(has(textBody(scene), "当前 n：Gauss Γ=1、等距 Γ=63.25"));
  assert.match(badgeTexts(scene), /等距规则出现 2 个负权重/);
});

test("2.4 gauss_vs_equispaced：缺 meta.n 不猜节点数（写 ? + 降级角标）", async () => {
  const scene = ok(gaussVsEquispacedModel(await gaussFrame({ dropMeta: ["n"] }), OPTS), "gauss_vs_equispaced");
  assert.match(badgeTexts(scene), new RegExp(DEGRADED_PREFIX));
  assert.match(badgeTexts(scene), /缺 n/);
  assert.ok(has(badgeTexts(scene), "帧 meta 缺 n"));
});

test("2.4 gauss_vs_equispaced：缺数组 → ModelInputError", async () => {
  await assert.rejects(async () => gaussVsEquispacedModel(await gaussFrame({ drop: ["cond_equi"] }), OPTS), ModelInputError);
});

// --------------------------------------------------------------------------- //
// romberg_table
// --------------------------------------------------------------------------- //

/** 三角长表（行主序 `row(row+1)/2 + col`）：第 j 列 ~ h^{2j+2}，对角线触地板。 */
function rombergTable(levels: number): { j: number[]; k: number[]; e: number[] } {
  const j: number[] = [];
  const k: number[] = [];
  const e: number[] = [];
  for (let row = 0; row < levels; row += 1) {
    for (let col = 0; col <= row; col += 1) {
      j.push(col);
      k.push(row);
      const v = 0.14 * Math.pow(2, -2 * (col + 1) * (row - col + 1));
      e.push(Math.max(v, 1.91e-16));
    }
  }
  return { j, k, e };
}

async function rombergFrame(over: Override = {}, levels = 8): Promise<Frame> {
  const tab = rombergTable(levels);
  return makeFrame(
    {
      viz: "romberg_table",
      arrays: [
        { name: "column_j", data: tab.j, unit: "1", label: "列号" },
        { name: "column_k", data: tab.k, unit: "1", label: "行号" },
        { name: "column_value", data: tab.e.map((v) => 1.71828 + v), unit: "1", label: "表中值" },
        { name: "error_table", data: tab.e, unit: "1", label: "表中误差" },
        { name: "level", data: Array.from({ length: levels }, (_, i) => i), unit: "1", label: "层号" },
        { name: "n_sub", data: Array.from({ length: levels }, (_, i) => 2 ** i), unit: "1", label: "子区间数" },
        { name: "evals", data: Array.from({ length: levels }, (_, i) => 2 ** i + 1), unit: "1", label: "求值次数" },
        { name: "diag_error", data: Array.from({ length: levels }, (_, i) => Math.max(0.14 * 2 ** (-2 * (i + 1)), 1.91e-16)), unit: "1", label: "对角线误差" },
        { name: "order_j", data: [0, 1, 2, 3], unit: "1", label: "列号" },
        { name: "order_measured", data: [1.99761, 3.9946, 5.97714, 7.971], unit: "1", label: "实测阶" },
        { name: "order_theory", data: [2, 4, 6, 8], unit: "1", label: "理论阶" },
        { name: "gain_j", data: [1, 1, 1], unit: "1", label: "列号" },
        { name: "gain_k", data: [1, 2, 3], unit: "1", label: "行号" },
        { name: "gain", data: [14.4, 58.3, 235.5], unit: "1", label: "同层相邻列误差比" },
        { name: "coef", data: [1, -0.25, 0.111, -0.0625, 0.04, -0.028, 0.02, -0.015], unit: "1", label: "外推系数" },
        { name: "coef_index", data: [0, 1, 2, 3, 4, 5, 6, 7], unit: "1", label: "系数下标" },
      ],
      meta: {
        step: levels,
        levels,
        func_smooth: 1,
        value: 1.7182818,
        exact: 1.7182818,
        diag_error_current: 1.91e-16,
        diag_error_prev: 3.06e-16,
        error_floor: 1.91e-16,
        clipped_points: 6,
        coefficient_abs_sum: 1.9691802,
        coefficient_sum: 1,
        n_columns: levels,
        order_col0: 1.99761,
        order_col1: 3.9946,
        order_col2: 5.97714,
        order_col3: 7.971,
        lambda_cusp: null,
        t: 10,
      },
      strings: {
        func: "exp",
        func_label: "e^x（解析，所有导数有界）",
        column_note: "第 j 列只有 m−j 个可用点：列号越大、可用行越少，m−j<3 的列在帧里被剔除（不编一个 0）。",
        window_note: "每列的拟合窗口都剔掉触到舍入地板的行（不剔时第 3 列实测 6.96、剔掉得 7.97，理论 8）。",
        floor_note: "对角线在 1.91e-16 处触地板：A_j 有界（→1.9692603），截断误差已经小到无意义。",
        nonsmooth_note: "被积函数足够光滑，Romberg 的列阶应当逐列 +2。",
        data_source: "backend",
        params_json: '{"func":"exp","levels":8}',
      },
      critical: { levels: 6 },
    },
    over,
  );
}

test("2.4 romberg_table：逐列误差 + 对角线 + 实测/理论阶，地板竖线只在揭示后出现", async () => {
  const scene = ok(rombergTableModel(await rombergFrame(), OPTS), "romberg_table");
  assert.deepEqual(
    scene.axes.map((a) => a.id),
    ["conv", "order"],
  );
  assert.equal(scene.axes[0]?.xscale, "log");
  assert.equal(scene.axes[0]?.yscale, "log");
  const convLines = layersOf(scene, "conv").filter((l) => l.kind === "line");
  assert.ok(convLines.length >= 8 + 1, `每列一条 + 对角线，收到 ${convLines.length}`);
  assert.ok(layersOf(scene, "order").length >= 2, "实测阶与理论阶两条线");
  const body = textBody(scene);
  assert.ok(has(body, "实测/理论（j=0…3）：1.99761/2"));
  assert.ok(has(body, "外推的代价：最后一列 Σ|c_i| = 1.9691802"));
  assert.ok(has(body, "Σ|c_i|=1、Σ|c_i|→1.9692603"), "manifest 的极限值要在屏上");
  assert.ok(has(body, "⚠ 竖线 = 第 6 层"));
  assertNoClippedText(scene, "romberg_table");
});

test("2.4 romberg_table：没有任何一列有阶时留白（不编 0 阶），并降级说明", async () => {
  const scene = ok(
    rombergTableModel(await rombergFrame({ drop: ["order_j", "order_measured", "order_theory"], meta: { step: 2, levels: 2 } }), OPTS),
    "romberg_table",
  );
  assert.ok(!layersOf(scene, "order").some((l) => l.kind === "line"), "没有阶就不画线");
  assert.match(badgeTexts(scene), new RegExp(DEGRADED_PREFIX));
  assert.ok(has(textBody(scene), "本帧没有任何一列凑够可拟合的点"));
});

test("2.4 romberg_table：缺 level/error_table → ModelInputError", async () => {
  await assert.rejects(async () => rombergTableModel(await rombergFrame({ drop: ["error_table"] }), OPTS), ModelInputError);
});

// --------------------------------------------------------------------------- //
// singular_endpoint
// --------------------------------------------------------------------------- //

const S_M = Array.from({ length: 64 }, (_, i) => i + 1);
const S_NODES = S_M.map((m) => 2 * m + 1);
const S_FLOOR = 1.49e-8;
/** tanh–sinh：先双指数下降，到 m≈51（节点 103）后贴地板；窗口之外后端填的是**占位值**。 */
const S_ERR = S_M.map((m) => (m < 51 ? 0.2 * Math.exp(-0.28 * m) : S_FLOOR));
const S_XMIN = S_M.map((m) => Math.max(0.5 * 2 ** -53, 0.45 * Math.exp(-0.26 * m)));
const S_TAIL = S_XMIN.map((x) => 2 * Math.sqrt(x));
const S_SAT = S_M.map((m) => (m < 51 ? 0 : 2 * (m - 50)));

async function singularFrame(over: Override = {}, subAvailable = true): Promise<Frame> {
  return makeFrame(
    {
      viz: "singular_endpoint",
      arrays: [
        { name: "ts_m", data: S_M, unit: "1", label: "每侧节点数 m" },
        { name: "ts_nodes", data: S_NODES, unit: "1", label: "总节点数" },
        { name: "error_ts", data: S_ERR, unit: "1", label: "tanh–sinh 误差" },
        { name: "ts_saturated", data: S_SAT, unit: "1", label: "饱和节点数" },
        { name: "ts_xmin", data: S_XMIN, unit: "", label: "最靠近端点的节点" },
        { name: "ts_tail", data: S_TAIL, unit: "1", label: "未采样尾巴" },
        { name: "sub_nodes", data: S_NODES, unit: "1", label: "换元后节点数" },
        { name: "error_sub", data: S_M.map((m) => Math.max(S_FLOOR, 0.5 * Math.pow(m, -2))), unit: "1", label: "换元 + 等距误差" },
        { name: "floor_nodes", data: S_NODES, unit: "1", label: "地板参考线横坐标" },
        { name: "floor_value", data: S_NODES.map(() => S_FLOOR), unit: "1", label: "地板理论值" },
        { name: "x", data: AREA_X, unit: "", label: "采样点" },
        { name: "f", data: AREA_X.map((x) => (x === 0 ? 1 : (1 + x) / Math.sqrt(x))), unit: "", label: "被积函数" },
        { name: "f_defined", data: AREA_X.map((x) => (x === 0 ? 0 : 1)), unit: "1", label: "是否有定义" },
      ],
      meta: {
        step: 64,
        nodes: 129,
        h: 0.0625,
        error_current: S_ERR[63],
        n_saturated: S_SAT[63],
        x_min: S_XMIN[63],
        tail_estimate: S_TAIL[63],
        tail_ratio: 0.54,
        tail_theory: S_FLOOR,
        error_floor: S_FLOOR,
        clipped_points: 3,
        sub_order: subAvailable ? 2 : null,
        ts_slope_final: -6.28984,
        ts_slope_r2: 0.268531,
        raw_equispaced_finite: 0,
        raw_equispaced_nonfinite: 1,
        raw_equispaced_value: null,
        raw_equispaced_error: null,
        constant_node_count: 129,
        sub_available: subAvailable ? 1 : 0,
        f_display_cap: 4.53,
        f_undefined_points: 1,
        t: 256,
      },
      strings: {
        func: "inv_sqrt",
        func_label: "(1+x)/√x（端点本身无界）",
        h_label: "h = 1/16 = 0.0625",
        smoothness: "x→0⁺ 时 f 本身发散（α=−1/2）：等距规则在 x=0 处的函数值无定义",
        floor_note: "tanh–sinh 的地板不是 u 而是未被采样的尾巴 ∫₀^{x_min}f：x_min≈u/2。",
        saturation_note: "饱和节点是舍入到 1 的那些点：权重下溢到 0，贡献必须按 0 计。",
        display_note: "被积函数曲线在未定义点处截断，并用 f_defined 掩码标出。",
        raw_note: "等距规则在原始被积函数上没有定义（x=0 处 f 无界）。",
        sub_note: "对照曲线用 x=t² 换元后的被积函数，同一套梯形法逐次加密。",
        data_source: "backend",
        params_json: '{"func":"inv_sqrt","h":"1/16","n":129}',
      },
      critical: { n: 103 },
    },
    over,
  );
}

test("2.4 singular_endpoint：误差曲线按 step 截断（不画占位值），尾巴/地板/对照同屏", async () => {
  const frame = await singularFrame({ meta: { step: 40 } });
  const scene = ok(singularEndpointModel(frame, OPTS), "singular_endpoint");
  assert.deepEqual(
    scene.axes.map((a) => a.id),
    ["conv", "tail"],
  );
  assert.equal(scene.axes[0]?.yscale, "log");
  assert.equal(scene.axes[1]?.yscale, "log");
  const main = layersOf(scene, "conv");
  const tanh = main
    .filter((l) => l.kind === "line")
    .flatMap((l) => (l as Extract<Layer, { kind: "line" }>).series)
    .find((s) => s.label === "tanh–sinh");
  assert.ok(tanh !== undefined, "缺 tanh–sinh 曲线");
  assert.equal(tanh.x.length, 40, "必须截断到 meta.step=40，不能把地板占位值当数据画出来");
  assert.ok(main.some((l) => l.kind === "scatter"), "当前 m 的点要标出来");
  assert.ok(layersOf(scene, "tail").length >= 3, "尾巴、x_min、地板三条线");
  {
    // 饱和节点只有到 m≥51 才出现：未饱和时**不画**那条恒为 0 的线（log 轴也画不出 0）。
    const late = ok(singularEndpointModel(await singularFrame({ meta: { step: 64 } }), OPTS), "singular_endpoint");
    assert.ok(layersOf(late, "tail").length >= 4, "饱和开始后要多出饱和节点数那条线");
  }
  const body = textBody(scene);
  assert.ok(has(body, "⚠ 同一个 log 轴上是三种单位"));
  assert.ok(has(body, "误差/尾巴 = 0.54"));
  assert.ok(has(body, "帧数组在已揭示窗口之外填的是地板占位值"));
  assert.ok(has(body, "换元对照的实测阶 2（理论 2.0000000）"));
  assert.match(badgeTexts(scene), /原始等距规则没有定义/);
  assert.equal(scene.overflow.flag, true, "被积函数的未定义点必须标出来（规则 4）");
  assertNoClippedText(scene, "singular_endpoint");
});

test("2.4 singular_endpoint：换元不可用时对照曲线留白（不画贴现地板的假曲线）", async () => {
  const scene = ok(singularEndpointModel(await singularFrame({}, false), OPTS), "singular_endpoint");
  const hasSub = layersOf(scene, "conv")
    .filter((l) => l.kind === "line")
    .flatMap((l) => (l as Extract<Layer, { kind: "line" }>).series)
    .some((s) => (s.label ?? "").includes("换元"));
  assert.ok(!hasSub, "不该有换元曲线");
  assert.ok(has(textBody(scene), "没有可用的有限区间换元"));
});

test("2.4 singular_endpoint：缺 floor_value → ModelInputError", async () => {
  await assert.rejects(async () => singularEndpointModel(await singularFrame({ drop: ["floor_value"] }), OPTS), ModelInputError);
});

// --------------------------------------------------------------------------- //
// 章节级
// --------------------------------------------------------------------------- //

test("2.4 章：manifest、models 与 modelFor 一一对应", () => {
  assert.deepEqual(
    manifest.vizzes.map((v) => v.id).sort(),
    [...VIZ_IDS].sort(),
  );
  assert.deepEqual(Object.keys(models).sort(), [...VIZ_IDS].sort());
  for (const id of VIZ_IDS) {
    assert.equal(modelFor(id), models[id], `${id} 的 modelFor 与 models 不一致`);
  }
  assert.equal(modelFor("nope"), null, "未知 viz 必须返回 null 而不是抛错");
});

test("2.4 章：manifest 声明的坐标轴/自适应标记与 model 产出的一致", async () => {
  const frames: Record<string, Frame> = {
    area_convergence: await areaClassic(),
    gauss_vs_equispaced: await gaussFrame(),
    romberg_table: await rombergFrame(),
    singular_endpoint: await singularFrame(),
  };
  for (const id of VIZ_IDS) {
    const spec = specOf(id);
    const scene = ok(models[id]!(frames[id] as Frame, OPTS), id);
    assert.deepEqual(
      scene.axes.map((a) => a.id),
      spec.axes.map((a) => a.id),
      `${id} 的轴 id 与 manifest 不一致`,
    );
    for (const a of scene.axes) {
      const declared = spec.axes.find((d) => d.id === a.id) ?? fail(`${id} 多出轴 ${a.id}`);
      assert.equal(a.autoscaled, declared.autoscaled, `${id}/${a.id} 的自适应标记与 manifest 不一致`);
      assert.equal(a.title, declared.title, `${id}/${a.id} 的标题与 manifest 不一致`);
      assert.equal(a.ylabel, declared.ylabel, `${id}/${a.id} 的 ylabel 与 manifest 不一致`);
      if (declared.scale === "log") assert.equal(a.yscale, "log", `${id}/${a.id} 应为对数纵轴`);
    }
  }
});
