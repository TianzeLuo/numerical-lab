/**
 * 2.1 章（插值与多项式逼近）model 层测试 —— **自足**：不依赖 Python，也不依赖 `data/frames` 是否存在。
 *
 * 夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，字段集合（数组名/dtype/shape/unit/label、
 * meta/strings/critical 键）逐字照抄 `server/chapters/2.1.py` 的真实帧；真实帧在的话再对拍一遍
 * （见文件末尾，缺帧时整组跳过）。数值一律是**合成值**：本层只搬运帧里的数，测试关心的是
 * "搬得对不对、缺字段有没有明说"。
 *
 * 本文件专门守 2.1 最容易画错的几条：
 * - **log 轴上的 0**：帧里恰为 0 的误差点不进对数层（否则曲线被抹掉），但必须在正文里报数；
 * - **自适应要明说**（规则 4）：`autoscaled` 与 manifest 一致，且挂"自适应"角标；
 * - **临界参考线**：竖线落在帧 `critical` 给的 n 上，并挂 `criticalBadge`；
 * - **requiredAux 真的画出来了**：误差/残差量必须能在对应轴上找到（按**数值**认，不按 label——
 *   `LineSeries.label` 会被渲染器画在曲线末端，本章一律不写长 label）；
 * - **时间语义**：`t` 只用于选帧，当前 k/n 一律取帧 meta；
 * - **降级不静默**：meta 缺键 / 值为 null → 少画什么就写明什么（status 角标点名）。
 */

import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";

import { formatNumber, linspace } from "../shared/src/axes.js";
import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions, type SceneModel } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { sceneToJson, validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import { barycentricVsVandermondeModel } from "../chapters/2.1/barycentric_vs_vandermonde.model.js";
import { errorCurveModel } from "../chapters/2.1/error_curve.model.js";
import { manifest, modelFor, models } from "../chapters/2.1/index.js";
import { lebesgueGrowthModel } from "../chapters/2.1/lebesgue_growth.model.js";
import { rungeNodesModel } from "../chapters/2.1/runge_nodes.model.js";

const PITFALLS =
  "它会怎样骗你：只画最大误差会把 Runge 现象讲反（最大误差是一个数，它的位置在两端）；" +
  "对数轴上的 0 是假的——插值误差在节点上恰好等于 0，直接画会得到 -inf 并把整条曲线抹掉；" +
  "严格包络不是误差估计，它可能宽松好几个量级。";

/** 一行数组 = `[名字, 数据, 标签]`；本章所有数组的 `unit` 都是 `"1"`（与真实帧一致）。 */
type Row = readonly [name: string, data: ArrayLike<number>, label: string];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical: Record<string, number>;
  /** 缺省用 `PITFALLS`；只有"pitfalls 过短"那条用例需要覆盖它。 */
  pitfalls?: string;
}

async function makeFrame(def: FrameDef): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const [name, data, label] of def.arrays) {
    const typed = Float64Array.from(data as ArrayLike<number>);
    specs.push({ name, dtype: "f8", shape: [data.length], order: "C", offset: 0, nbytes: typed.byteLength, unit: "1", label });
    arrays.set(name, typed);
  }
  const frame = new Frame({
    chapter: "2.1",
    viz: def.viz,
    kind: "frame",
    codec: "none",
    meta: def.meta,
    strings: def.strings,
    critical: def.critical,
    pitfalls: def.pitfalls ?? PITFALLS,
    specs,
    arrays,
  });
  return decodeNlf(await encodeNlf(frame));
}

interface Override {
  arrays?: Record<string, ArrayLike<number>>;
  meta?: Record<string, number | boolean | null>;
  strings?: Record<string, string>;
  drop?: readonly string[];
  dropMeta?: readonly string[];
  dropStrings?: readonly string[];
  pitfalls?: string;
}

/** 覆盖/删字段；`meta` 里给 `null` 用来造"键在但值是 null"的降级帧（`.nlf` 允许 null）。 */
function tweak(def: FrameDef, ov: Override): FrameDef {
  const arrays = def.arrays
    .filter(([name]) => !(ov.drop ?? []).includes(name))
    .map((row): Row => (ov.arrays && row[0] in ov.arrays ? [row[0], ov.arrays[row[0]], row[2]] : row));
  const meta = { ...def.meta };
  for (const k of ov.dropMeta ?? []) delete meta[k];
  const strings = { ...def.strings };
  for (const k of ov.dropStrings ?? []) delete strings[k];
  return {
    ...def,
    arrays,
    meta: { ...meta, ...ov.meta },
    strings: { ...strings, ...ov.strings },
    pitfalls: ov.pitfalls ?? def.pitfalls,
  };
}

// --------------------------------------------------------------------------- //
// 四个 viz 的夹具（字段集合照抄真实帧）
// --------------------------------------------------------------------------- //

const XS = linspace(-1, 1, 801);
const COS_F = Array.from(XS, (x) => Math.cos(1.5 * x));
/** 合成误差：在 `zeros` 指定的下标处**恰好为 0**，专门测"log 轴上的 0"。 */
const err = (scale: number, zeros: readonly number[] = []): Float64Array =>
  XS.map((x, i) => (zeros.includes(i) ? 0 : scale * Math.abs(Math.sin(7 * x)) + 1e-6));
const N = (count: number): number[] => Array.from({ length: count }, (_, i) => i + 1);
const geo = (count: number, a: number, b: number): number[] => N(count).map((n) => a * Math.pow(b, n - count));

const RUNGE_DEF: FrameDef = {
  viz: "runge_nodes",
  arrays: [
    ["x_grid", XS, "x"],
    ["f_values", COS_F, "f(x)"],
    ["p_selected", XS.map((x) => Math.cos(1.5 * x) + 0.05 * Math.sin(9 * x)), "等距节点插值多项式（k 个节点）"],
    ["p_other", XS.map((x) => Math.cos(1.5 * x) + 0.01 * Math.sin(20 * x)), "Chebyshev 节点插值多项式（同样 k 个节点）"],
    ["error_selected", err(0.04, [200, 600]), "|f-p|（当前分布，已夹地板）"],
    ["error_other", err(0.01), "|f-p|（另一种分布，已夹地板）"],
    ["error_selected_raw", err(0.04, [200, 600]), "|f-p|（真值，节点上恰为 0）"],
    ["envelope_selected", XS.map((x) => 5e-3 * (1 + Math.cos(3 * x))), "严格误差包络 sup|f^(k+1)|/(k+1)!·|∏(x-x_i)|"],
    ["nodes_selected_x", [-0.5, 0.5], "当前分布的节点"],
    ["nodes_selected_y", [Math.cos(-0.75), Math.cos(0.75)], "节点上的函数值"],
    ["nodes_other_x", [-0.8, 0.8], "另一种分布的节点"],
    ["nodes_other_y", [Math.cos(-1.2), Math.cos(1.2)], "节点上的函数值"],
    ["n_axis", N(40), "节点数 n"],
    ["max_error_vs_n_selected", Array.from({ length: 40 }, (_, i) => 0.2 * Math.pow(1.35, i)), "最大误差随 n（当前分布）"],
    ["max_error_vs_n_other", Array.from({ length: 40 }, (_, i) => 0.9 * Math.pow(0.7, i)), "最大误差随 n（另一种分布）"],
    ["mark_k", [1], "当前位置 k"],
    ["mark_error", [0.9615384615384616], "当前最大误差"],
  ],
  meta: {
    a: 25, divergence_ratio: 307086726.59996307, envelope_looseness: 26.000000000026002, envelope_max: 25.000000000025004,
    error_floor: 1.1102230246251565e-16, is_chebyshev: false, k: 1, lambda_other_k: 1, lambda_other_n: 2.5393083434232837,
    lambda_ratio_n: 0.028427772932651803, lambda_selected_k: 1, lambda_selected_n: 89.32491297996349,
    max_error_chebyshev_k: 0.9615384615384616, max_error_chebyshev_n: 0.000339877499893837,
    max_error_equispaced_k: 0.9615384615384616, max_error_equispaced_n: 104371.8688873777,
    max_error_other: 0.9615384615384616, max_error_selected: 0.9615384615384616, n: 12, n_error_at_floor: 2,
    n_error_nonfinite: 0, n_exact_zero_error: 2, n_exact_zero_error_raw: 2, n_nodes: 1, overflow: false,
    sup_derivative_over_factorial: 25.000000000025004, t: 0,
  },
  strings: {
    family: "等距节点", other_family: "Chebyshev 节点", function: "Runge 函数 1/(1+a x^2)",
    note: "k=1 个等距节点：最大误差 9.615e-01；等距误差在 n=7 触底 0.247，之后不单调。",
    data_source: "backend", params_json: '{"a":25.0,"distribution":"equispaced","n":12}',
  },
  critical: { n: 12 },
};

const LEB = N(30);
const LEB_DEF: FrameDef = {
  viz: "lebesgue_growth",
  arrays: [
    ["n_axis", LEB, "节点数 n"],
    ["lambda_equispaced", LEB.map((n) => 1.25 * Math.pow(1.3, n - 1)), "Λ_n（等距）实测"],
    ["lambda_chebyshev", LEB.map((n) => 1.25 + 0.3 * Math.log(n)), "Λ_n（Chebyshev）实测"],
    ["theory_equispaced", LEB.map((n) => 1.2 * Math.pow(1.3, n - 1)), "渐近公式 2^(n+1)/(e n ln n)"],
    ["theory_chebyshev", LEB.map((n) => 1.4 + 0.29 * Math.log(n)), "渐近公式 (2/π)ln n + 0.9625"],
    ["log10_lambda_equispaced", LEB.map(() => 1), "log10 Λ_n（等距）"],
    ["log10_lambda_chebyshev", LEB.map(() => 0.1), "log10 Λ_n（Chebyshev）"],
    ["log10_theory_equispaced", LEB.map(() => 1), "log10 渐近（等距）"],
    ["log10_theory_chebyshev", LEB.map(() => 0.1), "log10 渐近（Chebyshev）"],
    ["slope_measured_equispaced", LEB.map((n) => 0.4 + 0.01 * n), "实测 log-log 斜率（等距）"],
    ["slope_measured_chebyshev", LEB.map(() => 0.465), "实测 log-log 斜率（Chebyshev）"],
    ["slope_theory_equispaced", LEB.map((n) => 0.42 + 0.01 * n), "理论 log-log 斜率（等距）"],
    ["slope_theory_chebyshev", LEB.map(() => 0.4624), "理论 log-log 斜率（Chebyshev）"],
    ["error_asymptotic_equispaced", LEB.map(() => 0.147), "|Λ_实测/Λ_渐近 − 1|（等距）"],
    ["error_asymptotic_chebyshev", LEB.map(() => 2.56e-4), "|Λ_实测/Λ_渐近 − 1|（Chebyshev）"],
    ["mark_n", [15], "当前位置 n"],
    ["mark_lambda", [1.25 * Math.pow(1.3, 14), 1.25 + 0.3 * Math.log(15)], "当前位置 Λ"],
  ],
  meta: {
    error_asymptotic_chebyshev_n: 2.56e-4, error_asymptotic_equispaced_n: 0.147, grid: 2001,
    growth_factor_equispaced: 1.3049042475526071, lambda_chebyshev_n: 3.12697, lambda_chebyshev_times_u: 3.47e-16,
    lambda_equispaced_n: 6.601e6, lambda_equispaced_times_u: 7.33e-10, lambda_ratio: 2.11e6, log10_floor: -16,
    n_exact_zero_lambda: 0, n_exact_zero_slope: 0, n_lambda_at_floor: 0, n_mark: 2, n_max: 30, overflow: false,
    slope_gap_chebyshev: 0.0026, slope_gap_equispaced: 0.0825, slope_measured_chebyshev_n: 0.19597,
    slope_measured_equispaced_n: 19.1588, slope_theory_chebyshev_n: 0.20424, slope_theory_equispaced_n: 19.15,
    t: 0, theory_chebyshev_n: 3.12777, theory_equispaced_n: 7.743e6,
  },
  strings: {
    family_bad: "等距节点", family_good: "Chebyshev 节点", note: "n=30：Λ_等距=6.6e6（渐近公式 7.743e6）。",
    data_source: "backend", params_json: '{"grid":2001,"n_max":30}',
  },
  critical: { n_max: 50 },
};

const T_AXIS = linspace(0, 1, 21);
/** Λ(t) 从等距端降到 Chebyshev 端，但**不是严格单调**：全局最小在 t=0.85。 */
const LAMBDA_VS_T = T_AXIS.map((t, i) => (i === 17 ? 2.2156 : 89.32 * (1 - t) * (1 - t) + 2.54));
const EC_DEF: FrameDef = {
  viz: "error_curve",
  arrays: [
    ["x_grid", XS, "x"],
    ["f_values", COS_F, "f(x)"],
    ["p_morph", XS.map((x) => Math.cos(1.5 * x) + 0.05 * Math.sin(9 * x)), "插值多项式（变形参数 t）"],
    ["error", err(0.04, [100, 300, 500]), "|f-p|（已夹地板）"],
    ["error_raw", err(0.04, [100, 300, 500]), "|f-p|（真值，节点上恰为 0）"],
    ["envelope", XS.map(() => 5.442e5), "严格误差包络"],
    ["monic", err(4.8e-3, [100, 300, 500]), "|∏(x-x_i)|（首一乘积，节点上恰为 0）"],
    ["error_equispaced", err(0.06), "|f-p|（等距节点，同一 n）"],
    ["error_chebyshev", err(0.004), "|f-p|（Chebyshev 节点，同一 n）"],
    ["nodes_x", linspace(-1, 1, 13), "当前节点"],
    ["nodes_y", linspace(-1, 1, 13).map((x) => Math.cos(1.5 * x)), "节点值"],
    ["t_axis", T_AXIS, "节点分布变形 t"],
    ["lambda_vs_t", LAMBDA_VS_T, "Λ(t)"],
    ["mark_t", [0, 89.32491297996349], "当前位置 (t, Λ)"],
    ["mark_error", [3.6633758699408276], "当前最大误差"],
  ],
  meta: {
    a: 25, envelope_looseness: 1229750.196251938, error_floor: 1.1102230246251565e-16, is_runge: true,
    lambda_chebyshev: 2.5393083434232837, lambda_equispaced: 89.32491297996349, lambda_morph: 89.32491297996349,
    lambda_ratio: 35.17686743766733, max_envelope: 4505037.195004347, max_error: 3.6633758699408276,
    max_error_chebyshev: 0.08439197787392261, max_error_equispaced: 3.6633758699408276,
    monic_max: 0.004007495931982975, monic_ratio_to_theory: 8.207351668701133, monic_sup_theory: 0.00048828125,
    morph: 0, n: 12, n_error_at_floor: 3, n_error_nonfinite: 0, n_exact_zero_error: 3, n_exact_zero_monic: 3,
    n_monic_at_floor: 3, overflow: false, sup_derivative_over_factorial: 1124152655.789517, t: 0,
    tightness_median: 1.2630870161845074e-6, tightness_points: 796,
  },
  strings: {
    function_label: "Runge 函数 1/(1+a x^2)",
    note: "n=12, t=0.00：Λ=89.32，最大误差 3.663e+00，严格包络 4.505e+06。",
    data_source: "backend", params_json: '{"a":25.0,"function":"runge","n":12}',
  },
  critical: { n: 12 },
};

const BAR = N(30);
/** 系数通道与残差通道在 n=1、2 处**恰好为 0**（真实帧如此，对数轴画不出）。 */
const BAR_COEF = BAR.map((n) => (n <= 2 ? 0 : 2.69e-4 * Math.pow(1.2, n - 30)));
const BAR_RESID = BAR.map((n) => (n <= 2 ? 0 : 1.65e-17));
const BAR_DEF: FrameDef = {
  viz: "barycentric_vs_vandermonde",
  arrays: [
    ["n_axis", BAR, "节点数 n"],
    ["error_barycentric", geo(30, 2.451e-9, 1.1), "函数值误差：重心公式"],
    ["error_vandermonde", geo(30, 9.068e-12, 1.1), "函数值误差：Vandermonde + Horner"],
    ["error_coefficients_vandermonde", BAR_COEF, "系数误差：Vandermonde（对有理数精确解）"],
    ["kappa_vandermonde", BAR.map((n) => Math.pow(3, n) / 3), "κ₁(V) 实测"],
    ["kappa_bound_vandermonde", BAR.map((n) => (Math.pow(3, n) / 3) * 1.11e-16), "最坏情况系数误差尺度 κ₁(V)·u"],
    ["kappa_theory", BAR.map((n) => Math.pow(3, n)), "参考指数 3^n"],
    ["residual_vandermonde", BAR_RESID, "后向残差 ‖Vc−y‖/(‖V‖‖c‖+‖y‖)"],
    ["log10_error_barycentric", BAR.map(() => -9), "log10 函数值误差（重心）"],
    ["log10_error_vandermonde", BAR.map(() => -11), "log10 函数值误差（Vandermonde）"],
    ["log10_error_coefficients_vandermonde", BAR.map(() => -4), "log10 系数误差（Vandermonde）"],
    ["log10_kappa_vandermonde", BAR.map((n) => n * 0.477), "log10 κ₁(V)"],
    ["log10_kappa_theory", BAR.map((n) => n * 0.477), "log10 参考指数曲线"],
    ["slope_kappa_measured", BAR.map((n) => 1.6 * n), "实测 log-log 斜率 d log κ/d log n"],
    ["slope_kappa_theory", BAR.map((n) => 1.585 * n), "参考曲线的 log-log 斜率"],
    ["mark_n", [30], "当前位置 n"],
    ["mark_error", [-9.6, -11, -3.57], "当前位置三条通道的 log10 误差"],
  ],
  meta: {
    coefficient_over_bound: 0.009351753227000336, coefficient_over_value_error: 109735.13765348477,
    error_barycentric_n: 2.451e-9, error_coefficients_n: 2.69e-4, error_vandermonde_n: 9.068e-12,
    is_chebyshev: false, kappa_bound_n: 2.876e-2, kappa_log2_per_n: 1.596, kappa_n: 2.591e14,
    kappa_ratio_to_theory: 1.258, kappa_saturated: false, kappa_theory_base: 3, log10_floor: -18,
    n_coefficient_error_clamped: 0, n_error_at_floor: 0, n_exact_zero_error: 0, n_exact_zero_residual: 2,
    n_mark: 2, n_max: 30, overflow: false, residual_n: 1.6463220609421744e-17,
    slope_kappa_measured_n: 32.88940204325626, slope_kappa_theory_n: 32.405958853387325, t: 0,
  },
  strings: {
    family: "等距节点", function_label: "exp(x)",
    note: "n=30：函数值误差 重心 2.451e-09 / Vandermonde 9.068e-12，而系数误差 2.690e-04。",
    data_source: "backend", params_json: '{"distribution":"equispaced","function":"exp","n_max":30}',
  },
  critical: { n_max: 30 },
};

const DEFS: Record<string, FrameDef> = {
  runge_nodes: RUNGE_DEF, lebesgue_growth: LEB_DEF, error_curve: EC_DEF, barycentric_vs_vandermonde: BAR_DEF,
};
const VIZ: Record<string, SceneModel> = {
  runge_nodes: rungeNodesModel, lebesgue_growth: lebesgueGrowthModel,
  error_curve: errorCurveModel, barycentric_vs_vandermonde: barycentricVsVandermondeModel,
};
const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };
const VIZ_IDS = Object.keys(DEFS);
/** manifest 的 `requiredAux` 名 → （画在哪条轴，取帧里哪个数组）。 */
const AUX_WHERE: Record<string, [string, string]> = {
  runge_nodes: ["aux", "error_selected"],
  lebesgue_growth: ["dev", "error_asymptotic_equispaced"],
  error_curve: ["aux", "error"],
  barycentric_vs_vandermonde: ["aux", "residual_vandermonde"],
};

const sceneOf = async (viz: string): Promise<Scene> => VIZ[viz](await makeFrame(DEFS[viz]), { ...OPTS });
const specOf = (viz: string) => manifest.vizzes.find((v) => v.id === viz) ?? assert.fail(`manifest 缺 ${viz}`);
const textsOf = (s: Scene): string => s.texts.map((t) => t.text).join("\n");
const badgesOf = (s: Scene): string => s.badges.map((b) => `${b.kind}:${b.text}`).join("\n");
const badgeKinds = (s: Scene): string[] => s.badges.map((b) => b.kind);
const axisOf = (s: Scene, id: string): Axis => s.axes.find((a) => a.id === id) ?? assert.fail(`缺少轴 ${id}`);
const layersOn = (s: Scene, id: string): Layer[] => s.layers.filter((l) => l.axes === id);
const drawnY = (layers: readonly Layer[]): number[] =>
  layers.flatMap((l) =>
    l.kind === "line" ? l.series.flatMap((s) => Array.from(s.y, Number)) : l.kind === "scatter" ? Array.from(l.y, Number) : [],
  );

// --------------------------------------------------------------------------- //
// 装配层与四个 viz 的公共契约
// --------------------------------------------------------------------------- //

test("2.1：manifest 的四个 viz 都有 model，映射表没有多余键，未知 id 返回 null", () => {
  assert.deepEqual(Object.keys(models).sort(), manifest.vizzes.map((v) => v.id).sort());
  assert.equal(manifest.chapter, "2.1");
  for (const v of manifest.vizzes) assert.equal(modelFor(v.id), models[v.id]);
  assert.equal(modelFor("no_such_viz"), null);
});

for (const viz of VIZ_IDS) {
  test(`${viz}：默认帧 → validateScene 干净；chapter/viz/t/pitfalls 逐项取自帧`, async () => {
    const frame = await makeFrame(DEFS[viz]);
    const scene = VIZ[viz](frame, { ...OPTS });
    assert.deepEqual(validateScene(scene), []);
    assert.equal(scene.chapter, "2.1");
    assert.equal(scene.viz, viz);
    assert.equal(scene.t, frame.meta["t"]);
    assert.equal(scene.pitfalls, PITFALLS, "规则 7：pitfalls 逐字来自帧");
    assert.ok(scene.dataSource.startsWith("backend"), "规则 5：来源要写清");
  });

  test(`${viz}：autoscaled 与 manifest 一致、自适应轴挂角标、source 角标齐备（规则 4/5）`, async () => {
    const scene = await sceneOf(viz);
    const spec = specOf(viz);
    for (const a of spec.axes) assert.equal(axisOf(scene, a.id).autoscaled, a.autoscaled, `轴 ${a.id}`);
    if (spec.axes.some((a) => a.autoscaled)) assert.ok(badgesOf(scene).includes("自适应"), "自适应必须明说");
    assert.ok(badgeKinds(scene).includes("source"), "规则 5：必须有 source 角标");
  });

  test(`${viz}：requiredAux 的量真的画在轴上（按数值认，不靠 label）`, async () => {
    const frame = await makeFrame(DEFS[viz]);
    const scene = VIZ[viz](frame, { ...OPTS });
    assert.ok(specOf(viz).requiredAux.length > 0, "规则 3：requiredAux 非空");
    const [axis, arr] = AUX_WHERE[viz];
    const want = Array.from(frame.array(arr), Number).filter((v) => v > 0);
    const got = drawnY(layersOn(scene, axis));
    assert.ok(want.length > 0 && want.every((v) => got.includes(v)), `${arr} 的每个有限正值都要出现在 ${axis} 轴上`);
  });

  test(`${viz}：criticalBadge 用帧 critical；临界值落在某条轴的横轴上时要有竖线`, async () => {
    const frame = await makeFrame(DEFS[viz]);
    const scene = VIZ[viz](frame, { ...OPTS });
    const key = Object.keys(frame.critical)[0];
    const value = Number(frame.critical[key]);
    assert.ok(badgeKinds(scene).includes("critical"), "规则 6：必须挂 critical 角标");
    assert.ok(badgesOf(scene).includes(formatNumber(value, 3)), "角标里的临界值要来自帧");
    // 临界值只有在**某条轴的横轴坐标范围**里才画得成竖线：error_curve 的横轴是 x 与 t（n=12 不是坐标），
    // lebesgue 的横轴是 n∈[1,30]（n_max=50 在范围外）——这两种情况只挂角标，不画假的竖线。
    const inRange = scene.axes.some((a) => a.xlim[0] <= value && value <= a.xlim[1]);
    const xs = scene.layers.flatMap((l) =>
      l.kind === "line" && l.series.every((s) => new Set(Array.from(s.x, Number)).size === 1)
        ? l.series.flatMap((s) => [...new Set(Array.from(s.x, Number))])
        : [],
    );
    if (inRange) assert.ok(xs.includes(value), `竖线要落在帧 critical.${key}=${value} 上，实际 ${JSON.stringify(xs)}`);
  });
}

// --------------------------------------------------------------------------- //
// 2.1 专属：log 轴上的 0 / 时间语义 / 非单调
// --------------------------------------------------------------------------- //

test("runge_nodes：恰为 0 的误差点不进对数层，但地板值写进正文（不静默丢点）", async () => {
  const frame = await makeFrame(RUNGE_DEF);
  const scene = rungeNodesModel(frame, { ...OPTS });
  for (const layer of layersOn(scene, "aux")) {
    const ys = drawnY([layer]).filter((v) => Number.isFinite(v));
    assert.ok(ys.length > 0 && ys.every((v) => v > 0), "对数轴上不许出现 0 或负数");
  }
  assert.equal(frame.meta["n_exact_zero_error"], 2);
  assert.ok(textsOf(scene).includes(formatNumber(1.1102230246251565e-16, 3)), "地板值必须来自帧 meta.error_floor");
});

test("error_curve：t 轴语义（xlim=[0,1]）、全局最小点来自帧数组（Λ(t) 不单调）", async () => {
  const scene = await sceneOf("error_curve");
  const lam = axisOf(scene, "lam");
  assert.deepEqual(lam.xlim, [0, 1]);
  assert.equal(lam.autoscaled, true);
  assert.ok(textsOf(scene).includes("0.85"), "全局最小位置来自帧数组，不能被四舍五入掉");
  assert.ok(textsOf(scene).includes("2.2156"));
});

test("runge_nodes：时间轴只选帧；当前节点数取帧 meta.k（不把请求 t 当索引）", async () => {
  const frame = await makeFrame(RUNGE_DEF);
  const scene = rungeNodesModel(frame, { ...OPTS, t: 0 });
  assert.equal(frame.meta["k"], 1);
  assert.equal(frame.meta["t"], 0);
  assert.ok(textsOf(scene).includes("k=1"));
});

test("barycentric：κ 面板是 log-log，loglogfit 带实测/理论斜率与两条标注（规则 2）", async () => {
  const scene = await sceneOf("barycentric_vs_vandermonde");
  const aux = axisOf(scene, "aux");
  assert.equal(aux.xscale, "log");
  assert.equal(aux.yscale, "log");
  const fit = scene.layers.find((l) => l.kind === "loglogfit");
  assert.ok(fit, "斜率必须画成参考线（画在 log-log 轴上）");
  if (fit && fit.kind === "loglogfit") {
    assert.equal(fit.measuredSlope, BAR_DEF.meta["slope_kappa_measured_n"]);
    assert.equal(fit.theorySlope, BAR_DEF.meta["slope_kappa_theory_n"]);
    assert.ok(fit.seriesLabel.length > 0 && fit.theoryLabel.length > 0);
  }
});

test("barycentric / error_curve：恒 0 的系数、残差与首一乘积点被丢掉但在正文里有交代", async () => {
  const bar = barycentricVsVandermondeModel(await makeFrame(BAR_DEF), { ...OPTS });
  for (const id of ["main", "aux"]) {
    if (axisOf(bar, id).yscale !== "log") continue;
    assert.ok(drawnY(layersOn(bar, id)).every((v) => v > 0), `${id} 的对数轴上出现了 0`);
  }
  assert.ok(textsOf(bar).includes("n=3"), "曲线起点（n=1、2 画不出）要写清");
  const ec = errorCurveModel(await makeFrame(EC_DEF), { ...OPTS });
  assert.ok(drawnY(layersOn(ec, "aux")).every((v) => v > 0));
  assert.ok(textsOf(ec).includes("恰好为 0"), "|∏| 在节点上的 0 要说明");
});

// --------------------------------------------------------------------------- //
// 降级路径：meta 缺键 / 值为 null → 少画什么就写明什么
// --------------------------------------------------------------------------- //

test("runge_nodes：meta.error_floor = null → 不画地板线，status 角标点名该字段", async () => {
  const scene = rungeNodesModel(await makeFrame(tweak(RUNGE_DEF, { meta: { error_floor: null } })), { ...OPTS });
  assert.deepEqual(validateScene(scene), []);
  assert.ok(badgesOf(scene).includes("降级"), "缺字段必须挂 status 角标");
  assert.ok(badgesOf(scene).includes("error_floor"), "角标要点名缺的是哪个字段");
  assert.ok(!textsOf(scene).includes("灰虚线=地板"), "没画地板线就不能在正文里说有");
});

test("lebesgue_growth：缺 meta 斜率 → loglogfit 回落到帧数组末值（不回落成常数）", async () => {
  const frame = await makeFrame(tweak(LEB_DEF, { dropMeta: ["slope_measured_equispaced_n", "slope_theory_equispaced_n"] }));
  const scene = lebesgueGrowthModel(frame, { ...OPTS });
  assert.deepEqual(validateScene(scene), []);
  const fit = scene.layers.find((l) => l.kind === "loglogfit");
  if (fit && fit.kind === "loglogfit") {
    assert.equal(fit.measuredSlope, frame.array("slope_measured_equispaced")[29]);
    assert.equal(fit.theorySlope, frame.array("slope_theory_equispaced")[29]);
  }
  assert.ok(badgesOf(scene).includes("降级"));
});

test("缺 strings.family / note → 写明帧未给出（或降级点名），不自己编分布名与结论", async () => {
  const scene = rungeNodesModel(
    await makeFrame(tweak(RUNGE_DEF, { dropStrings: ["family", "other_family", "note"] })),
    { ...OPTS },
  );
  assert.deepEqual(validateScene(scene), []);
  assert.ok(textsOf(scene).includes("帧未给出") || badgesOf(scene).includes("降级"), "缺字符串要明说");
});

test("缺必需数组 / 长度不一致 / pitfalls 过短 → 抛 ModelInputError（不画半张图）", async () => {
  const cases: Array<[string, FrameDef]> = [
    ["runge_nodes", tweak(RUNGE_DEF, { drop: ["error_selected"] })],
    ["error_curve", tweak(EC_DEF, { drop: ["monic"] })],
    ["barycentric_vs_vandermonde", tweak(BAR_DEF, { drop: ["kappa_vandermonde"] })],
    ["lebesgue_growth", tweak(LEB_DEF, { arrays: { n_axis: [1, 2, 3] } })],
    ["runge_nodes", tweak(RUNGE_DEF, { pitfalls: "太短" })],
  ];
  for (const [viz, def] of cases) {
    const frame = await makeFrame(def);
    assert.throws(() => VIZ[viz](frame, { ...OPTS }), ModelInputError, `${viz} 应当抛 ModelInputError`);
  }
});

// --------------------------------------------------------------------------- //
// 纯函数性
// --------------------------------------------------------------------------- //

test("模型是纯函数：同一帧两次调用逐字节相同，且不改帧（无 IO / Date / random）", async () => {
  const frame = await makeFrame(EC_DEF);
  const fingerprint = (f: Frame): string => f.specs.map((s) => `${s.name}:${s.dtype}:${s.shape.join(",")}`).join("|");
  const before = fingerprint(frame);
  assert.equal(sceneToJson(errorCurveModel(frame, { ...OPTS })), sceneToJson(errorCurveModel(frame, { ...OPTS })));
  assert.equal(fingerprint(frame), before);
  const light = errorCurveModel(frame, { ...OPTS });
  const dark = errorCurveModel(frame, { ...OPTS, theme: "dark" });
  assert.equal(dark.layers.length, light.layers.length, "暗色只改配色，不改结构");
  assert.notEqual(dark.canvas.background, light.canvas.background);
});

// --------------------------------------------------------------------------- //
// 真实帧对拍（`data/frames/2.1/**` 不存在时整组跳过——测试不依赖它）
// --------------------------------------------------------------------------- //

/** 测试跑的是 `dist/viz/tests/`（与 `chapter00.test.ts` 同一套相对路径口径）。 */
const REAL_DIR = new URL("../../../data/frames/2.1/", import.meta.url);
const real = existsSync(REAL_DIR)
  ? {}
  : { skip: "没有 data/frames/2.1（先跑 python3 tools/precompute.py --chapter 2.1）" };

function realFrame(viz: string, last = false): Frame {
  const dir = new URL(`${viz}/`, REAL_DIR);
  const files = readdirSync(dir).filter((f) => f.endsWith(".nlf")).sort();
  return decodeNlf(readFileSync(new URL(last ? files[files.length - 1] : files[0], dir)));
}

test("真实帧：夹具与 build_frame 产物的字段集合一致（字段名/单位/标签不漂）", real, async () => {
  for (const viz of VIZ_IDS) {
    const mine = await makeFrame(DEFS[viz]);
    const theirs = realFrame(viz);
    assert.deepEqual(
      mine.specs.map((s) => [s.name, s.dtype, [...s.shape], s.unit, s.label]),
      theirs.specs.map((s) => [s.name, s.dtype, [...s.shape], s.unit, s.label]),
      `${viz}：数组名/dtype/shape/unit/label 逐项一致`,
    );
    for (const key of ["meta", "strings", "critical"] as const) {
      assert.deepEqual(Object.keys(mine[key]).sort(), Object.keys(theirs[key]).sort(), `${viz} ${key} 键`);
    }
  }
});

test("真实帧：首帧与末帧都能产出合法 Scene（时间轴滑到头也不炸）", real, async () => {
  for (const viz of VIZ_IDS) {
    for (const last of [false, true]) {
      const frame = realFrame(viz, last);
      const scene = VIZ[viz](frame, { ...OPTS, t: Number(frame.meta["t"] ?? 0) });
      assert.deepEqual(validateScene(scene), [], `${viz} 的 Scene 不合规`);
    }
  }
});
