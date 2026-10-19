/**
 * 5.4 章（UQ 与全局敏感性）model 层测试 —— 自足合成帧。
 *
 * 守的几条：本章误差数组存的是 log10 值，轴不能再取一次对数；
 * 误差带 band_valid=0 不是“误差很小”；Sobol 的三个“一阶”要分清；
 * PCE 的欠定区与外推没有收敛保证必须显式标出。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import {
  manifest,
  modelFor,
  models,
  outputHistModel,
  pceSpectrumModel,
  pointFillModel,
  sobolBarsModel,
} from "../chapters/5.4/index.js";

type Row = readonly [name: string, data: ArrayLike<number>, label: string, shape?: readonly number[]];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  pitfalls?: string;
}

const PITFALLS =
  "第一，点更均匀不等于积分更准；第二，关掉随机化后误差棒不存在，0 不是误差小；" +
  "第三，sigma/sqrt(N) 只对 MC 是理论；第四，直方图依赖 bin 数，U 形权衡不是越多越好；" +
  "第五，参考密度是 CLT 近似时密度差混着近似偏差；第六，Sobol 的三个一阶指数只在独立输入时相等；" +
  "第七，bootstrap 区间是估计量不确定性，不保证真值；第八，估计器漂移是偏差不是运气；" +
  "第九，训练误差到 1e-16 不说明泛化；第十，PCE 外推没有收敛保证；" +
  "第十一，本章误差数组是 log10 值，轴不能再取对数；第十二，地板值不是精确零。";

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
        chapter: "5.4",
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

interface Override {
  arrays?: Record<string, ArrayLike<number>>;
  meta?: Record<string, number | boolean | null>;
  strings?: Record<string, string>;
  drop?: readonly string[];
  dropMeta?: readonly string[];
}
function tweak(def: FrameDef, ov: Override): FrameDef {
  const arrays = def.arrays
    .filter(([name]) => !(ov.drop ?? []).includes(name))
    .map((row): Row => {
      if (!ov.arrays || !(row[0] in ov.arrays)) return row;
      const data = ov.arrays[row[0]] as ArrayLike<number>;
      const shape = row[3];
      const keepShape = shape !== undefined && shape.reduce((a, b) => a * b, 1) === data.length;
      return [row[0], data, row[2], keepShape ? shape : [data.length]];
    });
  const meta = { ...def.meta };
  for (const key of ov.dropMeta ?? []) delete meta[key];
  return { ...def, arrays, meta: { ...meta, ...ov.meta }, strings: { ...def.strings, ...ov.strings } };
}
async function sceneOf(viz: string, over: Override = {}, opts: Partial<ModelOptions> = {}): Promise<Scene> {
  const scene = models[viz]!(await makeFrame(tweak(DEFS[viz] as FrameDef, over)), { ...OPTS, ...opts });
  assert.deepEqual(validateScene(scene), [], `场景 ${viz} 未通过 validateScene`);
  return scene;
}
const axisOf = (scene: Scene, id: string): Axis => {
  const hit = scene.axes.find((a) => a.id === id);
  assert.ok(hit, `场景里没有轴 ${id}（有 ${scene.axes.map((a) => a.id).join(",")}）`);
  return hit;
};
const layersOn = (scene: Scene, id: string): Layer[] => scene.layers.filter((l) => l.axes === id);
const linesOn = (scene: Scene, id: string) =>
  layersOn(scene, id).flatMap((l) => (l.kind === "line" ? l.series.map((s) => ({ x: s.x, y: s.y, label: s.label })) : []));
const scatterOn = (scene: Scene, id: string) =>
  layersOn(scene, id).flatMap((l) => (l.kind === "scatter" ? [{ x: l.x, y: l.y, label: l.label }] : []));
const barsOn = (scene: Scene, id: string) =>
  layersOn(scene, id).flatMap((l) => (l.kind === "bars" ? [{ x: l.x, values: l.values, label: l.label, errors: l.errors }] : []));
const textsOf = (scene: Scene): string => scene.texts.map((t) => t.text).join(" | ");
const badgesOf = (scene: Scene): string => scene.badges.map((b) => b.text).join(" | ");

// --------------------------------------------------------------------------- //
const PF_DEF: FrameDef = {
  viz: "point_fill",
  arrays: [
    ["mc_x", [0.87, 0.2], "mc x"],
    ["mc_y", [0.39, 0.8], "mc y"],
    ["sobol_x", [0.59, 0.1], "sb x"],
    ["sobol_y", [0.97, 0.4], "sb y"],
    ["lattice_x", [0.87, 0.2], "la x"],
    ["lattice_y", [0.39, 0.8], "la y"],
    ["n_axis", [64, 128, 256], "N"],
    ["error", [-3.47, -3.99, -4.75], "sel"],
    ["error_valid", [1, 1, 1], "sel mask"],
    ["error_mc", [-1.83, -1.91, -2.09], "mc"],
    ["error_sobol", [-3.47, -3.99, -4.75], "sb"],
    ["error_lattice", [-2.51, -2.92, -3.25], "la"],
    ["error_grid", [-3.36, -3.63, -3.96], "grid"],
    ["error_ref", [-1.83, -1.98, -2.13], "ref"],
    ["band_lo", [-3.56, -4.09, -4.85], "lo"],
    ["band_hi", [-3.39, -3.91, -4.67], "hi"],
    ["band_valid", [1, 1, 1], "mask"],
  ],
  meta: {
    t: 128,
    n_shown: 2,
    n_max: 2048,
    n_curve: 3,
    replicates: 24,
    band_available: true,
    slope_current: -1.19,
    slope_mc: -0.5,
    slope_sobol: -1.19,
    slope_lattice: -0.73,
    slope_grid: -1.0,
    slope_reference: -0.5,
    error_last: 3.5e-6,
    error_mc_last: 1.7e-3,
    error_ratio_vs_mc: 0.002,
    sigma_measured: 0.116,
    overflow: false,
  },
  strings: { method: "sobol", method_name: "Sobol' 数字网", integrand: "smooth", integrand_label: "光滑（非张量积）" },
};

const OH_DEF: FrameDef = {
  viz: "output_hist",
  arrays: [
    ["bin_center", [-0.5, -0.1, 0.3, 0.7], "c"],
    ["density", [0.1, 0.35, 0.3, 0.1], "h"],
    ["density_ref", [0.08, 0.3, 0.32, 0.12], "ref"],
    ["error", [-1.2, -0.8, -1.0, -1.5], "err"],
    ["error_valid", [1, 1, 1, 0], "mask"],
    ["qq_ref", [-1.5, 0, 1.5], "qqr"],
    ["qq_emp", [-1.4, 0.1, 1.8], "qqe"],
    ["bins_curve", [8, 12, 16], "bins"],
    ["bins_error", [-1.29, -1.20, -1.28], "bins err"],
    ["counts", [2, 20, 18, 3], "counts"],
  ],
  meta: {
    t: 0,
    n: 20000,
    n_bins: 4,
    bin_width: 0.4,
    mean: 1.0,
    std: 1.02,
    skewness: 0.43,
    excess_kurtosis: -0.56,
    ks_to_normal: 0.059,
    ks_to_ref: 0.058,
    within_1p96_sigma: 0.962,
    ref_is_exact: false,
    spread: 1,
    overflow: false,
  },
  strings: { status: "ok", ref_kind: "normal" },
};

const SB_DEF: FrameDef = {
  viz: "sobol_bars",
  arrays: [
    ["index_x", [0, 1, 2], "i"],
    ["first", [0.69, 0.15, 0.008], "S"],
    ["total", [0.83, 0.25, 0.036], "ST"],
    ["error_first", [0.11, 0.07, 0.03], "e1"],
    ["error_total", [0.07, 0.03, 0.004], "e2"],
    ["first_other", [0.25, 0.83, 1.08], "other"],
    ["total_other", [0.27, 0.81, 0.99], "otherT"],
    ["first_indep", [0, 0, 0], "indep"],
    ["between_cond", [0, 0, 0], "cond"],
    ["corr_valid", [0, 0, 0], "corr mask"],
    ["drift_n", [64, 128, 256], "N"],
    ["drift_sum_first", [1.16, 0.84, 1.17], "d1"],
    ["drift_sum_total", [0.99, 0.9, 0.99], "d2"],
    ["drift_analytic_first", [0.929, 0.929, 0.929], "a1"],
    ["drift_analytic_total", [1.072, 1.072, 1.072], "a2"],
  ],
  meta: {
    t: 0,
    dim: 3,
    n: 1024,
    rho: 0,
    n_evals: 5120,
    sum_first: 0.857,
    sum_total: 1.115,
    sum_first_halfwidth: 0.139,
    sum_total_halfwidth: 0.073,
    interaction_share: 0.259,
    lower_bound_ok: true,
    upper_bound_ok: true,
    analytic_sum_first: 0.929,
    analytic_sum_total: 1.072,
    drift_span_first: 0.324,
    drift_span_total: 0.181,
    corr_applicable: false,
    corr_active: false,
    overlap_excess: null,
    rho_measured: null,
    sum_between: null,
    sum_indep: null,
    overflow: false,
  },
  strings: { model: "g_function", model_label: "g-函数", estimator: "Saltelli 2010" },
};

const PC_DEF: FrameDef = {
  viz: "pce_spectrum",
  arrays: [
    ["degree", [0, 1, 2], "k"],
    ["coeff_rms", [-0.8, -1.5, -2.2], "coef"],
    ["coeff_valid", [1, 1, 1], "mask"],
    ["coeff_count", [1, 1, 1], "count"],
    ["order_curve", [1, 2, 3], "p"],
    ["error", [-0.6, -1.2, -2.0], "out"],
    ["error_valid", [1, 1, 1], "mask"],
    ["error_train", [-0.7, -1.3, -2.0], "train"],
    ["n_basis_curve", [2, 3, 4], "nb"],
    ["under_flag", [0, 0, 1], "under"],
    ["extrap_x", [1, 1.5, 2], "mult"],
    ["error_extrap", [-4.7, -4.1, -3.8], "ex"],
    ["decay_ref", [-0.8, -1.6, -2.4], "ref"],
  ],
  meta: {
    t: 0,
    order: 2,
    n_basis: 3,
    n_train: 3,
    underdetermined: true,
    error_train_current: 1.7e-5,
    error_outsample_current: 2.1e-5,
    error_extrap_current: 0.0145,
    extrap_ratio: 706,
    extrap_factor: 2,
    decay_slope: -1.64,
    residual_train_l2: 3.1e-5,
    exact_at_order: null,
    min_outsample: 3.2e-12,
    overfit_factor: 7e10,
    n_underdetermined: 1,
    overflow: false,
  },
  strings: { status: "ok", target: "exp", basis: "legendre", fit: "regression" },
};

const DEFS: Record<string, FrameDef> = {
  point_fill: PF_DEF,
  output_hist: OH_DEF,
  sobol_bars: SB_DEF,
  pce_spectrum: PC_DEF,
};

// --------------------------------------------------------------------------- //
test("5.4：manifest 与模型表一致", () => {
  assert.equal(manifest.chapter, "5.4");
  assert.deepEqual(manifest.vizzes.map((v) => v.id).sort(), Object.keys(models).sort());
  assert.equal(modelFor("nope"), null);
  assert.equal(modelFor("point_fill"), pointFillModel);
  assert.equal(modelFor("output_hist"), outputHistModel);
  assert.equal(modelFor("sobol_bars"), sobolBarsModel);
  assert.equal(modelFor("pce_spectrum"), pceSpectrumModel);
});

test("5.4：误差类坐标轴是线性轴（数组本身已是 log10）", async () => {
  const pf = await sceneOf("point_fill");
  assert.equal(axisOf(pf, "aux").yscale, "linear");
  assert.equal(axisOf(pf, "aux").xscale, "log");
  const pce = await sceneOf("pce_spectrum");
  assert.equal(axisOf(pce, "main").yscale, "linear");
  assert.equal(axisOf(pce, "aux").yscale, "linear");
});

// point_fill
test("point_fill：三种点集、误差带与参考线", async () => {
  const scene = await sceneOf("point_fill");
  const mainScatter = scatterOn(scene, "main");
  for (const name of ["MC", "Sobol'", "秩一格点"]) assert.ok(mainScatter.some((s) => s.label === name));
  assert.ok(layersOn(scene, "aux").some((l) => l.kind === "band"));
  assert.equal(linesOn(scene, "aux").length, 5);
  assert.ok(linesOn(scene, "aux").every((s) => s.label === undefined), "曲线标签改为侧栏图例，避免右端堆叠");
  assert.match(textsOf(scene), /蓝=MC，橙=Sobol'，绿=格点，紫=张量格点/);
  assert.match(textsOf(scene), /灰虚=σ\/√N；只对 MC 是理论/);
  assert.equal(axisOf(scene, "main").autoscaled, false);
});

test("point_fill：band_valid=0 时不画误差带，并明说 0≠误差小", async () => {
  const scene = await sceneOf("point_fill", {
    arrays: { band_valid: [0, 0, 0] },
    meta: { band_available: false },
    strings: { note: "" },
  });
  assert.equal(layersOn(scene, "aux").some((l) => l.kind === "band"), false);
  assert.match(textsOf(scene), /band_valid=0/);
});

// output_hist
test("output_hist：直方图柱 + 参考密度，QQ 轴单独存在", async () => {
  const scene = await sceneOf("output_hist");
  assert.equal(barsOn(scene, "main").length, 1);
  const mainLines = linesOn(scene, "main");
  assert.ok(mainLines.length >= 1, "参考密度线要画出来");
  assert.ok(mainLines.every((s) => s.label === undefined), "参考密度不在柱子底部画行内标签");
  assert.match(textsOf(scene), /蓝=直方图，红=参考/);
  assert.match(textsOf(scene), /紫=密度差，灰虚=地板/);
  assert.ok(axisOf(scene, "qq"));
  assert.deepEqual(scene.axes.find((a) => a.id === "qq")?.ylim?.length, 2);
});

test("output_hist：CLT 参考必须标成近似，不能当真值", async () => {
  const scene = await sceneOf("output_hist");
  assert.match(textsOf(scene), /CLT 近似，不是真值/);
  assert.match(badgesOf(scene), /CLT 参考/);
});

test("output_hist：bin 数权衡的最优点要报出来", async () => {
  const scene = await sceneOf("output_hist");
  assert.match(textsOf(scene), /实测最小 RMS 在 bin=8/);
  assert.match(textsOf(scene), /不是“越多越好”/);
});

// sobol_bars
test("sobol_bars：分组柱、bootstrap 须线与上界参考", async () => {
  const scene = await sceneOf("sobol_bars");
  const bars = barsOn(scene, "main");
  assert.equal(bars.length, 2);
  assert.ok(bars.every((b) => b.errors !== undefined));
  assert.ok(linesOn(scene, "main").some((s) => s.label?.includes("上界 1")));
  const xs = bars.map((b) => Array.from(b.x));
  assert.notDeepEqual(xs[0], xs[1], "一阶与总效应必须错开，不能叠在一根柱上");
});

test("sobol_bars：模型不支持相关输入时必须明说 rho 无效", async () => {
  const scene = await sceneOf("sobol_bars");
  assert.match(textsOf(scene), /ρ 无效/);
  assert.equal(scatterOn(scene, "main").some((s) => s.label?.includes("条件方差")), false);
});

test("sobol_bars：漂移副轴同时画出实测与解析", async () => {
  const scene = await sceneOf("sobol_bars");
  const labels = linesOn(scene, "aux").map((s) => s.label ?? "");
  assert.ok(labels.some((l) => l.includes("ΣS_i(N)")));
  assert.ok(labels.some((l) => l.includes("解析 ΣS_i")));
  assert.equal(axisOf(scene, "aux").xscale, "log");
});

test("sobol_bars：上界违约 → 临界值角标", async () => {
  const scene = await sceneOf("sobol_bars", { meta: { lower_bound_ok: false } });
  assert.match(badgesOf(scene), /上界违约/);
});

// pce_spectrum
test("pce_spectrum：主图系数、衰减参考线、误差与欠定区", async () => {
  const scene = await sceneOf("pce_spectrum");
  assert.match(textsOf(scene), /橙虚=衰减参考/);
  assert.equal(linesOn(scene, "aux").filter((s) => s.x.length > 2).length, 2);
  assert.match(textsOf(scene), /蓝=训练，红=留出/);
  assert.ok(linesOn(scene, "aux").some((s) => s.x.length === 2 && s.x[0] === s.x[1]));
  assert.ok(linesOn(scene, "aux").every((s) => s.label === undefined), "误差曲线不在右端画标签");
  assert.ok(scatterOn(scene, "aux").some((s) => s.label === "欠定点"));
});

test("pce_spectrum：外推轴与放大倍率必须同屏", async () => {
  const scene = await sceneOf("pce_spectrum");
  assert.ok(axisOf(scene, "extrap"));
  assert.match(textsOf(scene), /外推（×2）误差/);
  assert.match(textsOf(scene), /放大 706/);
  assert.match(badgesOf(scene), /欠定/);
});

test("pce_spectrum：order 标记与当前留出误差对齐", async () => {
  const scene = await sceneOf("pce_spectrum");
  const mark = scatterOn(scene, "aux").find((s) => s.label?.includes("当前 order"));
  assert.ok(mark);
  assert.deepEqual(Array.from(mark?.x ?? []), [2]);
  assert.deepEqual(Array.from(mark?.y ?? []), [-1.2]);
});

test("5.4：窄画布四图都不抛错", async () => {
  for (const viz of Object.keys(models)) {
    const scene = await sceneOf(viz, {}, { width: 640, height: 520 });
    assert.ok(scene.axes.length >= 2);
  }
});
