/**
 * 5.2 章（反问题与正则化）model 层测试 —— 自足合成帧，字段与 `server/chapters/5.2.py` 同构。
 *
 * 守的几条：
 * - L 曲线的四个判据只在 valid=1 时画；偏差原则“没交点”不能被当成 λ=0；
 * - 同自由度比较必须用 dof 作横轴；TV 的探针曲线换了 b 就会变，不能当谱滤波；
 * - 四联热图是四个不同的量；直接法的 k 不进入重建，必须明说；
 * - 方差斜率副轴画的是方差、不是 MSE；0 段没出现就说没出现；
 * - 越窗与缺字段都要显式降级/溢出，不静默。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import {
  biasVarianceModel,
  filterShapesModel,
  imageDeconvolutionModel,
  lcurveTradeoffModel,
  manifest,
  modelFor,
  models,
} from "../chapters/5.2/index.js";

type Row = readonly [name: string, data: ArrayLike<number>, label: string, shape?: readonly number[]];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  pitfalls?: string;
}

const PITFALLS =
  "第一，oracle 用了真解，只作基准，不是可选判据；第二，L 曲线拐点不是良定义的，三个判据可能给出不同 λ；" +
  "第三，偏差原则没有交点时写 null/valid=0，不是 λ=0；第四，Picard 条件要分信号段与噪声段读，" +
  "噪声接管后整段必然变平；第五，滤波因子形状相同不等于精度相同，比较必须在同一有效自由度；" +
  "第六，TV 不对应谱滤波，换一个 b 同一个读数就会变；第七，ω_rel=2 是 Landweber 的发散线，" +
  "出界因子被夹值显示，掩码为 0 的点不是数据；第八，四联热图分别是真解、数据、重建、误差，不是四种滤镜；" +
  "第九，直接法的 k 不进入重建，图像不变只有游标在动；第十，半收敛时残差最小不等于解最准；" +
  "第十一，方差斜率副轴不能拿 MSE 的斜率冒充；第十二，地板值不是精确零，越窗必须点名。";

const OPTS: ModelOptions = { width: 960, height: 600, t: 0, params: {} };

async function makeFrame(def: FrameDef): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const [name, data, label, shape] of def.arrays) {
    const typed = Float64Array.from(data as ArrayLike<number>);
    specs.push({
      name,
      dtype: "f8",
      shape: (shape ?? [data.length]) as number[],
      order: "C",
      offset: 0,
      nbytes: typed.byteLength,
      unit: "",
      label,
    });
    arrays.set(name, typed);
  }
  return decodeNlf(
    await encodeNlf(
      new Frame({
        chapter: "5.2",
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
  const meta: Record<string, number | boolean | null> = { ...def.meta };
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
const lineSeriesOn = (scene: Scene, id: string): Array<{ x: ArrayLike<number>; y: ArrayLike<number>; color?: string; label?: string }> =>
  layersOn(scene, id).flatMap((l) =>
    l.kind === "line" ? l.series.map((s) => ({ x: s.x, y: s.y, color: s.color, label: s.label })) : [],
  );
const scatterOn = (scene: Scene, id: string): Array<{ x: ArrayLike<number>; y: ArrayLike<number>; label?: string }> =>
  layersOn(scene, id).flatMap((l) => (l.kind === "scatter" ? [{ x: l.x, y: l.y, label: l.label }] : []));
const textsOf = (scene: Scene): string => scene.texts.map((t) => t.text).join(" | ");
const badgesOf = (scene: Scene): string => scene.badges.map((b) => b.text).join(" | ");
const degradedText = (scene: Scene): string =>
  scene.badges.filter((b) => b.text.includes("降级")).map((b) => b.text).join(" | ");

// --------------------------------------------------------------------------- //
// lcurve_tradeoff
// --------------------------------------------------------------------------- //
const LC_DEF: FrameDef = {
  viz: "lcurve_tradeoff",
  arrays: [
    ["log10_lam", [-4, -3, -2, -1, 0], "lg lam"],
    ["log10_x_norm", [0.5, 0.3, 0.1, -0.1, -0.5], "lg ||x||"],
    ["log10_residual", [-4, -3.5, -2.5, -1.5, -0.5], "lg res"],
    ["error_curve", [1.5, 0.8, 0.1, 0.3, 1.0], "err"],
    ["curvature", [1e-17, 2e-2, 5e-2, 3e-2, 1e-2], "curv"],
    ["dof", [4, 3, 2, 1, 0.1], "dof"],
    ["gcv", [1e-2, 5e-3, 1e-3, 2e-3, 3e-3], "gcv"],
    ["criterion_log10_lam", [-2, -3, -2.5, -4.5], "crit lam"],
    ["criterion_error", [0.01, 0.2, 0.05, 0.0], "crit err"],
    ["criterion_valid", [1, 1, 1, 0], "crit valid"],
    ["corner_log10_lam", [-3, -2.6, -3.4], "corners"],
    ["picard_sigma", [1, 0.1, 0.01, 1e-3], "sigma"],
    ["picard_coeff", [1, 0.05, 1e-3, 1e-9], "coeff"],
    ["picard_ratio", [1, 0.5, 0.1, 1e-6], "ratio"],
    ["picard_below_noise", [0, 0, 1, 1], "below"],
  ],
  meta: {
    t: 0,
    kappa_measured: 1e6,
    noise_rel: 1e-3,
    log10_lam_current: -2,
    dof_current: 2,
    error_current: 0.1,
    residual_current: 0.003,
    log10_lam_oracle: -2,
    log10_lam_lcurve: -3,
    log10_lam_gcv: -2.5,
    log10_lam_discrepancy: null,
    error_oracle: 0.01,
    error_lcurve: 0.2,
    error_gcv: 0.05,
    error_discrepancy: null,
    inflation_lcurve: 20,
    inflation_gcv: 5,
    inflation_discrepancy: null,
    discrepancy_found: false,
    oracle_at_grid_edge: false,
    corner_index_spread: 2,
    picard_slope_signal: 1.9,
    picard_slope_all: 0.4,
    floor: 1e-17,
    overflow: false,
  },
  strings: { status: "ok", picard_class: "satisfied", spectrum: "decay" },
};

// --------------------------------------------------------------------------- //
// filter_shapes
// --------------------------------------------------------------------------- //
const FS_DEF: FrameDef = {
  viz: "filter_shapes",
  arrays: [
    ["index", [1, 2, 3, 4, 5], "i"],
    ["log10_sigma", [0, -1, -2, -3, -4], "lg sigma"],
    ["filter_tsvd", [1, 1, 1, 1, 1], "tsvd"],
    ["filter_tikhonov", [1, 0.9, 0.5, 0.1, 1e-3], "tik"],
    ["filter_landweber", [1, 0.8, 0.3, 0.05, 1e-4], "lw"],
    ["filter_landweber_valid", [1, 1, 1, 1, 1], "lw mask"],
    ["filter_tv_probe", [0.96, 0.9, 0.7, 0.4, 0.1], "tv"],
    ["filter_tv_probe_mixed", [0.6, 0.1, -0.04, 0.0, 0.0], "tv mixed"],
    ["dof_tsvd_curve", [1, 2, 3, 4, 5], "dof tsvd"],
    ["error_tsvd_curve", [1, 0.3, 0.1, 0.05, 0.1], "err tsvd"],
    ["dof_tikhonov_curve", [0.5, 1, 2, 4, 5], "dof tik"],
    ["error_tikhonov_curve", [2, 0.4, 0.05, 0.2, 1], "err tik"],
    ["log10_lam_curve", [-3, -2, -1, 0, 1], "lg lam"],
    ["dof_landweber_curve", [0.6, 1, 1.5, 2, 3], "dof lw"],
    ["error_landweber_curve", [1.5, 0.5, 0.2, 0.3, 0.4], "err lw"],
    ["error_matched", [0.05, 0.04, 0.2], "matched"],
  ],
  meta: {
    t: 0,
    kappa_measured: 1e6,
    lam: 0.01,
    k_tsvd: 4,
    n_iter: 5,
    omega_rel: 1,
    omega_divergence_threshold_rel: 2,
    landweber_diverges: false,
    landweber_max_abs_factor: 1,
    dof_tikhonov: 2,
    dof_tsvd: 4,
    dof_landweber: 1.5,
    error_tikhonov: 0.05,
    error_tsvd: 0.05,
    error_landweber: 0.2,
    matched_dof: 4,
    error_matched_tsvd: 0.05,
    error_matched_tikhonov: 0.04,
    error_matched_landweber: 0.2,
    tv_probe_dependence: 0.92,
    tv_probe_max_leakage: 0.04,
    half_power_index_tikhonov: 3,
    half_power_index_landweber: 2,
    floor: 1e-17,
    overflow: false,
  },
  strings: { status: "ok", winner: "tikhonov" },
};

// --------------------------------------------------------------------------- //
// image_deconvolution
// --------------------------------------------------------------------------- //
const N_IMG = 4;
const IMG_TRUE = [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0];
const IMG_DATA = IMG_TRUE.map((v) => v * 0.9 + 0.01);
const IMG_RECON = IMG_TRUE.map((v, i) => v * 0.8 + (i === 5 ? 0.2 : 0));
const IMG_ABS = IMG_RECON.map((v, i) => Math.abs(v - (IMG_TRUE[i] as number)));
const IMG_RES = IMG_RECON.map((v, i) => (IMG_DATA[i] as number) - v);
const ID_DEF: FrameDef = {
  viz: "image_deconvolution",
  arrays: [
    ["image_true", IMG_TRUE, "true", [N_IMG, N_IMG]],
    ["image_data", IMG_DATA, "data", [N_IMG, N_IMG]],
    ["image_recon", IMG_RECON, "recon", [N_IMG, N_IMG]],
    ["image_abs_error", IMG_ABS, "abs", [N_IMG, N_IMG]],
    ["image_residual", IMG_RES, "res", [N_IMG, N_IMG]],
    ["history_k", [0, 1, 2, 3, 4], "k"],
    ["error_history", [1, 0.5, 0.35, 0.5, 1.2], "err"],
    ["residual_history", [1e-17, 1e-17, 1e-17, 1e-17, 1e-17], "res"],
    ["log10_sigma", [0, -0.5, -1, -1.5], "lg sigma"],
    ["log10_lam_sweep", [-4, -3, -2, -1, 0], "lg sweep"],
    ["error_lam_sweep", [1.5, 0.8, 0.3, 0.6, 1.2], "err sweep"],
  ],
  meta: {
    t: 2,
    n_image: N_IMG,
    blur_sigma: 2,
    n_angles: 8,
    kappa: 1e6,
    kappa_1d: 1e3,
    n_modes: 16,
    rank_measured: 16,
    null_space_dim: 0,
    noise_rel: 1e-3,
    noise_abs: 0.01,
    data_norm: 1,
    lam: 0.01,
    log10_lam: -2,
    k_current: 2,
    error_rel_current: 0.35,
    residual_rel_current: 0.001,
    error_rel_best_lambda: 0.3,
    lam_best: 0.01,
    noise_dominated_modes: 5,
    error_rel_at_k_best: 0.32,
    k_best: 2,
    semi_convergence: true,
    floor: 1e-17,
    overflow: false,
  },
  strings: {
    status: "ok",
    method: "landweber",
    operator: "blur",
    history_kind: "iteration",
  },
};

// --------------------------------------------------------------------------- //
// bias_variance
// --------------------------------------------------------------------------- //
const BV_DEF: FrameDef = {
  viz: "bias_variance",
  arrays: [
    ["log10_lam", [-4, -2, 0, 2, 4], "lg lam"],
    ["bias2_analytic", [1e-8, 1e-6, 1e-3, 0.5, 1.6], "bias2"],
    ["variance_analytic", [2.3, 1.0, 0.1, 1e-4, 1e-17], "var"],
    ["mse_analytic", [2.3, 1.0, 0.1, 0.5, 1.6], "mse"],
    ["bias2_mc", [0.04, 0.03, 1e-3, 0.5, 1.6], "bias mc"],
    ["variance_mc", [2.0, 0.9, 0.09, 1e-4, 1e-17], "var mc"],
    ["mse_mc", [2.1, 0.9, 0.09, 0.5, 1.6], "mse mc"],
    ["mse_mc_std", [0.1, 0.05, 0.01, 1e-3, 1e-17], "std"],
    ["error_rel_analytic", [1.5, 1.0, 0.3, 0.7, 1.2], "err"],
    ["error_rel_mc", [1.4, 0.95, 0.3, 0.7, 1.2], "err mc"],
    ["dof", [10, 8, 3, 0.1, 1e-3], "dof"],
    ["local_slope", [-1.5, -1.2, -0.5, 0.2, 0.0], "mse slope"],
    ["local_slope_variance", [-1.49, -2.0, -3.9, -4.0, -3.6], "var slope"],
  ],
  meta: {
    t: 0,
    kappa_measured: 1e4,
    noise_rel: 1e-3,
    n_trials: 64,
    lam_current: 0.01,
    error_current: 0.3,
    dof_current: 3,
    log10_lam_oracle: -0.5,
    log10_lam_opt_total: -0.5,
    log10_lam_opt_mc: -0.5,
    mse_min: 0.09,
    error_min: 0.3,
    error_at_lam_min: 1.5,
    error_at_lam_max: 1.2,
    variance_slope_low: null,
    variance_slope_high: -4,
    bias2_at_lam_max: 1.6,
    variance_at_lam_min: 2.3,
    variance_over_mse_at_lam_min: 1,
    bias2_over_mse_at_lam_max: 1,
    mc_vs_analytic_max_rel: 0.1,
    mc_bias_max_abs_over_mse: 110,
    mc_mse_max_rel_near_min: 0.07,
    floor: 1e-17,
    overflow: false,
  },
  strings: { status: "ok", picard_class: "satisfied" },
};

const DEFS: Record<string, FrameDef> = {
  lcurve_tradeoff: LC_DEF,
  filter_shapes: FS_DEF,
  image_deconvolution: ID_DEF,
  bias_variance: BV_DEF,
};

// --------------------------------------------------------------------------- //
test("5.2：manifest 与模型表一致，四个 viz 都已实现", () => {
  assert.equal(manifest.chapter, "5.2");
  assert.deepEqual(manifest.vizzes.map((v) => v.id).sort(), Object.keys(models).sort());
  assert.equal(modelFor("nope"), null);
  assert.equal(modelFor("lcurve_tradeoff"), lcurveTradeoffModel);
  assert.equal(modelFor("filter_shapes"), filterShapesModel);
  assert.equal(modelFor("image_deconvolution"), imageDeconvolutionModel);
  assert.equal(modelFor("bias_variance"), biasVarianceModel);
});

// --------------------------------------------------------------------------- //
// lcurve_tradeoff
// --------------------------------------------------------------------------- //
test("lcurve：主图坐标是 log10 量、判据只在 valid=1 时出现", async () => {
  const scene = await sceneOf("lcurve_tradeoff");
  assert.equal(axisOf(scene, "lcurve").autoscaled, true);
  const texts = textsOf(scene);
  assert.match(texts, /偏差原则整条网格没有交点（null ≠ λ=0）/);
  const scatter = scatterOn(scene, "lcurve");
  assert.equal(scatter.filter((s) => s.label?.includes("偏差原则")).length, 0, "valid=0 的判据不能画点");
  assert.ok(scatter.some((s) => s.label === "L 曲线拐点"));
});

test("lcurve：Picard 图存在，噪声接管点单独标出", async () => {
  const scene = await sceneOf("lcurve_tradeoff");
  const picard = axisOf(scene, "picard");
  assert.equal(picard.xscale, "log");
  const below = scatterOn(scene, "picard").find((s) => s.label === "系数已低于噪声");
  assert.ok(below);
  assert.equal(below?.x.length, 2);
});

test("lcurve：缺 corner 数组时降级点名，不画假拐点", async () => {
  const scene = await sceneOf("lcurve_tradeoff", { drop: ["corner_log10_lam"] });
  assert.match(degradedText(scene), /corner_log10_lam/);
  assert.equal(scatterOn(scene, "lcurve").filter((s) => s.label === "拐点判据（3 个）").length, 0);
});

// --------------------------------------------------------------------------- //
// filter_shapes
// --------------------------------------------------------------------------- //
test("filter_shapes：滤波因子轴固定 0–1.15，误差横轴是有效自由度", async () => {
  const scene = await sceneOf("filter_shapes");
  const filters = axisOf(scene, "filters");
  assert.equal(filters.autoscaled, false);
  assert.deepEqual(filters.ylim, [0, 1.15]);
  const err = axisOf(scene, "dof_error");
  assert.equal(err.xscale, "log");
  assert.match(err.xlabel ?? "", /自由度/);
});

test("filter_shapes：同自由度竖线与三个方法的误差点都在", async () => {
  const scene = await sceneOf("filter_shapes");
  const matchedX = 4; // 夹具的 matched_dof
  assert.ok(lineSeriesOn(scene, "dof_error").some((s) => Array.from(s.x).every((v) => v === matchedX)));
  const pts = scatterOn(scene, "dof_error");
  for (const name of ["TSVD", "Tikhonov", "Landweber"]) {
    assert.ok(pts.some((p) => p.label?.startsWith(name)), `缺少 ${name} 的同自由度点`);
  }
});

test("filter_shapes：TV 换 b 后的探针曲线必须与单模态探针同屏", async () => {
  const scene = await sceneOf("filter_shapes");
  const filterLines = lineSeriesOn(scene, "filters");
  assert.ok(filterLines.length >= 5, "TSVD/Tik/LW/TV/混合 b 五条线都要在");
  // 混合 b 的负读数不进固定窗（不是 0，也不许连成跨窗直线）。
  assert.ok(filterLines.every((s) => Array.from(s.y).every((v) => v >= 0 && v <= 1.15)));
  const texts = textsOf(scene);
  assert.match(texts, /TV 不是谱滤波/);
  assert.match(texts, /橙=单模态.*绿=混合/);
});

test("filter_shapes：ω_rel 越过 2 时挂临界值角标，并点名出界因子", async () => {
  const scene = await sceneOf("filter_shapes", {
    arrays: { filter_landweber_valid: [1, 1, 0, 0, 0] },
    meta: { omega_rel: 2.5, landweber_diverges: true, landweber_max_abs_factor: 1.9e5 },
  });
  assert.match(badgesOf(scene), /已越界/);
  assert.ok(scatterOn(scene, "filters").some((s) => s.label?.includes("掩码=0")));
});

// --------------------------------------------------------------------------- //
// image_deconvolution
// --------------------------------------------------------------------------- //
test("image_deconvolution：四块热图 + 历史轴 + λ 扫描轴，热图是不同数组", async () => {
  const scene = await sceneOf("image_deconvolution");
  const heatAxes = scene.axes.filter((a) => a.id.startsWith("img_"));
  assert.equal(heatAxes.length, 4);
  assert.deepEqual(
    heatAxes.map((a) => a.title),
    ["真解 x*", "数据 b", "重建 x_k", "|重建−x*|"],
  );
  const heats = scene.layers.filter((l) => l.kind === "heat");
  assert.equal(heats.length, 4);
  assert.equal(new Set(heats.map((l) => (l.kind === "heat" ? l.values : undefined))).size, 4);
  assert.match(axisOf(scene, "history").title, /误差（红）.*残差（蓝）/);
  assert.ok(axisOf(scene, "lambda_sweep"));
});

test("image_deconvolution：迭代法的半收敛与 k* 明说；残差与误差分开", async () => {
  const scene = await sceneOf("image_deconvolution");
  assert.match(textsOf(scene), /半收敛：误差先降后升/);
  const hist = lineSeriesOn(scene, "history");
  assert.ok(hist.some((s) => Array.from(s.y).join(",") === "1,0.5,0.35,0.5,1.2"));
  assert.ok(hist.some((s) => Array.from(s.y).every((v) => v === 1e-17)));
  assert.ok(scatterOn(scene, "history").length >= 2);
});

test("image_deconvolution：CT 的 image_data 按正弦图 na×(na+2) 画，不硬套 n²", async () => {
  const scene = await sceneOf("image_deconvolution", {
    arrays: { image_data: Array.from({ length: 15 }, (_, i) => (i % 5) / 4) },
    meta: { n_angles: 3 },
    strings: { operator: "ct" },
  });
  const dataAxis = axisOf(scene, "img_1");
  assert.equal(dataAxis.title, "正弦图 b");
  assert.deepEqual(dataAxis.xlim, [0, 5]);
  assert.deepEqual(dataAxis.ylim, [0, 3]);
});

test("image_deconvolution：数据长度既不等于 n² 也不是正弦图 → 抛错（不猜形状）", async () => {
  await assert.rejects(
    async () => {
      const frame = await makeFrame(
        tweak(ID_DEF, { arrays: { image_data: Array.from({ length: 12 }, () => 0.5) } }),
      );
      imageDeconvolutionModel(frame, OPTS);
    },
    (err: unknown) => err instanceof Error,
  );
});

test("image_deconvolution：直接法必须说 k 不进入重建", async () => {
  const scene = await sceneOf("image_deconvolution", {
    strings: { method: "tikhonov", history_kind: "truncation_order" },
    meta: { semi_convergence: false },
  });
  assert.match(textsOf(scene), /直接法：k 不进入重建/);
  assert.match(textsOf(scene), /没有检测到半收敛/);
});

test("image_deconvolution：n_image 缺失 → 抛错（不能猜四联图尺寸）", async () => {
  await assert.rejects(
    async () => {
      const frame = await makeFrame(tweak(ID_DEF, { dropMeta: ["n_image"] }));
      imageDeconvolutionModel(frame, OPTS);
    },
    (err: unknown) => err instanceof ModelInputError || err instanceof Error,
  );
});

test("image_deconvolution：迭代法缺 λ 扫描 → 降级但不抛错", async () => {
  const scene = await sceneOf("image_deconvolution", { drop: ["log10_lam_sweep", "error_lam_sweep"] });
  assert.match(degradedText(scene), /λ 扫描/);
  assert.equal(scene.axes.some((a) => a.id === "lambda_sweep"), false);
});

// --------------------------------------------------------------------------- //
// bias_variance
// --------------------------------------------------------------------------- //
test("bias_variance：分解轴对数，斜率副轴同时画出 0 与 -4 参考线", async () => {
  const scene = await sceneOf("bias_variance");
  assert.equal(axisOf(scene, "decomposition").yscale, "log");
  const slope = axisOf(scene, "variance_slope");
  assert.ok(slope.ylim[0] <= -4 && slope.ylim[1] >= 0);
  const series = lineSeriesOn(scene, "variance_slope");
  assert.ok(series.some((s) => Array.from(s.y).every((v) => v === 0)), "缺少理论斜率 0 参考线");
  assert.ok(series.some((s) => Array.from(s.y).every((v) => v === -4)), "缺少理论斜率 -4 参考线");
});

test("bias_variance：副轴用方差斜率而不是 MSE 斜率", async () => {
  const scene = await sceneOf("bias_variance");
  const series = lineSeriesOn(scene, "variance_slope");
  const curve = series.find((s) => Array.from(s.y).join(",") === "-1.49,-2,-3.9,-4,-3.6");
  assert.ok(curve, "副轴没有画 variance_slope 的五个实测点");
  assert.match(textsOf(scene), /副轴画的是方差（不是 MSE）的斜率/);
});

test("bias_variance：缺 local_slope_variance 时降级并退回 MSE 斜率（明说）", async () => {
  const scene = await sceneOf("bias_variance", { drop: ["local_slope_variance"] });
  assert.match(degradedText(scene), /local_slope_variance/);
  assert.match(textsOf(scene), /副轴画的是 MSE/);
  assert.match(textsOf(scene), /不能当方差读/);
});

test("bias_variance：本窗口没有 0 段时必须说没有，不把 -1.5 硬读成 0", async () => {
  const scene = await sceneOf("bias_variance");
  assert.match(textsOf(scene), /本窗口没有出现 0 段/);
  assert.match(textsOf(scene), /没测到（null ≠ 0）/);
});

test("5.2：窄画布四图都不抛错（可降级）", async () => {
  for (const viz of Object.keys(models)) {
    const scene = await sceneOf(viz, {}, { width: 640, height: 520 });
    assert.ok(scene.axes.length >= 2);
  }
});
