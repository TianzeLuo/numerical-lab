/**
 * 5.3 章（伴随方法与灵敏度）model 层测试 —— 自足合成帧。
 *
 * 守的几条：方向重合不是证据（要看相对误差）；影响场的纵轴是反向时间；
 * V 形的两臂斜率不同；正向/伴随轨迹在相平面同屏只是视觉安排。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import {
  forwardAdjointPlaybackModel,
  gradientCheckModel,
  gradientFieldModel,
  influenceFieldModel,
  manifest,
  modelFor,
  models,
} from "../chapters/5.3/index.js";

type Row = readonly [name: string, data: ArrayLike<number>, label: string, shape?: readonly number[]];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  pitfalls?: string;
}

const PITFALLS =
  "第一，箭头方向重合不是证据，逐点相对误差才是；第二，箭头长度是对数压缩的显示量，不是灵敏度大小，" +
  "跨参数比较要用弹性；第三，影响场的纵轴是反向时间，tau=0 是终端；第四，质心不漂移不代表没有扩散，" +
  "扩散体现在宽度；第五，宽度连续律在支持触边界后不再适用；第六，梯度检验 V 形左臂舍入、右臂截断，" +
  "两条臂的斜率不同；第七，最优 eps 落在网格端点时不是真的谷底；第八，伴随代价与参数个数无关，" +
  "有限差分代价线性增长；第九，相平面上的正向与伴随轨迹属于不同空间，同屏只是视觉安排；" +
  "第十，|mu(0)|/|mu(T)| 是压缩率的倒数，不是数值不稳定；第十一，地板值不是精确零；" +
  "第十二，越窗与缺字段必须显式点名。";

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
        chapter: "5.3",
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
const heatOn = (scene: Scene, id: string) => layersOn(scene, id).filter((l) => l.kind === "heat");
const textsOf = (scene: Scene): string => scene.texts.map((t) => t.text).join(" | ");
const badgesOf = (scene: Scene): string => scene.badges.map((b) => b.text).join(" | ");

// --------------------------------------------------------------------------- //
// gradient_field
// --------------------------------------------------------------------------- //
const GF_DEF: FrameDef = {
  viz: "gradient_field",
  arrays: [
    ["objective", [0.1, 0.5, 0.2, 1.0], "J"],
    ["elasticity", [-6.8, -0.9], "elasticity"],
    ["gradient_center", [-2.5, -4.1], "grad"],
    ["grid_p1", [0.6, 1.6], "p1"],
    ["grid_p2", [0.03, 0.4], "p2"],
    ["arrow_p1", [0.6, 1.6, 0.6, 1.6], "ap1"],
    ["arrow_p2", [0.03, 0.03, 0.4, 0.4], "ap2"],
    ["adjoint_dir1", [0.47, -0.32, 0.83, 0.1], "d1"],
    ["adjoint_dir2", [-0.88, -0.95, -0.55, 0.99], "d2"],
    ["fd_dir1", [0.47, -0.32, 0.83, 0.1], "f1"],
    ["fd_dir2", [-0.88, -0.95, -0.55, 0.99], "f2"],
    ["arrow_length", [0.8, 1.0, 0.6, 0.3], "len"],
    ["grad_norm", [8.0, 16.9, 1.8, 0.5], "gnorm"],
    ["rel_error_log10", [-9.4, -10.2, -9.5, -8.8], "lg err"],
    ["rel_error", [4e-10, 6e-11, 3e-10, 1.5e-9], "err"],
  ],
  meta: {
    t: 0,
    n_arrows: 4,
    n_grid_p1: 2,
    n_grid_p2: 2,
    max_rel_error: 4.1e-10,
    median_rel_error: 1.3e-11,
    min_rel_error: 1.7e-12,
    grad_norm_max: 8,
    grad_norm_min: 8,
    objective_min: 0.1,
    objective_max: 1,
    resonance_p1: 1,
    n_grid_solves: 4,
    n_arrow_solves: 4,
    rhs_evals: 100,
    stiffness_center: 1,
    damping_center: 0.08,
    dominant_raw: 1,
    dominant_elasticity: 0,
    unit_roundoff: 1.1e-16,
    overflow: false,
  },
  strings: { status: "ok" },
};

// --------------------------------------------------------------------------- //
// influence_field
// --------------------------------------------------------------------------- //
const IF_DEF: FrameDef = {
  viz: "influence_field",
  arrays: [
    ["influence", [0.1, 0.0, 0.5, 0.2, 1.0, 0.4], "influence", [3, 2]],
    ["position", [0, 1], "x"],
    ["tau", [0.2, 0.1, 0], "tau"],
    ["width_measured", [0.15, 0.1, 0.05], "w"],
    ["width_theory", [0.15, 0.101, 0.05], "wth"],
    ["width_error", [0.0, 0.001, 1e-16], "werr"],
    ["center_error", [0, 1e-16, 0], "cerr"],
    ["boundary_flag", [1, 0, 0], "flag"],
    ["influence_now", [0.05, 0.1], "now"],
    ["semidiscrete_abs_error", [0, 1e-17], "semi"],
  ],
  meta: {
    t: 0,
    tau_current: 0,
    tau_max: 0.2,
    tau_fraction: 0,
    width_current: 0.05,
    width_theory_current: 0.05,
    width_ratio_at_max_tau: 0.0099,
    width_ratio_clean_max: 0.00016,
    var_slope_measured: 0.0999,
    var_slope_theory: 0.1,
    var_slope_ratio: 0.9997,
    center_drift_max: 1.1e-16,
    semidiscrete_rel_error_max: 1.6e-6,
    boundary_rows: 1,
    n_tau_rows: 3,
    alpha: 0.05,
    sigma0: 0.05,
    obs_center: 0.5,
    overflow: false,
  },
  strings: { status: "ok" },
};

// --------------------------------------------------------------------------- //
// gradient_check
// --------------------------------------------------------------------------- //
const GC_DEF: FrameDef = {
  viz: "gradient_check",
  arrays: [
    ["eps", [1e-8, 1e-6, 1e-4], "eps"],
    ["error_central", [1e-4, 4.5e-11, 1e-4], "central"],
    ["error_forward", [1e-6, 3e-8, 1e-4], "forward"],
    ["truncation_reference", [1e-10, 1e-6, 1e-2], "trunc"],
    ["roundoff_reference", [1e-8, 1e-10, 1e-12], "round"],
    ["cost_n_params", [2, 8, 32], "k"],
    ["cost_adjoint_evals", [1153, 1153, 1153], "adj"],
    ["cost_fd_evals", [1536, 6144, 24576], "fd"],
    ["cost_adjoint_solves", [1, 1, 1], "adj s"],
    ["cost_fd_solves", [4, 16, 64], "fd s"],
    ["cost_storage_floats", [2312, 2312, 2312], "store"],
    ["cost_checkpoint_floats", [588, 588, 588], "ck"],
    ["cost_checkpoint_evals", [768, 768, 768], "ck e"],
  ],
  meta: {
    t: 0,
    eps_min: 1e-16,
    eps_max: 0.1,
    n_eps: 3,
    param_index: 0,
    k_scanned: 1,
    adjoint_grad_scanned: -0.346,
    fd_grad_at_best: -0.346,
    eps_opt_central: 1e-6,
    eps_opt_forward: 1e-8,
    eps_opt_central_interior: true,
    eps_opt_forward_interior: true,
    error_best_central: 4.5e-11,
    error_best_forward: 3e-8,
    slope_down_central: -0.97,
    slope_up_central: 2,
    slope_down_forward: -0.96,
    slope_up_forward: 1,
    eps_opt_central_over_u13: 0.21,
    eps_opt_forward_over_sqrtu: 1.27,
    unit_roundoff: 1.11e-16,
    overflow: false,
  },
  strings: { status: "ok", method: "rk4" },
};

// --------------------------------------------------------------------------- //
// forward_adjoint_playback
// --------------------------------------------------------------------------- //
const FA_DEF: FrameDef = {
  viz: "forward_adjoint_playback",
  arrays: [
    ["time", [0, 1, 2], "t"],
    ["forward_y1", [1, 0.9, 0.7], "y1"],
    ["forward_y2", [0, 0.2, 0.3], "y2"],
    ["adjoint_mu1", [-0.04, 0.02, 0.05], "mu1"],
    ["adjoint_mu2", [0.212, 0.215, 0.22], "mu2"],
    ["forward_norm", [1, 0.92, 0.76], "fn"],
    ["adjoint_norm", [0.21, 0.22, 0.23], "an"],
    ["sensitivity_residual", [1e-11, 2.5e-10], "sens"],
    ["sensitivity_fd", [-0.038, 0.209], "fd"],
    ["sensitivity_adjoint", [-0.038, 0.209], "adj"],
    ["continuous_gap", [3e-5, 5e-5], "gap"],
  ],
  meta: {
    t: 0.5,
    play_fraction: 0.5,
    play_index: 1,
    t_play: 1,
    tau_play: 1,
    n_steps: 2,
    horizon: 2,
    norm_y_play: 0.92,
    norm_mu_play: 0.22,
    adjoint_growth_ratio: 0.246,
    norm_mu_min: 0.21,
    norm_mu_max: 0.23,
    sensitivity_residual_max: 2.5e-10,
    grad_stiffness: -2.55,
    grad_damping: -4.18,
    continuous_gap_max: 5e-5,
    n_rhs_forward: 12,
    n_floats_stored: 24,
    overflow: false,
  },
  strings: { status: "ok" },
};

const DEFS: Record<string, FrameDef> = {
  gradient_field: GF_DEF,
  influence_field: IF_DEF,
  gradient_check: GC_DEF,
  forward_adjoint_playback: FA_DEF,
};

// --------------------------------------------------------------------------- //
test("5.3：manifest 与模型表一致", () => {
  assert.equal(manifest.chapter, "5.3");
  assert.deepEqual(manifest.vizzes.map((v) => v.id).sort(), Object.keys(models).sort());
  assert.equal(modelFor("nope"), null);
  assert.equal(modelFor("gradient_field"), gradientFieldModel);
  assert.equal(modelFor("influence_field"), influenceFieldModel);
  assert.equal(modelFor("gradient_check"), gradientCheckModel);
  assert.equal(modelFor("forward_adjoint_playback"), forwardAdjointPlaybackModel);
});

test("gradient_field：等高线+箭头+误差底栏，误差必须给出", async () => {
  const scene = await sceneOf("gradient_field");
  assert.ok(layersOn(scene, "plane").some((l) => l.kind === "contour"));
  assert.ok(layersOn(scene, "plane").some((l) => l.kind === "vectorfield"));
  assert.equal(axisOf(scene, "error").autoscaled, true);
  assert.match(textsOf(scene), /橙虚=及格线 −6/);
  assert.ok(linesOn(scene, "error").every((s) => s.label === undefined), "底栏参考线不在数据区画行内标签");
  assert.match(textsOf(scene), /原始 p2，弹性 p1/, "主导参数是 0 基索引：原始梯度看 p2，弹性看 p1");
  assert.match(textsOf(scene), /看底栏误差/);
});

test("gradient_field：max 相对误差超阈 → 临界值角标", async () => {
  const scene = await sceneOf("gradient_field", { meta: { max_rel_error: 1e-3 } });
  assert.match(badgesOf(scene), /已越界/);
});

test("influence_field：纵轴是反向时间，热图行序已翻转（tau=0 在底）", async () => {
  const scene = await sceneOf("influence_field");
  const field = axisOf(scene, "field");
  assert.deepEqual(field.ylim, [0, 0.2]);
  const heat = heatOn(scene, "field")[0];
  assert.ok(heat && heat.kind === "heat");
  if (heat && heat.kind === "heat") {
    assert.equal(heat.ny, 3);
    assert.equal(heat.values[0], 1.0, "tau=0 的行（原最后一行）应在热图底部");
    assert.equal(heat.values[5], 0.0);
  }
});

test("influence_field：宽度偏差与质心漂移同轴，边界行单独标出", async () => {
  const scene = await sceneOf("influence_field");
  assert.ok(linesOn(scene, "width").filter((s) => s.x.length > 2).length >= 2);
  assert.match(textsOf(scene), /蓝=宽度偏差，紫=质心漂移/);
  assert.match(textsOf(scene), /橙点=支持触界/);
  assert.ok(scatterOn(scene, "width").some((s) => s.label?.includes("边界")));
  assert.match(textsOf(scene), /长 τ 处才开始偏/);
});

test("gradient_check：双对数 V 曲线、两条参考线与最优 eps", async () => {
  const scene = await sceneOf("gradient_check");
  assert.equal(axisOf(scene, "vcurve").xscale, "log");
  assert.equal(axisOf(scene, "vcurve").yscale, "log");
  const labels = linesOn(scene, "vcurve").map((s) => s.label ?? "");
  assert.ok(labels.some((l) => l.includes("中心差分 ε*")));
  assert.match(textsOf(scene), /蓝虚=截断参考/);
});

test("gradient_check：代价轴以参数个数为横轴，伴随平线 vs FD 增长", async () => {
  const scene = await sceneOf("gradient_check");
  const cost = axisOf(scene, "cost");
  assert.equal(cost.xscale, "log");
  const costLines = linesOn(scene, "cost");
  const adj = costLines.find((s) => s.y.length === 3 && Array.from(s.y).every((v) => v === 1153));
  const fd = costLines.find((s) => Array.from(s.y).join(",") === "1536,6144,24576");
  assert.deepEqual(Array.from(adj?.y ?? []), [1153, 1153, 1153]);
  assert.deepEqual(Array.from(fd?.y ?? []), [1536, 6144, 24576]);
  assert.match(textsOf(scene), /红=伴随，橙=FD/);
});

test("forward/adjoint：正向画到播放头、伴随从尾部倒着画回", async () => {
  const scene = await sceneOf("forward_adjoint_playback");
  const phaseLines = linesOn(scene, "phase");
  const fwd = phaseLines.find((s) => s.x[0] === 1);
  const adj = phaseLines.find((s) => s.x[0] === 0.02);
  assert.equal(fwd?.x.length, 2, "play_index=1 → 前 2 点");
  assert.equal(adj?.x.length, 2, "从 index 1 到尾部 2 点");
  assert.ok(phaseLines.every((s) => s.label === undefined), "两条轨迹改用侧栏图例");
  assert.match(textsOf(scene), /同屏不同空间/);
});

test("forward/adjoint：对偶残差必须有合格线，增长比不是不稳定", async () => {
  const scene = await sceneOf("forward_adjoint_playback");
  const resLines = linesOn(scene, "residual");
  assert.equal(resLines.length, 2);
  assert.ok(resLines.some((s) => Array.from(s.y).every((v) => Math.abs(v - 1e-8) < 1e-15)));
  assert.match(textsOf(scene), /橙虚=合格线 1e-8/);
  assert.match(badgesOf(scene), /压缩率的倒数/);
});

test("5.3：窄画布四图都不抛错", async () => {
  for (const viz of Object.keys(models)) {
    const scene = await sceneOf(viz, {}, { width: 640, height: 520 });
    assert.ok(scene.axes.length >= 2);
  }
});
