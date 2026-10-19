/**
 * 6.1 章（二维综合仿真）model 层测试 —— 自足合成帧。
 *
 * 守的几条：
 * 1. 方腔的残差与 Ghia RMSE 必须同时出现；残差小不是解准。
 * 2. Ghia 误差图的三个序列颜色要分得开（总 / u / v），不能全画成蓝色。
 * 3. 图灵判据：turing 是正波数增长带；stripes 的 λ(0)>0 是均匀失稳；
 *    spots 没有非平凡稳态；uniform 的 v≡0 时色标固定 0–1 并明确标注，
 *    主波长无定义（不是 0）。
 * 4. 浅水误差只画 error_valid 的点；t=0 没有推进步时面板为空并给出降级说明。
 * 5. 960 与 640 两种版式都要通过 validateScene。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import {
  cavityVorticityModel,
  manifest,
  modelFor,
  models,
} from "../chapters/6.1/index.js";

type Row = readonly [name: string, data: ArrayLike<number>, label: string, shape?: readonly number[]];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  pitfalls?: string;
}

const PITFALLS =
  "残差小不等于解准：残差只量离离散方程有多远，中心线 RMSE 才量离 Ghia 基准有多远。" +
  "图灵斑图不是图灵判据：spots 没有非平凡稳态，stripes 的底态均匀失稳；只有底态无扩散稳定且正波数 λ*>0 才是图灵带。" +
  "干底质量漂移来自正性地板注入的水，不是格式不守恒；扇内与全域 L1 必须分开报。" +
  "热图的色标与等值线各说一件事；误差掩码外的占位值不能当成数据点画出来。";

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
        chapter: "6.1",
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

const axisOf = (scene: Scene, id: string): Axis => {
  const hit = scene.axes.find((a) => a.id === id);
  assert.ok(hit, `场景里没有轴 ${id}（有 ${scene.axes.map((a) => a.id).join(",")}）`);
  return hit;
};
const layersOn = (scene: Scene, id: string): Layer[] => scene.layers.filter((l) => l.axes === id);
const layerKinds = (scene: Scene, id: string): string[] => layersOn(scene, id).map((l) => l.kind);
const textsOf = (scene: Scene): string => scene.texts.map((t) => t.text).join(" | ");
const badgesOf = (scene: Scene): string => scene.badges.map((b) => b.text).join(" | ");

function grid(n: number, f: (i: number, j: number) => number): number[] {
  const out: number[] = [];
  for (let j = 0; j < n; j += 1) for (let i = 0; i < n; i += 1) out.push(f(i, j));
  return out;
}

// --------------------------------------------------------------------------- //
// 合成帧
// --------------------------------------------------------------------------- //
const CAVITY: FrameDef = {
  viz: "cavity_vorticity",
  arrays: [
    ["x", [0, 1 / 3, 2 / 3, 1], "x"],
    ["y", [0, 1 / 3, 2 / 3, 1], "y"],
    ["vorticity", grid(4, (i, j) => (i - 1.5) * (j - 1.5) - (i === 0 ? 4 : 0)), "ω"],
    ["psi", grid(4, (i, j) => Math.sin((Math.PI * (i + 1)) / 5) * Math.sin((Math.PI * (j + 1)) / 5)), "ψ"],
    ["arrow_x", [0, 1], "ax"],
    ["arrow_y", [0, 1], "ay"],
    ["arrow_u", [1, 0.2, -0.1, 0.05], "au"],
    ["arrow_v", [-0.2, 0.3, 0.1, -0.05], "av"],
    ["residual", [-1.2, -3.5, -7.0], "log10 residual"],
    ["residual_index", [0, 50, 100], "iter"],
  ],
  meta: {
    re: 1000, n: 4, h: 0.3333, sweeps: 100, poisson: 100,
    residual_final: 1e-7, residual_tol: 1e-7, residual_initial: 0.06, max_divergence: 0,
    rmse: 0.227, rmse_u: 0.183, rmse_v: 0.264, center_valid: 1, center_x: 0.5, center_y: 0.625,
    omega_abs_max: 4, psi_min: -0.07, psi_max: 0.001, center_psi: -0.07,
    dt_cfl: 0.02, dt_diffusion: 0.156, re_cross: 6, dt_binding: 1,
    cell_re: 250, central_threshold_re: 6, converged: 1, finite: 1,
    enstrophy: 1, kinetic_energy: 0.1, overflow: false,
  },
  strings: { status: "ok" },
};

const GHIA: FrameDef = {
  viz: "ghia_benchmark",
  arrays: [
    ["y_ref", [0, 0.5, 1], "y"],
    ["u_ref", [0, 0.6, 1], "u_ref"],
    ["u_num_ref", [0, 0.58, 1], "u_num"],
    ["x_ref", [0, 0.5, 1], "x"],
    ["v_ref", [0, 0.02, 0], "v_ref"],
    ["v_num_ref", [0, 0.01, 0], "v_num"],
    ["y_profile", [0, 0.5, 1], "yp"],
    ["u_profile", [0, 0.58, 1], "up"],
    ["x_profile", [0, 0.5, 1], "xp"],
    ["v_profile", [0, 0.01, 0], "vp"],
    ["n_grid", [17, 25, 33, 49], "n"],
    ["error", [0.02, 0.012, 0.0087, 0.006], "RMSE"],
    ["error_u", [0.017, 0.011, 0.008, 0.0055], "RMSE u"],
    ["error_v", [0.023, 0.012, 0.008, 0.006], "RMSE v"],
  ],
  meta: {
    re: 100, n: 49, rmse: 0.006, rmse_u: 0.0055, rmse_v: 0.006,
    transcription_floor: 0.01, grid_slope: 1.1, grid_slope_u: 0.9, grid_slope_v: 1.2,
    best_n: 49, best_rmse: 0.006, center_dist: 0.017, residual: 1e-7,
    converged: 1, sweeps: 100, overflow: false,
  },
  strings: { status: "ok" },
};

function turingDef(preset: "turing" | "stripes" | "spots" | "uniform"): FrameDef {
  const common = {
    t: 2000, steps: 2000, time_physical: 190.7, n: 4, dx: 0.25,
    du: 8e-5, dv: 8e-6, feed: 0.03, kill: 0.045,
    dt: 0.095, dt_requested: 0.25, dt_limit: 0.095, dt_clamped: 1,
    grid_error_slope: 1.9, finite: 1, overflow: false, spot_count: 10,
  };
  const v = preset === "uniform" ? new Array(16).fill(0) : grid(4, (i, j) => 0.3 * Math.exp(-((i - 1.5) ** 2 + (j - 1.5) ** 2) / 2));
  const base: Record<string, number | boolean | null> = {
    ...common,
    amplitude: preset === "uniform" ? 0 : preset === "spots" ? 0.7 : 0.3,
    wavelength: preset === "uniform" ? 0 : preset === "stripes" ? 1 : 0.125,
    wavelength_valid: preset === "uniform" ? 0 : 1,
    vmax: preset === "uniform" ? 0 : preset === "spots" ? 0.36 : 0.3,
  };
  const variants: Record<string, { meta: Record<string, number | boolean | null>; strings: Record<string, string> }> = {
    turing: {
      meta: { ...base, phi_star: 53.9, lambda_star: 0.015, uniform_rate: -0.0225, turing_band: 1, base_valid: 1, base_u: 0.25, base_v: 0.3 },
      strings: { status: "ok", preset: "turing", turing_note: "λ*>0、φ*>0 且底态稳定：真图灵。", dispersion_note: "det 顶点只对真图灵带有意义。" },
    },
    stripes: {
      meta: { ...base, phi_star: 0, lambda_star: 0.0054, uniform_rate: 0.0054, turing_band: 0, base_valid: 1, base_u: 0.404, base_v: 0.21 },
      strings: { status: "ok", preset: "stripes", turing_note: "均匀模失稳。", dispersion_note: "没有正波数图灵带。" },
    },
    spots: {
      meta: { ...base, phi_star: 0, lambda_star: -0.035, uniform_rate: -0.035, turing_band: 0, base_valid: 0, base_u: 1, base_v: 0 },
      strings: { status: "ok", preset: "spots", turing_note: "没有非平凡稳态。", dispersion_note: "没有正波数图灵带。" },
    },
    uniform: {
      meta: { ...base, phi_star: 0, lambda_star: 0, uniform_rate: 0, turing_band: 0, base_valid: 0, base_u: 1, base_v: 0 },
      strings: { status: "ok", preset: "uniform", turing_note: "平凡态。", dispersion_note: "反应项为零：纯扩散衰减。" },
    },
  };
  const { meta, strings } = variants[preset] as { meta: Record<string, number | boolean | null>; strings: Record<string, string> };
  return {
    viz: "turing_pattern",
    arrays: [
      ["x", [0.125, 0.375, 0.625, 0.875], "x"],
      ["y", [0.125, 0.375, 0.625, 0.875], "y"],
      ["v", v, "v"],
      ["phi", [0, 10, 20, 30, 40, 50, 60, 80], "φ"],
      ["rate", [-0.02, -0.01, 0.005, 0.015, 0.01, 0.002, -0.01, -0.05], "λ"],
      ["n_error", [64, 128, 256, 512], "n"],
      ["error", [0.009, 0.003, 0.0008, 0.0002], "err"],
    ],
    meta,
    strings,
  };
}

const DAM: FrameDef = {
  viz: "dam_break",
  arrays: [
    ["x", [0.1, 0.3, 0.5, 0.7, 0.9], "x"],
    ["h", [1, 1, 0.6, 0.3, 0.01], "h"],
    ["hu", [0, 0, 0.5, 0.4, 0], "hu"],
    ["u", [0, 0, 0.83, 1.33, 0], "u"],
    ["h_exact", [1, 1, 0.62, 0.28, 0.02], "h_exact"],
    ["u_exact", [0, 0, 0.8, 1.4, 0], "u_exact"],
    ["h_initial", [1, 1, 0.5, 0, 0], "h0"],
    ["error", [1e-14, 1e-4, 0.02, 1e-14, 5e-3], "err"],
    ["error_valid", [0, 1, 1, 0, 1], "valid"],
    ["fan", [0, 1, 1, 0, 0], "fan"],
  ],
  meta: {
    t: 0.05, n: 5, dx: 0.2, cfl: 0.8, dt: 0.004, n_steps: 13, max_cfl: 0.8,
    mass: 0.5, mass0: 0.5, mass_drift: 9.9e-9, energy: 2.4, h_min: 1e-8,
    reached: 1, finite: 1, l1_fan: 0.0202, l1_all: 0.0221, l1_ratio: 1.09,
    fan_lo: 0.3, fan_hi: 0.88, overflow: false,
  },
  strings: { status: "ok", mass_note: "地板注水", error_note: "窗口分开报", window_note: "扇区" },
};

async function sceneOf(viz: string, def: FrameDef, opts: Partial<ModelOptions> = {}): Promise<Scene> {
  const scene = models[viz]!(await makeFrame(def), { ...OPTS, ...opts });
  assert.deepEqual(validateScene(scene), [], `场景 ${viz} 未通过 validateScene`);
  return scene;
}

// --------------------------------------------------------------------------- //
// 页面与装配
// --------------------------------------------------------------------------- //
test("6.1 的 manifest 与模型登记表一致，四个 viz 都换成 modelFor", () => {
  assert.deepEqual(manifest.vizzes.map((v) => v.id), ["cavity_vorticity", "ghia_benchmark", "turing_pattern", "dam_break"]);
  for (const id of manifest.vizzes.map((v) => v.id)) {
    assert.ok(models[id], `models 缺 ${id}`);
    assert.equal(modelFor(id), models[id]);
  }
  assert.equal(modelFor("nope"), null);
  for (const viz of manifest.vizzes) {
    assert.ok(viz.requiredAux.length >= 1);
    assert.equal(viz.dataSource, "backend");
  }
});

// --------------------------------------------------------------------------- //
// 方腔
// --------------------------------------------------------------------------- //
test("方腔：热图/等值线/箭头/残差同时存在，残差与 RMSE 不混为一谈", async () => {
  const scene = await sceneOf("cavity_vorticity", CAVITY);
  const kinds = layerKinds(scene, "main");
  assert.ok(kinds.includes("heat") && kinds.includes("contour") && kinds.includes("vectorfield"), `main 图层：${kinds}`);
  assert.ok(layerKinds(scene, "residual").includes("line"), "残差轴要有折线");
  const main = axisOf(scene, "main");
  assert.equal(main.autoscaled, false);
  const residual = axisOf(scene, "residual");
  assert.equal(residual.autoscaled, true);
  const residualLayer = layersOn(scene, "residual").find((l) => l.kind === "line") as Extract<Layer, { kind: "line" }>;
  assert.ok(residualLayer.series.some((s) => s.y.length === 3), "残差数据没进帧");
  const text = textsOf(scene);
  assert.match(text, /RMSE/);
  assert.match(text, /残差小 ≠ 解准/);
  assert.match(badgesOf(scene), /Re=1000/);
  assert.doesNotMatch(text, /NaN|undefined/, "文字里不许出现 NaN/undefined");
});

test("方腔：640 px 版式仍合法，且热图不会为色标挤掉坐标", async () => {
  const scene = await sceneOf("cavity_vorticity", CAVITY, { width: 640 });
  assert.ok(scene.axes.find((a) => a.id === "main")!.rect[3] > 60);
  assert.match(badgesOf(scene), /窄画布/);
});

// --------------------------------------------------------------------------- //
// Ghia
// --------------------------------------------------------------------------- //
test("Ghia：u/v 两个剖面 + 误差图，三条误差曲线颜色不同", async () => {
  const scene = await sceneOf("ghia_benchmark", GHIA);
  axisOf(scene, "main");
  axisOf(scene, "error");
  axisOf(scene, "vprofile");
  const errLayers = layersOn(scene, "error");
  const dots = errLayers.filter((l) => l.kind === "scatter") as Extract<Layer, { kind: "scatter" }>[];
  assert.ok(dots.length >= 3, `误差图至少三条点列，实际 ${dots.length}`);
  const colors = new Set(dots.map((d) => d.color));
  assert.ok(colors.size >= 3, `三条误差曲线的颜色必须分开，实际 ${[...colors].join(",")}`);
  assert.equal(axisOf(scene, "error").yscale, "log");
  const text = textsOf(scene);
  assert.match(text, /抄写地板/);
  assert.match(text, /数值解插到 Ghia 原表节点/);
  assert.match(badgesOf(scene), /RMSE/);
});

test("Ghia：640 px 省略 v(x) 时给出显式降级说明", async () => {
  const scene = await sceneOf("ghia_benchmark", GHIA, { width: 640 });
  assert.ok(scene.axes.some((a) => a.id === "main"));
  assert.ok(!scene.axes.some((a) => a.id === "vprofile"));
  assert.match(badgesOf(scene), /省略 v\(x\)/);
});

// --------------------------------------------------------------------------- //
// 图灵
// --------------------------------------------------------------------------- //
test("图灵：turing 是正波数增长带，stripes 的 λ(0)>0 是均匀失稳，spots 没有底态", async () => {
  const [tu, st, sp] = await Promise.all([sceneOf("turing_pattern", turingDef("turing")), sceneOf("turing_pattern", turingDef("stripes")), sceneOf("turing_pattern", turingDef("spots"))]);
  const heatT = layersOn(tu, "main").find((l) => l.kind === "heat") as Extract<Layer, { kind: "heat" }>;
  assert.equal(heatT.vmax, 0.3);
  assert.match(textsOf(tu), /真图灵带/);
  assert.match(badgesOf(tu), /图灵带/);

  assert.match(textsOf(st), /均匀模失稳/);
  assert.doesNotMatch(textsOf(st), /真图灵带/);
  assert.match(badgesOf(st), /非图灵/);

  assert.match(textsOf(sp), /自复制瞬态/);
  assert.doesNotMatch(textsOf(sp), /真图灵带/);
  for (const s of [tu, st, sp]) assert.ok(layersOn(s, "dispersion").some((l) => l.kind === "line"));
});

test("图灵：uniform 的 v≡0 用固定 0–1 色标；波长无定义不写成 0；无图灵带不显示“理论 ?”", async () => {
  const un = await sceneOf("turing_pattern", turingDef("uniform"));
  const heat = layersOn(un, "main").find((l) => l.kind === "heat") as Extract<Layer, { kind: "heat" }>;
  assert.equal(heat.vmax, 1, "v≡0 的色标固定到 0–1 并明说，而不是 1e-9 的假细节");
  const text = textsOf(un);
  assert.match(text, /v≡0/);
  assert.match(text, /主波长无定义/);
  assert.match(text, /纯扩散/);
  assert.doesNotMatch(text, /波长 0(?!\.)/, "不能把占位 0 写成真实波长 0");

  const st = await sceneOf("turing_pattern", turingDef("stripes"));
  assert.match(textsOf(st), /无图灵理论/);
  assert.doesNotMatch(textsOf(st), /理论 \?/);
});

test("图灵：t=0 与 640 px 版式都合法，窄版把色散图省略写进降级徽标", async () => {
  const narrow = await sceneOf("turing_pattern", turingDef("turing"), { width: 640, t: 0 });
  assert.ok(!narrow.axes.some((a) => a.id === "dispersion"));
  assert.match(badgesOf(narrow), /省略色散关系/);
  assert.ok(layersOn(narrow, "error").length > 0);
});

// --------------------------------------------------------------------------- //
// 干底溃坝
// --------------------------------------------------------------------------- //
test("溃坝：误差散点只包含 error_valid=1 的点，地板占位值不画", async () => {
  const scene = await sceneOf("dam_break", DAM);
  const dots = layersOn(scene, "error").filter((l) => l.kind === "scatter") as Extract<Layer, { kind: "scatter" }>[];
  assert.equal(dots.length, 1);
  const layer = dots[0]!;
  assert.equal(layer.x.length, 3, "5 个点里有 2 个掩码为 0，只该画 3 个");
  assert.ok([...Array(layer.y.length)].every((_, i) => Number(layer.y[i]) > 1e-14));
  const bands = layersOn(scene, "main").filter((l) => l.kind === "band");
  assert.equal(bands.length, 1, "扇区灰带应在主图上");
  assert.match(textsOf(scene), /扇内才是干净估计/);
  assert.match(textsOf(scene), /地板注入水/);
  assert.match(badgesOf(scene), /n=5/);
});

test("溃坝：t=0 没有有效误差 → 不画伪零点，降级徽标说明", async () => {
  const def: FrameDef = {
    ...DAM,
    meta: { ...DAM.meta, t: 0, n_steps: 0, max_cfl: 0, dt: 0, l1_fan: 0, l1_all: 0, l1_ratio: 0, mass_drift: 0, reached: 1 },
    arrays: DAM.arrays.map(([name, data, label, shape]) =>
      name === "error" ? [name, new Array(5).fill(1e-14), label, shape] as const : name === "error_valid" ? [name, new Array(5).fill(0), label, shape] as const : [name, data, label, shape] as const,
    ) as readonly Row[],
  };
  const scene = await sceneOf("dam_break", def, { t: 0 });
  assert.ok(!layersOn(scene, "error").some((l) => l.kind === "scatter"), "t=0 不该画误差点");
  const text = textsOf(scene);
  assert.match(text, /t=0/);
  assert.match(text, /无有效点/);
  assert.match(badgesOf(scene), /降级/);
});

test("溃坝：640 px 仍合法并保留主图与误差图", async () => {
  const scene = await sceneOf("dam_break", DAM, { width: 640 });
  assert.ok(scene.axes.some((a) => a.id === "main"));
  assert.ok(scene.axes.some((a) => a.id === "error"));
  assert.match(badgesOf(scene), /窄画布/);
});

// --------------------------------------------------------------------------- //
// 输入契约
// --------------------------------------------------------------------------- //
test("缺少必需数组时抛 ModelInputError，而不是画半张图", async () => {
  const def: FrameDef = { ...CAVITY, arrays: CAVITY.arrays.filter(([n]) => n !== "vorticity") };
  const frame = await makeFrame(def);
  assert.throws(() => cavityVorticityModel(frame, OPTS), (err: unknown) => err instanceof ModelInputError);
});

test("四页在 960 px 下都有 source/axis/status 徽标", async () => {
  for (const [viz, def] of [
    ["cavity_vorticity", CAVITY],
    ["ghia_benchmark", GHIA],
    ["turing_pattern", turingDef("turing")],
    ["dam_break", DAM],
  ] as const) {
    const scene = await sceneOf(viz, def);
    const kinds = scene.badges.map((b) => b.kind);
    assert.ok(kinds.includes("source"), `${viz} 缺 source 徽标`);
    assert.ok(kinds.includes("autoscale"), `${viz} 缺 axis mode 徽标`);
    assert.ok(kinds.includes("status"), `${viz} 缺 status 徽标`);
  }
});
