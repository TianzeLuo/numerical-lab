/**
 * 5.1 章（无约束优化与拟牛顿）model 层测试 —— **自足**：不读 `data/frames`，
 * 用 `Frame` + `encodeNlf`/`decodeNlf` 造与 `server/chapters/5.1.py` 同构的帧。
 *
 * 守的东西：
 * - 轨迹/误差只画到 `meta.k_actual`（把完整历史一次性画出来 = 把动画变成答案）；
 * - q 与 r 的量级差 130 个数量级，必须分轴；q 的无效掩码是“无定义”不是 0；
 * - 鞍点的阈值是实测值，null 必须显示为“测不到”，不能当 0；
 * - 信赖域的 ρ 可以为负，必须走线性轴；k=0 时 ρ 是“还没发生”，不是 0。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import {
  convergenceRatesModel,
  descentRaceModel,
  manifest,
  modelFor,
  models,
  saddleEscapeModel,
  trustRegionRadiusModel,
} from "../chapters/5.1/index.js";

type Row = readonly [name: string, data: ArrayLike<number>, label: string, shape?: readonly number[]];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  pitfalls?: string;
}

const PITFALLS =
  "第一，轨迹数组是完整历史，不按 k 切片就会把动画变成答案；第二，四种方法的步长口径不同，" +
  "拿谁先到当结论之前先看 α/α* 与发散线 1+1/κ；第三，q=e_{k+1}/e_k^2 趋于常数才是二次收敛，" +
  "贴到舍入地板之后 q 会爆炸，那是假超二次；第四，q 与 r 量级差 130 个数量级，同轴读会把 r 压成看不见的线；" +
  "第五，鞍点逃逸只取决于扰动在 u_min 上的投影，阈值 null 是没测到不是 0；第六，信赖域 ρ 可以为负，" +
  "对数轴画不出负值，必须单独用线性轴；第七，被拒绝的步不进轨迹，但仍在 accepted/ratio 历史里；" +
  "第八，k=0 时 ρ 还没有发生，写 0 是把未来值冒充当前读数；第九，等高线是底图不是数据；" +
  "第十，tMax 只覆盖默认步数窗口，增大 max_iter 不会加长动画。";

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
        chapter: "5.1",
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
const contourOn = (scene: Scene, id: string): Layer[] => layersOn(scene, id).filter((l) => l.kind === "contour");
const textsOf = (scene: Scene): string => scene.texts.map((t) => t.text).join(" | ");
const badgesOf = (scene: Scene): string => scene.badges.map((b) => b.text).join(" | ");
const degradedText = (scene: Scene): string =>
  scene.badges.filter((b) => b.text.includes("降级")).map((b) => b.text).join(" | ");

const METHODS = ["gd", "momentum", "bfgs", "newton"] as const;
const ITER6 = [0, 1, 2, 3, 4, 5];
const ITER7 = [0, 1, 2, 3, 4, 5, 6];
const CONTOUR3 = [0, 1, 2, 3, 4, 5, 6, 7, 8];

// --------------------------------------------------------------------------- //
// descent_race
// --------------------------------------------------------------------------- //
const DR_DEF: FrameDef = {
  viz: "descent_race",
  arrays: [
    ["contour_x", [-1, 0, 1], "x_1"],
    ["contour_y", [-1, 0, 1], "x_2"],
    ["contour_z", CONTOUR3, "f(x)"],
    ["iterations", ITER6, "迭代 k"],
    ...METHODS.flatMap((m): Row[] => [
      [`path_x_${m}`, [1, 0.8, 0.6, 0.4, 0.2, 0], `${m} x1`],
      [`path_y_${m}`, [1, 0.7, 0.5, 0.3, 0.1, 0], `${m} x2`],
      [`error_${m}`, [1.4, 1.0, 0.7, 0.5, 0.3, 0.1], `${m} err`],
    ]),
  ],
  meta: {
    t: 2,
    k_actual: 2,
    kappa_measured: 100,
    alpha_used: 0.0198,
    alpha_star: 0.0198,
    alpha_over_stable: 0.99,
    error_floor: 1e-17,
    n_iter_gd: 5,
    n_iter_momentum: 5,
    n_iter_bfgs: 3,
    n_iter_newton: 1,
    diverged_gd: false,
    overflow: false,
  },
  strings: { stop_gd: "max_iter", stop_momentum: "max_iter", stop_bfgs: "grad", stop_newton: "grad" },
};

// --------------------------------------------------------------------------- //
// convergence_rates
// --------------------------------------------------------------------------- //
const CR_DEF: FrameDef = {
  viz: "convergence_rates",
  arrays: [
    ["iterations", ITER7, "迭代 k"],
    ["error_gd", [1.4, 1.38, 1.36, 1.34, 1.32, 1.3, 1.28], "gd"],
    ["error_momentum", [1.4, 1.45, 1.47, 1.3, 0.8, 0.4, 0.2], "momentum"],
    ["error_bfgs", [1.4, 1.2, 0.3, 0.01, 1e-5, 1e-10, 1e-14], "bfgs"],
    ["error_newton", [1.4, 0.21, 1e-3, 1e-6, 1e-12, 1e-16, 1e-17], "newton"],
    ["q_newton", [0.107, 0.021, 9.9e-5, 1e-2, 3e12, 3.35e122], "q"],
    ["r_gd", [0.98, 0.99, 0.989, 0.99, 0.988, 0.99], "r"],
    ["q_newton_valid", [1, 1, 1, 1, 1, 0], "q mask"],
    ["r_gd_valid", [1, 1, 1, 1, 1, 1], "r mask"],
  ],
  meta: {
    t: 0,
    error_floor: 1e-17,
    ratio_floor: 1e-30,
    ratio_display_ceiling: 1e10,
    newton_enter_iter: 1,
    newton_floor_start: 5,
    newton_quadratic_constant: 9.9e-5,
    newton_order_tail: 2.998,
    gd_linear_rate_measured: 0.9895,
    gd_linear_rate_theory: 0.99,
    overflow: false,
  },
  strings: { status: "ok", stop_newton: "grad", stop_gd: "max_iter", stop_bfgs: "grad" },
};

// --------------------------------------------------------------------------- //
// saddle_escape
// --------------------------------------------------------------------------- //
const SE_DEF: FrameDef = {
  viz: "saddle_escape",
  arrays: [
    ["contour_x", [-2, 0, 2], "x1"],
    ["contour_y", [-2, 0, 2], "x2"],
    ["contour_z", [-4, 0, 4, 0, 0, 0, 4, 0, -4], "saddle"],
    ["traj_x", [1e-5, 2e-5, 3e-5], "traj x"],
    ["traj_y", [0, 0, 0], "traj y"],
    ["residual", [5e-7, 4e-7, 3e-7], "|grad|"],
    ["error_to_saddle", [1e-5, 2e-5, 3e-5], "dist"],
    ["stable_x", [0, 0], "stable x"],
    ["stable_y", [1e-3, 0], "stable y"],
  ],
  meta: {
    t: 2,
    k_actual: 2,
    lambda_min: -0.05,
    u_min_x: 1,
    u_min_y: 0,
    direction_overlap: 1,
    delta: 1e-5,
    step: 1,
    traj_final_distance: 3e-5,
    traj_final_residual: 3e-7,
    stable_final_distance: 0,
    threshold_neg_curv: 0.00924,
    threshold_random: 0.0139,
    threshold_ratio: 1.5088,
    escaped_random: true,
    escaped_neg_curv: true,
    escape_steps: 96,
    theory_threshold: 0.00924,
    error_floor: 1e-17,
    overflow: false,
  },
  strings: { status: "ok", direction: "neg_curv" },
};

// --------------------------------------------------------------------------- //
// trust_region_radius
// --------------------------------------------------------------------------- //
const TR_DEF: FrameDef = {
  viz: "trust_region_radius",
  arrays: [
    ["contour_x", [-1, 0, 1], "x"],
    ["contour_y", [0, 1, 2], "y"],
    ["contour_z", [100, 50, 100, 50, 0, 50, 100, 50, 100], "rosenbrock"],
    ["traj_x", [-1.2, -1.0, -0.8, -0.6], "x"],
    ["traj_y", [1, 0.9, 0.7, 0.5], "y"],
    ["iterations", ITER6, "k"],
    ["gradient_residual", [232, 100, 40, 10, 1, 0.1], "|grad|"],
    ["radius_history", [1, 0.5, 0.5, 1, 1, 1], "Delta"],
    ["ratio_history", [1.0, -4.2, 0.3, 0.9, 1.2], "rho"],
    ["ratio_valid", [1, 1, 1, 1, 1], "rho mask"],
    ["accepted", [1, 0, 1, 1, 1], "accepted"],
    ["neg_curv", [0, 0, 0, 0, 0], "neg curvature"],
  ],
  meta: {
    t: 3,
    k_actual: 3,
    t2: 0.75,
    delta0: 1,
    radius_current: 1,
    ratio_current: 0.9,
    accepted_current: true,
    n_iter: 5,
    n_accepted: 4,
    n_rejected: 1,
    radius_min: 0.5,
    radius_max: 1,
    ratio_min: -4.2,
    n_negative_curv: 0,
    final_dist: 1e-8,
    radius_floor: 1e-14,
    error_floor: 1e-17,
    overflow: false,
  },
  strings: { status: "ok", stop: "grad" },
};

const DEFS: Record<string, FrameDef> = {
  descent_race: DR_DEF,
  convergence_rates: CR_DEF,
  saddle_escape: SE_DEF,
  trust_region_radius: TR_DEF,
};

// --------------------------------------------------------------------------- //
// 注册表
// --------------------------------------------------------------------------- //
test("5.1：manifest 与模型表一致，四个 viz 都已实现", () => {
  assert.equal(manifest.chapter, "5.1");
  assert.deepEqual(manifest.vizzes.map((v) => v.id).sort(), Object.keys(models).sort());
  assert.equal(modelFor("nope"), null);
  assert.equal(modelFor("descent_race"), descentRaceModel);
  assert.equal(modelFor("convergence_rates"), convergenceRatesModel);
  assert.equal(modelFor("saddle_escape"), saddleEscapeModel);
  assert.equal(modelFor("trust_region_radius"), trustRegionRadiusModel);
});

test("5.1：时间轴 tMax 覆盖默认步数窗口（不冻结在半程）", () => {
  const byId = new Map(manifest.vizzes.map((v) => [v.id, v]));
  const dr = byId.get("descent_race");
  const se = byId.get("saddle_escape");
  const tr = byId.get("trust_region_radius");
  assert.equal(dr?.time.mode === "scrub" ? dr.time.tMax : undefined, 60);
  assert.equal(se?.time.mode === "scrub" ? se.time.tMax : undefined, 200);
  assert.equal(tr?.time.mode === "scrub" ? tr.time.tMax : undefined, 24);
});

// --------------------------------------------------------------------------- //
// descent_race
// --------------------------------------------------------------------------- //
test("descent_race：主图固定、副轴对数，等高线作底图", async () => {
  const scene = await sceneOf("descent_race");
  const main = axisOf(scene, "main");
  const aux = axisOf(scene, "aux");
  assert.equal(main.autoscaled, false);
  assert.equal(aux.autoscaled, true);
  assert.equal(aux.yscale, "log");
  assert.equal(contourOn(scene, "main").length, 1);
});

test("descent_race：路径与误差只画到 meta.k_actual（前 k+1 点）", async () => {
  const scene = await sceneOf("descent_race");
  const aux = lineSeriesOn(scene, "aux").filter((s) => s.y.length > 2);
  assert.equal(aux.length, 4, "四条误差曲线");
  for (const s of aux) {
    assert.equal(s.y.length, 3, "k=2 → 前 3 点");
    assert.equal(s.x.length, 3);
  }
  const paths = lineSeriesOn(scene, "main");
  assert.equal(paths.length, 4, "四条路径");
  for (const s of paths) {
    assert.equal(s.x.length, 3, "每条路径都只画到第 2 步");
    assert.equal(s.x[0], 1);
  }
});

test("descent_race：k=0 只画起点，不画整条路径", async () => {
  const scene = await sceneOf("descent_race", { meta: { k_actual: 0, t: 0 } });
  assert.equal(lineSeriesOn(scene, "main").length, 0, "没有线段可画");
  const points = scatterOn(scene, "main");
  assert.ok(points.length >= 4, "四个方法的当前点仍在（都重合在起点）");
});

test("descent_race：越界参数触发临界值角标（step_scale > 1+1/κ）", async () => {
  const scene = await sceneOf("descent_race", {}, { params: { step_scale: 1.2 } });
  const badges = badgesOf(scene);
  assert.match(badges, /step_scale/);
  assert.match(badges, /已越界/);
});

test("descent_race：缺 error_floor 时降级但不抛错", async () => {
  const scene = await sceneOf("descent_race", { dropMeta: ["error_floor"] });
  assert.match(degradedText(scene), /error_floor/);
});

test("descent_race：缺路径数组 → ModelInputError", async () => {
  await assert.rejects(
    async () => {
      const frame = await makeFrame(tweak(DR_DEF, { drop: ["path_x_bfgs"] }));
      descentRaceModel(frame, OPTS);
    },
    (err: unknown) => err instanceof ModelInputError,
  );
});

// --------------------------------------------------------------------------- //
// convergence_rates
// --------------------------------------------------------------------------- //
test("convergence_rates：q 走对数轴、r 走线性轴，两者不同轴", async () => {
  const scene = await sceneOf("convergence_rates");
  const ratio = axisOf(scene, "ratio");
  const rate = axisOf(scene, "linear_rate");
  assert.equal(ratio.yscale, "log");
  assert.equal(rate.yscale, "linear");
  assert.deepEqual(rate.ylim, [0.95, 1]);
  assert.ok(rate.rect[0] > ratio.rect[0] + ratio.rect[2], "r 轴在 q 轴右侧且不重叠");
});

test("convergence_rates：q 的无效点不参与连线，单独画成轴底散点", async () => {
  const scene = await sceneOf("convergence_rates");
  const qLines = lineSeriesOn(scene, "ratio").filter((s) => new Set(Array.from(s.x)).size > 2);
  assert.ok(qLines.length >= 1);
  assert.ok(qLines.every((s) => Array.from(s.x).every((x) => x <= 4)), "无效的第 6 点不应被连进来");
  const invalid = scatterOn(scene, "ratio").find((s) => s.label === "q 无定义（掩码=0）");
  assert.ok(invalid, "无效点必须有显式散点");
  assert.deepEqual(Array.from(invalid?.y ?? []), [axisOf(scene, "ratio").ylim[0]]);
});

test("convergence_rates：q=1 参考线与二次窗口竖线都在", async () => {
  const scene = await sceneOf("convergence_rates");
  const lines = lineSeriesOn(scene, "ratio");
  assert.ok(lines.some((s) => Array.from(s.y).every((v) => v === 1)), "缺少 q=1 参考线");
  assert.ok(lines.some((s) => s.x.length === 2 && s.x[0] === 1 && s.x[1] === 1), "缺少进入二次区竖线");
  assert.ok(lines.some((s) => s.x.length === 2 && s.x[0] === 5 && s.x[1] === 5), "缺少地板污染竖线");
});

test("convergence_rates：越窗的 q 由溢出角标点名，不静默裁剪", async () => {
  const scene = await sceneOf("convergence_rates");
  assert.equal(scene.overflow?.flag, true);
  assert.match(badgesOf(scene), /溢出/);
});

test("convergence_rates：缺 ratio_floor 时降级点名", async () => {
  const scene = await sceneOf("convergence_rates", { dropMeta: ["ratio_floor"] });
  assert.match(degradedText(scene), /ratio_floor/);
});

// --------------------------------------------------------------------------- //
// saddle_escape
// --------------------------------------------------------------------------- //
test("saddle_escape：轨迹切片、稳定轨迹完整，辅助轴对数", async () => {
  const scene = await sceneOf("saddle_escape");
  const aux = axisOf(scene, "aux");
  assert.equal(aux.yscale, "log");
  const mainLines = lineSeriesOn(scene, "main");
  const traj = mainLines.find((s) => s.x.length === 3 && s.x[0] === 1e-5);
  assert.ok(traj);
  const stable = mainLines.find((s) => s.x.length === 2 && s.x[0] === 0);
  assert.ok(stable);
  const auxLines = lineSeriesOn(scene, "aux").filter((s) => s.y.length === 3 && new Set(Array.from(s.y)).size > 1);
  assert.equal(auxLines.length, 2, "∇f 与距离两条曲线都在");
});

test("saddle_escape：δ 超过实测阈值时侧栏说「已逃逸」，不是靠随机扰动下结论", async () => {
  const scene = await sceneOf("saddle_escape", {
    arrays: { error_to_saddle: [0.1, 0.5, 1.2] },
    meta: { delta: 0.1, traj_final_distance: 1.2 },
  });
  assert.match(textsOf(scene), /→ 已逃逸/);
  assert.match(textsOf(scene), /1\/\|⟨u_min,d⟩\|/);
});

test("saddle_escape：正交方向没有阈值，显示 null 口径而不是 0", async () => {
  const scene = await sceneOf("saddle_escape", {
    strings: { direction: "orthogonal" },
    meta: { threshold_neg_curv: null, threshold_random: null, threshold_ratio: null, theory_threshold: null },
  });
  assert.match(textsOf(scene), /阈值测不到/);
  assert.match(textsOf(scene), /null≠0/);
});

test("saddle_escape：轨迹越出固定窗口 → 溢出角标", async () => {
  const scene = await sceneOf("saddle_escape", {
    arrays: { traj_x: [1e-5, 2e-5, 5.0], traj_y: [0, 0, 0] },
  });
  assert.equal(scene.overflow?.flag, true);
  assert.match(badgesOf(scene), /溢出/);
});

// --------------------------------------------------------------------------- //
// trust_region_radius
// --------------------------------------------------------------------------- //
test("trust_region_radius：ρ 走线性轴（允许负值），半径/梯度走对数轴", async () => {
  const scene = await sceneOf("trust_region_radius");
  assert.equal(axisOf(scene, "aux").yscale, "log");
  const rate = axisOf(scene, "ratio");
  assert.equal(rate.yscale, "linear");
  assert.ok(rate.ylim[0] < 0, "ρ>=-5 的负值必须在窗口内");
});

test("trust_region_radius：轨迹切到 k、圆画在当前点，接收/拒绝都保留", async () => {
  const scene = await sceneOf("trust_region_radius");
  const traj = lineSeriesOn(scene, "main").find((s) => s.x.length === 4);
  assert.equal(traj?.x.length, 4, "k=3 → 4 个点");
  const circle = lineSeriesOn(scene, "main").find((s) => s.x.length === 73);
  assert.ok(circle, "信赖域圆（73 点闭合折线）");
  const pts = scatterOn(scene, "ratio");
  assert.ok(pts.some((p) => p.label === "接受"));
  assert.ok(pts.some((p) => p.label === "拒绝"));
});

test("trust_region_radius：k=0 时 ρ 是「还没发生」，不是 0", async () => {
  const scene = await sceneOf("trust_region_radius", {
    arrays: { traj_x: [-1.2], traj_y: [1], ratio_history: [-4.2, 0.3, 0.9, 1.2, 1.0] },
    meta: { k_actual: 0, t: 0, ratio_current: null, accepted_current: false },
  });
  assert.match(textsOf(scene), /还没有执行任何一步/);
  assert.doesNotMatch(textsOf(scene), /ρ_当前=0/);
});

test("trust_region_radius：缺 radius_history → ModelInputError", async () => {
  await assert.rejects(
    async () => {
      const frame = await makeFrame(tweak(TR_DEF, { drop: ["radius_history"] }));
      trustRegionRadiusModel(frame, OPTS);
    },
    (err: unknown) => err instanceof ModelInputError,
  );
});

test("5.1：所有可视化在窄画布上不抛布局错误（要么降级要么自适配）", async () => {
  for (const viz of Object.keys(models)) {
    const scene = await sceneOf(viz, {}, { width: 640, height: 520 });
    assert.ok(scene.texts.length >= 0);
  }
});
