/**
 * 2.6 章（小波与多分辨分析）model 层测试 —— **自足**：不依赖 Python，也不依赖
 * `data/frames` 是否存在。夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，
 * 字段集合（数组名 / dtype / shape / unit / label、meta / strings / critical 键）
 * 逐字照抄 `server/chapters/2.6.py` 的真实帧（数值是测试用的合成值）。
 *
 * 本文件守的几条"这张图会不会骗人"的规矩：
 *
 * 1. **树的空位不是 0**：`approx_tree` / `detail_tree` 的补零格在测试里填成 999，
 *    模型一旦把整行都画出来，图层里立刻出现 999；测试逐条检查"没有任何图层含 999"。
 * 2. **第 0 层细节不画**：`detail_tree[0]` 在测试里填 ±777，场景里绝不允许出现它
 *    （它恒为 0，画出来会被读成"这一层被消失了"）。
 * 3. **掩码点不连线**：`vanishing_moments` 的 `interior_valid=0` 点只能以散点出现，
 *    折线不得跨过它们（否则会画出一条不存在的下坡）。
 * 4. **缺字段不静默**：meta / 可选数组缺项 → 出现降级角标并点名缺了什么；
 *    必需数组缺失 → 抛 `ModelInputError`（画半张图比不画更坏）。
 * 5. **临界值/参考线必须真画**：denoise 的当前 λ / 最优 λ / 通用阈值三条竖线，
 *    vanishing 的 u=2^-53 参考线，都是"图上的证据"，不能被省略。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Layer, type Scene } from "../shared/src/scene.js";
import { manifest, modelFor, models } from "../chapters/2.6/index.js";

type Row = readonly [name: string, data: ArrayLike<number>, label: string, shape?: readonly number[]];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical?: Record<string, number>;
  pitfalls?: string;
}

const PITFALLS = "这张图会怎么骗你：夹具里的数字是合成的，但字段集合、掩码口径与降级路径与真实帧完全一致。";
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
        chapter: "2.6",
        viz: def.viz,
        kind: "frame",
        codec: "none",
        meta: def.meta,
        strings: def.strings,
        critical: def.critical ?? {},
        pitfalls: def.pitfalls ?? PITFALLS,
        specs,
        arrays,
      }),
    ),
  );
}

const sceneOf = async (def: FrameDef, opts: Partial<ModelOptions> = {}): Promise<Scene> => {
  const model = models[def.viz]!;
  return model(await makeFrame(def), { ...OPTS, ...opts });
};
const layerArrays = (scene: Scene): number[] => {
  const out: number[] = [];
  for (const l of scene.layers) {
    if (l.kind === "line") for (const s of l.series) out.push(...Array.from(s.y as ArrayLike<number>));
    else if (l.kind === "scatter") out.push(...Array.from(l.y as ArrayLike<number>));
    else if (l.kind === "heat") out.push(...Array.from(l.values as ArrayLike<number>));
    else if (l.kind === "bars") out.push(...Array.from(l.values as ArrayLike<number>));
  }
  return out;
};
const degradedOf = (scene: Scene): string => scene.badges.filter((b) => b.text.includes("降级")).map((b) => b.text).join(" | ");
const lineSeriesOn = (scene: Scene, axis: string): { x: ArrayLike<number>; y: ArrayLike<number> }[] =>
  scene.layers
    .filter((l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === axis)
    .flatMap((l) => l.series.map((s) => ({ x: s.x, y: s.y })));

// --------------------------------------------------------------------------- //
// 夹具
// --------------------------------------------------------------------------- //

const DT_N = 32;
const DT_EXT = 40;
const DT_LEVEL = 2;
const DT_APPROX = Array.from({ length: (DT_LEVEL + 1) * DT_EXT }, (_, i) => {
  // 前 n 个是真实近似，其余是补零占位（测试里填 999，模型必须跳过）。
  const row = Math.floor(i / DT_EXT);
  const col = i % DT_EXT;
  return col < DT_N / 2 ** row ? Math.sin(col * 0.3 + row) : 999;
});
const DT_DETAIL = Array.from({ length: (DT_LEVEL + 1) * DT_EXT }, (_, i) => {
  const row = Math.floor(i / DT_EXT);
  const col = i % DT_EXT;
  if (row === 0) return 777; // 第 0 层细节：恒为 0，模型不许画
  return col < DT_N / 2 ** row ? 0.1 * Math.cos(col * 0.7 + row) : 888;
});

const DT_DEF: FrameDef = {
  viz: "decomp_tree",
  arrays: [
    ["signal", Array.from({ length: DT_N }, (_, i) => Math.sin(i * 0.4)), "原信号"],
    ["approx_tree", DT_APPROX, "各层近似", [DT_LEVEL + 1, DT_EXT]],
    ["detail_tree", DT_DETAIL, "各层细节", [DT_LEVEL + 1, DT_EXT]],
    [
      "coeff_length",
      [DT_N, DT_N / 2, DT_N / 4, 0, DT_N / 2, DT_N / 4],
      "各行真实长度",
    ],
    [
      "recon",
      Array.from({ length: (DT_LEVEL + 1) * DT_N }, (_, i) => {
        const row = Math.floor(i / DT_N);
        return Math.sin((i % DT_N) * 0.4) * (row === 0 ? 0.5 : row === 1 ? 0.8 : 1);
      }),
      "逐层重构",
      [DT_LEVEL + 1, DT_N],
    ],
    ["error", Array.from({ length: DT_N }, () => 0.12), "当前层误差"],
    ["energy_share", [0.5, 0.3, 0.2], "能量占比"],
    ["coeff_count", [8, 16, 16], "系数个数"],
  ],
  meta: {
    n: DT_N,
    level: DT_LEVEL,
    stage: 1,
    n_ext: DT_EXT,
    redundancy: 1.25,
    pr_error: 3e-16,
    mra_error: 1.8e-15,
    stage_error: 0.12,
    rel_stage_error: 0.2,
    boundary_share: 0.03,
    energy_top: 0.5,
    energy_detail_sum: 0.5,
    max_detail: 0.4,
    t: 0.5,
    t_actual: 0.5,
  },
  strings: {
    status: "ok",
    signal: "bumps",
    wavelet: "db2",
    mode: "symmetric",
    boundary_mode: "半样本对称反射",
    nonfinite_policy: "对数轴上的 0 用 1e-18 地板",
    mask_note: "空位是补零对齐用的，不是系数值",
    note: "往返误差都在 u 量级",
    data_source: "backend",
  },
  critical: { level: 4 },
};

const TS_N = 16;
const TS_LEVEL = 2;
const TS_BANDS = TS_LEVEL + 1;
const TS_NF = 4;
const TS_NT = 5;
const TS_DEF: FrameDef = {
  viz: "time_scale_map",
  arrays: [
    ["signal", Array.from({ length: TS_N }, (_, i) => Math.sin(i * 0.5)), "信号"],
    ["scalogram", Array.from({ length: TS_BANDS * TS_N }, (_, i) => Math.sin(i * 0.2)), "SWT", [TS_BANDS, TS_N]],
    ["stft_map", Array.from({ length: TS_NF * TS_NT }, (_, i) => 0.1 + 0.02 * i), "STFT", [TS_NF, TS_NT]],
    ["stft_time", [8, 10, 12, 14, 16], "窗中心"],
    ["stft_freq", [0, 1 / 6, 2 / 6, 0.5], "频率"],
    ["wav_width", [0.1, 0.2], "小波宽度"],
    ["wav_band", [0.5, 0.25], "小波带宽"],
    ["stft_width", [0.125, 0.125], "STFT 宽度"],
    ["stft_band", [0.0625, 0.0625], "STFT 带宽"],
    ["cell_area", [0.05, 0.05], "分辨率元"],
    ["error", Array.from({ length: TS_N }, () => 4.4e-16), "往返误差"],
  ],
  meta: {
    level: TS_LEVEL,
    n: TS_N,
    n_bands: TS_BANDS,
    n_stft_time: TS_NT,
    n_stft_freq: TS_NF,
    stft_window: 8,
    stft_hop: 2,
    pr_error: 4.4e-16,
    swt_energy_ratio: 9.03,
    swt_symbol_mean: 5.01,
    swt_redundancy: 5,
    cell_area_min: 0.05,
    cell_area_max: 0.05,
    t: 0,
    t_actual: 0,
  },
  strings: {
    status: "ok",
    signal: "chirp",
    wavelet: "db4",
    nonfinite_policy: "往返误差取 1e-18 地板",
    note: "能量比不是常数",
    shift_note: "SWT 平移一格系数逐位平移",
    data_source: "backend",
  },
  critical: { level: 5 },
};

const DN_GRID = [0, 0.4, 0.8, 1.2, 1.6];
const DN_DEF: FrameDef = {
  viz: "denoise_threshold",
  arrays: [
    ["clean", Array.from({ length: 20 }, (_, i) => Math.sin(i * 0.4)), "干净"],
    ["noisy", Array.from({ length: 20 }, (_, i) => Math.sin(i * 0.4) + 0.1 * Math.cos(i)), "含噪"],
    ["denoised", Array.from({ length: 20 }, (_, i) => 0.9 * Math.sin(i * 0.4)), "去噪"],
    ["error", Array.from({ length: 20 }, () => 0.3), "残差"],
    ["lambda_grid", DN_GRID, "λ 栅格"],
    ["mse_curve", [0.2, 0.1, 0.05, 0.08, 0.15], "MSE"],
    ["bias2_curve", [0.01, 0.02, 0.03, 0.06, 0.12], "偏差²"],
    ["var_curve", [0.19, 0.08, 0.02, 0.02, 0.03], "方差"],
    ["mse_noisy_line", DN_GRID.map(() => 0.09), "不去噪 MSE"],
    ["universal_line", DN_GRID.map(() => 1.0), "通用阈值"],
  ],
  meta: {
    n: 20,
    level: 2,
    sigma: 0.3,
    sigma_hat: 0.45,
    sigma_hat_bias: 0.5,
    lam: 0.8,
    mse: 0.05,
    mse_curve_here: 0.05,
    lam_grid_here: 0.8,
    lam_grid: 0.8,
    mse_noisy: 0.09,
    mse_opt: 0.05,
    lam_opt: 0.8,
    universal: 1.0,
    universal_over_opt: 1.25,
    mse_at_universal: 0.15,
    bias2_curve_here: 0.03,
    var_curve_here: 0.02,
    snr_in: 8,
    snr_out: 12,
    rel_error: 0.22,
    n_trials: 48,
    t: 0,
    t_actual: 0,
  },
  strings: {
    status: "ok",
    signal: "bumps",
    wavelet: "db4",
    rule: "soft",
    nonfinite_policy: "三条曲线取 1e-18 地板",
    note: "最优阈值 0.8，通用阈值 1.0",
    rule_note: "软阈值：大系数被收缩",
    sigma_note: "MAD 估计",
    data_source: "backend",
  },
  critical: { lam: 0.28 },
};

const VM_DEG = [0, 1, 2, 3, 4];
const VM_DEF: FrameDef = {
  viz: "vanishing_moments",
  arrays: [
    ["signal", Array.from({ length: 16 }, (_, i) => (i / 15) ** 2), "多项式"],
    [
      "detail_tree",
      Array.from({ length: 3 * 20 }, (_, i) => {
        const row = Math.floor(i / 20);
        const col = i % 20;
        if (row === 0) return 555;
        return col < 16 / 2 ** row ? (row === 1 ? 0.01 : 0.001) * Math.sin(col) : 666;
      }),
      "细节系数",
      [3, 20],
    ],
    ["detail_length", [16, 8, 4], "细节长度"],
    ["degree_axis", VM_DEG, "次数 k"],
    ["interior_ratio", [1e-18, 1e-18, 1e-18, 1e-16, 1e-5], "内部比值"],
    ["interior_valid", [0, 0, 0, 1, 1], "内部有效掩码"],
    ["boundary_ratio", [0.6, 0.6, 0.6, 0.6, 0.6], "边界比值"],
    ["u_line", VM_DEG.map(() => 1.11e-16), "u 参考线"],
    ["error", Array.from({ length: 16 }, () => 5.5e-16), "往返误差"],
    ["level_ratio", [1e-17, 1e-16], "逐层内部"],
    ["level_boundary", [0.5, 0.6], "逐层边界"],
  ],
  meta: {
    n: 16,
    degree: 3,
    moments: 4,
    level: 2,
    interior_max: 1e-16,
    boundary_max: 0.6,
    ratio_over_u: 0.9,
    killed: true,
    level1_ratio: 8e-17,
    n_interior: 12,
    n_ratio_floored: 3,
    u: 1.11e-16,
    pr_error: 5.5e-16,
    pollution_level1: 0.96,
    pollution_span1: 6,
    t: 0,
    t_actual: 0,
  },
  strings: {
    status: "ok",
    wavelet: "db4",
    mode: "zero",
    nonfinite_policy: "内部比值可能严格为 0：写 1e-18 地板",
    mask_note: "interior_valid = 0 不是很小，而是实测 0",
    note: "k<N 时内部比值贴 u",
    boundary_note: "边界污染",
    data_source: "backend",
  },
  critical: { degree: 4 },
};

const ALL: Record<string, FrameDef> = {
  decomp_tree: DT_DEF,
  time_scale_map: TS_DEF,
  denoise_threshold: DN_DEF,
  vanishing_moments: VM_DEF,
};

// --------------------------------------------------------------------------- //
// 1. 四个 viz 在时间轴上过校验器
// --------------------------------------------------------------------------- //

for (const [viz, def] of Object.entries(ALL)) {
  for (const t of [0, 0.5, 1]) {
    test(`${viz}: t=${t} 的 Scene 过 validateScene（规则 1/2/8）`, async () => {
      const scene = await sceneOf(def, { t });
      assert.deepEqual(validateScene(scene), []);
      assert.equal(scene.chapter, "2.6");
      assert.ok(scene.pitfalls.length > 0, "pitfalls 面板必须非空（规则 7）");
      assert.ok(scene.layers.length >= 3, "每个 viz 至少 3 个图层");
    });
  }
}

test("manifest 与登记表一致：4 个 viz 都有 model，未知 id 返回 null", () => {
  assert.deepEqual(Object.keys(models).sort(), [
    "decomp_tree",
    "denoise_threshold",
    "time_scale_map",
    "vanishing_moments",
  ]);
  for (const v of manifest.vizzes) assert.ok(modelFor(v.id), `manifest 里的 ${v.id} 没有 model`);
  assert.equal(modelFor("nope"), null);
});

// --------------------------------------------------------------------------- //
// 2. decomp_tree：空位不进图层、第 0 层细节不画、stage 对齐
// --------------------------------------------------------------------------- //

test("decomp_tree：补零占位（999/888）与第 0 层细节（777）绝不出现在任何图层里", async () => {
  const scene = await sceneOf(DT_DEF);
  const values = layerArrays(scene);
  assert.ok(!values.includes(999), "approx_tree 的补零占位 999 被画进了图层——t 的读者会以为那一层系数非零");
  assert.ok(!values.includes(888), "detail_tree 的补零占位 888 被画进了图层");
  assert.ok(!values.includes(777), "detail_tree[0]（第 0 层细节，恒为 0）被画进了图层");
});

test("decomp_tree：stage 由 meta 决定，scene.t 是层号而不是物理时间", async () => {
  const scene = await sceneOf(DT_DEF, { t: 0.99 });
  assert.equal(scene.t, 1, "meta.stage=1 时 scene.t 必须取 1（请求 t 只用于缺 meta 时的回落）");
  const meta = { ...DT_DEF.meta, stage: 0 };
  const stage0 = await sceneOf({ ...DT_DEF, meta }, { t: 0.99 });
  assert.equal(stage0.t, 0, "meta.stage=0 时不许按请求 t 画到第 1 层");
});

test("decomp_tree：缺 coeff_length → ModelInputError（不许猜真实长度）", async () => {
  const def: FrameDef = { ...DT_DEF, arrays: DT_DEF.arrays.filter(([n]) => n !== "coeff_length") };
  await assert.rejects(async () => sceneOf(def), (err) => err instanceof ModelInputError);
});

test("decomp_tree：缺 meta.stage 时降级并说明「按请求 t 取整」", async () => {
  const meta = { ...DT_DEF.meta };
  delete meta.stage;
  const scene = await sceneOf({ ...DT_DEF, meta }, { t: 0.9 });
  assert.match(degradedOf(scene), /stage/);
  assert.match(degradedOf(scene), /请求/);
});

// --------------------------------------------------------------------------- //
// 3. time_scale_map：热力图尺寸、缺坐标数组的降级
// --------------------------------------------------------------------------- //

test("time_scale_map：两张热力图的 nx×ny 与数组长度严格一致", async () => {
  const scene = await sceneOf(TS_DEF);
  const heat = scene.layers.filter((l): l is Extract<Layer, { kind: "heat" }> => l.kind === "heat");
  assert.equal(heat.length, 2);
  for (const h of heat) assert.equal(Array.from(h.values).length, h.nx * h.ny);
  const scale = heat.find((h) => h.axes === "scale")!;
  const stft = heat.find((h) => h.axes === "stft")!;
  assert.equal(scale.nx, TS_N);
  assert.equal(scale.ny, TS_BANDS);
  assert.equal(stft.nx, TS_NT);
  assert.equal(stft.ny, TS_NF);
});

test("time_scale_map：缺 stft_time/stft_freq → 降级点名并改用下标轴", async () => {
  const def: FrameDef = { ...TS_DEF, arrays: TS_DEF.arrays.filter(([n]) => n !== "stft_time" && n !== "stft_freq") };
  const scene = await sceneOf(def);
  assert.match(degradedOf(scene), /stft_time/);
  assert.match(degradedOf(scene), /stft_freq/);
  assert.match(degradedOf(scene), /下标/);
});

test("time_scale_map：scene.t 恒为 0（mode=none，不许偷用请求 t）", async () => {
  const scene = await sceneOf(TS_DEF, { t: 99 });
  assert.equal(scene.t, 0);
});

// --------------------------------------------------------------------------- //
// 4. denoise_threshold：三条竖线是真证据
// --------------------------------------------------------------------------- //

test("denoise_threshold：当前 λ、最优 λ、通用阈值三条竖线都画在 curves 面板上", async () => {
  const scene = await sceneOf(DN_DEF);
  const verticals = lineSeriesOn(scene, "curves").filter(
    (s) => s.x.length === 2 && s.x[0] === s.x[1] && s.y[0] !== s.y[1],
  );
  const xs = verticals.map((s) => s.x[0] as number).sort((a, b) => a - b);
  assert.deepEqual(xs, [0.8, 0.8, 1.0], "缺了当前 λ / 最优 λ / 通用阈值中的一条");
});

test("denoise_threshold：通用阈值与最优值来自 meta，不从参数猜", async () => {
  const meta = { ...DN_DEF.meta, lam: 1.2, lam_opt: 0.4, universal: 1.4 };
  const scene = await sceneOf({ ...DN_DEF, meta });
  const verticals = lineSeriesOn(scene, "curves").filter(
    (s) => s.x.length === 2 && s.x[0] === s.x[1] && s.y[0] !== s.y[1],
  );
  const xs = verticals.map((s) => s.x[0] as number).sort((a, b) => a - b);
  assert.deepEqual(xs, [0.4, 1.2, 1.4]);
});

test("denoise_threshold：缺 universal_line → 降级点名（不许静默少一条线）", async () => {
  const def: FrameDef = { ...DN_DEF, arrays: DN_DEF.arrays.filter(([n]) => n !== "universal_line") };
  const scene = await sceneOf(def);
  assert.match(degradedOf(scene), /universal_line/);
});

test("denoise_threshold：曲线长度不一致 → 抛错（拒绝画错位的图）", async () => {
  const def: FrameDef = { ...DN_DEF, arrays: DN_DEF.arrays.map((r) => (r[0] === "mse_curve" ? [r[0], [1, 2], r[2]] as Row : r)) };
  await assert.rejects(async () => sceneOf(def), /长度/);
});

// --------------------------------------------------------------------------- //
// 5. vanishing_moments：掩码点只能散点、u 参考线必须在
// --------------------------------------------------------------------------- //

test("vanishing_moments：interior_valid=0 的点只以散点出现，折线不跨过它们", async () => {
  const scene = await sceneOf(VM_DEF);
  const lines = lineSeriesOn(scene, "ratio");
  // 内部比值的折线只应覆盖有效点（k=3,4）——测试里 invalid 点是 0,1,2（地板值）。
  const interiorLines = lines.filter((s) => Array.from(s.y as ArrayLike<number>).some((v) => Math.abs(v - 1e-5) < 1e-15));
  assert.ok(interiorLines.length > 0, "没有找到内部比值的折线");
  for (const s of interiorLines) {
    const xs = Array.from(s.x as ArrayLike<number>);
    assert.ok(xs.every((x) => x >= 3), `内部比值折线穿过了无效点：x=${xs.join(",")}`);
  }
  const scatters = scene.layers.filter(
    (l): l is Extract<Layer, { kind: "scatter" }> => l.kind === "scatter" && l.axes === "ratio",
  );
  assert.ok(scatters.some((s) => Array.from(s.y as ArrayLike<number>).some((v) => v <= 1.1e-18)), "地板点必须用散点标出");
});

test("vanishing_moments：u=2^-53 参考线存在且数值来自 u_line 数组", async () => {
  const scene = await sceneOf(VM_DEF);
  const hit = lineSeriesOn(scene, "ratio").some(
    (s) => s.y.length === VM_DEG.length && Array.from(s.y).every((v) => Math.abs(v - 1.11e-16) < 1e-30),
  );
  assert.ok(hit, "ratio 面板上缺少 u 参考线");
});

test("vanishing_moments：缺 interior_valid → ModelInputError（掩码是画图前提）", async () => {
  const def: FrameDef = { ...VM_DEF, arrays: VM_DEF.arrays.filter(([n]) => n !== "interior_valid") };
  await assert.rejects(async () => sceneOf(def), (err) => err instanceof ModelInputError);
});

// --------------------------------------------------------------------------- //
// 6. 降级不静默：窄画布
// --------------------------------------------------------------------------- //

test("窄画布（640×520）仍能过校验器，并明确说明右列文字被放弃", async () => {
  const scene = await sceneOf(DT_DEF, { width: 480, height: 520 });
  assert.deepEqual(validateScene(scene), []);
  assert.match(degradedOf(scene), /窄画布/);
});

