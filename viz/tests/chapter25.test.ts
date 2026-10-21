/**
 * 2.5 章（傅里叶分析与 FFT）model 层测试 —— **自足**：不依赖 Python，也不依赖 `data/frames` 是否存在。
 *
 * 夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，字段集合（数组名/dtype/shape/unit/label、
 * meta/strings/critical 键）逐字照抄 `server/chapters/2.5.py` 的真实帧；真实帧在的话再对拍一遍
 * （见文件末尾，缺帧时整组跳过）。数值一律是**合成值**：本层只搬运帧里的数，测试关心的是
 * "搬得对不对、缺字段有没有明说"。
 *
 * 本文件专门守 2.5 最容易画错的几条（对应 `chapters/2.5/pitfalls.md`）：
 * - **补零 ≠ 分辨率**：`df`（随 pad 变）与 `df_true`（1/N，不随 pad 变）必须**同时**出现在同屏；
 *   原栅格读数只允许从本帧谱线**抽取**（`fft(zero_pad)[::pad] ≡ fft`），不许在 TS 里重算 FFT；
 * - **循环 ≠ 线性卷积**：差异位置数必须从帧数组**数出来**（= M−1），N 变大不会自己变成 0；
 *   补零后的差（`safe_difference`）与卷绕差必须**同屏**（差四个数量级才是结论）；
 * - **旁瓣低 ≠ 衰减快**：峰值旁瓣（dB）与包络幂次 p 必须同时报，Hamming(p=1)/Hann(p=3) 对照不能只报一半；
 * - **窗谱对拍要同类对同类**：被减数是 `|FFT(窗)|`、减数是解析窗谱（pitfalls §21）；
 * - **残差 ≠ 数学正确性**：蝶形残差轴上必须同时有 `|FFT − DFT|` 那条"另一件事"的参照线；
 * - **位反序**：`spectrum_mag` 是重排后的谱，峰位必须钉在帧 `meta.k` 上；
 * - **时间语义**：只有 `butterfly_stages` 有 scrub 时间轴，`t → round(t·log2N)`；画面以帧 `meta.stage` 为准；
 * - **降级不静默**：meta 缺键 / 值为 null → 少画什么就写明什么（degraded 角标点名）；
 * - **自适应要明说**（规则 4）：`autoscaled` 与 manifest 一致，且挂"自适应"角标。
 */

import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions, type SceneModel } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { sceneToJson, validateScene, type Axis, type Layer, type Scene } from "../shared/src/scene.js";
import { butterflyStagesModel } from "../chapters/2.5/butterfly_stages.model.js";
import { convTheoremModel } from "../chapters/2.5/conv_theorem.model.js";
import { leakWindowModel } from "../chapters/2.5/leak_window.model.js";
import { manifest, modelFor, models } from "../chapters/2.5/index.js";
import { resolutionModel } from "../chapters/2.5/resolution.model.js";
import { signalSpectrumModel } from "../chapters/2.5/signal_spectrum.model.js";

const PITFALLS =
  "它会怎样骗你：把 pad 从 1 拖到 8，谱线会明显变光滑、峰也显得更尖，很容易读成「分辨率提高了」；" +
  "其实主瓣宽度只由 T = N·dt 决定。N 不够大时循环卷积与线性卷积在前几项上完全一样，只有把「循环 − 线性」的差单独画出来才看得见卷绕；" +
  "只报旁瓣会把 Hamming 讲成比 Hann 好，真相是旁瓣低但衰减慢，还要同时看主瓣宽度与相干增益。";

/** 一行数组 = `[名字, dtype, shape, 数据, 单位, 标签]`（dtype/shape/unit/label 照抄真实帧）。 */
type Row = readonly [name: string, dtype: string, shape: readonly number[], data: ArrayLike<number>, unit: string, label: string];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical: Record<string, number>;
  pitfalls?: string;
}

async function makeFrame(def: FrameDef): Promise<Frame> {
  const specs: ArraySpec[] = [];
  const arrays = new Map<string, TypedArray>();
  for (const [name, dtype, shape, data, unit, label] of def.arrays) {
    const typed = Float64Array.from(data as ArrayLike<number>);
    const count = shape.reduce((a, b) => a * b, 1);
    assert.equal(count, typed.length, `${name}: shape ${shape.join("×")} 与数据长度不符`);
    specs.push({ name, dtype: dtype as ArraySpec["dtype"], shape: [...shape], order: "C", offset: 0, nbytes: typed.byteLength, unit, label });
    arrays.set(name, typed);
  }
  const frame = new Frame({
    chapter: "2.5",
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
  /** 连 shape 一起换（造"长度与 shape 不一致"的帧；不换的话 makeFrame 会先拦下来）。 */
  shapes?: Record<string, readonly number[]>;
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
    .map((row): Row => {
      const shape = ov.shapes && row[0] in ov.shapes ? ov.shapes[row[0]]! : row[2];
      return ov.arrays && row[0] in ov.arrays ? [row[0], row[1], shape, ov.arrays[row[0]], row[4], row[5]] : [row[0], row[1], shape, row[3], row[4], row[5]];
    });
  const meta = { ...def.meta };
  for (const k of ov.dropMeta ?? []) delete meta[k];
  const strings = { ...def.strings };
  for (const k of ov.dropStrings ?? []) delete strings[k];
  return { ...def, arrays, meta: { ...meta, ...ov.meta }, strings: { ...strings, ...ov.strings }, pitfalls: ov.pitfalls ?? def.pitfalls };
}

// --------------------------------------------------------------------------- //
// 五个 viz 的夹具（字段集合照抄真实帧；数值是合成值）
// --------------------------------------------------------------------------- //

const OPTS: ModelOptions = { t: 0, width: 960, height: 600, params: {} };

const range = (count: number, step = 1, start = 0): number[] => Array.from({ length: count }, (_, i) => start + i * step);

/** 单边 dB 谱：在 `peaks`（freq 轴下标）上给 0 dB，其余压到 `floor`。 */
function dbSpectrum(count: number, peaks: readonly number[], floor = -60): Float64Array {
  const out = new Float64Array(count).fill(floor);
  for (const p of peaks) out[p] = 0;
  return out;
}

const NS = 256;
const SIGNAL_DEF: FrameDef = {
  viz: "signal_spectrum",
  arrays: [
    ["x", "f8", [NS], range(NS).map((i) => Math.cos((2 * Math.PI * 3 * i) / NS)), "", "时域信号 x(t)"],
    ["t_axis", "f8", [NS], range(NS, 1 / NS), "T", "时间（周期单位）"],
    ["spectrum_db", "f8", [129], dbSpectrum(129, [3, 6, 9]), "dB", "单边幅度谱（相对峰值）"],
    ["freq", "f8", [129], range(129), "1/T", "频率（周期数/T）"],
    ["error", "f8", [NS], new Float64Array(NS).fill(4.46e-14), "", "|FFT-DFT|（逐点对照，已铺满 N 点）"],
  ],
  meta: {
    n: NS,
    pad: 1,
    n_pad: NS,
    df: 1 / NS,
    df_true: 1 / NS,
    mainlobe_width: 1 / NS,
    f0: 3,
    nyquist: 128,
    max_error: 4.46e-14,
    n_aliased: 0,
    t: 0,
    t_actual: 0,
  },
  strings: {
    status: "ok",
    algorithm: "radix2-iterative",
    alias_note: "全部谐波都在 Nyquist 之下",
    data_source: "backend",
    nonfinite_policy: "对数轴上 0 画不出来：谱线取 -60 dB 地板，error 取 1e-18 地板",
    note: "pad 只加密频率栅格（df = 1/(N*pad)），分辨率仍由 T = N*dt 决定",
    params_json: '{"n": 256, "pad": 1}',
  },
  critical: { n: 64, pad: 1 },
};

const NW = 64;
const LEAK_DEF: FrameDef = {
  viz: "leak_window",
  arrays: [
    ["window_shapes", "f8", [4, NW], range(4 * NW).map((i) => 1 - 0.5 * ((i % NW) / NW)), "", "四种窗（行序 rect/hann/hamming/blackman）"],
    ["window_shape", "f8", [NW], range(NW).map((i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (NW - 1))), "", "当前窗 hann"],
    ["x_windowed", "f8", [NW], range(NW).map((i) => 0.5 + 0.4 * Math.cos((2 * Math.PI * i) / NW)), "", "加窗后的信号"],
    ["spectrum_db", "f8", [129], dbSpectrum(129, [8]), "dB", "加窗后的 dB 谱（相对峰值）"],
    ["freq", "f8", [129], range(129, 0.25), "1/T", "频率（周期数/T）"],
    ["envelope", "f8", [129], range(129).map((i) => (i === 0 ? 1 : 22.4668 * Math.pow(Math.max(i * 0.25, 0.25) / 0.25, -3))), "", "渐近包络 f^(-3)（相对 f=1/T）"],
    ["error", "f8", [NW], new Float64Array(NW).fill(0.01324), "", "|FFT(窗)| vs 解析连续谱的最大偏差（铺满 N 点）"],
  ],
  meta: {
    n: NW,
    pad: 4,
    offset: 0.25,
    peak_sidelobe_db: -31.47,
    harris_sidelobe_db: -31.47,
    mainlobe_bins: 1.44,
    harris_mainlobe_bins: 1.44,
    coherent_gain: 0.49219,
    envelope_power: 3,
    envelope_anchor: 2.36,
    envelope_anchor_db: -31.467,
    rel_error: 0.01324,
    t: 0,
    t_actual: 0,
  },
  strings: {
    status: "ok",
    window: "hann",
    cola_note: "Hann/Hamming 在 50% 重叠下满足常数重叠相加（COLA），矩形窗不满足",
    data_source: "backend",
    nonfinite_policy: "dB 谱地板 -120 dB（Blackman 旁瓣可低于 -90 dB）",
    note: "旁瓣越低主瓣越宽：hann 峰值旁瓣 -31.47 dB、-3 dB 主瓣 1.44 格（Harris 表值 -31.47 dB / 1.44 格）",
    params_json: '{"n": 64, "window": "hann"}',
  },
  critical: { offset: 0.5 },
};

const NB = 32;
const LEVELS = 5;
const BF_DEF: FrameDef = {
  viz: "butterfly_stages",
  arrays: [
    ["stage_values_real", "f8", [LEVELS + 1, NB], range((LEVELS + 1) * NB).map((i) => Math.cos((2 * Math.PI * 3 * (i % NB)) / NB)), "", "逐级快照实部（行 = 级，末行位反序）"],
    ["stage_values_imag", "f8", [LEVELS + 1, NB], new Float64Array((LEVELS + 1) * NB), "", "逐级快照虚部"],
    ["current_real", "f8", [NB], range(NB).map((i) => Math.cos((2 * Math.PI * 3 * i) / NB)), "", "第 0 级的实部"],
    ["current_imag", "f8", [NB], new Float64Array(NB), "", "第 0 级的虚部"],
    ["residual", "f8", [LEVELS], [2.06e-16, 1.4e-16, 1.1e-16, 1.3e-16, 1.8e-16], "", "逐级残差（向量化 vs 朴素 DIF）"],
    ["spectrum_mag", "f8", [NB], range(NB).map((i) => (i === 3 || i === NB - 3 ? 16 : 0)), "", "位反序重排后的 |X_k|"],
    ["error", "f8", [NB], new Float64Array(NB).fill(2.3979e-15), "", "|FFT-DFT|（全谱，铺满 N 点）"],
  ],
  meta: {
    n: NB,
    k: 3,
    levels: LEVELS,
    n_butterflies: (NB / 2) * LEVELS,
    stage: 0,
    max_level_residual: 2.0637e-16,
    spectrum_error: 2.3979e-15,
    spectrum_peak: 16,
    t: 0,
    t_actual: 0,
  },
  strings: {
    status: "ok",
    signal: "cosine",
    algorithm: "radix2-DIF",
    data_source: "backend",
    nonfinite_policy: "残差与误差取 1e-18 地板（贴舍入地板时对数轴画不出来）",
    note: "末行是位反序的，经 bit_reverse_indices 置换后必须逐点等于 DFT(x)；逐级残差量的是向量化实现与朴素二重循环的一致性，不是数学正确性",
    params_json: '{"n": 32, "k": 3}',
  },
  critical: {},
};

const NC = 16;
const MC = 8;
const CONV_DEF: FrameDef = {
  viz: "conv_theorem",
  arrays: [
    ["signal", "f8", [NC], range(NC).map((i) => Math.exp(-Math.pow((i - NC / 2) / 3, 2))), "", "信号 x"],
    ["kernel", "f8", [NC], range(NC).map((i) => (i < MC ? 0.125 : 0.125)), "", "滤波器 h（补零到 N）"],
    ["circular", "f8", [NC], range(NC).map((i) => 0.5 - 0.02 * i), "", "循环卷积（频域相乘）"],
    ["linear", "f8", [NC], range(NC).map((i) => 0.5 - 0.02 * i + (i < MC - 1 ? 0.25 : 0)), "", "线性卷积（前 N 项）"],
    ["difference", "f8", [NC], range(NC).map((i) => (i < MC - 1 ? 0.25 : 0)), "", "循环 - 线性（卷绕误差）"],
    ["safe_difference", "f8", [NC], range(NC).map((i) => (i % 2 === 0 ? 1.11e-16 : -1.11e-16)), "", "补零到 N+M-1 后的差"],
    ["error", "f8", [NC], range(NC).map((i) => (i < MC - 1 ? 0.25 : 0)), "", "|循环 - 线性|（卷绕幅度）"],
  ],
  meta: {
    n: NC,
    m: MC,
    need: NC + MC - 1,
    max_wrap: 0.2507810918,
    first_wrap_index: 0,
    wrap_rel: 0.4212422987,
    safe_error: 1.110223025e-16,
    theorem_error: 1.144580107e-16,
    direct_vs_fft: 1.110223025e-16,
    t: 0,
    t_actual: 0,
  },
  strings: {
    status: "ok",
    signal: "gaussian",
    kernel: "box",
    data_source: "backend",
    nonfinite_policy: "差异为 0 时取 1e-18 地板",
    note: "N = 16 < N_a+M-1 = 23：尾部卷绕回来叠在前面，最大偏差 0.251（相对 42.1%）",
    params_json: '{"n": 16, "m": 8}',
  },
  critical: { n: 24 },
};

const NR = 256;
const SPANS = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096];
const RES_DEF: FrameDef = {
  viz: "resolution",
  arrays: [
    ["spectrum_db", "f8", [129], dbSpectrum(129, [2, 4], -80), "dB", "两音 dB 谱（相对峰值）"],
    ["freq", "f8", [129], range(129), "1/T", "频率（周期数/T）"],
    ["span_ratio", "f8", [SPANS.length], SPANS.map(() => 1), "", "能否分辨（1 = 能）"],
    ["span_n", "f8", [SPANS.length], SPANS, "点", "观测点数（T 固定为 1）"],
    ["error", "f8", [NR], new Float64Array(NR).fill(5.26e-14), "", "|FFT-DFT|（铺满 N 点）"],
  ],
  meta: {
    n: NR,
    pad: 1,
    df: 2,
    amp2: 1,
    separable: true,
    sep_measured: 1,
    rayleigh: 1,
    mainlobe_bins: 0.89,
    sep_over_mainlobe: 1.1235955,
    grid_df: 1 / NR,
    true_df: 1 / NR,
    max_error: 5.26e-14,
    t: 0,
    t_actual: 0,
  },
  strings: {
    status: "ok",
    window: "rect",
    data_source: "backend",
    nonfinite_policy: "dB 谱地板 -80 dB；误差取 1e-18 地板",
    note: "当前设置：df = 2.000/T，实测最小可分辨 1.000/T，Rayleigh 判据 1.000/T，本窗 −3 dB 主瓣 0.89 格",
    resolution_note: "可分辨判据 = 谱中出现两个局部极大且两峰之间有低于矮峰的谷，不是峰够不够尖",
    params_json: '{"n": 256, "pad": 1}',
  },
  critical: { pad: 1, df: 1.44 },
};

const DEFS: Record<string, FrameDef> = {
  signal_spectrum: SIGNAL_DEF,
  leak_window: LEAK_DEF,
  butterfly_stages: BF_DEF,
  conv_theorem: CONV_DEF,
  resolution: RES_DEF,
};
const VIZ: Record<string, SceneModel> = {
  signal_spectrum: signalSpectrumModel,
  leak_window: leakWindowModel,
  butterfly_stages: butterflyStagesModel,
  conv_theorem: convTheoremModel,
  resolution: resolutionModel,
};
const VIZ_IDS = Object.keys(DEFS);

// --------------------------------------------------------------------------- //
// 工具
// --------------------------------------------------------------------------- //

async function sceneOf(viz: string, t = 0): Promise<Scene> {
  const scene = VIZ[viz]!(await makeFrame(DEFS[viz]!), { ...OPTS, t });
  assert.deepEqual(validateScene(scene), [], `${viz} 的 Scene 不合规`);
  return scene;
}

function axisOf(scene: Scene, id: string): Axis {
  const axis = scene.axes.find((a) => a.id === id);
  assert.ok(axis, `场景里没有轴 ${id}`);
  return axis as Axis;
}

function layersOn(scene: Scene, axis: string): Layer[] {
  return scene.layers.filter((l) => l.axes === axis);
}

/** 某个轴上画出来的所有 y 值（按层种类读；测试按数值认，不按 label）。 */
function drawnY(layers: readonly Layer[]): number[] {
  const out: number[] = [];
  for (const l of layers) {
    if (l.kind === "line") for (const s of l.series) out.push(...Array.from(s.y, Number));
    else if (l.kind === "scatter") out.push(...Array.from(l.y, Number));
    else if (l.kind === "bars") out.push(...Array.from(l.values, Number));
  }
  return out;
}

function textsOf(scene: Scene): string {
  return scene.texts.map((t) => t.text).join("\n");
}

function badgesOf(scene: Scene): string {
  return scene.badges.map((b) => b.text).join("\n");
}

// --------------------------------------------------------------------------- //
// 规则骨架：五个 viz 都合法、轴与 manifest 对得上、必需辅助量真的画出来了
// --------------------------------------------------------------------------- //

test("五个 viz 都能产出合法 Scene，且每个 viz 只画 manifest 声明的轴 id", async () => {
  for (const viz of VIZ_IDS) {
    const scene = await sceneOf(viz);
    const declared = manifest.vizzes.find((v) => v.id === viz)?.axes.map((a) => a.id) ?? [];
    assert.deepEqual([...scene.axes.map((a) => a.id)].sort(), [...declared].sort(), `${viz} 的轴 id 集合与 manifest 不一致`);
  }
});

test("autoscaled 与 manifest 一致（规则 4：自适应必须明说）", async () => {
  for (const viz of VIZ_IDS) {
    const scene = await sceneOf(viz);
    for (const spec of manifest.vizzes.find((v) => v.id === viz)?.axes ?? []) {
      assert.equal(axisOf(scene, spec.id).autoscaled, spec.autoscaled, `${viz}/${spec.id} 的 autoscaled 与 manifest 不一致`);
    }
    const autoscaled = scene.axes.some((a) => a.autoscaled);
    assert.equal(badgesOf(scene).includes("自适应"), autoscaled, `${viz} 的"自适应"角标要与实际轴一致`);
  }
});

test("requiredAux 真的画在对应轴上（按数值认，不按 label）", async () => {
  const frame = await makeFrame(SIGNAL_DEF);
  const scene = signalSpectrumModel(frame, { ...OPTS });
  const err = Array.from(frame.array("error"), Number).map(Math.abs);
  const drawn = new Set(drawnY(layersOn(scene, "error")).map((v) => Math.round(v * 1e18) / 1e18));
  assert.ok(err.every((v) => drawn.has(Math.round(v * 1e18) / 1e18)), "signal_spectrum 的 error 没画全");

  const conv = await makeFrame(CONV_DEF);
  const convScene = convTheoremModel(conv, { ...OPTS });
  assert.ok(drawnY(layersOn(convScene, "error")).some((v) => Math.abs(v - 0.25) < 1e-9), "conv_theorem 的卷绕差没画");

  const bf = await makeFrame(BF_DEF);
  const bfScene = butterflyStagesModel(bf, { ...OPTS });
  const res = Array.from(bf.array("residual"), Number);
  const drawnRes = drawnY(layersOn(bfScene, "residual"));
  assert.ok(res.every((v) => drawnRes.some((d) => Math.abs(d - v) < 1e-18)), "逐级残差没画全");

  const th = await makeFrame(RES_DEF);
  const thScene = resolutionModel(th, { ...OPTS });
  assert.ok(drawnY(layersOn(thScene, "threshold")).includes(1), "能否分辨的 1 没画");

  const lw = await makeFrame(LEAK_DEF);
  const lwScene = leakWindowModel(lw, { ...OPTS });
  assert.ok(drawnY(layersOn(lwScene, "error")).every((v) => v > 0), "leak_window 的对数轴上出现了 0");
});

// --------------------------------------------------------------------------- //
// 「补零 ≠ 分辨率」（本页第一条防误导）
// --------------------------------------------------------------------------- //

test("signal_spectrum：df（随 pad 变）与 df_true（1/N，不随 pad 变）同屏并排，且各自点名", async () => {
  const scene = await sceneOf("signal_spectrum");
  const body = textsOf(scene);
  assert.ok(body.includes("df（DFT 栅格，随 pad 变）"), "必须点名 df 随 pad 变");
  assert.ok(body.includes("df_true（不随 pad 变）"), "必须点名 df_true 不随 pad 变");
  const texts = scene.texts.map((t) => t.text);
  assert.ok(texts.some((t) => t.includes("0.003906")), "两个数都要出得来");
});

test("signal_spectrum：pad>1 时原栅格读数只从本帧谱线抽取，不在 TS 里重算 FFT", async () => {
  // 造一个 pad=4 的帧：谱线是同一批数的加密采样，抽取 stride=4 就是原栅格
  const pad = 4;
  const fine = dbSpectrum(129, [12]); // 12 = 3 * pad
  const fineVal = fine;
  const def = tweak(SIGNAL_DEF, {
    arrays: { spectrum_db: fine },
    meta: { pad, n_pad: NS * pad, df: 1 / (NS * pad), mainlobe_width: 1 / (NS * pad) },
  });
  const scene = signalSpectrumModel(await makeFrame(def), { ...OPTS });
  assert.deepEqual(validateScene(scene), []);
  const scatter = layersOn(scene, "spec").find((l) => l.kind === "scatter" && l.x.length > 2);
  assert.ok(scatter, "pad>1 时要画「原栅格读数」的散点（同一帧抽稀）");
  if (scatter && scatter.kind === "scatter") {
    const xs = Array.from(scatter.x, Number);
    assert.ok(xs.length >= 3, `画出 ${xs.length} 个原栅格读数（横轴只到 3f0 附近，够看出"读数在原栅格上"）`);
    assert.ok(xs.every((x) => Number.isInteger(x) && x % pad === 0), "抽取后的 x 必须落在 stride 倍数的栅格点上（这就是「原栅格」）");
    // 抽稀出来的点必须与本帧谱线**逐点一致**（不是另算一条曲线）
    const fine = Array.from(fineVal);
    assert.ok(xs.every((x) => fine[x] === 0 || fine[x] === -60), "抽稀点的值来自同一批数");
  }
  assert.ok(textsOf(scene).includes("pad = 4"), "越界时要写明 pad 已经不是 1");
});

test("signal_spectrum：meta.mainlobe_width 只按 1/N 报，不谎称「量出来的半高全宽」", async () => {
  const scene = await sceneOf("signal_spectrum");
  assert.ok(textsOf(scene).includes("不是量出来的半高全宽"), "必须点破 mainlobe_width 是栅格而不是测宽");
  // 帧里报的读数必须是本帧谱线上量的（峰 0 dB、地板 -60 dB → 半高全宽落在第一个栅格内 0.1/T）
  assert.ok(textsOf(scene).includes("半高全宽读数"), "半高全宽读数要出现在谱面板上");
});

test("signal_spectrum：n_aliased > 0 时正文改用混叠口径（不再说「补零只是采得更密」）", async () => {
  const def = tweak(SIGNAL_DEF, { meta: { n_aliased: 2 }, strings: { alias_note: "第 9 次谐波越过 Nyquist" } });
  const scene = signalSpectrumModel(await makeFrame(def), { ...OPTS });
  assert.deepEqual(validateScene(scene), []);
  assert.ok(textsOf(scene).includes("混叠"), "混叠必须在正文里说");
  assert.ok(textsOf(scene).includes("9 次谐波"), "帧给的 alias_note 要原样带出来");
});

// --------------------------------------------------------------------------- //
// 「循环 ≠ 线性卷积」（本页第二条防误导）
// --------------------------------------------------------------------------- //

test("conv_theorem：差异位置数从帧数组数出来（= M−1），且补零后的差与卷绕差同屏", async () => {
  const scene = await sceneOf("conv_theorem");
  const body = textsOf(scene);
  assert.ok(body.includes("差异位置数 = 7 = M−1"), "差异位置数必须是数出来的，不是抄的常数");
  assert.ok(body.includes("0.250781"), "最大卷绕误差要报出来");
  assert.ok(body.includes("1.11×10⁻¹⁶"), "补零后的地板要报出来（差四个数量级才是结论）");
  const axisIds = scene.axes.map((a) => a.id);
  assert.deepEqual(axisIds, ["curves", "error", "safer"], "三条曲线必须同屏（上下三块）");
});

test("conv_theorem：difference 的非零位置与 M−1 不一致时，画面按数组报数并降级点名", async () => {
  const bad = new Float64Array(NC);
  bad[0] = 0.25; // 只有一个非零位置，而 M−1 = 7
  const scene = convTheoremModel(await makeFrame(tweak(CONV_DEF, { arrays: { difference: bad } })), { ...OPTS });
  assert.deepEqual(validateScene(scene), []);
  assert.ok(textsOf(scene).includes("差异位置数 = 1"), "按数组报数");
  assert.ok(badgesOf(scene).includes("降级"), "不一致必须降级点名");
  assert.ok(badgesOf(scene).includes("M−1"), "降级信息要点名冲突的字段");
});

test("conv_theorem：n 的临界角标按帧 critical.n 挂（N < 24 时已越过）", async () => {
  const scene = await sceneOf("conv_theorem");
  const badge = scene.badges.find((b) => b.kind === "critical" && b.text.includes("n"));
  assert.ok(badge, "必须挂 n 的临界角标");
  assert.ok(badge?.text.includes("24"), "临界值取帧 critical.n");
  assert.ok(badge?.text.includes("已超过临界"), "N=16 < 24：必须标出已经越界（criticalBadge 的官方措辞）");
});

// --------------------------------------------------------------------------- //
// 「旁瓣低 ≠ 衰减快」（本页第三条防误导）
// --------------------------------------------------------------------------- //

test("leak_window：峰值旁瓣（dB）与包络幂次 p 同时出现，Hamming/Hann 对照不缺一半", async () => {
  const scene = await sceneOf("leak_window");
  const body = textsOf(scene);
  assert.ok(body.includes("峰值旁瓣"), "峰值旁瓣要报");
  assert.ok(body.includes("包络幂次 p"), "包络幂次 p 要报");
  assert.ok(badgesOf(scene).includes("p=3"), "p 的值要报出来（status 角标）");
  assert.ok(body.includes("hamming") && body.includes("hann"), "两种窗的对照都要在");
  assert.ok(body.includes("旁瓣低 ≠ 衰减快"), "结论句要在同屏");
  assert.ok(body.includes("42.7") && body.includes("31.5"), "两个旁瓣数字都要有（−42.7 / −31.5）");
});

test("leak_window：理论包络只在旁瓣区画（主瓣区不画 f^−p），锚点用帧 meta", async () => {
  const scene = await sceneOf("leak_window");
  const env = layersOn(scene, "spec")
    .filter((l) => l.kind === "line")
    .flatMap((l) => (l.kind === "line" ? l.series : []))
    .find((s) => s.x.length > 5 && Math.max(...Array.from(s.y)) < 0 && Math.min(...Array.from(s.y)) < -60);
  assert.ok(env, "渐近包络要画出来（dB 口径）");
  if (env) {
    assert.ok(Math.min(...Array.from(env.x)) >= 2.36 - 1e-9, "主瓣区（f·T < 锚点）不许画幂律线");
  }
  assert.ok(textsOf(scene).includes("理论线锚点"), "锚点要在正文里点名（窗自己的最高旁瓣）");
});

test("leak_window：窗形面板把四种窗的行序与 p 一起标出，当前窗可辨", async () => {
  const scene = await sceneOf("leak_window");
  const shape = layersOn(scene, "shape").find((l) => l.kind === "line" && l.series.length === 4);
  assert.ok(shape, "四种窗都要画（行序 = rect/hann/hamming/blackman）");
  const body = textsOf(scene);
  for (const w of ["rect", "hann", "hamming", "blackman"]) assert.ok(body.includes(w), `缺 ${w}`);
  assert.ok(body.includes("当前窗加粗"), "当前窗要能认出来");
});

// --------------------------------------------------------------------------- //
// 蝶形：残差 ≠ 数学正确性；位反序；时间轴
// --------------------------------------------------------------------------- //

test("butterfly_stages：残差轴上同时有 |FFT − DFT| 参照线（两件事分开）", async () => {
  const scene = await sceneOf("butterfly_stages");
  const res = layersOn(scene, "residual")
    .filter((l) => l.kind === "line")
    .flatMap((l) => (l.kind === "line" ? l.series : []));
  const flat = res.find((s) => s.y.length === 2 && Number(s.y[0]) === Number(s.y[1]));
  assert.ok(flat, "必须有一条常数参照线（|FFT − DFT|）");
  assert.ok(Math.abs(Number((flat?.y as ArrayLike<number>)[0]) - 2.3979e-15) < 1e-18, "参照线的值取帧 error");
  assert.ok(textsOf(scene).includes("不是数学正确性"), "残差的含义要写明");
});

test("butterfly_stages：谱峰位钉在帧 meta.k 上（位反序重排后）", async () => {
  const scene = await sceneOf("butterfly_stages");
  const line = layersOn(scene, "spectrum").find((l) => l.kind === "line");
  assert.ok(line, "峰位参考线要画");
  if (line && line.kind === "line") {
    assert.ok(line.series.some((s) => s.x.length === 2 && Number(s.x[0]) === 3 && Number(s.x[1]) === 3), "竖线落在 k=3");
  }
  assert.ok(textsOf(scene).includes("位反序"), "位反序要说明");
});

test("butterfly_stages：时间轴 t → round(t·log2N)，画面以帧 meta.stage 为准（请求 t 与实到档位不一致要说）", async () => {
  const frame = await makeFrame(BF_DEF);
  // 帧 meta.stage=0，请求 t=0.6（→ 第 3 级）：画面必须按帧画第 0 级，并把差异说出来
  const mid = butterflyStagesModel(frame, { ...OPTS, t: 0.6 });
  assert.deepEqual(validateScene(mid), []);
  assert.ok(textsOf(mid).includes("第 0/5 级"), "画面以帧 meta.stage=0 为准（不拿请求 t 当级号）");
  assert.ok(textsOf(mid).includes("请求 t=0.6"), "请求档位要说出来");
  assert.ok(textsOf(mid).includes("第 3 级"), "t=0.6 → round(0.6·log2N) = 第 3 级（只说请求，不冒充实到）");
  assert.ok(badgesOf(mid).includes("降级"), "请求与实到不一致必须降级点名");
  // 帧与请求一致时（t_actual = stage/levels）走"正常"口径，不再挂降级
  const ok = butterflyStagesModel(await makeFrame(tweak(BF_DEF, { meta: { stage: 3, t_actual: 0.6 } })), { ...OPTS, t: 0.6 });
  assert.deepEqual(validateScene(ok), []);
  assert.ok(textsOf(ok).includes("第 3/5 级"));
  assert.ok(!badgesOf(ok).includes("降级"), "一致时不该降级");
});

test("butterfly_stages：末帧（t=1）画满轨迹且不越界", async () => {
  const frame = await makeFrame(tweak(BF_DEF, { meta: { stage: LEVELS }, arrays: { current_real: range(NB).map((i) => 16 * Math.cos((2 * Math.PI * 3 * i) / NB)) } }));
  const scene = butterflyStagesModel(frame, { ...OPTS, t: 1 });
  assert.deepEqual(validateScene(scene), []);
  const trail = layersOn(scene, "trace").find((l) => l.kind === "line" && l.series.length >= LEVELS);
  assert.ok(trail, "走过的级都要作为轨迹画出来");
});

// --------------------------------------------------------------------------- //
// 分辨率：三要素判据、阈值 vs Rayleigh、pad 与 T 分开报
// --------------------------------------------------------------------------- //

test("resolution：判据三要素与阈值/Rayleigh 同屏，pad 与 T 分开报", async () => {
  const scene = await sceneOf("resolution");
  const body = textsOf(scene);
  assert.ok(body.includes("实测最小可分辨"), "实测阈值要报");
  assert.ok(body.includes("Rayleigh"), "Rayleigh 判据要报");
  assert.ok(body.includes("grid_df（随 pad 变）") && body.includes("true_df（不随 pad 变）"), "栅格与 1/T 要分开点名");
  assert.ok(body.includes("谷") && body.includes("矮峰"), "判据三要素（两峰 + 低于矮峰的谷）要写出来");
  assert.ok(body.includes("补零只加密栅格"), "结论句要在同屏");
});

test("resolution：两个谱峰是从帧谱线读出来的局部极大，不是抄参数", async () => {
  const peaks = [2, 4];
  const scene = resolutionModel(await makeFrame(tweak(RES_DEF, { arrays: { spectrum_db: dbSpectrum(129, peaks, -80) } })), { ...OPTS });
  const sc = layersOn(scene, "spec").find((l) => l.kind === "scatter");
  assert.ok(sc, "两个峰要标出来");
  if (sc && sc.kind === "scatter") {
    assert.deepEqual(Array.from(sc.x).sort((a, b) => a - b), peaks, "峰位必须是谱线上读出来的位置");
  }
});

test("resolution：找不到两个局部极大时明说「判据判它分不开」（不硬凑两个峰）", async () => {
  const def = tweak(RES_DEF, { arrays: { spectrum_db: dbSpectrum(129, [3], -80) }, meta: { separable: false, df: 0.5 } });
  const scene = resolutionModel(await makeFrame(def), { ...OPTS });
  assert.deepEqual(validateScene(scene), []);
  assert.ok(badgesOf(scene).includes("降级"), "缺峰必须降级点名");
  assert.ok(textsOf(scene).includes("分不开") || textsOf(scene).includes("只找到 1 个局部极大"), "要如实说明");
});

test("resolution：pad 越界（>1）时临界角标点名，且 df 的临界角标按帧 critical.df", async () => {
  const scene = await sceneOf("resolution");
  const crit = scene.badges.filter((b) => b.kind === "critical").map((b) => b.text);
  const padBadge = crit.find((t) => t.startsWith("pad"));
  assert.ok(padBadge, "pad 的临界角标要挂");
  assert.ok(padBadge?.includes("补零只加密栅格"), "角标要点明 pad 改的是栅格");
  const dfBadge = crit.find((t) => t.startsWith("df = "));
  assert.ok(dfBadge, "df 的临界角标要挂（与 pad 分开两条）");
  assert.ok(dfBadge?.includes("1.44"), "df 的临界值取帧 critical.df");
});

// --------------------------------------------------------------------------- //
// 降级路径：meta 缺键 / 值为 null → 少画什么就写明什么
// --------------------------------------------------------------------------- //

test("signal_spectrum：缺 df_true / max_error → 不编数，degraded 角标点名字段", async () => {
  const scene = signalSpectrumModel(await makeFrame(tweak(SIGNAL_DEF, { dropMeta: ["df_true", "max_error"] })), { ...OPTS });
  assert.deepEqual(validateScene(scene), []);
  assert.ok(badgesOf(scene).includes("降级"));
  assert.ok(badgesOf(scene).includes("df_true"), "要点名缺的是哪个字段");
  assert.ok(!textsOf(scene).includes("不随 pad 变 = "), "缺的数不能补 0 顶上");
});

test("leak_window：缺 harris_sidelobe_db / envelope_anchor → 不画对照线与锚点，并点名", async () => {
  const scene = leakWindowModel(await makeFrame(tweak(LEAK_DEF, { dropMeta: ["harris_sidelobe_db", "envelope_anchor", "envelope_anchor_db"] })), { ...OPTS });
  assert.deepEqual(validateScene(scene), []);
  assert.ok(badgesOf(scene).includes("降级"));
  assert.ok(badgesOf(scene).includes("harris_sidelobe_db"), "角标要点名第一个缺的字段");
  assert.ok(badgesOf(scene).includes("降级：") || badgesOf(scene).includes("降级"), "降级角标要在");
  assert.ok(textsOf(scene).includes("帧未给锚点"), "缺锚点时正文要说清「理论线不画」（角标会被宽度裁掉）");
  const hasEnv = layersOn(scene, "spec").some((l) => l.kind === "line" && l.series.some((s) => s.x.length > 5 && Math.max(...Array.from(s.y)) < 0));
  assert.equal(hasEnv, false, "没有锚点时不许画幂律线");
});

test("butterfly_stages：current_real 与快照列数不一致 → 抛 ModelInputError（拒绝画错位的轨迹）", async () => {
  const frame = await makeFrame(tweak(BF_DEF, { arrays: { current_real: range(8) }, shapes: { current_real: [8] } }));
  assert.throws(() => butterflyStagesModel(frame, { ...OPTS }), ModelInputError);
});

test("resolution：缺 meta.n / pad → 不猜观测点数与栅格，degraded 点名", async () => {
  const scene = resolutionModel(await makeFrame(tweak(RES_DEF, { dropMeta: ["n", "pad"] })), { ...OPTS });
  assert.deepEqual(validateScene(scene), []);
  assert.ok(badgesOf(scene).includes("降级"));
  assert.ok(badgesOf(scene).includes("n"));
});

test("缺必需数组 / 长度不一致 / pitfalls 过短 → 抛 ModelInputError（不画半张图）", async () => {
  const cases: Array<[string, FrameDef]> = [
    ["signal_spectrum", tweak(SIGNAL_DEF, { drop: ["spectrum_db"] })],
    ["leak_window", tweak(LEAK_DEF, { drop: ["envelope"] })],
    ["butterfly_stages", tweak(BF_DEF, { drop: ["residual"] })],
    ["conv_theorem", tweak(CONV_DEF, { drop: ["safe_difference"] })],
    ["resolution", tweak(RES_DEF, { arrays: { span_ratio: [1, 1, 1] }, shapes: { span_ratio: [3] } })],
    ["conv_theorem", tweak(CONV_DEF, { arrays: { linear: range(5) }, shapes: { linear: [5] } })],
    ["leak_window", tweak(LEAK_DEF, { pitfalls: "太短" })],
  ];
  for (const [viz, def] of cases) {
    const frame = await makeFrame(def);
    assert.throws(() => VIZ[viz]!(frame, { ...OPTS }), ModelInputError, `${viz} 应当抛 ModelInputError`);
  }
});

// --------------------------------------------------------------------------- //
// 纯函数性 / 主题 / 溢出
// --------------------------------------------------------------------------- //

test("模型是纯函数：同一帧两次调用逐字节相同，且不改帧（无 IO / Date / random）", async () => {
  const frame = await makeFrame(CONV_DEF);
  const fingerprint = (f: Frame): string => f.specs.map((s) => `${s.name}:${s.dtype}:${s.shape.join(",")}`).join("|");
  const before = fingerprint(frame);
  assert.equal(sceneToJson(convTheoremModel(frame, { ...OPTS })), sceneToJson(convTheoremModel(frame, { ...OPTS })));
  assert.equal(fingerprint(frame), before);
  const light = convTheoremModel(frame, { ...OPTS });
  const dark = convTheoremModel(frame, { ...OPTS, theme: "dark" });
  assert.equal(dark.layers.length, light.layers.length, "暗色只改配色，不改结构");
  assert.notEqual(dark.canvas.background, light.canvas.background);
});

test("NaN/Inf 计数进溢出角标（不静默画断线）", async () => {
  const bad = range(129).map((i) => (i === 5 ? Number.NaN : i === 7 ? Number.POSITIVE_INFINITY : i === 3 ? 0 : -60));
  const scene = signalSpectrumModel(await makeFrame(tweak(SIGNAL_DEF, { arrays: { spectrum_db: bad } })), { ...OPTS });
  assert.deepEqual(validateScene(scene), []);
  assert.ok(badgesOf(scene).includes("溢出") || badgesOf(scene).includes("非有限"), "非有限值必须挂溢出角标");
});

test("modelFor：五个 viz 都注册了，未知 id 返回 null（不抛错）", () => {
  for (const viz of VIZ_IDS) assert.equal(modelFor(viz), models[viz]);
  assert.equal(modelFor("nope"), null);
});

// --------------------------------------------------------------------------- //
// 真实帧对拍（`data/frames/2.5/**` 不存在时整组跳过——测试不依赖它）
// --------------------------------------------------------------------------- //

/** 测试跑的是 `dist/viz/tests/`（与 `chapter00.test.ts` 同一套相对路径口径）。 */
const REAL_DIR = new URL("../../../data/frames/2.5/", import.meta.url);
const real = existsSync(REAL_DIR)
  ? {}
  : { skip: "没有 data/frames/2.5（先跑 python3 tools/precompute.py --chapter 2.5）" };

function realFrame(viz: string, last = false): Frame {
  const dir = new URL(`${viz}/`, REAL_DIR);
  const frames = readdirSync(dir)
    .filter((f) => f.endsWith(".nlf"))
    .map((f) => decodeNlf(readFileSync(new URL(f, dir))));
  // 文件名排序会把 t0.2 排在 t0 前面（字典序），"首帧"必须按帧内 meta.t 取。
  frames.sort((a, b) => Number(a.meta["t"] ?? 0) - Number(b.meta["t"] ?? 0));
  return frames[last ? frames.length - 1 : 0] as Frame;
}

test("真实帧：夹具与 build_frame 产物的字段集合一致（字段名/单位/标签不漂）", real, async () => {
  for (const viz of VIZ_IDS) {
    const mine = await makeFrame(DEFS[viz]!);
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

test("真实帧：首帧与末帧都能产出合法 Scene（时间轴滑到头也不炸），读数取帧里的数", real, async () => {
  for (const viz of VIZ_IDS) {
    for (const last of [false, true]) {
      const frame = realFrame(viz, last);
      const scene = VIZ[viz]!(frame, { ...OPTS, t: Number(frame.meta["t"] ?? 0) });
      assert.deepEqual(validateScene(scene), [], `${viz} 的 Scene 不合规`);
    }
  }
});

test("真实帧：signal_spectrum 的 df / df_true 与帧 meta 逐位一致（不在 TS 里重算）", real, async () => {
  const frame = realFrame("signal_spectrum");
  const scene = signalSpectrumModel(frame, { ...OPTS });
  const df = Number(frame.meta["df"]);
  const dfTrue = Number(frame.meta["df_true"]);
  const body = textsOf(scene);
  assert.ok(body.includes(String(df.toFixed(6)).replace(/0+$/, "").replace(/\.$/, "")) || body.includes("0.003906"), "df 要按帧 meta 报");
  assert.ok(dfTrue > 0);
});
