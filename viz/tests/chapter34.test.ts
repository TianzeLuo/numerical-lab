/**
 * 3.4 章（边值问题与谱方法）model 层测试 —— **自足**：不依赖 Python，也不依赖
 * `data/frames` 是否存在。夹具用 `new Frame(...)` + `encodeNlf`/`decodeNlf` 造帧，
 * 字段集合（数组名 / dtype / shape / 单位 / meta / strings 键）逐字照抄
 * `server/chapters/3.4.py` 的真实帧（数值是测试用的合成值）。
 *
 * 本文件守的几件"这张图会不会骗人"的规矩：
 *
 * 1. **截断占位不是数据**：`family`/`residual`/`error` 的 `_clip` 掩码位置填 1e6，
 *    模型必须把它们变成缺口；任何图层出现 1e6 都是失败。
 * 2. **规则 2**：`convergence_rate` 的 loglogfit 必须同时带实测斜率与理论斜率，
 *    且两个都是有限数；半对数主轴必须是 log。
 * 3. **节点不许假装均匀**：`node_clustering` 主面板既有真 Chebyshev 节点，也有明确
 *    标注的均匀对照；节点与均匀位置的偏离必须能被测出来。
 * 4. **稠密/奇异要画出来**：`diff_matrix_structure` 的热力图长度 = (N+1)²，
 *    奇异值谱跨十几个数量级，σmin+ 参考线在场。
 * 5. **降级不静默**：缺可选数组/ meta → 降级角标点名；缺必需数组 → `ModelInputError`。
 * 6. **time.mode="none" 的 viz 不许碰 t**：`scene.t` 恒为 0。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Frame, type ArraySpec, type TypedArray } from "../shared/src/frame.js";
import { ModelInputError, type ModelOptions } from "../shared/src/model.js";
import { decodeNlf, encodeNlf } from "../shared/src/nlf.js";
import { validateScene, type Layer, type Scene } from "../shared/src/scene.js";
import { manifest, modelFor, models } from "../chapters/3.4/index.js";

type Row = readonly [name: string, data: ArrayLike<number>, label: string, shape?: readonly number[]];
interface FrameDef {
  viz: string;
  arrays: readonly Row[];
  meta: Record<string, number | boolean | null>;
  strings: Record<string, string>;
  critical?: Record<string, number>;
  pitfalls?: string;
}

const PITFALLS =
  "它会怎样骗你：第一，残差贴地板不等于解准，衰减模态的舍入被 S 放大；" +
  "第二，插值误差在节点上恒为 0，那不是精度证据；第三，D_N 奇异，朴素条件数是零空间噪声。";
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
      unit: "1",
      label,
    });
    arrays.set(name, typed);
  }
  return decodeNlf(
    await encodeNlf(
      new Frame({
        chapter: "3.4",
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

interface Override {
  arrays?: Record<string, ArrayLike<number>>;
  meta?: Record<string, number | boolean | null>;
  strings?: Record<string, string>;
  critical?: Record<string, number>;
  drop?: readonly string[];
  dropMeta?: readonly string[];
  dropStrings?: readonly string[];
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
  for (const k of ov.dropMeta ?? []) delete meta[k];
  const strings = { ...def.strings };
  for (const k of ov.dropStrings ?? []) delete strings[k];
  return {
    ...def,
    arrays,
    meta: { ...meta, ...ov.meta },
    strings: { ...strings, ...ov.strings },
    critical: { ...(def.critical ?? {}), ...(ov.critical ?? {}) },
  };
}

const sceneOf = async (viz: string, over: Override = {}, opts: Partial<ModelOptions> = {}): Promise<Scene> =>
  (models[viz] as (f: Frame, o: ModelOptions) => Scene)(await makeFrame(tweak(DEFS[viz] as FrameDef, over)), { ...OPTS, ...opts });

const layersOn = (scene: Scene, id: string): Layer[] => scene.layers.filter((l) => l.axes === id);
/** 某个轴上的所有 y 值（line/scatter/loglogfit 都算），按图层顺序。 */
const ySeriesOn = (scene: Scene, id: string): ArrayLike<number>[] => {
  const out: ArrayLike<number>[] = [];
  for (const l of layersOn(scene, id)) {
    if (l.kind === "line") for (const s of l.series) out.push(s.y);
    else if (l.kind === "scatter") out.push(l.y);
    else if (l.kind === "loglogfit") out.push(l.err);
  }
  return out;
};
const badgesOf = (scene: Scene): string => scene.badges.map((b) => b.text).join(" | ");
const degradedOf = (scene: Scene): string => scene.badges.filter((b) => b.text.includes("降级")).map((b) => b.text).join(" | ");
const layerNumbers = (scene: Scene): number[] => {
  const out: number[] = [];
  for (const l of scene.layers) {
    if (l.kind === "line") for (const s of l.series) out.push(...Array.from(s.y as ArrayLike<number>));
    else if (l.kind === "scatter") out.push(...Array.from(l.y as ArrayLike<number>));
    else if (l.kind === "heat") out.push(...Array.from(l.values as ArrayLike<number>));
  }
  return out;
};
const hasArrayValues = (scene: Scene, axis: string, want: ArrayLike<number>, tol = 1e-12): boolean =>
  ySeriesOn(scene, axis).some(
    (s) => s.length === want.length && Array.from(s).every((v, i) => Math.abs(v - (want[i] as number)) <= tol * Math.max(1, Math.abs(want[i] as number))),
  );

// --------------------------------------------------------------------------- //
// 夹具（字段集合照抄真实帧）
// --------------------------------------------------------------------------- //

const SH_NX = 41;
const SH_X = Array.from({ length: SH_NX }, (_, i) => i / (SH_NX - 1));
const SH_EXACT = SH_X.map((x) => Math.exp(-3 * x));
const SH_M = 9;
const SH_SLOPES = Array.from({ length: SH_M }, (_, k) => -8 - (4 * k) / (SH_M - 1));
const SH_FAMILY: number[] = [];
const SH_FAMILY_CLIP: number[] = [];
for (let k = 0; k < SH_M; k += 1) {
  for (let c = 0; c < SH_NX; c += 1) {
    const x = SH_X[c] as number;
    const v = (SH_EXACT[c] as number) + (SH_SLOPES[k] as number + 10) * 0.08 * Math.sin(Math.PI * x);
    const clipped = (k === 6 && c === SH_NX - 1) || k === 7;
    SH_FAMILY.push(clipped ? 1e6 : v);
    SH_FAMILY_CLIP.push(clipped ? 1 : 0);
  }
}
const SH_RESIDUAL = [1e-12, 1e-8, 1e-4, 1e-2, 0.2, 1e-2, 1e-4, 1e-8, 1e-12];
const SH_ERROR = SH_RESIDUAL.map((v) => v * 1.01);

const SH_DEF: FrameDef = {
  viz: "shooting_scan",
  arrays: [
    ["x", SH_X, "x（打靶轨迹的积分网格）"],
    ["family", SH_FAMILY, "轨迹族", [SH_M, SH_NX]],
    ["family_clip", SH_FAMILY_CLIP, "轨迹截断掩码", [SH_M, SH_NX]],
    ["u_exact", SH_EXACT, "真解 u(x)"],
    ["slopes", SH_SLOPES, "本族初始斜率 s_k"],
    ["matched", Array.from({ length: SH_M }, () => 1e6), "右端匹配量（占位）"],
    ["residual", SH_RESIDUAL, "打靶残差 |u(b;s_k)-beta|"],
    ["residual_clip", Array.from({ length: SH_M }, () => 0), "残差截断掩码"],
    ["error", SH_ERROR, "每条轨迹与真解的最大偏差"],
    ["error_clip", Array.from({ length: SH_M }, () => 0), "误差截断掩码"],
    ["trace_x", [0, 1], "区间端点"],
  ],
  meta: {
    m: SH_M, span: 0.2, j: 0, node_index: 0, s_ref: -10, slope_node: -8,
    span_lo: -8, span_hi: -12, sensitivity: 1101.3233, digits_lost: 3.0419, converged: false,
    stagnated: true, iterations: 4, residual_ref: 4.21e-12, error_ref: 7.56e-11, n_steps: 800,
    n_family_clip: 42, n_errors_clip: 0, n_residual_clip: 0, n_blowup: 0, family_max_abs: 2202.65,
    error_max_raw: 2202.65, residual_max_raw: 2202.65, residual_min: 1e-12, residual_floor: 1.2227e-13,
    k: 10, L: 1, kL: 10, clip_fill: 1e6, roundoff_floor: 1.1102230246251565e-16, overflow: true,
  },
  strings: {
    problem: "decay",
    label: "衰减模态的边界层（打靶法在这里失效）",
    nonfinite_policy: "非有限值用 clip_fill 占位并配掩码",
    clip_fill: "1e+06",
    axis_note: "主轴是真解与轨迹族；副轴对数显示打靶残差 |F(s_k)|",
    failure_note: "衰减模态的舍入被 S 放大 e^{kL} 倍；增长模态相减不丢位数。",
    sensitivity_note: "实测灵敏度 S=1.1013e+03（变分方程），吃掉 3.04 位十进制",
    overflow: "有轨迹超上限：数组已换成 clip_fill 并配掩码",
    data_source: "backend",
    params_json: '{"L":1,"k":10,"m":9,"problem":"decay","span":0.2}',
  },
};

const CV_NFD = [26, 51, 90, 141, 205, 256];
const CV_FD = CV_NFD.map((n) => 0.00655644 * (26 / n) ** 2);
const CV_FDRES = CV_NFD.map((n) => 7.77e-13 * (26 / n) ** 2);
const CV_NC = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32];
const CV_CHEB = [0.2847, 0.09, 0.028, 0.008, 0.002, 4e-4, 8e-5, 1.5e-5, 3e-6, 7.5e-15, 7.5e-15, 7.5e-15];
const CV_INTERP = CV_CHEB.map((v, i) => Math.max(2.5e-15, v / (i < 8 ? 3 : 3.5)));
const CV_FITEXP = CV_NC.map((n) => Math.max(7.5e-15, 0.5327 * Math.exp(-2.0894 * (n - 4))));
const CV_FITPOW = CV_NC.map((n, i) => (i < 7 ? 4.15174 * (n / 4) ** -1.993 : 7.5e-15));

const CV_DEF: FrameDef = {
  viz: "convergence_rate",
  arrays: [
    ["n_fd", CV_NFD, "有限差分的区间数 n"],
    ["fd_error", CV_FD, "有限差分误差"],
    ["fd_error_interior", CV_FD.map((v) => v * 1.01), "有限差分内部误差"],
    ["fd_residual", CV_FDRES, "有限差分真实边界残差"],
    ["fd_residual_discrete", CV_FDRES.map((v) => v * 0.8), "有限差分离散边界行残差"],
    ["n_cheb", CV_NC, "Chebyshev 节点数 N"],
    ["cheb_error", CV_CHEB, "谱配置误差"],
    ["cheb_error_interior", CV_CHEB.map((v) => v * 1.02), "谱配置内部误差"],
    ["cheb_residual", CV_NC.map(() => 1.5e-14), "谱配置边界残差"],
    ["interp_error", CV_INTERP, "插值误差（负对照）"],
    ["fit_exponential", CV_FITEXP, "实测半对数拟合线"],
    ["fit_power", CV_FITPOW, "实测幂律拟合线"],
    ["floor", [7.549516567451064e-15], "实测舍入地板"],
    ["bc_true_fd", CV_FDRES, "有限差分真实边界残差（副轴用）"],
  ],
  meta: {
    bc_order: 2, nmax: 256, fd_window_lo: 26, fd_window_hi: 256, fd_p: 2.0002378, fd_r2: 0.99999999,
    fd_p_interior: 2.0002378, cheb_window_lo: 4, cheb_window_hi: 32, exp_fit_head: 7, exp_slope: -2.0894,
    exp_r2: 0.99491, exp_digits_per_node: 0.90741, power_p: 1.993, power_r2: 0.9997, interp_p: 3.0,
    interp_r2: 0.9999, floor: 7.549516567451064e-15, n_floor: 18, n_at_min: 24,
    error_min: 7.549516567451064e-15, cheb_error_first: 0.2847, cheb_error_last: 7.5e-15,
    fd_error_first: 0.00655644, fd_error_last: 6.75105e-5, n4_roundoff: 1.1641532182693481e-10,
    roundoff_floor: 1.1102230246251565e-16, overflow: false,
  },
  strings: {
    problem: "smooth", label: "解析解 e^x sin(pi x)（谱方法指数收敛）",
    nonfinite_policy: "非有限值不画", clip_fill: "1e+06",
    window_note: "差分窗口 n=26..256；谱窗口 N=4..16",
    rate_note: "实测：谱侧半对数斜率 -2.089（每节点 0.91 位）",
    floor_note: "舍入地板实测 7.55e-15，N>=18 起不再下降",
    bc_note: "离散行残差恒为舍入级，真实违反量才反映边界格式的阶",
    overflow: "本帧无截断：所有曲线取对数前已抬到 u/2 地板",
    data_source: "backend", params_json: '{"bc_order":"2","nmax":256,"problem":"smooth"}',
  },
};

const ND_N = 24;
const ND_NODES = Array.from({ length: ND_N + 1 }, (_, j) => Math.cos((j * Math.PI) / ND_N));
const ND_SPACING = ND_NODES.slice(0, -1).map((v, j) => Math.abs((ND_NODES[j + 1] as number) - v));
const ND_MID = ND_NODES.slice(0, -1).map((v, j) => 0.5 * (v + (ND_NODES[j + 1] as number)));
const ND_XF = Array.from({ length: 601 }, (_, i) => -0.998 + (1.996 * i) / 600);
const ND_RHO = ND_XF.map((x) => ND_N / (Math.PI * Math.sqrt(1 - x * x)));
const ND_ERR = ND_XF.map((x) => Math.max(5e-16, 1e-4 * Math.abs(Math.sin(3 * Math.PI * x))));
const ND_COLL = ND_XF.map((x) => Math.max(2e-6, 2e-2 * Math.abs(Math.sin(3 * Math.PI * x))));

const ND_DEF: FrameDef = {
  viz: "node_clustering",
  arrays: [
    ["nodes", ND_NODES, "Chebyshev-Lobatto 节点 x_j"],
    ["node_theta", ND_NODES.map((_, j) => (j * Math.PI) / ND_N), "theta_j"],
    ["spacing", ND_SPACING, "相邻间距"],
    ["spacing_mid", ND_MID, "间距中点"],
    ["density_fine", ND_RHO, "解析密度（细网格）"],
    ["density_nodes", ND_NODES.map((x) => ND_N / (Math.PI * Math.sqrt(Math.max(1e-12, 1 - x * x)))), "节点处密度"],
    ["x_fine", ND_XF, "细网格（不含端点）"],
    ["u_exact_fine", ND_XF.map((x) => Math.sin(Math.PI * x)), "真解"],
    ["u_interp_fine", ND_XF.map((x) => Math.sin(Math.PI * x)), "插值多项式"],
    ["error", ND_ERR, "插值误差"],
    ["collocation_error", ND_COLL, "配置解误差"],
    ["density_ratio_mark", [ND_RHO[0] as number, ND_RHO[300] as number], "端点/中点密度"],
  ],
  meta: {
    n: ND_N, n_nodes: ND_N + 1, k: 0, cluster_ratio: 15.257, ratio_over_n: 0.63571,
    ratio_over_n_theory: 0.63662, max_gap: Math.max(...ND_SPACING), min_gap: Math.min(...ND_SPACING),
    outer_nodes: 8, interp_error_max: Math.max(...ND_ERR), interp_error_mid: ND_ERR[300],
    collocation_error_max: Math.max(...ND_COLL), order_gap: -2.034, density_ratio: 2.29416,
    density_ratio_theory: 2.29416, cond_inf: 50712.9, roundoff_floor: 1.1102230246251565e-16,
    overflow: false,
  },
  strings: {
    problem: "abs3", label: "折点解 |x|^3（只 C2，退化成代数收敛）",
    nonfinite_policy: "非有限值不画", clip_fill: "1e+06",
    cluster_note: "最大/最小间距比 15.257（=0.6357·N，渐近常数 2/pi）",
    zero_note: "插值误差曲线在节点上恰好为 0，那不是精度证据",
    contrast_note: "插值误差峰值 7.76e-05，配置解误差 8.39e-03",
    overflow: "本帧无截断", data_source: "backend", params_json: '{"k":3,"n":24,"problem":"abs3"}',
  },
};

const DM_N = 8;
const DM_SIZE = DM_N + 1;
const DM_MATRIX: number[] = [];
for (let i = 0; i < DM_SIZE; i += 1) {
  for (let j = 0; j < DM_SIZE; j += 1) {
    DM_MATRIX.push(i === j ? ((2 * DM_N * DM_N + 1) / 6) * (i % 2 === 0 ? 1 : -1) : (Math.sin(i + j + 1) * (i - j)) / (i + j + 2));
  }
}
const DM_LOG = DM_MATRIX.map((v) => Math.log10(Math.max(Math.abs(v), 1.1102230246251565e-16)));
const DM_DIAG = Array.from({ length: DM_SIZE }, (_, i) => DM_MATRIX[i * DM_SIZE + i] as number);
const DM_NODES = Array.from({ length: DM_SIZE }, (_, j) => Math.cos((j * Math.PI) / DM_N));
const DM_SIGMA = [142.35, 80, 40, 20, 10, 5, 2.5, 1.1, 1.0466135227283007e-16];
const DM_DERR = [2.13e-10, 3.1e-10, 4.2e-10, 5.1e-10, 6.0e-10, 6.8e-10, 7.5e-10, 8.0e-10, 8.38858e-10];

const DM_DEF: FrameDef = {
  viz: "diff_matrix_structure",
  arrays: [
    ["matrix", DM_MATRIX, "微分矩阵（行主序）", [DM_SIZE, DM_SIZE]],
    ["log_abs_matrix", DM_LOG, "log10|matrix|", [DM_SIZE, DM_SIZE]],
    ["sign_matrix", DM_MATRIX.map((v) => (v > 0 ? 1 : v < 0 ? -1 : 0)), "符号", [DM_SIZE, DM_SIZE]],
    ["diagonal", DM_DIAG, "对角元"],
    ["diag_zero", DM_DIAG.map((v) => (Math.abs(v) <= 1e-12 ? 1 : 0)), "数值上为零的对角元掩码"],
    ["rowsum", DM_MATRIX.map(() => 1e-15), "行和"],
    ["residual", [1e-16, 2e-16, 4e-16, 8e-16, 1.2e-15, 2e-15, 3e-15, 3.8e-15, 4.4e-15], "行和残差"],
    ["sigma", DM_SIGMA, "奇异值（降序）"],
    ["sigma_index", DM_SIGMA.map((_, i) => i), "奇异值序号"],
    ["null_vector", DM_SIGMA.map(() => 1), "零空间向量"],
    ["error_derivative", DM_DERR, "谱微分逐点误差"],
    ["nodes", DM_NODES, "节点"],
  ],
  meta: {
    n: DM_N, size: DM_SIZE, norm_inf: 64, norm_inf_theory: 64, rowsum_residual: 4.440892098500626e-15,
    density: 1, d00: (2 * DM_N * DM_N + 1) / 6, d00_theory: (2 * DM_N * DM_N + 1) / 6,
    n_nonzero: 0, sigma_max: 142.35, sigma_min: 1.0466135227283007e-16,
    sigma_min_nonzero: 1.0953593389906768, kappa_naive: 1.36e18, kappa_nonzero: 129.96,
    d2_norm_inf: 1365.33, collocation_cond_inf: 140.2, derivative_error_max: 8.38858e-10,
    derivative_error_mid: 3.365996370519042e-10, n4_roundoff: 7.275957614183426e-12,
    roundoff_floor: 1.1102230246251565e-16, overflow: false,
  },
  strings: {
    which: "D", nonfinite_policy: "非有限值不画", clip_fill: "1e+06",
    singular_note: "D_N 奇异：朴素 kappa = 1.36e+18 是零空间噪声，挖掉后 kappa+ = 129.96",
    use_note: "微分矩阵不是用来解方程组的：配置矩阵 kappa 按 N^4.4 增长",
    roundoff_note: "谱微分逐点误差最大 8.39e-10；N^4·u = 7.3e-12 是预期地板",
    overflow: "本帧无截断", data_source: "backend", params_json: '{"n":16,"which":"D"}',
  },
};

const DEFS: Record<string, FrameDef> = {
  shooting_scan: SH_DEF,
  convergence_rate: CV_DEF,
  node_clustering: ND_DEF,
  diff_matrix_structure: DM_DEF,
};

// --------------------------------------------------------------------------- //
// 1. 四个 viz 的端点时间都过校验器
// --------------------------------------------------------------------------- //

for (const viz of Object.keys(DEFS)) {
  const tMax = manifest.vizzes.find((v) => v.id === viz)?.time.mode === "scrub" ? 8 : 0;
  const times = tMax > 0 ? [0, tMax / 2, tMax] : [0, 8];
  for (const t of times) {
    test(`${viz}: t=${t} 的 Scene 过 validateScene（规则 1/2/8 全绿）`, async () => {
      const scene = await sceneOf(viz, {}, { t });
      assert.deepEqual(validateScene(scene), []);
      assert.equal(scene.chapter, "3.4");
      assert.ok(scene.pitfalls.length > 0, "pitfalls 面板必须非空（规则 7）");
      assert.ok(scene.layers.length >= 3, "每个 viz 至少 3 个图层");
    });
  }
}

test("manifest 与登记表一致：4 个 viz 都有 model，未知 id 返回 null", () => {
  assert.deepEqual(Object.keys(models).sort(), [
    "convergence_rate",
    "diff_matrix_structure",
    "node_clustering",
    "shooting_scan",
  ]);
  for (const v of manifest.vizzes) assert.ok(modelFor(v.id), `manifest 里的 ${v.id} 没有 model`);
  assert.equal(modelFor("nope"), null);
  assert.equal(manifest.vizzes.length, 4);
});

test("autoscaled 与 manifest 一致（主轴/副轴的固定与否不许改口）", async () => {
  for (const spec of manifest.vizzes) {
    const scene = await sceneOf(spec.id);
    for (const ax of spec.axes) {
      const got = scene.axes.find((a) => a.id === ax.id);
      assert.ok(got, `${spec.id} 缺少 manifest 声明的轴 ${ax.id}`);
      assert.equal(got.autoscaled, ax.autoscaled, `${spec.id}.${ax.id} 的 autoscaled 与 manifest 不一致`);
    }
  }
});

// --------------------------------------------------------------------------- //
// 2. requiredAux 真的画在对应轴上（按数值认，不按 label）
// --------------------------------------------------------------------------- //

const AUX: Record<string, Record<string, readonly [string, string]>> = {
  shooting_scan: { error: ["aux", "error"] },
  convergence_rate: { error: ["main", "cheb_error"], residual: ["aux", "fd_residual"] },
  node_clustering: { error: ["aux", "error"], collocation_error: ["aux", "collocation_error"] },
  diff_matrix_structure: { residual: ["aux", "residual"], error_derivative: ["derivative", "error_derivative"] },
};

for (const [viz, need] of Object.entries(AUX)) {
  test(`${viz}: requiredAux 的每个名字都有一条真画出来的曲线`, async () => {
    const scene = await sceneOf(viz);
    const frame = await makeFrame(DEFS[viz] as FrameDef);
    for (const [aux, [axis, arrayName]] of Object.entries(need)) {
      const want = frame.array(arrayName);
      assert.ok(
        hasArrayValues(scene, axis, want as ArrayLike<number>, 1e-9),
        `${viz}.${aux}：轴 ${axis} 上没有画出 ${arrayName} 的数值`,
      );
    }
  });
}

// --------------------------------------------------------------------------- //
// 3. 各章特有不变式
// --------------------------------------------------------------------------- //

test("shooting_scan：截断占位 1e6 绝不进图层；残差与误差同轴；overflow 与截断都点名", async () => {
  const scene = await sceneOf("shooting_scan");
  const values = layerNumbers(scene);
  assert.ok(!values.includes(1e6), "被 family_clip 标出的 1e6 占位被画进了图层");
  assert.ok(!values.includes(1e6 * 1.01), "截断误差占位被画进了图层");
  assert.ok(scene.overflow.flag, "帧内有截断时 overflow.flag 必须为 true");
  assert.match(scene.overflow.note, /截断|溢出|clip/);
  assert.match(degradedOf(scene), /截断/);
  assert.match(badgesOf(scene), /kL = 10/);
  // 最小残差点（命中点）用散点标出。
  const scatters = layersOn(scene, "aux").filter((l): l is Extract<Layer, { kind: "scatter" }> => l.kind === "scatter");
  assert.ok(scatters.some((s) => Array.from(s.y as ArrayLike<number>).some((v) => Math.abs(v - 1e-12) < 1e-18)), "缺最小残差命中点");
});

test("shooting_scan：scene.t 取帧的 j（请求 t 只在缺 meta 时回落）", async () => {
  const at6 = await sceneOf("shooting_scan", { meta: { j: 6, node_index: 6 } }, { t: 0 });
  assert.equal(at6.t, 6);
  const noJ = await sceneOf("shooting_scan", { dropMeta: ["j"] }, { t: 3 });
  assert.equal(noJ.t, 3);
  assert.match(degradedOf(noJ), /j/);
});

test("convergence_rate：loglogfit 同时带有限实测斜率与理论斜率，且两轴都是 log", async () => {
  const scene = await sceneOf("convergence_rate");
  const fits = scene.layers.filter((l): l is Extract<Layer, { kind: "loglogfit" }> => l.kind === "loglogfit");
  assert.equal(fits.length, 1, "aux 轴上必须恰好有一条 loglogfit（规则 2）");
  const fit = fits[0]!;
  assert.ok(Number.isFinite(fit.measuredSlope) && Number.isFinite(fit.theorySlope));
  assert.ok(Math.abs(fit.measuredSlope - -2.0002378) < 1e-6, "实测斜率必须来自 meta.fd_p");
  assert.equal(fit.theorySlope, -2, "理论斜率必须来自 meta.bc_order");
  assert.ok(fit.seriesLabel.length > 0 && fit.theoryLabel.length > 0);
  const main = scene.axes.find((a) => a.id === "main")!;
  const aux = scene.axes.find((a) => a.id === "aux")!;
  assert.equal(main.yscale, "log");
  assert.equal(aux.xscale, "log");
  assert.equal(aux.yscale, "log");
});

test("convergence_rate：bc_order 缺 meta 时用请求参数并降级，理论斜率随之改变", async () => {
  const scene = await sceneOf("convergence_rate", { dropMeta: ["bc_order"] }, { params: { bc_order: 1 } });
  assert.match(degradedOf(scene), /bc_order/);
  const fit = scene.layers.find((l): l is Extract<Layer, { kind: "loglogfit" }> => l.kind === "loglogfit")!;
  assert.equal(fit.theorySlope, -1);
});

test("node_clustering：节点明显非均匀，均匀对照被明确标注", async () => {
  const scene = await sceneOf("node_clustering");
  const lines = layersOn(scene, "main").filter((l): l is Extract<Layer, { kind: "line" }> => l.kind === "line");
  const nodeSeries = lines.flatMap((l) => l.series).find((s) => s.y.length === ND_N + 1 && Math.abs((s.y[0] as number) - 1) < 1e-12);
  assert.ok(nodeSeries, "主面板上没有 Chebyshev 节点曲线");
  const uniformSeries = lines
    .flatMap((l) => l.series)
    .find((s) => s.y.length === ND_N + 1 && Math.abs((s.y[0] as number) + 1) < 1e-12);
  assert.ok(uniformSeries, "必须画出均匀节点对照（不能假装节点均匀）");
  let maxGap = 0;
  for (let j = 0; j <= ND_N; j += 1) maxGap = Math.max(maxGap, Math.abs((nodeSeries.y[j] as number) - (uniformSeries.y[j] as number)));
  assert.ok(maxGap > 0.1, `Chebyshev 节点与均匀对照的最大偏离只有 ${maxGap}，图上看不出聚集`);
  const spacing = ND_SPACING;
  assert.ok(Math.max(...spacing) / Math.min(...spacing) > 10, "端点/中心间距比必须 >10（真实的聚集度）");
  assert.match(badgesOf(scene), /间距比 15\.3/);
});

test("node_clustering：节点上的插值误差=0 画在地板，且副轴同时有插值与配置误差", async () => {
  const scene = await sceneOf("node_clustering");
  const scatters = layersOn(scene, "aux").filter((l): l is Extract<Layer, { kind: "scatter" }> => l.kind === "scatter");
  assert.ok(
    scatters.some((s) => s.x.length === ND_N + 1 && Array.from(s.y as ArrayLike<number>).every((v) => Math.abs(v - 1.1102230246251565e-16) < 1e-30)),
    "节点上的 0 误差必须画在地板上并标注",
  );
  assert.ok(hasArrayValues(scene, "aux", ND_ERR, 1e-9));
  assert.ok(hasArrayValues(scene, "aux", ND_COLL, 1e-9));
});

test("diff_matrix_structure：热力图长度 = (N+1)²，奇异值谱跨十几个数量级且 σmin+ 参考线在场", async () => {
  const scene = await sceneOf("diff_matrix_structure");
  const heat = scene.layers.find((l): l is Extract<Layer, { kind: "heat" }> => l.kind === "heat")!;
  assert.equal(Array.from(heat.values).length, DM_SIZE * DM_SIZE);
  assert.equal(heat.nx, DM_SIZE);
  assert.equal(heat.ny, DM_SIZE);
  const sigmas = scene.layers
    .filter((l): l is Extract<Layer, { kind: "line" }> => l.kind === "line" && l.axes === "aux")
    .flatMap((l) => l.series)
    .find((s) => s.y.length === DM_SIZE && Math.abs((s.y[0] as number) - 142.35) < 1e-9);
  assert.ok(sigmas, "副轴上没有奇异值谱");
  const decades = Math.log10(142.35) - Math.log10(1.0466135227283007e-16);
  assert.ok(decades > 17, "奇异值谱必须展示零空间噪声（跨 >17 个数量级）");
  const ref = scene.layers
    .filter((l): l is Extract<Layer, { kind: "line" }> => l.axes === "aux")
    .flatMap((l) => l.series)
    .some((s) => s.y.length === 2 && Math.abs((s.y[0] as number) - 1.0953593389906768) < 1e-12);
  assert.ok(ref, "缺 σmin+ 参考线，读者会把 1e-16 当成真实条件数");
  assert.match(badgesOf(scene), /κ\+=130/);
  assert.match(badgesOf(scene), /理论临界 24\.00/);
});

// --------------------------------------------------------------------------- //
// 4. 降级不静默 / 缺必需数组抛错
// --------------------------------------------------------------------------- //

test("缺可选数组或可选 meta → 不抛错，但 degraded 角标点名", async () => {
  const cases: Array<[string, Override, RegExp]> = [
    ["shooting_scan", { dropMeta: ["s_ref"] }, /s_ref/],
    ["convergence_rate", { drop: ["interp_error"] }, /interp_error/],
    ["node_clustering", { drop: ["density_fine"] }, /density_fine/],
    ["diff_matrix_structure", { drop: ["diagonal"] }, /diagonal/],
  ];
  for (const [viz, over, want] of cases) {
    const scene = await sceneOf(viz, over);
    assert.deepEqual(validateScene(scene), [], `${viz} 降级后仍要过校验器`);
    const degraded = degradedOf(scene);
    assert.ok(degraded.length > 0, `${viz} 降级后必须有 degraded 角标（不许静默）`);
    assert.ok(want.test(degraded) || /等 \d+ 项/.test(degraded), `${viz} 降级说明没点名 ${want}：${degraded}`);
  }
});

test("缺必需数组 → 抛 ModelInputError（不静默画空图）", async () => {
  for (const [viz, name] of [
    ["shooting_scan", "family"],
    ["convergence_rate", "n_cheb"],
    ["node_clustering", "nodes"],
    ["diff_matrix_structure", "residual"],
  ] as const) {
    await assert.rejects(async () => sceneOf(viz, { drop: [name] }), ModelInputError);
  }
});

test("pitfalls 为空 → 抛错（规则 7：防误导面板必须有内容）", async () => {
  await assert.rejects(async () => {
    const frame = await makeFrame({ ...SH_DEF, pitfalls: "" });
    (models.shooting_scan as (f: Frame, o: ModelOptions) => Scene)(frame, OPTS);
  }, ModelInputError);
});

test("time.mode=none 的三个 viz：请求 t 再大，scene.t 也恒为 0", async () => {
  for (const viz of ["convergence_rate", "node_clustering", "diff_matrix_structure"]) {
    const scene = await sceneOf(viz, {}, { t: 99 });
    assert.equal(scene.t, 0, `${viz} 偷用了请求 t`);
  }
});

// --------------------------------------------------------------------------- //
// 5. 窄画布（480×520）不崩，且明确说明哪些面板被放弃
// --------------------------------------------------------------------------- //

test("窄画布 480×520：四个 viz 都过校验器，轴矩形为正，degraded 点名", async () => {
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz, {}, { width: 480, height: 520 });
    assert.deepEqual(validateScene(scene), [], `${viz} 在 480×520 下不合法`);
    for (const ax of scene.axes) assert.ok(ax.rect[2] > 0 && ax.rect[3] > 0, `${viz}.${ax.id} 宽高必须为正`);
    assert.ok(degradedOf(scene).length > 0, `${viz} 放弃面板时必须点名降级`);
  }
});

test("中文文案里没有裸 ASCII 双引号（规则 8 的排版隐患）", async () => {
  for (const viz of Object.keys(DEFS)) {
    const scene = await sceneOf(viz);
    const bodies = [...scene.texts.map((t) => t.text), ...scene.badges.map((b) => b.text)];
    for (const body of bodies) assert.ok(!body.includes('"'), `${viz} 的文案含 ASCII 双引号：${body}`);
  }
});
