/**
 * `heat_ftcs`（0.0 章）的 model：一维热传导显式格式 u(x) 与 exact(x)、误差副轴、时空热力图。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 数值解 `u`（整段帧数组，非有限值处断开） |
 * | 2 | line | `main` | 精确解 `exact`（虚线 + 对比色，与 `u` 同轴，图例各自给标签） |
 * | 3 | line | `error` | `\|error\|` 逐点折线（对数纵轴；`=0` 的点对数轴画不出，单独说明） |
 * | 4 | heat | `xt` | `opts.history` 的 `u(x)` 拼成的时空热力图（**仅在可用帧 ≥ 2 时出现**） |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主面板轴模式 / 2 误差+时空轴自适应 / 3 `r` 临界 /
 * 4 溢出 / 5 前端预览（仅 `backend+preview` 时）/ 6 稳定性（`frame.strings.stability`）/
 * 7 降级说明（`⚠ 降级：…`，与稳定性分开挂，各自有宽度预算）。
 *
 * 三条刻意的决定
 * --------------
 * 1. **请求时刻 ≠ 到达时刻**：FTCS 的 `Δt = r·Δx²/α`、`n_steps = ⌈t_end/Δt⌉`，
 *    所以数组画的是 `meta.t_actual = n_steps·Δt` 时刻的解，而 `meta.t` 是请求时刻。
 *    两者相差超过 `dt/2` 时用 `text` 在图上**明说**（这是本 viz 的真陷阱，见 `frame.pitfalls`）；
 *    不超时**不出现这句话**——"只在需要时出现"正是它可信的原因。
 * 2. **主面板纵轴固定** `[-0.05, 1.05]`：初值 `sin(πx) ∈ [0,1]`、两端 Dirichlet 0，
 *    `r ≤ 1/2` 时离散极值原理保证解不越界；自适应纵轴会把衰减拉平，
 *    而 `r > 1/2` 时解会冲出画面——那时应当看到溢出角标（规则 1），而不是被缩轴藏起来。
 * 3. **时空热力图只由 `opts.history` 组成**，缺帧就**不画这一层**（不是画一层空的）；
 *    时间轴没攒够帧时，右侧改为一句说明"为什么没画"，避免读者以为那里本该有图。
 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import type { Frame } from "../../shared/src/frame.js";
import {
  resolveTextOverlaps,
  applyOverflow,
  canvasSize,
  clipBadgeText,
  columnsLayout,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finiteMinMax,
  finitePositiveMinMax,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  segmentedSeries,
  stringOf,
  toNumeric,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  pitfallsBand,
  textBoxOf,
  axisModeBadge,
  badge,
  criticalBadge,
  heatLayer,
  lineLayer,
  makeAxis,
  makeErrorAxis,
  previewBadge,
  sourceBadge,
  text,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "0.0";
const VIZ = "heat_ftcs";
const TITLE = manifest.vizzes.find((v) => v.id === VIZ)?.title ?? "一维热传导：显式格式与稳定阈值";
/** manifest 里 `r.critical = [0.5]`；`frame.critical.r` 缺失时的回落值。 */
const R_CRITICAL = 0.5;

/**
 * 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）：长文本角标（溢出、降级）各占一行左列，
 * 右侧留空，避免与邻居叠字。
 */
const SLOT = {
  source: 0,
  axisMode: 1,
  errorAxis: 2,
  critical: 3,
  overflow: 4,
  /** 仅当帧声明 `backend+preview` 时出现。 */
  preview: 5,
  stability: 6,
  degraded: 7,
} as const;
/** 主面板固定纵轴：初值 sin(πx) ∈ [0,1]（manifest theory）加 5% 余量，见文件头第 2 条。 */
const MAIN_YLIM: [number, number] = [-0.05, 1.05];

export const heatFtcsModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段 ——
  const x = requireArray(frame, "x");
  const u = requireArray(frame, "u");
  const exact = requireArray(frame, "exact");
  const error = requireArray(frame, "error");
  requireSameLength(x, u, "帧数组 x 与 u");
  requireSameLength(x, exact, "帧数组 x 与 exact");
  requireSameLength(x, error, "帧数组 x 与 error");
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段（缺了降级，不猜）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const dt = metaNumber(frame, "dt");
  const tActual = metaNumber(frame, "t_actual");
  const nSteps = metaNumber(frame, "n_steps");
  // 两条降级必须合并成**一条**：降级角标走 clipBadgeText（约 84 单位），拆成两条时
  // 后一条会被截掉——"帧里到底缺了什么"就说不全了（测试正好守这条：两个字段名都要在）。
  if (dt === undefined && tActual === undefined) {
    degraded.push("帧 meta 缺 dt 与 t_actual：不判断请求时刻是否到达、也不比对时刻（不假设相等）");
  } else if (dt === undefined) {
    degraded.push("帧 meta 缺 dt：不判断请求时刻是否到达");
  } else if (tActual === undefined) {
    degraded.push("帧 meta 缺 t_actual：不比对时刻（不假设相等）");
  }
  if (nSteps === undefined) degraded.push("帧 meta 缺 n_steps：不显示步数");
  const r = metaNumber(frame, "r");
  if (r === undefined) degraded.push("帧 meta 缺 r：不标 r 的临界值（不猜）");
  const stability = stringOf(frame, "stability");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const panels = columnsLayout(width, height, { share: 0.58 });
  const xRange = finiteMinMax(x) ?? [0, 1];
  const xlim = niceLimits(xRange[0], xRange[1], 7);

  scene.axes.push(
    makeAxis("main", panels.main, {
      xlim,
      ylim: MAIN_YLIM,
      xlabel: "",
      ylabel: "温度 $u$",
      title: "$u(x,t)$",
      autoscaled: false,
    }),
  );

  // 误差副轴：对数纵轴，只由 |error| 的正有限值定范围（规则 3）。
  const absError = absolute(error);
  const errPositive = finitePositiveMinMax(absError);
  const errorYlim: [number, number] = errPositive === undefined ? [1e-16, 1] : niceLimits(errPositive[0], errPositive[1], 6, { log: true, pad: 0.2 });
  scene.axes.push(
    makeErrorAxis("error", panels.error, {
      xlim,
      ylim: errorYlim,
      xlabel: "$x$",
      ylabel: "误差绝对值",
      title: "误差 $u-u_{\\mathrm{exact}}$",
      autoscaled: true,
    }),
  );

  // —— 图层 1~2：u(x) 与 exact(x) 同轴 ——
  const uSeries = segmentedSeries(x, u, pal.curves[0], { width: 1.8, label: "数值解 $u$（FTCS）" });
  const exactSeries = segmentedSeries(x, exact, pal.exact, { width: 1.4, dash: [6, 4], label: "精确解 $u_exact$" });
  if (uSeries.length + exactSeries.length > 0) scene.layers.push(lineLayer("main", [...uSeries, ...exactSeries]));
  else degraded.push(`帧数组 u/exact 没有可绘的有限值（${x.length} 个样本全为非有限）：主面板不画曲线`);

  // —— 图层 3：误差（逐点，对数轴；0 与 NaN 都不画，并在文本里报数）——
  const errSeries = segmentedSeries(x, absError, pal.residual, { width: 1, label: "$|u−u_exact|$（逐点）", accept: (v) => v > 0 });
  if (errSeries.length > 0) scene.layers.push(lineLayer("error", errSeries));
  else degraded.push(`帧 error 没有 > 0 的有限值（${x.length} 个样本）：对数轴上画不出任何点`);

  // —— 图层 4：时空热力图（只有 history 可用帧 ≥ 2 才画）——
  const grid = buildSpacetime(opts.history, x);
  if (grid !== undefined) {
    scene.axes.push(
      makeAxis("xt", panels.side, {
        xlim,
        ylim: [grid.extent[2], grid.extent[3]],
        xlabel: "$x$",
        ylabel: "$t$",
        title: `u(x,t) 时空（history ${grid.ny} 帧）`,
        autoscaled: true,
      }),
    );
    scene.layers.push(
      heatLayer("xt", grid.values, grid.nx, grid.ny, grid.extent, { cmap: "viridis", label: "$u(x,t)$（history 帧，行=时间）" }),
    );
  } else {
    const why =
      opts.history === undefined || opts.history.length === 0
        ? "opts.history 为空：没有历史帧，不画时空热力图（不画空层）"
        : `opts.history 有 ${opts.history.length} 帧，可用（同空间网格、时间递增）不足 2 帧：不画时空热力图`;
    scene.texts.push(
      text(panels.side[0], panels.side[1] + 6, `时空热力图未绘制——${why}`, {
        role: "note",
        size: 11,
        color: pal.warn,
        maxWidth: panels.side[2],
      }),
    );
  }

  // —— 文本 ——
  // `r` 在帧 meta 里（后端写入了），`nx` 不在——但 nx 就是帧数组 x 的长度，直接数出来即可，
  // 不必去请求参数里"借"一个值，也就不会产生一条名不副实的"降级"。
  const summary = paramSummary(frame, opts, [{ name: "r", label: "$r$" }]);
  degraded.push(...summary.degraded);
  const extras: string[] = [`nx=${x.length}（帧 x 数组长度）`];
  if (dt !== undefined) extras.push(`dt=${formatNumber(dt, 3)}`);
  if (nSteps !== undefined) extras.push(`n_steps=${formatNumber(nSteps, 6)}`);
  scene.texts.push(
    text(panels.main[0] + 6, panels.main[1] + 4, [summary.text, ...extras].join("　"), {
      role: "label",
      size: 11,
      color: pal.muted,
      maxWidth: panels.main[2] * 0.8,
    }),
  );

  // 请求时刻 vs 到达时刻：**只在真的不等时**出现（超过 dt/2）。
  const mismatch = tActual !== undefined && dt !== undefined && Math.abs(tActual - tNow) > dt / 2;
  if (mismatch) {
    // 位置按**上一块的真实盒子**算，不写死 +26：帧里的参数说明一变长就会多折一行，
    // 写死的偏移会把这条"请求时刻 ≠ 实际到达时刻"的警告压住（规则 8 的文本重叠当场报过）。
    const summaryBox = scene.texts[scene.texts.length - 1];
    const warnY = summaryBox === undefined
      ? panels.main[1] + 26
      : textBoxOf({ ...summaryBox, y: summaryBox.y ?? 0 }).y + textBoxOf({ ...summaryBox, y: summaryBox.y ?? 0 }).h + 4;
    scene.texts.push(
      text(
        panels.main[0] + 6,
        warnY,
        `⚠ 请求 t=${formatNumber(tNow, 4)} ≠ 实际到达 t_actual=${formatNumber(tActual, 4)}` +
          `（相差 ${formatNumber(Math.abs(tActual - tNow), 3)} > dt/2=${formatNumber(dt / 2, 3)}）：` +
          `曲线是 t_actual 时刻的解，不是请求时刻的。`,
        { role: "note", size: 11, color: pal.warn, maxWidth: panels.main[2] * 0.95 },
      ),
    );
  }

  const nBadError = countNonFinite(error);
  const nZeroError = countZero(error);
  scene.texts.push(
    text(
      panels.error[0] + 6,
      panels.error[1] + 4,
      `|误差| ${error.length} 点；${nZeroError} 点为 0（对数轴画不出）、${nBadError} 个非有限`,
      { role: "note", size: 11, color: nZeroError + nBadError > 0 ? pal.warn : pal.muted, maxWidth: panels.error[2] * 0.9 },
    ),
  );

  // —— 角标 ——
  // 规则 5：恒挂 source；预览型数据源另挂 preview（kind 各表一意，见 scene.ts）。
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "误差轴/时空轴：自适应（按本帧数据取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.errorAxis,
      width,
    ),
  );
  if (r !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("r", r, criticalOf(frame, "r", R_CRITICAL), { beyond: "above" }), SLOT.critical, width));
  }
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        clipBadgeText(stability === undefined ? "帧 strings 缺 stability：不自行判断稳定性" : `稳定性：${stability}（帧 strings）`),
        8,
        8,
      ),
      SLOT.stability,
      width,
    ),
  );
  const status = degradedBadge(degraded, SLOT.degraded, width);
  if (status !== undefined) scene.badges.push(status);

  // 规则 1：越界 + 非有限值（u / exact / error 一起算）。
  requireDrawableLayers(scene);
  applyOverflow(scene, countNonFinite(u) + countNonFinite(exact) + nBadError, { slot: SLOT.overflow, width });
  // 文本块之间不许重叠（规则 8，两侧校验器都查）：模型里各块按自己的行数排位，
  // 帧里的说明一变长就会多折一行压住下一块（fixture 里 t=0.5 的标记真压过参数行 49×8px）。
  // 底线取底部 pitfalls 面板的顶边（面板高度随 pitfalls 行数变，小画布上它会占掉大半屏）：
  // 第四个参数是**高度**（`textBoxOf` 口径），不是底线。
  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);

  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** `|error|`（副轴按 manifest 声明画的是"误差绝对值"）；非有限值原样保留，交给分段逻辑断开。 */
function absolute(a: ArrayLike<number>): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i += 1) out[i] = Math.abs(a[i] as number);
  return out;
}

/** 有限的 0 的个数（对数轴上"画不出"，与 NaN 不同，要分开报数）。 */
function countZero(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if (a[i] === 0) n += 1;
  return n;
}

/** 时空网格：行主序 `values[j*nx + i]`（行=时间、列=空间），与 `heatLayer` 的采样约定一致。 */
interface SpacetimeGrid {
  values: Float64Array;
  nx: number;
  ny: number;
  extent: [number, number, number, number];
}

/**
 * 用 `opts.history` 拼时空热力图。
 *
 * 返回 `undefined` 的情形（都**不画**该层，而不是画一层空的）：
 * 可用帧 < 2、时间不严格递增、空间网格与当前帧不一致、`nx < 2`。
 */
function buildSpacetime(history: readonly { t: number; frame: Frame }[] | undefined, x: ArrayLike<number>): SpacetimeGrid | undefined {
  const nx = x.length;
  if (history === undefined || history.length < 2 || nx < 2) return undefined;
  const x0 = x[0] as number;
  const x1 = x[nx - 1] as number;
  if (!(x1 > x0)) return undefined;

  const usable = history
    .filter((h) => Number.isFinite(h.t) && h.frame.has("u") && h.frame.has("x") && sameDomain(toNumeric(h.frame.array("x")), x))
    .slice()
    .sort((a, b) => a.t - b.t);
  if (usable.length < 2) return undefined;
  const t0 = usable[0].t;
  const t1 = usable[usable.length - 1].t;
  if (!(t1 > t0)) return undefined;

  const ny = usable.length;
  const values = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j += 1) {
    const arr = toNumeric(usable[j].frame.array("u"));
    for (let i = 0; i < nx; i += 1) values[j * nx + i] = arr[i] as number; // 非有限值原样保留（热力图会画成灰格）
  }
  return { values, nx, ny, extent: [x0, x1, t0, t1] };
}

/** 空间网格是否与当前帧一致（节点数 + 两端点；不一致的帧混进来会画错位）。 */
function sameDomain(other: ArrayLike<number>, x: ArrayLike<number>): boolean {
  if (other.length !== x.length) return false;
  const last = x.length - 1;
  const tol = 1e-9 * Math.max(1, Math.abs(x[0] as number), Math.abs(x[last] as number));
  return Math.abs((other[0] as number) - (x[0] as number)) <= tol && Math.abs((other[last] as number) - (x[last] as number)) <= tol;
}
