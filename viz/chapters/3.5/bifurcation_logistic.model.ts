/**
 * `bifurcation_logistic`（3.5 章）的 model：logistic 映射的分岔图 + 周期闭合残差 + 当前 r 的蛛网图。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | scatter | `main` | 混沌参数点（周期 0，灰） |
 * | 2 | scatter | `main` | 周期参数点（周期 > 0，蓝） |
 * | 3 | line | `main` | 6 条临界参数竖线（倍周期点 / 3 周期窗口起点 / 聚点） |
 * | 4 | line | `main` | 当前参数 r 的竖线 |
 * | 5 | scatter | `residual` | 每个参数的周期闭合残差（对数纵轴） |
 * | 6 | line | `residual` | 周期检测容差 1e-9（判据线）+ 地板 1e-18（占位说明） |
 * | 7 | line | `cobweb` | 映射曲线 f(x)=r·x(1−x) |
 * | 8 | line | `cobweb` | 对角线 y=x |
 * | 9 | line | `cobweb` | 蛛网折线（当前 r，从 x₀=0.4 迭代） |
 * | 10 | scatter | `cobweb` | 检测到的周期轨点（周期 0 时是 64 个迭代值，不是周期轨） |
 *
 * 三条刻意的决定
 * --------------
 * 1. **周期是检测出来的，不是画出来的**：点的颜色只区分"检测到闭合（周期>0）"与
 *    "在 max_period 内没闭合（周期 0）"，判据（|f^p(x)−x| ≤ 1e-9）与用到的
 *    `max_period` 都写在文本与角标里——把 max_period 调小，原本判为周期 32 的参数
 *    会掉进混沌桶，这件事必须能从画面上读出来。
 * 2. **副轴是"判据"，不是装饰**：周期点（≈1e-16，其中 0 被后端抬到地板 1e-18）与
 *    混沌点（≈1e-2）在同一个对数轴上分居两端；容差线 1e-9 画出来，是因为**地板与
 *    容差之间那条带是灰区**（近闭合但不是周期），不画这条线读者会把"贴着 1e-9"的
 *    点误当成"精确周期"。
 * 3. **蛛网图必须等比例**：`cobweb` 轴的 xlim/ylim 都是 [0,1] 且**像素宽度等于高度**
 *    （方图），否则对角线不是 45°，蛛网的"阶梯"会被拉歪——图形撒谎比不画更坏。
 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
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
  optionalArray,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  resolveTextOverlaps,
  sceneShell,
  segmentedSeries,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  pitfallsBand,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import { badgeUnitCap, stackTexts } from "./common.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.5";
const VIZ = "bifurcation_logistic";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "logistic 分岔图：拖动参数实时生成";

/**
 * `manifest.gen.ts` 里 `r.critical` 的**最紧**一条（= 3 周期窗口起点 1+√8）：帧 `critical.r` 缺失时的回落值。
 * 完整清单（3.0 / 3.4495 / 3.5441 / 3.5644 / 3.8284）会作为竖线画在主图上，来自帧数组 `critical_r`。
 */
const R_CRITICAL = 3.8284271247461903;
/** 周期检测容差：帧 `strings.period_protocol` 写明的判据 |f^p(x)−x| ≤ 1e-9。 */
const PERIOD_TOL = 1e-9;
/** 后端给残差数组的地板（`server/chapters/3.5.py::_floor(arr, 1e-18)`）；画出来并说明它是占位。 */
const RESIDUAL_FLOOR = 1e-18;
/** 主图纵轴：logistic 的迭代值恒在 [0,1]（固定轴，不随参数自适应）。 */
const MAIN_YLIM: [number, number] = [0, 1];
/** 蛛网图窗口：映射与蛛网都在 [0,1]²，方图（见文件头第 3 条）。 */
const COBWEB_LIM: [number, number] = [0, 1];

const SLOT = {
  source: 0,
  mainAxis: 1,
  auxAxis: 2,
  critical: 3,
  cobwebAxis: 4,
  overflow: 5,
  status: 6,
  degraded: 7,
} as const;

export const bifurcationLogisticModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const bifR = requireArray(frame, "bif_r");
  const bifX = requireArray(frame, "bif_x");
  const parGrid = requireArray(frame, "par_grid");
  const periodGrid = requireArray(frame, "period_grid");
  const residual = requireArray(frame, "residual_cycle");
  const mapX = requireArray(frame, "map_x");
  const mapY = requireArray(frame, "map_y");
  const cobwebX = requireArray(frame, "cobweb_x");
  const cobwebY = requireArray(frame, "cobweb_y");
  const pitfalls = requirePitfalls(frame);
  requireSameLength(bifR, bifX, "bif_r 与 bif_x");
  requireSameLength(parGrid, periodGrid, "par_grid 与 period_grid");
  requireSameLength(parGrid, residual, "par_grid 与 residual_cycle");
  requireSameLength(mapX, mapY, "map_x 与 map_y");
  requireSameLength(cobwebX, cobwebY, "cobweb_x 与 cobweb_y");
  const diagX = frame.has("return_map_x") ? requireArray(frame, "return_map_x") : undefined;
  const diagY = frame.has("return_map_y") ? requireArray(frame, "return_map_y") : undefined;
  const cycleX = optionalArray(frame, "cycle_x");
  const cycleY = optionalArray(frame, "cycle_y");

  // —— 可选字段（只影响标注的缺失走降级）——
  const degraded: string[] = [];
  const summary = paramSummary(frame, opts, [{ name: "max_period", label: "周期检测上限" }]);
  degraded.push(...summary.degraded);
  const nR = metaNumber(frame, "n_r");
  const nKeep = metaNumber(frame, "n_keep");
  const nTransient = metaNumber(frame, "n_transient");
  const maxPeriod = metaNumber(frame, "max_period");
  const nPoints = metaNumber(frame, "n_points");
  const nPeriodic = metaNumber(frame, "n_periodic");
  const nChaotic = metaNumber(frame, "n_chaotic");
  const dr = metaNumber(frame, "dr");
  const rNow = metaNumber(frame, "r_now");
  const periodNow = metaNumber(frame, "period_now");
  const residualNow = metaNumber(frame, "residual_now");
  const rAccum = metaNumber(frame, "r_accumulation");
  const residualMax = metaNumber(frame, "residual_max");
  if (nR === undefined || nKeep === undefined) degraded.push("帧 meta 缺 n_r/n_keep：不标参数网格与每参数点数（不猜）");
  if (maxPeriod === undefined) degraded.push("帧 meta 缺 max_period：不标周期检测上限（不猜，且它是本 viz 的关键口径）");
  if (rNow === undefined) degraded.push("帧 meta 缺 r_now：不画当前参数竖线（不猜）");
  if (dr === undefined) degraded.push("帧 meta 缺 dr：不标参数分辨率（不猜）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: metaNumber(frame, "t") ?? 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // ------------------------------------------------------------------------- //
  // 排版：左列「分岔图 + 残差」，右列「方蛛网图 + 说明」
  // ------------------------------------------------------------------------- //
  // headerExtra=18：轴的标题带画在 rect[1]−18，不留这 18px 它会落进角标区（角标最长到 headerBand−9）
const panels = columnsLayout(width, height, { share: 0.6, columnGap: 58, mainShare: 0.5, gap: 50, headerExtra: 18 });
  const rRange = finiteMinMax(parGrid) ?? [2.4, 4.0];
  const rlim: [number, number] = [Math.min(rRange[0], rRange[1]), Math.max(rRange[0], rRange[1])];
  if (!(rlim[1] > rlim[0])) degraded.push("par_grid 退化（只有一个 r）：横轴用 1-2-5 取整的退化范围画，判据线仍按数据画");

  scene.axes.push(
    makeAxis("main", panels.main, {
      xlim: rlim,
      ylim: MAIN_YLIM,
      xlabel: "",
      ylabel: "x_n",
      title: "分岔图：迭代值 vs 参数 r",
      autoscaled: false,
    }),
  );

  // —— 图层 1~2：把 44800 个点按"检测到周期 / 混沌"分两组着色（周期来自 period_grid）——
  const grouped = splitByPeriod(bifR, bifX, parGrid, periodGrid, nR, nKeep);
  if (grouped === undefined) {
    scene.layers.push(
      scatterLayer("main", bifR, bifX, { r: 0.9, color: pal.muted, label: "全部参数点（周期分组不可用）", alpha: 0.75 }),
    );
    degraded.push(
      `bif_r 的 ${bifR.length} 个点分不出"每个参数 64 个迭代值"的分组（n_r×n_keep 与点数不符）：主图不分色，只按同一种颜色画`,
    );
  } else {
    scene.layers.push(
      scatterLayer("main", grouped.chaoticR, grouped.chaoticX, { r: 0.9, color: pal.muted, label: `混沌参数点（周期 0，${grouped.nChaotic} 个参数）`, alpha: 0.8 }),
    );
    scene.layers.push(
      scatterLayer("main", grouped.periodicR, grouped.periodicX, { r: 0.9, color: pal.curves[0], label: `周期参数点（周期 > 0，${grouped.nPeriodic} 个参数）`, alpha: 0.8 }),
    );
  }

  // —— 图层 3：临界参数竖线（每条一段，不连线）——
  const critArr = frame.has("critical_r") ? requireArray(frame, "critical_r") : undefined;
  if (critArr !== undefined && critArr.length > 0) {
    const verticals = [];
    for (let i = 0; i < critArr.length; i += 1) {
      const r = critArr[i] as number;
      if (!Number.isFinite(r)) continue;
      verticals.push({ x: [r, r], y: MAIN_YLIM, color: pal.reference, width: 1, dash: [3, 3], label: undefined });
    }
    if (verticals.length > 0) scene.layers.push(lineLayer("main", verticals));
  } else {
    degraded.push("帧缺 critical_r：主图不画临界参数竖线（不猜位置）");
  }

  // —— 图层 4：当前参数竖线 ——
  if (rNow !== undefined) {
    scene.layers.push(
      lineLayer("main", [{ x: [rNow, rNow], y: MAIN_YLIM, color: pal.marker, width: 1.6 }]),
    );
  }

  // ------------------------------------------------------------------------- //
  // 副轴 `residual`：每个参数的闭合残差（对数），容差线与地板线就是判据
  // ------------------------------------------------------------------------- //
  const resPos = finitePositiveMinMax(concat(residual, Float64Array.from([PERIOD_TOL, RESIDUAL_FLOOR])));
  const resYlim: [number, number] = resPos === undefined ? [1e-18, 1] : niceLimits(resPos[0], resPos[1], 6, { log: true, pad: 0.04 });
  scene.axes.push(
    makeAxis("residual", panels.error, {
      xlim: rlim,
      ylim: resYlim,
      yscale: "log",
      xlabel: "参数 r",
      ylabel: "残差",
      title: "周期闭合残差 |f^p(x)−x|",
      autoscaled: true,
    }),
  );
  scene.layers.push(
    lineLayer("residual", [
      { x: rlim, y: [PERIOD_TOL, PERIOD_TOL], color: pal.marker, width: 1.2, dash: [4, 3], label: `周期判据容差 ${formatNumber(PERIOD_TOL, 2)}` },
      { x: rlim, y: [RESIDUAL_FLOOR, RESIDUAL_FLOOR], color: pal.muted, width: 1, dash: [1, 3], label: `地板 ${formatNumber(RESIDUAL_FLOOR, 2)}（= 浮点下的 0）` },
    ]),
  );
  scene.layers.push(
    scatterLayer("residual", parGrid, residual, { r: 1.2, color: pal.curves[2], label: "每个参数的闭合残差", alpha: 0.85 }),
  );

  // 残差带的点数（**从帧数组数出**的计数，不是统计量）
  let nAtFloor = 0;
  let nGrey = 0;
  for (let i = 0; i < residual.length; i += 1) {
    const v = residual[i] as number;
    if (!Number.isFinite(v)) continue;
    if (v <= RESIDUAL_FLOOR * (1 + 1e-12)) nAtFloor += 1;
    else if (v <= PERIOD_TOL) nGrey += 1;
  }

  // ------------------------------------------------------------------------- //
  // 右列 `cobweb`：方图（等比例 + 等像素跨度）
  // ------------------------------------------------------------------------- //
  const sideX = panels.side[0];
  const sideY = panels.side[1];
  const sideW = panels.side[2];
  // 方图让出下方说明区（方图必须等比，所以只能缩不能压）：说明放不下比图小一点严重得多
  const sideH = Math.max(170, Math.min(sideW, panels.side[3] - 200));
  const cobwebRect: Rect = [sideX, sideY, sideW, sideH];
  scene.axes.push(
    makeAxis("cobweb", cobwebRect, {
      xlim: COBWEB_LIM,
      ylim: COBWEB_LIM,
      xlabel: "x_n",
      ylabel: "x_n+1",
      title: "映射与蛛网图",
      autoscaled: false,
    }),
  );
  if (diagX !== undefined && diagY !== undefined) {
    scene.layers.push(lineLayer("cobweb", segmentedSeries(diagX, diagY, pal.reference, { width: 1, dash: [4, 4] })));
  } else {
    degraded.push("帧缺 return_map_x/return_map_y：蛛网图不画对角线（不猜）");
  }
  scene.layers.push(lineLayer("cobweb", segmentedSeries(cobwebX, cobwebY, pal.curves[0], { width: 1.1 })));
  scene.layers.push(lineLayer("cobweb", segmentedSeries(mapX, mapY, pal.curves[1], { width: 1.8 })));
  if (cycleX !== undefined && cycleY !== undefined && cycleX.length > 0 && cycleX.length === cycleY.length) {
    scene.layers.push(
      scatterLayer("cobweb", cycleX, cycleY, {
        r: 2.4,
        color: pal.marker,
      }),
    );
  } else {
    degraded.push("帧缺 cycle_x/cycle_y：蛛网图不标周期轨点（不猜）");
  }

  // ------------------------------------------------------------------------- //
  // 文本
  // ------------------------------------------------------------------------- //
  const mainLines = [
    "灰=混沌　蓝=周期　红=当前 r　灰虚=临界",
    `判据 |f^p−x|≤${formatNumber(PERIOD_TOL, 2)}，p≤${fmt(maxPeriod, 4)}；${fmt(nR, 6)}×${fmt(nKeep, 6)}（暂态 ${fmt(nTransient, 6)}，共 ${fmt(nPoints, 7)} 点）`,
  ];
  {
    // 分岔图左下角为空区（r<3.2 的轨道都在 x≈0.6 以上）：图例放这里，别压住分岔点云。
    const st = stackTexts(
      panels.main[0] + 6,
      panels.main[1] + panels.main[3] - 2 * 16 - 6,
      panels.main[2] * 0.98,
      panels.main[1] + panels.main[3] - 6,
      mainLines.map((body) => ({ body, color: pal.muted, size: 10 })),
    );
    scene.texts.push(...st.texts);
    if (st.dropped.length > 0) degraded.push(`主图图例放不下 ${st.dropped.length} 段：未显示`);
  }

  const resLines = [
    `红虚=周期检测容差 ${formatNumber(PERIOD_TOL, 2)}　细线=地板 ${formatNumber(RESIDUAL_FLOOR, 2)}（浮点 0）`,
    `周期 ${fmt(nPeriodic, 6)}/${fmt(nR, 6)}；地板 ${nAtFloor} 点；灰区：${nGrey} 个落在地板与容差之间（最大 ${fmt(residualMax, 3)}）`,
  ];
  {
    // 对数残差轴的中段是空的（周期点 ~1e-15、混沌点 ~1e-2）：图例放中段，避开两簇。
    const y0 = panels.error[1] + Math.round(panels.error[3] * 0.42);
    const st = stackTexts(
      panels.error[0] + 6,
      y0,
      panels.error[2] * 0.98,
      y0 + 3 * 16,
      resLines.map((body) => ({ body, color: pal.muted, size: 10 })),
    );
    scene.texts.push(...st.texts);
    if (st.dropped.length > 0) degraded.push(`残差轴图例放不下 ${st.dropped.length} 段：未显示`);
  }

  const cobwebLines = [
    `当前 r=${fmt(rNow, 4)}：周期 ${fmt(periodNow, 4)}，残差 ${fmt(residualNow, 3)}`,
    `周期 0 = max_period=${fmt(maxPeriod, 4)} 内未闭合（混沌）`,
    rAccum !== undefined ? `r∞≈${formatNumber(rAccum, 10)}（Feigenbaum 聚点）` : "r∞ 未给出",
    `Δr=${fmt(dr, 3)}：3 周期窗口仅约 5 个参数点（n_r=1200 才展开）`,
    "映射 f(x)=r·x(1−x)；灰虚 y=x",
  ];
  const cobwebTextTop = sideY + sideH + 30; // 让出方图自己的刻度带与轴名
  {
    const st = stackTexts(sideX, cobwebTextTop, sideW, panels.side[1] + panels.side[3] - 2, [
      { body: cobwebLines[0] as string, color: pal.ink, size: 10, role: "label" as const },
      ...cobwebLines.slice(1).map((body) => ({ body, color: pal.muted, size: 10, role: "note" as const })),
    ]);
    scene.texts.push(...st.texts);
    if (st.dropped.length > 0) degraded.push(`蛛网图下方的说明放不下 ${st.dropped.length} 段：未显示（口径见章节笔记）`);
  }

  // ------------------------------------------------------------------------- //
  // 角标（规则 4/5/6）
  // ------------------------------------------------------------------------- //
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.mainAxis, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "残差轴：自适应对数（按数据取整）", 8, 8), SLOT.auxAxis, width),
  );
  if (rNow !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("r", rNow, criticalOf(frame, "r", R_CRITICAL), { beyond: "above" }), SLOT.critical, width));
  }
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        clipBadgeText(
          `max_period=${fmt(maxPeriod, 4)}｜周期 ${fmt(nPeriodic, 6)} / 混沌 ${fmt(nChaotic, 6)}（调小上限会把周期 32 的推进混沌桶）`,
          badgeUnitCap(width),
        ),
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  const status = degradedBadge(degraded, SLOT.degraded, width);
  if (status !== undefined) scene.badges.push(status);

  requireDrawableLayers(scene);
  applyOverflow(scene, countNonFinite(bifX) + countNonFinite(residual) + countNonFinite(periodGrid), { slot: SLOT.overflow, width });

  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function fmt(v: number | undefined, sig: number): string {
  return v === undefined ? "?" : formatNumber(v, sig);
}

function concat(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const out = new Float64Array(a.length + b.length);
  for (let i = 0; i < a.length; i += 1) out[i] = a[i] as number;
  for (let i = 0; i < b.length; i += 1) out[a.length + i] = b[i] as number;
  return out;
}

/** 按"检测到周期 / 混沌"把点云分两组（点的周期 = `repeat(period_grid, n_keep)`，见帧 strings）。 */
function splitByPeriod(
  bifR: ArrayLike<number>,
  bifX: ArrayLike<number>,
  parGrid: ArrayLike<number>,
  periodGrid: ArrayLike<number>,
  nR: number | undefined,
  nKeep: number | undefined,
): { chaoticR: Float64Array; chaoticX: Float64Array; periodicR: Float64Array; periodicX: Float64Array; nPeriodic: number; nChaotic: number } | undefined {
  if (nR === undefined || nKeep === undefined || !(nR > 0) || !(nKeep > 0)) return undefined;
  if (parGrid.length !== nR || periodGrid.length !== nR) return undefined;
  if (bifR.length !== bifX.length || bifR.length % nR !== 0) return undefined;
  const keep = bifR.length / nR;
  if (keep !== Math.round(nKeep)) return undefined;

  const chaoticR: number[] = [];
  const chaoticX: number[] = [];
  const periodicR: number[] = [];
  const periodicX: number[] = [];
  let nChaotic = 0;
  let nPeriodic = 0;
  for (let j = 0; j < nR; j += 1) {
    const p = periodGrid[j] as number;
    const isChaotic = !(p > 0);
    if (isChaotic) nChaotic += 1;
    else nPeriodic += 1;
    const lo = j * keep;
    for (let k = 0; k < keep; k += 1) {
      const r = bifR[lo + k] as number;
      const x = bifX[lo + k] as number;
      if (isChaotic) {
        chaoticR.push(r);
        chaoticX.push(x);
      } else {
        periodicR.push(r);
        periodicX.push(x);
      }
    }
  }
  return {
    chaoticR: Float64Array.from(chaoticR),
    chaoticX: Float64Array.from(chaoticX),
    periodicR: Float64Array.from(periodicR),
    periodicX: Float64Array.from(periodicX),
    nPeriodic,
    nChaotic,
  };
}

