/**
 * `equioscillation`（2.3 章）的 model：Remez 逐轮误差曲线 + 交错点 + 与 L² 投影的对照。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | band | `main` | ±max\|e\|（收敛值）的等幅带：交错定理要求误差落在这两条线之间 |
 * | 2 | line | `main` | 零线（读数基准，浅色） |
 * | 3 | line | `main` | `error_final`（灰虚线：收敛后的误差曲线） |
 * | 4 | line | `main` | `error_l2`（紫点划线：同阶 **L² 投影**误差，对照） |
 * | 5 | line | `main` | `error`（蓝实线：第 k 轮的误差曲线，带符号） |
 * | 6 | scatter | `main` | `alt_x` / `alt_error`：第 k 轮的交错点 |
 * | 7 | line | `alt` | `abs_error` 第 2 行（最终 \|e\| 包络） |
 * | 8 | line | `alt` | 舍入地板 u（仅当确实有点触底时才画，见文件头第 3 条） |
 * | 9 | scatter | `alt` | 第 k 轮交错点的 \|e\|（在地板之上的点） |
 * | 10 | scatter | `alt` | 第 k 轮交错点里 \|e\| ≤ u 的那些（画在地板上并单独标注） |
 * | 11 | scatter | `alt` | 最终交错点的 \|e\| |
 *
 * 角标槽位（固定，见 `common.ts::SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 /
 * 3 定理参照（交错点应为 deg+2）/ 4 状态（证书、振幅比、地板）/ 5 前端预览 /
 * 6 溢出 / 7 降级。
 *
 * 这个 viz 最容易骗人的两处，以及它们在图上的防法
 * ------------------------------------------------
 * 1. **「交错点数 = n+2」被当成收敛判据**。它只是定理的必要条件：误差掉到舍入地板以后
 *    极值全是噪声，"数出 n+2 个等幅点"没有内容（pitfalls §8）。防法：`alt` 面板把
 *    **振幅比**（= max\|e\|/min\|e\|，等于 1 才是等幅）、**converged**、**alt_signs_alternate**、
 *    **clipped_points** 四个量全写出来；`amplitude_ratio_*` 为 `null` 时**照原样**写
 *    「无有限值（null）」并解释它的含义（某个交错点的 e 恰好为 0 → 比值是 ∞，
 *    **不是** 1、也不是 0），不拿 1 或 0 顶替；`alt_signs_alternate=false`、
 *    `converged=false` 时同样逐条说明，并把帧自带的 `converged_note` 原文照登。
 * 2. **把「最小二乘」当成「最小最大」**。`main` 面板同时画 L² 投影的误差曲线（紫点划线）：
 *    它与 Remez 误差（蓝实线）形状完全不同，端点附近明显更大。防法：文字里给出
 *    `max_error_final`、`max_error_l2` 与 `l2_over_minimax` 三个实测数（pole, n=8 时
 *    9.81e-2 / 1.50e-1 / 1.52），并写明"两个最优在最小化不同的范数，没有谁更正确"。
 *
 * 几条刻意的决定
 * --------------
 * 1. **交错点的 |e| 用对数轴（manifest 声明 `alt.scale="log"`），而 0 画不到**：帧里
 *    `alt_error` 是**带符号且未截断**的，初始轮次真的会出现 e ≡ 0 的点。这类点按
 *    地板 u 画成另一种颜色的标记并**报数**——绝不静默丢掉，也不用一条直线跨过去。
 * 2. **±max\|e\| 用 band 而不是两条线**：band 把"等幅"这件事画成一个区域，交错点是否
 *    贴在这条带的上下边缘一眼可见（这才是交错定理的内容）。
 * 3. **地板横线只在有点触底时才画**：收敛轮次里没有任何点触底，这时画一条 1e-16 的横线
 *    会把纵轴拉下十几个数量级、把 10 个等幅点压成一条线，反而看不出"等幅"。
 *    触底点数始终写在文字里（`meta.clipped_points`）。
 * 4. **时间轴是 Remez 轮次**，不是时间：以后端 `meta.step` 为准（`n_steps` 已收敛时，
 *    请求 t 再大也被夹住），夹取与不一致都写在图上。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import type { Frame } from "../../shared/src/frame.js";
import {
  applyOverflow,
  canvasSize,
  dataSourceOf,
  degradedBadge,
  metaNumber,
  palette,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  segmentedSeries,
  stackedPanels,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  bandLayer,
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  CHAPTER,
  SLOT,
  absolute,
  axisTitle,
  axisYlabel,
  countAtOrBelow,
  currentStep,
  fmt,
  fmtOpt,
  minMaxOfAll,
  noteOf,
  sci,
  specTitle,
  textBlock,
  unitBudget,
} from "./common.js";

const VIZ = "equioscillation";
const TITLE = specTitle(VIZ, "交错点：Remez 的等幅振荡");
/** 时间轴上限（manifest `time.tMax`）：只在帧缺 `n_steps` 时用它兜底。 */
const T_MAX = 24;

/** 把 `abs_error` 的第 `row` 行取出来（行主序、`rows × cols`）；形状不对时返回 undefined（降级）。 */
function matrixRow(frame: Frame, name: string, row: number): Float64Array | undefined {
  if (!frame.has(name)) return undefined;
  const shape = frame.shapeOf(name);
  if (shape.length !== 2 || shape[0] <= row || shape[1] <= 0) return undefined;
  const values = frame.array(name);
  const cols = shape[1] as number;
  const out = new Float64Array(cols);
  for (let i = 0; i < cols; i += 1) out[i] = values[row * cols + i] as number;
  return out;
}

export const equioscillationModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const x = requireArray(frame, "x");
  const error = requireArray(frame, "error");
  const errorFinal = requireArray(frame, "error_final");
  const errorL2 = requireArray(frame, "error_l2");
  const altX = requireArray(frame, "alt_x");
  const altError = requireArray(frame, "alt_error");
  const altXFinal = requireArray(frame, "alt_x_final");
  const altErrorFinal = requireArray(frame, "alt_error_final");
  requireSameLength(x, error, "帧数组 x 与 error");
  requireSameLength(x, errorFinal, "帧数组 x 与 error_final");
  requireSameLength(x, errorL2, "帧数组 x 与 error_l2");
  requireSameLength(altX, altError, "帧数组 alt_x 与 alt_error");
  requireSameLength(altXFinal, altErrorFinal, "帧数组 alt_x_final 与 alt_error_final");
  const pitfalls = requirePitfalls(frame);

  // —— 可选/降级字段 ——
  const degraded: string[] = [];
  const deg = metaNumber(frame, "deg");
  const nSteps = metaNumber(frame, "n_steps");
  const nAltExpected = metaNumber(frame, "n_alt_expected");
  const nAltCurrent = metaNumber(frame, "n_alt_current");
  const nAltFinal = metaNumber(frame, "n_alt_final");
  const ratioCurrent = metaNumber(frame, "amplitude_ratio_current");
  const ratioFinal = metaNumber(frame, "amplitude_ratio_final");
  const maxErrorFinal = metaNumber(frame, "max_error_final");
  const maxErrorL2 = metaNumber(frame, "max_error_l2");
  const l2OverMinimax = metaNumber(frame, "l2_over_minimax");
  const errorFloor = metaNumber(frame, "error_floor");
  const clippedPoints = metaNumber(frame, "clipped_points");
  const iterations = metaNumber(frame, "iterations");
  const converged = frame.meta["converged"];
  const altAlternate = frame.meta["alt_signs_alternate"];
  if (deg === undefined) degraded.push("帧 meta 缺 deg：次数留「?」，不按交错点数反推");
  if (maxErrorFinal === undefined) degraded.push("帧 meta 缺 max_error_final：等幅带的半宽无法确定，故不画 band");
  if (errorFloor === undefined) degraded.push("帧 meta 缺 error_floor：地板位置留「?」，对数轴下界改用数据下界");
  if (clippedPoints === undefined) degraded.push("帧 meta 缺 clipped_points：不声称有多少点被截断到地板");
  if (typeof converged !== "boolean") degraded.push("帧 meta 缺 converged：不下「拿到最优性证书」的结论");
  if (typeof altAlternate !== "boolean") degraded.push("帧 meta 缺 alt_signs_alternate：不声称交错点是否符号交替");
  // `amplitude_ratio_*` 的 `null` 是**如实标注**（比值为 ∞），不是缺失——单独走文本分支，不进降级列表。
  const funcName = stringOf(frame, "func");
  const objective = stringOf(frame, "objective");
  const contrast = stringOf(frame, "contrast");
  const convergedNote = noteOf(frame, "converged_note");
  const certificateNote = noteOf(frame, "certificate_note");
  const floorNote = noteOf(frame, "floor_note");

  const { step, notes: stepNotes } = currentStep(frame, opts, {
    lo: 1,
    hi: Math.max(1, Math.round(nSteps ?? T_MAX)),
    label: "Remez 轮次 k",
  });
  const k = step ?? 1;

  // —— 面板 ——
  const base = stackedPanels(width, height, { mainShare: 0.48, gap: 50 });

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: metaNumber(frame, "t") ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 主轴：带符号误差曲线（band = ±max|e|） ——
  const xRange = minMaxOfAll([x]) ?? [-1, 1];
  const xlim = niceLimits(xRange[0], xRange[1], 7, { pad: 0.02 });
  const envelope = maxErrorFinal;
  const bandX = new Float64Array([xlim[0], xlim[1]]);
  const yMainArrays = [error, errorFinal, errorL2, altError];
  const yMainRaw = minMaxOfAll(yMainArrays) ?? [-1, 1];
  const yMainLo = envelope === undefined ? yMainRaw[0] : Math.min(yMainRaw[0], -envelope);
  const yMainHi = envelope === undefined ? yMainRaw[1] : Math.max(yMainRaw[1], envelope);
  const ylimMain = niceLimits(yMainLo, yMainHi, 5, { pad: 0.08 });
  scene.axes.push(
    makeAxis("main", base.main, {
      xlim,
      ylim: ylimMain,
      xlabel: "x",
      ylabel: axisYlabel(VIZ, "main", "e(x)"),
      title: axisTitle(VIZ, "main", "误差曲线 e(x) = f − p"),
      autoscaled: false,
      xticks: niceTicks(xlim[0], xlim[1], 7),
      yticks: niceTicks(ylimMain[0], ylimMain[1], 5),
    }),
  );
  if (envelope !== undefined) {
    scene.layers.push(
      bandLayer("main", {
        x: bandX,
        ylo: new Float64Array([-envelope, -envelope]),
        yhi: new Float64Array([envelope, envelope]),
        color: pal.warn,
        alpha: 0.1,
        label: `±max|e| = ${sci(envelope)}（收敛值）`,
      }),
    );
  }
  scene.layers.push(lineLayer("main", segmentedSeries(bandX, new Float64Array([0, 0]), pal.cursor, { width: 1, label: "零线" })));
  scene.layers.push(lineLayer("main", segmentedSeries(x, errorFinal, pal.reference, { width: 1.4, dash: [6, 4], label: "收敛后的 e(x)" })));
  scene.layers.push(lineLayer("main", segmentedSeries(x, errorL2, pal.residual, { width: 1.3, dash: [2, 3], label: "同阶 L² 投影误差（对照）" })));
  scene.layers.push(lineLayer("main", segmentedSeries(x, error, pal.curves[0], { width: 1.9, label: `第 ${k} 轮的 e(x)` })));
  scene.layers.push(scatterLayer("main", altX, altError, { r: 4, color: pal.marker, label: "第 k 轮交错点" }));

  // —— 副轴：交错点上的 |e|（对数轴；0 与低于地板的点画在地板上并报数） ——
  const absAltCurrent = absolute(altError);
  const absAltFinal = absolute(altErrorFinal);
  const envelopeFinal = matrixRow(frame, "abs_error", 1);
  if (envelopeFinal === undefined) degraded.push("帧数组 abs_error 不是 2 维 3×N：副轴不画 |e| 包络，只画交错点");
  const floor = errorFloor ?? 1e-16;
  const belowFloor = countAtOrBelow(absAltCurrent, floor);
  const aboveX: number[] = [];
  const aboveY: number[] = [];
  const belowX: number[] = [];
  const belowY: number[] = [];
  for (let i = 0; i < absAltCurrent.length; i += 1) {
    const v = absAltCurrent[i] as number;
    const xv = altX[i] as number;
    if (!Number.isFinite(v) || !Number.isFinite(xv)) continue;
    if (v <= floor) {
      belowX.push(xv);
      belowY.push(floor);
    } else {
      aboveX.push(xv);
      aboveY.push(v);
    }
  }
  const finalBelow = countAtOrBelow(absAltFinal, floor);
  const altXFinalDraw: number[] = [];
  const altYFinalDraw: number[] = [];
  for (let i = 0; i < absAltFinal.length; i += 1) {
    const v = absAltFinal[i] as number;
    const xv = altXFinal[i] as number;
    if (!Number.isFinite(v) || !Number.isFinite(xv)) continue;
    altXFinalDraw.push(xv);
    altYFinalDraw.push(Math.max(floor, v));
  }
  const floorDrawn = belowFloor > 0 || finalBelow > 0 || (clippedPoints !== undefined && clippedPoints > 0);
  const yAltCandidates: ArrayLike<number>[] = [absAltFinal];
  if (envelopeFinal !== undefined) yAltCandidates.push(envelopeFinal);
  if (aboveY.length > 0) yAltCandidates.push(Float64Array.from(aboveY));
  if (floorDrawn) yAltCandidates.push(new Float64Array([floor]));
  const yAltRaw = minMaxOfAll(yAltCandidates) ?? [floor, 1];
  const ylimAlt = niceLimits(yAltRaw[0] > 0 ? yAltRaw[0] : floor, yAltRaw[1], 4, { log: true, pad: 0 });
  scene.axes.push(
    makeAxis("alt", base.error, {
      xlim,
      ylim: ylimAlt,
      yscale: "log",
      xlabel: "x",
      ylabel: axisYlabel(VIZ, "alt", "|e|"),
      title: axisTitle(VIZ, "alt", "交错点上的 |e|"),
      autoscaled: true,
      xticks: niceTicks(xlim[0], xlim[1], 6),
      yticks: niceTicks(ylimAlt[0], ylimAlt[1], 5, { log: true }),
    }),
  );
  if (envelopeFinal !== undefined) {
    scene.layers.push(lineLayer("alt", segmentedSeries(x, envelopeFinal, pal.reference, { width: 1.2, dash: [6, 4], label: "最终 |e(x)| 包络", accept: (v) => v > 0 })));
  }
  if (floorDrawn) {
    scene.layers.push(
      lineLayer("alt", segmentedSeries(bandX, new Float64Array([floor, floor]), pal.warn, { width: 1.2, dash: [2, 3], label: `舍入地板 u = ${sci(floor)}` })),
    );
  }
  if (aboveX.length > 0) {
    scene.layers.push(scatterLayer("alt", aboveX, aboveY, { r: 3.6, color: pal.curves[0], label: `第 ${k} 轮交错点 |e|` }));
  }
  if (belowX.length > 0) {
    scene.layers.push(
      scatterLayer("alt", belowX, belowY, { r: 3.6, color: pal.marker, label: `|e| ≤ u 的点（${belowX.length} 个，画在地板上）` }),
    );
  }
  scene.layers.push(
    scatterLayer("alt", altXFinalDraw, altYFinalDraw, {
      r: 3,
      color: pal.exact,
      label: finalBelow > 0 ? `最终交错点 |e|（其中 ${finalBelow} 个 ≤ u，画在地板上）` : "最终交错点 |e|",
    }),
  );

  // —— 文本 ——
  const mainLines: string[] = [
    `f = ${funcName ?? "?"}　次数 deg = ${fmtOpt(deg, 3)}　目标：${objective ?? "?"}　` +
      `时间轴 = Remez 轮次 k = ${k} / ${fmtOpt(nSteps, 3)}（meta.step）`,
    `蓝实线 = 第 k 轮误差 e(x)（带符号）　灰虚线 = 收敛后的 e(x)　紫点划线 = ${contrast ?? "同阶 L² 投影误差（对照）"}`,
    envelope === undefined
      ? "帧未给 max_error_final：等幅带（±max|e|）不画（不猜一个宽度出来）"
      : `红带 = ±max|e| = ${sci(envelope)}（收敛值）：交错定理要求误差在 n+2 个点上贴住带子的上下边缘、符号交替`,
    convergedNote === undefined ? "帧 strings 缺 converged_note：停机结论按 meta.converged 显示" : `停机：${convergedNote}`,
  ];
  mainLines.push(...stepNotes);
  scene.texts.push(...textBlock(base.main[0] + 6, base.main[1] + 4, base.main[1] + base.main[3] - 4, mainLines, pal.note, unitBudget(base.main[2]), "bottom"));

  const altLines: string[] = [
    `交错点：本帧 ${fmtOpt(nAltCurrent, 3)} 个（最终 ${fmtOpt(nAltFinal, 3)}），定理要求 deg+2 = ${fmtOpt(nAltExpected, 3)} 个；` +
      `符号交替 alt_signs_alternate = ${altAlternate === undefined ? "?" : String(altAlternate)}`,
    ratioCurrent === undefined
      ? `振幅比 max|e|/min|e|（= 1 才是等幅）：当前轮 = 无有限值（帧里是 null）⟺ 某个交错点的 e 恰好为 0、比值是 ∞ ——` +
        `它**不是** 1，也**不是** 0；最终 = ${fmtOpt(ratioFinal, 10)}`
      : `振幅比 max|e|/min|e|（= 1 才是等幅）：当前轮 = ${fmt(ratioCurrent, 10)}，最终 = ${fmtOpt(ratioFinal, 10)}`,
    `max|e| 最终 ${maxErrorFinal === undefined ? "?" : sci(maxErrorFinal)}　同阶 L² 投影 max|e| ${maxErrorL2 === undefined ? "?" : sci(maxErrorL2)}` +
      `　L²/minimax = ${fmtOpt(l2OverMinimax, 4)}（两个"最优"在最小化不同范数）`,
    `converged = ${converged === undefined ? "?" : String(converged)}　迭代次数 ${fmtOpt(iterations, 3)}　地板 u = ${sci(floor)}　` +
      `误差曲线（401 点网格）上被截断到地板的点（当前/最终/L² 合计）${fmtOpt(clippedPoints, 4)}`,
    belowFloor > 0
      ? `交错点里另有 ${belowFloor} 个 |e| ≤ u：对数轴画不到 0，已按地板画成红色标记（初始点集还没抓住极值就是这样）`
      : "本帧没有交错点落在地板上（|e| 全部在 u 之上）",
  ];
  if (altAlternate === false) {
    altLines.push(`⚠ 交错点已不再符号交替（alt_signs_alternate=false）：误差掉到舍入地板后，"数出 n+2 个等幅点"不再有内容`);
  }
  if (converged === false) {
    altLines.push(`⚠ converged=false：本帧没有拿到 de la Vallée Poussin 最优性证书，画的是历史最优迭代，不是"已经收敛"`);
  }
  if (certificateNote !== undefined) altLines.push(certificateNote);
  if (finalBelow > 0) {
    altLines.push(`最终交错点里另有 ${finalBelow} 个 |e| ≤ u：同样画在地板线上（对数轴没有 0）`);
  }
  if (floorNote !== undefined) altLines.push(floorNote);
  scene.texts.push(...textBlock(base.error[0] + 6, base.error[1] + 4, base.error[1] + base.error[3] - 4, altLines, pal.note, unitBudget(base.error[2]), "bottom"));

  // —— 角标 ——
  scene.badges.push(placeBadge(sourceBadge(scene.dataSource), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(placeBadge(badge("autoscale", "副轴 alt：自适应对数轴（|e| 可跨十几个数量级）", 8, 26), SLOT.auxAxis, width));
  scene.badges.push(
    placeBadge(badge("critical", `交错定理：交错点应为 deg+2 = ${fmtOpt(nAltExpected, 3)}（本帧 ${fmtOpt(nAltCurrent, 3)}）`), SLOT.critical, width),
  );
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        converged === false
          ? "⚠ 未拿到最优性证书（converged=false）"
          : ratioCurrent === undefined
            ? "当前轮振幅比无有限值（null）：交错点里有一个 e=0"
            : `振幅比 当前/最终 = ${fmt(ratioCurrent, 6)} / ${fmtOpt(ratioFinal, 6)}`,
      ),
      SLOT.status,
      width,
    ),
  );
  if (scene.dataSource.startsWith("backend+preview")) scene.badges.push(placeBadge(previewBadge(), SLOT.preview, width));

  let rawNonFinite = countNonFinite(error) + countNonFinite(errorFinal) + countNonFinite(errorL2);
  rawNonFinite += countNonFinite(altError) + countNonFinite(altErrorFinal);
  applyOverflow(scene, rawNonFinite, { slot: SLOT.overflow, width });
  const degradedObj = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedObj) scene.badges.push(degradedObj);

  requireDrawableLayers(scene);
  return scene;
};
