/**
 * `romberg_table`（2.4 章）的 model：Romberg 三角的每一列误差、对角线、实测阶与理论阶。
 *
 * 两个面板（顺序即绘制顺序，测试按此断言）
 * ----------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `conv` | 第 0 … k−1 列各自的误差曲线（第 j 列只有 k−j 个点，帧里给的是长表） |
 * | 2 | line | `conv` | 对角线 `diag_error`（每一层能拿到的最好估计，虚线 + 点） |
 * | 3 | line | `conv` | 舍入地板横线（`meta.error_floor`，对角线触底的那条线） |
 * | 4 | line | `conv` | 「第 6 层起只剩噪声」的竖线（`frame.critical.levels`，只在已揭示时画） |
 * | 5 | line | `order` | 每一列的**实测**阶 `order_measured`（点 + 线） |
 * | 6 | line | `order` | 每一列的**理论**阶 `2j+2`（虚线，规则 2 的对照物） |
 *
 * 角标槽位（固定，见 `common.ts::SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 /
 * 3 临界（第 6 层触地板）/ 4 状态 / 5 前端预览 / 6 溢出 / 7 降级。
 *
 * 这一页最容易骗人的两处，以及它们在图上的防法
 * --------------------------------------------
 * 1. **「只看最后一列」**：第 j 列只有 m−j 个可用点，列号越大可用行越少；帧在
 *    m−j 不足时**不给阶**（`order_j` 数组直接变短，而不是补 0）。防法：`conv` 面板
 *    逐列画曲线 + 各自点数，`order` 面板只画帧真的给了阶的那些列，并把
 *    `strings.column_note` 原文照登。
 * 2. **「外推一直加速」**：组合系数的绝对值之和 A_j 有界（→1.9692603，帧里给了
 *    最后一列的实测 Σ|c|），而**真正的限制在另一头**：对角线在第 6 层触到 2.2e-16，
 *    再往上加层只是把噪声放大近 2 倍。防法：临界竖线 + `strings.floor_note` 照登 +
 *    「不许把触地板的行算进拟合窗口」（`strings.window_note`：不剔得 6.96、剔掉得 7.97）。
 *
 * 几条刻意的决定
 * --------------
 * 1. **横轴用 `evals`（累计函数求值次数）**：它等于 `n_sub+1`（梯形每层的节点数），
 *    也正是"第 j 列误差 vs 节点数"那句话里的节点数——用**代价**当横坐标，
 *    "多算一层换到多少精度"才是可读的。
 * 2. **长表按 (行号 k, 列号 j) 还原成逐列曲线**：帧给的是 `column_j/column_k/error_table`
 *    三个等长数组（行主序），模型只做**索引还原**（`row*(row+1)/2 + col`），不插值。
 * 3. **`order` 面板纵轴不强行纳入 A_j**：阶是无量纲指数、A_j 是放大倍数，画在同一条
 *    纵轴上会读成"阶涨到 1.97"；A_j 进文字（数值 + manifest 里的极限值）。
 * 4. **`func=cusp` 时帧不给任何一列的阶**（折点落在节点上、误差恒在地板）：这时
 *    `order` 面板留白并说明原因，不画一条"0 阶"的假线。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  criticalOf,
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
  toNumeric,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  previewBadge,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  CHAPTER,
  SLOT,
  axisTitle,
  axisYlabel,
  currentStep,
  fmt,
  fmtOpt,
  head,
  maxAbs,
  minMaxOfAll,
  noteOf,
  paramSpec,
  plainNote,
  positiveMinMaxOfAll,
  sci,
  specTitle,
  splitColumns,
  textBlock,
  theoryOf,
  unitBudget,
} from "./common.js";

const VIZ = "romberg_table";
const TITLE = specTitle(VIZ, "Romberg 三角：每一列的阶与外推的代价");

/** 逐列曲线的配色：三色轮转，第 0 列（梯形）用主色。 */
const COLUMN_COLORS = 3;

export const rombergTableModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);
  const layout = splitColumns(width, height, { share: 0.58, gap: 64 });

  // —— 必需数组（长表 + 每层元信息）——
  const columnJ = requireArray(frame, "column_j");
  const columnK = requireArray(frame, "column_k");
  const errorTable = requireArray(frame, "error_table");
  const level = requireArray(frame, "level");
  const evals = requireArray(frame, "evals");
  const diagError = requireArray(frame, "diag_error");
  requireSameLength(columnJ, columnK, "帧数组 column_j 与 column_k");
  requireSameLength(columnJ, errorTable, "帧数组 column_j 与 error_table");
  requireSameLength(level, evals, "帧数组 level 与 evals");
  if (frame.has("n_sub")) requireSameLength(level, requireArray(frame, "n_sub"), "帧数组 level 与 n_sub");
  requireSameLength(level, diagError, "帧数组 level 与 diag_error");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const k = Math.max(1, Math.min(level.length, Math.round(metaNumber(frame, "step") ?? level.length)));
  const levelsParam = metaNumber(frame, "levels");
  const errorFloor = metaNumber(frame, "error_floor");
  const coefAbsSum = metaNumber(frame, "coefficient_abs_sum");
  const coefSum = metaNumber(frame, "coefficient_sum");
  const clipped = metaNumber(frame, "clipped_points");
  const diagCurrent = metaNumber(frame, "diag_error_current");
  const funcSmooth = metaNumber(frame, "func_smooth");
  if (metaNumber(frame, "step") === undefined) degraded.push("帧 meta 缺 step：已算层数按数组长度取并已点明（不假装是请求值）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: metaNumber(frame, "t") ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const stepInfo = currentStep(frame, opts, { lo: 1, hi: Math.max(1, Math.round(levelsParam ?? level.length)), label: "层数 k" });
  const revealed = Math.max(1, Math.min(k, evals.length));

  /* ------------------------------ 左列：各列误差 ------------------------------ */
  const xVals: number[] = [];
  for (let i = 0; i < revealed; i += 1) {
    const v = evals[i] as number;
    if (Number.isFinite(v) && v > 0) xVals.push(v);
  }
  const xLo = xVals.length > 0 ? Math.min(...xVals) : 2;
  const xHi = xVals.length > 0 ? Math.max(...xVals) : 4;
  const convAxis = makeAxis("conv", layout.left, {
    xscale: "log",
    yscale: "log",
    xlabel: "函数求值次数（= 节点数，对数轴）",
    ylabel: axisYlabel(VIZ, "conv", "|T_{k,j}−I|"),
    title: axisTitle(VIZ, "conv", "各列误差 vs 节点数（双对数）"),
    autoscaled: false,
  });
  // 右端留 2.2 倍：曲线自己的 `label` 画在**最后一个可见点**右侧，不留白就会被面板裁掉
  // （实测 1.25 倍时屏幕上是「第 0」「对角」这种半截标签）。
  convAxis.xlim = xHi > xLo ? [xLo / 1.25, xHi * 2.2] : [xLo / 1.6, xLo * 1.6];
  convAxis.xticks = niceTicks(convAxis.xlim[0], convAxis.xlim[1], 6, { log: true });

  // 长表 → 逐列曲线（纯索引还原：行 r 的起点是 r(r+1)/2，列 c 落在 +c 处）。
  const columns: { j: number; x: number[]; y: number[] }[] = [];
  for (let r = 0; r < revealed; r += 1) {
    for (let c = 0; c <= r; c += 1) {
      const idx = (r * (r + 1)) / 2 + c;
      if (idx >= errorTable.length) continue;
      const xv = evals[r] as number;
      const yv = errorTable[idx] as number;
      if (!Number.isFinite(xv) || !Number.isFinite(yv) || xv <= 0 || yv <= 0) continue;
      let col = columns[c];
      if (col === undefined) {
        col = { j: c, x: [], y: [] };
        columns[c] = col;
      }
      col.x.push(xv);
      col.y.push(yv);
    }
  }
  const drawable = columns.filter((c) => c !== undefined && c.x.length > 0);
  const yPool = positiveMinMaxOfAll([errorTable, diagError, errorFloor !== undefined && errorFloor > 0 ? [errorFloor] : undefined]) ?? [1e-16, 1];
  convAxis.ylim = niceLimits(yPool[0], yPool[1], 6, { log: true, pad: 0.03 });
  convAxis.yticks = niceTicks(convAxis.ylim[0], convAxis.ylim[1], 5, { log: true });
  scene.axes.push(convAxis);

  for (const col of drawable) {
    scene.layers.push(
      lineLayer(
        "conv",
        series(col.x, col.y, pal.curves[col.j % COLUMN_COLORS] as string, {
          width: col.j === 0 ? 2 : 1.5,
          marker: col.j === 0 ? "o" : "none",
          label: `第${col.j}列`,
        }),
      ),
    );
  }
  scene.layers.push(
    lineLayer(
      "conv",
      segmentedSeries(head(evals, revealed), head(diagError, revealed), pal.marker, {
        width: 1.6,
        dash: [4, 3],
        marker: "o",
        accept: (v) => v > 0,
        label: "对角线",
      }),
    ),
  );
  if (errorFloor !== undefined && errorFloor > 0 && errorFloor >= convAxis.ylim[0] / 10 && errorFloor <= convAxis.ylim[1]) {
    scene.layers.push(
      // 这条线横跨整个 x 轴、标注会正好落在右边界上被裁掉：地板值由下面的文字行给。
      lineLayer("conv", series(convAxis.xlim, [errorFloor, errorFloor], pal.muted, { width: 1, dash: [3, 3] })),
    );
  }
  const plateau = criticalOf(frame, "levels", paramSpec(VIZ, "levels")?.critical[0] ?? 6);
  const plateauLevel = Array.isArray(plateau) ? (plateau[0] as number) : plateau;
  const plateauRevealed = Number.isFinite(plateauLevel) && plateauLevel < revealed;
  if (plateauRevealed) {
    const xAt = evals[Math.round(plateauLevel)] as number;
    if (Number.isFinite(xAt) && xAt > 0) {
      scene.layers.push(
        lineLayer("conv", series([xAt, xAt], convAxis.ylim, pal.warn, { width: 1.4, dash: [5, 4], label: `第 ${fmt(plateauLevel, 3)} 层` })),
      );
    }
  }
  const convLines: string[] = [];
  convLines.push(
    `已算到第 ${fmt(revealed - 1, 3)} 层（上限 ${fmtOpt(levelsParam, 3)}）：对角线最后一点 ${fmtOpt(diagCurrent, 4)}、` +
      `地板 ${sci(errorFloor ?? Number.NaN)}、被截断点数 ${fmtOpt(clipped, 3)}`,
  );
  if (plateauRevealed) {
    convLines.push(`⚠ 竖线 = 第 ${fmt(plateauLevel, 3)} 层：对角线已触地板，再加层只放大噪声（截断误差已无意义）`);
  } else if (Number.isFinite(plateauLevel)) {
    convLines.push(`第 ${fmt(plateauLevel, 3)} 层（对角线触地板）还没揭示：拖到 k≥${fmt(plateauLevel + 1, 3)} 才看得到那条竖线`);
  }
  convLines.push("虚线 + 圆点 = 对角线 A_j（每一层能拿到的最好估计）；点线 = 地板；曲线右端的「第j列」= 列号，j 越大阶越高");
  pushIf(convLines, plainNote(noteOf(frame, "column_note")));
  pushIf(convLines, plainNote(noteOf(frame, "window_note")));
  if (funcSmooth !== undefined && funcSmooth < 0.5) pushIf(convLines, plainNote(noteOf(frame, "nonsmooth_note")));
  convLines.push(...stepInfo.notes);
  scene.texts.push(
    ...textBlock(
      convAxis.rect[0] + 6,
      convAxis.rect[1] + 4,
      convAxis.rect[1] + convAxis.rect[3] - 4,
      convLines,
      pal.note,
      // 只用面板宽度的 78%：右边 22% 留给曲线自己的 label（在最后可见点右侧）。
      unitBudget(Math.round(convAxis.rect[2] * 0.78)),
      "bottom",
    ),
  );

  /* ------------------------------ 右列：实测阶 vs 理论阶 ------------------------------ */
  const orderJ = frame.has("order_j") ? requireArray(frame, "order_j") : new Float64Array(0);
  const orderMeasured = frame.has("order_measured") ? requireArray(frame, "order_measured") : new Float64Array(0);
  const orderTheory = frame.has("order_theory") ? requireArray(frame, "order_theory") : new Float64Array(0);
  if (orderJ.length === 0) {
    degraded.push("帧里一列都没有可用的阶（order_j 为空）：右列留白（不编一个 0 阶），原因见 strings.column_note");
  }
  const orderAxis = makeAxis("order", layout.right, {
    xlim: [0, Math.max(1, revealed - 1)],
    ylim: orderRange(orderMeasured, orderTheory),
    xlabel: "列号 j",
    ylabel: axisYlabel(VIZ, "order", "阶 p_j"),
    title: axisTitle(VIZ, "order", "每一列的实测阶 vs 列号"),
    autoscaled: true,
  });
  orderAxis.xticks = niceTicks(orderAxis.xlim[0], orderAxis.xlim[1], 6);
  orderAxis.yticks = niceTicks(orderAxis.ylim[0], orderAxis.ylim[1], 6);
  scene.axes.push(orderAxis);
  if (orderMeasured.length > 0) {
    scene.layers.push(
      lineLayer("order", series(orderJ, orderMeasured, pal.curves[0], { width: 1.9, marker: "o", label: "实测阶" })),
    );
  }
  if (orderTheory.length > 0) {
    scene.layers.push(
      lineLayer("order", series(orderJ, orderTheory, pal.reference, { width: 1.4, dash: [6, 4], label: "理论阶 2j+2" })),
    );
  }

  const orderLines: string[] = [];
  if (orderMeasured.length > 0) {
    const pairs: string[] = [];
    for (let i = 0; i < orderMeasured.length; i += 1) {
      pairs.push(`${fmt(orderMeasured[i] as number, 6)}/${fmt(orderTheory[i] as number, 3)}`);
    }
    orderLines.push(`实测/理论（j=0…${fmt(orderJ[orderJ.length - 1] as number, 2)}）：${pairs.join("　")}`);
  } else {
    orderLines.push("本帧没有任何一列凑够可拟合的点：这里留白（不是 0 阶），原因见左列的 column_note");
  }
  orderLines.push(
    `外推的代价：最后一列 Σ|c_i| = ${fmtOpt(coefAbsSum, 8)}（Σc_i = ${fmtOpt(coefSum, 4)}），绝对值之和有界 ⇒ 噪声只放大近 2 倍`,
  );
  const theory = theoryOf(VIZ);
  if (theory.length >= 3) orderLines.push(theory[2] as string);
  if (funcSmooth !== undefined && funcSmooth < 0.5) {
    orderLines.push("f=|x|：折点在偶数层的节点上，梯形法从第 2 层起逐段精确——这时「列阶」没有意义（帧如实不给阶）");
  }
  const gainMax = maxAbs(frame.has("gain") ? requireArray(frame, "gain") : new Float64Array(0));
  if (gainMax !== undefined) orderLines.push(`同层相邻两列的误差比最大 ${fmt(gainMax, 4)}；第 j 列只有 k−j 个点，比值的个数也跟着少`);
  scene.texts.push(
    ...textBlock(
      orderAxis.rect[0] + 6,
      orderAxis.rect[1] + 4,
      orderAxis.rect[1] + orderAxis.rect[3] - 4,
      orderLines,
      pal.note,
      // 只用面板宽度的 62%：order 曲线从左下爬到右上，块窄一点才不会盖住它的爬升段。
      unitBudget(Math.round(orderAxis.rect[2] * 0.62)),
      "top",
    ),
  );

  /* ------------------------------ 角标 ------------------------------ */
  const funcLabel = noteOf(frame, "func_label") ?? noteOf(frame, "func") ?? "?";
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "右列 order：自适应（线性纵轴按本帧的阶取整；自适应会掩盖量级变化）", 8, 26), SLOT.auxAxis, width),
  );
  scene.badges.push(
    placeBadge(
      criticalBadge("levels", revealed, paramSpec(VIZ, "levels")?.critical ?? [], { beyond: "above", unit: " 层" }),
      SLOT.critical,
      width,
    ),
  );
  const status: string[] = [`${funcLabel}；已揭示 ${fmt(revealed, 3)} 层`];
  if (Number.isFinite(plateauLevel) && revealed > plateauLevel) status.push(`对角线已过第 ${fmt(plateauLevel, 3)} 层（触地板）`);
  if (orderMeasured.length === 0) status.push("没有任何一列有可信的阶（留白）");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 80), SLOT.status, width));
  if (scene.dataSource.startsWith("backend+preview")) scene.badges.push(placeBadge(previewBadge(), SLOT.preview, width));

  let rawNonFinite = 0;
  for (const name of frame.names) rawNonFinite += countNonFinite(toNumeric(frame.array(name)));
  applyOverflow(scene, rawNonFinite, { slot: SLOT.overflow, width });
  const degradedObj = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedObj) scene.badges.push(degradedObj);

  requireDrawableLayers(scene);
  return scene;
};

/* --------------------------------------------------------------------------- */
/* 局部工具                                                                      */
/* --------------------------------------------------------------------------- */

/** 阶面板的纵轴范围：实测与理论一起纳入、两端各留 12%（读数不贴边）。 */
function orderRange(measured: ArrayLike<number>, theory: ArrayLike<number>): [number, number] {
  const r = minMaxOfAll([measured, theory]);
  if (r === undefined) return [0, 10];
  const pad = Math.max(0.4, 0.12 * (r[1] - r[0]));
  return niceLimits(Math.min(0, r[0] - pad), r[1] + pad, 6);
}

/** 条件式入队（`undefined` 不入队）。 */
function pushIf(lines: string[], body: string | undefined): void {
  if (body !== undefined) lines.push(body);
}
