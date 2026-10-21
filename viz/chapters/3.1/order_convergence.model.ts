/**
 * `order_convergence`（3.1 章招牌 viz）的 model：真跑出来的解曲线 + 四条误差-步长曲线。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `solution` | 精确解 `y_exact_curve`（虚线）与数值解 `y_curve` |
 * | 2 | line | `solution` | 数值解的步点（两级：小点 + 当前步加粗）——**节点密度就是当前 h** |
 * | 3 | line | `error` | 误差地板 |y| = `meta.err_floor` 参考线（对数轴的下沿） |
 * | 4 | loglogfit | `error` | 整段最大误差 + **实测斜率**与**声明阶**两条参考（规则 2） |
 * | 5 | line | `error` | 终点误差、首步真局部误差、步内插值误差（三条同屏对照） |
 * | 6 | line+scatter | `error` | 当前 h 竖线 + 四个误差值的高亮点 |
 * | 7 | line | `error` | 洞见 1 的两条误差曲线在 `mark_h` 处连成的落差段（端点↔最大值） |
 *
 * 时间轴语义（manifest `time.mode="scrub"`，`tMax=10`）
 * ---------------------------------------------------
 * 滑块是**积分进度**：帧 `meta.t` 是进度坐标（0…10），`meta.t_now` 才是**真实时间**。
 * 模型把两者分开写：`scene.t` 用帧 `meta.t`（进度），面板文字给出"积分到 t=…（进度 …%）"。
 * 这两者被混为一谈正是本章 pitfalls 第 15 条，所以图上必须同时看得见。
 *
 * 防误导七条里本章的重点
 * ----------------------
 * * **规则 2**：`loglogfit` 同时给 `measuredSlope`（拟合窗口裁过）与 `theorySlope=declared_order`，
 *   两者都画；参考线锚在 `err` 末点（渲染层口径），所以按**当前 h** 高亮时两者的差当场可见。
 * * **规则 4**：误差轴是**固定窗口**（不随参数自适应）；`mark_*` 越出窗口、地板值被触到、
 *   拟合窗口丢弃的点数，全部进 `overflow.note`。
 * * **规则 7**：三条陷阱写进面板文字与 `pitfalls`，其中两条最容易读反的：
 *   ①"到了 1e-17 不是误差小，是双精度量不出来"；②"这条参考线的斜率是按**声明的阶**画的"。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  criticalOf,
  metaNumber,
  optionalArray,
  palette,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  stackedPanels,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  loglogFitLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  type LineSeries,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";
import {
  applyOverflowNote,
  axisTitleOf,
  axisYlabelOf,
  countAtCeiling,
  finiteMinMaxOf,
  logRange,
  positiveMinMax,
  stackPanels,
  textBlock,
  textBottom,
  textTop,
  unitBudget,
  wrapUnits,
  type PanelStack,
} from "./_panel.js";

const CHAPTER = "3.1";
const VIZ = "order_convergence";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "阶：误差-步长双对数阶梯与真跑出来的解";
const TIME_MAX = SPEC?.time.tMax ?? 10;

/**
 * 固定角标槽位（`model.ts::BADGE_LAYOUT`：两列 4 行 × 2 列）。
 * 长文案角标（状态、降级）各占一行左列，右边留空，避免与邻居叠字。
 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  status: 3,
  preview: 4,
  overflow: 5,
  critical: 6,
  degraded: 7,
} as const;

/** 四条误差曲线的显示名（顺序与面板文字、图层一致）。 */
const ERR_LABELS = ["整段最大 max|y_h−y|", "终点 |y_h(T)−y(T)|", "首步真局部（6 子步）", "步内插值（稠密输出）"] as const;

export const orderConvergenceModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const width = Math.max(320, Math.round(opts.width));
  const height = Math.max(240, Math.round(opts.height));

  // —— 必需：画不出来就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const tCurve = requireArray(frame, "t_curve");
  const yCurve = requireArray(frame, "y_curve");
  const yExact = requireArray(frame, "y_exact_curve");
  const errCurve = requireArray(frame, "error_curve");
  const hLadder = requireArray(frame, "h_ladder");
  const errorGlobal = requireArray(frame, "error_global");
  requireSameLength(tCurve, yCurve, "帧数组 t_curve 与 y_curve");
  requireSameLength(tCurve, yExact, "帧数组 t_curve 与 y_exact_curve");
  requireSameLength(tCurve, errCurve, "帧数组 t_curve 与 error_curve");
  const pitfalls = requirePitfalls(frame);

  // —— 可选：缺了就降级（逐条写进 degraded，最后挂成一枚 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：滑块位置改用请求 t（两者可能不同）");
  const tNow = metaNumber(frame, "t_now");
  if (tNow === undefined) degraded.push("帧 meta 缺 t_now：不显示真实积分时间（不拿滑块位置冒充）");
  const tEnd = metaNumber(frame, "t_end");
  if (tEnd === undefined) degraded.push("帧 meta 缺 t_end：不显示积分进度百分比");
  const nDone = metaNumber(frame, "n_steps_done");
  const nTotal = metaNumber(frame, "n_steps_total");
  if (nDone === undefined || nTotal === undefined) degraded.push("帧 meta 缺 n_steps_*：不显示步数（不按滑块位置反推）");
  const declared = metaNumber(frame, "declared_order");
  if (declared === undefined) degraded.push("帧 meta 缺 declared_order：不画理论斜率参考线（不拿拟合值充理论值）");
  const hNow = metaNumber(frame, "h");
  if (hNow === undefined) degraded.push("帧 meta 缺 h：当前 h 只能用帧数组 mark_h（不再退回请求参数）");
  const errFloor = metaNumber(frame, "err_floor");
  if (errFloor === undefined) degraded.push("帧 meta 缺 err_floor：不画误差地板参考线（地板值不许猜）");
  const ratioMaxOverEnd = metaNumber(frame, "ratio_max_over_endpoint");
  const methodLabel = stringOf(frame, "method_label");
  if (methodLabel === undefined) degraded.push("帧 strings 缺 method_label：方法名只在角标里用帧里的 method 字段");
  const problemLabel = stringOf(frame, "problem_label");
  const slopeNote = stringOf(frame, "slope_note");
  if (slopeNote === undefined) degraded.push("帧 strings 缺 slope_note：不声明拟合窗口（丢弃点数看不见就不许下结论）");
  const timeSemantics = stringOf(frame, "time_semantics");

  const errorEndpoint = optionalArray(frame, "error_endpoint");
  const errorLocal = optionalArray(frame, "error_local");
  const errorDense = optionalArray(frame, "error_dense");
  const theoryGlobal = optionalArray(frame, "theory_global");
  const theoryLocal = optionalArray(frame, "theory_local");
  const markH = optionalArray(frame, "mark_h");
  const markGlobal = optionalArray(frame, "mark_error_global");
  const markEndpoint = optionalArray(frame, "mark_error_endpoint");
  if (errorEndpoint === undefined) degraded.push("帧缺数组 error_endpoint：不画终点误差曲线");
  if (errorLocal === undefined) degraded.push("帧缺数组 error_local：不画首步真局部误差曲线");
  if (errorDense === undefined) degraded.push("帧缺数组 error_dense：不画步内插值误差曲线");
  if (theoryGlobal === undefined) degraded.push("帧缺数组 theory_global：不画全局理论斜率线（规则 2 仍由 loglogfit 承担）");
  if (markH === undefined || markGlobal === undefined) degraded.push("帧缺数组 mark_h/mark_error_global：不画当前 h 的竖线与高亮点");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: stringOf(frame, "data_source") ?? "backend",
    pitfalls,
  });

  // —— 排版 ——
  // **宽画布分两栏**：左栏解曲线（竖着放得下，才能看出"两条曲线贴在一起"），右栏误差双对数 +
  // 它下面的说明带。960×600 上竖直叠三块的话，每块只剩 ~100px，说明块会把曲线盖住（1.2 章的教训）。
  // 窄画布退回竖叠（说明块相应截断，`textBlock` 会写明"另有 n 行未画"）。
  const absolute = stackedPanels(width, height, { footer: width >= 712 ? 118 : 170 });
  const plotH = absolute.bottom - absolute.top;
  const wide = absolute.contentWidth >= 2 * 300 + 24;
  const COL_GAP = 54;
  const leftW = wide ? Math.round((absolute.contentWidth - COL_GAP) * 0.46) : absolute.contentWidth;
  const rightW = wide ? absolute.contentWidth - COL_GAP - leftW : absolute.contentWidth;
  const mainBudget = unitBudget(leftW - 12);
  /** 说明带的折行宽度：宽画布上说明带横跨两栏，窄画布上就是整幅宽度。 */
  const infoBandBudget = unitBudget(wide ? absolute.contentWidth - 12 : absolute.contentWidth - 12);

  const marksEarly = markPoints(markH, [markGlobal, markEndpoint]);
  const hMark = marksEarly.h ?? hNow;

  const mainLines: string[] = [
    `解曲线按当前 h=${fmtMaybe(hMark)} 步进到 t=${fmtMaybe(tNow)}` +
      (nDone !== undefined && nTotal !== undefined ? `（第 ${Math.round(nDone)}/${Math.round(nTotal)} 步）` : "") +
      "；点的疏密就是 h 的大小。",
    "精确解（虚线）画在同一根轴上：两者贴在一起不等于解得准——自适应的纵轴会把「分不开」放大成「很准」。",
  ];
  const infoLines: string[] = [];
  infoLines.push(`四条误差曲线：${ERR_LABELS.join(" / ")}（颜色与线型见图例）。`);
  const dropHead = metaNumber(frame, "slope_drop_head");
  const dropTail = metaNumber(frame, "slope_drop_tail");
  infoLines.push(
    `实测斜率：整段 ${fmtMaybe(measuredSlope(frame, "measured_slope_global"))}、局部 ${fmtMaybe(measuredSlope(frame, "measured_slope_local"))}、稠密 ${fmtMaybe(measuredSlope(frame, "measured_slope_dense"))}` +
      (declared !== undefined ? `（理论 −p 与 −(p+1)，p=${formatNumber(declared, 3)}）。` : "。"),
  );
  infoLines.push(
    `拟合窗口：头丢弃 ${fmtMaybe(dropHead)} 点、尾丢弃 ${fmtMaybe(dropTail)} 点；不裁窗口时预渐近点会把斜率抬高（rkf45 整窗 4.16）。`,
  );
  const denseSlope = measuredSlope(frame, "measured_slope_dense");
  if (declared !== undefined && denseSlope !== undefined) {
    infoLines.push(
      `步内插值的阶封顶在 min(p,4)=${formatNumber(Math.min(declared, 4), 3)}（实测 ${formatNumber(denseSlope, 4)}）：节点越来越准，插值算子不会跟着涨。`,
    );
  }
  infoLines.push(
    `「终点误差」≠「整段最大误差」：本帧 max/端点 = ${fmtMaybe(ratioMaxOverEnd)}` +
      (ratioMaxOverEnd !== undefined && ratioMaxOverEnd > 100 ? `——拿端点代表整段会低报 ${formatNumber(ratioMaxOverEnd, 3)} 倍。` : "。"),
  );
  const flooredCount =
    errFloor !== undefined
      ? countAtCeiling(errorGlobal, errFloor) + (errorDense !== undefined ? countAtCeiling(errorDense, errFloor) : 0)
      : 0;
  infoLines.push(
    `滑块是积分进度（manifest tMax=${formatNumber(TIME_MAX, 2)}，帧 meta.t=${fmtMaybe(tMeta)}）` +
      (tNow !== undefined ? `→ 真实时间 t=${formatNumber(tNow, 4)}。` : "。") +
      `误差地板 ${fmtMaybe(errFloor)} 是双精度下限，不是「很小」` +
      (flooredCount > 0 ? `（本帧 ${flooredCount} 个样本压在上面）。` : "。") +
      (timeSemantics !== undefined ? `帧声明：${clipText(timeSemantics, Math.max(30, infoBandBudget - 20))}` : ""),
  );
  if (slopeNote !== undefined) infoLines.push(clipText(slopeNote, Math.max(40, infoBandBudget - 4)));

  /** 一段文字折行后占的高度（与 `textBlock` 同一把尺子）。 */
  function textHeight(lines: readonly string[], lineBudget: number, cap: number): number {
    const rows = lines.flatMap((body) => wrapUnits(body, lineBudget)).length;
    return Math.min(rows * 15 + 6, cap);
  }
  let mainRect: [number, number, number, number];
  let errRect: [number, number, number, number];
  let infoRect: [number, number, number, number];
  let mainText: { top: number; bottom: number };
  let infoText: { top: number; bottom: number };
  if (wide) {
    // 说明带**横跨两栏**（全宽 = 每行放得下 60 个汉字），每条只占一行、不会被截断；
    // 左栏解曲线下面只留它自己那两行 caption。
    const infoTextH = textHeight(infoLines, infoBandBudget, Math.round(plotH * 0.45));
    const axesH = Math.max(120, plotH - infoTextH - 28);
    const mainTextH = textHeight(mainLines, mainBudget, Math.round(axesH * 0.34));
    mainRect = [absolute.left, absolute.top, leftW, Math.max(80, axesH - mainTextH - 28)];
    mainText = { top: mainRect[1] + mainRect[3] + 28, bottom: absolute.top + axesH };
    errRect = [absolute.left + leftW + COL_GAP, absolute.top, rightW, axesH];
    infoRect = [absolute.left, absolute.top + axesH + 28, absolute.contentWidth, infoTextH];
    infoText = { top: infoRect[1], bottom: absolute.top + plotH };
  } else {
    // 窄画布退回竖叠：说明块相应截断（`textBlock` 会写明"另有 n 行未画"）。
    const stack: PanelStack = stackPanels(absolute, [0.44, 0.34, 0.22], { minHeights: [110, 96, 60] });
    const mainPanel = stack.rects[0] as [number, number, number, number];
    const errPanel = stack.rects[1] as [number, number, number, number];
    const infoPanel = stack.rects[2] as [number, number, number, number];
    const mainTextH = textHeight(mainLines, mainBudget, Math.max(0, mainPanel[3] - 98));
    mainRect = [mainPanel[0], mainPanel[1], mainPanel[2], Math.max(70, mainPanel[3] - mainTextH - 28)];
    mainText = { top: mainRect[1] + mainRect[3] + 28, bottom: mainPanel[1] + mainPanel[3] };
    errRect = errPanel;
    infoRect = infoPanel;
    infoText = { top: textTop(infoPanel), bottom: textBottom(infoPanel, false) };
  }

  // —— 主轴：解曲线（自适应纵轴，如实标注）——
  const yRange = finiteMinMaxOf(yCurve) ?? [0, 1];
  const yExactRange = finiteMinMaxOf(yExact);
  const ylo = Math.min(yRange[0], yExactRange?.[0] ?? yRange[0]);
  const yhi = Math.max(yRange[1], yExactRange?.[1] ?? yRange[1]);
  const mainAxis = makeAxis("solution", mainRect, {
    ylim: padLinear(ylo, yhi, 0.08),
    xlim: [0, tEnd ?? (tCurve.length > 0 ? Math.max(0, (tCurve[tCurve.length - 1] as number) || 1) : 1)],
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "solution", "y"),
    title: axisTitleOf(SPEC?.axes, "solution", "解曲线（当前 h 的步点画在曲线上）"),
    autoscaled: true,
  });

  // —— 副轴：误差-步长（固定窗口的双对数）——
  const errAxis = makeAxis("error", errRect, {
    xlim: logRange(...ladderRange(hLadder)),
    ylim: errorYlim([errorGlobal, errorEndpoint, errorLocal, errorDense, theoryGlobal, theoryLocal, markGlobal, markEndpoint], errFloor),
    xscale: "log",
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "error", "误差"),
    title: axisTitleOf(SPEC?.axes, "error", "误差 vs 步长（双对数，固定窗口）"),
    autoscaled: false,
  });
  scene.axes.push(mainAxis, errAxis);

  // —— 图层 1~2：解曲线、精确解、步点 ——
  const stepX = toArray(tCurve);
  const stepY = toArray(yCurve);
  scene.layers.push(
    lineLayer("solution", [
      series(stepX, stepY, pal.curves[0], { width: 2, label: `数值解（${methodLabel ?? "当前方法"}）` }),
      series(stepX, toArray(yExact), pal.exact, { width: 1.3, dash: [6, 3], label: "精确解" }),
    ]),
  );
  scene.layers.push(
    lineLayer("solution", [
      series(stepX, stepY, pal.curves[0], { width: 1, marker: "o", alpha: 0.75, label: "步点（节点密度 = 当前 $h$）" }),
    ]),
  );

  // —— 图层 3：误差地板参考线（对数轴的下沿；触到它 = 双精度量不出来了）——
  if (errFloor !== undefined && errFloor > 0) {
    scene.layers.push(
      lineLayer(
        "error",
        series([errAxis.xlim[0], errAxis.xlim[1]], [errFloor, errFloor], pal.reference, {
          width: 1,
          dash: [2, 4],
          label: `误差地板 ${formatNumber(errFloor, 3)}（双精度下限，不是"误差很小"）`,
        }),
      ),
    );
  }

  // —— 图层 4：规则 2 的强制形态：实测斜率 + 理论斜率必须同图 ——
  const slope = measuredSlope(frame, "measured_slope_global");
  if (declared !== undefined && slope !== undefined && positiveLength(hLadder, errorGlobal) >= 2) {
    scene.layers.push(
      loglogFitLayer("error", hLadder, errorGlobal, slope, declared, {
        seriesLabel: `实测阶 ${formatNumber(slope, 4)}（最小二乘）`,
        theoryLabel: `理论 −${formatNumber(declared, 3)} = −p（声明阶，非拟合）`,
        color: pal.curves[0],
        theoryColor: pal.marker,
      }),
    );
  } else {
    if (slope === undefined) degraded.push("帧 meta 缺 measured_slope_global：不画 loglogfit（规则 2 不许只画一条理论线）");
    scene.layers.push(
      lineLayer("error", series(hLadder, errorGlobal, pal.curves[0], { width: 1.8, marker: "o", label: "整段最大误差" })),
    );
  }

  // —— 图层 5：另外三条误差曲线（局部 / 稠密 / 端点）+ 理论局部斜率线 ——
  const extra: LineSeries[] = [];
  const push = (h: ArrayLike<number> | undefined, e: ArrayLike<number> | undefined, color: string, label: string, dash?: number[]): void => {
    if (h === undefined || e === undefined || h.length !== e.length || h.length === 0) return;
    extra.push(...segments(h, e, color, dash, label));
  };
  if (theoryLocal !== undefined && theoryLocal.length === hLadder.length && declared !== undefined) {
    extra.push(
      ...segments(hLadder, theoryLocal, pal.warn, [7, 3], `理论局部斜率 −${formatNumber(declared + 1, 3)}（不得与下面的拟合值混读）`),
    );
  }
  push(hLadder, errorLocal, pal.curves[2], "首步真局部误差（6 子步参考）", [4, 3]);
  push(hLadder, errorDense, pal.residual, "步内插值误差（稠密输出，min(p,4) 阶）", [1, 3]);
  push(hLadder, errorEndpoint, pal.curves[1], "终点误差", [10, 4]);
  if (extra.length > 0) scene.layers.push(lineLayer("error", extra));
  else degraded.push("帧数组里没有可画的附加误差曲线：误差面板只剩整段最大误差");

  // —— 图层 6：当前 h 竖线 + 四个误差值 ——
  const marks = marksEarly;
  if (hMark !== undefined && Number.isFinite(hMark) && hMark > 0) {
    scene.layers.push(
      lineLayer(
        "error",
        series([hMark, hMark], errAxis.ylim, pal.marker, { width: 1.1, dash: [3, 3], label: `当前 h = ${formatNumber(hMark, 4)}` }),
      ),
    );
  }
  if (marks.y.length > 0 && hMark !== undefined) {
    scene.layers.push(
      scatterLayer("error", marks.y.map(() => hMark), marks.y, {
        r: 3.6,
        color: pal.marker,
        label: "当前 $h$ 处的误差（各一条曲线一个点）",
      }),
    );
  }

  // —— 图层 7：同一 h 上"终点误差 → 整段最大误差"的落差段（两地最容易被读混的量）——
  if (marks.h !== undefined && marks.global !== undefined && marks.endpoint !== undefined) {
    scene.layers.push(
      lineLayer(
        "error",
        series([marks.h, marks.h], [marks.endpoint, marks.global], pal.ink, {
          width: 3,
          alpha: 0.5,
          // 不写 series 标签：右上是 `loglogfit` 两条参考线标签的地盘，第三个标签会叠字；
          // 这个比值已经在底部说明带里（"终点误差 ≠ 整段最大误差" 那条）。
        }),
      ),
    );
  }

  scene.texts.push(...textBlock(mainRect[0] + 6, mainText.top, mainText.bottom, mainLines, pal.muted, mainBudget, "top"));
  scene.texts.push(...textBlock(infoRect[0] + 6, infoText.top, infoText.bottom, infoLines, pal.muted, infoBandBudget, wide ? "top" : "bottom"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "误差轴：固定窗口的双对数（不随参数自适应，纵轴自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );

  const statusBits: string[] = [];
  statusBits.push(
    `${methodLabel ?? stringOf(frame, "method") ?? "该方法"}@${problemLabel ?? stringOf(frame, "problem") ?? "本算例"}` +
      (declared !== undefined ? `　p=${formatNumber(declared, 3)}` : "") +
      (hMark !== undefined ? `　h=${formatNumber(hMark, 4)}` : ""),
  );
  statusBits.push(
    nDone !== undefined && nTotal !== undefined ? `第 ${Math.round(nDone)}/${Math.round(nTotal)} 步` : "步数未知",
  );
  if (marks.global !== undefined && marks.endpoint !== undefined) {
    statusBits.push(`max/端点 = ${formatNumber(marks.global / marks.endpoint, 3)}`);
  }
  scene.badges.push(placeBadge(badge("status", statusBits.join("　"), 8, 8), SLOT.status, width));

  const crit = criticalOf(frame, "h", Number.NaN);
  if (Number.isFinite(crit as number) && hMark !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("h", hMark, crit as number, { beyond: "above" }), SLOT.critical, width));
  }

  requireDrawableLayers(scene);
  const rawNonFinite =
    countNonFinite(tCurve) + countNonFinite(yCurve) + countNonFinite(yExact) + countNonFinite(errCurve) + countNonFinite(hLadder);
  const extras: string[] = [];
  if (flooredCount > 0 && errFloor !== undefined) extras.push(`误差轴地板 ${formatNumber(errFloor, 3)} 上有 ${flooredCount} 个样本（详见面板说明）`);
  if (marks.clipped > 0) extras.push(`${marks.clipped} 个当前-h 标记在对数轴上没有有限正值（未绘制）`);
  for (const [name, arr] of [
    ["error_global", errorGlobal],
    ["error_endpoint", errorEndpoint],
    ["error_local", errorLocal],
    ["error_dense", errorDense],
  ] as const) {
    if (arr === undefined) continue;
    const bad = countNonFinite(arr);
    if (bad > 0) extras.push(`${name} 有 ${bad} 个非有限样本（对数轴上未绘制）`);
  }
  applyOverflowNote(scene, extras.join("；"), rawNonFinite, { slot: SLOT.overflow, width });

  const degradedNode = degradedBadgeLocal(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（只做显示几何与文本，不碰物理量）
// --------------------------------------------------------------------------- //

/** 取一个可选 meta 标量（缺失/NaN 一律 undefined，不猜 0）。 */
function measuredSlope(frame: Parameters<SceneModel>[0], key: string): number | undefined {
  const v = metaNumber(frame, key);
  return v !== undefined && Number.isFinite(v) ? v : undefined;
}

/** `ArrayLike<number>` → 普通数组（帧数组是零拷贝视图，模型不许改它，只读地拷成新数组最省心）。 */
function toArray(a: ArrayLike<number>): number[] {
  const out = new Array<number>(a.length);
  for (let i = 0; i < a.length; i += 1) out[i] = a[i] as number;
  return out;
}

/**
 * 把 `(x, y)` 按"有限且 > 0"切成若干条曲线（对数轴上 0/负值不存在）。
 *
 * 为什么不把 NaN 交给渲染层：`validateScene` 明确拒绝含非有限纵坐标的 line 图层；
 * 而"对数轴上的 0"根本画不出来。断点语义是"这里没有数据"，不能连直线跨过去。
 */
function segments(x: ArrayLike<number>, y: ArrayLike<number>, color: string, dash: number[] | undefined, label: string): LineSeries[] {
  const n = Math.min(x.length, y.length);
  const runs: { start: number; end: number }[] = [];
  let start = -1;
  for (let i = 0; i < n; i += 1) {
    const xv = x[i] as number;
    const yv = y[i] as number;
    const ok = Number.isFinite(xv) && xv > 0 && Number.isFinite(yv) && yv > 0;
    if (ok && start < 0) start = i;
    if (!ok && start >= 0) {
      runs.push({ start, end: i });
      start = -1;
    }
  }
  if (start >= 0) runs.push({ start, end: n });
  return runs.map((run, k) => {
    const xs = new Float64Array(run.end - run.start);
    const ys = new Float64Array(run.end - run.start);
    for (let i = run.start; i < run.end; i += 1) {
      xs[i - run.start] = x[i] as number;
      ys[i - run.start] = y[i] as number;
    }
    return series(xs, ys, color, { width: 1.5, dash, marker: "o", label: k === 0 ? label : undefined });
  });
}

/** 有限、>0 的样本个数（决定"这条曲线画不画得出来"）。 */
function positiveLength(x: ArrayLike<number>, y: ArrayLike<number>): number {
  const n = Math.min(x.length, y.length);
  let k = 0;
  for (let i = 0; i < n; i += 1) {
    const xv = x[i] as number;
    const yv = y[i] as number;
    if (Number.isFinite(xv) && xv > 0 && Number.isFinite(yv) && yv > 0) k += 1;
  }
  return k;
}

/** 线性轴的 y 范围：两端各留 `frac` 的相对余量（退化时给 ±1）。 */
function padLinear(lo: number, hi: number, frac: number): [number, number] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (hi <= lo) {
    const d = Math.max(1e-12, Math.abs(lo) * 0.1);
    return [lo - d, hi + d];
  }
  const pad = frac * (hi - lo);
  return [lo - pad, hi + pad];
}

/** 步长阶梯横轴范围（对数轴要求 > 0）。 */
function ladderRange(h: ArrayLike<number>): [number, number] {
  const r = positiveMinMax(h);
  return r ?? [0.00390625, 0.5];
}

/** 误差轴（固定窗口）的纵轴范围：把四条曲线与所有参考线一起包住，再让出一格。 */
function errorYlim(arrays: (ArrayLike<number> | undefined)[], floor: number | undefined): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const a of arrays) {
    if (a === undefined) continue;
    const r = positiveMinMax(a);
    if (r === undefined) continue;
    if (r[0] < lo) lo = r[0];
    if (r[1] > hi) hi = r[1];
  }
  if (floor !== undefined && floor > 0) {
    lo = Math.min(lo, floor);
    hi = Math.max(hi, floor);
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [1e-17, 1];
  return logRange(Math.max(lo, Number.MIN_VALUE), hi);
}

/** 当前 h 的标记点：竖线位置 + 一组高亮值；越出纵轴窗口的点单独计数。 */
interface MarkPoints {
  h?: number;
  global?: number;
  endpoint?: number;
  y: number[];
  clipped: number;
}

function markPoints(markH: ArrayLike<number> | undefined, others: (ArrayLike<number> | undefined)[]): MarkPoints {
  const h = markH !== undefined && markH.length > 0 && Number.isFinite(markH[0] as number) ? (markH[0] as number) : undefined;
  const y: number[] = [];
  let clipped = 0;
  let global: number | undefined;
  let endpoint: number | undefined;
  others.forEach((a, k) => {
    if (a === undefined || a.length === 0) return;
    const v = a[0] as number;
    if (!Number.isFinite(v) || !(v > 0)) {
      clipped += 1;
      return;
    }
    y.push(v);
    if (k === 0) global = v;
    if (k === 1) endpoint = v;
  });
  return { h, global, endpoint, y, clipped };
}

/** 误差样本里有多少个越出固定窗口的下界（对数轴画不出来的与越界的都算）。 */
/** 把长串裁到 `units` 个显示宽度单位（角标与面板共用同一把尺子）。 */
function clipText(body: string, units: number): string {
  let used = 0;
  let out = "";
  for (const ch of body) {
    const w = (ch.codePointAt(0) ?? 0) > 0x2e80 ? 2 : 1;
    if (used + w > units) return `${out}…`;
    out += ch;
    used += w;
  }
  return out;
}

/** 数值的紧凑显示（undefined 显示为「—」而不是 0）。 */
function fmtMaybe(v: number | undefined): string {
  return v === undefined || !Number.isFinite(v) ? "—" : formatNumber(v, 4);
}

/** 降级角标（与 `model.ts::degradedBadge` 同口径，但走本文件的 SLOT 常量，槽位才固定在 7）。 */
function degradedBadgeLocal(items: readonly string[], slot: number, width: number) {
  if (items.length === 0) return undefined;
  const head = items.slice(0, 2).join("；");
  const rest = items.length > 2 ? `；等 ${items.length} 项` : "";
  return placeBadge(badge("status", `⚠ 降级：${head}${rest}`, 8, 8), slot, width);
}
