/**
 * `qr_schur`（1.4 章）的 model：次对角元衰减热力图 + Ritz 值到真谱的距离。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | heat | `main` | `subdiag_heat`：log10\|次对角元\|，行 = 下标 i、列 = QR 迭代步 |
 * | 2 | line | `main` | 当前步的扫描线（竖虚线，x = meta.step） |
 * | 3 | scatter | `main` | `subdiag_current`：当前步仍活着的次对角元（x = step, y = i） |
 * | 4 | scatter | `main` | `subdiag_final`：末步的次对角元（同一个 x，对照"压到什么程度"） |
 * | 5 | line | `aux` | `error`：Ritz 值到真谱的**实测**距离（对数纵轴） |
 * | 6 | line | `aux` | `theory_rate`：无位移 QR 的**理论**衰减 \|λ₂/λ₁\|^k（规则 2 的第二条斜率） |
 * | 7 | line | `aux` | 当前步游标 |
 * | 8 | scatter | `aux` | `error_mark`：当前步的谱距离 |
 *
 * 角标槽位：0 来源 / 1 主轴固定（热力图色阶不随帧动）/ 2 副轴自适应 / 4 状态 / 6 溢出 / 7 降级。
 * 本页没有理论临界值（manifest 里 `critical` 为空），所以**不挂** critical 角标——
 * 规则 6 要求的是"越过临界值时现身"，不是"每页都挂一枚"。
 *
 * 几条刻意的决定
 * --------------
 * 1. **热力图的两个方向不能反**：manifest 给的轴标题写死了「行 = i，列 = 迭代步」，
 *    帧里的 `subdiag_heat` 形状也正是 `(n−1, iters+1)`（`hist.T` 后的行主序缓冲）。
 *    `heatLayer` 的约定是"行主序、第 0 行 = `extent[2]`（y 最小）"，所以
 *    `nx = iters+1`（x = 步）、`ny = n−1`（y = 下标 i）、`values` 原样传——不必转置。
 * 2. **`-300` 是地板，不是数据**：后端写的是 `log10(max(|h|, 1e-300))`，收敛后整列都是
 *    −300。若照原样铺色，色阶会被拉到 300 个数量级，所有真实结构挤成一色。这里**夹在 −16**
 *    （1.2 章 `lu_heatmap` 的同一口径）并把夹掉的格点数报进文本与 overflow。
 * 3. **色阶固定 `[-16, 0]`**：自适应色阶会让"某一列开始变暗"这个**结论**在动画里消失
 *    （每帧都重新拉满对比度）——PROTOCOLS §6.2 规则 1/4。
 * 4. **`error` 里的 `1e-300` 是"恰好 0"**：`converged` 时 Ritz 距离可以是精确的 0
 *    （对称族在 Wilkinson 位移下真的会），后端把它抬到地板以便上对数轴。模型把地板值
 *    换成**轴底**并数出个数（"贴地板"与"真的等于轴底"在文本里区分），而不是丢掉。
 * 5. **两条曲线的量纲不同**（距离 vs 比值），共享一条对数轴只为**比斜率**——文本里写明。
 */

import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  canvasSize,
  clipBadgeText,
  dataSourceOf,
  degradedBadge,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  toNumeric,
  segmentedSeries,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  heatLayer,
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";
import {
  LOG_FLOOR,
  applyOverflowNote,
  atFloor,
  axisTitleOf,
  axisYlabelOf,
  channelRange,
  fmtSci,
  integerTicks,
  logRange,
  panelBudget,
  pitfallsGuard,
  replaceFloors,
  textBlock,
  textBottom,
  textTop,
  wideLayout,
} from "./_shared.js";

const CHAPTER = "1.4";
const VIZ = "qr_schur";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "QR 迭代：次对角元衰减热力图与 Ritz 值";
const AXES = SPEC?.axes;

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
const SLOT = { source: 0, axisMode: 1, auxAxis: 2, critical: 3, status: 4, preview: 5, overflow: 6, degraded: 7 } as const;

/** 热力图的显示下限（`log10`）：再往下都是"已经到机器精度"，不参与色阶。 */
const HEAT_FLOOR = -16;
/** 光栅器把色标画在轴右侧 46px（宽 14px + 数值标签），这里按 1.2 章的口径留 162px 过道。 */
const COLORBAR_GUTTER = 162;

export const qrSchurModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段 ——
  requireArray(frame, "subdiag_heat"); // 先过统一的存在性检查（缺数组 → ModelInputError）
  const heat = frame.matrix("subdiag_heat");
  const cur = requireArray(frame, "subdiag_current");
  const fin = requireArray(frame, "subdiag_final");
  const err = requireArray(frame, "error");
  const mark = requireArray(frame, "error_mark");
  const steps = requireArray(frame, "step_grid");
  const theory = requireArray(frame, "theory_rate");
  requireSameLength(steps, err, "step_grid/error");
  requireSameLength(steps, theory, "step_grid/theory_rate");
  if (heat.rows < 1 || heat.cols < 2) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 subdiag_heat 形状是 ${heat.rows}×${heat.cols}：至少要能画出一条随步数变化的次对角元带`,
    );
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const k = metaNumber(frame, "step");
  if (k === undefined) degraded.push("帧 meta 缺 step：不画当前步扫描线（不猜是第几步）");
  const nIter = metaNumber(frame, "n_iter");
  const converged = metaNumber(frame, "converged");
  const nDefl = metaNumber(frame, "n_deflations");
  const nBlocks = metaNumber(frame, "n_blocks");
  const gap = metaNumber(frame, "spectral_gap");
  const errCurrent = metaNumber(frame, "error_current");
  const errFinal = metaNumber(frame, "error_final");
  const subFinalMax = metaNumber(frame, "subdiag_final_max");
  const subInitMax = metaNumber(frame, "subdiag_initial_max");
  const complexSpectrum = (metaNumber(frame, "is_complex_spectrum") ?? 0) > 0.5;
  const status = stringOf(frame, "status");
  if (status === undefined) degraded.push("帧 strings 缺 status：状态角标不自行编造结论");
  const note = stringOf(frame, "note");
  const family = stringOf(frame, "family");
  const shift = stringOf(frame, "shift");
  if (family === undefined || shift === undefined) degraded.push("帧 strings 缺 family/shift：不显示族与位移策略（不按参数猜）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：上=热力图（宽、矮，形状跟着 (n−1)×(iters+1) 走），下=Ritz 距离曲线 ——
  const guard = pitfallsGuard(width, height, pitfalls);
  const layout = wideLayout(width, height, {
    footer: guard.footer,
    topShare: 0.44,
    gutter: COLORBAR_GUTTER,
    // 52 而不是 36：上面那块轴的 xlabel 带占到 `bottom+28`，下面那块的标题带又从 `top−18`
    // 起算；间距 < 46 时两条带会叠在同一行像素上（渲染出来像一句乱码）。
    gap: 52,
    minAux: 130,
  });
  const heatRect = layout.main;
  const auxRect = layout.aux;
  const heatBudget = panelBudget(heatRect);
  const auxBudget = panelBudget(auxRect);

  // —— 热力图：把地板夹到 −16 并数出来 ——
  let nClamped = 0;
  const heatVals = new Float64Array(heat.values.length);
  for (let i = 0; i < heat.values.length; i += 1) {
    const v = heat.values[i] as number;
    if (!Number.isFinite(v)) {
      heatVals[i] = Number.NaN; // 渲染层画成"无数据"灰，并计入非有限值
      continue;
    }
    if (v <= HEAT_FLOOR) {
      heatVals[i] = HEAT_FLOOR;
      nClamped += 1;
    } else {
      heatVals[i] = v;
    }
  }
  const nRows = heat.rows; // = n − 1（次对角元下标 i）
  const nCols = heat.cols; // = iters + 1（迭代步）
  const xMax = Math.max(1, nCols - 1);

  scene.axes.push(
    makeAxis("main", heatRect, {
      xlim: [-0.5, xMax + 0.5],
      ylim: [-0.5, nRows - 0.5],
      xlabel: "QR 迭代步 $k$",
      ylabel: axisYlabelOf(AXES, "main", "次对角元下标 i"),
      title: axisTitleOf(AXES, "main", "log10|次对角元|（行 = i，列 = 迭代步）"),
      autoscaled: false,
      xticks: integerTicks(heatRect[2], nCols),
      yticks: Array.from({ length: nRows }, (_, i) => i).filter((i) => i % Math.max(1, Math.ceil(nRows / 8)) === 0),
    }),
  );
  scene.layers.push(
    heatLayer("main", heatVals, nCols, nRows, [-0.5, xMax + 0.5, -0.5, nRows - 0.5], {
      cmap: "viridis",
      vmin: HEAT_FLOOR,
      vmax: 0,
      clipBelow: HEAT_FLOOR,
    }),
  );

  const kNow = k === undefined ? nCols - 1 : Math.max(0, Math.min(Math.round(k), nCols - 1));
  scene.layers.push(
    lineLayer("main", series([kNow, kNow], [-0.5, nRows - 0.5], pal.cursor, { dash: [4, 3], width: 1.4, label: "当前步" })),
  );
  requireSameLength(cur, new Float64Array(nRows), "subdiag_current/n−1");
  const idx = Array.from({ length: nRows }, (_, i) => i);
  if (cur.length !== nRows) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 subdiag_current 长度 ${cur.length} ≠ n−1 = ${nRows}：画不出「当前步还活着哪些次对角元」`);
  }
  scene.layers.push(scatterLayer("main", idx.map(() => kNow), idx, { r: 3.4, color: pal.marker, label: "当前步" }));
  if (fin.length !== nRows) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 subdiag_final 长度 ${fin.length} ≠ n−1 = ${nRows}`);
  }
  scene.layers.push(
    scatterLayer("main", idx.map(() => xMax), idx, { r: 3.0, color: pal.reference, alpha: 0.9, label: "末步" }),
  );

  // —— 副轴：实测 Ritz 距离 + 理论衰减率 ——
  // 纵轴范围排除地板（`channelRange`），地板值换成轴底再画（见文件头第 4 条）。
  const chan = channelRange(err, [1e-3]);
  const theoryChan = channelRange(theory, [1]);
  const auxYlim = logRange(Math.min(chan.lo, theoryChan.lo, 1e-16), Math.max(chan.hi, theoryChan.hi, 1));
  scene.axes.push(
    makeAxis("aux", auxRect, {
      xlim: [0, xMax],
      ylim: auxYlim,
      yscale: "log",
      xlabel: "QR 迭代步 $k$（不是物理时间；拖到头会被夹住）",
      ylabel: axisYlabelOf(AXES, "aux", "$\\max_i\\min_j|\\theta_i-\\lambda_j|$"),
      title: axisTitleOf(AXES, "aux", "Ritz 值到真谱的距离"),
      autoscaled: true,
      xticks: integerTicks(auxRect[2], nCols),
    }),
  );

  const errMapped = replaceFloors(err, auxYlim[0]);
  const errSeries = segmentedSeries(steps, errMapped.values, pal.residual, {
    width: 1.8,
    marker: "o",
    accept: (v) => v > 0,
    label: "特征值匹配误差",
  });
  if (errSeries.length > 0) scene.layers.push(lineLayer("aux", errSeries));
  const theorySeries = segmentedSeries(steps, theory, pal.reference, { width: 1.3, dash: [6, 3], accept: (v) => v > 0, label: "理论收敛率" });
  if (theorySeries.length > 0) scene.layers.push(lineLayer("aux", theorySeries));
  scene.layers.push(
    lineLayer("aux", series([steps[kNow] ?? kNow, steps[kNow] ?? kNow], auxYlim, pal.cursor, { dash: [3, 3], width: 1, label: "当前步" })),
  );
  const markY = mark.length > 0 ? (mark[0] as number) : Number.NaN;
  const markDrawn = Number.isFinite(markY) && markY > 0 ? (atFloor(markY) ? auxYlim[0] : markY) : undefined;
  if (markDrawn !== undefined) {
    scene.layers.push(scatterLayer("aux", [steps[kNow] ?? kNow], [markDrawn], { r: 4, color: pal.marker, label: "当前误差" }));
  } else if (mark.length > 0) {
    degraded.push("error_mark 不是有限正值（NaN 或 ≤0）：当前步的标记不画，只在文本里报数");
  }

  // —— 文本 ——
  // 只把**数值**参数交给 paramSummary：`family`/`shift` 是帧的 `strings`（枚举），
  // 拿它们去 meta/请求参数里找会得到 "?"，那是把"没找到"当成"没这个参数"。
  const summary = paramSummary(frame, opts, [{ name: "n", label: "$n$" }]);
  degraded.push(...summary.degraded);
  const familyShift = `族=${family ?? "?"}　位移=${shift ?? "?"}`;

  const clampedNote =
    nClamped > 0
      ? `：${nClamped}/${heat.values.length} 个格点已到机器精度地板（帧里写 −300，这里夹到 ${HEAT_FLOOR}，是「已经压没了」不是「数据缺失」）`
      : "";
  const heatLines: string[] = [
    `${summary.text}　${familyShift}${nIter === undefined ? "" : `　帧说迭代 ${fmtSci(nIter, 0)} 步`}`,
    k === undefined ? "当前步：帧未给 meta.step" : `扫描线 = 第 ${Math.round(kNow)} 步${status === undefined ? "" : `；帧 status：${status}`}`,
    `颜色 = log10|次对角元|（色阶固定 [${HEAT_FLOOR}, 0]）${clampedNote}`,
    `纵轴只有 ${nRows} 行（次对角元只有 n−1 个）：收敛与否读 status/converged，不是读颜色`,
  ];
  scene.texts.push(...textBlock(heatRect[0] + 6, textTop(heatRect), textBottom(heatRect), heatLines, pal.ink, heatBudget, "top"));

  const auxLines: string[] = [
    `实测（实线+点）= Ritz 值到真谱的距离；理论（长虚线）= 无位移 QR 的 $|\\lambda_2/\\lambda_1|^k=${gap === undefined ? "?" : fmtSci(gap)}^k$。两条线量纲不同，要比的是斜率`,
  ];
  if (errCurrent !== undefined || errFinal !== undefined) {
    auxLines.push(
      `当前谱距离 ${errCurrent === undefined ? "?" : fmtSci(errCurrent)}、末步 ${errFinal === undefined ? "?" : fmtSci(errFinal)}` +
        (subInitMax !== undefined && subFinalMax !== undefined
          ? `；次对角元 max 从 ${fmtSci(subInitMax)} 压到 ${fmtSci(subFinalMax)}`
          : ""),
    );
  }
  if (converged !== undefined) {
    auxLines.push(
      converged > 0.5
        ? `帧判 converged：deflation ${nDefl === undefined ? "?" : fmtSci(nDefl, 0)} 次、${nBlocks === undefined ? "?" : fmtSci(nBlocks, 0)} 块`
        : "⚠ 帧判未收敛：曲线停在平台上不是「到了」，见 status",
    );
  }
  if (complexSpectrum) {
    auxLines.push("⚠ 谱里有复共轭对：它们只能停在 2×2 块上（Wilkinson 位移的正常行为），别读成「卡住了」");
  }
  if (note !== undefined) auxLines.push(`帧 note：${note}`);
  if (errMapped.nFloor > 0) {
    auxLines.push(
      `${errMapped.nFloor}/${err.length} 个 Ritz 距离在地板 ${fmtSci(LOG_FLOOR)} 上（帧里是「恰好 0」）：画在轴底 ${fmtSci(auxYlim[0])}，不是真的等于轴底`,
    );
  }
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, auxBudget, "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(sourceBadge(scene.dataSource), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "副轴：自适应（log 纵轴按本帧 Ritz 距离取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  const statusBits: string[] = [];
  if (status !== undefined) statusBits.push(status);
  if (converged !== undefined) statusBits.push(converged > 0.5 ? "帧判 converged" : "帧判未收敛");
  if (nDefl !== undefined) statusBits.push(`deflation ${fmtSci(nDefl, 0)} 次`);
  if (complexSpectrum) statusBits.push("谱含复共轭对");
  if (statusBits.length === 0) statusBits.push("帧未给出可判读的状态（预期有 strings.status 与 meta.converged）");
  scene.badges.push(placeBadge(badge("status", clipBadgeText(statusBits.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);

  // 规则 1：热力图地板夹取 + 非有限值 + 副轴地板，全部如实记账。
  const extra: string[] = [];
  if (nClamped > 0) {
    extra.push(`subdiag_heat 有 ${nClamped}/${heat.values.length} 个格点是地板 −300，已夹到色阶下限 ${HEAT_FLOOR}（「已压到机器精度」）`);
  }
  if (errMapped.nFloor > 0) extra.push(`error 有 ${errMapped.nFloor}/${err.length} 个样本是地板 ${fmtSci(LOG_FLOOR)}（恰好 0），画在轴底 ${fmtSci(auxYlim[0])}`);
  const nBad = countNonFinite(toNumeric(heat.values)) + countNonFinite(err) + countNonFinite(theory) + countNonFinite(cur) + countNonFinite(fin);
  applyOverflowNote(scene, extra.join("；"), nBad, { slot: SLOT.overflow, width });
  return scene;
};
