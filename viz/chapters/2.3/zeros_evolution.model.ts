/**
 * `zeros_evolution`（2.3 章）的 model：三族正交多项式的零点随阶数演化（分岔图）+ 根误差。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 理论密度曲线（**横向**画在横轴左侧 20%，见文件头第 2 条） |
 * | 2 | line | `main` | 相邻根间距 `gap`（横轴 = 间距编号 `gap_index`，与阶数同刻度） |
 * | 3 | scatter | `main` | `root_contrast`：同阶 Chebyshev 零点仿射映射后的对照（灰） |
 * | 4 | scatter | `main` | 全部零点 `(degree, root)`（分岔图的主体） |
 * | 5 | scatter | `main` | 当前阶的全部零点 `root_current`（高亮） |
 * | 6 | line | `main` | 当前阶的游标竖线 |
 * | 7 | line | `resid` | 舍入地板 u（对数轴上的读数基准） |
 * | 8 | scatter | `resid` | 每个根的 Newton 根误差 `(degree, error)` |
 * | 9 | scatter | `resid` | 当前阶那些根的误差（高亮） |
 *
 * 角标槽位（固定，见 `common.ts::SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 /
 * 3 定理参照（零点互异、全在区间内、与低一阶交错）/ 4 状态（误差量是哪一个）/
 * 5 前端预览 / 6 溢出 / 7 降级。
 *
 * 这个 viz 最容易骗人的两处，以及它们在图上的防法
 * ------------------------------------------------
 * 1. **「误差」是哪一个？** 帧里的 `error` 是 **Newton 修正量 |p_n/p_n′|**（与多项式自身
 *    尺度无关），**不是**多项式残差 |p_n(x_k)|。同一个根，Laguerre n=40 时
 *    |L_n| ≈ 5.6e14 而 Newton 修正量 ≈ 1.2e-14——相差 28 个数量级（笔记陷阱 §4/§10）。
 *    防法：副轴标题、轴名、角标与文字块四处都写明"误差量 = |p_n/p_n′|"，并把帧自带的
 *    `strings.error_note` 原文照登；文字里同时点出"残差量 |L_n| 不在本帧里"（不假装画了）。
 * 2. **「零点看起来差不多是等距的」**。Legendre n=24 相邻根间距 max/min = 6.26 倍、
 *    Laguerre n=40 是 92.3 倍（meta 实测）。防法：主面板把 `gap` 曲线画出来（间距在中部
 *    最大、靠近端点最小），并把 `gap_ratio_current` / `gap_min` / `gap_max` 三个数写进文字；
 *    再用理论密度曲线（横向）说明"零点该往哪里挤"。
 *
 * 几条刻意的决定
 * --------------
 * 1. **密度曲线横向画**：它是以零点位置为自变量的函数，而主面板的横轴是**阶数**。把密度
 *    映射到横轴左侧 20% 的宽度上（向右=越密）、纵轴仍取零点位置，这样"哪些位置零点更密"
 *    与分岔图能对齐；文字里写明"这 20% 宽度上的横坐标读数不是阶数"。
 * 2. **n=1 的退化**：一个根、没有间距（`gap` 是空数组）、跨度 lo=hi。这时不画 gap 与密度
 *    （没有可画的），纵轴扩成 [lo−1, lo+1] 并**明说**是退化扩展，不是实测跨度。
 * 3. **误差轴的纵轴用对数**（manifest `resid.scale="log"`）：`error` 已被后端按 u 截断，
 *    所以恒有 > 0；落到地板的点会在地板线上排成一行——那正是"已经算到机器精度"的样子，
 *    数出有多少个点在地板上（`clipped_points`）并写在图上。
 * 4. **时间轴是"最高阶数 n"**，不是时间：以帧 `meta.step` 为准（$n_{\max}$ 之外被夹住），
 *    夹取与不一致都写在图上。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
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
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  CHAPTER,
  SLOT,
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

const VIZ = "zeros_evolution";
const TITLE = specTitle(VIZ, "正交多项式零点的演化");
/** 时间轴兜底上限（manifest `time.tMax`）：只在帧缺 `n_current`/`step` 时用。 */
const T_MAX = 40;
/** 密度曲线占主面板横轴宽度的比例（横向映射，见文件头第 1 条）。 */
const DENSITY_SHARE = 0.2;

export const zerosEvolutionModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const degree = requireArray(frame, "degree");
  const root = requireArray(frame, "root");
  const error = requireArray(frame, "error");
  const rootContrast = requireArray(frame, "root_contrast");
  requireSameLength(degree, root, "帧数组 degree 与 root");
  requireSameLength(degree, error, "帧数组 degree 与 error");
  requireSameLength(degree, rootContrast, "帧数组 degree 与 root_contrast");
  // `root_current` 与 `degree`/`root` 同属"画出来的东西"：缺了就没有当前阶的高亮，
  // 时间轴 scrub 也就失去了可见目标 → 归入必需数组（缺了就抛，不画半张图）。
  const rootCurrentRaw = requireArray(frame, "root_current");
  const gap = requireArray(frame, "gap");
  const gapIndex = requireArray(frame, "gap_index");
  requireSameLength(gap, gapIndex, "帧数组 gap 与 gap_index");
  const pitfalls = requirePitfalls(frame);

  // —— 可选/降级字段 ——
  const degraded: string[] = [];
  const nCurrent = metaNumber(frame, "n_current");
  const nRoots = metaNumber(frame, "n_roots_current");
  const maxRootError = metaNumber(frame, "max_root_error");
  const minRootError = metaNumber(frame, "min_root_error");
  const floor = metaNumber(frame, "root_error_floor");
  const clippedPoints = metaNumber(frame, "clipped_points");
  const iterMax = metaNumber(frame, "newton_iter_max");
  const iterMean = metaNumber(frame, "newton_iter_mean");
  const gapRatio = metaNumber(frame, "gap_ratio_current");
  const gapMin = metaNumber(frame, "gap_min_current");
  const gapMax = metaNumber(frame, "gap_max_current");
  const spanLo = metaNumber(frame, "root_span_lo");
  const spanHi = metaNumber(frame, "root_span_hi");
  if (floor === undefined) degraded.push("帧 meta 缺 root_error_floor：地板线不画（不猜一个地板出来）");
  if (maxRootError === undefined) degraded.push("帧 meta 缺 max_root_error：最大根误差留「?」，不按数组现算");
  if (clippedPoints === undefined) degraded.push("帧 meta 缺 clipped_points：不声称有多少个根误差落在地板上");
  if (spanLo === undefined || spanHi === undefined) degraded.push("帧 meta 缺 root_span_lo/hi：跨度读数留「?」（纵轴仍按数组实测值画）");
  const family = stringOf(frame, "family");
  const familyLabel = stringOf(frame, "family_label");
  const spacingNote = noteOf(frame, "spacing_note");
  const errorNote = noteOf(frame, "error_note");
  const densityNote = noteOf(frame, "density_note");

  const { step, notes: stepNotes } = currentStep(frame, opts, {
    lo: 1,
    hi: Math.max(1, Math.round(nCurrent ?? T_MAX)),
    label: "最高阶数 $n$",
  });
  const nCur = step ?? Math.max(1, Math.round(nCurrent ?? 1));
  if (nCurrent !== undefined && Math.round(nCurrent) !== nCur) {
    degraded.push(`帧 meta.n_current=${fmt(nCurrent)} 与 meta.step=${nCur} 不一致：以 step 为准`);
  }

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

  // —— 主轴：零点位置 vs 阶数 ——
  const yData = minMaxOfAll([root, rootContrast]) ?? [0, 1];
  // 退化判据：帧给了 root_span_lo/hi 就以它为准（n=1 时 lo=hi 是**精确**相等）；
  // 没给才退回数据跨度——数据跨度可能只是 ±1e-16 的浮点噪声，肉眼上同样是退化。
  const spanDegenerate =
    spanLo !== undefined && spanHi !== undefined ? !(spanHi > spanLo) : !(yData[1] - yData[0] > 0);
  const yMainRaw: [number, number] = spanDegenerate
    ? [yData[0] - 1, yData[0] + 1]
    : [yData[0], yData[1]];
  const ylimMain = niceLimits(yMainRaw[0], yMainRaw[1], 6, { pad: 0.06 });
  const xlimMain: [number, number] = [0.4, Math.max(2.6, nCur + 0.6)];
  scene.axes.push(
    makeAxis("main", base.main, {
      xlim: xlimMain,
      ylim: ylimMain,
      xlabel: "阶数 $n$",
      ylabel: axisYlabel(VIZ, "main", "零点 x"),
      title: axisTitle(VIZ, "main", "零点位置 vs 阶数"),
      autoscaled: false,
      xticks: niceTicks(xlimMain[0], xlimMain[1], 8).filter((v) => Number.isInteger(v)),
      yticks: niceTicks(ylimMain[0], ylimMain[1], 5),
    }),
  );

  // 1) 理论密度：横向映射到横轴左侧 20%（y = 零点位置，x = 密度大小）
  const densityX = frame.has("density_x") ? requireArray(frame, "density_x") : undefined;
  const densityY = frame.has("density_y") ? requireArray(frame, "density_y") : undefined;
  const densityRange = densityY === undefined ? undefined : minMaxOfAll([densityY]);
  if (densityX === undefined || densityY === undefined || densityRange === undefined || !(densityRange[1] > 0)) {
    degraded.push("帧缺 density_x/density_y（或密度全为 0）：不画理论密度曲线（不拿近似形状顶替）");
  } else {
    const span = xlimMain[1] - xlimMain[0];
    const mapped = new Float64Array(densityX.length);
    for (let i = 0; i < densityX.length; i += 1) {
      const d = densityY[i] as number;
      const v = Number.isFinite(d) && d > 0 ? d / densityRange[1] : 0;
      mapped[i] = xlimMain[0] + DENSITY_SHARE * span * Math.min(1, Math.max(0, v));
    }
    scene.layers.push(
      lineLayer("main", segmentedSeries(mapped, densityX, pal.residual, { width: 1.5, label: "理论密度（横向：向右=越密）" })),
    );
  }

  // 2) 相邻根间距：横轴用间距编号（与阶数同刻度），纵轴同为 x 单位
  if (gap.length > 0) {
    scene.layers.push(
      lineLayer("main", segmentedSeries(gapIndex, gap, pal.muted, { width: 1.3, label: "相邻根间距 gap" })),
    );
  }

  // 3) Chebyshev 对照零点（仿射映射后的解析闭式）
  scene.layers.push(
    scatterLayer("main", degree, rootContrast, { r: 1.8, color: pal.reference, alpha: 0.85, label: "同阶 Chebyshev 零点（对照）" }),
  );
  // 4) 全部零点
  scene.layers.push(scatterLayer("main", degree, root, { r: 2.1, color: pal.curves[0], label: "零点 $(n, x_k)$" }));
  // 5) 当前阶的零点 + 6) 游标
  const degCurrent = new Float64Array(rootCurrentRaw.length);
  for (let i = 0; i < rootCurrentRaw.length; i += 1) degCurrent[i] = nCur;
  scene.layers.push(scatterLayer("main", degCurrent, rootCurrentRaw, { r: 3.2, color: pal.marker, label: `当前阶 n=${nCur} 的零点` }));
  scene.layers.push(
    lineLayer("main", segmentedSeries([nCur, nCur], [ylimMain[0], ylimMain[1]], pal.cursor, { width: 1.2, dash: [4, 4], label: "当前阶" })),
  );

  // —— 副轴：Newton 根误差（对数轴） ——
  const nonPositiveResid = countAtOrBelow(error, 0);
  const errRange = minMaxOfAll([error]) ?? [floor ?? 1e-16, 1];
  const lo = floor ?? (errRange[0] > 0 ? errRange[0] : 1e-16);
  const ylimResid = niceLimits(lo > 0 ? lo : 1e-16, Math.max(errRange[1], lo * 10), 4, { log: true, pad: 0 });
  scene.axes.push(
    makeAxis("resid", base.error, {
      xlim: xlimMain,
      ylim: ylimResid,
      yscale: "log",
      xlabel: "阶数 $n$",
      ylabel: axisYlabel(VIZ, "resid", "根误差"),
      title: axisTitle(VIZ, "resid", "Newton 根误差 |p_n/p_n′|"),
      autoscaled: true,
      xticks: niceTicks(xlimMain[0], xlimMain[1], 8).filter((v) => Number.isInteger(v)),
      yticks: niceTicks(ylimResid[0], ylimResid[1], 5, { log: true }),
    }),
  );
  if (floor !== undefined) {
    scene.layers.push(
      lineLayer("resid", segmentedSeries([xlimMain[0], xlimMain[1]], [floor, floor], pal.warn, { width: 1.2, dash: [2, 3], label: `舍入地板 u = ${sci(floor)}` })),
    );
  }
  scene.layers.push(scatterLayer("resid", degree, error, { r: 1.8, color: pal.curves[1], label: "每个根的 $|p_n/p_n′$|" }));
  const curDegX: number[] = [];
  const curDegY: number[] = [];
  for (let i = 0; i < degree.length; i += 1) {
    if (Math.round(degree[i] as number) === nCur && Number.isFinite(error[i] as number)) {
      curDegX.push(degree[i] as number);
      curDegY.push(error[i] as number);
    }
  }
  if (curDegX.length > 0) {
    scene.layers.push(scatterLayer("resid", curDegX, curDegY, { r: 3, color: pal.marker, label: `当前阶 n=${nCur} 的根误差` }));
  }

  // —— 文本 ——
  const mainLines: string[] = [
    `族：${familyLabel ?? family ?? "?"}　时间轴 = 最高阶数 n = ${nCur}（meta.step = ${fmtOpt(step, 3)}）　本阶零点 ${fmtOpt(nRoots, 4)} 个`,
    `蓝点 = 零点 (n, x_k)　灰点 = 同阶 Chebyshev 零点（仿射映射后的解析闭式对照）　` +
      `紫线 = 理论密度（**横向**画在横轴左侧 ${Math.round(DENSITY_SHARE * 100)}%：越靠右越密，这段横坐标读数不是阶数）`,
    gap.length > 0
      ? `灰线 = 相邻根间距 gap（横轴用间距编号 1…${gap.length}，与阶数同刻度；纵轴同为 x 单位）：` +
        `max/min = ${fmtOpt(gapRatio, 4)}（min ${fmtOpt(gapMin, 4)}，max ${fmtOpt(gapMax, 4)}）——零点不是等距的`
      : "本阶只有一个根：没有相邻间距可画（gap 是空数组），所以主面板没有间距曲线",
    spanDegenerate
      ? `⚠ 本阶零点跨度退化（lo = hi = ${fmtOpt(spanLo, 4)}）：纵轴按 ±1 扩开才画得出来——这是**扩展**，不是实测跨度`
      : `零点跨度：${fmtOpt(spanLo, 4)} … ${fmtOpt(spanHi, 4)}（meta.root_span_lo/hi）`,
  ];
  if (spacingNote !== undefined) mainLines.push(spacingNote);
  mainLines.push(...stepNotes);
  scene.texts.push(...textBlock(base.main[0] + 6, base.main[1] + 4, base.main[1] + base.main[3] - 4, mainLines, pal.note, unitBudget(base.main[2]), "bottom"));

  const residLines: string[] = [
    `误差量 = **Newton 修正量 |p_n/p_n′|**（与多项式自身尺度无关），不是多项式残差 |p_n(x_k)|；本帧 max ` +
      `${maxRootError === undefined ? "?" : sci(maxRootError)}、min ${minRootError === undefined ? "?" : sci(minRootError)}、地板 u = ${floor === undefined ? "?" : sci(floor)}`,
    `同一个根的两个"误差"能差 28 个数量级（Laguerre n=40：|L_n| ≈ 5.6e14 vs Newton 修正量 1.2e-14）；` +
      `**残差量 |p_n(x_k)| 不在本帧里**——本页只能画 Newton 修正量`,
    `落在地板（≤ u）上的根误差 ${fmtOpt(clippedPoints, 4)} 个（共 ${degree.length} 个根）；` +
      `Newton 迭代次数 max/mean = ${fmtOpt(iterMax, 3)} / ${fmtOpt(iterMean, 4)}`,
  ];
  if (errorNote !== undefined) residLines.push(errorNote);
  if (densityNote !== undefined) residLines.push(densityNote);
  if (nonPositiveResid > 0) {
    residLines.push(`⚠ 有 ${nonPositiveResid} 个根误差 ≤ 0：对数轴画不到，已跳过（**不是**当成 0 画在轴上）`);
  }
  scene.texts.push(...textBlock(base.error[0] + 6, base.error[1] + 4, base.error[1] + base.error[3] - 4, residLines, pal.note, unitBudget(base.error[2]), "bottom"));

  // —— 角标 ——
  scene.badges.push(placeBadge(sourceBadge(scene.dataSource), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(placeBadge(badge("autoscale", "副轴 resid：自适应对数轴（根误差可跨十几个数量级）", 8, 26), SLOT.auxAxis, width));
  scene.badges.push(
    placeBadge(badge("critical", "定理：n 个零点互异、全在区间内，且与低一阶的零点交错（gap 曲线与之一致）"), SLOT.critical, width),
  );
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `误差量 = |p_n/p_n′|（不是 |p_n|）；max = ${maxRootError === undefined ? "?" : sci(maxRootError)}，地板 u = ${fmtOpt(floor === undefined ? undefined : floor, 3)}`,
      ),
      SLOT.status,
      width,
    ),
  );
  if (scene.dataSource.startsWith("backend+preview")) scene.badges.push(placeBadge(previewBadge(), SLOT.preview, width));

  const rawNonFinite = countNonFinite(degree) + countNonFinite(root) + countNonFinite(error) + countNonFinite(rootContrast) + countNonFinite(gap);
  applyOverflow(scene, rawNonFinite, { slot: SLOT.overflow, width });
  const degradedObj = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedObj) scene.badges.push(degradedObj);

  requireDrawableLayers(scene);
  return scene;
};
