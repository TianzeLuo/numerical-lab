/**
 * `cancellation`（1.1 章）的 model：灾难性抵消——有效位一位一位消失。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `digits` | 当前损失位数的游标（虚线竖线） |
 * | 2 | line | `digits` | `digits_remaining`：先验剩余有效位数随损失位数 t 的下降（斜率 −1） |
 * | 3 | scatter | `digits` | `digits_mark`：当前 (loss_mark, digits_mark) |
 * | 4 | line | `error` | `error`：输入扰动被放大后的**实测**相对误差（实线） |
 * | 5 | line | `error` | `error_theory`：理论上界 δ_in(|a|+|b|)/|a−b|（虚线） |
 * | 6 | line | `error` | `roundoff`：减法**自身**的舍入误差（点线；Sterbenz 区里恒为 0 → 地板） |
 * | 7 | scatter | `error` | `error_mark`：当前实测误差 |
 * | 8 | scatter | `error` | `roundoff` 贴地板的点（全部 31 个：这就是「恒为 0」的样子） |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴固定 0..16 / 2 副轴自适应 / 3 临界（`input_digits`
 * 的 15.95 = log10(2^53)）/ 4 状态（note + Sterbenz + 时间轴口径）/ 5 前端预览 / 6 溢出 /
 * 7 降级说明。
 *
 * 几条刻意的决定
 * --------------
 * 1. **主轴固定 `[0, 16]`，刻度只落在整数上**：manifest 声明 `autoscaled: false`，
 *    标题写着"剩余有效位数（固定 0..16）"。定死之后"每抵消 10 倍少一位"就是一条**斜率 −1
 *    的直线**，不需要看图例；`xticks`/`yticks` 都按整数位给（这条轴只能整位读）。
 *    注意曲线本身是**先验估计**（`log10(|a−b|/(δ_in·M))`，帧里是连续值），
 *    "整数"是**读法**不是数据的量化——文本里写清这一点（pitfalls 2b）。
 * 2. **副轴上要有三条线，尤其是那条"贴地板"的**：爆炸的是**输入自带**的误差
 *    （`error`），理论上界是 `error_theory`，而减法自身的舍入误差 `roundoff` 恒为 0
 *    （Sterbenz 引理）——后端把它抬到地板 `u/2` 才画得出来。三条同屏、颜色区分，
 *    "减法不稳定"这条都市传说在图上当场被反驳。
 * 3. **时间轴 t 不进入本 viz 的计算**：帧的数据（数组与 meta）只由 a、b、input_digits、
 *    perturb_ulps 决定——实测 t=0 与 t=16 两帧的数组**逐位相同**（见交付说明里的
 *    "发现的 Python 层问题"）。所以模型把游标画在**实测**损失位数 `meta.loss_mark`
 *    上，而不是请求的 `opts.t`；文本里如实写明这条口径，不假装拖动时间轴换了数据。
 * 4. **`degenerate` 如实说明**：b 与 a 完全相等时后端取相邻的 1 个 ulp 代替并置
 *    `meta.degenerate`，否则 `log(0)` 会让整张图消失。模型把它写进状态角标与文本。
 */

import { formatNumber, niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import type { Frame } from "../../shared/src/frame.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
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
  stackedPanels,
  stringOf,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  overflowBadge,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  constantArray,
  floorIndices,
  integerTicks,
  pick,
  textBlock,
  textBottom,
  textTop,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.1";
const VIZ = "cancellation";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "灾难性抵消：有效位一位一位消失";

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  status: 4,
  /** 仅当帧声明 `backend+preview` 时出现（本章的帧都是 backend）。 */
  preview: 5,
  overflow: 6,
  degraded: 7,
} as const;

/** manifest 声明主轴固定 0..16（`beyond: "below"` 的临界值是 log10(2^53)=15.95）。 */
const DIGITS_LIM: [number, number] = [0, 16];
/** `frame.critical.input_digits` 缺失时的回落值（与 manifest 的 `critical[0]` 同源）。 */
const INPUT_DIGITS_CRITICAL = 15.95;

export const cancellationModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const lossGrid = requireArray(frame, "loss_grid");
  const digitsRemaining = requireArray(frame, "digits_remaining");
  const error = requireArray(frame, "error");
  const errorTheory = requireArray(frame, "error_theory");
  const roundoff = requireArray(frame, "roundoff");
  requireSameLength(lossGrid, digitsRemaining, "帧数组 loss_grid 与 digits_remaining");
  requireSameLength(lossGrid, error, "帧数组 loss_grid 与 error");
  requireSameLength(lossGrid, errorTheory, "帧数组 loss_grid 与 error_theory");
  requireSameLength(lossGrid, roundoff, "帧数组 loss_grid 与 roundoff");
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组（当前点）：缺了就降级，不猜 ——
  const degraded: string[] = [];
  const digitsMark = optionalFirst(frame, "digits_mark");
  const lossMark = optionalFirst(frame, "loss_mark");
  const errorMark = optionalFirst(frame, "error_mark");
  const roundoffMark = optionalFirst(frame, "roundoff_mark");
  if (lossMark === undefined) degraded.push("帧缺 loss_mark：游标不画（不拿请求 t 冒充损失位数）");
  if (digitsMark === undefined) degraded.push("帧缺 digits_mark：主轴的当前点不画");

  // —— meta ——
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const a = metaNumber(frame, "a");
  const b = metaNumber(frame, "b");
  const gap = metaNumber(frame, "gap");
  const deltaIn = metaNumber(frame, "delta_in");
  const inputDigits = metaNumber(frame, "input_digits");
  const digitsLost = metaNumber(frame, "digits_lost");
  const condSubtraction = metaNumber(frame, "cond_subtraction");
  const amplification = metaNumber(frame, "perturbation_amplification");
  const errorBound = metaNumber(frame, "error_bound");
  const subtractionRelErr = metaNumber(frame, "subtraction_relative_error");
  const nRoundoffZero = metaNumber(frame, "roundoff_exact_zero_steps");
  const perturbUlps = metaNumber(frame, "perturb_ulps");
  const degenerate = (metaNumber(frame, "degenerate") ?? 0) > 0.5;
  const frameOverflow = (metaNumber(frame, "overflow") ?? 0) > 0.5;
  const sterbenzNote = stringOf(frame, "sterbenz_note");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  if (frame.critical["input_digits"] === undefined) {
    degraded.push(`帧 critical 缺 input_digits：临界值回落到 manifest 声明的 ${INPUT_DIGITS_CRITICAL}（不是帧给的）`);
  }
  const exactA = stringOf(frame, "exact_a");
  const exactB = stringOf(frame, "exact_b");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const base = stackedPanels(width, height);
  const mainRect: Rect = base.main;
  const auxRect: Rect = base.error;

  // —— 主轴：固定 0..16 的整数位（manifest: autoscaled=false）——
  const tRange = finiteMinMax(lossGrid) ?? [1, 16];
  const tLo = tRange[0];
  const tHi = tRange[1];
  const mainAxis = makeAxis("digits", mainRect, {
    xlim: [Math.min(0, Math.floor(tLo)), Math.max(DIGITS_LIM[1], Math.ceil(tHi))],
    ylim: DIGITS_LIM,
    // 主轴 xlabel 置空（下方面板顶部要放说明文字，规则 8）；横轴含义写进图例。
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "digits", "位数"),
    title: axisTitleOf(SPEC?.axes, "digits", "剩余有效位数（固定 0..16）"),
    autoscaled: false,
    xticks: integerTicks(0, 16, 2),
    yticks: integerTicks(0, 16, 2),
  });

  // —— 副轴：三条曲线共用的 log 纵轴（manifest: autoscaled=true）——
  const bErr = finitePositiveMinMax(error);
  const bTheory = finitePositiveMinMax(errorTheory);
  const bRound = finitePositiveMinMax(roundoff);
  const lo = Math.min(bErr?.[0] ?? Number.POSITIVE_INFINITY, bTheory?.[0] ?? Number.POSITIVE_INFINITY, bRound?.[0] ?? Number.POSITIVE_INFINITY);
  const hi = Math.max(bErr?.[1] ?? 0, bTheory?.[1] ?? 0, bRound?.[1] ?? 0);
  if (!(Number.isFinite(lo) && Number.isFinite(hi))) {
    degraded.push("error / error_theory / roundoff 里没有 > 0 的有限值：副轴用占位范围 [1e-18, 1]，不画曲线");
  }
  const auxYlim: [number, number] =
    Number.isFinite(lo) && Number.isFinite(hi) ? niceLimits(lo, hi, 7, { log: true }) : [1e-18, 1];
  const auxAxis = makeAxis("error", auxRect, {
    xlim: mainAxis.xlim,
    ylim: auxYlim,
    yscale: "log",
    xlabel: "损失位数 t",
    ylabel: axisYlabelOf(SPEC?.axes, "error", "相对误差"),
    title: axisTitleOf(SPEC?.axes, "error", "结果的相对误差（0 以 u/2 显示）"),
    autoscaled: true,
    xticks: niceTicks(mainAxis.xlim[0], mainAxis.xlim[1], 7),
  });
  scene.axes.push(mainAxis, auxAxis);

  // —— 图层 1：游标（画在**实测**损失位数上；t 不驱动本帧的数据，见文件头第 3 条）——
  const tNow = lossMark ?? clamp(opts.t, mainAxis.xlim[0], mainAxis.xlim[1]);
  scene.layers.push(
    lineLayer(
      "digits",
      series([tNow, tNow], DIGITS_LIM, pal.cursor, { width: 1, dash: [3, 3], label: "当前损失位数（帧 meta.loss_mark）" }),
    ),
    lineLayer("error", series([tNow, tNow], auxAxis.ylim, pal.cursor, { width: 1, dash: [3, 3] })),
  );

  // —— 图层 2：先验剩余有效位数（斜率 −1 的直线）——
  const digitsSeries = segmentedSeries(lossGrid, digitsRemaining, pal.curves[0] ?? "#2b6cb0", {
    width: 1.9,
    label: "先验剩余有效位数 log10(|a−b|/(δ_in·M))：每抵消 10 倍少一位",
  });
  if (digitsSeries.length > 0) scene.layers.push(lineLayer("digits", digitsSeries));
  else degraded.push("digits_remaining 没有可绘的有限值：主轴画不出下降曲线");

  // —— 图层 3：当前点 ——
  if (digitsMark !== undefined && lossMark !== undefined) {
    scene.layers.push(scatterLayer("digits", [lossMark], [digitsMark], { r: 4.5, color: pal.marker, label: "当前 (t, 剩余位数)" }));
  }

  // —— 图层 4~8：副轴的三条曲线 + 两个标记 ——
  const errSeries = segmentedSeries(lossGrid, error, pal.residual, {
    width: 2,
    label: "实测：输入扰动被放大后的相对误差（把 a、b 各推 ulp 个 ulp 实测）",
    accept: (v) => v > 0,
  });
  if (errSeries.length > 0) scene.layers.push(lineLayer("error", errSeries));
  else degraded.push("error 的样本没有一个 > 0：log 轴上画不出实测曲线");

  const theorySeries = segmentedSeries(lossGrid, errorTheory, pal.curves[2] ?? "#b7791f", {
    width: 1.5,
    dash: [7, 4],
    label: "理论上界 δ_in(|a|+|b|)/|a−b|",
    accept: (v) => v > 0,
  });
  if (theorySeries.length > 0) scene.layers.push(lineLayer("error", theorySeries));
  else degraded.push("error_theory 的样本没有一个 > 0：log 轴上画不出理论上界");

  const roundSeries = segmentedSeries(lossGrid, roundoff, pal.curves[1] ?? "#2f855a", {
    width: 1.5,
    dash: [2, 3],
    label: "减法**自身**的舍入误差（Sterbenz 区里恒为 0，被地板抬起）",
    accept: (v) => v > 0,
  });
  if (roundSeries.length > 0) scene.layers.push(lineLayer("error", roundSeries));
  else degraded.push("roundoff 的样本没有一个 > 0：log 轴上画不出这条对照曲线");

  const errorFloor = metaNumber(frame, "error_floor");
  const floorVec = errorFloor === undefined ? undefined : constantArray(roundoff.length, errorFloor);
  const onFloor = floorVec === undefined ? [] : floorIndices(roundoff, floorVec);
  if (onFloor.length > 0) {
    // 用同一把地板尺子标出"贴地板"的点：这里 31 个点全都贴在 u/2 上（减法精确）。
    const p = pick(lossGrid, roundoff, onFloor);
    scene.layers.push(
      scatterLayer("error", p.x, p.y, { r: 3, color: pal.curves[1] ?? "#2f855a", alpha: 0.9, label: `减法舍入误差贴地板的 ${onFloor.length} 个点（实测 0）` }),
    );
  }
  if (errorMark !== undefined && lossMark !== undefined && errorMark > 0) {
    scene.layers.push(scatterLayer("error", [lossMark], [errorMark], { r: 4.5, color: pal.marker, label: "当前实测放大后相对误差" }));
  }
  if (roundoffMark !== undefined && lossMark !== undefined && roundoffMark > 0) {
    scene.layers.push(scatterLayer("error", [lossMark], [roundoffMark], { r: 4, color: pal.curves[1] ?? "#2f855a", alpha: 0.8 }));
  }

  // —— 文本 ——
  const summary = paramSummary(frame, opts, [
    { name: "a", label: "a" },
    { name: "b", label: "b" },
    { name: "input_digits", label: "输入位数 d" },
  ]);
  degraded.push(...summary.degraded);

  const head: string[] = [summary.text];
  if (a !== undefined) head.push(`a=${formatNumber(a, 8)}`);
  if (b !== undefined) head.push(`b=${formatNumber(b, 8)}`);
  if (gap !== undefined) head.push(`|a−b|=${formatNumber(gap, 3)}`);
  if (deltaIn !== undefined) head.push(`δ_in=${formatNumber(deltaIn, 3)}`);
  if (perturbUlps !== undefined) head.push(`扰动=${formatNumber(perturbUlps, 4)} ulp`);
  const mainBudget = unitBudget(mainRect[2]);
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), [head.join("　")], pal.muted, mainBudget, "top"));

  // 三行"怎么读副轴"放在主轴**右上方那块空白**里：曲线（斜率 −1）从左上角下来，
  // 右下三角才是被占的，右上三角永远是空的。
  const wide = mainRect[2] >= 560;
  const midX = mainRect[0] + Math.round(mainRect[2] * 0.56);
  const midBudget = unitBudget(mainRect[2] * 0.44);
  const mid: string[] = [
    "红线=输入误差被放大（会爆炸）",
    "橙虚=理论上界 δ_in(|a|+|b|)/|a−b|",
    "绿点线=减法**自身**的舍入误差：Sterbenz 区内恒为 0",
    "爆炸的从来不是减法，是输入里本来就有的相对误差 δ_in",
  ];
  // 窄画布（<560px）上主面板只剩一列：右侧那块空白不存在，硬放就会与图例叠字。
  // 这时把同样的内容并进下方图例（全宽），由 textBlock 自己截断。
  if (wide) {
    scene.texts.push(...textBlock(midX, mainRect[1] + 30, textBottom(mainRect), mid, pal.muted, midBudget, "top"));
  }

  const legend: string[] = wide ? [] : [...mid];
  if (digitsLost !== undefined || digitsMark !== undefined) {
    legend.push(
      `当前：损失 ${digitsLost === undefined ? "?" : formatNumber(digitsLost, 4)} 位、` +
        `剩 ${digitsMark === undefined ? "?" : formatNumber(digitsMark, 4)} 位` +
        (condSubtraction === undefined ? "" : `（减法条件数 ${formatNumber(condSubtraction, 4)}）`),
    );
  }
  legend.push("横轴=损失位数 t = log10(M/|a−b|)　纵轴=剩余有效位数（固定 0..16，刻度按整数位读；纵轴是先验估计，不是对某一次计算的误差测量）");
  legend.push(
    "时间轴口径：t 不进入本帧的计算（实测 t=0 与 t=16 两帧的数组逐位相同）——" +
      `游标画在**实测**损失位数 ${lossMark === undefined ? "（缺 loss_mark）" : formatNumber(lossMark, 4)} 上，不假装 t 就是它`,
  );
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), legend, pal.muted, mainBudget, "bottom"));

  // 副轴只留 4 行（帧 note + 当前数值 + Sterbenz + 输入的精确十进制）：曲线才看得见。
  const auxLines: string[] = [];
  if (note !== undefined) auxLines.push(`帧 note：${note}`);
  const nums: string[] = [];
  if (amplification !== undefined) nums.push(`实测放大比=${formatNumber(amplification, 3)}`);
  if (errorBound !== undefined) nums.push(`理论上界=${formatNumber(errorBound, 3)}`);
  if (subtractionRelErr !== undefined) nums.push(`减法自身舍入误差=${formatNumber(subtractionRelErr, 1)}（精确 0）`);
  if (nRoundoffZero !== undefined) nums.push(`${onFloor.length}/${roundoff.length} 个采样点全为 0`);
  if (nums.length > 0) auxLines.push(nums.join("　"));
  if (sterbenzNote !== undefined) auxLines.push(sterbenzNote);
  if (exactA !== undefined && exactB !== undefined) auxLines.push(`输入的精确十进制：a=${exactA}　b=${exactB}`);
  if (degenerate) auxLines.push("⚠ b 与 a 完全相等：后端用相邻的 1 个 ulp 代替 b（否则 log(0) 会让整张图消失），帧已置 degenerate");
  const auxBudget = unitBudget(auxRect[2] * 0.72);
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, auxBudget, "top"));

  // 游标标签**不画**：底部那一行是底部锚定的说明块（规则 8 的"文本压文本"两次抓到
  // 21×14px 的重叠），顶部又已有"输入位数"一行；数值本身在说明块与角标里都写了，
  // 只留竖虚线做视觉指认（不为了多一个数字去挤掉一句话）。

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "副轴：自适应（log 纵轴按本帧三条误差曲线取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  if (inputDigits !== undefined) {
    scene.badges.push(
      placeBadge(
        criticalBadge("input_digits", inputDigits, criticalOf(frame, "input_digits", INPUT_DIGITS_CRITICAL), { beyond: "below" }),
        SLOT.critical,
        width,
      ),
    );
  }
  const status: string[] = [];
  if (degenerate) status.push("degenerate：b 已用 a 的前一个 ulp 代替（帧 meta）");
  status.push("抵消放大的是**输入**误差：Sterbenz 区内减法本身精确（实测 0）");
  if (note !== undefined) status.push(note);
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(lossGrid) +
    countNonFinite(digitsRemaining) +
    countNonFinite(error) +
    countNonFinite(errorTheory) +
    countNonFinite(roundoff);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  if (frameOverflow) {
    const note2 = [scene.overflow.note, "帧 meta.overflow=true：放大后的相对误差已经 > 1（结果一位有效数字都不剩）"]
      .filter((s) => s.length > 0)
      .join("；");
    scene.overflow = { flag: true, note: note2 };
    if (!scene.badges.some((b) => b.kind === "overflow")) {
      scene.badges.push(placeBadge(overflowBadge(note2), SLOT.overflow, width));
    }
  }
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** 取"长度 1 的数组"的第 0 个元素；缺数组、空数组、非有限值一律 undefined（降级，不猜）。 */
function optionalFirst(frame: Frame, name: string): number | undefined {
  if (!frame.has(name)) return undefined;
  const a = requireArray(frame, name);
  if (a.length === 0) return undefined;
  const v = a[0] as number;
  return Number.isFinite(v) ? v : undefined;
}
