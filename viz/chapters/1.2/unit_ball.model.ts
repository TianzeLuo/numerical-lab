/**
 * `unit_ball`（1.2 章）的 model：单位圆的三个像 + 前向误差随 κ 的实测曲线。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 单位圆（灰虚线参照） |
 * | 2 | line | `main` | `A` 的像（|det A| = 面积比） |
 * | 3 | line | `main` | `L` 的像（单位下三角：只剪切，det = 1） |
 * | 4 | line | `main` | `U` 的像（上三角：把圆压成细椭圆） |
 * | 5 | line | `main` | 向量 `b`（箭头：杆 + 两支箭头 + 端点圆点） |
 * | 6 | line | `main` | 数值解 `x̂` |
 * | 7 | line | `main` | 精确解 `x` |
 * | 8 | line | `aux` | `error` 随 `kappa_grid`（log 纵轴；在第一个不可解点处截断） |
 * | 9 | scatter | `aux` | 当前位置标记（`kappa_mark`, `error_mark`） |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴固定 ±3 / 2 副轴自适应 / 3 状态（κ、有效位）/
 * 4 溢出（帧 `meta.overflow` + 模型自己发现的"向量冲出画面"）/ 5 前端预览（仅 `backend+preview`）/
 * 7 降级说明。
 *
 * 几条刻意的决定
 * --------------
 * 1. **主轴固定在 ±3 且 `main` 是正方形**：单位圆必须看起来是圆的，否则"圆被压成椭圆"
 *    这件事本身就失真了。`stackedPanels` 给的主面板是宽扁的，所以这里用 `columnsLayout`
 *    的左列 + 自算方形（纯显示几何，与帧无关）。固定 ±3 是 manifest 的声明（规则 1/4）：
 *    越界要被标出来，而不是靠缩轴把 κ 的破坏力藏起来。
 * 2. **4 元"两段折线"是 `[x0, x1, y0, y1]`**：`b_vec`/`x_num`/`x_exact` 的排布是
 *    前半段给 x、后半段给 y（帧自带的 label 就写着"两段折线"，后端注释也说明它由
 *    `[0, bx, 0, by]` 拼成），因此 `x = [v[0], v[1]]`、`y = [v[2], v[3]]` 正是
 *    "从原点指向 (bx, by)"的那一段；模型把它画成带箭头的杆，而不是把 4 个数当成 4 个点。
 *    若按交错 (x,y) 读，会得到 (0,bx)→(0,by) 这条贴在 y 轴上的竖线——与 `b=(cosθ,sinθ)` 矛盾。
 * 3. **冲出画面的向量按方向裁到轴边，并挂溢出角标**：κ 大时 `x̂`/`x` 的真值在 1e15 量级
 *    （帧里就有），直接交给渲染层会被"离轴超过 1e6 像素"的过滤整段丢掉——图上什么都没有、
 *    账上也没有，正是规则 1 禁止的。所以模型自己算与轴框的交点、画到边上，并把真实端点写进
 *    数值行与溢出说明。`meta.overflow` 只覆盖"单位圆的像"是否越界（实际构造下恒为 false，
 *    像只会变小变细），向量的越界由模型这一侧负责补上。
 * 4. **帧里没有 NaN，改用"占位符 + 掩码"**（帧契约要求所有数组有限）：某 κ 上后端算不出解时，
 *    `error[i]=0` 是**占位符**、`error_valid[i]=0` 才是事实本身；`x_num_valid=0` 同理。
 *    模型必须①按掩码把曲线截断（占位 0 在对数轴上会画成"误差掉到地板"的假象）、
 *    ②奇异时不画 `x̂` 箭头（否则会画出"解在原点"的假箭头）、③在文本里把两个数分开报。
 * 5. **不画 κu 理论线**：那需要前端自己算 κ·u（P8 禁止前端造物理量）。帧 `strings.note`
 *    已经给出"实测前向误差 vs κu"的对比，模型只搬运，不重算。
 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import type { LineSeries } from "../../shared/src/scene.js";
import {
  ModelInputError,
  applyOverflow,
  canvasSize,
  clipBadgeText,
  dataSourceOf,
  degradedBadge,
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
  sceneShell,
  segmentedSeries,
  stackedPanels,
  stringOf,
  textWidthUnits,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  lineLayer,
  makeAxis,
  overflowBadge,
  pitfallsBand,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  text,
  type TextSpec,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.2";
const VIZ = "unit_ball";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "单位球的形变：行列式、条件数与解";

/**
 * 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。
 * 长文本角标（溢出、降级）各占一行的左列，右侧留空，避免与邻居叠字。
 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  status: 3,
  /** 仅当帧声明 `backend+preview` 时出现（本章的帧都是 backend）。 */
  preview: 4,
  overflow: 5,
  degraded: 7,
} as const;

/** manifest 声明主轴"固定坐标"：单位圆与三个像都画在这个框里。 */
const MAIN_LIM: [number, number] = [-3, 3];

/** 一条向量（帧里的 4 元数组解出来的"从原点出发"的那一段）。 */
interface Arrow {
  label: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** 端点是否在 ±3 之外（裁过）。 */
  clipped: boolean;
}

export const unitBallModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const circleX = requireArray(frame, "circle_x");
  const circleY = requireArray(frame, "circle_y");
  const imageX = requireArray(frame, "image_x");
  const imageY = requireArray(frame, "image_y");
  const lX = requireArray(frame, "l_x");
  const lY = requireArray(frame, "l_y");
  const uX = requireArray(frame, "u_x");
  const uY = requireArray(frame, "u_y");
  const bVec = requireArray(frame, "b_vec");
  const xNum = requireArray(frame, "x_num");
  const xExact = requireArray(frame, "x_exact");
  const kappaGrid = requireArray(frame, "kappa_grid");
  const error = requireArray(frame, "error");
  // 掩码（可选）：后端把"这个 κ 在浮点下不可解"写成占位符 0 + `error_valid[i]=0`。
  // 缺掩码时退化成"error>0 即有效"（老帧），并在降级角标里说明——**不猜**。
  const errorValid = optionalArray(frame, "error_valid");
  const errorMask = (i: number): boolean => (errorValid === undefined ? error[i]! > 0 : (errorValid[i] ?? 0) > 0.5);
  const errorMark = requireArray(frame, "error_mark");
  const kappaMark = requireArray(frame, "kappa_mark");
  requireSameLength(circleX, circleY, "帧数组 circle_x 与 circle_y");
  requireSameLength(imageX, imageY, "帧数组 image_x 与 image_y");
  requireSameLength(lX, lY, "帧数组 l_x 与 l_y");
  requireSameLength(uX, uY, "帧数组 u_x 与 u_y");
  requireSameLength(kappaGrid, error, "帧数组 kappa_grid 与 error");
  for (const [name, arr] of [
    ["b_vec", bVec],
    ["x_num", xNum],
    ["x_exact", xExact],
  ] as const) {
    if (arr.length !== 4) {
      throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 '${name}' 长度 ${arr.length} ≠ 4：向量按 [x0,x1,y0,y1] 编码（见文件头第 2 条）`);
    }
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const shear = metaNumber(frame, "shear");
  if (shear === undefined) degraded.push("帧 meta 缺 shear：参数摘要用请求值并标注");
  const log10Kappa = metaNumber(frame, "log10_kappa");
  if (log10Kappa === undefined) degraded.push("帧 meta 缺 log10_kappa：不显示 κ（不按 t 猜）");
  const kappa = metaNumber(frame, "kappa");
  const kappaSigma = metaNumber(frame, "kappa_sigma");
  const sigmaMin = metaNumber(frame, "sigma_min");
  const sigmaMax = metaNumber(frame, "sigma_max");
  const kappaMeasured = metaNumber(frame, "kappa_measured");
  const detA = metaNumber(frame, "det_a");
  const detL = metaNumber(frame, "det_l");
  const detU = metaNumber(frame, "det_u");
  const detError = metaNumber(frame, "det_error");
  const forwardError = metaNumber(frame, "forward_error");
  const backwardError = metaNumber(frame, "backward_error");
  const residual = metaNumber(frame, "residual");
  const nTheta = metaNumber(frame, "n_theta");
  const frameOverflow = (metaNumber(frame, "overflow") ?? 0) > 0.5;
  const singular = (metaNumber(frame, "singular") ?? 0) > 0.5;
  const xNumValid = errorValid === undefined ? !singular : (metaNumber(frame, "x_num_valid") ?? (singular ? 0 : 1)) > 0.5;
  if (singular && xNumValid) {
    degraded.push("帧 meta.singular=true 但 x_num_valid 不是 0：按 singular 处理（占位 0 不画成解）");
  }
  const pivotName = stringOf(frame, "pivot");
  if (pivotName === undefined) degraded.push("帧 strings 缺 pivot：不显示主元策略名（不按参数猜）");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左侧方形画几何，右侧画误差曲线 ——
  // 为什么不用 `columnsLayout`：它的左列高度是 `stackedPanels` 的主面板高度
  // （`mainShare` 缺省 0.56 → 960×600 下只有 211px），单位圆会被挤成一个小方块。
  // 这里直接拿内容区的 top/bottom 自己算方形，**圆必须看起来是圆的**（见文件头第 1 条）。
  // 底部 pitfalls 的文案长度会变：把内容底边抬到它的上方，再给轴下说明留 42px。
  // 不这么做时，改一段长文案就会让最后一行说明压进 pitfalls 面板（规则 8 会当场拒绝）。
  const band = pitfallsBand({ canvas: { w: width, h: height }, pitfalls });
  const footer = band === null ? 102 : Math.max(102, height - band[1] + 42);
  const base = stackedPanels(width, height, { footer });
  const side = Math.max(110, Math.floor(Math.min(base.bottom - base.top, base.contentWidth * 0.5)));
  const mainRect: Rect = [base.left, base.top, side, side];
  // 右列留出它自己的 y 轴刻度与 ylabel（render2d 把 ylabel 画在轴左侧 42px 处）。
  const auxX = base.left + side + 74;
  const auxRect: Rect = [auxX, base.top, Math.max(160, base.left + base.contentWidth - auxX), base.bottom - base.top];

  const mainAxis = makeAxis("main", mainRect, {
    xlim: MAIN_LIM,
    ylim: MAIN_LIM,
    xlabel: "x",
    ylabel: axisYlabel("main", "y"),
    title: axisTitle("main", "单位圆与它的三个像（固定坐标）"),
    autoscaled: false,
  });

  // 副轴（manifest 声明 autoscaled: true、scale: log）：前向误差随 log10 κ。
  const errPos = finitePositiveMinMax(error);
  const markPos = finitePositiveMinMax(errorMark);
  const auxLo = Math.min(errPos?.[0] ?? Number.POSITIVE_INFINITY, markPos?.[0] ?? Number.POSITIVE_INFINITY);
  const auxHi = Math.max(errPos?.[1] ?? Number.NEGATIVE_INFINITY, markPos?.[1] ?? Number.NEGATIVE_INFINITY);
  const auxYlim: [number, number] =
    Number.isFinite(auxLo) && Number.isFinite(auxHi) ? niceLimits(auxLo, auxHi, 7, { log: true }) : [1e-16, 1];
  if (!(Number.isFinite(auxLo) && Number.isFinite(auxHi))) {
    degraded.push("error/error_mark 里没有 >0 的有限值：副轴用占位范围 [1e-16, 1]，不画曲线");
  }
  const kappaRange = finiteMinMaxOr(kappaGrid, [0, 17]);
  const auxAxis = makeAxis("aux", auxRect, {
    xlim: niceLimits(kappaRange[0], kappaRange[1], 7),
    ylim: auxYlim,
    xscale: "linear",
    yscale: "log",
    xlabel: "log10 κ",
    ylabel: axisYlabel("aux", "相对误差"),
    title: axisTitle("aux", "前向误差 ‖x̂−x‖/‖x‖ 随 κ"),
    autoscaled: true,
  });
  scene.axes.push(mainAxis, auxAxis);

  // —— 图层 1~4：单位圆与三个像 ——
  const images: { x: ArrayLike<number>; y: ArrayLike<number>; color: string; label: string; dash?: number[]; width: number }[] = [
    { x: circleX, y: circleY, color: pal.reference, label: "单位圆 S¹（参照）", dash: [5, 4], width: 1.1 },
    { x: imageX, y: imageY, color: pal.curves[0], label: "A 的像（面积 = |det A|）", width: 1.8 },
    { x: lX, y: lY, color: pal.curves[1], label: "L 的像（det L = 1，纯剪切）", width: 1.4 },
    { x: uX, y: uY, color: pal.curves[2], label: "U 的像（细到这个尺度就看不见了）", width: 1.4 },
  ];
  for (const img of images) {
    const trimmed = trimNonFinite(img.x, img.y);
    if (trimmed === undefined) {
      degraded.push(`像曲线（${img.label}）没有可绘的有限点：跳过该层`);
      continue;
    }
    scene.layers.push(lineLayer("main", series(trimmed.x, trimmed.y, img.color, { width: img.width, dash: img.dash, label: img.label })));
  }

  // —— 图层 5~7：b、x̂、x 三条向量（箭头；冲出画面的按方向裁到轴边）——
  const arrows: { arrow: Arrow; color: string; dash?: number[] }[] = [
    { arrow: toArrow("b（右端项）", bVec), color: pal.note, dash: [7, 4] },
    { arrow: toArrow("x（精确解）", xExact), color: pal.ink, dash: [3, 3] },
  ];
  if (xNumValid) {
    arrows.splice(1, 0, { arrow: toArrow("x̂（数值解）", xNum), color: pal.marker });
  } else {
    // 奇异时 x̂ 是占位 0：画出来会得到一个"解在原点"的假箭头，必须明确不画。
    degraded.push("x̂ 在该参数下没有有限值（浮点奇异）：不画 x̂ 箭头，位置由状态角标与错位说明给出");
  }
  const clipped: string[] = [];
  for (const item of arrows) {
    const drawn = clipArrow(item.arrow, MAIN_LIM);
    if (drawn.clipped) {
      clipped.push(
        `${item.arrow.label.split("（")[0]} 的真实端点 (${formatNumber(item.arrow.x1, 3)}, ${formatNumber(item.arrow.y1, 3)})`,
      );
    }
    scene.layers.push(lineLayer("main", arrowSeries({ ...item.arrow, x1: drawn.x1, y1: drawn.y1 }, item.color, item.dash)));
  }

  // —— 图层 8~9：副轴（规则 3 的误差场）——
  // 不可解点（掩码 0）与"误差恰好为 0"是两件事，分开计数；
  // 曲线**在第一个不可解点处截断**：占位 0 在对数轴上会被画成"误差掉到地板"的假象。
  let nInvalid = 0;
  let nZero = 0;
  let firstInvalid = -1;
  let prefix = -1; // 第一个不可解点**之前**的有效点数：曲线的可画前缀
  const kappaOk: number[] = [];
  const errOk: number[] = [];
  for (let i = 0; i < error.length; i += 1) {
    const v = error[i] ?? Number.NaN;
    if (!errorMask(i) || !Number.isFinite(v)) {
      nInvalid += 1;
      if (firstInvalid < 0) {
        firstInvalid = i;
        prefix = kappaOk.length;
      }
      continue;
    }
    if (v === 0) nZero += 1;
    if (v > 0) {
      kappaOk.push(kappaGrid[i] ?? Number.NaN);
      errOk.push(v);
    }
  }
  if (errorValid === undefined) {
    degraded.push("帧里没有 error_valid 掩码：按「error>0 即有效」处理（老帧），不可解点无法与精确 0 区分");
  }
  // 只画"第一个不可解点之前"的连续前缀：跳过中间的洞继续画，会读成"这段 κ 上误差仍然很小"。
  const drawCount = prefix < 0 ? kappaOk.length : prefix;
  const kappaDraw = errorValid === undefined ? kappaOk : kappaOk.slice(0, drawCount);
  const errDraw = errorValid === undefined ? errOk : errOk.slice(0, drawCount);
  const errSeries = segmentedSeries(kappaDraw, errDraw, pal.residual, {
    width: 1.6,
    label: "实测前向误差（每个 κ 都真跑一次 LU + 回代）",
    accept: (v) => v > 0,
  });
  if (errSeries.length > 0) scene.layers.push(lineLayer("aux", errSeries));
  else degraded.push(`帧 error 的 ${error.length} 个样本没有一个 > 0：log 轴上画不出曲线`);
  const markX = kappaMark[0];
  const markY = errorMark[0];
  const markValid = (metaNumber(frame, "error_mark_valid") ?? (Number.isFinite(markY) && markY > 0 ? 1 : 0)) > 0.5;
  if (markValid && Number.isFinite(markX) && Number.isFinite(markY) && markY > 0) {
    scene.layers.push(scatterLayer("aux", [markX], [markY], { r: 4.5, color: pal.marker, label: "当前 log10 κ 处的误差" }));
  } else {
    degraded.push(`当前位置标记不可画（kappa_mark=${String(markX)}、error_mark=${String(markY)}）：只画曲线`);
  }

  // —— 文本（一行一个 text 层：render2d 不拆 \n，见 lu_heatmap 文件头第 6 条）——
  const summary = paramSummary(frame, opts, [
    { name: "shear", label: "剪切 s" },
    { name: "b_angle", label: "b 方向(度)" },
  ]);
  degraded.push(...summary.degraded);
  const head: string[] = [summary.text];
  if (pivotName !== undefined) head.push(`主元=${pivotName}`);
  if (log10Kappa !== undefined) head.push(`log10 κ=${formatNumber(log10Kappa, 4)}`);
  if (kappa !== undefined) head.push(`κ=${formatNumber(kappa, 3)}`);
  if (forwardError !== undefined) head.push(`前向误差=${formatNumber(forwardError, 3)}`);
  const mainBudget = unitBudget(mainRect[2]);
  const headSpecs = textBlock(mainRect[0] + 6, mainRect[1] + 4, mainRect[1] + 4, [head.join("　")], pal.muted, mainBudget, "top");
  scene.texts.push(...headSpecs);

  const legend: string[] = [
    "像：灰虚=单位圆　蓝=A　绿=L　橙=U（面积比给出 |det|）",
    "向量：蓝虚=b　红=x̂（数值解）　黑虚=x（精确解）；箭头按方向画到轴边",
  ];
  if (clipped.length > 0) legend.push(`⚠ ${clipped.length} 条向量已冲出 ±3：${clipped.join("；")}——看不到不等于不存在`);
  scene.texts.push(
    ...textBlock(
      mainRect[0] + 6,
      mainRect[1] + 4 + headSpecs.length * 15 + 6,
      mainRect[1] + mainRect[3] - 30,
      legend,
      clipped.length > 0 ? pal.warn : pal.muted,
      mainBudget,
      "bottom",
    ),
  );

  const nums: string[] = [];
  if (detA !== undefined) nums.push(`det A=${formatNumber(detA, 4)}`);
  if (detL !== undefined && detU !== undefined) nums.push(`det L·det U=${formatNumber(detL * detU, 4)}`);
  if (detError !== undefined) nums.push(`|det 误差|=${formatNumber(detError, 3)}`);
  if (kappaSigma !== undefined) nums.push(`κ_σ=σmax/σmin=${formatNumber(kappaSigma, 4)}`);
  if (kappaMeasured !== undefined) nums.push(`采样实测 κ=${formatNumber(kappaMeasured, 4)}（会系统性高估 σmin）`);
  if (sigmaMin !== undefined && sigmaMax !== undefined) {
    nums.push(`σ ∈ [${formatNumber(sigmaMin, 3)}, ${formatNumber(sigmaMax, 3)}]`);
  }
  if (nTheta !== undefined) nums.push(`n_θ=${formatNumber(nTheta, 6)}`);
  const auxBudget = unitBudget(auxRect[2]);
  let auxTop = auxRect[1] + 4;
  if (nums.length > 0) {
    const numSpecs = textBlock(auxRect[0] + 6, auxTop, auxRect[1] + auxRect[3] - 6, [nums.join("　")], pal.muted, auxBudget, "top");
    scene.texts.push(...numSpecs);
    auxTop += numSpecs.length * 15;
  }
  const auxLines: string[] = [];
  if (note !== undefined) auxLines.push(`帧 note：${note}`);
  auxLines.push(
    `误差曲线 ${error.length} 点：${nInvalid} 个不可解（error_valid=0，占位 0）、${nZero} 个恰好为 0` +
      (firstInvalid >= 0 ? `；曲线在 log10 κ=${formatNumber(kappaGrid[firstInvalid] ?? Number.NaN, 3)} 处截断` : "") +
      "——「没有有限值」与「误差为零」是两件事，分开计数",
  );
  if (backwardError !== undefined || residual !== undefined) {
    auxLines.push(
      `后向误差 ${formatNumber(backwardError ?? Number.NaN, 3)}、残差 ${formatNumber(residual ?? Number.NaN, 3)}（帧 meta；缺的显示为 NaN，不猜）`,
    );
  }
  auxLines.push("把 κ 拖到 1e16 附近：曲线右端抬高是 κu ~ 1 的直接后果，而不是绘图或插值的产物。");
  scene.texts.push(...textBlock(auxRect[0] + 6, auxTop, auxRect[1] + auxRect[3] - 6, auxLines, pal.muted, auxBudget, "top"));

  // —— 角标（固定槽位）——
  // 规则 5：恒挂 source；预览型数据源另挂 preview（kind 各表一意，见 scene.ts）。
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "误差轴：自适应（log 纵轴按本帧 error 取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  const status: string[] = [];
  if (log10Kappa !== undefined) {
    // 帧的 `strings.note` 里通常已经带了前向误差（后端的 `_ball_note` 就是这么写的），
    // 模型再写一遍会得到"前向误差 1.6e-8；前向误差 1.6e-8：与 κu 同量级"这种复读。
    // 有 note 时只报 κ，把误差与解释交给 note；没 note 时才自己补上数值。
    status.push(
      log10Kappa >= 16
        ? `κu ≫ 1（log10 κ=${formatNumber(log10Kappa, 3)}）：解已无有效位`
        : `log10 κ=${formatNumber(log10Kappa, 3)}` +
            (note === undefined ? `：前向误差 ${formatNumber(forwardError ?? Number.NaN, 3)}` : ""),
    );
  }
  if (note !== undefined) status.push(note);
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 meta.log10_kappa 与 strings.note）");
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  // 规则 1：图层越界 + 帧内非有限值 + "帧说像越界了" + "模型发现向量越界"，四者都要进 overflow。
  const nBad =
    countNonFinite(circleX) +
    countNonFinite(circleY) +
    countNonFinite(imageX) +
    countNonFinite(imageY) +
    countNonFinite(lX) +
    countNonFinite(lY) +
    countNonFinite(uX) +
    countNonFinite(uY) +
    countNonFinite(bVec) +
    countNonFinite(xNum) +
    countNonFinite(xExact) +
    countNonFinite(kappaGrid) +
    countNonFinite(error) +
    countNonFinite(errorMark) +
    countNonFinite(kappaMark);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  const extra: string[] = [];
  if (frameOverflow) extra.push("帧 meta.overflow=true：单位圆的像已经跑出固定的 ±3 坐标范围（看不到不等于不存在）");
  if (clipped.length > 0) extra.push(`${clipped.join("；")}——按方向裁到轴边显示`);
  if (extra.length > 0) {
    const note2 = clipBadgeText([scene.overflow.note, ...extra].filter((s) => s.length > 0).join("；"));
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

/** 4 元向量 `[x0, x1, y0, y1]` → 从原点出发的箭头（见文件头第 2 条）。 */
function toArrow(label: string, v: ArrayLike<number>): Arrow {
  const x0 = v[0];
  const x1 = v[1];
  const y0 = v[2];
  const y1 = v[3];
  if (![x0, y0, x1, y1].every((t) => Number.isFinite(t))) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的向量 '${label.split("（")[0]}' 含非有限分量：[${x0}, ${x1}, ${y0}, ${y1}]`);
  }
  return { label, x0, y0, x1, y1, clipped: false };
}

/** 箭头裁到固定坐标框里：原点在框内，取与框边最近的那个正向交点（纯显示几何）。 */
function clipArrow(a: Arrow, lim: [number, number]): { x1: number; y1: number; clipped: boolean } {
  const dx = a.x1 - a.x0;
  const dy = a.y1 - a.y0;
  let t = 1;
  const consider = (num: number, den: number): void => {
    if (den === 0 || !Number.isFinite(den)) return;
    const s = num / den;
    if (s > 0 && s < t) t = s;
  };
  if (dx > 0) consider(lim[1] - a.x0, dx);
  if (dx < 0) consider(lim[0] - a.x0, dx);
  if (dy > 0) consider(lim[1] - a.y0, dy);
  if (dy < 0) consider(lim[0] - a.y0, dy);
  const clipped = t < 1 - 1e-12;
  // 交点上再向内收一丝（1e-9 相对量），免得贴着轴框的坐标被判成"越界"。
  const shrink = clipped ? 1 - 1e-9 : 1;
  return { x1: a.x0 + dx * t * shrink, y1: a.y0 + dy * t * shrink, clipped };
}

/** 一条箭头 = 杆 + 两支箭头 + 端点圆点（4 条 series；圆点用 `marker:"o"` 画）。 */
function arrowSeries(a: Arrow, color: string, dash?: number[]): LineSeries[] {
  const dx = a.x1 - a.x0;
  const dy = a.y1 - a.y0;
  const len = Math.hypot(dx, dy);
  const out: LineSeries[] = [series([a.x0, a.x1], [a.y0, a.y1], color, { width: 2, dash, label: a.label })];
  if (len > 0) {
    const span = MAIN_LIM[1] - MAIN_LIM[0];
    const head = Math.min(len * 0.35, span * 0.045);
    const ux = dx / len;
    const uy = dy / len;
    const bx = a.x1 - head * ux;
    const by = a.y1 - head * uy;
    const px = -uy * head * 0.45;
    const py = ux * head * 0.45;
    out.push(
      series([a.x1, bx + px], [a.y1, by + py], color, { width: 2 }),
      series([a.x1, bx - px], [a.y1, by - py], color, { width: 2 }),
      series([a.x1], [a.y1], color, { width: 2.4, marker: "o" }),
    );
  }
  return out;
}

/** 去掉非有限点（保留顺序；有缺口就跳过整条，返回 undefined 让调用方记一条降级）。 */
function trimNonFinite(x: ArrayLike<number>, y: ArrayLike<number>): { x: Float64Array; y: Float64Array } | undefined {
  const xs: number[] = [];
  const ys: number[] = [];
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i += 1) {
    const xv = x[i];
    const yv = y[i];
    if (Number.isFinite(xv) && Number.isFinite(yv)) {
      xs.push(xv);
      ys.push(yv);
    }
  }
  if (xs.length === 0) return undefined;
  return { x: Float64Array.from(xs), y: Float64Array.from(ys) };
}

/** 有限值的 min/max；一个有限值都没有时用回落范围。 */
function finiteMinMaxOr(a: ArrayLike<number>, fallback: [number, number]): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : fallback;
}

/** manifest 里该轴的标题（文案直接取 manifest，缺省才用回落值）。 */
function axisTitle(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel。 */
function axisYlabel(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.ylabel ?? fallback;
}

/**
 * 一段说明文字 → 若干 `text` 层，**一行一个**（`render2d.ts::drawText` 不拆 `\n`）。
 *
 * 为什么要自己折行：两个渲染器都支持 `maxWidth` 自动折行，但折出来的续行**不占**模型这一侧的
 * 行距预算（光栅器的续行间距是 `cell_h + 4`，比这里排的 15px 大），续行会压到下一行文字上。
 * 所以模型先按"显示宽度单位"折到 `budget` 以内（`model.ts::textWidthUnits`：CJK=2、ASCII=1），
 * 再逐行排开：一行就是一层，两个渲染器都不会再折。
 *
 * `anchor`：`"top"` 从 `top` 往下排；`"bottom"` 让整块贴着 `bottom`。返回的数组长度 = 实际行数。
 */
function textBlock(
  x: number,
  top: number,
  bottom: number,
  lines: readonly string[],
  color: string,
  budget: number,
  anchor: "top" | "bottom",
): TextSpec[] {
  const size = 11;
  const rowH = 15;
  let pieces = lines.flatMap((body) => wrapUnits(body, budget));
  // 画布小的时候宁可**截断并说明**，也不让文字叠在一起（叠字等于看不见，见规则 1 的精神）：
  // 能放下的行数由 top/bottom 之间的净空决定，最后一行留给"还有 N 行"。
  if (bottom > top) {
    const maxRows = Math.max(1, Math.floor((bottom - top) / rowH) + 1);
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - (pieces.length - 1) * rowH);
  return pieces.map((body, i) => text(x, y + i * rowH, body, { role: "note", size, color, maxWidth: budget * 7 }));
}

/**
 * 按显示宽度单位折行（同一把尺子：`model.ts::textWidthUnits`，CJK=2、ASCII=1）。
 *
 * 先按空白（含全角空格）切"词"再往行里塞：`ρ=0.9991`、`k=3/63` 这类字段不会被从中间劈开
 * （劈开在光栅器里读起来像两个数）。单个词比预算还长时才硬切。
 */
function wrapUnits(body: string, budget: number): string[] {
  const out: string[] = [];
  let line = "";
  let used = 0;
  const push = (): void => {
    if (line !== "") out.push(line);
    line = "";
    used = 0;
  };
  for (const word of body.split(/(?<=[\s　])/)) {
    const w = textWidthUnits(word);
    if (used > 0 && used + w > budget) push();
    if (w <= budget) {
      line += word;
      used += w;
      continue;
    }
    let piece = "";
    let pieceUsed = 0;
    for (const ch of word) {
      const cw = textWidthUnits(ch);
      if (pieceUsed + cw > budget && piece !== "") {
        out.push(piece);
        piece = "";
        pieceUsed = 0;
      }
      piece += ch;
      pieceUsed += cw;
    }
    line = piece;
    used = pieceUsed;
  }
  push();
  return out;
}

/**
 * 面板宽度 → 显示宽度单位预算（size 11 时 1 单位 ≈ 7px 光栅单元）。
 * 留 12px 边距再折算，保证两个渲染器都不会自动折行（见 `textBlock` 的注释）。
 */
function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}
