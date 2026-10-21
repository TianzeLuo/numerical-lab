/**
 * `convergence_order`（4.1 章 viz 2）的 model：双对数误差阶梯 + 最细一级的逐点误差。
 *
 * 图层（顺序即绘制顺序）
 * ----------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | loglogfit | `ladder` | 实测误差阶梯 + **实测斜率**与**理论斜率**两条参考（规则 2） |
 * | 2 | line | `ladder` | 「对连续解」的对照阶梯（虚线）——换靶子的后果当场可见 |
 * | 3 | scatter | `ladder` | 最细一级的 (Δh, 误差) 高亮点 |
 * | 4 | line | `profile` | 最细一级的逐点误差（对数轴）+ 地板线 |
 *
 * 时间轴语义：manifest `time.mode="none"`——这个 viz **没有时间轴**，一次性扫完整条阶梯，
 * 所以模型里不出现任何 `tMax` / 时间控件（PROTOCOLS §5 的字段约束）。
 *
 * 防误导七条里本章的重点
 * ----------------------
 * * **规则 2**：`loglogfit` 同时给 `measured_order`（帧里拟合出来的）与 `theory_order`
 *   （θ 族的理论阶），两条线都画、都标；理论线的锚点在**最细一级**（与帧的
 *   `theory_line` 数组同一口径）。
 * * **规则 4**：`ladder` 是**固定窗口**（`autoscaled=false`，范围由本帧全部曲线取整后
 *   钉住，不随 t 变——本 viz 本来就没有时间轴）；`profile` 是自适应，如实挂
 *   `autoscale` 角标并说明"会掩盖量级变化"。
 * * **规则 7**：最容易读反的一条写进面板文字与状态角标——**靶子选错会得到相反的结论**：
 *   时间细化必须对半离散解测。对连续解测时 CN 的实测斜率塌到 0 附近（空间误差把时间
 *   误差盖住了），看起来像"格式不收敛"，其实是靶子的问题（`meta.order_vs_continuous`）。
 */

import { formatNumber, niceLimits, niceTicks } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  dataSourceOf,
  degradedBadge,
  finitePositiveMinMax,
  metaNumber,
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
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  lineLayer,
  loglogFitLayer,
  makeAxis,
  pitfallsBand,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";
import {
  arrayLabelOf,
  axisAutoscaledOf,
  axisTitleFitted,
  axisYlabelOf,
  bandLayout,
  clip,
  endLabel,
  endLabelOf,
  contentBox,
  hLine,
  splitColumns,
  textBlock,
} from "./_panel.js";

const CHAPTER = "4.1";
const VIZ = "convergence_order";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "收敛阶：空间细化与时间细化（以及选错靶子的后果）";
/** 帧 `strings.floor`（对数轴地板）缺失时的回落值，与后端 `_ERR_FLOOR` 同值。 */
const FLOOR_FALLBACK = 1e-17;

const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  status: 3,
  overflow: 4,
  preview: 5,
  degraded: 6,
} as const;

export const convergenceOrderModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const hLadder = requireArray(frame, "h_ladder");
  const errorLadder = requireArray(frame, "error_ladder");
  const errorContinuous = requireArray(frame, "error_ladder_continuous");
  const xProfile = requireArray(frame, "x_profile");
  const errorProfile = requireArray(frame, "error_profile");
  requireSameLength(hLadder, errorLadder, "帧数组 h_ladder 与 error_ladder");
  requireSameLength(hLadder, errorContinuous, "帧数组 h_ladder 与 error_ladder_continuous");
  requireSameLength(xProfile, errorProfile, "帧数组 x_profile 与 error_profile");
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const markH = optional(frame, "mark_h");
  const markErr = optional(frame, "mark_error");
  if (markH === undefined || markErr === undefined) {
    degraded.push("帧缺数组 mark_h/mark_error：不画最细一级的高亮点");
  }
  const theoryLine = optional(frame, "theory_line");
  if (theoryLine === undefined) degraded.push("帧缺数组 theory_line：理论斜率线仍由 loglogfit 层承担（规则 2）");

  // —— 标量 meta（缺了只少一条标注，不抛错）——
  const theoryOrder = metaNumber(frame, "theory_order");
  const measured = metaNumber(frame, "measured_order");
  if (theoryOrder === undefined) degraded.push("帧 meta 缺 theory_order：不给理论斜率（不猜阶数）");
  if (measured === undefined) degraded.push("帧 meta 缺 measured_order：不报实测斜率");
  const stderr = metaNumber(frame, "order_stderr");
  const r2 = metaNumber(frame, "r2");
  const orderContinuous = metaNumber(frame, "order_vs_continuous");
  const nUsed = metaNumber(frame, "n_used");
  const nDropped = metaNumber(frame, "n_dropped");
  const errFinest = metaNumber(frame, "err_finest");
  const errCoarsest = metaNumber(frame, "err_coarsest");
  const errRatio = metaNumber(frame, "err_ratio");
  const orderGap = metaNumber(frame, "order_gap");
  const fixedR = metaNumber(frame, "fixed_r");
  const isSpace = frame.meta["target_is_space"] === true;

  const scheme = stringOf(frame, "scheme");
  const target = stringOf(frame, "target");
  const reference = stringOf(frame, "reference");
  const floor = numberFromString(stringOf(frame, "floor")) ?? FLOOR_FALLBACK;
  // 帧的 ladder_note / reading_note / trap_note 是完整句子（74/146/114 个单位）：
  // 面板按通栏 6 行排版，塞不下三段散文；它们的内容在下面按"数字 + 最短判读"重述，
  // 完整口径留在帧里（strings）与底部 pitfalls 面板。
  const grad = paramSummary(frame, opts, [
    { name: "nx", label: "$nx$" },
    { name: "t_end", label: "$T$" },
  ]);
  degraded.push(...grad.degraded);

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：两块面板并排，底部通栏数值块（见 `_panel.ts::bandLayout`）——
  const box = contentBox(width, height, pitfallsBand(scene)?.[1] ?? null);
  const cols = splitColumns(box, 0.62);
  const band = bandLayout(box, { botH: 0, rows: 7 });
  const ladderRect: [number, number, number, number] = [cols.left.left, band.topY, cols.left.width, band.topH];
  const profileRect: [number, number, number, number] = [cols.right.left, band.topY, cols.right.width, band.topH];

  // —— 轴：ladder 双对数固定窗口；profile 对数纵轴自适应 ——
  const hPos = finitePositiveMinMax(hLadder);
  const allErr = concatPositive(errorLadder, errorContinuous, theoryLine);
  const errPos = finitePositiveMinMax(allErr);
  const ladderXlim: [number, number] =
    hPos === undefined ? [1e-3, 1] : niceLimits(hPos[0] / 1.6, hPos[1] * 1.6, 5, { log: true });
  // 阶梯面板不画地板线：范围只由数据定（把 1e-17 的地板也算进来会白占四个数量级）。
  const ladderYlim: [number, number] =
    errPos === undefined ? [floor, 1] : niceLimits(errPos[0] / 1.8, errPos[1] * 1.8, 6, { log: true });

  const ladderAxis = makeAxis("ladder", ladderRect, {
    xlim: ladderXlim,
    ylim: ladderYlim,
    xscale: "log",
    yscale: "log",
    ylabel: axisYlabelOf(SPEC, "ladder", "L2 误差"),
    title: axisTitleFitted(SPEC, "ladder", "误差 vs 步长（双对数，固定窗口）", ladderRect[2]),
    autoscaled: axisAutoscaledOf(SPEC, "ladder", false),
    xticks: niceTicks(ladderXlim[0], ladderXlim[1], 5, { log: true }),
    yticks: niceTicks(ladderYlim[0], ladderYlim[1], 7, { log: true }),
  });

  const profPos = finitePositiveMinMax(errorProfile);
  const profileYlim: [number, number] =
    profPos === undefined ? [floor, 1] : niceLimits(Math.min(profPos[0], floor), profPos[1], 6, { log: true });
  const xProfLo = xProfile.length > 0 ? (xProfile[0] as number) : 0;
  const xProfHi = xProfile.length > 0 ? (xProfile[xProfile.length - 1] as number) : 1;
  const profileAxis = makeAxis("profile", profileRect, {
    xlim: Number.isFinite(xProfLo) && Number.isFinite(xProfHi) && xProfHi > xProfLo ? [xProfLo, xProfHi] : [0, 1],
    ylim: profileYlim,
    yscale: "log",
    ylabel: axisYlabelOf(SPEC, "profile", "误差绝对值"),
    title: axisTitleFitted(SPEC, "profile", "最细一级的逐点误差（对数轴）", profileRect[2]),
    autoscaled: axisAutoscaledOf(SPEC, "profile", true),
  });
  scene.axes.push(ladderAxis, profileAxis);

  // —— 图层 1：loglogfit（实测阶梯 + 实测斜率线 + 理论斜率线，规则 2）——
  if (measured !== undefined && theoryOrder !== undefined) {
    scene.layers.push(
      loglogFitLayer("ladder", hLadder, errorLadder, measured, theoryOrder, {
        seriesLabel: `实测误差阶梯（${reference ?? "靶子见文本"}）`,
        theoryLabel: `理论斜率 −${formatNumber(theoryOrder, 2)}（锚在最细一级）`,
        color: pal.curves[0],
        theoryColor: pal.reference,
      }),
    );
  } else {
    const fallback = segmentedSeries(hLadder, errorLadder, pal.curves[0], {
      width: 1.8,
      label: endLabelOf(hLadder, ladderXlim, ladderRect[2], arrayLabelOf(frame, "error_ladder", "实测误差阶梯")),
    });
    if (fallback.length > 0) scene.layers.push(lineLayer("ladder", fallback));
    degraded.push("缺 measured_order/theory_order：改画实测阶梯，规则 2 的两条斜率线无法标注");
  }

  // —— 图层 2：对连续解的对照阶梯（换靶子的后果）——
  // 不挂末点标签：`loglogfit` 层已经在面板右上角写了两行标签，再来一行会在同一处叠字；
  // 这条对照阶梯的名字交给底部通栏的图例行（`endLabel` 在双对数轴上本来也常判"画不下"）。
  const control = segmentedSeries(hLadder, errorContinuous, pal.exact, {
    width: 1.3,
    dash: [6, 3],
    accept: (v) => v > 0,
  });
  if (control.length > 0) scene.layers.push(lineLayer("ladder", control));
  else degraded.push("帧 error_ladder_continuous 没有 > 0 的有限值：对数轴上画不出对照阶梯");

  // —— 图层 3：最细一级的高亮点 ——
  if (markH !== undefined && markErr !== undefined && markH.length > 0 && markErr.length > 0) {
    const mx = markH[0] as number;
    const my = markErr[0] as number;
    if (Number.isFinite(mx) && mx > 0 && Number.isFinite(my) && my > 0) {
      scene.layers.push(
        scatterLayer("ladder", [mx], [my], {
          r: 4,
          color: pal.marker,
          label: endLabel(mx, ladderXlim, ladderRect[2], "最细一级", { log: true }),
        }),
      );
    }
  }

  // —— 图层 4：逐点误差 + 地板线 ——
  const profSeries = segmentedSeries(xProfile, errorProfile, pal.residual, {
    width: 1.4,
    accept: (v) => v > 0,
    label: endLabelOf(xProfile, profileAxis.xlim, profileRect[2], arrayLabelOf(frame, "error_profile", "逐点误差（最细一级）")),
  });
  if (profSeries.length > 0) scene.layers.push(lineLayer("profile", profSeries));
  else degraded.push("帧 error_profile 没有 > 0 的有限值：逐点误差画不出来");
  const floorLine = hLine(profileAxis.xlim[0], profileAxis.xlim[1], floor);
  scene.layers.push(
    lineLayer("profile", [
      series(floorLine.x, floorLine.y, pal.reference, {
        width: 1,
        dash: [2, 3],
        label: endLabel(floorLine.x[1], profileAxis.xlim, profileRect[2], `地板 ${formatNumber(floor, 2)}`),
      }),
    ]),
  );

  // —— 文本：底部通栏（数字在前、图例在后；装不下时 `textBlock` 先砍图例）——
  // 帧里那几条长 note（ladder_note / reading_note / trap_note）不进面板：它们是"读图口径"的
  // 完整句子，底部 pitfalls 面板本来就是放这个的地方；面板只留数字与最短的判读。
  const lines: string[] = [
    `细化方向：${isSpace ? "空间（固定 r、加密网格）" : "时间（固定网格、Δt 每次减半）"}　格式=${scheme ?? "?"}` +
      `　靶子=${clip(reference ?? "帧未说明", 24)}　${grad.text}　固定 r=${formatNumber(fixedR ?? 0.4, 3)}`,
  ];
  if (theoryOrder !== undefined && measured !== undefined) {
    lines.push(
      `实测阶 ${formatNumber(measured, 6)}　理论阶 ${formatNumber(theoryOrder, 4)}　差 ${formatNumber(orderGap ?? Math.abs(measured - theoryOrder), 3)}` +
        `　拟合 R²=${formatNumber(r2 ?? NaN, 6)}　标准差 ${formatNumber(stderr ?? NaN, 3)}　用了 ${formatNumber(nUsed ?? 0, 6)}/${formatNumber((nUsed ?? 0) + (nDropped ?? 0), 6)} 级`,
    );
  }
  if (errFinest !== undefined && errCoarsest !== undefined) {
    lines.push(
      `误差 ${formatNumber(errFinest, 4)}（最细）→ ${formatNumber(errCoarsest, 4)}（最粗）　比值 ${formatNumber(errRatio ?? NaN, 4)}　` +
        "误差在空间上不是常数：用一个范数代表全部信息本身就是一次压缩；贴 1e-17 地板 = 那里本来就接近 0",
    );
  }
  if (orderContinuous !== undefined) {
    lines.push(
      `⚠ 换靶子（改用连续解）实测斜率 = ${formatNumber(orderContinuous, 4)}` +
        (target === "time" ? "：空间误差盖住了时间误差，不是格式不收敛" : "（本来就是连续解当靶子，应当一致）"),
    );
  }
  lines.push(
    "图例：蓝=实测阶梯、橙虚线=对照阶梯、灰虚线=理论斜率线（锚在最细一级）、红点=最细一级；副图=逐点误差（对数纵轴）。",
  );
  scene.texts.push(...textBlock(box.left, band.textTop, box.bottom, lines, pal.muted, band.budget, "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "阶梯轴：固定窗口（双对数）；逐点误差轴：自适应（会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  scene.badges.push(placeBadge(badge("status", orderStatus(target, measured, theoryOrder, orderContinuous), 8, 8), SLOT.status, width));
  const status = degradedBadge(degraded, SLOT.degraded, width);
  if (status !== undefined) scene.badges.push(status);

  requireDrawableLayers(scene);
  applyOverflow(scene, countNonFinite(errorLadder) + countNonFinite(errorContinuous) + countNonFinite(errorProfile), {
    slot: SLOT.overflow,
    width,
  });

  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);

  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function optional(frame: Frame, name: string): ArrayLike<number> | undefined {
  return frame.has(name) ? (frame.array(name) as ArrayLike<number>) : undefined;
}

/** 把若干数组（可能缺）里 > 0 的有限值并起来——只用于取显示范围。 */
function concatPositive(...arrays: (ArrayLike<number> | undefined)[]): Float64Array {
  const out: number[] = [];
  for (const a of arrays) {
    if (a === undefined) continue;
    for (let i = 0; i < a.length; i += 1) {
      const v = a[i] as number;
      if (Number.isFinite(v) && v > 0) out.push(v);
    }
  }
  return Float64Array.from(out);
}

function numberFromString(s: string | undefined): number | undefined {
  if (s === undefined) return undefined;
  const v = Number(s);
  return Number.isFinite(v) ? v : undefined;
}

/**
 * 状态角标：一句话说清"这条阶梯是不是可信"。
 *
 * 时间细化 + 连续解靶子是本 viz 的招牌陷阱（pitfall 第 1 条）：那时 `measured_order`
 * 本身很漂亮（CN 是 2.0000），**骗人的是 `order_vs_continuous`**——所以角标先报它。
 */
function orderStatus(
  target: string | undefined,
  measured: number | undefined,
  theory: number | undefined,
  orderContinuous: number | undefined,
): string {
  if (measured === undefined || theory === undefined) return "帧缺 measured_order/theory_order：不自行判断阶数";
  const ok = Math.abs(measured - theory) <= 0.1;
  const head = ok
    ? `实测阶 ${measured.toPrecision(5)} ≈ 理论 ${theory.toPrecision(3)}`
    : `⚠ 实测阶 ${measured.toPrecision(5)} 与理论 ${theory.toPrecision(3)} 差 ${Math.abs(measured - theory).toPrecision(3)}`;
  if (target === "time" && orderContinuous !== undefined) {
    return `${head}；换连续解当靶子会得到 ${orderContinuous.toPrecision(5)}（靶子陷阱，见文本）`;
  }
  return head;
}
