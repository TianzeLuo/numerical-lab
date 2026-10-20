/**
 * `float_zoom`（1.1 章）的 model：把浮点格点画成一维数轴，并展示 0.1 累加的残差。
 *
 * 上面一块图
 * ----------
 * 横轴是「与窗口中心的真实距离 / ulp(窗口中心)」，纵轴只有一行示意。每个圆点是一个真实存在的
 * double——不是采样，也不是任意实数。相邻点的横向距离就是它们的实际间距：
 *
 * - 0.3 附近的窗口里，相邻点等距（每格 1 个 ulp）；0.1+0.2 的实际值正好是 0.3 右边那一格。
 * - 0.25 附近，左半边每格 0.5、右半边每格 1：跨过 2 的幂，间距翻倍，圆点明显变稀。
 *
 * 括号表示相邻两点之间的间距。悬停任意圆点会给出它的精确十进制，帮助建立“打印成 0.3 的数
 * 可能不是 0.3”的直觉。
 *
 * 下面一块图
 * ----------
 * 0.1 连加 k 次的残差。紫色线是相对数学值 k/10 的总残差，绿色虚线是相对 k·fl(0.1) 的
 * 纯累加误差；两条线都在对数纵轴上。误差恰为 0 时用 u/2·max(|k/10|,0.1) 的显示地板抬高，
 * 地板上的点用空心圈标出——“贴地板”不等于“误差为零”。
 */

import { formatNumber, niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import type { Frame } from "../../shared/src/frame.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
  dataSourceOf,
  degradedBadge,
  finitePositiveMinMax,
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
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  type LineSeries,
} from "../../shared/src/scene.js";
import {
  TEXT_ROW_H,
  absMax,
  avoidTextOverlap,
  axisTitleOf,
  axisYlabelOf,
  floorIndices,
  pick,
  textBlock,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.1";
const VIZ = "float_zoom";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "0.1+0.2 为什么不是 0.3：把数轴放大到最后一个二进制位";

const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  status: 3,
  preview: 4,
  overflow: 5,
  degraded: 7,
} as const;

const TEXT_COLUMN_MIN_WIDTH = 900;
const TEXT_COLUMN_GAP = 26;
const TEXT_COLUMN_WIDTH = 300;
const PANEL_GAP = 44;

const GRID_ROW = 0.5;
const BRACKET_LOW = 0.68;
const BRACKET_HIGH = 0.79;

interface FloatZoomLayout {
  main: Rect;
  aux: Rect;
  mainText?: Rect;
  auxText?: Rect;
}

function floatZoomLayout(width: number, height: number): FloatZoomLayout {
  // 主图是一维数轴，不需要太高；把余下的高度留给残差图和轴名。
  const base = stackedPanels(width, height, { footer: 96 });
  const avail = base.bottom - base.top;
  const mainH = Math.round((avail - PANEL_GAP) * 0.46);
  const auxH = avail - PANEL_GAP - mainH;
  const twoCol = width >= TEXT_COLUMN_MIN_WIDTH;
  const sideW = twoCol ? TEXT_COLUMN_WIDTH : 0;
  const plotW = twoCol ? base.contentWidth - sideW - TEXT_COLUMN_GAP : base.contentWidth;
  const main: Rect = [base.left, base.top, plotW, mainH];
  const aux: Rect = [base.left, base.top + mainH + PANEL_GAP, plotW, auxH];
  if (!twoCol) return { main, aux };
  const textX = base.left + plotW + TEXT_COLUMN_GAP;
  const textW = base.left + base.contentWidth - textX;
  return {
    main,
    aux,
    mainText: [textX, main[1], textW, main[3]],
    auxText: [textX, aux[1], textW, aux[3]],
  };
}

export const floatZoomModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const offset = requireArray(frame, "offset");
  const values = requireArray(frame, "value");
  const kGrid = requireArray(frame, "k_grid");
  const residual = requireArray(frame, "residual");
  const residualSumOnly = requireArray(frame, "residual_sum_only");
  const residualFloor = requireArray(frame, "residual_floor");
  requireSameLength(offset, values, "帧数组 offset 与 value");
  requireSameLength(kGrid, residual, "帧数组 k_grid 与 residual");
  requireSameLength(kGrid, residualSumOnly, "帧数组 k_grid 与 residual_sum_only");
  requireSameLength(kGrid, residualFloor, "帧数组 k_grid 与 residual_floor");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const centerOffset = optionalFirst(frame, "center_offset");
  const centerY = optionalFirst(frame, "center_y");
  const specialOffset = optionalFirst(frame, "special_offset");
  const specialY = optionalFirst(frame, "special_y");
  const residualMark = optionalFirst(frame, "residual_mark");
  const kMark = optionalFirst(frame, "k_mark");
  const sumValue = frame.has("sum_value") ? requireArray(frame, "sum_value") : undefined;
  if (centerOffset === undefined || centerY === undefined) degraded.push("帧缺 center_offset/center_y：不画窗口中心");
  if (!frame.has("residual_mark") || !frame.has("k_mark")) {
    degraded.push("帧缺 residual_mark/k_mark：副图不画当前点（不拿请求 t 冒充实测 k）");
  }

  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const center = metaNumber(frame, "center");
  const spacingCenter = metaNumber(frame, "spacing_center");
  const spacingRatio = metaNumber(frame, "spacing_ratio");
  const nPoints = metaNumber(frame, "n_points");
  const radiusMeta = metaNumber(frame, "window_ulps");
  const specialUlps = metaNumber(frame, "special_ulps");
  const kMeta = metaNumber(frame, "k");
  if (kMeta === undefined) degraded.push("帧 meta 缺 k：游标按 k_mark 画（若也缺则回到 0）");
  const nExactZero = metaNumber(frame, "residual_exact_zero_steps");
  const nExactZeroSum = metaNumber(frame, "residual_sum_only_exact_zero_steps");
  const focusKind = stringOf(frame, "focus");
  const centerLabel = stringOf(frame, "center_label");
  const specialLabel = stringOf(frame, "special_label") ?? "";
  const note = stringOf(frame, "note");
  const spacingNote = stringOf(frame, "spacing_note");
  const exactCenter = stringOf(frame, "exact_center");
  const exactSpecial = stringOf(frame, "exact_special");
  if (focusKind === undefined) degraded.push("帧 strings 缺 focus：按通用文案画，不猜测当前观察位置");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  if (!frame.has("special_offset") && specialLabel !== "") {
    degraded.push("帧缺 special_offset：不画当前标注的特殊值");
  }

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const layout = floatZoomLayout(width, height);
  const mainRect = layout.main;
  const auxRect = layout.aux;

  const half = Math.max(1, radiusMeta ?? absMax(offset) ?? 1);
  const mainAxis = makeAxis("ladder", mainRect, {
    xlim: [-half, half],
    ylim: [0.18, 0.95],
    xlabel: "距窗口中心（ulp）",
    ylabel: axisYlabelOf(SPEC?.axes, "ladder", "浮点格点（示意行）"),
    title: axisTitleOf(SPEC?.axes, "ladder", "浮点格点：每个圆点是一个可表示的 double"),
    autoscaled: false,
    xticks: niceTicks(-half, half, 9),
    yticks: [],
    grid: false,
  });

  const bounds = finitePositiveMinMax(residual);
  const boundsSum = finitePositiveMinMax(residualSumOnly);
  const boundsFloor = finitePositiveMinMax(residualFloor);
  const lo = Math.min(
    bounds?.[0] ?? Number.POSITIVE_INFINITY,
    boundsSum?.[0] ?? Number.POSITIVE_INFINITY,
    boundsFloor?.[0] ?? Number.POSITIVE_INFINITY,
  );
  const hi = Math.max(bounds?.[1] ?? 0, boundsSum?.[1] ?? 0, boundsFloor?.[1] ?? 0);
  if (!(Number.isFinite(lo) && Number.isFinite(hi))) {
    degraded.push("residual / residual_sum_only / residual_floor 里没有 > 0 的有限值：副轴用占位范围 [1e-18, 1e-14]");
  }
  const auxYlim: [number, number] =
    Number.isFinite(lo) && Number.isFinite(hi) ? niceLimits(lo, hi, 6, { log: true, pad: 0.12 }) : [1e-18, 1e-14];
  const kMax = kGrid.length > 0 ? (kGrid[kGrid.length - 1] as number) : 1;
  const auxAxis = makeAxis("residual", auxRect, {
    xlim: [0, Math.max(1, kMax)],
    ylim: auxYlim,
    yscale: "log",
    xlabel: "0.1 的累加次数 k",
    ylabel: axisYlabelOf(SPEC?.axes, "residual", "绝对误差"),
    title: axisTitleOf(SPEC?.axes, "residual", "0.1 累加 k 次后的误差"),
    autoscaled: true,
    tickTarget: 5,
  });
  scene.axes.push(mainAxis, auxAxis);

  // —— 每一个可表示的 double：圆点就是数轴上的格点，不再另画一条会被误命中的基线 ——
  const notes = Array.from({ length: values.length }, (_, i) => {
    const v = values[i] as number;
    const x = offset[i] as number;
    return `可表示浮点数 ${v.toPrecision(17)}；距窗口中心 ${formatNumber(x, 5)} ulp`;
  });
  const gridY = new Float64Array(offset.length);
  gridY.fill(GRID_ROW);
  scene.layers.push(
    scatterLayer("ladder", offset, gridY, {
      r: 3.2,
      color: pal.reference,
      label: "可表示的 double（每个圆点）",
      notes,
      hideY: true,
    }),
  );

  // —— 相邻间距括号：真实宽度直接来自相邻横坐标之差 ——
  for (const bracketSeries of spacingBrackets(offset, focusKind, pal)) {
    scene.layers.push(lineLayer("ladder", bracketSeries));
  }

  // —— 窗口中心、0.1+0.2 的实际值 ——
  if (centerOffset !== undefined && centerY !== undefined) {
    scene.layers.push(
      scatterLayer("ladder", [centerOffset], [centerY], {
        r: 5.5,
        color: pal.curves[0] ?? "#2b6cb0",
        label: centerLabel ?? "窗口中心",
        notes: exactCenter === undefined ? undefined : [`这个 double 的完整十进制：${exactCenter}`],
        hideY: true,
      }),
    );
  }
  if (focusKind === "boundary") {
    scene.layers.push(
      lineLayer(
        "ladder",
        series([0, 0], [0.3, 0.92], pal.reference, {
          width: 1,
          dash: [4, 4],
          label: "2 的幂边界（左右两侧间距不同）",
        }),
      ),
    );
  }
  if (specialOffset !== undefined && specialY !== undefined) {
    scene.layers.push(
      scatterLayer("ladder", [specialOffset], [specialY], {
        r: 5.5,
        color: pal.marker,
        label: specialLabel === "" ? "当前标注的特殊值" : specialLabel,
        notes:
          exactSpecial === undefined || exactSpecial === ""
            ? undefined
            : [`这个 double 的完整十进制：${exactSpecial}`],
        hideY: true,
      }),
    );
  }

  // —— 残差图：总残差、纯累加误差、显示地板、当前点 ——
  const kNow = kMeta ?? kMark ?? 0;
  scene.layers.push(
    lineLayer(
      "residual",
      series([kNow, kNow], auxAxis.ylim, pal.cursor, { width: 1, dash: [3, 3] }),
    ),
  );
  const totalSeries = segmentedSeries(kGrid, residual, pal.residual, {
    width: 2,
    label: "与数学值 k/10 的总误差",
    accept: (v) => v > 0,
  });
  if (totalSeries.length > 0) scene.layers.push(lineLayer("residual", totalSeries));
  else degraded.push("residual 没有可绘的有限正值");

  const sumSeries = segmentedSeries(kGrid, residualSumOnly, pal.curves[1] ?? "#2f855a", {
    width: 1.5,
    dash: [7, 4],
    label: "与 k·fl(0.1) 的纯累加误差",
    accept: (v) => v > 0,
  });
  if (sumSeries.length > 0) scene.layers.push(lineLayer("residual", sumSeries));
  else degraded.push("residual_sum_only 没有可绘的有限正值");

  const floorSeries = segmentedSeries(kGrid, residualFloor, pal.reference, {
    width: 1,
    dash: [2, 4],
    accept: (v) => v > 0,
  });
  if (floorSeries.length > 0) scene.layers.push(lineLayer("residual", floorSeries));

  const onFloor = floorIndices(residual, residualFloor);
  if (onFloor.length > 0) {
    const p = pick(kGrid, residual, onFloor);
    scene.layers.push(
      scatterLayer("residual", p.x, p.y, {
        r: 2.8,
        color: pal.reference,
        alpha: 0.9,
      }),
    );
  }
  const kIdx = clamp(Math.round(kNow), 0, Math.max(0, kGrid.length - 1));
  if (residualMark !== undefined) {
    scene.layers.push(
      scatterLayer("residual", [kNow], [residualMark], {
        r: 4.5,
        color: pal.marker,
      }),
    );
  }

  // —— 说明（宽画布在右栏；窄画布留给 HTML 阅读区）——
  if (layout.mainText !== undefined) {
    const head: string[] = [];
    if (center !== undefined) head.push(`窗口中心 ${formatNumber(center, 10)}`);
    if (spacingCenter !== undefined) head.push(`相邻 double 相距 ${formatNumber(spacingCenter, 4)}`);
    if (nPoints !== undefined) head.push(`窗口里 ${formatNumber(nPoints, 0)} 个 double`);
    const budget = unitBudget(layout.mainText[2]);
    const bottom = layout.mainText[1] + layout.mainText[3] - 8;
    let y = layout.mainText[1] + 2;
    if (head.length > 0) {
      const headNodes = textBlock(layout.mainText[0], y, bottom, [head.join("；")], pal.muted, budget, "top");
      scene.texts.push(...headNodes);
      y += headNodes.length * TEXT_ROW_H + 6;
    }
    const bullets = [
      "灰点：每一个真实存在的 double；点距就是间距。",
      focusKind === "boundary"
        ? "蓝点：2 的幂 0.25。左侧每格 0.5 个 ulp，右侧每格 1 个 ulp——跨过边界后格点变稀。"
        : "蓝点：实数 0.3 最近的 double；红点：0.1+0.2 实际存下的 double，正好在右边一格。",
      spacingNote ?? "",
      exactCenter === undefined ? "" : `中心精确值 ${exactCenter}`,
      exactSpecial === undefined || exactSpecial === "" ? "" : `红点精确值 ${exactSpecial}`,
    ].filter((line) => line !== "");
    scene.texts.push(...textBlock(layout.mainText[0], y, bottom, bullets, pal.note, budget, "top"));
  }

  if (layout.auxText !== undefined) {
    const live: string[] = [`k=${formatNumber(kNow, 0)}`];
    if (sumValue !== undefined && kIdx < sumValue.length) {
      live.push(`累加结果 ${formatNumber(sumValue[kIdx] as number, 8)}`);
    }
    if (kIdx < residual.length) live.push(`总误差 ${formatNumber(residual[kIdx] as number, 3)}`);
    const lines = [live.join("　")];
    if (nExactZero !== undefined) {
      lines.push(
        `贴地板的总误差采样点：${onFloor.length}/${residual.length}（其中恰好为 0 的 ${formatNumber(nExactZero, 0)} 个）`,
      );
    }
    if (nExactZeroSum !== undefined) {
      lines.push(`纯累加误差里恰好为 0 的步数：${formatNumber(nExactZeroSum, 0)}`);
    }
    lines.push("对数轴上的 0 被地板抬高：贴地板不等于误差为零。");
    scene.texts.push(
      ...textBlock(
        layout.auxText[0],
        layout.auxText[1] + 2,
        layout.auxText[1] + layout.auxText[3] - 6,
        lines,
        pal.muted,
        unitBudget(layout.auxText[2]),
        "top",
      ),
    );
  }

  // —— 角标（网页显示层会把它放到画面正上方）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "下图：自适应 log 纵轴（按本帧三条曲线取整）", 8, 8), SLOT.auxAxis, width),
  );
  const status: string[] = [];
  if (note !== undefined) status.push(note);
  if (specialUlps !== undefined) status.push(`0.1+0.2 位于中心右侧 ${formatNumber(specialUlps, 3)} 个 ulp`);
  if (spacingRatio !== undefined && focusKind === "boundary") {
    status.push(`右侧 / 左侧间距 = ${formatNumber(spacingRatio, 3)}`);
  }
  if (status.length > 0) scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(offset) +
    countNonFinite(values) +
    countNonFinite(kGrid) +
    countNonFinite(residual) +
    countNonFinite(residualSumOnly) +
    countNonFinite(residualFloor);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });

  scene.texts = avoidTextOverlap(scene.texts, [8, 24, width - 16, layout.aux[1] + layout.aux[3] - 24]);
  return scene;
};

/** 从相邻横坐标直接造一个「括号」线：左右两条短竖线 + 一条水平线。 */
function bracket(a: number, b: number, y: number, color: string, label: string): LineSeries {
  const cap = 0.035;
  return series([a, a, b, b], [y - cap, y, y, y - cap], color, {
    width: 1.3,
    label,
    marker: "none",
  });
}

/**
 * 用实际横坐标差构造间距括号。0.25 的左、右两侧各画一个；0.3 则只画中心与右侧一格。
 * 标签里的数值来自帧 meta，模型不自行计算“物理间距”。
 */
function spacingBrackets(
  offset: ArrayLike<number>,
  focusKind: string | undefined,
  pal: ReturnType<typeof palette>,
): LineSeries[][] {
  const xs = Array.from({ length: offset.length }, (_, i) => offset[i] as number).filter((x) => Number.isFinite(x));
  if (xs.length < 2) return [];
  const color = pal.curves[2] ?? "#b7791f";
  let i0 = 0;
  for (let i = 1; i < xs.length; i += 1) {
    if ((xs[i] as number) >= 0) {
      i0 = i;
      break;
    }
  }
  if (focusKind === "boundary" && i0 >= 2 && i0 + 1 < xs.length) {
    const leftA = xs[i0 - 2] as number;
    const leftB = xs[i0 - 1] as number;
    const rightA = xs[i0] as number;
    const rightB = xs[i0 + 1] as number;
    return [
      [
        bracket(leftA, leftB, BRACKET_LOW, color, "左侧相邻间距（0.5 ulp）"),
      ],
      [
        bracket(rightA, rightB, BRACKET_HIGH, color, "右侧相邻间距（1 ulp）"),
      ],
    ];
  }
  if (i0 + 1 < xs.length) {
    return [
      [
        bracket(
          xs[i0] as number,
          xs[i0 + 1] as number,
          BRACKET_HIGH,
          color,
          "相邻两个 double 的间距（1 ulp(中心)）",
        ),
      ],
    ];
  }
  return [];
}

/** 取“长度 1 的数组”的第 0 个元素；缺数组、空数组、非有限值一律 undefined。 */
function optionalFirst(frame: Frame, name: string): number | undefined {
  if (!frame.has(name)) return undefined;
  const a = requireArray(frame, name);
  if (a.length === 0) return undefined;
  const v = a[0] as number;
  return Number.isFinite(v) ? v : undefined;
}
