/**
 * `work_precision`（3.1 章 viz 4）的 model：工作-精度前沿。**没有时间轴**（`time.mode="none"`）。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `vs_tol` | 参考线 y = tol（实到误差恰好等于请求容差的位置） |
 * | 2 | line | `vs_tol` | 四条自适应曲线（heun / bs23 / rkf45 / dopri5）+ 固定步长 RK4 的误差阶梯 |
 * | 3 | line+scatter | `vs_tol` | 当前 tol 的竖线 + 四条曲线在该 tol 上的实到误差 |
 * | 4 | line | `frontier` | 目标精度参考线（= 当前 tol 下**最差**方法的实到误差，四个方法都达到它） |
 * | 5 | line | `frontier` | 四条自适应曲线 + 固定步长 RK4（横轴换成"每单位时间的求值次数"） |
 * | 6 | line+scatter | `frontier` | 同一精度反查的工作量（`work_at_target`）；`-1` 掩码的点**不画成数值** |
 *
 * 三条最容易读反的地方（图上怎么防）
 * ----------------------------------
 * 1. **左图四条曲线不重合**：同一个 tol 下实到误差能差 $10\times$——"同样的容差谁更省"
 *    这个问题本身没有定义；要比就固定在精度上比（右图）。图上把差值写出来。
 * 2. **工作量单位是"每单位时间求值次数"**：换"步数"会让高阶方法凭空占便宜、换"总求值"
 *    又会被区间长度带跑。轴名与面板文字都写明这个单位。
 * 3. **下包络才是前沿**：任何一条曲线都不是"这个方法的性能"，它只是这个算例上的一条切面。
 *    反查拒绝外推（`-1` + `work_at_target_valid`），图上把不可用的方法**画成明确标记**，
 *    而不是悄悄少一个点。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  metaNumber,
  optionalArray,
  palette,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  sceneShell,
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
  series,
  sourceBadge,
  text,
  type LineSeries,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";
import {
  applyOverflowNote,
  axisTitleOf,
  axisYlabelOf,
  logRange,
  positiveMinMax,
  stackPanels,
  textBlock,
  textTop,
  unitBudget,
  wrapUnits,
  type PanelStack,
} from "./_panel.js";

const CHAPTER = "3.1";
const VIZ = "work_precision";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "工作-精度前沿：同一个 tol 不是同一个工作点";

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
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

/** 后端固定的方法顺序（帧 `strings.method_order` 是权威，这份只是回退值与配色索引）。 */
const FALLBACK_METHODS = ["heun", "bs23", "rkf45", "dopri5"] as const;

export const workPrecisionModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const width = Math.max(320, Math.round(opts.width));
  const height = Math.max(240, Math.round(opts.height));

  // —— 必需 ——
  const tolGrid = requireArray(frame, "tol_grid");
  const tolLine = requireArray(frame, "tol_line");
  const errorFixed = requireArray(frame, "error_fixed_rk4");
  const workFixed = requireArray(frame, "work_fixed_rk4");
  const pitfalls = requirePitfalls(frame);
  const order = (stringOf(frame, "method_order") ?? FALLBACK_METHODS.join("|")).split("|").filter((s) => s !== "");

  const degraded: string[] = [];
  if (order.length === 0) order.push(...FALLBACK_METHODS);

  // —— 可选 ——
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 用请求 t（本 viz 没有时间轴）");
  const tol = metaNumber(frame, "tol");
  if (tol === undefined) degraded.push("帧 meta 缺 tol：不画当前 tol 的竖线与高亮点");
  const extrap = (metaNumber(frame, "local_extrapolation") ?? 0) > 0.5;
  const errorSpread = metaNumber(frame, "error_spread_at_tol");
  const workSpread = metaNumber(frame, "work_spread_at_tol");
  const targetError = metaNumber(frame, "target_error");
  const targetSpread = metaNumber(frame, "work_at_target_spread");
  const nUnavailable = metaNumber(frame, "n_work_at_target_unavailable");
  const readingNote = stringOf(frame, "reading_note");
  const atTargetText = stringOf(frame, "at_target");
  const targetNote = stringOf(frame, "target_note");
  if (targetNote === undefined) degraded.push("帧 strings 缺 target_note：不声明反查的目标精度是怎么来的");
  const problem = stringOf(frame, "problem");

  const errors: Record<string, ArrayLike<number>> = {};
  const works: Record<string, ArrayLike<number>> = {};
  for (const m of order) {
    const e = optionalArray(frame, `error_${m}`);
    const w = optionalArray(frame, `work_${m}`);
    if (e === undefined || w === undefined) {
      degraded.push(`帧缺数组 error_${m}/work_${m}：不画这个方法的曲线`);
      continue;
    }
    if (e.length !== tolGrid.length || w.length !== tolGrid.length) {
      degraded.push(`数组 error_${m}/work_${m} 与 tol_grid 长度不一致（${e.length}/${w.length} vs ${tolGrid.length}）：不画`);
      continue;
    }
    errors[m] = e;
    works[m] = w;
  }
  const drawn = order.filter((m) => errors[m] !== undefined);
  if (drawn.length === 0) degraded.push("一个自适应方法的数组都不全：只有固定步长 RK4 的曲线");

  const markTol = optionalArray(frame, "mark_tol");
  const markError = optionalArray(frame, "mark_error");
  const markWork = optionalArray(frame, "mark_work");
  const atTarget = optionalArray(frame, "work_at_target");
  const atTargetValid = optionalArray(frame, "work_at_target_valid");
  if (atTarget === undefined || atTargetValid === undefined) {
    degraded.push("帧缺数组 work_at_target/work_at_target_valid：右图不画「同一精度反查」的标记");
  }

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: stringOf(frame, "data_source") ?? "backend",
    pitfalls,
  });

  // —— 排版：左（tol→误差）| 右（工作量→误差），下挂说明块；窄画布改成上下叠 ——
  // 说明块只有 ~7 行（`textBlock` 会按高度截断并如实标注"另有 n 行未画"），所以每行都压到
  // ≤60 个汉字（= 120 个显示单位 = 一行），并把最容易被读反的那条（下包络）放在最前面。
  const lines: string[] = [];
  lines.push(
    readingNote ??
      "左图看「说到做不到」：曲线在 y=tol 上方 = 实到误差大于请求容差；右图看代价，四条曲线的**下包络**才是前沿——任何一条曲线都不是「这个方法的性能」，它只是这个算例上的一条切面。",
  );
  lines.push(
    `同一个 tol 不是同一个工作点：本帧 tol=${fmt(tol)}（${extrap ? "开" : "关"}局部外推）下四个方法的实到误差相差 ${fmt(errorSpread)} 倍、工作量相差 ${fmt(workSpread)} 倍。`,
  );
  if (targetNote !== undefined || atTargetText !== undefined) {
    lines.push(
      (targetNote ?? "") +
        (atTargetText !== undefined ? `。本帧反查：${atTargetText}（跨方法 ${fmt(targetSpread)} 倍` + (nUnavailable !== undefined && nUnavailable > 0 ? `，${Math.round(nUnavailable)} 个方法拒绝作答）。` : "）。") : ""),
    );
  }
  lines.push("固定步长 RK4 的曲线由误差阶梯反查（线性插值，不外推）；反查不可用的方法画成红色标记贴在轴左端，不是数值。");
  lines.push(`工作量单位：每单位时间的函数求值次数（不是步数，也不是整段总求值）；算例 ${problem ?? "—"}。`);
  // 本 viz 的 pitfalls 文案短，footer 收到 `12 + 76 + 34`，把高度让给两块主图。
  const absolute = stackedPanels(width, height, { footer: 12 + 76 + 34 });
  const wide = absolute.contentWidth >= 2 * 300 + 24;
  // 两块主图 + 下方的说明块。**说明块的上边界必须由轴的矩形算出来**：
  // 上方面板的下沿外面还有 `[bottom+3, bottom+15]`（刻度标签）与 `[bottom+16, bottom+28]`（轴名）
  // 两条装饰带——说明块落进去就是规则 8 的当场违规（1.2 章踩过）。
  // 说明带**横跨两栏**、按整幅宽度折行（每行 ~60 个汉字 = 一行），所以它的高度是可以先算出来的：
  // 先给说明带留够高度，剩下的才是两块主图（主图绝不被文字盖住——1.2 章的教训）。
  const bandBudget = unitBudget(absolute.contentWidth - 12);
  const infoRows = lines.flatMap((body) => wrapUnits(body, bandBudget)).length;
  const infoH = Math.min(infoRows * 15 + 6, Math.round((absolute.bottom - absolute.top) * 0.5));
  const plotH = Math.max(120, Math.min(344, absolute.bottom - absolute.top - infoH - 36));
  const stack: PanelStack = wide
    ? sideBySide(absolute.left, absolute.top, absolute.contentWidth, plotH)
    : stackPanels({ ...absolute, bottom: absolute.top + plotH }, [0.5, 0.5], { minHeights: [110, 110] });
  const leftRect = stack.rects[0] as [number, number, number, number];
  const rightRect = stack.rects[1] as [number, number, number, number];
  /** 两块主图的下沿 + 28px 装饰带 + 8px 余量。 */
  const textTop0 = (wide ? leftRect[1] + leftRect[3] : rightRect[1] + rightRect[3]) + 36;
  const textBottom0 = absolute.bottom + 8;

  // —— 左轴：请求容差 → 实到误差 ——
  const tolRange = positiveMinMax(tolGrid) ?? [1e-4, 1e-8];
  const vsTol = makeAxis("vs_tol", leftRect, {
    xlim: logRange(Math.min(tolRange[0], tolRange[1]), Math.max(tolRange[0], tolRange[1])),
    ylim: errorYlim([...Object.values(errors), errorFixed, markError], undefined),
    xscale: "log",
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "vs_tol", "实到误差"),
    title: axisTitleOf(SPEC?.axes, "vs_tol", "请求容差 vs 实到误差（双对数）"),
    autoscaled: false,
  });

  // —— 右轴：工作量 → 实到误差（**前沿**）——
  const workAll: ArrayLike<number>[] = [...Object.values(works), workFixed];
  if (markWork !== undefined) workAll.push(markWork);
  if (atTarget !== undefined && atTargetValid !== undefined) workAll.push(positiveRun(atTarget, atTargetValid));
  const workRange = multiPositiveMinMax(workAll) ?? [1, 1e4];
  const frontier = makeAxis("frontier", rightRect, {
    xlim: logRange(workRange[0], workRange[1]),
    ylim: errorYlim([...Object.values(errors), errorFixed, markError], undefined),
    xscale: "log",
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "frontier", "实到误差"),
    title: axisTitleOf(SPEC?.axes, "frontier", "工作量 vs 实到误差（下包络才是前沿）"),
    autoscaled: false,
  });
  scene.axes.push(vsTol, frontier);

  // —— 图层 1：y = tol 对角线（"说到做不到"的判据）——
  // 直接用帧给的 tol_line 当横坐标：它和 tol_grid 是同一串值，所以这条线**正好**落在 y=x 上
  // （不要用 [min,max] 拼两端——那只是两根端点之间的弦，网格非等比时就不在对角线上）。
  const diagX = toArr(tolLine).filter((v) => Number.isFinite(v) && v > 0);
  const diagY = toArr(tolLine).filter((v) => Number.isFinite(v) && v > 0);
  scene.layers.push(
    lineLayer(
      "vs_tol",
      series(diagX.length >= 2 ? diagX : [tolRange[0], tolRange[1]], diagY.length >= 2 ? diagY : [tolRange[0], tolRange[1]], pal.reference, {
        width: 1.2,
        dash: [6, 4],
        label: "$y = tol$（曲线在它上方 = 实到误差大于请求容差）",
      }),
    ),
  );

  // —— 图层 2：四条自适应曲线 + 固定步长 RK4 ——
  const toSeries: LineSeries[] = [];
  drawn.forEach((m, i) => {
    const e = errors[m] as ArrayLike<number>;
    const pts = paired(tolGrid, e);
    if (pts.x.length === 0) return;
    toSeries.push(series(pts.x, pts.y, pal.curves[i % pal.curves.length] as string, { width: 1.8, marker: "o", label: `${m}（自适应）` }));
  });
  const fixedPts = paired(tolGrid, errorFixed);
  if (fixedPts.x.length > 0) {
    toSeries.push(series(fixedPts.x, fixedPts.y, pal.reference, { width: 1.4, dash: [7, 4], marker: "s", label: "固定步长 $RK4$（由误差阶梯反查，不是跑出来的）" }));
  }
  if (toSeries.length > 0) scene.layers.push(lineLayer("vs_tol", toSeries));
  else degraded.push("左图一条曲线都画不出来（误差数组全是非正/非有限）");

  // —— 图层 3：当前 tol 的竖线与四个高亮点 ——
  if (tol !== undefined && tol > 0) {
    scene.layers.push(lineLayer("vs_tol", series([tol, tol], vsTol.ylim, pal.marker, { width: 1.1, dash: [3, 3], label: `当前 tol = ${formatNumber(tol, 3)}` })));
  }
  if (markTol !== undefined && markError !== undefined && markError.length > 0) {
    const mx: number[] = [];
    const my: number[] = [];
    for (let i = 0; i < markError.length; i += 1) {
      const v = markError[i] as number;
      const x = markTol[0] as number;
      if (Number.isFinite(v) && v > 0 && Number.isFinite(x)) {
        mx.push(x);
        my.push(v);
      }
    }
    if (mx.length > 0) {
      scene.layers.push(scatterLayer("vs_tol", mx, my, { r: 3.4, color: pal.marker, label: "当前 tol 上的实到误差（同一个 tol，四个不同的工作点）" }));
    }
  }

  // —— 图层 4：目标精度参考线（四个方法都达到的那个精度）——
  if (targetError !== undefined && targetError > 0) {
    scene.layers.push(
      lineLayer("frontier", series(frontier.xlim, [targetError, targetError], pal.note, { width: 1.2, dash: [4, 3], label: `目标精度 ${formatNumber(targetError, 3)}（= 当前 tol 下最差方法的实到误差）` })),
    );
  }

  // —— 图层 5：右图的四条曲线（横轴换成工作量）+ 固定步长 RK4 ——
  const workSeries: LineSeries[] = [];
  drawn.forEach((m, i) => {
    const w = works[m] as ArrayLike<number>;
    const e = errors[m] as ArrayLike<number>;
    const pts = paired(w, e);
    if (pts.x.length === 0) return;
    workSeries.push(series(pts.x, pts.y, pal.curves[i % pal.curves.length] as string, { width: 1.8, marker: "o", label: `${m}（自适应）` }));
  });
  const fixedWork = paired(workFixed, errorFixed);
  if (fixedWork.x.length > 0) {
    workSeries.push(series(fixedWork.x, fixedWork.y, pal.reference, { width: 1.4, dash: [7, 4], marker: "s", label: "固定步长 $RK4$" }));
  }
  if (workSeries.length > 0) scene.layers.push(lineLayer("frontier", workSeries));

  // —— 图层 6：同一精度反查的工作量（掩码为 0 的点不画成数值）——
  if (atTarget !== undefined && atTargetValid !== undefined && targetError !== undefined && targetError > 0) {
    const vx: number[] = [];
    const vy: number[] = [];
    const bx: number[] = [];
    const by: number[] = [];
    for (let i = 0; i < atTarget.length; i += 1) {
      const v = atTarget[i] as number;
      const valid = (atTargetValid[i] as number) > 0.5;
      if (valid && Number.isFinite(v) && v > 0) {
        vx.push(v);
        vy.push(targetError);
      } else {
        bx.push(frontier.xlim[0]);
        by.push(targetError);
      }
    }
    if (vx.length > 0) {
      scene.layers.push(scatterLayer("frontier", vx, vy, { r: 4.6, color: pal.note, label: "同一精度反查的工作量（插值，不外推）" }));
    }
    if (bx.length > 0) {
      scene.layers.push(scatterLayer("frontier", bx, by, { r: 3.2, color: pal.warn, label: "反查不可用（实测区间没覆盖这个精度）：贴在轴左端，不是数值" }));
    }
  }
  if (markWork !== undefined && markError !== undefined && markWork.length === markError.length) {
    const mx: number[] = [];
    const my: number[] = [];
    for (let i = 0; i < markWork.length; i += 1) {
      const w = markWork[i] as number;
      const e = markError[i] as number;
      if (Number.isFinite(w) && w > 0 && Number.isFinite(e) && e > 0) {
        mx.push(w);
        my.push(e);
      }
    }
    if (mx.length > 0) {
      scene.layers.push(scatterLayer("frontier", mx, my, { r: 3, color: pal.marker, label: "当前 tol 的工作点" }));
    }
  }

  // —— 文字：两块面板各一行单位声明 + 共享的说明块 ——
  const unitLeft = `横轴：请求容差 tol（对数）；纵轴：实到最大误差（对数）。曲线在 y=tol 上方 = 「说到做不到」。`;
  const unitRight = `横轴：工作量 = **每单位时间**的函数求值次数（不是步数、不是总求值）；纵轴同上。x 越小、y 越低越好。`;
  scene.texts.push(
    text(leftRect[0] + 8, textTop(leftRect), unitLeft, { role: "note", size: 11, color: pal.muted, maxWidth: Math.max(60, leftRect[2] - 16) }),
    text(rightRect[0] + 8, textTop(rightRect), unitRight, { role: "note", size: 11, color: pal.muted, maxWidth: Math.max(60, rightRect[2] - 16) }),
  );


  scene.texts.push(...textBlock(stack.left + 6, textTop0, textBottom0, lines, pal.muted, bandBudget, "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "两轴都是固定对数窗口（数据由后端一次扫描给出，本 viz 没有时间轴）", 8, 8), SLOT.auxAxis, width),
  );
  const statusBits: string[] = [`本 viz 无时间轴（time.mode=none）`, `tol=${fmt(tol)}　${extrap ? "开外推" : "关外推"}`];
  if (errorSpread !== undefined) statusBits.push(`同 tol 误差差 ${formatNumber(errorSpread, 3)} 倍`);
  if (targetSpread !== undefined) statusBits.push(`同精度工作量差 ${formatNumber(targetSpread, 3)} 倍`);
  scene.badges.push(placeBadge(badge("status", statusBits.join("　"), 8, 8), SLOT.status, width));

  requireDrawableLayers(scene);
  const rawNonFinite =
    countNonFinite(tolGrid) + countNonFinite(errorFixed) + countNonFinite(workFixed) + drawn.reduce((s, m) => s + countNonFinite(errors[m] as ArrayLike<number>), 0);
  const extras: string[] = [];
  const invalid = atTargetValid !== undefined ? countValidFalse(atTargetValid) : 0;
  if (invalid > 0) extras.push(`${invalid} 个方法的"同一精度反查"超出实测区间（-1 哨兵 + 掩码，图上不画成数值）`);
  const zeroWork = countZero(workFixed);
  if (zeroWork > 0) extras.push(`固定步长曲线有 ${zeroWork} 个工作量恰为 0（对数轴上未绘制）`);
  applyOverflowNote(scene, extras.join("；"), rawNonFinite, { slot: SLOT.overflow, width });

  const deg = degradedBadgeLocal(degraded, SLOT.degraded, width);
  if (deg !== undefined) scene.badges.push(deg);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** 左右并排（等分）：与 `model.ts::stackedPanels` 同一套边距，只是把高度切成两列。 */
function sideBySide(left: number, top: number, contentWidth: number, h: number): PanelStack {
  const gap = 54; // 右列要给自己的 y 刻度标签与轴名留出 ~56px
  const colW = Math.max(60, Math.round((contentWidth - gap) / 2));
  return {
    rects: [
      [left, top, colW, h],
      [left + colW + gap, top, colW, h],
    ],
    left,
    contentWidth,
    top,
    bottom: top + h,
  };
}

/** `TypedArray` → `number[]`（联合类型下 `Array.from` 挑不出重载）。 */
function toArr(a: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < a.length; i += 1) out.push(a[i] as number);
  return out;
}

/** 两条数组按"两个都 > 0 且有限"配对（对数轴上 0/负值不存在）。 */
function paired(x: ArrayLike<number>, y: ArrayLike<number>): { x: number[]; y: number[] } {
  const n = Math.min(x.length, y.length);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const xv = x[i] as number;
    const yv = y[i] as number;
    if (Number.isFinite(xv) && xv > 0 && Number.isFinite(yv) && yv > 0) {
      xs.push(xv);
      ys.push(yv);
    }
  }
  return { x: xs, y: ys };
}

/** 按掩码筛选出"有效且 > 0"的子序列（掩码 ≤ 0.5 的哨兵一律不参与定界）。 */
function positiveRun(values: ArrayLike<number>, mask: ArrayLike<number>): Float64Array {
  const out: number[] = [];
  const n = Math.min(values.length, mask.length);
  for (let i = 0; i < n; i += 1) {
    const v = values[i] as number;
    if ((mask[i] as number) > 0.5 && Number.isFinite(v) && v > 0) out.push(v);
  }
  return Float64Array.from(out);
}

/** 多组正有限值的整体 [min, max]。 */
function multiPositiveMinMax(arrays: readonly ArrayLike<number>[]): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const a of arrays) {
    const r = positiveMinMax(a);
    if (r === undefined) continue;
    if (r[0] < lo) lo = r[0];
    if (r[1] > hi) hi = r[1];
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : undefined;
}

/** 误差轴（固定对数窗口）：把所有曲线与参考线一起包住，再各让一格。 */
function errorYlim(arrays: readonly (ArrayLike<number> | undefined)[], floor: number | undefined): [number, number] {
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
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [1e-9, 1e-2];
  return logRange(lo, hi);
}

/** 掩码里"不通过"（≤ 0.5）的个数。 */
function countValidFalse(mask: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < mask.length; i += 1) if ((mask[i] as number) <= 0.5) n += 1;
  return n;
}

/** 恰好为 0 的样本个数（对数轴上画不出来）。 */
function countZero(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if (a[i] === 0) n += 1;
  return n;
}

/** 数值的紧凑显示（undefined 显示为「—」而不是 0）。 */
function fmt(v: number | undefined): string {
  return v === undefined || !Number.isFinite(v) ? "—" : Number.isInteger(v) ? String(v) : formatNumber(v, 3);
}

/** 降级角标（槽位固定在本章的 SLOT.degraded）。 */
function degradedBadgeLocal(items: readonly string[], slot: number, width: number) {
  if (items.length === 0) return undefined;
  const head = items.slice(0, 2).join("；");
  const rest = items.length > 2 ? `；等 ${items.length} 项` : "";
  return placeBadge(badge("status", `⚠ 降级：${head}${rest}`, 8, 8), slot, width);
}
