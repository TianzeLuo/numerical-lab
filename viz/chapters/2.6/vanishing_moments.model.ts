/**
 * `vanishing_moments`（2.6 章）的 model：消失矩把多项式细节系数抹平。
 *
 * 三条阅读纪律（都落在具体图层上）：
 * 1. **内部与边界必须分开量**：只画 `interior_ratio` 会得出"db4 连四次多项式都能抹平"
 *    的错误结论；边界系数是 O(1)，因为延拓把多项式截断了。两条曲线同屏，颜色分开。
 * 2. **`interior_valid = 0` 的点不是"很小"，而是实测严格为 0**：它们在对数轴上被压到
 *    1e-18 地板。模型用空心红点把它们单独标出，折线只连有效点。
 * 3. **u 线是参考线，不是地板**：`u_line` 是 2^-53，内部比值贴住它表示"到舍入级"；
 *    贴住 1e-18 地板才是"严格 0"。
 *
 * `t` 对这个 viz 无意义（manifest: mode=none），scene.t 恒为 0。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  dataSourceOf,
  degradedBadge,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  sceneShell,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  badge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  firstFinite,
  indexAxis,
  legendBlock,
  linearRange,
  logRange,
  positiveMinMax,
  sliceRun,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "2.6";
const VIZ = "vanishing_moments";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "消失矩：多项式被细节系数抹平";
const FLOOR = 1e-18;

const SLOT = {
  source: 0,
  axisMode: 1,
  aux: 2,
  critical: 3,
  status: 4,
  preview: 5,
  overflow: 6,
  degraded: 7,
} as const;

export const vanishingMomentsModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const signal = requireArray(frame, "signal");
  const detailTree = requireArray(frame, "detail_tree");
  const detailLength = requireArray(frame, "detail_length");
  const degreeAxis = requireArray(frame, "degree_axis");
  const interiorRatio = requireArray(frame, "interior_ratio");
  const interiorValid = requireArray(frame, "interior_valid");
  const boundaryRatio = requireArray(frame, "boundary_ratio");
  const uLine = requireArray(frame, "u_line");
  const error = requireArray(frame, "error");
  const pitfalls = requirePitfalls(frame);

  for (const [name, arr] of [
    ["interior_ratio", interiorRatio],
    ["interior_valid", interiorValid],
    ["boundary_ratio", boundaryRatio],
    ["u_line", uLine],
  ] as const) {
    if (arr.length !== degreeAxis.length) {
      throw new Error(`帧 ${CHAPTER}/${VIZ} 的 ${name} 长度 ${arr.length} ≠ degree_axis 长度 ${degreeAxis.length}`);
    }
  }
  const level = Math.max(1, Math.round(metaNumber(frame, "level") ?? detailLength.length - 1));
  if (detailLength.length !== level + 1) {
    throw new Error(`帧 ${CHAPTER}/${VIZ} 的 detail_length 长度 ${detailLength.length} ≠ J+1=${level + 1}`);
  }
  const n = signal.length;
  const widthExt = Math.max(1, Math.round(metaNumber(frame, "n") ?? n));

  // —— 可选：当前次数下逐层比值（缺了只是少一块面板内的标注）——
  const degraded: string[] = [];
  const levelRatio = frame.has("level_ratio") ? requireArray(frame, "level_ratio") : undefined;
  const levelBoundary = frame.has("level_boundary") ? requireArray(frame, "level_boundary") : undefined;
  if (levelRatio === undefined || levelBoundary === undefined) {
    degraded.push("帧缺 level_ratio/level_boundary：不显示「当前次数下逐层比值」的柱状对照");
  }

  // —— meta ——
  const degree = metaNumber(frame, "degree");
  const moments = metaNumber(frame, "moments");
  const interiorMax = metaNumber(frame, "interior_max");
  const boundaryMax = metaNumber(frame, "boundary_max");
  const ratioOverU = metaNumber(frame, "ratio_over_u");
  const killed = frame.meta["killed"];
  const nFloored = metaNumber(frame, "n_ratio_floored");
  const nInterior = metaNumber(frame, "n_interior");
  const level1Ratio = metaNumber(frame, "level1_ratio");
  const pollution = metaNumber(frame, "pollution_level1");
  const pollutionSpan = metaNumber(frame, "pollution_span1");
  const u = metaNumber(frame, "u");
  const prError = metaNumber(frame, "pr_error");
  const wavelet = stringOf(frame, "wavelet");
  const mode = stringOf(frame, "mode");
  const note = stringOf(frame, "note");
  const boundaryNote = stringOf(frame, "boundary_note");
  const maskNote = stringOf(frame, "mask_note");
  const nonfinitePolicy = stringOf(frame, "nonfinite_policy");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.56, auxH: 128, gap: 48 });

  // —— 多项式与各层细节系数（只画到真实长度）——
  let lo = 0;
  let hi = 0;
  for (let i = 0; i < n; i += 1) {
    const v = signal[i] as number;
    if (Number.isFinite(v)) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  for (let j = 1; j <= level; j += 1) {
    const len = Math.max(0, Math.round(detailLength[j] ?? 0));
    for (let c = 0; c < len; c += 1) {
      const v = detailTree[j * widthExt + c] as number;
      if (Number.isFinite(v)) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
  }
  const polyAxis = makeAxis("poly", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [0, Math.max(1, n - 1)],
    ylim: linearRange(lo, hi, 0.08),
    title: axisTitleOf(SPEC?.axes, "poly", "多项式信号 t^k 与它的细节系数"),
    ylabel: axisYlabelOf(SPEC?.axes, "poly", "x"),
    xlabel: "样本下标",
    autoscaled: false,
  });

  // —— 内部 / 边界量级比（对数）——
  const allPositive: number[] = [];
  for (let i = 0; i < degreeAxis.length; i += 1) {
    for (const v of [interiorRatio[i], boundaryRatio[i], uLine[i]]) {
      if (Number.isFinite(v) && (v as number) > 0) allPositive.push(v as number);
    }
  }
  const ratioLo = Math.min(...allPositive, FLOOR);
  const ratioHi = Math.max(...allPositive, 1);
  const ratioAxis = makeAxis("ratio", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [firstFinite(degreeAxis) ?? 0, lastFinite(degreeAxis) ?? 1],
    ylim: logRange(ratioLo * 0.6, ratioHi * 2),
    yscale: "log",
    title: axisTitleOf(SPEC?.axes, "ratio", "内部量级比 max|d|/max|a|（对数轴）"),
    ylabel: axisYlabelOf(SPEC?.axes, "ratio", "ratio"),
    xlabel: "多项式次数 k",
    autoscaled: true,
  });

  // —— 误差副轴（规则 3）——
  const errLo = positiveMinMax(error)?.[0] ?? FLOOR;
  const errHi = positiveMinMax(error)?.[1] ?? 1;
  const errAxis = makeAxis("error", [L.rightX, L.topY, L.rightW, L.auxH], {
    xlim: [0, Math.max(1, n - 1)],
    ylim: logRange(errLo * 0.5, Math.max(errHi, FLOOR) * 2),
    yscale: "log",
    title: axisTitleOf(SPEC?.axes, "error", "该模式的往返误差"),
    ylabel: axisYlabelOf(SPEC?.axes, "error", "max|delta|"),
    xlabel: "样本下标",
    autoscaled: true,
  });
  scene.axes.push(polyAxis, ratioAxis, errAxis);

  const xIdx = indexAxis(n);
  scene.layers.push(lineLayer("poly", series(xIdx, signal, pal.curves[0], { width: 2.2, label: "多项式信号" })));
  const detailColors = [pal.exact, pal.curves[1], pal.curves[2], pal.residual, "#805ad5", "#2c7a7b"];
  for (let j = 1; j <= level; j += 1) {
    const len = Math.max(0, Math.round(detailLength[j] ?? 0));
    if (len === 0) continue;
    const y = new Float64Array(len);
    for (let c = 0; c < len; c += 1) y[c] = detailTree[j * widthExt + c] as number;
    const x = indexAxis(len);
    scene.layers.push(
      lineLayer(
        "poly",
        series(x, y, detailColors[j % detailColors.length] as string, {
          width: 1.2,
          dash: [4, 3],
          label: `d_${j}`,
        }),
      ),
    );
  }

  // —— 量级比：内部有效点连线、无效点（严格 0）单独标记 ——
  const validIdx: number[] = [];
  const flooredIdx: number[] = [];
  for (let i = 0; i < degreeAxis.length; i += 1) {
    if ((interiorValid[i] ?? 0) > 0.5) validIdx.push(i);
    else flooredIdx.push(i);
  }
  if (validIdx.length > 0) {
    const xs = Float64Array.from(validIdx, (i) => degreeAxis[i] as number);
    const ys = Float64Array.from(validIdx, (i) => interiorRatio[i] as number);
    // 只有**在 degree 轴上也相邻**的有效点才连线：跨过"严格为 0"的点时断开，
    // 否则折线会把地板两侧连成一条不存在的下坡（那是本页第 1 个陷阱）。
    const runs: { start: number; end: number }[] = [];
    let start = 0;
    for (let k = 1; k <= xs.length; k += 1) {
      const gap = k === xs.length || Math.abs((xs[k] as number) - (xs[k - 1] as number)) > 1 + 1e-9;
      if (gap) {
        runs.push({ start, end: k });
        start = k;
      }
    }
    for (const run of runs) {
      const part = sliceRun(xs, ys, run);
      scene.layers.push(
        lineLayer("ratio", series(part.x, part.y, pal.curves[0], { width: 2.2, label: "内部比值" })),
      );
    }
    scene.layers.push(scatterLayer("ratio", xs, ys, { r: 2.6, color: pal.curves[0] }));
  }
  if (flooredIdx.length > 0) {
    const xs = flooredIdx.map((i) => degreeAxis[i] as number);
    const ys = flooredIdx.map((i) => interiorRatio[i] as number);
    scene.layers.push(
      scatterLayer("ratio", Float64Array.from(xs), Float64Array.from(ys), {
        r: 3.2,
        color: pal.warn,
        label: "严格 0（地板占位）",
      }),
    );
  }
  scene.layers.push(
    lineLayer("ratio", series(degreeAxis, boundaryRatio, pal.exact, { width: 1.7, dash: [5, 3], label: "边界比值" })),
    lineLayer("ratio", series(degreeAxis, uLine, pal.reference, { width: 1.1, dash: [3, 3], label: "u=2^-53" })),
  );
  // 当前次数与消失矩阶数的竖直参考线。
  if (degree !== undefined) {
    scene.layers.push(
      lineLayer("ratio", series([degree, degree], [ratioAxis.ylim[0], ratioAxis.ylim[1]], pal.marker, { width: 1.2, dash: [4, 3] })),
    );
  }
  if (moments !== undefined && moments !== degree) {
    scene.layers.push(
      lineLayer(
        "ratio",
        series([moments, moments], [ratioAxis.ylim[0], ratioAxis.ylim[1]], pal.warn, { width: 1.1, dash: [2, 3] }),
      ),
    );
  }

  // 误差副轴。
  scene.layers.push(lineLayer("error", series(xIdx, error, pal.residual, { width: 2 })));
  if (prError !== undefined && prError > 0) {
    const y = Math.max(prError, FLOOR);
    scene.layers.push(
      lineLayer("error", series([errAxis.xlim[0], errAxis.xlim[1]], [y, y], pal.warn, { width: 1, dash: [4, 3] })),
    );
  }

  // —— 图例 ——
  scene.texts.push(
    ...legendBlock(
      L.left + 6,
      L.topY + 6,
      L.leftW,
      `蓝=多项式 t^k；虚线=第 1..J 层细节系数（只画到 detail_length）`,
      pal.muted,
    ),
    ...legendBlock(
      L.left + 6,
      L.botY + 6,
      L.leftW,
      "蓝=内部比值　橙虚=边界比值　灰虚=u=2^-53　红点=实测严格为 0（画在地板上）",
      pal.muted,
    ),
    ...legendBlock(
      L.left + 6,
      L.botY + 23,
      L.leftW,
      `竖线：红=当前 k=${degree ?? "?"}，红点线=消失矩 N=${moments ?? "?"}`,
      pal.muted,
    ),
  );

  // —— 右列文字 ——
  if (L.columns) {
    const head = paramSummary(frame, opts, [{ name: "degree", label: "k" }]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`小波 ${wavelet ?? "?"} / ${mode ?? "?"} 延拓　${head.text}`);
    if (moments !== undefined) lines.push(`消失矩阶数 N=${moments}：k<N 时内部系数应贴 u`);
    if (interiorMax !== undefined && ratioOverU !== undefined) {
      lines.push(
        `当前内部 max|d|/max|a| = ${formatNumber(interiorMax, 4)}` +
          `（${formatNumber(ratioOverU, 3)} u）`,
      );
    }
    if (killed === true) lines.push("判定：k<N，内部细节被抹平（到舍入级）");
    else if (killed === false) lines.push("判定：k≥N，内部细节不再被抹平");
    else degraded.push("帧 meta 缺 killed：不写「是否被抹平」的判定");
    if (boundaryMax !== undefined) lines.push(`边界量级比 ${formatNumber(boundaryMax, 4)}：延拓才是边界系数的来源`);
    if (nFloored !== undefined) {
      lines.push(
        `严格 0 的点 ${Math.round(nFloored)}/${degreeAxis.length}` +
          (nInterior !== undefined ? `，内部有效点 ${Math.round(nInterior)} 个` : ""),
      );
    }
    if (u !== undefined) lines.push(`u = ${formatNumber(u, 3)}（2^-53）`);
    if (level1Ratio !== undefined) lines.push(`第 1 层内部量级比 ${formatNumber(level1Ratio, 3)}（贴 u）`);
    if (pollution !== undefined) {
      lines.push(
        `不做延拓时的边界污染：第 1 层最大偏差 ${formatNumber(pollution, 4)}` +
          (pollutionSpan !== undefined ? `，影响 ${Math.round(pollutionSpan)} 个样本` : ""),
      );
    }
    if (prError !== undefined) lines.push(`本模式往返误差 ${formatNumber(prError, 3)}`);
    if (note !== undefined) lines.push(note);
    if (boundaryNote !== undefined) lines.push(boundaryNote);
    if (maskNote !== undefined) lines.push(`掩码：${maskNote}`);
    if (nonfinitePolicy !== undefined) lines.push(nonfinitePolicy);
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "坐标轴：多项式固定窗口；比值/误差轴自适应", 8, 26), SLOT.axisMode, width),
  );
  if (degree !== undefined && moments !== undefined) {
    const crossing = degree >= moments;
    scene.badges.push(
      placeBadge(
        badge(
          "critical",
          `k=${degree}　消失矩 N=${moments}　${crossing ? "k≥N：不再抹平" : "k<N：被抹平（到 u）"}`,
          8,
          62,
        ),
        SLOT.critical,
        width,
      ),
    );
  }
  const status = [
    wavelet ?? "?",
    `k=${degree ?? "?"}`,
    killed === true ? "已抹平" : killed === false ? "未抹平" : "判定缺失",
    nFloored !== undefined ? `${Math.round(nFloored)} 个地板点` : "",
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(signal) +
    countNonFinite(detailTree) +
    countNonFinite(interiorRatio) +
    countNonFinite(boundaryRatio) +
    countNonFinite(error);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

function lastFinite(a: ArrayLike<number>): number | undefined {
  for (let i = a.length - 1; i >= 0; i -= 1) if (Number.isFinite(a[i])) return a[i];
  return undefined;
}
