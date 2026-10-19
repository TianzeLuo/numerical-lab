/**
 * `shooting_scan`（3.4 章）的 model：一族初始斜率的轨迹 + 右端打靶残差。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 整族轨迹 `family`（逐行；`family_clip=1` 的格**不画**） |
 * | 2 | line | `main` | 真解 `u_exact`（橙虚线） |
 * | 3 | scatter | `main` | 右端目标 `u(b)=β`（真解端点） |
 * | 4 | line | `aux` | 打靶残差 `residual`（对数轴） |
 * | 5 | line | `aux` | 每条轨迹与真解的最大偏差 `error`（虚线，同一量纲） |
 * | 6 | line | `aux` | 舍入地板 `meta.residual_floor` |
 * | 7 | line | `aux` | 参考斜率 `meta.s_ref`（再打靶一次，不是真值） |
 * | 8 | scatter | `aux` | 当前 j、整族最小残差点 |
 *
 * 三条防误导决定
 * --------------
 * 1. **截断占位不画**：`family` / `residual` / `error` 里被 `_clip` 掩码标出的位置是
 *    `strings.clip_fill`（1e6）占位，不是数据。模型把它们变成缺口（折线断开），
 *    并在溢出角标里给出被截断的个数。
 * 2. **主轴窗口按真解与轨迹的稳健分位取**（q05…q95），飞出窗口的轨迹由
 *    `applyOverflow` 记进 `overflow.note`：读者看到的是「它飞出去了」，而不是
 *    「把 1e6 塞进轴里、真解压成一条线」。
 * 3. **副轴同时画 `residual` 与 `error`**：前者是右端打靶函数的残差，后者是整条轨迹
 *    与真解的偏差；衰减算例里两者一起被 S 放大，而「残差贴地板」并不等于「解准」。
 *
 * 时间轴：`manifest.time.mode="scrub"`，t 是斜率序号 j（不是物理时间）。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
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
  segmentedSeries,
  stringOf,
  ModelInputError,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
  overflowBadge,
  type LineSeries,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  finiteMinMax,
  legendBlock,
  logRange,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.4";
const VIZ = "shooting_scan";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "打靶法：初始斜率扫描与轨迹族";
const KRIT = SPEC?.params?.find((p) => p.name === "k")?.critical ?? [18, 36];

/** 固定角标槽位：缺一枚时其余不挪位。 */
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

export const shootingScanModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const x = requireArray(frame, "x");
  const familyArr = requireArray(frame, "family");
  requireArray(frame, "family_clip");
  const exact = requireArray(frame, "u_exact");
  const slopes = requireArray(frame, "slopes");
  const residual = requireArray(frame, "residual");
  const residualClip = requireArray(frame, "residual_clip");
  const error = requireArray(frame, "error");
  const errorClip = requireArray(frame, "error_clip");
  const pitfalls = requirePitfalls(frame);

  const fam = frame.matrix("family");
  const famClip = frame.matrix("family_clip");
  const m = fam.rows;
  const nx = fam.cols;
  if (!(m > 0) || !(nx > 0) || fam.values.length !== m * nx) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 family 形状 [${m}, ${nx}] 与数据长度不符（拒绝画错位的轨迹族）`);
  }
  if (famClip.rows !== m || famClip.cols !== nx) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 family_clip 形状 [${famClip.rows}, ${famClip.cols}] 与 family [${m}, ${nx}] 不一致`);
  }
  if (familyArr.length !== m * nx) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 family 数据长度 ${familyArr.length} ≠ m×nx=${m * nx}`);
  }
  requireSameLength(x, exact, "帧数组 x 与 u_exact");
  if (x.length !== nx) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 x 长度 ${x.length} ≠ family 列数 ${nx}（拒绝画错位的曲线）`);
  }
  for (const [name, arr] of [
    ["slopes", slopes],
    ["residual", residual],
    ["residual_clip", residualClip],
    ["error", error],
    ["error_clip", errorClip],
  ] as const) {
    if (arr.length !== m) {
      throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 ${name} 长度 ${arr.length} ≠ 轨迹条数 ${m}`);
    }
  }

  // —— 可选 meta / strings：缺了只降级，不猜 ——
  const degraded: string[] = [];
  const mMeta = metaNumber(frame, "m");
  if (mMeta !== undefined && Math.round(mMeta) !== m) {
    degraded.push(`帧 meta.m=${Math.round(mMeta)} 与 family 行数 ${m} 不一致：按数组行数画`);
  }
  const jMeta = metaNumber(frame, "j");
  const j = clamp(jMeta ?? opts.t, 0, Math.max(0, m - 1));
  if (jMeta === undefined) degraded.push("帧 meta 缺 j：时间轴按请求 t 夹到 [0,m-1]（写明为请求值）");
  const nodeMeta = metaNumber(frame, "node_index");
  const node = clamp(Math.round(nodeMeta ?? j), 0, Math.max(0, m - 1));
  if (nodeMeta === undefined) degraded.push("帧 meta 缺 node_index：高亮轨迹按 round(j) 取");
  const sRef = metaNumber(frame, "s_ref");
  if (sRef === undefined) degraded.push("帧 meta 缺 s_ref：不画参考斜率竖线（不按扫描中点猜）");
  const slopeNode = metaNumber(frame, "slope_node");
  const sensitivity = metaNumber(frame, "sensitivity");
  const digitsLost = metaNumber(frame, "digits_lost");
  const kL = metaNumber(frame, "kL");
  const kMeta = metaNumber(frame, "k");
  const lMeta = metaNumber(frame, "L");
  const residualFloor = metaNumber(frame, "residual_floor");
  const residualMin = metaNumber(frame, "residual_min");
  const residualRef = metaNumber(frame, "residual_ref");
  const errorRef = metaNumber(frame, "error_ref");
  const nBlowup = metaNumber(frame, "n_blowup");
  const clipFill = metaNumber(frame, "clip_fill");
  const frameOverflow = (metaNumber(frame, "overflow") ?? 0) > 0.5;
  const problem = stringOf(frame, "problem");
  const label = stringOf(frame, "label");

  // —— 截断掩码的计数（规则 1 要的是计数，不是估计）——
  let nFamilyClip = 0;
  for (let i = 0; i < famClip.values.length; i += 1) if ((famClip.values[i] ?? 0) > 0.5) nFamilyClip += 1;
  let nResidualClip = 0;
  for (let i = 0; i < residualClip.length; i += 1) if ((residualClip[i] ?? 0) > 0.5) nResidualClip += 1;
  let nErrorClip = 0;
  for (let i = 0; i < errorClip.length; i += 1) if ((errorClip[i] ?? 0) > 0.5) nErrorClip += 1;
  const nClip = nFamilyClip + nResidualClip + nErrorClip;
  if (nClip > 0) degraded.push(`帧内 ${nClip} 个点被截断（占位 ${formatNumber(clipFill ?? 1e6, 3)} 未画，掩码已生效）`);

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: j,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.999, auxH: 182, gap: 0 });
  const topH = L.floor - L.topY < 300 ? Math.max(60, L.floor - L.topY - 28) : L.topH;
  const auxH = Math.max(52, Math.min(L.auxH, topH));

  // —— 轨迹数据：把截断掩码变成缺口（NaN 只活在局部数组里，segmentedSeries 会切段）——
  const rowY = (row: number): Float64Array => {
    const out = new Float64Array(nx);
    for (let c = 0; c < nx; c += 1) {
      const v = fam.values[row * nx + c] as number;
      const clipped = (famClip.values[row * nx + c] ?? 0) > 0.5;
      out[c] = clipped || !Number.isFinite(v) ? Number.NaN : v;
    }
    return out;
  };
  const maskX = (values: ArrayLike<number>, clip: ArrayLike<number>, fallback: ArrayLike<number>): Float64Array => {
    const out = new Float64Array(values.length);
    for (let i = 0; i < values.length; i += 1) {
      const v = values[i] as number;
      const clipped = (clip[i] ?? 0) > 0.5;
      out[i] = clipped || !Number.isFinite(v) ? Number.NaN : fallback[i] as number;
    }
    return out;
  };

  // —— 主轴窗口：真解的取值范围（飞出去的轨迹交给 overflow）——
  // 为什么不按整族的分位取：kL=10 时整族到右端能长到 1e3，按分位取会把真解压成
  // 一条贴在 0 上的直线，"只有一条轨迹贴着真解"这件事反而看不见。
  const exactRange = finiteMinMax(exact);
  let yLo = exactRange?.[0] ?? -1;
  let yHi = exactRange?.[1] ?? 1;
  if (!Number.isFinite(yLo) || !Number.isFinite(yHi)) {
    yLo = -1;
    yHi = 1;
  }
  if (!(yHi > yLo)) {
    const c = Math.max(1, Math.abs(yLo) * 0.1);
    yLo -= c;
    yHi += c;
  }
  const padY = (yHi - yLo) * 0.25;
  const xMin = Math.min(x[0] as number, x[x.length - 1] as number);
  const xMax = Math.max(x[0] as number, x[x.length - 1] as number);
  const x0 = Number.isFinite(xMin) ? xMin : 0;
  const x1 = Number.isFinite(xMax) && xMax > x0 ? xMax : x0 + 1;

  // —— 副轴窗口：残差 / 轨迹误差 / 地板都是 > 0 的同一量纲 ——
  const auxPos = positiveMinMax([
    ...Array.from(residual, (v, i) => ((residualClip[i] ?? 0) > 0.5 ? Number.NaN : v)),
    ...Array.from(error, (v, i) => ((errorClip[i] ?? 0) > 0.5 ? Number.NaN : v)),
    residualFloor ?? Number.NaN,
  ]);
  const auxLo = auxPos?.[0] ?? 1e-16;
  const auxHi = auxPos?.[1] ?? 1;
  const slopeVals: number[] = [];
  for (let i = 0; i < slopes.length; i += 1) if (Number.isFinite(slopes[i])) slopeVals.push(slopes[i] as number);
  const sMin = slopeVals.length > 0 ? Math.min(...slopeVals) : 0;
  const sMax = slopeVals.length > 0 ? Math.max(...slopeVals) : 1;
  const sPad = Math.max((sMax - sMin) * 0.05, 1e-12);
  const sX0 = sMin - sPad;
  const sX1 = sMax + sPad;

  const main = makeAxis("main", [L.left, L.topY, L.leftW, topH], {
    xlim: [x0, x1],
    ylim: [yLo - padY, yHi + padY],
    title: L.columns
      ? axisTitleOf(SPEC?.axes, "main", "轨迹族 u(x; s_j) 与真解")
      : "轨迹族与真解",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "u"),
    xlabel: "x",
    autoscaled: false,
  });
  const aux = makeAxis("aux", [L.rightX, L.topY, L.rightW, auxH], {
    xlim: [sX0, sX1],
    ylim: logRange(auxLo * 0.5, auxHi * 2),
    yscale: "log",
    title: L.columns
      ? axisTitleOf(SPEC?.axes, "aux", "右端打靶残差 |F(s_j)| = |u(b;s_j) - beta|（对数轴）")
      : "打靶残差 |F|（对数）",
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "|F|"),
    xlabel: "s_j",
    autoscaled: false,
  });
  scene.axes.push(main, aux);

  // —— 图层 1~3：整族轨迹、真解、右端目标 ——
  const familySeries: LineSeries[] = [];
  for (let k = 0; k < m; k += 1) {
    const isNode = k === node;
    familySeries.push(
      ...segmentedSeries(x, rowY(k), isNode ? pal.marker : pal.curves[0], {
        width: isNode ? 2.5 : 1.2,
        alpha: isNode ? 1 : 0.55,
      }),
    );
  }
  if (familySeries.length > 0) scene.layers.push(lineLayer("main", familySeries));
  scene.layers.push(
    lineLayer("main", segmentedSeries(x, exact, pal.exact, { width: 1.7, dash: [7, 4], label: "真解 u(x)" })),
  );
  if (Number.isFinite(x1) && Number.isFinite(exact[exact.length - 1] as number)) {
    scene.layers.push(
      scatterLayer("main", [x1], [exact[exact.length - 1] as number], {
        r: 4,
        color: pal.exact,
      }),
    );
  }

  // —— 副轴：残差 / 轨迹误差 / 地板 / 参考斜率 / 命中点 ——
  const residualY = maskX(residual, residualClip, residual);
  const errorY = maskX(error, errorClip, error);
  const auxSeries = [
    ...segmentedSeries(slopes, residualY, pal.residual, { width: 2.2, marker: "o" }),
    ...segmentedSeries(slopes, errorY, pal.warn, { width: 1.3, dash: [5, 4] }),
  ];
  if (auxSeries.length > 0) scene.layers.push(lineLayer("aux", auxSeries));
  if (residualFloor !== undefined && residualFloor > 0) {
    scene.layers.push(
      lineLayer(
        "aux",
        series([aux.xlim[0], aux.xlim[1]], [residualFloor, residualFloor], pal.reference, {
          width: 1,
          dash: [2, 4],
          
        }),
      ),
    );
  }
  if (sRef !== undefined) {
    scene.layers.push(
      lineLayer("aux", series([sRef, sRef], aux.ylim, pal.reference, { width: 1, dash: [4, 3],  })),
    );
  }
  // 当前 j 的残差点。
  const nodeResidual = residual[node] as number;
  if ((residualClip[node] ?? 0) <= 0.5 && Number.isFinite(nodeResidual) && nodeResidual > 0) {
    scene.layers.push(scatterLayer("aux", [slopes[node] as number], [nodeResidual], { r: 4, color: pal.marker }));
  }
  // 整族最小残差点（命中点）。
  let best = -1;
  let bestVal = Number.POSITIVE_INFINITY;
  for (let i = 0; i < m; i += 1) {
    const v = residual[i] as number;
    if ((residualClip[i] ?? 0) > 0.5 || !Number.isFinite(v) || v <= 0) continue;
    if (v < bestVal) {
      bestVal = v;
      best = i;
    }
  }
  if (best >= 0) {
    scene.layers.push(
      scatterLayer("aux", [slopes[best] as number], [bestVal], {
        r: 5,
        color: pal.exact,
      }),
    );
  }

  // —— 面板图例 ——
  scene.texts.push(
    ...legendBlock(
      L.left + 6,
      L.topY + 6,
      L.leftW,
      `蓝细=整族 s_j　红粗=当前 j　橙虚=真解　${m} 条轨迹`,
      pal.muted,
    ),
  );
  scene.texts.push(
    ...legendBlock(L.rightX + 4, L.topY + auxH - 21, L.rightW, "紫实=残差|F|　红虚=轨迹偏差", pal.muted),
  );

  // —— 右列文字 ——
  if (L.columns) {
    const head = paramSummary(frame, opts, [
      { name: "k", label: "k" },
      { name: "L", label: "L" },
      { name: "span", label: "span" },
      { name: "m", label: "m" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`j=${formatNumber(j, 3)}/${m - 1}　${head.text}　${problem ?? "?"}`);
    if (label !== undefined) lines.push(label);
    if (sensitivity !== undefined) {
      lines.push(
        `S=${formatNumber(sensitivity, 3)} → 舍入被放大 ${formatNumber(digitsLost ?? 0, 3)} 位` +
          (kL !== undefined ? `（kL=${formatNumber(kL, 4)}）` : ""),
      );
    }
    if (sRef !== undefined) lines.push(`s_ref=${formatNumber(sRef, 5)}：再打靶一次得到的对照量，不是真值`);
    const nodeResidualShown = (residualClip[node] ?? 0) > 0.5 ? undefined : nodeResidual;
    lines.push(
      `当前 s=${formatNumber(slopeNode ?? (slopes[node] as number), 5)}，` +
        (nodeResidualShown !== undefined
          ? `残差 ${formatNumber(nodeResidualShown, 3)}`
          : "残差被截断（掩码，不把 1e6 占位当实测）"),
    );
    if (best >= 0 && residualMin !== undefined) {
      lines.push(`整族最小残差 ${formatNumber(residualMin, 3)}（地板 ${formatNumber(residualFloor ?? 1e-16, 3)}）`);
    } else if (nResidualClip >= m) {
      lines.push("副轴残差全部被截断（占位 1e6 未画）：没有命中点可读");
    }
    if (residualRef !== undefined || errorRef !== undefined) {
      lines.push(
        `参考打靶残差 ${formatNumber(residualRef ?? 0, 3)}；真解误差 ${formatNumber(errorRef ?? 0, 3)}`,
      );
    }
    if (nBlowup !== undefined && nBlowup > 0) lines.push(`⚠ 飞出 BLOWUP 的轨迹 ${formatNumber(nBlowup, 4)} 条（掩码已标）`);
    if (nClip > 0) lines.push(`截断显示 ${nClip} 点：占位值不进图层，真量级见 meta.*_max`);
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "坐标轴：主轴固定为真解窗口（飞出轨迹标 overflow）；残差轴固定", 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(badge("status", "副轴：残差与轨迹偏差共用对数标尺（同一量纲）", 8, 44), SLOT.aux, width),
  );
  if (kL !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("kL", kL, KRIT, { beyond: "above" }), SLOT.critical, width));
  } else if (kMeta !== undefined && lMeta !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("kL", kMeta * lMeta, KRIT, { beyond: "above" }), SLOT.critical, width));
  } else {
    degraded.push("帧 meta 缺 kL/k/L：临界值角标未画（不按参数猜）");
  }
  const status = [
    `j=${formatNumber(j, 2)}/${m - 1}`,
    sensitivity !== undefined ? `S=${formatNumber(sensitivity, 3)}` : "S 未给",
    digitsLost !== undefined ? `失 ${formatNumber(digitsLost, 2)} 位` : "",
    nClip > 0 ? `截断 ${nClip}` : "",
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(x) +
    countNonFinite(familyArr) +
    countNonFinite(exact) +
    countNonFinite(slopes) +
    countNonFinite(residual) +
    countNonFinite(error);
  const report = applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  if (nClip > 0 || frameOverflow) {
    const note = [
      nClip > 0 ? `${nClip} 个点被截断显示（占位 ${formatNumber(clipFill ?? 1e6, 3)} 未画）` : "",
      frameOverflow ? (stringOf(frame, "overflow") ?? "帧 meta.overflow=true") : "",
    ]
      .filter((s) => s !== "")
      .join("；");
    scene.overflow.flag = true;
    scene.overflow.note = scene.overflow.note === "" ? note : `${scene.overflow.note}；${note}`;
    if (!report.flag) scene.badges.push(placeBadge(overflowBadge(note), SLOT.overflow, width));
  }
  return scene;
};
