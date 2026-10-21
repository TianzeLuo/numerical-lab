/**
 * `decomp_tree`（2.6 章）的 model：系数树 + 当前层的重构 + 重构误差。
 *
 * 这张图最容易骗人的三个地方（都对应到具体图层/字段）：
 *
 * 1. **树的空位不是 0**：`approx_tree` / `detail_tree` 为了铺成矩形补了占位 0，
 *    真实长度在 `coeff_length`（前 J+1 个是近似、后 J+1 个是细节）。模型在每一行
 *    **只画到真实长度为止**；空位一格都不进图层。否则读者会看到"这一层系数全是 0"。
 * 2. **第 0 层细节恒为 0**：它没有对应的细节带，不是"消失矩把它抹平了"。图例明确写
 *    "细节从第 1 层开始"，并且在树里跳过 detail_tree[0]。
 * 3. **t 是层号，不是物理时间**：`meta.stage = round(t·J)`；拖动滑杆只看重构保留了几层。
 *    误差轴上贴 1e-18 地板表示"已到舍入级"，不是"误差恰好等于 1e-18"。
 *
 * 轴的自适应口径与 manifest 一致：树/重构轴**固定窗口**（同一组参数下不随层号变，
 * 否则动画里曲线会跳），误差轴自适应并挂 `autoscale` 徽标。
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
  sceneShell,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  text,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  indexAxis,
  legendBlock,
  linearRange,
  logRange,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "2.6";
const VIZ = "decomp_tree";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "逐层分解树与逐层重构";
const LEVEL_CRITICAL = SPEC?.params?.find((p) => p.name === "level")?.critical ?? [4];
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

/** 近似曲线的颜色循环（最多 7 层；颜色只用显示，不进入任何数值判断）。 */
const LEVEL_COLORS = ["#2b6cb0", "#2f855a", "#b7791f", "#805ad5", "#c53030", "#2c7a7b", "#975a16"] as const;

export const decompTreeModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const signal = requireArray(frame, "signal");
  const approxTree = requireArray(frame, "approx_tree");
  const detailTree = requireArray(frame, "detail_tree");
  const coeffLength = requireArray(frame, "coeff_length");
  const recon = requireArray(frame, "recon");
  const error = requireArray(frame, "error");
  const pitfalls = requirePitfalls(frame);

  // 层号 J 与数组形状必须自洽：树是 (J+1) × width，coeff_length 是 2(J+1)。
  const levelMeta = metaNumber(frame, "level");
  const nMeta = metaNumber(frame, "n");
  const nExtMeta = metaNumber(frame, "n_ext");
  const degraded: string[] = [];
  if (levelMeta === undefined) degraded.push("帧 meta 缺 level：层号按 coeff_length 推断并标注");
  if (nMeta === undefined) degraded.push("帧 meta 缺 n：信号长度只按数组长度显示");
  if (nExtMeta === undefined) degraded.push("帧 meta 缺 n_ext：树的横轴范围按数组宽度显示");
  const level = Math.max(1, Math.round(levelMeta ?? coeffLength.length / 2 - 1));
  if (coeffLength.length !== 2 * (level + 1)) {
    throw new Error(
      `帧 ${CHAPTER}/${VIZ} 的 coeff_length 长度 ${coeffLength.length} 与 level=${level} 不符` +
        `（应为 2×(J+1)=${2 * (level + 1)}）：拒绝画错位的树`,
    );
  }
  const widthExt = Math.max(1, Math.round(nExtMeta ?? approxTree.length / (level + 1)));
  const n = Math.min(signal.length, Math.round(nMeta ?? signal.length));
  if (approxTree.length !== (level + 1) * widthExt) {
    throw new Error(
      `帧 ${CHAPTER}/${VIZ} 的 approx_tree 长度 ${approxTree.length} ≠ (J+1)×n_ext=${(level + 1) * widthExt}`,
    );
  }
  if (detailTree.length !== (level + 1) * widthExt) {
    throw new Error(
      `帧 ${CHAPTER}/${VIZ} 的 detail_tree 长度 ${detailTree.length} ≠ (J+1)×n_ext=${(level + 1) * widthExt}`,
    );
  }
  if (recon.length !== (level + 1) * n) {
    throw new Error(`帧 ${CHAPTER}/${VIZ} 的 recon 长度 ${recon.length} ≠ (J+1)×n=${(level + 1) * n}`);
  }

  // —— meta ——
  const stageMeta = metaNumber(frame, "stage");
  const stage = clamp(Math.round(stageMeta ?? opts.t * level), 0, level);
  if (stageMeta === undefined) degraded.push("帧 meta 缺 stage：层号按请求 t 取整并夹到 [0,J]（写明为请求值）");
  const tActual = metaNumber(frame, "t_actual");
  if (tActual !== undefined && Math.abs(tActual - stage / level) > 1e-9) {
    degraded.push(`请求 t=${formatNumber(tActual, 4)} 映射到层号 ${stage}（帧 meta.t_actual 的实到层）`);
  }
  const stageError = metaNumber(frame, "stage_error");
  const relStageError = metaNumber(frame, "rel_stage_error");
  const prError = metaNumber(frame, "pr_error");
  const mraError = metaNumber(frame, "mra_error");
  const boundaryShare = metaNumber(frame, "boundary_share");
  const energyTop = metaNumber(frame, "energy_top");
  const energyDetail = metaNumber(frame, "energy_detail_sum");
  const maxDetail = metaNumber(frame, "max_detail");
  const redundancy = metaNumber(frame, "redundancy");
  const signalName = stringOf(frame, "signal");
  const wavelet = stringOf(frame, "wavelet");
  const mode = stringOf(frame, "mode");
  const boundaryMode = stringOf(frame, "boundary_mode");
  const maskNote = stringOf(frame, "mask_note");
  const nonfinitePolicy = stringOf(frame, "nonfinite_policy");
  const note = stringOf(frame, "note");

  // 可选数组：缺了只是少画一层"装饰"，如实降级。
  const energyShare = frame.has("energy_share") ? requireArray(frame, "energy_share") : undefined;
  if (energyShare === undefined) degraded.push("帧缺 energy_share：不画各层能量占比柱（不按系数平方自己算）");
  const coeffCount = frame.has("coeff_count") ? requireArray(frame, "coeff_count") : undefined;
  if (coeffCount === undefined) degraded.push("帧缺 coeff_count：不显示各层系数个数");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: stage,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.58, auxH: 128, gap: 68 });

  // —— 每行只取到真实长度（绝不把占位 0 当系数画）——
  const approxLength = (j: number): number => Math.max(0, Math.round(coeffLength[j] ?? 0));
  const detailLength = (j: number): number => Math.max(0, Math.round(coeffLength[level + 1 + j] ?? 0));
  const rowOf = (values: ArrayLike<number>, cols: number, r: number): Float64Array => {
    const out = new Float64Array(cols);
    for (let c = 0; c < cols; c += 1) out[c] = values[r * cols + c] as number;
    return out;
  };

  // 树/重构两轴的固定窗口：只由这一帧的参数决定，不随层号（scrub）变。
  let maxAbs = 0;
  for (let j = 0; j <= level; j += 1) {
    const a = rowOf(approxTree, widthExt, j);
    const d = rowOf(detailTree, widthExt, j);
    for (let c = 0; c < approxLength(j); c += 1) {
      const v = a[c] as number;
      if (Number.isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
    }
    for (let c = 0; c < detailLength(j); c += 1) {
      const v = d[c] as number;
      if (Number.isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
    }
  }
  const coeffScale = maxAbs > 0 ? maxAbs : 1;
  // 每一层有自己的基线（层号 × 间距），系数幅度**不归一化**：同一尺度下比较各层大小。
  const bandGap = 2.3 * coeffScale;
  const bandCount = level + 2;
  const tree = makeAxis("tree", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [0, Math.max(1, widthExt - 1)],
    ylim: [
      -1.2 * coeffScale,
      (bandCount - 1) * bandGap + 1.2 * coeffScale,
    ],
    title: axisTitleOf(SPEC?.axes, "tree", "系数树（近似 / 细节，按层）"),
    ylabel: axisYlabelOf(SPEC?.axes, "tree", "层基线 + 系数（显示坐标）"),
    xlabel: "系数下标（每层只画到 $coeff_length$ 为止）",
    autoscaled: false,
    tickTarget: Math.max(3, Math.round(L.topH / 70)),
  });

  let reconLo = Number.POSITIVE_INFINITY;
  let reconHi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < n; i += 1) {
    const v = signal[i] as number;
    if (Number.isFinite(v)) {
      reconLo = Math.min(reconLo, v);
      reconHi = Math.max(reconHi, v);
    }
  }
  for (let k = 0; k <= level; k += 1) {
    for (let i = 0; i < n; i += 1) {
      const v = recon[k * n + i] as number;
      if (Number.isFinite(v)) {
        reconLo = Math.min(reconLo, v);
        reconHi = Math.max(reconHi, v);
      }
    }
  }
  const reconAxis = makeAxis("recon", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [0, Math.max(1, n - 1)],
    ylim: linearRange(Number.isFinite(reconLo) ? reconLo : -1, Number.isFinite(reconHi) ? reconHi : 1, 0.08),
    title: axisTitleOf(SPEC?.axes, "recon", "逐层重构（当前层）"),
    ylabel: axisYlabelOf(SPEC?.axes, "recon", "x"),
    xlabel: "样本下标",
    autoscaled: false,
  });

  const errLo = positiveMinMax(error)?.[0] ?? FLOOR;
  const errHi = positiveMinMax(error)?.[1] ?? 1;
  const errAxis = makeAxis("error", [L.rightX, L.topY, L.rightW, L.auxH], {
    xlim: [0, Math.max(1, n - 1)],
    ylim: logRange(Math.min(errLo, errHi) * 0.5, Math.max(errHi, FLOOR) * 2),
    yscale: "log",
    title: axisTitleOf(SPEC?.axes, "error", "重构误差 |x_hat - x|"),
    ylabel: axisYlabelOf(SPEC?.axes, "error", "max|delta|"),
    xlabel: "样本下标",
    autoscaled: true,
  });
  scene.axes.push(tree, reconAxis, errAxis);

  // 每层一条极淡的基线 + 左端层名（层名画在面板内，不压 y 刻度带）。
  const yToPixel = (v: number): number =>
    tree.rect[1] + ((tree.ylim[1] - v) / (tree.ylim[1] - tree.ylim[0])) * tree.rect[3];
  const bandName = (b: number): string => {
    if (b === 0) return "a0=x";
    if (b === level + 1) return `a${level}`;
    return `d${b}`;
  };
  for (let b = 0; b < bandCount; b += 1) {
    const base = b * bandGap;
    scene.layers.push(
      lineLayer(
        "tree",
        series([tree.xlim[0], tree.xlim[1]], [base, base], "#e2e8f0", { width: 0.8 }),
      ),
    );
    // 窄画布下面板里放不下"图例 + 层名"两套文字：层名会让给图例（图例里按顺序写了 a0…aJ）。
    if (L.leftW >= 420) {
      scene.texts.push(
        text(tree.rect[0] + tree.rect[2] - 6, yToPixel(base + coeffScale * 0.35), bandName(b), {
          size: 10,
          color: LEVEL_COLORS[b % LEVEL_COLORS.length] as string,
          role: "note",
          align: "right",
        }),
      );
    }
  }
  const treeLegend: string[] = [
    "自下而上：a0=延拓后的原信号，d1…dJ=各层细节，aJ=最粗近似",
    "纵轴=「层基线 + 系数」；系数不做归一化，层间幅度可直接比较",
  ];
  // 第 0 层近似（=延拓后的信号）。
  const a0Len = approxLength(0);
  if (a0Len > 0) {
    const a0 = rowOf(approxTree, widthExt, 0);
    scene.layers.push(
      lineLayer("tree", series(indexAxis(a0Len), a0.subarray(0, a0Len), LEVEL_COLORS[0] as string, { width: 1.8 })),
    );
  }
  // 第 1..J 层细节，各自放在自己的基线上。
  for (let j = 1; j <= level; j += 1) {
    const dLen = detailLength(j);
    if (dLen === 0) continue;
    const d = rowOf(detailTree, widthExt, j);
    const y = new Float64Array(dLen);
    for (let c = 0; c < dLen; c += 1) y[c] = j * bandGap + (d[c] as number);
    scene.layers.push(
      lineLayer("tree", series(indexAxis(dLen), y, LEVEL_COLORS[j % LEVEL_COLORS.length] as string, { width: 1.5 })),
    );
  }
  // 最粗近似 aJ。
  const aJLen = approxLength(level);
  if (aJLen > 0) {
    const aJ = rowOf(approxTree, widthExt, level);
    const y = new Float64Array(aJLen);
    for (let c = 0; c < aJLen; c += 1) y[c] = (level + 1) * bandGap + (aJ[c] as number);
    scene.layers.push(
      lineLayer(
        "tree",
        series(indexAxis(aJLen), y, LEVEL_COLORS[(level + 1) % LEVEL_COLORS.length] as string, { width: 1.8 }),
      ),
    );
  }
  // —— 重构：原信号是参照，当前层是最粗的那条 ——
  const signalX = indexAxis(n);
  const reconCurrent = rowOf(recon, n, stage);
  const reconFull = rowOf(recon, n, level);
  const signalPlot = new Float64Array(n);
  for (let i = 0; i < n; i += 1) signalPlot[i] = signal[i] as number;
  scene.layers.push(
    lineLayer("recon", series(signalX, signalPlot, pal.reference, { width: 1.6, label: "原信号" })),
    lineLayer("recon", series(signalX, reconCurrent, pal.curves[0], { width: 2.3, label: `当前层 ${stage}` })),
  );
  if (stage !== level) {
    scene.layers.push(
      lineLayer("recon", series(signalX, reconFull, pal.exact, { width: 1.3, dash: [5, 4], label: "全重构" })),
    );
  }

  // —— 误差副轴（规则 3）：当前层的重构误差 ——
  const errPlot = new Float64Array(n);
  for (let i = 0; i < n; i += 1) errPlot[i] = error[i] as number;
  scene.layers.push(lineLayer("error", series(signalX, errPlot, pal.residual, { width: 2 })));
  if (stageError !== undefined && stageError > 0) {
    const y = Math.max(stageError, FLOOR);
    scene.layers.push(
      lineLayer(
        "error",
        series([errAxis.xlim[0], errAxis.xlim[1]], [y, y], pal.warn, { width: 1, dash: [4, 3] }),
      ),
    );
    // 当前层号的位置用一个点标出（误差数组本身是常量列，点只做"读数是这个"的索引）。
    scene.layers.push(scatterLayer("error", [0], [y], { r: 3, color: pal.marker, label: `stage ${stage}` }));
  }

  // —— 面板图例 ——
  let legendY = L.topY + 6;
  for (const line of treeLegend) {
    scene.texts.push(...legendBlock(L.left + 6, legendY, L.leftW, line, pal.muted));
    legendY += 17;
  }
  scene.texts.push(
    ...legendBlock(L.left + 6, L.botY + 6, L.leftW, `灰=原信号　蓝=当前层（保留 ${stage}/${level} 层）　橙虚=全重构`, pal.muted),
    ...legendBlock(
      L.left + 6,
      L.botY + 23,
      L.leftW,
      "误差轴是对数轴：贴 1e-18 地板表示已到舍入级，不是误差恰好为 1e-18",
      pal.muted,
    ),
  );

  // —— 右列文字 ——
  if (L.columns) {
    const head = paramSummary(frame, opts, [
      { name: "n", label: "$n$" },
      { name: "level", label: "$J$" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`层号 ${stage}/${level}　${head.text}`);
    if (signalName !== undefined || wavelet !== undefined) {
      lines.push(`信号 ${signalName ?? "?"}　小波 ${wavelet ?? "?"}（${mode ?? "?"} 延拓）`);
    }
    if (boundaryMode !== undefined) lines.push(boundaryMode);
    if (redundancy !== undefined) lines.push(`延拓后 n_ext=${widthExt}，冗余 ${formatNumber(redundancy, 4)}`);
    if (stageError !== undefined) {
      lines.push(
        `当前层误差 ${formatNumber(stageError, 4)}` +
          (relStageError !== undefined ? `（相对 ${formatNumber(relStageError, 3)}）` : ""),
      );
    }
    if (prError !== undefined && mraError !== undefined) {
      lines.push(`往返误差：idwt ${formatNumber(prError, 3)} / 逐带求和 ${formatNumber(mraError, 3)}`);
    }
    if (energyTop !== undefined && energyDetail !== undefined) {
      lines.push(`能量占比：最粗近似 ${formatPercentage(energyTop)}，全部细节 ${formatPercentage(energyDetail)}`);
    }
    if (energyShare !== undefined && energyShare.length > 0) {
      const top = Math.max(...Array.from(energyShare, (v) => (Number.isFinite(v) ? v : 0)));
      const idx = Array.from(energyShare).findIndex((v) => v === top);
      lines.push(`最大的一层能量占比 ${formatPercentage(top)}（第 ${idx} 项：a_J 起）`);
    }
    if (maxDetail !== undefined) lines.push(`第 1 层 |d| 最大 ${formatNumber(maxDetail, 4)}`);
    if (coeffCount !== undefined) {
      const counts = Array.from(coeffCount, (v) => Math.round(v)).join(" / ");
      lines.push(`各层系数个数 ${counts}`);
    }
    if (boundaryShare !== undefined) {
      lines.push(`第 1 层边界系数占细节能量 ${formatPercentage(boundaryShare)}（边界延拓的回声）`);
    }
    if (note !== undefined) lines.push(note);
    if (maskNote !== undefined) lines.push(`掩码：${maskNote}`);
    if (nonfinitePolicy !== undefined) lines.push(nonfinitePolicy);
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge(), SLOT.preview, width));
  }
  scene.badges.push(
    placeBadge(badge("autoscale", "坐标轴：树/重构固定，误差轴自适应（如实标注）", 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(placeBadge(criticalBadge("J", level, LEVEL_CRITICAL), SLOT.critical, width));
  const status = [
    `层 ${stage}/${level}`,
    stageError !== undefined ? `误差 ${formatNumber(stageError, 3)}` : "误差未给",
    boundaryShare !== undefined ? `边界能量 ${formatPercentage(boundaryShare)}` : "",
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(signal) + countNonFinite(approxTree) + countNonFinite(detailTree) + countNonFinite(recon);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

/** 占比的显示口径：小到 0.01% 以下时写 `<0.01%`，而不是四舍五入成 0.00%。 */
function formatPercentage(v: number): string {
  if (!Number.isFinite(v)) return "?";
  const pct = v * 100;
  if (pct > 0 && pct < 0.01) return "<0.01%";
  return `${pct.toFixed(pct >= 10 ? 1 : 2)}%`;
}
