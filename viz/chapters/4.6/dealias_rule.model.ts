/**
 * `dealias_rule`（4.6 章 viz 2）的 model：3/2 去混叠的三项账。
 *
 * 这张图最容易读反的三件事（都对应一个显式图层）
 * ----------------------------------------------
 * 1. **混叠落在哪**：副图 `alias` 画 `|W_naive-W_ref|` 折叠到 |k| 的谱——它铺满整个
 *    保留带，不只碰高波数。主图 `spectrum` 把「本尊」|p+q|=2N/3 与「别名」
 *    |p+q-N|=N/3 **都画在波数轴上**（x 轴是 k，不是幅度；绝不放进比值轴）。
 *    2N/3 > Nyquist N/2，所以主图 x 轴特意延伸到 2N/3，Nyquist 右侧留一条灰带：
 *    那里没有网格数据——这正是折返发生的原因。
 * 2. **截断的代价与混叠是两笔账**：扫描图上 `scan_alias`（真混叠）与
 *    `scan_truncation`（2/3 截断主动丢掉的）分开画；两者的比值写进右列文字。
 * 3. **只补零不截断不是「白补」**：`scan_pad_only` 单独一条线；残差集中在
 *    |k|=N/2 的边界模态（meta.cur_nyquist_rel），这是 3/2 规则与 2/3 规则可以
 *    分开使用的证据。地板、窗口与 k=0 的处理都在图例/文字里说明。
 *
 * 帧 `time.mode="none"`：`scene.t` 恒为 0。
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
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  bandLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  SAFE_GAP,
  TEXT_ROW_H,
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  countMask,
  legendBlock,
  selectValues,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.6";
const VIZ = "dealias_rule";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "3/2 规则：混叠去哪了、代价是什么";

/** 帧约定的地板（meta 优先；缺 meta 时用登记常量并降级）。 */
const COEF_FLOOR_FALLBACK = 1e-18;
const SCAN_FLOOR_FALLBACK = 1e-17;

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

export const dealiasRuleModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const kAbs = requireArray(frame, "k_abs");
  const prodRef = requireArray(frame, "prod_ref");
  const prodCur = requireArray(frame, "prod_cur");
  const aliasErr = requireArray(frame, "alias_error_spectrum");
  const scanDecay = requireArray(frame, "scan_decay");
  const scanAlias = requireArray(frame, "scan_alias");
  const scanTrunc = requireArray(frame, "scan_truncation");
  const scanPad = requireArray(frame, "scan_pad_only");
  const scanDeal = requireArray(frame, "scan_dealias");
  requireSameLength(kAbs, prodRef, "帧数组 k_abs 与 prod_ref");
  requireSameLength(kAbs, prodCur, "帧数组 k_abs 与 prod_cur");
  requireSameLength(kAbs, aliasErr, "帧数组 k_abs 与 alias_error_spectrum");
  for (const [name, arr] of [
    ["scan_alias", scanAlias],
    ["scan_truncation", scanTrunc],
    ["scan_pad_only", scanPad],
    ["scan_dealias", scanDeal],
  ] as const) {
    requireSameLength(scanDecay, arr, `帧数组 scan_decay 与 ${name}`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 / meta ——
  const degraded: string[] = [];
  const prodMask = frame.has("prod_floor_mask") ? requireArray(frame, "prod_floor_mask") : undefined;
  if (prodMask === undefined) degraded.push("帧缺 prod_floor_mask：不标「两条谱都在地板」的模态");
  const aliasMask = frame.has("alias_error_mask") ? requireArray(frame, "alias_error_mask") : undefined;
  if (aliasMask === undefined) degraded.push("帧缺 alias_error_mask：不标「偏差低于地板」的模态");
  const scanMask = frame.has("scan_floor_mask") ? requireArray(frame, "scan_floor_mask") : undefined;
  if (scanMask === undefined) degraded.push("帧缺 scan_floor_mask：不标「该 decay 有曲线在地板」的扫描点");
  const keepLine = frame.has("k_keep_line") ? requireArray(frame, "k_keep_line") : undefined;
  if (keepLine === undefined) {
    degraded.push("帧缺 k_keep_line：不画 2/3 带边缘与本尊/别名频率（不按 N 猜位置）");
  }

  const floorMeta = metaNumber(frame, "error_floor");
  const floor = floorMeta !== undefined && floorMeta > 0 ? floorMeta : COEF_FLOOR_FALLBACK;
  if (floorMeta === undefined) degraded.push("帧 meta 缺 error_floor：地板用登记常量 1e-18");
  const nMeta = metaNumber(frame, "n");
  const half = lastFinite(kAbs);
  const n = nMeta ?? (half !== undefined ? 2 * half : undefined);
  if (nMeta === undefined) degraded.push("帧 meta 缺 n：横轴上限按 k_max×2 推得（只用于显示范围）");

  const mode = stringOf(frame, "mode");
  const exactPad = frame.meta["exact_pad"] === true || (metaNumber(frame, "exact_pad") ?? 0) > 0.5;
  const mUsed = metaNumber(frame, "m_used");
  const aliasRel = metaNumber(frame, "alias_rel");
  const aliasLowRel = metaNumber(frame, "alias_low_rel");
  const truncRel = metaNumber(frame, "truncation_rel");
  const dealiasRel = metaNumber(frame, "dealias_rel");
  const padOnlyRel = metaNumber(frame, "pad_only_rel");
  const curNyqRel = metaNumber(frame, "cur_nyquist_rel");
  const energyNaive = metaNumber(frame, "energy_resid_naive");
  const energyDealias = metaNumber(frame, "energy_resid_dealias");
  const decayNow = metaNumber(frame, "decay");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });
  const L = columnsLayout(scene, width, height, { topShare: 0.55, auxH: 130, gap: 46 });
  const panelTextFits = L.floor > L.botY + L.botH + 8;

  // —— 主图：只保留 k>0（对数 x 轴；k=0 均值模态写进图例说明）——
  const keep: number[] = [];
  for (let i = 0; i < kAbs.length; i += 1) {
    const kv = kAbs[i] as number;
    if (Number.isFinite(kv) && kv > 0) keep.push(i);
  }
  if (keep.length === 0) {
    throw new Error(`帧 ${CHAPTER}/${VIZ} 的 k_abs 没有正波数：双对数主图无法成立`);
  }
  const kx = selectValues(kAbs, keep);
  const refY = selectValues(prodRef, keep);
  const curY = selectValues(prodCur, keep);
  const errY = selectValues(aliasErr, keep);
  const kLast = kx[kx.length - 1] as number;
  const kLo = kx[0] as number;

  // 「本尊」频率 |p+q|=2N/3：N/3 由帧数组 k_keep_line 给（不是前端发明）。
  const kKeep = firstFinite(keepLine);
  const kTruth = kKeep !== undefined ? 2 * kKeep : undefined;
  const xHi = kTruth !== undefined && kTruth > kLast ? Math.max(kLast * 1.02, kTruth * 1.05) : kLast * 1.02;

  const spectrum = makeAxis("spectrum", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [kLo, xHi],
    ylim: [floor, 10],
    xscale: "log",
    yscale: "log",
    xlabel: "k（波数；x 轴不是幅度）",
    title: axisTitleOf(SPEC?.axes, "spectrum", "乘积谱 |W_k|（双对数，1e-18–10）"),
    ylabel: axisYlabelOf(SPEC?.axes, "spectrum", "|W_k|"),
    autoscaled: false,
  });
  const aliasAxis = makeAxis("alias", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [kLo, Math.max(kLo * 1.2, kLast)],
    ylim: [floor, 10],
    xscale: "log",
    yscale: "log",
    xlabel: "k（波数）",
    title: axisTitleOf(SPEC?.axes, "alias", "混叠偏差 |dW_k| 折叠（1e-18–10）"),
    ylabel: axisYlabelOf(SPEC?.axes, "alias", "|dW_k|"),
    autoscaled: false,
  });
  const scanAxis = makeAxis("scan", [L.rightX, L.topY, L.rightW, L.auxH], {
    xlim: [scanDecay[0] as number, scanDecay[scanDecay.length - 1] as number],
    ylim: [SCAN_FLOOR_FALLBACK, 1],
    yscale: "log",
    xlabel: "输入谱指数 decay",
    title: axisTitleOf(SPEC?.axes, "scan", "相对偏差 vs decay（双对数）"),
    ylabel: axisYlabelOf(SPEC?.axes, "scan", "相对偏差"),
    autoscaled: false,
  });
  scene.axes.push(spectrum, aliasAxis, scanAxis);

  // —— 主图：Nyquist 右侧灰带（无网格数据）+ 三条频率竖线 ——
  if (kTruth !== undefined && kTruth > kLast) {
    scene.layers.push(
      bandLayer("spectrum", {
        x: [kLast, spectrum.xlim[1]],
        ylo: [floor, floor],
        yhi: [10, 10],
        color: pal.reference,
        alpha: 0.08,
        label: "Nyquist 右侧：网格没有数据",
      }),
    );
  }
  const vline = (axisId: string, x: number, color: string, dash: number[], label?: string): void => {
    if (!Number.isFinite(x) || x < spectrum.xlim[0] || x > spectrum.xlim[1]) return;
    scene.layers.push(
      lineLayer(
        axisId,
        series([x, x], [floor, 10], color, { dash, width: 1.2, label }),
      ),
    );
  };
  // 频率位置只画在**波数轴**上（不是比值轴）：实线=别名 N/3，虚线=本尊 2N/3，点线=Nyquist。
  if (kKeep !== undefined) {
    vline("spectrum", kKeep, pal.warn, [6, 3]);
  }
  if (kTruth !== undefined) {
    vline("spectrum", kTruth, pal.exact, [4, 3]);
  }
  vline("spectrum", kLast, pal.reference, [1, 3]);
  scene.layers.push(
    lineLayer("spectrum", series([spectrum.xlim[0], spectrum.xlim[1]], [floor, floor], pal.warn, { dash: [3, 3], width: 1, label: "谱地板" })),
    lineLayer("spectrum", series(kx, refY, pal.reference, { dash: [5, 4], width: 1.5, label: "参考谱" })),
    lineLayer("spectrum", series(kx, curY, pal.curves[0], { width: 2.2, label: "当前网格上的谱" })),
  );
  if (prodMask !== undefined && prodMask.length === kAbs.length) {
    const idx: number[] = [];
    for (const i of keep) if ((prodMask[i] ?? 0) > 0.5) idx.push(i);
    if (idx.length > 0) {
      scene.layers.push(
        scatterLayer("spectrum", selectValues(kAbs, idx), selectValues(prodRef, idx), {
          r: 2.4,
          color: pal.warn,
          label: "两条谱都在地板",
        }),
      );
    }
  }

  // —— 混叠偏差（error 副轴）：它铺满整个保留带 ——
  scene.layers.push(
    lineLayer("alias", series([aliasAxis.xlim[0], aliasAxis.xlim[1]], [floor, floor], pal.warn, { dash: [3, 3], width: 1, label: "偏差地板" })),
    lineLayer("alias", series(kx, errY, pal.residual, { width: 2.2, label: "混叠偏差" })),
  );
  if (kKeep !== undefined && kKeep >= aliasAxis.xlim[0] && kKeep <= aliasAxis.xlim[1]) {
    scene.layers.push(
      lineLayer("alias", series([kKeep, kKeep], [floor, 10], pal.warn, { dash: [6, 3], width: 1.2 })),
    );
  }
  if (aliasMask !== undefined && aliasMask.length === kAbs.length) {
    const idx: number[] = [];
    for (const i of keep) if ((aliasMask[i] ?? 0) > 0.5) idx.push(i);
    if (idx.length > 0) {
      scene.layers.push(
        scatterLayer("alias", selectValues(kAbs, idx), selectValues(aliasErr, idx), {
          r: 2.4,
          color: pal.warn,
          label: "偏差低于地板",
        }),
      );
    }
  }

  // —— 扫描图：四条账分开画（截断 ≠ 混叠）——
  scene.layers.push(
    lineLayer("scan", series([scanAxis.xlim[0], scanAxis.xlim[1]], [SCAN_FLOOR_FALLBACK, SCAN_FLOOR_FALLBACK], pal.reference, { dash: [3, 3], width: 1 })),
    lineLayer("scan", series(scanDecay, scanAlias, pal.warn, { width: 2, label: "混叠" })),
    lineLayer("scan", series(scanDecay, scanTrunc, pal.exact, { width: 2, dash: [6, 3], label: "截断" })),
    lineLayer("scan", series(scanDecay, scanPad, pal.curves[1], { width: 1.6, dash: [2, 3], label: "补零" })),
    lineLayer("scan", series(scanDecay, scanDeal, pal.curves[0], { width: 2.2, label: "去混叠" })),
  );
  if (decayNow !== undefined && decayNow >= scanAxis.xlim[0] && decayNow <= scanAxis.xlim[1]) {
    scene.layers.push(
      lineLayer("scan", series([decayNow, decayNow], [SCAN_FLOOR_FALLBACK, 1], pal.marker, { dash: [4, 3], width: 1.1 })),
    );
  }
  if (scanMask !== undefined && scanMask.length === scanDecay.length) {
    const idx: number[] = [];
    for (let i = 0; i < scanMask.length; i += 1) if ((scanMask[i] ?? 0) > 0.5) idx.push(i);
    if (idx.length > 0) {
      scene.layers.push(
        scatterLayer("scan", selectValues(scanDecay, idx), selectValues(scanDeal, idx), {
          r: 2.4,
          color: pal.warn,
          label: "该 decay 至少一条曲线在地板",
        }),
      );
    }
  }

  // —— 面板内图例（单块 textBlock，按面板宽度顺序折行，构造上互不重叠）——
  if (panelTextFits) {
    const legend: string[] = [];
    legend.push("灰虚=全带参照　蓝实=当前模式　红线=地板");
    if (kKeep !== undefined && kTruth !== undefined) {
      legend.push(
        `频率竖线：实线=别名 N/3=${formatNumber(kKeep, 3)}；虚线=本尊 2N/3=${formatNumber(kTruth, 3)}；点线=Nyquist`,
      );
      legend.push("本尊在 Nyquist 右侧（灰带无数据）→ 折回保留带 N/3；x 轴是波数，不是幅度");
    } else {
      legend.push("k=0 均值模态不进对数横轴；2/3 边缘位置缺帧字段，未画");
    }
    scene.texts.push(...textBlock(L.left + 6, L.topY + L.topH - SAFE_GAP - 8 * TEXT_ROW_H, L.topY + L.topH - SAFE_GAP - TEXT_ROW_H, legend, pal.muted, unitBudget(L.leftW), "bottom"));
    scene.texts.push(
      ...legendBlock(L.left + 6, L.botY + 6, L.leftW, "紫实=混叠偏差　红线=地板　竖线=N/3", pal.muted),
    );
    scene.texts.push(
      ...textBlock(
        L.rightX + 4,
        L.topY + L.auxH - SAFE_GAP - 7 * TEXT_ROW_H,
        L.topY + L.auxH - SAFE_GAP - TEXT_ROW_H,
        ["红=真混叠　橙虚=截断代价　绿虚=只补零　蓝=2/3+3/2", "紫竖线=当前 decay"],
        pal.muted,
        unitBudget(L.rightW),
        "bottom",
      ),
    );
  } else {
    degraded.push("画布过矮/过窄：面板说明与右列文字放不下（pitfalls 面板占满底部），关键读数见角标");
  }

  // —— 右列文字 ——
  if (L.columns && L.floor - L.textTop >= 32) {
    const lines: string[] = [];
    lines.push(`N=${n !== undefined ? Math.round(n) : "?"} ${mode ?? "?"} M=${mUsed !== undefined ? Math.round(mUsed) : "?"}（${exactPad ? "精确 3N/2" : "2 的幂"}）`);
    if (kKeep !== undefined) lines.push(`保留带 k≤${formatNumber(kKeep, 3)}；k=0 不进对数轴`);
    if (aliasRel !== undefined) {
      lines.push(
        `混叠 ${formatNumber(aliasRel, 3)}` +
          (aliasLowRel !== undefined ? `（低带 ${formatNumber(aliasLowRel, 3)}）` : ""),
      );
    }
    if (truncRel !== undefined && aliasRel !== undefined && aliasRel > 0) {
      lines.push(`截断 ${formatNumber(truncRel, 3)} = ${formatNumber(truncRel / aliasRel, 3)}× 混叠`);
    }
    if (padOnlyRel !== undefined) {
      lines.push(
        `只补零 ${formatNumber(padOnlyRel, 3)}` +
          (curNyqRel !== undefined ? `（边界 ${formatNumber(curNyqRel, 3)}）` : ""),
      );
    }
    if (dealiasRel !== undefined) lines.push(`3/2 残差 ${formatNumber(dealiasRel, 3)}：贴舍入`);
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else if (L.columns) {
    degraded.push("画布过矮：右列文字放不下（pitfalls 面板之下），关键读数见角标");
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键读数见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "窗口固定：谱/混叠 1e-18–10 · 扫描 1e-17–1", 8, 26), SLOT.aux, width),
  );
  const status: string[] = [];
  status.push(`mode=${mode ?? "?"}`);
  if (aliasRel !== undefined && truncRel !== undefined && aliasRel > 0) {
    status.push(`截断 ${formatNumber(truncRel / aliasRel, 3)}× 混叠`);
  }
  if (dealiasRel !== undefined) status.push(`3/2 残差 ${formatNumber(dealiasRel, 2)}`);
  scene.badges.push(placeBadge(badge("status", status.join("；")), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(kAbs) +
    countNonFinite(prodRef) +
    countNonFinite(prodCur) +
    countNonFinite(aliasErr) +
    countNonFinite(scanAlias) +
    countNonFinite(scanTrunc) +
    countNonFinite(scanPad) +
    countNonFinite(scanDeal);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  // 保留带扫描点数只用于状态条（帮助判断是不是只有一两个点）。
  const scanFloorCount = countMask(scanMask ?? new Float64Array(0));
  const caveat: string[] = [];
  if (energyNaive !== undefined && energyDealias !== undefined) caveat.push("能量残差=舍入级，不是去混叠证据");
  if (scanFloorCount > 0) caveat.push(`${Math.round(scanFloorCount)} 个 decay 有曲线在地板`);
  if (caveat.length > 0) {
    scene.badges.push(placeBadge(badge("status", caveat.join("；")), SLOT.preview, width));
  }
  return scene;
};

/** 数组里最后一个有限值（k_abs 升序；用来从帧数据推显示上限，不参与物理量）。 */
function lastFinite(a: ArrayLike<number>): number | undefined {
  for (let i = a.length - 1; i >= 0; i -= 1) {
    const v = a[i] as number;
    if (Number.isFinite(v)) return v;
  }
  return undefined;
}

/** 数组里第一个有限值（k_keep_line 是常量列，取一个就够）。 */
function firstFinite(a: ArrayLike<number> | undefined): number | undefined {
  if (a === undefined) return undefined;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v)) return v;
  }
  return undefined;
}
