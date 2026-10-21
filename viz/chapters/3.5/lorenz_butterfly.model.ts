/**
 * `lorenz_butterfly`（3.5 章）的 model：孪生轨道的三维投影 + 分离距离 d(t) + 两种步长的相对差。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `scene` | 轨道 2（虚线）+ 轨道 1（实线），都由**原始 x/y/z** 在显示空间投影 |
 * | 2 | scatter | `scene` | 8 个孪生对的落点（`start_x/y/z`） |
 * | 3 | scatter | `scene` | 当前时刻两个标记（帧内插值 + 同一投影） |
 * | 4 | line | `sep` | 8 条每一对各自的 d(t)（细线，对数纵轴） |
 * | 5 | line | `sep` | 平均 d(t)（粗线） |
 * | 6 | line | `sep` | h=0.01（2h）的平均 d(t)（虚线）——"步长证据"的图形部分 |
 * | 7 | line | `sep` | 实测拟合增长线（`fit_x`/`fit_y`） |
 * | 8 | line | `sep` | 公认 λ 给出的增长线（点线） |
 * | 9 | line | `sep` | 显示分辨率地板 1e-3、float64 分辨极限 4.44e-15（两条水平参考线） |
 * | 10 | line | `erreldiff` | h 与 2h 的**逐点相对差**（规则 3 的误差场，单独一条窄轴） |
 * | 11 | line | `erreldiff` | 1% 判据线 |
 *
 * 三条刻意的决定
 * --------------
 * 1. **"形状像"不是证据**。1e-8 的初值差在这张两百多像素的图上永远只有两个像素宽，
 *    所以主图不承担任何判据：判据全部落在 `sep` 副图（实测拟合线、公认增长线、地板线）
 *    与 `erreldiff` 窄条（h vs 2h 的逐点相对差）上。主图下方那句话是**常驻**的，
 *    不是"判据失败时才出现"——否则读者会以为主图本身说明了什么。
 * 2. **误差场单独一条轴**：`error_stepsize` 是**无量纲的相对差**（3e-6…11），而 `sep`
 *    轴画的是**长度** d(t)（1e-8…22）。叠在一条对数轴上，读者会把"相对差 11"读成
 *    "距离 11"——量纲不同，所以分轴画（窄条在底部、满宽）。
 * 3. **标记位置由显示空间插值给出**：帧的 `mark*_u/v` 是后端在**帧的角度**下算的；
 *    显示角度一旦与帧角度不同（拖动角度、多帧播放），拿它会把标记画到错误位置。
 *    所以标记统一走 `t_traj → 原始 x/y/z 线性插值 → 同一投影`，并在文本里写明
 *    标记时刻（帧 `meta.t_now` 与请求 t 不一致时两个都写出来）。
 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  applyOverflow,
  canvasSize,
  clipBadgeText,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finiteMinMax,
  finitePositiveMinMax,
  headerBand,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  resolveTextOverlaps,
  sampleLinear,
  sceneShell,
  segmentedSeries,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  pitfallsBand,
  scatterLayer,
  sourceBadge,
  text,
  type TextRole,
} from "../../shared/src/scene.js";
import { badgeUnitCap, columnOf, equalSpanLimits, equalSpanTicks, orthoProjection, projectInto, projectionOf, textHeight } from "./common.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.5";
const VIZ = "lorenz_butterfly";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Lorenz 孪生轨道：1e-8 的差多久看得见";

/** `manifest.gen.ts` 里 `rho.critical` 的回落值（帧 `critical.rho` 缺失时才用）。 */
/** ρ 的 Hopf 临界（manifest 里 `rho.critical` 的**最紧**一条）：帧 `critical.rho` 缺失时的回落值。 */
const RHO_CRITICAL = 24.736842105263158;
/** `log10_eps.critical` 的回落值：float64 在 |x|≈20 上的分辨极限 log10(20u)。 */
const LOG10_EPS_CRITICAL = -14.3522;
/** 拟合窗口与帧 `meta.fit_lo` / `meta.fit_hi` 同口径（帧里缺就用它并注明）。 */
const FIT_LO = 1e-6;
const FIT_HI = 1;
/** 陷阱里写的步长判据线：两种步长的相对差 1%。 */
const REL_STEP_CRIT = 1e-2;

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）：8 个槽位正好用满两列 4 行。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  sepAxis: 2,
  criticalRho: 3,
  criticalEps: 4,
  overflow: 5,
  status: 6,
  degraded: 7,
} as const;

export const lorenzButterflyModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const tTraj = requireArray(frame, "t_traj");
  const x1 = requireArray(frame, "traj1_x");
  const y1 = requireArray(frame, "traj1_y");
  const z1 = requireArray(frame, "traj1_z");
  const x2 = requireArray(frame, "traj2_x");
  const y2 = requireArray(frame, "traj2_y");
  const z2 = requireArray(frame, "traj2_z");
  const sepT = requireArray(frame, "sep_t");
  const sepMean = requireArray(frame, "sep_mean");
  const sepCoarse = requireArray(frame, "sep_coarse");
  const sepPairs = requireArray(frame, "sep_pairs");
  const errStep = requireArray(frame, "error_stepsize");
  const pitfalls = requirePitfalls(frame);
  const pairs: [ArrayLike<number>, ArrayLike<number>, string][] = [
    [tTraj, x1, "t_traj 与 traj1_x"],
    [tTraj, y1, "t_traj 与 traj1_y"],
    [tTraj, z1, "t_traj 与 traj1_z"],
    [tTraj, x2, "t_traj 与 traj2_x"],
    [tTraj, y2, "t_traj 与 traj2_y"],
    [tTraj, z2, "t_traj 与 traj2_z"],
    [sepT, sepMean, "sep_t 与 sep_mean"],
    [sepT, sepCoarse, "sep_t 与 sep_coarse"],
  ];
  for (const [a, b, what] of pairs) requireSameLength(a, b, what);
  const pairShape = frame.shapeOf("sep_pairs");
  const pairRows = pairShape[0] ?? 0;
  const pairCols = pairShape[1] ?? 0;
  if (pairShape.length !== 2 || pairRows !== sepT.length) {
    throw new ModelInputError(
      `帧 ${frame.chapter}/${frame.viz} 的 sep_pairs 形状应为 (${sepT.length}, n_pairs)，得到 [${pairShape.join(", ")}]——拒绝画错位的孪生对`,
    );
  }

  // —— 可选字段（只影响标注的缺失走降级，不抛错）——
  const degraded: string[] = [];
  const proj = projectionOf(frame, opts, degraded);
  const summary = paramSummary(frame, opts, [
    { name: "sigma", label: "$σ$" },
    { name: "rho", label: "$ρ$" },
    { name: "beta", label: "$β$" },
  ]);
  degraded.push(...summary.degraded);

  const tNowMeta = metaNumber(frame, "t_now");
  const epsMeta = metaNumber(frame, "eps");
  const log10Eps = metaNumber(frame, "log10_eps");
  const rate = metaNumber(frame, "rate");
  const rateR2 = metaNumber(frame, "rate_r2");
  const rateCoarse = metaNumber(frame, "rate_coarse");
  const rateDiff = metaNumber(frame, "rate_diff");
  const lambdaRef = metaNumber(frame, "lambda_reference");
  const sat = metaNumber(frame, "saturation");
  const tVisible = metaNumber(frame, "t_visible");
  const displayRes = metaNumber(frame, "display_res");
  const relMed = metaNumber(frame, "rel_step_median");
  const relMax = metaNumber(frame, "rel_step_max");
  const fitLoMeta = metaNumber(frame, "fit_lo");
  const fitHiMeta = metaNumber(frame, "fit_hi");
  const fitLo = fitLoMeta ?? FIT_LO;
  const fitHi = fitHiMeta ?? FIT_HI;
  if (log10Eps === undefined) degraded.push("帧 meta 缺 log10_eps：不标初值差的临界值（不猜）");
  if (rate === undefined) degraded.push("帧 meta 缺 rate：不显示实测增长率（不猜）");
  if (fitLoMeta === undefined || fitHiMeta === undefined) {
    degraded.push(`帧 meta 缺 fit_lo/fit_hi：拟合窗口按 ${FIT_LO}≤d≤${FIT_HI} 取（常数并在此注明）`);
  }

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNowMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // ------------------------------------------------------------------------- //
  // 排版：上排「三维方图 + d(t) 副轴」并排，中间一条常驻说明带，底部满宽窄条
  // ------------------------------------------------------------------------- //
  const left = 64;
  const right = 24;
  const top = headerBand(width) + 8;
  const footer = 8 + 60 + 34; // 底轴 xlabel + 渲染层现画的 pitfalls 面板（与 stackedPanels 同口径）
  const bottom = Math.max(top + 120, height - footer);
  const avail = bottom - top;
  const contentW = Math.max(120, width - left - right);
  const stripH = clampInt(Math.round(avail * 0.17), 44, 64);
  const rowTop = top + 18; // 让出 scene 轴的标题带

  // —— 常驻说明带的内容：**先量高再排版**（自己算的盒子才和自己算的排版对得上）——
  const evidence = evidenceLines({
    rate,
    rateR2,
    rateCoarse,
    rateDiff,
    sat,
    lambdaRef,
    relMed,
    relMax,
    tVisible,
    nFit: metaNumber(frame, "rate_n_fit"),
    tVisibleTheory: metaNumber(frame, "t_visible_theory"),
    displayRes,
  });
  const warnBody =
    `⚠ 「形状像」不是混沌证据：两条曲线的初值只差 ${fmt(epsMeta, 3)}，在三维图上必然逐像素重合` +
    `（任何数值噪声都能让两条线错开两个像素）；主图只回答"轨道长什么样"，判据在右侧 d(t) 副图与底部窄条。`;
  const bandBlocks: { body: string; size: number; color: string; role: TextRole }[] = [
    { body: warnBody, size: 11, color: pal.warn, role: "note" },
    ...evidence.map((body) => ({ body, size: 10, color: pal.muted, role: "note" as TextRole })),
  ];
  const bandGap = 4;
  const bandNeed = bandBlocks.reduce((acc, b) => acc + textHeight(b.body, b.size, contentW - 12) + bandGap, 6);
  const bandH = clampInt(bandNeed, 56, Math.max(56, Math.round(avail * 0.42)));
  const rowH = Math.max(96, avail - stripH - bandH - 18 - 30);
  const side = Math.max(96, Math.min(rowH, Math.round(contentW * 0.34)));
  const sepW = Math.max(160, Math.round(contentW * 0.36));
  const textX = left + side + 30 + sepW + 26;
  const textW = contentW - side - 30 - sepW - 26;
  const hasTextCol = textW >= 110;

  const sceneRect: Rect = [left, rowTop, side, side];
  const sepRect: Rect = [left + side + 30, rowTop, sepW, side];
  const gapTop = rowTop + side + 30; // 副轴的下装饰带（刻度 15 + 轴名 12）之下
  const stripRect: Rect = [left, bottom - stripH, contentW, stripH];

  // —— 显示空间投影（原始 x/y/z；见 common.ts 文件头）——
  const p1 = projectInto(proj, x1, y1, z1);
  const p2 = projectInto(proj, x2, y2, z2);
  const startX = frame.has("start_x") ? requireArray(frame, "start_x") : undefined;
  const startY = frame.has("start_y") ? requireArray(frame, "start_y") : undefined;
  const startZ = frame.has("start_z") ? requireArray(frame, "start_z") : undefined;
  const startPts =
    startX !== undefined && startY !== undefined && startZ !== undefined
      ? projectInto(proj, startX, startY, startZ)
      : undefined;

  const uAll = concat(concat(p1.u, p2.u), startPts?.u ?? new Float64Array(0));
  const vAll = concat(concat(p1.v, p2.v), startPts?.v ?? new Float64Array(0));
  const span = equalSpanLimits(uAll, vAll, 5);

  scene.axes.push(
    makeAxis("scene", sceneRect, {
      xlim: span.xlim,
      ylim: span.ylim,
      xticks: equalSpanTicks(span.xlim, 4),
      yticks: equalSpanTicks(span.ylim, 4),
      xlabel: "",
      // 纵轴名保持极短：光栅层把 CJK 竖排成一列，太长会挤在轴边
      ylabel: "$v$",
      title: "三维相空间（正交投影，等比例）",
      autoscaled: false,
    }),
  );

  // —— 图层 1：两条孪生轨道（相差 1e-8，图上必然重合）——
  const s2 = segmentedSeries(p2.u, p2.v, pal.warn, { width: 1.1, dash: [5, 4] });
  const s1 = segmentedSeries(p1.u, p1.v, pal.curves[0], { width: 1.6 });
  scene.layers.push(lineLayer("scene", [...s2, ...s1]));

  // —— 图层 2：8 个孪生对的落点 ——
  if (startPts !== undefined && startPts.u.length > 0 && startPts.u.length === startPts.v.length) {
    scene.layers.push(scatterLayer("scene", startPts.u, startPts.v, { r: 2.6, color: pal.muted, alpha: 0.85 }));
  } else {
    degraded.push("帧缺 start_x/start_y/start_z：三维图不画孪生对落点（不猜）");
  }

  // —— 图层 3：当前时刻标记（帧内插值 + 同一投影，见文件头第 3 条）——
  const tLo = tTraj[0] as number;
  const tHi = tTraj[tTraj.length - 1] as number;
  const tWant = Number.isFinite(opts.t) ? opts.t : (tNowMeta ?? tLo);
  const tMark = clampTo(tWant, tLo, tHi);
  const mark = markerAt(tMark, tTraj, x1, y1, z1, x2, y2, z2, proj);
  if (mark !== undefined) {
    scene.layers.push(
      scatterLayer("scene", [mark.u1, mark.u2], [mark.v1, mark.v2], {
        r: 3.6,
        color: pal.marker,
      }),
    );
  } else {
    degraded.push(`t=${formatNumber(tMark, 4)} 不在帧 t_traj 范围内：不画当前时刻标记（不外推）`);
  }
  if (Math.abs(tMark - tWant) > 1e-9) {
    degraded.push(
      `请求 t=${formatNumber(tWant, 4)} 超出帧范围 [${formatNumber(tLo, 3)}, ${formatNumber(tHi, 3)}]：夹到 ${formatNumber(tMark, 3)} 并按真实时刻标注`,
    );
  }
  if (tNowMeta !== undefined && Math.abs(tNowMeta - tMark) > 0.0051) {
    degraded.push(
      `帧 meta.t_now=${formatNumber(tNowMeta, 4)} 与请求时刻 ${formatNumber(tMark, 4)} 相差超过半步长：标记按帧内插值画，两个时刻都写在文本里`,
    );
  }

  // ------------------------------------------------------------------------- //
  // 副轴 `sep`：d(t)（对数）
  // ------------------------------------------------------------------------- //
  const fitX = frame.has("fit_x") ? requireArray(frame, "fit_x") : undefined;
  const fitY = frame.has("fit_y") ? requireArray(frame, "fit_y") : undefined;
  const growthT = frame.has("growth_line_t") ? requireArray(frame, "growth_line_t") : undefined;
  const growthD = frame.has("growth_line_d") ? requireArray(frame, "growth_line_d") : undefined;
  const floorT = frame.has("floor_line_t") ? requireArray(frame, "floor_line_t") : undefined;
  const floorD = frame.has("floor_line_d") ? requireArray(frame, "floor_line_d") : undefined;
  const floatD = frame.has("float_line_d") ? requireArray(frame, "float_line_d") : undefined;

  const sepPositive = finitePositiveMinMax(
    concat(
      concat(sepMean, sepCoarse),
      concat(
        concat(fitY ?? new Float64Array(0), growthD ?? new Float64Array(0)),
        concat(floorD ?? new Float64Array(0), floatD ?? new Float64Array(0)),
      ),
    ),
  );
  const sepYlim: [number, number] =
    sepPositive === undefined ? [1e-16, 1] : niceLimits(sepPositive[0], sepPositive[1], 6, { log: true, pad: 0.04 });
  const tSpan = finiteMinMax(sepT) ?? [0, 1];
  scene.axes.push(
    makeAxis("sep", sepRect, {
      xlim: tSpan,
      ylim: sepYlim,
      yscale: "log",
      xlabel: "$t$",
      ylabel: "$d(t)$",
      title: "孪生轨道分离距离 d(t)",
      autoscaled: true,
    }),
  );

  for (let c = 0; c < pairCols; c += 1) {
    const col = columnOf(frame, "sep_pairs", c, pairRows);
    if (col === undefined) break;
    scene.layers.push(
      lineLayer(
        "sep",
        segmentedSeries(sepT, col, pal.reference, {
          width: 0.8,
          alpha: 0.75,
          accept: (v) => v > 0,
        }),
      ),
    );
  }
  scene.layers.push(
    lineLayer("sep", segmentedSeries(sepT, sepMean, pal.curves[0], { width: 2.2, accept: (v) => v > 0 })),
  );
  scene.layers.push(
    lineLayer("sep", segmentedSeries(sepT, sepCoarse, pal.exact, { width: 1.3, dash: [6, 4], accept: (v) => v > 0 })),
  );
  if (fitX !== undefined && fitY !== undefined) {
    scene.layers.push(
      lineLayer("sep", segmentedSeries(fitX, fitY, pal.marker, { width: 1.4, dash: [3, 3], accept: (v) => v > 0 })),
    );
  } else {
    degraded.push("帧缺 fit_x/fit_y：不画实测拟合增长线（不猜）");
  }
  if (growthT !== undefined && growthD !== undefined) {
    scene.layers.push(
      lineLayer(
        "sep",
        segmentedSeries(growthT, growthD, pal.note, {
          width: 1.2,
          dash: [2, 3],
          accept: (v) => v > 0,
        }),
      ),
    );
  }
  if (floorT !== undefined && floorD !== undefined) {
    scene.layers.push(
      lineLayer(
        "sep",
        segmentedSeries(floorT, floorD, pal.muted, {
          width: 1,
          dash: [1, 3],
          accept: (v) => v > 0,
        }),
      ),
    );
  } else {
    degraded.push("帧缺 floor_line_t/floor_line_d：不画显示分辨率地板线（不猜）");
  }
  if (floatD !== undefined && floorT !== undefined && floorT.length >= 2) {
    scene.layers.push(
      lineLayer(
        "sep",
        [
          {
            x: [floorT[0] as number, floorT[floorT.length - 1] as number],
            y: [floatD[0] as number, floatD[0] as number],
            color: pal.residual,
            width: 1,
            dash: [2, 2],
          },
        ],
      ),
    );
  }

  // ------------------------------------------------------------------------- //
  // 窄条 `erreldiff`：h 与 2h 的逐点相对差（无量纲；见文件头第 2 条）
  // ------------------------------------------------------------------------- //
  const relWin = windowTimes(sepT, sepMean, errStep.length, fitLo, fitHi);
  const relPos = finitePositiveMinMax(concat(errStep, Float64Array.from([REL_STEP_CRIT])));
  const relYlim: [number, number] = relPos === undefined ? [1e-6, 1] : niceLimits(relPos[0], relPos[1], 5, { log: true, pad: 0.05 });
  scene.axes.push(
    makeAxis("erreldiff", stripRect, {
      xlim: tSpan,
      ylim: relYlim,
      yscale: "log",
      xlabel: "",
      ylabel: "相对差",
      title: "",
      autoscaled: true,
    }),
  );
  if (relWin !== undefined) {
    scene.layers.push(
      lineLayer(
        "erreldiff",
        segmentedSeries(relWin, errStep, pal.residual, {
          width: 1.1,
          accept: (v) => v > 0,
        }),
      ),
    );
    scene.layers.push(
      lineLayer("erreldiff", [
        { x: [tSpan[0], tSpan[1]], y: [REL_STEP_CRIT, REL_STEP_CRIT], color: pal.reference, width: 1, dash: [2, 3] },
      ]),
    );
    scene.texts.push(
      text(
        stripRect[0] + 6,
        stripRect[1] + 4,
        `h=0.005 vs 2h=0.01 的逐点相对差（无量纲，别当距离读）：中位 ${fmt(relMed, 3)}、最大 ${fmt(relMax, 3)}；虚线 = 1% 判据线`,
        { role: "note", size: 10, color: pal.muted, maxWidth: Math.max(140, stripRect[2] * 0.55) },
      ),
    );
  } else {
    degraded.push(
      `error_stepsize 有 ${errStep.length} 点，但帧里凑不出对应时刻（sep_mean 落在 ${FIT_LO}≤d≤${FIT_HI} 的点数对不上）：窄条不画相对差曲线`,
    );
  }

  // ------------------------------------------------------------------------- //
  // 文本
  // ------------------------------------------------------------------------- //
  // 右侧说明列：逐块量高、自上而下码放；放不下的当场记账（不让 resolveTextOverlaps 悄悄丢）
  const colBottom = rowTop + side;
  const colBlocks: { body: string; size: number; color: string; role: TextRole; gap: number }[] = [
    {
      body: "蓝=轨道 1　橙虚=轨道 2（x 上 +ε）　灰点=孪生对落点",
      size: 10,
      color: pal.muted,
      role: "note",
      gap: 4,
    },
    {
      body: `${summary.text}　初值差 ε=${fmt(epsMeta, 3)}（沿 x）　孪生对 ${pairCols}`,
      size: 11,
      color: pal.ink,
      role: "label",
      gap: 2,
    },
    {
      body: `当前时刻 t=${fmt(tMark, 4)}（帧 t_now=${fmt(tNowMeta, 4)}）　平均分离 d=${fmt(sampleLinear(sepT, sepMean, tMark), 3)}`,
      size: 11,
      color: pal.muted,
      role: "label",
      gap: 6,
    },
    {
      body: `正交投影（显示空间，无透视除法）：方位角 ${proj.azimuthDeg}°、仰角 ${proj.elevationDeg}°（${
        proj.source === "meta" ? "帧 meta" : proj.source === "params" ? "请求值" : "缺省值"
      }）；点的疏密只反映轨道停留的久暂，深度不可读。`,
      size: 10,
      color: pal.muted,
      role: "note",
      gap: 4,
    },
  ];
  if (hasTextCol) {
    let cursor = rowTop + 2;
    let dropped = 0;
    for (const b of colBlocks) {
      const h = textHeight(b.body, b.size, textW);
      if (cursor + h > colBottom) {
        dropped += 1;
        continue;
      }
      scene.texts.push(text(textX, cursor, b.body, { role: b.role, size: b.size, color: b.color, maxWidth: textW }));
      cursor += h + b.gap;
    }
    if (dropped > 0) degraded.push(`右侧说明列放不下 ${dropped} 段文字（列宽 ${Math.round(textW)}px、列高 ${Math.round(side)}px）：未显示的段落不再重复`);
  } else {
    degraded.push("画布太窄：主图右侧说明列放不下，参数与判据只保留角标与下方常驻声明");
  }

  // 中间说明带：常驻的反误读声明 + 三条判据（自下而上量高摆放，绝不压住上下装饰带）
  {
    let cursor = gapTop;
    for (const b of bandBlocks) {
      const h = textHeight(b.body, b.size, contentW - 12);
      if (cursor + h > stripRect[1] - 6) {
        degraded.push("说明带放不下：剩下的判据行未显示（角标仍给出成立与否）");
        break;
      }
      scene.texts.push(text(left, cursor, b.body, { role: b.role, size: b.size, color: b.color, maxWidth: contentW - 12 }));
      cursor += h + bandGap;
    }
  }

  // 副图内的参考线说明（sep 面板左上角是空的：那里 d 远大于 1）
  scene.texts.push(
    text(
      sepRect[0] + 6,
      sepRect[1] + 4,
      `灰虚线 = 显示分辨率地板 ${fmt(displayRes, 2)}（"看得见"的判据）；紫点线 = float64 在 |x|≈20 上的分辨极限 4.4e-15`,
      { role: "note", size: 10, color: pal.muted, maxWidth: Math.max(140, sepRect[2] * 0.92) },
    ),
  );

  // ------------------------------------------------------------------------- //
  // 角标（规则 4/5/6）
  // ------------------------------------------------------------------------- //
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "d(t) 轴与相对差轴：自适应（按数据取整）", 8, 8), SLOT.sepAxis, width),
  );
  const rho = metaNumber(frame, "rho");
  if (rho !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("ρ", rho, criticalOf(frame, "rho", RHO_CRITICAL), { beyond: "above" }), SLOT.criticalRho, width));
  } else {
    degraded.push("帧 meta 缺 rho：不标 ρ 的临界值（不猜）");
  }
  if (log10Eps !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("log10 ε", log10Eps, criticalOf(frame, "log10_eps", LOG10_EPS_CRITICAL), { beyond: "below" }), SLOT.criticalEps, width),
    );
  }
  scene.badges.push(
    placeBadge(
      badge("status", clipBadgeText(`判据：${criterionSummary(frame.meta["rate_ok"], rateR2, rateDiff, relMed, sat)}`,
          badgeUnitCap(width),), 8, 8),
      SLOT.status,
      width,
    ),
  );
  const status = degradedBadge(degraded, SLOT.degraded, width);
  if (status !== undefined) scene.badges.push(status);

  requireDrawableLayers(scene);
  applyOverflow(
    scene,
    countNonFinite(x1) + countNonFinite(y1) + countNonFinite(z1) + countNonFinite(sepMean) + countNonFinite(sepPairs) + countNonFinite(errStep),
    { slot: SLOT.overflow, width },
  );

  // 文本层之间不许互相压住、也不许被底部 pitfalls 面板压住（规则 7/8，两侧校验器都查）：
  // 第四个参数是**高度**（`textBoxOf` 口径），所以传「底线 − 顶」。
  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（纯显示几何）
// --------------------------------------------------------------------------- //

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(v)));
}

function clampTo(v: number, lo: number, hi: number): number {
  if (!(hi > lo)) return lo;
  return v < lo ? lo : v > hi ? hi : v;
}

function concat(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const out = new Float64Array(a.length + b.length);
  for (let i = 0; i < a.length; i += 1) out[i] = a[i] as number;
  for (let i = 0; i < b.length; i += 1) out[a.length + i] = b[i] as number;
  return out;
}

function fmt(v: number | undefined, sig: number): string {
  return v === undefined ? "?" : formatNumber(v, sig);
}

/** 标记点：在原始 x/y/z 上线性插值，再用**同一个**投影算子投影（见文件头第 3 条）。 */
function markerAt(
  tq: number,
  t: ArrayLike<number>,
  x1: ArrayLike<number>,
  y1: ArrayLike<number>,
  z1: ArrayLike<number>,
  x2: ArrayLike<number>,
  y2: ArrayLike<number>,
  z2: ArrayLike<number>,
  proj: ReturnType<typeof orthoProjection>,
): { u1: number; v1: number; u2: number; v2: number } | undefined {
  const a = [sampleLinear(t, x1, tq), sampleLinear(t, y1, tq), sampleLinear(t, z1, tq)];
  const b = [sampleLinear(t, x2, tq), sampleLinear(t, y2, tq), sampleLinear(t, z2, tq)];
  for (const v of [...a, ...b]) if (v === undefined) return undefined;
  const [ax, ay, az] = a as [number, number, number];
  const [bx, by, bz] = b as [number, number, number];
  return { u1: proj.u(ax, ay), v1: proj.v(ax, ay, az), u2: proj.u(bx, by), v2: proj.v(bx, by, bz) };
}

/**
 * 拟合窗口内的时间格点。
 *
 * 帧**没有**给 `error_stepsize` 的逐点时间（它 288 个点，而 `sep_t` 有 602 个），
 * 但它的口径写在帧的 `strings.display_policy` 里：`|d_h − d_2h| / d_2h` 取在
 * `1e-6 ≤ d ≤ 1` 的窗口上、按 `sep_t` 的顺序排列。所以这里按同一判据在
 * `sep_mean` 上取掩码（**从帧数组里选点**，不是重算物理量），点数与
 * `error_stepsize` 一致时才画；对不上就返回 `undefined` 并降级说明（不假装精确）。
 */
function windowTimes(
  sepT: ArrayLike<number>,
  sepMean: ArrayLike<number>,
  want: number,
  lo: number,
  hi: number,
): Float64Array | undefined {
  if (want <= 0 || sepT.length === 0) return undefined;
  const picked: number[] = [];
  for (let i = 0; i < sepMean.length && i < sepT.length; i += 1) {
    const d = sepMean[i] as number;
    if (Number.isFinite(d) && d >= lo && d <= hi) picked.push(sepT[i] as number);
  }
  return picked.length === want ? Float64Array.from(picked) : undefined;
}

/** 三条判据的读数（数字写清楚；缺的写 "?"，不猜）。 */
function evidenceLines(m: {
  rate: number | undefined;
  rateR2: number | undefined;
  rateCoarse: number | undefined;
  rateDiff: number | undefined;
  sat: number | undefined;
  lambdaRef: number | undefined;
  relMed: number | undefined;
  relMax: number | undefined;
  tVisible: number | undefined;
  tVisibleTheory: number | undefined;
  displayRes: number | undefined;
  nFit: number | undefined;
}): string[] {
  const dr = m.rate !== undefined && m.lambdaRef !== undefined ? m.rate - m.lambdaRef : undefined;
  return [
    `判据① 指数增长：窗口 1e-6≤d≤1、${fmt(m.nFit, 6)} 点，R²=${fmt(m.rateR2, 4)}，实测增长率 ${fmt(m.rate, 5)}（公认 ${fmt(m.lambdaRef, 5)}，差 ${fmt(dr, 3)}）`,
    `判据② 步长对照：h=0.005 vs 2h=0.01 的增长率 ${fmt(m.rate, 5)} vs ${fmt(m.rateCoarse, 5)}（差 ${fmt(m.rateDiff, 3)}）；逐点相对差中位 ${fmt(m.relMed, 3)}、最大 ${fmt(m.relMax, 3)}`,
    `判据③ 饱和：d 饱和在 ${fmt(m.sat, 4)}，与吸引子直径同量级（都是 O(20)）；不饱和就说明看到的只是积分器在舍入`,
    `可见时间：以显示分辨率 ${fmt(m.displayRes, 2)} 为准，实测 t≈${fmt(m.tVisible, 4)}，解析估计 ${fmt(m.tVisibleTheory, 4)}`,
  ];
}

/** 一行角标能放下的判据摘要（完整读数在文本层：角标只是索引）。 */
function criterionSummary(
  rateOk: boolean | number | null | undefined,
  rateR2: number | undefined,
  rateDiff: number | undefined,
  relMed: number | undefined,
  sat: number | undefined,
): string {
  const ok = rateOk === true || rateOk === 1;
  const head = ok ? "三条判据成立（读数见文本）" : "判据不成立：这张图只能说明积分器在舍入";
  return `${head}　R²=${fmt(rateR2, 3)}　Δ增长率=${fmt(rateDiff, 3)}　相对差中位=${fmt(relMed, 3)}　饱和=${fmt(sat, 3)}`;
}
