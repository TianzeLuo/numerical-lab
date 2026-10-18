/**
 * `precond_effect`（1.5 章）的 model：预条件前后的 Ritz 谱散点 + 收敛曲线与 CG 最坏界。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | band | `spectrum` | A 的谱区间 [λmin_pre, λmax_pre]（帧 meta 的两端） |
 * | 2 | line | `spectrum` | 预条件后谱的两端参考线（twin 的两条水平虚线） |
 * | 3 | scatter | `spectrum` | `ritz_pre`（A 的 Ritz 值，按序号） |
 * | 4 | scatter | `spectrum` | `ritz_post`（M⁻¹A 的 Ritz 值，按序号） |
 * | 5 | line | `convergence` | `residual`（requiredAux）、`error`、`bound` 三条曲线 |
 * | 6 | line | `convergence` | 实测收敛因子^k 参考线（斜率来自帧 meta） |
 * | 7 | scatter | `convergence` | 末步残差 `residual_mark` |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 / 3 ω 临界 /
 * 4 状态 / 5 溢出 / 6 降级。
 *
 * 几条刻意的决定
 * --------------
 * 1. **谱散点的纵轴是固定轴**（manifest 声明 `spectrum.autoscaled = false`）：
 *    范围取 [0, 8.2]——二维 5 点 Laplacian 的谱恒在 (0, 8) 内（λmax = 8cos²(π/(2(m+1)))·... < 8），
 *    **与 grid 无关**。固定它才能让"谱被压到 [λmin, λmax] 这一段"在同一个尺子下比较；
 *    若哪一帧真的越过 8.2，`checkOverflow` 会照实报溢出，而不是悄悄把轴拉长。
 *    横轴是 Ritz **序号**（索引轴，按帧给的个数铺满），不是物理量，按序号铺开不构成"自适应"。
 * 2. **`ilu_dropped = -1` 是哨兵**：只有 ILU(0) 才有"丢弃填充次数"，其它预条件子帧里写 -1。
 *    这个 -1 **不能当 0 画**（"丢弃 0 次"是完全相反的结论），所以文本里逐字说明；
 *    `docs/PROTOCOLS.md` §1.3 要求后端在 `strings` 里解释这个哨兵，当前 1.5 章**没有写**
 *    （见交付报告），前端在角标与面板文本里补上这层说明。
 * 3. **`meta.omega` 在非 SSOR 帧里是 0**：后端只在 `precond == "ssor"` 时写真实 ω，
 *    否则写 0——这同样是一个没有 `strings` 说明的哨兵。模型只在 `strings.precond == "ssor"`
 *    时显示 ω，否则明说"ω 不适用（帧 meta.omega = 0 是哨兵）"，绝不显示成"ω = 0.000"。
 * 4. **`kappa_effective` / `factor_measured` / `error_final` 可能是 null**：那是"没有有限值"
 *    的如实标注（残差已到 0 或点数不足），模型跳过该量并在文本里说明，**不按 0 处理**。
 * 5. **"κ 降幅 != 迭代数降幅"必须能从图上读出来**：谱的压缩（左图两条散点的横向差）与
 *    实际收敛速度（右图）是两件事——最坏界曲线 `bound` 与实测残差之间的那道缝就是
 *    "界比实测悲观"的证据（pp. 本章 pitfalls 第 10、11 条）。`strings.note` 里的
 *    "降 N 倍 / 迭代 M 步"原样进面板文本，不做二次解读。
 * 6. **两条参考线的斜率来自帧**：`bound` 是后端算好的数组（用实测 κ_post），
 *    实测因子^k 的斜率取 `meta.factor_measured`；前端只画线、并逐点裁到轴内
 *    （折线不被渲染器裁剪，越界值会触发溢出角标）。
 */

import { formatNumber, niceLimits, niceTicks } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  applyOverflow,
  canvasSize,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finitePositiveMinMax,
  metaNumber,
  optionalArray,
  palette,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
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
  bandLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
  text,
  type Badge,
  type TextSpec,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.5";
const VIZ = "precond_effect";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "预条件前后：谱分布与收敛曲线联动";
/** manifest 里 `omega.critical = [1.0]`（SSOR 在 ω=1 退化为 SSGS）；`frame.critical.omega` 缺失时的回落值。 */
const OMEGA_DEGENERATE = 1.0;
/** 二维 5 点 Laplacian 的谱上界：λmax < 8（与 grid 无关），固定轴的依据（见文件头第 1 条）。 */
const LAPLACIAN_2D_MAX = 8.0;
const SPECTRUM_YLIM: [number, number] = [0, LAPLACIAN_2D_MAX + 0.2];

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。本帧数据源恒为 backend，故不占 preview 槽。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  criticalOmega: 3,
  status: 4,
  overflow: 5,
  degraded: 6,
} as const;

export const precondEffectModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const ritzPre = requireArray(frame, "ritz_pre");
  const ritzPost = requireArray(frame, "ritz_post");
  const ritzIndexPre = requireArray(frame, "ritz_index_pre");
  const ritzIndexPost = requireArray(frame, "ritz_index_post");
  const convergence = requireArray(frame, "convergence");
  const residual = requireArray(frame, "residual");
  const error = requireArray(frame, "error");
  const bound = requireArray(frame, "bound");
  const stepGrid = requireArray(frame, "step_grid");
  for (const [name, a, b] of [
    ["ritz_pre / ritz_index_pre", ritzPre, ritzIndexPre],
    ["ritz_post / ritz_index_post", ritzPost, ritzIndexPost],
    ["convergence / residual", convergence, residual],
    ["error / residual", error, residual],
    ["bound / residual", bound, residual],
    ["step_grid / residual", stepGrid, residual],
  ] as const) {
    if (a.length !== b.length) {
      throw new ModelInputError(
        `帧 ${CHAPTER}/${VIZ} 的 ${name} 长度不一致（${a.length} vs ${b.length}）：逐点配对的曲线拒绝错位绘制`,
      );
    }
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（本可视化 time.mode = none，t 不参与取帧）");
  const tNow = tMeta ?? opts.t;
  const precondKey = stringOf(frame, "precond");
  const precondName = stringOf(frame, "precond_name") ?? precondKey;
  if (precondName === undefined) degraded.push("帧 strings 缺 precond/precond_name：角标只写「预条件子」而不猜是哪一个");
  const solver = stringOf(frame, "solver");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const kappaPre = metaNumber(frame, "kappa_pre");
  const kappaPost = metaNumber(frame, "kappa_post");
  const kappaRatio = metaNumber(frame, "kappa_ratio");
  const kappaEffective = metaNumber(frame, "kappa_effective");
  if (frame.meta["kappa_effective"] === null) {
    degraded.push("帧 meta.kappa_effective 是 null（没有有限值）：等效 κ 不显示，也不按 0 处理");
  }
  const factorMeasured = metaNumber(frame, "factor_measured");
  if (frame.meta["factor_measured"] === null) {
    degraded.push("帧 meta.factor_measured 是 null（没有有限值）：实测因子参考线不画");
  }
  const nIter = metaNumber(frame, "n_iter");
  const n = metaNumber(frame, "n");
  const grid = metaNumber(frame, "grid");
  const converged = (metaNumber(frame, "converged") ?? 0) > 0.5;
  const lambdaMinPre = metaNumber(frame, "lambda_min_pre");
  const lambdaMaxPre = metaNumber(frame, "lambda_max_pre");
  const lambdaMinPost = metaNumber(frame, "lambda_min_post");
  const lambdaMaxPost = metaNumber(frame, "lambda_max_post");
  const nRitzPre = metaNumber(frame, "n_ritz_pre");
  const nRitzPost = metaNumber(frame, "n_ritz_post");
  const spectrumSteps = metaNumber(frame, "spectrum_steps");
  const spectrumOrth = metaNumber(frame, "spectrum_orthogonality");
  const iluDroppedRaw = metaNumber(frame, "ilu_dropped");
  // 帧自己给了判定：`meta.ilu_dropped_applicable`（字符串解释见 strings.sentinel_note）。
  // 只有在帧没给这个键时才回落到 "-1 = 哨兵"，不自己发明判据。
  const iluApplicable = metaBool(frame, "ilu_dropped_applicable");
  const iluDropped = iluDroppedRaw !== undefined && iluDroppedRaw >= 0 ? iluDroppedRaw : undefined;
  const iluSentinel = iluApplicable === false || (iluApplicable === undefined && iluDroppedRaw !== undefined && iluDroppedRaw < 0);
  const sentinelNote = stringOf(frame, "sentinel_note");
  const precondNnz = metaNumber(frame, "precond_nnz");
  const matrixNnz = metaNumber(frame, "matrix_nnz");
  const omegaRaw = metaNumber(frame, "omega");
  const isSsor = precondKey === "ssor";
  const omega = isSsor ? omegaRaw : undefined;
  if (!isSsor && omegaRaw !== undefined) {
    degraded.push(
      `帧 meta.omega=${formatNumber(omegaRaw, 3)} 是哨兵（只有 SSOR 用 ω，strings 里没有解释）：ω 一栏显示"不适用"`,
    );
  }
  const residualMark = optionalArray(frame, "residual_mark");
  const paramsJson = paramsJsonOf(frame);

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左列谱散点（+ 文本），右列收敛曲线（+ 文本）——
  const base = stackedPanels(width, height);
  const contentH = base.bottom - base.top;
  const specW = Math.max(200, Math.round(base.contentWidth * 0.44));
  const specH = Math.max(140, Math.round(contentH * 0.53)); // 余下的高度要留给"这些数字怎么读"

  const specRect: Rect = [base.left, base.top, specW, specH];
  const convX = base.left + specW + 60;
  const convW = Math.max(180, base.left + base.contentWidth - convX);
  const convH = Math.max(140, Math.round(contentH * 0.53));
  const convRect: Rect = [convX, base.top, convW, convH];

  // 谱散点：纵轴固定（见文件头第 1 条）；横轴是 Ritz 序号。
  const nIdx = Math.max(ritzIndexPre.length, ritzIndexPost.length, 2);
  const specAxis = makeAxis("spectrum", specRect, {
    xlim: [-1, nIdx],
    ylim: SPECTRUM_YLIM,
    xlabel: "Ritz 序号（帧 ritz_index_pre / ritz_index_post）",
    ylabel: axisYlabel("spectrum", "λ(M⁻¹A)"),
    title: axisTitle("spectrum", "预条件前后的谱散点（Ritz 值，按序号排列）"),
    autoscaled: false,
    xticks: niceTicks(0, nIdx - 1, Math.max(3, Math.round(specRect[2] / 90))),
    yticks: niceTicks(SPECTRUM_YLIM[0], SPECTRUM_YLIM[1], Math.max(3, Math.round(specRect[3] / 55))),
  });

  // 收敛曲线（manifest 声明 `convergence.autoscaled = true`、`scale = "log"`）。
  const resPos = finitePositiveMinMax(residual);
  const errPos = finitePositiveMinMax(error);
  const boundPos = finitePositiveMinMax(bound);
  const lo = Math.min(resPos?.[0] ?? Number.POSITIVE_INFINITY, errPos?.[0] ?? Number.POSITIVE_INFINITY, boundPos?.[0] ?? Number.POSITIVE_INFINITY);
  const hi = Math.max(resPos?.[1] ?? Number.NEGATIVE_INFINITY, errPos?.[1] ?? Number.NEGATIVE_INFINITY, boundPos?.[1] ?? Number.NEGATIVE_INFINITY);
  const hasPositive = Number.isFinite(lo) && Number.isFinite(hi);
  const convYlim: [number, number] = hasPositive ? niceLimits(lo, hi, 7, { log: true, pad: 0.2 }) : [1e-16, 1];
  if (!hasPositive) degraded.push("residual/error/bound 里没有 >0 的有限值：副轴用占位范围 [1e-16, 1]，且不画曲线");
  const stepMax = Math.max(1, residual.length - 1);
  const convAxis = makeAxis("convergence", convRect, {
    xlim: [0, stepMax],
    ylim: convYlim,
    yscale: "log",
    xlabel: "迭代步 k",
    ylabel: axisYlabel("convergence", "相对残差"),
    title: axisTitle("convergence", "收敛曲线与 CG 最坏界（对数）"),
    autoscaled: true,
    xticks: niceTicks(0, stepMax, Math.max(3, Math.round(convRect[2] / 90))),
  });
  scene.axes.push(specAxis, convAxis);

  // —— 图层 1~2：谱区间带 + 预条件后谱的两端参考线 ——
  const sameSpectrum = sameValues(ritzPre, ritzPost);
  if (lambdaMinPre !== undefined && lambdaMaxPre !== undefined && lambdaMaxPre > lambdaMinPre) {
    scene.layers.push(
      bandLayer("spectrum", {
        x: [-1, nIdx],
        ylo: [lambdaMinPre, lambdaMinPre],
        yhi: [lambdaMaxPre, lambdaMaxPre],
        color: pal.curves[0],
        alpha: 0.14,
        label: `A 的谱区间 [${formatNumber(lambdaMinPre, 4)}, ${formatNumber(lambdaMaxPre, 4)}]（帧 meta）`,
      }),
    );
  } else {
    degraded.push("帧 meta 缺 lambda_min_pre/lambda_max_pre：不画 A 的谱区间带（不猜区间）");
  }
  const postRefs: ReturnType<typeof series>[] = [];
  if (lambdaMinPost !== undefined) {
    // 参考线横跨整个面板：末点标签会溢出画布，值写在面板文本里。
    postRefs.push(series([-1, nIdx], [lambdaMinPost, lambdaMinPost], pal.exact, { dash: [5, 4], width: 1 }));
  }
  if (lambdaMaxPost !== undefined) {
    postRefs.push(
      series([-1, nIdx], [lambdaMaxPost, lambdaMaxPost], pal.exact, {
        dash: [5, 4],
        width: 1,
        label: undefined,
      }),
    );
  }
  if (postRefs.length > 0) scene.layers.push(lineLayer("spectrum", postRefs));

  // —— 图层 3~4：两组 Ritz 值散点 ——
  scene.layers.push(
    scatterLayer("spectrum", ritzIndexPre, ritzPre, { r: 2.6, color: pal.curves[0], label: "A 的 Ritz 值（帧 ritz_pre）" }),
  );
  scene.layers.push(
    scatterLayer("spectrum", ritzIndexPost, ritzPost, {
      r: 3.2,
      color: pal.marker,
      alpha: sameSpectrum ? 0.85 : 1,
      label: sameSpectrum ? "M⁻¹A 的 Ritz 值（与上一组逐点重合：本帧 M = A）" : "M⁻¹A 的 Ritz 值（帧 ritz_post）",
    }),
  );

  // —— 图层 5~7：收敛曲线 + 最坏界 + 实测因子参考线 ——
  const convSeries: ReturnType<typeof series>[] = [];
  convSeries.push(
    ...segmentedSeries(stepGrid, residual, pal.residual, {
      width: 2.2,
      label: endLabel(stepGrid[stepGrid.length - 1] ?? Number.NaN, convAxis.xlim, convRect[2], "相对残差（帧 residual）"),
      accept: (v) => v > 0,
    }),
  );
  convSeries.push(
    ...segmentedSeries(stepGrid, error, pal.curves[1], {
      width: 1.4,
      dash: [6, 3],
      label: endLabel(stepGrid[stepGrid.length - 1] ?? Number.NaN, convAxis.xlim, convRect[2], "相对误差（帧 error）"),
      accept: (v) => v > 0,
    }),
  );
  convSeries.push(
    ...segmentedSeries(stepGrid, bound, pal.reference, {
      width: 1.4,
      dash: [4, 4],
      label: endLabel(stepGrid[stepGrid.length - 1] ?? Number.NaN, convAxis.xlim, convRect[2], "CG 最坏界（帧 bound）"),
      accept: (v) => v > 0,
    }),
  );
  let factorLineDrawn = false;
  if (factorMeasured !== undefined && factorMeasured > 0) {
    const line = decayLine(factorMeasured, stepMax, convYlim[0]);
    if (line.x.length >= 2) {
      convSeries.push(
        series(line.x, line.y, pal.note, {
          width: 1.2,
          dash: [2, 3],
          label: endLabel(line.x[line.x.length - 1] ?? Number.NaN, convAxis.xlim, convRect[2], `实测因子^k（${formatNumber(factorMeasured, 4)}）`),
        }),
      );
      factorLineDrawn = true;
    }
  }
  scene.layers.push(lineLayer("convergence", convSeries));
  const rLast = residualMark !== undefined && residualMark.length > 0 ? (residualMark[0] as number) : (residual[residual.length - 1] as number);
  if (Number.isFinite(rLast) && rLast > 0) {
    scene.layers.push(
      scatterLayer("convergence", [stepMax], [rLast], { r: 3.6, color: pal.marker, label: "末步残差（帧 residual_mark）" }),
    );
  }

  // —— 文本 ——
  const specLines: string[] = [];
  specLines.push(
    `${formatNumber(grid ?? 0, 3)}×${formatNumber(grid ?? 0, 3)} Laplacian：n=${formatNumber(n ?? 0, 3)}，` +
      `非零元 ${formatNumber(matrixNnz ?? 0, 3)} / 预条件子 ${formatNumber(precondNnz ?? 0, 3)}`,
  );
  specLines.push(
    (lambdaMinPre === undefined || lambdaMaxPre === undefined
      ? "A 的谱两端：帧没给"
      : `A 的谱 [${formatNumber(lambdaMinPre, 4)}, ${formatNumber(lambdaMaxPre, 4)}]`) +
      (lambdaMinPost === undefined || lambdaMaxPost === undefined
        ? "；预条件后：帧没给"
        : `；预条件后 [${formatNumber(lambdaMinPost, 4)}, ${formatNumber(lambdaMaxPost, 4)}]`),
  );
  specLines.push(
    `散点前 ${formatNumber(ritzIndexPre.length, 4)} / 后 ${formatNumber(ritzIndexPost.length, 4)} 个` +
      (nRitzPre === undefined || nRitzPost === undefined
        ? ""
        : `（meta ${formatNumber(nRitzPre, 3)}/${formatNumber(nRitzPost, 3)}）`) +
      (spectrumSteps === undefined ? "" : `；Lanczos ${formatNumber(spectrumSteps, 4)} 步`) +
      (spectrumOrth === undefined ? "" : `，正交性 ${formatNumber(spectrumOrth, 3)}`),
  );
  specLines.push("Ritz 值只有两端收敛快：散点 !=「整个谱都量到了」");
  specLines.push(
    sameSpectrum
      ? "本帧 M⁻¹A = A（两组散点逐点重合）：「预条件」不等于「一定有效」。"
      : "两组散点的横向差 =「谱被压到哪一段」；κ 只是两端之比。",
  );

  const convLines: string[] = [];
  convLines.push("曲线：(1) 残差 (2) 误差 (3) CG 最坏界 (4) 实测因子^k");
  convLines.push(
    `κ ${fmtOr(kappaPre, "帧没给")} → ${fmtOr(kappaPost, "帧没给")}` +
      (kappaRatio === undefined ? "" : `（降 ${formatNumber(kappaRatio, 4)} 倍）`) +
      `；${fmtOr(nIter, "?")} 步${converged ? "（已收敛）" : "（未到 tol）"}` +
      (omega === undefined ? "" : `；ω=${formatNumber(omega, 4)}`),
  );
  convLines.push(
    kappaEffective === undefined
      ? "等效 κ（由实测因子反推）：帧是 null，不显示（不按 0 处理）"
      : `等效 κ = ${formatNumber(kappaEffective, 4)}（帧 meta，由迭代数反推）`,
  );
  convLines.push(
    "κ 降幅 != 迭代数降幅：谱聚簇时 CG 远快于最坏界",
  );
  convLines.push(
    iluSentinel
      ? // 帧的 sentinel_note 有 3 行那么长：只取结论（-1 = 没有丢弃概念），
        // 判据写 applicable，剩下的留给帧的 note 行（全引会把 ω 那一行挤出画布）。
        `ilu_dropped=${formatNumber(iluDroppedRaw ?? -1, 3)}：帧 sentinel_note 说这是「没有丢弃概念」` +
        `${sentinelNote === undefined ? "（帧没给 sentinel_note）" : ""}；判据 applicable=` +
        `${iluApplicable === undefined ? "帧未给" : String(iluApplicable)}`
      : iluDropped === undefined
        ? "帧 meta 缺 ilu_dropped：不显示丢弃填充次数。"
        : `ILU(0) 丢弃填充 ${formatNumber(iluDropped, 4)} 次（帧 meta）；一维上 ILU(0) 是精确 LU（1 步收敛），不能外推到二维。`,
  );
  if (omega === undefined) {
    const requested = paramsJson["omega"];
    convLines.push(
      `ω：不适用（帧 meta.omega = ${formatNumber(omegaRaw ?? 0, 3)} 是哨兵：只有 SSOR 用 ω）` +
        (typeof requested === "number" ? `；帧 strings.params_json 里的请求值 ${formatNumber(requested, 4)} 本帧用不上。` : "。"),
    );
  } else {
    convLines.push(`ω=${formatNumber(omega, 4)}；ω=1 退化为对称 GS，需 0<ω<2 才保对称正定`);
  }
  if (bound.length > 0) {
    const bLast = bound[bound.length - 1] as number;
    convLines.push(`最坏界末值 ${formatNumber(bLast, 3)} vs 实测 ${formatNumber(rLast, 3)}：悲观 ${formatNumber(bLast / Math.max(rLast, Number.MIN_VALUE), 3)} 倍`);
  }
  if (!factorLineDrawn) {
    convLines.push(
      factorMeasured === undefined
        ? "实测因子参考线不画：帧 meta.factor_measured 为 null（残差已到 0 或尾部点不足）。"
        : `实测因子 ${formatNumber(factorMeasured, 4)} 的参考线在本轴内不足两点，不画。`,
    );
  }
  if (note !== undefined) convLines.push(`帧 note：${note}`);

  scene.texts.push(...textBlock(base.left, specRect[1] + specRect[3] + 28, base.bottom - 4, specLines, pal.muted, unitBudget(specW), "top"));
  scene.texts.push(...textBlock(convX, convRect[1] + convRect[3] + 28, base.bottom - 4, convLines, pal.muted, unitBudget(convW), "top"));

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", `${axisModeBadge(false).text}（spectrum：纵轴固定 [0, 8.2]，横轴是 Ritz 序号）`, 8, 8), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(badge("autoscale", "副轴 convergence：自适应（log 纵轴按本帧残差/误差/界取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  scene.badges.push(placeBadge(omegaBadge(omega, omegaRaw, isSsor, criticalOf(frame, "omega", OMEGA_DEGENERATE)), SLOT.criticalOmega, width));
  const status: string[] = [];
  if (precondName !== undefined) status.push(`预条件 ${precondName}`);
  if (solver !== undefined) status.push(`解法 ${solver.toUpperCase()}`);
  if (nIter !== undefined) status.push(`迭代 ${formatNumber(nIter, 4)} 步${converged ? "（已收敛）" : ""}`);
  if (kappaPre !== undefined && kappaPost !== undefined) {
    status.push(`κ ${formatNumber(kappaPre, 4)}→${formatNumber(kappaPost, 4)}`);
  }
  if (iluSentinel) status.push(`ilu_dropped 不适用（applicable=${iluApplicable === undefined ? "帧未给" : String(iluApplicable)}）`);
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 meta.n_iter 与 strings.note）");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(ritzPre) +
    countNonFinite(ritzPost) +
    countNonFinite(ritzIndexPre) +
    countNonFinite(ritzIndexPost) +
    countNonFinite(convergence) +
    countNonFinite(residual) +
    countNonFinite(error) +
    countNonFinite(bound) +
    countNonFinite(stepGrid);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function fmtOr(v: number | undefined, fallback: string): string {
  return v === undefined ? fallback : formatNumber(v, 4);
}

/** 两个等长数组是否逐点相等（用于识别 `precond = none` 时 ritz_post == ritz_pre）。 */
function sameValues(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length || a.length === 0) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (!Object.is(a[i], b[i])) return false;
  }
  return true;
}

/** `strings.params_json`（后端写的规范化参数）→ 对象；解析失败给空对象（不猜）。 */
function paramsJsonOf(frame: Frame): Record<string, number | string | boolean> {
  const raw = stringOf(frame, "params_json");
  if (raw === undefined) return {};
  try {
    const v: unknown = JSON.parse(raw);
    return typeof v === "object" && v !== null ? (v as Record<string, number | string | boolean>) : {};
  } catch {
    return {};
  }
}

/** 对数副轴上的因子^k 直线：逐点裁到轴内（折线不被渲染器裁剪，越界会触发溢出角标）。 */
function decayLine(rate: number, kMax: number, floor: number): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  if (!Number.isFinite(rate) || !(rate > 0)) return { x, y };
  for (let k = 0; k <= kMax; k += 1) {
    const v = Math.pow(rate, k);
    if (!Number.isFinite(v) || v < floor) break;
    x.push(k);
    y.push(v);
  }
  return { x, y };
}

/**
 * ω 徽标：SSOR 的两条边界（ω=1 退化、0<ω<2 才保持对称正定）。
 * 非 SSOR 的帧里 `meta.omega = 0` 是哨兵，这里明说"不适用"，不显示成 ω = 0。
 */
function omegaBadge(omega: number | undefined, omegaRaw: number | undefined, isSsor: boolean, degenerate: number | number[]): Badge {
  const deg = Array.isArray(degenerate) ? (degenerate[0] ?? OMEGA_DEGENERATE) : degenerate;
  if (!isSsor || omega === undefined) {
    const b = badge("critical", `ω：不适用（帧 meta.omega = ${formatNumber(omegaRaw ?? 0, 3)} 是哨兵：只有 SSOR 用 ω）`, 8, 8);
    b.color = "#718096";
    return b;
  }
  const parts = [`ω = ${formatNumber(omega, 4)}`, `ω=${formatNumber(deg, 4)} 退化为 SSGS`, "需 0<ω<2 才保持对称正定"];
  let warn = false;
  if (omega >= 2) {
    parts.push("! 已越界：预条件子不再对称正定");
    warn = true;
  } else if (Math.abs(omega - deg) < 1e-9) {
    parts.push("（恰在退化点）");
    warn = true;
  }
  const b = badge("critical", parts.join("　"), 8, 8);
  b.color = warn ? "#c53030" : "#2f855a";
  return b;
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
 * 折线图例标签只在「画得下」时给。
 *
 * 光栅器（`tools/raster.py`）把 series 的 label 画在**曲线末点的右侧**，而画布不裁剪文字：
 * 末点贴着面板右缘时标签会直接冲出画布（Scene 契约里没有「图例」节点，这是渲染器的加法）。
 * 所以这里按末点到轴右端的剩余像素判断，画不下就不给，改由面板文本里的文字图例承担。
 */
function endLabel(xLast: number, xlim: readonly [number, number], panelW: number, label: string): string | undefined {
  if (!Number.isFinite(xLast) || !(xlim[1] > xlim[0])) return undefined;
  const room = ((xlim[1] - xLast) / (xlim[1] - xlim[0])) * panelW;
  return room > textWidthUnits(label) * 7 + 10 ? label : undefined;
}

/** 帧 `meta` 里的真布尔值；缺失/`null`/数字一律 undefined（0/1 不当布尔猜）。 */
function metaBool(frame: Frame, key: string): boolean | undefined {
  const v = frame.meta[key];
  return typeof v === "boolean" ? v : undefined;
}

/** 面板宽度 → 显示宽度单位预算（size 11 时 1 单位 ≈ 7px）。 */
function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/** 一段说明文字 → 若干 `text` 层，**一行一个**（`render2d.ts::drawText` 不拆 `\n`）。 */
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
  // 折行后做一次「短行拼接」：把装得下的相邻短句并进同一行。
  // 不做这一步的话，一句 20 单位的短句会独占一行，白白吃掉一行的高度。
  let pieces: string[] = [];
  for (const body of lines) {
    for (const part of wrapUnits(body, budget)) {
      const last = pieces[pieces.length - 1];
      if (last !== undefined && textWidthUnits(last) + 1 + textWidthUnits(part) <= budget) {
        pieces[pieces.length - 1] = `${last}　${part}`;
      } else {
        pieces.push(part);
      }
    }
  }
  if (bottom > top) {
    const maxRows = Math.max(1, Math.floor((bottom - top) / rowH) + 1);
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `...（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - (pieces.length - 1) * rowH);
  return pieces.map((body, i) => text(x, y + i * rowH, body, { role: "note", size, color, maxWidth: budget * 7 }));
}

/** 按显示宽度单位折行（CJK=2、ASCII=1；口径同 `model.ts::textWidthUnits`）。 */
function wrapUnits(body: string, budget: number): string[] {
  const out: string[] = [];
  let line = "";
  let used = 0;
  const push = (): void => {
    if (line !== "") out.push(line);
    line = "";
    used = 0;
  };
  // 折行点：空白、顿号/逗号、句读、右括号之后（CJK 长句没有空格，只在空白处折会把
  // 「v0(0.297, 0.493)」这种坐标从中间劈开）。
  for (const word of body.split(/(?<=[\s　、，,。；;：:）)])/)) {
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
