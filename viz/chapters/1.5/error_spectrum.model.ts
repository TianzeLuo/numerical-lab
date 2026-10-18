/**
 * `error_spectrum`（1.5 章）的 model：误差在解析正弦模态上的分量条形（对数色阶）+ 误差/残差副轴。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | heat | `modes` | `components_log10`（行 = 模态 j，列 = 迭代步 k） |
 * | 2 | line | `modes` | 第 k 步的游标竖线 |
 * | 3 | scatter | `modes` | `components_mark`：第 k 步各模态的分量 |
 * | 4 | line | `decay` | `error`（2-范数）、`error_A`（A-范数）、`residual` 三条曲线 |
 * | 5 | line | `decay` | 理论最慢因子^k 与实测 ρ^k 两条参考线 |
 * | 6 | scatter | `decay` | 第 k 步的 `error_mark`/`residual_mark` |
 * | 7 | line | `rates` | 每个模态的解析因子 `mode_rates_theory`（折线 + 圆点） |
 * | 8 | scatter | `rates` | 每个模态的**实测**比值 `mode_rates_measured`（掩码为 1 的才画） |
 * | 9 | line | `rates` | \|λ\|=1 参考线与最慢模态因子参考线 |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 / 3 ω 临界 /
 * 4 状态 / 5 第三面板 rates 自适应 / 6 溢出 / 7 降级。
 *
 * 几条刻意的决定
 * --------------
 * 1. **热力图给的是整段 61 步，不是「到第 k 步为止」**：`components` 在后端一次算完
 *    （所有帧里逐位相同），按游标裁剪反而丢掉了这张图唯一要回答的问题——「谁先衰减完」。
 *    所以热力图恒画整段，**游标与副轴曲线**才表示第 k 步；这一点写在面板文本里，
 *    免得被读成「第 0 步就已经衰减完了」。
 * 2. **`mode_rates_measured` 的 `-1` 是哨兵，掩码才是真值**：`mode_rates_measured_valid`
 *    为 0 的模态"没测到"（幅值在 1e-14 以上的窗口不足 3 点、或比值算不出来）。
 *    这些点**不画**、也**不当 0**，只在文本与状态角标里报出个数——`-1` 画上去会变成
 *    「衰减因子 −1」这种荒谬读数。帧的 `strings` 里目前没有解释这个哨兵（见交付报告）。
 * 3. **色阶地板也是结论**：`components_log10` 已经把分量压到 `meta.component_floor = 1e-16`
 *    的地板上（后端 `np.maximum(comp, floor)`）。色标下端就取 `log10(floor)`，
 *    文本里写明"地板以下的条形是舍入噪声，不是误差真的更小"——否则一片深蓝会被读成
 *    "已经精确到 1e-300"。
 * 4. **第三面板 `rates` 是本章新增的轴**（manifest 只声明了 `modes`/`decay` 两根）：
 *    「每个模态旁标注理论因子与实测比值」如果塞成 16 行文本会挤爆版面，做成散点面板后
 *    「解析 vs 实测」一眼可比，`-1` 哨兵也自然缺席。它的 `autoscaled: true` 在角标里
 *    单独声明（manifest 的 `axes` 列表里没有它，见交付报告）。
 * 5. **模态分量不单调**：GS/SOR 下 A 的特征基不是 T 的特征基，低频模态会先混后衰减
 *    （实测 n=64/ω=1.9 时某模态的实测比值 1.542 > 1，即它先变大）。`rates` 面板的
 *    对数纵轴因此**必须容纳 >1 的值**，并在 1 处画参考线。
 * 6. **两条参考线的斜率来自帧 meta**（`slowest_rate` 解析式、`rho_measured` 幂法实测），
 *    前端只做"在对数轴上画斜率给定的直线"，并逐点裁到轴内（折线不被渲染器裁剪）。
 */

import { formatNumber, niceLimits, niceTicks } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  resolveTextOverlaps,
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
  pitfallsBand,
  axisModeBadge,
  badge,
  heatLayer,
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
const VIZ = "error_spectrum";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "误差的特征分量分解：谁先衰减完";
/** manifest 里 `omega.critical = [1.0]`；`frame.critical.omega` 缺失时的回落值。 */
const OMEGA_DEGENERATE = 1.0;
/** 后端 `_measured_mode_rates` 的"测得到"门槛（比值窗口里幅值必须 > 1e-14）。 */
const RATE_MEASURE_FLOOR = 1e-14;

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。本帧数据源恒为 backend，故不占 preview 槽。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  criticalOmega: 3,
  status: 4,
  ratesAxis: 5,
  overflow: 6,
  degraded: 7,
} as const;

export const errorSpectrumModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const components = frameMatrix(frame, "components");
  const componentsLog = frameMatrix(frame, "components_log10");
  const modeRatesTheory = requireArray(frame, "mode_rates_theory");
  const modeRatesMeasured = requireArray(frame, "mode_rates_measured");
  const error = requireArray(frame, "error");
  const residual = requireArray(frame, "residual");
  const stepGrid = requireArray(frame, "step_grid");
  if (components.rows !== componentsLog.rows || components.cols !== componentsLog.cols) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 components ${components.rows}×${components.cols} 与 components_log10 ` +
        `${componentsLog.rows}×${componentsLog.cols} 形状不一致：条形与色阶必须逐格对应`,
    );
  }
  if (modeRatesTheory.length !== components.rows || modeRatesMeasured.length !== components.rows) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的模态个数不一致：components ${components.rows} 行、` +
        `mode_rates_theory ${modeRatesTheory.length} 个、mode_rates_measured ${modeRatesMeasured.length} 个`,
    );
  }
  if (error.length > stepGrid.length || residual.length > stepGrid.length) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 error/residual 长度（${error.length}/${residual.length}）超过 step_grid 的 ` +
        `${stepGrid.length}：前缀曲线必须落在步网格之内（拒绝画错位的曲线）`,
    );
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const nStepsFrame = metaNumber(frame, "n_steps");
  const k = metaNumber(frame, "step");
  if (k === undefined) degraded.push("帧 meta 缺 step：不画游标与第 k 步标记（不猜是第几步）");
  if (nStepsFrame === undefined) degraded.push("帧 meta 缺 n_steps：热力图横轴按 components 的列数给（列数是数据，不是猜的）");
  if (k !== undefined && Number.isFinite(opts.t) && Math.round(opts.t) !== Math.round(k)) {
    degraded.push(
      `请求 t=${formatNumber(opts.t, 4)} 夹到第 ${formatNumber(k, 4)} 步：帧给的是四舍五入并夹到 [0, n_steps] 的整数步`,
    );
  }
  const nMeta = metaNumber(frame, "n");
  const nShown = metaNumber(frame, "n_modes_shown");
  if (nShown !== undefined && Math.round(nShown) !== components.rows) {
    degraded.push(
      `帧 meta.n_modes_shown=${formatNumber(nShown, 4)} 与 components 的 ${components.rows} 行不一致：按数组行数画`,
    );
  }
  const methodName = stringOf(frame, "method_name") ?? stringOf(frame, "method");
  const initName = stringOf(frame, "init_name") ?? stringOf(frame, "init");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const componentFloor = metaNumber(frame, "component_floor");
  const rhoTheory = metaNumber(frame, "rho_theory");
  const rhoMeasured = metaNumber(frame, "rho_measured");
  const slowestRate = metaNumber(frame, "slowest_rate");
  const slowestMode = metaNumber(frame, "slowest_mode");
  const nMeasured = metaNumber(frame, "n_modes_measured");
  const omega = metaNumber(frame, "omega");
  const omegaOpt = metaNumber(frame, "omega_opt");
  const errorMark = metaNumber(frame, "error_current") ?? metaNumber(frame, "error_final");
  const residualMarkMeta = metaNumber(frame, "residual");
  const componentMarkArr = optionalArray(frame, "components_mark");
  const errorMarkArr = optionalArray(frame, "error_mark");
  const residualMarkArr = optionalArray(frame, "residual_mark");
  const errorA = optionalArray(frame, "error_A");
  const ratesValidArr = optionalArray(frame, "mode_rates_measured_valid");
  const modeIndexArr = optionalArray(frame, "modes_index");
  if (ratesValidArr === undefined) {
    degraded.push("帧缺 mode_rates_measured_valid：实测比值按「有限且 >0」过滤，无法区分「没测到」与「真的为 0」");
  }
  if (modeIndexArr === undefined) degraded.push("帧缺 modes_index：模态横轴按 1...M 编号（M = components 行数）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左列热力图（+ 色标过道），右列副轴与 rates 面板 ——
  // 色标过道：`tools/raster.py` 会在每个 heat 轴右侧 46px 处画 14px 宽的色标条，数值标签再往右
  // ~60px（Scene 契约里没有色标节点，这是光栅器的加法）。不留这条过道，色标就压到右列上。
  const base = stackedPanels(width, height);
  const contentH = base.bottom - base.top;
  const COLORBAR_GUTTER = 120;
  const leftW = Math.max(200, Math.min(Math.round(base.contentWidth * 0.5), base.contentWidth - 260 - COLORBAR_GUTTER));
  const rightX = base.left + leftW + COLORBAR_GUTTER;
  const rightW = Math.max(180, base.left + base.contentWidth - rightX);
  const heatH = Math.max(120, Math.round(contentH * 0.47)); // 面板文本要 12 行：数字全在这一栏交代

  const heatRect: Rect = [base.left, base.top, leftW, heatH];
  const decayH = Math.max(110, Math.round(heatH * 0.62));
  const decayRect: Rect = [rightX, base.top, rightW, decayH];
  const ratesRect: Rect = [rightX, base.top + decayH + 54, rightW, Math.max(90, contentH - decayH - 54)];

  const nSteps = Math.max(1, (nStepsFrame ?? components.cols - 1) as number);
  const nModes = components.rows;
  // 热力图：行 = 模态 j（1...M，在 extent 里正好落在整数格点 [1, M]），列 = 迭代步 k（[0, nSteps]）。
  const modesAxis = makeAxis("modes", heatRect, {
    xlim: [-0.5, nSteps + 0.5],
    ylim: [0.5, nModes + 0.5],
    xlabel: "迭代步 k",
    ylabel: axisYlabel("modes", "模态 j"),
    title: axisTitle("modes", "各模态的误差分量（对数色阶条形）"),
    autoscaled: false,
    xticks: niceTicks(0, nSteps, Math.max(3, Math.round(heatRect[2] / 90))),
    yticks: modeTicks(nModes),
  });

  // 副轴（manifest 声明 `decay.autoscaled = true`、`scale = "log"`）：范围按整段取，不随 t 变。
  const errPos = finitePositiveMinMax(error);
  const resPos = finitePositiveMinMax(residual);
  const errAPos = errorA === undefined ? undefined : finitePositiveMinMax(errorA);
  const lo = Math.min(errPos?.[0] ?? Number.POSITIVE_INFINITY, resPos?.[0] ?? Number.POSITIVE_INFINITY, errAPos?.[0] ?? Number.POSITIVE_INFINITY);
  const hi = Math.max(errPos?.[1] ?? Number.NEGATIVE_INFINITY, resPos?.[1] ?? Number.NEGATIVE_INFINITY, errAPos?.[1] ?? Number.NEGATIVE_INFINITY);
  const hasPositive = Number.isFinite(lo) && Number.isFinite(hi);
  const decayYlim: [number, number] = hasPositive ? niceLimits(lo, hi, 7, { log: true, pad: 0.2 }) : [1e-16, 1];
  if (!hasPositive) degraded.push("error/residual 里没有 >0 的有限值：副轴用占位范围 [1e-16, 1]，且不画曲线");
  const stepLen = Math.max(1, stepGrid.length - 1);
  const decayAxis = makeAxis("decay", decayRect, {
    xlim: [0, stepLen],
    ylim: decayYlim,
    yscale: "log",
    xlabel: "迭代步 k",
    ylabel: axisYlabel("decay", "‖e_k‖/‖e_0‖ 与 ‖r_k‖/‖r_0‖"),
    title: axisTitle("decay", "误差与残差（对数）"),
    autoscaled: true,
    xticks: niceTicks(0, stepLen, Math.max(3, Math.round(decayRect[2] / 90))),
  });

  // —— 第三面板：解析 vs 实测衰减因子（见文件头第 4 条）——
  const validFlags = validMask(ratesValidArr, modeRatesMeasured);
  const theoryRates = Array.from({ length: nModes }, (_, j) => modeRatesTheory[j] as number);
  const measuredAll = Array.from({ length: nModes }, (_, j) => modeRatesMeasured[j] as number);
  const measuredShown = measuredAll.map((v, j) => (validFlags[j] && Number.isFinite(v) && v > 0 ? v : Number.NaN));
  const nSentinel = validFlags.reduce((acc, ok) => acc + (ok ? 0 : 1), 0);
  const nNonPositive = measuredShown.reduce((acc, v, j) => acc + (validFlags[j] && !(v > 0) ? 1 : 0), 0);
  const rateValues = [...theoryRates.filter((v) => Number.isFinite(v) && v > 0), ...measuredShown.filter((v) => Number.isFinite(v) && v > 0), 1];
  const rateLo = Math.min(...rateValues);
  const rateHi = Math.max(...rateValues);
  const rateYlim: [number, number] = niceLimits(rateLo, rateHi, 6, { log: true, pad: 0.15 });
  const ratesAxis = makeAxis("rates", ratesRect, {
    xlim: [0.5, nModes + 0.5],
    ylim: rateYlim,
    yscale: "log",
    xlabel: "模态编号 j",
    ylabel: "衰减因子 |λ_j(T)|",
    title: "解析因子 vs 实测比值（对数）",
    autoscaled: true,
    xticks: modeTicks(nModes),
  });
  scene.axes.push(modesAxis, decayAxis, ratesAxis);

  // —— 图层 1~3：模态热力图 + 游标 + 当前步分量 ——
  const floorLog = componentFloor !== undefined && componentFloor > 0 ? Math.log10(componentFloor) : minFinite(componentsLog.values);
  const maxLog = maxFinite(componentsLog.values, floorLog);
  const vmaxLog = Math.max(floorLog + 1, Math.ceil(maxLog));
  if (components.cols >= 2 && nModes >= 2) {
    scene.layers.push(
      heatLayer("modes", componentsLog.values, components.cols, nModes, [0, nSteps, 1, nModes], {
        cmap: "viridis",
        vmin: floorLog,
        vmax: vmaxLog,
        // 值已经是 log10（后端在地板处截断过）：再开 logScale 就成了 loglog，是另一张图。
        logScale: false,
        label: "log10|分量|",
      }),
    );
  } else {
    degraded.push(
      `components 是 ${components.rows}×${components.cols}：热力图至少需要 2×2 网格（节点采样含两端），本帧不画`,
    );
  }
  const cut = k === undefined ? Math.max(0, Math.min(stepGrid.length - 1, nSteps)) : clampInt(k, 0, nSteps);
  if (cut <= nSteps) {
    scene.layers.push(lineLayer("modes", [series([cut, cut], [0.5, nModes + 0.5], pal.cursor, { dash: [3, 3], width: 1.2 })]));
  }
  const markX: number[] = [];
  const markY: number[] = [];
  if (componentMarkArr !== undefined) {
    for (let j = 0; j < Math.min(nModes, componentMarkArr.length); j += 1) {
      const v = componentMarkArr[j] as number;
      if (Number.isFinite(v) && v > 0) {
        markX.push(cut);
        markY.push(j + 1);
      }
    }
  }
  if (markX.length > 0) {
    scene.layers.push(
      scatterLayer("modes", markX, markY, { r: 2.4, color: pal.marker, label: `第 ${formatNumber(cut, 4)} 步各模态的分量（帧 components_mark）` }),
    );
  } else {
    degraded.push("帧缺 components_mark（或没有一个 >0 的分量）：热力图上不标当前步的分量点");
  }

  // —— 图层 4~6：副轴的三条曲线 + 两条参考线 + 当前值 ——
  const nDraw = Math.min(cut + 1, stepGrid.length);
  const xs = prefixOf(stepGrid, nDraw);
  const decaySeries: ReturnType<typeof series>[] = [];
  decaySeries.push(
    ...segmentedSeries(xs, prefixOf(error, nDraw), pal.curves[0], {
      width: 1.8,
      label: endLabel(xs[xs.length - 1] ?? Number.NaN, decayAxis.xlim, decayRect[2], "相对误差（帧 error）"),
      accept: (v) => v > 0,
    }),
  );
  decaySeries.push(
    ...segmentedSeries(xs, prefixOf(residual, nDraw), pal.residual, {
      width: 1.4,
      dash: [6, 3],
      // 三条曲线末点都在同一处，只给第一条挂末点标签，其余靠下面的文字图例（避免叠字）。
      label: undefined,
      accept: (v) => v > 0,
    }),
  );
  if (errorA !== undefined) {
    decaySeries.push(
      ...segmentedSeries(xs, prefixOf(errorA, nDraw), pal.exact, {
        width: 1.2,
        dash: [2, 3],
        label: endLabel(xs[xs.length - 1] ?? Number.NaN, decayAxis.xlim, decayRect[2], "A-范数误差（帧 error_A）"),
        accept: (v) => v > 0,
      }),
    );
  }
  const refLines: string[] = [];
  refLines.push("曲线：(1) 相对误差 (2) 相对残差 (3) A-范数误差；参考线：(4) 最慢因子^k (5) ρ 实测^k");
  const refs: ReturnType<typeof series>[] = [];
  const slow = slowestRate !== undefined && slowestRate > 0 ? decayLine(slowestRate, stepLen, decayYlim[0]) : undefined;
  if (slow !== undefined && slow.x.length >= 2) {
    refs.push(
      series(slow.x, slow.y, pal.reference, {
        width: 1.2,
        dash: [6, 4],
        label: endLabel(slow.x[slow.x.length - 1] ?? Number.NaN, decayAxis.xlim, decayRect[2], `最慢模态因子^k（${formatNumber(slowestRate as number, 4)}）`),
      }),
    );
  }
  const measRho = rhoMeasured !== undefined && rhoMeasured > 0 ? decayLine(rhoMeasured, stepLen, decayYlim[0]) : undefined;
  if (measRho !== undefined && measRho.x.length >= 2) {
    refs.push(
      series(measRho.x, measRho.y, pal.note, {
        width: 1.2,
        dash: [2, 3],
        label: endLabel(measRho.x[measRho.x.length - 1] ?? Number.NaN, decayAxis.xlim, decayRect[2], `ρ^k 幂法实测（${formatNumber(rhoMeasured as number, 4)}）`),
      }),
    );
  }
  if (refs.length > 0) decaySeries.push(...refs);
  if (refs.length > 0) {
    refLines.push(
      `参考线：虚线 = 最慢因子^k（${slowestRate === undefined ? "帧没给" : formatNumber(slowestRate, 4)}）、点线 = ρ 实测^k（${rhoMeasured === undefined ? "帧没给" : formatNumber(rhoMeasured, 4)}）`,
    );
  } else {
    refLines.push("参考线：帧 meta 缺 slowest_rate/rho_measured（或不足两点），不画理论直线（不猜）");
  }
  if (decaySeries.length > 0) scene.layers.push(lineLayer("decay", decaySeries));

  const nowX: number[] = [];
  const nowY: number[] = [];
  const eNow = errorMarkArr !== undefined && errorMarkArr.length > 0 ? (errorMarkArr[0] as number) : errorMark;
  const rNow = residualMarkArr !== undefined && residualMarkArr.length > 0 ? (residualMarkArr[0] as number) : residualMarkMeta;
  if (eNow !== undefined && Number.isFinite(eNow) && eNow > 0) {
    nowX.push(cut);
    nowY.push(eNow);
  }
  if (rNow !== undefined && Number.isFinite(rNow) && rNow > 0) {
    nowX.push(cut);
    nowY.push(rNow);
  }
  if (nowX.length > 0) {
    scene.layers.push(scatterLayer("decay", nowX, nowY, { r: 3.6, color: pal.marker, label: "第 k 步的误差/残差（帧 *_mark）" }));
  }

  // —— 图层 7~9：解析因子 vs 实测比值 ——
  const jx = Array.from({ length: nModes }, (_, j) => j + 1);
  const theoryShown = theoryRates.map((v) => (Number.isFinite(v) && v > 0 ? v : Number.NaN));
  const theoryLine = theoryShown.filter((v) => Number.isFinite(v));
  if (theoryLine.length >= 2) {
    scene.layers.push(
      lineLayer("rates", [
        series(
          jx.filter((_, j) => Number.isFinite(theoryShown[j])),
          jx.map((_, j) => theoryShown[j]).filter((v) => Number.isFinite(v)),
          pal.curves[0],
          { width: 1.4, marker: "o" },
        ),
      ]),
    );
  }
  const measuredX = jx.filter((_, j) => Number.isFinite(measuredShown[j]));
  const measuredY = measuredShown.filter((v) => Number.isFinite(v));
  if (measuredX.length > 0) {
    scene.layers.push(
      scatterLayer("rates", measuredX, measuredY, { r: 3, color: pal.marker, label: "实测比值（帧 mode_rates_measured，掩码为 1 的才画）" }),
    );
  } else {
    degraded.push("没有一个模态的实测比值可用（掩码全 0）：rates 面板只画解析因子，不补点");
  }
  // 参考线：|λ|=1（不增不减）与最慢模态因子（决定渐近速率）。
  const refRate: ReturnType<typeof series>[] = [];
  if (rateYlim[0] <= 1 && 1 <= rateYlim[1]) {
    refRate.push(series([0.5, nModes + 0.5], [1, 1], pal.warn, { dash: [5, 4], width: 1 }));
  }
  if (slowestRate !== undefined && slowestRate > rateYlim[0] && slowestRate < rateYlim[1]) {
    refRate.push(
      series([0.5, nModes + 0.5], [slowestRate, slowestRate], pal.reference, {
        dash: [2, 4],
        width: 1,
        // 标签落在轴右缘会溢出画布：这条参考线的值写进面板文本。
        label: undefined,
      }),
    );
  }
  if (refRate.length > 0) scene.layers.push(lineLayer("rates", refRate));
  // 用 `bars` 把"每个模态的解析因子"再画一遍会与折线重叠，故这里只把最慢模态标出来（文字层）。
  if (slowestMode !== undefined && slowestMode >= 0 && slowestMode < nModes) {
    const jSlow = slowestMode + 1;
    scene.texts.push(
      text(
        ratesRect[0] + Math.max(4, Math.min(ratesRect[2] - 64, ((jSlow - 0.5) / nModes) * ratesRect[2] + 6)),
        ratesRect[1] + 3,
        `最慢 j=${formatNumber(slowestMode + 1, 3)}`,
        { role: "note", size: 10, color: pal.marker, maxWidth: unitBudget(ratesRect[2]) * 7 },
      ),
    );
  } else if (slowestMode === undefined) {
    degraded.push("帧 meta 缺 slowest_mode：不在 rates 面板上标最慢模态");
  }

  // —— 文本（一行一个 text 层）——
  const lines = summaryLines({
    nMeta,
    nModes,
    step: cut,
    nSteps,
    methodName,
    initName,
    omega,
    omegaOpt,
    rhoTheory,
    rhoMeasured,
    slowestRate,
    slowestMode,
    nMeasured,
    nSentinel,
    nNonPositive,
    componentFloor,
    errorMark: eNow,
    residualMark: rNow,
    vminLog: floorLog,
    vmaxLog,
    refLines,
    note,
  });
  // 左列文本块贴在热力图下方：让开热力图自己的刻度带(15)+轴名带(12) = 27px。
  // 横向只能铺到右列轴**左侧装饰带**的起点（rightX − 56）：再往右就被 y 轴名/刻度带压住（规则 8）。
  const heatBudget = Math.max(24, Math.floor((rightX - 56 - base.left - 8) / 7));
  scene.texts.push(
    ...textBlock(base.left, heatRect[1] + heatRect[3] + 28, base.bottom - 4, lines, pal.muted, heatBudget, "top"),
  );

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", `${axisModeBadge(false).text}（modes：模态与步网格都固定）`, 8, 8), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(
      badge("autoscale", "副轴 decay：自适应（log 纵轴按整段误差/残差取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  if (omega !== undefined) {
    scene.badges.push(placeBadge(omegaBadge(omega, omegaOpt, criticalOf(frame, "omega", OMEGA_DEGENERATE)), SLOT.criticalOmega, width));
  }
  const status: string[] = [];
  if (methodName !== undefined) status.push(`方法 ${methodName}`);
  status.push(`第 ${formatNumber(cut, 4)}/${formatNumber(nSteps, 4)} 步`);
  // 哨兵警告排在数值前面：状态角标会被 `clipBadgeText` 按宽度截断，
  // 排在末尾的话「N 个模态没测到」正好是被切掉的那一段（角标里看不见 = 没说）。
  if (nSentinel > 0) status.push(`! 实测比值里 -1 哨兵 ${nSentinel} 个（没测到），未画`);
  if (eNow !== undefined) status.push(`‖e‖/‖e_0‖=${formatNumber(eNow, 3)}`);
  if (rNow !== undefined) status.push(`‖r‖/‖r_0‖=${formatNumber(rNow, 3)}`);
  if (note !== undefined) status.push(note);
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "面板 rates：自适应（对数纵轴按本帧解析/实测因子取整，含 |λ|=1 参考线）", 8, 8),
      SLOT.ratesAxis,
      width,
    ),
  );
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(components.values) +
    countNonFinite(componentsLog.values) +
    countNonFinite(modeRatesTheory) +
    countNonFinite(modeRatesMeasured) +
    countNonFinite(error) +
    countNonFinite(residual) +
    countNonFinite(stepGrid);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  // 底部 pitfalls 面板（规则 7）由渲染层现画，模型看不见它：用共享的 pitfallsBand 拿到
  // 面板顶边，再让 resolveTextOverlaps 把放不下的文字如实计数后丢掉（窄画布下才生效）。
  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);

  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

interface Matrix2D {
  rows: number;
  cols: number;
  values: ArrayLike<number>;
}

/** 二维数组（行主序）：缺数组抛 `ModelInputError`；形状不是二维也抛。 */
function frameMatrix(frame: Frame, name: string): Matrix2D {
  const values = requireArray(frame, name);
  const shape = frame.shapeOf(name);
  if (shape.length !== 2) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 '${name}' 形状 [${shape.join(", ")}] 不是二维：本图按「行 = 模态、列 = 迭代步」消费`,
    );
  }
  const rows = shape[0] as number;
  const cols = shape[1] as number;
  if (values.length !== rows * cols) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 '${name}' 有 ${values.length} 个元素，与 ${rows}×${cols} 不符`);
  }
  return { rows, cols, values };
}

function prefixOf(a: ArrayLike<number>, n: number): Float64Array {
  const len = Math.max(0, Math.min(n, a.length));
  const out = new Float64Array(len);
  for (let i = 0; i < len; i += 1) out[i] = a[i] as number;
  return out;
}

function minFinite(a: ArrayLike<number>): number {
  let lo = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && v < lo) lo = v;
  }
  return Number.isFinite(lo) ? lo : -16;
}

function maxFinite(a: ArrayLike<number>, fallback: number): number {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : fallback;
}

function clampInt(v: number, lo: number, hi: number): number {
  const r = Math.round(v);
  if (!Number.isFinite(r)) return lo;
  return r < lo ? lo : r > hi ? hi : r;
}

/**
 * 实测比值的可用掩码：帧给了 `mode_rates_measured_valid` 就用它（哨兵 `-1` 由后端解释），
 * 缺掩码时只能退化成「有限且 >0」——这时 `-1` 会被当成不可用而不是一个数，仍然是安全的。
 */
function validMask(mask: ArrayLike<number> | undefined, measured: ArrayLike<number>): boolean[] {
  const out: boolean[] = [];
  for (let j = 0; j < measured.length; j += 1) {
    const v = measured[j] as number;
    if (mask === undefined) out.push(Number.isFinite(v) && v > 0);
    else out.push((mask[j] as number) > 0.5 && Number.isFinite(v) && v !== -1);
  }
  return out;
}

/** 模态编号刻度：1...M，太多时按步进取样（避免刻线糊成一片）。 */
function modeTicks(n: number): number[] {
  if (n <= 0) return [];
  const step = Math.max(1, Math.ceil(n / 10));
  const out: number[] = [];
  for (let j = 1; j <= n; j += step) out.push(j);
  return out;
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
 * ω 的双阈值徽标：ω=1 是"退化为 GS"的等式边界，ω_opt 是"ρ 最小"的最优点，
 * 越过 ω_opt 之后主特征值变成复共轭对、ρ 随 ω 上升（本章 pitfalls 第 5 条）。
 * `criticalBadge` 的单向语义表达不了这个形状，所以自己拼 `kind="critical"`。
 */
function omegaBadge(omega: number, omegaOpt: number | undefined, degenerate: number | number[]): Badge {
  const deg = Array.isArray(degenerate) ? (degenerate[0] ?? OMEGA_DEGENERATE) : degenerate;
  const parts = [`ω = ${formatNumber(omega, 4)}`, `ω=${formatNumber(deg, 4)} 退化为 GS`];
  if (omegaOpt !== undefined) parts.push(`ω_opt=${formatNumber(omegaOpt, 4)}（帧 meta）`);
  let warn = false;
  if (Math.abs(omega - deg) < 1e-9) {
    parts.push("（恰在退化点）");
    warn = true;
  } else if (omegaOpt !== undefined && omega > omegaOpt) {
    parts.push("已越过 ω_opt：主特征值为复共轭对，ρ 随 ω 上升");
    warn = true;
  }
  const b = badge("critical", parts.join("　"), 8, 8);
  b.color = warn ? "#c53030" : "#2f855a";
  return b;
}

/** 面板文本（一行一个 `text` 层；渲染层不拆 `\n`）。 */
function summaryLines(m: {
  nMeta?: number;
  nModes: number;
  step: number;
  nSteps: number;
  methodName?: string;
  initName?: string;
  omega?: number;
  omegaOpt?: number;
  rhoTheory?: number;
  rhoMeasured?: number;
  slowestRate?: number;
  slowestMode?: number;
  nMeasured?: number;
  nSentinel: number;
  nNonPositive: number;
  componentFloor?: number;
  errorMark?: number;
  residualMark?: number;
  vminLog: number;
  vmaxLog: number;
  refLines: string[];
  note?: string;
}): string[] {
  // 面板文本只有 9 行：每句都压到「一行装得下」的长度，长句拆成两个短句。
  const out: string[] = [];
  const head: string[] = [];
  if (m.nMeta !== undefined) head.push(`n=${formatNumber(m.nMeta, 4)}`);
  head.push(`模态数 ${formatNumber(m.nModes, 4)}`);
  if (m.methodName !== undefined) head.push(`方法 ${m.methodName}`);
  if (m.initName !== undefined) head.push(`初始误差 ${m.initName}`);
  out.push(`${head.join("　")}　第 ${formatNumber(m.step, 4)}/${formatNumber(m.nSteps, 4)} 步`);
  out.push(
    `ρ(T) 解析 ${fmtOr(m.rhoTheory, "帧没给")}、幂法实测 ${fmtOr(m.rhoMeasured, "帧没给")}；` +
      (m.slowestRate === undefined
        ? "帧没给最慢模态因子"
        : `最慢模态 j=${formatNumber((m.slowestMode ?? 0) + 1, 3)} 的 |λ|=${formatNumber(m.slowestRate, 6)}`),
  );
  out.push(
    `实测比值可测 ${m.nMeasured === undefined ? "帧没给" : formatNumber(m.nMeasured, 4)}/${formatNumber(m.nModes, 4)}` +
      `（窗口内幅值须高于 ${formatNumber(RATE_MEASURE_FLOOR, 1)}）；测不到处帧里写 -1 哨兵（不是 0），图上不画`,
  );
  out.push(
    `色阶 log10|分量| ∈ [${formatNumber(m.vminLog, 3)}, ${formatNumber(m.vmaxLog, 3)}]，` +
      `地板 ${m.componentFloor === undefined ? "取数组最小值（帧 meta 未给）" : formatNumber(m.componentFloor, 3)}，深色是舍入噪声`,
  );
  out.push(`热力图是整段 ${formatNumber(m.nSteps, 4)} 步（所有帧相同），游标与右上曲线才是第 ${formatNumber(m.step, 4)} 步`);
  out.push(...m.refLines);
  // 当前步的 ‖e‖/‖r‖ 由状态角标给出，这里不再重复占一行。
  if (m.note !== undefined) out.push(`帧 note：${m.note}`);
  return out;
}

/** `undefined` 显示成给定文字（帧没给就不编数）。 */
function fmtOr(v: number | undefined, fallback: string): string {
  return v === undefined ? fallback : formatNumber(v, 6);
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
