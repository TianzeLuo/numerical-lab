/**
 * `pivot_strategy`（1.2 章）的 model：三种主元策略的 log10 ρ 与前后向误差。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | `log10 ρ = 0` 参考线（ρ = 1 的位置；线性轴上必须画出来，否则负值柱没有基准） |
 * | 2 | bars | `main` | `rho_log10`（无主元 / 部分主元 / 完全主元；NaN 柱子由渲染层跳过） |
 * | 3 | line | `aux` | `u/2` 参考线（帧 `meta.unit_roundoff`，后向误差的理论地板） |
 * | 4 | bars | `aux` | `error`（前向）与 `error_backward`（后向）两组柱，log 纵轴 |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 / 3 状态（奇异策略数、
 * Cholesky、帧 note）/ 4 前端预览（仅 `backend+preview`）/ 5 溢出 / 7 降级说明。
 *
 * 几条刻意的决定
 * --------------
 * 1. **柱状用 `bars`，类别名用 `text` 手写**：`scene.ts::AxisOptions` 只有 `xticks: number[]`，
 *    **没有** `xtickLabels` 字段（不要臆造），所以三类策略的名字只能按"数据 → 像素"自己放
 *    text 层。为此把 `xticks` 清空，免得数字刻度与手写标签叠在同一行。
 * 2. **log 轴上的柱子必须给正 baseline**：两个渲染器都先算 `dataToPixel(axis, x, baseline)`，
 *    `baseline = 0` 在对数轴上得到 NaN → 整根柱子被 skip（图上什么都没有、账上也没有）。
 *    所以 `baseline = auxYlim[0]`；而**值为 0 或 NaN 的柱子本来就没法在 log 轴上画**，
 *    模型只负责跳过并如实报数（NaN = 该策略失败，0 = 误差恰好为零，两者含义不同）。
 * 3. **两组柱错开半个槽位**：前向与后向误差必须同屏对比（规则 3），但同一个 x 上画两组柱
 *    会完全重叠；这里把前向放到 `x−0.2`、后向放到 `x+0.2`，宽度 0.34，柱间留缝。
 * 4. **策略名表在本地，顺序取自帧**：`strategies_x` 的顺序就是后端 `_strategy_stats` 的顺序
 *    （none / partial / complete），本地表只提供中文名（与后端 `_pivot_name` 同口径）；
 *    帧 `strings.note` 原样进状态角标，名字若与后端漂移能立刻看出来。
 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite, dataToPixel } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clipBadgeText,
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
  sceneShell,
  stackedPanels,
  stringOf,
  textWidthUnits,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  barsLayer,
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  text,
  type TextSpec,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.2";
const VIZ = "pivot_strategy";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "主元策略对比：增长因子与前向/后向误差";

/**
 * 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。
 * 长文本角标（溢出、降级）各占一行的左列，右侧留空，避免与邻居叠字。
 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  status: 3,
  /** 仅当帧声明 `backend+preview` 时出现（本章的帧都是 backend）。 */
  preview: 4,
  overflow: 5,
  degraded: 7,
} as const;

/**
 * 三种主元策略的显示名（UI 文案，与后端 `server/chapters/1.2.py::_pivot_name` 同口径）。
 * 顺序不是这里定的：柱子按帧 `strategies_x` 排（后端 `_strategy_stats` 依次跑 none/partial/complete）。
 */
const STRATEGY_LABELS: readonly string[] = ["无主元", "部分主元", "完全主元"];

export const pivotStrategyModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const strategiesX = requireArray(frame, "strategies_x");
  const rhoLog10 = requireArray(frame, "rho_log10");
  const rho = requireArray(frame, "rho");
  const error = requireArray(frame, "error");
  const errorBackward = requireArray(frame, "error_backward");
  requireSameLength(strategiesX, rhoLog10, "帧数组 strategies_x 与 rho_log10");
  requireSameLength(strategiesX, rho, "帧数组 strategies_x 与 rho");
  requireSameLength(strategiesX, error, "帧数组 strategies_x 与 error");
  requireSameLength(strategiesX, errorBackward, "帧数组 strategies_x 与 error_backward");
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const n = metaNumber(frame, "n");
  if (n === undefined) degraded.push("帧 meta 缺 n：不显示矩阵阶数（不按参数猜）");
  const matrixSize = metaNumber(frame, "matrix_size");
  const kappa1 = metaNumber(frame, "kappa_1");
  if (kappa1 === undefined) degraded.push("帧 meta 缺 kappa_1：不显示 κ₁（不重算）");
  const rhoBound = metaNumber(frame, "rho_bound");
  // 后端 2026-10-18 起把语义拆开了：`rho_max` 是真正的最大增长因子，
  // `rho_log10_max` 才是柱高用的对数；这里用的是后者（旧版只有一个错名的字段）。
  const rhoLog10Max = metaNumber(frame, "rho_log10_max");
  const rhoMax = metaNumber(frame, "rho_max");
  const forwardMax = metaNumber(frame, "forward_max");
  const backwardMax = metaNumber(frame, "backward_max");
  const nSingular = metaNumber(frame, "n_singular");
  if (nSingular === undefined) degraded.push("帧 meta 缺 n_singular：不声明「有策略失败」（不猜）");
  const choleskyOk = metaNumber(frame, "cholesky_ok");
  const unitRoundoff = metaNumber(frame, "unit_roundoff");
  if (unitRoundoff === undefined) degraded.push("帧 meta 缺 unit_roundoff：不画 u/2 参考线");
  const frameOverflow = (metaNumber(frame, "overflow") ?? 0) > 0.5;
  const matrixName = stringOf(frame, "matrix_name");
  if (matrixName === undefined) degraded.push("帧 strings 缺 matrix_name：不显示矩阵名（不按参数猜）");
  const strategyKeys = stringOf(frame, "strategy");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左列上下两栏柱状（ρ 一栏、误差一栏），右列专门放说明文字 ——
  // 为什么把文字挪到右列：ρ 的柱子可以长到 log10 ρ = 18（无主元 + 小主元），
  // 任何压在柱面上的文字都会被整片纯色吃掉；spy_ordering 那张图里也是同一个理由。
  const base = stackedPanels(width, height);
  const barsW = Math.max(200, Math.round(base.contentWidth * 0.58));
  const mainH = Math.round((base.bottom - base.top) * 0.45);
  const auxTop = base.top + mainH + 52; // 52px 留给主面板的 x 类别名（画在轴下 4px）
  const mainRect: Rect = [base.left, base.top, barsW, mainH];
  const auxRect: Rect = [base.left, auxTop, barsW, Math.max(60, base.bottom - auxTop)];
  const textX = base.left + barsW + 20;
  const textW = Math.max(160, base.left + base.contentWidth - textX);
  const mainAxis = makeAxis("main", mainRect, {
    ylim: rhoYlim(rhoLog10),
    xlabel: "",
    ylabel: axisYlabel("main", "log10 ρ"),
    title: axisTitle("main", "log10 ρ（增长因子）"),
    autoscaled: false,
    // 类别名用 text 手写（AxisOptions 没有 xtickLabels）：先清空数字刻度，免得两行叠字。
    xticks: [],
  });
  const auxAxis = makeAxis("aux", auxRect, {
    ylim: errorYlim(error, errorBackward, unitRoundoff),
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabel("aux", "相对误差"),
    title: axisTitle("aux", "前向 / 后向误差（对数）"),
    autoscaled: true,
    xticks: [],
  });
  // x 范围由帧的 strategies_x 决定（两端各留半个槽位给柱子）。
  const xr = strategyRange(strategiesX);
  mainAxis.xlim = [xr[0] - 0.6, xr[1] + 0.6];
  auxAxis.xlim = [xr[0] - 0.6, xr[1] + 0.6];
  scene.axes.push(mainAxis, auxAxis);

  // —— 图层 1~2：log10 ρ 的柱 + ρ = 1 基准线 ——
  scene.layers.push(
    lineLayer(
      "main",
      series([mainAxis.xlim[0], mainAxis.xlim[1]], [0, 0], pal.reference, {
        dash: [5, 4],
        width: 1,
        label: "$\\log_{10}\\rho=0$（$\\rho=1$：没有增长）",
      }),
    ),
  );
  let rhoNaN = 0;
  for (let i = 0; i < rhoLog10.length; i += 1) if (!Number.isFinite(rhoLog10[i])) rhoNaN += 1;
  scene.layers.push(
    barsLayer("main", strategiesX, rhoLog10, {
      width: 0.56,
      color: pal.curves[0],
      baseline: 0,
      label: "$\\log_{10}\\rho$",
    }),
  );
  // 柱高为 0 的柱子（log10 ρ = 0，即 ρ = 1）在图上是**看不见**的——它和轴底重合。
  // 所以在每个可画的策略上再点一个圆点：数值靠点读，柱高只表达量级。
  const dotX: number[] = [];
  const dotY: number[] = [];
  for (let i = 0; i < strategiesX.length; i += 1) {
    if (!Number.isFinite(strategiesX[i]) || !Number.isFinite(rhoLog10[i])) continue;
    dotX.push(strategiesX[i]);
    dotY.push(rhoLog10[i]);
  }
  if (dotX.length > 0) {
    scene.layers.push(
      scatterLayer("main", dotX, dotY, {
        r: 4,
        color: pal.marker,
        alpha: 0.95,
        label: "每个策略的 $\\log_{10}\\rho$（柱高为 0 时靠它读数值）",
      }),
    );
  }

  // —— 图层 3~4：前后向误差的两组柱（错开半个槽位）+ u/2 参考线 ——
  // 见文件头第 2 条：log 轴的 baseline 必须 > 0，否则柱子会被渲染层整根跳过。
  const barBase = auxAxis.ylim[0];
  if (unitRoundoff !== undefined && unitRoundoff > 0 && unitRoundoff >= auxAxis.ylim[0] && unitRoundoff <= auxAxis.ylim[1]) {
    scene.layers.push(
      lineLayer(
        "aux",
        series([auxAxis.xlim[0], auxAxis.xlim[1]], [unitRoundoff, unitRoundoff], pal.reference, {
          dash: [5, 4],
          width: 1,
          label: `u/2 = ${formatNumber(unitRoundoff, 3)}（单位舍入）`,
        }),
      ),
    );
  } else if (unitRoundoff !== undefined) {
    degraded.push(`u/2 = ${formatNumber(unitRoundoff, 3)} 落在误差轴范围之外：不画参考线（不缩轴藏它）`);
  }
  const shift = 0.2;
  const xForward = shiftX(strategiesX, -shift);
  const xBackward = shiftX(strategiesX, shift);
  scene.layers.push(
    barsLayer("aux", xForward, error, {
      width: 0.34,
      color: pal.curves[0],
      baseline: barBase,
      label: "前向误差 $‖x̂−x‖/‖x‖$（实测）",
    }),
  );
  scene.layers.push(
    barsLayer("aux", xBackward, errorBackward, {
      width: 0.34,
      color: pal.residual,
      baseline: barBase,
      label: "后向误差（帧 meta 给出的同一口径）",
    }),
  );
  const fwdZero = countExactZero(error);
  const bwdZero = countExactZero(errorBackward);
  const fwdNaN = countNonFinite(error);
  const bwdNaN = countNonFinite(errorBackward);

  // —— 文本（一行一个 text 层：render2d 不拆 \n）——
  const summary = paramSummary(frame, opts, [
    { name: "n", label: "$n$" },
  ]);
  degraded.push(...summary.degraded);
  const head: string[] = [summary.text];
  if (matrixName !== undefined) head.push(`矩阵=${matrixName}`);
  if (matrixSize !== undefined) head.push(`阶数=${formatNumber(matrixSize, 4)}`);
  if (kappa1 !== undefined) head.push(`κ₁=${formatNumber(kappa1, 4)}`);
  if (strategyKeys !== undefined) head.push(`策略顺序=${strategyKeys}`);
  // 右列文字宽 ~346px → 约 47 个显示单位一行。
  const textBudget = unitBudget(textW);
  let textTop = base.top + 4;
  const headSpecs = textBlock(textX, textTop, textTop, [head.join("　")], pal.muted, textBudget, "top");
  scene.texts.push(...headSpecs);
  textTop += headSpecs.length * 15 + 6;

  const mainLines: string[] = [
    `柱高 = log10 ρ：${Math.max(0, rhoLog10.length - rhoNaN)} 根可画` + (rhoNaN > 0 ? `；${rhoNaN} 根非有限（该策略直接失败）已跳过` : ""),
  ];
  if (rhoBound !== undefined) mainLines.push(`理论上界 2^(n−1) = ${formatNumber(rhoBound, 4)}（ρ 是"后向"稳定性的度量，不是解准不准）`);
  mainLines.push("ρ 大 ≠ 解不准：真正的误差还要乘条件数 κ（看副轴与 unit_ball 那一页）。");
  {
    const specs = textBlock(textX, textTop, base.bottom - 8, mainLines, pal.warn, textBudget, "top");
    scene.texts.push(...specs);
    textTop += specs.length * 15 + 6;
  }

  const auxLines: string[] = [];
  if (note !== undefined) auxLines.push(`帧 note：${note}`);
  auxLines.push("蓝=前向误差　紫=后向误差（两组柱错开半个槽位；baseline 取轴底，因为 log 轴画不出 0）");
  const skipped: string[] = [];
  if (fwdNaN + bwdNaN > 0) skipped.push(`${fwdNaN + bwdNaN} 根 NaN（策略失败）`);
  if (fwdZero + bwdZero > 0) skipped.push(`前向 ${fwdZero} 根、后向 ${bwdZero} 根恰好为 0（对数轴画不出，含义与 NaN 不同）`);
  if (skipped.length > 0) auxLines.push(`跳过：${skipped.join("；")}`);
  if (forwardMax !== undefined || backwardMax !== undefined) {
    auxLines.push(
      `本帧最大：前向 ${formatNumber(forwardMax ?? Number.NaN, 3)}、后向 ${formatNumber(backwardMax ?? Number.NaN, 3)}` +
        (rhoLog10Max === undefined ? "" : `；max log10 ρ = ${formatNumber(rhoLog10Max, 4)}`) +
        (rhoMax === undefined ? "" : `（ρ_max = ${formatNumber(rhoMax, 3)}）`),
    );
  }
  if (choleskyOk !== undefined) {
    auxLines.push(choleskyOk === 1 ? "Cholesky 可行（对称正定）：无主元也稳定，ρ ≤ 1。" : "Cholesky 不可行：该矩阵不是对称正定（帧 meta.cholesky_ok）。");
  }
  scene.texts.push(...textBlock(textX, textTop, base.bottom - 8, auxLines, pal.muted, textBudget, "top"));

  // 两个面板的 x 类别名（手写；`AxisOptions` 没有 xtickLabels 字段，见文件头第 1 条）。
  // 写在轴下 4px：所以主面板与副面板之间留了 52px 的净空（见排版）。
  for (const axis of [mainAxis, auxAxis]) {
    for (let i = 0; i < strategiesX.length; i += 1) {
      const [px] = dataToPixel(axis, strategiesX[i], axis.ylim[0]);
      if (!Number.isFinite(px)) continue;
      scene.texts.push(
        text(px, axis.rect[1] + axis.rect[3] + 4, STRATEGY_LABELS[i] ?? `#${i}`, {
          role: "axis",
          size: 11,
          color: pal.muted,
          align: "center",
        }),
      );
    }
  }

  // —— 角标（固定槽位）——
  // 规则 5：恒挂 source；预览型数据源另挂 preview（kind 各表一意，见 scene.ts）。
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "误差轴：自适应（log 纵轴按本帧误差取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  const status: string[] = [];
  if (nSingular !== undefined && nSingular > 0) {
    status.push(`⚠ ${formatNumber(nSingular, 4)} 种策略直接失败（零主元）：对应柱子留空`);
  }
  if (frameOverflow) status.push("⚠ 帧 meta.overflow=true：ρ 里出现了非有限值（柱子被跳过，不是 0）");
  if (note !== undefined) status.push(note);
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 meta.n_singular 与 strings.note）");
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  // 规则 1：图层越界 + 帧内非有限值，两者都要进 overflow 与角标。
  const nBad =
    countNonFinite(strategiesX) + countNonFinite(rhoLog10) + countNonFinite(rho) + countNonFinite(error) + countNonFinite(errorBackward);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/**
 * 一段说明文字 → 若干 `text` 层，**一行一个**（`render2d.ts::drawText` 不拆 `\n`）。
 *
 * 为什么要自己折行：两个渲染器都支持 `maxWidth` 自动折行，但折出来的续行**不占**模型这一侧的
 * 行距预算（光栅器的续行间距是 `cell_h + 4`，比这里排的 15px 大），续行会压到下一行文字上。
 * 所以模型先按"显示宽度单位"折到 `budget` 以内（`model.ts::textWidthUnits`：CJK=2、ASCII=1），
 * 再逐行排开：一行就是一层，两个渲染器都不会再折。
 *
 * `anchor`：`"top"` 从 `top` 往下排；`"bottom"` 让整块贴着 `bottom`。返回的数组长度 = 实际行数。
 */
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
  let pieces = lines.flatMap((body) => wrapUnits(body, budget));
  // 画布小的时候宁可**截断并说明**，也不让文字叠在一起（叠字等于看不见，见规则 1 的精神）：
  // 能放下的行数由 top/bottom 之间的净空决定，最后一行留给"还有 N 行"。
  if (bottom > top) {
    const maxRows = Math.max(1, Math.floor((bottom - top) / rowH) + 1);
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - (pieces.length - 1) * rowH);
  return pieces.map((body, i) => text(x, y + i * rowH, body, { role: "note", size, color, maxWidth: budget * 7 }));
}

/**
 * 按显示宽度单位折行（同一把尺子：`model.ts::textWidthUnits`，CJK=2、ASCII=1）。
 *
 * 先按空白（含全角空格）切"词"再往行里塞：`ρ=0.9991`、`k=3/63` 这类字段不会被从中间劈开
 * （劈开在光栅器里读起来像两个数）。单个词比预算还长时才硬切。
 */
function wrapUnits(body: string, budget: number): string[] {
  const out: string[] = [];
  let line = "";
  let used = 0;
  const push = (): void => {
    if (line !== "") out.push(line);
    line = "";
    used = 0;
  };
  for (const word of body.split(/(?<=[\s　])/)) {
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

/**
 * 面板宽度 → 显示宽度单位预算（size 11 时 1 单位 ≈ 7px 光栅单元）。
 * 留 12px 边距再折算，保证两个渲染器都不会自动折行（见 `textBlock` 的注释）。
 */
function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/** log10 ρ 的线性轴范围：必须含 0（ρ = 1 的基准线与负柱都靠它）。 */
function rhoYlim(values: ArrayLike<number>): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-1, 1];
  return niceLimits(Math.min(0, lo) - 0.5, Math.max(1, hi + 0.5), 6);
}

/** 误差轴的 log 范围：前向、后向、以及 u/2 参考线一起纳入（正有限值）。 */
function errorYlim(fwd: ArrayLike<number>, bwd: ArrayLike<number>, u: number | undefined): [number, number] {
  const a = finitePositiveMinMax(fwd);
  const b = finitePositiveMinMax(bwd);
  const lo = Math.min(a?.[0] ?? Number.POSITIVE_INFINITY, b?.[0] ?? Number.POSITIVE_INFINITY, u !== undefined && u > 0 ? u : Number.POSITIVE_INFINITY);
  const hi = Math.max(a?.[1] ?? Number.NEGATIVE_INFINITY, b?.[1] ?? Number.NEGATIVE_INFINITY, u !== undefined && u > 0 ? u : Number.NEGATIVE_INFINITY);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [1e-18, 1];
  return niceLimits(lo, hi, 7, { log: true });
}

/** 策略横坐标的 `[min, max]`。 */
function strategyRange(values: ArrayLike<number>): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : [0, 2];
}

/** 把横坐标整体平移（两组柱错开用；纯显示几何）。 */
function shiftX(values: ArrayLike<number>, dx: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    out.push(Number.isFinite(v) ? v + dx : Number.NaN);
  }
  return out;
}

/** 恰好为 0 的个数（log 轴上画不出，但与 NaN 含义不同）。 */
function countExactZero(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if (a[i] === 0) n += 1;
  return n;
}

/** manifest 里该轴的标题（文案直接取 manifest，缺省才用回落值）。 */
function axisTitle(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel。 */
function axisYlabel(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.ylabel ?? fallback;
}
