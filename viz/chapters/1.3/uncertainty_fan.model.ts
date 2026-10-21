/**
 * `uncertainty_fan`（1.3 章）的 model：等残差谷底上的不确定扇面 + 不确定度随 κ 的增长。
 *
 * 面板与图层
 * ----------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `plane` | 等残差椭圆族（`level_x`/`level_y` 逐行，越界段**断开**） |
 * | 2 | line | `plane` | 噪声扇面（`fan_x`/`fan_y` 折线，随 t 逐步张开） |
 * | 3 | line | `plane` | 当前方向：从谷底中心指向当前解 |
 * | 4 | scatter | `plane` | 当前方向的解（`sol_now`）与谷底中心（`center`） |
 * | 5 | loglogfit | `growth` | 实测最大不确定度 vs κ（实测斜率 + κ·δ 理论参考线，规则 2） |
 * | 6 | line | `growth` | κ·δ 参考线（帧数组 `theory_kappa`） |
 * | 7 | scatter | `growth` | 当前 log10 κ 处的放大倍数 |
 *
 * 四条刻意的决定
 * --------------
 * 1. **主轴固定 + 等比例**：manifest 声明 `plane` 是 `autoscaled: false`，注释写的是
 *    "固定等比例坐标"。等比例不是装饰——把两个坐标轴各自缩放，κ=1e6 的**针**会被画成**圆**，
 *    "病态"这个结论当场反转（pitfalls 第 4 条）。这里用同一个半径算两个方向的范围，
 *    并在文本里写出窗口半径与"椭圆的长短轴之比 = κ"的实测值。
 * 2. **窗口是"看得见谷底"的那个窗口，不是"装得下整条椭圆"的那个**：最长的一条等残差椭圆
 *    延伸到 ±3.5e3（κ=1e6、ρ/‖b‖=1e-2），装下它的话扇面就只有几个像素。所以窗口取
 *    `center ± 3×扇面最大偏移`，被窗口切断的椭圆在 overflow 里如实计数——
 *    **看不到不等于不存在**（规则 1）。
 * 3. **守恒的那个量画不出来，就说出来**：所有方向的相对残差差异只有 1.7e-10（帧的
 *    `meta.residual_spread`），在对数轴上是一条死平的线。与其画一条看不出信息的线，
 *    不如把 `residual_spread`/`residual_roundoff`/`residual_flatness` 写成数字，
 *    并说明"残差完全相同"这件事在浮点里只能稳到 κu‖x‖。
 * 4. **`resolvable=false` 时要拦一句**：当 κu 已经大于噪声水平，扇面量到的是**舍入**而不是
 *    噪声（pitfalls 第 7 条）。帧的 `meta.resolvable` 就是这条判据，模型在角标与文本里
 *    都把它点出来——这种时候这张图不该被用来谈"噪声放大"。
 */

import { niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  canvasSize,
  clamp,
  clipBadgeText,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finitePositiveMinMax,
  metaNumber,
  palette,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  resolveTextOverlaps,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  stackedPanels,
  stringOf,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  clipBadge,
  criticalBadge,
  lineLayer,
  loglogFitLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  type LineSeries,
  pitfallsBand,
} from "../../shared/src/scene.js";
import { contentBottom, fmt, fmtOr, joinParts, kv, overflowBadgeFor, textBlock, textFloor, unitBudget } from "./_shared.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.3";
const VIZ = "uncertainty_fan";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "解的不确定扇面：等残差的谷底有多长";

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  status: 4,
  preview: 5,
  overflow: 6,
  degraded: 7,
} as const;

const KAPPA_CRITICAL = 15.95;

/** 窗口半径 = 这个倍数 × 扇面的最大偏移（见文件头第 2 条）。 */
/**
 * 窗口 = `WINDOW_FACTOR × 扇面最大偏移`（两个方向同一个半径，等比例不许打折）。
 *
 * 为什么不是 3：等比例窗口的用途是"看清扇面"，而 κ=1e6 时扇面只占窗口的几分之一，
 * 面板上就只剩一个点（实测：3 倍窗口下扇面宽度不到面板的 1/6，图基本是空的）。
 * 1.15 让扇面顶到面板边缘，同时**不裁掉任何一条实测方向**（偏移最大值就是 r1）。
 * 代价照旧用文字与角标承担：等残差椭圆族在这个窗口里几乎全部落在窗外，
 * 逐条标"超出窗口，已切断"并给出它的 ρ/‖b‖（见下面的椭圆图层）。
 */
const WINDOW_FACTOR = 1.15;

export const uncertaintyFanModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段 ——
  const levelX = requireArray(frame, "level_x");
  const levelY = requireArray(frame, "level_y");
  const fanX = requireArray(frame, "fan_x");
  const fanY = requireArray(frame, "fan_y");
  const error = requireArray(frame, "error");
  const residual = requireArray(frame, "residual");
  const thetaDeg = requireArray(frame, "theta_deg");
  const solNow = requireArray(frame, "sol_now");
  const center = requireArray(frame, "center");
  const v1 = requireArray(frame, "v1_dir");
  const v2 = requireArray(frame, "v2_dir");
  const sigma = requireArray(frame, "sigma");
  const semiAxes = requireArray(frame, "semi_axes");
  const kappaGrid = requireArray(frame, "kappa_grid");
  const errorKappa = requireArray(frame, "error_kappa");
  const theoryKappa = requireArray(frame, "theory_kappa");
  const levelRho = requireArray(frame, "level_rho");
  const pitfalls = requirePitfalls(frame);
  requireSameLength(fanX, fanY, "帧数组 fan_x 与 fan_y");
  requireSameLength(fanX, error, "帧数组 fan_x 与 error");
  requireSameLength(fanX, residual, "帧数组 fan_x 与 residual");
  requireSameLength(fanX, thetaDeg, "帧数组 fan_x 与 theta_deg");
  requireSameLength(kappaGrid, errorKappa, "帧数组 kappa_grid 与 error_kappa");
  requireSameLength(kappaGrid, theoryKappa, "帧数组 kappa_grid 与 theory_kappa");
  if (solNow.length !== 2 || center.length !== 2 || v1.length !== 2 || v2.length !== 2) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的二维向量长度必须为 2：sol_now=${solNow.length}、center=${center.length}、` +
        `v1_dir=${v1.length}、v2_dir=${v2.length}`,
    );
  }
  const nFan = fanX.length;
  const nLevel = levelRho.length;
  if (levelX.length % nLevel !== 0 || levelY.length !== levelX.length) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 level_x/level_y 无法按 ${nLevel} 条曲线等分：` +
        `len(level_x)=${levelX.length}、len(level_y)=${levelY.length}`,
    );
  }
  const perLevel = levelX.length / nLevel;

  // —— 可选字段 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const kappaTarget = metaNumber(frame, "kappa_target");
  const kappaMeasured = metaNumber(frame, "kappa_measured");
  const sigmaMin = metaNumber(frame, "sigma_min");
  const sigmaMax = metaNumber(frame, "sigma_max");
  const noiseRel = metaNumber(frame, "noise_rel");
  const rhoRel = metaNumber(frame, "rho_rel");
  const ampMax = metaNumber(frame, "amplification_max");
  const ampMin = metaNumber(frame, "amplification_min");
  const ampTheory = metaNumber(frame, "amplification_theory");
  const residualMax = metaNumber(frame, "residual_max");
  const residualMin = metaNumber(frame, "residual_min");
  const residualSpread = metaNumber(frame, "residual_spread");
  const residualRoundoff = metaNumber(frame, "residual_roundoff");
  const residualFlatness = metaNumber(frame, "residual_flatness");
  const thetaCurrentMeta = metaNumber(frame, "theta_current");
  const nSamplesMeta = metaNumber(frame, "n_samples");
  const nLevelsMeta = metaNumber(frame, "n_levels");
  const roundoffRel = metaNumber(frame, "roundoff_rel");
  const resolvableMeta = metaNumber(frame, "resolvable");
  const resolvable = resolvableMeta === undefined ? undefined : resolvableMeta > 0.5;
  if (resolvable === undefined) degraded.push("帧 meta 缺 resolvable：不判断「扇面量的是噪声还是舍入」（不猜）");
  const nVisibleMeta = metaNumber(frame, "n_visible");
  const semiRatio = metaNumber(frame, "semi_axis_ratio");
  const errorFloor = metaNumber(frame, "error_floor");
  const xNorm = metaNumber(frame, "x_norm");
  const fanWidthDeg = metaNumber(frame, "fan_width_deg");
  const rankNote = stringOf(frame, "rank_note");
  if (rankNote === undefined) degraded.push("帧 strings 缺 rank_note：不显示数值秩");
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

  // —— 排版：左列正方形解平面（固定等比例）+ 右列增长曲线 + 两块文字 ——
  // 像素账（`base.bottom = 498`，下面 102px 是 pitfalls 面板，谁也不能压）：
  //   右列：文字 114..204（7 行）| 标题带 222..238 | 增长轴 240..450 | 轴名带 466..478
  //   左列：解平面 122..402（280×280 正方形）| 文字 404..498（6 行）
  // 解平面**不设轴名**（那条带会让正方形挤不下）：轴的含义写在文字里。
  const base = stackedPanels(width, height);
  // 底部面板（渲染层现画的 `scene.pitfalls`）会随文案长短变高：把内容底边一起收上来，
  // 否则副轴与文本的下半截会被它盖住（实测 1.3 的 fan 帧面板顶边只有 442，而 base.bottom=498）。
  base.bottom = contentBottom(frame, width, height, base.bottom);
  const columnGap = 74; // 右列自己的 y 刻度 + ylabel
  const side = Math.max(180, base.bottom - base.top); // 正方形顶满内容高度（等比例不许打折）
  const planeRect: Rect = [base.left, base.top, side, side];
  const growthX = base.left + side + columnGap;
  const growthW = Math.max(160, base.left + base.contentWidth - growthX);
  // 增长轴压到 110px：底部 pitfalls 面板变高（实测顶边 442）以后，只有把轴压扁
  // 才腾得出右列下方 4 行数值块的位置（`bottom = 442−4`，每行 15px）。
  const growthRect: Rect = [growthX, base.top + 104, growthW, Math.max(80, Math.min(110, base.bottom - base.top - 190))];

  // —— 当前方向（时间轴 = 噪声方向角）——
  // 与后端 `_uncertainty_fan` 同口径：先按 |θ−t| 找最近的采样方向（帧里的实测解），
  // 再用 `meta.n_visible` 决定扇面画到第几条——**不插值**（插值出来的解不是帧里的解）。
  const thetaNowRaw = ((tNow % 360) + 360) % 360;
  let idxNow = 0;
  let best = Number.POSITIVE_INFINITY;
  for (let i = 0; i < nFan; i += 1) {
    const d = Math.abs((thetaDeg[i] as number) - thetaNowRaw);
    const dd = Math.min(d, 360 - d);
    if (dd < best) {
      best = dd;
      idxNow = i;
    }
  }
  if (best > 1e-9 && nFan > 1) {
    degraded.push(`请求角度 θ=${fmt(thetaNowRaw, 4)}° 落在两个采样方向之间：按最近方向 θ=${fmt(thetaDeg[idxNow] as number, 4)}° 画（不插值）`);
  }
  const nVisible = clamp(Math.round(nVisibleMeta ?? idxNow + 1), 1, nFan);
  if (nVisibleMeta === undefined) degraded.push("帧 meta 缺 n_visible：扇面按当前方向的下标画（与后端口径可能差一条）");

  // —— 窗口：center ± WINDOW_FACTOR × 扇面最大偏移（两个方向同一个半径 → 等比例）——
  const cx = center[0] as number;
  const cy = center[1] as number;
  let r1 = 0;
  for (let i = 0; i < nFan; i += 1) {
    r1 = Math.max(r1, Math.abs((fanX[i] as number) - cx), Math.abs((fanY[i] as number) - cy));
  }
  if (!(r1 > 0)) {
    r1 = Math.max(1e-12, Math.abs(cx) > 0 ? Math.abs(cx) * 1e-3 : 1e-3);
    degraded.push("扇面所有方向都落在中心上（偏移恒为 0）：窗口退化成 ±1e-3 的占位范围");
  }
  const radius = WINDOW_FACTOR * r1;
  const planeLim: [number, number] = [cx - radius, cx + radius];
  const planeYlim: [number, number] = [cy - radius, cy + radius];
  const planeAxis = makeAxis("plane", planeRect, {
    xlim: planeLim,
    ylim: planeYlim,
    // **不设轴名**：轴名带会在正方形下方占 16..28px，而这块正方形要顶到内容底边；
    // 轴的含义写在右下角的文字里（"横 = x₁、纵 = x₂，等比例、固定窗口"）。
    xlabel: "",
    ylabel: axisYlabel("plane", "$x_2$"),
    title: axisTitle("plane", "解平面（固定等比例坐标，等残差椭圆族）"),
    autoscaled: false,
    xticks: [],
    yticks: [],
    grid: false,
  });
  const growthPos = finitePositiveMinMax(errorKappa);
  const growthYlim: [number, number] =
    growthPos !== undefined ? niceLimits(growthPos[0], growthPos[1], 6, { log: true, pad: 0.08 }) : [1e-18, 1e4];
  if (growthPos === undefined) degraded.push("error_kappa 没有 >0 的有限值：增长轴用占位范围");
  const growthAxis = makeAxis("growth", growthRect, {
    xlim: niceLimits(kappaGrid[0] as number, kappaGrid[kappaGrid.length - 1] as number, 7, { log: true, pad: 0.04 }),
    ylim: growthYlim,
    xscale: "log",
    yscale: "log",
    xlabel: "$\\kappa$（对数）",
    ylabel: axisYlabel("growth", "相对不确定度"),
    title: axisTitle("growth", "实测不确定度 vs $\\kappa$（双对数）与 $\\kappa\\cdot\\delta$ 参考线"),
    autoscaled: true,
  });
  scene.axes.push(planeAxis, growthAxis);

  // —— 图层 1：等残差椭圆族（逐行取，越界处在窗口边界切断）——
  const win: ClipWindow = { u0: planeLim[0], u1: planeLim[1], v0: planeYlim[0], v1: planeYlim[1] };
  const ellSeries: LineSeries[] = [];
  let nLevelClipped = 0;
  for (let li = 0; li < nLevel; li += 1) {
    const path: [number, number][] = [];
    for (let i = 0; i < perLevel; i += 1) {
      const k = li * perLevel + i;
      path.push([levelX[k] as number, levelY[k] as number]);
    }
    const pieces = clipPolyline(path, win);
    if (pieces.clipped > 0) nLevelClipped += 1;
    for (const piece of pieces.parts) {
      if (piece.length < 2) continue;
      ellSeries.push(
        series(
          Float64Array.from(piece, (q) => q[0]),
          Float64Array.from(piece, (q) => q[1]),
          pal.reference,
          {
            width: 1.1,
            alpha: 0.85,
            // 椭圆越靠外 ρ 越大；图例里给出实测的 ρ/‖b‖，不靠颜色深浅去猜。
            label:
              li === 0 || li === nLevel - 1
                ? `等残差椭圆 ρ/‖b‖=${fmt(levelRho[li] as number, 2)}${pieces.clipped > 0 ? "（超出窗口，已切断）" : ""}`
                : undefined,
          },
        ),
      );
    }
  }
  if (ellSeries.length > 0) {
    scene.layers.push(lineLayer("plane", ellSeries));
  } else {
    degraded.push("等残差椭圆没有一段落在窗口内：只画扇面与中心");
  }

  // —— 图层 2：噪声扇面（画到第 nVisible 条）——
  const fanXs: number[] = [];
  const fanYs: number[] = [];
  let nFanClipped = 0;
  for (let i = 0; i < nVisible; i += 1) {
    const px = fanX[i] as number;
    const py = fanY[i] as number;
    if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
    if (px < win.u0 || px > win.u1 || py < win.v0 || py > win.v1) nFanClipped += 1;
    fanXs.push(px);
    fanYs.push(py);
  }
  if (fanXs.length >= 1) {
    scene.layers.push(
      lineLayer("plane", [
        series(Float64Array.from(fanXs), Float64Array.from(fanYs), pal.curves[0], {
          width: 1.8,
          marker: "o",
          label: `噪声扇面：各方向的解（已张开 ${nVisible}/${nFan} 条；每条都有同一个残差）`,
        }),
      ]),
    );
  } else {
    degraded.push("扇面在当前方向上没有可画的点");
  }

  // —— 图层 3：当前方向（中心 → 当前解）——
  const sx = solNow[0] as number;
  const sy = solNow[1] as number;
  scene.layers.push(
    lineLayer("plane", [
      series([cx, sx], [cy, sy], pal.marker, {
        width: 2.2,
        label: `当前方向 θ=${fmt(thetaDeg[idxNow] as number, 4)}°：δb 落在 u_min 附近时解被放大最多`,
      }),
    ]),
  );
  scene.layers.push(
    scatterLayer("plane", [sx], [sy], { r: 4, color: pal.marker, label: "当前方向的解 $\\hat x$" }),
  );
  scene.layers.push(
    scatterLayer("plane", [cx], [cy], { r: 3.4, color: pal.ink, label: "谷底中心 $x^*$（帧数组 center）" }),
  );

  // —— 图层 5~6：增长曲线 ——
  const slopeMeasured = powerLawSlope(kappaGrid, errorKappa);
  if (kappaGrid.length >= 2) {
    scene.layers.push(
      loglogFitLayer("growth", kappaGrid, errorKappa, slopeMeasured, 1, {
        seriesLabel: `实测最大不确定度（幂律拟合斜率 ${fmt(slopeMeasured, 3)}；理论 1）`,
        theoryLabel: "理论参考斜率 1（$\\kappa\\cdot\\delta$：扰动界取等号的方向）",
        color: pal.curves[0],
        theoryColor: pal.reference,
      }),
    );
  } else {
    degraded.push("kappa_grid 少于 2 个点：增长轴画不出幂律参考线（规则 2 要求两条斜率都在）");
  }
  const theoryX: number[] = [];
  const theoryY: number[] = [];
  for (let i = 0; i < kappaGrid.length; i += 1) {
    const x = kappaGrid[i] as number;
    const y = theoryKappa[i] as number;
    if (Number.isFinite(x) && x > 0 && Number.isFinite(y) && y > 0) {
      theoryX.push(x);
      theoryY.push(y);
    }
  }
  if (theoryX.length >= 2) {
    scene.layers.push(
      lineLayer("growth", [
        series(Float64Array.from(theoryX), Float64Array.from(theoryY), pal.exact, {
          width: 1,
          dash: [5, 4],
          label: "$\\kappa\\cdot\\delta$ 参考线（帧数组 theory_kappa，不是前端重算的）",
        }),
      ]),
    );
  }
  // 当前位置：κ_target 在网格上的最近点（帧里的实测值，不插值）
  if (kappaTarget !== undefined && kappaTarget > 0) {
    let gi = 0;
    let gBest = Number.POSITIVE_INFINITY;
    for (let i = 0; i < kappaGrid.length; i += 1) {
      const d = Math.abs(Math.log10((kappaGrid[i] as number) / kappaTarget));
      if (d < gBest) {
        gBest = d;
        gi = i;
      }
    }
    const gx = kappaGrid[gi] as number;
    const gy = errorKappa[gi] as number;
    if (Number.isFinite(gx) && gx > 0 && Number.isFinite(gy) && gy > 0) {
      scene.layers.push(
        scatterLayer("growth", [gx], [gy], { r: 3.6, color: pal.marker, label: `当前 κ=${fmt(gx, 3)} 处的实测最大不确定度` }),
      );
      if (Math.abs(Math.log10(gx / kappaTarget)) > 0.02) {
        degraded.push(
          `当前 κ=${fmt(kappaTarget, 3)} 不在帧的 κ 网格上：标记画在最近的实测点 κ=${fmt(gx, 3)}（不插值）`,
        );
      }
    }
  } else if (kappaTarget !== undefined) {
    degraded.push("帧 meta.kappa_target ≤ 0：增长轴不标当前位置（对数轴上没有负的 κ）");
  }

  // —— 文本 ——
  // 两块文字，各写各的（数值与说明都要落在图里，但**不许重算物理量**）：
  //   右列增长轴**上方** 7 行 = 参数与"这一帧在讲什么"；
  //   右列增长轴**下方** 6 行 = 读图要点。
  // 每行都按宽度预算（`unitBudget(growthW)`）手写成"一行放得下"的短句：
  // `textBlock` 的 `maxWidth` 与校验器同一把尺子，句子一超宽就会被折成两行、
  // 把整块文字顶出空档（实测会被规则 8 打成"文本压住坐标轴装饰"或被截断成"另有 N 行"）。
  // 帧的 `rank_note` / `note` 这类"随帧变长短"的字符串**不进这两块**，
  // 它们去状态角标（角标是摘要，超宽由 `clipBadgeText` 截断并留 `…`）。
  const headX = growthX;
  const headBudget = unitBudget(growthW);
  const numsBudget = headBudget;
  const head: string[] = [];
  head.push(
    joinParts([
      kv("log10 κ", kappaTarget !== undefined ? Math.log10(Math.max(kappaTarget, Number.MIN_VALUE)) : undefined, 2),
      kv("实测 κ", kappaMeasured, 2),
      kv("σmax", sigmaMax, 2),
      kv("σmin", sigmaMin, 2),
    ]) ?? "帧 meta 缺 κ/谱信息：这一行不显示",
  );
  head.push(`扇面 = 噪声球经 A⁻¹ 的像（实测椭圆，非理论估计）。`);
  head.push(`窗口 ±${fmt(radius, 3)} = ${WINDOW_FACTOR}× 最大扇面偏移（两方向同一半径）。`);
  head.push(
    joinParts([
      kv("半轴比（帧 meta）", semiRatio, 2),
      kv("ρ/‖b‖", rhoRel, 2),
    ]) ?? "帧 meta 缺椭圆信息：这一行不显示",
  );
  // θ 那一行**不在这里**重复：它在下面 `nums` 里已经写了（含 ‖δx‖/‖x*‖）。
  // 一行省下来，`resolvable=false` 的"不可分辨"警告才不会被 textBlock 截掉——
  // 那句话是这张图最重要的一句（量到的是舍入，不是噪声）。
  if (resolvable !== undefined) {
    head.push(
      resolvable
        ? `可分辨：κu = ${fmtOr(roundoffRel, 2)} ≪ δ = ${fmtOr(noiseRel, 2)} → 量到的是**噪声**。`
        : `⚠ **不可分辨**：κu ≥ δ → 量到的是**舍入**，不是噪声。`,
    );
  }
  scene.texts.push(...textBlock(headX, base.top - 8, growthRect[1] - 36, head, pal.muted, headBudget, "top"));

  const numsX = growthX;
  const nums: string[] = [];
  nums.push(`当前方向 θ=${fmt(thetaDeg[idxNow] as number, 3)}°：‖δx‖/‖x*‖ = ${fmt(error[idxNow] as number, 3)}。`);
  nums.push(
    `放大实测 ∈ [${fmtOr(ampMin, 2)}, ${fmtOr(ampMax, 2)}] ≤ 上界 κ·δ = ${fmtOr(ampTheory, 2)}。`,
  );
  // 四行（原来五行）：残差那两行合并——空档只放得下 4 行，多一行就会被截成"另有 N 行"。
  nums.push(
    `**所有方向残差相同**（帧 meta）；起伏 ${fmtOr(residualSpread, 2)}。`,
  );
  nums.push(
    `浮点里这条「守恒」只稳到 κu‖x‖ ≈ ${fmtOr(residualRoundoff, 2)}；地板 ${fmtOr(errorFloor, 1)}。`,
  );
// 采样计数、扇面张角、谷底范数这些"读图辅助量"进状态角标（下面 `statusBits`），
  // 数值块只留**结论性**的四行——空档只放得下 4 行（`maxRows = floor((bottom−top)/15)`）。
  // 数值块落在**右列增长轴下方**：宽度与增长轴同宽（预算宽），且不与任何装饰带相交
  // （增长轴轴名带止于 bottom+28）。左列正方形要顶满内容高度，下方没有空位。
  scene.texts.push(
    ...textBlock(numsX, growthRect[1] + growthRect[3] + 30, textFloor(scene, base.bottom - 4), nums, pal.muted, numsBudget, "top"),
  );



  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "增长轴：自适应（log 纵轴按本帧实测取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  if (kappaTarget !== undefined) {
    const lg = Math.log10(Math.max(kappaTarget, Number.MIN_VALUE));
    scene.badges.push(placeBadge(criticalBadge("log10 κ", lg, criticalOf(frame, "log10_kappa", KAPPA_CRITICAL)), SLOT.critical, width));
  }
  const statusBits: string[] = [
    `θ=${fmt(thetaDeg[idxNow] as number, 4)}°（帧 meta.theta_current=${fmtOr(thetaCurrentMeta, 4)}°；` +
      `第 ${idxNow + 1}/${nFan} 个采样方向）`,
    `实测放大 ${fmtOr(ampMax, 3)} / 上界 ${fmtOr(ampTheory, 3)}`,
  ];
  if (resolvable === false) statusBits.push("κu ≥ δ：扇面量的是舍入");
  // 帧的 rank_note / note 长度随帧变，放不进那两块固定行数的文字——它们在这里作摘要
  //（超宽由 `clipBadgeText` 截断并留 `…`，不会静默丢掉整条）。
  if (nLevelsMeta !== undefined) statusBits.push(`等残差条数 ${fmt(nLevelsMeta, 0)}`);
  // 数值块只留得下 4 行：残差的极值与平坦度挪到这里（角标是紧凑摘要，超宽会截断但不会丢整块）
  if (residualMin !== undefined && residualMax !== undefined) {
    statusBits.push(`所有方向残差 ∈[${fmt(residualMin, 2)}, ${fmt(residualMax, 2)}]`);
  }
  if (residualFlatness !== undefined) statusBits.push(`残差平坦度 ${fmt(residualFlatness, 2)}`);
  statusBits.push(
    `方向数 ${nSamplesMeta === undefined ? "?" : fmt(nSamplesMeta, 0)}（已画 ${nVisible}）、张角 ≈ ` +
      `${fanWidthDeg === undefined ? "?" : fmt(fanWidthDeg, 3)}°、谷底 ‖x*‖ = ${fmtOr(xNorm, 3)}`,
  );
  statusBits.push("地板 1e-18：0 与「没有定义」同形");
  if (rankNote !== undefined) statusBits.push(`rank_note：${rankNote}`);
  if (note !== undefined) statusBits.push(`note：${note}`);
  scene.badges.push(
    placeBadge(badge("status", clipBadgeText(statusBits.join("；")), 8, 8), SLOT.status, width),
  );
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(levelX) +
    countNonFinite(levelY) +
    countNonFinite(fanX) +
    countNonFinite(fanY) +
    countNonFinite(error) +
    countNonFinite(residual) +
    countNonFinite(thetaDeg) +
    countNonFinite(solNow) +
    countNonFinite(center) +
    countNonFinite(v1) +
    countNonFinite(v2) +
    countNonFinite(sigma) +
    countNonFinite(semiAxes) +
    countNonFinite(kappaGrid) +
    countNonFinite(errorKappa) +
    countNonFinite(theoryKappa) +
    countNonFinite(levelRho);
  const clipNotes: string[] = [];
  if (nLevelClipped > 0) {
    clipNotes.push(
      `${nLevelClipped}/${nLevel} 条等残差椭圆被窗口切断（最长的一条延伸到 ±${fmt(maxAbs(levelX, levelY), 3)}，` +
        `窗口只有 center ± ${fmt(radius, 3)}）：切断的段没有画出来，但椭圆本身还在`,
    );
  }
  if (nFanClipped > 0) {
    clipNotes.push(`${nFanClipped} 个扇面解落在窗口外（按数据保留，图上不可见）`);
  }
  // 固定窗口裁剪是这页的设计（不裁就看不见谷底），不是数据坏了。用中性的「视窗裁剪」，
  // 把红色「溢出」留给真正的非有限值/意外越界。
  overflowBadgeFor(scene, nBad, [], { slot: SLOT.overflow, width });
  if (!scene.overflow.flag) {
    if (clipNotes.length > 0) {
      const node = placeBadge(clipBadge(clipNotes.join("；")), SLOT.overflow, width);
      scene.badges.push(node);
    }
  }
  // 规则 8 的安全网（两侧校验器都查）：
  //  - 文本层之间不许互相重叠（帧里的说明一变长就会多折一行压住下一块）；
  //  - 文本不许被底部 pitfalls 面板压住（面板高度随帧里 pitfalls 的折行数变，模型事先看不见）。
  // 只丢不移：下移会推进轴的装饰带，那是另一条违规。丢弃的行数会写在画布顶部那一行里。
  const band = pitfallsBand(scene);
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, band === null ? height - 8 : band[1] - 4]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function axisTitle(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.title ?? fallback;
}

function axisYlabel(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.ylabel ?? fallback;
}

/** 两条坐标数组里绝对值最大的那个（只用于说明"椭圆延伸到哪里"，不参与绘图）。 */
function maxAbs(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
    const x = a[i] as number;
    const y = b[i] as number;
    if (Number.isFinite(x)) m = Math.max(m, Math.abs(x));
    if (Number.isFinite(y)) m = Math.max(m, Math.abs(y));
  }
  return m;
}

interface ClipWindow {
  u0: number;
  u1: number;
  v0: number;
  v1: number;
}

function inside(p: readonly [number, number], w: ClipWindow): boolean {
  return p[0] >= w.u0 && p[0] <= w.u1 && p[1] >= w.v0 && p[1] <= w.v1;
}

/** 线段与矩形窗口求交（Liang–Barsky）。 */
function clipParam(a: readonly [number, number], b: readonly [number, number], w: ClipWindow): [number, number] | undefined {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  let t0 = 0;
  let t1 = 1;
  for (const [p, q] of [
    [-dx, a[0] - w.u0],
    [dx, w.u1 - a[0]],
    [-dy, a[1] - w.v0],
    [dy, w.v1 - a[1]],
  ] as [number, number][]) {
    if (p === 0) {
      if (q < 0) return undefined;
      continue;
    }
    const r = q / p;
    if (p < 0) {
      if (r > t1) return undefined;
      if (r > t0) t0 = r;
    } else {
      if (r < t0) return undefined;
      if (r < t1) t1 = r;
    }
  }
  return [t0, t1];
}

/** 折线逐段裁剪；`clipped` 统计被裁过的线段数（用于说明"椭圆被切断"）。 */
function clipPolyline(path: readonly [number, number][], w: ClipWindow): { parts: [number, number][][]; clipped: number } {
  const parts: [number, number][][] = [];
  let current: [number, number][] = [];
  let clipped = 0;
  for (let i = 0; i + 1 < path.length; i += 1) {
    const a = path[i] as [number, number];
    const b = path[i + 1] as [number, number];
    if (!inside(a, w) || !inside(b, w)) clipped += 1;
    const t = clipParam(a, b, w);
    if (t === undefined) {
      if (current.length > 0) parts.push(current);
      current = [];
      continue;
    }
    const start: [number, number] = [a[0] + (b[0] - a[0]) * t[0], a[1] + (b[1] - a[1]) * t[0]];
    const end: [number, number] = [a[0] + (b[0] - a[0]) * t[1], a[1] + (b[1] - a[1]) * t[1]];
    if (current.length === 0) {
      current = [start, end];
    } else {
      const last = current[current.length - 1] as [number, number];
      if (Math.abs(last[0] - start[0]) < 1e-12 && Math.abs(last[1] - start[1]) < 1e-12) current.push(end);
      else {
        parts.push(current);
        current = [start, end];
      }
    }
  }
  if (current.length > 0) parts.push(current);
  return { parts, clipped };
}

/**
 * 双对数斜率（显示辅助，不是物理量）：只用来给"实测增长是不是 κ·δ 那一阶"一个标签。
 *
 * 结论性的东西（放大倍数、残差起伏、κu）全部来自帧 `meta`；这里算的斜率是为了满足
 * 规则 2 的"实测斜率必须与理论斜率同屏"，不参与任何判定。点少于 2 个返回 0。
 */
function powerLawSlope(xs: ArrayLike<number>, ys: ArrayLike<number>): number {
  const px: number[] = [];
  const py: number[] = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
    const x = xs[i] as number;
    const y = ys[i] as number;
    if (Number.isFinite(x) && x > 0 && Number.isFinite(y) && y > 0) {
      px.push(Math.log10(x));
      py.push(Math.log10(y));
    }
  }
  if (px.length < 2) return 0;
  const mx = px.reduce((a, b) => a + b, 0) / px.length;
  const my = py.reduce((a, b) => a + b, 0) / py.length;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < px.length; i += 1) {
    sxy += ((px[i] as number) - mx) * ((py[i] as number) - my);
    sxx += ((px[i] as number) - mx) ** 2;
  }
  return sxx === 0 ? 0 : sxy / sxx;
}
