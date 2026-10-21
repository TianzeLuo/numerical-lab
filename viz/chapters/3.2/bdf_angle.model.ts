/**
 * `bdf_angle`（3.2 章）的 model：BDF 的 A(α)-稳定角楔形。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | k=1…6 的稳定域边界家族（浅色；按 `family_k` **分段**画） |
 * | 2 | line | `main` | 当前 k 的边界（粗黑，`boundary_x/y`） |
 * | 3 | line | `main` | A(α) 楔形（**一条**四点折线：0→上边→回 0→下边，无 NaN） |
 * | 4 | line | `main` | 测试射线（α 方向，0→半径上限） |
 * | 5 | scatter | `main` | 射线上第一次越界的位置（由 `error>0` 扫出来） |
 * | 6 | line | `aux` | 测试射线上的超界量 max(0,\|R\|−1)（对数） |
 * | 7 | line | `aux` | 边界自洽残差 \|\|R(z_θ)\|−1\|（对数，虚线） |
 * | 8 | bars | `angles` | 各阶**实测** A(α) 角（线性 0…90°） |
 * | 9 | line | `angles` | 各阶**理论**角（同一条线性轴） |
 * | 10 | line | `angles` | 当前测试角 α（水平参考线） |
 * | 11 | scatter | `angles` | 当前 k 的实测角标记（`angle_mark_x/y`） |
 *
 * 时间轴语义：**测试角 α（度）**（manifest `time.mode="scrub"`，`tMax=90`）。帧 meta 的
 * `alpha_probe_deg` 是**实到**角（后端夹到 [0,90]），显示一律用它。
 *
 * 三条刻意的决定
 * --------------
 * 1. **楔形是"顺序拼接"的四点折线，不含 NaN**（帧 `strings.wedge_note` 写明）：本模型把它
 *    当成**一条** `line` 画（0→上边→回 0→下边，V 的拐点就是原点），不插任何 NaN、
 *    也不把它猜成"两条射线"。`family_x/family_y` 同理：那是 k=1…6 六条边界首尾相接，
 *    必须按 `family_k` 切段；当成一条画会多出五条横穿画面的假连线。
 * 2. **各阶角单独一块线性面板**：17.84°…90° 放进 aux 那条 16 个数量级的对数轴里会被挤成
 *    一条线（对数轴上 17.84 与 90 只差 0.7 个数量级）。两块面板共用同一批帧数组，
 *    前端不重算任何角度。manifest 把两者都写在 aux 的标题里，这里按"读得出来"优先拆开。
 * 3. **"射线没越界" ≠ "A-稳定"**：`strings.display_policy` 说得很清楚——超界量为 0 只说明
 *    这一条射线没出界。所以状态角标把 `a_stable`、`alpha_within_measured`、实测角三者分列，
 *    并且面板文本照抄这条 display_policy。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  canvasSize,
  clipBadgeText,
  criticalOf,
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
  badge,
  barsLayer,
  criticalBadge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  applyOverflowNote,
  argmaxIndex,
  axisTitleOf,
  axisYlabelOf,
  chapterLayout,
  clipRuns,
  integerTicks,
  logRange,
  runsBy,
  sliceRun,
  textBlock,
  textBottom,
  textTop,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.2";
const VIZ = "bdf_angle";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "BDF 的 A(α)-稳定角：阶数越高，楔形越窄";

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

/** 家族里"非当前阶"的浅色（六个 k 共用；当前 k 用粗黑压在上面）。 */
const FAMILY_COLOR = "#a0aec0";
/** A(α) 楔形的颜色（manifest 说"青色楔形"）。 */
const WEDGE_COLOR = "#0f766e";

export const bdfAngleModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const boundaryX = requireArray(frame, "boundary_x");
  const boundaryY = requireArray(frame, "boundary_y");
  const wedgeX = requireArray(frame, "wedge_x");
  const wedgeY = requireArray(frame, "wedge_y");
  const rayX = requireArray(frame, "ray_x");
  const rayY = requireArray(frame, "ray_y");
  const error = requireArray(frame, "error");
  const residual = requireArray(frame, "residual");
  const angleK = requireArray(frame, "angle_k");
  const angleMeasured = requireArray(frame, "angle_measured");
  const angleTheory = requireArray(frame, "angle_theory");
  const angleMarkX = requireArray(frame, "angle_mark_x");
  const angleMarkY = requireArray(frame, "angle_mark_y");
  const familyX = frame.has("family_x") ? requireArray(frame, "family_x") : undefined;
  const familyY = frame.has("family_y") ? requireArray(frame, "family_y") : undefined;
  const familyK = frame.has("family_k") ? requireArray(frame, "family_k") : undefined;

  requireSameLength(boundaryX, boundaryY, "帧数组 boundary_x 与 boundary_y");
  requireSameLength(wedgeX, wedgeY, "帧数组 wedge_x 与 wedge_y");
  requireSameLength(rayX, rayY, "帧数组 ray_x 与 ray_y");
  requireSameLength(rayX, error, "帧数组 ray_x 与 error");
  requireSameLength(boundaryX, residual, "帧数组 boundary_x 与 residual");
  requireSameLength(angleK, angleMeasured, "帧数组 angle_k 与 angle_measured");
  requireSameLength(angleK, angleTheory, "帧数组 angle_k 与 angle_theory");
  if (wedgeX.length !== 4 || wedgeY.length !== 4) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 wedge_x/wedge_y 长度应为 4（0→上边→回 0→下边），实际 ${wedgeX.length}/${wedgeY.length}`);
  }
  if (familyX !== undefined && familyK !== undefined && familyX.length !== familyK.length) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 family_x 与 family_k 长度不一致`);
  }
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const k = metaNumber(frame, "k");
  if (k === undefined) degraded.push("帧 meta 缺 k：标题不写当前阶（不按参数默认值猜）");
  const kMax = metaNumber(frame, "k_max");
  const alphaProbe = metaNumber(frame, "alpha_probe_deg");
  if (alphaProbe === undefined) degraded.push("帧 meta 缺 alpha_probe_deg：不画测试角参考线（不按请求 t 猜）");
  const angleCurrent = metaNumber(frame, "angle_current_deg");
  const angleTheoryDeg = metaNumber(frame, "angle_theory_deg");
  const angleErrorDeg = metaNumber(frame, "angle_error_deg");
  const aStable = (metaNumber(frame, "a_stable") ?? 0) > 0.5;
  const wedgeViolated = (metaNumber(frame, "wedge_violated") ?? 0) > 0.5;
  const violationMax = metaNumber(frame, "violation_max");
  const violationFloor = metaNumber(frame, "violation_floor");
  const alphaWithinMeasured = (metaNumber(frame, "alpha_within_measured") ?? 0) > 0.5;
  const residualMax = metaNumber(frame, "residual_max");
  const nTheta = metaNumber(frame, "n_theta_boundary");
  const familySize = metaNumber(frame, "family_size");
  const showFamily = (metaNumber(frame, "show_family") ?? 1) > 0.5;
  const anglesMonotone = (metaNumber(frame, "angles_monotone") ?? 0) > 0.5;
  const zLim = metaNumber(frame, "z_axis_limit") ?? 6;
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const displayPolicy = stringOf(frame, "display_policy");
  const nonfinitePolicy = stringOf(frame, "nonfinite_policy");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: metaNumber(frame, "t") ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const layout = chapterLayout(width, height, { rightShares: [0.5, 0.5], gutter: 0, minRight: 200 });
  const mainRect = layout.main;
  const auxRect = layout.right[0];
  const anglesRect = layout.right[1];
  scene.axes.push(
    makeAxis("main", mainRect, {
      xlim: [-zLim, zLim],
      ylim: [-zLim, zLim],
      xlabel: "$\\operatorname{Re} z$",
      ylabel: axisYlabelOf(SPEC?.axes, "main", "$\\mathrm{Im}\\,z$"),
      title: axisTitleOf(SPEC?.axes, "main", "稳定域边界家族与 A(α) 楔形（固定窗口）"),
      autoscaled: false,
    }),
  );
  const floorPos = violationFloor !== undefined && violationFloor > 0 ? violationFloor : Number.POSITIVE_INFINITY;
  const auxLo = Math.min(positiveMin(error) ?? Number.POSITIVE_INFINITY, positiveMin(residual) ?? Number.POSITIVE_INFINITY, floorPos);
  const auxHi = Math.max(finiteMax(error) ?? 0, finiteMax(residual) ?? 0, violationFloor ?? 0);
  const auxYlim: [number, number] =
    Number.isFinite(auxLo) && auxHi > 0 ? logRange(auxLo, auxHi) : [1e-18, 1];
  if (!(Number.isFinite(auxLo) && auxHi > 0)) degraded.push("error/residual 里没有 >0 的有限值：副轴用占位范围");
  scene.axes.push(
    makeAxis("aux", auxRect, {
      xlim: [0, 1],
      ylim: auxYlim,
      yscale: "log",
      xlabel: "归一化序号 $i/(N−1)$（实线：射线半径；虚线：边界 $θ$）",
      ylabel: axisYlabelOf(SPEC?.axes, "aux", "超界量 / 角度"),
      title: "测试射线上的超界量（对数）与边界残差",
      autoscaled: true,
    }),
  );
  // 各阶角单独一块线性面板（见文件头第 2 条）：固定 0…90°。
  // 纵轴下探到 −28°：柱子仍然从 0 往上长，而文字块住在 0 线**下方**的留白里——
  // 放到面板顶部会正好盖住 k=1、k=2 那两根 90° 的柱子（实测过）。
  scene.axes.push(
    makeAxis("angles", anglesRect, {
      xlim: [minOf(angleK) - 0.6, maxOf(angleK) + 0.6],
      ylim: [-28, 96],
      yticks: [0, 30, 60, 90],
      xticks: integerTicks(minOf(angleK), maxOf(angleK), 1),
      xlabel: "BDF 阶 $k$",
      ylabel: "$A(α)$ 角（度）",
      title: "各阶实测角 vs 理论角（线性 0…90°）",
      autoscaled: false,
    }),
  );
  const anglesAxis = scene.axes[2];

  // —— 图层 1：家族（按 family_k 分段；不插 NaN）——
  let nFamilyOutside = 0;
  const familyRuns = familyX !== undefined && familyY !== undefined && familyK !== undefined ? runsBy(familyK) : [];
  const familyKSeg = familyK;
  if (showFamily && familyRuns.length > 0 && familyY !== undefined) {
    for (const run of familyRuns) {
      const seg = sliceRun(familyX as ArrayLike<number>, familyY, run);
      for (let i = 0; i < seg.x.length; i += 1) {
        if (Math.abs(seg.x[i]) > zLim || Math.abs(seg.y[i]) > zLim) nFamilyOutside += 1;
      }
      const clipped = clipRuns(seg.x, seg.y, -zLim, zLim, (v) => Math.abs(v) <= zLim);
      const kOfRun = familyKSeg === undefined ? Number.NaN : familyKSeg[run.start];
      for (const piece of clipped.runs) {
        scene.layers.push(
          lineLayer(
            "main",
            series(piece.x, piece.y, kOfRun === k ? pal.ink : FAMILY_COLOR, {
              width: kOfRun === k ? 2.4 : 1.2,
              alpha: kOfRun === k ? 1 : 0.8,
              label: `BDF-${formatNumber(kOfRun, 2)}`,
            }),
          ),
        );
      }
    }
  } else if (!showFamily) {
    degraded.push("帧 meta.show_family=False：k=1…6 家族未画（当前 k 的边界仍然画）");
  } else if (familyRuns.length === 0) {
    degraded.push("帧里没有 family_x/family_y/family_k：家族未画（当前 k 的边界仍然画）");
  }

  // —— 图层 2：当前 k 的边界（窗口外的点不画，计数报账）——
  let nBoundaryOutside = 0;
  for (let i = 0; i < boundaryX.length; i += 1) {
    if (Math.abs(boundaryX[i]) > zLim || Math.abs(boundaryY[i]) > zLim) nBoundaryOutside += 1;
  }
  const boundaryRuns = clipRuns(boundaryX, boundaryY, -zLim, zLim, (v) => Math.abs(v) <= zLim).runs;
  for (const run of boundaryRuns) {
    scene.layers.push(lineLayer("main", series(run.x, run.y, pal.ink, { width: 2.4 })));
  }

  // —— 图层 3：楔形（一条四点折线；**不插 NaN**）——
  scene.layers.push(
    lineLayer(
      "main",
      series(wedgeX, wedgeY, WEDGE_COLOR, { width: 1.8, alpha: 0.95 }),
    ),
  );

  // —— 图层 4~5：测试射线 + 第一次越界点 ——
  scene.layers.push(lineLayer("main", series(rayX, rayY, pal.marker, { width: 1.6, dash: [5, 3] })));
  const firstViolation = firstAbove(error, 0);
  if (firstViolation >= 0) {
    scene.layers.push(
      scatterLayer("main", [rayX[firstViolation]], [rayY[firstViolation]], {
        r: 5,
        color: pal.warn,
        label: "射线上第一次越界（$error>0$）的位置",
      }),
    );
  }

  // —— 图层 6~7：副轴 ——
  const errX = normalizedIndex(error.length);
  const errClip = clipRuns(errX, error, auxYlim[0], auxYlim[1], (v) => v > 0);
  for (const run of errClip.runs) {
    scene.layers.push(lineLayer("aux", series(run.x, run.y, pal.warn, { width: 1.5 })));
  }
  const resX = normalizedIndex(residual.length);
  const resClip = clipRuns(resX, residual, auxYlim[0], auxYlim[1], (v) => v > 0);
  for (const run of resClip.runs) {
    scene.layers.push(lineLayer("aux", series(run.x, run.y, pal.residual, { width: 1.1, dash: [4, 3] })));
  }
  if (errClip.runs.length === 0 && resClip.runs.length === 0) degraded.push("两条辅助曲线在对数范围内都没有可画的点");
  const errAt = argmaxIndex(error);
  if (errAt >= 0 && error[errAt] > 0) {
    scene.layers.push(
      scatterLayer("aux", [errX[errAt]], [error[errAt]], {
        r: 4,
        color: pal.warn,
        label: "超界量最大值（位置由数组扫描得到，数值见 $meta.violation_max$）",
      }),
    );
  }

  // —— 图层 8~11：各阶角（线性面板）——
  scene.layers.push(
    barsLayer("angles", angleK, angleMeasured, { width: 0.5, color: WEDGE_COLOR, baseline: 0, label: "各阶实测 $A(α)$ 角" }),
  );
  scene.layers.push(
    lineLayer(
      "angles",
      series(angleK, angleTheory, pal.reference, { width: 1.4, dash: [4, 3], marker: "+" }),
    ),
  );
  if (alphaProbe !== undefined) {
    scene.layers.push(
      lineLayer(
        "angles",
        series([anglesAxis.xlim[0], anglesAxis.xlim[1]], [alphaProbe, alphaProbe], pal.marker, { width: 1.4 }),
      ),
    );
  }
  if (angleMarkX.length > 0 && angleMarkY.length > 0) {
    scene.layers.push(
      scatterLayer("angles", [angleMarkX[0]], [angleMarkY[0]], {
        r: 5.5,
        color: pal.exact,
        label: "当前 $k$ 的实测角（当前点标记）",
      }),
    );
  }

  // —— 文本 ——
  const kLabel = k === undefined ? "BDF-?" : `BDF-${formatNumber(k, 2)}`;
  const head: string[] = [
    `${kLabel}${kMax === undefined ? "" : `（本图最多 k=${formatNumber(kMax, 2)}：k≥7 连零稳定都没有）`}` +
      (angleCurrent === undefined ? "" : `　实测角 ${formatNumber(angleCurrent, 5)}°`) +
      (angleTheoryDeg === undefined ? "" : `（理论 ${formatNumber(angleTheoryDeg, 4)}°，差 ${formatNumber(angleErrorDeg ?? 0, 3)}°）`),
    `A-稳定=${aStable ? "是" : "否"}　测试角 α=${alphaProbe === undefined ? "?" : formatNumber(alphaProbe, 4)}°　楔形${wedgeViolated ? "已越界" : "仍在域内"}`,
  ];
  const mainLegend: string[] = [
    `浅灰=家族（k=1…6，按 family_k 分段画：${familyRuns.length} 段，窗口外 ${nFamilyOutside} 点未画）；粗黑=当前 k`,
    `青=楔形（一条四点折线：0→上边→回 0→下边，**没有 NaN 分隔符**）；红虚线=测试射线（半径 0…${formatNumber(zLim, 3)}）`,
  ];
  const mainLines = [...head, ...mainLegend];
  // 左列文字不许压到右列第一条轴（aux）的左侧装饰带（刻度数字 + ylabel，宽 56px）。
  const leftBand = layout.columns ? 56 + 4 : 0;
  const mainTextPx = layout.columns ? Math.max(120, auxRect[0] - leftBand - (mainRect[0] + 6) - 2) : mainRect[2] - 12;
  const mainTextMax = Math.max(24, Math.floor(mainTextPx / 7)); // textBlock 的 budget 是显示宽度单位
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), mainLines, pal.muted, mainTextMax, "top"));

  const auxLines: string[] = [];
  auxLines.push(
    `超界量 max=${formatNumber(violationMax ?? (errAt >= 0 ? error[errAt] : 0), 4)}` +
      (violationFloor === undefined ? "" : `（地板 ${formatNumber(violationFloor, 3)}=u/2：贴地板=这条射线没出界，**不代表** A-稳定）`),
  );
  auxLines.push(`边界残差 max=${formatNumber(residualMax ?? 0, 4)}（${formatNumber(nTheta ?? residual.length, 4)} 个 θ 点；边界是参数化算出来的，不是网格等高线）。`);
  if (displayPolicy !== undefined) auxLines.push(`读法：${displayPolicy}`);
  if (nonfinitePolicy !== undefined && note !== undefined) {
    auxLines.push("非有限值：本帧 |R| 与超界量全部有限——楔形整体在闭左半平面，而 BDF-k 的极点 z=α_k 全是正实数，永远不在楔形里。");
  }
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, unitBudget(auxRect[2]), "top"));

  const angleLines: string[] = [];
  if (angleK.length > 0) {
    const pairs: string[] = [];
    for (let i = 0; i < Math.min(angleK.length, 6); i += 1) {
      pairs.push(`${formatNumber(angleMeasured[i], 4)}/${formatNumber(angleTheory[i], 4)}`);
    }
    angleLines.push(`实测/理论（度，k=1…6）：${pairs.join("　")}`);
    angleLines.push(
      `k ${anglesMonotone ? "单调不增（帧 meta.angles_monotone=True）" : "⚠ 不是单调不增（帧 meta.angles_monotone=False）"}：` +
        "k=1、2 都是 90°（A-稳定），k=3 掉到 86.03°（Dahlquist 第二障碍）。",
    );
  }
  scene.texts.push(
    ...textBlock(anglesRect[0] + 6, textTop(anglesRect), textBottom(anglesRect) - 2, angleLines, pal.muted, unitBudget(anglesRect[2]), "bottom"),
  );

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", `主轴：固定窗口 ±${formatNumber(zLim, 3)}（楔形与射线都按这个半径画）`, 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(placeBadge(badge("autoscale", "副轴：自适应 log（按本帧超界量取整；会掩盖量级变化）", 8, 26), SLOT.auxAxis, width));
  const kCrit = criticalOf(frame, "k", 3);
  if (k !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("阶 k", k, kCrit, { beyond: "above" }), SLOT.critical, width));
  }
  const status: string[] = [];
  status.push(`A(α) 角 ${angleCurrent === undefined ? "?" : formatNumber(angleCurrent, 5)}°${aStable ? "（A-稳定）" : "（不是 A-稳定）"}`);
  status.push(
    alphaProbe === undefined
      ? "测试角缺失"
      : `测试角 α=${formatNumber(alphaProbe, 4)}° ${alphaWithinMeasured ? "≤ 实测角（楔形含在域内）" : "> 实测角（楔形已越界）"}`,
  );
  if (wedgeViolated) status.push("⚠ 这条射线上 |R|>1：楔形确实出界了");
  if (note !== undefined) status.push(note);
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  if (familySize !== undefined && familyX !== undefined && familySize !== familyX.length) {
    degraded.push(`帧 meta.family_size=${formatNumber(familySize, 4)} 与 family_x 长度 ${familyX.length} 不一致：按数组画`);
  }
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(boundaryX) + countNonFinite(boundaryY) + countNonFinite(wedgeX) + countNonFinite(wedgeY) + countNonFinite(rayX) + countNonFinite(rayY) + countNonFinite(error) + countNonFinite(residual);
  const extra: string[] = [];
  if (nFamilyOutside > 0) extra.push(`家族边界有 ${nFamilyOutside} 个点超出固定窗口 ±${formatNumber(zLim, 3)}（k≥3 的稳定域比窗口大），未绘制`);
  if (nBoundaryOutside > 0) extra.push(`当前 k 的边界有 ${nBoundaryOutside}/${boundaryX.length} 个点超出固定窗口，未绘制`);
  if ((metaNumber(frame, "overflow") ?? 0) > 0.5) extra.push("帧 meta.overflow=True：后端标记本帧存在非有限值");
  applyOverflowNote(scene, extra.join("；"), nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** `i/(N−1)`：把一条扫描序列映射到 [0,1]（纯显示几何）。 */
function normalizedIndex(n: number): Float64Array {
  const out = new Float64Array(Math.max(0, n));
  for (let i = 0; i < n; i += 1) out[i] = n <= 1 ? 0 : i / (n - 1);
  return out;
}

/**
 * 对数轴上能用的小值：**只收 > 0 的有限值**。
 *
 * 帧里恰为 0 的样本（后端把地板值抬到 u/2，但降级帧或别的章节可能真的给 0）如果进了
 * `niceLimits(lo, hi, {log:true})`，会当场抛 `RangeError`——那是"画不出来"，不是"帧坏了"。
 */
function positiveMin(a: ArrayLike<number>): number | undefined {
  let best: number | undefined;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v) || v <= 0) continue;
    if (best === undefined || v < best) best = v;
  }
  return best;
}

function finiteMax(a: ArrayLike<number>): number | undefined {
  let best: number | undefined;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    if (best === undefined || v > best) best = v;
  }
  return best;
}

/** 第一个严格大于 `level` 的下标（阈值比较，不是重算放大因子）。 */
function firstAbove(values: ArrayLike<number>, level: number): number {
  for (let i = 0; i < values.length; i += 1) {
    if (Number.isFinite(values[i]) && values[i] > level) return i;
  }
  return -1;
}

function minOf(a: ArrayLike<number>): number {
  let best = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) if (Number.isFinite(a[i]) && a[i] < best) best = a[i];
  return Number.isFinite(best) ? best : 0;
}

function maxOf(a: ArrayLike<number>): number {
  let best = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) if (Number.isFinite(a[i]) && a[i] > best) best = a[i];
  return Number.isFinite(best) ? best : 0;
}
