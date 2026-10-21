/**
 * `singular_endpoint`（2.4 章）的 model：端点奇异 —— tanh–sinh（双指数）vs 换元后的等距规则。
 *
 * 两个面板（顺序即绘制顺序，测试按此断言）
 * ----------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `conv` | **地板理论值** `floor_value` = \|∫₀^{u/2}f\|（横线，先画，压在数据下面） |
 * | 2 | line | `conv` | tanh–sinh 的误差 `error_ts`（按 `meta.step` 截断到已揭示的 m） |
 * | 3 | line | `conv` | 换元 x=t² 后等距规则的误差 `error_sub`（同一节点窗口，代数收敛） |
 * | 4 | line | `conv` | 饱和开始处（`critical.n` = 首个饱和节点数）竖线 |
 * | 5 | scatter | `conv` | 当前 m 的误差点（把时间轴与曲线接起来） |
 * | 6 | line | `tail` | 未采样尾巴 `ts_tail`（实测）与地板理论值 |
 * | 7 | line | `tail` | 最靠近奇异端点的节点 `ts_xmin`（地板 = 它下不去的直接原因） |
 * | 8 | line | `tail` | 饱和节点数 `ts_saturated`（只画 > 0 的部分：log 轴画不出 0） |
 *
 * 角标槽位（固定，见 `common.ts::SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 /
 * 3 临界（首个饱和节点数 103）/ 4 状态（原始等距规则有没有 inf）/ 5 前端预览 /
 * 6 溢出 / 7 降级。
 *
 * 这一页最容易骗人的两处，以及它们在图上的防法
 * --------------------------------------------
 * 1. **「双指数收敛 ⇒ 能到机器精度」**。实测它停在 `√(2u)≈1.5e-8`：`tail` 面板把
 *    "地板 = 未被采样的尾巴"这条因果链画出来（x_min 下到 u/2、尾巴就停在 |∫₀^{u/2}f|），
 *    `conv` 面板把地板画成横线，并把"最后两点的误差比"算出来（≈1 就是"一动不动"）。
 * 2. **「等距规则也能算这个积分」**。原始被积函数在 x=0 处是 inf：帧里
 *    `raw_equispaced_finite=false`、`raw_equispaced_nonfinite=1`——这不是"误差大"，
 *    是**公式里出现了 inf**。防法：状态角标 + 面板文字点名，并把"换元之后才重新可用"
 *    的对照曲线画在同一窗口里（`meta.sub_order` 给出代数阶）。
 *
 * 几条刻意的决定
 * --------------
 * 1. **`error_ts` 在已揭示窗口之外填的是地板占位值**（后端 `np.where(reveal, ts_c, floor)`）：
 *    直接整条画出来会读成"节点加多以后误差掉到 1e-16 并停住"。模型按 `meta.step` 截断，
 *    并在文字里说明数组尾部是占位值而不是数据（`docs/PROTOCOLS.md` §1.3.1 的掩码纪律）。
 * 2. **横轴跨整个节点预算**（不随 t 收缩）：拖时间轴时只reveal更多点，轴不跳，
 *    "再多的节点也没用"这件事才看得出来。
 * 3. **三条曲线的单位写在同屏文字里**：`tail` 面板上 x_min 是坐标、尾巴是面积、
 *    饱和节点数是计数——同一个 log 轴上必须点名单位，否则"都往下掉"会被读成一回事。
 * 4. **`func=log` 时 `sub_available=false`**：帧把整条对照曲线填成地板并明说不可用，
 *    这时只画 tanh–sinh 一条曲线（画一条"贴着地板的对照线"等于编数据）。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
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
  segmentedSeries,
  toNumeric,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  CHAPTER,
  SLOT,
  axisTitle,
  axisYlabel,
  currentStep,
  fmt,
  fmtOpt,
  head,
  noteOf,
  paramSpec,
  plainNote,
  positiveMinMaxOfAll,
  sci,
  specTitle,
  splitColumns,
  textBlock,
  unitBudget,
} from "./common.js";

const VIZ = "singular_endpoint";
const TITLE = specTitle(VIZ, "端点奇异：双指数变换 vs 换元");

export const singularEndpointModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);
  const layout = splitColumns(width, height, { share: 0.56, gap: 64 });

  // —— 必需数组 ——
  const tsNodes = requireArray(frame, "ts_nodes");
  const errTs = requireArray(frame, "error_ts");
  const tsTail = requireArray(frame, "ts_tail");
  const tsXmin = requireArray(frame, "ts_xmin");
  const tsSat = requireArray(frame, "ts_saturated");
  const floorValue = requireArray(frame, "floor_value");
  requireSameLength(tsNodes, errTs, "帧数组 ts_nodes 与 error_ts");
  requireSameLength(tsNodes, tsTail, "帧数组 ts_nodes 与 ts_tail");
  requireSameLength(tsNodes, tsXmin, "帧数组 ts_nodes 与 ts_xmin");
  requireSameLength(tsNodes, tsSat, "帧数组 ts_nodes 与 ts_saturated");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const step = currentStep(frame, opts, { lo: 1, hi: Math.max(1, tsNodes.length), label: "每侧节点数 $m$" });
  const m = Math.max(1, Math.min(tsNodes.length, Math.round(step.step ?? 1)));
  const subAvailable = (metaNumber(frame, "sub_available") ?? 1) > 0.5;
  const errorFloor = metaNumber(frame, "error_floor");
  const tailTheory = metaNumber(frame, "tail_theory");
  const tailEstimate = metaNumber(frame, "tail_estimate");
  const tailRatio = metaNumber(frame, "tail_ratio");
  const xMin = metaNumber(frame, "x_min");
  const nSat = metaNumber(frame, "n_saturated");
  const subOrder = metaNumber(frame, "sub_order");
  const slopeFinal = metaNumber(frame, "ts_slope_final");
  const slopeR2 = metaNumber(frame, "ts_slope_r2");
  const clipped = metaNumber(frame, "clipped_points");
  const rawFinite = metaNumber(frame, "raw_equispaced_finite");
  const rawNonFinite = metaNumber(frame, "raw_equispaced_nonfinite");
  const fUndef = metaNumber(frame, "f_undefined_points");
  const hLabel = noteOf(frame, "h_label");
  const funcLabel = noteOf(frame, "func_label") ?? noteOf(frame, "func") ?? "?";
  const smoothness = noteOf(frame, "smoothness");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: metaNumber(frame, "t") ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const nodeLo = tsNodes.length > 0 ? (tsNodes[0] as number) : 3;
  const nodeHi = tsNodes.length > 0 ? (tsNodes[tsNodes.length - 1] as number) : 129;

  /* ------------------------------ 左列：误差 vs 节点数 ------------------------------ */
  const convAxis = makeAxis("conv", layout.left, {
    // 右端多留 1/3：曲线自己的 `label` 画在**最后可见点**右侧（`未采样尾巴` 这类
    // 标签有 80px 宽），不留白就会被面板右边界裁成「未采样尾」。
    xlim: niceLimits(nodeLo, nodeLo + (nodeHi - nodeLo) * 1.34, 6),
    yscale: "log",
    xlabel: "总节点数 $2m+1$",
    ylabel: axisYlabel(VIZ, "conv", "|Q−I|"),
    title: axisTitle(VIZ, "conv", "误差 vs 节点数（半对数）"),
    autoscaled: true,
  });
  // 纵轴范围用**全量**数组算（含地板占位值）：拖动时间轴时轴不跳。
  const errRange = positiveMinMaxOfAll([errTs, floorValue, errorFloor !== undefined && errorFloor > 0 ? [errorFloor] : undefined]) ?? [1e-16, 1];
  convAxis.ylim = niceLimits(errRange[0], errRange[1], 6, { log: true, pad: 0.03 });
  convAxis.yticks = niceTicks(convAxis.ylim[0], convAxis.ylim[1], 5, { log: true });
  convAxis.xticks = niceTicks(convAxis.xlim[0], convAxis.xlim[1], 6);
  scene.axes.push(convAxis);

  scene.layers.push(
    lineLayer("conv", series(floorSpan(convAxis.xlim), [floorValue[0] as number, floorValue[0] as number], pal.reference, {
      width: 1.3,
      dash: [6, 4],
      label: "地板",
    })),
  );
  const nodesRev = head(tsNodes, m);
  const errRev = head(errTs, m);
  scene.layers.push(
    lineLayer("conv", segmentedSeries(nodesRev, errRev, pal.curves[0], { width: 2, marker: "o", accept: (v) => v > 0, label: "tanh–sinh" })),
  );
  if (subAvailable && frame.has("error_sub")) {
    const subNodes = requireArray(frame, "sub_nodes");
    const errSub = requireArray(frame, "error_sub");
    requireSameLength(subNodes, errSub, "帧数组 sub_nodes 与 error_sub");
    scene.layers.push(
      lineLayer(
        "conv",
        segmentedSeries(head(subNodes, m), head(errSub, m), pal.curves[1] as string, {
          width: 1.6,
          dash: [5, 3],
          accept: (v) => v > 0,
          label: "换元+等距",
        }),
      ),
    );
  } else if (subAvailable) {
    degraded.push("帧声明换元可用（sub_available=true）但没有 error_sub 数组：对照曲线画不出来，已说明");
  }
  const satNodes = criticalOf(frame, "n", paramSpec(VIZ, "n")?.critical[0] ?? 103);
  const satValue = Array.isArray(satNodes) ? (satNodes[0] as number) : satNodes;
  if (Number.isFinite(satValue) && satValue >= convAxis.xlim[0] && satValue <= convAxis.xlim[1]) {
    scene.layers.push(
      lineLayer("conv", series([satValue, satValue], convAxis.ylim, pal.warn, { width: 1.3, dash: [5, 4], label: `饱和起点 ${fmt(satValue, 4)}` })),
    );
  } else if (Number.isFinite(satValue)) {
    degraded.push(`临界值 n=${fmt(satValue, 4)}（首个饱和节点数）在当前节点预算之外：竖线不画，文字里保留`);
  }
  const curIdx = m - 1;
  if (curIdx >= 0 && curIdx < tsNodes.length) {
    scene.layers.push(
      scatterLayer("conv", [tsNodes[curIdx] as number], [errTs[curIdx] as number], { r: 3.8, color: pal.marker, label: `当前 m=${fmt(m, 3)}` }),
    );
  }

  const errNow = metaNumber(frame, "error_current");
  const convLines: string[] = [];
  convLines.push(`${funcLabel}　${hLabel ?? "h = ?"}　当前 m=${fmt(m, 3)}（节点 ${fmt(curIdx >= 0 ? (tsNodes[curIdx] as number) : nodeHi, 4)} 个）`);
  if (smoothness !== undefined) convLines.push(smoothness);
  convLines.push(`当前误差 ${fmtOpt(errNow, 4)}；地板 ${sci(floorValue[0] as number)}（= 未采样尾巴的理论值）；被截断点数 ${fmtOpt(clipped, 3)}`);
  if (subOrder !== undefined) {
    convLines.push(`换元对照的实测阶 ${fmt(subOrder, 8)}（理论 2.0000000）：奇异性被换元吃掉后，等距规则回到代数收敛`);
  } else if (!subAvailable) {
    convLines.push("这一格没有可用的有限区间换元：对照曲线留白（sub_available=false），不画一条贴着地板的假曲线");
  }
  const plateau = plateauRatio(nodesRev, errRev, floorValue[0] as number);
  if (plateau !== undefined) {
    convLines.push(
      `最后两点（节点 ${fmt(plateau.x0, 4)} → ${fmt(plateau.x1, 4)}）误差比 ${plateau.ratio.toFixed(6)}：≈1 就是「再加节点一动不动」`,
    );
  }
  if (slopeFinal !== undefined) {
    // 斜率本身是帧给的；**怎么读它**要看它的大小：末段已饱和时斜率≈0（R² 也低），
    // 这时说"几乎垂直"就是把地板读成收敛。
    convLines.push(
      `tanh–sinh 末段半对数斜率 ${fmt(slopeFinal, 4)}（R²=${fmtOpt(slopeR2, 4)}）：` +
        (Math.abs(slopeFinal) > 1 ? "双指数段在触地板前几乎垂直" : "末段已经压平——这一段是地板，不是收敛"),
    );
  }
  convLines.push("⚠ 帧数组在已揭示窗口之外填的是地板占位值：本图按 meta.step 截断，不把占位值当数据");
  convLines.push(...step.notes);
  scene.texts.push(
    ...textBlock(convAxis.rect[0] + 6, convAxis.rect[1] + 4, convAxis.rect[1] + convAxis.rect[3] - 4, convLines, pal.note, unitBudget(Math.round(convAxis.rect[2] * 0.8)), "bottom"),
  );

  /* ------------------------------ 右列：尾巴与饱和节点 ------------------------------ */
  const tailAxis = makeAxis("tail", layout.right, {
    // 右端多留 1/3：曲线自己的 `label` 画在**最后可见点**右侧（`未采样尾巴` 这类
    // 标签有 80px 宽），不留白就会被面板右边界裁成「未采样尾」。
    xlim: niceLimits(nodeLo, nodeLo + (nodeHi - nodeLo) * 1.34, 6),
    yscale: "log",
    xlabel: "总节点数 $2m+1$",
    ylabel: axisYlabel(VIZ, "tail", "∫₀^{x_min}f"),
    title: axisTitle(VIZ, "tail", "未采样尾巴与饱和节点"),
    autoscaled: true,
  });
  const tailRange = positiveMinMaxOfAll([tsTail, tsXmin, tsSat, floorValue]) ?? [1e-16, 1];
  tailAxis.ylim = niceLimits(tailRange[0], tailRange[1], 6, { log: true, pad: 0.03 });
  tailAxis.yticks = niceTicks(tailAxis.ylim[0], tailAxis.ylim[1], 5, { log: true });
  tailAxis.xticks = niceTicks(tailAxis.xlim[0], tailAxis.xlim[1], 6);
  scene.axes.push(tailAxis);

  scene.layers.push(
    lineLayer("tail", series(floorSpan(tailAxis.xlim), [floorValue[0] as number, floorValue[0] as number], pal.reference, {
      width: 1.3,
      dash: [6, 4],
      label: "地板",
    })),
  );
  scene.layers.push(
    lineLayer("tail", segmentedSeries(nodesRev, head(tsTail, m), pal.warn, { width: 2, marker: "o", accept: (v) => v > 0, label: "未采样尾巴" })),
  );
  scene.layers.push(
    lineLayer("tail", segmentedSeries(nodesRev, head(tsXmin, m), pal.muted, { width: 1.6, accept: (v) => v > 0, label: "`x_min`" })),
  );
  const satHead = head(tsSat, m);
  const satPositive = Array.from(satHead).some((v) => Number.isFinite(v) && v > 0);
  if (satPositive) {
    scene.layers.push(
      lineLayer("tail", segmentedSeries(nodesRev, satHead, pal.curves[2] as string, {
        width: 1.6,
        dash: [2, 3],
        marker: "o",
        accept: (v) => v > 0,
        label: "饱和节点数",
      })),
    );
  }
  const tailLines: string[] = ["⚠ 同一个 log 轴上是三种单位：尾巴 = 面积、x_min = 坐标、饱和节点数 = 计数"];
  tailLines.push(
    `x_min=${fmtOpt(xMin, 4)}（≈u/2=${sci(0.5 * 2 ** -53)}）；尾巴 ${fmtOpt(tailEstimate, 4)} vs 理论 ${fmtOpt(tailTheory, 4)}` +
      (tailRatio === undefined ? "" : `，误差/尾巴 = ${fmt(tailRatio, 4)}（≈1 ⇒ 误差就是这块没采到的面积）`),
  );
  tailLines.push(`饱和节点数 ${fmtOpt(nSat, 4)}：权重下溢到 0、函数值是 ∞，贡献必须按 0 计（0·inf 得 NaN）`);
  if (fUndef !== undefined && fUndef > 0) {
    tailLines.push(`被积函数 ${fmt(fUndef, 3)} 个采样点非有限（x=0 处），已按 u/2 处的值截断显示（f_defined 掩码）`);
  }
  pushIf(tailLines, plainNote(noteOf(frame, "floor_note")));
  pushIf(tailLines, plainNote(noteOf(frame, "sub_note")));
  scene.texts.push(
    ...textBlock(tailAxis.rect[0] + 6, tailAxis.rect[1] + 4, tailAxis.rect[1] + tailAxis.rect[3] - 4, tailLines, pal.note, unitBudget(Math.round(tailAxis.rect[2] * 0.8)), "bottom"),
  );

  /* ------------------------------ 角标 ------------------------------ */
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "两列纵轴都自适应（log 轴按本帧数据取整；自适应会掩盖量级变化）", 8, 26), SLOT.auxAxis, width),
  );
  scene.badges.push(
    placeBadge(
      criticalBadge("nodes", Number.isFinite(satValue) ? satValue : Number.NaN, paramSpec(VIZ, "n")?.critical ?? [], { beyond: "above" }),
      SLOT.critical,
      width,
    ),
  );
  const status: string[] = [];
  if (rawFinite !== undefined && rawFinite < 0.5) {
    status.push(`⚠ 原始等距规则没有定义（x=0 处 f=∞）：实测 ${fmtOpt(rawNonFinite, 3)} 个节点值非有限——不是误差大，是公式里出现了 inf`);
  } else {
    status.push("原始等距规则在这一格有限（可以对照）");
  }
  if (plateau !== undefined && plateau.ratio > 0.999 && plateau.ratio < 1.001) status.push("已到地板：再加节点误差不动");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 80), SLOT.status, width));
  if (scene.dataSource.startsWith("backend+preview")) scene.badges.push(placeBadge(previewBadge(), SLOT.preview, width));

  let rawNonFiniteCount = 0;
  for (const name of frame.names) rawNonFiniteCount += countNonFinite(toNumeric(frame.array(name)));
  applyOverflow(scene, rawNonFiniteCount, { slot: SLOT.overflow, width });
  if (fUndef !== undefined && fUndef > 0) {
    // 规则 4：被积函数的未定义点已经**如实标了出来**（掩码 + 文案），这里补一条溢出说明。
    scene.overflow.note = `${scene.overflow.note === "" ? "" : scene.overflow.note + "；"}` +
      `被积函数 ${fmt(fUndef, 3)} 个采样点非有限（x=0 处 f=∞），已截断到 u/2 处的值并用 f_defined 掩码标出`;
    scene.overflow.flag = true;
  }
  const degradedObj = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedObj) scene.badges.push(degradedObj);

  requireDrawableLayers(scene);
  return scene;
};

/* --------------------------------------------------------------------------- */
/* 局部工具                                                                      */
/* --------------------------------------------------------------------------- */

/**
 * 已揭示窗口里最后两个**在地板之上**的点的误差比（"再加节点还动不动"）。
 *
 * 这是两个帧值的相除（显示空间的读数），不是重算误差；找不到两个可比的点就返回
 * `undefined`（文字里留白，不编一个 1.000000）。
 */
function plateauRatio(
  x: ArrayLike<number>,
  err: ArrayLike<number>,
  floor: number,
): { ratio: number; x0: number; x1: number } | undefined {
  for (let i = err.length - 1; i > 0; i -= 1) {
    const b = err[i] as number;
    const a = err[i - 1] as number;
    if (!Number.isFinite(a) || !Number.isFinite(b) || a <= floor * 1.000001 || b <= floor * 1.000001) continue;
    return { ratio: b / a, x0: x[i - 1] as number, x1: x[i] as number };
  }
  return undefined;
}

/**
 * 横贯全轴的参考线只画 72% 宽：曲线自己的 `label` 画在**最后一个可见点右侧**，
 * 画满整轴时那个点就落在面板右边界上，标签必被裁（实测「地板理论值 1.49×10⁻⁸」被裁掉一大半）。
 */
function floorSpan(xlim: readonly number[]): number[] {
  const lo = xlim[0] as number;
  const hi = xlim[1] as number;
  return [lo, lo + (hi - lo) * 0.72];
}

/** 条件式入队（`undefined` 不入队）。 */
function pushIf(lines: string[], body: string | undefined): void {
  if (body !== undefined) lines.push(body);
}
