/**
 * `oscillator_regimes`（0.0 章）的 model：阻尼振子三种分支 + ODE 残差。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 当前时刻游标（虚线竖线，仅在标记可画时出现） |
 * | 2 | line | `main` | `x(t)` 全曲线——**整段**帧数组 `t`/`x`，非有限值处断开 |
 * | 3 | scatter | `main` | `meta.t` 处的运动标记（帧内样本线性插值，不外推） |
 * | 4 | line | `error` | `residual` 逐点折线（对数纵轴，**不平滑、不截断**） |
 * | 5 | line | `error` | `meta.residual_max` 参考线（后端给的标量，不是前端算的） |
 *
 * 角标槽位（固定，见 `SLOT`；缺一个不会让其余挪位）：0 数据来源 / 1 主面板轴模式 /
 * 2 误差轴自适应 / 3 ζ 临界 / 4 溢出（`checkOverflow` + 帧内非有限值计数）/
 * 5 前端预览（仅 `backend+preview` 时）/ 7 降级说明。
 *
 * 几条刻意的决定
 * --------------
 * 1. **不自己判阻尼分支**：分支说明文本直接取 `frame.strings.regime_note`（前面带上
 *    `frame.strings.regime` 这个机器标记，如 `underdamped`）；它缺失时显示"帧未给出
 *    regime_note"，绝不按 ζ 猜（那是后端的判定）。
 * 2. **主面板纵轴固定**（`autoscaled: false`）：范围由整段 `x(t)` 做 1-2-5 取整得到，
 *    与当前时刻无关，因此动画中恒定。自适应纵轴会把过阻尼的衰减尾巴"拉满"，
 *    让两种分支看起来一样陡——这正是本 viz 的陷阱（详见 `frame.pitfalls`）。
 * 3. **残差是噪声，就画成噪声**：逐点折线、无平滑、无抽稀、无截断；
 *    非有限值处**断开**（不跨缺口连线），并把个数写进溢出说明。
 * 4. `meta.t` 只用于定位标记：它由后端 `build_frame` 写入；缺失时回落到 `opts.t`
 *    （UI 请求时刻）并挂降级角标——不假装两者相等。
 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite, dataToPixel } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finiteMinMax,
  finitePositiveMinMax,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sampleLinear,
  sceneShell,
  segmentedSeries,
  stackedPanels,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  makeErrorAxis,
  scatterLayer,
  series,
  previewBadge,
  sourceBadge,
  text,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "0.0";
const VIZ = "oscillator_regimes";
const TITLE = manifest.vizzes.find((v) => v.id === VIZ)?.title ?? "阻尼振子：三种阻尼分支与残差";
/** manifest 里 `zeta.critical = [1.0]`；`frame.critical.zeta` 缺失时的回落值（不另设阈值）。 */
const ZETA_CRITICAL = 1.0;

/**
 * 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。
 * 长文本角标（溢出、降级）各占一行的左列，右边留空，避免与邻居叠字。
 */
const SLOT = {
  source: 0,
  axisMode: 1,
  errorAxis: 2,
  critical: 3,
  overflow: 4,
  /** 仅当帧声明 `backend+preview` 时出现（本章的帧都是 backend）。 */
  preview: 5,
  degraded: 7,
} as const;

export const oscillatorModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const tt = requireArray(frame, "t");
  const x = requireArray(frame, "x");
  const residual = requireArray(frame, "residual");
  requireSameLength(tt, x, "帧数组 t 与 x");
  requireSameLength(tt, residual, "帧数组 t 与 residual");
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：当前时刻改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const zeta = metaNumber(frame, "zeta");
  if (zeta === undefined) degraded.push("帧 meta 缺 zeta：不标 ζ 的临界值（不猜）");
  const regimeNote = stringOf(frame, "regime_note");
  const regimeToken = stringOf(frame, "regime");
  if (regimeNote === undefined) degraded.push("帧 strings 缺 regime_note：不自行判断阻尼分支");
  const residualMax = metaNumber(frame, "residual_max");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 轴范围：全部由帧数据 1-2-5 取整，不硬编码数值 ——
  const panels = stackedPanels(width, height);
  const tRange = finiteMinMax(tt) ?? [0, 1];
  const xRange = finiteMinMax(x) ?? [0, 1];
  const tlim = niceLimits(tRange[0], tRange[1], 8);
  // 主面板纵轴：整段解的取整范围 + 5% 余量；与 t 无关 ⇒ 动画中固定（规则 1/4）。
  const mainYlim = niceLimits(xRange[0], xRange[1], 6, { pad: 0.05 });

  scene.axes.push(
    makeAxis("main", panels.main, {
      xlim: tlim,
      ylim: mainYlim,
      xlabel: "",
      ylabel: "位移 x",
      title: "x(t)",
      autoscaled: false,
    }),
  );

  // 残差轴：对数纵轴，范围只由 >0 的有限残差决定（对数轴画不出 ≤0）。
  const resRange = finiteMinMax(residual);
  const resPositive = finitePositiveMinMax(residual);
  const errorYlim: [number, number] =
    resPositive === undefined ? [1e-16, 1] : niceLimits(resPositive[0], resPositive[1], 6, { log: true, pad: 0.2 });
  if (resPositive === undefined) degraded.push("帧 residual 没有正值：对数轴无可绘点，仅标出轴范围");
  const errorAxis = makeErrorAxis("error", panels.error, {
    xlim: tlim,
    ylim: errorYlim,
    xlabel: "t（帧时间轴）",
    ylabel: "|x''+2ζωx'+ω²x|",
    title: "ODE 残差",
    autoscaled: true,
  });
  scene.axes.push(errorAxis);

  // —— 图层 1~3：主面板 ——
  const xNow = sampleLinear(tt, x, tNow);
  if (xNow !== undefined) {
    scene.layers.push(
      lineLayer("main", series([tNow, tNow], [mainYlim[0], mainYlim[1]], pal.cursor, { dash: [3, 3], width: 1 })),
    );
  }
  const xSeries = segmentedSeries(tt, x, pal.curves[0], { width: 1.8, label: "x(t)（帧内全时段解析解）" });
  if (xSeries.length > 0) scene.layers.push(lineLayer("main", xSeries));
  else degraded.push(`帧数组 x 没有可绘的有限值（${x.length} 个样本全为非有限）：主面板不画曲线`);
  if (xNow !== undefined) {
    scene.layers.push(
      scatterLayer("main", [tNow], [xNow], { r: 4.5, color: pal.marker, label: "当前时刻标记" }),
    );
    const [mx, my] = dataToPixel(scene.axes[0]!, tNow, xNow);
    if (Number.isFinite(mx) && Number.isFinite(my)) {
      const onRight = mx > panels.main[0] + panels.main[2] * 0.62;
      scene.texts.push(
        text(mx + (onRight ? -8 : 8), clamp(my - 20, panels.main[1] + 4, panels.main[1] + panels.main[3] - 20), `t = ${formatNumber(tNow, 4)}`, {
          role: "note",
          size: 11,
          color: pal.marker,
          align: onRight ? "right" : "left",
        }),
      );
    }
  } else {
    degraded.push(
      `请求 t=${formatNumber(tNow, 4)} 不在帧时间范围 [${formatNumber(tRange[0], 4)}, ${formatNumber(tRange[1], 4)}] 内：不画当前时刻标记（不外推）`,
    );
  }

  // —— 图层 4~5：误差面板（规则 3）——
  // `accept: v > 0`：对数轴画不出 ≤0 的点；这些点不是"被裁掉"，而是本来就画不出来，单独报数。
  const resSeries = segmentedSeries(tt, residual, pal.residual, {
    width: 1,
    label: "residual（逐点，未平滑/未截断）",
    accept: (v) => v > 0,
  });
  if (resSeries.length > 0) scene.layers.push(lineLayer("error", resSeries));
  else degraded.push(`帧 residual 的 ${residual.length} 个样本没有一个 > 0：对数轴上画不出任何点`);
  if (resSeries.length > 0 && residualMax !== undefined && residualMax > 0) {
    scene.layers.push(
      lineLayer("error", series(tlim, [residualMax, residualMax], pal.reference, { dash: [5, 4], width: 1, label: "meta.residual_max" })),
    );
  }

  // —— 文本 ——
  const summary = paramSummary(frame, opts, [
    { name: "omega", label: "ω" },
    { name: "zeta", label: "ζ" },
    { name: "n_periods", label: "n_periods" },
  ]);
  degraded.push(...summary.degraded);
  scene.texts.push(
    text(panels.main[0] + 6, panels.main[1] + 4, summary.text, {
      role: "label",
      size: 11,
      color: pal.muted,
      maxWidth: panels.main[2] * 0.45,
    }),
  );
  const regimeLine =
    regimeNote === undefined
      ? "帧未给出 regime_note（不按 ζ 猜分支）"
      : `${regimeToken === undefined ? "" : `${regimeToken}｜`}${regimeNote}`;
  scene.texts.push(
    text(panels.main[0] + panels.main[2] - 6, panels.main[1] + 4, regimeLine, {
      role: "note",
      size: 12,
      color: regimeNote === undefined ? pal.warn : pal.note,
      align: "right",
      maxWidth: panels.main[2] * 0.5,
    }),
  );
  const nBad = countNonFinite(residual);
  const nNonPositive = countNonPositive(residual);
  scene.texts.push(
    text(
      panels.error[0] + 6,
      panels.error[1] + 4,
      `残差 ${residual.length} 点逐点画出（未平滑、未截断）；${nNonPositive} 个 ≤0（对数轴画不出）、${nBad} 个非有限` +
        (resRange === undefined ? "；整段没有有限值" : ""),
      { role: "note", size: 11, color: nBad + nNonPositive > 0 ? pal.warn : pal.muted, maxWidth: panels.error[2] * 0.66 },
    ),
  );

  // —— 角标（固定槽位：0 来源 / 1 主面板轴模式 / 2 误差轴 / 3 ζ 临界 / 4 溢出 / 7 降级）——
  // 规则 5：恒挂一枚 kind="source" 的数据来源徽标；`dataSource` 是预览型时另挂 preview 徽标
  // （两个 kind 各有唯一含义：`sourceBadge` 只回答"数据从哪来"，`previewBadge` 只回答"这是近似"）。
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "误差轴：自适应（log 纵轴按本帧残差取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.errorAxis,
      width,
    ),
  );
  if (zeta !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("ζ", zeta, criticalOf(frame, "zeta", ZETA_CRITICAL), { beyond: "above" }), SLOT.critical, width),
    );
  }
  const status = degradedBadge(degraded, SLOT.degraded, width);
  if (status !== undefined) scene.badges.push(status);

  requireDrawableLayers(scene);
  // 规则 1：图层越界 + 帧内非有限值，两者都要进 overflow 与角标。
  applyOverflow(scene, countNonFinite(tt) + countNonFinite(x) + countNonFinite(residual), { slot: SLOT.overflow, width });
  return scene;
};

/** 有限的 ≤0 的个数（对数轴上"画不出"，与 NaN 分开报数）。 */
function countNonPositive(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && v <= 0) n += 1;
  }
  return n;
}
