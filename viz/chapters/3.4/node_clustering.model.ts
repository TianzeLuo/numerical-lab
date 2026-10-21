/**
 * `node_clustering`（3.4 章）的 model：Chebyshev-Lobatto 节点的端点聚集与插值误差位置。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 均匀节点对照（灰虚线，明确标注「不是本图节点」） |
 * | 2 | line+scatter | `main` | Chebyshev 节点 `nodes`（x_j 对序号 j） |
 * | 3 | line+scatter | `main` | 相邻间距 `spacing`（半格偏移，另一条标尺） |
 * | 4 | line | `spacing` | 实测间距 vs 位置（端点 N^-2、中心 N^-1） |
 * | 5 | line | `spacing` | 解析密度的倒数 1/ρ（橙虚线：密度是公式，不是间距统计） |
 * | 6 | line | `spacing` | 均匀间距 2/N（灰虚线） |
 * | 7 | line | `error` | 插值误差 `error`（节点上为 0，节点之间才是误差） |
 * | 8 | line | `error` | 配置解误差 `collocation_error`（负对照） |
 * | 9 | scatter | `error` | 节点位置（画在地板上，表示「节点上误差=0」） |
 *
 * 三条防误导决定
 * --------------
 * 1. **绝不假装节点均匀**：主面板同时画均匀对照与真正的 Chebyshev 节点；两条曲线在
 *    端点附近的分离就是聚集。
 * 2. **插值误差为 0 ≠ 精度高**：节点上的 0 是插值定义，不是测到的；这些点画在
 *    `roundoff_floor` 地板上并明确标注，节点之间的峰值才是有意义的量。
 * 3. **密度是解析式**：`density_fine` 来自帧的 rho(x)=N/(pi sqrt(1-x^2))，图上画它的
 *    倒数与实测间距并排；小 N 时两者的差别本身就是第 3 个陷阱。
 *
 * 时间轴：manifest `mode="none"`，`scene.t` 恒为 0。
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
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  stringOf,
  ModelInputError,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  badge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  legendBlock,
  logRange,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.4";
const VIZ = "node_clustering";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Chebyshev 节点：端点聚集与插值误差的位置";

const SLOT = {
  source: 0,
  axisMode: 1,
  aux: 2,
  status: 3,
  preview: 4,
  overflow: 6,
  degraded: 7,
} as const;

export const nodeClusteringModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const nodes = requireArray(frame, "nodes");
  const spacing = requireArray(frame, "spacing");
  const spacingMid = requireArray(frame, "spacing_mid");
  const xFine = requireArray(frame, "x_fine");
  const interpError = requireArray(frame, "error");
  const collError = requireArray(frame, "collocation_error");
  const pitfalls = requirePitfalls(frame);
  requireSameLength(xFine, interpError, "帧数组 x_fine 与 error");
  requireSameLength(xFine, collError, "帧数组 x_fine 与 collocation_error");
  if (nodes.length < 3) throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 nodes 少于 3 个：无法画节点聚集`);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const densityFine = frame.has("density_fine") ? requireArray(frame, "density_fine") : undefined;
  if (densityFine === undefined) {
    degraded.push("帧缺 density_fine：不画 1/ρ 解析曲线（不按 N 自己造密度）");
  } else if (densityFine.length !== xFine.length) {
    degraded.push(`density_fine 长度 ${densityFine.length} ≠ x_fine ${xFine.length}：不画 1/ρ（拒绝错位）`);
  }
  const densityNodes = frame.has("density_nodes") ? requireArray(frame, "density_nodes") : undefined;
  if (densityNodes !== undefined && densityNodes.length !== nodes.length) {
    degraded.push("density_nodes 与 nodes 长度不符：不读节点处密度");
  }

  // —— meta ——
  const nMeta = metaNumber(frame, "n");
  const n = Math.max(2, nodes.length - 1);
  if (nMeta !== undefined && Math.round(nMeta) !== n) {
    degraded.push(`帧 meta.n=${Math.round(nMeta)} 与 nodes 长度-1=${n} 不一致：按数组画`);
  }
  const clusterRatio = metaNumber(frame, "cluster_ratio");
  const ratioOverN = metaNumber(frame, "ratio_over_n");
  const ratioTheory = metaNumber(frame, "ratio_over_n_theory");
  const maxGapMeta = metaNumber(frame, "max_gap");
  const minGapMeta = metaNumber(frame, "min_gap");
  const outerNodes = metaNumber(frame, "outer_nodes");
  const interpMax = metaNumber(frame, "interp_error_max");
  const collMax = metaNumber(frame, "collocation_error_max");
  const orderGap = metaNumber(frame, "order_gap");
  const densityRatio = metaNumber(frame, "density_ratio");
  const densityRatioTheory = metaNumber(frame, "density_ratio_theory");
  const condInf = metaNumber(frame, "cond_inf");
  const floorMeta = metaNumber(frame, "roundoff_floor");
  const problem = stringOf(frame, "problem");
  const label = stringOf(frame, "label");

  let maxGap = maxGapMeta;
  let minGap = minGapMeta;
  if (maxGap === undefined || minGap === undefined) {
    let lo = Number.POSITIVE_INFINITY;
    let hi = 0;
    for (let i = 0; i < spacing.length; i += 1) {
      const v = spacing[i] as number;
      if (!Number.isFinite(v) || v <= 0) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    if (maxGap === undefined) maxGap = Number.isFinite(hi) ? hi : undefined;
    if (minGap === undefined) minGap = Number.isFinite(lo) ? lo : undefined;
  }

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.58, auxH: 142, gap: 54, contentLeft: 120 });
  const short = L.floor - L.topY < 300;
  const topH = short ? Math.max(60, L.floor - L.topY - 28) : L.topH;
  const auxH = Math.max(52, Math.min(L.auxH, topH));

  // —— 主面板：节点 vs 序号；均匀对照与间距同图 ——
  const main = makeAxis("main", [L.left, L.topY, L.leftW, topH], {
    xlim: [0, n],
    ylim: [-1.08, 1.08],
    title: L.columns ? axisTitleOf(SPEC?.axes, "main", "节点、间距与密度") : "节点、间距与密度",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "x_j 与间距"),
    xlabel: "节点序号 $j$",
    autoscaled: false,
  });
  scene.axes.push(main);

  const uniformY = new Float64Array(n + 1);
  for (let j = 0; j <= n; j += 1) uniformY[j] = -1 + (2 * j) / n;
  scene.layers.push(
    lineLayer(
      "main",
      series(indexOf(n + 1, 0), uniformY, pal.reference, {
        width: 1.2,
        dash: [6, 4],
      }),
    ),
  );
  scene.layers.push(
    lineLayer("main", series(indexOf(n + 1, 0), nodes, pal.curves[0], { width: 1.8, marker: "o" })),
  );
  const spacingX = indexOf(Math.min(spacing.length, spacingMid.length), 0.5);
  scene.layers.push(
    lineLayer("main", series(spacingX, spacing, pal.curves[1], { width: 1.2, marker: "s" })),
  );

  // —— 间距面板：实测间距 / 1-ρ / 均匀间距 ——
  if (!short) {
    let invRhoMax = 0;
    const invRho = new Float64Array(densityFine !== undefined && densityFine.length === xFine.length ? xFine.length : 0);
    for (let i = 0; i < invRho.length; i += 1) {
      const rho = densityFine?.[i] as number;
      const v = Number.isFinite(rho) && rho > 0 ? 1 / rho : Number.NaN;
      invRho[i] = v;
      if (Number.isFinite(v)) invRhoMax = Math.max(invRhoMax, v);
    }
    const yMax = Math.max(maxGap ?? 0, invRhoMax, 1e-12) * 1.18;
    const spacingAxis = makeAxis("spacing", [L.left, L.botY, L.leftW, L.botH], {
      xlim: [-1.02, 1.02],
      ylim: [0, yMax],
      title: "间距 vs 位置：端点 ~N^-2，中心 ~N^-1",
      ylabel: "间距 / $1-ρ$",
      xlabel: "$x$",
      autoscaled: false,
    });
    scene.axes.push(spacingAxis);
    scene.layers.push(
      lineLayer("spacing", series(spacingMid, spacing, pal.curves[1], { width: 1.8, marker: "o" })),
    );
    if (invRho.length > 0) {
      scene.layers.push(
        lineLayer("spacing", series(xFine, invRho, pal.exact, { width: 1.4, dash: [6, 4] })),
      );
    }
    scene.layers.push(
      lineLayer(
        "spacing",
        series([spacingAxis.xlim[0], spacingAxis.xlim[1]], [2 / n, 2 / n], pal.reference, {
          width: 1.1,
          dash: [5, 4],
        }),
      ),
    );
    for (const xEdge of [-0.9, 0.9]) {
      scene.layers.push(
        lineLayer(
          "spacing",
          series([xEdge, xEdge], spacingAxis.ylim, pal.reference, { width: 0.9, dash: [2, 3] }),
        ),
      );
    }
    scene.texts.push(
      ...legendBlock(
        L.left + 6,
        L.botY + 6,
        L.leftW,
        `绿=实测间距　橙虚=1/ρ（解析）　灰虚=均匀 2/N；两端 10% 内 ${formatNumber(outerNodes ?? 0, 3)}/${n + 1} 个节点`,
        pal.muted,
      ),
    );
  } else {
    degraded.push("窄画布：间距面板与右列说明未排下（节点/误差仍在）");
  }

  // —— 误差副轴（规则 3）：插值误差 vs 配置解误差 ——
  const errPos = positiveMinMax([
    ...Array.from(interpError),
    ...Array.from(collError),
    floorMeta ?? Number.NaN,
  ]);
  const errLo = errPos?.[0] ?? 1e-16;
  const errHi = errPos?.[1] ?? 1;
  const errAxis = makeAxis("aux", [L.rightX, L.topY, L.rightW, auxH], {
    xlim: [-1, 1],
    ylim: logRange(errLo * 0.5, errHi * 2),
    yscale: "log",
    title: L.columns
      ? axisTitleOf(SPEC?.axes, "aux", "插值误差与配置解误差（对数轴）")
      : "插值/配置误差",
    ylabel: "误差",
    xlabel: "$x$",
    autoscaled: false,
  });
  scene.axes.push(errAxis);
  const floor = floorMeta ?? errLo;
  scene.layers.push(
    lineLayer(
      "aux",
      series([errAxis.xlim[0], errAxis.xlim[1]], [Math.max(floor, errAxis.ylim[0]), Math.max(floor, errAxis.ylim[0])], pal.reference, {
        width: 1,
        dash: [2, 4],
      }),
    ),
  );
  scene.layers.push(
    lineLayer("aux", series(xFine, interpError, pal.curves[0], { width: 2 })),
  );
  scene.layers.push(
    lineLayer("aux", series(xFine, collError, pal.residual, { width: 1.5, dash: [5, 4] })),
  );
  // 节点上的插值误差恒为 0：把这些位置画在地板上（明确标注，不冒充精度）。
  const nodeFloor = new Float64Array(nodes.length);
  for (let i = 0; i < nodes.length; i += 1) nodeFloor[i] = Math.max(floor, errAxis.ylim[0]);
  scene.layers.push(
    scatterLayer("aux", nodes, nodeFloor, { r: 2.2, color: pal.exact, alpha: 0.8 }),
  );
  if (interpMax !== undefined && interpMax > 0) {
    let best = 0;
    for (let i = 1; i < interpError.length; i += 1) if ((interpError[i] as number) > (interpError[best] as number)) best = i;
    scene.layers.push(
      scatterLayer("aux", [xFine[best] as number], [interpError[best] as number], {
        r: 4,
        color: pal.marker,
      }),
    );
  }

  // —— 面板图例 ——
  scene.texts.push(
    ...legendBlock(L.left + 6, L.topY + 6, L.leftW, "蓝=Chebyshev 节点（cos(jπ/N)）　灰虚=均匀对照　绿=间距", pal.muted),
  );
  scene.texts.push(...legendBlock(L.rightX + 4, L.topY + 6, L.rightW, "蓝=插值　紫虚=配置解　橙点=节点（误差 0）", pal.muted));

  // —— 右列文字 ——
  if (L.columns) {
    const head = paramSummary(frame, opts, [
      { name: "n", label: "$n$" },
      { name: "k", label: "$k$" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`N=${n}（${n + 1} 个节点）　${head.text}　${problem ?? "?"}`);
    if (label !== undefined) lines.push(label);
    if (clusterRatio !== undefined && ratioOverN !== undefined) {
      lines.push(
        `间距比 max/min=${formatNumber(clusterRatio, 4)}（=${formatNumber(ratioOverN, 4)}·N，渐近 2/π=${formatNumber(ratioTheory ?? 2 / Math.PI, 5)}）`,
      );
    }
    if (maxGap !== undefined && minGap !== undefined) {
      lines.push(`最大间距 ${formatNumber(maxGap, 4)}，最小 ${formatNumber(minGap, 4)}：端点密、中心疏`);
    }
    if (densityRatio !== undefined) {
      lines.push(
        `ρ(0.9)/ρ(0)=${formatNumber(densityRatio, 4)}（理论 ${formatNumber(densityRatioTheory ?? Math.sqrt(1 / 0.19), 4)}）：密度是解析式`,
      );
    }
    if (interpMax !== undefined && collMax !== undefined) {
      lines.push(`插值误差峰值 ${formatNumber(interpMax, 3)}；配置解 ${formatNumber(collMax, 3)}`);
      if (orderGap !== undefined) lines.push(`配置解比插值差 ${formatNumber(Math.abs(orderGap), 3)} 个量级（微分吃一阶光滑度）`);
    }
    if (condInf !== undefined) lines.push(`配置矩阵 κ∞=${formatNumber(condInf, 3)}：N 不能一味加大`);
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else if (!short) {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "坐标轴：固定（节点/间距/误差窗口不随帧回缩）", 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(badge("status", "副轴：插值误差与配置误差共用对数标尺", 8, 44), SLOT.aux, width),
  );
  const status = [
    `N=${n}`,
    clusterRatio !== undefined ? `间距比 ${formatNumber(clusterRatio, 3)}` : "",
    condInf !== undefined ? `κ=${formatNumber(condInf, 2)}` : "",
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(nodes) +
    countNonFinite(spacing) +
    countNonFinite(spacingMid) +
    countNonFinite(xFine) +
    countNonFinite(interpError) +
    countNonFinite(collError) +
    (densityFine !== undefined ? countNonFinite(densityFine) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

function indexOf(n: number, offset: number): Float64Array {
  const out = new Float64Array(Math.max(0, n));
  for (let i = 0; i < out.length; i += 1) out[i] = i + offset;
  return out;
}
